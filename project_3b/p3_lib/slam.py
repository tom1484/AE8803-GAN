import numpy as np
import gtsam
from typing import Tuple, List

from p3_lib.icp import icp
from scipy.spatial import KDTree


def simulate_uwb(
    poses_gt_list: List[np.ndarray],
    range_trigger: float = 0.60,
    range_sigma: float = 0.05,
    max_edges_per_pair: int = 5,
    seed: int = 0,
    fallback_to_nearest: bool = True,
) -> List[Tuple[int, int, int, float]]:
    """Generate noisy inter-robot range measurements (UWB-style).

    For each robot pair, finds keyframes where they are within range_trigger
    distance and produces noisy range measurements.

    Args:
        poses_gt_list: List of (n_kf, 3) ground truth poses per robot.
        range_trigger (float): Max inter-robot distance to generate a range edge (m).
        range_sigma (float): Std dev of range noise (m).
        max_edges_per_pair (int): Max range edges per robot pair.
        seed (int): Random seed.
        fallback_to_nearest (bool): If robots never close enough, use nearest keyframes.

    Returns:
        List of (robot_r, robot_s, keyframe_idx, noisy_range) tuples.
    """
    rng = np.random.default_rng(seed)
    n_robots = len(poses_gt_list)
    n_kf = len(poses_gt_list[0])

    for r in range(n_robots):
        assert len(poses_gt_list[r]) == n_kf, "Keyframes not synchronized across robots"

    range_edges = []

    for r_id in range(n_robots):
        for s_id in range(r_id + 1, n_robots):
            pr = poses_gt_list[r_id][:, :2]
            ps = poses_gt_list[s_id][:, :2]
            d = np.linalg.norm(pr - ps, axis=1)

            close_idx = np.where(d < range_trigger)[0]

            if len(close_idx) == 0:
                if not fallback_to_nearest:
                    continue
                chosen = np.argsort(d)[:max_edges_per_pair]
            else:
                if len(close_idx) > max_edges_per_pair:
                    sel = np.linspace(
                        0, len(close_idx) - 1, max_edges_per_pair, dtype=int
                    )
                    chosen = close_idx[sel]
                else:
                    chosen = close_idx

            for i in chosen:
                z = float(d[int(i)] + rng.normal(0.0, range_sigma))
                range_edges.append((int(r_id), int(s_id), int(i), z))

    return range_edges


def encoders_to_odometry(
    encoders: np.ndarray,
    initial_pose: np.ndarray,
    wheel_radius: float = 0.016,
    base_length: float = 0.105,
    ticks_per_rev: float = 28.0 * 100.37,
) -> np.ndarray:
    """Convert cumulative wheel encoder ticks to SE2 odometry poses.

    Uses differential drive kinematics to integrate encoder readings into
    a trajectory of poses.

    Steps:
    1. Compute delta ticks between consecutive timesteps.
    2. Convert to wheel arc lengths: d = delta_ticks * (2 * pi * wheel_radius) / ticks_per_rev
    3. Differential drive: d_center = (d_left + d_right) / 2
                           d_theta  = (d_right - d_left) / base_length
    4. Integrate: x += d_center * cos(theta), y += d_center * sin(theta), theta += d_theta

    Args:
        encoders (np.ndarray): (T, 2) cumulative encoder ticks [left, right].
        initial_pose (np.ndarray): (3,) starting pose [x, y, theta].
        wheel_radius (float): Wheel radius in meters.
        base_length (float): Distance between left and right wheels in meters.
        ticks_per_rev (float): Encoder ticks per full wheel revolution.

    Returns:
        np.ndarray: (T, 3) array of SE2 poses [x, y, theta]. poses[0] == initial_pose.
    """
    # ------- start solution ---------------------------------
    delta_ticks = encoders[1:] - encoders[:-1]
    d = delta_ticks * (2 * np.pi * wheel_radius) / ticks_per_rev
    d_center = (d[:, 0] + d[:, 1]) / 2
    d_theta = (d[:, 1] - d[:, 0]) / base_length

    d_center = np.concatenate(([0.0], d_center))
    d_theta = np.concatenate(([0.0], d_theta))

    theta = initial_pose[2] + np.cumsum(d_theta)
    # Translate along the heading *before* this step's rotation (step 4 order)
    theta_prev = np.concatenate(([initial_pose[2]], theta[:-1]))
    x = initial_pose[0] + np.cumsum(d_center * np.cos(theta_prev))
    y = initial_pose[1] + np.cumsum(d_center * np.sin(theta_prev))

    return np.stack((x, y, theta), axis=-1)
    # ------- end solution -----------------------------------


def build_single_robot_graph(
    poses_odom: np.ndarray,
    scans_local: List[np.ndarray],
    gt_first_pose: np.ndarray,
    search_radius: float = 0.55,
    icp_threshold: float = 0.06,
    odom_sigmas: np.ndarray = np.array([0.005, 0.005, 0.002]),
    scan_sigmas: np.ndarray = np.array([0.025, 0.025, 0.008]),
    prior_sigmas: np.ndarray = np.array([1e-4, 1e-4, 1e-4]),
    max_lc_per_node: int = 8,
    lc_stride: int = 1,
    base: int = -1,
    add_prior: bool = True,
) -> Tuple[gtsam.NonlinearFactorGraph, gtsam.Values, List[Tuple[int, int]]]:
    """Build a GTSAM factor graph for single-robot pose-graph SLAM.

    Creates a graph with three types of factors:
    1. Prior factor on the first pose (anchors the graph to world frame).
    2. Odometry BetweenFactorPose2 between every consecutive pair.
    3. ICP BetweenFactorPose2 between pairs within search_radius:
       - Consecutive pairs get BOTH odometry AND ICP factors.
       - Non-consecutive pairs within search_radius get ICP factors (loop closures).
       - Use odometry-estimated relative pose as ICP initial guess.
       - Only add ICP factor if mean error < icp_threshold.
       - Limit to max_lc_per_node closest candidates per keyframe.

    Keying: use gtsam.symbol_shorthand.X, so X(0), X(1), ..., X(N-1).

    Noise models: build from the sigma arrays using gtsam.noiseModel.Diagonal.Sigmas().

    ICP call pattern:
        odom_rel = gtsam.Pose2(*poses_odom[j]).between(gtsam.Pose2(*poses_odom[i]))
        aTb, err = icp(source=scans_local[i], target=scans_local[j],
                        max_corr_dist=search_radius + 0.2, initial_guess=???)

    Args:
        poses_odom (np.ndarray): (N, 3) odometry poses [x, y, theta] at keyframes.
        scans_local (List[np.ndarray]): List of (K, 2) scans in local frame per keyframe.
        gt_first_pose (np.ndarray): (3,) ground truth first pose for the prior factor.
        search_radius (float): Max distance between odom poses to consider for ICP.
        icp_threshold (float): Max ICP mean error to accept as a valid factor.
        odom_sigmas (np.ndarray): (3,) noise sigmas [x, y, theta] for odometry factors.
        scan_sigmas (np.ndarray): (3,) noise sigmas [x, y, theta] for ICP factors.
        prior_sigmas (np.ndarray): (3,) noise sigmas [x, y, theta] for prior factor.
        max_lc_per_node (int): Max number of ICP candidates to try per keyframe.
        lc_stride (int): Only attempt ICP every lc_stride keyframes.

    Returns:
        Tuple of:
            graph (gtsam.NonlinearFactorGraph): The factor graph.
            initial (gtsam.Values): Initial values (odometry poses).
            lc_pairs (List[Tuple[int, int]]): List of (j, i) index pairs where ICP
                factors were added.
    """
    # ------- start solution ---------------------------------
    graph = gtsam.NonlinearFactorGraph()
    lc_pairs = []
    initials = gtsam.Values()

    N = len(poses_odom)
    X = gtsam.symbol_shorthand.X
    if base == -1:
        to_key = lambda i: X(i)
    else:
        to_key = lambda i: X(i + base)

    poses = [gtsam.Pose2(*odom) for odom in poses_odom]
    for i in range(N):
        initials.insert(to_key(i), poses[i])

    # Build prior pose factor
    if add_prior:
        prior_mean = gtsam.Pose2(*gt_first_pose)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        graph.add(gtsam.PriorFactorPose2(to_key(0), prior_mean, prior_noise))

    # Build odometry factors
    for i in range(N - 1):
        relative = poses[i].between(poses[i + 1])
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(odom_sigmas)
        graph.add(
            gtsam.BetweenFactorPose2(to_key(i), to_key(i + 1), relative, odom_noise)
        )

    # Build lc pairs using KDTree
    tree = KDTree(poses_odom[:, :2])
    neighbors_list = tree.query_ball_tree(tree, search_radius)

    # total_candidates = 0
    # total_rejection = 0

    scan_noise = gtsam.noiseModel.Diagonal.Sigmas(scan_sigmas)
    for i in range(0, N, lc_stride):
        odom_src = poses_odom[i]
        neighbors = neighbors_list[i]

        candidates = np.array([j for j in sorted(neighbors) if j > i], dtype=int)
        distances = np.linalg.norm(poses_odom[candidates, :2] - odom_src[:2], axis=1)
        # Only try the max_lc_per_node closest candidates
        closest = candidates[np.argsort(distances)][:max_lc_per_node]

        # total_candidates += closest.shape[0]

        for j in closest:
            # icp(source=i, target=j) returns jTi (frame i -> frame j)
            odom_rel = poses[j].between(poses[i])
            aTb, err = icp(
                source=scans_local[i],
                target=scans_local[j],
                max_corr_dist=search_radius + 0.2,
                initial_guess=odom_rel,
            )
            if err >= icp_threshold:
                # total_rejection += 1
                continue

            # Measurement is jTi, so the factor goes from X(j) to X(i)
            graph.add(gtsam.BetweenFactorPose2(to_key(j), to_key(i), aTb, scan_noise))
            lc_pairs.append((j, i))

    # print(total_candidates, total_rejection)

    return graph, initials, lc_pairs
    # ------- end solution -----------------------------------


def build_multi_robot_graph(
    poses_odom_list: List[np.ndarray],
    scans_local_list: List[List[np.ndarray]],
    gt_first_poses: List[np.ndarray],
    range_edges: List[Tuple[int, int, int, float]],
    search_radius: float = 0.55,
    icp_threshold: float = 0.06,
    odom_sigmas: np.ndarray = np.array([0.005, 0.005, 0.002]),
    scan_sigmas: np.ndarray = np.array([0.025, 0.025, 0.008]),
    prior_sigmas: np.ndarray = np.array([1e-4, 1e-4, 1e-4]),
    range_sigma: float = 0.05,
    max_lc_per_node: int = 8,
    lc_stride: int = 1,
) -> Tuple[gtsam.NonlinearFactorGraph, gtsam.Values, List[Tuple[int, int, int]]]:
    """Build a combined factor graph for multi-robot pose-graph SLAM.

    Combines per-robot subgraphs (odometry + ICP, same as build_single_robot_graph)
    with cross-robot range factors.

    Key differences from single-robot:
    - Keying: gtsam.symbol('x', robot_id * 100000 + keyframe_idx) to avoid collisions.
    - Prior: only robot 0 gets a prior on its first pose (anchors the world frame).
    - Per-robot subgraph: same logic as build_single_robot_graph (odom + ICP factors).
    - Range factors: for each (r, s, i, z) from range_edges, add a
      gtsam.RangeFactorPose2 between robot r's keyframe i and robot s's keyframe i,
      with noise model gtsam.noiseModel.Isotropic.Sigma(1, range_sigma).

    Args:
        poses_odom_list: List of (N, 3) odometry poses per robot.
        scans_local_list: List of per-robot scan lists.
        gt_first_poses: List of (3,) ground truth first pose per robot.
        range_edges: List of (robot_r, robot_s, keyframe_idx, range_z) tuples.
        search_radius (float): Max distance for ICP candidates.
        icp_threshold (float): Max ICP error to accept.
        odom_sigmas (np.ndarray): Noise sigmas for odometry factors.
        scan_sigmas (np.ndarray): Noise sigmas for ICP factors.
        prior_sigmas (np.ndarray): Noise sigmas for prior factor.
        range_sigma (float): Noise sigma for range factors.
        max_lc_per_node (int): Max ICP candidates per keyframe.
        lc_stride (int): ICP attempt frequency.

    Returns:
        Tuple of:
            graph (gtsam.NonlinearFactorGraph): The combined factor graph.
            initial (gtsam.Values): Initial values from odometry.
            lc_pairs_all (List[Tuple[int, int, int]]): List of (robot_id, j, i)
                triples where ICP factors were added.
    """
    # ------- start solution ---------------------------------
    R = len(poses_odom_list)
    to_key = lambda r, i: gtsam.symbol("x", r * 100000 + i)

    # Build individual graph and merge
    graph = gtsam.NonlinearFactorGraph()
    initials = gtsam.Values()
    lc_pairs = []
    for r_id in range(R):
        r_graph, r_initials, r_lc_pairs = build_single_robot_graph(
            poses_odom_list[r_id],
            scans_local_list[r_id],
            gt_first_poses[r_id],
            search_radius=search_radius,
            icp_threshold=icp_threshold,
            odom_sigmas=odom_sigmas,
            scan_sigmas=scan_sigmas,
            prior_sigmas=prior_sigmas,
            max_lc_per_node=max_lc_per_node,
            lc_stride=lc_stride,
            base=r_id * 100000,
            add_prior=(r_id == 0),  # only robot 0 anchors the world frame
        )

        graph.push_back(r_graph)
        initials.insert(r_initials)
        lc_pairs.extend([(r_id, j, i) for j, i in r_lc_pairs])

    # Build cross-robot range factors
    for r, s, i, z in range_edges:
        range_noise = gtsam.noiseModel.Isotropic.Sigma(1, range_sigma)
        graph.add(gtsam.RangeFactorPose2(to_key(r, i), to_key(s, i), z, range_noise))

    return graph, initials, lc_pairs
    # ------- end solution -----------------------------------
