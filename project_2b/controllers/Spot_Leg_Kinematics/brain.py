import numpy as np
import gtsam
from robot import M, B_list


def forward_kinematics(M: gtsam.Pose3, B_list: list, theta: np.ndarray) -> gtsam.Pose3:
    """
    Computes the forward kinematics using the Product of Exponentials (POE) formula in the body frame.

    Args:
        M (gtsam.Pose3): The home configuration of the end-effector (pose at zero joint angles).
        B_list (list): A list of 6D screw axes in the body frame, one for each joint.
        theta (np.ndarray): A numpy array of joint angles (in radians).

    Returns:
        gtsam.Pose3: The pose of the end-effector in the body frame after applying the joint angles.
    """
    T_out = M
    # ------- start solution ---------------------------------
    for i, B in enumerate(B_list):
        A = gtsam.Pose3.Expmap(B * theta[i])
        T_out = T_out.compose(A)
    # ------- end solution -----------------------------------
    return T_out


def jacobian_body(B_list: list, theta: np.ndarray) -> np.ndarray:
    """
    Computes the 6x3 body Jacobian for a 3-joint leg.

    Args:
        B_list (list): A list of 6D screw axes in the body frame, one for each joint.
        theta (np.ndarray): A numpy array of joint angles (in radians).

    Returns:
        np.ndarray: A 6x3 numpy array representing the body Jacobian matrix.
    """
    out = np.zeros((6, 3), dtype=float)
    # ------- start solution ---------------------------------
    N = len(B_list)

    adj_list = [gtsam.Pose3.Identity()]
    for i, B in enumerate(B_list[1:]):
        E = gtsam.Pose3.Expmap(-B * theta[i + 1])
        adj_list.append(E.compose(adj_list[-1]))
    adj_list = adj_list[::-1]

    for i, B in enumerate(B_list):
        out[:, i] = gtsam.Pose3.AdjointMap(adj_list[i]) @ B
    # ------- end solution -----------------------------------

    return out


def inverse_kinematics(
    M: gtsam.Pose3,
    B_list: list,
    T_d: gtsam.Pose3,
    theta_init: np.ndarray,
    alpha: float = 0.1,
    max_iters: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Solves the Inverse Kinematics (IK) problem using gradient descent on the body-frame error twist.

    Args:
        M (gtsam.Pose3): The home configuration of the end-effector (pose at zero joint angles).
        B_list (list): A list of 6D screw axes in the body frame, one for each joint.
        T_d (gtsam.Pose3): The desired pose of the end-effector in the body frame.
        theta_init (np.ndarray): Initial guess for the joint angles (in radians).
        alpha (float, optional): Step size for the gradient descent. Defaults to 0.1.
        max_iters (int, optional): Maximum number of iterations for the solver. Defaults to 100.
        tol (float, optional): Convergence tolerance for the error twist norm. Defaults to 1e-4.

    Returns:
        np.ndarray: A numpy array of joint angles (in radians) that achieve the desired pose.
    """
    theta = theta_init
    # ------- start solution ---------------------------------
    W = np.zeros((6, 6), dtype=float)
    W[3:, 3:] = np.identity(3)

    for i in range(max_iters):
        T = forward_kinematics(M, B_list, theta)
        e = gtsam.Pose3.Logmap(T.inverse().compose(T_d))
        if np.linalg.norm(e) < tol:
            break
        J = jacobian_body(B_list, theta)
        G = -J.T @ W @ e
        theta -= alpha * G
    # ------- end solution -----------------------------------
    return theta

