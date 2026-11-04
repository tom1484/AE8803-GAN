"""Score a Project 2a foot trajectory the same way the Gradescope autograder does.

In the notebook, after launching the plotter:

    from trajectory_check import check
    check(plotter)

Scoring (10 points):
  - Targets, 5 points: 1 point for each Muybridge target the foot is within 10 cm of at that target's time.
  - Ground, 5 points: each separate trip below the ground line costs 1 point per started second
    (a 0.3 s dip costs 1 point, a 1.2 s dip costs 2). Dips shallower than 1 cm are ignored.
"""

import math

import numpy as np
import gtsam

# Leg and fit settings, the same as leg_plotter.py
T_END, L1, L2, X0, Y0 = 5.0, 2.0, 2.0, 2.5, 3.4
MEAS_NOISE, MIN_POINTS = 0.05, 3

# Muybridge targets, the same as leg_plotter.py: (time s, x m, y m, color)
TARGETS = [
    (0.0, 0.500, 0.0, "red"),
    (1.0, 2.500, 0.7, "pink"),
    (2.0, 4.500, 0.0, "yellow"),
    (3.0, 3.167, 0.0, "green"),
    (4.0, 1.833, 0.0, "blue"),
]

TARGET_TOL = 0.10   # m: a target counts if the foot is this close to it at the target's time
GROUND_POINTS = 5   # each trip below the ground line costs 1 point per started second
GROUND_EPS = 0.01   # m: trips below the ground line that stay shallower than this are ignored
N_SAMPLES = 5000    # samples per 5 s cycle for the ground check


def fit_fourier(points, t):
    """Same fit as LegPlotter.fit_fourier: joint angle (rad) at times t from (time s, angle rad) points."""
    points = np.asarray(points, dtype=float)
    x = 2.0 * np.pi * (points[:, 0] / T_END)
    sequence = {float(xi): float(yi) for xi, yi in zip(x, points[:, 1])}
    model = gtsam.noiseModel.Isotropic.Sigma(1, MEAS_NOISE)
    fit = gtsam.FitBasisFourierBasis(sequence, model, max(3, len(points)))
    params = fit.parameters()
    x_eval = 2.0 * np.pi * (np.asarray(t, dtype=float) / T_END)
    W = gtsam.FourierBasis.WeightMatrix(len(params), x_eval)
    return np.asarray(W @ params).reshape(-1)


def foot_position(theta_1, theta_2):
    """Same closed-form FK as LegPlotter.compute_fk (angles measured from straight down, CCW positive)."""
    a1 = theta_1 - np.pi / 2
    a2 = a1 + theta_2
    return X0 + L1 * np.cos(a1) + L2 * np.cos(a2), Y0 + L1 * np.sin(a1) + L2 * np.sin(a2)


def underground_trips(y, dt):
    """Separate stretches of the periodic foot height y below the ground line that reach deeper than GROUND_EPS.

    Returns a list of (start s, end s, depth m), or None if the foot never comes above the ground line.
    """
    below = y < 0.0
    if below.all():
        return None
    start = int(np.argmin(below))  # an above-ground sample, so no trip is split by the wrap from t = 5 s to 0 s
    below, y = np.roll(below, -start), np.roll(y, -start)
    n, trips, i = len(y), [], 0
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < n and below[j]:
            j += 1
        depth = -float(y[i:j].min())
        if depth > GROUND_EPS:
            trips.append((((i + start) % n) * dt, ((j + start) % n) * dt, depth))
        i = j
    return trips


def trip_cost(duration):
    """1 point per started second underground; durations are rounded to 0.01 s first, so a 1.00 s trip costs 1."""
    return max(1, math.ceil(round(duration, 2)))


def score(points_1, points_2):
    """Score the trajectory from (time s, angle rad) points. Returns Gradescope-style test dicts."""
    target_names = [f"Target {color} at t = {t:g} s" for t, _, _, color in TARGETS]
    ground_name = "Foot stays above ground"

    def all_zero(reason):
        tests = [dict(name=n, score=0.0, max_score=1.0, output=reason) for n in target_names]
        tests.append(dict(name=ground_name, score=0.0, max_score=float(GROUND_POINTS), output=reason))
        return tests

    if len(points_1) < MIN_POINTS or len(points_2) < MIN_POINTS:
        return all_zero(f"Need at least {MIN_POINTS} points on each joint-angle plot for a trajectory "
                        f"(found {len(points_1)} for theta_1 and {len(points_2)} for theta_2).")
    try:
        t = np.arange(N_SAMPLES) * (T_END / N_SAMPLES)
        _, y = foot_position(fit_fourier(points_1, t), fit_fourier(points_2, t))
        t_targets = np.array([target[0] for target in TARGETS])
        xt, yt = foot_position(fit_fourier(points_1, t_targets), fit_fourier(points_2, t_targets))
    except Exception as err:  # e.g. two points at the same time make the fit singular
        return all_zero(f"Could not fit a trajectory to these points ({err}).")

    tests = []
    for name, (_, gx, gy, _), fx, fy in zip(target_names, TARGETS, xt, yt):
        dist = float(np.hypot(fx - gx, fy - gy))
        fx, fy = round(float(fx), 3) + 0.0, round(float(fy), 3) + 0.0  # avoid printing -0.000
        tests.append(dict(
            name=name,
            score=1.0 if dist <= TARGET_TOL else 0.0,
            max_score=1.0,
            output=f"Foot at ({fx:.3f}, {fy:.3f}) m, {100 * dist:.1f} cm from the target ({gx:.3f}, {gy:.3f}) m. "
                   f"Tolerance: {100 * TARGET_TOL:.0f} cm.",
        ))

    trips = underground_trips(y, T_END / N_SAMPLES)
    if trips is None:
        ground, output = 0.0, "The foot is below the ground line for the whole cycle."
    elif not trips:
        ground, output = float(GROUND_POINTS), f"The foot stays above the ground line (lowest point y = {y.min():.3f} m)."
    else:
        lines = [f"The foot goes below the ground line {len(trips)} time(s); each trip costs 1 point per started second:"]
        cost = 0
        for a, b, depth in trips:
            duration = (b - a) % T_END
            cost += trip_cost(duration)
            wrap = " (wraps past t = 5 s)" if b < a else ""
            lines.append(f"  t = {a:.2f} s to {b:.2f} s{wrap}: {duration:.2f} s, down to {100 * depth:.1f} cm "
                         f"below ground, -{trip_cost(duration)}")
        ground, output = float(max(0, GROUND_POINTS - cost)), "\n".join(lines)
    tests.append(dict(name=ground_name, score=ground, max_score=float(GROUND_POINTS), output=output))
    return tests


def table_points(clicks):
    """Points as the notebook's table records them (2 decimals in seconds and degrees), so check() matches the autograder."""
    return np.reshape([(float(f"{t:.2f}"), np.radians(float(f"{np.degrees(a):.2f}"))) for t, a in clicks], (-1, 2))


def check(plotter):
    """Print the autograder's score for the plotter's current points."""
    tests = score(table_points(plotter.theta1_clicks), table_points(plotter.theta2_clicks))
    for test in tests:
        print(f"[{test['score']:g}/{test['max_score']:g}] {test['name']}")
        for line in test["output"].splitlines():
            print(f"    {line}")
    total = sum(test["score"] for test in tests)
    print(f"Trajectory score: {total:g} / {sum(test['max_score'] for test in tests):g}")
