"""Control, estimation, and command logic for the ROV controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import gtsam
import numpy as np
from gtsam import NavState, NavStateImuEKF, Point3, PreintegrationParams
from robot import Control, Measurements

# Manual command step sizes (forces in N, torques in N*m).
FORCE_X_STEP = 1.0
FORCE_Z_STEP = 2.0
TAU_Z_STEP = 1.0
TAU_X_STEP = 1.0

# IMU noise parameters.
SIGMA_ACC = 0.004
SIGMA_GYRO = 0.05

# Preintegration params for the EKF.
PREINTEGRATION_PARAMS: PreintegrationParams = gtsam.PreintegrationParams.MakeSharedU(
    9.81
)
PREINTEGRATION_PARAMS.setAccelerometerCovariance(np.diag([SIGMA_ACC**2] * 3))
PREINTEGRATION_PARAMS.setIntegrationCovariance(np.diag([1e-3] * 3))
PREINTEGRATION_PARAMS.setGyroscopeCovariance(np.diag([SIGMA_GYRO**2] * 3))

# Measurement covariances.
POSITION_MEAS_COV = np.eye(3) * 1.0**2
DEPTH_MEAS_COV = np.eye(1) * 0.1**2
RANGE_MEAS_COV = np.eye(1) * 1.0**2

# Trajectory follower gains.
KP_DIS = 5.0
KP_YAW = 1.0
KP_Z = 10.0
KP_ROLL = 1.0
KP_DIR = 0.2

# ITA = 0.95
# KI_DIS = 0.1
# KI_YAW = 0.02
# KI_Z = 0.02
# KI_ROLL = 0.002
# KI_DIR = 0.002


@dataclass(frozen=True)
class Wrench:
    """6-DOF force/torque command expressed in body coordinates."""

    torque: np.ndarray
    force: np.ndarray

    @staticmethod
    def torque_only(x: float, y: float, z: float) -> "Wrench":
        """Construct a pure torque wrench."""
        return Wrench(
            torque=np.array([x, y, z], dtype=float),
            force=np.zeros(3, dtype=float),
        )

    @staticmethod
    def force_only(x: float, y: float, z: float) -> "Wrench":
        """Construct a pure force wrench."""
        return Wrench(
            torque=np.zeros(3, dtype=float),
            force=np.array([x, y, z], dtype=float),
        )


B_MATRIX: np.ndarray = np.array(
    [
        [-1.0, 1.0, 0.0, 0.0],
        [0.1, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, 0.035, 0.035],
    ],
    dtype=float,
)

INVERSE_B_MATRIX: np.ndarray = np.linalg.inv(B_MATRIX)


def wrench_to_thrusters(wrench: Wrench) -> Tuple[float, float, float, float]:
    """
    Allocate thruster commands from a wrench.

    Uses the linear map y = B u where y = [Fx, tau_z, Fz, tau_x]^T.
    Return (u_lm, u_rm, u_vlm, u_vrm).
    """
    # ---------------- TODO: Implement this function --------------- #
    # TODO (students):
    # Use the provided INVERSE_B_MATRIX to convert the wrench into thruster commands.

    W = np.array(
        [[wrench.force[0], wrench.torque[2], wrench.force[2], wrench.torque[0]]]
    )
    U = (INVERSE_B_MATRIX @ W.T).T[0]
    u_lm, u_rm, u_vlm, u_vrm = U[0], U[1], U[2], U[3]

    # -------------------------------------------------------------- #
    return u_lm, u_rm, u_vlm, u_vrm


class Brain:
    def __init__(self, beacons: Sequence[Point3]):
        """Initialize the brain with known range beacons."""
        self.beacons: list[Point3] = list(beacons)
        self.ekf: Optional[NavStateImuEKF] = None

    def estimate(self, measurements: Measurements) -> Optional[NavState]:
        """Return a true-state snapshot for logging; estimation is handled elsewhere."""
        return measurements.X_est

    def initialize_EKF(self, X0: NavState, P0: np.ndarray) -> None:
        """Create the EKF with initial state and covariance."""
        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # Initialize self.ekf using the provided X0 and P0, and PREINTEGRATION_PARAMS.

        self.ekf: Optional[NavStateImuEKF] = gtsam.NavStateImuEKF(
            X0, P0, PREINTEGRATION_PARAMS
        )

        # -------------------------------------------------------------- #

        self.t_prev = 0.0
        self.eint_dis = 0.0
        self.eint_yaw = 0.0
        self.eint_z = 0.0
        self.eint_roll = 0.0
        self.eint_dir = 0.0

    def EKF_predict(
        self, omega_meas: np.ndarray, acc_meas: np.ndarray, dt: float
    ) -> Tuple[Optional[NavState], Optional[np.ndarray]]:
        """Run EKF prediction and return (state, covariance)."""
        if self.ekf is None:
            return NavState(), np.eye(9)
        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # Use self.ekf.predict to perform the prediction step with the provided omega_meas, acc_meas, and dt.
        # Return the updated state and covariance.
        self.ekf.predict(omega_meas, acc_meas, dt)
        X_est = self.ekf.state()
        P = self.ekf.covariance()

        return X_est, P
        # -------------------------------------------------------------- #

    def EKF_update(self, measurements: Measurements) -> None:
        """Apply measurement updates based on available measurements (position, depth, ranges)."""
        if self.ekf is None:
            return None

        X_est = self.ekf.state()
        R_est = X_est.attitude().matrix()

        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # For each available measurement in measurements (position, depth, ranges),
        # compute the appropriate measurement matrix H and call self.ekf.updateWithVector.

        p_est = X_est.position()

        hX_ = []
        z_ = []
        H_ = []
        R_ = []
        R_dim = 0

        if measurements.position is not None:
            # TODO (students)
            hX_.append(p_est)
            z_.append(measurements.position)
            H_.append(
                np.concatenate(
                    [
                        np.zeros((3, 3)),
                        R_est,
                        np.zeros((3, 3)),
                    ],
                    axis=1,
                )
            )
            R_.append(POSITION_MEAS_COV)
            R_dim += 3

        if measurements.depth is not None:
            # TODO (students)
            hX_.append(p_est[2:3])
            z_.append(np.array([measurements.depth]))
            H_.append(
                np.concatenate(
                    [
                        np.zeros((1, 3)),
                        R_est[2:3, :],
                        np.zeros((1, 3)),
                    ],
                    axis=1,
                )
            )
            R_.append(DEPTH_MEAS_COV)
            R_dim += 1

        if measurements.ranges is not None and self.beacons:
            # TODO (students)
            for _, (r, b) in enumerate(zip(measurements.ranges, self.beacons)):
                rv_est = (p_est - b).reshape((1, 3))  # (1, 3)
                r_est = np.sqrt(rv_est @ rv_est.T)[0]  # (1, )

                hX_.append(r_est)
                z_.append(np.array([r]))
                H_.append(
                    np.concatenate(
                        [
                            np.zeros((1, 3)),
                            rv_est @ R_est / r_est[0],
                            np.zeros((1, 3)),
                        ],
                        axis=1,
                    )
                )
                R_.append(RANGE_MEAS_COV)
                R_dim += 1

        if len(hX_) > 0:
            hX = np.concatenate(hX_)
            z = np.concatenate(z_)
            H = np.concatenate(H_)
            R = np.zeros((R_dim, R_dim))
            i = 0
            for r_ in R_:
                r_dim = r_.shape[0]
                R[i : i + r_dim, i : i + r_dim] = r_
                i += r_dim

            self.ekf.updateWithVector(hX, H, z, R)

        return None
        # -------------------------------------------------------------- #

    def act_on_command(self, key: int, keyboard: Any) -> Control:
        """Convert a keyboard command into a thruster Control."""

        wrench = None
        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # Replace the 0.0 entries below with the provided step-size constants:
        #   FORCE_X_STEP, FORCE_Z_STEP, TAU_Z_STEP, TAU_X_STEP.
        # Decide (for each key) whether it should apply a force or a torque, and put the
        # correct constant on the correct axis with the correct sign:
        #   - UP/DOWN:   +/- X force
        #   - W/S:       +/- Z force
        #   - E/Q:       +/- roll torque
        #   - LEFT/RIGHT:+/- yaw torque

        if key == keyboard.UP:
            wrench = Wrench.force_only(FORCE_X_STEP, 0.0, 0.0)
        elif key == keyboard.DOWN:
            wrench = Wrench.force_only(-FORCE_X_STEP, 0.0, 0.0)
        elif key == keyboard.LEFT:
            wrench = Wrench.torque_only(0.0, 0.0, TAU_Z_STEP)
        elif key == keyboard.RIGHT:
            wrench = Wrench.torque_only(0.0, 0.0, -TAU_Z_STEP)
        elif key == ord("S"):
            wrench = Wrench.force_only(0.0, 0.0, -FORCE_Z_STEP)
        elif key == ord("W"):
            wrench = Wrench.force_only(0.0, 0.0, FORCE_Z_STEP)
        elif key == ord("Q"):
            wrench = Wrench.torque_only(-TAU_X_STEP, 0.0, 0.0)
        elif key == ord("E"):
            wrench = Wrench.torque_only(TAU_X_STEP, 0.0, 0.0)

        # -------------------------------------------------------------- #

        if wrench is None:
            return Control(0.0, 0.0, 0.0, 0.0)

        u_lm, u_rm, u_vlm, u_vrm = wrench_to_thrusters(wrench)
        return Control(u_lm=u_lm, u_rm=u_rm, u_vlm=u_vlm, u_vrm=u_vrm)

    @staticmethod
    def wrap_to_pi(a: float) -> float:
        """Wrap an angle in radians to [-pi, pi)."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def follow_step(self, t: float, X: NavState, desired_state: NavState) -> Control:
        """Trajectory-following control step from desired pose."""

        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # Implement a simple P controller using the provided gains (KP_DIS, KP_YAW, KP_Z, KP_ROLL).
        # Compute the position error in the XY plane and Z axis, as well as the yaw and roll errors.
        # Remember to wrap the angular errors with self.wrap_to_pi.
        # Use these to compute desired forces/torques, then convert to thruster commands with wrench_to_thrusters.
        pos = X.position()
        att = X.attitude()
        desired_pos = desired_state.position()
        desired_att = desired_state.attitude()

        # Compute P control errors
        err_yaw = float(self.wrap_to_pi(desired_att.yaw() - att.yaw()))
        err_z = float(desired_pos[2] - pos[2])
        err_roll = float(self.wrap_to_pi(-att.roll()))

        z_axis = att.matrix()[:, 2]
        err_pos = desired_pos - pos
        err_pos_proj = err_pos - np.dot(err_pos, z_axis) * z_axis
        err_dis = float(np.linalg.norm(err_pos_proj))

        F_x = err_dis * KP_DIS * np.cos(err_yaw)
        tau_z = err_yaw * KP_YAW
        F_z = err_z * KP_Z
        tau_x = err_roll * KP_ROLL

        # --- Extensions ---
        # x_axis = att.matrix()[:, 0]
        # y_axis = att.matrix()[:, 1]
        # err_dir = np.arctan2(np.dot(err_pos, y_axis), np.dot(err_pos, x_axis))

        # F_x = err_dis * KP_DIS * np.cos(err_yaw)
        # tau_z = err_yaw * KP_YAW + err_dir * KP_DIR
        # F_z = err_z * KP_Z
        # tau_x = err_roll * KP_ROLL

        # Integrate errors for I control
        # dt = t - self.t_prev
        # self.eint_dis = ITA * self.eint_dis + err_dis * dt
        # self.eint_yaw = ITA * self.eint_yaw + err_yaw * dt
        # self.eint_z = ITA * self.eint_z + err_z * dt
        # self.eint_roll = ITA * self.eint_roll + err_roll * dt
        # self.eint_dir = ITA * self.eint_dir + err_dir * dt

        # F_x += self.eint_dis * KI_DIS * np.cos(err_yaw)
        # tau_z += self.eint_yaw * KI_YAW + self.eint_dir * KI_DIR
        # F_z += self.eint_z * KI_Z
        # tau_x += self.eint_roll * KI_ROLL

        # ------

        wrench = Wrench.force_only(F_x, 0.0, F_z)
        wrench.torque[0] = tau_x
        wrench.torque[2] = tau_z

        uL, uR, uvL, uvR = wrench_to_thrusters(wrench)

        # -------------------------------------------------------------- #

        # default compensation
        uvL -= 0.9
        uvR += 0.9
        return Control(u_lm=uL, u_rm=uR, u_vlm=uvL, u_vrm=uvR)
