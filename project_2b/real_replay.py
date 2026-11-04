from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from rosbags.highlevel import AnyReader

try:
    from .measurement import Measurement
except ImportError:
    from measurement import Measurement


class RealReplay:
    """Stream IMU/contact messages from a bag and emit Measurement objects online."""

    def __init__(self, bag_path: str, contact_state_value: int | None = None):
        self.bag_path = Path(bag_path)
        self.foot_names = ["FL", "FR", "RL", "RR"]
        self.contact_state_value = contact_state_value

        self._reader = AnyReader([self.bag_path])  # type: ignore[arg-type]
        self._reader.open()
        self._closed = False

        self._imu_connections = [c for c in self._reader.connections if c.topic == "/imu"]
        self._foot_connections = [
            c for c in self._reader.connections if c.topic == "/spot/status/feet"
        ]
        self._replay_connections = self._imu_connections + self._foot_connections

        if not self._imu_connections:
            raise ValueError("No /imu topic found in bag.")
        if not self._foot_connections:
            raise ValueError("No /spot/status/feet topic found in bag.")

        # Running state statistics for auto-detecting which contact-state code is stance.
        self._state_counts: dict[int, int] = {}
        self._state_mean_z: dict[int, float] = {}
        self._min_state_observations = 8

        self._reset_stream_state()

    def __enter__(self) -> RealReplay:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._reader.close()
        except Exception:
            pass
        self._closed = True

    @staticmethod
    def _ros_time_to_s(time_msg) -> float | None:
        """Convert a ROS-style time object with sec/nanosec fields to seconds."""
        if time_msg is None:
            return None
        sec = getattr(time_msg, "sec", None)
        nanosec = getattr(time_msg, "nanosec", None)
        if sec is None or nanosec is None:
            return None
        sec_i = int(sec)
        nsec_i = int(nanosec)
        if sec_i == 0 and nsec_i == 0:
            return None
        return float(sec_i) + float(nsec_i) * 1e-9

    @classmethod
    def _sensor_time_s(cls, msg) -> float | None:
        """Prefer explicit sensor/acquisition timestamps, then header.stamp."""
        for field in (
            "sensor_timestamp",
            "acquisition_timestamp",
            "measurement_timestamp",
            "source_timestamp",
        ):
            ts = cls._ros_time_to_s(getattr(msg, field, None))
            if ts is not None:
                return ts

        header = getattr(msg, "header", None)
        return cls._ros_time_to_s(getattr(header, "stamp", None))

    @classmethod
    def _msg_time_s(cls, msg, record_time_s: float) -> float:
        ts = cls._sensor_time_s(msg)
        if ts is not None:
            return ts
        return float(record_time_s)

    def _reset_stream_state(self) -> None:
        self._k = 0
        self._last_sample_time_s: float | None = None
        self._last_imu_time_s: float | None = None
        self._last_contact_emit_time_s: float | None = None
        self._last_omega = np.zeros(3, dtype=float)
        self._last_f = np.zeros(3, dtype=float)
        self._has_imu = False

        self._prev_raw_contacts: dict[str, int | None] = {
            foot: None for foot in self.foot_names
        }

    def _next_dt(self, current_time_s: float) -> float:
        if self._last_sample_time_s is None:
            return 0.0
        return max(current_time_s - self._last_sample_time_s, 0.0)

    def _emit_sample(
        self,
        timestamp_s: float,
        old_contacts: dict[str, np.ndarray],
        new_contacts: dict[str, np.ndarray],
        callback: Callable[[Measurement], None],
    ) -> None:
        measurement = Measurement(
            k=self._k,
            dt=self._next_dt(timestamp_s),
            omega_meas=self._last_omega.copy(),
            f_meas=self._last_f.copy(),
            old_contacts=old_contacts,
            new_contacts=new_contacts,
            timestamp_s=timestamp_s,
        )
        callback(measurement)
        self._k += 1
        self._last_sample_time_s = timestamp_s

    def _update_contact_state_stats(self, state_value: int, z_body: float) -> None:
        count = self._state_counts.get(state_value, 0) + 1
        prev_mean = self._state_mean_z.get(state_value, 0.0)
        mean = prev_mean + (z_body - prev_mean) / count
        self._state_counts[state_value] = count
        self._state_mean_z[state_value] = mean

    def _maybe_infer_contact_state(self) -> None:
        if self.contact_state_value is not None:
            return

        candidates = [
            (self._state_mean_z[state], state)
            for state, count in self._state_counts.items()
            if count >= self._min_state_observations
        ]
        if len(candidates) < 2:
            return

        # In body coordinates, stance feet tend to have lower (more negative) z.
        self.contact_state_value = min(candidates)[1]

    def _process_imu_message(
        self,
        msg,
        timestamp_s: float,
        callback: Callable[[Measurement], None],
    ) -> None:
        t_imu = float(timestamp_s)

        self._last_imu_time_s = t_imu
        self._last_omega = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],  # type: ignore[attr-defined]
            dtype=float,
        )
        self._last_f = np.array(
            [
                msg.linear_acceleration.x,  # type: ignore[attr-defined]
                msg.linear_acceleration.y,  # type: ignore[attr-defined]
                msg.linear_acceleration.z,  # type: ignore[attr-defined]
            ],
            dtype=float,
        )
        self._has_imu = True

        self._emit_sample(
            timestamp_s=t_imu,
            old_contacts={},
            new_contacts={},
            callback=callback,
        )

    def _process_foothold_message(
        self,
        msg,
        timestamp_s: float,
        callback: Callable[[Measurement], None],
    ) -> None:
        t_contact = float(timestamp_s)
        states = list(msg.states)  # type: ignore[attr-defined]

        raw_contacts: dict[str, int] = {}
        foot_measurements: dict[str, np.ndarray] = {}

        for i, foot_state in enumerate(states):
            if i >= len(self.foot_names):
                break
            foot = self.foot_names[i]
            raw = int(foot_state.contact)
            x = float(foot_state.foot_position_rt_body.x)
            y = float(foot_state.foot_position_rt_body.y)
            z = float(foot_state.foot_position_rt_body.z)

            raw_contacts[foot] = raw
            foot_measurements[foot] = np.array([x, y, z], dtype=float)
            self._update_contact_state_stats(raw, z)

        self._maybe_infer_contact_state()

        start_feet: set[str] = set()
        continuing_feet: set[str] = set()
        contact_state = self.contact_state_value

        if contact_state is not None:
            for foot, curr_raw in raw_contacts.items():
                prev_raw = self._prev_raw_contacts[foot]
                if prev_raw is not None:
                    prev_contact = prev_raw == contact_state
                    curr_contact = curr_raw == contact_state
                    if curr_contact and not prev_contact:
                        start_feet.add(foot)
                    if curr_contact and prev_contact:
                        continuing_feet.add(foot)

        for foot, curr_raw in raw_contacts.items():
            self._prev_raw_contacts[foot] = curr_raw

        # Emit foothold updates only on swing->stance transitions.
        if not start_feet or not self._has_imu:
            return
        if (
            self._last_contact_emit_time_s is not None
            and abs(t_contact - self._last_contact_emit_time_s) < 1e-12
        ):
            return

        old_contacts = {
            foot: foot_measurements[foot].copy()
            for foot in self.foot_names
            if foot in continuing_feet
        }
        new_contacts = {
            foot: foot_measurements[foot].copy()
            for foot in self.foot_names
            if foot in start_feet
        }

        self._emit_sample(
            timestamp_s=t_contact,
            old_contacts=old_contacts,
            new_contacts=new_contacts,
            callback=callback,
        )
        self._last_contact_emit_time_s = t_contact

    def replay(self, callback: Callable[[Measurement], None]) -> None:
        """Deliver interleaved IMU/contact measurements to a callback."""
        if self._closed:
            raise RuntimeError("RealReplay reader is closed.")

        self._reset_stream_state()
        events = []
        for order, (connection, record_time_ns, rawdata) in enumerate(
            self._reader.messages(connections=self._replay_connections)
        ):
            msg = self._reader.deserialize(rawdata, connection.msgtype)
            record_time_s = float(record_time_ns) * 1e-9
            msg_time_s = self._msg_time_s(msg, record_time_s)
            events.append((msg_time_s, record_time_s, order, connection.topic, msg))

        # Use sensor-time ordering for consistent dt computation across topics.
        events.sort(key=lambda event: (event[0], event[1], event[2]))

        for msg_time_s, _, _, topic, msg in events:
            if topic == "/imu":
                self._process_imu_message(msg, msg_time_s, callback)
            elif topic == "/spot/status/feet":
                self._process_foothold_message(msg, msg_time_s, callback)
