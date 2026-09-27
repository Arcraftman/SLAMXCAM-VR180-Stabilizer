from __future__ import annotations

import numpy as np

from .quaternion import Quat
from .stabilization import interpolate_quat


def build_rolling_shutter_matrices(
    imu_times: list[float],
    imu_quaternions: list[Quat],
    frame_start_times_s: list[float],
    readout_s: float,
    row_count: int,
    readout_times_s: list[float] | None = None,
) -> tuple[list[np.ndarray], float]:
    """Build q_row^-1 * q_mid camera-space matrices for every output row."""

    if readout_times_s is not None and len(readout_times_s) != len(frame_start_times_s):
        raise ValueError("Per-frame readout times must match frame timestamps.")
    if (readout_s <= 0.0 and not any(readout_times_s or [])) or row_count <= 0:
        return [], 0.0
    plans: list[np.ndarray] = []
    max_angle_deg = 0.0
    denominator = max(1, row_count - 1)
    for index, frame_start_s in enumerate(frame_start_times_s):
        frame_readout_s = max(0.0, readout_times_s[index] if readout_times_s is not None else readout_s)
        mid_time_s = frame_start_s + frame_readout_s * 0.5
        mid = interpolate_quat(imu_times, imu_quaternions, mid_time_s)
        matrices = np.empty((row_count, 3, 3), dtype=np.float32)
        for row in range(row_count):
            row_time_s = frame_start_s + frame_readout_s * (row / denominator)
            row_orientation = interpolate_quat(imu_times, imu_quaternions, row_time_s)
            relative = row_orientation.conjugate().mul(mid)
            max_angle_deg = max(max_angle_deg, Quat.identity().angular_distance_deg(relative))
            matrices[row] = np.asarray(relative.to_matrix3(), dtype=np.float32)
        plans.append(matrices)
    return plans, max_angle_deg
