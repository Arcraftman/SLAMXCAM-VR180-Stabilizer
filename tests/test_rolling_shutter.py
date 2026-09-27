from __future__ import annotations

from math import cos, sin
import numpy as np
import pytest

from slam_stabilizer.core.quaternion import Quat
from slam_stabilizer.core.rolling_shutter import build_rolling_shutter_matrices


def test_stationary_rolling_shutter_matrices_are_identity() -> None:
    matrices, max_angle = build_rolling_shutter_matrices(
        [0.0, 1.0],
        [Quat.identity(), Quat.identity()],
        [0.25],
        readout_s=0.03,
        row_count=4,
    )

    assert len(matrices) == 1
    assert max_angle < 1e-9
    assert abs(float(matrices[0][2][0][0]) - 1.0) < 1e-9


def test_rolling_shutter_rows_rotate_toward_mid_exposure() -> None:
    end = Quat(cos(0.5), 0.0, 0.0, sin(0.5))
    matrices, max_angle = build_rolling_shutter_matrices(
        [0.0, 1.0],
        [Quat.identity(), end],
        [0.2],
        readout_s=0.1,
        row_count=3,
    )

    assert 2.8 < max_angle < 3.0
    assert matrices[0][0][1][0] > 0.0
    assert matrices[0][2][1][0] < 0.0


def test_per_frame_readout_uses_each_frames_duration() -> None:
    end = Quat(cos(0.5), 0.0, 0.0, sin(0.5))
    matrices, maximum = build_rolling_shutter_matrices(
        [0.0, 1.0], [Quat.identity(), end], [0.2, 0.4],
        readout_s=0.05, row_count=3, readout_times_s=[0.0, 0.1],
    )
    np.testing.assert_allclose(matrices[0], np.repeat(np.eye(3)[None], 3, axis=0), atol=1e-6)
    np.testing.assert_allclose(matrices[1][1], np.eye(3), atol=1e-6)
    assert matrices[1][0][1][0] > 0.0
    assert 2.8 < maximum < 3.0


def test_per_frame_readout_requires_matching_frame_count() -> None:
    with pytest.raises(ValueError, match='must match'):
        build_rolling_shutter_matrices([0.0, 1.0], [Quat.identity()] * 2,
                                      [0.2], 0.03, 4, readout_times_s=[0.01, 0.02])
