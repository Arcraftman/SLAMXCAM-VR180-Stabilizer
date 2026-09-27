from __future__ import annotations

import numpy as np
import pytest

from slam_stabilizer.core.quaternion import Quat
from slam_stabilizer.imu import ImuSample
from slam_stabilizer.vqf_fusion import fuse_6d_vqf


def test_vqf_6d_stationary_aligns_acceleration_with_world_up() -> None:
    samples = [
        ImuSample(
            timestamp_s=index * 0.005,
            quaternion_wxyz=Quat.identity().as_tuple(),
            acceleration_xyz=(0.0, 0.0, 9.81),
            gyro_xyz=(0.0, 0.0, 0.0),
        )
        for index in range(400)
    ]

    fused, diagnostics = fuse_6d_vqf(samples)
    orientation = Quat.from_iter(fused[-1].quaternion_wxyz)

    np.testing.assert_allclose(
        orientation.rotate_vector((0.0, 0.0, 1.0)),
        (0.0, 1.0, 0.0),
        atol=1e-3,
    )
    assert abs(diagnostics.sample_rate_hz - 200.0) < 1e-6
    assert max(abs(value) for value in diagnostics.bias_deg_s_xyz) < 0.05


def test_vqf_dropped_samples_preserve_elapsed_rotation() -> None:
    samples = [
        ImuSample(i * 0.005, Quat.identity().as_tuple(),
                  (0.0, 0.0, 9.81), (0.0, 0.0, 1.0))
        for i in range(401)
    ]
    # A 250 ms gap must not be treated as a single 5 ms filter step.
    sparse = samples[:100] + samples[150:]
    reference, _ = fuse_6d_vqf(samples, orientation_window_s=0.0)
    actual, _ = fuse_6d_vqf(sparse, orientation_window_s=0.0)
    assert [s.timestamp_s for s in actual] == [s.timestamp_s for s in sparse]
    assert len(actual) == len(sparse)
    error = Quat.from_iter(actual[-1].quaternion_wxyz).angular_distance_deg(
        Quat.from_iter(reference[-1].quaternion_wxyz))
    assert error < 0.1


@pytest.mark.parametrize("times", [[0.0, 0.005, 0.005], [0.0, 0.01, 0.005],
                                   [0.0, float("nan"), 0.01]])
def test_vqf_rejects_invalid_timeline(times) -> None:
    samples = [ImuSample(t, Quat.identity().as_tuple(), (0.0, 0.0, 9.81),
                         (0.0, 0.0, 0.0)) for t in times]
    with pytest.raises(ValueError, match="strictly increasing"):
        fuse_6d_vqf(samples)
