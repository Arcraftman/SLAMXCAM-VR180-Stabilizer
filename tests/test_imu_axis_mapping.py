from __future__ import annotations

from math import sin

from slam_stabilizer.imu import load_imu_csv
from slam_stabilizer.models import LensProfile
from pathlib import Path
from slam_stabilizer.core.quaternion import Quat
from slam_stabilizer.core.stabilization import horizon_locked_target
from slam_stabilizer.vqf_fusion import fuse_6d_vqf


def test_load_imu_csv_applies_axis_mapping_before_integration(tmp_path) -> None:
    imu_path = tmp_path / "imu.csv"
    imu_path.write_text(
        "timestamp_s,gx,gy,gz\n"
        "0,1,0,0\n"
        "1,1,0,0\n",
        encoding="utf-8",
    )

    samples = load_imu_csv(
        imu_path,
        axis_rotation=[
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    )

    assert samples[-1].gyro_xyz == (0.0, 1.0, 0.0)
    assert abs(samples[-1].quaternion_wxyz[2] - sin(0.5)) < 1e-9


def test_load_imu_csv_applies_gyro_scale_before_integration(tmp_path) -> None:
    imu_path = tmp_path / "scaled_imu.csv"
    imu_path.write_text(
        "timestamp_s,gx,gy,gz\n"
        "0,0,0,1\n"
        "1,0,0,1\n",
        encoding="utf-8",
    )

    samples = load_imu_csv(imu_path, gyro_scale=0.45)

    assert samples[-1].gyro_xyz == (0.0, 0.0, 0.45)
    assert abs(samples[-1].quaternion_wxyz[3] - sin(0.225)) < 1e-9


def test_2026_profile_preserves_gravity_and_cancels_sensor_x_rotation(tmp_path) -> None:
    profile = LensProfile.from_file(Path(__file__).resolve().parents[1] / 'config/lenses/slam_xcam_2026.json')
    imu_path = tmp_path / 'landscape.csv'
    imu_path.write_text('timestamp_s,gx,gy,gz,ax,ay,az\n0,1,0,0,9.81,0,0\n1,1,0,0,9.81,0,0\n')
    samples = load_imu_csv(imu_path, axis_rotation=profile.raw['imu_to_camera_rotation'],
                          gyro_axis_rotation=profile.raw['imu_to_camera_gyro_rotation'])
    assert samples[-1].acceleration_xyz == (0.0, 9.81, 0.0)
    assert samples[-1].gyro_xyz == (0.0, -1.0, 0.0)
    assert abs(samples[-1].quaternion_wxyz[2] + sin(0.5)) < 1e-9


def test_2026_profile_fusion_keeps_level_camera_upright(tmp_path) -> None:
    profile = LensProfile.from_file(Path(__file__).resolve().parents[1] / 'config/lenses/slam_xcam_2026.json')
    path = tmp_path / 'stationary.csv'
    path.write_text('timestamp_s,gx,gy,gz,ax,ay,az\n' + ''.join(
        f'{i * 0.005},0,0,0,9.81,0,0\n' for i in range(400)))
    samples = load_imu_csv(path, axis_rotation=profile.raw['imu_to_camera_rotation'],
                          gyro_axis_rotation=profile.raw['imu_to_camera_gyro_rotation'])
    fused, _ = fuse_6d_vqf(samples)
    pose = Quat.from_iter(fused[-1].quaternion_wxyz)
    assert pose.angular_distance_deg(Quat.identity()) < 0.01
    assert horizon_locked_target(pose).angular_distance_deg(pose) < 0.01
