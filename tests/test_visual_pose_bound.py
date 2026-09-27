from math import cos, sin, radians
import pytest
from slam_stabilizer.core.quaternion import Quat
from slam_stabilizer.core.stabilization import bound_visual_pose


def roll(degrees):
    angle=radians(degrees)/2
    return Quat(cos(angle),0,0,sin(angle))


def test_visual_pose_outlier_cannot_pull_shared_pose_past_limit():
    raw=roll(35)
    limited=bound_visual_pose(raw,roll(100),1/30)
    assert raw.angular_distance_deg(limited)==pytest.approx(3,abs=1e-6)


def test_visual_pose_returns_to_imu_when_no_new_residual_is_observed():
    raw=roll(20)
    estimate=roll(23)
    for _ in range(90):
        estimate=bound_visual_pose(raw,estimate,1/30)
    assert raw.angular_distance_deg(estimate)<.001


def test_visual_pose_preserves_agreeing_estimate():
    raw=roll(47)
    assert raw.angular_distance_deg(bound_visual_pose(raw,raw,.1))<1e-6


@pytest.mark.parametrize('dt,tau,limit',[(0,.3,3),(.03,0,3),(.03,.3,0)])
def test_visual_pose_requires_positive_timing_and_limits(dt,tau,limit):
    with pytest.raises(ValueError,match='positive'):
        bound_visual_pose(Quat.identity(),Quat.identity(),dt,tau,limit)
