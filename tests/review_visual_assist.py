"""Bounded shared-stereo visual residual experiment for the supplied clip.

The production IMU pipeline remains unchanged. This local experiment fuses
stereo-agreed background rotations with IMU relative rotations and exports
through the existing official renderer. It is a per-clip fallback experiment.
"""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np

from review_real_videos import decode, observations, fit_rotation, rectilinear, pixel_motion
from slam_stabilizer.models import LensProfile
from slam_stabilizer.imu import load_slamimu
from slam_stabilizer.vqf_fusion import fuse_6d_vqf
from slam_stabilizer.core.quaternion import Quat
from slam_stabilizer.core.stabilization import interpolate_quat, build_frame_stabilization, SmoothParams, FrameStabilization, bound_visual_pose
from slam_stabilizer.core.rolling_shutter import build_rolling_shutter_matrices
from slam_stabilizer.calibration_runtime import CalibrationRuntime
from slam_stabilizer.runtime_renderer import render_stabilized_sbs_runtime
from slam_stabilizer.cpu_renderer import CpuRenderOptions

SOURCE=Path('D:/Video/Slam_20260927_180016_489.mp4')
ROOT=Path('outputs/optimization_180016')


def render(width, output):
    saved=json.loads((ROOT/'hybrid_plan.json').read_text(encoding='utf-8'))
    plan=[FrameStabilization(**f) for f in saved['frames']]
    data=np.load(ROOT/'imu_for_rows.npz')
    rows,_=build_rolling_shutter_matrices(data['times'].tolist(),
        [Quat.from_iter(q) for q in data['quats'].tolist()],data['row_start'].tolist(),
        float(data['readout']),width//2,readout_times_s=data['readouts'].tolist())
    last=[-1]
    def progress(p,message):
        if p!=last[0]: print(p,message,flush=True);last[0]=p
    result=render_stabilized_sbs_runtime(SOURCE,output,plan,30,
        CpuRenderOptions(output_width=width,distortion_correction=False),
        CalibrationRuntime(),'slam_xcam_2026',rolling_shutter_plan=rows,progress=progress)
    print('RENDERED',output,result,flush=True)


def analyze():
    ROOT.mkdir(parents=True,exist_ok=True)
    cfg=LensProfile.from_file(Path('config/lenses/slam_xcam_2026.json')).raw
    motion=load_slamimu(SOURCE.with_name(SOURCE.stem+'_motion.slamimu'),
        axis_rotation=cfg['imu_to_camera_rotation'],gyro_axis_rotation=cfg['imu_to_camera_gyro_rotation'],
        gyro_filter_window_s=0)
    fused,_=fuse_6d_vqf(motion.samples)
    times=[s.timestamp_s for s in fused];imu=[Quat.from_iter(s.quaternion_wxyz) for s in fused]
    raw=[interpolate_quat(times,imu,t) for t in motion.frame_pose_times_s]
    frames=decode(SOURCE,1280)
    left={o[0]:o for o in observations(frames,motion.frame_times_s,stride=1)}
    right={o[0]:o for o in observations([f[:,640:] for f in frames],motion.frame_times_s,stride=1)}
    hybrid=[raw[0]];accepted=0;weights=[];agreements=[]
    for i in range(len(frames)-1):
        inertial=raw[i+1].conjugate().mul(raw[i])
        step=inertial;weight=0.0
        if i in left and i in right:
            a,b=left[i],right[i]
            qa,qb=Quat.from_matrix3(a[4].tolist()),Quat.from_matrix3(b[4].tolist())
            agreement=qa.angular_distance_deg(qb);agreements.append(agreement)
            visual,error=fit_rotation(np.concatenate([a[6],b[6]]),np.concatenate([a[7],b[7]]))
            estimate=Quat.from_matrix3(visual.tolist())
            discrepancy=inertial.angular_distance_deg(estimate)
            # Reject stereo disagreements, poor rigid fits and large outliers.
            # IMU supplies the pose whenever image evidence is unreliable.
            if agreement < .25 and error < .008 and discrepancy < 2.0:
                weight=.8*min(1.,(.25-agreement)/.15)
                step=inertial.slerp(estimate,weight)
                accepted+=1
        prediction=hybrid[-1].mul(step.conjugate())
        dt=motion.frame_times_s[i+1]-motion.frame_times_s[i]
        hybrid.append(bound_visual_pose(raw[i+1],prediction,dt));weights.append(weight)
    plan=build_frame_stabilization(motion.frame_times_s,hybrid,len(frames),30,
        params=SmoothParams(smooth_ms=0),stabilization_mode='orientation-lock',
        frame_times_s=motion.frame_times_s)
    result=dict(method='shared stereo visual high-frequency residual; 300 ms return to VQF trajectory; 3 degree deviation cap; fixed target; IMU rolling shutter',
        accepted_intervals=accepted,total_intervals=len(frames)-1,mean_visual_weight=float(np.mean(weights)),
        median_stereo_rotation_disagreement_deg=float(np.median(agreements)),
        frames=[asdict(f) for f in plan])
    (ROOT/'hybrid_plan.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(ROOT/'imu_for_rows.npz',times=times,quats=[q.as_tuple() for q in imu],
        row_start=[t+e/2 for t,e in zip(motion.frame_times_s,motion.frame_exposure_times_s)],
        readout=motion.rolling_shutter_skew_s,readouts=motion.frame_readout_times_s)
    print({k:v for k,v in result.items() if k!='frames'},flush=True)
    render(1280,ROOT/'hybrid_preview.mp4')
    output=decode(ROOT/'hybrid_preview.mp4',1280)
    assert len(output)==len(frames)
    metrics=pixel_motion([rectilinear(f,np.eye(3)) for f in output])
    (ROOT/'hybrid_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('HYBRID METRICS',metrics,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true')
    if parser.parse_args().full:
        render(6000,Path('outputs/final_stabilized')/(SOURCE.stem+'_optimized.mp4'))
    else:
        analyze()
