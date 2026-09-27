"""Local real-video/SLAMIMU pairing and visual rotation diagnostics."""
from pathlib import Path
import json
import sqlite3
import argparse
import cv2
import numpy as np
from review_real_videos import probe, decode, observations, fit_axes_offset, angle
from slam_stabilizer.imu import load_slamimu
from slam_stabilizer.vqf_fusion import fuse_6d_vqf
from slam_stabilizer.core.quaternion import Quat
from slam_stabilizer.core.stabilization import interpolate_quat
from review_real_videos import rectilinear, encoder, pixel_motion

OUT=Path('outputs/slamimu_pair_review')
STEMS=['Slam_20260927_170347_823','Slam_20260927_170408_900']
PROFILE=np.array([[0,1,0],[1,0,0],[0,0,1]],float)


def residual(obs,times,quats,offset=0):
    values=[]
    for _,_,a,b,r,*_ in obs:
        qa=np.asarray(interpolate_quat(times,quats,a+offset).to_matrix3())
        qb=np.asarray(interpolate_quat(times,quats,b+offset).to_matrix3())
        values.append(angle(r.T@qb.T@qa))
    return values


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for stem in STEMS:
        path=Path('D:/Video')/(stem+'.mp4')
        sidecar=path.with_name(stem+'_motion.slamimu')
        meta=probe(path,'-show_streams','-show_format')
        v=next(s for s in meta['streams'] if s['codec_type']=='video')
        frame_data=probe(path,'-select_streams','v:0','-show_frames',
            '-show_entries','frame=best_effort_timestamp_time')['frames']
        pts=np.array([float(f['best_effort_timestamp_time']) for f in frame_data])
        raw=load_slamimu(sidecar,gyro_filter_window_s=0)
        current=load_slamimu(sidecar,axis_rotation=PROFILE.ravel().tolist(),gyro_filter_window_s=0)
        times=[s.timestamp_s for s in raw.samples]
        frame_times=np.asarray(raw.frame_times_s)
        frames=decode(path)
        assert len(frames)==len(frame_times)==len(pts)
        obs=observations(frames,frame_times)
        quats=[Quat.from_iter(s.quaternion_wxyz) for s in raw.samples]
        score,offset,basis=fit_axes_offset(obs,times,quats)
        converted=[Quat.from_matrix3((basis.T@np.asarray(q.to_matrix3())@basis).tolist()) for q in quats]
        baseline=[Quat.from_iter(s.quaternion_wxyz) for s in current.samples]
        fused,diagnostics=fuse_6d_vqf(current.samples)
        fused_q=[Quat.from_iter(s.quaternion_wxyz) for s in fused]
        half=len(obs)//2
        heldout=obs[half:]
        midpoint=(raw.rolling_shutter_skew_s or 0)*.5
        errors=residual(heldout,times,converted,offset)
        old_errors=residual(heldout,times,baseline,midpoint)
        fusion_errors=residual(heldout,times,fused_q,midpoint)
        gyro=np.array([s.gyro_xyz for s in raw.samples])
        acc=np.array([s.acceleration_xyz for s in raw.samples])
        codec=np.asarray(raw.frame_codec_pts_us)/1e6
        pts_diff=pts-(codec-codec[0])
        time_diff=pts-(frame_times-frame_times[0])
        row=dict(file=stem,width=v['width'],height=v['height'],frames=len(pts),
            gyro_samples=len(times),gyro_rate_hz=1/float(np.median(np.diff(times))),
            gyro_std_xyz=gyro.std(axis=0).tolist(),accel_std_xyz=acc.std(axis=0).tolist(),
            imu_coverage_s=[times[0],times[-1]],video_coverage_s=[frame_times[0],frame_times[-1]],
            video_pts_vs_codec_relative_max_error_ms=float(np.max(np.abs(pts_diff))*1000),
            video_pts_vs_sensor_relative_max_error_ms=float(np.max(np.abs(time_diff))*1000),
            rolling_shutter_readout_ms=raw.rolling_shutter_skew_s*1000,
            fitted_offset_s=offset,fitted_camera_to_sensor_basis=basis.tolist(),
            candidate_gyro_mapping=basis.T.tolist(),training_error_deg=score,
            heldout_candidate_error_deg=float(np.median(errors)),
            heldout_profile_gyro_error_deg=float(np.median(old_errors)),
            heldout_profile_vqf_error_deg=float(np.median(fusion_errors)),
            heldout_raw_visual_rotation_deg=float(np.median([angle(r) for _,_,_,_,r,*_ in heldout])),
            vqf_residual_bias_deg_s=list(diagnostics.bias_deg_s_xyz))
        results.append(row)
        np.savez_compressed(OUT/(stem+'_motion.npz'),frame_times=frame_times,pts=pts,
            imu_times=times,current_vqf=[q.as_tuple() for q in fused_q],
            candidate_gyro=[q.as_tuple() for q in converted],offset=offset,basis=basis)
        print(json.dumps(row,indent=2),flush=True)
    (OUT/'analysis.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


def compare(fixed=False):
    results=[]
    for stem in STEMS:
        source=Path('D:/Video')/(stem+'.mp4')
        print('Comparing '+stem,flush=True)
        originals=decode(source,1280)
        before=decode(OUT/(stem+('_after_vqf_normal.mp4' if fixed else '_before_vqf_normal.mp4')),1280)
        after=decode(OUT/(stem+('_after_vqf_orientation-lock.mp4' if fixed else '_after_vqf_normal.mp4')),1280)
        horizon=after if fixed else decode(OUT/(stem+'_after_vqf_horizon-lock.mp4'),1280)
        assert len(originals)==len(before)==len(after)==len(horizon)
        suffix='_fixed_comparison' if fixed else '_comparison'
        writer=encoder(OUT/(stem+suffix+'.mp4'),1920,512,source)
        views=[[],[],[],[]]
        for i,frames in enumerate(zip(originals,before,after,horizon)):
            rects=[rectilinear(frame,np.eye(3)) for frame in frames]
            for destination,frame in zip(views,rects): destination.append(frame)
            panel=np.zeros((512,1920,3),np.uint8)
            panel[32:]=np.concatenate(rects[:3],axis=1)
            labels=['ORIGINAL','AFTER / normal following','AFTER / fixed orientation'] if fixed else ['ORIGINAL','BEFORE / wrong gyro axis','AFTER / VQF + exposure timing']
            for x,label in zip([10,650,1290],labels):
                cv2.putText(panel,label,(x,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            writer.stdin.write(panel.tobytes())
            if i in [30,90,150,240]: cv2.imwrite(str(OUT/(stem+suffix+f'_{i:03d}.jpg')),panel)
        writer.stdin.close()
        if writer.wait()!=0: raise RuntimeError('Comparison encode failed')
        metrics=[pixel_motion(v) for v in views]
        row=dict(file=stem,frames=len(originals),fixed_comparison=fixed,original=metrics[0],before=metrics[1],after=metrics[2],horizon=metrics[3],
                 method='background feature motion remeasured in rendered 640x480 90-degree left-eye views; scene-specific proxy')
        results.append(row);print(json.dumps(row,indent=2),flush=True)
    (OUT/('rendered_fixed_motion_metrics.json' if fixed else 'rendered_motion_metrics.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--compare',action='store_true')
    parser.add_argument('--fixed',action='store_true')
    args=parser.parse_args()
    if args.compare: compare(args.fixed)
    else: main()
