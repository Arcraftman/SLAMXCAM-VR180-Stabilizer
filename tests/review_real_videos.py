"""Reproducible, offline CAMM/visual-motion review of the supplied clips.

This is an experiment, not a calibrated production importer. Camera rays use
the nominal equidistant 180-degree model; sensor axes/time offset are fitted
on the first half and evaluated on the second half of each recording.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import struct
import subprocess
import argparse

import cv2
import numpy as np

from slam_stabilizer.core.quaternion import Quat
from slam_stabilizer.core.stabilization import interpolate_quat, build_frame_stabilization, SmoothParams
from slam_stabilizer.cpu_renderer import _build_eye_maps

FF = Path('D:/slamworld-gaussian-client/portable-engine/tools')
OUT = Path('outputs/real_video_review')
VIDEOS = [Path('D:/Video/Slam_20260927_162750_083.mp4'),
          Path('D:/Video/Slam_20260927_162809_008.mp4')]


def probe(path, *args):
    return json.loads(subprocess.check_output([
        str(FF / 'ffprobe.exe'), '-v', 'error', *args, '-of', 'json', str(path)]))


def read_motion(path):
    packets = probe(path, '-select_streams', 'd:0', '-show_packets')['packets']
    streams = {0: [], 2: [], 3: []}
    with path.open('rb') as source:
        for packet in packets:
            source.seek(int(packet['pos']))
            data = source.read(int(packet['size']))
            if len(data) != 16:
                continue
            # These files use uint32 type followed by three float32 values,
            # unlike the standard reserved-uint16/type-uint16 CAMM header.
            kind, x, y, z = struct.unpack('<Ifff', data)
            if kind in streams:
                streams[kind].append((float(packet['pts_time']), [x, y, z]))
    if not streams[0]:
        raise ValueError('No orientation packets')
    times, quats = [], []
    for time, vector in streams[0]:
        angle = np.linalg.norm(vector)
        scale = math.sin(angle / 2) / angle if angle > 1e-10 else .5
        times.append(time)
        quats.append(Quat(math.cos(angle / 2), *(np.array(vector) * scale)))
    return times, quats, streams


def decode(path, width=960):
    proc = subprocess.Popen([str(FF / 'ffmpeg.exe'), '-v', 'error', '-i', str(path),
        '-an', '-vf', f'scale={width}:{width//2}', '-fps_mode', 'passthrough',
        '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
    size = width * (width // 2) * 3
    frames = []
    while True:
        data = proc.stdout.read(size)
        if not data:
            break
        if len(data) != size:
            raise RuntimeError('Truncated decoded frame')
        frames.append(np.frombuffer(data, np.uint8).reshape(width//2, width, 3).copy())
    proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError('Video decode failed')
    return frames


def rays(points, size):
    p = points.reshape(-1, 2)
    xy = np.column_stack((p[:, 0] - (size-1)/2, (size-1)/2 - p[:, 1])) / (size/2)
    radius = np.linalg.norm(xy, axis=1)
    theta = radius * np.pi / 2
    scale = np.sin(theta) / np.maximum(radius, 1e-10)
    return np.column_stack((xy * scale[:, None], np.cos(theta)))


def fit_rotation(a, b):
    weights = np.ones(len(a))
    for _ in range(5):
        u, _, vt = np.linalg.svd((b * weights[:, None]).T @ a)
        r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        error = np.linalg.norm(a @ r.T - b, axis=1)
        weights = np.minimum(1, .003 / np.maximum(error, 1e-10))
    return r, float(np.median(error))


def observations(frames, times, stride=3):
    size = frames[0].shape[0]
    yy, xx = np.indices((size, size))
    mask = (((xx-size/2)**2 + (yy-size/2)**2 < (size*.38)**2) &
            (yy < size*.83)).astype(np.uint8)*255
    result = []
    for i in range(0, len(frames)-stride, stride):
        a = cv2.cvtColor(frames[i][:, :size], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frames[i+stride][:, :size], cv2.COLOR_BGR2GRAY)
        p = cv2.goodFeaturesToTrack(a, 400, .015, 8, mask=mask)
        if p is None:
            continue
        q, status, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(25, 25), maxLevel=3)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(25, 25), maxLevel=3)
        good = (status.ravel()==1) & (back_status.ravel()==1) & (np.linalg.norm(back-p, axis=2).ravel()<.7)
        if good.sum() < 30:
            continue
        ra, rb = rays(p[good], size), rays(q[good], size)
        r, error = fit_rotation(ra, rb)
        result.append((i, i+stride, float(times[i]), float(times[i+stride]), r, error, ra, rb))
    return result


def angle(matrix):
    return np.rad2deg(np.arccos(np.clip((np.trace(matrix)-1)/2, -1, 1)))


def fit_axes_offset(obs, times, quats):
    train = obs[:len(obs)//2]
    candidates = []
    # +/-B are equivalent under conjugation; retain proper bases only.
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            basis = np.eye(3)[:, perm] @ np.diag(signs)
            if np.linalg.det(basis) > .5:
                candidates.append(basis)
    best = None
    for offset in np.arange(-.4, .401, .01):
        relative = [np.asarray(interpolate_quat(times, quats, b+offset).to_matrix3()).T @
                    np.asarray(interpolate_quat(times, quats, a+offset).to_matrix3())
                    for _, _, a, b, *_ in train]
        for basis in candidates:
            errors = [angle(r.T @ basis.T @ p @ basis) for p, (_, _, _, _, r, *_) in zip(relative, train)]
            score = float(np.median(errors))
            if best is None or score < best[0]:
                best = score, float(offset), basis
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in VIDEOS:
        print(f'Inspecting {path.name}', flush=True)
        meta = probe(path, '-show_streams', '-show_format')
        video = next(s for s in meta['streams'] if s['codec_type']=='video')
        pts = probe(path, '-select_streams', 'v:0', '-show_frames',
                    '-show_entries', 'frame=best_effort_timestamp_time')['frames']
        frame_times = [float(f['best_effort_timestamp_time']) for f in pts]
        times, quats, streams = read_motion(path)
        frames = decode(path)
        assert len(frames) == len(frame_times)
        obs = observations(frames, frame_times)
        frozen = all(q.angular_distance_deg(quats[0]) < 1e-6 for q in quats)
        if frozen:
            # Timing/axes cannot be estimated from a constant signal.
            score, offset, basis = None, 0.0, np.eye(3)
        else:
            score, offset, basis = fit_axes_offset(obs, times, quats)
        converted = [Quat.from_matrix3((basis.T @ np.asarray(q.to_matrix3()) @ basis).tolist()) for q in quats]
        errors, raw_angles = [], []
        for _, _, a, b, r, *_ in obs[len(obs)//2:]:
            qa = np.asarray(interpolate_quat(times, converted, a+offset).to_matrix3())
            qb = np.asarray(interpolate_quat(times, converted, b+offset).to_matrix3())
            errors.append(angle(r.T @ qb.T @ qa))
            raw_angles.append(angle(r))
        row = dict(file=str(path), width=video['width'], height=video['height'],
                   frame_count=len(frames), duration_s=float(video['duration']),
                   nominal_fps=video['r_frame_rate'], packet_counts={k:len(v) for k,v in streams.items()},
                   orientation_rate_hz=1/float(np.median(np.diff(times))),
                   frozen_motion_data=frozen,
                   unique_sensor_values={k:len(set(tuple(v) for _,v in rows)) for k,rows in streams.items()},
                   fitted_offset_s=None if frozen else offset,
                   camera_to_camm_basis=None if frozen else basis.tolist(),
                   training_median_error_deg=score, heldout_median_error_deg=float(np.median(errors)),
                   heldout_raw_rotation_deg=float(np.median(raw_angles)))
        summary.append(row)
        print(json.dumps(row, indent=2), flush=True)
        np.savez_compressed(OUT / (path.stem+'_analysis.npz'),
                           frame_times=frame_times, pose_times=times,
                           pose_quats=[q.as_tuple() for q in converted], basis=basis, offset=offset)
    (OUT/'analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def rectilinear(frame, correction, width=640, height=480):
    size = frame.shape[0]
    yy, xx = np.indices((height, width), dtype=np.float32)
    dirs = np.stack(((xx-(width-1)/2)/(width/2),
                     ((height-1)/2-yy)/(width/2), np.ones_like(xx))).reshape(3,-1)
    dirs /= np.linalg.norm(dirs, axis=0)
    src = correction @ dirs
    theta = np.arccos(np.clip(src[2],-1,1))
    phi = np.arctan2(src[1],src[0])
    radius = theta / (np.pi/2) * (size/2)
    x = ((size-1)/2 + radius*np.cos(phi)).reshape(height,width).astype(np.float32)
    y = ((size-1)/2 - radius*np.sin(phi)).reshape(height,width).astype(np.float32)
    result = cv2.remap(frame[:,:size],x,y,cv2.INTER_LINEAR)
    result[theta.reshape(height,width)>np.pi/2]=0
    return result


def encoder(path, width, height, source):
    return subprocess.Popen([str(FF/'ffmpeg.exe'), '-y','-v','error',
        '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r','30',
        '-i','-','-i',str(source),'-map','0:v:0','-map','1:a?',
        '-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p',
        '-c:a','aac','-movflags','+faststart',str(path)], stdin=subprocess.PIPE)


def pixel_motion(frames):
    values=[]
    height,width=frames[0].shape[:2]
    mask=np.zeros((height,width),np.uint8)
    # Ceiling/background, away from the seated person and near desk parallax.
    mask[20:height//2,60:width-60]=255
    for i in range(len(frames)-1):
        a=cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY)
        b=cv2.cvtColor(frames[i+1],cv2.COLOR_BGR2GRAY)
        p=cv2.goodFeaturesToTrack(a,250,.015,8,mask=mask)
        if p is None:
            continue
        q,status,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(21,21),maxLevel=3)
        back,back_status,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(21,21),maxLevel=3)
        good=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(back-p,axis=2).ravel()<.7)
        if good.sum()>20:
            values.append(float(np.median(np.linalg.norm(q-p,axis=2).ravel()[good])))
    return dict(valid_pairs=len(values),median_px_per_frame=float(np.median(values)),
                rms_px_per_frame=float(np.sqrt(np.mean(np.square(values)))))


def preview(mode='normal'):
    results=[]
    for source in VIDEOS:
        print('Rendering experiment: '+source.name, flush=True)
        saved=np.load(OUT/(source.stem+'_analysis.npz'))
        times=saved['frame_times'].tolist()
        frames=decode(source,1280)
        obs=observations(frames,times,stride=1)
        measured={i:r for i,_,_,_,r,*_ in obs}
        camera=[Quat.identity()]
        missing=[]
        for i in range(len(frames)-1):
            if i not in measured:
                missing.append(i)
            step=measured.get(i,np.eye(3))
            camera.append(camera[-1].mul(Quat.from_matrix3(step.T.tolist())))
        plans=build_frame_stabilization(times,camera,len(frames),30,stabilization_mode=mode,
            params=SmoothParams(smooth_ms=0 if mode=='orientation-lock' else 600,max_correction_deg=15),frame_times_s=times)
        corrections=[np.asarray(p.correction_matrix3,dtype=np.float32) for p in plans]
        prefix=source.stem+('_fixed' if mode=='orientation-lock' else '')
        comparison=encoder(OUT/(prefix+'_comparison.mp4'),1280,512,source)
        stereo=encoder(OUT/(prefix+'_visual_experiment_sbs.mp4'),1280,640,source)
        camm_compare=encoder(OUT/(prefix+'_camm_comparison.mp4'),1280,512,source) if mode=='normal' else None
        original_views=[];stable_views=[]
        raw_motion=[]; residual_motion=[]
        for i,j,_,_,r,*_ in obs:
            raw_motion.append(angle(r))
            residual_motion.append(angle(corrections[j].T @ r @ corrections[i]))
        for i,frame in enumerate(frames):
            original=rectilinear(frame,np.eye(3))
            stable=rectilinear(frame,corrections[i])
            original_views.append(original);stable_views.append(stable)
            panel=np.zeros((512,1280,3),np.uint8)
            panel[32:,:640]=original;panel[32:,640:]=stable
            cv2.putText(panel,'ORIGINAL / left eye / 90 deg',(12,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            label='VISUAL FIXED VIEW' if mode=='orientation-lock' else 'VISUAL SMOOTHING'
            cv2.putText(panel,label+' / experiment',(650,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            comparison.stdin.write(panel.tobytes())
            panel[32:,640:]=original
            panel[:32,640:]=0
            cv2.putText(panel,'CAMM NORMAL / frozen data = no correction',(650,23),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1)
            if camm_compare:
                camm_compare.stdin.write(panel.tobytes())
            x,y,valid=_build_eye_maps(640,corrections[i])
            left=cv2.remap(frame[:,:640],x,y,cv2.INTER_LINEAR)
            right=cv2.remap(frame[:,640:],x,y,cv2.INTER_LINEAR)
            left[~valid]=0;right[~valid]=0
            rendered=np.concatenate([left,right],axis=1)
            stereo.stdin.write(rendered.tobytes())
            if i in [30,90,150,240]:
                cv2.imwrite(str(OUT/(prefix+f'_compare_{i:03d}.jpg')),np.vstack([
                    np.concatenate([original,stable],axis=1),
                    np.concatenate([frame[:,:640],rendered[:,:640]],axis=1)]))
            if i%60==0:
                print(f'{prefix}: {i}/{len(frames)}',flush=True)
        for p in [comparison,stereo,camm_compare]:
            if p is None:
                continue
            p.stdin.close()
            if p.wait()!=0:
                raise RuntimeError('Preview encoding failed')
        row=dict(file=source.name,mode=mode,frames=len(frames),unmeasured_intervals=missing,
                 rendered_original_background_flow=pixel_motion(original_views),
                 rendered_stabilized_background_flow=pixel_motion(stable_views),
                 raw_median_rotation_deg_per_frame=float(np.median(raw_motion)),
                 visual_residual_median_rotation_deg_per_frame=float(np.median(residual_motion)),
                 raw_rms_rotation_deg_per_frame=float(np.sqrt(np.mean(np.square(raw_motion)))),
                 visual_residual_rms_rotation_deg_per_frame=float(np.sqrt(np.mean(np.square(residual_motion)))),
                 max_correction_deg=max(Quat.identity().angular_distance_deg(Quat.from_iter(p.correction_wxyz)) for p in plans),
                 method='optical flow fit on nominal fisheye rays; shared stereo correction; approximate model; not independent validation')
        results.append(row)
        print(json.dumps(row,indent=2),flush=True)
    (OUT/f'visual_experiment_{mode}_metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--mode',choices=['normal','orientation-lock'],default='normal')
    args=parser.parse_args()
    if args.preview:
        preview(args.mode)
    else:
        main()
