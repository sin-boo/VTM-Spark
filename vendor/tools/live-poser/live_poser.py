"""Live Poser — OpenSeeFace face tracking → KEYPOINT_SCHEMA bridge.

Pipeline: OpenSeeFace (live) → Label28 face + iris → (37,4) training schema.
Iris from custom iris_pose.pt and/or OpenSeeFace gaze, merged into slots 28/29.

  python live_poser.py              # UI
  python live_poser.py --cli -c 6   # headless OpenCV window
  python live_poser.py --list-cameras
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# Tk / ImageTk are only needed for LivePoserUI. Headless importers
# (real_stream live_poser_client, frozen builds) must not require them.
tk = None
ttk = None
messagebox = None
Image = None
ImageTk = None


def _require_tk() -> None:
    """Import Tk UI deps on demand; raise a clear error if missing."""
    global tk, ttk, messagebox, Image, ImageTk
    if tk is not None:
        return
    try:
        import tkinter as _tk
        from tkinter import messagebox as _messagebox
        from tkinter import ttk as _ttk
        from PIL import Image as _Image
        from PIL import ImageTk as _ImageTk
    except ImportError as exc:
        raise RuntimeError(
            'tkinter is required for the Live Poser UI. '
            'Use --cli for headless mode, or install a Python build with tk support.'
        ) from exc
    tk = _tk
    ttk = _ttk
    messagebox = _messagebox
    Image = _Image
    ImageTk = _ImageTk


from bridge import (
    BridgeFrame,
    build_bridge_frame,
    flip_iris_pair,
    write_bridge_json,
)
from cameras import (
    CameraCapture,
    CameraInfo,
    ensure_com,
    ensure_osf_on_path,
    list_cameras,
    resolve_openseeface,
    uninit_com,
)
from calibration import CenterCalibration, RelativePose, draw_coordinate_scope
from iris_tracker import (
    CustomIrisTracker,
    draw_iris,
    merge_iris,
    osf_gaze_to_iris,
    resolve_iris_weights,
)
from label_schema import draw_label28, flip_label28_x, osf_to_label28
from skeleton import (
    SkeletonHold,
    SkeletonLiteTracker,
    draw_skeleton,
    flip_body7_x,
    resolve_body7,
    resolve_skeleton_weights,
    synth_upper_body,
)
from tracking_filters import FilterSettings, MotionSmoother

ROOT = Path(__file__).resolve().parent
BRIDGE_JSON_PATH = ROOT / 'live_keypoints.json'
TORCH_TRAIN = ROOT.parents[1] / 'send2pod' / 'torch_train'
if str(TORCH_TRAIN) not in sys.path:
    sys.path.insert(0, str(TORCH_TRAIN))

# Downscale huge virtual-cam frames before tracking (keeps FPS usable).
MAX_TRACK_WIDTH = 1280
LIVE_IMAGE_SIZE = 768


def attach_norm_crop(
    bridge: BridgeFrame,
    *,
    body_lost: bool = False,
    calibrated: bool = False,
    image_size: int = LIVE_IMAGE_SIZE,
) -> BridgeFrame:
    """Fill model-space ``keypoints_norm`` + crop metadata on a pixel bridge frame."""
    try:
        from utils.coordinate_frames import (
            COORD_NORM_CROP,
            coord_meta_dict,
            webcam_pixels_to_norm_crop,
        )
    except Exception:
        return bridge

    src_w, src_h = int(bridge.image_wh[0]), int(bridge.image_wh[1])
    kps_norm, crop = webcam_pixels_to_norm_crop(
        bridge.keypoints, src_w, src_h, image_size=int(image_size)
    )
    bridge.keypoints_norm = kps_norm
    bridge.coord_space = COORD_NORM_CROP
    bridge.crop = crop
    bridge.image_size = int(image_size)
    meta = dict(bridge.meta or {})
    meta.update(
        coord_meta_dict(
            coord_space=COORD_NORM_CROP,
            source_wh=(src_w, src_h),
            crop=crop,
            image_size=int(image_size),
            mirrored=bool(bridge.mirrored),
            calibrated=bool(calibrated),
            skeleton_method=str(bridge.skeleton_method or 'none'),
            body_lost=bool(body_lost),
            extra={'fps': meta.get('fps')},
        )
    )
    # Keep original fps / notes if present.
    for key in ('fps', 'osf_model', 'sensitivity', 'smoothing', 'det_thr', 'note'):
        if key in (bridge.meta or {}) and key not in meta:
            meta[key] = bridge.meta[key]
        elif key in (bridge.meta or {}):
            meta[key] = bridge.meta[key]
    bridge.meta = meta
    return bridge


# --- drawing (14 OpenSeeFace pose points only) -------------------------------------

def draw_bbox(frame, face):
    # OpenSeeFace stores bbox as (y, x, h, w), not (x1, y1, x2, y2).
    if face.bbox is None or len(face.bbox) < 4:
        return
    y, x, h, w = [float(v) for v in face.bbox[:4]]
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 1, cv2.LINE_AA)


def format_euler(euler) -> str:
    if euler is None or len(euler) < 3:
        return 'n/a'
    return f'P:{float(euler[0]):6.1f}  Y:{float(euler[1]):6.1f}  R:{float(euler[2]):6.1f}'


def draw_hud(
    frame,
    faces_count: int,
    fps: float,
    mirror: bool,
    model: int,
    cam_name: str = '',
    reason: str = '',
    rel: RelativePose | None = None,
):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    line1 = f'Live Poser  |  KEYPOINT 37  |  m{model}  |  {fps:4.1f} FPS  |  {faces_count} face(s)'
    if mirror:
        line1 += '  |  mirror'
    if rel is not None and rel.calibrated:
        line1 += '  |  centered'
    cv2.putText(frame, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

    if cam_name:
        cv2.putText(frame, cam_name[:70], (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if faces_count <= 0:
        msg = reason or 'No face detected'
        cv2.putText(frame, msg[:90], (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 180, 255), 1, cv2.LINE_AA)
        return

    if rel is not None and rel.calibrated:
        pose_txt = f'rel P:{rel.pitch:+5.1f} Y:{rel.yaw:+5.1f} R:{rel.roll:+5.1f}'
    elif rel is not None:
        pose_txt = f'P:{rel.pitch:+5.1f} Y:{rel.yaw:+5.1f} R:{rel.roll:+5.1f}'
    else:
        pose_txt = ''
    cv2.putText(frame, pose_txt, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)


def annotate(
    frame,
    pts28: np.ndarray | None,
    fps,
    mirror,
    model,
    cam_name='',
    reason='',
    calib: CenterCalibration | None = None,
    rel: RelativePose | None = None,
    now: float = 0.0,
    faces_count: int = 0,
    eye_lower: np.ndarray | None = None,
    right_iris=None,
    left_iris=None,
    body7: np.ndarray | None = None,
    body_lost: bool = False,
):
    if pts28 is not None:
        draw_label28(frame, pts28, show_ids=True, eye_lower=eye_lower)
    draw_iris(frame, right_iris, left_iris)
    draw_skeleton(frame, body7, lost=body_lost)
    if calib is not None:
        draw_coordinate_scope(frame, calib, rel, now=now)
    draw_hud(
        frame,
        faces_count,
        fps,
        mirror,
        model,
        cam_name,
        reason=reason,
        rel=rel,
    )


def detection_reason(frame: np.ndarray, faces_alive: list) -> str:
    if faces_alive:
        return ''
    status, mean = frame_health(frame)
    if status == 'black':
        return 'Empty/black frames — try Refresh or another DroidCam resolution mode'
    if status == 'flat':
        return 'Flat/solid frame — virtual cam has no face content'
    return 'No face in view yet — center your face in DroidCam'


def fit_frame(frame: np.ndarray, max_w: int = MAX_TRACK_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# --- tracker factory ---------------------------------------------------------------

def build_tracker(
    width,
    height,
    model,
    faces,
    threads,
    detection_threshold,
    threshold,
    no_gaze,
    model_dir,
    osf: Path | None = None,
    try_hard: bool = True,
    use_retinaface: int = 0,
):
    from onnx_fix import patch_onnx_providers

    # Must run before OpenSeeFace creates InferenceSession (avoids TensorRT index crash).
    patch_onnx_providers(prefer_cuda=False)
    ensure_osf_on_path(resolve_openseeface(osf))
    from tracker import Tracker  # noqa: E402

    model = int(model)
    if model < -3 or model > 4:
        raise ValueError(f'Invalid OpenSeeFace model id {model} (expected -3..4)')

    # OpenSeeFace defaults (0.6) are too strict for many virtual-cam feeds.
    if detection_threshold is None:
        detection_threshold = 0.35
    if threshold is None:
        threshold = 0.35

    return Tracker(
        width,
        height,
        model_type=model,
        detection_threshold=float(detection_threshold),
        threshold=float(threshold),
        max_faces=faces,
        max_threads=threads,
        silent=True,
        model_dir=model_dir,
        no_gaze=no_gaze,
        use_retinaface=use_retinaface,
        try_hard=try_hard,
        max_feature_updates=900,
        static_model=True,
    )


def close_tracker(tracker) -> None:
    """Best-effort release of OSF ONNX sessions before rebuild/stop."""
    if tracker is None:
        return
    try:
        if hasattr(tracker, 'close'):
            tracker.close()
    except Exception:
        pass


def frame_health(frame: np.ndarray) -> tuple[str, float]:
    """Return (status, mean_luma). status: ok | black | flat."""
    if frame is None or frame.size == 0:
        return 'black', 0.0
    mean = float(np.mean(frame))
    std = float(np.std(frame))
    if mean < 5.0:
        return 'black', mean
    if std < 3.0:
        return 'flat', mean
    return 'ok', mean


def pick_default_camera(cameras: list[CameraInfo]) -> int:
    """Prefer a camera likely to show a real face feed."""
    if not cameras:
        return 0
    # Warudo placeholders are often solid color when unused.
    skip = ('warudo',)
    prefer_order = ('droidcam', 'obs', 'webcam', 'usb', 'hd ', 'integrated', 'nizima', 'camera')
    for key in prefer_order:
        for i, c in enumerate(cameras):
            lower = c.name.lower()
            if key in lower and not any(s in lower for s in skip):
                return i
    for i, c in enumerate(cameras):
        if not any(s in c.name.lower() for s in skip):
            return i
    return 0



# --- basic UI ----------------------------------------------------------------------

class LivePoserUI:
    """Basic face-tracking UI. Camera list includes Windows virtual cameras."""

    PREVIEW_MAX_W = 960

    def __init__(self, args: argparse.Namespace):
        _require_tk()
        self.args = args
        self.osf = resolve_openseeface(args.osf_dir)
        self.model_dir = str(args.model_dir or (self.osf / 'models'))
        self.cameras: list[CameraInfo] = []
        self.capture: CameraCapture | None = None
        self.tracker = None
        self.running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._photo = None
        self._latest_bgr = None
        self._fps = 0.0
        self._ema = 0.0
        self._status_faces = 0
        self._lock = threading.Lock()
        self.calib = CenterCalibration()
        self._latest_face = None
        self._latest_pts28 = None
        self._latest_shape = None
        self._latest_rel: RelativePose | None = None
        self._pose_out: dict | None = None
        self._latest_bridge: BridgeFrame | None = None
        self._bridge_text = ''
        self._mirror_prev = bool(self.args.mirror)
        self._iris_tracker: CustomIrisTracker | None = None
        self._skel_tracker: SkeletonLiteTracker | None = None
        self._smoother = MotionSmoother()
        self._skel_hold = SkeletonHold()

        ensure_com()
        self.root = tk.Tk()
        self.root.title('Live Poser — OSF → KEYPOINT_SCHEMA')
        self.root.minsize(980, 600)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.root.bind('<c>', lambda _e: self.center_now())
        self.root.bind('<C>', lambda _e: self.center_now())

        self._build()
        self.refresh_cameras()

    def _build(self):
        pad = {'padx': 8, 'pady': 4}
        top = ttk.Frame(self.root)
        top.pack(fill='x', **pad)

        ttk.Label(top, text='Camera').grid(row=0, column=0, sticky='w')
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(top, textvariable=self.cam_var, state='readonly', width=55)
        self.cam_combo.grid(row=0, column=1, sticky='ew', padx=4)
        self.refresh_btn = ttk.Button(top, text='Refresh', command=self.refresh_cameras)
        self.refresh_btn.grid(row=0, column=2)

        ttk.Label(top, text='Size').grid(row=1, column=0, sticky='w')
        size_row = ttk.Frame(top)
        size_row.grid(row=1, column=1, sticky='w')
        self.w_var = tk.IntVar(value=self.args.width)
        self.h_var = tk.IntVar(value=self.args.height)
        self.fps_var = tk.IntVar(value=self.args.fps)
        ttk.Spinbox(size_row, from_=160, to=1920, textvariable=self.w_var, width=6).pack(side='left')
        ttk.Label(size_row, text='x').pack(side='left', padx=2)
        ttk.Spinbox(size_row, from_=120, to=1080, textvariable=self.h_var, width=6).pack(side='left')
        ttk.Label(size_row, text='@').pack(side='left', padx=4)
        ttk.Spinbox(size_row, from_=1, to=60, textvariable=self.fps_var, width=4).pack(side='left')
        ttk.Label(size_row, text='FPS').pack(side='left', padx=2)

        ttk.Label(top, text='Model').grid(row=2, column=0, sticky='w')
        model_row = ttk.Frame(top)
        model_row.grid(row=2, column=1, sticky='w')
        self.model_var = tk.StringVar(value=str(self.args.model))
        ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=['-3', '-2', '-1', '0', '1', '2', '3', '4'],
            width=6,
            state='readonly',
        ).pack(side='left')
        ttk.Label(model_row, text='(3 = best / slower)').pack(side='left', padx=6)
        self.mirror_var = tk.BooleanVar(value=self.args.mirror)
        ttk.Checkbutton(
            model_row,
            text='Mirror',
            variable=self.mirror_var,
            command=self._on_mirror_toggle,
        ).pack(side='left', padx=8)
        self.gaze_var = tk.BooleanVar(value=not self.args.no_gaze)
        ttk.Checkbutton(model_row, text='Gaze (OSF)', variable=self.gaze_var).pack(side='left')
        self.iris_var = tk.BooleanVar(value=not getattr(self.args, 'no_iris', False))
        ttk.Checkbutton(model_row, text='Iris (custom)', variable=self.iris_var).pack(side='left', padx=4)
        self.skel_var = tk.BooleanVar(value=not getattr(self.args, 'no_skeleton', False))
        ttk.Checkbutton(model_row, text='Skeleton', variable=self.skel_var).pack(side='left', padx=4)

        ttk.Label(top, text='Sense').grid(row=3, column=0, sticky='w')
        sense_row = ttk.Frame(top)
        sense_row.grid(row=3, column=1, sticky='ew', pady=2)
        self.sensitivity_var = tk.DoubleVar(value=float(getattr(self.args, 'sensitivity', 50)))
        self.smoothing_var = tk.DoubleVar(value=float(getattr(self.args, 'smoothing', 40)))
        ttk.Label(sense_row, text='Sensitivity').pack(side='left')
        ttk.Scale(
            sense_row,
            from_=0,
            to=100,
            orient='horizontal',
            variable=self.sensitivity_var,
            length=160,
        ).pack(side='left', padx=4)
        self.sensitivity_lbl = ttk.Label(sense_row, text='50', width=3)
        self.sensitivity_lbl.pack(side='left')
        ttk.Label(sense_row, text='  Smoothing').pack(side='left', padx=(12, 0))
        ttk.Scale(
            sense_row,
            from_=0,
            to=100,
            orient='horizontal',
            variable=self.smoothing_var,
            length=160,
        ).pack(side='left', padx=4)
        self.smoothing_lbl = ttk.Label(sense_row, text='40', width=3)
        self.smoothing_lbl.pack(side='left')
        self.sensitivity_var.trace_add('write', lambda *_: self._update_filter_labels())
        self.smoothing_var.trace_add('write', lambda *_: self._update_filter_labels())
        self._update_filter_labels()

        top.columnconfigure(1, weight=1)

        note = ttk.Label(
            self.root,
            text='OSF face + MediaPipe body → KEYPOINT_SCHEMA  ·  Sensitivity↑ = stricter thr  ·  Smoothing = EMA',
            foreground='#555',
        )
        note.pack(anchor='w', padx=8)

        btns = ttk.Frame(self.root)
        btns.pack(fill='x', padx=8, pady=6)
        self.start_btn = ttk.Button(btns, text='Start', command=self.toggle)
        self.start_btn.pack(side='left')
        self.center_btn = ttk.Button(btns, text='Center', command=self.center_now)
        self.center_btn.pack(side='left', padx=(8, 0))
        ttk.Button(btns, text='Reset Center', command=self.reset_center).pack(side='left', padx=(6, 0))
        self.status_var = tk.StringVar(value='Select a camera, Start, look straight, then Center')
        ttk.Label(btns, textvariable=self.status_var).pack(side='left', padx=10)

        self.pose_var = tk.StringVar(value='coord: (not calibrated)  ·  press Center for instant origin')
        ttk.Label(self.root, textvariable=self.pose_var, foreground='#333').pack(anchor='w', padx=8)

        body = ttk.Frame(self.root)
        body.pack(fill='both', expand=True, padx=8, pady=8)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.preview = ttk.Label(body, anchor='center', background='#111')
        self.preview.grid(row=0, column=0, sticky='nsew', padx=(0, 8))

        side = ttk.Frame(body)
        side.grid(row=0, column=1, sticky='nsew')
        ttk.Label(side, text='Bridge · KEYPOINT_SCHEMA').pack(anchor='w')
        self.bridge_text = tk.Text(
            side,
            width=36,
            height=28,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#d8d8d8',
            insertbackground='#d8d8d8',
            relief='flat',
            wrap='word',
        )
        self.bridge_text.pack(fill='both', expand=True)
        self.bridge_text.insert('1.0', 'Start tracking to stream translated parameters…')
        self.bridge_text.configure(state='disabled')
        ttk.Label(
            side,
            text=f'Also writing {BRIDGE_JSON_PATH.name}',
            foreground='#666',
        ).pack(anchor='w', pady=(4, 0))

    def refresh_cameras(self):
        if self.running:
            return
        try:
            ensure_com()
            self.cameras = list_cameras(self.osf)
        except Exception as exc:
            messagebox.showerror('Camera list failed', str(exc))
            self.cameras = []
        labels = [c.label for c in self.cameras]
        self.cam_combo['values'] = labels
        if labels:
            pick = pick_default_camera(self.cameras)
            self.cam_combo.current(pick)
            self.status_var.set(
                f'Found {len(labels)} camera(s). Tip: use OBS/webcam with your real face in view.'
            )
        else:
            self.cam_var.set('')
            self.status_var.set('No cameras found — click Refresh')

    def _update_filter_labels(self):
        try:
            self.sensitivity_lbl.configure(text=str(int(round(float(self.sensitivity_var.get())))))
            self.smoothing_lbl.configure(text=str(int(round(float(self.smoothing_var.get())))))
        except Exception:
            pass

    def _filter_settings(self) -> FilterSettings:
        try:
            sens = float(self.sensitivity_var.get())
        except Exception:
            sens = 50.0
        try:
            smooth = float(self.smoothing_var.get())
        except Exception:
            smooth = 40.0
        return FilterSettings(sensitivity=sens, smoothing=smooth)

    def selected_camera(self) -> CameraInfo | None:
        i = self.cam_combo.current()
        if i < 0 or i >= len(self.cameras):
            # Fallback: match by label text
            label = self.cam_var.get()
            for c in self.cameras:
                if c.label == label:
                    return c
            return None
        return self.cameras[i]

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        cam = self.selected_camera()
        if cam is None:
            messagebox.showwarning('No camera', 'Pick a camera first (Refresh if the list is empty).')
            return
        if self._thread is not None and self._thread.is_alive():
            messagebox.showwarning('Busy', 'Wait for the previous session to stop.')
            return

        try:
            width = int(self.w_var.get())
            height = int(self.h_var.get())
            fps = int(self.fps_var.get())
            model_id = int(self.model_var.get())
        except Exception:
            messagebox.showerror('Invalid settings', 'Check width / height / FPS / model values.')
            return

        self._stop.clear()
        self.running = True
        self._latest_bgr = None
        self._latest_face = None
        self._latest_pts28 = None
        self._latest_shape = None
        self._latest_rel = None
        self._latest_bridge = None
        self._bridge_text = ''
        self._smoother.reset()
        self._skel_hold.reset()
        self.calib.clear()
        self.pose_var.set('coord: tracking… look straight at camera, then Center')
        self.start_btn.configure(text='Stop')
        self.cam_combo.configure(state='disabled')
        self.refresh_btn.configure(state='disabled')
        self.status_var.set(f'Starting {cam.name}…')
        self._set_bridge_text('Starting…')

        self._thread = threading.Thread(
            target=self._loop,
            args=(cam, width, height, fps, model_id),
            daemon=True,
        )
        self._thread.start()
        self.root.after(30, self._tick_ui)

    def stop(self):
        self._stop.set()
        self.running = False
        self.start_btn.configure(text='Start')
        self.cam_combo.configure(state='readonly')
        self.refresh_btn.configure(state='normal')
        self.status_var.set('Stopped')

    def _on_mirror_toggle(self):
        # Mirror changes display-space coords — clear center so it isn't flipped/wrong.
        self._smoother.reset()
        self._skel_hold.reset()
        if self.calib.ready:
            self.calib.clear()
            self.pose_var.set('coord: mirror toggled — press Center again')
            if self.running:
                self.status_var.set('Mirror changed — re-Center')

    def center_now(self):
        """Instant center calibration from the latest 28 labeling points."""
        with self._lock:
            pts28 = None if self._latest_pts28 is None else self._latest_pts28.copy()
            shape = self._latest_shape
            conf = 0.0
            face = self._latest_face
            if face is not None and getattr(face, 'conf', None) is not None:
                conf = float(face.conf)
        if not self.running:
            messagebox.showinfo('Center', 'Start tracking first, then press Center.')
            return
        if pts28 is None or shape is None:
            messagebox.showwarning('Center', 'No face tracked yet — look at the camera first.')
            return
        try:
            sample = self.calib.capture(pts28, shape, conf=conf)
            self.calib.flash_until = time.time() + 0.45
            ax, ay = sample.anchor_xy
            self.status_var.set(f'Centered instantly @ ({ax:.0f}, {ay:.0f})')
            self.pose_var.set(
                f'origin set  ·  P:{sample.pitch:.1f} Y:{sample.yaw:.1f} R:{sample.roll:.1f}  '
                f'·  scope locked (28 label pts)'
            )
        except Exception as exc:
            messagebox.showerror('Center failed', str(exc))

    def reset_center(self):
        self.calib.clear()
        self.pose_var.set('coord: (not calibrated)  ·  press Center for instant origin')
        if self.running:
            self.status_var.set('Center cleared')

    def _set_bridge_text(self, text: str):
        self.bridge_text.configure(state='normal')
        self.bridge_text.delete('1.0', 'end')
        self.bridge_text.insert('1.0', text)
        self.bridge_text.configure(state='disabled')

    def _resolve_iris(self, frame, face, pts28_raw, filters: FilterSettings):
        """Run custom iris + OSF gaze on raw (unmirrored) frame/pts."""
        custom = None
        osf = None
        if bool(self.iris_var.get()) and self._iris_tracker is not None and pts28_raw is not None:
            try:
                custom = self._iris_tracker.match_to_face(
                    frame,
                    pts28_raw,
                    conf=filters.iris_box_conf(),
                    pupil_vis_thr=filters.iris_pupil_vis(),
                )
            except Exception:
                custom = None
        if bool(self.gaze_var.get()) and face is not None:
            try:
                osf = osf_gaze_to_iris(face, conf_thr=filters.iris_pupil_vis())
            except Exception:
                osf = None
        return merge_iris(custom, osf, prefer='custom_then_osf')

    def _loop(self, cam: CameraInfo, width: int, height: int, fps: int, model_id: int):
        capture = None
        try:
            ensure_com()
            os.environ['OMP_NUM_THREADS'] = str(self.args.threads)
            ensure_osf_on_path(self.osf)

            # Custom iris (optional — soft-fail if weights / ultralytics missing)
            self._iris_tracker = None
            if bool(self.iris_var.get()):
                weights = resolve_iris_weights(getattr(self.args, 'iris_weights', None))
                if weights is None:
                    self.root.after(
                        0,
                        lambda: self.status_var.set('Iris weights not found — OSF gaze only'),
                    )
                else:
                    try:
                        self._iris_tracker = CustomIrisTracker(weights)
                        self.root.after(
                            0,
                            lambda w=str(weights.name): self.status_var.set(f'Iris model: {w}'),
                        )
                    except Exception as exc:
                        self.root.after(
                            0,
                            lambda e=str(exc): self.status_var.set(f'Iris load failed: {e}'),
                        )

            # Dedicated human body tracker (MediaPipe lite → KEYPOINT 30-36)
            self._skel_tracker = None
            if bool(self.skel_var.get()):
                sk_weights = resolve_skeleton_weights(getattr(self.args, 'skeleton_weights', None))
                backend = getattr(self.args, 'skeleton_backend', 'auto')
                try:
                    self.root.after(
                        0,
                        lambda: self.status_var.set('Loading MediaPipe Pose (lite)…'),
                    )
                    self._skel_tracker = SkeletonLiteTracker(
                        sk_weights,
                        device='cpu',
                        backend=backend,
                    )
                    method = self._skel_tracker.method
                    self.root.after(
                        0,
                        lambda m=method: self.status_var.set(f'Body tracker: {m}'),
                    )
                except Exception as exc:
                    self.root.after(
                        0,
                        lambda e=str(exc): self.status_var.set(
                            f'Body tracker failed ({e}) — synth fallback'
                        ),
                    )

            capture = CameraCapture(
                cam.index,
                width=width,
                height=height,
                fps=fps,
                backend='auto',
                osf=self.osf,
                camera_name=cam.name,
            )
            self.capture = capture

            frame = None
            for _ in range(30):
                if self._stop.is_set():
                    return
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)
            if frame is None:
                raise RuntimeError(
                    f'Camera opened ({capture.name} / {capture.backend}) but returned no frames. '
                    'Try another camera or click Refresh.'
                )

            health, mean = frame_health(frame)
            if health == 'black':
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        'Camera feed is black — try another DroidCam mode / Refresh'
                    ),
                )

            frame = fit_frame(frame)
            h, w = frame.shape[:2]
            filters0 = self._filter_settings()
            det_thr = filters0.detection_threshold()
            lms_thr = filters0.landmark_threshold()
            self.tracker = build_tracker(
                w,
                h,
                model_id,
                self.args.faces,
                self.args.threads,
                det_thr,
                lms_thr,
                no_gaze=not bool(self.gaze_var.get()),
                model_dir=self.model_dir,
                osf=self.osf,
                try_hard=True,
                use_retinaface=1,
            )
            name = capture.name
            backend = capture.backend
            self.root.after(0, lambda n=name, b=backend: self.status_var.set(f'Tracking: {n} [{b}]'))

            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                frame = fit_frame(frame)
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                # Track on the raw (unmirrored) frame so left/right stay correct.
                t0 = time.perf_counter()
                faces = self.tracker.predict(frame)
                dt = time.perf_counter() - t0
                inst = 1.0 / dt if dt > 0 else 0.0
                self._ema = inst if self._ema == 0 else (self._ema * 0.85 + inst * 0.15)
                self._fps = self._ema

                alive = [f for f in faces if getattr(f, 'alive', False) and f.lms is not None]
                filters = self._filter_settings()
                face_thr = filters.detection_threshold()
                alive = [
                    f
                    for f in alive
                    if float(getattr(f, 'conf', 1.0) or 0.0) >= face_thr * 0.5
                ]
                mirror = bool(self.mirror_var.get())
                pts28 = None
                eye_lower = None
                rel = None
                right_iris = left_iris = None
                iris_method = 'none'
                body7 = None
                skel_method = 'none'
                body_lost = False
                bridge = None

                # Body tracker can run even if face drops — hold last pose in red.
                raw_body = None
                raw_method = 'none'
                if bool(self.skel_var.get()):
                    raw_body, raw_method = resolve_body7(
                        frame,
                        None,  # don't synth from face here; hold handles misses
                        self._skel_tracker,
                        allow_synth_fallback=self._skel_tracker is None,
                        vis_thr=filters.body_vis_thr(),
                        box_conf=filters.iris_box_conf(),
                    )

                if alive:
                    face = alive[0]
                    pts28, eye_lower = osf_to_label28(face.lms)
                    lms_thr = filters.landmark_threshold()
                    if pts28 is not None:
                        weak = pts28[:, 2] < lms_thr
                        pts28[weak, 2] = 0.0
                    right_iris, left_iris, iris_method = self._resolve_iris(
                        frame, face, pts28, filters
                    )
                    # If dedicated body missed, optional face-driven synth only when
                    # we have never locked a pose yet (cold start).
                    if (
                        raw_body is None
                        and self._skel_hold.last is None
                        and bool(self.skel_var.get())
                    ):
                        raw_body, raw_method = resolve_body7(
                            frame,
                            pts28,
                            None,
                            allow_synth_fallback=True,
                            rel_pitch=0.0,
                            rel_yaw=0.0,
                            rel_roll=0.0,
                        )

                if bool(self.skel_var.get()):
                    body7, body_lost, skel_method = self._skel_hold.update(
                        raw_body, raw_method
                    )

                pts28, eye_lower, right_iris, left_iris, body7 = self._smoother.apply(
                    pts28=pts28,
                    eye_lower=eye_lower,
                    right_iris=right_iris,
                    left_iris=left_iris,
                    body7=body7,
                    smoothing=filters.smoothing,
                    freeze_body=body_lost,
                )

                display = cv2.flip(frame, 1) if mirror else frame
                if mirror:
                    if pts28 is not None:
                        pts28, eye_lower = flip_label28_x(
                            pts28, display.shape[1], eye_lower
                        )
                        right_iris, left_iris = flip_iris_pair(
                            right_iris, left_iris, display.shape[1]
                        )
                    if body7 is not None:
                        body7 = flip_body7_x(body7, display.shape[1])

                if alive and pts28 is not None:
                    rel = self.calib.remap(pts28, display.shape)
                    pose_dict = self.calib.to_dict(rel)
                    if (
                        bool(self.skel_var.get())
                        and skel_method == 'synthetic_from_face'
                        and not body_lost
                    ):
                        body7 = synth_upper_body(
                            pts28,
                            pitch=rel.pitch,
                            yaw=rel.yaw,
                            roll=rel.roll,
                        )
                        body7, body_lost, skel_method = self._skel_hold.update(
                            body7, 'synthetic_from_face'
                        )
                    bridge = build_bridge_frame(
                        pts28,
                        display.shape,
                        right_iris=right_iris,
                        left_iris=left_iris,
                        body7=body7,
                        mirrored=mirror,
                        iris_method=iris_method,
                        skeleton_method=skel_method,
                        pose=pose_dict,
                        meta={
                            'fps': self._fps,
                            'osf_model': model_id,
                            'sensitivity': filters.sensitivity,
                            'smoothing': filters.smoothing,
                            'det_thr': filters.detection_threshold(),
                            'body_lost': body_lost,
                        },
                    )
                    attach_norm_crop(
                        bridge,
                        body_lost=body_lost,
                        calibrated=bool(rel is not None and rel.calibrated),
                    )
                    try:
                        write_bridge_json(bridge, BRIDGE_JSON_PATH)
                    except Exception:
                        pass
                    with self._lock:
                        self._latest_face = alive[0]
                        self._latest_pts28 = pts28.copy()
                        self._latest_shape = display.shape
                        self._latest_rel = rel
                        self._pose_out = pose_dict
                        self._latest_bridge = bridge
                        self._bridge_text = bridge.summary_text()
                else:
                    # Face gone — still publish held skeleton so outputs don't drop.
                    if body7 is not None:
                        bridge = build_bridge_frame(
                            None,
                            display.shape,
                            body7=body7,
                            mirrored=mirror,
                            skeleton_method=skel_method,
                            meta={
                                'fps': self._fps,
                                'body_lost': True,
                                'note': 'face_lost_skeleton_held',
                            },
                        )
                        attach_norm_crop(
                            bridge,
                            body_lost=True,
                            calibrated=False,
                        )
                        try:
                            write_bridge_json(bridge, BRIDGE_JSON_PATH)
                        except Exception:
                            pass
                        with self._lock:
                            self._latest_face = None
                            self._latest_pts28 = None
                            self._latest_rel = None
                            self._latest_bridge = bridge
                            self._bridge_text = (
                                bridge.summary_text() + '\n\nface lost — skeleton HELD (red)'
                            )
                    else:
                        with self._lock:
                            self._latest_face = None
                            self._latest_pts28 = None
                            self._latest_rel = None
                            self._latest_bridge = None
                            self._bridge_text = 'No face — waiting…'

                reason = detection_reason(display, alive)
                annotate(
                    display,
                    pts28,
                    self._fps,
                    mirror,
                    model_id,
                    name,
                    reason=reason,
                    calib=self.calib,
                    rel=rel,
                    now=time.time(),
                    faces_count=len(alive),
                    eye_lower=eye_lower,
                    right_iris=right_iris,
                    left_iris=left_iris,
                    body7=body7,
                    body_lost=body_lost,
                )
                self._status_faces = len(alive)
                with self._lock:
                    self._latest_bgr = display
        except Exception as exc:
            err = f'{exc}\n\n{traceback.format_exc()}'
            print(err, file=sys.stderr)
            self.root.after(0, lambda e=str(exc): messagebox.showerror('Tracking error', e))
            self.root.after(0, self.stop)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            self.capture = None
            self.tracker = None
            self._iris_tracker = None
            if self._skel_tracker is not None:
                try:
                    self._skel_tracker.close()
                except Exception:
                    pass
            self._skel_tracker = None
            uninit_com()

    def _tick_ui(self):
        if not self.running:
            return
        with self._lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
            bridge_txt = self._bridge_text
        if bridge_txt:
            self._set_bridge_text(bridge_txt)
        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                max_w = self.PREVIEW_MAX_W
                if w > max_w:
                    scale = max_w / w
                    rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                img = Image.fromarray(rgb)
                self._photo = ImageTk.PhotoImage(img)
                self.preview.configure(image=self._photo)
                cam_name = getattr(self.capture, 'name', None) or 'camera'
                with self._lock:
                    rel = self._latest_rel
                    bridge = self._latest_bridge
                iris_tag = ''
                if bridge is not None:
                    iris_tag = f'  |  iris:{bridge.iris_method}'
                if rel is not None and rel.calibrated:
                    self.status_var.set(
                        f'{cam_name}  |  {self._fps:.1f} FPS  |  faces:{self._status_faces}  |  centered{iris_tag}'
                    )
                    self.pose_var.set(
                        f'rel  P:{rel.pitch:+6.1f}  Y:{rel.yaw:+6.1f}  R:{rel.roll:+6.1f}   '
                        f'nX:{rel.nx:+.2f}  nY:{rel.ny:+.2f}   dx:{rel.dx:+.0f} dy:{rel.dy:+.0f}'
                    )
                else:
                    self.status_var.set(
                        f'{cam_name}  |  {self._fps:.1f} FPS  |  faces:{self._status_faces}  |  press Center{iris_tag}'
                    )
            except Exception:
                pass
        self.root.after(33, self._tick_ui)

    def on_close(self):
        self._stop.set()
        self.running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


# --- CLI ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Live Poser — face tracking (OpenSeeFace). Virtual cameras included on Windows.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ui', action='store_true', default=True, help='Open basic UI (default)')
    p.add_argument('--cli', action='store_true', help='CLI / OpenCV window instead of UI')
    p.add_argument('-c', '--camera', default='0', help='Camera index (CLI mode)')
    p.add_argument('-l', '--list-cameras', action='store_true', help='List cameras and exit')
    p.add_argument('-W', '--width', type=int, default=640)
    p.add_argument('-H', '--height', type=int, default=360)
    p.add_argument('-F', '--fps', type=int, default=30)
    p.add_argument('--model', type=int, default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
    p.add_argument('--faces', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.35, help='Landmark confidence threshold')
    p.add_argument('--detection-threshold', type=float, default=0.35, help='Face detect threshold')
    p.add_argument(
        '--sensitivity',
        type=float,
        default=50.0,
        help='0-100; higher raises detection thresholds (stricter)',
    )
    p.add_argument(
        '--smoothing',
        type=float,
        default=40.0,
        help='0-100; higher smooths all motion more',
    )
    p.add_argument('--no-gaze', action='store_true', help='Disable OpenSeeFace gaze / pupils')
    p.add_argument('--no-iris', action='store_true', help='Disable custom iris_pose.pt tracker')
    p.add_argument(
        '--iris-weights',
        type=Path,
        default=None,
        help='Path to iris_pose.pt (default: pose-traker/models/iris_pose.pt)',
    )
    p.add_argument(
        '--no-skeleton',
        action='store_true',
        help='Disable body skeleton (MediaPipe / YOLO-n)',
    )
    p.add_argument(
        '--skeleton-weights',
        type=Path,
        default=None,
        help='pose_landmarker_lite.task or yolov8n-pose.pt',
    )
    p.add_argument(
        '--skeleton-backend',
        choices=['auto', 'mediapipe', 'yolo'],
        default='auto',
        help='Body tracker: MediaPipe Pose (default) or stock yolov8n-pose',
    )
    p.add_argument(
        '--skeleton-pt',
        action='store_true',
        help='(compat) Prefer YOLO .pt backend',
    )
    p.add_argument('--mirror', action='store_true')
    p.add_argument('--osf-dir', type=Path, default=None)
    p.add_argument('--model-dir', type=Path, default=None)
    return p.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    ensure_com()
    osf = resolve_openseeface(args.osf_dir)
    model_dir = str(args.model_dir or (osf / 'models'))
    ensure_osf_on_path(osf)
    os.environ['OMP_NUM_THREADS'] = str(args.threads)

    cam_index = int(args.camera) if str(args.camera).isdigit() else 0
    capture = CameraCapture(cam_index, args.width, args.height, args.fps, backend='auto', osf=osf)
    ok, probe = capture.read()
    if not ok or probe is None:
        raise RuntimeError('Camera opened but failed to read a frame')
    probe = fit_frame(probe)
    height, width = probe.shape[:2]
    tracker = build_tracker(
        width,
        height,
        args.model,
        args.faces,
        args.threads,
        args.detection_threshold,
        args.threshold,
        args.no_gaze,
        model_dir,
        osf=osf,
        try_hard=True,
        use_retinaface=1,
    )

    iris_tracker = None
    if not args.no_iris:
        weights = resolve_iris_weights(args.iris_weights)
        if weights is not None:
            try:
                iris_tracker = CustomIrisTracker(weights)
                print(f'Iris model: {weights}')
            except Exception as exc:
                print(f'Iris load failed ({exc}) — OSF gaze only')
        else:
            print('iris_pose.pt not found — OSF gaze only')

    skel_tracker = None
    skel_hold = SkeletonHold()
    if not args.no_skeleton:
        sk_w = resolve_skeleton_weights(args.skeleton_weights)
        backend = 'yolo' if args.skeleton_pt else args.skeleton_backend
        try:
            print('Loading body tracker…')
            skel_tracker = SkeletonLiteTracker(sk_w, device='cpu', backend=backend)
            print(f'Body tracker: {skel_tracker.method} ({skel_tracker.weights})')
        except Exception as exc:
            print(f'Body tracker failed ({exc}) — synth fallback')

    mirror = args.mirror
    paused = False
    fps = 0.0
    ema = 0.0
    calib = CenterCalibration()
    last_pts28 = None
    win = 'Live Poser — KEYPOINT_SCHEMA'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f'Tracking {capture.name} [{capture.backend}]… q quit / m mirror / c center / space pause')
    print(f'Bridge JSON → {BRIDGE_JSON_PATH}')
    health, _ = frame_health(probe)
    if health != 'ok':
        print(f'WARNING: camera frame looks {health}.')
    frame = probe
    faces = []
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame = fit_frame(frame)
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                t0 = time.perf_counter()
                faces = tracker.predict(frame)
                dt = time.perf_counter() - t0
                inst = 1.0 / dt if dt > 0 else 0.0
                ema = inst if ema == 0 else (ema * 0.85 + inst * 0.15)
                fps = ema
                alive = [f for f in faces if getattr(f, 'alive', False) and f.lms is not None]
                pts28 = None
                eye_lower = None
                rel = None
                right_iris = left_iris = None
                iris_method = 'none'
                body7 = None
                skel_method = 'none'
                body_lost = False
                if alive:
                    face = alive[0]
                    pts28, eye_lower = osf_to_label28(face.lms)
                    custom = None
                    osf_iris = None
                    if iris_tracker is not None:
                        try:
                            custom = iris_tracker.match_to_face(frame, pts28)
                        except Exception:
                            custom = None
                    if not args.no_gaze:
                        osf_iris = osf_gaze_to_iris(face)
                    right_iris, left_iris, iris_method = merge_iris(
                        custom, osf_iris, prefer='custom_then_osf'
                    )
                    body7 = None
                    skel_method = 'none'
                    if not args.no_skeleton:
                        raw_body, raw_method = resolve_body7(
                            frame,
                            pts28,
                            skel_tracker,
                            allow_synth_fallback=True,
                        )
                        body7, body_lost, skel_method = skel_hold.update(
                            raw_body, raw_method
                        )
                    display = cv2.flip(frame, 1) if mirror else frame
                    if mirror:
                        pts28, eye_lower = flip_label28_x(pts28, display.shape[1], eye_lower)
                        right_iris, left_iris = flip_iris_pair(
                            right_iris, left_iris, display.shape[1]
                        )
                        if body7 is not None:
                            body7 = flip_body7_x(body7, display.shape[1])
                    last_pts28 = pts28.copy()
                    rel = calib.remap(pts28, display.shape)
                    if (
                        not args.no_skeleton
                        and skel_method == 'synthetic_from_face'
                        and pts28 is not None
                        and not body_lost
                    ):
                        body7 = synth_upper_body(
                            pts28,
                            pitch=rel.pitch,
                            yaw=rel.yaw,
                            roll=rel.roll,
                        )
                        body7, body_lost, skel_method = skel_hold.update(
                            body7, 'synthetic_from_face'
                        )
                    bridge = build_bridge_frame(
                        pts28,
                        display.shape,
                        right_iris=right_iris,
                        left_iris=left_iris,
                        body7=body7,
                        mirrored=mirror,
                        iris_method=iris_method,
                        skeleton_method=skel_method,
                        pose=calib.to_dict(rel),
                        meta={'fps': fps, 'body_lost': body_lost},
                    )
                    attach_norm_crop(
                        bridge,
                        body_lost=body_lost,
                        calibrated=bool(rel is not None and rel.calibrated),
                    )
                    try:
                        write_bridge_json(bridge, BRIDGE_JSON_PATH)
                    except Exception:
                        pass
                else:
                    display = cv2.flip(frame, 1) if mirror else frame
                    # Face lost — keep publishing held skeleton when available.
                    if not args.no_skeleton:
                        held, body_lost, skel_method = skel_hold.update(None, 'none')
                        if held is not None:
                            bridge = build_bridge_frame(
                                None,
                                display.shape,
                                body7=held,
                                mirrored=mirror,
                                skeleton_method=skel_method,
                                meta={
                                    'fps': fps,
                                    'body_lost': True,
                                    'note': 'face_lost_skeleton_held',
                                },
                            )
                            attach_norm_crop(bridge, body_lost=True, calibrated=False)
                            try:
                                write_bridge_json(bridge, BRIDGE_JSON_PATH)
                            except Exception:
                                pass
                    last_pts28 = None
                annotate(
                    display,
                    pts28,
                    fps,
                    mirror,
                    args.model,
                    capture.name,
                    reason=detection_reason(display, alive),
                    calib=calib,
                    rel=rel,
                    now=time.time(),
                    faces_count=len(alive),
                    eye_lower=eye_lower,
                    right_iris=right_iris,
                    left_iris=left_iris,
                    body7=body7,
                )
                frame = display
            else:
                display = frame.copy()
                cv2.putText(display, 'PAUSED', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
            cv2.imshow(win, display if paused else frame)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('m'):
                mirror = not mirror
                calib.clear()
                print('Mirror toggled — re-Center')
            if key == ord(' '):
                paused = not paused
            if key == ord('c'):
                if last_pts28 is not None:
                    calib.capture(last_pts28, frame.shape, conf=0.0)
                    calib.flash_until = time.time() + 0.45
                    print('Centered instantly (28 label pts)')
                else:
                    print('Center failed: no face')
            if key == ord('r'):
                calib.clear()
                print('Center cleared')
    finally:
        capture.release()
        cv2.destroyAllWindows()
        uninit_com()
    return 0


def main() -> int:
    args = parse_args()
    ensure_com()
    osf = resolve_openseeface(args.osf_dir)

    if args.list_cameras:
        cams = list_cameras(osf)
        if not cams:
            print('No cameras found')
            return 1
        print('Available cameras (DirectShow, includes virtual):')
        for c in cams:
            print(f'  {c.label}')
        return 0

    if args.cli:
        return run_cli(args)

    try:
        _require_tk()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    LivePoserUI(args).run()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted')
        raise SystemExit(130)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
