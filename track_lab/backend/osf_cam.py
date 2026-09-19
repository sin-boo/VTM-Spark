"""Webcam + OpenSeeFace. Emits mix weights, never character pixels."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cameras import open_capture
from .presets import empty_weights
from .eye_bits import LID_MID_L, LID_MID_R, bits as eye_bits
from .mouth_bits import bits as mouth_bits
from .feel import feel
from .iris import hits_payload, merge_hits, osf_gaze_hits, track_camera
from .retarget import FACE_TRACK
from .visemes import reset_visemes, viseme_weights

ROOT = Path(__file__).resolve().parents[1]
OSF_DIR = ROOT / "osf"
MODELS_DIR = ROOT / "models"
CAM_W = 640
CAM_H = 480

if str(OSF_DIR) not in sys.path:
    sys.path.insert(0, str(OSF_DIR))


@dataclass
class OsfFrame:
    weights: dict[str, float] = field(default_factory=empty_weights)
    head: dict[str, float] = field(default_factory=lambda: {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
    blink: dict[str, float] = field(default_factory=lambda: {"l": 0.0, "r": 0.0})
    pose: dict[str, float] = field(
        default_factory=lambda: {
            "cx": 0.0,
            "cy": 0.0,
            "bx": 0.0,
            "by": 0.0,
            "scale": 1.0,
            "tz": 0.0,
            "tilt": 0.0,
            "ok": 0.0,
        }
    )
    faces: int = 0
    ms: float = 0.0
    camera_jpeg: bytes = b""
    error: str = ""
    pts_3d: np.ndarray | None = None
    mouth_2d: np.ndarray | None = None
    lms_xy: np.ndarray | None = None
    iris_cam: list[dict[str, object]] | None = None
    look: dict[str, float] | None = None
    source: str = ""
    brow: dict[str, float] = field(default_factory=dict)


def apply_mirror(frame: np.ndarray, mirror: bool) -> np.ndarray:
    """Horizontal flip of the preview only. Tracking always sees the raw
    frame; left/right is decided once in sides.py."""
    if not mirror:
        return frame
    return cv2.flip(frame, 1)


def _open_camera(index: int) -> cv2.VideoCapture | None:
    return open_capture(int(index), CAM_W, CAM_H)


def _make_tracker(width: int, height: int) -> object:
    if not (MODELS_DIR / "lm_model3_opt.onnx").is_file():
        raise FileNotFoundError(f"Face models missing in {MODELS_DIR}. Run track_lab/setup.ps1 or track_lab/start.bat")
    from tracker import Tracker  # noqa: E402

    return Tracker(
        width=width,
        height=height,
        model_type=3,
        detection_threshold=0.6,
        max_faces=1,
        max_threads=4,
        silent=True,
        model_dir=str(MODELS_DIR),
        no_gaze=False,
        use_retinaface=1,
        try_hard=False,
        feature_level=2,
        static_model=True,
    )


def _lms_xy(lms: np.ndarray) -> np.ndarray:
    if lms is None or len(lms) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.zeros((len(lms), 3), dtype=np.float32)
    out[:, 0] = lms[:, 1]
    out[:, 1] = lms[:, 0]
    if lms.shape[1] > 2:
        out[:, 2] = lms[:, 2]
    else:
        out[:, 2] = 1.0
    return out


def _stamp_id(
    vis: np.ndarray,
    px: int,
    py: int,
    text: str,
    color: tuple[int, int, int],
    font: float,
    pad: int = 3,
) -> None:
    """Index next to a landmark. Left-side points get the number on the left."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
    left = px < vis.shape[1] * 0.5
    tx = px - tw - pad if left else px + pad
    ty = py + th // 2
    cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font, (8, 10, 14), 1, cv2.LINE_AA)
    cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font, color, 1, cv2.LINE_AA)


def _draw_camera(
    frame: np.ndarray,
    face: object | None,
    pose: dict[str, float],
    iris: object = None,
) -> np.ndarray:
    vis = frame.copy()
    show_ids = feel.show_ids()
    if face is None or not (feel.show_face() or show_ids):
        return vis
    lms = getattr(face, "lms", None)
    if lms is None:
        return vis
    pts = _lms_xy(np.asarray(lms, dtype=np.float32))
    h, w = vis.shape[:2]
    scale = 2
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    h, w = vis.shape[:2]
    radius = 2
    font = 0.28
    ink = (236, 228, 208)
    for i, (x, y, conf) in enumerate(pts[:66]):
        if not show_ids and 48 <= i <= 65 and not mouth_bits.on(i):
            continue
        if conf < 0.12:
            continue
        px, py = int(round(x * scale)), int(round(y * scale))
        if px < 0 or py < 0 or px >= w or py >= h:
            continue
        used_mouth = 48 <= i <= 65 and mouth_bits.on(i)
        used_eye = 36 <= i <= 47 and eye_bits.on(i)
        used_face = i in FACE_TRACK and not (36 <= i <= 47)
        if used_mouth:
            color = (180, 80, 255)
        elif used_eye:
            color = (80, 230, 160)
        elif used_face:
            color = (80, 200, 255)
        else:
            color = (210, 220, 230)
        cv2.circle(vis, (px, py), radius + 1, (8, 10, 14), -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), radius, color, -1, cv2.LINE_AA)
        if show_ids or used_mouth or used_eye or used_face:
            _stamp_id(vis, px, py, str(i), ink, font, pad=radius + 2)
    _draw_lid_mids(vis, pts, scale, radius, font, show_ids=show_ids)
    _draw_iris(vis, iris, scale)
    return vis


def _draw_iris(vis: np.ndarray, iris: object, scale: int) -> None:
    if not isinstance(iris, list):
        return
    h, w = vis.shape[:2]
    marks = {str(row.get("side")): row for row in iris if isinstance(row, dict)}
    for side, color, tag in (("r", (0, 220, 255), "R"), ("l", (255, 180, 0), "L")):
        row = marks.get(side)
        if not row or not row.get("visible"):
            continue
        px = int(round(float(row.get("x") or 0.0) * scale))
        py = int(round(float(row.get("y") or 0.0) * scale))
        if px < 0 or py < 0 or px >= w or py >= h:
            continue
        cv2.circle(vis, (px, py), 5, (8, 10, 14), -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 4, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 7, color, 1, cv2.LINE_AA)
        _stamp_id(vis, px, py, tag, color, 0.32, pad=8)


def _draw_lid_mids(
    vis: np.ndarray,
    pts: np.ndarray,
    scale: int,
    radius: int,
    font: float,
    show_ids: bool = False,
) -> None:
    """Gold diamonds mark the artificial lid midpoints."""
    h, w = vis.shape[:2]
    marks = (
        (LID_MID_R, 37, 38),
        (LID_MID_L, 43, 44),
    )
    color = (40, 210, 255)
    for mid_id, a, b in marks:
        if (not show_ids and not eye_bits.on(mid_id)) or len(pts) <= max(a, b):
            continue
        mid = 0.5 * (pts[a, :2] + pts[b, :2])
        px, py = int(round(float(mid[0]) * scale)), int(round(float(mid[1]) * scale))
        if px < 0 or py < 0 or px >= w or py >= h:
            continue
        size = radius + 3
        diamond = np.array(
            ((px, py - size), (px + size, py), (px, py + size), (px - size, py)),
            dtype=np.int32,
        )
        cv2.fillConvexPoly(vis, diamond, (8, 10, 14), cv2.LINE_AA)
        cv2.fillConvexPoly(vis, diamond, color, cv2.LINE_AA)
        _stamp_id(vis, px, py, f"{mid_id}*", color, font, pad=size + 2)


def _encode_jpeg(bgr: np.ndarray) -> bytes:
    h, w = bgr.shape[:2]
    longest = float(max(h, w, 1))
    if longest > 1280.0:
        scale = 1280.0 / longest
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return b""
    return bytes(buf)


def _face_pose(face: object | None) -> dict[str, float]:
    out = {"cx": 0.0, "cy": 0.0, "bx": 0.0, "by": 0.0, "scale": 1.0, "tz": 0.0, "tilt": 0.0, "ok": 0.0}
    if face is None:
        return out
    lms = getattr(face, "lms", None)
    if lms is None:
        return out
    pts = _lms_xy(np.asarray(lms, dtype=np.float32))
    if len(pts) <= 45:
        return out
    nose = pts[30]
    if float(nose[2]) < 0.12:
        return out
    jaw = float(np.hypot(pts[16, 0] - pts[0, 0], pts[16, 1] - pts[0, 1])) if len(pts) > 16 else 0.0
    r_eye = 0.5 * (pts[36, :2] + pts[39, :2])
    l_eye = 0.5 * (pts[42, :2] + pts[45, :2])
    eye = float(np.hypot(l_eye[0] - r_eye[0], l_eye[1] - r_eye[1]))
    scale = max(jaw, eye)
    if scale < 8.0:
        return out
    mid = 0.5 * (l_eye + r_eye)
    out["cx"] = float(nose[0])
    out["cy"] = float(nose[1])
    # Eye midpoint is the walk/place. The nose slides when you look.
    out["bx"] = float(mid[0])
    out["by"] = float(mid[1])
    out["scale"] = scale
    # PnP distance does not shrink when you turn; the 2D box does.
    translation = getattr(face, "translation", None)
    if translation is not None:
        vals = np.asarray(translation, dtype=np.float32).reshape(-1)
        if vals.size >= 3 and np.isfinite(vals[2]) and abs(float(vals[2])) > 1e-3:
            out["tz"] = abs(float(vals[2]))
    out["tilt"] = float(np.degrees(np.arctan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0])))
    out["ok"] = 1.0
    return out


def _face_local_2d(
    face: object | None,
    pose: dict[str, float],
) -> np.ndarray | None:
    """Return image landmarks in a translation/scale/roll-neutral face frame.

    Used for the mouth only. Brows and eyes read OSF's pose-free 3D
    landmarks so a head turn is applied once, by the head rig.
    """
    if face is None or not pose.get("ok"):
        return None
    lms = getattr(face, "lms", None)
    if lms is None:
        return None
    pts = _lms_xy(np.asarray(lms, dtype=np.float32))
    if len(pts) < 66:
        return None
    scale = max(float(pose["scale"]), 1.0)
    angle = np.deg2rad(-float(pose.get("tilt", 0.0)))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    xy = pts[:, :2] - np.array([pose["cx"], pose["cy"]], dtype=np.float32)
    out = pts.copy()
    out[:, 0] = (xy[:, 0] * c - xy[:, 1] * s) / scale
    out[:, 1] = (xy[:, 0] * s + xy[:, 1] * c) / scale
    extra = np.zeros((2, out.shape[1]), dtype=out.dtype)
    extra[0, :2] = 0.5 * (out[37, :2] + out[38, :2])
    extra[0, 2] = min(float(out[37, 2]), float(out[38, 2]))
    extra[1, :2] = 0.5 * (out[43, :2] + out[44, :2])
    extra[1, 2] = min(float(out[43, 2]), float(out[44, 2]))
    return np.vstack((out, extra))


def _head(face: object | None) -> dict[str, float]:
    out = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    if face is None:
        return out
    euler = getattr(face, "euler", None)
    if euler is None:
        return out
    vals = np.asarray(euler, dtype=np.float32).reshape(-1)
    if vals.size >= 3:
        out["pitch"] = round(float(vals[0]), 2)
        out["yaw"] = round(float(vals[1]), 2)
        out["roll"] = round(float(vals[2]), 2)
    return out


def _blink(face: object | None) -> dict[str, float]:
    out = {"l": 0.0, "r": 0.0}
    if face is None:
        return out
    blink = getattr(face, "eye_blink", None)
    if blink is None:
        return out
    vals = np.asarray(blink, dtype=np.float32).reshape(-1)
    if vals.size >= 2:
        # OSF stores 1 = open. Slider is blink amount.
        # Canonical image sides: eye_blink[0] is OSF "eye_r" = landmarks
        # 36-41 = the image-left eye, so it is "l" here.
        out["l"] = round(float(np.clip(1.0 - vals[0], 0.0, 1.0)), 3)
        out["r"] = round(float(np.clip(1.0 - vals[1], 0.0, 1.0)), 3)
    return out


def _smooth(prev: dict[str, float], nxt: dict[str, float], alpha: float = 0.38) -> dict[str, float]:
    out = dict(nxt)
    for key, value in nxt.items():
        old = float(prev.get(key, value))
        out[key] = old + alpha * (float(value) - old)
    return out


class _LatestFrame:
    """Overwrite mailbox so a slow tracker never replays buffered camera frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: np.ndarray | None = None
        self._has = threading.Event()

    def put(self, item: np.ndarray) -> None:
        with self._lock:
            self._item = item
            self._has.set()

    def take(self, timeout: float | None = None) -> np.ndarray | None:
        if not self._has.wait(timeout):
            return None
        with self._lock:
            item = self._item
            self._item = None
            self._has.clear()
        return item


class OsfCam:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._grab_thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._tracker: object | None = None
        self.camera_index = 0
        # Flip the preview JPEG like a selfie. Landmarks stay raw.
        self.preview_flip = False
        self.latest = OsfFrame()

    @property
    def running(self) -> bool:
        return self._running

    def start(
        self,
        on_frame: Callable[[OsfFrame], None] | None = None,
        index: int = 0,
    ) -> None:
        if self._running:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Camera tracker is still stopping")
        cap = _open_camera(index)
        if cap is None:
            raise RuntimeError(
                f"Could not open camera {index}. Close the app using it, or pick another."
            )
        self.camera_index = int(index)
        self._cap = cap
        self._running = True
        self.latest = OsfFrame()
        reset_visemes()
        self._thread = threading.Thread(
            target=self._loop,
            args=(on_frame,),
            name="osf-cam",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        grab = self._grab_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        if grab is not None and grab.is_alive():
            grab.join(timeout=0.2)
        self._release()
        if grab is not None and grab.is_alive():
            grab.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            thread.join(timeout=8.0)
        if grab is None or not grab.is_alive():
            self._grab_thread = None
        if thread is None or not thread.is_alive():
            self._thread = None

    def _release(self) -> None:
        with self._lock:
            cap = self._cap
            self._cap = None
            tracker = self._tracker
            self._tracker = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if tracker is not None:
            closer = getattr(tracker, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    def _grab_loop(self, cap: cv2.VideoCapture, pending: _LatestFrame) -> None:
        """Keep eating camera frames so DirectShow never queues a delay."""
        try:
            while self._running:
                ok, raw = cap.read()
                if not ok or raw is None:
                    time.sleep(0.02)
                    continue
                # OpenCV reuses the capture buffer; copy before the next read.
                pending.put(raw.copy())
        finally:
            self._grab_thread = None

    def _loop(self, on_frame: Callable[[OsfFrame], None] | None) -> None:
        cap = self._cap
        smoothed = empty_weights()
        pending = _LatestFrame()
        try:
            if cap is None:
                raise RuntimeError("Camera closed")
            ok, probe = cap.read()
            if not ok or probe is None:
                raise RuntimeError("Camera produced no frames")
            frame0 = cv2.resize(probe, (CAM_W, CAM_H))
            self._tracker = _make_tracker(frame0.shape[1], frame0.shape[0])
            grab = threading.Thread(
                target=self._grab_loop,
                args=(cap, pending),
                name="osf-grab",
                daemon=True,
            )
            self._grab_thread = grab
            grab.start()
            while self._running:
                raw = pending.take(timeout=0.05)
                if raw is None:
                    continue
                frame = cv2.resize(raw, (CAM_W, CAM_H))
                started = time.perf_counter()
                faces = self._tracker.predict(frame)
                face = faces[0] if faces else None
                pose = _face_pose(face)
                head = _head(face)
                weights = viseme_weights(face, pose=pose)
                smoothed = _smooth(smoothed, weights, feel.alpha())
                pts_3d = None
                if face is not None:
                    raw3 = getattr(face, "pts_3d", None)
                    if raw3 is not None:
                        pts_3d = np.asarray(raw3, dtype=np.float32).copy()
                mouth_2d = _face_local_2d(face, pose)
                lms_xy = None
                if face is not None:
                    raw_lms = getattr(face, "lms", None)
                    if raw_lms is not None:
                        lms_xy = _lms_xy(np.asarray(raw_lms, dtype=np.float32))
                iris_cam: list[dict[str, object]] = []
                if lms_xy is not None and len(lms_xy) >= 48:
                    yolo_right, yolo_left = track_camera(frame, lms_xy)
                    gaze_right, gaze_left = osf_gaze_hits(lms_xy)
                    cam_right, cam_left, _method = merge_hits(
                        (yolo_right, yolo_left), (gaze_right, gaze_left)
                    )
                    iris_cam = hits_payload(cam_right, cam_left)
                snap = OsfFrame(
                    weights=smoothed,
                    head=head,
                    blink=_blink(face),
                    pose=pose,
                    faces=1 if face is not None else 0,
                    ms=(time.perf_counter() - started) * 1000.0,
                    camera_jpeg=_encode_jpeg(
                        apply_mirror(
                            _draw_camera(frame, face, pose, iris_cam), self.preview_flip
                        )
                    ),
                    pts_3d=pts_3d,
                    mouth_2d=mouth_2d,
                    lms_xy=lms_xy,
                    iris_cam=iris_cam,
                    source="osf",
                )
                self.latest = snap
                if on_frame is not None:
                    on_frame(snap)
        except Exception as exc:
            snap = OsfFrame(error=str(exc))
            self.latest = snap
            if on_frame is not None:
                on_frame(snap)
            self._running = False
        finally:
            self._release()
            self._thread = None
