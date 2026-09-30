"""Webcam + OpenSeeFace. Emits mix weights, never character pixels."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import debug_log
from .cameras import _ensure_com, open_capture
from .paths import LAB_ROOT, OSF_MODELS
from .presets import empty_weights
from .eye_bits import LID_MID_L, LID_MID_R, bits as eye_bits
from .mouth_bits import bits as mouth_bits
from .feel import feel
from .iris import hits_payload, merge_hits, osf_gaze_hits, track_camera
from .retarget import FACE_TRACK
from .visemes import reset_visemes, viseme_weights

ROOT = LAB_ROOT
OSF_DIR = ROOT / "osf"
MODELS_DIR = OSF_MODELS
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


# Seconds OpenSeeFace learns each face's feature scale (the blink's open /
# shut range: a running median, plus a min and max that creep toward it)
# after it first sees the face; then the scale holds, through face losses
# too. Left at the Tracker's default 0 it never stopped (and began again on
# every face loss), so the same lid read more or less shut as the session
# wore on. 30, not less: at a screen people can go ~10 s without a blink,
# and a window with none leaves the lid an on / off switch for the session.
_FEATURE_LEARN_S = 30


def _make_tracker(width: int, height: int) -> object:
    if not (MODELS_DIR / "lm_model3_opt.onnx").is_file():
        raise FileNotFoundError(
            f"Face models missing in {MODELS_DIR}. Run install.bat at the repo root."
        )
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
        max_feature_updates=_FEATURE_LEARN_S,
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


def _drain_queued(cap: object, frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop frames already waiting in DirectShow so the preview is not a backlog."""
    reader = getattr(cap, "_reader", None)
    if reader is None or not hasattr(reader, "timeout"):
        return frame, 0
    old = reader.timeout
    extra = 0
    try:
        # 0 often means "wait forever" in this capture DLL.
        reader.timeout = 1
        for _ in range(4):
            started = time.perf_counter()
            try:
                ok, nxt = cap.read()  # type: ignore[attr-defined]
            except Exception:
                break
            waited = time.perf_counter() - started
            if not ok or nxt is None:
                break
            frame = nxt
            extra += 1
            if waited > 0.012:
                break
    finally:
        try:
            reader.timeout = old
        except Exception:
            pass
    return frame, extra


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


# The solved head roll is only trusted near the eye line. Half-turn flips
# (roll ~180 on an upright face) are undone in _pnp_head; what is still this
# far off (roll 123 while the eyes read 3) is a bad solve: locked as rest, it
# tipped the hair to the roll limit for the whole session.
# 60, not tighter: a real turn + nod slants the eye line off the true roll by
# atan(sin(pitch) * tan(turn)) (~41 deg at 60 turn / 30 nod).
_PNP_TILT_TRUST = 60.0


def _wrap_deg(angle: float) -> float:
    """Angle difference in [-180, 180): 179 -> -179 is 2 deg, not 358."""
    return (float(angle) + 180.0) % 360.0 - 180.0


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
    # The eye line still frames the mouth: it slants with the mouth on a turn.
    eyes = float(np.degrees(np.arctan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0])))
    turn = _pnp_head(face)
    head_ok = turn is None or abs(_wrap_deg(turn["roll"] - eyes)) <= _PNP_TILT_TRUST
    out["tilt"] = turn["roll"] if turn is not None and head_ok else eyes
    out["tilt_eyes"] = eyes
    # 0 = this frame's solved head is a flipped / bad solve: do not lock rest
    # on it, and hold the last good yaw / pitch (see FaceRig._sync).
    out["head_ok"] = 1.0 if head_ok else 0.0
    out["ok"] = 1.0
    return out


# OSF runs PnP on (y, x) landmarks with a face model that looks down +z, so its
# rotation is -SWAP @ H, where H is the head turn in the image frame (x right,
# y down, z away) and H = identity when facing the camera.
_SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# Half a turn round the view axis: Rz(180) @ H keeps H's pitch and turn and
# moves its roll by 180 (Rz(r) Rx(p) Ry(t) -> Rz(r + 180) Rx(p) Ry(t)).
_HALF_TURN = np.diag([-1.0, -1.0, 1.0])


def _pnp_head(face: object | None) -> dict[str, float] | None:
    """Turn, nod, and tilt from OSF's solved rotation, in the rig's signs.

    OSF's own euler is taken on that swapped frame and sits near gimbal lock
    (pitch ~180, roll ~90 when facing the camera). A plain turn there leaked
    into roll by +-25 deg and into pitch, so the drawn head tipped over
    instead of turning. H = Rz(roll) @ Rx(pitch) @ Ry(turn) keeps them apart:
    a turn spins around the head's own up axis and leaves tilt alone.
    """
    # Not gated on face.success: the last solved rotation beats switching to
    # OSF's euler, whose zero is somewhere else entirely.
    if face is None:
        return None
    rvec = getattr(face, "rotation", None)
    if rvec is None:
        return None
    vals = np.asarray(rvec, dtype=np.float64).reshape(-1)
    if vals.size != 3 or not np.all(np.isfinite(vals)):
        return None
    rmat, _ = cv2.Rodrigues(vals)
    h = -_SWAP @ rmat
    if h[1, 1] < 0.0:
        # Upside down (|roll| > 90): the solve settled half a turn round the
        # view axis. On a real webcam this is the usual state, not a rare
        # glitch (roll ~176 on an upright face), and refusing it froze the
        # turn for good. Turn and nod are right as they are; only roll is off.
        h = _HALF_TURN @ h
    pitch = float(np.degrees(np.arcsin(np.clip(h[2, 1], -1.0, 1.0))))
    turn = float(np.degrees(np.arctan2(-h[2, 0], h[2, 2])))
    roll = float(np.degrees(np.arctan2(-h[0, 1], h[1, 1])))
    # Rig signs: pitch+ = look down, roll+ = clockwise on screen (same as the
    # eye line), yaw = -turn (what OSF's yaw read while it was near zero).
    return {"pitch": pitch, "yaw": -turn, "roll": roll}


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
    angle = np.deg2rad(-float(pose.get("tilt_eyes", pose.get("tilt", 0.0))))
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
    solved = _pnp_head(face)
    if solved is not None:
        return {key: round(value, 2) for key, value in solved.items()}
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
        self.seq = 0
        self.dropped = 0
        self.age_ms = 0.0
        self.read_ms = 0.0
        self._put_t = 0.0

    def put(self, item: np.ndarray, read_ms: float = 0.0) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self.seq += 1
            self._put_t = now
            self.read_ms = float(read_ms)
            self._has.set()

    def take(self, timeout: float | None = None) -> np.ndarray | None:
        if not self._has.wait(timeout):
            return None
        now = time.perf_counter()
        with self._lock:
            item = self._item
            self.age_ms = (now - self._put_t) * 1000.0 if self._put_t else 0.0
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
        self._preview_jpeg = b""
        self.latest = OsfFrame()

    @property
    def running(self) -> bool:
        return self._running

    def start(
        self,
        on_frame: Callable[[OsfFrame], None] | None = None,
        index: int = 0,
    ) -> None:
        if self._running and self._thread is not None and self._thread.is_alive():
            return
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        self._thread = None
        self._grab_thread = None
        # Model load can take seconds. Do it before the device is capturing,
        # or DirectShow keeps that unread backlog for the rest of the session.
        if self._tracker is None:
            self._tracker = _make_tracker(CAM_W, CAM_H)
        cap = _open_camera(index)
        if cap is None:
            self._drop_tracker()
            raise RuntimeError(
                f"Could not open camera {index}. "
                "Check that it is connected and streaming, or pick another."
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
        self._poke_reader()
        thread = self._thread
        grab = self._grab_thread
        if grab is not None and grab.is_alive():
            grab.join(timeout=0.4)
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.4)
        self._release()
        if grab is not None and grab.is_alive():
            grab.join(timeout=1.0)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        # A native read can outlive release. Forget the handle so the next
        # Track is not refused with "still stopping".
        if self._grab_thread is grab:
            self._grab_thread = None
        if self._thread is thread:
            self._thread = None

    def _poke_reader(self) -> None:
        """Ask an in-flight DirectShow read to return so stop can finish."""
        cap = self._cap
        reader = getattr(cap, "_reader", None) if cap is not None else None
        if reader is None:
            return
        try:
            reader.timeout = 1
        except Exception:
            pass

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

    def _drop_tracker(self) -> None:
        with self._lock:
            tracker = self._tracker
            self._tracker = None
        if tracker is None:
            return
        closer = getattr(tracker, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    def _publish_preview(self, raw: np.ndarray) -> None:
        """Newest grab, without landmark drawing or the 2× upscale."""
        try:
            if int(raw.shape[1]) != CAM_W or int(raw.shape[0]) != CAM_H:
                shown = cv2.resize(raw, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
            else:
                shown = raw
            jpeg = _encode_jpeg(apply_mirror(shown, self.preview_flip))
        except Exception:
            return
        with self._lock:
            self._preview_jpeg = jpeg

    def _latest_preview(self) -> bytes:
        with self._lock:
            return self._preview_jpeg

    def _grab_loop(self, cap: cv2.VideoCapture, pending: _LatestFrame) -> None:
        """Keep eating camera frames so DirectShow never queues a delay."""
        me = threading.current_thread()
        _ensure_com()
        reader = getattr(cap, "_reader", None)
        if reader is not None:
            try:
                reader.timeout = 150
            except Exception:
                pass
        last_log = 0.0
        fails = 0
        try:
            while self._running:
                started = time.perf_counter()
                try:
                    ok, raw = cap.read()
                except Exception as exc:
                    fails += 1
                    # #region agent log
                    if debug_log.ENABLED and time.perf_counter() - last_log >= 0.5:
                        last_log = time.perf_counter()
                        debug_log.log(
                            "A",
                            "osf_cam.py:_grab_loop",
                            "read raised",
                            {"error": str(exc), "fails": fails},
                        )
                    # #endregion
                    time.sleep(0.02)
                    continue
                read_ms = (time.perf_counter() - started) * 1000.0
                queued = 0
                if ok and raw is not None:
                    raw, queued = _drain_queued(cap, raw)
                if not ok or raw is None:
                    fails += 1
                    # #region agent log
                    if debug_log.ENABLED and time.perf_counter() - last_log >= 0.5:
                        last_log = time.perf_counter()
                        debug_log.log(
                            "A",
                            "osf_cam.py:_grab_loop",
                            "read empty",
                            {"fails": fails, "read_ms": round(read_ms, 1)},
                        )
                    # #endregion
                    time.sleep(0.02)
                    continue
                # OpenCV reuses the capture buffer; copy before the next read.
                fresh = raw.copy()
                pending.put(fresh, read_ms)
                self._publish_preview(fresh)
                now = time.perf_counter()
                if debug_log.ENABLED and now - last_log >= 0.5:
                    last_log = now
                    shape = [0, 0]
                    std = 0.0
                    try:
                        shape = [int(raw.shape[1]), int(raw.shape[0])]
                        step = max(1, min(int(raw.shape[0]), int(raw.shape[1])) // 16, 1)
                        std = round(float(np.std(raw[::step, ::step])), 2)
                    except Exception:
                        pass
                    # #region agent log
                    debug_log.log(
                        "A",
                        "osf_cam.py:_grab_loop",
                        "grab",
                        {
                            "read_ms": round(read_ms, 1),
                            "shape": shape,
                            "std": std,
                            "seq": pending.seq,
                            "dropped": pending.dropped,
                            "queued": queued,
                            "fails": fails,
                        },
                    )
                    # #endregion
        except Exception as exc:
            # #region agent log
            debug_log.log("A", "osf_cam.py:_grab_loop", "grab died", {"error": str(exc)})
            # #endregion
        finally:
            if self._grab_thread is me:
                self._grab_thread = None

    def _loop(self, on_frame: Callable[[OsfFrame], None] | None) -> None:
        me = threading.current_thread()
        _ensure_com()
        cap = self._cap
        smoothed = empty_weights()
        pending = _LatestFrame()
        try:
            if cap is None:
                raise RuntimeError("Camera closed")
            if self._tracker is None:
                self._tracker = _make_tracker(CAM_W, CAM_H)
            grab = threading.Thread(
                target=self._grab_loop,
                args=(cap, pending),
                name="osf-grab",
                daemon=True,
            )
            self._grab_thread = grab
            grab.start()
            last_log = 0.0
            stalls = 0
            last_seq = 0
            saw_frame = False
            while self._running:
                raw = pending.take(timeout=0.05)
                if raw is None:
                    stalls += 1
                    if not saw_frame and stalls >= 30:
                        raise RuntimeError("Camera produced no frames")
                    # #region agent log
                    if debug_log.ENABLED and time.perf_counter() - last_log >= 0.5:
                        last_log = time.perf_counter()
                        debug_log.log(
                            "A",
                            "osf_cam.py:_loop",
                            "tracker waiting",
                            {"stalls": stalls, "grab_alive": bool(grab.is_alive())},
                        )
                    # #endregion
                    continue
                saw_frame = True
                raw_wh = [int(raw.shape[1]), int(raw.shape[0])]
                frame = cv2.resize(raw, (CAM_W, CAM_H))
                started = time.perf_counter()
                faces = self._tracker.predict(frame)
                predict_ms = (time.perf_counter() - started) * 1000.0
                # #region agent log
                now_log = time.perf_counter()
                if debug_log.ENABLED and now_log - last_log >= 0.5:
                    last_log = now_log
                    skipped = max(0, pending.seq - last_seq - 1)
                    last_seq = pending.seq
                    debug_log.log(
                        "B",
                        "osf_cam.py:_loop",
                        "track",
                        {
                            "predict_ms": round(predict_ms, 1),
                            "age_ms": round(pending.age_ms, 1),
                            "read_ms": round(pending.read_ms, 1),
                            "faces": len(faces) if faces else 0,
                            "raw_wh": raw_wh,
                            "squashed": raw_wh != [CAM_W, CAM_H],
                            "skipped": skipped,
                            "dropped": pending.dropped,
                        },
                    )
                # #endregion
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
                    camera_jpeg=self._latest_preview(),
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
            if self._cap is cap:
                self._release()
            if self._thread is me:
                self._thread = None
