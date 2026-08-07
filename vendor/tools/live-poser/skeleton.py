"""Dedicated human body tracker → KEYPOINT_SCHEMA slots 30..36.

Primary: MediaPipe Pose Landmarker (lite) — real human skeleton tracker.
Fallback: stock yolov8n-pose (COCO-17) if MediaPipe weights missing.
Last resort: synthetic torso from face pose (OpenSeeFace has no body).

Slots
-----
30 nose  31 neck
32 right_shoulder  33 right_elbow
34 left_shoulder   35 left_elbow
36 chest (derived)
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from label_schema import LEFT_EYE, RIGHT_EYE, face_basis_from_label28, geom_head_angles

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'
_PACKAGE_TRACKERS = ROOT.parent.parent.parent / 'models' / 'trackers'

BODY_NOSE = 30
NECK = 31
RIGHT_SHOULDER = 32
RIGHT_ELBOW = 33
LEFT_SHOULDER = 34
LEFT_ELBOW = 35
CHEST = 36

KPT_NAMES = [
    'nose',
    'neck',
    'right_shoulder',
    'right_elbow',
    'left_shoulder',
    'left_elbow',
]

SKELETON_BONES = [
    (BODY_NOSE, NECK),
    (NECK, RIGHT_SHOULDER),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (NECK, LEFT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (RIGHT_SHOULDER, LEFT_SHOULDER),
    (NECK, CHEST),
]

# MediaPipe BlazePose indices (person-relative)
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14

# COCO-17 YOLO-pose indices
COCO_NOSE = 0
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8

MEDIAPIPE_LITE_URL = (
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
)
MEDIAPIPE_LITE_URL_FALLBACK = (
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
)

YOLO_N_CANDIDATES = [
    _PACKAGE_TRACKERS / 'yolov8n-pose.pt',
    MODELS_DIR / 'yolov8n-pose.pt',
    ROOT.parent / 'pose-traker' / 'iris-model' / 'yolov8n-pose.pt',
]

VIS_THR = 0.45
EDGE_SLACK = 24.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _clamp_xy(x: float, y: float, w: int, h: int) -> tuple[float, float, bool]:
    near = (-EDGE_SLACK <= x <= w - 1 + EDGE_SLACK) and (-EDGE_SLACK <= y <= h - 1 + EDGE_SLACK)
    if not near:
        return x, y, False
    return (
        min(max(float(x), 1.0), float(w - 1)),
        min(max(float(y), 1.0), float(h - 1)),
        True,
    )


def _derive_neck_chest(
    r_sh: dict,
    l_sh: dict,
) -> tuple[dict, dict]:
    if not r_sh.get('visible') or not l_sh.get('visible'):
        z = {'x': 0.0, 'y': 0.0, 'score': 0.0, 'visible': False}
        return z, z
    sw = abs(float(l_sh['x']) - float(r_sh['x'])) + 1e-6
    mx = 0.5 * (float(l_sh['x']) + float(r_sh['x']))
    my = 0.5 * (float(l_sh['y']) + float(r_sh['y']))
    score = min(float(l_sh['score']), float(r_sh['score']))
    neck = {'x': mx, 'y': my - sw * 0.08, 'score': score, 'visible': True}
    chest = {'x': mx, 'y': my + sw * 0.22, 'score': score, 'visible': True}
    return neck, chest


def joints_to_body7(joints: dict[str, dict]) -> np.ndarray:
    out = np.zeros((7, 4), dtype=np.float32)
    for i, name in enumerate(KPT_NAMES + ['chest']):
        j = joints.get(name) or {}
        vis = bool(j.get('visible'))
        out[i, 0] = float(j.get('x', 0.0))
        out[i, 1] = float(j.get('y', 0.0))
        out[i, 2] = float(j.get('score', 0.0))
        out[i, 3] = 1.0 if vis else 0.0
    return out


def ensure_mediapipe_lite(path: Path | None = None) -> Path | None:
    """Ensure pose_landmarker_lite.task exists (download once if needed)."""
    if path is None:
        for candidate in (
            _PACKAGE_TRACKERS / 'pose_landmarker_lite.task',
            MODELS_DIR / 'pose_landmarker_lite.task',
        ):
            if candidate.is_file() and candidate.stat().st_size > 100_000:
                return candidate
        path = MODELS_DIR / 'pose_landmarker_lite.task'
    else:
        path = Path(path)
        if path.is_file() and path.stat().st_size > 100_000:
            return path
    if path.is_file() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in (MEDIAPIPE_LITE_URL, MEDIAPIPE_LITE_URL_FALLBACK):
        try:
            print(f'[skeleton] Downloading MediaPipe lite -> {path.name} ...')
            urllib.request.urlretrieve(url, path)
            if path.is_file() and path.stat().st_size > 1000:
                return path
        except Exception as exc:
            print(f'[skeleton] download failed ({url}): {exc}')
    return None


def resolve_yolo_n_weights(explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    for c in YOLO_N_CANDIDATES:
        if c.is_file():
            return c
    return None


# Back-compat aliases used by live_poser
def resolve_skeleton_weights(explicit: Path | str | None = None) -> Path | None:
    """Prefer MediaPipe lite task, else yolov8n-pose."""
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    mp = ensure_mediapipe_lite()
    if mp is not None:
        return mp
    return resolve_yolo_n_weights()


class MediaPipePoseTracker:
    """Dedicated MediaPipe Pose Landmarker (lite) → body7."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        path = ensure_mediapipe_lite(Path(model_path) if model_path else None)
        if path is None:
            raise FileNotFoundError('pose_landmarker_lite.task not available')

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._mp_format = mp.ImageFormat.SRGB
        self.weights = path
        self.method = 'mediapipe_pose_lite'
        self._ts_ms = 0

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def predict_body7(
        self,
        image_bgr: np.ndarray,
        *,
        vis_thr: float | None = None,
        box_conf: float | None = None,  # unused; API parity with YOLO
    ) -> np.ndarray | None:
        thr = float(VIS_THR if vis_thr is None else vis_thr)
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # VIDEO mode requires strictly increasing timestamps
        self._ts_ms = max(self._ts_ms + 33, int(time.time() * 1000))
        mp_image = self._mp_image_cls(image_format=self._mp_format, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]

        def _pt(idx: int) -> dict:
            lm = lms[idx]
            x = float(lm.x) * w
            y = float(lm.y) * h
            score = float(getattr(lm, 'visibility', 0.0) or getattr(lm, 'presence', 0.0) or 0.0)
            x, y, in_frame = _clamp_xy(x, y, w, h)
            return {
                'x': x,
                'y': y,
                'score': score,
                'visible': bool(score >= thr and in_frame),
            }

        nose = _pt(MP_NOSE)
        r_sh = _pt(MP_RIGHT_SHOULDER)
        l_sh = _pt(MP_LEFT_SHOULDER)
        r_el = _pt(MP_RIGHT_ELBOW)
        l_el = _pt(MP_LEFT_ELBOW)
        neck, chest = _derive_neck_chest(r_sh, l_sh)
        joints = {
            'nose': nose,
            'neck': neck,
            'right_shoulder': r_sh,
            'right_elbow': r_el,
            'left_shoulder': l_sh,
            'left_elbow': l_el,
            'chest': chest,
        }
        body7 = joints_to_body7(joints)
        if float(np.sum(body7[:, 3])) < 3:
            return None
        return body7


class YoloCocoPoseTracker:
    """Stock COCO yolov8n-pose → body7 (fallback human tracker)."""

    def __init__(
        self,
        weights: Path | str | None = None,
        *,
        device: str | None = None,
        imgsz: int = 416,
        conf: float = 0.25,
    ):
        path = resolve_yolo_n_weights(weights)
        if path is None:
            raise FileNotFoundError('yolov8n-pose.pt not found')
        from ultralytics import YOLO

        self.weights = path
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.model = YOLO(str(path))
        self.method = f'yolov8n_pose_{self.imgsz}'

    def close(self) -> None:
        pass

    def predict_body7(
        self,
        image_bgr: np.ndarray,
        *,
        vis_thr: float | None = None,
        box_conf: float | None = None,
    ) -> np.ndarray | None:
        thr = float(VIS_THR if vis_thr is None else vis_thr)
        conf = float(self.conf if box_conf is None else box_conf)
        h, w = image_bgr.shape[:2]
        try:
            results = self.model.predict(
                source=image_bgr,
                conf=conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
        except Exception:
            return None
        if not results or results[0].keypoints is None or results[0].keypoints.data is None:
            return None
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None
        confs = r0.boxes.conf.cpu().numpy()
        best_i = int(np.argmax(confs))
        kpts = r0.keypoints.data.cpu().numpy()
        if best_i >= len(kpts):
            return None
        kp = kpts[best_i]

        def _pt(idx: int) -> dict:
            if idx >= len(kp):
                return {'x': 0.0, 'y': 0.0, 'score': 0.0, 'visible': False}
            px, py, pv = float(kp[idx][0]), float(kp[idx][1]), float(kp[idx][2])
            score = float(pv) if pv <= 1.0 else float(min(1.0, pv / 2.0))
            px, py, in_frame = _clamp_xy(px, py, w, h)
            return {
                'x': px,
                'y': py,
                'score': score,
                'visible': bool(score >= thr and in_frame),
            }

        nose = _pt(COCO_NOSE)
        r_sh = _pt(COCO_RIGHT_SHOULDER)
        l_sh = _pt(COCO_LEFT_SHOULDER)
        r_el = _pt(COCO_RIGHT_ELBOW)
        l_el = _pt(COCO_LEFT_ELBOW)
        neck, chest = _derive_neck_chest(r_sh, l_sh)
        joints = {
            'nose': nose,
            'neck': neck,
            'right_shoulder': r_sh,
            'right_elbow': r_el,
            'left_shoulder': l_sh,
            'left_elbow': l_el,
            'chest': chest,
        }
        body7 = joints_to_body7(joints)
        if float(np.sum(body7[:, 3])) < 3:
            return None
        return body7


class SkeletonLiteTracker:
    """Facade: MediaPipe human pose first, else YOLO-n COCO, else raise."""

    def __init__(
        self,
        weights: Path | str | None = None,
        *,
        device: str | None = None,
        prefer_onnx: bool = True,  # unused; kept for live_poser argv compat
        imgsz: int = 416,
        conf: float = 0.25,
        export_onnx: bool = True,  # unused; kept for live_poser argv compat
        backend: str | None = None,
    ):
        self.device = device
        self._impl = None
        self.method = 'none'
        self.weights = None

        want = (backend or 'auto').lower()
        errors: list[str] = []

        def _try_mp():
            path = None
            if weights is not None and str(weights).endswith('.task'):
                path = Path(weights)
            self._impl = MediaPipePoseTracker(path)
            self.method = self._impl.method
            self.weights = self._impl.weights

        def _try_yolo():
            w = weights if (weights is not None and str(weights).endswith('.pt')) else None
            self._impl = YoloCocoPoseTracker(w, device=device, imgsz=imgsz, conf=conf)
            self.method = self._impl.method
            self.weights = self._impl.weights

        if want in ('mediapipe', 'mp', 'auto'):
            try:
                _try_mp()
                return
            except Exception as exc:
                errors.append(f'mediapipe: {exc}')
                if want != 'auto':
                    raise
        if want in ('yolo', 'yolov8n', 'coco', 'auto'):
            try:
                _try_yolo()
                return
            except Exception as exc:
                errors.append(f'yolo: {exc}')
                if want != 'auto':
                    raise
        raise RuntimeError('No human pose tracker available: ' + '; '.join(errors))

    def predict_body7(
        self,
        image_bgr: np.ndarray,
        *,
        vis_thr: float | None = None,
        box_conf: float | None = None,
    ) -> np.ndarray | None:
        if self._impl is None:
            return None
        try:
            return self._impl.predict_body7(image_bgr, vis_thr=vis_thr, box_conf=box_conf)
        except TypeError:
            return self._impl.predict_body7(image_bgr, vis_thr=vis_thr)

    def close(self) -> None:
        if self._impl is not None and hasattr(self._impl, 'close'):
            self._impl.close()


def synth_upper_body(
    pts28: np.ndarray,
    *,
    pitch: float | None = None,
    yaw: float | None = None,
    roll: float | None = None,
    score: float = 0.85,
) -> np.ndarray:
    """Synthetic (7,4) body from face pose — last-resort fallback."""
    out = np.zeros((7, 4), dtype=np.float32)
    if pts28 is None or len(pts28) < 28:
        return out

    basis = face_basis_from_label28(pts28)
    if basis is None:
        return out
    nose_xy, face_x, face_y = basis
    down = -face_y
    person_right = -face_x

    gp, gy, gr = geom_head_angles(pts28)
    pitch = float(gp if pitch is None else pitch)
    yaw = float(gy if yaw is None else yaw)
    roll = float(gr if roll is None else roll)

    left_eye = pts28[list(LEFT_EYE), 0:2].astype(np.float64).mean(axis=0)
    right_eye = pts28[list(RIGHT_EYE), 0:2].astype(np.float64).mean(axis=0)
    chin = pts28[2, 0:2].astype(np.float64)
    eye_dist = float(np.linalg.norm(left_eye - right_eye)) + 1e-6
    face_h = float(np.linalg.norm(chin - nose_xy)) + eye_dist * 0.6

    yaw_n = _clamp(yaw / 40.0, -1.2, 1.2)
    pitch_n = _clamp(pitch / 30.0, -1.0, 1.0)
    roll_rad = np.radians(roll)

    def rot2(v: np.ndarray, ang: float) -> np.ndarray:
        c, s = np.cos(ang), np.sin(ang)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)

    pr = rot2(person_right, roll_rad)
    dn = rot2(down, roll_rad)
    look = person_right * yaw_n
    neck = chin + dn * (face_h * 0.22) + look * (eye_dist * 0.25)
    neck = neck + dn * (pitch_n * face_h * 0.06)
    body_nose = nose_xy.copy()
    chest = neck + dn * (face_h * 0.72) + look * (eye_dist * 0.12)

    yaw_twist = np.radians(yaw * 0.55)
    shoulder_half = eye_dist * 1.05
    r_off = rot2(pr * shoulder_half + dn * (face_h * 0.08), yaw_twist)
    l_off = rot2(-pr * shoulder_half + dn * (face_h * 0.08), yaw_twist)
    r_scale = 1.0 - 0.22 * max(0.0, -yaw_n)
    l_scale = 1.0 - 0.22 * max(0.0, yaw_n)
    r_shoulder = neck + r_off * r_scale
    l_shoulder = neck + l_off * l_scale
    arm = face_h * 0.7
    r_elbow = r_shoulder + dn * arm + pr * (eye_dist * 0.1) + look * (eye_dist * 0.18)
    l_elbow = l_shoulder + dn * arm - pr * (eye_dist * 0.1) + look * (eye_dist * 0.18)

    joints = [
        (BODY_NOSE, body_nose),
        (NECK, neck),
        (RIGHT_SHOULDER, r_shoulder),
        (RIGHT_ELBOW, r_elbow),
        (LEFT_SHOULDER, l_shoulder),
        (LEFT_ELBOW, l_elbow),
        (CHEST, chest),
    ]
    for slot, xy in joints:
        i = slot - BODY_NOSE
        out[i, 0] = float(xy[0])
        out[i, 1] = float(xy[1])
        out[i, 2] = float(score)
        out[i, 3] = 1.0
    return out


def resolve_body7(
    image_bgr: np.ndarray,
    pts28: np.ndarray | None,
    tracker: SkeletonLiteTracker | None,
    *,
    rel_pitch: float = 0.0,
    rel_yaw: float = 0.0,
    rel_roll: float = 0.0,
    allow_synth_fallback: bool = True,
    vis_thr: float | None = None,
    box_conf: float | None = None,
) -> tuple[np.ndarray | None, str]:
    """Prefer dedicated human tracker; optionally fall back to face-driven synth."""
    if tracker is not None:
        body = tracker.predict_body7(image_bgr, vis_thr=vis_thr, box_conf=box_conf)
        if body is not None:
            return body, tracker.method
    if allow_synth_fallback and pts28 is not None:
        return (
            synth_upper_body(pts28, pitch=rel_pitch, yaw=rel_yaw, roll=rel_roll),
            'synthetic_from_face',
        )
    return None, 'none'


def flip_body7_x(body7: np.ndarray, width: int) -> np.ndarray:
    if body7 is None or len(body7) < 7:
        return body7
    out = body7.copy()
    out[:, 0] = (width - 1) - out[:, 0]
    out[[2, 4]] = out[[4, 2]]
    out[[3, 5]] = out[[5, 3]]
    return out


class SkeletonHold:
    """Keep last good skeleton when tracking drops (UI turns red; output holds)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last: np.ndarray | None = None
        self.lost: bool = True
        self.method: str = 'none'

    def update(
        self,
        body7: np.ndarray | None,
        method: str = 'none',
    ) -> tuple[np.ndarray | None, bool, str]:
        """Return (body7_to_use, is_lost, method_label)."""
        live = (
            body7 is not None
            and len(body7) >= 7
            and float(np.sum(body7[:, 3])) >= 3.0
        )
        if live:
            self.last = body7.copy()
            self.lost = False
            self.method = method or 'tracked'
            return self.last.copy(), False, self.method
        if self.last is not None:
            self.lost = True
            held_method = self.method if self.method not in ('none', '') else 'held'
            if not str(held_method).endswith('_held'):
                held_method = f'{held_method}_held'
            return self.last.copy(), True, held_method
        self.lost = True
        return None, True, 'none'


def draw_skeleton(
    frame: np.ndarray,
    body7: np.ndarray | None,
    *,
    lost: bool = False,
    color: tuple[int, int, int] | None = None,
    joint_color: tuple[int, int, int] | None = None,
) -> None:
    if body7 is None or len(body7) < 7:
        return
    if lost:
        color = color or (40, 40, 255)       # red (BGR)
        joint_color = joint_color or (60, 60, 255)
    else:
        color = color or (80, 220, 120)
        joint_color = joint_color or (40, 255, 180)

    pts = {}
    for i in range(7):
        # Held poses keep visibility flags; always draw xy if score>0 or held
        if float(body7[i, 3]) < 0.5 and float(body7[i, 2]) <= 0 and not lost:
            continue
        if float(body7[i, 0]) <= 0 and float(body7[i, 1]) <= 0:
            continue
        pts[BODY_NOSE + i] = (int(round(body7[i, 0])), int(round(body7[i, 1])))

    for a, b in SKELETON_BONES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    for idx, p in pts.items():
        r = 5 if idx in (NECK, CHEST) else 4
        cv2.circle(frame, p, r, joint_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(idx),
            (p[0] + 4, p[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (230, 230, 255) if lost else (230, 255, 230),
            1,
            cv2.LINE_AA,
        )
    if lost and pts:
        # small banner near neck/chest
        anchor = pts.get(NECK) or pts.get(CHEST) or next(iter(pts.values()))
        cv2.putText(
            frame,
            'LOST',
            (anchor[0] + 8, anchor[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 255),
            1,
            cv2.LINE_AA,
        )
