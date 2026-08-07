"""Iris / pupil tracking for Live Poser.

Sources (prefer custom when available):
1. Custom YOLO-pose `iris_pose.pt` (same as pose-traker auto-labeler)
2. OpenSeeFace gaze (`eye_state` / lms 66-67) as fallback

Both are mapped into KEYPOINT_SCHEMA slots:
  28 = right_iris (person's right)
  29 = left_iris  (person's left)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bridge import IrisPoint
from label_schema import LEFT_EYE, RIGHT_EYE

ROOT = Path(__file__).resolve().parent
# Prefer package models/trackers (HF download / ship location), then vendor mirrors.
_PACKAGE_TRACKERS = ROOT.parent.parent.parent / 'models' / 'trackers'
DEFAULT_IRIS_CANDIDATES = [
    _PACKAGE_TRACKERS / 'iris_pose.pt',
    ROOT / 'models' / 'iris_pose.pt',
    ROOT.parent / 'pose-traker' / 'models' / 'iris_pose.pt',
    ROOT.parent / 'pose-traker' / 'iris-model' / 'models' / 'iris_pose.pt',
]

IRIS_DETECT_CONF = 0.25
IRIS_PUPIL_VIS_THR = 0.3


def resolve_iris_weights(explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    for c in DEFAULT_IRIS_CANDIDATES:
        if c.is_file():
            return c
    return None


class CustomIrisTracker:
    """Thin wrapper around Ultralytics YOLO-pose iris_pose.pt."""

    def __init__(self, weights: Path | str | None = None, device: str | None = None):
        path = resolve_iris_weights(weights)
        if path is None:
            raise FileNotFoundError(
                'iris_pose.pt not found. Place it at live-poser/models/iris_pose.pt '
                'or pose-traker/models/iris_pose.pt'
            )
        from ultralytics import YOLO  # lazy

        self.weights = Path(path)
        self.model = YOLO(str(self.weights))
        self.device = device
        self.model._iris_device = device

    def detect(
        self,
        image_bgr: np.ndarray,
        conf: float = IRIS_DETECT_CONF,
        pupil_vis_thr: float = IRIS_PUPIL_VIS_THR,
    ) -> list[dict]:
        results = self.model.predict(
            source=image_bgr,
            conf=conf,
            verbose=False,
            device=self.device,
        )
        eyes: list[dict] = []
        if not results:
            return eyes
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return eyes
        boxes = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        kpts = None
        if r0.keypoints is not None and r0.keypoints.data is not None:
            kpts = r0.keypoints.data.cpu().numpy()
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = [float(v) for v in box]
            pupil = None
            visible = False
            if kpts is not None and i < len(kpts):
                kp = kpts[i][0]
                px, py, pv = float(kp[0]), float(kp[1]), float(kp[2])
                if pv >= pupil_vis_thr:
                    pupil = (px, py)
                    visible = True
            eyes.append(
                {
                    'bbox': (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
                    'pupil': pupil,
                    'score': float(confs[i]),
                    'visible': visible,
                    'cx': (x0 + x1) / 2.0,
                    'cy': (y0 + y1) / 2.0,
                }
            )
        return eyes

    def match_to_face(
        self,
        image_bgr: np.ndarray,
        pts28: np.ndarray,
        *,
        conf: float = IRIS_DETECT_CONF,
        pupil_vis_thr: float = IRIS_PUPIL_VIS_THR,
    ) -> tuple[IrisPoint, IrisPoint]:
        """Return (right_iris, left_iris) matched to Label28 eye anchors."""
        try:
            dets = self.detect(image_bgr, conf=conf, pupil_vis_thr=pupil_vis_thr)
        except Exception:
            return IrisPoint(method='iris_pose'), IrisPoint(method='iris_pose')

        used: set[int] = set()

        def _eye_anchor(eye_ids: tuple[int, ...]) -> np.ndarray | None:
            pts = [
                pts28[i, 0:2].astype(np.float64)
                for i in eye_ids
                if i < len(pts28) and float(pts28[i, 2]) >= 0.15
            ]
            return np.mean(np.stack(pts, axis=0), axis=0) if pts else None

        right_anchor = _eye_anchor(RIGHT_EYE)
        left_anchor = _eye_anchor(LEFT_EYE)

        def _for_side(
            eye_ids: tuple[int, ...],
            other_anchor: np.ndarray | None,
        ) -> IrisPoint:
            pts = []
            for i in eye_ids:
                if i < len(pts28) and float(pts28[i, 2]) >= 0.15:
                    pts.append(pts28[i, 0:2].astype(np.float64))
            if not pts:
                return IrisPoint(method='iris_pose')
            pts_a = np.stack(pts, axis=0)
            anchor = pts_a.mean(axis=0)
            eye_w = float(pts_a[:, 0].max() - pts_a[:, 0].min())
            max_dist = max(40.0, eye_w * 1.5)

            best_i = None
            best_d = 1e9
            for i, det in enumerate(dets):
                if i in used:
                    continue
                d = float(np.hypot(det['cx'] - anchor[0], det['cy'] - anchor[1]))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is None or best_d > max_dist:
                return IrisPoint(method='iris_pose')
            det = dets[best_i]
            if other_anchor is not None:
                other_d = float(
                    np.hypot(det['cx'] - other_anchor[0], det['cy'] - other_anchor[1])
                )
                if other_d + max(2.0, eye_w * 0.10) < best_d:
                    return IrisPoint(method='iris_pose')
            used.add(best_i)
            if det['visible'] and det['pupil'] is not None:
                px, py = det['pupil']
                return IrisPoint(
                    x=float(px),
                    y=float(py),
                    score=float(det['score']),
                    visible=True,
                    method='iris_pose',
                    eye_bbox=list(det['bbox']),
                )
            return IrisPoint(
                score=float(det['score']),
                visible=False,
                method='iris_pose',
                eye_bbox=list(det['bbox']),
            )

        # Schema: RIGHT_EYE = person right, LEFT_EYE = person left
        right = _for_side(RIGHT_EYE, left_anchor)
        left = _for_side(LEFT_EYE, right_anchor)
        return right, left


def osf_gaze_to_iris(face, *, conf_thr: float = 0.15) -> tuple[IrisPoint, IrisPoint]:
    """Extract pupils from OpenSeeFace face.eye_state / extended lms.

    OSF eye_state[0] = person right, [1] = person left.
    Each entry: [openness, y, x, conf]  (y,x image coords).
    Extended lms[66]/[67] store (y, x, conf) the same way.
    """
    right = IrisPoint(method='openseeface_gaze')
    left = IrisPoint(method='openseeface_gaze')

    lms = getattr(face, 'lms', None)
    eye_state = getattr(face, 'eye_state', None)

    def _from_yx(y: float, x: float, conf: float) -> IrisPoint:
        ok = conf >= conf_thr and x > 1.0 and y > 1.0
        return IrisPoint(
            x=float(x),
            y=float(y),
            score=float(conf),
            visible=bool(ok),
            method='openseeface_gaze',
        )

    if lms is not None and len(lms) >= 68:
        # lms rows are (y, x, conf)
        right = _from_yx(float(lms[66, 0]), float(lms[66, 1]), float(lms[66, 2]))
        left = _from_yx(float(lms[67, 0]), float(lms[67, 1]), float(lms[67, 2]))
        return right, left

    if eye_state is not None and len(eye_state) >= 2:
        # eye_state[i] = [open, y, x, conf]
        er, el = eye_state[0], eye_state[1]
        right = _from_yx(float(er[1]), float(er[2]), float(er[3]))
        left = _from_yx(float(el[1]), float(el[2]), float(el[3]))
    return right, left


def merge_iris(
    custom: tuple[IrisPoint, IrisPoint] | None,
    osf: tuple[IrisPoint, IrisPoint] | None,
    *,
    prefer: str = 'custom',
) -> tuple[IrisPoint, IrisPoint, str]:
    """Merge custom + OSF iris. prefer: 'custom' | 'osf' | 'custom_then_osf'."""
    c_r, c_l = custom or (IrisPoint(), IrisPoint())
    o_r, o_l = osf or (IrisPoint(), IrisPoint())

    def _pick(c: IrisPoint, o: IrisPoint) -> IrisPoint:
        if prefer == 'osf':
            return o if o.visible else (c if c.visible else o)
        if prefer == 'custom':
            return c if c.visible else (o if o.visible else c)
        # custom_then_osf
        if c.visible:
            return c
        if o.visible:
            return o
        return c if c.method != 'none' else o

    right = _pick(c_r, o_r)
    left = _pick(c_l, o_l)
    methods = {right.method, left.method} - {'none'}
    if not methods:
        method = 'none'
    elif len(methods) == 1:
        method = next(iter(methods))
    else:
        method = 'mixed'
    return right, left, method


def draw_iris(
    frame: np.ndarray,
    right: IrisPoint | None,
    left: IrisPoint | None,
) -> None:
    import cv2

    for p, color, tag in (
        (right, (0, 220, 255), 'R'),
        (left, (255, 180, 0), 'L'),
    ):
        if p is None:
            continue
        if p.eye_bbox is not None and len(p.eye_bbox) == 4:
            x0, y0, x1, y1 = [int(v) for v in p.eye_bbox]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        if p.visible:
            x, y = int(round(p.x)), int(round(p.y))
            cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 7, color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                tag,
                (x + 6, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
