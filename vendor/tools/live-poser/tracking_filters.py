"""Live tracking filters: detection sensitivity + motion smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bridge import IrisPoint


@dataclass
class FilterSettings:
    """UI-driven tracking knobs.

    sensitivity 0..100 — higher = higher detection thresholds (stricter).
    smoothing 0..100 — higher = heavier EMA on all motion.
    """

    sensitivity: float = 50.0
    smoothing: float = 40.0

    def detection_threshold(self, base: float = 0.35, lo: float = 0.12, hi: float = 0.78) -> float:
        t = float(np.clip(self.sensitivity, 0.0, 100.0)) / 100.0
        return float(lo + t * (hi - lo))

    def landmark_threshold(self) -> float:
        return self.detection_threshold(base=0.35, lo=0.12, hi=0.72)

    def iris_box_conf(self) -> float:
        return self.detection_threshold(base=0.25, lo=0.10, hi=0.65)

    def iris_pupil_vis(self) -> float:
        return self.detection_threshold(base=0.30, lo=0.15, hi=0.70)

    def body_vis_thr(self) -> float:
        return self.detection_threshold(base=0.45, lo=0.20, hi=0.80)

    def mediapipe_conf(self) -> float:
        return self.detection_threshold(base=0.50, lo=0.25, hi=0.85)

    def ema_alpha(self) -> float:
        """Blend weight for the *new* sample. smoothing=0 → 1.0 (raw)."""
        s = float(np.clip(self.smoothing, 0.0, 100.0)) / 100.0
        # s=0 → alpha=1; s=1 → alpha≈0.08 (very sticky)
        return float(1.0 - 0.92 * s)


def _ema_xy(prev: np.ndarray | None, cur: np.ndarray, alpha: float) -> np.ndarray:
    """EMA on columns 0:2; copy scores/visibility from cur."""
    out = cur.copy()
    if prev is None or alpha >= 0.999 or prev.shape != cur.shape:
        return out
    out[:, 0:2] = alpha * cur[:, 0:2] + (1.0 - alpha) * prev[:, 0:2]
    return out


def _ema_iris(prev: IrisPoint | None, cur: IrisPoint, alpha: float) -> IrisPoint:
    if cur is None:
        return IrisPoint()
    if not cur.visible or prev is None or not prev.visible or alpha >= 0.999:
        return IrisPoint(
            x=cur.x,
            y=cur.y,
            score=cur.score,
            visible=cur.visible,
            method=cur.method,
            eye_bbox=cur.eye_bbox,
        )
    return IrisPoint(
        x=alpha * cur.x + (1.0 - alpha) * prev.x,
        y=alpha * cur.y + (1.0 - alpha) * prev.y,
        score=cur.score,
        visible=cur.visible,
        method=cur.method,
        eye_bbox=cur.eye_bbox,
    )


class MotionSmoother:
    """Exponential smoothing for face / iris / body points."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._pts28: np.ndarray | None = None
        self._eye_lower: np.ndarray | None = None
        self._iris_r: IrisPoint | None = None
        self._iris_l: IrisPoint | None = None
        self._body7: np.ndarray | None = None

    def apply(
        self,
        *,
        pts28: np.ndarray | None,
        eye_lower: np.ndarray | None,
        right_iris: IrisPoint | None,
        left_iris: IrisPoint | None,
        body7: np.ndarray | None,
        smoothing: float,
        freeze_body: bool = False,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        IrisPoint | None,
        IrisPoint | None,
        np.ndarray | None,
    ]:
        alpha = FilterSettings(smoothing=smoothing).ema_alpha()
        if pts28 is None and body7 is None:
            # Keep body hold path usable even when face is gone
            if freeze_body and self._body7 is not None and body7 is None:
                return None, None, right_iris, left_iris, self._body7.copy()
            if not freeze_body:
                self.reset()
            return None, None, right_iris, left_iris, body7

        pts_s = None
        if pts28 is not None:
            pts_s = _ema_xy(self._pts28, pts28, alpha)
            self._pts28 = pts_s.copy()

        eye_s = None
        if eye_lower is not None:
            eye_s = _ema_xy(self._eye_lower, eye_lower, alpha)
            self._eye_lower = eye_s.copy()
        elif pts28 is not None:
            self._eye_lower = None

        ri = right_iris or IrisPoint()
        li = left_iris or IrisPoint()
        if pts28 is not None:
            ri_s = _ema_iris(self._iris_r, ri, alpha)
            li_s = _ema_iris(self._iris_l, li, alpha)
            self._iris_r = ri_s
            self._iris_l = li_s
        else:
            ri_s, li_s = ri, li

        body_s = None
        if body7 is not None:
            if freeze_body:
                # Hold last pose — do not blend toward noise/absence
                body_s = body7.copy()
                self._body7 = body_s.copy()
            else:
                body_s = _ema_xy(self._body7, body7, alpha)
                if self._body7 is not None and self._body7.shape == body7.shape:
                    body_s[:, 2:4] = body7[:, 2:4]
                self._body7 = body_s.copy()
        return pts_s, eye_s, ri_s, li_s, body_s
