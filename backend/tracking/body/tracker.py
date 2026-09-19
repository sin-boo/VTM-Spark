"""Upper-body joints (slots 30–36): MediaPipe / YOLO, else face-synth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...paths import ensure_import_paths

ensure_import_paths()

from skeleton import (  # noqa: E402
    SkeletonHold,
    SkeletonLiteTracker,
    flip_body7_x,
    resolve_body7,
    resolve_skeleton_weights,
    synth_upper_body,
)
from tracking_filters import FilterSettings  # noqa: E402


@dataclass
class BodyTrack:
    """Seven upper-body joints in source pixels (KEYPOINT_SCHEMA 30–36)."""

    joints: np.ndarray | None = None
    method: str = "none"
    lost: bool = False


class BodyTracker:
    """Load a body pose model and hold last-good joints across dropouts."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._model: SkeletonLiteTracker | None = None
        self._hold = SkeletonHold()

    def load(self, *, device: str = "cpu") -> None:
        self.close()
        if not self.enabled:
            return
        weights = resolve_skeleton_weights(None)
        try:
            self._model = SkeletonLiteTracker(weights, device=device, backend="auto")
        except Exception as exc:
            print(f"Body tracker failed ({exc}) - synth fallback")
            self._model = None

    def close(self) -> None:
        model = self._model
        self._model = None
        self._hold.reset()
        if model is not None and hasattr(model, "close"):
            try:
                model.close()
            except Exception:
                pass

    def reset(self) -> None:
        self._hold.reset()

    @property
    def has_model(self) -> bool:
        return self._model is not None

    def resolve(
        self,
        frame,
        pts28: np.ndarray | None,
        filters: FilterSettings,
        *,
        allow_synth_fallback: bool | None = None,
        rel_pitch: float = 0.0,
        rel_yaw: float = 0.0,
        rel_roll: float = 0.0,
    ) -> tuple[np.ndarray | None, str]:
        """Detect joints without updating the hold buffer."""
        if not self.enabled:
            return None, "none"
        if allow_synth_fallback is None:
            allow_synth_fallback = self._model is None
        return resolve_body7(
            frame,
            pts28,
            self._model,
            allow_synth_fallback=allow_synth_fallback,
            rel_pitch=rel_pitch,
            rel_yaw=rel_yaw,
            rel_roll=rel_roll,
            vis_thr=filters.body_vis_thr(),
            box_conf=filters.iris_box_conf(),
        )

    def detect(
        self,
        frame,
        pts28: np.ndarray | None,
        filters: FilterSettings,
        *,
        allow_synth_fallback: bool | None = None,
        rel_pitch: float = 0.0,
        rel_yaw: float = 0.0,
        rel_roll: float = 0.0,
    ) -> BodyTrack:
        raw, method = self.resolve(
            frame,
            pts28,
            filters,
            allow_synth_fallback=allow_synth_fallback,
            rel_pitch=rel_pitch,
            rel_yaw=rel_yaw,
            rel_roll=rel_roll,
        )
        return self.hold_update(raw, method)

    def hold_update(self, joints: np.ndarray | None, method: str) -> BodyTrack:
        kept, lost, held_method = self._hold.update(joints, method)
        return BodyTrack(joints=kept, method=held_method, lost=bool(lost))

    def resynth_from_face(
        self,
        pts28: np.ndarray,
        *,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> BodyTrack:
        body = synth_upper_body(pts28, pitch=pitch, yaw=yaw, roll=roll)
        return self.hold_update(body, "synthetic_from_face")

    @property
    def has_hold(self) -> bool:
        return self._hold.last is not None

    @staticmethod
    def flip(track: BodyTrack, width: int) -> BodyTrack:
        if track.joints is None:
            return track
        return BodyTrack(
            joints=flip_body7_x(track.joints, int(width)),
            method=track.method,
            lost=track.lost,
        )
