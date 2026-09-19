"""Iris / pupil: custom ``iris_pose.pt`` merged with OpenSeeFace gaze."""

from __future__ import annotations

from dataclasses import dataclass

from ...paths import ensure_import_paths

ensure_import_paths()

from bridge import IrisPoint  # noqa: E402
from iris_tracker import (  # noqa: E402
    CustomIrisTracker,
    merge_iris,
    osf_gaze_to_iris,
    resolve_iris_weights,
)
from tracking_filters import FilterSettings  # noqa: E402


@dataclass
class IrisTrack:
    """KEYPOINT_SCHEMA slots 28 (right) and 29 (left)."""

    right: IrisPoint
    left: IrisPoint
    method: str = "none"


class IrisTracker:
    """Load iris_pose.pt and merge each frame with OSF gaze."""

    def __init__(self, *, use_iris: bool = True, use_gaze: bool = True) -> None:
        self.use_iris = bool(use_iris)
        self.use_gaze = bool(use_gaze)
        self._model: CustomIrisTracker | None = None

    def load(self, *, device: str = "cpu") -> None:
        self.close()
        if not self.use_iris:
            return
        weights = resolve_iris_weights(None)
        if weights is None:
            return
        try:
            # Always CPU by default — DiT/VAE own the GPU.
            self._model = CustomIrisTracker(weights, device=device)
        except Exception as exc:
            print(f"Iris load failed: {exc}")
            self._model = None

    def close(self) -> None:
        model = self._model
        self._model = None
        if model is not None and hasattr(model, "close"):
            try:
                model.close()
            except Exception:
                pass

    def detect(self, frame, face, pts28, filters: FilterSettings) -> IrisTrack:
        custom = None
        osf = None
        if self.use_iris and self._model is not None and pts28 is not None:
            try:
                custom = self._model.match_to_face(
                    frame,
                    pts28,
                    conf=filters.iris_box_conf(),
                    pupil_vis_thr=filters.iris_pupil_vis(),
                )
            except Exception:
                custom = None
        if self.use_gaze and face is not None:
            try:
                osf = osf_gaze_to_iris(face, conf_thr=filters.iris_pupil_vis())
            except Exception:
                osf = None
        right, left, method = merge_iris(custom, osf, prefer="custom_then_osf")
        return IrisTrack(right=right, left=left, method=method)

    @staticmethod
    def flip(track: IrisTrack, width: int) -> IrisTrack:
        from bridge import flip_iris_pair

        right, left = flip_iris_pair(track.right, track.left, int(width))
        return IrisTrack(right=right, left=left, method=track.method)
