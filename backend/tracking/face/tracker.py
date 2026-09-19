"""OpenSeeFace 66 landmarks → Label28 face slots 0–27 (mouth included)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...paths import ensure_import_paths

ensure_import_paths()

from label_schema import (  # noqa: E402
    flip_label28_x,
    normalize_mouth_osf_map,
    osf_to_label28,
)


@dataclass
class FaceTrack:
    """One frame of Label28 face landmarks (slots 0–27)."""

    pts28: np.ndarray | None = None
    eye_lower: np.ndarray | None = None
    mouth_osf: dict[int, int] = field(default_factory=dict)
    alive: bool = False


class FaceTracker:
    """Decode OpenSeeFace landmarks; mouth slots 20–27 are remappable live."""

    def __init__(self) -> None:
        self._mouth_osf = normalize_mouth_osf_map(None)

    def set_mouth_osf_map(self, raw: object | None) -> dict[int, int]:
        mapped = normalize_mouth_osf_map(raw)
        self._mouth_osf = mapped
        return mapped

    def mouth_osf_map(self) -> dict[int, int]:
        return dict(self._mouth_osf)

    def decode(self, face, *, landmark_threshold: float = 0.15) -> FaceTrack:
        """Map one OSF face (``face.lms``) into Label28."""
        lms = getattr(face, "lms", None)
        if lms is None:
            return FaceTrack(mouth_osf=self.mouth_osf_map(), alive=False)
        pts28, eye_lower = osf_to_label28(lms, mouth_osf=self._mouth_osf)
        if pts28 is not None:
            weak = pts28[:, 2] < float(landmark_threshold)
            pts28[weak, 2] = 0.0
        return FaceTrack(
            pts28=pts28,
            eye_lower=eye_lower,
            mouth_osf=self.mouth_osf_map(),
            alive=pts28 is not None,
        )

    @staticmethod
    def flip(track: FaceTrack, width: int) -> FaceTrack:
        if track.pts28 is None:
            return track
        pts28, eye_lower = flip_label28_x(track.pts28, int(width), track.eye_lower)
        return FaceTrack(
            pts28=pts28,
            eye_lower=eye_lower,
            mouth_osf=dict(track.mouth_osf),
            alive=track.alive,
        )
