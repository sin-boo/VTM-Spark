"""Hair-part polygons from animeseg_hair3.pt."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ...paths import ensure_import_paths

ensure_import_paths()

from hair_tracker import (  # noqa: E402
    HairHold,
    clone_hair_segments,
    create_hair_tracker,
    draw_hair_segments,
    flip_hair_pixels,
    resolve_hair_weights,
)


@dataclass
class HairTrack:
    """Three hair-part polygons (middle / left / right) in source pixels."""

    segments: list[dict[str, Any]] | None = None
    method: str = "none"
    lost: bool = False


class HairTracker:
    """Load the hair weights and hold last-good polygons across dropouts."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._model = None
        self._hold = HairHold()

    def load(self, *, device: str = "cpu") -> None:
        self.close()
        if not self.enabled:
            return
        path = resolve_hair_weights(None)
        if path is None:
            print("animeseg_hair3.pt not found — live hair tracking off")
            return
        try:
            self._model = create_hair_tracker(path, device=device)
            print(f"Hair model: {path.name} ({self._model.method})")
        except Exception as exc:
            print(f"Hair load failed: {exc}")
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

    def detect(self, frame, *, now: float | None = None) -> HairTrack:
        if not self.enabled:
            return HairTrack()
        raw = None
        raw_method = "none"
        if self._model is not None:
            try:
                raw = self._model.detect(frame)
                raw_method = self._model.method
            except Exception:
                raw = None
        segments, lost, method = self._hold.update(
            raw, raw_method, now=time.time() if now is None else float(now)
        )
        return HairTrack(segments=segments, method=method, lost=bool(lost))

    @staticmethod
    def flip(track: HairTrack, width: int) -> HairTrack:
        if not track.segments:
            return track
        return HairTrack(
            segments=flip_hair_pixels(track.segments, int(width)),
            method=track.method,
            lost=track.lost,
        )

    @staticmethod
    def draw(frame, track: HairTrack) -> None:
        if track.segments:
            draw_hair_segments(frame, track.segments, lost=track.lost)
