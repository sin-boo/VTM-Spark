"""Blink → eyelids on the posed character.

A real closed eye rarely reads 1.0, so the tracker's blink is reshaped per
source, eased (fast close, softer reopen), and the lid mid is laid on its
corner chord *after* smoothing and the head rig. Before, the lid moved a
linear share of the blink and then trailed the 28-point smoothing, so a
closed eye stopped part-open and the model kept drawing a pupil.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import numpy as np

# Raw blink (0 open, 1 shut) that reads as fully open / fully shut. ARKit
# tops out around 0.6-0.9 on a closed eye depending on the face, and
# OpenSeeFace's adaptive openness around 0.7-0.95 at 30 fps. Looking down
# lowers real lids too (ARKit ~0.3-0.45), which stays a partial lid.
SHUT_RANGE: dict[str, tuple[float, float]] = {
    "ifm": (0.15, 0.62),
    "osf": (0.18, 0.68),
}
# Seconds. Lids drop at once and reopen a touch slower, like a real blink,
# so one noisy frame cannot flash a closed eye open.
CLOSE_TAU = 0.008
OPEN_TAU = 0.045
# A gap longer than this is a stall, not a blink: jump to the new value.
_STALE_S = 0.5
# Character slots by screen side: (corner, lid mid, corner).
EYE_SLOTS: dict[str, tuple[int, int, int]] = {"l": (11, 12, 13), "r": (17, 18, 19)}


def _smoothstep(t: float) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else float(t)
    return t * t * (3.0 - 2.0 * t)


def _get(values: Mapping[str, float] | None, key: str) -> float:
    try:
        return float((values or {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def shape_blink(blink: Mapping[str, float] | None, source: str) -> dict[str, float]:
    """Raw tracker blink → lid amount (0 open, 1 shut) for ``source`` ("ifm" / "osf")."""
    lo, hi = SHUT_RANGE.get(source, SHUT_RANGE["osf"])
    return {
        key: round(_smoothstep((_get(blink, key) - lo) / (hi - lo)), 3)
        for key in ("l", "r")
    }


class LidFilter:
    """Per-eye ease: closing follows at once, reopening over ~50 ms."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last: dict[str, float] = {}
        self._t = 0.0

    def update(
        self, blink: Mapping[str, float] | None, now: float | None = None
    ) -> dict[str, float]:
        t = time.perf_counter() if now is None else float(now)
        dt = t - self._t if self._t else 0.0
        self._t = t
        out: dict[str, float] = {}
        for key in ("l", "r"):
            want = min(max(_get(blink, key), 0.0), 1.0)
            prev = self._last.get(key)
            if prev is None or dt <= 0.0 or dt > _STALE_S:
                out[key] = want
                continue
            tau = CLOSE_TAU if want > prev else OPEN_TAU
            out[key] = prev + (want - prev) * (1.0 - math.exp(-dt / tau))
        self._last = out
        return {key: round(value, 3) for key, value in out.items()}


def shut_lids(pts: np.ndarray | None, blink: Mapping[str, float] | None) -> np.ndarray | None:
    """Move each lid mid ``blink`` of the way onto its corner chord.

    Runs on the posed mesh, so the chord already carries the head's tilt and
    the lid closes along the eye's own up axis. Only the lid moves: levelling
    the corners here would undo the tilt. A lid already on or below its
    chord (an authored squint) is left alone.
    """
    if pts is None:
        return None
    out = np.array(pts, dtype=np.float32, copy=True)
    for key, (a, lid, b) in EYE_SLOTS.items():
        amount = min(max(_get(blink, key), 0.0), 1.0)
        if amount <= 0.0 or len(out) <= max(a, lid, b):
            continue
        if out.shape[1] > 2 and float(min(out[a, 2], out[lid, 2], out[b, 2])) < 0.05:
            continue
        ax, ay = float(out[a, 0]), float(out[a, 1])
        cx, cy = float(out[b, 0]) - ax, float(out[b, 1]) - ay
        span = cx * cx + cy * cy
        if span < 1e-9:
            continue
        px, py = float(out[lid, 0]) - ax, float(out[lid, 1]) - ay
        along = min(max((px * cx + py * cy) / span, 0.0), 1.0)
        fx, fy = along * cx, along * cy
        # The chord's normal that points up the screen (Y down), whichever
        # corner comes first. Positive height = the lid is open.
        nx, ny = (cy, -cx) if cx > 0.0 else (-cy, cx)
        if (px - fx) * nx + (py - fy) * ny <= 0.0:
            continue
        out[lid, 0] = ax + px + amount * (fx - px)
        out[lid, 1] = ay + py + amount * (fy - py)
    return out
