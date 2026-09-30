"""Tracker blink -> lid amount (0 open, 1 shut) per character eye.

A real closed eye rarely reads 1.0, so the blink is reshaped per source and
eased (fast close, softer reopen). The eyes themselves are the Eye open /
Eye closed shapes (presets.py), blended by this amount after the rig.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

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
