"""Smooth for the head: one ease, timed in seconds, for every source.

A One-Euro filter (Casiez 2012). Held still, the ease is at its heaviest and
hides tracker jitter; the faster the head moves, the lighter it gets, so a
turn is not dragged far behind. The move's speed is itself eased, so the
ease opens and closes over a few frames: a move starts and settles softly
instead of snapping to the raw reading and stopping dead.

Timed in seconds, not frames, so one Smooth feels the same on a 12 fps
webcam and a 60 fps iPhone. The old per-frame blend let go entirely once a
frame moved ~4 % of the face, which a low frame rate did on nearly every
frame of a move: Smooth only ever touched a head held still.
"""

from __future__ import annotations

import math

import numpy as np

# Smooth 0 -> 1, eased between the ends on a log scale. On a real phone
# recording a move trails ~20 ms at 0.25, ~40 ms at 0.5, ~85 ms at 0.75 and
# ~160-180 ms at 1, at 10 fps as at 35. Smooth 0 is the raw head.
# Cutoff with the head still (Hz): time constant ~30 ms -> ~640 ms.
_STILL_HZ = (5.3, 0.25)
# Cutoff added per face width a second of movement (Hz).
_OPEN = (20.0, 0.5)
# Cutoff for the move's own speed (Hz).
_SPEED_HZ = 1.0
# Frames closer than this are timed as this: a burst of queued frames must
# not read as the head flying. Longer gaps (a stall) ease no further.
_DT_MIN = 1.0 / 120.0
_DT_MAX = 0.25
_FIRST_DT = 1.0 / 30.0


def _between(ends: tuple[float, float], t: float) -> float:
    lo, hi = ends
    return float(lo * (hi / lo) ** t)


def cutoffs(smooth: float) -> tuple[float, float]:
    """(still cutoff Hz, Hz added per face width a second) for Smooth 0-1."""
    t = min(max(float(smooth), 0.0), 1.0)
    return _between(_STILL_HZ, t), _between(_OPEN, t)


def ease_weight(dt: float, cutoff_hz: float) -> float:
    """Share of the way to the new reading after ``dt`` seconds."""
    tau = 1.0 / (2.0 * math.pi * max(float(cutoff_hz), 1e-6))
    return 1.0 / (1.0 + tau / max(float(dt), 1e-6))


class HeadEase:
    """Eases a vector of head values with one shared weight.

    One weight keeps the parts of a move together: a turn and the slide it
    swings the head by land at the same time. ``units`` turns each value
    into face widths for the speed (0 leaves it out of the speed).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._value: np.ndarray | None = None
        self._raw: np.ndarray | None = None
        # Eased signed velocity (face widths a second, per value).
        self._velocity: np.ndarray | None = None
        self._t: float | None = None

    def step(
        self,
        target: np.ndarray | tuple[float, ...],
        units: np.ndarray | tuple[float, ...],
        smooth: float,
        now: float,
        *,
        snap: bool = False,
    ) -> np.ndarray:
        goal = np.asarray(target, dtype=np.float64)
        last_t = self._t
        last_raw = self._raw
        self._t = float(now)
        self._raw = goal.copy()
        if snap or smooth <= 0.0 or self._value is None or self._value.shape != goal.shape:
            self._value = goal.copy()
            self._velocity = np.zeros_like(goal)
            return self._value.copy()
        dt = _FIRST_DT if last_t is None else float(now) - last_t
        dt = min(max(dt, _DT_MIN), _DT_MAX)
        # The reading's own speed, not how far behind the ease is: that
        # gap over a frame read 5x faster at 60 fps than at 12.
        step = (goal - (goal if last_raw is None else last_raw)) * np.asarray(units, dtype=np.float64)
        # Ease the signed velocity, then take its size. Jitter flips sign
        # frame to frame and cancels here; easing the size alone kept a still
        # head reading as moving, which held the ease open at a third of its
        # time constant.
        if self._velocity is None or self._velocity.shape != step.shape:
            self._velocity = np.zeros_like(step)
        self._velocity += ease_weight(dt, _SPEED_HZ) * (step / dt - self._velocity)
        still_hz, open_hz = cutoffs(smooth)
        weight = ease_weight(dt, still_hz + open_hz * float(np.linalg.norm(self._velocity)))
        self._value += weight * (goal - self._value)
        return self._value.copy()
