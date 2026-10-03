"""Performance test: the character moves on a fixed loop while the desk times itself.

Run it once alone and once beside a game; the drop between the two runs is what
the game costs. The motion needs no camera, so every run sees the same poses.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

# One loop of head turn, nod and mouth: a few loops fit even the shortest test.
MOTION_PERIOD_S = 4.0
# Timed from the first picture: model load and graph build are not what the test
# measures, nor is the first second of replays.
WARMUP_S = 1.5
# No picture by then: the stream never got going.
START_TIMEOUT_S = 120.0
# A gap this long between shown pictures is a visible hitch.
STALL_S = 0.25
DURATIONS_S = (15, 30, 60)
MAX_RESULTS = 6

FACE = tuple(range(0, 31))  # face mesh, irises, body nose
NECK = 31
MOUTH_LOWER = (24, 25, 26, 27)
CHIN = 2


def motion_pose(base: np.ndarray, t: float) -> np.ndarray:
    """``base`` (37, 4) with the head tilting, turning and nodding, and the mouth opening."""
    out = np.asarray(base, dtype=np.float32).copy()
    face = list(FACE)
    xy = out[face, :2]
    span = float(np.ptp(xy[:, 1])) or 0.3
    phase = 2.0 * math.pi * (t / MOTION_PERIOD_S)
    pivot = out[NECK, :2].copy()
    roll = 0.12 * math.sin(phase)
    c, s = math.cos(roll), math.sin(roll)
    rel = xy - pivot
    xy = pivot + rel @ np.array([[c, s], [-s, c]], dtype=np.float32)
    xy[:, 0] += 0.18 * span * math.sin(2.0 * phase)
    xy[:, 1] += 0.06 * span * math.sin(phase + 1.0)
    out[face, :2] = xy
    jaw = 0.12 * span * max(0.0, math.sin(3.0 * phase))
    out[list(MOUTH_LOWER), 1] += jaw
    out[CHIN, 1] += 0.5 * jaw
    out[:, 0] += 0.04 * math.sin(phase + 2.0)
    return out


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class PerfRun:
    """One test: shown pictures, model calls and machine samples over ``seconds``."""

    def __init__(self, seconds: float, now: float, *, label: str = "", started_stream: bool = False) -> None:
        self.seconds = float(seconds)
        self.label = label
        self.started_stream = started_stream
        self.t0 = float(now)
        self._first: float | None = None
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._frames: list[float] = []
        self._calls: list[tuple[float, int]] = []
        self._gpu: list[float] = []
        self._vram: list[float] = []
        self._cpu: list[float] = []

    def clock(self, now: float) -> float:
        """Seconds since the test started (drives the motion)."""
        return float(now) - self.t0

    def _since_first(self, now: float) -> float | None:
        return None if self._first is None else float(now) - self._first

    def _measuring(self, now: float) -> bool:
        since = self._since_first(now)
        return since is not None and since >= WARMUP_S

    def done(self, now: float) -> bool:
        if self.stop.is_set():
            return True
        since = self._since_first(now)
        if since is None:
            return self.clock(now) >= START_TIMEOUT_S
        return since >= WARMUP_S + self.seconds

    def note_frame(self, now: float) -> None:
        with self._lock:
            if self._first is None:
                self._first = float(now)
            if self._measuring(now):
                self._frames.append(float(now))

    def note_call(self, now: float, elapsed: float, keys: int) -> None:
        if self._measuring(now):
            with self._lock:
                self._calls.append((float(elapsed), max(1, int(keys))))

    def sample(self, now: float, *, gpu: float | None, vram_mb: float | None, cpu: float | None) -> None:
        if not self._measuring(now):
            return
        with self._lock:
            if gpu is not None:
                self._gpu.append(float(gpu))
            if vram_mb is not None:
                self._vram.append(float(vram_mb))
            if cpu is not None:
                self._cpu.append(float(cpu))

    def progress(self, now: float) -> float:
        since = self._since_first(now)
        return 0.0 if since is None else max(0.0, min(1.0, since / (WARMUP_S + self.seconds)))

    def result(self) -> dict[str, Any]:
        with self._lock:
            frames = list(self._frames)
            calls = list(self._calls)
            gpu, vram, cpu = list(self._gpu), list(self._vram), list(self._cpu)
        span = frames[-1] - frames[0] if len(frames) >= 2 else 0.0
        gaps = list(np.diff(frames)) if len(frames) >= 2 else []
        # 1% low: frame rate over the slowest 1% of gaps (at least one), so a
        # single hitch in a short run still shows.
        slow = sorted(gaps)[-max(1, math.ceil(len(gaps) / 100)):] if gaps else []
        worst = _mean(slow)
        keys = sum(k for _, k in calls)
        per_key_ms = [1000.0 * e / k for e, k in calls]
        return {
            "label": self.label,
            "seconds": round(span, 1),
            "fps": round((len(frames) - 1) / span, 1) if span > 0 else 0.0,
            "fps_low": round(1.0 / worst, 1) if worst > 0 else 0.0,
            "stalls": sum(1 for g in gaps if g >= STALL_S),
            "keys_per_s": round(keys / span, 1) if span > 0 else 0.0,
            "key_ms": round(_pct(per_key_ms, 50), 1),
            "key_ms_p95": round(_pct(per_key_ms, 95), 1),
            "gpu": round(_mean(gpu)) if gpu else None,
            "vram_mb": round(max(vram)) if vram else None,
            "cpu": round(_mean(cpu)) if cpu else None,
        }
