"""Progress from measured stage times, not guesses.

Model loads and the stream warmup run a fixed list of stages (read the DiT,
load the VAE, compile, first decode …). Each one's real duration on this PC is
remembered in ``models/load_timings.json``; the next run shows the bar against
those times, so it reaches 90–100 % when the work really ends. A stage that
runs long eases toward the end of its slice instead of jumping past it.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import data_dir

TIMINGS_NAME = "load_timings.json"

# First-run guesses (seconds) before this PC has measured anything.
DEFAULT_SECONDS: dict[str, float] = {
    "dit": 7.0,  # per GB of checkpoint (see PER_GB)
    "vae": 4.0,
    "character": 3.0,
    "gpu_move": 2.0,
    "tiny_vae": 1.0,
    "compile_wrap": 0.5,
    "compile_run": 12.0,
    "warm_run": 0.3,
    "decode": 7.0,
    "verify": 0.3,
    "graph": 20.0,
}

# Stages whose time grows with the checkpoint file are stored per GB.
PER_GB = {"dit"}

# Weight of a new measurement against the stored one.
_BLEND = 0.5
# A stage fills linearly to this share of its slice by its expected time,
# then eases toward (but never reaches) the end of the slice.
_LINEAR_TO = 0.9
_EASE_TO = 0.99


def timings_path() -> Path:
    return data_dir() / TIMINGS_NAME


class StageClock:
    """Remembered stage durations for this PC."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or timings_path()
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._seen = {
                    str(k): float(v)
                    for k, v in raw.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0
                }
        except (OSError, ValueError):
            pass

    def expected(self, key: str, *, gb: float = 1.0) -> float:
        with self._lock:
            base = self._seen.get(key, DEFAULT_SECONDS.get(key, 2.0))
        return max(0.05, base * (gb if key in PER_GB else 1.0))

    def record(self, key: str, seconds: float, *, gb: float = 1.0) -> None:
        secs = float(seconds)
        if not math.isfinite(secs) or secs <= 0:
            return
        if key in PER_GB:
            secs /= max(gb, 0.05)
        with self._lock:
            old = self._seen.get(key)
            self._seen[key] = secs if old is None else old + _BLEND * (secs - old)
            snapshot = dict(self._seen)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass


def stage_fill(elapsed: float, expected: float) -> float:
    """Share of one stage's slice after ``elapsed`` seconds (0 … <1)."""
    exp = max(float(expected), 0.05)
    t = max(0.0, float(elapsed))
    if t <= exp:
        return _LINEAR_TO * t / exp
    over = (t - exp) / exp
    return _LINEAR_TO + (_EASE_TO - _LINEAR_TO) * (1.0 - math.exp(-over))


@dataclass
class StagePlan:
    """Ordered stages with expected seconds; turns stage + elapsed into 0..1."""

    stages: list[tuple[str, float]]
    _index: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._index = {key: i for i, (key, _secs) in enumerate(self.stages)}

    @property
    def total(self) -> float:
        return max(sum(secs for _key, secs in self.stages), 1e-6)

    def has(self, key: str) -> bool:
        return key in self._index

    def fraction(self, key: str, elapsed: float, *, inner: float | None = None) -> float:
        """Overall share with ``key`` running for ``elapsed`` s.

        ``inner`` (0..1) is a real measure of the stage itself (bytes read);
        the larger of it and the timed fill wins.
        """
        i = self._index.get(key)
        if i is None:
            return 0.0
        before = sum(secs for _key, secs in self.stages[:i])
        secs = self.stages[i][1]
        fill = stage_fill(elapsed, secs)
        if inner is not None:
            fill = max(fill, min(_EASE_TO, max(0.0, float(inner))))
        return min(1.0, (before + fill * secs) / self.total)

    def done_through(self, key: str) -> float:
        """Share once ``key`` has finished."""
        i = self._index.get(key)
        if i is None:
            return 0.0
        return min(1.0, sum(secs for _key, secs in self.stages[: i + 1]) / self.total)


class StageMeter:
    """Drive a progress bar from stage announcements.

    Feed it ``on_stage(key, label)`` from the engine. A small thread reports
    ``lo + (hi - lo) * plan.fraction(...)`` every tick; each finished stage's
    real duration goes back into the clock for next time. Unknown keys only
    change the label; ``"done"`` closes the last stage.
    """

    def __init__(
        self,
        clock: StageClock,
        keys: list[str],
        report: Callable[[float, str], None],
        *,
        lo: float = 0.0,
        hi: float = 1.0,
        gb: float = 1.0,
        inner: Callable[[str], float | None] | None = None,
        tick_s: float = 0.15,
        label: str = "",
    ) -> None:
        self._clock = clock
        self._gb = float(gb)
        self._plan = StagePlan([(key, clock.expected(key, gb=gb)) for key in keys])
        self._report = report
        self._lo = float(lo)
        self._hi = max(float(hi), self._lo)
        self._inner = inner
        self._tick_s = float(tick_s)
        self._lock = threading.Lock()
        self._key: str | None = None
        self._label = label
        self._t0 = 0.0
        self._shown = self._lo
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "StageMeter":
        self._tick()
        self._thread = threading.Thread(target=self._run, name="vtm-stage-meter", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def stage(self, key: str, label: str) -> None:
        now = time.monotonic()
        with self._lock:
            prev = self._key
            if prev is not None and prev != key:
                self._clock.record(prev, now - self._t0, gb=self._gb)
            if label:
                self._label = label
            if key == "done":
                self._key = None
            elif self._plan.has(key):
                if key != prev:
                    self._t0 = now
                self._key = key
        self._tick()

    def _value(self) -> float:
        with self._lock:
            key = self._key
            elapsed = time.monotonic() - self._t0
        if key is None:
            return self._shown
        inner = None
        if self._inner is not None:
            try:
                inner = self._inner(key)
            except Exception:
                inner = None
        frac = self._plan.fraction(key, elapsed, inner=inner)
        return self._lo + (self._hi - self._lo) * frac

    def _tick(self) -> None:
        value = self._value()
        with self._lock:
            self._shown = max(self._shown, min(self._hi, value))
            shown, label = self._shown, self._label
        try:
            self._report(shown, label)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.wait(self._tick_s):
            self._tick()
