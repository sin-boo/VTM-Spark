"""How fast this PC runs the model, per batch size — so Auto adapts to the card.

Saved to ``models/hw_profile.json`` per (GPU, checkpoint, steps, speed boost),
so a PC is tuned once and a different card or model is tuned on its own.
Stdlib-only; the timing itself comes from :meth:`StreamEngine.time_batch`.

Call time grows almost linearly with batch size (fixed overhead + per-pose
cost; on a 5060 Ti 102/142/175/227 ms for ×1–×4, a line through ×1 and ×2 is
within 6 %), so two measured sizes predict the rest.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

PROFILE_NAME = "hw_profile.json"
# Busy share Auto aims to stay under at the target rate. Above it a game, OBS
# or a hiccup pushes keys below target, and the GPU never idles.
MAX_DUTY = 0.8
# A bigger batch must buy at least this much more keys/s to be worth its lag.
MIN_GAIN = 1.10
# Measured is "close enough" to target.
TARGET_SLACK = 0.95
# Calls timed per batch size when tuning.
TIME_RUNS = 6

_lock = threading.Lock()


def profile_path() -> Path:
    from .paths import data_dir

    return data_dir() / PROFILE_NAME


def profile_key(gpu: str, checkpoint: str, steps: int, compiled: bool) -> str:
    return f"{gpu or 'cpu'}|{checkpoint}|steps={int(steps)}|{'boost' if compiled else 'eager'}"


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_rates(key: str, path: Path | None = None) -> dict[int, float]:
    """``{batch: seconds per call}`` measured on this PC for ``key``."""
    entry = _read(path or profile_path()).get(key)
    if not isinstance(entry, dict):
        return {}
    out: dict[int, float] = {}
    for b, secs in (entry.get("call_s") or {}).items():
        try:
            n, s = int(b), float(secs)
        except (TypeError, ValueError):
            continue
        if n >= 1 and s > 0.0:
            out[n] = s
    return out


def load_failed(key: str, path: Path | None = None) -> set[int]:
    """Batch sizes that ran out of memory on this PC."""
    entry = _read(path or profile_path()).get(key)
    if not isinstance(entry, dict):
        return set()
    return {int(b) for b in entry.get("oom") or [] if str(b).isdigit()}


def save_rate(
    key: str,
    batch: int,
    call_s: float,
    *,
    weight: float = 1.0,
    path: Path | None = None,
) -> dict[int, float]:
    """Record a timing. ``weight`` < 1 blends into what is there (live streams
    nudge the profile; a fresh tuning run replaces it)."""
    target = path or profile_path()
    with _lock:
        data = _read(target)
        entry = data.get(key) if isinstance(data.get(key), dict) else {}
        calls = dict(entry.get("call_s") or {})
        old = calls.get(str(int(batch)))
        w = min(max(float(weight), 0.0), 1.0)
        try:
            value = float(call_s) if old is None else (1.0 - w) * float(old) + w * float(call_s)
        except (TypeError, ValueError):
            value = float(call_s)
        calls[str(int(batch))] = round(value, 5)
        entry["call_s"] = calls
        data[key] = entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return load_rates(key, target)


def save_failed(key: str, batch: int, path: Path | None = None) -> None:
    target = path or profile_path()
    with _lock:
        data = _read(target)
        entry = data.get(key) if isinstance(data.get(key), dict) else {}
        oom = sorted({*(int(b) for b in entry.get("oom") or []), int(batch)})
        entry["oom"] = oom
        data[key] = entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def predict_call_s(rates: dict[int, float], batch: int) -> float | None:
    """Seconds per call at ``batch``: measured, else a line through the rest."""
    if batch in rates:
        return rates[batch]
    if len(rates) < 2:
        return None
    xs = sorted(rates)
    n = float(len(xs))
    mx = sum(xs) / n
    my = sum(rates[x] for x in xs) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0.0:
        return None
    slope = sum((x - mx) * (rates[x] - my) for x in xs) / var
    # A batch never costs less than the biggest one measured.
    return max(my + slope * (batch - mx), max(rates.values()))


def keys_per_s(rates: dict[int, float], batch: int) -> float | None:
    t = predict_call_s(rates, batch)
    return None if not t else batch / t


def plan_batch(
    rates: dict[int, float],
    target_keys_s: float,
    max_batch: int,
    *,
    failed: set[int] | None = None,
) -> tuple[int, int | None]:
    """``(batch to stream at, batch to time next or None)``.

    Smallest batch that reaches the target key rate while leaving the GPU
    ``1 - MAX_DUTY`` idle — smaller batches lag less. If none can, the one
    with the most keys/s, stepping up only while each step gains ``MIN_GAIN``.
    Sizes that ran out of memory are never picked.
    """
    # A size that ran out of memory rules out every bigger one too.
    ceiling = min([int(max_batch), *[b - 1 for b in (failed or set())]])
    sizes = list(range(1, max(1, ceiling) + 1))
    if 1 not in rates:
        return 1, 1
    target = max(float(target_keys_s), 0.1)

    def fits(b: int) -> bool:
        t = predict_call_s(rates, b)
        if not t:
            return False
        rate = b / t
        duty = t * min(rate, target) / b
        return rate >= target * TARGET_SLACK and duty <= MAX_DUTY

    if fits(1):
        return 1, None
    if len(rates) < 2 and 2 in sizes:
        return 1, 2
    best = 1
    for b in sizes:
        if b == 1:
            continue
        rate = keys_per_s(rates, b)
        if rate is None:
            break
        if fits(b):
            return b, (b if b not in rates else None)
        if rate >= (keys_per_s(rates, best) or 0.0) * MIN_GAIN:
            best = b
        else:
            break
    return best, (best if best not in rates else None)
