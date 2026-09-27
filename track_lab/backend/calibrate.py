"""Capture rest plus independent mouth setups that mix together.

Smile/sad/oh move a lot. I/E/U move a little. Matching each bank in its
own geometry scale lets both groups work instead of one crowding out the other.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

from .presets import FORM_IDS, MOUTH_BANKS, PRESET_LABELS, VOWEL_IDS, empty_weights

ROOT = Path(__file__).resolve().parents[1]
CALIB_PATH = ROOT / "output" / "webcam_calibration.json"

CAPTURE_SEC = 1.2
WARMUP_SEC = 0.18
CAPTURE_IDS = ("rest",) + FORM_IDS + VOWEL_IDS
_MIN_SPAN = {"open": 0.035, "width": 0.025, "inner": 0.02, "corner": 0.035}
_MIN_SIGMA = 0.16
_AXIS_W = {
    "corners": {"corner": 1.0, "width": 0.25, "open": 0.10},
    "spread": {"width": 1.0, "open": 0.80, "inner": 0.45, "corner": 0.22},
    "round": {"width": 1.0, "open": 0.80, "inner": 0.55, "corner": 0.18},
    "open": {"open": 1.0, "width": 0.15, "corner": 0.05},
}
_HINTS = {
    "rest": "Keep your mouth closed until Rest finishes",
    "smile": "Hold a smile until capture finishes",
    "sad": "Hold a frown until capture finishes",
    "A": "Hold 'ah' until capture finishes",
    "I": "Hold 'ee' until capture finishes",
    "U": "Hold 'oo' until capture finishes",
    "E": "Hold 'eh' until capture finishes",
}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else float(value)


def _sample(feat: dict[str, float]) -> dict[str, float]:
    out = {
        "open": round(float(feat["open"]), 5),
        "width": round(float(feat["width"]), 5),
        "lift": round(float(feat.get("lift", 0.0)), 5),
    }
    for key in ("corner", "inner"):
        if key in feat:
            try:
                out[key] = round(float(feat[key]), 5)
            except (TypeError, ValueError):
                continue
    return out


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    n = float(len(rows))
    out = {
        "open": sum(row["open"] for row in rows) / n,
        "width": sum(row["width"] for row in rows) / n,
        "lift": sum(float(row.get("lift", 0.0)) for row in rows) / n,
    }
    for key in ("corner", "inner"):
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[key] = sum(vals) / float(len(vals))
    return out


def _val(row: dict[str, float], key: str, rest: dict[str, float]) -> float:
    if key in row:
        return float(row[key])
    if key in rest:
        return float(rest[key])
    return 0.0


def _delta(row: dict[str, float], rest: dict[str, float], key: str) -> float:
    return _val(row, key, rest) - _val(rest, key, rest)


def _axes(bank_id: str, rest: dict[str, float], rows: list[dict[str, float]]) -> tuple[str, ...]:
    weights = _AXIS_W[bank_id]
    keys = []
    for key in weights:
        if key in rest or any(key in row for row in rows):
            keys.append(key)
    return tuple(keys)


def _spans(
    rest: dict[str, float],
    rows: list[dict[str, float]],
    axes: tuple[str, ...],
) -> dict[str, float]:
    out = {}
    for key in axes:
        diffs = [abs(_delta(row, rest, key)) for row in rows]
        floor = _MIN_SPAN.get(key, 0.03)
        out[key] = max(max(diffs) if diffs else 0.0, floor)
    return out


def _vec(
    row: dict[str, float],
    rest: dict[str, float],
    axes: tuple[str, ...],
    spans: dict[str, float],
    bank_id: str,
) -> tuple[float, ...]:
    weights = _AXIS_W[bank_id]
    return tuple(weights[key] * _delta(row, rest, key) / spans[key] for key in axes)


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _open_gate(feat: dict[str, float], rest: dict[str, float], bank_id: str) -> float:
    if bank_id not in ("spread", "round"):
        return 1.0
    delta = float(feat["open"]) - float(rest["open"])
    return _clip((delta - 0.012) / 0.055)


def _match_bank(
    feat: dict[str, float],
    rest: dict[str, float],
    captured: dict[str, dict[str, float]],
    names: tuple[str, ...],
    bank_id: str,
) -> dict[str, float]:
    rows = list(captured.values())
    axes = _axes(bank_id, rest, rows)
    if not axes:
        return {name: 0.0 for name in names}
    spans = _spans(rest, rows, axes)
    live = _vec(feat, rest, axes, spans, bank_id)
    protos = {"rest": _vec(rest, rest, axes, spans, bank_id)}
    for name, row in captured.items():
        protos[name] = _vec(row, rest, axes, spans, bank_id)
    dists = {name: _dist(live, proto) for name, proto in protos.items()}
    rest_pt = protos["rest"]
    radii = [_dist(rest_pt, protos[name]) for name in captured]
    sigma = max(_MIN_SIGMA, 0.32 * (sorted(radii)[len(radii) // 2] if radii else 0.4))
    floor = min(dists.values())
    exps = {name: math.exp(-(dist - floor) / sigma) for name, dist in dists.items()}
    total = sum(exps.values()) or 1.0
    soft = {name: value / total for name, value in exps.items()}
    rest_w = soft.get("rest", 0.0)
    scale = 1.0 if rest_w < 0.50 else _clip(1.0 - (rest_w - 0.50) / 0.50)
    scale *= _open_gate(feat, rest, bank_id)
    return {name: _clip(soft.get(name, 0.0) * scale) for name in names}


def _align_feat(
    feat: dict[str, float],
    disk_rest: dict[str, float],
    origin: dict[str, float],
) -> dict[str, float]:
    """Map this session's live mouth into the saved Rest frame.

    A new OSF run has a new 3D zero. Matching saved poses against that raw
    face looks like movement even when the mouth is still.
    """
    out = dict(feat)
    live_w = float(feat.get("width") or 0.0)
    origin_w = float(origin.get("width") or 0.0)
    disk_w = float(disk_rest.get("width") or 0.0)
    if live_w > 1e-5 and origin_w > 1e-5 and disk_w > 1e-5:
        out["width"] = disk_w * (live_w / origin_w)
    for key in ("open", "corner", "lift", "inner"):
        if key not in feat or key not in disk_rest or key not in origin:
            continue
        out[key] = float(disk_rest[key]) + float(feat[key]) - float(origin[key])
    return out


class Calibrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.samples: dict[str, dict[str, float]] = {}
        self.capturing = ""
        self._started = 0.0
        self._buf: list[dict[str, float]] = []
        self._error = ""
        self._load()

    def _load(self) -> None:
        if not CALIB_PATH.is_file():
            return
        try:
            data = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        raw = data.get("samples") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return
        for name, row in raw.items():
            if name not in CAPTURE_IDS or not isinstance(row, dict):
                continue
            try:
                self.samples[name] = _sample(
                    {
                        "open": float(row["open"]),
                        "width": float(row["width"]),
                        "lift": float(row.get("lift", 0.0)),
                        **({k: float(row[k]) for k in ("corner", "inner") if k in row}),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

    def save(self) -> None:
        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "samples": {name: self.samples[name] for name in CAPTURE_IDS if name in self.samples}
        }
        CALIB_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear(self) -> None:
        with self._lock:
            self.samples = {}
            self.capturing = ""
            self._buf = []
            self._error = ""
        if CALIB_PATH.is_file():
            try:
                CALIB_PATH.unlink()
            except OSError:
                pass

    def drop_poses(self, names: object = None) -> None:
        """Remove captured poses so those rows go back to auto. Rest stays."""
        wanted: set[str] | None
        if names is None:
            wanted = None
        elif isinstance(names, str):
            wanted = {names}
        elif isinstance(names, (list, tuple, set)):
            wanted = {str(name) for name in names}
        else:
            raise ValueError("Unknown poses to clear")
        with self._lock:
            if wanted is None:
                self.samples = {k: v for k, v in self.samples.items() if k == "rest"}
            else:
                for name in wanted:
                    if name == "rest":
                        continue
                    self.samples.pop(name, None)
            self.capturing = ""
            self._buf = []
            self._error = ""
        self.save()

    def start(self, name: str) -> None:
        if name not in CAPTURE_IDS:
            raise ValueError(f"Unknown capture pose: {name}")
        with self._lock:
            if self.capturing:
                if self._progress_unlocked() < 1.0:
                    raise ValueError("Already capturing")
                self.capturing = ""
                self._buf = []
            if name != "rest" and "rest" not in self.samples:
                raise ValueError("Set Rest first, then capture this pose")
            self.capturing = name
            self._started = time.perf_counter()
            self._buf = []
            self._error = ""

    def rest_snapshot(self) -> dict[str, float] | None:
        with self._lock:
            row = self.samples.get("rest")
            return dict(row) if row else None

    def ingest(self, feat: dict[str, float] | None) -> str:
        finished = ""
        with self._lock:
            name = self.capturing
            if not name:
                return ""
            now = time.perf_counter()
            elapsed = now - self._started
            if feat is not None and elapsed >= WARMUP_SEC:
                self._buf.append(_sample(feat))
            if elapsed < CAPTURE_SEC:
                return ""
            rows = self._buf
            self.capturing = ""
            self._buf = []
            if len(rows) < 4:
                self._error = "No face while calibrating — hold the pose and try again"
                return ""
            self.samples[name] = _sample(_mean(rows))
            self._error = ""
            finished = name
        self.save()
        return finished

    def _progress_unlocked(self) -> float:
        if not self.capturing:
            return 0.0
        return _clip((time.perf_counter() - self._started) / CAPTURE_SEC)

    def payload(self) -> dict[str, object]:
        with self._lock:
            capturing = self.capturing
            return {
                "rest": "rest" in self.samples,
                "smile": "smile" in self.samples,
                "sad": "sad" in self.samples,
                "vowels": {name: name in self.samples for name in VOWEL_IDS},
                "capturing": capturing,
                "progress": round(self._progress_unlocked(), 3),
                "error": self._error,
                "hint": _HINTS.get(capturing, ""),
                "banks": [
                    {
                        "id": bank_id,
                        "label": label,
                        "shapes": [
                            {
                                "id": name,
                                "label": PRESET_LABELS[name],
                                "ready": name in self.samples,
                            }
                            for name in names
                        ],
                        "ready": any(name in self.samples for name in names),
                    }
                    for bank_id, label, names in MOUTH_BANKS
                ],
            }

    def weights(
        self,
        feat: dict[str, float] | None,
        fallback,
        origin: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if feat is None:
            return empty_weights()
        mixed = fallback(feat)
        with self._lock:
            rest = self.samples.get("rest")
            copied = {name: dict(row) for name, row in self.samples.items()}
        if rest is None:
            return mixed
        # Saved poses are last session's 3D. Hold them until this camera
        # run has its own closed-mouth zero, then compare in that frame.
        if origin is None:
            return mixed
        live = _align_feat(feat, rest, origin)
        for bank_id, _label, names in MOUTH_BANKS:
            captured = {name: copied[name] for name in names if name in copied}
            if not captured:
                continue
            part = _match_bank(live, rest, captured, names, bank_id)
            for name in names:
                mixed[name] = part[name]
        total = sum(mixed[name] for name in VOWEL_IDS)
        if total > 1.0:
            for name in VOWEL_IDS:
                mixed[name] /= total
        return mixed


calibrator = Calibrator()
