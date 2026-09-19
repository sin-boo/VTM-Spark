"""Small, stable set of live mouth controls."""

from __future__ import annotations

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEEL_PATH = ROOT / "output" / "tracking_feel.json"

DEFAULTS = {
    "response": 0.65,
    "smoothing": 0.48,
    "mouth": 0.50,
    "use_visemes": 1.0,
    "show_face": 1.0,
    "show_skeleton": 1.0,
    "show_hair": 1.0,
    "show_ids": 0.0,
    "hair_pin": 0.7,
    "max_yaw": 1.0,
    "max_roll": 1.0,
    "max_pitch_up": 1.0,
    "max_pitch_down": 1.0,
    "max_look_x": 1.0,
    "max_look_y": 1.0,
    "gaze_gain": 1.0,
    "gaze_smooth": 0.28,
}
_LIMITS = {key: 1.0 for key in DEFAULTS}
_LIMITS["mouth"] = 2.0
_LIMITS["gaze_gain"] = 2.0


def _clip(value: float, hi: float) -> float:
    return 0.0 if value < 0.0 else hi if value > hi else float(value)


class Feel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.values = dict(DEFAULTS)
        self._load()

    def _load(self) -> None:
        if not FEEL_PATH.is_file():
            return
        try:
            data = json.loads(FEEL_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        for key in DEFAULTS:
            if key in data:
                try:
                    self.values[key] = _clip(float(data[key]), _LIMITS[key])
                except (TypeError, ValueError):
                    continue

    def save(self) -> None:
        FEEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEEL_PATH.write_text(json.dumps(self.values, indent=2), encoding="utf-8")

    def payload(self) -> dict[str, float | bool]:
        with self._lock:
            return {key: round(float(self.values[key]), 3) for key in DEFAULTS}

    def update(self, body: object) -> dict[str, float | bool]:
        if not isinstance(body, dict):
            return self.payload()
        with self._lock:
            for key in DEFAULTS:
                if key not in body:
                    continue
                try:
                    self.values[key] = _clip(float(body[key]), _LIMITS[key])
                except (TypeError, ValueError):
                    continue
        self.save()
        return self.payload()

    def _get(self, key: str) -> float:
        with self._lock:
            return float(self.values.get(key, DEFAULTS[key]))

    def alpha(self) -> float:
        return 0.72 - 0.56 * self._get("smoothing")

    def response(self) -> float:
        return 0.75 + 0.60 * self._get("response")

    def mouth_gain(self) -> float:
        """Overall mouth size. 0.5 = authored presets, 1.0 = 2×, 2.0 = 4×."""
        t = self._get("mouth")
        if t <= 0.5:
            return 0.35 + 1.30 * t
        return 1.0 + 2.0 * (t - 0.5)

    def use_visemes(self) -> bool:
        return self._get("use_visemes") >= 0.5

    def show_face(self) -> bool:
        return self._get("show_face") >= 0.5

    def show_skeleton(self) -> bool:
        return self._get("show_skeleton") >= 0.5

    def show_hair(self) -> bool:
        return self._get("show_hair") >= 0.5

    def show_ids(self) -> bool:
        return self._get("show_ids") >= 0.5

    def hair_pin(self) -> float:
        """0 = full 2.5D hair follow, 1 = outer silhouette stays put."""
        return self._get("hair_pin")

    def max_yaw(self) -> float:
        return self._get("max_yaw")

    def max_roll(self) -> float:
        return self._get("max_roll")

    def max_pitch_up(self) -> float:
        return self._get("max_pitch_up")

    def max_pitch_down(self) -> float:
        return self._get("max_pitch_down")

    def max_look_x(self) -> float:
        return self._get("max_look_x")

    def max_look_y(self) -> float:
        return self._get("max_look_y")

    def gaze_gain(self) -> float:
        return self._get("gaze_gain")

    def gaze_alpha(self) -> float:
        return 0.72 - 0.56 * self._get("gaze_smooth")


feel = Feel()
