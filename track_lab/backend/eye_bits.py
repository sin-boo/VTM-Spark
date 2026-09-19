"""Which OSF eye landmarks are drawn, tracked, and wired to the mesh."""

from __future__ import annotations

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "output" / "eye_bits.json"

# Artificial lid mids: 66 = (37+38)/2 (OSF "right" = image-left eye),
# 67 = (43+44)/2 (image-right eye).
LID_MID_R = 66
LID_MID_L = 67
ARTIFICIAL = frozenset((LID_MID_R, LID_MID_L))
EYE_ALL = tuple(range(36, 48)) + (LID_MID_R, LID_MID_L)
ANIME_SLOTS = tuple(range(28))
DEFAULT_ON = frozenset((36, 39, LID_MID_R, 42, 45, LID_MID_L))
# Canonical frame (sides.py): image-left camera eye 36-41 → screen-left
# slots 11-13. Mirror OFF swaps this at apply time, not here.
DEFAULT_TO = {
    36: 11,
    39: 13,
    LID_MID_R: 12,
    42: 17,
    45: 19,
    LID_MID_L: 18,
}
# Bump when DEFAULT_TO changes sides so a saved map from the old
# convention is dropped instead of silently crossing the eyes.
SIDES_VERSION = 2


class EyeBits:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._on = set(DEFAULT_ON)
        self._to = dict(DEFAULT_TO)
        self._load()

    def _load(self) -> None:
        if not PATH.is_file():
            return
        try:
            data = json.loads(PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        raw = data.get("on")
        if isinstance(raw, list):
            nxt = {int(i) for i in raw if int(i) in EYE_ALL}
            if nxt:
                self._on = nxt
        dest = data.get("to")
        if int(data.get("sides") or 1) < SIDES_VERSION:
            # Old file: eyes were mapped anatomically. Keep defaults.
            dest = None
        if isinstance(dest, dict):
            mapped: dict[int, int] = {}
            for key, value in dest.items():
                try:
                    src = int(key)
                    slot = int(value)
                except (TypeError, ValueError):
                    continue
                if src in EYE_ALL and slot in ANIME_SLOTS:
                    mapped[src] = slot
            self._to = mapped
        if LID_MID_L not in self._on:
            self._on.discard(44)
            self._on.update({42, 45, LID_MID_L})
            self._to.setdefault(42, DEFAULT_TO[42])
            self._to.setdefault(45, DEFAULT_TO[45])
            self._to.setdefault(LID_MID_L, DEFAULT_TO[LID_MID_L])
            self._to.pop(44, None)

    def save(self) -> None:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            on = sorted(self._on)
            to = {str(k): int(v) for k, v in sorted(self._to.items())}
        PATH.write_text(
            json.dumps({"on": on, "to": to, "sides": SIDES_VERSION}, indent=2),
            encoding="utf-8",
        )

    def on(self, index: int) -> bool:
        with self._lock:
            return int(index) in self._on

    def showing(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._on)

    def maps(self) -> dict[int, int]:
        with self._lock:
            return dict(self._to)

    def snapshot(self) -> tuple[frozenset[int], dict[int, int]]:
        with self._lock:
            return frozenset(self._on), dict(self._to)

    def restore(self, on: frozenset[int], to: dict[int, int]) -> None:
        with self._lock:
            self._on = set(on)
            self._to = dict(to)
        self.save()

    def set_on(self, index: int, enabled: bool) -> dict[str, object]:
        idx = int(index)
        if idx not in EYE_ALL:
            return self.payload()
        with self._lock:
            if enabled:
                self._on.add(idx)
            else:
                self._on.discard(idx)
        self.save()
        return self.payload()

    def set_to(self, index: int, slot: int | None) -> dict[str, object]:
        idx = int(index)
        if idx not in EYE_ALL:
            return self.payload()
        with self._lock:
            if slot is None:
                self._to.pop(idx, None)
            else:
                dest = int(slot)
                if dest not in ANIME_SLOTS:
                    return self.payload()
                self._to[idx] = dest
        self.save()
        return self.payload()

    def payload(self) -> dict[str, object]:
        with self._lock:
            on = set(self._on)
            to = dict(self._to)
        return {
            "eye_points": [
                {
                    "id": i,
                    "on": i in on,
                    "side": "l" if i == LID_MID_L or 42 <= i <= 47 else "r",
                    "artificial": i in ARTIFICIAL,
                    "to": to.get(i),
                }
                for i in EYE_ALL
            ]
        }


bits = EyeBits()
