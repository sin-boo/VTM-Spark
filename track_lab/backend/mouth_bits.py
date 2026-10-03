"""Which OSF mouth landmarks are drawn, tracked, and wired to the mesh."""

from __future__ import annotations

import json
import threading

from .paths import OUTPUT

PATH = OUTPUT / "mouth_bits.json"

MOUTH_ALL = tuple(range(48, 66))
ANIME_SLOTS = tuple(range(28))
# 8 camera lips 58-65 → 8 anime mouth slots 20-27.
DEFAULT_ON = frozenset((58, 59, 60, 61, 62, 63, 64, 65))
DEFAULT_TO = {
    58: 23,
    59: 20,
    60: 21,
    61: 22,
    62: 26,
    63: 27,
    64: 25,
    65: 24,
}
# Viseme L / upper / R / lower = anime 23 / 21 / 26 / 25.
ROLE_ANIME = (23, 21, 26, 25)
DEFAULT_ROLES = (58, 60, 62, 64)
ROLE_FALLBACK = (
    (58, 59, 48),
    (60, 61, 59, 50, 51),
    (62, 63, 61, 54),
    (64, 65, 63, 57, 55),
)


class MouthBits:
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
            nxt = {int(i) for i in raw if int(i) in MOUTH_ALL}
            if nxt:
                self._on = nxt
        dest = data.get("to")
        if isinstance(dest, dict):
            mapped: dict[int, int] = {}
            for key, value in dest.items():
                try:
                    src = int(key)
                    slot = int(value)
                except (TypeError, ValueError):
                    continue
                if src in MOUTH_ALL and slot in ANIME_SLOTS:
                    mapped[src] = slot
            self._to = mapped

    def save(self) -> None:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            on = sorted(self._on)
            to = {str(k): int(v) for k, v in sorted(self._to.items())}
        PATH.write_text(json.dumps({"on": on, "to": to}, indent=2), encoding="utf-8")

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

    def roles(self) -> tuple[int, int, int, int] | None:
        with self._lock:
            on = set(self._on)
            to = dict(self._to)
        picks: list[int] = []
        for anime, cands in zip(ROLE_ANIME, ROLE_FALLBACK):
            mapped = next((src for src, dest in to.items() if dest == anime and src in on), None)
            if mapped is not None:
                picks.append(mapped)
                continue
            hit = next((i for i in cands if i in on), None)
            if hit is None:
                return None
            picks.append(hit)
        return picks[0], picks[1], picks[2], picks[3]

    def set_on(self, index: int, enabled: bool) -> dict[str, object]:
        idx = int(index)
        if idx not in MOUTH_ALL:
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
        if idx not in MOUTH_ALL:
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
            "mouth_points": [
                {
                    "id": i,
                    "on": i in on,
                    "ring": "in" if i >= 60 else "out",
                    "to": to.get(i),
                }
                for i in MOUTH_ALL
            ]
        }


bits = MouthBits()
