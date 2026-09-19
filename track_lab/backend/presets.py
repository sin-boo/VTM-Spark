"""Mouth shape presets authored on top of the anime rest mesh."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRESET_PATH = ROOT / "output" / "mouth_presets.json"

MOUTH_SLOTS = tuple(range(20, 28))
VOWEL_IDS = ("A", "I", "U", "E", "O")
FORM_IDS = ("smile", "sad")
PRESET_IDS = ("rest",) + FORM_IDS + VOWEL_IDS
PRESET_LABELS = {
    "rest": "Rest",
    "smile": "Smile",
    "sad": "Sad",
    "A": "A",
    "I": "I",
    "U": "U",
    "E": "E",
    "O": "O",
}
# Independent tracking setups. Each bank has its own geometry scale, then
# all banks mix. Corners (smile/sad) stay coarse; spread/round (I E / U O)
# can use the smaller lip differences without being swamped by Oh/Smile.
MOUTH_BANKS = (
    ("corners", "Corners", ("smile", "sad")),
    ("spread", "Spread", ("I", "E")),
    ("round", "Round", ("U", "O")),
    ("open", "Open", ("A",)),
)
_STALE_IDS = {"closed", "open", "mouth_open", "mouth_closed"}
# Leftover iPhone I/U at rest (~0.06–0.09) still mixed authored open
# visemes and dropped MOUTH.D. Ignore that noise.
_MIX_DEAD = 0.10


def pts_to_json(pts: np.ndarray | None) -> list[list[float]]:
    if pts is None or len(pts) < 28:
        return []
    return [
        [round(float(row[0]), 3), round(float(row[1]), 3), round(float(row[2]), 4)]
        for row in np.asarray(pts, dtype=np.float32)[:28]
    ]


def json_to_pts(raw: object) -> np.ndarray | None:
    if not isinstance(raw, list) or len(raw) < 28:
        return None
    out = np.zeros((28, 3), dtype=np.float32)
    for i, row in enumerate(raw[:28]):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        out[i, 0] = float(row[0])
        out[i, 1] = float(row[1])
        out[i, 2] = float(row[2]) if len(row) > 2 else 1.0
    return out


def copy_pts(pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=np.float32)[:28].copy()


def apply_mouth(base: np.ndarray, mouth: object) -> np.ndarray:
    out = copy_pts(base)
    if not isinstance(mouth, dict):
        return out
    for key, value in mouth.items():
        try:
            slot = int(key)
        except (TypeError, ValueError):
            continue
        if slot not in MOUTH_SLOTS:
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        out[slot, 0] = float(value[0])
        out[slot, 1] = float(value[1])
        if len(value) > 2:
            out[slot, 2] = float(value[2])
        elif out[slot, 2] < 0.05:
            out[slot, 2] = 1.0
    return out


# Unsaved vowel / smile / sad start from rest. Units are fractions of mouth width.
# open = extra lip gap (image down). spread = corners out. lift = corners up.
_DRAFT = {
    "smile": {"open": 0.04, "spread": 0.10, "lift": 0.14},
    "sad": {"open": 0.05, "spread": 0.02, "lift": -0.14},
    "A": {"open": 0.50, "spread": 0.04, "lift": 0.0},
    "I": {"open": 0.07, "spread": 0.20, "lift": 0.04},
    "U": {"open": 0.12, "spread": -0.16, "lift": 0.0},
    "E": {"open": 0.12, "spread": 0.16, "lift": 0.02},
    "O": {"open": 0.24, "spread": -0.12, "lift": 0.0},
}
_OPEN_FRAC = 0.46
_OPEN_UP = ((20, 0.45), (21, 0.50), (22, 0.45))
_OPEN_DOWN = ((24, 0.55), (25, 0.62), (27, 0.55))


def _mouth_frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    right = np.asarray(pts[23, :2], dtype=np.float32)
    left = np.asarray(pts[26, :2], dtype=np.float32)
    across = left - right
    width = float(np.hypot(across[0], across[1]))
    if width < 1e-3:
        return np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32), 0.0
    along = across / width
    down = np.array([-along[1], along[0]], dtype=np.float32)
    if float(down[1]) < 0.0:
        down = -down
    return along, down, width


def _mouth_down(pts: np.ndarray) -> tuple[np.ndarray, float]:
    along, down, width = _mouth_frame(pts)
    del along
    return down, width


def lip_gap(pts: np.ndarray | None) -> float:
    if pts is None or len(pts) <= 25:
        return 0.0
    return abs(float(pts[25, 1] - pts[21, 1]))


def open_amount(weights: dict[str, float] | None) -> float:
    """How far the live mouth should split. A is the jaw; O/E still open."""
    if not weights:
        return 0.0
    a = float(np.clip(weights.get("A") or 0.0, 0.0, 1.0))
    o = float(np.clip(weights.get("O") or 0.0, 0.0, 1.0))
    e = float(np.clip(weights.get("E") or 0.0, 0.0, 1.0))
    return float(np.clip(a + 0.45 * o + 0.20 * e, 0.0, 1.0))


def apply_open_offset(
    mesh: np.ndarray | None,
    rest: np.ndarray | None,
    amount: float,
) -> np.ndarray | None:
    """Push U up and D down so an open mouth cannot stay a rest slit.

    Adds only the missing gap, so an authored A that already opened the
    lips is left alone.
    """
    if mesh is None or rest is None or len(mesh) <= 27 or len(rest) <= 27:
        return mesh
    amt = float(np.clip(amount, 0.0, 1.0))
    if amt < 0.08:
        return mesh
    down, width = _mouth_down(rest)
    if width < 1e-3:
        return mesh
    want = amt * _OPEN_FRAC * width
    have = max(0.0, lip_gap(mesh) - lip_gap(rest))
    missing = want - have
    if missing <= 0.75:
        return mesh
    for slot, weight in _OPEN_UP:
        mesh[slot, :2] -= down * (missing * weight)
    for slot, weight in _OPEN_DOWN:
        mesh[slot, :2] += down * (missing * weight)
    return mesh


def draft_mouth(name: str, rest: np.ndarray) -> np.ndarray:
    """Move rest lips into a starting A / I / U / E / O / smile / sad pose."""
    out = copy_pts(rest)
    spec = _DRAFT.get(name)
    if spec is None or len(out) < 28:
        return out
    down, width = _mouth_down(out)
    if width < 1e-3:
        return out
    right = out[23, :2]
    left = out[26, :2]
    along = (left - right) / width
    open_px = width * float(spec["open"])
    spread_px = width * float(spec["spread"])
    lift_px = width * float(spec["lift"])
    out[20, :2] -= down * (open_px * 0.45)
    out[21, :2] -= down * (open_px * 0.50)
    out[22, :2] -= down * (open_px * 0.45)
    out[24, :2] += down * (open_px * 0.55)
    out[25, :2] += down * (open_px * 0.62)
    out[27, :2] += down * (open_px * 0.55)
    out[23, :2] += -along * spread_px - down * lift_px
    out[26, :2] += along * spread_px - down * lift_px
    return out


def retarget_mouth(shape: np.ndarray, old_rest: np.ndarray, new_rest: np.ndarray) -> np.ndarray:
    """Move authored lips onto a new rest. Eyes/jaw stay on the new face."""
    out = copy_pts(new_rest)
    src = copy_pts(shape)
    prev = copy_pts(old_rest)
    nxt = copy_pts(new_rest)
    old_along, old_down, old_w = _mouth_frame(prev)
    new_along, new_down, new_w = _mouth_frame(nxt)
    if old_w < 1e-3 or new_w < 1e-3:
        return out
    scale = new_w / old_w
    for slot in MOUTH_SLOTS:
        delta = src[slot, :2] - prev[slot, :2]
        along_amt = float(np.dot(delta, old_along))
        down_amt = float(np.dot(delta, old_down))
        out[slot, :2] = nxt[slot, :2] + scale * (along_amt * new_along + down_amt * new_down)
        if src.shape[1] > 2:
            out[slot, 2] = float(src[slot, 2])
    return out


def empty_weights() -> dict[str, float]:
    out = {name: 0.0 for name in VOWEL_IDS}
    for name in FORM_IDS:
        out[name] = 0.0
    return out


class MouthBook:
    def __init__(self) -> None:
        self.shapes: dict[str, np.ndarray] = {}
        self.active: str = ""
        self._load()

    def _load(self) -> None:
        if not PRESET_PATH.is_file():
            return
        try:
            data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        stale = False
        shapes = data.get("shapes")
        if isinstance(shapes, dict):
            for name, raw in shapes.items():
                if name in _STALE_IDS or name not in PRESET_IDS:
                    stale = True
                    continue
                pts = json_to_pts(raw)
                if pts is not None:
                    self.shapes[name] = pts
        active = data.get("active")
        if isinstance(active, str) and active in self.shapes:
            self.active = active
        elif self.active not in self.shapes:
            self.active = "rest" if "rest" in self.shapes else ""
            stale = True
        if stale:
            self.save()

    def save(self) -> None:
        PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active": self.active if self.active in self.shapes else "",
            "shapes": {name: pts_to_json(self.shapes[name]) for name in PRESET_IDS if name in self.shapes},
        }
        PRESET_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.shapes = {}
        self.active = ""
        if PRESET_PATH.is_file():
            try:
                PRESET_PATH.unlink()
            except OSError:
                pass

    def seed_rest(self, pts: np.ndarray) -> None:
        self.shapes["rest"] = copy_pts(pts)
        self.active = "rest"
        self.save()

    def rebase(self, pts: np.ndarray) -> None:
        """Keep visemes when rest jumps; only the lips move onto the new face."""
        old = self.shapes.get("rest")
        nxt = copy_pts(pts)
        if old is not None:
            for name in FORM_IDS + VOWEL_IDS:
                stored = self.shapes.get(name)
                if stored is None:
                    continue
                self.shapes[name] = retarget_mouth(stored, old, nxt)
        self.shapes["rest"] = nxt
        self.active = "rest"
        self.save()

    def template(self, rest: np.ndarray | None) -> np.ndarray | None:
        if "rest" in self.shapes:
            return self.shapes["rest"]
        return rest

    def preview(self, name: str, rest: np.ndarray | None) -> np.ndarray:
        """Saved shape, or a drafted viseme from rest. Does not write."""
        if name not in PRESET_IDS:
            raise ValueError(f"Unknown mouth preset: {name}")
        stored = self.shapes.get(name)
        if stored is not None:
            return copy_pts(stored)
        source = self.template(rest)
        if source is None:
            raise ValueError("Track a face first")
        if name == "rest":
            return copy_pts(source)
        return draft_mouth(name, source)

    def apply(self, name: str, rest: np.ndarray | None) -> np.ndarray:
        if name not in PRESET_IDS:
            raise ValueError(f"Unknown mouth preset: {name}")
        stored = self.shapes.get(name)
        if stored is None:
            if name != "rest":
                raise ValueError(f"No saved {PRESET_LABELS[name]} yet")
            source = self.template(rest)
            if source is None:
                raise ValueError("Track a face first")
            self.shapes["rest"] = copy_pts(source)
            stored = self.shapes["rest"]
        self.active = name
        self.save()
        return copy_pts(stored)

    def set_mouth(self, name: str, mouth: object, rest: np.ndarray | None) -> np.ndarray:
        if name not in PRESET_IDS:
            raise ValueError(f"Unknown mouth preset: {name}")
        base = self.shapes.get(name)
        if base is None:
            source = self.template(rest)
            if source is None:
                raise ValueError("Track a face first")
            base = source
        pts = apply_mouth(base, mouth)
        self.shapes[name] = pts
        self.active = name
        self.save()
        return copy_pts(pts)

    def has_visemes(self) -> bool:
        return any(name in self.shapes for name in FORM_IDS + VOWEL_IDS)

    def mix(self, weights: dict[str, float] | None) -> np.ndarray | None:
        rest = self.shapes.get("rest")
        if rest is None:
            return None
        out = copy_pts(rest)
        if not weights:
            return out
        slots = list(MOUTH_SLOTS)
        vowels = {
            name: float(np.clip(weights.get(name, 0.0), 0.0, 1.0))
            for name in VOWEL_IDS
        }
        total = sum(vowels.values())
        if total > 1.0:
            vowels = {name: amount / total for name, amount in vowels.items()}
        amounts = {
            "smile": float(np.clip(weights.get("smile", 0.0), 0.0, 0.7)),
            "sad": float(np.clip(weights.get("sad", 0.0), 0.0, 0.7)),
            **vowels,
        }
        if amounts["smile"] >= amounts["sad"]:
            amounts["sad"] = 0.0
        else:
            amounts["smile"] = 0.0
        for name in FORM_IDS + VOWEL_IDS:
            amount = amounts[name]
            if amount <= _MIX_DEAD:
                continue
            amount = (amount - _MIX_DEAD) / (1.0 - _MIX_DEAD)
            shape = self.shapes.get(name)
            if shape is None:
                # Anime rest is often a small open circle. Draft from that
                # shape instead of skipping, so live A does not wait for a
                # saved viseme or replace rest with a camera slit.
                shape = draft_mouth(name, rest)
            out[slots, :2] += amount * (shape[slots, :2] - rest[slots, :2])
        amt = open_amount(weights)
        apply_open_offset(out, rest, amt)
        return out

    def current(self) -> np.ndarray | None:
        if self.active and self.active in self.shapes:
            return copy_pts(self.shapes[self.active])
        return None

    def payload(self, rest: np.ndarray | None) -> dict[str, object]:
        return {
            "active": self.active,
            "mouth_slots": list(MOUTH_SLOTS),
            "presets": [
                {
                    "id": name,
                    "label": PRESET_LABELS[name],
                    "ready": name in self.shapes,
                }
                for name in PRESET_IDS
            ],
            "points": pts_to_json(self.current() if self.active else rest),
            "shapes": {name: pts_to_json(self.shapes[name]) for name in PRESET_IDS if name in self.shapes},
        }


book = MouthBook()
