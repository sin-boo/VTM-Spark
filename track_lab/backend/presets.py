"""Mouth and eye shape presets authored on top of the anime rest mesh."""

from __future__ import annotations

import json

import numpy as np

from .paths import OUTPUT

PRESET_PATH = OUTPUT / "mouth_presets.json"

MOUTH_SLOTS = tuple(range(20, 28))
# Character eyes by screen side: (corner, lid mid, corner). Blink "l" drives
# 11-13, "r" drives 17-19.
EYE_L = (11, 12, 13)
EYE_R = (17, 18, 19)
EYE_SLOTS = EYE_L + EYE_R
VOWEL_IDS = ("A", "I", "U", "E")
FORM_IDS = ("smile", "sad")
# Blink blends Eye open -> Eye closed; they own only the eye points.
EYE_IDS = ("eye_open", "eye_closed")
PRESET_IDS = ("rest",) + FORM_IDS + VOWEL_IDS + EYE_IDS
# Every shape in PRESET_IDS is authored. O was folded into U (rounded lips).
AUTHOR_IDS = PRESET_IDS
PRESET_LABELS = {
    "rest": "Rest",
    "smile": "Smile",
    "sad": "Sad",
    "A": "A",
    "I": "I",
    "U": "U",
    "E": "E",
    "eye_open": "Eye open",
    "eye_closed": "Eye closed",
}


def pair_ends(name: str) -> tuple[str, str] | None:
    """Canonical ``a+b`` id, with ``a`` before ``b`` in ``PRESET_IDS``.

    A stop along that pair is ``a+b@NNN`` (thousandths, 1–999). ``a+b`` alone
    is the halfway stop.
    """
    if not isinstance(name, str):
        return None
    base, sep, suffix = name.partition("@")
    if sep:
        if not suffix.isdigit():
            return None
        slot = int(suffix)
        if slot <= 0 or slot >= 1000:
            return None
    if base.count("+") != 1:
        return None
    left, right = base.split("+", 1)
    if left not in PRESET_IDS or right not in PRESET_IDS or left == right:
        return None
    if PRESET_IDS.index(left) > PRESET_IDS.index(right):
        return None
    if (left in EYE_IDS) != (right in EYE_IDS):
        return None
    return left, right


def shape_slots(name: str) -> tuple[int, ...]:
    """Points a shape (or a stop between two) owns: the eyes or the lips."""
    ends = pair_ends(name)
    head = ends[0] if ends else name
    return EYE_SLOTS if head in EYE_IDS else MOUTH_SLOTS


def pair_id(a: str, b: str) -> str:
    if a not in PRESET_IDS or b not in PRESET_IDS or a == b:
        raise ValueError("Pair needs two different shapes")
    if (a in EYE_IDS) != (b in EYE_IDS):
        raise ValueError("Pair eyes with eyes and mouths with mouths")
    if PRESET_IDS.index(a) > PRESET_IDS.index(b):
        a, b = b, a
    return f"{a}+{b}"


def key_t(name: str) -> float | None:
    """Blend position of a saved stop. Halfway when the id has no ``@``."""
    if pair_ends(name) is None:
        return None
    if "@" not in name:
        return 0.5
    return int(name.split("@", 1)[1]) / 1000.0


def key_id(a: str, b: str, t: float) -> str:
    """Stop id at ``t`` in (0, 1). ``0.5`` stays the legacy ``a+b`` id."""
    base = pair_id(a, b)
    slot = int(round(float(np.clip(t, 0.0, 1.0)) * 1000))
    slot = min(999, max(1, slot))
    if slot == 500:
        return base
    return f"{base}@{slot}"


def pair_label(name: str) -> str:
    ends = pair_ends(name)
    if ends is None:
        return PRESET_LABELS.get(name, name)
    return f"{PRESET_LABELS[ends[0]]} · {PRESET_LABELS[ends[1]]}"


def _stored_ids(shapes: dict[str, np.ndarray]) -> list[str]:
    names = [name for name in PRESET_IDS if name in shapes]
    names.extend(name for name in shapes if pair_ends(name) is not None)
    return names


def _blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    out = copy_pts(a)
    u = float(np.clip(t, 0.0, 1.0))
    out[:, :2] = (1.0 - u) * np.asarray(a[:28, :2], dtype=np.float32) + u * np.asarray(
        b[:28, :2], dtype=np.float32
    )
    return out


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _blend(a, b, 0.5)


def _along(
    start: np.ndarray,
    shape: np.ndarray,
    stops: list[tuple[float, np.ndarray]],
    amount: float,
    slots: tuple[int, ...] = MOUTH_SLOTS,
) -> np.ndarray:
    """``slots`` at ``amount`` through start, the saved stops, and the end."""
    slots = list(slots)
    knots_t = [0.0]
    knots_xy = [np.asarray(start[slots, :2], dtype=np.float32)]
    seen: set[int] = set()
    for t, mid in stops:
        slot = int(round(float(t) * 1000))
        if slot in seen or slot <= 0 or slot >= 1000:
            continue
        seen.add(slot)
        knots_t.append(float(t))
        knots_xy.append(np.asarray(mid[slots, :2], dtype=np.float32))
    knots_t.append(1.0)
    knots_xy.append(np.asarray(shape[slots, :2], dtype=np.float32))
    amt = float(np.clip(amount, 0.0, 1.0))
    for i in range(len(knots_t) - 1):
        t0 = knots_t[i]
        t1 = knots_t[i + 1]
        if amt <= t1 or i == len(knots_t) - 2:
            span = 0.0 if t1 <= t0 else (amt - t0) / (t1 - t0)
            span = float(np.clip(span, 0.0, 1.0))
            return knots_xy[i] + span * (knots_xy[i + 1] - knots_xy[i])
    return knots_xy[-1]


# Independent tracking setups. Each bank has its own geometry scale, then
# all banks mix. Corners (smile/sad) stay coarse; spread/round (I E / U)
# can use the smaller lip differences without being swamped by Smile.
MOUTH_BANKS = (
    ("corners", "Corners", ("smile", "sad")),
    ("spread", "Spread", ("I", "E")),
    ("round", "Round", ("U",)),
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


def apply_mouth(
    base: np.ndarray, mouth: object, slots: tuple[int, ...] = MOUTH_SLOTS
) -> np.ndarray:
    """Write absolute ``slots`` from ``mouth`` ({slot: [x, y, score?]})."""
    out = copy_pts(base)
    if not isinstance(mouth, dict):
        return out
    for key, value in mouth.items():
        try:
            slot = int(key)
        except (TypeError, ValueError):
            continue
        if slot not in slots:
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
}
_OPEN_FRAC = 0.46
_OPEN_UP = ((20, 0.45), (21, 0.50), (22, 0.45))
_OPEN_DOWN = ((24, 0.55), (25, 0.62), (27, 0.55))


def _mouth_frame(
    pts: np.ndarray, a: int = 23, b: int = 26
) -> tuple[np.ndarray, np.ndarray, float]:
    """(along a to b, down the screen, width) for a pair of corners."""
    right = np.asarray(pts[a, :2], dtype=np.float32)
    left = np.asarray(pts[b, :2], dtype=np.float32)
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
    """How far the live mouth should split. A is the jaw; U/E still open."""
    if not weights:
        return 0.0
    a = float(np.clip(weights.get("A") or 0.0, 0.0, 1.0))
    u = float(np.clip(weights.get("U") or 0.0, 0.0, 1.0))
    e = float(np.clip(weights.get("E") or 0.0, 0.0, 1.0))
    return float(np.clip(a + 0.20 * u + 0.20 * e, 0.0, 1.0))


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
    """Move rest lips into a starting A / I / U / E / smile / sad pose."""
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


def draft_eyes(rest: np.ndarray) -> np.ndarray:
    """Unsaved Eye closed: each lid mid on its corner line."""
    out = copy_pts(rest)
    for a, lid, b in (EYE_L, EYE_R):
        chord = out[b, :2] - out[a, :2]
        span = float(np.dot(chord, chord))
        if span < 1e-6:
            continue
        t = float(np.clip(np.dot(out[lid, :2] - out[a, :2], chord) / span, 0.0, 1.0))
        out[lid, :2] = out[a, :2] + t * chord
    return out


def draft_shape(name: str, rest: np.ndarray) -> np.ndarray:
    return draft_eyes(rest) if name == "eye_closed" else draft_mouth(name, rest)


# Each group moves in the frame of its two corners.
_GROUPS = {
    "mouth": ((MOUTH_SLOTS, 23, 26),),
    "eyes": ((EYE_L, 11, 13), (EYE_R, 17, 19)),
}


def retarget_mouth(
    shape: np.ndarray, old_rest: np.ndarray, new_rest: np.ndarray, group: str = "mouth"
) -> np.ndarray:
    """Move an authored shape onto a new rest. Other points stay on the new face."""
    out = copy_pts(new_rest)
    src = copy_pts(shape)
    prev = copy_pts(old_rest)
    nxt = copy_pts(new_rest)
    for slots, a, b in _GROUPS[group]:
        old_along, old_down, old_w = _mouth_frame(prev, a, b)
        new_along, new_down, new_w = _mouth_frame(nxt, a, b)
        if old_w < 1e-3 or new_w < 1e-3:
            continue
        scale = new_w / old_w
        for slot in slots:
            delta = src[slot, :2] - prev[slot, :2]
            along_amt = float(np.dot(delta, old_along))
            down_amt = float(np.dot(delta, old_down))
            out[slot, :2] = nxt[slot, :2] + scale * (along_amt * new_along + down_amt * new_down)
            if src.shape[1] > 2:
                out[slot, 2] = float(src[slot, 2])
    return out


# A rest this close is the same face (a re-push, or saved at another
# precision). Re-projecting onto it only adds float and save rounding.
_SAME_REST_PX = 0.01


def _same_rest(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return False
    left = np.asarray(a, dtype=np.float64)[:28]
    right = np.asarray(b, dtype=np.float64)[:28]
    if left.shape != right.shape:
        return False
    return bool(
        np.all(np.abs(left[:, :2] - right[:, :2]) <= _SAME_REST_PX)
        and np.all(np.abs(left[:, 2:] - right[:, 2:]) <= 1e-4)
    )


def empty_weights() -> dict[str, float]:
    out = {name: 0.0 for name in VOWEL_IDS}
    for name in FORM_IDS:
        out[name] = 0.0
    return out


class MouthBook:
    def __init__(self) -> None:
        self.shapes: dict[str, np.ndarray] = {}
        self.active: str = ""
        # The shapes as last authored, rest included. Every rebase starts
        # here: moving on from the last rebase drifted them a little on each
        # character switch until the desk read a new plan.
        self._authored: dict[str, np.ndarray] = {}
        self._load()

    def _anchor(self) -> None:
        """What the book holds now is the authored plan."""
        self._authored = {name: copy_pts(pts) for name, pts in self.shapes.items()}

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
                if name in _STALE_IDS or (name not in PRESET_IDS and pair_ends(name) is None):
                    stale = True
                    continue
                pts = json_to_pts(raw)
                if pts is not None:
                    self.shapes[name] = pts
        authored = data.get("authored")
        if isinstance(authored, dict):
            for name, raw in authored.items():
                pts = json_to_pts(raw) if name in self.shapes else None
                if pts is not None:
                    self._authored[name] = pts
        if set(self._authored) != set(self.shapes):
            # Older files keep no authored copy: the saved shapes are the plan.
            self._anchor()
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
            "shapes": {name: pts_to_json(self.shapes[name]) for name in _stored_ids(self.shapes)},
            "authored": {
                name: pts_to_json(self._authored[name]) for name in _stored_ids(self._authored)
            },
        }
        PRESET_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.shapes = {}
        self._authored = {}
        self.active = ""
        if PRESET_PATH.is_file():
            try:
                PRESET_PATH.unlink()
            except OSError:
                pass

    def seed_rest(self, pts: np.ndarray) -> None:
        self.shapes["rest"] = copy_pts(pts)
        self._anchor()
        self.active = "rest"
        self.save()

    def rebase(self, pts: np.ndarray) -> None:
        """Keep shapes when rest jumps; only their own points move onto the new face.

        Shapes move from how they were authored, never from the last rebase,
        so switching A -> B -> A lands exactly back on A's shapes.
        """
        old = self.shapes.get("rest")
        nxt = copy_pts(pts)
        if _same_rest(old, nxt):
            if self.active != "rest":
                self.active = "rest"
                self.save()
            return
        if set(self._authored) != set(self.shapes):
            self._anchor()
        home = self._authored.get("rest")
        if _same_rest(home, nxt):
            self.shapes = {name: copy_pts(self._authored[name]) for name in self.shapes}
        else:
            if home is not None:
                for name in list(self.shapes):
                    if name == "rest":
                        continue
                    group = "eyes" if shape_slots(name) == EYE_SLOTS else "mouth"
                    self.shapes[name] = retarget_mouth(self._authored[name], home, nxt, group)
            self.shapes["rest"] = nxt
            if home is None:
                self._anchor()
        self.active = "rest"
        self.save()

    def template(self, rest: np.ndarray | None) -> np.ndarray | None:
        if "rest" in self.shapes:
            return self.shapes["rest"]
        return rest

    def _ends(self, name: str, rest: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        ends = pair_ends(name)
        if ends is None:
            raise ValueError(f"Unknown shape: {name}")
        return self.preview(ends[0], rest), self.preview(ends[1], rest)

    def preview(self, name: str, rest: np.ndarray | None) -> np.ndarray:
        """Saved shape, or a drafted viseme from rest. Does not write."""
        if pair_ends(name) is not None:
            stored = self.shapes.get(name)
            if stored is not None:
                return copy_pts(stored)
            left, right = self._ends(name, rest)
            return _blend(left, right, key_t(name) or 0.5)
        if name not in PRESET_IDS:
            raise ValueError(f"Unknown shape: {name}")
        stored = self.shapes.get(name)
        if stored is not None:
            return copy_pts(stored)
        source = self.template(rest)
        if source is None:
            raise ValueError("Track a face first")
        if name == "rest":
            return copy_pts(source)
        return draft_shape(name, source)

    def apply(self, name: str, rest: np.ndarray | None) -> np.ndarray:
        if name not in PRESET_IDS and pair_ends(name) is None:
            raise ValueError(f"Unknown shape: {name}")
        stored = self.shapes.get(name)
        if stored is None:
            if name != "rest":
                label = pair_label(name)
                raise ValueError(f"No saved {label} yet")
            source = self.template(rest)
            if source is None:
                raise ValueError("Track a face first")
            self.shapes["rest"] = copy_pts(source)
            stored = self.shapes["rest"]
            self._anchor()
        self.active = name
        self.save()
        return copy_pts(stored)

    def set_mouth(self, name: str, mouth: object, rest: np.ndarray | None) -> np.ndarray:
        """Write ``name``'s own points (lips, or eyes for the eye shapes)."""
        if name not in PRESET_IDS and pair_ends(name) is None:
            raise ValueError(f"Unknown shape: {name}")
        base = self.shapes.get(name)
        if base is None:
            if pair_ends(name) is not None:
                left, right = self._ends(name, rest)
                base = _blend(left, right, key_t(name) or 0.5)
            else:
                source = self.template(rest)
                if source is None:
                    raise ValueError("Track a face first")
                base = draft_shape(name, source) if name == "eye_closed" else source
        pts = apply_mouth(base, mouth, shape_slots(name))
        self.shapes[name] = pts
        self._anchor()
        self.active = name
        self.save()
        return copy_pts(pts)

    def _stops(self, left: str, right: str) -> list[tuple[float, np.ndarray]]:
        want = (left, right)
        found: list[tuple[float, np.ndarray]] = []
        for name, shape in self.shapes.items():
            if pair_ends(name) != want:
                continue
            t = key_t(name)
            if t is None:
                continue
            found.append((t, shape))
        found.sort(key=lambda item: item[0])
        return found

    def move_key(self, name: str, t: object) -> str:
        """Slide a saved stop along its pair. The mouth shape stays put."""
        if pair_ends(name) is None or name not in self.shapes:
            raise ValueError("No saved point")
        if isinstance(t, bool) or not isinstance(t, (int, float, str)):
            raise ValueError("Point needs a position")
        try:
            pos = float(t)
        except ValueError:
            raise ValueError("Point needs a position") from None
        ends = pair_ends(name)
        if ends is None:
            raise ValueError("No saved point")
        new_name = key_id(ends[0], ends[1], pos)
        if new_name == name:
            return name
        if new_name in self.shapes:
            raise ValueError("A point is already there")
        self.shapes[new_name] = self.shapes.pop(name)
        if name in self._authored:
            self._authored[new_name] = self._authored.pop(name)
        if self.active == name:
            self.active = new_name
        self.save()
        return new_name

    def drop_key(self, name: str) -> None:
        """Remove a saved in-between. End shapes stay."""
        if pair_ends(name) is None or name not in self.shapes:
            raise ValueError("No saved point")
        del self.shapes[name]
        self._authored.pop(name, None)
        if self.active == name:
            self.active = "rest" if "rest" in self.shapes else ""
        self.save()

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
            stops = self._stops("rest", name)
            if not stops:
                out[slots, :2] += amount * (shape[slots, :2] - rest[slots, :2])
            else:
                out[slots, :2] += _along(rest, shape, stops, amount) - rest[slots, :2]
        amt = open_amount(weights)
        apply_open_offset(out, rest, amt)
        return out

    def blink(
        self,
        mesh: np.ndarray | None,
        rest: np.ndarray | None,
        blink: dict[str, float] | None,
    ) -> np.ndarray | None:
        """Each eye at its blink (0 open, 1 shut) along Eye open -> Eye closed.

        Adds the shape's move from ``rest`` to ``mesh``, so whatever else
        drove the eye points stays on top. Unsaved, Eye open is rest and Eye
        closed lays each lid on its corner line.
        """
        if mesh is None or rest is None:
            return mesh
        rest = copy_pts(rest)
        opened = self.shapes.get("eye_open", rest)
        shut = self.shapes.get("eye_closed")
        if shut is None:
            shut = draft_eyes(opened)
        stops = self._stops("eye_open", "eye_closed")
        out = copy_pts(mesh)
        for key, slots in (("l", EYE_L), ("r", EYE_R)):
            try:
                amount = float(np.clip(float((blink or {}).get(key) or 0.0), 0.0, 1.0))
            except (TypeError, ValueError):
                amount = 0.0
            xy = _along(opened, shut, stops, amount, slots)
            out[list(slots), :2] += xy - rest[list(slots), :2]
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
                for name in AUTHOR_IDS
            ],
            "points": pts_to_json(self.current() if self.active else rest),
            "shapes": {name: pts_to_json(self.shapes[name]) for name in _stored_ids(self.shapes)},
            "mids": [name for name in self.shapes if pair_ends(name) is not None],
        }


book = MouthBook()
