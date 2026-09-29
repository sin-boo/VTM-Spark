"""Character travel limits in Track Lab character-pixel space.

Two walls, both fixed to the rest still: a head box and a body box. Room is
measured in face heights past each rest box. Each wall holds its own part:
the head (face, iris, hair) slides back into the head wall, the torso slides
back into the body wall, and the neck is the give between them. So the head
can go as far as the head room while the torso stops at the body room. Turn, tilt,
and look cap degrees on the FaceRig; size caps how much stepping toward or
away from the camera may zoom it; eye range caps how far the iris travels.

Do not import the parent package — this module is the lab's source of truth.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAVEL_PATH = ROOT / "output" / "travel_box.json"

NUM_KEYPOINTS = 37
KEYPOINT_DIM = 4
FACE_SLOTS = tuple(range(0, 28))
OUTLINE = (0, 1, 2, 3, 4)
BROWS = (5, 6, 7, 8, 9, 10)
L_EYE = (11, 12, 13)
R_EYE = (17, 18, 19)
NOSE = (14, 15, 16)
RIGHT_IRIS = 28  # IRIS.L / person-left
LEFT_IRIS = 29
IRIS_EYE_PAIRS = ((RIGHT_IRIS, L_EYE), (LEFT_IRIS, R_EYE))
NECK = 31
L_SHOULDER = 34
R_SHOULDER = 32
CHEST = 36
OVERLAY_BODY_SLOTS = (31, 32, 33, 34, 35, 36)
HEAD_SLOTS = OUTLINE + BROWS + L_EYE + R_EYE + NOSE
FACE_BLEND_SLOTS = FACE_SLOTS + (RIGHT_IRIS, LEFT_IRIS)
# Where the head and body actually are. The jaw drops when talking and the
# elbows swing — neither is a walk, so neither may push the character.
HEAD_ANCHOR = BROWS + L_EYE + R_EYE + NOSE
BODY_ANCHOR = (NECK, R_SHOULDER, L_SHOULDER, CHEST)

YAW_MAX_DEG = 80.0
ROLL_MAX_DEG = 80.0
PITCH_UP_MAX_DEG = 50.0
PITCH_DOWN_MAX_DEG = 32.0
# Size is how far stepping toward / away from the camera may grow or shrink
# the character, as a fraction of rest size. The rig itself stops at 0.62–1.70.
SIZE_MAX = 0.7

# Boxes saved before this version used other walls; their room values do not carry over.
TRAVEL_VERSION = 2
ROOM_KEYS = ("left", "right", "up", "down", "body_left", "body_right", "body_up", "body_down")

DEFAULT_TRAVEL_BOX: dict[str, Any] = {
    "version": TRAVEL_VERSION,
    "enabled": True,
    "left": 0.68,
    "right": 0.65,
    "up": 0.29,
    "down": 0.57,
    "body_left": 0.54,
    "body_right": 0.48,
    "body_up": 0.1,
    "body_down": 0.12,
    "turn_left": 21.0,
    "turn_right": 22.0,
    "tilt_left": 22.0,
    "tilt_right": 12.0,
    "pitch_up": 14.0,
    "pitch_down": 3.0,
    "eye": 0.56,
    "size": 0.0,
}

_RANGE = (0.0, 1.2)
_EYE = (0.0, 1.0)
_SIZE = (0.0, SIZE_MAX)
# Left / right are the screen sides the character's face turns or its crown
# tilts toward; positive yaw / roll is screen-right.
_DEG = {
    "turn_left": (0.0, YAW_MAX_DEG),
    "turn_right": (0.0, YAW_MAX_DEG),
    "tilt_left": (0.0, ROLL_MAX_DEG),
    "tilt_right": (0.0, ROLL_MAX_DEG),
    "pitch_up": (0.0, PITCH_UP_MAX_DEG),
    "pitch_down": (0.0, PITCH_DOWN_MAX_DEG),
}
# Boxes saved before the split had one ``yaw`` / ``roll`` for both sides.
_ONE_SIDED = {"turn_left": "yaw", "turn_right": "yaw", "tilt_left": "roll", "tilt_right": "roll"}

HEAD_COLOR = "#f45b69"
BODY_COLOR = "#5ba4f4"


def _clip(v: Any, lo: float, hi: float, default: float | None = None) -> float:
    try:
        value = float(v)
    except (TypeError, ValueError):
        value = float(lo if default is None else default)
    if not math.isfinite(value):
        value = float(lo if default is None else default)
    return float(min(max(value, float(lo)), float(hi)))


# Past the wall, this fraction of the allowed travel is the spring.
# Motion starts 1:1 at the wall and eases to a stop at wall + give.
_BARRIER_GIVE = 0.45


def soft_barrier(value: float, lo: float, hi: float, origin: float, *, give: float = _BARRIER_GIVE) -> float:
    """Keep 1:1 motion inside the wall. Outside, travel gets harder and then stops."""
    value = float(value)
    lo = float(lo)
    hi = float(hi)
    origin = float(origin)
    if hi < lo:
        lo, hi = hi, lo
    if lo <= value <= hi:
        return value
    if value > hi:
        allowed = max(hi - origin, 0.0)
        if allowed < 1e-6:
            return hi
        room = allowed * float(give)
        over = value - hi
        return hi + room * (1.0 - math.exp(-over / room))
    allowed = max(origin - lo, 0.0)
    if allowed < 1e-6:
        return lo
    room = allowed * float(give)
    over = lo - value
    return lo - room * (1.0 - math.exp(-over / room))


def _bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() not in {"0", "false", "off", "no", ""}
    return bool(v)


def default_travel_box() -> dict[str, Any]:
    return dict(DEFAULT_TRAVEL_BOX)


def normalize_travel_box(raw: Any) -> dict[str, Any]:
    base = default_travel_box()
    if not isinstance(raw, dict):
        return base
    legacy = raw.get("version") != TRAVEL_VERSION
    out: dict[str, Any] = {
        "version": TRAVEL_VERSION,
        "enabled": _bool(raw.get("enabled", base["enabled"]), base["enabled"]),
    }
    lo, hi = _RANGE
    for key in ROOM_KEYS:
        value = base[key] if legacy else raw.get(key, base[key])
        out[key] = _clip(value, lo, hi, base[key])
    for key, (dlo, dhi) in _DEG.items():
        fallback = raw.get(_ONE_SIDED.get(key, key), base[key])
        out[key] = _clip(raw.get(key, fallback), dlo, dhi, base[key])
    eye = raw.get("eye", raw.get("eye_x", base["eye"]))
    out["eye"] = _clip(eye, _EYE[0], _EYE[1], base["eye"])
    out["size"] = _clip(raw.get("size", base["size"]), _SIZE[0], _SIZE[1], base["size"])
    return out


def merge_travel_box(base: Any, patch: Any) -> dict[str, Any]:
    out = normalize_travel_box(base)
    if not isinstance(patch, dict):
        return out
    merged = dict(out)
    merged.update({k: v for k, v in patch.items() if v is not None})
    # An old ``yaw`` / ``roll`` in a patch sets both sides, unless a side is given.
    for side, old in _ONE_SIDED.items():
        if patch.get(old) is not None and patch.get(side) is None:
            merged[side] = patch[old]
    merged["version"] = TRAVEL_VERSION
    return normalize_travel_box(merged)


def boxes_equal(a: Any, b: Any) -> bool:
    left = normalize_travel_box(a)
    right = normalize_travel_box(b)
    for key, value in left.items():
        other = right[key]
        if isinstance(value, bool) or key == "version":
            if value != other:
                return False
        elif isinstance(value, (int, float)):
            if abs(float(value) - float(other)) > 1e-9:
                return False
        elif value != other:
            return False
    return True


def motion_caps(box: Any) -> dict[str, float]:
    spec = normalize_travel_box(box)
    if not spec["enabled"]:
        return {
            "turn_left": YAW_MAX_DEG,
            "turn_right": YAW_MAX_DEG,
            "tilt_left": ROLL_MAX_DEG,
            "tilt_right": ROLL_MAX_DEG,
            "pitch_up": PITCH_UP_MAX_DEG,
            "pitch_down": PITCH_DOWN_MAX_DEG,
        }
    return {
        "turn_left": spec["turn_left"],
        "turn_right": spec["turn_right"],
        "tilt_left": spec["tilt_left"],
        "tilt_right": spec["tilt_right"],
        "pitch_up": spec["pitch_up"],
        "pitch_down": spec["pitch_down"],
    }


def lab_feel_caps(box: Any) -> dict[str, float]:
    """Feel fractions (0–1) for FaceRig euler / size stops and iris travel."""
    caps = motion_caps(box)
    spec = normalize_travel_box(box)
    look = float(spec["eye"]) if spec["enabled"] else 1.0
    size = float(spec["size"]) / SIZE_MAX if spec["enabled"] else 1.0
    return {
        "max_size": size,
        "max_yaw_left": caps["turn_left"] / YAW_MAX_DEG,
        "max_yaw_right": caps["turn_right"] / YAW_MAX_DEG,
        "max_roll_left": caps["tilt_left"] / ROLL_MAX_DEG,
        "max_roll_right": caps["tilt_right"] / ROLL_MAX_DEG,
        "max_pitch_up": caps["pitch_up"] / PITCH_UP_MAX_DEG,
        "max_pitch_down": caps["pitch_down"] / PITCH_DOWN_MAX_DEG,
        "max_look_x": look,
        "max_look_y": look,
    }


def _as37(kps: np.ndarray) -> np.ndarray:
    out = np.asarray(kps, dtype=np.float32)
    if out.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(f"Expected ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {out.shape}")
    return out


def _vis(k: np.ndarray, i: int) -> bool:
    return float(k[i, 3]) >= 0.5


def face_height(kps: np.ndarray) -> float:
    k = _as37(kps)
    tops = [i for i in (5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19) if _vis(k, i)]
    bots = [i for i in (2, 20, 21, 22, 25) if _vis(k, i)]
    if not tops or not bots:
        return 1.0
    return max(1e-3, float(np.mean(k[bots, 1]) - np.mean(k[tops, 1])))


def face_center(kps: np.ndarray) -> np.ndarray:
    k = _as37(kps)
    pts = [k[i, :2] for i in FACE_SLOTS if _vis(k, i)]
    if not pts:
        return np.zeros(2, dtype=np.float32)
    return np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32)


def chin_y(kps: np.ndarray) -> float | None:
    k = _as37(kps)
    if _vis(k, 2):
        return float(k[2, 1])
    bots = [i for i in (25, 24, 27, 1, 3) if _vis(k, i)]
    if not bots:
        return None
    return float(np.max(k[bots, 1]))


def pack_overlay(
    face: np.ndarray | None,
    skeleton: list[dict[str, Any]] | None = None,
    iris: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Face / skeleton / iris lists → (37, 4) in character pixels."""
    k = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
    if face is not None:
        arr = np.asarray(face, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2:
            n = min(28, int(arr.shape[0]))
            k[:n, 0] = arr[:n, 0]
            k[:n, 1] = arr[:n, 1]
            if arr.shape[1] > 2:
                k[:n, 2] = arr[:n, 2]
            else:
                k[:n, 2] = 1.0
            k[:n, 3] = (k[:n, 2] >= 0.05).astype(np.float32)
    if isinstance(iris, list):
        for row in iris:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("id", -1))
            except (TypeError, ValueError):
                continue
            if idx not in (RIGHT_IRIS, LEFT_IRIS):
                continue
            if not row.get("visible", True):
                continue
            k[idx, 0] = float(row.get("x", 0.0))
            k[idx, 1] = float(row.get("y", 0.0))
            k[idx, 2] = float(row.get("score", 1.0) or 1.0)
            k[idx, 3] = 1.0 if k[idx, 2] >= 0.05 else 0.0
    if isinstance(skeleton, list):
        for joint in skeleton:
            if not isinstance(joint, dict):
                continue
            try:
                idx = int(joint.get("id", -1))
            except (TypeError, ValueError):
                continue
            if idx < 31 or idx > 36:
                continue
            k[idx, 0] = float(joint.get("x", 0.0))
            k[idx, 1] = float(joint.get("y", 0.0))
            k[idx, 2] = float(joint.get("score", 1.0) or 1.0)
            k[idx, 3] = 1.0 if k[idx, 2] >= 0.05 else 0.0
    return k


def unpack_face(k: np.ndarray, template: np.ndarray | None = None) -> np.ndarray:
    out = _as37(k)
    if template is not None:
        face = np.asarray(template, dtype=np.float32).copy()
    else:
        face = np.zeros((28, 3), dtype=np.float32)
        face[:, 2] = 1.0
    n = min(28, int(face.shape[0]))
    face[:n, 0] = out[:n, 0]
    face[:n, 1] = out[:n, 1]
    if face.shape[1] > 2:
        face[:n, 2] = np.where(out[:n, 3] >= 0.5, np.maximum(out[:n, 2], 0.05), 0.0)
    return face


def unpack_skeleton(k: np.ndarray, template: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    out = _as37(k)
    by_id = {int(j.get("id", -1)): dict(j) for j in (template or []) if isinstance(j, dict)}
    rows: list[dict[str, Any]] = []
    for idx in OVERLAY_BODY_SLOTS:
        if not _vis(out, idx):
            if idx in by_id:
                rows.append(by_id[idx])
            continue
        row = dict(by_id.get(idx) or {"id": idx, "score": 1.0})
        row["id"] = idx
        row["x"] = round(float(out[idx, 0]), 1)
        row["y"] = round(float(out[idx, 1]), 1)
        row["score"] = float(out[idx, 2]) if float(out[idx, 2]) > 0 else 1.0
        rows.append(row)
    return rows


def unpack_iris(k: np.ndarray, template: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    out = _as37(k)
    by_id = {int(j.get("id", -1)): dict(j) for j in (template or []) if isinstance(j, dict)}
    rows: list[dict[str, Any]] = []
    for idx in (RIGHT_IRIS, LEFT_IRIS):
        if not _vis(out, idx):
            continue
        row = dict(by_id.get(idx) or {"id": idx, "score": 1.0, "visible": True})
        row["id"] = idx
        row["x"] = float(out[idx, 0])
        row["y"] = float(out[idx, 1])
        row["score"] = float(out[idx, 2]) if float(out[idx, 2]) > 0 else 1.0
        row["visible"] = True
        rows.append(row)
    return rows


def _bbox(k: np.ndarray, slots: tuple[int, ...]) -> tuple[float, float, float, float] | None:
    pts = [k[i, :2] for i in slots if _vis(k, i)]
    if not pts:
        return None
    arr = np.stack(pts, axis=0)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    )


def _room(spec: dict[str, Any], fh: float, prefix: str) -> tuple[float, float, float, float]:
    """Allowed travel from rest: (-left, right, -up, down)."""
    return (
        -float(spec[f"{prefix}left"]) * fh,
        float(spec[f"{prefix}right"]) * fh,
        -float(spec[f"{prefix}up"]) * fh,
        float(spec[f"{prefix}down"]) * fh,
    )


def _wall(
    rest: np.ndarray | None,
    box: Any,
    slots: tuple[int, ...],
    prefix: str,
) -> tuple[float, float, float, float] | None:
    if rest is None:
        return None
    k = _as37(rest)
    rect = _bbox(k, slots)
    if rect is None:
        return None
    spec = normalize_travel_box(box)
    lo_x, hi_x, lo_y, hi_y = _room(spec, face_height(k), prefix)
    return rect[0] + lo_x, rect[1] + lo_y, rect[2] + hi_x, rect[3] + hi_y


def head_mesh_rect_px(rest: np.ndarray | None, box: Any = None) -> tuple[float, float, float, float] | None:
    """Rest head box: jaw, brows, eyes, nose."""
    return None if rest is None else _bbox(_as37(rest), HEAD_SLOTS)


def head_rect_px(rest: np.ndarray | None, box: Any) -> tuple[float, float, float, float] | None:
    """Head wall: the rest head box plus head room."""
    return _wall(rest, box, HEAD_SLOTS, "")


def body_mesh_rect_px(rest: np.ndarray | None, box: Any = None) -> tuple[float, float, float, float] | None:
    """Rest body box: neck, shoulders, elbows, chest."""
    return None if rest is None else _bbox(_as37(rest), OVERLAY_BODY_SLOTS)


def body_rect_px(rest: np.ndarray | None, box: Any) -> tuple[float, float, float, float] | None:
    """Body wall: the rest body box plus body room."""
    return _wall(rest, box, OVERLAY_BODY_SLOTS, "body_")


def _anchor_moved(live: np.ndarray, rest: np.ndarray, slots: tuple[int, ...]) -> np.ndarray | None:
    idx = [i for i in slots if _vis(live, i) and _vis(rest, i)]
    if not idx:
        return None
    return np.mean(live[idx, :2], axis=0) - np.mean(rest[idx, :2], axis=0)


def _points_moved(live: np.ndarray, rest: np.ndarray, slots: tuple[int, ...]) -> np.ndarray | None:
    idx = [i for i in slots if _vis(live, i) and _vis(rest, i)]
    if not idx:
        return None
    return live[idx, :2] - rest[idx, :2]


def _hold(moved: np.ndarray | None, room: tuple[float, float, float, float]) -> tuple[float, float]:
    """Smallest slide that keeps every moved row inside the room.

    A part that spread wider than its room (grown, tilted) is centered in it.
    """
    if moved is None:
        return 0.0, 0.0
    rows = np.atleast_2d(np.asarray(moved, dtype=np.float64))
    shift = [0.0, 0.0]
    for axis in (0, 1):
        lo = room[axis * 2] - float(rows[:, axis].min())
        hi = room[axis * 2 + 1] - float(rows[:, axis].max())
        shift[axis] = 0.5 * (lo + hi) if lo > hi else min(max(0.0, lo), hi)
    return float(shift[0]), float(shift[1])


def limit_shifts(
    live: np.ndarray, rest: np.ndarray, box: Any
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(head slide, body slide) that bring each part back inside its own wall.

    The head is placed by its features' center, so a turn is not a walk. The
    torso is held point by point: a torso that turns or swings about its own
    center keeps that center still, and would slip past a center-only wall.
    """
    spec = normalize_travel_box(box)
    if not spec["enabled"]:
        return (0.0, 0.0), (0.0, 0.0)
    k_live = _as37(live)
    k_rest = _as37(rest)
    fh = face_height(k_rest)
    head = _hold(_anchor_moved(k_live, k_rest, HEAD_ANCHOR), _room(spec, fh, ""))
    body = _hold(_points_moved(k_live, k_rest, BODY_ANCHOR), _room(spec, fh, "body_"))
    return head, body


def _shift_hair(segments: list[dict[str, Any]] | None, dx: float, dy: float) -> list[dict[str, Any]] | None:
    if segments is None:
        return None
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return segments
    out: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        poly = seg.get("polygon") or []
        pts: list[list[float]] = []
        for vertex in poly:
            if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                continue
            pts.append([float(vertex[0]) + dx, float(vertex[1]) + dy])
        rec = dict(seg)
        rec["polygon"] = pts
        out.append(rec)
    return out


def eye_range_rects(k: np.ndarray, box: Any) -> list[tuple[float, float, float, float]]:
    """Where each iris may sit: the eye opening scaled by the eye range."""
    spec = normalize_travel_box(box)
    frac = float(spec["eye"])
    out: list[tuple[float, float, float, float]] = []
    for _iris, eye in IRIS_EYE_PAIRS:
        corners = [i for i in (eye[0], eye[2]) if _vis(k, i)]
        if len(corners) < 2:
            continue
        xs = [float(k[i, 0]) for i in corners]
        ys = [float(k[i, 1]) for i in corners]
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
        hw = max((max(xs) - min(xs)) * 0.5, 1e-4)
        hh = max((max(ys) - min(ys)) * 0.5, hw * 0.22)
        lid = eye[1]
        if _vis(k, lid):
            hh = max(abs(cy - float(k[lid, 1])), hw * 0.22)
        out.append((cx - hw * frac, cy - hh * frac, cx + hw * frac, cy + hh * frac))
    return out


def _clamp_iris_in_eyes(k: np.ndarray, spec: dict[str, Any]) -> None:
    rects = eye_range_rects(k, spec)
    pairs = [(iris, eye) for iris, eye in IRIS_EYE_PAIRS if sum(_vis(k, i) for i in (eye[0], eye[2])) == 2]
    for (iris, _eye), (x0, y0, x1, y1) in zip(pairs, rects):
        if not _vis(k, iris):
            continue
        k[iris, 0] = float(min(max(float(k[iris, 0]), x0), x1))
        k[iris, 1] = float(min(max(float(k[iris, 1]), y0), y1))


def apply_limits(
    live: np.ndarray,
    rest: np.ndarray | None,
    box: Any,
    *,
    hair: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]] | None]:
    """Hold the character inside the head and body walls. Expression is untouched."""
    spec = normalize_travel_box(box)
    k_live = _as37(live)
    if rest is None or not spec["enabled"]:
        return k_live, hair
    k_rest = _as37(rest)
    (hx, hy), (bx, by) = limit_shifts(k_live, k_rest, spec)
    out = k_live.copy()
    vis = out[:, 3] >= 0.5
    body = np.zeros(NUM_KEYPOINTS, dtype=bool)
    body[list(OVERLAY_BODY_SLOTS)] = True
    for part, dx, dy in ((vis & ~body, hx, hy), (vis & body, bx, by)):
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            out[part, 0] += dx
            out[part, 1] += dy
    _clamp_iris_in_eyes(out, spec)
    return out, _shift_hair(hair, hx, hy)


def limiter_rects_px(rest: np.ndarray | None, box: Any) -> dict[str, list[float] | None]:
    """Rects for the lab overlay SVG: solid = rest box, dashed = wall."""
    spec = normalize_travel_box(box)
    if not spec["enabled"]:
        return {"head": None, "head_wall": None, "body": None, "body_wall": None}

    def _list(rect: tuple[float, float, float, float] | None) -> list[float] | None:
        if rect is None:
            return None
        return [round(float(v), 2) for v in rect]

    return {
        "head": _list(head_mesh_rect_px(rest)),
        "head_wall": _list(head_rect_px(rest, spec)),
        "body": _list(body_mesh_rect_px(rest)),
        "body_wall": _list(body_rect_px(rest, spec)),
    }


class TravelStore:
    """Persisted travel box + feel-cap sync."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.values = default_travel_box()
        self._load()

    def _load(self) -> None:
        if not TRAVEL_PATH.is_file():
            return
        try:
            data = json.loads(TRAVEL_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.values = normalize_travel_box(data)

    def save(self) -> None:
        TRAVEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRAVEL_PATH.write_text(json.dumps(self.values, indent=2), encoding="utf-8")

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.values)

    def update(self, patch: object) -> tuple[dict[str, Any], bool]:
        """Merge patch. Returns (box, changed)."""
        with self._lock:
            nxt = merge_travel_box(self.values, patch if isinstance(patch, dict) else {})
            if boxes_equal(self.values, nxt):
                return dict(self.values), False
            self.values = nxt
            self.save()
            return dict(self.values), True


travel = TravelStore()
