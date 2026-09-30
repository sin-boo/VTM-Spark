"""Character limiters: how far the pose may move, fixed to the rest still.

Two walls, both boxed around the character's rest pose: a head box (jaw,
brows, eyes, nose) and a body box (neck, shoulders, elbows, chest). Room is
measured in face heights past each rest box. The whole character moves by one
rigid shift, so it stops at whichever wall it reaches first and the head never
tears away from the body. Turn / tilt / look cap degrees; eye range caps the
iris inside each eye. Coordinates are ``norm_crop`` [-1, 1].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .pose_controller import (
    BROWS,
    CHEST,
    FACE_SLOTS,
    IRIS_EYE_PAIRS,
    L_EYE,
    L_SHOULDER,
    LEFT_IRIS,
    NECK,
    NOSE,
    NUM_KEYPOINTS,
    OUTLINE,
    OVERLAY_BODY_SLOTS,
    R_EYE,
    R_SHOULDER,
    RIGHT_IRIS,
    face_height,
    normalized_to_pixels,
)

KEYPOINT_DIM = 4
HEAD_SLOTS = OUTLINE + BROWS + L_EYE + R_EYE + NOSE
FACE_BLEND_SLOTS = FACE_SLOTS + (RIGHT_IRIS, LEFT_IRIS)
# Where the head and body actually are. The jaw drops when talking and the
# elbows swing — neither is a walk, so neither may push the character.
HEAD_ANCHOR = BROWS + L_EYE + R_EYE + NOSE
BODY_ANCHOR = (NECK, R_SHOULDER, L_SHOULDER, CHEST)

# Match Track Lab FaceRig hard stops so a slider at max is "current range".
YAW_MAX_DEG = 80.0
ROLL_MAX_DEG = 80.0
PITCH_UP_MAX_DEG = 50.0
PITCH_DOWN_MAX_DEG = 32.0
# Grow / shrink from rest when the performer steps toward or away from the camera.
SIZE_MAX = 0.7

# Boxes before version 2 used other walls; their room values do not carry over.
# Version 3 only moved look-down on (see _OLD_PITCH_DOWN).
TRAVEL_VERSION = 3
_ROOM_VERSIONS = (2, 3)
# The look-down default before version 3. Every character created then got it
# copied in, which capped a nod at 3 degrees; read it as the current default.
_OLD_PITCH_DOWN = 3.0
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
    "pitch_down": 12.0,
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
BOX_COLOR = (244, 91, 105)
BODY_BOX_COLOR = (91, 164, 244)


def _clip(v: Any, lo: float, hi: float, default: float | None = None) -> float:
    try:
        value = float(v)
    except (TypeError, ValueError):
        value = float(lo if default is None else default)
    if not math.isfinite(value):
        value = float(lo if default is None else default)
    return float(min(max(value, float(lo)), float(hi)))


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
    legacy = raw.get("version") not in _ROOM_VERSIONS
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
    if raw.get("version") != TRAVEL_VERSION and out["pitch_down"] == _OLD_PITCH_DOWN:
        out["pitch_down"] = base["pitch_down"]
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


def motion_caps(box: Any) -> dict[str, float]:
    """Degree caps for live retarget. Limiters off → the full range."""
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
    """Track Lab feel fractions (0–1) for FaceRig euler stops and iris travel."""
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


def _fh(rest: np.ndarray) -> float:
    return max(float(face_height(rest)), 1e-3)


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
    lo_x, hi_x, lo_y, hi_y = _room(spec, _fh(k), prefix)
    return rect[0] + lo_x, rect[1] + lo_y, rect[2] + hi_x, rect[3] + hi_y


def head_mesh_rect_norm(rest: np.ndarray | None, box: Any = None) -> tuple[float, float, float, float] | None:
    """Rest head box: jaw, brows, eyes, nose."""
    return None if rest is None else _bbox(_as37(rest), HEAD_SLOTS)


def head_rect_norm(rest: np.ndarray | None, box: Any) -> tuple[float, float, float, float] | None:
    """Head wall: the rest head box plus head room."""
    return _wall(rest, box, HEAD_SLOTS, "")


def body_mesh_rect_norm(rest: np.ndarray | None, box: Any = None) -> tuple[float, float, float, float] | None:
    """Rest body box: neck, shoulders, elbows, chest."""
    return None if rest is None else _bbox(_as37(rest), OVERLAY_BODY_SLOTS)


def body_rect_norm(rest: np.ndarray | None, box: Any) -> tuple[float, float, float, float] | None:
    """Body wall: the rest body box plus body room."""
    return _wall(rest, box, OVERLAY_BODY_SLOTS, "body_")


def _anchor_moved(live: np.ndarray, rest: np.ndarray, slots: tuple[int, ...]) -> np.ndarray | None:
    idx = [i for i in slots if _vis(live, i) and _vis(rest, i)]
    if not idx:
        return None
    return np.mean(live[idx, :2], axis=0) - np.mean(rest[idx, :2], axis=0)


def oval_stop(moved: np.ndarray | None, room: tuple[float, float, float, float]) -> tuple[float, float]:
    """Where a head that moved by ``moved`` stops on the oval head wall.

    The oval reaches each side's room, so a straight move stops where the
    old box did; a diagonal no longer runs out into the box's corners.
    Inside the oval the move is kept as it is. Same as Track Lab's.
    """
    if moved is None:
        return 0.0, 0.0
    dx, dy = (float(v) for v in np.asarray(moved, dtype=np.float64).reshape(-1)[:2])
    reach_x = room[1] if dx > 0.0 else -room[0]
    reach_y = room[3] if dy > 0.0 else -room[2]
    nx = dx / reach_x if reach_x > 1e-9 else 0.0
    ny = dy / reach_y if reach_y > 1e-9 else 0.0
    if reach_x <= 1e-9:
        dx = 0.0
    if reach_y <= 1e-9:
        dy = 0.0
    r = math.hypot(nx, ny)
    if r <= 1.0:
        return dx, dy
    return dx / r, dy / r


def oval_outline(
    tight: tuple[float, float, float, float] | None,
    wall: tuple[float, float, float, float] | None,
    steps: int = 10,
) -> list[tuple[float, float]]:
    """The oval head wall drawn round the rest head box.

    Straight sides at the old walls, each corner a quarter ellipse whose
    radii are the rooms on that side. Clockwise from the top-left.
    """
    if tight is None or wall is None:
        return []
    x0, y0, x1, y1 = tight
    corners = (
        (x0, y0, x0 - wall[0], y0 - wall[1], math.pi, 1.5 * math.pi),
        (x1, y0, wall[2] - x1, y0 - wall[1], 1.5 * math.pi, 2.0 * math.pi),
        (x1, y1, wall[2] - x1, wall[3] - y1, 0.0, 0.5 * math.pi),
        (x0, y1, x0 - wall[0], wall[3] - y1, 0.5 * math.pi, math.pi),
    )
    out: list[tuple[float, float]] = []
    for cx, cy, rx, ry, a0, a1 in corners:
        for i in range(steps + 1):
            a = a0 + (a1 - a0) * i / steps
            out.append((cx + max(rx, 0.0) * math.cos(a), cy + max(ry, 0.0) * math.sin(a)))
    return out


def limit_shift(live: np.ndarray, rest: np.ndarray, box: Any) -> tuple[float, float]:
    """One rigid shift that brings both the head and the body inside their walls.

    If the two walls cannot both hold (the head leaned farther than both rooms
    together), split the difference so neither side is let off.
    """
    spec = normalize_travel_box(box)
    if not spec["enabled"]:
        return 0.0, 0.0
    k_live = _as37(live)
    k_rest = _as37(rest)
    fh = _fh(k_rest)
    walls: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
    head = _anchor_moved(k_live, k_rest, HEAD_ANCHOR)
    if head is not None:
        # The head wall is an oval: past it, the room on each side the head
        # went is where the oval stops it along its own direction.
        room = list(_room(spec, fh, ""))
        tx, ty = oval_stop(head, tuple(room))
        for axis, stop in ((0, tx), (1, ty)):
            if abs(stop - float(head[axis])) > 1e-9:
                room[axis * 2 + (1 if stop > 0.0 else 0)] = stop
        walls.append((head, (room[0], room[1], room[2], room[3])))
    body = _anchor_moved(k_live, k_rest, BODY_ANCHOR)
    if body is not None:
        walls.append((body, _room(spec, fh, "body_")))
    if not walls:
        return 0.0, 0.0
    shift = [0.0, 0.0]
    for axis in (0, 1):
        lo = -math.inf
        hi = math.inf
        for moved, room in walls:
            lo = max(lo, room[axis * 2] - float(moved[axis]))
            hi = min(hi, room[axis * 2 + 1] - float(moved[axis]))
        shift[axis] = 0.5 * (lo + hi) if lo > hi else min(max(0.0, lo), hi)
    return float(shift[0]), float(shift[1])


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


def _shift_slots(out: np.ndarray, slots: tuple[int, ...] | range, dx: float, dy: float) -> None:
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return
    for i in slots:
        if not _vis(out, i):
            continue
        out[i, 0] = float(out[i, 0] + dx)
        out[i, 1] = float(out[i, 1] + dy)


def _eye_ranges(
    k: np.ndarray, frac: float
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """(iris slot, box the iris may sit in) — the eye opening scaled by ``frac``."""
    out: list[tuple[int, tuple[float, float, float, float]]] = []
    for iris, eye in IRIS_EYE_PAIRS:
        corners = [i for i in (eye[0], eye[2]) if _vis(k, i)]
        if len(corners) < 2:
            continue
        xs = [float(k[i, 0]) for i in corners]
        ys = [float(k[i, 1]) for i in corners]
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
        hw = max((max(xs) - min(xs)) * 0.5, 1e-4)
        hh = max((max(ys) - min(ys)) * 0.5, hw * 0.22)
        lid = eye[1] if len(eye) > 1 else None
        if lid is not None and _vis(k, lid):
            hh = max(abs(cy - float(k[lid, 1])), hw * 0.22)
        out.append((iris, (cx - hw * frac, cy - hh * frac, cx + hw * frac, cy + hh * frac)))
    return out


def _clamp_iris_in_eyes(k: np.ndarray, spec: dict[str, Any]) -> None:
    for iris, (x0, y0, x1, y1) in _eye_ranges(k, float(spec["eye"])):
        if not _vis(k, iris):
            continue
        k[iris, 0] = float(min(max(float(k[iris, 0]), x0), x1))
        k[iris, 1] = float(min(max(float(k[iris, 1]), y0), y1))


def apply_travel_box(
    live: np.ndarray,
    rest: np.ndarray | None,
    box: Any,
    *,
    hair: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]] | None, float]:
    """Hold the character inside the head and body walls.

    Returns ``(keypoints, hair, scale)``; scale is 1.0 because the shape is
    never squashed — only the whole character is slid back.
    """
    spec = normalize_travel_box(box)
    k_live = _as37(live)
    if rest is None or not spec["enabled"]:
        return k_live, hair, 1.0
    k_rest = _as37(rest)
    dx, dy = limit_shift(k_live, k_rest, spec)
    out = k_live.copy()
    _shift_slots(out, range(NUM_KEYPOINTS), dx, dy)
    _clamp_iris_in_eyes(out, spec)
    return out, _shift_hair(hair, dx, dy), 1.0


PREVIEW_AXES = (
    "left",
    "right",
    "up",
    "down",
    "body_left",
    "body_right",
    "body_up",
    "body_down",
    "pitch_up",
    "pitch_down",
    "eye",
)


def changed_preview_axis(old: Any, new: Any) -> str | None:
    """Which limiter slider moved. None if several changed (Reset) or none did."""
    a = normalize_travel_box(old)
    b = normalize_travel_box(new)
    hit = [key for key in PREVIEW_AXES if abs(float(a[key]) - float(b[key])) > 1e-6]
    if len(hit) != 1:
        return None
    return hit[0]


_ROOM_STEP = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
}

# Track Lab's nod (rig.FACE_DEPTH, NECK_FORWARD): each face point's depth
# in front of the eye line in head radii, and how far the eyes swing round
# the neck in face widths. The preview draws the nod the rig will draw.
_NOD_DEPTH = (
    -0.24, -0.14, 0.02, -0.14, -0.24,
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    -0.05, 0.0, 0.0,
    0.06, 0.10, 0.06,
    0.0, 0.0, -0.05,
    0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
)
_NECK_FORWARD = 0.55


def _nod(out: np.ndarray, deg: float) -> None:
    """Nod the face (+ looks down) round the neck; the torso stays."""
    if abs(deg) < 1e-9 or not (_vis(out, 0) and _vis(out, 4) and _vis(out, 15)):
        return
    width = float(np.hypot(*(out[4, :2] - out[0, :2])))
    if width < 1e-6:
        return
    rad = math.radians(float(deg))
    cos_p = math.cos(rad)
    sin_p = math.sin(rad)
    origin_y = float(out[15, 1])
    radius = 1.05 * width
    drop = _NECK_FORWARD * sin_p * width
    for i in FACE_BLEND_SLOTS:
        if not _vis(out, i):
            continue
        depth = _NOD_DEPTH[i] if i < len(_NOD_DEPTH) else 0.0
        rel = float(out[i, 1]) - origin_y
        out[i, 1] = origin_y + rel * cos_p + depth * radius * sin_p + drop


def preview_travel_pose(rest: np.ndarray, box: Any, axis: str) -> np.ndarray:
    """Rest pose walked to the max of one limiter, so the overlay can show it."""
    spec = normalize_travel_box(box)
    out = _as37(rest).copy()
    if not spec["enabled"] or axis not in PREVIEW_AXES:
        return out
    fh = _fh(out)
    side = axis[len("body_"):] if axis.startswith("body_") else axis
    if side in _ROOM_STEP:
        sx, sy = _ROOM_STEP[side]
        amount = float(spec[axis]) * fh
        _shift_slots(out, range(NUM_KEYPOINTS), sx * amount, sy * amount)
    elif axis in ("pitch_up", "pitch_down"):
        sign = -1.0 if axis == "pitch_up" else 1.0
        _nod(out, sign * float(spec[axis]))
    elif axis == "eye":
        for iris, (_x0, y0, x1, _y1) in _eye_ranges(out, float(spec["eye"])):
            if _vis(out, iris):
                out[iris, 0] = x1
                out[iris, 1] = y0
    return out


def _rect_pixels(
    image_size: tuple[int, int],
    rect: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, h = image_size
    dummy = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
    dummy[0, 0], dummy[0, 1], dummy[0, 3] = rect[0], rect[1], 1.0
    dummy[1, 0], dummy[1, 1], dummy[1, 3] = rect[2], rect[3], 1.0
    pix = normalized_to_pixels(dummy, w, h)
    x0, y0 = float(pix[0, 0]), float(pix[0, 1])
    x1, y1 = float(pix[1, 0]), float(pix[1, 1])
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    a: tuple[float, float],
    b: tuple[float, float],
    color: tuple[int, int, int],
    *,
    width: int = 2,
    dash: float = 8.0,
    gap: float = 5.0,
) -> None:
    x0, y0 = a
    x1, y1 = b
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    ux, uy = dx / length, dy / length
    t = 0.0
    on = True
    while t < length:
        step = dash if on else gap
        t2 = min(length, t + step)
        if on:
            draw.line(
                [(x0 + ux * t, y0 + uy * t), (x0 + ux * t2, y0 + uy * t2)],
                fill=color,
                width=width,
            )
        t = t2
        on = not on


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    rect: tuple[float, float, float, float],
    color: tuple[int, int, int],
    *,
    dashed: bool = False,
) -> None:
    x0, y0, x1, y1 = _rect_pixels(image_size, rect)
    w, h = image_size
    x0 = min(max(1.0, x0), max(2.0, w - 2.0))
    y0 = min(max(1.0, y0), max(2.0, h - 2.0))
    x1 = min(max(x0 + 1.0, x1), max(3.0, w - 1.0))
    y1 = min(max(y0 + 1.0, y1), max(3.0, h - 1.0))
    segs = [
        ((x0, y0), (x1, y0)),
        ((x0, y1), (x1, y1)),
        ((x0, y0), (x0, y1)),
        ((x1, y0), (x1, y1)),
    ]
    for a, b in segs:
        if dashed:
            _draw_dashed_line(draw, a, b, color, width=2)
        else:
            draw.line([a, b], fill=color, width=2)


def _draw_outline(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    outline: list[tuple[float, float]],
    color: tuple[int, int, int],
) -> None:
    """Dashed closed outline; points in the same space as the rects."""
    if len(outline) < 3:
        return
    w, h = image_size
    dummy = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
    pix: list[tuple[float, float]] = []
    for x, y in outline:
        dummy[0, 0], dummy[0, 1], dummy[0, 3] = x, y, 1.0
        p = normalized_to_pixels(dummy, w, h)
        pix.append((
            min(max(1.0, float(p[0, 0])), max(2.0, w - 2.0)),
            min(max(1.0, float(p[0, 1])), max(2.0, h - 2.0)),
        ))
    for a, b in zip(pix, pix[1:] + pix[:1]):
        _draw_dashed_line(draw, a, b, color, width=2)


def draw_travel_box(
    image: Image.Image,
    rest: np.ndarray | None,
    box: Any,
) -> Image.Image:
    """Draw the head and body walls on the still.

    Solid boxes are the rest pose; dashed lines are the walls (the head's
    is an oval). Both come from
    ``rest`` only, so they stay put while the character moves.
    """
    spec = normalize_travel_box(box)
    if image is None or rest is None or not spec["enabled"]:
        return image
    k = _as37(rest)
    base = image.convert("RGB")
    draw = ImageDraw.Draw(base)
    for tight, wall, color, oval in (
        (head_mesh_rect_norm(k), head_rect_norm(k, spec), BOX_COLOR, True),
        (body_mesh_rect_norm(k), body_rect_norm(k, spec), BODY_BOX_COLOR, False),
    ):
        if tight is not None:
            _draw_rect(draw, base.size, tight, color)
        if wall is not None and oval:
            _draw_outline(draw, base.size, oval_outline(tight, wall), color)
        elif wall is not None:
            _draw_rect(draw, base.size, wall, color, dashed=True)
    for _iris, rect in _eye_ranges(k, float(spec["eye"])):
        _draw_rect(draw, base.size, rect, BOX_COLOR, dashed=True)
    return base
