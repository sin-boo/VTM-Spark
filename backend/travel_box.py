"""Character travel limits: Track Lab may move the pose only this far.

Size walls come from the character reference: chroma-key the green screen to a
hard mask, box the whole foreground (hair included), then pad ``pad_px`` on
every side. Head left / right sliders are extra side room for the face mesh
(jaw, brows, eyes, nose — not a separate hair box). Look up / down cap nod in
degrees. Skeleton sliders cap how far the stick figure may slide; turn / tilt
cap yaw and roll. Mouth is not used to hit-test the crop.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .pose_controller import (
    FACE_SLOTS,
    IRIS_EYE_PAIRS,
    LEFT_IRIS,
    NUM_KEYPOINTS,
    OVERLAY_BODY_SLOTS,
    RIGHT_IRIS,
    BROWS,
    CHEST,
    L_EYE,
    L_SHOULDER,
    NECK,
    NOSE,
    OUTLINE,
    R_EYE,
    R_SHOULDER,
    enforce_head_body_attachment,
    face_center,
    face_height,
    normalized_to_pixels,
)

KEYPOINT_DIM = 4
HEAD_SLOTS = OUTLINE + BROWS + L_EYE + R_EYE + NOSE
FACE_BLEND_SLOTS = FACE_SLOTS + (RIGHT_IRIS, LEFT_IRIS)

# Match Track Lab FaceRig hard stops so a slider at max is "current range".
YAW_MAX_DEG = 80.0
ROLL_MAX_DEG = 80.0
PITCH_UP_MAX_DEG = 50.0
PITCH_DOWN_MAX_DEG = 32.0
BODY_YAW_MAX_DEG = 80.0
BODY_ROLL_MAX_DEG = 80.0

DEFAULT_TRAVEL_BOX: dict[str, Any] = {
    "enabled": True,
    "side": True,
    "rotate": True,
    "look_up": True,
    "look_down": True,
    "body": True,
    "body_rotate": True,
    "eyes": True,
    "left": 0.0,
    "right": 0.0,
    "up": 0.0,
    "down": 0.0,
    "body_left": 0.0,
    "body_right": 0.0,
    "body_up": 0.0,
    "body_down": 0.0,
    "yaw": YAW_MAX_DEG,
    "roll": ROLL_MAX_DEG,
    "pitch_up": PITCH_UP_MAX_DEG,
    "pitch_down": PITCH_DOWN_MAX_DEG,
    "body_yaw": BODY_YAW_MAX_DEG,
    "body_roll": BODY_ROLL_MAX_DEG,
    "eye_x": 0.78,
    "eye_y": 0.78,
    "pad_px": 50,
}

_RANGE = (0.0, 1.2)
_EYE = (0.0, 1.0)
_PAD_PX = (0, 200)
_DEG = {
    "yaw": (0.0, YAW_MAX_DEG),
    "roll": (0.0, ROLL_MAX_DEG),
    "pitch_up": (0.0, PITCH_UP_MAX_DEG),
    "pitch_down": (0.0, PITCH_DOWN_MAX_DEG),
    "body_yaw": (0.0, BODY_YAW_MAX_DEG),
    "body_roll": (0.0, BODY_ROLL_MAX_DEG),
}
# Match training chroma-key so the limiter sees the same character crop.
_GREEN_THRESH = 40.0
_GREEN_DOMINANCE = 20.0
SIZE_COLOR = (210, 210, 210)
BOX_COLOR = (244, 91, 105)
BODY_BOX_COLOR = (91, 164, 244)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


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
    lo, hi = _RANGE
    pad_lo, pad_hi = _PAD_PX
    side = _bool(raw.get("side", raw.get("head", base["side"])), base["side"])
    out = {
        "enabled": _bool(raw.get("enabled", base["enabled"]), base["enabled"]),
        "side": side,
        "rotate": _bool(raw.get("rotate", base["rotate"]), base["rotate"]),
        "look_up": _bool(raw.get("look_up", base["look_up"]), base["look_up"]),
        "look_down": _bool(raw.get("look_down", base["look_down"]), base["look_down"]),
        "body": _bool(raw.get("body", base["body"]), base["body"]),
        "body_rotate": _bool(raw.get("body_rotate", base["body_rotate"]), base["body_rotate"]),
        "eyes": _bool(raw.get("eyes", base["eyes"]), base["eyes"]),
        "left": _clip(raw.get("left", base["left"]), lo, hi),
        "right": _clip(raw.get("right", base["right"]), lo, hi),
        "up": _clip(raw.get("up", base["up"]), lo, hi),
        "down": _clip(raw.get("down", base["down"]), lo, hi),
        "body_left": _clip(raw.get("body_left", base["body_left"]), lo, hi),
        "body_right": _clip(raw.get("body_right", base["body_right"]), lo, hi),
        "body_up": _clip(raw.get("body_up", base["body_up"]), lo, hi),
        "body_down": _clip(raw.get("body_down", base["body_down"]), lo, hi),
        "pad_px": int(round(_clip(raw.get("pad_px", base["pad_px"]), pad_lo, pad_hi))),
        "eye_x": _clip(raw.get("eye_x", base["eye_x"]), _EYE[0], _EYE[1]),
        "eye_y": _clip(raw.get("eye_y", base["eye_y"]), _EYE[0], _EYE[1]),
    }
    for key, (dlo, dhi) in _DEG.items():
        out[key] = _clip(raw.get(key, base[key]), dlo, dhi)
    return out


def motion_caps(box: Any) -> dict[str, float]:
    """Degree / unit caps for live retarget. Off walls use the current max."""
    spec = normalize_travel_box(box)
    yaw = YAW_MAX_DEG
    roll = ROLL_MAX_DEG
    pitch_up = PITCH_UP_MAX_DEG
    pitch_down = PITCH_DOWN_MAX_DEG
    if spec["enabled"] and spec["rotate"]:
        yaw = spec["yaw"]
        roll = spec["roll"]
    if spec["enabled"] and spec["look_up"]:
        pitch_up = spec["pitch_up"]
    if spec["enabled"] and spec["look_down"]:
        pitch_down = spec["pitch_down"]
    return {
        "yaw": yaw,
        "roll": roll,
        "pitch_up": pitch_up,
        "pitch_down": pitch_down,
    }


def lab_feel_caps(box: Any) -> dict[str, float]:
    """Track Lab feel fractions (0–1) for FaceRig euler stops and iris travel."""
    caps = motion_caps(box)
    spec = normalize_travel_box(box)
    look_x = 1.0
    look_y = 1.0
    if spec["enabled"] and spec["eyes"]:
        look_x = float(spec["eye_x"])
        look_y = float(spec["eye_y"])
    return {
        "max_yaw": caps["yaw"] / YAW_MAX_DEG,
        "max_roll": caps["roll"] / ROLL_MAX_DEG,
        "max_pitch_up": caps["pitch_up"] / PITCH_UP_MAX_DEG,
        "max_pitch_down": caps["pitch_down"] / PITCH_DOWN_MAX_DEG,
        "max_look_x": look_x,
        "max_look_y": look_y,
    }


def merge_travel_box(base: Any, patch: Any) -> dict[str, Any]:
    out = normalize_travel_box(base)
    if not isinstance(patch, dict):
        return out
    merged = dict(out)
    merged.update({k: v for k, v in patch.items() if v is not None})
    return normalize_travel_box(merged)


def _as37(kps: np.ndarray) -> np.ndarray:
    out = np.asarray(kps, dtype=np.float32)
    if out.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(f"Expected ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {out.shape}")
    return out


def _vis(k: np.ndarray, i: int) -> bool:
    return float(k[i, 3]) >= 0.5


def limit_slots(box: Any) -> tuple[int, ...]:
    spec = normalize_travel_box(box)
    slots = list(HEAD_SLOTS)
    if spec["body"]:
        slots.extend(OVERLAY_BODY_SLOTS)
    return tuple(slots)


def _base_rect_norm(
    rest: np.ndarray | None,
    spec: dict[str, Any],
    *,
    silhouette: tuple[float, float, float, float] | None,
    slots: tuple[int, ...],
) -> tuple[float, float, float, float] | None:
    if silhouette is not None and len(silhouette) == 4:
        x0, y0, x1, y1 = (float(v) for v in silhouette)
    else:
        if rest is None:
            return None
        k = _as37(rest)
        pts = _points(k, slots)
        if pts is None:
            return None
        x0 = float(pts[:, 0].min())
        x1 = float(pts[:, 0].max())
        y0 = float(pts[:, 1].min())
        y1 = float(pts[:, 1].max())
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _expand_rect(
    rect: tuple[float, float, float, float],
    fh: float,
    *,
    left: float = 0.0,
    right: float = 0.0,
    up: float = 0.0,
    down: float = 0.0,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    x0 -= float(left) * fh
    x1 += float(right) * fh
    y0 -= float(up) * fh
    y1 += float(down) * fh
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def size_rect_norm(
    rest: np.ndarray | None,
    box: Any,
    *,
    silhouette: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Whole-character crop in ``norm_crop`` (hair included). Ignores head left / right."""
    spec = normalize_travel_box(box)
    if silhouette is not None and len(silhouette) == 4:
        x0, y0, x1, y1 = (float(v) for v in silhouette)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, y0, x1, y1
    return _base_rect_norm(
        rest, spec, silhouette=None, slots=HEAD_SLOTS + OVERLAY_BODY_SLOTS
    )


def head_mesh_rect_norm(
    rest: np.ndarray | None,
    box: Any,
    *,
    silhouette: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Tight axis-aligned box around jaw, brows, eyes, and nose.

    Head Left / Right extra room is a clamp allowance on the rest pose, not a
    drawn offset. Baking it into the overlay rectangle is what made the pink
    box sit off the face when those sliders were unequal.
    """
    spec = normalize_travel_box(box)
    return _base_rect_norm(rest, spec, silhouette=None, slots=HEAD_SLOTS)


def head_rect_norm(
    rest: np.ndarray | None,
    box: Any,
    *,
    silhouette: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Face-mesh box plus Head Left / Right extra slide room (clamp walls)."""
    spec = normalize_travel_box(box)
    rect = head_mesh_rect_norm(rest, spec)
    if rect is None:
        return None
    fh = 1e-3
    if rest is not None:
        fh = max(float(face_height(_as37(rest))), 1e-3)
    left = spec["left"] if spec["side"] else 0.0
    right = spec["right"] if spec["side"] else 0.0
    return _expand_rect(rect, fh, left=left, right=right)


def box_rect_norm(
    rest: np.ndarray | None,
    box: Any,
    *,
    silhouette: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Alias for the face Head box (kept for older callers)."""
    return head_rect_norm(rest, box, silhouette=silhouette)


def body_rect_norm(
    rest: np.ndarray | None,
    box: Any,
    *,
    silhouette: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Skeleton limiter in ``norm_crop`` — independent of head left / right."""
    spec = normalize_travel_box(box)
    if not spec["body"]:
        return None
    rect = _base_rect_norm(rest, spec, silhouette=None, slots=OVERLAY_BODY_SLOTS)
    if rect is None:
        rect = _base_rect_norm(rest, spec, silhouette=silhouette, slots=HEAD_SLOTS)
    if rect is None:
        return None
    fh = 1e-3
    if rest is not None:
        fh = max(float(face_height(_as37(rest))), 1e-3)
    return _expand_rect(
        rect,
        fh,
        left=spec["body_left"],
        right=spec["body_right"],
        up=spec["body_up"],
        down=spec["body_down"],
    )


def _points(k: np.ndarray, slots: tuple[int, ...]) -> np.ndarray | None:
    pts = [k[i, :2] for i in slots if _vis(k, i)]
    if not pts:
        pts = [k[i, :2] for i in FACE_SLOTS if _vis(k, i)]
    if not pts:
        return None
    return np.stack(pts, axis=0).astype(np.float32)


def character_mask(rgb: np.ndarray) -> np.ndarray:
    """Hard silhouette: True = character, False = chroma-key green."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.zeros(arr.shape[:2], dtype=bool)
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    is_green = (
        (g > _GREEN_THRESH)
        & (g - r > _GREEN_DOMINANCE)
        & (g - b > _GREEN_DOMINANCE)
    )
    return ~is_green


def hard_silhouette(rgb: np.ndarray) -> np.ndarray:
    """Uint8 HxW — green screen black, everything else white."""
    return np.where(character_mask(rgb), np.uint8(255), np.uint8(0))


def _px_to_norm(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    return float(x) / w * 2.0 - 1.0, float(y) / h * 2.0 - 1.0


def silhouette_rect_norm(
    rgb: np.ndarray | None,
    *,
    pad_px: int = 50,
) -> tuple[float, float, float, float] | None:
    """Foreground crop in ``norm_crop``, expanded by ``pad_px`` on every side."""
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    mask = character_mask(arr)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = int(mask.shape[0]), int(mask.shape[1])
    pad = max(0, int(pad_px))
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    nx0, ny0 = _px_to_norm(x0, y0, w, h)
    nx1, ny1 = _px_to_norm(x1, y1, w, h)
    if nx1 < nx0:
        nx0, nx1 = nx1, nx0
    if ny1 < ny0:
        ny0, ny1 = ny1, ny0
    return nx0, ny0, nx1, ny1


def _max_blend_1d(rest: float, live: float, lo: float, hi: float) -> float:
    delta = float(live) - float(rest)
    if abs(delta) < 1e-9:
        return 1.0
    if delta > 0.0:
        return max(0.0, min(1.0, (hi - float(rest)) / delta))
    return max(0.0, min(1.0, (lo - float(rest)) / delta))


def _scale_x(
    live: np.ndarray,
    rest: np.ndarray,
    slots: tuple[int, ...],
    x0: float,
    x1: float,
    enabled: bool,
) -> float:
    if not enabled:
        return 1.0
    scale = 1.0
    for i in slots:
        if not (_vis(live, i) and _vis(rest, i)):
            continue
        scale = min(scale, _max_blend_1d(float(rest[i, 0]), float(live[i, 0]), x0, x1))
    return float(max(0.0, min(1.0, scale)))


def _scale_y(
    live: np.ndarray,
    rest: np.ndarray,
    slots: tuple[int, ...],
    y0: float,
    y1: float,
    *,
    up: bool,
    enabled: bool,
) -> float:
    if not enabled:
        return 1.0
    scale = 1.0
    for i in slots:
        if not (_vis(live, i) and _vis(rest, i)):
            continue
        delta = float(live[i, 1]) - float(rest[i, 1])
        if up and delta >= 0.0:
            continue
        if not up and delta <= 0.0:
            continue
        scale = min(scale, _max_blend_1d(float(rest[i, 1]), float(live[i, 1]), y0, y1))
    return float(max(0.0, min(1.0, scale)))


def _blend_slots(
    live: np.ndarray,
    rest: np.ndarray,
    slots: tuple[int, ...],
    sx: float,
    sy_up: float,
    sy_down: float,
) -> np.ndarray:
    out = live.copy()
    for i in slots:
        if not (_vis(live, i) and _vis(rest, i)):
            continue
        dx = float(live[i, 0]) - float(rest[i, 0])
        dy = float(live[i, 1]) - float(rest[i, 1])
        sy = sy_up if dy < 0.0 else sy_down
        out[i, 0] = float(rest[i, 0] + sx * dx)
        out[i, 1] = float(rest[i, 1] + sy * dy)
    return out


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


def _angle(k: np.ndarray, a: int, b: int) -> float | None:
    if not (_vis(k, a) and _vis(k, b)):
        return None
    return math.atan2(float(k[b, 1]) - float(k[a, 1]), float(k[b, 0]) - float(k[a, 0]))


def _wrap_delta(delta: float) -> float:
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return float(delta)


def _rotate_about(out: np.ndarray, slots: tuple[int, ...], origin: tuple[float, float], rad: float) -> None:
    if abs(rad) < 1e-9:
        return
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    ox, oy = origin
    for i in slots:
        if not _vis(out, i):
            continue
        x = float(out[i, 0]) - ox
        y = float(out[i, 1]) - oy
        out[i, 0] = ox + cos_a * x - sin_a * y
        out[i, 1] = oy + sin_a * x + cos_a * y


def _origin(k: np.ndarray, *slots: int) -> tuple[float, float] | None:
    for i in slots:
        if _vis(k, i):
            return float(k[i, 0]), float(k[i, 1])
    return None


def _cap_body_rotation(out: np.ndarray, rest: np.ndarray, spec: dict[str, Any]) -> None:
    """Keep torso turn (neck→chest) and shoulder tilt within the skeleton caps."""
    if not spec["enabled"] or not spec["body_rotate"]:
        return
    origin = _origin(out, NECK, CHEST) or _origin(rest, NECK, CHEST)
    if origin is None:
        return
    turn = _angle(out, NECK, CHEST)
    rest_turn = _angle(rest, NECK, CHEST)
    if turn is not None and rest_turn is not None:
        cap = math.radians(float(spec["body_yaw"]))
        delta = _wrap_delta(turn - rest_turn)
        if abs(delta) > cap:
            _rotate_about(out, OVERLAY_BODY_SLOTS, origin, math.copysign(cap, delta) - delta)
    tilt = _angle(out, R_SHOULDER, L_SHOULDER)
    rest_tilt = _angle(rest, R_SHOULDER, L_SHOULDER)
    if tilt is not None and rest_tilt is not None:
        cap = math.radians(float(spec["body_roll"]))
        delta = _wrap_delta(tilt - rest_tilt)
        if abs(delta) > cap:
            mid = _origin(out, NECK) or origin
            _rotate_about(out, OVERLAY_BODY_SLOTS, mid, math.copysign(cap, delta) - delta)


def _clamp_group(
    live: np.ndarray,
    rest: np.ndarray,
    slots: tuple[int, ...],
    rect: tuple[float, float, float, float],
    *,
    side: bool,
    up: bool,
    down: bool,
) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = rect
    sx = _scale_x(live, rest, slots, x0, x1, side)
    sy_up = _scale_y(live, rest, slots, y0, y1, up=True, enabled=up)
    sy_down = _scale_y(live, rest, slots, y0, y1, up=False, enabled=down)
    out = _blend_slots(live, rest, slots, sx, sy_up, sy_down)
    return out, min(sx, sy_up, sy_down)


def _clamp_iris_in_eyes(k: np.ndarray, spec: dict[str, Any]) -> None:
    """Keep pupils inside an inset of each live eye socket."""
    if not spec["enabled"] or not spec["eyes"]:
        return
    fx = _clip(spec["eye_x"], 0.0, 1.0)
    fy = _clip(spec["eye_y"], 0.0, 1.0)
    for iris, eye in IRIS_EYE_PAIRS:
        if not _vis(k, iris):
            continue
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
        k[iris, 0] = float(np.clip(float(k[iris, 0]), cx - hw * fx, cx + hw * fx))
        k[iris, 1] = float(np.clip(float(k[iris, 1]), cy - hh * fy, cy + hh * fy))


def _rigid_fit_shift(
    live: np.ndarray,
    slots: tuple[int, ...],
    rect: tuple[float, float, float, float],
    *,
    side: bool,
    up: bool,
    down: bool,
) -> tuple[float, float]:
    """Translation that puts ``slots`` back inside ``rect`` without scaling look."""
    pts = [live[i, :2] for i in slots if _vis(live, i)]
    if not pts:
        return 0.0, 0.0
    stacked = np.stack(pts, axis=0)
    min_x, min_y = float(stacked[:, 0].min()), float(stacked[:, 1].min())
    max_x, max_y = float(stacked[:, 0].max()), float(stacked[:, 1].max())
    x0, y0, x1, y1 = rect
    dx = 0.0
    dy = 0.0
    if side:
        left_need = x0 - min_x
        right_need = x1 - max_x
        if left_need > 0.0 and right_need < 0.0:
            dx = 0.0
        elif left_need > 0.0:
            dx = left_need
        elif right_need < 0.0:
            dx = right_need
    top_need = y0 - min_y
    bot_need = y1 - max_y
    if up and down and top_need > 0.0 and bot_need < 0.0:
        dy = 0.0
    elif up and top_need > 0.0:
        dy = top_need
    elif down and bot_need < 0.0:
        dy = bot_need
    return float(dx), float(dy)


def apply_walk_box(
    live: np.ndarray,
    rest: np.ndarray | None,
    box: Any,
    *,
    hair: list[dict[str, Any]] | None = None,
    silhouette: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]] | None, float]:
    """Slide the whole mesh back when Track Lab walks past the size walls.

    Look / expression stay as the lab authored them — same shift on every
    point. The per-point blend in ``apply_travel_box`` would squash a look.
    """
    spec = normalize_travel_box(box)
    k_live = _as37(live)
    if rest is None or not spec["enabled"]:
        return k_live, hair, 1.0
    k_rest = _as37(rest)
    rect = size_rect_norm(k_rest, spec, silhouette=silhouette)
    if rect is None:
        return k_live, hair, 1.0
    dx, dy = _rigid_fit_shift(
        k_live,
        HEAD_SLOTS + OVERLAY_BODY_SLOTS,
        rect,
        side=True,
        up=True,
        down=True,
    )
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return k_live, hair, 1.0
    out = k_live.copy()
    for i in range(NUM_KEYPOINTS):
        if not _vis(out, i):
            continue
        out[i, 0] = float(out[i, 0] + dx)
        out[i, 1] = float(out[i, 1] + dy)
    return out, _shift_hair(hair, dx, dy), 1.0


def apply_travel_box(
    live: np.ndarray,
    rest: np.ndarray | None,
    box: Any,
    *,
    hair: list[dict[str, Any]] | None = None,
    silhouette: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]] | None, float]:
    """Clamp live pose toward rest on the enabled movement walls.

    Head and skeleton use separate boxes. Returns ``(keypoints, hair, scale)``.
    """
    spec = normalize_travel_box(box)
    k_live = _as37(live)
    if rest is None or not spec["enabled"]:
        return k_live, hair, 1.0
    k_rest = _as37(rest)
    out = k_live.copy()
    scale = 1.0
    head_rect = head_rect_norm(k_rest, spec)
    if head_rect is not None:
        out, head_scale = _clamp_group(
            out,
            k_rest,
            FACE_BLEND_SLOTS,
            head_rect,
            side=spec["side"],
            up=False,
            down=False,
        )
        scale = min(scale, head_scale)
    body_rect = body_rect_norm(k_rest, spec, silhouette=silhouette)
    if body_rect is not None:
        out, body_scale = _clamp_group(
            out,
            k_rest,
            OVERLAY_BODY_SLOTS,
            body_rect,
            side=True,
            up=True,
            down=True,
        )
        scale = min(scale, body_scale)
    _cap_body_rotation(out, k_rest, spec)
    _clamp_iris_in_eyes(out, spec)
    enforce_head_body_attachment(out, k_rest)
    c_live = face_center(k_live)
    c_out = face_center(out)
    hair_out = _shift_hair(hair, float(c_out[0] - c_live[0]), float(c_out[1] - c_live[1]))
    return out, hair_out, scale


PREVIEW_AXES = (
    "left",
    "right",
    "pitch_up",
    "pitch_down",
    "body_left",
    "body_right",
    "body_up",
    "body_down",
    "body_yaw",
    "body_roll",
    "eye_x",
    "eye_y",
)


def changed_preview_axis(old: Any, new: Any) -> str | None:
    """Which limiter slider moved. None if several changed (Reset) or none did."""
    a = normalize_travel_box(old)
    b = normalize_travel_box(new)
    hit = [key for key in PREVIEW_AXES if abs(float(a[key]) - float(b[key])) > 1e-6]
    if len(hit) != 1:
        return None
    return hit[0]


def preview_travel_pose(rest: np.ndarray, box: Any, axis: str) -> np.ndarray:
    """Rest pose walked to the max of one limiter, so the overlay can show it."""
    spec = normalize_travel_box(box)
    out = _as37(rest).copy()
    if not spec["enabled"] or axis not in PREVIEW_AXES:
        return out
    fh = max(float(face_height(out)), 1e-3)
    if axis == "left" and spec["side"]:
        _shift_slots(out, FACE_BLEND_SLOTS, -float(spec["left"]) * fh, 0.0)
    elif axis == "right" and spec["side"]:
        _shift_slots(out, FACE_BLEND_SLOTS, float(spec["right"]) * fh, 0.0)
    elif axis == "pitch_up" and spec["look_up"]:
        center = face_center(out)
        _rotate_about(
            out,
            FACE_BLEND_SLOTS,
            (float(center[0]), float(center[1])),
            math.radians(-float(spec["pitch_up"])),
        )
    elif axis == "pitch_down" and spec["look_down"]:
        center = face_center(out)
        _rotate_about(
            out,
            FACE_BLEND_SLOTS,
            (float(center[0]), float(center[1])),
            math.radians(float(spec["pitch_down"])),
        )
    elif axis == "body_left" and spec["body"]:
        _shift_slots(out, OVERLAY_BODY_SLOTS, -float(spec["body_left"]) * fh, 0.0)
    elif axis == "body_right" and spec["body"]:
        _shift_slots(out, OVERLAY_BODY_SLOTS, float(spec["body_right"]) * fh, 0.0)
    elif axis == "body_up" and spec["body"]:
        _shift_slots(out, OVERLAY_BODY_SLOTS, 0.0, -float(spec["body_up"]) * fh)
    elif axis == "body_down" and spec["body"]:
        _shift_slots(out, OVERLAY_BODY_SLOTS, 0.0, float(spec["body_down"]) * fh)
    elif axis == "body_yaw" and spec["body_rotate"]:
        origin = _origin(out, NECK, CHEST)
        if origin is not None:
            _rotate_about(out, OVERLAY_BODY_SLOTS, origin, math.radians(float(spec["body_yaw"])))
    elif axis == "body_roll" and spec["body_rotate"]:
        origin = _origin(out, NECK) or _origin(out, CHEST)
        if origin is not None:
            _rotate_about(out, OVERLAY_BODY_SLOTS, origin, math.radians(-float(spec["body_roll"])))
    elif axis == "eye_x" and spec["eyes"]:
        _preview_iris(out, spec, "x")
    elif axis == "eye_y" and spec["eyes"]:
        _preview_iris(out, spec, "y")
    return out


def _shift_slots(out: np.ndarray, slots: tuple[int, ...], dx: float, dy: float) -> None:
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return
    for i in slots:
        if not _vis(out, i):
            continue
        out[i, 0] = float(out[i, 0] + dx)
        out[i, 1] = float(out[i, 1] + dy)


def _preview_iris(k: np.ndarray, spec: dict[str, Any], axis: str) -> None:
    fx = _clip(spec["eye_x"], 0.0, 1.0)
    fy = _clip(spec["eye_y"], 0.0, 1.0)
    for iris, eye in IRIS_EYE_PAIRS:
        if not _vis(k, iris):
            continue
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
        if axis == "x":
            k[iris, 0] = cx + hw * fx
        else:
            k[iris, 1] = cy - hh * fy


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


def _norm_to_px(
    x: float, y: float, image_size: tuple[int, int]
) -> tuple[float, float]:
    dummy = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
    dummy[0, 0], dummy[0, 1], dummy[0, 3] = float(x), float(y), 1.0
    pix = normalized_to_pixels(dummy, image_size[0], image_size[1])
    return float(pix[0, 0]), float(pix[0, 1])


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
    top: bool,
    bottom: bool,
    sides: bool,
    dashed: bool = False,
    outset: float = 0.0,
) -> None:
    x0, y0, x1, y1 = _rect_pixels(image_size, rect)
    x0 -= outset
    y0 -= outset
    x1 += outset
    y1 += outset
    w, h = image_size
    x0 = min(max(1.0, x0), max(2.0, w - 2.0))
    y0 = min(max(1.0, y0), max(2.0, h - 2.0))
    x1 = min(max(x0 + 1.0, x1), max(3.0, w - 1.0))
    y1 = min(max(y0 + 1.0, y1), max(3.0, h - 1.0))
    width = 2
    segs = []
    if top:
        segs.append(((x0, y0), (x1, y0)))
    if bottom:
        segs.append(((x0, y1), (x1, y1)))
    if sides:
        segs.append(((x0, y0), (x0, y1)))
        segs.append(((x1, y0), (x1, y1)))
    for a, b in segs:
        if dashed:
            _draw_dashed_line(draw, a, b, color, width=width)
        else:
            draw.line([a, b], fill=color, width=width)


def _rects_close(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
    eps: float = 1e-4,
) -> bool:
    if a is None or b is None:
        return False
    return all(abs(float(x) - float(y)) < eps for x, y in zip(a, b))


def _rotate_xy(
    x: float, y: float, ox: float, oy: float, rad: float
) -> tuple[float, float]:
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = float(x) - ox, float(y) - oy
    return ox + c * dx - s * dy, oy + s * dx + c * dy


def _draw_rotated_rect(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    rect: tuple[float, float, float, float],
    origin: tuple[float, float],
    deg: float,
    color: tuple[int, int, int],
) -> None:
    if abs(float(deg)) < 0.4:
        return
    x0, y0, x1, y1 = rect
    ox, oy = origin
    rad = math.radians(float(deg))
    corners = [
        _rotate_xy(x0, y0, ox, oy, rad),
        _rotate_xy(x1, y0, ox, oy, rad),
        _rotate_xy(x1, y1, ox, oy, rad),
        _rotate_xy(x0, y1, ox, oy, rad),
    ]
    pix = [_norm_to_px(x, y, image_size) for x, y in corners]
    for a, b in zip(pix, pix[1:] + pix[:1]):
        _draw_dashed_line(draw, a, b, color, width=2)


def _body_mesh_rect_norm(
    rest: np.ndarray | None,
    spec: dict[str, Any],
    *,
    silhouette: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if not spec["body"]:
        return None
    rect = _base_rect_norm(rest, spec, silhouette=None, slots=OVERLAY_BODY_SLOTS)
    if rect is None:
        rect = _base_rect_norm(rest, spec, silhouette=silhouette, slots=HEAD_SLOTS)
    return rect


def _iris_range_rects(
    k: np.ndarray, spec: dict[str, Any]
) -> list[tuple[float, float, float, float]]:
    if not spec["eyes"]:
        return []
    fx = _clip(spec["eye_x"], 0.0, 1.0)
    fy = _clip(spec["eye_y"], 0.0, 1.0)
    out: list[tuple[float, float, float, float]] = []
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
        out.append((cx - hw * fx, cy - hh * fy, cx + hw * fx, cy + hh * fy))
    return out


def draw_travel_box(
    image: Image.Image,
    rest: np.ndarray | None,
    box: Any,
    *,
    silhouette: tuple[float, float, float, float] | None = None,
    tight: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Draw size, head, and skeleton limiters on the cel overlay.

    Solid boxes are rest / current overlay. Dashed boxes are the max travel
    for form (slide) and rotation (tilted copies of the same box). Size uses
    the chroma-key silhouette of the still, padded by ``pad_px``.
    """
    spec = normalize_travel_box(box)
    if image is None or not spec["enabled"]:
        return image
    if rest is None and silhouette is None:
        return image
    base = image.convert("RGB")
    draw = ImageDraw.Draw(base)
    size = size_rect_norm(rest, spec, silhouette=silhouette)
    if tight is not None:
        _draw_rect(
            draw,
            base.size,
            tight,
            SIZE_COLOR,
            top=True,
            bottom=True,
            sides=True,
            dashed=True,
        )
    if size is not None:
        _draw_rect(
            draw,
            base.size,
            size,
            SIZE_COLOR,
            top=True,
            bottom=True,
            sides=True,
            dashed=True,
            outset=0.0,
        )
    k = _as37(rest) if rest is not None else None
    head = head_mesh_rect_norm(rest, spec)
    if head is not None and spec["side"]:
        _draw_rect(
            draw,
            base.size,
            head,
            BOX_COLOR,
            top=True,
            bottom=True,
            sides=True,
        )
        walls = head_rect_norm(rest, spec)
        if walls is not None and not _rects_close(head, walls):
            _draw_rect(
                draw,
                base.size,
                walls,
                BOX_COLOR,
                top=True,
                bottom=True,
                sides=True,
                dashed=True,
            )
        origin = None
        if k is not None:
            c = face_center(k)
            origin = (float(c[0]), float(c[1]))
        if origin is not None and (abs(origin[0]) > 1e-6 or abs(origin[1]) > 1e-6):
            if spec["look_up"]:
                _draw_rotated_rect(
                    draw, base.size, head, origin, -float(spec["pitch_up"]), BOX_COLOR
                )
            if spec["look_down"]:
                _draw_rotated_rect(
                    draw, base.size, head, origin, float(spec["pitch_down"]), BOX_COLOR
                )
        if k is not None:
            for rect in _iris_range_rects(k, spec):
                _draw_rect(
                    draw,
                    base.size,
                    rect,
                    BOX_COLOR,
                    top=True,
                    bottom=True,
                    sides=True,
                    dashed=True,
                )
    body_tight = _body_mesh_rect_norm(rest, spec, silhouette=silhouette)
    body = body_rect_norm(rest, spec, silhouette=silhouette)
    if body_tight is not None:
        _draw_rect(
            draw,
            base.size,
            body_tight,
            BODY_BOX_COLOR,
            top=True,
            bottom=True,
            sides=True,
        )
    if body is not None and not _rects_close(body_tight, body):
        _draw_rect(
            draw,
            base.size,
            body,
            BODY_BOX_COLOR,
            top=True,
            bottom=True,
            sides=True,
            dashed=True,
        )
    if body_tight is not None and spec["body_rotate"] and k is not None:
        origin = _origin(k, NECK, CHEST)
        if origin is not None:
            _draw_rotated_rect(
                draw,
                base.size,
                body_tight,
                origin,
                float(spec["body_yaw"]),
                BODY_BOX_COLOR,
            )
            _draw_rotated_rect(
                draw,
                base.size,
                body_tight,
                origin,
                -float(spec["body_roll"]),
                BODY_BOX_COLOR,
            )
    return base
