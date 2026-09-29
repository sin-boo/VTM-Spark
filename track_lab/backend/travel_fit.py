"""Fit the limiters to a character still.

Walking room comes from how the character is framed: the free space between
its rest head and body and the picture's edges. Turn and tilt keep ``base``'s
amount (the mean of its two sides) but centre it on the pose the art is drawn
in: a still drawn turned 8 deg to the right has 8 deg less room to turn right
and 8 deg more to the left, so the view never goes further from straight on
than a frontal still allows. Look up / down, eye range, size and On come from
``base`` unchanged.

The margins reproduce limiters tuned by hand on a green-screen bust (face
about a fifth of the picture high); other framings get the same margins to
the picture's edges.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .travel_box import (
    BODY_ANCHOR,
    HEAD_SLOTS,
    L_EYE,
    OVERLAY_BODY_SLOTS,
    R_EYE,
    ROLL_MAX_DEG,
    YAW_MAX_DEG,
    _as37,
    _bbox,
    _vis,
    face_height,
    normalize_travel_box,
)

# Margins, in face heights. The face box keeps this far from the side edges;
# the chin this far above the bottom edge, so the neck and shoulders stay in.
SIDE_MARGIN = 0.95
BELOW_MARGIN = 1.1
# Brows to the top edge when the top of the head is unknown (no green
# screen): the forehead and hair above them.
ABOVE_MARGIN = 1.75
# How far the top of the head (hair, ears, a hat) may go out of the top edge.
TOP_CROP = 0.3
# Torso points keep this far inside the side / bottom edges.
BODY_SIDE_MARGIN = 0.1
BODY_BELOW_MARGIN = 0.02
# A torso cut off by the bottom edge may only rise this much: above that the
# model has to draw body the still never showed.
CUT_BODY_RISE = 0.1
# Without a green screen: a torso whose lowest point is this close to the
# bottom edge carries on past it.
CUT_BODY_REACH = 1.5
# Never frozen: a close-up still gets a little room to move.
ROOM_MIN = 0.1
ROOM_MAX = 1.2

# Drawn turn / tilt smaller than this is noise in the points, not the art.
DRAWN_DEADBAND = 3.0
DRAWN_MAX = 25.0
TURN_MIN = 5.0

MOUTH = tuple(range(20, 28))
CHIN = 2
JAW_LEFT = 0
JAW_RIGHT = 4


def _eye_center(k: np.ndarray, slots: tuple[int, ...]) -> np.ndarray | None:
    pts = [k[i, :2] for i in slots if _vis(k, i)]
    return None if not pts else np.mean(np.stack(pts), axis=0)


def _line_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Screen angle from the left point to the right one; + is clockwise."""
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    return math.degrees(math.atan2(float(right[1] - left[1]), float(right[0] - left[0])))


def drawn_pose(rest: np.ndarray) -> dict[str, float]:
    """The turn and tilt the still is drawn in, degrees, rig signs.

    Tilt is the slant of the eye line and the jaw (+ is clockwise, the crown
    toward screen-right). Turn is how far the face's middle (chin and mouth)
    sits off the middle of the cheeks: 0 on a frontal face, and + when the
    face looks toward screen-right. Both are 0 when the points are missing.
    """
    k = _as37(rest)
    out = {"yaw": 0.0, "roll": 0.0}
    slants: list[float] = []
    eye_a = _eye_center(k, L_EYE)
    eye_b = _eye_center(k, R_EYE)
    if eye_a is not None and eye_b is not None:
        slants.append(_line_deg(eye_a, eye_b))
    if _vis(k, JAW_LEFT) and _vis(k, JAW_RIGHT):
        slants.append(_line_deg(k[JAW_LEFT, :2], k[JAW_RIGHT, :2]))
    roll = float(np.mean(slants)) if slants else 0.0
    out["roll"] = roll
    mid_pts = [k[i, :2] for i in (CHIN, *MOUTH) if _vis(k, i)]
    if not (mid_pts and _vis(k, JAW_LEFT) and _vis(k, JAW_RIGHT)):
        return out
    # Measure along the eye line so a tilted still does not read as a turn.
    c, s = math.cos(math.radians(-roll)), math.sin(math.radians(-roll))

    def along(p: np.ndarray) -> float:
        return float(p[0]) * c - float(p[1]) * s

    x_left = along(k[JAW_LEFT, :2])
    x_right = along(k[JAW_RIGHT, :2])
    if x_left > x_right:
        x_left, x_right = x_right, x_left
    span = x_right - x_left
    if span < 1e-3:
        return out
    chin = along(k[CHIN, :2]) if _vis(k, CHIN) else None
    mouth = [along(k[i, :2]) for i in MOUTH if _vis(k, i)]
    mids = ([chin] if chin is not None else []) + ([float(np.mean(mouth))] if mouth else [])
    mid = float(np.mean(mids))
    lean = ((mid - x_left) - (x_right - mid)) / span
    out["yaw"] = math.degrees(math.asin(max(-0.9, min(0.9, lean))))
    return out


def _drawn(deg: float) -> float:
    if abs(deg) < DRAWN_DEADBAND:
        return 0.0
    return max(-DRAWN_MAX, min(DRAWN_MAX, deg))


def green_screen_mask(image_bgr: np.ndarray | None) -> np.ndarray | None:
    """Character pixels (True) when the still is on a green screen, else None.

    The border decides: a green-screen still is mostly green round its edges.
    """
    if image_bgr is None:
        return None
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] < 3 or min(img.shape[:2]) < 8:
        return None
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    green = (g > 40) & ((g - r) > 18) & ((g - b) > 18)
    border = np.concatenate((green[0], green[-1], green[:, 0], green[:, -1]))
    if float(np.mean(border)) < 0.5:
        return None
    return ~green


def _head_top(mask: np.ndarray, face: tuple[float, float, float, float], fh: float) -> float | None:
    """Topmost character row over the face (hair, ears, a hat), in pixels."""
    h, w = mask.shape[:2]
    x0 = max(0, int(math.floor(face[0] - 0.5 * fh)))
    x1 = min(w, int(math.ceil(face[2] + 0.5 * fh)))
    y1 = max(0, min(h, int(math.floor(face[1]))))
    if x1 <= x0 or y1 <= 0:
        return None
    # A few stray pixels (antialiasing, a loose strand) are not the head.
    rows = np.count_nonzero(mask[:y1, x0:x1], axis=1) >= 3
    hits = np.flatnonzero(rows)
    return float(hits[0]) if hits.size else None


def _body_cut(mask: np.ndarray | None, body: tuple[float, float, float, float], height: int, fh: float) -> bool:
    if mask is not None:
        bottom = mask[-2:]
        return float(np.mean(bottom)) > 0.1
    return (float(height) - body[3]) < CUT_BODY_REACH * fh


def _room(pixels: float, fh: float) -> float:
    return round(max(ROOM_MIN, min(ROOM_MAX, pixels / fh)), 2)


def fit_travel_box(
    rest: np.ndarray,
    width: int,
    height: int,
    *,
    base: Any = None,
    image_bgr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Limiters for this still. ``rest`` is its (37, 4) rest overlay in pixels."""
    k = _as37(rest)
    spec = normalize_travel_box(base)
    out = dict(spec)
    w = float(max(int(width), 1))
    h = float(max(int(height), 1))
    fh = face_height(k)
    mask = green_screen_mask(image_bgr)
    if mask is not None and mask.shape[:2] != (int(height), int(width)):
        mask = None

    face = _bbox(k, HEAD_SLOTS)
    if face is not None:
        out["left"] = _room(face[0] - SIDE_MARGIN * fh, fh)
        out["right"] = _room(w - face[2] - SIDE_MARGIN * fh, fh)
        out["down"] = _room(h - face[3] - BELOW_MARGIN * fh, fh)
        top = _head_top(mask, face, fh) if mask is not None else None
        if top is not None:
            out["up"] = _room(top + TOP_CROP * fh, fh)
        else:
            out["up"] = _room(face[1] - ABOVE_MARGIN * fh, fh)

    body = _bbox(k, OVERLAY_BODY_SLOTS)
    if body is not None and any(_vis(k, i) for i in BODY_ANCHOR):
        rise = CUT_BODY_RISE if _body_cut(mask, body, int(height), fh) else out["up"]
        fitted = {
            "left": _room(body[0] - BODY_SIDE_MARGIN * fh, fh),
            "right": _room(w - body[2] - BODY_SIDE_MARGIN * fh, fh),
            "up": rise,
            "down": _room(h - body[3] - BODY_BELOW_MARGIN * fh, fh),
        }
        # The torso goes no further than the head. Shoulders off the edge of
        # a close-up leave only the neck to measure, which read as room to
        # walk the whole way across.
        for side, room in fitted.items():
            out[f"body_{side}"] = min(room, float(out[side]))

    pose = drawn_pose(k)
    yaw = _drawn(pose["yaw"])
    roll = _drawn(pose["roll"])
    turn = 0.5 * (float(spec["turn_left"]) + float(spec["turn_right"]))
    tilt = 0.5 * (float(spec["tilt_left"]) + float(spec["tilt_right"]))
    # Right is positive: a still already facing right has less right to go.
    out["turn_left"] = round(max(TURN_MIN, min(YAW_MAX_DEG, turn + yaw)), 1)
    out["turn_right"] = round(max(TURN_MIN, min(YAW_MAX_DEG, turn - yaw)), 1)
    out["tilt_left"] = round(max(TURN_MIN, min(ROLL_MAX_DEG, tilt + roll)), 1)
    out["tilt_right"] = round(max(TURN_MIN, min(ROLL_MAX_DEG, tilt - roll)), 1)
    return normalize_travel_box(out)
