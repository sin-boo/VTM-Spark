"""Bend the drawn body to the torso points Track Lab sent.

The model draws the body where the still has it, whatever the torso points
say: Track Lab turns, leans and tilts the torso with the head
(track_lab/backend/skeleton.py) and only this makes the picture follow.

The turn is read back from the shoulders: Track Lab turns the torso on an
oval round the neck base (turn_on_oval), so how unevenly the shoulders sit
round the neck gives the turn whatever the size, walk or limiter shift.
The drawn body is wrapped on that oval and turned (the far side closes, the
near side opens, the front stays under the neck); the tilt, lean, hanging
arms and limiter shift go on top. The face and hair are left as drawn.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image

NECK, R_SHOULDER, R_ELBOW, L_SHOULDER, L_ELBOW, CHEST = 31, 32, 33, 34, 35, 36
TORSO = (NECK, R_SHOULDER, R_ELBOW, L_SHOULDER, L_ELBOW, CHEST)
CHIN = 2
JAW = (0, 1, 2, 3, 4)
# Track Lab's oval (track_lab/backend/skeleton.py BODY_HALF / BODY_DEPTH and
# its body turn cap); keep these in step with it.
BODY_HALF = 1.35
BODY_DEPTH = 0.40
MAX_TURN = math.radians(45.0)
# The field is built at 1/FIELD_STEP size and scaled up: it is smooth.
FIELD_STEP = 4
# Hair outlines stop short of the drawn tips; the head mask grows this much.
_HEAD_GROW = 0.035
_STILL_PX = 0.5


def _px(kps: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.asarray(kps, dtype=np.float64)[:, :2].copy()
    out[:, 0] = (out[:, 0] + 1.0) * 0.5 * w
    out[:, 1] = (out[:, 1] + 1.0) * 0.5 * h
    return out


def _seen(kps: np.ndarray, idx: int) -> bool:
    return float(kps[idx, 3]) >= 0.5


def body_oval(rest_px: np.ndarray) -> tuple[float, float]:
    """Half width and depth of the oval, from the rest torso (pixels)."""
    reach = max(abs(float(rest_px[j, 0] - rest_px[NECK, 0])) for j in TORSO)
    half = BODY_HALF * max(reach, 1.0)
    return half, BODY_DEPTH * half


def turn_on_oval(x: np.ndarray, half: float, depth: float, turn: float) -> np.ndarray:
    """Track Lab's turn_on_oval: ``x`` from the neck base, turned."""
    th = np.arcsin(np.clip(np.asarray(x, dtype=np.float64) / half, -1.0, 1.0))
    return half * np.sin(th) * math.cos(turn) + depth * (np.cos(th) - 1.0) * math.sin(turn)


def turn_back(xo: np.ndarray, half: float, depth: float, turn: float) -> np.ndarray:
    """Where on the unturned oval a turned ``xo`` (from the neck) came from.

    Past the oval's edges it carries on with the edge's own shift; pinning
    to the edge smeared the sleeves sideways across the background.
    """
    xo = np.asarray(xo, dtype=np.float64)
    amp = math.hypot(half * math.cos(turn), depth * math.sin(turn))
    lead = math.atan2(depth * math.sin(turn), half * math.cos(turn))
    rel = xo + depth * math.sin(turn)
    th = np.clip(np.arcsin(np.clip(rel / amp, -1.0, 1.0)) - lead, -math.pi / 2, math.pi / 2)
    src = half * np.sin(th)
    for side in (1.0, -1.0):
        edge_out = side * amp - depth * math.sin(turn)
        th_e = min(max(side * math.pi / 2 - lead, -math.pi / 2), math.pi / 2)
        src = np.where(side * rel > amp, xo - (edge_out - half * math.sin(th_e)), src)
    return src


def body_turn(kps_px: np.ndarray, rest_px: np.ndarray) -> float:
    """The torso's turn (radians, + toward screen-right), from the shoulders.

    Untilted, each shoulder sits at half*k*cos(turn) + depth*(c - 1)*sin(turn)
    from the neck (k, c: its sin / cos on the rest oval, times the size):
    two shoulders, two unknowns, the size drops out.
    """
    rest_line = rest_px[L_SHOULDER] - rest_px[R_SHOULDER]
    line = kps_px[L_SHOULDER] - kps_px[R_SHOULDER]
    tilt = math.atan2(line[1], line[0]) - math.atan2(rest_line[1], rest_line[0])
    c, s = math.cos(-tilt), math.sin(-tilt)
    half, depth = body_oval(rest_px)
    rows = []
    rhs = []
    for idx in (R_SHOULDER, L_SHOULDER):
        off = kps_px[idx] - kps_px[NECK]
        rhs.append(c * off[0] - s * off[1])
        k = float(np.clip((rest_px[idx, 0] - rest_px[NECK, 0]) / half, -1.0, 1.0))
        rows.append([half * k, depth * (math.sqrt(1.0 - k * k) - 1.0)])
    try:
        size_cos, size_sin = np.linalg.solve(np.array(rows), np.array(rhs))
    except np.linalg.LinAlgError:
        return 0.0
    if size_cos <= 0.0:
        return 0.0
    return float(np.clip(math.atan2(size_sin, size_cos), -MAX_TURN, MAX_TURN))


@lru_cache(maxsize=4)
def _grids(w: int, h: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pixel grids, full size and field size (cell centres). Building the
    full one each frame cost more than the whole warp; read only."""
    full_y, full_x = np.mgrid[0:h, 0:w].astype(np.float32)
    ys, xs = np.mgrid[0 : h // step, 0 : w // step].astype(np.float64)
    out = (full_x, full_y, (xs + 0.5) * step, (ys + 0.5) * step)
    for grid in out:
        grid.setflags(write=False)
    return out


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _head_mask(kps_px: np.ndarray, hair_px: list[np.ndarray], w: int, h: int, step: int) -> np.ndarray:
    """1 over the face (jaw closed up to the top) and the hair, feathered."""
    sw, sh = w // step, h // step
    mask = np.zeros((sh, sw), np.uint8)
    jaw = [kps_px[i] / step for i in JAW]
    poly = np.array([[jaw[0][0], 0.0]] + [list(p) for p in jaw] + [[jaw[-1][0], 0.0]], np.int32)
    cv2.fillPoly(mask, [poly], 255)
    for pts in hair_px:
        if len(pts) >= 3:
            cv2.fillPoly(mask, [(pts / step).astype(np.int32)], 255)
    grow = max(1, int(round(_HEAD_GROW * w / step)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), max(1.0, 4.0 / step))


def warp_body(
    image: Image.Image,
    kps: np.ndarray | None,
    rest: np.ndarray | None,
    hair: list[dict[str, Any]] | None = None,
) -> Image.Image:
    """``image`` with its body bent from ``rest`` (where the model draws it)
    to ``kps`` (where Track Lab put it). Both (37, 4) norm_crop. Returned as
    is when there is no torso to follow or it has not moved."""
    if kps is None or rest is None:
        return image
    kps = np.asarray(kps, dtype=np.float32)
    rest = np.asarray(rest, dtype=np.float32)
    if kps.shape[0] < 37 or rest.shape[0] < 37:
        return image
    if not all(_seen(kps, j) and _seen(rest, j) for j in TORSO + (CHIN,)):
        return image
    w, h = image.size
    live = _px(kps, w, h)
    still = _px(rest, w, h)
    turn = body_turn(live, still)
    half, depth = body_oval(still)
    torso = list(TORSO)
    turned = still[torso].copy()
    turned[:, 0] = still[NECK, 0] + turn_on_oval(still[torso, 0] - still[NECK, 0], half, depth, turn)
    moves = live[torso] - turned
    if abs(turn) < math.radians(0.2) and float(np.abs(moves).max()) < _STILL_PX:
        return image

    step = FIELD_STEP
    chin = float(live[CHIN, 1])
    collar = float(live[NECK, 1])
    # Nothing above the chin line moves: work from two cells above it down.
    row0 = int(min(max((chin - 2 * step) // step, 0), h // step - 1))
    top = row0 * step
    full_x, full_y, xs, ys = _grids(w, h, step)
    full_x, full_y, xs, ys = full_x[top:], full_y[top:], xs[row0:], ys[row0:]
    # Tilt, lean, arms and the limiter's shift: each joint's move after the
    # turn, blended by distance where it now is.
    num_x = np.zeros_like(xs)
    num_y = np.zeros_like(xs)
    den = np.zeros_like(xs)
    for (px, py), (dx, dy) in zip(live[torso], moves):
        wgt = 1.0 / (np.square(xs - px) + np.square(ys - py) + 400.0)
        num_x += wgt * dx
        num_y += wgt * dy
        den += wgt
    ramp = _smooth((ys - chin) / max(collar - chin, 8.0))
    # The body runs off the bottom of the frame: lifting it there opened a strip.
    edge = np.clip((h - 1 - ys) / (0.125 * h), 0.0, 1.0)
    mx = xs - num_x / den * ramp
    my = np.clip(ys - num_y / den * ramp * edge, 0.0, h - 1.0)
    # Then undo the turn, from the chin line down. The oval's front barely
    # moves, so the neck stays; starting at the collar bent the shoulder tips.
    src = still[NECK, 0] + turn_back(mx - still[NECK, 0], half, depth, turn)
    mx = mx + ramp * (src - mx)
    hair_px = []
    for seg in hair or []:
        poly = np.asarray(seg.get("polygon") or [], dtype=np.float64)
        if poly.ndim == 2 and len(poly) >= 3:
            hair_px.append(np.stack([(poly[:, 0] + 1.0) * 0.5 * w, (poly[:, 1] + 1.0) * 0.5 * h], axis=1))
    keep = 1.0 - _head_mask(live, hair_px, w, h, step)[row0:]
    off_x = (keep * (mx - xs)).astype(np.float32)
    off_y = (keep * (my - ys)).astype(np.float32)
    # Scale the field up: the offsets, not the coordinates, are interpolated.
    size = (w, h - top)
    off_x = cv2.resize(off_x, size, interpolation=cv2.INTER_LINEAR)
    off_y = cv2.resize(off_y, size, interpolation=cv2.INTER_LINEAR)
    rgb = np.asarray(image.convert("RGB"))
    out = rgb.copy()
    out[top:] = cv2.remap(rgb, full_x + off_x, full_y + off_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out)
