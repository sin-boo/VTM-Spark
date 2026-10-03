"""Manual six-point upper-body skeleton for the character."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import cv2
import numpy as np

# 31 neck, 32 R shoulder, 33 R elbow,
# 34 L shoulder, 35 L elbow, 36 chest. Slot 30 (body nose) is unused.
PARENT = {32: 31, 33: 32, 34: 31, 35: 34, 36: 31}
CHAIN = (32, 34, 36, 33, 35)
SKELETON_IDS = (31, 32, 33, 34, 35, 36)
# Without a locked rig the torso shares the face place (walk and size) and
# takes part of a tilt. With one it turns in 3D with the head: see
# _follow_with_rig.
_TORSO_ROLL = 0.45
# How far each joint sits in front (+) of the shoulder line, in shoulder
# widths: the neck base a little, the chest well in front, the elbows hang
# a little behind.
TORSO_DEPTH = {31: 0.08, 32: 0.0, 34: 0.0, 33: -0.05, 35: -0.05, 36: 0.28}
# The body turns on an oval round the neck base: as wide as the widest rest
# joint from the neck times BODY_HALF, BODY_DEPTH of that deep. The desk
# bends the drawn body on the same oval (backend/body_warp.py); keep these
# in step with it.
BODY_HALF = 1.35
BODY_DEPTH = 0.40
_FOCAL = 8.0  # weak perspective, in shoulder widths
# Past these the shoulders read as folding (cos 45 deg = 0.71 of their
# width) and a lean hides the arms.
_MAX_BODY_TURN = math.radians(45.0)
_MAX_BODY_LEAN = math.radians(30.0)
_MAX_BODY_TILT = math.radians(40.0)
# Upper arms swing this share of the way back to hanging after the
# shoulders turn, over about ARM_EASE_S.
ARM_GRAVITY = 0.8
ARM_EASE_S = 0.12
ARMS = ((32, 33), (34, 35))  # shoulder, elbow
_SHOULDER_FROM_MOUTH = 0.50
_SHOULDER_HALF = 0.58
_ELBOW_OUT = 0.08
_ELBOW_DOWN = 0.56
_CHEST_DOWN = 0.29
_NECK_ALONG = 0.68
_HSV_GREEN_LO = (35, 40, 40)
_HSV_GREEN_HI = (90, 255, 255)
_CHROMA_MIN = 0.12
_CHROMA_MAX = 0.88
_SHOULDER_INSET = 0.08
_ELBOW_ALONG = 0.82
_ELBOW_INSET = 0.18
_CHEST_ALONG = 0.45
_NECK_TO_SHOULDER = 0.32


def _xy(joint: dict[str, Any]) -> np.ndarray:
    return np.array([float(joint["x"]), float(joint["y"])], dtype=np.float32)


def _angle(parent: np.ndarray, child: np.ndarray) -> float:
    delta = child - parent
    return float(np.arctan2(float(delta[1]), float(delta[0])))


def _length(parent: np.ndarray, child: np.ndarray) -> float:
    return float(max(np.linalg.norm(child - parent), 1e-4))


def _pack_joints(
    neck: np.ndarray,
    right_shoulder: np.ndarray,
    right_elbow: np.ndarray,
    left_shoulder: np.ndarray,
    left_elbow: np.ndarray,
    chest: np.ndarray,
) -> list[dict[str, Any]]:
    rows = (
        (31, "neck", neck),
        (32, "right_shoulder", right_shoulder),
        (33, "right_elbow", right_elbow),
        (34, "left_shoulder", left_shoulder),
        (35, "left_elbow", left_elbow),
        (36, "chest", chest),
    )
    return [
        {
            "id": idx,
            "name": name,
            "x": round(float(point[0]), 1),
            "y": round(float(point[1]), 1),
            "score": 1.0,
        }
        for idx, name, point in rows
    ]


def skeleton_from_face(pts: np.ndarray) -> list[dict[str, Any]]:
    """Create an editable initial skeleton below the character face."""
    if pts is None or len(pts) < 17:
        return []
    face = np.asarray(pts, dtype=np.float32)
    chin = face[2, :2]
    left = face[0, :2]
    right = face[4, :2]
    mouth = face[21, :2] if len(face) > 21 else chin
    mid = 0.5 * (left + right)
    face_w = max(float(np.linalg.norm(right - left)), 36.0)
    # Shoulders sit on the body, not on nose→chin. A longer anime jaw must
    # not drag the deltoids down.
    shoulder_y = float(mouth[1] + _SHOULDER_FROM_MOUTH * face_w)
    gap = max(shoulder_y - float(chin[1]), 0.08 * face_w)
    # Collar sits near the shoulders, not under the jaw.
    neck = np.array(
        [float(chin[0]) * 0.65 + float(mid[0]) * 0.35, float(chin[1]) + _NECK_ALONG * gap],
        dtype=np.float32,
    )
    half = _SHOULDER_HALF * face_w
    right_shoulder = np.array([float(mid[0] - half), shoulder_y], dtype=np.float32)
    left_shoulder = np.array([float(mid[0] + half), shoulder_y], dtype=np.float32)
    right_elbow = right_shoulder + np.array(
        [-_ELBOW_OUT * face_w, _ELBOW_DOWN * face_w], dtype=np.float32
    )
    left_elbow = left_shoulder + np.array(
        [_ELBOW_OUT * face_w, _ELBOW_DOWN * face_w], dtype=np.float32
    )
    chest = np.array([float(mid[0]), shoulder_y + _CHEST_DOWN * face_w], dtype=np.float32)
    return _pack_joints(neck, right_shoulder, right_elbow, left_shoulder, left_elbow, chest)


def _fg_mask(bgr: np.ndarray) -> np.ndarray | None:
    """Character pixels on a green-screen still, or None when chroma is missing."""
    if bgr is None or bgr.ndim != 3 or bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, _HSV_GREEN_LO, _HSV_GREEN_HI)
    frac = float(np.mean(green > 0))
    if frac < _CHROMA_MIN or frac > _CHROMA_MAX:
        return None
    fg = np.where(green == 0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)


def _row_span(fg: np.ndarray, y: int) -> tuple[int, int, int] | None:
    xs = np.flatnonzero(fg[int(y)] > 0)
    if xs.size < 2:
        return None
    x0 = int(xs[0])
    x1 = int(xs[-1])
    return x0, x1, x1 - x0


def _torso_from_chroma(pts: np.ndarray, bgr: np.ndarray) -> list[dict[str, Any]] | None:
    """Place neck/shoulders/chest/elbows on the bust silhouette."""
    fg = _fg_mask(bgr)
    if fg is None:
        return None
    face = np.asarray(pts, dtype=np.float32)
    if face.ndim != 2 or len(face) < 5:
        return None
    h, w = fg.shape[:2]
    chin = face[2, :2]
    mid_x = float(0.5 * (float(face[0, 0]) + float(face[4, 0])))
    chin_y = int(round(float(chin[1])))
    y0 = min(max(chin_y + 4, 0), h - 2)
    spans: list[tuple[int, int, int, int]] = []
    for y in range(y0, h):
        hit = _row_span(fg, y)
        if hit is None:
            continue
        spans.append((y, hit[0], hit[1], hit[2]))
    if len(spans) < 12:
        return None
    # Neck is the narrowest row under the chin. Shoulders are the widest row
    # of the upper torso. A 2.55× neck rule drops any character whose collar
    # or hair is already wide, and the face-width fallback is then too small.
    neck_end = min(len(spans) - 1, max(6, int(0.35 * len(spans))))
    neck_i = min(range(neck_end + 1), key=lambda i: spans[i][3])
    neck_y, neck_x0, neck_x1, neck_span = spans[neck_i]
    if neck_span < 8:
        return None
    bust_end = min(len(spans) - 1, max(neck_i + 1, int(0.55 * len(spans))))
    sh_i = max(range(neck_i, bust_end + 1), key=lambda i: spans[i][3])
    sh_y, sh_x0, sh_x1, sh_span = spans[sh_i]
    if sh_y <= neck_y or sh_span < 8:
        return None
    sh_half = 0.5 * float(sh_span)
    inset = _SHOULDER_INSET * sh_half
    right_shoulder = np.array([float(sh_x0) + inset, float(sh_y)], dtype=np.float32)
    left_shoulder = np.array([float(sh_x1) - inset, float(sh_y)], dtype=np.float32)
    body_bottom = int(spans[-1][0])
    elbow_y = int(round(float(sh_y) + _ELBOW_ALONG * (body_bottom - float(sh_y))))
    elbow_y = min(max(elbow_y, sh_y + 8), body_bottom)
    hit = _row_span(fg, elbow_y) or (int(spans[-1][1]), int(spans[-1][2]), int(spans[-1][3]))
    el_x0, el_x1, _el_span = hit[0], hit[1], hit[2]
    center = 0.5 * (float(el_x0) + float(el_x1))
    right_elbow = np.array(
        [float(el_x0) + _ELBOW_INSET * (center - float(el_x0)), float(elbow_y)],
        dtype=np.float32,
    )
    left_elbow = np.array(
        [float(el_x1) - _ELBOW_INSET * (float(el_x1) - center), float(elbow_y)],
        dtype=np.float32,
    )
    chest_y = float(sh_y) + _CHEST_ALONG * (float(elbow_y) - float(sh_y))
    chest = np.array([mid_x, chest_y], dtype=np.float32)
    neck = np.array(
        [
            float(chin[0]) * 0.55 + 0.45 * 0.5 * (float(neck_x0) + float(neck_x1)),
            float(neck_y) + _NECK_TO_SHOULDER * (float(sh_y) - float(neck_y)),
        ],
        dtype=np.float32,
    )
    if not (float(neck[1]) < float(sh_y) < float(chest[1]) < float(elbow_y)):
        return None
    return _pack_joints(neck, right_shoulder, right_elbow, left_shoulder, left_elbow, chest)


def skeleton_from_still(pts: np.ndarray, bgr: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Rest skeleton from the bust pixels when chroma exists; face width otherwise."""
    fitted = _torso_from_chroma(pts, bgr) if bgr is not None else None
    return fitted if fitted else skeleton_from_face(pts)


def _tilt_offset(off: np.ndarray, roll: float) -> np.ndarray:
    """Rotate a joint around the neck in the picture. No yaw or pitch."""
    if abs(roll) < 1e-8:
        return np.asarray(off, dtype=np.float32)
    c = math.cos(roll)
    s = math.sin(roll)
    x = float(off[0])
    y = float(off[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def _face_xform(point: np.ndarray, place: dict[str, float] | None) -> np.ndarray:
    """Scale and slide the torso with the face."""
    if not place:
        return point
    cx = float(place.get("cx", 0.0))
    cy = float(place.get("cy", 0.0))
    scale = float(place.get("scale", 1.0))
    dx = float(place.get("dx", 0.0))
    dy = float(place.get("dy", 0.0))
    return np.array(
        [
            cx + dx + scale * (float(point[0]) - cx),
            cy + dy + scale * (float(point[1]) - cy),
        ],
        dtype=np.float32,
    )


def _torso_roll(head_roll_deg: float) -> float:
    """The share of a head tilt the shoulders take, in radians."""
    return math.radians(float(np.clip(head_roll_deg, -25.0, 25.0))) * _TORSO_ROLL


def body_oval(rest_by: dict[int, Any]) -> tuple[float, float]:
    """Half width and depth of the body's oval, from the rest joints."""
    neck = _xy(rest_by[31])
    reach = max(abs(float(_xy(j)[0] - neck[0])) for idx, j in rest_by.items() if idx in SKELETON_IDS)
    half = BODY_HALF * max(reach, 1.0)
    return half, BODY_DEPTH * half


def turn_on_oval(x: float, half: float, depth: float, turn: float) -> float:
    """A point ``x`` from the neck base on the front of the oval, turned.

    The neck base stays: the front of the chest barely moves, the far side
    closes toward its edge and the near side opens out.
    """
    th = math.asin(max(-1.0, min(1.0, float(x) / half)))
    return (
        half * math.sin(th) * math.cos(turn)
        + depth * math.cos(th) * math.sin(turn)
        - depth * math.sin(turn)
    )


class ArmEase:
    """Soft gravity: each elbow eases to where it hangs over ~ARM_EASE_S."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._dirs: dict[int, np.ndarray] = {}
        self._t: float | None = None

    def reset(self) -> None:
        self._dirs = {}
        self._t = None

    def step(self, targets: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        now = float(self._clock())
        dt = 0.0 if self._t is None else min(max(now - self._t, 0.0), 0.25)
        self._t = now
        keep = math.exp(-dt / ARM_EASE_S)
        out: dict[int, np.ndarray] = {}
        for idx, target in targets.items():
            prev = self._dirs.get(idx)
            cur = target if prev is None else keep * prev + (1.0 - keep) * target
            cur = cur / max(float(np.linalg.norm(cur)), 1e-6)
            self._dirs[idx] = cur
            out[idx] = cur
        return out


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-6)


def _follow_with_rig(
    rest: list[dict[str, Any]],
    rig: Any,
    *,
    share: float = 1.5,
    gravity: float = ARM_GRAVITY,
    arms: ArmEase | None = None,
) -> list[dict[str, Any]]:
    """The torso walks and grows with the body and turns in 3D with the head.

    Joint 31 is the base of the neck, on the torso. It walks and sizes with
    the body and stays the pivot: a turn swings the head round it while the
    torso takes ``share`` of the turn on its oval (turn_on_oval), of the nod
    as a lean and of the tilt, all round the neck base. Riding the head's
    slide dragged the shoulders down on every look-down and sideways on
    every turn. The upper arms then swing back toward hanging (``gravity``),
    eased by ``arms``: rigid on a turned torso, a tilt swung an elbow up.
    """
    place = rig.place()
    cx = float(place["cx"])
    cy = float(place["cy"])
    scale = float(place.get("scale", 1.0))
    joints = [joint for joint in rest if int(joint["id"]) in SKELETON_IDS]
    if not joints:
        return []
    rest_by = {int(joint["id"]): joint for joint in joints}
    neck = rest_by.get(31)
    if neck is None:
        return [dict(joint) for joint in joints]
    neck_rest = _xy(neck)
    neck_xy = np.array(
        [
            cx + float(place.get("body_dx", 0.0)) + scale * (float(neck_rest[0]) - cx),
            cy + float(place.get("body_dy", 0.0)) + scale * (float(neck_rest[1]) - cy),
        ],
        dtype=np.float32,
    )
    head = rig.turn()
    turn = max(-_MAX_BODY_TURN, min(_MAX_BODY_TURN, share * float(head["yaw"])))
    lean = max(-_MAX_BODY_LEAN, min(_MAX_BODY_LEAN, share * float(head["pitch"])))
    roll = max(-_MAX_BODY_TILT, min(_MAX_BODY_TILT, share * float(head["roll"])))
    span = 1.0
    if 32 in rest_by and 34 in rest_by:
        span = float(np.linalg.norm(_xy(rest_by[32]) - _xy(rest_by[34])))
    width = max(span, 1.0) * scale
    half, depth = body_oval(rest_by)
    half, depth = half * scale, depth * scale
    cl, sl = math.cos(lean), math.sin(lean)
    cr, sr = math.cos(roll), math.sin(roll)
    def posed(idx: int, off: np.ndarray) -> np.ndarray:
        x = turn_on_oval(float(off[0]), half, depth, turn)
        y = float(off[1])
        z0 = TORSO_DEPTH.get(idx, 0.0) * width
        # Lean: a look down brings the top toward the camera.
        y, z = cl * y + sl * z0, -sl * y + cl * z0
        x, y = cr * x - sr * y, sr * x + cr * y
        # Only a change of depth changes the size: the still already shows
        # the chest in front of the shoulders.
        persp = (_FOCAL * width - z0) / max(_FOCAL * width - z, 1e-3)
        return persp * np.array([x, y], dtype=np.float32)

    # The neck base is the pivot exactly: in front of the shoulder line, a
    # lean alone would have carried it.
    pivot = posed(31, np.zeros(2, dtype=np.float32))
    placed: dict[int, np.ndarray] = {}
    for idx, joint in rest_by.items():
        placed[idx] = neck_xy + posed(idx, (_xy(joint) - neck_rest) * scale) - pivot
    targets: dict[int, np.ndarray] = {}
    for sh, el in ARMS:
        if sh in placed and el in placed:
            arm = placed[el] - placed[sh]
            hang = _unit(_xy(rest_by[el]) - _xy(rest_by[sh]))
            targets[el] = _unit((1.0 - gravity) * _unit(arm) + gravity * hang)
    eased = arms.step(targets) if arms is not None else targets
    for sh, el in ARMS:
        if el in eased:
            # A hanging arm hangs across the view: gravity brings back its
            # length too, or a lean foreshortened it into the shoulder.
            turned = float(np.linalg.norm(placed[el] - placed[sh]))
            hanging = float(np.linalg.norm(_xy(rest_by[el]) - _xy(rest_by[sh]))) * scale
            length = (1.0 - gravity) * turned + gravity * hanging
            placed[el] = placed[sh] + length * eased[el]
    out: list[dict[str, Any]] = []
    for joint in joints:
        point = placed[int(joint["id"])]
        record = dict(joint)
        record["x"] = round(float(point[0]), 1)
        record["y"] = round(float(point[1]), 1)
        out.append(record)
    return out


def follow_skeleton(
    rest: list[dict[str, Any]],
    live_face: np.ndarray | None,
    cam_rest: list[dict[str, Any]] | None = None,
    cam_live: list[dict[str, Any]] | None = None,
    head: dict[str, float] | None = None,
    place: dict[str, float] | None = None,
    rig: Any = None,
    *,
    share: float = 1.5,
    arms: ArmEase | None = None,
) -> list[dict[str, Any]]:
    """Parent the manual skeleton to the face. Ignore camera body pose.

    ``share``: of the head's turn / nod / tilt the torso takes (feel
    body_turn); ``arms`` eases the elbows' gravity over time.
    """
    del live_face, cam_rest, cam_live
    if not rest:
        return []
    if rig is not None and getattr(rig, "locked", False):
        return _follow_with_rig(rest, rig, share=share, arms=arms)
    rest_by = {int(joint["id"]): joint for joint in rest}
    neck = rest_by.get(31)
    if neck is None:
        return [dict(joint) for joint in rest]
    neck_xy = _face_xform(_xy(neck), place)
    roll = _torso_roll(head.get("roll", 0.0)) if head else 0.0
    posed = {31: neck_xy}
    for idx, joint in rest_by.items():
        if idx != 31:
            posed[idx] = neck_xy + _tilt_offset(
                _face_xform(_xy(joint), place) - neck_xy, roll
            )
    placed = {31: neck_xy}
    for child in CHAIN:
        parent = PARENT[child]
        if child not in posed or parent not in placed:
            continue
        length = _length(posed[parent], posed[child])
        angle = _angle(posed[parent], posed[child])
        placed[child] = placed[parent] + length * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
    out: list[dict[str, Any]] = []
    for joint in rest:
        idx = int(joint["id"])
        if idx not in SKELETON_IDS:
            continue
        record = dict(joint)
        point = placed.get(idx, _xy(joint))
        record["x"] = round(float(point[0]), 1)
        record["y"] = round(float(point[1]), 1)
        out.append(record)
    return out


def merge_skeleton(
    detected: list[dict[str, Any]], face_pts: np.ndarray | None
) -> list[dict[str, Any]]:
    """Return the manual skeleton; legacy detections are intentionally ignored."""
    del detected
    return skeleton_from_face(face_pts) if face_pts is not None else []
