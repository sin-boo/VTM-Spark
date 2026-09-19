"""Pose controller: sanitize and overlay mesh for KEYPOINT_SCHEMA.

Merged from pose_sanitize / pose_overlay so travel clamps and mesh drawing
share one module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Shared schema
# ---------------------------------------------------------------------------

NUM_KEYPOINTS = 37
KEYPOINT_DIM = 4

FACE_SLOTS = tuple(range(0, 28))
OUTLINE = (0, 1, 2, 3, 4)
BROWS = (5, 6, 7, 8, 9, 10)
LEFT_BROW = (5, 6, 7)
RIGHT_BROW = (8, 9, 10)
L_EYE = LEFT_EYE = (11, 12, 13)
R_EYE = RIGHT_EYE = (17, 18, 19)
NOSE = (14, 15, 16)
MOUTH = (20, 21, 22, 23, 24, 25, 26, 27)
# Lip contour draw order (not raw 20..27).
MOUTH_DRAW = (20, 21, 22, 26, 27, 25, 24, 23)
MOUTH_CORNERS = (23, 26)
# Schema lip rows as (left, mid, right). 21 = upper mid, 25 = lower mid.
SCHEMA_UPPER_LIP = (20, 21, 22)
SCHEMA_LOWER_LIP = (24, 25, 27)
# Person left/right (iFacialMocap / OSF). On a mirrored selfie that is also
# image left/right. Slot 28 / IRIS.L sits in EYE.L (11-13). Training still
# calls 28 right_iris — that name is leftover from the unmirrored schema.
IRIS_L = RIGHT_IRIS = 28
IRIS_R = LEFT_IRIS = 29
IRIS_EYE_PAIRS: tuple[tuple[int, tuple[int, ...]], ...] = (
    (IRIS_L, L_EYE),
    (IRIS_R, R_EYE),
)

# Keep in sync with track_lab/harness/points.py — overlay IDs use these refs.
KEYPOINT_REFS: tuple[str, ...] = (
    "JAW.R",
    "JAW.R.MID",
    "CHIN",
    "JAW.L.MID",
    "JAW.L",
    "BROW.L.OUT",
    "BROW.L",
    "BROW.L.IN",
    "BROW.R.IN",
    "BROW.R",
    "BROW.R.OUT",
    "EYE.L.IN",
    "EYE.L.LID",
    "EYE.L.OUT",
    "NOSE.R",
    "NOSE",
    "NOSE.L",
    "EYE.R.IN",
    "EYE.R.LID",
    "EYE.R.OUT",
    "MOUTH.U.R",
    "MOUTH.U",
    "MOUTH.U.L",
    "MOUTH.R",
    "MOUTH.D.R",
    "MOUTH.D",
    "MOUTH.L",
    "MOUTH.D.L",
    "IRIS.L",
    "IRIS.R",
    "BODY.NOSE",
    "NECK",
    "SHO.R",
    "ELB.R",
    "SHO.L",
    "ELB.L",
    "CHEST",
)
KEYPOINT_NAMES: tuple[str, ...] = (
    "jaw_r",
    "jaw_r_mid",
    "chin",
    "jaw_l_mid",
    "jaw_l",
    "brow_l_outer",
    "brow_l",
    "brow_l_inner",
    "brow_r_inner",
    "brow_r",
    "brow_r_outer",
    "eye_l_inner",
    "eye_l_lid",
    "eye_l_outer",
    "nose_r",
    "nose",
    "nose_l",
    "eye_r_inner",
    "eye_r_lid",
    "eye_r_outer",
    "mouth_upper_r",
    "mouth_upper",
    "mouth_upper_l",
    "mouth_corner_r",
    "mouth_lower_r",
    "mouth_lower",
    "mouth_corner_l",
    "mouth_lower_l",
    "iris_l",
    "iris_r",
    "body_nose",
    "neck",
    "shoulder_r",
    "elbow_r",
    "shoulder_l",
    "elbow_l",
    "chest",
)
KEYPOINT_LEGACY: tuple[str, ...] = (
    *(f"face_{i}" for i in range(28)),
    "right_iris",
    "left_iris",
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "left_shoulder",
    "left_elbow",
    "chest",
)
# Legacy first, then names / refs, so "nose" stays the face tip (15), not body (30).
_SLOT_BY_TOKEN: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_LEGACY)}
_SLOT_BY_TOKEN.update({name: i for i, name in enumerate(KEYPOINT_NAMES)})
_SLOT_BY_TOKEN.update({ref: i for i, ref in enumerate(KEYPOINT_REFS)})
_SLOT_BY_TOKEN.update({ref.lower(): i for i, ref in enumerate(KEYPOINT_REFS)})
_SLOT_BY_TOKEN.update({str(i): i for i in range(len(KEYPOINT_REFS))})


def ref_of(slot: int) -> str:
    if 0 <= int(slot) < len(KEYPOINT_REFS):
        return KEYPOINT_REFS[int(slot)]
    return str(slot)


def slot_of(token: object) -> int:
    """Resolve a harness keypoint index, ref, name, or legacy name."""
    if isinstance(token, bool) or token is None:
        return -1
    if isinstance(token, int):
        return int(token) if 0 <= token < NUM_KEYPOINTS else -1
    if isinstance(token, float) and token.is_integer():
        idx = int(token)
        return idx if 0 <= idx < NUM_KEYPOINTS else -1
    text = str(token).strip()
    if not text:
        return -1
    if text in _SLOT_BY_TOKEN:
        return _SLOT_BY_TOKEN[text]
    return _SLOT_BY_TOKEN.get(text.lower(), -1)

BODY_NOSE = 30
NECK = 31
R_SHOULDER = RIGHT_SHOULDER = 32
R_ELBOW = RIGHT_ELBOW = 33
L_SHOULDER = LEFT_SHOULDER = 34
L_ELBOW = LEFT_ELBOW = 35
CHEST = 36
BODY_SLOTS = (30, 31, 32, 33, 34, 35, 36)
# Overlay skips the unused body-nose slot (30); neck is the head of the stick figure.
OVERLAY_BODY_SLOTS = (31, 32, 33, 34, 35, 36)
# Face nose tip (15) and body nose (30) stay off the cel and off the DiT.
BLOCKED_POSE_SLOTS = (15, 30)
BLOCKED_POSE_SLOT_SET = frozenset(BLOCKED_POSE_SLOTS)

# Travel caps as fractions of face height (character-relative).
# Tightened for Live2D-style constrained retarget (rig is the primary motion model).
TRAVEL_FACE = 0.14
TRAVEL_BROW = 0.12
TRAVEL_EYE = 0.16
TRAVEL_NOSE = 0.10
TRAVEL_MOUTH = 0.34
TRAVEL_IRIS = 0.08
TRAVEL_BODY = 0.22
TRAVEL_BODY_Y = 0.20
# Tracked body still slightly wider for arm swings, but no longer human-wide.
TRAVEL_BODY_TRACKED = 0.28
TRAVEL_BODY_Y_TRACKED = 0.26

# Topology margins as fractions of face height.
NECK_BELOW_CHIN = 0.12
SHOULDER_BELOW_NECK = 0.02
ELBOW_BELOW_SHOULDER = 0.02
CHEST_BELOW_NECK = 0.05
MOUTH_BELOW_NOSE = 0.01
LIP_GAP_MIN = 0.004

FACE_CENTER_BLEND = 0.85
BODY_FOLLOW_GAIN = 0.35
# When body is tracker-driven, do not drag it with face re-center.
BODY_FOLLOW_GAIN_TRACKED = 0.0

# Overlay colors (RGB).
FACE_CONNECTIONS: list[tuple[tuple[int, ...], tuple[int, int, int]]] = [
    (OUTLINE, (255, 200, 80)),
    (LEFT_BROW, (160, 255, 80)),
    (RIGHT_BROW, (160, 255, 80)),
    (NOSE, (80, 180, 255)),
    (MOUTH_DRAW, (255, 80, 180)),
]
FACE_POINT_COLOR = {
    "outline": (255, 200, 80),
    "brow": (160, 255, 80),
    "eye": (80, 120, 255),
    "nose": (80, 180, 255),
    "mouth": (255, 80, 180),
}
# Mouth overlay: purple while open/tracked, blue when near-closed snap is active.
MOUTH_COLOR = (255, 80, 180)
MOUTH_SNAPPED_COLOR = (60, 140, 255)
SKELETON_BONES = [
    (NECK, RIGHT_SHOULDER),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (NECK, LEFT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (RIGHT_SHOULDER, LEFT_SHOULDER),
    (NECK, CHEST),
]
SKELETON_COLOR = (120, 220, 80)
JOINT_COLOR = (180, 255, 40)
SKELETON_LOST_COLOR = (255, 60, 60)
JOINT_LOST_COLOR = (255, 100, 80)
# Keypoint index labels (not the dots themselves).
ID_LABEL_COLOR = (255, 40, 40)
IRIS_RIGHT = (255, 220, 0)
IRIS_LEFT = (0, 180, 255)
# RGB; matches live-poser hair overlay (those colors are BGR).
HAIR_OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "hair_middle": (255, 200, 0),
    "hair_left": (0, 180, 255),
    "hair_right": (255, 80, 160),
}
HAIR_OVERLAY_LABELS: dict[str, str] = {
    "hair_middle": "M",
    "hair_left": "L",
    "hair_right": "R",
}


@dataclass(frozen=True)
class LipRoles:
    """Which mouth slots are *visually* upper / lower on a given rest pose.

    Some characters are authored with schema lower-mid 25 parked on the
    cupid's bow above upper-mid 21. Every lip rule (snap, jaw drop, clamps)
    must act on visual roles, never on raw slot ids, or 25 gets dragged
    toward 21.
    """

    upper: tuple[int, int, int]  # (left, mid, right)
    lower: tuple[int, int, int]
    inverted: bool

    @property
    def upper_mid(self) -> int:
        return self.upper[1]

    @property
    def lower_mid(self) -> int:
        return self.lower[1]

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """(visual upper, visual lower) slot pairs, mid first."""
        return (
            (self.upper[1], self.lower[1]),
            (self.upper[0], self.lower[0]),
            (self.upper[2], self.lower[2]),
        )

    def live_slot(self, slot: int) -> int:
        """Human slot whose motion should drive this character slot."""
        if not self.inverted:
            return slot
        for a, b in zip(SCHEMA_UPPER_LIP, SCHEMA_LOWER_LIP):
            if slot == a:
                return b
            if slot == b:
                return a
        return slot


def lip_roles(rest: np.ndarray | None) -> LipRoles:
    """Read the visual lip layout from a rest pose (schema order when unknown)."""
    inverted = (
        rest is not None
        and float(rest[21, 3]) >= 0.5
        and float(rest[25, 3]) >= 0.5
        and float(rest[25, 1]) < float(rest[21, 1]) - 1e-4
    )
    if inverted:
        return LipRoles(SCHEMA_LOWER_LIP, SCHEMA_UPPER_LIP, True)
    return LipRoles(SCHEMA_UPPER_LIP, SCHEMA_LOWER_LIP, False)


def _as37(kps: np.ndarray) -> np.ndarray:
    out = np.asarray(kps, dtype=np.float32)
    if out.ndim != 2 or out.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(
            f"Expected keypoints ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {out.shape}"
        )
    return out


def _vis(k: np.ndarray, i: int) -> bool:
    return float(k[i, 3]) >= 0.5


def block_model_slots(keypoints: np.ndarray) -> np.ndarray:
    """Drop face nose 15 and body nose 30 so the DiT never sees them."""
    out = _as37(keypoints).copy()
    for idx in BLOCKED_POSE_SLOTS:
        out[idx] = 0.0
    return out


# ---------------------------------------------------------------------------
# Sanitize — travel clamp / re-center / topology
# ---------------------------------------------------------------------------


def face_height(kps: np.ndarray) -> float:
    """Rough face height from brows/eyes to chin/mouth (crop Y down = +)."""
    k = _as37(kps)
    tops = [i for i in (5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19) if _vis(k, i)]
    bots = [i for i in (2, 20, 21, 22, 25) if _vis(k, i)]
    if not tops or not bots:
        return 1.0
    return max(1e-3, float(np.mean(k[bots, 1]) - np.mean(k[tops, 1])))


def face_width(kps: np.ndarray) -> float:
    k = _as37(kps)
    left = [i for i in (0, 1, 11, 5) if _vis(k, i)]
    right = [i for i in (3, 4, 19, 10) if _vis(k, i)]
    if not left or not right:
        return 1.0
    return max(1e-3, float(np.mean(k[right, 0]) - np.mean(k[left, 0])))


def face_center(kps: np.ndarray) -> np.ndarray:
    """Mean of visible face slots 0..27."""
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


def _eye_box(k: np.ndarray, idxs: tuple[int, ...]) -> tuple[float, float, float, float] | None:
    vis = [i for i in idxs if _vis(k, i)]
    if len(vis) < 2:
        return None
    xs = [float(k[i, 0]) for i in vis]
    ys = [float(k[i, 1]) for i in vis]
    pad = 0.02
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def _clamp_point_to_ref(
    out: np.ndarray,
    ref: np.ndarray,
    idx: int,
    *,
    max_dx: float,
    max_dy: float,
) -> None:
    if not _vis(out, idx) or not _vis(ref, idx):
        return
    dx = float(out[idx, 0] - ref[idx, 0])
    dy = float(out[idx, 1] - ref[idx, 1])
    dx = float(np.clip(dx, -max_dx, max_dx))
    dy = float(np.clip(dy, -max_dy, max_dy))
    out[idx, 0] = float(ref[idx, 0] + dx)
    out[idx, 1] = float(ref[idx, 1] + dy)


def _clamp_travel(
    out: np.ndarray,
    ref: np.ndarray,
    face_h: float,
    *,
    sanitize_body: bool = True,
    body_tracked: bool = False,
    limit_face: bool = True,
    limit_brows: bool = True,
    limit_eyes: bool = True,
    limit_nose: bool = True,
    limit_mouth: bool = True,
) -> None:
    """Limit each slot's travel from the character ref by region."""
    fh = max(face_h, 1e-3)

    if limit_face:
        for i in OUTLINE:
            _clamp_point_to_ref(
                out, ref, i, max_dx=TRAVEL_FACE * fh, max_dy=TRAVEL_FACE * fh
            )
    if limit_brows:
        for i in BROWS:
            _clamp_point_to_ref(
                out, ref, i, max_dx=TRAVEL_BROW * fh, max_dy=TRAVEL_BROW * fh
            )
    if limit_eyes:
        for i in L_EYE + R_EYE:
            _clamp_point_to_ref(
                out, ref, i, max_dx=TRAVEL_EYE * fh, max_dy=TRAVEL_EYE * fh
            )
        for iris, eye in IRIS_EYE_PAIRS:
            _clamp_point_to_ref(
                out, ref, iris, max_dx=TRAVEL_IRIS * fh, max_dy=TRAVEL_IRIS * fh
            )
            box = _eye_box(out, eye)
            if box is not None and _vis(out, iris):
                x0, x1, y0, y1 = box
                out[iris, 0] = float(np.clip(out[iris, 0], x0, x1))
                out[iris, 1] = float(np.clip(out[iris, 1], y0, y1))
    if limit_nose:
        for i in NOSE:
            _clamp_point_to_ref(
                out, ref, i, max_dx=TRAVEL_NOSE * fh, max_dy=TRAVEL_NOSE * fh
            )
    if limit_mouth:
        for i in MOUTH:
            _clamp_point_to_ref(
                out, ref, i, max_dx=TRAVEL_MOUTH * fh, max_dy=TRAVEL_MOUTH * fh
            )

    if sanitize_body:
        max_dx = (TRAVEL_BODY_TRACKED if body_tracked else TRAVEL_BODY) * fh
        max_dy = (TRAVEL_BODY_Y_TRACKED if body_tracked else TRAVEL_BODY_Y) * fh
        for i in BODY_SLOTS:
            _clamp_point_to_ref(out, ref, i, max_dx=max_dx, max_dy=max_dy)


def _recenter_face(
    out: np.ndarray,
    ref: np.ndarray,
    *,
    body_tracked: bool = False,
    sanitize_body: bool = True,
) -> None:
    """Translate face (and lightly body) so face COM stays near the character ref."""
    c_out = face_center(out)
    c_ref = face_center(ref)
    drift = c_out - c_ref
    if float(np.linalg.norm(drift)) < 1e-5:
        return
    corr = -FACE_CENTER_BLEND * drift
    for i in FACE_SLOTS:
        if _vis(out, i):
            out[i, 0] = float(out[i, 0] + corr[0])
            out[i, 1] = float(out[i, 1] + corr[1])
    for i in (RIGHT_IRIS, LEFT_IRIS):
        if _vis(out, i):
            out[i, 0] = float(out[i, 0] + corr[0])
            out[i, 1] = float(out[i, 1] + corr[1])
    if not sanitize_body:
        return
    # Tracker-driven body should not be dragged by face re-centering.
    follow = BODY_FOLLOW_GAIN_TRACKED if body_tracked else BODY_FOLLOW_GAIN
    if abs(follow) < 1e-6:
        return
    body_corr = follow * corr
    for i in BODY_SLOTS:
        if _vis(out, i):
            out[i, 0] = float(out[i, 0] + body_corr[0])
            out[i, 1] = float(out[i, 1] + body_corr[1])


def _enforce_topology(
    out: np.ndarray,
    ref: np.ndarray,
    face_h: float,
    *,
    body_tracked: bool = False,
    sanitize_body: bool = True,
    limit_mouth: bool = True,
    limit_nose: bool = True,
) -> None:
    """Project points back onto anatomically allowed bands (Y down = +)."""
    fh = max(face_h, 1e-3)

    if limit_mouth or limit_nose:
        nose_y = None
        for i in (15, 14, 16, BODY_NOSE):
            if _vis(out, i):
                nose_y = float(out[i, 1])
                break

        # Lip rules act on the character's *visual* upper/lower rows.
        roles = lip_roles(ref)
        um, lm = roles.upper_mid, roles.lower_mid
        if limit_mouth and limit_nose and _vis(out, um) and nose_y is not None:
            out[um, 1] = max(float(out[um, 1]), nose_y + MOUTH_BELOW_NOSE * fh)
        if limit_mouth and _vis(out, um) and _vis(out, lm):
            out[lm, 1] = max(float(out[lm, 1]), float(out[um, 1]) + LIP_GAP_MIN * fh)
        if limit_mouth and _vis(out, um):
            uy = float(out[um, 1])
            for i in (roles.upper[0], roles.upper[2]):
                if _vis(out, i):
                    out[i, 1] = max(float(out[i, 1]), uy - 0.08 * fh)
        if limit_mouth and _vis(out, lm):
            ly = float(out[lm, 1])
            for i in (roles.lower[0], roles.lower[2]):
                if _vis(out, i) and float(out[i, 1]) < ly - 0.05 * fh:
                    out[i, 1] = ly
            uy = float(out[um, 1]) if _vis(out, um) else ly
            for i in MOUTH_CORNERS:
                if _vis(out, i):
                    out[i, 1] = float(np.clip(out[i, 1], uy, ly + 0.05 * fh))

        if limit_mouth:
            mouth_bottom = None
            for i in (*roles.lower, 2):
                if _vis(out, i):
                    y = float(out[i, 1])
                    mouth_bottom = y if mouth_bottom is None else max(mouth_bottom, y)
            if (
                _vis(out, 2)
                and mouth_bottom is not None
                and float(out[2, 1]) < mouth_bottom
            ):
                out[2, 1] = mouth_bottom

    if not sanitize_body:
        return

    cy = chin_y(out)
    if cy is None and _vis(ref, 2):
        cy = float(ref[2, 1])

    if _vis(out, NECK) and cy is not None:
        min_neck = cy + NECK_BELOW_CHIN * fh
        if float(out[NECK, 1]) < min_neck:
            dy = min_neck - float(out[NECK, 1])
            out[NECK, 1] = min_neck
            # Carry the whole upper-body chain so bone lengths stay intact.
            for i in BODY_SLOTS:
                if i == NECK or not _vis(out, i):
                    continue
                out[i, 1] = float(out[i, 1] + dy)

    if _vis(out, NECK):
        ny = float(out[NECK, 1])
        nx = float(out[NECK, 0])
        for i in (R_SHOULDER, L_SHOULDER):
            if _vis(out, i) and float(out[i, 1]) < ny + SHOULDER_BELOW_NECK * fh:
                out[i, 1] = ny + SHOULDER_BELOW_NECK * fh
        if _vis(out, CHEST) and float(out[CHEST, 1]) < ny + CHEST_BELOW_NECK * fh:
            out[CHEST, 1] = ny + CHEST_BELOW_NECK * fh
        if _vis(out, R_SHOULDER) and float(out[R_SHOULDER, 0]) > nx:
            out[R_SHOULDER, 0] = nx - 0.02 * fh
        if _vis(out, L_SHOULDER) and float(out[L_SHOULDER, 0]) < nx:
            out[L_SHOULDER, 0] = nx + 0.02 * fh

    if _vis(out, R_SHOULDER) and _vis(out, R_ELBOW):
        if float(out[R_ELBOW, 1]) < float(out[R_SHOULDER, 1]) + ELBOW_BELOW_SHOULDER * fh:
            out[R_ELBOW, 1] = float(out[R_SHOULDER, 1]) + ELBOW_BELOW_SHOULDER * fh
    if _vis(out, L_SHOULDER) and _vis(out, L_ELBOW):
        if float(out[L_ELBOW, 1]) < float(out[L_SHOULDER, 1]) + ELBOW_BELOW_SHOULDER * fh:
            out[L_ELBOW, 1] = float(out[L_SHOULDER, 1]) + ELBOW_BELOW_SHOULDER * fh

    # Only pin body nose → face nose when body is synthetic / untracked.
    if (not body_tracked) and _vis(out, BODY_NOSE) and _vis(out, 15):
        out[BODY_NOSE, 0] = float(out[15, 0])
        out[BODY_NOSE, 1] = float(out[15, 1])
    if _vis(out, BODY_NOSE) and _vis(out, NECK):
        if float(out[BODY_NOSE, 1]) > float(out[NECK, 1]) - 0.02 * fh:
            out[BODY_NOSE, 1] = float(out[NECK, 1]) - 0.05 * fh


# Chin/neck may scale a bit with head size, but they cannot disconnect.
ATTACH_LEN_MIN = 0.70
ATTACH_LEN_MAX = 1.30


def _mean_visible_xy(k: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray | None:
    pts = [k[i, :2].astype(np.float64) for i in idxs if _vis(k, i)]
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def _face_roll(k: np.ndarray) -> float:
    el = _mean_visible_xy(k, L_EYE)
    er = _mean_visible_xy(k, R_EYE)
    if el is None or er is None:
        return 0.0
    return float(np.arctan2(float(er[1] - el[1]), float(er[0] - el[0])))


def _rotate2(vec: np.ndarray, angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [c * float(vec[0]) - s * float(vec[1]), s * float(vec[0]) + c * float(vec[1])],
        dtype=np.float64,
    )


def _head_attach_xy(k: np.ndarray, prefer: int | None = None) -> tuple[int | None, np.ndarray]:
    if prefer is not None and _vis(k, prefer):
        return prefer, k[prefer, :2].astype(np.float64)
    for idx in (2, 15):
        if _vis(k, idx):
            return idx, k[idx, :2].astype(np.float64)
    return None, face_center(k).astype(np.float64)


def _snap_body_nose_to_face(out: np.ndarray, ref: np.ndarray, droll: float) -> None:
    if not (_vis(out, BODY_NOSE) and _vis(out, 15)):
        return
    if _vis(ref, BODY_NOSE) and _vis(ref, 15):
        offset = _rotate2(ref[BODY_NOSE, :2].astype(np.float64) - ref[15, :2].astype(np.float64), droll)
    else:
        offset = np.zeros(2, dtype=np.float64)
    out[BODY_NOSE, 0] = float(out[15, 0] + offset[0])
    out[BODY_NOSE, 1] = float(out[15, 1] + offset[1])


def enforce_head_body_attachment(out: np.ndarray, ref: np.ndarray) -> None:
    """Keep the skeleton plugged into the face using the reference rest offset.

    The face is authoritative. If chin→neck stretches, shrinks, or inverts, the
    whole body translates so neck stays at the authored offset (with scale slack).
    Body-nose stays on the face nose so the green neck bone cannot tear off.
    """
    o = _as37(out)
    r = _as37(ref)
    if not _vis(o, NECK) or not _vis(r, NECK):
        return

    ref_idx, ref_head = _head_attach_xy(r)
    _, out_head = _head_attach_xy(o, prefer=ref_idx)
    rest = r[NECK, :2].astype(np.float64) - ref_head
    rest_len = float(np.linalg.norm(rest))
    if rest_len < 1e-5:
        rest = np.array([0.0, 0.22 * max(face_height(r), 1e-3)], dtype=np.float64)
        rest_len = float(np.linalg.norm(rest))

    droll = _face_roll(o) - _face_roll(r)
    rest_rot = _rotate2(rest, droll)
    rest_dir = rest_rot / max(float(np.linalg.norm(rest_rot)), 1e-6)

    cur = o[NECK, :2].astype(np.float64) - out_head
    cur_len = float(np.linalg.norm(cur))
    min_len = ATTACH_LEN_MIN * rest_len
    max_len = ATTACH_LEN_MAX * rest_len
    neck_above = float(o[NECK, 1]) < float(out_head[1]) + 0.02 * rest_len
    if neck_above or cur_len < min_len or cur_len > max_len:
        scale = 1.0
        if cur_len > 1e-6 and not neck_above:
            scale = float(np.clip(cur_len / rest_len, ATTACH_LEN_MIN, ATTACH_LEN_MAX))
        desired = out_head + rest_dir * (rest_len * scale)
        shift = desired - o[NECK, :2].astype(np.float64)
        if float(np.linalg.norm(shift)) > 1e-6:
            for i in BODY_SLOTS:
                if not _vis(o, i):
                    continue
                o[i, 0] = float(o[i, 0] + shift[0])
                o[i, 1] = float(o[i, 1] + shift[1])
    _snap_body_nose_to_face(o, r, droll)


_BODY_NAMES = (
    (30, "body_nose"),
    (31, "neck"),
    (32, "r_shoulder"),
    (33, "r_elbow"),
    (34, "l_shoulder"),
    (35, "l_elbow"),
    (36, "chest"),
)


def format_body_diagnostics(kps: np.ndarray, *, label: str = "pose") -> str:
    """One-line + body-slot dump for console debugging."""
    k = _as37(kps)
    fh = face_height(k)
    cy = chin_y(k)
    fc = face_center(k)
    neck_y = float(k[NECK, 1]) if _vis(k, NECK) else float("nan")
    lines = [
        f"[pose-diag] {label}: face_h={fh:.4f} chin_y={cy} "
        f"face_c=({fc[0]:+.3f},{fc[1]:+.3f}) neck_y={neck_y:.4f} "
        f"neck-chin={(neck_y - cy) if cy is not None else float('nan'):+.4f}"
    ]
    for i, name in _BODY_NAMES:
        if _vis(k, i):
            lines.append(
                f"  {i:2d} {name:<11} x={float(k[i, 0]):+.4f} y={float(k[i, 1]):+.4f}"
            )
        else:
            lines.append(f"  {i:2d} {name:<11} INVISIBLE")
    flags: list[str] = []
    if cy is not None and _vis(k, NECK) and float(k[NECK, 1]) < cy:
        flags.append("NECK_ABOVE_CHIN")
    if _vis(k, NECK) and _vis(k, R_SHOULDER) and float(k[R_SHOULDER, 1]) < float(k[NECK, 1]):
        flags.append("R_SH_ABOVE_NECK")
    if _vis(k, NECK) and _vis(k, L_SHOULDER) and float(k[L_SHOULDER, 1]) < float(k[NECK, 1]):
        flags.append("L_SH_ABOVE_NECK")
    if _vis(k, 21) and _vis(k, 25) and float(k[25, 1]) < float(k[21, 1]):
        # Informational: some characters are authored this way (see LipRoles).
        flags.append("MOUTH_INVERTED")
    if flags:
        lines.append(f"  FLAGS: {', '.join(flags)}")
    else:
        lines.append("  FLAGS: ok")
    return "\n".join(lines)


def sanitize_pose(
    keypoints: np.ndarray,
    ref_keypoints: np.ndarray,
    *,
    recenter: bool = True,
    clamp_travel: bool = True,
    topology: bool = True,
    sanitize_body: bool = True,
    body_tracked: bool = False,
    lock_proportions: bool = False,
    lock_face_proportions: bool = True,
    limit_face: bool = True,
    limit_brows: bool = True,
    limit_eyes: bool = True,
    limit_nose: bool = True,
    limit_mouth: bool = True,
    log: bool = False,
    log_label: str = "sanitize",
) -> np.ndarray:
    """Make driven keypoints stay proportional to the character reference.

    Order: travel clamp → face re-center → topology guards → optional
    proportion locks (face/eye/mouth widths + body bone lengths).

    ``body_tracked``: when True, body slots use wider travel, are excluded from
    face re-centering, and body-nose is not forced onto face nose tip.

    ``limit_face`` / ``limit_brows`` / ``limit_eyes`` / ``limit_nose`` /
    ``limit_mouth``: when False, skip that region's travel clamp (and the
    matching face proportion lock when ``lock_face_proportions`` is on).
    """
    out = _as37(keypoints).copy()
    ref = _as37(ref_keypoints)
    fh = face_height(ref)
    before = out.copy()
    travel_kwargs = dict(
        sanitize_body=sanitize_body,
        body_tracked=body_tracked,
        limit_face=limit_face,
        limit_brows=limit_brows,
        limit_eyes=limit_eyes,
        limit_nose=limit_nose,
        limit_mouth=limit_mouth,
    )

    if log:
        print(format_body_diagnostics(ref, label=f"{log_label}/REF"))
        print(format_body_diagnostics(before, label=f"{log_label}/BEFORE"))

    if clamp_travel:
        _clamp_travel(out, ref, fh, **travel_kwargs)
    if recenter:
        _recenter_face(
            out, ref, body_tracked=body_tracked, sanitize_body=sanitize_body
        )
        if clamp_travel:
            _clamp_travel(out, ref, fh, **travel_kwargs)
    if topology:
        _enforce_topology(
            out,
            ref,
            fh,
            body_tracked=body_tracked,
            sanitize_body=sanitize_body,
            limit_mouth=limit_mouth,
            limit_nose=limit_nose,
        )
    if sanitize_body:
        enforce_head_body_attachment(out, ref)
    if lock_proportions:
        from .live_retarget import enforce_proportion_invariants

        enforce_proportion_invariants(
            out,
            ref,
            sanitize_face=lock_face_proportions,
            sanitize_body=sanitize_body,
            limit_face=limit_face,
            limit_eyes=limit_eyes,
            limit_mouth=limit_mouth,
        )

    invisible = out[:, 3] < 0.5
    out[invisible, 0:2] = 0.0

    if log:
        print(format_body_diagnostics(out, label=f"{log_label}/AFTER"))
        deltas = []
        for i in list(range(20, 28)) + list(range(30, 37)):
            if before[i, 3] < 0.5 and out[i, 3] < 0.5:
                continue
            dx = float(out[i, 0] - before[i, 0])
            dy = float(out[i, 1] - before[i, 1])
            if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                deltas.append(f"{i}:d=({dx:+.3f},{dy:+.3f})")
        print(
            f"[pose-diag] {log_label}/DELTAS "
            + (", ".join(deltas) if deltas else "(none)")
        )
    return out


# ---------------------------------------------------------------------------
# Overlay — draw mesh on character frame
# ---------------------------------------------------------------------------


def normalized_to_pixels(
    kps: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert (37,4) ``norm_crop`` [-1,1] coords → pixel xy (score/visible kept).

    Matches ``transform_keypoints_crop`` / training: ``x = (nx+1)/2 * out_w``.
    """
    k = _as37(kps).copy()
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    k[:, 0] = (k[:, 0] + 1.0) * 0.5 * w
    k[:, 1] = (k[:, 1] + 1.0) * 0.5 * h
    return k


def pixels_to_normalized(
    x: float,
    y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Map image pixel xy → ``norm_crop`` [-1, 1] (same formula as training)."""
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    nx = float(x) / w * 2.0 - 1.0
    ny = float(y) / h * 2.0 - 1.0
    return float(np.clip(nx, -1.5, 1.5)), float(np.clip(ny, -1.5, 1.5))


def overlay_visible_slots(status: dict | None) -> set[int]:
    """Keypoint indices currently drawn / pickable from overlay part toggles."""
    st = status or {}
    if not bool(st.get("show_mesh", True)):
        return set()
    ids: set[int] = set()
    if bool(st.get("show_outline", True)):
        ids.update(OUTLINE)
    if bool(st.get("show_brows", True)):
        ids.update(BROWS)
    if bool(st.get("show_eyes", True)):
        ids.update(L_EYE + R_EYE)
    if bool(st.get("show_nose", True)):
        ids.update(i for i in NOSE if i not in BLOCKED_POSE_SLOT_SET)
    if bool(st.get("show_mouth", True)):
        ids.update(MOUTH)
    if bool(st.get("show_iris_overlay", True)):
        ids.update((IRIS_L, IRIS_R))
    if bool(st.get("show_skeleton", True)):
        ids.update(OVERLAY_BODY_SLOTS)
    return ids


def nearest_keypoint(
    keypoints: np.ndarray,
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    *,
    max_dist: float = 28.0,
    include: Iterable[int] | None = None,
) -> int | None:
    """Return index of nearest visible keypoint in image pixel space, or None."""
    pix = normalized_to_pixels(keypoints, width, height)
    allowed = None if include is None else set(int(i) for i in include)
    best_i: int | None = None
    best_d = float(max_dist)
    for i in range(NUM_KEYPOINTS):
        if allowed is not None and i not in allowed:
            continue
        if float(pix[i, 3]) < 0.5:
            continue
        dx = float(pix[i, 0]) - float(pixel_x)
        dy = float(pix[i, 1]) - float(pixel_y)
        d = (dx * dx + dy * dy) ** 0.5
        if d <= best_d:
            best_d = d
            best_i = i
    return best_i


def _visible(k: np.ndarray, idx: int, *, min_score: float = 0.15) -> bool:
    return float(k[idx, 3]) >= 0.5 and float(k[idx, 2]) >= min_score


def _xy(k: np.ndarray, idx: int) -> tuple[int, int]:
    return int(round(float(k[idx, 0]))), int(round(float(k[idx, 1])))


def _group_of(i: int) -> str:
    if i <= 4:
        return "outline"
    if i <= 10:
        return "brow"
    if i in (11, 12, 13, 17, 18, 19):
        return "eye"
    if 14 <= i <= 16:
        return "nose"
    if 20 <= i <= 27:
        return "mouth"
    return "mouth"


def _polyline(
    draw: ImageDraw.ImageDraw,
    k: np.ndarray,
    idxs: Iterable[int],
    color: tuple[int, int, int],
    *,
    closed: bool = False,
    width: int = 2,
) -> None:
    pts: list[tuple[int, int]] = []
    for i in idxs:
        if _visible(k, i):
            pts.append(_xy(k, i))
    if len(pts) < 2:
        return
    if closed and len(pts) >= 3:
        pts = pts + [pts[0]]
    draw.line(pts, fill=color, width=width)


_EYE_LID_IDS = (11, 12, 13, 17, 18, 19)


def _id_label_xy(i: int, x: int, y: int) -> tuple[int, int]:
    """Pixel origin for a keypoint index label.

    Upper-lip 20/21 sit under the nose, so their numbers go *below* the dot
    instead of covering the point (and each other) above it.
    """
    if i in (20, 21):
        return (x + 4, y + 8)
    if i in _EYE_LID_IDS:
        return (x + 7, y - 12)
    if i >= BODY_NOSE:
        return (x + 5, y - 10)
    return (x + 5, y - 11)


def _face_group_enabled(group: str, *, outline: bool, brows: bool, eyes: bool, nose: bool, mouth: bool) -> bool:
    if group == "outline":
        return outline
    if group == "brow":
        return brows
    if group == "eye":
        return eyes
    if group == "nose":
        return nose
    return mouth


def _draw_eye_point_hollow(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    r: int = 6,
    width: int = 2,
) -> None:
    """Eye lid marker: thick ring, empty center (no fill, no connecting lines)."""
    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        outline=color,
        width=width,
    )


def _try_font(size: int = 12):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_keypoint_mesh(
    image: Image.Image,
    keypoints: np.ndarray,
    *,
    show_ids: bool = True,
    show_face: bool = True,
    show_skeleton: bool = True,
    show_iris: bool = True,
    show_outline: bool = True,
    show_brows: bool = True,
    show_eyes: bool = True,
    show_nose: bool = True,
    show_mouth: bool = True,
    skeleton_lost: bool = False,
    mouth_snapped: bool = False,
    keypoints_are_pixels: bool = False,
) -> Image.Image:
    """Overlay Live-Poser-style mesh using model-space (37,4) keypoints.

    By default ``keypoints`` are ``norm_crop`` [-1,1]. Pass
    ``keypoints_are_pixels=True`` when xy are already absolute image pixels
    (camera diagnostic preview). When ``skeleton_lost`` is True the body
    overlay draws red instead of green. When ``mouth_snapped`` is True the
    mouth overlay switches from purple to blue. ``show_face=False`` hides every
    face group; otherwise ``show_outline`` / ``show_brows`` / ``show_eyes`` /
    ``show_nose`` / ``show_mouth`` select parts independently.
    """
    if image is None or keypoints is None:
        return image
    if not show_face:
        show_outline = show_brows = show_eyes = show_nose = show_mouth = False
    base = image.convert("RGB")
    w, h = base.size
    if keypoints_are_pixels:
        pix = _as37(keypoints).copy()
    else:
        pix = normalized_to_pixels(keypoints, w, h)
    out = base.copy()
    draw = ImageDraw.Draw(out)
    font = _try_font(11) if show_ids else None
    mouth_color = MOUTH_SNAPPED_COLOR if mouth_snapped else MOUTH_COLOR
    show_any_face = show_outline or show_brows or show_eyes or show_nose or show_mouth

    if show_any_face:
        for idxs, color in FACE_CONNECTIONS:
            group = _group_of(idxs[0])
            if not _face_group_enabled(
                group,
                outline=show_outline,
                brows=show_brows,
                eyes=show_eyes,
                nose=show_nose,
                mouth=show_mouth,
            ):
                continue
            if list(idxs) == list(MOUTH_DRAW):
                color = mouth_color
            _polyline(
                draw,
                pix,
                tuple(i for i in idxs if i not in BLOCKED_POSE_SLOT_SET),
                color,
                closed=(list(idxs) == list(MOUTH_DRAW)),
                width=2,
            )
        # Eyes: hollow rings only — no aperture lines, no synthetic lower lid.
        if show_eyes:
            eye_color = FACE_POINT_COLOR["eye"]
            for i in _EYE_LID_IDS:
                if not _visible(pix, i):
                    continue
                x, y = _xy(pix, i)
                _draw_eye_point_hollow(draw, x, y, eye_color)
                if font is not None:
                    lx, ly = _id_label_xy(i, x, y)
                    draw.text((lx, ly), ref_of(i), fill=ID_LABEL_COLOR, font=font)

        for i in range(28):
            if i in _EYE_LID_IDS or i in BLOCKED_POSE_SLOT_SET:
                continue
            if not _visible(pix, i):
                continue
            group = _group_of(i)
            if not _face_group_enabled(
                group,
                outline=show_outline,
                brows=show_brows,
                eyes=show_eyes,
                nose=show_nose,
                mouth=show_mouth,
            ):
                continue
            x, y = _xy(pix, i)
            color = mouth_color if group == "mouth" else FACE_POINT_COLOR[group]
            r = 5 if i == 15 else 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255))
            if font is not None:
                lx, ly = _id_label_xy(i, x, y)
                draw.text((lx, ly), ref_of(i), fill=ID_LABEL_COLOR, font=font)

    if show_iris:
        for idx, color in (
            (IRIS_L, IRIS_RIGHT),
            (IRIS_R, IRIS_LEFT),
        ):
            if not _visible(pix, idx):
                continue
            x, y = _xy(pix, idx)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=color)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=1)
            if font is not None:
                draw.text((x + 8, y - 6), ref_of(idx), fill=color, font=font)

    if show_skeleton:
        bone_color = SKELETON_LOST_COLOR if skeleton_lost else SKELETON_COLOR
        joint_color = JOINT_LOST_COLOR if skeleton_lost else JOINT_COLOR
        body_pts: dict[int, tuple[int, int]] = {}
        for idx in OVERLAY_BODY_SLOTS:
            if not _visible(pix, idx, min_score=0.0):
                if float(pix[idx, 3]) < 0.5:
                    continue
            x, y = _xy(pix, idx)
            if x == 0 and y == 0 and float(pix[idx, 2]) <= 0:
                continue
            body_pts[idx] = (x, y)

        for a, b in SKELETON_BONES:
            if a in body_pts and b in body_pts:
                draw.line([body_pts[a], body_pts[b]], fill=bone_color, width=3)

        for idx, (x, y) in body_pts.items():
            r = 5 if idx in (NECK, CHEST) else 4
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=joint_color,
                outline=joint_color,
            )
            if font is not None:
                lx, ly = _id_label_xy(idx, x, y)
                draw.text((lx, ly), ref_of(idx), fill=ID_LABEL_COLOR, font=font)
        if skeleton_lost and body_pts:
            anchor = body_pts.get(NECK) or body_pts.get(CHEST) or next(iter(body_pts.values()))
            draw.text(
                (anchor[0] + 8, anchor[1] + 14),
                "LOST",
                fill=SKELETON_LOST_COLOR,
                font=font or _try_font(12),
            )

    return out


def _hair_poly_pixels(
    polygon,
    width: int,
    height: int,
    *,
    pixel_space: bool = False,
) -> list[tuple[float, float]]:
    """Map a hair polygon to image pixels.

    Default coords are ``norm_crop`` [-1, 1]. Pass ``pixel_space=True`` when
    the polygon is already in source-pixel coordinates.
    """
    if polygon is None:
        return []
    arr = np.asarray(polygon, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return []
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    out: list[tuple[float, float]] = []
    for x, y in arr[:, :2]:
        if pixel_space:
            out.append((float(x), float(y)))
        else:
            out.append((float((x + 1.0) * 0.5 * w), float((y + 1.0) * 0.5 * h)))
    return out


def draw_hair_overlay(
    image: Image.Image,
    segments,
    *,
    lost: bool = False,
    pixel_space: bool = False,
) -> Image.Image:
    """Tint hair-part polygons on the same preview as the skeleton mesh."""
    if image is None or not segments:
        return image
    base = image.convert("RGB")
    w, h = base.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    fill_a = 70 if lost else 140
    draw_l = ImageDraw.Draw(layer)
    labels: list[tuple[tuple[float, float], str, tuple[int, int, int]]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        cls = str(seg.get("class") or "")
        color = HAIR_OVERLAY_COLORS.get(cls)
        xy = _hair_poly_pixels(
            seg.get("polygon") or [], w, h, pixel_space=pixel_space
        )
        if color is None or len(xy) < 3:
            continue
        draw_l.polygon(xy, fill=(*color, fill_a))
        cx = sum(p[0] for p in xy) / len(xy)
        cy = sum(p[1] for p in xy) / len(xy)
        labels.append(((cx, cy), HAIR_OVERLAY_LABELS.get(cls, "?"), color))
    if not labels:
        return base
    out = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(out)
    font = _try_font(12)
    for (cx, cy), tag, color in labels:
        draw.text((int(cx), int(cy)), tag, fill=color, font=font)
    return out
