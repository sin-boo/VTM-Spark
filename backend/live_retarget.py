"""Constrained Live2D-style retarget: reference proportions + semantic controls.

Instead of transferring every human landmark delta independently (which stretches
anime characters to human proportions), this module:

1. Locks reference face layout and body bone lengths.
2. Extracts bounded semantic controls from live vs origin (head, blink, gaze,
   brows, mouth, body joint angles).
3. Reconstructs the character pose from the reference rig + those controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose_controller import (
    BODY_NOSE,
    BROWS,
    CHEST,
    FACE_SLOTS,
    IRIS_EYE_PAIRS,
    L_ELBOW,
    L_EYE,
    L_SHOULDER,
    LEFT_BROW,
    LEFT_IRIS,
    LipRoles,
    MOUTH,
    MOUTH_CORNERS,
    NECK,
    NOSE,
    NUM_KEYPOINTS,
    OUTLINE,
    R_ELBOW,
    R_EYE,
    R_SHOULDER,
    RIGHT_BROW,
    RIGHT_IRIS,
    enforce_head_body_attachment,
    face_center,
    face_height,
    face_width,
    lip_roles,
)

KEYPOINT_DIM = 4

# Head motion limits (fractions of character face height / degrees / scale).
HEAD_TX_MAX = 0.12
HEAD_TY_MAX = 0.10
HEAD_ROLL_MAX_DEG = 18.0
HEAD_YAW_MAX_DEG = 28.0
HEAD_PITCH_MAX_DEG = 18.0
HEAD_SCALE_MIN = 0.94
HEAD_SCALE_MAX = 1.06
# RelativePose nx/ny is ±1 at half-frame. Used only when landmark COM is quiet.
HEAD_TX_NORM_GAIN = 0.75
HEAD_TY_NORM_GAIN = 0.65
# Visible nod: pitch rotates around the neck, so midline points must actually
# travel in Y. The 2.5D warp alone only foreshortens |x| and is invisible on
# the nose/chin.
PITCH_NOD = 0.28  # × character face height at 90°

# Expression / gaze response limits.
BLINK_MIN = 0.0
BLINK_MAX = 1.0
BROW_RAISE_MAX = 0.08  # × face height
MOUTH_SMILE_MAX = 0.10
GAZE_MAX = 0.85  # normalized inside eye box
IRIS_HIDE_BLINK = 0.10

# Body angle clamps (degrees from rest).
SHOULDER_ANGLE_MAX_DEG = 35.0
ELBOW_ANGLE_MAX_DEG = 55.0
CHEST_ANGLE_MAX_DEG = 12.0

# Mild 2.5D warp strengths (character-relative).
# Strong enough that a ~20–30° turn is obvious on the anime mesh, not just on
# the raw camera landmarks.
YAW_WARP = 0.68
PITCH_WARP = 0.32
YAW_FORESHORTEN = 0.42  # how much X collapses at 90° yaw (0.55→0.13 old → stronger)
PITCH_FORESHORTEN = 0.38
YAW_FEATURE_SHIFT = 0.30
# Thin closed slit as a fraction of mouth width.
MOUTH_CLOSED_SLIT = 0.012
# How far a full open may drop the lower lip, as a fraction of mouth width.
MOUTH_OPEN_MAX = 0.42
# Slot 21 parked on the nose tip — ignore that mid and use 20/22 instead.
MOUTH_ON_NOSE = 0.06


def _as37(kps: np.ndarray) -> np.ndarray:
    out = np.asarray(kps, dtype=np.float32)
    if out.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(f"Expected ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {out.shape}")
    return out


def _vis(k: np.ndarray, i: int) -> bool:
    return float(k[i, 3]) >= 0.5


def _mean_xy(k: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray | None:
    pts = [k[i, :2] for i in idxs if _vis(k, i)]
    if not pts:
        return None
    return np.mean(np.stack(pts), axis=0).astype(np.float32)


# Mouth must stay below nose tip 15 (Y down). Live UV + head warp can park 21
# on the nose or invert 21/25; sanitize topology is off on the live path.
MOUTH_ON_NOSE_EPS = 1e-4
MOUTH_OFF_NOSE = 0.015  # × character face height, only for points on the nose
# Extra opening past the authored 21–25 gap. Inner-lower OSF 64 (slot 25)
# otherwise walks toward the chin and never comes back.
MOUTH_LOWER_GAP_PAD = 0.16  # × face height


def clamp_mouth_anatomy(
    out: np.ndarray,
    face_h: float | None = None,
    *,
    ref: np.ndarray | None = None,
    limit_mouth: bool = True,
) -> np.ndarray:
    """Last anatomy net: off the nose, upper-mid on its lip line, lower lip below it.

    Works on the character's *visual* lip rows (``lip_roles``), so a rest pose
    with 25 authored above 21 is respected instead of "corrected". Only slots
    whose rest pose is below the nose are pulled off it. Live sanitize leaves
    topology off, so this is what the overlay / DiT actually see.
    """
    k = _as37(out)
    fh = max(float(face_h if face_h is not None else face_height(k)), 1e-3)
    rest = _as37(ref) if ref is not None else None
    roles = lip_roles(rest)
    um, lm = roles.upper_mid, roles.lower_mid

    if _vis(k, 15):
        nose_y = float(k[15, 1])
        for i in MOUTH:
            if not _vis(k, i):
                continue
            rest_below = True
            if rest is not None and _vis(rest, i) and _vis(rest, 15):
                rest_below = float(rest[i, 1]) > float(rest[15, 1]) + 0.01 * fh
            if rest_below and float(k[i, 1]) <= nose_y + MOUTH_ON_NOSE_EPS:
                k[i, 1] = nose_y + MOUTH_OFF_NOSE * fh

    # Lower mid popped above the upper mid: restore a thin slit. Using the
    # authored rest gap here made a slightly open rest impossible to close.
    if _vis(k, um) and _vis(k, lm) and float(k[lm, 1]) < float(k[um, 1]):
        mw = _mouth_width(k) or (0.22 * fh)
        k[lm, 1] = float(k[um, 1]) + max(MOUTH_CLOSED_SLIT * mw, 0.004 * fh)

    _flatten_upper_mid_to_chord(k, roles.upper)
    if limit_mouth:
        _limit_lower_mid_drop(k, fh, roles, rest=rest)
    elif _vis(k, lm) and _vis(k, 2):
        k[lm, 1] = min(float(k[lm, 1]), float(k[2, 1]) - 0.01 * fh)
    return k


def _project_onto_segment(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    chord = end.astype(np.float64) - start.astype(np.float64)
    denom = float(np.dot(chord, chord))
    if denom < 1e-12:
        return start.astype(np.float64).copy()
    rel = point.astype(np.float64) - start.astype(np.float64)
    t = float(np.clip(np.dot(rel, chord) / denom, 0.0, 1.0))
    return start.astype(np.float64) + t * chord


def _flatten_upper_mid_to_chord(k: np.ndarray, upper: tuple[int, int, int]) -> None:
    """If the visual upper-mid peaks toward the nose, snap it onto that lip line."""
    left_i, mid_i, right_i = upper
    if not all(_vis(k, i) for i in upper):
        return
    mid = k[mid_i, :2].astype(np.float64)
    proj = _project_onto_segment(mid, k[left_i, :2], k[right_i, :2])
    # Image Y down: above the chord is toward the nose.
    if float(mid[1]) < float(proj[1]) - 1e-6:
        k[mid_i, :2] = proj


def _limit_lower_mid_drop(
    k: np.ndarray,
    face_h: float,
    roles: LipRoles,
    *,
    rest: np.ndarray | None = None,
) -> None:
    """Stop the visual lower-mid from walking toward the chin."""
    um, lm = roles.upper_mid, roles.lower_mid
    if not _vis(k, lm):
        return
    fh = max(float(face_h), 1e-3)
    if _vis(k, 2):
        k[lm, 1] = min(float(k[lm, 1]), float(k[2, 1]) - 0.01 * fh)
    if _vis(k, um):
        rest_gap = 0.05 * fh
        if rest is not None and _vis(rest, um) and _vis(rest, lm):
            rest_gap = max(float(rest[lm, 1] - rest[um, 1]), 0.0)
        k[lm, 1] = min(
            float(k[lm, 1]), float(k[um, 1]) + rest_gap + MOUTH_LOWER_GAP_PAD * fh
        )
    a, b = roles.lower[0], roles.lower[2]
    if _vis(k, a) and _vis(k, b):
        chord_y = 0.5 * (float(k[a, 1]) + float(k[b, 1]))
        mw = _mouth_width(k) or (0.22 * fh)
        k[lm, 1] = min(float(k[lm, 1]), chord_y + 0.45 * mw)


def _eye_width(k: np.ndarray, idxs: tuple[int, ...]) -> float | None:
    vis = [i for i in idxs if _vis(k, i)]
    if len(vis) < 2:
        return None
    xs = [float(k[i, 0]) for i in vis]
    return max(1e-4, max(xs) - min(xs))


def _eye_aperture(k: np.ndarray, idxs: tuple[int, ...]) -> float | None:
    """Vertical span of an eye (openness proxy)."""
    vis = [i for i in idxs if _vis(k, i)]
    if len(vis) < 2:
        return None
    ys = [float(k[i, 1]) for i in vis]
    return max(1e-4, max(ys) - min(ys))


def _eye_lid_depth(k: np.ndarray, idxs: tuple[int, ...]) -> float | None:
    """Perpendicular upper-lid depth from the eye-corner chord.

    The three-point eye schema is corner / upper-mid / corner, not an upper and
    lower lid chain. Distance to the corner chord therefore tracks closure
    without mistaking head roll or uneven corner heights for eye aperture.
    """
    if len(idxs) != 3 or not all(_vis(k, i) for i in idxs):
        return None
    a = k[idxs[0], :2].astype(np.float64)
    p = k[idxs[1], :2].astype(np.float64)
    b = k[idxs[2], :2].astype(np.float64)
    chord = b - a
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-5:
        return None
    return abs(float(chord[0] * (p[1] - a[1]) - chord[1] * (p[0] - a[0]))) / chord_len


def _mouth_gap(k: np.ndarray) -> float | None:
    if not (_vis(k, 21) and _vis(k, 25)):
        return None
    return float(k[25, 1] - k[21, 1])



def _mouth_width(k: np.ndarray) -> float | None:
    if not (_vis(k, 23) and _vis(k, 26)):
        return None
    return max(1e-4, abs(float(k[26, 0] - k[23, 0])))


def _mouth_local_metrics(k: np.ndarray) -> tuple[float, float, float, float] | None:
    """Width-normalized gap and corner lifts in the mouth's own basis."""
    needed = (21, 23, 25, 26)
    if not all(_vis(k, i) for i in needed):
        return None
    left = k[23, :2].astype(np.float64)
    right = k[26, :2].astype(np.float64)
    axis_x = right - left
    width = float(np.linalg.norm(axis_x))
    if width < 1e-5:
        return None
    axis_x /= width
    axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float64)
    if axis_y[1] < 0.0:
        axis_y *= -1.0
    upper = k[21, :2].astype(np.float64)
    if _vis(k, 15) and float(
        np.hypot(float(k[21, 0] - k[15, 0]), float(k[21, 1] - k[15, 1]))
    ) < MOUTH_ON_NOSE:
        if _vis(k, 20) and _vis(k, 22):
            upper = 0.5 * (
                k[20, :2].astype(np.float64) + k[22, :2].astype(np.float64)
            )
    lower = k[25, :2].astype(np.float64)
    gap = max(0.0, float(np.dot(lower - upper, axis_y)) / width)
    lift_l = float(np.dot(upper - left, axis_y)) / width
    lift_r = float(np.dot(upper - right, axis_y)) / width
    return gap, lift_l, lift_r, width


def _apply_mouth_expression(
    local_face: np.ndarray,
    ref: np.ndarray,
    rig: "ReferenceRig",
    controls: "SemanticControls",
    *,
    limit_mouth: bool,
) -> None:
    """Deform the authored mouth with open / smile / width. Upper lip stays put."""
    mw = max(float(rig.mouth_width), 1e-4)
    drop = float(controls.mouth_gap_delta) * mw
    if limit_mouth:
        drop = _clip(drop, -0.85 * float(rig.mouth_gap), MOUTH_OPEN_MAX * mw)
    else:
        drop = _clip(drop, -float(rig.mouth_gap), 0.75 * mw)
    smile_y = -float(controls.mouth_smile) * mw
    widen = float(controls.mouth_form) * (0.09 if limit_mouth else 0.14) * (0.5 * mw)
    asym = float(controls.mouth_asym) * 0.04 * mw
    roles = lip_roles(ref)
    weights = {
        roles.lower[0]: 0.85,
        roles.lower_mid: 1.0,
        roles.lower[2]: 0.85,
    }
    for i, w in weights.items():
        if _vis(ref, i):
            local_face[i, 1] = float(local_face[i, 1] + drop * w)
    if _vis(ref, 23):
        local_face[23, 1] = float(local_face[23, 1] + drop * 0.40 + smile_y + asym)
        local_face[23, 0] = float(local_face[23, 0] - widen)
    if _vis(ref, 26):
        local_face[26, 1] = float(local_face[26, 1] + drop * 0.40 + smile_y - asym)
        local_face[26, 0] = float(local_face[26, 0] + widen)
    cx = 0.5 * (float(local_face[23, 0]) + float(local_face[26, 0]))
    for i in (*roles.upper, *roles.lower):
        if i in MOUTH_CORNERS or not _vis(ref, i):
            continue
        side = 1.0 if float(local_face[i, 0]) >= cx else -1.0
        local_face[i, 0] = float(local_face[i, 0] + side * widen * 0.45)


def _angle(origin: np.ndarray, tip: np.ndarray) -> float:
    return float(np.arctan2(float(tip[1] - origin[1]), float(tip[0] - origin[0])))


def _length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(float(b[0] - a[0]), float(b[1] - a[1])))


def _rot2(v: np.ndarray, radians: float) -> np.ndarray:
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def _cap(v: float, limit: float, active: bool) -> float:
    if not active:
        return float(v)
    lim = abs(float(limit))
    return _clip(float(v), -lim, lim)


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    t = _clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def _soft_positive(value: float, response: float) -> float:
    """Smoothly map a positive signal to [0, 1) without a hard ceiling."""
    if value <= 0.0:
        return 0.0
    return float(np.tanh(value / max(response, 1e-4)))


@dataclass
class ReferenceRig:
    """Immutable character proportions derived from the reference pose."""

    face_center: np.ndarray
    face_height: float
    face_width: float
    local_face: np.ndarray  # (28, 2) offsets from face_center
    eye_l_center: np.ndarray
    eye_r_center: np.ndarray
    eye_l_width: float
    eye_r_width: float
    eye_l_aperture: float
    eye_r_aperture: float
    brow_l_y: float
    brow_r_y: float
    mouth_gap: float
    mouth_width: float
    # Body rest: absolute ref positions for length, vectors from neck.
    neck: np.ndarray
    bone_len: dict[int, float]  # slot -> length from parent
    bone_angle: dict[int, float]  # slot -> rest angle from parent
    parent: dict[int, int]


def build_reference_rig(ref_keypoints: np.ndarray) -> ReferenceRig:
    ref = _as37(ref_keypoints)
    fc = face_center(ref)
    fh = max(face_height(ref), 1e-3)
    fw = max(face_width(ref), 1e-3)

    local = np.zeros((28, 2), dtype=np.float32)
    for i in FACE_SLOTS:
        if _vis(ref, i):
            local[i] = ref[i, :2] - fc

    el_c = _mean_xy(ref, L_EYE)
    er_c = _mean_xy(ref, R_EYE)
    if el_c is None:
        el_c = fc + np.array([-0.18, -0.08], dtype=np.float32) * fh
    if er_c is None:
        er_c = fc + np.array([0.18, -0.08], dtype=np.float32) * fh

    el_w = _eye_width(ref, L_EYE) or (0.12 * fh)
    er_w = _eye_width(ref, R_EYE) or (0.12 * fh)
    el_a = _eye_aperture(ref, L_EYE) or (0.04 * fh)
    er_a = _eye_aperture(ref, R_EYE) or (0.04 * fh)

    bl = _mean_xy(ref, LEFT_BROW)
    br = _mean_xy(ref, RIGHT_BROW)
    brow_l_y = float(bl[1]) if bl is not None else float(el_c[1] - 0.08 * fh)
    brow_r_y = float(br[1]) if br is not None else float(er_c[1] - 0.08 * fh)

    mg = _mouth_gap(ref)
    mw = _mouth_width(ref)
    mouth_gap = float(mg) if mg is not None else 0.02 * fh
    mouth_width = float(mw) if mw is not None else 0.25 * fw

    neck = ref[NECK, :2].copy() if _vis(ref, NECK) else fc + np.array([0.0, 0.35 * fh], dtype=np.float32)

    # Parent map for upper-body IK-style reconstruction.
    parent = {
        BODY_NOSE: NECK,
        R_SHOULDER: NECK,
        L_SHOULDER: NECK,
        CHEST: NECK,
        R_ELBOW: R_SHOULDER,
        L_ELBOW: L_SHOULDER,
    }
    bone_len: dict[int, float] = {}
    bone_angle: dict[int, float] = {}
    for child, par in parent.items():
        if not (_vis(ref, child) and _vis(ref, par)):
            continue
        p = ref[par, :2]
        c = ref[child, :2]
        bone_len[child] = max(_length(p, c), 1e-4)
        bone_angle[child] = _angle(p, c)

    return ReferenceRig(
        face_center=fc.astype(np.float32),
        face_height=fh,
        face_width=fw,
        local_face=local,
        eye_l_center=el_c.astype(np.float32),
        eye_r_center=er_c.astype(np.float32),
        eye_l_width=float(el_w),
        eye_r_width=float(er_w),
        eye_l_aperture=float(el_a),
        eye_r_aperture=float(er_a),
        brow_l_y=brow_l_y,
        brow_r_y=brow_r_y,
        mouth_gap=max(mouth_gap, 1e-4),
        mouth_width=max(mouth_width, 1e-4),
        neck=neck.astype(np.float32),
        bone_len=bone_len,
        bone_angle=bone_angle,
        parent=parent,
    )


@dataclass
class SemanticControls:
    """Bounded Live2D-style channels extracted from live vs origin."""

    head_tx: float = 0.0  # × character face height
    head_ty: float = 0.0
    head_roll_deg: float = 0.0
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_scale: float = 1.0
    blink_l: float = 1.0  # 1=open, ~0=closed
    blink_r: float = 1.0
    brow_l: float = 0.0  # × fh, negative = raise (y-up screen is +down)
    brow_r: float = 0.0
    mouth_open: float = 0.0  # normalized 0=closed, 1=wide open (pose keys)
    mouth_smile: float = 0.0
    mouth_form: float = 0.0
    mouth_asym: float = 0.0
    mouth_gap_delta: float = 0.0  # live-origin gap / width
    mouth_snapped: bool = False
    gaze_l: tuple[float, float] = (0.0, 0.0)  # normalized [-1,1] in eye
    gaze_r: tuple[float, float] = (0.0, 0.0)
    # Body angle deltas (radians) from origin rest, keyed by child slot.
    body_d_angle: dict[int, float] | None = None


def _eye_line_angle(k: np.ndarray) -> float | None:
    el = _mean_xy(k, L_EYE)
    er = _mean_xy(k, R_EYE)
    if el is None or er is None:
        return None
    return _angle(el, er)


def _rigid_face_span(k: np.ndarray) -> float:
    """Face-size proxy unaffected by mouth opening or jaw articulation."""
    el = _mean_xy(k, L_EYE)
    er = _mean_xy(k, R_EYE)
    if el is not None and er is not None:
        span = float(np.linalg.norm(er.astype(np.float64) - el.astype(np.float64)))
        if span > 1e-4:
            return span
    return max(face_width(k), 1e-3)


def _geom_head_angles(k: np.ndarray) -> tuple[float, float, float] | None:
    """Pitch / yaw / roll proxies from face keypoints (degrees).

    Same convention as ``label_schema.geom_head_angles`` so landmark fallbacks
    match CenterCalibration RelativePose when that path is unavailable.
    """
    el = _mean_xy(k, L_EYE)
    er = _mean_xy(k, R_EYE)
    if el is None or er is None or not _vis(k, 15):
        return None
    nose = k[15, :2].astype(np.float64)
    mid = 0.5 * (el.astype(np.float64) + er.astype(np.float64))
    eye_dist = float(np.linalg.norm(el - er)) + 1e-6
    roll = float(
        np.degrees(np.arctan2(float(el[1] - er[1]), float(el[0] - er[0])))
    )
    yaw = float(np.degrees(np.arctan2(-(float(nose[0] - mid[0])), eye_dist))) * 1.4
    if _vis(k, 2):
        chin = k[2, :2].astype(np.float64)
        face_h = float(np.linalg.norm(chin - mid)) + 1e-6
        pitch = float(np.degrees(np.arctan2(float(nose[1] - mid[1]), face_h))) * 1.4
    else:
        pitch = 0.0
    return pitch, yaw, roll


def _normalized_gaze(
    k: np.ndarray,
    iris: int,
    eye_idxs: tuple[int, ...],
) -> tuple[float, float]:
    if not _vis(k, iris):
        return (0.0, 0.0)
    if len(eye_idxs) != 3 or not all(_vis(k, i) for i in eye_idxs):
        return (0.0, 0.0)
    a = k[eye_idxs[0], :2].astype(np.float64)
    b = k[eye_idxs[2], :2].astype(np.float64)
    axis_x = b - a
    w = float(np.linalg.norm(axis_x))
    if w < 1e-5:
        return (0.0, 0.0)
    axis_x /= w
    axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float64)
    if axis_y[1] < 0.0:
        axis_y *= -1.0
    center = _mean_xy(k, eye_idxs)
    if center is None:
        return (0.0, 0.0)
    rel = k[iris, :2].astype(np.float64) - center.astype(np.float64)
    aperture = _eye_lid_depth(k, eye_idxs) or 1e-3
    gx = _clip(2.0 * float(np.dot(rel, axis_x)) / w, -1.0, 1.0)
    gy = _clip(float(np.dot(rel, axis_y)) / max(aperture, 1e-3), -1.0, 1.0)
    return (gx, gy)


def extract_controls(
    live_keypoints: np.ndarray,
    origin_keypoints: np.ndarray,
    rig: ReferenceRig,
    *,
    head_yaw_deg: float | None = None,
    head_pitch_deg: float | None = None,
    head_roll_deg: float | None = None,
    head_tx_norm: float | None = None,
    head_ty_norm: float | None = None,
    gain: float = 1.0,
    limit_face: bool = True,
    limit_brows: bool = True,
    limit_eyes: bool = True,
    limit_mouth: bool = True,
    motion: dict[str, float] | None = None,
) -> SemanticControls:
    live = _as37(live_keypoints)
    origin = _as37(origin_keypoints)
    fh = rig.face_height
    g = float(gain)
    # Motion multiplier: semantic deltas are (live - origin) * g.
    # 1.0 = full webcam motion vs Center; 0.5 = half; overlay hide does not change g.

    # --- Head translation (face COM) in character fh units ---
    c_live = face_center(live)
    c_origin = face_center(origin)
    # Scale human pixel motion into character units via live face height.
    live_rigid_span = _rigid_face_span(live)
    origin_rigid_span = _rigid_face_span(origin)
    ref_rigid_span = max(
        float(np.linalg.norm(rig.eye_r_center - rig.eye_l_center)), 1e-3
    )
    scale_to_char = ref_rigid_span / live_rigid_span
    landmark_tx = scale_to_char * float(c_live[0] - c_origin[0]) / fh
    landmark_ty = scale_to_char * float(c_live[1] - c_origin[1]) / fh

    def _pick_translation(landmark: float, calibrated_norm: float | None, gain: float) -> float:
        """Prefer same-space landmark COM; fall back to RelativePose nx/ny.

        Replacing landmark delta with a tiny nx (old 0.40× path) froze the
        overlay while the webcam face was clearly moving.
        """
        if calibrated_norm is None:
            return g * landmark
        calibrated = gain * float(calibrated_norm)
        if abs(landmark) >= abs(calibrated):
            return g * landmark
        return g * calibrated

    caps = motion or {}
    # Left / right are screen sides of the character; right is positive.
    yaw_left = abs(float(caps.get("turn_left", HEAD_YAW_MAX_DEG)))
    yaw_right = abs(float(caps.get("turn_right", HEAD_YAW_MAX_DEG)))
    roll_left = abs(float(caps.get("tilt_left", HEAD_ROLL_MAX_DEG)))
    roll_right = abs(float(caps.get("tilt_right", HEAD_ROLL_MAX_DEG)))
    pitch_up = float(caps.get("pitch_up", HEAD_PITCH_MAX_DEG))
    pitch_down = float(caps.get("pitch_down", HEAD_PITCH_MAX_DEG))

    head_tx = _pick_translation(landmark_tx, head_tx_norm, HEAD_TX_NORM_GAIN)
    head_ty = _pick_translation(landmark_ty, head_ty_norm, HEAD_TY_NORM_GAIN)
    head_tx = _cap(head_tx, HEAD_TX_MAX, limit_face)
    head_ty = _cap(head_ty, HEAD_TY_MAX, limit_face)

    # --- Roll from eye line (landmark fallback when RelativePose missing) ---
    a_live = _eye_line_angle(live)
    a_origin = _eye_line_angle(origin)
    if head_roll_deg is not None:
        roll = float(head_roll_deg) * g
    elif a_live is not None and a_origin is not None:
        roll = np.degrees(_wrap_pi(a_live - a_origin)) * g
    else:
        roll = 0.0
    if limit_face:
        roll = _clip(roll, -roll_left, roll_right)

    # --- Yaw / pitch: prefer calibrated RelativePose, else landmark geometry ---
    # Camera mesh shows raw landmark rotation even when CenterCalibration has
    # not marked rel.calibrated yet. Without this fallback, yaw/pitch stay 0
    # and the anime face looks locked while the webcam overlay turns.
    geom_live = _geom_head_angles(live)
    geom_origin = _geom_head_angles(origin)

    def _wrap_deg(delta: float) -> float:
        return float(((delta + 180.0) % 360.0) - 180.0)

    geom_yaw = (
        _wrap_deg(geom_live[1] - geom_origin[1]) * g
        if geom_live is not None and geom_origin is not None
        else None
    )
    geom_pitch = (
        _wrap_deg(geom_live[0] - geom_origin[0]) * g
        if geom_live is not None and geom_origin is not None
        else None
    )

    def _fuse_angle(calibrated: float | None, geometric: float | None) -> float:
        """Keep calibrated angles authoritative without accepting stale zeros."""
        if calibrated is None:
            return float(geometric or 0.0)
        calibrated = float(calibrated) * g
        if geometric is None:
            return calibrated
        # CenterCalibration can briefly publish a near-zero sample after a
        # recenter while the landmark geometry already shows a turn.
        if abs(calibrated) < 0.35 * abs(geometric):
            return float(geometric)
        if calibrated * geometric > 0.0:
            return float(0.80 * calibrated + 0.20 * geometric)
        return calibrated

    yaw = _fuse_angle(head_yaw_deg, geom_yaw)
    pitch = _fuse_angle(head_pitch_deg, geom_pitch)
    # Even with optional tight limiters disabled, keep turns in the range the
    # 2D rig can represent cleanly. A physical 90° turn maps to a strong 45°
    # character turn instead of folding the mesh.
    if limit_face:
        yaw = _clip(yaw, -yaw_left, yaw_right)
    if limit_face:
        pitch = _clip(pitch, -pitch_up, pitch_down)
    yaw = _clip(yaw, -min(45.0, max(yaw_left, 1e-6)), min(45.0, max(yaw_right, 1e-6)))
    pitch = _clip(pitch, -min(30.0, max(pitch_up, 1e-6)), min(30.0, max(pitch_down, 1e-6)))

    # --- Forward/back as calibrated face-size scale ---
    # Eye-to-eye span shrinks under yaw (perspective). Undo that so a turn is
    # not read as leaning away. Without this, returning to center after a turn
    # often leaves a slightly larger live span than the locked origin and the
    # character mesh ratchets wider.
    yaw_cos = max(abs(float(np.cos(np.radians(yaw)))), 0.55)
    live_span_for_scale = float(live_rigid_span) / yaw_cos
    head_scale = live_span_for_scale / max(origin_rigid_span, 1e-3)
    if limit_face:
        head_scale = _clip(head_scale, HEAD_SCALE_MIN, HEAD_SCALE_MAX)
    # Soften: blend toward 1 so leaning doesn't dominate.
    head_scale = 1.0 + 0.55 * (head_scale - 1.0)
    # Facing the camera: crush residual span noise. Expand especially hard so
    # "look left → come back" can't leave a permanently wider mesh.
    frontal = max(0.0, 1.0 - (abs(yaw) / 16.0 + abs(pitch) / 20.0))
    expand_keep = 1.0 - 0.85 * frontal
    shrink_keep = 1.0 - 0.45 * frontal
    if head_scale > 1.0:
        head_scale = 1.0 + (head_scale - 1.0) * expand_keep
    else:
        head_scale = 1.0 + (head_scale - 1.0) * shrink_keep

    # --- Blink ---
    def _blink(eye: tuple[int, ...]) -> float:
        depth_live = _eye_lid_depth(live, eye)
        depth_origin = _eye_lid_depth(origin, eye)
        if depth_live is None or depth_origin is None or depth_origin < 1e-5:
            return 1.0
        # A centered open eye is 1.0. Tracker noise below 12% is closed, while
        # 92% and above is fully open; smoothstep keeps half blinks continuous.
        ratio = float(depth_live / depth_origin)
        return _smoothstep(0.12, 0.92, ratio)

    blink_l = _blink(L_EYE)
    blink_r = _blink(R_EYE)

    # --- Brows (vertical vs eye center, relative to origin) ---
    def _brow(brow: tuple[int, ...], eye: tuple[int, ...]) -> float:
        b_l = _mean_xy(live, brow)
        b_o = _mean_xy(origin, brow)
        e_l = _mean_xy(live, eye)
        e_o = _mean_xy(origin, eye)
        if b_l is None or b_o is None or e_l is None or e_o is None:
            return 0.0
        # Relative brow-eye distance change, mapped to character fh.
        d_live = float(b_l[1] - e_l[1])
        d_origin = float(b_o[1] - e_o[1])
        delta = g * scale_to_char * (d_live - d_origin) / fh
        return _cap(delta, BROW_RAISE_MAX, limit_brows)

    brow_l = _brow(LEFT_BROW, L_EYE)
    brow_r = _brow(RIGHT_BROW, R_EYE)

    # --- Mouth: 4 scalars on the authored contour (open / smile / width / asym) ---
    mouth_live = _mouth_local_metrics(live)
    mouth_origin = _mouth_local_metrics(origin)
    mouth_open = 0.0
    mouth_form = 0.0
    mouth_asym = 0.0
    mouth_gap_delta = 0.0
    smile = 0.0
    if mouth_live is not None and mouth_origin is not None:
        open_live, lift_l_live, lift_r_live, width_live = mouth_live
        open_origin, lift_l_origin, lift_r_origin, width_origin = mouth_origin
        closed_floor = _clip(open_origin, 0.04, 0.12)
        mouth_open = _smoothstep(closed_floor, closed_floor + 0.42, open_live)
        mouth_gap_delta = (open_live - open_origin) * g
        width_delta = width_live / max(width_origin, 1e-4) - 1.0
        mouth_form = float(np.tanh(width_delta / 0.18))
        asym_live = lift_l_live - lift_r_live
        asym_origin = lift_l_origin - lift_r_origin
        mouth_asym = float(np.tanh((asym_live - asym_origin) / 0.12))
        live_lift = 0.5 * (lift_l_live + lift_r_live)
        origin_lift = 0.5 * (lift_l_origin + lift_r_origin)
        smile = g * (live_lift - origin_lift)
        smile = _cap(smile, MOUTH_SMILE_MAX, limit_mouth)
        mouth_open = _clip(mouth_open, 0.0, 1.0)
        mouth_form = _clip(mouth_form, -1.0, 1.0)

    raw_gaze: dict[int, tuple[float, float]] = {}
    for iris, eye in IRIS_EYE_PAIRS:
        if not (_vis(live, iris) and _vis(origin, iris)):
            continue
        g_live = _normalized_gaze(live, iris, eye)
        g_origin = _normalized_gaze(origin, iris, eye)
        gx = g_live[0] - g_origin[0]
        gy = g_live[1] - g_origin[1]
        gx = _cap(gx, GAZE_MAX, limit_eyes)
        gy = _cap(gy, GAZE_MAX, limit_eyes)
        raw_gaze[iris] = (gx, gy)

    # Human eyes normally move conjugately. Fuse their shared motion and keep
    # only a small residual so independent detector noise cannot make the
    # character cross-eyed. Both eye tuples use image-left→right X orientation.
    if raw_gaze:
        shared_x = float(np.mean([v[0] for v in raw_gaze.values()]))
        shared_y = float(np.mean([v[1] for v in raw_gaze.values()]))
        gaze_by_iris = {}
        for iris in (RIGHT_IRIS, LEFT_IRIS):
            measured = raw_gaze.get(iris, (shared_x, shared_y))
            residual_x = measured[0] - shared_x
            residual_y = measured[1] - shared_y
            if limit_eyes:
                residual_x = _clip(residual_x, -0.15, 0.15)
                residual_y = _clip(residual_y, -0.15, 0.15)
                gaze_by_iris[iris] = (
                    _clip(shared_x + 0.25 * residual_x, -GAZE_MAX, GAZE_MAX),
                    _clip(shared_y + 0.25 * residual_y, -GAZE_MAX, GAZE_MAX),
                )
            else:
                gaze_by_iris[iris] = (
                    shared_x + residual_x,
                    shared_y + residual_y,
                )
    else:
        gaze_by_iris = {}
    gaze_r = gaze_by_iris.get(RIGHT_IRIS, (0.0, 0.0))
    gaze_l = gaze_by_iris.get(LEFT_IRIS, (0.0, 0.0))

    # --- Body joint angle deltas ---
    body_d: dict[int, float] = {}
    parents = {
        BODY_NOSE: NECK,
        R_SHOULDER: NECK,
        L_SHOULDER: NECK,
        CHEST: NECK,
        R_ELBOW: R_SHOULDER,
        L_ELBOW: L_SHOULDER,
    }
    angle_caps = {
        BODY_NOSE: np.radians(18.0),
        R_SHOULDER: np.radians(SHOULDER_ANGLE_MAX_DEG),
        L_SHOULDER: np.radians(SHOULDER_ANGLE_MAX_DEG),
        CHEST: np.radians(CHEST_ANGLE_MAX_DEG),
        R_ELBOW: np.radians(ELBOW_ANGLE_MAX_DEG),
        L_ELBOW: np.radians(ELBOW_ANGLE_MAX_DEG),
    }
    for child, par in parents.items():
        if not (
            _vis(live, child)
            and _vis(live, par)
            and _vis(origin, child)
            and _vis(origin, par)
        ):
            continue
        a_l = _angle(live[par, :2], live[child, :2])
        a_o = _angle(origin[par, :2], origin[child, :2])
        d = _wrap_pi(a_l - a_o)
        cap = angle_caps.get(child, np.radians(30.0))
        # Ignore sub-degree tracker jitter only. A 3° deadzone plus 0.65 damp
        # ate desk-webcam arm/shoulder motion and left the overlay frozen.
        deadzone = np.radians(0.35)
        if abs(d) <= deadzone:
            d = 0.0
        else:
            d = np.sign(d) * (abs(d) - deadzone)
        body_d[child] = _clip(d * g, -cap, cap)
        body_d[child] = _clip(body_d[child], -np.radians(90.0), np.radians(90.0))

    return SemanticControls(
        head_tx=head_tx,
        head_ty=head_ty,
        head_roll_deg=roll,
        head_yaw_deg=yaw,
        head_pitch_deg=pitch,
        head_scale=head_scale,
        blink_l=blink_l,
        blink_r=blink_r,
        brow_l=brow_l,
        brow_r=brow_r,
        mouth_open=mouth_open,
        mouth_smile=smile,
        mouth_form=mouth_form,
        mouth_asym=mouth_asym,
        mouth_gap_delta=mouth_gap_delta,
        mouth_snapped=False,
        gaze_l=gaze_l,
        gaze_r=gaze_r,
        body_d_angle=body_d,
    )


def _apply_head_warp(
    local: np.ndarray,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    scale: float,
) -> np.ndarray:
    """Rigid-ish local face transform with mild 2.5D foreshortening."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    # Roll first.
    xy = _rot2(local, roll)
    # Yaw: foreshorten X + X shear from Y (turn toward camera-left/right).
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    xy = np.array(
        [
            xy[0] * (1.0 - YAW_FORESHORTEN + YAW_FORESHORTEN * cos_y)
            - YAW_WARP * sin_y * abs(xy[1]),
            xy[1] + 0.14 * sin_y * xy[0],
        ],
        dtype=np.float32,
    )
    # Pitch: foreshorten Y + vertical shift of upper/lower features.
    cos_p = float(np.cos(pitch))
    sin_p = float(np.sin(pitch))
    xy = np.array(
        [
            xy[0] + 0.10 * sin_p * xy[1],
            xy[1] * (1.0 - PITCH_FORESHORTEN + PITCH_FORESHORTEN * cos_p)
            - PITCH_WARP * sin_p * abs(xy[0]),
        ],
        dtype=np.float32,
    )
    return (xy * float(scale)).astype(np.float32)


def apply_eye_expression(
    out: np.ndarray,
    eye_idxs: tuple[int, ...],
    blink: float,
) -> None:
    """Move the upper-lid midpoint onto the corner chord as the eye closes."""
    if len(eye_idxs) != 3 or not all(_vis(out, i) for i in eye_idxs):
        return
    left, upper, right = eye_idxs
    ax, ay = (float(out[left, 0]), float(out[left, 1]))
    bx, by = (float(out[right, 0]), float(out[right, 1]))
    px = float(out[upper, 0])
    if abs(bx - ax) > 1e-5:
        along = _clip((px - ax) / (bx - ax), 0.0, 1.0)
        chord_y = ay + along * (by - ay)
    else:
        chord_y = 0.5 * (ay + by)
    openness = _clip(blink, BLINK_MIN, BLINK_MAX)
    out[upper, 1] = float(chord_y + (float(out[upper, 1]) - chord_y) * openness)
    for i in eye_idxs:
        out[i, 2] = max(float(out[i, 2]), 0.85)
        out[i, 3] = 1.0


def reconstruct_pose(
    ref_keypoints: np.ndarray,
    rig: ReferenceRig,
    controls: SemanticControls,
    *,
    body_method: str = "unknown",
    body_lost: bool = False,
    prev_body: np.ndarray | None = None,
    limit_eyes: bool = True,
    limit_mouth: bool = True,
) -> np.ndarray:
    """Build character pose from reference proportions + semantic controls."""
    from .live_poser_client import body_method_kind

    ref = _as37(ref_keypoints)
    out = ref.copy()
    fh = rig.face_height
    warp_mix = 1.0

    # --- Expressions in the reference-local face frame ---
    local_face = rig.local_face.copy()

    for brow, delta in ((LEFT_BROW, controls.brow_l), (RIGHT_BROW, controls.brow_r)):
        for i in brow:
            if _vis(ref, i):
                local_face[i, 1] = float(local_face[i, 1] + delta * fh)

    _apply_mouth_expression(
        local_face, ref, rig, controls, limit_mouth=limit_mouth
    )

    # --- One authoritative head transform ---
    # Pitch nod is applied here (after in-place warp offset removal) so a
    # 15–20° look-down actually moves the chin instead of only foreshortening.
    pitch_nod = float(np.sin(np.radians(controls.head_pitch_deg))) * PITCH_NOD * fh
    new_center = rig.face_center + np.array(
        [controls.head_tx * fh, controls.head_ty * fh + pitch_nod], dtype=np.float32
    )
    yaw_sin = float(np.sin(np.radians(controls.head_yaw_deg)))

    def _yaw_depth(slot: int) -> float:
        if slot in NOSE:
            return 1.0
        if slot in MOUTH:
            return 0.52
        if slot in L_EYE + R_EYE:
            return 0.34
        if slot in BROWS:
            return 0.25
        if slot in OUTLINE:
            return -0.18
        return 0.0

    warped_face: dict[int, np.ndarray] = {}
    warp_deltas: list[np.ndarray] = []
    for i in FACE_SLOTS:
        if not _vis(ref, i):
            continue
        local = local_face[i]
        warped = _apply_head_warp(
            local,
            yaw_deg=controls.head_yaw_deg,
            pitch_deg=controls.head_pitch_deg,
            roll_deg=controls.head_roll_deg,
            scale=controls.head_scale,
        )
        if warp_mix < 0.999:
            rest = _apply_head_warp(
                local,
                yaw_deg=0.0,
                pitch_deg=0.0,
                roll_deg=controls.head_roll_deg,
                scale=1.0,
            )
            warped = (rest * (1.0 - warp_mix) + warped * warp_mix).astype(np.float32)
        warped[0] -= yaw_sin * YAW_FEATURE_SHIFT * fh * _yaw_depth(i)
        baseline = _apply_head_warp(
            local,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=controls.head_roll_deg,
            scale=controls.head_scale,
        )
        warped_face[i] = warped
        warp_deltas.append(warped - baseline)

    # Yaw/pitch shears must turn features inside the head, not translate the
    # entire face away from its neck. Remove their mean displacement and reuse
    # the same offset for irises.
    head_warp_offset = (
        np.mean(np.stack(warp_deltas), axis=0).astype(np.float32)
        if warp_deltas
        else np.zeros(2, dtype=np.float32)
    )
    for i, warped in warped_face.items():
        warped = warped - head_warp_offset
        out[i, 0] = float(new_center[0] + warped[0])
        out[i, 1] = float(new_center[1] + warped[1])
        out[i, 2] = max(float(out[i, 2]), 0.85)
        out[i, 3] = 1.0

    # --- Blink on top of warped eyes ---
    # The schema stores corner / upper-mid / corner. Keep both corners fixed
    # and bend the upper lid onto their chord, which preserves eye width.
    apply_eye_expression(out, L_EYE, controls.blink_l)
    apply_eye_expression(out, R_EYE, controls.blink_r)

    # --- Iris: head-warp from ref, then apply relative gaze in the correct eye ---
    for iris, eye_idxs in IRIS_EYE_PAIRS:
        if not _vis(ref, iris):
            continue
        # Start from the same head transform as the face (keeps iris on its eye).
        local = ref[iris, :2] - rig.face_center
        warped = _apply_head_warp(
            local.astype(np.float32),
            yaw_deg=controls.head_yaw_deg,
            pitch_deg=controls.head_pitch_deg,
            roll_deg=controls.head_roll_deg,
            scale=controls.head_scale,
        )
        if warp_mix < 0.999:
            rest = _apply_head_warp(
                local.astype(np.float32),
                yaw_deg=0.0,
                pitch_deg=0.0,
                roll_deg=controls.head_roll_deg,
                scale=1.0,
            )
            warped = (rest * (1.0 - warp_mix) + warped * warp_mix).astype(np.float32)
        warped[0] -= yaw_sin * YAW_FEATURE_SHIFT * fh * 0.34 * warp_mix
        base = new_center + warped - head_warp_offset
        out[iris, 0] = float(base[0])
        out[iris, 1] = float(base[1])
        out[iris, 2] = max(float(out[iris, 2]), 0.85)
        out[iris, 3] = 1.0

        gaze = controls.gaze_r if iris == RIGHT_IRIS else controls.gaze_l
        center = _mean_xy(out, eye_idxs)
        if center is None:
            continue
        paired_blink = controls.blink_l if iris == RIGHT_IRIS else controls.blink_r
        if paired_blink <= IRIS_HIDE_BLINK:
            # A visible pupil on a line-shaped lid reads as an open eye. Hide
            # only at near-full closure; partial blinks keep normal gaze.
            out[iris, 0] = float(center[0])
            out[iris, 1] = float(center[1])
            out[iris, 2] = 0.0
            out[iris, 3] = 0.0
            continue
        w = _eye_width(out, eye_idxs) or (0.1 * fh)
        a = _eye_aperture(out, eye_idxs) or (0.03 * fh)
        eye_a = out[eye_idxs[0], :2].astype(np.float64)
        eye_b = out[eye_idxs[2], :2].astype(np.float64)
        axis_x = eye_b - eye_a
        axis_len = float(np.linalg.norm(axis_x))
        if axis_len < 1e-5:
            continue
        axis_x /= axis_len
        axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float64)
        if axis_y[1] < 0.0:
            axis_y *= -1.0
        # Offset along the post-transform eye basis, not global screen axes.
        gaze_strength = _smoothstep(IRIS_HIDE_BLINK, 0.45, paired_blink)
        pupil = (
            base.astype(np.float64)
            + axis_x * (gaze_strength * 0.42 * w * gaze[0])
            + axis_y * (gaze_strength * 0.28 * a * gaze[1])
        )
        if limit_eyes:
            # One eye-local socket clamp. Static-reference iris travel clamps are
            # intentionally skipped for constrained retarget output.
            rel = pupil - center.astype(np.float64)
            h = _clip(float(np.dot(rel, axis_x)), -0.55 * w, 0.55 * w)
            v_limit = max(0.55 * a, 0.025 * fh)
            v = _clip(float(np.dot(rel, axis_y)), -v_limit, v_limit)
            pupil = center.astype(np.float64) + axis_x * h + axis_y * v
        out[iris, 0] = float(pupil[0])
        out[iris, 1] = float(pupil[1])

    # Last net after the head warp: keep the mouth off the nose and inside
    # the authored lip rows. Live sanitize leaves topology off.
    out = clamp_mouth_anatomy(out, fh, ref=ref, limit_mouth=limit_mouth)

    # --- Body: held / angle-based reconstruction with ref bone lengths ---
    kind = body_method_kind(body_method)
    if body_lost or kind == "held":
        if prev_body is not None and np.asarray(prev_body).shape == (7, 4):
            pb = np.asarray(prev_body, dtype=np.float32)
            for i in range(7):
                if float(pb[i, 3]) >= 0.5:
                    out[30 + i] = pb[i]
        enforce_head_body_attachment(out, ref)
        return out

    # Neck follows face (0.35 / 0.25 mix).
    if _vis(ref, NECK):
        follow = 0.35
        out[NECK, 0] = float(rig.neck[0] + controls.head_tx * fh * follow)
        out[NECK, 1] = float(
            rig.neck[1] + controls.head_ty * fh * follow * (0.25 / 0.35)
        )
        out[NECK, 2] = max(float(out[NECK, 2]), 0.85)
        out[NECK, 3] = 1.0

    d_angles = controls.body_d_angle or {}
    # Reconstruct children from parents in dependency order.
    order = (R_SHOULDER, L_SHOULDER, CHEST, R_ELBOW, L_ELBOW)
    for child in order:
        if child not in rig.bone_len or child not in rig.bone_angle:
            continue
        par = rig.parent[child]
        if not _vis(out, par):
            continue
        rest = rig.bone_angle[child]
        d = float(d_angles.get(child, 0.0))
        ang = rest + d
        length = rig.bone_len[child]
        parent_xy = out[par, :2]
        out[child, 0] = float(parent_xy[0] + length * np.cos(ang))
        out[child, 1] = float(parent_xy[1] + length * np.sin(ang))
        out[child, 2] = max(float(out[child, 2]), 0.85)
        out[child, 3] = 1.0

    # Body-nose and face-nose represent the same physical head anchor. Point
    # the reference-length neck bone toward the transformed face nose for every
    # active body tracker so head turns cannot send the body the other way.
    if _vis(out, BODY_NOSE) and _vis(out, NECK) and _vis(out, 15):
        direction = out[15, :2].astype(np.float64) - out[NECK, :2].astype(np.float64)
        direction_len = float(np.linalg.norm(direction))
        bone_len = rig.bone_len.get(BODY_NOSE, direction_len)
        if direction_len > 1e-5 and bone_len > 1e-5:
            body_nose = out[NECK, :2] + direction / direction_len * bone_len
            out[BODY_NOSE, 0] = float(body_nose[0])
            out[BODY_NOSE, 1] = float(body_nose[1])

    enforce_head_body_attachment(out, ref)
    return out


def apply_constrained_retarget(
    ref_keypoints: np.ndarray,
    live_keypoints: np.ndarray,
    live_origin: np.ndarray,
    *,
    gain: float = 1.0,
    body_method: str = "unknown",
    body_lost: bool = False,
    prev_body: np.ndarray | None = None,
    head_yaw_deg: float | None = None,
    head_pitch_deg: float | None = None,
    head_roll_deg: float | None = None,
    head_tx_norm: float | None = None,
    head_ty_norm: float | None = None,
    limit_face: bool = True,
    limit_brows: bool = True,
    limit_eyes: bool = True,
    limit_nose: bool = True,
    limit_mouth: bool = True,
    reference_rig: ReferenceRig | None = None,
    prev_controls: SemanticControls | None = None,
    controls_out: list | None = None,
    motion: dict[str, float] | None = None,
) -> np.ndarray:
    """End-to-end constrained retarget (pre-sanitize).

    ``limit_nose`` is accepted for API symmetry with sanitize travel clamps;
    the nose has no separate semantic expression channel.
    """
    rig = reference_rig if reference_rig is not None else build_reference_rig(ref_keypoints)
    controls = extract_controls(
        live_keypoints,
        live_origin,
        rig,
        head_yaw_deg=head_yaw_deg,
        head_pitch_deg=head_pitch_deg,
        head_roll_deg=head_roll_deg,
        head_tx_norm=head_tx_norm,
        head_ty_norm=head_ty_norm,
        gain=gain,
        limit_face=limit_face,
        limit_brows=limit_brows,
        limit_eyes=limit_eyes,
        limit_mouth=limit_mouth,
        motion=motion,
    )
    if controls_out is not None:
        controls_out.append(controls)
    return reconstruct_pose(
        ref_keypoints,
        rig,
        controls,
        body_method=body_method,
        body_lost=body_lost,
        prev_body=prev_body,
        limit_eyes=limit_eyes,
        limit_mouth=limit_mouth,
    )


def enforce_proportion_invariants(
    out: np.ndarray,
    ref: np.ndarray,
    *,
    sanitize_face: bool = True,
    sanitize_body: bool = True,
    tol_face: float = 0.08,
    tol_body: float = 0.06,
    limit_face: bool = True,
    limit_eyes: bool = True,
    limit_mouth: bool = True,
) -> None:
    """Final safety net: snap defining distances back toward reference.

    ``tol_*`` are fractional allowances relative to the reference measurement.
    """
    o = _as37(out)
    r = _as37(ref)

    # Face width: scale outline / brows / eyes / nose / mouth laterally about face center.
    fw_ref = face_width(r)
    fw_out = face_width(o)
    if sanitize_face and limit_face and fw_ref > 1e-4 and fw_out > 1e-4:
        ratio = fw_out / fw_ref
        if abs(ratio - 1.0) > tol_face:
            target = _clip(ratio, 1.0 - tol_face, 1.0 + tol_face)
            fix = target / ratio
            c = face_center(o)
            for i in list(FACE_SLOTS) + [RIGHT_IRIS, LEFT_IRIS]:
                if _vis(o, i):
                    o[i, 0] = float(c[0] + (float(o[i, 0]) - float(c[0])) * fix)

    # Eye widths: restore each eye's horizontal span about its center.
    for eye in ((L_EYE, R_EYE) if (sanitize_face and limit_eyes) else ()):
        w_ref = _eye_width(r, eye)
        w_out = _eye_width(o, eye)
        c = _mean_xy(o, eye)
        if w_ref is None or w_out is None or c is None or w_out < 1e-5:
            continue
        ratio = w_out / w_ref
        if abs(ratio - 1.0) > tol_face:
            target = _clip(ratio, 1.0 - tol_face, 1.0 + tol_face)
            fix = target / ratio
            for i in eye:
                if _vis(o, i):
                    o[i, 0] = float(c[0] + (float(o[i, 0]) - float(c[0])) * fix)

    if not sanitize_body:
        return

    # Body bone lengths from current parents.
    parents = {
        BODY_NOSE: NECK,
        R_SHOULDER: NECK,
        L_SHOULDER: NECK,
        CHEST: NECK,
        R_ELBOW: R_SHOULDER,
        L_ELBOW: L_SHOULDER,
    }
    for child, par in parents.items():
        if not (_vis(o, child) and _vis(o, par) and _vis(r, child) and _vis(r, par)):
            continue
        len_ref = _length(r[par, :2], r[child, :2])
        len_out = _length(o[par, :2], o[child, :2])
        if len_ref < 1e-5 or len_out < 1e-5:
            continue
        ratio = len_out / len_ref
        if abs(ratio - 1.0) <= tol_body:
            continue
        target = _clip(ratio, 1.0 - tol_body, 1.0 + tol_body)
        # Rebuild child along current direction at corrected length.
        direction = o[child, :2] - o[par, :2]
        n = float(np.linalg.norm(direction))
        if n < 1e-6:
            continue
        new_len = len_ref * target
        o[child, 0] = float(o[par, 0] + direction[0] / n * new_len)
        o[child, 1] = float(o[par, 1] + direction[1] / n * new_len)

    # Shoulder span soft lock.
    if all(_vis(o, i) and _vis(r, i) for i in (R_SHOULDER, L_SHOULDER, NECK)):
        sw_ref = _length(r[R_SHOULDER, :2], r[L_SHOULDER, :2])
        sw_out = _length(o[R_SHOULDER, :2], o[L_SHOULDER, :2])
        if sw_ref > 1e-4 and sw_out > 1e-4:
            ratio = sw_out / sw_ref
            if abs(ratio - 1.0) > tol_body:
                target = _clip(ratio, 1.0 - tol_body, 1.0 + tol_body)
                fix = (sw_ref * target) / sw_out
                mid = 0.5 * (o[R_SHOULDER, :2] + o[L_SHOULDER, :2])
                for i in (R_SHOULDER, L_SHOULDER):
                    o[i, 0] = float(mid[0] + (float(o[i, 0]) - float(mid[0])) * fix)
                    o[i, 1] = float(mid[1] + (float(o[i, 1]) - float(mid[1])) * fix)
