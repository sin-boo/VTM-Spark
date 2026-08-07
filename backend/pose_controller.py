"""Pose controller: sanitize and overlay mesh for KEYPOINT_SCHEMA.

Merged from pose_sanitize / pose_overlay so travel clamps and mesh drawing
share one module.
"""

from __future__ import annotations

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
RIGHT_IRIS = 28
LEFT_IRIS = 29

# Iris are named anatomically (person's L/R). Eye lid slots are named by image
# side: L_EYE (11-13) sits on image-left with the person's RIGHT iris (28);
# R_EYE (17-19) sits on image-right with the person's LEFT iris (29).
IRIS_EYE_PAIRS: tuple[tuple[int, tuple[int, ...]], ...] = (
    (RIGHT_IRIS, L_EYE),
    (LEFT_IRIS, R_EYE),
)

BODY_NOSE = 30
NECK = 31
R_SHOULDER = RIGHT_SHOULDER = 32
R_ELBOW = RIGHT_ELBOW = 33
L_SHOULDER = LEFT_SHOULDER = 34
L_ELBOW = LEFT_ELBOW = 35
CHEST = 36
BODY_SLOTS = (30, 31, 32, 33, 34, 35, 36)

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
    (BODY_NOSE, NECK),
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
IRIS_RIGHT = (255, 220, 0)
IRIS_LEFT = (0, 180, 255)


def _as37(kps: np.ndarray) -> np.ndarray:
    out = np.asarray(kps, dtype=np.float32)
    if out.ndim != 2 or out.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(
            f"Expected keypoints ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {out.shape}"
        )
    return out


def _vis(k: np.ndarray, i: int) -> bool:
    return float(k[i, 3]) >= 0.5


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

        if limit_mouth and limit_nose and _vis(out, 21) and nose_y is not None:
            min_um = nose_y + MOUTH_BELOW_NOSE * fh
            if float(out[21, 1]) < min_um:
                out[21, 1] = min_um
        if limit_mouth and _vis(out, 21) and _vis(out, 25):
            uy = float(out[21, 1])
            min_lm = uy + LIP_GAP_MIN * fh
            if float(out[25, 1]) < min_lm:
                out[25, 1] = min_lm
        if limit_mouth and _vis(out, 21):
            uy = float(out[21, 1])
            for i in (20, 22):
                if _vis(out, i) and float(out[i, 1]) < uy - 0.08 * fh:
                    out[i, 1] = uy - 0.08 * fh
        if limit_mouth and _vis(out, 25):
            ly = float(out[25, 1])
            for i in (24, 27):
                if _vis(out, i) and float(out[i, 1]) < ly - 0.05 * fh:
                    out[i, 1] = ly
            for i in (23, 26):
                if _vis(out, i):
                    um = float(out[21, 1]) if _vis(out, 21) else ly
                    out[i, 1] = float(np.clip(out[i, 1], um, ly + 0.05 * fh))

        if limit_mouth:
            mouth_bottom = None
            for i in (25, 24, 27, 2):
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


def nearest_keypoint(
    keypoints: np.ndarray,
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    *,
    max_dist: float = 18.0,
) -> int | None:
    """Return index of nearest visible keypoint in image pixel space, or None."""
    pix = normalized_to_pixels(keypoints, width, height)
    best_i: int | None = None
    best_d = float(max_dist)
    for i in range(NUM_KEYPOINTS):
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
    skeleton_lost: bool = False,
    mouth_snapped: bool = False,
    keypoints_are_pixels: bool = False,
) -> Image.Image:
    """Overlay Live-Poser-style mesh using model-space (37,4) keypoints.

    By default ``keypoints`` are ``norm_crop`` [-1,1]. Pass
    ``keypoints_are_pixels=True`` when xy are already absolute image pixels
    (camera diagnostic preview). When ``skeleton_lost`` is True the body
    overlay draws red instead of green. When ``mouth_snapped`` is True the
    mouth overlay switches from purple to blue.
    """
    if image is None or keypoints is None:
        return image
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

    if show_face:
        for idxs, color in FACE_CONNECTIONS:
            if list(idxs) == list(MOUTH_DRAW):
                color = mouth_color
            _polyline(
                draw,
                pix,
                idxs,
                color,
                closed=(list(idxs) == list(MOUTH_DRAW)),
                width=2,
            )
        # Eyes: hollow rings only — no aperture lines, no synthetic lower lid.
        eye_color = FACE_POINT_COLOR["eye"]
        for i in _EYE_LID_IDS:
            if not _visible(pix, i):
                continue
            x, y = _xy(pix, i)
            _draw_eye_point_hollow(draw, x, y, eye_color)
            if font is not None:
                draw.text((x + 7, y - 12), str(i), fill=(255, 255, 255), font=font)

        for i in range(28):
            if i in _EYE_LID_IDS:
                continue
            if not _visible(pix, i):
                continue
            x, y = _xy(pix, i)
            color = mouth_color if _group_of(i) == "mouth" else FACE_POINT_COLOR[_group_of(i)]
            r = 5 if i == 15 else 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255))
            if font is not None:
                draw.text((x + 5, y - 11), str(i), fill=(255, 255, 255), font=font)

    if show_iris:
        for idx, color, tag in (
            (28, IRIS_RIGHT, "R"),
            (29, IRIS_LEFT, "L"),
        ):
            if not _visible(pix, idx):
                continue
            x, y = _xy(pix, idx)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=color)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=1)
            if font is not None:
                draw.text((x + 8, y - 6), tag, fill=color, font=font)

    if show_skeleton:
        bone_color = SKELETON_LOST_COLOR if skeleton_lost else SKELETON_COLOR
        joint_color = JOINT_LOST_COLOR if skeleton_lost else JOINT_COLOR
        id_color = (255, 200, 200) if skeleton_lost else (230, 255, 230)
        body_pts: dict[int, tuple[int, int]] = {}
        for idx in range(BODY_NOSE, CHEST + 1):
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
                draw.text((x + 5, y - 10), str(idx), fill=id_color, font=font)
        if skeleton_lost and body_pts:
            anchor = body_pts.get(NECK) or body_pts.get(CHEST) or next(iter(body_pts.values()))
            draw.text(
                (anchor[0] + 8, anchor[1] + 14),
                "LOST",
                fill=SKELETON_LOST_COLOR,
                font=font or _try_font(12),
            )

    return out
