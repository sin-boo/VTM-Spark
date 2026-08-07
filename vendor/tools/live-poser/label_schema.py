"""28-point face schema used by auto-labeling / training (KEYPOINT_SCHEMA).

Live Poser maps OpenSeeFace landmarks into this exact order so the overlay
matches what pose-traker writes into *_landmarks.json / full_stack labels.
"""

from __future__ import annotations

import cv2
import numpy as np

# Training / labeling face landmark ids 0..27 (see send2pod/KEYPOINT_SCHEMA.md)
FACE_LANDMARK_LABELS = {
    **{i: 'face_outline' for i in range(0, 5)},
    **{i: 'eyebrow' for i in range(5, 11)},
    **{i: 'eye' for i in list(range(11, 14)) + list(range(17, 20))},
    **{i: 'nose' for i in range(14, 17)},
    **{i: 'mouth' for i in range(20, 28)},
}

# Person-relative (character's left / right) — matches KEYPOINT_SCHEMA.
LEFT_BROW = (5, 6, 7)
RIGHT_BROW = (8, 9, 10)
LEFT_EYE = (11, 12, 13)
NOSE = (14, 15, 16)
RIGHT_EYE = (17, 18, 19)
MOUTH = tuple(range(20, 28))
OUTLINE = (0, 1, 2, 3, 4)

# Draw connections (same groups as labeling UI intent).
# Mouth order traces the lip contour (not raw id order 20..27).
MOUTH_DRAW = (20, 21, 22, 26, 27, 25, 24, 23)

LABEL_CONNECTIONS = [
    (OUTLINE, (80, 200, 255)),
    (LEFT_BROW, (80, 255, 160)),
    (RIGHT_BROW, (80, 255, 160)),
    (NOSE, (255, 180, 80)),
    (MOUTH_DRAW, (180, 80, 255)),
]

# OpenSeeFace 66 landmarks → our 28 training slots.
# OSF mouth is NOT iBUG-68: corners are 58 (person right) / 62 (person left),
# upper outer 48→52 (person right→left), lower outer 53→57 (person left→right).
# Eyes: 36-41 person right, 42-47 person left (43/44 upper, 46/47 lower).
_OSF_TO_28 = {
    # outline
    0: 0,
    1: 4,
    2: 8,
    3: 12,
    4: 16,
    # left brow (person left) ← OSF 22-26
    5: 22,
    6: 24,
    7: 26,
    # right brow (person right) ← OSF 17-21
    8: 21,
    9: 19,
    10: 17,
    # left eye: outer / upper-lid mid / inner (upper mid shows open vs closed)
    11: 45,
    12: 43,
    13: 42,
    # nose L / tip / R
    14: 31,
    15: 30,
    16: 35,
    # right eye: inner / upper-lid mid / outer
    17: 39,
    18: 37,
    19: 36,
    # mouth (person-relative) — OSF 66 layout
    # Use INNER lip (59-65) for upper/lower so a closed mouth collapses to a
    # slit. Outer lip (48-57) is lip thickness and always looks "open".
    20: 61,  # upper L (inner)
    21: 60,  # upper mid (inner)
    22: 59,  # upper R (inner)
    23: 62,  # corner L
    24: 63,  # lower L (inner)
    25: 64,  # lower mid (inner)
    26: 58,  # corner R
    27: 65,  # lower R (inner)
}


def osf_lms_to_image_xy(lms: np.ndarray) -> np.ndarray:
    """OpenSeeFace stores landmarks as (y, x, conf), not (x, y, conf).

    Confirmed by facetracker.py:
        cv2.circle(frame, (y, x), ...)
        bounds check: x >= height or y >= width

    Returns (N,3) as image (x, y, conf) for OpenCV drawing / our label schema.
    """
    if lms is None or len(lms) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.zeros((len(lms), 3), dtype=np.float32)
    out[:, 0] = lms[:, 1]  # image x
    out[:, 1] = lms[:, 0]  # image y
    if lms.shape[1] > 2:
        out[:, 2] = lms[:, 2]
    else:
        out[:, 2] = 1.0
    return out


def _mean_xy(lms_xy: np.ndarray, idxs: list[int]) -> np.ndarray:
    pts = []
    for i in idxs:
        if i < len(lms_xy):
            pts.append(lms_xy[i, 0:2].astype(np.float64))
    if not pts:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.mean(pts, axis=0)


def osf_to_label28(lms: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert OpenSeeFace landmarks → (28,3) schema + optional eye lower lids.

    Returns
    -------
    pts28 : (28, 3) image (x, y, conf)
    eye_lower : (2, 3) lower-lid midpoints for [person-left, person-right],
                or None if eyes unavailable. Used so open eyes draw as a
                closed aperture (upper + lower), not a flat line.
    """
    out = np.zeros((28, 3), dtype=np.float32)
    if lms is None or len(lms) == 0:
        return out, None
    xy = osf_lms_to_image_xy(lms)
    for dst, src in _OSF_TO_28.items():
        if src < len(xy):
            out[dst, 0] = float(xy[src, 0])
            out[dst, 1] = float(xy[src, 1])
            out[dst, 2] = float(xy[src, 2])

    eye_lower = None
    if len(xy) >= 48:
        # Upper lids (schema "center") — rises when the eye opens
        le_up = _mean_xy(xy, [43, 44])
        re_up = _mean_xy(xy, [37, 38])
        out[12, 0], out[12, 1] = le_up[0], le_up[1]
        out[12, 2] = max(float(out[12, 2]), 0.5)
        out[18, 0], out[18, 1] = re_up[0], re_up[1]
        out[18, 2] = max(float(out[18, 2]), 0.5)

        # Lower lids for aperture drawing
        le_lo = _mean_xy(xy, [46, 47])
        re_lo = _mean_xy(xy, [40, 41])
        eye_lower = np.zeros((2, 3), dtype=np.float32)
        eye_lower[0, 0:2] = le_lo
        eye_lower[0, 2] = 1.0
        eye_lower[1, 0:2] = re_lo
        eye_lower[1, 2] = 1.0

        # If inner lips are nearly closed, snap upper/lower to the mid-line so
        # tracker noise doesn't leave a false open oval. Keep a tight threshold
        # so mid-lip shape still tracks while the mouth is slightly open.
        if len(xy) >= 66:
            gap = abs(float(out[21, 1] - out[25, 1]))
            mouth_w = abs(float(out[23, 0] - out[26, 0])) + 1e-6
            if gap / mouth_w < 0.035:
                for up, lo in ((20, 24), (21, 25), (22, 27)):
                    mid_y = 0.5 * (float(out[up, 1]) + float(out[lo, 1]))
                    out[up, 1] = mid_y
                    out[lo, 1] = mid_y
    return out, eye_lower


def flip_label28_x(
    pts: np.ndarray,
    width: int,
    eye_lower: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Mirror points horizontally (for selfie display). Swaps L/R schema ids."""
    if pts is None or len(pts) == 0:
        return pts, eye_lower
    out = pts.copy()
    out[:, 0] = (width - 1) - out[:, 0]
    pairs = [
        (0, 4),
        (1, 3),
        (5, 10),
        (6, 9),
        (7, 8),
        (11, 19),
        (12, 18),
        (13, 17),
        (14, 16),
        (20, 22),
        (23, 26),
        (24, 27),
    ]
    for a, b in pairs:
        out[[a, b]] = out[[b, a]]
    lo = None
    if eye_lower is not None and len(eye_lower) >= 2:
        lo = eye_lower.copy()
        lo[:, 0] = (width - 1) - lo[:, 0]
        lo[[0, 1]] = lo[[1, 0]]
    return out, lo


def face_basis_from_label28(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Upright face basis from 28 labeling points.

    X = along eyes toward person's left (screen-right when facing camera, unmirrored)
    Y = up the face toward brows (screen-up when head is upright)
    Origin = nose tip (15)
    """
    if pts is None or not isinstance(pts, np.ndarray) or len(pts) < 28:
        return None
    left_eye = pts[list(LEFT_EYE), 0:2].astype(np.float64).mean(axis=0)
    right_eye = pts[list(RIGHT_EYE), 0:2].astype(np.float64).mean(axis=0)
    nose = pts[15, 0:2].astype(np.float64)
    left_brow = pts[list(LEFT_BROW), 0:2].astype(np.float64).mean(axis=0)
    right_brow = pts[list(RIGHT_BROW), 0:2].astype(np.float64).mean(axis=0)
    brow = (left_brow + right_brow) * 0.5
    chin = pts[2, 0:2].astype(np.float64)

    # Person left − person right ≈ screen-right when facing camera
    x = left_eye - right_eye
    xn = np.linalg.norm(x)
    if xn < 1e-6:
        return None
    x = x / xn

    y = brow - chin
    y = y - x * float(np.dot(y, x))
    yn = np.linalg.norm(y)
    if yn < 1e-6:
        y = np.array([-x[1], x[0]], dtype=np.float64)
        yn = np.linalg.norm(y)
    y = y / max(yn, 1e-6)
    if np.dot(y, brow - nose) < 0:
        y = -y

    return nose.copy(), x, y


def geom_head_angles(pts: np.ndarray) -> tuple[float, float, float]:
    """Pitch / yaw / roll proxies from 28-point geometry (degrees).

    More stable for centering than raw OpenSeeFace euler (which can look
    90° sideways). Roll from eye line; yaw from nose vs eye midline;
    pitch from nose vs eye line vertical.
    """
    if pts is None or len(pts) < 28:
        return 0.0, 0.0, 0.0
    left_eye = pts[list(LEFT_EYE), 0:2].astype(np.float64).mean(axis=0)
    right_eye = pts[list(RIGHT_EYE), 0:2].astype(np.float64).mean(axis=0)
    nose = pts[15, 0:2].astype(np.float64)
    chin = pts[2, 0:2].astype(np.float64)
    mid = (left_eye + right_eye) * 0.5
    eye_dist = float(np.linalg.norm(left_eye - right_eye)) + 1e-6

    roll = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
    # Yaw: nose left/right of eye midline, normalized by eye width
    yaw = float(np.degrees(np.arctan2(-(nose[0] - mid[0]), eye_dist))) * 1.4
    # Pitch: nose below/above eyes vs chin stretch
    face_h = float(np.linalg.norm(chin - mid)) + 1e-6
    pitch = float(np.degrees(np.arctan2((nose[1] - mid[1]), face_h))) * 1.4
    return pitch, yaw, roll


def _draw_eye_aperture(
    frame: np.ndarray,
    pts: np.ndarray,
    outer_id: int,
    upper_id: int,
    inner_id: int,
    lower_xy: np.ndarray | None,
    color: tuple[int, int, int],
) -> None:
    """Draw eye as outer→upper→inner→lower so open eyes aren't a flat line."""
    ids = (outer_id, upper_id, inner_id)
    if any(float(pts[i, 2]) < 0.15 for i in ids):
        return
    poly = [
        [int(pts[outer_id, 0]), int(pts[outer_id, 1])],
        [int(pts[upper_id, 0]), int(pts[upper_id, 1])],
        [int(pts[inner_id, 0]), int(pts[inner_id, 1])],
    ]
    if lower_xy is not None and float(lower_xy[2]) >= 0.15:
        poly.append([int(lower_xy[0]), int(lower_xy[1])])
        cv2.polylines(frame, [np.array(poly, np.int32)], True, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (int(lower_xy[0]), int(lower_xy[1])), 2, color, -1, cv2.LINE_AA)
    else:
        cv2.polylines(frame, [np.array(poly, np.int32)], False, color, 1, cv2.LINE_AA)


def draw_label28(
    frame: np.ndarray,
    pts: np.ndarray,
    *,
    show_ids: bool = True,
    eye_lower: np.ndarray | None = None,
) -> None:
    """Draw all 28 labeling landmarks + connections (same set as auto-labeler)."""
    if pts is None or len(pts) < 28:
        return
    for idxs, color in LABEL_CONNECTIONS:
        poly = []
        for i in idxs:
            if float(pts[i, 2]) < 0.15:
                continue
            poly.append([int(pts[i, 0]), int(pts[i, 1])])
        if len(poly) >= 2:
            closed = list(idxs) == list(MOUTH_DRAW)
            cv2.polylines(frame, [np.array(poly, np.int32)], closed, color, 1, cv2.LINE_AA)

    eye_color = (255, 120, 80)
    left_lo = eye_lower[0] if eye_lower is not None and len(eye_lower) >= 1 else None
    right_lo = eye_lower[1] if eye_lower is not None and len(eye_lower) >= 2 else None
    # person-left eye: outer(11) → upper(12) → inner(13) → lower
    _draw_eye_aperture(frame, pts, 11, 12, 13, left_lo, eye_color)
    # person-right eye: outer(19) → upper(18) → inner(17) → lower
    _draw_eye_aperture(frame, pts, 19, 18, 17, right_lo, eye_color)

    for i in range(28):
        if float(pts[i, 2]) < 0.15:
            continue
        x, y = int(pts[i, 0]), int(pts[i, 1])
        group = FACE_LANDMARK_LABELS.get(i, '')
        if group == 'face_outline':
            color = (80, 200, 255)
        elif group == 'eyebrow':
            color = (80, 255, 160)
        elif group == 'eye':
            color = (255, 120, 80)
        elif group == 'nose':
            color = (255, 180, 80)
        else:
            color = (180, 80, 255)
        cv2.circle(frame, (x, y), 3 if i != 15 else 5, color, -1, cv2.LINE_AA)
        if show_ids:
            cv2.putText(
                frame,
                str(i),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
