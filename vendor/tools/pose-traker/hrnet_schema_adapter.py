"""Convert native anime-HRNet 28 landmarks → project KEYPOINT_SCHEMA (label28).

Native checkpoint layout (demo.ipynb CONTOURS / hrnetv2 flip_indices):
  0–4   face contour
  5–10  eyebrows
  11–13 left eye top lid
  14–16 left eye bottom lid
  17–19 right eye top lid
  20–22 right eye bottom lid
  23    nose (single tip)
  24–27 mouth quad: L corner, upper mid, R corner, lower mid

Project KEYPOINT_SCHEMA / label28:
  0–10  same contour + brows
  11–13 left eye (outer / upper / inner)  ← native top lid
  14–16 nose L / tip / R                  ← synthesized around native 23
  17–19 right eye (inner / upper / outer) ← native top lid
  20–27 8-pt mouth                        ← expanded from native 24–27

Call this at the HRNet detector boundary BEFORE collapse repair, iris match,
JSON write, or training ingest. OpenSeeFace / OSF paths already produce
label28 and must not go through this adapter.
"""

from __future__ import annotations

import numpy as np

# Nose half-width as fraction of inter-eye (upper-lid mid) distance.
_NOSE_HALF_W = 0.14
# Slight upward offset for nose L/R relative to tip (fraction of eye_dist).
_NOSE_SIDE_LIFT = 0.04


def _finite_xy(pt: np.ndarray) -> bool:
    return bool(np.isfinite(pt[0]) and np.isfinite(pt[1]))


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def _eye_mid(pts: np.ndarray, idxs: tuple[int, ...]) -> tuple[float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for i in idxs:
        if i >= len(pts):
            continue
        if not _finite_xy(pts[i]):
            continue
        if pts.shape[1] > 2 and float(pts[i, 2]) < 0.05:
            continue
        xs.append(float(pts[i, 0]))
        ys.append(float(pts[i, 1]))
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _set_row(
    out: np.ndarray,
    i: int,
    x: float,
    y: float,
    score: float,
) -> None:
    out[i, 0] = float(x)
    out[i, 1] = float(y)
    if out.shape[1] > 2:
        out[i, 2] = float(score)
    if out.shape[1] > 3:
        out[i, 3] = 1.0 if float(score) >= 0.05 else 0.0


def hrnet_native_to_label28(pts: np.ndarray) -> np.ndarray:
    """Map native HRNet ``(28, 3|4)`` keypoints → project label28 schema.

    Returns a new array with the same trailing dim as ``pts``. Malformed /
    short inputs are returned unchanged (caller should validate separately).
    """
    src = np.asarray(pts, dtype=np.float32)
    if src.ndim != 2 or src.shape[0] < 28 or src.shape[1] < 2:
        return src

    cols = int(src.shape[1])
    out = np.zeros((28, cols), dtype=np.float32)

    # Contour + brows + upper lids copy through 1:1.
    for i in list(range(0, 14)) + list(range(17, 20)):
        out[i] = src[i]

    # Inter-eye distance from upper lids (schema eye slots after copy).
    l_eye = _eye_mid(src, (11, 12, 13))
    r_eye = _eye_mid(src, (17, 18, 19))
    eye_dist = 1.0
    if l_eye is not None and r_eye is not None:
        eye_dist = abs(float(r_eye[0]) - float(l_eye[0]))
        if eye_dist < 1e-6:
            eye_dist = 1.0

    # Nose: tip = native 23; L/R synthesized around tip.
    if _finite_xy(src[23]) and (src.shape[1] < 3 or float(src[23, 2]) >= 0.05):
        tip_x, tip_y = float(src[23, 0]), float(src[23, 1])
        tip_sc = float(src[23, 2]) if cols > 2 else 1.0
        half_w = _NOSE_HALF_W * eye_dist
        lift = _NOSE_SIDE_LIFT * eye_dist
        _set_row(out, 14, tip_x - half_w, tip_y - lift, tip_sc)
        _set_row(out, 15, tip_x, tip_y, tip_sc)
        _set_row(out, 16, tip_x + half_w, tip_y - lift, tip_sc)

    # Mouth: native 24/25/26/27 → schema 8-pt contour.
    # native: 24 L corner, 25 upper mid, 26 R corner, 27 lower mid
    mouth_ok = all(
        _finite_xy(src[i]) and (cols < 3 or float(src[i, 2]) >= 0.05)
        for i in (24, 25, 26, 27)
    )
    if mouth_ok:
        c_l = src[24, :2].astype(np.float64)
        u_m = src[25, :2].astype(np.float64)
        c_r = src[26, :2].astype(np.float64)
        l_m = src[27, :2].astype(np.float64)
        sc = min(float(src[i, 2]) if cols > 2 else 1.0 for i in (24, 25, 26, 27))

        # Schema:
        # 20 upper L, 21 upper mid, 22 upper R
        # 23 corner L, 24 lower L, 25 lower mid, 26 corner R, 27 lower R
        u_l = _lerp(c_l, u_m, 0.5)
        u_r = _lerp(u_m, c_r, 0.5)
        lo_l = _lerp(c_l, l_m, 0.5)
        lo_r = _lerp(c_r, l_m, 0.5)

        layout = {
            20: u_l,
            21: u_m,
            22: u_r,
            23: c_l,
            24: lo_l,
            25: l_m,
            26: c_r,
            27: lo_r,
        }
        for i, xy in layout.items():
            _set_row(out, i, float(xy[0]), float(xy[1]), sc)

    return out


def map_face_detector_results(face_results: list[dict]) -> list[dict]:
    """In-place map each face's ``keypoints`` from native HRNet → label28.

    Returns the same list for chaining. Non-dict / short keypoints are skipped.
    """
    for pred in face_results:
        kps = pred.get("keypoints")
        if kps is None:
            continue
        arr = np.asarray(kps, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 28:
            continue
        pred["keypoints"] = hrnet_native_to_label28(arr)
    return face_results
