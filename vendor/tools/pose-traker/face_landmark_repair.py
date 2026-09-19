"""Repair anime HRNet face landmarks that collapse nose/mouth onto eyes.

Works on detector pixel coords ``(N,3)`` (x,y,score) and on real_stream
crop / sidecar tensors ``(N,4)`` (x,y,score,visible). Schema stays 28 face
pts (3 lid + iris elsewhere); this only rebuilds nose 14–16 and mouth 20–27
when collapse is detected.

IMPORTANT: When the mouth needs repair, we keep any detector points that
already sit in the mouth band and only rewrite the collapsed slots. A full
synthetic rebuild from nose/chin is last resort — that was shoving mouths
to the wrong place even when HRNet lower lips were usable.
"""

from __future__ import annotations

import numpy as np


def _set(out: np.ndarray, i: int, x: float, y: float, score: float = 1.0) -> None:
    out[i, 0] = float(x)
    out[i, 1] = float(y)
    out[i, 2] = float(score)
    if out.shape[1] > 3:
        out[i, 3] = 1.0


def _mean_xy(pts: np.ndarray, idxs: tuple[int, ...]) -> tuple[float, float] | None:
    xs, ys = [], []
    for i in idxs:
        if i < 0 or i >= len(pts):
            continue
        if pts.shape[1] > 2 and float(pts[i, 2]) < 0.05:
            continue
        if pts.shape[1] > 3 and float(pts[i, 3]) < 0.5:
            continue
        xs.append(float(pts[i, 0]))
        ys.append(float(pts[i, 1]))
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _face_anchors(
    pts: np.ndarray,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    float,
]:
    """Return (chin, nose, left_eye, right_eye, eye_dist)."""
    chin = _mean_xy(pts, (2,))
    nose = _mean_xy(pts, (15,)) or _mean_xy(pts, (14, 15, 16))
    l_eye = _mean_xy(pts, (11, 12, 13))
    r_eye = _mean_xy(pts, (17, 18, 19))
    eye_dist = 1.0
    if l_eye is not None and r_eye is not None:
        # Do NOT clamp with max(1.0): crop-normalized [-1,1] eye spans are ~0.2–0.5.
        # That clamp made mouth-band / collapse thresholds huge and dropped valid lips.
        eye_dist = abs(float(r_eye[0]) - float(l_eye[0]))
        if eye_dist < 1e-6:
            eye_dist = 1.0
    return chin, nose, l_eye, r_eye, eye_dist


def nose_needs_repair(pts: np.ndarray) -> bool:
    """True when nose cluster has collapsed onto an eye (common HRNet failure)."""
    chin, nose, l_eye, r_eye, eye_dist = _face_anchors(pts)
    if chin is None or nose is None or l_eye is None or r_eye is None:
        return True
    d_eye = min(
        float(np.hypot(nose[0] - l_eye[0], nose[1] - l_eye[1])),
        float(np.hypot(nose[0] - r_eye[0], nose[1] - r_eye[1])),
    )
    if d_eye < 0.22 * eye_dist:
        return True
    eye_mid_y = 0.5 * (float(l_eye[1]) + float(r_eye[1]))
    # Nose tip should sit below the eye band — use a soft margin so slightly
    # high noses (asymmetric wink / head tilt) are not rebuilt.
    if float(nose[1]) < eye_mid_y + 0.02 * eye_dist and d_eye < 0.45 * eye_dist:
        return True
    if float(nose[1]) > float(chin[1]) - 0.05 * eye_dist:
        return True
    return False


def _repair_nose_pixels(pts: np.ndarray, *, log_prefix: str | None) -> np.ndarray:
    """Rebuild nose L/tip/R between the eyes and chin."""
    out = np.asarray(pts, dtype=np.float32).copy()
    chin, _nose, l_eye, r_eye, eye_dist = _face_anchors(out)
    if chin is None or l_eye is None or r_eye is None:
        return out
    face_mid_x = 0.5 * (float(l_eye[0]) + float(r_eye[0]))
    eye_mid_y = 0.5 * (float(l_eye[1]) + float(r_eye[1]))
    tip_y = eye_mid_y + 0.32 * (float(chin[1]) - eye_mid_y)
    half_w = 0.14 * eye_dist
    layout = {
        14: (face_mid_x - half_w, tip_y - 0.04 * eye_dist),
        15: (face_mid_x, tip_y),
        16: (face_mid_x + half_w, tip_y - 0.04 * eye_dist),
    }
    for i, (x, y) in layout.items():
        _set(out, i, x, y, 0.95)
    if log_prefix:
        print(
            f"{log_prefix} Repaired nose landmarks (was on/above eye band); "
            f"tip=({face_mid_x:.4f},{tip_y:.4f})"
        )
    return out


def _point_in_mouth_band(
    x: float,
    y: float,
    *,
    nose: tuple[float, float],
    chin: tuple[float, float],
    l_eye: tuple[float, float],
    r_eye: tuple[float, float],
    face_mid_x: float,
    eye_dist: float,
) -> bool:
    if y < float(nose[1]) - 0.02 * eye_dist:
        return False
    # On the nose tip — that slot is a mouth id confused with the nose, not a lip.
    if float(np.hypot(x - float(nose[0]), y - float(nose[1]))) < 0.10 * eye_dist:
        return False
    if y > float(chin[1]) + 0.08 * eye_dist:
        return False
    if abs(x - face_mid_x) > 0.55 * eye_dist:
        return False
    d_eye = min(
        float(np.hypot(x - l_eye[0], y - l_eye[1])),
        float(np.hypot(x - r_eye[0], y - r_eye[1])),
    )
    if d_eye < 0.28 * eye_dist:
        return False
    return True


def mouth_needs_repair(pts: np.ndarray) -> bool:
    """True when upper-lip cluster is laterally/vertically off the face mid."""
    if pts.shape[0] < 28:
        return True
    chin, nose, l_eye, r_eye, eye_dist = _face_anchors(pts)
    if chin is None or nose is None or l_eye is None or r_eye is None:
        return True
    face_mid_x = 0.5 * (float(l_eye[0]) + float(r_eye[0]))
    upper = [i for i in (20, 21, 22) if float(pts[i, 2]) >= 0.15]
    lower = [i for i in (24, 25, 27) if float(pts[i, 2]) >= 0.15]
    if not upper:
        return True
    ux = float(np.mean([pts[i, 0] for i in upper]))
    uy = float(np.mean([pts[i, 1] for i in upper]))
    # Far from face midline (classic failure: upper mouth on right eye).
    if abs(ux - face_mid_x) > 0.28 * eye_dist:
        return True
    # Above nose tip (mouth cannot sit in the eye band).
    if uy < float(nose[1]) - 0.05 * eye_dist:
        return True
    # Upper-mid / upper lip parked on the nose while the lower lip is clearly below.
    # The old band test kept those slots (y ≈ nose) and the mouth contour stretched
    # from the real lips up to the nose tip.
    if lower:
        ly = float(np.mean([pts[i, 1] for i in lower]))
        if ly > float(nose[1]) + 0.12 * eye_dist:
            for i in (20, 21, 22):
                if float(pts[i, 2]) < 0.15:
                    continue
                d_nose = float(
                    np.hypot(float(pts[i, 0]) - float(nose[0]), float(pts[i, 1]) - float(nose[1]))
                )
                if d_nose < 0.10 * eye_dist:
                    return True
    # Upper cluster parked on an eye (stricter than old chin-distance ratio,
    # which false-triggered on valid mouths that sit closer to eyes than chin).
    d_eye = min(
        float(np.hypot(ux - l_eye[0], uy - l_eye[1])),
        float(np.hypot(ux - r_eye[0], uy - r_eye[1])),
    )
    if d_eye < 0.30 * eye_dist:
        return True
    if uy < 0.5 * (float(l_eye[1]) + float(r_eye[1])) + 0.08 * eye_dist:
        return True
    # Lower lip OK but far from upper → inconsistent mouth.
    if lower:
        lx = float(np.mean([pts[i, 0] for i in lower]))
        if abs(lx - ux) > 0.45 * eye_dist:
            return True
    return False


def _gap_floor(eye_dist: float) -> float:
    """Absolute min lip gap: pixels for image space, tiny for crop [-1,1]."""
    return 2.0 if eye_dist > 5.0 else 0.008


def _mouth_zone_samples(
    pts: np.ndarray,
    *,
    nose: tuple[float, float],
    chin: tuple[float, float],
    l_eye: tuple[float, float],
    r_eye: tuple[float, float],
    face_mid_x: float,
    eye_dist: float,
) -> tuple[list[float], list[float], list[int]]:
    """Collect mouth-slot xy that sit in the mouth band (not on an eye)."""
    xs: list[float] = []
    ys: list[float] = []
    ids: list[int] = []
    for i in range(20, 28):
        if i >= len(pts):
            continue
        if pts.shape[1] > 2 and float(pts[i, 2]) < 0.15:
            continue
        if pts.shape[1] > 3 and float(pts[i, 3]) < 0.5:
            continue
        x, y = float(pts[i, 0]), float(pts[i, 1])
        if not _point_in_mouth_band(
            x,
            y,
            nose=nose,
            chin=chin,
            l_eye=l_eye,
            r_eye=r_eye,
            face_mid_x=face_mid_x,
            eye_dist=eye_dist,
        ):
            continue
        xs.append(x)
        ys.append(y)
        ids.append(i)
    return xs, ys, ids


def _slot_ok(
    pts: np.ndarray,
    i: int,
    *,
    nose: tuple[float, float],
    chin: tuple[float, float],
    l_eye: tuple[float, float],
    r_eye: tuple[float, float],
    face_mid_x: float,
    eye_dist: float,
) -> bool:
    if i >= len(pts):
        return False
    if pts.shape[1] > 2 and float(pts[i, 2]) < 0.15:
        return False
    if pts.shape[1] > 3 and float(pts[i, 3]) < 0.5:
        return False
    return _point_in_mouth_band(
        float(pts[i, 0]),
        float(pts[i, 1]),
        nose=nose,
        chin=chin,
        l_eye=l_eye,
        r_eye=r_eye,
        face_mid_x=face_mid_x,
        eye_dist=eye_dist,
    )


def _repair_mouth_pixels(pts: np.ndarray, *, log_prefix: str | None) -> np.ndarray:
    """Fix collapsed mouth slots — keep detector points that already look right.

    Previous versions always rewrote all 8 mouth points from nose/chin geometry,
    which moved a usable mouth to the wrong place. Now:
      1. Keep any of 20–27 that already sit in the mouth band.
      2. Rebuild only the bad slots around the kept band (or a closed slit).
      3. Full synthetic layout only when nothing in-band remains.
    """
    out = np.asarray(pts, dtype=np.float32).copy()
    chin, nose, l_eye, r_eye, eye_dist = _face_anchors(out)
    if chin is None or nose is None or l_eye is None or r_eye is None:
        return out
    face_mid_x = 0.5 * (float(l_eye[0]) + float(r_eye[0]))
    gap_floor = _gap_floor(eye_dist)
    default_half_w = 0.22 * eye_dist

    xs, ys, kept_ids = _mouth_zone_samples(
        out,
        nose=nose,
        chin=chin,
        l_eye=l_eye,
        r_eye=r_eye,
        face_mid_x=face_mid_x,
        eye_dist=eye_dist,
    )
    kept = set(kept_ids)
    upper_kept = [i for i in (20, 21, 22) if i in kept]
    lower_kept = [i for i in (23, 24, 25, 26, 27) if i in kept]

    if len(ys) >= 2:
        ys_arr = np.asarray(ys, dtype=np.float64)
        xs_arr = np.asarray(xs, dtype=np.float64)
        q25, q75 = np.percentile(ys_arr, (25.0, 75.0))
        iqr = float(q75 - q25)
        if iqr > 1e-8:
            lo = float(q25 - 1.5 * iqr)
            hi = float(q75 + 1.5 * iqr)
            mask = (ys_arr >= lo) & (ys_arr <= hi)
            if int(np.count_nonzero(mask)) >= 2:
                ys_arr = ys_arr[mask]
                xs_arr = xs_arr[mask]
        face_mid_x = float(np.median(xs_arr))
        span_x = float(np.max(xs_arr) - np.min(xs_arr))
        half_w = max(0.55 * span_x, 0.16 * eye_dist)
        half_w = min(half_w, 0.35 * eye_dist)

        if upper_kept and lower_kept:
            um_y = float(np.mean([out[i, 1] for i in upper_kept]))
            lm_y = float(np.mean([out[i, 1] for i in lower_kept]))
            gap = max(gap_floor, lm_y - um_y)
            mode = f"keep-detector gap={gap:.4f} kept={sorted(kept)}"
        elif lower_kept and not upper_kept:
            # Classic HRNet failure: upper on eye, lower still on the mouth.
            # Anchor a closed slit on the lower-band median. Clear `kept` so we
            # rewrite a coherent mouth at that place (keeping raw lower scatter
            # leaves corners above the new upper lip).
            face_mid_x = float(np.median([out[i, 0] for i in lower_kept]))
            band_y = float(np.median([out[i, 1] for i in lower_kept]))
            gap = min(max(gap_floor, 0.025 * eye_dist), 0.05 * eye_dist)
            um_y = band_y - 0.5 * gap
            # width from lower-band span
            span_x = float(
                max(out[i, 0] for i in lower_kept) - min(out[i, 0] for i in lower_kept)
            )
            half_w = max(0.55 * span_x, 0.16 * eye_dist)
            half_w = min(half_w, 0.35 * eye_dist)
            kept = set()
            mode = (
                f"upper-from-lower-band center=({face_mid_x:.1f},{band_y:.1f}) "
                f"gap={gap:.4f} from_lower={sorted(lower_kept)}"
            )
        elif upper_kept and not lower_kept:
            um_y = float(np.mean([out[i, 1] for i in upper_kept]))
            face_mid_x = float(np.median([out[i, 0] for i in upper_kept]))
            gap = min(max(gap_floor, 0.025 * eye_dist), 0.05 * eye_dist)
            kept = set(upper_kept)  # keep good upper; fill lower from layout
            mode = (
                f"lower-from-upper-band um={um_y:.1f} gap={gap:.4f} "
                f"kept_upper={sorted(upper_kept)}"
            )
        else:
            # Corners-only etc. — use band median, closed-biased gap.
            face_mid_x = float(np.median(xs_arr))
            band_y = float(np.median(ys_arr))
            gap = min(max(gap_floor, 0.025 * eye_dist), 0.05 * eye_dist)
            um_y = band_y - 0.5 * gap
            kept = set()
            mode = f"band-slit center=({face_mid_x:.1f},{band_y:.1f}) gap={gap:.4f}"
    else:
        um_y = float(nose[1]) + 0.35 * (float(chin[1]) - float(nose[1]))
        gap = max(gap_floor, 0.06 * (float(chin[1]) - float(nose[1])))
        gap = min(gap, max(gap_floor * 2.5, 0.03 * eye_dist))
        half_w = default_half_w
        mode = f"synthetic closed gap={gap:.4f} (no in-band detector pts)"

    # Default layout for slots we must fill; kept slots retain detector xy.
    layout = {
        20: (face_mid_x - 0.55 * half_w, um_y),
        21: (face_mid_x, um_y),
        22: (face_mid_x + 0.55 * half_w, um_y),
        23: (face_mid_x - half_w, um_y + 0.35 * gap),
        24: (face_mid_x - 0.45 * half_w, um_y + gap),
        25: (face_mid_x, um_y + gap),
        26: (face_mid_x + half_w, um_y + 0.35 * gap),
        27: (face_mid_x + 0.45 * half_w, um_y + gap),
    }

    rewritten: list[int] = []
    for i, (x, y) in layout.items():
        if i in kept and _slot_ok(
            out,
            i,
            nose=nose,
            chin=chin,
            l_eye=l_eye,
            r_eye=r_eye,
            face_mid_x=face_mid_x,
            eye_dist=eye_dist,
        ):
            # Keep detector point; only nudge lower mid below upper if inverted.
            continue
        _set(out, i, x, y, max(float(out[i, 2]) if out.shape[1] > 2 else 0.95, 0.9))
        rewritten.append(i)

    # If we kept a lower mid above the (possibly new) upper mid, pull it down.
    if float(out[21, 3] if out.shape[1] > 3 else 1.0) >= 0.5 and float(
        out[25, 3] if out.shape[1] > 3 else 1.0
    ) >= 0.5:
        if float(out[25, 1]) < float(out[21, 1]) + gap_floor:
            out[25, 1] = float(out[21, 1]) + max(gap, gap_floor)

    if log_prefix:
        print(
            f"{log_prefix} Repaired mouth landmarks; UM=({float(out[21,0]):.4f},"
            f"{float(out[21,1]):.4f}) {mode}; rewritten={rewritten}"
        )
    return out


def repair_collapsed_face_landmarks(
    kps: np.ndarray,
    *,
    log_prefix: str | None = "[face-repair]",
    repair_nose: bool = True,
    repair_mouth: bool = True,
) -> np.ndarray:
    """Fix nose/mouth slots that collapsed onto eyes (pixel or crop [-1,1]).

    Accepts ``(N,3)`` detector keypoints or ``(N,4)`` sidecar / fitted refs.
    Iris snap runs only when iris rows exist (37-pt tensors).
    Pass ``log_prefix=None`` to stay quiet (caller can log once per pose).
    Set ``repair_mouth=False`` to trust detector/sidecar lips (debug / calibrate).
    """
    out = np.asarray(kps, dtype=np.float32)
    if out.ndim != 2 or out.shape[0] < 28 or out.shape[1] < 3:
        return out
    out = out.copy()
    changed = False
    if repair_nose and nose_needs_repair(out):
        out = _repair_nose_pixels(out, log_prefix=log_prefix)
        changed = True
    if repair_mouth and mouth_needs_repair(out):
        out = _repair_mouth_pixels(out, log_prefix=log_prefix)
        changed = True
    if changed:
        # Keep iris inside repaired eye boxes when present (37-pt only).
        for iris, eyes in ((28, (11, 12, 13)), (29, (17, 18, 19))):
            if iris >= out.shape[0]:
                continue
            if out.shape[1] > 3 and float(out[iris, 3]) < 0.5:
                continue
            if out.shape[1] > 2 and float(out[iris, 2]) < 0.05:
                continue
            mid = _mean_xy(out, eyes)
            if mid is None:
                continue
            d = float(np.hypot(out[iris, 0] - mid[0], out[iris, 1] - mid[1]))
            _, _, _, _, eye_dist = _face_anchors(out)
            if d > 0.35 * eye_dist:
                _set(out, iris, mid[0], mid[1], max(float(out[iris, 2]), 0.8))
    return out


def repair_face_detector_results(
    face_results: list[dict],
    *,
    log_prefix: str = "[label]",
) -> bool:
    """Mutate each face's ``keypoints`` in place when collapse is detected.

    Returns True if any face was repaired. Logs once for the whole batch.
    """
    repaired_parts: list[str] = []
    for pred in face_results:
        kps = np.asarray(pred.get("keypoints"), dtype=np.float32)
        if kps.ndim != 2 or kps.shape[0] < 28:
            continue
        did_nose = nose_needs_repair(kps)
        did_mouth = mouth_needs_repair(kps)
        if not did_nose and not did_mouth:
            continue
        pred["keypoints"] = repair_collapsed_face_landmarks(kps, log_prefix=None)
        if did_nose and "nose" not in repaired_parts:
            repaired_parts.append("nose")
        if did_mouth and "mouth" not in repaired_parts:
            repaired_parts.append("mouth")
    if repaired_parts:
        print(
            f"{log_prefix} repaired {'/'.join(repaired_parts)} landmarks "
            f"(collapsed onto eyes)"
        )
        return True
    return False
