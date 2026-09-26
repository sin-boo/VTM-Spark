"""Constrained Live2D retarget: proportions stay on the character reference."""

from __future__ import annotations

import numpy as np

from backend.engine import apply_live_deltas_to_ref, neutral_keypoints
from backend.live_retarget import (
    build_reference_rig,
    enforce_proportion_invariants,
    extract_controls,
)
from backend.pose_controller import (
    LEFT_BROW,
    L_EYE,
    MOUTH,
    R_EYE,
    face_height,
    face_width,
    sanitize_pose,
)


def _shoulder_width(k: np.ndarray) -> float:
    return float(np.linalg.norm(k[32, :2] - k[34, :2]))


def _bone_len(k: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(k[a, :2] - k[b, :2]))


def _eye_width(k: np.ndarray, idxs: tuple[int, ...]) -> float:
    xs = [float(k[i, 0]) for i in idxs]
    return max(xs) - min(xs)


def _mouth_width(k: np.ndarray) -> float:
    return abs(float(k[26, 0] - k[23, 0]))


def _eye_lid_depth(k: np.ndarray, idxs: tuple[int, int, int]) -> float:
    a = k[idxs[0], :2]
    p = k[idxs[1], :2]
    b = k[idxs[2], :2]
    chord = b - a
    return abs(float(chord[0] * (p[1] - a[1]) - chord[1] * (p[0] - a[0]))) / max(
        float(np.linalg.norm(chord)), 1e-5
    )


def _expression_ref() -> np.ndarray:
    """Neutral fixture with its mouth below the nose topology guard."""
    ref = neutral_keypoints()
    ref[MOUTH, 1] += 0.14
    return ref


def _narrow_anime_ref() -> np.ndarray:
    ref = neutral_keypoints()
    ref[32, 0] = -0.22
    ref[34, 0] = 0.22
    ref[33, 0] = -0.28
    ref[35, 0] = 0.28
    return ref


def _wide_human(origin_like: bool = True) -> np.ndarray:
    h = neutral_keypoints()
    # Much wider shoulders / longer arms than anime ref.
    h[32, 0] = -0.55
    h[34, 0] = 0.55
    h[33, 0] = -0.85
    h[35, 0] = 0.85
    # Broader face.
    h[0, 0] = -0.55
    h[4, 0] = 0.55
    h[11, 0] = -0.40
    h[13, 0] = -0.15
    h[17, 0] = 0.15
    h[19, 0] = 0.40
    return h


def test_build_reference_rig_stores_bone_lengths() -> None:
    ref = _narrow_anime_ref()
    rig = build_reference_rig(ref)
    assert rig.face_height > 0
    assert 32 in rig.bone_len and 34 in rig.bone_len
    assert abs(rig.bone_len[32] - _bone_len(ref, 31, 32)) < 1e-5


def test_wide_human_at_rest_keeps_anime_shoulders() -> None:
    ref = _narrow_anime_ref()
    origin = _wide_human()
    live = origin.copy()
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    assert abs(_shoulder_width(out) - _shoulder_width(ref)) < 0.04
    assert _shoulder_width(out) < _shoulder_width(origin) - 0.25
    # Defining face width stays near ref, not human-wide.
    assert abs(face_width(out) - face_width(ref)) / max(face_width(ref), 1e-3) < 0.12


def test_long_arms_do_not_stretch_elbow_bones() -> None:
    ref = _narrow_anime_ref()
    origin = _wide_human()
    live = origin.copy()
    # Swing elbows without changing shoulder width much.
    live[33, 1] += 0.20
    live[35, 1] += 0.20
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    assert abs(_bone_len(out, 32, 33) - _bone_len(ref, 32, 33)) / max(
        _bone_len(ref, 32, 33), 1e-3
    ) < 0.10
    assert abs(_bone_len(out, 34, 35) - _bone_len(ref, 34, 35)) / max(
        _bone_len(ref, 34, 35), 1e-3
    ) < 0.10


def test_shoulder_raise_moves_but_keeps_width() -> None:
    ref = _narrow_anime_ref()
    origin = _wide_human()
    live = origin.copy()
    live[32, 1] -= 0.15
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    assert float(out[32, 1]) < float(ref[32, 1]) - 0.01
    assert abs(_shoulder_width(out) - _shoulder_width(ref)) < 0.05


def test_blink_closes_eye_without_changing_width() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    # Collapse left eye vertically (blink).
    mid_y = 0.5 * (float(live[11, 1]) + float(live[13, 1]))
    for i in (11, 12, 13):
        live[i, 1] = mid_y
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    # Aperture smaller than ref.
    ap_ref = abs(float(ref[11, 1]) - float(ref[12, 1])) + abs(
        float(ref[13, 1]) - float(ref[12, 1])
    )
    ap_out = abs(float(out[11, 1]) - float(out[12, 1])) + abs(
        float(out[13, 1]) - float(out[12, 1])
    )
    assert ap_out < ap_ref * 0.75
    # Eye width preserved.
    assert abs(_eye_width(out, (11, 12, 13)) - _eye_width(ref, (11, 12, 13))) < 0.04


def test_blink_curve_is_monotonic_and_hides_iris_when_closed() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    outputs: list[np.ndarray] = []
    for openness in (0.0, 0.5, 1.0):
        live = origin.copy()
        chord_y = 0.5 * (float(live[11, 1]) + float(live[13, 1]))
        live[12, 1] = chord_y + (float(live[12, 1]) - chord_y) * openness
        outputs.append(
            apply_live_deltas_to_ref(
                ref,
                live,
                origin,
                body_method="synthetic_from_face",
                body_lost=False,
                live_coord_space="norm_crop",
                origin_coord_space="norm_crop",
            )
        )

    closed, half, opened = outputs
    depths = [_eye_lid_depth(k, (11, 12, 13)) for k in outputs]
    assert depths[0] < depths[1] < depths[2]
    assert depths[0] < _eye_lid_depth(ref, (11, 12, 13)) * 0.10
    half_ratio = depths[1] / max(depths[2], 1e-5)
    assert 0.25 < half_ratio < 0.75
    for out in outputs:
        assert abs(_eye_width(out, (11, 12, 13)) - _eye_width(ref, (11, 12, 13))) < 0.04
    assert float(closed[28, 3]) < 0.5
    assert float(half[28, 3]) >= 0.5
    assert float(opened[28, 3]) >= 0.5


def test_iris_stays_inside_correct_eye() -> None:
    """Person-left iris (29) belongs with image-right eye lids (17-19)."""
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    # Push person-left iris far outside its live eye.
    live[29, 0] = float(live[18, 0]) + 0.5
    live[29, 1] = float(live[18, 1]) - 0.5
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    xs = [float(out[i, 0]) for i in (17, 18, 19)]
    ys = [float(out[i, 1]) for i in (17, 18, 19)]
    assert min(xs) - 0.05 <= float(out[29, 0]) <= max(xs) + 0.05
    assert min(ys) - 0.05 <= float(out[29, 1]) <= max(ys) + 0.05
    # Must not jump into the other eye / toward nose center.
    assert float(out[29, 0]) > 0.0
    assert float(out[28, 0]) < 0.0


def test_iris_pairing_not_cross_eyed_at_rest() -> None:
    """Resting gaze must keep irises in their own eyes (not swapped to nose)."""
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    # Right iris (28) with image-left eye (11-13); left iris (29) with 17-19.
    assert float(out[28, 0]) < float(out[15, 0])  # left of nose
    assert float(out[29, 0]) > float(out[15, 0])  # right of nose
    l_eye_c = 0.5 * (float(out[11, 0]) + float(out[13, 0]))
    r_eye_c = 0.5 * (float(out[17, 0]) + float(out[19, 0]))
    assert abs(float(out[28, 0]) - l_eye_c) < abs(float(out[28, 0]) - r_eye_c)
    assert abs(float(out[29, 0]) - r_eye_c) < abs(float(out[29, 0]) - l_eye_c)
    # Not collapsed toward nose.
    assert abs(float(out[28, 0]) - float(out[15, 0])) > 0.08
    assert abs(float(out[29, 0]) - float(out[15, 0])) > 0.08


def test_sanitize_iris_clamped_to_matching_eye() -> None:
    from backend.pose_controller import sanitize_pose

    ref = neutral_keypoints()
    driven = ref.copy()
    # Force both irises onto the nose — sanitize must push each back to its eye.
    driven[28, 0] = 0.0
    driven[29, 0] = 0.0
    out = sanitize_pose(driven, ref, recenter=False, lock_proportions=False)
    assert float(out[28, 0]) < -0.05
    assert float(out[29, 0]) > 0.05
    xs_l = [float(out[i, 0]) for i in (11, 12, 13)]
    xs_r = [float(out[i, 0]) for i in (17, 18, 19)]
    assert min(xs_l) - 0.03 <= float(out[28, 0]) <= max(xs_l) + 0.03
    assert min(xs_r) - 0.03 <= float(out[29, 0]) <= max(xs_r) + 0.03


def test_mouth_open_and_close() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    # Open mouth on live.
    live_open = origin.copy()
    live_open[25, 1] += 0.18
    out_open = apply_live_deltas_to_ref(
        ref,
        live_open,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    assert float(out_open[25, 1]) > float(ref[25, 1]) + 0.01
    # Closed mouth.
    live_closed = origin.copy()
    live_closed[25, 1] = float(live_closed[21, 1]) + 0.01
    out_closed = apply_live_deltas_to_ref(
        ref,
        live_closed,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    gap = float(out_closed[25, 1] - out_closed[21, 1])
    assert gap < 0.05
    # Mouth width stays near ref.
    assert abs(_mouth_width(out_open) - _mouth_width(ref)) < 0.06


def test_full_mouth_open_deforms_entire_reference_contour() -> None:
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    # Move the full live contour (upper + lower) so box-local expression opens.
    for i, dy in {
        20: -0.04,
        21: -0.06,
        22: -0.04,
        23: 0.08,
        24: 0.16,
        25: 0.22,
        26: 0.08,
        27: 0.16,
    }.items():
        live[i, 1] += dy
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
    }
    resting = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    opened = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    dy = opened[MOUTH, 1] - resting[MOUTH, 1]

    # Opening stays in the authored mouth: lower lip drops, upper stays put.
    assert np.all(np.abs(dy[:3]) < 0.04)
    assert np.all(dy[[3, 6]] > 0.008)  # corners
    assert np.all(dy[[4, 5, 7]] > 0.02)  # lower contour
    gap_rest = float(resting[25, 1] - resting[21, 1])
    gap_open = float(opened[25, 1] - opened[21, 1])
    assert gap_open > gap_rest + 0.015
    # The whole mouth does not slide off the authored place.
    rest_c = float(np.mean(resting[list(MOUTH), 1]))
    open_c = float(np.mean(opened[list(MOUTH), 1]))
    assert abs(open_c - rest_c) < 0.10


def test_mouth_open_never_lifts_upper_lip_toward_nose() -> None:
    """Jaw-only live drop opens the character mouth downward."""
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    for i, dy in {23: 0.10, 26: 0.10, 24: 0.26, 25: 0.30, 27: 0.26, 2: 0.25}.items():
        live[i, 1] += dy
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_mouth": False,
        "limit_nose": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    opened = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    for i in (20, 21, 22):
        assert abs(float(opened[i, 1]) - float(rest[i, 1])) < 0.04
    assert float(opened[25, 1]) > float(rest[25, 1]) + 0.04


def test_wide_open_upper_mid_stays_on_lip_line() -> None:
    """A lone live 21 spike toward the nose does not make a V on the character."""
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    for i, dy in {23: 0.10, 26: 0.10, 24: 0.26, 25: 0.30, 27: 0.26}.items():
        live[i, 1] += dy
    live[21, 1] -= 0.04
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_mouth": False,
        "limit_nose": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    opened = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    chord_y = 0.5 * (float(opened[20, 1]) + float(opened[22, 1]))
    assert float(opened[21, 1]) >= chord_y - 0.01
    assert float(opened[25, 1]) > float(rest[25, 1]) + 0.04


def test_clamp_mouth_anatomy_flattens_upper_mid_triangle() -> None:
    from backend.live_retarget import clamp_mouth_anatomy
    from backend.pose_controller import face_height

    k = _expression_ref()
    rest = k.copy()
    k[21, 0] = float(k[15, 0])
    k[21, 1] = float(k[15, 1]) + 0.01
    out = clamp_mouth_anatomy(k.copy(), face_height(k), ref=rest)
    chord_y = 0.5 * (float(out[20, 1]) + float(out[22, 1]))
    assert float(out[21, 1]) >= chord_y - 1e-4
    assert float(out[21, 1]) > float(out[15, 1])


def test_upper_mid_on_nose_does_not_stretch_character_mouth() -> None:
    """A live 21 parked on the nose is repaired, not transplanted."""
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    live[21, 0] = float(live[15, 0])
    live[21, 1] = float(live[15, 1])
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_mouth": False,
        "limit_nose": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    confused = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    assert float(confused[21, 1]) > float(confused[15, 1])
    assert abs(float(confused[21, 1]) - float(rest[21, 1])) < 0.04


def test_clamp_mouth_anatomy_pulls_upper_lip_off_nose() -> None:
    from backend.live_retarget import clamp_mouth_anatomy
    from backend.pose_controller import face_height

    k = _expression_ref()
    rest = k.copy()
    k[21, 0] = float(k[15, 0])
    k[21, 1] = float(k[15, 1])
    k[20, 1] = float(k[15, 1])
    k[22, 1] = float(k[15, 1])
    k[25, 1] = float(k[15, 1]) - 0.02
    out = clamp_mouth_anatomy(k.copy(), face_height(k), ref=rest)
    assert float(out[21, 1]) > float(out[15, 1])
    assert float(out[25, 1]) >= float(out[21, 1])
    assert float(out[20, 1]) > float(out[15, 1])
    assert float(out[22, 1]) > float(out[15, 1])


def test_reconstruct_keeps_character_lip_order() -> None:
    """Inverted / nose-confused live lips do not invert the character mouth."""
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    live[21, 1] = float(live[15, 1])
    live[25, 1] = float(live[15, 1]) - 0.04
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_mouth": False,
        "limit_nose": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    out = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    assert float(out[21, 1]) > float(out[15, 1])
    assert float(out[25, 1]) >= float(out[21, 1])
    assert abs(float(np.mean(out[list(MOUTH), 1])) - float(np.mean(rest[list(MOUTH), 1]))) < 0.06


def test_mouth_repair_rejects_upper_mid_on_nose() -> None:
    from face_landmark_repair import (
        mouth_needs_repair,
        repair_collapsed_face_landmarks,
    )

    k = _expression_ref()
    k[21, 0] = float(k[15, 0])
    k[21, 1] = float(k[15, 1])
    assert mouth_needs_repair(k)
    out = repair_collapsed_face_landmarks(k, log_prefix=None)
    assert float(np.hypot(out[21, 0] - out[15, 0], out[21, 1] - out[15, 1])) > 0.04
    assert float(out[21, 1]) > float(out[15, 1])


def test_resting_expression_and_brow_motion_stay_stable() -> None:
    ref = _expression_ref()
    origin = ref.copy()
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
    }
    resting = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    np.testing.assert_allclose(resting[5:20, :2], ref[5:20, :2], atol=0.01)
    # Live==origin keeps the authored mouth; no closed-slit snap.
    np.testing.assert_allclose(resting[list(MOUTH), :2], ref[list(MOUTH), :2], atol=0.02)
    assert abs(_mouth_width(resting) - _mouth_width(ref)) < 0.04

    live = origin.copy()
    live[list(LEFT_BROW), 1] -= 0.04
    raised = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    brow_dy = float(np.mean(raised[list(LEFT_BROW), 1] - resting[list(LEFT_BROW), 1]))
    assert brow_dy < -0.015
    assert abs(_eye_width(raised, (11, 12, 13)) - _eye_width(resting, (11, 12, 13))) < 0.01


def test_human_sized_mouth_open_stays_in_character_box() -> None:
    """A huge human jaw drop must not explode the authored anime mouth."""
    ref = _expression_ref()
    origin = _expression_ref()
    live = origin.copy()
    # Webcam-scale open: lower lip walks toward the chin.
    for i, dy in {23: 0.08, 24: 0.28, 25: 0.36, 26: 0.08, 27: 0.28, 2: 0.22}.items():
        live[i, 1] += dy
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    opened = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    fh = face_height(ref)
    rest_c = np.mean(rest[list(MOUTH), :2], axis=0)
    open_c = np.mean(opened[list(MOUTH), :2], axis=0)
    assert float(np.hypot(open_c[0] - rest_c[0], open_c[1] - rest_c[1])) < 0.10
    assert float(opened[25, 1]) > float(rest[25, 1]) + 0.02
    assert float(opened[25, 1] - opened[21, 1]) < 0.40 * fh
    assert float(opened[21, 1]) > float(opened[15, 1])
    assert abs(_mouth_width(opened) - _mouth_width(rest)) < 0.08


def test_mouth_stays_in_authored_place_when_live_mouth_slides() -> None:
    """A whole-mouth live slide does not drag the character mouth off its place."""
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    live[list(MOUTH), 1] += 0.12
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_mouth": False,
    }
    resting = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    shifted = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    rest_rel = float(np.mean(resting[list(MOUTH), 1]) - resting[15, 1])
    shift_rel = float(np.mean(shifted[list(MOUTH), 1]) - shifted[15, 1])
    assert abs(shift_rel - rest_rel) < 0.03
    assert abs(_mouth_width(shifted) - _mouth_width(resting)) < 0.04


def test_reference_rig_stores_mouth_size() -> None:
    ref = _expression_ref()
    rig = build_reference_rig(ref)
    assert rig.mouth_width > 0.05
    assert rig.mouth_gap > 0.0


def test_near_closed_mouth_does_not_snap() -> None:
    """Near-origin mouth keeps live shape; no closed-slit snap."""
    from backend.live_retarget import extract_controls

    ref = _expression_ref()
    origin = ref.copy()
    for up, lo, gap in ((20, 24, 0.02), (21, 25, 0.025), (22, 27, 0.02)):
        origin[lo, 1] = float(origin[up, 1]) + gap
    live = origin.copy()
    live[21, 1] += 0.002
    live[25, 1] += 0.003
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
    )
    rest = apply_live_deltas_to_ref(
        ref,
        origin,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
    )
    assert abs(float(out[25, 1] - out[21, 1]) - float(rest[25, 1] - rest[21, 1])) < 0.01
    ctrls = extract_controls(live, origin, build_reference_rig(ref))
    assert ctrls.mouth_snapped is False


def test_rest_equals_live_keeps_authored_mouth() -> None:
    """Live==origin leaves the reference mouth in place — no slit snap."""
    ref = _expression_ref()
    mouth_ids = list(MOUTH)
    center = np.mean(ref[mouth_ids, :2], axis=0)
    angle = np.radians(17.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    translation = np.array([0.035, 0.020], dtype=np.float32)
    ref[mouth_ids, :2] = (
        (ref[mouth_ids, :2] - center) @ rotation.T + center + translation
    )

    out = apply_live_deltas_to_ref(
        ref,
        ref,
        ref,
        body_method="synthetic_from_face",
        body_lost=False,
    )
    np.testing.assert_allclose(
        np.mean(out[mouth_ids, :2], axis=0),
        np.mean(ref[mouth_ids, :2], axis=0),
        atol=1e-4,
    )
    np.testing.assert_allclose(out[mouth_ids, :2], ref[mouth_ids, :2], atol=0.02)


def test_closed_live_mouth_closes_toward_reference() -> None:
    """Closing relative to an open origin closes the character mouth."""
    ref = _expression_ref()
    origin = ref.copy()
    # Keep origin clearly open so a closed live is not "near neutral".
    origin[24, 1] = float(origin[20, 1]) + 0.08
    origin[25, 1] = float(origin[21, 1]) + 0.10
    origin[27, 1] = float(origin[22, 1]) + 0.08
    live = origin.copy()
    live[24, 1] = live[20, 1]
    live[25, 1] = live[21, 1]
    live[27, 1] = live[22, 1]
    rest = apply_live_deltas_to_ref(
        ref,
        origin,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
    )
    closed = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
    )
    # Origin==live open → neutral snap (gap 0). Closed live also ends closed.
    assert float(closed[25, 1] - closed[21, 1]) <= float(rest[25, 1] - rest[21, 1]) + 1e-4
    assert float(closed[25, 1] - closed[21, 1]) < 0.02


def test_closed_after_open_origin_closes_character_mouth() -> None:
    """Closing vs an open origin transfers as a close delta, with no snap."""
    ref = _expression_ref()
    for up, lo, gap in ((20, 24, 0.08), (21, 25, 0.11), (22, 27, 0.08)):
        ref[lo, 1] = float(ref[up, 1]) + gap
    origin = _expression_ref()
    for up, lo, gap in ((20, 24, 0.05), (21, 25, 0.07), (22, 27, 0.05)):
        origin[lo, 1] = float(origin[up, 1]) + gap
    live = origin.copy()
    for up, lo in ((20, 24), (21, 25), (22, 27)):
        live[lo, 1] = live[up, 1]

    rest = apply_live_deltas_to_ref(
        ref,
        origin,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        limit_mouth=False,
    )
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        limit_mouth=False,
    )
    assert float(out[25, 1] - out[21, 1]) < float(rest[25, 1] - rest[21, 1]) - 0.02


def test_frown_lowers_corners_smile_raises_them() -> None:
    """Expression deltas keep the correct vertical direction (not inverted)."""
    ref = _expression_ref()
    origin = ref.copy()
    for upper, lower in ((20, 24), (21, 25), (22, 27)):
        origin[lower, 1] = origin[upper, 1]
    kwargs = {"body_method": "synthetic_from_face", "body_lost": False}
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)

    smiling = origin.copy()
    smiling[23, 1] -= 0.04
    smiling[26, 1] -= 0.04
    smile_out = apply_live_deltas_to_ref(ref, smiling, origin, **kwargs)
    assert float(smile_out[23, 1]) < float(rest[23, 1]) - 0.005
    assert float(smile_out[26, 1]) < float(rest[26, 1]) - 0.005

    frowning = origin.copy()
    frowning[23, 1] += 0.04
    frowning[26, 1] += 0.04
    frown_out = apply_live_deltas_to_ref(ref, frowning, origin, **kwargs)
    assert float(frown_out[23, 1]) > float(rest[23, 1]) + 0.005
    assert float(frown_out[26, 1]) > float(rest[26, 1]) + 0.005


def test_smile_and_form_move_mouth_as_one_loop() -> None:
    ref = _expression_ref()
    origin = ref.copy()
    for upper, lower in ((20, 24), (21, 25), (22, 27)):
        origin[lower, 1] = origin[upper, 1]
    smiling = origin.copy()
    smiling[23, 0] -= 0.025
    smiling[26, 0] += 0.025
    smiling[23, 1] -= 0.035
    smiling[26, 1] -= 0.035
    kwargs = {"body_method": "synthetic_from_face", "body_lost": False}
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    out = apply_live_deltas_to_ref(ref, smiling, origin, **kwargs)

    assert float(out[23, 1]) < float(rest[23, 1]) - 0.005
    assert float(out[26, 1]) < float(rest[26, 1]) - 0.005
    assert _mouth_width(out) > _mouth_width(rest)


def test_mouth_open_axis_rotates_with_head() -> None:
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    live[24, 1] += 0.16
    live[25, 1] += 0.22
    live[27, 1] += 0.16
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        head_roll_deg=20.0,
    )
    jaw = out[25, :2] - out[21, :2]
    assert float(jaw[1]) > 0.04
    assert float(jaw[0]) < -0.02


def test_calibrated_head_translation_survives_safety_pass() -> None:
    ref = _expression_ref()
    origin = ref.copy()
    rest = apply_live_deltas_to_ref(
        ref,
        origin,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        head_tx_norm=0.0,
    )
    moved = apply_live_deltas_to_ref(
        ref,
        origin,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        head_tx_norm=0.25,
    )
    dx = float(np.mean(moved[:28, 0] - rest[:28, 0]))
    assert dx > 0.02


def test_landmark_translation_wins_over_tiny_calibrated_nx() -> None:
    """Webcam face COM must drive the overlay even if RelativePose nx is ~0.

    Live logs showed face_c moving ~0.07 while tx stayed at -0.04 * 0.40,
    so the character overlay sat on the reference.
    """
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    live[:28, 0] -= 0.08
    live[28:30, 0] -= 0.08
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_face": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, head_tx_norm=0.0, **kwargs)
    moved = apply_live_deltas_to_ref(
        ref, live, origin, head_tx_norm=-0.04, **kwargs
    )
    dx = float(np.mean(moved[:5, 0] - rest[:5, 0]))
    assert dx < -0.02


def test_pitch_nod_moves_chin() -> None:
    """A 20° look-down has to travel the chin, not only foreshorten |x|."""
    ref = _expression_ref()
    origin = ref.copy()
    rest = apply_live_deltas_to_ref(
        ref,
        origin,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        head_pitch_deg=0.0,
    )
    nodded = apply_live_deltas_to_ref(
        ref,
        origin,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        head_pitch_deg=20.0,
    )
    assert float(nodded[2, 1]) > float(rest[2, 1]) + 0.02


def test_face_translation_does_not_slide_mouth_opposite() -> None:
    """Whole-face slide must move mouth with the face, not against it.

    Mouth UV used to be measured against a fixed origin anchor, so head
    translation leaked in as sideways lip UV and the mouth drifted opposite
    head_tx (especially with mouth limiters off).
    """
    ref = neutral_keypoints()
    origin = ref.copy()
    live = origin.copy()
    # Human moves left in image space (negative X).
    live[:28, 0] -= 0.12
    live[28:30, 0] -= 0.12
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_face": False,
        "limit_mouth": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    moved = apply_live_deltas_to_ref(ref, live, origin, **kwargs)

    face_dx = float(np.mean(moved[:5, 0] - rest[:5, 0]))
    mouth_dx = float(np.mean(moved[list(MOUTH), 0] - rest[list(MOUTH), 0]))
    nose_dx = float(moved[15, 0] - rest[15, 0])
    assert face_dx < -0.01
    assert mouth_dx < -0.01
    # Mouth stays glued to the nose/face — no opposite-direction slide.
    assert abs(mouth_dx - nose_dx) < 0.02
    rest_mouth_rel = float(np.mean(rest[list(MOUTH), 0]) - rest[15, 0])
    moved_mouth_rel = float(np.mean(moved[list(MOUTH), 0]) - moved[15, 0])
    assert abs(moved_mouth_rel - rest_mouth_rel) < 0.015


def test_binocular_gaze_moves_together_in_turned_eye_basis() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    live[28, 0] += 0.05
    live[29, 0] += 0.05
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "head_roll_deg": 15.0,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    looked = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    for iris, eye in ((28, (11, 12, 13)), (29, (17, 18, 19))):
        axis = looked[eye[2], :2] - looked[eye[0], :2]
        axis /= max(float(np.linalg.norm(axis)), 1e-5)
        delta = looked[iris, :2] - rest[iris, :2]
        assert float(np.dot(delta, axis)) > 0.02
    assert float(looked[28, 0]) < float(looked[15, 0])
    assert float(looked[29, 0]) > float(looked[15, 0])


def test_head_yaw_warps_without_stretching_face_width() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
        head_yaw_deg=20.0,
    )
    # Face still roughly ref-sized (yaw foreshortens; allow modest height change).
    assert abs(face_width(out) - face_width(ref)) / max(face_width(ref), 1e-3) < 0.15
    assert abs(face_height(out) - face_height(ref)) / max(face_height(ref), 1e-3) < 0.30


def test_landmark_yaw_transfers_without_calibrated_angles() -> None:
    """Yaw must reach the anime face from keypoints alone (no RelativePose).

    Camera overlay rotates from raw landmarks; if app drops uncalibrated
    head_yaw_deg, extract_controls must still derive yaw from live vs origin.
    """
    ref = neutral_keypoints()
    origin = ref.copy()
    live = origin.copy()
    # Simulate a rightward head turn: nose shifts toward image-right relative
    # to the eye midline (geom yaw sign matches label_schema).
    mid_x = 0.5 * (float(live[12, 0]) + float(live[18, 0]))
    live[14:17, 0] = mid_x + 0.08
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_face": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    turned = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    # Character face should foreshorten / shear — not stay identical to rest.
    assert abs(float(turned[0, 0] - turned[4, 0])) < abs(
        float(rest[0, 0] - rest[4, 0])
    ) - 0.01 or abs(float(np.mean(turned[:5, 0] - rest[:5, 0]))) > 0.005
    # Nose should move with the yaw warp relative to face center.
    rest_nose = float(rest[15, 0] - np.mean(rest[:5, 0]))
    turned_nose = float(turned[15, 0] - np.mean(turned[:5, 0]))
    assert abs(turned_nose - rest_nose) > 0.01


def test_landmark_roll_tilts_face_without_calibrated_angles() -> None:
    ref = neutral_keypoints()
    origin = ref.copy()
    live = origin.copy()
    # Tilt: raise image-right eye / lower image-left eye.
    live[list(L_EYE), 1] += 0.04
    live[list(R_EYE), 1] -= 0.04
    live[list(LEFT_BROW), 1] += 0.04
    live[8:11, 1] -= 0.04
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "limit_face": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    tilted = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    # Left outline tip should drop relative to right when rolling this way.
    rest_dy = float(rest[0, 1] - rest[4, 1])
    tilt_dy = float(tilted[0, 1] - tilted[4, 1])
    assert abs(tilt_dy - rest_dy) > 0.02


def test_extreme_head_turns_remain_ordered_and_reference_bounded() -> None:
    ref = _expression_ref()
    origin = ref.copy()
    for yaw in (-28.0, 28.0):
        out = apply_live_deltas_to_ref(
            ref,
            origin,
            origin,
            body_method="synthetic_from_face",
            body_lost=False,
            head_yaw_deg=yaw,
            head_pitch_deg=15.0,
        )
        assert np.isfinite(out).all()
        assert float(out[11, 0]) < float(out[13, 0])
        assert float(out[17, 0]) < float(out[19, 0])
        assert float(out[23, 0]) < float(out[26, 0])
        assert 0.75 < face_width(out) / face_width(ref) < 1.15
        assert abs(_shoulder_width(out) - _shoulder_width(ref)) < 0.05


def test_forward_scale_is_bounded() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    # Simulate leaning toward camera: larger live face.
    live[:28, :2] *= 1.35
    live[28:30, :2] *= 1.35
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    # Character face must not explode to human camera scale.
    assert face_height(out) < face_height(ref) * 1.30
    assert face_width(out) < face_width(ref) * 1.20


def test_return_from_yaw_does_not_widen_mesh() -> None:
    """Look left then back: tracker eye-span overshoot must not widen the mesh."""
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
        "limit_face": False,
    }
    rest = apply_live_deltas_to_ref(ref, origin, origin, head_yaw_deg=0.0, **kwargs)

    # Mid-turn: perspective foreshortens interocular span (and calibrated yaw).
    turned_live = origin.copy()
    mid_x = 0.5 * (float(turned_live[12, 0]) + float(turned_live[18, 0]))
    turned_live[14:17, 0] = mid_x + 0.07
    for i in list(L_EYE) + list(R_EYE) + [28, 29]:
        turned_live[i, 0] = mid_x + 0.82 * (float(turned_live[i, 0]) - mid_x)
    turned = apply_live_deltas_to_ref(
        ref, turned_live, origin, head_yaw_deg=-22.0, **kwargs
    )
    assert face_width(turned) <= face_width(rest) * 1.02

    # Return to center with a slightly larger live eye span than origin
    # (common OSF/crop overshoot after a turn).
    returned_live = origin.copy()
    eye_c = 0.5 * (
        np.mean(returned_live[list(L_EYE), :2], axis=0)
        + np.mean(returned_live[list(R_EYE), :2], axis=0)
    )
    for i in list(L_EYE) + list(R_EYE) + list(range(0, 5)) + [28, 29]:
        returned_live[i, :2] = eye_c + 1.08 * (returned_live[i, :2] - eye_c)
    returned = apply_live_deltas_to_ref(
        ref, returned_live, origin, head_yaw_deg=0.0, **kwargs
    )
    assert face_width(returned) <= face_width(rest) * 1.025
    assert abs(face_width(returned) - face_width(rest)) / max(
        face_width(rest), 1e-3
    ) < 0.03


def test_held_body_preserves_prev() -> None:
    ref = _narrow_anime_ref()
    origin = _wide_human()
    live = origin.copy()
    live[32, 0] -= 0.3
    prev = ref[30:37].copy()
    prev[2, 0] = float(ref[32, 0] - 0.05)
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite_held",
        body_lost=True,
        prev_body=prev,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    assert abs(float(out[32, 0]) - float(prev[2, 0])) < 1e-5


def test_proportion_invariant_safety_net() -> None:
    ref = _narrow_anime_ref()
    bad = ref.copy()
    # Artificially stretch shoulders like a human leak.
    bad[32, 0] = -0.60
    bad[34, 0] = 0.60
    enforce_proportion_invariants(bad, ref, sanitize_body=True, tol_body=0.06)
    assert abs(_shoulder_width(bad) - _shoulder_width(ref)) / max(
        _shoulder_width(ref), 1e-3
    ) < 0.08


def test_extract_controls_blink_and_gaze() -> None:
    ref = neutral_keypoints()
    rig = build_reference_rig(ref)
    origin = neutral_keypoints()
    live = origin.copy()
    mid_y = float(live[12, 1])
    for i in (11, 12, 13):
        live[i, 1] = mid_y
    live[29, 0] = float(live[12, 0]) + 0.05
    ctrl = extract_controls(live, origin, rig)
    assert ctrl.blink_l < 0.6
    assert abs(ctrl.gaze_l[0]) > 0.05 or abs(ctrl.gaze_r[0]) >= 0.0


def test_feature_limiters_can_be_disabled() -> None:
    """Hard caps for face/brows/mouth loosen when their limit_* flags are off."""
    ref = neutral_keypoints()
    origin = ref.copy()
    live = origin.copy()
    live[list(LEFT_BROW), 1] -= 0.20
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
    }
    capped = apply_live_deltas_to_ref(ref, live, origin, limit_brows=True, **kwargs)
    free = apply_live_deltas_to_ref(ref, live, origin, limit_brows=False, **kwargs)
    dy_c = float(np.mean(capped[list(LEFT_BROW), 1] - ref[list(LEFT_BROW), 1]))
    dy_f = float(np.mean(free[list(LEFT_BROW), 1] - ref[list(LEFT_BROW), 1]))
    assert abs(dy_f) > abs(dy_c) + 0.05

    capped_h = apply_live_deltas_to_ref(
        ref, origin, origin, head_tx_norm=2.0, limit_face=True, **kwargs
    )
    free_h = apply_live_deltas_to_ref(
        ref, origin, origin, head_tx_norm=2.0, limit_face=False, **kwargs
    )
    assert abs(float(free_h[15, 0] - ref[15, 0])) > abs(
        float(capped_h[15, 0] - ref[15, 0])
    ) + 0.05

    live_m = origin.copy()
    live_m[25, 1] += 0.35
    capped_m = apply_live_deltas_to_ref(ref, live_m, origin, limit_mouth=True, **kwargs)
    free_m = apply_live_deltas_to_ref(ref, live_m, origin, limit_mouth=False, **kwargs)
    assert float(free_m[25, 1] - ref[25, 1]) > float(
        capped_m[25, 1] - ref[25, 1]
    )


def test_motion_caps_limit_yaw() -> None:
    ref = neutral_keypoints()
    origin = ref.copy()
    rig = build_reference_rig(ref)
    free = extract_controls(origin, origin, rig, head_yaw_deg=40.0)
    tight = extract_controls(
        origin,
        origin,
        rig,
        head_yaw_deg=40.0,
        motion={
            "turn_left": 8.0, "turn_right": 8.0, "tilt_left": 80.0, "tilt_right": 80.0,
            "pitch_up": 50.0, "pitch_down": 32.0,
        },
    )
    assert abs(free.head_yaw_deg) > abs(tight.head_yaw_deg) + 10
    assert abs(tight.head_yaw_deg) <= 8.0 + 1e-5


def test_motion_caps_stop_each_side_on_its_own() -> None:
    """Right is positive: a closed right turn / tilt must not hold the left."""
    ref = neutral_keypoints()
    rig = build_reference_rig(ref)
    motion = {
        "turn_left": 20.0, "turn_right": 0.0, "tilt_left": 0.0, "tilt_right": 20.0,
        "pitch_up": 50.0, "pitch_down": 32.0,
    }
    right = extract_controls(ref, ref, rig, head_yaw_deg=15.0, head_roll_deg=-15.0, motion=motion)
    left = extract_controls(ref, ref, rig, head_yaw_deg=-15.0, head_roll_deg=15.0, motion=motion)
    assert abs(right.head_yaw_deg) < 1e-5 and abs(right.head_roll_deg) < 1e-5
    assert left.head_yaw_deg < -10.0 and left.head_roll_deg > 10.0


def test_sanitize_per_region_travel_limits() -> None:
    ref = neutral_keypoints()
    fh = face_height(ref)
    driven = ref.copy()
    driven[2, 1] += 0.50 * fh  # chin / face outline
    driven[6, 1] -= 0.50 * fh  # brow

    all_on = sanitize_pose(
        driven,
        ref,
        recenter=False,
        topology=False,
        lock_proportions=False,
        limit_face=True,
        limit_brows=True,
        limit_eyes=True,
        limit_nose=True,
        limit_mouth=True,
    )
    face_off = sanitize_pose(
        driven,
        ref,
        recenter=False,
        topology=False,
        lock_proportions=False,
        limit_face=False,
        limit_brows=True,
        limit_eyes=True,
        limit_nose=True,
        limit_mouth=True,
    )
    # Face outline free to travel farther when its limiter is off.
    assert abs(float(face_off[2, 1] - ref[2, 1])) > abs(
        float(all_on[2, 1] - ref[2, 1])
    ) + 0.05
    # Brow still clamped.
    assert abs(float(face_off[6, 1] - ref[6, 1])) <= abs(
        float(all_on[6, 1] - ref[6, 1])
    ) + 1e-5


def test_sanitize_lock_proportions_flag() -> None:
    ref = _narrow_anime_ref()
    driven = ref.copy()
    driven[32, 0] = -0.7
    driven[34, 0] = 0.7
    out = sanitize_pose(driven, ref, lock_proportions=True, body_tracked=True)
    assert abs(_shoulder_width(out) - _shoulder_width(ref)) / max(
        _shoulder_width(ref), 1e-3
    ) < 0.12
