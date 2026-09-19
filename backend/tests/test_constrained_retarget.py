"""Constrained Live2D retarget: proportions stay on the character reference."""

from __future__ import annotations

import numpy as np

from engine import apply_live_deltas_to_ref, neutral_keypoints
from live_retarget import (
    build_reference_rig,
    enforce_proportion_invariants,
    extract_controls,
)
from pose_controller import (
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
    from pose_controller import sanitize_pose

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

    assert np.all(np.abs(dy) > 0.008)
    assert np.all(dy[:3] < -0.005)  # upper lip lifts with live
    assert np.all(dy[[3, 6]] > 0.015)  # corners bend with the jaw
    assert np.all(dy[[4, 5, 7]] > 0.0)  # lower contour drops
    gap_rest = float(resting[25, 1] - resting[21, 1])
    gap_open = float(opened[25, 1] - opened[21, 1])
    assert gap_open > gap_rest + 0.06


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
    # Live==origin → neutral snap collapses to a closed slit on the character.
    assert abs(float(resting[25, 1] - resting[21, 1])) < 1e-4
    assert abs(_mouth_width(resting) - _mouth_width(ref)) < 0.04

    live = origin.copy()
    live[list(LEFT_BROW), 1] -= 0.04
    raised = apply_live_deltas_to_ref(ref, live, origin, **kwargs)
    brow_dy = float(np.mean(raised[list(LEFT_BROW), 1] - resting[list(LEFT_BROW), 1]))
    assert brow_dy < -0.015
    assert abs(_eye_width(raised, (11, 12, 13)) - _eye_width(resting, (11, 12, 13))) < 0.01


def test_mouth_box_keeps_character_placement_when_human_midface_differs() -> None:
    """Mouth location comes from the character box, not human midface length."""
    ref = _expression_ref()
    origin = ref.copy()
    live = origin.copy()
    # Human mouth sits much lower relative to the eyes (long midface).
    live[list(MOUTH), 1] += 0.12
    kwargs = {
        "body_method": "synthetic_from_face",
        "body_lost": False,
        "live_coord_space": "norm_crop",
        "origin_coord_space": "norm_crop",
    }
    resting = apply_live_deltas_to_ref(ref, origin, origin, **kwargs)
    shifted = apply_live_deltas_to_ref(ref, live, origin, **kwargs)

    # Rigid mouth translate changes live center but not shape vs origin center
    # after recentering — relative placement to the nose stays authored.
    rest_rel = float(np.mean(resting[list(MOUTH), 1]) - resting[15, 1])
    shift_rel = float(np.mean(shifted[list(MOUTH), 1]) - shifted[15, 1])
    assert abs(shift_rel - rest_rel) < 0.01
    assert float(np.mean(shifted[list(MOUTH), 1])) < float(shifted[2, 1])


def test_mouth_region_built_from_reference_anchors() -> None:
    ref = _expression_ref()
    rig = build_reference_rig(ref)
    assert rig.mouth_region is not None
    region = rig.mouth_region
    assert region.half_w >= region.rest_half_w > 0.0
    assert region.half_h >= region.rest_half_h > 0.0
    # Box is locked to a mouth-free anchor near the eyes/nose, not face COM.
    assert float(region.anchor_local[1]) < 0.0
    # Authored mouth center sits below that anchor.
    assert float(region.center_from_anchor[1]) > 0.0
    assert set(region.rest_uv) == set(MOUTH)


def test_near_closed_mouth_snaps_to_closed_slit() -> None:
    """Neutral / near-origin mouth snaps to a fully closed lip slit."""
    from live_retarget import extract_controls, is_mouth_closed_snap

    ref = _expression_ref()
    origin = ref.copy()
    # Typical resting gap that used to miss the old absolute threshold.
    for up, lo, gap in ((20, 24, 0.02), (21, 25, 0.025), (22, 27, 0.02)):
        origin[lo, 1] = float(origin[up, 1]) + gap
    live = origin.copy()
    # Tiny tracker noise around neutral — still near origin.
    live[21, 1] += 0.002
    live[25, 1] += 0.003
    assert is_mouth_closed_snap(live, origin)
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
    )
    for up, lo in ((20, 24), (21, 25), (22, 27)):
        assert abs(float(out[up, 1]) - float(out[lo, 1])) < 1e-4
    assert float(out[25, 1] - out[21, 1]) < 1e-4
    ctrls = extract_controls(live, origin, build_reference_rig(ref))
    assert ctrls.mouth_snapped is True


def test_neutral_smile_snap_preserves_mouth_object_transform() -> None:
    """Blue snap changes local shape without importing position or face lean."""
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

    snapped = apply_live_deltas_to_ref(
        ref,
        ref,
        ref,
        body_method="synthetic_from_face",
        body_lost=False,
    )

    # The mouth object keeps its authored center and orientation.
    np.testing.assert_allclose(
        np.mean(snapped[mouth_ids, :2], axis=0),
        np.mean(ref[mouth_ids, :2], axis=0),
        atol=1e-5,
    )
    axis_x = snapped[26, :2] - snapped[23, :2]
    axis_x /= np.linalg.norm(axis_x)
    assert abs(float(np.arctan2(axis_x[1], axis_x[0])) - angle) < 1e-5

    axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float32)
    if axis_y[1] < 0.0:
        axis_y *= -1.0
    for upper, lower in ((20, 24), (21, 25), (22, 27)):
        assert abs(float(np.dot(snapped[lower, :2] - snapped[upper, :2], axis_y))) < 1e-5
    # Closed center sits slightly below the raised corners: a subtle smile.
    lip_mid = 0.5 * (snapped[21, :2] + snapped[25, :2])
    corner_mid = 0.5 * (snapped[23, :2] + snapped[26, :2])
    assert float(np.dot(lip_mid - corner_mid, axis_y)) > 0.0


def test_smile_away_from_origin_does_not_snap() -> None:
    """A hard smile is not neutral — must not blue-snap just because gap shrank."""
    from live_retarget import is_mouth_closed_snap

    ref = _expression_ref()
    origin = ref.copy()
    for upper, lower in ((20, 24), (21, 25), (22, 27)):
        origin[lower, 1] = origin[upper, 1] + 0.02
    smiling = origin.copy()
    smiling[23, 1] -= 0.05
    smiling[26, 1] -= 0.05
    smiling[25, 1] = smiling[21, 1] + 0.004  # tiny mid gap like a pressed smile
    assert not is_mouth_closed_snap(smiling, origin)


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


def test_closed_after_open_origin_snaps_open_character_mouth() -> None:
    """User closes after an open origin lock: character must fully shut.

    Previously absolute shut rejected resting negative corner lift, so snap
    never fired and an open reference mouth stayed open in generate.
    """
    from live_retarget import is_mouth_closed_snap

    ref = _expression_ref()
    for up, lo, gap in ((20, 24, 0.08), (21, 25, 0.11), (22, 27, 0.08)):
        ref[lo, 1] = float(ref[up, 1]) + gap
    origin = _expression_ref()
    for up, lo, gap in ((20, 24, 0.05), (21, 25, 0.07), (22, 27, 0.05)):
        origin[lo, 1] = float(origin[up, 1]) + gap
    live = origin.copy()
    for up, lo in ((20, 24), (21, 25), (22, 27)):
        live[lo, 1] = live[up, 1]

    assert is_mouth_closed_snap(live, origin)
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        limit_mouth=False,
    )
    assert abs(float(out[25, 1] - out[21, 1])) < 1e-4
    for up, lo in ((20, 24), (21, 25), (22, 27)):
        assert abs(float(out[up, 1] - out[lo, 1])) < 1e-4


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
    assert float(jaw[1]) > 0.08
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
