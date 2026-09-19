"""Pure-numpy tests for sanitize / overlay coordinate helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.pose_controller import (
    ATTACH_LEN_MAX,
    BODY_NOSE,
    IRIS_L,
    IRIS_R,
    KEYPOINT_REFS,
    TRAVEL_BODY,
    TRAVEL_BODY_TRACKED,
    block_model_slots,
    draw_keypoint_mesh,
    enforce_head_body_attachment,
    face_height,
    normalized_to_pixels,
    overlay_visible_slots,
    pixels_to_normalized,
    ref_of,
    sanitize_pose,
    slot_of,
)


def _blank37() -> np.ndarray:
    k = np.zeros((37, 4), dtype=np.float32)
    # Face outline / brows / eyes / nose / mouth — simple upright layout.
    k[0] = [-0.4, 0.0, 1, 1]
    k[1] = [-0.25, 0.25, 1, 1]
    k[2] = [0.0, 0.45, 1, 1]  # chin
    k[3] = [0.25, 0.25, 1, 1]
    k[4] = [0.4, 0.0, 1, 1]
    for i, x in enumerate((-0.25, -0.15, -0.05)):
        k[5 + i] = [x, -0.35, 1, 1]  # left brow
    for i, x in enumerate((0.05, 0.15, 0.25)):
        k[8 + i] = [x, -0.35, 1, 1]  # right brow
    k[11] = [-0.28, -0.15, 1, 1]
    k[12] = [-0.18, -0.18, 1, 1]
    k[13] = [-0.08, -0.15, 1, 1]
    k[14] = [-0.05, 0.0, 1, 1]
    k[15] = [0.0, 0.05, 1, 1]  # face nose tip
    k[16] = [0.05, 0.0, 1, 1]
    k[17] = [0.08, -0.15, 1, 1]
    k[18] = [0.18, -0.18, 1, 1]
    k[19] = [0.28, -0.15, 1, 1]
    for i, (x, y) in enumerate(
        (
            (-0.08, 0.22),
            (0.0, 0.20),
            (0.08, 0.22),
            (-0.12, 0.25),
            (-0.04, 0.28),
            (0.0, 0.30),
            (0.12, 0.25),
            (0.04, 0.28),
        )
    ):
        k[20 + i] = [x, y, 1, 1]
    k[28] = [-0.18, -0.16, 1, 1]  # IRIS.L in EYE.L
    k[29] = [0.18, -0.16, 1, 1]  # IRIS.R in EYE.R
    # Body
    k[30] = [0.0, 0.05, 1, 1]
    k[31] = [0.0, 0.55, 1, 1]
    k[32] = [-0.35, 0.60, 1, 1]
    k[33] = [-0.40, 0.85, 1, 1]
    k[34] = [0.35, 0.60, 1, 1]
    k[35] = [0.40, 0.85, 1, 1]
    k[36] = [0.0, 0.80, 1, 1]
    return k


def test_normalized_pixels_roundtrip() -> None:
    k = _blank37()
    px = normalized_to_pixels(k, 768, 768)
    back = k.copy()
    for i in range(37):
        if k[i, 3] < 0.5:
            continue
        nx, ny = pixels_to_normalized(float(px[i, 0]), float(px[i, 1]), 768, 768)
        back[i, 0] = nx
        back[i, 1] = ny
    vis = k[:, 3] >= 0.5
    np.testing.assert_allclose(back[vis, :2], k[vis, :2], atol=1e-4)


def test_sanitize_body_nose_aligns_when_untracked() -> None:
    ref = _blank37()
    driven = ref.copy()
    driven[BODY_NOSE, 0] = 0.4
    driven[BODY_NOSE, 1] = 0.4
    out = sanitize_pose(driven, ref, body_tracked=False)
    assert abs(float(out[BODY_NOSE, 0]) - float(out[15, 0])) < 1e-5
    assert abs(float(out[BODY_NOSE, 1]) - float(out[15, 1])) < 1e-5 or float(
        out[BODY_NOSE, 1]
    ) <= float(out[31, 1])


def test_sanitize_body_nose_stays_attached_when_tracked() -> None:
    ref = _blank37()
    driven = ref.copy()
    driven[BODY_NOSE, 0] = 0.12
    driven[BODY_NOSE, 1] = 0.10
    out = sanitize_pose(driven, ref, body_tracked=True)
    np.testing.assert_allclose(out[BODY_NOSE, :2], out[15, :2], atol=2e-4)


def test_sanitize_tracked_allows_wider_body_travel() -> None:
    ref = _blank37()
    fh = face_height(ref)
    driven = ref.copy()
    # Move right shoulder by more than untracked cap but within tracked cap.
    mid = 0.5 * (TRAVEL_BODY + TRAVEL_BODY_TRACKED) * fh
    driven[32, 0] = float(ref[32, 0] - mid)
    out_loose = sanitize_pose(driven, ref, body_tracked=True, recenter=False)
    out_tight = sanitize_pose(driven, ref, body_tracked=False, recenter=False)
    # Tracked should preserve more of the intentional shoulder shift.
    d_loose = abs(float(out_loose[32, 0] - ref[32, 0]))
    d_tight = abs(float(out_tight[32, 0] - ref[32, 0]))
    assert d_loose > d_tight + 1e-4


def test_sanitize_mouth_inverted_cleared() -> None:
    ref = _blank37()
    driven = ref.copy()
    driven[25, 1] = float(driven[21, 1]) - 0.05
    out = sanitize_pose(driven, ref)
    assert float(out[25, 1]) >= float(out[21, 1])


def test_sanitize_preserves_large_mouth_expression() -> None:
    ref = _blank37()
    driven = ref.copy()
    driven[20:23, 1] -= 0.03
    driven[23, 1] += 0.08
    driven[24, 1] += 0.16
    driven[25, 1] += 0.18
    driven[26, 1] += 0.08
    driven[27, 1] += 0.16
    out = sanitize_pose(driven, ref, recenter=False)

    assert float(out[25, 1] - ref[25, 1]) > 0.15
    assert float(out[21, 1] - ref[21, 1]) < -0.02
    assert float(out[25, 1]) > float(out[21, 1])
    assert float(out[23, 1]) <= float(out[25, 1]) + 0.05 * face_height(ref)


def test_structural_only_safety_preserves_driven_face_transform() -> None:
    ref = _blank37()
    driven = ref.copy()
    driven[:28, 0] += 0.08
    driven[28:30, 0] += 0.08
    driven[21, 1] -= 0.03
    driven[25, 1] += 0.12
    out = sanitize_pose(
        driven,
        ref,
        recenter=False,
        clamp_travel=False,
        topology=False,
        lock_proportions=True,
        lock_face_proportions=False,
    )
    np.testing.assert_allclose(out[:28, 0], driven[:28, 0], atol=1e-5)
    assert float(out[25, 1] - out[21, 1]) > float(ref[25, 1] - ref[21, 1]) + 0.10


def test_draw_skeleton_only_without_face_iris() -> None:
    """Test Skeleton mode: body overlay works with face/iris drawing off."""
    k = _blank37()
    img = Image.new("RGB", (128, 128), color=(20, 20, 20))
    out = draw_keypoint_mesh(
        img,
        k,
        show_ids=False,
        show_face=False,
        show_iris=False,
        show_skeleton=True,
    )
    assert out.size == img.size
    # Skeleton drawing should change some pixels (bones/joints).
    assert not np.array_equal(np.asarray(out), np.asarray(img))
    # Face-only drawing should also work independently.
    face_out = draw_keypoint_mesh(
        img,
        k,
        show_ids=False,
        show_face=True,
        show_iris=False,
        show_skeleton=False,
    )
    assert not np.array_equal(np.asarray(face_out), np.asarray(out))


def test_overlay_skips_blocked_nose_slots() -> None:
    ids = overlay_visible_slots({"show_mesh": True, "show_skeleton": True, "show_nose": True})
    assert BODY_NOSE not in ids
    assert 15 not in ids
    assert 14 in ids
    assert 16 in ids
    assert 31 in ids


def test_block_model_slots_zeros_nose_tips() -> None:
    k = _blank37()
    assert float(k[15, 3]) >= 0.5
    assert float(k[30, 3]) >= 0.5
    out = block_model_slots(k)
    for idx in (15, 30):
        assert float(out[idx, 3]) < 0.5
        assert abs(float(out[idx, 0])) < 1e-8
        assert abs(float(out[idx, 1])) < 1e-8
    np.testing.assert_allclose(out[14, :2], k[14, :2], atol=1e-8)
    np.testing.assert_allclose(out[31, :2], k[31, :2], atol=1e-8)


def test_draw_skeleton_lost_uses_red() -> None:
    """Held/lost body overlay should tint red vs active green."""
    k = _blank37()
    img = Image.new("RGB", (128, 128), color=(10, 10, 10))
    active = draw_keypoint_mesh(
        img,
        k,
        show_ids=False,
        show_face=False,
        show_iris=False,
        show_skeleton=True,
        skeleton_lost=False,
    )
    lost = draw_keypoint_mesh(
        img,
        k,
        show_ids=False,
        show_face=False,
        show_iris=False,
        show_skeleton=True,
        skeleton_lost=True,
    )
    a = np.asarray(active)
    l = np.asarray(lost)
    assert not np.array_equal(a, l)
    # Lost overlay should contain more red-dominant pixels.
    lost_red = np.sum((l[:, :, 0] > 180) & (l[:, :, 1] < 120) & (l[:, :, 2] < 120))
    active_red = np.sum((a[:, :, 0] > 180) & (a[:, :, 1] < 120) & (a[:, :, 2] < 120))
    assert lost_red > active_red


def test_draw_mouth_snapped_uses_blue() -> None:
    """Near-closed snap turns mouth overlay from purple to blue."""
    from backend.pose_controller import MOUTH, MOUTH_COLOR, MOUTH_SNAPPED_COLOR

    k = _blank37()
    # Visible closed mouth slit in norm_crop space.
    for i, x in zip(MOUTH, (-0.12, 0.0, 0.12, -0.18, -0.10, 0.0, 0.18, 0.10)):
        k[i] = [x, 0.05, 1.0, 1.0]
    img = Image.new("RGB", (256, 256), color=(0, 0, 0))
    open_m = draw_keypoint_mesh(
        img, k, show_ids=False, show_iris=False, show_skeleton=False, mouth_snapped=False
    )
    snap_m = draw_keypoint_mesh(
        img, k, show_ids=False, show_iris=False, show_skeleton=False, mouth_snapped=True
    )
    o = np.asarray(open_m)
    s = np.asarray(snap_m)
    assert not np.array_equal(o, s)
    # Purple mouth pixels present when not snapped.
    purple = np.sum(
        (np.abs(o[:, :, 0].astype(np.int16) - MOUTH_COLOR[0]) < 30)
        & (np.abs(o[:, :, 1].astype(np.int16) - MOUTH_COLOR[1]) < 30)
        & (np.abs(o[:, :, 2].astype(np.int16) - MOUTH_COLOR[2]) < 30)
    )
    blue = np.sum(
        (np.abs(s[:, :, 0].astype(np.int16) - MOUTH_SNAPPED_COLOR[0]) < 30)
        & (np.abs(s[:, :, 1].astype(np.int16) - MOUTH_SNAPPED_COLOR[1]) < 30)
        & (np.abs(s[:, :, 2].astype(np.int16) - MOUTH_SNAPPED_COLOR[2]) < 30)
    )
    assert purple > 0
    assert blue > 0


def test_draw_keypoints_are_pixels() -> None:
    """Camera diagnostic path: draw absolute pixel keypoints without remapping."""
    img = Image.new("RGB", (200, 160), color=(5, 5, 5))
    k = np.zeros((37, 4), dtype=np.float32)
    # Place body nose / neck / shoulders in absolute pixels.
    k[30] = [100, 40, 1, 1]
    k[31] = [100, 70, 1, 1]
    k[32] = [70, 80, 1, 1]
    k[34] = [130, 80, 1, 1]
    k[36] = [100, 100, 1, 1]
    out = draw_keypoint_mesh(
        img,
        k,
        show_ids=False,
        show_face=False,
        show_iris=False,
        show_skeleton=True,
        keypoints_are_pixels=True,
    )
    assert out.size == img.size
    assert not np.array_equal(np.asarray(out), np.asarray(img))
    # Pixel near neck should be painted (green bone/joint).
    arr = np.asarray(out)
    assert arr[70, 100].sum() > arr[10, 10].sum()


def test_full_mesh_differs_from_body_only() -> None:
    k = _blank37()
    img = Image.new("RGB", (128, 128), color=(20, 20, 20))
    full = draw_keypoint_mesh(img, k, show_ids=False)
    body = draw_keypoint_mesh(
        img,
        k,
        show_ids=False,
        show_face=False,
        show_iris=False,
        show_skeleton=True,
    )
    assert not np.array_equal(np.asarray(full), np.asarray(body))


def test_draw_hair_overlay_tints_polygon() -> None:
    from backend.pose_controller import draw_hair_overlay

    img = Image.new("RGB", (64, 64), color=(10, 10, 10))
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.2]],
        }
    ]
    out = draw_hair_overlay(img, segs, lost=False)
    assert out.size == img.size
    assert not np.array_equal(np.asarray(out), np.asarray(img))
    pix = [
        {
            "class": "hair_left",
            "polygon": [[4.0, 4.0], [40.0, 4.0], [20.0, 40.0]],
        }
    ]
    out_px = draw_hair_overlay(img, pix, pixel_space=True)
    assert not np.array_equal(np.asarray(out_px), np.asarray(img))
    blank = draw_hair_overlay(img, None)
    assert np.array_equal(np.asarray(blank), np.asarray(img))


def test_draw_hair_overlay_has_no_shell_outline() -> None:
    import inspect

    from backend.pose_controller import draw_hair_overlay

    src = inspect.getsource(draw_hair_overlay)
    assert "outline=" not in src


def _chin_neck(k: np.ndarray) -> float:
    return float(np.linalg.norm(k[31, :2] - k[2, :2]))


def test_attachment_identity_on_reference() -> None:
    ref = _blank37()
    out = ref.copy()
    enforce_head_body_attachment(out, ref)
    np.testing.assert_allclose(out[31, :2], ref[31, :2], atol=1e-5)
    np.testing.assert_allclose(out[30, :2], out[15, :2], atol=1e-5)


def test_attachment_pulls_neck_when_face_flies_up() -> None:
    ref = _blank37()
    rest = _chin_neck(ref)
    out = ref.copy()
    # Slide the whole face up; leave the skeleton behind.
    for i in list(range(28)) + [28, 29]:
        out[i, 1] -= 0.55
    enforce_head_body_attachment(out, ref)
    assert _chin_neck(out) <= rest * ATTACH_LEN_MAX + 1e-4
    assert float(out[31, 1]) >= float(out[2, 1]) - 1e-4
    np.testing.assert_allclose(out[30, :2], out[15, :2], atol=2e-4)
    # Shoulders ride with the neck so the torso stays one piece.
    sh_off = (ref[32, :2] - ref[31, :2])
    np.testing.assert_allclose(out[32, :2] - out[31, :2], sh_off, atol=2e-4)


def test_attachment_allows_modest_scale_but_clamps_a_tear() -> None:
    ref = _blank37()
    rest = _chin_neck(ref)
    mild = ref.copy()
    mild[31, 1] = float(ref[2, 1]) + rest * 1.15
    enforce_head_body_attachment(mild, ref)
    assert abs(_chin_neck(mild) - rest * 1.15) < 0.02

    torn = ref.copy()
    torn[31, 1] = float(ref[2, 1]) + rest * 3.0
    enforce_head_body_attachment(torn, ref)
    assert _chin_neck(torn) <= rest * ATTACH_LEN_MAX + 1e-4


def test_harness_refs_match_person_left_iris() -> None:
    assert len(KEYPOINT_REFS) == 37
    assert ref_of(IRIS_L) == "IRIS.L"
    assert ref_of(IRIS_R) == "IRIS.R"
    assert slot_of("IRIS.L") == 28
    assert slot_of("right_iris") == 28
    assert slot_of("iris_l") == 28
    assert slot_of("EYE.L.IN") == 11
    assert slot_of("nose") == 15
    assert slot_of("BODY.NOSE") == 30
    assert slot_of(None) == -1
