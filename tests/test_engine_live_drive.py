"""Tests for neck-anchored live body retargeting."""

from __future__ import annotations

import numpy as np

from engine import apply_live_deltas_to_ref, neutral_keypoints
from live_poser_client import (
    body_method_kind,
    body_tracking_active,
    body_tracking_label,
    is_body_tracked,
)


def test_body_method_classification() -> None:
    assert is_body_tracked("mediapipe_pose_lite")
    assert is_body_tracked("yolov8n-pose")
    assert not is_body_tracked("synthetic_from_face")
    assert not is_body_tracked("mediapipe_pose_lite_held")
    assert body_method_kind("synthetic_from_face") == "synthetic"
    assert body_method_kind("yolov8n-pose_held") == "held"
    assert body_method_kind("none") == "none"


def test_body_tracking_active_strict() -> None:
    assert body_tracking_active("mediapipe_pose_lite", lost=False)
    assert not body_tracking_active("mediapipe_pose_lite", lost=True)
    assert not body_tracking_active("synthetic_from_face", lost=False)
    assert not body_tracking_active("mediapipe_pose_lite_held", lost=True)
    assert not body_tracking_active("none", lost=False)


def test_body_tracking_label_states() -> None:
    assert body_tracking_label(None, tracking=False) == "Body: OFF"
    waiting = body_tracking_label("none", tracking=True, waiting=True)
    assert "WAITING" in waiting
    active = body_tracking_label(
        "mediapipe_pose_lite",
        tracking=True,
        lost=False,
        visible_body=7,
        age=0.05,
    )
    assert "ACTIVE (human)" in active
    assert "mediapipe_pose_lite" in active
    assert "joints=7/7" in active
    synth = body_tracking_label("synthetic_from_face", tracking=True, lost=False)
    assert "SYNTHETIC" in synth
    held = body_tracking_label(
        "mediapipe_pose_lite_held", tracking=True, lost=True
    )
    assert "HELD/LOST" in held
    none = body_tracking_label("none", tracking=True, lost=False)
    assert "NONE" in none
    drive_off = body_tracking_label(
        "mediapipe_pose_lite", tracking=True, drive_pose=False
    )
    assert "drive=off" in drive_off


def test_apply_live_deltas_identity_at_origin() -> None:
    ref = neutral_keypoints()
    live = ref.copy()
    # Webcam pose identical to origin → body joints stay on the character ref.
    # Sanitize may still repair neutral template topology (neck/mouth), so only
    # assert shoulder/elbow/chest identity.
    out = apply_live_deltas_to_ref(
        ref,
        live,
        live.copy(),
        body_method="mediapipe_pose_lite",
        body_lost=False,
    )
    for i in (32, 33, 34, 35, 36):
        np.testing.assert_allclose(out[i, :2], ref[i, :2], atol=5e-3)


def test_neck_anchored_body_ignores_global_translation() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    # Shift whole body+face right — neck-anchored body should stay near ref.
    live[:, 0] += 0.25
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite",
        body_lost=False,
    )
    for i in (31, 32, 34, 36):
        assert abs(float(out[i, 0] - ref[i, 0])) < 0.05, f"slot {i} drifted"


def test_shoulder_raise_retargets() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    # Raise both elbows (smaller y in our +y-down convention? wait +y is down,
    # so raising arms means decreasing y).
    live[33, 1] -= 0.15
    live[35, 1] -= 0.15
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite",
        body_lost=False,
    )
    assert float(out[33, 1]) < float(ref[33, 1]) - 0.02
    assert float(out[35, 1]) < float(ref[35, 1]) - 0.02


def test_held_body_keeps_prev() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    live[32, 0] -= 0.2
    prev = ref[30:37].copy()
    prev[2, 0] = float(ref[32, 0] - 0.18)  # remembered right shoulder
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite_held",
        body_lost=True,
        prev_body=prev,
    )
    assert abs(float(out[32, 0]) - float(prev[2, 0])) < 1e-5


def test_apply_live_deltas_rejects_norm_full() -> None:
    ref = neutral_keypoints()
    live = ref.copy()
    try:
        apply_live_deltas_to_ref(
            ref,
            live,
            live.copy(),
            live_coord_space="norm_full",
            origin_coord_space="norm_crop",
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "norm_crop" in str(exc)


def test_isotropic_shoulder_delta_in_norm_crop() -> None:
    """Equal Δx/Δy on shoulders in shared norm_crop stay directionally equal."""
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    live[32, 0] -= 0.08
    live[32, 1] -= 0.08
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    dx = float(out[32, 0] - ref[32, 0])
    dy = float(out[32, 1] - ref[32, 1])
    assert dx < -0.01
    assert dy < -0.01
    # Shoulder scale is uniform; magnitudes should stay close after sanitize.
    assert abs(abs(dx) - abs(dy)) < 0.08


def test_face_mouth_still_moves() -> None:
    ref = neutral_keypoints()
    origin = neutral_keypoints()
    live = origin.copy()
    # Open mouth on live.
    live[25, 1] += 0.12
    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="synthetic_from_face",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    assert float(out[25, 1]) > float(ref[25, 1]) + 0.01


def _shoulder_width(k: np.ndarray) -> float:
    return float(np.linalg.norm(k[32, :2] - k[34, :2]))


def test_mismatched_shoulder_width_preserves_ref_at_rest() -> None:
    """Wide human shoulders must not replace the anime character's width.

    At zero live delta (live == origin), retargeted shoulders stay on the
    character reference proportions regardless of how wide the human is.
    """
    ref = neutral_keypoints()
    # Anime character: relatively narrow shoulders.
    ref[32, 0] = -0.22
    ref[34, 0] = 0.22
    ref_sw = _shoulder_width(ref)

    # Human rest pose: much wider shoulders, different absolute placement.
    origin = neutral_keypoints()
    origin[32, 0] = -0.55
    origin[34, 0] = 0.55
    live = origin.copy()  # zero delta

    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    out_sw = _shoulder_width(out)
    # Character width preserved; human absolute width must not leak through.
    assert abs(out_sw - ref_sw) < 0.03
    assert abs(out_sw - _shoulder_width(origin)) > 0.2
    for i in (32, 34):
        np.testing.assert_allclose(out[i, :2], ref[i, :2], atol=2e-2)


def test_mismatched_shoulder_motion_is_relative() -> None:
    """Raise one human shoulder → character shoulder moves, width stays anime."""
    ref = neutral_keypoints()
    ref[32, 0] = -0.22
    ref[34, 0] = 0.22
    ref_sw = _shoulder_width(ref)

    origin = neutral_keypoints()
    origin[32, 0] = -0.55
    origin[34, 0] = 0.55

    live = origin.copy()
    # Raise right shoulder (smaller y = up in +y-down space).
    live[32, 1] -= 0.12

    out = apply_live_deltas_to_ref(
        ref,
        live,
        origin,
        body_method="mediapipe_pose_lite",
        body_lost=False,
        live_coord_space="norm_crop",
        origin_coord_space="norm_crop",
    )
    assert float(out[32, 1]) < float(ref[32, 1]) - 0.02
    # Still character-proportioned, not human-wide.
    assert abs(_shoulder_width(out) - ref_sw) < 0.08
    assert _shoulder_width(out) < _shoulder_width(origin) - 0.15
