"""Model-boundary conversion and per-checkpoint keypoint layout detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import backend.engine as engine_module
from backend.engine import StreamEngine, keypoints_for_model, neutral_keypoints
from backend.model_layout import (
    EYE_LIDS,
    LAYOUT_HRNET_NATIVE,
    LAYOUT_SCHEMA,
    LOWER_LID_JOIN,
    SHUT_LINE,
    keypoint_layout_from_config,
    lower_lid_shape,
    schema37_to_hrnet_native37,
    stamp_checkpoint_keypoint_layout,
    valid_lower_lids,
)
from backend.pose_controller import block_model_slots


def test_closed_schema_mouth_becomes_native_closed_signature() -> None:
    closed = neutral_keypoints()
    closed[25, :2] = closed[21, :2]

    native = schema37_to_hrnet_native37(closed)

    np.testing.assert_allclose(native[25, :2], native[27, :2], atol=1e-7)
    np.testing.assert_allclose(native[24], closed[23])
    np.testing.assert_allclose(native[26], closed[26])

    opened = closed.copy()
    opened[25, 1] += 0.16
    native_open = schema37_to_hrnet_native37(opened)
    assert float(native_open[27, 1] - native_open[25, 1]) > 0.15


def test_nose_iris_and_body_map_to_native_rows() -> None:
    schema = neutral_keypoints()
    native = schema37_to_hrnet_native37(schema)

    np.testing.assert_allclose(native[23], schema[15])
    np.testing.assert_allclose(native[28:37], schema[28:37])


def test_lower_lids_are_reflected_from_schema_upper_lids() -> None:
    schema = neutral_keypoints()
    native = schema37_to_hrnet_native37(schema)

    # Upper midpoint is above its corner chord; synthesized lower midpoint is
    # the same distance below it in +Y-down image coordinates.
    left_chord_y = 0.5 * (float(schema[11, 1]) + float(schema[13, 1]))
    assert float(schema[12, 1]) < left_chord_y
    assert float(native[15, 1]) > left_chord_y
    assert abs(
        (left_chord_y - float(schema[12, 1]))
        - (float(native[15, 1]) - left_chord_y)
    ) < 1e-6

    # On a blink the upper midpoint reaches the chord, so the synthetic lower
    # midpoint reaches the same chord as well.
    blink = schema.copy()
    blink[12, :2] = 0.5 * (blink[11, :2] + blink[13, :2])
    blink[18, :2] = 0.5 * (blink[17, :2] + blink[19, :2])
    native_blink = schema37_to_hrnet_native37(blink)
    np.testing.assert_allclose(native_blink[15, :2], blink[12, :2], atol=1e-7)
    np.testing.assert_allclose(native_blink[21, :2], blink[18, :2], atol=1e-7)


def _tall_eye_native() -> np.ndarray:
    """Native rows of a tall anime eye pair: lower lids well below the corners."""
    native = np.zeros((28, 3), dtype=np.float32)
    native[11:17, :2] = [[0, 0], [30, -20], [70, -10], [20, 40], [40, 45], [60, 35]]
    native[17:23, :2] = [[160, -10], [200, -20], [230, 0], [170, 35], [190, 45], [210, 40]]
    native[:, 2] = 1.0
    return native


def _rest_eyes(native: np.ndarray) -> np.ndarray:
    rest = neutral_keypoints()
    for i in (11, 12, 13, 17, 18, 19):
        rest[i, :2] = native[i, :2] / 400.0
    return rest


def test_lower_lid_shape_is_kept_in_each_eyes_corner_frame() -> None:
    native = _tall_eye_native()
    shape = lower_lid_shape(native)
    assert shape is not None
    moved = native.copy()
    moved[:, :2] = moved[:, :2] * 2.5 + 7.0
    np.testing.assert_allclose(lower_lid_shape(moved), shape, atol=1e-3)
    # A "lower lid" on or above the corner line is a missed detection.
    bad = native.copy()
    bad[14:17, 1] = -5.0
    assert lower_lid_shape(bad) is None
    assert valid_lower_lids(shape) == shape
    assert valid_lower_lids([[1, 2]]) is None


def _track_lab_shut(rest: np.ndarray) -> np.ndarray:
    """Track Lab's shut eye: each lid mid on its corner line."""
    shut = rest.copy()
    for (a, m, b), _lower in EYE_LIDS:
        chord = shut[b, :2] - shut[a, :2]
        t = float(np.dot(shut[m, :2] - shut[a, :2], chord) / np.dot(chord, chord))
        shut[m, :2] = shut[a, :2] + t * chord
    return shut


def test_own_lower_lids_at_rest_and_a_trained_shut_line_when_shut() -> None:
    native = _tall_eye_native()
    shape = lower_lid_shape(native)
    rest = _rest_eyes(native)

    opened = schema37_to_hrnet_native37(rest, ref=rest, lower_lids=shape)
    np.testing.assert_allclose(opened[14:17, :2], native[14:17, :2] / 400.0, atol=1e-5)
    np.testing.assert_allclose(opened[20:23, :2], native[20:23, :2] / 400.0, atol=1e-5)
    np.testing.assert_allclose(opened[11:14, :2], rest[11:14, :2], atol=1e-7)

    closed = schema37_to_hrnet_native37(_track_lab_shut(rest), ref=rest, lower_lids=shape)
    for eye, ((a, m, b), lower) in enumerate(EYE_LIDS):
        pts = closed[[a, m, b, *lower], :2]
        # One straight line: every point on the corner-to-corner line.
        chord = pts[2] - pts[0]
        normal = np.array([-chord[1], chord[0]]) / np.hypot(*chord)
        assert np.abs((pts - pts[0]) @ normal).max() < 1e-5
        # The lower lid's ends on the corners, both mids at the centre.
        np.testing.assert_allclose(closed[lower[0], :2], closed[a, :2], atol=1e-6)
        np.testing.assert_allclose(closed[lower[2], :2], closed[b, :2], atol=1e-6)
        np.testing.assert_allclose(closed[m, :2], 0.5 * (closed[a, :2] + closed[b, :2]), atol=1e-6)
        np.testing.assert_allclose(closed[lower[1], :2], closed[m, :2], atol=1e-6)
        # The line, corners too, sits SHUT_LINE of the way from the open upper
        # lid mid down to the open lower lid mid, in the rest corner frame.
        rest_chord = rest[b, :2] - rest[a, :2]
        rest_normal = np.array([-rest_chord[1], rest_chord[0]])
        upper_v = float((rest[m, :2] - rest[a, :2]) @ rest_normal / (rest_chord @ rest_chord))
        line_v = upper_v + SHUT_LINE * (shape[eye][1][1] - upper_v)
        np.testing.assert_allclose(closed[a, :2], rest[a, :2] + line_v * rest_normal, atol=1e-6)


def _part_shut(rest: np.ndarray, amount: float) -> np.ndarray:
    shut = _track_lab_shut(rest)
    out = rest.copy()
    out[[12, 18], :2] = rest[[12, 18], :2] + amount * (shut[[12, 18], :2] - rest[[12, 18], :2])
    return out


def test_a_squint_drops_the_upper_lid_and_keeps_corners_and_lower_lid() -> None:
    native = _tall_eye_native()
    shape = lower_lid_shape(native)
    rest = _rest_eyes(native)
    opened = schema37_to_hrnet_native37(rest, ref=rest, lower_lids=shape)
    closed = schema37_to_hrnet_native37(_track_lab_shut(rest), ref=rest, lower_lids=shape)
    for amount in (0.35, 0.5, LOWER_LID_JOIN):
        squint = schema37_to_hrnet_native37(_part_shut(rest, amount), ref=rest, lower_lids=shape)
        # Only the upper lid mid moves, its share of the way to the shut line.
        still = [11, 13, 14, 15, 16, 17, 19, 20, 21, 22]
        np.testing.assert_allclose(squint[still, :2], opened[still, :2], atol=1e-5)
        np.testing.assert_allclose(
            squint[[12, 18], 1],
            opened[[12, 18], 1] + amount * (closed[[12, 18], 1] - opened[[12, 18], 1]),
            atol=2e-3,
        )


def test_corners_and_lower_lid_meet_the_upper_lid_after_the_join() -> None:
    native = _tall_eye_native()
    shape = lower_lid_shape(native)
    rest = _rest_eyes(native)
    opened = schema37_to_hrnet_native37(rest, ref=rest, lower_lids=shape)
    closed = schema37_to_hrnet_native37(_track_lab_shut(rest), ref=rest, lower_lids=shape)
    rows = [11, 13, 14, 15, 16, 17, 19, 20, 21, 22]
    share = []
    for amount in (0.8, 0.9):
        out = schema37_to_hrnet_native37(_part_shut(rest, amount), ref=rest, lower_lids=shape)
        moved = np.abs(out[rows, :2] - opened[rows, :2]).max()
        share.append(moved / np.abs(closed[rows, :2] - opened[rows, :2]).max())
    assert 0.0 < share[0] < share[1] < 1.0


def test_a_lost_lid_point_keeps_the_seen_corners() -> None:
    native = _tall_eye_native()
    shape = lower_lid_shape(native)
    rest = _rest_eyes(native)
    lost = rest.copy()
    lost[12] = 0.0
    out = schema37_to_hrnet_native37(lost, ref=rest, lower_lids=shape)
    np.testing.assert_allclose(out[[11, 13], :2], rest[[11, 13], :2], atol=1e-7)
    np.testing.assert_array_equal(out[14:17], schema37_to_hrnet_native37(lost)[14:17])
    # The other eye still gets its own lower lid.
    np.testing.assert_allclose(out[20:23, :2], native[20:23, :2] / 400.0, atol=1e-5)


def test_a_wide_eye_keeps_its_corners_and_lower_lid() -> None:
    native = _tall_eye_native()
    shape = lower_lid_shape(native)
    rest = _rest_eyes(native)
    wide = rest.copy()
    wide[12, 1] -= 0.02
    out = schema37_to_hrnet_native37(wide, ref=rest, lower_lids=shape)
    np.testing.assert_allclose(out[[11, 12, 13], :2], wide[[11, 12, 13], :2], atol=1e-7)
    np.testing.assert_allclose(out[14:17, :2], native[14:17, :2] / 400.0, atol=1e-5)


def test_lower_lids_follow_the_eye_corners() -> None:
    native = _tall_eye_native()
    shape = lower_lid_shape(native)
    rest = _rest_eyes(native)
    moved = rest.copy()
    moved[list(range(11, 20)), 0] += 0.1
    out = schema37_to_hrnet_native37(moved, ref=rest, lower_lids=shape)
    np.testing.assert_allclose(out[14:17, 0], native[14:17, 0] / 400.0 + 0.1, atol=1e-5)


def test_no_lower_lids_keeps_the_mirrored_lids() -> None:
    schema = neutral_keypoints()
    np.testing.assert_allclose(
        schema37_to_hrnet_native37(schema, ref=schema, lower_lids=None),
        schema37_to_hrnet_native37(schema),
    )


def test_engine_reference_uses_its_lower_lids() -> None:
    native = _tall_eye_native()
    rest = _rest_eyes(native)
    eng = StreamEngine.__new__(StreamEngine)
    eng.keypoint_layout = LAYOUT_HRNET_NATIVE
    eng._ref_keypoints = rest
    eng._lower_lids = None
    eng._last_gen_latent = None
    eng.clear_last_gen_latent = lambda: None
    eng.set_lower_lids(lower_lid_shape(native))
    np.testing.assert_allclose(eng._ref_keypoints_model[15, :2], native[15, :2] / 400.0, atol=1e-5)


def test_native_layout_matches_legacy_training_signature() -> None:
    native = schema37_to_hrnet_native37(neutral_keypoints())
    legacy_mouth_channel = native[20:28, :2]
    ids = np.arange(20, 28)

    assert int(ids[np.argmax(legacy_mouth_channel[:, 0])]) == 22
    distances = np.linalg.norm(
        legacy_mouth_channel[:, None, :] - legacy_mouth_channel[None, :, :],
        axis=-1,
    )
    a, b = np.unravel_index(np.argmax(distances), distances.shape)
    assert {int(ids[a]), int(ids[b])} == {22, 24}


def test_layout_detector_explicit_schema_aliases() -> None:
    assert keypoint_layout_from_config({"keypoint_layout": "schema"}) == LAYOUT_SCHEMA
    assert keypoint_layout_from_config({"model_keypoint_layout": "label28"}) == LAYOUT_SCHEMA
    assert keypoint_layout_from_config({"label_layout": "full_stack"}) == LAYOUT_SCHEMA
    assert keypoint_layout_from_config({"keypoint_layout": "KEYPOINT_SCHEMA"}) == LAYOUT_SCHEMA


def test_layout_detector_explicit_native_aliases() -> None:
    assert keypoint_layout_from_config({"keypoint_layout": "hrnet_native"}) == LAYOUT_HRNET_NATIVE
    assert keypoint_layout_from_config({"keypoint_layout": "legacy"}) == LAYOUT_HRNET_NATIVE
    assert keypoint_layout_from_config({"model_keypoint_layout": "hrnet"}) == LAYOUT_HRNET_NATIVE


def test_layout_detector_missing_key_defaults_native() -> None:
    assert keypoint_layout_from_config(None) == LAYOUT_HRNET_NATIVE
    assert keypoint_layout_from_config({}) == LAYOUT_HRNET_NATIVE
    assert keypoint_layout_from_config({"num_keypoints": 37}) == LAYOUT_HRNET_NATIVE


def test_layout_detector_names_slot_14_is_nose() -> None:
    names = [f"face_{i}" for i in range(28)]
    names[14] = "face_14_nose"
    assert keypoint_layout_from_config({"keypoint_names": names}) == LAYOUT_SCHEMA
    assert keypoint_layout_from_config({"KEYPOINT_NAMES": names}) == LAYOUT_SCHEMA
    names[14] = "nose"
    assert keypoint_layout_from_config({"keypoint_names": names}) == LAYOUT_SCHEMA


def test_layout_detector_explicit_wins_over_names() -> None:
    names = [f"face_{i}" for i in range(28)]
    names[14] = "face_14_nose"
    assert (
        keypoint_layout_from_config(
            {"keypoint_layout": "hrnet_native", "keypoint_names": names}
        )
        == LAYOUT_HRNET_NATIVE
    )


def test_engine_defaults_to_native_until_load() -> None:
    stream = StreamEngine(device="cpu")
    assert stream.keypoint_layout == LAYOUT_HRNET_NATIVE


def test_runtime_status_reports_engine_layout() -> None:
    from backend.stream import StreamRuntime

    class _Runtime:
        engine = type(
            "E",
            (),
            {
                "keypoint_layout": LAYOUT_SCHEMA,
                "compile_status": "off",
                "compile_detail": "",
            },
        )()

    fields = StreamRuntime._compile_status_fields(_Runtime())
    assert fields["keypoint_layout"] == LAYOUT_SCHEMA
    assert fields["compile_model"] is False


def test_keypoints_for_model_schema_is_identity() -> None:
    schema = neutral_keypoints()
    out = keypoints_for_model(schema, LAYOUT_SCHEMA)
    np.testing.assert_allclose(out, schema)
    assert out is not schema


def test_keypoints_for_model_native_matches_remap() -> None:
    schema = neutral_keypoints()
    np.testing.assert_allclose(
        keypoints_for_model(schema, LAYOUT_HRNET_NATIVE),
        schema37_to_hrnet_native37(schema),
    )


def test_stamp_checkpoint_keypoint_layout(tmp_path: Path) -> None:
    path = tmp_path / "toy.pt"
    torch.save({"config": {"num_keypoints": 37}, "ema": {}}, path)
    assert stamp_checkpoint_keypoint_layout(path, "schema") == LAYOUT_SCHEMA
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["config"]["keypoint_layout"] == LAYOUT_SCHEMA
    assert ckpt["config"]["num_keypoints"] == 37
    assert keypoint_layout_from_config(ckpt["config"]) == LAYOUT_SCHEMA


def _fake_engine(layout: str) -> StreamEngine:
    stream = StreamEngine(device="cpu")
    stream._ready = True
    stream.model = torch.nn.Linear(1, 1)
    stream.vae = object()
    stream.keypoint_layout = layout
    stream._ref_path = Path("reference.png")
    stream._ref_latent = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    stream._ref_face_latent = None
    stream._ref_keypoints = neutral_keypoints()
    stream._ref_keypoints_model = keypoints_for_model(
        stream._ref_keypoints, layout
    )
    return stream


def _patch_generate(monkeypatch, captured: dict[str, np.ndarray]) -> None:
    def fake_denoise(_model, **kwargs):
        captured["target"] = np.asarray(kwargs["keypoints_target"]).copy()
        captured["ref"] = np.asarray(kwargs["ref_keypoints"]).copy()
        return torch.zeros((1, 4, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(engine_module, "denoise_keypoint", fake_denoise)
    monkeypatch.setattr(
        engine_module,
        "decode_sd_vae",
        lambda _vae, _latents: np.zeros((1, 8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        engine_module,
        "decode_tiny_vae",
        lambda _vae, _latents: np.zeros((1, 8, 8, 3), dtype=np.uint8),
    )


def test_generate_sends_native_layout_but_keeps_schema_overlay(monkeypatch) -> None:
    stream = _fake_engine(LAYOUT_HRNET_NATIVE)
    captured: dict[str, np.ndarray] = {}
    _patch_generate(monkeypatch, captured)

    target = neutral_keypoints()
    target[25, :2] = target[21, :2]
    stream.generate_from_keypoints(target, num_steps=1, sanitize="none")

    sent = np.asarray(captured["target"])
    if sent.ndim == 3:
        sent = sent[0]
    # Nose tips are blocked before the DiT; the overlay keeps them.
    np.testing.assert_allclose(sent, schema37_to_hrnet_native37(block_model_slots(target)))
    ref_sent = np.asarray(captured["ref"])
    expected_ref = np.asarray(stream._ref_keypoints_model)
    if ref_sent.ndim == 3:
        ref_sent = ref_sent[0]
    if expected_ref.ndim == 3:
        expected_ref = expected_ref[0]
    np.testing.assert_allclose(ref_sent, expected_ref)
    np.testing.assert_allclose(stream.last_target_keypoints, target)


def test_generate_sends_schema_layout_and_keeps_schema_overlay(monkeypatch) -> None:
    stream = _fake_engine(LAYOUT_SCHEMA)
    captured: dict[str, np.ndarray] = {}
    _patch_generate(monkeypatch, captured)

    target = neutral_keypoints()
    target[25, :2] = target[21, :2]
    stream.generate_from_keypoints(target, num_steps=1, sanitize="none")

    sent = np.asarray(captured["target"])
    if sent.ndim == 3:
        sent = sent[0]
    np.testing.assert_allclose(sent, block_model_slots(target))
    ref_sent = np.asarray(captured["ref"])
    if ref_sent.ndim == 3:
        ref_sent = ref_sent[0]
    np.testing.assert_allclose(ref_sent, stream._ref_keypoints)
    np.testing.assert_allclose(stream.last_target_keypoints, target)
