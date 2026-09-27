"""Model-boundary conversion and per-checkpoint keypoint layout detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import backend.engine as engine_module
from backend.engine import StreamEngine, keypoints_for_model, neutral_keypoints
from backend.model_layout import (
    LAYOUT_HRNET_NATIVE,
    LAYOUT_SCHEMA,
    keypoint_layout_from_config,
    schema37_to_hrnet_native37,
    stamp_checkpoint_keypoint_layout,
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
