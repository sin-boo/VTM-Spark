"""Model-boundary conversion for legacy anime-HRNet checkpoint labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import engine as engine_module
from engine import (
    MODEL_KEYPOINT_LAYOUT,
    StreamEngine,
    _keypoints_for_model,
    neutral_keypoints,
)
from model_layout import schema37_to_hrnet_native37


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


def test_engine_model_boundary_uses_native_layout() -> None:
    assert MODEL_KEYPOINT_LAYOUT == "hrnet_native"
    schema = neutral_keypoints()
    np.testing.assert_allclose(
        _keypoints_for_model(schema),
        schema37_to_hrnet_native37(schema),
    )


def test_generate_sends_native_layout_but_keeps_schema_overlay(monkeypatch) -> None:
    stream = StreamEngine(device="cpu")
    stream._ready = True
    stream.model = torch.nn.Linear(1, 1)
    stream.vae = object()
    stream._ref_path = Path("reference.png")
    stream._ref_latent = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    stream._ref_face_latent = None
    stream._ref_keypoints = neutral_keypoints()
    stream._ref_keypoints_model = schema37_to_hrnet_native37(
        stream._ref_keypoints
    )

    captured: dict[str, np.ndarray] = {}

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

    target = neutral_keypoints()
    target[25, :2] = target[21, :2]
    stream.generate_from_keypoints(target, num_steps=1, sanitize="none")

    np.testing.assert_allclose(
        captured["target"],
        schema37_to_hrnet_native37(target),
    )
    np.testing.assert_allclose(captured["ref"], stream._ref_keypoints_model)
    np.testing.assert_allclose(stream.last_target_keypoints, target)
