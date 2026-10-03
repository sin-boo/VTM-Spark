import numpy as np
import torch

import backend.engine as engine_module
from backend.engine import (
    STREAM_HOLD_LAST,
    STREAM_HOLD_LAST_T,
    STREAM_HOLD_REF_PULL,
    StreamEngine,
    anchor_hold_latent,
    face_pose_delta,
    hold_ease,
    hold_plan,
    lid_delta,
    neutral_keypoints,
    pose_move,
    snap_move,
    snap_alpha,
)
from backend.model_layout import LAYOUT_HRNET_NATIVE
from backend.ui_session import load_ui_session, save_ui_session
from inference_keypoint import flow_start_from_last


def test_flow_start_skips_without_last_or_zero_t() -> None:
    noise = torch.ones((1, 4, 2, 2))
    times = torch.tensor([0.0, 0.5, 1.0])
    latents, used = flow_start_from_last(noise, times, None, 0.6)
    assert latents is noise
    assert used is times
    last = torch.zeros_like(noise)
    latents, used = flow_start_from_last(noise, times, last, 0.0)
    assert latents is noise
    assert used is times


def test_flow_start_mixes_and_trims_schedule() -> None:
    noise = torch.ones((1, 4, 2, 2))
    last = torch.zeros((1, 4, 2, 2))
    times = torch.tensor([0.0, 0.23, 1.0])
    latents, used = flow_start_from_last(noise, times, last, 0.6)
    torch.testing.assert_close(latents, 0.4 * noise)
    assert abs(used[0].item() - 0.6) < 1e-6
    assert used[-1].item() == 1.0
    assert used.numel() == 2


def test_flow_start_expands_last_to_batch() -> None:
    noise = torch.ones((2, 4, 2, 2))
    last = torch.zeros((1, 4, 2, 2))
    times = torch.linspace(0.0, 1.0, 3)
    latents, used = flow_start_from_last(noise, times, last, 0.5)
    assert latents.shape == noise.shape
    torch.testing.assert_close(latents, 0.5 * noise)
    assert abs(used[0].item() - 0.5) < 1e-6


def test_engine_default_hold_last_on() -> None:
    engine = StreamEngine(checkpoint="missing.pt", device="cpu")
    assert engine.hold_last is STREAM_HOLD_LAST
    assert STREAM_HOLD_LAST is True
    assert STREAM_HOLD_LAST_T == 0.28
    assert STREAM_HOLD_REF_PULL == 0.22
    engine.set_hold_last(False)
    assert engine.hold_last is False


def _fake_engine() -> StreamEngine:
    stream = StreamEngine(device="cpu")
    stream._ready = True
    stream.model = torch.nn.Linear(1, 1)
    stream.vae = object()
    stream.keypoint_layout = LAYOUT_HRNET_NATIVE
    stream._ref_path = None
    stream._ref_latent = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    stream._ref_face_latent = None
    stream._ref_keypoints = neutral_keypoints()
    stream._ref_keypoints_model = stream._ref_keypoints.copy()
    return stream


def test_second_generate_passes_last_latent(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_denoise(_model, **kwargs):
        captured.append(
            {
                "last": kwargs.get("last_latent"),
                "start_t": float(kwargs.get("start_t") or 0.0),
            }
        )
        return torch.full((1, 4, 2, 2), float(len(captured)), dtype=torch.float32)

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

    stream = _fake_engine()
    stream.set_hold_last(True)
    kps = neutral_keypoints()
    stream.generate_from_keypoints(kps, num_steps=1, sanitize="none")
    stream.generate_from_keypoints(kps, num_steps=1, sanitize="none")

    assert captured[0]["last"] is None
    assert captured[0]["start_t"] == 0.0
    assert captured[1]["last"] is not None
    torch.testing.assert_close(
        captured[1]["last"],
        anchor_hold_latent(torch.full((1, 4, 2, 2), 1.0), stream._ref_latent),
    )
    assert captured[1]["start_t"] == STREAM_HOLD_LAST_T


def test_hold_eases_off_when_the_face_moves() -> None:
    assert hold_ease(0.0) == 1.0
    assert hold_ease(0.01) == 1.0
    assert hold_ease(0.2) == 0.0
    mid = hold_ease(0.054)
    assert 0.2 < mid < 0.8


def test_hold_plan_drops_on_rest_drift_even_when_the_step_is_tiny() -> None:
    start_t, pull = hold_plan(0.01, 0.01)
    assert start_t == STREAM_HOLD_LAST_T
    assert pull == STREAM_HOLD_REF_PULL
    start_t, pull = hold_plan(0.01, 0.2)
    assert start_t == 0.0
    assert pull == 0.0


def test_face_pose_delta_ignores_parked_and_measures_travel() -> None:
    rest = neutral_keypoints()
    moved = rest.copy()
    moved[:28, 0] += 0.2
    assert face_pose_delta(rest, rest) == 0.0
    assert face_pose_delta(rest, moved) > 0.15
    parked = rest.copy()
    parked[3, :2] = (0.9, 0.9)
    parked[3, 3] = 0.0
    assert face_pose_delta(rest, parked) < 0.02


def _blink_step(rest: np.ndarray, amount: float) -> np.ndarray:
    """Both lid mids ``amount`` of the way down to their corner lines."""
    out = rest.copy()
    for a, m, b in ((11, 12, 13), (17, 18, 19)):
        chord = out[b, :2] - out[a, :2]
        t = float(np.dot(out[m, :2] - out[a, :2], chord) / np.dot(chord, chord))
        on = out[a, :2] + t * chord
        out[m, :2] += amount * (on - out[m, :2])
    return out


def test_a_blink_counts_as_a_move_for_hold_and_snap() -> None:
    rest = neutral_keypoints()
    step = _blink_step(rest, 0.5)
    # The mean face travel barely sees two lids...
    assert hold_ease(face_pose_delta(rest, step)) == 1.0
    # ...so a lid's move counts on its own: the hold drops and Snap shows the new key.
    assert lid_delta(rest, step) > 0.0
    assert hold_ease(pose_move(rest, step)) == 0.0
    assert snap_alpha(0.58, snap_move(rest, step)) == 1.0


def test_lid_delta_ignores_a_head_slide_and_lid_jitter() -> None:
    rest = neutral_keypoints()
    slid = rest.copy()
    slid[:28, 0] += 0.05
    assert lid_delta(rest, slid) < 1e-6
    jitter = _blink_step(rest, 0.02)
    assert hold_ease(pose_move(rest, jitter)) == 1.0


def test_snap_still_blends_through_a_pixel_of_lid_wobble() -> None:
    rest = neutral_keypoints()
    wobble = rest.copy()
    wobble[12, 1] += 0.002  # about a pixel at 512, a tenth of this shallow lid
    assert snap_alpha(0.58, snap_move(rest, wobble)) == 0.58
    assert hold_ease(pose_move(rest, wobble)) == 1.0
    # A third of a blink in one key is a move on either scale.
    quarter = _blink_step(rest, 0.33)
    assert snap_alpha(0.58, snap_move(rest, quarter)) == 1.0


def test_anchor_hold_pulls_toward_ref() -> None:
    last = torch.ones((1, 4, 2, 2))
    ref = torch.zeros((1, 4, 2, 2))
    out = anchor_hold_latent(last, ref, pull=0.25)
    torch.testing.assert_close(out, torch.full((1, 4, 2, 2), 0.75))
    assert anchor_hold_latent(last, None) is last


def test_big_face_move_drops_hold(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_denoise(_model, **kwargs):
        captured.append(
            {
                "last": kwargs.get("last_latent"),
                "start_t": float(kwargs.get("start_t") or 0.0),
            }
        )
        return torch.ones((1, 4, 2, 2), dtype=torch.float32)

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

    stream = _fake_engine()
    stream.set_hold_last(True)
    rest = neutral_keypoints()
    turned = rest.copy()
    turned[:28, 0] += 0.2
    stream.generate_from_keypoints(rest, num_steps=1, sanitize="none")
    stream.generate_from_keypoints(turned, num_steps=1, sanitize="none")
    assert captured[1]["last"] is None
    assert captured[1]["start_t"] == 0.0


def test_slow_side_walk_drops_hold(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_denoise(_model, **kwargs):
        captured.append(
            {
                "last": kwargs.get("last_latent"),
                "start_t": float(kwargs.get("start_t") or 0.0),
            }
        )
        return torch.ones((1, 4, 2, 2), dtype=torch.float32)

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

    stream = _fake_engine()
    stream.set_hold_last(True)
    rest = neutral_keypoints()
    stream.generate_from_keypoints(rest, num_steps=1, sanitize="none")
    for i in range(1, 12):
        stepped = rest.copy()
        stepped[:28, 0] += 0.02 * i
        stream.generate_from_keypoints(stepped, num_steps=1, sanitize="none")
    assert captured[-1]["last"] is None
    assert captured[-1]["start_t"] == 0.0


def test_hold_last_off_does_not_pass_last_latent(monkeypatch) -> None:
    captured: list[object] = []

    def fake_denoise(_model, **kwargs):
        captured.append(kwargs.get("last_latent"))
        return torch.ones((1, 4, 2, 2), dtype=torch.float32)

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

    stream = _fake_engine()
    stream.set_hold_last(False)
    kps = neutral_keypoints()
    stream.generate_from_keypoints(kps, num_steps=1, sanitize="none")
    stream.generate_from_keypoints(kps, num_steps=1, sanitize="none")
    assert captured[0] is None
    assert captured[1] is None
    assert stream._last_gen_latent is not None


def _counting_engine(monkeypatch) -> tuple[StreamEngine, list]:
    captured: list = []

    def fake_denoise(_model, **kwargs):
        captured.append(kwargs.get("last_latent"))
        return torch.full((1, 4, 2, 2), float(len(captured)), dtype=torch.float32)

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
    stream = _fake_engine()
    stream.set_hold_last(True)
    return stream, captured


def test_a_long_hold_never_restarts_from_noise(monkeypatch) -> None:
    """A still face keeps holding: a periodic restart from noise redrew the
    line work and flickered every ~2 s. The still pull keeps it anchored."""
    stream, captured = _counting_engine(monkeypatch)
    kps = neutral_keypoints()
    for _ in range(40):
        stream.generate_from_keypoints(kps, num_steps=1, sanitize="none")
    fresh = [i for i, last in enumerate(captured) if last is None]
    assert fresh == [0]


def test_a_new_rest_drops_the_hold(monkeypatch) -> None:
    stream, captured = _counting_engine(monkeypatch)
    kps = neutral_keypoints()
    stream.generate_from_keypoints(kps, num_steps=1, sanitize="none")
    stream.adopt_ref_keypoints(kps, persist=False)
    assert stream._last_gen_latent is None
    stream.generate_from_keypoints(kps, num_steps=1, sanitize="none")
    assert captured[1] is None


def test_session_stores_hold_last(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.ui_session.models_root", lambda: tmp_path)
    save_ui_session(hold_last=False)
    st = load_ui_session()
    assert st["hold_last"] is False
    save_ui_session(hold_last=True)
    st = load_ui_session()
    assert st["hold_last"] is True
