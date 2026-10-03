"""Shut eyes: the model's reference keypoints turn so the still's open eye is not copied in."""

from __future__ import annotations

import numpy as np
import torch

import backend.engine as engine_module
from backend.engine import (
    SHUT_REF_TURN_AT,
    SHUT_REF_TURN_DEG,
    SHUT_REF_TURN_FROM,
    StreamEngine,
    keypoints_for_model,
    neutral_keypoints,
    shut_ref_turn,
    turned_ref,
)
from backend.model_layout import (
    LAYOUT_HRNET_NATIVE,
    eye_shut,
    lower_lid_shape,
    schema37_to_hrnet_native37,
)

_EYES = ((11, 12, 13), (17, 18, 19))


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


def _shut(rest: np.ndarray, amount: float, eyes=_EYES) -> np.ndarray:
    """Each lid mid in ``eyes`` ``amount`` of the way down to its corner line."""
    out = rest.copy()
    for a, m, b in eyes:
        chord = out[b, :2] - out[a, :2]
        t = float(np.dot(out[m, :2] - out[a, :2], chord) / np.dot(chord, chord))
        on = out[a, :2] + t * chord
        out[m, :2] += amount * (on - out[m, :2])
    return out


def _angle(ref: np.ndarray, turned: np.ndarray, centre: np.ndarray, row: int) -> float:
    a = ref[row, :2].astype(np.float64) - centre
    b = turned[row, :2].astype(np.float64) - centre
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a, b)))))


# --- eye_shut -------------------------------------------------------------


def test_eye_shut_is_zero_at_rest_and_one_with_each_lid_on_its_corner_line() -> None:
    rest = _rest_eyes(_tall_eye_native())
    assert eye_shut(rest, rest) == 0.0
    assert abs(eye_shut(_shut(rest, 1.0), rest) - 1.0) < 1e-5
    # A wider-than-rest eye is open, not negative.
    wide = rest.copy()
    wide[12, 1] -= 0.02
    assert eye_shut(wide, rest) == 0.0


def test_eye_shut_is_partial_part_way() -> None:
    rest = _rest_eyes(_tall_eye_native())
    for amount in (0.25, 0.5, 0.75):
        assert abs(eye_shut(_shut(rest, amount), rest) - amount) < 1e-4


def test_eye_shut_takes_the_more_shut_eye() -> None:
    rest = _rest_eyes(_tall_eye_native())
    one = _shut(rest, 0.8, eyes=(_EYES[0],))
    one = _shut(one, 0.3, eyes=(_EYES[1],))
    assert abs(eye_shut(one, rest) - 0.8) < 1e-4
    other = _shut(rest, 0.6, eyes=(_EYES[1],))
    assert abs(eye_shut(other, rest) - 0.6) < 1e-4


def test_eye_shut_ignores_hidden_eye_points_and_bad_shapes() -> None:
    rest = _rest_eyes(_tall_eye_native())
    shut = _shut(rest, 1.0)
    hidden = shut.copy()
    hidden[[12, 18], 3] = 0.0
    assert eye_shut(hidden, rest) == 0.0
    hidden_rest = rest.copy()
    hidden_rest[[11, 17], 3] = 0.0
    assert eye_shut(shut, hidden_rest) == 0.0
    # One eye hidden: the other still counts.
    half_hidden = shut.copy()
    half_hidden[12, 3] = 0.0
    assert abs(eye_shut(half_hidden, rest) - 1.0) < 1e-5
    assert eye_shut(shut[:30], rest) == 0.0
    assert eye_shut(shut, rest[:, :3]) == 0.0
    assert eye_shut(np.zeros((37, 4), np.float32), rest) == 0.0


def test_eye_shut_is_the_conversions_shut_measure() -> None:
    """Half shut by eye_shut puts the model's upper lid mid half way from open to shut."""
    native = _tall_eye_native()
    shape = lower_lid_shape(native)
    rest = _rest_eyes(native)
    half = _shut(rest, 0.5)
    shut = eye_shut(half, rest)
    assert abs(shut - 0.5) < 1e-3
    opened = schema37_to_hrnet_native37(rest, ref=rest, lower_lids=shape)
    closed = schema37_to_hrnet_native37(_shut(rest, 1.0), ref=rest, lower_lids=shape)
    mid = schema37_to_hrnet_native37(half, ref=rest, lower_lids=shape)
    rows = [12, 18]
    np.testing.assert_allclose(
        mid[rows, :2], opened[rows, :2] + shut * (closed[rows, :2] - opened[rows, :2]), atol=2e-3
    )


# --- turned_ref -----------------------------------------------------------


def _ref_model() -> np.ndarray:
    native = _tall_eye_native()
    rest = _rest_eyes(native)
    return keypoints_for_model(rest, LAYOUT_HRNET_NATIVE, ref=rest, lower_lids=lower_lid_shape(native))


def test_turned_ref_by_zero_is_an_equal_copy() -> None:
    ref = _ref_model()
    out = turned_ref(ref, 0.0)
    assert out is not ref
    np.testing.assert_array_equal(out, ref)
    out[0, 0] += 1.0
    assert out[0, 0] != ref[0, 0]


def test_turned_ref_keeps_distances_and_turns_about_the_face_centre() -> None:
    ref = _ref_model()
    out = turned_ref(ref, 20.0)
    seen = ref[:, 3] >= 0.5
    assert seen.sum() > 3

    def dists(k: np.ndarray) -> np.ndarray:
        pts = k[seen, :2].astype(np.float64)
        return np.linalg.norm(pts[:, None] - pts[None], axis=-1)

    np.testing.assert_allclose(dists(out), dists(ref), atol=1e-5)
    face = ref[:28][seen[:28], :2].astype(np.float64)
    centre = face.mean(axis=0)
    np.testing.assert_allclose(out[:28][seen[:28], :2].mean(axis=0), centre, atol=1e-5)
    for row in np.flatnonzero(seen):
        if np.linalg.norm(ref[row, :2] - centre) > 1e-3:
            assert abs(_angle(ref, out, centre, int(row)) - 20.0) < 1e-2
    # Only x, y move.
    np.testing.assert_array_equal(out[:, 2:], ref[:, 2:])


def test_turned_ref_leaves_hidden_rows_alone() -> None:
    ref = _ref_model()
    ref[[5, 30], 3] = 0.0
    ref[[5, 30], :2] = [[0.9, -0.9], [0.7, 0.4]]
    out = turned_ref(ref, 15.0)
    np.testing.assert_array_equal(out[[5, 30]], ref[[5, 30]])
    # The hidden face row is not part of the centre.
    seen = ref[:, 3] >= 0.5
    centre = ref[:28][seen[:28], :2].astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(out[:28][seen[:28], :2].mean(axis=0), centre, atol=1e-5)


# --- StreamEngine._shut_eye_ref --------------------------------------------


def _ref_engine() -> StreamEngine:
    native = _tall_eye_native()
    rest = _rest_eyes(native)
    eng = StreamEngine.__new__(StreamEngine)
    eng.keypoint_layout = LAYOUT_HRNET_NATIVE
    eng._ref_keypoints = rest
    eng._lower_lids = lower_lid_shape(native)
    eng._ref_keypoints_model = keypoints_for_model(
        rest, LAYOUT_HRNET_NATIVE, ref=rest, lower_lids=eng._lower_lids
    )
    return eng


def test_open_eyes_keep_the_stored_reference() -> None:
    eng = _ref_engine()
    rest = eng._ref_keypoints
    assert eng._shut_eye_ref(rest[None]) is eng._ref_keypoints_model
    wide = rest.copy()
    wide[12, 1] -= 0.02
    assert eng._shut_eye_ref(np.stack([rest, wide])) is eng._ref_keypoints_model


def test_shut_eyes_turn_the_reference() -> None:
    eng = _ref_engine()
    rest = eng._ref_keypoints
    stored = eng._ref_keypoints_model.copy()
    out = eng._shut_eye_ref(_shut(rest, 1.0)[None])
    np.testing.assert_allclose(out, turned_ref(stored, SHUT_REF_TURN_DEG), atol=1e-5)
    assert SHUT_REF_TURN_DEG == 20.0
    # The stored reference is untouched.
    np.testing.assert_array_equal(eng._ref_keypoints_model, stored)
    # A squint keeps the still's own eye; nearly shut turns part way.
    assert eng._shut_eye_ref(_shut(rest, 0.5)[None]) is eng._ref_keypoints_model
    near = eng._shut_eye_ref(_shut(rest, 0.8)[None])
    np.testing.assert_allclose(near, turned_ref(stored, shut_ref_turn(0.8)), atol=1e-3)
    assert 0.0 < shut_ref_turn(0.8) < SHUT_REF_TURN_DEG


def test_the_reference_turn_starts_near_shut() -> None:
    assert shut_ref_turn(0.0) == 0.0
    assert shut_ref_turn(SHUT_REF_TURN_FROM) == 0.0
    assert shut_ref_turn(SHUT_REF_TURN_AT) == SHUT_REF_TURN_DEG
    assert shut_ref_turn(1.0) == SHUT_REF_TURN_DEG
    steps = [shut_ref_turn(s / 20.0) for s in range(21)]
    assert steps == sorted(steps)


def test_a_batch_turns_by_its_most_shut_key() -> None:
    eng = _ref_engine()
    rest = eng._ref_keypoints
    batch = np.stack([_shut(rest, 1.0), _shut(rest, 0.25), rest])
    np.testing.assert_allclose(
        eng._shut_eye_ref(batch), turned_ref(eng._ref_keypoints_model, SHUT_REF_TURN_DEG), atol=1e-5
    )


def test_no_reference_returns_what_is_stored() -> None:
    eng = _ref_engine()
    eng._ref_keypoints = None
    assert eng._shut_eye_ref(_shut(_rest_eyes(_tall_eye_native()), 1.0)[None]) is eng._ref_keypoints_model
    eng._ref_keypoints_model = None
    assert eng._shut_eye_ref(neutral_keypoints()[None]) is None


# --- _denoise_to_latents_locked ---------------------------------------------


def _gen_engine() -> StreamEngine:
    native = _tall_eye_native()
    rest = _rest_eyes(native)
    stream = StreamEngine(device="cpu")
    stream._ready = True
    stream.model = torch.nn.Linear(1, 1)
    stream.vae = object()
    stream.keypoint_layout = LAYOUT_HRNET_NATIVE
    stream._ref_path = None
    stream._ref_latent = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    stream._ref_face_latent = None
    stream._ref_keypoints = rest
    stream._lower_lids = lower_lid_shape(native)
    stream._ref_keypoints_model = stream._model_keypoints(rest, ref=rest)
    stream.set_hold_last(False)
    return stream


def _graph_ref(monkeypatch, target: np.ndarray) -> tuple[StreamEngine, np.ndarray]:
    stream = _gen_engine()
    captured: dict[str, np.ndarray] = {}

    def fake_graph(kps_model, hair_maps, prev, start_t, kps_ref=None):
        captured["ref"] = np.asarray(kps_ref).copy()
        bsz = int(np.shape(kps_model)[0])
        return torch.zeros((bsz, 4, 2, 2)), np.zeros((bsz, 8, 8, 3), np.uint8), 0.0

    def no_eager(*_a, **_k):
        raise AssertionError("graph path expected")

    monkeypatch.setattr(stream, "_graph_ok", lambda *a: True)
    monkeypatch.setattr(stream, "_run_graph_frame", fake_graph)
    monkeypatch.setattr(engine_module, "denoise_keypoint", no_eager)
    stream.denoise_to_latents(target, num_steps=1, sanitize="none")
    return stream, captured["ref"]


def test_graph_frame_gets_the_turned_reference_when_eyes_shut(monkeypatch) -> None:
    rest = _rest_eyes(_tall_eye_native())
    stream, sent = _graph_ref(monkeypatch, _shut(rest, 1.0))
    np.testing.assert_allclose(sent, turned_ref(stream._ref_keypoints_model, SHUT_REF_TURN_DEG), atol=1e-5)
    assert np.abs(sent[:28, :2] - stream._ref_keypoints_model[:28, :2]).max() > 1e-3


def test_graph_frame_gets_the_stored_reference_when_eyes_open(monkeypatch) -> None:
    rest = _rest_eyes(_tall_eye_native())
    stream, sent = _graph_ref(monkeypatch, rest)
    np.testing.assert_array_equal(sent, stream._ref_keypoints_model)


def test_eager_denoise_gets_the_turned_reference_when_eyes_shut(monkeypatch) -> None:
    stream = _gen_engine()
    captured: dict[str, np.ndarray] = {}

    def fake_denoise(_model, **kwargs):
        captured["ref"] = np.asarray(kwargs["ref_keypoints"]).copy()
        return torch.zeros((1, 4, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(stream, "_graph_ok", lambda *a: False)
    monkeypatch.setattr(engine_module, "denoise_keypoint", fake_denoise)
    stream.denoise_to_latents(_shut(stream._ref_keypoints, 1.0), num_steps=1, sanitize="none")
    np.testing.assert_allclose(
        captured["ref"], turned_ref(stream._ref_keypoints_model, SHUT_REF_TURN_DEG), atol=1e-5
    )
