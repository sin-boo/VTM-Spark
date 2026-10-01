"""The drawn body is bent to the torso Track Lab posed, and the two agree."""

from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import body_warp
from backend.body_warp import _px, body_turn, warp_body

_LAB_SKELETON = Path(__file__).resolve().parents[2] / "track_lab" / "backend" / "skeleton.py"
W = H = 256


@pytest.fixture(scope="module")
def lab():
    """Track Lab's torso, loaded on its own: it imports only numpy and cv2."""
    spec = importlib.util.spec_from_file_location("lab_skeleton", _LAB_SKELETON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Rig:
    locked = True

    def __init__(self, yaw=0.0, pitch=0.0, roll=0.0, scale=1.0, dx=0.0) -> None:
        self._turn = {"yaw": yaw, "pitch": pitch, "roll": roll}
        self._place = {"cx": 128.0, "cy": 128.0, "scale": scale, "body_dx": dx, "body_dy": 0.0}

    def place(self):
        return dict(self._place)

    def turn(self):
        return dict(self._turn)


def _face() -> np.ndarray:
    pts = np.zeros((28, 3), np.float32)
    pts[:, 2] = 1.0
    for i, xy in {0: (98, 88), 1: (104, 132), 2: (128, 156), 3: (152, 132), 4: (158, 88), 15: (128, 120), 21: (128, 140)}.items():
        pts[i, :2] = xy
    return pts


def _kps(lab, rig: _Rig) -> np.ndarray:
    body = lab.skeleton_from_face(_face())
    k = np.zeros((37, 4), np.float32)
    k[:28, :2] = _face()[:, :2]
    k[:28, 3] = 1.0
    for joint in lab.follow_skeleton(body, None, rig=rig):
        k[int(joint["id"]), :2] = (joint["x"], joint["y"])
        k[int(joint["id"]), 3] = 1.0
    k[:, 0] = k[:, 0] / W * 2.0 - 1.0
    k[:, 1] = k[:, 1] / H * 2.0 - 1.0
    return k


def _body_picture() -> Image.Image:
    """Green field, a striped body below the neck, a face block above it."""
    arr = np.zeros((H, W, 3), np.uint8)
    arr[:] = (0, 200, 0)
    arr[165:, 40:216] = (90, 140, 230)
    arr[165:, 120:136] = (240, 240, 240)  # a stripe down the front
    arr[80:158, 98:158] = (250, 210, 190)  # face
    return Image.fromarray(arr)


def test_constants_are_track_labs(lab) -> None:
    assert body_warp.BODY_HALF == lab.BODY_HALF
    assert body_warp.BODY_DEPTH == lab.BODY_DEPTH
    assert math.isclose(body_warp.MAX_TURN, lab._MAX_BODY_TURN)


@pytest.mark.parametrize("deg", [-40.0, -20.0, -5.0, 0.0, 12.0, 30.0])
@pytest.mark.parametrize("roll", [0.0, 12.0])
def test_the_turn_reads_back_from_the_shoulders(lab, deg, roll) -> None:
    """Whatever the tilt, size or walk, the shoulders give the turn back."""
    rest = _kps(lab, _Rig())
    live = _kps(lab, _Rig(yaw=math.radians(deg) / 1.5, roll=math.radians(roll) / 1.5, scale=1.1, dx=9.0))
    got = math.degrees(body_turn(_px(live, W, H), _px(rest, W, H)))
    assert abs(got - deg) < 1.0


def test_the_rest_pose_leaves_the_frame_alone(lab) -> None:
    rest = _kps(lab, _Rig())
    image = _body_picture()
    assert warp_body(image, rest, rest) is image
    assert warp_body(image, None, rest) is image
    assert warp_body(image, rest, None) is image


def test_a_shifted_torso_moves_the_body_and_not_the_face(lab) -> None:
    rest = _kps(lab, _Rig())
    moved = _kps(lab, _Rig(dx=12.0))
    image = _body_picture()
    out = np.asarray(warp_body(image, moved, rest))
    src = np.asarray(image)
    # Low on the body the stripe follows the torso to the right.
    row = 230
    stripe = np.where((out[row, :, 0] > 230) & (out[row, :, 1] > 230))[0]
    assert abs(float(stripe.mean()) - (127.5 + 12.0)) < 2.5
    # The face is left as drawn.
    assert np.array_equal(out[80:150, 98:158], src[80:150, 98:158])


def test_a_turn_closes_the_far_side_of_the_body(lab) -> None:
    rest = _kps(lab, _Rig())
    turned = _kps(lab, _Rig(yaw=math.radians(30.0) / 1.5))
    out = np.asarray(warp_body(_body_picture(), turned, rest))
    row = 230
    body = np.where(out[row, :, 2] > 200)[0]
    # Narrower than the drawn 176 px.
    assert body.max() - body.min() < 170


def test_the_display_bends_keys_before_blending_and_single_frames() -> None:
    from backend.stream import StreamRuntime

    play = inspect.getsource(StreamRuntime._play_display_job)
    assert play.index("self._bend_body(image, keypoints)") < play.index("self._blend_display_frame(")
    assert "self._bend_body(image, keypoints)" in inspect.getsource(StreamRuntime._on_frame)


def test_a_bend_that_fails_shows_the_frame_as_drawn(monkeypatch) -> None:
    from types import SimpleNamespace

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt.engine = SimpleNamespace(_ref_keypoints=np.zeros((37, 4), np.float32))

    def broken(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(body_warp, "warp_body", broken)
    image = _body_picture()
    assert StreamRuntime._bend_body(rt, image, np.zeros((37, 4), np.float32)) is image
    assert StreamRuntime._bend_body(rt, None, None) is None
