import numpy as np
from PIL import Image

from backend.engine import neutral_keypoints
from backend.frame_interp import (
    blend_images,
    inbetween_frames,
    inbetween_image,
    inbetween_slot_s,
    inbetween_ts,
    lerp_stream_pose,
    playout_gap,
    print_inbetween_count,
)


def test_lerp_pose_midpoint() -> None:
    a = neutral_keypoints()
    b = a.copy()
    a[:, 0] = -0.4
    b[:, 0] = 0.4
    mid = lerp_stream_pose(a, b, 0.5)
    np.testing.assert_allclose(mid[:, 0], 0.0, atol=1e-5)


def test_blend_images_midpoint() -> None:
    black = Image.new("RGB", (8, 8), (0, 0, 0))
    white = Image.new("RGB", (8, 8), (255, 255, 255))
    mid = np.asarray(blend_images(black, white, 0.5))
    assert 120 <= int(mid.mean()) <= 140


def test_inbetween_ts_one_is_half() -> None:
    assert inbetween_ts(0) == []
    assert inbetween_ts(1) == [0.5]
    assert inbetween_ts(2) == [1.0 / 3.0, 2.0 / 3.0]


def test_inbetween_slot_does_not_sleep() -> None:
    assert inbetween_slot_s(22.0, 0) == 0.0
    assert inbetween_slot_s(22.0, 1) == 0.0
    assert inbetween_slot_s(10.0, 2) == 0.0


def test_print_inbetween_skips_when_behind() -> None:
    assert print_inbetween_count(1) == 1
    assert print_inbetween_count(2, busy=True) == 0
    assert print_inbetween_count(2, queued=1) == 0
    assert print_inbetween_count(1, last_interp_s=0.08, gen_fps=10.0) == 0
    assert print_inbetween_count(1, last_interp_s=0.02, gen_fps=10.0) == 1


def test_inbetween_image_moves_box() -> None:
    left = np.zeros((48, 48, 3), dtype=np.uint8)
    right = np.zeros((48, 48, 3), dtype=np.uint8)
    left[20:28, 4:12] = 255
    right[20:28, 36:44] = 255
    mid = np.asarray(inbetween_image(Image.fromarray(left), Image.fromarray(right), 0.5))

    def _col_mass(arr: np.ndarray) -> float:
        cols = np.where(arr.sum(axis=(0, 2)) > 0)[0]
        return float(cols.mean()) if cols.size else -1.0

    assert _col_mass(left) < _col_mass(mid) < _col_mass(right)


def test_inbetween_frames_count() -> None:
    a = Image.new("RGB", (16, 16), (10, 10, 10))
    b = Image.new("RGB", (16, 16), (200, 200, 200))
    frames = inbetween_frames(a, b, 2)
    assert len(frames) == 2
    assert frames[0][0] < frames[1][0]
    first = int(np.asarray(frames[0][1]).mean())
    second = int(np.asarray(frames[1][1]).mean())
    assert first < second


def test_playout_holds_then_caps_at_20() -> None:
    wait, nxt = playout_gap(10.0, 0.0)
    assert wait == 0.25
    assert abs(nxt - 10.30) < 1e-9
    # Ready just after the first picture: wait out the rest of the 50 ms slot.
    wait, nxt = playout_gap(10.26, nxt)
    assert abs(wait - 0.04) < 1e-9
    # Late frame shows now. The one after it is a full slot later, not a burst.
    wait, nxt = playout_gap(11.0, 10.5)
    assert wait == 0.0
    assert nxt == 11.05
