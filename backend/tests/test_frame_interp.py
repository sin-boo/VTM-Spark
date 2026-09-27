import numpy as np
from PIL import Image

from backend.engine import neutral_keypoints
from backend.frame_interp import (
    blend_images,
    inbetween_frames,
    inbetween_image,
    inbetween_pacing,
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


def _blob(cx: float, size: int = 192) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    g = np.exp(-((xx - cx) ** 2 + (yy - size / 2) ** 2) / (2 * 12.0**2)) * 255.0
    return np.dstack([g, g, g]).astype(np.uint8)


def _centre_x(arr: np.ndarray) -> float:
    g = arr[..., 0].astype(np.float32)
    xx = np.arange(arr.shape[1], dtype=np.float32)[None, :]
    return float((g * xx).sum() / g.sum())


def _spread_x(arr: np.ndarray) -> float:
    g = arr[..., 0].astype(np.float32)
    xx = np.arange(arr.shape[1], dtype=np.float32)[None, :]
    c = _centre_x(arr)
    return float(np.sqrt((g * (xx - c) ** 2).sum() / g.sum()))


def test_inbetween_image_lands_halfway() -> None:
    """The mid must sit at the midpoint, not backwards or as two ghosts.

    Sampling ``x + flow`` moved the old key the wrong way; a plain crossfade
    would also pass a loose "somewhere between" check.
    """
    left, right = _blob(80), _blob(100)
    mid = np.asarray(inbetween_image(Image.fromarray(left), Image.fromarray(right), 0.5))
    assert abs(_centre_x(mid) - 90.0) < 2.0
    # One blob, not the old and new positions faded together.
    fade = np.asarray(blend_images(Image.fromarray(left), Image.fromarray(right), 0.5))
    # Fade: ~15.3 px wide; warped mid: ~13.1; the key itself: ~11.8.
    assert _spread_x(mid) < _spread_x(fade) * 0.9
    assert _spread_x(mid) < _spread_x(left) * 1.2


def test_inbetween_pacing_spreads_evenly() -> None:
    # 6 keys/s, one mid: two 83 ms steps, not 50 ms then 117 ms.
    n, gap = inbetween_pacing(6.0, 1)
    assert n == 1
    assert abs(gap - 1.0 / 12.0) < 1e-6
    # 10 keys/s cannot fit three mids under 20 fps: only one survives.
    n, gap = inbetween_pacing(10.0, 3)
    assert n == 1
    assert abs(gap - 0.05) < 1e-6
    # 20 keys/s: no room for mids at all.
    assert inbetween_pacing(20.0, 2)[0] == 0
    # Unknown rate: keep the count, fall back to the 20 fps slot.
    assert inbetween_pacing(0.0, 2) == (2, 0.05)


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
