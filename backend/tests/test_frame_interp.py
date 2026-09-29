import numpy as np
from PIL import Image

from backend.engine import neutral_keypoints
from backend.frame_interp import (
    TWEEN_MAX_S,
    blend_images,
    ema_blend,
    inbetween_frames,
    inbetween_image,
    inbetween_pacing,
    inbetween_slot_s,
    inbetween_ts,
    lerp_stream_pose,
    inbetween_maker,
    playout_gap,
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
    # 6 keys/s, one mid: two even ~80 ms steps, not 50 ms then 117 ms.
    n, gap = inbetween_pacing(6.0, 1)
    assert n == 1
    assert abs(gap - 0.95 / 12.0) < 1e-6
    # 10 keys/s cannot fit three mids under 20 fps: only one survives.
    n, gap = inbetween_pacing(10.0, 3)
    assert n == 1
    assert abs(gap - 0.0475) < 1e-6
    # 20 keys/s: no room for mids at all.
    assert inbetween_pacing(20.0, 2)[0] == 0
    # Unknown rate (first key): keys only.
    assert inbetween_pacing(0.0, 2) == (0, 0.05)


def test_inbetween_pacing_survives_rate_jitter() -> None:
    """Auto holds keys at 10/s; a measured 10.3 must not drop the mid.

    The old exact fit flipped to zero mids on any jitter above 10 keys/s.
    """
    assert inbetween_pacing(10.3, 1)[0] == 1
    assert inbetween_pacing(9.7, 1)[0] == 1


def test_inbetween_pacing_spreads_keys_without_mids() -> None:
    # Batch×2 keys with no mids: one per key interval, not 50 ms apart.
    n, gap = inbetween_pacing(2.0, 0)
    assert n == 0
    assert abs(gap - 0.475) < 1e-6


def test_inbetween_pacing_drops_mids_that_render_too_slow() -> None:
    assert inbetween_pacing(10.0, 1, mid_cost_s=0.02)[0] == 1
    assert inbetween_pacing(10.0, 1, mid_cost_s=0.06)[0] == 0
    # Slow keys leave time: 3 mids at 2 keys/s even at 60 ms each.
    assert inbetween_pacing(2.0, 3, mid_cost_s=0.06)[0] == 3


def test_capped_tween_does_not_hold_a_slow_key_back() -> None:
    """Mids sit before their key. Spread over a 3 keys/s gap they showed the
    key ~0.3 s after it was drawn; capped, the tween runs at display rate."""
    n, gap = inbetween_pacing(3.0, 3)
    assert (n + 1) * gap > 0.3
    n, gap = inbetween_pacing(3.0, 3, span_max=TWEEN_MAX_S)
    assert n >= 1
    assert (n + 1) * gap <= TWEEN_MAX_S + 1e-9
    assert gap >= 0.045 - 1e-9
    # Fast keys are under the cap already: nothing changes.
    assert inbetween_pacing(10.0, 1, span_max=TWEEN_MAX_S) == inbetween_pacing(10.0, 1)
    # Never tighter than one display slot.
    assert inbetween_pacing(1.0, 0, span_max=0.001)[1] == 0.05


def test_ema_blend_mixes_uint8_frames() -> None:
    new = np.full((4, 4, 3), 200, dtype=np.uint8)
    held = np.zeros((4, 4, 3), dtype=np.uint8)
    out = ema_blend(new, held, 0.5)
    assert out.dtype == np.uint8
    assert int(out.mean()) == 100
    assert np.array_equal(ema_blend(new, held, 1.0), new)


def test_inbetween_maker_matches_frames() -> None:
    a = Image.new("RGB", (16, 16), (10, 10, 10))
    b = Image.new("RGB", (16, 16), (200, 200, 200))
    make = inbetween_maker(a, b)
    lazy = [np.asarray(make(t)) for t in (1.0 / 3.0, 2.0 / 3.0)]
    eager = [np.asarray(img) for _, img in inbetween_frames(a, b, 2)]
    for x, y in zip(lazy, eager):
        np.testing.assert_array_equal(x, y)


def test_inbetween_frames_count() -> None:
    a = Image.new("RGB", (16, 16), (10, 10, 10))
    b = Image.new("RGB", (16, 16), (200, 200, 200))
    frames = inbetween_frames(a, b, 2)
    assert len(frames) == 2
    assert frames[0][0] < frames[1][0]
    first = int(np.asarray(frames[0][1]).mean())
    second = int(np.asarray(frames[1][1]).mean())
    assert first < second


def test_playout_starts_at_once_then_caps_at_20() -> None:
    # No start hold: a quarter-second head start never drained (standing lag).
    wait, nxt = playout_gap(10.0, 0.0)
    assert wait == 0.0
    assert abs(nxt - 10.05) < 1e-9
    # Ready just after the first picture: wait out the rest of the 50 ms slot.
    wait, nxt = playout_gap(10.01, nxt)
    assert abs(wait - 0.04) < 1e-9
    # Late frame shows now. The one after it is a full slot later, not a burst.
    wait, nxt = playout_gap(11.0, 10.5)
    assert wait == 0.0
    assert nxt == 11.05
