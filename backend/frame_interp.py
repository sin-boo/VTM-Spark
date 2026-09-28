"""In-between frames between two generated pictures (print / inbetweening).

DiT still draws the keys. This fills the gaps with optical-flow warps so the
stream looks like more FPS without stretching the last still toward live pose.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

# Small on purpose — Farneback at 384 with two passes was stealing the GPU
# (OpenCL) and dropping generated keys from ~10 to ~6.
_FLOW_SIZE = 192


def lerp_stream_pose(prev: np.ndarray, current: np.ndarray, t: float) -> np.ndarray:
    """Lerp visible keypoints; keep a point parked if only one side can see it."""
    a = np.asarray(prev, dtype=np.float32)
    b = np.asarray(current, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Pose shapes must match, got {a.shape} vs {b.shape}")
    amount = float(min(max(t, 0.0), 1.0))
    out = a.copy()
    vis_a = a[..., 3] >= 0.5
    vis_b = b[..., 3] >= 0.5
    both = vis_a & vis_b
    out[both] = (1.0 - amount) * a[both] + amount * b[both]
    only_b = (~vis_a) & vis_b
    out[only_b] = b[only_b]
    out[..., 3] = np.maximum(a[..., 3], b[..., 3])
    return out


def lerp_stream_hair(
    prev: np.ndarray | None, current: np.ndarray, t: float
) -> np.ndarray:
    cur = np.asarray(current, dtype=np.float32)
    if prev is None:
        return cur
    old = np.asarray(prev, dtype=np.float32)
    if old.shape != cur.shape:
        return cur
    amount = float(min(max(t, 0.0), 1.0))
    return (1.0 - amount) * old + amount * cur


def blend_images(prev: Image.Image, current: Image.Image, t: float) -> Image.Image:
    """Crossfade two RGB frames. ``t=1`` is the new picture."""
    amount = float(min(max(t, 0.0), 1.0))
    a = np.asarray(prev.convert("RGB"), dtype=np.float32)
    b = np.asarray(current.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        return current
    out = (1.0 - amount) * a + amount * b
    return Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8), mode="RGB")


def inbetween_ts(count: int) -> list[float]:
    """Fractions strictly between 0 and 1. ``count=1`` → ``[0.5]``."""
    n = max(0, int(count))
    if n <= 0:
        return []
    return [float(i) / float(n + 1) for i in range(1, n + 1)]


def inbetween_slot_s(gen_fps: float, count: int) -> float:
    """Legacy hold. Always 0 — sleeping here drops generated keys."""
    del gen_fps, count
    return 0.0


# Shown pictures, not generated ones. 20 fps is a frame every 50 ms.
SHOW_FPS_MAX = 20.0
# Let a short burst sit before the first picture so it does not all hit at once.
PLAYOUT_DELAY_S = 0.25
# Quarter-second of 20 fps pictures. Newer keys replace anything older.
PLAYOUT_QUEUE_MAX = 5


def playout_gap(
    now: float,
    next_at: float,
    *,
    fps_max: float = SHOW_FPS_MAX,
    delay_s: float = PLAYOUT_DELAY_S,
) -> tuple[float, float]:
    """Seconds to wait before the next shown picture, and the deadline after it.

    ``next_at <= 0`` is the start of a stream: hold ``delay_s``, then show.
    A late clock shows immediately. The next picture is one slot later, so a
    backlog is not dumped in one burst.
    """
    slot = 1.0 / max(1.0, float(fps_max))
    if next_at <= 0.0:
        return float(delay_s), float(now) + float(delay_s) + slot
    if next_at <= now:
        return 0.0, float(now) + slot
    return float(next_at) - float(now), float(next_at) + slot


# Pictures are spaced over this share of the key interval, so the next key
# lands in a short idle instead of behind a backlog (which skips its mids).
PACE_MARGIN = 0.95
# A mid may run a touch under the 20 fps slot (~22 fps) rather than drop out
# the moment the measured key rate jitters above 10.
SLOT_SLACK = 0.9
# A mid has to render in this share of its own slot or it would show late.
RENDER_SHARE = 0.9


def inbetween_pacing(
    gen_fps: float,
    count: int,
    *,
    fps_max: float = SHOW_FPS_MAX,
    mid_cost_s: float = 0.0,
) -> tuple[int, float]:
    """``(mids per key gap, seconds between shown pictures)`` at ``gen_fps`` keys/s.

    Mids and keys share the gap between two keys evenly instead of firing
    every 50 ms and then waiting: at 6 keys/s one mid gives ~80 ms steps, not
    50 ms then 117 ms. Mids that would not fit under ``fps_max``, or that take
    too long to render, are dropped so the display never backs up and bursts.
    Unknown rate (first key): keys only.
    """
    n = max(0, int(count))
    slot = 1.0 / max(1.0, float(fps_max))
    fps = float(gen_fps)
    if fps <= 0.0:
        return 0, slot
    span = PACE_MARGIN / fps
    fit = int(span / (slot * SLOT_SLACK) + 1e-6) - 1
    n = max(0, min(n, fit))
    cost = max(0.0, float(mid_cost_s))
    while n > 0 and cost > RENDER_SHARE * span / float(n + 1):
        n -= 1
    return n, max(slot * SLOT_SLACK, span / float(n + 1))


def _cv2():
    import cv2

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    return cv2


_GRIDS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    key = (int(height), int(width))
    grid = _GRIDS.get(key)
    if grid is None:
        grid = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        _GRIDS.clear()
        _GRIDS[key] = grid
    return grid


def _warp_rgb(image: np.ndarray, flow: np.ndarray, amount: float) -> np.ndarray:
    """Move ``image`` ``amount`` of the way along ``flow`` (prev → next).

    ``remap`` samples backwards: the pixel that lands at ``x`` comes from
    ``x - amount * flow``. Adding the flow instead sent every moving part the
    wrong way, so each in-between showed the old and the new position at once.
    """
    cv2 = _cv2()

    height, width = image.shape[:2]
    grid_x, grid_y = _pixel_grid(height, width)
    map_x = (grid_x - flow[..., 0] * float(amount)).astype(np.float32)
    map_y = (grid_y - flow[..., 1] * float(amount)).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _flow_forward(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    cv2 = _cv2()

    height, width = prev.shape[:2]
    scale = float(_FLOW_SIZE) / float(max(height, width))
    if scale < 1.0:
        small_w = max(16, int(round(width * scale)))
        small_h = max(16, int(round(height * scale)))
        a = cv2.resize(prev, (small_w, small_h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(nxt, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        a, b = prev, nxt
        small_w, small_h = width, height
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    # One cheap pass. A second backward pass plus OpenCL was cutting gen FPS.
    # Three pyramid levels follow a fast head turn (~60 px at 768); two lost it.
    fwd = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 11, 2, 5, 1.1, 0)
    if (small_w, small_h) != (width, height):
        fwd = cv2.resize(fwd, (width, height), interpolation=cv2.INTER_LINEAR)
        fwd[..., 0] *= float(width) / float(small_w)
        fwd[..., 1] *= float(height) / float(small_h)
    return fwd


def _mix_warp(prev: np.ndarray, nxt: np.ndarray, flow: np.ndarray, t: float) -> Image.Image:
    """Both keys meet at ``t``: prev moves forward by ``t``, next back by ``1-t``.

    Warping only prev and fading in a still next left a faint double image on
    anything that moved.
    """
    cv2 = _cv2()

    left = _warp_rgb(prev, flow, t)
    right = _warp_rgb(nxt, flow, t - 1.0)
    # addWeighted: ~1 ms at 768², a float32 numpy mix was ~15 ms per mid.
    out = cv2.addWeighted(left, 1.0 - float(t), right, float(t), 0.0)
    return Image.fromarray(out, mode="RGB")


def inbetween_image(prev: Image.Image, current: Image.Image, t: float) -> Image.Image:
    """One in-between: warp the last key toward ``t`` and crossfade."""
    amount = float(min(max(t, 0.0), 1.0))
    if amount <= 1e-4:
        return prev
    if amount >= 1.0 - 1e-4:
        return current
    a = np.asarray(prev.convert("RGB"))
    b = np.asarray(current.convert("RGB"))
    if a.shape != b.shape:
        return blend_images(prev, current, amount)
    try:
        return _mix_warp(a, b, _flow_forward(a, b), amount)
    except Exception:
        return blend_images(prev, current, amount)


def inbetween_maker(
    prev: Image.Image, current: Image.Image
) -> Callable[[float], Image.Image]:
    """``make(t)`` for pictures between two keys. Flow is worked out once, on
    the first call, so each mid can be drawn just before it is due."""
    a = np.asarray(prev.convert("RGB"))
    b = np.asarray(current.convert("RGB"))
    if a.shape != b.shape:
        return lambda t: blend_images(prev, current, t)
    flow: list[np.ndarray | None] = []

    def make(t: float) -> Image.Image:
        if not flow:
            try:
                flow.append(_flow_forward(a, b))
            except Exception:
                flow.append(None)
        if flow[0] is None:
            return blend_images(prev, current, t)
        return _mix_warp(a, b, flow[0], t)

    return make


def inbetween_frames(
    prev: Image.Image, current: Image.Image, count: int
) -> list[tuple[float, Image.Image]]:
    """``count`` pictures strictly between ``prev`` and ``current``."""
    amounts = inbetween_ts(count)
    if not amounts:
        return []
    make = inbetween_maker(prev, current)
    return [(t, make(t)) for t in amounts]
