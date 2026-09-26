"""In-between frames between two generated pictures (print / inbetweening).

DiT still draws the keys. This fills the gaps with optical-flow warps so the
stream looks like more FPS without stretching the last still toward live pose.
"""

from __future__ import annotations

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


def print_inbetween_count(
    wanted: int,
    *,
    busy: bool = False,
    queued: int = 0,
    last_interp_s: float = 0.0,
    gen_fps: float = 0.0,
) -> int:
    """How many mids to print. 0 if a mid would make us skip a generated key."""
    n = max(0, int(wanted))
    if n <= 0 or busy or int(queued) > 0:
        return 0
    fps = float(gen_fps)
    elapsed = float(last_interp_s)
    if elapsed > 0.0 and fps > 1.0 and elapsed > (0.45 / fps):
        return 0
    return n


def _cv2():
    import cv2

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    return cv2


def _warp_rgb(image: np.ndarray, flow: np.ndarray, amount: float) -> np.ndarray:
    cv2 = _cv2()

    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = (grid_x + flow[..., 0] * float(amount)).astype(np.float32)
    map_y = (grid_y + flow[..., 1] * float(amount)).astype(np.float32)
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
    fwd = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 2, 11, 2, 5, 1.1, 0)
    if (small_w, small_h) != (width, height):
        fwd = cv2.resize(fwd, (width, height), interpolation=cv2.INTER_LINEAR)
        fwd[..., 0] *= float(width) / float(small_w)
        fwd[..., 1] *= float(height) / float(small_h)
    return fwd


def _mix_warp(prev: np.ndarray, nxt: np.ndarray, flow: np.ndarray, t: float) -> Image.Image:
    left = _warp_rgb(prev, flow, t)
    out = (1.0 - t) * left.astype(np.float32) + t * nxt.astype(np.float32)
    return Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8), mode="RGB")


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


def inbetween_frames(
    prev: Image.Image, current: Image.Image, count: int
) -> list[tuple[float, Image.Image]]:
    """``count`` pictures strictly between ``prev`` and ``current``."""
    amounts = inbetween_ts(count)
    if not amounts:
        return []
    a = np.asarray(prev.convert("RGB"))
    b = np.asarray(current.convert("RGB"))
    if a.shape != b.shape:
        return [(t, blend_images(prev, current, t)) for t in amounts]
    try:
        flow = _flow_forward(a, b)
    except Exception:
        return [(t, blend_images(prev, current, t)) for t in amounts]
    return [(t, _mix_warp(a, b, flow, t)) for t in amounts]
