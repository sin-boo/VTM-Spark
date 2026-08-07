"""Rasterize keypoints into multi-channel pose maps at latent resolution.

Channels (see utils.keypoints.POSE_CHANNEL_GROUPS):
  0 face_outline, 1 eyebrow, 2 eye, 3 iris, 4 nose, 5 mouth,
  6 skeleton (bones as lines), 7 joints (body joint gaussians).

Input keypoints are expected in normalized [-1, 1] crop space.
Output is (B, NUM_POSE_CHANNELS, H, W) float32 in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.keypoints import (
    BODY_BONE_INDICES,
    KEYPOINT_DIM,
    NUM_KEYPOINTS,
    NUM_POSE_CHANNELS,
    POSE_CHANNEL_GROUPS,
    group_indices,
)


def _norm_xy_to_pixel(xy: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Map [-1, 1] coords to pixel coords in [0, W-1] / [0, H-1]. xy: (..., 2)."""
    x = (xy[..., 0] + 1.0) * 0.5 * (w - 1)
    y = (xy[..., 1] + 1.0) * 0.5 * (h - 1)
    return torch.stack([x, y], dim=-1)


def _splat_gaussians(
    canvas: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    weights: torch.Tensor,
    sigma: float,
) -> None:
    """Add isotropic gaussians onto canvas (B, H, W) in-place.

    xs, ys, weights: (B, K)
    """
    bsz, h, w = canvas.shape
    device = canvas.device
    yy = torch.arange(h, device=device, dtype=canvas.dtype).view(1, h, 1)
    xx = torch.arange(w, device=device, dtype=canvas.dtype).view(1, 1, w)
    # Loop over points — K is small (~37); vectorizing all K at once can OOM on large H*W.
    k = xs.shape[1]
    inv = 1.0 / (2.0 * sigma * sigma + 1e-8)
    for i in range(k):
        w_i = weights[:, i]
        if float(w_i.abs().max().item()) < 1e-8:
            continue
        dx = xx - xs[:, i].view(bsz, 1, 1)
        dy = yy - ys[:, i].view(bsz, 1, 1)
        g = torch.exp(-(dx * dx + dy * dy) * inv) * w_i.view(bsz, 1, 1)
        canvas.add_(g)


def _draw_bones(
    canvas: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    visible: torch.Tensor,
    thickness: float = 1.0,
) -> None:
    """Draw skeleton bones as soft lines onto canvas (B, H, W).

    xs, ys, visible: (B, NUM_KEYPOINTS)
    """
    bsz, h, w = canvas.shape
    device = canvas.device
    yy = torch.arange(h, device=device, dtype=canvas.dtype).view(1, h, 1)
    xx = torch.arange(w, device=device, dtype=canvas.dtype).view(1, 1, w)
    for ia, ib in BODY_BONE_INDICES:
        xa = xs[:, ia]
        ya = ys[:, ia]
        xb = xs[:, ib]
        yb = ys[:, ib]
        both = (visible[:, ia] > 0.5) & (visible[:, ib] > 0.5)
        if not bool(both.any()):
            continue
        # Distance from each pixel to the segment.
        # Parametrize: p = a + t (b-a), t in [0,1]
        dx = xb - xa
        dy = yb - ya
        len2 = dx * dx + dy * dy
        # Avoid zero-length
        len2 = torch.clamp(len2, min=1e-4)
        # t = ((p-a)·(b-a)) / |b-a|^2
        t = ((xx - xa.view(bsz, 1, 1)) * dx.view(bsz, 1, 1) + (yy - ya.view(bsz, 1, 1)) * dy.view(bsz, 1, 1)) / len2.view(
            bsz, 1, 1
        )
        t = t.clamp(0.0, 1.0)
        px = xa.view(bsz, 1, 1) + t * dx.view(bsz, 1, 1)
        py = ya.view(bsz, 1, 1) + t * dy.view(bsz, 1, 1)
        dist2 = (xx - px) ** 2 + (yy - py) ** 2
        line = torch.exp(-dist2 / (2.0 * thickness * thickness + 1e-8))
        line = line * both.view(bsz, 1, 1).to(canvas.dtype)
        canvas.add_(line)


@torch.no_grad()
def rasterize_pose_maps(
    keypoints: torch.Tensor,
    height: int,
    width: int,
    *,
    sigma: float = 1.5,
    bone_thickness: float = 1.0,
    clamp: bool = True,
) -> torch.Tensor:
    """Rasterize a batch of keypoints to pose maps.

    Args:
        keypoints: (B, NUM_KEYPOINTS, 4) with (x, y, score, visible) in [-1,1] xy.
        height, width: output spatial size (typically latent H/W).
        sigma: gaussian std in pixels for point splats.
        bone_thickness: soft-line thickness for skeleton channel.
        clamp: if True, clamp output to [0, 1].

    Returns:
        (B, NUM_POSE_CHANNELS, height, width) float tensor.
    """
    if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(
            f"Expected keypoints (B, {NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {tuple(keypoints.shape)}"
        )
    bsz = keypoints.shape[0]
    device = keypoints.device
    dtype = torch.float32
    kps = keypoints.to(device=device, dtype=dtype)
    xy = _norm_xy_to_pixel(kps[..., :2], height, width)
    xs = xy[..., 0]
    ys = xy[..., 1]
    scores = kps[..., 2].clamp(min=0.0)
    visible = kps[..., 3]
    weights = scores * (visible > 0.5).to(dtype)

    out = torch.zeros((bsz, NUM_POSE_CHANNELS, height, width), device=device, dtype=dtype)

    for ch, group in enumerate(POSE_CHANNEL_GROUPS):
        if group == "skeleton":
            _draw_bones(out[:, ch], xs, ys, visible, thickness=bone_thickness)
            continue
        idxs = group_indices(group)
        if not idxs:
            continue
        idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
        _splat_gaussians(
            out[:, ch],
            xs.index_select(1, idx_t),
            ys.index_select(1, idx_t),
            weights.index_select(1, idx_t),
            sigma=sigma,
        )

    if clamp:
        out = out.clamp(0.0, 1.0)
    return out


def pose_map_channels() -> int:
    return NUM_POSE_CHANNELS


# Face + iris slots (indices 0..29) for loss weighting / face crop.
FACE_POINT_SLICE = slice(0, 30)


@torch.no_grad()
def face_weight_mask(
    keypoints: torch.Tensor,
    height: int,
    width: int,
    weight: float = 4.0,
    margin: float = 0.15,
    feather: float = 0.05,
) -> torch.Tensor:
    """Soft spatial weight map emphasizing the face region.

    Args:
        keypoints: (B, NUM_KEYPOINTS, 4) in normalized [-1, 1] crop space.
        height, width: latent spatial size.
        weight: multiplier inside the face bbox (1.0 = no-op).
        margin: fractional pad of the face bbox (relative to image size).
        feather: soft edge width as a fraction of image size.

    Returns:
        (B, 1, height, width) float tensor with 1.0 outside and ``weight`` inside.
    """
    if abs(float(weight) - 1.0) < 1e-6:
        return torch.ones(
            (keypoints.shape[0], 1, height, width),
            device=keypoints.device,
            dtype=torch.float32,
        )
    if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(
            f"Expected keypoints (B, {NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {tuple(keypoints.shape)}"
        )
    bsz = keypoints.shape[0]
    device = keypoints.device
    dtype = torch.float32
    face = keypoints[:, FACE_POINT_SLICE].to(device=device, dtype=dtype)
    vis = face[..., 3] > 0.5
    xy = face[..., :2]

    # Default full-image soft weight when no face points are visible.
    yy = torch.arange(height, device=device, dtype=dtype).view(1, height, 1).expand(bsz, height, width)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, width).expand(bsz, height, width)
    # Pixel coords of face points
    pix = _norm_xy_to_pixel(xy, height, width)  # (B, K, 2)
    # Mask invisible points with large/small sentinels for min/max
    big = float(max(height, width) + 1)
    pix_x = pix[..., 0].masked_fill(~vis, big)
    pix_y = pix[..., 1].masked_fill(~vis, big)
    x0 = pix_x.min(dim=1).values.clamp(0, width - 1)
    y0 = pix_y.min(dim=1).values.clamp(0, height - 1)
    pix_x_hi = pix[..., 0].masked_fill(~vis, -1.0)
    pix_y_hi = pix[..., 1].masked_fill(~vis, -1.0)
    x1 = pix_x_hi.max(dim=1).values.clamp(0, width - 1)
    y1 = pix_y_hi.max(dim=1).values.clamp(0, height - 1)
    # Fallback center box when all invisible (handled below via has_face)
    x0 = torch.where(vis.any(dim=1), x0, torch.full_like(x0, width * 0.25))
    x1 = torch.where(vis.any(dim=1), x1, torch.full_like(x1, width * 0.75))
    y0 = torch.where(vis.any(dim=1), y0, torch.full_like(y0, height * 0.15))
    y1 = torch.where(vis.any(dim=1), y1, torch.full_like(y1, height * 0.55))

    pad_x = margin * width
    pad_y = margin * height
    x0 = (x0 - pad_x).clamp(0, width - 1)
    x1 = (x1 + pad_x).clamp(0, width - 1)
    y0 = (y0 - pad_y).clamp(0, height - 1)
    y1 = (y1 + pad_y).clamp(0, height - 1)

    # Soft box via distance to edge (feather).
    feather_px = max(feather * min(height, width), 1.0)
    # Signed distance: positive inside
    dx = torch.minimum(xx - x0.view(bsz, 1, 1), x1.view(bsz, 1, 1) - xx)
    dy = torch.minimum(yy - y0.view(bsz, 1, 1), y1.view(bsz, 1, 1) - yy)
    dist_in = torch.minimum(dx, dy)
    soft = (dist_in / feather_px).clamp(0.0, 1.0)
    # Outside gets soft=0, deep inside soft=1
    mask = 1.0 + (float(weight) - 1.0) * soft
    # If no visible face points, keep weight=1 (don't boost random center)
    has_face = vis.any(dim=1).to(dtype).view(bsz, 1, 1)
    mask = torch.where(has_face > 0.5, mask, torch.ones_like(mask))
    return mask.unsqueeze(1)


@torch.no_grad()
def crop_face_latent(
    ref_latents: torch.Tensor,
    keypoints_ref: torch.Tensor,
    out_size: int = 32,
    margin: float = 0.25,
) -> torch.Tensor:
    """Crop a square face region from each ref latent and resize to ``out_size``.

    Args:
        ref_latents: (B, C, H, W) VAE latents.
        keypoints_ref: (B, NUM_KEYPOINTS, 4) in normalized [-1, 1] crop space.
        out_size: output spatial size (default 32 → 16x16 tokens at patch 2).
        margin: pad fraction around face bbox.

    Returns:
        (B, C, out_size, out_size) face-cropped latents.
        Falls back to center crop when no face points are visible.
    """
    if ref_latents.ndim != 4:
        raise ValueError(f"Expected ref_latents (B,C,H,W), got {tuple(ref_latents.shape)}")
    bsz, c, h, w = ref_latents.shape
    device = ref_latents.device
    dtype = ref_latents.dtype
    face = keypoints_ref[:, FACE_POINT_SLICE].to(device=device, dtype=torch.float32)
    vis = face[..., 3] > 0.5
    pix = _norm_xy_to_pixel(face[..., :2], h, w)
    big = float(max(h, w) + 1)
    pix_x = pix[..., 0].masked_fill(~vis, big)
    pix_y = pix[..., 1].masked_fill(~vis, big)
    x0 = pix_x.min(dim=1).values
    y0 = pix_y.min(dim=1).values
    pix_x_hi = pix[..., 0].masked_fill(~vis, -1.0)
    pix_y_hi = pix[..., 1].masked_fill(~vis, -1.0)
    x1 = pix_x_hi.max(dim=1).values
    y1 = pix_y_hi.max(dim=1).values
    has_face = vis.any(dim=1)
    x0 = torch.where(has_face, x0, torch.full_like(x0, w * 0.25))
    x1 = torch.where(has_face, x1, torch.full_like(x1, w * 0.75))
    y0 = torch.where(has_face, y0, torch.full_like(y0, h * 0.15))
    y1 = torch.where(has_face, y1, torch.full_like(y1, h * 0.55))

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = torch.maximum(x1 - x0, y1 - y0)
    side = side * (1.0 + 2.0 * margin)
    side = torch.clamp(side, min=4.0)
    # Center-crop fallback
    fallback = float(min(h, w)) * 0.5
    cx = torch.where(has_face, cx, torch.full_like(cx, 0.5 * (w - 1)))
    cy = torch.where(has_face, cy, torch.full_like(cy, 0.5 * (h - 1)))
    side = torch.where(has_face, side, torch.full_like(side, fallback))

    half = 0.5 * side
    x0i = (cx - half).clamp(0, w - 1)
    y0i = (cy - half).clamp(0, h - 1)
    x1i = (cx + half).clamp(1, w)
    y1i = (cy + half).clamp(1, h)
    # Keep square within bounds
    side_i = torch.minimum(x1i - x0i, y1i - y0i).clamp(min=1.0)
    x1i = x0i + side_i
    y1i = y0i + side_i
    # Shift if past edge
    x_shift = torch.clamp(x1i - w, min=0)
    y_shift = torch.clamp(y1i - h, min=0)
    x0i = x0i - x_shift
    x1i = x1i - x_shift
    y0i = y0i - y_shift
    y1i = y1i - y_shift

    crops = []
    for i in range(bsz):
        xa = int(x0i[i].floor().item())
        ya = int(y0i[i].floor().item())
        xb = int(x1i[i].ceil().item())
        yb = int(y1i[i].ceil().item())
        xa = max(0, min(xa, w - 1))
        ya = max(0, min(ya, h - 1))
        xb = max(xa + 1, min(xb, w))
        yb = max(ya + 1, min(yb, h))
        patch = ref_latents[i : i + 1, :, ya:yb, xa:xb]
        patch = F.interpolate(patch.float(), size=(out_size, out_size), mode="bilinear", align_corners=False)
        crops.append(patch.to(dtype=dtype))
    return torch.cat(crops, dim=0)
