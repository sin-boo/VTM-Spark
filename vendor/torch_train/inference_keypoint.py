"""Keypoint-conditioning inference with dual CFG (pose + identity).

Batched three-way guidance per step::

    v = v_uncond + s_pose * (v_pose - v_uncond) + s_id * (v_id - v_uncond)

where:
  - v_pose: pose on, ref null
  - v_id:   ref on, pose null
  - v_uncond: both null
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.dit import DualStreamDiTConfig, i1DiT
from utils.keypoints import KEYPOINT_DIM, NUM_KEYPOINTS, NUM_POSE_CHANNELS, keypoint_deltas
from utils.pose_map import crop_face_latent, rasterize_pose_maps
from vae.vae import encode_images_to_latents, load_vae, reverse_scale_latents, scale_latents


def is_keypoint_checkpoint(checkpoint: dict[str, Any] | Path | str) -> bool:
    if isinstance(checkpoint, (str, Path)):
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    else:
        ckpt = checkpoint
    cfg = ckpt.get("config") or {}
    return bool(cfg.get("use_keypoint_conditioning"))


def build_keypoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[i1DiT, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["config"])
    use_ref_face = bool(cfg_dict.get("use_ref_face_tokens", False))
    ds_cfg = DualStreamDiTConfig(
        input_size=int(cfg_dict["input_size"]),
        image_resolution=int(cfg_dict.get("image_resolution", 768)),
        patch_size=int(cfg_dict.get("patch_size", 2)),
        in_channels=int(cfg_dict["in_channels"]),
        hidden_size=int(cfg_dict["hidden_size"]),
        depth=int(cfg_dict["depth"]),
        num_heads=int(cfg_dict["num_heads"]),
        mlp_ratio=float(cfg_dict.get("mlp_ratio", 4.0)),
        text_embed_dim=int(cfg_dict.get("text_embed_dim", cfg_dict["hidden_size"])),
        text_num_tokens=int(cfg_dict.get("text_num_tokens", cfg_dict.get("num_keypoints", NUM_KEYPOINTS))),
        use_numeric_conditioning=False,
        use_keypoint_conditioning=True,
        num_keypoints=int(cfg_dict.get("num_keypoints", NUM_KEYPOINTS)),
        num_pose_channels=int(cfg_dict.get("num_pose_channels", NUM_POSE_CHANNELS)),
        use_adaln=bool(cfg_dict.get("use_adaln", True)),
        use_ref_conditioning=True,
        use_ref_tokens=bool(cfg_dict.get("use_ref_tokens", True)),
        ref_spatial_downsample=int(cfg_dict.get("ref_spatial_downsample", 2)),
        use_ref_face_tokens=use_ref_face,
        ref_face_size=int(cfg_dict.get("ref_face_size", 32)),
        use_keypoint_rope=bool(cfg_dict.get("use_keypoint_rope", True)),
        drop_ref_prob=0.0,
        drop_pose_prob=0.0,
        drop_text_prob=0.0,
        use_long_skip=bool(cfg_dict.get("use_long_skip", True)),
        use_sandwich_norm=bool(cfg_dict.get("use_sandwich_norm", True)),
        use_qknorm=bool(cfg_dict.get("use_qknorm", True)),
        use_swiglu=bool(cfg_dict.get("use_swiglu", True)),
        use_rmsnorm=bool(cfg_dict.get("use_rmsnorm", True)),
        rope_theta=float(cfg_dict.get("rope_theta", 10000.0)),
        use_repa=False,
    )
    model = i1DiT(ds_cfg).to(device=device, dtype=dtype).eval()
    state = ckpt.get("ema") or ckpt.get("model")
    if isinstance(state, dict) and "shadow" in state:
        state = state["shadow"]
    model.load_state_dict(state, strict=False)
    model = model.to(device=device, dtype=dtype).eval()
    return model, cfg_dict


def _timestep_schedule(
    num_steps: int,
    device: torch.device,
    inference_timestep_shift: float,
) -> torch.Tensor:
    times = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32, device=device)
    if inference_timestep_shift != 0.0:
        shift = inference_timestep_shift
        times = (shift * times) / (1.0 + (shift - 1.0) * times)
    return times


def flow_start_from_last(
    noise: torch.Tensor,
    times: torch.Tensor,
    last_latent: torch.Tensor | None,
    start_t: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Start a flow at ``start_t`` mixed toward the previous frame latent.

    ``times`` is the full ``[0, …, 1]`` schedule (noise → image). ``start_t``
    of 0 keeps today's full denoise. Values in (0, 1) mix
    ``(1-t) * noise + t * last`` and drop schedule points before ``t``.
    """
    if last_latent is None:
        return noise, times
    t0 = float(start_t)
    if t0 <= 0.0:
        return noise, times
    t0 = min(max(t0, 0.0), 1.0 - 1e-4)
    last = last_latent.to(device=noise.device, dtype=noise.dtype)
    if last.shape[-3:] != noise.shape[-3:]:
        return noise, times
    if last.shape[0] != noise.shape[0]:
        last = last[-1:].expand(noise.shape[0], -1, -1, -1)
    latents = (1.0 - t0) * noise + t0 * last
    rest = times[times > (t0 + 1e-6)]
    if rest.numel() == 0:
        rest = times[-1:]
    t0_t = torch.as_tensor([t0], device=times.device, dtype=times.dtype)
    return latents, torch.cat([t0_t, rest], dim=0)


def _as_batch_keypoints(keypoints: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(keypoints, np.ndarray):
        t = torch.from_numpy(keypoints.astype(np.float32))
    else:
        t = keypoints.detach().float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.shape[-2:] != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(f"Expected keypoints (..., {NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {tuple(t.shape)}")
    return t.to(device=device)


def _square_pad_offsets(width: int, height: int) -> tuple[int, int, int]:
    side = max(int(width), int(height), 1)
    pad_x = (side - int(width)) // 2
    pad_y = (side - int(height)) // 2
    return pad_x, pad_y, side


def _pad_rgb_to_square(rgb: np.ndarray, fill: tuple[int, int, int] | None = None) -> tuple[np.ndarray, int, int, int]:
    """Match ``build_dataset --skip-crop``: full frame centered on matching green square."""
    from scripts.build_dataset import sample_green_fill

    h, w = rgb.shape[:2]
    pad_x, pad_y, side = _square_pad_offsets(w, h)
    if fill is None:
        fill = sample_green_fill(rgb)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    canvas[pad_y : pad_y + h, pad_x : pad_x + w] = rgb
    return canvas, pad_x, pad_y, side


def _chroma_crop_to_square(
    rgb: np.ndarray,
    *,
    image_size: int,
    margin: int = 32,
    margin_top: int = 0,
    margin_bottom: int = 0,
    margin_x: int = 32,
    green_thresh: float = 40.0,
    green_dominance: float = 20.0,
    target_aspect: float = 1.0,
) -> tuple[np.ndarray, float, float, float, float]:
    """Match training side-only chroma crop: trim left/right, keep full height.

    Returns ``(resized_rgb, crop_x0, crop_y0, crop_w, crop_h)`` for keypoint remapping.
    """
    del margin_top, margin_bottom, target_aspect  # side-only path ignores vertical crop margins
    from scripts.build_dataset import (
        BBox,
        crop_and_resize,
        horizontal_crop_bbox,
        chroma_key_mask,
        sample_green_fill,
    )

    h, w = rgb.shape[:2]
    mask = chroma_key_mask(rgb, green_thresh=green_thresh, green_dominance=green_dominance)
    fill = sample_green_fill(rgb, green_thresh=green_thresh, green_dominance=green_dominance)
    mx = int(margin_x if margin_x is not None else margin)
    tight = horizontal_crop_bbox(mask, margin_x=mx)
    if tight is None or tight.w < 8:
        tight = BBox(0, 0, w, h)
    side = h
    cx = (tight.x0 + tight.x1) / 2.0
    x0 = int(round(cx - side / 2.0))
    bbox = BBox(x0, 0, x0 + side, h)
    crop = crop_and_resize(rgb, bbox, image_size, image_size, fill=fill)
    return crop, float(bbox.x0), float(bbox.y0), float(bbox.w), float(bbox.h)


def _looks_like_absolute_pixels(kps: torch.Tensor) -> bool:
    """Heuristic: normalized crop coords live in ~[-1, 1]; raw labels are pixel units.

    Only visible points count — training ``*_keypoints.npy`` can store garbage
    xy on invisible slots (e.g. missing iris at -1.69), which must not flip
    already-normalized tensors into the absolute-pixel transform path.
    """
    xy = kps[..., :2]
    vis = kps[..., 3] >= 0.5
    if bool(vis.any().item()):
        xy = xy[vis]
    if xy.numel() == 0:
        return False
    return bool(xy.abs().max().item() > 1.5)


@torch.no_grad()
def encode_reference(
    vae,
    image_rgb: np.ndarray | torch.Tensor,
    keypoints: np.ndarray | torch.Tensor,
    *,
    image_size: int = 768,
    ref_face_size: int = 32,
    use_ref_face_tokens: bool = True,
    skip_crop: bool = False,
    margin: int = 32,
    margin_top: int = 0,
    margin_bottom: int = 0,
    margin_x: int = 32,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Encode a reference RGB + keypoints once per session.

    Preprocessing matches training ``build_dataset`` by default:
    chroma-key crop left/right only (full height kept), pad to a square with
    matching green, then resize to ``image_size x image_size``. Set
    ``skip_crop=True`` to keep the full frame and only pad to square.

    Keypoint contract:
      - Prefer keypoints already in normalized ``[-1, 1]`` crop space
        (same as ``*_keypoints.npy`` from ``build_dataset``).
      - Absolute pixel keypoints from the raw image are accepted and transformed
        through the same crop/pad+resize.

    Returns:
        ref_latent: (1, C, H, W) scaled SD latent
        ref_face_latent: (1, C, ref_face_size, ref_face_size) or None
        ref_keypoints: (1, 37, 4) on the same device as the latent
    """
    from PIL import Image as PILImage
    from utils.config import ConfigDict
    from utils.keypoints import transform_keypoints_crop

    if isinstance(image_rgb, torch.Tensor):
        arr = image_rgb.detach().cpu().numpy()
    else:
        arr = np.asarray(image_rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {arr.shape}")

    if skip_crop:
        canvas, pad_x, pad_y, side = _pad_rgb_to_square(arr)
        img = PILImage.fromarray(canvas, mode="RGB").resize(
            (image_size, image_size), resample=PILImage.BILINEAR
        )
        crop_x0, crop_y0, crop_w, crop_h = float(-pad_x), float(-pad_y), float(side), float(side)
        arr_out = np.asarray(img)
    else:
        arr_out, crop_x0, crop_y0, crop_w, crop_h = _chroma_crop_to_square(
            arr,
            image_size=image_size,
            margin=margin,
            margin_top=margin_top,
            margin_bottom=margin_bottom,
            margin_x=margin_x,
        )

    tensor = torch.from_numpy(arr_out.astype(np.float32) / 127.5 - 1.0)[None]
    device = next(vae.parameters()).device
    tensor = tensor.to(device)
    latents = encode_images_to_latents(vae, tensor, sample=False)
    latents = scale_latents(latents, ConfigDict(dict(vae_type="sd"))).float()

    kps = _as_batch_keypoints(keypoints, device)
    if _looks_like_absolute_pixels(kps):
        transformed = []
        for i in range(kps.shape[0]):
            transformed.append(
                transform_keypoints_crop(
                    kps[i].detach().cpu().numpy(),
                    crop_x0=crop_x0,
                    crop_y0=crop_y0,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    out_w=image_size,
                    out_h=image_size,
                    normalize=True,
                )
            )
        kps = torch.from_numpy(np.stack(transformed, axis=0)).to(device=device, dtype=torch.float32)

    face = None
    if use_ref_face_tokens:
        face = crop_face_latent(latents, kps, out_size=ref_face_size)
    return latents, face, kps


@torch.no_grad()
def denoise_keypoint(
    model: i1DiT,
    *,
    keypoints_target: np.ndarray | torch.Tensor,
    ref_latent: torch.Tensor,
    ref_keypoints: np.ndarray | torch.Tensor,
    ref_face_latent: torch.Tensor | None = None,
    num_steps: int = 30,
    pose_cfg_scale: float = 1.5,
    id_cfg_scale: float = 2.0,
    inference_timestep_shift: float = 0.3,
    pose_sigma: float = 1.5,
    seed: int | None = None,
    shared_noise: bool = False,
    hair_maps: np.ndarray | torch.Tensor | None = None,
    last_latent: torch.Tensor | None = None,
    start_t: float = 0.0,
) -> torch.Tensor:
    """Denoise with separate pose / identity CFG scales (3-way batched forward).

    When ``shared_noise`` is True, every batch item starts from the same noise
    sample (expanded). Useful for stream Batch×N A/B frames so items do not
    flicker from independent noise draws.

    ``hair_maps`` is optional ``(3,H,W)`` or ``(B,3,H,W)`` in [0, 1] and is
    written onto pose-map channels 8–10 when the checkpoint has 11 channels.

    ``last_latent`` + ``start_t`` > 0 starts the flow from a mix of noise and
    the previous generated latent (img2img hold) instead of a fresh still.
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    kps_t = _as_batch_keypoints(keypoints_target, device)
    kps_r = _as_batch_keypoints(ref_keypoints, device)
    bsz = kps_t.shape[0]
    if kps_r.shape[0] == 1 and bsz > 1:
        kps_r = kps_r.expand(bsz, -1, -1)
    ref_latent = ref_latent.to(device=device, dtype=model_dtype)
    if ref_latent.shape[0] == 1 and bsz > 1:
        ref_latent = ref_latent.expand(bsz, -1, -1, -1)
    face = ref_face_latent
    if face is not None:
        face = face.to(device=device, dtype=model_dtype)
        if face.shape[0] == 1 and bsz > 1:
            face = face.expand(bsz, -1, -1, -1)

    deltas_np = []
    for i in range(bsz):
        deltas_np.append(keypoint_deltas(kps_t[i].cpu().numpy(), kps_r[i].cpu().numpy()))
    deltas = torch.from_numpy(np.stack(deltas_np, axis=0)).to(device=device, dtype=model_dtype)
    hair_t = None
    if hair_maps is not None:
        if isinstance(hair_maps, np.ndarray):
            hair_t = torch.from_numpy(np.asarray(hair_maps, dtype=np.float32))
        else:
            hair_t = hair_maps.float()
        if hair_t.ndim == 3:
            hair_t = hair_t.unsqueeze(0)
        hair_t = hair_t.to(device=device, dtype=torch.float32)
        if hair_t.shape[0] == 1 and bsz > 1:
            hair_t = hair_t.expand(bsz, -1, -1, -1)
    pose = rasterize_pose_maps(
        kps_t, model.input_size, model.input_size, sigma=pose_sigma, hair_maps=hair_t
    ).to(device=device, dtype=model_dtype)
    need = int(getattr(model, "num_pose_channels", pose.shape[1]))
    if pose.shape[1] > need:
        pose = pose[:, :need]
    elif pose.shape[1] < need:
        pad = torch.zeros(
            (bsz, need - pose.shape[1], pose.shape[2], pose.shape[3]),
            device=pose.device,
            dtype=pose.dtype,
        )
        pose = torch.cat([pose, pad], dim=1)

    gen = torch.Generator(device=device)
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little") % (2**63 - 1)
    gen.manual_seed(int(seed))
    if shared_noise and bsz > 1:
        one = torch.randn(
            (1, model.in_channels, model.input_size, model.input_size),
            generator=gen,
            device=device,
            dtype=model_dtype,
        )
        latents = one.expand(bsz, -1, -1, -1).contiguous()
    else:
        latents = torch.randn(
            (bsz, model.in_channels, model.input_size, model.input_size),
            generator=gen,
            device=device,
            dtype=model_dtype,
        )

    times = _timestep_schedule(num_steps, device, inference_timestep_shift)
    latents, times = flow_start_from_last(latents, times, last_latent, start_t)
    n_steps = int(times.numel() - 1)
    s_pose = float(pose_cfg_scale)
    s_id = float(id_cfg_scale)
    use_cfg = abs(s_pose - 1.0) > 1e-6 or abs(s_id - 1.0) > 1e-6

    for i in range(n_steps):
        t = times[i].expand(bsz)
        if not use_cfg:
            velocity = model(
                latents, t, deltas, None, train=False,
                ref_latent=ref_latent, pose_map=pose, keypoint_deltas=deltas,
                ref_face_latent=face,
            )
        else:
            # Batch: [pose-only, id-only, uncond]
            latents_in = torch.cat([latents, latents, latents], dim=0)
            t_in = torch.cat([t, t, t], dim=0)
            deltas_in = torch.cat([deltas, deltas, deltas], dim=0)
            pose_in = torch.cat([pose, pose, pose], dim=0)
            ref_in = torch.cat([ref_latent, ref_latent, ref_latent], dim=0)
            face_in = torch.cat([face, face, face], dim=0) if face is not None else None
            # pose branch: pose on, ref null
            # id branch:   ref on, pose null (pose map + keypoint deltas)
            # uncond:      both null
            force_null_pose = torch.tensor(
                [False] * bsz + [True] * bsz + [True] * bsz, device=device, dtype=torch.bool
            )
            force_null_ref = torch.tensor(
                [True] * bsz + [False] * bsz + [True] * bsz, device=device, dtype=torch.bool
            )
            # Pose information lives in both pose maps and keypoint-delta tokens.
            # Identity-only and uncond branches must null both.
            force_null_kp = torch.tensor(
                [False] * bsz + [True] * bsz + [True] * bsz, device=device, dtype=torch.bool
            )
            velocity_all = model(
                latents_in, t_in, deltas_in, None, train=False,
                force_null=force_null_kp,
                ref_latent=ref_in,
                force_null_ref=force_null_ref,
                pose_map=pose_in,
                keypoint_deltas=deltas_in,
                force_null_pose=force_null_pose,
                ref_face_latent=face_in,
            )
            v_pose, v_id, v_uncond = velocity_all.chunk(3, dim=0)
            velocity = v_uncond + s_pose * (v_pose - v_uncond) + s_id * (v_id - v_uncond)
        latents = latents + (times[i + 1] - times[i]).to(dtype=model_dtype) * velocity
    return latents


def _latents_to_uint8(decoded: torch.Tensor) -> np.ndarray:
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = (image.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return image


@torch.no_grad()
def decode_sd_vae(vae, latents: torch.Tensor) -> np.ndarray:
    # Stay in the VAE's runtime dtype. Live path is float32 so TF32 matmul
    # applies without a half-precision cast on every frame.
    vae_dtype = next(vae.parameters()).dtype
    latents = reverse_scale_latents(latents.to(dtype=vae_dtype), "sd")
    decoded = vae.decode(latents).sample
    return _latents_to_uint8(decoded.float())


def preferred_sd_vae_dtype(device: torch.device) -> torch.dtype:
    """Float32 for encode and decode.

    CUDA float32 matmul runs as TF32 when the engine enables it, which is
    faster here than bfloat16: the DiT is small, and half precision adds a
    cast on every latent without a faster kernel.
    """
    del device
    return torch.float32


def load_sd_vae(device: torch.device, dtype: torch.dtype | None = None):
    from utils.config import ConfigDict

    if dtype is None:
        dtype = preferred_sd_vae_dtype(device)
    return load_vae(ConfigDict(dict(vae_type="sd")), device, dtype=dtype)


# Hybrid TinyVAE (TAESD finetune) — same SD latent API, much faster decode.
DEFAULT_TINY_VAE_ID = "cqyan/hybrid-sd-tinyvae"
FALLBACK_TINY_VAE_ID = "madebyollin/taesd"


def load_tiny_vae(
    device: torch.device,
    dtype: torch.dtype | None = None,
    model_id: str | None = None,
):
    """Load AutoencoderTiny for fast SD-latent decode (Hybrid, else TAESD)."""
    from diffusers import AutoencoderTiny

    if dtype is None:
        dtype = preferred_sd_vae_dtype(device)
    requested = model_id or DEFAULT_TINY_VAE_ID
    candidates: list[str] = [requested]
    if requested != FALLBACK_TINY_VAE_ID:
        candidates.append(FALLBACK_TINY_VAE_ID)

    last_err: Exception | None = None
    for i, candidate in enumerate(candidates):
        try:
            vae = AutoencoderTiny.from_pretrained(candidate)
            vae = vae.to(device=device, dtype=dtype).eval()
            for p in vae.parameters():
                p.requires_grad_(False)
            if i > 0:
                print(f"TinyVAE fallback loaded: {candidate}")
            return vae, candidate
        except Exception as exc:
            last_err = exc
            if i + 1 < len(candidates):
                print(
                    f"TinyVAE '{candidate}' failed ({exc}); "
                    f"trying {candidates[i + 1]} ..."
                )
    raise RuntimeError(f"Failed to load TinyVAE ({requested})") from last_err


@torch.no_grad()
def decode_tiny_vae(vae, latents: torch.Tensor) -> np.ndarray:
    """Decode SD-scaled latents with AutoencoderTiny (Hybrid / TAESD).

    Diffusers TinyVAE configs use ``scaling_factor=1.0``, meaning they expect the
    same scaled latents the DiT outputs (already ``* 0.18215``). Dividing by
    the SD factor first (as we do for AutoencoderKL) overdrives the decoder and
    produces the harsh, over-bright look.
    """
    vae_dtype = next(vae.parameters()).dtype
    z = latents.to(dtype=vae_dtype)
    scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
    if abs(scale - 1.0) > 1e-6:
        z = z / scale
    decoded = vae.decode(z).sample
    return _latents_to_uint8(decoded.float())


__all__ = [
    "is_keypoint_checkpoint",
    "build_keypoint_model",
    "encode_reference",
    "flow_start_from_last",
    "denoise_keypoint",
    "decode_sd_vae",
    "decode_tiny_vae",
    "load_sd_vae",
    "load_tiny_vae",
    "preferred_sd_vae_dtype",
    "DEFAULT_TINY_VAE_ID",
    "FALLBACK_TINY_VAE_ID",
]
