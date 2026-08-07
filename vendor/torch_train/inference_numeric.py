"""Numeric-conditioning inference helpers (no CLIP).

Supports optional reference-image conditioning (channel-concat ref latent).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage

from models.dit import DualStreamDiTConfig, i1DiT
from utils.params import (
    NUM_PARAMS,
    PARAM_NAMES,
    normalize_params,
    parse_caption_params,
)
from vae.vae import encode_images_to_latents, load_vae, reverse_scale_latents, scale_latents


def is_numeric_checkpoint(checkpoint: dict[str, Any] | Path | str) -> bool:
    if isinstance(checkpoint, (str, Path)):
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    else:
        ckpt = checkpoint
    cfg = ckpt.get("config") or {}
    return bool(cfg.get("use_numeric_conditioning"))


def build_numeric_model(checkpoint_path: str | Path, device: torch.device) -> tuple[i1DiT, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["config"])
    ds_cfg = DualStreamDiTConfig(
        input_size=int(cfg_dict["input_size"]),
        image_resolution=int(cfg_dict.get("image_resolution", 512)),
        patch_size=int(cfg_dict.get("patch_size", 2)),
        in_channels=int(cfg_dict["in_channels"]),
        hidden_size=int(cfg_dict["hidden_size"]),
        depth=int(cfg_dict["depth"]),
        num_heads=int(cfg_dict["num_heads"]),
        mlp_ratio=float(cfg_dict.get("mlp_ratio", 4.0)),
        text_embed_dim=int(cfg_dict.get("text_embed_dim", cfg_dict["hidden_size"])),
        text_num_tokens=int(cfg_dict.get("text_num_tokens", cfg_dict.get("num_params", NUM_PARAMS))),
        use_numeric_conditioning=True,
        num_params=int(cfg_dict.get("num_params", NUM_PARAMS)),
        use_adaln=bool(cfg_dict.get("use_adaln", True)),
        use_ref_conditioning=bool(cfg_dict.get("use_ref_conditioning", False)),
        drop_ref_prob=0.0,
        use_long_skip=bool(cfg_dict.get("use_long_skip", True)),
        use_sandwich_norm=bool(cfg_dict.get("use_sandwich_norm", True)),
        use_qknorm=bool(cfg_dict.get("use_qknorm", True)),
        use_swiglu=bool(cfg_dict.get("use_swiglu", True)),
        use_rmsnorm=bool(cfg_dict.get("use_rmsnorm", True)),
        rope_theta=float(cfg_dict.get("rope_theta", 10000.0)),
        drop_text_prob=0.0,
    )
    model = i1DiT(ds_cfg).to(device=device, dtype=torch.bfloat16).eval()
    state = ckpt.get("ema") or ckpt.get("model")
    if isinstance(state, dict) and "shadow" in state:
        state = state["shadow"]
    model.load_state_dict(state, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return model, cfg_dict


def prompt_to_params(
    prompt: str,
    param_stats: dict | None = None,
) -> torch.Tensor:
    """Parse legacy caption text (or JSON list) into a normalized (1, N) float tensor."""
    prompt = (prompt or "").strip()
    if prompt.startswith("[") and prompt.endswith("]"):
        values = [float(x) for x in json.loads(prompt)]
        if len(values) != NUM_PARAMS:
            raise ValueError(f"Expected {NUM_PARAMS} floats, got {len(values)}")
    else:
        values = parse_caption_params(prompt)
    if param_stats and param_stats.get("normalized"):
        values = normalize_params(values, param_stats["param_mins"], param_stats["param_maxs"])
    return torch.tensor([values], dtype=torch.float32)


def load_param_stats(data_dir: str | Path | None) -> dict | None:
    if not data_dir:
        return None
    path = Path(data_dir) / "param_stats.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


@torch.no_grad()
def denoise_numeric(
    model: i1DiT,
    params: torch.Tensor,
    *,
    ref_latent: torch.Tensor | None = None,
    num_steps: int = 30,
    cfg_scale: float = 1.5,
    inference_timestep_shift: float = 0.3,
) -> torch.Tensor:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    bsz = params.shape[0]
    shape = (bsz, model.in_channels, model.input_size, model.input_size)
    gen = torch.Generator(device=device)
    gen.manual_seed(int.from_bytes(os.urandom(8), "little") % (2**63 - 1))
    latents = torch.randn(shape, generator=gen, device=device, dtype=model_dtype)
    params = params.to(device=device, dtype=model_dtype)
    if model.use_ref_conditioning:
        if ref_latent is None:
            raise ValueError("ref_latent required for reference-conditioned checkpoints")
        ref_latent = ref_latent.to(device=device, dtype=model_dtype)
        if ref_latent.shape[0] == 1 and bsz > 1:
            ref_latent = ref_latent.expand(bsz, -1, -1, -1)

    times = _timestep_schedule(num_steps, device, inference_timestep_shift)

    for i in range(num_steps):
        t = times[i].expand(bsz)
        if cfg_scale == 1.0:
            velocity = model(latents, t, params, None, train=False, ref_latent=ref_latent)
        else:
            vel_cond = model(
                latents, t, params, None, train=False,
                force_null=False, ref_latent=ref_latent, force_null_ref=False,
            )
            vel_uncond = model(
                latents, t, params, None, train=False,
                force_null=True, ref_latent=ref_latent, force_null_ref=True,
            )
            velocity = vel_cond + (cfg_scale - 1.0) * (vel_cond - vel_uncond)
        latents = latents + (times[i + 1] - times[i]).to(dtype=model_dtype) * velocity
    return latents


@torch.no_grad()
def denoise_numeric_turbo(
    model: i1DiT,
    params: torch.Tensor,
    *,
    ref_latent: torch.Tensor | None = None,
    num_steps: int = 8,
    cfg_scale: float = 1.5,
    inference_timestep_shift: float = 0.3,
    seed: int | None = None,
) -> torch.Tensor:
    """Faster denoise: fewer default steps + batched CFG (one forward per step).

    If ``seed`` is set, initial noise is deterministic (same seed → same noise).
    If ``seed`` is None, noise is random each call.
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    bsz = params.shape[0]
    shape = (bsz, model.in_channels, model.input_size, model.input_size)
    gen = torch.Generator(device=device)
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little") % (2**63 - 1)
    gen.manual_seed(int(seed))
    latents = torch.randn(shape, generator=gen, device=device, dtype=model_dtype)
    params = params.to(device=device, dtype=model_dtype)
    if model.use_ref_conditioning:
        if ref_latent is None:
            raise ValueError("ref_latent required for reference-conditioned checkpoints")
        ref_latent = ref_latent.to(device=device, dtype=model_dtype)
        if ref_latent.shape[0] == 1 and bsz > 1:
            ref_latent = ref_latent.expand(bsz, -1, -1, -1)

    times = _timestep_schedule(num_steps, device, inference_timestep_shift)

    for i in range(num_steps):
        t = times[i].expand(bsz)
        if abs(cfg_scale - 1.0) < 1e-6:
            velocity = model(latents, t, params, None, train=False, ref_latent=ref_latent)
        else:
            latents_in = torch.cat([latents, latents], dim=0)
            params_in = torch.cat([params, params], dim=0)
            t_in = torch.cat([t, t], dim=0)
            force_null = torch.zeros(2 * bsz, dtype=torch.bool, device=device)
            force_null[bsz:] = True
            ref_in = None
            force_null_ref: bool | torch.Tensor = False
            if model.use_ref_conditioning:
                ref_in = torch.cat([ref_latent, ref_latent], dim=0)
                force_null_ref = force_null
            velocity_both = model(
                latents_in, t_in, params_in, None, train=False,
                force_null=force_null, ref_latent=ref_in, force_null_ref=force_null_ref,
            )
            vel_cond, vel_uncond = velocity_both.chunk(2, dim=0)
            velocity = vel_cond + (cfg_scale - 1.0) * (vel_cond - vel_uncond)
        latents = latents + (times[i + 1] - times[i]).to(dtype=model_dtype) * velocity
    return latents


def encode_reference_image(
    vae,
    image_rgb: np.ndarray | torch.Tensor,
    *,
    image_size: int = 512,
) -> torch.Tensor:
    """Encode a reference RGB image to scaled SD latent (1,4,h,w)."""
    from utils.config import ConfigDict

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
    img = PILImage.fromarray(arr, mode="RGB")
    w, h = img.size
    if w <= h:
        new_w, new_h = image_size, int(round(h * (image_size / w)))
    else:
        new_h, new_w = image_size, int(round(w * (image_size / h)))
    img = img.resize((new_w, new_h), resample=PILImage.BILINEAR)
    left = (new_w - image_size) // 2
    top = (new_h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))
    tensor = torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1.0)[None]
    device = next(vae.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        latents = encode_images_to_latents(vae, tensor, sample=False)
        latents = scale_latents(latents, ConfigDict(dict(vae_type="sd"))).float()
    return latents


def _latents_to_uint8(decoded: torch.Tensor) -> np.ndarray:
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = (image.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return image


@torch.no_grad()
def decode_sd_vae(vae, latents: torch.Tensor) -> np.ndarray:
    latents = reverse_scale_latents(latents.float(), "sd")
    decoded = vae.decode(latents.to(dtype=vae.dtype)).sample
    return _latents_to_uint8(decoded)


@torch.no_grad()
def decode_sd_vae_fast(vae, latents: torch.Tensor) -> np.ndarray:
    """Decode with the VAE's native dtype (e.g. bf16) instead of forcing float32."""
    latents = reverse_scale_latents(latents.to(dtype=vae.dtype), "sd")
    decoded = vae.decode(latents).sample
    return _latents_to_uint8(decoded)


def load_sd_vae(device: torch.device, dtype=torch.float32):
    from utils.config import ConfigDict

    return load_vae(ConfigDict(dict(vae_type="sd")), device, dtype=dtype)


def load_taesd(device: torch.device, dtype=torch.float32):
    """Tiny AutoEncoder for SD latents (drop-in decoder, much faster than SD-VAE)."""
    from diffusers import AutoencoderTiny

    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=device, dtype=dtype).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def decode_taesd(taesd, latents: torch.Tensor) -> np.ndarray:
    latents = reverse_scale_latents(latents.to(dtype=taesd.dtype), "sd")
    decoded = taesd.decode(latents).sample
    return _latents_to_uint8(decoded)


__all__ = [
    "PARAM_NAMES",
    "NUM_PARAMS",
    "is_numeric_checkpoint",
    "build_numeric_model",
    "prompt_to_params",
    "load_param_stats",
    "denoise_numeric",
    "denoise_numeric_turbo",
    "encode_reference_image",
    "decode_sd_vae",
    "decode_sd_vae_fast",
    "decode_taesd",
    "load_sd_vae",
    "load_taesd",
]
