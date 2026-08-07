from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclasses.dataclass
class DualStreamDiTConfig:
    input_size: int = 16
    patch_size: int = 2
    in_channels: int = 32
    hidden_size: int = 1152
    depth: int = 29
    num_heads: int = 16
    mlp_ratio: float = 4.0
    use_qknorm: bool = True
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    image_resolution: int = 256
    text_embed_dim: int = 2304
    text_num_tokens: int = 256
    drop_text_prob: float = 0.1
    use_long_skip: bool = True
    text_encoder_adapter_type: str = "transformer"
    text_encoder_adapter_num_blocks: int = 2
    use_sandwich_norm: bool = True
    use_separate_norms: bool = False
    use_grad_ckpt: bool = False
    rope_theta: float = 10000.0
    rope_axes_dims: Optional[tuple] = None
    rope_axes_lens: Optional[tuple] = None
    # Numeric conditioning (Live2D / face-tracking floats).
    use_numeric_conditioning: bool = False
    num_params: int = 16
    use_adaln: bool = False
    # Reference-image conditioning: concat VAE-encoded ref latent to noisy latent.
    use_ref_conditioning: bool = False
    drop_ref_prob: float = 0.1
    # Keypoint conditioning: pose-map concat + ref identity tokens in second stream.
    use_keypoint_conditioning: bool = False
    num_keypoints: int = 37
    num_pose_channels: int = 8
    drop_pose_prob: float = 0.1
    # If True with keypoint mode, ref is cross-attn tokens (not channel-concat).
    use_ref_tokens: bool = True
    # Spatially downsample ref before patchify (1=full, 2=half) to control token count.
    ref_spatial_downsample: int = 2
    # Full-res face-crop ref tokens (dual-scale identity).
    use_ref_face_tokens: bool = False
    ref_face_size: int = 32  # latent spatial size of face crop
    # VAE-REPA: align mid features to VAE latents (training accelerator).
    use_repa: bool = False
    # Use 2-D RoPE for second-stream tokens in keypoint mode.
    use_keypoint_rope: bool = True


DualStreamDiT_models = {
    # ~1.95M trainable params with CLIP (768-d, 77 tok) + SD-VAE (4ch) @ 512.
    # Adapter blocks=2 to match torch_inference TextEncoderAdapterTransformer.
    "DiT-2M": dict(depth=5, hidden_size=104, num_heads=4, mlp_ratio=4.0),
    # ~30M with numeric ParamEmbedder + SD-VAE (4ch) @ 512 (dual-stream).
    "DiT-30M": dict(depth=10, hidden_size=320, num_heads=5, mlp_ratio=4.0),
    # ~100M for multi-character keypoint + ref-token training.
    "DiT-100M": dict(depth=14, hidden_size=512, num_heads=8, mlp_ratio=4.0),
    "DiT-XL": dict(depth=29, hidden_size=1152, num_heads=16, mlp_ratio=4.0),
    "DiT-XL_1296": dict(depth=29, hidden_size=1296, num_heads=18, mlp_ratio=4.0),
    "DiT-XL_1440": dict(depth=29, hidden_size=1440, num_heads=20, mlp_ratio=4.0),
    "DiT-XL_1584": dict(depth=29, hidden_size=1584, num_heads=22, mlp_ratio=4.0),
    "DiT-XL_1728": dict(depth=29, hidden_size=1728, num_heads=24, mlp_ratio=4.0),
    "DiT-XL_1872": dict(depth=29, hidden_size=1872, num_heads=26, mlp_ratio=4.0),
    "DiT-XL_2016": dict(depth=29, hidden_size=2016, num_heads=28, mlp_ratio=4.0),
}


def _get_1d_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    out = np.outer(pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_interpolated_pos_embed(
    embed_dim: int,
    grid_size: int,
    image_resolution: int,
    base_image_resolution: int = 256,
) -> np.ndarray:
    scale = float(base_image_resolution) / float(image_resolution)
    grid_h = np.arange(grid_size, dtype=np.float32) * scale
    grid_w = np.arange(grid_size, dtype=np.float32) * scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _default_rope_axes_dims(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE.")
    time_dim = head_dim // 2
    if time_dim % 2 != 0:
        time_dim -= 1
    remaining = head_dim - time_dim
    row_dim = remaining // 2
    col_dim = remaining - row_dim
    if row_dim % 2 != 0:
        row_dim -= 1
        col_dim += 1
    if col_dim % 2 != 0:
        col_dim -= 1
        row_dim += 1
    if min(time_dim, row_dim, col_dim) <= 0:
        raise ValueError("Each RoPE axis must receive at least two dimensions.")
    return time_dim, row_dim, col_dim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * self.scale.float()).to(dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = (x_float - mean).square().mean(dim=-1, keepdim=True)
        x_float = (x_float - mean) * torch.rsqrt(var + self.eps)
        return (x_float * self.scale.float() + self.bias.float()).to(dtype)


def _norm(use_rmsnorm: bool):
    return RMSNorm if use_rmsnorm else LayerNorm


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int, in_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        # Sinusoidal emb is float32; match Linear weight dtype (bf16 under numeric inference).
        x = x.to(dtype=self.linear1.weight.dtype)
        return self.linear2(F.silu(self.linear1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(hidden_size, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class MlpBlock(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def _make_ffn(use_swiglu: bool, hidden_size: int, mlp_ratio: float) -> nn.Module:
    mlp_hidden = int(hidden_size * mlp_ratio)
    if use_swiglu:
        return SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden))
    return MlpBlock(hidden_size, mlp_hidden)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        norm = _norm(use_rmsnorm)
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.proj(out)


class TextEncoderAdapterTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        drop_text_prob: float,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        token_len: int,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.drop_text_prob = drop_text_prob
        self.num_blocks = num_blocks
        self.learnable_null_caption = nn.Parameter(torch.empty(1, token_len, in_channels))
        nn.init.normal_(self.learnable_null_caption, std=in_channels ** -0.5)
        self.connector_in = nn.Linear(in_channels, hidden_size)
        norm = _norm(use_rmsnorm)
        for block_idx in range(num_blocks):
            suffix = "" if block_idx == 0 else str(block_idx + 1)
            setattr(self, f"connector_norm{2 * block_idx + 1}", norm(hidden_size))
            setattr(self, f"connector_norm{2 * block_idx + 2}", norm(hidden_size))
            setattr(self, f"connector_attn{suffix}", Attention(hidden_size, num_heads, use_qknorm, use_rmsnorm))
            setattr(self, f"connector_mlp{suffix}", _make_ffn(use_swiglu, hidden_size, mlp_ratio))

    def forward(self, caption: torch.Tensor, train: bool) -> torch.Tensor:
        if train and self.drop_text_prob > 0:
            drop = torch.rand(caption.shape[0], device=caption.device) < self.drop_text_prob
            null = self.learnable_null_caption.to(caption.dtype)
            caption = torch.where(drop[:, None, None], null, caption)
        x = self.connector_in(caption)
        for block_idx in range(self.num_blocks):
            suffix = "" if block_idx == 0 else str(block_idx + 1)
            norm1 = getattr(self, f"connector_norm{2 * block_idx + 1}")
            norm2 = getattr(self, f"connector_norm{2 * block_idx + 2}")
            attn = getattr(self, f"connector_attn{suffix}")
            mlp = getattr(self, f"connector_mlp{suffix}")
            x = x + attn(norm1(x))
            x = x + mlp(norm2(x))
        return x


class MultimodalRopeEmbedder(nn.Module):
    def __init__(
        self,
        axes_dims: tuple,
        axes_lens: tuple,
        axes_scales: tuple,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        cos_tables = []
        sin_tables = []
        for dim, axis_len, axis_scale in zip(axes_dims, axes_lens, axes_scales):
            steps = torch.arange(0, dim, 2, dtype=torch.float32)
            base = 1.0 / (theta ** (steps / dim))
            positions = torch.arange(axis_len, dtype=torch.float32) * axis_scale
            angles = positions[:, None] * base[None, :]
            cos_tables.append(angles.cos())
            sin_tables.append(angles.sin())
        self.cos_tables = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in cos_tables])
        self.sin_tables = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in sin_tables])
        self.axes_lens = tuple(int(x) for x in axes_lens)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = [], []
        for axis_idx, (cos_table, sin_table) in enumerate(zip(self.cos_tables, self.sin_tables)):
            pos = position_ids[:, :, axis_idx].clamp(0, cos_table.shape[0] - 1)
            cos.append(cos_table[pos])
            sin.append(sin_table[pos])
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def _apply_multimodal_rope(
    x: torch.Tensor,
    freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if freqs is None:
        return x
    cos, sin = freqs
    dtype = x.dtype
    x_pair = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pair.unbind(dim=-1)
    cos = cos[:, None].float()
    sin = sin[:, None].float()
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape_as(x).to(dtype)


class MMDiTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_image = nn.Linear(hidden_size, 3 * hidden_size)
        self.qkv_text = nn.Linear(hidden_size, 3 * hidden_size)
        norm = _norm(use_rmsnorm)
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.proj_image = nn.Linear(hidden_size, hidden_size)
        self.proj_text = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, image_len, _ = image_tokens.shape
        text_len = text_tokens.shape[1]

        def project(linear: nn.Linear, x: torch.Tensor):
            qkv = linear(x).reshape(bsz, x.shape[1], 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q_image, k_image, v_image = project(self.qkv_image, image_tokens)
        q_text, k_text, v_text = project(self.qkv_text, text_tokens)
        if self.q_norm is not None:
            q_image = self.q_norm(q_image)
            k_image = self.k_norm(k_image)
            q_text = self.q_norm(q_text)
            k_text = self.k_norm(k_text)
        q_image = _apply_multimodal_rope(q_image, image_freqs)
        k_image = _apply_multimodal_rope(k_image, image_freqs)
        q_text = _apply_multimodal_rope(q_text, text_freqs)
        k_text = _apply_multimodal_rope(k_text, text_freqs)
        q = torch.cat([q_image, q_text], dim=2)
        k = torch.cat([k_image, k_text], dim=2)
        v = torch.cat([v_image, v_text], dim=2)
        key_mask = None
        attn_mask = None
        if text_mask is not None:
            image_mask = torch.ones((bsz, image_len), dtype=torch.bool, device=text_tokens.device)
            key_mask = torch.cat([image_mask, text_mask.bool()], dim=1)
            attn_mask = key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, image_len + text_len, self.hidden_size)
        if key_mask is not None:
            out = out * key_mask[:, :, None].to(out.dtype)
        return self.proj_image(out[:, :image_len]), self.proj_text(out[:, image_len:])


class DualStreamDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        use_sandwich_norm: bool,
        use_separate_norms: bool,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        self.use_sandwich_norm = use_sandwich_norm
        self.use_separate_norms = use_separate_norms
        if use_skip:
            self.skip_linear_image = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_linear_text = nn.Linear(2 * hidden_size, hidden_size)
        norm = _norm(use_rmsnorm)
        if use_separate_norms:
            self.norm1_image = norm(hidden_size)
            self.norm1_text = norm(hidden_size)
            self.norm2_image = norm(hidden_size)
            self.norm2_text = norm(hidden_size)
        else:
            self.norm1 = norm(hidden_size)
            self.norm2 = norm(hidden_size)
        if use_sandwich_norm:
            self.norm3 = norm(hidden_size)
            self.norm4 = norm(hidden_size)
        self.attn = MMDiTAttention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        self.mlp_image = _make_ffn(use_swiglu, hidden_size, mlp_ratio)
        self.mlp_text = _make_ffn(use_swiglu, hidden_size, mlp_ratio)

    def _norm1(self):
        return (self.norm1_image, self.norm1_text) if self.use_separate_norms else (self.norm1, self.norm1)

    def _norm2(self):
        return (self.norm2_image, self.norm2_text) if self.use_separate_norms else (self.norm2, self.norm2)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs,
        text_freqs,
        text_mask: Optional[torch.Tensor],
        skip: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_skip:
            if skip is None:
                raise ValueError("Skip connection is required.")
            image_tokens = self.skip_linear_image(torch.cat([image_tokens, skip[0]], dim=-1))
            text_tokens = self.skip_linear_text(torch.cat([text_tokens, skip[1]], dim=-1))
        norm1_image, norm1_text = self._norm1()
        norm2_image, norm2_text = self._norm2()
        image_attn, text_attn = self.attn(
            norm1_image(image_tokens),
            norm1_text(text_tokens),
            image_freqs,
            text_freqs,
            text_mask,
        )
        if self.use_sandwich_norm:
            image_attn = self.norm3(image_attn)
            text_attn = self.norm3(text_attn)
        image_tokens = image_tokens + image_attn
        text_tokens = text_tokens + text_attn
        image_mlp = self.mlp_image(norm2_image(image_tokens))
        text_mlp = self.mlp_text(norm2_text(text_tokens))
        if self.use_sandwich_norm:
            image_mlp = self.norm4(image_mlp)
            text_mlp = self.norm4(text_mlp)
        image_tokens = image_tokens + image_mlp
        text_tokens = text_tokens + text_mlp
        if text_mask is not None:
            text_tokens = text_tokens * text_mask[:, :, None].to(text_tokens.dtype)
        return image_tokens, text_tokens


class FinalLayerNoAdaLN(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool) -> None:
        super().__init__()
        self.norm_final = _norm(use_rmsnorm)(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        del c
        return self.linear(self.norm_final(x))


class FinalLayerAdaLN(nn.Module):
    """Final projection with adaLN from a conditioning vector (timestep + pooled params)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool) -> None:
        super().__init__()
        self.norm_final = _norm(use_rmsnorm)(hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if c is None:
            return self.linear(self.norm_final(x))
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class ParamEmbedder(nn.Module):
    """Map a fixed-length float param vector to conditioning tokens.

    Each scalar is projected by a shared MLP and added to a per-param ID embedding,
    producing `num_params` tokens of `hidden_size`. Supports CFG dropout via a
    learnable null token bank (same role as learnable_null_caption).
    """

    def __init__(
        self,
        num_params: int,
        hidden_size: int,
        drop_prob: float = 0.1,
        mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.num_params = int(num_params)
        self.hidden_size = int(hidden_size)
        self.drop_prob = float(drop_prob)
        hid = mlp_hidden or max(hidden_size, 128)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, hid),
            nn.SiLU(),
            nn.Linear(hid, hidden_size),
        )
        self.param_id_embed = nn.Embedding(self.num_params, hidden_size)
        self.learnable_null_caption = nn.Parameter(torch.empty(1, self.num_params, hidden_size))
        nn.init.normal_(self.learnable_null_caption, std=hidden_size ** -0.5)
        # Marker so stack_matches_model can read .in_features (reports hidden_size).
        self.connector_in = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.connector_in.weight)
        self.connector_in.weight.requires_grad_(False)

    def forward(
        self,
        params: torch.Tensor,
        train: bool = False,
        force_null: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        # params: (B, num_params)
        # force_null: bool (all samples) or bool tensor of shape (B,) for batched CFG.
        if params.ndim != 2 or params.shape[1] != self.num_params:
            raise ValueError(f"Expected params shape (B, {self.num_params}), got {tuple(params.shape)}")
        bsz = params.shape[0]
        weight_dtype = self.value_mlp[0].weight.dtype
        null = self.learnable_null_caption.to(dtype=weight_dtype)
        if isinstance(force_null, torch.Tensor):
            if force_null.shape != (bsz,):
                raise ValueError(f"force_null tensor must have shape ({bsz},), got {tuple(force_null.shape)}")
            # Avoid .all().item() host sync — illegal during CUDA graph capture.
            values = params.unsqueeze(-1).to(dtype=weight_dtype)  # (B, N, 1)
            tokens = self.value_mlp(values)  # (B, N, H)
            ids = torch.arange(self.num_params, device=params.device)
            tokens = tokens + self.param_id_embed(ids)[None].to(dtype=tokens.dtype)
            tokens = torch.where(force_null[:, None, None], null.to(dtype=tokens.dtype), tokens)
            if train and self.drop_prob > 0:
                drop = torch.rand(bsz, device=params.device) < self.drop_prob
                # Do not overwrite samples already forced to null.
                drop = drop & ~force_null
                tokens = torch.where(drop[:, None, None], null.to(dtype=tokens.dtype), tokens)
            return tokens
        if force_null:
            return null.expand(bsz, -1, -1)
        values = params.unsqueeze(-1).to(dtype=weight_dtype)  # (B, N, 1)
        tokens = self.value_mlp(values)  # (B, N, H)
        ids = torch.arange(self.num_params, device=params.device)
        tokens = tokens + self.param_id_embed(ids)[None].to(dtype=tokens.dtype)
        if train and self.drop_prob > 0:
            drop = torch.rand(bsz, device=params.device) < self.drop_prob
            tokens = torch.where(drop[:, None, None], null.to(dtype=tokens.dtype), tokens)
        return tokens

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


class KeypointDeltaEmbedder(nn.Module):
    """Embed per-point pose deltas (dx, dy, score, visible) as conditioning tokens.

    Produces ``num_keypoints`` tokens of ``hidden_size``. Supports CFG dropout via
    a learnable null token bank.
    """

    def __init__(
        self,
        num_keypoints: int,
        hidden_size: int,
        drop_prob: float = 0.1,
        in_features: int = 4,
        mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.hidden_size = int(hidden_size)
        self.drop_prob = float(drop_prob)
        self.in_features = int(in_features)
        hid = mlp_hidden or max(hidden_size, 128)
        self.value_mlp = nn.Sequential(
            nn.Linear(self.in_features, hid),
            nn.SiLU(),
            nn.Linear(hid, hidden_size),
        )
        self.point_id_embed = nn.Embedding(self.num_keypoints, hidden_size)
        self.learnable_null = nn.Parameter(torch.empty(1, self.num_keypoints, hidden_size))
        nn.init.normal_(self.learnable_null, std=hidden_size ** -0.5)

    def forward(
        self,
        deltas: torch.Tensor,
        train: bool = False,
        force_null: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        # deltas: (B, num_keypoints, in_features)
        if deltas.ndim != 3 or deltas.shape[1] != self.num_keypoints or deltas.shape[2] != self.in_features:
            raise ValueError(
                f"Expected deltas (B, {self.num_keypoints}, {self.in_features}), got {tuple(deltas.shape)}"
            )
        bsz = deltas.shape[0]
        weight_dtype = self.value_mlp[0].weight.dtype
        null = self.learnable_null.to(dtype=weight_dtype)
        values = deltas.to(dtype=weight_dtype)
        tokens = self.value_mlp(values)
        ids = torch.arange(self.num_keypoints, device=deltas.device)
        tokens = tokens + self.point_id_embed(ids)[None].to(dtype=tokens.dtype)

        if isinstance(force_null, torch.Tensor):
            if force_null.shape != (bsz,):
                raise ValueError(f"force_null tensor must have shape ({bsz},), got {tuple(force_null.shape)}")
            tokens = torch.where(force_null[:, None, None], null.to(dtype=tokens.dtype), tokens)
            if train and self.drop_prob > 0:
                drop = torch.rand(bsz, device=deltas.device) < self.drop_prob
                drop = drop & ~force_null
                tokens = torch.where(drop[:, None, None], null.to(dtype=tokens.dtype), tokens)
            return tokens
        if force_null:
            return null.expand(bsz, -1, -1)
        if train and self.drop_prob > 0:
            drop = torch.rand(bsz, device=deltas.device) < self.drop_prob
            tokens = torch.where(drop[:, None, None], null.to(dtype=tokens.dtype), tokens)
        return tokens

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


class i1DiT(nn.Module):
    def __init__(self, config: DualStreamDiTConfig) -> None:
        super().__init__()
        cfg = config
        self.config = cfg
        self.input_size = cfg.input_size
        self.patch_size = cfg.patch_size
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.in_channels
        self.use_grad_ckpt = cfg.use_grad_ckpt
        self.use_numeric_conditioning = bool(cfg.use_numeric_conditioning)
        self.use_keypoint_conditioning = bool(cfg.use_keypoint_conditioning)
        self.use_adaln = bool(cfg.use_adaln)
        self.num_params = int(cfg.num_params)
        self.num_keypoints = int(cfg.num_keypoints)
        self.num_pose_channels = int(cfg.num_pose_channels)
        self.drop_pose_prob = float(cfg.drop_pose_prob)
        self.use_ref_tokens = bool(cfg.use_ref_tokens) and self.use_keypoint_conditioning
        self.ref_spatial_downsample = max(1, int(cfg.ref_spatial_downsample))
        self.use_ref_face_tokens = bool(cfg.use_ref_face_tokens) and self.use_ref_tokens
        self.ref_face_size = int(cfg.ref_face_size)
        self.use_keypoint_rope = bool(getattr(cfg, "use_keypoint_rope", True)) and self.use_keypoint_conditioning

        self.use_ref_conditioning = bool(cfg.use_ref_conditioning) or self.use_keypoint_conditioning
        self.drop_ref_prob = float(cfg.drop_ref_prob)

        # Patch input channels:
        # - legacy ref concat: 2 * in_channels
        # - keypoint pose-map: in_channels + num_pose_channels (ref goes to token stream)
        # - plain: in_channels
        if self.use_keypoint_conditioning:
            patch_in = cfg.in_channels + self.num_pose_channels
        elif self.use_ref_conditioning and not self.use_ref_tokens:
            patch_in = cfg.in_channels * 2
        else:
            patch_in = cfg.in_channels
        self.patch_in_channels = patch_in
        self.out_channels = cfg.in_channels

        self.x_embedder = PatchEmbed(cfg.patch_size, cfg.hidden_size, patch_in)
        hw = cfg.input_size // cfg.patch_size
        self.hw = hw
        if int(cfg.image_resolution) == 256:
            pos = _get_pos_embed(cfg.hidden_size, hw)
        else:
            pos = _get_interpolated_pos_embed(cfg.hidden_size, hw, int(cfg.image_resolution))
        self.pos_embed = nn.Parameter(torch.from_numpy(pos.reshape(1, hw * hw, cfg.hidden_size).astype(np.float32)))

        self.t_embedder = TimestepEmbedder(cfg.hidden_size)

        # Reference token encoder (keypoint mode): patchify (possibly downsampled) ref latent.
        self.ref_embedder = None
        self.ref_pos_embed = None
        self.ref_null_tokens = None
        self.ref_hw = 0
        self.ref_face_embedder = None
        self.ref_face_pos_embed = None
        self.ref_face_null_tokens = None
        self.ref_face_hw = 0
        if self.use_ref_tokens:
            ref_latent_size = max(1, cfg.input_size // self.ref_spatial_downsample)
            self.ref_hw = max(1, ref_latent_size // cfg.patch_size)
            self.ref_embedder = PatchEmbed(cfg.patch_size, cfg.hidden_size, cfg.in_channels)
            ref_pos = _get_interpolated_pos_embed(
                cfg.hidden_size, self.ref_hw, int(cfg.image_resolution) // self.ref_spatial_downsample
            )
            self.ref_pos_embed = nn.Parameter(
                torch.from_numpy(ref_pos.reshape(1, self.ref_hw * self.ref_hw, cfg.hidden_size).astype(np.float32))
            )
            self.ref_null_tokens = nn.Parameter(
                torch.empty(1, self.ref_hw * self.ref_hw, cfg.hidden_size)
            )
            nn.init.normal_(self.ref_null_tokens, std=cfg.hidden_size ** -0.5)

        if self.use_ref_face_tokens:
            self.ref_face_hw = max(1, self.ref_face_size // cfg.patch_size)
            self.ref_face_embedder = PatchEmbed(cfg.patch_size, cfg.hidden_size, cfg.in_channels)
            # Treat face crop as a small square "image" at native resolution of ref_face_size * 8.
            face_img_res = max(self.ref_face_size * 8, 64)
            face_pos = _get_interpolated_pos_embed(cfg.hidden_size, self.ref_face_hw, face_img_res)
            self.ref_face_pos_embed = nn.Parameter(
                torch.from_numpy(
                    face_pos.reshape(1, self.ref_face_hw * self.ref_face_hw, cfg.hidden_size).astype(np.float32)
                )
            )
            self.ref_face_null_tokens = nn.Parameter(
                torch.empty(1, self.ref_face_hw * self.ref_face_hw, cfg.hidden_size)
            )
            nn.init.normal_(self.ref_face_null_tokens, std=cfg.hidden_size ** -0.5)

        self.keypoint_embedder = None
        if self.use_keypoint_conditioning:
            self.keypoint_embedder = KeypointDeltaEmbedder(
                num_keypoints=self.num_keypoints,
                hidden_size=cfg.hidden_size,
                drop_prob=cfg.drop_text_prob,
            )
            n_ref = self.ref_hw * self.ref_hw if self.use_ref_tokens else 0
            n_face = self.ref_face_hw * self.ref_face_hw if self.use_ref_face_tokens else 0
            cond_tokens = n_ref + n_face + self.num_keypoints
            self.param_embedder = None
            self.text_encoder_adapter = self.keypoint_embedder
        elif self.use_numeric_conditioning:
            cond_tokens = cfg.num_params
            self.param_embedder = ParamEmbedder(
                num_params=cfg.num_params,
                hidden_size=cfg.hidden_size,
                drop_prob=cfg.drop_text_prob,
            )
            self.text_encoder_adapter = self.param_embedder
        else:
            cond_tokens = cfg.text_num_tokens
            self.param_embedder = None
            if cfg.text_encoder_adapter_type != "transformer":
                raise ValueError("Only the transformer text adapter (used by i1) is supported.")
            self.text_encoder_adapter = TextEncoderAdapterTransformer(
                cfg.text_embed_dim,
                cfg.hidden_size,
                cfg.drop_text_prob,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.use_qknorm,
                cfg.use_swiglu,
                cfg.use_rmsnorm,
                cfg.text_num_tokens,
                num_blocks=cfg.text_encoder_adapter_num_blocks,
            )
        self.cond_num_tokens = cond_tokens

        head_dim = cfg.hidden_size // cfg.num_heads
        axes_dims = cfg.rope_axes_dims or _default_rope_axes_dims(head_dim)
        if sum(axes_dims) != head_dim:
            raise ValueError(f"Sum of rope_axes_dims ({axes_dims}) must equal head_dim={head_dim}")
        # Keypoint RoPE needs time axis for: coarse(0) + face(1) + keypoints(2..K+1) + image(K+2)
        if self.use_keypoint_rope:
            time_len = 2 + self.num_keypoints + 1
        else:
            time_len = max(cond_tokens, 1) + 1
        axes_lens = cfg.rope_axes_lens or (time_len, hw, hw)
        if axes_lens[0] < time_len:
            axes_lens = (time_len, axes_lens[1], axes_lens[2])
        image_scale = 256.0 / cfg.image_resolution
        self.rope_embedder = MultimodalRopeEmbedder(
            axes_dims, axes_lens, (1.0, image_scale, image_scale), theta=cfg.rope_theta
        )
        self.register_buffer("image_row_ids", torch.repeat_interleave(torch.arange(hw), hw), persistent=False)
        self.register_buffer("image_col_ids", torch.tile(torch.arange(hw), (hw,)), persistent=False)
        if self.use_ref_tokens and self.ref_hw > 0:
            self.register_buffer(
                "ref_row_ids",
                torch.repeat_interleave(torch.arange(self.ref_hw), self.ref_hw),
                persistent=False,
            )
            self.register_buffer(
                "ref_col_ids",
                torch.tile(torch.arange(self.ref_hw), (self.ref_hw,)),
                persistent=False,
            )
        if self.use_ref_face_tokens and self.ref_face_hw > 0:
            self.register_buffer(
                "ref_face_row_ids",
                torch.repeat_interleave(torch.arange(self.ref_face_hw), self.ref_face_hw),
                persistent=False,
            )
            self.register_buffer(
                "ref_face_col_ids",
                torch.tile(torch.arange(self.ref_face_hw), (self.ref_face_hw,)),
                persistent=False,
            )

        def block(use_skip=False):
            return DualStreamDiTBlock(
                cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio,
                cfg.use_qknorm, cfg.use_swiglu, cfg.use_rmsnorm,
                cfg.use_sandwich_norm, cfg.use_separate_norms, use_skip=use_skip,
            )

        if cfg.use_long_skip:
            num_in_blocks = cfg.depth // 2
            self.in_blocks = nn.ModuleList([block() for _ in range(num_in_blocks)])
            self.mid_block = block()
            self.out_blocks = nn.ModuleList([block(use_skip=True) for _ in range(num_in_blocks)])
            self.blocks = None
        else:
            self.in_blocks = self.mid_block = self.out_blocks = None
            self.blocks = nn.ModuleList([block() for _ in range(cfg.depth)])

        if self.use_adaln:
            self.final_layer = FinalLayerAdaLN(cfg.hidden_size, cfg.patch_size, self.out_channels, cfg.use_rmsnorm)
        else:
            self.final_layer = FinalLayerNoAdaLN(cfg.hidden_size, cfg.patch_size, self.out_channels, cfg.use_rmsnorm)

        # Optional VAE-REPA projector (aligned to VAE latent channels * spatial tokens via conv).
        self.repa_projector = None
        if bool(getattr(cfg, "use_repa", False)):
            # Project mid image tokens (H) -> VAE feature dim (in_channels) with a 1x1-style MLP,
            # then reshape handled outside. Keep a small conv over token grid.
            self.repa_projector = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size, cfg.in_channels),
            )

    def init_weights(self) -> None:
        trunc_std_correction = 0.8796256610342398

        def lecun_normal_(weight: torch.Tensor, fan_in: int) -> None:
            std = math.sqrt(1.0 / fan_in) / trunc_std_correction
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                lecun_normal_(module.weight, module.in_features)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                lecun_normal_(module.weight, fan_in)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (RMSNorm, LayerNorm)):
                nn.init.ones_(module.scale)
                if isinstance(module, LayerNorm):
                    nn.init.zeros_(module.bias)

        if self.use_keypoint_conditioning and self.keypoint_embedder is not None:
            nn.init.normal_(self.keypoint_embedder.learnable_null, std=self.config.hidden_size ** -0.5)
            nn.init.normal_(self.keypoint_embedder.point_id_embed.weight, std=0.02)
            if self.ref_null_tokens is not None:
                nn.init.normal_(self.ref_null_tokens, std=self.config.hidden_size ** -0.5)
            if self.ref_face_null_tokens is not None:
                nn.init.normal_(self.ref_face_null_tokens, std=self.config.hidden_size ** -0.5)
        elif self.use_numeric_conditioning:
            nn.init.normal_(self.param_embedder.learnable_null_caption, std=self.config.hidden_size ** -0.5)
            nn.init.normal_(self.param_embedder.param_id_embed.weight, std=0.02)
            nn.init.eye_(self.param_embedder.connector_in.weight)
            self.param_embedder.connector_in.weight.requires_grad_(False)
        else:
            nn.init.xavier_uniform_(self.text_encoder_adapter.connector_in.weight)
            nn.init.zeros_(self.text_encoder_adapter.connector_in.bias)
            in_channels = self.text_encoder_adapter.learnable_null_caption.shape[-1]
            nn.init.normal_(self.text_encoder_adapter.learnable_null_caption, std=in_channels ** -0.5)

        for blk in (self.out_blocks or []):
            nn.init.xavier_uniform_(blk.skip_linear_image.weight)
            nn.init.zeros_(blk.skip_linear_image.bias)
            nn.init.xavier_uniform_(blk.skip_linear_text.weight)
            nn.init.zeros_(blk.skip_linear_text.bias)

        # Zero-init adaLN modulation so the model starts as FinalLayerNoAdaLN.
        if self.use_adaln and isinstance(self.final_layer, FinalLayerAdaLN):
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)

    def _build_position_ids(self, text_mask: torch.Tensor, text_lengths: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
        bsz, text_len = text_mask.shape
        caption_positions = torch.arange(text_len, dtype=torch.long, device=text_mask.device)[None].expand(bsz, text_len)
        caption_positions = torch.where(text_mask.bool(), caption_positions, torch.zeros_like(caption_positions))
        zeros = torch.zeros_like(caption_positions)
        caption_ids = torch.stack((caption_positions, zeros, zeros), dim=-1)
        row_ids = self.image_row_ids[:num_image_tokens][None].expand(bsz, num_image_tokens)
        col_ids = self.image_col_ids[:num_image_tokens][None].expand(bsz, num_image_tokens)
        image_time = text_lengths[:, None].expand(bsz, num_image_tokens)
        image_ids = torch.stack((image_time, row_ids, col_ids), dim=-1)
        return torch.cat([caption_ids, image_ids], dim=1)

    def _cond_position_ids(self, bsz: int, device: torch.device, num_image_tokens: int) -> torch.Tensor:
        """2-D RoPE positions for keypoint mode second-stream + image tokens.

        Layout:
          coarse ref: time=0, row/col scaled to image grid
          face ref:   time=1, row/col scaled to image grid
          keypoints:  time=2..K+1, row/col=0
          image:      time=K+2, row/col=image grid
        """
        pieces = []
        if self.use_ref_tokens and self.ref_hw > 0:
            n = self.ref_hw * self.ref_hw
            scale = float(self.hw) / float(max(self.ref_hw, 1))
            rows = (self.ref_row_ids.float() * scale).long().clamp(0, self.hw - 1)
            cols = (self.ref_col_ids.float() * scale).long().clamp(0, self.hw - 1)
            t0 = torch.zeros(bsz, n, dtype=torch.long, device=device)
            pieces.append(torch.stack((t0, rows[None].expand(bsz, -1), cols[None].expand(bsz, -1)), dim=-1))
        if self.use_ref_face_tokens and self.ref_face_hw > 0:
            n = self.ref_face_hw * self.ref_face_hw
            scale = float(self.hw) / float(max(self.ref_face_hw, 1))
            rows = (self.ref_face_row_ids.float() * scale).long().clamp(0, self.hw - 1)
            cols = (self.ref_face_col_ids.float() * scale).long().clamp(0, self.hw - 1)
            t1 = torch.ones(bsz, n, dtype=torch.long, device=device)
            pieces.append(torch.stack((t1, rows[None].expand(bsz, -1), cols[None].expand(bsz, -1)), dim=-1))
        # Keypoint tokens
        kp_t = torch.arange(2, 2 + self.num_keypoints, dtype=torch.long, device=device)[None].expand(bsz, -1)
        zeros = torch.zeros(bsz, self.num_keypoints, dtype=torch.long, device=device)
        pieces.append(torch.stack((kp_t, zeros, zeros), dim=-1))

        cond_ids = torch.cat(pieces, dim=1)
        img_time = 2 + self.num_keypoints
        row_ids = self.image_row_ids[:num_image_tokens][None].expand(bsz, num_image_tokens)
        col_ids = self.image_col_ids[:num_image_tokens][None].expand(bsz, num_image_tokens)
        t_img = torch.full((bsz, num_image_tokens), img_time, dtype=torch.long, device=device)
        image_ids = torch.stack((t_img, row_ids, col_ids), dim=-1)
        return torch.cat([cond_ids, image_ids], dim=1)

    def _rope_freqs(self, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor], num_image_tokens: int):
        seq_text = text_tokens.shape[1]
        if self.use_keypoint_rope:
            position_ids = self._cond_position_ids(text_tokens.shape[0], text_tokens.device, num_image_tokens)
            cos, sin = self.rope_embedder(position_ids)
            text_freqs = (cos[:, :seq_text], sin[:, :seq_text])
            image_freqs = (cos[:, seq_text:seq_text + num_image_tokens], sin[:, seq_text:seq_text + num_image_tokens])
            return image_freqs, text_freqs
        pos_mask = (
            text_mask if text_mask is not None
            else torch.ones((text_tokens.shape[0], seq_text), dtype=torch.bool, device=text_tokens.device)
        )
        text_lengths = pos_mask.to(torch.int32).sum(dim=1)
        position_ids = self._build_position_ids(pos_mask, text_lengths, num_image_tokens)
        cos, sin = self.rope_embedder(position_ids)
        text_freqs = (cos[:, :seq_text], sin[:, :seq_text])
        image_freqs = (cos[:, seq_text:seq_text + num_image_tokens], sin[:, seq_text:seq_text + num_image_tokens])
        return image_freqs, text_freqs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        caption: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        train: bool = False,
        force_null: bool | torch.Tensor = False,
        ref_latent: Optional[torch.Tensor] = None,
        force_null_ref: bool | torch.Tensor = False,
        pose_map: Optional[torch.Tensor] = None,
        keypoint_deltas: Optional[torch.Tensor] = None,
        force_null_pose: bool | torch.Tensor = False,
        ref_face_latent: Optional[torch.Tensor] = None,
        return_mid_features: bool = False,
    ):
        """Forward pass.

        Keypoint mode expects:
          - pose_map: (B, num_pose_channels, H, W) spatial pose conditioning
          - keypoint_deltas: (B, num_keypoints, 4) target-minus-ref deltas
          - ref_latent: (B, C, H, W) identity reference (encoded as second-stream tokens)
          - ref_face_latent: optional (B, C, ref_face_size, ref_face_size) face crop
          - caption is unused (pass zeros or deltas; deltas preferred via keypoint_deltas)
        """
        bsz = x.shape[0]
        mid_features = None

        # --- Build patch input (noise [+ pose map] [+ legacy ref concat]) ---
        if self.use_keypoint_conditioning:
            if pose_map is None:
                raise ValueError("pose_map is required when use_keypoint_conditioning=True")
            if pose_map.shape[0] != bsz or pose_map.shape[1] != self.num_pose_channels:
                raise ValueError(
                    f"pose_map shape {tuple(pose_map.shape)} expected (B, {self.num_pose_channels}, H, W)"
                )
            if pose_map.shape[-2:] != x.shape[-2:]:
                pose_map = F.interpolate(pose_map.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
            pose = pose_map.to(dtype=x.dtype)
            if isinstance(force_null_pose, torch.Tensor):
                drop_pose = force_null_pose
            else:
                drop_pose = torch.full((bsz,), bool(force_null_pose), dtype=torch.bool, device=x.device)
            if train and self.drop_pose_prob > 0:
                drop_pose = drop_pose | (torch.rand(bsz, device=x.device) < self.drop_pose_prob)
            pose = torch.where(drop_pose[:, None, None, None], torch.zeros_like(pose), pose)
            x_in = torch.cat([x, pose], dim=1)
        elif self.use_ref_conditioning and not self.use_ref_tokens:
            if ref_latent is None:
                raise ValueError("ref_latent is required when use_ref_conditioning=True")
            if ref_latent.shape != x.shape:
                raise ValueError(
                    f"ref_latent shape {tuple(ref_latent.shape)} must match x {tuple(x.shape)}"
                )
            ref = ref_latent.to(dtype=x.dtype)
            if isinstance(force_null_ref, torch.Tensor):
                drop_ref = force_null_ref
            else:
                drop_ref = torch.full((bsz,), bool(force_null_ref), dtype=torch.bool, device=x.device)
            if train and self.drop_ref_prob > 0:
                drop_ref = drop_ref | (torch.rand(bsz, device=x.device) < self.drop_ref_prob)
            ref = torch.where(drop_ref[:, None, None, None], torch.zeros_like(ref), ref)
            x_in = torch.cat([x, ref], dim=1)
        else:
            x_in = x

        tokens = self.x_embedder(x_in) + self.pos_embed.to(dtype=x.dtype)
        t_emb = self.t_embedder(t).to(dtype=x.dtype)

        # --- Second-stream conditioning tokens ---
        if self.use_keypoint_conditioning:
            deltas = keypoint_deltas if keypoint_deltas is not None else caption
            if deltas is None:
                raise ValueError("keypoint_deltas (or caption) required in keypoint mode")
            if deltas.ndim == 2:
                deltas = deltas.view(bsz, self.num_keypoints, -1)
            kp_tokens = self.keypoint_embedder(deltas, train=train, force_null=force_null)

            stream_parts = []
            if self.use_ref_tokens:
                if ref_latent is None:
                    raise ValueError("ref_latent is required for keypoint ref-token mode")
                ref = ref_latent.to(dtype=x.dtype)
                if isinstance(force_null_ref, torch.Tensor):
                    drop_ref = force_null_ref
                else:
                    drop_ref = torch.full((bsz,), bool(force_null_ref), dtype=torch.bool, device=x.device)
                if train and self.drop_ref_prob > 0:
                    drop_ref = drop_ref | (torch.rand(bsz, device=x.device) < self.drop_ref_prob)
                if self.ref_spatial_downsample > 1:
                    ref_ds = F.avg_pool2d(
                        ref, kernel_size=self.ref_spatial_downsample, stride=self.ref_spatial_downsample
                    )
                else:
                    ref_ds = ref
                ref_tokens = self.ref_embedder(ref_ds) + self.ref_pos_embed.to(dtype=x.dtype)
                null_ref = self.ref_null_tokens.to(dtype=ref_tokens.dtype).expand(bsz, -1, -1)
                ref_tokens = torch.where(drop_ref[:, None, None], null_ref, ref_tokens)
                stream_parts.append(ref_tokens)

                if self.use_ref_face_tokens:
                    if ref_face_latent is None:
                        # Fallback: center-crop + resize from full ref
                        face = F.interpolate(
                            ref.float(), size=(self.ref_face_size, self.ref_face_size),
                            mode="bilinear", align_corners=False,
                        ).to(dtype=x.dtype)
                    else:
                        face = ref_face_latent.to(dtype=x.dtype)
                        if face.shape[-2:] != (self.ref_face_size, self.ref_face_size):
                            face = F.interpolate(
                                face.float(), size=(self.ref_face_size, self.ref_face_size),
                                mode="bilinear", align_corners=False,
                            ).to(dtype=x.dtype)
                    face_tokens = self.ref_face_embedder(face) + self.ref_face_pos_embed.to(dtype=x.dtype)
                    null_face = self.ref_face_null_tokens.to(dtype=face_tokens.dtype).expand(bsz, -1, -1)
                    # Share identity CFG drop with coarse ref
                    face_tokens = torch.where(drop_ref[:, None, None], null_face, face_tokens)
                    stream_parts.append(face_tokens)

            stream_parts.append(kp_tokens)
            text_tokens = torch.cat(stream_parts, dim=1)
            text_mask = torch.ones(
                (text_tokens.shape[0], text_tokens.shape[1]),
                dtype=torch.bool,
                device=text_tokens.device,
            )
            pooled = kp_tokens.mean(dim=1)
        elif self.use_numeric_conditioning:
            text_tokens = self.param_embedder(caption, train=train, force_null=force_null)
            text_mask = torch.ones(
                (text_tokens.shape[0], text_tokens.shape[1]),
                dtype=torch.bool,
                device=text_tokens.device,
            )
            pooled = self.param_embedder.pool(text_tokens)
        else:
            text_mask = mask.bool() if mask is not None else None
            text_tokens = self.text_encoder_adapter(caption, train=train)
            pooled = text_tokens.mean(dim=1)

        cond = t_emb + pooled.to(dtype=t_emb.dtype)
        image_freqs, text_freqs = self._rope_freqs(text_tokens, text_mask, tokens.shape[1])

        use_ckpt = self.use_grad_ckpt and train and torch.is_grad_enabled()

        def run_block(blk, img, txt, skip=None):
            if not use_ckpt:
                return blk(img, txt, image_freqs, text_freqs, text_mask, skip)
            if skip is None:
                def fn(a, b):
                    return blk(a, b, image_freqs, text_freqs, text_mask)
                return torch.utils.checkpoint.checkpoint(fn, img, txt, use_reentrant=False)
            def fn(a, b, s0, s1):
                return blk(a, b, image_freqs, text_freqs, text_mask, (s0, s1))
            return torch.utils.checkpoint.checkpoint(fn, img, txt, skip[0], skip[1], use_reentrant=False)

        image_tokens = tokens
        if self.blocks is None:
            skips = []
            for blk in self.in_blocks:
                image_tokens, text_tokens = run_block(blk, image_tokens, text_tokens)
                skips.append((image_tokens, text_tokens))
            image_tokens, text_tokens = run_block(self.mid_block, image_tokens, text_tokens)
            if return_mid_features:
                mid_features = image_tokens
            for blk in self.out_blocks:
                image_tokens, text_tokens = run_block(blk, image_tokens, text_tokens, skips.pop())
        else:
            mid_idx = len(self.blocks) // 2
            for i, blk in enumerate(self.blocks):
                image_tokens, text_tokens = run_block(blk, image_tokens, text_tokens)
                if return_mid_features and i == mid_idx:
                    mid_features = image_tokens

        out_tokens = self.final_layer(image_tokens, cond if self.use_adaln else None)
        h = w = self.input_size // self.patch_size
        p = self.patch_size
        out_tokens = out_tokens.reshape(bsz, h, w, p, p, self.out_channels)
        out_tokens = out_tokens.permute(0, 1, 3, 2, 4, 5).reshape(bsz, h * p, w * p, self.out_channels)
        pred = out_tokens.permute(0, 3, 1, 2)
        if return_mid_features:
            return pred, mid_features
        return pred


def build_dit_model(config, latent_size: int, text_embed_dim: int, text_num_tokens: int) -> i1DiT:
    preset = DualStreamDiT_models[config.model_size]
    model_kwargs = dict(config.model_kwargs)
    use_numeric = bool(config.get("use_numeric_conditioning", False))
    use_keypoint = bool(config.get("use_keypoint_conditioning", False))
    num_params = int(config.get("num_params", text_num_tokens if use_numeric else 16))
    num_keypoints = int(config.get("num_keypoints", 37))
    num_pose_channels = int(config.get("num_pose_channels", 8))
    cfg = DualStreamDiTConfig(
        input_size=latent_size,
        image_resolution=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        text_embed_dim=text_embed_dim,
        text_num_tokens=(
            num_keypoints if use_keypoint else (num_params if use_numeric else text_num_tokens)
        ),
        use_qknorm=config.use_qknorm,
        use_swiglu=config.use_swiglu,
        use_rmsnorm=config.use_rmsnorm,
        depth=preset["depth"],
        hidden_size=preset["hidden_size"],
        num_heads=preset["num_heads"],
        mlp_ratio=preset["mlp_ratio"],
        use_long_skip=model_kwargs.get("use_long_skip", True),
        text_encoder_adapter_type=model_kwargs.get("text_encoder_adapter_type", "transformer"),
        text_encoder_adapter_num_blocks=model_kwargs.get("text_encoder_adapter_num_blocks", 2),
        use_sandwich_norm=model_kwargs.get("use_sandwich_norm", True),
        use_separate_norms=model_kwargs.get("use_separate_norms", False),
        use_grad_ckpt=getattr(config, "use_grad_ckpt", False),
        rope_theta=model_kwargs.get("rope_theta", 10000.0),
        rope_axes_dims=model_kwargs.get("rope_axes_dims", None),
        rope_axes_lens=model_kwargs.get("rope_axes_lens", None),
        use_numeric_conditioning=use_numeric and not use_keypoint,
        num_params=num_params,
        use_adaln=bool(config.get("use_adaln", model_kwargs.get("use_adaln", False))),
        drop_text_prob=float(config.get("drop_text_prob", 0.1)),
        use_ref_conditioning=bool(config.get("use_ref_conditioning", False)) or use_keypoint,
        drop_ref_prob=float(config.get("drop_ref_prob", 0.1)),
        use_keypoint_conditioning=use_keypoint,
        num_keypoints=num_keypoints,
        num_pose_channels=num_pose_channels,
        drop_pose_prob=float(config.get("drop_pose_prob", 0.1)),
        use_ref_tokens=bool(config.get("use_ref_tokens", True)),
        ref_spatial_downsample=int(config.get("ref_spatial_downsample", 2)),
        use_ref_face_tokens=bool(config.get("use_ref_face_tokens", False)),
        ref_face_size=int(config.get("ref_face_size", 32)),
        use_keypoint_rope=bool(config.get("use_keypoint_rope", True)),
        use_repa=bool(config.get("use_repa", False)),
    )
    return i1DiT(cfg)
