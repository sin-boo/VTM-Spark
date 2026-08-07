"""Count DualStreamDiT params for fast / numeric / keypoint presets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dit import DualStreamDiTConfig, DualStreamDiT_models, i1DiT
from utils.keypoints import NUM_KEYPOINTS, NUM_POSE_CHANNELS
from utils.params import NUM_PARAMS


def count_numeric(depth, hidden, heads):
    cfg = DualStreamDiTConfig(
        input_size=64,
        patch_size=2,
        in_channels=4,
        hidden_size=hidden,
        depth=depth,
        num_heads=heads,
        mlp_ratio=4.0,
        image_resolution=512,
        text_embed_dim=hidden,
        text_num_tokens=NUM_PARAMS,
        use_numeric_conditioning=True,
        num_params=NUM_PARAMS,
        use_adaln=True,
        use_long_skip=True,
        use_sandwich_norm=True,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
    )
    model = i1DiT(cfg)
    model.init_weights()
    return sum(p.numel() for p in model.parameters())


def count_keypoint(depth, hidden, heads, image_size=768):
    latent = image_size // 8
    cfg = DualStreamDiTConfig(
        input_size=latent,
        patch_size=2,
        in_channels=4,
        hidden_size=hidden,
        depth=depth,
        num_heads=heads,
        mlp_ratio=4.0,
        image_resolution=image_size,
        text_embed_dim=hidden,
        text_num_tokens=NUM_KEYPOINTS,
        use_keypoint_conditioning=True,
        use_ref_tokens=True,
        ref_spatial_downsample=2,
        num_keypoints=NUM_KEYPOINTS,
        num_pose_channels=NUM_POSE_CHANNELS,
        use_adaln=True,
        use_long_skip=True,
        use_sandwich_norm=True,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        use_repa=True,
    )
    model = i1DiT(cfg)
    model.init_weights()
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    preset = DualStreamDiT_models["DiT-30M"]
    n = count_numeric(preset["depth"], preset["hidden_size"], preset["num_heads"])
    print(f"DiT-30M numeric {preset} -> {n/1e6:.2f}M params")
    preset100 = DualStreamDiT_models["DiT-100M"]
    n100 = count_keypoint(preset100["depth"], preset100["hidden_size"], preset100["num_heads"])
    print(f"DiT-100M keypoint {preset100} @768 -> {n100/1e6:.2f}M params")
    n30k = count_keypoint(preset["depth"], preset["hidden_size"], preset["num_heads"], image_size=512)
    print(f"DiT-30M keypoint @512 -> {n30k/1e6:.2f}M params")
