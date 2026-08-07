"""Resolve quality vs fast (SD-VAE + CLIP) vs numeric / keypoint stacks."""

from __future__ import annotations

from typing import TypedDict


class StackSpec(TypedDict):
    vae_type: str
    text_encoder_type: str | None
    in_channels: int
    text_embed_dim: int
    token_len: int
    use_numeric_conditioning: bool
    use_keypoint_conditioning: bool


QUALITY_STACK: StackSpec = {
    "vae_type": "flux2",
    "text_encoder_type": "T5Gemma",
    "in_channels": 32,
    "text_embed_dim": 2304,
    "token_len": 256,
    "use_numeric_conditioning": False,
    "use_keypoint_conditioning": False,
}

FAST_STACK: StackSpec = {
    "vae_type": "sd",
    "text_encoder_type": "CLIP",
    "in_channels": 4,
    "text_embed_dim": 768,
    "token_len": 77,
    "use_numeric_conditioning": False,
    "use_keypoint_conditioning": False,
}

# Fast VAE + direct float param conditioning (no text encoder).
NUMERIC_FAST_STACK: StackSpec = {
    "vae_type": "sd",
    "text_encoder_type": None,
    "in_channels": 4,
    "text_embed_dim": 320,  # matches DiT-30M hidden; connector_in reports this
    "token_len": 16,
    "use_numeric_conditioning": True,
    "use_keypoint_conditioning": False,
}

# Keypoint pose-map + ref-token identity (no text encoder).
KEYPOINT_STACK: StackSpec = {
    "vae_type": "sd",
    "text_encoder_type": None,
    "in_channels": 4,
    "text_embed_dim": 320,  # matches DiT-30M hidden by default
    "token_len": 37,
    "use_numeric_conditioning": False,
    "use_keypoint_conditioning": True,
}


def resolve_stack(
    use_fast_stack: bool = False,
    use_numeric_conditioning: bool = False,
    use_keypoint_conditioning: bool = False,
) -> StackSpec:
    """Map flags -> VAE/text-encoder types and DiT I/O dims."""
    if use_keypoint_conditioning:
        return dict(KEYPOINT_STACK)
    if use_numeric_conditioning:
        return dict(NUMERIC_FAST_STACK)
    return dict(FAST_STACK if use_fast_stack else QUALITY_STACK)


def apply_stack_to_config(config, use_fast_stack: bool | None = None):
    """Set vae_type / text_encoder_type / in_channels / token_len on a train config."""
    if use_fast_stack is None:
        use_fast_stack = bool(config.get("use_fast_stack", False))
    else:
        config.use_fast_stack = bool(use_fast_stack)
    use_numeric = bool(config.get("use_numeric_conditioning", False))
    use_keypoint = bool(config.get("use_keypoint_conditioning", False))
    stack = resolve_stack(
        bool(config.get("use_fast_stack", False)),
        use_numeric_conditioning=use_numeric and not use_keypoint,
        use_keypoint_conditioning=use_keypoint,
    )
    config.vae_type = stack["vae_type"]
    config.text_encoder_type = stack["text_encoder_type"]
    # VAE latent channels stay at stack default (4 for SD). Pose-map / ref concat
    # is handled inside the model; do not inflate in_channels here.
    config.in_channels = stack["in_channels"]
    if use_keypoint:
        config.token_len = int(config.get("num_keypoints", stack["token_len"]))
        config.num_keypoints = int(config.get("num_keypoints", stack["token_len"]))
        config.use_numeric_conditioning = False
    elif use_numeric:
        config.token_len = int(config.get("num_params", stack["token_len"]))
        config.num_params = int(config.get("num_params", stack["token_len"]))
    elif config.get("token_len") is None:
        config.token_len = stack["token_len"]
    return stack
