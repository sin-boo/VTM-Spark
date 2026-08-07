"""From-scratch ~30M DiT on character_1 with numeric param conditioning (SD-VAE).

No CLIP / T5 — Live2D floats go straight into ParamEmbedder.
Paths are relative to send2pod / override with SEND2POD_ROOT.
"""

import os
from pathlib import Path

from utils.config import ConfigDict
from utils.params import NUM_PARAMS


def _root() -> Path:
    env = os.environ.get("SEND2POD_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def get_config():
    root = _root()
    data_dir = str(root / "data" / "character_1_512")
    latent_cache = root / "data" / "character_1_512" / "latents_cache.pt"

    config = ConfigDict()

    config.seed = 0
    config.total_steps = 30000
    config.resume = ""
    config.warmup_steps = 500

    config.image_size = 512
    config.use_fast_stack = True
    config.use_numeric_conditioning = True
    config.use_adaln = True
    config.num_params = NUM_PARAMS
    config.drop_text_prob = 0.1
    config.vae_type = "sd"
    config.in_channels = 4
    config.text_encoder_type = None
    config.text_encoder_precision = "bf16"
    config.token_len = NUM_PARAMS
    config.backbone = "dual_stream"
    config.model_size = "DiT-30M"
    config.patch_size = 2
    config.model_kwargs = ConfigDict(
        dict(
            rope_axes_dims=None,
            rope_axes_lens=None,
            rope_theta=10000.0,
            use_long_skip=True,
            text_encoder_adapter_type="transformer",
            text_encoder_adapter_num_blocks=0,
            use_image_connector=False,
            use_adaln=True,
            repeat_text_emb=False,
            position_embedding="sinusoidal_and_rope",
            use_sandwich_norm=True,
            use_separate_norms=False,
        )
    )
    config.use_qknorm = True
    config.use_swiglu = True
    config.use_rmsnorm = True
    config.use_grad_ckpt = True
    config.amp = True

    config.transport = ConfigDict(
        dict(
            prediction="velocity",
            use_lognorm=True,
            lognorm_mu=0.0,
            lognorm_sigma=1.0,
            train_timestep_shift=0.0,
            cfg_interval_start=0,
        )
    )

    config.tensor_parallel_size = 1
    config.fsdp_axis_size = 1

    config.input = ConfigDict()
    config.input.data = [(dict(split="train", data_dir=data_dir), 1.0)]
    # Fast 2x5090 default (override with --batch_size). Global batch; split across ranks.
    config.input.batch_size = 256
    config.input.shuffle_buffer_size = 512
    config.input.preprocess = (
        "|decode_png()"
        f"|ensure_params(num_params={NUM_PARAMS})"
        '|sample_caption(key="caption")'
        '|copy("caption", "labels")'
        "|value_range(-1, 1)"
        '|keep("image", "params", "labels")'
    )

    # Prefer latent cache when present (created by scripts/cache_latents.py).
    config.latent_cache = str(latent_cache) if latent_cache.is_file() else str(latent_cache)

    config.grad_accum_steps = 1
    config.log_training_steps = 20
    config.ckpt_steps = 500
    config.keep_ckpt_steps = 2000
    config.save_ckpt = True

    config.grad_clip_norm = 1.0
    config.b1 = 0.9
    config.b2 = 0.95
    config.adam_eps = 1e-8
    config.mu_dtype = "bfloat16"
    config.lr = 1e-4
    config.freeze_patterns = []

    config.use_ema = True
    config.ema_decay_rate = 0.999

    config.wandb = ConfigDict(
        dict(
            log_wandb=False,
            project="i1-scratch-numeric",
            experiment="character_1_30m_runpod",
        )
    )
    return config
