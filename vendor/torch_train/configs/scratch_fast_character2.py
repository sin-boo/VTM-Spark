"""From-scratch ~2M DiT on character_2 with fast stack (SD-VAE + CLIP).

Paths are relative to send2pod / override with SEND2POD_ROOT.
"""

import os
from pathlib import Path

from utils.config import ConfigDict


def _root() -> Path:
    env = os.environ.get("SEND2POD_ROOT")
    if env:
        return Path(env).resolve()
    # send2pod/torch_train/configs/this_file.py -> send2pod/
    return Path(__file__).resolve().parents[2]


def get_config():
    root = _root()
    data_dir = str(root / "data" / "character_2_512")

    config = ConfigDict()

    config.seed = 0
    config.total_steps = 10000
    config.resume = ""

    config.image_size = 512
    config.use_fast_stack = True
    config.vae_type = "sd"
    config.in_channels = 4
    config.text_encoder_type = "CLIP"
    config.text_encoder_precision = "bf16"
    config.token_len = None
    config.backbone = "dual_stream"
    config.model_size = "DiT-2M"
    config.patch_size = 2
    config.model_kwargs = ConfigDict(
        dict(
            rope_axes_dims=None,
            rope_axes_lens=None,
            rope_theta=10000.0,
            use_long_skip=True,
            text_encoder_adapter_type="transformer",
            text_encoder_adapter_num_blocks=2,
            use_image_connector=False,
            use_adaln=False,
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
    config.input.batch_size = 8
    config.input.shuffle_buffer_size = 256
    config.input.preprocess = (
        "|decode_png()"
        '|sample_caption(key="caption")'
        "|value_range(-1, 1)"
        '|copy("caption", "labels")'
        '|keep("image", "labels")'
    )

    config.grad_accum_steps = 8
    config.log_training_steps = 20
    config.ckpt_steps = 500
    config.keep_ckpt_steps = 1000
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
            project="i1-scratch-fast",
            experiment="character_2_2m_runpod",
        )
    )
    return config
