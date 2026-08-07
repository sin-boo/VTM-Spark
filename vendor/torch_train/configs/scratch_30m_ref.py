"""From-scratch ~30M DiT with numeric pose + reference-image conditioning.

Requires:
  - TFRecords with character_id (scripts/build_tfrecords.py --sessions-root ...)
  - Latent cache with character_ids (scripts/cache_latents.py)

Paths relative to send2pod / override with SEND2POD_ROOT.
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
    data_dir = str(root / "data" / "train_512")
    latent_cache = root / "data" / "train_512" / "latents_cache.pt"

    config = ConfigDict()

    config.seed = 0
    config.total_steps = 100000
    config.resume = ""
    config.warmup_steps = 1000

    config.image_size = 512
    config.use_fast_stack = True
    config.use_numeric_conditioning = True
    config.use_ref_conditioning = True
    config.use_adaln = True
    config.num_params = NUM_PARAMS
    config.drop_text_prob = 0.1
    config.drop_ref_prob = 0.1
    config.flip_prob = 0.5
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
    # Override on pod with --batch_size if VRAM is tight.
    config.input.batch_size = 64
    config.input.shuffle_buffer_size = 1024
    config.input.preprocess = (
        "|decode_png()"
        f"|ensure_params(num_params={NUM_PARAMS})"
        '|sample_caption(key="caption")'
        '|copy("caption", "labels")'
        "|value_range(-1, 1)"
        '|keep("image", "params", "labels", "character_id")'
    )

    config.latent_cache = str(latent_cache)

    config.grad_accum_steps = 1
    config.log_training_steps = 20
    config.ckpt_steps = 5000
    config.keep_ckpt_steps = 5000
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
            project="i1-scratch-ref",
            experiment="ref_30m_train_512",
        )
    )
    return config
