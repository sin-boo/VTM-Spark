"""From-scratch DiT with keypoint pose maps + reference-token identity.

Data prep (user runs when ready)::

  python -m scripts.remove_original_samples --raw-root ../data/raw
  python -m scripts.build_dataset --raw-root ../data/raw --stats-only
  python -m scripts.build_dataset --raw-root ../data/raw --out-dir ../data/train_crop \\
      --out-size 768 768 --aspect 1.0 --margin-x 32
  python -m scripts.cache_latents --data-dir ../data/train_crop \\
      --output ../data/train_crop/latents_cache.pt --mode keypoints

Note: default preprocessing chroma-keys the character and crops left/right only
(full source height is kept), pads to a square, and resizes to 768.
Keypoints are transformed automatically — no relabel.
Pad green is sampled from each image. Use ``--skip-crop`` for full-frame padding.
Non-square (e.g. 768x1152) can be used after adding rectangular RoPE/pos support.
Non-square (e.g. 768x1152) can be used after adding rectangular RoPE/pos support.

Holdout: put a separate character subset through the same pipeline and point
``val_latent_cache`` at its cache (optional until you create holdout images).
"""

import os
from pathlib import Path

from utils.config import ConfigDict
from utils.keypoints import NUM_KEYPOINTS, NUM_POSE_CHANNELS


def _root() -> Path:
    env = os.environ.get("SEND2POD_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def get_config():
    root = _root()
    data_dir = str(root / "data" / "train_crop")
    latent_cache = root / "data" / "train_crop" / "latents_cache.pt"
    val_cache = root / "data" / "holdout_crop" / "latents_cache.pt"

    config = ConfigDict()

    config.seed = 0
    config.total_steps = 100000
    config.resume = ""
    config.warmup_steps = 1000

    # Square crop (see module docstring). Change after rectangular DiT support lands.
    config.image_size = 768
    config.use_fast_stack = True
    config.use_numeric_conditioning = False
    config.use_keypoint_conditioning = True
    config.use_ref_conditioning = True
    config.use_ref_tokens = True
    config.ref_spatial_downsample = 2
    # Dual-scale identity: coarse whole-body tokens + full-res face crop tokens.
    config.use_ref_face_tokens = True
    config.ref_face_size = 32  # latent px → 16x16 = 256 tokens at patch 2
    config.use_keypoint_rope = True
    config.use_adaln = True
    config.num_keypoints = NUM_KEYPOINTS
    config.num_pose_channels = NUM_POSE_CHANNELS
    config.drop_text_prob = 0.1
    config.drop_ref_prob = 0.1
    config.drop_pose_prob = 0.1
    config.flip_prob = 0.5
    config.pose_sigma = 1.5
    # Face-weighted velocity loss (1.0 = plain MSE).
    config.face_loss_weight = 4.0
    config.face_loss_margin = 0.15
    # Ref curriculum: prefer similar-pose refs early, decay to uniform.
    config.ref_pose_bias_start = 0.7
    config.ref_pose_bias_end = 0.0
    # Inference CFG defaults (also stored in checkpoint_config).
    config.pose_cfg_scale = 1.5
    config.id_cfg_scale = 2.0
    config.vae_type = "sd"
    config.in_channels = 4
    config.text_encoder_type = None
    config.text_encoder_precision = "bf16"
    config.token_len = NUM_KEYPOINTS
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

    # VAE-REPA (disable with use_repa=False). Stops after repa_stop_step if > 0.
    config.use_repa = True
    config.repa_weight = 0.05
    config.repa_stop_step = 40000

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
    # 4x RTX 5090 (32GB): global batch 32 is comfortable with DiT-30M + face tokens
    # (~869 second-stream tokens). Raise toward 48–64 if VRAM allows.
    config.input.batch_size = 32
    config.input.shuffle_buffer_size = 1024
    # Preprocess unused when latent_cache is present.
    config.input.preprocess = "|decode_png()|value_range(-1, 1)|keep(\"image\")"

    config.latent_cache = str(latent_cache)
    # Optional — ignored until the file exists.
    config.val_latent_cache = str(val_cache) if val_cache.is_file() else ""
    config.val_steps = 1000
    config.sample_steps = 2000

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
    # "adam" (default) or "muon" (Muon on 2D weights + Adam on rest).
    config.optimizer = "adam"
    config.muon_lr = 0.02
    config.freeze_patterns = []

    config.use_ema = True
    config.ema_decay_rate = 0.9999

    config.wandb = ConfigDict(
        dict(
            log_wandb=False,
            project="i1-scratch-keypoint",
            experiment="keypoint_30m_train_crop",
        )
    )
    return config
