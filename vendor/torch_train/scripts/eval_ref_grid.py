"""Eval grid: held-out character reference x fixed pose vectors.

Example:
  python -m scripts.eval_ref_grid \\
    --checkpoint /workspace/workdir/checkpoint.pt \\
    --holdout-dir ../data/holdout_512 \\
    --output-dir ../data/eval_grids \\
    --steps 14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tensorflow_datasets as tfds
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_numeric import (  # noqa: E402
    build_numeric_model,
    decode_sd_vae,
    denoise_numeric_turbo,
    load_param_stats,
    load_sd_vae,
)
from utils.params import NUM_PARAMS, PARAM_NAMES, normalize_params  # noqa: E402


# Fixed pose presets in RAW (unnormalized) space — remapped with param_stats.
POSE_PRESETS_RAW = {
    "neutral": {
        "ParamAngleX": 0, "ParamAngleY": 0, "ParamAngleZ": 0,
        "ParamMouthOpenY": 0, "ParamMouthForm": -0.4,
        "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
        "ParamEyeBallX": 0, "ParamEyeBallY": 0,
        "FaceAngleX": 0, "FaceAngleY": 0, "FaceAngleZ": 0,
        "MouthOpen": 0, "MouthSmile": 0.2,
        "EyeOpenLeft": 0.7, "EyeOpenRight": 0.7,
    },
    "smile": {
        "ParamAngleX": 0, "ParamAngleY": 0, "ParamAngleZ": 0,
        "ParamMouthOpenY": 0.1, "ParamMouthForm": 0.8,
        "ParamEyeLOpen": 0.9, "ParamEyeROpen": 0.9,
        "ParamEyeBallX": 0, "ParamEyeBallY": 0,
        "FaceAngleX": 0, "FaceAngleY": 0, "FaceAngleZ": 0,
        "MouthOpen": 0.05, "MouthSmile": 0.95,
        "EyeOpenLeft": 0.65, "EyeOpenRight": 0.65,
    },
    "mouth_open": {
        "ParamAngleX": 0, "ParamAngleY": 0, "ParamAngleZ": 0,
        "ParamMouthOpenY": 0.9, "ParamMouthForm": 0.0,
        "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
        "ParamEyeBallX": 0, "ParamEyeBallY": 0,
        "FaceAngleX": 0, "FaceAngleY": 0, "FaceAngleZ": 0,
        "MouthOpen": 0.9, "MouthSmile": 0.2,
        "EyeOpenLeft": 0.7, "EyeOpenRight": 0.7,
    },
    "eyes_closed": {
        "ParamAngleX": 0, "ParamAngleY": 0, "ParamAngleZ": 0,
        "ParamMouthOpenY": 0, "ParamMouthForm": -0.2,
        "ParamEyeLOpen": 0.0, "ParamEyeROpen": 0.0,
        "ParamEyeBallX": 0, "ParamEyeBallY": 0,
        "FaceAngleX": 0, "FaceAngleY": 0, "FaceAngleZ": 0,
        "MouthOpen": 0, "MouthSmile": 0.3,
        "EyeOpenLeft": 0.0, "EyeOpenRight": 0.0,
    },
    "head_left": {
        "ParamAngleX": -20, "ParamAngleY": 5, "ParamAngleZ": -8,
        "ParamMouthOpenY": 0, "ParamMouthForm": -0.3,
        "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
        "ParamEyeBallX": 0.3, "ParamEyeBallY": 0,
        "FaceAngleX": -18, "FaceAngleY": 4, "FaceAngleZ": -6,
        "MouthOpen": 0, "MouthSmile": 0.25,
        "EyeOpenLeft": 0.7, "EyeOpenRight": 0.7,
    },
    "head_right": {
        "ParamAngleX": 20, "ParamAngleY": 5, "ParamAngleZ": 8,
        "ParamMouthOpenY": 0, "ParamMouthForm": -0.3,
        "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
        "ParamEyeBallX": -0.3, "ParamEyeBallY": 0,
        "FaceAngleX": 18, "FaceAngleY": 4, "FaceAngleZ": 6,
        "MouthOpen": 0, "MouthSmile": 0.25,
        "EyeOpenLeft": 0.7, "EyeOpenRight": 0.7,
    },
    "head_tilt": {
        "ParamAngleX": 5, "ParamAngleY": -10, "ParamAngleZ": 18,
        "ParamMouthOpenY": 0.05, "ParamMouthForm": 0.2,
        "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
        "ParamEyeBallX": 0, "ParamEyeBallY": 0.2,
        "FaceAngleX": 4, "FaceAngleY": -8, "FaceAngleZ": 15,
        "MouthOpen": 0.02, "MouthSmile": 0.5,
        "EyeOpenLeft": 0.7, "EyeOpenRight": 0.7,
    },
    "combo": {
        "ParamAngleX": -12, "ParamAngleY": 8, "ParamAngleZ": -5,
        "ParamMouthOpenY": 0.45, "ParamMouthForm": 0.6,
        "ParamEyeLOpen": 0.85, "ParamEyeROpen": 0.85,
        "ParamEyeBallX": 0.2, "ParamEyeBallY": -0.1,
        "FaceAngleX": -10, "FaceAngleY": 6, "FaceAngleZ": -4,
        "MouthOpen": 0.35, "MouthSmile": 0.8,
        "EyeOpenLeft": 0.6, "EyeOpenRight": 0.6,
    },
}


def _raw_to_tensor(raw: dict, stats: dict | None) -> torch.Tensor:
    values = [float(raw.get(n, 0.0)) for n in PARAM_NAMES]
    if stats and stats.get("normalized"):
        values = normalize_params(values, stats["param_mins"], stats["param_maxs"])
    return torch.tensor([values], dtype=torch.float32)


def _load_holdout_by_character(holdout_dir: Path):
    builder = tfds.builder_from_directory(str(holdout_dir))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    by_char: dict[str, list] = {}
    for ex in ds:
        image = ex["image"]
        if hasattr(image, "numpy"):
            image = image.numpy()
        if isinstance(image, (bytes, bytearray)) or (
            isinstance(image, np.ndarray) and image.dtype == object
        ):
            import tensorflow as tf

            image = tf.io.decode_png(
                image if not isinstance(image, np.ndarray) else image.item(), channels=3
            ).numpy()
        image = np.asarray(image)
        cid = ex.get("character_id", b"unknown")
        if hasattr(cid, "numpy"):
            cid = cid.numpy()
        if isinstance(cid, (bytes, bytearray)):
            cid = cid.decode("utf-8", errors="replace")
        else:
            cid = str(cid)
        by_char.setdefault(cid, []).append(image)
    return by_char


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refs-per-char", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    holdout_dir = Path(args.holdout_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_param_stats(holdout_dir)
    model, cfg = build_numeric_model(args.checkpoint, device)
    if not bool(cfg.get("use_ref_conditioning", False)):
        raise SystemExit("Checkpoint is not reference-conditioned.")
    vae = load_sd_vae(device, dtype=torch.float32)

    by_char = _load_holdout_by_character(holdout_dir)
    pose_names = list(POSE_PRESETS_RAW.keys())
    pose_batch = torch.cat(
        [_raw_to_tensor(POSE_PRESETS_RAW[n], stats) for n in pose_names], dim=0
    ).to(device)

    from inference_numeric import encode_reference_image

    for cid, images in by_char.items():
        for r in range(min(args.refs_per_char, len(images))):
            ref_img = images[r]
            ref_latent = encode_reference_image(
                vae, ref_img, image_size=int(cfg.get("image_resolution", 512))
            )
            latents = denoise_numeric_turbo(
                model,
                pose_batch,
                ref_latent=ref_latent,
                num_steps=args.steps,
                cfg_scale=args.cfg,
            )
            decoded = decode_sd_vae(vae, latents)
            # Grid: ref | pose0 | pose1 | ...
            tiles = [ref_img if ref_img.dtype == np.uint8 else (np.clip(ref_img, 0, 255)).astype(np.uint8)]
            if tiles[0].max() <= 1.5:
                tiles[0] = (tiles[0] * 255).astype(np.uint8)
            for i, name in enumerate(pose_names):
                tiles.append(decoded[i])
            h = max(t.shape[0] for t in tiles)
            w = max(t.shape[1] for t in tiles)
            row = np.zeros((h, w * len(tiles), 3), dtype=np.uint8)
            for i, t in enumerate(tiles):
                hh, ww = t.shape[:2]
                row[:hh, i * w : i * w + ww] = t[..., :3]
            out_path = out_dir / f"{cid}__ref{r}.png"
            Image.fromarray(row).save(out_path)
            print(f"wrote {out_path}")

    meta = {"poses": pose_names, "param_names": list(PARAM_NAMES), "num_params": NUM_PARAMS}
    (out_dir / "grid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
