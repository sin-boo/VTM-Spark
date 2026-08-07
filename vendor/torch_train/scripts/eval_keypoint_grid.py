"""Holdout character x pose grid, sweeping identity CFG scale.

Example::

  python -m scripts.eval_keypoint_grid \\
    --checkpoint ../workdir/keypoint_30m/checkpoint.pt \\
    --holdout-cache ../data/holdout_crop/latents_cache.pt \\
    --output-dir ../data/eval_keypoint_grids \\
    --id-scales 1.0,1.5,2.0,3.0 \\
    --steps 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_keypoint import (  # noqa: E402
    build_keypoint_model,
    decode_sd_vae,
    denoise_keypoint,
    load_sd_vae,
)
from utils.keypoints import NUM_KEYPOINTS  # noqa: E402


def _load_holdout(cache_path: Path):
    payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    latents = payload["latents"].float()
    keypoints = payload["keypoints"].float()
    char_ids = payload.get("character_ids")
    if char_ids is None:
        # Fallback: treat each sample as its own character
        char_ids = [f"sample_{i}" for i in range(latents.shape[0])]
    else:
        if isinstance(char_ids, torch.Tensor):
            char_ids = [str(x) for x in char_ids.tolist()]
        elif isinstance(char_ids, (list, tuple)) and char_ids and isinstance(char_ids[0], bytes):
            char_ids = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in char_ids]
        else:
            char_ids = [str(c) for c in char_ids]
    return latents, keypoints, char_ids


def _pick_chars(char_ids: list[str], max_chars: int) -> list[str]:
    seen = []
    for c in char_ids:
        if c not in seen:
            seen.append(c)
        if len(seen) >= max_chars:
            break
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--holdout-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--id-scales", default="1.0,1.5,2.0,3.0")
    ap.add_argument("--pose-cfg", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=4)
    ap.add_argument("--poses-per-char", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg = build_keypoint_model(args.checkpoint, device)
    vae = load_sd_vae(device, dtype=torch.float32)
    use_face = bool(cfg.get("use_ref_face_tokens", False))
    face_size = int(cfg.get("ref_face_size", 32))
    image_size = int(cfg.get("image_resolution", 768))

    latents, keypoints, char_ids = _load_holdout(Path(args.holdout_cache))
    chars = _pick_chars(char_ids, args.max_chars)
    id_scales = [float(x) for x in args.id_scales.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from utils.pose_map import crop_face_latent

    for cid in chars:
        idxs = [i for i, c in enumerate(char_ids) if c == cid]
        if not idxs:
            continue
        ref_i = idxs[0]
        pose_idxs = idxs[1 : 1 + args.poses_per_char] or idxs[:1]
        ref_lat = latents[ref_i : ref_i + 1].to(device)
        ref_kps = keypoints[ref_i : ref_i + 1].to(device)
        ref_face = None
        if use_face:
            ref_face = crop_face_latent(ref_lat, ref_kps, out_size=face_size)

        rows = []
        for pose_i in pose_idxs:
            tgt_kps = keypoints[pose_i : pose_i + 1]
            row_imgs = []
            for s_id in id_scales:
                lat = denoise_keypoint(
                    model,
                    keypoints_target=tgt_kps,
                    ref_latent=ref_lat,
                    ref_keypoints=ref_kps,
                    ref_face_latent=ref_face,
                    num_steps=args.steps,
                    pose_cfg_scale=args.pose_cfg,
                    id_cfg_scale=s_id,
                    seed=args.seed + pose_i,
                )
                rgb = decode_sd_vae(vae, lat)[0]
                row_imgs.append(rgb)
            rows.append(np.concatenate(row_imgs, axis=1))

        # Header strip: reference image decoded from latent for context
        ref_rgb = decode_sd_vae(vae, ref_lat)[0]
        # Pad ref to full row width
        cell_w = rows[0].shape[1] // max(len(id_scales), 1)
        ref_cell = np.asarray(
            Image.fromarray(ref_rgb).resize((cell_w, ref_rgb.shape[0]), Image.BILINEAR)
        )
        # Label row as first id-scale column only; leave rest black
        header = np.zeros_like(rows[0])
        header[:, :cell_w] = ref_cell
        grid = np.concatenate([header] + rows, axis=0)
        out_path = out_dir / f"{cid}_idcfg.png"
        Image.fromarray(grid).save(out_path)
        print(f"wrote {out_path}  chars={cid}  id_scales={id_scales}  image_size={image_size}")


if __name__ == "__main__":
    main()
