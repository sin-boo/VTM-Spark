"""Pre-encode dataset images with the frozen VAE for faster DiT training.

Supports two input modes:

1. **Cropped keypoint dataset** (preferred)::

       data_dir/
         images/<character_id>/<pose_id>.png
         labels/<character_id>/<pose_id>_keypoints.npy

2. **Legacy TFRecords** with params / captions (numeric path).

Writes a single .pt cache::

  {
    "latents": FloatTensor[N, C, H, W],
    "keypoints": FloatTensor[N, 37, 4],   # keypoint mode
    # OR
    "params": FloatTensor[N, P],         # numeric mode
    "character_ids": list[str],
    "image_names": list[str],
    "vae_type": str,
    "scaling_applied": True,
    "mode": "keypoints" | "params",
  }

Hue augmentation has been removed (it broke reference identity conditioning).

Example::

  python -m scripts.cache_latents \\
    --data-dir ../data/train_crop \\
    --output ../data/train_crop/latents_cache.pt \\
    --mode keypoints \\
    --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import ConfigDict  # noqa: E402
from utils.keypoints import KEYPOINT_DIM, NUM_KEYPOINTS, empty_keypoints  # noqa: E402
from utils.params import NUM_PARAMS, parse_caption_params  # noqa: E402
from vae.vae import encode_images_to_latents, load_vae, scale_latents  # noqa: E402


def _as_str(value) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.item()
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")
    return str(value)


def _to_neg1_1(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.5:
        image = image / 127.5 - 1.0
    return image


def _list_keypoint_samples(data_dir: Path) -> list[tuple[Path, Path, str, str]]:
    """Return list of (img_path, kps_path_or_empty, image_name, character_id)."""
    images_root = data_dir / "images"
    labels_root = data_dir / "labels"
    if not images_root.is_dir():
        raise FileNotFoundError(f"Expected {images_root} for keypoint mode")
    samples: list[tuple[Path, Path, str, str]] = []
    for char_dir in sorted(images_root.iterdir()):
        if not char_dir.is_dir():
            continue
        cid = char_dir.name
        lab_dir = labels_root / cid
        for img_path in sorted(char_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            pose_id = img_path.stem
            kps_path = lab_dir / f"{pose_id}_keypoints.npy"
            samples.append((img_path, kps_path, f"{cid}/{pose_id}", cid))
    return samples


class KeypointImageDataset(Dataset):
    """Threaded DataLoader-friendly keypoint image reader."""

    def __init__(self, samples: list[tuple[Path, Path, str, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, kps_path, name, cid = self.samples[idx]
        if kps_path.is_file():
            kps = np.load(kps_path).astype(np.float32)
        else:
            kps = empty_keypoints()
        if kps.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
            raise ValueError(f"Bad keypoints shape {kps.shape} in {kps_path}")
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        image = _to_neg1_1(rgb)
        return image, kps, name, cid


def _keypoint_collate(batch):
    images = np.stack([b[0] for b in batch], axis=0)
    kps = np.stack([b[1] for b in batch], axis=0)
    names = [b[2] for b in batch]
    cids = [b[3] for b in batch]
    return images, kps, names, cids


def iter_keypoint_dataset(data_dir: Path):
    """Yield (image_neg1_1, keypoints[37,4], image_name, character_id)."""
    for img_path, kps_path, name, cid in _list_keypoint_samples(data_dir):
        if kps_path.is_file():
            kps = np.load(kps_path).astype(np.float32)
        else:
            kps = empty_keypoints()
        if kps.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
            raise ValueError(f"Bad keypoints shape {kps.shape} in {kps_path}")
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        yield _to_neg1_1(rgb), kps, name, cid


def iter_tfrecord_dataset(data_dir: Path):
    """Yield (image, params, image_name, character_id) from TFDS builder dir."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(data_dir))
    raw = builder.as_dataset(split="train", shuffle_files=False)
    for ex in raw:
        image = ex["image"]
        if hasattr(image, "numpy"):
            image = image.numpy()
        if isinstance(image, (bytes, bytearray)) or (isinstance(image, np.ndarray) and image.dtype == object):
            image = tf.io.decode_png(
                image if not isinstance(image, np.ndarray) else image.item(), channels=3
            ).numpy()
        image = _to_neg1_1(image)
        if "params" in ex:
            params = np.asarray(
                ex["params"].numpy() if hasattr(ex["params"], "numpy") else ex["params"],
                dtype=np.float32,
            )
        else:
            caption = _as_str(ex.get("caption", ex.get("labels", b"")))
            params = np.asarray(parse_caption_params(caption), dtype=np.float32)
        if params.shape[-1] != NUM_PARAMS:
            # pad / truncate
            p = np.zeros((NUM_PARAMS,), dtype=np.float32)
            n = min(NUM_PARAMS, int(np.prod(params.shape)))
            p[:n] = params.reshape(-1)[:n]
            params = p
        name = _as_str(ex.get("image_name", b""))
        if "character_id" in ex:
            character_id = _as_str(ex["character_id"])
        else:
            character_id = "legacy"
        yield image, params, name, character_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae-type", default="sd")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mode",
        choices=("auto", "keypoints", "params"),
        default="auto",
        help="auto: keypoint if images/ exists, else TFRecords params.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(2, min(8, (os.cpu_count() or 4) // 2)),
        help="DataLoader workers for image decode (keypoint mode). Use 0 on Windows to avoid freezes.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Encode at most this many samples (0 = all). Useful for quick smoke trains.",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable CUDA pin_memory (recommended on Windows desktops to keep the OS responsive).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor when workers>0 (lower = less host RAM pressure).",
    )
    parser.add_argument(
        "--yield-ms",
        type=int,
        default=0,
        help="Sleep this many ms every batch so the desktop UI can breathe (e.g. 5–20).",
    )
    parser.add_argument(
        "--gentle",
        action="store_true",
        help="Desktop-safe preset: workers=0, batch<=4, no pin_memory, yield 10ms, fewer torch threads.",
    )
    args = parser.parse_args()
    if args.gentle:
        args.workers = 0
        args.batch_size = min(int(args.batch_size), 4)
        args.no_pin_memory = True
        if int(args.yield_ms) <= 0:
            args.yield_ms = 10
        # Leave most CPU cores for Windows / UI.
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
            os.environ.setdefault(key, "4")
        try:
            torch.set_num_threads(4)
        except Exception:
            pass

    data_dir = Path(args.data_dir)
    mode = args.mode
    if mode == "auto":
        mode = "keypoints" if (data_dir / "images").is_dir() else "params"

    device = torch.device(args.device)
    config = ConfigDict(dict(vae_type=args.vae_type))
    vae = load_vae(config, device, dtype=torch.float32)

    print(
        f"Encoding dataset mode={mode} vae={args.vae_type} from {data_dir} "
        f"(batch={args.batch_size}, workers={args.workers}) ..."
    )

    latent_chunks = []
    cond_list, names_list, cids_list = [], [], []
    n = 0

    if mode == "keypoints":
        samples = _list_keypoint_samples(data_dir)
        if args.max_samples and args.max_samples > 0:
            samples = samples[: int(args.max_samples)]
        if not samples:
            raise SystemExit("No samples found to encode.")
        print(f"Encoding {len(samples)} keypoint samples ...", flush=True)
        ds = KeypointImageDataset(samples)
        n_workers = max(0, int(args.workers))
        pin_memory = bool(device.type == "cuda" and not args.no_pin_memory)
        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=_keypoint_collate,
            pin_memory=pin_memory,
        )
        if n_workers > 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))
        loader = DataLoader(ds, **loader_kwargs)
        yield_s = max(0, int(args.yield_ms)) / 1000.0
        for images, kps, names, cids in loader:
            imgs = torch.from_numpy(images).to(device, non_blocking=pin_memory)
            with torch.no_grad():
                latents = encode_images_to_latents(vae, imgs, sample=False)
                latents = scale_latents(latents, config).float().cpu()
            latent_chunks.append(latents)
            cond_list.extend(list(kps))
            names_list.extend(names)
            cids_list.extend(cids)
            n += len(names)
            if n % max(64, args.batch_size * 4) == 0 or n == len(samples):
                print(f"  encoded {n}/{len(samples)} ...", flush=True)
            if yield_s > 0:
                # Keep the Windows desktop scheduler from starving under sustained CUDA load.
                time.sleep(yield_s)
                if n % max(256, args.batch_size * 16) == 0:
                    torch.cuda.empty_cache()
        cond_key = "keypoints"
        all_cond = torch.from_numpy(np.stack(cond_list, axis=0)).float()
    else:
        iterator = iter_tfrecord_dataset(data_dir)
        batch_imgs, batch_cond, batch_names, batch_cids = [], [], [], []

        def flush():
            nonlocal batch_imgs, batch_cond, batch_names, batch_cids
            if not batch_imgs:
                return
            imgs = torch.from_numpy(np.stack(batch_imgs, axis=0)).to(device)
            with torch.no_grad():
                latents = encode_images_to_latents(vae, imgs, sample=False)
                latents = scale_latents(latents, config).float().cpu()
            latent_chunks.append(latents)
            cond_list.extend(batch_cond)
            names_list.extend(batch_names)
            cids_list.extend(batch_cids)
            batch_imgs, batch_cond, batch_names, batch_cids = [], [], [], []

        for image, cond, name, character_id in iterator:
            batch_imgs.append(image)
            batch_cond.append(cond)
            batch_names.append(name)
            batch_cids.append(character_id)
            n += 1
            if len(batch_imgs) >= args.batch_size:
                flush()
                if n % (args.batch_size * 20) == 0:
                    print(f"  encoded {n} ...")
        flush()
        cond_key = "params"
        all_cond = torch.from_numpy(np.stack(cond_list, axis=0)).float()

    if not latent_chunks:
        raise SystemExit("No samples found to encode.")

    all_latents = torch.cat(latent_chunks, dim=0)

    out = {
        "latents": all_latents,
        cond_key: all_cond,
        "character_ids": cids_list,
        "image_names": names_list,
        "vae_type": args.vae_type,
        "scaling_applied": True,
        "mode": mode,
        "hue_augments": 0,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    n_chars = len(set(cids_list))
    print(f"Wrote {all_latents.shape[0]} latents {tuple(all_latents.shape[1:])} -> {out_path}")
    print(f"{cond_key} shape: {tuple(all_cond.shape)} | characters: {n_chars}")


if __name__ == "__main__":
    main()
