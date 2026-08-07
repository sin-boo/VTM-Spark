"""Build cropped training dataset from green-screen characters + keypoint labels.

Expected raw layout (flexible — also accepts pose-traker output layout)::

    raw_root/<character_id>/
      images/<pose_id>.png|.jpg     # OR <pose_id>.png at character root
      labels/<pose_id>_landmarks.json
      labels/<pose_id>_iris.json
      labels/<pose_id>_upper_body_pose.json
      # OR labels/<pose_id>_full_stack.json

Writes::

    out_dir/
      images/<character_id>/<pose_id>.png
      labels/<character_id>/<pose_id>_keypoints.npy   # (37, 4) normalized [-1,1]
      split_info.json
      crop_stats.json

Usage::

  # Stats only (pick aspect ratio from data):
  python -m scripts.build_dataset --raw-root ../data/raw --stats-only

  # Recommended: chroma-key crop character, pad to square, resize (keypoints auto-mapped):
  python -m scripts.build_dataset \\
    --raw-root ../data/raw \\
    --out-dir ../data/train_crop \\
    --out-size 768 768 \\
    --aspect 1.0 \\
    --margin-x 32

  # Optional: keep full frame + green pad (no crop / no stretch):
  python -m scripts.build_dataset \\
    --raw-root ../data/raw \\
    --out-dir ../data/train_crop \\
    --out-size 768 768 \\
    --aspect 1.0 \\
    --skip-crop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def _default_workers() -> int:
    """Use all logical CPUs for crop/build (CPU-bound; processes beat threads under GIL)."""
    return max(1, os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.keypoints import (  # noqa: E402
    NUM_KEYPOINTS,
    load_keypoints_from_label_dir,
    transform_keypoints_crop,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def aspect(self) -> float:
        return self.w / max(self.h, 1)


def chroma_key_mask(
    rgb: np.ndarray,
    *,
    green_thresh: float = 40.0,
    green_dominance: float = 20.0,
) -> np.ndarray:
    """Boolean mask True = foreground (not green screen).

    ``rgb`` is uint8 HxWx3.
    A pixel is background if G is high and G exceeds R and B by ``green_dominance``.
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    is_green = (g > green_thresh) & (g - r > green_dominance) & (g - b > green_dominance)
    return ~is_green


def sample_green_fill(
    rgb: np.ndarray,
    *,
    green_thresh: float = 40.0,
    green_dominance: float = 20.0,
    fallback: tuple[int, int, int] = (8, 240, 20),
) -> tuple[int, int, int]:
    """Pick a pad color that matches this image's green screen (not pure #00FF00)."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    is_green = (g > green_thresh) & (g - r > green_dominance) & (g - b > green_dominance)
    if not bool(is_green.any()):
        return fallback
    med = np.median(rgb[is_green], axis=0)
    return (int(round(med[0])), int(round(med[1])), int(round(med[2])))


def bbox_from_mask(
    mask: np.ndarray,
    margin: int = 8,
    *,
    margin_top: int | None = None,
    margin_bottom: int | None = None,
    margin_x: int | None = None,
) -> BBox | None:
    """Tight foreground bbox expanded by margins.

    Margins may extend past the image edge so ``crop_and_resize`` can pad with
    green.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    mt = int(margin if margin_top is None else margin_top)
    mb = int(margin if margin_bottom is None else margin_bottom)
    mx = int(margin if margin_x is None else margin_x)
    x0 = int(xs.min()) - mx
    y0 = int(ys.min()) - mt
    x1 = int(xs.max()) + 1 + mx
    y1 = int(ys.max()) + 1 + mb
    return BBox(x0, y0, x1, y1)


def horizontal_crop_bbox(
    mask: np.ndarray,
    *,
    margin_x: int = 32,
) -> BBox | None:
    """Crop left/right only; keep the full image height (no top/bottom crop)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, _w = mask.shape
    x0 = int(xs.min()) - int(margin_x)
    x1 = int(xs.max()) + 1 + int(margin_x)
    return BBox(x0, 0, x1, h)



def pad_bbox_to_aspect(bbox: BBox, img_w: int, img_h: int, target_aspect: float) -> BBox:
    """Expand bbox (never shrink) so width/height ~= target_aspect.

    The result may extend outside the image; ``crop_and_resize`` fills those
    regions with green so hair near frame edges keeps breathing room.
    """
    del img_w, img_h  # kept for call-site compatibility; no longer clamped here
    w, h = bbox.w, bbox.h
    if w <= 0 or h <= 0:
        return bbox
    cur = w / h
    if cur < target_aspect:
        # Too tall — expand width.
        new_w = int(round(h * target_aspect))
        new_h = h
    else:
        new_h = int(round(w / target_aspect))
        new_w = w
    cx = (bbox.x0 + bbox.x1) / 2.0
    cy = (bbox.y0 + bbox.y1) / 2.0
    x0 = int(round(cx - new_w / 2.0))
    y0 = int(round(cy - new_h / 2.0))
    x1 = x0 + new_w
    y1 = y0 + new_h
    return BBox(x0, y0, x1, y1)


def crop_and_resize(
    rgb: np.ndarray,
    bbox: BBox,
    out_w: int,
    out_h: int,
    fill: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Crop bbox from rgb; if bbox exceeds image, pad with fill; then resize."""
    if fill is None:
        fill = sample_green_fill(rgb)
    h, w = rgb.shape[:2]
    canvas = np.full((bbox.h, bbox.w, 3), fill, dtype=np.uint8)
    # Region of bbox that overlaps the image.
    src_x0 = max(bbox.x0, 0)
    src_y0 = max(bbox.y0, 0)
    src_x1 = min(bbox.x1, w)
    src_y1 = min(bbox.y1, h)
    dst_x0 = src_x0 - bbox.x0
    dst_y0 = src_y0 - bbox.y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = rgb[src_y0:src_y1, src_x0:src_x1]
    img = Image.fromarray(canvas, mode="RGB")
    img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def square_pad_offsets(width: int, height: int) -> tuple[int, int, int]:
    """Centered pad amounts and square side length for a full-frame image."""
    side = max(int(width), int(height), 1)
    pad_x = (side - int(width)) // 2
    pad_y = (side - int(height)) // 2
    return pad_x, pad_y, side


def pad_full_image_to_square(
    rgb: np.ndarray,
    fill: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, int, int, int]:
    """Keep the full image, center it on a green square canvas.

    Returns ``(canvas, pad_x, pad_y, side)`` where the original pixels sit at
    ``canvas[pad_y:pad_y+h, pad_x:pad_x+w]``.
    """
    if fill is None:
        fill = sample_green_fill(rgb)
    h, w = rgb.shape[:2]
    pad_x, pad_y, side = square_pad_offsets(w, h)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    canvas[pad_y : pad_y + h, pad_x : pad_x + w] = rgb
    return canvas, pad_x, pad_y, side


def _is_trainable_pose_id(pose_id: str) -> bool:
    """Skip incomplete body-only ``*_original`` captures."""
    return not pose_id.endswith("_original")


def _discover_poses(char_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (pose_id, image_path, label_dir)."""
    images_dir = char_dir / "images"
    labels_dir = char_dir / "labels"
    # Layout A: images/ + labels/
    if images_dir.is_dir():
        label_root = labels_dir if labels_dir.is_dir() else char_dir
        out = []
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            if not _is_trainable_pose_id(p.stem):
                continue
            out.append((p.stem, p, label_root))
        return out
    # Layout B: pose-traker style — images and jsons co-located in character folder
    # OR images live elsewhere and labels are here (labels-only character dir).
    out = []
    for p in sorted(char_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if not _is_trainable_pose_id(p.stem):
            continue
        out.append((p.stem, p, char_dir))
    if out:
        return out
    # Layout C: label jsons only — image path inferred from source_image in json, or sibling input/
    for p in sorted(char_dir.glob("*_landmarks.json")):
        pose_id = p.name.replace("_landmarks.json", "")
        if not _is_trainable_pose_id(pose_id):
            continue
        # Try sibling image in same dir or parent images
        candidates = [
            char_dir / f"{pose_id}.png",
            char_dir / f"{pose_id}.jpg",
            char_dir / f"{pose_id}.jpeg",
            char_dir.parent.parent / "input" / char_dir.name / f"{pose_id}.png",
            char_dir.parent.parent / "input" / char_dir.name / f"{pose_id}.jpg",
        ]
        img = next((c for c in candidates if c.is_file()), None)
        if img is None:
            # Try reading source_image from full_stack / landmarks
            for jname in (f"{pose_id}_full_stack.json", f"{pose_id}_landmarks.json"):
                jp = char_dir / jname
                if not jp.is_file():
                    continue
                try:
                    meta = json.loads(jp.read_text(encoding="utf-8"))
                    src = meta.get("source_image")
                    if src and Path(src).is_file():
                        img = Path(src)
                        break
                except Exception:
                    pass
        if img is not None:
            out.append((pose_id, img, char_dir))
    return out


def discover_characters(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root not found: {raw_root}")
    chars = [p for p in sorted(raw_root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    return chars


def _stats_one(payload: tuple) -> tuple[float, int, int] | None:
    """Worker: return (aspect, width, height) or None on failure."""
    img_path_s, green_thresh, green_dominance, margin, margin_x = payload
    try:
        rgb = np.asarray(Image.open(img_path_s).convert("RGB"))
    except Exception:
        return None
    mask = chroma_key_mask(rgb, green_thresh=green_thresh, green_dominance=green_dominance)
    mx = int(margin if margin_x is None else margin_x)
    bbox = horizontal_crop_bbox(mask, margin_x=mx)
    if bbox is None or bbox.w < 8 or bbox.h < 8:
        return None
    side = int(rgb.shape[0])  # square = full image height; sides only
    return (1.0, side, side)


def collect_bbox_stats(
    raw_root: Path,
    *,
    green_thresh: float,
    green_dominance: float,
    margin: int,
    margin_top: int | None = None,
    margin_bottom: int | None = None,
    margin_x: int | None = None,
    max_samples: int = 0,
    workers: int | None = None,
) -> dict:
    del margin_top, margin_bottom  # side-only path
    jobs: list[tuple] = []
    for char_dir in discover_characters(raw_root):
        for pose_id, img_path, _ in _discover_poses(char_dir):
            jobs.append(
                (str(img_path), float(green_thresh), float(green_dominance), int(margin), margin_x)
            )
            if max_samples and len(jobs) >= max_samples:
                break
        if max_samples and len(jobs) >= max_samples:
            break

    aspects: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    n_fail = 0
    n_workers = max(1, int(workers if workers is not None else _default_workers()))
    if n_workers == 1 or len(jobs) <= 1:
        results = [_stats_one(j) for j in jobs]
    else:
        print(f"Collecting bbox stats with {n_workers} workers ({len(jobs)} images) ...")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_stats_one, jobs, chunksize=max(8, len(jobs) // (n_workers * 8) or 1)))
    for r in results:
        if r is None:
            n_fail += 1
            continue
        aspects.append(r[0])
        widths.append(r[1])
        heights.append(r[2])
    n = len(aspects)
    if not aspects:
        return {"num_samples": 0, "num_fail": n_fail}
    arr = np.asarray(aspects, dtype=np.float64)
    return {
        "num_samples": int(n),
        "num_fail": int(n_fail),
        "aspect_mean": float(arr.mean()),
        "aspect_median": float(np.median(arr)),
        "aspect_p50": float(np.percentile(arr, 50)),
        "aspect_p90": float(np.percentile(arr, 90)),
        "aspect_p95": float(np.percentile(arr, 95)),
        "aspect_p99": float(np.percentile(arr, 99)),
        "width_p95": float(np.percentile(widths, 95)),
        "height_p95": float(np.percentile(heights, 95)),
        "recommended_aspect": float(np.percentile(arr, 95)),
        "note": "Pad tight boxes to recommended_aspect (width/height), then resize.",
    }


def _process_and_write(job: dict) -> dict:
    """Worker: crop one sample and write outputs. Returns a small result dict (no image IPC)."""
    cid = job["character_id"]
    pose_id = job["pose_id"]
    try:
        crop, kps, meta = process_one(
            Path(job["img_path"]),
            Path(job["label_dir"]),
            pose_id,
            out_w=int(job["out_w"]),
            out_h=int(job["out_h"]),
            target_aspect=float(job["target_aspect"]),
            green_thresh=float(job["green_thresh"]),
            green_dominance=float(job["green_dominance"]),
            margin=int(job["margin"]),
            skip_crop=bool(job["skip_crop"]),
            margin_top=job.get("margin_top"),
            margin_bottom=job.get("margin_bottom"),
            margin_x=job.get("margin_x"),
        )
        img_out = Path(job["img_out"])
        lab_out = Path(job["lab_out"])
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(img_out / f"{pose_id}.png")
        np.save(lab_out / f"{pose_id}_keypoints.npy", kps.astype(np.float32))
        (lab_out / f"{pose_id}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"ok": True, "character_id": cid, "pose_id": pose_id}
    except Exception as e:
        return {"ok": False, "character_id": cid, "pose_id": pose_id, "error": str(e)}


def process_one(
    img_path: Path,
    label_dir: Path,
    pose_id: str,
    *,
    out_w: int,
    out_h: int,
    target_aspect: float,
    green_thresh: float,
    green_dominance: float,
    margin: int,
    skip_crop: bool,
    margin_top: int | None = None,
    margin_bottom: int | None = None,
    margin_x: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    h, w = rgb.shape[:2]
    kps_abs = load_keypoints_from_label_dir(label_dir, pose_id)

    if skip_crop:
        # Keep the full frame: center on a green square, then resize. Do not stretch.
        canvas, pad_x, pad_y, side = pad_full_image_to_square(rgb)
        bbox = BBox(-pad_x, -pad_y, -pad_x + side, -pad_y + side)
        crop = np.asarray(Image.fromarray(canvas).resize((out_w, out_h), Image.Resampling.LANCZOS))
        kps = transform_keypoints_crop(
            kps_abs,
            crop_x0=-pad_x,
            crop_y0=-pad_y,
            crop_w=side,
            crop_h=side,
            out_w=out_w,
            out_h=out_h,
            normalize=True,
        )
    else:
        mask = chroma_key_mask(rgb, green_thresh=green_thresh, green_dominance=green_dominance)
        fill = sample_green_fill(
            rgb, green_thresh=green_thresh, green_dominance=green_dominance
        )
        # Side-only crop: always keep full source height (y0=0 .. y1=h).
        # Square size is the image height; only left/right are trimmed or padded.
        mx = int(margin if margin_x is None else margin_x)
        tight = horizontal_crop_bbox(mask, margin_x=mx)
        if tight is None or tight.w < 8:
            tight = BBox(0, 0, w, h)
        side = h
        cx = (tight.x0 + tight.x1) / 2.0
        x0 = int(round(cx - side / 2.0))
        bbox = BBox(x0, 0, x0 + side, h)
        crop = crop_and_resize(rgb, bbox, out_w, out_h, fill=fill)
        kps = transform_keypoints_crop(
            kps_abs,
            crop_x0=bbox.x0,
            crop_y0=bbox.y0,
            crop_w=bbox.w,
            crop_h=bbox.h,
            out_w=out_w,
            out_h=out_h,
            normalize=True,
        )

    meta = {
        "source": str(img_path),
        "orig_size": [w, h],
        "bbox": asdict(bbox),
        "out_size": [out_w, out_h],
        "num_keypoints": NUM_KEYPOINTS,
        "visible_count": int((kps[:, 3] > 0.5).sum()),
        "skip_crop": bool(skip_crop),
        "pad_mode": "full_frame_square_green" if skip_crop else "chroma_key_sides_only",
        "margin": {
            "base": int(margin),
            "top": 0,
            "bottom": 0,
            "x": int(margin if margin_x is None else margin_x),
        },
    }
    return crop, kps, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--out-size", type=int, nargs=2, default=[768, 768], metavar=("W", "H"),
                        help="Output crop size (width height). Default 768x768 (square for current DiT).")
    parser.add_argument("--aspect", type=float, default=None,
                        help="Target width/height. Default: use --out-size aspect, or p95 from stats.")
    parser.add_argument("--green-thresh", type=float, default=40.0)
    parser.add_argument("--green-dominance", type=float, default=20.0)
    parser.add_argument(
        "--margin",
        type=int,
        default=32,
        help="Fallback side margin if --margin-x is omitted.",
    )
    parser.add_argument(
        "--margin-top",
        type=int,
        default=0,
        help="Ignored (side-only crop keeps full image height).",
    )
    parser.add_argument(
        "--margin-bottom",
        type=int,
        default=0,
        help="Ignored (side-only crop keeps full image height).",
    )
    parser.add_argument(
        "--margin-x",
        type=int,
        default=32,
        help="Extra pixels on left/right only (default 32). Top/bottom are never cropped.",
    )
    parser.add_argument(
        "--skip-crop",
        action="store_true",
        help="Skip chroma-key crop: keep full image, pad to square with green, then resize (no stretch).",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit for stats or dry processing.")
    parser.add_argument("--characters", nargs="*", default=None, help="Optional character_id filter.")
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Parallel worker processes for crop/stats (default: all CPU cores).",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    out_w, out_h = int(args.out_size[0]), int(args.out_size[1])
    n_workers = max(1, int(args.workers))

    # Full bbox scan is only needed for --stats-only or when --aspect is omitted.
    # Skip it when the caller already fixed the aspect (avoids reading every image twice).
    if args.stats_only or args.aspect is None:
        stats = collect_bbox_stats(
            raw_root,
            green_thresh=args.green_thresh,
            green_dominance=args.green_dominance,
            margin=args.margin,
            margin_top=args.margin_top,
            margin_bottom=args.margin_bottom,
            margin_x=args.margin_x,
            max_samples=args.max_samples,
            workers=n_workers,
        )
        print(json.dumps(stats, indent=2))
    else:
        stats = {
            "num_samples": 0,
            "num_fail": 0,
            "note": "bbox stats skipped (aspect provided)",
            "recommended_aspect": float(args.aspect),
        }
        print(json.dumps(stats, indent=2))

    if args.stats_only:
        return

    if args.out_dir is None:
        raise SystemExit("--out-dir is required unless --stats-only")

    target_aspect = args.aspect
    if target_aspect is None:
        target_aspect = float(stats.get("recommended_aspect") or (out_w / max(out_h, 1)))
    print(
        f"Using target aspect (w/h)={target_aspect:.4f}, out_size={out_w}x{out_h}, "
        f"margins top/bottom/x={args.margin_top}/{args.margin_bottom}/{args.margin_x}, "
        f"workers={n_workers}"
    )

    out_dir = args.out_dir.resolve()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    char_filter = set(args.characters) if args.characters else None
    characters = discover_characters(raw_root)
    if char_filter:
        characters = [c for c in characters if c.name in char_filter]

    jobs: list[dict] = []
    for char_dir in characters:
        cid = char_dir.name
        poses = _discover_poses(char_dir)
        if not poses:
            continue
        img_out = out_dir / "images" / cid
        lab_out = out_dir / "labels" / cid
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        for pose_id, img_path, label_dir in poses:
            jobs.append(
                {
                    "character_id": cid,
                    "pose_id": pose_id,
                    "img_path": str(img_path),
                    "label_dir": str(label_dir),
                    "img_out": str(img_out),
                    "lab_out": str(lab_out),
                    "out_w": out_w,
                    "out_h": out_h,
                    "target_aspect": float(target_aspect),
                    "green_thresh": float(args.green_thresh),
                    "green_dominance": float(args.green_dominance),
                    "margin": int(args.margin),
                    "skip_crop": bool(args.skip_crop),
                    "margin_top": args.margin_top,
                    "margin_bottom": args.margin_bottom,
                    "margin_x": args.margin_x,
                }
            )
            if args.max_samples and len(jobs) >= args.max_samples:
                break
        if args.max_samples and len(jobs) >= args.max_samples:
            break

    print(f"Processing {len(jobs)} samples ...", flush=True)
    index: list[dict] = []
    n_ok = 0
    n_fail = 0
    done = 0

    def _consume(result: dict) -> None:
        nonlocal n_ok, n_fail, done
        done += 1
        if result["ok"]:
            n_ok += 1
            index.append({"character_id": result["character_id"], "pose_id": result["pose_id"]})
        else:
            n_fail += 1
            print(f"FAIL {result['character_id']}/{result['pose_id']}: {result.get('error')}", flush=True)
        if done % 200 == 0 or done == len(jobs):
            print(f"  progress {done}/{len(jobs)} ok={n_ok} fail={n_fail}", flush=True)

    if n_workers == 1 or len(jobs) <= 1:
        for result in (_process_and_write(j) for j in jobs):
            _consume(result)
    else:
        chunksize = max(4, len(jobs) // (n_workers * 16) or 1)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for result in ex.map(_process_and_write, jobs, chunksize=chunksize):
                _consume(result)

    # Stable order for split_info
    index.sort(key=lambda e: (e["character_id"], e["pose_id"]))

    split_info = {
        "num_samples": n_ok,
        "num_fail": n_fail,
        "num_characters": len({e["character_id"] for e in index}),
        "out_size": [out_w, out_h],
        "aspect": target_aspect,
        "workers": n_workers,
        "samples": index,
    }
    (out_dir / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    (out_dir / "crop_stats.json").write_text(
        json.dumps({"bbox_stats": stats, "target_aspect": target_aspect, "out_size": [out_w, out_h]}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {n_ok} samples ({n_fail} failed) -> {out_dir}")


if __name__ == "__main__":
    main()
