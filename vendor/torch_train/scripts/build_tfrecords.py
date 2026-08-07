"""Build TFRecords with image + float32[NUM_PARAMS] params + character_id.

Examples:
  # From captions.jsonl + images folder
  python -m scripts.build_tfrecords \\
    --captions-jsonl path/to/captions.jsonl \\
    --images-dir path/to/images \\
    --output-dir path/to/out_512 \\
    --image-size 512

  # From a single VTS session (images/ + values/)
  python -m scripts.build_tfrecords \\
    --session path/to/session \\
    --output-dir path/to/out_512 \\
    --image-size 512

  # From many VTS sessions (identity = session.json vts.model_id)
  python -m scripts.build_tfrecords \\
    --sessions-root path/to/data-ch-5-30mil \\
    --output-dir path/to/data/train_512 \\
    --holdout-dir path/to/data/holdout_512 \\
    --image-size 512 \\
    --min-samples 200 \\
    --holdout-count 2
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.params import (  # noqa: E402
    NUM_PARAMS,
    PARAM_NAMES,
    format_caption,
    normalize_params,
    params_from_mapping,
    parse_caption_params,
)

try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE = Image.BILINEAR

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    if width <= height:
        new_w = size
        new_h = int(round(height * (size / width)))
    else:
        new_h = size
        new_w = int(round(width * (size / height)))
    if (new_w, new_h) != (width, height):
        img = img.resize((new_w, new_h), resample=RESAMPLE)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _safe_character_id(model_id: str | None, model_name: str | None, fallback: str) -> str:
    raw = (model_id or "").strip() or (model_name or "").strip() or fallback
    cleaned = _SAFE_ID_RE.sub("_", raw).strip("._-")
    return cleaned or fallback


def _read_session_identity(session_dir: Path) -> tuple[str, str]:
    """Return (character_id, display_name) from session.json when present."""
    session_path = session_dir / "session.json"
    fallback = session_dir.name
    if not session_path.is_file():
        return fallback, fallback
    doc = json.loads(session_path.read_text(encoding="utf-8"))
    vts = doc.get("vts") or {}
    model_id = vts.get("model_id")
    model_name = vts.get("model_name") or (doc.get("character") or fallback)
    character_id = _safe_character_id(model_id, model_name, fallback)
    display = str(model_name or character_id)
    return character_id, display


def _iter_from_session(session_dir: Path, character_id: str | None = None, display_name: str | None = None):
    if character_id is None or display_name is None:
        character_id, display_name = _read_session_identity(session_dir)
    values_dir = session_dir / "values"
    images_dir = session_dir / "images"
    for values_path in sorted(values_dir.glob("sample_*.json")):
        sample_id = values_path.stem
        image_path = images_dir / f"{sample_id}.png"
        if not image_path.is_file():
            continue
        doc = json.loads(values_path.read_text(encoding="utf-8"))
        training = doc.get("training") or {}
        params = params_from_mapping(training.get("params") or {})
        # Unique across sessions: character_id + sample stem.
        global_sample_id = f"{character_id}__{sample_id}"
        yield global_sample_id, image_path, character_id, display_name, params


def _iter_from_captions(captions_jsonl: Path, images_dir: Path):
    with captions_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or Path(row["image"]).stem)
            image_name = row.get("image") or f"{sample_id}.png"
            image_path = images_dir / image_name
            if not image_path.is_file():
                alt = captions_jsonl.parent / image_name
                if alt.is_file():
                    image_path = alt
                else:
                    raise FileNotFoundError(f"Missing image for {sample_id}: {image_path}")
            character = str(row.get("character") or "character_unknown")
            character_id = str(row.get("character_id") or character)
            if "params" in row and isinstance(row["params"], dict):
                params = params_from_mapping(row["params"])
            else:
                params = parse_caption_params(row.get("caption") or "")
            yield sample_id, image_path, character_id, character, params


def _discover_sessions(sessions_root: Path) -> list[Path]:
    sessions = []
    for child in sorted(sessions_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images").is_dir() and (child / "values").is_dir():
            sessions.append(child)
    return sessions


def _compute_norm_stats(all_params: list[list[float]]) -> tuple[list[float], list[float]]:
    arr = np.asarray(all_params, dtype=np.float64)
    mins = arr.min(axis=0).tolist()
    maxs = arr.max(axis=0).tolist()
    for i in range(NUM_PARAMS):
        if maxs[i] - mins[i] < 1e-6:
            mins[i] -= 1.0
            maxs[i] += 1.0
    return mins, maxs


def _write_metadata(output_dir: Path, image_size: int) -> None:
    import tensorflow as tf  # noqa: F401
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.folder_dataset import (
        compute_split_info_from_directory,
        write_metadata,
    )

    features = tfds.features.FeaturesDict(
        {
            "image": tfds.features.Image(
                shape=(image_size, image_size, 3),
                encoding_format="png",
                doc="PNG-encoded RGB bytes (decode to uint8).",
            ),
            "image/shape": tfds.features.Tensor(
                shape=(3,),
                dtype=tf.int64,
                doc="Height, width, channels.",
            ),
            "caption": tfds.features.Sequence(
                tfds.features.Text(doc="Legacy text caption (debug / UI)"),
            ),
            "params": tfds.features.Tensor(
                shape=(NUM_PARAMS,),
                dtype=tf.float32,
                doc=f"Normalized Live2D/face params in order: {list(PARAM_NAMES)}",
            ),
            "params_raw": tfds.features.Tensor(
                shape=(NUM_PARAMS,),
                dtype=tf.float32,
                doc="Unnormalized params in the same order.",
            ),
            "image_name": tfds.features.Text(
                doc="Image identifier (from filenames or dataset ids)",
            ),
            "character_id": tfds.features.Text(
                doc="Stable identity key (VTS model_id).",
            ),
        }
    )
    split_infos = compute_split_info_from_directory(data_dir=str(output_dir))
    write_metadata(
        data_dir=str(output_dir),
        features=features,
        split_infos=split_infos,
        version="1.2.0",
        description="VTS character training image-param pairs (numeric + character_id)",
        supervised_keys=("image", "params"),
    )


def _write_split(
    samples: list[tuple],
    output_dir: Path,
    image_size: int,
    shard_size: int,
    mins: list[float],
    maxs: list[float],
    no_normalize: bool,
) -> int:
    import tensorflow as tf

    if not samples:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear old shards so rebuilds stay clean.
    for old in output_dir.glob("character_vts-train.tfrecord-*"):
        old.unlink()

    shard_count = max(1, math.ceil(len(samples) / shard_size))
    writers = [
        tf.io.TFRecordWriter(
            str(output_dir / f"character_vts-train.tfrecord-{i:05d}-of-{shard_count:05d}")
        )
        for i in range(shard_count)
    ]

    captions_path = output_dir / "captions.jsonl"
    written = 0
    with captions_path.open("w", encoding="utf-8") as captions_file:
        for index, (sample_id, image_path, character_id, display_name, params_raw) in enumerate(samples):
            if no_normalize:
                params_norm = list(params_raw)
            else:
                params_norm = normalize_params(params_raw, mins, maxs)
            caption = format_caption(display_name, params_raw)
            captions_file.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "character": display_name,
                        "character_id": character_id,
                        "caption": caption,
                        "params": {n: float(v) for n, v in zip(PARAM_NAMES, params_raw)},
                        "params_norm": params_norm,
                        "image": image_path.name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            with Image.open(image_path) as img:
                cropped = _resize_center_crop(img, image_size)
            buffer = io.BytesIO()
            cropped.save(buffer, format="PNG", compress_level=3)
            img_bytes = buffer.getvalue()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[img_bytes])
                        ),
                        "image/shape": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[image_size, image_size, 3]
                            )
                        ),
                        "caption": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[caption.encode("utf-8")])
                        ),
                        "params": tf.train.Feature(
                            float_list=tf.train.FloatList(value=params_norm)
                        ),
                        "params_raw": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[float(x) for x in params_raw])
                        ),
                        "image_name": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[sample_id.encode("utf-8")])
                        ),
                        "character_id": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[character_id.encode("utf-8")])
                        ),
                    }
                )
            )
            writers[index % shard_count].write(example.SerializeToString())
            written += 1

    for w in writers:
        w.close()

    _write_metadata(output_dir, image_size)

    info = {
        "num_samples": written,
        "num_params": NUM_PARAMS,
        "param_names": list(PARAM_NAMES),
        "normalized": not no_normalize,
        "param_mins": mins,
        "param_maxs": maxs,
        "image_size": image_size,
        "characters": sorted({cid for _, _, cid, _, _ in samples}),
    }
    (output_dir / "param_stats.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return written


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--session", type=Path, help="VTS session folder (images/ + values/)")
    src.add_argument("--sessions-root", type=Path, help="Folder of many VTS sessions")
    src.add_argument("--captions-jsonl", type=Path, help="captions.jsonl path")
    parser.add_argument("--images-dir", type=Path, default=None, help="Images dir for --captions-jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=None,
        help="Optional dir for held-out characters (requires --sessions-root).",
    )
    parser.add_argument("--image-size", type=int, default=512, choices=(256, 512, 1024))
    parser.add_argument("--shard-size", type=int, default=200)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Drop sessions/characters with fewer than this many frames (sessions-root only).",
    )
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=2,
        help="Number of characters to hold out for unseen-identity eval.",
    )
    parser.add_argument(
        "--holdout-ids",
        type=str,
        default="",
        help="Comma-separated character_ids to hold out (overrides --holdout-count).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Store raw params without per-dim [-1,1] normalization.",
    )
    args = parser.parse_args()

    samples: list[tuple] = []
    if args.session is not None:
        session_dir = args.session.resolve()
        if not (session_dir / "values").is_dir() or not (session_dir / "images").is_dir():
            raise SystemExit(f"Not a VTS session (need images/ + values/): {session_dir}")
        samples = list(_iter_from_session(session_dir))
    elif args.sessions_root is not None:
        root = args.sessions_root.resolve()
        sessions = _discover_sessions(root)
        if not sessions:
            raise SystemExit(f"No VTS sessions found under {root}")
        by_char: dict[str, list[tuple]] = defaultdict(list)
        char_names: dict[str, str] = {}
        for session_dir in sessions:
            character_id, display_name = _read_session_identity(session_dir)
            char_names[character_id] = display_name
            sess_samples = list(_iter_from_session(session_dir, character_id, display_name))
            by_char[character_id].extend(sess_samples)
            _safe_print(f"  {session_dir.name}: {len(sess_samples)} frames -> {character_id} ({display_name})")

        kept = {cid: rows for cid, rows in by_char.items() if len(rows) >= args.min_samples}
        dropped = sorted(set(by_char) - set(kept))
        if dropped:
            _safe_print(f"Dropped {len(dropped)} characters under min-samples={args.min_samples}:")
            for cid in dropped:
                _safe_print(f"  - {cid}: {len(by_char[cid])} frames ({char_names.get(cid, cid)})")

        if not kept:
            raise SystemExit("No characters left after min-samples filter.")

        if args.holdout_ids.strip():
            holdout_ids = [x.strip() for x in args.holdout_ids.split(",") if x.strip()]
        else:
            # Deterministic: sort by id, take last N (stable across rebuilds).
            sorted_ids = sorted(kept.keys())
            holdout_ids = sorted_ids[-max(0, args.holdout_count) :] if args.holdout_count > 0 else []

        missing = [cid for cid in holdout_ids if cid not in kept]
        if missing:
            raise SystemExit(f"holdout ids not found after filter: {missing}")

        train_samples = []
        holdout_samples = []
        for cid, rows in kept.items():
            if cid in holdout_ids:
                holdout_samples.extend(rows)
            else:
                train_samples.extend(rows)

        _safe_print(
            f"Train characters={len(kept) - len(holdout_ids)} frames={len(train_samples)} | "
            f"Holdout characters={len(holdout_ids)} frames={len(holdout_samples)}"
        )
        for cid in holdout_ids:
            _safe_print(f"  holdout: {cid} ({char_names.get(cid, cid)}) n={len(kept[cid])}")

        samples = train_samples
        # Stash for later write.
        args._holdout_samples = holdout_samples  # type: ignore[attr-defined]
        args._holdout_ids = holdout_ids  # type: ignore[attr-defined]
        args._char_names = char_names  # type: ignore[attr-defined]
        args._kept = kept  # type: ignore[attr-defined]
    else:
        captions_jsonl = args.captions_jsonl.resolve()
        images_dir = (args.images_dir or captions_jsonl.parent).resolve()
        samples = list(_iter_from_captions(captions_jsonl, images_dir))

    if not samples and not getattr(args, "_holdout_samples", None):
        raise SystemExit("No samples found.")

    # Norm stats from ALL kept data (train + holdout) so inference uses one scale.
    all_for_stats = list(samples)
    if hasattr(args, "_holdout_samples"):
        all_for_stats = all_for_stats + list(args._holdout_samples)
    raw_params = [p for *_, p in all_for_stats]
    mins, maxs = _compute_norm_stats(raw_params)

    output_dir = args.output_dir.resolve()
    written = _write_split(
        samples,
        output_dir,
        args.image_size,
        args.shard_size,
        mins,
        maxs,
        args.no_normalize,
    )
    print(f"Wrote {written} train examples to {output_dir}")

    if hasattr(args, "_holdout_samples") and args.holdout_dir is not None:
        holdout_dir = args.holdout_dir.resolve()
        h_written = _write_split(
            list(args._holdout_samples),
            holdout_dir,
            args.image_size,
            args.shard_size,
            mins,
            maxs,
            args.no_normalize,
        )
        split_info = {
            "train_dir": str(output_dir),
            "holdout_dir": str(holdout_dir),
            "holdout_ids": list(args._holdout_ids),
            "holdout_names": {cid: args._char_names.get(cid, cid) for cid in args._holdout_ids},
            "train_characters": sorted(set(args._kept) - set(args._holdout_ids)),
            "train_samples": written,
            "holdout_samples": h_written,
            "param_mins": mins,
            "param_maxs": maxs,
            "normalized": not args.no_normalize,
            "image_size": args.image_size,
        }
        parent = holdout_dir.parent
        (parent / "split_info.json").write_text(
            json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir.parent / "split_info.json").write_text(
            json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {h_written} holdout examples to {holdout_dir}")
        print(f"split_info: {parent / 'split_info.json'}")


if __name__ == "__main__":
    main()
