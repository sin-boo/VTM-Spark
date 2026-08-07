"""Delete incomplete ``*_original`` pose images and matching label JSONs.

Scans ``raw_root/<character>/{images,labels}`` and removes any file whose
pose stem ends with ``_original`` (e.g. ``000001_original.jpg`` and
``000001_original_full_stack.json``).

Usage::

  # Preview only
  python -m scripts.remove_original_samples --raw-root ../data/raw --dry-run

  # Delete images + matching labels
  python -m scripts.remove_original_samples --raw-root ../data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LABEL_SUFFIXES = (
    "_full_stack.json",
    "_landmarks.json",
    "_iris.json",
    "_upper_body_pose.json",
    "_keypoints.npy",
    "_meta.json",
)


def is_original_stem(stem: str) -> bool:
    return stem.endswith("_original")


def discover_character_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root not found: {raw_root}")
    return [p for p in sorted(raw_root.iterdir()) if p.is_dir() and not p.name.startswith(".")]


def collect_original_targets(char_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (files_to_delete, orphan_labels_without_image)."""
    images_dir = char_dir / "images"
    labels_dir = char_dir / "labels"
    to_delete: list[Path] = []
    orphans: list[Path] = []
    seen_stems: set[str] = set()

    search_dirs = []
    if images_dir.is_dir():
        search_dirs.append(images_dir)
    else:
        search_dirs.append(char_dir)

    for d in search_dirs:
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            if not is_original_stem(p.stem):
                continue
            to_delete.append(p)
            seen_stems.add(p.stem)
            label_root = labels_dir if labels_dir.is_dir() else char_dir
            for suffix in LABEL_SUFFIXES:
                lab = label_root / f"{p.stem}{suffix}"
                if lab.is_file():
                    to_delete.append(lab)

    # Labels that look like originals but have no matching image.
    label_roots = [labels_dir] if labels_dir.is_dir() else [char_dir]
    for label_root in label_roots:
        for p in sorted(label_root.iterdir()):
            if not p.is_file():
                continue
            name = p.name
            stem = None
            for suffix in LABEL_SUFFIXES:
                if name.endswith(suffix):
                    stem = name[: -len(suffix)]
                    break
            if stem is None or not is_original_stem(stem):
                continue
            if stem in seen_stems:
                continue
            orphans.append(p)
            to_delete.append(p)

    # Deduplicate while preserving order.
    uniq: list[Path] = []
    seen_paths: set[Path] = set()
    for p in to_delete:
        rp = p.resolve()
        if rp in seen_paths:
            continue
        seen_paths.add(rp)
        uniq.append(p)
    return uniq, orphans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be deleted.")
    parser.add_argument("--characters", nargs="*", default=None, help="Optional character_id filter.")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    chars = discover_character_dirs(raw_root)
    if args.characters:
        allow = set(args.characters)
        chars = [c for c in chars if c.name in allow]

    n_files = 0
    n_chars = 0
    n_orphans = 0
    for char_dir in chars:
        targets, orphans = collect_original_targets(char_dir)
        if not targets:
            continue
        n_chars += 1
        n_orphans += len(orphans)
        print(f"{char_dir.name}: {len(targets)} file(s)")
        for p in targets:
            rel = p.relative_to(raw_root) if p.is_relative_to(raw_root) else p
            action = "DRY" if args.dry_run else "DEL"
            print(f"  [{action}] {rel}")
            if not args.dry_run:
                p.unlink(missing_ok=True)
            n_files += 1

    mode = "would delete" if args.dry_run else "deleted"
    print(f"{mode} {n_files} file(s) across {n_chars} character folder(s); orphan labels={n_orphans}")


if __name__ == "__main__":
    main()
