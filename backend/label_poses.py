"""Load training ``*_full_stack.json`` poses for Generate once."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_LABEL_DIR = Path(
    os.environ.get("VTM_LABEL_DIR", r"E:\A-data\input\00001\labels")
)


class LabelPoseError(ValueError):
    """Invalid label folder, file, or payload."""


def label_dir(folder: Path | str | None = None) -> Path:
    return Path(folder) if folder is not None else DEFAULT_LABEL_DIR


def list_label_files(*, dest_dir: Path | str | None = None) -> list[dict[str, str]]:
    folder = label_dir(dest_dir)
    if not folder.is_dir():
        return []
    out: list[dict[str, str]] = []
    for path in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json"):
        stem = path.stem
        name = stem.removesuffix("_full_stack")
        out.append({"id": path.name, "name": name, "path": str(path)})
    return out


def resolve_label_file(ident: str, *, dest_dir: Path | str | None = None) -> Path:
    folder = label_dir(dest_dir).resolve()
    name = Path(str(ident or "").strip()).name
    if not name or name in {".", ".."}:
        raise LabelPoseError("Invalid label id")
    path = (folder / name).resolve()
    if path.parent != folder or not path.is_file():
        raise LabelPoseError(f"Label not found: {name}")
    return path


def _hair_to_norm(segments: list[dict[str, Any]], crop) -> list[dict[str, Any]]:
    w = max(float(crop.w), 1e-6)
    h = max(float(crop.h), 1e-6)
    out: list[dict[str, Any]] = []
    for seg in segments:
        poly = []
        for x, y in seg.get("polygon") or []:
            nx = 2.0 * ((float(x) - float(crop.x0)) / w) - 1.0
            ny = 2.0 * ((float(y) - float(crop.y0)) / h) - 1.0
            poly.append([nx, ny])
        if len(poly) < 3:
            continue
        out.append({"class": seg["class"], "polygon": poly})
    return out


def load_label_for_generate(
    ident: str,
    *,
    dest_dir: Path | str | None = None,
    image_size: int = 768,
) -> dict[str, Any]:
    """Parse a full-stack label into model ``norm_crop`` keypoints and hair maps."""
    from utils.coordinate_frames import webcam_pixels_to_norm_crop
    from utils.hair import empty_hair_maps, hair_from_full_stack, rasterize_hair_maps
    from utils.keypoints import keypoints_from_full_stack

    path = resolve_label_file(ident, dest_dir=dest_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise LabelPoseError(f"Label is not an object: {path.name}")
    width = int(data.get("image_width") or 0)
    height = int(data.get("image_height") or 0)
    if width < 1 or height < 1:
        raise LabelPoseError(f"{path.name} is missing image_width / image_height")
    kps_px = keypoints_from_full_stack(data)
    kps, crop = webcam_pixels_to_norm_crop(
        kps_px, width, height, image_size=int(image_size or 768)
    )
    hair_px = hair_from_full_stack(data)
    hair_norm = _hair_to_norm(hair_px, crop)
    hair_maps = rasterize_hair_maps(hair_norm) if hair_norm else empty_hair_maps()
    return {
        "id": path.name,
        "name": path.stem.removesuffix("_full_stack"),
        "keypoints": np.asarray(kps, dtype=np.float32),
        "hair_maps": np.asarray(hair_maps, dtype=np.float32),
        "hair": hair_norm,
        "pose_id": str(data.get("pose_id") or path.stem.removesuffix("_full_stack")),
    }
