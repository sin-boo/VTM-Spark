"""Hair-part polygons for pose-map conditioning.

Classes match the pose-tracker hair model:
  0 hair_middle  (bangs / center)
  1 hair_left    (screen-left)
  2 hair_right   (screen-right)

Polygons are stored in the same crop space as keypoints (norm_crop [-1, 1]).
They are rasterized into 3 extra pose-map channels (after the 8 keypoint channels).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HAIR_CLASSES: tuple[str, ...] = (
    "hair_middle",
    "hair_left",
    "hair_right",
)
NUM_HAIR_CHANNELS = len(HAIR_CLASSES)
HAIR_CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(HAIR_CLASSES)}
HAIR_CLASS_ALIASES: dict[str, str] = {
    "bangs": "hair_middle",
    "middle": "hair_middle",
    "left": "hair_left",
    "right": "hair_right",
    "hair_bangs": "hair_middle",
}
# After a horizontal flip, swap left/right channel indices.
HAIR_FLIP_CHANNEL_ORDER: tuple[int, ...] = (0, 2, 1)
HAIR_MAP_SIZE = 96
HAIR_SWAP_LR = {"hair_left": "hair_right", "hair_right": "hair_left"}


def normalize_hair_class(name: str | None) -> str | None:
    if not name:
        return None
    key = str(name).strip().lower()
    key = HAIR_CLASS_ALIASES.get(key, key)
    return key if key in HAIR_CLASS_TO_INDEX else None


def empty_hair_maps(height: int = HAIR_MAP_SIZE, width: int | None = None, dtype=np.float32) -> np.ndarray:
    w = int(width if width is not None else height)
    return np.zeros((NUM_HAIR_CHANNELS, int(height), w), dtype=dtype)


def hair_from_full_stack(data: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return ``[{class, polygon: [[x,y], ...]}, ...]`` in source-pixel space."""
    if not data:
        return []
    hair = data.get("hair") or {}
    segments = hair.get("segments") or hair.get("polygons") or []
    out: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, Mapping):
            continue
        cls = normalize_hair_class(seg.get("class") or seg.get("label") or seg.get("name"))
        poly = seg.get("polygon") or seg.get("points") or []
        pts = _as_xy(poly)
        if cls is None or pts.shape[0] < 3:
            continue
        out.append({"class": cls, "polygon": pts.tolist()})
    return out


def load_hair_from_label_dir(label_dir: Path | str, pose_id: str) -> list[dict[str, Any]]:
    """Load source-pixel hair polygons from ``*_full_stack.json``."""
    label_dir = Path(label_dir)
    full = label_dir / f"{pose_id}_full_stack.json"
    if full.is_file():
        with open(full, "r", encoding="utf-8") as f:
            return hair_from_full_stack(json.load(f))
    return []


def transform_hair_crop(
    segments: Sequence[Mapping[str, Any]],
    *,
    crop_x0: float,
    crop_y0: float,
    crop_w: float,
    crop_h: float,
    out_w: float,
    out_h: float,
    normalize: bool = True,
) -> list[dict[str, Any]]:
    """Map source-pixel polygons through the same crop used for keypoints."""
    sx = float(out_w) / max(float(crop_w), 1e-6)
    sy = float(out_h) / max(float(crop_h), 1e-6)
    out: list[dict[str, Any]] = []
    for seg in segments:
        cls = normalize_hair_class(seg.get("class"))
        pts = _as_xy(seg.get("polygon") or [])
        if cls is None or pts.shape[0] < 3:
            continue
        x = (pts[:, 0] - float(crop_x0)) * sx
        y = (pts[:, 1] - float(crop_y0)) * sy
        if normalize:
            x = 2.0 * (x / max(float(out_w), 1.0)) - 1.0
            y = 2.0 * (y / max(float(out_h), 1.0)) - 1.0
        out.append({"class": cls, "polygon": np.stack([x, y], axis=1).astype(np.float32).tolist()})
    return out


def transform_hair_crop_rect(
    segments: Sequence[Mapping[str, Any]],
    crop: Any,
    *,
    normalize: bool = True,
) -> list[dict[str, Any]]:
    """``transform_hair_crop`` using a ``CropRect`` / dict with x0,y0,w,h,out_w,out_h."""
    if crop is None:
        return []
    if hasattr(crop, "as_dict"):
        d = crop.as_dict()
    elif isinstance(crop, Mapping):
        d = crop
    else:
        return []
    return transform_hair_crop(
        segments,
        crop_x0=float(d.get("x0", 0)),
        crop_y0=float(d.get("y0", 0)),
        crop_w=float(d.get("w", 1)),
        crop_h=float(d.get("h", 1)),
        out_w=float(d.get("out_w", d.get("w", 1))),
        out_h=float(d.get("out_h", d.get("h", 1))),
        normalize=normalize,
    )


def hair_payload(segments: Sequence[Mapping[str, Any]], *, coord_space: str = "norm_crop") -> dict[str, Any]:
    return {
        "schema": "hair_v1",
        "coord_space": coord_space,
        "classes": list(HAIR_CLASSES),
        "segments": [
            {"class": s["class"], "polygon": s["polygon"]}
            for s in segments
            if normalize_hair_class(s.get("class")) and _as_xy(s.get("polygon") or []).shape[0] >= 3
        ],
    }


def load_normalized_hair(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(payload.get("segments") or [])


def flip_hair_horizontal(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mirror x → -x and swap hair_left ↔ hair_right (norm_crop space)."""
    out: list[dict[str, Any]] = []
    for seg in segments:
        cls = normalize_hair_class(seg.get("class"))
        pts = _as_xy(seg.get("polygon") or [])
        if cls is None or pts.shape[0] < 3:
            continue
        pts = pts.copy()
        pts[:, 0] = -pts[:, 0]
        out.append({"class": HAIR_SWAP_LR.get(cls, cls), "polygon": pts.tolist()})
    return out


def flip_hair_pixels(segments: Sequence[Mapping[str, Any]], width: float) -> list[dict[str, Any]]:
    """Mirror polygons in source-pixel space and swap hair_left ↔ hair_right."""
    w = float(width)
    out: list[dict[str, Any]] = []
    for seg in segments:
        cls = normalize_hair_class(seg.get("class"))
        pts = _as_xy(seg.get("polygon") or [])
        if cls is None or pts.shape[0] < 3:
            continue
        pts = pts.copy()
        pts[:, 0] = (w - 1.0) - pts[:, 0]
        rec = {"class": HAIR_SWAP_LR.get(cls, cls), "polygon": pts.tolist()}
        if "score" in seg:
            rec["score"] = seg["score"]
        if "area" in seg:
            rec["area"] = seg["area"]
        out.append(rec)
    return out


def flip_hair_maps(maps: np.ndarray):
    """Horizontal flip raster hair maps ``(3,H,W)`` or ``(B,3,H,W)`` and swap L/R."""
    out = np.flip(maps, axis=-1).copy()
    order = list(HAIR_FLIP_CHANNEL_ORDER)
    if out.ndim == 3:
        return out[order]
    if out.ndim == 4:
        return out[:, order]
    raise ValueError(f"Expected (3,H,W) or (B,3,H,W), got {out.shape}")


def flip_hair_maps_torch(maps):
    """Torch version of ``flip_hair_maps``."""
    import torch

    out = torch.flip(maps, dims=[-1])
    order = torch.tensor(HAIR_FLIP_CHANNEL_ORDER, device=out.device, dtype=torch.long)
    if out.ndim == 3:
        return out.index_select(0, order)
    if out.ndim == 4:
        return out.index_select(1, order)
    raise ValueError(f"Expected (3,H,W) or (B,3,H,W), got {tuple(out.shape)}")


def rasterize_hair_maps(
    segments: Sequence[Mapping[str, Any]],
    height: int = HAIR_MAP_SIZE,
    width: int | None = None,
) -> np.ndarray:
    """Fill polygons into ``(3, H, W)`` float32 maps in [0, 1]. Coords are norm_crop."""
    from PIL import Image, ImageDraw

    w = int(width if width is not None else height)
    h = int(height)
    canvases = [Image.new("L", (w, h), 0) for _ in HAIR_CLASSES]
    draws = [ImageDraw.Draw(im) for im in canvases]
    for seg in segments:
        cls = normalize_hair_class(seg.get("class"))
        pts = _as_xy(seg.get("polygon") or [])
        if cls is None or pts.shape[0] < 3:
            continue
        xy: list[tuple[float, float]] = []
        for x, y in pts:
            px = float((x + 1.0) * 0.5 * (w - 1))
            py = float((y + 1.0) * 0.5 * (h - 1))
            xy.append((px, py))
        draws[HAIR_CLASS_TO_INDEX[cls]].polygon(xy, fill=255)
    maps = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in canvases], axis=0)
    return maps


def _as_xy(poly: Sequence) -> np.ndarray:
    if poly is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2]
