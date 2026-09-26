"""Still-space hair, skeleton, and limiter boxes for the create fit step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .character_pack import CharacterPackError, character_still_path
from .pose_controller import face_height
from .travel_box import (
    body_mesh_rect_norm,
    body_rect_norm,
    head_mesh_rect_norm,
    head_rect_norm,
    normalize_travel_box,
)

SKELETON_LABELS = {
    31: "Neck",
    32: "R shoulder",
    33: "R elbow",
    34: "L shoulder",
    35: "L elbow",
    36: "Chest",
}

FIT_NAME = "fit.json"


def character_fit_path(ident: str, *, dest_dir: Path | None = None) -> Path:
    return character_still_path(ident, dest_dir=dest_dir).with_name(FIT_NAME)


def read_character_fit(ident: str, *, dest_dir: Path | None = None) -> dict[str, Any]:
    path = character_fit_path(ident, dest_dir=dest_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_character_fit(ident: str, data: dict[str, Any], *, dest_dir: Path | None = None) -> Path:
    path = character_fit_path(ident, dest_dir=dest_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def update_character_fit(ident: str, patch: dict[str, Any], *, dest_dir: Path | None = None) -> Path:
    current = read_character_fit(ident, dest_dir=dest_dir)
    current.update(patch)
    return write_character_fit(ident, current, dest_dir=dest_dir)


def _px_rect(
    rect: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> list[float] | None:
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    return [
        round((float(x0) + 1.0) * 0.5 * w, 1),
        round((float(y0) + 1.0) * 0.5 * h, 1),
        round((float(x1) + 1.0) * 0.5 * w, 1),
        round((float(y1) + 1.0) * 0.5 * h, 1),
    ]


def norm_hair_to_pixels(
    segments: list | None,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        cls = str(seg.get("class") or "")
        poly = []
        for vertex in seg.get("polygon") or []:
            if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                continue
            try:
                poly.append(
                    [
                        round((float(vertex[0]) + 1.0) * 0.5 * w, 1),
                        round((float(vertex[1]) + 1.0) * 0.5 * h, 1),
                    ]
                )
            except (TypeError, ValueError):
                continue
        if cls and len(poly) >= 3:
            out.append({"class": cls, "polygon": poly})
    return out


def pixels_hair_to_norm(
    segments: list | None,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        cls = str(seg.get("class") or "")
        poly = []
        for vertex in seg.get("polygon") or []:
            if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                continue
            try:
                poly.append(
                    [
                        round(float(vertex[0]) / w * 2.0 - 1.0, 5),
                        round(float(vertex[1]) / h * 2.0 - 1.0, 5),
                    ]
                )
            except (TypeError, ValueError):
                continue
        if cls and len(poly) >= 3:
            out.append({"class": cls, "polygon": poly})
    return out


def build_fit_view(
    *,
    width: int,
    height: int,
    keypoints: np.ndarray | None,
    hair_norm: list | None,
    box: Any,
) -> dict[str, Any]:
    """Pixel overlay the fit canvas draws on the character still."""
    spec = normalize_travel_box(box)
    w = max(int(width), 1)
    h = max(int(height), 1)
    k = None if keypoints is None else np.asarray(keypoints, dtype=np.float32)
    fh = 1e-3
    if k is not None and k.ndim == 2 and k.shape[0] >= 37:
        fh = max(float(face_height(k)), 1e-3)
    skeleton: list[dict[str, Any]] = []
    if k is not None and k.ndim == 2 and k.shape[0] >= 37:
        for idx, label in SKELETON_LABELS.items():
            skeleton.append(
                {
                    "id": idx,
                    "label": label,
                    "x": round((float(k[idx, 0]) + 1.0) * 0.5 * w, 1),
                    "y": round((float(k[idx, 1]) + 1.0) * 0.5 * h, 1),
                }
            )
    head_tight = head_mesh_rect_norm(k) if k is not None else None
    head = head_rect_norm(k, spec) if k is not None else None
    body_tight = body_mesh_rect_norm(k) if k is not None else None
    body = body_rect_norm(k, spec) if k is not None else None
    return {
        "width": w,
        "height": h,
        "face_height": round(fh, 6),
        "hair": norm_hair_to_pixels(hair_norm, w, h),
        "skeleton": skeleton,
        "boxes": {
            "head_tight": _px_rect(head_tight, w, h),
            "head": _px_rect(head, w, h),
            "body_tight": _px_rect(body_tight, w, h),
            "body": _px_rect(body, w, h),
        },
    }


def replace_pack_keypoints(path: Path, keypoints: np.ndarray) -> None:
    """Swap ``keypoints.npy`` inside a ``.vtm`` without touching the latents."""
    import io
    import zipfile

    from .character_pack import KEYPOINTS_NAME, _read_zip

    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.shape != (37, 4):
        raise CharacterPackError(f"Expected keypoints (37, 4), got {kps.shape}")
    buf = io.BytesIO()
    np.save(buf, kps)
    raw = buf.getvalue()
    zf = _read_zip(path)
    try:
        others = [(name, zf.read(name)) for name in zf.namelist() if name != KEYPOINTS_NAME]
    finally:
        zf.close()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as dest:
        for name, data in others:
            dest.writestr(name, data)
        dest.writestr(KEYPOINTS_NAME, raw)
    path.write_bytes(out.getvalue())
