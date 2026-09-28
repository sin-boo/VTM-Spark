"""Still-space hair, tracking points, skeleton, and limiter boxes for the create fit step.

The fit (hair mask, skeleton, limiters) lives in the character's ``.vtm`` as
``fit.json``; the pack is the source of truth. Older installs kept it beside
the preview still: the first read folds that sidecar into the pack (pack keys
win) and deletes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .character_pack import (
    KEYPOINTS_NAME,
    CharacterPackError,
    character_still_path,
    read_pack_fit,
    replace_pack_members,
    resolve_character_id,
    upgrade_character_pack,
    write_pack_fit,
)
from .pose_controller import KEYPOINT_REFS, face_height
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
# Tracking points the fit canvas can move: face 0–27, irises 28–29. Labels
# match the live overlay so a point reads the same on both screens.
FACE_POINT_SLOTS = tuple(range(28))
IRIS_POINT_SLOTS = (28, 29)
POINT_SLOTS = FACE_POINT_SLOTS + IRIS_POINT_SLOTS + tuple(SKELETON_LABELS)

FIT_NAME = "fit.json"


def character_fit_path(ident: str, *, dest_dir: Path | None = None) -> Path:
    """Legacy sidecar beside the preview still."""
    return character_still_path(ident, dest_dir=dest_dir).with_name(FIT_NAME)


def _pack_path(ident: str, dest_dir: Path | None) -> Path | None:
    try:
        return resolve_character_id(ident, dest_dir=dest_dir)
    except CharacterPackError:
        return None


def _read_sidecar(ident: str, dest_dir: Path | None) -> dict[str, Any]:
    path = character_fit_path(ident, dest_dir=dest_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _fold_sidecar(ident: str, pack: Path, dest_dir: Path | None) -> None:
    """Move a legacy ``fit.json`` sidecar into the pack once, then delete it."""
    sidecar = character_fit_path(ident, dest_dir=dest_dir)
    if not sidecar.is_file():
        return
    legacy = _read_sidecar(ident, dest_dir)
    try:
        upgrade_character_pack(pack, fit=legacy)
    except (CharacterPackError, OSError):
        return  # keep the sidecar; the pack could not take it
    try:
        sidecar.unlink()
    except OSError:
        pass


def read_character_fit(ident: str, *, dest_dir: Path | None = None) -> dict[str, Any]:
    pack = _pack_path(ident, dest_dir)
    if pack is None:
        return _read_sidecar(ident, dest_dir)
    _fold_sidecar(ident, pack, dest_dir)
    try:
        return read_pack_fit(pack)
    except CharacterPackError:
        return _read_sidecar(ident, dest_dir)


def write_character_fit(ident: str, data: dict[str, Any], *, dest_dir: Path | None = None) -> Path:
    pack = _pack_path(ident, dest_dir)
    sidecar = character_fit_path(ident, dest_dir=dest_dir)
    if pack is not None:
        write_pack_fit(pack, data)
        if sidecar.is_file():
            sidecar.unlink()
        return pack
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return sidecar


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
    points: list[dict[str, Any]] = []
    if k is not None and k.ndim == 2 and k.shape[0] >= 37:

        def _px(idx: int) -> tuple[float, float]:
            return (
                round((float(k[idx, 0]) + 1.0) * 0.5 * w, 1),
                round((float(k[idx, 1]) + 1.0) * 0.5 * h, 1),
            )

        for idx, label in SKELETON_LABELS.items():
            x, y = _px(idx)
            skeleton.append({"id": idx, "label": label, "x": x, "y": y})
        for idx in POINT_SLOTS:
            if float(k[idx, 3]) < 0.5:
                continue
            group = "face" if idx in FACE_POINT_SLOTS else "iris" if idx in IRIS_POINT_SLOTS else "body"
            label = SKELETON_LABELS.get(idx) or KEYPOINT_REFS[idx]
            x, y = _px(idx)
            points.append({"id": idx, "label": label, "group": group, "x": x, "y": y})
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
        "points": points,
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

    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.shape != (37, 4):
        raise CharacterPackError(f"Expected keypoints (37, 4), got {kps.shape}")
    buf = io.BytesIO()
    np.save(buf, kps)
    replace_pack_members(path, {KEYPOINTS_NAME: buf.getvalue()})
