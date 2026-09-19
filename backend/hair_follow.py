"""Bind character hair polygons once, then follow the head with offsets.

The segmentation model only authors a rest mesh (three parts: middle / left /
right). Live frames warp those polygons in the current face frame so hair
tracks the character without running the detector again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .pose_controller import L_EYE, R_EYE, face_center, face_height

HAIR_CLASSES = ("hair_middle", "hair_left", "hair_right")


def _as_xy(poly: Sequence) -> np.ndarray:
    arr = np.asarray(poly, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[:, :2]


def _mean_xy(k: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray | None:
    pts = []
    for i in idxs:
        if i < int(k.shape[0]) and float(k[i, 3]) >= 0.5:
            pts.append(k[i, :2].astype(np.float64))
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def face_frame(keypoints: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Head frame: origin at face COM, +X along the eye line, units of face height."""
    k = np.asarray(keypoints, dtype=np.float32)
    center = face_center(k).astype(np.float64)
    height = max(float(face_height(k)), 1e-3)
    el = _mean_xy(k, L_EYE)
    er = _mean_xy(k, R_EYE)
    if el is not None and er is not None:
        angle = float(np.arctan2(float(er[1] - el[1]), float(er[0] - el[0])))
    else:
        angle = 0.0
    return center, height, angle


def _to_local(pts: np.ndarray, center: np.ndarray, height: float, angle: float) -> np.ndarray:
    d = pts.astype(np.float64) - center
    ca, sa = float(np.cos(-angle)), float(np.sin(-angle))
    x = d[:, 0] * ca - d[:, 1] * sa
    y = d[:, 0] * sa + d[:, 1] * ca
    return np.stack([x, y], axis=1) / max(float(height), 1e-3)


def _from_local(
    local: np.ndarray, center: np.ndarray, height: float, angle: float
) -> np.ndarray:
    p = local.astype(np.float64) * float(height)
    ca, sa = float(np.cos(angle)), float(np.sin(angle))
    x = p[:, 0] * ca - p[:, 1] * sa
    y = p[:, 0] * sa + p[:, 1] * ca
    return (np.stack([x, y], axis=1) + center).astype(np.float32)


@dataclass
class HairPart:
    cls: str
    local: np.ndarray  # (N, 2) in rest face-height units


@dataclass
class HairRig:
    parts: list[HairPart]
    rest_center: np.ndarray
    rest_height: float
    rest_angle: float


def build_hair_rig(
    segments: Sequence[Mapping[str, Any]] | None,
    rest_keypoints: np.ndarray,
) -> HairRig | None:
    """Author rest-pose hair in the character head frame. Returns None if empty."""
    center, height, angle = face_frame(rest_keypoints)
    parts: list[HairPart] = []
    for seg in segments or []:
        cls = str(seg.get("class") or "")
        pts = _as_xy(seg.get("polygon") or [])
        if cls not in HAIR_CLASSES or pts.shape[0] < 3:
            continue
        parts.append(HairPart(cls=cls, local=_to_local(pts, center, height, angle)))
    if not parts:
        return None
    return HairRig(
        parts=parts,
        rest_center=center.astype(np.float32),
        rest_height=float(height),
        rest_angle=float(angle),
    )


def follow_hair(
    rig: HairRig | None,
    driven_keypoints: np.ndarray,
) -> list[dict[str, Any]]:
    """Rebuild the three hair parts in the current character head frame."""
    if rig is None or not rig.parts:
        return []
    center, height, angle = face_frame(driven_keypoints)
    out: list[dict[str, Any]] = []
    for part in rig.parts:
        pts = _from_local(part.local, center, height, angle)
        out.append({"class": part.cls, "polygon": pts.astype(np.float32).tolist()})
    return out


def rest_hair(rig: HairRig | None) -> list[dict[str, Any]]:
    """Author rest polygons with no live follow (the captured silhouette)."""
    if rig is None or not rig.parts:
        return []
    center = np.asarray(rig.rest_center, dtype=np.float64)
    out: list[dict[str, Any]] = []
    for part in rig.parts:
        pts = _from_local(
            part.local, center, float(rig.rest_height), float(rig.rest_angle)
        )
        out.append({"class": part.cls, "polygon": pts.astype(np.float32).tolist()})
    return out
