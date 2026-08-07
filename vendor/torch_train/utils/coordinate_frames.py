"""Canonical coordinate frames for KEYPOINT_SCHEMA live ↔ training.

Spaces
------
- ``pixel_src``  — absolute pixels in the source image (post-mirror)
- ``norm_full``  — [-1, 1] vs full source W×H (debug / legacy only)
- ``norm_crop``  — [-1, 1] vs square crop of size ``image_size`` (MODEL SPACE)

All tensors reaching ``denoise_keypoint`` / training must be ``norm_crop``.
Live webcam frames use a centered pad-to-square virtual crop so x/y units are
isotropic and compatible with anime reference / training space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .keypoints import NUM_KEYPOINTS, KEYPOINT_DIM, transform_keypoints_crop

COORD_PIXEL_SRC = "pixel_src"
COORD_NORM_FULL = "norm_full"
COORD_NORM_CROP = "norm_crop"

DEFAULT_IMAGE_SIZE = 768


@dataclass(frozen=True)
class CropRect:
    """Virtual or real crop window in source-pixel coordinates.

    For pad-to-square, ``x0``/``y0`` may be negative (content is inset on a
    larger square canvas). ``w`` and ``h`` are the crop side lengths before
    resize to ``out_w`` × ``out_h``.
    """

    x0: float
    y0: float
    w: float
    h: float
    out_w: float = float(DEFAULT_IMAGE_SIZE)
    out_h: float = float(DEFAULT_IMAGE_SIZE)
    mode: str = "pad_square"

    def as_dict(self) -> dict[str, Any]:
        return {
            "x0": float(self.x0),
            "y0": float(self.y0),
            "w": float(self.w),
            "h": float(self.h),
            "out_w": float(self.out_w),
            "out_h": float(self.out_h),
            "mode": str(self.mode),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> CropRect | None:
        if not isinstance(d, dict):
            return None
        try:
            return cls(
                x0=float(d["x0"]),
                y0=float(d["y0"]),
                w=float(d["w"]),
                h=float(d["h"]),
                out_w=float(d.get("out_w", DEFAULT_IMAGE_SIZE)),
                out_h=float(d.get("out_h", DEFAULT_IMAGE_SIZE)),
                mode=str(d.get("mode", "pad_square")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def identity_key(self) -> tuple:
        """Hashable identity for detecting crop/resolution changes."""
        return (
            round(self.x0, 3),
            round(self.y0, 3),
            round(self.w, 3),
            round(self.h, 3),
            round(self.out_w, 3),
            round(self.out_h, 3),
            self.mode,
        )


def square_pad_offsets(width: int, height: int) -> tuple[int, int, int]:
    """Return ``(pad_x, pad_y, side)`` for centered pad-to-square."""
    w = max(int(width), 1)
    h = max(int(height), 1)
    side = max(w, h)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    return pad_x, pad_y, side


def pad_square_crop_rect(
    width: int,
    height: int,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> CropRect:
    """Centered pad-to-square crop matching ``encode_reference(skip_crop=True)``.

    A 1280×720 frame becomes a 1280×1280 virtual canvas with the content
    centered; keypoints then normalize into ``image_size`` square space.
    """
    pad_x, pad_y, side = square_pad_offsets(width, height)
    return CropRect(
        x0=float(-pad_x),
        y0=float(-pad_y),
        w=float(side),
        h=float(side),
        out_w=float(image_size),
        out_h=float(image_size),
        mode="pad_square",
    )


def to_norm_crop(
    kps_px: np.ndarray,
    crop: CropRect,
) -> np.ndarray:
    """Map pixel-space keypoints through ``crop`` into model ``norm_crop``."""
    return transform_keypoints_crop(
        kps_px,
        crop_x0=crop.x0,
        crop_y0=crop.y0,
        crop_w=crop.w,
        crop_h=crop.h,
        out_w=crop.out_w,
        out_h=crop.out_h,
        normalize=True,
    )


def to_norm_full(
    kps_px: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Map absolute pixels into full-frame [-1, 1] (debug / legacy only)."""
    out = np.asarray(kps_px, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != KEYPOINT_DIM:
        raise ValueError(f"Expected (N, {KEYPOINT_DIM}), got {out.shape}")
    w = max(float(image_width), 1.0)
    h = max(float(image_height), 1.0)
    out[:, 0] = 2.0 * (out[:, 0] / w) - 1.0
    out[:, 1] = 2.0 * (out[:, 1] / h) - 1.0
    return out


def webcam_pixels_to_norm_crop(
    kps_px: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, CropRect]:
    """Convenience: pad-square + normalize webcam keypoints to model space."""
    crop = pad_square_crop_rect(image_width, image_height, image_size=image_size)
    return to_norm_crop(kps_px, crop), crop


def assert_norm_crop(coord_space: str | None, *, label: str = "keypoints") -> None:
    """Raise if ``coord_space`` is not model ``norm_crop``."""
    space = str(coord_space or "").strip().lower()
    if space != COORD_NORM_CROP:
        raise ValueError(
            f"{label} must be in '{COORD_NORM_CROP}' space for model retarget; "
            f"got '{coord_space or 'unset'}'"
        )


def crop_rects_equal(a: CropRect | None, b: CropRect | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.identity_key() == b.identity_key()


def coord_meta_dict(
    *,
    coord_space: str,
    source_wh: tuple[int, int],
    crop: CropRect | None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mirrored: bool = False,
    calibrated: bool = False,
    skeleton_method: str = "none",
    body_lost: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard metadata envelope for live / bridge packets."""
    meta: dict[str, Any] = {
        "coord_space": str(coord_space),
        "source_wh": [int(source_wh[0]), int(source_wh[1])],
        "image_size": int(image_size),
        "mirrored": bool(mirrored),
        "calibrated": bool(calibrated),
        "skeleton_method": str(skeleton_method or "none"),
        "body_lost": bool(body_lost),
    }
    if crop is not None:
        meta["crop"] = crop.as_dict()
    if extra:
        meta.update(extra)
    return meta


__all__ = [
    "COORD_PIXEL_SRC",
    "COORD_NORM_FULL",
    "COORD_NORM_CROP",
    "DEFAULT_IMAGE_SIZE",
    "CropRect",
    "square_pad_offsets",
    "pad_square_crop_rect",
    "to_norm_crop",
    "to_norm_full",
    "webcam_pixels_to_norm_crop",
    "assert_norm_crop",
    "crop_rects_equal",
    "coord_meta_dict",
    "NUM_KEYPOINTS",
    "KEYPOINT_DIM",
    "asdict",
]
