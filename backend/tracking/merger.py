"""Merge per-model tracks into one KEYPOINT_SCHEMA tensor for the generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..paths import ensure_import_paths

ensure_import_paths()

from bridge import BridgeFrame, build_bridge_frame  # noqa: E402
from utils.coordinate_frames import (  # noqa: E402
    COORD_NORM_CROP,
    DEFAULT_IMAGE_SIZE,
    CropRect,
    coord_meta_dict,
    webcam_pixels_to_norm_crop,
)
from utils.hair import transform_hair_crop_rect  # noqa: E402

from .body import BodyTrack
from .face import FaceTrack
from .hair import HairTrack, clone_hair_segments
from .iris import IrisTrack


@dataclass
class MergedTrack:
    """Unified tracking packet consumed by the live generator."""

    bridge: BridgeFrame
    keypoints_px: np.ndarray
    keypoints_norm: np.ndarray
    crop: CropRect | None
    hair_segments_px: list[dict[str, Any]] | None = None
    hair_segments_norm: list[dict[str, Any]] | None = None


def merge_tracks(
    *,
    image_shape: tuple[int, ...],
    face: FaceTrack | None = None,
    iris: IrisTrack | None = None,
    body: BodyTrack | None = None,
    hair: HairTrack | None = None,
    mirrored: bool = False,
    pose: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    calibrated: bool = False,
) -> MergedTrack:
    """Pack face + iris + body + hair into (37,4) pixels and norm_crop."""
    pts28 = None if face is None else face.pts28
    right = None if iris is None else iris.right
    left = None if iris is None else iris.left
    body7 = None if body is None else body.joints
    hair_segs = None if hair is None or not hair.segments else hair.segments
    iris_method = "none" if iris is None else iris.method
    skeleton_method = "none" if body is None else body.method
    hair_method = "none" if hair is None else hair.method
    body_lost = True if body is None else bool(body.lost)

    extra = dict(meta or {})
    extra.setdefault("body_lost", body_lost)

    bridge = build_bridge_frame(
        pts28,
        image_shape,
        right_iris=right,
        left_iris=left,
        body7=body7,
        mirrored=mirrored,
        iris_method=iris_method,
        skeleton_method=skeleton_method,
        hair_method=hair_method,
        hair_segments=hair_segs,
        pose=pose or {},
        meta=extra,
    )
    kps_px = bridge.keypoints.copy()
    src_w = int(bridge.image_wh[0])
    src_h = int(bridge.image_wh[1])
    kps_norm, crop = webcam_pixels_to_norm_crop(
        kps_px, src_w, src_h, image_size=int(image_size)
    )
    bridge.keypoints_norm = kps_norm.copy()
    bridge.coord_space = COORD_NORM_CROP
    bridge.crop = crop
    bridge.image_size = int(image_size)
    bridge.meta.update(
        coord_meta_dict(
            coord_space=COORD_NORM_CROP,
            source_wh=(src_w, src_h),
            crop=crop,
            image_size=int(image_size),
            mirrored=mirrored,
            calibrated=bool(calibrated),
            skeleton_method=skeleton_method,
            body_lost=body_lost,
        )
    )
    hair_norm = (
        transform_hair_crop_rect(hair_segs, crop)
        if hair_segs and crop is not None
        else None
    )
    return MergedTrack(
        bridge=bridge,
        keypoints_px=kps_px,
        keypoints_norm=kps_norm,
        crop=crop,
        hair_segments_px=None if not hair_segs else clone_hair_segments(hair_segs),
        hair_segments_norm=hair_norm,
    )
