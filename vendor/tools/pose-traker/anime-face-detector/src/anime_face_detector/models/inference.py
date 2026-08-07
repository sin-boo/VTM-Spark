"""Inference utilities for HRNetV2 landmark detection."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import torch
from cv2.typing import MatLike


def get_affine_transform(center, scale, rot, output_size, shift=(0.0, 0.0), inv=False):
    """Get affine transformation matrix compatible with mmpose.

    Args:
        center: Center point coordinates (x, y)
        scale: Bounding box scale as [w, h] / 200.0
        rot: Rotation angle in degrees
        output_size: Output image size (width, height)
        shift: Shift offset (default: (0, 0))
        inv: If True, return inverse transformation matrix

    Returns:
        Affine transformation matrix (2x3)
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    # Scale back to pixel units
    scale_tmp = scale * 200.0

    # Use the width component for direction calculation
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # Calculate rotation
    rot_rad = np.pi * rot / 180

    # Source direction vector
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    # Destination direction vector
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    # Define 3 points for affine transform
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    # Point 1: center
    src[0, :] = center
    if isinstance(shift, (tuple, list)):
        shift = np.array(shift, dtype=np.float32)
    src[0, :] += scale_tmp * shift

    # Point 2: center + direction
    src[1, :] = center + src_dir
    src[1, :] += scale_tmp * shift

    # Destination points
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir

    # Point 3: perpendicular to form triangle
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    src_mat = cast(MatLike, np.float32(src))
    dst_mat = cast(MatLike, np.float32(dst))

    if inv:
        trans = cv2.getAffineTransform(dst_mat, src_mat)
    else:
        trans = cv2.getAffineTransform(src_mat, dst_mat)

    return trans


def get_dir(src_point, rot_rad):
    """Get direction vector after rotation.

    Args:
        src_point: Source point (x, y)
        rot_rad: Rotation angle in radians

    Returns:
        Direction vector after rotation
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    """Get the third point to form a right triangle.

    Args:
        a: First point
        b: Second point

    Returns:
        Third point coordinates
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def box_to_center_scale(box, image_size, scale_factor=1.25):
    """Convert bounding box to center and scale.

    Args:
        box: Bounding box [x0, y0, x1, y1]
        image_size: Image size (width, height)
        scale_factor: Scale factor for box expansion

    Returns:
        center: Center point (x, y)
        scale: Scale as array [w, h] / 200.0
    """
    x0, y0, x1, y1 = box[:4]
    box_width = x1 - x0
    box_height = y1 - y0
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float32)

    # Aspect ratio matching (square crop for 256x256 input)
    aspect_ratio = image_size[0] / image_size[1]  # typically 1.0 for 256x256
    if box_width > aspect_ratio * box_height:
        box_height = box_width / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio

    # Scale represents the bbox size relative to 200 pixels
    # mmpose uses 200.0 as the base scale
    scale = np.array([box_width, box_height], dtype=np.float32) / 200.0 * scale_factor

    return center, scale


def affine_transform_keypoints(keypoints, trans_matrix):
    """Apply affine transformation to keypoints.

    Args:
        keypoints: Keypoint coordinates, shape (K, 2)
        trans_matrix: Affine transformation matrix (2x3)

    Returns:
        Transformed keypoints, shape (K, 2)
    """
    keypoints_homo = np.concatenate(
        [keypoints, np.ones((keypoints.shape[0], 1))], axis=1
    )
    keypoints_trans = keypoints_homo @ trans_matrix.T
    return keypoints_trans


def decode_heatmaps(heatmaps, input_size=(256, 256)):
    """Decode heatmaps to keypoint coordinates using argmax with sub-pixel refinement.

    Args:
        heatmaps: Heatmap array, shape (K, H, W)
        input_size: Input image size (width, height)

    Returns:
        keypoints: Keypoint coordinates, shape (K, 2)
        scores: Keypoint scores, shape (K,)
    """
    K, H, W = heatmaps.shape

    # Flatten heatmaps
    heatmaps_reshaped = heatmaps.reshape(K, -1)

    # Get max values and indices
    scores = np.max(heatmaps_reshaped, axis=1)
    indices = np.argmax(heatmaps_reshaped, axis=1)

    # Convert indices to coordinates (in heatmap space)
    y = indices // W
    x = indices % W

    # Refine coordinates with sub-pixel accuracy using Taylor expansion
    keypoints = np.zeros((K, 2), dtype=np.float32)

    for i in range(K):
        px = int(x[i])
        py = int(y[i])

        if 1 < px < W - 1 and 1 < py < H - 1:
            # Calculate derivatives for Taylor expansion
            diff_x = (heatmaps[i, py, px + 1] - heatmaps[i, py, px - 1]) / 2.0
            diff_y = (heatmaps[i, py + 1, px] - heatmaps[i, py - 1, px]) / 2.0
            diff_xx = (
                heatmaps[i, py, px + 1]
                - 2 * heatmaps[i, py, px]
                + heatmaps[i, py, px - 1]
            )
            diff_yy = (
                heatmaps[i, py + 1, px]
                - 2 * heatmaps[i, py, px]
                + heatmaps[i, py - 1, px]
            )
            diff_xy = (
                heatmaps[i, py + 1, px + 1]
                - heatmaps[i, py + 1, px - 1]
                - heatmaps[i, py - 1, px + 1]
                + heatmaps[i, py - 1, px - 1]
            ) / 4.0

            # Calculate offset
            derivative = np.array([diff_x, diff_y])
            hessian = np.array([[diff_xx, diff_xy], [diff_xy, diff_yy]])

            # Avoid division by zero
            if diff_xx * diff_yy - diff_xy**2 != 0:
                offset = -np.linalg.inv(hessian) @ derivative
                # Limit offset to [-0.5, 0.5]
                offset = np.clip(offset, -0.5, 0.5)
                keypoints[i, 0] = px + offset[0]
                keypoints[i, 1] = py + offset[1]
            else:
                keypoints[i, 0] = px
                keypoints[i, 1] = py
        else:
            keypoints[i, 0] = px
            keypoints[i, 1] = py

    # Scale to input size (heatmap is typically 64x64, input is 256x256)
    scale_x = input_size[0] / W
    scale_y = input_size[1] / H
    keypoints[:, 0] *= scale_x
    keypoints[:, 1] *= scale_y

    return keypoints, scores


def inference_single_image(
    model, image, box, device='cuda:0', input_size=(256, 256), scale_factor=1.25
):
    """Run inference on a single image with a single bounding box.

    Args:
        model: HRNetV2 model
        image: Input image in BGR format, shape (H, W, 3)
        box: Bounding box [x0, y0, x1, y1, score]
        device: Device to run inference on
        input_size: Model input size (width, height)
        scale_factor: Scale factor for bounding box expansion (default: 1.25)

    Returns:
        keypoints: Keypoint coordinates in original image space, shape (K, 2)
        scores: Keypoint scores, shape (K,)
    """
    model.eval()

    # Get center and scale
    image_h, image_w = image.shape[:2]
    center, scale = box_to_center_scale(box, input_size, scale_factor)

    # Get affine transformation
    trans = get_affine_transform(center, scale, 0, input_size)

    # Apply affine transformation
    input_img = cv2.warpAffine(image, trans, input_size, flags=cv2.INTER_LINEAR)

    # Convert to tensor (BGR format)
    input_tensor = torch.from_numpy(input_img).float().permute(2, 0, 1).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Preprocess
    input_tensor = model.preprocess(input_tensor)

    # Inference
    with torch.no_grad():
        heatmaps = model(input_tensor)

    # Convert to numpy
    heatmaps = heatmaps.cpu().numpy()[0]  # (K, H, W)

    # Decode heatmaps
    keypoints, scores = decode_heatmaps(heatmaps, input_size)

    # Transform back to original image space
    trans_inv = get_affine_transform(center, scale, 0, input_size, inv=True)
    keypoints = affine_transform_keypoints(keypoints, trans_inv)

    return keypoints, scores


def inference_batch(
    model, image, boxes, device='cuda:0', input_size=(256, 256), scale_factor=1.25
):
    """Run inference on a single image with multiple bounding boxes.

    Args:
        model: HRNetV2 model
        image: Input image in BGR format, shape (H, W, 3)
        boxes: List of bounding boxes, each [x0, y0, x1, y1, score]
        device: Device to run inference on
        input_size: Model input size (width, height)
        scale_factor: Scale factor for bounding box expansion (default: 1.25)

    Returns:
        results: List of detection results, each containing:
            - bbox: Bounding box [x0, y0, x1, y1, score]
            - keypoints: Keypoint coordinates with scores, shape (K, 3) [x, y, score]
    """
    if len(boxes) == 0:
        return []

    results = []
    for box in boxes:
        keypoints, scores = inference_single_image(
            model, image, box, device, input_size, scale_factor
        )

        # Combine keypoints and scores
        keypoints_with_scores = np.concatenate(
            [keypoints, scores[:, np.newaxis]], axis=1
        )  # (K, 3)

        results.append({'bbox': box, 'keypoints': keypoints_with_scores})

    return results
