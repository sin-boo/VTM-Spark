"""Canonical keypoint schema for pose-conditioned training.

37 points = 28 face landmarks + 2 iris + 7 upper-body joints.
Order is fixed and must match between dataset build, training, and inference.

Each point is stored as (x, y, score, visible) with x,y in normalized crop space
[-1, 1] by default (after build_dataset), or absolute pixels before transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Face landmark labels in detector order (ids 0..27).
FACE_LANDMARK_LABELS: tuple[str, ...] = (
    "face_outline",  # 0
    "face_outline",  # 1
    "face_outline",  # 2
    "face_outline",  # 3
    "face_outline",  # 4
    "eyebrow",       # 5
    "eyebrow",       # 6
    "eyebrow",       # 7
    "eyebrow",       # 8
    "eyebrow",       # 9
    "eyebrow",       # 10
    "eye",          # 11
    "eye",          # 12
    "eye",          # 13
    "nose",         # 14
    "nose",         # 15
    "nose",         # 16
    "eye",          # 17
    "eye",          # 18
    "eye",          # 19
    "mouth",        # 20
    "mouth",        # 21
    "mouth",        # 22
    "mouth",        # 23
    "mouth",        # 24
    "mouth",        # 25
    "mouth",        # 26
    "mouth",        # 27
)

NUM_FACE_LANDMARKS = len(FACE_LANDMARK_LABELS)  # 28

# Iris points appended after face landmarks.
IRIS_NAMES: tuple[str, ...] = (
    "right_iris",  # 28
    "left_iris",   # 29
)
NUM_IRIS = len(IRIS_NAMES)

# Upper-body joints (from upper_body_pose.people[0].joints).
BODY_JOINT_NAMES: tuple[str, ...] = (
    "nose",            # 30
    "neck",            # 31
    "right_shoulder",  # 32
    "right_elbow",     # 33
    "left_shoulder",   # 34
    "left_elbow",      # 35
    "chest",           # 36
)
NUM_BODY_JOINTS = len(BODY_JOINT_NAMES)

# Canonical point names (length 37).
KEYPOINT_NAMES: tuple[str, ...] = tuple(
    f"face_{i}_{FACE_LANDMARK_LABELS[i]}" for i in range(NUM_FACE_LANDMARKS)
) + IRIS_NAMES + BODY_JOINT_NAMES

NUM_KEYPOINTS = len(KEYPOINT_NAMES)  # 37
assert NUM_KEYPOINTS == 37

# Features per keypoint: x, y, score, visible
KEYPOINT_DIM = 4  # (x, y, score, visible)

# Pose-map channel groups (order = channel index).
POSE_CHANNEL_GROUPS: tuple[str, ...] = (
    "face_outline",
    "eyebrow",
    "eye",
    "iris",
    "nose",
    "mouth",
    "skeleton",
    "joints",
)
NUM_POSE_CHANNELS = len(POSE_CHANNEL_GROUPS)  # 8

# Map face landmark index -> pose channel group name.
FACE_ID_TO_GROUP: dict[int, str] = {
    i: FACE_LANDMARK_LABELS[i] for i in range(NUM_FACE_LANDMARKS)
}

# Bone connectivity using body joint names (indices into BODY_JOINT_NAMES offset).
BODY_BONES: tuple[tuple[str, str], ...] = (
    ("nose", "neck"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "left_shoulder"),
    ("neck", "chest"),
)

# Indices into the full 37-vector for body joints, keyed by name.
BODY_NAME_TO_INDEX: dict[str, int] = {
    name: NUM_FACE_LANDMARKS + NUM_IRIS + i for i, name in enumerate(BODY_JOINT_NAMES)
}

# Bone index pairs in the full 37-vector.
BODY_BONE_INDICES: tuple[tuple[int, int], ...] = tuple(
    (BODY_NAME_TO_INDEX[a], BODY_NAME_TO_INDEX[b]) for a, b in BODY_BONES
)

# Horizontal flip: pairs of indices that should be swapped (L/R).
# Face landmark L/R pairs (eyebrows 5-7 vs 8-10, eyes 11-13 vs 17-19).
# Mouth / outline / nose are mirrored in x only (no index swap for unpaired).
FLIP_SWAP_PAIRS: tuple[tuple[int, int], ...] = (
    # eyebrows: left trio 5,6,7 <-> right trio 10,9,8 (mirror order)
    (5, 10),
    (6, 9),
    (7, 8),
    # eyes: left 11,12,13 <-> right 19,18,17
    (11, 19),
    (12, 18),
    (13, 17),
    # face outline: 0,1 <-> 4,3 (2 is chin center)
    (0, 4),
    (1, 3),
    # mouth corners / sides — approximate symmetric pairs
    (20, 22),
    (23, 26),
    (24, 27),
    # iris
    (28, 29),  # right_iris <-> left_iris
    # body
    (32, 34),  # right_shoulder <-> left_shoulder
    (33, 35),  # right_elbow <-> left_elbow
)


def empty_keypoints(dtype=np.float32) -> np.ndarray:
    """Return zeros of shape (NUM_KEYPOINTS, KEYPOINT_DIM)."""
    return np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=dtype)


def _set_point(out: np.ndarray, idx: int, x: float, y: float, score: float, visible: bool) -> None:
    out[idx, 0] = float(x)
    out[idx, 1] = float(y)
    out[idx, 2] = float(score)
    out[idx, 3] = 1.0 if visible else 0.0


def keypoints_from_landmarks_json(data: Mapping[str, Any] | None) -> np.ndarray:
    """Parse a *_landmarks.json (or faces section) into (37, 4). Body/iris left zero."""
    out = empty_keypoints()
    if not data:
        return out
    faces = data.get("faces") or []
    if not faces:
        return out
    face = faces[0]
    kps = face.get("keypoints") or []
    for kp in kps:
        try:
            idx = int(kp["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= NUM_FACE_LANDMARKS:
            continue
        score = float(kp.get("score", 0.0))
        visible = bool(kp.get("visible", score > 0.1))
        _set_point(out, idx, float(kp.get("x", 0.0)), float(kp.get("y", 0.0)), score, visible)
    # Optional embedded upper_body_pose (landmarks.json sometimes has it).
    ub = data.get("upper_body_pose")
    if ub:
        _fill_body(out, ub)
    return out


def keypoints_from_iris_json(data: Mapping[str, Any] | None, out: np.ndarray | None = None) -> np.ndarray:
    """Fill iris slots (28, 29) from *_iris.json."""
    if out is None:
        out = empty_keypoints()
    if not data:
        return out
    faces = data.get("faces") or []
    if not faces:
        return out
    irises = (faces[0].get("irises") or {})
    for name, slot in (("right_iris", 28), ("left_iris", 29)):
        entry = irises.get(name) or {}
        score = float(entry.get("score", 0.0))
        visible = bool(entry.get("visible", False))
        _set_point(
            out,
            slot,
            float(entry.get("x", 0.0)),
            float(entry.get("y", 0.0)),
            score,
            visible and score > 0.0,
        )
    return out


def _fill_body(out: np.ndarray, upper_body_pose: Mapping[str, Any]) -> None:
    people = upper_body_pose.get("people") or []
    if not people:
        return
    joints = people[0].get("joints") or {}
    for name, idx in BODY_NAME_TO_INDEX.items():
        entry = joints.get(name) or {}
        score = float(entry.get("score", 0.0))
        visible = bool(entry.get("visible", score > 0.1))
        _set_point(
            out,
            idx,
            float(entry.get("x", 0.0)),
            float(entry.get("y", 0.0)),
            score,
            visible,
        )


def keypoints_from_upper_body_json(data: Mapping[str, Any] | None, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        out = empty_keypoints()
    if not data:
        return out
    # Accept either wrapped {"upper_body_pose": ...} or the pose object itself.
    ub = data.get("upper_body_pose", data)
    _fill_body(out, ub)
    return out


def keypoints_from_full_stack(data: Mapping[str, Any]) -> np.ndarray:
    """Parse a *_full_stack.json combining landmarks + iris + body."""
    out = keypoints_from_landmarks_json(data)
    iris = data.get("iris")
    if iris:
        keypoints_from_iris_json(iris, out)
    ub = data.get("upper_body_pose")
    if ub:
        keypoints_from_upper_body_json({"upper_body_pose": ub}, out)
    return out


def load_keypoints_from_label_dir(label_dir: Path | str, pose_id: str) -> np.ndarray:
    """Load keypoints for a pose from a labels directory.

    Prefers ``{pose_id}_full_stack.json``; otherwise merges landmarks + iris + upper_body.
    """
    label_dir = Path(label_dir)
    full = label_dir / f"{pose_id}_full_stack.json"
    if full.is_file():
        with open(full, "r", encoding="utf-8") as f:
            return keypoints_from_full_stack(json.load(f))

    out = empty_keypoints()
    lm = label_dir / f"{pose_id}_landmarks.json"
    if lm.is_file():
        with open(lm, "r", encoding="utf-8") as f:
            out = keypoints_from_landmarks_json(json.load(f))
    iris = label_dir / f"{pose_id}_iris.json"
    if iris.is_file():
        with open(iris, "r", encoding="utf-8") as f:
            keypoints_from_iris_json(json.load(f), out)
    ub = label_dir / f"{pose_id}_upper_body_pose.json"
    if ub.is_file():
        with open(ub, "r", encoding="utf-8") as f:
            keypoints_from_upper_body_json(json.load(f), out)
    return out


def normalize_keypoints_xy(
    kps: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Map absolute pixel x,y into full-frame [-1, 1] (``norm_full``).

    Score/visible unchanged. This is **not** model space — for DiT / training
    use ``transform_keypoints_crop`` or ``coordinate_frames.to_norm_crop``.
    """
    out = np.asarray(kps, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != KEYPOINT_DIM:
        raise ValueError(f"Expected (N, {KEYPOINT_DIM}), got {out.shape}")
    w = max(float(image_width), 1.0)
    h = max(float(image_height), 1.0)
    out[:, 0] = 2.0 * (out[:, 0] / w) - 1.0
    out[:, 1] = 2.0 * (out[:, 1] / h) - 1.0
    return out


def transform_keypoints_crop(
    kps: np.ndarray,
    *,
    crop_x0: float,
    crop_y0: float,
    crop_w: float,
    crop_h: float,
    out_w: float,
    out_h: float,
    normalize: bool = True,
) -> np.ndarray:
    """Map absolute keypoints through a crop+resize into output image space.

    If ``normalize``, result x,y are in [-1, 1] w.r.t. (out_w, out_h).
    Points that fall outside the crop keep their transformed coords but
    ``visible`` is set to 0.
    """
    out = np.asarray(kps, dtype=np.float32).copy()
    sx = float(out_w) / max(float(crop_w), 1e-6)
    sy = float(out_h) / max(float(crop_h), 1e-6)
    x = (out[:, 0] - float(crop_x0)) * sx
    y = (out[:, 1] - float(crop_y0)) * sy
    # Mark out-of-crop as invisible (with small margin).
    margin = 2.0
    outside = (x < -margin) | (y < -margin) | (x > out_w + margin) | (y > out_h + margin)
    out[:, 3] = np.where(outside, 0.0, out[:, 3])
    if normalize:
        out[:, 0] = 2.0 * (x / max(float(out_w), 1.0)) - 1.0
        out[:, 1] = 2.0 * (y / max(float(out_h), 1.0)) - 1.0
    else:
        out[:, 0] = x
        out[:, 1] = y
    return out


def flip_keypoints_horizontal(kps: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Mirror x -> -x (normalized) or reflect about image center, and swap L/R pairs.

    Assumes x is already in [-1, 1] (normalized crop space). For pixel space,
    call with x mirrored as ``width - 1 - x`` separately.
    """
    out = np.asarray(kps, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[0] != NUM_KEYPOINTS or out.shape[1] != KEYPOINT_DIM:
        raise ValueError(f"Expected ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {out.shape}")
    out[:, 0] = -out[:, 0]
    for a, b in FLIP_SWAP_PAIRS:
        tmp = out[a].copy()
        out[a] = out[b]
        out[b] = tmp
    return out


def keypoint_deltas(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-point delta (target - ref) for x,y; keep target score/visible.

    Returns (NUM_KEYPOINTS, 4) with columns (dx, dy, score_target, visible_target).
    Invisible points in either get dx=dy=0 and visible=0.
    """
    t = np.asarray(target, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    out = np.zeros_like(t)
    both = (t[:, 3] > 0.5) & (r[:, 3] > 0.5)
    out[both, 0] = t[both, 0] - r[both, 0]
    out[both, 1] = t[both, 1] - r[both, 1]
    out[:, 2] = t[:, 2]
    out[:, 3] = both.astype(np.float32)
    return out


def group_indices(group: str) -> list[int]:
    """Indices belonging to a pose-map channel group."""
    if group == "iris":
        return [28, 29]
    if group == "skeleton" or group == "joints":
        return list(range(NUM_FACE_LANDMARKS + NUM_IRIS, NUM_KEYPOINTS))
    return [i for i, lab in enumerate(FACE_LANDMARK_LABELS) if lab == group]
