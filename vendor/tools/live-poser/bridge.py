"""OSF → KEYPOINT_SCHEMA bridge.

Live Poser tracks with OpenSeeFace, then projects every frame into the same
(37, 4) slot order used for training (see send2pod/KEYPOINT_SCHEMA).

Slots
-----
0..27  face (Label28)
28     right_iris
29     left_iris
30..36 body (synthetic upper-body from face pose; OSF has no skeleton)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NUM_KEYPOINTS = 37
FACE_COUNT = 28
RIGHT_IRIS = 28
LEFT_IRIS = 29
BODY_START = 30
BODY_COUNT = 7

KEYPOINT_NAMES = [
    *[f'face_{i}' for i in range(28)],
    'right_iris',
    'left_iris',
    'nose',
    'neck',
    'right_shoulder',
    'right_elbow',
    'left_shoulder',
    'left_elbow',
    'chest',
]


@dataclass
class IrisPoint:
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    visible: bool = False
    method: str = 'none'
    eye_bbox: list[int] | None = None

    def as_row(self) -> np.ndarray:
        vis = 1.0 if self.visible and self.score > 0 else 0.0
        return np.array([self.x, self.y, self.score, vis], dtype=np.float32)


@dataclass
class BridgeFrame:
    """One frame of training-schema parameters.

    ``keypoints`` stay in source pixel space for diagnostics/overlays.
    ``keypoints_norm`` (when set) are model ``norm_crop`` coordinates.
    """

    t: float
    image_wh: tuple[int, int]
    keypoints: np.ndarray  # (37, 4) x,y,score,visible — pixel_src
    mirrored: bool = False
    iris_method: str = 'none'
    skeleton_method: str = 'none'
    pose: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    keypoints_norm: np.ndarray | None = None  # (37, 4) norm_crop
    coord_space: str = 'pixel_src'
    crop: Any = None  # CropRect | dict | None
    image_size: int = 768

    def to_dict(self, *, round_xy: int = 2) -> dict[str, Any]:
        k = self.keypoints
        rows = []
        for i in range(NUM_KEYPOINTS):
            rows.append(
                {
                    'i': i,
                    'name': KEYPOINT_NAMES[i],
                    'x': round(float(k[i, 0]), round_xy),
                    'y': round(float(k[i, 1]), round_xy),
                    'score': round(float(k[i, 2]), 3),
                    'visible': bool(k[i, 3] >= 0.5),
                }
            )
        out: dict[str, Any] = {
            't': self.t,
            'schema': 'KEYPOINT_SCHEMA',
            'shape': [NUM_KEYPOINTS, 4],
            'image_wh': [int(self.image_wh[0]), int(self.image_wh[1])],
            'source_wh': [int(self.image_wh[0]), int(self.image_wh[1])],
            'mirrored': self.mirrored,
            'iris_method': self.iris_method,
            'skeleton_method': self.skeleton_method,
            'coord_space': str(self.coord_space or 'pixel_src'),
            'image_size': int(self.image_size),
            'pose': self.pose,
            'meta': self.meta,
            'keypoints': rows,
        }
        crop = self.crop
        if crop is not None:
            if hasattr(crop, 'as_dict'):
                out['crop'] = crop.as_dict()
            elif isinstance(crop, dict):
                out['crop'] = crop
        if self.keypoints_norm is not None:
            kn = np.asarray(self.keypoints_norm, dtype=np.float32)
            out['keypoints_norm'] = [
                {
                    'i': i,
                    'name': KEYPOINT_NAMES[i],
                    'x': round(float(kn[i, 0]), 5),
                    'y': round(float(kn[i, 1]), 5),
                    'score': round(float(kn[i, 2]), 3),
                    'visible': bool(kn[i, 3] >= 0.5),
                }
                for i in range(NUM_KEYPOINTS)
            ]
            out['coord_space_norm'] = 'norm_crop'
        # Prefer body_lost from meta for consumers.
        if 'body_lost' not in out['meta']:
            out['meta'] = dict(out['meta'])
            out['meta']['body_lost'] = False
        return out

    def summary_text(self) -> str:
        """Compact side-panel text."""
        k = self.keypoints
        w, h = self.image_wh
        lines = [
            f'KEYPOINT_SCHEMA  (37,4)  {w}x{h}',
            f'iris={self.iris_method}  skel={self.skeleton_method}',
            f'mirror={self.mirrored}',
            '',
        ]
        if self.pose:
            lines.append(
                f"pose P:{self.pose.get('pitch', 0):+.1f} "
                f"Y:{self.pose.get('yaw', 0):+.1f} "
                f"R:{self.pose.get('roll', 0):+.1f}"
            )
            lines.append('')

        lines.append('face 0..27 (visible):')
        vis_face = int(np.sum(k[:FACE_COUNT, 3] >= 0.5))
        lines.append(f'  {vis_face}/28 pts')
        nose = k[15]
        if nose[3] >= 0.5:
            lines.append(f'  nose15  x={nose[0]:.1f} y={nose[1]:.1f} s={nose[2]:.2f}')

        lines.append('')
        lines.append('iris:')
        for idx, label in ((RIGHT_IRIS, '28 right'), (LEFT_IRIS, '29 left')):
            r = k[idx]
            if r[3] >= 0.5:
                lines.append(f'  {label}  x={r[0]:.1f} y={r[1]:.1f} s={r[2]:.2f}')
            else:
                lines.append(f'  {label}  (hidden)')

        lines.append('')
        lines.append('body 30..36:')
        vis_body = int(np.sum(k[BODY_START:, 3] >= 0.5))
        if vis_body == 0:
            lines.append('  (off)')
        else:
            lines.append(f'  {vis_body}/7 joints')
            for idx, name in (
                (30, 'nose'),
                (31, 'neck'),
                (32, 'r_sh'),
                (34, 'l_sh'),
                (36, 'chest'),
            ):
                r = k[idx]
                if r[3] >= 0.5:
                    lines.append(f'  {idx} {name}  x={r[0]:.0f} y={r[1]:.0f}')
        return '\n'.join(lines)


def pts28_to_keypoints37(
    pts28: np.ndarray | None,
    *,
    right_iris: IrisPoint | None = None,
    left_iris: IrisPoint | None = None,
    body7: np.ndarray | None = None,
    face_score_thr: float = 0.15,
) -> np.ndarray:
    """Build (37,4) tensor in image pixel space."""
    out = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)
    if pts28 is not None and len(pts28) >= FACE_COUNT:
        for i in range(FACE_COUNT):
            x, y, s = float(pts28[i, 0]), float(pts28[i, 1]), float(pts28[i, 2])
            out[i, 0] = x
            out[i, 1] = y
            out[i, 2] = s
            out[i, 3] = 1.0 if s >= face_score_thr else 0.0
    ri = right_iris or IrisPoint()
    li = left_iris or IrisPoint()
    out[RIGHT_IRIS] = ri.as_row()
    out[LEFT_IRIS] = li.as_row()
    if body7 is not None and len(body7) >= BODY_COUNT:
        out[BODY_START : BODY_START + BODY_COUNT] = body7[:BODY_COUNT]
    return out


def flip_keypoints37_x(kpts: np.ndarray, width: int) -> np.ndarray:
    """Mirror KEYPOINT_SCHEMA tensor; swaps person L/R pairs including iris."""
    if kpts is None or len(kpts) < NUM_KEYPOINTS:
        return kpts
    out = kpts.copy()
    out[:, 0] = (width - 1) - out[:, 0]
    pairs = [
        (0, 4),
        (1, 3),
        (5, 10),
        (6, 9),
        (7, 8),
        (11, 19),
        (12, 18),
        (13, 17),
        (14, 16),
        (20, 22),
        (23, 26),
        (24, 27),
        (RIGHT_IRIS, LEFT_IRIS),
        (32, 34),  # shoulders
        (33, 35),  # elbows
    ]
    for a, b in pairs:
        out[[a, b]] = out[[b, a]]
    return out


def flip_iris_pair(
    right: IrisPoint,
    left: IrisPoint,
    width: int,
) -> tuple[IrisPoint, IrisPoint]:
    """Mirror iris points and swap sides (after display flip)."""

    def _flip(p: IrisPoint) -> IrisPoint:
        if not p.visible and p.score <= 0:
            return IrisPoint(method=p.method)
        bbox = None
        if p.eye_bbox is not None and len(p.eye_bbox) == 4:
            x0, y0, x1, y1 = p.eye_bbox
            bbox = [width - 1 - x1, y0, width - 1 - x0, y1]
        return IrisPoint(
            x=(width - 1) - float(p.x),
            y=float(p.y),
            score=float(p.score),
            visible=bool(p.visible),
            method=p.method,
            eye_bbox=bbox,
        )

    # After flip, image-left becomes person-left → schema left_iris, etc.
    new_left = _flip(right)   # was person-right on image-left
    new_right = _flip(left)   # was person-left on image-right
    return new_right, new_left


def build_bridge_frame(
    pts28: np.ndarray | None,
    image_shape: tuple[int, ...],
    *,
    right_iris: IrisPoint | None = None,
    left_iris: IrisPoint | None = None,
    body7: np.ndarray | None = None,
    mirrored: bool = False,
    iris_method: str = 'none',
    skeleton_method: str = 'none',
    pose: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    t: float | None = None,
) -> BridgeFrame:
    h, w = int(image_shape[0]), int(image_shape[1])
    kpts = pts28_to_keypoints37(
        pts28,
        right_iris=right_iris,
        left_iris=left_iris,
        body7=body7,
    )
    return BridgeFrame(
        t=time.time() if t is None else float(t),
        image_wh=(w, h),
        keypoints=kpts,
        mirrored=mirrored,
        iris_method=iris_method,
        skeleton_method=skeleton_method,
        pose=pose or {},
        meta=meta or {},
    )


def write_bridge_json(frame: BridgeFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frame.to_dict(), indent=2), encoding='utf-8')
