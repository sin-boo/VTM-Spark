"""Instant center calibration + face-attached coordinate system.

Uses the same 28 labeling landmarks as pose-traker auto-labeling.
Center zeros head angles from face geometry (not raw OpenSeeFace euler,
which can look 90° sideways). Scope sticks to the nose with Y = up face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from label_schema import face_basis_from_label28, geom_head_angles


def wrap_deg(delta: float) -> float:
    return ((float(delta) + 180.0) % 360.0) - 180.0


@dataclass
class PoseSample:
    pitch: float
    yaw: float
    roll: float
    anchor_xy: tuple[float, float]
    frame_wh: tuple[int, int]
    basis_x: np.ndarray
    basis_y: np.ndarray
    conf: float = 0.0


@dataclass
class RelativePose:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    nx: float = 0.0
    ny: float = 0.0
    anchor_xy: tuple[float, float] = (0.0, 0.0)
    calibrated: bool = False
    axis_x: np.ndarray | None = None
    axis_y: np.ndarray | None = None


@dataclass
class CenterCalibration:
    origin: PoseSample | None = None
    flash_until: float = 0.0

    @property
    def ready(self) -> bool:
        return self.origin is not None

    def clear(self) -> None:
        self.origin = None

    def capture(self, pts28: np.ndarray, frame_shape: tuple[int, ...], conf: float = 0.0) -> PoseSample:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        basis = face_basis_from_label28(pts28)
        if basis is None:
            raise RuntimeError('Need visible eyes/nose/brows to calibrate')
        origin_xy, bx, by = basis
        pitch, yaw, roll = geom_head_angles(pts28)
        sample = PoseSample(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            anchor_xy=(float(origin_xy[0]), float(origin_xy[1])),
            frame_wh=(w, h),
            basis_x=bx,
            basis_y=by,
            conf=float(conf),
        )
        self.origin = sample
        return sample

    def remap(self, pts28: np.ndarray, frame_shape: tuple[int, ...]) -> RelativePose:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        basis = face_basis_from_label28(pts28)
        if basis is None:
            return RelativePose(calibrated=self.ready)
        origin_xy, live_x, live_y = basis
        anchor = (float(origin_xy[0]), float(origin_xy[1]))
        pitch, yaw, roll = geom_head_angles(pts28)

        if self.origin is None:
            dx = anchor[0] - w * 0.5
            dy = anchor[1] - h * 0.5
            return RelativePose(
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                dx=dx,
                dy=dy,
                nx=float(np.clip(dx / max(w * 0.5, 1e-6), -1.0, 1.0)),
                ny=float(np.clip(dy / max(h * 0.5, 1e-6), -1.0, 1.0)),
                anchor_xy=anchor,
                calibrated=False,
                axis_x=live_x,
                axis_y=live_y,
            )

        o = self.origin
        return RelativePose(
            pitch=wrap_deg(pitch - o.pitch),
            yaw=wrap_deg(yaw - o.yaw),
            roll=wrap_deg(roll - o.roll),
            dx=anchor[0] - o.anchor_xy[0],
            dy=anchor[1] - o.anchor_xy[1],
            nx=float(np.clip((anchor[0] - o.anchor_xy[0]) / max(w * 0.5, 1e-6), -1.0, 1.0)),
            ny=float(np.clip((anchor[1] - o.anchor_xy[1]) / max(h * 0.5, 1e-6), -1.0, 1.0)),
            anchor_xy=anchor,
            calibrated=True,
            axis_x=live_x,
            axis_y=live_y,
        )

    def to_dict(self, rel: RelativePose) -> dict[str, Any]:
        return {
            'calibrated': rel.calibrated,
            'pitch': rel.pitch,
            'yaw': rel.yaw,
            'roll': rel.roll,
            'dx': rel.dx,
            'dy': rel.dy,
            'nx': rel.nx,
            'ny': rel.ny,
            'anchor_xy': list(rel.anchor_xy),
            'origin_xy': list(self.origin.anchor_xy) if self.origin else None,
        }


def draw_coordinate_scope(
    frame: np.ndarray,
    calib: CenterCalibration,
    rel: RelativePose | None,
    *,
    now: float = 0.0,
) -> None:
    if rel is None or rel.axis_x is None or rel.axis_y is None:
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (120, 120, 120), cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)
        return

    ox, oy = rel.anchor_xy
    origin = (int(round(ox)), int(round(oy)))
    x_axis = rel.axis_x
    y_axis = rel.axis_y
    length = 55.0
    x_end = (int(round(ox + x_axis[0] * length)), int(round(oy + x_axis[1] * length)))
    y_end = (int(round(ox + y_axis[0] * length)), int(round(oy + y_axis[1] * length)))

    ring = (0, 255, 220) if rel.calibrated else (160, 160, 160)
    cv2.circle(frame, origin, 22, ring, 1, cv2.LINE_AA)
    cv2.circle(frame, origin, 3, ring, -1, cv2.LINE_AA)
    cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(frame, 'X', (x_end[0] + 4, x_end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Y up', (y_end[0] + 4, y_end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    if now and now < calib.flash_until:
        cv2.circle(frame, origin, 34, (0, 255, 255), 2, cv2.LINE_AA)

    if calib.origin is not None:
        cx, cy = calib.origin.anchor_xy
        cv2.drawMarker(
            frame,
            (int(round(cx)), int(round(cy))),
            (0, 180, 255),
            cv2.MARKER_TILTED_CROSS,
            14,
            1,
            cv2.LINE_AA,
        )

    h = frame.shape[0]
    y0 = h - 58
    if rel.calibrated:
        line = f'rel  P:{rel.pitch:+6.1f}  Y:{rel.yaw:+6.1f}  R:{rel.roll:+6.1f}   nX:{rel.nx:+.2f} nY:{rel.ny:+.2f}   [28 label pts]'
        color = (180, 255, 180)
    else:
        line = 'look straight → Center  ·  28 labeling points  ·  Y sticks up your face'
        color = (160, 160, 160)
    cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f'face-local  dx:{rel.dx:+.0f}  dy:{rel.dy:+.0f}',
        (10, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
