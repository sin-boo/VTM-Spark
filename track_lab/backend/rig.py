"""Drive the full 28-point mesh from OSF head pose.

The tracked face writes the character's place: location from the eye
midpoint, size from the face box. Looking left or right stays a rotation
around the rest nose. The skeleton only follows this place.
"""

from __future__ import annotations

import math

import numpy as np

from .feel import feel
from .visemes import rest_stamp, session_rest_locked

_MAX_TURN = 80.0
_MAX_LOOK_DOWN = 32.0
_MAX_LOOK_UP = 50.0
# Front hemisphere. Hair outside this disk is pinned here; extra length is fluff.
_RN_MAX = 0.92
_FOCAL = 1.8
_PERSP_MIN = 0.72
_PERSP_MAX = 1.28


def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else float(value)


def _posed_sphere(
    xs: np.ndarray,
    ys: np.ndarray,
    yaw_r: float,
    pitch_r: float,
    roll_r: float,
    radius: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Front-hemisphere rotate. Hair past the disk is fluff, not extra yaw X."""
    xs = np.asarray(xs, dtype=np.float64) * scale
    ys = np.asarray(ys, dtype=np.float64) * scale
    radius = max(float(radius), 1.0)
    xy_r = np.hypot(xs, ys)
    cap = radius * math.sqrt(_RN_MAX)
    pull = np.minimum(cap / np.maximum(xy_r, 1e-8), 1.0)
    xs_s = xs * pull
    ys_s = ys * pull
    fluff = xy_r - np.hypot(xs_s, ys_s)
    rn = (xs_s * xs_s + ys_s * ys_s) / (radius * radius)
    zs = radius * np.sqrt(np.maximum(1.0 - rn, 0.0))
    # OSF pitch+ is look-down (nose points down). This bowl is Y-down with
    # Z toward the camera, so that same sign grew the chin — look-up. Flip
    # so a nod tucks the chin and looking up brings it toward the camera.
    pitch_r = -float(pitch_r)
    x2 = xs_s * math.cos(yaw_r) + zs * math.sin(yaw_r)
    z2 = -xs_s * math.sin(yaw_r) + zs * math.cos(yaw_r)
    y2 = ys_s * math.cos(pitch_r) - z2 * math.sin(pitch_r)
    z3 = ys_s * math.sin(pitch_r) + z2 * math.cos(pitch_r)
    xr = x2 * math.cos(roll_r) - y2 * math.sin(roll_r)
    yr = x2 * math.sin(roll_r) + y2 * math.cos(roll_r)
    return xr, yr, z3, zs, fluff


def _with_fluff(
    xr: np.ndarray,
    yr: np.ndarray,
    zr: np.ndarray,
    zs: np.ndarray,
    fluff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    posed_r = np.maximum(np.hypot(xr, yr), 1e-8)
    facing = np.clip(zr / np.maximum(zs, 1e-3), 0.15, 1.0)
    return xr + fluff * (xr / posed_r) * facing, yr + fluff * (yr / posed_r) * facing


def orbit_xyz(
    xs: np.ndarray,
    ys: np.ndarray,
    yaw_r: float,
    pitch_r: float,
    roll_r: float,
    radius: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a front hemisphere. Hair beyond the disk keeps fluff, not extra X."""
    xr, yr, zr, zs, fluff = _posed_sphere(
        xs, ys, yaw_r, pitch_r, roll_r, radius, scale
    )
    xr, yr = _with_fluff(xr, yr, zr, zs, fluff)
    return xr, yr, zr, zs


def orbit_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    yaw_r: float,
    pitch_r: float,
    roll_r: float,
    radius: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """2.5D spherical bowl used by the mesh rig and the IFM pip."""
    xr, yr, _zr, _zs = orbit_xyz(xs, ys, yaw_r, pitch_r, roll_r, radius, scale)
    return xr, yr


def project_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    yaw_r: float,
    pitch_r: float,
    roll_r: float,
    radius: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nose-locked projection: near side grows, far side shrinks."""
    xr, yr, zr, zs, fluff = _posed_sphere(
        xs, ys, yaw_r, pitch_r, roll_r, radius, scale
    )
    ox, oy, oz, ozs, _fluff0 = _posed_sphere(
        np.zeros(1), np.zeros(1), yaw_r, pitch_r, roll_r, radius, scale
    )
    # Depth from the turn only. Rest pose (euler 0) keeps scale 1.
    z_rel = (zr - zs) - (float(oz[0]) - float(ozs[0]))
    focal = _FOCAL * max(float(radius), 1.0)
    persp = np.clip(focal / np.maximum(focal - z_rel, 1e-3), _PERSP_MIN, _PERSP_MAX)
    px = xr - float(ox[0])
    py = yr - float(oy[0])
    # Perspective is for the face disk. Hair spikes only tuck with facing;
    # scaling their rest length from the nose is what ballooned one side.
    inner = fluff <= 1e-6
    px = np.where(inner, px * persp, px)
    py = np.where(inner, py * persp, py)
    return _with_fluff(px, py, zr, zs, fluff)


def project_head(
    xs: np.ndarray,
    ys: np.ndarray,
    head: dict[str, float] | None,
    radius: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute head euler, same signs as FaceRig (pitch+ = look-down)."""
    if not head:
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    yaw_deg = float(head.get("yaw", 0.0))
    roll_deg = float(head.get("roll", 0.0))
    yaw = math.radians(_clip(yaw_deg, -_MAX_TURN * feel.max_yaw(), _MAX_TURN * feel.max_yaw()))
    pitch_deg = float(head.get("pitch", 0.0))
    pitch = math.radians(
        _clip(
            pitch_deg,
            -_MAX_LOOK_UP * feel.max_pitch_up(),
            _MAX_LOOK_DOWN * feel.max_pitch_down(),
        )
    )
    roll = math.radians(_clip(roll_deg, -_MAX_TURN * feel.max_roll(), _MAX_TURN * feel.max_roll()))
    return project_xy(xs, ys, yaw, pitch, roll, radius, scale)


def mesh_center(pts: np.ndarray) -> tuple[float, float]:
    if len(pts) > 15 and float(pts[15, 2]) >= 0.05:
        return float(pts[15, 0]), float(pts[15, 1])
    jaw = pts[:5, :2]
    return float(np.mean(jaw[:, 0])), float(np.mean(jaw[:, 1]))


def _mesh_scale(pts: np.ndarray) -> float:
    if len(pts) > 4:
        span = float(np.linalg.norm(pts[4, :2] - pts[0, :2]))
        if span > 1.0:
            return span
    xs = pts[: min(17, len(pts)), 0]
    return float(max(xs.max() - xs.min(), 1.0))


def _body_xy(pose: dict[str, float]) -> tuple[float, float]:
    """Walk from the face box, not the nose — looking must not count as a step."""
    if "bx" in pose and "by" in pose:
        return float(pose["bx"]), float(pose["by"])
    return float(pose["cx"]), float(pose["cy"])


class FaceRig:
    def __init__(self) -> None:
        # Selfie (Mirror OFF): the camera turn is reflected onto the still.
        # Yaw, roll, and the head slide negate; pitch is not a side.
        self.selfie = False
        self.reset()

    def reset(self) -> None:
        self.locked = False
        self._token: object = None
        self._rest_cx = 0.0
        self._rest_cy = 0.0
        self._rest_ms = 1.0
        self._cam_cx = 0.0
        self._cam_cy = 0.0
        self._cam_bx = 0.0
        self._cam_by = 0.0
        self._cam_scale = 1.0
        self._cam_tz = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._roll = 0.0
        self._live: dict[str, float] | None = None
        self._dx = 0.0
        self._dy = 0.0
        self._s = 1.0
        self._yaw_r = 0.0
        self._pitch_r = 0.0
        self._roll_r = 0.0

    def _lock(self, rest: np.ndarray, head: dict[str, float], pose: dict[str, float]) -> None:
        self._rest_cx, self._rest_cy = mesh_center(rest)
        self._rest_ms = _mesh_scale(rest)
        self._cam_cx = float(pose["cx"])
        self._cam_cy = float(pose["cy"])
        self._cam_bx, self._cam_by = _body_xy(pose)
        self._cam_scale = max(float(pose["scale"]), 1.0)
        self._cam_tz = max(float(pose.get("tz", 0.0) or 0.0), 0.0)
        self._pitch = float(head.get("pitch", 0.0))
        self._yaw = float(head.get("yaw", 0.0))
        self._roll = float(pose.get("tilt", head.get("roll", 0.0)))
        self._live = {
            "cx": self._cam_cx,
            "cy": self._cam_cy,
            "bx": self._cam_bx,
            "by": self._cam_by,
            "scale": self._cam_scale,
            "tz": self._cam_tz,
            "pitch": self._pitch,
            "yaw": self._yaw,
            "roll": self._roll,
        }
        self.locked = True
        self._token = rest_stamp()

    def _sync(self, rest: np.ndarray, head: dict[str, float], pose: dict[str, float]) -> bool:
        if not pose.get("ok"):
            return False
        stamp = rest_stamp()
        if not self.locked:
            self._lock(rest, head, pose)
        elif stamp != self._token and session_rest_locked():
            snap = stamp[1] if isinstance(stamp, tuple) and len(stamp) > 1 else None
            if snap is not None:
                self._lock(rest, head, pose)
        raw_p = float(head.get("pitch", 0.0))
        if abs(raw_p - self._pitch) > 70.0:
            raw_p = float(self._live["pitch"]) if self._live is not None else self._pitch
        body_x, body_y = _body_xy(pose)
        nxt = {
            "cx": float(pose["cx"]),
            "cy": float(pose["cy"]),
            "bx": body_x,
            "by": body_y,
            "scale": max(float(pose["scale"]), 1.0),
            "tz": max(float(pose.get("tz", 0.0) or 0.0), 0.0),
            "pitch": raw_p,
            "yaw": float(head.get("yaw", 0.0)),
            "roll": float(pose.get("tilt", head.get("roll", 0.0))),
        }
        alpha = max(feel.alpha(), 0.28)
        if self._live is None:
            self._live = nxt
        else:
            self._live = {
                key: self._live[key] + alpha * (nxt[key] - self._live[key]) for key in nxt
            }
        live = self._live
        side = -1.0 if self.selfie else 1.0
        yaw_delta = side * (live["yaw"] - self._yaw)
        roll_delta = side * (live["roll"] - self._roll)
        self._yaw_r = math.radians(
            _clip(yaw_delta, -_MAX_TURN * feel.max_yaw(), _MAX_TURN * feel.max_yaw())
        )
        # Positive pitch is look-down. Up / down has no side, so no flip.
        pitch_delta = live["pitch"] - self._pitch
        self._pitch_r = math.radians(
            _clip(
                pitch_delta,
                -_MAX_LOOK_UP * feel.max_pitch_up(),
                _MAX_LOOK_DOWN * feel.max_pitch_down(),
            )
        )
        self._roll_r = math.radians(
            _clip(
                roll_delta,
                -_MAX_TURN * feel.max_roll(),
                _MAX_TURN * feel.max_roll(),
            )
        )
        cam_s = max(self._cam_scale, 1.0)
        img_s = self._rest_ms / cam_s
        dx = side * (live["bx"] - self._cam_bx) * img_s
        self._dx = _clip(dx, -self._rest_ms * 2.2, self._rest_ms * 2.2)
        self._dy = _clip(
            (live["by"] - self._cam_by) * img_s, -self._rest_ms * 1.6, self._rest_ms * 1.6
        )
        # Size from solved distance. It is the same whether you face the
        # camera or turn, so a look never reads as a zoom. Box width is the
        # fallback when PnP is missing.
        if self._cam_tz > 1e-3 and live["tz"] > 1e-3:
            raw_s = self._cam_tz / live["tz"]
        else:
            raw_s = live["scale"] / cam_s
        self._s = _clip(raw_s, 0.62, 1.70)
        return True

    def place(self) -> dict[str, float]:
        """Face-owned location and size for child overlays (skeleton, hair)."""
        return {
            "dx": float(self._dx),
            "dy": float(self._dy),
            "scale": float(self._s),
            "cx": float(self._rest_cx),
            "cy": float(self._rest_cy),
        }

    def _project(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        max_turn: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        radius = max(self._rest_ms * 1.05 * self._s, 1.0)
        yaw = self._yaw_r
        pitch = self._pitch_r
        if max_turn is not None:
            cap = math.radians(float(max_turn))
            yaw = _clip(yaw, -cap, cap)
            pitch = _clip(pitch, -cap, cap)
        return project_xy(xs, ys, yaw, pitch, self._roll_r, radius, self._s)

    def map_local(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        max_turn: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rest-centered coords into posed image space (same 2.5D as the face)."""
        x2, y2 = self._project(
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            max_turn=max_turn,
        )
        return self._rest_cx + self._dx + x2, self._rest_cy + self._dy + y2

    def map_flat(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Translate / scale / roll only — no yaw or pitch."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        c = math.cos(self._roll_r)
        s = math.sin(self._roll_r)
        xr = self._s * (xs * c - ys * s)
        yr = self._s * (xs * s + ys * c)
        return self._rest_cx + self._dx + xr, self._rest_cy + self._dy + yr

    def apply(
        self,
        mixed: np.ndarray | None,
        rest: np.ndarray | None,
        head: dict[str, float],
        pose: dict[str, float],
    ) -> np.ndarray | None:
        if mixed is None or rest is None:
            return mixed
        if not self._sync(rest, head, pose) and not self.locked:
            return mixed
        src = mixed
        out = src.copy()
        xs = src[:, 0] - self._rest_cx
        ys = src[:, 1] - self._rest_cy
        x2, y2 = self._project(xs, ys)
        out[:, 0] = self._rest_cx + self._dx + x2
        out[:, 1] = self._rest_cy + self._dy + y2
        return out
