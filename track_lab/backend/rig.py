"""Drive the full 28-point mesh from OSF head pose.

The tracked face writes one place for the whole character: slide, size,
and turn. The chin rides with that place. Hair uses the same transform; the
skeleton shares the slide and size, and only its neck rides the turn, so the
body does not stay behind a moving face nor turn with a look.

The face and the skeleton turn as a flat drawing. A bowl would cave the
jaw and fold distant bones into the head; a long lens keeps the authored
spacing while a nod or turn still foreshortens.
"""

from __future__ import annotations

import math

import numpy as np

from .feel import feel
from .travel_box import SIZE_MAX, soft_barrier
from .visemes import rest_stamp, session_rest_locked

_MAX_TURN = 80.0
# Turn / tilt walls are nearly hard. The shared 45% spring let a 33 deg roll
# box reach ~48 deg, which tipped the drawn head sideways on a big turn.
_TURN_GIVE = 0.1
_MAX_LOOK_DOWN = 32.0
_MAX_LOOK_UP = 50.0
# Size zero is a still distance, not the first solve. PnP's opening guess is
# often far; locking it makes the overlay sit at the zoom cap until Set Rest.
_SIZE_HOLD = 5
_SIZE_BAND = 0.08
# Front hemisphere. Hair outside this disk is pinned here; extra length is fluff.
_RN_MAX = 0.92
_FOCAL = 1.8
_PERSP_MIN = 0.72
_PERSP_MAX = 1.28
# Portrait lens for the drawn face. The wide lens above is for hair and the
# IFM preview; on the character it ballooned the near eye and crushed the jaw.
_FACE_FOCAL = 6.0
_FACE_PERSP_MIN = 0.94
_FACE_PERSP_MAX = 1.06


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


def plane_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    yaw_r: float,
    pitch_r: float,
    roll_r: float,
    radius: float,
    scale: float = 1.0,
    *,
    focal: float | None = None,
    persp_min: float | None = None,
    persp_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Yaw/pitch a flat card. Overlay polygons stay shapes; they do not fold."""
    xs = np.asarray(xs, dtype=np.float64) * scale
    ys = np.asarray(ys, dtype=np.float64) * scale
    radius = max(float(radius), 1.0)
    yaw_r = float(yaw_r)
    # Same look-up sign as _posed_sphere.
    pitch_r = -float(pitch_r)
    roll_r = float(roll_r)
    cy, sy = math.cos(yaw_r), math.sin(yaw_r)
    cp, sp = math.cos(pitch_r), math.sin(pitch_r)
    cr, sr = math.cos(roll_r), math.sin(roll_r)
    z = radius

    def _card(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x2 = px * cy + z * sy
        z2 = -px * sy + z * cy
        y2 = py * cp - z2 * sp
        z3 = py * sp + z2 * cp
        xr = x2 * cr - y2 * sr
        yr = x2 * sr + y2 * cr
        return xr, yr, z3

    xr, yr, zr = _card(xs, ys)
    ox, oy, oz = _card(np.zeros(1), np.zeros(1))
    z_rel = (zr - z) - (float(oz[0]) - z)
    focal_len = (_FOCAL if focal is None else float(focal)) * radius
    lo = _PERSP_MIN if persp_min is None else float(persp_min)
    hi = _PERSP_MAX if persp_max is None else float(persp_max)
    persp = np.clip(focal_len / np.maximum(focal_len - z_rel, 1e-3), lo, hi)
    return (xr - float(ox[0])) * persp, (yr - float(oy[0])) * persp


def face_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    yaw_r: float,
    pitch_r: float,
    roll_r: float,
    radius: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn the drawn face. Feature spacing stays put; the jaw does not cave."""
    return plane_xy(
        xs,
        ys,
        yaw_r,
        pitch_r,
        roll_r,
        radius,
        scale,
        focal=_FACE_FOCAL,
        persp_min=_FACE_PERSP_MIN,
        persp_max=_FACE_PERSP_MAX,
    )


def _turn_stop(deg: float, frac: tuple[float, float]) -> float:
    """Stop a turn or tilt at its (left, right) fractions; right is positive."""
    left, right = frac
    return soft_barrier(deg, -_MAX_TURN * left, _MAX_TURN * right, 0.0, give=_TURN_GIVE)


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
    yaw = math.radians(
        _turn_stop(yaw_deg, feel.max_yaw())
    )
    pitch_deg = float(head.get("pitch", 0.0))
    pitch = math.radians(
        _clip(
            pitch_deg,
            -_MAX_LOOK_UP * feel.max_pitch_up(),
            _MAX_LOOK_DOWN * feel.max_pitch_down(),
        )
    )
    roll = math.radians(
        _turn_stop(roll_deg, feel.max_roll())
    )
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
        self._size_ready = False
        self._size_mode = ""
        self._size_ring: list[tuple[float, float]] = []
        self._size_pending: float | None = None
        self._size_confirm = 0

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

    def _seal_size(self, tz: float, scale: float) -> None:
        """Distance that means overlay scale 1."""
        if tz > 1e-3:
            self._cam_tz = float(tz)
            self._cam_scale = float(scale)
        else:
            self._cam_tz = 0.0
            self._cam_scale = float(scale)
        self._size_ready = True
        self._size_ring = []
        self._size_pending = None
        self._size_confirm = 0

    def _note_size(self, pose: dict[str, float], *, force: bool) -> None:
        tz = max(float(pose.get("tz", 0.0) or 0.0), 0.0)
        scale = max(float(pose["scale"]), 1.0)
        if force:
            self._seal_size(tz, scale)
            return
        if self._size_ready:
            return
        mode = "tz" if tz > 1e-3 else "scale"
        if mode != self._size_mode:
            # Solved distance and box width are different units. Start over.
            self._size_mode = mode
            self._size_ring = []
            self._size_pending = None
            self._size_confirm = 0
        self._size_ring.append((tz, scale))
        if len(self._size_ring) > _SIZE_HOLD:
            self._size_ring.pop(0)
        if len(self._size_ring) < _SIZE_HOLD:
            return
        keys = [row[0] if mode == "tz" else row[1] for row in self._size_ring]
        med = float(np.median(keys))
        if med <= 1e-3 or any(abs(key / med - 1.0) > _SIZE_BAND for key in keys):
            self._size_pending = None
            self._size_confirm = 0
            return
        if self._size_pending is None or abs(med / self._size_pending - 1.0) > _SIZE_BAND:
            self._size_pending = med
            self._size_confirm = 1
            return
        self._size_confirm += 1
        if self._size_confirm < _SIZE_HOLD:
            return
        if mode == "tz":
            seal_tz = med
            seal_scale = float(np.median([row[1] for row in self._size_ring]))
        else:
            seal_tz = 0.0
            seal_scale = med
        self._seal_size(seal_tz, seal_scale)

    def _sync(self, rest: np.ndarray, head: dict[str, float], pose: dict[str, float]) -> bool:
        if not pose.get("ok"):
            return False
        stamp = rest_stamp()
        relock = False
        if not self.locked:
            self._lock(rest, head, pose)
        elif stamp != self._token and session_rest_locked():
            snap = stamp[1] if isinstance(stamp, tuple) and len(stamp) > 1 else None
            if snap is not None:
                self._lock(rest, head, pose)
                relock = True
        # Set Rest grabs the face in hand. Otherwise wait until distance
        # holds still, so the opening solve cannot pin the overlay large.
        self._note_size(pose, force=relock)
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
            "tilt_eyes": float(pose.get("tilt_eyes", 0.0)),
            "pnp_pitch": float(head.get("pitch", 0.0)),
            "pnp_roll": float(head.get("roll", 0.0)),
        }
        # One ease only: the drawn points use Feel.smoothing. Blending the
        # pose here too made the head trail a second time, so tracking felt late.
        self._live = nxt
        live = self._live
        side = -1.0 if self.selfie else 1.0
        yaw_delta = side * (live["yaw"] - self._yaw)
        # Tilt is an angle: 179 -> -179 is 2 deg, not a 358 deg flip.
        roll_delta = side * (((live["roll"] - self._roll) + 180.0) % 360.0 - 180.0)
        self._yaw_r = math.radians(
            _turn_stop(yaw_delta, feel.max_yaw())
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
            _turn_stop(roll_delta, feel.max_roll())
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
        # fallback when PnP is missing. Until the distance holds still, stay
        # at rest size — the first solves are the ones that read too far.
        if not self._size_ready:
            self._s = 1.0
        else:
            if self._cam_tz > 1e-3 and live["tz"] > 1e-3:
                raw_s = self._cam_tz / live["tz"]
            else:
                raw_s = live["scale"] / cam_s
            # Size limiter: stepping back must not shrink the face off the art.
            room = SIZE_MAX * feel.max_size()
            self._s = _clip(soft_barrier(raw_s, 1.0 - room, 1.0 + room, 1.0), 0.62, 1.70)
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

    def turn(self) -> dict[str, float]:
        """Mirrored yaw / pitch / roll in radians (selfie already applied)."""
        return {
            "yaw": float(self._yaw_r),
            "pitch": float(self._pitch_r),
            "roll": float(self._roll_r),
        }

    def debug(self) -> dict[str, float]:
        """Drawn turn in degrees plus the raw tilt it came from."""
        live = self._live or {}
        return {
            "yaw": round(math.degrees(self._yaw_r), 2),
            "pitch": round(math.degrees(self._pitch_r), 2),
            "roll": round(math.degrees(self._roll_r), 2),
            "tilt_live": round(float(live.get("roll", 0.0)), 2),
            "tilt_rest": round(float(self._roll), 2),
            "yaw_live": round(float(live.get("yaw", 0.0)), 2),
            "yaw_rest": round(float(self._yaw), 2),
            "tilt_eyes": round(float(live.get("tilt_eyes", 0.0)), 2),
            "pnp_pitch": round(float(live.get("pnp_pitch", 0.0)), 2),
            "pnp_roll": round(float(live.get("pnp_roll", 0.0)), 2),
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
        return face_xy(xs, ys, yaw, pitch, self._roll_r, radius, self._s)

    def _plane(
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
        return plane_xy(xs, ys, yaw, pitch, self._roll_r, radius, self._s)

    def map_local(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        max_turn: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rest-centered coords into posed image space (same card turn as the face)."""
        x2, y2 = self._project(
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            max_turn=max_turn,
        )
        return self._rest_cx + self._dx + x2, self._rest_cy + self._dy + y2

    def map_plane(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        max_turn: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rest-centered overlay shapes: same place as the face, planar turn."""
        x2, y2 = self._plane(
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
