"""Drive the full 28-point mesh from OSF head pose.

The tracked face writes one place for the whole character: slide, size,
and turn. The chin rides with that place and hair uses the same transform.
The head's slide is the body's walk plus its swing round the neck; the
skeleton takes the walk and size only, so a turn or nod moves the head and
leaves the torso where it is.

The face turns as a flat drawing lifted by a little depth per point. A
bowl would cave the jaw and fold distant bones into the head; a long lens
keeps the authored spacing while a nod or turn still foreshortens, and the
depth carries the nose and mouth into the turn or nod.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .calibrate import WARMUP_SEC, calibrator
from .ease import HeadEase
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


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def head_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Head rotation for rig angles in degrees.

    Image frame (x right, y down, z away from the camera), identity facing
    it: H = Rz(roll) Rx(pitch) Ry(-yaw), the order osf_cam._pnp_head reads
    a solve in. Yaw + faces image-right, pitch + looks down, roll + is
    clockwise.
    """
    return (
        _rot_z(math.radians(roll))
        @ _rot_x(math.radians(pitch))
        @ _rot_y(math.radians(-yaw))
    )


def head_matrix_yaw_outer(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """The same angles composed yaw outermost, Ry(-yaw) Rx(pitch) Rz(roll):
    the Unity order iFacialMocap sends ARKit's head in."""
    return (
        _rot_y(math.radians(-yaw))
        @ _rot_x(math.radians(pitch))
        @ _rot_z(math.radians(roll))
    )


def head_angles(rot: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees: the inverse of ``head_matrix``."""
    pitch = math.degrees(math.asin(_clip(float(rot[2, 1]), -1.0, 1.0)))
    yaw = -math.degrees(math.atan2(-float(rot[2, 0]), float(rot[2, 2])))
    roll = math.degrees(math.atan2(-float(rot[0, 1]), float(rot[1, 1])))
    return yaw, pitch, roll


# Where the eyes sit from the neck's pivot, in face widths (the jaw span the
# rig scales by). A webcam watches the eyes swing round that pivot on every
# turn, nod and tilt, and the head slides with them; the iPhone only sends
# angles. Both sources swing the drawn head here, by the turn as drawn, so a
# turn / nod / tilt stop also stops the swing the same way on either.
NECK_FORWARD = 0.55
NECK_UP = 0.5
# What the webcam's eyes still move once the head's swing is taken out is
# the body walking or leaning. The torso ignores this much of it (face
# widths), so a neck that swings a little unlike NECK_FORWARD does not drag
# the shoulders on a nod.
_WALK_DEADBAND = 0.08
# Past this nod (degrees from rest) the torso's vertical walk holds. A real
# neck swings the eyes by up to ~1.5x (or 0.5x) what NECK_UP models; on a
# 20-30 deg nod that miss read as a walk of 20+ px and dragged the shoulders.
# The head still follows the eyes the camera sees. Blends in from rest.
_NOD_WALK_HOLD_DEG = 6.0
# How far each face point sits in front of the eye line (+) or behind it,
# in head radii. A flat card only squashed: a nod read as the face sliding
# down and a turn as it slimming, with the nose and mouth still centred
# between the cheeks. With depth a look-down drops the nose toward the mouth
# and raises the cheeks, and a turn carries the nose and mouth toward it
# while the near cheek widens and the far one closes in.
FACE_DEPTH = np.array(
    [
        -0.24, -0.14, 0.02, -0.14, -0.24,  # outline: cheek, jaw, chin, jaw, cheek
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02,  # brows
        -0.05, 0.0, 0.0,  # eye: outer corner, lid, inner corner
        0.06, 0.10, 0.06,  # nose
        0.0, 0.0, -0.05,  # eye: inner corner, lid, outer corner
        0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,  # mouth
    ],
    dtype=np.float64,
)
# Each outline point and the feature it stays outside of: cheek past the
# eye's outer corner, jaw past the mouth corner. Depth swings the far cheek
# in on a turn; past the eye it drew the eye outside the face. The gap may
# close to this share of the drawn one.
_SILHOUETTE = ((0, 11), (1, 23), (4, 19), (3, 26))
_SILHOUETTE_KEEP = 0.3


def neck_offset(yaw_r: float, pitch_r: float, roll_r: float) -> tuple[float, float]:
    """Screen (x right, y down) move of the eyes, in face widths, for the
    drawn turn (radians, relative to rest): a turn swings them sideways
    round a neck behind them, a nod down or up round it, a tilt sideways
    round a pivot below them.

    The pivot is below the eyes for a nod too, so a look-down drops them a
    little further than a look-up raises them. Without that, a webcam's
    look-down past the stop read the extra drop as the body walking down,
    and its look-up sank back.

    Each motion keeps to its own axis. As one rigid rotation, the small
    tilt a real turn carries rolled the swung-out eyes up or down, and a
    plain turn read as the head rising (measured on a real phone).
    """
    x = NECK_FORWARD * math.sin(yaw_r) + NECK_UP * math.sin(roll_r)
    y = (
        NECK_FORWARD * math.sin(pitch_r)
        + NECK_UP * (1.0 - math.cos(pitch_r))
        + NECK_UP * (1.0 - math.cos(roll_r))
    )
    return x, y


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
    depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Yaw/pitch a flat card. Overlay polygons stay shapes; they do not fold.

    ``depth`` (pixels toward the camera, per point) lifts points off the
    card: one in front goes further on a turn or a nod. The rest pose and
    the lens are the flat card's.
    """
    xs = np.asarray(xs, dtype=np.float64) * scale
    ys = np.asarray(ys, dtype=np.float64) * scale
    ds = np.zeros_like(xs)
    if depth is not None:
        ds = np.broadcast_to(np.asarray(depth, dtype=np.float64), xs.shape)
    radius = max(float(radius), 1.0)
    yaw_r = float(yaw_r)
    # Same look-up sign as _posed_sphere.
    pitch_r = -float(pitch_r)
    roll_r = float(roll_r)
    cy, sy = math.cos(yaw_r), math.sin(yaw_r)
    cp, sp = math.cos(pitch_r), math.sin(pitch_r)
    cr, sr = math.cos(roll_r), math.sin(roll_r)
    z = radius

    def _card(px: np.ndarray, py: np.ndarray, pd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x2 = px * cy + (z + pd) * sy
        z2 = -px * sy + (z + pd) * cy
        y2 = py * cp - z2 * sp
        z3 = py * sp + z2 * cp - pd
        xr = x2 * cr - y2 * sr
        yr = x2 * sr + y2 * cr
        return xr, yr, z3

    xr, yr, zr = _card(xs, ys, ds)
    ox, oy, oz = _card(np.zeros(1), np.zeros(1), np.zeros(1))
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
    depth: np.ndarray | None = None,
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
        depth=depth,
    )


def face_depth(count: int, radius: float) -> np.ndarray:
    """FACE_DEPTH for ``count`` face rows, in pixels at this radius."""
    out = np.zeros(int(count), dtype=np.float64)
    n = min(int(count), len(FACE_DEPTH))
    out[:n] = FACE_DEPTH[:n] * float(radius)
    return out


def keep_silhouette(
    x2: np.ndarray,
    y2: np.ndarray,
    flat: np.ndarray,
    roll_r: float,
    scale: float = 1.0,
) -> None:
    """Hold each cheek / jaw point outside its eye / mouth corner, in place.

    ``x2`` / ``y2`` are the turned face rows, ``flat`` the same rows before
    the turn ((n, 3): x, y, score). The gap is read along the tilted face's
    across axis; it closes freely down to twice the floor, then eases onto
    the floor and never crosses it.
    """
    ux, uy = math.cos(roll_r), math.sin(roll_r)
    n = min(len(x2), len(flat))
    for edge, inner in _SILHOUETTE:
        if max(edge, inner) >= n or min(float(flat[edge, 2]), float(flat[inner, 2])) < 0.05:
            continue
        drawn = (float(flat[edge, 0]) - float(flat[inner, 0])) * scale
        if abs(drawn) < 1e-6:
            continue
        side = 1.0 if drawn > 0.0 else -1.0
        floor = _SILHOUETTE_KEEP * abs(drawn)
        gap = side * ((x2[edge] - x2[inner]) * ux + (y2[edge] - y2[inner]) * uy)
        if gap >= 2.0 * floor:
            continue
        held = floor * (1.0 + math.exp((gap - 2.0 * floor) / floor))
        x2[edge] += side * (held - gap) * ux
        y2[edge] += side * (held - gap) * uy


def _turn_stop(deg: float, frac: tuple[float, float]) -> float:
    """Stop a turn or tilt at its (left, right) fractions; right is positive."""
    left, right = frac
    return soft_barrier(deg, -_MAX_TURN * left, _MAX_TURN * right, 0.0, give=_TURN_GIVE)


def _nod_stop(deg: float) -> float:
    """Stop a nod at Look up / Look down, eased like a turn; + looks down."""
    return soft_barrier(
        deg,
        -_MAX_LOOK_UP * feel.max_pitch_up(),
        _MAX_LOOK_DOWN * feel.max_pitch_down(),
        0.0,
        give=_TURN_GIVE,
    )


def _dead(x: float, y: float, band: float) -> tuple[float, float]:
    """Zero inside the band, then carries on from its edge (no jump)."""
    size = math.hypot(x, y)
    if size <= band:
        return 0.0, 0.0
    keep = (size - band) / size
    return x * keep, y * keep


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
    pitch = math.radians(_nod_stop(float(head.get("pitch", 0.0))))
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


# Frames of bad head solves before the rig locks on the eye line instead.
_BAD_SOLVE_LOCK = 15

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
        self._rest_rot = np.eye(3)
        self._live: dict[str, float] | None = None
        # Head slide = walk + the head's swing round the neck; the torso
        # only walks (_bdx / _bdy), so a turn or nod never moves it.
        self._dx = 0.0
        self._dy = 0.0
        self._walk = (0.0, 0.0)
        self._torso_walk_y = 0.0
        self._bdx = 0.0
        self._bdy = 0.0
        self._s = 1.0
        self._yaw_r = 0.0
        self._pitch_r = 0.0
        self._roll_r = 0.0
        self._size_ready = False
        self._size_mode = ""
        self._size_ring: list[tuple[float, float]] = []
        self._size_pending: float | None = None
        self._size_confirm = 0
        self._bad_solves = 0
        # Locked on the eye line because no clean solve came: a stand-in.
        self._provisional = False
        # Solved roll minus the eye-line tilt on the last clean solve.
        self._eye_roll_gap = 0.0
        # Clean frames of Set Rest's capture window (after its warm-up), and
        # when the window was first seen.
        self._rest_hold: list[tuple[dict[str, float], dict[str, float]]] = []
        self._rest_hold_t: float | None = None
        self._ease = HeadEase()

    def _lock(
        self,
        rest: np.ndarray,
        head: dict[str, float],
        pose: dict[str, float],
        *,
        keep_size: bool = False,
    ) -> None:
        """``keep_size``: keep the sealed distance (see _note_size)."""
        self._rest_cx, self._rest_cy = mesh_center(rest)
        self._rest_ms = _mesh_scale(rest)
        self._cam_cx = float(pose["cx"])
        self._cam_cy = float(pose["cy"])
        self._cam_bx, self._cam_by = _body_xy(pose)
        if not keep_size:
            self._cam_scale = max(float(pose["scale"]), 1.0)
            self._cam_tz = max(float(pose.get("tz", 0.0) or 0.0), 0.0)
        self._pitch = float(head.get("pitch", 0.0))
        self._yaw = float(head.get("yaw", 0.0))
        self._roll = float(pose.get("tilt", head.get("roll", 0.0)))
        self._rest_rot = head_matrix(self._yaw, self._pitch, self._roll)
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
        self._walk = (0.0, 0.0)
        self._torso_walk_y = 0.0
        # A new zero lands at once, not eased in from the old one.
        self._ease.reset()
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

    def _hold_rest(self, head: dict[str, float], pose: dict[str, float], head_ok: bool) -> None:
        """Keep the clean frames of Set Rest's capture window, the way
        calibrate keeps the mouth's: past its warm-up, while the face is held."""
        if calibrator.capturing != "rest":
            self._rest_hold = []
            self._rest_hold_t = None
            return
        now = time.perf_counter()
        if self._rest_hold_t is None:
            self._rest_hold_t = now
        if head_ok and now - self._rest_hold_t >= WARMUP_SEC:
            self._rest_hold.append((dict(head), dict(pose)))

    def _held_rest(
        self, head: dict[str, float], pose: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Set Rest's zero: the median over its capture window and this
        frame, not the one frame the window ended on (a twitch there was the
        zero for the session). No window (a snapshot from disk): this frame."""
        rows = self._rest_hold + [(head, pose)]
        self._rest_hold = []
        self._rest_hold_t = None
        if len(rows) < 2:
            return head, pose

        def angle(values: list[float], near: float) -> float:
            # Unwrapped round this frame's reading: 179 and -179 are 2 apart.
            return float(np.median([near + ((v - near + 180.0) % 360.0 - 180.0) for v in values]))

        out_head = dict(head)
        for key in ("pitch", "yaw", "roll"):
            out_head[key] = angle([float(h.get(key, 0.0)) for h, _p in rows], float(head.get(key, 0.0)))
        out_pose = dict(pose)
        if "tilt" in pose:
            out_pose["tilt"] = angle([float(p.get("tilt", 0.0)) for _h, p in rows], float(pose["tilt"]))
        for key in ("cx", "cy", "bx", "by", "scale"):
            if key in pose:
                out_pose[key] = float(np.median([float(p.get(key, pose[key])) for _h, p in rows]))
        if float(pose.get("tz", 0.0) or 0.0) > 1e-3:
            # Solved distance only where there was one (0 = no PnP).
            dists = [float(p.get("tz", 0.0) or 0.0) for _h, p in rows]
            out_pose["tz"] = float(np.median([d for d in dists if d > 1e-3]))
        return out_head, out_pose

    def _sync(
        self,
        rest: np.ndarray,
        head: dict[str, float],
        pose: dict[str, float],
        *,
        snap: bool = False,
    ) -> bool:
        if not pose.get("ok"):
            return False
        stamp = rest_stamp()
        relock = False
        size_pose = pose
        # A flipped head solve must never become the zero: every later frame
        # would read as a turn / tilt away from it.
        head_ok = bool(pose.get("head_ok", 1.0))
        if not self.locked:
            if not head_ok:
                # Some cameras / faces never give a clean solve. After ~half a
                # second, lock on the eye-line tilt with the head straight on
                # rather than leaving the overlay frozen for good.
                self._bad_solves += 1
                if self._bad_solves < _BAD_SOLVE_LOCK:
                    return False
                head = {"pitch": 0.0, "yaw": 0.0, "roll": float(pose.get("tilt", 0.0))}
            self._bad_solves = 0
            self._lock(rest, head, pose)
            self._provisional = not head_ok
        elif self._provisional and head_ok:
            # The first clean solve is the real zero. Kept on the stand-in, an
            # off-level camera read as a nod pinned at the pitch stop. A size
            # zero already sealed stays: it is a 5-frame median, and taking
            # this one frame's instead ended its settling for the session.
            self._lock(rest, head, pose, keep_size=self._size_ready)
            self._provisional = False
        elif stamp != self._token and session_rest_locked() and head_ok:
            snap = stamp[1] if isinstance(stamp, tuple) and len(stamp) > 1 else None
            if snap is not None:
                held_head, size_pose = self._held_rest(head, pose)
                self._lock(rest, held_head, size_pose)
                relock = True
        self._hold_rest(head, pose, head_ok)
        # Set Rest grabs the face in hand. Otherwise wait until distance
        # holds still, so the opening solve cannot pin the overlay large.
        self._note_size(size_pose, force=relock)
        raw_p = float(head.get("pitch", 0.0))
        raw_y = float(head.get("yaw", 0.0))
        if not head_ok and self._live is not None:
            # Bad solve this frame: keep the last good turn / nod.
            raw_p = float(self._live["pitch"])
            raw_y = float(self._live["yaw"])
        if abs(raw_p - self._pitch) > 70.0:
            raw_p = float(self._live["pitch"]) if self._live is not None else self._pitch
        roll = float(pose.get("tilt", head.get("roll", 0.0)))
        eyes = float(pose.get("tilt_eyes", roll))
        if head_ok:
            self._eye_roll_gap = ((roll - eyes) + 180.0) % 360.0 - 180.0
        elif not self._provisional:
            # A bad frame's tilt is the bare eye line, which sits off the
            # solved roll on a turn (~10 deg at a 30 deg turn, camera off
            # level). Keep the last clean gap so the tilt does not twitch.
            roll = eyes + self._eye_roll_gap
        body_x, body_y = _body_xy(pose)
        nxt = {
            "cx": float(pose["cx"]),
            "cy": float(pose["cy"]),
            "bx": body_x,
            "by": body_y,
            "scale": max(float(pose["scale"]), 1.0),
            "tz": max(float(pose.get("tz", 0.0) or 0.0), 0.0),
            "pitch": raw_p,
            "yaw": raw_y,
            "roll": roll,
            "tilt_eyes": float(pose.get("tilt_eyes", 0.0)),
            "pnp_pitch": float(head.get("pitch", 0.0)),
            "pnp_roll": float(head.get("roll", 0.0)),
        }
        # The raw reading. Smooth eases the drawn head once, at the end.
        self._live = nxt
        live = self._live
        side = -1.0 if self.selfie else 1.0
        # The turn relative to rest, as one rotation. Subtracting the rest
        # angle by angle only holds for a head that rests facing the camera:
        # with the phone or webcam off-level (rest pitch ~15 deg here), a
        # plain turn read as a turn plus a nod and a tilt.
        rel_yaw, pitch_delta, rel_roll = head_angles(
            self._rest_rot.T @ head_matrix(live["yaw"], live["pitch"], live["roll"])
        )
        yaw_delta = side * rel_yaw
        roll_delta = side * rel_roll
        self._yaw_r = math.radians(
            _turn_stop(yaw_delta, feel.max_yaw())
        )
        # Positive pitch is look-down. Up / down has no side, so no flip.
        self._pitch_r = math.radians(_nod_stop(pitch_delta))
        self._roll_r = math.radians(
            _turn_stop(roll_delta, feel.max_roll())
        )
        ms = self._rest_ms
        cam_s = max(self._cam_scale, 1.0)
        # The head swings round the neck by the turn as drawn, so it slides
        # as far as it turns: past a turn / nod / tilt stop the slide stops
        # too, the same on either source. The torso never swings with it.
        nx, ny = neck_offset(self._yaw_r, self._pitch_r, self._roll_r)
        if "sway" in pose:
            # iPhone: angles only; nothing sees the head or the body move.
            sway = float(pose.get("sway") or 0.0)
            swing = (sway * nx * ms, sway * ny * ms)
            walk = (0.0, 0.0)
            torso_walk = walk
        else:
            # Webcam: the eyes move by the head's real swing plus any walk or
            # lean. Take the real turn's swing out and what is left walks.
            img_s = ms / cam_s
            seen_x = side * (live["bx"] - self._cam_bx) * img_s
            seen_y = (live["by"] - self._cam_by) * img_s
            real_x, real_y = neck_offset(
                math.radians(yaw_delta), math.radians(pitch_delta), math.radians(roll_delta)
            )
            if head_ok or self._provisional:
                self._walk = (seen_x - real_x * ms, seen_y - real_y * ms)
                hold = min(1.0, abs(pitch_delta) / _NOD_WALK_HOLD_DEG)
                self._torso_walk_y += (self._walk[1] - self._torso_walk_y) * (1.0 - hold)
            # A bad solve holds the last turn, so its swing cannot be taken
            # out of the eyes: the last walk holds with it.
            swing = (nx * ms, ny * ms)
            walk = self._walk
            torso_walk = (walk[0], self._torso_walk_y)
        self._dx = _clip(walk[0] + swing[0], -ms * 2.2, ms * 2.2)
        self._dy = _clip(walk[1] + swing[1], -ms * 1.6, ms * 1.6)
        torso_x, torso_y = _dead(torso_walk[0], torso_walk[1], _WALK_DEADBAND * ms)
        self._bdx = _clip(torso_x, -ms * 2.2, ms * 2.2)
        self._bdy = _clip(torso_y, -ms * 1.6, ms * 1.6)
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
        # Smooth: the one ease of the drawn head, for every source (ease.py).
        # Turn, nod, tilt, slide and size share one weight so they land
        # together, and past a stop the head settles into it softly. Hair and
        # the skeleton read this turn and place, so they ease with it; the
        # mouth and brows are eased apart (FaceBench) so lip sync stays quick.
        unit = 1.0 / max(ms, 1.0)
        eased = self._ease.step(
            (self._yaw_r, self._pitch_r, self._roll_r, self._dx, self._dy, self._bdx, self._bdy, self._s),
            (1.0, 1.0, 1.0, unit, unit, 0.0, 0.0, 1.0),
            feel.smooth(),
            time.perf_counter(),
            snap=snap,
        )
        (
            self._yaw_r,
            self._pitch_r,
            self._roll_r,
            self._dx,
            self._dy,
            self._bdx,
            self._bdy,
            self._s,
        ) = (float(value) for value in eased)
        return True

    def place(self) -> dict[str, float]:
        """Face-owned location and size for child overlays (skeleton, hair).

        dx / dy move the head (walk plus its swing round the neck);
        body_dx / body_dy move the torso (walk only).
        """
        return {
            "dx": float(self._dx),
            "dy": float(self._dy),
            "body_dx": float(self._bdx),
            "body_dy": float(self._bdy),
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
        *,
        face_rows: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``face_rows``: xs / ys are the 28 face points in order, so the turn
        and nod get their depth (FACE_DEPTH)."""
        radius = max(self._rest_ms * 1.05 * self._s, 1.0)
        yaw = self._yaw_r
        pitch = self._pitch_r
        if max_turn is not None:
            cap = math.radians(float(max_turn))
            yaw = _clip(yaw, -cap, cap)
            pitch = _clip(pitch, -cap, cap)
        depth = face_depth(len(xs), radius) if face_rows else None
        return face_xy(xs, ys, yaw, pitch, self._roll_r, radius, self._s, depth=depth)

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
        *,
        snap: bool = False,
    ) -> np.ndarray | None:
        """``snap``: land on this pose without easing (a held pose re-drawn
        after a limiter change)."""
        if mixed is None or rest is None:
            return mixed
        if not self._sync(rest, head, pose, snap=snap) and not self.locked:
            return mixed
        return self.project_face(mixed)

    def project_face(self, src: np.ndarray) -> np.ndarray:
        """The 28 face points through the current head, without easing it."""
        if not self.locked:
            return src.copy()
        out = src.copy()
        xs = src[:, 0] - self._rest_cx
        ys = src[:, 1] - self._rest_cy
        x2, y2 = self._project(xs, ys, face_rows=True)
        if src.shape[1] > 2:
            keep_silhouette(x2, y2, src, self._roll_r, self._s)
        out[:, 0] = self._rest_cx + self._dx + x2
        out[:, 1] = self._rest_cy + self._dy + y2
        return out
