"""Turn webcam mouth geometry into a bounded character-shape mix."""

from __future__ import annotations

import math

import numpy as np

from .calibrate import calibrator
from .feel import feel
from .mouth_bits import bits as mouth_bits
from .presets import MOUTH_BANKS, VOWEL_IDS, empty_weights

_OPEN_SPAN = 0.16
_FORM_SCALE = 0.18
_CORNER_SCALE = 0.10
_SEED_OPEN = 0.55
# I/E/U move much less than smile/oh. Their own scale, mixed in later.
_FINE_FORM = 0.10
_FINE_DEAD = 0.07
_VOWEL_SPAN = 0.18
# Old snapshots stored absolute 3D Y (~-3). Smile is lip-relative (~±1).
_ABS_CORNER = 1.6
# dlib outer lip mid. Inner 60/64 often stay shut while 51/57 open.
_OUTER_UPPER = 51
_OUTER_LOWER = 57


class RestState:
    """This camera session's closed mouth. Disk rest from last time is not reused."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.open_rest: float | None = None
        self.width_rest: float | None = None
        self.corner_rest: float | None = None
        self.inner_rest: float | None = None
        self.misses = 0
        self.locked = False
        self._seed_n = 0
        self._snap: tuple[float, float] | None = None
        self._prev: tuple[float, float, float] | None = None
        self._yaw0: float | None = None
        self._cx0: float | None = None
        self._scale0: float | None = None
        self._branch = "init"
        # Phone input has a fixed synthetic rest. Never re-learn it from motion.
        self.frozen = False

    def snapshot(self) -> dict[str, float] | None:
        if not self.locked or self.open_rest is None or not self.width_rest:
            return None
        if self.width_rest <= 1e-5:
            return None
        out = {
            "open": float(self.open_rest),
            "width": float(self.width_rest),
            "lift": -float(self.corner_rest or 0.0),
        }
        if self.corner_rest is not None:
            out["corner"] = float(self.corner_rest)
        if self.inner_rest is not None:
            out["inner"] = float(self.inner_rest)
        return out

    def use_snapshot(self, row: dict[str, float]) -> None:
        """Closed-mouth Rest click: this face's zero, like VTS Webcam Calibrate."""
        opened = float(row["open"])
        width = float(row["width"])
        key = (round(opened, 4), round(width, 4))
        if self.locked and self._snap == key:
            return
        self.open_rest = opened
        self.width_rest = width
        corner = row.get("corner")
        if corner is not None and abs(float(corner)) < _ABS_CORNER:
            self.corner_rest = float(corner)
        else:
            self.corner_rest = None
        inner = row.get("inner")
        self.inner_rest = float(inner) if inner is not None else None
        self.locked = True
        self._snap = key
        self._seed_n = 99
        self.misses = 0
        self._prev = (opened, width, float(self.corner_rest or 0.0))

    def _still(self, opened: float, width: float, corner: float) -> bool:
        prev = self._prev
        self._prev = (opened, width, corner)
        if prev is None or prev[1] <= 1e-5:
            return False
        po, pw, pc = prev
        return (
            abs(opened - po) <= 0.018
            and abs(width / pw - 1.0) <= 0.025
            and abs(corner - pc) <= 0.035
        )

    def _mix(
        self,
        opened: float,
        width: float,
        corner: float,
        inner: float | None,
        rate: float,
    ) -> None:
        rest_o = float(self.open_rest or 0.0)
        rest_w = float(self.width_rest or width)
        self.open_rest = (1.0 - rate) * rest_o + rate * opened
        self.width_rest = (1.0 - rate) * rest_w + rate * width
        self.corner_rest = (1.0 - rate) * float(self.corner_rest or corner) + rate * corner
        if inner is None:
            return
        if self.inner_rest is None:
            self.inner_rest = inner
            return
        self.inner_rest = (1.0 - rate) * self.inner_rest + rate * inner

    def observe(
        self,
        opened: float,
        width: float,
        corner: float,
        seen: bool,
        inner: float | None = None,
        yaw: float | None = None,
        cx: float | None = None,
        scale: float | None = None,
    ) -> None:
        if self.frozen:
            self._branch = "frozen"
            return
        if not seen:
            self.misses += 1
            self._prev = None
            if self.misses > 40:
                self.reset()
            return
        self.misses = 0
        if width <= 1e-5:
            return
        still = self._still(opened, width, corner)
        if not still:
            self._branch = "moving"
            return
        if self.width_rest is None:
            if opened > _SEED_OPEN:
                self._branch = "seed_skip_open"
                return
            self.open_rest = opened
            self.width_rest = width
            self.corner_rest = corner
            self.inner_rest = inner
            self._yaw0 = yaw
            self._cx0 = cx
            self._scale0 = scale
            self._seed_n = 1
            self._branch = "seed"
            return
        if self._yaw0 is None and yaw is not None:
            self._yaw0 = yaw
        if self._cx0 is None and cx is not None:
            self._cx0 = cx
            self._scale0 = scale
        rest_o = float(self.open_rest or 0.0)
        rest_w = float(self.width_rest)
        quieter = opened + 0.002 < rest_o
        closed = opened <= rest_o + 0.03
        near = abs(width - rest_w) <= 0.035 * max(rest_w, 1e-4)
        home = False
        if self._yaw0 is not None and yaw is not None:
            home = abs(yaw - self._yaw0) <= 12.0
        if self._cx0 is not None and cx is not None and self._scale0:
            at_cx = abs(cx - self._cx0) <= 0.16 * max(self._scale0, 1.0)
            home = at_cx if self._yaw0 is None or yaw is None else (home and at_cx)
        # Walked off-center, then back: PnP hysteresis looks like an open
        # mouth. Re-zero rest at the start pose instead of keeping A on.
        if home and opened <= rest_o + 0.14:
            self._mix(opened, width, corner, inner, 0.28 if self.locked else 0.22)
            self._branch = "home_rezero"
            if not self.locked:
                self._seed_n += 1
                if self._seed_n >= 10:
                    self.locked = True
            return
        if not closed:
            self._branch = "skip_not_closed"
            return
        if not (near or quieter):
            self._branch = "skip_not_near"
            return
        self._mix(opened, width, corner, inner, 0.08 if self.locked else 0.22)
        self._branch = "follow_closed"
        if not self.locked:
            self._seed_n += 1
            if self._seed_n >= 10:
                self.locked = True


_rest = RestState()


def reset_visemes() -> None:
    _rest.reset()


def session_rest_locked() -> bool:
    return _rest.locked


def rest_stamp() -> object:
    return (_rest.locked, _rest._snap)


def apply_calibrated_rest() -> None:
    row = calibrator.rest_snapshot()
    if row:
        _rest.use_snapshot(row)


def freeze_rest(face: object, pose: dict[str, float] | None = None) -> bool:
    """Lock this session's zero to ``face`` (a closed synthetic mouth) and stop
    the still-frame re-learning that lets a talking phone user drift rest open."""
    _rest.reset()
    feat = mouth_features(face, pose)
    if feat is None:
        return False
    _rest.use_snapshot(feat)
    _rest.frozen = True
    return True


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else float(value)


def _clip11(value: float) -> float:
    return -1.0 if value < -1.0 else 1.0 if value > 1.0 else float(value)


def _smoothstep(lo: float, hi: float, value: float) -> float:
    span = hi - lo
    if span <= 1e-6:
        return 1.0 if value >= hi else 0.0
    t = _clip((value - lo) / span)
    return t * t * (3.0 - 2.0 * t)


def mouth_features(
    face: object | None,
    pose: dict[str, float] | None = None,
) -> dict[str, float] | None:
    if face is None:
        _rest.observe(0.0, 0.0, 0.0, False)
        return None
    pnp_error = float(getattr(face, "pnp_error", 0.0) or 0.0)
    lms = getattr(face, "lms", None)
    if pnp_error > 300.0:
        _rest.observe(0.0, 0.0, 0.0, False)
        return None
    roles = mouth_bits.roles()
    if roles is None:
        _rest.observe(0.0, 0.0, 0.0, False)
        return None
    left, upper_i, right, lower = roles
    need = max(left, upper_i, right, lower)
    if lms is not None:
        landmark_rows = np.asarray(lms, dtype=np.float32)
        if landmark_rows.ndim == 2 and landmark_rows.shape[0] > need and landmark_rows.shape[1] > 2:
            conf = float(np.mean(landmark_rows[[left, upper_i, right, lower], 2]))
            if conf < 0.35:
                _rest.observe(0.0, 0.0, 0.0, False)
                return None
    pts = getattr(face, "pts_3d", None)
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= need or arr.shape[1] < 2:
        return None
    xy = arr[:, :2]
    norm_x = float(np.mean([xy[0, 0] - xy[16, 0], xy[1, 0] - xy[15, 0]]))
    norm_y = float(np.mean([xy[27, 1] - xy[28, 1], xy[28, 1] - xy[29, 1], xy[29, 1] - xy[30, 1]]))
    if abs(norm_x) < 1e-5 or abs(norm_y) < 1e-5:
        return None
    opened = abs(float(xy[upper_i, 1] - xy[lower, 1]))
    if len(xy) > _OUTER_LOWER:
        # Inner ring can stay a slit. Outer 51/57 still split on a real open.
        opened = max(opened, abs(float(xy[_OUTER_UPPER, 1] - xy[_OUTER_LOWER, 1])))
    mouth_w = abs(float(xy[left, 0] - xy[right, 0]))
    if mouth_w <= 1e-5:
        return None
    # Lip aspect, not nose length. Yaw shrinks the nose and used to look like A.
    opened = opened / mouth_w
    width = mouth_w / abs(norm_x)
    # Corners vs upper lip, not absolute 3D Y. Head pitch / new PnP must not
    # look like a smile after Reset.
    upper = float(xy[upper_i, 1]) / abs(norm_y)
    corners = 0.5 * float(xy[left, 1] + xy[right, 1]) / abs(norm_y)
    corner = corners - upper
    lift = -corner
    yaw = None
    euler = getattr(face, "euler", None)
    if euler is not None:
        vals = np.asarray(euler, dtype=np.float32).reshape(-1)
        if vals.size >= 2:
            yaw = float(vals[1])
    pose = pose if isinstance(pose, dict) else None
    _rest.observe(
        opened,
        width,
        corner,
        True,
        yaw=yaw,
        cx=float(pose["cx"]) if pose and pose.get("ok") else None,
        scale=float(pose["scale"]) if pose and pose.get("ok") else None,
    )
    return {
        "open": opened,
        "width": width,
        "corner": corner,
        "lift": lift,
    }


def _deadzone(value: float, dead: float = 0.12) -> float:
    mag = abs(value)
    if mag <= dead:
        return 0.0
    signed = 1.0 if value > 0.0 else -1.0
    return signed * _clip((mag - dead) / (1.0 - dead))


def _geo_axes(feat: dict[str, float]) -> tuple[float, float, float]:
    raw_open = float(feat["open"])
    raw_width = float(feat["width"])
    raw_corner = float(feat.get("corner", 0.0))
    rest_w = _rest.width_rest
    rest_o = _rest.open_rest
    rest_c = _rest.corner_rest
    if not _rest.locked or rest_o is None or not rest_w or rest_w <= 1e-5:
        return 0.0, 0.0, 0.0
    opened = _smoothstep(rest_o + 0.02, rest_o + _OPEN_SPAN, raw_open)
    width_delta = raw_width / rest_w - 1.0
    form_w = math.tanh(width_delta / _FORM_SCALE)
    corner = 0.0
    if rest_c is not None:
        corner = math.tanh((raw_corner - rest_c) / _CORNER_SCALE)
    return opened, _clip11(form_w), _clip11(corner)


def _axes(feat: dict[str, float]) -> tuple[float, float, float]:
    opened, form, corner = _geo_axes(feat)
    # Blank face jitter is mostly corners. Need a real grin to leave rest.
    c_dead = 0.22 if opened < 0.10 else 0.12
    return opened, _deadzone(form, 0.12), _deadzone(corner, c_dead)


def _fine_form(feat: dict[str, float]) -> float:
    rest_w = _rest.width_rest
    if not _rest.locked or not rest_w or rest_w <= 1e-5:
        return 0.0
    width_delta = float(feat["width"]) / rest_w - 1.0
    return _deadzone(math.tanh(width_delta / _FINE_FORM), _FINE_DEAD)


def _vowel_open(feat: dict[str, float]) -> float:
    rest_o = _rest.open_rest
    if not _rest.locked or rest_o is None:
        return 0.0
    return _smoothstep(rest_o + 0.012, rest_o + _VOWEL_SPAN, float(feat["open"]))


def _heuristic_bank(feat: dict[str, float], bank_id: str) -> dict[str, float]:
    """One setup's weights. Spread/round use a finer lip scale than corners/open."""
    opened, form, corner = _axes(feat)
    amount = _clip(opened * feel.response())
    out: dict[str, float] = {}
    if bank_id == "corners":
        out["smile"] = _clip(max(corner, 0.0) * (1.0 - 0.55 * amount), 0.0, 0.7)
        out["sad"] = _clip(max(-corner, 0.0) * (1.0 - 0.55 * amount), 0.0, 0.7)
        return out
    if bank_id == "open":
        out["A"] = amount * (1.0 - abs(form))
        return out
    fine = _fine_form(feat)
    gain = _clip(_vowel_open(feat) * feel.response())
    high = _smoothstep(0.32, 0.72, _vowel_open(feat))
    if bank_id == "spread":
        spread = max(fine, 0.0)
        out["I"] = gain * spread * (1.0 - high)
        out["E"] = gain * spread * high
        return out
    if bank_id == "round":
        rounded = max(-fine, 0.0)
        out["U"] = gain * rounded
        return out
    return out


def _heuristic(feat: dict[str, float]) -> dict[str, float]:
    out = empty_weights()
    for bank_id, _label, names in MOUTH_BANKS:
        part = _heuristic_bank(feat, bank_id)
        for name in names:
            out[name] = float(part.get(name, 0.0))
    total = sum(out[name] for name in VOWEL_IDS)
    if total > 1.0:
        for name in VOWEL_IDS:
            out[name] /= total
    return out


def viseme_weights(
    face: object | None,
    pose: dict[str, float] | None = None,
) -> dict[str, float]:
    feat = mouth_features(face, pose)
    if calibrator.ingest(feat) == "rest":
        apply_calibrated_rest()
    out = calibrator.weights(feat, _heuristic, origin=_rest.snapshot())
    return out
