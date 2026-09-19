"""Drive jaw, brows, eyes, nose, and mouth from OSF.

Brows, eyes, and nose copy the same image landmarks the camera preview
draws. 3D landmarks are only a fallback, and their Y axis is flipped
when the source face is Y-up.

Authored smile / sad / A I U E O mix inside the mouth box when those
shapes exist. Otherwise camera lips are copied into that same box.
"""

from __future__ import annotations

import numpy as np

from .feel import feel
from .eye_bits import EYE_ALL, bits as eye_bits
from .mouth_bits import MOUTH_ALL, bits as mouth_bits
from .presets import apply_open_offset, lip_gap
from .sides import mirror_map, mirror_osf, mirror_slot, mirror_sources, x_sign
from .visemes import rest_stamp, session_rest_locked

# Character slot → OSF dlib indices to average, in the canonical frame:
# image-left camera landmarks feed screen-left slots (see sides.py).
# 0–16 jaw, 17–21 / 22–26 brows, 36–41 / 42–47 eyes, all image-left first.
FACE_SOURCES: tuple[tuple[int, tuple[int, ...]], ...] = (
    (0, (0,)),
    (1, (4,)),
    (2, (8,)),
    (3, (12,)),
    (4, (16,)),
    (5, (17, 18)),
    (6, (18, 19, 20)),
    (7, (20, 21)),
    (8, (22, 23)),
    (9, (23, 24, 25)),
    (10, (25, 26)),
    # Eyes sweep left→right like the training flip pairs (11↔19, 13↔17):
    # 11 / 19 are the outer corners, 13 / 17 the inner ones by the nose.
    (11, (36,)),
    (12, (37, 38, 41, 40)),
    (13, (39,)),
    (14, (31,)),
    (15, (30,)),
    (16, (35,)),
    (17, (42,)),
    (18, (43, 44, 47, 46)),
    (19, (45,)),
)
_SELFIE_SOURCES = mirror_sources(FACE_SOURCES)
_BROW_OSF = tuple(range(17, 27))
FACE_TRACK = frozenset(i for _slot, src in FACE_SOURCES for i in src) | frozenset(_BROW_OSF)
_SLOTS = tuple(slot for slot, _src in FACE_SOURCES)
_SLOT_INDEX = {slot: i for i, slot in enumerate(_SLOTS)}
_NEED = 66
_CLIP = 0.32
_JAW = frozenset((0, 1, 2, 3, 4))
_MOUTH_ALONG = 0.38
_NOSTRIL_TO_MOUTH = 2.2
_JAW_MOUTH = 0.36
_NOSTRIL_MIN = 0.05
_BOX_ASPECT = 0.45
_CAGE_PAD = 0.02
_MOUTH_SLOTS = tuple(range(20, 28))
_LEFT_EYE = (11, 12, 13)
_RIGHT_EYE = (17, 18, 19)
_LEFT_BROW = (5, 6, 7)
_RIGHT_BROW = (8, 9, 10)
_EYE_SLOTS = _LEFT_EYE + _RIGHT_EYE
_BROW_SLOTS = _LEFT_BROW + _RIGHT_BROW
_EYE_IDX = tuple(i for i, slot in enumerate(_SLOTS) if slot in _EYE_SLOTS)
_BROW_IDX = tuple(i for i, slot in enumerate(_SLOTS) if slot in _BROW_SLOTS)
_REMAP_STEP = 0.04


def _ease(t: float) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else float(t)
    return t * t * (3.0 - 2.0 * t)


def _sample(xy: np.ndarray, src: tuple[int, ...]) -> np.ndarray:
    return np.mean(xy[list(src), :2], axis=0)


def _source_xy(
    pts3: np.ndarray,
    mouth_pts: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    """Prefer the same image landmarks the camera preview draws."""
    if mouth_pts is not None:
        arr = np.asarray(mouth_pts, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= _NEED and arr.shape[1] >= 2:
            return arr[:, :2], True
    return np.asarray(pts3, dtype=np.float32)[:, :2], False


def _face_sources(selfie: bool) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return _SELFIE_SOURCES if selfie else FACE_SOURCES


def _source_y_sign(rest: np.ndarray) -> float:
    """+1 if source Y grows downward like the mesh, -1 if the source is Y-up."""
    if len(rest) <= 10:
        return 1.0
    brow_y = float(np.mean(rest[5:11, 1]))
    chin_y = float(rest[2, 1])
    return 1.0 if brow_y < chin_y else -1.0


def _source_x_sign(xy: np.ndarray) -> float:
    """+1 if source X grows to image-right (OSF 0 left of 16), -1 if reversed."""
    if len(xy) <= 16:
        return 1.0
    return 1.0 if float(xy[0, 0]) <= float(xy[16, 0]) else -1.0


def _swap_rows(arr: np.ndarray | None) -> np.ndarray | None:
    """Swap L/R slot rows of a per-slot (len(_SLOTS), 2) array."""
    if arr is None:
        return None
    out = arr.copy()
    for i, slot in enumerate(_SLOTS):
        j = _SLOT_INDEX.get(mirror_slot(slot))
        if j is not None:
            out[i] = arr[j]
    return out


def _span(pts: np.ndarray) -> float:
    if len(pts) > 4:
        span = float(np.linalg.norm(pts[4, :2] - pts[0, :2]))
        if span > 1.0:
            return span
    return 1.0


def _aabb(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    xy = np.asarray(pts, dtype=np.float32)[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    size = np.maximum(hi - lo, 1e-6)
    return 0.5 * (lo + hi), float(size[0]), float(size[1])


def _scale_mouth(mesh: np.ndarray, gain: float) -> None:
    """Grow or shrink every mouth slot around the mouth's own center."""
    slots = [slot for slot in _MOUTH_SLOTS if slot < len(mesh)]
    if len(slots) < 2:
        return
    gain = float(gain)
    if abs(gain - 1.0) < 1e-4:
        return
    xy = mesh[slots, :2]
    center = np.mean(xy, axis=0)
    mesh[slots, :2] = center + (xy - center) * gain


def dest_mouth_width(face: np.ndarray) -> float:
    jaw = _span(face)
    if len(face) > 16:
        nostril = float(np.linalg.norm(face[16, :2] - face[14, :2]))
        if nostril > _NOSTRIL_MIN * max(jaw, 1.0):
            return max(nostril * _NOSTRIL_TO_MOUTH, 1e-3)
    return max(jaw * _JAW_MOUTH, 1e-3)


def dest_mouth_center(face: np.ndarray) -> np.ndarray:
    nose = np.asarray(face[15, :2], dtype=np.float32)
    chin = np.asarray(face[2, :2], dtype=np.float32) if len(face) > 2 else nose
    center = nose + _MOUTH_ALONG * (chin - nose)
    center[0] = nose[0]
    return center


def dest_mouth_cage(face: np.ndarray) -> tuple[float, float, float, float]:
    """Left, top, right, bottom from mid-nose, chin, and jaw sides."""
    left = float(min(face[0, 0], face[4, 0]))
    right = float(max(face[0, 0], face[4, 0]))
    nose_y = float(face[15, 1])
    chin_y = float(face[2, 1]) if len(face) > 2 else nose_y + 1.0
    top = min(nose_y, chin_y)
    bottom = max(nose_y, chin_y)
    span = max(bottom - top, 1.0)
    pad_x = _CAGE_PAD * max(right - left, 1.0)
    pad_y = _CAGE_PAD * span
    # Extra room below the rest chin so jawOpen is not clipped to rest lips.
    return left + pad_x, top + pad_y, right - pad_x, bottom - pad_y + 0.22 * span


def _clip_to_box(
    xy: np.ndarray,
    dest_c: np.ndarray,
    dest_w: float,
    dest_h: float,
) -> np.ndarray:
    """Lips may move inside the box. They are not allowed to leave it."""
    return np.array(
        [
            float(np.clip(xy[0], dest_c[0] - 0.5 * dest_w, dest_c[0] + 0.5 * dest_w)),
            float(np.clip(xy[1], dest_c[1] - 0.5 * dest_h, dest_c[1] + 0.5 * dest_h)),
        ],
        dtype=np.float32,
    )


def fit_mouth_box(
    face: np.ndarray,
    src_w: float,
    src_h: float,
    natural_w: float,
    rest: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Automatically fit camera lips to the character's authored Rest mouth."""
    left, top, right, bottom = dest_mouth_cage(face)
    cage_w = max(right - left, 1e-3)
    cage_h = max(bottom - top, 1e-3)
    aspect = src_h / src_w if src_w > 1e-5 and src_h > 1e-5 else _BOX_ASPECT
    anchor = np.asarray(rest if rest is not None else face, dtype=np.float32)
    mouth = anchor[list(_MOUTH_SLOTS), :2] if len(anchor) > max(_MOUTH_SLOTS) else np.empty((0, 2))
    if len(mouth) and np.all(np.isfinite(mouth)):
        lo = mouth.min(axis=0)
        hi = mouth.max(axis=0)
        base_w = max(float(hi[0] - lo[0]), natural_w * 0.35, 1e-3)
        base_h = max(float(hi[1] - lo[1]), base_w * _BOX_ASPECT, 1e-3)
        dest_c = 0.5 * (lo + hi)
    else:
        base_w = max(natural_w, 1e-3)
        base_h = base_w * _BOX_ASPECT
        dest_c = dest_mouth_center(face)
    wanted_w = max(base_w, 1e-3)
    wanted_h = wanted_w * aspect
    # Fit the authored rest mouth. Overall size is applied later by
    # expanding every lip around the mouth center.
    wanted_h = max(wanted_h, base_h)
    fit_w = min(1.0, cage_w / wanted_w)
    dest_w = max(wanted_w * fit_w, 1e-3)
    dest_h = min(max(wanted_h, 1e-3), cage_h)
    dest_c[0] = float(np.clip(dest_c[0], left + 0.5 * dest_w, right - 0.5 * dest_w))
    dest_c[1] = float(np.clip(dest_c[1], top + 0.5 * dest_h, bottom - 0.5 * dest_h))
    scale = dest_w / max(src_w, 1e-6)
    return dest_c, dest_w, dest_h, scale


def _keep_inside_cage(
    face: np.ndarray,
    dest_c: np.ndarray,
    placed: list[np.ndarray],
    scale: float,
) -> float:
    if not placed:
        return scale
    left, top, right, bottom = dest_mouth_cage(face)
    xs = [float(p[0]) for p in placed]
    ys = [float(p[1]) for p in placed]
    extra = 1.0
    max_x, min_x = max(xs), min(xs)
    max_y, min_y = max(ys), min(ys)
    if max_x > dest_c[0] and max_x > right:
        extra = min(extra, (right - dest_c[0]) / (max_x - dest_c[0]))
    if min_x < dest_c[0] and min_x < left:
        extra = min(extra, (dest_c[0] - left) / (dest_c[0] - min_x))
    if max_y > dest_c[1] and max_y > bottom:
        extra = min(extra, (bottom - dest_c[1]) / (max_y - dest_c[1]))
    if min_y < dest_c[1] and min_y < top:
        extra = min(extra, (dest_c[1] - top) / (dest_c[1] - min_y))
    return scale * max(float(extra), 0.05)


class FaceExpr:
    def __init__(self) -> None:
        self._selfie = False
        self.reset()

    @property
    def selfie(self) -> bool:
        return self._selfie

    def set_selfie(self, selfie: bool) -> None:
        """Flip which camera side feeds which slot. Rest stays locked.

        Rest / live are per-slot samples, so mirroring the source table is
        the same as swapping L/R rows. Nothing is re-captured, so the pose
        reflects at once instead of re-centering.
        """
        flag = bool(selfie)
        if flag == self._selfie:
            return
        self._selfie = flag
        self._sources = _face_sources(flag)
        self._psources = _face_sources(flag)
        if self.locked:
            self._rest = _swap_rows(self._rest)
            self._live = _swap_rows(self._live)
            self._prest = _swap_rows(self._prest)
            self._plive = _swap_rows(self._plive)

    def reset(self) -> None:
        self.locked = False
        self._token: object = None
        self._rest: np.ndarray | None = None
        self._osf_span = 1.0
        self._live: np.ndarray | None = None
        self._mrest = np.zeros((66, 2), dtype=np.float32)
        self._mlive = np.zeros((66, 2), dtype=np.float32)
        self._erest = np.zeros((68, 2), dtype=np.float32)
        self._elive = np.zeros((68, 2), dtype=np.float32)
        self._mouth_uses_image_y = False
        self._uses_image = False
        self._sources = _face_sources(self._selfie)
        self._y_sign = 1.0
        self._x_sign = 1.0
        self._face_scale = 1.0
        # Pose-free (PnP head-local) brow/eye frame. A real head turn does
        # not move these, so yaw is applied once, by the head rig.
        self._psources = _face_sources(self._selfie)
        self._prest: np.ndarray | None = None
        self._plive: np.ndarray | None = None
        self._p_span = 1.0
        self._p_ysign = 1.0
        self._p_xsign = 1.0
        self._p_face_scale = 1.0
        self._gain = 1.0
        self._cap = 1.0
        self._src_center = np.zeros(2, dtype=np.float32)
        self._src_w = 1.0
        self._src_h = 1.0
        self._box_ids: frozenset[int] = frozenset()
        self._dest_box: tuple[float, float, float, float] | None = None
        self._cage: tuple[float, float, float, float] | None = None
        self._held: np.ndarray | None = None
        self._box_w: float | None = None
        self._box_h: float | None = None
        self._maps_key: object = None
        self._remap_t = 1.0

    def _lock(self, pts3: np.ndarray, mouth_pts: np.ndarray | None = None) -> None:
        xy, uses_image = _source_xy(pts3, mouth_pts)
        self._uses_image = uses_image
        self._sources = _face_sources(self._selfie)
        rest = np.zeros((len(self._sources), 2), dtype=np.float32)
        for i, (_slot, src) in enumerate(self._sources):
            rest[i] = _sample(xy, src)
        self._rest = rest
        # Image landmarks already share the mesh axes. Only 3D may be flipped.
        self._y_sign = 1.0 if uses_image else _source_y_sign(rest)
        self._x_sign = 1.0 if uses_image else _source_x_sign(xy)
        self._osf_span = max(float(np.linalg.norm(rest[4] - rest[0])), 1e-3)
        self._live = rest.copy()
        self._mrest[:] = 0
        mouth_xy = (
            np.asarray(mouth_pts, dtype=np.float32)[:, :2]
            if mouth_pts is not None
            else xy
        )
        self._mouth_uses_image_y = uses_image
        for i in MOUTH_ALL:
            if i < len(mouth_xy):
                self._mrest[i] = mouth_xy[i]
        self._mlive = self._mrest.copy()
        p3 = np.asarray(pts3, dtype=np.float32)[:, :2]
        self._psources = _face_sources(self._selfie)
        prest = np.zeros((len(self._psources), 2), dtype=np.float32)
        for i, (_slot, src) in enumerate(self._psources):
            prest[i] = _sample(p3, src)
        self._prest = prest
        self._plive = prest.copy()
        self._p_span = max(float(np.linalg.norm(prest[4] - prest[0])), 1e-3)
        self._p_ysign = _source_y_sign(prest)
        self._p_xsign = _source_x_sign(p3)
        self._erest = _eye_sources(p3)
        self._elive = self._erest.copy()
        self._box_ids = frozenset()
        self._dest_box = None
        self._box_w = None
        self._box_h = None
        self.locked = True
        self._token = rest_stamp()

    def place_brows(
        self,
        mesh: np.ndarray | None,
        rest: np.ndarray | None,
        face_rig: object | None = None,
    ) -> np.ndarray | None:
        """Sit brows on the current eyes using camera brow-to-eye motion.

        Head pitch also orbits the brows and often reads a frown as a nod,
        which lifts them. Placing after the rig, relative to the eyes, keeps
        camera-down as character-down. When the head rig is present, the rest
        brow offset is projected with it so a turn does not flatten the brows.
        """
        if mesh is None or rest is None or self._prest is None or self._plive is None:
            return mesh
        if not _EYE_IDX or not _BROW_IDX:
            return mesh
        src_live_eye = np.mean(self._plive[list(_EYE_IDX)], axis=0)
        src_rest_eye = np.mean(self._prest[list(_EYE_IDX)], axis=0)
        dest_slots = [slot for slot in _EYE_SLOTS if slot < len(mesh) and slot < len(rest)]
        if len(dest_slots) < 2:
            return mesh
        dest_eye = np.mean(mesh[dest_slots, :2], axis=0)
        rest_eye = np.mean(rest[dest_slots, :2], axis=0)
        scale = self._p_face_scale * self._gain
        cap = self._cap
        y_sign = self._p_ysign
        sx = self._p_xsign * x_sign(self._selfie)
        project = getattr(face_rig, "_project", None) if face_rig is not None else None
        rest_cx = float(getattr(face_rig, "_rest_cx", 0.0) or 0.0)
        rest_cy = float(getattr(face_rig, "_rest_cy", 0.0) or 0.0)
        for i in _BROW_IDX:
            slot = _SLOTS[i]
            if slot >= len(mesh) or slot >= len(rest):
                continue
            src = (self._plive[i] - src_live_eye) - (self._prest[i] - src_rest_eye)
            dx = float(np.clip(sx * src[0] * scale, -cap, cap))
            dy = float(np.clip(y_sign * src[1] * scale, -cap, cap))
            if callable(project) and getattr(face_rig, "locked", False):
                xs, ys = project(
                    np.array(
                        [float(rest[slot, 0] - rest_cx), float(rest_eye[0] - rest_cx)],
                        dtype=np.float64,
                    ),
                    np.array(
                        [float(rest[slot, 1] - rest_cy), float(rest_eye[1] - rest_cy)],
                        dtype=np.float64,
                    ),
                )
                base = np.array(
                    [float(xs[0] - xs[1]), float(ys[0] - ys[1])], dtype=np.float32
                )
            else:
                base = rest[slot, :2] - rest_eye
            mesh[slot, 0] = float(dest_eye[0] + base[0] + dx)
            mesh[slot, 1] = float(dest_eye[1] + base[1] + dy)
        return mesh

    def mouth_box(self) -> list[float] | None:
        if self._dest_box is None:
            return None
        x, y, w, h = self._dest_box
        return [round(float(x), 2), round(float(y), 2), round(float(w), 2), round(float(h), 2)]

    def mouth_cage(self) -> list[float] | None:
        if self._cage is None:
            return None
        return [round(float(v), 2) for v in self._cage]

    def _src_gap(self, arr: np.ndarray, upper: int, lower: int) -> float:
        if len(arr) <= max(upper, lower):
            return 0.0
        return abs(float(arr[upper, 1] - arr[lower, 1]))

    def _camera_open(self) -> float:
        """Live lip split vs the rest-locked camera ring. 0 = shut, 1 = wide."""
        if not self.locked:
            return 0.0
        rest_w = max(
            float(self._src_w),
            abs(float(self._mrest[58, 0] - self._mrest[62, 0])),
            1e-6,
        )
        rest_g = max(self._src_gap(self._mrest, 60, 64), self._src_gap(self._mrest, 51, 57))
        live_g = max(self._src_gap(self._mlive, 60, 64), self._src_gap(self._mlive, 51, 57))
        delta = (live_g - rest_g) / rest_w
        if delta <= 0.02:
            return 0.0
        t = (delta - 0.02) / 0.14
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return t * t * (3.0 - 2.0 * t)

    def _lock_mouth_box(self, ids: tuple[int, ...]) -> None:
        src = np.array([self._mrest[i] for i in ids], dtype=np.float32)
        center, src_w, src_h = _aabb(src)
        self._src_center = center
        self._src_w = max(src_w, 1e-6)
        self._src_h = src_h
        self._box_ids = frozenset(ids)

    def apply(
        self,
        mixed: np.ndarray | None,
        rest: np.ndarray | None,
        pts3: np.ndarray | None,
        blink: dict[str, float] | None = None,
        mouth_pts: np.ndarray | None = None,
        keep_mouth: bool = False,
    ) -> np.ndarray | None:
        if mixed is None or rest is None:
            return mixed
        if pts3 is None:
            return self._held if self._held is not None else mixed
        arr = np.asarray(pts3, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < _NEED or arr.shape[1] < 2:
            return self._held if self._held is not None else mixed
        stamp = rest_stamp()
        _, uses_image = _source_xy(arr, mouth_pts)
        if not self.locked:
            self._lock(arr, mouth_pts)
        elif uses_image != self._uses_image:
            self._lock(arr, mouth_pts)
        elif stamp != self._token and session_rest_locked():
            self._lock(arr, mouth_pts)
        if self._rest is None or self._live is None:
            return mixed
        xy, _uses_image = _source_xy(arr, mouth_pts)
        nxt = np.zeros_like(self._rest)
        for i, (_slot, src) in enumerate(self._sources):
            nxt[i] = _sample(xy, src)
        alpha = max(feel.alpha(), 0.22)
        self._live = self._live + alpha * (nxt - self._live)
        p3 = arr[:, :2]
        if self._prest is None or self._plive is None:
            self._lock(arr, mouth_pts)
        pnxt = np.zeros_like(self._prest)
        for i, (_slot, src) in enumerate(self._psources):
            pnxt[i] = _sample(p3, src)
        self._plive = self._plive + alpha * (pnxt - self._plive)
        mouth_xy = (
            np.asarray(mouth_pts, dtype=np.float32)[:, :2]
            if mouth_pts is not None and len(mouth_pts) >= _NEED
            else arr[:, :2]
        )
        for i in MOUTH_ALL:
            if i < len(mouth_xy):
                self._mlive[i] = mouth_xy[i]
        self._elive = _eye_sources(p3)
        gain = feel.response()
        mesh_span = _span(rest)
        face_scale = mesh_span / self._osf_span
        p_face_scale = mesh_span / self._p_span
        cap = _CLIP * mesh_span
        self._face_scale = face_scale
        self._p_face_scale = p_face_scale
        self._gain = gain
        self._cap = cap
        out = mixed.copy()
        try:
            nose_i = _SLOTS.index(15)
            nose_d = (self._live[nose_i] - self._rest[nose_i]) * face_scale * gain
            p_nose_d = (self._plive[nose_i] - self._prest[nose_i]) * p_face_scale * gain
        except ValueError:
            nose_d = np.zeros(2, dtype=np.float32)
            p_nose_d = np.zeros(2, dtype=np.float32)
        selfie = self._selfie
        sx = x_sign(selfie)
        eye_maps = eye_bits.maps()
        mapped_eyes = {
            slot
            for osf_i, slot in eye_maps.items()
            if eye_bits.on(osf_i) and slot < len(out)
        }
        y_sign = self._y_sign
        p_ysign = self._p_ysign
        x_img = self._x_sign * sx
        x_p3 = self._p_xsign * sx
        for i, slot in enumerate(_SLOTS):
            if slot in _JAW or slot in _BROW_SLOTS or slot in mapped_eyes:
                continue
            if slot in _EYE_SLOTS:
                delta = (self._plive[i] - self._prest[i]) * p_face_scale * gain - p_nose_d
                dx = float(np.clip(x_p3 * delta[0], -cap, cap))
                dy = float(np.clip(p_ysign * delta[1], -cap, cap))
            else:
                delta = (self._live[i] - self._rest[i]) * face_scale * gain - nose_d
                dx = float(np.clip(x_img * delta[0], -cap, cap))
                dy = float(np.clip(y_sign * delta[1], -cap, cap))
            out[slot, 0] = float(rest[slot, 0]) + dx
            out[slot, 1] = float(rest[slot, 1]) + dy
        self.place_brows(out, rest)
        # Mouth / eye maps are edited in the canonical camera frame; the
        # selfie rule swaps which camera lip feeds each slot.
        mouth_maps = mouth_bits.maps()
        mouth_on = {osf_i for osf_i in mouth_maps if mouth_bits.on(osf_i)}
        if selfie:
            mouth_maps = mirror_map(mouth_maps)
            mouth_on = {mirror_osf(i) for i in mouth_on}
        pairs = [
            (osf_i, slot)
            for osf_i, slot in mouth_maps.items()
            if osf_i in mouth_on and slot < len(out)
        ]
        if keep_mouth:
            dest_c, dest_w, dest_h, _scale = fit_mouth_box(
                out,
                self._src_w,
                self._src_h,
                dest_mouth_width(rest),
                rest,
            )
            slots = [slot for slot in _MOUTH_SLOTS if slot < len(out)]
            if slots:
                # Authored A sits below the rest slit. A rest-height box
                # would clip it back shut.
                mix = out[slots, :2]
                lo = mix.min(axis=0)
                hi = mix.max(axis=0)
                dest_c = 0.5 * (lo + hi)
                dest_w = max(float(hi[0] - lo[0]), dest_w, 1e-3)
                dest_h = max(float(hi[1] - lo[1]), dest_h, 1e-3)
                left, top, right, bottom = dest_mouth_cage(out)
                dest_w = min(dest_w, max(right - left, 1e-3))
                dest_h = min(dest_h, max(bottom - top, 1e-3))
                dest_c[0] = float(
                    np.clip(dest_c[0], left + 0.5 * dest_w, right - 0.5 * dest_w)
                )
                dest_c[1] = float(
                    np.clip(dest_c[1], top + 0.5 * dest_h, bottom - 0.5 * dest_h)
                )
            for slot in _MOUTH_SLOTS:
                if slot < len(out):
                    xy = _clip_to_box(out[slot, :2], dest_c, dest_w, dest_h)
                    out[slot, 0] = float(xy[0])
                    out[slot, 1] = float(xy[1])
            self._dest_box = (
                float(dest_c[0] - 0.5 * dest_w),
                float(dest_c[1] - 0.5 * dest_h),
                float(dest_w),
                float(dest_h),
            )
            self._cage = dest_mouth_cage(out)
            apply_open_offset(out, rest, self._camera_open())
            left, top, right, bottom = dest_mouth_cage(out)
            for slot in _MOUTH_SLOTS:
                if slot < len(out):
                    out[slot, 0] = float(np.clip(out[slot, 0], left, right))
                    out[slot, 1] = float(np.clip(out[slot, 1], top, bottom))
        elif pairs:
            if self._mouth_uses_image_y and len(out) > 16:
                # Wrap camera lips in a rest-locked box. Dest comes from the
                # live nose / nostrils / chin, never from anime mouth slots.
                ids = tuple(osf_i for osf_i, _slot in pairs)
                if not self._box_ids:
                    self._lock_mouth_box(ids)
                else:
                    # Keep the rest AABB. Relocking when a lip is toggled
                    # rescales every mouth point and reads as a snap.
                    self._box_ids = frozenset(ids)
                live_src = np.array([self._mlive[i] for i in ids], dtype=np.float32)
                dest_c, dest_w, dest_h, scale = fit_mouth_box(
                    out,
                    self._src_w,
                    self._src_h,
                    dest_mouth_width(rest),
                    rest,
                )
                # Cover live lips from the rest-locked center. An open jaw is
                # not centered on the rest AABB, so dest_h must grow down.
                max_dx = float(np.max(np.abs(live_src[:, 0] - self._src_center[0])))
                max_dy = float(np.max(np.abs(live_src[:, 1] - self._src_center[1])))
                dest_w = max(dest_w, 2.0 * max_dx * scale)
                dest_h = max(dest_h, 2.0 * max_dy * scale)
                left, top, right, bottom = dest_mouth_cage(out)
                dest_w = min(dest_w, max(right - left, 1e-3))
                dest_h = min(dest_h, max(bottom - top, 1e-3))
                dest_c[0] = float(
                    np.clip(dest_c[0], left + 0.5 * dest_w, right - 0.5 * dest_w)
                )
                dest_c[1] = float(
                    np.clip(dest_c[1], top + 0.5 * dest_h, bottom - 0.5 * dest_h)
                )
                if self._box_w is not None and dest_w < self._box_w:
                    dest_w = self._box_w + 0.22 * (dest_w - self._box_w)
                if self._box_h is not None and dest_h < self._box_h:
                    dest_h = self._box_h + 0.22 * (dest_h - self._box_h)
                self._box_w = dest_w
                self._box_h = dest_h
                flip = np.array([sx, 1.0], dtype=np.float32)
                placed = [
                    dest_c + (self._mlive[osf_i] - self._src_center) * flip * scale
                    for osf_i, _slot in pairs
                ]
                scale = _keep_inside_cage(out, dest_c, placed, scale)
                for osf_i, slot in pairs:
                    xy = dest_c + (self._mlive[osf_i] - self._src_center) * flip * scale
                    xy = _clip_to_box(xy, dest_c, dest_w, dest_h)
                    out[slot, 0] = float(xy[0])
                    out[slot, 1] = float(xy[1])
                self._dest_box = (
                    float(dest_c[0] - 0.5 * dest_w),
                    float(dest_c[1] - 0.5 * dest_h),
                    float(dest_w),
                    float(dest_h),
                )
                self._cage = dest_mouth_cage(out)
            else:
                # Legacy 3D input has a different axis convention and is kept
                # only as a fallback when image landmarks are unavailable.
                src = np.array([self._mrest[i] for i, _ in pairs], dtype=np.float32)
                dst = rest[[slot for _i, slot in pairs], :2]
                osf_w = float(np.ptp(src[:, 0]))
                anime_w = float(np.ptp(dst[:, 0]))
                fallback_scale = (
                    anime_w / osf_w
                    if osf_w > 1e-5 and anime_w > 1e-5
                    else 1.0
                )
                for osf_i, slot in pairs:
                    d = self._mlive[osf_i] - self._mrest[osf_i]
                    out[slot, 0] = float(rest[slot, 0]) + sx * float(d[0]) * fallback_scale
                    out[slot, 1] = float(rest[slot, 1]) - float(d[1]) * fallback_scale
        mouth_scale = feel.mouth_gain()
        _scale_mouth(out, mouth_scale)
        if self._dest_box is not None and abs(mouth_scale - 1.0) >= 1e-4:
            x, y, w, h = self._dest_box
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            w *= mouth_scale
            h *= mouth_scale
            self._dest_box = (cx - 0.5 * w, cy - 0.5 * h, w, h)
        eye_on = {osf_i for osf_i in eye_maps if eye_bits.on(osf_i)}
        if selfie:
            eye_maps = mirror_map(eye_maps)
            eye_on = {mirror_osf(i) for i in eye_on}
        mapped_eyes = _apply_eye_maps(
            out,
            rest,
            self._erest,
            self._elive,
            maps={osf_i: slot for osf_i, slot in eye_maps.items() if osf_i in eye_on},
            scale=p_face_scale,
            gain=gain,
            nose_d=p_nose_d,
            cap=cap,
            y_sign=p_ysign,
            x_sign=x_p3,
        )
        if blink:
            _close_eyes(out, rest, blink, skip=mapped_eyes)
        maps_key = (
            tuple(sorted(mouth_bits.maps().items())),
            tuple(sorted(eye_bits.maps().items())),
            mouth_bits.showing(),
            eye_bits.showing(),
        )
        if maps_key != self._maps_key:
            self._maps_key = maps_key
            self._remap_t = 1.0 if self._held is None else 0.0
        if self._held is not None and self._remap_t < 1.0:
            t = _ease(self._remap_t)
            mixed_xy = (1.0 - t) * self._held[:, :2] + t * out[:, :2]
            out = out.copy()
            out[:, :2] = mixed_xy
            self._remap_t = min(1.0, self._remap_t + _REMAP_STEP)
        self._held = out
        return out


def _eye_sources(xy: np.ndarray) -> np.ndarray:
    """OSF 3D/2D landmarks plus lid mids, same frame the brows use."""
    out = np.zeros((68, 2), dtype=np.float32)
    n = min(len(xy), 66)
    out[:n] = np.asarray(xy[:n, :2], dtype=np.float32)
    if n > 38:
        out[66] = 0.5 * (out[37] + out[38])
    if n > 44:
        out[67] = 0.5 * (out[43] + out[44])
    return out


def _dest_eye(slot: int) -> tuple[int, ...]:
    if slot in _LEFT_EYE:
        return _LEFT_EYE
    if slot in _RIGHT_EYE:
        return _RIGHT_EYE
    return _LEFT_EYE if slot <= 13 else _RIGHT_EYE


def _apply_eye_maps(
    out: np.ndarray,
    rest: np.ndarray,
    erest: np.ndarray,
    elive: np.ndarray,
    *,
    scale: float,
    gain: float,
    nose_d: np.ndarray,
    cap: float,
    y_sign: float = 1.0,
    x_sign: float = 1.0,
    maps: dict[int, int] | None = None,
) -> set[int]:
    """Drive mapped eye slots from the same camera frame as the brows."""
    if maps is None:
        maps = {osf_i: slot for osf_i, slot in eye_bits.maps().items() if eye_bits.on(osf_i)}
    pairs = [
        (osf_i, slot)
        for osf_i, slot in maps.items()
        if slot < len(out) and osf_i < len(erest)
    ]
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for osf_i, slot in pairs:
        groups.setdefault(_dest_eye(slot), []).append((osf_i, slot))
    used: set[int] = set()
    for dest_slots, group in groups.items():
        dest_c, dest_w, dest_h = _aabb(rest[list(dest_slots)])
        dest_h = max(dest_h, dest_w * 0.35)
        dest_w = max(dest_w, dest_h * 0.80)
        box_w = dest_w * 2.2
        box_h = dest_h * 2.4
        for osf_i, slot in group:
            delta = (elive[osf_i] - erest[osf_i]) * scale * gain - nose_d
            xy = rest[slot, :2] + np.array(
                [
                    float(np.clip(x_sign * delta[0], -cap, cap)),
                    float(np.clip(y_sign * delta[1], -cap, cap)),
                ],
                dtype=np.float32,
            )
            xy = _clip_to_box(xy, dest_c, box_w, box_h)
            out[slot, 0] = float(xy[0])
            out[slot, 1] = float(xy[1])
            used.add(slot)
    return used


def _close_eyes(
    pts: np.ndarray,
    rest: np.ndarray,
    blink: dict[str, float],
    skip: set[int] | None = None,
) -> None:
    skip = skip or set()
    for slots, key in ((_LEFT_EYE, "l"), (_RIGHT_EYE, "r")):
        amount = float(np.clip(float(blink.get(key, 0.0)), 0.0, 1.0))
        if amount < 0.03:
            continue
        lid = slots[1]
        corners = [i for i in (slots[0], slots[2]) if i < len(pts)]
        if not corners and lid >= len(rest):
            continue
        chord_y = (
            float(np.mean(pts[corners, 1])) if corners else float(rest[lid, 1])
        )
        # Lid mid always follows blink so a wink slams even when maps drove corners.
        # Only drop toward the aperture; never lift a lid that maps already shut.
        if lid < len(pts) and float(pts[lid, 1]) < chord_y:
            pts[lid, 1] = float(pts[lid, 1]) * (1.0 - amount) + chord_y * amount
        for i in corners:
            if i in skip:
                continue
            pts[i, 1] = float(pts[i, 1]) * (1.0 - amount) + chord_y * amount
