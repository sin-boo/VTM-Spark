"""Adapt project KEYPOINT_SCHEMA poses to the legacy HRNet model layout.

Tracking and retargeting always use the corrected project schema. Conversion
belongs only at the DiT conditioning boundary, and only when the loaded
checkpoint was trained on historical anime-HRNet face ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

NUM_KEYPOINTS = 37
KEYPOINT_DIM = 4

LAYOUT_SCHEMA = "schema"
LAYOUT_HRNET_NATIVE = "hrnet_native"
DEFAULT_KEYPOINT_LAYOUT = LAYOUT_HRNET_NATIVE

_SCHEMA_ALIASES = frozenset(
    {"schema", "label28", "keypoint_schema", "full_stack"}
)
_NATIVE_ALIASES = frozenset({"hrnet_native", "hrnet", "native", "legacy"})
_LAYOUT_KEYS = ("keypoint_layout", "model_keypoint_layout", "label_layout")


def normalize_keypoint_layout(value: str | None) -> str | None:
    token = str(value or "").strip().lower()
    if token in _SCHEMA_ALIASES:
        return LAYOUT_SCHEMA
    if token in _NATIVE_ALIASES:
        return LAYOUT_HRNET_NATIVE
    return None


def _name_slot_is_nose(name: Any) -> bool:
    token = str(name or "").strip().lower()
    if not token:
        return False
    return token == "nose" or token.endswith("_nose")


def keypoint_layout_from_config(cfg: Mapping[str, Any] | None) -> str:
    """Resolve DiT keypoint layout from a checkpoint ``config`` dict.

    Explicit ``keypoint_layout`` (and aliases) win. Otherwise a
    ``keypoint_names[14]`` that looks like a nose means KEYPOINT_SCHEMA.
    Historical checkpoints have neither, so they stay ``hrnet_native``.
    """
    if not cfg:
        return DEFAULT_KEYPOINT_LAYOUT
    for key in _LAYOUT_KEYS:
        resolved = normalize_keypoint_layout(cfg.get(key) if hasattr(cfg, "get") else None)
        if resolved is not None:
            return resolved
    names = None
    if hasattr(cfg, "get"):
        names = cfg.get("keypoint_names") or cfg.get("KEYPOINT_NAMES")
    if names is not None:
        try:
            slot = names[14]
        except (IndexError, KeyError, TypeError):
            slot = None
        if _name_slot_is_nose(slot):
            return LAYOUT_SCHEMA
    return DEFAULT_KEYPOINT_LAYOUT


def keypoints_for_model(
    keypoints: np.ndarray,
    layout: str,
    *,
    ref: np.ndarray | None = None,
    lower_lids: Any = None,
) -> np.ndarray:
    """Map KEYPOINT_SCHEMA rows to the layout a checkpoint expects.

    ``ref`` / ``lower_lids``: see ``schema37_to_hrnet_native37``."""
    resolved = normalize_keypoint_layout(layout)
    if resolved is None:
        raise ValueError(f"Unknown model keypoint layout: {layout!r}")
    source = np.asarray(keypoints, dtype=np.float32)
    if source.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(
            f"Expected ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {source.shape}"
        )
    if resolved == LAYOUT_SCHEMA:
        return source.copy()
    return schema37_to_hrnet_native37(source, ref=ref, lower_lids=lower_lids)


def stamp_checkpoint_keypoint_layout(path: Path | str, layout: str) -> str:
    """Write ``config.keypoint_layout`` into an existing checkpoint (no retrain)."""
    resolved = normalize_keypoint_layout(layout)
    if resolved is None:
        raise ValueError(f"Unknown model keypoint layout: {layout!r}")
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(ckpt).__name__}")
    cfg = dict(ckpt.get("config") or {})
    cfg["keypoint_layout"] = resolved
    ckpt["config"] = cfg
    torch.save(ckpt, str(ckpt_path))
    return resolved


def _visible(kps: np.ndarray, index: int) -> bool:
    return float(kps[index, 3]) >= 0.5


def _set_synthesized(
    out: np.ndarray,
    index: int,
    xy: np.ndarray,
    source: np.ndarray,
    source_indices: tuple[int, ...],
) -> None:
    scores = [float(source[i, 2]) for i in source_indices]
    visible = all(_visible(source, i) for i in source_indices)
    out[index, 0:2] = np.asarray(xy, dtype=np.float32)
    out[index, 2] = min(scores) if scores else 0.0
    out[index, 3] = 1.0 if visible else 0.0
    if not visible:
        out[index, 0:2] = 0.0


def _synthesize_lower_lid(
    out: np.ndarray,
    source: np.ndarray,
    *,
    upper_indices: tuple[int, int, int],
    output_indices: tuple[int, int, int],
) -> None:
    """Reflect an upper-lid midpoint across its corner chord."""
    outer_i, upper_i, inner_i = upper_indices
    outer = source[outer_i, 0:2].astype(np.float64)
    upper = source[upper_i, 0:2].astype(np.float64)
    inner = source[inner_i, 0:2].astype(np.float64)
    chord = inner - outer
    chord_len2 = float(np.dot(chord, chord))
    if chord_len2 < 1e-10:
        lower_mid = upper.copy()
    else:
        along = float(np.dot(upper - outer, chord) / chord_len2)
        projection = outer + np.clip(along, 0.0, 1.0) * chord
        lower_mid = 2.0 * projection - upper

    lower_outer = 0.5 * (outer + lower_mid)
    lower_inner = 0.5 * (lower_mid + inner)
    sources = (outer_i, upper_i, inner_i)
    for index, xy in zip(
        output_indices,
        (lower_outer, lower_mid, lower_inner),
        strict=True,
    ):
        _set_synthesized(out, index, xy, source, sources)


# Each eye: upper lid (corner, mid, corner) in both layouts, and the native
# lower lid rows the model was trained with.
EYE_LIDS = (((11, 12, 13), (14, 15, 16)), ((17, 18, 19), (20, 21, 22)))
# A shut eye, as the training labels draw one (2,876 open/shut pairs of the
# same character): one straight line through both corners, the lower lid's
# ends on the corners, both mids at its centre, and the whole line, corners
# too, dropped this share of the way from the open upper lid mid down to the
# open lower lid mid. The labels' median is 0.6 (quartiles 0.5-0.72); on a
# tall anime eye 0.6-0.75 left a sliver of eye under the line on a turned or
# raised head, 0.8 shut it in every pose rendered.
SHUT_LINE = 0.8
# A real lid closes from the top: the upper lid mid drops toward the shut line
# over the whole blink, and the corners and lower lid wait until the eye is
# this shut, then meet it there. Moving all six from the start drew a squint
# as a pale, flattened eye; the late join keeps a heavy-lidded eye up to
# about 0.65 shut (rendered at 0.55 / 0.7 joins; 0.7 picked).
LOWER_LID_JOIN = 0.7


def _smoothstep(t: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _eye_frame(a: np.ndarray, b: np.ndarray) -> tuple | None:
    """Corner frame: origin ``a``, x along a→b, y its perpendicular (image
    y down for a left→right chord), units of the chord length."""
    a = np.asarray(a, dtype=np.float64)
    chord = b - a
    span = float(np.hypot(chord[0], chord[1]))
    if span < 1e-6:
        return None
    x = chord / span
    return a, x, np.array([-x[1], x[0]]), span


def _to_frame(p: np.ndarray, frame) -> np.ndarray:
    a, x, y, span = frame
    d = p - a
    return np.array([d @ x, d @ y]) / span


def _from_frame(uv: np.ndarray, frame) -> np.ndarray:
    a, x, y, span = frame
    return a + span * (uv[0] * x + uv[1] * y)


def lower_lid_shape(native: np.ndarray) -> list[list[list[float]]] | None:
    """Each eye's lower lid from native anime-HRNet rows (any isotropic pixel
    or crop coords), in its corner frame: ``[[u, v] x3] x2``. None if the
    detector did not find a lower lid under both eyes."""
    pts = np.asarray(native, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 23 or pts.shape[1] < 2:
        return None
    eyes: list[list[list[float]]] = []
    for (outer, _mid, inner), lower in EYE_LIDS:
        frame = _eye_frame(pts[outer, :2], pts[inner, :2])
        if frame is None:
            return None
        uv = [_to_frame(pts[i, :2], frame) for i in lower]
        if min(v for _u, v in uv) <= 0.02:
            return None
        eyes.append([[round(float(u), 4), round(float(v), 4)] for u, v in uv])
    return eyes


def valid_lower_lids(value: Any) -> list[list[list[float]]] | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (2, 3, 2) or not np.all(np.isfinite(arr)) or float(arr[..., 1].min()) <= 0.0:
        return None
    return arr.round(4).tolist()


def _lid(
    source: np.ndarray, ref: np.ndarray, upper_indices: tuple[int, int, int]
) -> tuple[tuple, np.ndarray, np.ndarray, float] | None:
    """One eye's corner frame, rest and current lid mid in it, and how shut
    it is: 0 open as at rest (or wider), 1 lid mid on its corner line."""
    outer_i, upper_i, inner_i = upper_indices
    rest_frame = _eye_frame(ref[outer_i, 0:2], ref[inner_i, 0:2])
    frame = _eye_frame(source[outer_i, 0:2], source[inner_i, 0:2])
    if rest_frame is None or frame is None:
        return None
    rest_mid = _to_frame(ref[upper_i, 0:2], rest_frame)
    if rest_mid[1] > -0.02:
        return None  # rest lid already on its corner line: nothing to scale
    mid = _to_frame(source[upper_i, 0:2], frame)
    return frame, rest_mid, mid, 1.0 - float(np.clip(mid[1] / rest_mid[1], 0.0, 1.0))


def eye_shut(keypoints: np.ndarray, ref: np.ndarray) -> float:
    """How shut the more shut eye is (0 open .. 1 shut), schema layout."""
    source = np.asarray(keypoints, dtype=np.float32)
    rest = np.asarray(ref, dtype=np.float32)
    if source.shape != (NUM_KEYPOINTS, KEYPOINT_DIM) or rest.shape != source.shape:
        return 0.0
    shut = 0.0
    for upper, _lower in EYE_LIDS:
        if not all(_visible(source, i) and _visible(rest, i) for i in upper):
            continue
        lid = _lid(source, rest, upper)
        if lid is not None:
            shut = max(shut, lid[3])
    return shut


def _place_eye(
    out: np.ndarray,
    source: np.ndarray,
    ref: np.ndarray,
    lower: np.ndarray,
    *,
    upper_indices: tuple[int, int, int],
    output_indices: tuple[int, int, int],
) -> bool:
    """The model's six-point eye from Track Lab's three: the character's own
    lower lid, the upper lid mid moved toward the ``SHUT_LINE`` line as far as
    it fell from rest toward its corner line, and the corners and lower lid
    following once the eye is ``LOWER_LID_JOIN`` shut."""
    lid = _lid(source, ref, upper_indices)
    if lid is None:
        return False
    frame, rest_mid, mid, shut = lid
    shut_v = rest_mid[1] + SHUT_LINE * (lower[1, 1] - rest_mid[1])
    late = _smoothstep((shut - LOWER_LID_JOIN) / (1.0 - LOWER_LID_JOIN))

    def toward_shut(opened, shut_u: float, amount: float) -> np.ndarray:
        opened = np.asarray(opened, dtype=np.float64)
        return _from_frame(opened + amount * (np.array([shut_u, shut_v]) - opened), frame)

    along = (0.0, 0.5, 1.0)  # corner, centre, corner on the shut line
    # An eye open as at rest or wider keeps Track Lab's upper lid as drawn.
    if shut > 0.0:
        upper = ([0.0, 0.0], [mid[0], rest_mid[1]], [1.0, 0.0])
        for index, opened, shut_u, amount in zip(upper_indices, upper, along, (late, shut, late)):
            out[index, 0:2] = toward_shut(opened, shut_u, amount)
    for index, opened, shut_u in zip(output_indices, lower, along, strict=True):
        _set_synthesized(out, index, toward_shut(opened, shut_u, late), source, upper_indices)
    return True


def schema37_to_hrnet_native37(
    keypoints: np.ndarray,
    *,
    ref: np.ndarray | None = None,
    lower_lids: Any = None,
) -> np.ndarray:
    """Convert one corrected-schema ``(37,4)`` pose to model-native layout.

    Body and iris rows stay unchanged. Face rows are rearranged to match the
    legacy anime-HRNet labels used to train the current DiT checkpoint.

    ``lower_lids`` (``lower_lid_shape``) with the rest pose ``ref`` places the
    character's own lower lids and shuts each eye the way the model learnt
    (``_place_eye``); without them each lower lid is the upper lid mirrored
    across its corner line, which on a tall eye never closes it.
    """
    source = np.asarray(keypoints, dtype=np.float32)
    if source.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(
            f"Expected ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {source.shape}"
        )

    out = np.zeros_like(source)
    copy_indices = tuple(range(0, 14)) + tuple(range(17, 20)) + tuple(range(28, 37))
    out[list(copy_indices)] = source[list(copy_indices)]

    # Native 14..16 and 20..22 are lower eyelids, despite the historical
    # training channel names calling them nose/mouth.
    shape = valid_lower_lids(lower_lids)
    rest = None if ref is None else np.asarray(ref, dtype=np.float32)
    own = shape is not None and rest is not None and rest.shape == source.shape
    for eye, (upper, lower) in enumerate(EYE_LIDS):
        # A lid point lost in this pose would read as a shut eye and drop the
        # corners still seen; that eye keeps the mirrored lid.
        placed = (
            own
            and all(_visible(rest, i) and _visible(source, i) for i in upper)
            and _place_eye(
                out,
                source,
                rest,
                np.asarray(shape[eye]),
                upper_indices=upper,
                output_indices=lower,
            )
        )
        if not placed:
            _synthesize_lower_lid(out, source, upper_indices=upper, output_indices=lower)

    # Native 23 is the single nose tip.
    out[23] = source[15]

    # Native 24..27 are mouth L corner, upper mid, R corner, lower mid.
    out[24] = source[23]
    out[25] = source[21]
    out[26] = source[26]
    out[27] = source[25]
    return out
