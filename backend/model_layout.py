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


def keypoints_for_model(keypoints: np.ndarray, layout: str) -> np.ndarray:
    """Map KEYPOINT_SCHEMA rows to the layout a checkpoint expects."""
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
    return schema37_to_hrnet_native37(source)


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


def schema37_to_hrnet_native37(keypoints: np.ndarray) -> np.ndarray:
    """Convert one corrected-schema ``(37,4)`` pose to model-native layout.

    Body and iris rows stay unchanged. Face rows are rearranged to match the
    legacy anime-HRNet labels used to train the current DiT checkpoint.
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
    _synthesize_lower_lid(
        out,
        source,
        upper_indices=(11, 12, 13),
        output_indices=(14, 15, 16),
    )
    _synthesize_lower_lid(
        out,
        source,
        upper_indices=(17, 18, 19),
        output_indices=(20, 21, 22),
    )

    # Native 23 is the single nose tip.
    out[23] = source[15]

    # Native 24..27 are mouth L corner, upper mid, R corner, lower mid.
    out[24] = source[23]
    out[25] = source[21]
    out[26] = source[26]
    out[27] = source[25]
    return out
