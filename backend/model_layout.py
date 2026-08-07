"""Adapt project KEYPOINT_SCHEMA poses to the legacy HRNet model layout.

The current DiT checkpoint was trained on historical train_crop labels whose
28 face rows still used the anime-HRNet native ordering.  Tracking and
retargeting use the corrected project schema, so conversion belongs only at
the model-conditioning boundary.
"""

from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 37
KEYPOINT_DIM = 4


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
