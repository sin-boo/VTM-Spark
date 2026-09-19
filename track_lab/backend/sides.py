"""One left/right rule for every tracker.

Canonical frame: the raw, unflipped camera image. A landmark on the
image-left drives the character slot that sits on the screen-left of the
still. That is an anatomical copy — the person's right eye (image-left on
a webcam) moves the character's right eye (screen-left).

``mirror`` is the only user switch. Mirror OFF is the VTuber default: the
character behaves like your reflection, so person-left lands on
screen-left. Internally that is ``selfie = not mirror`` and it means one
thing everywhere: swap each L/R pair and negate X. No per-feature flags.

iFacialMocap has no image. Its ``_L`` / ``_R`` shapes are the person's own
sides, so they are first written into the canonical frame (person-right →
image-left) and then the same selfie rule applies.
"""

from __future__ import annotations

from typing import Mapping

# Character slot pairs that swap under a horizontal flip of the still.
MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
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
    (28, 29),
    (32, 34),
    (33, 35),
)
SLOT_MIRROR: dict[int, int] = {}
for _a, _b in MIRROR_PAIRS:
    SLOT_MIRROR[_a] = _b
    SLOT_MIRROR[_b] = _a

# Slots on the screen-left of a facing still (the image-left camera side
# drives these in the canonical frame).
SCREEN_LEFT: frozenset[int] = frozenset(a for a, _b in MIRROR_PAIRS)
SCREEN_RIGHT: frozenset[int] = frozenset(b for _a, b in MIRROR_PAIRS)

# OSF 66-point (dlib minus inner-lip corners) plus lid mids 66 / 67.
_OSF_PAIRS: tuple[tuple[int, int], ...] = (
    *((i, 16 - i) for i in range(8)),
    *((17 + i, 26 - i) for i in range(5)),
    (31, 35),
    (32, 34),
    (36, 45),
    (37, 44),
    (38, 43),
    (39, 42),
    (40, 47),
    (41, 46),
    (48, 54),
    (49, 53),
    (50, 52),
    (55, 57),
    (58, 62),
    (59, 61),
    (63, 65),
    (66, 67),
)
OSF_MIRROR: dict[int, int] = {}
for _a, _b in _OSF_PAIRS:
    OSF_MIRROR[_a] = _b
    OSF_MIRROR[_b] = _a


def selfie_of(mirror: bool) -> bool:
    """Mirror OFF = reflection (selfie). Mirror ON = anatomical copy."""
    return not bool(mirror)


def mirror_slot(slot: int) -> int:
    return SLOT_MIRROR.get(int(slot), int(slot))


def mirror_osf(index: int) -> int:
    return OSF_MIRROR.get(int(index), int(index))


def mirror_sources(
    sources: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Feed each slot from the mirrored camera landmarks."""
    return tuple((slot, tuple(mirror_osf(i) for i in src)) for slot, src in sources)


def mirror_map(maps: Mapping[int, int]) -> dict[int, int]:
    """osf → slot map with the camera side swapped."""
    return {mirror_osf(i): slot for i, slot in maps.items()}


def swap_lr(values: Mapping[str, float] | None) -> dict[str, float]:
    """Swap ``l``/``r`` (and ``*_l``/``*_r``) keys."""
    if not values:
        return {}
    out: dict[str, float] = {}
    for key, value in values.items():
        if key == "l":
            out["r"] = value
        elif key == "r":
            out["l"] = value
        elif key.endswith("_l"):
            out[key[:-2] + "_r"] = value
        elif key.endswith("_r"):
            out[key[:-2] + "_l"] = value
        else:
            out[key] = value
    return out


def to_screen(values: Mapping[str, float] | None, selfie: bool) -> dict[str, float]:
    """Canonical (image-side) l/r → character screen-side l/r."""
    return swap_lr(values) if selfie else dict(values or {})


def ifm_canonical(values: Mapping[str, float] | None) -> dict[str, float]:
    """ARKit person-side l/r → canonical image-side (person-right = image-left)."""
    return swap_lr(values)


def ifm_look_canonical(look: Mapping[str, float] | None) -> dict[str, float] | None:
    """look.x > 0 is the person's right; in the raw image that is -x."""
    if not isinstance(look, Mapping):
        return None
    out = dict(look)
    try:
        out["x"] = -float(look.get("x") or 0.0)
    except (TypeError, ValueError):
        out["x"] = 0.0
    return out


def x_sign(selfie: bool) -> float:
    return -1.0 if selfie else 1.0
