"""One left/right rule, checked per source.

Canonical: image-left camera → screen-left slot. Mirror OFF (selfie) swaps
every pair and negates X, with no rest recapture.
"""

from __future__ import annotations

import numpy as np

from .sides import (
    MIRROR_PAIRS,
    OSF_MIRROR,
    SCREEN_LEFT,
    SCREEN_RIGHT,
    ifm_canonical,
    ifm_look_canonical,
    mirror_map,
    mirror_osf,
    mirror_slot,
    mirror_sources,
    selfie_of,
    swap_lr,
    to_screen,
)
from .test_visemes import _camera_from_osf, _toy_face


def test_slot_pairs_match_catalog_screen_sides() -> None:
    from harness.points import BY_ID

    for a, b in MIRROR_PAIRS:
        assert BY_ID[a].screen == "l", (a, BY_ID[a].ref)
        assert BY_ID[b].screen == "r", (b, BY_ID[b].ref)
        assert BY_ID[a].group == BY_ID[b].group
        assert BY_ID[a].role == BY_ID[b].role
    for point in BY_ID.values():
        if point.screen == "c":
            assert point.id not in SCREEN_LEFT and point.id not in SCREEN_RIGHT
            assert mirror_slot(point.id) == point.id


def test_osf_pairs_are_involutions_and_keep_midline() -> None:
    for i in range(68):
        assert mirror_osf(mirror_osf(i)) == i
    for mid in (8, 27, 28, 29, 30, 33, 51, 56, 60, 64):
        assert mid not in OSF_MIRROR
    assert mirror_osf(0) == 16
    assert mirror_osf(17) == 26
    assert mirror_osf(36) == 45
    assert mirror_osf(39) == 42
    assert mirror_osf(58) == 62
    assert mirror_osf(66) == 67


def test_canonical_face_sources_feed_screen_left_from_image_left() -> None:
    from .retarget import FACE_SOURCES

    rest, osf = _toy_face()
    for slot, src in FACE_SOURCES:
        x = float(np.mean(osf[list(src), 0]))
        if slot in SCREEN_LEFT:
            assert x < 0.0, (slot, src)
        elif slot in SCREEN_RIGHT:
            assert x > 0.0, (slot, src)
    mirrored = dict(mirror_sources(FACE_SOURCES))
    plain = dict(FACE_SOURCES)
    for slot in SCREEN_LEFT & set(plain):
        assert sorted(mirrored[slot]) == sorted(plain[mirror_slot(slot)])


def test_mirror_map_swaps_camera_side_only() -> None:
    from .eye_bits import DEFAULT_TO

    flipped = mirror_map(DEFAULT_TO)
    assert flipped[45] == DEFAULT_TO[36]
    assert flipped[42] == DEFAULT_TO[39]
    assert flipped[36] == DEFAULT_TO[45]
    assert set(flipped.values()) == set(DEFAULT_TO.values())


def test_swap_helpers() -> None:
    assert swap_lr({"l": 1.0, "r": 0.0}) == {"r": 1.0, "l": 0.0}
    assert swap_lr({"down_l": 0.2, "down_r": 0.4, "inner": 0.1}) == {
        "down_r": 0.2,
        "down_l": 0.4,
        "inner": 0.1,
    }
    assert to_screen({"l": 1.0, "r": 0.0}, selfie=False) == {"l": 1.0, "r": 0.0}
    assert to_screen({"l": 1.0, "r": 0.0}, selfie=True) == {"r": 1.0, "l": 0.0}
    assert selfie_of(False) is True and selfie_of(True) is False


def test_ifm_person_side_becomes_canonical_image_side() -> None:
    # Person's LEFT eye is on image-RIGHT of an unflipped camera.
    assert ifm_canonical({"l": 1.0, "r": 0.0}) == {"r": 1.0, "l": 0.0}
    look = ifm_look_canonical({"x": 0.6, "y": 0.2})
    assert look is not None
    assert look["x"] == -0.6 and look["y"] == 0.2
    # Then selfie puts the person's left back on screen-left.
    assert to_screen(ifm_canonical({"l": 1.0, "r": 0.0}), selfie=True) == {"l": 1.0, "r": 0.0}


def _brow_raise(selfie: bool) -> tuple[np.ndarray, np.ndarray]:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera = _camera_from_osf(osf)
    live = osf.copy()
    live[[17, 18, 19, 20, 21], 1] -= 0.25  # image-left brow up
    live_cam = _camera_from_osf(live)
    expr = FaceExpr()
    expr.set_selfie(selfie)
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=camera)
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live, {"l": 0.0, "r": 0.0}, mouth_pts=live_cam)
    finally:
        feel.update(prev)
    assert out is not None
    return rest, out


def test_image_left_brow_drives_screen_left_then_flips_with_selfie() -> None:
    rest, plain = _brow_raise(selfie=False)
    assert float(plain[6, 1]) < float(rest[6, 1]) - 1.0
    assert abs(float(plain[9, 1]) - float(rest[9, 1])) < 0.5
    rest, flipped = _brow_raise(selfie=True)
    assert float(flipped[9, 1]) < float(rest[9, 1]) - 1.0
    assert abs(float(flipped[6, 1]) - float(rest[6, 1])) < 0.5


def test_toggle_selfie_mid_session_swaps_without_relock() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera = _camera_from_osf(osf)
    live = osf.copy()
    live[[17, 18, 19, 20, 21], 1] -= 0.25
    live_cam = _camera_from_osf(live)
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=camera)
        token = expr._token
        for _ in range(8):
            plain = expr.apply(rest, rest, live, {"l": 0.0, "r": 0.0}, mouth_pts=live_cam)
        expr.set_selfie(True)
        for _ in range(8):
            flipped = expr.apply(rest, rest, live, {"l": 0.0, "r": 0.0}, mouth_pts=live_cam)
        assert expr._token == token and expr.locked
    finally:
        feel.update(prev)
    assert plain is not None and flipped is not None
    assert float(plain[6, 1]) < float(rest[6, 1]) - 1.0
    assert float(flipped[9, 1]) < float(rest[9, 1]) - 1.0
    assert abs(float(flipped[6, 1]) - float(rest[6, 1])) < 0.5


def test_osf_blink_index_zero_is_image_left() -> None:
    from .osf_cam import _blink

    class Face:
        eye_blink = [0.2, 1.0]  # OSF eye_r (36-41) mostly shut

    out = _blink(Face())
    assert out["l"] > 0.7 and out["r"] < 0.05


def test_screen_blink_closes_matching_eye_slots() -> None:
    from .retarget import _close_eyes

    rest, _osf = _toy_face()
    pts = rest.copy()
    _close_eyes(pts, rest, {"l": 1.0, "r": 0.0})
    assert abs(float(pts[12, 1]) - float(rest[11, 1])) < 0.6
    assert abs(float(pts[18, 1]) - float(rest[18, 1])) < 1e-6


def test_ifm_drive_blink_left_person_eye() -> None:
    from .ifm import drive_ifm

    rest, _osf = _toy_face()
    # Person's left eye shuts → canonical "r" → selfie puts it screen-left (slots 11-13).
    canon = ifm_canonical({"l": 1.0, "r": 0.0})
    assert canon == {"r": 1.0, "l": 0.0}
    selfie = drive_ifm(rest, {}, to_screen(canon, True), {})
    plain = drive_ifm(rest, {}, to_screen(canon, False), {})
    assert selfie is not None and plain is not None
    assert float(selfie[12, 1]) > float(rest[12, 1]) + 0.5
    assert abs(float(selfie[18, 1]) - float(rest[18, 1])) < 1e-6
    assert float(plain[18, 1]) > float(rest[18, 1]) + 0.5
    assert abs(float(plain[12, 1]) - float(rest[12, 1])) < 1e-6


def test_iris_cam_pairs_follow_rule() -> None:
    from .iris import IrisHit, LEFT_EYE_SLOTS, RIGHT_EYE_SLOTS, RIGHT_IRIS, LEFT_IRIS, _cam_pairs

    right = IrisHit(x=1.0, visible=True, side="r")
    left = IrisHit(x=2.0, visible=True, side="l")
    blink = {"l": 0.1, "r": 0.9}
    plain = _cam_pairs(right, left, blink, selfie=False)
    # image-left pupil ("r" = OSF 36-41) → screen-left iris 28 / eye 11-13
    assert plain[0][0] is right and plain[0][2] == LEFT_EYE_SLOTS and plain[0][3] == RIGHT_IRIS
    assert plain[0][4] == 0.1
    assert plain[1][0] is left and plain[1][2] == RIGHT_EYE_SLOTS and plain[1][3] == LEFT_IRIS
    flipped = _cam_pairs(right, left, blink, selfie=True)
    assert flipped[0][0] is left and flipped[0][2] == LEFT_EYE_SLOTS and flipped[0][3] == RIGHT_IRIS
    assert flipped[0][4] == 0.1
    assert flipped[1][0] is right and flipped[1][3] == LEFT_IRIS


def test_rig_selfie_negates_yaw_roll_slide_not_pitch() -> None:
    from .feel import feel
    from .rig import FaceRig

    rest, _osf = _toy_face()
    origin = {"cx": 200.0, "cy": 200.0, "bx": 200.0, "by": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    moved = {"cx": 230.0, "cy": 200.0, "bx": 230.0, "by": 200.0, "scale": 100.0, "tilt": 8.0, "ok": 1.0}
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    head = {"pitch": 12.0, "yaw": 20.0, "roll": 8.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0})
    try:
        rig = FaceRig()
        rig.apply(rest, rest, head0, origin)
        for _ in range(10):
            rig.apply(rest, rest, head, moved)
        yaw, roll, pitch, dx = rig._yaw_r, rig._roll_r, rig._pitch_r, rig._dx
        rig.selfie = True
        for _ in range(10):
            rig.apply(rest, rest, head, moved)
        assert rig.locked
        assert abs(rig._yaw_r + yaw) < 1e-4 and abs(yaw) > 0.1
        assert abs(rig._roll_r + roll) < 1e-4 and abs(roll) > 0.05
        assert abs(rig._dx + dx) < 1e-3 and abs(dx) > 1.0
        assert abs(rig._pitch_r - pitch) < 1e-4 and abs(pitch) > 0.1
    finally:
        feel.update(prev)
