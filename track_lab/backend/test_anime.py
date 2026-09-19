from __future__ import annotations

import numpy as np
import pytest

from backend.anime import AnimeMeshError, pick_face_box, rest_shifted, expand_tiny_head_box, rest_too_small


def test_pick_face_box_prefers_large_face_over_hair_swirl() -> None:
    hair = np.array([240.0, 250.0, 520.0, 510.0, 0.92], dtype=np.float32)
    face = np.array([430.0, 280.0, 820.0, 720.0, 0.41], dtype=np.float32)
    picked = pick_face_box([hair, face], 1254, 1254)
    np.testing.assert_allclose(picked, face)


def test_pick_face_box_ignores_tiny_high_score_when_a_face_exists() -> None:
    swirl = np.array([80.0, 60.0, 140.0, 120.0, 0.99], dtype=np.float32)
    face = np.array([400.0, 260.0, 860.0, 740.0, 0.22], dtype=np.float32)
    picked = pick_face_box([swirl, face], 1254, 1254)
    np.testing.assert_allclose(picked, face)


def test_pick_face_box_falls_back_when_every_hit_is_tiny() -> None:
    only = np.array([10.0, 12.0, 40.0, 50.0, 0.8], dtype=np.float32)
    picked = pick_face_box([only], 1254, 1254)
    np.testing.assert_allclose(picked, only)


def test_pick_face_box_empty_raises() -> None:
    with pytest.raises(AnimeMeshError, match="No anime face"):
        pick_face_box([], 640, 640)


def _mesh(x: float, y: float) -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    for i in range(28):
        pts[i, 0] = x + (i % 7) * 4.0
        pts[i, 1] = y + (i // 7) * 4.0
    return pts


def test_tighten_head_box_keeps_upper_face_on_bust() -> None:
    from backend.anime import tighten_head_box

    bust = np.array([319.5, 300.0, 935.2, 920.2, 0.9], dtype=np.float32)
    tight = tighten_head_box(bust, 1254, 1254)
    assert float(tight[3]) < float(bust[3])
    assert float(tight[3]) > 720.0
    assert float(tight[1]) < 400.0
    assert float(tight[0]) > float(bust[0])
    assert float(tight[2]) < float(bust[2])


def test_tighten_head_box_leaves_a_close_face_alone() -> None:
    from backend.anime import tighten_head_box

    face = np.array([430.0, 220.0, 820.0, 620.0, 0.8], dtype=np.float32)
    tight = tighten_head_box(face, 1254, 1254)
    np.testing.assert_allclose(tight[:4], face[:4], atol=0.05)


def test_expand_tiny_head_box_grows_a_bangs_hit() -> None:
    swirl = np.array([252.0, 266.0, 535.0, 517.0, 0.9], dtype=np.float32)
    grown = expand_tiny_head_box(swirl, 1254, 1254)
    need = 0.30 * 1254
    assert float(grown[2] - grown[0]) + 1e-3 >= need
    assert float(grown[3] - grown[1]) + 1e-3 >= need
    assert float(grown[1]) <= float(swirl[1]) + 1.0
    assert float(grown[3]) > float(swirl[3]) + 80.0


def test_expand_tiny_head_box_leaves_a_face_alone() -> None:
    face = np.array([430.0, 220.0, 820.0, 720.0, 0.8], dtype=np.float32)
    grown = expand_tiny_head_box(face, 1254, 1254)
    np.testing.assert_allclose(grown[:4], face[:4], atol=0.05)


def test_rest_too_small_on_a_bangs_mesh() -> None:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    pts[0, 0], pts[4, 0] = 252.0, 523.0
    pts[:, 1] = 400.0
    assert rest_too_small(pts, 1254, 1254) is True
    pts[0, 0], pts[4, 0] = 430.0, 820.0
    assert rest_too_small(pts, 1254, 1254) is False


def test_rest_shifted_when_mesh_jumps_to_the_real_face() -> None:
    hair = _mesh(300.0, 330.0)
    face = _mesh(620.0, 400.0)
    assert rest_shifted(hair, face, 1254, 1254) is True
    assert rest_shifted(face, face, 1254, 1254) is False
    jitter = _mesh(624.0, 403.0)
    assert rest_shifted(face, jitter, 1254, 1254) is False


def test_drop_anime_chin_extends_a_cropped_jaw() -> None:
    from backend.anime import drop_anime_chin

    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    pts[15] = [100.0, 100.0, 1.0]
    pts[21] = [100.0, 140.0, 1.0]
    pts[25] = [100.0, 142.0, 1.0]
    pts[2] = [100.0, 155.0, 0.2]
    pts[1] = [70.0, 148.0, 1.0]
    pts[3] = [130.0, 148.0, 1.0]
    out = drop_anime_chin(pts)
    assert float(out[2, 1]) > float(pts[25, 1]) + 50.0
    assert abs(float(out[21, 1]) - 140.0) < 1e-3
    assert float(out[1, 1]) > float(pts[1, 1])


def test_drop_anime_chin_leaves_a_deep_jaw_alone() -> None:
    from backend.anime import drop_anime_chin

    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    pts[15] = [100.0, 100.0, 1.0]
    pts[25] = [100.0, 140.0, 1.0]
    pts[2] = [100.0, 210.0, 1.0]
    out = drop_anime_chin(pts)
    assert abs(float(out[2, 1]) - 210.0) < 1e-3

