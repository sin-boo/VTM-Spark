import numpy as np

from backend.engine import neutral_keypoints
from backend.stream import lerp_stream_pose, pack_stream_batch


def test_pack_batch1_is_current_only() -> None:
    now = neutral_keypoints()
    now[0, 0] = 0.4
    hair = np.zeros((3, 8, 8), dtype=np.float32)
    kps, maps = pack_stream_batch(now, hair, now, hair, 1)
    assert kps.shape == now.shape
    np.testing.assert_allclose(kps, now)
    np.testing.assert_allclose(maps, hair)


def test_pack_batch2_uses_mid_and_now_not_replay() -> None:
    prev = neutral_keypoints()
    now = prev.copy()
    now[:, 0] = 0.8
    prev[:, 0] = 0.2
    hair_now = np.ones((3, 4, 4), dtype=np.float32)
    hair_prev = np.zeros((3, 4, 4), dtype=np.float32)
    kps, maps = pack_stream_batch(now, hair_now, prev, hair_prev, 2)
    assert kps.shape == (2, *now.shape)
    np.testing.assert_allclose(kps[0, :, 0], 0.5)
    np.testing.assert_allclose(kps[1], now)
    np.testing.assert_allclose(maps[0], 0.5)
    np.testing.assert_allclose(maps[1], hair_now)


def test_lerp_keeps_parked_point_when_only_one_side_visible() -> None:
    prev = neutral_keypoints()
    now = prev.copy()
    prev[5, :2] = (0.1, 0.2)
    prev[5, 3] = 1.0
    now[5, :2] = (0.9, 0.8)
    now[5, 3] = 0.0
    out = lerp_stream_pose(prev, now, 0.5)
    np.testing.assert_allclose(out[5, :2], (0.1, 0.2))
    assert out[5, 3] >= 0.5
