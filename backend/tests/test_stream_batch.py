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


def test_pack_batch2_first_call_keeps_the_warmed_batch_size() -> None:
    now = neutral_keypoints()
    now[:, 0] = 0.3
    hair = np.ones((3, 4, 4), dtype=np.float32)
    kps, maps = pack_stream_batch(now, hair, None, None, 2)
    assert kps.shape == (2, *now.shape)
    np.testing.assert_allclose(kps[0], now)
    np.testing.assert_allclose(kps[1], now)
    assert maps.shape == (2, *hair.shape)


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


def test_pack_batch_n_steps_evenly_to_now() -> None:
    prev = neutral_keypoints()
    now = prev.copy()
    prev[:, 0] = 0.0
    now[:, 0] = 0.9
    hair_prev = np.zeros((3, 4, 4), dtype=np.float32)
    hair_now = np.full((3, 4, 4), 0.9, dtype=np.float32)
    for n in (3, 4):
        kps, maps = pack_stream_batch(now, hair_now, prev, hair_prev, n)
        assert kps.shape == (n, *now.shape)
        want = [0.9 * (i + 1) / n for i in range(n)]
        np.testing.assert_allclose(kps[:, 0, 0], want, atol=1e-6)
        np.testing.assert_allclose(maps[:, 0, 0, 0], want, atol=1e-6)
        np.testing.assert_allclose(kps[-1], now)
    kps, _ = pack_stream_batch(now, hair_now, None, None, 4)
    assert kps.shape == (4, *now.shape)
