from __future__ import annotations

import numpy as np

from backend.hair_follow import (
    build_hair_rig,
    follow_hair,
    rest_hair,
)


def _rest_kps() -> np.ndarray:
    k = np.zeros((37, 4), dtype=np.float32)
    k[:, 2:] = 1.0
    k[11] = [-0.28, -0.15, 1, 1]
    k[12] = [-0.18, -0.18, 1, 1]
    k[13] = [-0.08, -0.15, 1, 1]
    k[17] = [0.08, -0.15, 1, 1]
    k[18] = [0.18, -0.18, 1, 1]
    k[19] = [0.28, -0.15, 1, 1]
    k[2] = [0.0, 0.45, 1, 1]
    return k


def test_rest_hair_ignores_live_pose() -> None:
    rest = _rest_kps()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[-0.1, -0.6], [0.1, -0.6], [0.0, -0.3]],
        }
    ]
    rig = build_hair_rig(segs, rest)
    assert rig is not None
    parked = rest_hair(rig)
    live = rest.copy()
    live[:, 0] += 0.4
    followed = follow_hair(rig, live)
    rest_xy = np.asarray(parked[0]["polygon"], dtype=np.float32)
    follow_xy = np.asarray(followed[0]["polygon"], dtype=np.float32)
    assert float(np.linalg.norm(rest_xy.mean(axis=0) - follow_xy.mean(axis=0))) > 0.2
    parked2 = rest_hair(rig)
    rest2 = np.asarray(parked2[0]["polygon"], dtype=np.float32)
    assert np.allclose(rest_xy, rest2, atol=1e-5)
