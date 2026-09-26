from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.record import MovementRecorder, pack_hair, pack_pose


def _face() -> list[list[float]]:
    return [[10.0 + i, 20.0 + i, 1.0] for i in range(28)]


def test_pack_pose_matches_benchmark_norm() -> None:
    kps = pack_pose(
        _face(),
        [{"id": 31, "x": 40.0, "y": 80.0, "score": 1.0}],
        [{"id": 28, "x": 12.0, "y": 18.0, "score": 0.9, "visible": True}],
        100,
        200,
    )
    assert kps.shape == (37, 4)
    assert abs(float(kps[0, 0]) - (10.0 / 100.0 * 2.0 - 1.0)) < 1e-5
    assert abs(float(kps[0, 1]) - (20.0 / 200.0 * 2.0 - 1.0)) < 1e-5
    assert abs(float(kps[31, 0]) - (40.0 / 100.0 * 2.0 - 1.0)) < 1e-5
    assert float(kps[28, 3]) == 1.0
    shut = pack_pose(_face(), [], [{"id": 29, "x": 1, "y": 1, "visible": False}], 100, 200)
    assert float(shut[29, 3]) == 0.0


def test_pack_hair_normalizes_polygons() -> None:
    hair = pack_hair(
        [{"class": "hair_left", "polygon": [[0, 0], [50, 0], [50, 100]]}],
        100,
        200,
    )
    assert hair[0]["polygon"][1] == [0.0, -1.0]
    assert hair[0]["polygon"][2] == [0.0, 0.0]


def test_stop_writes_benchmark_take(tmp_path: Path) -> None:
    rec = MovementRecorder(tmp_path)
    still = np.zeros((40, 30, 3), np.uint8)
    still[:, :] = (10, 20, 30)
    rec.start(still, "ref.png")
    rec.note(_face(), [{"id": 36, "x": 15.0, "y": 30.0, "score": 1.0}], [], [], {"A": 0.4}, 30, 40)
    rec.note(_face(), [{"id": 36, "x": 16.0, "y": 31.0, "score": 1.0}], [], [], {"smile": 0.2}, 30, 40)
    saved = rec.stop()
    assert saved["recording"] is False
    assert saved["record_frames"] == 2
    dest = Path(str(saved["record_path"]))
    assert (dest / "base.png").is_file()
    assert (dest / "poses.npz").is_file()
    packed = np.load(dest / "poses.npz")
    assert packed["keypoints"].shape == (2, 37, 4)
    hair = json.loads((dest / "hair.json").read_text(encoding="utf-8"))
    assert len(hair) == 2
    track = json.loads((dest / "track.json").read_text(encoding="utf-8"))
    assert track["frames"] == 2
    assert track["peaks"]["A"] == 0.4
    assert track["track"][0]["weights"]["A"] == 0.4


def test_start_without_still_raises(tmp_path: Path) -> None:
    rec = MovementRecorder(tmp_path)
    try:
        rec.start(None, "")
    except ValueError as exc:
        assert "reference" in str(exc)
    else:
        raise AssertionError("expected ValueError")
