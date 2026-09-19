from __future__ import annotations

from backend.retarget import FACE_SOURCES
from backend.mouth_bits import DEFAULT_TO
from harness.pack import pack_keypoints
from harness.points import (
    BY_ID,
    BY_REF,
    CAM_BY_ID,
    IRIS_L,
    IRIS_R,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    POINTS,
    RIGHT_IRIS,
    cam_ref_of,
    ref_of,
    row_meta,
)


def test_catalog_covers_schema() -> None:
    assert len(POINTS) == NUM_KEYPOINTS == 37
    assert [p.id for p in POINTS] == list(range(37))
    assert len(BY_REF) == 37
    assert len(set(KEYPOINT_NAMES)) == 37


def test_iris_refs_match_the_eyes_they_sit_in() -> None:
    assert IRIS_L == 28 == RIGHT_IRIS
    assert IRIS_R == 29
    assert ref_of(28) == "IRIS.L"
    assert ref_of(29) == "IRIS.R"
    assert BY_ID[28].legacy == "right_iris"
    assert BY_ID[29].legacy == "left_iris"
    assert BY_ID[11].side == BY_ID[28].side == "l"
    assert BY_ID[17].side == BY_ID[29].side == "r"


def test_face_sources_match_catalog_osf() -> None:
    for slot, src in FACE_SOURCES:
        assert BY_ID[slot].osf == src


def test_mouth_bits_match_catalog_osf() -> None:
    for osf_i, slot in DEFAULT_TO.items():
        assert BY_ID[slot].osf == (osf_i,)


def test_pack_rows_include_ref() -> None:
    face = [[100.0 + i, 200.0, 1.0] for i in range(28)]
    k = pack_keypoints(face, [], [{"id": 28, "x": 10.0, "y": 20.0, "score": 1.0, "visible": True}])
    assert abs(float(k[28, 0]) - 10.0) < 1e-6
    meta = row_meta(28)
    assert meta["ref"] == "IRIS.L"
    assert meta["name"] == "iris_l"


def test_cam_refs() -> None:
    assert cam_ref_of(36) == "CAM.EYE.R.OUT"
    assert cam_ref_of(42) == "CAM.EYE.L.IN"
    assert cam_ref_of(67) == "CAM.LID.L"
    assert CAM_BY_ID[58].side == "r"
    assert CAM_BY_ID[62].side == "l"
