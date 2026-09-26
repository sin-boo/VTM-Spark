from __future__ import annotations

from backend.hair_edit import stamp_hair_stroke


def test_stamp_adds_hair_middle_blob() -> None:
    stamped = stamp_hair_stroke(
        [],
        [(40, 40), (55, 42), (70, 40)],
        radius=10,
        part="hair_middle",
        width=120,
        height=100,
        erase=False,
    )
    middles = [seg for seg in stamped if seg["class"] == "hair_middle"]
    assert len(middles) == 1
    assert len(middles[0]["polygon"]) >= 3
    assert middles[0]["area"] > 24


def test_erase_from_edge_removes_or_shrinks_blob() -> None:
    painted = stamp_hair_stroke(
        [],
        [(20, 20), (40, 20), (40, 40), (20, 40)],
        radius=8,
        part="hair_left",
        width=80,
        height=80,
        erase=False,
    )
    before = next(seg for seg in painted if seg["class"] == "hair_left")
    erased = stamp_hair_stroke(
        painted,
        [(0, 20), (20, 20), (40, 20)],
        radius=12,
        part="hair_left",
        width=80,
        height=80,
        erase=True,
    )
    left = [seg for seg in erased if seg["class"] == "hair_left"]
    if left:
        assert left[0]["area"] < before["area"]
    else:
        assert left == []
