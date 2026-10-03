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


def test_strokes_leave_untouched_parts_exactly_as_they_were() -> None:
    """Every stroke re-traced all three parts at whole pixels; the parts it
    never touched crept and a thin lock wore away stroke by stroke."""
    left = {"class": "hair_left", "polygon": [[10.3, 10.7], [30.6, 11.2], [29.4, 60.8], [12.1, 58.5]]}
    # A thin lock, three pixels wide.
    right = {"class": "hair_right", "polygon": [[90.4, 5.2], [93.3, 5.6], [93.1, 70.9], [90.2, 70.4]]}
    hair = [left, right]
    for i in range(12):
        hair = stamp_hair_stroke(
            hair,
            [(50, 30 + i), (60, 32 + i)],
            radius=5,
            part="hair_middle",
            width=120,
            height=100,
            erase=bool(i % 2),
        )
    assert [seg for seg in hair if seg["class"] == "hair_left"] == [left]
    assert [seg for seg in hair if seg["class"] == "hair_right"] == [right]


def test_a_stroke_over_a_part_retraces_only_that_part() -> None:
    left = {"class": "hair_left", "polygon": [[10.3, 10.7], [30.6, 11.2], [29.4, 60.8], [12.1, 58.5]]}
    right = {"class": "hair_right", "polygon": [[80.4, 5.2], [95.3, 5.6], [95.1, 70.9], [80.2, 70.4]]}
    out = stamp_hair_stroke(
        [left, right],
        [(20, 30), (20, 40)],
        radius=4,
        part="hair_middle",
        width=120,
        height=100,
    )
    assert [seg for seg in out if seg["class"] == "hair_right"] == [right]
    new_left = [seg for seg in out if seg["class"] == "hair_left"]
    assert new_left and new_left != [left]
    assert any(seg["class"] == "hair_middle" for seg in out)
