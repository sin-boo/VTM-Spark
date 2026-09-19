from __future__ import annotations

from backend.offsets import apply_points, apply_rows, clear, dump, nudge, parse


def test_nudge_and_reset_roundtrip() -> None:
    offsets = nudge({}, 28, (100.0, 80.0), (112.0, 70.0))
    assert offsets[28] == (12.0, -10.0)
    rows = apply_rows(
        [{"id": 28, "x": 100.0, "y": 80.0, "score": 1.0}],
        offsets,
    )
    assert abs(float(rows[0]["x"]) - 112.0) < 1e-6
    assert abs(float(rows[0]["y"]) - 70.0) < 1e-6
    packed = dump(offsets)
    assert packed[0]["id"] == 28
    again = parse(packed)
    assert again[28] == (12.0, -10.0)
    assert clear(offsets) == {}
    assert 28 not in clear(offsets, 28)


def test_tiny_nudge_is_ignored() -> None:
    offsets = nudge({}, 21, (10.0, 10.0), (10.2, 10.1))
    assert offsets == {}


def test_absolute_mouth_plus_offset_is_double() -> None:
    """Dragging rest then applying those pixels again must not keep the nudge."""
    rest = [[float(i), 10.0, 1.0] for i in range(28)]
    rest[21] = [60.0, 80.0, 1.0]
    offsets = nudge({}, 21, (60.0, 80.0), (60.0, 96.0))
    authored = [row[:] for row in rest]
    authored[21] = [60.0, 96.0, 1.0]
    doubled = apply_points(authored, offsets)
    assert abs(float(doubled[21][1]) - 112.0) < 1e-6
    cleared = apply_points(authored, clear(offsets, 21))
    assert abs(float(cleared[21][1]) - 96.0) < 1e-6


def test_apply_points_keeps_other_slots() -> None:
    offsets = nudge({}, 2, (0.0, 0.0), (5.0, -3.0))
    pts = [[1.0, 2.0, 1.0], [3.0, 4.0, 1.0], [0.0, 0.0, 1.0]]
    out = apply_points(pts, offsets)
    assert out[0][0] == 1.0
    assert abs(float(out[2][0]) - 5.0) < 1e-6
    assert abs(float(out[2][1]) + 3.0) < 1e-6
