"""User overlay nudges. Applied after tracking so live motion still follows."""

from __future__ import annotations

from typing import Any

from harness.protocol import NUM_KEYPOINTS

_MIN = 0.5


def parse(raw: object) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    if isinstance(raw, dict):
        items = raw.items()
        for key, value in items:
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            dx = dy = None
            if isinstance(value, dict):
                dx, dy = value.get("dx"), value.get("dy")
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                dx, dy = value[0], value[1]
            if dx is None or dy is None:
                continue
            _put(out, idx, dx, dy)
        return out
    if not isinstance(raw, list):
        return out
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("id", -1))
            dx = float(row.get("dx", 0.0))
            dy = float(row.get("dy", 0.0))
        except (TypeError, ValueError):
            continue
        _put(out, idx, dx, dy)
    return out


def dump(offsets: dict[int, tuple[float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in sorted(offsets):
        dx, dy = offsets[idx]
        rows.append(
            {
                "id": int(idx),
                "dx": round(float(dx), 3),
                "dy": round(float(dy), 3),
            }
        )
    return rows


def nudge(
    offsets: dict[int, tuple[float, float]],
    idx: int,
    current: tuple[float, float],
    target: tuple[float, float],
) -> dict[int, tuple[float, float]]:
    """Set the offset so ``current + offset == target``."""
    dx = float(target[0]) - float(current[0])
    dy = float(target[1]) - float(current[1])
    return _put(dict(offsets), idx, dx, dy)


def clear(
    offsets: dict[int, tuple[float, float]],
    idx: int | None = None,
) -> dict[int, tuple[float, float]]:
    if idx is None:
        return {}
    out = dict(offsets)
    out.pop(int(idx), None)
    return out


def apply_xy(
    x: float,
    y: float,
    idx: int,
    offsets: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    delta = offsets.get(int(idx))
    if delta is None:
        return float(x), float(y)
    return float(x) + float(delta[0]), float(y) + float(delta[1])


def apply_points(points: object, offsets: dict[int, tuple[float, float]]) -> object:
    if not offsets or not isinstance(points, list):
        return points
    out: list[object] = []
    for i, row in enumerate(points):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            out.append(row)
            continue
        x, y = apply_xy(float(row[0]), float(row[1]), i, offsets)
        next_row = [x, y, *list(row[2:])]
        out.append(next_row)
    return out


def apply_rows(rows: object, offsets: dict[int, tuple[float, float]]) -> object:
    if not offsets or not isinstance(rows, list):
        return rows
    out: list[object] = []
    for row in rows:
        if not isinstance(row, dict):
            out.append(row)
            continue
        try:
            idx = int(row.get("id", -1))
        except (TypeError, ValueError):
            out.append(row)
            continue
        copied = dict(row)
        try:
            x, y = apply_xy(float(copied.get("x") or 0.0), float(copied.get("y") or 0.0), idx, offsets)
        except (TypeError, ValueError):
            out.append(copied)
            continue
        copied["x"] = round(x, 3)
        copied["y"] = round(y, 3)
        out.append(copied)
    return out


def _put(
    offsets: dict[int, tuple[float, float]],
    idx: int,
    dx: object,
    dy: object,
) -> dict[int, tuple[float, float]]:
    if idx < 0 or idx >= NUM_KEYPOINTS:
        return offsets
    try:
        ox = float(dx)
        oy = float(dy)
    except (TypeError, ValueError):
        return offsets
    if (ox * ox + oy * oy) ** 0.5 < _MIN:
        offsets.pop(idx, None)
        return offsets
    offsets[idx] = (ox, oy)
    return offsets
