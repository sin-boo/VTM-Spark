"""Add or trim hair polygons with a brush stroke.

The hair model only emits middle / left / right. A stroke stamps those masks
so a missed lock can be painted in, or an extra blob trimmed from the edge.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

HAIR_PARTS = ("hair_middle", "hair_left", "hair_right")
_MIN_AREA = 24.0
_APPROX = 0.0015


def _points(raw: Sequence) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in raw or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            out.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return out


def _paint_stroke(layer: np.ndarray, points: Sequence[tuple[float, float]], radius: int) -> None:
    if not points:
        return
    rad = max(1, int(radius))
    first = (int(round(points[0][0])), int(round(points[0][1])))
    cv2.circle(layer, first, rad, 255, -1)
    for start, end in zip(points, points[1:]):
        a = (int(round(start[0])), int(round(start[1])))
        b = (int(round(end[0])), int(round(end[1])))
        cv2.line(layer, a, b, 255, rad * 2)
        cv2.circle(layer, b, rad, 255, -1)


def _polygons(layer: np.ndarray, cls: str) -> list[dict[str, Any]]:
    if int(layer.max()) == 0:
        return []
    height, width = layer.shape[:2]
    contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < _MIN_AREA:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(0.4, _APPROX * peri), True)
        if len(approx) < 3:
            continue
        poly = [
            [
                round(float(np.clip(point[0][0], 0, width - 1)), 1),
                round(float(np.clip(point[0][1], 0, height - 1)), 1),
            ]
            for point in approx
        ]
        out.append({"class": cls, "polygon": poly, "area": round(area, 1), "score": 1.0})
    return out


def stamp_hair_stroke(
    segments: Sequence[dict[str, Any]] | None,
    points: Sequence,
    *,
    radius: float,
    part: str,
    width: int,
    height: int,
    erase: bool = False,
) -> list[dict[str, Any]]:
    """Return hair polygons after painting ``points`` with a round brush."""
    w = int(width)
    h = int(height)
    if w < 2 or h < 2:
        return [dict(seg) for seg in (segments or []) if isinstance(seg, dict)]
    layers = {name: np.zeros((h, w), dtype=np.uint8) for name in HAIR_PARTS}
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        name = str(seg.get("class") or "")
        if name not in layers:
            continue
        poly = _points(seg.get("polygon") or [])
        if len(poly) < 3:
            continue
        cv2.fillPoly(layers[name], [np.round(poly).astype(np.int32)], 255)
    stroke = np.zeros((h, w), dtype=np.uint8)
    _paint_stroke(stroke, _points(points), max(1, int(round(float(radius)))))
    if int(stroke.max()) == 0:
        return [dict(seg) for seg in (segments or []) if isinstance(seg, dict)]
    if erase:
        clear = cv2.bitwise_not(stroke)
        for name in layers:
            layers[name] = cv2.bitwise_and(layers[name], clear)
    else:
        chosen = part if part in layers else "hair_middle"
        keep = cv2.bitwise_not(stroke)
        for name in layers:
            if name == chosen:
                layers[name] = cv2.bitwise_or(layers[name], stroke)
            else:
                layers[name] = cv2.bitwise_and(layers[name], keep)
    out: list[dict[str, Any]] = []
    for name in HAIR_PARTS:
        out.extend(_polygons(layers[name], name))
    return out
