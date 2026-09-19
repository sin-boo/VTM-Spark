"""Dump hair-mask overlays from the still, without opening the desk."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "track_lab" / "input" / "source.png"
PARTS = ROOT / "track_lab" / "output" / "overlay_parts.json"
VTM = ROOT / "models" / "characters" / "ChatGPT-Image-Aug-3-2026-11_36_05-PM.vtm"
OUT = ROOT / "track_lab" / "output"

COLORS = {
    "hair_middle": (255, 200, 0),
    "hair_left": (0, 180, 255),
    "hair_right": (255, 80, 160),
}


def _draw(base: Image.Image, segs: list[dict], dest: Path) -> None:
    img = base.convert("RGB")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for seg in segs:
        cls = str(seg.get("class") or "")
        color = COLORS.get(cls)
        pts = [(float(x), float(y)) for x, y in (seg.get("polygon") or [])]
        if color is None or len(pts) < 3:
            continue
        draw.polygon(pts, fill=(*color, 140), outline=(*color, 255))
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        tag = "M" if cls.endswith("middle") else ("L" if cls.endswith("left") else "R")
        draw.text((cx, cy), tag, fill=(*color, 255))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)
    print("wrote", dest)


def _rest_pts(w: int, h: int) -> np.ndarray | None:
    if not VTM.is_file():
        return None
    with zipfile.ZipFile(VTM) as zf:
        raw = zf.read("keypoints.npy")
    k = np.load(__import__("io").BytesIO(raw)).astype(np.float32)
    if k.ndim != 2 or k.shape[0] < 20:
        return None
    px = np.zeros((len(k), 3), dtype=np.float32)
    px[:, 0] = (k[:, 0] + 1.0) * 0.5 * (w - 1)
    px[:, 1] = (k[:, 1] + 1.0) * 0.5 * (h - 1)
    px[:, 2] = 1.0
    return px


def main() -> None:
    from backend.hair import detect_hair

    rgb = np.asarray(Image.open(SRC).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]
    still = Image.fromarray(rgb)

    if PARTS.is_file():
        saved = list(json.loads(PARTS.read_text(encoding="utf-8")).get("hair") or [])
        if saved:
            _draw(still, saved, OUT / "hair_overlay_saved.png")

    segs = detect_hair(bgr, _rest_pts(w, h))
    _draw(still, segs, OUT / "hair_overlay_detect.png")


if __name__ == "__main__":
    main()
