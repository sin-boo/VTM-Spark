"""Raw vision / pose stack test — no DiT, VAE, crop, normalize, or repair.

Reads images from ``test-pose/input/``, runs face+iris+dwpose full stack in
TEST_MODE (schema-mapped detector pixels, geometric repair off), draws overlay
onto the image, writes to ``test-pose/output/``.

Toggle::

    TEST_MODE = True   # run vision test (default for this script)
    TEST_MODE = False  # exit immediately (safety)

Launch via ``test.bat`` — does not start the stream UI or generation model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Toggle
# ---------------------------------------------------------------------------
TEST_MODE = True

ROOT = Path(__file__).resolve().parent
TEST_POSE_DIR = ROOT / "test-pose"
INPUT_DIR = TEST_POSE_DIR / "input"
OUTPUT_DIR = TEST_POSE_DIR / "output"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _discover_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(path)
    return out


def _pixels_to_normalized(kps: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map absolute pixel (37,4) → fake crop [-1,1] for overlay draw only."""
    out = np.asarray(kps, dtype=np.float32).copy()
    w = max(int(width), 2)
    h = max(int(height), 2)
    vis = out[:, 3] >= 0.5
    out[vis, 0] = out[vis, 0] / float(w - 1) * 2.0 - 1.0
    out[vis, 1] = out[vis, 1] / float(h - 1) * 2.0 - 1.0
    return out


def _keypoints_json(kps: np.ndarray) -> list[dict]:
    rows = []
    for i in range(int(kps.shape[0])):
        rows.append(
            {
                "i": i,
                "x": float(kps[i, 0]),
                "y": float(kps[i, 1]),
                "score": float(kps[i, 2]),
                "visible": bool(float(kps[i, 3]) >= 0.5),
            }
        )
    return rows


def run_one(image_path: Path, out_dir: Path, *, device: str | None = None) -> Path:
    from pose_controller import draw_keypoint_mesh
    from ref_pose_fit import detect_face37_pixels

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = rgb.shape[:2]
    print(f"\n=== {image_path.name} ({w}x{h}) ===")

    # Full stack: face YOLO+HRNet (schema-mapped) + iris_pose + dwpose_v2 —
    # no geometric repair / eye refine / flip / crop / normalize.
    kps_px = detect_face37_pixels(rgb, device=device, test_mode=True)

    stem = image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{stem}_keypoints_px.npy"
    json_path = out_dir / f"{stem}_keypoints_px.json"
    overlay_path = out_dir / f"{stem}_overlay.png"

    np.save(npy_path, kps_px.astype(np.float32))
    json_path.write_text(
        json.dumps(
            {
                "source": str(image_path),
                "width": w,
                "height": h,
                "test_mode": True,
                "schema_mapped": True,
                "space": "pixels",
                "keypoints": _keypoints_json(kps_px),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Draw mesh onto the original image (convert px → [-1,1] only for drawer).
    kps_norm = _pixels_to_normalized(kps_px, w, h)
    overlay = draw_keypoint_mesh(Image.fromarray(rgb), kps_norm)
    overlay.save(overlay_path)

    vis = int(np.sum(kps_px[:, 3] >= 0.5))
    gap = None
    if kps_px[21, 3] >= 0.5 and kps_px[25, 3] >= 0.5:
        gap = float(kps_px[25, 1] - kps_px[21, 1])
    print(
        f"  visible {vis}/37  mouth_gap_px={gap if gap is not None else 'n/a'}"
    )
    print(f"  wrote {overlay_path.name}")
    print(f"  wrote {npy_path.name}")
    print(f"  wrote {json_path.name}")
    return overlay_path


def main() -> int:
    if not TEST_MODE:
        print("TEST_MODE is False — nothing to do. Set TEST_MODE = True in test_pose.py.")
        return 0

    if not INPUT_DIR.is_dir():
        print(f"Missing input folder: {INPUT_DIR}")
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = _discover_images(INPUT_DIR)
    if not images:
        print(f"No images in {INPUT_DIR}")
        print("Put PNG/JPG stills in test-pose/input/, then re-run test.bat.")
        return 1

    print(f"TEST_MODE=ON  input={INPUT_DIR}")
    print(f"              output={OUTPUT_DIR}")
    print(f"Found {len(images)} image(s). Vision stack only (no DiT / VAE).")

    ok = 0
    fail = 0
    for path in images:
        try:
            run_one(path, OUTPUT_DIR)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"FAILED {path.name}: {exc}")

    print(f"\nDone. ok={ok} fail={fail} → {OUTPUT_DIR}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    # Keep console UTF-8 on Windows.
    for _stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(_stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    raise SystemExit(main())
