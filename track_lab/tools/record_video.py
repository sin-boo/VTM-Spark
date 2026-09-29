"""Record a short webcam video while a voice guides the moves.

    python -m tools.record_video               # guided, Track Lab's camera
    python -m tools.record_video --camera 0 --free 30

Writes output/captures/cam_<time>.mp4 and cam_<time>.jsonl (a header, then
one line per frame: time and guide step), opened the way Track Lab opens the
camera so the frames match what the tracker sees. Track Lab must not hold
the camera meanwhile.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from backend.cameras import open_capture
from backend.osf_cam import CAM_H, CAM_W
from tools.guide import Guide, Voice

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "output" / "captures"
CAMERA_SETTINGS = ROOT / "output" / "camera.json"
FPS = 30.0
FIRST_FRAME_SEC = 10.0


def saved_camera() -> dict[str, Any]:
    """The camera Track Lab last used."""
    try:
        data = json.loads(CAMERA_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record(out: Path, *, camera: int | None, free: float, voice_on: bool) -> int:
    saved = saved_camera()
    index = int(camera if camera is not None else saved.get("index", 0) or 0)
    cap = open_capture(index, CAM_W, CAM_H)
    if cap is None:
        print(f"Camera {index} did not open. Is Track Lab using it? Stop tracking there first.", flush=True)
        return 1
    voice = Voice(voice_on and free <= 0)
    guide = None if free > 0 else Guide(voice=voice)
    length = free if free > 0 else guide.total  # type: ignore[union-attr]
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".jsonl")
    writer = None
    frames = 0
    try:
        waited = time.perf_counter()
        frame = None
        while frame is None:
            ok, grabbed = cap.read()
            if ok and grabbed is not None and grabbed.size:
                frame = grabbed
            elif time.perf_counter() - waited > FIRST_FRAME_SEC:
                print(f"Camera {index} sends no picture. Is Track Lab using it?", flush=True)
                return 1
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        if not writer.isOpened():
            print(f"Could not write {out}", flush=True)
            return 1
        print(f"Camera {index} ({saved.get('name') or '?'}) {w}x{h}. Recording {length:.0f} s.", flush=True)
        with log_path.open("w", encoding="utf-8") as fh:
            header = {
                "kind": "header",
                "tool": "record_video",
                "started": datetime.now().isoformat(timespec="seconds"),
                "camera": index,
                "camera_name": saved.get("name"),
                "size": [w, h],
                "video": out.name,
                "guided": guide is not None,
                "steps": [list(step) for step in guide.steps] if guide is not None else [],
            }
            fh.write(json.dumps(header) + "\n")
            start = time.perf_counter()
            step = "free"
            while True:
                t = time.perf_counter() - start
                if guide is not None:
                    step, _changed, done = guide.tick(t)
                    if done:
                        break
                elif t >= length:
                    break
                if frame is None:
                    ok, frame = cap.read()
                    if not ok or frame is None or not frame.size:
                        frame = None
                        continue
                    t = time.perf_counter() - start
                writer.write(frame)
                fh.write(json.dumps({"i": frames, "t": round(t, 4), "step": step}) + "\n")
                frames += 1
                frame = None
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        voice.close()
    print(f"Saved {frames} frames to {out} (steps in {log_path.name})", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path, default=CAPTURES / f"cam_{stamp}.mp4")
    parser.add_argument("--camera", type=int, default=None, help="camera index (default: Track Lab's)")
    parser.add_argument("--free", type=float, default=0.0, help="record this many seconds, no prompts")
    parser.add_argument("--quiet", action="store_true", help="print the prompts, do not speak them")
    args = parser.parse_args()
    return record(args.out, camera=args.camera, free=args.free, voice_on=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
