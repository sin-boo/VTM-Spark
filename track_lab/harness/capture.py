"""Record live harness frames to JSON lines, for replaying real motion in tests.

    python -m harness.capture --seconds 300 --out output/captures/turns.jsonl

Polls ``/harness/frame`` and keeps a row whenever the head, turn or blink
moved, so a 60 fps phone is not written twice per packet. Stops after
``--seconds`` or as soon as ``<out>.stop`` exists.
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
PORT = 8780


def row_of(frame: dict[str, Any], now: float) -> dict[str, Any]:
    """The parts of a frame a motion test needs, compact."""
    points = [
        [round(float(r.get("x", 0.0)), 2), round(float(r.get("y", 0.0)), 2), 1 if r.get("visible") else 0]
        for r in frame.get("keypoints") or []
    ]
    return {
        "t": round(now, 4),
        "source": frame.get("source"),
        "faces": frame.get("faces"),
        "head": frame.get("head"),
        "turn": frame.get("turn"),
        "blink": frame.get("blink"),
        "look": frame.get("look"),
        "weights": frame.get("weights"),
        "image_wh": frame.get("image_wh"),
        "keypoints": points,
    }


def _moved(row: dict[str, Any], last: dict[str, Any] | None) -> bool:
    if last is None:
        return True
    return any(row.get(key) != last.get(key) for key in ("head", "turn", "blink", "look"))


def capture(out: Path, seconds: float, hz: float = 120.0) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = out.with_name(out.name + ".stop")
    stop.unlink(missing_ok=True)
    conn = http.client.HTTPConnection(HOST, PORT, timeout=2.0)
    end = time.monotonic() + float(seconds)
    period = 1.0 / max(float(hz), 1.0)
    last: dict[str, Any] | None = None
    rows = 0
    with out.open("w", encoding="utf-8") as fh:
        while time.monotonic() < end and not stop.exists():
            started = time.monotonic()
            try:
                conn.request("GET", "/harness/frame")
                frame = json.loads(conn.getresponse().read())
            except (OSError, http.client.HTTPException, ValueError):
                conn.close()
                conn = http.client.HTTPConnection(HOST, PORT, timeout=2.0)
                time.sleep(0.2)
                continue
            row = row_of(frame, time.time())
            if _moved(row, last):
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                fh.flush()
                rows += 1
                last = row
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--hz", type=float, default=120.0)
    args = parser.parse_args()
    rows = capture(args.out, args.seconds, args.hz)
    print(f"[capture] {rows} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
