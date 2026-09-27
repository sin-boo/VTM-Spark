"""Record live character motion for the fine-tune benchmark.

Each sample is the overlay the lab is already drawing: face, skeleton, iris,
and hair in character pixels. On stop, that stream is written the same way
``generate_baseline.py`` reads a clip: reference still, ``poses.npz``,
``hair.json``, and per-frame mouth weights.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from harness.points import BODY_END, BODY_START, FACE_COUNT, NUM_KEYPOINTS

from .paths import REPO

SHAPES = ("smile", "sad", "A", "I", "U", "E")
BENCHMARK = REPO.parent / "post-prosesing" / "fine-tuning" / "benchmark" / "recordings"


def _norm(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    return float(x) / float(w) * 2.0 - 1.0, float(y) / float(h) * 2.0 - 1.0


def pack_pose(
    points: object,
    skeleton: object,
    iris: object,
    w: int,
    h: int,
) -> np.ndarray:
    """(37, 4) keypoints in [-1, 1], matching ``generate_baseline._pack``."""
    kps = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)
    if w < 1 or h < 1:
        return kps
    if isinstance(points, (list, tuple)) and len(points) >= FACE_COUNT:
        for i in range(FACE_COUNT):
            row = points[i]
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            nx, ny = _norm(float(row[0]), float(row[1]), w, h)
            score = float(row[2]) if len(row) > 2 else 1.0
            visible = 1.0 if score >= 0.05 else 0.0
            kps[i] = (nx, ny, max(score, 0.0), visible)
    if isinstance(skeleton, list):
        for joint in skeleton:
            if not isinstance(joint, dict):
                continue
            try:
                idx = int(joint["id"])
                nx, ny = _norm(float(joint["x"]), float(joint["y"]), w, h)
                score = float(joint.get("score", 1.0))
            except (KeyError, TypeError, ValueError):
                continue
            if idx < BODY_START or idx > BODY_END:
                continue
            kps[idx] = (nx, ny, max(score, 0.0), 1.0)
    if isinstance(iris, list):
        for row in iris:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row["id"])
                nx, ny = _norm(float(row["x"]), float(row["y"]), w, h)
                score = float(row.get("score", 1.0))
            except (KeyError, TypeError, ValueError):
                continue
            if idx not in (28, 29) or not bool(row.get("visible", True)):
                continue
            kps[idx] = (nx, ny, max(score, 0.0), 1.0)
    return kps


def pack_hair(hair: object, w: int, h: int) -> list[dict]:
    if w < 1 or h < 1 or not isinstance(hair, list):
        return []
    sx = 2.0 / float(w)
    sy = 2.0 / float(h)
    out: list[dict] = []
    for part in hair:
        if not isinstance(part, dict):
            continue
        poly = part.get("polygon") or []
        cls = str(part.get("class") or "")
        if not cls or not isinstance(poly, list) or len(poly) < 3:
            continue
        pts = []
        for vertex in poly:
            if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                continue
            pts.append([float(vertex[0]) * sx - 1.0, float(vertex[1]) * sy - 1.0])
        if len(pts) >= 3:
            out.append({"class": cls, "polygon": pts})
    return out


def _weights(raw: object) -> dict[str, float]:
    src = raw if isinstance(raw, dict) else {}
    return {name: round(float(src.get(name) or 0.0), 4) for name in SHAPES}


class MovementRecorder:
    """One take. Start while tracking is live, stop to write the clip."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else BENCHMARK
        self.recording = False
        self.frames = 0
        self.started = 0.0
        self.path = ""
        self.error = ""
        self._rows: list[np.ndarray] = []
        self._hair: list[list[dict]] = []
        self._weights: list[dict[str, float]] = []
        self._times: list[float] = []
        self._still: np.ndarray | None = None
        self._source = ""

    def payload(self) -> dict[str, object]:
        elapsed = 0.0
        if self.recording and self.started:
            elapsed = max(0.0, time.time() - self.started)
        return {
            "recording": self.recording,
            "record_frames": self.frames,
            "record_seconds": round(elapsed, 2),
            "record_path": self.path,
            "record_error": self.error,
        }

    def start(self, still: np.ndarray | None, source: str) -> None:
        if still is None or getattr(still, "size", 0) == 0:
            raise ValueError("Load a reference image before recording movement")
        self._rows = []
        self._hair = []
        self._weights = []
        self._times = []
        self._still = np.asarray(still).copy()
        self._source = source
        self.frames = 0
        self.path = ""
        self.error = ""
        self.started = time.time()
        self.recording = True

    def note(
        self,
        points: object,
        skeleton: object,
        iris: object,
        hair: object,
        weights: object,
        width: int,
        height: int,
    ) -> None:
        if not self.recording:
            return
        self._rows.append(pack_pose(points, skeleton, iris, width, height))
        self._hair.append(pack_hair(hair, width, height))
        self._weights.append(_weights(weights))
        self._times.append(time.time())
        self.frames = len(self._rows)

    def stop(self) -> dict[str, object]:
        self.recording = False
        if not self._rows or self._still is None:
            self.error = "Recording had no tracked frames"
            self.frames = 0
            return self.payload()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = self.root / stamp
        dest.mkdir(parents=True, exist_ok=True)
        still_path = dest / "base.png"
        if not cv2.imwrite(str(still_path), self._still):
            # Windows paths with odd characters: write via imencode.
            ok, buf = cv2.imencode(".png", self._still)
            if not ok:
                self.error = "Could not write the reference image"
                return self.payload()
            still_path.write_bytes(buf.tobytes())
        spans = [
            self._times[i] - self._times[i - 1]
            for i in range(1, len(self._times))
            if self._times[i] > self._times[i - 1]
        ]
        if spans:
            fps = float(np.clip(1.0 / float(np.median(spans)), 1.0, 60.0))
        else:
            fps = 24.0
        np.savez_compressed(
            dest / "poses.npz",
            keypoints=np.stack(self._rows),
            fps=np.float32(fps),
        )
        (dest / "hair.json").write_text(json.dumps(self._hair), encoding="utf-8")
        peaks = {
            name: round(max(float(row[name]) for row in self._weights), 4)
            for name in SHAPES
        }
        track = {
            "source": "track_lab",
            "reference": self._source,
            "fps": round(fps, 3),
            "frames": len(self._rows),
            "image": [int(self._still.shape[1]), int(self._still.shape[0])],
            "shapes": list(SHAPES),
            "peaks": peaks,
            "files": {
                "base": "base.png",
                "poses": "poses.npz",
                "hair": "hair.json",
            },
            "track": [
                {"i": i, "t": round(i / fps, 4), "weights": row}
                for i, row in enumerate(self._weights)
            ],
        }
        (dest / "track.json").write_text(json.dumps(track, indent=2), encoding="utf-8")
        self.path = str(dest)
        self.error = ""
        self._rows = []
        self._hair = []
        self._weights = []
        self._times = []
        self._still = None
        return self.payload()
