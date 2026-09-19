from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from backend import vtm_gen
from harness.dispatch import handle
from harness.protocol import parse_command


def _face(x0: float = 10.0, y0: float = 20.0) -> list[list[float]]:
    return [[x0 + i, y0, 1.0] for i in range(28)]


def test_overlay_norm_maps_character_px_to_norm_crop() -> None:
    frame = {
        "image_wh": [200, 100],
        "keypoints": [
            {"i": 0, "x": 150.0, "y": 25.0, "score": 1.0, "visible": True},
            {"i": 21, "x": 0.0, "y": 100.0, "score": 0.9, "visible": True},
        ],
    }
    k = vtm_gen.overlay_norm(frame)
    assert k is not None
    assert k.shape == (37, 4)
    assert abs(float(k[0, 0]) - 0.5) < 1e-5
    assert abs(float(k[0, 1]) - -0.5) < 1e-5
    assert abs(float(k[21, 0]) - -1.0) < 1e-5
    assert abs(float(k[21, 1]) - 1.0) < 1e-5
    assert float(k[0, 3]) == 1.0


def test_hair_norm_maps_polygons() -> None:
    segs = vtm_gen.hair_norm(
        {
            "image_wh": [640, 480],
            "hair": [
                {
                    "class": "hair_middle",
                    "polygon": [[0, 0], [640, 0], [640, 480], [0, 480]],
                }
            ],
        }
    )
    assert len(segs) == 1
    poly = segs[0]["polygon"]
    assert segs[0]["class"] == "hair_middle"
    assert abs(poly[0][0] - -1.0) < 1e-5
    assert abs(poly[1][0] - 1.0) < 1e-5
    assert abs(poly[2][1] - 1.0) < 1e-5


def test_overlay_norm_empty_is_none() -> None:
    assert vtm_gen.overlay_norm({"image_wh": [64, 64], "keypoints": []}) is None
    assert vtm_gen.overlay_norm(None) is None


class _Shape:
    shape = (480, 640, 3)


class _Bench:
    source_bgr = _Shape()
    generation = 3

    def live_status(self) -> dict[str, object]:
        return {
            "live": False,
            "points": _face(),
            "hair": [],
            "skeleton": [],
            "iris": [],
        }

    def status(self) -> dict[str, object]:
        return {"ok": True, "ready": True}


def test_generate_runs_engine_on_screen_overlay(monkeypatch, tmp_path: Path) -> None:
    vtm_gen._jpeg = None
    still = tmp_path / "source.png"
    still.write_bytes(b"not-a-real-png")
    seen: dict[str, object] = {}

    class _Engine:
        def generate_from_keypoints(self, kps, sanitize="none", hair_maps=None):
            seen["kps"] = np.asarray(kps)
            seen["sanitize"] = sanitize
            seen["hair_maps"] = hair_maps
            return Image.new("RGB", (8, 8), (12, 34, 56)), 0.25

    monkeypatch.setattr(vtm_gen, "_source", lambda: still)
    monkeypatch.setattr(vtm_gen, "_engine_for", lambda src: _Engine())
    monkeypatch.setattr(vtm_gen, "_hair_maps", lambda segs: np.ones((3, 4, 4), dtype=np.float32) if segs else None)

    overlay = {
        "points": [[320.0, 240.0, 1.0] for _ in range(28)],
        "hair": [{"class": "hair_middle", "polygon": [[0, 0], [640, 0], [640, 480], [0, 480]]}],
        "skeleton": [{"id": 31, "x": 320.0, "y": 400.0, "score": 1.0}],
        "iris": [],
    }
    out = vtm_gen.generate(_Bench(), overlay)
    assert out["gen"] is True
    assert out["gen_ms"] == 250.0
    assert seen["sanitize"] == "none"
    kps = seen["kps"]
    assert abs(float(kps[0, 0])) < 1e-5
    assert abs(float(kps[0, 1])) < 1e-5
    assert seen["hair_maps"] is not None
    jpeg = vtm_gen.last_jpeg()
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"


def test_generate_needs_points(monkeypatch, tmp_path: Path) -> None:
    still = tmp_path / "source.png"
    still.write_bytes(b"x")
    monkeypatch.setattr(vtm_gen, "_source", lambda: still)

    class Empty:
        source_bgr = _Shape()
        generation = 0

        def live_status(self) -> dict[str, object]:
            return {"points": [], "hair": []}

        def status(self) -> dict[str, object]:
            return {}

    try:
        vtm_gen.generate(Empty(), {"points": []})
    except RuntimeError as exc:
        assert "overlay" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_parse_and_dispatch_generate(monkeypatch) -> None:
    msg = parse_command({"op": "generate", "body": {"points": [[1.0, 2.0, 1.0]]}})
    assert msg["op"] == "generate"
    assert msg["body"]["points"][0][1] == 2.0
    called: dict[str, object] = {}

    def fake(bench, overlay=None):
        called["overlay"] = overlay
        return {"ok": True, "gen": True, "error": ""}

    monkeypatch.setattr(vtm_gen, "generate", fake)
    reply = handle(_Bench(), {"op": "generate", "body": {"points": _face(3.0, 4.0)}})
    assert reply["ok"] is True
    assert called["overlay"]["points"][0][0] == 3.0


def test_lab_http_exposes_generate() -> None:
    from backend.server import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/generate" in paths
