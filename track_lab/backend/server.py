"""Minimal API: load a still, Track it, Reset the Python tracker.

This process is the harness host. It does not import torch. FaceBench lives
in ``python -m backend.worker`` and attaches over localhost IPC.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from harness.bridge import spawn_tracker_worker
from harness.hub import hub
from harness.pack import warming_status
from harness.server import attach as attach_harness

from .ports import HOST, PORT, SERVICE

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
SOURCE_NAME = "source.png"

app = FastAPI(title="Track Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
attach_harness(app, None)


@app.on_event("startup")
def _spawn_worker() -> None:
    spawn_tracker_worker()


def _from_ack(result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status") if isinstance(result.get("status"), dict) else {}
    out = dict(status) if isinstance(status, dict) else {}
    out["ok"] = bool(result.get("ok")) and not result.get("error")
    if result.get("error"):
        out["error"] = result.get("error")
    return out


def _command(op: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    return _from_ack(hub.command({"op": op, "body": body or {}}))


def _latest_status() -> dict:
    packet = hub.latest_status()
    if packet is None:
        packet = warming_status(clients=hub.clients)
        hub.publish(packet)
    return dict(packet)


@app.get("/api/health")
def health() -> dict[str, object]:
    packet = hub.latest_status() or {}
    return {
        "ok": True,
        "service": SERVICE,
        "port": PORT,
        "loaded": bool(packet.get("loaded")),
    }


@app.get("/api/status")
def status() -> dict:
    return _latest_status()


@app.post("/api/source")
async def source(request: Request) -> dict:
    payload = await request.body()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = INPUT_DIR / SOURCE_NAME
    dest.write_bytes(payload)
    return await asyncio.to_thread(_command, "set_source", {"path": str(dest)})


@app.post("/api/track")
def track() -> dict:
    return _command("track")


@app.post("/api/reset")
def reset() -> dict:
    return _command("reset")


@app.post("/api/preset/apply")
async def apply_preset(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "apply_preset", {"id": str(body.get("id", ""))})


@app.post("/api/preset/mouth")
async def set_mouth(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(
        _command, "set_mouth", {"id": str(body.get("id", "")), "mouth": body.get("mouth")}
    )


@app.post("/api/preset/move")
async def move_key(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(
        _command, "move_key", {"id": str(body.get("id", "")), "t": body.get("t")}
    )


@app.post("/api/preset/drop")
async def drop_key(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "drop_key", {"id": str(body.get("id", ""))})


@app.post("/api/osf/start")
async def osf_start(request: Request) -> dict:
    camera = None
    source = None
    host = None
    port = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            if body.get("camera") is not None:
                camera = int(body["camera"])
            if body.get("source") is not None:
                source = str(body["source"])
            if body.get("host") is not None:
                host = str(body["host"])
            if body.get("port") is not None:
                port = int(body["port"])
    except Exception:
        pass
    payload: dict[str, Any] = {}
    if camera is not None:
        payload["camera"] = camera
    if source is not None:
        payload["source"] = source
    if host is not None:
        payload["host"] = host
    if port is not None:
        payload["port"] = port
    return await asyncio.to_thread(_command, "start", payload)


@app.post("/api/osf/stop")
def osf_stop() -> dict:
    return _command("stop")


@app.post("/api/input")
async def track_input(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "set_input", {"source": str(body.get("source", "camera"))})


@app.post("/api/ifm")
async def ifm_settings(request: Request) -> dict:
    body = await request.json()
    payload: dict[str, Any] = {}
    if body.get("host") is not None:
        payload["host"] = str(body.get("host"))
    if body.get("port") is not None:
        payload["port"] = int(body["port"])
    return await asyncio.to_thread(_command, "set_ifm", payload)


@app.post("/api/camera")
async def camera(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "set_camera", {"index": int(body.get("index", 0))})


@app.post("/api/cameras/refresh")
async def refresh_cameras() -> dict:
    return await asyncio.to_thread(_command, "refresh_cameras")


@app.post("/api/feel")
async def set_feel(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "set_feel", body if isinstance(body, dict) else {})


@app.post("/api/travel")
async def set_travel(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "set_travel", body if isinstance(body, dict) else {})


@app.post("/api/mirror")
async def set_mirror(request: Request) -> dict:
    body = await request.json()
    on = body.get("on") if isinstance(body, dict) else None
    if on is None and isinstance(body, dict):
        on = body.get("mirror")
    return await asyncio.to_thread(_command, "set_mirror", {"on": bool(on)})


@app.post("/api/mouth-points")
async def set_mouth_point(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "set_mouth_point", body if isinstance(body, dict) else {})


@app.post("/api/eye-points")
async def set_eye_point(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "set_eye_point", body if isinstance(body, dict) else {})


@app.post("/api/skeleton-point")
async def set_skeleton_point(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "set_skeleton_point", body if isinstance(body, dict) else {})


@app.post("/api/overlay-point")
async def set_overlay_point(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "set_point", body if isinstance(body, dict) else {})


@app.post("/api/generate")
async def generate(request: Request) -> dict:
    body: dict[str, Any] = {}
    try:
        parsed = await request.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:
        body = {}
    return await asyncio.to_thread(_command, "generate", body)


@app.post("/api/overlay-points/reset")
async def reset_overlay_points(request: Request) -> dict:
    body = {}
    try:
        parsed = await request.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:
        body = {}
    return await asyncio.to_thread(_command, "reset_points", body)


@app.post("/api/record")
async def record_movement(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "record", {"on": bool(body.get("on"))})


@app.post("/api/calibrate")
async def calibrate(request: Request) -> dict:
    body = await request.json()
    return await asyncio.to_thread(_command, "calibrate", {"id": str(body.get("id", ""))})


@app.post("/api/calibrate/reset")
def calibrate_reset() -> dict:
    return _command("reset_calibrate")


@app.get("/api/live")
def live() -> dict:
    packet = hub.latest_frame()
    if packet is None:
        return _latest_status()
    return dict(packet)


@app.get("/api/frame/{kind}")
def frame(kind: str) -> Response:
    from harness.bridge import bridge

    payload = bridge.jpeg(kind)
    if payload is None:
        return Response(status_code=503)
    return Response(content=payload, media_type="image/jpeg")
