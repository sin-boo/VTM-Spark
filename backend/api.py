"""FastAPI surface for the VTM Noble desktop app."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .paths import (
    default_ref_path,
    display_path,
    ensure_under_models,
    models_dir,
    refs_dir,
    ui_dist_dir,
)
from .stream import get_runtime, shutdown_runtime

HOST = "127.0.0.1"
PORT = 8765

__all__ = ["app", "configure_runtime", "mount_frontend", "shutdown_runtime"]

app = FastAPI(title="VTM Noble", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ws_clients: list[WebSocket] = []
_ws_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_frontend_mounted = False
_dialog_lock = threading.Lock()


def _pick_checkpoint_file() -> str | None:
    """Native OS file dialog scoped to package ``models/`` (opens in models/dit).

    Returns a package-relative path under ``models/``, or None if cancelled.
    Raises HTTPException if the user picks a file outside ``models/``.
    """
    with _dialog_lock:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return None
        start = models_dir()
        start.mkdir(parents=True, exist_ok=True)
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            path = filedialog.askopenfilename(
                title="Select DiT checkpoint (models/)",
                initialdir=str(start),
                filetypes=[
                    ("Checkpoints", "*.pt*"),
                    ("PyTorch weights", "*.pt"),
                    ("All files", "*.*"),
                ],
            )
        finally:
            try:
                root.destroy()
            except Exception:
                pass
        picked = str(path).strip() or None
        if not picked:
            return None
        try:
            return display_path(ensure_under_models(picked))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


class SettingsBody(BaseModel):
    steps: int | None = None
    track_fps: float | None = None
    drive_pose: bool | None = None
    show_mesh: bool | None = None
    mirror: bool | None = None
    use_iris: bool | None = None
    use_body: bool | None = None
    fast_mode: bool | None = None
    batch2: bool | None = None
    auto_sync_track: bool | None = None
    camera_index: int | None = None


class PathBody(BaseModel):
    path: str = Field(..., min_length=1)


class MeshBody(BaseModel):
    x: float
    y: float


def _broadcast(event: dict[str, Any]) -> None:
    loop = _loop
    if loop is None:
        return
    with _ws_lock:
        clients = list(_ws_clients)

    async def _send_all() -> None:
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        if dead:
            with _ws_lock:
                for ws in dead:
                    if ws in _ws_clients:
                        _ws_clients.remove(ws)

    try:
        asyncio.run_coroutine_threadsafe(_send_all(), loop)
    except Exception:
        pass


def configure_runtime() -> None:
    rt = get_runtime()
    rt.add_listener(_broadcast)


def mount_frontend() -> None:
    global _frontend_mounted
    if _frontend_mounted:
        return
    dist = ui_dist_dir()
    if not (dist / "index.html").is_file():
        return
    app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(dist / "index.html")

    _frontend_mounted = True


@app.on_event("startup")
async def _on_startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    configure_runtime()
    # First-run setup: fetch VTM-ELF.pt into models/dit when missing.
    try:
        from .model_download import ensure_default_model

        ensure_default_model(blocking=False)
    except Exception:
        pass


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"ok": "1"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return get_runtime().status()


@app.get("/api/checkpoints")
def checkpoints() -> list[dict[str, str]]:
    return get_runtime().list_checkpoints()


@app.post("/api/checkpoint")
def set_checkpoint(body: PathBody) -> dict[str, Any]:
    try:
        get_runtime().set_checkpoint(body.path)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/checkpoint/browse")
def browse_checkpoint() -> dict[str, Any]:
    """Open a native file picker and apply the selected checkpoint immediately."""
    path = _pick_checkpoint_file()
    if not path:
        return {"cancelled": True, "status": get_runtime().status()}
    try:
        get_runtime().set_checkpoint(path)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"cancelled": False, "status": get_runtime().status()}


@app.post("/api/settings")
def settings(body: SettingsBody) -> dict[str, Any]:
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    return get_runtime().update_settings(**data)


@app.get("/api/cameras")
def cameras() -> dict[str, Any]:
    rt = get_runtime()
    cams = rt.list_cameras()
    return {"cameras": cams, "preferred": rt.preferred_camera()}


@app.post("/api/reference")
def reference(body: PathBody) -> dict[str, Any]:
    try:
        frame = get_runtime().apply_reference(body.path)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "frame": frame, "status": get_runtime().status()}


@app.post("/api/reference/upload")
async def reference_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "ref.png").suffix or ".png"
    dest = refs_dir() / f"upload{suffix}"
    data = await file.read()
    dest.write_bytes(data)
    try:
        # Apply on a worker thread so the event loop can keep flushing
        # websocket status ("Encoding reference…") instead of freezing the UI.
        frame = await asyncio.to_thread(get_runtime().apply_reference, dest)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "ok": True,
        "path": display_path(dest),
        "frame": frame,
        "status": get_runtime().status(),
    }


@app.get("/api/reference/default")
def reference_default() -> dict[str, str]:
    path = default_ref_path()
    return {"path": display_path(path), "exists": str(path.is_file()).lower()}


@app.get("/api/models/status")
def models_status_api() -> dict[str, Any]:
    from .model_download import models_status

    return models_status()


@app.get("/api/models/download/status")
def models_download_status() -> dict[str, Any]:
    from .model_download import download_status

    return download_status()


@app.post("/api/models/download")
def models_download_start() -> dict[str, Any]:
    from .model_download import start_model_download

    return start_model_download()


@app.post("/api/models/reload")
def models_reload() -> dict[str, Any]:
    """After files land in models/dit, load the default checkpoint."""
    try:
        from .engine import default_stream_checkpoint

        ckpt = default_stream_checkpoint()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"No checkpoint in models/dit yet (looked for {ckpt})"
            )
        get_runtime().set_checkpoint(ckpt)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/tracking/start")
def tracking_start() -> dict[str, Any]:
    try:
        get_runtime().start_tracking()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/tracking/stop")
def tracking_stop() -> dict[str, Any]:
    get_runtime().stop_tracking()
    return get_runtime().status()


@app.post("/api/tracking/calibrate")
def tracking_calibrate() -> dict[str, Any]:
    try:
        get_runtime().calibrate_ref()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/tracking/recenter")
def tracking_recenter() -> dict[str, Any]:
    try:
        get_runtime().recenter()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/generate")
def generate() -> dict[str, Any]:
    try:
        get_runtime().generate_once()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/stream/start")
def stream_start() -> dict[str, Any]:
    try:
        get_runtime().start_stream()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/stream/stop")
def stream_stop() -> dict[str, Any]:
    get_runtime().stop_stream()
    return get_runtime().status()


@app.post("/api/virtual-cam/start")
def virtual_cam_start() -> dict[str, Any]:
    try:
        get_runtime().start_virtual_cam()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/virtual-cam/stop")
def virtual_cam_stop() -> dict[str, Any]:
    get_runtime().stop_virtual_cam()
    return get_runtime().status()


@app.post("/api/mesh/press")
def mesh_press(body: MeshBody) -> dict[str, str]:
    get_runtime().mesh_press(body.x, body.y)
    return {"ok": "1"}


@app.post("/api/mesh/drag")
def mesh_drag(body: MeshBody) -> dict[str, str]:
    get_runtime().mesh_drag(body.x, body.y)
    return {"ok": "1"}


@app.post("/api/mesh/release")
def mesh_release() -> dict[str, str]:
    get_runtime().mesh_release()
    return {"ok": "1"}


@app.post("/api/mesh/reset")
def mesh_reset() -> dict[str, str]:
    get_runtime().mesh_reset()
    return {"ok": "1"}


@app.get("/api/outputs/last")
def last_output() -> None:
    """Disabled — frames stay in-memory / UI only; nothing is written to outputs/."""
    raise HTTPException(404, "Output folder writes are disabled")


@app.websocket("/api/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    with _ws_lock:
        _ws_clients.append(ws)
    try:
        await ws.send_json({"type": "status", "status": get_runtime().status()})
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)
