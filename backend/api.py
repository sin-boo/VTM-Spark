"""FastAPI surface for the VTM Spark desktop app."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .desk_splash import ui_public_files
from .engine import checkpoint_browse_start_dir, remember_checkpoint_location
from .lab_harness import lab as lab_harness
from .paths import (
    default_ref_path,
    display_path,
    refs_dir,
    resolve_user_path,
    ui_dist_dir,
)
from .stream import get_runtime, shutdown_runtime
from .gpu_select import gpu_snapshot, list_gpus, save_gpu_pref
from .ui_prefs import load_ui_prefs, save_ui_prefs

HOST = "127.0.0.1"
PORT = 8765

__all__ = ["app", "configure_runtime", "mount_frontend", "shutdown_runtime"]

app = FastAPI(title="VTM Spark", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ws_clients: list[WebSocket] = []
_ws_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_frontend_mounted = False
_runtime_configured = False
_dialog_lock = threading.Lock()


_lab_boot_lock = threading.Lock()
_lab_boot_started = False


def _boot_track_lab() -> None:
    global _lab_boot_started
    with _lab_boot_lock:
        if _lab_boot_started:
            return
        _lab_boot_started = True
    try:
        from .lab_process import watch_lab

        watch_lab()
    except Exception as exc:
        print(f"[track-lab] ensure failed: {exc}", flush=True)


def _pick_checkpoint_file() -> str | None:
    """Native OS file dialog for a DiT checkpoint (.pt / .pth).

    Opens in the last custom folder when one exists, otherwise models/dit.
    Returns a display path, or None if cancelled.
    """
    with _dialog_lock:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return None
        start = checkpoint_browse_start_dir()
        start.mkdir(parents=True, exist_ok=True)
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            # Windows only honors `*.ext` (semicolon-separated). `*.pt*` hides
            # most real checkpoints and can make a folder look like it has one file.
            path = filedialog.askopenfilename(
                title="Select DiT checkpoint",
                initialdir=str(start),
                filetypes=[
                    ("PyTorch checkpoint", "*.pt *.pth *.ckpt;*.pt;*.pth;*.ckpt"),
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
        resolved = resolve_user_path(picked).resolve()
        remember_checkpoint_location(resolved)
        return display_path(resolved)


class SettingsBody(BaseModel):
    steps: int | None = None
    pose_cfg: float | None = None
    id_cfg: float | None = None
    frame_blend: float | None = None
    inbetweens: int | None = None
    interpolate: bool | None = None
    max_fps: int | None = None
    hold_last: bool | None = None
    track_fps: float | None = None
    drive_pose: bool | None = None
    show_mesh: bool | None = None
    show_hair: bool | None = None
    show_outline: bool | None = None
    show_brows: bool | None = None
    show_eyes: bool | None = None
    show_nose: bool | None = None
    show_mouth: bool | None = None
    show_iris_overlay: bool | None = None
    show_skeleton: bool | None = None
    show_limiters: bool | None = None
    mirror: bool | None = None
    use_iris: bool | None = None
    use_body: bool | None = None
    fast_mode: bool | None = None
    compile_model: bool | None = None
    batch: int | None = None
    batch2: bool | None = None
    auto_sync_track: bool | None = None
    camera_index: int | None = None
    travel_box: dict[str, Any] | None = None


class UiPrefsBody(BaseModel):
    language: str | None = None


class GpuBody(BaseModel):
    uuid: str = ""


class PathBody(BaseModel):
    path: str = Field(..., min_length=1)


class CharacterIdBody(BaseModel):
    id: str = Field(..., min_length=1)
    repair: bool = False


class CharacterRenameBody(BaseModel):
    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)


class CharacterMetaBody(BaseModel):
    id: str = Field(..., min_length=1)
    name: str | None = None
    author: str | None = None
    license: str | None = None
    description: str | None = None


class CharacterExportBody(BaseModel):
    id: str = Field(..., min_length=1)


class MeshBody(BaseModel):
    x: float
    y: float


class HairStrokeBody(BaseModel):
    part: str = "hair_middle"
    points: list[list[float]]
    radius: float = 16
    erase: bool = False


class SkeletonMoveBody(BaseModel):
    id: int
    x: float
    y: float


class BootCharacterBody(BaseModel):
    id: str = ""
    loaded: bool = False


class LabCommandBody(BaseModel):
    op: str = Field(..., min_length=1)
    body: dict[str, Any] = Field(default_factory=dict)


class DownloadBody(BaseModel):
    name: str | None = None
    names: list[str] | None = None


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
    global _runtime_configured
    if _runtime_configured:
        return
    rt = get_runtime()
    rt.add_listener(_broadcast)
    threading.Thread(target=_boot_track_lab, daemon=True, name="track-lab").start()
    rt.start_boot()
    _runtime_configured = True


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
        return FileResponse(
            dist / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    def _register_ui_file(filename: str, file_path: Path) -> None:
        @app.get("/" + filename, name=f"ui_public_{filename}", include_in_schema=False)
        async def serve_ui_public() -> FileResponse:
            return FileResponse(file_path)

        return None

    for name, path in ui_public_files(dist).items():
        _register_ui_file(name, path)

    _frontend_mounted = True


@app.on_event("startup")
async def _on_startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    configure_runtime()
    # First-run setup: fetch VTM-1.5.1.pt into models/dit when missing.
    try:
        from .model_download import ensure_default_model

        ensure_default_model(blocking=False)
    except Exception:
        pass


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"ok": "1"}


@app.post("/api/reload")
def reload_backend(request: Request) -> dict[str, Any]:
    """Spawn the out-of-process hold window, then exit so Python reimports."""
    from .desk_reload import request_reload

    host = request.url.hostname or HOST
    if host in {"localhost", "0.0.0.0", "::"}:
        host = HOST
    port = int(request.url.port or PORT)
    if port in {80, 443, 5173}:
        port = PORT
    try:
        return request_reload(host=str(host), port=port)
    except OSError as exc:
        raise HTTPException(500, f"Could not start reload window: {exc}") from exc


@app.get("/api/boot")
def boot_status() -> dict[str, Any]:
    return get_runtime().boot_snapshot()


@app.post("/api/boot")
def boot_start() -> dict[str, Any]:
    return get_runtime().start_boot()


@app.post("/api/gpu/repair")
def gpu_repair() -> dict[str, Any]:
    """Open install.bat in its own window: it closes the desk, installs the AI engine
    build for this graphics card and tests it (backend.gpu_check)."""
    from .gpu_check import launch_repair

    try:
        launch_repair()
    except (OSError, FileNotFoundError) as exc:
        raise HTTPException(500, f"Could not start the repair: {exc}") from exc
    return {"ok": True}


@app.post("/api/boot/character")
def boot_character(body: BootCharacterBody) -> dict[str, Any]:
    try:
        return get_runtime().finish_boot_character(
            body.id, already_loaded=bool(body.loaded)
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/status")
def status() -> dict[str, Any]:
    return get_runtime().status()


@app.get("/api/checkpoints")
def checkpoints() -> list[dict[str, str]]:
    return get_runtime().list_checkpoints()


@app.post("/api/checkpoint")
def set_checkpoint(body: PathBody) -> dict[str, Any]:
    """Select a model. It loads on the next Start stream / Generate."""
    try:
        get_runtime().select_checkpoint(body.path)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_runtime().status()


@app.post("/api/checkpoint/browse")
def browse_checkpoint() -> dict[str, Any]:
    """Open a native file picker and select the file (loads on Start stream)."""
    path = _pick_checkpoint_file()
    if not path:
        return {"cancelled": True, "path": None, "status": get_runtime().status()}
    rt = get_runtime()
    try:
        rt.select_checkpoint(path)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"cancelled": False, "path": path, "status": rt.status()}


@app.post("/api/settings")
def settings(body: SettingsBody) -> dict[str, Any]:
    data = {k: v for k, v in body.model_dump(exclude_none=True).items()}
    return get_runtime().update_settings(**data)


@app.get("/api/ui-prefs")
def ui_prefs() -> dict[str, Any]:
    return load_ui_prefs()


@app.post("/api/ui-prefs")
def set_ui_prefs(body: UiPrefsBody) -> dict[str, Any]:
    try:
        return save_ui_prefs(body.model_dump(exclude_none=True))
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/gpus")
def gpus() -> dict[str, Any]:
    return gpu_snapshot()


@app.post("/api/gpu")
def set_gpu(body: GpuBody) -> dict[str, Any]:
    """Save the GPU pick. It takes effect on the next start / Reload backend."""
    uuid = body.uuid.strip()
    if uuid and not any(g["uuid"] == uuid for g in list_gpus()):
        raise HTTPException(400, f"No GPU with id {uuid}")
    try:
        save_gpu_pref(uuid)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return gpu_snapshot()


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


@app.get("/api/characters")
def characters_list() -> dict[str, Any]:
    return {"characters": get_runtime().list_characters()}


@app.get("/api/characters/fit")
def character_fit() -> dict[str, Any]:
    try:
        return get_runtime().character_fit()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/characters/fit/hair")
def character_fit_hair(body: HairStrokeBody) -> dict[str, Any]:
    try:
        view = get_runtime().paint_character_hair(
            body.points,
            radius=body.radius,
            part=body.part,
            erase=body.erase,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "fit": view}


@app.post("/api/characters/fit/skeleton")
def character_fit_skeleton(body: SkeletonMoveBody) -> dict[str, Any]:
    try:
        view = get_runtime().move_character_skeleton(body.id, body.x, body.y)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "fit": view}


@app.post("/api/characters/fit/point")
def character_fit_point(body: SkeletonMoveBody) -> dict[str, Any]:
    try:
        view = get_runtime().move_character_point(body.id, body.x, body.y)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "fit": view}


@app.get("/api/characters/{ident}/preview")
def character_preview(ident: str) -> Response:
    from .character_pack import (
        CharacterPackError,
        ensure_character_still,
        resolve_character_id,
    )

    try:
        if ident.strip().lower() == "current":
            path = get_runtime().current_character_path()
            if path is None:
                return Response(status_code=204)
        else:
            path = resolve_character_id(ident)
        still = ensure_character_still(path)
        data = still.read_bytes()
    except CharacterPackError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/characters/{ident}/thumb")
def character_thumb(ident: str) -> Response:
    from .character_pack import (
        CharacterPackError,
        read_character_thumb_png,
        resolve_character_id,
    )

    try:
        data = read_character_thumb_png(resolve_character_id(ident))
    except CharacterPackError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/characters/{ident}/info")
def character_info(ident: str) -> dict[str, Any]:
    try:
        return get_runtime().character_info(ident)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/characters/meta")
def character_meta(body: CharacterMetaBody) -> dict[str, Any]:
    try:
        result = get_runtime().update_character_meta(
            body.id,
            name=body.name,
            author=body.author,
            license=body.license,
            description=body.description,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


def _attachment_header(filename: str) -> str:
    from urllib.parse import quote

    ascii_name = filename.encode("ascii", "ignore").decode("ascii") or "character.vtm"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@app.get("/api/characters/{ident}/export")
def character_export(ident: str) -> Response:
    try:
        filename, data = get_runtime().export_character_bytes(ident)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": _attachment_header(filename),
            "Cache-Control": "no-store",
        },
    )


def _desk_window() -> Any:
    """The pywebview desk window, or None when the UI runs in a browser."""
    try:
        import webview
    except Exception:
        return None
    windows = list(getattr(webview, "windows", None) or [])
    return windows[0] if windows else None


def _save_dialog_kind() -> Any:
    import webview

    dialog = getattr(webview, "FileDialog", None)
    if dialog is not None and hasattr(dialog, "SAVE"):
        return dialog.SAVE
    return webview.SAVE_DIALOG


@app.post("/api/characters/reveal")
def character_reveal(body: CharacterExportBody) -> dict[str, Any]:
    try:
        return get_runtime().reveal_character(body.id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/characters/export-save")
def character_export_save(body: CharacterExportBody) -> dict[str, Any]:
    """Native Save dialog in the desk. 404 without a window so the UI downloads instead."""
    window = _desk_window()
    if window is None:
        raise HTTPException(404, "No desk window; use the download export")
    try:
        filename, data = get_runtime().export_character_bytes(body.id)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    try:
        picked = window.create_file_dialog(
            _save_dialog_kind(),
            save_filename=filename,
            file_types=("VTM character (*.vtm)", "All files (*.*)"),
        )
    except Exception as exc:
        return {"ok": False, "error": f"Could not open the save dialog: {exc}"}
    if not picked:
        return {"ok": False, "cancelled": True}
    target = Path(picked[0] if isinstance(picked, (list, tuple)) else picked)
    if target.suffix.lower() != ".vtm":
        target = target.with_name(f"{target.name}.vtm")
    try:
        target.write_bytes(data)
    except OSError as exc:
        return {"ok": False, "error": f"Could not save {target.name}: {exc}"}
    return {"ok": True, "path": str(target)}


@app.post("/api/characters/create")
async def character_create(file: UploadFile = File(...)) -> dict[str, Any]:
    from .character_pack import stage_create_still

    suffix = Path(file.filename or "character.png").suffix or ".png"
    dest = stage_create_still(await file.read(), suffix=suffix)
    try:
        result = await asyncio.to_thread(
            get_runtime().create_character, dest, name=Path(file.filename or dest.stem).stem
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


@app.post("/api/characters/add")
async def character_add(file: UploadFile = File(...)) -> dict[str, Any]:
    from .character_pack import unique_character_path
    from .paths import characters_dir

    raw = await file.read()
    tmp = unique_character_path(Path(file.filename or "import.vtm").stem, dest_dir=characters_dir())
    # Write to a sidecar tmp so add_character can copy/validate without clobbering.
    staging = refs_dir() / f"character_import_{tmp.stem}.vtm"
    staging.write_bytes(raw)
    try:
        result = await asyncio.to_thread(get_runtime().add_character, staging)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        try:
            staging.unlink()
        except OSError:
            pass
    return {"ok": True, **result}


@app.post("/api/characters/load")
def character_load(body: CharacterIdBody) -> dict[str, Any]:
    try:
        result = get_runtime().load_character(body.id, repair=bool(body.repair))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


@app.post("/api/characters/remove")
def character_remove(body: CharacterIdBody) -> dict[str, Any]:
    try:
        result = get_runtime().remove_character(body.id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


@app.post("/api/characters/rename")
def character_rename(body: CharacterRenameBody) -> dict[str, Any]:
    try:
        result = get_runtime().rename_character(body.id, body.name)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


@app.get("/api/models/status")
def models_status_api() -> dict[str, Any]:
    from .model_download import models_status

    return models_status()


@app.get("/api/models/catalog")
def models_catalog() -> dict[str, Any]:
    from .model_download import hub_catalog_offers

    return {"offers": hub_catalog_offers()}


@app.get("/api/models/download/status")
def models_download_status() -> dict[str, Any]:
    from .model_download import download_status

    return download_status()


@app.post("/api/models/download")
def models_download_start(body: DownloadBody | None = None) -> dict[str, Any]:
    from .model_download import start_model_download

    names: list[str] = []
    if body is not None:
        if body.name:
            names.append(body.name)
        names.extend(n for n in (body.names or []) if n)
    return start_model_download(names or None)


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


@app.post("/api/limiters/fit")
def fit_limiters() -> dict[str, Any]:
    """Fit the limiters to the loaded character's still (Track Lab does the fit)."""
    try:
        return get_runtime().fit_character_limiters()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/lab/status")
def lab_status() -> dict[str, Any]:
    # Read before the harness GET: a fit edit landing meanwhile makes this packet stale.
    try:
        fit_gen: int | None = get_runtime().fit_generation()
    except Exception:
        fit_gen = None
    packet = lab_harness.status()
    try:
        if packet.get("online"):
            rt = get_runtime()
            rt.adopt_lab_travel_box(packet)
            # Live overlay is copied on the track thread. Doing it here too
            # stacks harness GETs on the desk API and makes the face lag.
            if not bool(getattr(rt, "_tracking", False)):
                rt.adopt_lab_overlay(packet, emit=True, fit_gen=fit_gen)
    except Exception:
        pass
    return packet


@app.post("/api/lab/connect")
def lab_connect() -> dict[str, Any]:
    from .lab_process import CONNECT_WAIT, connect_lab

    packet = connect_lab(timeout=CONNECT_WAIT)
    try:
        if packet.get("online"):
            get_runtime().adopt_lab_travel_box(packet)
    except Exception:
        pass
    return packet


@app.post("/api/lab/command")
def lab_command(body: LabCommandBody) -> dict[str, Any]:
    print(f"[lab-harness] api op={body.op} body={body.body}", flush=True)
    result = lab_harness.command(body.op, body.body)
    if not result.get("online", True) and not result.get("ok"):
        raise HTTPException(503, str(result.get("error") or "Track Lab is not running"))
    if body.op == "calibrate":
        try:
            get_runtime().apply_lab_calibrate(result)
        except Exception:
            pass
    if body.op == "set_input":
        try:
            restarted = get_runtime().restart_tracking_on_input(str(body.body.get("source") or ""))
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        if restarted is not None:
            result = restarted
    return result


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


@app.post("/api/stream/pause")
def stream_pause() -> dict[str, Any]:
    get_runtime().pause_stream()
    return get_runtime().status()


@app.post("/api/stream/resume")
def stream_resume() -> dict[str, Any]:
    try:
        get_runtime().resume_stream()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
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


@app.post("/api/pose/freeze")
def pose_freeze() -> dict[str, Any]:
    try:
        return get_runtime().freeze_pose()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pose/unfreeze")
def pose_unfreeze() -> dict[str, Any]:
    return get_runtime().unfreeze_pose()


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
        rt = get_runtime()
        await ws.send_json({"type": "status", "status": rt.status()})
        frame = rt.current_frame_event()
        if frame is not None:
            await ws.send_json(frame)
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
