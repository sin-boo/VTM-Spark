"""HTTP + WebSocket surface for the harness.

GET  /harness/status   control surface
GET  /harness/frame    latest tracking packet
POST /harness/command  {op, body?, id?}
WS   /harness/ws       push frames + status, accept commands
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect

from .dispatch import bind
from .hub import hub
from .pack import frame_from_bench, status_from_bench, warming_frame, warming_status

router = APIRouter()
_bench: Any = None


def _plain(value: Any) -> Any:
    """What FastAPI's encoder would have turned into JSON: numpy values,
    paths, sets."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _json(packet: dict[str, Any]) -> Response:
    """Packets are plain JSON already. Returning the dict ran FastAPI's
    generic encoder over every keypoint and hair vertex (~3 ms a poll, on
    the desk's path between two DiT calls)."""
    body = json.dumps(
        packet, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=_plain
    )
    return Response(content=body, media_type="application/json")


def attach(app: Any, bench: Any = None) -> None:
    """Mount routes. ``bench`` is the in-process tracker; omit it for the host."""
    global _bench
    _bench = bench
    if bench is not None:
        bind(bench)
        try:
            hub.publish(status_from_bench(bench, clients=hub.clients))
            hub.publish(frame_from_bench(bench, clients=hub.clients))
        except Exception:
            pass
    else:
        from .bridge import install_host

        install_host()
    app.include_router(router, prefix="/harness", tags=["harness"])
    print("[harness] mounted GET /harness/status /frame  POST /harness/command  WS /harness/ws", flush=True)


@router.get("/status")
def status() -> Any:
    packet = hub.latest_status()
    if packet is None and _bench is not None:
        packet = status_from_bench(_bench, clients=hub.clients)
        hub.publish(packet)
    if packet is None:
        packet = warming_status(clients=hub.clients)
        hub.publish(packet)
    packet = dict(packet)
    packet["clients"] = hub.clients
    return _json(packet)


@router.get("/frame")
def frame() -> Any:
    packet = hub.latest_frame()
    if packet is None and _bench is not None:
        packet = frame_from_bench(_bench, clients=hub.clients)
        hub.publish(packet)
    if packet is None:
        packet = warming_frame(clients=hub.clients)
        hub.publish(packet)
    packet = dict(packet)
    packet["clients"] = hub.clients
    return _json(packet)


@router.post("/command")
def command(body: dict[str, Any]) -> dict[str, Any]:
    """Sync so start/track cannot freeze GET /status and /frame."""
    return hub.command(body)


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=2)

    def on_packet(packet: dict[str, Any] | None) -> None:
        def push() -> None:
            if incoming.full():
                try:
                    incoming.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                incoming.put_nowait(packet)
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(push)

    off = hub.listen(on_packet)

    async def writer() -> None:
        while True:
            packet = await incoming.get()
            if packet is None:
                return
            await ws.send_json(packet)

    pump = asyncio.create_task(writer())
    try:
        while True:
            try:
                text = await ws.receive_text()
            except WebSocketDisconnect:
                break
            raw: object = "ping" if text == "ping" else text
            if text and text != "ping":
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError:
                    raw = text
            await ws.send_json(hub.command(raw))
    except WebSocketDisconnect:
        pass
    finally:
        off()
        try:
            incoming.put_nowait(None)
        except asyncio.QueueFull:
            pass
        pump.cancel()
        try:
            await pump
        except (asyncio.CancelledError, Exception):
            pass
