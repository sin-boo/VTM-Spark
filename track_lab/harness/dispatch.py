"""Map harness commands onto FaceBench methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hub import hub
from .pack import frame_from_bench, status_from_bench
from .protocol import ack


def bind(bench: Any) -> None:
    hub.set_handler(lambda msg: handle(bench, msg))


def handle(bench: Any, msg: dict[str, Any]) -> dict[str, Any]:
    op = str(msg.get("op") or "")
    body = msg.get("body") if isinstance(msg.get("body"), dict) else {}
    ident = msg.get("id")
    try:
        result = _call(bench, op, body)
    except Exception as exc:
        print(f"[harness] op={op} raised: {exc}", flush=True)
        return ack(ident=ident, ok=False, error=str(exc))
    payload = result if isinstance(result, dict) else bench.status()
    status = status_from_bench(bench, payload, clients=hub.clients)
    frame = frame_from_bench(bench, clients=hub.clients)
    hub.publish(status)
    hub.publish(frame)
    error = str(payload.get("error") or "") if isinstance(payload, dict) else ""
    print(
        f"[harness] op={op} ok={not error} error={error or '—'} "
        f"source={status.get('source') or '—'} live={bool(status.get('live'))}",
        flush=True,
    )
    return ack(ident=ident, ok=not error, error=error, status=status, frame=frame)


def _call(bench: Any, op: str, body: dict[str, Any]) -> Any:
    if op == "start":
        return bench.start_live(
            body.get("camera"),
            body.get("source"),
            body.get("host"),
            body.get("port"),
            body.get("mirror"),
        )
    if op == "stop":
        return bench.stop_live()
    if op == "track":
        return bench.track()
    if op == "set_source":
        path = Path(str(body.get("path") or ""))
        if not path.is_file():
            raise ValueError("set_source needs a readable file path")
        return bench.set_source(path.read_bytes(), path.name)
    if op == "reset":
        return bench.reset()
    if op == "set_camera":
        return bench.set_camera(int(body.get("index", 0)))
    if op == "set_input":
        return bench.set_source_mode(str(body.get("source", "camera")))
    if op == "set_ifm":
        host = body.get("host")
        port = body.get("port")
        return bench.set_ifm(
            str(host) if host is not None else None,
            int(port) if port is not None else None,
        )
    if op == "set_mirror":
        on = body.get("on")
        if on is None:
            on = body.get("mirror")
        return bench.set_mirror(bool(on))
    if op == "set_feel":
        return bench.set_feel(body)
    if op == "set_travel":
        return bench.set_travel(body)
    if op == "calibrate":
        return bench.start_calibrate(str(body.get("id", "")))
    if op == "reset_calibrate":
        return bench.reset_calibrate()
    if op == "apply_preset":
        return bench.apply_preset(str(body.get("id", "")))
    if op == "set_mouth":
        return bench.set_mouth(str(body.get("id", "")), body.get("mouth"))
    if op == "move_key":
        return bench.move_key(str(body.get("id", "")), body.get("t"))
    if op == "drop_key":
        return bench.drop_key(str(body.get("id", "")))
    if op == "set_mouth_point":
        return bench.set_mouth_point(body)
    if op == "set_eye_point":
        return bench.set_eye_point(body)
    if op == "set_skeleton_point":
        return bench.set_skeleton_point(body)
    if op == "set_hair":
        return bench.set_hair(body)
    if op == "set_point":
        return bench.set_point(body)
    if op == "reset_points":
        return bench.reset_points(body)
    if op == "generate":
        from backend.vtm_gen import generate as run_generate

        return run_generate(bench, body)
    if op == "record":
        return bench.record_movement(bool(body.get("on")))
    raise ValueError(f"unknown op '{op}'")
