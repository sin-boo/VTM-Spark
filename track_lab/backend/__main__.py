"""python -m backend — start the face-tracking lab API."""

from __future__ import annotations

import time

from .bind import (
    already_running_message,
    kill_listeners,
    occupied_message,
    probe_state,
    stale_message,
    _pause,
)
from .ports import HOST, PORT


def _wait_free(seconds: float = 8.0) -> str:
    deadline = time.monotonic() + seconds
    state = probe_state(HOST, PORT)
    while state != "free" and time.monotonic() < deadline:
        time.sleep(0.2)
        state = probe_state(HOST, PORT)
    return state


def main() -> int:
    print(f"[track-lab] probing {HOST}:{PORT}", flush=True)
    state = probe_state(HOST, PORT)
    print(
        f"[track-lab] probe={state} health=http://{HOST}:{PORT}/api/health "
        f"harness=http://{HOST}:{PORT}/harness/status",
        flush=True,
    )
    if state == "stale":
        print(f"[track-lab] {stale_message(PORT)}", flush=True)
        killed = kill_listeners(PORT)
        print(f"[track-lab] killed stale pids={killed or '—'}", flush=True)
        state = _wait_free()
        print(f"[track-lab] probe={state} after stale restart", flush=True)
    if state == "lab":
        _pause(already_running_message(HOST, PORT))
        return 0
    if state == "busy":
        print(f"[track-lab] {occupied_message(PORT)}", flush=True)
        _pause(occupied_message(PORT))
        return 2
    if state == "stale":
        print(f"[track-lab] {stale_message(PORT)} still bound", flush=True)
        _pause(stale_message(PORT))
        return 2

    print(
        f"[track-lab] binding http://{HOST}:{PORT}  harness=/harness  ws=/harness/ws",
        flush=True,
    )
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
