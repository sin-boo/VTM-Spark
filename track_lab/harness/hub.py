"""In-process pub/sub for harness frames, status, and commands.

Camera threads publish. WebSocket / HTTP consumers subscribe. A slow client
drops old frames instead of stalling tracking.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

Handler = Callable[[dict[str, Any]], dict[str, Any]]
Listener = Callable[[dict[str, Any] | None], None]


class LatestSlot:
    """One mailbox. ``put`` overwrites; ``take`` returns the newest item.

    Camera / IPC producers can outrun tracking. Keeping only the latest
    packet means a slow consumer never replays stale motion.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Any = None
        self._has = threading.Event()

    def put(self, item: Any) -> None:
        with self._lock:
            self._item = item
            self._has.set()

    def take(self, timeout: float | None = None) -> Any:
        if not self._has.wait(timeout):
            return None
        with self._lock:
            item = self._item
            self._item = None
            self._has.clear()
        return item


class HarnessHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: dict[str, Any] | None = None
        self._status: dict[str, Any] | None = None
        self._listeners: list[Listener] = []
        self._offs: dict[int, Callable[[], None]] = {}
        self._handler: Handler | None = None

    def set_handler(self, handler: Handler | None) -> None:
        with self._lock:
            self._handler = handler

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._listeners)

    def latest_frame(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._frame is None else dict(self._frame)

    def latest_status(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._status is None else dict(self._status)

    def listen(self, fn: Listener) -> Callable[[], None]:
        """Call ``fn`` on every packet. Returns an unsubscribe function."""
        with self._lock:
            self._listeners.append(fn)
            status = None if self._status is None else dict(self._status)
            frame = None if self._frame is None else dict(self._frame)
        if status is not None:
            fn(status)
        if frame is not None:
            fn(frame)

        def off() -> None:
            with self._lock:
                if fn in self._listeners:
                    self._listeners.remove(fn)

        return off

    def subscribe(self) -> queue.Queue[dict[str, Any] | None]:
        q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=2)

        def fn(packet: dict[str, Any] | None) -> None:
            _offer(q, packet)

        off = self.listen(fn)
        self._offs[id(q)] = off
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any] | None]) -> None:
        off = self._offs.pop(id(q), None)
        if off is not None:
            off()
        _offer(q, None)

    def publish(self, packet: dict[str, Any]) -> None:
        kind = str(packet.get("type") or "")
        with self._lock:
            packet = dict(packet)
            packet["clients"] = len(self._listeners)
            if kind == "status":
                self._status = packet
            elif kind == "frame":
                self._frame = packet
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(dict(packet))
            except Exception:
                continue

    def command(self, raw: object) -> dict[str, Any]:
        from .protocol import ack, parse_command

        try:
            msg = parse_command(raw)
        except ValueError as exc:
            return ack(ok=False, error=str(exc))
        if msg["op"] == "ping":
            return ack(ident=msg["id"], ok=True, status=self.latest_status())
        with self._lock:
            handler = self._handler
        if handler is None:
            return ack(ident=msg["id"], ok=False, error="harness not bound")
        try:
            return handler(msg)
        except Exception as exc:
            return ack(ident=msg["id"], ok=False, error=str(exc))


def _offer(q: queue.Queue[dict[str, Any] | None], item: dict[str, Any] | None) -> None:
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


hub = HarnessHub()
