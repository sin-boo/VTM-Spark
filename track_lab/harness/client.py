"""HTTP client for the Track Lab harness.

Use this from another process (VTM Noble) to read tracking packets and
change settings. Live push is `ws://127.0.0.1:8780/harness/ws`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:8780/harness"


class HarnessClient:
    def __init__(self, base: str = DEFAULT_BASE, timeout: float = 2.0) -> None:
        self.base = str(base).rstrip("/")
        self.timeout = float(timeout)

    def status(self) -> dict[str, Any]:
        payload = self._get("/status")
        return payload if isinstance(payload, dict) else {}

    def frame(self) -> dict[str, Any] | None:
        return self._get("/frame", allow_empty=True)

    def command(self, op: str, body: dict[str, Any] | None = None, *, ident: object = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": op, "body": body or {}}
        if ident is not None:
            payload["id"] = ident
        result = self._post("/command", payload)
        return result if isinstance(result, dict) else {"ok": False, "error": "empty reply"}

    def set_feel(self, **values: object) -> dict[str, Any]:
        return self.command("set_feel", dict(values))

    def set_travel(self, **values: object) -> dict[str, Any]:
        return self.command("set_travel", dict(values))

    def start(self, **opts: object) -> dict[str, Any]:
        return self.command("start", dict(opts))

    def stop(self) -> dict[str, Any]:
        return self.command("stop")

    def _get(self, path: str, *, allow_empty: bool = False) -> dict[str, Any] | None:
        req = urllib.request.Request(self.base + path, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                if res.status == 204:
                    return None
                raw = res.read()
        except urllib.error.HTTPError as exc:
            if allow_empty and exc.code == 204:
                return None
            raise
        if not raw:
            return None if allow_empty else {}
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as res:
            raw = res.read()
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
