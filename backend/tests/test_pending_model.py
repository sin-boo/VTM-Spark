"""Picking a model only selects it; Start stream / Generate load it."""

from __future__ import annotations

import threading
from pathlib import Path

from backend.stream import StreamRuntime


def _runtime(tmp_path: Path, monkeypatch, *, loaded: Path) -> tuple[StreamRuntime, list[str]]:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._model_load_lock = threading.Lock()
    rt._pending_checkpoint = None
    rt._status = {"checkpoint": loaded.stem, "pending_checkpoint": "", "message": ""}

    def _set(**kw: object) -> None:
        rt._status.update(kw)

    rt._set_status = _set  # type: ignore[method-assign]

    class _Engine:
        checkpoint = loaded
        _ready = True

    rt.engine = _Engine()  # type: ignore[assignment]
    loads: list[str] = []
    rt.set_checkpoint = lambda path: loads.append(Path(path).name)  # type: ignore[method-assign]
    monkeypatch.setattr("backend.stream.is_stream_checkpoint_file", lambda _p: True)
    monkeypatch.setattr("backend.stream.remember_checkpoint_location", lambda _p: None)
    monkeypatch.setattr("backend.ui_session.save_ui_session", lambda **_k: None)
    return rt, loads


def test_select_checkpoint_does_not_load(tmp_path: Path, monkeypatch) -> None:
    old = tmp_path / "old.pt"
    new = tmp_path / "new.pt"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    rt, loads = _runtime(tmp_path, monkeypatch, loaded=old)

    StreamRuntime.select_checkpoint(rt, new)

    assert loads == []
    assert rt._pending_checkpoint == new.resolve()
    assert rt._status["pending_checkpoint"] == "new"
    assert "Start stream" in str(rt._status["message"])


def test_reselecting_loaded_model_clears_pending(tmp_path: Path, monkeypatch) -> None:
    old = tmp_path / "old.pt"
    new = tmp_path / "new.pt"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    rt, loads = _runtime(tmp_path, monkeypatch, loaded=old.resolve())

    StreamRuntime.select_checkpoint(rt, new)
    StreamRuntime.select_checkpoint(rt, old)

    assert loads == []
    assert rt._pending_checkpoint is None
    assert rt._status["pending_checkpoint"] == ""


def test_ensure_model_loads_pending_first(tmp_path: Path, monkeypatch) -> None:
    old = tmp_path / "old.pt"
    new = tmp_path / "new.pt"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    rt, loads = _runtime(tmp_path, monkeypatch, loaded=old)
    StreamRuntime.select_checkpoint(rt, new)

    StreamRuntime._apply_pending_checkpoint(rt)

    assert loads == ["new.pt"]


def test_start_and_generate_go_through_ensure_model() -> None:
    import inspect

    assert "_apply_pending_checkpoint()" in inspect.getsource(StreamRuntime.ensure_model)
    assert "self.ensure_model(" in inspect.getsource(StreamRuntime.start_stream)
    assert "self.ensure_model(" in inspect.getsource(StreamRuntime.generate_once)
