"""The character bar must start from zero and move through real steps."""

import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend import character_pack
from backend.stream import StreamRuntime


def _runtime() -> StreamRuntime:
    rt = object.__new__(StreamRuntime)
    rt._lock = threading.Lock()
    rt._status = {}
    rt._set_status = lambda **kw: rt._status.update(kw)
    return rt


def test_boot_leaves_no_stale_progress() -> None:
    rt = _runtime()
    rt._boot_lock = threading.Lock()
    rt._boot = {"running": True, "ready": False}
    rt.boot_snapshot = lambda: dict(rt._boot)

    def lab_done() -> None:
        # The last boot stage reports 100%, as the real Track Lab handshake does.
        rt._status.update(progress=1.0, progress_kind="lab", progress_label="Track Lab connected")

    rt._boot_model = lambda: None
    rt._boot_character = lambda: None
    rt._boot_lab = lab_done
    rt._boot_lab_source = lambda: None
    StreamRuntime._run_boot(rt)
    assert rt._boot["ready"] is True
    assert rt._status["progress"] == 0.0
    assert rt._status["progress_kind"] == ""


def test_selecting_a_character_reports_real_steps(monkeypatch) -> None:
    rt = _runtime()
    steps: list[tuple[float, str]] = []
    bands: list[str] = []
    cleared: list[dict] = []
    rt._set_progress = lambda frac, *, label, kind, **_kw: steps.append((frac, label))
    rt._clear_progress = lambda **kw: cleared.append(kw)

    @contextmanager
    def band(**kw):
        bands.append(kw["label"])
        yield

    rt._run_progress_band = band
    rt._streaming = False
    rt._last_lab_hair = {"ok": True}
    rt._last_image = None
    rt._last_overlay_kps = None
    rt.engine = SimpleNamespace(
        _ready=True,
        _ref_keypoints=np.zeros((37, 4), dtype=np.float32),
        image_size=768,
        load_encoded_reference=lambda **_kw: None,
    )
    pack = SimpleNamespace(
        name="Mika",
        keypoints=None,
        ref_latent=None,
        ref_face_latent=None,
        skip_crop=False,
        image_size=768,
        preview_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(character_pack, "resolve_character_id", lambda ident: Path(f"{ident}.vtm"))
    monkeypatch.setattr(character_pack, "read_character_pack", lambda path: pack)
    for name in (
        "_migrate_character_pack",
        "_mark_current_character",
        "_apply_character_limiters",
        "_sync_lab_character",
        "_character_shape_gate",
    ):
        setattr(rt, name, lambda *a, **k: None)
    rt._pack_latents_usable = lambda _pack: True
    rt._cel_still = lambda rgb, **_kw: rgb
    rt._install_loaded_reference = lambda preview, kps: {"type": "frame"}
    rt._apply_character_fit = lambda: False
    rt._character_card_safe = lambda path: {"id": path.stem}
    rt.status = lambda: dict(rt._status)

    StreamRuntime.load_character(rt, "Mika", require_compatible=False)

    fractions = [frac for frac, _label in steps]
    assert len(steps) >= 4
    assert fractions == sorted(fractions)
    assert fractions[0] < 0.1
    assert "Fitting overlay…" in bands
    assert cleared and cleared[-1].get("busy") is False


def test_quiet_load_never_touches_the_bar(monkeypatch) -> None:
    import inspect

    src = inspect.getsource(StreamRuntime.load_character)
    assert "if not quiet:\n                self._set_progress(frac" in src
    assert "nullcontext()\n                if quiet" in src


def test_create_starts_from_an_empty_bar() -> None:
    import inspect

    src = inspect.getsource(StreamRuntime.apply_reference)
    start = src.index('message="Creating character…" if silent else "Applying reference…"')
    head = src[start : start + 200]
    assert "progress=0.0" in head
    assert 'progress_kind=""' in head


def test_create_window_ignores_other_bars() -> None:
    ui = Path(__file__).resolve().parents[2] / "ui" / "src" / "components"
    rail = (ui / "ControlRail.tsx").read_text(encoding="utf-8")
    lib = (ui / "CharacterLibrary.tsx").read_text(encoding="utf-8")
    assert "Math.max(createOwnsBar ? progress : 0, 0.04)" in rail
    assert "createPhase={modelBar ? 'model' : 'character'}" in rail
    assert "key={props.createPhase ?? 'character'}" in lib
