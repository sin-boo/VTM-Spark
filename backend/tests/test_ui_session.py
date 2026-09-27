
from backend.model_download import _is_dit_weight_name, hub_checkpoint_names
from backend.ui_session import load_ui_session, save_ui_session, session_path


def test_session_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.ui_session.models_root", lambda: tmp_path)
    save_ui_session(
        checkpoint="models/dit/VTM-1.5.1.pt",
        reference_path="models/refs/upload.png",
        character_path="characters/gigi.vtm",
        hub_files=["VTM-ELF.pt"],
    )
    assert session_path() == tmp_path / "session.json"
    st = load_ui_session()
    assert st["checkpoint"] == "models/dit/VTM-1.5.1.pt"
    assert st["reference_path"] == "models/refs/upload.png"
    assert st["character_path"] == "characters/gigi.vtm"
    assert st["hub_files"] == ["VTM-ELF.pt"]


def test_session_stores_travel_box(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.ui_session.models_root", lambda: tmp_path)
    save_ui_session(travel_box={"enabled": True, "left": 0.1, "right": 0.2})
    st = load_ui_session()
    assert st["travel_box"]["left"] == 0.1
    assert st["travel_box"]["right"] == 0.2


def test_session_stores_compile_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.ui_session.models_root", lambda: tmp_path)
    save_ui_session(compile_model=True)
    st = load_ui_session()
    assert st["compile_model"] is True
    save_ui_session(compile_model=False)
    st = load_ui_session()
    assert st["compile_model"] is False


def test_session_stores_model_guidance(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.ui_session.models_root", lambda: tmp_path)
    save_ui_session(
        steps=4,
        pose_cfg=2.5,
        id_cfg=0.8,
        frame_blend=0.9,
        inbetweens=2,
        interpolate=False,
        hold_last=False,
    )
    st = load_ui_session()
    assert st["steps"] == 4
    assert st["pose_cfg"] == 2.5
    assert st["id_cfg"] == 0.8
    assert st["frame_blend"] == 0.9
    assert st["inbetweens"] == 2
    assert st["interpolate"] is False
    assert st["hold_last"] is False


def test_hub_vs_local_names(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.ui_session.models_root", lambda: tmp_path)
    save_ui_session(hub_files=["VTM-ELF.pt"])
    names = hub_checkpoint_names()
    assert "VTM-ELF.pt" in names
    assert "VTM-Elf_2.0_checkpoints_VTM-Elf_2.0-000450000.pt" not in names
    assert _is_dit_weight_name("VTM-1.5.1.pt")
    assert not _is_dit_weight_name("readme.md")


def test_list_stream_checkpoints_reads_available_files(tmp_path, monkeypatch) -> None:
    from backend.engine import list_stream_checkpoints

    dit = tmp_path / "dit"
    dit.mkdir()
    (dit / "VTM-ELF.pt").write_bytes(b"x" * 1_000_001)
    (dit / "VTM-1.5.1.pt").write_bytes(b"x" * 1_000_001)
    nested = dit / "pack"
    nested.mkdir()
    (nested / "nested-run.pt").write_bytes(b"x" * 1_000_001)
    (dit / "tiny.pt").write_bytes(b"nope")
    (dit / "readme.md").write_text("skip", encoding="utf-8")
    monkeypatch.setattr("backend.engine.models_dir", lambda: dit)
    monkeypatch.setattr(
        "backend.engine._load_extra_checkpoint_state",
        lambda: {"dirs": [], "last_dir": ""},
    )
    labels = [label for label, _ in list_stream_checkpoints()]
    assert labels == ["nested-run", "VTM-1.5.1", "VTM-ELF"]


def test_default_checkpoint_uses_last_session(tmp_path, monkeypatch) -> None:
    from backend.engine import default_stream_checkpoint, is_stream_checkpoint_file

    ckpt = tmp_path / "custom.pt"
    ckpt.write_bytes(b"x" * 1_000_001)
    assert is_stream_checkpoint_file(ckpt)
    monkeypatch.setattr("backend.ui_session.models_root", lambda: tmp_path)
    save_ui_session(checkpoint=str(ckpt))
    assert default_stream_checkpoint() == ckpt.resolve()
