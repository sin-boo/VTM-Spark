from pathlib import Path

from backend.paths import characters_dir, display_path, resolve_user_path


def test_characters_dir_is_package_relative(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.paths.package_root", lambda: tmp_path)
    folder = characters_dir()
    assert folder == tmp_path / "characters"
    assert folder.is_dir()
    assert display_path(folder / "gigi.vtm") == "characters/gigi.vtm"


def test_resolve_uses_characters_not_models(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.paths.package_root", lambda: tmp_path)
    pack = tmp_path / "characters" / "gigi.vtm"
    pack.parent.mkdir()
    pack.write_bytes(b"vtm")
    assert resolve_user_path("characters/gigi.vtm") == pack.resolve()
    assert resolve_user_path("models/characters/gigi.vtm") == pack.resolve()


def test_migrate_legacy_models_characters(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.paths.package_root", lambda: tmp_path)
    legacy = tmp_path / "models" / "characters"
    nested = legacy / "gigi"
    nested.mkdir(parents=True)
    (legacy / "gigi.vtm").write_bytes(b"vtm")
    (nested / "preview.png").write_bytes(b"png")
    (legacy / ".gitkeep").write_bytes(b"")
    folder = characters_dir()
    assert (folder / "gigi.vtm").read_bytes() == b"vtm"
    assert (folder / "gigi" / "preview.png").read_bytes() == b"png"
    assert not (tmp_path / "models" / "characters").exists()
