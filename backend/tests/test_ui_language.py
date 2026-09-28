import json
import re
from pathlib import Path

import pytest

from backend import ui_prefs
from backend.desk_splash import early_splash_html
from backend.ui_prefs import load_ui_prefs, save_ui_prefs, ui_language

UI = Path(__file__).resolve().parents[2] / "ui" / "src"


def _ui(*parts: str) -> str:
    return UI.joinpath(*parts).read_text(encoding="utf-8")


def test_prefs_start_blank_and_save_a_language(tmp_path: Path) -> None:
    path = tmp_path / "ui_prefs.json"
    assert load_ui_prefs(path) == {"language": ""}
    assert save_ui_prefs({"language": "ja"}, path) == {"language": "ja"}
    assert json.loads(path.read_text(encoding="utf-8")) == {"language": "ja"}
    assert load_ui_prefs(path) == {"language": "ja"}
    assert ui_language(path) == "ja"


def test_prefs_reject_unknown_languages(tmp_path: Path) -> None:
    path = tmp_path / "ui_prefs.json"
    save_ui_prefs({"language": "en"}, path)
    with pytest.raises(ValueError):
        save_ui_prefs({"language": "fr"}, path)
    assert load_ui_prefs(path) == {"language": "en"}


def test_prefs_survive_a_broken_file(tmp_path: Path) -> None:
    path = tmp_path / "ui_prefs.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_ui_prefs(path) == {"language": ""}
    path.write_text(json.dumps({"language": "xx"}), encoding="utf-8")
    assert load_ui_prefs(path) == {"language": ""}


def test_unset_language_follows_the_os(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui_prefs, "system_language", lambda: "ja")
    assert ui_language(tmp_path / "missing.json") == "ja"


def test_api_exposes_ui_prefs() -> None:
    src = (Path(__file__).resolve().parents[1] / "api.py").read_text(encoding="utf-8")
    assert '@app.get("/api/ui-prefs")' in src
    assert '@app.post("/api/ui-prefs")' in src
    api = _ui("api.ts")
    assert "fetch('/api/ui-prefs')" in api


def test_native_splash_speaks_japanese() -> None:
    page = early_splash_html(api_origin="http://127.0.0.1:8765", language="ja")
    assert '<html lang="ja">' in page
    assert "起動中…" in page
    assert '"Loading model": "モデルを読み込み中"' in page
    assert "tr(label" in page


def test_native_splash_stays_english() -> None:
    page = early_splash_html(api_origin="http://127.0.0.1:8765", language="en")
    assert '<html lang="en">' in page
    assert "const WORDS = {};" in page
    assert "Starting…" in page


def test_settings_has_a_language_picker() -> None:
    rail = _ui("components", "ControlRail.tsx")
    app = _ui("App.tsx")
    words = _ui("i18n.ts")
    settings = rail.split("railPane === 'desk' ? (")[1].split(") : (", 1)[1]
    assert "t('lang.title')" in settings
    assert "LANGUAGES.map" in settings
    assert "props.onLanguage(row.id)" in settings
    assert "setLang(next)" in app
    assert "{ id: 'en', label: 'English' }" in words
    assert "{ id: 'ja', label: '日本語' }" in words
    assert "LanguageProvider" in _ui("main.tsx")


def _table(words: str, name: str) -> dict[str, str]:
    start = words.index(f"const {name}")
    body = words[start : words.index("\n}\n", start)]
    rows = re.findall(r"""^  '([\w.]+)':\s*\n?\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")""", body, re.M)
    return {key: single or double for key, single, double in rows}


def test_every_message_has_japanese() -> None:
    words = _ui("i18n.ts")
    en = _table(words, "en = {")
    ja = _table(words, "ja: Record<MessageKey, string> = {")
    assert len(en) > 250
    assert set(en) == set(ja)
    # Placeholders must survive translation.
    for key, text in en.items():
        assert set(re.findall(r"\{\w+\}", text)) == set(re.findall(r"\{\w+\}", ja[key])), key


def test_backstage_lines_line_up() -> None:
    words = _ui("i18n.ts")
    block = words[words.index("export const BACKSTAGE_LINES") :]
    en = block[block.index("en: [") : block.index("],", block.index("en: ["))]
    ja = block[block.index("ja: [") : block.index("],", block.index("ja: ["))]
    count = lambda part: len(re.findall(r"^\s+'", part, re.M))  # noqa: E731
    assert count(en) == count(ja) > 40


def test_japanese_font_fallback() -> None:
    css = _ui("styles", "tokens.css")
    assert ":root:lang(ja)" in css
    assert "Yu Gothic UI" in css


def test_first_run_asks_for_a_language_after_the_splash() -> None:
    app = _ui("App.tsx")
    provider = _ui("LanguageProvider.tsx")
    pick = _ui("components", "LanguagePick.tsx")
    # Splash until loading is done and we know if a language was ever saved.
    assert "if (!deskReady || picked === null) {" in app
    assert app.index("<Splash") < app.index("<LanguagePick />") < app.index('className="desk-shell"')
    assert "if (!picked) {" in app
    # Blank saved language = first run; an old backend never blocks the desk.
    assert "setPicked(isLang(prefs.language))" in provider
    assert "setPicked(true)" in provider
    # Every line shows in both languages.
    assert "messageIn('en', 'pick.title')" in pick
    assert "messageIn('ja', 'pick.title')" in pick
    assert "LANGUAGES.map" in pick
    assert "setLang(next)" in pick
    words = _ui("i18n.ts")
    assert "'pick.title': 'Choose your language'" in words
    assert "'pick.title': '言語を選択してください'" in words
