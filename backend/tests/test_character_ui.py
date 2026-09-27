from pathlib import Path


def _ui(*parts: str) -> str:
    return (Path(__file__).resolve().parents[2] / "ui" / "src").joinpath(*parts).read_text(
        encoding="utf-8"
    )


def test_toon_preview_opens_library() -> None:
    lib = _ui("components", "CharacterLibrary.tsx")
    rail = _ui("components", "ControlRail.tsx")
    css = _ui("App.css")
    assert "char-preview" in lib
    assert "char-dock" not in lib
    assert "No character" in lib
    assert "Click to add" in lib
    assert "libraryOpen" in lib
    assert "setLibraryOpen(true)" in lib
    assert "pickFile()" not in lib.split("function openDock")[1].split("async function beginCreate")[0]
    assert "                    Create" in lib
    ctx = lib.split('className="char-ctx"')[1]
    assert "Repair" in ctx
    assert "shapes_compatible === false" in ctx
    assert ctx.index("Edit") < ctx.index("Repair")
    assert ctx.index("Repair") < ctx.index("Rename")
    assert ctx.index("Rename") < ctx.index("Remove")
    assert "beginEdit" in lib
    assert "chooseCard" in lib
    assert "onDoubleClick" not in lib
    assert "props.onRefresh?.()" in lib
    assert "char-empty-well" in lib
    assert "pickRef" not in lib
    assert "libraryOpen" not in rail
    assert "Create character" not in rail
    assert "char-stage-open" not in rail
    assert rail.index("char-stage") < rail.index("<CharacterLibrary")
    assert rail.index("<MixMeters") > rail.index("<CharacterLibrary")
    assert rail.index("<MixMeters") < rail.index("stream-panel")
    preview = css.split(".char-preview {")[1].split("}")[0]
    img = css.split(".char-preview img {")[1].split("}")[0]
    assert "168px" not in preview
    assert "152px" not in preview
    assert "height: auto" in img
    assert "width: 100%" in img
    assert "object-fit: contain" in img
    assert "100cqh" not in css
    assert "container-type" not in css
    # The rail fits without a tall-screen spread; only the width steps remain.
    assert "@media (min-height: 880px)" not in css
    assert "--rail-width: 336px" in css
    assert "--rail-width: 380px" in css


def test_desk_settings_has_no_lab_overlay_flags() -> None:
    rail = _ui("components", "ControlRail.tsx")
    css = _ui("App.css")
    assert "LAB_OVERLAY" not in rail
    assert "use_visemes" not in rail
    assert "invert_look" not in rail
    assert "invert_pitch" not in rail
    assert 'label="Mirror"' in rail
    assert 'label="Invert"' not in rail
    assert "Start Track Lab to edit these." not in rail
    assert 'group-subtitle">Lab</h3>' not in rail
    assert ".overlay-controls .group-subtitle" not in css


def test_cel_stage_resets_zoom_when_character_changes() -> None:
    stage = _ui("components", "CelStage.tsx")
    app = _ui("App.tsx")
    assert "stillId?: string" in stage
    assert "[stillId]" in stage
    assert "stillId={String(status?.character_id || status?.reference_path || '')}" in app


def test_virtual_cam_has_no_obs_hint() -> None:
    rail = _ui("components", "ControlRail.tsx")
    assert "set Resolution to Custom" not in rail
    assert "OBS → Video Capture Device" not in rail


def test_character_fit_screen() -> None:
    lib = _ui("components", "CharacterLibrary.tsx")
    fit = _ui("components", "CharacterFit.tsx")
    css = _ui("App.css")
    assert "CharacterFit" in lib
    assert "from './CharacterFit'" in lib
    assert "is-fit" in lib
    assert "Hair" in fit
    assert "Skeleton" in fit
    assert "Limiters" in fit
    assert ".char-create.is-fit" in css
    assert ".char-fit" in css
    assert ".fit-frame" in css
