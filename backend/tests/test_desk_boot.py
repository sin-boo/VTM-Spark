from backend.desk_boot import (
    new_boot_state,
    overall_progress,
    patch_boot_stage,
    previous_load_target,
    snapshot_boot,
    STAGE_WEIGHTS,
)


def test_previous_load_target_character_pack() -> None:
    kind, ident = previous_load_target(
        {"character_path": "characters/Gigi-Mood.vtm", "reference_path": "x.png"}
    )
    assert kind == "character"
    assert ident == "Gigi-Mood"


def test_previous_load_target_still() -> None:
    kind, ident = previous_load_target({"reference_path": "models/refs/upload.png"})
    assert kind == "still"
    assert ident.endswith("upload.png")


def test_previous_load_target_none() -> None:
    assert previous_load_target({}) == ("none", "")
    assert previous_load_target(None) == ("none", "")


def test_boot_stage_patch() -> None:
    state = new_boot_state()
    snap = patch_boot_stage(state, "lab", stage_state="run", progress=0.4, label="Handshake")
    assert snap["stages"]["lab"]["state"] == "run"
    assert snap["stages"]["lab"]["progress"] == 0.4
    assert snap["stages"]["model"]["state"] == "idle"
    assert snap["awaiting"] == ""
    copied = snapshot_boot(state)
    copied["stages"]["lab"]["progress"] = 0.0
    assert state["stages"]["lab"]["progress"] == 0.4


def test_boot_wait_character_fields() -> None:
    state = new_boot_state()
    state["awaiting"] = "character"
    state["suggested"] = "Gigi-Mood"
    patch_boot_stage(state, "character", stage_state="wait", progress=0.08, label="Select a character")
    snap = snapshot_boot(state)
    assert snap["awaiting"] == "character"
    assert snap["suggested"] == "Gigi-Mood"
    assert snap["stages"]["character"]["state"] == "wait"


def test_overall_progress_weights_running_stage() -> None:
    state = new_boot_state()
    patch_boot_stage(state, "model", stage_state="done", progress=1.0, label="Model ready")
    patch_boot_stage(state, "character", stage_state="run", progress=0.5, label="Loading character")
    frac, label = overall_progress(state)
    expected = STAGE_WEIGHTS["model"] * 1.0 + STAGE_WEIGHTS["character"] * 0.5
    assert abs(frac - expected) < 1e-6
    assert label == "Loading character"
    snap = snapshot_boot(state)
    assert abs(snap["progress"] - expected) < 1e-6
    assert snap["progress_label"] == "Loading character"


def test_overall_prefers_model_over_lab() -> None:
    state = new_boot_state()
    patch_boot_stage(state, "model", stage_state="run", progress=0.4, label="Loading model")
    patch_boot_stage(state, "lab", stage_state="run", progress=0.2, label="Connecting Track Lab")
    _frac, label = overall_progress(state)
    assert label == "Loading model"

