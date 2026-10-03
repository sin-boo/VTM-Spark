from datetime import datetime, timedelta, timezone

from backend.model_download import NEW_MODEL_DAYS, hub_catalog_offers


def test_catalog_hides_owned_and_ages_out(tmp_path, monkeypatch) -> None:
    dit = tmp_path / "dit"
    dit.mkdir()
    (dit / "VTM-ELF.pt").write_bytes(b"x" * 8)
    monkeypatch.setattr("backend.model_download.models_dir", lambda: dit)
    monkeypatch.setattr(
        "backend.model_download.load_model_sources",
        lambda: {"enabled": True, "hf_repo": "unit/test"},
    )
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    listing = [
        {
            "name": "VTM-ELF.pt",
            "hf": "VTM-ELF.pt",
            "dest": "models/dit/VTM-ELF.pt",
            "published": now - timedelta(days=2),
        },
        {
            "name": "VTM-2.0.pt",
            "hf": "VTM-2.0.pt",
            "dest": "models/dit/VTM-2.0.pt",
            "published": now - timedelta(days=5),
        },
        {
            "name": "VTM-old.pt",
            "hf": "VTM-old.pt",
            "dest": "models/dit/VTM-old.pt",
            "published": now - timedelta(days=NEW_MODEL_DAYS + 1),
        },
    ]
    offers = hub_catalog_offers(now=now, listing=listing)
    names = [row["name"] for row in offers]
    assert "VTM-ELF.pt" not in names
    fresh = next(row for row in offers if row["name"] == "VTM-2.0.pt")
    aged = next(row for row in offers if row["name"] == "VTM-old.pt")
    assert fresh["is_new"] is True
    assert fresh["badge"] == "New"
    assert aged["is_new"] is False
    assert aged["badge"] == "Available"


def test_catalog_off_when_sources_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.model_download.load_model_sources",
        lambda: {"enabled": False},
    )
    assert hub_catalog_offers(listing=[{"name": "x.pt"}]) == []


def test_auto_download_checklist_items_have_a_source() -> None:
    from backend.model_checklist import CHECKLIST
    from backend.model_download import _asset_jobs, load_model_sources

    sources = load_model_sources()
    assert sources.get("enabled")
    dests = {str(job["dest"]) for job in _asset_jobs(sources)}
    missing = [
        item.candidates[0]
        for item in CHECKLIST
        if item.auto_download and item.candidates[0] not in dests
    ]
    assert missing == []


def test_http_sources_are_https_with_dest() -> None:
    from backend.model_download import _asset_jobs, load_model_sources

    http_jobs = [j for j in _asset_jobs(load_model_sources()) if j["kind"] == "http"]
    assert http_jobs
    for job in http_jobs:
        assert str(job["url"]).startswith("https://")
        assert str(job["dest"]).startswith("models/trackers/")
        assert int(job["min_bytes"]) >= 1_000_000


def test_checklist_reports_the_fast_decoder(tmp_path, monkeypatch) -> None:
    from backend import model_checklist as mc
    from backend.engine import fast_decoder_path
    from backend.paths import package_root

    item = next(x for x in mc.CHECKLIST if x.id == "fast_decoder")
    # Downloaded with the models; optional (the stream falls back to TinyVAE).
    assert not item.required and item.auto_download
    assert item.candidates[0] == fast_decoder_path().relative_to(package_root()).as_posix()

    dit = tmp_path / "models" / "dit"
    dit.mkdir(parents=True)
    monkeypatch.setattr(mc, "package_root", lambda: tmp_path)
    monkeypatch.setattr(mc, "models_dir", lambda: dit)

    def entry() -> dict:
        return next(x for x in mc.scan_models()["items"] if x["id"] == "fast_decoder")

    assert entry()["ok"] is False
    path = tmp_path / item.candidates[0]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * item.min_bytes)
    assert entry()["ok"] is True
    assert entry()["bytes"] == item.min_bytes
