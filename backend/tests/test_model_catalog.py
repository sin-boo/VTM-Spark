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
