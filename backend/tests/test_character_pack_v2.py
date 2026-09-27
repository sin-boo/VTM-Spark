from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import character_pack as cp
from backend.character_pack import (
    BLENDSHAPES_NAME,
    FIT_NAME,
    KEYPOINTS_NAME,
    LATENTS_NAME,
    MANIFEST_NAME,
    POSE_KEYS_NAME,
    PREVIEW_NAME,
    THUMB_NAME,
    CharacterPackError,
    character_card,
    export_character_pack_bytes,
    pack_model_matches,
    peek_character_manifest,
    read_character_pack,
    read_character_thumb_png,
    read_pack_fit,
    read_pack_json,
    rename_character_pack,
    replace_pack_members,
    update_pack_meta,
    upgrade_character_pack,
    validate_character_pack,
    write_character_pack,
    write_pack_fit,
    write_pack_json,
)
from backend.engine import neutral_keypoints

REPO = Path(__file__).resolve().parents[2]


def _payload(**extra) -> dict:
    preview = np.zeros((300, 200, 3), dtype=np.uint8)
    preview[:, :] = (40, 80, 60)
    base = {
        "name": "Gigi",
        "preview_rgb": preview,
        "keypoints": neutral_keypoints(),
        "ref_latent": np.full((1, 4, 8, 8), 0.5, dtype=np.float16),
        "ref_face_latent": np.full((1, 4, 4, 4), 0.25, dtype=np.float16),
        "image_size": 768,
        "skip_crop": True,
        "source_name": "gigi.png",
    }
    base.update(extra)
    return base


def _full_payload() -> dict:
    src = io.BytesIO()
    Image.new("RGB", (40, 30), (1, 2, 3)).save(src, format="JPEG")
    return _payload(
        fit={"travel_box": {"left": 0.3}, "hair": [{"class": "hair_left", "polygon": [[0, 0], [1, 0], [0, 1]]}]},
        source_bytes=src.getvalue(),
        source_suffix="JPG",
        pose_keys=[{"delta": [[0.1, 0.2]], "params": [1, 2]}],
        blendshapes={"format": "vtm-blendshapes", "shapes": {"smile": [1, 2, 3]}},
        model={"checkpoint": "D:/models/dit/Noble-768.pt", "image_size": 768},
        meta={"author": "  Ann  ", "license": "CC-BY", "description": "line one\nline two"},
        created_at="2026-01-02T03:04:05Z",
    )


def _write_v1(dest: Path, *, fit: dict | None = None, name: str = "Old") -> Path:
    """A pack exactly the way the v1 writer made it."""
    p = _payload(name=name)
    preview_buf = io.BytesIO()
    Image.fromarray(p["preview_rgb"]).save(preview_buf, format="PNG")
    latents_buf = io.BytesIO()
    np.savez(latents_buf, ref_latent=p["ref_latent"], ref_face_latent=p["ref_face_latent"])
    kps_buf = io.BytesIO()
    np.save(kps_buf, np.asarray(p["keypoints"], dtype=np.float32))
    manifest = {
        "format": "vtm-character",
        "version": 1,
        "name": name,
        "image_size": 768,
        "skip_crop": True,
        "has_face_latent": True,
        "source_name": "old.png",
    }
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        zf.writestr(PREVIEW_NAME, preview_buf.getvalue())
        zf.writestr(LATENTS_NAME, latents_buf.getvalue())
        zf.writestr(KEYPOINTS_NAME, kps_buf.getvalue())
        if fit:
            zf.writestr(FIT_NAME, json.dumps(fit))
    return dest


def _rezip(src: Path, dest: Path, *, mutate=None, extra: dict[str, bytes] | None = None) -> Path:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dest, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if mutate is not None:
                data = mutate(info.filename, data)
            if data is not None:
                zout.writestr(info.filename, data)
        for name, data in (extra or {}).items():
            zout.writestr(name, data)
    return dest


def test_v2_roundtrip_every_field(tmp_path: Path) -> None:
    dest = tmp_path / "gigi.vtm"
    payload = _full_payload()
    write_character_pack(dest, **payload)
    pack = read_character_pack(dest)
    assert pack.version == 2
    assert pack.name == "Gigi"
    assert pack.image_size == 768 and pack.skip_crop is True
    np.testing.assert_array_equal(pack.preview_rgb, payload["preview_rgb"])
    np.testing.assert_allclose(pack.keypoints, payload["keypoints"], atol=1e-6)
    np.testing.assert_array_equal(pack.ref_latent, payload["ref_latent"])
    np.testing.assert_array_equal(pack.ref_face_latent, payload["ref_face_latent"])
    assert pack.fit == payload["fit"]
    assert pack.pose_keys == payload["pose_keys"]
    assert pack.blendshapes == payload["blendshapes"]
    assert pack.source_bytes == payload["source_bytes"]
    assert pack.source_suffix == ".jpg"
    assert pack.model == {
        "checkpoint": "Noble-768.pt",
        "image_size": 768,
        "latent_shape": [1, 4, 8, 8],
        "latent_dtype": "float16",
    }
    assert pack.meta == {"author": "Ann", "license": "CC-BY", "description": "line one\nline two"}
    assert pack.created_at == "2026-01-02T03:04:05Z"
    assert pack.updated_at.endswith("Z") and pack.updated_at != pack.created_at

    raw = peek_character_manifest(dest)
    assert raw["version"] == 2
    assert raw["app"] == {"name": "VTM Noble", "pack_version": 2}
    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
        assert names == {
            MANIFEST_NAME, PREVIEW_NAME, THUMB_NAME, LATENTS_NAME, KEYPOINTS_NAME,
            FIT_NAME, POSE_KEYS_NAME, BLENDSHAPES_NAME, "source.jpg",
        }
        assert set(raw["members"]) == names - {MANIFEST_NAME}
        assert zf.getinfo(PREVIEW_NAME).compress_type == zipfile.ZIP_STORED
        assert zf.getinfo(THUMB_NAME).compress_type == zipfile.ZIP_STORED
        assert zf.getinfo(LATENTS_NAME).compress_type == zipfile.ZIP_DEFLATED
        assert zf.getinfo(FIT_NAME).compress_type == zipfile.ZIP_DEFLATED
    manifest = pack.manifest()
    assert manifest["version"] == 2 and manifest["meta"]["author"] == "Ann"
    assert validate_character_pack(dest)["model"]["checkpoint"] == "Noble-768.pt"


def test_minimal_v2_defaults(tmp_path: Path) -> None:
    dest = tmp_path / "plain.vtm"
    write_character_pack(dest, **_payload())
    pack = read_character_pack(dest)
    assert pack.fit == {} and pack.pose_keys == [] and pack.blendshapes == {}
    assert pack.source_bytes is None and pack.source_suffix == ".png"
    assert pack.meta == {"author": "", "license": "", "description": ""}
    assert pack.model["checkpoint"] == ""
    assert read_pack_json(dest, POSE_KEYS_NAME, {"keys": []}) == {"keys": []}
    assert read_pack_json(dest, BLENDSHAPES_NAME, None) is None


def test_thumb_size_and_fallback(tmp_path: Path) -> None:
    dest = tmp_path / "big.vtm"
    big = np.zeros((600, 400, 3), dtype=np.uint8)
    write_character_pack(dest, **_payload(preview_rgb=big))
    with Image.open(io.BytesIO(read_character_thumb_png(dest))) as img:
        assert max(img.size) == 256 and img.size == (171, 256)
    old = _write_v1(tmp_path / "old.vtm")
    with Image.open(io.BytesIO(read_character_thumb_png(old))) as img:
        assert max(img.size) <= 256


def test_v1_reads_and_upgrade_folds_sidecars_without_overwriting(tmp_path: Path) -> None:
    dest = _write_v1(tmp_path / "old.vtm", fit={"travel_box": {"left": 0.1}})
    pack = read_character_pack(dest)
    assert pack.version == 1
    assert pack.fit == {"travel_box": {"left": 0.1}}
    assert pack.pose_keys == [] and pack.model["latent_shape"] == [1, 4, 8, 8]
    card = character_card(dest)
    assert card["version"] == 1 and card["author"] == ""

    changed = upgrade_character_pack(
        dest,
        fit={"travel_box": {"left": 0.9}, "hair": [{"class": "hair_left"}]},
        pose_keys=[{"delta": [[1, 1]]}],
        blendshapes={"shapes": {"a": 1}},
        model={"checkpoint": "C:\\m\\x.pt"},
    )
    assert changed is True
    up = read_character_pack(dest)
    assert up.version == 2
    assert up.fit["travel_box"] == {"left": 0.1}  # pack data wins
    assert up.fit["hair"] == [{"class": "hair_left"}]  # sidecar-only key folded
    assert up.pose_keys == [{"delta": [[1, 1]]}]
    assert up.blendshapes == {"shapes": {"a": 1}}
    assert up.model["checkpoint"] == "x.pt"
    assert up.created_at and up.updated_at
    with zipfile.ZipFile(dest) as zf:
        assert THUMB_NAME in zf.namelist()

    # A second upgrade has nothing to do and must not clobber anything.
    assert upgrade_character_pack(
        dest,
        fit={"travel_box": {"left": 0.5}},
        pose_keys=[{"delta": [[9, 9]]}],
        blendshapes={"shapes": {"b": 2}},
        model={"checkpoint": "other.pt"},
    ) is False
    again = read_character_pack(dest)
    assert again.pose_keys == [{"delta": [[1, 1]]}]
    assert again.blendshapes == {"shapes": {"a": 1}}
    assert again.model["checkpoint"] == "x.pt"
    assert again.created_at == up.created_at


def test_upgrade_plain_v1_and_card(tmp_path: Path) -> None:
    dest = _write_v1(tmp_path / "old.vtm")
    assert upgrade_character_pack(dest) is True
    card = character_card(dest)
    assert card["version"] == 2
    assert card["thumb_url"].startswith("/api/characters/old/thumb?v=")
    assert "preview_url" in card and card["name"] == "Old"


def test_writers_upgrade_v1(tmp_path: Path) -> None:
    fit_dest = _write_v1(tmp_path / "a.vtm")
    write_pack_fit(fit_dest, {"travel_box": {"up": 0.4}})
    assert peek_character_manifest(fit_dest)["version"] == 2
    assert read_pack_fit(fit_dest) == {"travel_box": {"up": 0.4}}

    json_dest = _write_v1(tmp_path / "b.vtm")
    write_pack_json(json_dest, POSE_KEYS_NAME, [{"delta": 1}, "junk"])
    assert read_character_pack(json_dest).pose_keys == [{"delta": 1}]

    ren = _write_v1(tmp_path / "c.vtm")
    card = rename_character_pack(ren, "New Name")
    assert card["name"] == "New Name" and card["version"] == 2
    read_character_pack(ren)

    kp = _write_v1(tmp_path / "d.vtm")
    buf = io.BytesIO()
    np.save(buf, np.ones((37, 4), dtype=np.float32))
    replace_pack_members(kp, {KEYPOINTS_NAME: buf.getvalue()})
    pack = read_character_pack(kp)
    assert pack.version == 2 and float(pack.keypoints[0, 0]) == 1.0


def test_writes_keep_created_bump_updated_and_hashes(tmp_path: Path) -> None:
    dest = tmp_path / "gigi.vtm"
    write_character_pack(dest, **_full_payload())
    before = peek_character_manifest(dest)
    write_pack_json(dest, BLENDSHAPES_NAME, {"shapes": {}})
    after = peek_character_manifest(dest)
    assert after["created_at"] == before["created_at"]
    assert after["updated_at"] >= before["updated_at"]
    assert after["members"][BLENDSHAPES_NAME] != before["members"][BLENDSHAPES_NAME]
    assert after["members"][PREVIEW_NAME] == before["members"][PREVIEW_NAME]
    pack = read_character_pack(dest)
    assert pack.blendshapes == {"shapes": {}}
    assert pack.source_bytes == _full_payload()["source_bytes"]
    assert pack.pose_keys == _full_payload()["pose_keys"]


def test_update_pack_meta_sanitises(tmp_path: Path) -> None:
    dest = tmp_path / "gigi.vtm"
    write_character_pack(dest, **_full_payload())
    manifest = update_pack_meta(dest, author="  " + "x" * 300 + "  ", description="a\x00b\n" + "d" * 3000)
    assert manifest["meta"]["author"] == "x" * 120
    assert manifest["meta"]["license"] == "CC-BY"  # None = unchanged
    assert manifest["meta"]["description"].startswith("ab\n")
    assert len(manifest["meta"]["description"]) == 2000
    assert read_character_pack(dest).meta == manifest["meta"]
    manifest = update_pack_meta(dest, license="")
    assert manifest["meta"]["license"] == ""
    assert character_card(dest)["author"] == "x" * 120


def test_hash_corruption_detected(tmp_path: Path) -> None:
    good = tmp_path / "good.vtm"
    write_character_pack(good, **_full_payload())

    def _flip_fit(name: str, data: bytes) -> bytes:
        return data.replace(b"0.3", b"0.4") if name == FIT_NAME else data

    bad = _rezip(good, tmp_path / "bad.vtm", mutate=_flip_fit)
    with pytest.raises(CharacterPackError, match="corrupt: fit.json"):
        read_character_pack(bad)
    with pytest.raises(CharacterPackError, match="corrupt"):
        validate_character_pack(bad)
    with pytest.raises(CharacterPackError, match="corrupt"):
        read_pack_fit(bad)
    with pytest.raises(CharacterPackError, match="corrupt"):
        export_character_pack_bytes(bad)
    # Rewrites never launder corruption into fresh hashes.
    with pytest.raises(CharacterPackError, match="corrupt"):
        rename_character_pack(bad, "x")

    missing = _rezip(good, tmp_path / "missing.vtm", mutate=lambda n, d: None if n == POSE_KEYS_NAME else d)
    with pytest.raises(CharacterPackError, match="corrupt: pose_keys.json"):
        read_character_pack(missing)
    with pytest.raises(CharacterPackError, match="corrupt: pose_keys.json"):
        read_pack_json(missing, POSE_KEYS_NAME, {})


def test_unlisted_known_member_is_rejected(tmp_path: Path) -> None:
    good = tmp_path / "good.vtm"
    write_character_pack(good, **_payload())
    sneaky = _rezip(good, tmp_path / "sneaky.vtm", extra={FIT_NAME: b'{"x": 1}'})
    with pytest.raises(CharacterPackError, match="corrupt: fit.json"):
        read_character_pack(sneaky)


def test_unknown_members_ignored(tmp_path: Path) -> None:
    good = tmp_path / "good.vtm"
    write_character_pack(good, **_payload())
    extra = _rezip(good, tmp_path / "extra.vtm", extra={"readme.txt": b"hello"})
    pack = read_character_pack(extra)
    assert pack.name == "Gigi"
    write_pack_fit(extra, {"a": 1})
    with zipfile.ZipFile(extra) as zf:
        assert "readme.txt" not in zf.namelist()


@pytest.mark.parametrize("bad_name", ["../evil.png", "sub/fit.json", "a\\b.json", "C:x.json", ".."])
def test_hostile_member_names_rejected(tmp_path: Path, bad_name: str) -> None:
    good = tmp_path / "good.vtm"
    write_character_pack(good, **_payload())
    hostile = tmp_path / "hostile.vtm"
    shutil.copy(good, hostile)
    with zipfile.ZipFile(hostile, "a") as zf:
        zf.writestr(zipfile.ZipInfo(bad_name), b"x")
    with pytest.raises(CharacterPackError, match="unsafe"):
        read_character_pack(hostile)
    with pytest.raises(CharacterPackError):
        peek_character_manifest(hostile)


def test_oversized_pack_rejected(tmp_path: Path, monkeypatch) -> None:
    good = tmp_path / "good.vtm"
    write_character_pack(good, **_payload())
    monkeypatch.setattr(cp, "MAX_PACK_BYTES", 1000)
    with pytest.raises(CharacterPackError, match="too large"):
        read_character_pack(good)


def test_newer_version_refused(tmp_path: Path) -> None:
    good = tmp_path / "good.vtm"
    write_character_pack(good, **_payload())

    def _bump(name: str, data: bytes) -> bytes:
        if name != MANIFEST_NAME:
            return data
        raw = json.loads(data)
        raw["version"] = 3
        return json.dumps(raw).encode()

    future = _rezip(good, tmp_path / "future.vtm", mutate=_bump)
    with pytest.raises(CharacterPackError, match="newer"):
        read_character_pack(future)


def test_atomic_write_keeps_original_on_failure(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "gigi.vtm"
    write_character_pack(dest, **_full_payload())
    original = dest.read_bytes()

    def _boom(_fd: int) -> None:
        raise OSError("disk yanked")

    monkeypatch.setattr(cp.os, "fsync", _boom)
    with pytest.raises(OSError):
        write_pack_fit(dest, {"travel_box": {"left": 0.99}})
    with pytest.raises(OSError):
        rename_character_pack(dest, "Broken")
    with pytest.raises(OSError):
        write_character_pack(dest, **_payload(name="Clobber"))
    assert dest.read_bytes() == original
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    monkeypatch.undo()
    assert read_character_pack(dest).fit["travel_box"]["left"] == 0.3


def test_atomic_write_failure_on_replace(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "gigi.vtm"
    write_character_pack(dest, **_payload())
    original = dest.read_bytes()

    def _boom(*_a, **_k) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(cp.os, "replace", _boom)
    with pytest.raises(OSError):
        update_pack_meta(dest, author="Z")
    assert dest.read_bytes() == original
    assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_pack_model_matches(tmp_path: Path) -> None:
    dest = tmp_path / "gigi.vtm"
    write_character_pack(dest, **_full_payload())
    pack = read_character_pack(dest)
    shape = (1, 4, 8, 8)
    assert pack_model_matches(pack, checkpoint="E:/other/NOBLE-768.PT", image_size=768, latent_shape=shape)
    assert pack_model_matches(pack, checkpoint="noble-768.pt", image_size=768, latent_shape=list(shape))
    assert not pack_model_matches(pack, checkpoint="different.pt", image_size=768, latent_shape=shape)
    assert not pack_model_matches(pack, checkpoint="", image_size=768, latent_shape=shape)
    assert not pack_model_matches(pack, checkpoint="noble-768.pt", image_size=512, latent_shape=shape)
    assert not pack_model_matches(pack, checkpoint="noble-768.pt", image_size=768, latent_shape=(1, 4, 16, 16))

    unknown = read_character_pack(_write_v1(tmp_path / "old.vtm"))
    assert pack_model_matches(unknown, checkpoint="anything.pt", image_size=768, latent_shape=shape)
    assert not pack_model_matches(unknown, checkpoint="anything.pt", image_size=1024, latent_shape=shape)


def test_export_bytes_readable(tmp_path: Path) -> None:
    v2 = tmp_path / "gigi.vtm"
    write_character_pack(v2, **_full_payload())
    data = export_character_pack_bytes(v2)
    out = tmp_path / "copy.vtm"
    out.write_bytes(data)
    assert read_character_pack(out).pose_keys == _full_payload()["pose_keys"]

    v1 = _write_v1(tmp_path / "old.vtm", fit={"travel_box": {"left": 0.2}})
    before = v1.read_bytes()
    data = export_character_pack_bytes(v1)
    assert v1.read_bytes() == before  # file untouched
    out1 = tmp_path / "old-copy.vtm"
    out1.write_bytes(data)
    pack = read_character_pack(out1)
    assert pack.version == 2 and pack.fit == {"travel_box": {"left": 0.2}}
    with zipfile.ZipFile(out1) as zf:
        assert THUMB_NAME in zf.namelist()


def test_real_library_packs_still_read(tmp_path: Path) -> None:
    packs = sorted((REPO / "characters").glob("*.vtm"))
    if not packs:
        pytest.skip("no packs in characters/")
    for src in packs:
        before = src.read_bytes()
        pack = read_character_pack(src)
        assert pack.keypoints.shape == (37, 4)
        assert read_character_thumb_png(src)[:8] == b"\x89PNG\r\n\x1a\n"
        copy = tmp_path / src.name
        shutil.copy(src, copy)
        sidecar = src.with_suffix("") / "fit.json"
        fit = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
        assert upgrade_character_pack(copy, fit=fit) is True
        up = read_character_pack(copy)
        assert up.version == 2
        np.testing.assert_array_equal(up.ref_latent, pack.ref_latent)
        np.testing.assert_array_equal(up.preview_rgb, pack.preview_rgb)
        if fit:
            assert up.fit == {**fit, **pack.fit}
        assert src.read_bytes() == before
