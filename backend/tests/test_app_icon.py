from pathlib import Path

from PIL import Image

from backend.app_icon import APP_USER_MODEL_ID, hat_png_path, write_hat_ico


def test_write_hat_ico_keeps_transparent_brim(tmp_path: Path) -> None:
    src = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for y in range(20, 44):
        for x in range(20, 44):
            src.putpixel((x, y), (37, 44, 70, 255))
    png = tmp_path / "hat.png"
    src.save(png)
    ico = tmp_path / "hat.ico"
    write_hat_ico(png, ico)
    assert ico.is_file()
    frame = Image.open(ico).convert("RGBA")
    assert frame.getpixel((0, 0))[3] < 10
    mid = frame.getpixel((frame.size[0] // 2, frame.size[1] // 2))
    assert mid[3] > 200
    assert mid[0] < 80


def test_repo_hat_ico_is_transparent() -> None:
    png = hat_png_path()
    assert png is not None
    ico = Path(__file__).resolve().parents[1] / "packaging" / "run.ico"
    if ico.is_file() and ico.stat().st_mtime < png.stat().st_mtime:
        write_hat_ico(png, ico)
    frame = Image.open(ico).convert("RGBA")
    assert frame.getpixel((0, 0))[3] < 10


def test_app_id_is_not_pythonw() -> None:
    assert APP_USER_MODEL_ID == "VTM.Studio"
