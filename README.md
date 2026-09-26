# VTM Noble

Windows desktop app for real-time pose-driven anime generation (keypoint DiT + live tracking).

**Requirements**

- Windows 10/11
- NVIDIA GeForce RTX 30 / 40 / 50 (CUDA). No AMD / macOS / Linux package yet.
- [WebView2](https://developer.microsoft.com/microsoft-edge/webview2/) (usually already installed)
- Internet on first setup (downloads model weights from Hugging Face — **no API key**)

**Quick start**

1. Clone this repo.
2. Double-click `install.bat` the first time (creates `.venv-build` with **uv**, installs deps, builds the UI, downloads models).
3. Double-click `run.exe` (hat icon) to open the desk.

Models download automatically into:

| Asset | Local path |
|-------|------------|
| DiT checkpoint | `models/dit/VTM-ELF.pt` |
| Iris / body trackers | `models/trackers/` |
| OpenSeeFace face stack | `vendor/tools/openseeface/models/` |

Source weights: [sinBoo1/VTM-Elf-0.01](https://huggingface.co/sinBoo1/VTM-Elf-0.01)

Rebuild runs a **model checklist**. If DiT weights are still missing, Start will download them before launch.

**Layout**

- `backend/` — app (`python -m backend`) plus `backend/packaging/` (start menu + optional package)
- `ui/` — Vite/React operator UI
- `vendor/` — inference + LivePoser + OpenSeeFace + virtual cam
- `models/` — DiT, trackers, download map, refs
- `characters/` — user VTM packs (package-relative; lives next to the app root)

Regression tests (optional): `python -m pytest backend/tests`

**Optional packaged exe** (not used by `run.exe`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\packaging\build.ps1
```

That writes `dist/VTMNoble/` (thin launcher + CUDA `runtime/`). Day-to-day use is `run.exe`.

Vendor code is committed under `vendor/`. Only pass `-SyncVendor` if you are developing inside the optional parent monorepo and need to refresh vendor copies.

**License**

Apache License 2.0 covers VTM Noble source and **our** original / fine-tune training work — see [LICENSE](LICENSE). That grant does **not** cover third-party model weights, and it does not apply to models added later.

Third-party weights keep the official license of the publisher who released them. Inventory and source URLs: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). OpenSeeFace binary library notices: `vendor/tools/openseeface/Licenses/`.
