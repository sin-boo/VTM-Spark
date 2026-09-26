# VTM Noble

Windows desktop app for real-time pose-driven anime generation (keypoint DiT + live tracking).

**Requirements**

- Windows 10/11
- NVIDIA GeForce RTX 30 / 40 / 50 with a current NVIDIA driver (CUDA). No AMD / macOS / Linux package yet.
- [Node.js LTS](https://nodejs.org/) (builds the UI)
- Python 3 is optional — `install.bat` fetches one through **uv** if none is found
- [WebView2](https://developer.microsoft.com/microsoft-edge/webview2/) (usually already installed)
- Internet on first setup (downloads model weights from Hugging Face — **no API key**)

**Quick start**

1. Clone this repo.
2. Double-click `install.bat` the first time (creates `.venv-build` with **uv**, installs deps, builds the UI, sets up Track Lab, downloads models).
3. Double-click `run.exe` (hat icon) to open the desk.

Models download automatically into:

| Asset | Local path |
|-------|------------|
| DiT checkpoint | `models/dit/VTM-1.5.1.pt` |
| Iris / body / hair trackers | `models/trackers/` |
| Anime face box + landmarks (`face_yolov8n.pt`, `mmpose_anime-face_hrnetv2.pth`) | `models/trackers/` |
| OpenSeeFace face stack | `vendor/tools/openseeface/models/` |

Source weights: [sinBoo1/VTM-Elf-0.01](https://huggingface.co/sinBoo1/VTM-Elf-0.01). Anime face weights come from [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) and [hysts/anime-face-detector](https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1).

Hair tracking (`animeseg_hair3.pt`) is built on Mask2Former weights licensed CC BY-NC 4.0 — non-commercial use only (see THIRD_PARTY_NOTICES.md).

Rebuild runs a **model checklist**. If DiT weights are still missing, Start will download them before launch.

**Layout**

- `backend/` — app (`python -m backend`) plus `backend/packaging/` (install / start menu scripts)
- `ui/` — Vite/React operator UI
- `vendor/` — inference + LivePoser + OpenSeeFace + virtual cam
- `models/` — DiT, trackers, download map, refs
- `characters/` — user VTM packs (package-relative; lives next to the app root)

Regression tests (optional): `python -m pytest backend/tests`

Vendor code is committed under `vendor/`. Only pass `-SyncVendor` to `backend\packaging\build.ps1` if you are developing inside the optional parent monorepo and need to refresh vendor copies.

**License**

Apache License 2.0 covers VTM Noble source and **our** original / fine-tune training work — see [LICENSE](LICENSE). That grant does **not** cover third-party model weights, and it does not apply to models added later.

Third-party weights keep the official license of the publisher who released them. Inventory and source URLs: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). OpenSeeFace binary library notices: `vendor/tools/openseeface/Licenses/`.
