# VTM Noble

Windows desktop app for real-time pose-driven anime generation (keypoint DiT + live tracking).

**Requirements**

- Windows 10/11
- NVIDIA GeForce RTX 30 / 40 / 50 (CUDA). No AMD / macOS / Linux package yet.
- [WebView2](https://developer.microsoft.com/microsoft-edge/webview2/) (usually already installed)
- Internet on first setup (downloads model weights from Hugging Face — **no API key**)

**Quick start**

1. Clone this repo.
2. Double-click `start.bat`.
3. Choose **[1] Smart Build** (creates `.venv-build`, builds the UI, downloads models).
4. After build, double-click **`VTMNoble.exe`** in this same folder (next to `start.bat`).  
   Or use menu **[2] Start** from `start.bat`.

Smart Build places `VTMNoble.exe` beside `start.bat` on purpose — you should not need to open `dist\VTMNoble\` to run the app. Package files still live under `dist\VTMNoble\` (runtime, backend, vendor); the root exe just points at them. Re-running Smart Build skips heavy steps when they are already up to date (deps, UI, runtime copy, launcher).

Models download automatically into:

| Asset | Local path |
|-------|------------|
| DiT checkpoint | `models/dit/VTM-ELF.pt` |
| Iris / body trackers | `models/trackers/` |
| OpenSeeFace face stack | `vendor/tools/openseeface/models/` |

Source weights: [sinBoo1/VTM-Elf-0.01](https://huggingface.co/sinBoo1/VTM-Elf-0.01)

Setup also runs a **model checklist** — all required files must be present before Start succeeds.

**Dev layout**

- `backend/` — FastAPI + stream engine
- `ui/` — Vite/React operator UI
- `packaging/` — Smart Build / thin launcher
- `vendor/` — lean torch_train + LivePoser + OpenSeeFace
- `data/model_sources.json` — Hugging Face download map

**Package a desktop build**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build.ps1
```

Output:

- **Run locally:** `VTMNoble.exe` appears next to `start.bat` (convenience launcher).
- **Ship / zip:** copy the whole `dist/VTMNoble/` folder (thin exe + CUDA `runtime/` — not the exe alone).

Vendor code is committed under `vendor/`. Only pass `-SyncVendor` if you are developing inside the optional parent monorepo and need to refresh vendor copies.

**License**

Apache License 2.0 — see [LICENSE](LICENSE).  
Third-party notices for OpenSeeFace binaries live under `vendor/tools/openseeface/Licenses/`.
