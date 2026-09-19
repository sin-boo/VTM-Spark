# Track lab

Face-tracking bench. Shares the main VTM Noble venv (`.venv-build`). Source edits stay in this folder.

**Still goes here:** `input/source.png`

## Setup (once)

From this folder, after the main app has a venv (`start.bat` → **[1] Smart Build**):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
cd ui
npm install
```

`setup.ps1` copies OpenSeeFace Python + face models here. It does **not** create a second venv. `start.ps1` runs setup on its own if the face models are missing.

## Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Or double-click `start.bat`. Open [http://127.0.0.1:5174](http://127.0.0.1:5174).

- **Track** — fit the rest mesh on `input/source.png`
- **Reset** — tear down the Python tracker / ONNX sessions

Python is `..\.venv-build\Scripts\python.exe`. `PYTHONPATH` is this folder so `backend` / `harness` resolve here, not the main app.

## Harness

`harness/` is the outbound tracking + inbound settings port. The lab UI talks to `/api`; VTM Noble talks to `/harness`.

Port **8780** is fixed. A second `start.ps1` reuses a live lab **with harness**. An old process that only has `/api/health` is killed and replaced. Anything else on that port is an error — we do not hop.

| Path | What |
|---|---|
| `GET /harness/status` | cameras, feel/overlay flags, calibration, advertised commands |
| `GET /harness/frame` | latest character-space packet (Label28 + skeleton + hair + visemes) |
| `POST /harness/command` | `{ "op": "set_feel", "body": { "smoothing": 0.2 } }` |
| `WS /harness/ws` | pushed `frame` / `status` packets; send the same command JSON back |

Keypoints are already retargeted onto the character still (`coord_space: "character_px"`, KEYPOINT_SCHEMA 37 slots). Overlay flags live in `feel` (`show_face`, `show_skeleton`, `show_hair`, `show_ids`).

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path
..\.venv-build\Scripts\python.exe -c "from harness.client import HarnessClient; c=HarnessClient(); print(c.status().get('commands')); print(c.command('ping'))"
```

From this folder: `..\.venv-build\Scripts\python.exe -m pytest harness`

## Layout

- `backend/` — FastAPI + still-image tracker
- `harness/` — tracking packets out, settings commands in
- `osf/` — local OpenSeeFace Python (safe to edit)
- `models/` — local face ONNX weights
- `ui/` — React bench
