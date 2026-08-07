# Memory-safe VTM Noble packaging.
# Does NOT freeze torch/transformers with PyInstaller (that can use tens of GB RAM).
# Instead:
#   1) ensure .venv-build has deps (CUDA torch cu128 last)
#   2) copy that venv to dist/VTMNoble/runtime
#   3) copy app code + lean vendor/
#   4) build a TINY launcher exe (stdlib only) that runs: runtime\Scripts\python.exe -m backend
# DiT weights download into models/dit on Smart Build / first launch (not frozen into the zip).
param(
  [string]$BasePython = "",
  [switch]$RecreateVenv,
  [switch]$SkipDeps,
  [switch]$ForceBundle,
  [switch]$SkipLauncher,
  [switch]$SkipVendorSync
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Out = Join-Path $Root "dist\VTMNoble"
$VenvDir = Join-Path $Root ".venv-build"
$Py = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "==> VTM Noble package (thin launcher + side-by-side runtime)"
Write-Host "    NOTE: PyInstaller will NOT analyze torch. Peak RAM should stay normal."

Write-Host "==> Killing leftover VTM Noble / backend processes"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "kill-orphans.ps1")

# --- Vendor (self-contained) ----------------------------------------------
if (-not $SkipVendorSync) {
  Write-Host "==> Syncing lean vendor/"
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "sync-vendor.ps1")
  if ($LASTEXITCODE -ne 0) { throw "vendor sync failed" }
}
if (-not (Test-Path (Join-Path $Root "vendor\torch_train\inference_keypoint.py"))) {
  throw "vendor/torch_train missing — run packaging\sync-vendor.ps1 from the monorepo once"
}
if (-not (Test-Path (Join-Path $Root "vendor\tools\live-poser\live_poser.py"))) {
  throw "vendor/tools/live-poser missing"
}

# --- UI --------------------------------------------------------------------
Write-Host "==> Building UI (Vite)"
Push-Location "$Root\ui"
if (-not (Test-Path "node_modules")) {
  npm ci
  if ($LASTEXITCODE -ne 0) { npm install }
}
$UiDist = Join-Path $Root "ui\dist\index.html"
$NeedUiBuild = $true
if (Test-Path $UiDist) {
  $distTime = (Get-Item $UiDist).LastWriteTimeUtc
  $newestSrc = Get-ChildItem -Path (Join-Path $Root "ui\src"), (Join-Path $Root "ui\index.html") -Recurse -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
  if ($null -eq $newestSrc -or $newestSrc.LastWriteTimeUtc -le $distTime) {
    Write-Host "    ui/dist is up to date - skipping vite build"
    $NeedUiBuild = $false
  }
}
if ($NeedUiBuild) {
  npm run build
  if ($LASTEXITCODE -ne 0) { throw "UI build failed" }
}
Pop-Location

# --- Python discovery / venv ----------------------------------------------
function Test-PythonExe {
  param([string]$PythonExe)
  if (-not $PythonExe) { return $false }
  if (-not (Test-Path -LiteralPath $PythonExe)) { return $false }
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $PythonExe -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 1>$null 2>$null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  } finally {
    $ErrorActionPreference = $prev
  }
}

function Resolve-BasePython {
  param([string]$Preferred)
  if ($Preferred) {
    if (Test-Path -LiteralPath $Preferred) {
      if (Test-PythonExe $Preferred) { return $Preferred }
      throw "BasePython path is not a usable Python 3.10+: $Preferred"
    }
    return $Preferred
  }
  $pathCandidates = @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    "C:\Program Files\Python312\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python310\python.exe"
  )
  foreach ($c in $pathCandidates) {
    if (Test-PythonExe $c) {
      Write-Host "==> Found base Python: $c"
      return $c
    }
  }
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    if (Get-Command python -ErrorAction SilentlyContinue) {
      $out = & python -c 'import sys; print(sys.executable)' 2>$null
      if ($LASTEXITCODE -eq 0 -and $out -and (Test-PythonExe $out.Trim())) {
        return $out.Trim()
      }
    }
  } finally {
    $ErrorActionPreference = $prev
  }
  return $null
}

if ($RecreateVenv -and (Test-Path $VenvDir)) {
  Write-Host "==> Removing existing build venv"
  Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $Py)) {
  $BasePython = Resolve-BasePython -Preferred $BasePython
  if (-not $BasePython) {
    throw "No Python 3.10+ found. Pass -BasePython <path-to-python.exe>."
  }
  Write-Host "==> Creating build venv at $VenvDir"
  & $BasePython -m venv $VenvDir
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Py)) { throw "Failed to create build venv" }
  & $Py -m pip install --disable-pip-version-check -q --upgrade pip
}

Write-Host "==> Using Python: $Py"

function Test-RuntimeImports {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $Py -c "import importlib; bad=[]
for m in ['torch','transformers','diffusers','fastapi','uvicorn','webview','PIL','cv2']:
  try: importlib.import_module(m)
  except Exception: bad.append(m)
raise SystemExit(1 if bad else 0)"
  $ok = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = $prev
  return $ok
}

function Test-CudaTorch {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $raw = & $Py -c "import torch; print(torch.__version__); print(torch.version.cuda or '')" 2>&1
  $code = $LASTEXITCODE
  $ErrorActionPreference = $prev
  $text = (($raw | ForEach-Object { "$_" }) -join "`n").Trim()
  if ($code -ne 0) {
    Write-Host "==> torch probe FAILED (exit $code): $text"
    return $false
  }
  $lines = @($text -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
  $ver = if ($lines.Count -ge 1) { $lines[0] } else { "" }
  $cuda = if ($lines.Count -ge 2) { $lines[1] } else { "" }
  Write-Host "==> torch probe: version=$ver cuda_tag=$cuda"
  if ($ver -match '\+cu\d+' -or ($cuda -and $cuda -ne "None" -and $cuda -ne "")) {
    return $true
  }
  Write-Host "==> torch probe: not a CUDA wheel"
  return $false
}

# Capture pip exit code WITHOUT piping (pipeline resets LASTEXITCODE in PowerShell).
function Invoke-Pip {
  param([Parameter(Mandatory = $true)][string[]]$PipArgs)
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $output = & $Py -m pip @PipArgs 2>&1
  $code = $LASTEXITCODE
  $ErrorActionPreference = $prev
  foreach ($item in @($output)) {
    $text = "$item"
    if ($text -match '^(WARNING:|WARN:)') {
      continue
    }
    if ($text.Trim() -ne "") {
      Write-Host $text
    }
  }
  return [int]$code
}

function Install-CudaTorch {
  Write-Host "==> Ensuring CUDA torch (cu128)"
  $null = Invoke-Pip @("uninstall", "-y", "torch", "torchvision", "torchaudio")
  # Avoid -q so failures are visible. Index-only install from pytorch cu128.
  $code = Invoke-Pip @(
    "install", "--disable-pip-version-check",
    "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cu128"
  )
  if ($code -ne 0) {
    throw "CUDA torch install failed (pip exit $code)"
  }
  if (-not (Test-CudaTorch)) {
    throw "CUDA torch not installed (still CPU / import failed). Aborting package."
  }
}

$NeedDeps = -not $SkipDeps
if (-not $NeedDeps) {
  Write-Host "==> -SkipDeps set - checking imports only"
  if (-not (Test-RuntimeImports) -or -not (Test-CudaTorch)) {
    Write-Host "    imports/CUDA incomplete - installing deps anyway"
    $NeedDeps = $true
  } else {
    Write-Host "    build venv imports OK - skipping pip/torch install"
  }
}

if ($NeedDeps) {
  # Install CUDA torch FIRST so transformers/diffusers/ultralytics see it as satisfied
  # and do not pull a CPU torch from PyPI.
  if (-not (Test-CudaTorch)) {
    Install-CudaTorch
  } else {
    Write-Host "==> CUDA torch wheel already present"
  }

  Write-Host "==> Installing requirements (torch already provided by cu128 wheel)"
  $code = Invoke-Pip @("install", "--disable-pip-version-check", "-q", "-r", "$Root\requirements.txt")
  if ($code -ne 0) { throw "pip install failed (exit $code)" }

  # If anything clobbered the CUDA wheel, put it back.
  if (-not (Test-CudaTorch)) {
    Write-Host "==> torch was replaced by a CPU wheel — reinstalling cu128"
    Install-CudaTorch
  }

  $code = Invoke-Pip @("install", "--disable-pip-version-check", "-q", "pyinstaller")
  if ($code -ne 0) { throw "PyInstaller install failed (exit $code)" }

  # Windows: official PyPI triton has no wheels; torch.compile needs triton-windows.
  # Pin to Triton's minor for the installed torch (2.11 → 3.6.x).
  Write-Host "==> Ensuring triton-windows for torch.compile"
  $code = Invoke-Pip @("install", "--disable-pip-version-check", "-q", "triton-windows>=3.6,<3.7")
  if ($code -ne 0) {
    Write-Host "    WARNING: triton-windows install failed (exit $code) — Fast will stay eager"
  } else {
    Write-Host "    triton-windows ready"
  }
}

if (-not (Test-RuntimeImports)) { throw "Build venv is missing required packages" }
if (-not (Test-CudaTorch)) { throw "Build venv must have torch+cu128" }
Write-Host "    runtime imports OK (CUDA torch)"

# --- Prepare dist folder (overwrite in place — no .old stashes) ------------
function Clear-DistSoft {
  param([string]$DistDir)
  Write-Host "==> Preparing $DistDir (overwrite in place)"
  Get-Process -Name "VTMNoble","RealStream","robocopy" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
  cmd /c "taskkill /F /IM VTMNoble.exe >nul 2>&1" | Out-Null
  cmd /c "taskkill /F /IM RealStream.exe >nul 2>&1" | Out-Null
  Start-Sleep -Seconds 1

  $parent = Split-Path -Parent $DistDir
  $leaf = Split-Path -Leaf $DistDir
  if (Test-Path -LiteralPath $parent) {
    Get-ChildItem -LiteralPath $parent -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -like "$leaf.old.*" } |
      ForEach-Object {
        Write-Host "    removing leftover $($_.Name)"
        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
      }
  }

  if (-not (Test-Path -LiteralPath $DistDir)) {
    New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
    return
  }

  if ($ForceBundle) {
    Write-Host "    removing existing dist (-ForceBundle)"
    Remove-Item -LiteralPath $DistDir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
    return
  }

  # Keep runtime/ (multi-GB) unless -ForceBundle; wipe the rest so stale files go away.
  Get-ChildItem -LiteralPath $DistDir -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "runtime" } |
    ForEach-Object {
      Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
  Write-Host "    overwriting package (kept runtime/ — use -ForceBundle to refresh it)"
}

Clear-DistSoft -DistDir $Out

# Copy runtime venv
$RuntimeDest = Join-Path $Out "runtime"
$RuntimeMarker = Join-Path $RuntimeDest "Scripts\python.exe"
if ((Test-Path $RuntimeMarker) -and -not $ForceBundle) {
  Write-Host "==> runtime/ already present - skipping venv copy (use -ForceBundle to refresh)"
} else {
  Write-Host "==> Copying Python runtime (venv -> dist/VTMNoble/runtime)"
  Write-Host "    This uses disk, not tens of GB of RAM. Please wait..."
  New-Item -ItemType Directory -Force -Path $RuntimeDest | Out-Null
  robocopy $VenvDir $RuntimeDest /E /J /MT:4 `
    /XD __pycache__ .git .pytest_cache Tests tests test `
    /XF *.pyc *.pyo `
    /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
  if (-not (Test-Path $RuntimeMarker)) { throw "Failed to copy runtime python.exe" }
}

# App code
Write-Host "==> Copying app code (backend, ui/dist, data, models placeholders)"
robocopy (Join-Path $Root "backend") (Join-Path $Out "backend") /E /XD __pycache__ .pytest_cache /XF *.pyc /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if (Test-Path (Join-Path $Root "ui\dist")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "ui\dist") | Out-Null
  robocopy (Join-Path $Root "ui\dist") (Join-Path $Out "ui\dist") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
}
if (Test-Path (Join-Path $Root "data")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "data") | Out-Null
  # Keep config; skip local user uploads under data/refs and cached data/models.
  robocopy (Join-Path $Root "data") (Join-Path $Out "data") /E /XD models refs /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "data\refs") | Out-Null
}

# models/dit empty placeholder + models/trackers weights (DiT downloaded on setup)
Write-Host "==> Copying models/trackers (DiT weights download separately)"
New-Item -ItemType Directory -Force -Path (Join-Path $Out "models\dit") | Out-Null
Set-Content -Path (Join-Path $Out "models\dit\.gitkeep") -Value ""
if (Test-Path (Join-Path $Root "models\trackers")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "models\trackers") | Out-Null
  robocopy (Join-Path $Root "models\trackers") (Join-Path $Out "models\trackers") /E /XF .gitkeep /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
}

# Download default DiT into source models/dit (packaged app also downloads on first launch).
Write-Host "==> Ensuring DiT checkpoint (models/dit/VTM-ELF.pt from Hugging Face)"
$prevPyPath = $env:PYTHONPATH
$env:PYTHONPATH = $Root
try {
  & $Py -m backend.model_download
  if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: DiT download failed — app will retry on first launch." -ForegroundColor Yellow
  }
} finally {
  if ($null -eq $prevPyPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
  else { $env:PYTHONPATH = $prevPyPath }
}
# Keep package lean: do not copy multi-hundred-MB DiT weights into dist/.
# Runtime downloads into <install>/models/dit via data/model_sources.json.

# Lean vendor only (never monorepo tools/ datasets)
Write-Host "==> Copying vendor/ (lean)"
robocopy (Join-Path $Root "vendor") (Join-Path $Out "vendor") /E /MT:4 `
  /XD __pycache__ .git .pytest_cache tests Tests training datasets data input `
  /XF *.pyc `
  /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null

# Tiny launcher exe
$LauncherExe = Join-Path $Out "VTMNoble.exe"
if ((Test-Path $LauncherExe) -and $SkipLauncher) {
  Write-Host "==> Launcher exe present - skipping (-SkipLauncher)"
} else {
  Write-Host "==> Building tiny launcher exe (stdlib only - low memory)"
  & $Py -m pip install --disable-pip-version-check -q pyinstaller
  $LaunchWork = Join-Path $Root "build\launcher"
  New-Item -ItemType Directory -Force -Path $LaunchWork | Out-Null
  & $Py -m PyInstaller --noconfirm --clean `
    --distpath (Join-Path $Root "build\launcher_dist") `
    --workpath $LaunchWork `
    (Join-Path $Root "packaging\launcher.spec")
  if ($LASTEXITCODE -ne 0) { throw "Launcher PyInstaller failed" }
  $built = Join-Path $Root "build\launcher_dist\VTMNoble.exe"
  if (-not (Test-Path $built)) { throw "Launcher exe not produced" }
  Copy-Item -Force $built $LauncherExe
}

$Bat = @"
@echo off
cd /d "%~dp0"
"%~dp0runtime\Scripts\python.exe" -m backend %*
"@
Set-Content -Path (Join-Path $Out "VTMNoble.bat") -Value $Bat -Encoding ASCII

Write-Host ""
Write-Host "==> Done: $Out"
Write-Host "Run: $Out\VTMNoble.exe"
Write-Host "Fallback: $Out\VTMNoble.bat"
Write-Host "DiT models: auto-download VTM-ELF.pt into $Out\models\dit (sinBoo1/VTM-Elf-0.01)"
Write-Host "Supported GPUs: GeForce RTX 30 / 40 / 50 (CUDA). No AMD."
