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
  [switch]$SkipVendorSync,
  [switch]$SyncVendor
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Out = Join-Path $Root "dist\VTMNoble"
$VenvDir = Join-Path $Root ".venv-build"
$Py = Join-Path $VenvDir "Scripts\python.exe"

. (Join-Path $PSScriptRoot "console-progress.ps1")

Write-Host "==> VTM Noble package (thin launcher + side-by-side runtime)"
Write-Host "    NOTE: PyInstaller will NOT analyze torch. Peak RAM should stay normal."

Write-Host "==> Killing leftover VTM Noble / backend processes"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "kill-orphans.ps1")

# --- Vendor: use committed vendor/ by default (opt-in -SyncVendor for monorepo) ----
$MonorepoMarker = Join-Path (Split-Path $Root -Parent) "send2pod\torch_train"
$HasMonorepo = Test-Path -LiteralPath $MonorepoMarker
$DoSync = $SyncVendor -and (-not $SkipVendorSync)
if ($DoSync) {
  if (-not $HasMonorepo) {
    throw "-SyncVendor requires parent monorepo (send2pod/torch_train). Standalone clones use committed vendor/."
  }
  Write-Host "==> Syncing lean vendor/ from monorepo parent (-SyncVendor)"
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "sync-vendor.ps1")
  if ($LASTEXITCODE -ne 0) { throw "vendor sync failed" }
} else {
  Write-Host "==> Using committed vendor/ (pass -SyncVendor only when refreshing from monorepo)"
}
if (-not (Test-Path (Join-Path $Root "vendor\torch_train\inference_keypoint.py"))) {
  throw "vendor/torch_train missing - re-clone the repo (vendor/ must be present)"
}
if (-not (Test-Path (Join-Path $Root "vendor\tools\live-poser\live_poser.py"))) {
  throw "vendor/tools/live-poser missing - re-clone the repo (vendor/ must be present)"
}

# --- UI --------------------------------------------------------------------
Write-Host "==> Building UI (Vite)"
Push-Location "$Root\ui"
try {
  if (-not (Test-Path "node_modules")) {
    Write-LongStepHint "Installing UI npm packages (first run can take a few minutes)..."
    $code = Invoke-ProcessWithHeartbeat `
      -FilePath "cmd.exe" `
      -ArgumentList @("/c", "npm", "ci") `
      -Activity "npm ci" `
      -HeartbeatSeconds 12
    if ($code -ne 0) {
      $code = Invoke-ProcessWithHeartbeat `
        -FilePath "cmd.exe" `
        -ArgumentList @("/c", "npm", "install") `
        -Activity "npm install" `
        -HeartbeatSeconds 12
    }
    if ($code -ne 0) { throw "npm install failed" }
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
    Write-LongStepHint "Running Vite production build..."
    $code = Invoke-ProcessWithHeartbeat `
      -FilePath "cmd.exe" `
      -ArgumentList @("/c", "npm", "run", "build") `
      -Activity "vite build" `
      -HeartbeatSeconds 10
    if ($code -ne 0) { throw "UI build failed" }
  }
} finally {
  Pop-Location
}

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
  Write-LongStepHint "Creating venv can take a minute on first run..."
  & $BasePython -m venv $VenvDir
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Py)) { throw "Failed to create build venv" }
  Write-Host "==> Upgrading pip in build venv"
  $null = Invoke-NativeWithHeartbeat `
    -FilePath $Py `
    -ArgumentList @("-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip") `
    -Activity "pip upgrade" `
    -HeartbeatSeconds 10
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

# Run pip via call operator (keeps real progress bars).
# IMPORTANT: PowerShell variables are case-insensitive — never name a local $pipArgs
# when the parameter is $PipArgs (that bug produced: python -m pip -m pip install...).
function Invoke-Pip {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$PipArgs,
    [string]$Activity = "pip",
    [int]$HeartbeatSeconds = 12
  )
  $exeArgs = [System.Collections.Generic.List[string]]::new()
  [void]$exeArgs.Add("-m")
  [void]$exeArgs.Add("pip")
  foreach ($a in @($PipArgs)) { [void]$exeArgs.Add([string]$a) }
  if ($PipArgs -contains "install" -and -not ($PipArgs -contains "--progress-bar")) {
    [void]$exeArgs.Add("--progress-bar")
    [void]$exeArgs.Add("on")
  }
  $code = Invoke-NativeWithHeartbeat `
    -FilePath $Py `
    -ArgumentList @($exeArgs.ToArray()) `
    -Activity $Activity `
    -HeartbeatSeconds $HeartbeatSeconds
  # Defensive: never let a multi-object pipeline result blow up the [int] cast.
  if ($code -is [System.Array]) {
    $code = $code | Select-Object -Last 1
  }
  return [int]$code
}

function Install-CudaTorch {
  Write-Host "==> Ensuring CUDA torch (cu128)"
  Write-LongStepHint "CUDA wheels are large (often 2+ GB). Downloads can take several minutes."
  Write-LongStepHint "pip shows its own download progress below."
  $null = Invoke-Pip -PipArgs @("uninstall", "-y", "torch", "torchvision", "torchaudio") `
    -Activity "uninstalling previous torch" `
    -HeartbeatSeconds 8
  # Avoid -q so failures and download progress stay visible. Index-only from pytorch cu128.
  $code = Invoke-Pip -PipArgs @(
    "install", "--disable-pip-version-check",
    "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cu128"
  ) -Activity "CUDA torch cu128 download/install" -HeartbeatSeconds 10
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
  Write-LongStepHint "Installing Python deps from requirements.txt (can take a few minutes)..."
  $code = Invoke-Pip -PipArgs @(
    "install", "--disable-pip-version-check", "-r", "$Root\requirements.txt"
  ) -Activity "requirements.txt" -HeartbeatSeconds 12
  if ($code -ne 0) { throw "pip install failed (exit $code)" }

  # If anything clobbered the CUDA wheel, put it back.
  if (-not (Test-CudaTorch)) {
    Write-Host "==> torch was replaced by a CPU wheel - reinstalling cu128"
    Install-CudaTorch
  }

  Write-Host "==> Ensuring PyInstaller"
  $code = Invoke-Pip -PipArgs @("install", "--disable-pip-version-check", "pyinstaller") `
    -Activity "pyinstaller" -HeartbeatSeconds 10
  if ($code -ne 0) { throw "PyInstaller install failed (exit $code)" }

  # Windows: official PyPI triton has no wheels; torch.compile needs triton-windows.
  # Pin to Triton's minor for the installed torch (2.11 -> 3.6.x).
  Write-Host "==> Ensuring triton-windows for torch.compile"
  $code = Invoke-Pip -PipArgs @(
    "install", "--disable-pip-version-check", "triton-windows>=3.6,<3.7"
  ) -Activity "triton-windows" -HeartbeatSeconds 10
  if ($code -ne 0) {
    Write-Host "    WARNING: triton-windows install failed (exit $code) - Fast will stay eager"
  } else {
    Write-Host "    triton-windows ready"
  }
}

if (-not (Test-RuntimeImports)) { throw "Build venv is missing required packages" }
if (-not (Test-CudaTorch)) { throw "Build venv must have torch+cu128" }
Write-Host "    runtime imports OK (CUDA torch)"

# --- Prepare dist folder (overwrite in place - no .old stashes) ------------
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
  Write-Host "    overwriting package (kept runtime/ - use -ForceBundle to refresh it)"
}

Clear-DistSoft -DistDir $Out

# Copy runtime venv
$RuntimeDest = Join-Path $Out "runtime"
$RuntimeMarker = Join-Path $RuntimeDest "Scripts\python.exe"
if ((Test-Path $RuntimeMarker) -and -not $ForceBundle) {
  Write-Host "==> runtime/ already present - skipping venv copy (use -ForceBundle to refresh)"
} else {
  New-Item -ItemType Directory -Force -Path $RuntimeDest | Out-Null
  Write-LongStepHint "Copy uses disk I/O (not tens of GB of RAM)."
  $null = Invoke-RobocopyWithProgress `
    -Source $VenvDir `
    -Dest $RuntimeDest `
    -Label "Copying Python runtime (venv -> dist/VTMNoble/runtime)" `
    -ExtraArgs @("/E", "/J", "/MT:4", "/XD", "__pycache__", ".git", ".pytest_cache", "Tests", "tests", "test", "/XF", "*.pyc", "*.pyo")
  if (-not (Test-Path $RuntimeMarker)) { throw "Failed to copy runtime python.exe" }
}

# App code
Write-Host "==> Copying app code (backend, ui/dist, data, models placeholders)"
$null = Invoke-RobocopyWithProgress `
  -Source (Join-Path $Root "backend") `
  -Dest (Join-Path $Out "backend") `
  -Label "Copying backend/" `
  -ExtraArgs @("/E", "/XD", "__pycache__", ".pytest_cache", "/XF", "*.pyc")
if (Test-Path (Join-Path $Root "ui\dist")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "ui\dist") | Out-Null
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "ui\dist") `
    -Dest (Join-Path $Out "ui\dist") `
    -Label "Copying ui/dist/" `
    -ExtraArgs @("/E")
}
if (Test-Path (Join-Path $Root "data")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "data") | Out-Null
  # Keep config; skip local user uploads under data/refs and cached data/models.
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "data") `
    -Dest (Join-Path $Out "data") `
    -Label "Copying data/" `
    -ExtraArgs @("/E", "/XD", "models", "refs")
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "data\refs") | Out-Null
}

# models/dit empty placeholder + models/trackers weights (DiT downloaded on setup)
Write-Host "==> Copying models/trackers (DiT weights download separately)"
New-Item -ItemType Directory -Force -Path (Join-Path $Out "models\dit") | Out-Null
Set-Content -Path (Join-Path $Out "models\dit\.gitkeep") -Value ""
if (Test-Path (Join-Path $Root "models\trackers")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "models\trackers") | Out-Null
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "models\trackers") `
    -Dest (Join-Path $Out "models\trackers") `
    -Label "Copying models/trackers/" `
    -ExtraArgs @("/E", "/XF", ".gitkeep")
}

# Download default DiT into source models/dit (packaged app also downloads on first launch).
Write-Host "==> Ensuring DiT checkpoint (models/dit/VTM-ELF.pt from Hugging Face)"
Write-LongStepHint "Large model download - Hugging Face / tqdm progress appears below when fetching."
$prevPyPath = $env:PYTHONPATH
$env:PYTHONPATH = $Root
try {
  & $Py -m backend.model_download
  if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: DiT download failed - app will retry on first launch." -ForegroundColor Yellow
  }
} finally {
  if ($null -eq $prevPyPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
  else { $env:PYTHONPATH = $prevPyPath }
}
# Keep package lean: do not copy multi-hundred-MB DiT weights into dist/.
# Runtime downloads into <install>/models/dit via data/model_sources.json.

# Lean vendor only (never monorepo tools/ datasets)
$null = Invoke-RobocopyWithProgress `
  -Source (Join-Path $Root "vendor") `
  -Dest (Join-Path $Out "vendor") `
  -Label "Copying vendor/ (lean)" `
  -ExtraArgs @("/E", "/MT:4", "/XD", "__pycache__", ".git", ".pytest_cache", "tests", "Tests", "training", "datasets", "data", "input", "/XF", "*.pyc")

# Tiny launcher exe — also dropped next to start.bat for one-click run.
$LauncherExe = Join-Path $Out "VTMNoble.exe"
$RootExe = Join-Path $Root "VTMNoble.exe"
$NeedLauncher = $true
if ($SkipLauncher) {
  $NeedLauncher = $false
  Write-Host "==> Launcher skip requested (-SkipLauncher)"
} elseif (-not $ForceBundle -and (Test-Path -LiteralPath $RootExe)) {
  $exeTime = (Get-Item -LiteralPath $RootExe).LastWriteTimeUtc
  $launchSrcs = @(
    (Join-Path $Root "packaging\launcher.py"),
    (Join-Path $Root "packaging\launcher.spec")
  ) | Where-Object { Test-Path -LiteralPath $_ }
  $newestSrc = (
    $launchSrcs | ForEach-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc } |
      Sort-Object -Descending | Select-Object -First 1
  )
  if ($null -ne $newestSrc -and $newestSrc -le $exeTime) {
    $NeedLauncher = $false
    Write-Host "==> Launcher up to date - reusing root VTMNoble.exe (no PyInstaller reload)"
  }
}

if ($NeedLauncher) {
  Write-Host "==> Building tiny launcher exe (stdlib only - low memory)"
  $null = Invoke-Pip -PipArgs @("install", "--disable-pip-version-check", "pyinstaller") `
    -Activity "pyinstaller" -HeartbeatSeconds 10
  $LaunchWork = Join-Path $Root "build\launcher"
  New-Item -ItemType Directory -Force -Path $LaunchWork | Out-Null
  Write-LongStepHint "PyInstaller is packing the tiny launcher - usually under a minute..."
  $code = Invoke-ProcessWithHeartbeat `
    -FilePath $Py `
    -ArgumentList @(
      "-m", "PyInstaller", "--noconfirm", "--clean",
      "--distpath", (Join-Path $Root "build\launcher_dist"),
      "--workpath", $LaunchWork,
      (Join-Path $Root "packaging\launcher.spec")
    ) `
    -Activity "PyInstaller launcher" `
    -HeartbeatSeconds 10
  if ($code -ne 0) { throw "Launcher PyInstaller failed" }
  $built = Join-Path $Root "build\launcher_dist\VTMNoble.exe"
  if (-not (Test-Path $built)) { throw "Launcher exe not produced" }
  Copy-Item -Force $built $LauncherExe
  Copy-Item -Force $built $RootExe
} else {
  if (-not (Test-Path -LiteralPath $RootExe)) {
    throw "No VTMNoble.exe at repo root to reuse. Re-run without -SkipLauncher."
  }
  # dist/ was wiped (kept runtime/); put the clickable exe back into the package folder too.
  Copy-Item -Force $RootExe $LauncherExe
}

$Bat = @"
@echo off
cd /d "%~dp0"
"%~dp0runtime\Scripts\python.exe" -m backend %*
"@
Set-Content -Path (Join-Path $Out "VTMNoble.bat") -Value $Bat -Encoding ASCII

# Root convenience bat (same folder as start.bat) — launches via root exe → dist package.
$RootBat = @"
@echo off
cd /d "%~dp0"
start "" "%~dp0VTMNoble.exe" %*
"@
Set-Content -Path (Join-Path $Root "VTMNoble.bat") -Value $RootBat -Encoding ASCII

Write-Host ""
Write-Host "==> Done"
Write-Host "    Click to run:  $RootExe"
Write-Host "    (same folder as start.bat — you do not need to open dist\ )"
Write-Host "    Package data:  $Out"
Write-Host "    Fallback:      $Out\VTMNoble.bat"
Write-Host "DiT models: auto-download VTM-ELF.pt into models\dit (sinBoo1/VTM-Elf-0.01)"
Write-Host "Supported GPUs: GeForce RTX 30 / 40 / 50 (CUDA). No AMD."
