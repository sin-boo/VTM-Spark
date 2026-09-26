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
  [switch]$SyncVendor,
  [switch]$SkipPackage
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Out = Join-Path $Root "dist\VTMNoble"
$VenvDir = Join-Path $Root ".venv-build"
$Py = Join-Path $VenvDir "Scripts\python.exe"

. (Join-Path $PSScriptRoot "console-progress.ps1")
. (Join-Path $PSScriptRoot "venv-home.ps1")

function Get-UvExe {
  $cmd = Get-Command uv -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) { return [string]$cmd.Source }
  $local = Join-Path $Root ".tools\uv.exe"
  if (Test-Path -LiteralPath $local) { return $local }
  Write-Host "==> Downloading uv into .tools\ (one-time)"
  $tools = Join-Path $Root ".tools"
  New-Item -ItemType Directory -Force -Path $tools | Out-Null
  $zip = Join-Path $env:TEMP "uv-x86_64-pc-windows-msvc.zip"
  $url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
  $extract = Join-Path $env:TEMP "uv-extract"
  if (Test-Path -LiteralPath $extract) { Remove-Item -LiteralPath $extract -Recurse -Force }
  Expand-Archive -Path $zip -DestinationPath $extract -Force
  $found = Get-ChildItem -Path $extract -Filter "uv.exe" -Recurse | Select-Object -First 1
  if (-not $found) { throw "uv.exe missing from Astral download" }
  Copy-Item -Force $found.FullName $local
  $uvx = Get-ChildItem -Path $extract -Filter "uvx.exe" -Recurse | Select-Object -First 1
  if ($uvx) { Copy-Item -Force $uvx.FullName (Join-Path $tools "uvx.exe") }
  return $local
}

$script:UvExe = Get-UvExe
Write-Host "==> Using uv: $($script:UvExe)"

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
    (Join-Path $Root ".venv\Scripts\python.exe")
  ) + @(Get-VtmHostPythonCandidates)
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

if ((Test-Path $Py) -and -not (Test-VtmPythonExe $Py)) {
  Write-Host "==> Build venv launcher is dead (missing base pythonw / encodings)"
  if (Repair-VtmVenvHome -VenvDir $VenvDir -PythonExe $Py) {
    Write-Host "==> Retargeted .venv-build to a working Python of the same version"
  } else {
    Write-Host "==> Could not retarget. Recreating build venv"
    Remove-Item -Recurse -Force $VenvDir
  }
}

if (-not (Test-Path $Py)) {
  $BasePython = Resolve-BasePython -Preferred $BasePython
  if (-not $BasePython) {
    throw "No Python 3.10+ found. Pass -BasePython <path-to-python.exe>."
  }
  Write-Host "==> Creating build venv with uv at $VenvDir"
  Write-LongStepHint "First run downloads a Python environment into .venv-build..."
  $code = Invoke-NativeWithHeartbeat `
    -FilePath $script:UvExe `
    -ArgumentList @("venv", $VenvDir, "--python", $BasePython) `
    -Activity "uv venv" `
    -HeartbeatSeconds 8
  if ($code -ne 0 -or -not (Test-Path $Py)) { throw "Failed to create build venv with uv" }
}

Write-Host "==> Using Python: $Py"

function Test-RuntimeImports {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  # Keep in sync with requirements.txt essentials (incl. virtual camera for OBS).
  & $Py -c "import importlib; bad=[]
for m in ['torch','transformers','diffusers','fastapi','uvicorn','webview','PIL','cv2','pyvirtualcam']:
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

# Run uv pip against the build (or packaged) Python. Faster than python -m pip.
function Invoke-Pip {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$PipArgs,
    [string]$Activity = "uv pip",
    [int]$HeartbeatSeconds = 12,
    [string]$PythonExe = ""
  )
  $targetPy = if ($PythonExe) { $PythonExe } else { $Py }
  $exeArgs = [System.Collections.Generic.List[string]]::new()
  [void]$exeArgs.Add("pip")
  foreach ($a in @($PipArgs)) {
    $s = [string]$a
    if ($s -in @("--disable-pip-version-check", "--progress-bar", "on")) { continue }
    [void]$exeArgs.Add($s)
  }
  [void]$exeArgs.Add("--python")
  [void]$exeArgs.Add($targetPy)
  $code = Invoke-NativeWithHeartbeat `
    -FilePath $script:UvExe `
    -ArgumentList @($exeArgs.ToArray()) `
    -Activity $Activity `
    -HeartbeatSeconds $HeartbeatSeconds
  if ($code -is [System.Array]) {
    $code = $code | Select-Object -Last 1
  }
  return [int]$code
}

function Install-CudaTorch {
  Write-Host "==> Ensuring CUDA torch (cu128)"
  Write-LongStepHint "CUDA wheels are large (often 2+ GB). Downloads can take several minutes."
  Write-LongStepHint "uv shows download progress below."
  $null = Invoke-Pip -PipArgs @("uninstall", "-y", "torch", "torchvision", "torchaudio") `
    -Activity "uninstalling previous torch" `
    -HeartbeatSeconds 8
  # Avoid -q so failures and download progress stay visible. Index-only from pytorch cu128.
  $code = Invoke-Pip -PipArgs @(
    "install", "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cu128"
  ) -Activity "CUDA torch cu128 download/install" -HeartbeatSeconds 10
  if ($code -ne 0) {
    throw "CUDA torch install failed (uv pip exit $code)"
  }
  if (-not (Test-CudaTorch)) {
    throw "CUDA torch not installed (still CPU / import failed). Aborting package."
  }
}

$NeedDeps = -not $SkipDeps
$DidInstallDeps = $false
if (-not $NeedDeps) {
  Write-Host "==> -SkipDeps set - checking imports only"
  if (-not (Test-RuntimeImports) -or -not (Test-CudaTorch)) {
    Write-Host "    imports/CUDA incomplete - installing deps anyway (includes pyvirtualcam)"
    $NeedDeps = $true
  } else {
    Write-Host "    build venv imports OK - skipping uv/torch install"
  }
}

if ($NeedDeps) {
  $DidInstallDeps = $true
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
    "install", "-r", "$Root\requirements.txt"
  ) -Activity "requirements.txt" -HeartbeatSeconds 12
  if ($code -ne 0) { throw "uv pip install failed (exit $code)" }

  # If anything clobbered the CUDA wheel, put it back.
  if (-not (Test-CudaTorch)) {
    Write-Host "==> torch was replaced by a CPU wheel - reinstalling cu128"
    Install-CudaTorch
  }

  Write-Host "==> Ensuring PyInstaller"
  $code = Invoke-Pip -PipArgs @("install", "pyinstaller") `
    -Activity "pyinstaller" -HeartbeatSeconds 10
  if ($code -ne 0) { throw "PyInstaller install failed (exit $code)" }

  # Windows: official PyPI triton has no wheels; torch.compile needs triton-windows.
  # Pin to Triton's minor for the installed torch (2.11 -> 3.6.x).
  Write-Host "==> Ensuring triton-windows for torch.compile"
  $code = Invoke-Pip -PipArgs @(
    "install", "triton-windows>=3.6,<3.7"
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

if ($SkipPackage) {
  Write-Host ""
  Write-Host "==> SkipPackage: deps and UI only (no VTMNoble.exe)"
  Write-Host "    Use run.exe to run the operator desk from source."
  exit 0
}

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
    Write-Host "    refreshing dist except models/ and characters/ (-ForceBundle)"
    Get-ChildItem -LiteralPath $DistDir -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -ne "models" -and $_.Name -ne "characters" } |
      ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
      }
    return
  }

  # Keep runtime/, models/, and characters/ (user DiT drops, VTM packs, session). Wipe the rest.
  Get-ChildItem -LiteralPath $DistDir -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "runtime" -and $_.Name -ne "models" -and $_.Name -ne "characters" } |
    ForEach-Object {
      Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
  Write-Host "    overwriting package (kept runtime/, models/, characters/ - use -ForceBundle to refresh runtime)"
}

Clear-DistSoft -DistDir $Out

# Copy runtime venv
$RuntimeDest = Join-Path $Out "runtime"
$RuntimeMarker = Join-Path $RuntimeDest "Scripts\python.exe"
$RuntimePy = Join-Path $RuntimeDest "Scripts\python.exe"
if ((Test-Path $RuntimeMarker) -and -not $ForceBundle) {
  Write-Host "==> runtime/ already present - skipping venv copy (use -ForceBundle to refresh)"
  # Still sync new wheels (e.g. pyvirtualcam) into the packaged runtime when deps changed.
  if ($DidInstallDeps -and (Test-Path -LiteralPath $RuntimePy)) {
    Write-Host "==> Syncing requirements into existing dist runtime (new packages)"
    $code = Invoke-Pip `
      -PipArgs @("install", "-r", "$Root\requirements.txt") `
      -PythonExe $RuntimePy `
      -Activity "runtime requirements sync" `
      -HeartbeatSeconds 12
    if ($code -ne 0) {
      Write-Host "    WARNING: runtime uv pip sync failed (exit $code) - use -ForceBundle to refresh runtime/"
    } else {
      Write-Host "    runtime packages updated"
    }
  }
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
Write-Host "==> Copying app code (backend, ui/dist, models placeholders)"
$null = Invoke-RobocopyWithProgress `
  -Source (Join-Path $Root "backend") `
  -Dest (Join-Path $Out "backend") `
  -Label "Copying backend/" `
  -ExtraArgs @("/E", "/XD", "__pycache__", ".pytest_cache", "tests", "packaging", "/XF", "*.pyc")
if (Test-Path (Join-Path $Root "ui\dist")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Out "ui\dist") | Out-Null
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "ui\dist") `
    -Dest (Join-Path $Out "ui\dist") `
    -Label "Copying ui/dist/" `
    -ExtraArgs @("/E")
}

# Sync models without deleting custom DiT / refs the user dropped in.
Write-Host "==> Syncing models/ (keep existing checkpoints; do not purge)"
New-Item -ItemType Directory -Force -Path (Join-Path $Out "models\dit") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Out "models\trackers") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Out "models\refs") | Out-Null
if (Test-Path (Join-Path $Root "models\trackers")) {
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "models\trackers") `
    -Dest (Join-Path $Out "models\trackers") `
    -Label "Syncing models/trackers/" `
    -ExtraArgs @("/E", "/XO", "/XF", ".gitkeep")
}
if (Test-Path (Join-Path $Root "models\dit")) {
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "models\dit") `
    -Dest (Join-Path $Out "models\dit") `
    -Label "Syncing models/dit/ (no deletes)" `
    -ExtraArgs @("/E", "/XO", "/XF", ".gitkeep")
}
if (Test-Path (Join-Path $Root "models\refs")) {
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "models\refs") `
    -Dest (Join-Path $Out "models\refs") `
    -Label "Syncing models/refs/" `
    -ExtraArgs @("/E", "/XO", "/XF", ".gitkeep")
}
New-Item -ItemType Directory -Force -Path (Join-Path $Out "characters") | Out-Null
if (Test-Path (Join-Path $Root "characters")) {
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "characters") `
    -Dest (Join-Path $Out "characters") `
    -Label "Syncing characters/" `
    -ExtraArgs @("/E", "/XO", "/XF", ".gitkeep")
}
$sourcesJson = Join-Path $Root "models\model_sources.json"
if (Test-Path -LiteralPath $sourcesJson) {
  Copy-Item -Force $sourcesJson (Join-Path $Out "models\model_sources.json")
}

# Ping Hub for *new* DiT files only; never overwrite or delete local weights.
Write-Host "==> Checking Hugging Face for new DiT checkpoints (skip files already present)"
Write-LongStepHint "Only missing Hub files download. Custom local .pt files are left alone."
$prevPyPath = $env:PYTHONPATH
$env:PYTHONPATH = $Root
try {
  & $Py -m backend.model_download
  if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Hub sync failed - existing local models are unchanged." -ForegroundColor Yellow
  }
} finally {
  if ($null -eq $prevPyPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
  else { $env:PYTHONPATH = $prevPyPath }
}
# Copy any newly downloaded repo dit files into the package (still no deletes).
if (Test-Path (Join-Path $Root "models\dit")) {
  $null = Invoke-RobocopyWithProgress `
    -Source (Join-Path $Root "models\dit") `
    -Dest (Join-Path $Out "models\dit") `
    -Label "Syncing new DiT files into package" `
    -ExtraArgs @("/E", "/XO", "/XF", ".gitkeep")
}

# Lean vendor only (never monorepo tools/ datasets)
$null = Invoke-RobocopyWithProgress `
  -Source (Join-Path $Root "vendor") `
  -Dest (Join-Path $Out "vendor") `
  -Label "Copying vendor/ (lean)" `
  -ExtraArgs @("/E", "/MT:4", "/XD", "__pycache__", ".git", ".pytest_cache", "tests", "Tests", "training", "datasets", "data", "input", "/XF", "*.pyc")

# Tiny launcher exe — also dropped next to run.exe for one-click run.
# Always rebuild unless -SkipLauncher: Explorer "Date modified" on root VTMNoble.exe
# must reflect this Smart Build (skipping left a stale  stamp while dist/ refreshed).
$LauncherExe = Join-Path $Out "VTMNoble.exe"
$RootExe = Join-Path $Root "VTMNoble.exe"
if ($SkipLauncher) {
  Write-Host "==> Launcher skip requested (-SkipLauncher)"
  if (-not (Test-Path -LiteralPath $RootExe)) {
    throw "No VTMNoble.exe at repo root to reuse. Re-run without -SkipLauncher."
  }
  Copy-Item -Force $RootExe $LauncherExe
  $now = Get-Date
  (Get-Item -LiteralPath $RootExe).LastWriteTime = $now
  (Get-Item -LiteralPath $LauncherExe).LastWriteTime = $now
  Write-Host "    stamped Date modified -> $now"
} else {
  Write-Host "==> Building tiny launcher exe (stdlib only - low memory)"
  $null = Invoke-Pip -PipArgs @("install", "pyinstaller") `
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
      (Join-Path $PSScriptRoot "launcher.spec")
    ) `
    -Activity "PyInstaller launcher" `
    -HeartbeatSeconds 10
  if ($code -ne 0) { throw "Launcher PyInstaller failed" }
  $built = Join-Path $Root "build\launcher_dist\VTMNoble.exe"
  if (-not (Test-Path $built)) { throw "Launcher exe not produced" }
  Copy-Item -Force $built $LauncherExe
  Copy-Item -Force $built $RootExe
  Write-Host "    wrote $RootExe"
}

$Bat = @"
@echo off
cd /d "%~dp0"
"%~dp0runtime\Scripts\python.exe" -m backend %*
"@
Set-Content -Path (Join-Path $Out "VTMNoble.bat") -Value $Bat -Encoding ASCII

# Register VTM Noble Cam DirectShow filter (bundled Unity Capture, custom name).
$VcamInstall = Join-Path $Root "vendor\tools\vtm_noble_cam\Install-VTMNobleCam.bat"
if (Test-Path -LiteralPath $VcamInstall) {
  Write-Host "==> Ensuring virtual camera device: VTM Noble Cam"
  Write-LongStepHint "Approve UAC once if Windows asks (DirectShow registration)."
  $prevPyPath = $env:PYTHONPATH
  $env:PYTHONPATH = $Root
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $vcamReady = $false
  try {
    & $Py -c "from backend.vcam_device import device_available; raise SystemExit(0 if device_available() else 1)"
    $vcamReady = ($LASTEXITCODE -eq 0)
  } catch {
    $vcamReady = $false
  }
  if (-not $vcamReady) {
    try {
      Start-Process -FilePath $VcamInstall -WorkingDirectory (Split-Path $VcamInstall -Parent) -Wait -Verb RunAs
      Start-Sleep -Milliseconds 800
      & $Py -c "from backend.vcam_device import device_available; raise SystemExit(0 if device_available() else 1)"
      $vcamReady = ($LASTEXITCODE -eq 0)
    } catch {
      Write-Host "    WARNING: VTM Noble Cam install skipped ($_)" -ForegroundColor Yellow
    }
  }
  $ErrorActionPreference = $prev
  if ($null -eq $prevPyPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
  else { $env:PYTHONPATH = $prevPyPath }
  if ($vcamReady) {
    Write-Host "    VTM Noble Cam ready"
  } else {
    Write-Host "    WARNING: VTM Noble Cam not registered yet - app will prompt on first use" -ForegroundColor Yellow
  }
} else {
  Write-Host "==> WARNING: vendor\tools\vtm_noble_cam missing - virtual camera unavailable" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==> Done"
Write-Host "    Click to run:  $RootExe"
Write-Host "    (same folder as run.exe — you do not need to open dist\ )"
Write-Host "    Package data:  $Out"
Write-Host "    Fallback:      $Out\VTMNoble.bat"
Write-Host "DiT models: Hub ping for new files only; local models/dit is never purged"
Write-Host "Virtual cam: VTM Noble Cam (OBS Video Capture Device)"
Write-Host "Supported GPUs: GeForce RTX 30 / 40 / 50 (CUDA). No AMD."
