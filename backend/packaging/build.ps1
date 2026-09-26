# VTM Noble setup: builds the UI and ensures .venv-build has deps (CUDA torch cu128 last).
# run.exe runs the desk from source with that venv.
# DiT weights download into models/dit on Smart Build / first launch.
param(
  [string]$BasePython = "",
  [switch]$RecreateVenv,
  [switch]$SkipDeps,
  [switch]$SkipVendorSync,
  [switch]$SyncVendor
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
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

Write-Host "==> VTM Noble setup (UI + .venv-build)"

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

Write-Host ""
Write-Host "==> Done - use run.exe to open the operator desk."
