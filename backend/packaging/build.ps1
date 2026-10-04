# VTM Spark setup: builds the UI and ensures .venv-build has deps (CUDA torch for this GPU last).
# run.exe runs the desk from source with that venv.
# DiT weights download into models/dit on install.bat / first launch.
param(
  [string]$BasePython = "",
  [switch]$RecreateVenv,
  [switch]$SkipDeps,
  [switch]$SkipVendorSync,
  [switch]$SyncVendor
)

$ErrorActionPreference = "Stop"
$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
Set-Location -LiteralPath $Root
$VenvDir = Join-Path $Root ".venv-build"
$Py = Join-Path $VenvDir "Scripts\python.exe"

# Pinned uv release (downloaded into .tools\ only when uv is not already on PATH).
$UvVersion = "0.12.9"
$UvZipUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
# From https://github.com/astral-sh/uv/releases/download/0.12.9/uv-x86_64-pc-windows-msvc.zip.sha256
$UvZipSha256 = "ddbfcee1ac615a0499f6aa97b5ec8ebdf3ee4a7714a48055ec2ba0030e3cf810"
# Managed Python that uv downloads when no usable system Python is found.
$ManagedPythonVersion = "3.13"

. (Join-Path $PSScriptRoot "console-progress.ps1")
. (Join-Path $PSScriptRoot "venv-home.ps1")
. (Join-Path $PSScriptRoot "node-tools.ps1")
. (Join-Path $PSScriptRoot "install-stamps.ps1")

function Get-UvExe {
  $cmd = Get-Command uv -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) { return [string]$cmd.Source }
  $local = Join-Path $Root ".tools\uv.exe"
  if (Test-Path -LiteralPath $local) { return $local }
  Write-Host "==> Downloading uv $UvVersion into .tools\ (one-time)"
  $tools = Join-Path $Root ".tools"
  New-Item -ItemType Directory -Force -Path $tools | Out-Null
  # Staged inside the app folder, not %TEMP%.
  $stage = Join-Path $tools "downloads"
  New-Item -ItemType Directory -Force -Path $stage | Out-Null
  $tag = [guid]::NewGuid().ToString("N")
  $zip = Join-Path $stage "uv-$tag.zip"
  $extract = Join-Path $stage "uv-extract-$tag"
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    try {
      Invoke-WebRequest -Uri $UvZipUrl -OutFile $zip -UseBasicParsing
    } catch {
      throw "Could not download uv $UvVersion from $UvZipUrl ($($_.Exception.Message)). Check your internet connection / proxy and re-run install.bat, or install uv yourself (https://docs.astral.sh/uv/) so it is on PATH."
    }
    $hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $UvZipSha256) {
      throw "Checksum mismatch for the uv $UvVersion download (got $hash) - it was damaged or altered. Re-run install.bat."
    }
    Expand-Archive -Path $zip -DestinationPath $extract -Force
    $found = Get-ChildItem -Path $extract -Filter "uv.exe" -Recurse | Select-Object -First 1
    if (-not $found) {
      throw "uv.exe missing from the Astral download ($UvZipUrl). Delete .tools\ and re-run install.bat, or install uv yourself so it is on PATH."
    }
    Copy-Item -Force $found.FullName $local
    $uvx = Get-ChildItem -Path $extract -Filter "uvx.exe" -Recurse | Select-Object -First 1
    if ($uvx) { Copy-Item -Force $uvx.FullName (Join-Path $tools "uvx.exe") }
  } finally {
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $extract) { Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue }
  }
  return $local
}

$script:UvExe = Get-UvExe
Write-Host "==> Using uv: $($script:UvExe)"

Write-Host "==> VTM Spark setup (UI + .venv-build)"

# --- NVIDIA GPU / driver (warn only: cu128 wheels still install without one) ---
function Test-NvidiaDriver {
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return $true }
  if (${env:SystemRoot} -and (Test-Path -LiteralPath (Join-Path ${env:SystemRoot} "System32\nvidia-smi.exe"))) { return $true }
  return (Test-Path -LiteralPath "C:\Windows\System32\nvidia-smi.exe")
}

if (Test-NvidiaDriver) {
  Write-Host "==> NVIDIA driver found (nvidia-smi)"
} else {
  Write-Host ""
  Write-Host "    WARNING: nvidia-smi not found - no NVIDIA GPU driver detected."
  Write-Host "    WARNING: VTM Spark needs an NVIDIA GPU + current NVIDIA driver to run CUDA torch."
  Write-Host "    WARNING: Setup will continue (CUDA torch wheels still install), but the desk will not"
  Write-Host "    WARNING: run until you install a driver from https://www.nvidia.com/Download/index.aspx"
  Write-Host ""
}

Write-Host "==> Killing leftover VTM Spark / backend processes"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "kill-orphans.ps1")

# --- Vendor: use committed vendor/ by default (opt-in -SyncVendor for monorepo) ----
$MonorepoMarker = Join-Path (Split-Path $Root -Parent) "send2pod\torch_train"
$HasMonorepo = Test-Path -LiteralPath $MonorepoMarker
$DoSync = $SyncVendor -and (-not $SkipVendorSync)
if ($DoSync) {
  if (-not $HasMonorepo) {
    throw "-SyncVendor requires parent monorepo (send2pod/torch_train). Standalone clones use committed vendor/ - re-run install.bat without -SyncVendor."
  }
  Write-Host "==> Syncing lean vendor/ from monorepo parent (-SyncVendor)"
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "sync-vendor.ps1")
  if ($LASTEXITCODE -ne 0) { throw "vendor sync failed - see sync-vendor.ps1 output above; fix the monorepo checkout or re-run without -SyncVendor to use the committed vendor/." }
} else {
  Write-Host "==> Using committed vendor/ (pass -SyncVendor only when refreshing from monorepo)"
}
if (-not (Test-Path (Join-Path $Root "vendor\torch_train\inference_keypoint.py"))) {
  throw "vendor/torch_train missing - re-clone the repo (vendor/ must be present), then re-run install.bat"
}
if (-not (Test-Path (Join-Path $Root "vendor\tools\live-poser\live_poser.py"))) {
  throw "vendor/tools/live-poser missing - re-clone the repo (vendor/ must be present), then re-run install.bat"
}

# --- UI --------------------------------------------------------------------
if (-not (Use-VtmNode -RepoRoot $Root -Install)) {
  throw "Could not download Node.js $NodeVersion into .tools\node (needed to build the UI). Check your internet connection / proxy and re-run install.bat."
}
Write-Host "==> Using Node.js: $((Get-Command node -ErrorAction SilentlyContinue).Source)"
Write-Host "==> Building UI (Vite)"
Push-Location "$Root\ui"
# Vite's live progress moves the console cursor and the next lines print over
# its summary. CI=true makes it log plainly (it checks isTTY && !CI).
$prevCi = $env:CI
$env:CI = "true"
try {
  function Install-UiPackages {
    $npmCode = Invoke-ProcessWithHeartbeat `
      -FilePath "cmd.exe" `
      -ArgumentList @("/c", "npm", "ci") `
      -Activity "npm ci" `
      -HeartbeatSeconds 12
    if ($npmCode -ne 0) {
      $npmCode = Invoke-ProcessWithHeartbeat `
        -FilePath "cmd.exe" `
        -ArgumentList @("/c", "npm", "install") `
        -Activity "npm install" `
        -HeartbeatSeconds 12
    }
    if ($npmCode -eq 0) { Save-NpmStamp (Join-Path $Root "ui") }
    return [int]$npmCode
  }
  function Reset-UiPackages {
    # A half-finished or corrupt node_modules is the usual npm failure; start clean.
    if (Test-Path "node_modules") {
      Write-Host "    Clearing ui\node_modules and retrying..."
      Remove-Item -LiteralPath "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
    }
    return (Install-UiPackages)
  }

  # An update can bring a new package-lock.json; node_modules has to follow it.
  $UiDir = Join-Path $Root "ui"
  if (-not (Test-NpmPackagesCurrent $UiDir)) {
    if (Test-Path "node_modules") {
      Write-LongStepHint "UI packages changed - reinstalling UI npm packages..."
    } else {
      Write-LongStepHint "Installing UI npm packages (first run can take a few minutes)..."
    }
    $code = Install-UiPackages
    if ($code -ne 0) { $code = Reset-UiPackages }
    if ($code -ne 0) { throw "npm install failed twice - see npm output above. Check your internet connection and re-run install.bat." }
  }
  $UiHash = Get-UiSourceHash $Root
  $NeedUiBuild = -not (Test-UiBuildCurrent $Root)
  if (-not $NeedUiBuild) {
    Write-Host "    ui/dist is up to date - skipping vite build"
  }
  if ($NeedUiBuild) {
    Write-LongStepHint "Running Vite production build..."
    $code = Invoke-ProcessWithHeartbeat `
      -FilePath "cmd.exe" `
      -ArgumentList @("/c", "npm", "run", "build") `
      -Activity "vite build" `
      -HeartbeatSeconds 10
    if ($code -ne 0) {
      Write-Host "    UI build failed - reinstalling UI packages and building again"
      $code = Reset-UiPackages
      if ($code -eq 0) {
        $code = Invoke-ProcessWithHeartbeat `
          -FilePath "cmd.exe" `
          -ArgumentList @("/c", "npm", "run", "build") `
          -Activity "vite build (retry)" `
          -HeartbeatSeconds 10
      }
    }
    if ($code -ne 0) { throw "UI build failed twice - see npm output above. Re-run install.bat; if it keeps failing, report the error above." }
    Save-VtmStamp (Get-UiBuildStampPath $Root) $UiHash
  }
} finally {
  $env:CI = $prevCi
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
      throw "BasePython path is not a usable Python 3.10+: $Preferred. Point -BasePython at a working python.exe, or omit it to let uv download Python $ManagedPythonVersion."
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
  if ($BasePython) {
    $VenvPython = $BasePython
  } else {
    # No usable system Python: let uv download a managed CPython instead.
    Write-Host "==> No system Python 3.10+ found - uv will download a managed Python $ManagedPythonVersion"
    $VenvPython = $ManagedPythonVersion
  }
  Write-Host "==> Creating build venv with uv at $VenvDir (python: $VenvPython)"
  Write-LongStepHint "First run downloads a Python environment into .venv-build..."
  $code = Invoke-NativeWithHeartbeat `
    -FilePath $script:UvExe `
    -ArgumentList @("venv", $VenvDir, "--python", $VenvPython) `
    -Activity "uv venv" `
    -HeartbeatSeconds 8
  if ($code -is [System.Array]) { $code = $code | Select-Object -Last 1 }
  if ($code -ne 0 -or -not (Test-Path $Py)) {
    # A half-made venv or a broken system Python: start clean on uv's own Python.
    Write-Host "==> Venv creation failed - clearing .venv-build and retrying with uv's managed Python $ManagedPythonVersion"
    Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
    $VenvPython = $ManagedPythonVersion
    $code = Invoke-NativeWithHeartbeat `
      -FilePath $script:UvExe `
      -ArgumentList @("venv", $VenvDir, "--python", $VenvPython) `
      -Activity "uv venv (retry)" `
      -HeartbeatSeconds 8
    if ($code -is [System.Array]) { $code = $code | Select-Object -Last 1 }
  }
  if ($code -ne 0 -or -not (Test-Path $Py)) {
    throw "Failed to create build venv with uv (python: $VenvPython, exit $code) - see uv output above. Check your internet connection and re-run install.bat."
  }
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

# torch build tag (cu128 / cu126) of the installed wheel matches $Build.
function Test-TorchBuild {
  param([string]$Build)
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $ver = (& $Py -c "import torch; print(torch.__version__)" 2>$null | Select-Object -Last 1)
  $ErrorActionPreference = $prev
  return ("$ver".Trim() -match ("\+" + [regex]::Escape($Build) + "$"))
}

# backend.gpu_check (stdlib only, so it runs before torch exists): which torch build
# fits this graphics card, and whether the card really runs the installed one.
function Invoke-GpuCheck {
  param([string[]]$CliArgs)
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $out = & $Py -c "import sys; sys.path.insert(0, sys.argv.pop(1)); from backend.gpu_check import main; raise SystemExit(main(sys.argv[1:]))" $Root @CliArgs 2>&1
  $code = $LASTEXITCODE
  $ErrorActionPreference = $prev
  return @{ Code = $code; Out = @($out | ForEach-Object { "$_" }) }
}

function Install-CudaTorch {
  param([string]$Build = "cu128")
  Write-Host "==> Ensuring CUDA torch ($Build)"
  Write-LongStepHint "CUDA wheels are large (often 2+ GB). Downloads can take several minutes."
  Write-LongStepHint "uv shows download progress below."
  # Remove an old (e.g. CPU-only) torch first, but only what is really there:
  # on a fresh venv uv would just warn "Skipping torch as it is not installed".
  $sitePackages = Join-Path $VenvDir "Lib\site-packages"
  $oldTorch = @("torch", "torchvision", "torchaudio") |
    Where-Object { Test-Path -LiteralPath (Join-Path $sitePackages $_) }
  if ($oldTorch.Count -gt 0) {
    $null = Invoke-Pip -PipArgs (@("uninstall") + $oldTorch) `
      -Activity "removing the previous torch ($($oldTorch -join ', '))" `
      -HeartbeatSeconds 8
  }
  # Avoid -q so failures and download progress stay visible. Index-only from pytorch.
  $code = Invoke-Pip -PipArgs @(
    "install", "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/$Build"
  ) -Activity "CUDA torch $Build download/install" -HeartbeatSeconds 10
  if ($code -ne 0) {
    throw "CUDA torch install failed (uv pip exit $code) - see uv output above. Check your internet connection and free disk space (2+ GB), then re-run install.bat."
  }
  if (-not (Test-CudaTorch)) {
    throw "CUDA torch not installed (still CPU / import failed) - see torch probe above. Delete .venv-build and re-run install.bat."
  }
}

# CUDA 12.8 builds run RTX 20 and newer (Blackwell too) but dropped GTX 900 / 10;
# CUDA 12.6 builds still run those. Pick by the card nvidia-smi reports.
$TorchBuild = "cu128"
$want = Invoke-GpuCheck @("want-build")
$wantLine = ($want.Out | Where-Object { $_ -match '^cu\d+$' } | Select-Object -Last 1)
if ($want.Code -eq 0 -and $wantLine) { $TorchBuild = "$wantLine".Trim() }
Write-Host "==> torch build for this graphics card: $TorchBuild"

# requirements.txt (and the torch build) the venv was last installed from.
# An update that adds or bumps a package changes it, so -SkipDeps installs anyway.
$DepsStamp = Join-Path $VenvDir ".vtm-requirements"
$DepsHash = "$(Get-VtmFileHash (Join-Path $Root "requirements.txt")) $TorchBuild"
$NeedDeps = -not $SkipDeps
$DidInstallDeps = $false
if (-not $NeedDeps) {
  Write-Host "==> -SkipDeps set - checking imports only"
  if (-not (Test-VtmStamp $DepsStamp $DepsHash)) {
    Write-Host "    requirements.txt changed since the last install - installing deps"
    $NeedDeps = $true
  } elseif (-not (Test-RuntimeImports) -or -not (Test-CudaTorch) -or -not (Test-TorchBuild $TorchBuild)) {
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
  if (-not (Test-CudaTorch) -or -not (Test-TorchBuild $TorchBuild)) {
    Install-CudaTorch -Build $TorchBuild
  } else {
    Write-Host "==> CUDA torch wheel ($TorchBuild) already present"
  }

  Write-Host "==> Installing requirements (torch already provided by the $TorchBuild wheel)"
  Write-LongStepHint "Installing Python deps from requirements.txt (can take a few minutes)..."
  $code = Invoke-Pip -PipArgs @(
    "install", "-r", "$Root\requirements.txt"
  ) -Activity "requirements.txt" -HeartbeatSeconds 12
  if ($code -ne 0) { throw "uv pip install -r requirements.txt failed (exit $code) - see uv output above. Check your internet connection and re-run install.bat; if it keeps failing, delete .venv-build and re-run." }

  # If anything clobbered the CUDA wheel, put it back.
  if (-not (Test-CudaTorch) -or -not (Test-TorchBuild $TorchBuild)) {
    Write-Host "==> torch was replaced by another wheel - reinstalling $TorchBuild"
    Install-CudaTorch -Build $TorchBuild
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

if (-not (Test-RuntimeImports)) { throw "Build venv is missing required packages - re-run install.bat without -SkipDeps; if it persists, delete .venv-build and re-run install.bat." }
if (-not (Test-CudaTorch)) { throw "Build venv must have CUDA torch - delete .venv-build and re-run install.bat." }
Write-Host "    runtime imports OK (CUDA torch)"
if ($DidInstallDeps) { Save-VtmStamp $DepsStamp $DepsHash }

# A CUDA wheel is not a working card: make it run real work. If it fails and the
# other torch build supports the card, put that one in and test again. What was
# tried goes to models\gpu_check.json; the desk shows the same story at start.
Write-Host "==> Testing the graphics card"
$gpu = Invoke-GpuCheck @("verify")
$retryLine = ($gpu.Out | Where-Object { $_ -match '^retry:cu\d+$' } | Select-Object -Last 1)
$gpu.Out | Where-Object { $_ -notmatch '^retry:' } | ForEach-Object { Write-Host $_ }
if ($gpu.Code -eq 3 -and $retryLine) {
  $retry = "$retryLine".Trim() -replace '^retry:', ''
  Write-Host "==> Trying the $retry torch build for this graphics card"
  Install-CudaTorch -Build $retry
  $gpu = Invoke-GpuCheck @("verify")
  $gpu.Out | Where-Object { $_ -notmatch '^retry:' } | ForEach-Object { Write-Host $_ }
}
if ($gpu.Code -ne 0) {
  Write-Host "    The desk will show this when it starts. The rest of the install continues."
}

# The Ultra fast stream decoder comes down from the Hub with the DiT.
# Without it the stream still runs, decoding with the slower TinyVAE.
$FastDecoder = Join-Path $Root "models\decoder\vtm-fast-decoder.pt"
if ((Test-Path -LiteralPath $FastDecoder) -and (Get-Item -LiteralPath $FastDecoder).Length -ge 1000000) {
  Write-Host "    fast decoder OK (models\decoder\vtm-fast-decoder.pt)"
} else {
  Write-Host "    WARNING: models\decoder\vtm-fast-decoder.pt is missing - VTM Spark downloads it on its next start; until then the stream decodes with the slower TinyVAE."
}

Write-Host ""
Write-Host "==> Done - use run.exe to open the operator desk."
