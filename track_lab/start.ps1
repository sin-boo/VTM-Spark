# Start the face-tracking lab (API + React UI) using the main app venv.
# API and UI are one process tree (backend\pair.py): close either and both stop.
# Keep this file ASCII. Windows PowerShell 5.1 reads .ps1 as ANSI unless it has a BOM,
# so a UTF-8 em dash becomes a stray quote and the script never parses.

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Root
$Py = Join-Path $Repo ".venv-build\Scripts\python.exe"
$Ui = Join-Path $Root "ui"
$Setup = Join-Path $Root "setup.ps1"
$FaceModel = Join-Path $Repo "vendor\tools\openseeface\models\lm_model3_opt.onnx"
$OsfPy = Join-Path $Root "osf\tracker.py"

if (-not (Test-Path $Py)) {
  throw "Missing $Py. Run start.bat -> [1] Smart Build first."
}

$env:PYTHONPATH = $Root

Set-Location $Root
if (-not (Test-Path $OsfPy)) {
  Write-Host "OpenSeeFace Python missing -- running setup.ps1"
  & powershell -NoProfile -ExecutionPolicy Bypass -File $Setup
}
if (-not (Test-Path $FaceModel)) {
  throw "Face models missing in vendor\tools\openseeface\models. Run start.bat -> [1] Smart Build first."
}

Set-Location $Ui
if (-not (Test-Path (Join-Path $Ui "node_modules"))) {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    npm install
    if ($LASTEXITCODE -ne 0) {
      throw "npm install failed with code $LASTEXITCODE"
    }
  } finally {
    $ErrorActionPreference = $prev
  }
}
Write-Host "API      http://127.0.0.1:8780"
Write-Host "UI       http://127.0.0.1:5174"
Write-Host "Harness  ws://127.0.0.1:8780/harness/ws"
Write-Host "Python   $Py"
Write-Host "Put a still at track_lab\input\source.png then press Track."
Write-Host "API and UI run together: closing this window or either one stops both."
# One process: the API serves here and the UI is its child (backend\pair.py).
Set-Location $Root
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & $Py -m backend.pair
  $code = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $prev
}
if ($code -ne 0) {
  throw "Track Lab exited with code $code"
}
