# Start the face-tracking lab (API + React UI) using the main app venv.
# One owner for 8780: reuse a live harness, replace an old build, refuse anything else.
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
$FaceModel = Join-Path $Root "models\lm_model3_opt.onnx"

if (-not (Test-Path $Py)) {
  throw "Missing $Py. Run start.bat -> [1] Smart Build first."
}

$env:PYTHONPATH = $Root

function Get-LabPortState {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $raw = & $Py -c 'from backend.bind import probe_state; print(probe_state())' 2>&1
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prev
  }
  if ($code -ne 0) {
    throw "Could not check port 8780.`n$raw"
  }
  $known = @("lab", "stale", "busy", "free")
  $line = @(
    $raw | ForEach-Object { ([string]$_).Trim() } | Where-Object { $known -contains $_ }
  ) | Select-Object -Last 1
  if (-not $line) {
    throw "Could not check port 8780.`n$raw"
  }
  return $line
}

function Clear-StaleLab {
  Write-Host "Old Track Lab on 8780 has no harness -- restarting it."
  & $Py -c 'from backend.bind import kill_listeners; print("killed", kill_listeners())'
  $deadline = (Get-Date).AddSeconds(8)
  while ((Get-Date) -lt $deadline) {
    $now = Get-LabPortState
    if ($now -eq "free" -or $now -eq "lab") { return }
    Start-Sleep -Milliseconds 200
  }
  Write-Host "Port 8780 still looks stale; the new process will replace it."
}

function Wait-LabHealth {
  param([int]$Seconds = 45)
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    $state = Get-LabPortState
    if ($state -eq "lab") { return }
    if ($state -eq "stale") {
      throw "Track Lab came up without /harness/status. Stop it and start again."
    }
    if ($state -eq "busy") {
      throw "Port 8780 is already in use by another app. Close it, then retry."
    }
    Start-Sleep -Milliseconds 250
  }
  throw "Track Lab did not come up on port 8780."
}

Set-Location $Root
if (-not (Test-Path $FaceModel)) {
  Write-Host "Face models missing -- running setup.ps1"
  & powershell -NoProfile -ExecutionPolicy Bypass -File $Setup
}

$state = Get-LabPortState
if ($state -eq "busy") {
  throw "Port 8780 is already in use by another app. Close it, then start Track Lab again."
}
if ($state -eq "stale") {
  Clear-StaleLab
  $state = "free"
}

$api = $null
$owned = $false
if ($state -eq "lab") {
  Write-Host "API already running on http://127.0.0.1:8780 -- reusing it."
} else {
  $api = Start-Process -FilePath $Py -ArgumentList "-m", "backend" -WorkingDirectory $Root -PassThru -WindowStyle Normal
  $owned = $true
  Wait-LabHealth
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
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  npm run dev
  if ($LASTEXITCODE -ne 0) {
    throw "UI exited with code $LASTEXITCODE"
  }
} finally {
  $ErrorActionPreference = $prev
  if ($owned -and $null -ne $api -and -not $api.HasExited) {
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
  }
}
