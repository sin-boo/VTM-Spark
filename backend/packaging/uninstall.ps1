# VTM Spark uninstall (uninstall.bat): stops the app, removes the virtual camera
# and everything install.bat and the app put on this PC, then (if asked) the app
# folder itself. Exit 0 = all removed, 2 = some steps need attention,
# 3 = uninstall.bat deletes the folder once this script has exited.
$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$CamSetup = Join-Path $Root "vendor\tools\vtm_spark_cam\VTM Spark Camera Setup.exe"
$script:Problems = 0

function Confirm-Vtm([string]$question, [bool]$default = $false) {
  if ($default) {
    $answer = Read-Host "$question [Y/n]"
    return $answer -notmatch '^\s*(n|no)\s*$'
  }
  $answer = Read-Host "$question [y/N]"
  return $answer -match '^\s*(y|yes)\s*$'
}

function Remove-VtmPath([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return }
  Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
  # Deep node_modules trees can outlast Remove-Item in Windows PowerShell 5.1.
  if ((Test-Path -LiteralPath $path -PathType Container)) {
    cmd /c "rd /s /q `"$path`" >nul 2>&1" | Out-Null
  }
  if (Test-Path -LiteralPath $path) {
    Write-Host "    could not remove $path (in use?)" -ForegroundColor Yellow
    $script:Problems++
  }
}

# Files in $dir matching $patterns; keeps the .gitkeep placeholders.
function Remove-VtmFiles([string]$dir, [string[]]$patterns) {
  if (-not (Test-Path -LiteralPath $dir)) { return }
  foreach ($pattern in $patterns) {
    Get-ChildItem -LiteralPath $dir -Filter $pattern -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -ne ".gitkeep" } |
      ForEach-Object { Remove-VtmPath $_.FullName }
  }
}

Write-Host ""
Write-Host "VTM Spark - Uninstall" -ForegroundColor Cyan
Write-Host ""
Write-Host "This removes from this PC:"
Write-Host "  - the VTM Spark virtual camera (Windows asks for admin once)"
Write-Host "  - the Python environment, Node.js and uv (.venv-build, .tools)"
Write-Host "  - downloaded models, tracker weights and caches"
Write-Host "  - the built UI, logs and this PC's settings"
Write-Host "  - the desk window's browser data (%LOCALAPPDATA%\pywebview)"
Write-Host ""
Write-Host "Microsoft Edge WebView2 stays: Windows and other apps use it." -ForegroundColor DarkGray
Write-Host ""
if (-not (Confirm-Vtm "Uninstall VTM Spark?")) {
  Write-Host "Nothing was removed."
  exit 0
}
# A git checkout may hold unpushed work, so there the folder stays unless asked.
$isCheckout = Test-Path -LiteralPath (Join-Path $Root ".git")
$removeFolder = $false
if ((Test-Path -LiteralPath (Join-Path $Root "install.bat")) -and
    (Test-Path -LiteralPath (Join-Path $Root "backend\__main__.py"))) {
  if ($isCheckout) {
    Write-Host "This folder is a git checkout: deleting it also deletes any work you have not pushed." -ForegroundColor Yellow
  }
  $removeFolder = Confirm-Vtm "Also delete the VTM Spark folder itself, with your characters ($Root)?" (-not $isCheckout)
}
$removeUserData = $removeFolder
if (-not $removeFolder) {
  $removeUserData = Confirm-Vtm "Also delete your characters, reference images and outputs?"
}
$removeUvCache = Confirm-Vtm "Also clear uv's download cache and its managed Python (shared with other uv projects; frees several GB)?"
Write-Host ""

Write-Host "==> Stopping VTM Spark" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "stop-app.ps1")

Write-Host "==> Removing the virtual camera" -ForegroundColor Cyan
$camDir = Join-Path ${env:ProgramW6432} "VTM Spark"
if (-not ${env:ProgramW6432}) { $camDir = Join-Path ${env:ProgramFiles} "VTM Spark" }
if (Test-Path -LiteralPath $CamSetup) {
  Write-Host "    Windows will ask for administrator permission ('VTM Spark Camera Setup'). Click Yes." -ForegroundColor DarkGray
  try {
    $p = Start-Process -FilePath $CamSetup -ArgumentList "--uninstall" -WorkingDirectory (Split-Path $CamSetup -Parent) -Wait -PassThru -Verb RunAs
    if ($p.ExitCode -eq 0) {
      Write-Host "    virtual camera removed" -ForegroundColor Green
      if (Test-Path -LiteralPath $camDir) {
        Write-Host "    $camDir goes at the next restart (an app still has the camera open)" -ForegroundColor DarkGray
      }
    } else {
      Write-Host "    camera removal failed (exit $($p.ExitCode))" -ForegroundColor Yellow
      $script:Problems++
    }
  } catch {
    Write-Host "    skipped - the Windows permission prompt was declined" -ForegroundColor Yellow
    $script:Problems++
  }
} else {
  Write-Host "    camera setup missing ($CamSetup) - skipped" -ForegroundColor Yellow
  $script:Problems++
}

Write-Host "==> Removing the Python environment, tools and built UI" -ForegroundColor Cyan
foreach ($rel in @(
    ".venv-build", ".venv", ".tools",
    "ui\node_modules", "ui\dist", "ui\public\_early_splash.html",
    "track_lab\.venv", "track_lab\ui\node_modules", "track_lab\ui\dist", "track_lab\output"
  )) {
  Remove-VtmPath (Join-Path $Root $rel)
}

Write-Host "==> Removing downloaded models and caches" -ForegroundColor Cyan
foreach ($rel in @("models\cache", "models\decoder", "models\iamge")) {
  Remove-VtmPath (Join-Path $Root $rel)
}
Remove-VtmFiles (Join-Path $Root "models\dit") @("*")
Remove-VtmFiles (Join-Path $Root "models\trackers") @("*.pt", "*.pth", "*.onnx", "*.task")
Remove-VtmFiles (Join-Path $Root "track_lab\models") @("*")
Remove-VtmFiles (Join-Path $Root "vendor\tools\openseeface\models") @("*.onnx", "priorbox_640x640.json")
Remove-VtmFiles (Join-Path $Root "vendor\tools\live-poser\models") @("*.pt", "*.onnx", "*.task")
Get-ChildItem -LiteralPath (Join-Path $Root "vendor\tools\pose-traker") -Recurse -File -Force -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -in @("iris_pose.pt", "dwpose_v2.pt", "dwpose_v2.onnx") } |
  ForEach-Object { Remove-VtmPath $_.FullName }

Write-Host "==> Removing logs, settings and caches" -ForegroundColor Cyan
Remove-VtmFiles (Join-Path $Root "models") @(
  "*.lock", "*.log", "extra_checkpoint_dirs.json", "session.json", "load_timings.json",
  "ui_prefs.json", "gpu.json", "gpu_check.json", "hw_profile.json"
)
Remove-VtmFiles $Root @("*.log")
Get-ChildItem -LiteralPath $Root -Recurse -Directory -Force -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache") } |
  ForEach-Object { Remove-VtmPath $_.FullName }
foreach ($base in @(${env:LOCALAPPDATA}, ${env:APPDATA})) {
  if ($base) { Remove-VtmPath (Join-Path $base "pywebview") }
}

if ($removeUserData -and -not $removeFolder) {
  Write-Host "==> Removing your characters, references and outputs" -ForegroundColor Cyan
  foreach ($rel in @("characters", "models\refs", "models\blendshapes", "track_lab\input")) {
    Remove-VtmFiles (Join-Path $Root $rel) @("*")
  }
  Remove-VtmPath (Join-Path $Root "outputs")
}

if ($removeUvCache) {
  Write-Host "==> Clearing uv's cache and managed Python" -ForegroundColor Cyan
  if (${env:LOCALAPPDATA}) { Remove-VtmPath (Join-Path ${env:LOCALAPPDATA} "uv\cache") }
  if (${env:APPDATA}) { Remove-VtmPath (Join-Path ${env:APPDATA} "uv\python") }
}

Write-Host ""
if ($script:Problems -eq 0) {
  Write-Host "VTM Spark is uninstalled." -ForegroundColor Green
} else {
  Write-Host "VTM Spark is uninstalled, but $($script:Problems) item(s) need attention (see above)." -ForegroundColor Yellow
  if (-not $removeFolder) {
    Write-Host "Close anything using VTM Spark and run uninstall.bat again." -ForegroundColor Yellow
  }
}
Write-Host ""
if ($removeFolder) {
  # Neither this script nor uninstall.bat can delete the folder while running
  # from it; uninstall.bat does it as its last step.
  Write-Host "The folder $Root is deleted when you press a key."
  exit 3
}
Write-Host "The app folder stays: $Root"
Write-Host ""
if ($script:Problems -eq 0) { exit 0 }
exit 2
