# Sync lean runtime deps into vendor/ (self-contained GitHub / package source).
# Pulls from the monorepo parent when present; safe to re-run.
param(
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Repo = Split-Path $Root -Parent
$Vendor = Join-Path $Root "vendor"

Write-Host "==> Syncing lean vendor into $Vendor"

function Copy-Lean {
  param(
    [string]$Src,
    [string]$Dest,
    [string[]]$ExcludeDirs = @()
  )
  if (-not (Test-Path -LiteralPath $Src)) {
    Write-Host "    skip missing: $Src"
    return
  }
  New-Item -ItemType Directory -Force -Path $Dest | Out-Null
  $xd = @(".git", ".pytest_cache", "__pycache__", ".venv", "debug_frames", "training", "datasets", "tests", "Tests") + $ExcludeDirs
  $xdArgs = @()
  foreach ($d in $xd) { $xdArgs += @("/XD", $d) }
  & robocopy $Src $Dest /E /MT:4 @xdArgs /XF *.pyc *.pyo /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
  Write-Host "    synced $Src -> $Dest"
}

# --- torch_train (inference only) ------------------------------------------
$TorchSrc = Join-Path $Repo "send2pod\torch_train"
$TorchDest = Join-Path $Vendor "torch_train"
if ($Force -and (Test-Path $TorchDest)) { Remove-Item -Recurse -Force $TorchDest }
Copy-Lean -Src $TorchSrc -Dest $TorchDest -ExcludeDirs @("datasets", "training", "tests", ".pytest_cache")

# --- live-poser ------------------------------------------------------------
$LpSrc = Join-Path $Repo "tools\live-poser"
$LpDest = Join-Path $Vendor "tools\live-poser"
if ($Force -and (Test-Path $LpDest)) { Remove-Item -Recurse -Force $LpDest }
Copy-Lean -Src $LpSrc -Dest $LpDest

# --- OpenSeeFace (flattened) -----------------------------------------------
$OsfSrc = Join-Path $Repo "tools\vedio traker\OpenSeeFace"
$OsfDest = Join-Path $Vendor "tools\openseeface"
if ($Force -and (Test-Path $OsfDest)) { Remove-Item -Recurse -Force $OsfDest }
Copy-Lean -Src $OsfSrc -Dest $OsfDest -ExcludeDirs @("Unity", "Examples", "Images", ".github")

# --- lean pose-traker (ref-fit helpers + anime detector src) ---------------
$PtDest = Join-Path $Vendor "tools\pose-traker"
New-Item -ItemType Directory -Force -Path $PtDest | Out-Null
$PtSrc = Join-Path $Repo "tools\pose-traker"
foreach ($file in @("face_landmark_repair.py", "hrnet_schema_adapter.py")) {
  $src = Join-Path $PtSrc $file
  if (Test-Path $src) { Copy-Item -Force $src (Join-Path $PtDest $file) }
}
$AfdSrc = Join-Path $PtSrc "anime-face-detector"
$AfdDest = Join-Path $PtDest "anime-face-detector"
if (Test-Path $AfdSrc) {
  New-Item -ItemType Directory -Force -Path $AfdDest | Out-Null
  robocopy (Join-Path $AfdSrc "src") (Join-Path $AfdDest "src") /E /XD __pycache__ .pytest_cache /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
  foreach ($extra in @("pyproject.toml", "README.md", "LICENSE")) {
    $p = Join-Path $AfdSrc $extra
    if (Test-Path $p) { Copy-Item -Force $p (Join-Path $AfdDest $extra) }
  }
}

# --- tracker weights (small, required) -------------------------------------
$Trackers = Join-Path $Root "models\trackers"
$LpModels = Join-Path $LpDest "models"
$PtModels = Join-Path $PtDest "models"
$PtIris = Join-Path $PtDest "iris-model\models"
New-Item -ItemType Directory -Force -Path $Trackers, $LpModels, $PtModels, $PtIris | Out-Null

$weightPairs = @(
  @{ Src = Join-Path $Repo "tools\pose-traker\models\iris_pose.pt"; Names = @("iris_pose.pt") },
  @{ Src = Join-Path $Repo "tools\live-poser\models\dwpose_v2.onnx"; Names = @("dwpose_v2.onnx") },
  @{ Src = Join-Path $Repo "tools\live-poser\models\pose_landmarker_lite.task"; Names = @("pose_landmarker_lite.task") },
  @{ Src = Join-Path $Repo "tools\pose-traker\iris-model\models\dwpose_v2.pt"; Names = @("dwpose_v2.pt") }
)

foreach ($w in $weightPairs) {
  if (-not (Test-Path -LiteralPath $w.Src)) {
    Write-Host "    missing weight: $($w.Src)"
    continue
  }
  $name = Split-Path $w.Src -Leaf
  Copy-Item -Force $w.Src (Join-Path $Trackers $name)
  Copy-Item -Force $w.Src (Join-Path $LpModels $name)
  if ($name -eq "iris_pose.pt") {
    Copy-Item -Force $w.Src (Join-Path $PtModels $name)
    Copy-Item -Force $w.Src (Join-Path $PtIris $name)
  }
  if ($name -like "dwpose*") {
    Copy-Item -Force $w.Src (Join-Path $PtIris $name)
  }
  Write-Host "    weight $name"
}

# Ensure dit folder exists (empty — downloads only)
New-Item -ItemType Directory -Force -Path (Join-Path $Root "models\dit") | Out-Null
if (-not (Test-Path (Join-Path $Root "models\dit\.gitkeep"))) {
  Set-Content -Path (Join-Path $Root "models\dit\.gitkeep") -Value ""
}

Write-Host "==> Vendor sync done"
Write-Host "    torch_train marker: $(Test-Path (Join-Path $Vendor 'torch_train\inference_keypoint.py'))"
Write-Host "    live-poser: $(Test-Path (Join-Path $Vendor 'tools\live-poser\live_poser.py'))"
Write-Host "    openseeface: $(Test-Path (Join-Path $Vendor 'tools\openseeface\tracker.py'))"
