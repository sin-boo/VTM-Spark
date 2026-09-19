# Copy OpenSeeFace Python + face models into track_lab.
# Uses the main app venv (.venv-build). Does not create a lab-only venv.

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Root
$SrcPy = Join-Path $Repo ".venv-build\Scripts\python.exe"
$SrcOs = Join-Path $Repo "vendor\tools\openseeface"
$DstOs = Join-Path $Root "osf"
$SrcModels = Join-Path $SrcOs "models"
$DstModels = Join-Path $Root "models"

if (-not (Test-Path $SrcPy)) {
  throw "Missing $SrcPy. Run start.bat -> [1] Smart Build first."
}

Write-Host "==> Copy OpenSeeFace Python"
New-Item -ItemType Directory -Force -Path $DstOs | Out-Null
@(
  "tracker.py",
  "retinaface.py",
  "similaritytransform.py",
  "remedian.py"
) | ForEach-Object {
  Copy-Item (Join-Path $SrcOs $_) (Join-Path $DstOs $_) -Force
}

Write-Host "==> Copy face models"
New-Item -ItemType Directory -Force -Path $DstModels | Out-Null
Get-ChildItem $SrcModels -File | ForEach-Object {
  $dest = Join-Path $DstModels $_.Name
  if (-not (Test-Path $dest)) {
    Write-Host "  $($_.Name)"
    Copy-Item $_.FullName $dest
  }
}

Write-Host "==> Probe main venv imports"
& $SrcPy -c "import torch, cv2, onnxruntime, numpy, fastapi, uvicorn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ok')"
if ($LASTEXITCODE -ne 0) {
  throw "Main venv is missing lab packages. Run start.bat -> [1] Smart Build."
}
Write-Host "DONE. Start with track_lab\start.bat (uses .venv-build)."
