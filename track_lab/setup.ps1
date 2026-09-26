# Copy OpenSeeFace Python into track_lab. Face weights stay in
# vendor\tools\openseeface\models and models\trackers.
# Uses the main app venv (.venv-build). Does not create a lab-only venv.

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Root
$SrcPy = Join-Path $Repo ".venv-build\Scripts\python.exe"
$SrcOs = Join-Path $Repo "vendor\tools\openseeface"
$DstOs = Join-Path $Root "osf"

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

Write-Host "==> Probe main venv imports"
& $SrcPy -c "import torch, cv2, onnxruntime, numpy, fastapi, uvicorn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ok')"
if ($LASTEXITCODE -ne 0) {
  throw "Main venv is missing lab packages. Run start.bat -> [1] Smart Build."
}
Write-Host "DONE. Start with track_lab\start.bat (uses .venv-build)."
