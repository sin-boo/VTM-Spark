# Bake splash-mark.png onto run.exe with a transparent ICO. A .bat cannot carry an Explorer icon.
$ErrorActionPreference = "Stop"
$Pack = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $Pack "..\..")).Path
$Png = Join-Path $Root "ui\public\splash-mark.png"
$Ico = Join-Path $Pack "run.ico"
$Cs = Join-Path $Pack "run-stub.cs"
$Out = Join-Path $Root "run.exe"
$Csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $Csc)) {
  $Csc = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
}
if (-not (Test-Path -LiteralPath $Csc)) {
  throw "csc.exe not found. Need .NET Framework 4 to stamp the hat on run.exe"
}
if (-not (Test-Path -LiteralPath $Png)) { throw "missing $Png" }

$py = Join-Path $Root ".venv-build\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) { $py = "python" }
$env:PYTHONPATH = $Root
& $py -m backend.app_icon $Png $Ico
if ($LASTEXITCODE -ne 0) { throw "failed to write $Ico" }
$Fav = Join-Path $Root "ui\public\favicon.ico"
& $py -m backend.app_icon $Png $Fav
if ($LASTEXITCODE -ne 0) { throw "failed to write $Fav" }

& $Csc /nologo /optimize /target:exe /win32icon:"$Ico" /out:"$Out" "$Cs"
if ($LASTEXITCODE -ne 0) { throw "csc failed" }
Write-Host "wrote $Out"
