# Bake app-icon.png (the VTM Spark logo) onto run.exe with a transparent ICO. A .bat cannot carry an Explorer icon.
# winexe: no console window. The stub opens start-menu.ps1 in a console only when it must.
$ErrorActionPreference = "Stop"
$Pack = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $Pack "..\..")).Path
$Png = Join-Path $Root "ui\public\app-icon.png"
$Ico = Join-Path $Pack "run.ico"
$Cs = Join-Path $Pack "run-stub.cs"
$Out = Join-Path $Root "run.exe"
$Csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $Csc)) {
  $Csc = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
}
if (-not (Test-Path -LiteralPath $Csc)) {
  throw "csc.exe not found. Need .NET Framework 4 to stamp the logo on run.exe"
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

& $Csc /nologo /optimize /target:winexe /reference:System.Management.dll /win32icon:"$Ico" /out:"$Out" "$Cs"
if ($LASTEXITCODE -ne 0) { throw "csc failed" }
Write-Host "wrote $Out"

# The virtual camera's admin prompt shows this exe's name and logo, not cmd's.
$CamCs = Join-Path $Pack "cam-setup.cs"
$CamOut = Join-Path $Root "vendor\tools\vtm_spark_cam\VTM Spark Camera Setup.exe"
& $Csc /nologo /optimize /target:winexe /platform:anycpu /win32icon:"$Ico" /out:"$CamOut" "$CamCs"
if ($LASTEXITCODE -ne 0) { throw "csc failed on cam-setup.cs" }
Write-Host "wrote $CamOut"
