# Stops VTM Spark and everything running from this folder (Track Lab, Node, the
# venv's Python and the interpreters it starts), so the folder's files can be
# replaced or deleted (update.ps1, uninstall.ps1).
# -List only prints what is running: exit 1 = something is, 0 = nothing.
param([switch]$List)

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$rootPrefix = $Root.TrimEnd('\') + '\'

if (-not $List) {
  & (Join-Path $PSScriptRoot "kill-orphans.ps1") -Quiet
}

$all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
$byId = @{}
foreach ($p in $all) { $byId[[int]$p.ProcessId] = $p }

# Never this script's own process or what started it.
$keep = @{}
$id = [int]$PID
while ($byId.ContainsKey($id) -and -not $keep.ContainsKey($id)) {
  $keep[$id] = $true
  $id = [int]$byId[$id].ParentProcessId
}

$hit = @{}
foreach ($p in $all) {
  $id = [int]$p.ProcessId
  if ($keep.ContainsKey($id)) { continue }
  $fromRoot = $p.ExecutablePath -and $p.ExecutablePath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)
  $namesRoot = $p.Name -match '^(python|pythonw|node)\.exe$' -and $p.CommandLine -and
    $p.CommandLine.IndexOf($rootPrefix, [StringComparison]::OrdinalIgnoreCase) -ge 0
  if ($fromRoot -or $namesRoot) { $hit[$id] = $p }
}
# Their children too: the venv's pythonw.exe is a launcher whose interpreter
# lives outside this folder. A "child" older than its parent only inherited a
# reused process id.
do {
  $added = $false
  foreach ($p in $all) {
    $id = [int]$p.ProcessId
    $parent = [int]$p.ParentProcessId
    if ($hit.ContainsKey($id) -or $keep.ContainsKey($id) -or -not $hit.ContainsKey($parent)) { continue }
    if ($p.CreationDate -lt $hit[$parent].CreationDate) { continue }
    $hit[$id] = $p
    $added = $true
  }
} while ($added)

foreach ($p in $hit.Values) {
  if ($List) {
    Write-Host "    running: $($p.Name) pid=$($p.ProcessId)"
  } else {
    Write-Host "    stopping $($p.Name) pid=$($p.ProcessId)"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
  }
}
if ($List) {
  if ($hit.Count -gt 0) { exit 1 }
  exit 0
}
Start-Sleep -Milliseconds 500
exit 0
