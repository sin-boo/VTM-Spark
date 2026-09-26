# Kill leftover VTM Noble / Real Stream processes so close/build doesn't leave
# multi-GB Python orphans around (was freezing 96GB hosts at ~90% RAM).
param(
  [switch]$Quiet,
  [switch]$Fast
)

$ErrorActionPreference = "Continue"
$names = @("VTMNoble", "RealStream")
foreach ($n in $names) {
  Get-Process -Name $n -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $Quiet) { Write-Host "    killing $($_.ProcessName) pid=$($_.Id)" }
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    cmd /c "taskkill /F /T /PID $($_.Id) >nul 2>&1" | Out-Null
  }
  cmd /c "taskkill /F /IM $n.exe >nul 2>&1" | Out-Null
}

if (-not $Fast) {
  # Child runtimes started as: ...\python.exe -m backend
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -match '^(python|pythonw)\.exe$' -and
      $_.CommandLine -and
      (
        $_.CommandLine -match '(?i)-m\s+backend(\s|$)' -and
        (
          $_.CommandLine -match '(?i)[\\/](vtm[ -]?noble|real_stream|VTMNoble|RealStream)[\\/]' -or
          $_.CommandLine -match '(?i)[\\/]dist[\\/](VTMNoble|RealStream)[\\/]' -or
          $_.CommandLine -match '(?i)VTM_NOBLE|REAL_STREAM'
        )
      )
    } |
    ForEach-Object {
      if (-not $Quiet) { Write-Host "    killing python pid=$($_.ProcessId) (backend)" }
      cmd /c "taskkill /F /T /PID $($_.ProcessId) >nul 2>&1" | Out-Null
    }

  # Broader sweep: any python whose command line is clearly our packaged runtime
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -match '^(python|pythonw)\.exe$' -and
      $_.CommandLine -and
      $_.CommandLine -match '(?i)[\\/]dist[\\/]VTMNoble[\\/]runtime[\\/].*python'
    } |
    ForEach-Object {
      if (-not $Quiet) { Write-Host "    killing python pid=$($_.ProcessId) (VTMNoble runtime)" }
      cmd /c "taskkill /F /T /PID $($_.ProcessId) >nul 2>&1" | Out-Null
    }
}

if ($Fast) {
  Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe' OR Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
      $_.CommandLine -and (
        $_.CommandLine -match '(?i)-m\s+backend(\s|$)' -or
        $_.CommandLine -match '(?i)_mock_boot\.py'
      )
    } |
    ForEach-Object {
      cmd /c "taskkill /F /T /PID $($_.ProcessId) >nul 2>&1" | Out-Null
    }

  # Vite's mock API sits on 8765 and has no "-m backend" in its command line.
  # Leave it and the desk hands off to JSON ``{}`` (Edge "Pretty-print").
  try {
    $deskPort = Get-NetTCPConnection -State Listen -LocalPort 8765 -ErrorAction SilentlyContinue
    foreach ($c in @($deskPort)) {
      if ($null -eq $c) { continue }
      $owningPid = [int]$c.OwningProcess
      if ($owningPid -le 4) { continue }
      $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$owningPid" -ErrorAction SilentlyContinue
      if ($null -eq $proc) { continue }
      if ($proc.Name -match '^(python|pythonw)') {
        if (-not $Quiet) { Write-Host "    killing pid=$owningPid listening on port 8765 ($($proc.Name))" }
        cmd /c "taskkill /F /T /PID $owningPid >nul 2>&1" | Out-Null
      }
    }
  } catch {}
}

if (-not $Fast) {
  # Leftover Python on 8780 is a stale Track Lab (uvicorn reload ghosts).
  # A live lab is started again by the app from .venv-build.
  try {
    $conns = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
      Where-Object { $_.LocalPort -ge 8765 -and $_.LocalPort -le 8828 }
    foreach ($c in $conns) {
      $owningPid = [int]$c.OwningProcess
      if ($owningPid -le 4) { continue }
      $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$owningPid" -ErrorAction SilentlyContinue
      if ($null -eq $proc) { continue }
      if ($proc.Name -match '^(python|pythonw|VTMNoble|RealStream)') {
        if (-not $Quiet) {
          Write-Host "    killing pid=$owningPid listening on port $($c.LocalPort) ($($proc.Name))"
        }
        cmd /c "taskkill /F /T /PID $owningPid >nul 2>&1" | Out-Null
      }
    }
  } catch {
    # Get-NetTCPConnection may be unavailable; ignore.
  }

  Start-Sleep -Milliseconds 500
}

if (-not $Quiet) { Write-Host "    orphan sweep done" }
# taskkill returns 128 when the process is already gone. Never leak that to Start.
exit 0
