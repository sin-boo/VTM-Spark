# Shared helpers: detect a dead venv home (missing pythonw / encodings) and retarget.

function Test-VtmPythonExe {
  param([string]$PythonExe)
  if (-not $PythonExe) { return $false }
  if (-not (Test-Path -LiteralPath $PythonExe)) { return $false }
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $PythonExe -c "import encodings, sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 1>$null 2>$null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  } finally {
    $ErrorActionPreference = $prev
  }
}

function Get-VtmVenvCfgPath {
  param([string]$VenvDir)
  return (Join-Path $VenvDir "pyvenv.cfg")
}

function Get-VtmVenvCfg {
  param([string]$VenvDir)
  $cfg = Get-VtmVenvCfgPath $VenvDir
  $info = [ordered]@{
    Path    = $cfg
    Home    = ""
    Version = ""
  }
  if (-not (Test-Path -LiteralPath $cfg)) { return [pscustomobject]$info }
  foreach ($line in Get-Content -LiteralPath $cfg) {
    if ($line -match '^\s*home\s*=\s*(.+)$') {
      $info.Home = $Matches[1].Trim()
    } elseif ($line -match '^\s*version\s*=\s*(.+)$') {
      $info.Version = $Matches[1].Trim()
    }
  }
  return [pscustomobject]$info
}

function Get-VtmPythonMinor {
  param([string]$Version)
  if ($Version -match '(\d+)\.(\d+)') {
    return "$($Matches[1]).$($Matches[2])"
  }
  return ""
}

function Get-VtmHostPythonCandidates {
  $local = [Environment]::GetFolderPath("LocalApplicationData")
  $pf = ${env:ProgramFiles}
  $home = $env:USERPROFILE
  return @(
    (Join-Path $local "Programs\Python\Python313\python.exe"),
    (Join-Path $local "Programs\Python\Python312\python.exe"),
    (Join-Path $local "Programs\Python\Python311\python.exe"),
    (Join-Path $pf "Python313\python.exe"),
    (Join-Path $pf "Python312\python.exe"),
    (Join-Path $pf "Python311\python.exe"),
    (Join-Path $pf "Python310\python.exe"),
    (Join-Path $home "miniconda3\python.exe"),
    (Join-Path $home "anaconda3\python.exe")
  )
}

function Resolve-VtmMatchingHostPython {
  param([string]$WantMinor)
  foreach ($c in Get-VtmHostPythonCandidates) {
    if (-not (Test-VtmPythonExe $c)) { continue }
    if (-not $WantMinor) { return $c }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
      $got = & $c -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
      if ($LASTEXITCODE -eq 0 -and $got -and $got.Trim() -eq $WantMinor) {
        return $c
      }
    } catch {
    } finally {
      $ErrorActionPreference = $prev
    }
  }
  return $null
}

function Repair-VtmVenvHome {
  param(
    [string]$VenvDir,
    [string]$PythonExe
  )
  if (Test-VtmPythonExe $PythonExe) { return $true }
  $cfg = Get-VtmVenvCfg $VenvDir
  if (-not (Test-Path -LiteralPath $cfg.Path)) { return $false }
  $wantMinor = Get-VtmPythonMinor $cfg.Version
  $hostPy = Resolve-VtmMatchingHostPython -WantMinor $wantMinor
  if (-not $hostPy) { return $false }
  $hostHome = Split-Path -Parent $hostPy
  $hostVer = ""
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $hostVer = (& $hostPy -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2>$null)
    if ($LASTEXITCODE -ne 0) { $hostVer = $wantMinor }
  } catch {
    $hostVer = $wantMinor
  } finally {
    $ErrorActionPreference = $prev
  }
  $lines = @(
    "home = $hostHome"
    "include-system-site-packages = false"
    "version = $($hostVer.Trim())"
    "executable = $hostPy"
    "command = $hostPy -m venv $VenvDir"
  )
  Set-Content -LiteralPath $cfg.Path -Value $lines -Encoding ascii
  return (Test-VtmPythonExe $PythonExe)
}

function Resolve-VtmDeskPython {
  param(
    [string]$VenvDir,
    [string]$PythonExe
  )
  $cfg = Get-VtmVenvCfg $VenvDir
  $pyw = Join-Path $VenvDir "Scripts\pythonw.exe"
  if ($cfg.Home -and (Test-Path -LiteralPath (Join-Path $cfg.Home "pythonw.exe")) -and (Test-Path -LiteralPath $pyw)) {
    return $pyw
  }
  return $PythonExe
}
