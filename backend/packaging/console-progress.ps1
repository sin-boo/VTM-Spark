# Thin process helpers for install.bat / packaging.
# No custom ASCII bars — let uv/npm/vite show their own progress when possible.
# Dot-source from build.ps1 / start-menu.ps1.

function Write-LongStepHint {
  param([string]$Message)
  Write-Host "    $Message"
}

function Format-ProcessArgumentList {
  param([Parameter(Mandatory = $true)][string[]]$Args)
  $parts = foreach ($a in @($Args)) {
    if ($null -eq $a) { continue }
    $s = "$a"
    if ($s -match '[\s"]') {
      '"' + ($s -replace '"', '\"') + '"'
    } else {
      $s
    }
  }
  return (($parts) -join " ")
}

function Invoke-NativeWithHeartbeat {
  <#
  .SYNOPSIS
    Run a native exe on the real console so uv/npm keep their own progress UI.
    Returns only the exit code (stdout must not enter the PowerShell pipeline).
  #>
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [string]$Activity = "Working",
    [int]$HeartbeatSeconds = 12
  )
  $null = $HeartbeatSeconds
  Write-LongStepHint $Activity

  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $argString = Format-ProcessArgumentList -Args $ArgumentList
    $p = Start-Process -FilePath $FilePath `
      -ArgumentList $argString `
      -NoNewWindow -Wait -PassThru
    if ($null -eq $p) { return 1 }
    return [int]$p.ExitCode
  } finally {
    $ErrorActionPreference = $prev
  }
}

function Invoke-ProcessWithHeartbeat {
  <#
  .SYNOPSIS
    Same as Invoke-NativeWithHeartbeat (name kept for build.ps1 callers).
  #>
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [string]$Activity = "Working",
    [int]$HeartbeatSeconds = 15
  )
  return [int](Invoke-NativeWithHeartbeat `
    -FilePath $FilePath `
    -ArgumentList $ArgumentList `
    -Activity $Activity `
    -HeartbeatSeconds $HeartbeatSeconds)
}

function Invoke-RobocopyWithProgress {
  <#
  .SYNOPSIS
    Quiet robocopy with a start/done line (no size polling or fake bars).
  .NOTES
    Robocopy exit codes 0-7 are success.
  #>
  param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Dest,
    [Parameter(Mandatory = $true)][string]$Label,
    [string[]]$ExtraArgs = @()
  )

  if (-not (Test-Path -LiteralPath $Source)) {
    throw "Robocopy source missing: $Source"
  }
  New-Item -ItemType Directory -Force -Path $Dest | Out-Null

  Write-Host "==> $Label"
  $argList = @($Source, $Dest) + $ExtraArgs + @(
    "/NFL", "/NDL", "/NJH", "/NJS", "/nc", "/ns", "/np"
  )
  $argString = Format-ProcessArgumentList -Args $argList
  $p = Start-Process -FilePath "robocopy.exe" `
    -ArgumentList $argString `
    -NoNewWindow -Wait -PassThru
  $code = if ($null -eq $p) { 16 } else { [int]$p.ExitCode }
  if ($code -ge 8) {
    throw "robocopy failed (exit $code) for $Label"
  }
  Write-Host "    done"
  return $code
}
