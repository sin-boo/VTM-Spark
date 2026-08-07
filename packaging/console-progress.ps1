# Console progress helpers for Smart Build / packaging (ASCII-safe bars).
# Dot-source from build.ps1 / start-menu.ps1.

function Get-FolderByteSize {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return [int64]0 }
  $sum = [int64]0
  Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue |
    ForEach-Object { $sum += $_.Length }
  return $sum
}

function Format-ByteSize {
  param([int64]$Bytes)
  if ($Bytes -ge 1GB) { return ("{0:N2} GB" -f ($Bytes / 1GB)) }
  if ($Bytes -ge 1MB) { return ("{0:N1} MB" -f ($Bytes / 1MB)) }
  if ($Bytes -ge 1KB) { return ("{0:N0} KB" -f ($Bytes / 1KB)) }
  return "$Bytes B"
}

# Last in-place progress line length (for clearing leftovers without full-width pad).
$script:ProgressLineLen = 0
$script:ProgressLastNewlineAt = [datetime]::MinValue

function Test-ProgressSupportsInPlace {
  <#
  .SYNOPSIS
    True when carriage-return overwrite is likely to work (real console, not redirected).
  #>
  try {
    if ([Console]::IsOutputRedirected) { return $false }
  } catch { }
  # Cursor/VS Code / some hosts report RawUI width but still redirect agent shells.
  # Prefer in-place when we have an interactive ConsoleHost and no redirect flag.
  try {
    if ($Host.Name -ne "ConsoleHost") { return $false }
  } catch {
    return $false
  }
  return $true
}

function Get-ProgressMaxCols {
  <#
  .SYNOPSIS
    Safe column budget for a single progress line.
    Never use full RawUI width — IDE terminals often report 120 while the panel is narrower,
    which wraps every update and scrolls the console instead of showing one bar.
  #>
  $cols = 72
  try {
    $w = [Console]::WindowWidth
    if ($w -gt 20) { $cols = $w - 1 }
  } catch {
    try {
      if ($Host.UI.RawUI -and $Host.UI.RawUI.WindowSize.Width -gt 20) {
        $cols = $Host.UI.RawUI.WindowSize.Width - 1
      }
    } catch { }
  }
  # Cap hard: keep the bar on one visual row even when RawUI lies about width.
  return [Math]::Max(40, [Math]::Min(76, $cols))
}

function Write-ProgressLine {
  <#
  .SYNOPSIS
    Overwrite the current console line with an ASCII progress bar.
    Falls back to throttled newlines when stdout is redirected / non-TTY.
  #>
  param(
    [ValidateRange(0, 100)][double]$Percent,
    [string]$Label = "",
    [string]$Detail = "",
    [int]$Width = 28,
    [switch]$Force
  )
  $pct = [Math]::Max(0.0, [Math]::Min(100.0, $Percent))
  $filled = [int][Math]::Floor($Width * $pct / 100.0)
  if ($filled -gt $Width) { $filled = $Width }
  $bar = ("#" * $filled) + ("-" * ($Width - $filled))
  $line = "    [{0}] {1,5:N1}%  {2}" -f $bar, $pct, $Label
  if ($Detail) { $line = "$line  $Detail" }

  $cols = Get-ProgressMaxCols
  if ($line.Length -ge $cols) {
    $line = $line.Substring(0, [Math]::Max(1, $cols - 1))
  }

  $inPlace = Test-ProgressSupportsInPlace
  if (-not $inPlace) {
    # Redirected / captured output: \r does not overwrite — throttle so we don't spam.
    $now = Get-Date
    if (-not $Force -and (($now - $script:ProgressLastNewlineAt).TotalSeconds -lt 1.5) -and $pct -lt 99.5) {
      return
    }
    Write-Host $line
    $script:ProgressLastNewlineAt = $now
    $script:ProgressLineLen = 0
    return
  }

  # Clear only as far as the previous render (avoids full-width pad → wrap → scroll).
  $pad = [Math]::Max(0, $script:ProgressLineLen - $line.Length)
  $render = $line + (" " * $pad)
  try {
    [Console]::Write("`r$render")
  } catch {
    Write-Host "`r$render" -NoNewline
  }
  $script:ProgressLineLen = $line.Length
}

function Complete-ProgressLine {
  param(
    [string]$Label = "done",
    [string]$Detail = ""
  )
  Write-ProgressLine -Percent 100 -Label $Label -Detail $Detail -Force
  if (Test-ProgressSupportsInPlace) {
    Write-Host ""
  }
  $script:ProgressLineLen = 0
}

function Write-LongStepHint {
  param([string]$Message)
  Write-Host "    $Message"
}

function Invoke-RobocopyWithProgress {
  <#
  .SYNOPSIS
    Run robocopy while showing a size-based progress bar (avoids looking frozen).
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
  Write-LongStepHint "Measuring source size (one-time)..."
  $srcBytes = Get-FolderByteSize -Path $Source
  Write-LongStepHint ("Source size ~ {0} - progress updates while files copy." -f (Format-ByteSize $srcBytes))

  $argList = @(
    $Source, $Dest
  ) + $ExtraArgs + @(
    "/NFL", "/NDL", "/NJH", "/NJS", "/nc", "/ns", "/np"
  )
  $argString = Format-ProcessArgumentList -Args $argList

  $stdoutLog = [System.IO.Path]::GetTempFileName()
  $stderrLog = [System.IO.Path]::GetTempFileName()
  try {
    $p = Start-Process -FilePath "robocopy.exe" `
      -ArgumentList $argString `
      -NoNewWindow -PassThru `
      -RedirectStandardOutput $stdoutLog `
      -RedirectStandardError $stderrLog

    $spin = @("|", "/", "-", "\")
    $spinIdx = 0
    $lastPct = 0.0
    $dstBytes = [int64]0
    $lastSizeCheck = Get-Date
    # Full-tree size scans are expensive on multi-GB venvs — sample every 2s, spin every tick.
    while (-not $p.HasExited) {
      $now = Get-Date
      if ((($now - $lastSizeCheck).TotalSeconds -ge 2.0) -or ($dstBytes -eq 0)) {
        $dstBytes = Get-FolderByteSize -Path $Dest
        $lastSizeCheck = $now
        if ($srcBytes -gt 0) {
          $pct = [Math]::Min(99.0, (100.0 * [double]$dstBytes / [double]$srcBytes))
          if ($pct -gt $lastPct) { $lastPct = $pct }
        }
      }
      $detail = "{0} / {1}  {2}" -f (Format-ByteSize $dstBytes), (Format-ByteSize $srcBytes), $spin[$spinIdx]
      Write-ProgressLine -Percent $lastPct -Label $Label -Detail $detail
      $spinIdx = ($spinIdx + 1) % $spin.Length
      Start-Sleep -Milliseconds 350
    }
    $p.WaitForExit()
    $code = [int]$p.ExitCode
    Complete-ProgressLine -Label $Label -Detail ("{0} copied" -f (Format-ByteSize (Get-FolderByteSize -Path $Dest)))

    # Robocopy: 0-7 = success with optional extras; >= 8 = failure.
    if ($code -ge 8) {
      $errTail = ""
      if (Test-Path -LiteralPath $stderrLog) {
        $errTail = (Get-Content -LiteralPath $stderrLog -ErrorAction SilentlyContinue | Select-Object -Last 5) -join " | "
      }
      throw "robocopy failed (exit $code) for $Label. $errTail"
    }
    return $code
  } finally {
    Remove-Item -LiteralPath $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue
  }
}

function Format-ProcessArgumentList {
  <#
  .SYNOPSIS
    Build a single Start-Process command-line string (array ArgumentList is unreliable).
  #>
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

function Invoke-ProcessWithHeartbeat {
  <#
  .SYNOPSIS
    Run an external command, streaming stdout/stderr live, with a heartbeat if quiet.
  #>
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [string]$Activity = "Working",
    [int]$HeartbeatSeconds = 15
  )

  Write-LongStepHint "$Activity"
  Write-LongStepHint "Live output below. A heartbeat appears if this step goes quiet."

  $argString = Format-ProcessArgumentList -Args $ArgumentList
  $stdoutLog = [System.IO.Path]::GetTempFileName()
  $stderrLog = [System.IO.Path]::GetTempFileName()
  $lastBeat = Get-Date
  $spin = @("|", "/", "-", "\")
  $spinIdx = 0

  try {
    $p = Start-Process -FilePath $FilePath `
      -ArgumentList $argString `
      -NoNewWindow -PassThru `
      -RedirectStandardOutput $stdoutLog `
      -RedirectStandardError $stderrLog

    $outPos = 0L
    $errPos = 0L

    function Read-NewLines {
      param([string]$LogPath, [ref]$Pos)
      if (-not (Test-Path -LiteralPath $LogPath)) { return @() }
      $fs = [System.IO.File]::Open($LogPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
      try {
        if ($fs.Length -lt $Pos.Value) { $Pos.Value = 0 }
        $fs.Seek($Pos.Value, [System.IO.SeekOrigin]::Begin) | Out-Null
        $reader = New-Object System.IO.StreamReader($fs)
        $chunk = $reader.ReadToEnd()
        $Pos.Value = $fs.Position
        if (-not $chunk) { return @() }
        # pip/npm progress uses \r; keep the latest segment of each update only.
        $chunk = $chunk -replace "`r`n", "`n" -replace "`r", "`n"
        return @(
          $chunk -split "`n" |
            ForEach-Object { $_.TrimEnd() } |
            Where-Object { $_.Trim() -ne "" }
        )
      } finally {
        $fs.Close()
      }
    }

    while (-not $p.HasExited) {
      $newOut = Read-NewLines -LogPath $stdoutLog -Pos ([ref]$outPos)
      $newErr = Read-NewLines -LogPath $stderrLog -Pos ([ref]$errPos)
      $got = $false
      foreach ($line in @($newOut + $newErr)) {
        if ($line -match '^(WARNING:|WARN:)') { continue }
        Write-Host $line
        $got = $true
        $lastBeat = Get-Date
      }
      if (-not $got) {
        $idle = ((Get-Date) - $lastBeat).TotalSeconds
        if ($idle -ge $HeartbeatSeconds) {
          Write-Host ("    [{0}] still working on: {1} ({2:N0}s quiet)..." -f $spin[$spinIdx], $Activity, $idle)
          $lastBeat = Get-Date
          $spinIdx = ($spinIdx + 1) % $spin.Length
        }
      }
      Start-Sleep -Milliseconds 200
    }
    $p.WaitForExit()

    foreach ($line in @(Read-NewLines -LogPath $stdoutLog -Pos ([ref]$outPos))) {
      if ($line -match '^(WARNING:|WARN:)') { continue }
      Write-Host $line
    }
    foreach ($line in @(Read-NewLines -LogPath $stderrLog -Pos ([ref]$errPos))) {
      if ($line -match '^(WARNING:|WARN:)') { continue }
      Write-Host $line
    }

    return [int]$p.ExitCode
  } finally {
    Remove-Item -LiteralPath $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue
  }
}

function Invoke-NativeWithHeartbeat {
  <#
  .SYNOPSIS
    Run via the call operator so TTY progress bars (pip/npm) work.
    Avoid Start-Process redirection — that turns progress into blank scrolling lines.
  #>
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [string]$Activity = "Working",
    [int]$HeartbeatSeconds = 12
  )

  Write-LongStepHint "$Activity"
  Write-LongStepHint "Progress should update live below (large downloads can take several minutes)."
  # HeartbeatSeconds reserved for API compatibility with callers.
  $null = $HeartbeatSeconds

  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $FilePath @ArgumentList
    if ($null -eq $LASTEXITCODE) { return 0 }
    return [int]$LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prev
  }
}
