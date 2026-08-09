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
# Buffer row of the active progress line (-1 = not anchored yet).
$script:ProgressAnchorTop = -1
# Optional tighter col budget after we detect a wrap.
$script:ProgressColsCap = 56

function Test-ProgressSupportsInPlace {
  <#
  .SYNOPSIS
    True when carriage-return overwrite is likely to work (real console, not redirected).
  #>
  try {
    if ([Console]::IsOutputRedirected) { return $false }
  } catch { }
  try {
    if ($Host.Name -ne "ConsoleHost") { return $false }
  } catch {
    return $false
  }
  # Cursor APIs throw "The handle is invalid" when stdout is a pipe even if
  # IsOutputRedirected is wrong — treat that as non-TTY.
  try {
    $null = [Console]::CursorTop
  } catch {
    return $false
  }
  return $true
}

function Stop-ProgressLine {
  <#
  .SYNOPSIS
    Finish an active in-place line before ordinary process output is written.
  #>
  $had = ($script:ProgressLineLen -gt 0)
  if ($had -and (Test-ProgressSupportsInPlace)) {
    Write-Host ""
  }
  $script:ProgressLineLen = 0
  $script:ProgressAnchorTop = -1
}

function Get-ProgressMaxCols {
  <#
  .SYNOPSIS
    Safe column budget for a single progress line.
    Never use full RawUI width — IDE terminals often report 120 while the panel is narrower,
    which wraps every update and scrolls the console instead of showing one bar.
  #>
  $cols = 56
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
  $cap = [Math]::Max(36, [int]$script:ProgressColsCap)
  return [Math]::Max(36, [Math]::Min($cap, $cols))
}

function Write-ProgressText {
  param(
    [Parameter(Mandatory = $true)][string]$Line,
    [switch]$Force
  )

  $line = $Line
  $cols = Get-ProgressMaxCols
  if ($line.Length -ge $cols) {
    $line = $line.Substring(0, [Math]::Max(1, $cols - 1))
  }

  $inPlace = Test-ProgressSupportsInPlace
  if (-not $inPlace) {
    $now = Get-Date
    $throttled = (-not $Force -and (($now - $script:ProgressLastNewlineAt).TotalSeconds -lt 1.5))
    if ($throttled) {
      return
    }
    Write-Host $line
    $script:ProgressLastNewlineAt = $now
    $script:ProgressLineLen = 0
    $script:ProgressAnchorTop = -1
    return
  }

  # Clear only as far as the previous render (full-width padding can wrap and
  # make a single progress update consume an entire console screen).
  $pad = [Math]::Max(0, $script:ProgressLineLen - $line.Length)
  $render = $line + (" " * $pad)
  try {
    # Always rewrite the same buffer row. SetCursorPosition(0, CursorTop) alone
    # drifts after wrap in IDE terminals (reported width > visible width), which
    # stacks a new bar + blank row on every heartbeat tick.
    if ($script:ProgressAnchorTop -lt 0) {
      $script:ProgressAnchorTop = [Console]::CursorTop
    }
    $anchor = $script:ProgressAnchorTop
    [Console]::SetCursorPosition(0, $anchor)
    # CR + erase-line + text: overwrite one physical row without emitting \n.
    # Avoid Write-Host here — some hosts turn a leading CR into a blank row.
    $esc = [char]27
    [Console]::Write(("`r{0}[2K{1}" -f $esc, $render))
    $afterTop = [Console]::CursorTop
    $wrapped = ($afterTop -gt $anchor)
    if ($wrapped) {
      # Wrapped despite budget: erase spilled rows and tighten for next ticks.
      for ($y = $anchor + 1; $y -le $afterTop; $y++) {
        try {
          [Console]::SetCursorPosition(0, $y)
          [Console]::Write(("$esc[2K"))
        } catch { break }
      }
      [Console]::SetCursorPosition(0, $anchor)
      [Console]::Write(("`r{0}[2K{1}" -f $esc, $line))
      $script:ProgressColsCap = [Math]::Max(36, [Math]::Min($script:ProgressColsCap, [Math]::Max(36, $line.Length - 2)))
      $render = $line
    }
    # Keep cursor on the anchor row so the next tick does not start below.
    try {
      [Console]::SetCursorPosition([Math]::Min([Math]::Max($render.Length, 0), [Math]::Max(0, $cols - 1)), $anchor)
    } catch {
      try { [Console]::SetCursorPosition(0, $anchor) } catch { }
    }
  } catch {
    # If cursor APIs are unavailable, use visible throttled lines instead of
    # emitting carriage returns that the host may render as empty rows.
    $now = Get-Date
    if ($Force -or (($now - $script:ProgressLastNewlineAt).TotalSeconds -ge 1.5)) {
      Write-Host $line
      $script:ProgressLastNewlineAt = $now
    }
    $script:ProgressLineLen = 0
    $script:ProgressAnchorTop = -1
    return
  }
  $script:ProgressLineLen = $line.Length
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
    [int]$Width = 20,
    [switch]$Force
  )
  $pct = [Math]::Max(0.0, [Math]::Min(100.0, $Percent))
  $filled = [int][Math]::Floor($Width * $pct / 100.0)
  if ($filled -gt $Width) { $filled = $Width }
  $bar = ("#" * $filled) + ("-" * ($Width - $filled))
  $line = "  [{0}] {1,5:N1}%  {2}" -f $bar, $pct, $Label
  if ($Detail) { $line = "$line  $Detail" }

  Write-ProgressText -Line $line -Force:$Force
}

function Write-IndeterminateProgressLine {
  <#
  .SYNOPSIS
    Show a moving bar for a process whose total amount of work is unknown.
  #>
  param(
    [string]$Label = "Working",
    [double]$ElapsedSeconds = 0,
    [int]$Frame = 0,
    [int]$Width = 20,
    [switch]$Force
  )

  $blockWidth = [Math]::Min(5, [Math]::Max(2, $Width))
  $travel = [Math]::Max(1, $Width - $blockWidth)
  $cycle = [Math]::Max(1, 2 * $travel)
  $step = $Frame % $cycle
  $pos = if ($step -le $travel) { $step } else { $cycle - $step }
  $bar = ("-" * $pos) + ("#" * $blockWidth) + ("-" * ($Width - $pos - $blockWidth))
  $line = "  [{0}]  {1}  {2:N1}s" -f $bar, $Label, $ElapsedSeconds
  Write-ProgressText -Line $line -Force:$Force
}

function Complete-ProgressLine {
  param(
    [string]$Label = "done",
    [string]$Detail = ""
  )
  Write-ProgressLine -Percent 100 -Label $Label -Detail $Detail -Force
  Stop-ProgressLine
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
    $lastStatusAt = [datetime]::MinValue
    $progressId = 42
    $usePane = Test-ProgressPaneReliable
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
      if (($now - $lastStatusAt).TotalSeconds -ge 1.0) {
        $lastStatusAt = $now
        $detail = "{0} / {1}  {2}" -f (Format-ByteSize $dstBytes), (Format-ByteSize $srcBytes), $spin[$spinIdx]
        if ($usePane) {
          Write-Progress -Id $progressId -Activity $Label -Status $detail -PercentComplete ([int][Math]::Floor($lastPct))
        } else {
          Write-Host ("    [{0}] {1,5:N1}%  {2}" -f $Label, $lastPct, $detail)
        }
      }
      $spinIdx = ($spinIdx + 1) % $spin.Length
      Start-Sleep -Milliseconds 350
    }
    $p.WaitForExit()
    $code = [int]$p.ExitCode
    $doneDetail = ("{0} copied" -f (Format-ByteSize (Get-FolderByteSize -Path $Dest)))
    if ($usePane) {
      Write-Progress -Id $progressId -Activity $Label -Status $doneDetail -PercentComplete 100 -Completed
    }
    Write-Host ("    {0} - {1}" -f $Label, $doneDetail)

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
    Write-Progress -Id 42 -Activity "done" -Completed -ErrorAction SilentlyContinue
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

function Test-ProgressPaneReliable {
  <#
  .SYNOPSIS
    True when Write-Progress can use the host progress pane without scrolling blanks.
  #>
  try {
    if ([Console]::IsOutputRedirected) { return $false }
  } catch { return $false }
  # VS Code / Cursor ConPTY often cannot render the progress pane and instead
  # prints a blank line for every Write-Progress call.
  if ($env:TERM_PROGRAM -eq "vscode" -or $env:TERM_PROGRAM -eq "cursor") { return $false }
  if ($env:VSCODE_PID -or $env:VSCODE_INJECTION -or $env:CURSOR_TRACE_ID) { return $false }
  try {
    $null = [Console]::CursorTop
  } catch {
    return $false
  }
  return $true
}

function Invoke-ProcessWithHeartbeat {
  <#
  .SYNOPSIS
    Run an external command with live output + non-scrolling progress.
    Uses Write-Progress when the host progress pane works; otherwise throttled status lines.
  #>
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [string]$Activity = "Working",
    [int]$HeartbeatSeconds = 15
  )

  $usePane = Test-ProgressPaneReliable
  Write-LongStepHint "$Activity"
  if ($usePane) {
    Write-LongStepHint "Live output below. Host progress bar updates while this step runs."
  } else {
    Write-LongStepHint "Live output below. Status lines update about once per second (no stacking bars)."
  }

  $argString = Format-ProcessArgumentList -Args $ArgumentList
  $stdoutLog = [System.IO.Path]::GetTempFileName()
  $stderrLog = [System.IO.Path]::GetTempFileName()
  $lastBeat = Get-Date
  $lastStatusAt = [datetime]::MinValue
  $startedAt = Get-Date
  $frame = 0
  $progressId = 41
  $statusWrites = 0
  # HeartbeatSeconds kept for caller compatibility.
  $null = $HeartbeatSeconds

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
        # Captured CLI spinners contain cursor-control ANSI and many carriage
        # returns. Replaying those bytes can move the console cursor down by a
        # screenful. Strip terminal controls and retain only the latest visible
        # carriage-return segment from each logical line.
        $chunk = $chunk -replace "$([char]27)\][^$([char]7)]*($([char]7)|$([char]27)\\)", ""
        $chunk = $chunk -replace "$([char]27)\[[0-?]*[ -/]*[@-~]", ""
        $safeLines = [System.Collections.Generic.List[string]]::new()
        foreach ($logicalLine in @($chunk -split "`n")) {
          $segments = @($logicalLine -split "`r")
          $visible = ""
          for ($i = $segments.Count - 1; $i -ge 0; $i--) {
            $candidate = ($segments[$i] -replace '[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '').TrimEnd()
            if ($candidate.Trim() -ne "") {
              $visible = $candidate
              break
            }
          }
          if ($visible) { [void]$safeLines.Add($visible) }
        }
        return @($safeLines.ToArray())
      } finally {
        $fs.Close()
      }
    }

    if ($usePane) {
      Write-Progress -Id $progressId -Activity $Activity -Status "starting..." -PercentComplete 0
    }
    while (-not $p.HasExited) {
      $newOut = Read-NewLines -LogPath $stdoutLog -Pos ([ref]$outPos)
      $newErr = Read-NewLines -LogPath $stderrLog -Pos ([ref]$errPos)
      foreach ($line in @($newOut + $newErr)) {
        if ($line -match '^(WARNING:|WARN:)') { continue }
        Write-Host $line
        $lastBeat = Get-Date
      }
      $elapsed = ((Get-Date) - $startedAt).TotalSeconds
      $now = Get-Date
      # Throttle UI updates: Write-Progress every 200ms floods ConPTY with blank rows.
      if (($now - $lastStatusAt).TotalSeconds -ge 1.0) {
        $lastStatusAt = $now
        $statusWrites++
        $status = ("elapsed {0:N1}s" -f $elapsed)
        if ($usePane) {
          $pct = [Math]::Min(99, [int](3 * $statusWrites) % 99)
          Write-Progress -Id $progressId -Activity $Activity -Status $status -PercentComplete $pct
        } else {
          Write-Host ("    [{0}] {1}" -f $Activity, $status)
        }
      }
      $frame++
      Start-Sleep -Milliseconds 200
    }
    $p.WaitForExit()

    $remaining = @(
      @(Read-NewLines -LogPath $stdoutLog -Pos ([ref]$outPos)) +
      @(Read-NewLines -LogPath $stderrLog -Pos ([ref]$errPos))
    )
    foreach ($line in $remaining) {
      if ($line -match '^(WARNING:|WARN:)') { continue }
      Write-Host $line
    }

    $elapsed = ((Get-Date) - $startedAt).TotalSeconds
    $doneStatus = ("finished in {0:N1}s" -f $elapsed)
    if ($usePane) {
      Write-Progress -Id $progressId -Activity $Activity -Status $doneStatus -PercentComplete 100 -Completed
    }
    Write-Host ("    {0} - {1}" -f $Activity, $doneStatus)
    return [int]$p.ExitCode
  } finally {
    if ($usePane) {
      Write-Progress -Id $progressId -Activity $Activity -Completed -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue
  }
}

function Invoke-NativeWithHeartbeat {
  <#
  .SYNOPSIS
    Run a native exe with the console inherited so TTY progress bars (pip/npm) work.
  .NOTES
    Must NOT leave native stdout on the PowerShell success stream — callers cast
    the result to [int]. Use Start-Process -NoNewWindow -Wait (no redirect).
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
    $argString = Format-ProcessArgumentList -Args $ArgumentList
    $p = Start-Process -FilePath $FilePath `
      -ArgumentList $argString `
      -NoNewWindow `
      -Wait `
      -PassThru
    if ($null -eq $p) { return 1 }
    return [int]$p.ExitCode
  } finally {
    $ErrorActionPreference = $prev
  }
}
