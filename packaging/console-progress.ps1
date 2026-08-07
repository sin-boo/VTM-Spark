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

function Write-ProgressLine {
  <#
  .SYNOPSIS
    Overwrite the current console line with an ASCII progress bar.
  #>
  param(
    [ValidateRange(0, 100)][double]$Percent,
    [string]$Label = "",
    [string]$Detail = "",
    [int]$Width = 28
  )
  $pct = [Math]::Max(0.0, [Math]::Min(100.0, $Percent))
  $filled = [int][Math]::Floor($Width * $pct / 100.0)
  if ($filled -gt $Width) { $filled = $Width }
  $bar = ("#" * $filled) + ("-" * ($Width - $filled))
  $line = "    [{0}] {1,5:N1}%  {2}" -f $bar, $pct, $Label
  if ($Detail) { $line = "$line  $Detail" }
  $cols = 100
  try {
    if ($Host.UI.RawUI -and $Host.UI.RawUI.WindowSize.Width -gt 20) {
      $cols = [Math]::Max(40, $Host.UI.RawUI.WindowSize.Width - 1)
    }
  } catch { }
  if ($line.Length -ge $cols) {
    $line = $line.Substring(0, $cols - 1)
  } else {
    $line = $line.PadRight($cols)
  }
  Write-Host "`r$line" -NoNewline
}

function Complete-ProgressLine {
  param(
    [string]$Label = "done",
    [string]$Detail = ""
  )
  Write-ProgressLine -Percent 100 -Label $Label -Detail $Detail
  Write-Host ""
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

  $stdoutLog = [System.IO.Path]::GetTempFileName()
  $stderrLog = [System.IO.Path]::GetTempFileName()
  try {
    $p = Start-Process -FilePath "robocopy.exe" `
      -ArgumentList $argList `
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
  Write-LongStepHint "Live output below. If quiet for a bit, a heartbeat keeps the line alive."

  $stdoutLog = [System.IO.Path]::GetTempFileName()
  $stderrLog = [System.IO.Path]::GetTempFileName()
  $lastBeat = Get-Date
  $spin = @("|", "/", "-", "\")
  $spinIdx = 0

  try {
    $p = Start-Process -FilePath $FilePath `
      -ArgumentList $ArgumentList `
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
        return @($chunk -split "`r?`n" | Where-Object { $_ -ne "" })
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

    # Drain remaining output
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
