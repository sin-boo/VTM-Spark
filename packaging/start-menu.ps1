# VTM Noble start menu - Smart Build + Start. Kill orphans is automatic.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPy = Join-Path $Root ".venv-build\Scripts\python.exe"
$UiIndex = Join-Path $Root "ui\dist\index.html"
$KillScript = Join-Path $PSScriptRoot "kill-orphans.ps1"
$BuildScript = Join-Path $PSScriptRoot "build.ps1"
$Esc = [char]27

. (Join-Path $PSScriptRoot "console-progress.ps1")

function Enable-PrettyConsole {
  try {
    $Host.UI.RawUI.WindowTitle = "VTM Noble"
  } catch {}

  try {
    if (-not ("Win32.ConsoleMode" -as [type])) {
      Add-Type -Namespace Win32 -Name ConsoleMode -MemberDefinition @"
[DllImport("kernel32.dll", SetLastError = true)]
public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
"@
    }
    $hout = [Win32.ConsoleMode]::GetStdHandle(-11)
    $mode = 0
    if ([Win32.ConsoleMode]::GetConsoleMode($hout, [ref]$mode)) {
      [void][Win32.ConsoleMode]::SetConsoleMode($hout, ($mode -bor 0x0004))
    }
  } catch {}

  try {
    if (-not ("Win32.ConsoleFont" -as [type])) {
      Add-Type -Namespace Win32 -Name ConsoleFont -MemberDefinition @"
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
public struct CONSOLE_FONT_INFOEX {
  public uint cbSize;
  public uint nFont;
  public short dwFontSizeX;
  public short dwFontSizeY;
  public uint FontFamily;
  public uint FontWeight;
  [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
  public string FaceName;
}
[DllImport("kernel32.dll", SetLastError = true)]
public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool SetCurrentConsoleFontEx(IntPtr hConsoleOutput, bool bMaximumWindow, ref CONSOLE_FONT_INFOEX lpConsoleCurrentFontEx);
"@
    }

    $faces = @("Cascadia Mono", "Cascadia Code", "JetBrains Mono", "Consolas")
    $hout = [Win32.ConsoleFont]::GetStdHandle(-11)
    foreach ($face in $faces) {
      $info = New-Object Win32.ConsoleFont+CONSOLE_FONT_INFOEX
      $info.cbSize = [uint32]84
      $info.nFont = 0
      $info.dwFontSizeX = 0
      $info.dwFontSizeY = 18
      $info.FontFamily = 54
      $info.FontWeight = 400
      $info.FaceName = $face
      if ([Win32.ConsoleFont]::SetCurrentConsoleFontEx($hout, $false, [ref]$info)) {
        break
      }
    }
  } catch {}

  try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $script:OutputEncoding = [Console]::OutputEncoding
  } catch {}
}

function Write-Ansi {
  param(
    [string]$Text,
    [string]$Color = "",
    [switch]$NoNewline
  )
  $reset = "$Esc[0m"
  $map = @{
    dim     = "$Esc[2m"
    bold    = "$Esc[1m"
    cyan    = "$Esc[38;2;94;234;212m"
    teal    = "$Esc[38;2;45;212;191m"
    mint    = "$Esc[38;2;167;243;208m"
    gold    = "$Esc[38;2;251;191;36m"
    rose    = "$Esc[38;2;251;113;133m"
    slate   = "$Esc[38;2;148;163;184m"
    white   = "$Esc[38;2;248;250;252m"
    green   = "$Esc[38;2;74;222;128m"
    amber   = "$Esc[38;2;245;158;11m"
  }
  $prefix = if ($Color -and $map.ContainsKey($Color)) { $map[$Color] } else { "" }
  if ($NoNewline) {
    Write-Host -NoNewline ($prefix + $Text + $reset)
  } else {
    Write-Host ($prefix + $Text + $reset)
  }
}

function Get-ModelReady {
  $dit = Join-Path $Root "models\dit"
  if (-not (Test-Path -LiteralPath $dit)) { return $false }
  $hit = Get-ChildItem -LiteralPath $dit -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "*.pt*" -and $_.Length -ge 1000000 } |
    Select-Object -First 1
  return [bool]$hit
}

function Get-RunState {
  [pscustomobject]@{
    HasVenv  = [bool](Test-Path -LiteralPath $VenvPy)
    HasUi    = [bool](Test-Path -LiteralPath $UiIndex)
    HasModel = Get-ModelReady
  }
}

function Invoke-KillOrphans {
  & $KillScript -Quiet
}

function Invoke-EnsureModel {
  param([switch]$Required)

  if (-not (Test-Path -LiteralPath $VenvPy)) {
    if ($Required) {
      Write-Ansi "No .venv-build Python - run [1] Smart Build first." rose
      return $false
    }
    return $false
  }

  if (Get-ModelReady) {
    Write-Ansi "==> DiT model ready:" cyan -NoNewline
    Write-Ansi " models\dit\VTM-ELF.pt" mint
  } else {
    Write-Host ""
    Write-Ansi "==> Downloading DiT model (Hugging Face -> models\dit\VTM-ELF.pt)" cyan
    Write-Ansi "    sinBoo1/VTM-Elf-0.01 - first setup can take several minutes." slate
    Write-Ansi "    Progress below means it is still working - not frozen." slate
    Write-Host ""

    $prevPyPath = $env:PYTHONPATH
    $env:PYTHONPATH = $Root
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $code = 1
    try {
      & $VenvPy -m backend.model_download
      $code = [int]$LASTEXITCODE
    } catch {
      Write-Ansi "Model download threw: $_" rose
      $code = 1
    } finally {
      $ErrorActionPreference = $prev
      if ($null -eq $prevPyPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
      } else {
        $env:PYTHONPATH = $prevPyPath
      }
    }

    if (($code -eq 0) -and (Get-ModelReady)) {
      Write-Ansi "Model download finished." green
    } else {
      Write-Ansi "Model download failed or incomplete." rose
      Write-Ansi "Place VTM-ELF.pt in models\dit or retry Smart Build / Start." slate
      if ($Required) { return $false }
      return $false
    }
  }

  # Full weight checklist (DiT + trackers + OpenSeeFace).
  Write-Host ""
  Write-Ansi "==> Scanning model checklist..." cyan
  $prevPyPath = $env:PYTHONPATH
  $env:PYTHONPATH = $Root
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $checkCode = 1
  try {
    & $VenvPy -m backend.model_download --checklist-only
    $checkCode = [int]$LASTEXITCODE
  } catch {
    Write-Ansi "Checklist scan threw: $_" rose
    $checkCode = 1
  } finally {
    $ErrorActionPreference = $prev
    if ($null -eq $prevPyPath) {
      Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
      $env:PYTHONPATH = $prevPyPath
    }
  }

  if ($checkCode -eq 0) {
    Write-Ansi "Checklist passed - all required models found." green
    return $true
  }

  Write-Ansi "Checklist incomplete - some required models are missing." amber
  if ($Required) { return $false }
  return $false
}

function Show-Menu {
  $state = Get-RunState
  Clear-Host
  Write-Host ""
  Write-Ansi "  ================================================" cyan
  Write-Ansi "         V T M   N O B L E" white
  Write-Ansi "      smart build  ·  source start" slate
  Write-Ansi "  ================================================" cyan
  Write-Host ""
  Write-Ansi "  Start runs python -m backend (webview + ui\dist)." slate
  Write-Ansi "  Setup downloads VTM-ELF.pt into models\dit when missing." slate
  Write-Ansi "  Leftovers are cleared automatically before each action." slate
  Write-Host ""

  Write-Ansi "  -- actions --------------------------------------" teal
  Write-Ansi "  [1]  Smart Build" gold -NoNewline
  Write-Ansi "   deps / UI + drop VTMNoble.exe here" slate
  if ($state.HasVenv -and $state.HasUi) {
    Write-Ansi "  [2]  Start" gold -NoNewline
    Write-Ansi "       ensure model, then run app" slate
  } else {
    Write-Ansi "  [2]  Start" dim -NoNewline
    Write-Ansi "       (need venv + ui\dist - use [1])" dim
  }
  Write-Ansi "  ------------------------------------------------" teal
  Write-Host ""

  if ($state.HasVenv) {
    Write-Ansi "  venv    " slate -NoNewline
    Write-Ansi "* ready" green -NoNewline
    Write-Ansi "  .venv-build" mint
  } else {
    Write-Ansi "  venv    " slate -NoNewline
    Write-Ansi "o missing" amber -NoNewline
    Write-Ansi "  Smart Build installs deps" slate
  }
  if ($state.HasUi) {
    Write-Ansi "  ui      " slate -NoNewline
    Write-Ansi "* ready" green -NoNewline
    Write-Ansi "  ui\dist" mint
  } else {
    Write-Ansi "  ui      " slate -NoNewline
    Write-Ansi "o missing" amber -NoNewline
    Write-Ansi "  Smart Build builds UI" slate
  }
  if ($state.HasModel) {
    Write-Ansi "  model   " slate -NoNewline
    Write-Ansi "* ready" green -NoNewline
    Write-Ansi "  models\dit" mint
  } else {
    Write-Ansi "  model   " slate -NoNewline
    Write-Ansi "o missing" amber -NoNewline
    Write-Ansi "  downloads on Smart Build / Start" slate
  }
  Write-Host ""
}

function Invoke-SmartBuild {
  Write-Host ""
  Write-Ansi "==> Clearing leftovers..." amber
  Invoke-KillOrphans
  $state = Get-RunState
  if ($state.HasVenv) {
    Write-Ansi "==> Smart build (checks deps - skips pip when venv is ready)" cyan
  } else {
    Write-Ansi "==> Smart build (first install - creating venv and installing deps)" cyan
  }
  Write-Ansi "    Long steps use each tool's own progress (pip/npm); copies print start/done." slate
  Write-Host ""
  $code = 0
  try {
    # First install: do not pass -SkipDeps so pip/torch run immediately.
    # Later builds: -SkipDeps, and build.ps1 still installs if imports/CUDA are incomplete.
    if ($state.HasVenv) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File $BuildScript -SkipDeps
    } else {
      & powershell -NoProfile -ExecutionPolicy Bypass -File $BuildScript
    }
    $code = [int]$LASTEXITCODE
  } catch {
    Write-Ansi "Build threw: $_" rose
    $code = 1
  }
  if ($code -ne 0) {
    Write-Host ""
    Write-Ansi "Smart Build failed. Review the message above." rose
    Write-Host ""
    Write-Ansi "Press Enter to return..." slate
    [void][Console]::ReadLine()
    return
  }

  # Explicit setup step: fetch DiT weights into models\dit if missing.
  [void](Invoke-EnsureModel)

  Write-Host ""
  Write-Ansi "Smart Build finished." green
  $rootExe = Join-Path $Root "VTMNoble.exe"
  if (Test-Path -LiteralPath $rootExe) {
    Write-Ansi "  Double-click " slate -NoNewline
    Write-Ansi "VTMNoble.exe" mint -NoNewline
    Write-Ansi " next to start.bat to launch the app." slate
    Write-Ansi "  (You do not need to dig into dist\VTMNoble\.)" slate
  } else {
    Write-Ansi "  Or use menu [2] Start after models are ready." slate
  }
  Write-Host ""
  Write-Ansi "Press Enter to return..." slate
  [void][Console]::ReadLine()
}

function Invoke-StartApp {
  $state = Get-RunState
  if (-not $state.HasVenv) {
    Write-Host ""
    Write-Ansi "No .venv-build Python. Run [1] Smart Build first." rose
    Write-Host ""
    Write-Ansi "Press Enter to return..." slate
    [void][Console]::ReadLine()
    return
  }
  if (-not $state.HasUi) {
    Write-Host ""
    Write-Ansi "UI dist missing. Run [1] Smart Build first." rose
    Write-Host ""
    Write-Ansi "Press Enter to return..." slate
    [void][Console]::ReadLine()
    return
  }

  Write-Host ""
  Write-Ansi "==> Clearing leftovers..." amber
  Invoke-KillOrphans

  if (-not (Invoke-EnsureModel -Required)) {
    Write-Host ""
    Write-Ansi "Press Enter to return..." slate
    [void][Console]::ReadLine()
    return
  }

  Write-Ansi "==> Starting:" cyan -NoNewline
  Write-Ansi " python -m backend" mint
  Write-Ansi "    Close the app (or Ctrl+C) to return to the menu." slate
  Write-Host ""
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $prevPyPath = $env:PYTHONPATH
  $env:PYTHONPATH = $Root
  try {
    & $VenvPy -m backend --ui webview
    $code = $LASTEXITCODE
  } catch {
    Write-Ansi "Failed to start: $_" rose
    $code = 1
  } finally {
    $ErrorActionPreference = $prev
    if ($null -eq $prevPyPath) {
      Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
      $env:PYTHONPATH = $prevPyPath
    }
  }
  Write-Host ""
  if ($code -and $code -ne 0) {
    Write-Ansi "Process exited with code $code" amber
  } else {
    Write-Ansi "App closed." green
  }
  Write-Host ""
  Write-Ansi "Press Enter to return..." slate
  [void][Console]::ReadLine()
}

Enable-PrettyConsole

while ($true) {
  Show-Menu
  Write-Ansi "  Choose: " gold -NoNewline
  $choice = (Read-Host).Trim()
  switch -Regex ($choice) {
    '^1$' { Invoke-SmartBuild }
    '^2$' { Invoke-StartApp }
    default {
      Write-Host ""
      Write-Ansi "Invalid choice." rose
      Start-Sleep -Milliseconds 700
    }
  }
}
