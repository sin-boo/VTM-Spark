# VTM Studio — install.bat / run.exe
param(
  [ValidateSet("install", "run")]
  [string]$Action = "run"
)

$ErrorActionPreference = "Stop"
# PS 7+ can treat taskkill's "process not found" (128) as a terminating error.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$VenvPy = Join-Path $Root ".venv-build\Scripts\python.exe"
$UiIndex = Join-Path $Root "ui\dist\index.html"
$KillScript = Join-Path $PSScriptRoot "kill-orphans.ps1"
$BuildScript = Join-Path $PSScriptRoot "build.ps1"
$Esc = [char]27

. (Join-Path $PSScriptRoot "console-progress.ps1")
. (Join-Path $PSScriptRoot "venv-home.ps1")
. (Join-Path $PSScriptRoot "node-tools.ps1")

function Enable-PrettyConsole {
  try {
    $Host.UI.RawUI.WindowTitle = "VTM Studio"
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
  [void](Repair-VtmVenvHome -VenvDir (Join-Path $Root ".venv-build") -PythonExe $VenvPy)
  [pscustomobject]@{
    HasVenv  = [bool](Test-VtmPythonExe $VenvPy)
    HasUi    = [bool](Test-Path -LiteralPath $UiIndex)
    HasModel = Get-ModelReady
  }
}

function Invoke-KillOrphans {
  param([switch]$Fast)
  if ($Fast) {
    & $KillScript -Quiet -Fast
  } else {
    & $KillScript -Quiet
  }
}

function Get-VtmConsoleHwnd {
  if (-not ("Win32.SplashWnd" -as [type])) {
    Add-Type -Namespace Win32 -Name SplashWnd -MemberDefinition @"
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
[DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
"@
  }
  return [Win32.SplashWnd]::GetConsoleWindow()
}

function Hide-VtmConsole {
  param($Hwnd)
  if ($Hwnd -ne [IntPtr]::Zero) {
    [void][Win32.SplashWnd]::ShowWindow($Hwnd, 0)
  }
}

function Show-VtmConsole {
  param($Hwnd)
  if ($Hwnd -ne [IntPtr]::Zero) {
    [void][Win32.SplashWnd]::ShowWindow($Hwnd, 5)
  }
}

function Invoke-EnsureModel {
  param([switch]$Required)

  if (-not (Test-Path -LiteralPath $VenvPy)) {
    if ($Required) {
      Write-Ansi "No .venv-build Python - run install.bat first." rose
      return $false
    }
    return $false
  }

  Write-Host ""
  Write-Ansi "==> Checking Hugging Face for new DiT checkpoints" cyan
  Write-Ansi "    Existing local files are kept. Only missing Hub weights download." slate
  Write-Host ""

    $prevPyPath = $env:PYTHONPATH
    $env:PYTHONPATH = $Root
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $code = 1
    try {
      # Out-Host: show stdout and keep it out of this function's return value.
      & $VenvPy -m backend.model_download | Out-Host
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
      Write-Ansi "Model catalog sync finished." green
    } else {
      Write-Ansi "Model download failed or incomplete." rose
      Write-Ansi "Place a .pt file in models\dit or re-run install.bat." slate
      if ($Required) { return $false }
      return $false
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
    & $VenvPy -m backend.model_download --checklist-only | Out-Host
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

function Invoke-EnsureVtmStudioCam {
  # Register bundled DirectShow filter as 'VTM Studio Cam' (UAC once).
  $installBat = Join-Path $Root "vendor\tools\vtm_studio_cam\Install-VTMStudioCam.bat"
  $dll64 = Join-Path $Root "vendor\tools\vtm_studio_cam\UnityCaptureFilter64.dll"
  if (-not (Test-Path -LiteralPath $installBat) -or -not (Test-Path -LiteralPath $dll64)) {
    Write-Ansi "==> VTM Studio Cam filters missing under vendor\tools\vtm_studio_cam" amber
    return $false
  }
  if (-not (Test-Path -LiteralPath $VenvPy)) {
    return $false
  }

  $prevPyPath = $env:PYTHONPATH
  $env:PYTHONPATH = $Root
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $ready = $false
  try {
    & $VenvPy -c "from backend.vcam_device import device_available; raise SystemExit(0 if device_available() else 1)" | Out-Host
    $ready = ($LASTEXITCODE -eq 0)
  } catch {
    $ready = $false
  } finally {
    $ErrorActionPreference = $prev
    if ($null -eq $prevPyPath) {
      Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
      $env:PYTHONPATH = $prevPyPath
    }
  }

  if ($ready) {
    Write-Ansi "==> Virtual camera ready:" cyan -NoNewline
    Write-Ansi " VTM Studio Cam" mint
    return $true
  }

  Write-Host ""
  Write-Ansi "==> Installing the virtual camera: VTM Studio Cam" cyan
  Write-Host ""
  Write-Ansi "    Windows will now ask for administrator permission." white
  Write-Ansi "    This is for the virtual camera. It lets OBS, Discord, Zoom and other apps" slate
  Write-Ansi "    use VTM Studio as a webcam, and Windows only allows adding a camera as admin." slate
  Write-Ansi "    The prompt may say 'Windows Command Processor' - that is this step." slate
  Write-Ansi "    Click Yes. You are only asked once." slate
  Write-Host ""
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    Start-Process -FilePath $installBat -WorkingDirectory (Split-Path $installBat -Parent) -Wait -Verb RunAs
  } catch {
    Write-Ansi "Virtual camera skipped - the Windows permission prompt was declined." amber
    Write-Ansi "Everything else still works. Click Virtual camera in the app to add it later." slate
    $ErrorActionPreference = $prev
    return $false
  }
  $ErrorActionPreference = $prev

  Start-Sleep -Milliseconds 800
  $prevPyPath = $env:PYTHONPATH
  $env:PYTHONPATH = $Root
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $VenvPy -c "from backend.vcam_device import device_available; raise SystemExit(0 if device_available() else 1)" | Out-Host
    $ready = ($LASTEXITCODE -eq 0)
  } catch {
    $ready = $false
  } finally {
    $ErrorActionPreference = $prev
    if ($null -eq $prevPyPath) {
      Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
      $env:PYTHONPATH = $prevPyPath
    }
  }

  if ($ready) {
    Write-Ansi "VTM Studio Cam installed." green
    return $true
  }
  Write-Ansi "VTM Studio Cam not detected yet — open Virtual camera in the app after Start." amber
  return $false
}

function Test-WebView2 {
  # Same registry keys pywebview checks. Without the runtime pywebview falls
  # back to the IE engine, which cannot run the desk UI (blank window).
  $guid = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
  $keys = @(
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$guid",
    "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$guid",
    "HKCU:\Software\Microsoft\EdgeUpdate\Clients\$guid"
  )
  foreach ($key in $keys) {
    $pv = (Get-ItemProperty -LiteralPath $key -Name pv -ErrorAction SilentlyContinue).pv
    if ($pv -and $pv -ne "0.0.0.0") { return $true }
  }
  return $false
}

function Invoke-EnsureWebView2 {
  if (Test-WebView2) {
    Write-Ansi "==> WebView2 runtime present" cyan
    return $true
  }
  Write-Host ""
  Write-Ansi "==> Installing Microsoft Edge WebView2 runtime (the desk window needs it)" cyan
  # Microsoft's Evergreen bootstrapper, run silently. Unelevated it installs
  # per-user with no UAC; if that is blocked, retry once system-wide (one UAC).
  $url = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
  $exe = Join-Path $env:TEMP ("vtm-webview2-" + [guid]::NewGuid().ToString("N") + ".exe")
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $prevProgress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
      Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    } finally {
      $ProgressPreference = $prevProgress
    }
    $proc = Start-Process -FilePath $exe -ArgumentList "/silent", "/install" -Wait -PassThru
    Write-Ansi "    WebView2 installer exited $($proc.ExitCode)" slate
    if (-not (Test-WebView2)) {
      Write-Host ""
      Write-Ansi "    Windows will now ask for administrator permission." white
      Write-Ansi "    This is for WebView2, Microsoft's component that draws the VTM Studio window." slate
      Write-Ansi "    The quick install did not take, so it installs for all users instead." slate
      Write-Ansi "    The prompt will say 'Microsoft Edge Update Setup' - that is this step. Click Yes." slate
      Write-Host ""
      $proc = Start-Process -FilePath $exe -ArgumentList "/silent", "/install" -Verb RunAs -Wait -PassThru
      Write-Ansi "    WebView2 installer exited $($proc.ExitCode)" slate
    }
  } catch {
    Write-Ansi "WebView2 install failed: $_" amber
  } finally {
    Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
  }
  if (Test-WebView2) {
    Write-Ansi "WebView2 runtime installed." green
    return $true
  }
  Write-Ansi "WebView2 runtime still missing." amber
  return $false
}

function Invoke-EnsureTrackLab {
  # Track Lab shares .venv-build: copy OpenSeeFace Python, probe imports, and
  # install the lab UI npm packages. Same steps track_lab\start.ps1 runs lazily.
  $labDir = Join-Path $Root "track_lab"
  $setup = Join-Path $labDir "setup.ps1"
  $labUi = Join-Path $labDir "ui"
  if (-not (Test-Path -LiteralPath $setup)) {
    Write-Ansi "==> Track Lab missing under track_lab\ - skipped" amber
    return $false
  }

  Write-Host ""
  Write-Ansi "==> Setting up Track Lab" cyan
  $ok = $true

  $setupCode = 1
  try {
    $setupCode = Invoke-ProcessWithHeartbeat `
      -FilePath "powershell.exe" `
      -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $setup) `
      -Activity "track_lab setup.ps1" `
      -HeartbeatSeconds 10
  } catch {
    Write-Ansi "Track Lab setup threw: $_" rose
    $setupCode = 1
  }
  if ($setupCode -ne 0) {
    Write-Ansi "Track Lab setup.ps1 exited $setupCode." amber
    $ok = $false
  }

  if (Test-Path -LiteralPath (Join-Path $labUi "node_modules")) {
    Write-Ansi "    track_lab\ui\node_modules present - skipping npm" slate
  } elseif (-not (Use-VtmNode -RepoRoot $Root)) {
    Write-Ansi "Node.js missing (.tools\node) - Track Lab UI packages not installed." amber
    $ok = $false
  } else {
    $npmCode = 1
    Push-Location $labUi
    try {
      $npmCode = Invoke-ProcessWithHeartbeat `
        -FilePath "cmd.exe" `
        -ArgumentList @("/c", "npm", "ci") `
        -Activity "npm ci (track_lab\ui)" `
        -HeartbeatSeconds 12
      if ($npmCode -ne 0) {
        $npmCode = Invoke-ProcessWithHeartbeat `
          -FilePath "cmd.exe" `
          -ArgumentList @("/c", "npm", "install") `
          -Activity "npm install (track_lab\ui)" `
          -HeartbeatSeconds 12
      }
    } catch {
      Write-Ansi "Track Lab npm install threw: $_" rose
      $npmCode = 1
    } finally {
      Pop-Location
    }
    if ($npmCode -ne 0) {
      Write-Ansi "Track Lab npm install exited $npmCode." amber
      $ok = $false
    }
  }

  if ($ok) {
    Write-Ansi "Track Lab ready." green
  }
  return $ok
}

function Write-InstallStep {
  param(
    [string]$Label,
    [bool]$Ok,
    [string]$Hint = ""
  )
  Write-Ansi ("  " + $Label.PadRight(16)) white -NoNewline
  if ($Ok) {
    Write-Ansi "OK" green
    return
  }
  Write-Ansi "needs attention" amber
  if ($Hint) {
    Write-Ansi "    $Hint" slate
  }
}

function Invoke-SmartBuild {
  # Sets $script:InstallExitCode: 0 all OK, 1 app build failed,
  # 2 app built but a later step (models / virtual cam / Track Lab) needs attention.
  $script:InstallExitCode = 1
  Write-Host ""
  Write-Ansi "==> Clearing leftovers..." amber
  Invoke-KillOrphans
  $state = Get-RunState
  if ($state.HasVenv) {
    Write-Ansi "==> Install / update (checks deps - installs anything missing, e.g. pyvirtualcam)" cyan
  } else {
    Write-Ansi "==> Install (first run - creating venv and installing deps)" cyan
  }
  Write-Ansi "    Long steps use each tool's own progress (uv/npm); copies print start/done." slate
  Write-Host ""
  $code = 0
  try {
    # First install: do not pass -SkipDeps so uv/torch run immediately.
    # Later builds: -SkipDeps, and build.ps1 still installs if imports/CUDA
    # (or new requirements like pyvirtualcam) are incomplete.
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
    Write-Ansi "Install failed during the app build (exit $code). Review the message above." rose
    Write-Ansi "  Common causes:" slate
    Write-Ansi "    - No internet connection (uv, Node.js, PyTorch and npm download packages)" slate
    Write-Ansi "    - A proxy or firewall blocking nodejs.org, github.com or pypi.org" slate
    Write-Ansi "    - No NVIDIA driver - the CUDA check needs a recent NVIDIA GPU driver" slate
    Write-Ansi "    - Antivirus or a running app locking .venv-build - close VTM Studio / pause AV and retry" slate
    Write-Ansi "  Re-running install.bat is safe; it keeps what finished and retries the rest." slate
    Write-Host ""
    Write-Ansi "Press Enter to return..." slate
    [void][Console]::ReadLine()
    $script:InstallExitCode = 1
    return
  }

  # The desk window is WebView2; without it pywebview shows a blank IE window.
  $webviewOk = [bool](@(Invoke-EnsureWebView2)[-1])

  # Track Lab (face-tracking bench): OSF Python copy + lab UI npm packages.
  # Uses the portable Node build.ps1 put in .tools\node.
  $trackLabOk = [bool](@(Invoke-EnsureTrackLab)[-1])

  # Explicit setup step: fetch DiT weights into models\dit if missing.
  $modelsOk = [bool](@(Invoke-EnsureModel)[-1])

  # Register bundled DirectShow virtual camera (VTM Studio Cam) once.
  $vcamOk = [bool](@(Invoke-EnsureVtmStudioCam)[-1])

  $allOk = $webviewOk -and $modelsOk -and $vcamOk -and $trackLabOk
  Write-Host ""
  Write-Ansi "==> Install summary" cyan
  Write-InstallStep "App build" $true
  Write-InstallStep "WebView2" $webviewOk "Check your internet. run.exe installs it automatically on the next start."
  Write-InstallStep "Models" $modelsOk "Check your internet. Missing models download automatically when you start the app."
  Write-InstallStep "Virtual camera" $vcamOk "Click Virtual camera in the app to add it (Windows asks for admin once)."
  Write-InstallStep "Track Lab" $trackLabOk "Check your internet and re-run install.bat, or run track_lab\start.bat."
  Write-Host ""
  if ($allOk) {
    Write-Ansi "Install finished." green
    $script:InstallExitCode = 0
  } else {
    Write-Ansi "Install finished, but some steps need attention (see above)." amber
    Write-Ansi "  The desk may still open. Re-running install.bat is safe." slate
    $script:InstallExitCode = 2
  }
  Write-Ansi "  Double-click run.exe to open the desk." slate
  Write-Host ""
  Write-Ansi "Press Enter to return..." slate
  [void][Console]::ReadLine()
}

function Test-UiStale {
  if (-not (Test-Path -LiteralPath $UiIndex)) { return $true }
  $distTime = (Get-Item -LiteralPath $UiIndex).LastWriteTimeUtc
  $newestSrc = Get-ChildItem -Path (Join-Path $Root "ui\src"), (Join-Path $Root "ui\index.html"), (Join-Path $Root "ui\public") -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike "_*" } |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
  return ($null -ne $newestSrc -and $newestSrc.LastWriteTimeUtc -gt $distTime)
}

function Invoke-BuildDeskUi {
  Write-Ansi "Building desk UI from current source…" cyan
  $uiDir = Join-Path $Root "ui"
  if (-not (Use-VtmNode -RepoRoot $Root)) {
    Write-Ansi "Node.js missing (.tools\node). Run install.bat." rose
    return 1
  }
  $buildCode = 1
  Push-Location $uiDir
  # Plain vite output; its live progress overwrites the next console lines.
  $prevCi = $env:CI
  $env:CI = "true"
  try {
    # Start-Process ExitCode. kill-orphans' taskkill (128) used to leak into
    # the next native status and mark a good vite build as failed.
    $buildCode = Invoke-ProcessWithHeartbeat `
      -FilePath "cmd.exe" `
      -ArgumentList @("/c", "npm", "run", "build") `
      -Activity "vite build" `
      -HeartbeatSeconds 10
  } catch {
    Write-Ansi "UI rebuild threw: $_" rose
    $buildCode = 1
  } finally {
    $env:CI = $prevCi
    Pop-Location
  }
  if ($buildCode -ne 0) {
    Write-Ansi "UI rebuild exited $buildCode." rose
  }
  return [int]$buildCode
}

function Test-VtmDeskWindow {
  $named = Get-Process -Name pythonw,python,VTMNoble -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero -and $_.MainWindowTitle -match 'VTM Studio' }
  if ($named) { return $true }
  try {
    if (-not ("Win32.FindVtm" -as [type])) {
      Add-Type -Namespace Win32 -Name FindVtm -MemberDefinition @"
[DllImport("user32.dll", CharSet = CharSet.Unicode)]
public static extern IntPtr FindWindowW(string lpClassName, string lpWindowName);
"@
    }
    $hwnd = [Win32.FindVtm]::FindWindowW($null, "VTM Studio")
    return ($hwnd -ne [IntPtr]::Zero)
  } catch {
    return $false
  }
}

function Wait-VtmDeskWindow {
  param($Proc, [int]$TimeoutSec = 40)
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    if ($null -eq $Proc) { return $false }
    try { $null = $Proc.Refresh() } catch {}
    if ($Proc.HasExited) { return $false }
    if (Test-VtmDeskWindow) { return $true }
    Start-Sleep -Milliseconds 400
  }
  return $false
}

function Show-StartFailure {
  param($ConsoleHwnd, [string]$Reason)
  Show-VtmConsole $ConsoleHwnd
  Write-Host ""
  Write-Ansi $Reason rose
  $log = Join-Path $Root "models\vtm_noble.log"
  if (Test-Path -LiteralPath $log) {
    Write-Host ""
    Write-Ansi "Last log lines:" slate
    Get-Content -LiteralPath $log -Tail 30
  }
  Write-Host ""
  Write-Ansi "Press Enter to return..." slate
  [void][Console]::ReadLine()
}

function Invoke-StartApp {
  $state = Get-RunState
  if (-not $state.HasVenv) {
    Write-Host ""
    Write-Ansi "No working .venv-build Python. Run install.bat first." rose
    Write-Ansi "    A leftover venv whose conda/pythonw home was deleted also needs a rebuild." slate
    Write-Host ""
    Write-Ansi "Press Enter to return..." slate
    [void][Console]::ReadLine()
    return
  }

  $consoleHwnd = Get-VtmConsoleHwnd
  Write-Host ""
  Write-Ansi "==> Starting operator desk…" cyan
  Write-Ansi "    This window stays until the splash appears." slate
  Invoke-KillOrphans -Fast

  $Pyw = Resolve-VtmDeskPython -VenvDir (Join-Path $Root ".venv-build") -PythonExe $VenvPy

  $prevPyPath = $env:PYTHONPATH
  $env:PYTHONPATH = $Root
  $proc = $null
  $code = 0
  try {
    if (Test-UiStale) {
      $buildCode = Invoke-BuildDeskUi
      if ($buildCode -ne 0 -and -not (Test-Path -LiteralPath $UiIndex)) {
        Write-Host ""
        Write-Ansi "UI build failed and ui\dist is missing. Run install.bat." rose
        Write-Host ""
        Write-Ansi "Press Enter to return..." slate
        [void][Console]::ReadLine()
        return
      }
      if ($buildCode -ne 0) {
        Write-Ansi "UI rebuild failed; launching last ui\dist." rose
      }
    }

    $proc = Start-Process -FilePath $Pyw -ArgumentList "-m","backend","--ui","webview" -WorkingDirectory $Root -PassThru

    if (-not (Wait-VtmDeskWindow $proc)) {
      if ($null -ne $proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Invoke-KillOrphans -Fast
      }
      $why = "Desk window did not open."
      if ($null -ne $proc -and $proc.HasExited) {
        $why = "Desk process exited before the splash appeared."
      }
      Show-StartFailure $consoleHwnd $why
      return
    }
    Hide-VtmConsole $consoleHwnd
    exit 0
  } catch {
    $code = 1
    Show-VtmConsole $consoleHwnd
    Write-Ansi "Failed to start: $_" rose
  } finally {
    if ($null -eq $prevPyPath) {
      Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
      $env:PYTHONPATH = $prevPyPath
    }
  }
  if ($code -and $code -ne 0) {
    Show-StartFailure $consoleHwnd "Process exited with code $code"
    return
  }
}

Enable-PrettyConsole

if ($Action -eq "install") {
  $script:InstallExitCode = 1
  Invoke-SmartBuild
  exit $script:InstallExitCode
} else {
  Invoke-StartApp
}
