# VTM Spark update (update.bat): brings this folder up to the latest version,
# then runs the install so new packages, models and the UI build follow.
# It looks first and only closes VTM Spark when there is something new.
# A git checkout fetches VTM Spark's main from GitHub and fast-forwards to it
# (local work is never overwritten), whatever remote the clone tracks.
# Without git, or in a folder unpacked from the ZIP, it downloads the latest
# ZIP from GitHub and copies it over. Characters, models and settings are not
# in the download, so they stay.
# An install that failed or was closed runs again on the next update.bat, even
# when there is nothing new to download.
# Exit: 0 updated or already current, 1 failed, otherwise the install's code.
# Keep this file ASCII: Windows PowerShell 5.1 reads BOM-less .ps1 as ANSI.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$Repo = "sin-boo/VTM-Spark"
$Branch = "main"
$StateDir = Join-Path $Root ".tools\update"
# What the last ZIP update copied in, so files removed upstream go too.
$Manifest = Join-Path $StateDir "files.txt"
$CommitFile = Join-Path $StateDir "commit.txt"
# Written before the install runs, removed once it finished.
$Pending = Join-Path $StateDir "install-pending.txt"
# GitHub writes the commit into this file when it builds the ZIP (export-subst
# in .gitattributes), so even a fresh ZIP folder knows its version.
$SourceCommit = "backend\packaging\source-commit.txt"
# Short on purpose: the deepest file adds ~100 characters to this path and
# Windows PowerShell 5.1 cannot write past 260.
$Stage = Join-Path $StateDir "new"
$StageZip = Join-Path $StateDir "download.zip"

function Confirm-Vtm([string]$question) {
  $answer = Read-Host "$question [Y/n]"
  return $answer -notmatch '^\s*(n|no)\s*$'
}

function Write-Fail([string]$text) {
  Write-Host $text -ForegroundColor Red
}

function Read-Commit([string]$path) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  $text = ([string](Get-Content -LiteralPath $path -Raw -ErrorAction SilentlyContinue)).Trim()
  if ($text -match '^[0-9a-f]{40}$') { return $text }
  return $null
}

# Folders on another drive or made by another Windows user trip git's
# "dubious ownership" check; this folder is the user's own install.
function Invoke-Git {
  & $script:GitExe -C $Root -c "safe.directory=$($Root.Replace('\', '/'))" -c core.safecrlf=false @args
}

# git on PATH, or the copies Git for Windows and GitHub Desktop bring along
# without putting them on PATH.
function Find-Git {
  $cmd = Get-Command git.exe -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cmd) { return $cmd.Source }
  $candidates = @(
    (Join-Path $env:ProgramFiles "Git\cmd\git.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe")
  )
  $desktop = Join-Path $env:LOCALAPPDATA "GitHubDesktop"
  if (Test-Path -LiteralPath $desktop) {
    $candidates += @(Get-ChildItem -LiteralPath $desktop -Directory -Filter "app-*" -ErrorAction SilentlyContinue |
      Sort-Object Name -Descending |
      ForEach-Object { Join-Path $_.FullName "resources\app\git\cmd\git.exe" })
  }
  foreach ($c in $candidates) {
    if ($c -and (Test-Path -LiteralPath $c -PathType Leaf)) { return $c }
  }
  return $null
}

# --- git checkout -----------------------------------------------------------

# Fetches; returns the commit to move to, or $null when git failed. Always the
# public main: a clone tracking another remote or branch would never see updates.
function Get-GitTarget {
  Invoke-Git fetch --quiet "https://github.com/$Repo.git" $Branch | Out-Host
  if ($LASTEXITCODE -ne 0) {
    Write-Fail "git could not get the latest version from GitHub (see above). Check your internet connection and run update.bat again."
    return $null
  }
  $target = Invoke-Git rev-parse FETCH_HEAD
  if ($LASTEXITCODE -ne 0) { return $null }
  return ([string]$target).Trim()
}

# Before anything closes: can git fast-forward to $target without touching
# the user's own changes?
function Test-GitUpdate([string]$target) {
  Invoke-Git merge-base --is-ancestor HEAD $target
  if ($LASTEXITCODE -ne 0) {
    Write-Fail "This folder has commits that are not on GitHub, so git cannot fast-forward it."
    Write-Host "Merge or rebase onto the latest version yourself, then run update.bat again."
    return $false
  }
  $incoming = @(Invoke-Git diff --name-only HEAD $target)
  $clash = @(Invoke-Git diff --name-only HEAD | Where-Object { $incoming -contains $_ })
  if ($clash.Count -gt 0) {
    Write-Fail "These files were changed in this folder, and the update changes them too:"
    $clash | ForEach-Object { Write-Host "    $_" }
    Write-Host "Commit or stash them (or undo them with: git checkout -- <file>), then run update.bat again."
    return $false
  }
  return $true
}

function Update-WithGit([string]$target) {
  Invoke-Git merge --ff-only $target | Out-Host
  if ($LASTEXITCODE -eq 0) { return $true }
  Write-Fail "git could not update this folder (see above). Nothing was changed; fix what git names, then run update.bat again."
  return $false
}

# --- ZIP folder -------------------------------------------------------------

function Get-LatestCommit {
  try {
    $info = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/commits/$Branch" `
      -Headers @{ "User-Agent" = "VTM-Spark-update" } -UseBasicParsing -ErrorAction Stop
    return [string]$info.sha
  } catch {
    Write-Host "    could not ask GitHub for the latest version - downloading it to compare" -ForegroundColor DarkGray
    return $null
  }
}

# Unpacks the ZIP into $Stage without GitHub's "VTM-Spark-<commit>" top folder.
function Expand-Update([string]$zip) {
  if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $Stage | Out-Null
  $base = $Stage.TrimEnd('\') + '\'
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive = [IO.Compression.ZipFile]::OpenRead($zip)
  try {
    foreach ($entry in $archive.Entries) {
      $slash = $entry.FullName.IndexOf('/')
      if ($slash -lt 0) { continue }
      $rel = $entry.FullName.Substring($slash + 1)
      if (-not $rel) { continue }
      $dest = [IO.Path]::GetFullPath($base + $rel.Replace('/', '\'))
      if (-not $dest.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) { continue }
      if ($rel.EndsWith('/')) {
        [void][IO.Directory]::CreateDirectory($dest)
        continue
      }
      [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($dest))
      [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
    }
  } finally {
    $archive.Dispose()
  }
}

# Downloads $ref into $Stage. Returns the commit it holds ("" when unknown),
# or $null when the download failed.
function Get-Update([string]$ref) {
  New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
  Write-Host "==> Downloading the latest VTM Spark" -ForegroundColor Cyan
  # WebClient: Invoke-WebRequest -OutFile reads [ ] in the folder path as wildcards.
  $web = New-Object Net.WebClient
  $web.Headers["User-Agent"] = "VTM-Spark-update"
  try {
    $web.DownloadFile("https://github.com/$Repo/archive/$ref.zip", $StageZip)
  } catch {
    Write-Fail "Download failed: $($_.Exception.Message)"
    Write-Host "Check your internet connection and run update.bat again."
    return $null
  }
  try {
    Expand-Update $StageZip
  } catch {
    Write-Fail "Could not unpack the download: $($_.Exception.Message)"
    Write-Host "If the message mentions a path that is too long, move the VTM Spark folder somewhere shorter (like C:\VTM Spark) and run update.bat there."
    return $null
  } finally {
    Remove-Item -LiteralPath $StageZip -Force -ErrorAction SilentlyContinue
  }
  if (-not (Test-Path -LiteralPath (Join-Path $Stage "install.bat")) -or
      -not (Test-Path -LiteralPath (Join-Path $Stage "backend\__main__.py"))) {
    Write-Fail "The download does not look like VTM Spark - nothing was changed."
    return $null
  }
  $sha = Read-Commit (Join-Path $Stage $SourceCommit)
  if ($sha) { return $sha }
  return ""
}

# Shipped files of a version, from GitHub (for a first update with no list yet).
function Get-CommitFiles([string]$sha) {
  try {
    $tree = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/git/trees/$sha`?recursive=1" `
      -Headers @{ "User-Agent" = "VTM-Spark-update" } -UseBasicParsing -ErrorAction Stop
    if ($tree.truncated) { return @() }
    return @($tree.tree | Where-Object { $_.type -eq "blob" } | ForEach-Object { $_.path.Replace('/', '\') })
  } catch {
    return @()
  }
}

function Install-Update([string]$current, [string]$sha) {
  $prefix = $Stage.TrimEnd('\').Length + 1
  $files = @(Get-ChildItem -LiteralPath $Stage -Recurse -File -Force |
    ForEach-Object { $_.FullName.Substring($prefix) })
  $old = @()
  if (Test-Path -LiteralPath $Manifest) {
    $old = @(Get-Content -LiteralPath $Manifest)
  } elseif ($current) {
    $old = Get-CommitFiles $current
  }

  Write-Host "==> Copying the new version in" -ForegroundColor Cyan
  & robocopy $Stage $Root /E /IS /IT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Host
  if ($LASTEXITCODE -ge 8) {
    Write-Fail "Copying the new files failed (robocopy $LASTEXITCODE). Close anything using files in this folder and run update.bat again."
    return $false
  }

  $now = @{}
  foreach ($f in $files) { $now[$f.ToLowerInvariant()] = $true }
  foreach ($f in $old) {
    if (-not $f -or $now.ContainsKey($f.ToLowerInvariant())) { continue }
    $path = Join-Path $Root $f
    if (Test-Path -LiteralPath $path -PathType Leaf) {
      Write-Host "    removing $f (no longer part of VTM Spark)"
      Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
  }
  Set-Content -LiteralPath $Manifest -Value $files -Encoding UTF8
  if ($sha) {
    Set-Content -LiteralPath $CommitFile -Value $sha -Encoding ASCII
  } else {
    Remove-Item -LiteralPath $CommitFile -Force -ErrorAction SilentlyContinue
  }
  Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
  return $true
}

# --- main -------------------------------------------------------------------

Write-Host ""
Write-Host "VTM Spark - Update" -ForegroundColor Cyan
Write-Host ""
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$script:GitExe = $null
if (Test-Path -LiteralPath (Join-Path $Root ".git")) { $script:GitExe = Find-Git }
$useGit = [bool]$script:GitExe
$gitTarget = $null
$zipCommit = $null
$current = $null
$hasNew = $false

Write-Host "==> Looking for a new version" -ForegroundColor Cyan
if ($useGit) {
  $gitTarget = Get-GitTarget
  if (-not $gitTarget) { exit 1 }
  Invoke-Git merge-base --is-ancestor $gitTarget HEAD
  $hasNew = ($LASTEXITCODE -ne 0)
  if ($hasNew -and -not (Test-GitUpdate $gitTarget)) { exit 1 }
} else {
  $current = Read-Commit $CommitFile
  if (-not $current) { $current = Read-Commit (Join-Path $Root $SourceCommit) }
  $latest = Get-LatestCommit
  if (-not $latest -or $latest -ne $current) {
    $ref = $Branch
    if ($latest) { $ref = $latest }
    $zipCommit = Get-Update $ref
    if ($null -eq $zipCommit) { exit 1 }
    if (-not $zipCommit) { $zipCommit = $latest }
    $hasNew = (-not $zipCommit -or $zipCommit -ne $current)
    if (-not $hasNew) { Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue }
  }
}

$finishInstall = (-not $hasNew) -and (Test-Path -LiteralPath $Pending)
if (-not $hasNew -and -not $finishInstall) {
  Write-Host ""
  Write-Host "VTM Spark is already up to date." -ForegroundColor Green
  exit 0
}
if ($finishInstall) {
  Write-Host "    no new version, but the last update did not finish installing - finishing it now"
}

$stopApp = Join-Path $PSScriptRoot "stop-app.ps1"
& $stopApp -List
if ($LASTEXITCODE -ne 0) {
  if (-not (Confirm-Vtm "VTM Spark is running and will close for the update. Continue?")) {
    Write-Host "Nothing was changed."
    exit 0
  }
}
Write-Host "==> Closing VTM Spark" -ForegroundColor Cyan
& $stopApp

if ($hasNew) {
  if ($useGit) {
    Write-Host "==> Moving this folder to the latest version with git" -ForegroundColor Cyan
    $ok = Update-WithGit $gitTarget
  } else {
    $ok = Install-Update $current $zipCommit
  }
  if (-not $ok) { exit 1 }
}

Write-Host ""
Write-Host "==> Finishing the update (install)" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
Set-Content -LiteralPath $Pending -Value "install.bat has not finished since the last update" -Encoding ASCII
# The freshly copied start-menu.ps1, the same script install.bat runs.
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start-menu.ps1") -Action install -NoPause
$code = $LASTEXITCODE
# 2 = the app is built and only optional steps need attention; install.bat
# tells the user about those, update.bat does not have to keep retrying it.
if ($code -eq 0 -or $code -eq 2) {
  Remove-Item -LiteralPath $Pending -Force -ErrorAction SilentlyContinue
} else {
  Write-Fail "The install did not finish. Run update.bat again to retry it."
}
exit $code
