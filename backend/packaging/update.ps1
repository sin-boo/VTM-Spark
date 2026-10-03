# VTM Spark update (update.bat): brings this folder up to the latest version,
# then runs install.bat so new packages, models and the UI build follow.
# A git checkout pulls (fast-forward only, so local work is never overwritten).
# Without git, or in a folder unpacked from the ZIP, it downloads the latest
# ZIP from GitHub and copies it over. Characters, models and settings are not
# in the download, so they stay.
# Exit: 0 updated or already current, 1 failed, otherwise install.bat's code.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Repo = "sin-boo/VTM-Spark"
$Branch = "main"
# What the last ZIP update copied in, so files removed upstream go too.
$StateDir = Join-Path $Root ".tools\update"
$Manifest = Join-Path $StateDir "files.txt"
$CommitFile = Join-Path $StateDir "commit.txt"

function Confirm-Vtm([string]$question) {
  $answer = Read-Host "$question [Y/n]"
  return $answer -notmatch '^\s*(n|no)\s*$'
}

function Write-Fail([string]$text) {
  Write-Host $text -ForegroundColor Red
}

# $true = new files, $false = already current, $null = failed.
function Update-WithGit {
  $before = (& git -C $Root rev-parse HEAD)
  & git -C $Root rev-parse --abbrev-ref --symbolic-full-name "@{u}" *> $null
  if ($LASTEXITCODE -eq 0) {
    & git -C $Root pull --ff-only | Out-Host
  } else {
    & git -C $Root pull --ff-only "https://github.com/$Repo.git" $Branch | Out-Host
  }
  if ($LASTEXITCODE -ne 0) {
    Write-Fail "git could not update this folder (see above)."
    Write-Host "Your own changes or commits conflict with the update. Commit or stash them, then run update.bat again."
    return $null
  }
  return ((& git -C $Root rev-parse HEAD) -ne $before)
}

function Update-FromZip {
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  $sha = $null
  try {
    $info = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/commits/$Branch" `
      -Headers @{ "User-Agent" = "VTM-Spark-update" } -UseBasicParsing -ErrorAction Stop
    $sha = [string]$info.sha
  } catch {
    Write-Host "    could not ask GitHub for the latest version - downloading it anyway" -ForegroundColor DarkGray
  }
  if ($sha -and (Test-Path -LiteralPath $CommitFile) -and
      ((Get-Content -LiteralPath $CommitFile -Raw).Trim() -eq $sha)) {
    return $false
  }

  $ref = $Branch
  if ($sha) { $ref = $sha }
  $stage = Join-Path $Root ".tools\downloads\update"
  if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $stage | Out-Null
  $zip = Join-Path $stage "vtm-spark.zip"
  $extract = Join-Path $stage "x"
  try {
    Write-Host "==> Downloading the latest VTM Spark" -ForegroundColor Cyan
    Invoke-WebRequest -Uri "https://github.com/$Repo/archive/$ref.zip" -OutFile $zip -UseBasicParsing -ErrorAction Stop
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $extract)
  } catch {
    Write-Fail "Download failed: $($_.Exception.Message)"
    Write-Host "Check your internet connection and run update.bat again."
    return $null
  }
  $src = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
  if (-not $src -or -not (Test-Path -LiteralPath (Join-Path $src.FullName "install.bat")) -or
      -not (Test-Path -LiteralPath (Join-Path $src.FullName "backend\__main__.py"))) {
    Write-Fail "The download does not look like VTM Spark - nothing was changed."
    return $null
  }

  $prefix = $src.FullName.Length + 1
  $files = @(Get-ChildItem -LiteralPath $src.FullName -Recurse -File -Force |
    ForEach-Object { $_.FullName.Substring($prefix) })

  Write-Host "==> Copying the new version in" -ForegroundColor Cyan
  & robocopy $src.FullName $Root /E /IS /IT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Host
  if ($LASTEXITCODE -ge 8) {
    Write-Fail "Copying the new files failed (robocopy $LASTEXITCODE). Close anything using VTM Spark and run update.bat again."
    return $null
  }

  if (Test-Path -LiteralPath $Manifest) {
    $now = @{}
    foreach ($f in $files) { $now[$f.ToLowerInvariant()] = $true }
    foreach ($old in (Get-Content -LiteralPath $Manifest)) {
      if (-not $old -or $now.ContainsKey($old.ToLowerInvariant())) { continue }
      $path = Join-Path $Root $old
      if (Test-Path -LiteralPath $path -PathType Leaf) {
        Write-Host "    removing $old (no longer part of VTM Spark)"
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
      }
    }
  }
  New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
  Set-Content -LiteralPath $Manifest -Value $files -Encoding UTF8
  if ($sha) { Set-Content -LiteralPath $CommitFile -Value $sha -Encoding ASCII }
  Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
  return $true
}

Write-Host ""
Write-Host "VTM Spark - Update" -ForegroundColor Cyan
Write-Host ""

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

$git = Get-Command git.exe -ErrorAction SilentlyContinue
if ($git -and (Test-Path -LiteralPath (Join-Path $Root ".git"))) {
  Write-Host "==> Pulling the latest version with git" -ForegroundColor Cyan
  $changed = Update-WithGit
} else {
  $changed = Update-FromZip
}
if ($null -eq $changed) { exit 1 }
if (-not $changed) {
  Write-Host ""
  Write-Host "VTM Spark is already up to date." -ForegroundColor Green
  exit 0
}

Write-Host ""
Write-Host "==> Finishing the update with install.bat" -ForegroundColor Cyan
& cmd /c "`"$(Join-Path $Root 'install.bat')`""
exit $LASTEXITCODE
