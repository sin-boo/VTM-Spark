# Portable Node.js for the UI builds (dot-source; needs $Root).
# install.bat downloads a pinned Node into .tools\node, the same way it fetches
# uv, so users never install Node themselves and every PC builds with the same
# version. A system Node on PATH is only a fallback when the download fails.
# Keep this file ASCII: Windows PowerShell 5.1 reads BOM-less .ps1 as ANSI.

$NodeVersion = "24.21.0"
$NodeZipName = "node-v$NodeVersion-win-x64.zip"
$NodeZipUrl = "https://nodejs.org/dist/v$NodeVersion/$NodeZipName"
# From https://nodejs.org/dist/v24.21.0/SHASUMS256.txt
$NodeZipSha256 = "158f7685b44de51f6c0df1d153526cbcd3e1bc739a8dfc607721cef75de9e541"

function Get-VtmNodeDir {
  param([string]$RepoRoot)
  return (Join-Path $RepoRoot ".tools\node")
}

function Test-VtmNodeReady {
  param([string]$RepoRoot)
  $dir = Get-VtmNodeDir $RepoRoot
  $node = Join-Path $dir "node.exe"
  if (-not (Test-Path -LiteralPath $node)) { return $false }
  if (-not (Test-Path -LiteralPath (Join-Path $dir "npm.cmd"))) { return $false }
  try {
    $ver = (& $node --version 2>$null | Select-Object -First 1)
    return ([string]$ver).Trim() -eq "v$NodeVersion"
  } catch {
    return $false
  }
}

function Install-VtmNode {
  # Download + verify + unpack the pinned Node into .tools\node. Throws on failure.
  param([string]$RepoRoot)
  $dir = Get-VtmNodeDir $RepoRoot
  Write-Host "==> Downloading Node.js $NodeVersion into .tools\node (one-time, ~35 MB)"
  $tag = [guid]::NewGuid().ToString("N")
  $zip = Join-Path $env:TEMP "vtm-node-$tag.zip"
  $extract = Join-Path $env:TEMP "vtm-node-extract-$tag"
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $prevProgress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"  # PS 5.1's progress bar makes big downloads crawl
    try {
      Invoke-WebRequest -Uri $NodeZipUrl -OutFile $zip -UseBasicParsing
    } finally {
      $ProgressPreference = $prevProgress
    }
    $hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $NodeZipSha256) {
      throw "checksum mismatch for $NodeZipName (got $hash)"
    }
    Expand-Archive -Path $zip -DestinationPath $extract -Force
    $inner = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
    if (-not $inner -or -not (Test-Path -LiteralPath (Join-Path $inner.FullName "node.exe"))) {
      throw "node.exe missing from $NodeZipName"
    }
    if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path (Split-Path $dir -Parent) | Out-Null
    Move-Item -LiteralPath $inner.FullName -Destination $dir
  } finally {
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $extract) { Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue }
  }
}

function Use-VtmNode {
  # Put a working npm on PATH for this process and its children.
  # -Install downloads the pinned Node when missing; without it only an
  # existing .tools\node or a system Node is used. Returns $true when npm works.
  param(
    [string]$RepoRoot,
    [switch]$Install
  )
  if ($Install -and -not (Test-VtmNodeReady $RepoRoot)) {
    try {
      Install-VtmNode $RepoRoot
    } catch {
      Write-Host "    WARNING: could not set up portable Node.js ($($_.Exception.Message))"
    }
  }
  $dir = Get-VtmNodeDir $RepoRoot
  if (Test-Path -LiteralPath (Join-Path $dir "npm.cmd")) {
    $parts = @($env:PATH -split ";" | Where-Object { $_ -and ($_.TrimEnd("\") -ne $dir.TrimEnd("\")) })
    $env:PATH = (@($dir) + $parts) -join ";"
    # Keep npm's download cache inside the app folder too.
    $env:npm_config_cache = Join-Path $RepoRoot ".tools\npm-cache"
    $env:npm_config_update_notifier = "false"
    $env:npm_config_fund = "false"
    # Build-time advisories are for us to fix in the lockfile, not for users.
    $env:npm_config_audit = "false"
    return $true
  }
  if (Get-Command npm -ErrorAction SilentlyContinue) {
    if ($Install) { Write-Host "    Using the system Node.js on PATH instead." }
    return $true
  }
  return $false
}
