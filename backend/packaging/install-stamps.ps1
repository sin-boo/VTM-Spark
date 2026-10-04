# What install.bat last built or installed from, kept as content hashes (dot-source).
# File dates cannot say: an update from the GitHub ZIP stamps every file with
# the commit time (in GitHub's time zone), often older than the user's last
# build, so a date check skipped the UI build and new packages after updates.
# Keep this file ASCII: Windows PowerShell 5.1 reads BOM-less .ps1 as ANSI.

# Top-level ui\ files the build reads; ui\src and ui\public count whole.
# run-stub.cs UiSourceHash hashes the same files the same way.
$UiTopFiles = @(
  "index.html", "package.json", "package-lock.json", "vite.config.ts",
  "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json"
)

function Get-VtmTextHash {
  param([string]$Text)
  $sha = [Security.Cryptography.SHA256]::Create()
  try {
    $bytes = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text))
  } finally {
    $sha.Dispose()
  }
  return (($bytes | ForEach-Object { $_.ToString("x2") }) -join "")
}

function Get-VtmFileHash {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return "" }
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

# One hash over every file the desk UI is built from: "path hash\n" per file,
# paths relative to ui\ with forward slashes, sorted ordinal. Files starting
# with "_" are local scratch (ui\_mock_boot.py and the like) and do not count.
function Get-UiSourceHash {
  param([string]$RepoRoot)
  $ui = Join-Path $RepoRoot "ui"
  $files = New-Object System.Collections.Generic.List[string]
  foreach ($name in $UiTopFiles) {
    if (Test-Path -LiteralPath (Join-Path $ui $name) -PathType Leaf) { $files.Add($name) }
  }
  foreach ($dir in @("src", "public")) {
    $path = Join-Path $ui $dir
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { continue }
    foreach ($f in [IO.Directory]::EnumerateFiles($path, "*", [IO.SearchOption]::AllDirectories)) {
      if ([IO.Path]::GetFileName($f).StartsWith("_")) { continue }
      $files.Add($f.Substring($ui.Length + 1).Replace('\', '/'))
    }
  }
  $sorted = $files.ToArray()
  [Array]::Sort($sorted, [StringComparer]::Ordinal)
  $sb = New-Object System.Text.StringBuilder
  foreach ($rel in $sorted) {
    [void]$sb.Append($rel).Append(' ').Append((Get-VtmFileHash (Join-Path $ui $rel))).Append("`n")
  }
  return (Get-VtmTextHash $sb.ToString())
}

function Get-UiBuildStampPath {
  param([string]$RepoRoot)
  return (Join-Path $RepoRoot "ui\dist\.vtm-source")
}

function Test-VtmStamp {
  param([string]$Path, [string]$Value)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
  $saved = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
  return ($null -ne $saved -and $saved.Trim() -eq $Value)
}

function Save-VtmStamp {
  param([string]$Path, [string]$Value)
  $dir = Split-Path -Parent $Path
  if (-not (Test-Path -LiteralPath $dir)) { return }
  [IO.File]::WriteAllText($Path, $Value)
}

# $true when ui\dist was built from the UI sources as they are now.
function Test-UiBuildCurrent {
  param([string]$RepoRoot)
  if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "ui\dist\index.html"))) { return $false }
  return (Test-VtmStamp (Get-UiBuildStampPath $RepoRoot) (Get-UiSourceHash $RepoRoot))
}

# The lockfile node_modules was installed from, kept inside node_modules so a
# deleted node_modules takes its stamp with it.
function Get-NpmStampPath {
  param([string]$UiDir)
  return (Join-Path $UiDir "node_modules\.vtm-lock")
}

function Test-NpmPackagesCurrent {
  param([string]$UiDir)
  if (-not (Test-Path -LiteralPath (Join-Path $UiDir "node_modules"))) { return $false }
  return (Test-VtmStamp (Get-NpmStampPath $UiDir) (Get-VtmFileHash (Join-Path $UiDir "package-lock.json")))
}

function Save-NpmStamp {
  param([string]$UiDir)
  Save-VtmStamp (Get-NpmStampPath $UiDir) (Get-VtmFileHash (Join-Path $UiDir "package-lock.json"))
}
