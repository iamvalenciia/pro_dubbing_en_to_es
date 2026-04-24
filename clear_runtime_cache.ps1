param(
    [string]$RepoRoot = (Get-Location).Path,
    [switch]$Deep
)

$ErrorActionPreference = "Stop"

function Remove-PathSafe {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Write-Host "[cache] removing: $Path"
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Clear-ChildrenSafe {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Write-Host "[cache] clearing children in: $Path"
        Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue | ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$tempWorkspace = Join-Path $repo "temp_workspace"
$pyTmp = Join-Path $repo "pyvideotrans\tmp"
$pyOutput = Join-Path $repo "pyvideotrans\output"

Write-Host "[cache] repo: $repo"

# Runtime caches and temporary artifacts (safe default)
Clear-ChildrenSafe -Path $tempWorkspace
Clear-ChildrenSafe -Path $pyTmp

# Remove transient outputs only (keep manually curated media in root output/)
if (Test-Path -LiteralPath $pyOutput) {
    Get-ChildItem -LiteralPath $pyOutput -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.Name.ToLowerInvariant()
        if ($name.StartsWith("test") -or $name.Contains("temp") -or $name.Contains("clip")) {
            Remove-PathSafe -Path $_.FullName
        }
    }
}

if ($Deep) {
    $hf = Join-Path $env:USERPROFILE ".cache\huggingface"
    $torch = Join-Path $env:USERPROFILE ".cache\torch"
    Write-Host "[cache] deep mode enabled"
    Clear-ChildrenSafe -Path $hf
    Clear-ChildrenSafe -Path $torch
}

Write-Host "[cache] done"