<#
.SYNOPSIS
    Distribute MODA Lancaster Heritage branding assets to the web and Flutter APK source trees.

.DESCRIPTION
    Run this script after changing any file in branding/logo/ to push the
    canonical assets to FastMODA (web) and APP (Flutter).  The script is
    idempotent — safe to run repeatedly.

.EXAMPLE
    .\branding\distribute.ps1
#>

[CmdletBinding()]
param()

$root   = Split-Path $PSScriptRoot -Parent   # repo root (parent of branding/)
$src    = "$PSScriptRoot\logo"

$webDst = "$root\FastMODA\static\images"
$appDst = "$root\APP\assets\images"

Write-Host "MODA Branding Distribute" -ForegroundColor Yellow
Write-Host "  Source : $src"
Write-Host "  Web    : $webDst"
Write-Host "  Flutter: $appDst"
Write-Host ""

function Copy-Asset {
    param([string]$From, [string]$To)
    if (-not (Test-Path $From)) {
        Write-Warning "Missing source: $From"
        return
    }
    $dir = Split-Path $To
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
    Copy-Item -Force $From $To
    Write-Host "  [OK] $(Split-Path $From -Leaf)  ->  $(Split-Path $To -Parent | Split-Path -Leaf)/$(Split-Path $To -Leaf)"
}

# ── Web (FastMODA / Flask static) ────────────────────────────────────────────
Write-Host "Web assets:" -ForegroundColor Cyan
Copy-Asset "$src\moda-logo.svg"   "$webDst\moda-logo.svg"
Copy-Asset "$src\moda-banner.svg" "$webDst\moda-banner.svg"
Copy-Asset "$src\wave-icon.svg"   "$webDst\wave-icon.svg"

# ── Flutter / APK ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Flutter assets:" -ForegroundColor Cyan
Copy-Asset "$src\moda_logo.png"   "$appDst\moda_logo.png"

# ── Android native icon foreground (wave-icon regenerated from SVG paths) ──────
# The Android vector drawable at
#   APP/android/app/src/main/res/drawable/ic_launcher_foreground.xml
# is the authoritative Android form of wave-icon.svg.  Both share the same
# path data; ic_launcher_foreground.xml is NOT overwritten here because it
# is already in sync.  To regenerate it from branding/logo/wave-icon.svg,
# use Android Studio's Vector Asset wizard or svg2vectordrawable.

Write-Host ""
Write-Host "Done." -ForegroundColor Green
