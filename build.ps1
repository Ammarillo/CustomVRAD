# Builds vrad_dll.dll (Release x64) safely.
#
# Why this script exists: an interrupted/partial MSBuild once linked stale object
# files against new headers (struct layout mismatch) and produced a vrad_dll.dll
# that corrupted memory and crashed mid-bake. This script prevents that class of
# failure by:
#   1. Defaulting to a FULL rebuild (/t:Rebuild) so no stale .obj can be linked.
#   2. Picking an MSBuild whose VS instance actually has the v143 C++ toolset.
#   3. Verifying afterwards that bin\vrad_dll.dll (and the game copy) really got
#      written by this build - a "successful" build with a missing/old DLL fails.
#
# Usage:
#   .\build.ps1                 full rebuild (recommended)
#   .\build.ps1 -Incremental    fast incremental build (only for quick iteration)

param(
    [switch]$Incremental
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$proj = Join-Path $root 'src\utils\vrad\vrad_dll_win64.vcxproj'
if ( -not ( Test-Path $proj ) ) { throw "Project not found: $proj" }

# --- Locate an MSBuild backed by a real v143 C++ toolset -----------------------
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if ( -not ( Test-Path $vswhere ) ) { throw "vswhere.exe not found - is Visual Studio installed?" }

$msbuild = $null
$installs = & $vswhere -all -products * -property installationPath
foreach ( $inst in $installs )
{
    $mb = Join-Path $inst 'MSBuild\Current\Bin\MSBuild.exe'
    $vc = Join-Path $inst 'VC\Tools\MSVC'
    if ( -not ( Test-Path $mb ) ) { continue }
    if ( -not ( Test-Path $vc ) ) { continue }
    # v143 toolset = MSVC 14.30+ compiler directories
    $has143 = Get-ChildItem $vc -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^14\.(3\d|4\d|5\d)' } | Select-Object -First 1
    if ( $has143 ) { $msbuild = $mb; break }
}
if ( -not $msbuild )
{
    throw "No Visual Studio instance with the v143 C++ toolset found. Install 'Desktop development with C++' (MSVC v143)."
}
Write-Host "MSBuild: $msbuild"

# --- Build ---------------------------------------------------------------------
$target = 'Rebuild'
if ( $Incremental ) { $target = 'Build' }
Write-Host "Target:  $target (Release x64)"
$buildStart = Get-Date

& $msbuild $proj /t:$target /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo
if ( $LASTEXITCODE -ne 0 )
{
    throw "MSBuild failed with exit code $LASTEXITCODE - vrad_dll.dll was NOT updated. Do not run vrad until this is fixed."
}

# --- Verify outputs are present and fresh ---------------------------------------
function Assert-FreshFile( [string]$path, [datetime]$notBefore, [bool]$required )
{
    if ( -not [System.IO.File]::Exists( $path ) )
    {
        if ( $required ) { throw "Build reported success but output is missing: $path" }
        Write-Warning "Optional output missing: $path"
        return
    }
    $t = [System.IO.File]::GetLastWriteTime( $path )
    # Skip freshness when notBefore is MinValue (incremental "don't care" sentinel).
    if ( $notBefore -gt [datetime]::MinValue -and $t -lt $notBefore.AddSeconds( -5 ) )
    {
        throw "Output is STALE (written $t, build started $notBefore): $path`nA stale DLL means mixed object files - delete src\utils\vrad\Release and rebuild."
    }
    Write-Host ("OK  {0}  ({1:yyyy-MM-dd HH:mm:ss}, {2:N0} bytes)" -f $path, $t, (Get-Item $path).Length)
}

# Incremental builds may legitimately skip the link if nothing changed.
$checkTime = $buildStart
if ( $Incremental ) { $checkTime = [datetime]::MinValue }

Assert-FreshFile ( Join-Path $root 'bin\vrad_dll.dll' )            $checkTime $true
Assert-FreshFile ( Join-Path $root 'game\bin\x64\vrad_dll.dll' )   $checkTime $false
Assert-FreshFile ( Join-Path $root 'src\utils\vrad\Release\x64\vrad_dll.pdb' ) $checkTime $false

Write-Host ''
Write-Host 'Build OK - vrad_dll.dll is consistent and up to date.' -ForegroundColor Green
