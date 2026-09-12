# Build vrad.exe + vrad_dll.dll (Release x64).
#
# A half-finished MSBuild once linked old .obj files against new headers and the
# DLL crashed mid-bake. This script avoids that by:
#   1. Doing a full rebuild by default (/t:Rebuild) so leftover .obj files cannot
#      be linked.
#   2. Picking an MSBuild whose Visual Studio install actually has the v143 C++
#      toolset.
#   3. Checking that bin\vrad.exe and bin\vrad_dll.dll were written by this run.
#      A "successful" build that left an old DLL behind still fails here.
#
# Usage:
#   .\build.ps1                 full rebuild (use this)
#   .\build.ps1 -Incremental    faster; only for local iteration

param(
    [switch]$Incremental
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$projDll = Join-Path $root 'src\utils\vrad\vrad_dll_win64.vcxproj'
$projExe = Join-Path $root 'src\utils\vrad_launcher\vrad_launcher_win64.vcxproj'
if ( -not ( Test-Path $projDll ) ) { throw "Project not found: $projDll" }
if ( -not ( Test-Path $projExe ) ) { throw "Project not found: $projExe" }

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

function Invoke-VradMsBuild( [string]$project )
{
    Write-Host "Building $project"
    & $msbuild $project /t:$target /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo
    if ( $LASTEXITCODE -ne 0 )
    {
        throw "MSBuild failed with exit code $LASTEXITCODE for $project. Do not run vrad until this is fixed."
    }
}

Invoke-VradMsBuild $projDll
Invoke-VradMsBuild $projExe

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

Assert-FreshFile ( Join-Path $root 'bin\vrad.exe' )                $checkTime $true
Assert-FreshFile ( Join-Path $root 'bin\vrad_dll.dll' )            $checkTime $true
Assert-FreshFile ( Join-Path $root 'game\bin\x64\vrad_dll.dll' )   $checkTime $false
Assert-FreshFile ( Join-Path $root 'src\utils\vrad\Release\x64\vrad_dll.pdb' ) $checkTime $false

Write-Host ''
Write-Host 'Build OK - vrad.exe and vrad_dll.dll are up to date.' -ForegroundColor Green
