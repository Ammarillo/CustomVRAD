# Assemble a portable PathRAD runtime folder.
#
# Copies vrad.exe, vrad_dll.dll, Valve GMod DLLs, OIDN/TBB, CUDA runtime,
# fog shaders, configs, and FGDs into one directory Hammer can point at.
# Keep that folder off GMod's stock vrad/bin tree - use a separate path or
# its own subfolder so PathRAD DLLs do not overwrite GMod's.
#
# Usage:
#   .\bundle.ps1                  write .\bundle
#   .\bundle.ps1 -Build           rebuild first, then bundle
#   .\bundle.ps1 -OutDir C:\PathRAD
#   .\bundle.ps1 -GmodDir "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod"

param(
    [string]$OutDir = ( Join-Path $PSScriptRoot 'bundle' ),
    [string]$GmodDir,
    [switch]$Build,
    [switch]$Incremental
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Copy-FileTo( [string]$Src, [string]$Dst, [bool]$Required )
{
    if ( -not $Src -or -not ( Test-Path -LiteralPath $Src ) )
    {
        if ( $Required ) { throw "Missing required file: $Src" }
        return $false
    }
    $dir = Split-Path -Parent $Dst
    if ( $dir -and -not ( Test-Path -LiteralPath $dir ) )
    {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
    Write-Host ("  {0}" -f $Dst)
    return $true
}

function Find-SteamLibraries
{
    $steamRoots = New-Object System.Collections.Generic.List[string]
    foreach ( $reg in @(
        'HKCU:\Software\Valve\Steam',
        'HKLM:\SOFTWARE\WOW6432Node\Valve\Steam',
        'HKLM:\SOFTWARE\Valve\Steam'
    ) )
    {
        if ( -not ( Test-Path $reg ) ) { continue }
        $p = Get-ItemProperty $reg -ErrorAction SilentlyContinue
        foreach ( $name in @( 'SteamPath', 'InstallPath' ) )
        {
            $v = $p.$name
            if ( $v ) { [void]$steamRoots.Add( $v.TrimEnd( '\', '/' ) ) }
        }
    }
    foreach ( $fallback in @(
        'C:\Program Files (x86)\Steam',
        'C:\Program Files\Steam'
    ) )
    {
        [void]$steamRoots.Add( $fallback )
    }

    $libs = New-Object System.Collections.Generic.List[string]
    foreach ( $steam in $steamRoots )
    {
        if ( -not $steam ) { continue }
        [void]$libs.Add( $steam )
        $vdf = Join-Path $steam 'steamapps\libraryfolders.vdf'
        if ( -not ( Test-Path -LiteralPath $vdf ) ) { continue }
        $text = Get-Content -LiteralPath $vdf -Raw -ErrorAction SilentlyContinue
        if ( -not $text ) { continue }
        [regex]::Matches( $text, '"path"\s+"([^"]+)"' ) | ForEach-Object {
            $lib = $_.Groups[1].Value -replace '\\\\', '\'
            if ( $lib ) { [void]$libs.Add( $lib.TrimEnd( '\', '/' ) ) }
        }
    }
    $libs | Select-Object -Unique
}

function Find-GmodDir
{
    if ( $script:GmodDir )
    {
        if ( -not ( Test-Path -LiteralPath $script:GmodDir ) )
        {
            throw "GmodDir not found: $($script:GmodDir)"
        }
        return ( Resolve-Path -LiteralPath $script:GmodDir ).Path
    }
    foreach ( $envName in @( 'GMOD_DIR', 'GMODDIR' ) )
    {
        $v = [Environment]::GetEnvironmentVariable( $envName )
        if ( $v -and ( Test-Path -LiteralPath $v ) )
        {
            return ( Resolve-Path -LiteralPath $v ).Path
        }
    }
    foreach ( $steam in ( Find-SteamLibraries ) )
    {
        $candidate = Join-Path $steam 'steamapps\common\GarrysMod'
        $probe = Join-Path $candidate 'bin\win64\tier0.dll'
        if ( Test-Path -LiteralPath $probe )
        {
            return ( Resolve-Path -LiteralPath $candidate ).Path
        }
    }
    return $null
}

function Find-CudaRuntime
{
    $roots = New-Object System.Collections.Generic.List[string]
    foreach ( $envName in @( 'CUDA_PATH', 'CUDA_PATH_V12_6', 'CUDA_PATH_V12_5', 'CUDA_PATH_V12_4' ) )
    {
        $v = [Environment]::GetEnvironmentVariable( $envName )
        if ( $v ) { [void]$roots.Add( $v ) }
    }
    $toolkit = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if ( Test-Path -LiteralPath $toolkit )
    {
        Get-ChildItem -LiteralPath $toolkit -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { [void]$roots.Add( $_.FullName ) }
    }
    foreach ( $rootCuda in $roots )
    {
        $dll = Get-ChildItem -LiteralPath ( Join-Path $rootCuda 'bin' ) -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ( $dll ) { return $dll.FullName }
    }
    return $null
}

# --- Optional rebuild ----------------------------------------------------------
if ( $Build )
{
    $buildScript = Join-Path $root 'build.ps1'
    if ( -not ( Test-Path -LiteralPath $buildScript ) ) { throw "build.ps1 not found: $buildScript" }
    if ( $Incremental )
    {
        & $buildScript -Incremental
    }
    else
    {
        & $buildScript
    }
    if ( $LASTEXITCODE -ne 0 ) { throw "build.ps1 failed with exit code $LASTEXITCODE" }
}

$vradExe = Join-Path $root 'bin\vrad.exe'
$vradDll = Join-Path $root 'bin\vrad_dll.dll'
if ( -not ( Test-Path -LiteralPath $vradExe ) -or -not ( Test-Path -LiteralPath $vradDll ) )
{
    throw "bin\vrad.exe / bin\vrad_dll.dll missing. Run .\build.ps1 first, or pass -Build."
}

$outFull = [System.IO.Path]::GetFullPath( $OutDir )
$rootFull = [System.IO.Path]::GetFullPath( $root )
if ( $outFull.TrimEnd( '\' ) -ieq $rootFull.TrimEnd( '\' ) )
{
    throw "OutDir cannot be the repo root."
}
foreach ( $banned in @( 'bin', 'src', 'shaders', 'configs', 'fgd' ) )
{
    if ( $outFull.TrimEnd( '\' ) -ieq ( Join-Path $rootFull $banned ) )
    {
        throw "OutDir cannot be the repo $banned folder."
    }
}

$gmod = Find-GmodDir
$gmodWin64 = $null
if ( $gmod )
{
    $gmodWin64 = Join-Path $gmod 'bin\win64'
    Write-Host "GMod:    $gmod"
}
else
{
    Write-Warning "Garry's Mod not found. Will use Valve DLLs already in repo bin\ if present. Pass -GmodDir to set the install."
}

Write-Host "OutDir:  $outFull"
if ( Test-Path -LiteralPath $outFull )
{
    Remove-Item -LiteralPath $outFull -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $outFull | Out-Null

Write-Host ''
Write-Host 'PathRAD binaries'
Copy-FileTo $vradExe ( Join-Path $outFull 'vrad.exe' ) $true | Out-Null
Copy-FileTo $vradDll ( Join-Path $outFull 'vrad_dll.dll' ) $true | Out-Null

Write-Host ''
Write-Host 'Valve runtime DLLs'
# Prefer the copies already next to this build (repo bin\). Live GMod DLLs can
# be newer than the SDK libs vrad_dll was linked against and then fail to load.
$repoBin = Join-Path $root 'bin'
$valveMap = @(
    @{ Dest = 'tier0.dll';                    Repo = 'tier0.dll';                    Gmod = 'tier0.dll' },
    @{ Dest = 'vstdlib.dll';                  Repo = 'vstdlib.dll';                  Gmod = 'vstdlib.dll' },
    @{ Dest = 'vphysics.dll';                 Repo = 'vphysics.dll';                 Gmod = 'vphysics_stub.dll' },
    @{ Dest = 'filesystem_stdio.dll';         Repo = 'filesystem_stdio.dll';         Gmod = 'filesystem_stdio.dll' },
    @{ Dest = 'bin\x64\filesystem_stdio.dll'; Repo = 'bin\x64\filesystem_stdio.dll'; Gmod = 'filesystem_stdio.dll' },
    @{ Dest = 'bin\x64\vphysics.dll';         Repo = 'bin\x64\vphysics.dll';         Gmod = 'vphysics_stub.dll' }
)
$usedGmodValve = $false
foreach ( $item in $valveMap )
{
    $src = $null
    $fromRepo = Join-Path $repoBin $item.Repo
    if ( Test-Path -LiteralPath $fromRepo )
    {
        $src = $fromRepo
    }
    elseif ( $item.Gmod -eq 'vphysics_stub.dll' )
    {
        $fromRepoStub = Join-Path $repoBin 'vphysics_stub.dll'
        if ( Test-Path -LiteralPath $fromRepoStub ) { $src = $fromRepoStub }
    }
    if ( -not $src -and $gmodWin64 )
    {
        $fromGmod = Join-Path $gmodWin64 $item.Gmod
        if ( Test-Path -LiteralPath $fromGmod )
        {
            $src = $fromGmod
            $usedGmodValve = $true
        }
    }
    if ( -not $src )
    {
        throw "Could not find $($item.Dest). Copy Valve DLLs into bin\ (see README) or pass -GmodDir."
    }
    Copy-FileTo $src ( Join-Path $outFull $item.Dest ) $true | Out-Null
}
if ( $usedGmodValve )
{
    Write-Warning "Some Valve DLLs came from GMod rather than repo bin\. If vrad fails to start, copy the README DLL set into bin\ and re-run this script."
}

Write-Host ''
Write-Host 'OIDN / TBB'
$oidnCopied = 0
$oidnDirs = @(
    ( Join-Path $root 'bin' ),
    ( Join-Path $root 'thirdparty\oidn\bin' )
)
$oidnNames = @(
    'OpenImageDenoise.dll',
    'OpenImageDenoise_core.dll',
    'OpenImageDenoise_device_cpu.dll',
    'OpenImageDenoise_device_cuda.dll',
    'tbb12.dll'
)
foreach ( $name in $oidnNames )
{
    $found = $null
    foreach ( $dir in $oidnDirs )
    {
        $p = Join-Path $dir $name
        if ( Test-Path -LiteralPath $p ) { $found = $p; break }
    }
    if ( $found -and ( Copy-FileTo $found ( Join-Path $outFull $name ) $false ) ) { $oidnCopied++ }
}
$tbbBindCopied = $false
foreach ( $dir in $oidnDirs )
{
    if ( -not ( Test-Path -LiteralPath $dir ) ) { continue }
    Get-ChildItem -LiteralPath $dir -Filter 'tbbbind*.dll' -ErrorAction SilentlyContinue | ForEach-Object {
        if ( Copy-FileTo $_.FullName ( Join-Path $outFull $_.Name ) $false )
        {
            $oidnCopied++
            $tbbBindCopied = $true
        }
    }
    if ( $tbbBindCopied ) { break }
}
if ( $oidnCopied -eq 0 )
{
    Write-Warning "OIDN DLLs not found (thirdparty\oidn or bin\). -pt_denoiser oidn will be unavailable."
}

Write-Host ''
Write-Host 'CUDA runtime (linked by vrad_dll)'
$cudart = Find-CudaRuntime
$repoCudart = Get-ChildItem -LiteralPath $repoBin -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
if ( -not $cudart -and $repoCudart ) { $cudart = $repoCudart.FullName }
if ( $cudart )
{
    Copy-FileTo $cudart ( Join-Path $outFull ( Split-Path -Leaf $cudart ) ) $true | Out-Null
}
else
{
    Write-Warning "cudart64_*.dll not found. vrad_dll.dll may fail to load without the CUDA runtime on PATH. Install the CUDA toolkit or copy cudart64_*.dll next to vrad.exe."
}

Write-Host ''
Write-Host 'Shaders'
$shaderSrc = Join-Path $root 'shaders\fxc'
$shaderDst = Join-Path $outFull 'shaders\fxc'
$shaderCount = 0
Get-ChildItem -LiteralPath $shaderSrc -Filter '*.vcs' -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-FileTo $_.FullName ( Join-Path $shaderDst $_.Name ) $true | Out-Null
    $shaderCount++
}
if ( $shaderCount -eq 0 )
{
    throw "No compiled shaders in shaders\fxc\*.vcs. Run .\shaders\build_shaders.ps1"
}

Write-Host ''
Write-Host 'Configs'
$configDst = Join-Path $outFull 'configs'
Get-ChildItem -LiteralPath ( Join-Path $root 'configs' ) -Filter '*.cfg' | ForEach-Object {
    Copy-FileTo $_.FullName ( Join-Path $configDst $_.Name ) $true | Out-Null
}

Write-Host ''
Write-Host 'FGD'
$fgdDst = Join-Path $outFull 'fgd'
Copy-FileTo ( Join-Path $root 'fgd\pathrad.fgd' ) ( Join-Path $fgdDst 'pathrad.fgd' ) $true | Out-Null
Copy-FileTo ( Join-Path $root 'fgd\customvrad.fgd' ) ( Join-Path $fgdDst 'customvrad.fgd' ) $true | Out-Null

$readme = @"
PathRAD runtime bundle
======================

Install
-------
Do not copy these files into Garry's Mod's own VRAD folder
(GarrysMod\bin, bin\win64, or wherever stock vrad.exe lives).
PathRAD's DLLs will clash with GMod's and both tools can break.

Put this bundle somewhere else on the machine, or in its own
subfolder (for example C:\PathRAD or GarrysMod\pathrad\).
Point Hammer's RAD executable at THAT copy of vrad.exe.

Hammer Expert / RAD executable:
  C:\PathRAD\vrad.exe

Parameters (keep -game and the map path on this line, not in the cfg):
  -config "C:\PathRAD\configs\full" -game `$gamedir `$path\`$file

Or, because configs live next to vrad.exe:
  -config full -game `$gamedir `$path\`$file

FGD (Tools -> Options -> Game Data Files, last in the list, then restart Hammer):
  C:\PathRAD\fgd\pathrad.fgd

Do not use GMod's real vphysics.dll next to this tool - the bundle already has the stub renamed to vphysics.dll.
"@
Set-Content -LiteralPath ( Join-Path $outFull 'README.txt' ) -Value $readme -Encoding ASCII

Write-Host ''
Write-Host "Bundle OK -> $outFull" -ForegroundColor Green
Write-Host ''
Write-Host 'Hammer executable:'
Write-Host "  $outFull\vrad.exe"
Write-Host 'Hammer parameters:'
Write-Host "  -config `"$outFull\configs\full`" -game `$gamedir `$path\`$file"
Write-Host 'FGD:'
Write-Host "  $outFull\fgd\pathrad.fgd"
Write-Host ''
Write-Host 'Do not copy this bundle into GMod bin / stock vrad. Keep it in its own folder.' -ForegroundColor Yellow
