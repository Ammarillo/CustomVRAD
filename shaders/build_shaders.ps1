# CustomVRAD shaders (fog_volume)
#
# Compile HLSL to GMod .vcs via SCell555 ShaderCompile:
#   https://github.com/SCell555/ShaderCompile
#
# Output in shaders/fxc/:
#   cvrad_fog_ps30.vcs / cvrad_fog_vs30.vcs
# VRAD packs these into the BSP when fog_volume entities are present.

param(
    [string]$ShaderCompile = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $OutDir) { $OutDir = Join-Path $Root "fxc" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Find-ShaderCompile {
    param([string]$Hint)
    if ($Hint -and (Test-Path $Hint)) { return (Resolve-Path $Hint).Path }

    $candidates = @(
        "$Root\tools\ShaderCompile.exe",
        "$Root\tools\ShaderCompile\build\Release\ShaderCompile.exe",
        "$Root\..\bin\ShaderCompile.exe",
        "$env:GMOD_DIR\bin\ShaderCompile.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path }
    }

    $cmd = Get-Command ShaderCompile.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

$compiler = Find-ShaderCompile -Hint $ShaderCompile
if (-not $compiler) {
    Write-Host @"
ERROR: ShaderCompile.exe not found.

Clone/build https://github.com/SCell555/ShaderCompile under shaders/tools/ShaderCompile
or pass -ShaderCompile C:\path\to\ShaderCompile.exe
"@
    exit 1
}

Write-Host "Using: $compiler"
Write-Host "Output: $OutDir"

$shaders = @(
    "cvrad_fog_vs30.hlsl",
    "cvrad_fog_ps30.hlsl"
)

Push-Location $Root
try {
    foreach ($src in $shaders) {
        Write-Host "Compiling $src ..."
        & $compiler /O 3 -ver 30 -shaderpath $Root ".\$src"
        if ($LASTEXITCODE -ne 0) {
            throw "ShaderCompile failed for $src (exit $LASTEXITCODE)"
        }
    }
} finally {
    Pop-Location
}

# SCell555 writes to <shaderpath>/shaders/fxc/
$nested = Join-Path $Root "shaders\fxc"
if (Test-Path $nested) {
    Copy-Item -Force (Join-Path $nested "*.vcs") $OutDir
    Remove-Item -Recurse -Force (Join-Path $Root "shaders")
}

foreach ($name in @(
    "cvrad_fog_vs30.vcs", "cvrad_fog_ps30.vcs"
)) {
    $path = Join-Path $OutDir $name
    if (-not (Test-Path $path)) { throw "Missing output: $path" }
    Write-Host ("OK  {0}  ({1:N0} bytes)" -f $path, (Get-Item $path).Length)
}

Write-Host "Done. VRAD will pack files from: $OutDir"
