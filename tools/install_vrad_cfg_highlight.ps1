#Requires -Version 5.1
$repoRoot = Split-Path -Parent $PSScriptRoot
& python (Join-Path $repoRoot 'tools\install_vrad_cfg_highlight.py') @args
exit $LASTEXITCODE
