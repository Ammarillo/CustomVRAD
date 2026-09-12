# Fog shaders

These are the compiled Valve shaders for GMod `screenspace_general`.

Build them with:

```powershell
.\shaders\build_shaders.ps1
```

You need `ShaderCompile.exe` (see the script if it isn't on PATH).

Required files in this folder:

- `pathrad_fog_vs30.vcs` (fullscreen VS)
- `pathrad_fog_ps30.vcs` (fog)

VRAD packs this pair into the BSP when the map has a `fog_volume`.
