# PathRAD

PathRAD is a 64-bit VRAD for Garry's Mod. Point Hammer at this `vrad.exe` instead of Valve's. The lightmaps it writes are still the ones the engine already knows how to load.

You can bake the usual way (radiosity, optional OpenCL), or turn on `-pathtrace` and light the world on the GPU. The rest of this repo is the extras: lighting volumes, Kelvin color temperature on lights, IES / projected spots, colored glass, underwater light, fog stored in the BSP, and `bake_volume` so you can relight one room without compiling the whole map again.

Windows only. Load the map in GMod and look at it before you ship — path tracing in particular can surprise you.

| | |
| --- | --- |
| Repo | [https://github.com/Ammarillo/PathRAD](https://github.com/Ammarillo/PathRAD) |
| Hammer entities | [fgd/pathrad.fgd](fgd/pathrad.fgd) |
| Math / algorithms | [docs/algorithms.tex](docs/algorithms.tex) → [docs/algorithms.pdf](docs/algorithms.pdf) |
| Flag list | [configs/full.cfg](configs/full.cfg) |

---

## Install

Do this once. After that, Hammer just calls this `vrad.exe` instead of Valve's.

### 1. What you need

- Windows 10 or 11
- [Visual Studio 2022](https://visualstudio.microsoft.com/) with **Desktop development with C++** (MSVC v143, Windows 10/11 SDK)
- Garry's Mod installed (you copy a few of its DLLs next to VRAD)
- A GPU that can run DirectX Raytracing if you want `-pathtrace` (most DX12 cards from the last several years)

OpenCL (`-gpu`) comes with your GPU driver. You don't install a separate SDK for that.

### 2. Get the source

```powershell
git clone https://github.com/Ammarillo/PathRAD.git
cd PathRAD
```

### 3. Copy Valve DLLs next to the compiler

VRAD is a Source tool. It needs Valve's runtime DLLs beside `vrad.exe`. Copy these from your GMod install.

Typical GMod folder:

`C:\Program Files (x86)\Steam\steamapps\common\GarrysMod`

Create `bin\` in the PathRAD repo (the build will also write here) and copy:

| Copy from GMod | Put it here | Notes |
| --- | --- | --- |
| `bin\win64\tier0.dll` | `PathRAD\bin\tier0.dll` | |
| `bin\win64\vstdlib.dll` | `PathRAD\bin\vstdlib.dll` | |
| `bin\win64\vphysics_stub.dll` | `PathRAD\bin\vphysics.dll` | **Rename** stub → `vphysics.dll` |
| `bin\win64\filesystem_stdio.dll` | `PathRAD\bin\bin\x64\filesystem_stdio.dll` | Keep the extra `bin\x64` folder |
| same `vphysics_stub.dll` | `PathRAD\bin\bin\x64\vphysics.dll` | Same rename, second copy |

Do **not** use GMod's real `bin\win64\vphysics.dll`. That one is built against a different `tier0` and will crash the tool.

You should end up with:

```
PathRAD\bin\vrad.exe          (after the build)
PathRAD\bin\vrad_dll.dll      (after the build)
PathRAD\bin\tier0.dll
PathRAD\bin\vstdlib.dll
PathRAD\bin\vphysics.dll
PathRAD\bin\bin\x64\filesystem_stdio.dll
PathRAD\bin\bin\x64\vphysics.dll
```

If you use OIDN denoise (`-pt_denoiser oidn`), the build copies `OpenImageDenoise*.dll` and TBB next to `vrad_dll.dll` when those files exist under `thirdparty\oidn\`.

### 4. Build

From the repo root:

```powershell
.\build.ps1
```

That does a full rebuild of `vrad.exe` and `vrad_dll.dll` (Release x64) and checks that `bin\vrad_dll.dll` is actually new. Use `.\build.ps1 -Incremental` only when you're iterating on code.

If MSBuild says it can't find the v143 toolset, open Visual Studio Installer and add **Desktop development with C++**.

### 5. Point Hammer at this VRAD

In Hammer (or Hammer++):

1. Open **Expert** compile (or your custom compile setup).
2. Find the VRAD / RAD step.
3. Set the executable to your new `vrad.exe`, for example:

```
C:\PathRAD\vrad.exe
```

4. Set the parameters to something like:

```
-config "C:\PathRAD\configs\full" -game $gamedir $path\$file
```

Use **your** install path. Keep `-game` and the map path on this line — never put them inside the `.cfg` file.

Stock flags (`-hdr`, `-final`, `-StaticPropLighting`, …) still work. Running `vrad.exe` with no arguments still prints Valve's help.

### 6. Add the FGD

1. Hammer → **Tools → Options → Game Configurations → Game Data Files**
2. Add `C:\PathRAD\fgd\pathrad.fgd` (this must be the **last** FGD, after `garrysmod.fgd`)
3. If the list still has `fgd\customvrad.fgd`, leave it or replace it — that file only `@include`s `pathrad.fgd`
4. **Restart Hammer** (FGDs are not reloaded while it is open)

You should now see entities like `light_env_vol`, `fog_volume`, `bake_volume`, and extra keys on lights (IES, color temperature, …). If Smart Edit is empty or only shows raw key names, the FGD is missing or not last in the list.

### 7. Compile a map once

1. Save the VMF, run **VBSP** as usual (stock VBSP is fine).
2. Run **VIS** as usual.
3. Run this VRAD with the line from step 5.
4. Load the BSP in GMod and look at it.

First bakes are slow if `-pathtrace` is on and `-pt_samples` is high. `configs\full.cfg` is a full-quality sheet — comment flags out (put `#` in front) if you want a faster preview.

If the map is huge and HDR lightmaps make clients hitch on load, put `r_hunkalloclightmaps 0` in the client/server cfg.

---

## Day-to-day use

### Config file

`configs\full.cfg` is the flag sheet. One flag per line. `#` or `//` starts a comment. Uncomment a line to turn it on.

```bat
vrad.exe -config "C:\PathRAD\configs\full" -game "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod" "C:\maps\map"
```

CLI args after `-config` override the file. VRAD looks for the file as given, then with `.cfg` / `.txt`, then next to `vrad.exe` and in `vrad.exe\configs\`.

Check a config:

```powershell
python tools/lint_vrad_cfg.py --style configs
```

Syntax colors in Cursor / VS Code:

```powershell
python tools/install_vrad_cfg_highlight.py
```

Then **Developer: Reload Window**.

### Two lighting modes (pick one for world faces)

| Mode | How you get it | What it does |
| --- | --- | --- |
| Classic radiosity | Default, or if path tracing fails | Stock `BuildFacelights` + bounce. Optional OpenCL with `-gpu`. |
| Path tracing | `-pathtrace` or `-dxr` | Replaces world-face direct light and bounce. Needs a DXR GPU. |

Only one of those owns world-face lighting. Static props can use the path tracer too if you pass `-StaticPropLighting`.

`-gpu` (OpenCL bounce) and `-pt_gpu` (DXR path tracer) are different backends. You can have both on the command line; the path tracer wins for world faces if it starts.

### Physical vs “make it look good”

Some controls try to match real light (linear color, IES files, water extinction). Others are just knobs: `-bounce_boost`, `-bounce_chroma`, `-ao_force`, bounce-volume tint. The knobs are marked as such below.

### Example commands

Config (normal Hammer line):

```bat
C:\PathRAD\vrad.exe -config "C:\PathRAD\configs\full" -game $gamedir $path\$file
```

Classic radiosity, no path tracer:

```bat
vrad.exe -hdr -final -StaticPropLighting -textureshadows -texbounce -ao -ao_samples 32 -gpu -game "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod" "C:\maps\map"
```

Path trace:

```bat
vrad.exe -hdr -pathtrace -pt_samples 256 -pt_bounces 6 -pt_aa 2 -StaticPropLighting -textureshadows -game "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod" "C:\maps\map"
```

Underwater attenuation:

```bat
vrad.exe -config full -underwater -game "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod" "C:\maps\map"
```

---

## What you can do

| Feature | Kind | Short version |
| --- | --- | --- |
| `light_env_vol` | Brush | Local sun / sky / ambient. Bake only — not exported as a second sun. |
| Leaf-cube volume sun | Bake | Ambient cubes pick up volume-blended sun when they see the sky. |
| `light_ao` / `light_ao_vol` | Point / brush | Hemisphere AO. Skipped under path-trace GI unless `-ao_force`. |
| `light_absorb` | Brush | Darkens bounce (optional direct). |
| `light_bounce_vol` | Brush | Local bounce boost / chroma / energy. |
| `light_volume` | Point | Soft-sphere origin. Engine still gets a hard point light at the center. |
| `fog_volume` (+ blocker) | Brush | 3D light grid + screenspace fog packed into the BSP. |
| `water_light_vol` | Brush | Override underwater optics (`-underwater` / `-uw_volume`). |
| `bake_volume` | Brush | Rebake only samples inside the brush; keep the rest from last compile. |
| `light_spot` IES / projected VTF | Keys | Bake-only beam shape / gobo. Engine cone keys stay for dynamic light. |
| `LightTemperature` / `AmbientTemperature` | Key | Kelvin on `light`, `light_spot`, `light_volume`, `light_environment`, and `light_env_vol`. Replaces RGB, keeps brightness. `0` = use the color pickers. |
| Soft sun | Bake | Cone of sun rays instead of a hard disk. |
| `-pathtrace` | CLI | DXR path tracer (GPU RayQuery by default). |
| `-pt_spectral` | CLI | Hero-wavelength color on GPU (default). `-pt_nospectral` = RGB. |
| `-underwater` / `-uw_volume` | CLI | Water extinction, or full volume path tracing in water. |
| `$vrad_emit*` | VMT | Textured emission. Path tracer treats it as an area light. |
| `$vrad_filter*` | VMT | Colored glass (path tracer only). |
| `-gpu` | CLI | OpenCL bounce gather (classic path). |
| `-config` | CLI | Load flags from a text file. |
| Seam stitch / edge pull | Bake | On by default. Stitch split faces; pull edge samples inward. |

Threading: auto-detects cores (including more than 64). Cap is **256** workers.

---

## Lighting volumes

Brush volumes share the same membership rules: a soft distance to the brush, a quintic fade (`BlendDistance` / `BlendMode`), priority (higher wins; same priority → smaller brush), and a soft split so two volumes don't paint each other's outsides. Sun shadow filters on env volumes use the **hard** box, not the fade shell.

Place them like a trigger: `tools/toolstrigger` or nodraw, then **Tie to Entity**.

### `light_env_vol`

Overrides sun / sky / ambient **inside** the brush. Outside (and in the fade) it mixes with the map `light_environment`. Keep one global `light_environment` as the default.

Old classname `light_environment_volume` still works.


| Key | Default | What it does |
| --- | --- | --- |
| `_light` / `_ambient` | (stock) | Sun and ambient |
| `LightTemperature` / `AmbientTemperature` | `0` | Kelvin; `0` = use RGB. Keeps the 4th brightness number |
| `_lightHDR` / `_ambientHDR` | `-1 -1 -1 1` | HDR override; default copies SDR |
| `_lightscaleHDR` / `_AmbientScaleHDR` | `1` | HDR scales |
| `pitch` | `0` | Overrides pitch in Angles |
| `SunSpreadAngle` | `0` | Soft sun cone, degrees |
| `BlendDistance` | `0` | Fade width (`0` = hard cut) |
| `BlendMode` | `2` Center | `0` Inside · `1` Outside · `2` Center |
| `priority` | `0` | Overlap |
| `BounceVolColor` / `BounceVolBright` | No | Recolor / rescale bounce to this volume's `_light` |
| `OutsideCastShadowIn` / `InsideCastShadowOut` | Yes | Whether geo on one side shadows the other |

Old keys `OutsideCastShadow` / `InsideCastShadow` still work (you'll get a warning).

The engine still only has **one** exported `light_environment` for dynamic lighting. Leaf cubes try to match the volume look for ambient-lit dynamics.

### `light_absorb`


| Key | Default | What it does |
| --- | --- | --- |
| `Strength` | `0.5` | 0 = off, 1 = full |
| `AbsorbDirect` | No | Also darken direct light at samples |
| `AbsorbBounce` | Yes | Darken bounce |
| Blend / priority | (shared) | Same as other volumes |

### `light_bounce_vol`

Local versions of `-bounce_boost` / `-bounce_chroma` / `-energy`, faded with the CLI values.


| Key | Default | What it does |
| --- | --- | --- |
| `BounceBoost` | `-1` (inherit) | Bounce scale, 0–16 |
| `BounceChroma` | `-1` (inherit) | Early-bounce saturation, 0–8 |
| `EnergyMode` | Inherit | Cavity damp on/off |

Meant for use with `-texbounce`. Path tracing ignores `-bounce_chroma`. Boost can still change albedo/throughput through volumes.

### `bake_volume` — rebake one room

If **any** enabled `bake_volume` exists, VRAD only recomputes samples whose world position is inside those brushes (AABB plus `Padding`). Faces and props that stick out of the volume keep lighting **outside** it from the last good bake (`mapsrc/<map>.vradlm`, written after every compile).

Geometry outside still sits in the ray scene (it still shadows and bounces into the volume). Its own luxels / vertices are not recalculated.

Typical flow: full-bake once (no volumes, or `-nobakevolume`), then add volumes for room-sized updates. Faces match by origFace / displacement start + lightmap size; props match by origin + model. Displacements use the sculpted surface box for the coarse overlap test; luxels still use the displaced sample position. Anything with no cache hit is baked so it can't go black.

A prop that straddles the volume only rebakes **vertices (and prop lightmap texels) inside the box**. Same for world lightmaps: only inside luxels are recomputed, then cached luxels outside are copied back.


| Key | Default | What it does |
| --- | --- | --- |
| `Enabled` | Yes | `0` ignore this brush |
| `Rebake` | Both | `0` Both · `1` Lightmaps only · `2` Props only |
| `Padding` | `8` | Extra world units on the box |

`-nobakevolume` ignores the entities and rebakes the whole map (still updates the cache). Several volumes add together. The trigger faces themselves are `TEX_SPECIAL` and are not lit.

### `light_volume` — soft sphere point light

Same keys as stock `light`, plus `Radius` (default **16**) and optional `LightTemperature`. Bake samples the origin inside the sphere so you get a soft penumbra. Sample count scales with radius (about 16 at default, cap 40; fewer with `-fast`). The engine still gets a **hard** point light at the center.

### `fog_volume` — fog in the BSP

Bakes a 3D lighting grid into the BSP pak and packs a GMod screenspace fog draw. No addon. The server must allow map `lua_run` / client Lua.

1. Brush → Tie to Entity → `fog_volume` (`tools/toolstrigger`). Vertex-edited / clipped brushes work (convex). They don't have to be axis-aligned boxes.
2. Compile the fog shaders once (below).
3. VRAD writes `materials/maps/<map>/fog_volume_<hammerid>.vtf/.vmt`, a shared `fog_noise.vtf`, packs `shaders/fxc/pathrad_fog_*.vcs`, and rewrites the entity into two `lua_run`s (setup + draw).


| Key | Default | What it does |
| --- | --- | --- |
| `Enabled` / `StartEnabled` | Yes | Skip bake / runtime `cvf_en` |
| `FogColor` | `200 220 255` | Tint |
| `Density` | `0.02` | Optical density per world unit |
| `LightBoost` | `1.0` | Direct-light multiplier in fog (0–16) |
| `Anisotropy` | `0` | Phase, −1 to 1 |
| `GridSpacing` | `32` | Grid step; at most **64³** samples |
| `StepCount` | `32` | Raymarch steps (8–128) |
| `BlendDistance` / `BlendMode` | `0` / Center | Soft edge; Outside/Center grow bake + march |
| `NoiseScale` / `NoiseCoverage` | `0` / `0.45` | FBM noise (`0` = off) |
| `WindSpeed` / `WindDir` | `8` / `0` | World-space scroll; yaw in 45° steps |

Live tweak (no rebake): `cvf_en`, `cvf_de`, `cvf_ns`, `cvf_cov`, `cvf_wspd`, `cvf_wyaw`, `cvf_bd`. FogColor and lighting still need a rebake.

`fog_volume_blocker`: carves fog out of overlapping fog (atlas alpha). Rewritten to `info_null`.

#### Fog shaders

```powershell
.\shaders\build_shaders.ps1 -ShaderCompile C:\path\to\ShaderCompile.exe
```

Output: `shaders/fxc/pathrad_fog_ps30.vcs`, `pathrad_fog_vs30.vcs`. Clients/servers that block map client Lua will not draw fog.

---

## Underwater lighting

VRAD finds BSP water from `leafWaterData` / `CONTENTS_WATER`. Optics come from the water surface VMT (`$fogenable`, `$fogcolor`, `$fogend`) or `$vrad_uw_*` / Jerlov presets. `water_light_vol` overrides overlapping water.


| Flag | Mode | What it does |
| --- | --- | --- |
| `-underwater` | A | Beer-Lambert along rays + Kd downwelling for sun/sky |
| `-uw_volume` | B | Homogeneous volume path tracing in water (implies underwater) |
| `-uw_off` | — | Force off |
| `-uw_maxscatter N` | B | Cap medium bounces (default **8**, max **64**) |

Worth knowing:

- Source `$fogend` is screenspace fog, not lightmap optical depth. The bake scales extinction (`s_bake = 0.28`), uses scatter fraction 0.45, and `k_d = 0.35 σ_t`, with a per-channel Kd floor of 0.04.
- Mode A uses **either** segment transmittance (local lights / path segments) **or** Kd (sky/sun) — not both stacked.
- Mode B only scatters **inside** the water box. Soft sky fill covers the rest.
- Optional VMT: `$vrad_uw_jerlov`, `$vrad_uw_sigma_a`, `$vrad_uw_sigma_s`, `$vrad_uw_g`.

CPU and GPU path tracing both implement A and B. Mode A also scales classic direct gathering when path tracing is off. Details: algorithms report, underwater section.

---

## Ambient occlusion

Hemisphere AO in `FinalLightFace`.

With `-pathtrace` and `-pt_bounces` greater than 0, this multiply is **skipped** (the path tracer already sees occlusion). Use `-ao_force` only if you want extra darkening on purpose. With `-pt_bounces 0` (direct + sky), AO still runs as a stand-in for missing bounce.


| Flag | Default | What it does |
| --- | --- | --- |
| `-ao` | off | Enable |
| `-ao_force` | off | Keep AO under path-trace GI |
| `-ao_samples N` | `16` | Rays per luxel (implies `-ao`) |
| `-ao_distance N` | `48` | Ray length |
| `-ao_strength N` | `1.0` | Darkening, 0–8 |
| `-ao_bias N` | `0.25` | Normal offset |
| `-ao_denoise` | off | Bilateral denoise |
| `-ao_denoise_radius N` | `1` | 1–4 |
| `-ao_denoise_strength N` | `1.0` | Blend 0–1 |

`light_ao` / `light_ao_vol` share `Enabled`, `Samples`, `Distance`, `Strength`, `Bias`. Denoise keys are on `light_ao` only; blend/priority on `light_ao_vol`. With `-gpu`, AO can use batched OpenCL.

---

## IES spots, projected textures, color temperature

These are bake-only extras on stock `light_spot`. Engine worldlights still use cone keys for dynamic lighting.

### IES (`IES`)


| Key | Default | What it does |
| --- | --- | --- |
| `IES` | (empty) | LM-63 file under `garrysmod/IES/` (also `IES/maps/<map>/`) |
| `IESScale` | `1` | Peak-normalized table scale |
| `IESBrightness` | `-1` | Bake brightness (`-1` = stock `_light`) |
| `IESMaxIntensity` | `0` | Per-luxel RGB cap (`0` = off) |

When `IES` is set, `_inner_cone` / `_cone` / `_exponent` are ignored for the bake. Distance falloff is unchanged. **TILT=NONE** only. Works in classic and path tracing.

### `ProjectedTexture`

Material under `materials/`. Detected from the VTF:


| Mode | Detection | Domain |
| --- | --- | --- |
| Planar 2D | Ordinary 2D VTF | Outer cone (`fill` / `fit`) |
| Envmap / cubemap | Cubemap VTF | Full sphere; entity angles rotate it |

Projected RGB multiplies intensity. With IES, the table is an angular mask. Cubemap UVs follow Direct3D / Source `CUBEMAP_FACE_*` (Z-up).

### Color temperature (`LightTemperature`)

Optional Kelvin on `light`, `light_spot`, `light_volume`, `light_environment`, and `light_env_vol`. `0` (default) leaves `_light` RGB alone. When set (1000–40000 K), RGB is replaced by a blackbody; brightness stays the 4th `_light` number, so `255 255 255 200` at **3000** K is as bright as white 200.

`AmbientTemperature` does the same for `_ambient`. The tinted color is also written to engine worldlights. With `IESBrightness`, temperature is applied after that rebuild.

Ballpark: 2700 incandescent, **3000** halogen / warm LED, 4000 cool white, 5600 daylight, 6500 overcast.

---

## Soft sun

Instead of Valve's fixed 30 random cone rays:

- Low-discrepancy spiral
- Count scales with angle (about 9 at 2°, about 40 at 50°)
- Stops early when fully lit or fully shadowed

`SunSpreadAngle` / `-softsun`: `0` = hard; a soft look is usually **0.5–3°**.

---

## Lightmap seams

### `-bounce_weld` (off by default)

Coplanar neighbors that share an edge contribute with distance falloff when bounce is written. `-bounce_soft N` (default `1`, range 0.5–4).

### Seam stitching (on; `-nostitch` off)

After `FinalLightFace`, luxels near overlapping edges of coplanar faces blend toward the neighbor at the same world position. Per lightstyle and bump layer. Displacements skipped.

### Edge pull (on; `-noedgepull` off)

Pulls edge samples inward (`-edgepull [N]`, default **0.5** luxels) so thin walls and jambs don't go black.

---

## Materials

### Textured emission (`$vrad_emit*`)


| Key | Default | What it does |
| --- | --- | --- |
| `$vrad_emit` | `0` | Enable (optional if strength > 0) |
| `$vrad_emitstrength` | `200` if emit-on | Intensity |
| `$vrad_emitdensity` | `1` | Finer patch chop |
| `$vrad_emitmask` | unset | Greyscale mask |
| `$vrad_emitmap` / `$vrad_emissivemap` | unset | Color map; else `$basetexture` |

Path tracing: fan-triangulated area mesh (`-pt_emit_samples`, default **64**; `0` = all tris). Classic: `emit_surface` proxies + self-illum. Worldlights cap **65536**.

### Colored glass (`$vrad_filter*`)

Path tracing only. Ordinary `$translucent` does **not** tint light unless `$vrad_filter 1`.


| Key | Default | What it does |
| --- | --- | --- |
| `$vrad_filter` | `0` | Enable |
| `$vrad_filtermap` | unset | Else `$basetexture` |
| `$vrad_filterstrength` | `1` | Mix white → filter color |
| `$vrad_filteropacity` | `0.05` | Flat absorption |
| `$vrad_filterthickness` | `1` | Extra density |
| `$vrad_filter_onesided` | `0` | Front-only if `1` |

Stacked panes mix in Oklab (L multiply, a,b add); energy stays linear. Up to 64 filter hits; entrance/exit on the same sheet are merged within 12 units.

---

## Path tracing (`-pathtrace` / `-dxr`)

Unidirectional diffuse paths with next-event estimation at every bounce. Default backend: **GPU RayQuery** (`-pt_gpu`). `-pt_cpu` forces the SSE CPU path (RGB only; no spectral).

**Spectral (GPU default):** `-pt_spectral` — four wavelengths, Smits RGB↔spectrum, CIE 1931 → linear sRGB (white-balanced). Saturated red lights stay red. `-pt_nospectral` for linear RGB.

Useful flags: `-pt_samples`, `-pt_bounces` (indirect hops after the luxel; `0` = direct + sky), `-pt_aa`, `-pt_emit_samples`, `-pt_denoise` / `-pt_denoiser` (`oidn`, `optix`, `sakai`). OIDN/OptiX denoise connected coplanar faces as one image. Soft shadows: `-pt_lightradius`, `-pt_lightpenumbra`, `-pt_softsamples`, `-pt_softmode`. Full list is in the CLI tables below.

Needs a DXR GPU. Easiest path: `-config full` with the path-trace flags uncommented.

### Static props

With `-StaticPropLighting`, the GPU baker lights prop lightmap texels and prop vertices. Vertices use **8×8** per-triangle charts packed into a **64×64**-cell atlas (`-pt_prop_vertgrid`, default **4**; `0` = one sample per vert). Charts are hole-filled and can be denoised (`-pt_denoise`). Each vertex takes the closest sample from every adjacent triangle and averages them. Override with `-pt_prop_samples` / `-pt_prop_bounces`. Classic prop path: 4-wide SSE direct + optional GPU bounce cull.

### Detail props

Grass and `prop_detail` get one GPU path-trace sample at the sprite/model centre when `-pathtrace` world bake succeeded (same `-pt_prop_samples` / `-pt_prop_bounces` as prop verts). Styled lights (1–63) still use classic direct. If the GPU baker is down, VRAD samples nearby world lightmaps and does **not** add classic sun on top (that double-counted after a path-trace bake). `-nodetaillight` skips this.

---

## Classic radiosity and OpenCL


| Flag | Role |
| --- | --- |
| `-coarse` / `-adaptivechop` / `-chop` / `-maxchop` | Patch size |
| `-texbounce` / `-texbounce_clean` | Textured albedo; kill chroma noise on near-greys |
| `-energy` / `-cavity` / `-noenergy` | Optional cavity-damped gather (default = stock Valve) |
| `-bounce_boost` / `-bounce_chroma` / `-bounce_weld` / `-bounce_soft` | Artistic / weld |
| `-maxtransfer N` | Cull far patch transfers |
| `-gpu` / `-gpu_batch` / `-gpu_maxtris` | OpenCL gather; sky occlusion stays CPU with `-textureshadows` |

OpenCL comes with the GPU driver. If it fails, VRAD falls back to CPU.

---

## Command-line flags

Valve flags plus PathRAD. Bold = typical defaults. Put `-game` / the map path on the Hammer line, not in `-config` files. The commented sheet is [configs/full.cfg](configs/full.cfg).

### Launcher / paths


| Parameter | Description |
| --- | --- |
| `-game` / `-vproject` | Game directory |
| `-insert_search_path` | Extra search path |
| `-config` / `-cfg <file>` | Load preset (`file`, `.cfg`, `.txt`, or next to `vrad.exe`) |
| `-novconfig` / `-steam` / `-allowdebug` | Tooling |

### Quality / mode


| Parameter | Description |
| --- | --- |
| `-hdr` / `-ldr` | Lightmap format |
| `-final` | High quality (= `-extrasky 16`) |
| `-fast` / `-fastambient` | Reduced sampling |
| `-extrasky N` / `-bounce N` | Sky rays / radiosity rounds (**100**; ignored for world if path tracing succeeds) |

### Threading / process


| Parameter | Description |
| --- | --- |
| `-threads N` | **1–256** (default = auto cores) |
| `-low` / `-verbose` / `-v` / `-StopOnExit` / `-FullMinidumps` / `-rederrors` | Process / log |

### Static props / detail / shadows


| Parameter | Description |
| --- | --- |
| `-StaticPropLighting` / `-StaticPropPolys` / `-StaticPropNormals` | Prop bake / debug |
| `-OnlyStaticProps` / `-nossprops` | Prop-only / no self-shadow |
| `-textureshadows` | Alpha textures cast shadows on `$alphatest` / `$translucent` props (no MDL flag). Path tracing: `$alphatest` cutout, `$translucent` soft `T*=(1-a)` |
| `-nodetaillight` / `-onlydetail` / `-noskyboxrecurse` | Skip detail bake / detail-only / skybox |

### GPU (OpenCL radiosity)


| Parameter | Description |
| --- | --- |
| `-gpu` | OpenCL bounce gather (+ batched AO) |
| `-gpu_maxtris N` / `-gpu_batch N` | BVH cap / rays per dispatch (**32768**) |

### Path tracing — DXR


| Parameter | Description |
| --- | --- |
| `-pathtrace` / `-dxr` | Enable path tracer |
| `-pt_gpu` / `-pt_cpu` | GPU (default) / CPU SSE |
| `-pt_samples N` | Samples per luxel (CLI default **4** / **2** fast / **8** final; max **4096**; configs often 256–1024) |
| `-pt_bounces N` | Indirect hops (**3**; **1** fast; **0** = direct+sky; max **16**) |
| `-pt_prop_samples` / `-pt_prop_bounces` / `-pt_prop_vertgrid` | Prop spp / hops / vert atlas on (**4**; `0`=per-vert) |
| `-pt_aa N` | Luxel footprint grid **1–5** (default **3**) |
| `-pt_lights N` | Local light samples (`0` = all) |
| `-pt_emit_samples N` | Area-emit samples (**64**; `0` = all tris) |
| `-pt_lightradius` / `-pt_lightpenumbra` / `-pt_softsamples` / `-pt_softmode` | Soft shadows |
| `-pt_spectral` / `-pt_nospectral` | Hero-wavelength (default on GPU) / RGB |
| `-pt_device N` | DXGI adapter index |
| `-pt_denoise` / `-pt_nodennoise` / `-pt_denoiser` / `-pt_denoise_radius` / `-pt_denoise_strength` | Denoise |

### Underwater


| Parameter | Description |
| --- | --- |
| `-underwater` | Mode A: Beer-Lambert + Kd |
| `-uw_volume` | Mode B: volume path tracing |
| `-uw_off` | Force off |
| `-uw_maxscatter N` | Scatter cap (**8**) |

### Ambient occlusion


| Parameter | Description |
| --- | --- |
| `-ao` / `-ao_force` / `-ao_samples` / `-ao_distance` / `-ao_strength` / `-ao_bias` | Enable / force / params |
| `-ao_denoise` / `-ao_denoise_radius` / `-ao_denoise_strength` | AO denoise |

### Radiosity / bounce


| Parameter | Description |
| --- | --- |
| `-texbounce` / `-texbounce_clean` / `-texbounce_noclean` | Textured bounce |
| `-coarse` / `-adaptivechop` / `-chop` / `-maxchop` | Patch chop |
| `-bounce_soft` / `-bounce_boost` / `-bounce_chroma` / `-bounce_weld` | Splat / artistic / weld |
| `-energy` / `-noenergy` / `-valve` / `-cavity` | Cavity damp |
| `-maxtransfer N` | Far transfer cull |

### Lightmap samples / seams


| Parameter | Description |
| --- | --- |
| `-edgepull [N]` / `-noedgepull` | Edge inset (default on, **0.5**) |
| `-nostitch` | Disable seam stitch |
| `-nobakevolume` | Ignore `bake_volume` (full map) |
| `-centersamples` / `-noextra` / `-debugextra` / `-dlightmap` / `-luxeldensity` / `-smooth` | Stock sample controls |

Never use `-luxeldensity` with Hammer lightmap scale 2 — values greater than 1 invert (e.g. 4 becomes 0.25 = coarser). After removing it, re-run VBSP once.

### Sun / sky / displacements


| Parameter | Description |
| --- | --- |
| `-softsun N` | Soft sun degrees (prefer entity `SunSpreadAngle`) |
| `-LargeDispSampleRadius` / `-dispchop` / `-disppatchradius` / `-maxdispsamplesize` | Displacements |

### Extra / debug / MPI


| Parameter | Description |
| --- | --- |
| `-lights` / `-dump*` / `-loghash` / `-mpi*` | Extra lights / debug / VMPI |

### Debug-build only (`ALLOWDEBUGOPTIONS`)

`-scale`, `-ambient`, `-dlight`, `-sky`, `-notexscale`, `-coring` — ignored in Release.

---

## License

**SOURCE 1 SDK LICENSE** — see [LICENSE](LICENSE). This is a derived work and stays under the same non-commercial terms.

Based on Source SDK 2013 tooling for Garry's Mod (64-bit), including work from [Ficool2's Source SDK 2013 fork](https://github.com/ficool2/source-sdk-2013). When this README and `vrad.exe -help` / `configs/full.cfg` disagree, trust the compiler and the config file.
