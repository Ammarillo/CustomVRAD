# CustomVRAD

Custom **64-bit VRAD** for **Garry’s Mod** / Source SDK 2013. Drop-in lighting compile with map entities and CLI options built on top of stock VRAD.

Compatible lightmap / BSP lighting output for the engine. Experimental — validate looks on your maps before shipping.

**Repo:** https://github.com/Ammarillo/CustomVRAD  
**FGD:** [`fgd/customvrad.fgd`](fgd/customvrad.fgd)  
**Algorithms:** [`docs/algorithms.tex`](docs/algorithms.tex) (technical report; compile with `pdflatex`)

---

## What’s included

| Feature | Type | Summary |
|---------|------|---------|
| `light_env_vol` | brush entity | Local sky / sun / ambient override volumes |
| Volume sun in leaf cubes | bake | Sky-visible leaf samples accumulate volume-blended sun into ambient cubes |
| `light_ao` / `light_ao_vol` | point / brush | Baked AO with multi-scale, selective direct/bounce/sky factors, bent normals |
| `light_absorb` | brush entity | Volumes that damp bounce (and optional direct) light |
| `light_volume` | point entity | Soft sphere point light (scattered origins, like soft sun) |
| `light_spot` `IES` / `IESScale` / `IESBrightness` / `IESMaxIntensity` | entity keys | IESNA LM-63 photometric intensity for spots (bake-only; replaces cone angles) |
| `light_spot` `ProjectedTexture` | entity key | Planar, cubemap, or spherical projected VTF modulation (auto-detect; bake-only) |
| Soft sun | bake | Faster, smoother `SunSpreadAngle` / `-softsun` cone sampling |
| Cross-face bounce weld | bake | Opt-in (`-bounce_weld`): edge-weighted bounce across coplanar seams |
| Lightmap seam stitching | bake | Blends luxels across coplanar VBSP face splits (`-nostitch` to disable) |
| `-gpu` | CLI | OpenCL bounce gather; sky occlusion stays CPU with `-TextureShadows` |
| `-pathtrace` / `-dxr` | CLI | D3D12 DXR path-traced world + static-prop lightmaps (GPU baker + `$vrad_emit` area lights) |
| `-config` / `-cfg` | CLI | Load a text preset of VRAD flags (keeps Hammer compile strings short) |
| `-coarse` / `-adaptivechop` / `-texbounce` / `-energy` / `-cavity` / `-maxtransfer` / `-bounce_soft` / `-bounce_boost` / `-bounce_chroma` | CLI | Faster / tunable radiosity (`-energy` cavity damp opt-in) |
| VMT `$vrad_emit*` | material | Textured emission; pathtrace uses PBRT-style area mesh NEE |
| VMT `$vrad_filter*` | material | Colored glass light transmission (pathtrace CPU + GPU) |
| Prop lighting speedups | bake | 4-wide SSE direct + GPU-culled bounce; `-pathtrace` GPU PathLi for props |
| Threading | runtime | Auto core detect (incl. >64), up to **256** threads |

Stock VRAD flags (`-hdr`, `-final`, `-StaticPropLighting`, `-textureshadows`, etc.) still work. Run `vrad.exe` with no args for the stock help text. Full CLI tables: [VRAD CLI reference](#vrad-cli-reference--all-parameters).

---

## Install / Hammer

1. Build or copy `bin\vrad.exe` + `bin\vrad_dll.dll` (and required Valve DLLs — see [Build](#build-windows)).
2. Point Hammer’s **Custom VRAD** (or expert compile) at this `vrad.exe`.
3. **Tools → Options → Game Configurations → Game Data Files** → add `fgd/customvrad.fgd` **after** `garrysmod.fgd` → restart Hammer.

| Classname | Place as |
|-----------|----------|
| `light_env_vol` | Tie brush to entity (trigger / nodraw) |
| `light_ao` | Point entity (Entity Tool) |
| `light_ao_vol` | Tie brush to entity |
| `light_absorb` | Tie brush to entity |
| `light_volume` | Point entity (Entity Tool) |

Legacy classname `light_environment_volume` is still accepted for env volumes. Stock **VBSP** is fine — these entities are compile-time only.

---

## `light_env_vol` — sky / sun volumes

Brush that overrides sky, sun, and ambient **inside** its bounds. Outside (and in the blend shell) mixes with the map `light_environment`. Volume lights are **not** exported as engine worldlights.

1. Keep one map `light_environment` as the global default.
2. Brush → Tie to Entity → `light_env_vol`.
3. Set the same lighting keys as `light_environment`.

| Key | Default | Description |
|-----|---------|-------------|
| `_light` / `_ambient` | (standard) | Sun and ambient color + brightness |
| `_lightHDR` / `_ambientHDR` | `-1 -1 -1 1` | HDR overrides; leave default to use SDR |
| `_lightscaleHDR` / `_AmbientScaleHDR` | `1` | HDR scales |
| `pitch` | `0` | Overrides pitch in Angles |
| `SunSpreadAngle` | `0` | Soft sun cone (degrees); uses CustomVRAD soft-sun sampler |
| `BlendDistance` | `0` | Soft fade length (`0` = hard cut). Perlin smootherstep |
| `BlendMode` | `2` Center | `0` Inside · `1` Outside · `2` Center |
| `priority` | `0` | Overlapping cores: higher wins; ties → smaller AABB |
| `BounceVolColor` | No | Recolor bounced light **inside** this volume to the volume’s `_light` hue |
| `BounceVolBright` | No | With BounceVolColor: also scale bounce luminance vs map env |
| `OutsideCastShadowIn` | Yes | Outside geometry casts sun/sky shadows **into** this volume |
| `InsideCastShadowOut` | Yes | Inside geometry casts sun/sky shadows **outside** this volume |

If an entity still has legacy keys (`OutsideCastShadow` / `InsideCastShadow`) that disagree with the new names, VRAD prefers the **legacy** value and prints a warning — delete the unused key in Hammer and re-save.

Bounds use the brush model AABB. Neighbor volumes use a soft Voronoi split so one volume’s outside halo does not tint another’s side.

**Dynamic entities** (players, NPCs, physics props) are lit by the per-leaf ambient cubes. CustomVRAD bakes those volume-aware: rays that hit lit geometry pick up volume-lit lightmaps; rays that hit sky use volume-blended `_ambient`; and when a sample can see the sun through sky, **volume-blended sun** (direction + color × visibility) is accumulated into the cube faces. Shared sky helpers also keep detail-prop and leaf-cube sky paths consistent.

**Note:** the engine’s exported `light_environment` worldlight is still a single global sun for dynamic RT lighting. Leaf-cube sun fill covers most of the volume look for ambient-lit dynamics; full per-volume dynamic sun would need engine changes. Enclosed volumes (no sky visibility) are unaffected.

**Shadow filters** use the hard volume AABB (not the blend shell). Set `OutsideCastShadowIn` to No so outdoor walls/props don’t darken an interior volume; set `InsideCastShadowOut` to No so interior blockers don’t shadow the courtyard outside.

---

## Ambient occlusion — `-ao`, `light_ao`, `light_ao_vol`

Multi-scale hemisphere AO baked into lightmaps in `FinalLightFace`.

**Enable via:** CLI `-ao` / `-ao_*`, point `light_ao` (map defaults), and/or brush `light_ao_vol` (local overrides). Combine freely.

### CLI

| Flag | Default | Effect |
|------|---------|--------|
| `-ao` | off | Enable AO pass |
| `-ao_samples N` | `16` | Rays per luxel (implies `-ao`; reduced with `-fast`) |
| `-ao_distance N` | `48` | Ray length (implies `-ao`) |
| `-ao_strength N` | `1.0` | Darkening `0`–`8` (implies `-ao`) |
| `-ao_bias N` | `0.25` | Normal offset (implies `-ao`) |
| `-ao_denoise` | off | Edge-preserving bilateral denoise |
| `-ao_denoise_radius N` | `1` | Radius `1`–`4` |
| `-ao_denoise_strength N` | `1.0` | Blend `0`–`1` |

### Entity keys

Shared by `light_ao` and `light_ao_vol` (defaults match CLI): `Enabled`, `Samples`, `Distance`, `Strength`, `Bias`.

`light_ao` only: `Denoise`, `DenoiseRadius`, `DenoiseStrength`.  
`light_ao_vol` only: `BlendDistance` / `BlendMode` / `priority` (same idea as env vols).

Volumes soft-blend settings. `Enabled=No` carves AO out; `Enabled=Yes` can add AO when the map default is off. With `-gpu`, AO can use batched OpenCL.

---

## `light_absorb` — absorber volumes

Damps lighting inside a soft-blended brush volume.

| Key | Default | Description |
|-----|---------|-------------|
| `Strength` | `0.5` | Absorb amount (`0`–`1`) |
| `AbsorbDirect` | No | Also damp direct lights at luxels |
| `AbsorbBounce` | Yes | Damp radiosity gather |
| `BlendDistance` / `BlendMode` / `priority` | like env vols | Soft fade + overlap |

Start low — high strength can crush bounce.

---

## `light_bounce_vol` — bounce boost / chroma / energy volumes

Soft-blended brush volume that overrides artistic radiosity controls locally. Mixes with CLI `-bounce_boost` / `-bounce_chroma` / `-energy` in the blend shell.

| Key | Default | Description |
|-----|---------|-------------|
| `BounceBoost` | `-1` (inherit CLI) | Bounce brightness scale `0`–`16` inside the volume |
| `BounceChroma` | `-1` (inherit CLI) | Early-bounce saturation `0`–`8` (strongest on bounce #1) |
| `EnergyMode` | Inherit | Local cavity-damp: **Energy** or **No Energy**. Soft-blends with CLI |
| `BlendDistance` / `BlendMode` / `priority` | like env vols | Soft fade + overlap |

Use with `-texbounce` so there is chroma to boost. Example: `-texbounce -bounce_chroma 1.5` map-wide, then a volume with `BounceChroma 3` over a colorful floor.

---

## `light_volume` — soft sphere point light

Point entity with the same keys as a normal `light` (inherits the stock `Light` base / editor icon), plus **`Radius`** (default **16**). Soft-samples the light origin with low-discrepancy points scattered **inside a sphere** of that radius — same idea as soft sun, but omnidirectional — so shadows get a soft penumbra instead of a hard point-light edge.

| Key | Default | Description |
|-----|---------|-------------|
| `_light` / `_lightHDR` / `_lightscaleHDR` | (standard) | Color and brightness |
| `style` | `0` | Appearance / lightstyle |
| `_constant_attn` / `_linear_attn` / `_quadratic_attn` | `0` / `0` / `1` | Falloff |
| `_fifty_percent_distance` / `_zero_percent_distance` | `0` | Alternate falloff (overrides Constant/Linear/Quadratic) |
| `_hardfalloff` | `0` | Hard fade to zero with falloff distances |
| `_distance` | `0` | Max distance for engine worldlights |
| `Radius` | `16` | Soft-sample sphere radius (world units). `0` = hard point |

Sample count scales with radius (~16 at default 16, capped at 40; reduced with `-fast`). Fully lit / fully shadowed luxels early-out after a short probe. Exported to the engine as a normal **point** light at the center (softness is bake-only).

---

## `light_spot` photometry and projected textures

Bake-only extensions on stock `light_spot`. Engine worldlights continue to use
cone keys for dynamic lighting; the lightmap bake may replace or modulate that
response with an IESNA table and/or a projected VTF.

### IESNA profiles (`IES`)

| Key | Default | Description |
|-----|---------|-------------|
| `IES` | (empty) | LM-63 `.ies` path under `garrysmod/IES/` (also `IES/maps/<map>/`) |
| `IESScale` | `1` | Scales the peak-normalized angular table |
| `IESBrightness` | `-1` | Bake brightness (same units as `_light`’s 4th component). `-1` uses stock `_light`. Values `≥ 0` override bake intensity only; set `_light` brightness to `0` to suppress engine dynamic light |
| `IESMaxIntensity` | `0` | Per-luxel RGB cap on this light’s direct contribution (`0` = off) |

When `IES` is set, the bake ignores `_inner_cone`, `_cone`, and `_exponent`; the
candela table defines the beam. Distance attenuation is unchanged. Supported on
classic radiosity and path tracing (CPU/GPU). Only `TILT=NONE` files are accepted.

Place e.g. `garrysmod/IES/light01.ies`, set `IES` to `light01`, and optionally
`IESBrightness` for bake intensity independent of `_light`.

### `ProjectedTexture`

Material path under `materials/` (e.g. `lights/gobos/logo`). Classification is
automatic from the VTF:

| VTF | Behaviour |
|-----|-----------|
| Planar 2D | Image framed to the outer cone (`_cone`); see `ProjectedTextureMode` |
| Cubemap (6 faces) | Omnidirectional cube sample; entity angles rotate the frame |
| 2D + ENVMAP (matcap / spheremap / equirect) | Omnidirectional 2D map — square → spheremap, wide → equirect |

**`ProjectedTextureMode`** (planar only; default `fill`):

| Mode | Geometry |
|------|----------|
| `fill` | Square sides lie on the cone; corners are clipped |
| `fit` | Square inscribed in the cone (`×√2`); full image visible |

Projected RGB multiplies light intensity. With `IES` also set, the IES table is
an angular mask on the projection. Cubemap / spherical modes are
omnidirectional (no cone cull). Face UVs follow Direct3D / Source
`CUBEMAP_FACE_*` conventions (Z-up axes). See
[`docs/algorithms.tex`](docs/algorithms.tex) § Photometric spots and projected textures.

---

## Soft sun (`SunSpreadAngle` / `-softsun`)

Replaces stock’s fixed 30 random rays:

- **Low-discrepancy cone** (golden-angle spiral) — smooth penumbras, less noise at large angles
- **Adaptive sample count** — scales with angle (~9 at 2°, up to ~40 at 50°)
- **Early-out** — fully lit / fully shadowed luxels stop after a short probe; only penumbra pays full cost

`0` = hard sun. Typical soft look: **0.5–3°**. Very large angles (e.g. 50°) still cost more, but less than stock and look cleaner.

---

## Cross-face bounce welding

**Off by default.** Enable with `-bounce_weld`.

When enabled, coplanar neighbor faces that share an edge contribute with **edge-distance falloff** when bounce is written into lightmaps. Distant patches no longer smear rectangular GI across the seam. Non-coplanar neighbors keep the stock neighbor splat. Leaving this off matches stock neighbor bleed (smoother ceilings; possible rectangular GI on coplanar splits).

Tunable with `-bounce_soft N` (default `1`; range `0.5`–`4`; `<1` tighter).

---

## Lightmap seam stitching

Stock VRAD filters every face's lightmap independently, so coplanar faces split by VBSP (grid splits, brush boundaries) can land on slightly different luxel values along the shared edge — a faint brightness step even on flat, evenly lit surfaces.

After `FinalLightFace`, CustomVRAD blends luxels near each shared edge of coplanar faces toward the neighbor's value at the same world position; at the edge both sides converge to the same average, removing the step. Applies per lightstyle and bump layer; displacements are skipped (they have their own edge rules).

**On by default.** Disable with `-nostitch`.

---

## GPU (`-gpu`)

Optional OpenCL path (project links `OpenCL.lib` from `src/lib/public/x64`).

**OpenCL ICD:** usually already on the machine — NVIDIA, AMD, and Intel GPU drivers ship the OpenCL ICD / loader with a normal driver install. You do **not** need a separate OpenCL download for CustomVRAD, and it is **not** bundled in the release zip. If `-gpu` fails to find a device, update your GPU driver; without OpenCL, everything falls back to CPU.

| Flag | Effect |
|------|--------|
| `-gpu` | Bounce gather (+ batched AO). Sky/ambient + soft-sun occlusion stay on CPU when `-TextureShadows` is set |
| `-gpu_maxtris N` | Optional BVH triangle cap (`0` = unlimited) |
| `-gpu_batch N` | Rays per dispatch (default **32768**; lower = safer vs TDR) |

Also used when present:

- Soft-sun / sky closest-hit batches (**CPU** when `-TextureShadows` — GPU BVH is opaque-only and was causing jagged wall/ceiling contact strips)
- Static-prop **indirect** ray pre-cull (skip BSP walk for sky / miss)

**Stability:** small ray batches, chunked uploads, device alloc checks, auto CPU fallback on OpenCL errors.  
**Throughput:** each CPU worker can use its own OpenCL queue + ray buffers (pool up to 64) so dispatches overlap instead of serializing on one queue.

Without `-gpu`, everything falls back to CPU.

---

## Radiosity / speed CLI

| Flag | Effect |
|------|--------|
| `-coarse` | Patch chop `8` — fewer patches, faster VisLeafs / bounce |
| `-adaptivechop` | Finer patches where sky visibility contrasts (floor `4` under `-coarse`) |
| `-texbounce` | Sample `$basetexture` albedo per patch for color bleed (supports VTF 7.5; fallback: flat texdata average) |
| `-texbounce_clean N` | Soft-kill DXT/JPEG chroma noise on near-greys (Oklab C; default **0.04**; **0**/`-texbounce_noclean` = off). Stops yellowish bounce from “white” walls |
| `-energy` | Opt-in — cavity-damped radiosity (enclosed bounce darkened; outdoor nearly unchanged) |
| `-noenergy` / `-valve` | Stock Valve radiosity (no enclosure damp; **default**) |
| `-cavity N` | Fully-enclosed gather scale vs stock (default `0.70`; range `0.25`–`1`; implies `-energy`) |
| `-maxtransfer N` | Skip patch transfers farther than N units |
| `-bounce_soft N` | Bounce luxel splat scale (default `1` = stock) |
| `-bounce_weld` | Opt-in: cull distant coplanar neighbor bounce (can blotch ceilings; **off by default**) |
| `-bounce_boost N` | **Artistic** scale of final bounced light after radiosity (`1` = stock; `0`–`16`). Direct lights unchanged. Does not compound across bounces. |
| `-bounce_chroma N` | **Artistic** early-bounce saturation (`0` = off; `0`–`8`). Keeps luminance; strongest on bounce #1. Use with `-texbounce`. Overridable per-area via `light_bounce_vol`. **Ignored by `-pathtrace`** (linear RGB × texbounce). |
| `-nostitch` | Disable lightmap seam stitching across coplanar face splits |
| `-edgepull [N]` | **On by default** — pull edge luxel samples inward (`N` luxels, default `0.5`). Reduces black strips on thin walls / door jambs |
| `-noedgepull` | Stock Valve sample positions (no edge inset) |
| `-threads N` | Override thread count (`1`–`256`) |

Auto-detects logical processors (including >64 via processor groups). Work dispatch uses atomics so high thread counts scale better.

---

## VMT textured emission (`$vrad_emit*`)

Per-material emissive surfaces driven by VMT keys (no `lights.rad` entry required). Emission color comes from texels — same idea as `-texbounce`.

| Key | Default | Description |
|-----|---------|-------------|
| `$vrad_emit` | `0` | Set to `1` to enable (optional if strength &gt; 0) |
| `$vrad_emitstrength` | `200` if emit on without value | Intensity scaler (same role as the 4th number in `lights.rad`) |
| `$vrad_emitdensity` | `1` | Optional finer patch chop only (`>1` subdivides more) |
| `$vrad_emitmask` | unset | Optional greyscale mask — white = full emit, black = none |
| `$vrad_emitmap` / `$vrad_emissivemap` | unset | Optional color map for emitted light. If unset, uses `$basetexture` |

Example:

```
LightmappedGeneric
{
	"$basetexture" "aui/random/rainbow"
	"$vrad_emit" "1"
	"$vrad_emitstrength" "400"
	"$vrad_emitmask" "aui/random/rainbow_emitmask"
	// optional: "$vrad_emitmap" "aui/random/rainbow_emitcolor"
}
```

**How it bakes**

- Samples color × mask × strength from the emitmap / `$basetexture`
- Adds **self-illum** on the emitting face so it glows in the lightmap
- Under **`-pathtrace`**: builds a **textured triangle area-light mesh** (PBRT-style NEE — power-pick tri, uniform area sample, emission from albedo at the sample). Soft continuous lighting, not a grid of face-center point lights
- Under classic radiosity: still creates `emit_surface` direct lights for leaf ambient / worldlights
- `lights.rad` texlights stay on the stock point-proxy path
- Works with VTF 7.5. Worldlights cap is **65536** (stock 8192)

**Noise control (pathtrace):** raise `-pt_emit_samples` (default **64**) or use `0` for all emit tris every NEE. Pair with `-pt_denoise`.

---

## VMT colored glass (`$vrad_filter*`)

Thin-sheet colored light transmission through brush glass under **`-pathtrace`** (CPU + GPU). Classic radiosity is unchanged. Opt-in only — ordinary `$translucent` / WINDOW glass still lets light through unfiltered unless you set `$vrad_filter 1`.

| Key | Default | Description |
|-----|---------|-------------|
| `$vrad_filter` | `0` | `1` enables colored transmission |
| `$vrad_filtermap` | unset | Optional RGB filter map; else `$basetexture` |
| `$vrad_filterstrength` | `1` | `0` = no tint, `1` = full texture tint |
| `$vrad_filteropacity` | `0.05` | Flat absorption (keep low; color comes from the texture) |
| `$vrad_filterthickness` | `1` | `pow(filterRGB, thickness)` for denser stained glass |
| `$vrad_filter_onesided` | `0` | `1` = front-face tint only; default is two-sided |
| `$nocull` | — | Forces two-sided if set |

Stacked panes / strength / bounce chroma / env inbound tint use **Oklab** (Björn Ottosson). Energy transport (`albedo × light`, NEE `intensity × vis`) stays linear RGB.

- Strength: Oklab lerp white → filter  
- Stacked filters: `L' = L0·L1`, `a' = a0+a1`, `b' = b0+b1`  
- `-bounce_chroma`: scale Oklab `a,b` (keep `L`)  
- `light_env_vol` inbound tint: keep light `L`, take tint `a,b`

Transmittance: `T = OklabLerp(white, pow(rgb, thickness), strength) * (1 - opacity)`.

Example:

```
LightmappedGeneric
{
	"$basetexture" "aui/glass/stained_red"
	"%keywords" "glass"
	"$translucent" "1"
	"$vrad_filter" "1"
	"$vrad_filterstrength" "1"
	"$vrad_filteropacity" "0.05"
	"$vrad_filterthickness" "1.5"
}
```

**Notes**

- Pathtrace only (`-pathtrace` / `-pt_gpu` / `-pt_cpu`). Rays multiply by each pane’s `T` and continue.
- Two-sided by default so angled/stacked panes blend regardless of facing. Use `$vrad_filter_onesided 1` for front-only.
- Do not rely on `$alpha` for light opacity (that is framebuffer blend); set `$vrad_filteropacity` explicitly if needed.
- CPU and GPU both sample the filter texture at the hit (GPU: one `Texture2DArray` layer per unique filter VTF). Face-average fallback if a texture won't fit.
- Static-prop `-textureshadows` stays grayscale. Classic radiosity form-factor glass is a follow-up.

---

## Static prop lighting (`-StaticPropLighting`)

CustomVRAD speedups on top of stock prop vertex lighting:

- **4 vertices per SSE gather** for direct light (stock duplicated one vert across all lanes)
- **GPU bounce-ray culling** with `-gpu` (sky / miss rays skip the BSP lightmap walk)
- **Cleaner bounce on props** — bilinear luxel reads + cheap vertex soften (keeps stock-fast sample counts)
- **`-pathtrace` + `-StaticPropLighting`** — GPU pathtraces props: **lightmap texels** match world spp/bounces/NEE; **vertex** lighting samples a virtual per-triangle lightmap (`-pt_prop_vertgrid`, default **4**) and barycentric-weights those samples onto verts (denoiser-friendly spatial filter; `0` = old noisy per-vert). Multi-LOD models inherit the same vertex/texel colors on every LOD (shared studio verts + nearest-lit fill). Override spp/bounces with `-pt_prop_samples` / `-pt_prop_bounces`. CPU gather fallback if DXR bake fails.

Quality flags like `-StaticPropPolys` / `-TextureShadows` still apply and are expensive — drop them for preview compiles.

---

## Path tracing (`-pathtrace` / `-dxr`)

Optional **D3D12 DXR** baker that replaces stock `BuildFacelights` + radiosity bounce for **world faces** (direct + multi-bounce GI + sky, soft lights, textured `$vrad_emit*` area lights). With **`-StaticPropLighting`**, the same GPU PathLi pass also bakes static prop vertex/lightmap samples. Leaf ambient / detail props stay on the stock path. Falls back to radiosity (world) or CPU prop gather if DXR init/bake fails.

**Colored light (physical):** default **`-pt_spectral`** — hero-wavelength path transport (Smits 1999 RGB→spectrum via PBRT tables, CIE 1931 XYZ → linear sRGB). Lights and albedo multiply in λ-space so colored bounce mixes like a spectral renderer, then projects back for lightmaps. Use **`-pt_nospectral`** for linear RGB. Firefly clamp is luminance-preserving on all bounces; artistic chroma/env tint ignored.

Default bake path is the **GPU RayQuery luxel baker** (`-pt_gpu`). Use `-pt_cpu` for the SSE CPU integrator.

Common flags: `-pt_samples`, `-pt_bounces`, `-pt_aa`, `-pt_emit_samples`, `-pt_denoise` / `-pt_denoiser`. Full list: [Path tracing — DXR](#path-tracing--dxr) in the CLI reference below.

**`$vrad_emit*` under pathtrace:** fan-triangulated mesh lights with PBRT-style area sampling. Noise vs speed: `-pt_emit_samples` (`0` = all tris).

**`$vrad_filter*` under pathtrace:** colored transmission through opt-in glass panes (see [VMT colored glass](#vmt-colored-glass-vrad_filter)).

Needs a DXR-capable GPU. Prefer [`configs/full.cfg`](configs/full.cfg).

```bat
vrad.exe -config full -game "<gmod>\garrysmod" "<map>"
```

---

## VRAD CLI reference — all parameters

Complete command-line flag list for CustomVRAD (stock Valve + CustomVRAD extensions). Values in **bold** are typical defaults. Pass `-game` / map path on the Hammer line — not inside `-config` files.

Also see [`configs/full.cfg`](configs/full.cfg) (same flags, commented for toggling).

### Launcher / paths

| Parameter | Description |
|-----------|-------------|
| `-game <dir>` / `-vproject <dir>` | Game directory (materials, maps search path). Required for normal compiles. |
| `-insert_search_path <dir>` | Extra filesystem search path |
| `-config` / `-cfg <file>` | Load flags from a text preset (`file`, `file.cfg`, `file.txt`, or `vrad.exe\configs\…`) |
| `-novconfig` | Do not show GUI on VProject errors |
| `-steam` | Steam mode (tooling) |
| `-allowdebug` | Allow debug (tooling) |

### Quality / mode

| Parameter | Description |
|-----------|-------------|
| `-hdr` | Bake HDR lightmaps |
| `-ldr` | Bake LDR lightmaps |
| `-final` | High quality (= `-extrasky 16`) |
| `-fast` | Quick/dirty lighting (fewer rays; lower pathtrace defaults) |
| `-fastambient` | Lower-quality per-leaf ambient sampling |
| `-extrasky N` | Trace N× as many rays for indirect / sky ambient |
| `-bounce N` | Max radiosity bounces (default **100**; ignored for world faces when pathtrace succeeds) |

### Threading / process

| Parameter | Description |
|-----------|-------------|
| `-threads N` | Worker threads (`1`–`256`; default = auto-detect cores) |
| `-low` | Idle process priority |
| `-verbose` / `-v` | Verbose log output |
| `-StopOnExit` | Wait for keypress on exit |
| `-FullMinidumps` | Large crash minidumps |
| `-rederrors` / `-rederror` | Show errors in red |

### Static props / detail / shadows

| Parameter | Description |
|-----------|-------------|
| `-StaticPropLighting` | Bake static prop vertex lighting |
| `-StaticPropPolys` | Prop shadows at polygon precision (slower, sharper) |
| `-StaticPropNormals` | Debug: show prop normals instead of lighting |
| `-OnlyStaticProps` | Only direct static prop lighting (debug) |
| `-nossprops` | Disable self-shadowing on static props |
| `-textureshadows` | Alpha textures block light (sampled along rays) |
| `-nodetaillight` | Do not light detail props |
| `-onlydetail` | Only detail props + per-leaf lighting |
| `-noskyboxrecurse` | No 3D skybox recursion (skybox shadows on world) |

### GPU (OpenCL radiosity)

| Parameter | Description |
|-----------|-------------|
| `-gpu` | OpenCL bounce gather; sky occlusion stays CPU when `-textureshadows` |
| `-gpu_maxtris N` | Optional BVH triangle cap (`0` = unlimited) |
| `-gpu_batch N` | Rays per OpenCL dispatch (default **32768**) |

### Path tracing — DXR

| Parameter | Description |
|-----------|-------------|
| `-pathtrace` / `-dxr` | Path-traced world (+ prop) lightmaps (replaces BuildFacelights + world radiosity) |
| `-pt_gpu` | GPU RayQuery luxel baker (default when DXR ready) |
| `-pt_cpu` | Force CPU SSE path tracer (full soft-shadow path) |
| `-pt_samples N` | Samples per luxel (CLI default **4** / **2** `-fast` / **8** `-final`; max **4096**; configs often **256**–**1024**) |
| `-pt_bounces N` | Indirect hops after the luxel (**0** = direct+sky only; default **3**; **1** `-fast`; max **16**) |
| `-pt_prop_samples N` | Prop spp (default **max(8, pt_samples/4)**) |
| `-pt_prop_bounces N` | Prop indirect hops (**0** = direct+sky; default **same as `-pt_bounces`**) |
| `-pt_prop_vertgrid N` | Virtual triangle lightmap edge subdiv for vertex lighting (default **4**; **0** = one sample/vert) |
| `-pt_aa N` | Luxel footprint AA grid **1**–**5** (`1` = off; default **3**) |
| `-pt_lights N` | Local point/spot NEE samples (`0` = all; sky always all) |
| `-pt_emit_samples N` | `$vrad_emit` **area** NEE samples (`0` = all tris; default **64**; higher = less noise) |
| `-pt_lightradius N` | Soft disk radius for `light` / `light_spot` (`0` = hard; world units) |
| `-pt_lightpenumbra N` | Softness growth vs distance (default **1**) |
| `-pt_softsamples N` | Max soft visibility rays per NEE (default **16**) |
| `-pt_softmode direct\|all` | Soft only at luxel (**direct**) or every bounce (**all**) |
| `-pt_device N` | DXGI adapter index |
| `-pt_denoise` | Enable pathtrace denoise |
| `-pt_nodennoise` | Disable pathtrace denoise |
| `-pt_denoiser oidn\|optix\|sakai` | Denoiser backend (implies `-pt_denoise`; default **oidn**) |
| `-pt_denoise_radius N` | Sakai filter radius **1**–**8** (default **3**; OIDN/OptiX ignore) |
| `-pt_denoise_strength N` | Blend **0**–**1** noisy→denoised (default **1**) |

### Ambient occlusion

| Parameter | Description |
|-----------|-------------|
| `-ao` | Bake cosine-weighted AO into lightmaps |
| `-ao_samples N` | AO rays per luxel (default **16**; implies `-ao`) |
| `-ao_distance N` | AO ray length, world units (default **48**; implies `-ao`) |
| `-ao_strength N` | Darkening strength (default **1**, max **8**; implies `-ao`) |
| `-ao_bias N` | Offset along normal (default **0.25**; implies `-ao`) |
| `-ao_denoise` | Edge-preserving AO denoise (implies `-ao`) |
| `-ao_denoise_radius N` | Denoise radius **1**–**4** (default **1**) |
| `-ao_denoise_strength N` | Denoise blend **0**–**1** (default **1**) |

### Radiosity / bounce (world faces skipped if pathtrace succeeds)

| Parameter | Description |
|-----------|-------------|
| `-texbounce` | Sample `$basetexture` albedo per patch for colored bounce |
| `-texbounce_clean N` | Kill compression chroma on near-greys (default **0.04**; **0** = off) |
| `-texbounce_noclean` | Keep raw texture tint (disable clean) |
| `-coarse` | Larger lighting patches (chop **8**) — faster VisLeafs/bounce |
| `-adaptivechop` | Finer patch floor under `-coarse` (elongated patches) |
| `-chop N` | Smallest luxel widths for a bounce patch (edges) |
| `-maxchop N` | Coarsest luxel widths for a patch (face interiors) |
| `-bounce_soft N` | Bounce luxel splat scale (default **1** = stock; **0.5**–**4**) |
| `-bounce_boost N` | Scale final bounced light (**1** = stock; **0**–**16**; direct unchanged) |
| `-bounce_chroma N` | Early-bounce saturation (**0** = off; **0**–**8**; use with `-texbounce`) |
| `-bounce_weld` | Cull distant coplanar neighbor bounce (can blotch; **off** by default) |
| `-energy` | Cavity-damped radiosity — darkens enclosed bounce (**off** by default) |
| `-noenergy` / `-valve` | Stock Valve radiosity (no enclosure damp; default) |
| `-cavity N` | Fully-enclosed gather scale vs stock (default **0.70**; **0.25**–**1**; implies `-energy`) |
| `-maxtransfer N` | Skip patch transfers farther than N units (`0` = off) |

### Lightmap samples / seams

| Parameter | Description |
|-----------|-------------|
| `-edgepull [N]` | Pull edge luxel samples inward (**on** by default; N luxels, default **0.5**) |
| `-noedgepull` | Stock Valve sample positions (no edge inset) |
| `-nostitch` | Disable lightmap seam stitching across coplanar face splits |
| `-centersamples` | Move sample centers |
| `-noextra` | Disable supersampling |
| `-debugextra` | Debug data in lightmaps to visualize supersampling |
| `-dlightmap` | Force direct lighting into a different lightmap than radiosity |
| `-luxeldensity N` | Rescale all luxels (default **1**). **Avoid with Hammer lightmap scale 2** — values &gt;1 invert and coarsen |
| `-smooth N` | Smoothing-group threshold in degrees (default **45**) |

### Sun / sky / displacements

| Parameter | Description |
|-----------|-------------|
| `-softsun N` | Treat sun as area light of size N degrees (soft shadows; typical **0**–**5**; default **0**). Prefer entity `SunSpreadAngle` when possible |
| `-LargeDispSampleRadius` | Wider bounce gather on displacements (fixes splotches; slower) |
| `-dispchop N` | Displacement chop size (default **8**) |
| `-disppatchradius N` | Displacement patch radius (default **512**) |
| `-maxdispsamplesize N` | Max displacement sample size (default **512**) |

### Extra lights / debug / MPI

| Parameter | Description |
|-----------|-------------|
| `-lights <file>` | Additional lights file beyond `lights.rad` / map |
| `-dump` | Write debugging `.txt` files |
| `-dumpnormals` | Write normals to debug files |
| `-dumptrace` | Write ray-tracing environment to debug files |
| `-dumppropmaps` | Dump prop lightmaps |
| `-loghash` | Log sample hash table to `samplehash.txt` |
| `-mpi` | Use VMPI distributed compile |
| `-mpi_ListParams` | List VMPI parameters |
| `-mpi_pw <pw>` | Password for a specific VMPI worker set |

### Debug-build only (`ALLOWDEBUGOPTIONS` — ignored in Release)

| Parameter | Description |
|-----------|-------------|
| `-scale N` | Global light scale |
| `-ambient R G B` | Add flat ambient |
| `-dlight N` | Direct light threshold tweak |
| `-sky N` | Sky scale |
| `-notexscale` | Disable texture scaling in lighting |
| `-coring N` | Coring threshold |

---

## Config presets (`-config`)

Put VRAD flags in a text file so Hammer expert compile stays short:

```bat
-config "PathToCustomVRAD\configs\quality" -game $gamedir $path\$file
```

| Preset | File | Intent |
|--------|------|--------|
| **full** | [`configs/full.cfg`](configs/full.cfg) | **All flags**, grouped — uncomment to toggle |
| quality | [`configs/quality.cfg`](configs/quality.cfg) | Final + props + AO + GPU |
| preview | [`configs/preview.cfg`](configs/preview.cfg) | Fast look |
| pathtrace | [`configs/pathtrace.cfg`](configs/pathtrace.cfg) | DXR path-traced lightmaps |
| ao | [`configs/ao.cfg`](configs/ao.cfg) | Strong AO focus |

Start from **`full.cfg`** if you want every option listed. Commented lines (`# …`) are off; uncomment to enable.

**Format:** one flag (or `flag value`) per line / space-separated. `#` and `//` comments. Do **not** put `-game` or the map path in the config — keep those on the Hammer command line.

**Path lookup:** tries `path`, `path.cfg`, `path.txt`, then the same next to `vrad.exe` and `vrad.exe\configs\`.

CLI args after `-config` still apply and can override (e.g. `-config preview -final`).

---

## Example commands

**Using a config preset (recommended for Hammer):**

```bat
PathToCustomVRAD\bin\vrad.exe -config "PathToCustomVRAD\configs\quality" -game $gamedir $path\$file
```

**Quality (explicit flags):**

```bat
PathToCustomVRAD\bin\vrad.exe -hdr -final -StaticPropLighting -textureshadows -texbounce -ao -ao_samples 32 -gpu -game "PathToGarrysMod\garrysmod" "PathToMap\map"
```

**Preview:**

```bat
PathToCustomVRAD\bin\vrad.exe -hdr -fast -coarse -ao -ao_samples 8 -gpu -game "PathToGarrysMod\garrysmod" "PathToMap\map"
```

**AO-focused:**

```bat
vrad.exe -ao -ao_samples 32 -ao_distance 64 -ao_strength 0.85 -ao_denoise -gpu -hdr -game "<gmod>\garrysmod" "<map>"
```

---

## Build (Windows)

**Needs:** Visual Studio 2022 (Desktop C++, MSVC v143, Windows 10/11 SDK). OpenCL SDK/ICD only if you use `-gpu`.

**Recommended:** run `build.ps1` at the repo root. It always does a full rebuild of `vrad_dll` (no stale object files — a mixed incremental build once produced a DLL that crashed mid-bake), auto-picks an MSBuild with the v143 toolset, and verifies `bin\vrad_dll.dll` was actually written. `.\build.ps1 -Incremental` for fast iteration only.

Manual alternative:

1. Open `src/Source GPU compiles tool (L-I).sln`
2. Build **Release | x64**
3. Outputs under `/bin/` — mainly `vrad_dll_win64` / launcher

**Runtime next to `vrad.exe`:**

```
vrad.exe
vrad_dll.dll
tier0.dll
vstdlib.dll
vphysics.dll          ← use GMod's vphysics_stub.dll copied as vphysics.dll
bin/x64/filesystem_stdio.dll
bin/x64/vphysics.dll  ← same stub
```

Do **not** use GMod `bin/win64/vphysics.dll` here — it needs GMod’s `tier0` and fails with `Unable to load vphysics DLL`.

**Platform:** Windows only.

---

## License

**SOURCE 1 SDK LICENSE** — see [LICENSE](LICENSE). Derived work remains under the same non-commercial terms.

Based on Source SDK 2013 tooling adapted for Garry’s Mod (64-bit), including work from [Ficool2’s Source SDK 2013 fork](https://github.com/ficool2/source-sdk-2013).
