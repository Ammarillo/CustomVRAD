# CustomVRAD

Custom **64-bit VRAD** for **Garry’s Mod** / Source SDK 2013. Drop-in lighting compile with map entities and CLI options built on top of stock VRAD.

Compatible lightmap / BSP lighting output for the engine. Experimental — validate looks on your maps before shipping.

**Repo:** https://github.com/Ammarillo/CustomVRAD  
**FGD:** [`fgd/customvrad.fgd`](fgd/customvrad.fgd)

---

## What’s included

| Feature | Type | Summary |
|---------|------|---------|
| `light_env_vol` | brush entity | Local sky / sun / ambient override volumes |
| `light_ao` / `light_ao_vol` | point / brush | Baked ambient occlusion (map-wide or local) |
| `light_absorb` | brush entity | Volumes that damp bounce (and optional direct) light |
| Soft sun | bake | Faster, smoother `SunSpreadAngle` / `-softsun` cone sampling |
| Cross-face bounce weld | bake | Edge-weighted bounce across coplanar face seams |
| Lightmap seam stitching | bake | Blends luxels across coplanar VBSP face splits (`-nostitch` to disable) |
| `-gpu` | CLI | OpenCL bounce gather, AO / sky occlusion, prop bounce culling |
| `-coarse` / `-maxtransfer` / `-bounce_soft` | CLI | Faster / tunable radiosity |
| Prop lighting speedups | bake | 4-wide SSE direct + GPU-culled bounce for `-StaticPropLighting` |
| Threading | runtime | Auto core detect (incl. >64), up to **256** threads |

Stock VRAD flags (`-hdr`, `-final`, `-StaticPropLighting`, `-textureshadows`, etc.) still work. Run `vrad.exe` with no args for the full stock list.

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
| `BounceVolColor` | No | Recolor inbound radiosity to this volume’s `_light` hue |
| `BounceVolBright` | No | With BounceVolColor: also scale bounce luminance vs map env |
| `OutsideCastShadowIn` | Yes | Outside geometry casts sun/sky shadows **into** this volume |
| `InsideCastShadowOut` | Yes | Inside geometry casts sun/sky shadows **outside** this volume |

Bounds use the brush model AABB. Neighbor volumes use a soft Voronoi split so one volume’s outside halo does not tint another’s side.

**Dynamic entities** (players, NPCs, physics props) are lit by the per-leaf ambient cubes, which CustomVRAD bakes volume-aware: rays that hit lit geometry pick up the volume-lit lightmaps, and rays that hit sky use the volume-blended `_ambient` at the sample position instead of the map default. Detail props get the same treatment. Limitation: the engine’s **dynamic sun** (`light_environment` worldlight) is global — a dynamic entity that can trace to sky still receives the map sun’s color/direction, not the volume’s. Enclosed volumes (no sky visibility) are unaffected by this.

**Shadow filters** use the hard volume AABB (not the blend shell). Set `OutsideCastShadowIn` to No so outdoor walls/props don’t darken an interior volume; set `InsideCastShadowOut` to No so interior blockers don’t shadow the courtyard outside.

---

## Ambient occlusion — `-ao`, `light_ao`, `light_ao_vol`

Cosine-weighted hemisphere AO baked into lightmaps in `FinalLightFace` (after direct + bounce).

**Enable via:** CLI `-ao` / `-ao_*`, point `light_ao` (map defaults), and/or brush `light_ao_vol` (local overrides). Combine freely.

### CLI

| Flag | Default | Effect |
|------|---------|--------|
| `-ao` | off | Enable AO pass |
| `-ao_samples N` | `16` | Rays per luxel (implies `-ao`; reduced with `-fast`) |
| `-ao_distance N` | `48` | Max ray length (implies `-ao`) |
| `-ao_strength N` | `1.0` | Darkening `0`–`8` (implies `-ao`) |
| `-ao_bias N` | `0.25` | Normal offset (implies `-ao`) |
| `-ao_denoise` | off | Edge-preserving bilateral denoise |
| `-ao_denoise_radius N` | `1` | Radius `1`–`4` |
| `-ao_denoise_strength N` | `1.0` | Blend `0`–`1` |

### Entity keys

Shared by `light_ao` and `light_ao_vol`: `Enabled`, `Samples`, `Distance`, `Strength`, `Bias`.

`light_ao` only: `Denoise`, `DenoiseRadius`, `DenoiseStrength`.  
`light_ao_vol` only: `BlendDistance` / `BlendMode` / `priority` (same idea as env vols).

Volumes soft-blend settings. `Enabled=No` carves AO out; `Enabled=Yes` can add AO when the map default is off. With `-gpu`, AO uses batched OpenCL occlusion when available.

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

## Soft sun (`SunSpreadAngle` / `-softsun`)

Replaces stock’s fixed 30 random rays:

- **Low-discrepancy cone** (golden-angle spiral) — smooth penumbras, less noise at large angles
- **Adaptive sample count** — scales with angle (~9 at 2°, up to ~40 at 50°)
- **Early-out** — fully lit / fully shadowed luxels stop after a short probe; only penumbra pays full cost

`0` = hard sun. Typical soft look: **0.5–3°**. Very large angles (e.g. 50°) still cost more, but less than stock and look cleaner.

---

## Cross-face bounce welding

When bounce is written into lightmaps, coplanar neighbor faces that share an edge contribute with **edge-distance falloff**. Distant patches no longer smear rectangular GI across the seam. Non-coplanar neighbors keep the stock neighbor splat.

Tunable with `-bounce_soft N` (default `1`; range `0.5`–`4`; `<1` tighter).

---

## Lightmap seam stitching

Stock VRAD filters every face's lightmap independently, so coplanar faces split by VBSP (grid splits, brush boundaries) can land on slightly different luxel values along the shared edge — a faint brightness step even on flat, evenly lit surfaces.

After `FinalLightFace`, CustomVRAD blends luxels near each shared edge of coplanar faces toward the neighbor's value at the same world position; at the edge both sides converge to the same average, removing the step. Applies per lightstyle and bump layer; displacements are skipped (they have their own edge rules).

**On by default.** Disable with `-nostitch`.

---

## GPU (`-gpu`)

Optional OpenCL path (needs a working OpenCL ICD; project links `OpenCL.lib` from `src/lib/public/x64`).

| Flag | Effect |
|------|--------|
| `-gpu` | Bounce gather + batched AO / sky occlusion BVH |
| `-gpu_maxtris N` | Optional BVH triangle cap (`0` = unlimited) |
| `-gpu_batch N` | Rays per dispatch (default **32768**; lower = safer vs TDR) |
| `-gpu_transfers` | Experimental GPU transfer rays — usually **slower**, not recommended |

Also used when present:

- Soft-sun / sky closest-hit batches
- Static-prop **indirect** ray pre-cull (skip BSP walk for sky / miss)

**Stability:** small ray batches, chunked uploads, device alloc checks, auto CPU fallback on OpenCL errors.  
**Throughput:** each CPU worker can use its own OpenCL queue + ray buffers (pool up to 64) so dispatches overlap instead of serializing on one queue.

Without `-gpu`, everything falls back to CPU.

---

## Radiosity / speed CLI

| Flag | Effect |
|------|--------|
| `-coarse` | Patch chop `8` — fewer patches, faster VisLeafs / bounce |
| `-maxtransfer N` | Skip patch transfers farther than N units |
| `-bounce_soft N` | Bounce luxel splat scale (see bounce weld) |
| `-nostitch` | Disable lightmap seam stitching across coplanar face splits |
| `-threads N` | Override thread count (`1`–`256`) |

Auto-detects logical processors (including >64 via processor groups). Work dispatch uses atomics so high thread counts scale better.

---

## Static prop lighting (`-StaticPropLighting`)

CustomVRAD speedups on top of stock prop vertex lighting:

- **4 vertices per SSE gather** for direct light (stock duplicated one vert across all lanes)
- **GPU bounce-ray culling** with `-gpu` (sky / miss rays skip the BSP lightmap walk)

Quality flags like `-StaticPropPolys` / `-TextureShadows` still apply and are expensive — drop them for preview compiles.

---

## Example commands

**Quality:**

```bat
PathToCustomVRAD\bin\vrad.exe -hdr -final -StaticPropLighting -textureshadows -ao -ao_samples 32 -gpu -game "PathToGarrysMod\garrysmod" "PathToMap\map"
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
