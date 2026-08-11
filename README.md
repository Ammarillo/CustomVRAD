# CustomVRAD

**A 64-bit lightmap compiler for Garry's Mod / Source SDK 2013.**

CustomVRAD is a drop-in replacement for Valve's VRAD that preserves engine-compatible HDR/LDR lightmap and BSP lighting output while extending the bake with path-traced global illumination, spectral transport, photometric spots, spatial volume overrides, physically motivated underwater media, and BSP-shipped volumetric fog. The implementation is experimental: validate appearance on target maps before shipping content.


| Resource              | Location                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Repository            | [https://github.com/Ammarillo/CustomVRAD](https://github.com/Ammarillo/CustomVRAD)                           |
| Entity definitions    | `[fgd/customvrad.fgd](fgd/customvrad.fgd)`                                                                   |
| Algorithmic monograph | `[docs/algorithms.tex](docs/algorithms.tex)` → `[docs/algorithms.pdf](docs/algorithms.pdf)`                  |
| Flag catalog / linter | `[tools/vrad_cfg_flags.json](tools/vrad_cfg_flags.json)`, `[tools/lint_vrad_cfg.py](tools/lint_vrad_cfg.py)` |
| Config preset         | `[configs/full.cfg](configs/full.cfg)`                                                                       |


Formal estimators, MIS weights, IOP fits, and coordinate conventions are stated in the algorithm report. This README is the operational companion: scope, authoring contract, CLI semantics, and build procedure.

---



## 1. Scope and bake contract



### 1.1 What is produced

CustomVRAD writes Source lightmaps (world faces; optionally static-prop vertex/texel lighting) and related BSP lighting data. Leaf ambient cubes, detail-prop lighting, and engine-exported worldlights remain on stock VRAD paths, with CustomVRAD volume-aware extensions where noted below.

### 1.2 Two mutually exclusive world-face pipelines


| Pipeline              | Activation                             | Role                                                                              |
| --------------------- | -------------------------------------- | --------------------------------------------------------------------------------- |
| **Classic radiosity** | Default, or pathtrace failure fallback | `BuildFacelights` + iterative patch gather; optional OpenCL (`-gpu`)              |
| **DXR path tracing**  | `-pathtrace` / `-dxr`                  | Unidirectional PathLi with NEE; replaces world-face direct + radiosity on success |


Exactly one pipeline owns world-face irradiance. Static props may share PathLi under `-StaticPropLighting`. Stock CLI (`-hdr`, `-final`, `-StaticPropLighting`, `-textureshadows`, …) remains valid; `vrad.exe` with no arguments prints Valve's help text.

### 1.3 Physical vs artistic operators

Operators fall into two classes:

- **Physical / physically motivated** -- linear RGB or hero-λ transport, Beer-Lambert media, PBRT-style area lights, IESNA tables, Kd downwelling. Prefer these for predictive bakes.
- **Artistic** -- `-bounce_boost`, `-bounce_chroma`, `-ao_force`, bounce-volume chroma, env-vol inbound Oklab tint. Documented explicitly; they are not unbiased estimators.

---



## 2. Feature inventory


| Feature                          | Kind          | Estimator / contract (summary)                                              |
| -------------------------------- | ------------- | --------------------------------------------------------------------------- |
| `light_env_vol`                  | Brush         | Local sun / sky / ambient override; bake-only (not exported as worldlights) |
| Leaf-cube volume sun             | Bake          | Sky-visible samples accumulate volume-blended sun into ambient cubes        |
| `light_ao` / `light_ao_vol`      | Point / brush | Multi-scale hemisphere AO; skipped under pathtrace GI unless `-ao_force`    |
| `light_absorb`                   | Brush         | Soft damp of bounce (optional direct)                                       |
| `light_bounce_vol`               | Brush         | Local `-bounce_boost` / `-bounce_chroma` / energy mode                      |
| `light_volume`                   | Point         | Soft-sphere origin sampling; exported as hard point light                   |
| `fog_volume` (+ blocker)         | Brush         | 3D light grid + screenspace fog packed into BSP (`lua_run`)                 |
| `water_light_vol`                | Brush         | Optional underwater IOP override (`-underwater` / `-uw_volume`)             |
| `light_spot` IES / projected VTF | Entity keys   | Bake-only photometry / gobo; engine cone keys unchanged for dynamics        |
| Soft sun                         | Bake          | Low-discrepancy cone + adaptive CHSS-style count                            |
| `-pathtrace` / `-dxr`            | CLI           | DXR PathLi (GPU RayQuery default; CPU SSE optional)                         |
| `-pt_spectral`                   | CLI           | Hero-wavelength GPU transport (default on); `-pt_nospectral` → RGB          |
| `-underwater` / `-uw_volume`     | CLI           | `underwater` = Beer-Lambert+Kd; `uw_volume` = free-flight volume PT         |
| `$vrad_emit*`                    | VMT           | Textured emission; pathtrace area-mesh NEE                                  |
| `$vrad_filter*`                  | VMT           | Colored thin-sheet transmission (pathtrace only)                            |
| `-gpu`                           | CLI           | OpenCL radiosity gather / batched AO (independent of DXR)                   |
| `-config`                        | CLI           | Text preset of flags                                                        |
| Seam stitch / edge pull          | Bake          | Coplanar luxel stitch (default on); edge sample inset (default on)          |
| Threading                        | Runtime       | Auto core detect (incl. >64); up to **256** workers                         |


---



## 3. Installation and Hammer

1. Build or deploy `bin\vrad.exe` + `bin\vrad_dll.dll` and required Valve DLLs ([§14](#14-build-windows)).
2. Point Hammer **Custom VRAD** (or expert compile) at this `vrad.exe`.
3. **Tools → Options → Game Configurations → Game Data Files** → add `fgd/customvrad.fgd` **after** `garrysmod.fgd` → restart Hammer.


| Classname                           | Type         |
| ----------------------------------- | ------------ |
| `light_env_vol`                     | Brush        |
| `light_ao`                          | Point entity |
| `light_ao_vol`                      | Brush        |
| `light_absorb`                      | Brush        |
| `light_bounce_vol`                  | Brush        |
| `light_volume`                      | Point entity |
| `fog_volume` / `fog_volume_blocker` | Brush        |
| `water_light_vol`                   | Brush        |


Legacy classname `light_environment_volume` is accepted for env volumes. Stock **VBSP** is sufficient: lighting volumes are compile-time. `fog_volume` is rewritten by VRAD into map-embedded `lua_run` (no addon).

---



## 4. Spatial volumes (shared membership model)

Brush volumes share a common soft AABB membership: soft-min signed distance, quintic smootherstep shell (`BlendDistance` / `BlendMode`), priority (higher wins; ties → smaller AABB), and soft Voronoi separation so outside halos do not tint neighbours. Shadow filters for env volumes use the **hard** AABB, not the blend shell.

### 4.1 `light_env_vol` -- local sky / sun / ambient

Overrides irradiance **inside** the volume; outside and in the blend shell mix with map `light_environment`. Keep one global `light_environment` as the default.


| Key                                           | Default      | Semantics                                       |
| --------------------------------------------- | ------------ | ----------------------------------------------- |
| `_light` / `_ambient`                         | (stock)      | Sun and ambient                                 |
| `_lightHDR` / `_ambientHDR`                   | `-1 -1 -1 1` | HDR overrides; default inherits SDR             |
| `_lightscaleHDR` / `_AmbientScaleHDR`         | `1`          | HDR scales                                      |
| `pitch`                                       | `0`          | Overrides pitch in Angles                       |
| `SunSpreadAngle`                              | `0`          | Soft sun cone (degrees)                         |
| `BlendDistance`                               | `0`          | Soft fade (`0` = hard cut)                      |
| `BlendMode`                                   | `2` Center   | `0` Inside · `1` Outside · `2` Center           |
| `priority`                                    | `0`          | Overlap resolution                              |
| `BounceVolColor` / `BounceVolBright`          | No           | Recolor / rescale bounce to volume `_light` hue |
| `OutsideCastShadowIn` / `InsideCastShadowOut` | Yes          | Cross-boundary sun/sky shadow filters           |


Legacy keys `OutsideCastShadow` / `InsideCastShadow` override the new names when present (warning emitted). Leaf ambient cubes accumulate volume-blended sun when samples see the sun through sky. The engine's exported `light_environment` worldlight remains a **single** global sun for dynamic RT lighting; leaf-cube fill approximates the volume look for ambient-lit dynamics only.

### 4.2 `light_absorb`


| Key              | Default  | Semantics                  |
| ---------------- | -------- | -------------------------- |
| `Strength`       | `0.5`    | Absorb weight ∈ [0,1]      |
| `AbsorbDirect`   | No       | Also damp direct at luxels |
| `AbsorbBounce`   | Yes      | Damp radiosity gather      |
| Blend / priority | (shared) | Soft membership            |




### 4.3 `light_bounce_vol`

Local artistic radiosity controls soft-blended with CLI `-bounce_boost` / `-bounce_chroma` / `-energy`.


| Key            | Default        | Semantics                             |
| -------------- | -------------- | ------------------------------------- |
| `BounceBoost`  | `-1` (inherit) | Bounce scale ∈ [0,16]                 |
| `BounceChroma` | `-1` (inherit) | Early-bounce Oklab saturation ∈ [0,8] |
| `EnergyMode`   | Inherit        | Local cavity damp on/off              |


Use with `-texbounce`. Pathtrace ignores `-bounce_chroma` (non-physical); boost may still affect pathtrace albedo/throughput via volumes.

### 4.4 `light_volume` -- soft sphere point light

Same keys as stock `light`, plus `Radius` (default **16**). Bake samples origins with low-discrepancy points inside the sphere (omnidirectional soft penumbra). Sample count scales with radius (~16 at default, cap 40; reduced under `-fast`). Exported to the engine as a **hard** point light at the centre.

### 4.5 `fog_volume` -- BSP-shipped volumetric fog

Bakes a 3D irradiance grid into the BSP pak and packs a GMod screenspace fog draw (**no addon**).

1. Brush → Tie to Entity → `fog_volume` (`tools/toolstrigger`).
2. Compile fog shaders once ([§4.5.1](#451-shaders)).
3. VRAD writes `materials/maps/<map>/fog_volume_<hammerid>.vtf/.vmt`, shared `fog_noise.vtf`, packs `shaders/fxc/cvrad_fog_*.vcs`, and rewrites the entity to `lua_run`.


| Key                            | Default       | Semantics                                   |
| ------------------------------ | ------------- | ------------------------------------------- |
| `Enabled` / `StartEnabled`     | Yes           | Bake skip / runtime `cvf_en`                |
| `FogColor`                     | `200 220 255` | Tint                                        |
| `Density`                      | `0.02`        | Optical density per world unit              |
| `LightBoost`                   | `1.0`         | Direct-light multiplier in fog (0-16)       |
| `Anisotropy`                   | `0`           | HG g ∈ [−1,1]                               |
| `GridSpacing`                  | `32`          | Grid step; ≤ **64³** samples                |
| `StepCount`                    | `32`          | Raymarch steps (8-128)                      |
| `BlendDistance` / `BlendMode`  | `0` / Center  | Soft edge; Outside/Center expand bake+march |
| `NoiseScale` / `NoiseCoverage` | `0` / `0.45`  | FBM atlas (`0` = off)                       |
| `WindSpeed` / `WindDir`        | `8` / `0`     | World-space scroll; yaw quantized 45°       |


**Live tweak (BSP-only):** `cvf_en`, `cvf_de`, `cvf_ns`, `cvf_cov`, `cvf_wspd`, `cvf_wyaw`, `cvf_bd`. FogColor / lighting still need a rebake.

`fog_volume_blocker`**:** soft carve into overlapping fog (atlas alpha); rewritten to `info_null`.

#### 4.5.1 Shaders

```powershell
.\shaders\build_shaders.ps1 -ShaderCompile C:\path\to\ShaderCompile.exe
```

Outputs: `shaders/fxc/cvrad_fog_*.vcs`. **Caveat:** clients/servers that block map client Lua will not draw fog.

---



## 5. Underwater lighting (physically based bake)

Auto-detects BSP water via `leafWaterData` / `CONTENTS_WATER`. Optical properties (IOPs) are fit from the water surface VMT (`$fogenable`, `$fogcolor`, `$fogend`) or optional `$vrad_uw_*` / Jerlov presets. Brush `water_light_vol` overrides overlapping bodies.


| Flag               | Mode  | Estimator                                                              |
| ------------------ | ----- | ---------------------------------------------------------------------- |
| `-underwater`      | **A** | Beer-Lambert segment transmittance + Kd downwelling for sun/sky        |
| `-uw_volume`       | **C** | Homogeneous free-flight volume path tracing (implies underwater media) |
| `-uw_off`          | --    | Force disable                                                          |
| `-uw_maxscatter N` | C     | Cap medium scatter events (default **8**, max **64**)                  |


**Authoring notes (important):**

- Source `$fogend` is screenspace fog, not lightmap optical depth. The bake applies an extinction scale s_{\mathrm{bake}}=0.28, scatter fraction 0.45, and \mathbf{k}_d = 0.35\boldsymbol{\sigma}_t, with a per-channel Kd floor of 0.04.
- Mode A applies **either** segment T (local lights / path segments) **or** Kd (sky/sun irradiance)--not both stacked.
- Mode C accepts free-flight scatters only **inside** the water AABB (no fake boundary albedo); soft sky veiling reinjects ambient fill.
- Optional VMT: `$vrad_uw_jerlov`, `$vrad_uw_sigma_a`, `$vrad_uw_sigma_s`, `$vrad_uw_g`.

CPU + GPU pathtrace implement both modes; Mode A also scales classic direct gathering when pathtrace is off. See algorithms report § Underwater media.

---



## 6. Ambient occlusion

Multi-scale hemisphere ambient obscurance in `FinalLightFace`.

**Pathtrace policy:** with `-pathtrace` and `-pt_bounces` N>0, post-multiply AO is **skipped** (occlusion is already in PathLi). Use `-ao_force` only for artistic double-darkening. With N=0 (direct+sky), AO still applies as a stand-in for missing GI.


| Flag                     | Default | Effect                       |
| ------------------------ | ------- | ---------------------------- |
| `-ao`                    | off     | Enable AO                    |
| `-ao_force`              | off     | Apply under pathtrace GI     |
| `-ao_samples N`          | `16`    | Rays / luxel (implies `-ao`) |
| `-ao_distance N`         | `48`    | Ray length                   |
| `-ao_strength N`         | `1.0`   | Darkening ∈ [0,8]            |
| `-ao_bias N`             | `0.25`  | Normal offset                |
| `-ao_denoise`            | off     | Bilateral denoise            |
| `-ao_denoise_radius N`   | `1`     | ∈ [1,4]                      |
| `-ao_denoise_strength N` | `1.0`   | Blend ∈ [0,1]                |


Entities `light_ao` / `light_ao_vol` share `Enabled`, `Samples`, `Distance`, `Strength`, `Bias`. Denoise keys on `light_ao` only; blend/priority on `light_ao_vol`. With `-gpu`, AO may use batched OpenCL.

---



## 7. Photometric spots and projected textures

Bake-only extensions on stock `light_spot`. Engine worldlights keep cone keys for dynamic lighting.

### 7.1 IESNA (`IES`)


| Key               | Default | Semantics                                                  |
| ----------------- | ------- | ---------------------------------------------------------- |
| `IES`             | (empty) | LM-63 path under `garrysmod/IES/` (also `IES/maps/<map>/`) |
| `IESScale`        | `1`     | Peak-normalized table scale                                |
| `IESBrightness`   | `-1`    | Bake brightness override (`-1` = stock `_light`)           |
| `IESMaxIntensity` | `0`     | Per-luxel RGB cap (`0` = off)                              |


When `IES` is set, `_inner_cone` / `_cone` / `_exponent` are ignored for the bake. Distance attenuation unchanged. **TILT=NONE** only. Classic + pathtrace (CPU/GPU).

### 7.2 `ProjectedTexture`

Material under `materials/`. Auto classification:


| Mode             | Detection       | Domain                            |
| ---------------- | --------------- | --------------------------------- |
| Planar 2D        | Ordinary 2D VTF | Outer cone (`fill` / `fit`)       |
| Envmap / cubemap | Cubemap VTF     | Full sphere; entity angles rotate |


Projected RGB multiplies intensity; with IES, the table is an angular mask. Cubemap UVs follow Direct3D / Source `CUBEMAP_FACE_*` (Z-up). Algorithms report § Photometric spots.

---



## 8. Soft sun

Replaces stock's fixed 30 random cone rays:

- Low-discrepancy golden-angle spiral  
- Adaptive count (~9 at 2°, ~40 at 50°)  
- Early-out when fully lit / fully shadowed

`SunSpreadAngle` / `-softsun`: `0` = hard; typical soft look **0.5-3°**.

---



## 9. Lightmap sample quality



### 9.1 Cross-face bounce welding (`-bounce_weld`, off by default)

Coplanar neighbours sharing an edge contribute with edge-distance falloff when bounce is written. Tunable with `-bounce_soft N` (default `1`, range 0.5-4).

### 9.2 Seam stitching (on by default; `-nostitch` disables)

After `FinalLightFace`, luxels near shared edges of coplanar faces blend toward the neighbour value at the same world position. Per lightstyle and bump layer; displacements skipped.

### 9.3 Edge pull (on by default; `-noedgepull` disables)

Pulls edge luxel samples inward (`-edgepull [N]`, default **0.5** luxels) to reduce black strips on thin walls / jambs.

---



## 10. Materials



### 10.1 Textured emission (`$vrad_emit*`)


| Key                                   | Default          | Semantics                         |
| ------------------------------------- | ---------------- | --------------------------------- |
| `$vrad_emit`                          | `0`              | Enable (optional if strength > 0) |
| `$vrad_emitstrength`                  | `200` if emit-on | Intensity scaler                  |
| `$vrad_emitdensity`                   | `1`              | Optional finer patch chop         |
| `$vrad_emitmask`                      | unset            | Greyscale mask                    |
| `$vrad_emitmap` / `$vrad_emissivemap` | unset            | Colour map; else `$basetexture`   |


Under `-pathtrace`: fan-triangulated area mesh, PBRT-style NEE (`-pt_emit_samples`, default **64**; `0` = all tris). Classic: `emit_surface` proxies + self-illum. Worldlights cap **65536**.

### 10.2 Colored glass (`$vrad_filter*`)

Pathtrace-only thin-sheet transmission. Ordinary `$translucent` does **not** tint light unless `$vrad_filter 1`.


| Key                     | Default | Semantics                 |
| ----------------------- | ------- | ------------------------- |
| `$vrad_filter`          | `0`     | Enable                    |
| `$vrad_filtermap`       | unset   | Else `$basetexture`       |
| `$vrad_filterstrength`  | `1`     | Oklab lerp white → filter |
| `$vrad_filteropacity`   | `0.05`  | Flat absorption           |
| `$vrad_filterthickness` | `1`     | T^{\mathrm{thickness}}    |
| `$vrad_filter_onesided` | `0`     | Front-only if `1`         |


Stacked panes use Oklab (L multiply, a,b add); energy transport stays linear RGB/λ. Up to 64 filter hits; sheet entrance/exit dedup within 12 units.

---



## 11. Path tracing (`-pathtrace` / `-dxr`)

Unidirectional Lambertian integrator with NEE at every vertex. Default backend: **GPU RayQuery** (`-pt_gpu`). `-pt_cpu` forces the SSE reference (RGB only; ignores spectral).

**Spectral (GPU default):** `-pt_spectral` -- four hero wavelengths, Smits RGB↔SPD, CIE 1931 → linear sRGB. `-pt_nospectral` for linear RGB. Firefly clamp is luminance-preserving.

**Common flags:** `-pt_samples`, `-pt_bounces` (N = indirect hops after luxel; depth =N+1; N=0 ⇒ direct+sky), `-pt_aa`, `-pt_emit_samples`, `-pt_denoise` / `-pt_denoiser` (`oidn`  `optix`  `sakai`), soft-shadow CHSS (`-pt_lightradius`, `-pt_lightpenumbra`, `-pt_softsamples`, `-pt_softmode`). Full table: [§13](#13-vrad-cli-reference).

Needs a DXR-capable GPU. Prefer `-config full` with pathtrace flags uncommented.

```bat
vrad.exe -config full -game "<gmod>\garrysmod" "<map>"
```



### 11.1 Static props under pathtrace

With `-StaticPropLighting`, GPU PathLi bakes prop lightmap texels and a virtual per-triangle vertex lattice (`-pt_prop_vertgrid`, default **4**; `0` = one sample/vert). Override spp/bounces with `-pt_prop_samples` / `-pt_prop_bounces`. Classic prop path: 4-wide SSE direct + optional GPU bounce cull.

---



## 12. Classic radiosity and OpenCL


| Flag                                                                 | Role                                                          |
| -------------------------------------------------------------------- | ------------------------------------------------------------- |
| `-coarse` / `-adaptivechop` / `-chop` / `-maxchop`                   | Patch density                                                 |
| `-texbounce` / `-texbounce_clean`                                    | Textured albedo; Oklab near-grey chroma kill                  |
| `-energy` / `-cavity` / `-noenergy`                                  | Opt-in cavity-damped gather (default = stock Valve)           |
| `-bounce_boost` / `-bounce_chroma` / `-bounce_weld` / `-bounce_soft` | Artistic / weld controls                                      |
| `-maxtransfer N`                                                     | Cull far patch transfers                                      |
| `-gpu` / `-gpu_batch` / `-gpu_maxtris`                               | OpenCL gather; sky occlusion stays CPU with `-textureshadows` |


`-gpu` and `-pt_gpu` are **independent** backends. OpenCL ICD ships with GPU drivers; not bundled. Errors fall back to CPU.

---



## 13. VRAD CLI reference

Complete flag list (Valve + CustomVRAD). Bold = typical defaults. Pass `-game` / map path on the Hammer line -- **never** inside `-config` files. Authoritative toggle sheet: `[configs/full.cfg](configs/full.cfg)`.

### Launcher / paths


| Parameter                               | Description                                                   |
| --------------------------------------- | ------------------------------------------------------------- |
| `-game` / `-vproject`                   | Game directory                                                |
| `-insert_search_path`                   | Extra search path                                             |
| `-config` / `-cfg <file>`               | Load preset (`file`, `.cfg`, `.txt`, or `vrad.exe\configs\…`) |
| `-novconfig` / `-steam` / `-allowdebug` | Tooling                                                       |




### Quality / mode


| Parameter                   | Description                                                                    |
| --------------------------- | ------------------------------------------------------------------------------ |
| `-hdr` / `-ldr`             | Lightmap format                                                                |
| `-final`                    | High quality (= `-extrasky 16`)                                                |
| `-fast` / `-fastambient`    | Reduced sampling                                                               |
| `-extrasky N` / `-bounce N` | Sky rays / radiosity rounds (**100**; ignored for world if pathtrace succeeds) |




### Threading / process


| Parameter                                                                    | Description                      |
| ---------------------------------------------------------------------------- | -------------------------------- |
| `-threads N`                                                                 | **1-256** (default = auto cores) |
| `-low` / `-verbose` / `-v` / `-StopOnExit` / `-FullMinidumps` / `-rederrors` | Process / log                    |




### Static props / detail / shadows


| Parameter                                                         | Description                 |
| ----------------------------------------------------------------- | --------------------------- |
| `-StaticPropLighting` / `-StaticPropPolys` / `-StaticPropNormals` | Prop bake / debug           |
| `-OnlyStaticProps` / `-nossprops`                                 | Prop-only / no self-shadow  |
| `-textureshadows`                                                 | Alpha textures cast shadows |
| `-nodetaillight` / `-onlydetail` / `-noskyboxrecurse`             | Detail / skybox             |




### GPU (OpenCL radiosity)


| Parameter                         | Description                             |
| --------------------------------- | --------------------------------------- |
| `-gpu`                            | OpenCL bounce gather (+ batched AO)     |
| `-gpu_maxtris N` / `-gpu_batch N` | BVH cap / rays per dispatch (**32768**) |




### Path tracing -- DXR


| Parameter                                                                                         | Description                                                                              |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `-pathtrace` / `-dxr`                                                                             | Enable PathLi baker                                                                      |
| `-pt_gpu` / `-pt_cpu`                                                                             | GPU (default) / CPU SSE                                                                  |
| `-pt_samples N`                                                                                   | spp (CLI default **4** / **2** fast / **8** final; max **4096**; configs often 256-1024) |
| `-pt_bounces N`                                                                                   | Indirect hops (**3**; **1** fast; **0** = direct+sky; max **16**)                        |
| `-pt_prop_samples` / `-pt_prop_bounces` / `-pt_prop_vertgrid`                                     | Prop spp / hops / vert lattice (**4**)                                                   |
| `-pt_aa N`                                                                                        | Luxel footprint grid **1-5** (default **3**)                                             |
| `-pt_lights N`                                                                                    | Local NEE samples (`0` = all)                                                            |
| `-pt_emit_samples N`                                                                              | Area-emit NEE (**64**; `0` = all tris)                                                   |
| `-pt_lightradius` / `-pt_lightpenumbra` / `-pt_softsamples` / `-pt_softmode`                      | Soft shadows / CHSS                                                                      |
| `-pt_spectral` / `-pt_nospectral`                                                                 | Hero-λ (default on GPU) / RGB                                                            |
| `-pt_device N`                                                                                    | DXGI adapter index                                                                       |
| `-pt_denoise` / `-pt_nodennoise` / `-pt_denoiser` / `-pt_denoise_radius` / `-pt_denoise_strength` | Denoise                                                                                  |




### Underwater


| Parameter          | Description                   |
| ------------------ | ----------------------------- |
| `-underwater`      | Mode A: Beer-Lambert + Kd     |
| `-uw_volume`       | Mode C: free-flight volume PT |
| `-uw_off`          | Force disable                 |
| `-uw_maxscatter N` | Medium scatter cap (**8**)    |




### Ambient occlusion


| Parameter                                                                          | Description                |
| ---------------------------------------------------------------------------------- | -------------------------- |
| `-ao` / `-ao_force` / `-ao_samples` / `-ao_distance` / `-ao_strength` / `-ao_bias` | AO enable / force / params |
| `-ao_denoise` / `-ao_denoise_radius` / `-ao_denoise_strength`                      | AO denoise                 |




### Radiosity / bounce


| Parameter                                                            | Description             |
| -------------------------------------------------------------------- | ----------------------- |
| `-texbounce` / `-texbounce_clean` / `-texbounce_noclean`             | Textured bounce         |
| `-coarse` / `-adaptivechop` / `-chop` / `-maxchop`                   | Patch chop              |
| `-bounce_soft` / `-bounce_boost` / `-bounce_chroma` / `-bounce_weld` | Splat / artistic / weld |
| `-energy` / `-noenergy` / `-valve` / `-cavity`                       | Cavity damp             |
| `-maxtransfer N`                                                     | Far transfer cull       |




### Lightmap samples / seams


| Parameter                                                                                  | Description                      |
| ------------------------------------------------------------------------------------------ | -------------------------------- |
| `-edgepull [N]` / `-noedgepull`                                                            | Edge inset (default on, **0.5**) |
| `-nostitch`                                                                                | Disable seam stitch              |
| `-centersamples` / `-noextra` / `-debugextra` / `-dlightmap` / `-luxeldensity` / `-smooth` | Stock sample controls            |




### Sun / sky / displacements


| Parameter                                                                          | Description                                       |
| ---------------------------------------------------------------------------------- | ------------------------------------------------- |
| `-softsun N`                                                                       | Soft sun degrees (prefer entity `SunSpreadAngle`) |
| `-LargeDispSampleRadius` / `-dispchop` / `-disppatchradius` / `-maxdispsamplesize` | Displacement                                      |




### Extra / debug / MPI


| Parameter                                   | Description                 |
| ------------------------------------------- | --------------------------- |
| `-lights` / `-dump*` / `-loghash` / `-mpi*` | Extra lights / debug / VMPI |




### Debug-build only (`ALLOWDEBUGOPTIONS`)

`-scale`, `-ambient`, `-dlight`, `-sky`, `-notexscale`, `-coring` -- ignored in Release.

---



## 14. Config presets, lint, and editor support

```bat
-config "PathToCustomVRAD\configs\full" -game $gamedir $path\$file
```


| Preset   | File                                   | Role                                                 |
| -------- | -------------------------------------- | ---------------------------------------------------- |
| **full** | `[configs/full.cfg](configs/full.cfg)` | Complete commented flag sheet -- uncomment to enable |


**Format:** one flag (or `flag value`) per line; `#` / `//` comments. Do **not** embed `-game`, `-config`, or the map path. CLI args after `-config` override the file.

**Path lookup:** `path`, `path.cfg`, `path.txt`, then beside `vrad.exe` and `vrad.exe\configs\`.

**Lint:**

```powershell
python tools/lint_vrad_cfg.py --style configs
```

Validates arity, enums, banned Hammer-only flags, and conflicts. Catalog: `tools/vrad_cfg_flags.json`. Cursor/VS Code task: **Lint VRAD configs**.

**Syntax highlighting:**

```powershell
python tools/install_vrad_cfg_highlight.py
```

Then **Developer: Reload Window**. Grammar: `tools/vscode-vrad-cfg/`; colours: `.vscode/settings.json`.

---



## 15. Example commands

**Config-driven (recommended):**

```bat
PathToCustomVRAD\bin\vrad.exe -config "PathToCustomVRAD\configs\full" -game $gamedir $path\$file
```

**Quality radiosity:**

```bat
PathToCustomVRAD\bin\vrad.exe -hdr -final -StaticPropLighting -textureshadows -texbounce -ao -ao_samples 32 -gpu -game "PathToGarrysMod\garrysmod" "PathToMap\map"
```

**Pathtrace:**

```bat
vrad.exe -hdr -pathtrace -pt_samples 256 -pt_bounces 6 -pt_aa 2 -StaticPropLighting -textureshadows -game "<gmod>\garrysmod" "<map>"
```

**Underwater (attenuation):**

```bat
vrad.exe -config full -underwater -game "<gmod>\garrysmod" "<map>"
```

---



## 16. Build (Windows)

**Requires:** Visual Studio 2022 (Desktop C++, MSVC v143, Windows 10/11 SDK). OpenCL only if using `-gpu`.

**Recommended:** `.\build.ps1` at repo root -- full rebuild of `vrad_dll`, MSBuild auto-select, verifies `bin\vrad_dll.dll`. Use `.\build.ps1 -Incremental` only for iteration.

Manual: open `src/Source GPU compiles tool (L-I).sln` → **Release | x64**.

**Runtime next to** `vrad.exe`**:**

```
vrad.exe
vrad_dll.dll
tier0.dll
vstdlib.dll
vphysics.dll          ← GMod vphysics_stub.dll renamed
bin/x64/filesystem_stdio.dll
bin/x64/vphysics.dll  ← same stub
```

Do **not** use GMod `bin/win64/vphysics.dll` (wrong `tier0`). **Platform:** Windows only.

---



## 17. License and provenance

**SOURCE 1 SDK LICENSE** -- see [LICENSE](LICENSE). Derived work remains under the same non-commercial terms.

Based on Source SDK 2013 tooling adapted for Garry's Mod (64-bit), including work from [Ficool2's Source SDK 2013 fork](https://github.com/ficool2/source-sdk-2013). Algorithms and defaults are as implemented in the source tree; `vrad.exe` help and `configs/full.cfg` define shipping presets when they diverge from older prose.