# CustomVRAD

Custom **VRAD** for **Garry’s Mod (64-bit)** / Source SDK 2013. Drop-in lighting compile with map entities and CLI options for environment volumes, baked ambient occlusion, and optional OpenCL acceleration.

Compatible lightmap output for the engine. Experimental — validate looks on your maps before shipping.

**Repo:** https://github.com/Ammarillo/CustomVRAD

---

## Features

| Feature | What it does |
|---------|----------------|
| `light_env_vol` | Brush volumes that override sky/sun/ambient (same keys as `light_environment`) |
| `light_ao` / `light_ao_vol` | Map-wide or local baked ambient occlusion in lightmaps |
| `-gpu` | OpenCL bounce gather + batched sky/ambient occlusion |
| `-coarse`, `-maxtransfer`, `-bounce_soft` | Faster / tunable radiosity |
| High thread counts | Auto-detects cores (incl. >64 via processor groups), up to **256** threads |

FGD for all custom entities: [`fgd/customvrad.fgd`](fgd/customvrad.fgd)

---

## Hammer setup

1. **Tools → Options → Game Configurations → Game Data Files**
2. Add `fgd/customvrad.fgd` **after** your game FGD (e.g. `garrysmod.fgd`)
3. Restart Hammer

| Classname | Type | How to place |
|-----------|------|----------------|
| `light_env_vol` | brush | Tie brush to entity (trigger/nodraw) |
| `light_ao` | **point** | Entity Tool → classname `light_ao` |
| `light_ao_vol` | brush | Tie brush to entity (trigger/nodraw) |

Legacy classname `light_environment_volume` is still accepted by VRAD for env volumes.

---

## `light_env_vol` — sky/sun override volumes

Compile-time brush that overrides sky/sun/ambient **inside** a volume. Outside (and in the blend shell) lighting mixes with the map’s `light_environment`. Volume lights are **not** exported as engine worldlights.

### Setup

1. Keep one map `light_environment` (global default).
2. Make a brush with `tools/toolstrigger` or nodraw → **Tie to Entity** → `light_env_vol`.
3. Set the same lighting keys as `light_environment`: `_light`, `_ambient`, `angles`, `pitch`, HDR keys, `SunSpreadAngle`.
4. Tune optional blend / bounce keys below.

### Keys

| Key | Default | Description |
|-----|---------|-------------|
| `_light` / `_ambient` | (standard) | Sun and ambient color + brightness |
| `_lightHDR` / `_ambientHDR` | `-1 -1 -1 1` | HDR overrides; leave default to use SDR |
| `_lightscaleHDR` / `_AmbientScaleHDR` | `1` | HDR scales |
| `pitch` | `0` | Overrides pitch in Angles |
| `SunSpreadAngle` | `0` | Soft sun shadows (degrees) |
| `BlendDistance` | `0` | Soft fade length in world units (`0` = hard cut). Uses Perlin smootherstep (C2); corners use softmin |
| `BlendMode` | `2` Center | `0` Inside · `1` Outside · `2` Center (fade straddles face) |
| `priority` | `0` | When cores fully overlap: higher wins; ties → smaller AABB |
| `BounceVolColor` | No | Yes = recolor inbound radiosity bounce to this volume’s `_light` hue |
| `BounceVolBright` | No | Only if BounceVolColor=Yes: also scale bounce luminance vs map `light_environment` |

### Behavior notes

- Bounds use the brush **model AABB** (axis-aligned).
- Neighbor volumes use a soft **Voronoi** split so one volume’s outside halo does not tint another’s side.
- Overlapping blend shells mix by weight; leftover weight goes to the default `light_environment`.
- Env weights are cached per sample group for threading performance.

---

## Ambient occlusion — `-ao`, `light_ao`, `light_ao_vol`

Optional cosine-weighted hemisphere AO baked into lightmaps during `FinalLightFace` (after direct + bounce, before `_minlight` / pack).

### Ways to enable

1. **CLI:** `-ao` / `-ao_*` (compile defaults)
2. **Point entity `light_ao`:** map-wide toggle + settings (overrides CLI defaults when present)
3. **Brush `light_ao_vol`:** local override (blend like `light_env_vol`)

You can combine them: CLI or `light_ao` for the map default, volumes for local stronger/weaker/off regions.

### CLI flags

| Flag | Default | Effect |
|------|---------|--------|
| `-ao` | off | Enable AO pass |
| `-ao_samples N` | `16` | Rays per luxel (implies `-ao`). Reduced automatically with `-fast` |
| `-ao_distance N` | `48` | Max ray length in world units (implies `-ao`) |
| `-ao_strength N` | `1.0` | Darkening; may be **>1** to boost (max `8`). Implies `-ao` |
| `-ao_bias N` | `0.25` | Offset along surface normal to reduce self-hit (implies `-ao`) |
| `-ao_denoise` | off | Edge-preserving bilateral denoise on AO luxels |
| `-ao_denoise_radius N` | `1` | Denoise radius `1..4` (implies denoise) |
| `-ao_denoise_strength N` | `1.0` | Denoise blend `0..1` (implies denoise) |

### Entity keys

**`light_ao` (point)** and **`light_ao_vol` (brush)** share:

| Key | Default | Description |
|-----|---------|-------------|
| `Enabled` | Yes | Master / local AO on-off |
| `Samples` | `16` | Hemisphere rays per luxel |
| `Distance` | `48` | Max ray length |
| `Strength` | `1.0` | Darkening (`0`–`8`) |
| `Bias` | `0.25` | Normal offset |

`light_ao` only (global face filter):

| Key | Default | Description |
|-----|---------|-------------|
| `Denoise` | No | Edge-preserving AO filter |
| `DenoiseRadius` | `1` | Luxel radius (`1` = 3×3, `2` = 5×5) |
| `DenoiseStrength` | `1.0` | Blend toward filtered AO (`0..1`) |

`light_ao_vol` only:

| Key | Default | Description |
|-----|---------|-------------|
| `BlendDistance` / `BlendMode` / `priority` | same idea as `light_env_vol` | Soft fade + overlap resolution |

### AO behavior notes

- Volumes soft-blend settings (samples use the **max** of contributors; distance/strength/bias are weight-averaged). Soft Voronoi keeps neighboring shells from fighting.
- `Enabled=No` on a volume carves out AO when the map default is on; `Enabled=Yes` can add AO when the default is off.
- With `-gpu`, AO uses batched OpenCL occlusion when available; otherwise CPU `TestLine`.
- Affects baked lightmaps only (not a runtime effect).

### Example

```bat
vrad.exe -ao -ao_samples 32 -ao_distance 64 -ao_strength 0.85 -ao_denoise -gpu -hdr -game "<gmod>\garrysmod" "<map>"
```

---

## Speed & GPU options

```bat
vrad.exe -gpu -coarse -maxtransfer 2048 -hdr -game "<gmod>\garrysmod" "<map>"
```

| Flag | Effect |
|------|--------|
| `-gpu` | OpenCL bounce gather + batched ambient/sky occlusion (BVH). Transfer rays stay on CPU SSE |
| `-coarse` | Patch chop `8` (fewer patches → faster VisLeafs/bounce) |
| `-bounce_soft N` | Bounce luxel splat scale (default **0.75** = tighter than stock; `1` = stock; range `0.5`–`4`) |
| `-maxtransfer N` | Skip patch transfers farther than N units (helps large maps) |
| `-gpu_transfers` | Experimental GPU transfer rays — usually **slower**, not recommended |

Requires a working **OpenCL ICD**. The project links `OpenCL.lib` from `src/lib/public/x64`.

### Threading

- Auto-detects logical processors (including >64 via processor groups), up to **256** threads.
- Override with `-threads N` (clamped to `1`–`256`).
- Work dispatch uses atomics (not a global lock) so high thread counts scale better.

---

## Typical compile commands

**Quality (HDR + AO + GPU):**

```bat
PathToCustomVRAD\bin\vrad.exe -hdr -final -StaticPropLighting -textureshadows -ao -ao_samples 32 -gpu -game "PathToGarrysMod\garrysmod" "PathToMap\map"
```

**Preview (fast):**

```bat
PathToCustomVRAD\bin\vrad.exe -hdr -fast -coarse -ao -ao_samples 8 -gpu -game "PathToGarrysMod\garrysmod" "PathToMap\map"
```

Stock VRAD flags (`-bounce`, `-final`, `-StaticPropLighting`, etc.) work as usual — run `vrad.exe` with no args for the full list.

---

## Build (Windows)

### Requirements

- Visual Studio 2022 (Desktop C++, MSVC v143, Windows 10/11 SDK)
- OpenCL SDK / ICD for `-gpu` (optional if you never use GPU)

### Build

1. Open `src/Source GPU compiles tool (L-I).sln`
2. Build **Release | x64** (not Debug)
3. Outputs land under `/bin/`

Relevant projects: `vrad_dll_win64`, `vrad_launcher_win64`.

### Runtime layout under `/bin/`

```
vrad.exe
vrad_dll.dll
tier0.dll
vstdlib.dll
vphysics.dll          <- use GMod's vphysics_stub.dll renamed/copied as vphysics.dll
vphysics_stub.dll
bin/x64/filesystem_stdio.dll
bin/x64/vphysics.dll  <- same stub copy
```

Do **not** place GMod `bin/win64/vphysics.dll` next to this build — it needs GMod’s `tier0` and fails with `Unable to load vphysics DLL`. Use **`vphysics_stub.dll`** from GMod win64 (copied as `vphysics.dll`) instead.

---

## Platform

- **Windows** only for now
- Linux not supported

---

## Lineage

Based on Source SDK 2013 tooling adapted for Garry’s Mod (64-bit), including work from [Ficool2’s Source SDK 2013 fork](https://github.com/ficool2/source-sdk-2013) and related compile-tool experiments. This repo focuses on **CustomVRAD** features above; older VVIS/OpenCL experiments are not the product focus here.

---

## License

**SOURCE 1 SDK LICENSE** — see [LICENSE](LICENSE). Derived work remains under the same non-commercial terms.

---

## References

- [Source SDK 2013](https://developer.valvesoftware.com/wiki/Source_SDK_2013)
- [Ficool2 — Source SDK 2013](https://github.com/ficool2/source-sdk-2013)
- [Ficool2 — Hammer++ tools](https://ficool2.github.io/HammerPlusPlus-Website/tools.html)
- [OpenCL](https://www.khronos.org/opencl/)
- [Garry’s Mod Wiki](https://wiki.facepunch.com/gmod/)
