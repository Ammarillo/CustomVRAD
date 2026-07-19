# Source SDK 2013 - GMOD Tools X64 Mod

# DO NOT USE

---

##  Project Status - IMPORTANT

***After thinking about this project, I'm going to start over and go back to a CUDA base with a STABLE plan. I'll leave this repo open for those curious about the beginnings of the work, but there's little chance I'll finish the version with OpenCL. For now, I'm focusing on my studies, my GMOD map, and the other small projects on my git that need to be finished. Thanks for your patience and understanding :)***

---

## Project Summary

Source code for **Source SDK 2013 (Garry’s Mod, 64-bit)** - custom experimental build tools for:

- **VBSP**
- **VVIS**
- **VRAD**

The long-term goal is to explore **modernization, profiling, and acceleration** of Source Engine build tools while preserving:
- Perfect format compatibility
- Deterministic results
- Zero visual regressions

This repository is an **experimental research project**, not a drop-in replacement for official tools.

---

## Background & Motivation

This fork is based on **Source SDK 2013**, originally adapted from  
[Ficool2’s Source SDK 2013 fork](https://github.com/ficool2/source-sdk-2013),  
and modified specifically for **Garry’s Mod (64-bit)**.

The original ambition of this project was to:
- Explore **GPU-assisted visibility computation (VVIS)**
- Use **OpenCL** for broad GPU compatibility
- Reduce compile times on large, complex maps
- Preserve **strict correctness guarantees**

However, practical experimentation revealed that:
> **VVIS is extremely sensitive to implementation details**  
> Any optimization is meaningless unless the CPU version is first proven identical to Valve’s.

---


### GPU / OpenCL Work - ON HOLD

The OpenCL-based **VVIS_GPU** work is currently **paused**, not abandoned.

Reasons:
- CPU correctness has absolute priority
- Ficool2’s **VVIS++** already provides excellent CPU-side performance
- GPU acceleration must never compromise correctness

GPU work will resume only when:
- CPU baseline matches Valve exactly
- All differences are fully understood and documented
- GPU is used only for **safe, conservative pruning**

---

### VRAD — `light_env_vol`

CustomVRAD adds a compile-time brush entity that overrides sky/sun/ambient lighting inside a volume.  
Short classname (`light_env_vol`, 13 chars). Legacy `light_environment_volume` is still accepted by VRAD.

**Setup**
1. Keep one map `light_environment` (global default).
2. Create a brush textured with tools/toolstrigger (or nodraw), tie it to `light_env_vol`.
3. Set the same lighting keys as `light_environment` on the volume (`_light`, `_ambient`, `angles`, `pitch`, HDR keys, `SunSpreadAngle`).
4. Optional: `BlendDistance` (world units) — soft fade length using Perlin smootherstep (C2), with rounded corners via softmin (`0` = hard cut).
5. Optional: `BlendMode` — per volume: `0` Inside (fade inward from face), `1` Outside (fade outward from face), `2` Center (fade straddles face, default).
6. Optional: `priority` — when volume cores fully overlap, higher wins (ties → smaller AABB).
7. Optional: `BounceVolColor` — if Yes, radiosity bounce entering the volume from outside is recolored to this volume’s `_light` tint; if No (default), outside bounce color is kept.
8. Optional: `BounceVolBright` — only with (7)=Yes; if Yes, also scales bounce luminance by volume `_light` intensity vs the map `light_environment`; if No (default), luminance is preserved.

FGD: [`fgd/customvrad.fgd`](fgd/customvrad.fgd)

**Notes**
- Bounds use the brush **model AABB** (axis-aligned).
- Neighbor volumes use a soft Voronoi territory split so one volume’s outside halo cannot tint another volume’s side.
- Overlapping blend shells mix by weight; leftover weight goes to the default `light_environment`.
- Volume lights are not exported to engine worldlights (compile-only).

**Threading**
- Auto-detects logical processors (including >64 via processor groups) and uses up to **256** threads.
- Override with `-threads N` (clamped to 1–256).
- Work dispatch uses atomics (not a global lock) so high thread counts scale better.
- Volume lighting caches env weights per sample group (avoids re-walking volumes for every sky light).

### VRAD — speed options (`-gpu`, `-coarse`, `-maxtransfer`)

```
vrad.exe -gpu -coarse -maxtransfer 2048 -hdr -game "<gmod>\garrysmod" "<map>"
```

| Flag | Effect |
|------|--------|
| `-gpu` | OpenCL bounce gather + batched ambient/sky occlusion (BVH). Transfer rays stay on CPU SSE. |
| `-coarse` | Patch chop 8 (far fewer patches → faster VisLeafs/bounce). |
| `-bounce_soft N` | Bounce luxel splat scale (default **0.75** = tighter than stock; `1` = stock; range 0.5–4). |
| `-maxtransfer N` | Skip patch transfers farther than N units (helps large maps). |
| `-gpu_transfers` | Experimental GPU transfer rays — usually **slower**, not recommended. |

Requires a working OpenCL ICD. Links `OpenCL.lib` from `src/lib/public/x64`.

### VRAD — ambient occlusion (`-ao`, `light_ao`, `light_ao_vol`)

Optional cosine-weighted hemisphere AO baked into lightmaps during `FinalLightFace` (after direct + bounce, before `_minlight` / pack).

**Ways to enable**
1. CLI `-ao` / `-ao_*` (defaults for the compile)
2. Point entity `light_ao` — map-wide toggle + settings (overrides CLI defaults when present)
3. Brush entity `light_ao_vol` — local override (blend like `light_env_vol`)

```
vrad.exe -ao -ao_samples 32 -ao_distance 64 -ao_strength 0.85 -gpu -hdr -game "<gmod>\garrysmod" "<map>"
```

| Flag | Default | Effect |
|------|---------|--------|
| `-ao` | off | Enable AO pass. |
| `-ao_samples N` | 16 | Rays per luxel (implies `-ao`). Reduced automatically with `-fast`. |
| `-ao_distance N` | 48 | Max ray length in world units (implies `-ao`). |
| `-ao_strength N` | 1.0 | Darkening strength; may be **>1** to boost (max 8). Implies `-ao`. |
| `-ao_bias N` | 0.25 | Offset along surface normal to reduce self-hit (implies `-ao`). |
| `-ao_denoise` | off | Edge-preserving bilateral denoise on AO luxels. |
| `-ao_denoise_radius N` | 1 | Denoise radius 1..4 (implies denoise). |
| `-ao_denoise_strength N` | 1.0 | Denoise blend 0..1 (implies denoise). |

**Entities**

| Classname | Type | Role |
|-----------|------|------|
| `light_ao` | point | Global enable + Samples / Distance / Strength / Bias / Denoise |
| `light_ao_vol` | brush | Same AO keys + `BlendDistance` / `BlendMode` / `priority` |

FGD (single file): [`fgd/customvrad.fgd`](fgd/customvrad.fgd) — add in Hammer after your game FGD, then restart.

**Hammer tip:** `light_ao` is a **point** entity (Entity Tool → classname `light_ao`). `light_ao_vol` / `light_env_vol` are **brush** entities (Tie to Entity).

- Volumes soft-blend settings (samples use the max of contributors; distance/strength/bias are weight-averaged). Soft Voronoi keeps neighboring volume shells from fighting.
- `Enabled=No` on a volume carves out AO when the map default is on; a volume with `Enabled=Yes` can add AO when the default is off.
- With `-gpu`, AO uses batched OpenCL occlusion when available; otherwise CPU `TestLine`. Baked lightmaps only.

---

###  VBSP

Minor profiling and structural analysis only:
- Compile-time behavior
- I/O bottlenecks
- No functional changes planned for now

- Upgrades limits ?
  
---

## Development Activity Notice

Development is **slow, irregular, and priority-based**.

There may be:
- Long periods without commits
- Experimental branches that are later discarded
- Major refactors with no short-term payoff

The goal is to **preserve knowledge and experiments**, not to rush releases.

---

## Platform Support

- Windows only (for now)
- Linux not supported (may be explored later)

---

## Build Instructions (Windows)

### Requirements

- **Visual Studio 2022**
  - Desktop development with C++
  - MSVC v143
  - Windows 10 / 11 SDK
- **CMake** (recommended)
- **OpenCL SDK** (optional, currently unused)

### Build Notes

- Use **Release** configuration (**NOT Debug**)
- Debugging can still be done via *Local Windows Debugger*
- Tools are built as standalone executables

Compiled binaries will appear in:
```
/bin/
```

Example:
```
/bin/vvis_GPU.exe
```

### Runtime Dependencies

Keep this layout under `/bin/`:
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

**Important:** Do **not** use GMod `bin/win64/vphysics.dll` next to this build — it requires GMod’s `tier0` and will fail with `Unable to load vphysics DLL`. Use **`vphysics_stub.dll`** from GMod win64 (copied as `vphysics.dll`) instead.

---

## Tool Usage

Usage syntax remains identical to the original Source tools.

Example:
```
PathToYourClone\bin\vvis_GPU.exe -game "PathToGarrysMod\garrysmod" "PathToMap\map.bsp"
```

---

## References & Resources

- Valve Developer Wiki - Source SDK 2013  
  https://developer.valvesoftware.com/wiki/Source_SDK_2013

- Ficool2 - Source SDK 2013 fork  
  https://github.com/ficool2/source-sdk-2013

- Ficool2 - VVIS++, VRAD++, Tools  
  https://ficool2.github.io/HammerPlusPlus-Website/tools.html

- OpenCL Specification  
  https://www.khronos.org/opencl/

- Garry’s Mod Developer Wiki  
  https://wiki.facepunch.com/gmod/

---

## License

This project is licensed under the **SOURCE 1 SDK LICENSE**.  
See the [LICENSE](LICENSE) file for details.

All derived work, experiments, and modifications remain subject to the same non-commercial license.

---

## Author Notes

This repository reflects an **honest, technical exploration** of Source Engine tooling.

Progress is nonlinear.  
Mistakes are part of the process.  
Correctness always comes before speed.

> “Make it correct.  
> Then make it fast.  
> Never the other way around.”
