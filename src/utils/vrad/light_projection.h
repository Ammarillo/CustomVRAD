//========= Copyright CustomVRAD contributors. ============//
// light_spot ProjectedTexture - 2D planar (fit to outer cone) or cubemap envmap.
// Auto-detect from VTF; bake-only. IES (if set) is an angular mask on top.
//=============================================================================//
#ifndef VRAD_LIGHT_PROJECTION_H
#define VRAD_LIGHT_PROJECTION_H
#pragma once

#include "mathlib/vector.h"
#include <vector>

enum ProjKind_t
{
	PROJ_NONE = 0,
	PROJ_2D = 1,		// planar framed to outer cone
	PROJ_CUBE = 2,		// 6-face cubemap (omni)
	PROJ_SPHERE = 3,	// 2D spheremap / equirect / matcap tagged ENVMAP (omni)
};

// 2D planar framing relative to the outer cone.
enum ProjFrameMode_t
{
	PROJ_FRAME_FILL = 0,	// square sides touch the cone (corners clipped)
	PROJ_FRAME_FIT = 1,		// whole square inside the cone (corners on cone)
};

struct ProjTex;

// Find or load materials/<path>.vtf - auto-detects 2D vs cubemap.
const ProjTex *Proj_FindOrLoad( const char *pPath );

ProjKind_t Proj_Kind( const ProjTex *p );
int Proj_GpuLayer( const ProjTex *p ); // after Proj_AssignGpuLayers / BuildGpuArrays
bool Proj_IsEquirect( const ProjTex *p ); // spheremap uses latlong when wider than tall

// RGB multiplier (1,1,1 if none). lightToSample = direction from light toward sample.
// frameMode: FILL vs FIT (2D only; ignored for cubemaps).
Vector Proj_EvalRGB( const ProjTex *p, const Vector &beamDir, const Vector &rightDir,
					 const Vector &upDir, float outerConeCos,
					 const Vector &lightToSample, ProjFrameMode_t frameMode = PROJ_FRAME_FILL );

// Convenience for a direct light that already has basis + stopdot2 + frame mode.
struct directlight_t;
Vector Proj_EvalLightRGB( const directlight_t *dl, const Vector &lightToSample );

// True when projection is omnidirectional (cubemap or 2D spheremap/equirect).
bool Proj_IsOmniEnvmap( const directlight_t *dl );

// Assign GPU layers for textures referenced by active lights; fill CPU-side
// Texture2DArray blobs (one full image per layer - no 2D atlas packing).
// Cube faces are 6 consecutive layers per cubemap (face order = Source CUBEMAP_FACE_*).
struct ProjGpuArray_t
{
	int width = 1;
	int height = 1;
	int layers = 1; // planar: N; cubes: N*6
	std::vector<unsigned char> rgba; // layers * W * H * 4
};

void Proj_AssignGpuLayers();
void Proj_BuildGpuCookieArray( ProjGpuArray_t &out );
void Proj_BuildGpuCubeArray( ProjGpuArray_t &out );

#endif // VRAD_LIGHT_PROJECTION_H
