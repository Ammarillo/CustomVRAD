//========= Copyright CustomVRAD contributors. ============//
//
// Optional D3D12 + DXR path-traced lightmap baker (-pathtrace / -dxr).
// Independent of the OpenCL -gpu path.
//
//=============================================================================//

#ifndef PATHTRACE_DXR_H
#define PATHTRACE_DXR_H
#pragma once

class RayTracingEnvironment;

// CLI / state
extern bool		g_bPathTraceRequested;
extern bool		g_bPathTraceActive;		// true after successful bake this run
extern int		g_nPathTraceSamples;	// spp per luxel (0 = auto from quality)
extern int		g_nPathTraceBounces;	// max path depth (0 = auto)
extern int		g_nPathTraceDevice;		// adapter index (-1 = default)
extern bool		g_bPathTraceDenoise;	// -pt_denoise
extern int		g_nPathTraceDenoiseRadius;	// Sakai spatial (1..4); OIDN/OptiX ignore
extern float	g_flPathTraceDenoiseStrength;	// blend 0..1 noisy→denoised
extern int		g_nPathTraceLuxelAA;	// luxel footprint AA grid 1..5 (1=off, 3=3x3)
extern int		g_nPathTraceLightSamples;	// local NEE samples (0=all lights)
extern int		g_nPathTraceEmitSamples;	// $vrad_emit area NEE samples (0=all tris)
extern float	g_flPtLightRadius;		// -pt_lightradius: soft disk for light/light_spot (0=hard)
extern float	g_flPtLightPenumbra;	// -pt_lightpenumbra: CHSS growth scale
extern int		g_nPtSoftSamples;		// -pt_softsamples: max soft rays per NEE
extern int		g_nPtSoftMode;			// 0=direct only (fast), 1=all bounces
extern bool		g_bPathTraceGpu;		// GPU RayQuery luxel baker (default on when DXR ready)
extern bool		g_bPathTraceCpuForced;	// -pt_cpu: force SSE CPU integrator

#include "pathtrace_denoise.h"	// g_PathTraceDenoiser / PathTraceDenoiser_t

void PathTraceDXR_SetRequested( bool bRequested );
bool PathTraceDXR_IsRequested();
bool PathTraceDXR_IsActive();

// Capture tris before g_RtEnv.SetupAccelerationStructure() (geometry format).
void PathTraceDXR_CaptureScene( RayTracingEnvironment &rtEnv );

// Bake world-face lightmaps. On failure returns false (caller falls back to stock).
bool PathTraceDXR_BakeWorldFaces();

void PathTraceDXR_Shutdown();

#endif // PATHTRACE_DXR_H
