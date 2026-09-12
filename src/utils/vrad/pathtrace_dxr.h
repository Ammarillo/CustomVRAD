//========= Copyright PathRAD contributors. ============//
//
// Path-traced lightmaps (-pathtrace / -dxr). Separate from the OpenCL -gpu bounce path.
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
extern int		g_nPathTraceBounces;	// indirect hops (-1 = auto; 0 = direct+sky only)
extern int		g_nPathTraceDevice;		// adapter index (-1 = default)
extern bool		g_bPathTraceDenoise;	// -pt_denoise
extern int		g_nPathTraceDenoiseRadius;	// Sakai spatial (1..4); OIDN/OptiX ignore
extern float	g_flPathTraceDenoiseStrength;	// blend 0..1 noisy->denoised
extern int		g_nPathTraceLuxelAA;	// luxel footprint AA grid 1..5 (1=off, 3=3x3)
extern int		g_nPathTraceLightSamples;	// local NEE samples (0=all lights)
extern int		g_nPathTraceEmitSamples;	// $vrad_emit area NEE samples (0=all tris)
extern int		g_nPathTracePropSamples;	// prop spp (0 = auto: max(8, world/4))
extern int		g_nPathTracePropBounces;	// prop bounces (-1 = auto: world; 0 = direct+sky)
extern int		g_nPathTracePropVertGrid;	// 0=per-vert; >0=8x8 tri atlas + closest gather (+denoise)
extern bool		g_bPathTraceSpectral;		// hero-wavelength Smits+CIE (default true)
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

// GPU-bake static prop samples. bLightmapQuality=true -> same spp/bounces/NEE as world luxels
// (for prop lightmap texels). false -> cheaper vertex defaults (-pt_prop_samples/bounces).
// bEndSession=false keeps the GPU bake session open for additional chunks (same quality).
// progressBase/progressTotal: overall stream progress (one \r line). total=0 -> no % (silent chunks).
bool PathTraceDXR_BakePropSamples( const struct PtGpuBakeLuxel *samples, unsigned nSamples,
								   struct PtGpuBakeResult *outResults, bool bLightmapQuality = false,
								   bool bEndSession = true,
								   unsigned progressBase = 0, unsigned progressTotal = 0,
								   const char *tagOverride = nullptr );

// Direct light only (0 bounces): local lights, sun, sky, emit surfaces — including
// $vrad_filter colored-glass transmission. Used for fog_volume LightBoost.
bool PathTraceDXR_BakeDirectOnlySamples( const struct PtGpuBakeLuxel *samples, unsigned nSamples,
										 struct PtGpuBakeResult *outResults, int spp = 64 );

// Close an open prop bake session with no leftover jobs (finish progress line + dark stats).
void PathTraceDXR_PropBakeClose( const char *tag, unsigned totalSamples );

// End GPU session without claiming stream complete (quality switch / suspend).
void PathTraceDXR_PropBakeSuspend( const char *tag );

// True when -pathtrace world bake succeeded and GPU baker can light props.
bool PathTraceDXR_CanBakeProps();

// Phase 2: register unique-model local shadow meshes + instances (before DeviceInit).
void PathTraceDXR_ClearPropInstances();
// shadowMatIndices: per-triangle index into shadow material DB (-1 = opaque). Optional.
void PathTraceDXR_RegisterPropModel( int modelIdx, const class Vector *verts, int nTris,
									 const int *shadowMatIndices = nullptr );
void PathTraceDXR_RegisterPropInstance( int propIndex, int modelIdx, const float xform[3][4] );

void PathTraceDXR_Shutdown();

#endif // PATHTRACE_DXR_H
