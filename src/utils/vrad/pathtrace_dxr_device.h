//========= Copyright CustomVRAD contributors. ============//
// Internal DXR device API (shared by pathtrace modules).
//=============================================================================//
#ifndef PATHTRACE_DXR_DEVICE_H
#define PATHTRACE_DXR_DEVICE_H
#pragma once

#include "mathlib/vector.h"
#include <stdint.h>

bool PathTraceDXR_DeviceInit( int adapterIndex );
void PathTraceDXR_DeviceShutdown();
bool PathTraceDXR_DeviceReady();

// Batched closest-hit. dirs should be unit or any length (tmax is absolute distance).
// outHit[i]=1 on triangle hit; outFlags = TRACE_ID_* from scene.
// outNormal may be NULL; when non-NULL, receives geometric shading normal (unit) on hit.
// Uses DXR RayQuery for large batches when the device is ready; else SSE Trace4Rays.
bool PathTraceDXR_TraceClosest( const Vector *origins, const Vector *dirs,
								const float *tmins, const float *tmaxs,
								float *outT, unsigned int *outFlags, unsigned int *outHit,
								int nRays, Vector *outNormal = NULL );

// ---------------------------------------------------------------------------
// Full GPU luxel path baker (RayQuery integrator). Hard NEE + cosine GI.
// ---------------------------------------------------------------------------
enum PtGpuLightType : int
{
	PT_GPU_LIGHT_POINT = 0,
	PT_GPU_LIGHT_SPOT = 1,
	PT_GPU_LIGHT_SKY = 2,
	PT_GPU_LIGHT_SURFACE = 3,
	PT_GPU_LIGHT_SKYAMB = 4,
};

struct PtGpuBakeLight
{
	float	origin[3];
	float	type;			// PtGpuLightType
	float	intensity[3];
	float	pad0;
	float	dir[3];			// emission normal, or sun axis toward scene
	float	stopdot;
	float	stopdot2;
	float	exponent;
	float	constant_attn;
	float	linear_attn;
	float	quadratic_attn;
	float	fadeStart;
	float	fadeEnd;
	float	capDist;
	float	areaR2;
	float	sunExtent;		// sin(angle)
	float	volumeRadius;
	float	power;			// for CDF / power sampling (locals)
	int		facenum;
	int		envId;
	int		isLocal;		// 1 = point/spot (power-sampled pool)
	int		pad1;
};

struct PtGpuBakeLuxel
{
	float		pos[3];
	float		luxelWorld;
	float		normal[3];
	uint32_t	seed;
	float		axisU[3];
	float		padU;
	float		axisV[3];
	float		padV;
	int			faceNum;
	int			aaN;		// 1..5
	int			skipPropIndex;	// -1 = allow self-shadow; >=0 skip TRACE_ID_STATICPROP|id
	int			pad1;
};

struct PtGpuBakeResult
{
	float	radiance[3];
	float	sunAmt;
};

struct PtGpuBakeParams
{
	uint32_t	spp;
	uint32_t	bounces;
	uint32_t	lightSamples;	// 0 = all locals
	uint32_t	softSamplesMax;	// cap soft rays (GPU uses hard if 0)
	float		occludeBias;
	float		maxTraceLen;
	float		fireflyCap;
	float		penumbraScale;
	uint32_t	softMode;		// 0=direct only
	uint32_t	triCount;
	float		skyAmb[3];		// pre-summed emit_skyambient (no volumes)
	uint32_t	emitTriCount;	// $vrad_emit textured area tris (0 = none)
	uint32_t	emitSamples;	// power-sampled area NEE draws per vertex
	uint32_t	envVolCount;	// light_env_vol count (0 = no volume blend)
	float		defaultBounceIntensity; // for BounceVolBright scale
	uint32_t	bounceVolCount;	// light_bounce_vol count
	float		cliBounceBoost;
	float		cliBounceChroma;
	uint32_t	spectralMode;	// 1 = hero-wavelength Smits+CIE
};

// GPU copy of LightEnvVolumeInfo_t (AABB + blend + shadow + bounce tint).
struct PtGpuEnvVol
{
	float	mins[3];
	float	blendDistance;
	float	maxs[3];
	float	blendMode;		// LightEnvBlendMode_t as float
	float	priority;
	float	volumeSize;
	float	envId;			// 1..N
	float	flags;			// bit0 OutsideCast, bit1 InsideCast, bit2 BounceVolColor, bit3 BounceVolBright
	float	bounceTint[3];
	float	bounceIntensity;
};

// GPU emit triangle (matches HLSL packing helpers).
struct PtGpuEmitTri
{
	float	v0[3]; float area;
	float	v1[3]; float power;
	float	v2[3]; float facenum;
	float	n[3];  float padN;
	float	e0[3]; float pad0;
	float	e1[3]; float pad1;
	float	e2[3]; float pad2;
};

// Upload lights + per-triangle albedo once per bake. albedoRGB = nTris*3 floats.
// filterMeta = nTris*5 float4s; filterArrayRGBA = Texture2DArray layers (W*H*4*layers).
// emitTris/cdf optional ($vrad_emit area mesh); pass nullptr/0 if unused.
bool PathTraceDXR_GpuBakeBegin( const PtGpuBakeLight *lights, uint32_t nLights,
								const float *albedoRGB, uint32_t nTris,
								const PtGpuBakeParams &params,
								const PtGpuEmitTri *emitTris = nullptr, uint32_t nEmitTris = 0,
								const float *emitCdf = nullptr,
								const float *filterMeta = nullptr,
								const unsigned char *filterArrayRGBA = nullptr,
								uint32_t filterArrayW = 0, uint32_t filterArrayH = 0,
								uint32_t filterArrayLayers = 0 );

// spp / sampleOffset for multi-pass high-spp bakes (chunked for GPU occupancy).
void PathTraceDXR_GpuBakeConfigurePass( uint32_t spp, uint32_t sampleOffset );

// Bake a batch of luxels. outResults must hold nLuxels entries.
bool PathTraceDXR_GpuBakeLuxels( const PtGpuBakeLuxel *luxels, uint32_t nLuxels,
								 PtGpuBakeResult *outResults );

void PathTraceDXR_GpuBakeEnd();
bool PathTraceDXR_GpuBakeIsActive();

// Emit one summary for all-dark GPU batches accumulated since last Begin (then clear).
void PathTraceDXR_ReportGpuBakeDarkStats( const char *phase );

uint32_t PathTraceDXR_CapturedTriCount();

// Triangle TRACE_ID_* captured before KD convert (safe after SetupAccelerationStructure).
uint32_t PathTraceDXR_CapturedTriFlags( uint32_t triIndex );

// World-space (or local for prop-model) verts for albedo / debug.
bool PathTraceDXR_GetCapturedTri( uint32_t triIndex, Vector &a, Vector &b, Vector &c, uint32_t *outFlags );

#endif
