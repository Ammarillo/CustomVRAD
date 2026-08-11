//========= Copyright CustomVRAD, All rights reserved. ============//
//
// water_medium - physically based underwater lighting for bake (CustomVRAD).
// Auto from BSP leafWaterData + water VMT fog; optional water_light_vol override.
// Mode A: Beer-Lambert + Kd. Mode C: homogeneous volume path tracing.
//
//=============================================================================//

#ifndef VRAD_WATER_MEDIUM_H
#define VRAD_WATER_MEDIUM_H
#pragma once

#include "mathlib/vector.h"

struct entity_t;

#define WATER_MEDIUM_MAX_BODIES 64
#define WATER_MEDIUM_MAX_OVERRIDES 32

enum WaterMediumMode_t
{
	WATER_MEDIUM_OFF = 0,
	WATER_MEDIUM_ATTENUATION = 1,	// Mode A: Beer-Lambert + Kd
	WATER_MEDIUM_VOLUME = 2,		// Mode C: free-flight volume PT
};

struct WaterIops_t
{
	Vector	sigmaA;		// absorption (1 / hammer unit)
	Vector	sigmaS;		// scattering (1 / hammer unit)
	Vector	sigmaT;		// extinction = a + s
	Vector	kd;			// diffuse downwelling (approx)
	Vector	fogColor;	// veiling / residual tint (linear 0..1)
	float	g;			// HG anisotropy
	float	surfaceZ;
	float	minZ;
};

// GPU pack: 5 x float4 per body
struct WaterGpuBody_t
{
	float	mins[3];
	float	sigmaAx;
	float	maxs[3];
	float	sigmaAy;
	float	sigmaAz;
	float	sigmaSx;
	float	sigmaSy;
	float	sigmaSz;
	float	g;
	float	surfaceZ;
	float	kdX;
	float	kdY;
	float	kdZ;
	float	fogR;
	float	fogG;
	float	fogB;
};

void WaterMedium_Clear();
void WaterMedium_ParseOverrideEntity( entity_t *e );

// Build after CreateDirectLights entity parse (needs BSP + materials).
void WaterMedium_BuildFromBSP();

bool WaterMedium_IsActive();
WaterMediumMode_t WaterMedium_GetMode();
int WaterMedium_BodyCount();
int WaterMedium_MaxScatter();

// CLI (set from vrad.cpp)
void WaterMedium_SetCli( bool enable, bool volumeMode, bool forceOff, int maxScatter );

bool WaterMedium_PointInWater( const Vector &pos, int *pBodyIndex = nullptr );
float WaterMedium_DepthBelowSurface( const Vector &pos, int bodyIndex = -1 );

// Beer-Lambert transmittance along segment (1 = clear). Mode A + C both use this for NEE.
Vector WaterMedium_SegmentTransmittance( const Vector &from, const Vector &to );

// Kd scale for sky/sun at submerged receiver (1 = no attenuation).
Vector WaterMedium_DownwellingScale( const Vector &pos );

const WaterIops_t *WaterMedium_GetIops( int bodyIndex );
bool WaterMedium_GetGpuBodies( WaterGpuBody_t *out, int maxOut, int *pCount );

// Volume PT helpers (Mode C)
bool WaterMedium_SampleFreeFlight( const Vector &pos, const Vector &dir, float u,
								  float *pTFree, Vector *pSigmaT, int *pBodyIndex );
Vector WaterMedium_ScatterAlbedo( int bodyIndex );	// sigmaS / sigmaT
float WaterMedium_PhaseHG( float cosTheta, float g );
void WaterMedium_SampleHG( float g, float u1, float u2, const Vector &wo, Vector &wiOut );

#endif // VRAD_WATER_MEDIUM_H
