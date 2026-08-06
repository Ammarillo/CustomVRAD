//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Optional baked ambient occlusion for CustomVRAD.
// CLI (-ao*) and/or map entities: light_ao (global) + light_ao_vol (volumes).
//
//=============================================================================//

#ifndef VRAD_AO_H
#define VRAD_AO_H
#pragma once

#include "mathlib/vector.h"

struct facelight_t;
struct entity_t;

#define AO_MAX_VOLUMES		64

enum AOBlendMode_t
{
	AO_BLEND_INSIDE = 0,
	AO_BLEND_OUTSIDE = 1,
	AO_BLEND_CENTER = 2,
};

struct AOSettings_t
{
	bool	enabled;
	int		samples;
	float	distance;
	float	strength;
	float	bias;
};

// CLI / resolved global defaults (light_ao overwrites these when present).
extern bool		g_bAO;
extern int		g_nAOSamples;
extern float	g_flAODistance;
extern float	g_flAOStrength;
extern float	g_flAOBias;
// Optional edge-preserving AO denoise (face lightmap filter; from light_ao / CLI).
extern bool		g_bAODenoise;
extern int		g_nAODenoiseRadius;		// 1=3x3, 2=5x5, ...
extern float	g_flAODenoiseStrength;	// 0..1 blend toward filtered

void AO_ClearEntities();
// Point entity light_ao - map-wide defaults + enable toggle.
void AO_ParseGlobalEntity( entity_t *e );
// Brush entity light_ao_vol - per-volume override (same keys + blend).
void AO_ParseVolumeEntity( entity_t *e );

bool AO_HasVolumes();
bool AO_ShouldRun(); // CLI/global enable or any volume present
int  AO_VolumeCount();

// Weighted blend of default + volumes at world position.
void AO_ResolveSettings( const Vector &pos, AOSettings_t &out );

// Fills pAOOut[numluxels] with occlusion factors in [0,1] (1 = fully open).
// Per-luxel distance/bias from resolved settings; samples = face max.
void ComputeFaceAmbientOcclusion( int facenum, const facelight_t *fl, float *pAOOut );

// scale = Lerp( 1, ao, strength ) at pos. Strength may be >1 (boosted darkening).
float AOScaleFromFactor( float ao, const Vector &pos );

#endif // VRAD_AO_H
