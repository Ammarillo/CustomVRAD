//========= Copyright CustomVRAD contributors. ============//
// light_bounce_vol - local artistic bounce overrides (boost / chroma).
//=============================================================================//
#ifndef VRAD_BOUNCE_VOL_H
#define VRAD_BOUNCE_VOL_H
#pragma once

#include "mathlib/vector.h"

struct entity_t;

#define BOUNCEVOL_MAX_VOLUMES 64

struct BounceVolSettings_t
{
	float boost;	// effective bounce boost at position (1 = stock)
	float chroma;	// effective bounce chroma at position (0 = off)
	float energy;	// 0 = stock Valve gather, 1 = full cavity damp (blended)
};

// GPU / tooling copy of one light_bounce_vol (0-based index).
struct BounceVolDesc_t
{
	Vector	mins;
	Vector	maxs;
	float	blendDistance;
	int		blendMode;
	int		priority;
	float	volumeSize;
	float	bounceBoost;	// < 0 = inherit CLI
	float	bounceChroma;	// < 0 = inherit CLI
	float	energyMode;		// < 0 = inherit
};

void BounceVol_Clear();
void BounceVol_ParseVolumeEntity( entity_t *e );
bool BounceVol_HasVolumes();
int  BounceVol_VolumeCount();
bool BounceVol_GetVolume( int index, BounceVolDesc_t &out );

// Blends CLI globals with any light_bounce_vol overrides at pos.
void BounceVol_Resolve( const Vector &pos, BounceVolSettings_t &out );

#endif // VRAD_BOUNCE_VOL_H
