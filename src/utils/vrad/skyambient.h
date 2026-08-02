//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Shared sky ambient evaluation for lightmaps, props, and leaf cubes.
// Flat _ambient with light_env_vol blend.
//
//=============================================================================//

#ifndef VRAD_SKYAMBIENT_H
#define VRAD_SKYAMBIENT_H
#pragma once

#include "mathlib/vector.h"

struct directlight_t;

// Per-env sky ambient params (intensity units matching dworldlight intensity).
struct SkyAmbientParams_t
{
	Vector	ambient;		// flat _ambient
};

// Fill params from an emit_skyambient light.
void SkyAmbient_GetParams( const directlight_t *pAmbient, SkyAmbientParams_t &out );

// Volume-blended sky ambient at pos (0..255 intensity scale).
void SkyAmbient_ComputeAtPos( const Vector &pos, Vector &intensity );

// Volume-blended sun at pos for leaf ambient cubes.
// Returns false if no skylight. sunDir = toward sun; sunIntensity in 0..255 scale.
bool SkyAmbient_ComputeSunAtPos( const Vector &pos, Vector &sunDir, Vector &sunIntensity );

#endif // VRAD_SKYAMBIENT_H
