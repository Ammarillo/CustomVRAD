//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
//=============================================================================//

#ifndef VRADDETAILPROPS_H
#define VRADDETAILPROPS_H
#ifdef _WIN32
#pragma once
#endif


#include "bspfile.h"
#include "mathlib/anorms.h"


// Calculate the lighting at whatever surface the ray hits.
// Note: this ADDS to the values already in color. So if you want absolute
// values in there, then clear the values in color[] first.
// pSkyAmbientAtStart optionally supplies the volume-blended sky ambient
// intensity (0..255 scale) at vStart, so callers shooting many rays from one
// point don't recompute the light_env_vol weights per ray.
void CalcRayAmbientLighting(
	int iThread,
	const Vector &vStart,
	const Vector &vEnd,
	float tanTheta,			// tangent of the inner angle of the cone
	Vector color[MAX_LIGHTSTYLES],	// The color contribution from each lightstyle.
	const Vector *pSkyAmbientAtStart = NULL
	);

// Sky ambient intensity (0..255 scale) at a position, blending
// light_env_vol ambient overrides by volume weight.
void ComputeSkyAmbientAtPos( const Vector &pos, Vector &intensity );

bool CastRayInLeaf( int iThread, const Vector &start, const Vector &end, int leafIndex, float *pFraction, Vector *pNormal );

void ComputeDetailPropLighting( int iThread );


#endif // VRADDETAILPROPS_H
