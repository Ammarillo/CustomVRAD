//========= Copyright PathRAD contributors. ============//
// Area lights from $vrad_emit* faces (triangle mesh, PBRT-style sampling).
//=============================================================================//
#ifndef VRAD_EMIT_AREA_H
#define VRAD_EMIT_AREA_H
#pragma once

#include "mathlib/vector.h"

// One emissive triangle with vertex emission (albedoxstrength) for area NEE.
// Refs: PBRT DiffuseAreaLight (area->solid-angle PDF), Veach MIS.
struct VRadEmitTri_t
{
	Vector	v0, v1, v2;
	Vector	n;			// unit geometric normal (emission side)
	Vector	e0, e1, e2;	// radiance at verts (VRAD direct units, includes DIRECT_SCALE)
	float	area;
	float	power;		// avg(|e|) * area - for power CDF
	int		facenum;
};

// Build / clear triangle mesh from $vrad_emit faces (fan-triangulated).
void VRadEmitArea_Clear( void );
int  VRadEmitArea_Build( void );

int  VRadEmitArea_TriCount( void );
const VRadEmitTri_t *VRadEmitArea_Tris( void );
float VRadEmitArea_PowerSum( void );
const float *VRadEmitArea_PowerCdf( void ); // inclusive prefix, size TriCount, last=1

// True if this face is covered by the area mesh (skip point-proxy NEE).
bool VRadEmitArea_FaceHasMesh( int facenum );

#endif // VRAD_EMIT_AREA_H
