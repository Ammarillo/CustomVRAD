//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
// $Workfile:     $
// $Date:         $
//
//-----------------------------------------------------------------------------
// $Log: $
//
// $NoKeywords: $
//=============================================================================//

#ifndef RADIAL_H
#define RADIAL_H
#pragma once

#include "mathlib/bumpvects.h"
#include "mathlib/ssemath.h"
#include "lightmap.h"

#define RADIALDIST2	2 // (1.25*1.25+1.25*1.25)
#define RADIALDIST	1.42 // 1.77 // sqrt( RADIALDIST2 )

// PathRAD: bounce splat radius (1 = stock). Larger = softer patch GI seams.
extern float g_flBounceRadialScale;
// Opt-in coplanar neighbor weld (-bounce_weld). Default off = stock bleed.
extern bool g_bBounceWeld;

#define WEIGHT_EPS	0.00001f

//-----------------------------------------------------------------------------
// The radial_t data structure is used to accumulate irregularly spaced and irregularly 
// shaped direct and indirect lighting samples into a uniformly spaced and shaped luxel grid.
// 
// The name "radial" is more historical than discriptive; it stems from the filtering method, 
// one of several methods initially tried.  Since all the other methods have since been deleted, 
// it would probably be more accurate to rename it something like "LuxelAccumulationBucket" or 
// something similar, but since "radial" is fairly meaningless it's not like it's actually confusing 
// the issue.
//-----------------------------------------------------------------------------
typedef struct radial_s
{
	int	facenum;
	lightinfo_t l;
	int w, h;
	float weight[SINGLEMAP];
	LightingValue_t light[NUM_BUMP_VECTS + 1][SINGLEMAP];
} radial_t;


void WorldToLuxelSpace( lightinfo_t const *l, Vector const &world, Vector2D &coord );
void LuxelSpaceToWorld( lightinfo_t const *l, float s, float t, Vector &world );

void WorldToLuxelSpace( lightinfo_t const *l, FourVectors const &world, FourVectors &coord );
void LuxelSpaceToWorld( lightinfo_t const *l, fltx4 s, fltx4 t, FourVectors &world );

void AddDirectToRadial( radial_t *rad, 
				  Vector const &pnt, 
				  Vector2D const &coordmins, Vector2D const &coordmaxs, 
				  Vector const light[NUM_BUMP_VECTS+1],
				  bool hasBumpmap, bool neighborHasBumpmap );

void AddBounceToRadial( radial_t *rad, 
				  Vector const &pnt, 
				  Vector2D const &coordmins, Vector2D const &coordmaxs, 
				  Vector const light[NUM_BUMP_VECTS+1],
				  bool hasBumpmap, bool neighborHasBumpmap );

bool SampleRadial( radial_t *rad, Vector& pnt, Vector light[NUM_BUMP_VECTS+1], int bumpSampleCount );

radial_t *AllocateRadial( int facenum );
void FreeRadial( radial_t *rad );

bool SampleRadial( radial_t *rad, Vector& pnt, Vector light[NUM_BUMP_VECTS + 1], int bumpSampleCount );
radial_t *BuildPatchRadial( int facenum );

// PathRAD: after FinalLightFace, blend luxels along shared edges of
// coplanar faces so VBSP splits don't show a brightness step.
// bPathTraceNarrow: tighter blend (path-trace soft shadows without blotching).
void StitchLightmapSeams( bool bPathTraceNarrow = false );

// Dilate dark outer luxels on concave creases (and any non-coplanar border
// whose last luxel is a sharp outlier) from that row's own stable interior texel.
// A light/shadow terminator keeps each side. Diagonal concave edges fill luxels
// that sit mostly outside the face. Coplanar VBSP splits are left for StitchLightmapSeams.
// bConcaveOnly skips the unknown-border fallback; use that after stitching.
void ReconstructLightmapEdgeLuxels( Vector *grid, int width, int height, int facenum, bool bConcaveOnly = false );

// Lit-side hard shadow edges: small lift toward bright neighbors (capped) so
// bilinear filtering does not smear darkness into open areas.
void CompressBilinearShadowLeak( Vector *grid, int width, int height );

// utilities
bool FloatLess( float const& src1, float const& src2 );

#endif
