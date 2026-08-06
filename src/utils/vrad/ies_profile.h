//========= Copyright CustomVRAD contributors. ============//
// IESNA LM-63 photometric profiles for light_spot (bake-only).
// When set, replaces _inner_cone / _cone / _exponent angular falloff.
//=============================================================================//
#ifndef VRAD_IES_PROFILE_H
#define VRAD_IES_PROFILE_H
#pragma once

#include "mathlib/vector.h"
#include <vector>

struct IesProfile;

// Find or load an IES file (game FS). Tries IES/<path>, then as-is / maps/.
// Returns nullptr on failure (caller keeps stock cone).
const IesProfile *IES_FindOrLoad( const char *pPath );

// Relative candela (peak = 1) along direction from light toward sample.
// beamDir = emission axis (light.normal); rightDir = orthonormal "right" for phi=0.
float IES_Eval( const IesProfile *p, const Vector &beamDir, const Vector &rightDir,
				const Vector &lightToSample );

// Build orthonormal right/up for IES phi.
// angles: entity pitch/yaw/roll (same convention as light_spot). Yaw+roll spin
// asymmetric lobes around the beam - critical when pitch~=+/-90 where beam
// direction no longer carries yaw.
void IES_BuildBasis( const Vector &beamDir, const QAngle &angles,
					 Vector &rightOut, Vector &upOut );

// Fallback when no entity angles are available (world-up preference).
void IES_BuildBasis( const Vector &beamDir, Vector &rightOut, Vector &upOut );

// GPU atlas: each unique loaded profile -> one layer of ResV x ResH floats (peak-normalized).
// Resolution is chosen adaptively from the loaded IES tables (min 64x32, max 1024x1440)
// so high-res profiles (e.g. 361x720) are not crushed to the old fixed 64x32 grid.
static const int IES_GPU_RES_V_MIN = 64;
static const int IES_GPU_RES_H_MIN = 32;
static const int IES_GPU_RES_V_MAX = 1024;
static const int IES_GPU_RES_H_MAX = 1440;

// Assign layer indices for profiles referenced by active lights; fill atlas
// (layers * ResV * ResH). Returns layer count (min 1 dummy layer of zeros).
// Call IES_GpuResV/H afterwards for the chosen resolution.
int IES_BuildGpuAtlas( std::vector<float> &atlasOut );

// Layer for a profile after IES_BuildGpuAtlas (-1 = unused / no IES).
int IES_GpuLayer( const IesProfile *p );

// Atlas resolution from the last IES_BuildGpuAtlas call.
int IES_GpuResV();
int IES_GpuResH();

#endif // VRAD_IES_PROFILE_H
