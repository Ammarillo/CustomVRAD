//========= Copyright PathRAD contributors. ============//
// Sample $basetexture albedo per texel for -texbounce.
//=============================================================================//
#ifndef BOUNCE_ALBEDO_H
#define BOUNCE_ALBEDO_H
#pragma once

#include "mathlib/vector.h"

// Sample linear (gamma-decoded) albedo at a world position on a face.
// Returns false if no basetexture could be loaded (caller keeps flat reflectivity).
bool BounceAlbedo_SampleFace( int facenum, const Vector &worldPos, Vector &outLinearRGB );

// Soft-kill DXT/JPEG chroma on near-greys (uses -texbounce_clean threshold).
void BounceAlbedo_SanitizeCompressionChroma( Vector &linRGB );

// Preload all texdata albedos (safe to call multiple times).
void BounceAlbedo_EnsureCache( void );

#endif // BOUNCE_ALBEDO_H
