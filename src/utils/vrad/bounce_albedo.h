//========= Copyright CustomVRAD contributors. ============//
// Per-texel $basetexture sampling for textured radiosity bounce (-texbounce).
//=============================================================================//
#ifndef BOUNCE_ALBEDO_H
#define BOUNCE_ALBEDO_H
#pragma once

#include "mathlib/vector.h"

// Sample linear (gamma-decoded) albedo at a world position on a face.
// Returns false if no basetexture could be loaded (caller keeps flat reflectivity).
bool BounceAlbedo_SampleFace( int facenum, const Vector &worldPos, Vector &outLinearRGB );

// Preload all texdata albedos (safe to call multiple times).
void BounceAlbedo_EnsureCache( void );

#endif // BOUNCE_ALBEDO_H
