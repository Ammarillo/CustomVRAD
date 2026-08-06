//========= Copyright CustomVRAD contributors. ============//
// VMT-driven textured emissive surfaces for VRAD.
//=============================================================================//
#ifndef VRAD_EMIT_H
#define VRAD_EMIT_H
#pragma once

#include "mathlib/vector.h"

// VMT keys (CustomVRAD):
//   $vrad_emit            "1"              - enable emission (optional if strength > 0)
//   $vrad_emitstrength    "200"            - intensity scaler (same idea as lights.rad 4th number)
//   $vrad_emitdensity     "1"              - optional finer patch chop only (contact is area-softened; no light grid)
//   $vrad_emitmask        "path/to/mask"   - optional greyscale mask (R/luma)
//   $vrad_emitmap         "path/to/color"  - optional emissive color map
//   $vrad_emissivemap     ...              - alias for $vrad_emitmap
// If no emitmap: emission color comes from $basetexture (like -texbounce).

bool VRadEmit_MaterialEmits( const char *pMaterialName );

// True if this face's material has $vrad_emit* enabled.
bool VRadEmit_FaceEmits( int facenum );

// Emit density from VMT (1 = default). Only affects patch chop (finer subdiv).
float VRadEmit_GetDensity( const char *pMaterialName );

// Sample textured emission at a world position on a face (baselight units).
bool VRadEmit_SampleAtPos( int facenum, const Vector &worldPos, Vector &outEmit );

// Create emit_surface direct lights from VMT emission (call from CreateDirectLights).
int VRadEmit_CreateDirectLights( void );

// After BuildFacelights / pathtrace bake: add per-sample self-illumination.
void VRadEmit_AddSelfIllumToFaceLights( void );

#endif // VRAD_EMIT_H
