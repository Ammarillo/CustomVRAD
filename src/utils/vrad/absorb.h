//========= Copyright Valve Corporation, All rights reserved. ============//
//
// light_absorb — compile-time light absorber volumes (CustomVRAD).
//
//=============================================================================//

#ifndef VRAD_ABSORB_H
#define VRAD_ABSORB_H
#pragma once

#include "mathlib/vector.h"

struct entity_t;

#define ABSORB_MAX_VOLUMES 64

void Absorb_Clear();
void Absorb_ParseVolumeEntity( entity_t *e );
bool Absorb_HasVolumes();
int  Absorb_VolumeCount();

// 0 = no absorb, 1 = full absorb at pos (blended across volumes).
float Absorb_GetWeight( const Vector &pos );

// Scale factor for lighting (1 - weight), considering volume flags.
float Absorb_DirectScale( const Vector &pos );
float Absorb_BounceScale( const Vector &emitterPos, const Vector &receiverPos );

#endif // VRAD_ABSORB_H
