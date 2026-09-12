//========= Copyright PathRAD, All rights reserved. ============//
//
// fog_volume - bake a 3D light grid and rewrite the entity to lua_run.
//
//=============================================================================//

#ifndef VRAD_FOG_VOLUME_H
#define VRAD_FOG_VOLUME_H
#pragma once

struct entity_t;

#define FOG_VOLUME_MAX 64
#define FOG_BLOCKER_MAX 128

void FogVolume_Clear();
void FogVolume_ParseEntity( entity_t *e );
void FogVolume_ParseBlockerEntity( entity_t *e );
bool FogVolume_HasVolumes();
int  FogVolume_VolumeCount();

// After leaf ambient lighting: bake grids, pack VTF/VMT/shaders, rewrite entities.
void FogVolume_BakeAndEmbed();

#endif // VRAD_FOG_VOLUME_H
