//========= Copyright Valve Corporation, All rights reserved. ============//
//
// light_environment_volume support: per-volume sky/ambient with blend.
//
//=============================================================================//

#ifndef ENVVOLUME_H
#define ENVVOLUME_H
#pragma once

#include "mathlib/vector.h"

// Env id 0 = default light_environment. 1..N = light_environment_volume.
// Non-sky lights use LIGHTENV_ID_NONE and always get weight 1.
#define LIGHTENV_ID_NONE		(-1)
#define LIGHTENV_ID_DEFAULT		0
#define LIGHTENV_MAX_VOLUMES	64

// Per-volume BlendDistance placement relative to the brush face.
enum LightEnvBlendMode_t
{
	LIGHTENV_BLEND_INSIDE = 0,	// fade only inside (0 at face → 1 at +BlendDistance)
	LIGHTENV_BLEND_OUTSIDE = 1,	// full inside; fade only outside (1 at face → 0 at -BlendDistance)
	LIGHTENV_BLEND_CENTER = 2,	// fade straddles the face (center shell)
};

struct LightEnvVolumeInfo_t
{
	Vector	mins;
	Vector	maxs;
	float	blendDistance;
	int		blendMode;	// LightEnvBlendMode_t
	int		priority;
	float	volumeSize;	// AABB volume for tie-break
	int		envId;		// 1..N
	bool	bInboundBounceUsesVolumeColor;
	bool	bInboundBounceUsesVolumeBrightness;
	Vector	bounceTint;			// volume _light chromaticity (sums to 1)
	float	bounceIntensity;	// sum of volume _light RGB intensity
};

void LightEnv_ClearVolumes();
int  LightEnv_AddVolume( const Vector &mins, const Vector &maxs, float blendDistance, int blendMode, int priority,
						 bool inboundBounceUsesVolumeColor, bool inboundBounceUsesVolumeBrightness );
void LightEnv_SetVolumeBounceTint( int envId, const Vector &lightColor );
void LightEnv_SetDefaultBounceIntensity( const Vector &lightColor );
float LightEnv_GetDefaultBounceIntensity();
int  LightEnv_VolumeCount();
// envId 1..N; NULL if invalid.
const LightEnvVolumeInfo_t *LightEnv_GetVolumeInfo( int envId );

// Weight for a sky/ambient light with the given env id at world position.
// Non-volume lights should pass LIGHTENV_ID_NONE (returns 1).
float LightEnv_GetWeight( int envId, const Vector &pos );

// Fill weights for env ids 0..VolumeCount() at one point (index 0 = default).
// pOutWeights must hold at least (LightEnv_VolumeCount()+1) floats.
void LightEnv_ComputeAllWeights( const Vector &pos, float *pOutWeights );

// Dominant env id at position (0 = default).
int LightEnv_GetDominantEnvId( const Vector &pos );

// True if any light_environment_volume was parsed.
bool LightEnv_HasVolumes();

// True if any volume wants inbound bounce recoloring.
bool LightEnv_HasInboundBounceTinting();

// If receiverEnv wants volume-colored inbound bounce and emitterEnv is outside
// that volume, recolor light in-place. Luminance is preserved unless the volume
// also has InboundBounceUsesVolumeBrightness set (then scaled vs default env).
void LightEnv_MaybeTintInboundBounce( int receiverEnvId, int emitterEnvId, Vector &light );

// Copy default sky PVS onto all volume sky/ambient lights after BuildVisForLightEnvironment.
void LightEnv_CopySkyVisToVolumeLights();

// Cache dominant env id per patch for radiosity (call before BounceLight).
void LightEnv_BuildPatchEnvCache();
int  LightEnv_GetPatchEnvId( int patchIndex );

#endif // ENVVOLUME_H
