//========= Copyright Valve Corporation, All rights reserved. ============//
//
// light_environment_volume: AABB volumes with blended sky/ambient weights.
//
//=============================================================================//

#include "vrad.h"
#include "envvolume.h"
#include "oklab.h"

static LightEnvVolumeInfo_t s_Volumes[LIGHTENV_MAX_VOLUMES];
static int s_nVolumes = 0;
static bool s_bAnyInboundBounceTint = false;
static bool s_bAnyShadowCastFilter = false;
static float s_flDefaultBounceIntensity = 1.0f;
static CUtlVector<int> s_PatchEnvId;

void LightEnv_ClearVolumes()
{
	s_nVolumes = 0;
	s_bAnyInboundBounceTint = false;
	s_bAnyShadowCastFilter = false;
	s_flDefaultBounceIntensity = 1.0f;
	s_PatchEnvId.RemoveAll();
}

int LightEnv_VolumeCount()
{
	return s_nVolumes;
}

bool LightEnv_HasVolumes()
{
	return s_nVolumes > 0;
}

bool LightEnv_HasInboundBounceTinting()
{
	return s_bAnyInboundBounceTint;
}

int LightEnv_AddVolume( const Vector &mins, const Vector &maxs, float blendDistance, int blendMode, int priority,
						bool inboundBounceUsesVolumeColor, bool inboundBounceUsesVolumeBrightness,
						bool outsideCastShadow, bool insideCastShadow )
{
	if ( s_nVolumes >= LIGHTENV_MAX_VOLUMES )
	{
		Warning( "WARNING: too many light_environment_volume entities (max %d)\n", LIGHTENV_MAX_VOLUMES );
		return LIGHTENV_ID_NONE;
	}

	if ( blendMode < LIGHTENV_BLEND_INSIDE || blendMode > LIGHTENV_BLEND_CENTER )
		blendMode = LIGHTENV_BLEND_CENTER;

	LightEnvVolumeInfo_t &v = s_Volumes[s_nVolumes];
	v.mins = mins;
	v.maxs = maxs;
	v.blendDistance = ( blendDistance > 0.0f ) ? blendDistance : 0.0f;
	v.blendMode = blendMode;
	v.priority = priority;
	v.bInboundBounceUsesVolumeColor = inboundBounceUsesVolumeColor;
	v.bInboundBounceUsesVolumeBrightness = inboundBounceUsesVolumeBrightness;
	v.bOutsideCastShadow = outsideCastShadow;
	v.bInsideCastShadow = insideCastShadow;
	v.bounceTint.Init( 1.0f, 1.0f, 1.0f );
	v.bounceIntensity = 1.0f;

	Vector size = maxs - mins;
	float sx = ( size.x > 0.0f ) ? size.x : 0.0f;
	float sy = ( size.y > 0.0f ) ? size.y : 0.0f;
	float sz = ( size.z > 0.0f ) ? size.z : 0.0f;
	v.volumeSize = sx * sy * sz;
	v.envId = s_nVolumes + 1; // 1-based; 0 is default

	if ( inboundBounceUsesVolumeColor )
		s_bAnyInboundBounceTint = true;
	if ( !outsideCastShadow || !insideCastShadow )
		s_bAnyShadowCastFilter = true;

	++s_nVolumes;
	return v.envId;
}

bool LightEnv_HasShadowCastFilters()
{
	return s_bAnyShadowCastFilter;
}

static bool PointInVolumeAABB( const LightEnvVolumeInfo_t &v, const Vector &pos )
{
	return pos.x >= v.mins.x && pos.x <= v.maxs.x &&
		   pos.y >= v.mins.y && pos.y <= v.maxs.y &&
		   pos.z >= v.mins.z && pos.z <= v.maxs.z;
}

bool LightEnv_ShouldIgnoreSkyOccluder( const Vector &samplePos, const Vector &hitPos )
{
	if ( !s_bAnyShadowCastFilter )
		return false;

	for ( int i = 0; i < s_nVolumes; ++i )
	{
		const LightEnvVolumeInfo_t &v = s_Volumes[i];
		if ( v.bOutsideCastShadow && v.bInsideCastShadow )
			continue;

		const bool sampleIn = PointInVolumeAABB( v, samplePos );
		const bool hitIn = PointInVolumeAABB( v, hitPos );

		// Outside geo → sample inside volume
		if ( sampleIn && !hitIn && !v.bOutsideCastShadow )
			return true;
		// Inside geo → sample outside volume
		if ( !sampleIn && hitIn && !v.bInsideCastShadow )
			return true;
	}
	return false;
}

void LightEnv_SetDefaultBounceIntensity( const Vector &lightColor )
{
	float sum = 0.0f;
	if ( lightColor.x > 0.0f ) sum += lightColor.x;
	if ( lightColor.y > 0.0f ) sum += lightColor.y;
	if ( lightColor.z > 0.0f ) sum += lightColor.z;
	s_flDefaultBounceIntensity = ( sum > 1e-6f ) ? sum : 1.0f;
}

float LightEnv_GetDefaultBounceIntensity()
{
	return s_flDefaultBounceIntensity;
}

const LightEnvVolumeInfo_t *LightEnv_GetVolumeInfo( int envId )
{
	int idx = envId - 1;
	if ( idx < 0 || idx >= s_nVolumes )
		return NULL;
	return &s_Volumes[idx];
}

void LightEnv_SetVolumeBounceTint( int envId, const Vector &lightColor )
{
	int idx = envId - 1;
	if ( idx < 0 || idx >= s_nVolumes )
		return;

	Vector tint = lightColor;
	if ( tint.x < 0.0f ) tint.x = 0.0f;
	if ( tint.y < 0.0f ) tint.y = 0.0f;
	if ( tint.z < 0.0f ) tint.z = 0.0f;

	float sum = tint.x + tint.y + tint.z;
	s_Volumes[idx].bounceIntensity = ( sum > 1e-6f ) ? sum : 1.0f;

	if ( sum < 1e-6f )
		tint.Init( 1.0f, 1.0f, 1.0f );
	else
		tint *= ( 1.0f / sum );

	s_Volumes[idx].bounceTint = tint;
}

// Soft-min of three values (rounded corners vs hard min of face distances).
static float SoftMin3( float a, float b, float c, float k )
{
	if ( k < 1e-4f )
		return min( a, min( b, c ) );

	float m = min( a, min( b, c ) );
	float ea = expf( -( a - m ) / k );
	float eb = expf( -( b - m ) / k );
	float ec = expf( -( c - m ) / k );
	return ( a * ea + b * eb + c * ec ) / ( ea + eb + ec );
}

// Inside: positive distance to boundary (softmin). Outside: negative Euclidean gap.
static float SignedBlendDistance( const Vector &p, const Vector &mins, const Vector &maxs, float softK )
{
	float dx = min( p.x - mins.x, maxs.x - p.x );
	float dy = min( p.y - mins.y, maxs.y - p.y );
	float dz = min( p.z - mins.z, maxs.z - p.z );

	if ( dx >= 0.0f && dy >= 0.0f && dz >= 0.0f )
		return SoftMin3( dx, dy, dz, softK );

	// Outside: Euclidean distance to the AABB, negated.
	float ox = 0.0f, oy = 0.0f, oz = 0.0f;
	if ( p.x < mins.x ) ox = mins.x - p.x;
	else if ( p.x > maxs.x ) ox = p.x - maxs.x;
	if ( p.y < mins.y ) oy = mins.y - p.y;
	else if ( p.y > maxs.y ) oy = p.y - maxs.y;
	if ( p.z < mins.z ) oz = mins.z - p.z;
	else if ( p.z > maxs.z ) oz = p.z - maxs.z;
	return -sqrtf( ox * ox + oy * oy + oz * oz );
}

// Perlin smootherstep (C2). Softer than cosine/smoothstep; less mid-band banding.
static float SmootherStep01( float t )
{
	if ( t <= 0.0f ) return 0.0f;
	if ( t >= 1.0f ) return 1.0f;
	return t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
}

// Soft Voronoi: keep outside halos from invading a neighboring volume's side.
// Without this, two volumes BlendDistance apart (e.g. 128 gap, blend 64 each)
// overlap in the middle and tint each other's blend shell (orange leak).
static float TerritoryFactor( int idx, const Vector &pos, float signedDist )
{
	float factor = 1.0f;
	const LightEnvVolumeInfo_t &v = s_Volumes[idx];

	for ( int j = 0; j < s_nVolumes; ++j )
	{
		if ( j == idx )
			continue;

		const LightEnvVolumeInfo_t &o = s_Volumes[j];
		const float softKJ = ( o.blendDistance > 0.0f ) ? ( o.blendDistance * 0.15f ) : 0.0f;
		const float distJ = SignedBlendDistance( pos, o.mins, o.maxs, softKJ );
		const float advantage = distJ - signedDist; // >0 => neighbor is closer / more interior
		if ( advantage <= 0.0f )
			continue;

		// Higher-priority neighbor claims territory faster.
		float handoff = 0.25f * ( v.blendDistance + o.blendDistance );
		if ( o.priority > v.priority )
			handoff *= 0.5f;
		else if ( v.priority > o.priority )
			handoff *= 2.0f;
		if ( handoff < 8.0f )
			handoff = 8.0f;

		factor *= SmootherStep01( 1.0f - advantage / handoff );
		if ( factor <= 1e-5f )
			return 0.0f;
	}
	return factor;
}

static float RawVolumeWeight( int idx, const Vector &pos )
{
	const LightEnvVolumeInfo_t &v = s_Volumes[idx];

	if ( v.blendDistance <= 0.0f )
	{
		// Hard cut: inside = 1, outside = 0.
		if ( pos.x < v.mins.x || pos.x > v.maxs.x ||
			 pos.y < v.mins.y || pos.y > v.maxs.y ||
			 pos.z < v.mins.z || pos.z > v.maxs.z )
			return 0.0f;
		return 1.0f;
	}

	// Soft radius ~15% of blend distance rounds AABB corners in the weight field.
	const float softK = v.blendDistance * 0.15f;
	const float dist = SignedBlendDistance( pos, v.mins, v.maxs, softK );
	const float blend = v.blendDistance;

	float t;
	switch ( v.blendMode )
	{
	case LIGHTENV_BLEND_INSIDE:
		// 0 at face / outside → 1 at +BlendDistance inside.
		if ( dist <= 0.0f )
			return 0.0f;
		t = dist / blend;
		break;

	case LIGHTENV_BLEND_OUTSIDE:
		// 1 inside / at face → 0 at -BlendDistance outside.
		if ( dist >= 0.0f )
			return 1.0f;
		t = 1.0f + dist / blend;
		break;

	case LIGHTENV_BLEND_CENTER:
	default:
		// Shell centered on the face: 0 at -BlendDistance, 0.5 at face, 1 at +BlendDistance.
		t = ( dist + blend ) / ( 2.0f * blend );
		break;
	}

	const float w = SmootherStep01( t );
	if ( w <= 0.0f )
		return 0.0f;

	return w * TerritoryFactor( idx, pos, dist );
}

void LightEnv_ComputeAllWeights( const Vector &pos, float *pOutWeights )
{
	const int nOut = s_nVolumes + 1; // [0]=default, [1..N]=volumes
	for ( int i = 0; i < nOut; ++i )
		pOutWeights[i] = 0.0f;

	if ( s_nVolumes == 0 )
	{
		pOutWeights[LIGHTENV_ID_DEFAULT] = 1.0f;
		return;
	}

	float raw[LIGHTENV_MAX_VOLUMES];
	int saturatedIdx = -1;

	for ( int i = 0; i < s_nVolumes; ++i )
	{
		raw[i] = RawVolumeWeight( i, pos );

		if ( raw[i] >= 1.0f - 1e-4f )
		{
			if ( saturatedIdx < 0 )
			{
				saturatedIdx = i;
			}
			else
			{
				const LightEnvVolumeInfo_t &a = s_Volumes[saturatedIdx];
				const LightEnvVolumeInfo_t &b = s_Volumes[i];
				if ( b.priority > a.priority ||
					 ( b.priority == a.priority && b.volumeSize < a.volumeSize ) )
				{
					saturatedIdx = i;
				}
			}
		}
	}

	if ( saturatedIdx >= 0 )
	{
		pOutWeights[s_Volumes[saturatedIdx].envId] = 1.0f;
		return;
	}

	float sum = 0.0f;
	for ( int i = 0; i < s_nVolumes; ++i )
		sum += raw[i];

	if ( sum <= 1.0f )
	{
		pOutWeights[LIGHTENV_ID_DEFAULT] = 1.0f - sum;
		for ( int i = 0; i < s_nVolumes; ++i )
			pOutWeights[s_Volumes[i].envId] = raw[i];
		return;
	}

	// Overlapping blend shells: normalize among volumes only.
	for ( int i = 0; i < s_nVolumes; ++i )
		pOutWeights[s_Volumes[i].envId] = raw[i] / sum;
}

float LightEnv_GetWeight( int envId, const Vector &pos )
{
	if ( envId == LIGHTENV_ID_NONE )
		return 1.0f;

	if ( s_nVolumes == 0 )
		return ( envId == LIGHTENV_ID_DEFAULT ) ? 1.0f : 0.0f;

	float weights[LIGHTENV_MAX_VOLUMES + 1];
	LightEnv_ComputeAllWeights( pos, weights );

	if ( envId < 0 || envId > s_nVolumes )
		return 0.0f;
	return weights[envId];
}

int LightEnv_GetDominantEnvId( const Vector &pos )
{
	if ( s_nVolumes == 0 )
		return LIGHTENV_ID_DEFAULT;

	float weights[LIGHTENV_MAX_VOLUMES + 1];
	LightEnv_ComputeAllWeights( pos, weights );

	int best = LIGHTENV_ID_DEFAULT;
	float bestW = weights[0];
	for ( int e = 1; e <= s_nVolumes; ++e )
	{
		if ( weights[e] > bestW )
		{
			bestW = weights[e];
			best = e;
		}
	}
	return best;
}

void LightEnv_MaybeTintInboundBounce( int receiverEnvId, int emitterEnvId, Vector &light )
{
	if ( receiverEnvId <= LIGHTENV_ID_DEFAULT )
		return;

	int idx = receiverEnvId - 1;
	if ( idx < 0 || idx >= s_nVolumes )
		return;

	const LightEnvVolumeInfo_t &vol = s_Volumes[idx];
	if ( !vol.bInboundBounceUsesVolumeColor )
		return;

	// Emitter is inside the same volume — keep natural bounce color.
	if ( emitterEnvId == receiverEnvId )
		return;

	float lum = light.x + light.y + light.z;
	if ( lum < 1e-10f )
		return;

	float scale = 1.0f;
	if ( vol.bInboundBounceUsesVolumeBrightness )
		scale = vol.bounceIntensity / s_flDefaultBounceIntensity;

	light = Oklab_ApplyTintPreserveL( light, vol.bounceTint, scale );
}

void LightEnv_BuildPatchEnvCache()
{
	s_PatchEnvId.RemoveAll();
	if ( !s_nVolumes )
		return;

	unsigned int n = g_Patches.Size();
	s_PatchEnvId.SetCount( (int)n );
	for ( unsigned int i = 0; i < n; ++i )
	{
		s_PatchEnvId[i] = LightEnv_GetDominantEnvId( g_Patches[i].origin );
	}
}

int LightEnv_GetPatchEnvId( int patchIndex )
{
	if ( !s_PatchEnvId.Count() || patchIndex < 0 || patchIndex >= s_PatchEnvId.Count() )
		return LIGHTENV_ID_DEFAULT;
	return s_PatchEnvId[patchIndex];
}

void LightEnv_CopySkyVisToVolumeLights()
{
	if ( !s_nVolumes )
		return;

	directlight_t *pSrc = NULL;
	for ( directlight_t *dl = activelights; dl; dl = dl->next )
	{
		if ( dl->light.type == emit_skylight && dl->m_nEnvId == LIGHTENV_ID_DEFAULT && dl->pvs )
		{
			pSrc = dl;
			break;
		}
	}
	if ( !pSrc || !pSrc->pvs )
		return;

	int nBytes = ( dvis->numclusters / 8 ) + 1;

	for ( directlight_t *dl = activelights; dl; dl = dl->next )
	{
		if ( dl->m_nEnvId <= LIGHTENV_ID_DEFAULT )
			continue;
		if ( dl->light.type != emit_skylight && dl->light.type != emit_skyambient )
			continue;

		if ( !dl->pvs )
			dl->pvs = (byte *)calloc( 1, nBytes );

		memcpy( dl->pvs, pSrc->pvs, nBytes );
	}
}
