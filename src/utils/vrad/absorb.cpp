//========= Copyright Valve Corporation, All rights reserved. ============//
//
// light_absorb volume weights (soft AABB blend).
//
//=============================================================================//

#include "vrad.h"
#include "absorb.h"

struct AbsorbVolume_t
{
	Vector	mins;
	Vector	maxs;
	float	blendDistance;
	int		blendMode;	// 0 inside, 1 outside, 2 center
	int		priority;
	float	volumeSize;
	float	strength;
	bool	bAbsorbDirect;
	bool	bAbsorbBounce;
};

static AbsorbVolume_t s_Volumes[ABSORB_MAX_VOLUMES];
static int s_nVolumes = 0;

void Absorb_Clear()
{
	s_nVolumes = 0;
}

bool Absorb_HasVolumes()
{
	return s_nVolumes > 0;
}

int Absorb_VolumeCount()
{
	return s_nVolumes;
}

void Absorb_ParseVolumeEntity( entity_t *e )
{
	char *pModel = ValueForKey( e, "model" );
	if ( !pModel || pModel[0] != '*' )
	{
		Warning( "WARNING: light_absorb without brush model ignored\n" );
		return;
	}
	int modelIndex = atoi( pModel + 1 );
	if ( modelIndex <= 0 || modelIndex >= nummodels )
	{
		Warning( "WARNING: light_absorb has invalid model \"%s\"\n", pModel );
		return;
	}
	if ( s_nVolumes >= ABSORB_MAX_VOLUMES )
	{
		Warning( "WARNING: too many light_absorb entities (max %d)\n", ABSORB_MAX_VOLUMES );
		return;
	}

	dmodel_t *pModelData = &dmodels[modelIndex];
	AbsorbVolume_t &v = s_Volumes[s_nVolumes];
	v.mins = pModelData->mins;
	v.maxs = pModelData->maxs;
	v.blendDistance = FloatForKeyWithDefault( e, "BlendDistance", 0.0f );
	if ( v.blendDistance < 0.0f )
		v.blendDistance = 0.0f;
	v.blendMode = IntForKeyWithDefault( e, "BlendMode", 2 );
	if ( v.blendMode < 0 || v.blendMode > 2 )
		v.blendMode = 2;
	v.priority = IntForKeyWithDefault( e, "priority", 0 );
	v.strength = FloatForKeyWithDefault( e, "Strength", 0.5f );
	if ( v.strength < 0.0f ) v.strength = 0.0f;
	if ( v.strength > 1.0f ) v.strength = 1.0f;
	v.bAbsorbDirect = IntForKeyWithDefault( e, "AbsorbDirect", 0 ) != 0;
	v.bAbsorbBounce = IntForKeyWithDefault( e, "AbsorbBounce", 1 ) != 0;

	Vector size = v.maxs - v.mins;
	v.volumeSize = max( 0.0f, size.x ) * max( 0.0f, size.y ) * max( 0.0f, size.z );

	Msg( "light_absorb %d  strength=%.2f direct=%s bounce=%s blend=%.1f\n",
		 s_nVolumes + 1, v.strength,
		 v.bAbsorbDirect ? "yes" : "no",
		 v.bAbsorbBounce ? "yes" : "no",
		 v.blendDistance );
	++s_nVolumes;
}

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

static float SignedDist( const Vector &p, const Vector &mins, const Vector &maxs, float softK )
{
	float dx = min( p.x - mins.x, maxs.x - p.x );
	float dy = min( p.y - mins.y, maxs.y - p.y );
	float dz = min( p.z - mins.z, maxs.z - p.z );
	if ( dx >= 0.0f && dy >= 0.0f && dz >= 0.0f )
		return SoftMin3( dx, dy, dz, softK );
	float ox = 0, oy = 0, oz = 0;
	if ( p.x < mins.x ) ox = mins.x - p.x;
	else if ( p.x > maxs.x ) ox = p.x - maxs.x;
	if ( p.y < mins.y ) oy = mins.y - p.y;
	else if ( p.y > maxs.y ) oy = p.y - maxs.y;
	if ( p.z < mins.z ) oz = mins.z - p.z;
	else if ( p.z > maxs.z ) oz = p.z - maxs.z;
	return -sqrtf( ox * ox + oy * oy + oz * oz );
}

static float Smoother01( float t )
{
	if ( t <= 0.0f ) return 0.0f;
	if ( t >= 1.0f ) return 1.0f;
	return t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
}

static float RawWeight( int idx, const Vector &pos )
{
	const AbsorbVolume_t &v = s_Volumes[idx];
	if ( v.blendDistance <= 0.0f )
	{
		if ( pos.x < v.mins.x || pos.x > v.maxs.x ||
			 pos.y < v.mins.y || pos.y > v.maxs.y ||
			 pos.z < v.mins.z || pos.z > v.maxs.z )
			return 0.0f;
		return v.strength;
	}
	float softK = v.blendDistance * 0.15f;
	float dist = SignedDist( pos, v.mins, v.maxs, softK );
	float t;
	switch ( v.blendMode )
	{
	case 0:
		if ( dist <= 0.0f ) return 0.0f;
		t = dist / v.blendDistance;
		break;
	case 1:
		if ( dist >= 0.0f ) return v.strength;
		t = 1.0f + dist / v.blendDistance;
		break;
	default:
		t = ( dist + v.blendDistance ) / ( 2.0f * v.blendDistance );
		break;
	}
	return Smoother01( t ) * v.strength;
}

float Absorb_GetWeight( const Vector &pos )
{
	if ( s_nVolumes == 0 )
		return 0.0f;
	float w = 0.0f;
	for ( int i = 0; i < s_nVolumes; ++i )
	{
		float r = RawWeight( i, pos );
		if ( r > w )
			w = r;
	}
	if ( w > 1.0f ) w = 1.0f;
	return w;
}

float Absorb_DirectScale( const Vector &pos )
{
	if ( s_nVolumes == 0 )
		return 1.0f;
	float w = 0.0f;
	for ( int i = 0; i < s_nVolumes; ++i )
	{
		if ( !s_Volumes[i].bAbsorbDirect )
			continue;
		float r = RawWeight( i, pos );
		if ( r > w )
			w = r;
	}
	return 1.0f - min( 1.0f, w );
}

float Absorb_BounceScale( const Vector &emitterPos, const Vector &receiverPos )
{
	if ( s_nVolumes == 0 )
		return 1.0f;
	float we = 0.0f, wr = 0.0f;
	for ( int i = 0; i < s_nVolumes; ++i )
	{
		if ( !s_Volumes[i].bAbsorbBounce )
			continue;
		float re = RawWeight( i, emitterPos );
		float rr = RawWeight( i, receiverPos );
		if ( re > we ) we = re;
		if ( rr > wr ) wr = rr;
	}
	float w = max( we, wr );
	return 1.0f - min( 1.0f, w );
}
