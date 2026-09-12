//========= Copyright PathRAD contributors. ============//
// light_bounce_vol - fade local bounce boost / chroma with the CLI values.
//=============================================================================//

#include "vrad.h"
#include "bounce_vol.h"
#include "bsplib.h"

struct BounceVolInfo_t
{
	Vector	mins;
	Vector	maxs;
	float	blendDistance;
	int		blendMode;	// 0 inside, 1 outside, 2 center
	int		priority;
	float	volumeSize;
	float	bounceBoost;	// < 0 = inherit CLI
	float	bounceChroma;	// < 0 = inherit CLI
	float	energyMode;		// < 0 = inherit; else 0..1 (0=noenergy, 1=energy)
};

static BounceVolInfo_t s_Volumes[BOUNCEVOL_MAX_VOLUMES];
static int s_nVolumes = 0;

void BounceVol_Clear()
{
	s_nVolumes = 0;
}

bool BounceVol_HasVolumes()
{
	return s_nVolumes > 0;
}

int BounceVol_VolumeCount()
{
	return s_nVolumes;
}

bool BounceVol_GetVolume( int index, BounceVolDesc_t &out )
{
	if ( index < 0 || index >= s_nVolumes )
		return false;
	const BounceVolInfo_t &v = s_Volumes[index];
	out.mins = v.mins;
	out.maxs = v.maxs;
	out.blendDistance = v.blendDistance;
	out.blendMode = v.blendMode;
	out.priority = v.priority;
	out.volumeSize = v.volumeSize;
	out.bounceBoost = v.bounceBoost;
	out.bounceChroma = v.bounceChroma;
	out.energyMode = v.energyMode;
	return true;
}

void BounceVol_ParseVolumeEntity( entity_t *e )
{
	char *pModel = ValueForKey( e, "model" );
	if ( !pModel || pModel[0] != '*' )
	{
		Warning( "WARNING: light_bounce_vol without brush model ignored\n" );
		return;
	}
	int modelIndex = atoi( pModel + 1 );
	if ( modelIndex <= 0 || modelIndex >= nummodels )
	{
		Warning( "WARNING: light_bounce_vol has invalid model \"%s\"\n", pModel );
		return;
	}
	if ( s_nVolumes >= BOUNCEVOL_MAX_VOLUMES )
	{
		Warning( "WARNING: too many light_bounce_vol entities (max %d)\n", BOUNCEVOL_MAX_VOLUMES );
		return;
	}

	dmodel_t *pModelData = &dmodels[modelIndex];
	BounceVolInfo_t &v = s_Volumes[s_nVolumes];
	v.mins = pModelData->mins;
	v.maxs = pModelData->maxs;
	v.blendDistance = FloatForKeyWithDefault( e, "BlendDistance", 0.0f );
	if ( v.blendDistance < 0.0f )
		v.blendDistance = 0.0f;
	v.blendMode = IntForKeyWithDefault( e, "BlendMode", 2 );
	if ( v.blendMode < 0 || v.blendMode > 2 )
		v.blendMode = 2;
	v.priority = IntForKeyWithDefault( e, "priority", 0 );

	v.bounceBoost = FloatForKeyWithDefault( e, "BounceBoost", -1.0f );
	v.bounceChroma = FloatForKeyWithDefault( e, "BounceChroma", -1.0f );
	// EnergyMode: -1 inherit, 0 noenergy, 1 energy (choices in FGD)
	v.energyMode = (float)IntForKeyWithDefault( e, "EnergyMode", -1 );

	if ( v.bounceBoost >= 0.0f && v.bounceBoost > 16.0f )
		v.bounceBoost = 16.0f;
	if ( v.bounceChroma >= 0.0f && v.bounceChroma > 8.0f )
		v.bounceChroma = 8.0f;
	if ( v.energyMode > 1.0f )
		v.energyMode = 1.0f;

	Vector size = v.maxs - v.mins;
	v.volumeSize = max( 0.0f, size.x ) * max( 0.0f, size.y ) * max( 0.0f, size.z );

	const char *pEnergy =
		( v.energyMode < 0.0f ) ? "inherit" :
		( v.energyMode > 0.5f ) ? "on" : "off";
	if ( v.bounceBoost < 0.0f && v.bounceChroma < 0.0f )
	{
		Msg( "light_bounce_vol %d  boost=inherit chroma=inherit energy=%s blend=%.1f priority=%d\n",
			 s_nVolumes + 1, pEnergy, v.blendDistance, v.priority );
	}
	else if ( v.bounceBoost < 0.0f )
	{
		Msg( "light_bounce_vol %d  boost=inherit chroma=%.2f energy=%s blend=%.1f priority=%d\n",
			 s_nVolumes + 1, v.bounceChroma, pEnergy, v.blendDistance, v.priority );
	}
	else if ( v.bounceChroma < 0.0f )
	{
		Msg( "light_bounce_vol %d  boost=%.2f chroma=inherit energy=%s blend=%.1f priority=%d\n",
			 s_nVolumes + 1, v.bounceBoost, pEnergy, v.blendDistance, v.priority );
	}
	else
	{
		Msg( "light_bounce_vol %d  boost=%.2f chroma=%.2f energy=%s blend=%.1f priority=%d\n",
			 s_nVolumes + 1, v.bounceBoost, v.bounceChroma, pEnergy, v.blendDistance, v.priority );
	}
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

static float Membership( int idx, const Vector &pos )
{
	const BounceVolInfo_t &v = s_Volumes[idx];
	if ( v.blendDistance <= 0.0f )
	{
		if ( pos.x < v.mins.x || pos.x > v.maxs.x ||
			 pos.y < v.mins.y || pos.y > v.maxs.y ||
			 pos.z < v.mins.z || pos.z > v.maxs.z )
			return 0.0f;
		return 1.0f;
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
		if ( dist >= 0.0f ) return 1.0f;
		t = 1.0f + dist / v.blendDistance;
		break;
	default:
		t = ( dist + v.blendDistance ) / ( 2.0f * v.blendDistance );
		break;
	}
	return Smoother01( t );
}

void BounceVol_Resolve( const Vector &pos, BounceVolSettings_t &out )
{
	out.boost = g_flBounceBoost;
	out.chroma = g_flBounceChroma;
	out.energy = g_bEnergyConserve ? 1.0f : 0.0f;

	if ( s_nVolumes == 0 )
		return;

	int best = -1;
	float bestW = 0.0f;
	for ( int i = 0; i < s_nVolumes; ++i )
	{
		float w = Membership( i, pos );
		if ( w <= 1e-6f )
			continue;

		if ( best < 0 )
		{
			best = i;
			bestW = w;
			continue;
		}

		const BounceVolInfo_t &a = s_Volumes[best];
		const BounceVolInfo_t &b = s_Volumes[i];
		if ( b.priority != a.priority )
		{
			if ( b.priority > a.priority )
			{
				best = i;
				bestW = w;
			}
			continue;
		}
		if ( w > bestW + 1e-5f )
		{
			best = i;
			bestW = w;
		}
		else if ( fabsf( w - bestW ) <= 1e-5f && b.volumeSize < a.volumeSize )
		{
			best = i;
			bestW = w;
		}
	}

	if ( best < 0 || bestW <= 1e-6f )
		return;

	const BounceVolInfo_t &v = s_Volumes[best];
	if ( v.bounceBoost >= 0.0f )
		out.boost = g_flBounceBoost + ( v.bounceBoost - g_flBounceBoost ) * bestW;
	if ( v.bounceChroma >= 0.0f )
		out.chroma = g_flBounceChroma + ( v.bounceChroma - g_flBounceChroma ) * bestW;
	if ( v.energyMode >= 0.0f )
	{
		const float cliEnergy = g_bEnergyConserve ? 1.0f : 0.0f;
		const float volEnergy = ( v.energyMode > 0.5f ) ? 1.0f : 0.0f;
		out.energy = cliEnergy + ( volEnergy - cliEnergy ) * bestW;
	}
}
