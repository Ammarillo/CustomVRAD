//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Optional baked ambient occlusion (-ao / light_ao / light_ao_vol).
// Cosine-weighted hemisphere rays; modulates final lightmap luxels.
//
//=============================================================================//

#include "vrad.h"
#include "lightmap.h"
#include "ao.h"
#include "vrad_gpu.h"
#include "mathlib/halton.h"
#include <vector>

bool	g_bAO = false;
int		g_nAOSamples = 16;
float	g_flAODistance = 48.0f;
float	g_flAOStrength = 1.0f;
float	g_flAOBias = 0.25f;
bool	g_bAODenoise = false;
int		g_nAODenoiseRadius = 1;
float	g_flAODenoiseStrength = 1.0f;

// Allow rays slightly behind the geometric hemisphere so wall/floor
// contacts occlude each other (pure hemisphere leaves a bright rim).
static const float kAOWrap = 0.25f;
// Fraction of Bias used as along-ray start offset (reduces skim misses).
static const float kAORayBiasScale = 0.5f;
// Mild bright-spike soften only (heavy min-filters caused black voids + blockiness).
static const float kAOSpikeGap = 0.18f;
static const float kAOSpikeSoft = 0.35f;

struct AOVolumeInfo_t
{
	Vector	mins;
	Vector	maxs;
	float	blendDistance;
	int		blendMode;
	int		priority;
	float	volumeSize;
	AOSettings_t settings;
};

static AOVolumeInfo_t s_Volumes[AO_MAX_VOLUMES];
static int s_nVolumes = 0;
static bool s_bParsedGlobalEntity = false;

static void ClampSettings( AOSettings_t &s )
{
	if ( s.samples < 1 )
		s.samples = 1;
	if ( s.samples > 256 )
		s.samples = 256;
	if ( s.distance < 1.0f )
		s.distance = 1.0f;
	if ( s.strength < 0.0f )
		s.strength = 0.0f;
	// Allow >1 to over-darken contacts (final luxel scale is clamped to >= 0).
	if ( s.strength > 8.0f )
		s.strength = 8.0f;
	if ( s.bias < 0.0f )
		s.bias = 0.0f;
}

static void ClampDenoiseGlobals()
{
	if ( g_nAODenoiseRadius < 1 )
		g_nAODenoiseRadius = 1;
	if ( g_nAODenoiseRadius > 4 )
		g_nAODenoiseRadius = 4;
	if ( g_flAODenoiseStrength < 0.0f )
		g_flAODenoiseStrength = 0.0f;
	if ( g_flAODenoiseStrength > 1.0f )
		g_flAODenoiseStrength = 1.0f;
}

static void ReadAOSettingsFromEntity( entity_t *e, AOSettings_t &s, bool bDefaultEnabled )
{
	s.enabled = IntForKeyWithDefault( e, "Enabled", bDefaultEnabled ? 1 : 0 ) != 0;
	s.samples = IntForKeyWithDefault( e, "Samples", g_nAOSamples );
	s.distance = FloatForKeyWithDefault( e, "Distance", g_flAODistance );
	s.strength = FloatForKeyWithDefault( e, "Strength", g_flAOStrength );
	s.bias = FloatForKeyWithDefault( e, "Bias", g_flAOBias );
	ClampSettings( s );
}

static void ReadAODenoiseFromEntity( entity_t *e )
{
	// Only light_ao (global) drives denoise — it's a face filter, not per-volume.
	g_bAODenoise = IntForKeyWithDefault( e, "Denoise", g_bAODenoise ? 1 : 0 ) != 0;
	g_nAODenoiseRadius = IntForKeyWithDefault( e, "DenoiseRadius", g_nAODenoiseRadius );
	g_flAODenoiseStrength = FloatForKeyWithDefault( e, "DenoiseStrength", g_flAODenoiseStrength );
	ClampDenoiseGlobals();
}

void AO_ClearEntities()
{
	s_nVolumes = 0;
	s_bParsedGlobalEntity = false;
}

bool AO_HasVolumes()
{
	return s_nVolumes > 0;
}

int AO_VolumeCount()
{
	return s_nVolumes;
}

bool AO_ShouldRun()
{
	return g_bAO || s_nVolumes > 0;
}

void AO_ParseGlobalEntity( entity_t *e )
{
	if ( s_bParsedGlobalEntity )
	{
		Warning( "WARNING: multiple light_ao entities; using the first\n" );
		return;
	}
	s_bParsedGlobalEntity = true;

	AOSettings_t s;
	ReadAOSettingsFromEntity( e, s, true );

	g_bAO = s.enabled;
	g_nAOSamples = s.samples;
	g_flAODistance = s.distance;
	g_flAOStrength = s.strength;
	g_flAOBias = s.bias;
	ReadAODenoiseFromEntity( e );

	Msg( "light_ao: enabled=%s samples=%d distance=%.1f strength=%.2f bias=%.2f denoise=%s radius=%d\n",
		 g_bAO ? "yes" : "no", g_nAOSamples, g_flAODistance, g_flAOStrength, g_flAOBias,
		 g_bAODenoise ? "yes" : "no", g_nAODenoiseRadius );
}

void AO_ParseVolumeEntity( entity_t *e )
{
	char *pModel = ValueForKey( e, "model" );
	if ( !pModel || pModel[0] != '*' )
	{
		Warning( "WARNING: light_ao_vol without brush model ignored\n" );
		return;
	}

	int modelIndex = atoi( pModel + 1 );
	if ( modelIndex <= 0 || modelIndex >= nummodels )
	{
		Warning( "WARNING: light_ao_vol has invalid model \"%s\"\n", pModel );
		return;
	}

	if ( s_nVolumes >= AO_MAX_VOLUMES )
	{
		Warning( "WARNING: too many light_ao_vol entities (max %d)\n", AO_MAX_VOLUMES );
		return;
	}

	dmodel_t *pModelData = &dmodels[modelIndex];
	AOVolumeInfo_t &v = s_Volumes[s_nVolumes];
	v.mins = pModelData->mins;
	v.maxs = pModelData->maxs;
	v.blendDistance = FloatForKeyWithDefault( e, "BlendDistance", 0.0f );
	if ( v.blendDistance < 0.0f )
		v.blendDistance = 0.0f;
	v.blendMode = IntForKeyWithDefault( e, "BlendMode", AO_BLEND_CENTER );
	if ( v.blendMode < AO_BLEND_INSIDE || v.blendMode > AO_BLEND_CENTER )
		v.blendMode = AO_BLEND_CENTER;
	v.priority = IntForKeyWithDefault( e, "priority", 0 );

	ReadAOSettingsFromEntity( e, v.settings, true );

	Vector size = v.maxs - v.mins;
	float sx = ( size.x > 0.0f ) ? size.x : 0.0f;
	float sy = ( size.y > 0.0f ) ? size.y : 0.0f;
	float sz = ( size.z > 0.0f ) ? size.z : 0.0f;
	v.volumeSize = sx * sy * sz;

	const char *pBlendModeName = "center";
	if ( v.blendMode == AO_BLEND_INSIDE )
		pBlendModeName = "inside";
	else if ( v.blendMode == AO_BLEND_OUTSIDE )
		pBlendModeName = "outside";

	Msg( "light_ao_vol %d  mins(%.0f %.0f %.0f) maxs(%.0f %.0f %.0f) blend %.1f mode %s priority %d enabled=%s samples=%d distance=%.1f strength=%.2f\n",
		 s_nVolumes + 1,
		 v.mins.x, v.mins.y, v.mins.z,
		 v.maxs.x, v.maxs.y, v.maxs.z,
		 v.blendDistance, pBlendModeName, v.priority,
		 v.settings.enabled ? "yes" : "no",
		 v.settings.samples, v.settings.distance, v.settings.strength );

	++s_nVolumes;
}

// ---- volume blend (same softmin / smootherstep / Voronoi idea as light_env_vol) ----

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

static float SignedBlendDistance( const Vector &p, const Vector &mins, const Vector &maxs, float softK )
{
	float dx = min( p.x - mins.x, maxs.x - p.x );
	float dy = min( p.y - mins.y, maxs.y - p.y );
	float dz = min( p.z - mins.z, maxs.z - p.z );

	if ( dx >= 0.0f && dy >= 0.0f && dz >= 0.0f )
		return SoftMin3( dx, dy, dz, softK );

	float ox = 0.0f, oy = 0.0f, oz = 0.0f;
	if ( p.x < mins.x ) ox = mins.x - p.x;
	else if ( p.x > maxs.x ) ox = p.x - maxs.x;
	if ( p.y < mins.y ) oy = mins.y - p.y;
	else if ( p.y > maxs.y ) oy = p.y - maxs.y;
	if ( p.z < mins.z ) oz = mins.z - p.z;
	else if ( p.z > maxs.z ) oz = p.z - maxs.z;
	return -sqrtf( ox * ox + oy * oy + oz * oz );
}

static float SmootherStep01( float t )
{
	if ( t <= 0.0f ) return 0.0f;
	if ( t >= 1.0f ) return 1.0f;
	return t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
}

static float TerritoryFactor( int idx, const Vector &pos, float signedDist )
{
	float factor = 1.0f;
	const AOVolumeInfo_t &v = s_Volumes[idx];

	for ( int j = 0; j < s_nVolumes; ++j )
	{
		if ( j == idx )
			continue;

		const AOVolumeInfo_t &o = s_Volumes[j];
		const float softKJ = ( o.blendDistance > 0.0f ) ? ( o.blendDistance * 0.15f ) : 0.0f;
		const float distJ = SignedBlendDistance( pos, o.mins, o.maxs, softKJ );
		const float advantage = distJ - signedDist;
		if ( advantage <= 0.0f )
			continue;

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
	const AOVolumeInfo_t &v = s_Volumes[idx];

	if ( v.blendDistance <= 0.0f )
	{
		if ( pos.x < v.mins.x || pos.x > v.maxs.x ||
			 pos.y < v.mins.y || pos.y > v.maxs.y ||
			 pos.z < v.mins.z || pos.z > v.maxs.z )
			return 0.0f;
		return 1.0f;
	}

	const float softK = v.blendDistance * 0.15f;
	const float dist = SignedBlendDistance( pos, v.mins, v.maxs, softK );
	const float blend = v.blendDistance;

	float t;
	switch ( v.blendMode )
	{
	case AO_BLEND_INSIDE:
		if ( dist <= 0.0f )
			return 0.0f;
		t = dist / blend;
		break;
	case AO_BLEND_OUTSIDE:
		if ( dist >= 0.0f )
			return 1.0f;
		t = 1.0f + dist / blend;
		break;
	case AO_BLEND_CENTER:
	default:
		t = ( dist + blend ) / ( 2.0f * blend );
		break;
	}

	const float w = SmootherStep01( t );
	if ( w <= 0.0f )
		return 0.0f;
	return w * TerritoryFactor( idx, pos, dist );
}

static void ComputeAllWeights( const Vector &pos, float *pOutWeights )
{
	const int nOut = s_nVolumes + 1;
	for ( int i = 0; i < nOut; ++i )
		pOutWeights[i] = 0.0f;

	if ( s_nVolumes == 0 )
	{
		pOutWeights[0] = 1.0f;
		return;
	}

	float raw[AO_MAX_VOLUMES];
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
				const AOVolumeInfo_t &a = s_Volumes[saturatedIdx];
				const AOVolumeInfo_t &b = s_Volumes[i];
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
		pOutWeights[saturatedIdx + 1] = 1.0f;
		return;
	}

	float sum = 0.0f;
	for ( int i = 0; i < s_nVolumes; ++i )
		sum += raw[i];

	if ( sum <= 1.0f )
	{
		pOutWeights[0] = 1.0f - sum;
		for ( int i = 0; i < s_nVolumes; ++i )
			pOutWeights[i + 1] = raw[i];
		return;
	}

	for ( int i = 0; i < s_nVolumes; ++i )
		pOutWeights[i + 1] = raw[i] / sum;
}

static void GetDefaultSettings( AOSettings_t &s )
{
	s.enabled = g_bAO;
	s.samples = g_nAOSamples;
	s.distance = g_flAODistance;
	s.strength = g_flAOStrength;
	s.bias = g_flAOBias;
	if ( !s.enabled )
		s.strength = 0.0f;
	ClampSettings( s );
}

void AO_ResolveSettings( const Vector &pos, AOSettings_t &out )
{
	AOSettings_t def;
	GetDefaultSettings( def );

	if ( s_nVolumes == 0 )
	{
		out = def;
		return;
	}

	float weights[AO_MAX_VOLUMES + 1];
	ComputeAllWeights( pos, weights );

	float wSum = 0.0f;
	float dist = 0.0f;
	float strength = 0.0f;
	float bias = 0.0f;
	int maxSamples = 1;

	for ( int i = 0; i <= s_nVolumes; ++i )
	{
		float w = weights[i];
		if ( w <= 1e-6f )
			continue;

		const AOSettings_t &s = ( i == 0 ) ? def : s_Volumes[i - 1].settings;
		wSum += w;
		float st = s.enabled ? s.strength : 0.0f;
		dist += w * s.distance;
		strength += w * st;
		bias += w * s.bias;
		if ( s.enabled && s.samples > maxSamples )
			maxSamples = s.samples;
	}

	if ( wSum <= 1e-6f )
	{
		out = def;
		out.strength = 0.0f;
		out.enabled = false;
		return;
	}

	out.enabled = ( strength > 1e-5f );
	out.samples = maxSamples;
	out.distance = dist / wSum;
	out.strength = strength / wSum;
	out.bias = bias / wSum;
	ClampSettings( out );
}

float AOScaleFromFactor( float ao, const Vector &pos )
{
	AOSettings_t s;
	AO_ResolveSettings( pos, s );
	float st = s.strength;
	if ( st < 0.0f ) st = 0.0f;
	// scale = lerp(1, ao, strength); strength>1 boosts darkening.
	float scale = 1.0f - st + st * ao;
	if ( scale < 0.0f )
		scale = 0.0f;
	return scale;
}

// ---- ray AO ----

static void GetLuxelNormal( int facenum, const facelight_t *fl, int luxel, Vector &outNormal )
{
	if ( fl->luxelNormals )
	{
		outNormal = fl->luxelNormals[luxel];
	}
	else
	{
		GetPhongNormal( facenum, fl->luxel[luxel], outNormal );
	}
	if ( VectorLength( outNormal ) < 1.0e-6f )
	{
		outNormal = faceneighbor[facenum].facenormal;
	}
	VectorNormalize( outNormal );
}

// Build ray start/end for one AO sample. Uses wrapped-sphere directions so
// concave contacts (wall/floor) see each other, plus along-ray bias.
static bool MakeAORay( const Vector &pos, const Vector &normal,
					   float flDist, float flBias,
					   DirectionalSampler_t &sampler,
					   Vector &outStart, Vector &outEnd, float &outWeight )
{
	// Try a few sphere samples until one is inside the wrapped cone.
	for ( int attempt = 0; attempt < 8; ++attempt )
	{
		Vector dir = sampler.NextValue();
		float nd = DotProduct( dir, normal );
		if ( nd < -kAOWrap )
			continue;

		float rayBias = max( 0.05f, flBias * kAORayBiasScale );
		outStart = pos + normal * flBias + dir * rayBias;
		outEnd = outStart + dir * flDist;
		// Keep a little weight on near-horizon / slight backface rays so
		// contact occlusion contributes, without over-darkening flats.
		outWeight = max( nd, 0.08f );
		return true;
	}

	// Fallback: straight along normal
	Vector dir = normal;
	float rayBias = max( 0.05f, flBias * kAORayBiasScale );
	outStart = pos + normal * flBias + dir * rayBias;
	outEnd = outStart + dir * flDist;
	outWeight = 1.0f;
	return true;
}

static void TraceAORays4( const Vector *starts, const Vector *ends, const float *weights,
						  int count, float *pSumVis, float *pSumW )
{
	if ( count <= 0 )
		return;

	FourVectors start4, end4;
	for ( int i = 0; i < 4; ++i )
	{
		int src = ( i < count ) ? i : 0;
		start4.X( i ) = starts[src].x;
		start4.Y( i ) = starts[src].y;
		start4.Z( i ) = starts[src].z;
		end4.X( i ) = ends[src].x;
		end4.Y( i ) = ends[src].y;
		end4.Z( i ) = ends[src].z;
	}

	fltx4 fractionVisible = Four_Ones;
	TestLine( start4, end4, &fractionVisible, -1 );

	for ( int i = 0; i < count; ++i )
	{
		float w = weights[i];
		*pSumVis += SubFloat( fractionVisible, i ) * w;
		*pSumW += w;
	}
}

static float ComputeLuxelAO_CPU( const Vector &pos, const Vector &normal, int nSamples, float flDist, float flBias )
{
	DirectionalSampler_t sampler;

	float sumVis = 0.0f;
	float sumW = 0.0f;
	Vector starts[4];
	Vector ends[4];
	float weights[4];
	int pending = 0;

	for ( int s = 0; s < nSamples; ++s )
	{
		float w = 1.0f;
		if ( !MakeAORay( pos, normal, flDist, flBias, sampler, starts[pending], ends[pending], w ) )
			continue;
		weights[pending] = w;
		++pending;
		if ( pending == 4 )
		{
			TraceAORays4( starts, ends, weights, 4, &sumVis, &sumW );
			pending = 0;
		}
	}
	if ( pending > 0 )
		TraceAORays4( starts, ends, weights, pending, &sumVis, &sumW );

	if ( sumW <= 1e-6f )
		return 1.0f;
	return sumVis / sumW;
}

static bool ComputeFaceAO_GPU( int facenum, const facelight_t *fl, float *pAOOut,
							   const int *pSamples, const float *pDist, const float *pBias, int nFaceSamples )
{
	if ( !VRadGPU_HasScene() )
		return false;

	const int nLuxels = fl->numluxels;
	if ( nLuxels <= 0 || nFaceSamples < 1 )
		return false;

	const int nRays = nLuxels * nFaceSamples;
	const int nPad = ( nRays < 32 ) ? 32 : nRays;

	std::vector<Vector> starts( nPad );
	std::vector<Vector> ends( nPad );
	std::vector<float> weights( nPad, 0.0f );
	std::vector<unsigned char> visible( nPad, 1 );
	std::vector<unsigned char> validRay( nPad, 0 );

	DirectionalSampler_t sampler;

	int ray = 0;
	for ( int j = 0; j < nLuxels; ++j )
	{
		Vector normal;
		GetLuxelNormal( facenum, fl, j, normal );

		int nUse = pSamples[j];
		if ( nUse < 1 )
			nUse = 1;
		if ( nUse > nFaceSamples )
			nUse = nFaceSamples;

		for ( int s = 0; s < nFaceSamples; ++s )
		{
			if ( s < nUse )
			{
				float w = 1.0f;
				MakeAORay( fl->luxel[j], normal, pDist[j], pBias[j], sampler,
						   starts[ray], ends[ray], w );
				weights[ray] = w;
				validRay[ray] = 1;
			}
			else
			{
				starts[ray] = starts[0];
				ends[ray] = starts[0];
				weights[ray] = 0.0f;
				validRay[ray] = 0;
			}
			++ray;
		}
	}

	while ( ray < nPad )
	{
		starts[ray] = starts[0];
		ends[ray] = starts[0];
		validRay[ray] = 0;
		weights[ray] = 0.0f;
		++ray;
	}

	if ( !VRadGPU_TraceOcclusion( starts.data(), ends.data(), visible.data(), nPad ) )
		return false;

	for ( int j = 0; j < nLuxels; ++j )
	{
		float sum = 0.0f;
		float wsum = 0.0f;
		int base = j * nFaceSamples;
		for ( int s = 0; s < nFaceSamples; ++s )
		{
			if ( !validRay[base + s] )
				continue;
			float w = weights[base + s];
			sum += ( visible[base + s] ? 1.0f : 0.0f ) * w;
			wsum += w;
		}
		pAOOut[j] = ( wsum > 1e-6f ) ? ( sum / wsum ) : 1.0f;
	}
	return true;
}

// Soften only clear bright outliers next to darker neighbors (no morphological crush).
static void ApplyAOSpikeSoften( float *pAO, int width, int height )
{
	if ( !pAO || width < 2 || height < 2 || kAOSpikeSoft <= 0.0f )
		return;

	const int n = width * height;
	std::vector<float> tmp( n );

	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			float center = pAO[s + t * width];
			float darkest = center;
			float sum = 0.0f;
			int count = 0;

			for ( int dt = -1; dt <= 1; ++dt )
			{
				int nt = t + dt;
				if ( nt < 0 || nt >= height )
					continue;
				for ( int ds = -1; ds <= 1; ++ds )
				{
					int ns = s + ds;
					if ( ns < 0 || ns >= width )
						continue;
					float v = pAO[ns + nt * width];
					sum += v;
					++count;
					if ( v < darkest )
						darkest = v;
				}
			}

			float avg = ( count > 0 ) ? ( sum / (float)count ) : center;
			float out = center;
			// Only pull down bright spikes; never spread darkness into midtones.
			if ( ( center - darkest ) > kAOSpikeGap && center > avg )
			{
				float target = 0.5f * ( avg + darkest );
				out = center + ( target - center ) * kAOSpikeSoft;
			}
			tmp[s + t * width] = out;
		}
	}

	memcpy( pAO, tmp.data(), n * sizeof( float ) );
}

// Edge-preserving bilateral filter on the AO luxel grid (reduces sample noise).
static void ApplyAODenoise( float *pAO, int width, int height )
{
	if ( !pAO || !g_bAODenoise || width < 2 || height < 2 )
		return;

	ClampDenoiseGlobals();
	if ( g_flAODenoiseStrength <= 1e-5f )
		return;

	const int radius = g_nAODenoiseRadius;
	const float spatialSigma = 0.65f * (float)radius + 0.35f;
	const float rangeSigma = 0.12f; // AO units; keeps hard contacts
	const float invTwoSpat = 1.0f / ( 2.0f * spatialSigma * spatialSigma );
	const float invTwoRange = 1.0f / ( 2.0f * rangeSigma * rangeSigma );
	const float blend = g_flAODenoiseStrength;

	const int n = width * height;
	std::vector<float> tmp( n );

	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			const float center = pAO[idx];
			float sum = 0.0f;
			float wsum = 0.0f;

			for ( int dt = -radius; dt <= radius; ++dt )
			{
				int nt = t + dt;
				if ( nt < 0 || nt >= height )
					continue;
				for ( int ds = -radius; ds <= radius; ++ds )
				{
					int ns = s + ds;
					if ( ns < 0 || ns >= width )
						continue;

					float v = pAO[ns + nt * width];
					float spat = (float)( ds * ds + dt * dt );
					float range = v - center;
					float w = expf( -spat * invTwoSpat - range * range * invTwoRange );
					sum += v * w;
					wsum += w;
				}
			}

			float filtered = ( wsum > 1e-8f ) ? ( sum / wsum ) : center;
			tmp[idx] = center + ( filtered - center ) * blend;
		}
	}

	memcpy( pAO, tmp.data(), n * sizeof( float ) );
}

void ComputeFaceAmbientOcclusion( int facenum, const facelight_t *fl, float *pAOOut )
{
	if ( !fl || !pAOOut || fl->numluxels <= 0 )
		return;

	const int nLuxels = fl->numluxels;
	std::vector<int> samples( nLuxels );
	std::vector<float> dist( nLuxels );
	std::vector<float> bias( nLuxels );
	std::vector<float> strength( nLuxels );

	int nFaceSamples = 1;
	bool anyActive = false;

	for ( int j = 0; j < nLuxels; ++j )
	{
		AOSettings_t s;
		AO_ResolveSettings( fl->luxel[j], s );
		if ( do_fast && s.samples > 4 )
			s.samples = max( 4, s.samples / 4 );

		samples[j] = s.samples;
		dist[j] = s.distance;
		bias[j] = s.bias;
		strength[j] = s.strength;
		pAOOut[j] = 1.0f; // clear before GPU solid-luxel path

		if ( s.strength > 1e-5f )
		{
			anyActive = true;
			if ( s.samples > nFaceSamples )
				nFaceSamples = s.samples;
		}
	}

	if ( !anyActive )
		return;

	bool bGPU = ComputeFaceAO_GPU( facenum, fl, pAOOut, samples.data(), dist.data(), bias.data(), nFaceSamples );
	if ( !bGPU )
	{
		for ( int j = 0; j < nLuxels; ++j )
		{
			if ( strength[j] <= 1e-5f )
			{
				pAOOut[j] = 1.0f;
				continue;
			}
			Vector normal;
			GetLuxelNormal( facenum, fl, j, normal );
			pAOOut[j] = ComputeLuxelAO_CPU( fl->luxel[j], normal, samples[j], dist[j], bias[j] );
		}
	}

	dface_t *f = &g_pFaces[facenum];
	int width = f->m_LightmapTextureSizeInLuxels[0] + 1;
	int height = f->m_LightmapTextureSizeInLuxels[1] + 1;
	if ( width * height == nLuxels )
	{
		ApplyAOSpikeSoften( pAOOut, width, height );
		ApplyAODenoise( pAOOut, width, height );
	}
}
