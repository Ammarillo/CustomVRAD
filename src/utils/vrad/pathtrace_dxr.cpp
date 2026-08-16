//========= Copyright CustomVRAD contributors. ============//
//
// DXR path-traced world-face lightmap baker (-pathtrace / -dxr).
// Hardware closest-hit via DXR RayQuery; path integration on CPU with
// soft area/sun sampling and CustomVRAD volume helpers.
//
//=============================================================================//

#include "vrad.h"
#include "oklab.h"
#include "lightmap.h"
#include "pathtrace_dxr.h"
#include "pathtrace_dxr_device.h"
#include "ies_profile.h"
#include "light_projection.h"
#include "pathtrace_denoise.h"
#include "envvolume.h"
#include "skyambient.h"
#include "bounce_vol.h"
#include "bounce_albedo.h"
#include "absorb.h"
#include "water_medium.h"
#include "vrad_emit.h"
#include "vrad_emit_area.h"
#include "vrad_filter.h"
#include "oklab.h"
#include "bsplib.h"
#include "pt_spectral.h"
#include "worldsize.h"
#include "coordsize.h"
#include "tier0/dbg.h"
#include "commonmacros.h"
#include "mathlib/bumpvects.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <vector>
#include <atomic>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool	g_bPathTraceRequested = false;
bool	g_bPathTraceActive = false;
int		g_nPathTraceSamples = 0;	// 0 = auto
int		g_nPathTraceBounces = -1;	// -1 = auto; 0 = direct+sky only
int		g_nPathTraceDevice = -1;
bool	g_bPathTraceDenoise = false;	// opt-in (-pt_denoise)
int		g_nPathTraceDenoiseRadius = 3;	// Sakai spatial radius (OIDN/OptiX ignore)
float	g_flPathTraceDenoiseStrength = 1.0f;	// blend noisy->denoised (1 = full)
int		g_nPathTraceLuxelAA = 3;	// 1=off .. 5=5x5 free spatial AA
int		g_nPathTraceLightSamples = 0;	// 0 = evaluate all local lights (legacy)
int		g_nPathTraceEmitSamples = 64;	// $vrad_emit area NEE samples (0 = all tris)
int		g_nPathTracePropSamples = 0;	// 0 = auto (max(8, world_spp/4))
int		g_nPathTracePropBounces = -1;	// -1 = auto (match world); 0 = direct+sky
int		g_nPathTracePropVertGrid = 4;	// 0=per-vert; >0=8x8 tri atlas + closest gather
bool	g_bPathTraceSpectral = true;	// hero-wavelength Smits+CIE (default on)
float	g_flPtLightRadius = 0.0f;	// soft disk for light/light_spot (0 = hard)
float	g_flPtLightPenumbra = 1.0f;	// CHSS growth scale
int		g_nPtSoftSamples = 16;		// max soft visibility rays per NEE
int		g_nPtSoftMode = 0;			// 0=direct only, 1=all path vertices
bool	g_bPathTraceGpu = true;		// prefer GPU baker when DXR device is ready
bool	g_bPathTraceCpuForced = false;

extern qboolean do_fast;
extern float g_flSkySampleScale;
extern void BuildPatchLights( int facenum );
extern void FreeSampleWindings( facelight_t *fl );
extern void GetBumpNormals( const float* sVect, const float* tVect, const Vector& flatNormal,
					 const Vector& phongNormal, Vector bumpNormals[NUM_BUMP_VECTS] );
extern void CalcPoints( lightinfo_t *pLightInfo, facelight_t *pFaceLight, int ndxFace );
extern bool BuildSamplesAndLuxels_DoFast( lightinfo_t *pLightInfo, facelight_t *pFaceLight, int ndxFace );

void PathTraceDXR_SetRequested( bool bRequested )
{
	g_bPathTraceRequested = bRequested;
}

bool PathTraceDXR_IsRequested()
{
	return g_bPathTraceRequested;
}

bool PathTraceDXR_IsActive()
{
	return g_bPathTraceActive;
}

void PathTraceDXR_Shutdown()
{
	PathTraceDenoise_Shutdown();
	PathTraceDXR_DeviceShutdown();
	g_bPathTraceActive = false;
}

// ---------------------------------------------------------------------------
// RNG / sampling helpers
// ---------------------------------------------------------------------------
static inline float PtRadicalInverse( unsigned int i )
{
	i = ( i << 16 ) | ( i >> 16 );
	i = ( ( i & 0x00ff00ffu ) << 8 ) | ( ( i & 0xff00ff00u ) >> 8 );
	i = ( ( i & 0x0f0f0f0fu ) << 4 ) | ( ( i & 0xf0f0f0f0u ) >> 4 );
	i = ( ( i & 0x33333333u ) << 2 ) | ( ( i & 0xccccccccu ) >> 2 );
	i = ( ( i & 0x55555555u ) << 1 ) | ( ( i & 0xaaaaaaaau ) >> 1 );
	return (float)( i * 2.3283064365386963e-10 );
}

static inline float PtHash( unsigned int x )
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return ( x & 0xFFFFFF ) / 16777216.0f;
}

// Continuous across VBSP coplanar face splits: seed from quantized world pos, not face index.
// Independent face RNGs were the main cause of vertical "strip" seams on walls/floors.
static inline unsigned int PtWorldSeed( const Vector &pos, float luxelWorld, int ni )
{
	const float cell = ( luxelWorld > 0.5f ) ? luxelWorld : 2.0f;
	const int qx = (int)floorf( pos.x / cell );
	const int qy = (int)floorf( pos.y / cell );
	const int qz = (int)floorf( pos.z / cell );
	unsigned int h = (unsigned int)qx * 73856093u ^ (unsigned int)qy * 19349663u ^ (unsigned int)qz * 83492791u;
	h ^= (unsigned int)ni * 2654435761u;
	h ^= h >> 16;
	return h ? h : 1u;
}

static void PtOrthonormalBasis( const Vector &n, Vector &t, Vector &b )
{
	if ( fabsf( n.x ) > 0.1f )
		t.Init( n.y, -n.x, 0 );
	else
		t.Init( 0, n.z, -n.y );
	VectorNormalize( t );
	CrossProduct( n, t, b );
}

static Vector PtCosineHemisphere( const Vector &n, float u1, float u2 )
{
	const float r = sqrtf( u1 );
	const float phi = 2.0f * (float)M_PI * u2;
	Vector t, b;
	PtOrthonormalBasis( n, t, b );
	Vector d = t * ( r * cosf( phi ) ) + b * ( r * sinf( phi ) ) + n * sqrtf( fmaxf( 0.0f, 1.0f - u1 ) );
	VectorNormalize( d );
	return d;
}

static void PtSoftSunDir( const Vector &axis, float sinExtent, int i, int nSoft, unsigned int seed, Vector &out )
{
	Vector t1, t2;
	PtOrthonormalBasis( axis, t1, t2 );
	const float cosThetaMax = sqrtf( fmaxf( 0.0f, 1.0f - sinExtent * sinExtent ) );
	// Stratified + Cranley-Patterson rotate so every luxel gets a unique soft pattern
	// (fixed dirs everywhere created hard penumbra "steps"/blocks).
	const float uRot = PtHash( seed + 17u );
	const float phiRot = PtHash( seed + 91u ) * 2.0f * (float)M_PI;
	float u;
	if ( nSoft <= 1 )
		u = 0.0f;
	else
		u = ( (float)i + uRot ) / (float)nSoft;
	const float cosTheta = 1.0f - u * ( 1.0f - cosThetaMax );
	const float sinTheta = sqrtf( fmaxf( 0.0f, 1.0f - cosTheta * cosTheta ) );
	const float phi = (float)i * 2.3999632f + phiRot;
	out = axis * cosTheta + ( t1 * cosf( phi ) + t2 * sinf( phi ) ) * sinTheta;
	VectorNormalize( out );
}

static void PtSoftSphereOffset( float radius, int i, unsigned int seed, Vector &offsetOut )
{
	if ( radius <= 0.0f )
	{
		offsetOut.Init();
		return;
	}
	const float u = PtHash( seed + (unsigned int)i * 13u + 3u );
	const float r = radius * cbrtf( u );
	const float z = 1.0f - 2.0f * PtHash( seed + (unsigned int)i * 29u + 7u );
	const float phi = PtHash( seed + (unsigned int)i * 47u + 11u ) * 2.0f * (float)M_PI;
	const float sinTheta = sqrtf( fmaxf( 0.0f, 1.0f - z * z ) );
	offsetOut.x = r * sinTheta * cosf( phi );
	offsetOut.y = r * sinTheta * sinf( phi );
	offsetOut.z = r * z;
}

static int PtSoftSunCount( float sinExtent )
{
	if ( sinExtent <= 0.0f )
		return 1;
	float s = min( 1.0f, max( 0.0f, sinExtent ) );
	const float angleDeg = (float)( asin( s ) * ( 180.0 / M_PI ) );
	int n = 8 + (int)( angleDeg * 0.6f );
	if ( n > 40 ) n = 40;
	if ( do_fast )
	{
		n /= 3;
		if ( n < 4 ) n = 4;
	}
	return n;
}

// Soft sample budget cap (respects -pt_softsamples / -fast).
static int PtSoftSampleCap()
{
	int cap = g_nPtSoftSamples;
	if ( cap < 1 )
		cap = 1;
	if ( do_fast )
	{
		cap /= 2;
		if ( cap < 4 )
			cap = 4;
	}
	if ( cap > 64 )
		cap = 64;
	return cap;
}

// PCSS-style penumbra -> sample count. Contact (small tHit) -> few rays; wide gap -> up to cap.
// distLight: receiver->light distance (use large value for sun). tHit: occluder distance along probe.
// blocked: true if probe hit an occluder before the light.
static int PtEstimatePenumbraSamples( float lightSize, float distLight, float tHit, bool blocked,
									  float penumbraScale )
{
	const int cap = PtSoftSampleCap();
	if ( lightSize <= 1e-6f || penumbraScale <= 1e-6f )
		return 1;

	if ( !blocked )
	{
		// Fully lit along center: cheap off-axis soft only.
		const float ang = lightSize / max( distLight, 1.0f );
		int n = 1 + (int)( ang * 10.0f * penumbraScale );
		const int litCap = min( 4, cap );
		if ( n < 1 ) n = 1;
		if ( n > litCap ) n = litCap;
		return n;
	}

	// Distances from light: dReceiver = distLight, dBlocker = distLight - tHit.
	const float dBlocker = max( distLight - tHit, 1e-3f );
	const float wPen = ( tHit * lightSize * penumbraScale ) / dBlocker;
	int n = 1 + (int)( wPen * 0.35f + 0.5f );
	if ( n < 1 ) n = 1;
	if ( n > cap ) n = cap;
	return n;
}

static const int kPtMaxFilterHits = 64;
static const float kPtFilterAdvance = 0.15f;
static const float kPtFilterSheetThick = 12.0f;

static float PtFilterAdvancePast( float t )
{
	float eps = fabsf( t ) * 1e-5f + 1e-3f;
	return t + max( kPtFilterAdvance, eps );
}

static bool PtFilterIsSheetExit( float t, const Vector &faceN, float lastT, const Vector &lastN )
{
	if ( lastT < -1e20f )
		return false;
	if ( ( t - lastT ) > kPtFilterSheetThick )
		return false;
	return DotProduct( faceN, lastN ) < -0.85f;
}

static Vector PtFilterFaceNormal( int facenum )
{
	if ( facenum < 0 || facenum >= numfaces )
		return Vector( 0, 0, 1 );
	return dplanes[g_pFaces[facenum].planenum].normal;
}

// Probe along dir with thin-sheet $vrad_filter any-hit. Returns true if an opaque
// occluder blocks before tmax. outT multiplies RGB transmittance (1 = clear).
static void PtMulWaterSegment( Vector &T, const Vector &a, const Vector &b )
{
	if ( !WaterMedium_IsActive() )
		return;
	T *= WaterMedium_SegmentTransmittance( a, b );
}

// Sky/sun irradiance at a submerged point: Kd only (Mode A). Do not stack with segment T.
static void PtMulWaterSky( Vector &T, const Vector &pos )
{
	if ( !WaterMedium_IsActive() )
		return;
	T *= WaterMedium_DownwellingScale( pos );
}

static bool PtProbeOccluder( const Vector &pos, const Vector &dir, float tmin, float tmax,
							 float *pHitT, bool bSunSkyCountsAsClear, Vector *pTrans = nullptr )
{
	Vector T( 1, 1, 1 );
	if ( tmax <= tmin + 1e-4f )
	{
		if ( pHitT ) *pHitT = tmax;
		if ( pTrans ) *pTrans = T;
		return false;
	}
	const float dlen2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	if ( dlen2 < 1e-12f || !_finite( dir.x ) || !_finite( dir.y ) || !_finite( dir.z ) )
	{
		if ( pHitT ) *pHitT = tmax;
		if ( pTrans ) *pTrans = T;
		return false;
	}

	float curTMin = tmin;
	int lastFace = -2;
	float lastFilterT = -1e30f;
	Vector lastFilterN( 0, 0, 0 );
	for ( int pass = 0; pass < kPtMaxFilterHits + 1; ++pass )
	{
		if ( tmax <= curTMin + 1e-4f )
			break;

		Vector origins[1] = { pos };
		Vector dirs[1] = { dir };
		float tmins[1] = { curTMin };
		float tmaxs[1] = { tmax };
		float t[1];
		unsigned int flags[1], hit[1];
		if ( !PathTraceDXR_TraceClosest( origins, dirs, tmins, tmaxs, t, flags, hit, 1 ) )
		{
			if ( pHitT ) *pHitT = tmax;
			if ( pTrans )
			{
				if ( bSunSkyCountsAsClear )
					PtMulWaterSky( T, pos );
				else
					PtMulWaterSegment( T, pos + dir * tmin, pos + dir * tmax );
				*pTrans = T;
			}
			return false;
		}
		if ( !hit[0] )
		{
			if ( pHitT ) *pHitT = tmax;
			if ( pTrans )
			{
				if ( bSunSkyCountsAsClear )
					PtMulWaterSky( T, pos );
				else
					PtMulWaterSegment( T, pos + dir * tmin, pos + dir * tmax );
				*pTrans = T;
			}
			return false;
		}
		if ( bSunSkyCountsAsClear && ( flags[0] & TRACE_ID_SKY ) )
		{
			if ( pHitT ) *pHitT = tmax;
			if ( pTrans )
			{
				PtMulWaterSky( T, pos );
				*pTrans = T;
			}
			return false;
		}
		if ( bSunSkyCountsAsClear && LightEnv_HasShadowCastFilters() )
		{
			Vector hitP = pos + dir * t[0];
			if ( LightEnv_ShouldIgnoreSkyOccluder( pos, hitP ) )
			{
				if ( pHitT ) *pHitT = tmax;
				if ( pTrans )
				{
					PtMulWaterSky( T, pos );
					*pTrans = T;
				}
				return false;
			}
		}
		if ( flags[0] & TRACE_ID_FILTER )
		{
			const int facenum = (int)( flags[0] & 0x00FFFFFFu );
			Vector faceN = PtFilterFaceNormal( facenum );
			if ( facenum == lastFace || PtFilterIsSheetExit( t[0], faceN, lastFilterT, lastFilterN ) )
			{
				curTMin = PtFilterAdvancePast( t[0] );
				continue;
			}
			lastFace = facenum;
			if ( VRadFilter_ShouldApply( facenum, dir ) )
			{
				Vector hitP = pos + dir * t[0];
				Vector Ft( 1, 1, 1 );
				if ( VRadFilter_SampleAtPos( facenum, hitP, Ft ) )
				{
					Ft.x = max( Ft.x, 1e-4f );
					Ft.y = max( Ft.y, 1e-4f );
					Ft.z = max( Ft.z, 1e-4f );
					T = Oklab_StackFilterLinearSRGB( T, Ft );
				}
				lastFilterT = t[0];
				lastFilterN = faceN;
			}
			curTMin = PtFilterAdvancePast( t[0] );
			continue;
		}
		if ( pHitT ) *pHitT = t[0];
		if ( pTrans )
		{
			// Shadowed: water along path irrelevant (hard block). Keep filter T unused.
			*pTrans = T;
		}
		return true;
	}
	if ( pHitT ) *pHitT = tmax;
	if ( pTrans )
	{
		if ( bSunSkyCountsAsClear )
			PtMulWaterSky( T, pos );
		else
			PtMulWaterSegment( T, pos + dir * tmin, pos + dir * tmax );
		*pTrans = T;
	}
	return false;
}

static float PtApplyFade( const directlight_t *dl, float dist, float falloff )
{
	if ( dl->m_flEndFadeDistance <= dl->m_flStartFadeDistance )
		return falloff;
	if ( dist > dl->m_flEndFadeDistance )
		return 0.0f;
	float t = ( dist - dl->m_flStartFadeDistance ) / ( dl->m_flEndFadeDistance - dl->m_flStartFadeDistance );
	t = min( 1.0f, max( 0.0f, t ) );
	t = 1.0f - t;
	// Quintic smoothstep (same family as stock GatherSampleLight)
	float mult = t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
	return falloff * mult;
}

// Point / spot attenuation: 1 / (c + l*d + q*d^2). CapDist<=0 (calloc) => uncapped.
static float PtPointFalloff( const directlight_t *dl, float dist )
{
	dist = max( dist, 1.0f );
	const float capDist = ( dl->m_flCapDist > 0.0f ) ? dl->m_flCapDist : 1.0e22f;
	float d = min( dist, capDist );
	float falloff = dl->light.constant_attn + dl->light.linear_attn * d + dl->light.quadratic_attn * d * d;
	if ( falloff <= 1e-8f )
		return 0.0f;
	return PtApplyFade( dl, dist, 1.0f / falloff );
}

// emit_surface: stock GatherSampleLight - emitter_cos / (r^2 + R^2).
// R^2 = m_flAreaRadius2 (equiv. disk radius^2 from patch area); 0 => classic 1/r^2.
static float PtSurfaceFalloff( const directlight_t *dl, float dist, float emitterCos )
{
	if ( emitterCos <= 0.0f )
		return 0.0f;
	dist = max( dist, 1.0f );
	float dist2 = dist * dist;
	float denom = dist2;
	if ( dl->m_flAreaRadius2 > 0.0f )
		denom = dist2 + dl->m_flAreaRadius2;
	if ( denom <= 1e-8f )
		return 0.0f;
	return PtApplyFade( dl, dist, emitterCos / denom );
}

static float PtSpotCone( const directlight_t *dl, const Vector &lightToSample )
{
	// lightToSample = direction from light toward the receiving sample.

	// Cubemap ProjectedTexture: omnidirectional - no stock cone.
	if ( Proj_IsOmniEnvmap( dl ) )
	{
		if ( dl->m_pIes )
		{
			float w = IES_Eval( dl->m_pIes, dl->light.normal, dl->m_vecIesRight, lightToSample );
			return w * dl->m_flIesScale;
		}
		return 1.0f;
	}

	// IES photometric profile replaces _inner_cone / _cone / _exponent entirely.
	if ( dl->m_pIes )
	{
		float w = IES_Eval( dl->m_pIes, dl->light.normal, dl->m_vecIesRight, lightToSample );
		return w * dl->m_flIesScale;
	}

	Vector dir = lightToSample;
	VectorNormalize( dir );
	float dot2 = DotProduct( dir, dl->light.normal );
	if ( dot2 <= dl->light.stopdot2 )
		return 0.0f;
	if ( dot2 >= dl->light.stopdot )
		return 1.0f;
	float mult = ( dot2 - dl->light.stopdot2 ) / ( dl->light.stopdot - dl->light.stopdot2 );
	mult = max( 0.0f, min( 1.0f, mult ) );
	if ( dl->light.exponent != 0.0f && dl->light.exponent != 1.0f )
		mult = powf( mult, dl->light.exponent );
	return mult;
}

// Disk soft sample in the emitter plane (area / texlight penumbra).
static void PtSoftDiskOffset( const Vector &normal, float radius, int i, unsigned int seed, Vector &offsetOut )
{
	if ( radius <= 0.0f )
	{
		offsetOut.Init();
		return;
	}
	Vector t1, t2;
	PtOrthonormalBasis( normal, t1, t2 );
	const float u = PtHash( seed + (unsigned int)i * 19u + 5u );
	const float r = radius * sqrtf( u );
	const float phi = PtHash( seed + (unsigned int)i * 41u + 13u ) * 2.0f * (float)M_PI;
	offsetOut = t1 * ( r * cosf( phi ) ) + t2 * ( r * sinf( phi ) );
}

static Vector PtClampFirefly( const Vector &v, float maxComp )
{
	Vector o = v;
	float m = max( o.x, max( o.y, o.z ) );
	if ( m > maxComp && m > 0.0f )
		o *= ( maxComp / m );
	return o;
}

static float PtLuma( const Vector &v )
{
	return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

// Fast albedo: bound cluster walk (adaptivechop maps can have huge patch lists).
// bApplyArt: apply -bounce_boost / -bounce_chroma / light_bounce_vol (false when
// uploading raw TriAlbedo to the GPU - shader applies art with bounce falloff).
static Vector PtGetHitAlbedo( const Vector &hitPos, const Vector &hitNormal, bool bAllowTexSample,
							  bool bApplyArt = true, int bounceIndex = 0 )
{
	Vector albedo( 0.45f, 0.45f, 0.45f );
	const int cluster = ClusterFromPoint( hitPos );
	if ( cluster >= 0 && cluster < clusterChildren.Count() )
	{
		float best = 1e30f;
		int bestFace = -1;
		Vector bestRefl = albedo;
		int checked = 0;
		for ( int p = clusterChildren[cluster]; p != g_Patches.InvalidIndex(); p = g_Patches[p].ndxNextClusterChild )
		{
			if ( ++checked > 64 )
				break;
			CPatch *patch = &g_Patches[p];
			Vector d = hitPos - patch->origin;
			float dist2 = DotProduct( d, d );
			float nd = DotProduct( patch->normal, hitNormal );
			if ( nd < 0.15f )
				continue;
			if ( dist2 < best )
			{
				best = dist2;
				bestFace = patch->faceNumber;
				bestRefl = patch->reflectivity;
			}
		}
		albedo = bestRefl;
		if ( bAllowTexSample && g_bTexturedBounce && bestFace >= 0 )
		{
			Vector tex;
			if ( BounceAlbedo_SampleFace( bestFace, hitPos, tex ) )
				albedo = tex;
			else
				BounceAlbedo_SanitizeCompressionChroma( albedo );
		}
		else
		{
			BounceAlbedo_SanitizeCompressionChroma( albedo );
		}
	}

	if ( !bApplyArt )
		return albedo;
	if ( !BounceVol_HasVolumes() && g_flBounceBoost == 1.0f && g_flBounceChroma <= 0.0f )
		return albedo;

	BounceVolSettings_t bv;
	BounceVol_Resolve( hitPos, bv );
	albedo *= bv.boost;

	if ( bv.chroma > 0.0f )
	{
		const float falloff = 1.0f / ( 1.0f + (float)max( 0, bounceIndex ) );
		const float sat = 1.0f + bv.chroma * falloff;
		albedo = Oklab_ScaleChroma( albedo, sat );
	}
	return albedo;
}

// ---------------------------------------------------------------------------
// Visibility helpers
// ---------------------------------------------------------------------------
// Near-hit bias (world units): hits closer than this from a luxel are ignored so
// corner/edge samples act as if they sit in front of the adjacent occluder.
static float g_flPtOccludeBias = 2.5f;

// RGB transmittance from->to (1 = clear, 0 = blocked). Multiplies through $vrad_filter panes.
static Vector PtVisibilityRGB( const Vector &from, const Vector &to )
{
	Vector delta = to - from;
	float dist = VectorNormalize( delta );
	if ( dist <= g_flPtOccludeBias * 2.0f )
	{
		Vector T( 1, 1, 1 );
		PtMulWaterSegment( T, from, to );
		return T;
	}

	const float tmax = dist - min( g_flPtOccludeBias, dist * 0.25f );
	Vector T;
	const bool blocked = PtProbeOccluder( from, delta, g_flPtOccludeBias, tmax, nullptr, false, &T );
	if ( blocked )
		return Vector( 0, 0, 0 );
	return T;
}

static float PtVisibility( const Vector &from, const Vector &to )
{
	Vector T = PtVisibilityRGB( from, to );
	return max( T.x, max( T.y, T.z ) );
}

// Place luxel origin in front of nearby side walls (floor/wall corners, jambs).
static Vector PtLuxelOriginInFront( const Vector &samplePos, const Vector &faceNormal, float luxelWorld )
{
	const float baseBias = max( 1.0f, max( g_flPtOccludeBias * 0.4f, luxelWorld * 0.35f ) );
	Vector pos = samplePos + faceNormal * baseBias;

	Vector t1, t2;
	PtOrthonormalBasis( faceNormal, t1, t2 );
	const float probe = max( g_flPtOccludeBias, luxelWorld * 1.25f );
	float extra = 0.0f;

	for ( int i = 0; i < 8; ++i )
	{
		const float ang = (float)i * ( 2.0f * (float)M_PI / 8.0f );
		Vector dir = t1 * cosf( ang ) + t2 * sinf( ang );
		Vector origins[1] = { pos };
		Vector dirs[1] = { dir };
		float tmins[1] = { 0.1f };
		float tmaxs[1] = { probe };
		float t[1];
		unsigned int flags[1], hit[1];
		Vector hitN[1];
		if ( !PathTraceDXR_TraceClosest( origins, dirs, tmins, tmaxs, t, flags, hit, 1, hitN ) )
			continue;
		if ( !hit[0] || ( flags[0] & TRACE_ID_SKY ) )
			continue;
		// Side occluder ~= wall meeting this face (hit normal perp face normal).
		if ( fabsf( DotProduct( hitN[0], faceNormal ) ) > 0.55f )
			continue;
		extra = max( extra, ( probe - t[0] ) * 0.5f + 0.5f );
	}

	const float maxExtra = max( probe, luxelWorld * 2.0f );
	if ( extra > maxExtra )
		extra = maxExtra;
	return pos + faceNormal * extra;
}

enum
{
	PT_NEE_SOFT		= 0x01,
	PT_NEE_SURFACE	= 0x02,
};

static int g_ptSpp = 16;
static int g_ptBounces = 3;

// Veach power heuristic (beta = 2).
static inline float PtMisWeight( float pdfA, float pdfB )
{
	if ( pdfA <= 0.0f )
		return 0.0f;
	if ( pdfB <= 0.0f )
		return 1.0f;
	const float a2 = pdfA * pdfA;
	const float b2 = pdfB * pdfB;
	return a2 / ( a2 + b2 );
}

// ---------------------------------------------------------------------------
// Direct lighting NEE with MIS vs cosine-hemisphere BSDF pdf.
// Sky lights: always all (few). Local lights: all, or K power-sampled (-pt_lights).
// Convention matches stock GatherSampleLight-style irradiance (no albedo here).
// ---------------------------------------------------------------------------
static std::vector<directlight_t *> g_ptSkyLights;
static std::vector<directlight_t *> g_ptLocalLights;	// point/spot only (power-sampled)
static std::vector<directlight_t *> g_ptEmitLights;		// emit_surface - always all NEE (stock-style)
static std::vector<float> g_ptLocalCdf;	// inclusive prefix of power weights
static float g_ptLocalWeightSum = 0.0f;

static float PtLightPower( const directlight_t *dl )
{
	const Vector &I = dl->light.intensity;
	return max( I.x + I.y + I.z, 1e-4f );
}

static void PtRebuildLightLists()
{
	g_ptSkyLights.clear();
	g_ptLocalLights.clear();
	g_ptEmitLights.clear();
	g_ptLocalCdf.clear();
	g_ptLocalWeightSum = 0.0f;

	for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
	{
		if ( dl->light.style != 0 )
			continue;
		switch ( dl->light.type )
		{
		case emit_skylight:
			g_ptSkyLights.push_back( dl );
			break;
		case emit_skyambient:
			break;
		case emit_surface:
			// $vrad_emit* faces use textured mesh area lights (PBRT NEE).
			if ( VRadEmitArea_FaceHasMesh( dl->facenum ) )
				break;
			g_ptEmitLights.push_back( dl );
			break;
		case emit_point:
		case emit_spotlight:
			g_ptLocalLights.push_back( dl );
			break;
		default:
			break;
		}
	}

	g_ptLocalCdf.resize( g_ptLocalLights.size() );
	float sum = 0.0f;
	for ( size_t i = 0; i < g_ptLocalLights.size(); ++i )
	{
		sum += PtLightPower( g_ptLocalLights[i] );
		g_ptLocalCdf[i] = sum;
	}
	g_ptLocalWeightSum = sum;

	Msg( "[PathTrace-DXR] Lights: %d sky, %d emit_surface (NEE=all), %d local",
		 (int)g_ptSkyLights.size(), (int)g_ptEmitLights.size(), (int)g_ptLocalLights.size() );
	if ( g_nPathTraceLightSamples > 0 && (int)g_ptLocalLights.size() > g_nPathTraceLightSamples )
		Msg( " (NEE samples %d / %d point/spot)\n", g_nPathTraceLightSamples, (int)g_ptLocalLights.size() );
	else
		Msg( " (NEE=all point/spot)\n" );
}

static int PtPickLocalLight( unsigned int seed, int draw )
{
	const int n = (int)g_ptLocalLights.size();
	if ( n <= 0 || g_ptLocalWeightSum <= 0.0f )
		return -1;
	// Scramble per draw so K samples aren't correlated.
	unsigned int h = seed ^ (unsigned int)( draw * 0x9E3779B9u );
	h ^= h >> 16;
	h *= 0x7FEB352Du;
	h ^= h >> 15;
	const float u = ( ( h >> 8 ) & 0xFFFFFFu ) / 16777216.0f;
	const float target = u * g_ptLocalWeightSum;
	int lo = 0, hi = n - 1;
	while ( lo < hi )
	{
		const int mid = ( lo + hi ) >> 1;
		if ( g_ptLocalCdf[mid] < target )
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

// Evaluate one light's NEE contribution at (pos,normal). Adds to sunAmt for skylight.
static Vector PtEvalOneDirectLight( directlight_t *dl, const Vector &pos, const Vector &normal,
									unsigned int seed, float *pSunAmt, int facenum, unsigned neeFlags )
{
	Vector sum( 0, 0, 0 );
	const bool bSoft = ( neeFlags & PT_NEE_SOFT ) != 0;
	const bool bSurface = ( neeFlags & PT_NEE_SURFACE ) != 0;
	const float absScale = Absorb_DirectScale( pos );
	const float invPi = 1.0f / (float)M_PI;

	switch ( dl->light.type )
	{
	case emit_skylight:
	{
		float wEnv = LightEnv_GetWeight( dl->m_nEnvId, pos );
		if ( wEnv <= 0.0f )
			break;

		Vector axis = -dl->light.normal;
		VectorNormalize( axis );
		if ( DotProduct( normal, axis ) <= 0.0f )
			break;

		float sinExt = 0.0f;
		int nSoft = 1;
		if ( bSoft && dl->m_flSunAngularExtent > 0.0f )
		{
			sinExt = dl->m_flSunAngularExtent;
			// Treat angular size as light "radius" at unit distance; probe along axis.
			float tHit = (float)MAX_TRACE_LENGTH;
			Vector Tprobe;
			const bool blocked = PtProbeOccluder( pos, axis, g_flPtOccludeBias, (float)MAX_TRACE_LENGTH,
												  &tHit, true, &Tprobe );
			const bool hardBlocked = blocked || ( Tprobe.x + Tprobe.y + Tprobe.z < 1e-4f );
			// Equivalent world size at a reference distance so PCSS formula is dimensionally ok.
			const float refDist = max( tHit, 64.0f );
			const float lightSize = sinExt * refDist;
			nSoft = PtEstimatePenumbraSamples( lightSize, refDist, tHit, hardBlocked, g_flPtLightPenumbra );
			// Also never exceed legacy soft-sun budget for huge extents.
			const int legacy = PtSoftSunCount( sinExt );
			if ( nSoft > legacy )
				nSoft = legacy;
		}

		const float cosThetaMax = sqrtf( fmaxf( 0.0f, 1.0f - sinExt * sinExt ) );
		const float sunSolidAngle = max( 1e-6f, 2.0f * (float)M_PI * ( 1.0f - cosThetaMax ) );
		const float pdfLight = 1.0f / sunSolidAngle;

		Vector visAccum( 0, 0, 0 );
		float ndlAccum = 0.0f;
		float misAccum = 0.0f;
		float nVis = 0.0f;
		for ( int s = 0; s < nSoft; ++s )
		{
			Vector dir;
			PtSoftSunDir( axis, sinExt, s, nSoft, seed + (unsigned int)s * 97u, dir );
			const float ndlS = DotProduct( normal, dir );
			if ( ndlS <= 0.0f )
				continue;

			float tHit = 0.0f;
			Vector T;
			const bool blocked = PtProbeOccluder( pos, dir, g_flPtOccludeBias, (float)MAX_TRACE_LENGTH,
												  &tHit, true, &T );
			if ( blocked || ( T.x + T.y + T.z < 1e-6f ) )
				continue;

			const float pdfBsdf = ndlS * invPi;
			const float mis = PtMisWeight( pdfLight, pdfBsdf );
			visAccum += T;
			ndlAccum += ndlS;
			misAccum += mis;
			nVis += 1.0f;
		}
		if ( nVis > 0.0f )
		{
			const Vector vis = visAccum * ( 1.0f / (float)nSoft );
			const float ndlSoft = ndlAccum / nVis;
			const float mis = misAccum / nVis;
			sum.x += dl->light.intensity.x * ( ndlSoft * vis.x * wEnv * absScale * mis );
			sum.y += dl->light.intensity.y * ( ndlSoft * vis.y * wEnv * absScale * mis );
			sum.z += dl->light.intensity.z * ( ndlSoft * vis.z * wEnv * absScale * mis );
			if ( pSunAmt )
				*pSunAmt += ( ( vis.x + vis.y + vis.z ) * ( 1.0f / 3.0f ) ) * wEnv;
		}
		break;
	}
	case emit_surface:
	{
		if ( !bSurface )
			break;
		if ( facenum >= 0 && dl->facenum == facenum )
			break;

		// Hard NEE to face center. Soft-disk sampling used sqrt(area/pi) as radius -
		// on medium/large $vrad_emit faces that disk extends into solids and every
		// soft sample fails occlusion -> zero light. Softness comes from area falloff
		// (cos/(r^2+R^2)) instead; optional small CHSS disk only when soft is on.
		const float areaR2 = max( dl->m_flAreaRadius2, 1.0f );
		float softRadius = 0.0f;
		int ns = 1;
		if ( bSoft )
		{
			softRadius = min( 8.0f, sqrtf( areaR2 ) * 0.15f );
			if ( softRadius > 0.5f )
			{
				Vector toLight = dl->light.origin - pos;
				float distL = VectorNormalize( toLight );
				if ( distL > g_flPtOccludeBias + 1.0f )
				{
					float tHit = distL;
					const bool blocked = PtProbeOccluder( pos, toLight, g_flPtOccludeBias, distL, &tHit, false );
					ns = PtEstimatePenumbraSamples( softRadius, distL, tHit, blocked, g_flPtLightPenumbra );
				}
			}
		}
		Vector accum( 0, 0, 0 );
		for ( int s = 0; s < ns; ++s )
		{
			Vector offset( 0, 0, 0 );
			// Always keep sample 0 at the center so large faces still contribute.
			if ( s > 0 && softRadius > 0.5f )
				PtSoftDiskOffset( dl->light.normal, softRadius, s, seed + (unsigned int)s * 31u, offset );
			// Push well off the emitter plane so DXR shadow rays don't self-hit.
			Vector lightPos = dl->light.origin + offset + dl->light.normal * 1.0f;

			Vector delta = lightPos - pos;
			float dist = VectorNormalize( delta );
			float ndl = DotProduct( normal, delta );
			if ( ndl <= 0.0f )
				continue;

			float emitterCos = -DotProduct( delta, dl->light.normal );
			if ( emitterCos <= 0.0f )
				continue;

			float falloff = PtSurfaceFalloff( dl, dist, emitterCos );
			if ( falloff <= 0.0f )
				continue;

			Vector est = dl->light.intensity * ( ndl * falloff * absScale );
			if ( est.x + est.y + est.z < 0.02f )
				continue;

			Vector vis = PtVisibilityRGB( pos, lightPos );
			if ( vis.x + vis.y + vis.z < 1e-6f )
				continue;

			// No MIS: stock GatherSampleLight is NEE-only for emit_surface.
			accum.x += dl->light.intensity.x * ( ndl * falloff * vis.x * absScale );
			accum.y += dl->light.intensity.y * ( ndl * falloff * vis.y * absScale );
			accum.z += dl->light.intensity.z * ( ndl * falloff * vis.z * absScale );
		}
		accum *= ( 1.0f / (float)ns );
		sum += accum;
		break;
	}
	case emit_point:
	case emit_spotlight:
	{
		// Entity Radius (light_volume) wins; else CLI -pt_lightradius for light/light_spot.
		float radius = dl->m_flVolumeRadius;
		if ( radius <= 0.0f )
			radius = g_flPtLightRadius;

		Vector toCenter = dl->light.origin - pos;
		float distCenter = VectorNormalize( toCenter );
		if ( distCenter < 1e-3f )
			break;

		int ns = 1;
		if ( bSoft && radius > 0.0f )
		{
			// Too close for a stable soft probe (tmax would be <= occlude bias).
			if ( distCenter > g_flPtOccludeBias + 1.0f )
			{
				float tHit = distCenter;
				const bool blocked = PtProbeOccluder( pos, toCenter, g_flPtOccludeBias, distCenter,
													  &tHit, false );
				ns = PtEstimatePenumbraSamples( radius, distCenter, tHit, blocked, g_flPtLightPenumbra );
			}
		}

		// Facing disk toward the receiver (contact-hardening area light).
		Vector diskN = -toCenter;

		Vector accum( 0, 0, 0 );
		for ( int s = 0; s < ns; ++s )
		{
			Vector offset;
			offset.Init();
			if ( radius > 0.0f && ns > 1 )
				PtSoftDiskOffset( diskN, radius, s, seed + (unsigned int)s * 53u, offset );

			Vector lightPos = dl->light.origin + offset;

			Vector delta = lightPos - pos;
			float dist = VectorNormalize( delta );
			if ( dist < 1e-3f || !_finite( dist ) )
				continue;
			float ndl = DotProduct( normal, delta );
			if ( ndl <= 0.0f )
				continue;

			if ( dl->light.type == emit_spotlight )
			{
				// lightToSample = sample - light = -delta (delta is light - sample)
				Vector lightToSample = -delta;
				float cone = PtSpotCone( dl, lightToSample );
				if ( cone <= 0.0f )
					continue;
				ndl *= cone;
			}

			float falloff = PtPointFalloff( dl, dist );
			if ( falloff <= 0.0f )
				continue;

			Vector intensity = dl->light.intensity;
			if ( dl->light.type == emit_spotlight && dl->m_pProj )
			{
				Vector rgb = Proj_EvalLightRGB( dl, -delta );
				intensity.x *= rgb.x;
				intensity.y *= rgb.y;
				intensity.z *= rgb.z;
			}

			Vector est = intensity * ( ndl * falloff * absScale );
			if ( est.x + est.y + est.z < 0.02f )
				continue;

			Vector vis = PtVisibilityRGB( pos, lightPos );
			if ( vis.x + vis.y + vis.z < 1e-6f )
				continue;

			accum.x += intensity.x * ( ndl * falloff * vis.x * absScale );
			accum.y += intensity.y * ( ndl * falloff * vis.y * absScale );
			accum.z += intensity.z * ( ndl * falloff * vis.z * absScale );
		}
		accum *= ( 1.0f / (float)ns );
		sum += IES_ClampDirectSample( dl, accum );
		break;
	}
	default:
		break;
	}
	// Sky/sun water attenuation is already in the probe transmittance (Kd / Mode C skip).
	return sum;
}

static int PtPickEmitTri( unsigned int seed, int draw )
{
	const int n = VRadEmitArea_TriCount();
	const float *cdf = VRadEmitArea_PowerCdf();
	if ( n <= 0 || !cdf || VRadEmitArea_PowerSum() <= 0.0f )
		return -1;
	unsigned int h = seed ^ (unsigned int)( draw * 0x85EBCA6Bu );
	h ^= h >> 16;
	h *= 0x7FEB352Du;
	h ^= h >> 15;
	const float u = ( ( h & 0xFFFFFFu ) + 1u ) * ( 1.0f / 16777217.0f );
	int lo = 0, hi = n;
	while ( lo + 1 < hi )
	{
		const int mid = ( lo + hi ) >> 1;
		if ( cdf[mid] < u )
			lo = mid;
		else
			hi = mid;
	}
	return lo;
}

// Uniform area sample on triangle (PBRT UniformSampleTriangle).
static void PtSampleTriangle( const VRadEmitTri_t &t, float u0, float u1, Vector &outP, Vector &outE )
{
	const float su0 = sqrtf( max( u0, 0.0f ) );
	const float b0 = 1.0f - su0;
	const float b1 = u1 * su0;
	const float b2 = 1.0f - b0 - b1;
	outP = t.v0 * b0 + t.v1 * b1 + t.v2 * b2;
	outE = t.e0 * b0 + t.e1 * b1 + t.e2 * b2;
}

// PBRT DiffuseAreaLight NEE: power-pick triangle (or all tris), uniform area sample,
// textured L_e via vertex colors / live sample. -pt_emit_samples controls K.
static Vector PtSampleEmitArea( const Vector &pos, const Vector &normal, unsigned int seed,
								int skipFace, int nSamples )
{
	Vector sum( 0, 0, 0 );
	const int nTris = VRadEmitArea_TriCount();
	if ( nTris <= 0 || VRadEmitArea_PowerSum() <= 0.0f )
		return sum;

	const float *cdf = VRadEmitArea_PowerCdf();
	const VRadEmitTri_t *tris = VRadEmitArea_Tris();
	const float absScale = Absorb_DirectScale( pos );

	// 0 or K>=nTris -> evaluate every triangle once (lowest variance).
	const bool sampleAll = ( nSamples <= 0 || nSamples >= nTris );
	const int k = sampleAll ? nTris : nSamples;

	for ( int s = 0; s < k; ++s )
	{
		int idx;
		float pPick;
		if ( sampleAll )
		{
			idx = s;
			pPick = 1.0f; // each tri once; pdf_A only
		}
		else
		{
			// Stratified pick in CDF domain: less clumping than pure random.
			const float uPick = ( (float)s + PtHash( seed + (unsigned)s * 91u + 7u ) ) / (float)k;
			unsigned int h = seed ^ (unsigned int)( s * 0x85EBCA6Bu );
			(void)h;
			int lo = 0, hi = nTris;
			while ( lo + 1 < hi )
			{
				const int mid = ( lo + hi ) >> 1;
				if ( cdf[mid] < uPick )
					lo = mid;
				else
					hi = mid;
			}
			idx = lo;
			const float cdf0 = ( idx == 0 ) ? 0.0f : cdf[idx - 1];
			pPick = max( cdf[idx] - cdf0, 1e-12f );
		}

		const VRadEmitTri_t &t = tris[idx];
		if ( skipFace >= 0 && t.facenum == skipFace )
			continue;

		const float u0 = PtHash( seed + (unsigned)s * 97u + 11u );
		const float u1 = PtHash( seed + (unsigned)s * 131u + 23u );
		Vector lightPos, Le;
		PtSampleTriangle( t, u0, u1, lightPos, Le );
		lightPos += t.n * 1.0f;

		Vector live;
		if ( VRadEmit_SampleAtPos( t.facenum, lightPos - t.n * 1.0f, live ) )
			VectorScale( live, lightscale * ( 100.0f * 100.0f ) / ( 128.0f * 128.0f ), Le );

		Vector delta = lightPos - pos;
		float dist = VectorNormalize( delta );
		if ( dist < 1e-3f )
			continue;
		const float ndl = DotProduct( normal, delta );
		if ( ndl <= 0.0f )
			continue;
		const float emitterCos = -DotProduct( delta, t.n );
		if ( emitterCos <= 0.0f )
			continue;

		const float pdfA = 1.0f / max( t.area, 1e-8f );
		const float pdf = sampleAll ? pdfA : ( pPick * pdfA );
		if ( pdf <= 1e-20f )
			continue;

		Vector vis = PtVisibilityRGB( pos, lightPos );
		if ( vis.x + vis.y + vis.z < 1e-6f )
			continue;

		sum.x += Le.x * ( ndl * emitterCos * absScale * vis.x / ( dist * dist * pdf ) );
		sum.y += Le.y * ( ndl * emitterCos * absScale * vis.y / ( dist * dist * pdf ) );
		sum.z += Le.z * ( ndl * emitterCos * absScale * vis.z / ( dist * dist * pdf ) );
	}
	if ( !sampleAll )
		sum *= ( 1.0f / (float)k );
	return sum;
}

static Vector PtSampleDirect( const Vector &pos, const Vector &normal, unsigned int seed, float *pSunAmt,
							  int facenum, unsigned neeFlags )
{
	Vector sum( 0, 0, 0 );
	float sunAmt = 0.0f;

	for ( size_t i = 0; i < g_ptSkyLights.size(); ++i )
		sum += PtEvalOneDirectLight( g_ptSkyLights[i], pos, normal, seed, &sunAmt, facenum, neeFlags );

	// lights.rad texlights still as emit_surface point proxies (not $vrad_emit mesh).
	for ( size_t i = 0; i < g_ptEmitLights.size(); ++i )
		sum += PtEvalOneDirectLight( g_ptEmitLights[i], pos, normal, seed + (unsigned int)i * 19u,
									 nullptr, facenum, neeFlags );

	// $vrad_emit*: textured triangle area lights (not a grid of face centers).
	if ( VRadEmitArea_TriCount() > 0 )
		sum += PtSampleEmitArea( pos, normal, seed + 0xA11Eu, facenum, g_nPathTraceEmitSamples );

	const int nLocal = (int)g_ptLocalLights.size();
	int k = g_nPathTraceLightSamples;
	if ( k <= 0 || k >= nLocal )
	{
		for ( int i = 0; i < nLocal; ++i )
			sum += PtEvalOneDirectLight( g_ptLocalLights[i], pos, normal, seed + (unsigned int)i * 17u,
										 nullptr, facenum, neeFlags );
	}
	else
	{
		for ( int s = 0; s < k; ++s )
		{
			const int idx = PtPickLocalLight( seed, s );
			if ( idx < 0 )
				break;
			const float p = PtLightPower( g_ptLocalLights[idx] ) / g_ptLocalWeightSum;
			if ( p <= 1e-12f )
				continue;
			Vector c = PtEvalOneDirectLight( g_ptLocalLights[idx], pos, normal,
											 seed + (unsigned int)( s + 1 ) * 131u,
											 nullptr, facenum, neeFlags );
			sum += c * ( 1.0f / ( p * (float)k ) );
		}
	}

	if ( pSunAmt )
		*pSunAmt = sunAmt;
	return sum;
}

static Vector PtSampleSkyAmbient( const Vector &pos, const Vector &dir )
{
	(void)dir;
	Vector intensity;
	SkyAmbient_ComputeAtPos( pos, intensity );
	intensity *= Absorb_DirectScale( pos );
	if ( WaterMedium_IsActive() )
		intensity *= WaterMedium_DownwellingScale( pos );
	return intensity;
}

// BSDF hit on emit_surface. Disabled: emit lights are NEE-only (stock GatherSampleLight).
// Re-enabling without MIS on surface NEE would double-count.
static bool PtSurfaceEmissionContrib( const Vector &prevPos, const Vector &prevNormal,
									  const Vector &hitPos, const Vector &hitNormal,
									  int skipFace, Vector &outContrib, float &outPdfSa )
{
	(void)prevPos; (void)prevNormal; (void)hitPos; (void)hitNormal; (void)skipFace;
	outContrib.Init();
	outPdfSa = 0.0f;
	return false;
}

// Full unidirectional path tracer (Veach/PBRT style):
// NEE+MIS at every vertex (incl. luxel = direct), BSDF bounce, emissive hits with MIS, RR.
// With -uw_volume: homogeneous free-flight medium events inside water.
static Vector PtPathLi( const Vector &posIn, const Vector &normalIn, unsigned int seed, int sppIndex,
						int maxBounces, int facenum, float *pSunAmt )
{
	Vector radiance( 0, 0, 0 );
	Vector throughput( 1, 1, 1 );
	Vector pos = posIn;
	Vector normal = normalIn;
	int skipFace = facenum;
	float sunAmt = 0.0f;
	const float invPi = 1.0f / (float)M_PI;
	const bool bVolMode = WaterMedium_IsActive() && WaterMedium_GetMode() == WATER_MEDIUM_VOLUME;
	int mediumScatters = 0;
	const int maxMed = WaterMedium_MaxScatter();
	bool bInMedium = false;
	Vector medDir( 0, 0, 1 );

	// -pt_bounces N = N indirect hops after the luxel (primary NEE). Path depth = N+1.
	int pathDepth = maxBounces + 1;
	if ( pathDepth < 1 )
		pathDepth = 1;

	for ( int bounce = 0; bounce < pathDepth; ++bounce )
	{
		// --- Next event estimation (all light types) + MIS ---
		float sunB = 0.0f;
		unsigned neeFlags = PT_NEE_SURFACE;
		if ( bounce == 0 || g_nPtSoftMode != 0 )
			neeFlags |= PT_NEE_SOFT;
		// Medium vertices: use last scatter dir as shading normal so NEE prefers forward lobe.
		Vector neeN = bInMedium ? medDir : normal;
		Vector nee = PtSampleDirect( pos, neeN,
									 seed + (unsigned)bounce * 31u + (unsigned)sppIndex * 17u,
									 &sunB, skipFace, neeFlags );
		nee = PtClampFirefly( nee, 2500.0f );
		radiance += throughput * nee;
		if ( bounce == 0 )
			sunAmt = sunB;

		// --- Direction sample: BSDF on surface, or continue medium dir ---
		float u1 = PtHash( seed + (unsigned)bounce * 13u + sppIndex * 7u + 11u );
		float u2 = PtHash( seed + (unsigned)bounce * 17u + sppIndex * 11u + 13u );
		Vector dir;
		float pdfBsdf = 0.0f;
		if ( bInMedium )
		{
			dir = medDir;
			pdfBsdf = 1.0f; // already sampled via HG previous event
		}
		else
		{
			dir = PtCosineHemisphere( normal, u1, u2 );
			const float cosTheta = max( 0.0f, DotProduct( normal, dir ) );
			pdfBsdf = cosTheta * invPi;
		}

		float maxDist = (float)MAX_TRACE_LENGTH;
		float tFree = maxDist;
		int medBody = -1;
		Vector sigmaT( 0, 0, 0 );
		bool bTryMedium = false;
		if ( bVolMode && mediumScatters < maxMed && WaterMedium_PointInWater( pos, &medBody ) )
		{
			float uMed = PtHash( seed + (unsigned)bounce * 53u + sppIndex * 29u + 77u );
			if ( WaterMedium_SampleFreeFlight( pos, dir, uMed, &tFree, &sigmaT, &medBody ) )
			{
				bTryMedium = true;
				maxDist = tFree + 1e-3f;
			}
		}

		float curTMin = 0.25f;
		float tHit = -1.0f;
		Vector hitN;
		hitN.Init( 0, 0, 1 );
		bool bHit = false;
		int lastFace = -2;
		float lastFilterT = -1e30f;
		Vector lastFilterN( 0, 0, 0 );
		for ( int fpass = 0; fpass < kPtMaxFilterHits + 1; ++fpass )
		{
			Vector origins[1] = { pos };
			Vector dirs[1] = { dir };
			float tmins[1] = { curTMin };
			float tmaxs[1] = { maxDist };
			float t[1];
			unsigned int flags[1], hit[1];
			Vector nOut[1];
			PathTraceDXR_TraceClosest( origins, dirs, tmins, tmaxs, t, flags, hit, 1, nOut );
			if ( !hit[0] || ( flags[0] & TRACE_ID_SKY ) )
			{
				bHit = false;
				break;
			}
			if ( flags[0] & TRACE_ID_FILTER )
			{
				const int fn = (int)( flags[0] & 0x00FFFFFFu );
				Vector faceN = PtFilterFaceNormal( fn );
				if ( fn == lastFace || PtFilterIsSheetExit( t[0], faceN, lastFilterT, lastFilterN ) )
				{
					curTMin = PtFilterAdvancePast( t[0] );
					continue;
				}
				lastFace = fn;
				if ( VRadFilter_ShouldApply( fn, dir ) )
				{
					Vector hitP = pos + dir * t[0];
					Vector Ft( 1, 1, 1 );
					if ( VRadFilter_SampleAtPos( fn, hitP, Ft ) )
					{
						Ft.x = max( Ft.x, 1e-4f );
						Ft.y = max( Ft.y, 1e-4f );
						Ft.z = max( Ft.z, 1e-4f );
						throughput = Oklab_StackFilterLinearSRGB( throughput, Ft );
					}
					lastFilterT = t[0];
					lastFilterN = faceN;
				}
				curTMin = PtFilterAdvancePast( t[0] );
				continue;
			}
			bHit = true;
			tHit = t[0];
			hitN = nOut[0];
			break;
		}

		// Mode C: free-flight scatter before surface
		if ( bTryMedium && ( !bHit || tHit > tFree ) )
		{
			Vector scatterPos = pos + dir * tFree;
			// Beer-Lambert to scatter point
			throughput *= WaterMedium_SegmentTransmittance( pos, scatterPos );
			Vector albedoM = WaterMedium_ScatterAlbedo( medBody );
			// Soft sky veiling at medium events (single-scatter fill). Without this,
			// HG + absorption eats ambient and pools go black.
			{
				Vector veilAmb( 0, 0, 0 );
				SkyAmbient_ComputeAtPos( scatterPos, veilAmb );
				const WaterIops_t *ioV = WaterMedium_GetIops( medBody );
				Vector fog = ioV ? ioV->fogColor : Vector( 0.05f, 0.12f, 0.18f );
				const float kVeil = 0.12f;
				radiance.x += throughput.x * albedoM.x * veilAmb.x * fog.x * kVeil;
				radiance.y += throughput.y * albedoM.y * veilAmb.y * fog.y * kVeil;
				radiance.z += throughput.z * albedoM.z * veilAmb.z * fog.z * kVeil;
			}
			throughput *= albedoM;
			++mediumScatters;

			float ug1 = PtHash( seed + (unsigned)bounce * 71u + sppIndex * 37u + 3u );
			float ug2 = PtHash( seed + (unsigned)bounce * 73u + sppIndex * 41u + 5u );
			const WaterIops_t *io = WaterMedium_GetIops( medBody );
			float g = io ? io->g : 0.8f;
			Vector wi;
			WaterMedium_SampleHG( g, ug1, ug2, dir, wi );
			pos = scatterPos;
			medDir = wi;
			bInMedium = true;
			normal = wi;
			skipFace = -1;

			if ( throughput.x + throughput.y + throughput.z < 1e-4f )
				break;
			if ( bounce >= 1 )
			{
				float q = min( 0.95f, max( throughput.x, max( throughput.y, throughput.z ) ) );
				float r = PtHash( seed + (unsigned)bounce * 41u + 99u + (unsigned)sppIndex );
				if ( r > q )
					break;
				throughput *= ( 1.0f / max( q, 1e-3f ) );
			}
			continue;
		}

		if ( !bHit )
		{
			// Path exited to sky. Mode C: Beer-Lambert through water (Kd is off in volume mode).
			// Mode A: sky irradiance uses Kd only via PtSampleSkyAmbient (no segment stack).
			if ( bVolMode )
				throughput *= WaterMedium_SegmentTransmittance( pos, pos + dir * (float)MAX_TRACE_LENGTH );
			radiance += PtClampFirefly( throughput * PtSampleSkyAmbient( pos, dir ), 2500.0f );
			break;
		}

		Vector hitPos = pos + dir * tHit;
		Vector hitNormal = hitN;
		if ( hitNormal.LengthSqr() < 1e-8f )
			hitNormal = -dir;
		if ( DotProduct( hitNormal, dir ) > 0.0f )
			hitNormal = -hitNormal;

		// Mode A/C: Beer-Lambert along path segment to the surface
		if ( WaterMedium_IsActive() )
			throughput *= WaterMedium_SegmentTransmittance( pos, hitPos );

		// --- Emissive surface hit (MIS vs NEE; same units as surface NEE) ---
		float pdfLight = 0.0f;
		Vector emitContrib;
		if ( pdfBsdf > 0.0f &&
			 PtSurfaceEmissionContrib( pos, normal, hitPos, hitNormal, skipFace, emitContrib, pdfLight ) )
		{
			const float mis = PtMisWeight( pdfBsdf, pdfLight );
			radiance += PtClampFirefly( throughput * emitContrib * mis, 2500.0f );
			break; // emitters terminate the path
		}

		Vector albedo = PtGetHitAlbedo( hitPos, hitNormal, true, false, (int)bounce );
		albedo *= Absorb_BounceScale( hitPos, pos );
		{
			BounceVolSettings_t bv;
			BounceVol_Resolve( hitPos, bv );
			albedo *= bv.boost;
		}
		albedo.x = min( 1.0f, max( 0.0f, albedo.x ) );
		albedo.y = min( 1.0f, max( 0.0f, albedo.y ) );
		albedo.z = min( 1.0f, max( 0.0f, albedo.z ) );

		throughput *= albedo;

		if ( bounce >= 1 )
		{
			float q = min( 0.95f, max( throughput.x, max( throughput.y, throughput.z ) ) );
			float r = PtHash( seed + (unsigned)bounce * 41u + 99u + (unsigned)sppIndex );
			if ( r > q )
				break;
			throughput *= ( 1.0f / max( q, 1e-3f ) );
		}

		pos = hitPos + hitNormal * 0.5f;
		normal = hitNormal;
		bInMedium = false;
		skipFace = -1;

		if ( throughput.x + throughput.y + throughput.z < 1e-4f )
			break;
	}

	if ( pSunAmt )
		*pSunAmt = sunAmt;
	return radiance;
}

// Kept name for call sites: returns full path Li (direct+indirect). outIndirect unused (zero).
// outVarMean: optional Var(E[luminance]) for Sakai Welch denoise (= sampleVar / nTaken).
// aaOff / nAa: optional luxel-footprint offsets (world units). Path sample s uses
// aaOff[s % nAa] so spatial AA costs nothing extra vs center-only sampling.
static void PtBakeAtPosition( const Vector &pos, const Vector &normal, float luxelWorld, int facenum,
							  int ni, Vector &outDirect, float &outSun, Vector &outIndirect,
							  float *outVarMean, const Vector *aaOff = nullptr, int nAa = 0 )
{
	const unsigned int seed = PtWorldSeed( pos - normal * 1.0f, luxelWorld, ni );

	const int sppBase = g_ptSpp;
	int sppMax = sppBase * 4;
	if ( do_fast )
		sppMax = sppBase * 2;
	if ( sppMax < sppBase )
		sppMax = sppBase;
	if ( sppMax > 64 )
		sppMax = 64;

	Vector sum( 0, 0, 0 );
	float sunSum = 0.0f;
	float sumL = 0.0f, sumL2 = 0.0f;
	int nTaken = 0;
	int nNearZero = 0;

	for ( int s = 0; ; ++s )
	{
		Vector samplePos = pos;
		if ( aaOff && nAa > 0 )
			samplePos = pos + aaOff[s % nAa];

		float sunAmt = 0.0f;
		Vector li = PtPathLi( samplePos, normal, seed, s, g_ptBounces, facenum, &sunAmt );
		li = PtClampFirefly( li, 2500.0f );
		sum += li;
		sunSum += sunAmt;
		const float L = PtLuma( li );
		sumL += L;
		sumL2 += L * L;
		++nTaken;
		if ( L < 1e-5f )
			++nNearZero;

		if ( nTaken < sppBase )
			continue;

		const float mean = sumL / (float)nTaken;
		const float var = max( 0.0f, sumL2 / (float)nTaken - mean * mean );
		const float cv = sqrtf( var ) / max( mean, 1e-5f );

		bool needMore = false;
		if ( nTaken < sppMax )
		{
			if ( cv > 0.25f )
				needMore = true;
			if ( nNearZero * 3 >= nTaken )
				needMore = true;
			if ( mean < 5e-4f )
				needMore = true;
		}
		if ( !needMore || nTaken >= sppMax )
			break;
	}

	outDirect = sum * ( 1.0f / (float)nTaken );
	outSun = sunSum * ( 1.0f / (float)nTaken );
	outIndirect.Init(); // full path already in outDirect
	if ( outVarMean )
	{
		const float mean = sumL / (float)nTaken;
		const float sampleVar = max( 0.0f, sumL2 / (float)nTaken - mean * mean );
		// Variance of the mean estimator (Sakai / Welch input).
		*outVarMean = sampleVar / (float)max( nTaken, 1 );
	}
}

// ---------------------------------------------------------------------------
// Face bake + shell progress
// ---------------------------------------------------------------------------
static std::atomic<int> g_ptFaceProgress{ 0 };
static std::atomic<int> g_ptFacesBaked{ 0 };
static std::atomic<int> g_ptLuxelsBaked{ 0 };
static std::atomic<int> g_ptLuxelsInFlight{ 0 };	// luxels finished (running total across all faces)
static std::atomic<long long> g_ptLastReportMs{ 0 };
static int g_ptFaceTotal = 0;
static int g_ptLuxelTotalEst = 0;		// pre-pass estimate of bakeable luxels
static int g_ptBakeableFacesEst = 0;		// faces that contribute luxels
static double g_ptBakeStartTime = 0.0;

// ETA smoother state (single-threaded write from progress reporter)
static double g_ptEmaLuxelsPerSec = 0.0;
static double g_ptSmoothedEtaSec = -1.0;
static int g_ptEtaLastLuxels = 0;
static double g_ptEtaLastTime = 0.0;
static double g_ptEtaLastDecayTime = 0.0;

// Soften luxel stair-steps where neighbor brightness jumps hard (shadow edges).
// Relative contrast drives blend - soft gradients stay put, binary jags AA.
static void PtAntialiasHighContrast( Vector *grid, const float *wgt, int width, int height )
{
	if ( !grid || !wgt || width < 2 || height < 2 )
		return;

	const int n = width * height;
	std::vector<Vector> out( (size_t)n );
	std::vector<float> luma( (size_t)n );

	for ( int pass = 0; pass < 2; ++pass )
	{
		for ( int i = 0; i < n; ++i )
			luma[i] = ( wgt[i] > 0.0f ) ? PtLuma( grid[i] ) : 0.0f;

		for ( int y = 0; y < height; ++y )
		{
			for ( int x = 0; x < width; ++x )
			{
				const int ci = y * width + x;
				out[ci] = grid[ci];
				if ( wgt[ci] <= 0.0f )
					continue;

				const float lc = luma[ci];
				float maxRel = 0.0f;
				Vector sum( 0, 0, 0 );
				float wsum = 0.0f;

				// 3x3 gather; stronger weight on axis neighbors (FXAA-ish).
				for ( int dy = -1; dy <= 1; ++dy )
				{
					int ny = y + dy;
					if ( ny < 0 || ny >= height )
						continue;
					for ( int dx = -1; dx <= 1; ++dx )
					{
						int nx = x + dx;
						if ( nx < 0 || nx >= width )
							continue;
						const int ni = ny * width + nx;
						if ( wgt[ni] <= 0.0f )
							continue;
						const float ln = luma[ni];
						const float denom = max( max( lc, ln ), 1e-4f );
						const float rel = fabsf( lc - ln ) / denom;
						if ( rel > maxRel )
							maxRel = rel;
						const float spat = ( dx == 0 && dy == 0 ) ? 1.0f
							: ( ( dx == 0 || dy == 0 ) ? 0.85f : 0.45f );
						sum += grid[ni] * spat;
						wsum += spat;
					}
				}

				// Start AA around ~25% relative jump; full by ~55%.
				float aa = ( maxRel - 0.25f ) / 0.30f;
				if ( aa < 0.0f )
					aa = 0.0f;
				if ( aa > 1.0f )
					aa = 1.0f;
				aa *= aa; // bias toward keeping mild edges
				if ( aa < 1e-4f || wsum < 1e-6f )
					continue;

				const Vector avg = sum * ( 1.0f / wsum );
				out[ci] = grid[ci] + ( avg - grid[ci] ) * ( aa * 0.65f );
			}
		}

		for ( int i = 0; i < n; ++i )
			grid[i] = out[i];
	}
}

// Path-trace luxel denoise. varSamples: optional [ni*numsamples+i] Var(mean) for Sakai.
static void PtDenoiseFaceSamples( facelight_t *fl, dface_t *f, int normalCount, bool bIndirect,
								  const float *varSamples, bool bNetworkDenoise = true )
{
	(void)bIndirect;
	if ( !fl || !f || fl->numsamples <= 0 )
		return;

	const bool bDenoise = g_bPathTraceDenoise && PathTraceDenoise_IsReady();
	// Always heal dark edge-bleed luxels on pathtraced faces (even if denoise is off).

	int width = f->m_LightmapTextureSizeInLuxels[0] + 1;
	int height = f->m_LightmapTextureSizeInLuxels[1] + 1;
	if ( width < 1 || height < 1 )
		return;

	if ( width * height < 4 )
		return;

	const int n = width * height;
	const float blend = min( 1.0f, max( 0.0f, g_flPathTraceDenoiseStrength ) );

	std::vector<Vector> grid( n );
	std::vector<float> wgt( n, 0.0f );
	std::vector<float> rgb( (size_t)n * 3u );
	std::vector<float> varGrid( n, 0.0f );
	std::vector<float> varWgt( n, 0.0f );

	for ( int ni = 0; ni < normalCount; ++ni )
	{
		if ( !fl->light[0][ni] )
			continue;

		for ( int i = 0; i < n; ++i )
		{
			grid[i].Init();
			wgt[i] = 0.0f;
			varGrid[i] = 0.0f;
			varWgt[i] = 0.0f;
		}

		for ( int i = 0; i < fl->numsamples; ++i )
		{
			int s = fl->sample[i].s;
			int t = fl->sample[i].t;
			if ( ( s < 0 || t < 0 || s >= width || t >= height ) && fl->numsamples == n )
			{
				s = i % width;
				t = i / width;
			}
			if ( s < 0 || t < 0 || s >= width || t >= height )
				continue;
			const int idx = s + t * width;
			grid[idx] += fl->light[0][ni][i].m_vecLighting;
			wgt[idx] += 1.0f;
			if ( varSamples )
			{
				varGrid[idx] += varSamples[ni * fl->numsamples + i];
				varWgt[idx] += 1.0f;
			}
		}
		for ( int i = 0; i < n; ++i )
		{
			if ( wgt[i] > 0.0f )
				grid[i] *= ( 1.0f / wgt[i] );
			if ( varWgt[i] > 0.0f )
				varGrid[i] *= ( 1.0f / varWgt[i] );
		}

		// Fill holes so OIDN sees a dense chart.
		std::vector<Vector> tmp = grid;
		std::vector<float> wtmp = wgt;
		for ( int pass = 0; pass < 4; ++pass )
		{
			bool any = false;
			tmp = grid;
			wtmp = wgt;
			for ( int t = 0; t < height; ++t )
			{
				for ( int s = 0; s < width; ++s )
				{
					const int idx = s + t * width;
					if ( wgt[idx] > 0.0f )
						continue;
					Vector sum( 0, 0, 0 );
					float sw = 0.0f;
					float vsum = 0.0f;
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
							int nidx = ns + nt * width;
							if ( wgt[nidx] <= 0.0f )
								continue;
							sum += grid[nidx];
							vsum += varGrid[nidx];
							sw += 1.0f;
						}
					}
					if ( sw > 0.0f )
					{
						tmp[idx] = sum * ( 1.0f / sw );
						varGrid[idx] = vsum / sw;
						wtmp[idx] = 0.001f;
						any = true;
					}
				}
			}
			grid.swap( tmp );
			wgt.swap( wtmp );
			if ( !any )
				break;
		}

		// Dilate "mostly hidden" / edge-bleed luxels: dark outliers relative to
		// neighbors get replaced so they cannot smear into denoise or bilinear filter.
		{
			std::vector<unsigned char> bad( n, 0 );
			std::vector<float> neighL;
			neighL.reserve( 8 );
			for ( int t = 0; t < height; ++t )
			{
				for ( int s = 0; s < width; ++s )
				{
					const int idx = s + t * width;
					if ( wgt[idx] <= 0.0f )
						continue;
					neighL.clear();
					float maxN = 0.0f;
					for ( int dt = -1; dt <= 1; ++dt )
					{
						int nt = t + dt;
						if ( nt < 0 || nt >= height )
							continue;
						for ( int ds = -1; ds <= 1; ++ds )
						{
							if ( !ds && !dt )
								continue;
							int ns = s + ds;
							if ( ns < 0 || ns >= width )
								continue;
							int nidx = ns + nt * width;
							if ( wgt[nidx] <= 0.0f )
								continue;
							float nl = PtLuma( grid[nidx] );
							neighL.push_back( nl );
							maxN = max( maxN, nl );
						}
					}
					if ( neighL.size() < 3 )
						continue;
					// median
					std::sort( neighL.begin(), neighL.end() );
					float med = neighL[neighL.size() / 2];
					float L = PtLuma( grid[idx] );
					// Dark spike in a brighter neighborhood -> likely sample-in-solid / backface.
					if ( maxN > 1e-3f && L < maxN * 0.28f && L < med * 0.55f )
						bad[idx] = 1;
				}
			}
			for ( int pass = 0; pass < 6; ++pass )
			{
				bool any = false;
				tmp = grid;
				for ( int t = 0; t < height; ++t )
				{
					for ( int s = 0; s < width; ++s )
					{
						const int idx = s + t * width;
						if ( !bad[idx] || wgt[idx] <= 0.0f )
							continue;
						Vector sum( 0, 0, 0 );
						float sw = 0.0f;
						float vsum = 0.0f;
						for ( int dt = -1; dt <= 1; ++dt )
						{
							int nt = t + dt;
							if ( nt < 0 || nt >= height )
								continue;
							for ( int ds = -1; ds <= 1; ++ds )
							{
								if ( !ds && !dt )
									continue;
								int ns = s + ds;
								if ( ns < 0 || ns >= width )
									continue;
								int nidx = ns + nt * width;
								if ( wgt[nidx] <= 0.0f || bad[nidx] )
									continue;
								sum += grid[nidx];
								vsum += varGrid[nidx];
								sw += 1.0f;
							}
						}
						if ( sw > 0.0f )
						{
							tmp[idx] = sum * ( 1.0f / sw );
							varGrid[idx] = vsum / sw;
							bad[idx] = 0; // healed
							any = true;
						}
					}
				}
				grid.swap( tmp );
				if ( !any )
					break;
			}
		}

		for ( int i = 0; i < n; ++i )
		{
			rgb[(size_t)i * 3u + 0] = grid[i].x;
			rgb[(size_t)i * 3u + 1] = grid[i].y;
			rgb[(size_t)i * 3u + 2] = grid[i].z;
		}

		const float *varPtr = varSamples ? varGrid.data() : nullptr;
		if ( bNetworkDenoise && bDenoise && width * height >= 32 )
		{
			if ( PathTraceDenoise_DenoiseRGB( rgb.data(), width, height, varPtr ) )
			{
				for ( int i = 0; i < n; ++i )
				{
					grid[i].x = rgb[(size_t)i * 3u + 0];
					grid[i].y = rgb[(size_t)i * 3u + 1];
					grid[i].z = rgb[(size_t)i * 3u + 2];
				}
			}
		}

		// Soften remaining luxel stair-steps on hard shadow edges.
		if ( bNetworkDenoise )
			PtAntialiasHighContrast( grid.data(), wgt.data(), width, height );

		for ( int i = 0; i < fl->numsamples; ++i )
		{
			int s = fl->sample[i].s;
			int t = fl->sample[i].t;
			if ( ( s < 0 || t < 0 || s >= width || t >= height ) && fl->numsamples == n )
			{
				s = i % width;
				t = i / width;
			}
			if ( s < 0 || t < 0 || s >= width || t >= height )
				continue;
			const int idx = s + t * width;
			if ( wgt[idx] <= 0.0f )
				continue;
			const Vector &den = grid[idx];
			Vector &dst = fl->light[0][ni][i].m_vecLighting;
			if ( !bDenoise || blend >= 0.999f )
				dst = den;
			else
				dst = dst + ( den - dst ) * blend;
		}
	}
}

static bool PtFaceIsBakeable( int facenum );

static int PtDenoiseUfFind( std::vector<int> &p, int i )
{
	while ( p[i] != i )
	{
		p[i] = p[p[i]];
		i = p[i];
	}
	return i;
}

static void PtWorldToSharedLuxel( const lightinfo_t &l, const Vector &world, float &s, float &t )
{
	Vector pos;
	VectorSubtract( world, l.luxelOrigin, pos );
	s = DotProduct( pos, l.worldToLuxelSpace[0] );
	t = DotProduct( pos, l.worldToLuxelSpace[1] );
}

static bool PtDenoiseSameLuxelFrame( int a, int b )
{
	const dface_t *fa = &g_pFaces[a];
	const dface_t *fb = &g_pFaces[b];
	const texinfo_t *ta = &texinfo[fa->texinfo];
	const texinfo_t *tb = &texinfo[fb->texinfo];
	for ( int i = 0; i < 2; ++i )
	{
		Vector va( ta->lightmapVecsLuxelsPerWorldUnits[i][0],
				   ta->lightmapVecsLuxelsPerWorldUnits[i][1],
				   ta->lightmapVecsLuxelsPerWorldUnits[i][2] );
		Vector vb( tb->lightmapVecsLuxelsPerWorldUnits[i][0],
				   tb->lightmapVecsLuxelsPerWorldUnits[i][1],
				   tb->lightmapVecsLuxelsPerWorldUnits[i][2] );
		const float na = va.Length();
		const float nb = vb.Length();
		if ( na < 1e-8f || nb < 1e-8f )
			return false;
		if ( fabsf( na - nb ) / max( na, nb ) > 0.02f )
			return false;
		if ( DotProduct( va, vb ) / ( na * nb ) < 0.999f )
			return false;
	}
	if ( ( face_offset[a] - face_offset[b] ).LengthSqr() > 0.01f )
		return false;
	const bool bumpA = ( ta->flags & SURF_BUMPLIGHT ) != 0;
	const bool bumpB = ( tb->flags & SURF_BUMPLIGHT ) != 0;
	return bumpA == bumpB;
}

static bool PtDenoiseCoplanarNeighbors( int a, int b )
{
	if ( DotProduct( faceneighbor[a].facenormal, faceneighbor[b].facenormal ) < 0.999f )
		return false;
	const dplane_t &pa = dplanes[g_pFaces[a].planenum];
	const dface_t *fb = &g_pFaces[b];
	if ( fb->numedges <= 0 )
		return false;
	const int e = dsurfedges[fb->firstedge];
	const int v = ( e >= 0 ) ? dedges[e].v[0] : dedges[-e].v[1];
	const float dist = fabsf( DotProduct( pa.normal, dvertexes[v].point ) - pa.dist );
	return dist < 0.75f;
}

static bool PtDenoiseIslandEligible( int facenum )
{
	if ( facenum < 0 || facenum >= numfaces )
		return false;
	if ( !PtFaceIsBakeable( facenum ) )
		return false;
	const dface_t *f = &g_pFaces[facenum];
	if ( f->dispinfo != -1 )
		return false;
	const facelight_t *fl = &facelight[facenum];
	return fl->numsamples > 0 && fl->sample && fl->light[0][0];
}

static void PtFormatDuration( int totalSec, char *buf, int bufSize );
static void PtProgressLine( bool bFinish, const char *text );

// Pack coplanar connected faces into one luxel image and denoise it (OIDN/OptiX).
static void PtDenoiseCoplanarIslands()
{
	if ( !g_bPathTraceDenoise || !PathTraceDenoise_IsReady() )
		return;
	if ( g_PathTraceDenoiser == PT_DENOISER_SAKAI )
		return;

	std::vector<int> parent( (size_t)numfaces );
	for ( int i = 0; i < numfaces; ++i )
		parent[i] = i;

	int nEligible = 0;
	for ( int facenum = 0; facenum < numfaces; ++facenum )
	{
		if ( !PtDenoiseIslandEligible( facenum ) )
			continue;
		++nEligible;
		const faceneighbor_t *fn = &faceneighbor[facenum];
		for ( int j = 0; j < fn->numneighbors; ++j )
		{
			const int nb = fn->neighbor[j];
			if ( nb <= facenum || !PtDenoiseIslandEligible( nb ) )
				continue;
			if ( !PtDenoiseCoplanarNeighbors( facenum, nb ) )
				continue;
			if ( !PtDenoiseSameLuxelFrame( facenum, nb ) )
				continue;
			const int ra = PtDenoiseUfFind( parent, facenum );
			const int rb = PtDenoiseUfFind( parent, nb );
			if ( ra != rb )
				parent[ra] = rb;
		}
	}

	std::vector<std::vector<int>> islands( (size_t)numfaces );
	for ( int facenum = 0; facenum < numfaces; ++facenum )
	{
		if ( !PtDenoiseIslandEligible( facenum ) )
			continue;
		islands[PtDenoiseUfFind( parent, facenum )].push_back( facenum );
	}

	int nIslandTotal = 0;
	int nFaceTotal = 0;
	for ( int root = 0; root < numfaces; ++root )
	{
		if ( islands[root].empty() )
			continue;
		++nIslandTotal;
		nFaceTotal += (int)islands[root].size();
	}
	if ( nIslandTotal <= 0 )
		return;

	Msg( "[PathTrace-DXR] Denoising %d coplanar island(s) (%d faces, %s)...\n",
		 nIslandTotal, nFaceTotal, PathTraceDenoise_Name() );
	fflush( stdout );

	const float blend = min( 1.0f, max( 0.0f, g_flPathTraceDenoiseStrength ) );
	const int kMaxSide = 2048;
	int nIslands = 0, nGroupedFaces = 0, nFallback = 0;
	const double t0 = Plat_FloatTime();
	double lastReport = 0.0;

	auto reportDenoise = [&]( bool bFinish )
	{
		char elapsedStr[32], etaStr[48];
		const double elapsed = Plat_FloatTime() - t0;
		PtFormatDuration( (int)elapsed, elapsedStr, sizeof( elapsedStr ) );
		const float pct = ( nIslandTotal > 0 )
			? ( 100.0f * (float)nIslands / (float)nIslandTotal ) : 100.0f;
		etaStr[0] = '\0';
		if ( !bFinish && nIslands > 0 && nIslands < nIslandTotal && elapsed > 0.4 )
		{
			const double rate = (double)nIslands / elapsed;
			const double eta = (double)( nIslandTotal - nIslands ) / max( rate, 1e-6 );
			char etaBuf[32];
			PtFormatDuration( (int)( eta + 0.5 ), etaBuf, sizeof( etaBuf ) );
			Q_snprintf( etaStr, sizeof( etaStr ), "  ETA %s", etaBuf );
		}
		char line[256];
		Q_snprintf( line, sizeof( line ),
					"[PathTrace-DXR] Denoise  %5.1f%%  %d / %d islands  (%d faces)  %s elapsed%s",
					pct, nIslands, nIslandTotal, nGroupedFaces, elapsedStr, etaStr );
		PtProgressLine( bFinish, line );
	};

	reportDenoise( false );

	for ( int root = 0; root < numfaces; ++root )
	{
		std::vector<int> &members = islands[root];
		if ( members.empty() )
			continue;
		++nIslands;
		nGroupedFaces += (int)members.size();

		int bumpCount = 1;
		{
			const dface_t *f0 = &g_pFaces[members[0]];
			if ( texinfo[f0->texinfo].flags & SURF_BUMPLIGHT )
				bumpCount = NUM_BUMP_VECTS + 1;
		}

		lightinfo_t lRoot;
		InitLightinfo( &lRoot, members[0] );

		float minS = 1e30f, minT = 1e30f, maxS = -1e30f, maxT = -1e30f;
		for ( int facenum : members )
		{
			const facelight_t *fl = &facelight[facenum];
			for ( int i = 0; i < fl->numsamples; ++i )
			{
				float s, t;
				PtWorldToSharedLuxel( lRoot, fl->sample[i].pos, s, t );
				minS = min( minS, s );
				minT = min( minT, t );
				maxS = max( maxS, s );
				maxT = max( maxT, t );
			}
		}

		int width = (int)ceilf( maxS - minS ) + 3;
		int height = (int)ceilf( maxT - minT ) + 3;
		if ( width < 2 )
			width = 2;
		if ( height < 2 )
			height = 2;

		if ( width > kMaxSide || height > kMaxSide )
		{
			for ( int facenum : members )
			{
				dface_t *f = &g_pFaces[facenum];
				facelight_t *fl = &facelight[facenum];
				int nc = 1;
				if ( texinfo[f->texinfo].flags & SURF_BUMPLIGHT )
					nc = NUM_BUMP_VECTS + 1;
				PtDenoiseFaceSamples( fl, f, nc, false, nullptr, true );
			}
			++nFallback;
			continue;
		}

		const int nPix = width * height;
		std::vector<Vector> grid( (size_t)nPix );
		std::vector<float> wgt( (size_t)nPix, 0.0f );
		std::vector<float> rgb( (size_t)nPix * 3u );

		for ( int ni = 0; ni < bumpCount; ++ni )
		{
			for ( int i = 0; i < nPix; ++i )
			{
				grid[i].Init();
				wgt[i] = 0.0f;
			}

			for ( int facenum : members )
			{
				const facelight_t *fl = &facelight[facenum];
				if ( !fl->light[0][ni] )
					continue;
				for ( int i = 0; i < fl->numsamples; ++i )
				{
					float fs, ft;
					PtWorldToSharedLuxel( lRoot, fl->sample[i].pos, fs, ft );
					const int s = clamp( (int)floorf( fs - minS + 1.0f + 0.5f ), 0, width - 1 );
					const int t = clamp( (int)floorf( ft - minT + 1.0f + 0.5f ), 0, height - 1 );
					const int idx = s + t * width;
					grid[idx] += fl->light[0][ni][i].m_vecLighting;
					wgt[idx] += 1.0f;
				}
			}
			for ( int i = 0; i < nPix; ++i )
			{
				if ( wgt[i] > 0.0f )
					grid[i] *= ( 1.0f / wgt[i] );
			}

			std::vector<Vector> tmp = grid;
			std::vector<float> wtmp = wgt;
			for ( int pass = 0; pass < 6; ++pass )
			{
				bool any = false;
				tmp = grid;
				wtmp = wgt;
				for ( int t = 0; t < height; ++t )
				{
					for ( int s = 0; s < width; ++s )
					{
						const int idx = s + t * width;
						if ( wgt[idx] > 0.0f )
							continue;
						Vector sum( 0, 0, 0 );
						float sw = 0.0f;
						for ( int dt = -1; dt <= 1; ++dt )
						{
							const int nt = t + dt;
							if ( nt < 0 || nt >= height )
								continue;
							for ( int ds = -1; ds <= 1; ++ds )
							{
								const int ns = s + ds;
								if ( ns < 0 || ns >= width )
									continue;
								const int nidx = ns + nt * width;
								if ( wgt[nidx] <= 0.0f )
									continue;
								sum += grid[nidx];
								sw += 1.0f;
							}
						}
						if ( sw > 0.0f )
						{
							tmp[idx] = sum * ( 1.0f / sw );
							wtmp[idx] = 0.001f;
							any = true;
						}
					}
				}
				grid.swap( tmp );
				wgt.swap( wtmp );
				if ( !any )
					break;
			}

			for ( int i = 0; i < nPix; ++i )
			{
				rgb[(size_t)i * 3u + 0] = grid[i].x;
				rgb[(size_t)i * 3u + 1] = grid[i].y;
				rgb[(size_t)i * 3u + 2] = grid[i].z;
			}

			if ( nPix >= 32 )
				PathTraceDenoise_DenoiseRGB( rgb.data(), width, height, nullptr );

			for ( int i = 0; i < nPix; ++i )
			{
				grid[i].x = rgb[(size_t)i * 3u + 0];
				grid[i].y = rgb[(size_t)i * 3u + 1];
				grid[i].z = rgb[(size_t)i * 3u + 2];
			}
			PtAntialiasHighContrast( grid.data(), wgt.data(), width, height );

			auto sampleBilinear = [&]( float fs, float ft ) -> Vector
			{
				const float x = clamp( fs - minS + 1.0f, 0.0f, (float)( width - 1 ) );
				const float y = clamp( ft - minT + 1.0f, 0.0f, (float)( height - 1 ) );
				const int s0 = (int)floorf( x );
				const int t0 = (int)floorf( y );
				const int s1 = min( s0 + 1, width - 1 );
				const int t1 = min( t0 + 1, height - 1 );
				const float fx = x - (float)s0;
				const float fy = y - (float)t0;
				const Vector &c00 = grid[s0 + t0 * width];
				const Vector &c10 = grid[s1 + t0 * width];
				const Vector &c01 = grid[s0 + t1 * width];
				const Vector &c11 = grid[s1 + t1 * width];
				return c00 * ( ( 1.0f - fx ) * ( 1.0f - fy ) ) + c10 * ( fx * ( 1.0f - fy ) ) +
					   c01 * ( ( 1.0f - fx ) * fy ) + c11 * ( fx * fy );
			};

			for ( int facenum : members )
			{
				facelight_t *fl = &facelight[facenum];
				if ( !fl->light[0][ni] )
					continue;
				for ( int i = 0; i < fl->numsamples; ++i )
				{
					float fs, ft;
					PtWorldToSharedLuxel( lRoot, fl->sample[i].pos, fs, ft );
					const Vector den = sampleBilinear( fs, ft );
					Vector &dst = fl->light[0][ni][i].m_vecLighting;
					if ( blend >= 0.999f )
						dst = den;
					else
						dst = dst + ( den - dst ) * blend;
				}
			}
		}

		const double now = Plat_FloatTime();
		if ( nIslands == nIslandTotal || ( now - lastReport ) >= 0.25 )
		{
			reportDenoise( false );
			lastReport = now;
		}
	}

	reportDenoise( true );
	Msg( "[PathTrace-DXR] Coplanar denoise: %d island(s), %d faces%s\n",
		 nIslands, nGroupedFaces,
		 nFallback ? " (some islands too large; per-face fallback)" : "" );
	(void)nEligible;
}

static void PtFormatDuration( int totalSec, char *buf, int bufSize )
{
	if ( totalSec < 0 )
		totalSec = 0;
	const int h = totalSec / 3600;
	const int m = ( totalSec / 60 ) % 60;
	const int s = totalSec % 60;
	if ( h > 0 )
		Q_snprintf( buf, bufSize, "%d:%02d:%02d", h, m, s );
	else
		Q_snprintf( buf, bufSize, "%d:%02d", m, s );
}

// In-place progress line (\r overwrite). Finish with bFinish=true to emit newline.
static int g_ptProgressLineLen = 0;

static void PtEndProgressLine()
{
	if ( g_ptProgressLineLen <= 0 )
		return;
	Msg( "\n" );
	fflush( stdout );
	g_ptProgressLineLen = 0;
}

static void PtProgressLine( bool bFinish, const char *text )
{
	char line[700];
	const int len = (int)V_strlen( text );
	int padTo = ( g_ptProgressLineLen > len ) ? g_ptProgressLineLen : len;
	if ( padTo > (int)sizeof( line ) - 8 )
		padTo = (int)sizeof( line ) - 8;

	int n = 0;
	line[n++] = '\r';
	for ( int i = 0; i < len && n < (int)sizeof( line ) - 2; ++i )
		line[n++] = text[i];
	while ( ( n - 1 ) < padTo && n < (int)sizeof( line ) - 2 )
		line[n++] = ' ';
	line[n] = '\0';
	Msg( "%s", line );
	if ( bFinish )
	{
		Msg( "\n" );
		g_ptProgressLineLen = 0;
	}
	else
	{
		g_ptProgressLineLen = len;
	}
	fflush( stdout );
}

// Cheap pre-pass: luxels we expect to bake (lightmap grid; skips invisible / special / no-patch).
static int PtEstimateFaceLuxels( int facenum )
{
	if ( facenum < 0 || facenum >= numfaces )
		return 0;
	if ( !( g_FacesVisibleToLights[facenum >> 3] & ( 1 << ( facenum & 7 ) ) ) )
		return 0;
	dface_t *f = &g_pFaces[facenum];
	if ( texinfo[f->texinfo].flags & TEX_SPECIAL )
		return 0;
	if ( g_FacePatches.Element( facenum ) == g_FacePatches.InvalidIndex() )
		return 0;

	int w = f->m_LightmapTextureSizeInLuxels[0] + 1;
	int h = f->m_LightmapTextureSizeInLuxels[1] + 1;
	if ( w < 1 ) w = 1;
	if ( h < 1 ) h = 1;
	return w * h;
}

static int PtEstimateTotalLuxels()
{
	int total = 0;
	g_ptBakeableFacesEst = 0;
	int maxDim = 0;
	int sumMaxDim = 0;
	for ( int i = 0; i < numfaces; ++i )
	{
		const int n = PtEstimateFaceLuxels( i );
		if ( n > 0 )
		{
			total += n;
			++g_ptBakeableFacesEst;
			dface_t *f = &g_pFaces[i];
			const int w = f->m_LightmapTextureSizeInLuxels[0] + 1;
			const int h = f->m_LightmapTextureSizeInLuxels[1] + 1;
			const int dim = ( w > h ) ? w : h;
			if ( dim > maxDim )
				maxDim = dim;
			sumMaxDim += dim;
		}
	}
	if ( g_ptBakeableFacesEst > 0 )
	{
		Msg( "[PathTrace-DXR] Lightmap dims on lit faces: avg max-side %.1f, largest %d luxels (Hammer scale 2 ~= many faces near 16-35).\n",
			 (double)sumMaxDim / (double)g_ptBakeableFacesEst, maxDim );
	}
	return total;
}

// Rate-limited in-place console progress (~1 Hz, \r overwrite).
// ETA is luxel-work based with EMA rate + smoothed remaining time (faces vary wildly in cost).
static void PtReportProgress( bool bForce )
{
	const int done = g_ptFaceProgress.load( std::memory_order_relaxed );
	const double now = Plat_FloatTime();
	const long long nowMs = (long long)( now * 1000.0 );

	if ( !bForce )
	{
		long long last = g_ptLastReportMs.load( std::memory_order_relaxed );
		if ( nowMs - last < 1000 )
			return;
		if ( !g_ptLastReportMs.compare_exchange_strong( last, nowMs, std::memory_order_relaxed ) )
			return;
		if ( nowMs - last < 1000 )
			return;
	}
	else
	{
		g_ptLastReportMs.store( nowMs, std::memory_order_relaxed );
	}

	const double elapsed = now - g_ptBakeStartTime;
	const int baked = g_ptFacesBaked.load( std::memory_order_relaxed );
	const int luxelsDone = g_ptLuxelsInFlight.load( std::memory_order_relaxed );

	// Keep estimate honest if actual luxels exceed the lightmap-size guess.
	int luxelTotal = g_ptLuxelTotalEst;
	if ( luxelsDone > luxelTotal )
		luxelTotal = luxelsDone;
	if ( luxelTotal < 1 )
		luxelTotal = 1;

	// Prefer luxel work for % (empty faces finish instantly and skew face %).
	float pct = ( luxelsDone > 0 || done >= g_ptFaceTotal )
		? ( 100.0f * (float)luxelsDone / (float)luxelTotal )
		: ( g_ptFaceTotal > 0 ? ( 100.0f * (float)done / (float)g_ptFaceTotal ) : 0.0f );

	char elapsedStr[32];
	PtFormatDuration( (int)elapsed, elapsedStr, sizeof( elapsedStr ) );

	const bool bBaking = ( done < g_ptFaceTotal ) && ( luxelsDone < luxelTotal || done + 8 < g_ptFaceTotal );
	bool bShowEta = false;
	int etaSec = 0;

	if ( bBaking && luxelsDone > 0 && elapsed > 2.0 )
	{
		// Instantaneous rate over the last progress tick, EMA-smoothed.
		const double dt = now - g_ptEtaLastTime;
		const int dLux = luxelsDone - g_ptEtaLastLuxels;
		if ( dt >= 0.75 && dLux > 0 )
		{
			const double instRate = (double)dLux / dt;
			if ( g_ptEmaLuxelsPerSec <= 1e-6 )
				g_ptEmaLuxelsPerSec = instRate;
			else
				g_ptEmaLuxelsPerSec = 0.28 * instRate + 0.72 * g_ptEmaLuxelsPerSec;
			g_ptEtaLastLuxels = luxelsDone;
			g_ptEtaLastTime = now;
		}
		else if ( g_ptEtaLastTime <= 0.0 )
		{
			g_ptEtaLastLuxels = luxelsDone;
			g_ptEtaLastTime = now;
		}

		const double avgRate = (double)luxelsDone / elapsed;
		double rate = avgRate;
		if ( g_ptEmaLuxelsPerSec > 1e-6 )
		{
			// Early on lean on average; later trust the recent EMA more.
			const double emaW = ( elapsed < 15.0 ) ? 0.35 : ( elapsed < 60.0 ? 0.55 : 0.70 );
			rate = emaW * g_ptEmaLuxelsPerSec + ( 1.0 - emaW ) * avgRate;
		}

		if ( rate > 1e-6 )
		{
			const int remaining = luxelTotal - luxelsDone;
			double rawEta = ( remaining > 0 ) ? ( (double)remaining / rate ) : 0.0;

			// Face-based sanity check: if many faces remain but few luxels, blend in face ETA
			// so we don't report ~0 while skip-faces are still being walked.
			if ( done > 0 && done < g_ptFaceTotal )
			{
				const double faceRate = (double)done / elapsed;
				const double faceEta = (double)( g_ptFaceTotal - done ) / faceRate;
				if ( remaining <= 0 )
					rawEta = faceEta;
				else if ( faceEta > rawEta * 1.25 )
					rawEta = 0.85 * rawEta + 0.15 * faceEta;
			}

			if ( g_ptSmoothedEtaSec < 0.0 )
			{
				g_ptSmoothedEtaSec = rawEta;
			}
			else
			{
				// Drop ETA quickly when speed improves; rise slowly when it worsens (less flicker).
				if ( rawEta < g_ptSmoothedEtaSec )
					g_ptSmoothedEtaSec = 0.45 * rawEta + 0.55 * g_ptSmoothedEtaSec;
				else
					g_ptSmoothedEtaSec = 0.18 * rawEta + 0.82 * g_ptSmoothedEtaSec;
			}

			// Also decay smoothed ETA by wall-clock between reports so it counts down naturally.
			if ( g_ptEtaLastDecayTime > 0.0 )
			{
				const double decay = now - g_ptEtaLastDecayTime;
				if ( decay > 0.0 && decay < 5.0 )
					g_ptSmoothedEtaSec = max( 0.0, g_ptSmoothedEtaSec - decay );
			}
			g_ptEtaLastDecayTime = now;

			etaSec = (int)( g_ptSmoothedEtaSec + 0.5 );
			bShowEta = true;
		}
	}

	const bool doneAll = ( done >= g_ptFaceTotal ) || ( luxelsDone >= luxelTotal && g_ptFaceTotal > 0 );
	const bool finish = bForce || ( !bBaking && doneAll );
	// Snap to 100% only when work is actually complete.
	if ( finish && doneAll && pct < 100.0f )
		pct = 100.0f;

	char line[512];
	if ( bShowEta && !finish )
	{
		char etaStr[32];
		PtFormatDuration( etaSec, etaStr, sizeof( etaStr ) );
		Q_snprintf( line, sizeof( line ),
					"[PathTrace-DXR] %5.1f%%  %d / %d faces  (%d lit, %d / %d luxels)  %s elapsed  ~%s left",
					pct, done, g_ptFaceTotal, baked, luxelsDone, luxelTotal, elapsedStr, etaStr );
	}
	else
	{
		Q_snprintf( line, sizeof( line ),
					"[PathTrace-DXR] %5.1f%%  %d / %d faces  (%d lit, %d / %d luxels)  %s elapsed",
					pct, done, g_ptFaceTotal, baked, luxelsDone, luxelTotal, elapsedStr );
	}
	PtProgressLine( finish, line );
}
static bool PtFaceIsBakeable( int facenum )
{
	if ( facenum < 0 || facenum >= numfaces )
		return false;
	if ( !( g_FacesVisibleToLights[facenum >> 3] & ( 1 << ( facenum & 7 ) ) ) )
		return false;
	dface_t *f = &g_pFaces[facenum];
	if ( texinfo[f->texinfo].flags & TEX_SPECIAL )
		return false;
	// Displacements are included - BuildSamplesAndLuxels_DoFast routes to Disp samples.
	if ( g_FacePatches.Element( facenum ) == g_FacePatches.InvalidIndex() )
		return false;
	return true;
}

static void PtFinishFaceLighting( int facenum, int normalCount, LightingValue_t *pIndirect[NUM_BUMP_VECTS + 1] )
{
	facelight_t *fl = &facelight[facenum];
	dface_t *f = &g_pFaces[facenum];

	// Build Var(mean) grid for Sakai from values stashed in pIndirect sun channel.
	std::vector<float> luxelVar;
	const float *varPtr = nullptr;
	if ( g_bPathTraceDenoise && g_PathTraceDenoiser == PT_DENOISER_SAKAI && fl->numsamples > 0 )
	{
		luxelVar.assign( (size_t)normalCount * (size_t)fl->numsamples, 0.0f );
		for ( int n = 0; n < normalCount; ++n )
		{
			for ( int i = 0; i < fl->numsamples; ++i )
				luxelVar[n * fl->numsamples + i] = pIndirect[n][i].m_flDirectSunAmount;
		}
		varPtr = luxelVar.data();
	}

	const bool bDeferIsland = g_bPathTraceDenoise && PathTraceDenoise_IsReady()
		&& g_PathTraceDenoiser != PT_DENOISER_SAKAI && f->dispinfo == -1;
	PtDenoiseFaceSamples( fl, f, normalCount, false, varPtr, !bDeferIsland );
	for ( int n = 0; n < normalCount; ++n )
	{
		for ( int i = 0; i < fl->numsamples; ++i )
			fl->light[0][n][i].m_vecLighting += pIndirect[n][i].m_vecLighting;
		free( pIndirect[n] );
		pIndirect[n] = nullptr;
	}
	BuildPatchLights( facenum );
	FreeSampleWindings( fl );
}

static bool PtPrepareFaceSamples( int facenum, lightinfo_t &l, int &normalCount, bool &needsBump,
								  Vector bumpVects[NUM_BUMP_VECTS] )
{
	dface_t *f = &g_pFaces[facenum];
	facelight_t *fl = &facelight[facenum];
	InitLightinfo( &l, facenum );
	if ( !BuildSamplesAndLuxels_DoFast( &l, fl, facenum ) )
		CalcPoints( &l, fl, facenum );

	needsBump = ( texinfo[f->texinfo].flags & SURF_BUMPLIGHT ) != 0;
	normalCount = needsBump ? ( NUM_BUMP_VECTS + 1 ) : 1;
	f->styles[0] = 0;
	for ( int n = 0; n < normalCount; ++n )
		fl->light[0][n] = (LightingValue_t *)calloc( fl->numsamples, sizeof( LightingValue_t ) );

	if ( needsBump )
	{
		texinfo_t *pTexinfo = &texinfo[f->texinfo];
		GetBumpNormals( pTexinfo->textureVecsTexelsPerWorldUnits[0],
						pTexinfo->textureVecsTexelsPerWorldUnits[1],
						l.facenormal, l.facenormal, bumpVects );
	}
	return fl->numsamples > 0;
}

static void PathTraceBakeOneFace( int facenum )
{
	facelight_t *fl = &facelight[facenum];
	lightinfo_t l;
	int normalCount = 1;
	bool needsBump = false;
	Vector bumpVects[NUM_BUMP_VECTS];
	if ( !PtPrepareFaceSamples( facenum, l, normalCount, needsBump, bumpVects ) )
		return;

	LightingValue_t *pIndirect[NUM_BUMP_VECTS + 1];
	for ( int n = 0; n < normalCount; ++n )
		pIndirect[n] = (LightingValue_t *)calloc( fl->numsamples, sizeof( LightingValue_t ) );

	const float luxelWorld = ( fl->worldAreaPerLuxel > 1e-6f ) ? sqrtf( fl->worldAreaPerLuxel ) : 2.0f;

	// Free spatial AA: cycle path samples across an NxN luxel footprint (same spp).
	int aaN = g_nPathTraceLuxelAA;
	if ( aaN < 1 ) aaN = 1;
	if ( aaN > 5 ) aaN = 5;
	const int nAa = aaN * aaN;
	Vector aaOff[25];
	const Vector *pAa = nullptr;
	int nAaUse = 0;
	if ( aaN > 1 )
	{
		const float span = 0.66f; // corners at +/-0.33 luxels
		int tap = 0;
		for ( int ty = 0; ty < aaN; ++ty )
		{
			const float v = ( (float)ty / (float)( aaN - 1 ) - 0.5f ) * span;
			for ( int tx = 0; tx < aaN; ++tx )
			{
				const float u = ( (float)tx / (float)( aaN - 1 ) - 0.5f ) * span;
				aaOff[tap].Init();
				VectorMA( aaOff[tap], u, l.luxelToWorldSpace[0], aaOff[tap] );
				VectorMA( aaOff[tap], v, l.luxelToWorldSpace[1], aaOff[tap] );
				++tap;
			}
		}
		pAa = aaOff;
		nAaUse = nAa;
	}

	for ( int i = 0; i < fl->numsamples; ++i )
	{
		if ( g_bInterrupt )
			break;

		Vector n0 = fl->sample[i].normal;
		if ( n0.LengthSqr() < 1e-6f )
			n0 = l.facenormal;
		VectorNormalize( n0 );
		// Bias along sample normal (required for displacements; base face plane is wrong).
		Vector pos = PtLuxelOriginInFront( fl->sample[i].pos, n0, luxelWorld );

		Vector normals[NUM_BUMP_VECTS + 1];
		normals[0] = n0;
		for ( int bb = 0; bb < NUM_BUMP_VECTS; ++bb )
			normals[bb + 1] = needsBump ? bumpVects[bb] : n0;

		Vector emitGlow( 0, 0, 0 );
		const bool bEmitGlow = VRadEmit_SampleAtPos( facenum, fl->sample[i].pos, emitGlow );

		for ( int ni = 0; ni < normalCount; ++ni )
		{
			Vector direct, indirect;
			float sunAmt = 0.0f;
			float varMean = 0.0f;
			PtBakeAtPosition( pos, normals[ni], luxelWorld, facenum, ni, direct, sunAmt, indirect,
							  &varMean, pAa, nAaUse );

			// $vrad_emit* self-glow (pathtrace replaces BuildFacelights; classic uses post-pass).
			if ( bEmitGlow )
				direct += emitGlow;

			fl->light[0][ni][i].m_vecLighting = direct;
			fl->light[0][ni][i].m_flDirectSunAmount = sunAmt;
			pIndirect[ni][i].m_vecLighting = indirect;
			// Stash Var(mean luminance) for Sakai (unused by FinalLightFace).
			pIndirect[ni][i].m_flDirectSunAmount = varMean;
		}

		g_ptLuxelsInFlight.fetch_add( 1, std::memory_order_relaxed );
		if ( ( i & 7 ) == 0 )
			PtReportProgress( false );
	}

	PtFinishFaceLighting( facenum, normalCount, pIndirect );
	g_ptFacesBaked.fetch_add( 1, std::memory_order_relaxed );
	g_ptLuxelsBaked.fetch_add( fl->numsamples, std::memory_order_relaxed );
}

static void PathTraceFaceWorker( int iThread, int facenum )
{
	(void)iThread;
	if ( facenum < 0 || facenum >= numfaces )
		return;

	dface_t *f = &g_pFaces[facenum];
	f->lightofs = -1;
	for ( int j = 0; j < MAXLIGHTMAPS; j++ )
		f->styles[j] = 255;

	if ( g_bInterrupt )
	{
		g_ptFaceProgress.fetch_add( 1, std::memory_order_relaxed );
		PtReportProgress( false );
		return;
	}

	if ( PtFaceIsBakeable( facenum ) )
		PathTraceBakeOneFace( facenum );

	g_ptFaceProgress.fetch_add( 1, std::memory_order_relaxed );
	PtReportProgress( g_ptFaceProgress.load() >= g_ptFaceTotal );
}

static void PathTraceClearAllFaces()
{
	for ( int facenum = 0; facenum < numfaces; ++facenum )
	{
		dface_t *f = &g_pFaces[facenum];
		f->lightofs = -1;
		for ( int j = 0; j < MAXLIGHTMAPS; j++ )
			f->styles[j] = 255;
	}
}


// ---------------------------------------------------------------------------
// GPU luxel baker (DXR RayQuery) - Frostbite-style: all work on GPU, luxels batched.
// Hard NEE (no CHSS soft); use -pt_cpu for full CPU soft-shadow path.
// ---------------------------------------------------------------------------
static void PtPackGpuLights( std::vector<PtGpuBakeLight> &out, std::vector<float> &iesAtlasOut,
							 ProjGpuArray_t &cookieArr, ProjGpuArray_t &cubeArr )
{
	iesAtlasOut.clear();
	const int nIesLayers = IES_BuildGpuAtlas( iesAtlasOut );
	Proj_AssignGpuLayers();
	Proj_BuildGpuCookieArray( cookieArr );
	Proj_BuildGpuCubeArray( cubeArr );

	out.clear();
	int nEmit = 0, nLocal = 0, nSky = 0, nIes = 0, nProj = 0;
	for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
	{
		if ( dl->light.style != 0 )
			continue;

		PtGpuBakeLight L = {};
		L.origin[0] = dl->light.origin.x;
		L.origin[1] = dl->light.origin.y;
		L.origin[2] = dl->light.origin.z;
		L.intensity[0] = dl->light.intensity.x;
		L.intensity[1] = dl->light.intensity.y;
		L.intensity[2] = dl->light.intensity.z;
		L.dir[0] = dl->light.normal.x;
		L.dir[1] = dl->light.normal.y;
		L.dir[2] = dl->light.normal.z;
		L.stopdot = dl->light.stopdot;
		L.stopdot2 = dl->light.stopdot2;
		L.exponent = dl->light.exponent;
		L.constant_attn = dl->light.constant_attn;
		L.linear_attn = dl->light.linear_attn;
		L.quadratic_attn = dl->light.quadratic_attn;
		L.fadeStart = dl->m_flStartFadeDistance;
		L.fadeEnd = dl->m_flEndFadeDistance;
		L.capDist = dl->m_flCapDist;
		L.areaR2 = dl->m_flAreaRadius2;
		L.sunExtent = dl->m_flSunAngularExtent;
		L.volumeRadius = ( dl->m_flVolumeRadius > 0.0f ) ? dl->m_flVolumeRadius : g_flPtLightRadius;
		L.facenum = dl->facenum;
		L.envId = dl->m_nEnvId;
		L.isLocal = 0;
		L.power = 0.0f;
		L.iesScale = 0.0f;
		L.iesMaxIntensity = 0.0f;
		L.iesLayer = -1;
		L.projRight[0] = dl->m_vecProjRight.x;
		L.projRight[1] = dl->m_vecProjRight.y;
		L.projRight[2] = dl->m_vecProjRight.z;
		L.projOuterCos = dl->light.stopdot2;
		L.projUp[0] = dl->m_vecProjUp.x;
		L.projUp[1] = dl->m_vecProjUp.y;
		L.projUp[2] = dl->m_vecProjUp.z;
		L.projMode = 0.0f;
		L.projLayer = -1;
		L.projFrameMode = 0;
		if ( dl->m_pProj )
		{
			const ProjKind_t kind = Proj_Kind( dl->m_pProj );
			const int layer = Proj_GpuLayer( dl->m_pProj );
			if ( layer >= 0 && kind == PROJ_2D )
			{
				L.projMode = 1.0f;
				L.projLayer = layer;
				L.projFrameMode = ( dl->m_nProjFrameMode == (int)PROJ_FRAME_FIT ) ? 1 : 0;
				++nProj;
			}
			else if ( layer >= 0 && kind == PROJ_CUBE )
			{
				L.projMode = 2.0f;
				L.projLayer = layer;
				L.projFrameMode = 0;
				++nProj;
			}
			else if ( layer >= 0 && kind == PROJ_SPHERE )
			{
				L.projMode = 3.0f;
				L.projLayer = layer;
				L.projFrameMode = Proj_IsEquirect( dl->m_pProj ) ? 1 : 0; // 1=equirect, 0=matcap
				++nProj;
			}
		}
		if ( dl->light.type == emit_spotlight && dl->m_pIes )
		{
			L.iesScale = dl->m_flIesScale;
			L.iesMaxIntensity = dl->m_flIesMaxIntensity;
			L.iesLayer = IES_GpuLayer( dl->m_pIes );
			if ( L.iesLayer >= 0 )
			{
				// Cone fields unused when IES is set - store IES right basis here.
				L.stopdot = dl->m_vecIesRight.x;
				L.stopdot2 = dl->m_vecIesRight.y;
				L.exponent = dl->m_vecIesRight.z;
				// Spots don't use areaR2 - pack IES max clamp into that GPU slot.
				L.areaR2 = L.iesMaxIntensity;
				++nIes;
			}
		}

		switch ( dl->light.type )
		{
		case emit_point:
			L.type = (float)PT_GPU_LIGHT_POINT;
			L.isLocal = 1;
			L.power = PtLightPower( dl );
			++nLocal;
			break;
		case emit_spotlight:
			L.type = (float)PT_GPU_LIGHT_SPOT;
			L.isLocal = 1;
			L.power = PtLightPower( dl );
			++nLocal;
			break;
		case emit_skylight:
			L.type = (float)PT_GPU_LIGHT_SKY;
			L.dir[0] = -dl->light.normal.x;
			L.dir[1] = -dl->light.normal.y;
			L.dir[2] = -dl->light.normal.z;
			++nSky;
			break;
		case emit_surface:
			// Textured $vrad_emit faces -> GPU area mesh, not point proxies.
			if ( VRadEmitArea_FaceHasMesh( dl->facenum ) )
				continue;
			L.type = (float)PT_GPU_LIGHT_SURFACE;
			++nEmit;
			break;
		case emit_skyambient:
			L.type = (float)PT_GPU_LIGHT_SKYAMB;
			break;
		default:
			continue;
		}
		out.push_back( L );
	}
	Msg( "[PathTrace-DXR] GPU lights packed: %d emit_surface, %d sky, %d local (%d IES, %d proj), %d IES atlas layer(s)\n",
		 nEmit, nSky, nLocal, nIes, nProj, nIesLayers );
}

static void PtBuildTriAlbedo( std::vector<float> &rgb )
{
	const uint32_t n = PathTraceDXR_CapturedTriCount();
	rgb.assign( (size_t)n * 3u, 0.45f );

	// Physical GI: per-triangle albedo from -texbounce ($basetexture). Must use captured
	// g_ptTris positions (OptimizedTriangleList indices diverge when props are instanced).
	if ( !g_bTexturedBounce )
	{
		g_bTexturedBounce = true;
		Msg( "[PathTrace-DXR] Enabling -texbounce for physical colored bounce.\n" );
	}
	BounceAlbedo_EnsureCache();

	int nTex = 0, nFlat = 0, nProp = 0, nFaced = 0;
	for ( uint32_t i = 0; i < n; ++i )
	{
		Vector a, b, c;
		uint32_t flags = 0;
		if ( !PathTraceDXR_GetCapturedTri( i, a, b, c, &flags ) )
			continue;

		// Unique-model prop meshes are local-space; keep neutral albedo (shadow casters).
		if ( flags & TRACE_ID_STATICPROP )
		{
			rgb[i * 3 + 0] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 0.40f;
			++nProp;
			continue;
		}

		Vector centroid = ( a + b + c ) * ( 1.0f / 3.0f );
		Vector nrm;
		Vector e1 = b - a;
		Vector e2 = c - a;
		CrossProduct( e1, e2, nrm );
		if ( VectorNormalize( nrm ) < 1e-6f )
			nrm.Init( 0, 0, 1 );

		Vector alb( 0.45f, 0.45f, 0.45f );
		bool got = false;

		// Prefer face id packed into TRACE_ID_OPAQUE (facenum+1) / FILTER (raw facenum).
		int facenum = -1;
		if ( flags & TRACE_ID_FILTER )
			facenum = (int)( flags & 0x00FFFFFFu );
		else if ( flags & TRACE_ID_OPAQUE )
		{
			const unsigned packed = flags & 0x00FFFFFFu;
			if ( packed != 0 )
				facenum = (int)packed - 1;
		}

		if ( facenum >= 0 && facenum < numfaces )
		{
			Vector acc( 0, 0, 0 );
			int ns = 0;
			const Vector pts[4] = { a, b, c, centroid };
			for ( int s = 0; s < 4; ++s )
			{
				Vector tex;
				if ( BounceAlbedo_SampleFace( facenum, pts[s], tex ) )
				{
					acc += tex;
					++ns;
				}
			}
			if ( ns > 0 )
			{
				alb = acc * ( 1.0f / (float)ns );
				got = true;
				++nFaced;
			}
			else
			{
				const int ti = g_pFaces[facenum].texinfo;
				if ( ti >= 0 && ti < texinfo.Count() )
				{
					const int td = texinfo[ti].texdata;
					if ( td >= 0 && td < numtexdata )
					{
						alb = dtexdata[td].reflectivity;
						BounceAlbedo_SanitizeCompressionChroma( alb );
						got = true;
						++nFaced;
					}
				}
			}
		}

		if ( !got )
		{
			// Legacy: nearest patch in cluster (can bleed from colorful neighbors).
			alb = PtGetHitAlbedo( centroid, nrm, true, false, 0 );
			BounceAlbedo_SanitizeCompressionChroma( alb );
		}

		rgb[i * 3 + 0] = alb.x;
		rgb[i * 3 + 1] = alb.y;
		rgb[i * 3 + 2] = alb.z;
		if ( g_bTexturedBounce && ( alb.x != alb.y || alb.y != alb.z ) )
			++nTex;
		else
			++nFlat;
	}
	Msg( "[PathTrace-DXR] Tri albedo: %d textured, %d flat/grey, %d prop-model (neutral), %d face-tagged.\n",
		 nTex, nFlat, nProp, nFaced );
}

// Per-triangle $vrad_filter GPU meta + Texture2DArray of unique filter images.
static void PtBuildTriFilterGpu( std::vector<float> &metaFloat4s, VRadFilterGpuArray_t &arr )
{
	const uint32_t n = PathTraceDXR_CapturedTriCount();
	metaFloat4s.clear();
	arr.width = 1;
	arr.height = 1;
	arr.layers = 1;
	arr.rgba.assign( 4, 255 );
	if ( n == 0 )
		return;

	// Must use captured flags - after SetupAccelerationStructure, GeometryData.m_nTriangleID
	// is no longer valid (union overwritten; IntersectData keeps ID at a different offset).
	std::vector<uint32_t> ids( n, 0 );
	int nFilterIds = 0;
	for ( uint32_t i = 0; i < n; ++i )
	{
		ids[i] = PathTraceDXR_CapturedTriFlags( i );
		if ( ids[i] & TRACE_ID_FILTER )
			++nFilterIds;
	}
	if ( nFilterIds <= 0 )
	{
		Warning( "[PathTrace-DXR] GPU TriFilter: no TRACE_ID_FILTER tris in capture (filter glass will hard-shadow).\n" );
		metaFloat4s.assign( (size_t)n * (size_t)kVRadFilterMetaFloat4s * 4u, 0.0f );
		return;
	}

	VRadFilter_BuildGpuUpload( ids.data(), n, arr, metaFloat4s );
}

// -textureshadows alpha atlas for stochastic GPU any-hit (dappled foliage).
static void PtBuildTriAlphaGpu( std::vector<float> &metaFloat4s, ShadowAlphaGpuArray_t &arr )
{
	const uint32_t n = PathTraceDXR_CapturedTriCount();
	metaFloat4s.assign( (size_t)n * (size_t)kShadowAlphaMetaFloat4s * 4u, 0.0f );
	arr.width = 1;
	arr.height = 1;
	arr.layers = 1;
	arr.rgba.assign( 4, 255 );
	if ( n == 0 || !g_bTextureShadows )
		return;
	const int *mats = PathTraceDXR_CapturedShadowMats();
	if ( !mats )
		return;
	ShadowTexture_BuildGpuUpload( mats, n, arr, metaFloat4s );
}

struct PtGpuFaceJob
{
	int facenum;
	int normalCount;
	int sampleBase;
	LightingValue_t *pIndirect[NUM_BUMP_VECTS + 1];
};

static bool PathTraceBakeWorldFacesGPU()
{
	if ( !PathTraceDXR_DeviceReady() )
		return false;

	std::vector<PtGpuBakeLight> lights;
	std::vector<float> iesAtlas;
	ProjGpuArray_t projCookies, projCubes;
	PtPackGpuLights( lights, iesAtlas, projCookies, projCubes );
	if ( lights.empty() )
	{
		Warning( "[PathTrace-DXR] GPU bake: no lights - falling back to CPU.\n" );
		return false;
	}

	Msg( "[PathTrace-DXR] Building per-triangle albedo for GPU baker...\n" );
	std::vector<float> albedo;
	PtBuildTriAlbedo( albedo );
	std::vector<float> filterMeta;
	VRadFilterGpuArray_t filterArray;
	PtBuildTriFilterGpu( filterMeta, filterArray );
	std::vector<float> alphaMeta;
	ShadowAlphaGpuArray_t alphaArray;
	PtBuildTriAlphaGpu( alphaMeta, alphaArray );

	PtGpuBakeParams params = {};
	params.spp = 1; // real spp set per pass via GpuBakeConfigurePass
	params.bounces = (uint32_t)max( 0, g_ptBounces );
	params.lightSamples = (uint32_t)max( 0, g_nPathTraceLightSamples );
	params.softSamplesMax = (uint32_t)max( 1, g_nPtSoftSamples );
	params.occludeBias = g_flPtOccludeBias;
	params.maxTraceLen = (float)MAX_TRACE_LENGTH;
	params.fireflyCap = 2500.0f;
	params.penumbraScale = g_flPtLightPenumbra;
	params.softMode = (uint32_t)( ( g_nPtSoftMode != 0 ) ? 1 : 0 );
	params.triCount = PathTraceDXR_CapturedTriCount();
	if ( g_flPtLightRadius > 0.0f || params.softSamplesMax > 1 )
	{
		Msg( "[PathTrace-DXR] GPU soft shadows: radius=%.1f penumbra=%.2f softsamples=%u mode=%s\n",
			 g_flPtLightRadius, g_flPtLightPenumbra, params.softSamplesMax,
			 params.softMode ? "all" : "direct" );
	}
	{
		Vector skyAmb( 0, 0, 0 );
		for ( size_t i = 0; i < lights.size(); ++i )
		{
			if ( lights[i].type > 3.5f ) // SKYAMB
			{
				skyAmb.x += lights[i].intensity[0];
				skyAmb.y += lights[i].intensity[1];
				skyAmb.z += lights[i].intensity[2];
			}
		}
		params.skyAmb[0] = skyAmb.x;
		params.skyAmb[1] = skyAmb.y;
		params.skyAmb[2] = skyAmb.z;
	}

	std::vector<PtGpuEmitTri> emitGpu;
	std::vector<float> emitCdf;
	const int nEmitTris = VRadEmitArea_TriCount();
	params.emitTriCount = (uint32_t)max( 0, nEmitTris );
	// 0 = shader samples all tris; else power-sample N (capped).
	params.emitSamples = (uint32_t)max( 0, g_nPathTraceEmitSamples );
	params.envVolCount = (uint32_t)max( 0, LightEnv_VolumeCount() );
	params.defaultBounceIntensity = LightEnv_GetDefaultBounceIntensity();
	params.bounceVolCount = (uint32_t)max( 0, BounceVol_VolumeCount() );
	params.cliBounceBoost = g_flBounceBoost;
	// Pathtrace keeps linear RGB lightxalbedo. -bounce_chroma is artistic Oklab sat - ignore.
	params.cliBounceChroma = 0.0f;
	params.spectralMode = g_bPathTraceSpectral ? 1u : 0u;
	if ( g_flBounceChroma > 0.0f )
	{
		Msg( "[PathTrace-DXR] Ignoring -bounce_chroma %.2f (artistic). Bounce uses -texbounce albedo in linear RGB.\n",
			 g_flBounceChroma );
	}
	if ( params.spectralMode )
		Msg( "[PathTrace-DXR] Spectral transport: 4 stratified wavelengths/path (compact RGB lobes + CIE).\n" );
	if ( params.envVolCount > 0 )
		Msg( "[PathTrace-DXR] GPU light_env_vol: %u volume(s) - weighted sky/ambient, shadow filters, BounceVol tint\n",
			 params.envVolCount );
	if ( params.bounceVolCount > 0 )
		Msg( "[PathTrace-DXR] GPU light_bounce_vol: %u volume(s) - boost/chroma at hit (CLI boost=%.2f chroma=%.2f)\n",
			 params.bounceVolCount, g_flBounceBoost, g_flBounceChroma );
	if ( nEmitTris > 0 )
	{
		const VRadEmitTri_t *src = VRadEmitArea_Tris();
		const float *cdf = VRadEmitArea_PowerCdf();
		emitGpu.resize( (size_t)nEmitTris );
		emitCdf.resize( (size_t)nEmitTris );
		for ( int i = 0; i < nEmitTris; ++i )
		{
			PtGpuEmitTri &D = emitGpu[i];
			const VRadEmitTri_t &S = src[i];
			D.v0[0] = S.v0.x; D.v0[1] = S.v0.y; D.v0[2] = S.v0.z; D.area = S.area;
			D.v1[0] = S.v1.x; D.v1[1] = S.v1.y; D.v1[2] = S.v1.z; D.power = S.power;
			D.v2[0] = S.v2.x; D.v2[1] = S.v2.y; D.v2[2] = S.v2.z; D.facenum = (float)S.facenum;
			D.n[0] = S.n.x; D.n[1] = S.n.y; D.n[2] = S.n.z; D.padN = 0;
			D.e0[0] = S.e0.x; D.e0[1] = S.e0.y; D.e0[2] = S.e0.z; D.pad0 = 0;
			D.e1[0] = S.e1.x; D.e1[1] = S.e1.y; D.e1[2] = S.e1.z; D.pad1 = 0;
			D.e2[0] = S.e2.x; D.e2[1] = S.e2.y; D.e2[2] = S.e2.z; D.pad2 = 0;
			emitCdf[i] = cdf[i];
		}
		Msg( "[PathTrace-DXR] GPU area emit: %d tris, emit_samples=%u%s\n",
			 nEmitTris, params.emitSamples,
			 ( params.emitSamples == 0 ) ? " (all tris)" : "" );
	}

	if ( !PathTraceDXR_GpuBakeBegin( lights.data(), (uint32_t)lights.size(),
									 albedo.data(), (uint32_t)( albedo.size() / 3 ), params,
									 nEmitTris > 0 ? emitGpu.data() : nullptr,
									 (uint32_t)nEmitTris,
									 nEmitTris > 0 ? emitCdf.data() : nullptr,
									 filterMeta.data(),
									 filterArray.rgba.data(),
									 (uint32_t)filterArray.width,
									 (uint32_t)filterArray.height,
									 (uint32_t)filterArray.layers,
									 iesAtlas.data(),
									 (uint32_t)max( 1, (int)( iesAtlas.size() / max( 1, IES_GpuResV() * IES_GpuResH() ) ) ),
									 projCookies.rgba.data(),
									 (uint32_t)projCookies.width,
									 (uint32_t)projCookies.height,
									 (uint32_t)projCookies.layers,
									 projCubes.rgba.data(),
									 (uint32_t)projCubes.width,
									 (uint32_t)projCubes.height,
									 (uint32_t)projCubes.layers,
									 alphaMeta.data(),
									 alphaArray.rgba.data(),
									 (uint32_t)alphaArray.width,
									 (uint32_t)alphaArray.height,
									 (uint32_t)alphaArray.layers ) )
	{
		Warning( "[PathTrace-DXR] GPU baker init failed - falling back to CPU.\n" );
		return false;
	}
	fflush( stdout );

	// Prefer large luxel batches + thicker spp chunks (fewer upload/readback syncs).
	// Only throttle under settings that previously TDR'd the driver (very high spp,
	// and all-emit tris combined with high spp). A mid-bake all-black batch is often
	// just fully-shadowed faces - not a reason to shrink the schedule.
	int kBatch = 65536;
	int sppChunk = ( g_ptSpp > 32 ) ? 16 : max( 1, g_ptSpp );
	if ( g_ptSpp >= 512 )
	{
		kBatch = 16384;
		sppChunk = 8;
	}
	if ( g_ptSpp >= 1024 )
	{
		kBatch = 8192;
		sppChunk = 4;
	}
	if ( g_nPathTraceEmitSamples == 0 && nEmitTris > 16 && g_ptSpp >= 512 )
	{
		kBatch = min( kBatch, 8192 );
		sppChunk = min( sppChunk, 4 );
	}
	Msg( "[PathTrace-DXR] GPU bake schedule: %d luxels/batch, %d spp/dispatch (%d spp total)\n",
		 kBatch, sppChunk, g_ptSpp );
	fflush( stdout );

	std::vector<PtGpuBakeLuxel> jobs;
	std::vector<PtGpuFaceJob> faces;
	jobs.reserve( kBatch );
	faces.reserve( 256 );

	auto flush = [&]() -> bool
	{
		if ( jobs.empty() )
			return true;

		const int sppTotal = max( 1, g_ptSpp );
		const int chunk = max( 1, min( sppChunk, sppTotal ) );

		std::vector<PtGpuBakeResult> acc( jobs.size() );
		std::vector<PtGpuBakeResult> temp( jobs.size() );
		for ( size_t i = 0; i < acc.size(); ++i )
		{
			acc[i].radiance[0] = acc[i].radiance[1] = acc[i].radiance[2] = 0.0f;
			acc[i].sunAmt = 0.0f;
		}

		for ( int off = 0; off < sppTotal; off += chunk )
		{
			const int n = min( chunk, sppTotal - off );
			PathTraceDXR_GpuBakeConfigurePass( (uint32_t)n, (uint32_t)off );
			if ( !PathTraceDXR_GpuBakeLuxels( jobs.data(), (uint32_t)jobs.size(), temp.data() ) )
				return false;
			// GPU returns mean over n samples - weight by n for the overall mean.
			const float w = (float)n;
			for ( size_t i = 0; i < acc.size(); ++i )
			{
				acc[i].radiance[0] += temp[i].radiance[0] * w;
				acc[i].radiance[1] += temp[i].radiance[1] * w;
				acc[i].radiance[2] += temp[i].radiance[2] * w;
				acc[i].sunAmt += temp[i].sunAmt * w;
			}
		}
		const float inv = 1.0f / (float)sppTotal;
		for ( size_t i = 0; i < acc.size(); ++i )
		{
			acc[i].radiance[0] *= inv;
			acc[i].radiance[1] *= inv;
			acc[i].radiance[2] *= inv;
			acc[i].sunAmt *= inv;
		}

		for ( size_t fi = 0; fi < faces.size(); ++fi )
		{
			PtGpuFaceJob &fj = faces[fi];
			facelight_t *fl = &facelight[fj.facenum];
			const int nSamp = fl->numsamples;
			const int nc = fj.normalCount;
			for ( int s = 0; s < nSamp; ++s )
			{
				for ( int ni = 0; ni < nc; ++ni )
				{
					const int idx = fj.sampleBase + s * nc + ni;
					if ( idx < 0 || idx >= (int)acc.size() )
					{
						Warning( "[PathTrace-DXR] GPU bake index OOB (idx=%d acc=%d) - aborting batch.\n",
								 idx, (int)acc.size() );
						return false;
					}
					const PtGpuBakeResult &r = acc[idx];
					Vector rad( r.radiance[0], r.radiance[1], r.radiance[2] );
					if ( !_finite( rad.x ) || !_finite( rad.y ) || !_finite( rad.z ) )
						rad.Init();
					fl->light[0][ni][s].m_vecLighting = rad;
					fl->light[0][ni][s].m_flDirectSunAmount = r.sunAmt;
					fj.pIndirect[ni][s].m_vecLighting.Init();
					fj.pIndirect[ni][s].m_flDirectSunAmount = 0.0f;
				}

				Vector emitGlow( 0, 0, 0 );
				if ( VRadEmit_SampleAtPos( fj.facenum, fl->sample[s].pos, emitGlow ) )
				{
					for ( int ni = 0; ni < nc; ++ni )
						fl->light[0][ni][s].m_vecLighting += emitGlow;
				}
			}
			PtFinishFaceLighting( fj.facenum, fj.normalCount, fj.pIndirect );
			g_ptFacesBaked.fetch_add( 1, std::memory_order_relaxed );
			g_ptLuxelsBaked.fetch_add( nSamp, std::memory_order_relaxed );
			g_ptLuxelsInFlight.fetch_add( nSamp, std::memory_order_relaxed );
			g_ptFaceProgress.fetch_add( 1, std::memory_order_relaxed );
		}
		PtReportProgress( false );
		jobs.clear();
		faces.clear();
		return true;
	};

	Msg( "[PathTrace-DXR] GPU baking world lightmaps (%d spp, %d bounces, hard NEE)...\n",
		 g_ptSpp, g_ptBounces );

	for ( int facenum = 0; facenum < numfaces; ++facenum )
	{
		if ( g_bInterrupt )
			break;

		dface_t *f = &g_pFaces[facenum];
		f->lightofs = -1;
		for ( int j = 0; j < MAXLIGHTMAPS; j++ )
			f->styles[j] = 255;

		if ( !PtFaceIsBakeable( facenum ) )
		{
			g_ptFaceProgress.fetch_add( 1, std::memory_order_relaxed );
			continue;
		}

		lightinfo_t l;
		int normalCount = 1;
		bool needsBump = false;
		Vector bumpVects[NUM_BUMP_VECTS];
		if ( !PtPrepareFaceSamples( facenum, l, normalCount, needsBump, bumpVects ) )
		{
			g_ptFaceProgress.fetch_add( 1, std::memory_order_relaxed );
			continue;
		}

		facelight_t *fl = &facelight[facenum];
		PtGpuFaceJob fj = {};
		fj.facenum = facenum;
		fj.normalCount = normalCount;
		fj.sampleBase = (int)jobs.size();
		for ( int n = 0; n < normalCount; ++n )
			fj.pIndirect[n] = (LightingValue_t *)calloc( fl->numsamples, sizeof( LightingValue_t ) );

		const float luxelWorld = ( fl->worldAreaPerLuxel > 1e-6f ) ? sqrtf( fl->worldAreaPerLuxel ) : 2.0f;
		int aaN = g_nPathTraceLuxelAA;
		if ( aaN < 1 ) aaN = 1;
		if ( aaN > 5 ) aaN = 5;
		// Base-face luxel axes dig into displaced height - skip planar AA on disps.
		if ( f->dispinfo != -1 )
			aaN = 1;

		for ( int i = 0; i < fl->numsamples; ++i )
		{
			Vector n0 = fl->sample[i].normal;
			if ( n0.LengthSqr() < 1e-6f )
				n0 = l.facenormal;
			VectorNormalize( n0 );
			// Bias along sample normal (required for displacements; base face plane is wrong).
			Vector pos = PtLuxelOriginInFront( fl->sample[i].pos, n0, luxelWorld );

			Vector normals[NUM_BUMP_VECTS + 1];
			normals[0] = n0;
			for ( int bb = 0; bb < NUM_BUMP_VECTS; ++bb )
				normals[bb + 1] = needsBump ? bumpVects[bb] : n0;

			for ( int ni = 0; ni < normalCount; ++ni )
			{
				PtGpuBakeLuxel job = {};
				job.pos[0] = pos.x; job.pos[1] = pos.y; job.pos[2] = pos.z;
				job.luxelWorld = luxelWorld;
				job.normal[0] = normals[ni].x;
				job.normal[1] = normals[ni].y;
				job.normal[2] = normals[ni].z;
				job.seed = PtWorldSeed( pos - normals[ni] * 1.0f, luxelWorld, ni );
				job.axisU[0] = l.luxelToWorldSpace[0].x;
				job.axisU[1] = l.luxelToWorldSpace[0].y;
				job.axisU[2] = l.luxelToWorldSpace[0].z;
				job.axisV[0] = l.luxelToWorldSpace[1].x;
				job.axisV[1] = l.luxelToWorldSpace[1].y;
				job.axisV[2] = l.luxelToWorldSpace[1].z;
				job.faceNum = facenum;
				job.aaN = aaN;
				job.skipPropIndex = -1;
				jobs.push_back( job );
			}
		}
		faces.push_back( fj );

		if ( (int)jobs.size() >= kBatch )
		{
			if ( !flush() )
			{
				PathTraceDXR_GpuBakeEnd();
				PtEndProgressLine();
				PathTraceDXR_ReportGpuBakeDarkStats( "world lightmaps" );
				return false;
			}
		}
	}

	if ( !g_bInterrupt )
	{
		if ( !flush() )
		{
			PathTraceDXR_GpuBakeEnd();
			PtEndProgressLine();
			PathTraceDXR_ReportGpuBakeDarkStats( "world lightmaps" );
			return false;
		}
	}
	else
	{
		for ( size_t fi = 0; fi < faces.size(); ++fi )
		{
			for ( int n = 0; n < faces[fi].normalCount; ++n )
				free( faces[fi].pIndirect[n] );
		}
	}

	PathTraceDXR_GpuBakeEnd();
	if ( g_bInterrupt )
	{
		PtEndProgressLine();
		PathTraceDXR_ReportGpuBakeDarkStats( "world lightmaps" );
		return false;
	}
	PtReportProgress( true );
	PathTraceDXR_ReportGpuBakeDarkStats( "world lightmaps" );
	return true;
}

bool PathTraceDXR_BakeWorldFaces()
{
	g_bPathTraceActive = false;
	if ( !g_bPathTraceRequested )
		return false;

	Msg( "\n[PathTrace-DXR] Initializing path tracer...\n" );

	const bool bDxrOk = PathTraceDXR_DeviceInit( g_nPathTraceDevice );
	if ( !bDxrOk )
	{
		Warning( "[PathTrace-DXR] DXR device init failed - using SSE Trace4Rays only.\n" );
	}
	if ( g_RtEnv.OptimizedTriangleList.Count() <= 0 )
	{
		Warning( "[PathTrace-DXR] No ray-trace scene - falling back to stock radiosity.\n" );
		PathTraceDXR_DeviceShutdown();
		return false;
	}

	if ( g_bPathTraceDenoise )
	{
		if ( !PathTraceDenoise_Init() )
			Warning( "[PathTrace-DXR] OIDN init failed - continuing without denoise.\n" );
	}
	else
	{
		PathTraceDenoise_Shutdown();
	}

	g_ptSpp = g_nPathTraceSamples;
	if ( g_ptSpp <= 0 )
	{
		if ( g_flSkySampleScale >= 4.0f )
			g_ptSpp = 16;
		else if ( do_fast )
			g_ptSpp = 4;
		else
			g_ptSpp = 12;
	}
	g_ptBounces = g_nPathTraceBounces;
	if ( g_ptBounces < 0 )
		g_ptBounces = do_fast ? 1 : 3;
	if ( g_ptBounces > 16 )
		g_ptBounces = 16;

	VRadEmitArea_Build();
	PtRebuildLightLists();
	if ( VRadEmitArea_TriCount() > 0 )
	{
		Msg( "[PathTrace-DXR] $vrad_emit area lights: %d tris (PBRT area NEE; point proxies skipped)\n",
			 VRadEmitArea_TriCount() );
	}

	PathTraceClearAllFaces();

	g_ptFaceProgress = 0;
	g_ptFacesBaked = 0;
	g_ptLuxelsBaked = 0;
	g_ptLuxelsInFlight = 0;
	g_ptFaceTotal = numfaces;
	g_ptLuxelTotalEst = PtEstimateTotalLuxels();
	g_ptBakeStartTime = Plat_FloatTime();
	g_ptLastReportMs = (long long)( g_ptBakeStartTime * 1000.0 );
	g_ptEmaLuxelsPerSec = 0.0;
	g_ptSmoothedEtaSec = -1.0;
	g_ptEtaLastLuxels = 0;
	g_ptEtaLastTime = 0.0;
	g_ptEtaLastDecayTime = 0.0;

	const char *denoiseTag = " (denoise off)";
	if ( g_bPathTraceDenoise && PathTraceDenoise_IsReady() )
	{
		static char tagBuf[64];
		Q_snprintf( tagBuf, sizeof( tagBuf ), " + %s", PathTraceDenoise_Name() );
		denoiseTag = tagBuf;
	}
	else if ( g_bPathTraceDenoise )
		denoiseTag = " + denoise(init failed)";
	Msg( "[PathTrace-DXR] Baking world lightmaps: %d faces (~%d luxels), direct 1x + %d indirect spp, %d bounces, AA %dx%d%s\n",
		 g_ptBakeableFacesEst, g_ptLuxelTotalEst, g_ptSpp, g_ptBounces,
		 g_nPathTraceLuxelAA, g_nPathTraceLuxelAA, denoiseTag );
	if ( g_ptBakeableFacesEst > 0 )
	{
		Msg( "[PathTrace-DXR] ~%.1f luxels/lit-face - Hammer scale 2 should be high; if low, re-run VBSP.\n",
			 (double)g_ptLuxelTotalEst / (double)g_ptBakeableFacesEst );
	}
	Msg( "[PathTrace-DXR] Progress updates ~1/sec on one line (%% by luxel work, faces, ETA)...\n" );
	PtReportProgress( false );

	bool bGpuOk = false;
	bool bGpuAttempted = false;
	if ( g_bPathTraceGpu && !g_bPathTraceCpuForced && bDxrOk && PathTraceDXR_DeviceReady() )
	{
		Msg( "[PathTrace-DXR] Using GPU RayQuery luxel baker (hard NEE; -pt_cpu for CPU soft path).\n" );
		bGpuAttempted = true;
		bGpuOk = PathTraceBakeWorldFacesGPU();
		if ( !bGpuOk && !g_bInterrupt )
		{
			PtEndProgressLine();
			Warning( "[PathTrace-DXR] GPU bake failed after %d lit faces / %d luxels.\n",
					 g_ptFacesBaked.load(), g_ptLuxelsBaked.load() );
			// CPU path is ~50-200x slower; mid-bake fallback turns a 2-min GPU job
			// into hours. Only fall back if GPU never produced anything.
			if ( g_ptFacesBaked.load() > 0 )
			{
				Warning( "[PathTrace-DXR] Refusing CPU fallback (partial GPU bake). "
						 "Re-run with -pt_lights 8 or lower -pt_samples, or raise Windows TDR delay.\n" );
				PathTraceDenoise_Shutdown();
				PathTraceDXR_DeviceShutdown();
				return false;
			}
			Warning( "[PathTrace-DXR] GPU produced no faces - falling back to CPU path tracer.\n" );
		}
	}
	else if ( g_bPathTraceCpuForced )
	{
		Msg( "[PathTrace-DXR] CPU path tracer forced (-pt_cpu).\n" );
	}
	else if ( !bDxrOk )
	{
		Msg( "[PathTrace-DXR] DXR unavailable - CPU SSE path tracer.\n" );
	}

	if ( !bGpuOk && !g_bInterrupt && !( bGpuAttempted && g_ptFacesBaked.load() > 0 ) )
		RunThreadsOnIndividual( numfaces, false, PathTraceFaceWorker );

	const double elapsed = Plat_FloatTime() - g_ptBakeStartTime;
	char elapsedStr[32];
	PtFormatDuration( (int)elapsed, elapsedStr, sizeof( elapsedStr ) );

	if ( g_bInterrupt )
	{
		PtEndProgressLine();
		Warning( "[PathTrace-DXR] Interrupted after %s (%d / %d faces, %d lit).\n",
				 elapsedStr,
				 g_ptFaceProgress.load(), g_ptFaceTotal,
				 g_ptFacesBaked.load() );
		PathTraceDenoise_Shutdown();
		PathTraceDXR_DeviceShutdown();
		return false;
	}

	// GPU path already finished progress + dark stats; CPU path needs a final 100% line.
	if ( !bGpuOk )
		PtReportProgress( true );
	PtDenoiseCoplanarIslands();
	PathTraceDenoise_Shutdown();
	g_bPathTraceActive = true;
	Msg( "[PathTrace-DXR] World-face path trace complete in %s - %d lit faces, %d luxels.\n",
		 elapsedStr, g_ptFacesBaked.load(), g_ptLuxelsBaked.load() );
	return true;
}

bool PathTraceDXR_CanBakeProps()
{
	return g_bPathTraceActive && g_bPathTraceGpu && !g_bPathTraceCpuForced && PathTraceDXR_DeviceReady();
}

static bool PathTraceBeginGpuBakeSession( int bounces, int lightSamples, int softSamplesMax, int emitSamples )
{
	std::vector<PtGpuBakeLight> lights;
	std::vector<float> iesAtlas;
	ProjGpuArray_t projCookies, projCubes;
	PtPackGpuLights( lights, iesAtlas, projCookies, projCubes );
	if ( lights.empty() )
		return false;

	std::vector<float> albedo;
	PtBuildTriAlbedo( albedo );
	std::vector<float> filterMeta;
	VRadFilterGpuArray_t filterArray;
	PtBuildTriFilterGpu( filterMeta, filterArray );
	std::vector<float> alphaMeta;
	ShadowAlphaGpuArray_t alphaArray;
	PtBuildTriAlphaGpu( alphaMeta, alphaArray );

	PtGpuBakeParams params = {};
	params.spp = 1;
	params.bounces = (uint32_t)max( 0, bounces );
	params.lightSamples = (uint32_t)max( 0, lightSamples );
	params.softSamplesMax = (uint32_t)max( 1, softSamplesMax );
	params.occludeBias = g_flPtOccludeBias;
	params.maxTraceLen = (float)MAX_TRACE_LENGTH;
	params.fireflyCap = 2500.0f;
	params.penumbraScale = g_flPtLightPenumbra;
	params.softMode = (uint32_t)( ( g_nPtSoftMode != 0 ) ? 1 : 0 );
	params.triCount = PathTraceDXR_CapturedTriCount();
	{
		Vector skyAmb( 0, 0, 0 );
		for ( size_t i = 0; i < lights.size(); ++i )
		{
			if ( lights[i].type > 3.5f )
			{
				skyAmb.x += lights[i].intensity[0];
				skyAmb.y += lights[i].intensity[1];
				skyAmb.z += lights[i].intensity[2];
			}
		}
		params.skyAmb[0] = skyAmb.x;
		params.skyAmb[1] = skyAmb.y;
		params.skyAmb[2] = skyAmb.z;
	}

	std::vector<PtGpuEmitTri> emitGpu;
	std::vector<float> emitCdf;
	const int nEmitTris = VRadEmitArea_TriCount();
	params.emitTriCount = (uint32_t)max( 0, nEmitTris );
	params.emitSamples = (uint32_t)max( 0, emitSamples );
	params.envVolCount = (uint32_t)max( 0, LightEnv_VolumeCount() );
	params.defaultBounceIntensity = LightEnv_GetDefaultBounceIntensity();
	params.bounceVolCount = (uint32_t)max( 0, BounceVol_VolumeCount() );
	params.cliBounceBoost = g_flBounceBoost;
	// Pathtrace keeps linear RGB lightxalbedo. -bounce_chroma is artistic Oklab sat - ignore.
	params.cliBounceChroma = 0.0f;
	params.spectralMode = g_bPathTraceSpectral ? 1u : 0u;
	if ( g_flBounceChroma > 0.0f )
	{
		Msg( "[PathTrace-DXR] Ignoring -bounce_chroma %.2f (artistic). Bounce uses -texbounce albedo in linear RGB.\n",
			 g_flBounceChroma );
	}
	if ( params.spectralMode )
		Msg( "[PathTrace-DXR] Spectral transport: 4 stratified wavelengths/path (compact RGB lobes + CIE).\n" );
	if ( nEmitTris > 0 )
	{
		const VRadEmitTri_t *src = VRadEmitArea_Tris();
		const float *cdf = VRadEmitArea_PowerCdf();
		emitGpu.resize( (size_t)nEmitTris );
		emitCdf.resize( (size_t)nEmitTris );
		for ( int i = 0; i < nEmitTris; ++i )
		{
			PtGpuEmitTri &D = emitGpu[i];
			const VRadEmitTri_t &S = src[i];
			D.v0[0] = S.v0.x; D.v0[1] = S.v0.y; D.v0[2] = S.v0.z; D.area = S.area;
			D.v1[0] = S.v1.x; D.v1[1] = S.v1.y; D.v1[2] = S.v1.z; D.power = S.power;
			D.v2[0] = S.v2.x; D.v2[1] = S.v2.y; D.v2[2] = S.v2.z; D.facenum = (float)S.facenum;
			D.n[0] = S.n.x; D.n[1] = S.n.y; D.n[2] = S.n.z; D.padN = 0;
			D.e0[0] = S.e0.x; D.e0[1] = S.e0.y; D.e0[2] = S.e0.z; D.pad0 = 0;
			D.e1[0] = S.e1.x; D.e1[1] = S.e1.y; D.e1[2] = S.e1.z; D.pad1 = 0;
			D.e2[0] = S.e2.x; D.e2[1] = S.e2.y; D.e2[2] = S.e2.z; D.pad2 = 0;
			emitCdf[i] = cdf[i];
		}
	}

	return PathTraceDXR_GpuBakeBegin( lights.data(), (uint32_t)lights.size(),
									  albedo.data(), (uint32_t)( albedo.size() / 3 ), params,
									  nEmitTris > 0 ? emitGpu.data() : nullptr,
									  (uint32_t)nEmitTris,
									  nEmitTris > 0 ? emitCdf.data() : nullptr,
									  filterMeta.data(),
									  filterArray.rgba.data(),
									  (uint32_t)filterArray.width,
									  (uint32_t)filterArray.height,
									  (uint32_t)filterArray.layers,
									  iesAtlas.data(),
									  (uint32_t)max( 1, (int)( iesAtlas.size() / max( 1, IES_GpuResV() * IES_GpuResH() ) ) ),
									  projCookies.rgba.data(),
									  (uint32_t)projCookies.width,
									  (uint32_t)projCookies.height,
									  (uint32_t)projCookies.layers,
									  projCubes.rgba.data(),
									  (uint32_t)projCubes.width,
									  (uint32_t)projCubes.height,
									  (uint32_t)projCubes.layers,
									  alphaMeta.data(),
									  alphaArray.rgba.data(),
									  (uint32_t)alphaArray.width,
									  (uint32_t)alphaArray.height,
									  (uint32_t)alphaArray.layers );
}

bool PathTraceDXR_BakePropSamples( const PtGpuBakeLuxel *samples, unsigned nSamples,
								   PtGpuBakeResult *outResults, bool bLightmapQuality,
								   bool bEndSession, unsigned progressBase, unsigned progressTotal )
{
	if ( !PathTraceDXR_CanBakeProps() || !samples || !outResults || nSamples == 0 )
		return false;

	int spp, bounces, lightSamples, emitSamples, softSamplesMax;
	const char *tag;
	if ( bLightmapQuality )
	{
		// Same integrator settings as world brush luxels.
		spp = max( 1, g_ptSpp );
		bounces = max( 0, g_ptBounces );
		if ( bounces > 16 ) bounces = 16;
		lightSamples = max( 0, g_nPathTraceLightSamples );
		emitSamples = max( 0, g_nPathTraceEmitSamples );
		softSamplesMax = max( 1, g_nPtSoftSamples );
		tag = "prop lightmaps";
	}
	else
	{
		spp = g_nPathTracePropSamples;
		if ( spp <= 0 )
			spp = max( 8, g_ptSpp / 4 );
		if ( spp < 1 ) spp = 1;
		if ( spp > 4096 ) spp = 4096;

		// Match world bounce depth - indoor props need multi-bounce GI.
		bounces = g_nPathTracePropBounces;
		if ( bounces < 0 )
			bounces = max( 0, g_ptBounces );
		if ( bounces > 16 ) bounces = 16;

		// Locals: bounce 0 always evaluates all lights (shader). Later bounces power-sample
		// when LightSamples > 0. Default 32 keeps indoor direct correct without allxsppxbounce cost.
		lightSamples = g_nPathTraceLightSamples;
		if ( lightSamples <= 0 )
			lightSamples = 32;
		emitSamples = g_nPathTraceEmitSamples;
		if ( emitSamples <= 0 )
			emitSamples = 32;
		softSamplesMax = 1;
		tag = "prop verts";
	}

	const bool bNewSession = !PathTraceDXR_GpuBakeIsActive();
	if ( bNewSession )
	{
		if ( !PathTraceBeginGpuBakeSession( bounces, lightSamples, softSamplesMax, emitSamples ) )
		{
			Warning( "[PathTrace-DXR] Prop GPU bake session failed (%s).\n", tag );
			return false;
		}
		const unsigned showTotal = progressTotal > 0 ? progressTotal : nSamples;
		Msg( "[PathTrace-DXR] GPU baking %s: %u samples (%d spp, %d bounces, pt_lights=%d%s)...\n",
			 tag, showTotal, spp, bounces, lightSamples,
			 bLightmapQuality ? "" : ", hard NEE" );
		fflush( stdout );
	}

	int kBatch = 65536;
	int sppChunk = ( spp > 32 ) ? 16 : max( 1, spp );
	if ( spp >= 512 ) { kBatch = 16384; sppChunk = 8; }
	if ( spp >= 1024 ) { kBatch = 8192; sppChunk = 4; }

	std::vector<PtGpuBakeResult> temp( (size_t)kBatch );
	std::vector<PtGpuBakeResult> pass( (size_t)kBatch );
	// Accumulate directly into caller's outResults (avoid a second full-size buffer).
	memset( outResults, 0, (size_t)nSamples * sizeof( PtGpuBakeResult ) );

	const unsigned progTotal = progressTotal > 0 ? progressTotal : nSamples;
	// Wall-clock for the whole stream (not each chunk).
	static double s_propStreamStart = 0.0;
	static long long s_propLastReportMs = 0;
	if ( bNewSession || progressBase == 0 )
	{
		s_propStreamStart = Plat_FloatTime();
		s_propLastReportMs = (long long)( s_propStreamStart * 1000.0 );
	}

	auto reportProp = [&]( unsigned localDone, bool bForce )
	{
		const double now = Plat_FloatTime();
		const long long nowMs = (long long)( now * 1000.0 );
		if ( !bForce && nowMs - s_propLastReportMs < 1000 )
			return;
		s_propLastReportMs = nowMs;

		const unsigned overallDone = progressBase + localDone;
		const unsigned showDone = ( overallDone > progTotal ) ? progTotal : overallDone;
		const double elapsed = now - s_propStreamStart;
		float pct = progTotal > 0 ? ( 100.0f * (float)showDone / (float)progTotal ) : 100.0f;
		const bool streamDone = ( progressTotal > 0 )
			? ( progressBase + nSamples >= progressTotal && localDone >= nSamples )
			: ( localDone >= nSamples );
		const bool finish = bForce && ( bEndSession || streamDone );
		if ( finish )
			pct = 100.0f;
		char elapsedStr[32], etaStr[32];
		PtFormatDuration( (int)elapsed, elapsedStr, sizeof( elapsedStr ) );
		etaStr[0] = '\0';
		if ( !finish && showDone > 0 && showDone < progTotal && elapsed > 1.0 )
		{
			const double rate = (double)showDone / elapsed;
			const int etaSec = (int)( ( (double)( progTotal - showDone ) / max( rate, 1e-6 ) ) + 0.5 );
			char tmp[32];
			PtFormatDuration( etaSec, tmp, sizeof( tmp ) );
			Q_snprintf( etaStr, sizeof( etaStr ), "  ~%s left", tmp );
		}
		char line[512];
		Q_snprintf( line, sizeof( line ),
					"[PathTrace-DXR] %s  %5.1f%%  %u / %u  %s elapsed%s",
					tag, pct, finish ? progTotal : showDone, progTotal, elapsedStr, etaStr );
		PtProgressLine( finish, line );
	};

	bool ok = true;
	for ( unsigned base = 0; base < nSamples && ok && !g_bInterrupt; base += (unsigned)kBatch )
	{
		const unsigned n = min( (unsigned)kBatch, nSamples - base );
		for ( unsigned i = 0; i < n; ++i )
		{
			temp[i].radiance[0] = temp[i].radiance[1] = temp[i].radiance[2] = 0.0f;
			temp[i].sunAmt = 0.0f;
		}
		for ( int so = 0; so < spp && ok; so += sppChunk )
		{
			const int nPass = min( sppChunk, spp - so );
			PathTraceDXR_GpuBakeConfigurePass( (uint32_t)nPass, (uint32_t)so );
			if ( !PathTraceDXR_GpuBakeLuxels( samples + base, n, pass.data() ) )
			{
				ok = false;
				break;
			}
			const float w = (float)nPass;
			for ( unsigned i = 0; i < n; ++i )
			{
				temp[i].radiance[0] += pass[i].radiance[0] * w;
				temp[i].radiance[1] += pass[i].radiance[1] * w;
				temp[i].radiance[2] += pass[i].radiance[2] * w;
				temp[i].sunAmt += pass[i].sunAmt * w;
			}
			{
				const double sppFrac = (double)( so + nPass ) / (double)spp;
				const unsigned approxDone = base + (unsigned)( (double)n * sppFrac );
				reportProp( min( approxDone, nSamples ), false );
			}
		}
		const float inv = 1.0f / (float)spp;
		for ( unsigned i = 0; i < n; ++i )
		{
			outResults[base + i].radiance[0] = temp[i].radiance[0] * inv;
			outResults[base + i].radiance[1] = temp[i].radiance[1] * inv;
			outResults[base + i].radiance[2] = temp[i].radiance[2] * inv;
			outResults[base + i].sunAmt = temp[i].sunAmt * inv;
		}
		reportProp( min( base + n, nSamples ), false );
	}

	if ( !ok || g_bInterrupt )
	{
		PtEndProgressLine();
		if ( bEndSession || !ok || g_bInterrupt )
		{
			PathTraceDXR_GpuBakeEnd();
			PathTraceDXR_ReportGpuBakeDarkStats( tag );
		}
		return false;
	}

	reportProp( nSamples, true );
	if ( bEndSession )
	{
		PathTraceDXR_GpuBakeEnd();
		PathTraceDXR_ReportGpuBakeDarkStats( tag );
		Msg( "[PathTrace-DXR] %s complete (%u samples).\n", tag, progTotal > 0 ? progTotal : nSamples );
		fflush( stdout );
	}
	return true;
}

bool PathTraceDXR_BakeDirectOnlySamples( const PtGpuBakeLuxel *samples, unsigned nSamples,
										 PtGpuBakeResult *outResults, int spp )
{
	if ( !PathTraceDXR_CanBakeProps() || !samples || !outResults || nSamples == 0 )
		return false;

	// A 0-bounce session returns direct light only — locals, sun, sky, emit
	// surfaces — with $vrad_filter glass transmission applied by shadow rays.
	// Any open session has different bounce settings; close it first.
	if ( PathTraceDXR_GpuBakeIsActive() )
		PathTraceDXR_GpuBakeEnd();

	const int savedSpp = g_nPathTracePropSamples;
	const int savedBounces = g_nPathTracePropBounces;
	g_nPathTracePropSamples = max( 1, spp );
	g_nPathTracePropBounces = 0;

	const bool ok = PathTraceDXR_BakePropSamples( samples, nSamples, outResults,
												  false /* prop quality */, true /* end session */ );

	g_nPathTracePropSamples = savedSpp;
	g_nPathTracePropBounces = savedBounces;
	return ok;
}

void PathTraceDXR_PropBakeClose( const char *tag, unsigned totalSamples )
{
	const char *t = tag ? tag : "prop bake";
	if ( g_ptProgressLineLen > 0 )
	{
		char line[512];
		Q_snprintf( line, sizeof( line ),
					"[PathTrace-DXR] %s  100.0%%  %u / %u",
					t, totalSamples, totalSamples );
		PtProgressLine( true, line );
	}
	else
	{
		PtEndProgressLine();
	}
	if ( PathTraceDXR_GpuBakeIsActive() )
	{
		PathTraceDXR_GpuBakeEnd();
		PathTraceDXR_ReportGpuBakeDarkStats( t );
	}
	Msg( "[PathTrace-DXR] %s complete (%u samples).\n", t, totalSamples );
	fflush( stdout );
}

void PathTraceDXR_PropBakeSuspend( const char *tag )
{
	const char *t = tag ? tag : "prop bake";
	PtEndProgressLine();
	if ( PathTraceDXR_GpuBakeIsActive() )
	{
		PathTraceDXR_GpuBakeEnd();
		PathTraceDXR_ReportGpuBakeDarkStats( t );
	}
}
