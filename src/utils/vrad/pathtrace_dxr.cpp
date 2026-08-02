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
#include "pathtrace_denoise.h"
#include "envvolume.h"
#include "skyambient.h"
#include "bounce_vol.h"
#include "bounce_albedo.h"
#include "absorb.h"
#include "vrad_emit.h"
#include "vrad_emit_area.h"
#include "vrad_filter.h"
#include "oklab.h"
#include "bsplib.h"
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
int		g_nPathTraceBounces = 0;	// 0 = auto
int		g_nPathTraceDevice = -1;
bool	g_bPathTraceDenoise = false;	// opt-in (-pt_denoise)
int		g_nPathTraceDenoiseRadius = 3;	// Sakai spatial radius (OIDN/OptiX ignore)
float	g_flPathTraceDenoiseStrength = 1.0f;	// blend noisy→denoised (1 = full)
int		g_nPathTraceLuxelAA = 3;	// 1=off .. 5=5x5 free spatial AA
int		g_nPathTraceLightSamples = 0;	// 0 = evaluate all local lights (legacy)
int		g_nPathTraceEmitSamples = 64;	// $vrad_emit area NEE samples (0 = all tris)
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

// PCSS-style penumbra → sample count. Contact (small tHit) → few rays; wide gap → up to cap.
// distLight: receiver→light distance (use large value for sun). tHit: occluder distance along probe.
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
			if ( pTrans ) *pTrans = T;
			return false;
		}
		if ( !hit[0] )
		{
			if ( pHitT ) *pHitT = tmax;
			if ( pTrans ) *pTrans = T;
			return false;
		}
		if ( bSunSkyCountsAsClear && ( flags[0] & TRACE_ID_SKY ) )
		{
			if ( pHitT ) *pHitT = tmax;
			if ( pTrans ) *pTrans = T;
			return false;
		}
		if ( bSunSkyCountsAsClear && LightEnv_HasShadowCastFilters() )
		{
			Vector hitP = pos + dir * t[0];
			if ( LightEnv_ShouldIgnoreSkyOccluder( pos, hitP ) )
			{
				if ( pHitT ) *pHitT = tmax;
				if ( pTrans ) *pTrans = T;
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
		if ( pTrans ) *pTrans = T;
		return true;
	}
	if ( pHitT ) *pHitT = tmax;
	if ( pTrans ) *pTrans = T;
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

// Point / spot attenuation: 1 / (c + lÂ·d + qÂ·dÂ²). CapDist<=0 (calloc) â‡’ uncapped.
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

// emit_surface: stock GatherSampleLight â€” emitter_cos / (rÂ² + RÂ²).
// RÂ² = m_flAreaRadius2 (equiv. disk radiusÂ² from patch area); 0 â‡’ classic 1/rÂ².
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
	// light.normal points along emission; sample sees -normal
	Vector dir = lightToSample;
	VectorNormalize( dir );
	float dot2 = -DotProduct( dir, dl->light.normal );
	if ( dot2 <= dl->light.stopdot2 )
		return 0.0f;
	if ( dot2 >= dl->light.stopdot )
		return 1.0f;
	float mult = ( dot2 - dl->light.stopdot2 ) / ( dl->light.stopdot - dl->light.stopdot2 );
	return max( 0.0f, min( 1.0f, mult ) );
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
// uploading raw TriAlbedo to the GPU — shader applies art with bounce falloff).
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

// RGB transmittance from→to (1 = clear, 0 = blocked). Multiplies through $vrad_filter panes.
static Vector PtVisibilityRGB( const Vector &from, const Vector &to )
{
	Vector delta = to - from;
	float dist = VectorNormalize( delta );
	if ( dist <= g_flPtOccludeBias * 2.0f )
		return Vector( 1, 1, 1 );

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
		// Side occluder ≈ wall meeting this face (hit normal ⟂ face normal).
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

// Veach power heuristic (β = 2).
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
static std::vector<directlight_t *> g_ptEmitLights;		// emit_surface — always all NEE (stock-style)
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

		// Hard NEE to face center. Soft-disk sampling used sqrt(area/π) as radius —
		// on medium/large $vrad_emit faces that disk extends into solids and every
		// soft sample fails occlusion → zero light. Softness comes from area falloff
		// (cos/(r²+R²)) instead; optional small CHSS disk only when soft is on.
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
				float cone = PtSpotCone( dl, delta );
				if ( cone <= 0.0f )
					continue;
				ndl *= cone;
			}

			float falloff = PtPointFalloff( dl, dist );
			if ( falloff <= 0.0f )
				continue;

			Vector est = dl->light.intensity * ( ndl * falloff * absScale );
			if ( est.x + est.y + est.z < 0.02f )
				continue;

			Vector vis = PtVisibilityRGB( pos, lightPos );
			if ( vis.x + vis.y + vis.z < 1e-6f )
				continue;

			accum.x += dl->light.intensity.x * ( ndl * falloff * vis.x * absScale );
			accum.y += dl->light.intensity.y * ( ndl * falloff * vis.y * absScale );
			accum.z += dl->light.intensity.z * ( ndl * falloff * vis.z * absScale );
		}
		accum *= ( 1.0f / (float)ns );
		sum += accum;
		break;
	}
	default:
		break;
	}
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

	// 0 or K>=nTris → evaluate every triangle once (lowest variance).
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
	return intensity * Absorb_DirectScale( pos );
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

	// -pt_bounces N = N indirect hops after the luxel (primary NEE). Path depth = N+1.
	int pathDepth = maxBounces + 1;
	if ( pathDepth < 1 )
		pathDepth = 1;

	for ( int bounce = 0; bounce < pathDepth; ++bounce )
	{
		// --- Next event estimation (all light types) + MIS ---
		// Soft area sampling is expensive: default only on luxel (direct). Indirect uses hard NEE.
		float sunB = 0.0f;
		unsigned neeFlags = PT_NEE_SURFACE;
		if ( bounce == 0 || g_nPtSoftMode != 0 )
			neeFlags |= PT_NEE_SOFT;
		Vector nee = PtSampleDirect( pos, normal,
									 seed + (unsigned)bounce * 31u + (unsigned)sppIndex * 17u,
									 &sunB, skipFace, neeFlags );
		nee = PtClampFirefly( nee, bounce == 0 ? 2500.0f : 400.0f );
		radiance += throughput * nee;
		if ( bounce == 0 )
			sunAmt = sunB;

		// --- BSDF sample (cosine-weighted Lambert) ---
		float u1 = PtHash( seed + (unsigned)bounce * 13u + sppIndex * 7u + 11u );
		float u2 = PtHash( seed + (unsigned)bounce * 17u + sppIndex * 11u + 13u );
		Vector dir = PtCosineHemisphere( normal, u1, u2 );
		const float cosTheta = max( 0.0f, DotProduct( normal, dir ) );
		const float pdfBsdf = cosTheta * invPi;

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
			float tmaxs[1] = { (float)MAX_TRACE_LENGTH };
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

		if ( !bHit )
		{
			// Sky only via BSDF (no skyambient NEE) → weight 1.
			radiance += PtClampFirefly( throughput * PtSampleSkyAmbient( pos, dir ), 400.0f );
			break;
		}

		Vector hitPos = pos + dir * tHit;
		Vector hitNormal = hitN;
		if ( hitNormal.LengthSqr() < 1e-8f )
			hitNormal = -dir;
		if ( DotProduct( hitNormal, dir ) > 0.0f )
			hitNormal = -hitNormal;

		// --- Emissive surface hit (MIS vs NEE; same units as surface NEE) ---
		float pdfLight = 0.0f;
		Vector emitContrib;
		if ( pdfBsdf > 0.0f &&
			 PtSurfaceEmissionContrib( pos, normal, hitPos, hitNormal, skipFace, emitContrib, pdfLight ) )
		{
			const float mis = PtMisWeight( pdfBsdf, pdfLight );
			radiance += PtClampFirefly( throughput * emitContrib * mis, bounce == 0 ? 2500.0f : 400.0f );
			break; // emitters terminate the path
		}

		Vector albedo = PtGetHitAlbedo( hitPos, hitNormal, true, true, (int)bounce );
		albedo *= Absorb_BounceScale( hitPos, pos );
		albedo.x = min( 0.92f, max( 0.0f, albedo.x ) );
		albedo.y = min( 0.92f, max( 0.0f, albedo.y ) );
		albedo.z = min( 0.92f, max( 0.0f, albedo.z ) );

		// Lambert + cosine sampling ⇒ throughput *= albedo.
		throughput *= albedo;

		if ( LightEnv_HasInboundBounceTinting() )
		{
			int recvEnv = LightEnv_GetDominantEnvId( posIn );
			int emitEnv = LightEnv_GetDominantEnvId( hitPos );
			LightEnv_MaybeTintInboundBounce( recvEnv, emitEnv, throughput );
		}

		// Russian roulette after first bounce (Veach efficiency-optimized style).
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
		skipFace = -1; // allow all surface lights after leaving the receiver face

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
// Relative contrast drives blend — soft gradients stay put, binary jags AA.
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
								  const float *varSamples )
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
	if ( bDenoise && width * height < 32 )
		return; // tiny charts: skip denoise, still allow dilate below for any size

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
					// Dark spike in a brighter neighborhood → likely sample-in-solid / backface.
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
		if ( bDenoise && width * height >= 32 )
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
		Msg( "[PathTrace-DXR] Lightmap dims on lit faces: avg max-side %.1f, largest %d luxels (Hammer scale 2 â‰ˆ many faces near 16â€“35).\n",
			 (double)sumMaxDim / (double)g_ptBakeableFacesEst, maxDim );
	}
	return total;
}

// Rate-limited console progress (Hammer-friendly newlines, ~1 Hz).
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
	const float pct = ( luxelsDone > 0 || done >= g_ptFaceTotal )
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

	if ( bShowEta )
	{
		char etaStr[32];
		PtFormatDuration( etaSec, etaStr, sizeof( etaStr ) );
		Msg( "[PathTrace-DXR] %5.1f%%  %d / %d faces  (%d lit, %d / %d luxels)  %s elapsed  ~%s left\n",
			 pct, done, g_ptFaceTotal, baked, luxelsDone, luxelTotal, elapsedStr, etaStr );
	}
	else
	{
		Msg( "[PathTrace-DXR] %5.1f%%  %d / %d faces  (%d lit, %d / %d luxels)  %s elapsed\n",
			 pct, done, g_ptFaceTotal, baked, luxelsDone, luxelTotal, elapsedStr );
	}
	fflush( stdout );
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
	// Displacements are included — BuildSamplesAndLuxels_DoFast routes to Disp samples.
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

	PtDenoiseFaceSamples( fl, f, normalCount, false, varPtr );
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
		const float span = 0.66f; // corners at ±0.33 luxels
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
// GPU luxel baker (DXR RayQuery) — Frostbite-style: all work on GPU, luxels batched.
// Hard NEE (no CHSS soft); use -pt_cpu for full CPU soft-shadow path.
// ---------------------------------------------------------------------------
static void PtPackGpuLights( std::vector<PtGpuBakeLight> &out )
{
	out.clear();
	int nEmit = 0, nLocal = 0, nSky = 0;
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
			// Textured $vrad_emit faces → GPU area mesh, not point proxies.
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
	Msg( "[PathTrace-DXR] GPU lights packed: %d emit_surface, %d sky, %d local (of %d total)\n",
		 nEmit, nSky, nLocal, (int)out.size() );
}

static void PtBuildTriAlbedo( std::vector<float> &rgb )
{
	const uint32_t n = PathTraceDXR_CapturedTriCount();
	rgb.assign( (size_t)n * 3u, 0.45f );
	const int nEnv = g_RtEnv.OptimizedTriangleList.Count();
	// Raw albedo only — GPU ApplyBounceVolAlbedo applies boost/chroma with falloff.
	// Use -texbounce colors when enabled so -bounce_chroma has chroma to boost.
	const bool bTex = g_bTexturedBounce;
	if ( g_flBounceChroma > 0.0f && !bTex )
	{
		Warning( "[PathTrace-DXR] -bounce_chroma %.2f with no -texbounce: flat reflectivity is often near-grey — "
				 "chroma will be weak. Enable -texbounce.\n", g_flBounceChroma );
	}
	if ( bTex )
		BounceAlbedo_EnsureCache();
	for ( uint32_t i = 0; i < n && (int)i < nEnv; ++i )
	{
		const CacheOptimizedTriangle &tri = g_RtEnv.OptimizedTriangleList[i];
		Vector centroid = ( tri.Vertex( 0 ) + tri.Vertex( 1 ) + tri.Vertex( 2 ) ) * ( 1.0f / 3.0f );
		Vector nrm;
		Vector e1 = tri.Vertex( 1 ) - tri.Vertex( 0 );
		Vector e2 = tri.Vertex( 2 ) - tri.Vertex( 0 );
		CrossProduct( e1, e2, nrm );
		VectorNormalize( nrm );
		Vector alb = PtGetHitAlbedo( centroid, nrm, bTex, false, 0 );
		rgb[i * 3 + 0] = alb.x;
		rgb[i * 3 + 1] = alb.y;
		rgb[i * 3 + 2] = alb.z;
	}
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

	// Must use captured flags — after SetupAccelerationStructure, GeometryData.m_nTriangleID
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
	PtPackGpuLights( lights );
	if ( lights.empty() )
	{
		Warning( "[PathTrace-DXR] GPU bake: no lights — falling back to CPU.\n" );
		return false;
	}

	Msg( "[PathTrace-DXR] Building per-triangle albedo for GPU baker...\n" );
	std::vector<float> albedo;
	PtBuildTriAlbedo( albedo );
	std::vector<float> filterMeta;
	VRadFilterGpuArray_t filterArray;
	PtBuildTriFilterGpu( filterMeta, filterArray );

	PtGpuBakeParams params = {};
	params.spp = 1; // real spp set per pass via GpuBakeConfigurePass
	params.bounces = (uint32_t)max( 1, g_ptBounces );
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
	params.cliBounceChroma = g_flBounceChroma;
	if ( params.envVolCount > 0 )
		Msg( "[PathTrace-DXR] GPU light_env_vol: %u volume(s) — weighted sky/ambient, shadow filters, BounceVol tint\n",
			 params.envVolCount );
	if ( params.bounceVolCount > 0 )
		Msg( "[PathTrace-DXR] GPU light_bounce_vol: %u volume(s) — boost/chroma at hit (CLI boost=%.2f chroma=%.2f)\n",
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
									 (uint32_t)filterArray.layers ) )
	{
		Warning( "[PathTrace-DXR] GPU baker init failed — falling back to CPU.\n" );
		return false;
	}
	fflush( stdout );

	// Prefer large luxel batches + thicker spp chunks (fewer upload/readback syncs).
	// Only throttle under settings that previously TDR'd the driver (very high spp,
	// and all-emit tris combined with high spp). A mid-bake all-black batch is often
	// just fully-shadowed faces — not a reason to shrink the schedule.
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
			// GPU returns mean over n samples — weight by n for the overall mean.
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
						Warning( "[PathTrace-DXR] GPU bake index OOB (idx=%d acc=%d) — aborting batch.\n",
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
		// Base-face luxel axes dig into displaced height — skip planar AA on disps.
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
				jobs.push_back( job );
			}
		}
		faces.push_back( fj );

		if ( (int)jobs.size() >= kBatch )
		{
			if ( !flush() )
			{
				PathTraceDXR_GpuBakeEnd();
				return false;
			}
		}
	}

	if ( !g_bInterrupt )
	{
		if ( !flush() )
		{
			PathTraceDXR_GpuBakeEnd();
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
	PtReportProgress( true );
	return !g_bInterrupt;
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
	if ( g_ptBounces <= 0 )
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
	Msg( "[PathTrace-DXR] Progress updates ~1/sec (%% by luxel work, faces, ETA)...\n" );
	PtReportProgress( true );

	bool bGpuOk = false;
	bool bGpuAttempted = false;
	if ( g_bPathTraceGpu && !g_bPathTraceCpuForced && bDxrOk && PathTraceDXR_DeviceReady() )
	{
		Msg( "[PathTrace-DXR] Using GPU RayQuery luxel baker (hard NEE; -pt_cpu for CPU soft path).\n" );
		bGpuAttempted = true;
		bGpuOk = PathTraceBakeWorldFacesGPU();
		if ( !bGpuOk && !g_bInterrupt )
		{
			Warning( "[PathTrace-DXR] GPU bake failed after %d lit faces / %d luxels.\n",
					 g_ptFacesBaked.load(), g_ptLuxelsBaked.load() );
			// CPU path is ~50–200× slower; mid-bake fallback turns a 2‑min GPU job
			// into hours. Only fall back if GPU never produced anything.
			if ( g_ptFacesBaked.load() > 0 )
			{
				Warning( "[PathTrace-DXR] Refusing CPU fallback (partial GPU bake). "
						 "Re-run with -pt_lights 8 or lower -pt_samples, or raise Windows TDR delay.\n" );
				PathTraceDenoise_Shutdown();
				PathTraceDXR_DeviceShutdown();
				return false;
			}
			Warning( "[PathTrace-DXR] GPU produced no faces — falling back to CPU path tracer.\n" );
		}
	}
	else if ( g_bPathTraceCpuForced )
	{
		Msg( "[PathTrace-DXR] CPU path tracer forced (-pt_cpu).\n" );
	}
	else if ( !bDxrOk )
	{
		Msg( "[PathTrace-DXR] DXR unavailable — CPU SSE path tracer.\n" );
	}

	if ( !bGpuOk && !g_bInterrupt && !( bGpuAttempted && g_ptFacesBaked.load() > 0 ) )
		RunThreadsOnIndividual( numfaces, false, PathTraceFaceWorker );

	const double elapsed = Plat_FloatTime() - g_ptBakeStartTime;
	char elapsedStr[32];
	PtFormatDuration( (int)elapsed, elapsedStr, sizeof( elapsedStr ) );

	if ( g_bInterrupt )
	{
		Warning( "[PathTrace-DXR] Interrupted after %s (%d / %d faces, %d lit).\n",
				 elapsedStr,
				 g_ptFaceProgress.load(), g_ptFaceTotal,
				 g_ptFacesBaked.load() );
		PathTraceDenoise_Shutdown();
		PathTraceDXR_DeviceShutdown();
		return false;
	}

	PtReportProgress( true );
	PathTraceDenoise_Shutdown();
	g_bPathTraceActive = true;
	Msg( "[PathTrace-DXR] World-face path trace complete in %s - %d lit faces, %d luxels.\n",
		 elapsedStr, g_ptFacesBaked.load(), g_ptLuxelsBaked.load() );
	return true;
}
