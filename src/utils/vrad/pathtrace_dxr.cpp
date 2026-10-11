//========= Copyright PathRAD contributors. ============//
//
// Path-traced lightmaps (-pathtrace / -dxr).
// GPU does closest-hit (DXR RayQuery). The CPU integrator walks the path,
// including soft lights, sun, and PathRAD volumes.
//
//=============================================================================//

#include "vrad.h"
#include "oklab.h"
#include "qmc_sobol.h"
#include "lightmap.h"
#include "radial.h"
#include "pathtrace_dxr.h"
#include "bake_volume.h"
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
#include "vrad_brdf.h"
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
bool	g_bPathTraceAdaptive = true;	// extra spp where luminance CV stays high
int		g_nPathTraceAdaptiveMax = 0;	// 0 = extra cap equals base spp
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
	if ( nSoft <= 1 )
	{
		out = axis;
		VectorNormalize( out );
		return;
	}
	// Sobol cone + Cranley-Patterson rotate so neighboring luxels don't share a pattern.
	const float u = QmcSobolRot( (unsigned int)i, 0, PtHash( seed + 17u ) );
	const float v = QmcSobolRot( (unsigned int)i, 1, PtHash( seed + 91u ) );
	const float cosTheta = 1.0f - u * ( 1.0f - cosThetaMax );
	const float sinTheta = sqrtf( fmaxf( 0.0f, 1.0f - cosTheta * cosTheta ) );
	const float phi = v * 2.0f * (float)M_PI;
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
	const float u = QmcSobolRot( (unsigned int)i, 0, PtHash( seed + 3u ) );
	const float v = QmcSobolRot( (unsigned int)i, 1, PtHash( seed + 7u ) );
	const float w = QmcSobolRot( (unsigned int)i, 2, PtHash( seed + 11u ) );
	const float r = radius * cbrtf( u );
	const float z = 1.0f - 2.0f * v;
	const float phi = w * 2.0f * (float)M_PI;
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
	const float u = QmcSobolRot( (unsigned int)i, 0, PtHash( seed + 5u ) );
	const float v = QmcSobolRot( (unsigned int)i, 1, PtHash( seed + 13u ) );
	const float r = radius * sqrtf( u );
	const float phi = v * 2.0f * (float)M_PI;
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

static void PtWelfordChunk( float L, float &meanL, float &m2, int &nChunks )
{
	++nChunks;
	const float d = L - meanL;
	meanL += d / (float)nChunks;
	m2 += d * (L - meanL);
}

// Relative standard error of the chunk-mean luminance. Path CV does not shrink
// as spp grows, so testing it ran most luxels out to -pt_adaptive_max and pinned
// the GPU for the whole cap on every compile.
static bool PtLuxelNeedsMoreSpp( float meanL, float m2, int nChunks, int chunkSpp, int nSpp, int sppCap )
{
	(void)chunkSpp;
	if ( nSpp >= sppCap || nChunks < 2 )
		return false;
	if ( meanL < 1e-4f )
		return false;
	const float varChunk = m2 / (float)( nChunks - 1 );
	if ( varChunk <= 0.0f )
		return false;
	const float se = sqrtf( varChunk / (float)nChunks ) / meanL;
	return se > 0.05f;
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

// Place luxel origin in the open room, not in a concave crease.
// Lifting along the receiving normal is not enough: a floor sample next to a
// wall is still in the contact shadow. Also push away from each nearby side
// wall so hemisphere/NEE rays don't immediately hit the adjacent occluder.
static Vector PtLuxelOriginInFront( const Vector &samplePos, const Vector &faceNormal, float luxelWorld, int skipTriId = 0 )
{
	const float baseBias = max( 1.0f, max( g_flPtOccludeBias * 0.4f, luxelWorld * 0.35f ) );
	Vector pos = samplePos + faceNormal * baseBias;

	Vector t1, t2;
	PtOrthonormalBasis( faceNormal, t1, t2 );
	const float probe = max( g_flPtOccludeBias * 2.0f, luxelWorld * 1.5f );
	const float wantDist = max( g_flPtOccludeBias + luxelWorld * 0.40f, luxelWorld * 0.75f );
	float extraN = 0.0f;

	Vector wallN[8];
	float wallT[8];
	int nWalls = 0;

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
		if ( !PathTraceDXR_TraceClosest( origins, dirs, tmins, tmaxs, t, flags, hit, 1, hitN, skipTriId ) )
			continue;
		if ( !hit[0] || ( flags[0] & TRACE_ID_SKY ) )
			continue;
		// Side occluder ~= wall meeting this face (hit normal perp face normal).
		if ( fabsf( DotProduct( hitN[0], faceNormal ) ) > 0.55f )
			continue;

		Vector n = hitN[0];
		if ( n.LengthSqr() < 1e-8f )
			continue;
		VectorNormalize( n );
		// Inward (into-room) wall normal: oppose the probe direction.
		if ( DotProduct( n, dir ) > 0.0f )
			n = -n;

		extraN = max( extraN, ( probe - t[0] ) * 0.35f + 0.35f );

		int slot = -1;
		for ( int w = 0; w < nWalls; ++w )
		{
			if ( DotProduct( wallN[w], n ) > 0.85f )
			{
				slot = w;
				break;
			}
		}
		if ( slot < 0 )
		{
			if ( nWalls >= 8 )
				continue;
			slot = nWalls++;
			wallT[slot] = t[0];
			wallN[slot] = n;
		}
		else if ( t[0] < wallT[slot] )
		{
			wallT[slot] = t[0];
			wallN[slot] = n;
		}
	}

	Vector planar( 0, 0, 0 );
	for ( int w = 0; w < nWalls; ++w )
	{
		const float need = wantDist - wallT[w];
		if ( need > 0.0f )
			planar += wallN[w] * need;
	}
	const float maxPlanar = max( wantDist, luxelWorld * 1.25f );
	const float pLen = planar.Length();
	if ( pLen > maxPlanar && pLen > 1e-6f )
		planar *= ( maxPlanar / pLen );

	const float maxExtra = max( probe, luxelWorld * 2.0f );
	if ( extraN > maxExtra )
		extraN = maxExtra;
	return pos + faceNormal * extraN + planar;
}

void PathTraceDXR_LuxelOriginInFront( const Vector &samplePos, const Vector &faceNormal, float luxelWorld, Vector &outPos, int skipTriId )
{
	outPos = PtLuxelOriginInFront( samplePos, faceNormal, luxelWorld, skipTriId );
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

		float sinExt = 0.0f;
		int nSoft = 1;
		if ( bSoft && dl->m_flSunAngularExtent > 0.0f )
		{
			sinExt = dl->m_flSunAngularExtent;
			nSoft = PtSoftSunCount( sinExt );
			const int cap = PtSoftSampleCap();
			if ( nSoft > cap )
				nSoft = cap;
		}
		// Finite sun wraps past 90°: cull only when the whole disk is below the horizon.
		if ( DotProduct( normal, axis ) <= -sinExt )
			break;

		// No MIS vs cosine BSDF: sky misses add ambient, not the sun disk.
		Vector visAccum( 0, 0, 0 );
		Vector contrib( 0, 0, 0 );
		float nVis = 0.0f;
		for ( int s = 0; s < nSoft; ++s )
		{
			Vector dir;
			PtSoftSunDir( axis, sinExt, s, nSoft, seed, dir );
			const float ndlS = DotProduct( normal, dir );
			if ( ndlS <= 0.0f )
				continue;

			float tHit = 0.0f;
			Vector T;
			const bool blocked = PtProbeOccluder( pos, dir, g_flPtOccludeBias, (float)MAX_TRACE_LENGTH,
												  &tHit, true, &T );
			if ( blocked || ( T.x + T.y + T.z < 1e-6f ) )
				continue;

			visAccum += T;
			contrib.x += ndlS * T.x;
			contrib.y += ndlS * T.y;
			contrib.z += ndlS * T.z;
			nVis += 1.0f;
		}
		if ( nVis > 0.0f )
		{
			const float inv = 1.0f / (float)nSoft;
			sum.x += dl->light.intensity.x * ( contrib.x * inv * wEnv * absScale );
			sum.y += dl->light.intensity.y * ( contrib.y * inv * wEnv * absScale );
			sum.z += dl->light.intensity.z * ( contrib.z * inv * wEnv * absScale );
			if ( pSunAmt )
				*pSunAmt += ( ( visAccum.x + visAccum.y + visAccum.z ) * ( 1.0f / 3.0f ) ) * inv * wEnv;
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
				PtSoftDiskOffset( dl->light.normal, softRadius, s, seed, offset );
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
				PtSoftDiskOffset( diskN, radius, s, seed, offset );

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
	// Sky/sun water attenuation is already in the probe transmittance (Kd / Mode B skip).
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
								int skipFace, int nSamples, float specRough = -1.0f, const Vector *pSpecWo = nullptr )
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

		Vector add;
		add.x = Le.x * ( ndl * emitterCos * absScale * vis.x / ( dist * dist * pdf ) );
		add.y = Le.y * ( ndl * emitterCos * absScale * vis.y / ( dist * dist * pdf ) );
		add.z = Le.z * ( ndl * emitterCos * absScale * vis.z / ( dist * dist * pdf ) );
		if ( specRough >= 0.0f && pSpecWo )
			add *= VRadBrdf_GgxShape( normal, *pSpecWo, delta, specRough );
		sum += add;
	}
	if ( !sampleAll )
		sum *= ( 1.0f / (float)k );
	return sum;
}

static Vector PtLightPrimaryDir( directlight_t *dl, const Vector &pos )
{
	if ( dl->light.type == emit_skylight )
	{
		Vector axis = -dl->light.normal;
		VectorNormalize( axis );
		return axis;
	}
	Vector wi = dl->light.origin - pos;
	VectorNormalize( wi );
	return wi;
}

static bool PtRayTri( const Vector &orig, const Vector &dir, float tmax,
					 const Vector &v0, const Vector &v1, const Vector &v2,
					 float &tOut, Vector &hitOut )
{
	const Vector e1 = v1 - v0;
	const Vector e2 = v2 - v0;
	Vector pvec;
	CrossProduct( dir, e2, pvec );
	const float det = DotProduct( e1, pvec );
	if ( fabsf( det ) < 1e-8f )
		return false;
	const float inv = 1.0f / det;
	const Vector s = orig - v0;
	const float u = DotProduct( s, pvec ) * inv;
	if ( u < 0.0f || u > 1.0f )
		return false;
	Vector qvec;
	CrossProduct( s, e1, qvec );
	const float v = DotProduct( dir, qvec ) * inv;
	if ( v < 0.0f || u + v > 1.0f )
		return false;
	const float t = DotProduct( e2, qvec ) * inv;
	if ( t < 1.0f || t > tmax - 0.5f )
		return false;
	tOut = t;
	hitOut = orig + dir * t;
	return true;
}

static Vector PtSchlick( const Vector &f0, float cosTheta )
{
	const float c = max( 0.0f, min( 1.0f, cosTheta ) );
	const float m = 1.0f - c;
	const float m2 = m * m;
	const float m5 = m2 * m2 * m;
	Vector F;
	F.x = f0.x + ( 1.0f - f0.x ) * m5;
	F.y = f0.y + ( 1.0f - f0.y ) * m5;
	F.z = f0.z + ( 1.0f - f0.z ) * m5;
	return F;
}

static Vector PtClosestOnTri( const Vector &p, const Vector &a, const Vector &b, const Vector &c )
{
	const Vector ab = b - a;
	const Vector ac = c - a;
	const Vector ap = p - a;
	const float d1 = DotProduct( ab, ap );
	const float d2 = DotProduct( ac, ap );
	if ( d1 <= 0.0f && d2 <= 0.0f )
		return a;
	const Vector bp = p - b;
	const float d3 = DotProduct( ab, bp );
	const float d4 = DotProduct( ac, bp );
	if ( d3 >= 0.0f && d4 <= d3 )
		return b;
	const float vc = d1 * d4 - d3 * d2;
	if ( vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f )
	{
		const float v = d1 / ( d1 - d3 );
		return a + ab * v;
	}
	const Vector cp = p - c;
	const float d5 = DotProduct( ab, cp );
	const float d6 = DotProduct( ac, cp );
	if ( d6 >= 0.0f && d5 <= d6 )
		return c;
	const float vb = d5 * d2 - d1 * d6;
	if ( vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f )
	{
		const float w = d2 / ( d2 - d6 );
		return a + ac * w;
	}
	const float va = d3 * d6 - d5 * d4;
	if ( va <= 0.0f && ( d4 - d3 ) >= 0.0f && ( d5 - d6 ) >= 0.0f )
	{
		const float w = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) );
		return b + ( c - b ) * w;
	}
	const float denom = 1.0f / ( va + vb + vc );
	return a + ab * ( vb * denom ) + ac * ( vc * denom );
}

// GGX peak ratio at the surface. 1 on the mirror direction, falling off with roughness.
static float PtGlossPeakRatio( float ndoth, float rough )
{
	float r = rough;
	if ( r < 0.02f )
		r = 0.02f;
	if ( r > 1.0f )
		r = 1.0f;
	const float alpha = r * r;
	const float a2 = alpha * alpha;
	const float d = ndoth * ndoth * ( a2 - 1.0f ) + 1.0f;
	const float q = a2 / max( d, 1e-8f );
	return q * q;
}

// Deterministic reflection. The ray has to land on the triangle. A miss uses the
// closest point only while the half-vector there still sits in the GGX lobe, so
// the beam widens around the real reflection and does not light the rest of the room.
// pLightPos is a point/spot/area sample. pToLight is the sun direction (surface toward sun).
static bool PtMirrorSpreadHit( const VRadBrdfMirrorTri_t &m, const Vector &pos, const Vector &dirIn,
							   float tmax, Vector &hit, Vector &dirOut, float &tOut, float &weight,
							   const Vector *pLightPos, const Vector *pToLight )
{
	const float denom = DotProduct( dirIn, m.n );
	if ( fabsf( denom ) < 1e-6f )
		return false;
	const float t = DotProduct( m.v0 - pos, m.n ) / denom;
	if ( t < 1.0f || t > tmax - 0.5f )
		return false;
	const Vector planeHit = pos + dirIn * t;
	const Vector closest = PtClosestOnTri( planeHit, m.v0, m.v1, m.v2 );
	const float miss = ( closest - planeHit ).Length();
	const float gloss = 1.0f - max( 0.0f, min( 1.0f, m.roughness ) );
	if ( gloss < 1e-3f )
		return false;
	if ( miss <= 0.02f )
	{
		weight = gloss;
		hit = planeHit;
		dirOut = dirIn;
		tOut = t;
		return true;
	}
	// Roughness 0 is the triangle itself. A slop on the shared edge lights it twice.
	if ( m.roughness < kVRadBrdfMirrorRough )
		return false;

	Vector wo = pos - closest;
	if ( VectorNormalize( wo ) < 1e-6f )
		return false;
	Vector wi;
	if ( pLightPos )
		wi = *pLightPos - closest;
	else if ( pToLight )
		wi = *pToLight;
	else
		return false;
	if ( pLightPos && VectorNormalize( wi ) < 1e-6f )
		return false;
	if ( !pLightPos )
		VectorNormalize( wi );
	if ( DotProduct( m.n, wi ) <= 1e-4f || DotProduct( m.n, wo ) <= 1e-4f )
		return false;
	Vector half = wi + wo;
	if ( VectorNormalize( half ) < 1e-6f )
		return false;
	const float ndoth = DotProduct( m.n, half );
	if ( ndoth <= 0.0f )
		return false;
	const float ratio = PtGlossPeakRatio( ndoth, m.roughness );
	if ( ratio < 0.04f )
		return false;
	Vector to = closest - pos;
	const float dist = VectorNormalize( to );
	if ( dist < 1.0f || DotProduct( to, dirIn ) <= 0.0f )
		return false;
	weight = gloss * ratio;
	hit = closest;
	dirOut = to;
	tOut = dist;
	return true;
}

// One-bounce specular connection (image method). Point, spot, sun, emit_surface, $vrad_emit.
static Vector PtConnectMirrorLight( const VRadBrdfMirrorTri_t &m, directlight_t *dl,
									const Vector &pos, const Vector &normal, int facenum )
{
	Vector zero( 0, 0, 0 );
	if ( !dl )
		return zero;
	if ( m.face >= 0 && facenum >= 0 && m.face == facenum )
		return zero;
	if ( DotProduct( m.n, pos - m.v0 ) <= 0.5f )
		return zero;

	const float absScale = Absorb_DirectScale( pos );
	Vector dir;
	float distPath = 0.0f;
	Vector hit;
	float ndl = 0.0f;
	float falloff = 1.0f;
	float cone = 1.0f;
	Vector lightPos;

	if ( dl->light.type == emit_skylight )
	{
		Vector axis = -dl->light.normal;
		VectorNormalize( axis );
		// Direction from the receiver toward the mirror that sees the sun.
		dir = axis - m.n * ( 2.0f * DotProduct( axis, m.n ) );
		if ( VectorNormalize( dir ) < 1e-6f )
			return zero;
		float t = 0.0f;
		float spread = 1.0f;
		Vector dirUse;
		if ( !PtMirrorSpreadHit( m, pos, dir, (float)MAX_TRACE_LENGTH, hit, dirUse, t, spread, nullptr, &axis ) )
			return zero;
		dir = dirUse;
		ndl = DotProduct( normal, dir );
		if ( ndl <= 0.0f )
			return zero;
		if ( DotProduct( m.n, axis ) <= 0.0f )
			return zero;
		const float wEnv = LightEnv_GetWeight( dl->m_nEnvId, pos );
		if ( wEnv <= 0.0f )
			return zero;
		Vector visFrom = pos + normal * g_flPtOccludeBias;
		if ( DotProduct( visFrom - pos, dir ) > t * 0.5f )
			visFrom = pos;
		Vector visR = PtVisibilityRGB( visFrom, hit );
		float tHit = 0.0f;
		Vector T;
		const bool blocked = PtProbeOccluder( hit + m.n * 1.0f, axis, g_flPtOccludeBias,
											  (float)MAX_TRACE_LENGTH, &tHit, true, &T );
		if ( blocked )
			return zero;
		const float cosI = max( 0.0f, DotProduct( m.n, -dir ) );
		const Vector F = PtSchlick( m.f0, cosI );
		Vector sum;
		sum.x = dl->light.intensity.x * F.x * ndl * wEnv * absScale * visR.x * T.x;
		sum.y = dl->light.intensity.y * F.y * ndl * wEnv * absScale * visR.y * T.y;
		sum.z = dl->light.intensity.z * F.z * ndl * wEnv * absScale * visR.z * T.z;
		sum *= spread;
		return sum;
	}

	if ( dl->light.type == emit_surface )
	{
		if ( facenum >= 0 && dl->facenum == facenum )
			return zero;
	}
	else if ( dl->light.type != emit_point && dl->light.type != emit_spotlight )
		return zero;

	lightPos = dl->light.origin;
	const float planeD = DotProduct( lightPos - m.v0, m.n );
	const Vector virt = lightPos - m.n * ( 2.0f * planeD );
	dir = virt - pos;
	distPath = VectorNormalize( dir );
	if ( distPath < 1.0f )
		return zero;
	float t = 0.0f;
	float spread = 1.0f;
	Vector dirUse;
	if ( !PtMirrorSpreadHit( m, pos, dir, distPath, hit, dirUse, t, spread, &lightPos, nullptr ) )
		return zero;
	dir = dirUse;
	ndl = DotProduct( normal, dir );
	if ( ndl <= 0.0f )
		return zero;
	Vector toL = lightPos - hit;
	const float distL = VectorNormalize( toL );
	if ( distL < 1.0f || DotProduct( m.n, toL ) <= 0.0f )
		return zero;
	if ( dl->light.type == emit_spotlight )
	{
		cone = PtSpotCone( dl, -toL );
		if ( cone <= 0.0f )
			return zero;
	}
	const float pathLen = max( t + distL, 1.0f );
	if ( dl->light.type == emit_surface )
	{
		const float emitterCos = -DotProduct( toL, dl->light.normal );
		falloff = PtSurfaceFalloff( dl, pathLen, emitterCos );
	}
	else
		falloff = PtPointFalloff( dl, pathLen );
	if ( falloff <= 0.0f )
		return zero;
	Vector visFrom = pos + normal * g_flPtOccludeBias;
	if ( DotProduct( visFrom - pos, dir ) > t * 0.5f )
		visFrom = pos;
	Vector visR = PtVisibilityRGB( visFrom, hit );
	Vector visL = PtVisibilityRGB( hit + m.n * 1.0f, lightPos );
	if ( visR.x + visR.y + visR.z < 1e-6f || visL.x + visL.y + visL.z < 1e-6f )
		return zero;
	const float cosI = max( 0.0f, DotProduct( m.n, -dir ) );
	const Vector F = PtSchlick( m.f0, cosI );
	Vector intensity = dl->light.intensity;
	Vector sum;
	sum.x = intensity.x * F.x * ( ndl * falloff * cone * absScale * visR.x * visL.x );
	sum.y = intensity.y * F.y * ( ndl * falloff * cone * absScale * visR.y * visL.y );
	sum.z = intensity.z * F.z * ( ndl * falloff * cone * absScale * visR.z * visL.z );
	sum *= spread;
	return sum;
}

// Area-light image: reflect one sample on an emit triangle across the mirror plane.
static Vector PtConnectMirrorEmitTri( const VRadBrdfMirrorTri_t &m, const VRadEmitTri_t &t,
									  const Vector &lightPos, const Vector &Le, float pdf,
									  const Vector &pos, const Vector &normal, int facenum )
{
	Vector zero( 0, 0, 0 );
	if ( pdf <= 1e-20f )
		return zero;
	if ( m.face >= 0 && facenum >= 0 && m.face == facenum )
		return zero;
	if ( facenum >= 0 && t.facenum == facenum )
		return zero;
	if ( DotProduct( m.n, pos - m.v0 ) <= 0.5f )
		return zero;

	const float planeD = DotProduct( lightPos - m.v0, m.n );
	const Vector virt = lightPos - m.n * ( 2.0f * planeD );
	Vector dir = virt - pos;
	const float distPath = VectorNormalize( dir );
	if ( distPath < 1.0f )
		return zero;
	float th = 0.0f;
	float spread = 1.0f;
	Vector hit;
	Vector dirUse;
	if ( !PtMirrorSpreadHit( m, pos, dir, distPath, hit, dirUse, th, spread, &lightPos, nullptr ) )
		return zero;
	dir = dirUse;
	const float ndl = DotProduct( normal, dir );
	if ( ndl <= 0.0f )
		return zero;
	Vector toL = lightPos - hit;
	const float distL = VectorNormalize( toL );
	if ( distL < 1.0f || DotProduct( m.n, toL ) <= 0.0f )
		return zero;
	const float emitterCos = -DotProduct( toL, t.n );
	if ( emitterCos <= 0.0f )
		return zero;

	Vector visFrom = pos + normal * g_flPtOccludeBias;
	if ( DotProduct( visFrom - pos, dir ) > th * 0.5f )
		visFrom = pos;
	Vector visR = PtVisibilityRGB( visFrom, hit );
	Vector visL = PtVisibilityRGB( hit + m.n * 1.0f, lightPos );
	if ( visR.x + visR.y + visR.z < 1e-6f || visL.x + visL.y + visL.z < 1e-6f )
		return zero;

	const float cosI = max( 0.0f, DotProduct( m.n, -dir ) );
	const Vector F = PtSchlick( m.f0, cosI );
	const float absScale = Absorb_DirectScale( pos );
	const float pathLen = max( th + distL, 1.0f );
	const float geom = ndl * emitterCos * absScale * spread / ( pathLen * pathLen * pdf );
	Vector sum;
	sum.x = Le.x * F.x * geom * visR.x * visL.x;
	sum.y = Le.y * F.y * geom * visR.y * visL.y;
	sum.z = Le.z * F.z * geom * visR.z * visL.z;
	return sum;
}

static Vector PtSampleMirrorConnections( const Vector &pos, const Vector &normal, unsigned int seed, int facenum )
{
	Vector sum( 0, 0, 0 );
	const int nM = VRadBrdf_MirrorCount();
	const VRadBrdfMirrorTri_t *tris = VRadBrdf_MirrorTris();
	if ( nM <= 0 || !tris )
		return sum;
	for ( int mi = 0; mi < nM; ++mi )
	{
		for ( size_t i = 0; i < g_ptLocalLights.size(); ++i )
			sum += PtConnectMirrorLight( tris[mi], g_ptLocalLights[i], pos, normal, facenum );
		for ( size_t i = 0; i < g_ptSkyLights.size(); ++i )
			sum += PtConnectMirrorLight( tris[mi], g_ptSkyLights[i], pos, normal, facenum );
		for ( size_t i = 0; i < g_ptEmitLights.size(); ++i )
			sum += PtConnectMirrorLight( tris[mi], g_ptEmitLights[i], pos, normal, facenum );
	}

	const int nTris = VRadEmitArea_TriCount();
	if ( nTris > 0 && VRadEmitArea_PowerSum() > 0.0f )
	{
		const VRadEmitTri_t *etris = VRadEmitArea_Tris();
		const float *cdf = VRadEmitArea_PowerCdf();
		const bool sampleAll = ( g_nPathTraceEmitSamples <= 0 ) || ( g_nPathTraceEmitSamples >= nTris );
		const int k = sampleAll ? nTris : g_nPathTraceEmitSamples;
		Vector areaSum( 0, 0, 0 );
		for ( int s = 0; s < k; ++s )
		{
			int idx;
			float pPick = 1.0f;
			if ( sampleAll )
				idx = s;
			else
			{
				float uPick = ( (float)s + PtHash( seed + (unsigned)s * 91u + 7u ) ) / (float)k;
				if ( uPick < 1e-6f )
					uPick = 1e-6f;
				if ( uPick > 0.999999f )
					uPick = 0.999999f;
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

			const VRadEmitTri_t &t = etris[idx];
			const float u0 = PtHash( seed + (unsigned)s * 97u + 11u );
			const float u1 = PtHash( seed + (unsigned)s * 131u + 23u );
			Vector lightPos, Le;
			PtSampleTriangle( t, u0, u1, lightPos, Le );
			lightPos += t.n * 1.0f;
			Vector live;
			if ( VRadEmit_SampleAtPos( t.facenum, lightPos - t.n * 1.0f, live ) )
				VectorScale( live, lightscale * ( 100.0f * 100.0f ) / ( 128.0f * 128.0f ), Le );

			const float pdfA = 1.0f / max( t.area, 1e-8f );
			const float pdf = sampleAll ? pdfA : ( pPick * pdfA );
			for ( int mi = 0; mi < nM; ++mi )
				areaSum += PtConnectMirrorEmitTri( tris[mi], t, lightPos, Le, pdf, pos, normal, facenum );
		}
		if ( !sampleAll )
			areaSum *= ( 1.0f / (float)k );
		sum += areaSum;
	}
	return sum;
}

static Vector PtSampleDirect( const Vector &pos, const Vector &normal, unsigned int seed, float *pSunAmt,
							  int facenum, unsigned neeFlags, float specRough = -1.0f, const Vector *pSpecWo = nullptr )
{
	Vector sum( 0, 0, 0 );
	float sunAmt = 0.0f;

	for ( size_t i = 0; i < g_ptSkyLights.size(); ++i )
	{
		Vector c = PtEvalOneDirectLight( g_ptSkyLights[i], pos, normal, seed, &sunAmt, facenum, neeFlags );
		if ( specRough >= 0.0f && pSpecWo )
			c *= VRadBrdf_GgxShape( normal, *pSpecWo, PtLightPrimaryDir( g_ptSkyLights[i], pos ), specRough );
		sum += c;
	}

	// lights.rad texlights still as emit_surface point proxies (not $vrad_emit mesh).
	for ( size_t i = 0; i < g_ptEmitLights.size(); ++i )
	{
		Vector c = PtEvalOneDirectLight( g_ptEmitLights[i], pos, normal, seed + (unsigned int)i * 19u,
										 nullptr, facenum, neeFlags );
		if ( specRough >= 0.0f && pSpecWo )
			c *= VRadBrdf_GgxShape( normal, *pSpecWo, PtLightPrimaryDir( g_ptEmitLights[i], pos ), specRough );
		sum += c;
	}

	// $vrad_emit*: textured triangle area lights (not a grid of face centers).
	if ( VRadEmitArea_TriCount() > 0 )
		sum += PtSampleEmitArea( pos, normal, seed + 0xA11Eu, facenum, g_nPathTraceEmitSamples, specRough, pSpecWo );

	const int nLocal = (int)g_ptLocalLights.size();
	int k = g_nPathTraceLightSamples;
	if ( k <= 0 || k >= nLocal )
	{
		for ( int i = 0; i < nLocal; ++i )
		{
			Vector c = PtEvalOneDirectLight( g_ptLocalLights[i], pos, normal, seed + (unsigned int)i * 17u,
											 nullptr, facenum, neeFlags );
			if ( specRough >= 0.0f && pSpecWo )
				c *= VRadBrdf_GgxShape( normal, *pSpecWo, PtLightPrimaryDir( g_ptLocalLights[i], pos ), specRough );
			sum += c;
		}
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
			if ( specRough >= 0.0f && pSpecWo )
				c *= VRadBrdf_GgxShape( normal, *pSpecWo, PtLightPrimaryDir( g_ptLocalLights[idx], pos ), specRough );
			sum += c * ( 1.0f / ( p * (float)k ) );
		}
	}

	// Smooth mirrors: connect this Lambertian vertex to lights through each mirror
	// triangle (planar virtual light). GGX NEE cannot see a point light in a delta lobe.
	if ( specRough < 0.0f )
		sum += PtSampleMirrorConnections( pos, normal, seed, facenum );

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

// Unidirectional path tracer (Veach / PBRT): next-event estimation and MIS
// at every bounce, including the luxel. -uw_volume adds scatter events inside water.
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
	// <0 = Lambertian. The luxel stays Lambertian; roughness changes the surfaces it hits.
	float lobeSpec = -1.0f;
	float lobeRough = 1.0f;
	Vector lobeWo = normal;
	Vector lobeF0( 0, 0, 0 );
	bool lobeVndf = false;
	bool lobeHasDir = false;
	Vector lobeDir = normal;

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
		const float specRough = bInMedium ? -1.0f : lobeSpec;
		Vector nee = PtSampleDirect( pos, neeN,
									 seed + (unsigned)bounce * 31u + (unsigned)sppIndex * 17u,
									 &sunB, skipFace, neeFlags, specRough, ( specRough >= 0.0f ) ? &lobeWo : nullptr );
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
		else if ( lobeHasDir )
		{
			dir = lobeDir;
			pdfBsdf = 1.0f;
		}
		else if ( lobeSpec >= 0.0f && lobeRough < kVRadBrdfMirrorRough )
		{
			dir = VRadBrdf_SampleGloss( normal, -lobeWo, lobeRough, u1, u2 );
			pdfBsdf = 1.0f;
		}
		else if ( lobeSpec >= 0.0f )
		{
			// VNDF sample. Weight is F * G1(wi) / pSpec (the f*cos/pdf of visible-normal
			// sampling). pSpec was applied when the lobe was chosen.
			dir = VRadBrdf_SampleGloss( normal, -lobeWo, lobeRough, u1, u2 );
			pdfBsdf = 1.0f;
			if ( lobeVndf )
			{
				const float ndotl = max( 0.0f, DotProduct( normal, dir ) );
				const float k = ( lobeRough + 1.0f ) * ( lobeRough + 1.0f ) / 8.0f;
				const float g1 = ( ndotl > 1e-4f ) ? ( ndotl / ( ndotl * ( 1.0f - k ) + k ) ) : 0.0f;
				Vector H = lobeWo + dir;
				float cosH = 1.0f;
				if ( VectorNormalize( H ) > 1e-6f )
					cosH = max( 0.0f, DotProduct( lobeWo, H ) );
				const Vector F = PtSchlick( lobeF0, cosH );
				if ( lobeF0.x > 1e-5f )
					throughput.x *= F.x * g1 / lobeF0.x;
				if ( lobeF0.y > 1e-5f )
					throughput.y *= F.y * g1 / lobeF0.y;
				if ( lobeF0.z > 1e-5f )
					throughput.z *= F.z * g1 / lobeF0.z;
			}
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
		unsigned int hitFlags = 0;
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
			hitFlags = flags[0];
			break;
		}

		// Mode B: free-flight scatter before surface
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
			lobeSpec = -1.0f;
			lobeHasDir = false;
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
			// Path exited to sky. Mode B: Beer-Lambert through water (Kd is off in volume mode).
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

		int brdfFace = -1;
		if ( hitFlags & TRACE_ID_FILTER )
			brdfFace = (int)( hitFlags & 0x00FFFFFFu );
		else if ( hitFlags & TRACE_ID_OPAQUE )
		{
			const unsigned packed = hitFlags & 0x00FFFFFFu;
			if ( packed != 0 )
				brdfFace = (int)packed - 1;
		}
		VRadBrdfSample_t rm = VRadBrdf_SampleFace( brdfFace, hitPos );
		VRadBrdfLobes_t lobes = VRadBrdf_Lobes( albedo, rm.roughness, rm.metalness );
		Vector lobeW = albedo;
		lobeSpec = -1.0f;
		lobeRough = 1.0f;
		lobeVndf = false;
		lobeHasDir = false;
		// Reflection stays on, but a ray that dies in the seam beside the hit is
		// replaced by the diffuse bounce. That seam hit is the noisy edge.
		// Direct lights still use the steady connection. lobeSpec stays clear so
		// the light sample does not add a second random spike.
		if ( rm.authored && !lobes.legacy && rm.roughness < 0.98f && lobes.pSpec > 1e-4f )
		{
			const float uLobe = PtHash( seed + (unsigned)bounce * 59u + (unsigned)sppIndex * 23u + 17u );
			Vector refl = dir - hitNormal * ( 2.0f * DotProduct( dir, hitNormal ) );
			if ( uLobe < lobes.pSpec && VectorNormalize( refl ) > 1e-6f )
			{
				float seamT = 0.0f;
				const Vector origin = hitPos + hitNormal * 1.0f;
				const bool seam = PtProbeOccluder( origin, refl, 0.5f, 24.0f, &seamT, false, nullptr );
				if ( !seam )
				{
					lobeHasDir = true;
					lobeDir = refl;
					lobeW = lobes.sharp * ( 1.0f / lobes.pSpec );
				}
				else
					lobeW = lobes.wide + lobes.sharp;
			}
			else
			{
				const float inv = 1.0f / max( 1.0f - lobes.pSpec, 1e-4f );
				lobeW = lobes.wide * inv;
			}
		}
		else if ( rm.authored && !lobes.legacy )
			lobeW = lobes.wide;
		throughput *= lobeW;

		if ( bounce >= 1 )
		{
			float q = min( 0.95f, max( throughput.x, max( throughput.y, throughput.z ) ) );
			float r = PtHash( seed + (unsigned)bounce * 41u + 99u + (unsigned)sppIndex );
			if ( r > q )
				break;
			throughput *= ( 1.0f / max( q, 1e-3f ) );
		}

		pos = hitPos + hitNormal * 1.0f;
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

// Returns full path Li (direct+indirect). outIndirect is unused (left at zero).
// outVarMean: optional variance of mean luminance for Sakai/Welch denoise.
// aaOff / nAa: luxel-footprint offsets. Sample s uses aaOff[s % nAa].
static void PtBakeAtPosition( const Vector &pos, const Vector &normal, float luxelWorld, int facenum,
							  int ni, Vector &outDirect, float &outSun, Vector &outIndirect,
							  float *outVarMean, const Vector *aaOff = nullptr, int nAa = 0 )
{
	const unsigned int seed = PtWorldSeed( pos - normal * 1.0f, luxelWorld, ni );

	const int sppBase = g_ptSpp;
	int extraCap = g_nPathTraceAdaptiveMax;
	if ( extraCap <= 0 )
		extraCap = sppBase;
	if ( extraCap > 4096 )
		extraCap = 4096;
	int sppMax = sppBase + extraCap;
	if ( !g_bPathTraceAdaptive )
		sppMax = sppBase;
	if ( do_fast )
		sppMax = min( sppMax, max( sppBase, sppBase * 2 ) );
	if ( sppMax < sppBase )
		sppMax = sppBase;
	if ( sppMax > 4096 )
		sppMax = 4096;

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
// Variance of the sample mean (input for Sakai / Welch denoise).
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
static std::atomic<int> g_ptAdaptiveLuxels{ 0 };
static std::atomic<int> g_ptAdaptiveJobs{ 0 };
static std::atomic<long long> g_ptAdaptiveExtraSpp{ 0 };
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
				// One-sided AA: soften shadow-side stair-steps into light, but never
				// pull darkness outward onto already-lit luxels (bilinear leak).
				if ( PtLuma( avg ) < lc * 0.92f )
					continue;

				out[ci] = grid[ci] + ( avg - grid[ci] ) * ( aa * 0.65f );
			}
		}

		for ( int i = 0; i < n; ++i )
			grid[i] = out[i];
	}
}

// Clamp AA / leak filters to a reference image (post-denoise, or post-heal if
// denoise is off). Do not pass the raw noisy bake here: a 6% brighten revert
// and 12% darken floor undoes OIDN / OptiX on typical Monte Carlo noise.
static void PtRestoreLitLuxelsAfterFilter( Vector *grid, const float *wgt, const Vector *preFilter,
	int width, int height, float minPreLuma = 0.04f, float maxDarken = 0.88f )
{
	if ( !grid || !wgt || !preFilter || width < 1 || height < 1 )
		return;

	const int n = width * height;
	for ( int i = 0; i < n; ++i )
	{
		if ( wgt[i] <= 0.0f )
			continue;

		const float pre = PtLuma( preFilter[i] );
		const float post = PtLuma( grid[i] );

		if ( pre > 0.01f && post > pre * 1.06f )
		{
			grid[i] = preFilter[i];
			continue;
		}

		if ( pre < minPreLuma )
			continue;

		if ( post >= pre * maxDarken )
			continue;

		grid[i] = preFilter[i] * maxDarken + grid[i] * ( 1.0f - maxDarken );
	}
}

static void PtSuppressLuxelOutliers( Vector *grid, const float *wgt, int width, int height )
{
	if ( !grid || !wgt || width < 2 || height < 2 )
		return;

	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			if ( wgt[idx] <= 0.0f )
				continue;

			std::vector<float> neighL;
			for ( int dt = -1; dt <= 1; ++dt )
			{
				const int nt = t + dt;
				if ( nt < 0 || nt >= height )
					continue;
				for ( int ds = -1; ds <= 1; ++ds )
				{
					if ( !ds && !dt )
						continue;
					const int ns = s + ds;
					if ( ns < 0 || ns >= width )
						continue;
					const int nidx = ns + nt * width;
					if ( wgt[nidx] <= 0.0f )
						continue;
					neighL.push_back( PtLuma( grid[nidx] ) );
				}
			}

			if ( neighL.size() < 5 )
				continue;

			std::sort( neighL.begin(), neighL.end() );
			const float med = neighL[neighL.size() / 2];
			float maxN = 0.0f;
			for ( size_t k = 0; k < neighL.size(); ++k )
				maxN = max( maxN, neighL[k] );

			const float L = PtLuma( grid[idx] );

			if ( med > 0.02f && L > med * 3.0f && L > maxN * 2.0f )
			{
				const float scale = ( med * 1.05f ) / max( L, 1e-4f );
				grid[idx] *= scale;
			}
		}
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

		PtSuppressLuxelOutliers( grid.data(), wgt.data(), width, height );

		const int facenum = (int)( f - g_pFaces );
		// Drop crease energy-loss before denoise so it cannot spread.
		ReconstructLightmapEdgeLuxels( grid.data(), width, height, facenum );

		for ( int i = 0; i < n; ++i )
		{
			rgb[(size_t)i * 3u + 0] = grid[i].x;
			rgb[(size_t)i * 3u + 1] = grid[i].y;
			rgb[(size_t)i * 3u + 2] = grid[i].z;
		}

		const float *varPtr = varSamples ? varGrid.data() : nullptr;
		// The OIDN wrapper pads tiny charts to a usable 16x16 footprint.
		// Do not skip the narrow concave faces that need denoising most.
		if ( bNetworkDenoise && bDenoise )
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

		// Clamp AA to the denoised (or healed) image, not the raw bake.
		std::vector<Vector> postDenoise( grid.begin(), grid.end() );

		// Soften remaining luxel stair-steps on hard shadow edges.
		if ( bNetworkDenoise )
			PtAntialiasHighContrast( grid.data(), wgt.data(), width, height );

		if ( bNetworkDenoise )
			PtRestoreLitLuxelsAfterFilter( grid.data(), wgt.data(), postDenoise.data(), width, height );
		CompressBilinearShadowLeak( grid.data(), width, height );

		// After denoise: restore energy on the last edge/corner luxel so
		// bilinear filtering cannot paint a dark AO band into the face.
		ReconstructLightmapEdgeLuxels( grid.data(), width, height, facenum );

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

	// CustomVBSP charts: PairEdges often misses T-junction / duplicated-vert
	// coplanar splits, so neighbor-based islands stay per-face. Union by chart
	// id so OIDN sees continuous soft shadows across BSP cuts.
	int nChartUnions = 0;
	const int nCharts = LightmapChartCount();
	if ( nCharts > 0 )
	{
		std::vector<int> chartRep( (size_t)nCharts, -1 );
		for ( int facenum = 0; facenum < numfaces; ++facenum )
		{
			if ( !PtDenoiseIslandEligible( facenum ) )
				continue;
			const int cid = GetLightmapChartId( facenum );
			if ( cid < 0 || cid >= nCharts )
				continue;
			if ( chartRep[(size_t)cid] < 0 )
			{
				chartRep[(size_t)cid] = facenum;
				continue;
			}
			const int rep = chartRep[(size_t)cid];
			if ( !PtDenoiseSameLuxelFrame( facenum, rep ) )
				continue;
			if ( !PtDenoiseCoplanarNeighbors( facenum, rep ) )
				continue;
			const int ra = PtDenoiseUfFind( parent, facenum );
			const int rb = PtDenoiseUfFind( parent, rep );
			if ( ra != rb )
			{
				parent[ra] = rb;
				++nChartUnions;
			}
		}
		if ( nChartUnions > 0 )
			Msg( "[PathTrace-DXR] Chart denoise: merged %d face link(s) across %d chart(s)\n",
				 nChartUnions, nCharts );
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

			PathTraceDenoise_DenoiseRGB( rgb.data(), width, height, nullptr );

			for ( int i = 0; i < nPix; ++i )
			{
				grid[i].x = rgb[(size_t)i * 3u + 0];
				grid[i].y = rgb[(size_t)i * 3u + 1];
				grid[i].z = rgb[(size_t)i * 3u + 2];
			}

			std::vector<Vector> postIslandDenoise( grid.begin(), grid.end() );
			PtAntialiasHighContrast( grid.data(), wgt.data(), width, height );
			PtRestoreLitLuxelsAfterFilter( grid.data(), wgt.data(), postIslandDenoise.data(), width, height );
			CompressBilinearShadowLeak( grid.data(), width, height );

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
	if ( !BakeVolume_ShouldBakeFace( facenum ) )
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
	if ( !BakeVolume_ShouldBakeFace( facenum ) )
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

	// Add CPU-split GI before denoise. GPU stores full radiance in fl->light and
	// leaves pIndirect at zero; CPU PathLi writes noisy indirect here, so denoising
	// first then adding GI made -pt_denoise a no-op on the CPU baker.
	for ( int n = 0; n < normalCount; ++n )
	{
		for ( int i = 0; i < fl->numsamples; ++i )
			fl->light[0][n][i].m_vecLighting += pIndirect[n][i].m_vecLighting;
		free( pIndirect[n] );
		pIndirect[n] = nullptr;
	}

	const bool bDeferIsland = g_bPathTraceDenoise && PathTraceDenoise_IsReady()
		&& g_PathTraceDenoiser != PT_DENOISER_SAKAI && f->dispinfo == -1;
	PtDenoiseFaceSamples( fl, f, normalCount, false, varPtr, !bDeferIsland );
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

		if ( !BakeVolume_ShouldBakeSample( facenum, fl->sample[i].pos ) )
		{
			BakeVolume_FillSampleFromCache( facenum, i );
			g_ptLuxelsInFlight.fetch_add( 1, std::memory_order_relaxed );
			continue;
		}

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
// GPU luxel baker (DXR RayQuery): all work on GPU, luxels in batches.
// Hard NEE (no CHSS soft); use -pt_cpu if you need the full CPU soft-shadow path.
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
				L.projFrameMode = dl->m_nProjFrameMode; // 0=fill, 1=fit, 2=hdri, 3=six-face
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

	int nTex = 0, nFlat = 0, nPropTex = 0, nPropFlat = 0, nFaced = 0;
	for ( uint32_t i = 0; i < n; ++i )
	{
		Vector a, b, c;
		uint32_t flags = 0;
		if ( !PathTraceDXR_GetCapturedTri( i, a, b, c, &flags ) )
			continue;

		// Prop meshes are local-space, so face UV lookup does not apply.
		// Textured albedo is sampled from the studio $basetexture at register time.
		if ( flags & TRACE_ID_STATICPROP )
		{
			Vector propAlb;
			if ( PathTraceDXR_GetCapturedTriAlbedo( i, propAlb ) )
			{
				rgb[i * 3 + 0] = propAlb.x;
				rgb[i * 3 + 1] = propAlb.y;
				rgb[i * 3 + 2] = propAlb.z;
				++nPropTex;
				if ( propAlb.x != propAlb.y || propAlb.y != propAlb.z )
					++nTex;
				else
					++nFlat;
			}
			else
			{
				rgb[i * 3 + 0] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 0.40f;
				++nPropFlat;
			}
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
	Msg( "[PathTrace-DXR] Tri albedo: %d textured, %d flat/grey, %d prop textured, %d prop neutral, %d face-tagged.\n",
		 nTex, nFlat, nPropTex, nPropFlat, nFaced );
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
	std::vector<int> jobIndex; // [s * nc + ni] -> index relative to sampleBase, or -1 (cache)
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
	std::vector<float> brdfMeta;
	VRadBrdfGpuArray_t brdfArray;
	VRadBrdf_BuildGpuUpload( brdfArray, brdfMeta );

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
		Msg( "[PathTrace-DXR] Spectral transport: 4 stratified wavelengths/path (Smits RGB->SPD + CIE).\n" );
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
									 (uint32_t)alphaArray.layers,
									 brdfMeta.data(),
									 brdfArray.rgba.data(),
									 (uint32_t)brdfArray.width,
									 (uint32_t)brdfArray.height,
									 (uint32_t)brdfArray.layers ) )
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
		if ( jobs.empty() && faces.empty() )
			return true;

		std::vector<PtGpuBakeResult> acc;
		if ( !jobs.empty() )
		{
			const int sppTotal = max( 1, g_ptSpp );
			int chunk = max( 1, min( sppChunk, sppTotal ) );
			int extraMax = 0;
			if ( g_bPathTraceAdaptive && !do_fast )
			{
				extraMax = g_nPathTraceAdaptiveMax;
				if ( extraMax <= 0 )
					extraMax = sppTotal;
				if ( extraMax > 4096 )
					extraMax = 4096;
				// Need >=2 chunks so per-luxel CV is defined.
				if ( chunk >= sppTotal && sppTotal >= 2 )
					chunk = max( 1, sppTotal / 2 );
			}
			const int sppCap = min( 4096, sppTotal + extraMax );

			acc.resize( jobs.size() );
			std::vector<PtGpuBakeResult> temp( jobs.size() );
			std::vector<int> nSpp( jobs.size(), 0 );
			std::vector<float> meanL( jobs.size(), 0.0f );
			std::vector<float> m2( jobs.size(), 0.0f );
			std::vector<int> nChunks( jobs.size(), 0 );
			std::vector<unsigned char> gotExtra( jobs.size(), 0 );
			for ( size_t i = 0; i < acc.size(); ++i )
			{
				acc[i].radiance[0] = acc[i].radiance[1] = acc[i].radiance[2] = 0.0f;
				acc[i].sunAmt = 0.0f;
			}

			auto addChunk = [&]( const PtGpuBakeResult *src, const int *remap, int nJobs, int n ) -> void
			{
				const float w = (float)n;
				for ( int j = 0; j < nJobs; ++j )
				{
					const int i = remap ? remap[j] : j;
					acc[i].radiance[0] += src[j].radiance[0] * w;
					acc[i].radiance[1] += src[j].radiance[1] * w;
					acc[i].radiance[2] += src[j].radiance[2] * w;
					acc[i].sunAmt += src[j].sunAmt * w;
					nSpp[i] += n;
					const Vector rgb( src[j].radiance[0], src[j].radiance[1], src[j].radiance[2] );
					PtWelfordChunk( PtLuma( rgb ), meanL[i], m2[i], nChunks[i] );
				}
			};

			for ( int off = 0; off < sppTotal; off += chunk )
			{
				const int n = min( chunk, sppTotal - off );
				PathTraceDXR_GpuBakeConfigurePass( (uint32_t)n, (uint32_t)off );
				if ( !PathTraceDXR_GpuBakeLuxels( jobs.data(), (uint32_t)jobs.size(), temp.data() ) )
					return false;
				addChunk( temp.data(), nullptr, (int)jobs.size(), n );
			}

			if ( extraMax > 0 )
			{
				std::vector<PtGpuBakeLuxel> extraJobs;
				std::vector<int> extraMap;
				std::vector<PtGpuBakeResult> extraTemp;
				extraJobs.reserve( jobs.size() / 4 + 8 );
				extraMap.reserve( jobs.size() / 4 + 8 );
				g_ptAdaptiveJobs.fetch_add( (int)jobs.size(), std::memory_order_relaxed );

				int off = sppTotal;
				while ( off < sppCap )
				{
					extraJobs.clear();
					extraMap.clear();
					const int probe = min( chunk, sppCap - off );
					for ( size_t i = 0; i < jobs.size(); ++i )
					{
						if ( PtLuxelNeedsMoreSpp( meanL[i], m2[i], nChunks[i], chunk, nSpp[i], sppCap ) )
						{
							extraMap.push_back( (int)i );
							extraJobs.push_back( jobs[i] );
						}
					}
					if ( extraJobs.empty() )
						break;

					extraTemp.resize( extraJobs.size() );
					PathTraceDXR_GpuBakeConfigurePass( (uint32_t)probe, (uint32_t)off );
					if ( !PathTraceDXR_GpuBakeLuxels( extraJobs.data(), (uint32_t)extraJobs.size(), extraTemp.data() ) )
						return false;
					addChunk( extraTemp.data(), extraMap.data(), (int)extraJobs.size(), probe );
					for ( int idx : extraMap )
					{
						if ( !gotExtra[idx] )
						{
							gotExtra[idx] = 1;
							g_ptAdaptiveLuxels.fetch_add( 1, std::memory_order_relaxed );
						}
					}
					g_ptAdaptiveExtraSpp.fetch_add( (long long)extraJobs.size() * (long long)probe,
													 std::memory_order_relaxed );
					off += probe;
				}
			}

			for ( size_t i = 0; i < acc.size(); ++i )
			{
				const float inv = 1.0f / (float)max( nSpp[i], 1 );
				acc[i].radiance[0] *= inv;
				acc[i].radiance[1] *= inv;
				acc[i].radiance[2] *= inv;
				acc[i].sunAmt *= inv;
			}
		}

		for ( size_t fi = 0; fi < faces.size(); ++fi )
		{
			PtGpuFaceJob &fj = faces[fi];
			facelight_t *fl = &facelight[fj.facenum];
			const int nSamp = fl->numsamples;
			const int nc = fj.normalCount;
			for ( int s = 0; s < nSamp; ++s )
			{
				bool bakedSample = false;
				for ( int ni = 0; ni < nc; ++ni )
				{
					const int slot = s * nc + ni;
					int idx = -1;
					if ( slot >= 0 && slot < (int)fj.jobIndex.size() )
						idx = fj.jobIndex[slot];
					if ( idx < 0 )
					{
						if ( ni == 0 )
							BakeVolume_FillSampleFromCache( fj.facenum, s );
						continue;
					}
					idx += fj.sampleBase;
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
					bakedSample = true;
				}

				if ( bakedSample )
				{
					Vector emitGlow( 0, 0, 0 );
					if ( VRadEmit_SampleAtPos( fj.facenum, fl->sample[s].pos, emitGlow ) )
					{
						for ( int ni = 0; ni < nc; ++ni )
							fl->light[0][ni][s].m_vecLighting += emitGlow;
					}
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

	g_ptAdaptiveLuxels = 0;
	g_ptAdaptiveJobs = 0;
	g_ptAdaptiveExtraSpp = 0;

	Msg( "[PathTrace-DXR] GPU baking world lightmaps (%d spp, %d bounces, hard NEE%s)...\n",
		 g_ptSpp, g_ptBounces, ( g_bPathTraceAdaptive && !do_fast ) ? ", adaptive" : "" );

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
		fj.jobIndex.assign( (size_t)fl->numsamples * (size_t)normalCount, -1 );
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
			if ( !BakeVolume_ShouldBakeSample( facenum, fl->sample[i].pos ) )
				continue;

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
				fj.jobIndex[i * normalCount + ni] = (int)jobs.size() - fj.sampleBase;
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
				job.sampleScale = 1;
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
	if ( g_bPathTraceAdaptive && !do_fast )
	{
		const int nJobs = g_ptAdaptiveJobs.load();
		const int nExtra = g_ptAdaptiveLuxels.load();
		const long long extraSpp = g_ptAdaptiveExtraSpp.load();
		const double avg = ( nExtra > 0 ) ? ( (double)extraSpp / (double)nExtra ) : 0.0;
		Msg( "[PathTrace-DXR] Adaptive sampling: %d / %d luxels got extra spp (avg +%.0f, %.1f%% of luxels)\n",
			 nExtra, nJobs, avg, ( nJobs > 0 ) ? ( 100.0 * nExtra / (double)nJobs ) : 0.0 );
	}
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
		VRadBrdf_EnsureCache();
	}
	else if ( !bDxrOk )
	{
		Msg( "[PathTrace-DXR] DXR unavailable - CPU SSE path tracer.\n" );
	}

	if ( !bGpuOk && !g_bInterrupt && !( bGpuAttempted && g_ptFacesBaked.load() > 0 ) )
	{
		VRadBrdf_EnsureCache();
		RunThreadsOnIndividual( numfaces, false, PathTraceFaceWorker );
	}

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
	std::vector<float> brdfMeta;
	VRadBrdfGpuArray_t brdfArray;
	VRadBrdf_BuildGpuUpload( brdfArray, brdfMeta );

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
		Msg( "[PathTrace-DXR] Spectral transport: 4 stratified wavelengths/path (Smits RGB->SPD + CIE).\n" );
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
									  (uint32_t)alphaArray.layers,
									  brdfMeta.data(),
									  brdfArray.rgba.data(),
									  (uint32_t)brdfArray.width,
									  (uint32_t)brdfArray.height,
									  (uint32_t)brdfArray.layers );
}

bool PathTraceDXR_BakePropSamples( const PtGpuBakeLuxel *samples, unsigned nSamples,
								   PtGpuBakeResult *outResults, bool bLightmapQuality,
								   bool bEndSession, unsigned progressBase, unsigned progressTotal,
								   const char *tagOverride )
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
	if ( tagOverride && tagOverride[0] )
		tag = tagOverride;

	// Prop lightmaps share the world adaptive schedule. Vertex samples stay fixed-spp.
	const bool bAdaptiveLm = bLightmapQuality && g_bPathTraceAdaptive && !do_fast;
	int extraMax = 0;
	int sppCap = spp;

	const bool bNewSession = !PathTraceDXR_GpuBakeIsActive();
	if ( bNewSession )
	{
		if ( !PathTraceBeginGpuBakeSession( bounces, lightSamples, softSamplesMax, emitSamples ) )
		{
			Warning( "[PathTrace-DXR] Prop GPU bake session failed (%s).\n", tag );
			return false;
		}
		const unsigned showTotal = progressTotal > 0 ? progressTotal : nSamples;
		if ( bLightmapQuality )
		{
			Msg( "[PathTrace-DXR] GPU baking %s: %u samples (%d spp, %d bounces, pt_lights=%d, AA %dx%d%s)...\n",
				 tag, showTotal, spp, bounces, lightSamples,
				 g_nPathTraceLuxelAA, g_nPathTraceLuxelAA,
				 bAdaptiveLm ? ", adaptive" : "" );
		}
		else
		{
			Msg( "[PathTrace-DXR] GPU baking %s: %u samples (%d spp, %d bounces, pt_lights=%d, hard NEE)...\n",
				 tag, showTotal, spp, bounces, lightSamples );
		}
		fflush( stdout );
	}

	int kBatch = 65536;
	int sppChunk = ( spp > 32 ) ? 16 : max( 1, spp );
	if ( spp >= 512 ) { kBatch = 16384; sppChunk = 8; }
	if ( spp >= 1024 ) { kBatch = 8192; sppChunk = 4; }
	if ( bAdaptiveLm )
	{
		extraMax = g_nPathTraceAdaptiveMax;
		if ( extraMax <= 0 )
			extraMax = spp;
		if ( extraMax > 4096 )
			extraMax = 4096;
		sppCap = min( 4096, spp + extraMax );
		if ( sppChunk >= spp && spp >= 2 )
			sppChunk = max( 1, spp / 2 );
	}

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

	std::vector<int> nSpp, nChunks, extraMap;
	std::vector<float> meanL, m2;
	std::vector<PtGpuBakeLuxel> extraJobs;
	if ( bAdaptiveLm )
	{
		nSpp.assign( (size_t)kBatch, 0 );
		nChunks.assign( (size_t)kBatch, 0 );
		meanL.assign( (size_t)kBatch, 0.0f );
		m2.assign( (size_t)kBatch, 0.0f );
		extraJobs.reserve( (size_t)kBatch / 4 + 8 );
		extraMap.reserve( (size_t)kBatch / 4 + 8 );
	}

	bool ok = true;
	for ( unsigned base = 0; base < nSamples && ok && !g_bInterrupt; base += (unsigned)kBatch )
	{
		const unsigned n = min( (unsigned)kBatch, nSamples - base );
		for ( unsigned i = 0; i < n; ++i )
		{
			temp[i].radiance[0] = temp[i].radiance[1] = temp[i].radiance[2] = 0.0f;
			temp[i].sunAmt = 0.0f;
			if ( bAdaptiveLm )
			{
				nSpp[i] = 0;
				nChunks[i] = 0;
				meanL[i] = 0.0f;
				m2[i] = 0.0f;
			}
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
				if ( bAdaptiveLm )
				{
					nSpp[i] += nPass;
					const Vector rgb( pass[i].radiance[0], pass[i].radiance[1], pass[i].radiance[2] );
					PtWelfordChunk( PtLuma( rgb ), meanL[i], m2[i], nChunks[i] );
				}
			}
			{
				const double sppFrac = (double)( so + nPass ) / (double)spp;
				const unsigned approxDone = base + (unsigned)( (double)n * sppFrac );
				reportProp( min( approxDone, nSamples ), false );
			}
		}

		if ( ok && bAdaptiveLm )
		{
			int off = spp;
			while ( off < sppCap && ok && !g_bInterrupt )
			{
				extraJobs.clear();
				extraMap.clear();
				const int probe = min( sppChunk, sppCap - off );
				for ( unsigned i = 0; i < n; ++i )
				{
					if ( PtLuxelNeedsMoreSpp( meanL[i], m2[i], nChunks[i], sppChunk, nSpp[i], sppCap ) )
					{
						extraMap.push_back( (int)i );
						extraJobs.push_back( samples[base + i] );
					}
				}
				if ( extraJobs.empty() )
					break;

				PathTraceDXR_GpuBakeConfigurePass( (uint32_t)probe, (uint32_t)off );
				if ( !PathTraceDXR_GpuBakeLuxels( extraJobs.data(), (uint32_t)extraJobs.size(), pass.data() ) )
				{
					ok = false;
					break;
				}
				const float w = (float)probe;
				for ( size_t j = 0; j < extraJobs.size(); ++j )
				{
					const int i = extraMap[j];
					temp[i].radiance[0] += pass[j].radiance[0] * w;
					temp[i].radiance[1] += pass[j].radiance[1] * w;
					temp[i].radiance[2] += pass[j].radiance[2] * w;
					temp[i].sunAmt += pass[j].sunAmt * w;
					nSpp[i] += probe;
					const Vector rgb( pass[j].radiance[0], pass[j].radiance[1], pass[j].radiance[2] );
					PtWelfordChunk( PtLuma( rgb ), meanL[i], m2[i], nChunks[i] );
				}
				off += probe;
			}
		}

		for ( unsigned i = 0; i < n; ++i )
		{
			const int taken = bAdaptiveLm ? max( nSpp[i], 1 ) : spp;
			const float inv = 1.0f / (float)taken;
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
