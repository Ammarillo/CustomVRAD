//========= Copyright PathRAD, All rights reserved. ============//
//
// fog_volume - bake a 3D light grid into the BSP, rewrite the entity to lua_run.
//
//=============================================================================//

#include "vrad.h"
#include "fog_volume.h"
#include "vtf/vtf.h"
#include "bitmap/imageformat.h"
#include "tier1/utlbuffer.h"
#include "tier1/checksum_crc.h"
#include "filesystem_tools.h"
#include "pathtrace_dxr.h"
#include "pathtrace_dxr_device.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

// vismat.cpp implements the recursive form; header declares a 1-arg overload that is unused.
dleaf_t *PointInLeaf( int iNode, Vector const &point );

namespace
{

struct FogVolume_t
{
	entity_t *pEntity;
	Vector	mins;
	Vector	maxs;
	Vector	fogColor;		// 0..1
	float	density;
	float	lightBoost;		// multiplier on baked lighting (direct light visibility)
	float	anisotropy;
	float	gridSpacing;
	int		stepCount;
	float	blendDistance;
	int		blendMode;
	float	noiseScale;		// 0 = off
	float	noiseCoverage;	// 0..1
	float	windSpeed;
	float	windYaw;		// degrees
	int		priority;
	int		hammerId;
	bool	bEnabled;		// bake this volume
	bool	bStartEnabled;	// runtime cvf_en default
	char	materialPath[160];	// materials/ relative, no extension
	// Convex brush planes (Source: solid is Dot(n,p) <= dist). Empty → AABB fallback.
	Vector	planeN[24];
	float	planeD[24];
	int		nPlanes;
};

struct FogBlocker_t
{
	entity_t *pEntity;
	Vector	mins;
	Vector	maxs;
	float	blendDistance;
	bool	bEnabled;
	Vector	planeN[24];
	float	planeD[24];
	int		nPlanes;
};

FogVolume_t s_Volumes[FOG_VOLUME_MAX];
int s_nVolumes = 0;
FogBlocker_t s_Blockers[FOG_BLOCKER_MAX];
int s_nBlockers = 0;

static const int kMaxGridAxis = 64;
static const int kMaxGridCells = kMaxGridAxis * kMaxGridAxis * kMaxGridAxis;

// Shared tileable 3D noise atlas (Z-sliced 2D VTF). Sampled as $texture2.
static const int kFogNoiseRes = 32; // power-of-two, multiple of 4
static bool s_bNoiseAtlasPacked = false;
static char s_NoiseMaterialPath[160]; // materials/ relative, no extension

static void GetMapBaseName( char *out, int outSize )
{
	Q_FileBase( source, out, outSize );
	if ( !out[0] )
		Q_strncpy( out, "unknown", outSize );
}

static void ParseColor255( entity_t *e, const char *key, Vector &out, float defaultR, float defaultG, float defaultB )
{
	out.x = defaultR;
	out.y = defaultG;
	out.z = defaultB;
	const char *p = ValueForKey( e, (char *)key );
	if ( !p || !p[0] )
		return;
	float r = defaultR * 255.0f, g = defaultG * 255.0f, b = defaultB * 255.0f;
	sscanf( p, "%f %f %f", &r, &g, &b );
	out.x = clamp( r / 255.0f, 0.0f, 1.0f );
	out.y = clamp( g / 255.0f, 0.0f, 1.0f );
	out.z = clamp( b / 255.0f, 0.0f, 1.0f );
}

static int RoundUpPow2Clamped( int v, int minV, int maxV )
{
	if ( v < minV ) v = minV;
	if ( v > maxV ) v = maxV;
	int p = 1;
	while ( p < v )
		p <<= 1;
	if ( p > maxV ) p = maxV;
	// VTF Init requires multiples of 4.
	if ( p < 4 ) p = 4;
	return p;
}

static void ComputeGridDims( const Vector &mins, const Vector &maxs, float spacing,
							 int &nx, int &ny, int &nz )
{
	if ( spacing < 1.0f )
		spacing = 1.0f;
	Vector size = maxs - mins;
	nx = clamp( (int)ceilf( max( size.x, 1.0f ) / spacing ) + 1, 2, kMaxGridAxis );
	ny = clamp( (int)ceilf( max( size.y, 1.0f ) / spacing ) + 1, 2, kMaxGridAxis );
	nz = clamp( (int)ceilf( max( size.z, 1.0f ) / spacing ) + 1, 2, kMaxGridAxis );

	// Volume VTFs need power-of-two dims that are multiples of 4.
	nx = RoundUpPow2Clamped( nx, 4, kMaxGridAxis );
	ny = RoundUpPow2Clamped( ny, 4, kMaxGridAxis );
	nz = RoundUpPow2Clamped( nz, 4, kMaxGridAxis );

	while ( (int64)nx * ny * nz > kMaxGridCells )
	{
		if ( nx >= ny && nx >= nz && nx > 4 ) nx >>= 1;
		else if ( ny >= nz && ny > 4 ) ny >>= 1;
		else if ( nz > 4 ) nz >>= 1;
		else break;
	}
}

// Encode brush signed distance into atlas alpha (0..1). Positive dist = inside.
// Range must match kFogSdfRange in pathrad_fog_ps30.hlsl.
static const float kFogSdfRange = 1024.0f;

static void AddAABBPlanes( const Vector &mins, const Vector &maxs,
						   Vector *planeN, float *planeD, int *nPlanes, int maxPlanes )
{
	*nPlanes = 0;
	struct { float nx, ny, nz, d; } addList[6] = {
		{ 1, 0, 0, maxs.x },
		{ -1, 0, 0, -mins.x },
		{ 0, 1, 0, maxs.y },
		{ 0, -1, 0, -mins.y },
		{ 0, 0, 1, maxs.z },
		{ 0, 0, -1, -mins.z },
	};
	for ( int i = 0; i < 6 && *nPlanes < maxPlanes; ++i )
	{
		planeN[*nPlanes].Init( addList[i].nx, addList[i].ny, addList[i].nz );
		planeD[*nPlanes] = addList[i].d;
		++( *nPlanes );
	}
}

// Collect unique outward planes from a brush submodel (supports vertex-edited shapes).
static int ExtractBrushPlanes( int modelIndex, Vector *planeN, float *planeD, int maxPlanes,
							   const Vector &mins, const Vector &maxs )
{
	int n = 0;
	if ( modelIndex > 0 && modelIndex < nummodels && maxPlanes > 0 )
	{
		const dmodel_t &m = dmodels[modelIndex];
		for ( int fi = 0; fi < m.numfaces && n < maxPlanes; ++fi )
		{
			const int faceIndex = m.firstface + fi;
			if ( faceIndex < 0 || faceIndex >= numfaces )
				continue;
			const dface_t &f = dfaces[faceIndex];
			const int pi = f.planenum & ~1; // canonical orientation
			if ( pi < 0 || pi >= numplanes )
				continue;
			dplane_t pl = dplanes[f.planenum];
			// Skip near-degenerate / tiny faces.
			if ( pl.normal.LengthSqr() < 0.5f )
				continue;

			bool dup = false;
			for ( int j = 0; j < n; ++j )
			{
				if ( DotProduct( pl.normal, planeN[j] ) > 0.999f &&
					 fabsf( pl.dist - planeD[j] ) < 0.25f )
				{
					dup = true;
					break;
				}
			}
			if ( dup )
				continue;
			planeN[n] = pl.normal;
			planeD[n] = pl.dist;
			++n;
		}
	}

	if ( n < 4 )
	{
		AddAABBPlanes( mins, maxs, planeN, planeD, &n, maxPlanes );
		return n;
	}
	return n;
}

// Signed distance to convex brush: >0 inside, <0 outside (matches AABB helper).
static float SignedInsidePlanes( const Vector &p, const Vector *planeN, const float *planeD, int nPlanes )
{
	if ( nPlanes <= 0 )
		return -kFogSdfRange;

	float minInside = 1e20f;
	Vector outside( 0, 0, 0 );
	bool anyOut = false;
	for ( int i = 0; i < nPlanes; ++i )
	{
		// Positive when behind plane (inside halfspace).
		const float s = planeD[i] - DotProduct( planeN[i], p );
		if ( s >= 0.0f )
			minInside = min( minInside, s );
		else
		{
			anyOut = true;
			outside -= planeN[i] * s; // s<0 → accumulate outward displacement
		}
	}
	if ( !anyOut )
		return minInside;
	return -outside.Length();
}

static float EncodeFogSdfAlpha( float signedDist, float blockerAllow )
{
	// Pull toward deep-outside when blockers carve fog.
	const float a = clamp( blockerAllow, 0.0f, 1.0f );
	const float d = -kFogSdfRange + ( signedDist + kFogSdfRange ) * a;
	const float enc = 0.5f + 0.5f * clamp( d / kFogSdfRange, -1.0f, 1.0f );
	return enc;
}

static int LeafIndexFromPoint( const Vector &pos )
{
	dleaf_t *leaf = ::PointInLeaf( 0, pos );
	if ( !leaf )
		return 0;
	return (int)( leaf - dleafs );
}

static void SampleLeafAmbientAverage( const Vector &pos, Vector &outColor )
{
	outColor.Init();
	if ( !g_pLeafAmbientIndex || !g_pLeafAmbientLighting || numleafs <= 0 )
		return;

	int leafIndex = LeafIndexFromPoint( pos );
	if ( leafIndex < 0 || leafIndex >= numleafs )
		return;

	dleafambientindex_t idx = g_pLeafAmbientIndex->Element( leafIndex );
	// Solid / empty leaves store neighbor leaf in firstAmbientSample when count==0.
	if ( idx.ambientSampleCount == 0 )
	{
		int ref = idx.firstAmbientSample;
		if ( ref < 0 || ref >= numleafs )
			return;
		idx = g_pLeafAmbientIndex->Element( ref );
		leafIndex = ref;
		if ( idx.ambientSampleCount == 0 )
			return;
	}

	const dleaf_t &leaf = dleafs[leafIndex];
	float total = 0.0f;
	Vector acc( 0, 0, 0 );
	for ( int i = 0; i < idx.ambientSampleCount; ++i )
	{
		int sampleIndex = idx.firstAmbientSample + i;
		if ( sampleIndex < 0 || sampleIndex >= g_pLeafAmbientLighting->Count() )
			continue;
		const dleafambientlighting_t &s = g_pLeafAmbientLighting->Element( sampleIndex );
		Vector samplePos;
		samplePos.x = leaf.mins[0] + ( leaf.maxs[0] - leaf.mins[0] ) * ( s.x / 255.0f );
		samplePos.y = leaf.mins[1] + ( leaf.maxs[1] - leaf.mins[1] ) * ( s.y / 255.0f );
		samplePos.z = leaf.mins[2] + ( leaf.maxs[2] - leaf.mins[2] ) * ( s.z / 255.0f );
		float distSqr = ( samplePos - pos ).LengthSqr();
		float w = 1.0f / ( distSqr + 1.0f );
		Vector cubeAvg( 0, 0, 0 );
		for ( int side = 0; side < 6; ++side )
		{
			Vector c;
			ColorRGBExp32ToVector( s.cube.m_Color[side], c );
			cubeAvg += c;
		}
		cubeAvg *= ( 1.0f / 6.0f );
		acc += cubeAvg * w;
		total += w;
	}
	if ( total > 0.0f )
		outColor = acc * ( 1.0f / total );
}

static const Vector s_FogAxes[6] =
{
	Vector(  1,  0,  0 ),
	Vector( -1,  0,  0 ),
	Vector(  0,  1,  0 ),
	Vector(  0, -1,  0 ),
	Vector(  0,  0,  1 ),
	Vector(  0,  0, -1 ),
};

static void PackFogBakeLuxel( PtGpuBakeLuxel &job, const Vector &pos, const Vector &nrm, int cellIndex )
{
	memset( &job, 0, sizeof( job ) );
	job.pos[0] = pos.x;
	job.pos[1] = pos.y;
	job.pos[2] = pos.z;
	job.luxelWorld = 2.0f;
	Vector n = nrm;
	if ( n.LengthSqr() < 1e-8f )
		n.Init( 0, 0, 1 );
	else
		n.NormalizeInPlace();
	job.normal[0] = n.x;
	job.normal[1] = n.y;
	job.normal[2] = n.z;
	unsigned int h = (unsigned int)cellIndex * 747796405u;
	h ^= (unsigned int)( pos.x * 12.9898f + pos.y * 78.233f + pos.z * 37.719f );
	h *= 1597334677u;
	job.seed = h ^ (unsigned int)( n.x * 1000.0f + n.y * 2000.0f + n.z * 3000.0f );
	job.faceNum = -1;
	job.aaN = 1;
	job.skipPropIndex = -1;
}

static Vector FogGridSamplePos( const FogVolume_t &v, int x, int y, int z, int nx, int ny, int nz )
{
	Vector pos;
	pos.x = ( nx == 1 ) ? ( v.mins.x + v.maxs.x ) * 0.5f
						: v.mins.x + ( v.maxs.x - v.mins.x ) * ( x / (float)( nx - 1 ) );
	pos.y = ( ny == 1 ) ? ( v.mins.y + v.maxs.y ) * 0.5f
						: v.mins.y + ( v.maxs.y - v.mins.y ) * ( y / (float)( ny - 1 ) );
	pos.z = ( nz == 1 ) ? ( v.mins.z + v.maxs.z ) * 0.5f
						: v.mins.z + ( v.maxs.z - v.mins.z ) * ( z / (float)( nz - 1 ) );

	// Pull samples off walls/floor so pathtrace doesn't bake dark contact shadow into fog.
	const float bias = 4.0f;
	Vector center = ( v.mins + v.maxs ) * 0.5f;
	Vector toCenter = center - pos;
	float len = toCenter.Length();
	if ( len > 1e-3f )
	{
		Vector n = toCenter * ( 1.0f / len );
		Vector candidate = pos + n * bias;
		candidate.x = clamp( candidate.x, v.mins.x + 1.0f, v.maxs.x - 1.0f );
		candidate.y = clamp( candidate.y, v.mins.y + 1.0f, v.maxs.y - 1.0f );
		candidate.z = clamp( candidate.z, v.mins.z + 1.0f, v.maxs.z - 1.0f );
		if ( !PositionInSolid( candidate ) )
			pos = candidate;
	}
	return pos;
}

// Positive = distance inside blocker (to nearest face). Negative = outside.
static float SignedInsideBlocker( const Vector &p, const FogBlocker_t &b )
{
	if ( b.nPlanes > 0 )
		return SignedInsidePlanes( p, b.planeN, b.planeD, b.nPlanes );

	const Vector &bmin = b.mins;
	const Vector &bmax = b.maxs;
	const float dx = min( p.x - bmin.x, bmax.x - p.x );
	const float dy = min( p.y - bmin.y, bmax.y - p.y );
	const float dz = min( p.z - bmin.z, bmax.z - p.z );
	if ( dx >= 0.0f && dy >= 0.0f && dz >= 0.0f )
		return min( dx, min( dy, dz ) );

	Vector o;
	o.x = max( bmin.x - p.x, p.x - bmax.x );
	o.y = max( bmin.y - p.y, p.y - bmax.y );
	o.z = max( bmin.z - p.z, p.z - bmax.z );
	o.x = max( o.x, 0.0f );
	o.y = max( o.y, 0.0f );
	o.z = max( o.z, 0.0f );
	return -o.Length();
}

// 1 = full fog allowed, 0 = fully blocked. Soft edge via blocker BlendDistance.
static float FogAllowAtPoint( const Vector &pos )
{
	float allow = 1.0f;
	for ( int i = 0; i < s_nBlockers; ++i )
	{
		const FogBlocker_t &b = s_Blockers[i];
		if ( !b.bEnabled )
			continue;
		const float inside = SignedInsideBlocker( pos, b );
		if ( inside < 0.0f )
			continue;
		if ( b.blendDistance <= 1e-3f )
		{
			allow = 0.0f;
			break;
		}
		// 0 on face, 1 deep inside → kill fog deeper in.
		const float u = clamp( inside / b.blendDistance, 0.0f, 1.0f );
		allow *= ( 1.0f - u * u );
		if ( allow <= 1e-4f )
		{
			allow = 0.0f;
			break;
		}
	}
	return allow;
}

// Atlas alpha = encoded brush SDF (convex planes) * blocker allow.
// Shader decodes signed distance for Lengyel soft edges on non-box brushes.
static void BuildFogAllowMask( const FogVolume_t &v, int nx, int ny, int nz, unsigned char *pAllow )
{
	int carved = 0;
	int outsideBrush = 0;
	for ( int z = 0; z < nz; ++z )
	{
		for ( int y = 0; y < ny; ++y )
		{
			for ( int x = 0; x < nx; ++x )
			{
				Vector pos = FogGridSamplePos( v, x, y, z, nx, ny, nz );
				float brushDist;
				if ( v.nPlanes > 0 )
					brushDist = SignedInsidePlanes( pos, v.planeN, v.planeD, v.nPlanes );
				else
				{
					// AABB fallback (same convention as shader SignedBlendDistance).
					const float dx = min( pos.x - v.mins.x, v.maxs.x - pos.x );
					const float dy = min( pos.y - v.mins.y, v.maxs.y - pos.y );
					const float dz = min( pos.z - v.mins.z, v.maxs.z - pos.z );
					if ( dx >= 0.0f && dy >= 0.0f && dz >= 0.0f )
						brushDist = min( dx, min( dy, dz ) );
					else
					{
						Vector o;
						o.x = max( v.mins.x - pos.x, pos.x - v.maxs.x );
						o.y = max( v.mins.y - pos.y, pos.y - v.maxs.y );
						o.z = max( v.mins.z - pos.z, pos.z - v.maxs.z );
						o.x = max( o.x, 0.0f );
						o.y = max( o.y, 0.0f );
						o.z = max( o.z, 0.0f );
						brushDist = -o.Length();
					}
				}
				const float allow = FogAllowAtPoint( pos );
				const float enc = EncodeFogSdfAlpha( brushDist, allow );
				const int index = x + y * nx + z * nx * ny;
				pAllow[index] = (unsigned char)clamp( (int)( enc * 255.0f + 0.5f ), 0, 255 );
				if ( brushDist < 0.0f )
					++outsideBrush;
				if ( allow < 0.999f )
					++carved;
			}
		}
	}
	if ( outsideBrush > 0 || carved > 0 )
		Msg( "fog_volume %d: SDF alpha — %d/%d cells outside brush, %d affected by blockers (%d planes)\n",
			 v.hammerId, outsideBrush, nx * ny * nz, carved, v.nPlanes );
}

static void SampleClassicDirectAtPoint( Vector pos, Vector &outDirect )
{
	outDirect.Init();
	if ( PositionInSolid( pos ) )
		return;
	Vector n( 0, 0, 1 );
	ComputeDirectLightingAtPoint( pos, n, outDirect, 0, -1, GATHERLFLAGS_IGNORE_NORMALS );
}

static void SampleClassicLightingAtPoint( Vector pos, float directBoost, Vector &outColor )
{
	outColor.Init();
	if ( PositionInSolid( pos ) )
	{
		SampleLeafAmbientAverage( pos, outColor );
		return;
	}

	Vector direct( 0, 0, 0 );
	Vector indirect( 0, 0, 0 );
	Vector n( 0, 0, 1 );
	ComputeDirectLightingAtPoint( pos, n, direct, 0, -1, GATHERLFLAGS_IGNORE_NORMALS );
	ComputeIndirectLightingAtPoint( pos, n, indirect, 0, true, true );
	outColor = direct * directBoost + indirect;

	if ( outColor.LengthSqr() < 1e-10f )
		SampleLeafAmbientAverage( pos, outColor );
}

static bool BakeLightGridPathTrace( const FogVolume_t &v, int nx, int ny, int nz, ColorRGBExp32 *pSamples )
{
	const int count = nx * ny * nz;
	CUtlVector<Vector> cellPos;
	CUtlVector<int> cellIndex;
	cellPos.EnsureCapacity( count );
	cellIndex.EnsureCapacity( count );

	for ( int z = 0; z < nz; ++z )
	{
		for ( int y = 0; y < ny; ++y )
		{
			for ( int x = 0; x < nx; ++x )
			{
			Vector pos = FogGridSamplePos( v, x, y, z, nx, ny, nz );

				const int index = x + y * nx + z * nx * ny;
				if ( PositionInSolid( pos ) )
				{
					Vector color;
					SampleLeafAmbientAverage( pos, color );
					VectorToColorRGBExp32( color, pSamples[index] );
					continue;
				}
				cellPos.AddToTail( pos );
				cellIndex.AddToTail( index );
			}
		}
	}

	const int nCells = cellPos.Count();
	if ( nCells <= 0 )
		return true;

	const int nJobs = nCells * 6;
	CUtlVector<PtGpuBakeLuxel> jobs;
	CUtlVector<PtGpuBakeResult> results;
	jobs.SetCount( nJobs );
	results.SetCount( nJobs );

	for ( int i = 0; i < nCells; ++i )
	{
		for ( int a = 0; a < 6; ++a )
			PackFogBakeLuxel( jobs[i * 6 + a], cellPos[i], s_FogAxes[a], cellIndex[i] * 6 + a );
	}

	Msg( "fog_volume %d: pathtrace lighting %d cells (%d axis samples)...\n",
		 v.hammerId, nCells, nJobs );

	if ( !PathTraceDXR_BakePropSamples( jobs.Base(), (unsigned)nJobs, results.Base(),
										true /* lightmap quality */, true ) )
	{
		Warning( "fog_volume: PathTraceDXR_BakePropSamples failed — classic fallback\n" );
		return false;
	}

	// LightBoost scales DIRECT light only. The GPU result mixes direct and
	// indirect, so run a second 0-bounce GPU pass (direct incl. $vrad_filter
	// glass transmission) and add (boost-1)x that on top.
	const bool bBoostDirect = fabsf( v.lightBoost - 1.0f ) > 1e-3f;
	CUtlVector<PtGpuBakeResult> directResults;
	bool bHaveGpuDirect = false;
	if ( bBoostDirect )
	{
		directResults.SetCount( nJobs );
		bHaveGpuDirect = PathTraceDXR_BakeDirectOnlySamples( jobs.Base(), (unsigned)nJobs,
															 directResults.Base(), 64 );
		if ( !bHaveGpuDirect )
			Warning( "fog_volume %d: direct-only GPU pass failed — LightBoost uses CPU direct (no glass filter)\n",
					 v.hammerId );
	}

	for ( int i = 0; i < nCells; ++i )
	{
		Vector acc( 0, 0, 0 );
		for ( int a = 0; a < 6; ++a )
		{
			const PtGpuBakeResult &r = results[i * 6 + a];
			acc.x += r.radiance[0];
			acc.y += r.radiance[1];
			acc.z += r.radiance[2];
		}
		acc *= ( 1.0f / 6.0f );

		if ( bBoostDirect )
		{
			Vector direct( 0, 0, 0 );
			if ( bHaveGpuDirect )
			{
				for ( int a = 0; a < 6; ++a )
				{
					const PtGpuBakeResult &r = directResults[i * 6 + a];
					direct.x += r.radiance[0];
					direct.y += r.radiance[1];
					direct.z += r.radiance[2];
				}
				direct *= ( 1.0f / 6.0f );
			}
			else
			{
				SampleClassicDirectAtPoint( cellPos[i], direct );
			}
			acc += direct * ( v.lightBoost - 1.0f );
			acc.x = max( acc.x, 0.0f );
			acc.y = max( acc.y, 0.0f );
			acc.z = max( acc.z, 0.0f );
		}

		// Bake FogColor into the atlas so the shader does not hardcode tint.
		acc.x *= v.fogColor.x;
		acc.y *= v.fogColor.y;
		acc.z *= v.fogColor.z;
		VectorToColorRGBExp32( acc, pSamples[cellIndex[i]] );
	}
	return true;
}

static void BakeLightGridClassic( const FogVolume_t &v, int nx, int ny, int nz, ColorRGBExp32 *pSamples )
{
	Msg( "fog_volume %d: classic direct+indirect lighting...\n", v.hammerId );
	for ( int z = 0; z < nz; ++z )
	{
		for ( int y = 0; y < ny; ++y )
		{
			for ( int x = 0; x < nx; ++x )
			{
				Vector pos = FogGridSamplePos( v, x, y, z, nx, ny, nz );

				Vector color;
				SampleClassicLightingAtPoint( pos, v.lightBoost, color );
				color.x *= v.fogColor.x;
				color.y *= v.fogColor.y;
				color.z *= v.fogColor.z;
				const int index = x + y * nx + z * nx * ny;
				VectorToColorRGBExp32( color, pSamples[index] );
			}
		}
	}
}

// Replace near-black / in-solid grid cells with averages of lit neighbors so
// bilinear sampling near walls does not pull fog to black.
static void DilateFogLightGrid( ColorRGBExp32 *pSamples, int nx, int ny, int nz,
								const FogVolume_t &v, int passes = 6 )
{
	const int count = nx * ny * nz;
	CUtlVector<Vector> colors;
	CUtlVector<char> valid;
	colors.SetCount( count );
	valid.SetCount( count );

	float lumSum = 0.0f;
	int lumCount = 0;
	for ( int i = 0; i < count; ++i )
	{
		ColorRGBExp32ToVector( pSamples[i], colors[i] );
		const float lum = 0.2126f * colors[i].x + 0.7152f * colors[i].y + 0.0722f * colors[i].z;
		const int x = i % nx;
		const int y = ( i / nx ) % ny;
		const int z = i / ( nx * ny );
		Vector pos = FogGridSamplePos( v, x, y, z, nx, ny, nz );
		const bool solid = PositionInSolid( pos );
		valid[i] = ( !solid && lum > 1e-4f ) ? 1 : 0;
		if ( valid[i] )
		{
			lumSum += lum;
			++lumCount;
		}
	}

	const float meanLum = ( lumCount > 0 ) ? ( lumSum / (float)lumCount ) : 0.0f;
	const float darkThresh = max( 1e-4f, meanLum * 0.05f );
	for ( int i = 0; i < count; ++i )
	{
		const float lum = 0.2126f * colors[i].x + 0.7152f * colors[i].y + 0.0722f * colors[i].z;
		if ( lum < darkThresh )
			valid[i] = 0;
	}

	static const int kOff[6][3] = {
		{ 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }
	};

	int filled = 0;
	for ( int pass = 0; pass < passes; ++pass )
	{
		CUtlVector<Vector> next;
		CUtlVector<char> nextValid;
		next.SetCount( count );
		nextValid.SetCount( count );
		for ( int i = 0; i < count; ++i )
		{
			next[i] = colors[i];
			nextValid[i] = valid[i];
		}
		int passFilled = 0;
		for ( int z = 0; z < nz; ++z )
		{
			for ( int y = 0; y < ny; ++y )
			{
				for ( int x = 0; x < nx; ++x )
				{
					const int i = x + y * nx + z * nx * ny;
					if ( valid[i] )
						continue;

					Vector acc( 0, 0, 0 );
					int n = 0;
					for ( int k = 0; k < 6; ++k )
					{
						const int xx = x + kOff[k][0];
						const int yy = y + kOff[k][1];
						const int zz = z + kOff[k][2];
						if ( xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz )
							continue;
						const int j = xx + yy * nx + zz * nx * ny;
						if ( !valid[j] )
							continue;
						acc += colors[j];
						++n;
					}
					if ( n <= 0 )
						continue;
					next[i] = acc * ( 1.0f / (float)n );
					nextValid[i] = 1;
					++passFilled;
				}
			}
		}
		for ( int i = 0; i < count; ++i )
		{
			colors[i] = next[i];
			valid[i] = nextValid[i];
		}
		filled += passFilled;
		if ( passFilled == 0 )
			break;
	}

	for ( int i = 0; i < count; ++i )
		VectorToColorRGBExp32( colors[i], pSamples[i] );

	Msg( "fog_volume %d: dilated %d dark/solid light-grid cell(s) (%d lit seed(s))\n",
		 v.hammerId, filled, lumCount );
}

static bool WriteLightGridAtlasVTF( const char *pakPath, int nx, int ny, int nz,
									int atlasW, int atlasH, const ColorRGBExp32 *pSamples,
									const unsigned char *pAllow )
{
	IVTFTexture *pTex = CreateVTFTexture();
	if ( !pTex )
		return false;

	// Source/GMod volume VTFs are unreliable here — pack Z slices into a 2D atlas.
	// Alpha = encoded brush signed distance (and blocker carve). See EncodeFogSdfAlpha.
	const int flags = TEXTUREFLAGS_NOMIP | TEXTUREFLAGS_NOLOD | TEXTUREFLAGS_TRILINEAR |
					  TEXTUREFLAGS_CLAMPS | TEXTUREFLAGS_CLAMPT | TEXTUREFLAGS_EIGHTBITALPHA;
	if ( !pTex->Init( atlasW, atlasH, 1, IMAGE_FORMAT_RGBA8888, flags, 1 ) )
	{
		Warning( "fog_volume: failed to Init atlas VTF %dx%d (grid %dx%dx%d)\n", atlasW, atlasH, nx, ny, nz );
		DestroyVTFTexture( pTex );
		return false;
	}

	unsigned char *pDst = pTex->ImageData( 0, 0, 0 );
	if ( !pDst )
	{
		DestroyVTFTexture( pTex );
		return false;
	}

	memset( pDst, 0, (size_t)atlasW * (size_t)atlasH * 4u );
	for ( int z = 0; z < nz; ++z )
	{
		for ( int y = 0; y < ny; ++y )
		{
			for ( int x = 0; x < nx; ++x )
			{
				const int src = x + y * nx + z * nx * ny;
				const int dx = x;
				const int dy = y + z * ny;
				if ( dx >= atlasW || dy >= atlasH )
					continue;
				unsigned char rgba[4];
				ConvertRGBExp32ToRGBA8888( &pSamples[src], rgba );
				rgba[3] = pAllow ? pAllow[src] : 255;
				unsigned char *p = pDst + ( dy * atlasW + dx ) * 4;
				p[0] = rgba[0];
				p[1] = rgba[1];
				p[2] = rgba[2];
				p[3] = rgba[3];
			}
		}
		// Pad atlas borders within this Z slice so bilinear filtering never
		// samples cleared (black) texels outside the grid.
		for ( int y = 0; y < ny; ++y )
		{
			const int dy = y + z * ny;
			if ( dy >= atlasH )
				continue;
			unsigned char *row = pDst + dy * atlasW * 4;
			if ( nx > 0 && nx < atlasW )
			{
				unsigned char *edge = row + ( nx - 1 ) * 4;
				for ( int x = nx; x < atlasW; ++x )
				{
					unsigned char *p = row + x * 4;
					p[0] = edge[0]; p[1] = edge[1]; p[2] = edge[2]; p[3] = edge[3];
				}
			}
		}
	}

	CUtlBuffer buf;
	if ( !pTex->Serialize( buf ) )
	{
		Warning( "fog_volume: failed to serialize atlas VTF\n" );
		DestroyVTFTexture( pTex );
		return false;
	}
	DestroyVTFTexture( pTex );

	AddBufferToPak( GetPakFile(), pakPath, buf.Base(), buf.TellPut(), false );
	return true;
}

// ---- Tileable 3D FBM bake (period = kFogNoiseRes in cell space) ----

static float FogNoiseHash3( int x, int y, int z )
{
	const int n = kFogNoiseRes;
	x = ( ( x % n ) + n ) % n;
	y = ( ( y % n ) + n ) % n;
	z = ( ( z % n ) + n ) % n;
	unsigned int h = (unsigned int)( x * 374761393u + y * 668265263u + z * 2147483647u );
	h = ( h ^ ( h >> 13 ) ) * 1274126177u;
	h ^= h >> 16;
	return ( h & 0x00FFFFFFu ) / 16777215.0f;
}

static float FogNoiseSmoothstep( float t )
{
	return t * t * ( 3.0f - 2.0f * t );
}

static float FogValueNoisePeriodic( float px, float py, float pz )
{
	const float ix = floorf( px );
	const float iy = floorf( py );
	const float iz = floorf( pz );
	float fx = px - ix;
	float fy = py - iy;
	float fz = pz - iz;
	fx = FogNoiseSmoothstep( fx );
	fy = FogNoiseSmoothstep( fy );
	fz = FogNoiseSmoothstep( fz );
	const int x0 = (int)ix, y0 = (int)iy, z0 = (int)iz;
	const float n000 = FogNoiseHash3( x0,     y0,     z0 );
	const float n100 = FogNoiseHash3( x0 + 1, y0,     z0 );
	const float n010 = FogNoiseHash3( x0,     y0 + 1, z0 );
	const float n110 = FogNoiseHash3( x0 + 1, y0 + 1, z0 );
	const float n001 = FogNoiseHash3( x0,     y0,     z0 + 1 );
	const float n101 = FogNoiseHash3( x0 + 1, y0,     z0 + 1 );
	const float n011 = FogNoiseHash3( x0,     y0 + 1, z0 + 1 );
	const float n111 = FogNoiseHash3( x0 + 1, y0 + 1, z0 + 1 );
	const float nx00 = n000 + ( n100 - n000 ) * fx;
	const float nx10 = n010 + ( n110 - n010 ) * fx;
	const float nx01 = n001 + ( n101 - n001 ) * fx;
	const float nx11 = n011 + ( n111 - n011 ) * fx;
	const float nxy0 = nx00 + ( nx10 - nx00 ) * fy;
	const float nxy1 = nx01 + ( nx11 - nx01 ) * fy;
	return nxy0 + ( nxy1 - nxy0 ) * fz;
}

// Tileable FBM: integer octave frequencies so the tile repeats without an edge.
static float FogFbmPeriodic( float px, float py, float pz )
{
	float a = 0.0f;
	float w = 0.5f;
	float freq = 1.0f;
	for ( int o = 0; o < 4; ++o )
	{
		a += w * FogValueNoisePeriodic( px * freq, py * freq, pz * freq );
		freq *= 2.0f;
		w *= 0.5f;
	}
	return a;
}

// Pack shared 32^3 noise as Z-slice atlas (same layout as light grid). R=G=B=noise, A=255.
static bool EnsureFogNoiseAtlasPacked()
{
	if ( s_bNoiseAtlasPacked )
		return s_NoiseMaterialPath[0] != 0;

	char mapName[64];
	GetMapBaseName( mapName, sizeof( mapName ) );
	Q_snprintf( s_NoiseMaterialPath, sizeof( s_NoiseMaterialPath ), "maps/%s/fog_noise", mapName );

	const int n = kFogNoiseRes;
	const int atlasW = n;
	const int atlasH = n * n; // ny * nz with ny=nz=n

	IVTFTexture *pTex = CreateVTFTexture();
	if ( !pTex )
		return false;

	const int flags = TEXTUREFLAGS_NOMIP | TEXTUREFLAGS_NOLOD | TEXTUREFLAGS_TRILINEAR |
					  TEXTUREFLAGS_CLAMPS | TEXTUREFLAGS_CLAMPT;
	if ( !pTex->Init( atlasW, atlasH, 1, IMAGE_FORMAT_RGBA8888, flags, 1 ) )
	{
		Warning( "fog_volume: failed to Init noise atlas VTF %dx%d\n", atlasW, atlasH );
		DestroyVTFTexture( pTex );
		return false;
	}

	unsigned char *pDst = pTex->ImageData( 0, 0, 0 );
	if ( !pDst )
	{
		DestroyVTFTexture( pTex );
		return false;
	}

	memset( pDst, 0, (size_t)atlasW * (size_t)atlasH * 4u );
	for ( int z = 0; z < n; ++z )
	{
		for ( int y = 0; y < n; ++y )
		{
			for ( int x = 0; x < n; ++x )
			{
				// Cell centers in [0, n) — matches shader frac(world*scale) * n sampling.
				const float px = (float)x + 0.5f;
				const float py = (float)y + 0.5f;
				const float pz = (float)z + 0.5f;
				float v = FogFbmPeriodic( px, py, pz );
				v = clamp( v, 0.0f, 1.0f );
				const unsigned char b = (unsigned char)( v * 255.0f + 0.5f );
				const int dx = x;
				const int dy = y + z * n;
				unsigned char *p = pDst + ( dy * atlasW + dx ) * 4;
				p[0] = b;
				p[1] = b;
				p[2] = b;
				p[3] = 255;
			}
		}
	}

	CUtlBuffer buf;
	if ( !pTex->Serialize( buf ) )
	{
		Warning( "fog_volume: failed to serialize noise atlas VTF\n" );
		DestroyVTFTexture( pTex );
		return false;
	}
	DestroyVTFTexture( pTex );

	char vtfPak[256];
	Q_snprintf( vtfPak, sizeof( vtfPak ), "materials/%s.vtf", s_NoiseMaterialPath );
	AddBufferToPak( GetPakFile(), vtfPak, buf.Base(), buf.TellPut(), false );
	s_bNoiseAtlasPacked = true;
	Msg( "fog_volume: packed shared noise atlas %s (%d^3, atlas %dx%d)\n",
		 vtfPak, n, atlasW, atlasH );
	return true;
}

// Unique per-bake suffix for shader + material names. GMod caches:
//   1) IMaterial by path (Lua Material())
//   2) pixel shaders by $pixshader name
//   3) VCS by filename under garrysmod/shaders/fxc (often preferred over BSP pak)
// A timestamp-only id is not enough if the same second is reused or if a stale
// on-disk .vcs / Material() entry keeps the old bytecode. Mix VCS bytes + time.
// Must keep the _ps30/_vs30 postfix on the packed names.
static char s_ShaderId[16];

static bool FileExistsDisk( const char *path );
static bool FindShaderFile( const char *fileName, char *outPath, int outSize );

static unsigned FogVolume_HashFile( const char *diskPath )
{
	FILE *f = fopen( diskPath, "rb" );
	if ( !f )
		return 0;
	fseek( f, 0, SEEK_END );
	long sz = ftell( f );
	fseek( f, 0, SEEK_SET );
	if ( sz <= 0 )
	{
		fclose( f );
		return 0;
	}
	CUtlBuffer buf( 0, (int)sz, 0 );
	buf.EnsureCapacity( (int)sz );
	fread( buf.Base(), 1, sz, f );
	fclose( f );
	return (unsigned)CRC32_ProcessSingleBuffer( buf.Base(), (int)sz );
}

static const char *FogVolume_ShaderId()
{
	if ( s_ShaderId[0] )
		return s_ShaderId;

	unsigned h = (unsigned)time( NULL );
#ifdef _WIN32
	h ^= (unsigned)GetTickCount();
	LARGE_INTEGER qpc;
	if ( QueryPerformanceCounter( &qpc ) )
		h ^= (unsigned)qpc.LowPart ^ (unsigned)qpc.HighPart;
#endif

	char disk[MAX_PATH];
	const char *src[] = {
		"pathrad_fog_vs30.vcs",
		"pathrad_fog_ps30.vcs"
	};
	for ( int i = 0; i < 2; ++i )
	{
		if ( FindShaderFile( src[i], disk, sizeof( disk ) ) )
			h ^= FogVolume_HashFile( disk ) + (unsigned)( i * 0x9e3779b9 );
	}

	if ( !h )
		h = 0x6d5a4c3u;
	Q_snprintf( s_ShaderId, sizeof( s_ShaderId ), "%08x", h );
	return s_ShaderId;
}

static void StampFogMaterialPaths()
{
	char mapName[64];
	GetMapBaseName( mapName, sizeof( mapName ) );
	const char *id = FogVolume_ShaderId();
	for ( int i = 0; i < s_nVolumes; ++i )
	{
		Q_snprintf( s_Volumes[i].materialPath, sizeof( s_Volumes[i].materialPath ),
					"maps/%s/fog_volume_%d_%s", mapName, s_Volumes[i].hammerId, id );
	}
}

static int ActiveFogCount()
{
	int n = 0;
	for ( int i = 0; i < s_nVolumes; ++i )
	{
		if ( s_Volumes[i].bEnabled )
			++n;
	}
	return n;
}

static int FogGridLog2Minus2( int n )
{
	// nx/ny/nz are power-of-two in [4,64] after ComputeGridDims.
	int log = 0;
	int v = 4;
	while ( v < n && log < 4 )
	{
		v <<= 1;
		++log;
	}
	return log;
}

// Bit-pack for shader c3.w (see pathrad_fog_ps30.hlsl). Fits in float32 ints (<2^24).
// NoiseScale + WindSpeed are full floats (c2.z / c2.w) — not packed.
static int PackFogShaderConst( int nz, int blendMode, float coverage, float windYaw, float blendDist )
{
	const int lz = FogGridLog2Minus2( nz );
	const int mode = clamp( blendMode, 0, 2 );
	const int cov = clamp( (int)( coverage * 15.0f + 0.5f ), 0, 15 );
	float yaw = windYaw;
	while ( yaw < 0.0f )
		yaw += 360.0f;
	while ( yaw >= 360.0f )
		yaw -= 360.0f;
	const int yawQ = ( (int)( yaw / 45.0f + 0.5f ) ) & 7;
	const int dist = clamp( (int)( blendDist + 0.5f ), 0, 2047 );
	// bits 0..2 lz, 3..4 mode, 5..8 coverage, 9..11 yaw, 12..22 BlendDistance
	return lz | ( mode << 3 ) | ( cov << 5 ) | ( yawQ << 9 ) | ( dist << 12 );
}

// Light-grid AABB: brush ± BlendDistance for Outside/Center so the outer fade
// has real baked samples (no UV-clamped edge stretch).
static void FogLightGridBounds( const FogVolume_t &v, Vector &outMins, Vector &outMaxs )
{
	outMins = v.mins;
	outMaxs = v.maxs;
	if ( v.blendDistance > 1e-3f && v.blendMode >= 1 )
	{
		const Vector e( v.blendDistance, v.blendDistance, v.blendDistance );
		outMins -= e;
		outMaxs += e;
	}
}

static void WriteVMT( const FogVolume_t &v, int nx, int ny, int nz, int atlasW, int atlasH )
{
	char vmtPath[256];
	Q_snprintf( vmtPath, sizeof( vmtPath ), "materials/%s.vmt", v.materialPath );

	// c0 = eye.xyz, density (eye/frustum set each frame from Lua)
	// c1 = mins.xyz, maxs.z  (brush AABB — density fade)
	// c2 = maxs.xy, NoiseScale, CurTime()*WindSpeed
	// c3 = forward*thx, pack (incl. integer BlendDistance)
	const int shaderPack = PackFogShaderConst( nz, v.blendMode, v.noiseCoverage, v.windYaw, v.blendDistance );
	char body[2048];
	const char *noisePath = ( s_NoiseMaterialPath[0] ) ? s_NoiseMaterialPath : "maps/unknown/fog_noise";
	Q_snprintf( body, sizeof( body ),
		"\"screenspace_general\"\n"
		"{\n"
		"\t\"$pixshader\" \"%s_%s_ps30\"\n"
		"\t\"$vertexshader\" \"%s_%s_vs30\"\n"
		"\t\"$basetexture\" \"%s\"\n"
		"\t\"$texture1\" \"_rt_ResolvedFullFrameDepth\"\n"
		"\t\"$texture2\" \"%s\"\n"
		"\t\"$ignorez\" \"1\"\n"
		"\t\"$additive\" \"0\"\n"
		"\t\"$alphablend\" \"1\"\n"
		"\t\"$depthtest\" \"0\"\n"
		"\t\"$depthwrite\" \"0\"\n"
		"\t\"$cull\" \"0\"\n"
		"\t\"$linearwrite\" \"1\"\n"
		"\t\"$linearread_basetexture\" \"1\"\n"
		"\t\"$linearread_texture2\" \"1\"\n"
		"\t\"$copyalpha\" \"0\"\n"
		"\t\"$vertexcolor\" \"1\"\n"
		"\t\"$vertextransform\" \"0\"\n"
		"\t\"$x360appfix\" \"0\"\n"
		"\t\"$c0_x\" \"0\"\n"
		"\t\"$c0_y\" \"0\"\n"
		"\t\"$c0_z\" \"0\"\n"
		"\t\"$c0_w\" \"%.6f\"\n"
		"\t\"$c1_x\" \"%.4f\"\n"
		"\t\"$c1_y\" \"%.4f\"\n"
		"\t\"$c1_z\" \"%.4f\"\n"
		"\t\"$c1_w\" \"%.4f\"\n"
		"\t\"$c2_x\" \"%.4f\"\n"
		"\t\"$c2_y\" \"%.4f\"\n"
		"\t\"$c2_z\" \"%.6f\"\n"
		"\t\"$c2_w\" \"0\"\n"
		"\t\"$c3_x\" \"0.5\"\n"
		"\t\"$c3_y\" \"0\"\n"
		"\t\"$c3_z\" \"0\"\n"
		"\t\"$c3_w\" \"%d\"\n"
		"}\n",
		"pathrad_fog", FogVolume_ShaderId(), "pathrad_fog", FogVolume_ShaderId(),
		v.materialPath, noisePath,
		v.density,
		v.mins.x, v.mins.y, v.mins.z, v.maxs.z,
		v.maxs.x, v.maxs.y, v.noiseScale,
		shaderPack );
	(void)v.anisotropy; (void)v.stepCount; (void)v.fogColor;
	(void)nx; (void)ny;

	// Binary mode — avoid text transforms mangling the VMT.
	AddBufferToPak( GetPakFile(), vmtPath, body, (int)strlen( body ), false );
	(void)atlasW; (void)atlasH;
}

static bool FileExistsDisk( const char *path )
{
	FILE *f = fopen( path, "rb" );
	if ( !f )
		return false;
	fclose( f );
	return true;
}

static bool FindShaderFile( const char *fileName, char *outPath, int outSize )
{
	char candidate[MAX_PATH];

	// 1) Next to the running module (bin/ or game/bin/x64)
	char modulePath[MAX_PATH] = { 0 };
#ifdef _WIN32
	HMODULE hMod = NULL;
	GetModuleHandleExA( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
						(LPCSTR)&FindShaderFile, &hMod );
	if ( hMod )
		GetModuleFileNameA( hMod, modulePath, sizeof( modulePath ) );
#endif
	if ( modulePath[0] )
	{
		char dir[MAX_PATH];
		Q_strncpy( dir, modulePath, sizeof( dir ) );
		Q_StripFilename( dir );
		// PathRAD/bin -> PathRAD/shaders/fxc
		Q_snprintf( candidate, sizeof( candidate ), "%s\\..\\shaders\\fxc\\%s", dir, fileName );
		if ( FileExistsDisk( candidate ) )
		{
			Q_strncpy( outPath, candidate, outSize );
			return true;
		}
		// PathRAD/game/bin/x64 -> ../../shaders/fxc
		Q_snprintf( candidate, sizeof( candidate ), "%s\\..\\..\\..\\shaders\\fxc\\%s", dir, fileName );
		if ( FileExistsDisk( candidate ) )
		{
			Q_strncpy( outPath, candidate, outSize );
			return true;
		}
		Q_snprintf( candidate, sizeof( candidate ), "%s\\shaders\\fxc\\%s", dir, fileName );
		if ( FileExistsDisk( candidate ) )
		{
			Q_strncpy( outPath, candidate, outSize );
			return true;
		}
	}

	// 2) CWD
	Q_snprintf( candidate, sizeof( candidate ), "shaders\\fxc\\%s", fileName );
	if ( FileExistsDisk( candidate ) )
	{
		Q_strncpy( outPath, candidate, outSize );
		return true;
	}
	return false;
}

static void ClearOldGameShaders()
{
	if ( !gamedir[0] )
		return;
	char pattern[MAX_PATH];
	Q_snprintf( pattern, sizeof( pattern ), "%sshaders\\fxc\\pathrad_*.vcs", gamedir );
	WIN32_FIND_DATAA fd;
	HANDLE h = FindFirstFileA( pattern, &fd );
	if ( h == INVALID_HANDLE_VALUE )
		return;
	int n = 0;
	do
	{
		if ( fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY )
			continue;
		char del[MAX_PATH];
		Q_snprintf( del, sizeof( del ), "%sshaders\\fxc\\%s", gamedir, fd.cFileName );
		if ( DeleteFileA( del ) )
			++n;
	} while ( FindNextFileA( h, &fd ) );
	FindClose( h );
	if ( n > 0 )
		Msg( "screenspace: removed %d stale pathrad_*.vcs from %sshaders\\fxc\n", n, gamedir );
}

static void InstallShaderToGame( const char *pakFileName, const void *data, int size )
{
	if ( !gamedir[0] || !data || size <= 0 )
		return;
	char shadersDir[MAX_PATH], fxcDir[MAX_PATH], dest[MAX_PATH];
	Q_snprintf( shadersDir, sizeof( shadersDir ), "%sshaders", gamedir );
	Q_snprintf( fxcDir, sizeof( fxcDir ), "%s\\fxc", shadersDir );
	CreateDirectoryA( shadersDir, NULL );
	CreateDirectoryA( fxcDir, NULL );
	Q_snprintf( dest, sizeof( dest ), "%s\\%s", fxcDir, pakFileName );
	FILE *f = fopen( dest, "wb" );
	if ( !f )
	{
		Warning( "screenspace: could not write %s (GMod will keep any cached shader)\n", dest );
		return;
	}
	fwrite( data, 1, size, f );
	fclose( f );
	Msg( "screenspace: installed %s (%d bytes) — GMod loads this over the BSP pak\n", dest, size );
}

static void PackShaderIfPresent( const char *fileName, const char *pakFileName )
{
	char diskPath[MAX_PATH];
	if ( !FindShaderFile( fileName, diskPath, sizeof( diskPath ) ) )
	{
		Warning( "fog_volume: missing shader %s (run shaders/build_shaders.ps1). Fog will not draw until packed.\n",
				 fileName );
		return;
	}

	FILE *f = fopen( diskPath, "rb" );
	if ( !f )
	{
		Warning( "fog_volume: could not read %s\n", diskPath );
		return;
	}
	fseek( f, 0, SEEK_END );
	long sz = ftell( f );
	fseek( f, 0, SEEK_SET );
	if ( sz <= 0 )
	{
		fclose( f );
		return;
	}
	CUtlBuffer buf( 0, (int)sz, 0 );
	buf.EnsureCapacity( (int)sz );
	fread( buf.Base(), 1, sz, f );
	buf.SeekPut( CUtlBuffer::SEEK_HEAD, (int)sz );
	fclose( f );

	char pakName[128];
	Q_snprintf( pakName, sizeof( pakName ), "shaders/fxc/%s", pakFileName );
	AddBufferToPak( GetPakFile(), pakName, buf.Base(), buf.TellPut(), false );
	Msg( "fog_volume: packed %s (%d bytes)\n", pakName, buf.TellPut() );
	InstallShaderToGame( pakFileName, buf.Base(), buf.TellPut() );
}

static void PackScreenspaceShaders()
{
	ClearOldGameShaders();
	char psName[80], vsName[80];
	Q_snprintf( psName, sizeof( psName ), "pathrad_fog_%s_ps30.vcs", FogVolume_ShaderId() );
	Q_snprintf( vsName, sizeof( vsName ), "pathrad_fog_%s_vs30.vcs", FogVolume_ShaderId() );
	PackShaderIfPresent( "pathrad_fog_ps30.vcs", psName );
	PackShaderIfPresent( "pathrad_fog_vs30.vcs", vsName );
	Msg( "screenspace: packed pathrad_fog id=%s (unique names bust GMod shader cache)\n",
		 FogVolume_ShaderId() );
}

static void BuildLuaCodeSetup( const FogVolume_t &v, char *out, int outSize )
{
	// Part A: volume locals, CreateClientConVar defaults, CVFS/CVFG/CVFP helpers.
	// Part B (separate lua_run) draws. No double-quotes. Keep under MAX_VALUE (1024).
	int nx, ny, nz;
	Vector lightMins, lightMaxs;
	FogLightGridBounds( v, lightMins, lightMaxs );
	ComputeGridDims( lightMins, lightMaxs, v.gridSpacing, nx, ny, nz );
	const int lz = FogGridLog2Minus2( nz );
	const int bm = clamp( v.blendMode, 0, 2 );
	(void)nx; (void)ny;

	// Force-set ConVars each load so archived client values cannot ignore the bake.
	Q_snprintf( out, outSize,
		"local c=[["
		"local P,mi,ma,lz,bm='%s',Vector(%.0f,%.0f,%.0f),Vector(%.0f,%.0f,%.0f),%d,%d "
		"for k,v in pairs{de=%.4f,ns=%.4f,cov=%.3f,wspd=%.4f,wyaw=%.0f,bd=%.0f,en=%d}"
		"do local n='cvf_'..k CreateClientConVar(n,tostring(v),false,false)"
		"local c=GetConVar(n)if c then c:SetString(tostring(v))end end "
		"_CVF={P=P,mi=mi,ma=ma,lz=lz,bm=bm}"
		"function CVFS(m,p,x,y,z,w)m:SetFloat(p..'_x',x)m:SetFloat(p..'_y',y)m:SetFloat(p..'_z',z)m:SetFloat(p..'_w',w)end "
		"function CVFG(n)return GetConVarNumber('cvf_'..n)end "
		"function CVFP()local t=_CVF if not t then return 0 end "
		"return t.lz+t.bm*8+math.Clamp(math.floor(CVFG('cov')*15+.5),0,15)*32+"
		"(math.floor(CVFG('wyaw')/45+.5)%%8)*512+math.Clamp(math.floor(CVFG('bd')+.5),0,2047)*4096 end]]"
		"timer.Simple(0,function()BroadcastLua(c)end)"
		"hook.Add('PlayerInitialSpawn','cvf_%d_a',function(p)p:SendLua(c)end)",
		v.materialPath,
		v.mins.x, v.mins.y, v.mins.z,
		v.maxs.x, v.maxs.y, v.maxs.z,
		lz, bm,
		v.density, v.noiseScale, v.noiseCoverage, v.windSpeed, v.windYaw, v.blendDistance,
		v.bStartEnabled ? 1 : 0,
		v.hammerId );
}

static void BuildLuaCodeDraw( const FogVolume_t &v, char *out, int outSize )
{
	// Part B: delayed so setup ConVars/helpers arrive first. Reads cvf_* live each frame.
	Q_snprintf( out, outSize,
		"local c=[["
		"hook.Add('NeedsDepthPass','cvf_%d_d',function()return true end)"
		"hook.Add('RenderScreenspaceEffects','cvf_%d',function()"
		"local t=_CVF if not t or not CVFS or CVFG('en')<1 then return end "
		"render.UpdateFullScreenDepthTexture()"
		"local m=Material(t.P)if m:IsError()then return end "
		"local e,f,v=EyePos(),EyeAngles():Forward(),render.GetViewSetup()"
		"local thx=math.tan(math.rad(v.fov)*.5)"
		"CVFS(m,'$c0',e.x,e.y,e.z,CVFG('de'))"
		"CVFS(m,'$c1',t.mi.x,t.mi.y,t.mi.z,t.ma.z)"
		"CVFS(m,'$c2',t.ma.x,t.ma.y,math.max(0,CVFG('ns')),CurTime()*CVFG('wspd'))"
		"CVFS(m,'$c3',f.x*thx,f.y*thx,f.z*thx,CVFP())"
		"local d=render.GetResolvedFullFrameDepth()if d then m:SetTexture('$texture1',d)end "
		"render.OverrideBlend(true,1,5,0)"
		"render.SetMaterial(m)render.DrawScreenQuad()render.OverrideBlend(false)end)]]"
		"timer.Simple(.1,function()BroadcastLua(c)end)"
		"hook.Add('PlayerInitialSpawn','cvf_%d_b',function(p)"
		"timer.Simple(.1,function()if IsValid(p)then p:SendLua(c)end end)end)",
		v.hammerId, v.hammerId, v.hammerId );
}

static entity_t *AppendMapEntity()
{
	if ( num_entities >= MAX_MAP_ENTITIES )
	{
		Warning( "fog_volume: MAX_MAP_ENTITIES — cannot add second lua_run\n" );
		return NULL;
	}
	entity_t *e = &entities[num_entities];
	memset( e, 0, sizeof( *e ) );
	num_entities++;
	return e;
}

static void WriteLuaRunEntity( entity_t *e, const Vector &origin, int hammerId,
							   const char *part, const char *code )
{
	char originStr[64];
	char hammerIdStr[32];
	Q_snprintf( originStr, sizeof( originStr ), "%.1f %.1f %.1f", origin.x, origin.y, origin.z );
	Q_snprintf( hammerIdStr, sizeof( hammerIdStr ), "%d", hammerId );

	SetKeyValue( e, "classname", "lua_run" );
	SetKeyValue( e, "origin", originStr );
	SetKeyValue( e, "Code", code );
	SetKeyValue( e, "spawnflags", "1" ); // run on spawn
	SetKeyValue( e, "_fog_volume", "1" );
	SetKeyValue( e, "_fog_part", part );
	SetKeyValue( e, "hammerid", hammerIdStr );
}

// MakePatches() re-parses entities without clearing epairs, so keys are duplicated.
// Engine keyvalues are last-wins — a trailing classname "fog_volume" would override our rewrite.
static void ClearEntityEpairs( entity_t *e )
{
	epair_t *ep = e->epairs;
	while ( ep )
	{
		epair_t *next = ep->next;
		free( ep->key );
		free( ep->value );
		free( ep );
		ep = next;
	}
	e->epairs = NULL;
}

static void RewriteEntity( FogVolume_t &v )
{
	entity_t *e = v.pEntity;
	if ( !e )
		return;

	char codeA[MAX_VALUE];
	char codeB[MAX_VALUE];
	BuildLuaCodeSetup( v, codeA, sizeof( codeA ) );
	BuildLuaCodeDraw( v, codeB, sizeof( codeB ) );
	Msg( "fog_volume %d: lua_run setup=%d draw=%d / %d\n",
		 v.hammerId, (int)strlen( codeA ), (int)strlen( codeB ), MAX_VALUE - 1 );
	if ( (int)strlen( codeA ) >= MAX_VALUE - 1 || (int)strlen( codeB ) >= MAX_VALUE - 1 )
	{
		Warning( "fog_volume: generated Code too long for hammerid %d (setup=%d draw=%d)\n",
				 v.hammerId, (int)strlen( codeA ), (int)strlen( codeB ) );
		return;
	}

	Vector center = ( v.mins + v.maxs ) * 0.5f;

	ClearEntityEpairs( e );
	WriteLuaRunEntity( e, center, v.hammerId, "setup", codeA );

	entity_t *eDraw = AppendMapEntity();
	if ( !eDraw )
		return;
	// Offset slightly so Hammer/engine don't merge identical origins.
	Vector drawOrigin = center;
	drawOrigin.z += 1.0f;
	WriteLuaRunEntity( eDraw, drawOrigin, v.hammerId, "draw", codeB );

	Msg( "fog_volume %d: two lua_run entities (setup+draw ConVars cvf_*)\n", v.hammerId );
}

static void BakeOne( FogVolume_t &v )
{
	// Bake over brush±BlendDistance for Outside/Center so outer fade has real lighting.
	FogVolume_t bakeV = v;
	FogLightGridBounds( v, bakeV.mins, bakeV.maxs );

	int nx, ny, nz;
	ComputeGridDims( bakeV.mins, bakeV.maxs, bakeV.gridSpacing, nx, ny, nz );
	const int count = nx * ny * nz;

	ColorRGBExp32 *pSamples = new ColorRGBExp32[count];
	memset( pSamples, 0, sizeof( ColorRGBExp32 ) * count );

	if ( v.blendDistance > 1e-3f && v.blendMode >= 1 )
		Msg( "fog_volume %d: light grid expanded by BlendDistance %.0f (mode %d)\n",
			 v.hammerId, v.blendDistance, v.blendMode );
	Msg( "fog_volume %d: baking light grid %dx%dx%d (spacing~%.1f)...\n",
		 v.hammerId, nx, ny, nz, v.gridSpacing );

	bool bOk = false;
	if ( PathTraceDXR_CanBakeProps() )
		bOk = BakeLightGridPathTrace( bakeV, nx, ny, nz, pSamples );
	if ( !bOk )
		BakeLightGridClassic( bakeV, nx, ny, nz, pSamples );

	DilateFogLightGrid( pSamples, nx, ny, nz, bakeV );

	unsigned char *pAllow = new unsigned char[count];
	BuildFogAllowMask( bakeV, nx, ny, nz, pAllow );
	// Zero lighting in fully blocked cells so dilation/filter cannot reintroduce fog glow.
	for ( int i = 0; i < count; ++i )
	{
		if ( pAllow[i] == 0 )
		{
			pSamples[i].r = 0;
			pSamples[i].g = 0;
			pSamples[i].b = 0;
			pSamples[i].exponent = 0;
		}
	}

	char vtfPak[256];
	Q_snprintf( vtfPak, sizeof( vtfPak ), "materials/%s.vtf", v.materialPath );
	const int atlasW = RoundUpPow2Clamped( nx, 4, kMaxGridAxis );
	const int atlasH = RoundUpPow2Clamped( ny * nz, 4, kMaxGridAxis * kMaxGridAxis );
	if ( !WriteLightGridAtlasVTF( vtfPak, nx, ny, nz, atlasW, atlasH, pSamples, pAllow ) )
	{
		Warning( "fog_volume: failed writing %s — skipping VMT/lua rewrite for this volume\n", vtfPak );
		delete[] pAllow;
		delete[] pSamples;
		return;
	}
	Msg( "fog_volume: wrote %s (grid %dx%dx%d, atlas %dx%d)\n", vtfPak, nx, ny, nz, atlasW, atlasH );
	delete[] pAllow;
	delete[] pSamples;

	if ( !EnsureFogNoiseAtlasPacked() )
		Warning( "fog_volume: noise atlas pack failed — procedural look unavailable (NoiseScale ignored)\n" );

	WriteVMT( v, nx, ny, nz, atlasW, atlasH );
	RewriteEntity( v );
	Msg( "fog_volume %d: rewritten to 2x lua_run (setup+draw), material '%s'\n", v.hammerId, v.materialPath );
}

static void RewriteBlockerEntity( FogBlocker_t &b )
{
	entity_t *e = b.pEntity;
	if ( !e )
		return;
	ClearEntityEpairs( e );
	Vector center = ( b.mins + b.maxs ) * 0.5f;
	char origin[64];
	Q_snprintf( origin, sizeof( origin ), "%.1f %.1f %.1f", center.x, center.y, center.z );
	SetKeyValue( e, "classname", "info_null" );
	SetKeyValue( e, "origin", origin );
}

} // namespace

void FogVolume_Clear()
{
	s_nVolumes = 0;
	s_nBlockers = 0;
	s_bNoiseAtlasPacked = false;
	s_NoiseMaterialPath[0] = 0;
	s_ShaderId[0] = 0;
}

bool FogVolume_HasVolumes()
{
	return s_nVolumes > 0;
}

int FogVolume_VolumeCount()
{
	return s_nVolumes;
}

void FogVolume_ParseBlockerEntity( entity_t *e )
{
	if ( !e )
		return;

	char *pModel = ValueForKey( e, "model" );
	if ( !pModel || pModel[0] != '*' )
	{
		Warning( "WARNING: fog_volume_blocker without brush model ignored\n" );
		return;
	}
	int modelIndex = atoi( pModel + 1 );
	if ( modelIndex <= 0 || modelIndex >= nummodels )
	{
		Warning( "WARNING: fog_volume_blocker has invalid model \"%s\"\n", pModel );
		return;
	}
	if ( s_nBlockers >= FOG_BLOCKER_MAX )
	{
		Warning( "WARNING: too many fog_volume_blocker entities (max %d)\n", FOG_BLOCKER_MAX );
		return;
	}

	FogBlocker_t &b = s_Blockers[s_nBlockers];
	memset( &b, 0, sizeof( b ) );
	b.pEntity = e;
	b.mins = dmodels[modelIndex].mins;
	b.maxs = dmodels[modelIndex].maxs;
	b.nPlanes = ExtractBrushPlanes( modelIndex, b.planeN, b.planeD, 24, b.mins, b.maxs );
	b.bEnabled = IntForKeyWithDefault( e, "Enabled", 1 ) != 0;
	b.blendDistance = FloatForKeyWithDefault( e, "BlendDistance", 16.0f );
	if ( b.blendDistance < 0.0f )
		b.blendDistance = 0.0f;

	Msg( "fog_volume_blocker  mins(%.0f %.0f %.0f) maxs(%.0f %.0f %.0f) blend=%.0f planes=%d enabled=%s\n",
		 b.mins.x, b.mins.y, b.mins.z,
		 b.maxs.x, b.maxs.y, b.maxs.z,
		 b.blendDistance, b.nPlanes,
		 b.bEnabled ? "yes" : "no" );
	++s_nBlockers;
}

void FogVolume_ParseEntity( entity_t *e )
{
	if ( !e )
		return;

	char *pModel = ValueForKey( e, "model" );
	if ( !pModel || pModel[0] != '*' )
	{
		Warning( "WARNING: fog_volume without brush model ignored\n" );
		return;
	}
	int modelIndex = atoi( pModel + 1 );
	if ( modelIndex <= 0 || modelIndex >= nummodels )
	{
		Warning( "WARNING: fog_volume has invalid model \"%s\"\n", pModel );
		return;
	}
	if ( s_nVolumes >= FOG_VOLUME_MAX )
	{
		Warning( "WARNING: too many fog_volume entities (max %d)\n", FOG_VOLUME_MAX );
		return;
	}

	FogVolume_t &v = s_Volumes[s_nVolumes];
	memset( &v, 0, sizeof( v ) );
	v.pEntity = e;
	v.mins = dmodels[modelIndex].mins;
	v.maxs = dmodels[modelIndex].maxs;
	v.nPlanes = ExtractBrushPlanes( modelIndex, v.planeN, v.planeD, 24, v.mins, v.maxs );
	v.bEnabled = IntForKeyWithDefault( e, "Enabled", 1 ) != 0;
	v.bStartEnabled = IntForKeyWithDefault( e, "StartEnabled", 1 ) != 0;
	ParseColor255( e, "FogColor", v.fogColor, 200.0f / 255.0f, 220.0f / 255.0f, 1.0f );
	v.density = FloatForKeyWithDefault( e, "Density", 0.02f );
	if ( v.density < 0.0f ) v.density = 0.0f;
	v.lightBoost = FloatForKeyWithDefault( e, "LightBoost", 1.0f );
	v.lightBoost = clamp( v.lightBoost, 0.0f, 16.0f );
	v.anisotropy = FloatForKeyWithDefault( e, "Anisotropy", 0.0f );
	v.anisotropy = clamp( v.anisotropy, -1.0f, 1.0f );
	v.gridSpacing = FloatForKeyWithDefault( e, "GridSpacing", 32.0f );
	if ( v.gridSpacing < 1.0f ) v.gridSpacing = 1.0f;
	v.stepCount = IntForKeyWithDefault( e, "StepCount", 32 );
	v.stepCount = clamp( v.stepCount, 8, 128 );
	v.blendDistance = FloatForKeyWithDefault( e, "BlendDistance", 0.0f );
	if ( v.blendDistance < 0.0f ) v.blendDistance = 0.0f;
	if ( v.blendDistance > 2047.0f ) v.blendDistance = 2047.0f;
	v.blendMode = IntForKeyWithDefault( e, "BlendMode", 2 );
	v.noiseScale = FloatForKeyWithDefault( e, "NoiseScale", 0.0f );
	if ( v.noiseScale < 0.0f ) v.noiseScale = 0.0f;
	v.noiseCoverage = FloatForKeyWithDefault( e, "NoiseCoverage", 0.45f );
	v.noiseCoverage = clamp( v.noiseCoverage, 0.0f, 1.0f );
	v.windSpeed = FloatForKeyWithDefault( e, "WindSpeed", 8.0f );
	if ( v.windSpeed < 0.0f ) v.windSpeed = 0.0f;
	v.windYaw = FloatForKeyWithDefault( e, "WindDir", 0.0f );
	v.priority = IntForKeyWithDefault( e, "priority", 0 );
	v.hammerId = IntForKeyWithDefault( e, "hammerid", s_nVolumes + 1 );
	if ( v.hammerId <= 0 )
		v.hammerId = s_nVolumes + 1;

	char mapName[64];
	GetMapBaseName( mapName, sizeof( mapName ) );
	Q_snprintf( v.materialPath, sizeof( v.materialPath ), "maps/%s/fog_volume_%d", mapName, v.hammerId );

	Msg( "fog_volume %d  mins(%.0f %.0f %.0f) maxs(%.0f %.0f %.0f) planes=%d density=%.4f spacing=%.1f color=(%.2f %.2f %.2f) boost=%.2f blend=%.0f mode=%d noise=%.4f cov=%.2f wind=%.1f@%.0f enabled=%s\n",
		 v.hammerId,
		 v.mins.x, v.mins.y, v.mins.z,
		 v.maxs.x, v.maxs.y, v.maxs.z,
		 v.nPlanes,
		 v.density, v.gridSpacing,
		 v.fogColor.x, v.fogColor.y, v.fogColor.z,
		 v.lightBoost,
		 v.blendDistance, v.blendMode,
		 v.noiseScale, v.noiseCoverage, v.windSpeed, v.windYaw,
		 v.bEnabled ? "yes" : "no" );
	++s_nVolumes;
}

void FogVolume_BakeAndEmbed()
{
	const int nActive = ActiveFogCount();
	const bool bFog = nActive > 0;

	if ( !bFog && s_nBlockers <= 0 )
		return;

	if ( bFog )
	{
		StampFogMaterialPaths();
		PackScreenspaceShaders();
	}

	if ( bFog )
	{
		if ( s_nBlockers > 0 )
			Msg( "fog_volume: baking %d volume(s) with %d blocker(s)...\n",
				 nActive, s_nBlockers );
		else
			Msg( "fog_volume: baking %d volume(s)...\n", nActive );

		for ( int i = 0; i < s_nVolumes; ++i )
		{
			if ( !s_Volumes[i].bEnabled )
				continue;
			BakeOne( s_Volumes[i] );
		}
	}
	else if ( s_nVolumes > 0 )
	{
		Msg( "fog_volume: %d parsed, none enabled — skipping bake\n", s_nVolumes );
	}

	for ( int i = 0; i < s_nBlockers; ++i )
		RewriteBlockerEntity( s_Blockers[i] );
	if ( s_nBlockers > 0 )
		Msg( "fog_volume: rewrote %d fog_volume_blocker(s) to info_null\n", s_nBlockers );

	// WriteBSPFile emits dentdata as-is; must rebuild it after classname/Code rewrite.
	UnparseEntities();
	Msg( "fog_volume: done (entity lump updated)\n" );
}
