//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Lightmapped props sample one lightmap. Charts that cover the same 0..1 UV
// region would write the same texels, so those islands are packed into the
// fewest bins whose charts do not overlap (graph coloring of the overlap
// graph; exact when the graph is small). Seams that only touch stay together.
// Higher LODs use that LOD's own vertex buffer and follow the LOD0
// surface they sit on, so the mesh stays intact when the engine switches.
//
//=============================================================================//

#include "vrad.h"
#include "studio.h"
#include "optimize.h"
#include "prop_uv_split.h"

#include <stdlib.h>
#include <intrin.h>
#include <math.h>

enum { kPolyMax = 32 };

struct Poly2
{
	Vector2D v[kPolyMax];
	int n;
};

struct TriRec
{
	int body;
	int model;
	int lod;
	int mesh;
	int stripGroup;
	int indexOffset;
	int serial;
	int island;
	int v0, v1, v2;
	int meshKey;
	Vector p0, p1, p2;
	bool hasPos;
	Poly2 clip;
	float area;
	float minx, miny, maxx, maxy;
};

struct NodeRec
{
	CUtlVector<int> tris;
	float minx, miny, maxx, maxy;
	float area;
};

struct TriHit
{
	int body;
	int model;
	int lod;
	int mesh;
	int stripGroup;
	int indexOffset;
	unsigned short *ix;
	int v0, v1, v2;
	Vector2D uv0, uv1, uv2;
	Vector p0, p1, p2;
	bool hasPos;
};

static float Orient( const Vector2D &a, const Vector2D &b, const Vector2D &p )
{
	return ( b.x - a.x ) * ( p.y - a.y ) - ( b.y - a.y ) * ( p.x - a.x );
}

static float PolyArea( const Poly2 &p )
{
	float a = 0.0f;
	for ( int i = 0; i < p.n; ++i )
	{
		const Vector2D &u = p.v[i];
		const Vector2D &w = p.v[( i + 1 ) % p.n];
		a += u.x * w.y - w.x * u.y;
	}
	return fabsf( a ) * 0.5f;
}

static void PushPoly( Poly2 &p, const Vector2D &v )
{
	if ( p.n > 0 )
	{
		const float dx = p.v[p.n - 1].x - v.x;
		const float dy = p.v[p.n - 1].y - v.y;
		if ( dx * dx + dy * dy < 1e-16f )
			return;
	}
	if ( p.n >= kPolyMax )
		return;
	p.v[p.n++] = v;
}

static Poly2 ClipHalf( const Poly2 &in, float nx, float ny, float d )
{
	Poly2 out;
	out.n = 0;
	if ( in.n <= 0 )
		return out;
	for ( int i = 0; i < in.n; ++i )
	{
		const Vector2D &s = in.v[i];
		const Vector2D &e = in.v[( i + 1 ) % in.n];
		const float ds = nx * s.x + ny * s.y + d;
		const float de = nx * e.x + ny * e.y + d;
		const bool ins = ds >= -1e-7f;
		const bool ine = de >= -1e-7f;
		if ( ins && ine )
		{
			PushPoly( out, e );
			continue;
		}
		const float den = ds - de;
		if ( fabsf( den ) < 1e-12f )
		{
			if ( ine )
				PushPoly( out, e );
			continue;
		}
		const float t = ds / den;
		Vector2D hit( s.x + ( e.x - s.x ) * t, s.y + ( e.y - s.y ) * t );
		if ( ins && !ine )
			PushPoly( out, hit );
		else if ( !ins && ine )
		{
			PushPoly( out, hit );
			PushPoly( out, e );
		}
	}
	return out;
}

static Poly2 Clip01( Poly2 p )
{
	p = ClipHalf( p, 1.0f, 0.0f, 0.0f );
	p = ClipHalf( p, -1.0f, 0.0f, 1.0f );
	p = ClipHalf( p, 0.0f, 1.0f, 0.0f );
	p = ClipHalf( p, 0.0f, -1.0f, 1.0f );
	return p;
}

static Poly2 ClipOrient( const Poly2 &in, const Vector2D &a, const Vector2D &b, float sign )
{
	Poly2 out;
	out.n = 0;
	if ( in.n <= 0 )
		return out;
	for ( int i = 0; i < in.n; ++i )
	{
		const Vector2D &s = in.v[i];
		const Vector2D &e = in.v[( i + 1 ) % in.n];
		const float ds = sign * Orient( a, b, s );
		const float de = sign * Orient( a, b, e );
		const bool ins = ds >= -1e-7f;
		const bool ine = de >= -1e-7f;
		if ( ins && ine )
		{
			PushPoly( out, e );
			continue;
		}
		const float den = ds - de;
		if ( fabsf( den ) < 1e-12f )
		{
			if ( ine )
				PushPoly( out, e );
			continue;
		}
		const float t = ds / den;
		Vector2D hit( s.x + ( e.x - s.x ) * t, s.y + ( e.y - s.y ) * t );
		if ( ins && !ine )
			PushPoly( out, hit );
		else if ( !ins && ine )
		{
			PushPoly( out, hit );
			PushPoly( out, e );
		}
	}
	return out;
}

static float IntersectArea( const Poly2 &A, const Poly2 &B )
{
	if ( A.n < 3 || B.n < 3 )
		return 0.0f;
	float w = 0.0f;
	for ( int i = 0; i < B.n; ++i )
	{
		const Vector2D &u = B.v[i];
		const Vector2D &v = B.v[( i + 1 ) % B.n];
		w += u.x * v.y - v.x * u.y;
	}
	if ( fabsf( w ) < 1e-12f )
		return 0.0f;
	const float sign = ( w >= 0.0f ) ? 1.0f : -1.0f;
	Poly2 cur = A;
	for ( int i = 0; i < B.n; ++i )
	{
		cur = ClipOrient( cur, B.v[i], B.v[( i + 1 ) % B.n], sign );
		if ( cur.n < 3 )
			return 0.0f;
	}
	return PolyArea( cur );
}

static bool CoverageOverlaps( float overlap, float areaA, float areaB )
{
	if ( !( overlap > 1e-8f ) )
		return false;
	const float m = ( areaA < areaB ) ? areaA : areaB;
	if ( !( m > 1e-8f ) )
		return false;
	return overlap > m * 1e-4f;
}

static bool BoxHit( const TriRec &a, const TriRec &b )
{
	return a.minx < b.maxx && a.maxx > b.minx && a.miny < b.maxy && a.maxy > b.miny;
}

static bool TrisOverlap( const CUtlVector<TriRec> &tris, int ia, int ib )
{
	const TriRec &a = tris[ia];
	const TriRec &b = tris[ib];
	if ( a.area <= 0.0f || b.area <= 0.0f )
		return false;
	if ( !BoxHit( a, b ) )
		return false;
	const float overlap = IntersectArea( a.clip, b.clip );
	return CoverageOverlaps( overlap, a.area, b.area );
}

static void BoxOfPoly( TriRec &tri )
{
	tri.area = PolyArea( tri.clip );
	if ( tri.clip.n <= 0 || tri.area <= 0.0f )
	{
		tri.minx = 1.0f;
		tri.miny = 1.0f;
		tri.maxx = 0.0f;
		tri.maxy = 0.0f;
		tri.area = 0.0f;
		return;
	}
	tri.minx = tri.maxx = tri.clip.v[0].x;
	tri.miny = tri.maxy = tri.clip.v[0].y;
	for ( int i = 1; i < tri.clip.n; ++i )
	{
		const Vector2D &p = tri.clip.v[i];
		if ( p.x < tri.minx ) tri.minx = p.x;
		if ( p.y < tri.miny ) tri.miny = p.y;
		if ( p.x > tri.maxx ) tri.maxx = p.x;
		if ( p.y > tri.maxy ) tri.maxy = p.y;
	}
}

template <typename Fn>
static bool VisitModelTris( studiohdr_t *pHdr, OptimizedModel::FileHeader_t *pVtx, int materialFilter, bool *pSawTristrip, Fn fn )
{
	if ( !pHdr || !pVtx )
		return false;
	const int nBody = min( pHdr->numbodyparts, pVtx->numBodyParts );
	for ( int bodyID = 0; bodyID < nBody; ++bodyID )
	{
		mstudiobodyparts_t *pBody = pHdr->pBodypart( bodyID );
		OptimizedModel::BodyPartHeader_t *pVtxBody = pVtx->pBodyPart( bodyID );
		if ( !pBody || !pVtxBody )
			continue;
		const int nModel = min( pBody->nummodels, pVtxBody->numModels );
		for ( int modelID = 0; modelID < nModel; ++modelID )
		{
			mstudiomodel_t *pModel = pBody->pModel( modelID );
			OptimizedModel::ModelHeader_t *pVtxModel = pVtxBody->pModel( modelID );
			if ( !pModel || !pVtxModel )
				continue;
			const int nLod = min( pVtx->numLODs, pVtxModel->numLODs );
			for ( int lod = 0; lod < nLod; ++lod )
			{
				OptimizedModel::ModelLODHeader_t *pLod = pVtxModel->pLOD( lod );
				if ( !pLod )
					continue;
				const int nMesh = min( pModel->nummeshes, pLod->numMeshes );
				for ( int meshID = 0; meshID < nMesh; ++meshID )
				{
					mstudiomesh_t *pMesh = pModel->pMesh( meshID );
					if ( !pMesh )
						continue;
					if ( materialFilter >= 0 && pMesh->material != materialFilter )
						continue;
					const mstudio_meshvertexdata_t *vertData = pMesh->GetVertexData( (void *)pHdr );
					if ( !vertData )
						continue;
					OptimizedModel::MeshHeader_t *pVtxMesh = pLod->pMesh( meshID );
					if ( !pVtxMesh )
						continue;
					for ( int group = 0; group < pVtxMesh->numStripGroups; ++group )
					{
						OptimizedModel::StripGroupHeader_t *pStripGroup = pVtxMesh->pStripGroup( group );
						if ( !pStripGroup )
							continue;
						for ( int strip = 0; strip < pStripGroup->numStrips; ++strip )
						{
							OptimizedModel::StripHeader_t *pStrip = pStripGroup->pStrip( strip );
							if ( !pStrip )
								continue;
							if ( pStrip->flags & OptimizedModel::STRIP_IS_TRISTRIP )
							{
								if ( pSawTristrip )
									*pSawTristrip = true;
								continue;
							}
							if ( ( pStrip->flags & OptimizedModel::STRIP_IS_TRILIST ) == 0 )
								continue;
							for ( int i = 0; i + 2 < pStrip->numIndices; i += 3 )
							{
								const int idx = pStrip->indexOffset + i;
								unsigned short *ix = pStripGroup->pIndex( idx );
								if ( !ix )
									continue;
								const int i0 = ix[0];
								const int i1 = ix[1];
								const int i2 = ix[2];
								if ( i0 < 0 || i1 < 0 || i2 < 0 ||
									i0 >= pStripGroup->numVerts || i1 >= pStripGroup->numVerts || i2 >= pStripGroup->numVerts )
									continue;
								const int v0 = pStripGroup->pVertex( i0 )->origMeshVertID;
								const int v1 = pStripGroup->pVertex( i1 )->origMeshVertID;
								const int v2 = pStripGroup->pVertex( i2 )->origMeshVertID;
								// LOD N indexes the vertex buffer after that LOD's fixups,
								// not the LOD0 mesh the rest of VRAD keeps loaded.
								int vertLimit = pMesh->numvertices;
								if ( lod > 0 )
								{
									const int lodCount = pMesh->vertexdata.numLODVertexes[lod];
									if ( lodCount > 0 )
										vertLimit = lodCount;
								}
								if ( v0 < 0 || v1 < 0 || v2 < 0 || v0 >= vertLimit || v1 >= vertLimit || v2 >= vertLimit )
									continue;
								TriHit hit;
								hit.body = bodyID;
								hit.model = modelID;
								hit.lod = lod;
								hit.mesh = meshID;
								hit.stripGroup = group;
								hit.indexOffset = idx;
								hit.ix = ix;
								hit.v0 = v0;
								hit.v1 = v1;
								hit.v2 = v2;
								hit.hasPos = false;
								hit.p0.Init();
								hit.p1.Init();
								hit.p2.Init();
								hit.uv0.Init();
								hit.uv1.Init();
								hit.uv2.Init();
								if ( lod == 0 && vertData &&
									v0 < pMesh->numvertices && v1 < pMesh->numvertices && v2 < pMesh->numvertices )
								{
									hit.uv0 = *vertData->Texcoord( v0 );
									hit.uv1 = *vertData->Texcoord( v1 );
									hit.uv2 = *vertData->Texcoord( v2 );
									hit.p0 = *vertData->Position( v0 );
									hit.p1 = *vertData->Position( v1 );
									hit.p2 = *vertData->Position( v2 );
									hit.hasPos = true;
								}
								fn( hit );
							}
						}
					}
				}
			}
		}
	}
	return true;
}

struct EdgeRec
{
	int serial;
	int v0;
	int v1;
	int tri;
};

static int CmpEdge( const void *va, const void *vb )
{
	const EdgeRec *a = (const EdgeRec *)va;
	const EdgeRec *b = (const EdgeRec *)vb;
	if ( a->serial != b->serial ) return a->serial - b->serial;
	if ( a->v0 != b->v0 ) return a->v0 - b->v0;
	if ( a->v1 != b->v1 ) return a->v1 - b->v1;
	return a->tri - b->tri;
}

static int CmpDegDesc( void *ctx, const void *va, const void *vb )
{
	const int *deg = (const int *)ctx;
	const int ia = *(const int *)va;
	const int ib = *(const int *)vb;
	if ( deg[ia] != deg[ib] )
		return deg[ib] - deg[ia];
	return ia - ib;
}

static void AddAdj( CUtlVector<int> &a, int v )
{
	for ( int i = 0; i < a.Count(); ++i )
	{
		if ( a[i] == v )
			return;
	}
	a.AddToTail( v );
}

static int CountColors( const CUtlVector<int> &color )
{
	int n = 0;
	for ( int i = 0; i < color.Count(); ++i )
	{
		if ( color[i] + 1 > n )
			n = color[i] + 1;
	}
	return n;
}

static int Dsatur( const CUtlVector< CUtlVector<int> > &adj, CUtlVector<int> &color )
{
	const int n = adj.Count();
	color.SetCount( n );
	CUtlVector<int> deg;
	deg.SetCount( n );
	for ( int i = 0; i < n; ++i )
	{
		color[i] = -1;
		deg[i] = adj[i].Count();
	}
	CUtlVector<int> stamp;
	stamp.SetCount( n );
	for ( int i = 0; i < n; ++i )
		stamp[i] = 0;
	int tick = 0;
	int used = 0;
	for ( int placed = 0; placed < n; ++placed )
	{
		int best = -1;
		int bestSat = -1;
		int bestDeg = -1;
		for ( int v = 0; v < n; ++v )
		{
			if ( color[v] >= 0 )
				continue;
			if ( ++tick == 0x7fffffff )
			{
				for ( int s = 0; s < n; ++s )
					stamp[s] = 0;
				tick = 1;
			}
			int sat = 0;
			for ( int e = 0; e < adj[v].Count(); ++e )
			{
				const int c = color[adj[v][e]];
				if ( c < 0 || c >= n )
					continue;
				if ( stamp[c] != tick )
				{
					stamp[c] = tick;
					++sat;
				}
			}
			if ( sat > bestSat || ( sat == bestSat && ( deg[v] > bestDeg || ( deg[v] == bestDeg && ( best < 0 || v < best ) ) ) ) )
			{
				bestSat = sat;
				bestDeg = deg[v];
				best = v;
			}
		}
		if ( best < 0 )
			break;
		if ( ++tick == 0x7fffffff )
		{
			for ( int s = 0; s < n; ++s )
				stamp[s] = 0;
			tick = 1;
		}
		for ( int e = 0; e < adj[best].Count(); ++e )
		{
			const int c = color[adj[best][e]];
			if ( c >= 0 && c < n )
				stamp[c] = tick;
		}
		int c = 0;
		while ( c < n && stamp[c] == tick )
			++c;
		color[best] = c;
		if ( c + 1 > used )
			used = c + 1;
	}
	return used;
}

static int WelshPowell( const CUtlVector< CUtlVector<int> > &adj, CUtlVector<int> &color )
{
	const int n = adj.Count();
	color.SetCount( n );
	CUtlVector<int> deg;
	CUtlVector<int> order;
	deg.SetCount( n );
	order.SetCount( n );
	for ( int i = 0; i < n; ++i )
	{
		color[i] = -1;
		deg[i] = adj[i].Count();
		order[i] = i;
	}
	if ( n > 1 )
		qsort_s( order.Base(), n, sizeof( int ), CmpDegDesc, deg.Base() );
	int used = 0;
	CUtlVector<int> stamp;
	stamp.SetCount( n );
	for ( int i = 0; i < n; ++i )
		stamp[i] = 0;
	int tick = 0;
	for ( int k = 0; k < n; ++k )
	{
		const int v = order[k];
		if ( ++tick == 0x7fffffff )
		{
			for ( int s = 0; s < n; ++s )
				stamp[s] = 0;
			tick = 1;
		}
		for ( int e = 0; e < adj[v].Count(); ++e )
		{
			const int c = color[adj[v][e]];
			if ( c >= 0 && c < n )
				stamp[c] = tick;
		}
		int c = 0;
		while ( c < n && stamp[c] == tick )
			++c;
		color[v] = c;
		if ( c + 1 > used )
			used = c + 1;
	}
	return used;
}

struct ExactCtx
{
	int n;
	unsigned int mask[16];
	int order[16];
	int cur[16];
	int bestColor[16];
	int best;
	int nodes;
	int lower;
};

static void ExactDfs( ExactCtx *cx, int k, int used )
{
	if ( cx->nodes++ > 2000000 )
		return;
	if ( used >= cx->best )
		return;
	if ( k == cx->n )
	{
		cx->best = used;
		memcpy( cx->bestColor, cx->cur, cx->n * sizeof( int ) );
		return;
	}
	const int v = cx->order[k];
	unsigned int forbid = 0;
	unsigned int bits = cx->mask[v];
	while ( bits )
	{
		unsigned long u = 0;
		_BitScanForward( &u, bits );
		bits &= bits - 1;
		if ( cx->cur[u] >= 0 )
			forbid |= 1u << cx->cur[u];
	}
	for ( int c = 0; c <= used && c < cx->best; ++c )
	{
		if ( forbid & ( 1u << c ) )
			continue;
		const int newUsed = ( c == used ) ? used + 1 : used;
		if ( newUsed >= cx->best )
			continue;
		cx->cur[v] = c;
		ExactDfs( cx, k + 1, newUsed );
		cx->cur[v] = -1;
		if ( cx->best <= cx->lower )
			return;
	}
}

static int CliqueBound( const CUtlVector< CUtlVector<int> > &adj )
{
	const int n = adj.Count();
	int best = 1;
	for ( int seed = 0; seed < n; ++seed )
	{
		CUtlVector<int> clique;
		clique.AddToTail( seed );
		for ( ;; )
		{
			int pick = -1;
			int pickDeg = -1;
			for ( int v = 0; v < n; ++v )
			{
				bool ok = true;
				for ( int k = 0; k < clique.Count(); ++k )
				{
					bool hit = false;
					const CUtlVector<int> &a = adj[v];
					for ( int e = 0; e < a.Count(); ++e )
					{
						if ( a[e] == clique[k] )
						{
							hit = true;
							break;
						}
					}
					if ( !hit )
					{
						ok = false;
						break;
					}
				}
				if ( !ok )
					continue;
				if ( adj[v].Count() > pickDeg )
				{
					pickDeg = adj[v].Count();
					pick = v;
				}
			}
			if ( pick < 0 )
				break;
			clique.AddToTail( pick );
		}
		if ( clique.Count() > best )
			best = clique.Count();
	}
	return best;
}

static int ColorGraph( const CUtlVector< CUtlVector<int> > &adj, CUtlVector<int> &color )
{
	const int n = adj.Count();
	color.SetCount( n );
	if ( n <= 0 )
		return 0;
	if ( n == 1 )
	{
		color[0] = 0;
		return 1;
	}
	CUtlVector<int> a, b;
	const int ca = WelshPowell( adj, a );
	int bestN = ca;
	color = a;
	if ( n <= 500 )
	{
		const int cb = Dsatur( adj, b );
		if ( cb > 0 && cb < bestN )
		{
			bestN = cb;
			color = b;
		}
	}
	if ( n <= 16 && bestN > 1 )
	{
		const int lower = CliqueBound( adj );
		if ( bestN > lower )
		{
			ExactCtx cx;
			memset( &cx, 0, sizeof( cx ) );
			cx.n = n;
			cx.best = bestN;
			cx.lower = lower;
			CUtlVector<int> deg;
			deg.SetCount( n );
			for ( int v = 0; v < n; ++v )
			{
				cx.cur[v] = -1;
				cx.order[v] = v;
				deg[v] = adj[v].Count();
				for ( int e = 0; e < adj[v].Count(); ++e )
				{
					const int u = adj[v][e];
					if ( u >= 0 && u < n )
						cx.mask[v] |= 1u << u;
				}
			}
			qsort_s( cx.order, n, sizeof( int ), CmpDegDesc, deg.Base() );
			ExactDfs( &cx, 0, 0 );
			if ( cx.best < bestN && cx.best > 0 )
			{
				bestN = cx.best;
				for ( int v = 0; v < n; ++v )
					color[v] = cx.bestColor[v];
			}
		}
	}
	for ( int v = 0; v < n; ++v )
	{
		for ( int e = 0; e < adj[v].Count(); ++e )
		{
			const int u = adj[v][e];
			if ( u > v && color[u] == color[v] )
				Warning( "UV split coloring left an overlap in one bin.\n" );
		}
	}
	return CountColors( color );
}

static void RemapLargestFirst( CUtlVector<int> &color, const CUtlVector<float> &weight )
{
	const int nColors = CountColors( color );
	if ( nColors <= 1 )
		return;
	CUtlVector<float> w;
	w.SetCount( nColors );
	for ( int i = 0; i < nColors; ++i )
		w[i] = 0.0f;
	for ( int i = 0; i < color.Count(); ++i )
	{
		if ( color[i] >= 0 && color[i] < nColors )
			w[color[i]] += ( i < weight.Count() ) ? weight[i] : 1.0f;
	}
	CUtlVector<int> order;
	order.SetCount( nColors );
	for ( int i = 0; i < nColors; ++i )
		order[i] = i;
	for ( int a = 0; a < nColors; ++a )
	{
		for ( int b = a + 1; b < nColors; ++b )
		{
			const bool swap = ( w[order[b]] > w[order[a]] ) ||
				( w[order[b]] == w[order[a]] && order[b] < order[a] );
			if ( swap )
			{
				const int t = order[a];
				order[a] = order[b];
				order[b] = t;
			}
		}
	}
	CUtlVector<int> map;
	map.SetCount( nColors );
	for ( int i = 0; i < nColors; ++i )
		map[order[i]] = i;
	for ( int i = 0; i < color.Count(); ++i )
	{
		if ( color[i] >= 0 && color[i] < nColors )
			color[i] = map[color[i]];
	}
}

static bool AnyListOverlap( const CUtlVector<TriRec> &tris, const CUtlVector<int> &a, const CUtlVector<int> &b )
{
	if ( a.Count() <= 0 || b.Count() <= 0 )
		return false;
	if ( a.Count() * b.Count() <= 48 )
	{
		for ( int i = 0; i < a.Count(); ++i )
		{
			for ( int j = 0; j < b.Count(); ++j )
			{
				if ( TrisOverlap( tris, a[i], b[j] ) )
					return true;
			}
		}
		return false;
	}

	float minx = 1.0f, miny = 1.0f, maxx = 0.0f, maxy = 0.0f;
	for ( int i = 0; i < b.Count(); ++i )
	{
		const TriRec &t = tris[b[i]];
		if ( t.area <= 0.0f )
			continue;
		if ( t.minx < minx ) minx = t.minx;
		if ( t.miny < miny ) miny = t.miny;
		if ( t.maxx > maxx ) maxx = t.maxx;
		if ( t.maxy > maxy ) maxy = t.maxy;
	}
	if ( !( maxx > minx ) || !( maxy > miny ) )
		return false;
	const int grid = 16;
	CUtlVector<int> cells[grid * grid];
	const float invX = (float)grid / ( maxx - minx );
	const float invY = (float)grid / ( maxy - miny );
	for ( int i = 0; i < b.Count(); ++i )
	{
		const TriRec &t = tris[b[i]];
		if ( t.area <= 0.0f )
			continue;
		int x0 = (int)( ( t.minx - minx ) * invX );
		int y0 = (int)( ( t.miny - miny ) * invY );
		int x1 = (int)( ( t.maxx - minx ) * invX );
		int y1 = (int)( ( t.maxy - miny ) * invY );
		if ( x0 < 0 ) x0 = 0;
		if ( y0 < 0 ) y0 = 0;
		if ( x1 >= grid ) x1 = grid - 1;
		if ( y1 >= grid ) y1 = grid - 1;
		for ( int y = y0; y <= y1; ++y )
		{
			for ( int x = x0; x <= x1; ++x )
				cells[x + y * grid].AddToTail( b[i] );
		}
	}
	for ( int i = 0; i < a.Count(); ++i )
	{
		const TriRec &t = tris[a[i]];
		if ( t.area <= 0.0f )
			continue;
		int x0 = (int)( ( t.minx - minx ) * invX );
		int y0 = (int)( ( t.miny - miny ) * invY );
		int x1 = (int)( ( t.maxx - minx ) * invX );
		int y1 = (int)( ( t.maxy - miny ) * invY );
		if ( x1 < 0 || y1 < 0 || x0 >= grid || y0 >= grid )
			continue;
		if ( x0 < 0 ) x0 = 0;
		if ( y0 < 0 ) y0 = 0;
		if ( x1 >= grid ) x1 = grid - 1;
		if ( y1 >= grid ) y1 = grid - 1;
		for ( int y = y0; y <= y1; ++y )
		{
			for ( int x = x0; x <= x1; ++x )
			{
				const CUtlVector<int> &cell = cells[x + y * grid];
				for ( int k = 0; k < cell.Count(); ++k )
				{
					if ( TrisOverlap( tris, a[i], cell[k] ) )
						return true;
				}
			}
		}
	}
	return false;
}

static int CmpTriKey( const void *va, const void *vb )
{
	const PropUvTriKey *a = (const PropUvTriKey *)va;
	const PropUvTriKey *b = (const PropUvTriKey *)vb;
	if ( a->body != b->body ) return ( a->body < b->body ) ? -1 : 1;
	if ( a->model != b->model ) return ( a->model < b->model ) ? -1 : 1;
	if ( a->lod != b->lod ) return ( a->lod < b->lod ) ? -1 : 1;
	if ( a->mesh != b->mesh ) return ( a->mesh < b->mesh ) ? -1 : 1;
	if ( a->stripGroup != b->stripGroup ) return ( a->stripGroup < b->stripGroup ) ? -1 : 1;
	if ( a->indexOffset != b->indexOffset ) return ( a->indexOffset < b->indexOffset ) ? -1 : 1;
	return 0;
}

// VTX LOD N indexes vertices after that LOD's VVD fixups. The baker keeps LOD0
// loaded, so a lower LOD has to be cooked on its own or its triangles land on
// the wrong charts and get stripped out of the model.
struct LodLibrary
{
	struct Span
	{
		int body, model, mesh, start, count;
	};
	struct Page
	{
		byte *block;
		const mstudiovertex_t *verts;
		CUtlVector<Span> spans;
	};

	byte *raw;
	Page pages[MAX_NUM_LODS];

	LodLibrary()
	{
		raw = NULL;
		for ( int i = 0; i < MAX_NUM_LODS; ++i )
		{
			pages[i].block = NULL;
			pages[i].verts = NULL;
		}
	}
	~LodLibrary()
	{
		if ( raw )
			free( raw );
		for ( int i = 0; i < MAX_NUM_LODS; ++i )
		{
			if ( pages[i].block )
				free( pages[i].block );
		}
	}

	bool ReadRaw( studiohdr_t *pHdr )
	{
		if ( raw )
			return true;
		char fileName[MAX_PATH];
		strcpy( fileName, "models/" );
		strcat( fileName, pHdr->pszName() );
		Q_StripExtension( fileName, fileName, sizeof( fileName ) );
		strcat( fileName, ".vvd" );
		FileHandle_t fh = g_pFileSystem->Open( fileName, "rb" );
		if ( !fh )
			return false;
		const int sz = g_pFileSystem->Size( fh );
		if ( sz <= (int)sizeof( vertexFileHeader_t ) )
		{
			g_pFileSystem->Close( fh );
			return false;
		}
		raw = (byte *)malloc( sz );
		if ( !raw )
		{
			g_pFileSystem->Close( fh );
			return false;
		}
		g_pFileSystem->Read( raw, sz, fh );
		g_pFileSystem->Close( fh );
		const vertexFileHeader_t *vh = (const vertexFileHeader_t *)raw;
		if ( vh->id != MODEL_VERTEX_FILE_ID || vh->version != MODEL_VERTEX_FILE_VERSION || vh->checksum != pHdr->checksum )
		{
			free( raw );
			raw = NULL;
			return false;
		}
		return true;
	}

	bool Cook( studiohdr_t *pHdr, int lod )
	{
		if ( lod <= 0 || lod >= MAX_NUM_LODS )
			return false;
		if ( pages[lod].verts )
			return true;
		if ( !ReadRaw( pHdr ) )
			return false;
		const vertexFileHeader_t *src = (const vertexFileHeader_t *)raw;
		if ( lod >= src->numLODs )
			return false;
		const int size = Studio_VertexDataSize( src, lod, false );
		byte *block = (byte *)malloc( size );
		if ( !block )
			return false;
		memset( block, 0, size );
		Studio_LoadVertexes( src, (vertexFileHeader_t *)block, lod, false );
		pages[lod].block = block;
		pages[lod].verts = ( (vertexFileHeader_t *)block )->GetVertexData();
		int cursor = 0;
		const int nBody = pHdr->numbodyparts;
		for ( int body = 0; body < nBody; ++body )
		{
			mstudiobodyparts_t *pBody = pHdr->pBodypart( body );
			if ( !pBody )
				continue;
			for ( int model = 0; model < pBody->nummodels; ++model )
			{
				mstudiomodel_t *pModel = pBody->pModel( model );
				if ( !pModel )
					continue;
				int meshCursor = 0;
				for ( int mesh = 0; mesh < pModel->nummeshes; ++mesh )
				{
					mstudiomesh_t *pMesh = pModel->pMesh( mesh );
					const int count = ( pMesh && lod < MAX_NUM_LODS ) ? pMesh->vertexdata.numLODVertexes[lod] : 0;
					Span span;
					span.body = body;
					span.model = model;
					span.mesh = mesh;
					span.start = cursor + meshCursor;
					span.count = count;
					pages[lod].spans.AddToTail( span );
					meshCursor += count;
				}
				cursor += meshCursor;
			}
		}
		return pages[lod].verts != NULL;
	}

	bool Vertex( int lod, int body, int model, int mesh, int local, Vector &pos, Vector2D &uv ) const
	{
		if ( lod <= 0 || lod >= MAX_NUM_LODS || !pages[lod].verts )
			return false;
		const CUtlVector<Span> &spans = pages[lod].spans;
		for ( int i = 0; i < spans.Count(); ++i )
		{
			if ( spans[i].body != body || spans[i].model != model || spans[i].mesh != mesh )
				continue;
			if ( local < 0 || local >= spans[i].count )
				return false;
			const mstudiovertex_t *v = pages[lod].verts + spans[i].start + local;
			pos = v->m_vecPosition;
			uv = v->m_vecTexCoord;
			return true;
		}
		return false;
	}
};

static float DistSqPointTriangle( const Vector &p, const Vector &a, const Vector &b, const Vector &c )
{
	const Vector ab = b - a;
	const Vector ac = c - a;
	const Vector ap = p - a;
	const float d1 = ab.Dot( ap );
	const float d2 = ac.Dot( ap );
	if ( d1 <= 0.0f && d2 <= 0.0f )
		return ap.LengthSqr();

	const Vector bp = p - b;
	const float d3 = ab.Dot( bp );
	const float d4 = ac.Dot( bp );
	if ( d3 >= 0.0f && d4 <= d3 )
		return bp.LengthSqr();

	const float vc = d1 * d4 - d3 * d2;
	if ( vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f )
	{
		const float v = d1 / ( d1 - d3 );
		return ( a + ab * v - p ).LengthSqr();
	}

	const Vector cp = p - c;
	const float d5 = ab.Dot( cp );
	const float d6 = ac.Dot( cp );
	if ( d6 >= 0.0f && d5 <= d6 )
		return cp.LengthSqr();

	const float vb = d5 * d2 - d1 * d6;
	if ( vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f )
	{
		const float w = d2 / ( d2 - d6 );
		return ( a + ac * w - p ).LengthSqr();
	}

	const float va = d3 * d6 - d5 * d4;
	if ( va <= 0.0f && ( d4 - d3 ) >= 0.0f && ( d5 - d6 ) >= 0.0f )
	{
		const float w = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) );
		return ( b + ( c - b ) * w - p ).LengthSqr();
	}

	const float denom = va + vb + vc;
	if ( fabsf( denom ) < 1e-12f )
		return ap.LengthSqr();
	const float v = vb / denom;
	const float w = vc / denom;
	return ( a + ab * v + ac * w - p ).LengthSqr();
}

bool PropUvBuildPlan( studiohdr_t *pHdr, OptimizedModel::FileHeader_t *pVtx, int materialFilter, PropUvSplitPlan &plan )
{
	const int srcDict = plan.m_SrcDict;
	const int material = plan.m_Material;
	plan.m_Tris.Purge();
	plan.m_EmitDict.Purge();
	plan.m_NumBins = 1;
	plan.m_IslandCount = 0;
	plan.m_SrcDict = srcDict;
	plan.m_Material = material;

	if ( !pHdr || !pVtx )
		return false;

	bool sawTristrip = false;
	CUtlVector<TriRec> tris;
	int serial = 0;
	int curSerial = -1;
	int lastBody = -1, lastModel = -1, lastLod = -1, lastMesh = -1;
	LodLibrary lodVerts;
	if ( pVtx->numLODs > 1 )
	{
		for ( int lod = 1; lod < pVtx->numLODs && lod < MAX_NUM_LODS; ++lod )
			lodVerts.Cook( pHdr, lod );
	}

	const bool visited = VisitModelTris( pHdr, pVtx, materialFilter, &sawTristrip, [&]( const TriHit &hit )
	{
		if ( hit.body != lastBody || hit.model != lastModel || hit.lod != lastLod || hit.mesh != lastMesh )
		{
			lastBody = hit.body;
			lastModel = hit.model;
			lastLod = hit.lod;
			lastMesh = hit.mesh;
			curSerial = serial++;
		}
		TriRec tri;
		memset( &tri, 0, sizeof( tri ) );
		tri.body = hit.body;
		tri.model = hit.model;
		tri.lod = hit.lod;
		tri.mesh = hit.mesh;
		tri.stripGroup = hit.stripGroup;
		tri.indexOffset = hit.indexOffset;
		tri.serial = curSerial;
		tri.island = -1;
		tri.v0 = hit.v0;
		tri.v1 = hit.v1;
		tri.v2 = hit.v2;
		tri.meshKey = hit.body * 100000 + hit.model * 1000 + hit.mesh;
		tri.p0 = hit.p0;
		tri.p1 = hit.p1;
		tri.p2 = hit.p2;
		tri.hasPos = hit.hasPos;
		if ( hit.lod > 0 )
		{
			Vector a, b, c;
			Vector2D ua, ub, uc;
			if ( lodVerts.Vertex( hit.lod, hit.body, hit.model, hit.mesh, hit.v0, a, ua ) &&
				lodVerts.Vertex( hit.lod, hit.body, hit.model, hit.mesh, hit.v1, b, ub ) &&
				lodVerts.Vertex( hit.lod, hit.body, hit.model, hit.mesh, hit.v2, c, uc ) )
			{
				tri.p0 = a;
				tri.p1 = b;
				tri.p2 = c;
				tri.hasPos = true;
			}
		}
		tri.clip.n = 3;
		tri.clip.v[0] = hit.uv0;
		tri.clip.v[1] = hit.uv1;
		tri.clip.v[2] = hit.uv2;
		tri.clip = Clip01( tri.clip );
		BoxOfPoly( tri );
		tris.AddToTail( tri );
	} );
	if ( !visited )
		return false;
	if ( sawTristrip )
		return false;
	if ( tris.Count() <= 1 )
		return true;

	CUtlVector<EdgeRec> edges;
	edges.EnsureCapacity( tris.Count() * 3 );
	for ( int t = 0; t < tris.Count(); ++t )
	{
		const int vs[3] = { tris[t].v0, tris[t].v1, tris[t].v2 };
		for ( int e = 0; e < 3; ++e )
		{
			int a = vs[e];
			int b = vs[( e + 1 ) % 3];
			if ( a == b )
				continue;
			if ( a > b )
			{
				const int s = a;
				a = b;
				b = s;
			}
			EdgeRec edge;
			edge.serial = tris[t].serial;
			edge.v0 = a;
			edge.v1 = b;
			edge.tri = t;
			edges.AddToTail( edge );
		}
	}
	if ( edges.Count() > 1 )
		qsort( edges.Base(), edges.Count(), sizeof( EdgeRec ), CmpEdge );

	CUtlVector<int> parent;
	parent.SetCount( tris.Count() );
	for ( int i = 0; i < parent.Count(); ++i )
		parent[i] = i;
	auto Find = [&]( int x ) -> int
	{
		while ( parent[x] != x )
		{
			parent[x] = parent[parent[x]];
			x = parent[x];
		}
		return x;
	};
	for ( int i = 0; i + 1 < edges.Count(); ++i )
	{
		int j = i + 1;
		while ( j < edges.Count() &&
			edges[j].serial == edges[i].serial &&
			edges[j].v0 == edges[i].v0 &&
			edges[j].v1 == edges[i].v1 )
		{
			const int a = Find( edges[i].tri );
			const int b = Find( edges[j].tri );
			if ( a != b )
				parent[b] = a;
			++j;
		}
	}

	CUtlVector<int> dense;
	dense.SetCount( tris.Count() );
	for ( int i = 0; i < dense.Count(); ++i )
		dense[i] = -1;
	int nIslands = 0;
	for ( int i = 0; i < tris.Count(); ++i )
	{
		const int r = Find( i );
		if ( dense[r] < 0 )
			dense[r] = nIslands++;
		tris[i].island = dense[r];
	}
	plan.m_IslandCount = nIslands;

	CUtlVector< CUtlVector<int> > islandTris;
	islandTris.SetCount( nIslands );
	CUtlVector<int> islandLod;
	islandLod.SetCount( nIslands );
	for ( int i = 0; i < nIslands; ++i )
		islandLod[i] = -1;
	for ( int i = 0; i < tris.Count(); ++i )
	{
		const int id = tris[i].island;
		islandTris[id].AddToTail( i );
		islandLod[id] = tris[i].lod;
	}

	CUtlVector<NodeRec> nodes;
	for ( int id = 0; id < nIslands; ++id )
	{
		if ( islandLod[id] != 0 )
			continue;
		const CUtlVector<int> &members = islandTris[id];
		CUtlVector< CUtlVector<int> > localAdj;
		localAdj.SetCount( members.Count() );
		// Grid the island so a clean chart does not test every triangle pair.
		float minx = 1.0f, miny = 1.0f, maxx = 0.0f, maxy = 0.0f;
		for ( int a = 0; a < members.Count(); ++a )
		{
			const TriRec &t = tris[members[a]];
			if ( t.area <= 0.0f )
				continue;
			if ( t.minx < minx ) minx = t.minx;
			if ( t.miny < miny ) miny = t.miny;
			if ( t.maxx > maxx ) maxx = t.maxx;
			if ( t.maxy > maxy ) maxy = t.maxy;
		}
		const int grid = 32;
		CUtlVector<int> cells[grid * grid];
		const bool haveBox = ( maxx > minx ) && ( maxy > miny );
		if ( haveBox )
		{
			const float invX = (float)grid / ( maxx - minx );
			const float invY = (float)grid / ( maxy - miny );
			for ( int a = 0; a < members.Count(); ++a )
			{
				const TriRec &t = tris[members[a]];
				if ( t.area <= 0.0f )
					continue;
				int x0 = (int)( ( t.minx - minx ) * invX );
				int y0 = (int)( ( t.miny - miny ) * invY );
				int x1 = (int)( ( t.maxx - minx ) * invX );
				int y1 = (int)( ( t.maxy - miny ) * invY );
				if ( x0 < 0 ) x0 = 0;
				if ( y0 < 0 ) y0 = 0;
				if ( x1 >= grid ) x1 = grid - 1;
				if ( y1 >= grid ) y1 = grid - 1;
				for ( int y = y0; y <= y1; ++y )
				{
					for ( int x = x0; x <= x1; ++x )
						cells[x + y * grid].AddToTail( a );
				}
			}
			for ( int c = 0; c < grid * grid; ++c )
			{
				const CUtlVector<int> &cell = cells[c];
				for ( int ia = 0; ia < cell.Count(); ++ia )
				{
					for ( int ib = ia + 1; ib < cell.Count(); ++ib )
					{
						const int a = cell[ia];
						const int b = cell[ib];
						if ( !TrisOverlap( tris, members[a], members[b] ) )
							continue;
						AddAdj( localAdj[a], b );
						AddAdj( localAdj[b], a );
					}
				}
			}
		}
		bool dirty = false;
		for ( int a = 0; a < localAdj.Count(); ++a )
		{
			if ( localAdj[a].Count() > 0 )
			{
				dirty = true;
				break;
			}
		}
		CUtlVector<int> localColor;
		int localBins = 1;
		if ( !dirty )
		{
			localColor.SetCount( members.Count() );
			for ( int a = 0; a < members.Count(); ++a )
				localColor[a] = 0;
		}
		else
		{
			localBins = ColorGraph( localAdj, localColor );
		}
		for ( int bin = 0; bin < localBins; ++bin )
		{
			const int nodeIndex = nodes.AddToTail();
			NodeRec &node = nodes[nodeIndex];
			node.minx = 1.0f;
			node.miny = 1.0f;
			node.maxx = 0.0f;
			node.maxy = 0.0f;
			node.area = 0.0f;
			for ( int a = 0; a < members.Count(); ++a )
			{
				if ( localColor[a] != bin )
					continue;
				const int ti = members[a];
				node.tris.AddToTail( ti );
				node.area += tris[ti].area;
				if ( tris[ti].area <= 0.0f )
					continue;
				if ( tris[ti].minx < node.minx ) node.minx = tris[ti].minx;
				if ( tris[ti].miny < node.miny ) node.miny = tris[ti].miny;
				if ( tris[ti].maxx > node.maxx ) node.maxx = tris[ti].maxx;
				if ( tris[ti].maxy > node.maxy ) node.maxy = tris[ti].maxy;
			}
			if ( node.tris.Count() <= 0 )
				nodes.FastRemove( nodeIndex );
		}
	}

	if ( nodes.Count() <= 0 )
		return true;

	CUtlVector< CUtlVector<int> > adj;
	adj.SetCount( nodes.Count() );
	for ( int a = 0; a < nodes.Count(); ++a )
	{
		for ( int b = a + 1; b < nodes.Count(); ++b )
		{
			if ( nodes[a].maxx <= nodes[a].minx || nodes[b].maxx <= nodes[b].minx )
				continue;
			if ( nodes[a].minx >= nodes[b].maxx || nodes[a].maxx <= nodes[b].minx ||
				nodes[a].miny >= nodes[b].maxy || nodes[a].maxy <= nodes[b].miny )
				continue;
			if ( !AnyListOverlap( tris, nodes[a].tris, nodes[b].tris ) )
				continue;
			AddAdj( adj[a], b );
			AddAdj( adj[b], a );
		}
	}

	CUtlVector<int> color;
	int nColors = ColorGraph( adj, color );
	CUtlVector<float> weight;
	weight.SetCount( nodes.Count() );
	for ( int i = 0; i < nodes.Count(); ++i )
		weight[i] = nodes[i].area;
	RemapLargestFirst( color, weight );
	nColors = CountColors( color );
	if ( nColors < 1 )
		nColors = 1;

	CUtlVector<int> triBin;
	triBin.SetCount( tris.Count() );
	for ( int i = 0; i < triBin.Count(); ++i )
		triBin[i] = -1;
	for ( int n = 0; n < nodes.Count(); ++n )
	{
		const int bin = color[n];
		for ( int k = 0; k < nodes[n].tris.Count(); ++k )
			triBin[nodes[n].tris[k]] = bin;
	}

	// A lower LOD is the same surface with fewer triangles. Give each one the
	// bin of the LOD0 triangle it sits on, so that prop still has a mesh when
	// the engine drops to that LOD. Do not open extra bins for this.
	struct MeshLod0_t
	{
		int key;
		CUtlVector<int> tris;
	};
	CUtlVector<MeshLod0_t> meshLod0;
	for ( int i = 0; i < tris.Count(); ++i )
	{
		if ( tris[i].lod != 0 || triBin[i] < 0 || !tris[i].hasPos )
			continue;
		int slot = -1;
		for ( int m = 0; m < meshLod0.Count(); ++m )
		{
			if ( meshLod0[m].key == tris[i].meshKey )
			{
				slot = m;
				break;
			}
		}
		if ( slot < 0 )
		{
			slot = meshLod0.AddToTail();
			meshLod0[slot].key = tris[i].meshKey;
		}
		meshLod0[slot].tris.AddToTail( i );
	}
	for ( int i = 0; i < tris.Count(); ++i )
	{
		if ( tris[i].lod == 0 || triBin[i] >= 0 )
			continue;
		int bestBin = 0;
		if ( tris[i].hasPos )
		{
			const Vector mid = ( tris[i].p0 + tris[i].p1 + tris[i].p2 ) * ( 1.0f / 3.0f );
			float best = FLT_MAX;
			const CUtlVector<int> *cands = NULL;
			for ( int m = 0; m < meshLod0.Count(); ++m )
			{
				if ( meshLod0[m].key == tris[i].meshKey )
				{
					cands = &meshLod0[m].tris;
					break;
				}
			}
			if ( cands )
			{
				for ( int k = 0; k < cands->Count(); ++k )
				{
					const TriRec &lod0 = tris[( *cands )[k]];
					const int bin = triBin[( *cands )[k]];
					if ( bin < 0 || !lod0.hasPos )
						continue;
					const float d = DistSqPointTriangle( mid, lod0.p0, lod0.p1, lod0.p2 );
					if ( d < best || ( d == best && bin < bestBin ) )
					{
						best = d;
						bestBin = bin;
					}
				}
			}
		}
		triBin[i] = bestBin;
	}

	for ( int i = 0; i < triBin.Count(); ++i )
	{
		if ( triBin[i] < 0 )
			triBin[i] = 0;
	}

	if ( nColors <= 1 )
		return true;

	plan.m_NumBins = nColors;
	plan.m_Tris.EnsureCapacity( tris.Count() );
	for ( int i = 0; i < tris.Count(); ++i )
	{
		PropUvTriKey key;
		key.body = (unsigned short)tris[i].body;
		key.model = (unsigned short)tris[i].model;
		key.lod = (unsigned short)tris[i].lod;
		key.mesh = (unsigned short)tris[i].mesh;
		key.stripGroup = (unsigned short)tris[i].stripGroup;
		key.bin = (unsigned short)triBin[i];
		key.indexOffset = tris[i].indexOffset;
		plan.m_Tris.AddToTail( key );
	}
	if ( plan.m_Tris.Count() > 1 )
		qsort( plan.m_Tris.Base(), plan.m_Tris.Count(), sizeof( PropUvTriKey ), CmpTriKey );
	return true;
}

bool PropUvTriKept( const PropUvSplitPlan &plan, int bin, int body, int model, int lod, int mesh, int stripGroup, int indexOffset )
{
	if ( bin < 0 || plan.m_NumBins <= 1 || plan.m_Tris.Count() <= 0 )
		return true;
	PropUvTriKey key;
	key.body = (unsigned short)body;
	key.model = (unsigned short)model;
	key.lod = (unsigned short)lod;
	key.mesh = (unsigned short)mesh;
	key.stripGroup = (unsigned short)stripGroup;
	key.bin = 0;
	key.indexOffset = indexOffset;
	const PropUvTriKey *found = (const PropUvTriKey *)bsearch( &key, plan.m_Tris.Base(), plan.m_Tris.Count(), sizeof( PropUvTriKey ), CmpTriKey );
	if ( !found )
		return true;
	return (int)found->bin == bin;
}

void PropUvDegenerateOutsideBin( studiohdr_t *pHdr, OptimizedModel::FileHeader_t *pVtx, const PropUvSplitPlan &plan, int bin )
{
	if ( !pHdr || !pVtx || bin < 0 || plan.m_NumBins <= 1 )
		return;
	VisitModelTris( pHdr, pVtx, -1, NULL, [&]( const TriHit &hit )
	{
		if ( !hit.ix )
			return;
		if ( PropUvTriKept( plan, bin, hit.body, hit.model, hit.lod, hit.mesh, hit.stripGroup, hit.indexOffset ) )
			return;
		hit.ix[1] = hit.ix[0];
		hit.ix[2] = hit.ix[0];
	} );
}
