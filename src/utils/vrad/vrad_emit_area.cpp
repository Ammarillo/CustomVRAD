//========= Copyright PathRAD contributors. ============//
// Area lights from $vrad_emit* faces (triangle mesh sampling).
//=============================================================================//

#include "vrad_emit_area.h"
#include "vrad_emit.h"
#include "vrad.h"
#include "lightmap.h"
#include "bsplib.h"
#include "cmdlib.h"
#include "tier1/utlvector.h"
#include <math.h>

extern int EdgeVertex( dface_t *f, int edge );
extern float lightscale;

static CUtlVector<VRadEmitTri_t> g_EmitTris;
static CUtlVector<float> g_EmitCdf;		// inclusive prefix, normalized to 1
static CUtlVector<char> g_EmitFaceHasMesh;	// numfaces bytes
static float g_EmitPowerSum = 0.0f;

// Match prior $vrad_emit cast calibration (128^2 reference sheet).
static const float kEmitRefArea = 128.0f * 128.0f;
static const float kDirectScale = 100.0f * 100.0f;

void VRadEmitArea_Clear( void )
{
	g_EmitTris.RemoveAll();
	g_EmitCdf.RemoveAll();
	g_EmitFaceHasMesh.RemoveAll();
	g_EmitPowerSum = 0.0f;
}

static Vector EmitRadianceAt( int facenum, const Vector &pos )
{
	Vector e( 0, 0, 0 );
	if ( !VRadEmit_SampleAtPos( facenum, pos, e ) )
		return e;
	// L_e in GatherSampleLight-compatible units: emit already has lightscale once.
	// Extra lightscale + /kRef + DIRECT_SCALE mirrors previous point-proxy energy.
	VectorScale( e, lightscale * kDirectScale / kEmitRefArea, e );
	return e;
}

int VRadEmitArea_Build( void )
{
	VRadEmitArea_Clear();
	g_EmitFaceHasMesh.SetCount( numfaces );
	memset( g_EmitFaceHasMesh.Base(), 0, g_EmitFaceHasMesh.Count() );

	int nFaces = 0;
	for ( int facenum = 0; facenum < numfaces; ++facenum )
	{
		if ( !VRadEmit_FaceEmits( facenum ) )
			continue;

		dface_t *f = &g_pFaces[facenum];
		if ( f->numedges < 3 )
			continue;

		const Vector &off = face_offset[facenum];
		Vector v0;
		VectorAdd( dvertexes[EdgeVertex( f, 0 )].point, off, v0 );

		bool any = false;
		for ( int e = 1; e + 1 < f->numedges; ++e )
		{
			Vector v1, v2;
			VectorAdd( dvertexes[EdgeVertex( f, e )].point, off, v1 );
			VectorAdd( dvertexes[EdgeVertex( f, e + 1 )].point, off, v2 );

			Vector e1 = v1 - v0;
			Vector e2 = v2 - v0;
			Vector n;
			CrossProduct( e1, e2, n );
			const float twiceArea = VectorNormalize( n );
			if ( twiceArea < 1e-4f )
				continue;
			const float area = 0.5f * twiceArea;

			// Prefer plane normal (winding can flip); keep emission into the room.
			dplane_t *pl = &dplanes[f->planenum];
			Vector pn = pl->normal;
			if ( f->side )
				VectorSubtract( vec3_origin, pn, pn );
			if ( DotProduct( n, pn ) < 0.0f )
			{
				n = pn;
				// swap v1/v2 to keep winding with normal
				Vector tmp = v1; v1 = v2; v2 = tmp;
			}
			else
			{
				n = pn;
			}
			VectorNormalize( n );

			VRadEmitTri_t t;
			t.v0 = v0; t.v1 = v1; t.v2 = v2;
			t.area = area;
			t.facenum = facenum;
			// Face plane normal (emission into the open half-space).
			VectorCopy( dplanes[f->planenum].normal, t.n );
			VectorNormalize( t.n );
			if ( DotProduct( n, t.n ) < 0.0f )
			{
				// Fan winding opposite plane - swap to match.
				Vector tmp = t.v1; t.v1 = t.v2; t.v2 = tmp;
			}

			t.e0 = EmitRadianceAt( facenum, t.v0 );
			t.e1 = EmitRadianceAt( facenum, t.v1 );
			t.e2 = EmitRadianceAt( facenum, t.v2 );
			const float avgE = ( VectorAvg( t.e0 ) + VectorAvg( t.e1 ) + VectorAvg( t.e2 ) ) * ( 1.0f / 3.0f );
			t.power = max( avgE * area, 1e-12f );
			g_EmitTris.AddToTail( t );
			any = true;
		}
		if ( any )
		{
			g_EmitFaceHasMesh[facenum] = 1;
			++nFaces;
		}
	}

	g_EmitPowerSum = 0.0f;
	g_EmitCdf.SetCount( g_EmitTris.Count() );
	for ( int i = 0; i < g_EmitTris.Count(); ++i )
	{
		g_EmitPowerSum += g_EmitTris[i].power;
		g_EmitCdf[i] = g_EmitPowerSum;
	}
	if ( g_EmitPowerSum > 0.0f )
	{
		const float inv = 1.0f / g_EmitPowerSum;
		for ( int i = 0; i < g_EmitCdf.Count(); ++i )
			g_EmitCdf[i] *= inv;
	}

	if ( g_EmitTris.Count() > 0 )
	{
		Msg( "VRad emit area: %d tris on %d faces (PBRT area NEE, power=%.3g)\n",
			 g_EmitTris.Count(), nFaces, g_EmitPowerSum );
	}
	return g_EmitTris.Count();
}

int VRadEmitArea_TriCount( void )
{
	return g_EmitTris.Count();
}

const VRadEmitTri_t *VRadEmitArea_Tris( void )
{
	return g_EmitTris.Base();
}

float VRadEmitArea_PowerSum( void )
{
	return g_EmitPowerSum;
}

const float *VRadEmitArea_PowerCdf( void )
{
	return g_EmitCdf.Base();
}

bool VRadEmitArea_FaceHasMesh( int facenum )
{
	if ( facenum < 0 || facenum >= g_EmitFaceHasMesh.Count() )
		return false;
	return g_EmitFaceHasMesh[facenum] != 0;
}
