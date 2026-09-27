//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
// $NoKeywords: $
//
//=============================================================================//

#include "vrad.h"
#include "lightmap.h"
#include "radial.h"
#include "mathlib/bumpvects.h"
#include "utlrbtree.h"
#include "mathlib/VMatrix.h"
#include "macro_texture.h"
#include "ao.h"
#include "tier1/utlvector.h"
#include "pathtrace_dxr_device.h"

// Bounce luxel splat vs stock RADIALDIST. 1 = stock; <1 tighter; >1 wider.
float g_flBounceRadialScale = 1.0f;

// Cross-face bounce weld (opt-in via -bounce_weld): coplanar neighbor patches
// near a shared edge contribute with edge falloff. Off by default - culling
// distant coplanar neighbors removes stock GI bleed and blotches ceilings.
bool g_bBounceWeld = false;
static const float BOUNCE_WELD_COPLANAR_DOT = 0.985f;	// ~10 deg
static const float BOUNCE_WELD_EDGE_DIST = 24.0f;

static int RadialEdgeVertex( dface_t *f, int edge )
{
	if ( edge < 0 )
		edge += f->numedges;
	else if ( edge >= f->numedges )
		edge = edge % f->numedges;

	int k = dsurfedges[f->firstedge + edge];
	if ( k < 0 )
		return dedges[-k].v[1];
	return dedges[k].v[0];
}

static float DistPointToSegment( const Vector &p, const Vector &a, const Vector &b )
{
	Vector ab = b - a;
	float ab2 = DotProduct( ab, ab );
	if ( ab2 < 1e-6f )
	{
		Vector d = p - a;
		return sqrtf( DotProduct( d, d ) );
	}
	float t = DotProduct( p - a, ab ) / ab2;
	if ( t < 0.0f ) t = 0.0f;
	if ( t > 1.0f ) t = 1.0f;
	Vector closest = a + ab * t;
	Vector d = p - closest;
	return sqrtf( DotProduct( d, d ) );
}

// 0 = skip (far from shared edge on coplanar neighbor), 1 = full.
// Non-coplanar neighbors return 1 (keep stock neighbor bleed).
static float BounceWeldWeight( int facenum, int neighborFace, const Vector &patchOrigin, float patchRadius )
{
	if ( !g_bBounceWeld )
		return 1.0f;

	faceneighbor_t *fn = &faceneighbor[facenum];
	faceneighbor_t *fnN = &faceneighbor[neighborFace];

	float ndot = DotProduct( fn->facenormal, fnN->facenormal );
	if ( ndot < BOUNCE_WELD_COPLANAR_DOT )
		return 1.0f;

	dface_t *fA = &g_pFaces[facenum];
	dface_t *fB = &g_pFaces[neighborFace];

	float bestDist = 1e30f;
	bool bFoundShared = false;
	for ( int ea = 0; ea < fA->numedges; ++ea )
	{
		int va0 = RadialEdgeVertex( fA, ea );
		int va1 = RadialEdgeVertex( fA, ea + 1 );
		for ( int eb = 0; eb < fB->numedges; ++eb )
		{
			int vb0 = RadialEdgeVertex( fB, eb );
			int vb1 = RadialEdgeVertex( fB, eb + 1 );
			bool shared = ( va0 == vb0 && va1 == vb1 ) || ( va0 == vb1 && va1 == vb0 );
			if ( !shared )
				continue;
			bFoundShared = true;
			float d = DistPointToSegment( patchOrigin, dvertexes[va0].point, dvertexes[va1].point );
			if ( d < bestDist )
				bestDist = d;
		}
	}

	// No exact shared edge (T-junction etc.) - keep stock neighbor splat.
	if ( !bFoundShared )
		return 1.0f;

	// Falloff length scales with patch size so the nearest row of neighbor
	// patches always contributes (24 alone is below default chop spacing).
	float weldDist = BOUNCE_WELD_EDGE_DIST + patchRadius;
	if ( bestDist > weldDist )
		return 0.0f;

	float t = 1.0f - ( bestDist / weldDist );
	// smootherstep
	return t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
}


void WorldToLuxelSpace( lightinfo_t const *l, Vector const &world, Vector2D &coord )
{
	Vector pos;

	VectorSubtract( world, l->luxelOrigin, pos );
	coord[0] = DotProduct( pos, l->worldToLuxelSpace[0] ) - l->face->m_LightmapTextureMinsInLuxels[0];
	coord[1] = DotProduct( pos, l->worldToLuxelSpace[1] ) - l->face->m_LightmapTextureMinsInLuxels[1];
}

void LuxelSpaceToWorld( lightinfo_t const *l, float s, float t, Vector &world )
{
	Vector pos;

	s += l->face->m_LightmapTextureMinsInLuxels[0];
	t += l->face->m_LightmapTextureMinsInLuxels[1];
	VectorMA( l->luxelOrigin, s, l->luxelToWorldSpace[0], pos );
	VectorMA( pos, t, l->luxelToWorldSpace[1], world );
}

void WorldToLuxelSpace( lightinfo_t const *l, FourVectors const &world, FourVectors &coord )
{
	FourVectors luxelOrigin;
	luxelOrigin.DuplicateVector ( l->luxelOrigin );

	FourVectors pos = world;
	pos -= luxelOrigin;

	coord.x = pos * l->worldToLuxelSpace[0];
	coord.x = SubSIMD ( coord.x, ReplicateX4 ( l->face->m_LightmapTextureMinsInLuxels[0] ) );
	coord.y = pos * l->worldToLuxelSpace[1];
	coord.y = SubSIMD ( coord.y, ReplicateX4 ( l->face->m_LightmapTextureMinsInLuxels[1] ) );
	coord.z = Four_Zeros;
}

void LuxelSpaceToWorld( lightinfo_t const *l, fltx4 s, fltx4 t, FourVectors &world )
{
	world.DuplicateVector ( l->luxelOrigin );
	FourVectors st;

	s = AddSIMD ( s, ReplicateX4 ( l->face->m_LightmapTextureMinsInLuxels[0] ) );
	st.DuplicateVector ( l->luxelToWorldSpace[0] );
	st *= s;
	world += st;

	t = AddSIMD ( t, ReplicateX4 ( l->face->m_LightmapTextureMinsInLuxels[1] ) );
	st.DuplicateVector ( l->luxelToWorldSpace[1] );
	st *= t;
	world += st;
}



void AddDirectToRadial( radial_t *rad, 
						Vector const &pnt, 
						Vector2D const &coordmins, Vector2D const &coordmaxs, 
						LightingValue_t const light[NUM_BUMP_VECTS+1],
						bool hasBumpmap, bool neighborHasBumpmap  )
{
	int     s_min, s_max, t_min, t_max;
	Vector2D  coord;
	int	    s, t;
	float   ds, dt;
	float   r;
	float	area;
	int		bumpSample;

	// convert world pos into local lightmap texture coord
	WorldToLuxelSpace( &rad->l, pnt, coord );

	s_min = ( int )( coordmins[0] );
	t_min = ( int )( coordmins[1] );
	s_max = ( int )( coordmaxs[0] + 0.9999f ) + 1; // ????
	t_max = ( int )( coordmaxs[1] + 0.9999f ) + 1;

	s_min = max( s_min, 0 );
	t_min = max( t_min, 0 );
	s_max = min( s_max, rad->w );
	t_max = min( t_max, rad->h );

	for( s = s_min; s < s_max; s++ )
	{
		for( t = t_min; t < t_max; t++ )
		{
			float s0 = max( coordmins[0] - s, -1.0 );
			float t0 = max( coordmins[1] - t, -1.0 );
			float s1 = min( coordmaxs[0] - s, 1.0 );
			float t1 = min( coordmaxs[1] - t, 1.0 );

			area = (s1 - s0) * (t1 - t0);

			if (area > EQUAL_EPSILON)
			{
				ds = fabs( coord[0] - s );
				dt = fabs( coord[1] - t );

				r = max( ds, dt );

				if (r < 0.1)
				{
					r = area / 0.1;
				}
				else
				{
					r = area / r;
				}

				int i = s+t*rad->w;

				if( hasBumpmap )
				{
					if( neighborHasBumpmap )
					{
						for( bumpSample = 0; bumpSample < NUM_BUMP_VECTS + 1; bumpSample++ )
						{
							rad->light[bumpSample][i].AddWeighted( light[bumpSample], r );
						}
					}
					else
					{
						rad->light[0][i].AddWeighted(light[0],r );
						for( bumpSample = 1; bumpSample < NUM_BUMP_VECTS + 1; bumpSample++ )
						{
							rad->light[bumpSample][i].AddWeighted( light[0], r * OO_SQRT_3 );
						}
					}
				}
				else
				{
					rad->light[0][i].AddWeighted( light[0], r );
				}
				
				rad->weight[i] += r;
			}
		}
	}
}



void AddBouncedToRadial( radial_t *rad, 
						 Vector const &pnt, 
						 Vector2D const &coordmins, Vector2D const &coordmaxs, 
						 Vector const light[NUM_BUMP_VECTS+1],
						 bool hasBumpmap, bool neighborHasBumpmap,
						 float flWeightScale = 1.0f )
{
	int     s_min, s_max, t_min, t_max;
	Vector2D  coord;
	int	    s, t;
	float   ds, dt;
	float   r;
	int		bumpSample;

	// convert world pos into local lightmap texture coord
	WorldToLuxelSpace( &rad->l, pnt, coord );

	float dists, distt;

	dists = (coordmaxs[0] - coordmins[0]);
	distt = (coordmaxs[1] - coordmins[1]);

	// patches less than a luxel in size could be mistakeningly filtered, so clamp.
	dists = max( 1.0, dists );
	distt = max( 1.0, distt );

	// Tighter than stock reduces overlapping patch GI (wider splat made seams worse).
	float bounceScale = g_flBounceRadialScale;
	if ( bounceScale < 0.5f )
		bounceScale = 0.5f;
	if ( bounceScale > 4.0f )
		bounceScale = 4.0f;
	const float bounceDist = RADIALDIST * bounceScale;
	const float bounceDist2 = bounceDist * bounceDist;

	// find possible domain of patch influence
  	s_min = ( int )( coord[0] - dists * bounceDist );
  	t_min = ( int )( coord[1] - distt * bounceDist );
  	s_max = ( int )( coord[0] + dists * bounceDist + 1.0f );
  	t_max = ( int )( coord[1] + distt * bounceDist + 1.0f );

	// clamp to valid luxel
	s_min = max( s_min, 0 );
	t_min = max( t_min, 0 );
	s_max = min( s_max, rad->w );
	t_max = min( t_max, rad->h );

	for( s = s_min; s < s_max; s++ )
	{
		for( t = t_min; t < t_max; t++ )
		{
			// patch influence is based on patch size
  			ds = ( coord[0] - s ) / dists;
  			dt = ( coord[1] - t ) / distt;
  
  			r = bounceDist2 - (ds * ds + dt * dt);

			int i = s+t*rad->w;
   
  			if (r > 0)
			{
				// Weld falloff scales both light and weight so the normalized
				// result blends toward own-face patches instead of darkening.
				r *= flWeightScale;

				if( hasBumpmap )
				{
					if( neighborHasBumpmap )
					{
						for( bumpSample = 0; bumpSample < NUM_BUMP_VECTS + 1; bumpSample++ )
						{
							rad->light[bumpSample][i].AddWeighted( light[bumpSample], r );
						}
					}
					else
					{
						rad->light[0][i].AddWeighted( light[0], r );
						for( bumpSample = 1; bumpSample < NUM_BUMP_VECTS + 1; bumpSample++ )
						{
							rad->light[bumpSample][i].AddWeighted( light[0], r * OO_SQRT_3 );
						}
					}
				}
				else
				{
					rad->light[0][i].AddWeighted( light[0], r );
				}
				
				rad->weight[i] += r;
			}
		}
	}
}

void PatchLightmapCoordRange( radial_t *rad, int ndxPatch, Vector2D &mins, Vector2D &maxs )
{
	winding_t	*w;
	int i;
	Vector2D coord;

	mins.Init( 1E30, 1E30 );
	maxs.Init( -1E30, -1E30 );

	CPatch *patch = &g_Patches.Element( ndxPatch );
	w = patch->winding;

	for (i = 0; i < w->numpoints; i++)
	{
		WorldToLuxelSpace( &rad->l, w->p[i], coord );
		mins[0] = min( mins[0], coord[0] );
		maxs[0] = max( maxs[0], coord[0] );
		mins[1] = min( mins[1], coord[1] );
		maxs[1] = max( maxs[1], coord[1] );
	}
}

radial_t *AllocateRadial( int facenum )
{
	radial_t *rad;

	rad = ( radial_t* )calloc( 1, sizeof( *rad ) );

	rad->facenum = facenum;
	InitLightinfo( &rad->l, facenum );

	rad->w = rad->l.face->m_LightmapTextureSizeInLuxels[0]+1;
	rad->h = rad->l.face->m_LightmapTextureSizeInLuxels[1]+1;

	return rad;
}

void FreeRadial( radial_t *rad )
{
	if (rad)
		free( rad );
}


radial_t *BuildPatchRadial( int facenum )
{
	int             j;
	radial_t        *rad;
	CPatch	        *patch;
	faceneighbor_t  *fn;
	Vector2D        mins, maxs;
	bool			needsBumpmap, neighborNeedsBumpmap;

	needsBumpmap = texinfo[g_pFaces[facenum].texinfo].flags & SURF_BUMPLIGHT ? true : false;

	rad = AllocateRadial( facenum );
	
	fn = &faceneighbor[ rad->facenum ];

	CPatch *pNextPatch;

	if( g_FacePatches.Element( rad->facenum ) != g_FacePatches.InvalidIndex() )
	{
		for( patch = &g_Patches.Element( g_FacePatches.Element( rad->facenum ) ); patch; patch = pNextPatch )
		{
			// next patch
			pNextPatch = NULL;
			if( patch->ndxNext != g_Patches.InvalidIndex() )
			{
				pNextPatch = &g_Patches.Element( patch->ndxNext );
			}
			
			// skip patches with children
			if (patch->child1 != g_Patches.InvalidIndex() )
				continue;
			
			// get the range of patch lightmap texture coords
			int ndxPatch = patch - g_Patches.Base();
			PatchLightmapCoordRange( rad, ndxPatch, mins, maxs );
			
			if (patch->numtransfers == 0)
			{
				// Error, using patch that was never evaluated or has no samples
				// patch->totallight[1] = 255;
			}
			
			//
			// displacement surface patch origin position and normal vectors have been changed to
			// represent the displacement surface position and normal -- for radial "blending"
			// we need to get the base surface patch origin!
			//
			if( ValidDispFace( &g_pFaces[facenum] ) )
			{
				Vector patchOrigin;
				WindingCenter (patch->winding, patchOrigin );
				AddBouncedToRadial( rad, patchOrigin, mins, maxs, patch->totallight.light,  
					needsBumpmap, needsBumpmap );			
			}
			else
			{
				AddBouncedToRadial( rad, patch->origin, mins, maxs, patch->totallight.light, 
					needsBumpmap, needsBumpmap );
			}
		}
	}

	for (j=0 ; j<fn->numneighbors; j++)
	{
		const int neighborFace = fn->neighbor[j];
		if( g_FacePatches.Element( neighborFace ) != g_FacePatches.InvalidIndex() )
		{
			for( patch = &g_Patches.Element( g_FacePatches.Element( neighborFace ) ); patch; patch = pNextPatch )
			{
				// next patch
				pNextPatch = NULL;
				if( patch->ndxNext != g_Patches.InvalidIndex() )
				{
					pNextPatch = &g_Patches.Element( patch->ndxNext );
				}
				
				// skip patches with children
				if (patch->child1 != g_Patches.InvalidIndex() )
					continue;
				
				// get the range of patch lightmap texture coords
				int ndxPatch = patch - g_Patches.Base();
				PatchLightmapCoordRange( rad, ndxPatch, mins, maxs  );
				
				neighborNeedsBumpmap = texinfo[g_pFaces[facenum].texinfo].flags & SURF_BUMPLIGHT ? true : false;

				Vector patchOrigin;
				if( ValidDispFace( &g_pFaces[neighborFace] ) )
					WindingCenter( patch->winding, patchOrigin );
				else
					patchOrigin = patch->origin;

				Vector patchExtent = patch->maxs - patch->mins;
				float patchRadius = 0.5f * patchExtent.Length();
				float weld = BounceWeldWeight( facenum, neighborFace, patchOrigin, patchRadius );
				if ( weld <= 1e-4f )
					continue;

				AddBouncedToRadial( rad, patchOrigin, mins, maxs, patch->totallight.light,
					needsBumpmap, needsBumpmap, weld );
			}
		}
	}

	return rad;
}


radial_t *BuildLuxelRadial( int facenum, int style )
{
	LightingValue_t light[NUM_BUMP_VECTS + 1];

	facelight_t *fl = &facelight[facenum];
	faceneighbor_t *fn = &faceneighbor[facenum];

	radial_t *rad = AllocateRadial( facenum );

	bool needsBumpmap = texinfo[g_pFaces[facenum].texinfo].flags & SURF_BUMPLIGHT ? true : false;

	for (int k=0 ; k<fl->numsamples ; k++)
	{
		if( needsBumpmap )
		{
			for( int bumpSample = 0; bumpSample < NUM_BUMP_VECTS + 1; bumpSample++ )
			{
				light[bumpSample] = fl->light[style][bumpSample][k];
			}
		}
		else
		{
			light[0] = fl->light[style][0][k];
		}

		AddDirectToRadial( rad, fl->sample[k].pos, fl->sample[k].mins, fl->sample[k].maxs, light, needsBumpmap, needsBumpmap );
	}

	for (int j = 0; j < fn->numneighbors; j++)
	{
		fl = &facelight[fn->neighbor[j]];

		bool neighborHasBumpmap = false;
		
		if( texinfo[g_pFaces[fn->neighbor[j]].texinfo].flags & SURF_BUMPLIGHT )
		{
			neighborHasBumpmap = true;
		}

		int nstyle = 0;

		// look for style that matches
		if (g_pFaces[fn->neighbor[j]].styles[nstyle] != g_pFaces[facenum].styles[style])
		{
			for (nstyle = 1; nstyle < MAXLIGHTMAPS; nstyle++ )
				if ( g_pFaces[fn->neighbor[j]].styles[nstyle] == g_pFaces[facenum].styles[style] )
					break;

			// if not found, skip this neighbor
			if (nstyle >= MAXLIGHTMAPS)
				continue;
		}

		lightinfo_t l;

		InitLightinfo( &l, fn->neighbor[j] );

		for (int k=0 ; k<fl->numsamples ; k++)
		{
			if( neighborHasBumpmap )
			{
				for( int bumpSample = 0; bumpSample < NUM_BUMP_VECTS + 1; bumpSample++ )
				{
					light[bumpSample] = fl->light[nstyle][bumpSample][k];
				}
			}
			else
			{
				light[0]=fl->light[nstyle][0][k];
			}

			Vector tmp;
			Vector2D mins, maxs;

			LuxelSpaceToWorld( &l, fl->sample[k].mins[0], fl->sample[k].mins[1], tmp );
			WorldToLuxelSpace( &rad->l, tmp, mins );
			LuxelSpaceToWorld( &l, fl->sample[k].maxs[0], fl->sample[k].maxs[1], tmp );
			WorldToLuxelSpace( &rad->l, tmp, maxs );

			AddDirectToRadial( rad, fl->sample[k].pos, mins, maxs, light,
				         needsBumpmap, neighborHasBumpmap );
		}
	}

	return rad;
}


//-----------------------------------------------------------------------------
// Purpose: returns the closest light value for a given point on the surface
//			this is normally a 1:1 mapping
//-----------------------------------------------------------------------------
bool SampleRadial( radial_t *rad, Vector& pnt, LightingValue_t light[NUM_BUMP_VECTS + 1], int bumpSampleCount )
{
	int bumpSample;
	Vector2D coord;

	WorldToLuxelSpace( &rad->l, pnt, coord );
	int u = ( int )( coord[0] + 0.5f );
	int v = ( int )( coord[1] + 0.5f );
	// Stock used `u > w` / `v > h`, so u==w indexed the next row (or past the
	// allocation). That dropped energy on the last luxel of each edge/corner.
	if ( rad->w <= 0 || rad->h <= 0 )
		return false;
	if ( u < 0 ) u = 0;
	if ( v < 0 ) v = 0;
	if ( u >= rad->w ) u = rad->w - 1;
	if ( v >= rad->h ) v = rad->h - 1;
	int i = u + v * rad->w;

	bool baseSampleOk = true;
	for( bumpSample = 0; bumpSample < bumpSampleCount; bumpSample++ )
	{
		light[bumpSample].Zero();

		if (rad->weight[i] > WEIGHT_EPS)
		{
			light[bumpSample]= rad->light[bumpSample][i];
			light[bumpSample].Scale( 1.0 / rad->weight[i] );
		}
		else
		{
			if ( bRed2Black )
			{
				// Error, luxel has no samples
				light[bumpSample].m_vecLighting.Init( 0, 0, 0 );
			}
			else
			{
				// Error, luxel has no samples
				// Yes, it actually should be 2550
				light[bumpSample].m_vecLighting.Init( 2550, 0, 0 );
			}

			if (bumpSample == 0)
				baseSampleOk = false;
		}
	}

	return baseSampleOk;
}

bool FloatLess( float const& src1, float const& src2 )
{
	return src1 < src2;
}


//-----------------------------------------------------------------------------
// Debugging!
//-----------------------------------------------------------------------------
void GetRandomColor( unsigned char *color )
{
	static bool firstTime = true;
				
	if( firstTime )
	{
		firstTime = false;
		srand( 0 );
	}
	
	color[0] = ( unsigned char )( rand() * ( 255.0f / VALVE_RAND_MAX ) ); 
	color[1] = ( unsigned char )( rand() * ( 255.0f / VALVE_RAND_MAX ) ); 
	color[2] = ( unsigned char )( rand() * ( 255.0f / VALVE_RAND_MAX ) ); 
}


#if 0
// debugging! -- not accurate!
void DumpLuxels( facelight_t *pFaceLight, Vector *luxelColors, int ndxFace )
{
	static FileHandle_t pFpLuxels = NULL;

	ThreadLock();

	if( !pFpLuxels )
	{
		pFpLuxels = g_pFileSystem->Open( "luxels.txt", "w" );
	}

	dface_t *pFace = &g_pFaces[ndxFace];
	bool bDisp = ( pFace->dispinfo != -1 );

	for( int ndx = 0; ndx < pFaceLight->numluxels; ndx++ )
	{
		WriteWinding( pFpLuxels, pFaceLight->sample[ndx].w, luxelColors[ndx] );
		if( bDumpNormals && bDisp )
		{
			WriteNormal( pFpLuxels, pFaceLight->luxel[ndx], pFaceLight->luxelNormals[ndx], 15.0f, Vector( 255, 255, 0 ) );
		}
	}

	ThreadUnlock();
}
#endif


static FileHandle_t pFileLuxels[4] = { NULL, NULL, NULL, NULL };

void DumpDispLuxels( int iFace, Vector &color, int iLuxel, int nBump )
{
	// Lock the thread and dump the luxel data.
	ThreadLock();

	// Get the face and facelight data.
	facelight_t *pFaceLight = &facelight[iFace];

	// Open the luxel files.
	char szFileName[512];
	for ( int iBump = 0; iBump < ( NUM_BUMP_VECTS+1 ); ++iBump )
	{
		if ( pFileLuxels[iBump] == NULL )
		{
			sprintf( szFileName, "luxels_bump%d.txt", iBump );
			pFileLuxels[iBump] = g_pFileSystem->Open( szFileName, "w" );
		}
	}

	WriteWinding( pFileLuxels[nBump], pFaceLight->sample[iLuxel].w, color );

	ThreadUnlock();
}

void CloseDispLuxels()
{
	for ( int iBump = 0; iBump < ( NUM_BUMP_VECTS+1 ); ++iBump )
	{
		if ( pFileLuxels[iBump] )
		{
			g_pFileSystem->Close( pFileLuxels[iBump] );
		}
	}
}

static float ShadowLeakLuxelLuma( const Vector &v )
{
	return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

void CompressBilinearShadowLeak( Vector *grid, int width, int height )
{
	if ( !grid || width < 2 || height < 2 )
		return;

	const int n = width * height;
	CUtlVector<Vector> src;
	src.SetCount( n );
	for ( int i = 0; i < n; ++i )
		src[i] = grid[i];

	static const int kDx[4] = { 0, 0, -1, 1 };
	static const int kDy[4] = { -1, 1, 0, 0 };

	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			const float L = ShadowLeakLuxelLuma( src[idx] );

			float maxN = 0.0f;
			float minN = 1e9f;
			int nCard = 0;
			for ( int k = 0; k < 4; ++k )
			{
				const int ns = s + kDx[k];
				const int nt = t + kDy[k];
				if ( ns < 0 || ns >= width || nt < 0 || nt >= height )
					continue;
				const float ln = ShadowLeakLuxelLuma( src[ns + nt * width] );
				maxN = max( maxN, ln );
				minN = min( minN, ln );
				++nCard;
			}

			if ( nCard < 2 || maxN < 0.02f )
				continue;

			const float edgeRatio = maxN / max( minN, 1e-4f );
			if ( edgeRatio < 3.0f )
				continue;

			// Lit-side boundary luxel undervalued by filters: small capped lift only.
			if ( L > minN * 1.6f && L < maxN * 0.62f )
			{
				const float targetL = min( maxN * 0.90f, L * 1.18f );
				if ( targetL > L * 1.03f )
				{
					const float scale = targetL / max( L, 1e-4f );
					grid[idx] = src[idx] * scale;
				}
				continue;
			}

			// Isolated firefly on a hard edge or open face.
			if ( L > maxN * 2.5f && L > 0.08f && maxN > 0.01f )
			{
				grid[idx] = src[idx] * ( ( maxN * 1.15f ) / max( L, 1e-4f ) );
			}
		}
	}
}

static void LiftPackedBorderLuxels( byte *pStyleBase, int width, int height, int bumpCount, int facenum )
{
	if ( !pStyleBase || width < 3 || height < 3 || bumpCount < 1 )
		return;

	const int n = width * height;
	CUtlVector<Vector> grid;
	grid.SetCount( n );

	for ( int bump = 0; bump < bumpCount; ++bump )
	{
		byte *p = pStyleBase + bump * n * 4;
		for ( int i = 0; i < n; ++i )
			ColorRGBExp32ToVector( *( ColorRGBExp32 * )( p + i * 4 ), grid[i] );
		ReconstructLightmapEdgeLuxels( grid.Base(), width, height, facenum );
		CompressBilinearShadowLeak( grid.Base(), width, height );
		for ( int i = 0; i < n; ++i )
			VectorToColorRGBExp32( grid[i], *( ColorRGBExp32 * )( p + i * 4 ) );
	}
}

/*
=============
FinalLightFace

Add the indirect lighting on top of the direct
lighting and save into final map format
=============
*/
void FinalLightFace( int iThread, int facenum )
{
	dface_t	        *f;
	int		        i, j, k;
	facelight_t	    *fl;
	float		    minlight;
	int			    lightstyles;
	LightingValue_t lb[NUM_BUMP_VECTS + 1], v[NUM_BUMP_VECTS + 1];
	unsigned char   *pdata[NUM_BUMP_VECTS + 1];
	int				bumpSample;
	radial_t	    *rad = NULL;
	radial_t	    *prad = NULL;

   	f = &g_pFaces[facenum];

    // test for non-lit texture
    if ( texinfo[f->texinfo].flags & TEX_SPECIAL)
        return;		

	fl = &facelight[facenum];


	for (lightstyles=0; lightstyles < MAXLIGHTMAPS; lightstyles++ )
	{
		if ( f->styles[lightstyles] == 255 )
			break;
	}
	if ( !lightstyles )
		return;

	
	//
	// sample the triangulation
	//
	minlight = FloatForKey (face_entity[facenum], "_minlight") * 128;

	bool needsBumpmap = ( texinfo[f->texinfo].flags & SURF_BUMPLIGHT ) ? true : false;
	int bumpSampleCount = needsBumpmap ? NUM_BUMP_VECTS + 1 : 1;

	bool bDisp = ( f->dispinfo != -1 );

//#define RANDOM_COLOR

#ifdef RANDOM_COLOR
	unsigned char randomColor[3];
	GetRandomColor( randomColor );
#endif

	// Optional AO: compute once per face (style-independent geometry occlusion).
	CUtlVector<float> aoFactors;
	if ( AO_ShouldApplyToLightmaps() && fl->numluxels > 0 )
	{
		aoFactors.SetCount( fl->numluxels );
		ComputeFaceAmbientOcclusion( facenum, fl, aoFactors.Base() );
	}

	// NOTE: I'm using these RB trees to sort all the illumination values
	// to compute median colors. Turns out that this is a somewhat better
	// method that using the average; usually if there are surfaces
	// with a large light intensity variation, the extremely bright regions
	// have a very small area and tend to influence the average too much.
	CUtlRBTree< float, int >	m_Red( 0, 256, FloatLess );
	CUtlRBTree< float, int >	m_Green( 0, 256, FloatLess );
	CUtlRBTree< float, int >	m_Blue( 0, 256, FloatLess );

	for (k=0 ; k < lightstyles; k++ )
	{
		m_Red.RemoveAll();
		m_Green.RemoveAll();
		m_Blue.RemoveAll();

		if (!do_fast)
		{
			if( !bDisp )
			{
				rad = BuildLuxelRadial( facenum, k );
			}
			else
			{
				rad = StaticDispMgr()->BuildLuxelRadial( facenum, k, needsBumpmap );
			}
		}

		if (numbounce > 0 && k == 0)
		{
			// currently only radiosity light non-displacement surfaces!
			if( !bDisp )
			{
				prad = BuildPatchRadial( facenum );
			}
			else
			{
				prad = StaticDispMgr()->BuildPatchRadial( facenum, needsBumpmap );
			}
		}

		// pack the nonbump texture and the three bump texture for the given 
		// lightstyle right next to each other.
		// NOTE: Even though it's building positions for all bump-mapped data,
		// it isn't going to use those positions (see loop over bumpSample below)
		// The file offset is correctly computed to only store space for 1 set
		// of light data if we don't have bumped lighting.
		for( bumpSample = 0; bumpSample < bumpSampleCount; ++bumpSample )
		{
			pdata[bumpSample] = &(*pdlightdata)[f->lightofs + (k * bumpSampleCount + bumpSample) * fl->numluxels*4]; 
		}

		// Compute the average luxel color, but not for the bump samples
		Vector avg( 0.0f, 0.0f, 0.0f );
		int avgCount = 0;

		for (j=0 ; j<fl->numluxels; j++)
		{
			// garymct - direct lighting
			bool baseSampleOk = true;

			if (!do_fast)
			{
				if( !bDisp )
				{
					baseSampleOk = SampleRadial( rad, fl->luxel[j], lb, bumpSampleCount );
				}
				else
				{
					baseSampleOk = StaticDispMgr()->SampleRadial( facenum, rad, fl->luxel[j], j, lb, bumpSampleCount, false );
				}
			}
			else
			{
				for ( int iBump = 0 ; iBump < bumpSampleCount; iBump++ )
				{
					lb[iBump] = fl->light[0][iBump][j];
				}
			}

			if (prad)
			{
				// garymct - bounced light
				// v is indirect light that is received on the luxel.
				if( !bDisp )
				{
					SampleRadial( prad, fl->luxel[j], v, bumpSampleCount );
				}
				else
				{
					StaticDispMgr()->SampleRadial( facenum, prad, fl->luxel[j], j, v, bumpSampleCount, true );
				}

				for( bumpSample = 0; bumpSample < bumpSampleCount; ++bumpSample )
				{
					lb[bumpSample].AddLight( v[bumpSample] );
				}
			}

			if ( AO_ShouldApplyToLightmaps() && aoFactors.Count() == fl->numluxels )
			{
				float aoScale = AOScaleFromFactor( aoFactors[j], fl->luxel[j] );
				for ( bumpSample = 0; bumpSample < bumpSampleCount; ++bumpSample )
				{
					lb[bumpSample].m_vecLighting *= aoScale;
				}
			}

			if ( bDisp && g_bDumpPatches )
			{
				for( bumpSample = 0; bumpSample < bumpSampleCount; ++bumpSample )
				{
					DumpDispLuxels( facenum, lb[bumpSample].m_vecLighting, j, bumpSample );
				}
			}

			if (fl->numsamples == 0)
			{
				for( i = 0; i < bumpSampleCount; i++ )
				{
					lb[i].Init( 255, 0, 0 );
				}
				baseSampleOk = false;
			}

			int bumpSample;
			for( bumpSample = 0; bumpSample < bumpSampleCount; bumpSample++ )
			{
				// clip from the bottom first
				// garymct: minlight is a per entity minimum light value?
				for( i=0; i<3; i++ )
				{
					lb[bumpSample].m_vecLighting[i] = max( lb[bumpSample].m_vecLighting[i], minlight );
				}
				
				// Do the average light computation, I'm assuming (perhaps incorrectly?)
				// that all luxels in a particular lightmap have the same area here. 
				// Also, don't bother doing averages for the bump samples. Doing it here 
				// because of the minlight clamp above + the random color testy thingy. 
				// Also have to do it before Vec3toColorRGBExp32 because it 
				// destructively modifies lb[bumpSample] (Feh!)
				if ((bumpSample == 0) && baseSampleOk)
				{
					++avgCount;

					ApplyMacroTextures( facenum, fl->luxel[j], lb[0].m_vecLighting );

					// For median computation
					m_Red.Insert( lb[bumpSample].m_vecLighting[0] );
					m_Green.Insert( lb[bumpSample].m_vecLighting[1] );
					m_Blue.Insert( lb[bumpSample].m_vecLighting[2] );
				}

#ifdef RANDOM_COLOR
				pdata[bumpSample][0] = randomColor[0] / ( bumpSample + 1 );
				pdata[bumpSample][1] = randomColor[1] / ( bumpSample + 1 );
				pdata[bumpSample][2] = randomColor[2] / ( bumpSample + 1 );
				pdata[bumpSample][3] = 0;
#else
				// convert to a 4 byte r,g,b,signed exponent format
				VectorToColorRGBExp32( Vector( lb[bumpSample].m_vecLighting.x, lb[bumpSample].m_vecLighting.y,
											   lb[bumpSample].m_vecLighting.z ), *( ColorRGBExp32 *)pdata[bumpSample] );
#endif

				pdata[bumpSample] += 4;
			}
		}
		FreeRadial( rad );
		if (prad)
		{
			FreeRadial( prad );
			prad = NULL;
		}

		// Last lightmap row/column sits in the geometric fold on concave edges.
		// Dilate those dark outliers from the first stable interior luxel.
		{
			const int width = f->m_LightmapTextureSizeInLuxels[0] + 1;
			const int height = f->m_LightmapTextureSizeInLuxels[1] + 1;
			if ( width >= 2 && height >= 2 && width * height == fl->numluxels )
			{
				byte *pStyle = &(*pdlightdata)[f->lightofs + k * bumpSampleCount * fl->numluxels * 4];
				LiftPackedBorderLuxels( pStyle, width, height, bumpSampleCount, facenum );
			}
		}

		// Compute the median color for this lightstyle
		// Remember, the data goes *before* the specified light_ofs, in *reverse order*
		ColorRGBExp32 *pAvgColor = dface_AvgLightColor( f, k );
		if (avgCount == 0)
		{
			Vector median( 0, 0, 0 );
			VectorToColorRGBExp32( median, *pAvgColor );
		}
		else
		{
			unsigned int r, g, b;
			r = m_Red.FirstInorder();
			g = m_Green.FirstInorder();
			b = m_Blue.FirstInorder();
			avgCount >>= 1;
			while (avgCount > 0)
			{
				r = m_Red.NextInorder(r);
				g = m_Green.NextInorder(g);
				b = m_Blue.NextInorder(b);
				--avgCount;
			}

			Vector median( m_Red[r], m_Green[g], m_Blue[b] );
			VectorToColorRGBExp32( median, *pAvgColor );
		}
	}
}


//-----------------------------------------------------------------------------
// Lightmap seam stitching
//
// Every face gets its final lightmap filtered/normalized independently, so two
// coplanar faces split by VBSP end up with slightly different luxel values
// along their shared edge (a faint brightness step, also present in stock
// VRAD). After FinalLightFace, blend luxels near each shared edge toward the
// value the neighbor face has at the same world position; at the edge itself
// both sides converge to the same average, removing the seam.
//
// Shared edges are matched geometrically (collinear overlap), so T-junctions
// and duplicated verts stitch the same as identical dvertex indices.
//-----------------------------------------------------------------------------

static const float STITCH_COPLANAR_DOT = 0.999f;	// ~2.5 degrees
static const float STITCH_LUXEL_RANGE = 1.75f;		// stitch band, in luxels from the shared edge
static const int   STITCH_MAX_EDGES = 32;			// shared segments per face pair
static const float STITCH_EDGE_DIST_EPS = 0.5f;	// world-unit tolerance for collinear overlap
static const float STITCH_EDGE_PARALLEL = 0.99f;	// |dot| of unit edge dirs
static const float STITCH_EDGE_MIN_OVERLAP = 0.5f;	// ignore tiny accidental overlaps (WU)

static float DistPointToSegment2D( float px, float py, const Vector2D &a, const Vector2D &b )
{
	float abx = b.x - a.x, aby = b.y - a.y;
	float ab2 = abx * abx + aby * aby;
	float t = 0.0f;
	if ( ab2 > 1e-6f )
	{
		t = ( ( px - a.x ) * abx + ( py - a.y ) * aby ) / ab2;
		t = clamp( t, 0.0f, 1.0f );
	}
	float dx = px - ( a.x + abx * t );
	float dy = py - ( a.y + aby * t );
	return sqrtf( dx * dx + dy * dy );
}

static float StitchDistPointToLine( const Vector &p, const Vector &o, const Vector &dir )
{
	Vector d = p - o;
	Vector perp = d - dir * DotProduct( d, dir );
	return perp.Length();
}

// True if A and B are nearly collinear and overlap; writes the overlapping
// segment in world space. Matches T-junctions / duplicated verts, not just
// identical dvertex indices.
static bool StitchEdgesOverlap( const Vector &a0, const Vector &a1,
								const Vector &b0, const Vector &b1,
								Vector &outStart, Vector &outEnd )
{
	Vector da = a1 - a0;
	float lenA = da.Length();
	if ( lenA < STITCH_EDGE_MIN_OVERLAP )
		return false;

	Vector db = b1 - b0;
	float lenB = db.Length();
	if ( lenB < STITCH_EDGE_MIN_OVERLAP )
		return false;

	Vector dira = da * ( 1.0f / lenA );
	Vector dirb = db * ( 1.0f / lenB );
	if ( fabsf( DotProduct( dira, dirb ) ) < STITCH_EDGE_PARALLEL )
		return false;

	if ( StitchDistPointToLine( b0, a0, dira ) > STITCH_EDGE_DIST_EPS ||
		 StitchDistPointToLine( b1, a0, dira ) > STITCH_EDGE_DIST_EPS ||
		 StitchDistPointToLine( a0, b0, dirb ) > STITCH_EDGE_DIST_EPS ||
		 StitchDistPointToLine( a1, b0, dirb ) > STITCH_EDGE_DIST_EPS )
		return false;

	float t0 = DotProduct( b0 - a0, dira );
	float t1 = DotProduct( b1 - a0, dira );
	float tMin = min( t0, t1 );
	float tMax = max( t0, t1 );
	float overlapStart = max( 0.0f, tMin );
	float overlapEnd = min( lenA, tMax );
	if ( overlapEnd - overlapStart < STITCH_EDGE_MIN_OVERLAP )
		return false;

	outStart = a0 + dira * overlapStart;
	outEnd = a0 + dira * overlapEnd;
	return true;
}

static float EdgeLuxelLuma( const Vector &v )
{
	return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

// Which lightmap rectangle borders a luxel-space segment sits on.
// bit0=s0, bit1=sMax, bit2=t0, bit3=tMax.
static int LuxelSegCreaseBorderMask( const Vector2D &a, const Vector2D &b, int w, int h )
{
	// Stay on one side of the face. A flat 2-luxel tolerance on a 3-luxel face
	// would flag the opposite border from the same edge.
	float eps = 2.0f;
	const float limit = 0.45f * (float)min( w, h );
	if ( eps > limit )
		eps = limit;
	if ( eps < 0.75f )
		eps = 0.75f;
	const float sMin = min( a.x, b.x ), sMax = max( a.x, b.x );
	const float tMin = min( a.y, b.y ), tMax = max( a.y, b.y );
	int mask = 0;
	if ( sMax <= eps )
		mask |= 1;
	if ( sMin >= (float)( w - 1 ) - eps )
		mask |= 2;
	if ( tMax <= eps )
		mask |= 4;
	if ( tMin >= (float)( h - 1 ) - eps )
		mask |= 8;
	return mask;
}

// Vertex positions quantized to 4-unit cells so a 90° neighbor is found even
// when VBSP did not weld the shared corner (duplicated verts / T-junctions).
enum { kCreaseHashBuckets = 2048 };

struct CreaseHashNode_t
{
	unsigned __int64 key;
	int face;
	int next;
};

static int g_CreaseHashHead[kCreaseHashBuckets];
static CUtlVector<CreaseHashNode_t> g_CreaseHashNodes;
static bool g_CreaseHashReady = false;

static unsigned __int64 CreaseCellKey( int x, int y, int z )
{
	unsigned __int64 ux = (unsigned)x & 0x1FFFFFu;
	unsigned __int64 uy = (unsigned)y & 0x1FFFFFu;
	unsigned __int64 uz = (unsigned)z & 0x1FFFFFu;
	return ux | ( uy << 21 ) | ( uz << 42 );
}

static int CreaseCellBucket( unsigned __int64 key )
{
	unsigned h = (unsigned)key ^ (unsigned)( key >> 21 ) ^ (unsigned)( key >> 42 );
	h *= 0x9E3779B9u;
	return (int)( h & ( kCreaseHashBuckets - 1 ) );
}

static void CreaseQuantize( const Vector &p, int &x, int &y, int &z )
{
	x = (int)floorf( p.x * 0.25f );
	y = (int)floorf( p.y * 0.25f );
	z = (int)floorf( p.z * 0.25f );
}

static void EnsureCreaseVertexHash()
{
	if ( g_CreaseHashReady )
		return;

	ThreadLock();
	if ( !g_CreaseHashReady )
	{
		for ( int i = 0; i < kCreaseHashBuckets; ++i )
			g_CreaseHashHead[i] = -1;
		g_CreaseHashNodes.RemoveAll();

		for ( int face = 0; face < numfaces; ++face )
		{
			dface_t *pf = &g_pFaces[face];
			if ( texinfo[pf->texinfo].flags & TEX_SPECIAL )
				continue;
			for ( int e = 0; e < pf->numedges; ++e )
			{
				int x, y, z;
				CreaseQuantize( dvertexes[RadialEdgeVertex( pf, e )].point, x, y, z );
				unsigned __int64 key = CreaseCellKey( x, y, z );
				int bucket = CreaseCellBucket( key );
				CreaseHashNode_t node;
				node.key = key;
				node.face = face;
				node.next = g_CreaseHashHead[bucket];
				g_CreaseHashNodes.AddToTail( node );
				g_CreaseHashHead[bucket] = g_CreaseHashNodes.Count() - 1;
			}
		}
		g_CreaseHashReady = true;
	}
	ThreadUnlock();
}

static void CreaseAddNearFaces( const Vector &p, int selfFace, int *candidates, int &nCand, int cap )
{
	int x, y, z;
	CreaseQuantize( p, x, y, z );
	for ( int dz = -1; dz <= 1; ++dz )
	{
		for ( int dy = -1; dy <= 1; ++dy )
		{
			for ( int dx = -1; dx <= 1; ++dx )
			{
				unsigned __int64 key = CreaseCellKey( x + dx, y + dy, z + dz );
				int bucket = CreaseCellBucket( key );
				for ( int i = g_CreaseHashHead[bucket]; i >= 0; i = g_CreaseHashNodes[i].next )
				{
					if ( g_CreaseHashNodes[i].key != key )
						continue;
					const int other = g_CreaseHashNodes[i].face;
					if ( other == selfFace )
						continue;
					bool dup = false;
					for ( int c = 0; c < nCand; ++c )
					{
						if ( candidates[c] == other )
						{
							dup = true;
							break;
						}
					}
					if ( dup )
						continue;
					if ( nCand >= cap )
						return;
					candidates[nCand++] = other;
				}
			}
		}
	}
}

// Steps from (s,t) to the first inward texel that matches the one beyond it
// and is actually brighter than the start. A light/shadow terminator is not a
// plateau, so the walk stops there instead of mixing the two sides.
// Returns 0 when the outer texel already agrees with a stable neighbor.
static int CreaseStableSteps( const Vector *src, int width, int height,
	int s, int t, int ds, int dt, int maxStep, float gate )
{
	const float Lstart = EdgeLuxelLuma( src[s + t * width] );
	int cs = s;
	int ct = t;
	for ( int i = 0; i < maxStep; ++i )
	{
		const int s1 = cs + ds;
		const int t1 = ct + dt;
		if ( s1 < 0 || t1 < 0 || s1 >= width || t1 >= height )
			break;
		const float L1 = EdgeLuxelLuma( src[s1 + t1 * width] );
		const int s2 = s1 + ds;
		const int t2 = t1 + dt;
		bool plateau = true;
		if ( s2 >= 0 && t2 >= 0 && s2 < width && t2 < height )
		{
			const float L2 = EdgeLuxelLuma( src[s2 + t2 * width] );
			const float rel = fabsf( L1 - L2 ) / max( max( L1, L2 ), 1e-4f );
			plateau = rel <= 0.30f;
		}
		if ( plateau )
		{
			if ( L1 > 0.02f && Lstart < L1 * gate && ( L1 - Lstart ) > 0.01f )
				return i + 1;
			return 0;
		}
		// The next texel already left the dark band and is not stable with
		// the one beyond it. Stop, so a shadow edge is not replaced by light
		// further in.
		const float relBand = fabsf( L1 - Lstart ) / max( max( max( L1, Lstart ), 0.05f ), 1e-4f );
		if ( relBand > 0.45f )
			return 0;
		cs = s1;
		ct = t1;
	}
	return 0;
}

// True when copying `plateau` onto this row would paint the lit side of a
// terminator onto an outer texel that still belongs to the shadow side.
static bool CreaseRowCrossesTerminator( float plateau, float outer, float sideA, float sideB )
{
	const float shadowRef = min( sideA, sideB );
	return shadowRef > 0.02f && plateau > shadowRef * 1.8f && outer <= shadowRef * 1.15f + 0.01f;
}

// One row or column. Copies the raw plateau texel — no tangent blur, so a
// shadow edge keeps its own side of the terminator.
static void ApplyCreaseBorder( Vector *grid, const Vector *src, int width, int height,
	int border, int maxStep, float gate, bool skipT0, bool skipT1, bool skipS0, bool skipS1 )
{
	const int sMax = width - 1;
	const int tMax = height - 1;
	if ( border == 0 || border == 1 )
	{
		const int ds = ( border == 0 ) ? 1 : -1;
		const int s0 = ( border == 0 ) ? 0 : sMax;
		for ( int t = 0; t <= tMax; ++t )
		{
			if ( ( skipT0 && t == 0 ) || ( skipT1 && t == tMax ) )
				continue;
			const int d = CreaseStableSteps( src, width, height, s0, t, ds, 0, maxStep, gate );
			if ( d <= 0 )
				continue;
			const int srcS = s0 + ds * d;
			if ( srcS < 0 || srcS >= width )
				continue;
			const Vector &v = src[srcS + t * width];
			const float lv = EdgeLuxelLuma( v );
			const float Louter = EdgeLuxelLuma( src[s0 + t * width] );
			const float sideA = ( t > 0 ) ? EdgeLuxelLuma( src[srcS + ( t - 1 ) * width] ) : lv;
			const float sideB = ( t < tMax ) ? EdgeLuxelLuma( src[srcS + ( t + 1 ) * width] ) : lv;
			if ( CreaseRowCrossesTerminator( lv, Louter, sideA, sideB ) )
				continue;
			for ( int k = 0; k < d; ++k )
			{
				const int s = s0 + ds * k;
				const int idx = s + t * width;
				if ( EdgeLuxelLuma( src[idx] ) < lv * gate )
					grid[idx] = v;
			}
		}
	}
	else
	{
		const int dt = ( border == 2 ) ? 1 : -1;
		const int t0 = ( border == 2 ) ? 0 : tMax;
		for ( int s = 0; s <= sMax; ++s )
		{
			if ( ( skipS0 && s == 0 ) || ( skipS1 && s == sMax ) )
				continue;
			const int d = CreaseStableSteps( src, width, height, s, t0, 0, dt, maxStep, gate );
			if ( d <= 0 )
				continue;
			const int srcT = t0 + dt * d;
			if ( srcT < 0 || srcT >= height )
				continue;
			const Vector &v = src[s + srcT * width];
			const float lv = EdgeLuxelLuma( v );
			const float Louter = EdgeLuxelLuma( src[s + t0 * width] );
			const float sideA = ( s > 0 ) ? EdgeLuxelLuma( src[( s - 1 ) + srcT * width] ) : lv;
			const float sideB = ( s < sMax ) ? EdgeLuxelLuma( src[( s + 1 ) + srcT * width] ) : lv;
			if ( CreaseRowCrossesTerminator( lv, Louter, sideA, sideB ) )
				continue;
			for ( int k = 0; k < d; ++k )
			{
				const int t = t0 + dt * k;
				const int idx = s + t * width;
				if ( EdgeLuxelLuma( src[idx] ) < lv * gate )
					grid[idx] = v;
			}
		}
	}
}

// Single corner texel, stepped diagonally into the face. Rejected when that
// sample is much brighter than both adjacent edge interiors (sun leaking
// into a shadowed corner).
static void ApplyCreaseCorner( Vector *grid, const Vector *src, int width, int height,
	int s0, int t0, int ds, int dt, int maxStep, float gate )
{
	const int steps = CreaseStableSteps( src, width, height, s0, t0, ds, dt, maxStep, gate );
	if ( steps <= 0 )
		return;
	const int ss = s0 + ds * steps;
	const int tt = t0 + dt * steps;
	if ( ss < 0 || tt < 0 || ss >= width || tt >= height )
		return;

	const int sAxis = s0 + ds;
	const int tAxis = t0 + dt;
	float local = 0.0f;
	if ( sAxis >= 0 && sAxis < width )
		local = max( local, EdgeLuxelLuma( src[sAxis + t0 * width] ) );
	if ( tAxis >= 0 && tAxis < height )
		local = max( local, EdgeLuxelLuma( src[s0 + tAxis * width] ) );

	const Vector &v = src[ss + tt * width];
	const float lv = EdgeLuxelLuma( v );
	if ( local > 0.02f && lv > local * 1.35f )
		return;

	const int idx = s0 + t0 * width;
	if ( EdgeLuxelLuma( src[idx] ) < lv * gate )
		grid[idx] = v;
}

// Concave edge that does not lie on the lightmap rectangle (stairs, other
// non-axis brush edges). Luxel centers along it can fall outside the face.
struct CreaseDiagSeg_t
{
	Vector2D a;
	Vector2D b;
	float ix;
	float iy;
};

static const int kCreaseMaxDiag = 24;

static void CreasePushDiag( CreaseDiagSeg_t *diag, int &nDiag, int maxDiag,
	const Vector2D &ls0, const Vector2D &ls1, const Vector2D &centroidLS )
{
	if ( !diag || nDiag >= maxDiag )
		return;

	const float dx = ls1.x - ls0.x;
	const float dy = ls1.y - ls0.y;
	const float len = sqrtf( dx * dx + dy * dy );
	if ( len < 1.25f )
		return;

	const float adx = fabsf( dx );
	const float ady = fabsf( dy );
	// Axis-aligned edges are the rectangle borders. A diagonal has extent on
	// both luxel axes and is not a shallow cut along one of them.
	if ( adx <= 1.0f || ady <= 1.0f || adx <= 0.30f * len || ady <= 0.30f * len )
		return;

	float ix = -dy / len;
	float iy = dx / len;
	const float mx = 0.5f * ( ls0.x + ls1.x );
	const float my = 0.5f * ( ls0.y + ls1.y );
	if ( ix * ( centroidLS.x - mx ) + iy * ( centroidLS.y - my ) < 0.0f )
	{
		ix = -ix;
		iy = -iy;
	}

	for ( int i = 0; i < nDiag; ++i )
	{
		const float ex = 0.5f * ( diag[i].a.x + diag[i].b.x ) - mx;
		const float ey = 0.5f * ( diag[i].a.y + diag[i].b.y ) - my;
		if ( ex * ex + ey * ey < 0.36f && ix * diag[i].ix + iy * diag[i].iy > 0.8f )
			return;
	}

	diag[nDiag].a = ls0;
	diag[nDiag].b = ls1;
	diag[nDiag].ix = ix;
	diag[nDiag].iy = iy;
	++nDiag;
}

// Signed luxel distance to the segment, positive toward the face interior.
// Points past either endpoint are rejected so the fill stays on this edge.
static bool CreaseDiagSample( float px, float py, const CreaseDiagSeg_t &seg, float &outSigned )
{
	const float abx = seg.b.x - seg.a.x;
	const float aby = seg.b.y - seg.a.y;
	const float ab2 = abx * abx + aby * aby;
	if ( ab2 < 1e-6f )
		return false;

	float t = ( ( px - seg.a.x ) * abx + ( py - seg.a.y ) * aby ) / ab2;
	t = clamp( t, 0.0f, 1.0f );
	const float vx = px - ( seg.a.x + abx * t );
	const float vy = py - ( seg.a.y + aby * t );
	const float invLen = 1.0f / sqrtf( ab2 );
	const float tang = ( vx * abx + vy * aby ) * invLen;
	if ( fabsf( tang ) > 0.75f )
		return false;

	outSigned = vx * seg.ix + vy * seg.iy;
	return true;
}

// Luxels that sit on or just outside a diagonal concave edge are mostly
// occluded by the neighbor solid. Copy the raw interior plateau of that edge.
static void ApplyCreaseDiagonals( Vector *grid, const Vector *src, int width, int height,
	const CreaseDiagSeg_t *diag, int nDiag )
{
	if ( !diag || nDiag <= 0 )
		return;

	const float gate = 0.75f;
	const int maxStep = 2;
	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			if ( EdgeLuxelLuma( grid[idx] ) > EdgeLuxelLuma( src[idx] ) + 0.01f )
				continue;

			float bestAbs = 1.0e9f;
			float bestSigned = 0.0f;
			int best = -1;
			for ( int e = 0; e < nDiag; ++e )
			{
				float sd;
				if ( !CreaseDiagSample( (float)s, (float)t, diag[e], sd ) )
					continue;
				const float ad = fabsf( sd );
				if ( ad < bestAbs )
				{
					bestAbs = ad;
					bestSigned = sd;
					best = e;
				}
			}
			if ( best < 0 || bestSigned < -1.15f || bestSigned > 0.55f )
				continue;

			int ds = 0;
			int dt = 0;
			const float ix = diag[best].ix;
			const float iy = diag[best].iy;
			if ( fabsf( ix ) >= 0.4f )
				ds = ( ix > 0.0f ) ? 1 : -1;
			if ( fabsf( iy ) >= 0.4f )
				dt = ( iy > 0.0f ) ? 1 : -1;
			if ( !ds && !dt )
			{
				if ( fabsf( ix ) >= fabsf( iy ) )
					ds = ( ix >= 0.0f ) ? 1 : -1;
				else
					dt = ( iy >= 0.0f ) ? 1 : -1;
			}

			const int d = CreaseStableSteps( src, width, height, s, t, ds, dt, maxStep, gate );
			if ( d <= 0 )
				continue;
			const int ss = s + ds * d;
			const int tt = t + dt * d;
			if ( ss < 0 || tt < 0 || ss >= width || tt >= height )
				continue;

			const Vector &v = src[ss + tt * width];
			const float lv = EdgeLuxelLuma( v );
			if ( EdgeLuxelLuma( src[idx] ) < lv * gate )
				grid[idx] = v;
		}
	}
}

// concave[b] / coplanar[b]: lightmap rectangle borders.
// bit0=s0, bit1=sMax, bit2=t0, bit3=tMax.
// diag: concave edges that cut across the luxel grid (not the rectangle).
//
// PairEdges only records neighbors with ndot >= smoothing_threshold (45°), so
// 90° walls never appear in faceneighbor. Walk vertex-sharing faces, plus
// any face whose verts land in the same 4-unit cell.
static void ClassifyLightmapCreaseBorders( int facenum, int width, int height, bool concave[4], bool coplanar[4],
	CreaseDiagSeg_t *diag, int *nDiag, int maxDiag )
{
	for ( int b = 0; b < 4; ++b )
	{
		concave[b] = false;
		coplanar[b] = false;
	}
	if ( nDiag )
		*nDiag = 0;
	if ( facenum < 0 || facenum >= numfaces || width < 2 || height < 2 )
		return;

	dface_t *f = &g_pFaces[facenum];
	if ( texinfo[f->texinfo].flags & TEX_SPECIAL || f->dispinfo != -1 )
		return;

	lightinfo_t lA;
	InitLightinfo( &lA, facenum );
	const Vector &nA = faceneighbor[facenum].facenormal;

	Vector centroidA( 0, 0, 0 );
	for ( int ea = 0; ea < f->numedges; ++ea )
		centroidA += dvertexes[RadialEdgeVertex( f, ea )].point;
	if ( f->numedges > 0 )
		centroidA *= ( 1.0f / (float)f->numedges );
	Vector2D centroidLS( 0, 0 );
	WorldToLuxelSpace( &lA, centroidA, centroidLS );

	int candidates[256];
	int nCand = 0;
	for ( int ea = 0; ea < f->numedges; ++ea )
	{
		const int v = RadialEdgeVertex( f, ea );
		if ( v < 0 || v >= numvertexes || !vertexface[v] )
			continue;
		for ( int k = 0; k < vertexref[v]; ++k )
		{
			const int other = vertexface[v][k];
			if ( other == facenum )
				continue;
			bool dup = false;
			for ( int c = 0; c < nCand; ++c )
			{
				if ( candidates[c] == other )
				{
					dup = true;
					break;
				}
			}
			if ( !dup && nCand < (int)ARRAYSIZE( candidates ) )
				candidates[nCand++] = other;
		}
	}

	// Duplicated / unwelded verts still share a 4-unit cell.
	EnsureCreaseVertexHash();
	for ( int ea = 0; ea < f->numedges && nCand < (int)ARRAYSIZE( candidates ); ++ea )
		CreaseAddNearFaces( dvertexes[RadialEdgeVertex( f, ea )].point, facenum, candidates, nCand, (int)ARRAYSIZE( candidates ) );

	for ( int j = 0; j < nCand; ++j )
	{
		const int neighborFace = candidates[j];
		dface_t *fB = &g_pFaces[neighborFace];
		if ( texinfo[fB->texinfo].flags & TEX_SPECIAL || fB->dispinfo != -1 )
			continue;

		const Vector &nB = faceneighbor[neighborFace].facenormal;
		const float ndot = DotProduct( nA, nB );
		const bool isCoplanar = ndot >= STITCH_COPLANAR_DOT;
		const bool isPerp = fabsf( ndot ) <= 0.35f;
		if ( !isCoplanar && !isPerp )
			continue;

		for ( int ea = 0; ea < f->numedges; ++ea )
		{
			const Vector &a0 = dvertexes[RadialEdgeVertex( f, ea )].point;
			const Vector &a1 = dvertexes[RadialEdgeVertex( f, ea + 1 )].point;
			for ( int eb = 0; eb < fB->numedges; ++eb )
			{
				Vector overlap0, overlap1;
				if ( !StitchEdgesOverlap( a0, a1,
										  dvertexes[RadialEdgeVertex( fB, eb )].point,
										  dvertexes[RadialEdgeVertex( fB, eb + 1 )].point,
										  overlap0, overlap1 ) )
					continue;

				if ( isPerp )
				{
					// Source normals point into the room. Skip when this face sits
					// behind the neighbor plane (convex / exterior corner).
					if ( DotProduct( centroidA - overlap0, nB ) <= 0.0f )
						continue;
				}

				Vector2D ls0, ls1;
				WorldToLuxelSpace( &lA, overlap0, ls0 );
				WorldToLuxelSpace( &lA, overlap1, ls1 );
				if ( isPerp && diag && nDiag )
					CreasePushDiag( diag, *nDiag, maxDiag, ls0, ls1, centroidLS );

				const int mask = LuxelSegCreaseBorderMask( ls0, ls1, width, height );
				if ( !mask )
					continue;
				for ( int b = 0; b < 4; ++b )
				{
					if ( !( mask & ( 1 << b ) ) )
						continue;
					if ( isCoplanar )
						coplanar[b] = true;
					if ( isPerp )
						concave[b] = true;
				}
			}
		}
	}
}

// Luxel center sits in another brush, or inside a static-prop mesh that
// intersects this face. A point on this face's own plane is nudged into the
// room first so the owner brush is not classified as solid.
static void BuildLuxelEmbedMask( const lightinfo_t &l, int width, int height, CUtlVector<unsigned char> &embedded )
{
	const int n = width * height;
	embedded.SetCount( n );
	memset( embedded.Base(), 0, n );

	CUtlVector<Vector> origins;
	CUtlVector<Vector> dirs;
	CUtlVector<float> tmins;
	CUtlVector<float> tmaxs;
	CUtlVector<int> rayMap;
	origins.EnsureCapacity( n );
	dirs.EnsureCapacity( n );
	tmins.EnsureCapacity( n );
	tmaxs.EnsureCapacity( n );
	rayMap.EnsureCapacity( n );

	const Vector &normal = l.facenormal;
	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			Vector world;
			LuxelSpaceToWorld( &l, (float)s, (float)t, world );
			Vector nudged = world + normal * 0.5f;
			const int leaf = PointLeafnum( nudged );
			if ( leaf >= 0 && leaf < numleafs && ( dleafs[leaf].contents & CONTENTS_SOLID ) )
			{
				embedded[idx] = 1;
				continue;
			}

			// Inside a prop the nearest hit along the normal is the shell's
			// backface. A prop in front of the face is a frontface hit.
			origins.AddToTail( world + normal * 0.25f );
			dirs.AddToTail( normal );
			tmins.AddToTail( 0.0f );
			tmaxs.AddToTail( 256.0f );
			rayMap.AddToTail( idx );
		}
	}

	const int nRays = rayMap.Count();
	CUtlVector<float> outT;
	CUtlVector<unsigned int> outFlags, outHit;
	CUtlVector<Vector> outN;
	if ( nRays > 0 )
	{
		outT.SetCount( nRays );
		outFlags.SetCount( nRays );
		outHit.SetCount( nRays );
		outN.SetCount( nRays );
		if ( PathTraceDXR_TraceClosest( origins.Base(), dirs.Base(), tmins.Base(), tmaxs.Base(),
				outT.Base(), outFlags.Base(), outHit.Base(), nRays, outN.Base() ) )
		{
			for ( int i = 0; i < nRays; ++i )
			{
				if ( !outHit[i] || ( outFlags[i] & TRACE_ID_SKY ) )
					continue;
				if ( ( outFlags[i] & TRACE_ID_STATICPROP ) == 0 )
					continue;
				Vector hn = outN[i];
				if ( hn.LengthSqr() < 1e-8f )
					continue;
				VectorNormalize( hn );
				if ( DotProduct( hn, normal ) > 0.2f )
					embedded[rayMap[i]] = 2;
			}
		}
	}

	// The visible dark band sits on the brush just outside the prop, so a
	// ray along the normal never enters the mesh. Probe across the face and
	// mark luxels that meet the prop within one and a half luxels.
	float lenS = l.luxelToWorldSpace[0].Length();
	float lenT = l.luxelToWorldSpace[1].Length();
	float luxelLen = 0.5f * ( lenS + lenT );
	if ( luxelLen < 1.0f )
		luxelLen = 1.0f;
	Vector axisS = l.luxelToWorldSpace[0];
	Vector axisT = l.luxelToWorldSpace[1];
	if ( lenS > 1e-6f )
		axisS *= ( 1.0f / lenS );
	else
		axisS.Init( 1, 0, 0 );
	if ( lenT > 1e-6f )
		axisT *= ( 1.0f / lenT );
	else
		axisT.Init( 0, 1, 0 );
	const float reach = luxelLen * 2.0f;

	origins.RemoveAll();
	dirs.RemoveAll();
	tmins.RemoveAll();
	tmaxs.RemoveAll();
	rayMap.RemoveAll();

	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			if ( embedded[idx] )
				continue;
			Vector world;
			LuxelSpaceToWorld( &l, (float)s, (float)t, world );
			Vector probe = world + normal * 0.5f;
			const int leaf = PointLeafnum( probe );
			if ( leaf >= 0 && StaticPropMgr()->IsPointInsideStaticProp( 0, leaf, probe ) )
			{
				embedded[idx] = 2;
				continue;
			}
			for ( int a = 0; a < 8; ++a )
			{
				const float ang = (float)a * ( 2.0f * (float)M_PI / 8.0f );
				Vector dir = axisS * cosf( ang ) + axisT * sinf( ang );
				origins.AddToTail( probe );
				dirs.AddToTail( dir );
				tmins.AddToTail( 0.05f );
				tmaxs.AddToTail( reach );
				rayMap.AddToTail( idx );
			}
		}
	}

	const int nSide = rayMap.Count();
	if ( nSide <= 0 )
		return;

	outT.SetCount( nSide );
	outFlags.SetCount( nSide );
	outHit.SetCount( nSide );
	outN.SetCount( nSide );
	if ( !PathTraceDXR_TraceClosest( origins.Base(), dirs.Base(), tmins.Base(), tmaxs.Base(),
			outT.Base(), outFlags.Base(), outHit.Base(), nSide, outN.Base() ) )
		return;

	for ( int i = 0; i < nSide; ++i )
	{
		if ( !outHit[i] || ( outFlags[i] & TRACE_ID_SKY ) )
			continue;
		if ( ( outFlags[i] & TRACE_ID_STATICPROP ) == 0 )
			continue;
		if ( outT[i] > reach )
			continue;
		embedded[rayMap[i]] = 2;
	}
}

static CUtlVector<CUtlVector<unsigned char>> g_LuxelEmbedMask;
static bool g_LuxelEmbedMaskInit = false;

static const unsigned char *LuxelEmbedMask( int facenum, int width, int height, const lightinfo_t &l )
{
	const int n = width * height;
	if ( !g_LuxelEmbedMaskInit )
	{
		ThreadLock();
		if ( !g_LuxelEmbedMaskInit )
		{
			g_LuxelEmbedMask.SetCount( numfaces );
			g_LuxelEmbedMaskInit = true;
		}
		ThreadUnlock();
	}
	if ( facenum >= 0 && facenum < g_LuxelEmbedMask.Count() && g_LuxelEmbedMask[facenum].Count() == n )
		return g_LuxelEmbedMask[facenum].Base();

	CUtlVector<unsigned char> built;
	BuildLuxelEmbedMask( l, width, height, built );

	ThreadLock();
	if ( facenum >= 0 && facenum < g_LuxelEmbedMask.Count() )
	{
		if ( g_LuxelEmbedMask[facenum].Count() != n )
			g_LuxelEmbedMask[facenum] = built;
		ThreadUnlock();
		return g_LuxelEmbedMask[facenum].Base();
	}
	ThreadUnlock();
	return NULL;
}

// Black texels inside an intersecting solid bilinear-filter onto the visible
// face. Paint those texels, and the one straddling luxel, from the visible
// side. A real shadow stays: it is not inside solid, and a light/shadow split
// along the cut is left alone.
static void HealIntersectingOcclusion( Vector *grid, int width, int height, int facenum )
{
	if ( !grid || facenum < 0 || facenum >= numfaces || width < 2 || height < 2 )
		return;

	dface_t *f = &g_pFaces[facenum];
	if ( texinfo[f->texinfo].flags & TEX_SPECIAL || f->dispinfo != -1 )
		return;

	lightinfo_t l;
	InitLightinfo( &l, facenum );

	const unsigned char *embedSrc = LuxelEmbedMask( facenum, width, height, l );
	if ( !embedSrc )
		return;

	const int n = width * height;
	CUtlVector<char> embedded;
	CUtlVector<char> propTouch;
	embedded.SetCount( n );
	propTouch.SetCount( n );
	int nEmb = 0;
	int nProp = 0;
	for ( int i = 0; i < n; ++i )
	{
		embedded[i] = ( embedSrc[i] & 1 ) ? 1 : 0;
		propTouch[i] = ( embedSrc[i] & 2 ) ? 1 : 0;
		if ( embedded[i] )
			++nEmb;
		if ( propTouch[i] )
			++nProp;
	}
	if ( ( !nEmb && !nProp ) || ( nEmb == n && !nProp ) )
		return;

	CUtlVector<Vector> src;
	src.SetCount( n );
	for ( int i = 0; i < n; ++i )
		src[i] = grid[i];

	const int kDx[4] = { 1, -1, 0, 0 };
	const int kDy[4] = { 0, 0, 1, -1 };

	// Straddling luxel: in the room, but one side is inside the other brush.
	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			if ( embedded[idx] )
				continue;

			float refL = -1.0f;
			Vector ref( 0, 0, 0 );
			bool agree = true;
			int nRef = 0;
			bool touch = false;
			for ( int k = 0; k < 4; ++k )
			{
				const int ns = s + kDx[k];
				const int nt = t + kDy[k];
				if ( ns < 0 || nt < 0 || ns >= width || nt >= height )
					continue;
				if ( !embedded[ns + nt * width] )
					continue;
				touch = true;

				const int os = s - kDx[k];
				const int ot = t - kDy[k];
				if ( os < 0 || ot < 0 || os >= width || ot >= height )
					continue;
				if ( embedded[os + ot * width] )
					continue;

				const Vector &ov = src[os + ot * width];
				const float ol = EdgeLuxelLuma( ov );
				if ( !nRef )
				{
					refL = ol;
					ref = ov;
				}
				else
				{
					const float rel = fabsf( ol - refL ) / max( max( ol, refL ), 1e-4f );
					if ( rel > 0.35f )
						agree = false;
				}
				++nRef;
			}
			if ( !touch || !agree || !nRef || refL <= 0.02f )
				continue;
			if ( EdgeLuxelLuma( src[idx] ) < refL * 0.90f && EdgeLuxelLuma( grid[idx] ) < refL )
				grid[idx] = ref;
		}
	}

	// One luxel of the buried side takes the visible neighbor, so the engine
	// bilinear kernel does not read black.
	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			if ( !embedded[idx] )
				continue;

			float minL = 1.0e9f;
			float maxL = -1.0f;
			Vector darker( 0, 0, 0 );
			Vector anyV( 0, 0, 0 );
			int nVis = 0;
			for ( int k = 0; k < 4; ++k )
			{
				const int ns = s + kDx[k];
				const int nt = t + kDy[k];
				if ( ns < 0 || nt < 0 || ns >= width || nt >= height )
					continue;
				if ( embedded[ns + nt * width] )
					continue;
				const Vector &v = grid[ns + nt * width];
				const float lv = EdgeLuxelLuma( v );
				if ( !nVis || lv < minL )
				{
					minL = lv;
					darker = v;
				}
				maxL = max( maxL, lv );
				anyV = v;
				++nVis;
			}
			if ( !nVis )
				continue;

			const bool agree = ( minL >= maxL * 0.65f );
			const Vector &pick = agree ? anyV : darker;
			if ( EdgeLuxelLuma( grid[idx] ) < EdgeLuxelLuma( pick ) )
				grid[idx] = pick;
		}
	}

	// Brush luxels that touch a prop. Copy the nearest luxel that does not,
	// so the contact band matches the open face instead of the occluded sample.
	if ( nProp )
	{
		for ( int t = 0; t < height; ++t )
		{
			for ( int s = 0; s < width; ++s )
			{
				const int idx = s + t * width;
				if ( !propTouch[idx] )
					continue;

				int bestD = 99;
				float bestL = -1.0f;
				Vector best( 0, 0, 0 );
				for ( int dy = -3; dy <= 3; ++dy )
				{
					const int nt = t + dy;
					if ( nt < 0 || nt >= height )
						continue;
					for ( int dx = -3; dx <= 3; ++dx )
					{
						if ( !dx && !dy )
							continue;
						const int ns = s + dx;
						if ( ns < 0 || ns >= width )
							continue;
						const int ni = ns + nt * width;
						if ( propTouch[ni] || embedded[ni] )
							continue;
						const int d = max( abs( dx ), abs( dy ) );
						const float lv = EdgeLuxelLuma( src[ni] );
						if ( d < bestD || ( d == bestD && lv < bestL ) )
						{
							bestD = d;
							bestL = lv;
							best = src[ni];
						}
					}
				}
				if ( bestD <= 3 && bestL > 0.02f && EdgeLuxelLuma( grid[idx] ) < bestL * 0.92f )
					grid[idx] = best;
			}
		}
	}
}

// Outer luxels on a concave crease lose hemisphere energy. Each row copies
// its own stable interior texel, so a light/shadow edge is not averaged.
// The shared corner texel is one diagonal sample. Diagonal concave edges
// fill luxels that landed mostly outside the face. Coplanar splits are not touched.
void ReconstructLightmapEdgeLuxels( Vector *grid, int width, int height, int facenum, bool bConcaveOnly )
{
	if ( !grid || width < 2 || height < 2 )
		return;

	bool concave[4], coplanar[4];
	CreaseDiagSeg_t diag[kCreaseMaxDiag];
	int nDiag = 0;
	ClassifyLightmapCreaseBorders( facenum, width, height, concave, coplanar, diag, &nDiag, kCreaseMaxDiag );

	const int capS = min( 4, max( 1, ( width - 1 ) / 2 ) );
	const int capT = min( 4, max( 1, ( height - 1 ) / 2 ) );

	bool active[4];
	float gate[4];
	int maxStep[4];
	bool any = nDiag > 0;
	for ( int b = 0; b < 4; ++b )
	{
		active[b] = false;
		gate[b] = 0.72f;
		maxStep[b] = ( b < 2 ) ? capS : capT;
		if ( coplanar[b] && !concave[b] )
			continue;
		if ( concave[b] )
		{
			active[b] = true;
			gate[b] = 0.86f;
		}
		else if ( !bConcaveOnly )
		{
			active[b] = true;
			gate[b] = 0.72f;
			if ( maxStep[b] > 3 )
				maxStep[b] = 3;
		}
		if ( active[b] )
			any = true;
	}
	if ( !any )
	{
		HealIntersectingOcclusion( grid, width, height, facenum );
		return;
	}

	const int n = width * height;
	CUtlVector<Vector> src;
	src.SetCount( n );
	for ( int i = 0; i < n; ++i )
		src[i] = grid[i];

	const int sMax = width - 1;
	const int tMax = height - 1;
	// Both borders own this corner: keep it out of either edge walk and fill
	// it with one diagonal sample so neither side's light is dragged across.
	const bool cornerBoth[4] = {
		active[0] && active[2],
		active[1] && active[2],
		active[0] && active[3],
		active[1] && active[3],
	};

	for ( int b = 0; b < 4; ++b )
	{
		if ( !active[b] )
			continue;
		bool skipT0 = false, skipT1 = false, skipS0 = false, skipS1 = false;
		if ( b == 0 )
		{
			skipT0 = cornerBoth[0];
			skipT1 = cornerBoth[2];
		}
		else if ( b == 1 )
		{
			skipT0 = cornerBoth[1];
			skipT1 = cornerBoth[3];
		}
		else if ( b == 2 )
		{
			skipS0 = cornerBoth[0];
			skipS1 = cornerBoth[1];
		}
		else
		{
			skipS0 = cornerBoth[2];
			skipS1 = cornerBoth[3];
		}
		ApplyCreaseBorder( grid, src.Base(), width, height, b, maxStep[b], gate[b],
			skipT0, skipT1, skipS0, skipS1 );
	}

	struct CornerSpec
	{
		int b0, b1;
		int s0, t0;
		int ds, dt;
	};
	const CornerSpec corners[4] = {
		{ 0, 2, 0,    0,    +1, +1 },
		{ 1, 2, sMax, 0,    -1, +1 },
		{ 0, 3, 0,    tMax, +1, -1 },
		{ 1, 3, sMax, tMax, -1, -1 },
	};
	for ( int c = 0; c < 4; ++c )
	{
		if ( !cornerBoth[c] )
			continue;
		const int b0 = corners[c].b0;
		const int b1 = corners[c].b1;
		const float g = max( gate[b0], gate[b1] );
		int mx = max( maxStep[b0], maxStep[b1] );
		mx = min( mx, min( capS, capT ) );
		ApplyCreaseCorner( grid, src.Base(), width, height,
			corners[c].s0, corners[c].t0, corners[c].ds, corners[c].dt,
			mx, g );
	}

	ApplyCreaseDiagonals( grid, src.Base(), width, height, diag, nDiag );
	HealIntersectingOcclusion( grid, width, height, facenum );
}

// Bilinear sample of a face's lightmap (decoded to linear) from the snapshot buffer.
static void StitchSampleLightmap( const byte *pSnapshot, const dface_t *f, int nLuxelsWide, int nLuxelsTall,
								  int style, int bump, int bumpCount, float s, float t, Vector &out )
{
	const int numLuxels = nLuxelsWide * nLuxelsTall;
	const byte *pBase = pSnapshot + f->lightofs + ( style * bumpCount + bump ) * numLuxels * 4;

	s = clamp( s, 0.0f, (float)( nLuxelsWide - 1 ) );
	t = clamp( t, 0.0f, (float)( nLuxelsTall - 1 ) );

	int s0 = (int)s, t0 = (int)t;
	int s1 = min( s0 + 1, nLuxelsWide - 1 );
	int t1 = min( t0 + 1, nLuxelsTall - 1 );
	float fs = s - s0, ft = t - t0;

	Vector c00, c10, c01, c11;
	ColorRGBExp32ToVector( *(const ColorRGBExp32 *)( pBase + ( s0 + t0 * nLuxelsWide ) * 4 ), c00 );
	ColorRGBExp32ToVector( *(const ColorRGBExp32 *)( pBase + ( s1 + t0 * nLuxelsWide ) * 4 ), c10 );
	ColorRGBExp32ToVector( *(const ColorRGBExp32 *)( pBase + ( s0 + t1 * nLuxelsWide ) * 4 ), c01 );
	ColorRGBExp32ToVector( *(const ColorRGBExp32 *)( pBase + ( s1 + t1 * nLuxelsWide ) * 4 ), c11 );

	out = c00 * ( ( 1.0f - fs ) * ( 1.0f - ft ) ) + c10 * ( fs * ( 1.0f - ft ) ) +
		  c01 * ( ( 1.0f - fs ) * ft ) + c11 * ( fs * ft );
}

// Stitch can pull a wall-corner luxel back toward a darker coplanar neighbor.
// Re-dilate concave borders only, so coplanar seam welds stay put.
static void RehealConcaveCreasesAfterStitch()
{
	if ( !pdlightdata || !pdlightdata->Count() )
		return;

	for ( int facenum = 0; facenum < numfaces; ++facenum )
	{
		dface_t *f = &g_pFaces[facenum];
		if ( texinfo[f->texinfo].flags & TEX_SPECIAL )
			continue;
		if ( f->lightofs < 0 || f->dispinfo != -1 )
			continue;

		facelight_t *fl = &facelight[facenum];
		if ( !fl->numluxels )
			continue;

		const int width = f->m_LightmapTextureSizeInLuxels[0] + 1;
		const int height = f->m_LightmapTextureSizeInLuxels[1] + 1;
		if ( width < 2 || height < 2 || width * height != fl->numluxels )
			continue;

		const int bumpCount = ( texinfo[f->texinfo].flags & SURF_BUMPLIGHT ) ? ( NUM_BUMP_VECTS + 1 ) : 1;
		const int n = width * height;
		CUtlVector<Vector> grid;
		grid.SetCount( n );

		for ( int k = 0; k < MAXLIGHTMAPS && f->styles[k] != 255; ++k )
		{
			for ( int bump = 0; bump < bumpCount; ++bump )
			{
				byte *p = &(*pdlightdata)[f->lightofs + ( k * bumpCount + bump ) * n * 4];
				for ( int i = 0; i < n; ++i )
					ColorRGBExp32ToVector( *( ColorRGBExp32 * )( p + i * 4 ), grid[i] );
				ReconstructLightmapEdgeLuxels( grid.Base(), width, height, facenum, true );
				for ( int i = 0; i < n; ++i )
					VectorToColorRGBExp32( grid[i], *( ColorRGBExp32 * )( p + i * 4 ) );
			}
		}
	}
}

void StitchLightmapSeams( bool bPathTraceNarrow )
{
	if ( !pdlightdata->Count() )
		return;

	// Snapshot the post-FinalLightFace data so every read sees unstitched
	// values; this keeps the blend symmetric for both faces of a seam.
	CUtlVector<byte> snapshot;
	snapshot.SetCount( pdlightdata->Count() );
	Q_memcpy( snapshot.Base(), pdlightdata->Base(), pdlightdata->Count() );

	// Pathtrace: only weld the edge itself (~1 luxel). Wider radiosity band
	// blurs PT detail into blotches; skipping stitch entirely leaves penumbra shifts.
	const float luxelRange = bPathTraceNarrow ? 1.25f : STITCH_LUXEL_RANGE;

	// Chart membership — PairEdges misses T-junction splits (duplicated verts),
	// so faceneighbor alone almost never sees coplanar floor cuts.
	const int nCharts = LightmapChartCount();
	CUtlVector< CUtlVector<int> > chartMembers;
	if ( nCharts > 0 )
	{
		chartMembers.SetSize( nCharts );
		for ( int i = 0; i < numfaces; ++i )
		{
			const int cid = GetLightmapChartId( i );
			if ( cid >= 0 && cid < nCharts )
				chartMembers[cid].AddToTail( i );
		}
	}

	int nStitchedLuxels = 0;
	int nChartSeams = 0;

	for ( int facenum = 0; facenum < numfaces; ++facenum )
	{
		dface_t *f = &g_pFaces[facenum];
		if ( texinfo[f->texinfo].flags & TEX_SPECIAL )
			continue;
		if ( f->lightofs < 0 || f->dispinfo != -1 )
			continue;

		facelight_t *fl = &facelight[facenum];
		if ( !fl->numluxels || !fl->luxel )
			continue;

		lightinfo_t lA;
		InitLightinfo( &lA, facenum );
		const int wA = f->m_LightmapTextureSizeInLuxels[0] + 1;
		const int bumpCountA = ( texinfo[f->texinfo].flags & SURF_BUMPLIGHT ) ? ( NUM_BUMP_VECTS + 1 ) : 1;

		faceneighbor_t *fn = &faceneighbor[facenum];

		// Candidate partners: vertex neighbors + same lightmap chart.
		CUtlVector<int> candidates;
		candidates.EnsureCapacity( fn->numneighbors + 8 );
		for ( int j = 0; j < fn->numneighbors; ++j )
			candidates.AddToTail( fn->neighbor[j] );
		const int cid = GetLightmapChartId( facenum );
		if ( cid >= 0 && cid < chartMembers.Count() )
		{
			const CUtlVector<int> &mates = chartMembers[cid];
			for ( int m = 0; m < mates.Count(); ++m )
			{
				const int other = mates[m];
				if ( other == facenum )
					continue;
				bool bDup = false;
				for ( int c = 0; c < candidates.Count(); ++c )
				{
					if ( candidates[c] == other )
					{
						bDup = true;
						break;
					}
				}
				if ( !bDup )
					candidates.AddToTail( other );
			}
		}

		for ( int j = 0; j < candidates.Count(); ++j )
		{
			const int neighborFace = candidates[j];
			dface_t *fB = &g_pFaces[neighborFace];
			if ( texinfo[fB->texinfo].flags & TEX_SPECIAL )
				continue;
			if ( fB->lightofs < 0 || fB->dispinfo != -1 )
				continue;

			// only stitch across (nearly) coplanar seams
			if ( DotProduct( fn->facenormal, faceneighbor[neighborFace].facenormal ) < STITCH_COPLANAR_DOT )
				continue;

			const bool bSameChart = FacesShareMergedLightmapChart( facenum, neighborFace );

			// Collect geometrically overlapping coplanar edges (identical vert
			// indices OR T-junction / duplicated-vert seams), in A's luxel space.
			Vector2D edgeStart[STITCH_MAX_EDGES], edgeEnd[STITCH_MAX_EDGES];
			int nEdges = 0;
			for ( int ea = 0; ea < f->numedges && nEdges < STITCH_MAX_EDGES; ++ea )
			{
				int va0 = RadialEdgeVertex( f, ea );
				int va1 = RadialEdgeVertex( f, ea + 1 );
				const Vector &a0 = dvertexes[va0].point;
				const Vector &a1 = dvertexes[va1].point;
				for ( int eb = 0; eb < fB->numedges && nEdges < STITCH_MAX_EDGES; ++eb )
				{
					int vb0 = RadialEdgeVertex( fB, eb );
					int vb1 = RadialEdgeVertex( fB, eb + 1 );
					Vector overlap0, overlap1;
					if ( !StitchEdgesOverlap( a0, a1, dvertexes[vb0].point, dvertexes[vb1].point,
											  overlap0, overlap1 ) )
						continue;

					WorldToLuxelSpace( &lA, overlap0, edgeStart[nEdges] );
					WorldToLuxelSpace( &lA, overlap1, edgeEnd[nEdges] );
					++nEdges;
				}
			}
			if ( !nEdges )
				continue;

			if ( bSameChart )
				++nChartSeams;

			lightinfo_t lB;
			InitLightinfo( &lB, neighborFace );
			const int wB = fB->m_LightmapTextureSizeInLuxels[0] + 1;
			const int hB = fB->m_LightmapTextureSizeInLuxels[1] + 1;
			const int bumpCountB = ( texinfo[fB->texinfo].flags & SURF_BUMPLIGHT ) ? ( NUM_BUMP_VECTS + 1 ) : 1;
			const int bumpCount = min( bumpCountA, bumpCountB );

			// Chart seams under pathtrace: slightly wider weld.
			const float pairRange = ( bPathTraceNarrow && bSameChart ) ? 1.75f : luxelRange;

			for ( int i = 0; i < fl->numluxels; ++i )
			{
				const float s = (float)( i % wA );
				const float t = (float)( i / wA );

				float dist = 1e30f;
				for ( int e = 0; e < nEdges; ++e )
				{
					float d = DistPointToSegment2D( s, t, edgeStart[e], edgeEnd[e] );
					if ( d < dist )
						dist = d;
				}
				if ( dist > pairRange )
					continue;

				// 0.5 at the edge (both sides meet at the average), fading to 0
				float x = 1.0f - ( dist / pairRange );
				float blend = 0.5f * x * x * ( 3.0f - 2.0f * x );
				// Chart members under pathtrace: force a hard 50/50 at the edge row.
				if ( bPathTraceNarrow && bSameChart && dist <= 0.75f )
					blend = 0.5f;

				Vector2D coordB;
				WorldToLuxelSpace( &lB, fl->luxel[i], coordB );

				for ( int k = 0; k < MAXLIGHTMAPS && f->styles[k] != 255; ++k )
				{
					// matching style on the neighbor
					int nstyle = -1;
					for ( int ks = 0; ks < MAXLIGHTMAPS && fB->styles[ks] != 255; ++ks )
					{
						if ( fB->styles[ks] == f->styles[k] )
						{
							nstyle = ks;
							break;
						}
					}
					if ( nstyle < 0 )
						continue;

					for ( int b = 0; b < bumpCount; ++b )
					{
						const int ofsA = f->lightofs + ( k * bumpCountA + b ) * fl->numluxels * 4 + i * 4;

						Vector colorA, colorB;
						ColorRGBExp32ToVector( *(const ColorRGBExp32 *)( snapshot.Base() + ofsA ), colorA );
						StitchSampleLightmap( snapshot.Base(), fB, wB, hB, nstyle, b, bumpCountB,
											  coordB.x, coordB.y, colorB );

						const float la = EdgeLuxelLuma( colorA );
						const float lb = EdgeLuxelLuma( colorB );
						const float maxL = max( la, lb );
						float stitchBlend = blend;
						if ( maxL > 1e-3f )
						{
							const float rel = fabsf( la - lb ) / maxL;
							if ( rel > 0.50f )
								stitchBlend = 0.0f;
							else if ( rel > 0.35f )
							{
								const float gate = clamp( ( 0.50f - rel ) / 0.15f, 0.0f, 1.0f );
								stitchBlend *= gate * gate;
							}
						}

						Vector stitched = colorA;
						if ( stitchBlend > 0.0f )
							stitched = colorA + ( colorB - colorA ) * stitchBlend;
						VectorToColorRGBExp32( stitched, *(ColorRGBExp32 *)&(*pdlightdata)[ofsA] );
					}
				}

				++nStitchedLuxels;
			}
		}
	}

	RehealConcaveCreasesAfterStitch();

	if ( bPathTraceNarrow )
		Msg( "Stitched %d pathtrace seam luxel(s) (narrow band, %d chart seam pair(s))\n", nStitchedLuxels, nChartSeams );
	else
		Msg( "Stitched %d seam luxel(s)\n", nStitchedLuxels );
}
