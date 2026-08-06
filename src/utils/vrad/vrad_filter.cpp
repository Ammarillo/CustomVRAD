//========= Copyright CustomVRAD contributors. ============//
// VMT-driven colored light transmission through thin-sheet glass.
//=============================================================================//

#include "vrad_filter.h"
#include "oklab.h"
#include "vrad.h"
#include "bsplib.h"
#include "cmdlib.h"
#include "vtf/vtf.h"
#include "tier1/utldict.h"
#include "tier1/utlbuffer.h"
#include "tier1/KeyValues.h"
#include "bitmap/imageformat.h"
#include "tier1/strtools.h"
#include <math.h>

extern char level_name[MAX_PATH];

struct FilterRGBA_t
{
	int width;      // loaded mip width
	int height;
	int origWidth;  // full VTF size (textureVecs are in this space)
	int origHeight;
	bool clampU;
	bool clampV;
	CUtlMemory<unsigned char> rgba;
};

struct FilterMaterial_t
{
	bool enabled;
	bool nocull; // $nocull: tint from both sides; else front-face hits only
	float strength;
	float opacity;
	float thickness;
	FilterRGBA_t *pTex;
	Vector flatRGB; // fallback when no texture (texdata reflectivity)
};

static CUtlDict<FilterRGBA_t *, int> g_FilterTextures;
static CUtlDict<FilterMaterial_t *, int> g_FilterMaterials;
static CUtlVector<FilterMaterial_t *> g_FilterByTexdata;
static bool g_bFilterCacheBuilt = false;
static bool g_bFilterAny = false;
static int g_nFilterLoaded = 0;

static bool LoadFileToBuffer( const char *pPath, CUtlBuffer &buf )
{
	FileHandle_t h = g_pFileSystem->Open( pPath, "rb" );
	if ( h == FILESYSTEM_INVALID_HANDLE )
		return false;
	int n = (int)g_pFileSystem->Size( h );
	if ( n <= 0 )
	{
		g_pFileSystem->Close( h );
		return false;
	}
	buf.EnsureCapacity( n );
	buf.SeekPut( CUtlBuffer::SEEK_HEAD, 0 );
	g_pFileSystem->Read( buf.Base(), n, h );
	buf.SeekPut( CUtlBuffer::SEEK_HEAD, n );
	buf.SeekGet( CUtlBuffer::SEEK_HEAD, 0 );
	g_pFileSystem->Close( h );
	return true;
}

static bool ParseBoolish( const char *p )
{
	if ( !p || !p[0] )
		return false;
	if ( p[0] == '0' && p[1] == '\0' )
		return false;
	if ( !Q_stricmp( p, "false" ) || !Q_stricmp( p, "no" ) || !Q_stricmp( p, "off" ) )
		return false;
	return true;
}

static void NormalizeMaterialPath( char *p )
{
	Q_FixSlashes( p );
	if ( !Q_strnicmp( p, "materials/", 10 ) )
		Q_memmove( p, p + 10, strlen( p + 10 ) + 1 );
	char *ext = Q_strrchr( p, '.' );
	if ( ext && ( !Q_stricmp( ext, ".vmt" ) || !Q_stricmp( ext, ".vtf" ) ) )
		*ext = 0;
}

static KeyValues *LoadMergedVmtKeys( const char *pMaterialName, int nDepth = 0 )
{
	if ( !pMaterialName || !pMaterialName[0] || nDepth > 8 )
		return NULL;

	char szVmt[MAX_PATH];
	Q_snprintf( szVmt, sizeof( szVmt ), "materials/%s.vmt", pMaterialName );
	Q_FixSlashes( szVmt );

	CUtlBuffer buf( 0, 0, CUtlBuffer::TEXT_BUFFER );
	if ( !LoadFileToBuffer( szVmt, buf ) )
		return NULL;

	KeyValues *pVMT = new KeyValues( "vmt" );
	if ( !pVMT->LoadFromBuffer( pMaterialName, buf ) )
	{
		pVMT->deleteThis();
		return NULL;
	}

	const char *pInclude = NULL;
	KeyValues *pInsert = NULL;
	if ( pVMT->FindKey( "patch" ) )
	{
		KeyValues *pPatch = pVMT->FindKey( "patch" );
		pInclude = pPatch->GetString( "include", NULL );
		pInsert = pPatch->FindKey( "insert" );
	}
	if ( ( !pInclude || !pInclude[0] ) && !Q_stricmp( pVMT->GetName(), "patch" ) )
	{
		pInclude = pVMT->GetString( "include", NULL );
		pInsert = pVMT->FindKey( "insert" );
	}
	if ( pInclude && pInclude[0] )
	{
		char szInc[MAX_PATH];
		Q_strncpy( szInc, pInclude, sizeof( szInc ) );
		NormalizeMaterialPath( szInc );
		KeyValues *pBase = LoadMergedVmtKeys( szInc, nDepth + 1 );
		if ( pBase )
		{
			if ( pInsert )
			{
				for ( KeyValues *pSub = pInsert->GetFirstValue(); pSub; pSub = pSub->GetNextValue() )
					pBase->SetString( pSub->GetName(), pSub->GetString() );
			}
			// Overlay root-level $vrad_filter* from this patch file if present
			for ( KeyValues *pSub = pVMT->GetFirstValue(); pSub; pSub = pSub->GetNextValue() )
			{
				const char *n = pSub->GetName();
				if ( n && !Q_strnicmp( n, "$vrad_filter", 12 ) )
					pBase->SetString( n, pSub->GetString() );
			}
			pVMT->deleteThis();
			return pBase;
		}
	}
	return pVMT;
}

static FilterRGBA_t *LoadFilterVTF( const char *pBaseTexture )
{
	if ( !pBaseTexture || !pBaseTexture[0] )
		return NULL;

	char szClean[MAX_PATH];
	Q_strncpy( szClean, pBaseTexture, sizeof( szClean ) );
	NormalizeMaterialPath( szClean );

	int found = g_FilterTextures.Find( szClean );
	if ( g_FilterTextures.IsValidIndex( found ) )
		return g_FilterTextures[found];

	char szPath[MAX_PATH];
	Q_snprintf( szPath, sizeof( szPath ), "materials/%s.vtf", szClean );
	Q_FixSlashes( szPath );

	CUtlBuffer buf;
	if ( !LoadFileToBuffer( szPath, buf ) )
	{
		g_FilterTextures.Insert( szClean, NULL );
		return NULL;
	}

	if ( buf.TellPut() >= 12 )
	{
		unsigned char *pRaw = (unsigned char *)buf.Base();
		if ( pRaw[0] == 'V' && pRaw[1] == 'T' && pRaw[2] == 'F' && pRaw[3] == 0 )
		{
			unsigned int major = *(unsigned int *)( pRaw + 4 );
			unsigned int minor = *(unsigned int *)( pRaw + 8 );
			if ( major == 7 && minor > 4 )
				*(unsigned int *)( pRaw + 8 ) = 4;
		}
	}

	IVTFTexture *pTex = CreateVTFTexture();
	if ( !pTex->Unserialize( buf ) )
	{
		DestroyVTFTexture( pTex );
		g_FilterTextures.Insert( szClean, NULL );
		return NULL;
	}
	if ( pTex->IsCubeMap() || ( pTex->Flags() & TEXTUREFLAGS_ENVMAP ) != 0 )
	{
		DestroyVTFTexture( pTex );
		g_FilterTextures.Insert( szClean, NULL );
		return NULL;
	}

	int iWidth = pTex->Width();
	int iHeight = pTex->Height();
	if ( iWidth <= 0 || iHeight <= 0 || iWidth > 8192 || iHeight > 8192 )
	{
		DestroyVTFTexture( pTex );
		g_FilterTextures.Insert( szClean, NULL );
		return NULL;
	}

	const int origW = iWidth;
	const int origH = iHeight;

	// Full-resolution filter maps (textureVecs are in this space). Cap only pathological sizes.
	int mip = 0;
	while ( ( iWidth > 2048 || iHeight > 2048 ) && mip + 1 < pTex->MipCount() )
	{
		++mip;
		iWidth = max( 1, iWidth >> 1 );
		iHeight = max( 1, iHeight >> 1 );
	}

	ImageFormat srcFormat = pTex->Format();
	unsigned char *pSrcImage = pTex->ImageData( 0, 0, mip, 0, 0, 0 );
	if ( !pSrcImage )
		pSrcImage = pTex->ImageData( 0, 0, 0, 0, 0, 0 );

	const int nDstBytes = ImageLoader::GetMemRequired( iWidth, iHeight, 1, IMAGE_FORMAT_RGBA8888, false );
	unsigned char *pDstImage = new unsigned char[nDstBytes];
	bool bOk = pSrcImage && ImageLoader::ConvertImageFormat( pSrcImage, srcFormat, pDstImage, IMAGE_FORMAT_RGBA8888, iWidth, iHeight, 0, 0 );
	if ( !bOk )
	{
		delete[] pDstImage;
		pTex->ConvertImageFormat( IMAGE_FORMAT_RGBA8888, false );
		pSrcImage = pTex->ImageData( 0, 0, 0, 0, 0, 0 );
		iWidth = pTex->Width();
		iHeight = pTex->Height();
		if ( !pSrcImage || pTex->Format() != IMAGE_FORMAT_RGBA8888 )
		{
			DestroyVTFTexture( pTex );
			g_FilterTextures.Insert( szClean, NULL );
			return NULL;
		}
		FilterRGBA_t *pData = new FilterRGBA_t;
		pData->width = iWidth;
		pData->height = iHeight;
		pData->origWidth = origW;
		pData->origHeight = origH;
		pData->clampU = ( pTex->Flags() & TEXTUREFLAGS_CLAMPS ) != 0;
		pData->clampV = ( pTex->Flags() & TEXTUREFLAGS_CLAMPT ) != 0;
		const int nBytes = iWidth * iHeight * 4;
		pData->rgba.EnsureCapacity( nBytes );
		memcpy( pData->rgba.Base(), pSrcImage, nBytes );
		DestroyVTFTexture( pTex );
		g_FilterTextures.Insert( szClean, pData );
		++g_nFilterLoaded;
		return pData;
	}

	FilterRGBA_t *pData = new FilterRGBA_t;
	pData->width = iWidth;
	pData->height = iHeight;
	pData->origWidth = origW;
	pData->origHeight = origH;
	pData->clampU = ( pTex->Flags() & TEXTUREFLAGS_CLAMPS ) != 0;
	pData->clampV = ( pTex->Flags() & TEXTUREFLAGS_CLAMPT ) != 0;
	pData->rgba.EnsureCapacity( nDstBytes );
	memcpy( pData->rgba.Base(), pDstImage, nDstBytes );
	delete[] pDstImage;
	DestroyVTFTexture( pTex );
	g_FilterTextures.Insert( szClean, pData );
	++g_nFilterLoaded;
	return pData;
}

static FilterMaterial_t *FindOrLoadFilterMaterial( const char *pMaterialName, const Vector &flatRefl )
{
	int idx = g_FilterMaterials.Find( pMaterialName );
	if ( g_FilterMaterials.IsValidIndex( idx ) )
		return g_FilterMaterials[idx];

	FilterMaterial_t *pMat = new FilterMaterial_t;
	memset( pMat, 0, sizeof( *pMat ) );
	pMat->strength = 1.0f;
	pMat->opacity = 0.05f;
	pMat->thickness = 1.0f;
	pMat->flatRGB = flatRefl;

	KeyValues *pVMT = LoadMergedVmtKeys( pMaterialName );
	if ( pVMT )
	{
		bool bFlag = pVMT->GetInt( "$vrad_filter", 0 ) != 0 || ParseBoolish( pVMT->GetString( "$vrad_filter", NULL ) );
		pMat->enabled = bFlag;
		if ( bFlag )
		{
			pMat->strength = pVMT->GetFloat( "$vrad_filterstrength", 1.0f );
			pMat->thickness = pVMT->GetFloat( "$vrad_filterthickness", 1.0f );
			// Light filters are two-sided by default (stacked gels multiply).
			// "$vrad_filter_onesided" "1" -> front-face tint only; "$nocull" "1" forces two-sided.
			const bool onesided = pVMT->GetInt( "$vrad_filter_onesided", 0 ) != 0
				|| ParseBoolish( pVMT->GetString( "$vrad_filter_onesided", NULL ) );
			const bool nocullKey = pVMT->GetInt( "$nocull", 0 ) != 0
				|| ParseBoolish( pVMT->GetString( "$nocull", NULL ) );
			pMat->nocull = nocullKey || !onesided;
			if ( pVMT->FindKey( "$vrad_filteropacity" ) )
				pMat->opacity = pVMT->GetFloat( "$vrad_filteropacity", 0.05f );
			else
				// Do NOT map $alpha -> opacity: glass $alpha is for framebuffer blend,
				// not optical density - it made stacked panes go black.
				pMat->opacity = 0.05f;

			pMat->strength = clamp( pMat->strength, 0.0f, 1.0f );
			pMat->opacity = clamp( pMat->opacity, 0.0f, 1.0f );
			if ( pMat->thickness < 0.01f )
				pMat->thickness = 0.01f;
			if ( pMat->thickness > 8.0f )
				pMat->thickness = 8.0f;

			const char *fm = pVMT->GetString( "$vrad_filtermap", NULL );
			const char *bt = pVMT->GetString( "$basetexture", NULL );
			const char *tex = ( fm && fm[0] ) ? fm : bt;
			if ( tex && tex[0] )
				pMat->pTex = LoadFilterVTF( tex );
		}
		pVMT->deleteThis();
	}

	g_FilterMaterials.Insert( pMaterialName, pMat );
	if ( pMat->enabled )
		g_bFilterAny = true;
	return pMat;
}

static void BuildFilterCache( void )
{
	if ( g_bFilterCacheBuilt )
		return;
	g_bFilterCacheBuilt = true;
	g_bFilterAny = false;
	g_nFilterLoaded = 0;

	g_FilterByTexdata.SetCount( numtexdata );
	for ( int i = 0; i < numtexdata; ++i )
		g_FilterByTexdata[i] = NULL;

	int nEnabled = 0;
	for ( int i = 0; i < numtexdata; ++i )
	{
		const char *pName = TexDataStringTable_GetString( dtexdata[i].nameStringTableID );
		if ( !pName || !pName[0] )
			continue;
		Vector flat( dtexdata[i].reflectivity[0], dtexdata[i].reflectivity[1], dtexdata[i].reflectivity[2] );
		g_FilterByTexdata[i] = FindOrLoadFilterMaterial( pName, flat );
		if ( g_FilterByTexdata[i] && g_FilterByTexdata[i]->enabled )
			++nEnabled;
	}

	if ( nEnabled > 0 )
	{
		Msg( "Colored glass ($vrad_filter): %d texdata(s) enabled, %d filter texture(s) loaded.\n",
			 nEnabled, g_nFilterLoaded );
	}
}

void VRadFilter_EnsureCache( void )
{
	BuildFilterCache();
}

bool VRadFilter_HasAny( void )
{
	BuildFilterCache();
	return g_bFilterAny;
}

static FilterMaterial_t *GetFilterForFace( int facenum )
{
	if ( facenum < 0 || facenum >= numfaces )
		return NULL;
	BuildFilterCache();
	dface_t *f = &g_pFaces[facenum];
	texinfo_t *tx = &texinfo[f->texinfo];
	if ( tx->texdata < 0 || tx->texdata >= g_FilterByTexdata.Count() )
		return NULL;
	return g_FilterByTexdata[tx->texdata];
}

bool VRadFilter_FaceFilters( int facenum )
{
	FilterMaterial_t *pMat = GetFilterForFace( facenum );
	return pMat && pMat->enabled;
}

bool VRadFilter_FaceNoCull( int facenum )
{
	FilterMaterial_t *pMat = GetFilterForFace( facenum );
	return pMat && pMat->enabled && pMat->nocull;
}

bool VRadFilter_ShouldApply( int facenum, const Vector &rayDir )
{
	FilterMaterial_t *pMat = GetFilterForFace( facenum );
	if ( !pMat || !pMat->enabled )
		return false;
	if ( pMat->nocull )
		return true;
	if ( facenum < 0 || facenum >= numfaces )
		return true;
	// Front-face only: ray travels against the BSP face normal (drawn side).
	// Backface hits are skipped so a brush with filter on both sides does not
	// double-multiply (which makes stacked panes look opaque).
	const dplane_t *pl = &dplanes[g_pFaces[facenum].planenum];
	return DotProduct( pl->normal, rayDir ) < 0.0f;
}

bool VRadFilter_TexdataFilters( int texdata )
{
	BuildFilterCache();
	if ( texdata < 0 || texdata >= g_FilterByTexdata.Count() )
		return false;
	FilterMaterial_t *pMat = g_FilterByTexdata[texdata];
	return pMat && pMat->enabled;
}

static inline int WrapOrClamp( int v, int size, bool bClamp )
{
	if ( size <= 0 )
		return 0;
	if ( bClamp )
	{
		if ( v < 0 ) return 0;
		if ( v >= size ) return size - 1;
		return v;
	}
	v %= size;
	if ( v < 0 )
		v += size;
	return v;
}

static Vector ComputeTransmittance( const FilterMaterial_t *pMat, float r, float g, float b )
{
	// Display RGB (0..1) -> linear via square (matches bounce_albedo / Valve reflectivity).
	Vector rgb( r * r, g * g, b * b );
	if ( pMat->thickness != 1.0f )
	{
		rgb.x = powf( max( rgb.x, 1e-6f ), pMat->thickness );
		rgb.y = powf( max( rgb.y, 1e-6f ), pMat->thickness );
		rgb.z = powf( max( rgb.z, 1e-6f ), pMat->thickness );
	}
	// Strength: perceptual lerp white -> filter in Oklab (Ottosson / CSS color-mix).
	const Vector white( 1.0f, 1.0f, 1.0f );
	Vector tint = Oklab_LerpLinearSRGB( white, rgb, clamp( pMat->strength, 0.0f, 1.0f ) );
	const float pass = 1.0f - pMat->opacity;
	tint *= pass;
	tint.x = clamp( tint.x, 0.0f, 1.0f );
	tint.y = clamp( tint.y, 0.0f, 1.0f );
	tint.z = clamp( tint.z, 0.0f, 1.0f );
	return tint;
}

bool VRadFilter_SampleAtPos( int facenum, const Vector &worldPos, Vector &outT )
{
	FilterMaterial_t *pMat = GetFilterForFace( facenum );
	if ( !pMat || !pMat->enabled )
		return false;

	dface_t *f = &g_pFaces[facenum];
	texinfo_t *tx = &texinfo[f->texinfo];

	float r = pMat->flatRGB.x, g = pMat->flatRGB.y, b = pMat->flatRGB.z;
	// flatRGB already linear reflectivity; convert back to 0..1 "display" for ComputeTransmittance square
	r = sqrtf( max( r, 0.0f ) );
	g = sqrtf( max( g, 0.0f ) );
	b = sqrtf( max( b, 0.0f ) );

	FilterRGBA_t *pTex = pMat->pTex;
	if ( pTex && pTex->width > 0 && pTex->height > 0 && pTex->rgba.Base() )
	{
		float s = worldPos.x * tx->textureVecsTexelsPerWorldUnits[0][0]
			+ worldPos.y * tx->textureVecsTexelsPerWorldUnits[0][1]
			+ worldPos.z * tx->textureVecsTexelsPerWorldUnits[0][2]
			+ tx->textureVecsTexelsPerWorldUnits[0][3];
		float t = worldPos.x * tx->textureVecsTexelsPerWorldUnits[1][0]
			+ worldPos.y * tx->textureVecsTexelsPerWorldUnits[1][1]
			+ worldPos.z * tx->textureVecsTexelsPerWorldUnits[1][2]
			+ tx->textureVecsTexelsPerWorldUnits[1][3];

		// textureVecs are in full-res texel space; scale into loaded mip.
		const int ow = max( pTex->origWidth, 1 );
		const int oh = max( pTex->origHeight, 1 );
		s *= (float)pTex->width / (float)ow;
		t *= (float)pTex->height / (float)oh;

		int u = WrapOrClamp( (int)floorf( s ), pTex->width, pTex->clampU );
		int v = WrapOrClamp( (int)floorf( t ), pTex->height, pTex->clampV );
		const int nOff = ( v * pTex->width + u ) * 4;
		if ( nOff >= 0 && ( nOff + 3 ) < pTex->width * pTex->height * 4 )
		{
			const unsigned char *p = &pTex->rgba[nOff];
			r = p[0] * ( 1.0f / 255.0f );
			g = p[1] * ( 1.0f / 255.0f );
			b = p[2] * ( 1.0f / 255.0f );
		}
	}

	outT = ComputeTransmittance( pMat, r, g, b );
	return true;
}

bool VRadFilter_AverageFace( int facenum, Vector &outT )
{
	FilterMaterial_t *pMat = GetFilterForFace( facenum );
	if ( !pMat || !pMat->enabled )
		return false;

	dface_t *f = &g_pFaces[facenum];
	Vector origin( 0, 0, 0 );
	if ( facenum >= 0 && facenum < ARRAYSIZE( face_offset ) )
		origin = face_offset[facenum];

	winding_t *w = WindingFromFace( f, origin );
	if ( !w || w->numpoints < 3 )
	{
		if ( w ) FreeWinding( w );
		// Fall back to flat
		float r = sqrtf( max( pMat->flatRGB.x, 0.0f ) );
		float g = sqrtf( max( pMat->flatRGB.y, 0.0f ) );
		float b = sqrtf( max( pMat->flatRGB.z, 0.0f ) );
		outT = ComputeTransmittance( pMat, r, g, b );
		return true;
	}

	Vector centroid( 0, 0, 0 );
	for ( int i = 0; i < w->numpoints; ++i )
		centroid += w->p[i];
	centroid *= ( 1.0f / (float)w->numpoints );

	Vector sum( 0, 0, 0 );
	int n = 0;
	Vector t;
	if ( VRadFilter_SampleAtPos( facenum, centroid, t ) )
	{
		sum += t;
		++n;
	}
	for ( int i = 0; i < w->numpoints && i < 4; ++i )
	{
		Vector p = centroid + ( w->p[i] - centroid ) * 0.85f;
		if ( VRadFilter_SampleAtPos( facenum, p, t ) )
		{
			sum += t;
			++n;
		}
	}
	FreeWinding( w );

	if ( n <= 0 )
	{
		float r = sqrtf( max( pMat->flatRGB.x, 0.0f ) );
		float g = sqrtf( max( pMat->flatRGB.y, 0.0f ) );
		float b = sqrtf( max( pMat->flatRGB.z, 0.0f ) );
		outT = ComputeTransmittance( pMat, r, g, b );
		return true;
	}
	outT = sum * ( 1.0f / (float)n );
	return true;
}

// ---------------------------------------------------------------------------
// GPU Texture2DArray (one layer per unique filter VTF) + per-tri meta
// ---------------------------------------------------------------------------
bool VRadFilter_BuildGpuUpload( const uint32_t *triIds, uint32_t nTris,
								VRadFilterGpuArray_t &outArray,
								std::vector<float> &outMetaFloat4s )
{
	outArray.width = 1;
	outArray.height = 1;
	outArray.layers = 1;
	outArray.rgba.assign( 4, 255 );
	outMetaFloat4s.assign( (size_t)nTris * (size_t)kVRadFilterMetaFloat4s * 4u, 0.0f );
	if ( !triIds || nTris == 0 )
		return true;

	VRadFilter_EnsureCache();
	if ( !g_bFilterAny )
		return true;

	// Unique textures -> layer index
	CUtlDict<int, int> layerOf; // key = "%p" -> layer
	CUtlVector<FilterRGBA_t *> layerTex;
	const int kMaxLayers = 256;
	const int kMaxDim = 2048;

	auto GetLayer = [&]( FilterRGBA_t *pTex ) -> int
	{
		if ( !pTex || !pTex->rgba.Base() || pTex->width <= 0 || pTex->height <= 0 )
			return -1;
		char key[32];
		Q_snprintf( key, sizeof( key ), "%p", (void *)pTex );
		int idx = layerOf.Find( key );
		if ( layerOf.IsValidIndex( idx ) )
			return layerOf[idx];
		if ( layerTex.Count() >= kMaxLayers )
		{
			Warning( "Colored glass ($vrad_filter): GPU texture array full (%d) - remaining use face averages.\n",
					 kMaxLayers );
			return -1;
		}
		if ( pTex->width > kMaxDim || pTex->height > kMaxDim )
		{
			Warning( "Colored glass ($vrad_filter): texture %dx%d exceeds GPU slice limit %d - using average.\n",
					 pTex->width, pTex->height, kMaxDim );
			return -1;
		}
		const int layer = layerTex.Count();
		layerTex.AddToTail( pTex );
		layerOf.Insert( key, layer );
		return layer;
	};

	int nTextured = 0, nAvg = 0;
	int maxW = 1, maxH = 1;
	for ( uint32_t i = 0; i < nTris; ++i )
	{
		float *m = &outMetaFloat4s[(size_t)i * (size_t)kVRadFilterMetaFloat4s * 4u];
		const uint32_t id = triIds[i];
		if ( !( id & TRACE_ID_FILTER ) )
			continue;

		const int facenum = (int)( id & 0x00FFFFFFu );
		FilterMaterial_t *pMat = GetFilterForFace( facenum );
		if ( !pMat || !pMat->enabled )
			continue;

		Vector avgT;
		if ( !VRadFilter_AverageFace( facenum, avgT ) )
			avgT.Init( 1, 1, 1 );

		m[0] = avgT.x;
		m[1] = avgT.y;
		m[2] = avgT.z;
		m[3] = 1.0f; // mode = avg fallback

		m[4] = pMat->strength;
		m[5] = pMat->opacity;
		m[6] = pMat->thickness;
		// flags: 1=clampU, 2=clampV, 4=$nocull (both sides)
		m[7] = pMat->nocull ? 4.0f : 0.0f;

		dface_t *f = &g_pFaces[facenum];
		texinfo_t *tx = &texinfo[f->texinfo];
		for ( int k = 0; k < 4; ++k )
		{
			m[8 + k] = tx->textureVecsTexelsPerWorldUnits[0][k];
			m[12 + k] = tx->textureVecsTexelsPerWorldUnits[1][k];
		}

		const int layer = GetLayer( pMat->pTex );
		if ( layer >= 0 )
		{
			FilterRGBA_t *pTex = layerTex[layer];
			m[3] = 2.0f; // textured
			m[7] = ( pTex->clampU ? 1.0f : 0.0f ) + ( pTex->clampV ? 2.0f : 0.0f ) + ( pMat->nocull ? 4.0f : 0.0f );
			m[16] = (float)layer;
			// y = scale from full-res textureVecs -> loaded mip (fixes false tiling)
			m[17] = (float)pTex->width / (float)max( pTex->origWidth, 1 );
			m[18] = (float)pTex->width;
			m[19] = (float)pTex->height;
			maxW = max( maxW, pTex->width );
			maxH = max( maxH, pTex->height );
			++nTextured;
		}
		else
		{
			++nAvg;
		}
	}

	if ( layerTex.Count() == 0 )
	{
		if ( nAvg > 0 )
			Msg( "[PathTrace-DXR] GPU TriFilter: %d avg-only filter tri(s) (no textures).\n", nAvg );
		return true;
	}

	outArray.width = maxW;
	outArray.height = maxH;
	outArray.layers = layerTex.Count();
	outArray.rgba.assign( (size_t)outArray.layers * (size_t)maxW * (size_t)maxH * 4u, 0 );
	for ( int layer = 0; layer < layerTex.Count(); ++layer )
	{
		FilterRGBA_t *pTex = layerTex[layer];
		unsigned char *dstBase = &outArray.rgba[(size_t)layer * (size_t)maxW * (size_t)maxH * 4u];
		for ( int y = 0; y < pTex->height; ++y )
		{
			const unsigned char *src = &pTex->rgba[( y * pTex->width ) * 4];
			unsigned char *dst = dstBase + (size_t)y * (size_t)maxW * 4u;
			memcpy( dst, src, (size_t)pTex->width * 4u );
		}
	}

	Msg( "[PathTrace-DXR] GPU TriFilter array: %dx%d x %d layer(s), %d textured / %d avg-only tri(s).\n",
		 maxW, maxH, layerTex.Count(), nTextured, nAvg );
	return true;
}


