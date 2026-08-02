//========= Copyright CustomVRAD contributors. ============//
// Per-texel $basetexture sampling for textured radiosity bounce (-texbounce).
//=============================================================================//

#include "bounce_albedo.h"
#include "vrad.h"
#include "bsplib.h"
#include "cmdlib.h"
#include "vtf/vtf.h"
#include "tier1/utldict.h"
#include "tier1/utlbuffer.h"
#include "tier1/KeyValues.h"
#include "bitmap/imageformat.h"
#include "tier1/strtools.h"

struct BounceAlbedoTexture_t
{
	int width;
	int height;
	bool clampU;
	bool clampV;
	CUtlMemory<unsigned char> rgba; // RGBA8888
};

static CUtlDict<BounceAlbedoTexture_t *, int> g_AlbedoByMaterial;
static CUtlVector<BounceAlbedoTexture_t *> g_AlbedoByTexdata; // parallel to dtexdata, nullable
static bool g_bAlbedoCacheBuilt = false;
static int g_nAlbedoLoaded = 0;
static int g_nAlbedoMissing = 0;

extern char level_name[MAX_PATH];
extern bool g_bTexturedBounce;

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

// maps/<level>/<mat>_<x>_<y>_<z>  →  <mat>  (same as LightForTexture)
static bool ResolveCubemapPatchMaterial( const char *pName, char *pOut, int nOut )
{
	if ( !pName || Q_strncmp( pName, "maps/", 5 ) != 0 )
		return false;

	const char *pAfterMaps = pName + 5;
	const int nLevelLen = (int)Q_strlen( level_name );
	if ( nLevelLen <= 0 || Q_strncmp( pAfterMaps, level_name, nLevelLen ) != 0 )
	{
		// Fallback: skip first path component after maps/
		const char *slash = strchr( pAfterMaps, '/' );
		if ( !slash )
			return false;
		Q_strncpy( pOut, slash + 1, nOut );
	}
	else
	{
		const char *base = pAfterMaps + nLevelLen;
		if ( *base != '/' )
			return false;
		Q_strncpy( pOut, base + 1, nOut );
	}

	bool foundSeparators = true;
	for ( int i = 0; i < 3; ++i )
	{
		char *underscore = Q_strrchr( pOut, '_' );
		if ( underscore && *underscore )
			*underscore = '\0';
		else
			foundSeparators = false;
	}
	return foundSeparators && pOut[0] != 0;
}

static bool ShouldSkipMaterial( const char *pMaterialName )
{
	if ( !pMaterialName || !pMaterialName[0] )
		return true;
	if ( !Q_strnicmp( pMaterialName, "tools/", 6 ) ||
		 !Q_strnicmp( pMaterialName, "skybox/", 7 ) ||
		 !Q_strnicmp( pMaterialName, "water/", 6 ) ||
		 !Q_strnicmp( pMaterialName, "nature/water", 12 ) ||
		 !Q_strnicmp( pMaterialName, "effects/", 8 ) ||
		 !Q_strnicmp( pMaterialName, "debug/", 6 ) ||
		 !Q_strnicmp( pMaterialName, "lights/", 7 ) )
	{
		return true;
	}
	return false;
}

static BounceAlbedoTexture_t *LoadAlbedoFromBaseTextureName( const char *pBaseTexture )
{
	if ( !pBaseTexture || !pBaseTexture[0] )
		return NULL;

	// Strip materials/ prefix and .vtf if a full path leaked in
	char szClean[MAX_PATH];
	Q_strncpy( szClean, pBaseTexture, sizeof( szClean ) );
	Q_FixSlashes( szClean );
	if ( !Q_strnicmp( szClean, "materials/", 10 ) )
		Q_memmove( szClean, szClean + 10, strlen( szClean + 10 ) + 1 );
	char *ext = Q_strrchr( szClean, '.' );
	if ( ext && ( !Q_stricmp( ext, ".vtf" ) || !Q_stricmp( ext, ".vmt" ) ) )
		*ext = 0;

	char szPath[MAX_PATH];
	Q_snprintf( szPath, sizeof( szPath ), "materials/%s.vtf", szClean );
	Q_FixSlashes( szPath );

	CUtlBuffer buf;
	if ( !LoadFileToBuffer( szPath, buf ) )
		return NULL;

	// SDK VTF loader stops at 7.4; many modern VTFs are 7.5 with a compatible layout.
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
		return NULL;
	}

	if ( pTex->IsCubeMap() || ( pTex->Flags() & TEXTUREFLAGS_ENVMAP ) != 0 )
	{
		DestroyVTFTexture( pTex );
		return NULL;
	}

	// Bounce only needs low-frequency albedo — pick the smallest mip whose
	// max edge is still >= 256 (or the lowest mip). Cuts VRAM and sample cost.
	const int kMaxBounceAlbedoEdge = 256;
	int mip = 0;
	int iWidth = pTex->Width();
	int iHeight = pTex->Height();
	const int nMips = pTex->MipCount();
	while ( mip + 1 < nMips && ( iWidth > kMaxBounceAlbedoEdge || iHeight > kMaxBounceAlbedoEdge ) )
	{
		++mip;
		iWidth = max( 1, iWidth >> 1 );
		iHeight = max( 1, iHeight >> 1 );
	}
	// Prefer API sizes if available via ImageData dimensions at this mip.
	if ( iWidth <= 0 || iHeight <= 0 || iWidth > 8192 || iHeight > 8192 )
	{
		DestroyVTFTexture( pTex );
		return NULL;
	}

	ImageFormat srcFormat = pTex->Format();
	ImageFormat dstFormat = IMAGE_FORMAT_RGBA8888;
	unsigned char *pSrcImage = pTex->ImageData( 0, 0, mip );
	if ( !pSrcImage )
	{
		DestroyVTFTexture( pTex );
		return NULL;
	}

	const int nDstBytes = ImageLoader::GetMemRequired( iWidth, iHeight, 1, dstFormat, false );
	unsigned char *pDstImage = new unsigned char[nDstBytes];
	bool bConverted = ImageLoader::ConvertImageFormat( pSrcImage, srcFormat, pDstImage, dstFormat, iWidth, iHeight, 0, 0 );
	if ( !bConverted )
	{
		delete[] pDstImage;
		// Fall back: convert full texture then (still use mip 0 after convert — rare path).
		pTex->ConvertImageFormat( IMAGE_FORMAT_RGBA8888, false );
		pSrcImage = pTex->ImageData( 0, 0, mip );
		if ( !pSrcImage || pTex->Format() != IMAGE_FORMAT_RGBA8888 )
		{
			DestroyVTFTexture( pTex );
			return NULL;
		}
		// Recompute dims from converted texture mip chain
		iWidth = pTex->Width();
		iHeight = pTex->Height();
		for ( int m = 0; m < mip; ++m )
		{
			iWidth = max( 1, iWidth >> 1 );
			iHeight = max( 1, iHeight >> 1 );
		}
		BounceAlbedoTexture_t *pData = new BounceAlbedoTexture_t;
		pData->width = iWidth;
		pData->height = iHeight;
		pData->clampU = ( pTex->Flags() & TEXTUREFLAGS_CLAMPS ) != 0;
		pData->clampV = ( pTex->Flags() & TEXTUREFLAGS_CLAMPT ) != 0;
		const int nBytes = pData->width * pData->height * 4;
		pData->rgba.EnsureCapacity( nBytes );
		memcpy( pData->rgba.Base(), pSrcImage, nBytes );
		DestroyVTFTexture( pTex );
		return pData;
	}

	BounceAlbedoTexture_t *pData = new BounceAlbedoTexture_t;
	pData->width = iWidth;
	pData->height = iHeight;
	pData->clampU = ( pTex->Flags() & TEXTUREFLAGS_CLAMPS ) != 0;
	pData->clampV = ( pTex->Flags() & TEXTUREFLAGS_CLAMPT ) != 0;
	pData->rgba.EnsureCapacity( nDstBytes );
	memcpy( pData->rgba.Base(), pDstImage, nDstBytes );
	delete[] pDstImage;
	DestroyVTFTexture( pTex );
	return pData;
}

static void NormalizeMaterialPath( char *pPath )
{
	Q_FixSlashes( pPath );
	if ( !Q_strnicmp( pPath, "materials/", 10 ) )
		Q_memmove( pPath, pPath + 10, strlen( pPath + 10 ) + 1 );
	char *ext = Q_strrchr( pPath, '.' );
	if ( ext && !Q_stricmp( ext, ".vmt" ) )
		*ext = 0;
}

// Returns $basetexture path into pOut. Follows patch { include "..." } once.
static const char *ReadBaseTextureFromVmt( const char *pMaterialName, char *pOut, int nOut, int nDepth = 0 )
{
	pOut[0] = 0;
	if ( !pMaterialName || !pMaterialName[0] || nDepth > 4 )
		return NULL;

	char szMat[MAX_PATH];
	Q_strncpy( szMat, pMaterialName, sizeof( szMat ) );
	NormalizeMaterialPath( szMat );

	char szVmt[MAX_PATH];
	Q_snprintf( szVmt, sizeof( szVmt ), "materials/%s.vmt", szMat );
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

	const char *pBase = pVMT->GetString( "$basetexture", NULL );
	if ( !pBase || !pBase[0] )
		pBase = pVMT->GetString( "$basetexture2", NULL );
	if ( pBase && pBase[0] )
	{
		Q_strncpy( pOut, pBase, nOut );
		pVMT->deleteThis();
		return pOut;
	}

	// Cubemap / patched materials: patch { include "materials/foo.vmt" ... }
	const char *pInclude = pVMT->GetString( "include", NULL );
	if ( ( !pInclude || !pInclude[0] ) && pVMT->FindKey( "patch" ) )
	{
		KeyValues *pPatch = pVMT->FindKey( "patch" );
		if ( pPatch )
			pInclude = pPatch->GetString( "include", NULL );
	}
	// Sometimes the root node itself is named "patch"
	if ( ( !pInclude || !pInclude[0] ) && !Q_stricmp( pVMT->GetName(), "patch" ) )
		pInclude = pVMT->GetString( "include", NULL );

	if ( pInclude && pInclude[0] )
	{
		char szInclude[MAX_PATH];
		Q_strncpy( szInclude, pInclude, sizeof( szInclude ) );
		NormalizeMaterialPath( szInclude );
		pVMT->deleteThis();
		return ReadBaseTextureFromVmt( szInclude, pOut, nOut, nDepth + 1 );
	}

	pVMT->deleteThis();
	return NULL;
}

static BounceAlbedoTexture_t *FindOrLoadAlbedoForMaterial( const char *pMaterialName )
{
	int idx = g_AlbedoByMaterial.Find( pMaterialName );
	if ( g_AlbedoByMaterial.IsValidIndex( idx ) )
		return g_AlbedoByMaterial[idx];

	if ( ShouldSkipMaterial( pMaterialName ) )
	{
		g_AlbedoByMaterial.Insert( pMaterialName, NULL );
		return NULL;
	}

	char szResolved[MAX_PATH];
	szResolved[0] = 0;
	const char *pLookupName = pMaterialName;
	if ( ResolveCubemapPatchMaterial( pMaterialName, szResolved, sizeof( szResolved ) ) )
	{
		pLookupName = szResolved;
		if ( ShouldSkipMaterial( pLookupName ) )
		{
			g_AlbedoByMaterial.Insert( pMaterialName, NULL );
			return NULL;
		}
		// Reuse albedo already loaded for the original material
		int idxResolved = g_AlbedoByMaterial.Find( pLookupName );
		if ( g_AlbedoByMaterial.IsValidIndex( idxResolved ) )
		{
			BounceAlbedoTexture_t *pShared = g_AlbedoByMaterial[idxResolved];
			g_AlbedoByMaterial.Insert( pMaterialName, pShared );
			return pShared;
		}
	}

	char szBase[MAX_PATH];
	const char *pBaseTextureName = ReadBaseTextureFromVmt( pLookupName, szBase, sizeof( szBase ) );

	// Also try the unresolved maps/ VMT (pakfile patch) for include → $basetexture
	if ( !pBaseTextureName && pLookupName != pMaterialName )
		pBaseTextureName = ReadBaseTextureFromVmt( pMaterialName, szBase, sizeof( szBase ) );

	BounceAlbedoTexture_t *pData = NULL;
	if ( pBaseTextureName )
		pData = LoadAlbedoFromBaseTextureName( pBaseTextureName );

	// Material path often matches the VTF name
	if ( !pData )
		pData = LoadAlbedoFromBaseTextureName( pLookupName );

	if ( pData )
	{
		++g_nAlbedoLoaded;
		if ( verbose )
		{
			Msg( "Textured bounce: loaded albedo %s (%dx%d) [base=%s]\n",
				 pMaterialName, pData->width, pData->height,
				 pBaseTextureName ? pBaseTextureName : pLookupName );
		}
	}
	else
	{
		++g_nAlbedoMissing;
		if ( verbose )
		{
			Warning( "Textured bounce: no albedo for '%s' (flat reflectivity)\n", pMaterialName );
		}
	}

	g_AlbedoByMaterial.Insert( pMaterialName, pData );
	if ( pLookupName != pMaterialName )
		g_AlbedoByMaterial.Insert( pLookupName, pData );
	return pData;
}

static void BuildAlbedoCache( void )
{
	if ( g_bAlbedoCacheBuilt )
		return;
	g_bAlbedoCacheBuilt = true;
	g_nAlbedoLoaded = 0;
	g_nAlbedoMissing = 0;

	g_AlbedoByTexdata.SetCount( numtexdata );
	for ( int i = 0; i < numtexdata; i++ )
		g_AlbedoByTexdata[i] = NULL;

	Msg( "Textured bounce: caching $basetexture for %d texdatas...\n", numtexdata );
	for ( int i = 0; i < numtexdata; i++ )
	{
		const char *pName = TexDataStringTable_GetString( dtexdata[i].nameStringTableID );
		if ( !pName || !pName[0] )
			continue;
		g_AlbedoByTexdata[i] = FindOrLoadAlbedoForMaterial( pName );
	}

	Msg( "Textured bounce (-texbounce): %d albedo textures loaded, %d using flat reflectivity.\n",
		 g_nAlbedoLoaded, g_nAlbedoMissing );
}

void BounceAlbedo_EnsureCache( void )
{
	if ( !g_bTexturedBounce )
		return;
	BuildAlbedoCache();
}

static BounceAlbedoTexture_t *GetAlbedoForTexdata( int texdataIndex )
{
	if ( texdataIndex < 0 || texdataIndex >= numtexdata )
		return NULL;
	BuildAlbedoCache();
	if ( texdataIndex >= g_AlbedoByTexdata.Count() )
		return NULL;
	return g_AlbedoByTexdata[texdataIndex];
}

static inline int WrapOrClamp( int v, int size, bool bClamp )
{
	if ( size <= 0 )
		return 0;
	if ( bClamp )
	{
		if ( v < 0 )
			return 0;
		if ( v >= size )
			return size - 1;
		return v;
	}
	v %= size;
	if ( v < 0 )
		v += size;
	return v;
}

bool BounceAlbedo_SampleFace( int facenum, const Vector &worldPos, Vector &outLinearRGB )
{
	if ( facenum < 0 || facenum >= numfaces )
		return false;

	dface_t *f = &g_pFaces[facenum];
	texinfo_t *tx = &texinfo[f->texinfo];
	BounceAlbedoTexture_t *pTex = GetAlbedoForTexdata( tx->texdata );
	if ( !pTex || pTex->width <= 0 || pTex->height <= 0 || !pTex->rgba.Base() )
		return false;

	float s = worldPos.x * tx->textureVecsTexelsPerWorldUnits[0][0]
		+ worldPos.y * tx->textureVecsTexelsPerWorldUnits[0][1]
		+ worldPos.z * tx->textureVecsTexelsPerWorldUnits[0][2]
		+ tx->textureVecsTexelsPerWorldUnits[0][3];
	float t = worldPos.x * tx->textureVecsTexelsPerWorldUnits[1][0]
		+ worldPos.y * tx->textureVecsTexelsPerWorldUnits[1][1]
		+ worldPos.z * tx->textureVecsTexelsPerWorldUnits[1][2]
		+ tx->textureVecsTexelsPerWorldUnits[1][3];

	int u = WrapOrClamp( (int)floorf( s ), pTex->width, pTex->clampU );
	int v = WrapOrClamp( (int)floorf( t ), pTex->height, pTex->clampV );

	const int nOff = ( v * pTex->width + u ) * 4;
	if ( nOff < 0 || ( nOff + 3 ) >= pTex->width * pTex->height * 4 )
		return false;

	const unsigned char *p = &pTex->rgba[nOff];
	// Match Valve texdata reflectivity: approximate sRGB -> linear via square
	float r = p[0] * ( 1.0f / 255.0f );
	float g = p[1] * ( 1.0f / 255.0f );
	float b = p[2] * ( 1.0f / 255.0f );
	outLinearRGB.x = r * r;
	outLinearRGB.y = g * g;
	outLinearRGB.z = b * b;
	return true;
}
