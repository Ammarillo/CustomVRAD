//========= Copyright PathRAD contributors. ============//
// Compile-time roughness / metalness for bounce ($vrad_roughness*, $vrad_metalness*).
//=============================================================================//

#include "vrad_brdf.h"
#include "vrad.h"
#include "pathtrace_dxr.h"
#include "pathtrace_dxr_device.h"
#include "bounce_albedo.h"
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

static const int kBrdfMaxEdge = 256;
static const int kBrdfMaxLayers = 128;

static const int kFlagAuthored = 1;
static const int kFlagRoughMap = 2;
static const int kFlagMetalMap = 4;
static const int kFlagClampU = 8;
static const int kFlagClampV = 16;

struct BrdfTex_t
{
	int width;
	int height;
	bool clampU;
	bool clampV;
	CUtlMemory<unsigned char> rgba;
};

struct BrdfMat_t
{
	bool authored;
	bool hasRoughMap;
	bool hasMetalMap;
	float roughValue; // scalar, or map multiplier
	float metalValue;
	bool clampU;
	bool clampV;
	BrdfTex_t *pRough;
	BrdfTex_t *pMetal;
};

static CUtlDict<BrdfTex_t *, int> g_BrdfTextures;
static CUtlDict<BrdfMat_t *, int> g_BrdfMaterials;
static bool g_bBrdfCacheBuilt = false;
static bool g_bBrdfAny = false;
static int g_nBrdfAuthored = 0;
static int g_nBrdfRoughMaps = 0;
static int g_nBrdfMetalMaps = 0;

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

static void NormalizeMaterialPath( char *p )
{
	Q_FixSlashes( p );
	if ( !Q_strnicmp( p, "materials/", 10 ) )
		Q_memmove( p, p + 10, strlen( p + 10 ) + 1 );
	char *ext = Q_strrchr( p, '.' );
	if ( ext && ( !Q_stricmp( ext, ".vmt" ) || !Q_stricmp( ext, ".vtf" ) ) )
		*ext = 0;
}

static bool IsBrdfKey( const char *n )
{
	if ( !n )
		return false;
	return !Q_strnicmp( n, "$vrad_roughness", 15 ) || !Q_strnicmp( n, "$vrad_metal", 11 );
}

static void OverlayBrdfKeys( KeyValues *pDst, KeyValues *pSrc )
{
	if ( !pDst || !pSrc )
		return;
	for ( KeyValues *pSub = pSrc->GetFirstValue(); pSub; pSub = pSub->GetNextValue() )
	{
		if ( IsBrdfKey( pSub->GetName() ) )
			pDst->SetString( pSub->GetName(), pSub->GetString() );
	}
}

static KeyValues *LoadMergedVmtKeys( const char *pMaterialName, int nDepth = 0 )
{
	if ( !pMaterialName || !pMaterialName[0] || nDepth > 8 )
		return NULL;

	char szMat[MAX_PATH];
	Q_strncpy( szMat, pMaterialName, sizeof( szMat ) );
	NormalizeMaterialPath( szMat );
	if ( !szMat[0] )
		return NULL;

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
			OverlayBrdfKeys( pBase, pVMT );
			pVMT->deleteThis();
			return pBase;
		}
	}
	return pVMT;
}

// maps/<level>/<mat>_<x>_<y>_<z> -> <mat>
static bool ResolveCubemapPatchMaterial( const char *pName, char *pOut, int nOut )
{
	if ( !pName || Q_strncmp( pName, "maps/", 5 ) != 0 )
		return false;

	const char *pAfterMaps = pName + 5;
	const int nLevelLen = (int)Q_strlen( level_name );
	if ( nLevelLen <= 0 || Q_strncmp( pAfterMaps, level_name, nLevelLen ) != 0 )
	{
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

static BrdfTex_t *LoadBrdfVTF( const char *pTexture )
{
	if ( !pTexture || !pTexture[0] )
		return NULL;

	char szClean[MAX_PATH];
	Q_strncpy( szClean, pTexture, sizeof( szClean ) );
	NormalizeMaterialPath( szClean );
	if ( !szClean[0] )
		return NULL;

	int found = g_BrdfTextures.Find( szClean );
	if ( g_BrdfTextures.IsValidIndex( found ) )
		return g_BrdfTextures[found];

	char szPath[MAX_PATH];
	Q_snprintf( szPath, sizeof( szPath ), "materials/%s.vtf", szClean );
	Q_FixSlashes( szPath );

	CUtlBuffer buf;
	if ( !LoadFileToBuffer( szPath, buf ) )
	{
		g_BrdfTextures.Insert( szClean, NULL );
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
		g_BrdfTextures.Insert( szClean, NULL );
		return NULL;
	}
	if ( pTex->IsCubeMap() || ( pTex->Flags() & TEXTUREFLAGS_ENVMAP ) != 0 )
	{
		DestroyVTFTexture( pTex );
		g_BrdfTextures.Insert( szClean, NULL );
		return NULL;
	}

	int iWidth = pTex->Width();
	int iHeight = pTex->Height();
	if ( iWidth <= 0 || iHeight <= 0 || iWidth > 8192 || iHeight > 8192 )
	{
		DestroyVTFTexture( pTex );
		g_BrdfTextures.Insert( szClean, NULL );
		return NULL;
	}

	int mip = 0;
	while ( ( iWidth > kBrdfMaxEdge || iHeight > kBrdfMaxEdge ) && mip + 1 < pTex->MipCount() )
	{
		++mip;
		iWidth = max( 1, iWidth >> 1 );
		iHeight = max( 1, iHeight >> 1 );
	}

	ImageFormat srcFormat = pTex->Format();
	unsigned char *pSrcImage = pTex->ImageData( 0, 0, mip );
	if ( !pSrcImage )
		pSrcImage = pTex->ImageData( 0, 0, 0 );

	const int nDstBytes = ImageLoader::GetMemRequired( iWidth, iHeight, 1, IMAGE_FORMAT_RGBA8888, false );
	unsigned char *pDstImage = new unsigned char[nDstBytes];
	bool bOk = pSrcImage && ImageLoader::ConvertImageFormat( pSrcImage, srcFormat, pDstImage, IMAGE_FORMAT_RGBA8888, iWidth, iHeight, 0, 0 );
	BrdfTex_t *pData = NULL;
	if ( bOk )
	{
		pData = new BrdfTex_t;
		pData->width = iWidth;
		pData->height = iHeight;
		pData->clampU = ( pTex->Flags() & TEXTUREFLAGS_CLAMPS ) != 0;
		pData->clampV = ( pTex->Flags() & TEXTUREFLAGS_CLAMPT ) != 0;
		pData->rgba.EnsureCapacity( nDstBytes );
		memcpy( pData->rgba.Base(), pDstImage, nDstBytes );
		delete[] pDstImage;
	}
	else
	{
		delete[] pDstImage;
		pTex->ConvertImageFormat( IMAGE_FORMAT_RGBA8888, false );
		pSrcImage = pTex->ImageData( 0, 0, 0 );
		iWidth = pTex->Width();
		iHeight = pTex->Height();
		if ( pSrcImage && pTex->Format() == IMAGE_FORMAT_RGBA8888 && iWidth > 0 && iHeight > 0 )
		{
			pData = new BrdfTex_t;
			pData->width = iWidth;
			pData->height = iHeight;
			pData->clampU = ( pTex->Flags() & TEXTUREFLAGS_CLAMPS ) != 0;
			pData->clampV = ( pTex->Flags() & TEXTUREFLAGS_CLAMPT ) != 0;
			const int nBytes = iWidth * iHeight * 4;
			pData->rgba.EnsureCapacity( nBytes );
			memcpy( pData->rgba.Base(), pSrcImage, nBytes );
		}
	}
	DestroyVTFTexture( pTex );
	g_BrdfTextures.Insert( szClean, pData );
	return pData;
}

static float SaturateF( float v )
{
	if ( v < 0.0f )
		return 0.0f;
	if ( v > 1.0f )
		return 1.0f;
	return v;
}

static int WrapOrClamp( int v, int size, bool bClamp )
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

// Channel as painted (byte/255). Roughness and metalness are linear data, not sRGB color.
static float SampleMapR( const BrdfTex_t *pTex, float u, float v )
{
	if ( !pTex || pTex->width <= 0 || pTex->height <= 0 || !pTex->rgba.Base() )
		return 1.0f;
	const int x = WrapOrClamp( (int)floorf( u * (float)pTex->width ), pTex->width, pTex->clampU );
	const int y = WrapOrClamp( (int)floorf( v * (float)pTex->height ), pTex->height, pTex->clampV );
	const int nOff = ( y * pTex->width + x ) * 4;
	if ( nOff < 0 || ( nOff + 3 ) >= pTex->width * pTex->height * 4 )
		return 1.0f;
	return pTex->rgba[nOff] * ( 1.0f / 255.0f );
}

static const char *FirstString( KeyValues *pVMT, const char *a, const char *b )
{
	const char *s = pVMT->GetString( a, NULL );
	if ( s && s[0] )
		return s;
	s = pVMT->GetString( b, NULL );
	if ( s && s[0] )
		return s;
	return NULL;
}

static BrdfMat_t *FindOrLoadBrdfMaterial( const char *pMaterialName )
{
	if ( !pMaterialName || !pMaterialName[0] )
		return NULL;

	char szKey[MAX_PATH];
	Q_strncpy( szKey, pMaterialName, sizeof( szKey ) );
	NormalizeMaterialPath( szKey );

	int idx = g_BrdfMaterials.Find( szKey );
	if ( g_BrdfMaterials.IsValidIndex( idx ) )
		return g_BrdfMaterials[idx];

	char szResolved[MAX_PATH];
	const char *pLookup = szKey;
	if ( ResolveCubemapPatchMaterial( szKey, szResolved, sizeof( szResolved ) ) )
		pLookup = szResolved;

	KeyValues *pVMT = LoadMergedVmtKeys( pLookup );
	if ( !pVMT && pLookup != szKey )
		pVMT = LoadMergedVmtKeys( szKey );
	else if ( pVMT && pLookup != szKey )
	{
		// Cubemap patch file may add $vrad_* on top of the included material.
		KeyValues *pPatch = LoadMergedVmtKeys( szKey );
		if ( pPatch )
		{
			OverlayBrdfKeys( pVMT, pPatch );
			pPatch->deleteThis();
		}
	}

	BrdfMat_t *pMat = new BrdfMat_t;
	memset( pMat, 0, sizeof( *pMat ) );
	pMat->roughValue = 1.0f;
	pMat->metalValue = 0.0f;

	if ( pVMT )
	{
		const bool roughKey = pVMT->FindKey( "$vrad_roughness" ) != NULL;
		const bool metalKey = pVMT->FindKey( "$vrad_metalness" ) != NULL || pVMT->FindKey( "$vrad_metallic" ) != NULL;
		const char *roughMap = pVMT->GetString( "$vrad_roughnessmap", NULL );
		const char *metalMap = FirstString( pVMT, "$vrad_metalnessmap", "$vrad_metallicmap" );
		const bool hasRoughMapName = roughMap && roughMap[0];
		const bool hasMetalMapName = metalMap && metalMap[0];

		pMat->authored = roughKey || metalKey || hasRoughMapName || hasMetalMapName;
		if ( pMat->authored )
		{
			const float roughScalar = roughKey ? SaturateF( pVMT->GetFloat( "$vrad_roughness", 1.0f ) ) : 1.0f;
			// Omitted metalness on an authored material is metal. Explicit 0 stays dielectric.
			// No $vrad_* keys at all stays unauthored Lambertian (metalValue left at 0).
			float metalScalar = 1.0f;
			if ( pVMT->FindKey( "$vrad_metalness" ) )
				metalScalar = SaturateF( pVMT->GetFloat( "$vrad_metalness", 1.0f ) );
			else if ( pVMT->FindKey( "$vrad_metallic" ) )
				metalScalar = SaturateF( pVMT->GetFloat( "$vrad_metallic", 1.0f ) );

			if ( hasRoughMapName )
			{
				pMat->pRough = LoadBrdfVTF( roughMap );
				if ( pMat->pRough )
				{
					pMat->hasRoughMap = true;
					pMat->roughValue = roughKey ? roughScalar : 1.0f;
					pMat->clampU = pMat->pRough->clampU;
					pMat->clampV = pMat->pRough->clampV;
					++g_nBrdfRoughMaps;
				}
				else
				{
					Warning( "Surface BRDF: roughness map '%s' failed to load for '%s' (using %.2f).\n",
						roughMap, szKey, roughScalar );
					pMat->roughValue = roughScalar;
				}
			}
			else
				pMat->roughValue = roughScalar;

			if ( hasMetalMapName )
			{
				pMat->pMetal = LoadBrdfVTF( metalMap );
				if ( pMat->pMetal )
				{
					pMat->hasMetalMap = true;
					pMat->metalValue = metalScalar;
					if ( !pMat->hasRoughMap )
					{
						pMat->clampU = pMat->pMetal->clampU;
						pMat->clampV = pMat->pMetal->clampV;
					}
					++g_nBrdfMetalMaps;
				}
				else
				{
					Warning( "Surface BRDF: metalness map '%s' failed to load for '%s' (using %.2f).\n",
						metalMap, szKey, metalScalar );
					pMat->metalValue = metalScalar;
				}
			}
			else
				pMat->metalValue = metalScalar;

			++g_nBrdfAuthored;
		}
		pVMT->deleteThis();
	}

	g_BrdfMaterials.Insert( szKey, pMat );
	if ( pLookup != szKey )
	{
		int idxRes = g_BrdfMaterials.Find( pLookup );
		if ( !g_BrdfMaterials.IsValidIndex( idxRes ) )
			g_BrdfMaterials.Insert( pLookup, pMat );
	}
	if ( pMat->authored )
		g_bBrdfAny = true;
	return pMat;
}

static void BuildBrdfCache( void )
{
	if ( g_bBrdfCacheBuilt )
		return;
	g_bBrdfCacheBuilt = true;
	g_bBrdfAny = false;
	g_nBrdfAuthored = 0;
	g_nBrdfRoughMaps = 0;
	g_nBrdfMetalMaps = 0;

	for ( int i = 0; i < numtexdata; ++i )
	{
		const char *pName = TexDataStringTable_GetString( dtexdata[i].nameStringTableID );
		if ( !pName || !pName[0] )
			continue;
		FindOrLoadBrdfMaterial( pName );
	}

	if ( g_nBrdfAuthored > 0 )
	{
		Msg( "Surface BRDF ($vrad_roughness / $vrad_metalness): %d material(s), %d roughness map(s), %d metalness map(s).\n",
			g_nBrdfAuthored, g_nBrdfRoughMaps, g_nBrdfMetalMaps );
	}
}

void VRadBrdf_EnsureCache( void )
{
	BuildBrdfCache();
	VRadBrdf_BuildMirrorList();
}

bool VRadBrdf_HasAny( void )
{
	BuildBrdfCache();
	return g_bBrdfAny;
}

static VRadBrdfSample_t SampleMat( const BrdfMat_t *pMat, float u, float v )
{
	VRadBrdfSample_t s;
	s.roughness = 1.0f;
	s.metalness = 0.0f;
	s.authored = false;
	if ( !pMat || !pMat->authored )
		return s;
	s.authored = true;
	if ( pMat->hasRoughMap && pMat->pRough )
		s.roughness = SaturateF( SampleMapR( pMat->pRough, u, v ) * pMat->roughValue );
	else
		s.roughness = SaturateF( pMat->roughValue );
	if ( pMat->hasMetalMap && pMat->pMetal )
		s.metalness = SaturateF( SampleMapR( pMat->pMetal, u, v ) * pMat->metalValue );
	else
		s.metalness = SaturateF( pMat->metalValue );
	return s;
}

static const char *MaterialForFace( int facenum )
{
	if ( facenum < 0 || facenum >= numfaces )
		return NULL;
	const dface_t *f = &g_pFaces[facenum];
	if ( f->texinfo < 0 || f->texinfo >= texinfo.Count() )
		return NULL;
	const int td = texinfo[f->texinfo].texdata;
	if ( td < 0 || td >= numtexdata )
		return NULL;
	return TexDataStringTable_GetString( dtexdata[td].nameStringTableID );
}

static void FaceToUV( int facenum, const Vector &worldPos, float &u, float &v )
{
	u = 0.0f;
	v = 0.0f;
	if ( facenum < 0 || facenum >= numfaces )
		return;
	const dface_t *f = &g_pFaces[facenum];
	if ( f->texinfo < 0 || f->texinfo >= texinfo.Count() )
		return;
	const texinfo_t *tx = &texinfo[f->texinfo];
	float w = 1.0f;
	float h = 1.0f;
	if ( tx->texdata >= 0 && tx->texdata < numtexdata )
	{
		w = (float)dtexdata[tx->texdata].width;
		h = (float)dtexdata[tx->texdata].height;
	}
	if ( w < 1.0f )
		w = 1.0f;
	if ( h < 1.0f )
		h = 1.0f;
	const float *s = tx->textureVecsTexelsPerWorldUnits[0];
	const float *t = tx->textureVecsTexelsPerWorldUnits[1];
	u = ( worldPos.x * s[0] + worldPos.y * s[1] + worldPos.z * s[2] + s[3] ) / w;
	v = ( worldPos.x * t[0] + worldPos.y * t[1] + worldPos.z * t[2] + t[3] ) / h;
}

VRadBrdfSample_t VRadBrdf_SampleFace( int facenum, const Vector &worldPos )
{
	BuildBrdfCache();
	float u, v;
	FaceToUV( facenum, worldPos, u, v );
	return SampleMat( FindOrLoadBrdfMaterial( MaterialForFace( facenum ) ), u, v );
}

VRadBrdfSample_t VRadBrdf_SampleMaterialUV( const char *pMaterialName, float u, float v )
{
	BuildBrdfCache();
	return SampleMat( FindOrLoadBrdfMaterial( pMaterialName ), u, v );
}

static float Luma( const Vector &v )
{
	return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

VRadBrdfLobes_t VRadBrdf_Lobes( const Vector &albedo, float roughness, float metalness )
{
	VRadBrdfLobes_t L;
	L.wide = albedo;
	L.sharp.Init();
	L.pSpec = 0.0f;
	L.legacy = true;

	roughness = SaturateF( roughness );
	metalness = SaturateF( metalness );
	if ( roughness >= 0.999f && metalness <= 0.001f )
		return L;

	const float gloss = 1.0f - roughness;
	Vector F0;
	F0.x = 0.04f + ( albedo.x - 0.04f ) * metalness;
	F0.y = 0.04f + ( albedo.y - 0.04f ) * metalness;
	F0.z = 0.04f + ( albedo.z - 0.04f ) * metalness;
	Vector diff;
	diff.x = albedo.x * ( 1.0f - metalness );
	diff.y = albedo.y * ( 1.0f - metalness );
	diff.z = albedo.z * ( 1.0f - metalness );

	L.sharp.x = F0.x * gloss;
	L.sharp.y = F0.y * gloss;
	L.sharp.z = F0.z * gloss;
	const float wideDielectric = 1.0f - 0.04f * gloss;
	const float wideMetal = ( 1.0f - gloss ) * metalness;
	L.wide.x = diff.x * wideDielectric + F0.x * wideMetal;
	L.wide.y = diff.y * wideDielectric + F0.y * wideMetal;
	L.wide.z = diff.z * wideDielectric + F0.z * wideMetal;

	const float specL = Luma( L.sharp );
	const float wideL = Luma( L.wide );
	L.pSpec = specL / ( specL + wideL + 1e-5f );
	if ( L.pSpec < 0.0f )
		L.pSpec = 0.0f;
	if ( L.pSpec > 1.0f )
		L.pSpec = 1.0f;
	L.legacy = false;
	return L;
}

float VRadBrdf_GgxShape( const Vector &n, const Vector &wo, const Vector &wi, float roughness )
{
	const float ndotl = DotProduct( n, wi );
	const float ndotv = DotProduct( n, wo );
	if ( ndotl <= 1e-4f || ndotv <= 1e-4f )
		return 0.0f;
	// Near-mirrors are connected explicitly. A tiny GGX peak never hits a point light.
	if ( roughness < kVRadBrdfMirrorRough )
		return 0.0f;

	float r = roughness;
	if ( r < 0.02f )
		r = 0.02f;
	if ( r > 1.0f )
		r = 1.0f;
	const float alpha = r * r;
	const float a2 = alpha * alpha;
	Vector H = wo + wi;
	if ( VectorNormalize( H ) < 1e-6f )
		return 0.0f;
	const float ndoth = max( 0.0f, DotProduct( n, H ) );
	const float d = ndoth * ndoth * ( a2 - 1.0f ) + 1.0f;
	const float D = a2 / ( (float)M_PI * d * d + 1e-7f );
	const float k = ( r + 1.0f ) * ( r + 1.0f ) / 8.0f;
	const float G1v = ndotv / ( ndotv * ( 1.0f - k ) + k );
	const float G1l = ndotl / ( ndotl * ( 1.0f - k ) + k );
	const float G = G1v * G1l;
	return ( (float)M_PI * D * G ) / ( 4.0f * ndotl * ndotv + 1e-4f );
}

Vector VRadBrdf_SampleGloss( const Vector &n, const Vector &incomingDir, float roughness, float u1, float u2 )
{
	// incomingDir points at the surface. Heitz GGX VNDF (JCGT 2018) needs the view direction.
	Vector wo = -incomingDir;
	VectorNormalize( wo );
	Vector mirror = n * ( 2.0f * DotProduct( n, wo ) ) - wo;
	VectorNormalize( mirror );
	if ( roughness < kVRadBrdfMirrorRough )
		return mirror;

	float r = SaturateF( roughness );
	const float alpha = max( r * r, 1e-4f );

	Vector t, b;
	if ( fabsf( n.x ) > 0.1f )
		t.Init( n.y, -n.x, 0 );
	else
		t.Init( 0, n.z, -n.y );
	VectorNormalize( t );
	CrossProduct( n, t, b );

	const float wox = DotProduct( wo, t );
	const float woy = DotProduct( wo, b );
	const float woz = max( DotProduct( wo, n ), 0.0f );
	Vector Vh( alpha * wox, alpha * woy, woz );
	if ( VectorNormalize( Vh ) < 1e-6f )
		return mirror;

	const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	Vector T1;
	if ( lensq > 1e-7f )
	{
		const float invLen = 1.0f / sqrtf( lensq );
		T1.Init( -Vh.y * invLen, Vh.x * invLen, 0 );
	}
	else
		T1.Init( 1, 0, 0 );
	Vector T2;
	CrossProduct( Vh, T1, T2 );

	const float ru = sqrtf( max( u1, 0.0f ) );
	const float phi = u2 * 2.0f * (float)M_PI;
	float t1 = ru * cosf( phi );
	float t2 = ru * sinf( phi );
	const float s = 0.5f * ( 1.0f + Vh.z );
	t2 = ( 1.0f - s ) * sqrtf( max( 0.0f, 1.0f - t1 * t1 ) ) + s * t2;
	const float nz = sqrtf( max( 0.0f, 1.0f - t1 * t1 - t2 * t2 ) );
	Vector Nh = T1 * t1 + T2 * t2 + Vh * nz;
	if ( VectorNormalize( Nh ) < 1e-6f )
		return mirror;

	Vector Ne( alpha * Nh.x, alpha * Nh.y, max( Nh.z, 0.0f ) );
	if ( VectorNormalize( Ne ) < 1e-6f )
		return mirror;

	Vector H = t * Ne.x + b * Ne.y + n * Ne.z;
	if ( VectorNormalize( H ) < 1e-6f )
		return mirror;
	Vector R = H * ( 2.0f * DotProduct( wo, H ) ) - wo;
	if ( DotProduct( R, n ) <= 0.0f )
		return mirror;
	VectorNormalize( R );
	return R;
}

void VRadBrdf_ApplyPatchReflectivity( void )
{
	BuildBrdfCache();
	if ( !g_bBrdfAny )
		return;

	int n = 0;
	const unsigned int uiPatchCount = g_Patches.Size();
	for ( unsigned int i = 0; i < uiPatchCount; ++i )
	{
		CPatch *patch = &g_Patches[i];
		if ( patch->child1 != g_Patches.InvalidIndex() )
			continue;
		if ( patch->sky )
			continue;

		VRadBrdfSample_t s = VRadBrdf_SampleFace( patch->faceNumber, patch->origin );
		if ( !s.authored )
			continue;
		VRadBrdfLobes_t lobes = VRadBrdf_Lobes( patch->reflectivity, s.roughness, s.metalness );
		if ( lobes.legacy )
			continue;
		patch->reflectivity = lobes.wide;
		for ( int c = 0; c < 3; ++c )
		{
			if ( patch->reflectivity[c] < 0.0f )
				patch->reflectivity[c] = 0.0f;
			if ( patch->reflectivity[c] > 0.99f )
				patch->reflectivity[c] = 0.99f;
		}
		++n;
	}
	if ( n > 0 )
	{
		Msg( "Surface BRDF: classic radiosity keeps the diffuse lobe on %d patches (smooth metals do not diffuse-fill).\n", n );
	}
}

static int FaceFromTriFlags( uint32_t flags )
{
	if ( flags & TRACE_ID_STATICPROP )
		return -1;
	if ( flags & TRACE_ID_FILTER )
		return (int)( flags & 0x00FFFFFFu );
	if ( flags & TRACE_ID_OPAQUE )
	{
		const unsigned packed = flags & 0x00FFFFFFu;
		if ( packed != 0 )
			return (int)packed - 1;
	}
	return -1;
}

static void BlitMapLayer( unsigned char *pLayer, int atlasW, int atlasH, const BrdfMat_t *pMat )
{
	for ( int y = 0; y < atlasH; ++y )
	{
		for ( int x = 0; x < atlasW; ++x )
		{
			const float u = ( (float)x + 0.5f ) / (float)atlasW;
			const float v = ( (float)y + 0.5f ) / (float)atlasH;
			float rough = 1.0f;
			float metal = 0.0f;
			if ( pMat->hasRoughMap && pMat->pRough )
				rough = SampleMapR( pMat->pRough, u, v );
			if ( pMat->hasMetalMap && pMat->pMetal )
				metal = SampleMapR( pMat->pMetal, u, v );
			unsigned char *p = pLayer + ( ( y * atlasW + x ) * 4 );
			p[0] = (unsigned char)( SaturateF( rough ) * 255.0f + 0.5f );
			p[1] = (unsigned char)( SaturateF( metal ) * 255.0f + 0.5f );
			p[2] = 0;
			p[3] = 255;
		}
	}
}

void VRadBrdf_BuildGpuUpload( VRadBrdfGpuArray_t &outArray, std::vector<float> &outMeta )
{
	BuildBrdfCache();
	const uint32_t n = PathTraceDXR_CapturedTriCount();
	outMeta.assign( (size_t)n * (size_t)kVRadBrdfMetaFloat4s * 4u, 0.0f );
	for ( uint32_t i = 0; i < n; ++i )
	{
		float *m = &outMeta[(size_t)i * (size_t)kVRadBrdfMetaFloat4s * 4u];
		m[0] = 1.0f; // roughness
		m[1] = 0.0f; // metalness
		m[2] = -1.0f; // layer
		m[3] = 0.0f; // flags
	}
	outArray.width = 1;
	outArray.height = 1;
	outArray.layers = 1;
	outArray.rgba.assign( 4, 255 );

	if ( n == 0 || !g_bBrdfAny )
		return;

	int atlasW = 1;
	int atlasH = 1;
	CUtlDict<int, int> layerOfMat;
	CUtlVector<BrdfMat_t *> layerMats;

	for ( uint32_t i = 0; i < n; ++i )
	{
		const char *matName = PathTraceDXR_CapturedTriMaterial( i );
		int face = -1;
		uint32_t flags = 0;
		Vector a, b, c;
		if ( !PathTraceDXR_GetCapturedTri( i, a, b, c, &flags ) )
			continue;
		if ( !matName || !matName[0] )
		{
			face = FaceFromTriFlags( flags );
			matName = MaterialForFace( face );
		}
		BrdfMat_t *pMat = FindOrLoadBrdfMaterial( matName );
		if ( !pMat || !pMat->authored || ( !pMat->hasRoughMap && !pMat->hasMetalMap ) )
			continue;

		char szKey[MAX_PATH];
		Q_strncpy( szKey, matName, sizeof( szKey ) );
		NormalizeMaterialPath( szKey );
		int found = layerOfMat.Find( szKey );
		if ( !layerOfMat.IsValidIndex( found ) )
		{
			if ( layerMats.Count() >= kBrdfMaxLayers )
				continue;
			const int layer = layerMats.AddToTail( pMat );
			layerOfMat.Insert( szKey, layer );
			const BrdfTex_t *pSize = pMat->pRough ? pMat->pRough : pMat->pMetal;
			if ( pSize )
			{
				atlasW = max( atlasW, pSize->width );
				atlasH = max( atlasH, pSize->height );
			}
		}
	}

	if ( atlasW > kBrdfMaxEdge )
		atlasW = kBrdfMaxEdge;
	if ( atlasH > kBrdfMaxEdge )
		atlasH = kBrdfMaxEdge;
	const int nLayers = max( 1, layerMats.Count() );
	outArray.width = atlasW;
	outArray.height = atlasH;
	outArray.layers = nLayers;
	outArray.rgba.assign( (size_t)nLayers * (size_t)atlasW * (size_t)atlasH * 4u, 255 );
	for ( int layer = 0; layer < layerMats.Count(); ++layer )
	{
		unsigned char *pLayer = outArray.rgba.data() + (size_t)layer * (size_t)atlasW * (size_t)atlasH * 4u;
		BlitMapLayer( pLayer, atlasW, atlasH, layerMats[layer] );
	}

	int nMapped = 0;
	int nScalar = 0;
	for ( uint32_t i = 0; i < n; ++i )
	{
		float *m = &outMeta[(size_t)i * (size_t)kVRadBrdfMetaFloat4s * 4u];
		const char *matName = PathTraceDXR_CapturedTriMaterial( i );
		uint32_t flags = 0;
		Vector a, b, c;
		if ( !PathTraceDXR_GetCapturedTri( i, a, b, c, &flags ) )
			continue;
		int face = -1;
		if ( !matName || !matName[0] )
		{
			face = FaceFromTriFlags( flags );
			matName = MaterialForFace( face );
		}
		BrdfMat_t *pMat = FindOrLoadBrdfMaterial( matName );
		if ( !pMat || !pMat->authored )
			continue;

		int flagsBits = kFlagAuthored;
		if ( pMat->clampU )
			flagsBits |= kFlagClampU;
		if ( pMat->clampV )
			flagsBits |= kFlagClampV;
		m[0] = pMat->roughValue;
		m[1] = pMat->metalValue;
		m[2] = -1.0f;

		float uv[6];
		bool hasUv = PathTraceDXR_CapturedTriUV( i, uv );
		if ( !hasUv && face >= 0 )
		{
			FaceToUV( face, a, uv[0], uv[1] );
			FaceToUV( face, b, uv[2], uv[3] );
			FaceToUV( face, c, uv[4], uv[5] );
			hasUv = true;
		}

		if ( pMat->hasRoughMap || pMat->hasMetalMap )
		{
			char szKey[MAX_PATH];
			Q_strncpy( szKey, matName ? matName : "", sizeof( szKey ) );
			NormalizeMaterialPath( szKey );
			int found = layerOfMat.Find( szKey );
			if ( layerOfMat.IsValidIndex( found ) && hasUv )
			{
				if ( pMat->hasRoughMap )
					flagsBits |= kFlagRoughMap;
				if ( pMat->hasMetalMap )
					flagsBits |= kFlagMetalMap;
				m[2] = (float)layerOfMat[found];
				m[4] = uv[0]; m[5] = uv[1];
				m[6] = uv[2]; m[7] = uv[3];
				m[8] = uv[4]; m[9] = uv[5];
				++nMapped;
			}
			else
			{
				// No UV or the atlas is full: one texel at the triangle center.
				VRadBrdfSample_t s = hasUv
					? SampleMat( pMat, ( uv[0] + uv[2] + uv[4] ) / 3.0f, ( uv[1] + uv[3] + uv[5] ) / 3.0f )
					: SampleMat( pMat, 0.5f, 0.5f );
				m[0] = s.roughness;
				m[1] = s.metalness;
				m[2] = -1.0f;
				++nScalar;
			}
		}
		else
			++nScalar;
		m[3] = (float)flagsBits;
	}

	if ( nMapped > 0 || nScalar > 0 )
	{
		Msg( "Surface BRDF GPU: %d textured tris, %d scalar tris, atlas %dx%d x %d.\n",
			nMapped, nScalar, atlasW, atlasH, layerMats.Count() );
	}
	VRadBrdf_BuildMirrorList();
}

static std::vector<VRadBrdfMirrorTri_t> g_MirrorTris;
static const int kMirrorCap = 4096;

void VRadBrdf_BuildMirrorList( void )
{
	BuildBrdfCache();
	g_MirrorTris.clear();
	const uint32_t n = PathTraceDXR_CapturedTriCount();
	if ( n == 0 || !g_bBrdfAny )
		return;
	BounceAlbedo_EnsureCache();

	int nSkip = 0;
	for ( uint32_t i = 0; i < n; ++i )
	{
		Vector a, b, c;
		uint32_t flags = 0;
		if ( !PathTraceDXR_GetCapturedTri( i, a, b, c, &flags ) )
			continue;
		const int face = FaceFromTriFlags( flags );
		const Vector centroid = ( a + b + c ) * ( 1.0f / 3.0f );
		VRadBrdfSample_t s;
		if ( flags & TRACE_ID_STATICPROP )
		{
			const char *mat = PathTraceDXR_CapturedTriMaterial( i );
			float uv[6];
			float u = 0.5f, v = 0.5f;
			if ( PathTraceDXR_CapturedTriUV( i, uv ) )
			{
				u = ( uv[0] + uv[2] + uv[4] ) / 3.0f;
				v = ( uv[1] + uv[3] + uv[5] ) / 3.0f;
			}
			s = VRadBrdf_SampleMaterialUV( mat, u, v );
		}
		else
			s = VRadBrdf_SampleFace( face, centroid );
		// Roughness near 1 has no sharp lobe. Below that, direct lights connect
		// through the triangle (steady; a reflection ray speckles seams).
		if ( !s.authored || s.roughness >= 0.98f )
			continue;
		if ( (int)g_MirrorTris.size() >= kMirrorCap )
		{
			++nSkip;
			continue;
		}

		Vector e1 = b - a;
		Vector e2 = c - a;
		Vector nrm;
		CrossProduct( e1, e2, nrm );
		if ( VectorNormalize( nrm ) < 1e-6f )
			continue;
		if ( face >= 0 && face < numfaces )
		{
			const Vector fn = dplanes[g_pFaces[face].planenum].normal;
			if ( DotProduct( nrm, fn ) < 0.0f )
				nrm = -nrm;
		}

		Vector alb( 0.45f, 0.45f, 0.45f );
		if ( flags & TRACE_ID_STATICPROP )
		{
			Vector propAlb;
			if ( PathTraceDXR_GetCapturedTriAlbedo( i, propAlb ) )
				alb = propAlb;
		}
		else if ( face >= 0 )
			BounceAlbedo_SampleFace( face, centroid, alb );
		alb.x = SaturateF( alb.x );
		alb.y = SaturateF( alb.y );
		alb.z = SaturateF( alb.z );

		VRadBrdfMirrorTri_t m;
		m.v0 = a;
		m.v1 = b;
		m.v2 = c;
		m.n = nrm;
		m.f0.x = 0.04f + ( alb.x - 0.04f ) * s.metalness;
		m.f0.y = 0.04f + ( alb.y - 0.04f ) * s.metalness;
		m.f0.z = 0.04f + ( alb.z - 0.04f ) * s.metalness;
		m.face = face;
		m.roughness = s.roughness;
		g_MirrorTris.push_back( m );
	}
	if ( !g_MirrorTris.empty() || nSkip > 0 )
	{
		Msg( "Surface BRDF gloss: %d triangle(s) roughness < 0.98 (steady light connection)%s.\n",
			(int)g_MirrorTris.size(),
			nSkip > 0 ? ", list capped" : "" );
	}
}

int VRadBrdf_MirrorCount( void )
{
	return (int)g_MirrorTris.size();
}

const VRadBrdfMirrorTri_t *VRadBrdf_MirrorTris( void )
{
	return g_MirrorTris.empty() ? NULL : &g_MirrorTris[0];
}

void VRadBrdf_PackMirrorGpu( std::vector<float> &outFloats )
{
	const int n = max( 1, (int)g_MirrorTris.size() );
	outFloats.assign( (size_t)n * 20u, 0.0f );
	for ( int i = 0; i < (int)g_MirrorTris.size(); ++i )
	{
		const VRadBrdfMirrorTri_t &m = g_MirrorTris[i];
		float *p = &outFloats[(size_t)i * 20u];
		p[0] = m.v0.x; p[1] = m.v0.y; p[2] = m.v0.z; p[3] = (float)m.face;
		p[4] = m.v1.x; p[5] = m.v1.y; p[6] = m.v1.z; p[7] = m.roughness;
		p[8] = m.v2.x; p[9] = m.v2.y; p[10] = m.v2.z; p[11] = 0.0f;
		p[12] = m.n.x; p[13] = m.n.y; p[14] = m.n.z; p[15] = 0.0f;
		p[16] = m.f0.x; p[17] = m.f0.y; p[18] = m.f0.z; p[19] = 0.0f;
	}
}
