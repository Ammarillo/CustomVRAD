//========= Copyright PathRAD contributors. ============//
// Textured emission from VMT ($vrad_emit*).
//=============================================================================//

#include "vrad_emit.h"
#include "vrad.h"
#include "lightmap.h"
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

// from lightmap.cpp
directlight_t *AllocDLight( Vector& origin, bool bAddToList );

struct EmitRGBATexture_t
{
	int width;
	int height;
	bool clampU;
	bool clampV;
	CUtlMemory<unsigned char> rgba;
};

struct EmitMaterial_t
{
	bool enabled;
	float strength;		// lights.rad-style scaler
	float density;		// 1 = default; higher = denser lights / finer chop
	EmitRGBATexture_t *pColor;	// emitmap or basetexture
	EmitRGBATexture_t *pMask;	// optional
	char colorName[MAX_PATH];
	char maskName[MAX_PATH];
};

static CUtlDict<EmitRGBATexture_t *, int> g_EmitTextures;
static CUtlDict<EmitMaterial_t *, int> g_EmitMaterials;
static CUtlVector<EmitMaterial_t *> g_EmitByTexdata;
static bool g_bEmitCacheBuilt = false;

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

static EmitRGBATexture_t *LoadEmitVTF( const char *pBaseTexture )
{
	if ( !pBaseTexture || !pBaseTexture[0] )
		return NULL;

	int found = g_EmitTextures.Find( pBaseTexture );
	if ( g_EmitTextures.IsValidIndex( found ) )
		return g_EmitTextures[found];

	char szPath[MAX_PATH];
	Q_snprintf( szPath, sizeof( szPath ), "materials/%s.vtf", pBaseTexture );
	Q_FixSlashes( szPath );

	CUtlBuffer buf;
	if ( !LoadFileToBuffer( szPath, buf ) )
	{
		g_EmitTextures.Insert( pBaseTexture, NULL );
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
		g_EmitTextures.Insert( pBaseTexture, NULL );
		return NULL;
	}
	if ( pTex->IsCubeMap() || ( pTex->Flags() & TEXTUREFLAGS_ENVMAP ) != 0 )
	{
		DestroyVTFTexture( pTex );
		g_EmitTextures.Insert( pBaseTexture, NULL );
		return NULL;
	}

	const int iWidth = pTex->Width();
	const int iHeight = pTex->Height();
	if ( iWidth <= 0 || iHeight <= 0 || iWidth > 8192 || iHeight > 8192 )
	{
		DestroyVTFTexture( pTex );
		g_EmitTextures.Insert( pBaseTexture, NULL );
		return NULL;
	}

	ImageFormat srcFormat = pTex->Format();
	unsigned char *pSrcImage = pTex->ImageData( 0, 0, 0, 0, 0, 0 );
	if ( !pSrcImage )
	{
		DestroyVTFTexture( pTex );
		g_EmitTextures.Insert( pBaseTexture, NULL );
		return NULL;
	}

	const int nDstBytes = ImageLoader::GetMemRequired( iWidth, iHeight, 1, IMAGE_FORMAT_RGBA8888, false );
	unsigned char *pDstImage = new unsigned char[nDstBytes];
	bool bOk = ImageLoader::ConvertImageFormat( pSrcImage, srcFormat, pDstImage, IMAGE_FORMAT_RGBA8888, iWidth, iHeight, 0, 0 );
	if ( !bOk )
	{
		delete[] pDstImage;
		pTex->ConvertImageFormat( IMAGE_FORMAT_RGBA8888, false );
		pSrcImage = pTex->ImageData( 0, 0, 0, 0, 0, 0 );
		if ( !pSrcImage || pTex->Format() != IMAGE_FORMAT_RGBA8888 )
		{
			DestroyVTFTexture( pTex );
			g_EmitTextures.Insert( pBaseTexture, NULL );
			return NULL;
		}
		EmitRGBATexture_t *pData = new EmitRGBATexture_t;
		pData->width = pTex->Width();
		pData->height = pTex->Height();
		pData->clampU = ( pTex->Flags() & TEXTUREFLAGS_CLAMPS ) != 0;
		pData->clampV = ( pTex->Flags() & TEXTUREFLAGS_CLAMPT ) != 0;
		const int nBytes = pData->width * pData->height * 4;
		pData->rgba.EnsureCapacity( nBytes );
		memcpy( pData->rgba.Base(), pSrcImage, nBytes );
		DestroyVTFTexture( pTex );
		g_EmitTextures.Insert( pBaseTexture, pData );
		return pData;
	}

	EmitRGBATexture_t *pData = new EmitRGBATexture_t;
	pData->width = iWidth;
	pData->height = iHeight;
	pData->clampU = ( pTex->Flags() & TEXTUREFLAGS_CLAMPS ) != 0;
	pData->clampV = ( pTex->Flags() & TEXTUREFLAGS_CLAMPT ) != 0;
	pData->rgba.EnsureCapacity( nDstBytes );
	memcpy( pData->rgba.Base(), pDstImage, nDstBytes );
	delete[] pDstImage;
	DestroyVTFTexture( pTex );
	g_EmitTextures.Insert( pBaseTexture, pData );
	return pData;
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
	if ( v < 0 ) v += size;
	return v;
}

static bool SampleRGBA( const EmitRGBATexture_t *pTex, const texinfo_t *tx, const Vector &worldPos,
						unsigned char outRGBA[4] )
{
	if ( !pTex || !pTex->rgba.Base() || !tx )
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
	memcpy( outRGBA, &pTex->rgba[nOff], 4 );
	return true;
}

static bool ResolveCubemapPatchMaterial( const char *pName, char *pOut, int nOut )
{
	if ( !pName || Q_strncmp( pName, "maps/", 5 ) != 0 )
		return false;

	const char *pAfterMaps = pName + 5;
	const int nLevelLen = (int)Q_strlen( level_name );
	if ( nLevelLen > 0 && Q_strncmp( pAfterMaps, level_name, nLevelLen ) == 0 && pAfterMaps[nLevelLen] == '/' )
	{
		Q_strncpy( pOut, pAfterMaps + nLevelLen + 1, nOut );
	}
	else
	{
		const char *slash = strchr( pAfterMaps, '/' );
		if ( !slash )
			return false;
		Q_strncpy( pOut, slash + 1, nOut );
	}

	for ( int i = 0; i < 3; ++i )
	{
		char *underscore = Q_strrchr( pOut, '_' );
		if ( !underscore || !*underscore )
			return false;
		*underscore = '\0';
	}
	return pOut[0] != '\0';
}

static void NormalizeMatPath( char *p )
{
	Q_FixSlashes( p );
	Q_strlower( p );
	// Strip materials/ prefix and .vmt if present
	if ( !Q_strnicmp( p, "materials/", 10 ) )
		memmove( p, p + 10, strlen( p + 10 ) + 1 );
	char *ext = Q_strrchr( p, '.' );
	if ( ext && !Q_stricmp( ext, ".vmt" ) )
		*ext = '\0';
}

static float ParseFloatOrDefault( const char *p, float def )
{
	if ( !p || !p[0] )
		return def;
	return (float)atof( p );
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

// Merge patch{ include / insert } into a flat KeyValues of shader params.
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
	if ( !Q_stricmp( pVMT->GetName(), "patch" ) )
	{
		pInclude = pVMT->GetString( "include", NULL );
		pInsert = pVMT->FindKey( "insert" );
	}
	else
	{
		KeyValues *pPatch = pVMT->FindKey( "patch" );
		if ( pPatch )
		{
			pInclude = pPatch->GetString( "include", NULL );
			pInsert = pPatch->FindKey( "insert" );
		}
	}

	if ( pInclude && pInclude[0] )
	{
		char szInc[MAX_PATH];
		Q_strncpy( szInc, pInclude, sizeof( szInc ) );
		NormalizeMatPath( szInc );
		KeyValues *pBase = LoadMergedVmtKeys( szInc, nDepth + 1 );
		if ( pBase )
		{
			if ( pInsert )
			{
				for ( KeyValues *pSub = pInsert->GetFirstValue(); pSub; pSub = pSub->GetNextValue() )
					pBase->SetString( pSub->GetName(), pSub->GetString() );
			}
			pVMT->deleteThis();
			return pBase;
		}
	}

	return pVMT;
}

static EmitMaterial_t *ParseEmitMaterial( const char *pMaterialName )
{
	if ( !pMaterialName || !pMaterialName[0] )
		return NULL;

	char szResolved[MAX_PATH];
	szResolved[0] = 0;
	const char *pLookup = pMaterialName;
	if ( ResolveCubemapPatchMaterial( pMaterialName, szResolved, sizeof( szResolved ) ) )
		pLookup = szResolved;

	// Cache under resolved name (maps/ patches share one entry).
	int idx = g_EmitMaterials.Find( pLookup );
	if ( g_EmitMaterials.IsValidIndex( idx ) )
	{
		if ( pLookup != pMaterialName )
		{
			int idxOrig = g_EmitMaterials.Find( pMaterialName );
			if ( !g_EmitMaterials.IsValidIndex( idxOrig ) )
				g_EmitMaterials.Insert( pMaterialName, g_EmitMaterials[idx] );
		}
		return g_EmitMaterials[idx];
	}

	EmitMaterial_t *pMat = new EmitMaterial_t;
	memset( pMat, 0, sizeof( *pMat ) );
	pMat->strength = 0.0f;
	pMat->density = 1.0f;

	if ( !Q_strnicmp( pLookup, "tools/", 6 ) ||
		 !Q_strnicmp( pLookup, "skybox/", 7 ) )
	{
		g_EmitMaterials.Insert( pLookup, pMat );
		if ( pLookup != pMaterialName )
			g_EmitMaterials.Insert( pMaterialName, pMat );
		return pMat;
	}

	bool bEmitFlag = false;
	float strength = 0.0f;
	float density = 1.0f;
	char szMask[MAX_PATH] = {};
	char szMap[MAX_PATH] = {};
	char szBase[MAX_PATH] = {};

	// KeyValues only - Do NOT use GetMaterialVar("$basetexture"): texture vars
	// crash on GetStringValue(). Custom $vrad_emit* are plain strings/floats in the VMT.
	KeyValues *pVMT = LoadMergedVmtKeys( pLookup );
	if ( !pVMT && pLookup != pMaterialName )
		pVMT = LoadMergedVmtKeys( pMaterialName );
	if ( pVMT )
	{
		bEmitFlag = pVMT->GetInt( "$vrad_emit", 0 ) != 0 || ParseBoolish( pVMT->GetString( "$vrad_emit", NULL ) );
		strength = pVMT->GetFloat( "$vrad_emitstrength", 0.0f );
		density = pVMT->GetFloat( "$vrad_emitdensity", 1.0f );
		const char *m = pVMT->GetString( "$vrad_emitmask", NULL );
		const char *em = pVMT->GetString( "$vrad_emitmap", NULL );
		if ( !em || !em[0] )
			em = pVMT->GetString( "$vrad_emissivemap", NULL );
		const char *bt = pVMT->GetString( "$basetexture", NULL );
		if ( m && m[0] )
			Q_strncpy( szMask, m, sizeof( szMask ) );
		if ( em && em[0] )
			Q_strncpy( szMap, em, sizeof( szMap ) );
		if ( bt && bt[0] )
			Q_strncpy( szBase, bt, sizeof( szBase ) );
		pVMT->deleteThis();
	}

	if ( strength < 0.0f )
		strength = 0.0f;
	if ( density < 0.25f )
		density = 0.25f;
	if ( density > 16.0f )
		density = 16.0f;

	pMat->strength = strength;
	pMat->density = density;
	pMat->enabled = ( bEmitFlag || pMat->strength > 0.0f );
	if ( pMat->enabled && pMat->strength <= 0.0f )
		pMat->strength = 200.0f;

	if ( pMat->enabled )
	{
		const char *pMask = szMask[0] ? szMask : NULL;
		const char *pMap = szMap[0] ? szMap : NULL;
		const char *pBase = szBase[0] ? szBase : NULL;

		if ( pMap && pMap[0] )
		{
			Q_strncpy( pMat->colorName, pMap, sizeof( pMat->colorName ) );
			pMat->pColor = LoadEmitVTF( pMap );
		}
		else if ( pBase && pBase[0] )
		{
			Q_strncpy( pMat->colorName, pBase, sizeof( pMat->colorName ) );
			pMat->pColor = LoadEmitVTF( pBase );
		}
		else
		{
			Q_strncpy( pMat->colorName, pLookup, sizeof( pMat->colorName ) );
			pMat->pColor = LoadEmitVTF( pLookup );
		}

		if ( pMask && pMask[0] )
		{
			Q_strncpy( pMat->maskName, pMask, sizeof( pMat->maskName ) );
			pMat->pMask = LoadEmitVTF( pMask );
		}

		if ( !pMat->pColor )
		{
			Warning( "VRad emit: '%s' enabled but no color texture (emitmap/basetexture)\n", pLookup );
			pMat->enabled = false;
		}
		else
		{
			Msg( "VRad emit: %s  strength=%.1f  density=%.2f  color=%s  mask=%s\n",
				 pLookup, pMat->strength, pMat->density, pMat->colorName,
				 pMat->maskName[0] ? pMat->maskName : "(none)" );
		}
	}

	g_EmitMaterials.Insert( pLookup, pMat );
	if ( pLookup != pMaterialName )
		g_EmitMaterials.Insert( pMaterialName, pMat );
	return pMat;
}

static void BuildEmitCache( void )
{
	if ( g_bEmitCacheBuilt )
		return;
	g_bEmitCacheBuilt = true;

	g_EmitByTexdata.SetCount( numtexdata );
	int nEmit = 0;
	for ( int i = 0; i < numtexdata; i++ )
	{
		g_EmitByTexdata[i] = NULL;
		const char *pName = TexDataStringTable_GetString( dtexdata[i].nameStringTableID );
		if ( !pName || !pName[0] )
			continue;
		EmitMaterial_t *pMat = ParseEmitMaterial( pName );
		g_EmitByTexdata[i] = pMat;
		if ( pMat && pMat->enabled )
			++nEmit;
	}
	if ( nEmit > 0 )
		Msg( "VRad emit: %d emissive material(s) active\n", nEmit );
}

static EmitMaterial_t *GetEmitForTexdata( int texdataIndex )
{
	if ( texdataIndex < 0 || texdataIndex >= numtexdata )
		return NULL;
	BuildEmitCache();
	if ( texdataIndex >= g_EmitByTexdata.Count() )
		return NULL;
	return g_EmitByTexdata[texdataIndex];
}

static bool SampleEmitAtPos( int facenum, const Vector &worldPos, Vector &outEmit )
{
	outEmit.Init( 0, 0, 0 );
	if ( facenum < 0 || facenum >= numfaces )
		return false;

	dface_t *f = &g_pFaces[facenum];
	texinfo_t *tx = &texinfo[f->texinfo];
	EmitMaterial_t *pMat = GetEmitForTexdata( tx->texdata );
	if ( !pMat || !pMat->enabled || !pMat->pColor )
		return false;

	unsigned char rgba[4];
	if ( !SampleRGBA( pMat->pColor, tx, worldPos, rgba ) )
		return false;

	float mask = 1.0f;
	if ( pMat->pMask )
	{
		unsigned char m[4];
		if ( SampleRGBA( pMat->pMask, tx, worldPos, m ) )
		{
			// greyscale: average RGB (or just R)
			mask = ( m[0] + m[1] + m[2] ) * ( 1.0f / ( 3.0f * 255.0f ) );
		}
		else
		{
			mask = 0.0f;
		}
	}

	if ( mask <= 1e-6f )
		return false;

	// Match lights.rad / LightForString: sRGB -> linear * 255, then * (strength/255) * lightscale
	float r = powf( rgba[0] / 255.0f, 2.2f ) * 255.0f;
	float g = powf( rgba[1] / 255.0f, 2.2f ) * 255.0f;
	float b = powf( rgba[2] / 255.0f, 2.2f ) * 255.0f;
	const float scale = ( pMat->strength / 255.0f ) * mask * lightscale;
	outEmit.x = r * scale;
	outEmit.y = g * scale;
	outEmit.z = b * scale;
	return VectorAvg( outEmit ) > 1e-6f;
}

bool VRadEmit_MaterialEmits( const char *pMaterialName )
{
	if ( !pMaterialName || !pMaterialName[0] )
		return false;
	EmitMaterial_t *pMat = ParseEmitMaterial( pMaterialName );
	return pMat && pMat->enabled;
}

bool VRadEmit_FaceEmits( int facenum )
{
	if ( facenum < 0 || facenum >= numfaces )
		return false;
	dface_t *f = &g_pFaces[facenum];
	if ( f->texinfo < 0 )
		return false;
	EmitMaterial_t *pMat = GetEmitForTexdata( texinfo[f->texinfo].texdata );
	return pMat && pMat->enabled;
}

float VRadEmit_GetDensity( const char *pMaterialName )
{
	if ( !pMaterialName || !pMaterialName[0] )
		return 1.0f;
	EmitMaterial_t *pMat = ParseEmitMaterial( pMaterialName );
	if ( !pMat || !pMat->enabled )
		return 1.0f;
	return pMat->density;
}

bool VRadEmit_SampleAtPos( int facenum, const Vector &worldPos, Vector &outEmit )
{
	return SampleEmitAtPos( facenum, worldPos, outEmit );
}

int VRadEmit_CreateDirectLights( void )
{
	BuildEmitCache();

	// One soft area light per face (sum of leaf patches). Keeps pathtrace NEE cheap
	// while preserving total emitted power vs per-patch lights.
	struct FaceEmitAccum_t
	{
		float area;
		Vector origin;		// area-weighted
		Vector normal;		// area-weighted
		Vector intensity;	// pre-scaled sum (before DIRECT_SCALE)
		bool any;
	};

	CUtlVector<FaceEmitAccum_t> accum;
	accum.SetCount( numfaces );
	memset( accum.Base(), 0, accum.Count() * sizeof( FaceEmitAccum_t ) );

	const float kDirectScale = 100.0f * 100.0f;
	unsigned int nPatches = g_Patches.Size();
	for ( unsigned int i = 0; i < nPatches; i++ )
	{
		CPatch *patch = &g_Patches[i];
		if ( patch->child1 != g_Patches.InvalidIndex() )
			continue;
		if ( patch->sky )
			continue;
		if ( patch->area < 1e-6f )
			continue;

		const int face = patch->faceNumber;
		if ( face < 0 || face >= numfaces )
			continue;

		dface_t *f = &g_pFaces[face];
		EmitMaterial_t *pMat = GetEmitForTexdata( texinfo[f->texinfo].texdata );
		if ( !pMat || !pMat->enabled )
			continue;

		Vector emit;
		if ( !SampleEmitAtPos( face, patch->origin, emit ) )
			continue;
		if ( VectorAvg( emit ) < dlight_threshold )
			continue;

		FaceEmitAccum_t &a = accum[face];
		a.any = true;
		a.area += patch->area;
		VectorMA( a.origin, patch->area, patch->origin, a.origin );
		VectorMA( a.normal, patch->area, patch->normal, a.normal );

		// Brightness scale for textured $vrad_emit*:
		//   /basearea (stock lights.rad) -> invisible under pathtrace area falloff
		//   no divide -> blown out
		//   /sqrt(basearea) -> still too hot
		// Fixed ref sheet (128^2) keeps $vrad_emitstrength in a usable range regardless
		// of VTF resolution.
		Vector patchI;
		const float kRefArea = 128.0f * 128.0f;
		VectorScale( emit,
					 lightscale * patch->area * patch->scale[0] * patch->scale[1] / kRefArea,
					 patchI );
		a.intensity += patchI;

		texinfo[f->texinfo].flags |= SURF_LIGHT;
	}

	int nLights = 0;
	float maxAvg = 0.0f;
	for ( int face = 0; face < numfaces; ++face )
	{
		FaceEmitAccum_t &a = accum[face];
		if ( !a.any || a.area < 1e-6f )
			continue;
		if ( VectorAvg( a.intensity ) < 1e-8f )
			continue;

		Vector org = a.origin * ( 1.0f / a.area );
		Vector nrm = a.normal;
		if ( VectorNormalize( nrm ) < 1e-6f )
			continue;
		// Push off the face so pathtrace shadow rays don't self-intersect.
		VectorMA( org, 1.0f, nrm, org );

		directlight_t *dl = AllocDLight( org, true );
		dl->light.type = emit_surface;
		dl->facenum = face;
		VectorCopy( nrm, dl->light.normal );
		// Cap R^2: full face-area R made cos/(r^2+R^2) kill stock-scale lights on large faces.
		// Small R keeps near-contact soft without crushing room lighting.
		dl->m_flAreaRadius2 = max( 4.0f, min( a.area / (float)M_PI, 48.0f ) );
		dl->m_flEndFadeDistance = -1.0f; // calloc skips ctor; uncapped
		dl->m_flCapDist = 1.0e22f;
		VectorScale( a.intensity, kDirectScale, dl->light.intensity );
		maxAvg = max( maxAvg, VectorAvg( dl->light.intensity ) );
		++nLights;
	}

	if ( nLights > 0 )
		Msg( "VRad emit: created %d area-softened surface lights (1/face, maxAvgI=%.1f)\n",
			 nLights, maxAvg );
	return nLights;
}

void VRadEmit_AddSelfIllumToFaceLights( void )
{
	BuildEmitCache();
	int nFaces = 0;
	for ( int facenum = 0; facenum < numfaces; facenum++ )
	{
		dface_t *f = &g_pFaces[facenum];
		if ( f->texinfo < 0 )
			continue;
		EmitMaterial_t *pMat = GetEmitForTexdata( texinfo[f->texinfo].texdata );
		if ( !pMat || !pMat->enabled )
			continue;

		facelight_t *fl = &facelight[facenum];
		if ( fl->numsamples <= 0 || !fl->sample || !fl->light[0][0] )
			continue;

		const bool needsBump = ( texinfo[f->texinfo].flags & SURF_BUMPLIGHT ) != 0;
		for ( int i = 0; i < fl->numsamples; i++ )
		{
			Vector emit;
			if ( !SampleEmitAtPos( facenum, fl->sample[i].pos, emit ) )
				continue;
			// Self-illum: surface appears to glow (in addition to cast direct lights)
			fl->light[0][0][i].m_vecLighting += emit;
			if ( needsBump )
			{
				for ( int b = 1; b < NUM_BUMP_VECTS + 1; b++ )
				{
					if ( fl->light[0][b] )
						fl->light[0][b][i].m_vecLighting += emit;
				}
			}
		}
		++nFaces;
	}
	if ( nFaces > 0 )
		Msg( "VRad emit: self-illum added on %d faces\n", nFaces );
}
