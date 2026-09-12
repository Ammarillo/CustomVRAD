//========= Copyright PathRAD contributors. ============//
// Load ProjectedTexture VTFs and sample them on the CPU.
//=============================================================================//

#include "light_projection.h"
#include "vrad.h"
#include "vtf/vtf.h"
#include "tier1/utldict.h"
#include "tier1/utlbuffer.h"
#include "bitmap/imageformat.h"
#include "tier1/strtools.h"
#include "mathlib/mathlib.h"
#include <cmath>

static const int kProjMaxCookieEdge = 1024;
static const int kProjMaxCubeFace = 512;

struct ProjTex
{
	char		path[MAX_PATH];
	ProjKind_t	kind;
	int			width;
	int			height;
	int			gpuLayer;	// planar layer index, or cube index (faces = layer*6..+5)
	CUtlMemory<unsigned char> rgba; // 2D: W*H*4; cube: 6*W*H*4 (Source face order)
};

static CUtlDict<ProjTex *, int> g_ProjByPath;
static CUtlVector<ProjTex *> g_ProjList;

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

static void PatchVtfMinorVersion( CUtlBuffer &buf )
{
	if ( buf.TellPut() < 12 )
		return;
	unsigned char *pRaw = (unsigned char *)buf.Base();
	if ( pRaw[0] == 'V' && pRaw[1] == 'T' && pRaw[2] == 'F' && pRaw[3] == 0 )
	{
		unsigned int major = *(unsigned int *)( pRaw + 4 );
		unsigned int minor = *(unsigned int *)( pRaw + 8 );
		// SDK loader tops out at 7.4; 7.5 is usually layout-compatible.
		if ( major == 7 && minor > 4 )
			*(unsigned int *)( pRaw + 8 ) = 4;
	}
}

// VTF 7.3+ may include a CRC resource that some SDK builds reject. Keep only
// IMAGE (and optional low-res) so Unserialize can succeed. Dictionary starts at
// 0x50 with 8 bytes of compiler padding after numResources @ 0x44 (see vtf.h).
static void PatchVtfDropAuxResources( CUtlBuffer &buf )
{
	if ( buf.TellPut() < 96 )
		return;
	unsigned char *pRaw = (unsigned char *)buf.Base();
	if ( !( pRaw[0] == 'V' && pRaw[1] == 'T' && pRaw[2] == 'F' && pRaw[3] == 0 ) )
		return;
	const unsigned int headerSize = *(unsigned int *)( pRaw + 12 );
	if ( headerSize < 80 || headerSize > (unsigned)buf.TellPut() )
		return;

	const int kDictBase = 0x50;
	if ( (int)headerSize < kDictBase + 8 )
		return;

	const unsigned int nListed = *(unsigned int *)( pRaw + 0x44 );
	if ( nListed == 0 || nListed > 32 )
		return;

	int imageSlot = -1;
	int lowResSlot = -1;
	for ( unsigned int i = 0; i < nListed; ++i )
	{
		const int off = kDictBase + (int)i * 8;
		if ( off + 8 > (int)headerSize )
			break;
		const unsigned int type = *(unsigned int *)( pRaw + off ) & 0x00FFFFFF;
		if ( type == 0x00000030 ) // VTF_LEGACY_RSRC_IMAGE
			imageSlot = (int)i;
		else if ( type == 0x00000001 ) // VTF_LEGACY_RSRC_LOW_RES_IMAGE
			lowResSlot = (int)i;
	}
	if ( imageSlot < 0 )
		return;

	// Compact kept entries to the front of the dictionary.
	unsigned char keep[16];
	int nKeep = 0;
	if ( lowResSlot >= 0 )
	{
		memcpy( keep, pRaw + kDictBase + lowResSlot * 8, 8 );
		nKeep = 1;
	}
	memcpy( keep + nKeep * 8, pRaw + kDictBase + imageSlot * 8, 8 );
	++nKeep;

	*(unsigned int *)( pRaw + 0x44 ) = (unsigned int)nKeep;
	memset( pRaw + kDictBase, 0, (size_t)headerSize - (size_t)kDictBase );
	memcpy( pRaw + kDictBase, keep, (size_t)nKeep * 8u );
	*(unsigned int *)( pRaw + 12 ) = (unsigned int)( kDictBase + nKeep * 8 );
}

// NOMIP + multi-mip cubemaps confuse some SDK loaders (size / face layout).
static void PatchVtfClearNomipIfMipped( CUtlBuffer &buf )
{
	if ( buf.TellPut() < 57 )
		return;
	unsigned char *pRaw = (unsigned char *)buf.Base();
	if ( !( pRaw[0] == 'V' && pRaw[1] == 'T' && pRaw[2] == 'F' && pRaw[3] == 0 ) )
		return;
	const unsigned char nMips = pRaw[56];
	if ( nMips <= 1 )
		return;
	unsigned int flags = *(unsigned int *)( pRaw + 20 );
	if ( flags & TEXTUREFLAGS_NOMIP )
		*(unsigned int *)( pRaw + 20 ) = flags & ~TEXTUREFLAGS_NOMIP;
}

// Only strip ENVMAP when the file cannot possibly hold 6 faces (true 2D matcap /
// equirect wrongly flagged). Real DXT cubemaps must keep the flag.
static bool PatchFakeCubemapEnvmapFlag( CUtlBuffer &buf )
{
	if ( buf.TellPut() < 56 )
		return false;
	unsigned char *pRaw = (unsigned char *)buf.Base();
	if ( !( pRaw[0] == 'V' && pRaw[1] == 'T' && pRaw[2] == 'F' && pRaw[3] == 0 ) )
		return false;

	unsigned int flags = *(unsigned int *)( pRaw + 20 );
	if ( ( flags & TEXTUREFLAGS_ENVMAP ) == 0 )
		return false;

	const unsigned int headerSize = *(unsigned int *)( pRaw + 12 );
	const unsigned short w = *(unsigned short *)( pRaw + 16 );
	const unsigned short h = *(unsigned short *)( pRaw + 18 );
	// bumpScale @48, format @52 in 7.1+ packed header (matches Valve layout)
	const ImageFormat format = (ImageFormat)( *(int *)( pRaw + 52 ) );

	const int faceBytes = ImageLoader::GetMemRequired( (int)w, (int)h, 1, format, true );
	if ( faceBytes <= 0 )
		return false;

	const size_t need = (size_t)headerSize + (size_t)faceBytes * 6u;
	// Allow some slack for alignment / low-res thumbnail.
	if ( (size_t)buf.TellPut() + 256 < need )
	{
		*(unsigned int *)( pRaw + 20 ) = flags & ~TEXTUREFLAGS_ENVMAP;
		return true;
	}
	return false;
}

static void ChooseMip( IVTFTexture *pTex, int maxEdge, int &mipOut, int &wOut, int &hOut )
{
	mipOut = 0;
	wOut = pTex->Width();
	hOut = pTex->Height();
	const int nMips = pTex->MipCount();
	while ( mipOut + 1 < nMips && ( wOut > maxEdge || hOut > maxEdge ) )
	{
		++mipOut;
		wOut = max( 1, wOut >> 1 );
		hOut = max( 1, hOut >> 1 );
	}
}

static bool CopyFaceRGBA( IVTFTexture *pTex, int face, int mip, int w, int h, unsigned char *dst )
{
	unsigned char *pSrc = pTex->ImageData( 0, face, mip );
	if ( !pSrc )
		return false;
	ImageFormat srcFormat = pTex->Format();
	if ( srcFormat == IMAGE_FORMAT_RGBA8888 )
	{
		memcpy( dst, pSrc, (size_t)w * (size_t)h * 4u );
		return true;
	}
	return ImageLoader::ConvertImageFormat( pSrc, srcFormat, dst, IMAGE_FORMAT_RGBA8888, w, h, 0, 0 );
}

// Manual 6-face decode when SDK Unserialize won't yield a cubemap (7.5 + CRC + NOMIP, etc.).
static ProjTex *LoadCubeRawFromVtfBuffer( const char *pCleanPath, const char *szPath, CUtlBuffer &buf )
{
	if ( buf.TellPut() < 96 )
		return NULL;
	unsigned char *pRaw = (unsigned char *)buf.Base();
	if ( !( pRaw[0] == 'V' && pRaw[1] == 'T' && pRaw[2] == 'F' && pRaw[3] == 0 ) )
		return NULL;

	const unsigned int flags = *(unsigned int *)( pRaw + 20 );
	if ( ( flags & TEXTUREFLAGS_ENVMAP ) == 0 )
		return NULL;

	const unsigned int headerSize = *(unsigned int *)( pRaw + 12 );
	const int w0 = (int)*(unsigned short *)( pRaw + 16 );
	const int h0 = (int)*(unsigned short *)( pRaw + 18 );
	const ImageFormat format = (ImageFormat)( *(int *)( pRaw + 52 ) );
	const int nMips = (int)pRaw[56];
	const int nFrames = (int)*(unsigned short *)( pRaw + 24 );
	if ( w0 <= 0 || h0 <= 0 || nMips <= 0 || nMips > 20 || nFrames <= 0 )
		return NULL;

	int imageOff = (int)headerSize;
	const unsigned int minor = *(unsigned int *)( pRaw + 8 );
	if ( minor >= 3 && headerSize >= 0x58 )
	{
		const unsigned int nRes = *(unsigned int *)( pRaw + 0x44 );
		for ( unsigned int i = 0; i < nRes && i < 32; ++i )
		{
			const int e = 0x50 + (int)i * 8;
			if ( e + 8 > (int)headerSize )
				break;
			const unsigned int type = *(unsigned int *)( pRaw + e ) & 0x00FFFFFF;
			if ( type == 0x00000030 )
			{
				imageOff = (int)*(unsigned int *)( pRaw + e + 4 );
				break;
			}
		}
	}
	if ( imageOff < 0 || imageOff >= buf.TellPut() )
		return NULL;

	const int faceFull = ImageLoader::GetMemRequired( w0, h0, 1, format, true );
	if ( faceFull <= 0 )
		return NULL;
	if ( (size_t)buf.TellPut() < (size_t)imageOff + (size_t)faceFull * 6u )
		return NULL;

	int mip = 0;
	int w = w0, h = h0;
	while ( mip + 1 < nMips && ( w > kProjMaxCubeFace || h > kProjMaxCubeFace ) )
	{
		++mip;
		w = max( 1, w >> 1 );
		h = max( 1, h >> 1 );
	}

	const int faceBytes = w * h * 4;
	ProjTex *p = new ProjTex;
	Q_strncpy( p->path, pCleanPath, sizeof( p->path ) );
	p->kind = PROJ_CUBE;
	p->width = w;
	p->height = h;
	p->gpuLayer = -1;
	p->rgba.EnsureCapacity( faceBytes * 6 );
	unsigned char *base = p->rgba.Base();
	memset( base, 0, (size_t)faceBytes * 6u );

	const bool bResourceOrder = ( minor >= 3 ); // 7.3+: mip small->large
	const unsigned char *img = pRaw + imageOff;
	const int imgBytes = buf.TellPut() - imageOff;
	int cursor = 0;
	bool ok = true;

	auto stepMip = [&]( int mw, int mh ) -> bool
	{
		const int faceSize = ImageLoader::GetMemRequired( mw, mh, 1, format, false );
		if ( faceSize <= 0 )
			return false;
		const int chunk = faceSize * 6 * nFrames;
		if ( cursor + chunk > imgBytes )
			return false;
		cursor += chunk;
		return true;
	};

	if ( bResourceOrder )
	{
		// Skip smaller mips (disk stores 1x1 first).
		for ( int disk = nMips - 1; disk > mip; --disk )
		{
			const int mw = max( 1, w0 >> disk );
			const int mh = max( 1, h0 >> disk );
			if ( !stepMip( mw, mh ) )
			{
				ok = false;
				break;
			}
		}
		if ( ok )
		{
			const int faceSize = ImageLoader::GetMemRequired( w, h, 1, format, false );
			if ( faceSize <= 0 || cursor + faceSize * 6 > imgBytes )
				ok = false;
			else
			{
				for ( int f = 0; f < 6; ++f )
				{
					const unsigned char *src = img + cursor + f * faceSize;
					if ( format == IMAGE_FORMAT_RGBA8888 && faceSize >= faceBytes )
						memcpy( base + f * faceBytes, src, (size_t)faceBytes );
					else if ( !ImageLoader::ConvertImageFormat( src, format, base + f * faceBytes,
																 IMAGE_FORMAT_RGBA8888, w, h, 0, 0 ) )
					{
						ok = false;
						break;
					}
				}
			}
		}
	}
	else
	{
		// 7.2 and earlier: largest mip first.
		for ( int disk = 0; disk < mip; ++disk )
		{
			const int mw = max( 1, w0 >> disk );
			const int mh = max( 1, h0 >> disk );
			if ( !stepMip( mw, mh ) )
			{
				ok = false;
				break;
			}
		}
		if ( ok )
		{
			const int faceSize = ImageLoader::GetMemRequired( w, h, 1, format, false );
			if ( faceSize <= 0 || cursor + faceSize * 6 > imgBytes )
				ok = false;
			else
			{
				for ( int f = 0; f < 6; ++f )
				{
					const unsigned char *src = img + cursor + f * faceSize;
					if ( format == IMAGE_FORMAT_RGBA8888 && faceSize >= faceBytes )
						memcpy( base + f * faceBytes, src, (size_t)faceBytes );
					else if ( !ImageLoader::ConvertImageFormat( src, format, base + f * faceBytes,
																 IMAGE_FORMAT_RGBA8888, w, h, 0, 0 ) )
					{
						ok = false;
						break;
					}
				}
			}
		}
	}

	if ( !ok )
	{
		delete p;
		return NULL;
	}

	Msg( "ProjectedTexture: loaded cubemap '%s' (%dx%d, 6 faces, raw VTF decode)\n", szPath, w, h );
	return p;
}

static ProjTex *LoadProjVTF( const char *pCleanPath )
{
	char szPath[MAX_PATH];
	Q_snprintf( szPath, sizeof( szPath ), "materials/%s.vtf", pCleanPath );
	Q_FixSlashes( szPath );

	auto Reload = [&]( CUtlBuffer &out ) -> bool
	{
		out.Clear();
		return LoadFileToBuffer( szPath, out );
	};

	CUtlBuffer buf;
	if ( !Reload( buf ) )
	{
		Warning( "ProjectedTexture: could not open '%s'\n", szPath );
		return NULL;
	}

	bool bHadEnvFlag = false;
	unsigned short peekW = 0, peekH = 0;
	int peekFmt = -1;
	if ( buf.TellPut() >= 56 )
	{
		unsigned char *pRaw = (unsigned char *)buf.Base();
		bHadEnvFlag = ( ( *(unsigned int *)( pRaw + 20 ) & TEXTUREFLAGS_ENVMAP ) != 0 );
		peekW = *(unsigned short *)( pRaw + 16 );
		peekH = *(unsigned short *)( pRaw + 18 );
		peekFmt = *(int *)( pRaw + 52 );
	}

	auto PrepareCubeFriendly = []( CUtlBuffer &b )
	{
		PatchVtfMinorVersion( b );
		PatchVtfDropAuxResources( b );
		PatchVtfClearNomipIfMipped( b );
	};

	IVTFTexture *pTex = CreateVTFTexture();
	bool bLoaded = false;

	// Pass 1: keep ENVMAP - real DXT cubemaps (e.g. Matcap7: 256 DXT1 x 6 x 9 mips).
	{
		PrepareCubeFriendly( buf );
		buf.SeekGet( CUtlBuffer::SEEK_HEAD, 0 );
		bLoaded = pTex->Unserialize( buf );
	}

	// Pass 2: size-heuristic strip ENVMAP only if file cannot hold 6 faces.
	if ( !bLoaded && bHadEnvFlag )
	{
		DestroyVTFTexture( pTex );
		pTex = CreateVTFTexture();
		if ( !Reload( buf ) )
		{
			DestroyVTFTexture( pTex );
			return NULL;
		}
		PrepareCubeFriendly( buf );
		PatchFakeCubemapEnvmapFlag( buf );
		buf.SeekGet( CUtlBuffer::SEEK_HEAD, 0 );
		bLoaded = pTex->Unserialize( buf );
	}

	// Pass 3: force strip ENVMAP (2D spheremap / matcap wrongly flagged).
	if ( !bLoaded && bHadEnvFlag )
	{
		DestroyVTFTexture( pTex );
		pTex = CreateVTFTexture();
		if ( !Reload( buf ) )
		{
			DestroyVTFTexture( pTex );
			return NULL;
		}
		PatchVtfMinorVersion( buf );
		if ( buf.TellPut() >= 24 )
		{
			unsigned char *pRaw = (unsigned char *)buf.Base();
			*(unsigned int *)( pRaw + 20 ) &= ~TEXTUREFLAGS_ENVMAP;
		}
		buf.SeekGet( CUtlBuffer::SEEK_HEAD, 0 );
		bLoaded = pTex->Unserialize( buf );
	}

	if ( !bLoaded )
	{
		DestroyVTFTexture( pTex );
		pTex = NULL;
		// Raw path for real cubemaps the SDK refuses (common with VTF 7.5 + CRC).
		if ( bHadEnvFlag )
		{
			if ( !Reload( buf ) )
				return NULL;
			if ( ProjTex *raw = LoadCubeRawFromVtfBuffer( pCleanPath, szPath, buf ) )
				return raw;
		}
		Warning( "ProjectedTexture: failed to parse '%s' (file %d bytes, %dx%d, fmt=%d, ENVMAP=%d)\n",
				 szPath, buf.TellPut(), peekW, peekH, peekFmt, bHadEnvFlag ? 1 : 0 );
		return NULL;
	}

	const int nFaces = pTex->FaceCount();
	bool bRealCube = ( pTex->IsCubeMap() || bHadEnvFlag ) && nFaces >= 6;
	if ( bHadEnvFlag && !bRealCube )
	{
		// SDK accepted the file but as 2D - try raw 6-face decode before spheremap.
		DestroyVTFTexture( pTex );
		pTex = NULL;
		if ( !Reload( buf ) )
			return NULL;
		if ( ProjTex *raw = LoadCubeRawFromVtfBuffer( pCleanPath, szPath, buf ) )
			return raw;

		// Fall through: reload via Unserialize as 2D omni.
		pTex = CreateVTFTexture();
		PrepareCubeFriendly( buf );
		PatchFakeCubemapEnvmapFlag( buf );
		if ( buf.TellPut() >= 24 )
			*(unsigned int *)( (unsigned char *)buf.Base() + 20 ) &= ~TEXTUREFLAGS_ENVMAP;
		buf.SeekGet( CUtlBuffer::SEEK_HEAD, 0 );
		if ( !pTex->Unserialize( buf ) )
		{
			DestroyVTFTexture( pTex );
			Warning( "ProjectedTexture: '%s' ENVMAP FaceCount=%d and raw cube decode failed\n",
					 szPath, nFaces );
			return NULL;
		}
		Msg( "ProjectedTexture: '%s' has ENVMAP but FaceCount=%d (need 6) - 2D omni envmap. file=%d bytes %dx%d fmt=%d\n",
			 szPath, nFaces, buf.TellPut(), peekW, peekH, peekFmt );
		bRealCube = false;
	}

	const bool bSphereEnv = !bRealCube && bHadEnvFlag;
	const int maxEdge = bRealCube ? kProjMaxCubeFace : kProjMaxCookieEdge;
	int mip, w, h;
	ChooseMip( pTex, maxEdge, mip, w, h );
	if ( w <= 0 || h <= 0 || w > 8192 || h > 8192 )
	{
		DestroyVTFTexture( pTex );
		return NULL;
	}

	if ( pTex->Format() != IMAGE_FORMAT_RGBA8888 )
		pTex->ConvertImageFormat( IMAGE_FORMAT_RGBA8888, false );

	ChooseMip( pTex, maxEdge, mip, w, h );

	ProjTex *p = new ProjTex;
	Q_strncpy( p->path, pCleanPath, sizeof( p->path ) );
	if ( bRealCube )
		p->kind = PROJ_CUBE;
	else if ( bSphereEnv )
		p->kind = PROJ_SPHERE;
	else
		p->kind = PROJ_2D;
	p->width = w;
	p->height = h;
	p->gpuLayer = -1;

	if ( p->kind == PROJ_CUBE )
	{
		const int faceBytes = w * h * 4;
		p->rgba.EnsureCapacity( faceBytes * 6 );
		unsigned char *base = p->rgba.Base();
		memset( base, 0, (size_t)faceBytes * 6u );
		for ( int f = 0; f < 6; ++f )
		{
			if ( !CopyFaceRGBA( pTex, f, mip, w, h, base + f * faceBytes ) )
			{
				Warning( "ProjectedTexture: cubemap face %d/%d failed for '%s'\n", f, nFaces, szPath );
				delete p;
				DestroyVTFTexture( pTex );
				return NULL;
			}
		}
		Msg( "ProjectedTexture: loaded cubemap '%s' (%dx%d, %d faces)\n", szPath, w, h, nFaces );
	}
	else
	{
		const int bytes = w * h * 4;
		p->rgba.EnsureCapacity( bytes );
		if ( !CopyFaceRGBA( pTex, 0, mip, w, h, p->rgba.Base() ) )
		{
			Warning( "ProjectedTexture: convert failed for '%s'\n", szPath );
			delete p;
			DestroyVTFTexture( pTex );
			return NULL;
		}
		if ( p->kind == PROJ_SPHERE )
		{
			const char *mapStr = ( w >= ( h * 3 ) / 2 ) ? "equirect" : "spheremap/matcap";
			Msg( "ProjectedTexture: loaded 2D envmap '%s' (%dx%d, %s, omni; FaceCount=%d)\n",
				 szPath, w, h, mapStr, nFaces );
		}
		else
		{
			Msg( "ProjectedTexture: loaded 2D planar '%s' (%dx%d)\n", szPath, w, h );
		}
	}

	DestroyVTFTexture( pTex );
	return p;
}

const ProjTex *Proj_FindOrLoad( const char *pPath )
{
	if ( !pPath || !pPath[0] )
		return NULL;

	char szClean[MAX_PATH];
	Q_strncpy( szClean, pPath, sizeof( szClean ) );
	Q_FixSlashes( szClean );
	if ( !Q_strnicmp( szClean, "materials/", 10 ) )
		Q_memmove( szClean, szClean + 10, strlen( szClean + 10 ) + 1 );
	char *ext = Q_strrchr( szClean, '.' );
	if ( ext && ( !Q_stricmp( ext, ".vtf" ) || !Q_stricmp( ext, ".vmt" ) ) )
		*ext = 0;

	int idx = g_ProjByPath.Find( szClean );
	if ( g_ProjByPath.IsValidIndex( idx ) )
		return g_ProjByPath[idx];

	ProjTex *p = LoadProjVTF( szClean );
	g_ProjByPath.Insert( szClean, p ); // may insert NULL to avoid retry spam
	if ( p )
		g_ProjList.AddToTail( p );
	return p;
}

ProjKind_t Proj_Kind( const ProjTex *p )
{
	return p ? p->kind : PROJ_NONE;
}

int Proj_GpuLayer( const ProjTex *p )
{
	return p ? p->gpuLayer : -1;
}

bool Proj_IsEquirect( const ProjTex *p )
{
	return p && p->kind == PROJ_SPHERE && p->width >= ( p->height * 3 ) / 2;
}

static Vector SampleRGBABilinear( const unsigned char *rgba, int w, int h, float u, float v, bool wrapU )
{
	if ( !rgba || w <= 0 || h <= 0 )
		return Vector( 0, 0, 0 );
	if ( wrapU )
	{
		u = u - floorf( u );
	}
	else if ( u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f )
	{
		return Vector( 0, 0, 0 ); // hard planar edge
	}
	v = max( 0.0f, min( 1.0f, v ) );

	float x = u * (float)( w - 1 );
	float y = v * (float)( h - 1 );
	int x0 = (int)floorf( x );
	int y0 = (int)floorf( y );
	int x1 = x0 + 1;
	int y1 = min( y0 + 1, h - 1 );
	if ( wrapU )
	{
		x0 = ( ( x0 % w ) + w ) % w;
		x1 = ( x1 % w + w ) % w;
	}
	else
	{
		x0 = max( 0, min( x0, w - 1 ) );
		x1 = max( 0, min( x1, w - 1 ) );
	}
	float tx = x - floorf( x );
	float ty = y - (float)y0;

	auto fetch = [&]( int ix, int iy ) -> Vector
	{
		const unsigned char *p = rgba + ( ( iy * w + ix ) * 4 );
		return Vector( p[0] / 255.0f, p[1] / 255.0f, p[2] / 255.0f );
	};
	Vector c00 = fetch( x0, y0 );
	Vector c10 = fetch( x1, y0 );
	Vector c01 = fetch( x0, y1 );
	Vector c11 = fetch( x1, y1 );
	Vector c0 = c00 + ( c10 - c00 ) * tx;
	Vector c1 = c01 + ( c11 - c01 ) * tx;
	return c0 + ( c1 - c0 ) * ty;
}

static Vector SampleCubeFace( const ProjTex *p, int face, float u, float v )
{
	// Cube faces clamp at edges (no black outside).
	u = max( 0.0f, min( 1.0f, u ) );
	v = max( 0.0f, min( 1.0f, v ) );
	const int w = p->width;
	const int h = p->height;
	const unsigned char *base = p->rgba.Base() + face * w * h * 4;
	// Temporarily allow in-range by sampling with open edge: use clamp path
	float x = u * (float)( w - 1 );
	float y = v * (float)( h - 1 );
	int x0 = (int)floorf( x );
	int y0 = (int)floorf( y );
	int x1 = min( x0 + 1, w - 1 );
	int y1 = min( y0 + 1, h - 1 );
	float tx = x - (float)x0;
	float ty = y - (float)y0;
	auto fetch = [&]( int ix, int iy ) -> Vector
	{
		const unsigned char *px = base + ( ( iy * w + ix ) * 4 );
		return Vector( px[0] / 255.0f, px[1] / 255.0f, px[2] / 255.0f );
	};
	Vector c00 = fetch( x0, y0 );
	Vector c10 = fetch( x1, y0 );
	Vector c01 = fetch( x0, y1 );
	Vector c11 = fetch( x1, y1 );
	Vector c0 = c00 + ( c10 - c00 ) * tx;
	Vector c1 = c01 + ( c11 - c01 ) * tx;
	return c0 + ( c1 - c0 ) * ty;
}

// Source VTF cubemap faces match D3D cube sampling (CUBEMAP_FACE_*:
// +X -X +Y -Y +Z -Z). sc/tc from the D3D cubemap face table.
static Vector SampleCubeDir( const ProjTex *p, const Vector &dirIn )
{
	Vector d = dirIn;
	float len = VectorNormalize( d );
	if ( len < 1e-8f )
		return Vector( 1, 1, 1 );

	const float ax = fabsf( d.x ), ay = fabsf( d.y ), az = fabsf( d.z );
	int face;
	float sc, tc, ma;
	if ( ax >= ay && ax >= az )
	{
		ma = ax;
		if ( d.x > 0 ) { face = 0; sc = -d.z; tc = -d.y; } // +X RIGHT
		else           { face = 1; sc =  d.z; tc = -d.y; } // -X LEFT
	}
	else if ( ay >= ax && ay >= az )
	{
		ma = ay;
		if ( d.y > 0 ) { face = 2; sc =  d.x; tc =  d.z; } // +Y BACK
		else           { face = 3; sc =  d.x; tc = -d.z; } // -Y FRONT
	}
	else
	{
		ma = az;
		if ( d.z > 0 ) { face = 4; sc =  d.x; tc = -d.y; } // +Z UP
		else           { face = 5; sc = -d.x; tc = -d.y; } // -Z DOWN
	}
	float u = 0.5f * ( sc / ma + 1.0f );
	float v = 0.5f * ( tc / ma + 1.0f );
	return SampleCubeFace( p, face, u, v );
}

// Omnidirectional 2D env: equirect if wide, else OpenGL spheremap (matcaps).
static Vector SampleSphereDir( const ProjTex *p, const Vector &dirIn )
{
	Vector d = dirIn;
	float len = VectorNormalize( d );
	if ( len < 1e-8f )
		return Vector( 1, 1, 1 );

	float u, v;
	const bool bEquirect = ( p->width >= ( p->height * 3 ) / 2 );
	if ( bEquirect )
	{
		u = atan2f( d.y, d.x ) * ( 0.5f / (float)M_PI ) + 0.5f;
		v = acosf( max( -1.0f, min( 1.0f, d.z ) ) ) * ( 1.0f / (float)M_PI );
		return SampleRGBABilinear( p->rgba.Base(), p->width, p->height, u, v, true );
	}

	// Spheremap / matcap (OpenGL-style)
	float m = 2.0f * sqrtf( d.x * d.x + d.y * d.y + ( d.z + 1.0f ) * ( d.z + 1.0f ) );
	if ( m < 1e-6f )
	{
		u = 0.5f;
		v = 0.5f;
	}
	else
	{
		u = d.x / m + 0.5f;
		v = d.y / m + 0.5f;
	}
	u = max( 0.0f, min( 1.0f, u ) );
	v = max( 0.0f, min( 1.0f, v ) );
	return SampleRGBABilinear( p->rgba.Base(), p->width, p->height, u, v, false );
}

Vector Proj_EvalRGB( const ProjTex *p, const Vector &beamDir, const Vector &rightDir,
					 const Vector &upDir, float outerConeCos,
					 const Vector &lightToSample, ProjFrameMode_t frameMode )
{
	if ( !p || p->kind == PROJ_NONE || !p->rgba.Base() )
		return Vector( 1, 1, 1 );

	Vector dir = lightToSample;
	VectorNormalize( dir );

	if ( p->kind == PROJ_CUBE || p->kind == PROJ_SPHERE )
	{
		// Light-local: x=right, y=up, z=forward(beam).
		Vector local;
		local.x = DotProduct( dir, rightDir );
		local.y = DotProduct( dir, upDir );
		local.z = DotProduct( dir, beamDir );
		if ( p->kind == PROJ_CUBE )
		{
			// Source cube axes are Z-up: X=right, Y=forward, Z=up.
			const Vector cubeDir( local.x, local.z, local.y );
			return SampleCubeDir( p, cubeDir );
		}
		return SampleSphereDir( p, local );
	}

	// 2D planar framed to outer cone.
	float cosTheta = DotProduct( dir, beamDir );
	if ( cosTheta <= 1e-5f )
		return Vector( 0, 0, 0 );
	float outerCos = outerConeCos;
	if ( outerCos >= 0.999f )
		outerCos = 0.999f;
	if ( outerCos <= -0.999f )
		return Vector( 0, 0, 0 );
	float alpha = acosf( max( -1.0f, min( 1.0f, outerCos ) ) );
	float tanA = tanf( alpha );
	if ( tanA < 1e-6f )
		return Vector( 0, 0, 0 );

	float u = DotProduct( dir, rightDir ) / ( cosTheta * tanA );
	float v = DotProduct( dir, upDir ) / ( cosTheta * tanA );
	const float frameScale = ( frameMode == PROJ_FRAME_FIT ) ? 1.41421356f : 1.0f;
	float uu = 0.5f + 0.5f * u * frameScale;
	float vv = 0.5f + 0.5f * v * frameScale;
	return SampleRGBABilinear( p->rgba.Base(), p->width, p->height, uu, vv, false );
}

Vector Proj_EvalLightRGB( const directlight_t *dl, const Vector &lightToSample )
{
	if ( !dl || !dl->m_pProj )
		return Vector( 1, 1, 1 );
	Vector up;
	CrossProduct( dl->light.normal, dl->m_vecProjRight, up );
	if ( up.LengthSqr() < 1e-10f )
		up = dl->m_vecProjUp;
	else
		VectorNormalize( up );
	const ProjFrameMode_t frame = ( dl->m_nProjFrameMode == (int)PROJ_FRAME_FIT )
		? PROJ_FRAME_FIT : PROJ_FRAME_FILL;
	return Proj_EvalRGB( dl->m_pProj, dl->light.normal, dl->m_vecProjRight, up,
						 dl->light.stopdot2, lightToSample, frame );
}

bool Proj_IsOmniEnvmap( const directlight_t *dl )
{
	if ( !dl || !dl->m_pProj )
		return false;
	const ProjKind_t k = Proj_Kind( dl->m_pProj );
	return k == PROJ_CUBE || k == PROJ_SPHERE;
}

void Proj_AssignGpuLayers()
{
	for ( int i = 0; i < g_ProjList.Count(); ++i )
		g_ProjList[i]->gpuLayer = -1;

	int nCookie = 0, nCube = 0;
	for ( int i = 0; i < g_ProjList.Count(); ++i )
	{
		ProjTex *p = g_ProjList[i];
		if ( !p )
			continue;
		if ( p->kind == PROJ_2D || p->kind == PROJ_SPHERE )
			p->gpuLayer = nCookie++;
		else if ( p->kind == PROJ_CUBE )
			p->gpuLayer = nCube++;
	}
}

static void PackArray( ProjKind_t kind, ProjGpuArray_t &out )
{
	out.width = 1;
	out.height = 1;
	out.layers = 1;
	out.rgba.assign( 4, 255 );

	CUtlVector<ProjTex *> layers;
	int maxW = 1, maxH = 1;
	for ( int i = 0; i < g_ProjList.Count(); ++i )
	{
		ProjTex *p = g_ProjList[i];
		if ( !p || p->gpuLayer < 0 )
			continue;
		const bool match =
			( kind == PROJ_CUBE && p->kind == PROJ_CUBE ) ||
			( kind == PROJ_2D && ( p->kind == PROJ_2D || p->kind == PROJ_SPHERE ) );
		if ( !match )
			continue;
		while ( layers.Count() <= p->gpuLayer )
			layers.AddToTail( NULL );
		layers[p->gpuLayer] = p;
		maxW = max( maxW, p->width );
		maxH = max( maxH, p->height );
	}
	if ( layers.Count() == 0 )
		return;

	if ( kind == PROJ_2D )
	{
		out.width = maxW;
		out.height = maxH;
		out.layers = layers.Count();
		out.rgba.assign( (size_t)out.layers * (size_t)maxW * (size_t)maxH * 4u, 0 );
		for ( int layer = 0; layer < layers.Count(); ++layer )
		{
			ProjTex *p = layers[layer];
			if ( !p )
				continue;
			unsigned char *dstBase = &out.rgba[(size_t)layer * (size_t)maxW * (size_t)maxH * 4u];
			for ( int y = 0; y < p->height; ++y )
			{
				memcpy( dstBase + (size_t)y * (size_t)maxW * 4u,
						p->rgba.Base() + (size_t)y * (size_t)p->width * 4u,
						(size_t)p->width * 4u );
			}
		}
		Msg( "ProjectedTexture: GPU 2D array %dx%d x %d layer(s) (planar+spheremaps)\n",
			 maxW, maxH, layers.Count() );
	}
	else
	{
		out.width = maxW;
		out.height = maxH;
		out.layers = layers.Count() * 6;
		out.rgba.assign( (size_t)out.layers * (size_t)maxW * (size_t)maxH * 4u, 0 );
		for ( int ci = 0; ci < layers.Count(); ++ci )
		{
			ProjTex *p = layers[ci];
			if ( !p )
				continue;
			for ( int f = 0; f < 6; ++f )
			{
				const int layer = ci * 6 + f;
				unsigned char *dstBase = &out.rgba[(size_t)layer * (size_t)maxW * (size_t)maxH * 4u];
				const unsigned char *srcFace = p->rgba.Base() + f * p->width * p->height * 4;
				for ( int y = 0; y < p->height; ++y )
				{
					memcpy( dstBase + (size_t)y * (size_t)maxW * 4u,
							srcFace + (size_t)y * (size_t)p->width * 4u,
							(size_t)p->width * 4u );
				}
			}
		}
		Msg( "ProjectedTexture: GPU cubemap array %dx%d x %d cube(s) (%d layers)\n",
			 maxW, maxH, layers.Count(), out.layers );
	}
}

void Proj_BuildGpuCookieArray( ProjGpuArray_t &out )
{
	PackArray( PROJ_2D, out );
}

void Proj_BuildGpuCubeArray( ProjGpuArray_t &out )
{
	PackArray( PROJ_CUBE, out );
}
