//========= Copyright CustomVRAD contributors. ============//
// IESNA LM-63-1995 / 2002 photometric profile loader + sampler.
//=============================================================================//

#include "ies_profile.h"
#include "vrad.h"
#include "tier1/utldict.h"
#include "tier1/utlbuffer.h"
#include "tier1/strtools.h"
#include "cmdlib.h"
#include "filesystem_tools.h"
#include <ctype.h>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include "tier1/utlstring.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct IesProfile
{
	char					szPath[MAX_PATH];
	int						nVert;			// vertical angles
	int						nHoriz;			// horizontal angles
	int						photometricType; // 1 = Type C (common)
	float					candelaMult;
	CUtlVector<float>		vertAngles;		// degrees
	CUtlVector<float>		horizAngles;	// degrees
	CUtlVector<float>		candela;		// nVert * nHoriz, row-major [h][v] or [v][h]?
	// Stored as candela[ h * nVert + v ] - matches common LM-63 layout
	// (all verticals for horiz[0], then horiz[1], ...).
	float					peak;			// max candela (pre-normalize sample)
	int						gpuLayer;		// set by IES_BuildGpuAtlas
};

static CUtlDict<IesProfile *, int> g_IesByPath;
static CUtlVector<IesProfile *> g_IesList; // ownership for atlas order

extern char level_name[MAX_PATH];

static bool IesLoadFile( const char *pPath, CUtlBuffer &buf )
{
	// 1) VFS relative (GAME/MOD search paths)
	const char *pathIds[] = { "GAME", "MOD", nullptr };
	for ( int i = 0; i < 3; ++i )
	{
		FileHandle_t h = g_pFileSystem->Open( pPath, "rb", pathIds[i] );
		if ( h == FILESYSTEM_INVALID_HANDLE )
			continue;
		int n = (int)g_pFileSystem->Size( h );
		if ( n <= 0 )
		{
			g_pFileSystem->Close( h );
			continue;
		}
		buf.EnsureCapacity( n + 1 );
		buf.SeekPut( CUtlBuffer::SEEK_HEAD, 0 );
		g_pFileSystem->Read( buf.Base(), n, h );
		( (char *)buf.Base() )[n] = 0;
		buf.SeekPut( CUtlBuffer::SEEK_HEAD, n + 1 );
		buf.SeekGet( CUtlBuffer::SEEK_HEAD, 0 );
		g_pFileSystem->Close( h );
		return true;
	}

	// 2) Absolute / CRT fopen (bypasses VFS - catches gamedir\IES\file.ies)
	FILE *fp = fopen( pPath, "rb" );
	if ( !fp )
		return false;
	if ( fseek( fp, 0, SEEK_END ) != 0 )
	{
		fclose( fp );
		return false;
	}
	long n = ftell( fp );
	if ( n <= 0 )
	{
		fclose( fp );
		return false;
	}
	fseek( fp, 0, SEEK_SET );
	buf.EnsureCapacity( (int)n + 1 );
	buf.SeekPut( CUtlBuffer::SEEK_HEAD, 0 );
	size_t got = fread( buf.Base(), 1, (size_t)n, fp );
	fclose( fp );
	if ( got != (size_t)n )
		return false;
	( (char *)buf.Base() )[n] = 0;
	buf.SeekPut( CUtlBuffer::SEEK_HEAD, (int)n + 1 );
	buf.SeekGet( CUtlBuffer::SEEK_HEAD, 0 );
	return true;
}

static void IesAddCandidate( CUtlVector<CUtlString> &list, const char *fmt, ... )
{
	char tmp[MAX_PATH * 2];
	va_list ap;
	va_start( ap, fmt );
	Q_vsnprintf( tmp, sizeof( tmp ), fmt, ap );
	va_end( ap );
	Q_FixSlashes( tmp );

	for ( int i = 0; i < list.Count(); ++i )
	{
		if ( !Q_stricmp( list[i].Get(), tmp ) )
			return;
	}
	list.AddToTail( tmp );
}

static bool IesTryOpen( const char *pRel, CUtlBuffer &buf, char *pResolved, int nResolved )
{
	char rel[MAX_PATH];
	Q_strncpy( rel, pRel ? pRel : "", sizeof( rel ) );
	Q_FixSlashes( rel );
	// Strip quotes Hammer sometimes leaves
	if ( rel[0] == '\"' )
	{
		int n = (int)strlen( rel );
		if ( n >= 2 && rel[n - 1] == '\"' )
		{
			rel[n - 1] = 0;
			Q_memmove( rel, rel + 1, n - 1 );
		}
	}
	// Strip accidental IES/ prefix
	if ( !Q_strnicmp( rel, "IES/", 4 ) || !Q_strnicmp( rel, "ies/", 4 ) )
		Q_memmove( rel, rel + 4, strlen( rel + 4 ) + 1 );
	// Strip leading slash
	while ( rel[0] == '/' || rel[0] == '\\' )
		Q_memmove( rel, rel + 1, strlen( rel + 1 ) + 1 );

	char relNoExt[MAX_PATH];
	Q_strncpy( relNoExt, rel, sizeof( relNoExt ) );
	char *dot = Q_strrchr( relNoExt, '.' );
	if ( dot && !Q_stricmp( dot, ".ies" ) )
		*dot = 0;

	CUtlVector<CUtlString> cands;
	const char *gd = gamedir[0] ? gamedir : "";

	IesAddCandidate( cands, "IES/%s", rel );
	IesAddCandidate( cands, "IES/%s.ies", relNoExt );
	if ( gd[0] )
	{
		IesAddCandidate( cands, "%sIES/%s", gd, rel );
		IesAddCandidate( cands, "%sIES/%s.ies", gd, relNoExt );
	}
	IesAddCandidate( cands, "%s", rel );
	IesAddCandidate( cands, "%s.ies", relNoExt );

	if ( level_name[0] )
	{
		IesAddCandidate( cands, "IES/maps/%s/%s", level_name, rel );
		IesAddCandidate( cands, "IES/maps/%s/%s.ies", level_name, relNoExt );
		if ( gd[0] )
		{
			IesAddCandidate( cands, "%sIES/maps/%s/%s", gd, level_name, rel );
			IesAddCandidate( cands, "%sIES/maps/%s/%s.ies", gd, level_name, relNoExt );
		}
	}

	for ( int i = 0; i < cands.Count(); ++i )
	{
		if ( IesLoadFile( cands[i].Get(), buf ) )
		{
			Q_strncpy( pResolved, cands[i].Get(), nResolved );
			return true;
		}
	}

	Warning( "IES: could not open '%s'\n", pRel );
	Warning( "IES: gamedir='%s' - put files in <gamedir>IES\\ (e.g. garrysmod\\IES\\light01.ies)\n",
			 gd[0] ? gd : "(unset)" );
	Warning( "IES: tried %d paths, first few:\n", cands.Count() );
	for ( int i = 0; i < cands.Count() && i < 8; ++i )
		Warning( "  - %s\n", cands[i].Get() );
	return false;
}

// Tokenize whitespace-separated floats/ints; skip TILT=INCLUDE blocks simply by
// requiring TILT=NONE or ignoring tilt data when present with zero lamps path.
static bool IesParseTokens( const char *text, CUtlVector<float> &nums, char *tiltOut, int nTilt )
{
	nums.RemoveAll();
	tiltOut[0] = 0;
	const char *p = text;

	// Find TILT=
	const char *tilt = Q_stristr( p, "TILT=" );
	if ( tilt )
	{
		tilt += 5;
		while ( *tilt == ' ' || *tilt == '\t' )
			++tilt;
		int i = 0;
		while ( *tilt && *tilt != '\r' && *tilt != '\n' && i < nTilt - 1 )
			tiltOut[i++] = *tilt++;
		tiltOut[i] = 0;
	}

	// Skip to first numeric line after TILT line (photometric header).
	// Scan for a line that starts with an integer (num lamps).
	const char *scan = tilt ? tilt : p;
	while ( *scan )
	{
		// Advance to line start
		while ( *scan == '\r' || *scan == '\n' )
			++scan;
		const char *line = scan;
		while ( *scan && *scan != '\r' && *scan != '\n' )
			++scan;

		// Skip keyword lines / empty
		while ( line < scan && ( *line == ' ' || *line == '\t' ) )
			++line;
		if ( line >= scan )
			continue;
		if ( isalpha( (unsigned char)*line ) || *line == '[' || *line == ']' )
			continue;
		if ( *line == 'T' || *line == 't' ) // TILT residual
			continue;

		// From here: photometric numbers to EOF
		p = line;
		break;
	}

	while ( *p )
	{
		while ( *p && ( isspace( (unsigned char)*p ) || *p == ',' ) )
			++p;
		if ( !*p )
			break;
		if ( isalpha( (unsigned char)*p ) || *p == '[' )
		{
			// Skip token
			while ( *p && !isspace( (unsigned char)*p ) )
				++p;
			continue;
		}
		char *end = nullptr;
		float v = (float)strtod( p, &end );
		if ( end == p )
		{
			++p;
			continue;
		}
		nums.AddToTail( v );
		p = end;
	}
	return nums.Count() > 10;
}

static IesProfile *IesParseBuffer( const char *resolvedPath, CUtlBuffer &buf )
{
	CUtlVector<float> nums;
	char tilt[64];
	if ( !IesParseTokens( (const char *)buf.Base(), nums, tilt, sizeof( tilt ) ) )
	{
		Warning( "IES: failed to tokenize '%s'\n", resolvedPath );
		return NULL;
	}

	if ( Q_stricmp( tilt, "NONE" ) != 0 && tilt[0] != 0 )
	{
		Warning( "IES: '%s' has TILT=%s (only TILT=NONE supported) - skipping.\n",
				 resolvedPath, tilt );
		return NULL;
	}

	int idx = 0;
	auto need = [&]( int n ) -> bool {
		return idx + n <= nums.Count();
	};
	if ( !need( 10 ) )
	{
		Warning( "IES: '%s' photometric header too short\n", resolvedPath );
		return NULL;
	}

	/* numLamps */ (void)nums[idx++];
	/* lumens   */ (void)nums[idx++];
	float candelaMult = nums[idx++];
	if ( candelaMult <= 0.0f )
		candelaMult = 1.0f;
	int nVert = (int)( nums[idx++] + 0.5f );
	int nHoriz = (int)( nums[idx++] + 0.5f );
	int photoType = (int)( nums[idx++] + 0.5f );
	/* units */ (void)nums[idx++];
	/* width, length, height */ idx += 3;
	if ( !need( 3 ) )
		return NULL;
	/* ballast, ballastFactor, future */ idx += 3;

	if ( nVert < 2 || nVert > 1024 || nHoriz < 1 || nHoriz > 1024 )
	{
		Warning( "IES: '%s' bad angle counts (%d vert, %d horiz)\n", resolvedPath, nVert, nHoriz );
		return NULL;
	}
	if ( !need( nVert + nHoriz + nVert * nHoriz ) )
	{
		Warning( "IES: '%s' truncated candela table (have %d nums)\n", resolvedPath, nums.Count() );
		return NULL;
	}

	IesProfile *p = new IesProfile;
	Q_strncpy( p->szPath, resolvedPath, sizeof( p->szPath ) );
	p->nVert = nVert;
	p->nHoriz = nHoriz;
	p->photometricType = photoType;
	p->candelaMult = candelaMult;
	p->gpuLayer = -1;
	p->vertAngles.SetCount( nVert );
	p->horizAngles.SetCount( nHoriz );
	for ( int i = 0; i < nVert; ++i )
		p->vertAngles[i] = nums[idx++];
	for ( int i = 0; i < nHoriz; ++i )
		p->horizAngles[i] = nums[idx++];

	p->candela.SetCount( nVert * nHoriz );
	float peak = 0.0f;
	for ( int h = 0; h < nHoriz; ++h )
	{
		for ( int v = 0; v < nVert; ++v )
		{
			float c = nums[idx++] * candelaMult;
			if ( c < 0.0f )
				c = 0.0f;
			p->candela[h * nVert + v] = c;
			if ( c > peak )
				peak = c;
		}
	}
	p->peak = ( peak > 1e-8f ) ? peak : 1.0f;
	return p;
}

const IesProfile *IES_FindOrLoad( const char *pPath )
{
	if ( !pPath || !pPath[0] )
		return NULL;

	char key[MAX_PATH];
	Q_strncpy( key, pPath, sizeof( key ) );
	Q_FixSlashes( key );
	Q_strlower( key );

	int found = g_IesByPath.Find( key );
	if ( g_IesByPath.IsValidIndex( found ) )
		return g_IesByPath[found];

	CUtlBuffer buf( 0, 0, 0 ); // binary - raw IES bytes
	char resolved[MAX_PATH];
	if ( !IesTryOpen( pPath, buf, resolved, sizeof( resolved ) ) )
	{
		g_IesByPath.Insert( key, NULL );
		return NULL;
	}

	IesProfile *p = IesParseBuffer( resolved, buf );
	g_IesByPath.Insert( key, p );
	if ( p )
	{
		g_IesList.AddToTail( p );
		Msg( "IES: loaded '%s' (%d vert x %d horiz, peak=%.1f)\n",
			 resolved, p->nVert, p->nHoriz, p->peak );
	}
	return p;
}

void IES_BuildBasis( const Vector &beamDir, Vector &rightOut, Vector &upOut )
{
	Vector beam = beamDir;
	if ( VectorNormalize( beam ) < 1e-6f )
		beam.Init( 0, 0, -1 );

	Vector upPref( 0, 0, 1 );
	if ( fabsf( DotProduct( beam, upPref ) ) > 0.99f )
		upPref.Init( 0, 1, 0 );

	CrossProduct( upPref, beam, rightOut );
	if ( VectorNormalize( rightOut ) < 1e-6f )
	{
		rightOut.Init( 1, 0, 0 );
	}
	CrossProduct( beam, rightOut, upOut );
	VectorNormalize( upOut );
}

void IES_BuildBasis( const Vector &beamDir, const QAngle &angles,
					 Vector &rightOut, Vector &upOut )
{
	Vector beam = beamDir;
	if ( VectorNormalize( beam ) < 1e-6f )
		beam.Init( 0, 0, -1 );

	// Entity local frame from Hammer angles (pitch/yaw/roll). Even when the
	// beam is straight down and SetupLightNormalFromProps drops yaw from the
	// normal, AngleVectors still rotates right/up with yaw+roll - that is the
	// spin axis for asymmetric IES (Type C horizontal table).
	Vector entFwd, entRight, entUp;
	AngleVectors( angles, &entFwd, &entRight, &entUp );

	// Project entity right onto the plane perpendicular to the photometric axis.
	Vector right = entRight - beam * DotProduct( entRight, beam );
	if ( right.LengthSqr() < 1e-8f )
	{
		right = entUp - beam * DotProduct( entUp, beam );
	}
	if ( right.LengthSqr() < 1e-8f )
	{
		// Degenerate Euler frame - fall back to world-up basis.
		IES_BuildBasis( beam, rightOut, upOut );
		return;
	}
	VectorNormalize( right );
	rightOut = right;
	CrossProduct( beam, rightOut, upOut );
	if ( VectorNormalize( upOut ) < 1e-6f )
	{
		IES_BuildBasis( beam, rightOut, upOut );
		return;
	}
	// Re-orthogonalize right against (beam, up)
	CrossProduct( upOut, beam, rightOut );
	VectorNormalize( rightOut );
}

static float IesInterp1D( const CUtlVector<float> &angles, float x, int &i0, int &i1, float &t )
{
	const int n = angles.Count();
	if ( n <= 1 )
	{
		i0 = i1 = 0;
		t = 0;
		return angles[0];
	}
	// angles are typically monotonic increasing
	if ( x <= angles[0] )
	{
		i0 = i1 = 0;
		t = 0;
		return angles[0];
	}
	if ( x >= angles[n - 1] )
	{
		i0 = i1 = n - 1;
		t = 0;
		return angles[n - 1];
	}
	int lo = 0, hi = n - 1;
	while ( hi - lo > 1 )
	{
		int mid = ( lo + hi ) >> 1;
		if ( angles[mid] <= x )
			lo = mid;
		else
			hi = mid;
	}
	i0 = lo;
	i1 = hi;
	float denom = angles[hi] - angles[lo];
	t = ( denom > 1e-8f ) ? ( ( x - angles[lo] ) / denom ) : 0.0f;
	return x;
}

static float IesLookup( const IesProfile *p, float thetaDeg, float phiDeg )
{
	// Wrap / mirror horizontal according to span
	const float h0 = p->horizAngles[0];
	const float h1 = p->horizAngles[p->nHoriz - 1];
	float span = h1 - h0;
	if ( span < 0.0f )
		span = -span;

	if ( p->nHoriz == 1 )
	{
		phiDeg = h0;
	}
	else if ( span <= 90.0f + 0.5f )
	{
		// Quadrant symmetry
		while ( phiDeg < 0.0f )
			phiDeg += 360.0f;
		while ( phiDeg >= 360.0f )
			phiDeg -= 360.0f;
		if ( phiDeg > 180.0f )
			phiDeg = 360.0f - phiDeg;
		if ( phiDeg > 90.0f )
			phiDeg = 180.0f - phiDeg;
	}
	else if ( span <= 180.0f + 0.5f )
	{
		while ( phiDeg < 0.0f )
			phiDeg += 360.0f;
		while ( phiDeg >= 360.0f )
			phiDeg -= 360.0f;
		if ( phiDeg > 180.0f )
			phiDeg = 360.0f - phiDeg;
	}
	else
	{
		while ( phiDeg < h0 )
			phiDeg += 360.0f;
		while ( phiDeg > h1 + 0.01f )
			phiDeg -= 360.0f;
	}

	if ( thetaDeg < p->vertAngles[0] || thetaDeg > p->vertAngles[p->nVert - 1] )
		return 0.0f;

	int v0, v1, hA, hB;
	float tv, th;
	IesInterp1D( p->vertAngles, thetaDeg, v0, v1, tv );
	IesInterp1D( p->horizAngles, phiDeg, hA, hB, th );

	const float c00 = p->candela[hA * p->nVert + v0];
	const float c01 = p->candela[hA * p->nVert + v1];
	const float c10 = p->candela[hB * p->nVert + v0];
	const float c11 = p->candela[hB * p->nVert + v1];
	const float c0 = c00 + ( c01 - c00 ) * tv;
	const float c1 = c10 + ( c11 - c10 ) * tv;
	return c0 + ( c1 - c0 ) * th;
}

float IES_Eval( const IesProfile *p, const Vector &beamDir, const Vector &rightDir,
				const Vector &lightToSample )
{
	if ( !p )
		return 0.0f;

	Vector d = lightToSample;
	float len = VectorNormalize( d );
	if ( len < 1e-8f )
		return 0.0f;

	Vector beam = beamDir;
	VectorNormalize( beam );
	Vector right = rightDir;
	VectorNormalize( right );

	float cosTheta = DotProduct( d, beam );
	cosTheta = max( -1.0f, min( 1.0f, cosTheta ) );
	float thetaDeg = acosf( cosTheta ) * ( 180.0f / (float)M_PI );

	Vector up;
	CrossProduct( beam, right, up );
	VectorNormalize( up );

	Vector proj = d - beam * cosTheta;
	float phiDeg = 0.0f;
	if ( proj.LengthSqr() > 1e-10f )
	{
		VectorNormalize( proj );
		float c = DotProduct( proj, right );
		float s = DotProduct( proj, up );
		phiDeg = atan2f( s, c ) * ( 180.0f / (float)M_PI );
		if ( phiDeg < 0.0f )
			phiDeg += 360.0f;
	}

	float cd = IesLookup( p, thetaDeg, phiDeg );
	return cd / p->peak;
}

static int g_iesGpuResV = IES_GPU_RES_V_MIN;
static int g_iesGpuResH = IES_GPU_RES_H_MIN;

int IES_GpuResV() { return g_iesGpuResV; }
int IES_GpuResH() { return g_iesGpuResH; }

static int IesChooseAtlasResV( const IesProfile *p )
{
	if ( !p || p->nVert < 2 || p->vertAngles.Count() < 2 )
		return IES_GPU_RES_V_MIN;
	const float span = p->vertAngles[p->nVert - 1] - p->vertAngles[0];
	if ( span < 1e-3f )
		return IES_GPU_RES_V_MIN;
	// Cover 0..180 deg at the source's angular density.
	const int need = (int)ceilf( (float)p->nVert * ( 180.0f / span ) );
	return max( IES_GPU_RES_V_MIN, min( IES_GPU_RES_V_MAX, need ) );
}

static int IesChooseAtlasResH( const IesProfile *p )
{
	if ( !p || p->nHoriz <= 1 || p->horizAngles.Count() < 2 )
		return IES_GPU_RES_H_MIN; // radial - coarse phi grid is fine
	float span = p->horizAngles[p->nHoriz - 1] - p->horizAngles[0];
	if ( span < 0.0f )
		span = -span;
	if ( span < 1e-3f )
		return IES_GPU_RES_H_MIN;
	// Cover 0..360 deg at the source's angular density (symmetry tables expand via IesLookup).
	const int need = (int)ceilf( (float)p->nHoriz * ( 360.0f / span ) );
	return max( IES_GPU_RES_H_MIN, min( IES_GPU_RES_H_MAX, need ) );
}

int IES_BuildGpuAtlas( std::vector<float> &atlasOut )
{
	for ( int i = 0; i < g_IesList.Count(); ++i )
		g_IesList[i]->gpuLayer = -1;

	int nLayers = 0;
	int resV = IES_GPU_RES_V_MIN;
	int resH = IES_GPU_RES_H_MIN;
	for ( int i = 0; i < g_IesList.Count(); ++i )
	{
		if ( !g_IesList[i] )
			continue;
		g_IesList[i]->gpuLayer = nLayers++;
		resV = max( resV, IesChooseAtlasResV( g_IesList[i] ) );
		resH = max( resH, IesChooseAtlasResH( g_IesList[i] ) );
	}
	g_iesGpuResV = resV;
	g_iesGpuResH = resH;

	if ( nLayers <= 0 )
	{
		atlasOut.assign( (size_t)resV * (size_t)resH, 0.0f );
		return 1;
	}

	// Fixed domain so the GPU can sample without per-layer angle metadata:
	// vertical 0..180 deg, horizontal 0..360 deg (IesLookup applies file symmetry).
	const size_t layerFloats = (size_t)resV * (size_t)resH;
	atlasOut.assign( (size_t)nLayers * layerFloats, 0.0f );
	Msg( "IES: GPU atlas %dx%d x %d layer(s) (%.1f MB)\n",
		 resV, resH, nLayers,
		 (double)( nLayers * layerFloats * sizeof( float ) ) / ( 1024.0 * 1024.0 ) );

	for ( int i = 0; i < g_IesList.Count(); ++i )
	{
		IesProfile *p = g_IesList[i];
		if ( !p || p->gpuLayer < 0 )
			continue;
		float *layer = atlasOut.data() + (size_t)p->gpuLayer * layerFloats;
		for ( int iv = 0; iv < resV; ++iv )
		{
			float theta = 180.0f * ( (float)iv / (float)( resV - 1 ) );
			for ( int ih = 0; ih < resH; ++ih )
			{
				float phi = 360.0f * ( (float)ih / (float)( resH - 1 ) );
				layer[ih * resV + iv] = IesLookup( p, theta, phi ) / p->peak;
			}
		}
	}
	return nLayers;
}

int IES_GpuLayer( const IesProfile *p )
{
	return p ? p->gpuLayer : -1;
}
