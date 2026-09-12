//========= Copyright PathRAD contributors. ============//
// OIDN RTLightmap wrapper for path-traced lightmaps.
//=============================================================================//

#include "pathtrace_oidn.h"

#include <mutex>
#include <vector>

#include "tier0/dbg.h"

#include <OpenImageDenoise/oidn.h>

static OIDNDevice g_oidnDevice = nullptr;
static std::mutex g_oidnMutex;
static bool g_oidnReady = false;
static std::vector<OIDNFilter> g_oidnFilters; // released on shutdown

struct PtOidnTls
{
	OIDNFilter filter = nullptr;
};

static void PathTraceOIDN_ClearUnlocked()
{
	for ( OIDNFilter f : g_oidnFilters )
	{
		if ( f )
			oidnReleaseFilter( f );
	}
	g_oidnFilters.clear();
	if ( g_oidnDevice )
	{
		oidnReleaseDevice( g_oidnDevice );
		g_oidnDevice = nullptr;
	}
	g_oidnReady = false;
}

bool PathTraceOIDN_Init()
{
	std::lock_guard<std::mutex> lock( g_oidnMutex );
	if ( g_oidnReady )
		return true;

	PathTraceOIDN_ClearUnlocked();

	// Host luxel grids -> CPU device (GPU rejects shared host pointers).
	// numThreads=1 per execute: each bake worker has its own filter. Default OIDN
	// thread pools (all cores x N workers) oversubscribe and blow bake time.
	g_oidnDevice = oidnNewDevice( OIDN_DEVICE_TYPE_CPU );
	if ( !g_oidnDevice )
	{
		Warning( "[PathTrace-OIDN] Failed to create OIDN CPU device.\n" );
		return false;
	}

	oidnSetDeviceInt( g_oidnDevice, "numThreads", 1 );
	oidnCommitDevice( g_oidnDevice );
	{
		const char *msg = nullptr;
		OIDNError err = oidnGetDeviceError( g_oidnDevice, &msg );
		if ( err != OIDN_ERROR_NONE )
		{
			Warning( "[PathTrace-OIDN] commit device: %s\n", msg ? msg : "error" );
			PathTraceOIDN_ClearUnlocked();
			return false;
		}
	}

	Msg( "[PathTrace-OIDN] Ready (RTLightmap, CPU, 1 thread/filter, parallel)\n" );
	g_oidnReady = true;
	return true;
}

void PathTraceOIDN_Shutdown()
{
	std::lock_guard<std::mutex> lock( g_oidnMutex );
	PathTraceOIDN_ClearUnlocked();
}

bool PathTraceOIDN_IsReady()
{
	return g_oidnReady;
}

static OIDNFilter PathTraceOIDN_TlsFilter()
{
	static thread_local PtOidnTls tls;
	if ( tls.filter )
		return tls.filter;

	std::lock_guard<std::mutex> lock( g_oidnMutex );
	if ( !g_oidnReady || !g_oidnDevice )
		return nullptr;
	if ( tls.filter )
		return tls.filter;

	OIDNFilter f = oidnNewFilter( g_oidnDevice, "RTLightmap" );
	if ( !f )
	{
		Warning( "[PathTrace-OIDN] Failed to create RTLightmap filter.\n" );
		return nullptr;
	}
	tls.filter = f;
	g_oidnFilters.push_back( f );
	return f;
}

bool PathTraceOIDN_DenoiseRGB( float *rgb, int width, int height )
{
	if ( !rgb || width < 1 || height < 1 || !g_oidnReady )
		return false;

	OIDNFilter filter = PathTraceOIDN_TlsFilter();
	if ( !filter )
		return false;

	// Tiny charts: pad by edge clamp so the network has a usable footprint.
	const int minDim = 16;
	const int padW = ( width < minDim ) ? minDim : width;
	const int padH = ( height < minDim ) ? minDim : height;
	const bool bPad = ( padW != width || padH != height );

	std::vector<float> scratch;
	float *work = rgb;
	if ( bPad )
	{
		scratch.assign( (size_t)padW * (size_t)padH * 3u, 0.0f );
		for ( int y = 0; y < padH; ++y )
		{
			const int sy = ( y < height ) ? y : ( height - 1 );
			for ( int x = 0; x < padW; ++x )
			{
				const int sx = ( x < width ) ? x : ( width - 1 );
				const float *src = rgb + ( (size_t)sy * width + sx ) * 3u;
				float *dst = scratch.data() + ( (size_t)y * padW + x ) * 3u;
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
			}
		}
		work = scratch.data();
	}

	// CPU device: shared host images - no OIDNBuffer copy round-trip.
	oidnSetSharedFilterImage( filter, "color", work, OIDN_FORMAT_FLOAT3,
							  (size_t)padW, (size_t)padH, 0, 0, 0 );
	oidnSetSharedFilterImage( filter, "output", work, OIDN_FORMAT_FLOAT3,
							  (size_t)padW, (size_t)padH, 0, 0, 0 );
	oidnSetFilterBool( filter, "hdr", true );
	oidnCommitFilter( filter );

	const char *msg = nullptr;
	OIDNError err = oidnGetDeviceError( g_oidnDevice, &msg );
	if ( err != OIDN_ERROR_NONE )
	{
		Warning( "[PathTrace-OIDN] commit: %s\n", msg ? msg : "error" );
		return false;
	}

	oidnExecuteFilter( filter );
	err = oidnGetDeviceError( g_oidnDevice, &msg );
	if ( err != OIDN_ERROR_NONE )
	{
		Warning( "[PathTrace-OIDN] execute: %s\n", msg ? msg : "error" );
		return false;
	}

	if ( bPad )
	{
		for ( int y = 0; y < height; ++y )
		{
			for ( int x = 0; x < width; ++x )
			{
				const float *src = work + ( (size_t)y * padW + x ) * 3u;
				float *dst = rgb + ( (size_t)y * width + x ) * 3u;
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
			}
		}
	}
	return true;
}
