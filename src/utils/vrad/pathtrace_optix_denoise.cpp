//========= Copyright CustomVRAD contributors. ============//
// NVIDIA OptiX AI HDR denoise for path-traced luxels.
//=============================================================================//

#include "pathtrace_optix_denoise.h"
#include "pathtrace_dxr.h"

#include <mutex>
#include <vector>
#include <cstring>

#include "tier0/dbg.h"

#if defined( VRAD_HAS_OPTIX )

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

static OptixDeviceContext g_optixCtx = nullptr;
static OptixDenoiser g_denoiser = nullptr;
static CUstream g_stream = nullptr;
static CUdeviceptr g_state = 0;
static CUdeviceptr g_scratch = 0;
static CUdeviceptr g_intensity = 0;
static CUdeviceptr g_input = 0;
static CUdeviceptr g_output = 0;
static size_t g_stateBytes = 0;
static size_t g_scratchBytes = 0;
static size_t g_imageBytes = 0;
static unsigned int g_setupW = 0;
static unsigned int g_setupH = 0;
static std::mutex g_mutex;
static bool g_ready = false;

static void OptixLog( unsigned int level, const char *tag, const char *msg, void * )
{
	if ( level >= 3 )
		Warning( "[PathTrace-OptiX][%s] %s\n", tag ? tag : "?", msg ? msg : "" );
}

static void PathTraceOptixDenoise_ClearUnlocked()
{
	if ( g_denoiser )
	{
		optixDenoiserDestroy( g_denoiser );
		g_denoiser = nullptr;
	}
	if ( g_input ) { cudaFree( (void *)g_input ); g_input = 0; }
	if ( g_output ) { cudaFree( (void *)g_output ); g_output = 0; }
	if ( g_state ) { cudaFree( (void *)g_state ); g_state = 0; }
	if ( g_scratch ) { cudaFree( (void *)g_scratch ); g_scratch = 0; }
	if ( g_intensity ) { cudaFree( (void *)g_intensity ); g_intensity = 0; }
	g_stateBytes = g_scratchBytes = g_imageBytes = 0;
	g_setupW = g_setupH = 0;
	if ( g_stream )
	{
		cudaStreamDestroy( g_stream );
		g_stream = nullptr;
	}
	if ( g_optixCtx )
	{
		optixDeviceContextDestroy( g_optixCtx );
		g_optixCtx = nullptr;
	}
	g_ready = false;
}

static bool PathTraceOptixDenoise_EnsureSize( unsigned int w, unsigned int h )
{
	const size_t imgBytes = (size_t)w * (size_t)h * 3u * sizeof( float );
	if ( !g_input || g_imageBytes < imgBytes )
	{
		if ( g_input ) cudaFree( (void *)g_input );
		if ( g_output ) cudaFree( (void *)g_output );
		g_input = g_output = 0;
		if ( cudaMalloc( (void **)&g_input, imgBytes ) != cudaSuccess ||
			 cudaMalloc( (void **)&g_output, imgBytes ) != cudaSuccess )
		{
			Warning( "[PathTrace-OptiX] cudaMalloc image failed.\n" );
			return false;
		}
		g_imageBytes = imgBytes;
	}

	// Setup for max seen size (invoke allows smaller).
	unsigned int needW = w > g_setupW ? w : g_setupW;
	unsigned int needH = h > g_setupH ? h : g_setupH;
	if ( needW < 16 ) needW = 16;
	if ( needH < 16 ) needH = 16;
	if ( g_denoiser && needW == g_setupW && needH == g_setupH )
		return true;

	OptixDenoiserSizes sizes = {};
	if ( optixDenoiserComputeMemoryResources( g_denoiser, needW, needH, &sizes ) != OPTIX_SUCCESS )
	{
		Warning( "[PathTrace-OptiX] ComputeMemoryResources failed.\n" );
		return false;
	}

	if ( g_state ) cudaFree( (void *)g_state );
	if ( g_scratch ) cudaFree( (void *)g_scratch );
	g_state = g_scratch = 0;
	g_stateBytes = sizes.stateSizeInBytes;
	g_scratchBytes = sizes.withoutOverlapScratchSizeInBytes;
	if ( cudaMalloc( (void **)&g_state, g_stateBytes ) != cudaSuccess ||
		 cudaMalloc( (void **)&g_scratch, g_scratchBytes ) != cudaSuccess )
	{
		Warning( "[PathTrace-OptiX] cudaMalloc state/scratch failed.\n" );
		return false;
	}

	if ( optixDenoiserSetup( g_denoiser, g_stream, needW, needH,
							 g_state, g_stateBytes, g_scratch, g_scratchBytes ) != OPTIX_SUCCESS )
	{
		Warning( "[PathTrace-OptiX] Setup failed.\n" );
		return false;
	}
	g_setupW = needW;
	g_setupH = needH;
	return true;
}

bool PathTraceOptixDenoise_Init()
{
	std::lock_guard<std::mutex> lock( g_mutex );
	if ( g_ready )
		return true;
	PathTraceOptixDenoise_ClearUnlocked();

	if ( cudaFree( 0 ) != cudaSuccess )
	{
		Warning( "[PathTrace-OptiX] CUDA runtime not available.\n" );
		return false;
	}
	if ( optixInit() != OPTIX_SUCCESS )
	{
		Warning( "[PathTrace-OptiX] optixInit failed (driver/OptiX).\n" );
		return false;
	}

	OptixDeviceContextOptions opts = {};
	opts.logCallbackFunction = OptixLog;
	opts.logCallbackLevel = 3;
	CUcontext cuCtx = 0;
	if ( optixDeviceContextCreate( cuCtx, &opts, &g_optixCtx ) != OPTIX_SUCCESS )
	{
		Warning( "[PathTrace-OptiX] DeviceContextCreate failed.\n" );
		PathTraceOptixDenoise_ClearUnlocked();
		return false;
	}
	if ( cudaStreamCreate( &g_stream ) != cudaSuccess )
	{
		Warning( "[PathTrace-OptiX] Stream create failed.\n" );
		PathTraceOptixDenoise_ClearUnlocked();
		return false;
	}

	OptixDenoiserOptions dopt = {};
	dopt.guideAlbedo = 0;
	dopt.guideNormal = 0;
	dopt.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
	if ( optixDenoiserCreate( g_optixCtx, OPTIX_DENOISER_MODEL_KIND_HDR, &dopt, &g_denoiser ) != OPTIX_SUCCESS )
	{
		Warning( "[PathTrace-OptiX] DenoiserCreate(HDR) failed.\n" );
		PathTraceOptixDenoise_ClearUnlocked();
		return false;
	}
	if ( cudaMalloc( (void **)&g_intensity, sizeof( float ) ) != cudaSuccess )
	{
		Warning( "[PathTrace-OptiX] intensity alloc failed.\n" );
		PathTraceOptixDenoise_ClearUnlocked();
		return false;
	}

	Msg( "[PathTrace-OptiX] Ready (AI HDR denoiser)\n" );
	g_ready = true;
	return true;
}

void PathTraceOptixDenoise_Shutdown()
{
	std::lock_guard<std::mutex> lock( g_mutex );
	PathTraceOptixDenoise_ClearUnlocked();
}

bool PathTraceOptixDenoise_IsReady()
{
	return g_ready;
}

bool PathTraceOptixDenoise_DenoiseRGB( float *rgb, int width, int height )
{
	if ( !rgb || width < 1 || height < 1 )
		return false;

	std::lock_guard<std::mutex> lock( g_mutex );
	if ( !g_ready || !g_denoiser )
		return false;

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
				dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
			}
		}
		work = scratch.data();
	}

	if ( !PathTraceOptixDenoise_EnsureSize( (unsigned)padW, (unsigned)padH ) )
		return false;

	const size_t bytes = (size_t)padW * (size_t)padH * 3u * sizeof( float );
	if ( cudaMemcpyAsync( (void *)g_input, work, bytes, cudaMemcpyHostToDevice, g_stream ) != cudaSuccess )
		return false;

	OptixImage2D inImg = {};
	inImg.data = g_input;
	inImg.width = (unsigned)padW;
	inImg.height = (unsigned)padH;
	inImg.rowStrideInBytes = (size_t)padW * 3u * sizeof( float );
	inImg.pixelStrideInBytes = 3u * sizeof( float );
	inImg.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	OptixImage2D outImg = inImg;
	outImg.data = g_output;

	OptixDenoiserLayer layer = {};
	layer.input = inImg;
	layer.output = outImg;

	OptixDenoiserGuideLayer guide = {};

	OptixDenoiserParams params = {};
	params.hdrIntensity = g_intensity;
	params.hdrAverageColor = 0;
	params.temporalModeUsePreviousLayers = 0;
	params.flowMulX = 0.0f;
	params.flowMulY = 0.0f;
	params.blendFactor = 0.0f; // strength applied by host writeback

	if ( optixDenoiserComputeIntensity( g_denoiser, g_stream, &inImg, g_intensity,
										g_scratch, g_scratchBytes ) != OPTIX_SUCCESS )
	{
		Warning( "[PathTrace-OptiX] ComputeIntensity failed.\n" );
		return false;
	}
	if ( optixDenoiserInvoke( g_denoiser, g_stream, &params, g_state, g_stateBytes,
							  &guide, &layer, 1, 0, 0, g_scratch, g_scratchBytes ) != OPTIX_SUCCESS )
	{
		Warning( "[PathTrace-OptiX] Invoke failed.\n" );
		return false;
	}
	if ( cudaMemcpyAsync( work, (void *)g_output, bytes, cudaMemcpyDeviceToHost, g_stream ) != cudaSuccess )
		return false;
	if ( cudaStreamSynchronize( g_stream ) != cudaSuccess )
		return false;

	if ( bPad )
	{
		for ( int y = 0; y < height; ++y )
		{
			for ( int x = 0; x < width; ++x )
			{
				const float *src = work + ( (size_t)y * padW + x ) * 3u;
				float *dst = rgb + ( (size_t)y * width + x ) * 3u;
				dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
			}
		}
	}
	return true;
}

#else // !VRAD_HAS_OPTIX

bool PathTraceOptixDenoise_Init()
{
	Warning( "[PathTrace-OptiX] Build without VRAD_HAS_OPTIX.\n" );
	return false;
}
void PathTraceOptixDenoise_Shutdown() {}
bool PathTraceOptixDenoise_IsReady() { return false; }
bool PathTraceOptixDenoise_DenoiseRGB( float *, int, int ) { return false; }

#endif
