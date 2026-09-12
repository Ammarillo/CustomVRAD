//========= Copyright PathRAD contributors. ============//
// Pick OIDN / OptiX / Sakai for path-traced lightmaps.
//=============================================================================//

#include "pathtrace_denoise.h"
#include "pathtrace_oidn.h"
#include "pathtrace_optix_denoise.h"
#include "pathtrace_sakai.h"
#include "tier0/dbg.h"

PathTraceDenoiser_t g_PathTraceDenoiser = PT_DENOISER_OIDN;

bool PathTraceDenoise_Init()
{
	PathTraceDenoise_Shutdown();

	switch ( g_PathTraceDenoiser )
	{
	case PT_DENOISER_OPTIX:
		if ( PathTraceOptixDenoise_Init() )
			return true;
		Warning( "[PathTrace] OptiX denoise init failed - falling back to OIDN.\n" );
		g_PathTraceDenoiser = PT_DENOISER_OIDN;
		return PathTraceOIDN_Init();
	case PT_DENOISER_SAKAI:
		return PathTraceSakai_Init();
	case PT_DENOISER_OIDN:
	default:
		g_PathTraceDenoiser = PT_DENOISER_OIDN;
		return PathTraceOIDN_Init();
	}
}

void PathTraceDenoise_Shutdown()
{
	PathTraceOIDN_Shutdown();
	PathTraceOptixDenoise_Shutdown();
	PathTraceSakai_Shutdown();
}

bool PathTraceDenoise_IsReady()
{
	switch ( g_PathTraceDenoiser )
	{
	case PT_DENOISER_OPTIX:	return PathTraceOptixDenoise_IsReady();
	case PT_DENOISER_SAKAI:	return PathTraceSakai_IsReady();
	case PT_DENOISER_OIDN:
	default:				return PathTraceOIDN_IsReady();
	}
}

const char *PathTraceDenoise_Name()
{
	switch ( g_PathTraceDenoiser )
	{
	case PT_DENOISER_OPTIX:	return "OptiX";
	case PT_DENOISER_SAKAI:	return "Sakai2024";
	case PT_DENOISER_OIDN:
	default:				return "OIDN";
	}
}

bool PathTraceDenoise_DenoiseRGB( float *rgb, int width, int height, const float *varMean )
{
	switch ( g_PathTraceDenoiser )
	{
	case PT_DENOISER_OPTIX:	return PathTraceOptixDenoise_DenoiseRGB( rgb, width, height );
	case PT_DENOISER_SAKAI:	return PathTraceSakai_DenoiseRGB( rgb, width, height, varMean );
	case PT_DENOISER_OIDN:
	default:				return PathTraceOIDN_DenoiseRGB( rgb, width, height );
	}
}
