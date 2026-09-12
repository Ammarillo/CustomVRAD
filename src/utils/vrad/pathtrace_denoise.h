//========= Copyright PathRAD contributors. ============//
// Path-trace denoise: OIDN, OptiX, or Sakai 2024.
//=============================================================================//
#ifndef PATHTRACE_DENOISE_H
#define PATHTRACE_DENOISE_H
#pragma once

enum PathTraceDenoiser_t
{
	PT_DENOISER_OIDN	= 0,
	PT_DENOISER_OPTIX	= 1,
	PT_DENOISER_SAKAI	= 2,
};

extern PathTraceDenoiser_t	g_PathTraceDenoiser;

bool PathTraceDenoise_Init();
void PathTraceDenoise_Shutdown();
bool PathTraceDenoise_IsReady();
const char *PathTraceDenoise_Name();

// varMean: optional per-pixel Var(E[L]) for Sakai; ignored by OIDN/OptiX.
bool PathTraceDenoise_DenoiseRGB( float *rgb, int width, int height, const float *varMean = nullptr );

#endif
