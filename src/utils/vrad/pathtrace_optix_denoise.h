//========= Copyright CustomVRAD contributors. ============//
// NVIDIA OptiX AI HDR denoise for path-traced luxels.
//=============================================================================//
#ifndef PATHTRACE_OPTIX_DENOISE_H
#define PATHTRACE_OPTIX_DENOISE_H
#pragma once

bool PathTraceOptixDenoise_Init();
void PathTraceOptixDenoise_Shutdown();
bool PathTraceOptixDenoise_IsReady();
bool PathTraceOptixDenoise_DenoiseRGB( float *rgb, int width, int height );

#endif
