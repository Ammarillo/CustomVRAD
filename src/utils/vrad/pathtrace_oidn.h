//========= Copyright CustomVRAD contributors. ============//
// Intel Open Image Denoise (OIDN) RTLightmap for path-traced luxels.
//=============================================================================//
#ifndef PATHTRACE_OIDN_H
#define PATHTRACE_OIDN_H
#pragma once

bool PathTraceOIDN_Init();
void PathTraceOIDN_Shutdown();
bool PathTraceOIDN_IsReady();

// Denoise an HDR RGB lightmap in-place (row-major float3, Source luxel irradiance).
// Thread-safe (serializes OIDN execute). Returns false on failure (buffer unchanged).
bool PathTraceOIDN_DenoiseRGB( float *rgb, int width, int height );

#endif
