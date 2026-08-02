//========= Copyright CustomVRAD contributors. ============//
// VMT-driven colored light transmission through thin-sheet glass.
//=============================================================================//
#ifndef VRAD_FILTER_H
#define VRAD_FILTER_H
#pragma once

#include "mathlib/vector.h"
#include <vector>
#include <stdint.h>

// VMT keys (CustomVRAD):
//   $vrad_filter            "1"            — enable colored transmission
//   $vrad_filtermap         "path/to/tex"  — optional RGB filter map (else $basetexture)
//   $vrad_filterstrength    "1"            — 0 = no tint, 1 = full texture tint
//   $vrad_filteropacity     "0.05"         — flat absorption (0..1); keep low for stacked panes
//   $vrad_filterthickness   "1"            — pow(filterRGB, thickness) for denser stained glass
//   $nocull                 "1"            — two-sided (default for filters). Explicit "0" ignored
//                                           unless $vrad_filter_onesided is set.
//   $vrad_filter_onesided   "0"            — 1 = front-face tint only (no backface multiply)
//
// Stacked panes: Oklab gel stack (L0*L1, a0+a1, b0+b1) — see oklab.h.
// Transmittance: Oklab lerp(white, pow(rgb, thickness), strength) * (1 - opacity)

bool VRadFilter_FaceFilters( int facenum );

// True if face filters and is two-sided (default), or VMT has $nocull.
bool VRadFilter_FaceNoCull( int facenum );

// Whether this ray direction should multiply transmittance at facenum.
// Two-sided by default; with $vrad_filter_onesided only front-face hits.
bool VRadFilter_ShouldApply( int facenum, const Vector &rayDir );

// Same check by texdata index (brush sides / no face number).
bool VRadFilter_TexdataFilters( int texdata );

// Sample linear RGB transmittance at a world position on a filter face.
// Returns false if face does not filter (caller treats as opaque / no atten).
bool VRadFilter_SampleAtPos( int facenum, const Vector &worldPos, Vector &outT );

// Average transmittance for a face (centroid + corner taps). Fallback when no texture.
bool VRadFilter_AverageFace( int facenum, Vector &outT );

// Preload filter materials for all faces that opt in (safe to call multiple times).
void VRadFilter_EnsureCache( void );

// True if any face in the map has $vrad_filter enabled.
bool VRadFilter_HasAny( void );

// ---------------------------------------------------------------------------
// GPU upload: Texture2DArray (one layer per unique filter VTF) + per-tri meta
//   [0] avgT.xyz, mode (0=opaque, 1=avg-only, 2=textured)
//   [1] strength, opacity, thickness, flags (1=clampU, 2=clampV, 4=$nocull)
//   [2] texVecU (world → texel S)
//   [3] texVecV (world → texel T)
//   [4] layerIndex, uvScale (mipW/origW), mipW, mipH
// ---------------------------------------------------------------------------
static const int kVRadFilterMetaFloat4s = 5;

struct VRadFilterGpuArray_t
{
	int width;   // slice width (max of textures, min 1)
	int height;  // slice height
	int layers;  // array length (min 1)
	std::vector<unsigned char> rgba; // layers * width * height * 4, layer-major
};

// Build Texture2DArray of unique filter VTFs + per-tri meta for nTris.
// triIds[i] = full triangle ID (TRACE_ID_* | faceIndex). Opaque tris get mode=0.
bool VRadFilter_BuildGpuUpload( const uint32_t *triIds, uint32_t nTris,
								VRadFilterGpuArray_t &outArray,
								std::vector<float> &outMetaFloat4s );

#endif // VRAD_FILTER_H
