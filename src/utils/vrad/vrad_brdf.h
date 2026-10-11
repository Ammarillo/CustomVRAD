//========= Copyright PathRAD contributors. ============//
// Compile-time surface roughness / metalness ($vrad_roughness*, $vrad_metalness*).
// The game never reads these. They only change how VRAD bounces light.
//=============================================================================//
#ifndef VRAD_BRDF_H
#define VRAD_BRDF_H
#pragma once

#include "mathlib/vector.h"
#include <math.h>
#include <vector>
#include <stdint.h>

// VMT keys (PathRAD, compile only):
//   $vrad_roughness       "1"     - 0 = mirror, 1 = fully rough (default 1)
//   $vrad_roughnessmap    "path"  - linear greyscale, R channel. 0 = smooth, 1 = rough
//   $vrad_metalness       "1"     - 0 = dielectric, 1 = metal
//   $vrad_metallic        "1"     - alias of $vrad_metalness
//   $vrad_metalnessmap    "path"  - linear greyscale, R channel
//   $vrad_metallicmap     "path"  - alias of $vrad_metalnessmap
//
// A map replaces the scalar per texel. If both are set, the scalar multiplies the map.
// If only the map is set, the multiplier is 1.
// No keys: Lambertian bounce (same as roughness 1, metalness 0).
// Metalness omitted on a material that sets roughness: 1 (metal).
// Explicit $vrad_metalness 0 is glossy paint (~4% sharp when roughness is 0).
// A mirror is metalness 1 (or omitted) and roughness 0.

struct VRadBrdfSample_t
{
	float roughness; // 0 mirror .. 1 rough
	float metalness; // 0 dielectric .. 1 metal
	bool authored;
};

struct VRadBrdfLobes_t
{
	Vector wide;   // energy that stays in the diffuse lobe
	Vector sharp;  // energy in the directional specular lobe
	float pSpec;   // probability of sampling the sharp lobe
	bool legacy;   // true = exact Lambertian (throughput *= albedo)
};

// wide = diffuse albedo of a rough dielectric. sharp = 0. legacy = true.
VRadBrdfLobes_t VRadBrdf_Lobes( const Vector &albedo, float roughness, float metalness );

// GGX specular shape without Fresnel. Multiply a Lambert NEE sample by this
// when throughput already carries the specular color. 0 if the light is below the horizon.
float VRadBrdf_GgxShape( const Vector &n, const Vector &wo, const Vector &wi, float roughness );

// incomingDir points at the surface (the path ray). n faces the ray.
// Roughness below kVRadBrdfMirrorRough returns the perfect reflection.
// Above that, the direction is a GGX visible-normal sample (Heitz, JCGT 2018).
Vector VRadBrdf_SampleGloss( const Vector &n, const Vector &incomingDir, float roughness, float u1, float u2 );

void VRadBrdf_EnsureCache( void );
bool VRadBrdf_HasAny( void );

// Brush face at a world position. Unauthored materials return roughness 1, metalness 0.
VRadBrdfSample_t VRadBrdf_SampleFace( int facenum, const Vector &worldPos );

// Studio / material name at a mesh UV (static props).
VRadBrdfSample_t VRadBrdf_SampleMaterialUV( const char *pMaterialName, float u, float v );

// Classic radiosity: fold authored materials into the diffuse lobe.
// Smooth metals stop filling the room; rough metals still color-bleed.
// Directional specular is path-trace only.
void VRadBrdf_ApplyPatchReflectivity( void );

// GPU: 3 x float4 per triangle + Texture2DArray (R = roughness, G = metalness).
//   [0] roughValueOrMul, metalValueOrMul, layer (-1 = no map), flags
//   [1] uv0.xy, uv1.xy
//   [2] uv2.xy, unused
// flags: bit0 authored, bit1 roughness map, bit2 metalness map, bit3 clamp U, bit4 clamp V
static const int kVRadBrdfMetaFloat4s = 3;

struct VRadBrdfGpuArray_t
{
	int width;
	int height;
	int layers;
	std::vector<unsigned char> rgba; // layers * width * height * 4
};

// Fills meta for every captured DXR triangle (legacy row if the material opts out).
void VRadBrdf_BuildGpuUpload( VRadBrdfGpuArray_t &outArray, std::vector<float> &outMetaFloat4s );

// Below this roughness the lobe is a delta. Point lights have no area, so a BSDF
// sample cannot see them; Lambertian vertices connect those lights through the
// triangle instead. Rougher gloss samples the GGX lobe and evaluates it toward
// each light (PBRT path tracer, delta-light case).
static const float kVRadBrdfMirrorRough = 0.08f;

// PBRT path regularization: after a non-specular bounce, double a too-small
// GGX alpha and clamp it so a later specular hit can still be light-sampled.
inline float VRadBrdf_Regularize( float roughness )
{
	float alpha = roughness * roughness;
	if ( alpha < 0.3f )
	{
		alpha *= 2.0f;
		if ( alpha < 0.1f )
			alpha = 0.1f;
		if ( alpha > 0.3f )
			alpha = 0.3f;
	}
	return sqrtf( alpha );
}

// One planar mirror triangle. f0 is Schlick F0 (0.04 .. albedo).
struct VRadBrdfMirrorTri_t
{
	Vector v0, v1, v2;
	Vector n;
	Vector f0;
	int face; // brush face, or -1 for a static prop
	float roughness;
};

// Rebuild from the captured DXR mesh. Safe to call more than once.
void VRadBrdf_BuildMirrorList( void );
int VRadBrdf_MirrorCount( void );
const VRadBrdfMirrorTri_t *VRadBrdf_MirrorTris( void );
// 5 x float4 per triangle: v0+face, v1+rough, v2, n, f0. Empty -> one zero record.
void VRadBrdf_PackMirrorGpu( std::vector<float> &outFloats );

#endif // VRAD_BRDF_H
