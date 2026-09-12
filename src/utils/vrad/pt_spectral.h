//========= Copyright PathRAD contributors. ============//
// Hero-wavelength helpers for path tracing.
//
// RGB->spectrum: Smits 1999 via PBRT-v3 tabulated basis (BSD).
//   Matt Pharr, Greg Humphreys, Wenzel Jakob - pbrt-v3 spectrum.cpp
// CIE XYZ CMFs: Wyman et al. analytic fit to CIE 1931 2 deg.
// XYZ->linear sRGB: IEC 61966-2-1 / Rec.709 matrix (PBRT).
//
// Lights add and albedo multiplies in wavelength space; the result is
// converted through CIE XYZ to linear RGB for the lightmap.
//=============================================================================//
#ifndef VRAD_PT_SPECTRAL_H
#define VRAD_PT_SPECTRAL_H
#pragma once

#include "mathlib/vector.h"
#include <math.h>

static const int kPtSpecN = 32;
static const float kPtSpecLambdaMin = 380.0f;
static const float kPtSpecLambdaMax = 780.0f;
static const float kPtCieYIntegral = 106.856895f;

// PBRT-v3 RGB2SpectLambda + Smits bases (nRGB2SpectSamples = 32).
static const float kPtRgb2SpectLambda[kPtSpecN] = {
	380.000000f, 390.967743f, 401.935486f, 412.903229f, 423.870972f, 434.838715f,
	445.806458f, 456.774200f, 467.741943f, 478.709686f, 489.677429f, 500.645172f,
	511.612915f, 522.580627f, 533.548340f, 544.516052f, 555.483765f, 566.451477f,
	577.419189f, 588.386902f, 599.354614f, 610.322327f, 621.290039f, 632.257751f,
	643.225464f, 654.193176f, 665.160889f, 676.128601f, 687.096313f, 698.064026f,
	709.031738f, 720.000000f
};

static const float kPtReflWhite[kPtSpecN] = {
	1.061896f, 1.061502f, 1.061434f, 1.062271f, 1.062204f, 1.062506f, 1.062394f, 1.062471f,
	1.062505f, 1.062437f, 1.062069f, 1.061317f, 1.061033f, 1.061387f, 1.061422f, 1.062034f,
	1.062550f, 1.062432f, 1.062525f, 1.062428f, 1.062475f, 1.062554f, 1.062533f, 1.062392f,
	1.062365f, 1.062526f, 1.061228f, 1.059426f, 1.059981f, 1.060255f, 1.060126f, 1.060657f
};
static const float kPtReflCyan[kPtSpecN] = {
	1.041463f, 1.032866f, 1.012615f, 1.035046f, 1.007866f, 1.042228f, 1.044260f, 1.053524f,
	1.018078f, 1.044273f, 1.052936f, 1.053703f, 1.053390f, 1.053778f, 1.052709f, 1.053045f,
	1.055055f, 1.055367f, 1.045431f, 0.623490f, 0.180381f, -0.007630f, -0.000152f, -0.007510f,
	-0.002171f, 0.000659f, 0.012279f, -0.004467f, 0.017120f, 0.004921f, 0.005876f, 0.025259f
};
static const float kPtReflMagenta[kPtSpecN] = {
	0.994221f, 0.989869f, 0.982937f, 0.996279f, 1.019896f, 1.016640f, 1.022091f, 0.996517f,
	1.009777f, 1.021542f, 0.640320f, 0.002501f, 0.006534f, 0.002833f, 0.000000f, -0.009059f,
	0.003394f, -0.003064f, 0.222039f, 0.631411f, 0.974810f, 0.972096f, 1.017377f, 0.998752f,
	0.947017f, 0.852586f, 0.948978f, 0.947519f, 0.995989f, 0.863014f, 0.891510f, 0.848665f
};
static const float kPtReflYellow[kPtSpecN] = {
	0.005574f, -0.004798f, -0.005254f, -0.006457f, -0.005969f, -0.002184f, 0.016781f, 0.096096f,
	0.212174f, 0.361691f, 0.539610f, 0.744088f, 0.922096f, 1.046030f, 1.051382f, 1.051199f,
	1.051053f, 1.051740f, 1.051604f, 1.051194f, 1.051159f, 1.051661f, 1.051404f, 1.051594f,
	1.051146f, 1.051512f, 1.050887f, 1.050892f, 1.047749f, 1.049327f, 1.043596f, 1.039228f
};
static const float kPtReflRed[kPtSpecN] = {
	0.165756f, 0.118464f, 0.124083f, 0.113713f, 0.078992f, 0.032206f, -0.010798f, 0.018052f,
	0.005341f, 0.013655f, -0.005956f, -0.001844f, -0.010572f, -0.002938f, -0.010790f, -0.008022f,
	-0.002267f, 0.007020f, -0.008153f, 0.607729f, 0.988316f, 0.993917f, 1.003934f, 0.992345f,
	0.999265f, 1.008462f, 0.983583f, 1.008502f, 0.974511f, 0.985433f, 0.934958f, 0.987139f
};
static const float kPtReflGreen[kPtSpecN] = {
	0.002649f, -0.005018f, -0.012547f, -0.009455f, -0.012526f, -0.007917f, -0.007996f, -0.009356f,
	0.065469f, 0.395729f, 0.752440f, 0.963765f, 0.998544f, 0.999930f, 0.999391f, 0.999944f,
	0.999391f, 0.999112f, 0.960196f, 0.631863f, 0.257974f, 0.009401f, -0.003080f, -0.004523f,
	-0.006893f, -0.009035f, -0.008591f, -0.008369f, -0.007869f, -0.000008f, 0.005430f, -0.002775f
};
static const float kPtReflBlue[kPtSpecN] = {
	0.992098f, 0.988764f, 0.995390f, 0.995293f, 0.991814f, 1.000258f, 0.999685f, 0.999881f,
	0.985040f, 0.790298f, 0.560822f, 0.331335f, 0.136924f, 0.018915f, -0.000005f, -0.000424f,
	-0.000419f, 0.001747f, 0.003800f, -0.000551f, -0.000044f, 0.007587f, 0.025796f, 0.038168f,
	0.049490f, 0.049596f, 0.049815f, 0.039841f, 0.030501f, 0.021243f, 0.006960f, 0.004173f
};

static const float kPtIllumWhite[kPtSpecN] = {
	1.156523f, 1.156723f, 1.156620f, 1.155578f, 1.156218f, 1.156767f, 1.156802f, 1.156768f,
	1.156356f, 1.156705f, 1.156513f, 1.156434f, 1.156802f, 1.147315f, 1.133932f, 1.129388f,
	1.129052f, 1.050486f, 1.045970f, 0.993667f, 0.956017f, 0.924675f, 0.914999f, 0.899395f,
	0.895425f, 0.888706f, 0.882228f, 0.879983f, 0.876352f, 0.880004f, 0.880657f, 0.883047f
};
static const float kPtIllumCyan[kPtSpecN] = {
	1.133448f, 1.126676f, 1.134683f, 1.135740f, 1.135637f, 1.136115f, 1.136218f, 1.136482f,
	1.135511f, 1.136406f, 1.136036f, 1.136012f, 1.135427f, 1.136310f, 1.135545f, 1.135373f,
	1.134950f, 1.111111f, 0.905987f, 0.611608f, 0.295398f, 0.095954f, -0.011651f, -0.012145f,
	-0.011148f, -0.011998f, -0.005051f, -0.007998f, -0.009472f, -0.005533f, -0.004543f, -0.012541f
};
static const float kPtIllumMagenta[kPtSpecN] = {
	1.037189f, 1.058754f, 1.076727f, 1.076271f, 1.079529f, 1.074364f, 1.072703f, 1.073245f,
	1.082376f, 1.084055f, 0.956076f, 0.551979f, 0.084191f, 0.000088f, -0.002309f, -0.001125f,
	0.000000f, -0.000273f, 0.014466f, 0.258831f, 0.529080f, 0.909666f, 1.069057f, 1.088733f,
	1.063762f, 1.020181f, 1.026220f, 1.078309f, 0.983338f, 1.070725f, 1.063425f, 1.015088f
};
static const float kPtIllumYellow[kPtSpecN] = {
	0.002776f, 0.003967f, -0.000146f, 0.000362f, -0.000258f, -0.000050f, -0.000244f, -0.000078f,
	0.049690f, 0.485160f, 1.029573f, 1.033321f, 1.036810f, 1.036488f, 1.036543f, 1.036860f,
	1.036565f, 1.036394f, 1.036721f, 1.036524f, 1.036153f, 1.034879f, 1.004273f, 0.842185f,
	0.737594f, 0.658532f, 0.605317f, 0.595498f, 0.594193f, 0.565177f, 0.560612f, 0.582286f
};
static const float kPtIllumRed[kPtSpecN] = {
	0.054711f, 0.055609f, 0.060756f, 0.056233f, 0.046170f, 0.038013f, 0.024424f, 0.003898f,
	-0.000561f, 0.000965f, 0.000373f, -0.000434f, -0.000094f, -0.000124f, -0.000145f, -0.000200f,
	-0.000499f, 0.027255f, 0.160674f, 0.350698f, 0.573575f, 0.763921f, 0.891445f, 0.963946f,
	0.988795f, 0.998974f, 0.986051f, 0.995325f, 0.974335f, 0.991344f, 0.988663f, 0.997139f
};
static const float kPtIllumGreen[kPtSpecN] = {
	0.025168f, 0.039427f, 0.006206f, 0.007112f, 0.000218f, 0.000000f, -0.021623f, 0.015670f,
	0.002802f, 0.324948f, 1.016492f, 1.032948f, 1.032159f, 1.035867f, 1.015124f, 1.033808f,
	1.037137f, 1.036138f, 1.022982f, 0.969103f, -0.005179f, 0.001113f, 0.006668f, 0.000740f,
	0.021592f, 0.005148f, 0.001456f, 0.000164f, -0.006463f, 0.010251f, 0.042387f, 0.021253f
};
static const float kPtIllumBlue[kPtSpecN] = {
	1.057049f, 1.053847f, 1.055049f, 1.053041f, 1.057993f, 1.057844f, 1.058313f, 1.057971f,
	1.056188f, 1.057140f, 1.042580f, 0.326031f, -0.001926f, -0.001296f, -0.001436f, -0.001296f,
	-0.001923f, 0.001262f, -0.001610f, -0.001303f, -0.001767f, -0.001233f, 0.010317f, 0.031285f,
	0.088774f, 0.138736f, 0.155351f, 0.148785f, 0.166243f, 0.169976f, 0.157697f, 0.190691f
};

inline float PtSpecLerpTab( const float *tab, float lambda )
{
	if ( lambda <= kPtRgb2SpectLambda[0] )
		return tab[0];
	if ( lambda >= kPtRgb2SpectLambda[kPtSpecN - 1] )
		return tab[kPtSpecN - 1];
	int i = 0;
	while ( i + 1 < kPtSpecN && kPtRgb2SpectLambda[i + 1] < lambda )
		++i;
	const float l0 = kPtRgb2SpectLambda[i];
	const float l1 = kPtRgb2SpectLambda[i + 1];
	const float t = ( lambda - l0 ) / ( l1 - l0 );
	return tab[i] * ( 1.0f - t ) + tab[i + 1] * t;
}

// Smits combination (PBRT SampledSpectrum::FromRGB). Returns SPD(lambda).
inline float PtSpecFromRgb( float r, float g, float b, float lambda, bool illuminant )
{
	const float *W, *C, *M, *Y, *R, *G, *B;
	if ( illuminant )
	{
		W = kPtIllumWhite; C = kPtIllumCyan; M = kPtIllumMagenta; Y = kPtIllumYellow;
		R = kPtIllumRed; G = kPtIllumGreen; B = kPtIllumBlue;
	}
	else
	{
		r = ( r < 0.0f ) ? 0.0f : ( ( r > 1.0f ) ? 1.0f : r );
		g = ( g < 0.0f ) ? 0.0f : ( ( g > 1.0f ) ? 1.0f : g );
		b = ( b < 0.0f ) ? 0.0f : ( ( b > 1.0f ) ? 1.0f : b );
		W = kPtReflWhite; C = kPtReflCyan; M = kPtReflMagenta; Y = kPtReflYellow;
		R = kPtReflRed; G = kPtReflGreen; B = kPtReflBlue;
	}

	// Illuminants may exceed 1 - scale by max, upsample unit RGB, rescale.
	float scale = 1.0f;
	if ( illuminant )
	{
		scale = r;
		if ( g > scale ) scale = g;
		if ( b > scale ) scale = b;
		if ( scale > 1.0f )
		{
			r /= scale; g /= scale; b /= scale;
		}
		else
			scale = 1.0f;
		r = ( r < 0.0f ) ? 0.0f : r;
		g = ( g < 0.0f ) ? 0.0f : g;
		b = ( b < 0.0f ) ? 0.0f : b;
	}

	float s = 0.0f;
	if ( r <= g && r <= b )
	{
		s += r * PtSpecLerpTab( W, lambda );
		if ( g <= b )
		{
			s += ( g - r ) * PtSpecLerpTab( C, lambda );
			s += ( b - g ) * PtSpecLerpTab( B, lambda );
		}
		else
		{
			s += ( b - r ) * PtSpecLerpTab( C, lambda );
			s += ( g - b ) * PtSpecLerpTab( G, lambda );
		}
	}
	else if ( g <= r && g <= b )
	{
		s += g * PtSpecLerpTab( W, lambda );
		if ( r <= b )
		{
			s += ( r - g ) * PtSpecLerpTab( M, lambda );
			s += ( b - r ) * PtSpecLerpTab( B, lambda );
		}
		else
		{
			s += ( b - g ) * PtSpecLerpTab( M, lambda );
			s += ( r - b ) * PtSpecLerpTab( R, lambda );
		}
	}
	else
	{
		s += b * PtSpecLerpTab( W, lambda );
		if ( r <= g )
		{
			s += ( r - b ) * PtSpecLerpTab( Y, lambda );
			s += ( g - r ) * PtSpecLerpTab( G, lambda );
		}
		else
		{
			s += ( g - b ) * PtSpecLerpTab( Y, lambda );
			s += ( r - g ) * PtSpecLerpTab( R, lambda );
		}
	}
	if ( !illuminant )
		s *= 0.94f;
	else
		s *= scale;
	return ( s > 0.0f ) ? s : 0.0f;
}

inline float PtSpecRefl( const Vector &rgb, float lambda )
{
	return PtSpecFromRgb( rgb.x, rgb.y, rgb.z, lambda, false );
}
inline float PtSpecIllum( const Vector &rgb, float lambda )
{
	return PtSpecFromRgb( rgb.x, rgb.y, rgb.z, lambda, true );
}

// Wyman et al. analytic CIE 1931 XYZ CMFs (nm).
inline void PtCieXyzCmf( float lambda, float &x, float &y, float &z )
{
	const float t1 = ( lambda - 442.0f ) * ( ( lambda < 442.0f ) ? 0.0624f : 0.0374f );
	const float t2 = ( lambda - 599.8f ) * ( ( lambda < 599.8f ) ? 0.0264f : 0.0323f );
	const float t3 = ( lambda - 501.1f ) * ( ( lambda < 501.1f ) ? 0.0490f : 0.0382f );
	x = 0.362f * expf( -0.5f * t1 * t1 ) + 1.056f * expf( -0.5f * t2 * t2 )
		- 0.065f * expf( -0.5f * t3 * t3 );

	const float t4 = ( lambda - 568.8f ) * ( ( lambda < 568.8f ) ? 0.0213f : 0.0247f );
	const float t5 = ( lambda - 530.9f ) * ( ( lambda < 530.9f ) ? 0.0613f : 0.0322f );
	y = 0.821f * expf( -0.5f * t4 * t4 ) + 0.286f * expf( -0.5f * t5 * t5 );

	const float t6 = ( lambda - 437.0f ) * ( ( lambda < 437.0f ) ? 0.0845f : 0.0278f );
	const float t7 = ( lambda - 459.0f ) * ( ( lambda < 459.0f ) ? 0.0385f : 0.0725f );
	z = 1.217f * expf( -0.5f * t6 * t6 ) + 0.681f * expf( -0.5f * t7 * t7 );
}

inline Vector PtXyzToLinearSrgb( float X, float Y, float Z )
{
	Vector rgb;
	rgb.x = 3.240479f * X - 1.537150f * Y - 0.498535f * Z;
	rgb.y = -0.969256f * X + 1.875991f * Y + 0.041556f * Z;
	rgb.z = 0.055648f * X - 0.204043f * Y + 1.057311f * Z;
	if ( rgb.x < 0.0f ) rgb.x = 0.0f;
	if ( rgb.y < 0.0f ) rgb.y = 0.0f;
	if ( rgb.z < 0.0f ) rgb.z = 0.0f;
	return rgb;
}

// Monochromatic radiance L(lambda) -> linear RGB contribution.
// Equal-energy spectrum is pink in Rec.709 (R/G~1.5); white-balance so
// SpecIllum(1,1,1) / flat SPD round-trips to neutral RGB.
inline Vector PtSpectralToRgb( float L, float lambda, float pdfLambda )
{
	float x, y, z;
	PtCieXyzCmf( lambda, x, y, z );
	const float denom = pdfLambda * kPtCieYIntegral;
	const float w = ( denom > 1e-20f ) ? ( L / denom ) : 0.0f;
	Vector rgb = PtXyzToLinearSrgb( x * w, y * w, z * w );
	rgb.x *= 0.607021f;
	rgb.y *= 0.925574f;
	rgb.z *= 0.977565f;
	return rgb;
}

#endif // VRAD_PT_SPECTRAL_H

