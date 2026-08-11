//========= Copyright CustomVRAD contributors. ============//
// Compact HLSL spectral helpers for the luxel baker.
// Uses smooth RGB lobe upsampling (not full Smits tables) so DXC stays
// stable - full Smits remains available on CPU in pt_spectral.h.
//=============================================================================//
#include "pt_spectral_hlsl.h"
#include <string>

std::string PtSpectralHLSL()
{
	// Keep this small: Smits 14x32 tables + dual PathLi previously AV'd DXC (0xC0000005).
	return R"HLSL(
static const float SPEC_LMIN = 380.0f;
static const float SPEC_LMAX = 780.0f;
static const float SPEC_CIE_Y_INT = 106.856895f;
static const uint SPEC_WAVES = 4;

// Smooth RGB->SPD lobes (stable, low register pressure). Illuminant may be >1.
float SpecFromRgb( float3 rgb, float lambda, bool illuminant )
{
	float3 c = illuminant ? max( rgb, 0.0f ) : saturate( rgb );
	float scale = 1.0f;
	if ( illuminant )
	{
		scale = max( c.x, max( c.y, c.z ) );
		if ( scale > 1.0f )
			c /= scale;
		else
			scale = 1.0f;
	}
	float3 peak = float3( 610.0f, 550.0f, 450.0f );
	float3 sig = float3( 48.0f, 36.0f, 40.0f );
	float3 d = ( lambda - peak ) / sig;
	float3 g = exp( -0.5f * d * d );
	float s = dot( c, g ) / max( g.x + g.y + g.z, 1e-3f );
	if ( !illuminant )
		s *= 0.94f;
	return max( s * scale, 0.0f );
}

float SpecRefl( float3 rgb, float lambda ) { return SpecFromRgb( rgb, lambda, false ); }
float SpecIllum( float3 rgb, float lambda ) { return SpecFromRgb( rgb, lambda, true ); }

float3 CieXyzCmf( float lambda )
{
	float t1 = ( lambda - 442.0f ) * ( ( lambda < 442.0f ) ? 0.0624f : 0.0374f );
	float t2 = ( lambda - 599.8f ) * ( ( lambda < 599.8f ) ? 0.0264f : 0.0323f );
	float t3 = ( lambda - 501.1f ) * ( ( lambda < 501.1f ) ? 0.0490f : 0.0382f );
	float x = 0.362f * exp( -0.5f * t1 * t1 ) + 1.056f * exp( -0.5f * t2 * t2 ) - 0.065f * exp( -0.5f * t3 * t3 );
	float t4 = ( lambda - 568.8f ) * ( ( lambda < 568.8f ) ? 0.0213f : 0.0247f );
	float t5 = ( lambda - 530.9f ) * ( ( lambda < 530.9f ) ? 0.0613f : 0.0322f );
	float y = 0.821f * exp( -0.5f * t4 * t4 ) + 0.286f * exp( -0.5f * t5 * t5 );
	float t6 = ( lambda - 437.0f ) * ( ( lambda < 437.0f ) ? 0.0845f : 0.0278f );
	float t7 = ( lambda - 459.0f ) * ( ( lambda < 459.0f ) ? 0.0385f : 0.0725f );
	float z = 1.217f * exp( -0.5f * t6 * t6 ) + 0.681f * exp( -0.5f * t7 * t7 );
	return float3( x, y, z );
}

float3 XyzToLinearSrgb( float3 xyz )
{
	float3 rgb;
	rgb.x = 3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z;
	rgb.y = -0.969256f * xyz.x + 1.875991f * xyz.y + 0.041556f * xyz.z;
	rgb.z = 0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z;
	return max( rgb, 0.0f );
}

// Equal-energy / SpecIllum(1,1,1) through CIE->sRGB is pink (R/G~1.5).
// Divide so white lights round-trip to neutral RGB.
static const float3 SPEC_WHITE_BALANCE = float3( 0.607021f, 0.925574f, 0.977565f );

float3 SpectralToRgb( float L, float lambda, float pdfLam )
{
	float3 cmf = CieXyzCmf( lambda );
	float w = L / max( pdfLam * SPEC_CIE_Y_INT, 1e-20f );
	return XyzToLinearSrgb( cmf * w ) * SPEC_WHITE_BALANCE;
}

float3 LightContrib( float3 intensity, float geo, float3 vis, float lambda )
{
	(void)lambda;
	return intensity * geo * vis;
}
)HLSL";
}
