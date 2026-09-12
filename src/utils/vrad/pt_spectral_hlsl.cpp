//========= Copyright PathRAD contributors. ============//
// Compact HLSL spectral helpers for the GPU luxel baker.
// Same Smits 1999 RGB->SPD mix as CPU pt_spectral.h, 16-knot tables so DXC
// stays stable. CIE XYZ -> linear sRGB, white-balanced so SpecIllum(1,1,1)
// comes back as neutral RGB.
//=============================================================================//
#include "pt_spectral_hlsl.h"
#include <string>

std::string PtSpectralHLSL()
{
	return R"HLSL(
static const float SPEC_LMIN = 380.0f;
static const float SPEC_LMAX = 780.0f;
static const float SPEC_CIE_Y_INT = 106.856895f;
static const uint SPEC_WAVES = 4;

// 16 knots (even PBRT RGB2Spect samples + 720 nm). Values clamped >= 0.
static const float specLam[16] = {
	380.000f, 401.935f, 423.871f, 445.806f, 467.742f, 489.677f, 511.613f, 533.548f,
	555.484f, 577.419f, 599.355f, 621.290f, 643.225f, 665.161f, 687.096f, 720.000f
};

float SpecLerp16( float lambda, float4 a, float4 b, float4 c, float4 d )
{
	float v[16] = {
		a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w,
		c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w
	};
	if ( lambda <= specLam[0] )
		return v[0];
	if ( lambda >= specLam[15] )
		return v[15];
	int i = 0;
	[unroll]
	for ( int k = 0; k < 14; ++k )
	{
		if ( specLam[k + 1] < lambda )
			i = k + 1;
	}
	float t = ( lambda - specLam[i] ) / ( specLam[i + 1] - specLam[i] );
	return v[i] * ( 1.0f - t ) + v[i + 1] * t;
}

float SpecBasis( float lambda, bool illuminant, int which )
{
	// which: 0 W, 1 C, 2 M, 3 Y, 4 R, 5 G, 6 B
	if ( illuminant )
	{
		if ( which == 0 ) return SpecLerp16( lambda, float4( 1.1565f, 1.1566f, 1.1562f, 1.1568f ), float4( 1.1564f, 1.1565f, 1.1568f, 1.1339f ), float4( 1.1291f, 1.0460f, 0.9560f, 0.9150f ), float4( 0.8954f, 0.8822f, 0.8764f, 0.8830f ) );
		if ( which == 1 ) return SpecLerp16( lambda, float4( 1.1334f, 1.1347f, 1.1356f, 1.1362f ), float4( 1.1355f, 1.1360f, 1.1354f, 1.1355f ), float4( 1.1350f, 0.9060f, 0.2954f, 0.0000f ), float4( 0.0000f, 0.0000f, 0.0000f, 0.0000f ) );
		if ( which == 2 ) return SpecLerp16( lambda, float4( 1.0372f, 1.0767f, 1.0795f, 1.0727f ), float4( 1.0824f, 0.9561f, 0.0842f, 0.0000f ), float4( 0.0000f, 0.0145f, 0.5291f, 1.0691f ), float4( 1.0638f, 1.0262f, 0.9833f, 1.0151f ) );
		if ( which == 3 ) return SpecLerp16( lambda, float4( 0.0028f, 0.0000f, 0.0000f, 0.0000f ), float4( 0.0497f, 1.0296f, 1.0368f, 1.0365f ), float4( 1.0366f, 1.0367f, 1.0362f, 1.0043f ), float4( 0.7376f, 0.6053f, 0.5942f, 0.5823f ) );
		if ( which == 4 ) return SpecLerp16( lambda, float4( 0.0547f, 0.0608f, 0.0462f, 0.0244f ), float4( 0.0000f, 0.0004f, 0.0000f, 0.0000f ), float4( 0.0000f, 0.1607f, 0.5736f, 0.8914f ), float4( 0.9888f, 0.9861f, 0.9743f, 0.9971f ) );
		if ( which == 5 ) return SpecLerp16( lambda, float4( 0.0252f, 0.0062f, 0.0002f, 0.0000f ), float4( 0.0028f, 1.0165f, 1.0322f, 1.0151f ), float4( 1.0371f, 1.0230f, 0.0000f, 0.0067f ), float4( 0.0216f, 0.0015f, 0.0000f, 0.0213f ) );
		return SpecLerp16( lambda, float4( 1.0570f, 1.0550f, 1.0580f, 1.0583f ), float4( 1.0562f, 1.0426f, 0.0000f, 0.0000f ), float4( 0.0000f, 0.0000f, 0.0000f, 0.0103f ), float4( 0.0888f, 0.1554f, 0.1662f, 0.1907f ) );
	}
	if ( which == 0 ) return SpecLerp16( lambda, float4( 1.0619f, 1.0614f, 1.0622f, 1.0624f ), float4( 1.0625f, 1.0621f, 1.0610f, 1.0614f ), float4( 1.0626f, 1.0625f, 1.0625f, 1.0625f ), float4( 1.0624f, 1.0612f, 1.0600f, 1.0607f ) );
	if ( which == 1 ) return SpecLerp16( lambda, float4( 1.0415f, 1.0126f, 1.0079f, 1.0443f ), float4( 1.0181f, 1.0529f, 1.0534f, 1.0527f ), float4( 1.0551f, 1.0454f, 0.1804f, 0.0000f ), float4( 0.0000f, 0.0123f, 0.0171f, 0.0253f ) );
	if ( which == 2 ) return SpecLerp16( lambda, float4( 0.9942f, 0.9829f, 1.0199f, 1.0221f ), float4( 1.0098f, 0.6403f, 0.0065f, 0.0000f ), float4( 0.0034f, 0.2220f, 0.9748f, 1.0174f ), float4( 0.9470f, 0.9490f, 0.9960f, 0.8487f ) );
	if ( which == 3 ) return SpecLerp16( lambda, float4( 0.0056f, 0.0000f, 0.0000f, 0.0168f ), float4( 0.2122f, 0.5396f, 0.9221f, 1.0514f ), float4( 1.0511f, 1.0516f, 1.0512f, 1.0514f ), float4( 1.0511f, 1.0509f, 1.0477f, 1.0392f ) );
	if ( which == 4 ) return SpecLerp16( lambda, float4( 0.1658f, 0.1241f, 0.0790f, 0.0000f ), float4( 0.0053f, 0.0000f, 0.0000f, 0.0000f ), float4( 0.0000f, 0.0000f, 0.9883f, 1.0039f ), float4( 0.9993f, 0.9836f, 0.9745f, 0.9871f ) );
	if ( which == 5 ) return SpecLerp16( lambda, float4( 0.0026f, 0.0000f, 0.0000f, 0.0000f ), float4( 0.0655f, 0.7524f, 0.9985f, 0.9994f ), float4( 0.9994f, 0.9602f, 0.2580f, 0.0000f ), float4( 0.0000f, 0.0000f, 0.0000f, 0.0000f ) );
	return SpecLerp16( lambda, float4( 0.9921f, 0.9954f, 0.9918f, 0.9997f ), float4( 0.9850f, 0.5608f, 0.1369f, 0.0000f ), float4( 0.0000f, 0.0038f, 0.0000f, 0.0258f ), float4( 0.0495f, 0.0498f, 0.0305f, 0.0042f ) );
}

// Smits combination (PBRT SampledSpectrum::FromRGB).
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
	float r = c.x, g = c.y, b = c.z;
	float s = 0.0f;
	if ( r <= g && r <= b )
	{
		s += r * SpecBasis( lambda, illuminant, 0 );
		if ( g <= b )
		{
			s += ( g - r ) * SpecBasis( lambda, illuminant, 1 );
			s += ( b - g ) * SpecBasis( lambda, illuminant, 6 );
		}
		else
		{
			s += ( b - r ) * SpecBasis( lambda, illuminant, 1 );
			s += ( g - b ) * SpecBasis( lambda, illuminant, 5 );
		}
	}
	else if ( g <= r && g <= b )
	{
		s += g * SpecBasis( lambda, illuminant, 0 );
		if ( r <= b )
		{
			s += ( r - g ) * SpecBasis( lambda, illuminant, 2 );
			s += ( b - r ) * SpecBasis( lambda, illuminant, 6 );
		}
		else
		{
			s += ( b - g ) * SpecBasis( lambda, illuminant, 2 );
			s += ( r - b ) * SpecBasis( lambda, illuminant, 4 );
		}
	}
	else
	{
		s += b * SpecBasis( lambda, illuminant, 0 );
		if ( r <= g )
		{
			s += ( r - b ) * SpecBasis( lambda, illuminant, 3 );
			s += ( g - r ) * SpecBasis( lambda, illuminant, 5 );
		}
		else
		{
			s += ( g - b ) * SpecBasis( lambda, illuminant, 3 );
			s += ( r - g ) * SpecBasis( lambda, illuminant, 4 );
		}
	}
	if ( !illuminant )
		s *= 0.94f;
	else
		s *= scale;
	return max( s, 0.0f );
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

// SpecIllum(1,1,1) through 4-lambda CIE->sRGB (16-knot Smits).
static const float3 SPEC_WHITE_BALANCE = float3( 0.628752f, 0.828451f, 0.845463f );

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
