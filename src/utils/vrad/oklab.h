//========= Copyright CustomVRAD contributors. ============//
// Oklab - Bjoern Ottosson (2020), public domain / MIT.
// https://bottosson.github.io/posts/oklab/
//
// Perceptual color ops for CustomVRAD (filters, bounce chroma, env tint).
// Energy transport (albedoxlight, NEE) stays linear RGB.
//=============================================================================//
#ifndef VRAD_OKLAB_H
#define VRAD_OKLAB_H
#pragma once

#include "mathlib/vector.h"
#include <math.h>

struct Oklab_t
{
	float L; // perceptual lightness (D65 white -> 1; HDR lights may be >1)
	float a; // green (-) ... red (+)
	float b; // blue (-) ... yellow (+)
};

inline Oklab_t Oklab_FromLinearSRGB( float r, float g, float b )
{
	r = ( r > 0.0f ) ? r : 0.0f;
	g = ( g > 0.0f ) ? g : 0.0f;
	b = ( b > 0.0f ) ? b : 0.0f;

	const float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
	const float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
	const float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

	const float l_ = cbrtf( l );
	const float m_ = cbrtf( m );
	const float s_ = cbrtf( s );

	Oklab_t o;
	o.L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
	o.a = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
	o.b = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
	return o;
}

inline Oklab_t Oklab_FromLinearSRGB( const Vector &c )
{
	return Oklab_FromLinearSRGB( c.x, c.y, c.z );
}

inline Vector Oklab_ToLinearSRGB( const Oklab_t &c )
{
	const float l_ = c.L + 0.3963377774f * c.a + 0.2158037573f * c.b;
	const float m_ = c.L - 0.1055613458f * c.a - 0.0638541728f * c.b;
	const float s_ = c.L - 0.0894841775f * c.a - 1.2914855480f * c.b;

	const float l = l_ * l_ * l_;
	const float m = m_ * m_ * m_;
	const float s = s_ * s_ * s_;

	Vector rgb;
	rgb.x = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
	rgb.y = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
	rgb.z = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
	return rgb;
}

inline Vector Oklab_Clamp01( const Vector &rgb )
{
	Vector o = rgb;
	o.x = ( o.x < 0.0f ) ? 0.0f : ( ( o.x > 1.0f ) ? 1.0f : o.x );
	o.y = ( o.y < 0.0f ) ? 0.0f : ( ( o.y > 1.0f ) ? 1.0f : o.y );
	o.z = ( o.z < 0.0f ) ? 0.0f : ( ( o.z > 1.0f ) ? 1.0f : o.z );
	return o;
}

inline Vector Oklab_ClampNonNeg( const Vector &rgb )
{
	Vector o = rgb;
	if ( o.x < 0.0f ) o.x = 0.0f;
	if ( o.y < 0.0f ) o.y = 0.0f;
	if ( o.z < 0.0f ) o.z = 0.0f;
	return o;
}

inline Vector Oklab_ToLinearSRGBClamped( const Oklab_t &c )
{
	return Oklab_Clamp01( Oklab_ToLinearSRGB( c ) );
}

// CSS / Photoshop-style perceptual mix.
inline Vector Oklab_LerpLinearSRGB( const Vector &c0, const Vector &c1, float t )
{
	if ( t <= 0.0f )
		return c0;
	if ( t >= 1.0f )
		return c1;
	const Oklab_t a = Oklab_FromLinearSRGB( c0 );
	const Oklab_t b = Oklab_FromLinearSRGB( c1 );
	Oklab_t o;
	o.L = a.L + ( b.L - a.L ) * t;
	o.a = a.a + ( b.a - a.a ) * t;
	o.b = a.b + ( b.b - a.b ) * t;
	return Oklab_ToLinearSRGBClamped( o );
}

// Stacked colored filters (gels): L*=L, a+=a, b+=b.
inline Vector Oklab_StackFilterLinearSRGB( const Vector &accumT, const Vector &filterT )
{
	const Oklab_t t = Oklab_FromLinearSRGB( accumT );
	const Oklab_t f = Oklab_FromLinearSRGB( filterT );
	Oklab_t o;
	o.L = t.L * f.L;
	o.a = t.a + f.a;
	o.b = t.b + f.b;
	if ( o.L < 1e-4f )
		o.L = 1e-4f;
	return Oklab_ToLinearSRGBClamped( o );
}

// -bounce_chroma: scale opponent channels, keep perceptual lightness.
inline Vector Oklab_ScaleChroma( const Vector &c, float sat )
{
	if ( sat <= 1.0001f )
		return c;
	Oklab_t o = Oklab_FromLinearSRGB( c );
	o.a *= sat;
	o.b *= sat;
	return Oklab_ClampNonNeg( Oklab_ToLinearSRGB( o ) );
}

// light_env_vol inbound tint: keep light's Oklab L (x lumScale), take a,b from tint.
inline Vector Oklab_ApplyTintPreserveL( const Vector &light, const Vector &tint, float lumScale )
{
	Oklab_t ol = Oklab_FromLinearSRGB( light );
	const Oklab_t ot = Oklab_FromLinearSRGB( tint );
	ol.L *= ( lumScale > 0.0f ) ? lumScale : 0.0f;
	ol.a = ot.a;
	ol.b = ot.b;
	return Oklab_ClampNonNeg( Oklab_ToLinearSRGB( ol ) );
}

#endif // VRAD_OKLAB_H
