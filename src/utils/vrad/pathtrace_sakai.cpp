//========= Copyright CustomVRAD contributors. ============//
// Sakai et al. SIGGRAPH Asia 2024 - Welch-gated filter for luxels.
// Small multi-pass kernels (no large windows / dilated A-Trous) to avoid
// lightmap "block" plateaus. Variance is spatially stabilized and a mild
// flat-region evening pass kills mottling in soft shadows.
//=============================================================================//

#include "pathtrace_sakai.h"
#include "pathtrace_dxr.h"

#include <cmath>
#include <vector>

#include "tier0/dbg.h"

static bool g_sakaiReady = false;

bool PathTraceSakai_Init()
{
	g_sakaiReady = true;
	Msg( "[PathTrace-Sakai] Ready (Welch multi-pass 5-tap + evening, radius=%d)\n",
		 g_nPathTraceDenoiseRadius );
	return true;
}

void PathTraceSakai_Shutdown()
{
	g_sakaiReady = false;
}

bool PathTraceSakai_IsReady()
{
	return g_sakaiReady;
}

static inline float SakaiLuma( float r, float g, float b )
{
	return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static void SakaiBlur3( const float *src, float *dst, int width, int height )
{
	static const float k[3] = { 0.25f, 0.5f, 0.25f };
	for ( int y = 0; y < height; ++y )
	{
		for ( int x = 0; x < width; ++x )
		{
			float s = 0.0f, wsum = 0.0f;
			for ( int ky = 0; ky < 3; ++ky )
			{
				int ny = y + ky - 1;
				if ( ny < 0 || ny >= height )
					continue;
				for ( int kx = 0; kx < 3; ++kx )
				{
					int nx = x + kx - 1;
					if ( nx < 0 || nx >= width )
						continue;
					const float w = k[kx] * k[ky];
					s += src[ny * width + nx] * w;
					wsum += w;
				}
			}
			dst[y * width + x] = ( wsum > 1e-8f ) ? ( s / wsum ) : src[y * width + x];
		}
	}
}

static void SakaiEnsureVars( const float *rgb, float *luma, float *varBuf, int width, int height,
							 const float *varMean )
{
	const int n = width * height;
	for ( int i = 0; i < n; ++i )
		luma[i] = SakaiLuma( rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2] );

	std::vector<float> raw( (size_t)n );
	if ( varMean )
	{
		for ( int i = 0; i < n; ++i )
		{
			float v = varMean[i];
			if ( v < 1e-10f )
				v = 1e-10f;
			raw[i] = v;
		}
	}
	else
	{
		// Fallback: mild noise floor (do NOT use full spatial variance - that no-ops Welch).
		for ( int i = 0; i < n; ++i )
		{
			float L = luma[i];
			raw[i] = max( 1e-8f, L * L * 1e-4f + 1e-6f );
		}
	}

	// Spatially stabilize Var(mean) so Welch accept/reject maps aren't mottled.
	SakaiBlur3( raw.data(), varBuf, width, height );
	SakaiBlur3( varBuf, raw.data(), width, height );
	for ( int i = 0; i < n; ++i )
		varBuf[i] = max( 1e-10f, raw[i] );
}

// One 5x5 Welch-gated bilateral pass (step=1 only - dilated steps look blocky on luxels).
static void SakaiPass5( float *rgb, const float *luma, const float *varBuf, int width, int height,
						float tCrit )
{
	const int n = width * height;
	std::vector<float> out( (size_t)n * 3u );
	static const float k[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };

	for ( int y = 0; y < height; ++y )
	{
		for ( int x = 0; x < width; ++x )
		{
			const int ci = y * width + x;
			const float lc = luma[ci];
			const float vc = varBuf[ci];
			float sr = 0.0f, sg = 0.0f, sb = 0.0f, wsum = 0.0f;

			for ( int ky = 0; ky < 5; ++ky )
			{
				int ny = y + ( ky - 2 );
				if ( ny < 0 || ny >= height )
					continue;
				for ( int kx = 0; kx < 5; ++kx )
				{
					int nx = x + ( kx - 2 );
					if ( nx < 0 || nx >= width )
						continue;
					const int ni = ny * width + nx;
					const float wSpat = k[kx] * k[ky];
					const float ln = luma[ni];
					const float vn = varBuf[ni];
					// Relative floor: dark flats share tiny absolute noise that would
					// otherwise look "significant" vs a tiny Var(mean).
					const float Lref = max( max( lc, ln ), 1e-4f );
					float se2 = vc + vn + ( 0.04f * Lref ) * ( 0.04f * Lref ) + 1e-8f;
					const float t = fabsf( lc - ln ) / sqrtf( se2 );
					float wWelch = 1.0f;
					if ( t > tCrit )
					{
						const float u = ( t - tCrit ) / max( tCrit, 1e-3f );
						wWelch = expf( -u * u * 1.5f ); // softer than before
					}
					const float w = wSpat * wWelch;
					sr += rgb[ni * 3 + 0] * w;
					sg += rgb[ni * 3 + 1] * w;
					sb += rgb[ni * 3 + 2] * w;
					wsum += w;
				}
			}

			if ( wsum > 1e-8f )
			{
				out[ci * 3 + 0] = sr / wsum;
				out[ci * 3 + 1] = sg / wsum;
				out[ci * 3 + 2] = sb / wsum;
			}
			else
			{
				out[ci * 3 + 0] = rgb[ci * 3 + 0];
				out[ci * 3 + 1] = rgb[ci * 3 + 1];
				out[ci * 3 + 2] = rgb[ci * 3 + 2];
			}
		}
	}

	for ( int i = 0; i < n * 3; ++i )
		rgb[i] = out[i];
}

// Mild 3x3 isotropic blend in low-gradient regions only - kills shadow mottling
// without soft-blurring hard shadow edges (stairs, wall contacts).
static void SakaiEveningPass( float *rgb, const float *luma, int width, int height )
{
	const int n = width * height;
	std::vector<float> out( (size_t)n * 3u );
	static const float k[3] = { 0.25f, 0.5f, 0.25f };

	for ( int y = 0; y < height; ++y )
	{
		for ( int x = 0; x < width; ++x )
		{
			const int ci = y * width + x;
			const float lc = luma[ci];
			float sr = 0.0f, sg = 0.0f, sb = 0.0f, wsum = 0.0f;
			float grad = 0.0f;
			int nNbr = 0;

			for ( int ky = 0; ky < 3; ++ky )
			{
				int ny = y + ky - 1;
				if ( ny < 0 || ny >= height )
					continue;
				for ( int kx = 0; kx < 3; ++kx )
				{
					int nx = x + kx - 1;
					if ( nx < 0 || nx >= width )
						continue;
					const int ni = ny * width + nx;
					const float w = k[kx] * k[ky];
					sr += rgb[ni * 3 + 0] * w;
					sg += rgb[ni * 3 + 1] * w;
					sb += rgb[ni * 3 + 2] * w;
					wsum += w;
					if ( ni != ci )
					{
						grad += fabsf( luma[ni] - lc );
						++nNbr;
					}
				}
			}

			if ( wsum < 1e-8f )
			{
				out[ci * 3 + 0] = rgb[ci * 3 + 0];
				out[ci * 3 + 1] = rgb[ci * 3 + 1];
				out[ci * 3 + 2] = rgb[ci * 3 + 2];
				continue;
			}

			const float meanGrad = ( nNbr > 0 ) ? ( grad / (float)nNbr ) : 0.0f;
			const float Lref = max( lc, 1e-4f );
			// flatAmt -> 1 in soft shadows / gradients, -> 0 on hard edges.
			const float flatAmt = 1.0f - min( 1.0f, meanGrad / ( 0.08f * Lref + 1e-4f ) );
			const float a = flatAmt * flatAmt; // bias toward preserving edges

			out[ci * 3 + 0] = rgb[ci * 3 + 0] * ( 1.0f - a ) + ( sr / wsum ) * a;
			out[ci * 3 + 1] = rgb[ci * 3 + 1] * ( 1.0f - a ) + ( sg / wsum ) * a;
			out[ci * 3 + 2] = rgb[ci * 3 + 2] * ( 1.0f - a ) + ( sb / wsum ) * a;
		}
	}

	for ( int i = 0; i < n * 3; ++i )
		rgb[i] = out[i];
}

bool PathTraceSakai_DenoiseRGB( float *rgb, int width, int height, const float *varMean )
{
	if ( !rgb || width < 1 || height < 1 || !g_sakaiReady )
		return false;

	const int n = width * height;
	std::vector<float> luma( (size_t)n );
	std::vector<float> varBuf( (size_t)n );
	SakaiEnsureVars( rgb, luma.data(), varBuf.data(), width, height, varMean );

	int radius = g_nPathTraceDenoiseRadius;
	if ( radius < 1 )
		radius = 1;
	if ( radius > 4 )
		radius = 4;

	// More passes = cleaner; each pass stays 5x5 (no big windows -> no block plateaus).
	const int nPasses = 1 + radius; // 2..5
	// Slightly looser than 1.75: more neighbor share in noisy soft shadows.
	const float tCrit = 1.35f;

	for ( int pass = 0; pass < nPasses; ++pass )
	{
		SakaiPass5( rgb, luma.data(), varBuf.data(), width, height, tCrit );
		for ( int i = 0; i < n; ++i )
			luma[i] = SakaiLuma( rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2] );
	}

	// Two light evening passes - flattens mottled shadow interiors.
	SakaiEveningPass( rgb, luma.data(), width, height );
	for ( int i = 0; i < n; ++i )
		luma[i] = SakaiLuma( rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2] );
	SakaiEveningPass( rgb, luma.data(), width, height );

	return true;
}
