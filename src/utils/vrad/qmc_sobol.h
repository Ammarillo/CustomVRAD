//========= Copyright PathRAD contributors. ============//
//
// Joe-Kuo Sobol (dims 0-2). Same digital net as scipy.stats.qmc.Sobol /
// optuna.samplers.QMCSampler. Gray-code construction; point 0 is the origin
// (sun axis / sphere center). Strongest when the sample count is 2^m.
//
//=============================================================================//

#ifndef QMC_SOBOL_H
#define QMC_SOBOL_H

#include <cmath>

static const unsigned int g_qmcSobolV[3][32] =
{
	{
		0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
		0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
		0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
		0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
		0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u,
		0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
		0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u,
		0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u
	},
	{
		0x80000000u, 0xC0000000u, 0xA0000000u, 0xF0000000u,
		0x88000000u, 0xCC000000u, 0xAA000000u, 0xFF000000u,
		0x80800000u, 0xC0C00000u, 0xA0A00000u, 0xF0F00000u,
		0x88880000u, 0xCCCC0000u, 0xAAAA0000u, 0xFFFF0000u,
		0x80008000u, 0xC000C000u, 0xA000A000u, 0xF000F000u,
		0x88008800u, 0xCC00CC00u, 0xAA00AA00u, 0xFF00FF00u,
		0x80808080u, 0xC0C0C0C0u, 0xA0A0A0A0u, 0xF0F0F0F0u,
		0x88888888u, 0xCCCCCCCCu, 0xAAAAAAAAu, 0xFFFFFFFFu
	},
	{
		0x80000000u, 0xC0000000u, 0x60000000u, 0x90000000u,
		0xE8000000u, 0x5C000000u, 0x8E000000u, 0xC5000000u,
		0x68800000u, 0x9CC00000u, 0xEE600000u, 0x55900000u,
		0x80680000u, 0xC09C0000u, 0x60EE0000u, 0x90550000u,
		0xE8808000u, 0x5CC0C000u, 0x8E606000u, 0xC5909000u,
		0x6868E800u, 0x9C9C5C00u, 0xEEEE8E00u, 0x5555C500u,
		0x8000E880u, 0xC0005CC0u, 0x60008E60u, 0x9000C590u,
		0xE8006868u, 0x5C009C9Cu, 0x8E00EEEEu, 0xC5005555u
	}
};

static inline float QmcSobol( unsigned int index, int dim )
{
	if ( dim < 0 || dim > 2 )
		dim = 0;
	unsigned int g = index ^ ( index >> 1 );
	unsigned int x = 0;
	for ( int b = 0; g; ++b, g >>= 1 )
	{
		if ( g & 1 )
			x ^= g_qmcSobolV[dim][b];
	}
	return (float)( x * 2.3283064365386963e-10 );
}

static inline float QmcFrac( float x )
{
	return x - floorf( x );
}

// Cranley-Patterson rotate: one scramble per luxel, one Sobol index per sample.
static inline float QmcSobolRot( unsigned int index, int dim, float rot )
{
	return QmcFrac( QmcSobol( index, dim ) + rot );
}

#endif // QMC_SOBOL_H
