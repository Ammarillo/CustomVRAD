#pragma once
#include "shaderlib/cshader.h"
class cvrad_fog_ps30_Static_Index
{
public:
	cvrad_fog_ps30_Static_Index( IShaderShadow* pShaderShadow, IMaterialVar** params )
	{
	}

	int GetIndex() const
	{
		return 0;
	}
};

#define shaderStaticTest_cvrad_fog_ps30 1


class cvrad_fog_ps30_Dynamic_Index
{
public:
	cvrad_fog_ps30_Dynamic_Index( IShaderDynamicAPI* pShaderAPI )
	{
	}

	int GetIndex() const
	{
		return 0;
	}
};

#define shaderDynamicTest_cvrad_fog_ps30 1

