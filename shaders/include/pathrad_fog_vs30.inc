#pragma once
#include "shaderlib/cshader.h"
class pathrad_fog_vs30_Static_Index
{
public:
	pathrad_fog_vs30_Static_Index( IShaderShadow* pShaderShadow, IMaterialVar** params )
	{
	}

	int GetIndex() const
	{
		return 0;
	}
};

#define shaderStaticTest_pathrad_fog_vs30 1


class pathrad_fog_vs30_Dynamic_Index
{
public:
	pathrad_fog_vs30_Dynamic_Index( IShaderDynamicAPI* pShaderAPI )
	{
	}

	int GetIndex() const
	{
		return 0;
	}
};

#define shaderDynamicTest_pathrad_fog_vs30 1

