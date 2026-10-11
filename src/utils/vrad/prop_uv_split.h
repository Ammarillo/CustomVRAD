//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Split a lightmapped static prop so each packed model has no overlapping
// UV islands inside the lightmap's 0..1 square. The cut is per source model
// and material, so every lightmapped instance of that model shares it.
//
//=============================================================================//

#ifndef PROP_UV_SPLIT_H
#define PROP_UV_SPLIT_H

#ifdef _WIN32
#pragma once
#endif

#include "utlvector.h"

struct studiohdr_t;
namespace OptimizedModel
{
struct FileHeader_t;
}

struct PropUvTriKey
{
	unsigned short body;
	unsigned short model;
	unsigned short lod;
	unsigned short mesh;
	unsigned short stripGroup;
	unsigned short bin;
	int indexOffset;
};

struct PropUvSplitPlan
{
	int m_SrcDict;
	int m_Material;
	int m_NumBins;
	int m_IslandCount;
	CUtlVector<PropUvTriKey> m_Tris;
	CUtlVector<int> m_EmitDict;
};

bool PropUvBuildPlan( studiohdr_t *pHdr, OptimizedModel::FileHeader_t *pVtx, int materialFilter, PropUvSplitPlan &plan );
bool PropUvTriKept( const PropUvSplitPlan &plan, int bin, int body, int model, int lod, int mesh, int stripGroup, int indexOffset );
void PropUvDegenerateOutsideBin( studiohdr_t *pHdr, OptimizedModel::FileHeader_t *pVtx, const PropUvSplitPlan &plan, int bin );

#endif
