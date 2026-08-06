//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Optional OpenCL acceleration for VRAD (-gpu).
//
//=============================================================================//

#ifndef VRAD_GPU_H
#define VRAD_GPU_H
#pragma once

#include "mathlib/vector.h"

class RayTracingEnvironment;

extern bool g_bVRadGPURequested;
extern bool g_bVRadCoarsePatches; // larger chop -> fewer patches
extern float g_flMaxTransferDist; // 0 = off; world units squared cull in VisLeafs

bool VRadGPU_IsActive();
bool VRadGPU_HasScene();
void VRadGPU_SetRequested( bool bRequested );
void VRadGPU_SetMaxTris( int nMaxTris );       // 0 = unlimited (default)
void VRadGPU_SetRayBatchSize( int nBatch );    // default 32768

void VRadGPU_CaptureScene( RayTracingEnvironment &rtEnv );
bool VRadGPU_InitAfterScene();
void VRadGPU_Shutdown();

// Batch occlusion: visible[i]=1 if unblocked. Uses persistent buffers + mutex.
bool VRadGPU_TraceOcclusion( const Vector *pStarts, const Vector *pEnds, unsigned char *pVisible, int nRays );

// Closest hit: pHitT[i], pHitFlags[i] (0 = miss). For sky tests.
bool VRadGPU_TraceClosest( const Vector *pStarts, const Vector *pEnds,
						   float *pHitT, int *pHitFlags, int nRays );

struct VRadGPUTransfer_t
{
	int		patch;
	float	transfer;
};

bool VRadGPU_UploadTransferGraph( const Vector *pReflectivity,
								  const VRadGPUTransfer_t *pTransfers,
								  const int *pOffsets,
								  const int *pPatchEnvIds,
								  int nPatches );

bool VRadGPU_GatherBounce( const Vector *pEmitLight, Vector *pAddLight, int nPatches );

#endif // VRAD_GPU_H
