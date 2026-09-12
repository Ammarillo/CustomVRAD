//========= Copyright Valve Corporation, All rights reserved. ============//
//
// OpenCL for classic bounce (-gpu): occlusion / closest hit + gather.
// Small ray batches (avoids TDR), size checks, chunked uploads, fall back to
// CPU if OpenCL errors.
//
//=============================================================================//

#include "vrad.h"
#include "vrad_gpu.h"
#include "envvolume.h"
#include "raytrace.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <vector>
#include <cstring>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <atomic>

bool g_bVRadGPURequested = false;
bool g_bVRadCoarsePatches = false;
float g_flMaxTransferDist = 0.0f;

namespace
{
std::mutex g_clMutex;
bool g_bActive = false;
bool g_bSceneUploaded = false;
bool g_bCaptureDone = false;
std::atomic<bool> g_bGPUFailed{ false };

cl_platform_id g_platform = nullptr;
cl_device_id g_device = nullptr;
cl_context g_context = nullptr;
cl_command_queue g_queue = nullptr;
cl_program g_program = nullptr;
cl_kernel g_kOcclusion = nullptr;
cl_kernel g_kClosest = nullptr;
cl_kernel g_kGather = nullptr;

cl_mem g_dTris = nullptr;
cl_mem g_dBVH = nullptr;
cl_mem g_dPrims = nullptr;
int g_nTris = 0;
int g_nBVH = 0;
cl_ulong g_deviceMaxAlloc = 0;
cl_ulong g_deviceGlobalMem = 0;

// Per-thread trace contexts: own queue + kernels + ray buffers, so all CPU
// worker threads can dispatch to the GPU concurrently (no global queue lock).
struct TraceCtx
{
	cl_command_queue queue;
	cl_kernel kOcclusion;
	cl_kernel kClosest;
	cl_mem dRays;
	cl_mem dVis;
	cl_mem dHitT;
	cl_mem dHitFlags;
	int capacity;
};
enum { kMaxTraceCtx = 64 };
std::vector<TraceCtx *> g_ctxPool;	// all created (guarded by g_clMutex)
std::vector<TraceCtx *> g_ctxFree;	// available for checkout

cl_mem g_dEmit = nullptr;
cl_mem g_dRefl = nullptr;
cl_mem g_dTrans = nullptr;
cl_mem g_dOff = nullptr;
cl_mem g_dAdd = nullptr;
cl_mem g_dPatchEnv = nullptr;
cl_mem g_dVolTint = nullptr;
cl_mem g_dVolUseColor = nullptr;
cl_mem g_dVolUseBright = nullptr;
int g_nPatchesCached = 0;
int g_nTransCached = 0;
float g_flDefaultBounceIntensity = 1.0f;

// Keep each OpenCL dispatch short to avoid Windows TDR / driver hangs.
enum { kMinGPURays = 32 };
enum { kDefaultRayBatch = 32768 };
enum { kUploadChunkBytes = 16 * 1024 * 1024 };
// Soft cap for GPU BVH (0 = unlimited). Override with -gpu_maxtris.
// Default unlimited - large scenes are handled via small ray batches instead.
enum { kDefaultMaxTris = 0 };

int g_nMaxRayBatch = kDefaultRayBatch;
int g_nMaxTris = kDefaultMaxTris;

struct Float4Host { float x, y, z, w; };

#pragma pack(push, 4)
struct Float3 { float x, y, z, w; };
struct TriGPU { Float3 a, b, c; int flags; int pad0, pad1, pad2; };
struct BVHNodeGPU { Float3 bmin, bmax; int left, right, firstPrim, primCount; };
struct RayGPU { Float3 origin, dir; };
#pragma pack(pop)

struct HostTri
{
	Vector a, b, c;
	int flags;
};

std::vector<HostTri> g_hostTris;
std::vector<BVHNodeGPU> g_hostBVH;
std::vector<int> g_primIndices;

const char *g_kernelSource = R"CLC(
typedef struct { float x,y,z,w; } Float3;
typedef struct { Float3 a,b,c; int flags; int pad0,pad1,pad2; } TriGPU;
typedef struct { Float3 bmin,bmax; int left,right,firstPrim,primCount; } BVHNodeGPU;
typedef struct { Float3 origin, dir; } RayGPU;
typedef struct { int patch; float transfer; } TransferGPU;
typedef struct { float x,y,z; } Vec3;
typedef struct { float x,y,z,w; } Tint4;

static Float3 make3(float x,float y,float z){ Float3 v; v.x=x; v.y=y; v.z=z; v.w=0; return v; }
static Float3 sub3(Float3 a,Float3 b){ return make3(a.x-b.x,a.y-b.y,a.z-b.z); }
static Float3 add3(Float3 a,Float3 b){ return make3(a.x+b.x,a.y+b.y,a.z+b.z); }
static Float3 mul3(Float3 a,float s){ return make3(a.x*s,a.y*s,a.z*s); }
static float dot3(Float3 a,Float3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static Float3 cross3(Float3 a,Float3 b){
  return make3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
static float length3(Float3 a){ return sqrt(dot3(a,a)); }

static int intersectAABB(Float3 o, Float3 d, float tmax, Float3 bmin, Float3 bmax)
{
  float tmin = 0.0f;
  for (int axis=0; axis<3; ++axis) {
    float orig = (axis==0)?o.x:((axis==1)?o.y:o.z);
    float dir  = (axis==0)?d.x:((axis==1)?d.y:d.z);
    float mn   = (axis==0)?bmin.x:((axis==1)?bmin.y:bmin.z);
    float mx   = (axis==0)?bmax.x:((axis==1)?bmax.y:bmax.z);
    if (fabs(dir) < 1e-20f) {
      if (orig < mn || orig > mx) return 0;
    } else {
      float inv = 1.0f / dir;
      float t0 = (mn - orig) * inv;
      float t1 = (mx - orig) * inv;
      if (t0 > t1) { float tmp=t0; t0=t1; t1=tmp; }
      if (t0 > tmin) tmin = t0;
      if (t1 < tmax) tmax = t1;
      if (tmin > tmax) return 0;
    }
  }
  return 1;
}

static float intersectTri(Float3 o, Float3 d, TriGPU t)
{
  Float3 e1 = sub3(t.b, t.a);
  Float3 e2 = sub3(t.c, t.a);
  Float3 pvec = cross3(d, e2);
  float det = dot3(e1, pvec);
  if (fabs(det) < 1e-12f) return -1.0f;
  float invDet = 1.0f / det;
  Float3 tvec = sub3(o, t.a);
  float u = dot3(tvec, pvec) * invDet;
  if (u < 0.0f || u > 1.0f) return -1.0f;
  Float3 qvec = cross3(tvec, e1);
  float v = dot3(d, qvec) * invDet;
  if (v < 0.0f || u + v > 1.0f) return -1.0f;
  float tt = dot3(e2, qvec) * invDet;
  return tt;
}

// Oklab (Ottosson) - inbound bounce tint only in this kernel.
static Float3 oklabFromLin(float r, float g, float b)
{
  if (r < 0.0f) r = 0.0f;
  if (g < 0.0f) g = 0.0f;
  if (b < 0.0f) b = 0.0f;
  float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
  float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
  float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;
  float l_ = cbrt(l);
  float m_ = cbrt(m);
  float s_ = cbrt(s);
  return make3(
    0.2104542553f*l_ + 0.7936177850f*m_ - 0.0040720468f*s_,
    1.9779984951f*l_ - 2.4285922050f*m_ + 0.4505937099f*s_,
    0.0259040371f*l_ + 0.7827717662f*m_ - 0.8086757660f*s_);
}
static Float3 oklabToLin(Float3 lab)
{
  float l_ = lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z;
  float m_ = lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z;
  float s_ = lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z;
  float l = l_*l_*l_;
  float m = m_*m_*m_;
  float s = s_*s_*s_;
  Float3 rgb = make3(
    +4.0767416621f*l - 3.3077115913f*m + 0.2309699292f*s,
    -1.2684380046f*l + 2.6097574011f*m - 0.3413193965f*s,
    -0.0041960863f*l - 0.7034186147f*m + 1.7076147010f*s);
  if (rgb.x < 0.0f) rgb.x = 0.0f;
  if (rgb.y < 0.0f) rgb.y = 0.0f;
  if (rgb.z < 0.0f) rgb.z = 0.0f;
  return rgb;
}
static void oklabApplyTintPreserveL(float *vx, float *vy, float *vz, Tint4 tint, float lumScale)
{
  Float3 ol = oklabFromLin(*vx, *vy, *vz);
  Float3 ot = oklabFromLin(tint.x, tint.y, tint.z);
  ol.x *= (lumScale > 0.0f) ? lumScale : 0.0f;
  ol.y = ot.y;
  ol.z = ot.z;
  Float3 rgb = oklabToLin(ol);
  *vx = rgb.x; *vy = rgb.y; *vz = rgb.z;
}

__kernel void kOcclusion(
  __global const RayGPU* rays,
  __global uchar* visible,
  __global const TriGPU* tris,
  __global const BVHNodeGPU* bvh,
  __global const int* prims,
  const int nRays)
{
  int i = get_global_id(0);
  if (i >= nRays) return;

  RayGPU r = rays[i];
  float tmax = length3(r.dir);
  if (tmax < 1e-8f) { visible[i] = 1; return; }
  Float3 d = mul3(r.dir, 1.0f / tmax);
  Float3 o = r.origin;

  int stack[64];
  int sp = 0;
  stack[sp++] = 0;
  int hit = 0;

  while (sp > 0) {
    int ni = stack[--sp];
    BVHNodeGPU node = bvh[ni];
    if (!intersectAABB(o, d, tmax, node.bmin, node.bmax))
      continue;
    if (node.left < 0) {
      for (int p = 0; p < node.primCount; ++p) {
        int ti = prims[node.firstPrim + p];
        float t = intersectTri(o, d, tris[ti]);
        if (t > 1e-4f && t < tmax) { hit = 1; sp = 0; break; }
      }
    } else {
      if (sp < 62) {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }
  }
  visible[i] = hit ? 0 : 1;
}

__kernel void kClosest(
  __global const RayGPU* rays,
  __global float* hitT,
  __global int* hitFlags,
  __global const TriGPU* tris,
  __global const BVHNodeGPU* bvh,
  __global const int* prims,
  const int nRays)
{
  int i = get_global_id(0);
  if (i >= nRays) return;

  RayGPU r = rays[i];
  float tmax = length3(r.dir);
  hitT[i] = tmax;
  hitFlags[i] = 0;
  if (tmax < 1e-8f) return;
  Float3 d = mul3(r.dir, 1.0f / tmax);
  Float3 o = r.origin;

  int stack[64];
  int sp = 0;
  stack[sp++] = 0;
  float bestT = tmax;
  int bestFlags = 0;

  while (sp > 0) {
    int ni = stack[--sp];
    BVHNodeGPU node = bvh[ni];
    if (!intersectAABB(o, d, bestT, node.bmin, node.bmax))
      continue;
    if (node.left < 0) {
      for (int p = 0; p < node.primCount; ++p) {
        int ti = prims[node.firstPrim + p];
        float t = intersectTri(o, d, tris[ti]);
        if (t > 1e-4f && t < bestT) {
          bestT = t;
          bestFlags = tris[ti].flags;
        }
      }
    } else {
      if (sp < 62) {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }
  }
  hitT[i] = bestT;
  hitFlags[i] = (bestT < tmax - 1e-4f) ? bestFlags : 0;
}

__kernel void kGatherBounce(
  __global const Vec3* emitlight,
  __global const Vec3* reflectivity,
  __global const TransferGPU* transfers,
  __global const int* offsets,
  __global const int* patchEnv,
  __global const Tint4* volTint,
  __global const int* volUseColor,
  __global const int* volUseBright,
  __global Vec3* addlight,
  const int nPatches,
  const float defaultIntensity)
{
  int i = get_global_id(0);
  if (i >= nPatches) return;
  int begin = offsets[i];
  int end = offsets[i+1];
  int recvEnv = patchEnv[i];
  float sx=0.0f, sy=0.0f, sz=0.0f;
  for (int k = begin; k < end; ++k) {
    TransferGPU tr = transfers[k];
    Vec3 e = emitlight[tr.patch];
    Vec3 r = reflectivity[tr.patch];
    float s = tr.transfer;
    float vx = e.x * r.x * s;
    float vy = e.y * r.y * s;
    float vz = e.z * r.z * s;

    int emitEnv = patchEnv[tr.patch];
    if (recvEnv > 0 && volUseColor[recvEnv] != 0 && emitEnv != recvEnv) {
      float lum = vx + vy + vz;
      if (lum >= 1e-10f) {
        Tint4 tint = volTint[recvEnv];
        float scale = 1.0f;
        if (volUseBright[recvEnv] != 0)
          scale = tint.w / defaultIntensity;
        oklabApplyTintPreserveL(&vx, &vy, &vz, tint, scale);
      }
    }
    sx += vx; sy += vy; sz += vz;
  }
  Vec3 o;
  o.x = sx; o.y = sy; o.z = sz;
  addlight[i] = o;
}
)CLC";

static Float3 ToF3( const Vector &v )
{
	Float3 f;
	f.x = v.x;
	f.y = v.y;
	f.z = v.z;
	f.w = 0.0f;
	return f;
}

static bool CheckCL( cl_int err, const char *what )
{
	if ( err == CL_SUCCESS )
		return true;
	Warning( "[VRAD-GPU] %s failed (cl_int %d)\n", what, (int)err );
	return false;
}

static void ReleaseSceneBuffers();
static void ReleaseGatherBuffers();
static void ShutdownUnlocked();

// Only flips the flag - resources are freed in VRadGPU_Shutdown so other
// threads still inside a trace call never touch released handles.
static void MarkGPUFailed( const char *why )
{
	bool expected = false;
	if ( g_bGPUFailed.compare_exchange_strong( expected, true ) )
		Warning( "[VRAD-GPU] Disabling GPU for this run: %s (CPU fallback)\n", why );
}

static void BoundsOfTris( const int *idx, int count, Vector &bmin, Vector &bmax )
{
	bmin.Init( 1e30f, 1e30f, 1e30f );
	bmax.Init( -1e30f, -1e30f, -1e30f );
	for ( int i = 0; i < count; ++i )
	{
		const HostTri &t = g_hostTris[idx[i]];
		for ( int v = 0; v < 3; ++v )
		{
			const Vector *p = ( v == 0 ) ? &t.a : ( ( v == 1 ) ? &t.b : &t.c );
			bmin = bmin.Min( *p );
			bmax = bmax.Max( *p );
		}
	}
}

static int BuildBVHRecursive( std::vector<int> &indices, int begin, int end, int depth )
{
	BVHNodeGPU node;
	memset( &node, 0, sizeof( node ) );

	Vector bmin, bmax;
	BoundsOfTris( &indices[begin], end - begin, bmin, bmax );
	node.bmin = ToF3( bmin );
	node.bmax = ToF3( bmax );

	const int count = end - begin;
	const int nodeIndex = (int)g_hostBVH.size();
	g_hostBVH.push_back( node );

	// Slightly larger leaves -> shallower tree, less stack pressure in kernels.
	if ( count <= 8 || depth > 40 )
	{
		BVHNodeGPU leaf = g_hostBVH[nodeIndex];
		leaf.left = -1;
		leaf.right = -1;
		leaf.firstPrim = (int)g_primIndices.size();
		leaf.primCount = count;
		for ( int i = begin; i < end; ++i )
			g_primIndices.push_back( indices[i] );
		g_hostBVH[nodeIndex] = leaf;
		return nodeIndex;
	}

	Vector ext = bmax - bmin;
	int axis = 0;
	if ( ext.y > ext.x )
		axis = 1;
	if ( ext.z > ( ( axis == 0 ) ? ext.x : ext.y ) )
		axis = 2;
	const float mid = 0.5f * ( ( &bmin.x )[axis] + ( &bmax.x )[axis] );

	int pivot = begin;
	for ( int i = begin; i < end; ++i )
	{
		const HostTri &t = g_hostTris[indices[i]];
		Vector c = ( t.a + t.b + t.c ) * ( 1.0f / 3.0f );
		if ( ( &c.x )[axis] < mid )
		{
			int tmp = indices[pivot];
			indices[pivot] = indices[i];
			indices[i] = tmp;
			++pivot;
		}
	}
	if ( pivot == begin || pivot == end )
		pivot = begin + count / 2;

	const int left = BuildBVHRecursive( indices, begin, pivot, depth + 1 );
	const int right = BuildBVHRecursive( indices, pivot, end, depth + 1 );

	BVHNodeGPU inner = g_hostBVH[nodeIndex];
	inner.left = left;
	inner.right = right;
	inner.firstPrim = 0;
	inner.primCount = 0;
	g_hostBVH[nodeIndex] = inner;
	return nodeIndex;
}

static void BuildBVH()
{
	g_hostBVH.clear();
	g_primIndices.clear();
	if ( g_hostTris.empty() )
		return;

	std::vector<int> indices( g_hostTris.size() );
	for ( int i = 0; i < (int)g_hostTris.size(); ++i )
		indices[i] = i;
	BuildBVHRecursive( indices, 0, (int)indices.size(), 0 );
}

static void ReleaseSceneBuffers()
{
	if ( g_dTris ) { clReleaseMemObject( g_dTris ); g_dTris = nullptr; }
	if ( g_dBVH ) { clReleaseMemObject( g_dBVH ); g_dBVH = nullptr; }
	if ( g_dPrims ) { clReleaseMemObject( g_dPrims ); g_dPrims = nullptr; }
	g_nTris = 0;
	g_nBVH = 0;
	g_bSceneUploaded = false;
}

static void DestroyTraceCtx( TraceCtx *ctx )
{
	if ( !ctx )
		return;
	if ( ctx->dRays ) clReleaseMemObject( ctx->dRays );
	if ( ctx->dVis ) clReleaseMemObject( ctx->dVis );
	if ( ctx->dHitT ) clReleaseMemObject( ctx->dHitT );
	if ( ctx->dHitFlags ) clReleaseMemObject( ctx->dHitFlags );
	if ( ctx->kOcclusion ) clReleaseKernel( ctx->kOcclusion );
	if ( ctx->kClosest ) clReleaseKernel( ctx->kClosest );
	if ( ctx->queue ) clReleaseCommandQueue( ctx->queue );
	delete ctx;
}

static void ReleaseTraceCtxPool()
{
	for ( TraceCtx *ctx : g_ctxPool )
		DestroyTraceCtx( ctx );
	g_ctxPool.clear();
	g_ctxFree.clear();
}

static TraceCtx *CreateTraceCtx()
{
	TraceCtx *ctx = new TraceCtx;
	memset( ctx, 0, sizeof( *ctx ) );
	ctx->capacity = g_nMaxRayBatch;

	cl_int err = CL_SUCCESS;
	ctx->queue = clCreateCommandQueue( g_context, g_device, 0, &err );
	if ( !CheckCL( err, "ctx queue" ) ) { DestroyTraceCtx( ctx ); return nullptr; }
	ctx->kOcclusion = clCreateKernel( g_program, "kOcclusion", &err );
	if ( !CheckCL( err, "ctx kOcclusion" ) ) { DestroyTraceCtx( ctx ); return nullptr; }
	ctx->kClosest = clCreateKernel( g_program, "kClosest", &err );
	if ( !CheckCL( err, "ctx kClosest" ) ) { DestroyTraceCtx( ctx ); return nullptr; }
	ctx->dRays = clCreateBuffer( g_context, CL_MEM_READ_ONLY, sizeof( RayGPU ) * ctx->capacity, nullptr, &err );
	if ( !CheckCL( err, "ctx dRays" ) ) { DestroyTraceCtx( ctx ); return nullptr; }
	ctx->dVis = clCreateBuffer( g_context, CL_MEM_WRITE_ONLY, ctx->capacity, nullptr, &err );
	if ( !CheckCL( err, "ctx dVis" ) ) { DestroyTraceCtx( ctx ); return nullptr; }
	ctx->dHitT = clCreateBuffer( g_context, CL_MEM_WRITE_ONLY, sizeof( float ) * ctx->capacity, nullptr, &err );
	if ( !CheckCL( err, "ctx dHitT" ) ) { DestroyTraceCtx( ctx ); return nullptr; }
	ctx->dHitFlags = clCreateBuffer( g_context, CL_MEM_WRITE_ONLY, sizeof( int ) * ctx->capacity, nullptr, &err );
	if ( !CheckCL( err, "ctx dHitFlags" ) ) { DestroyTraceCtx( ctx ); return nullptr; }

	// Bind the static scene args once.
	clSetKernelArg( ctx->kOcclusion, 0, sizeof( cl_mem ), &ctx->dRays );
	clSetKernelArg( ctx->kOcclusion, 1, sizeof( cl_mem ), &ctx->dVis );
	clSetKernelArg( ctx->kOcclusion, 2, sizeof( cl_mem ), &g_dTris );
	clSetKernelArg( ctx->kOcclusion, 3, sizeof( cl_mem ), &g_dBVH );
	clSetKernelArg( ctx->kOcclusion, 4, sizeof( cl_mem ), &g_dPrims );
	clSetKernelArg( ctx->kClosest, 0, sizeof( cl_mem ), &ctx->dRays );
	clSetKernelArg( ctx->kClosest, 1, sizeof( cl_mem ), &ctx->dHitT );
	clSetKernelArg( ctx->kClosest, 2, sizeof( cl_mem ), &ctx->dHitFlags );
	clSetKernelArg( ctx->kClosest, 3, sizeof( cl_mem ), &g_dTris );
	clSetKernelArg( ctx->kClosest, 4, sizeof( cl_mem ), &g_dBVH );
	clSetKernelArg( ctx->kClosest, 5, sizeof( cl_mem ), &g_dPrims );
	return ctx;
}

static TraceCtx *AcquireTraceCtx()
{
	std::lock_guard<std::mutex> lock( g_clMutex );
	if ( !g_bActive || !g_bSceneUploaded || g_bGPUFailed )
		return nullptr;
	if ( !g_ctxFree.empty() )
	{
		TraceCtx *ctx = g_ctxFree.back();
		g_ctxFree.pop_back();
		return ctx;
	}
	if ( (int)g_ctxPool.size() >= kMaxTraceCtx )
		return nullptr;	// all busy - caller falls back to CPU tracing
	TraceCtx *ctx = CreateTraceCtx();
	if ( ctx )
		g_ctxPool.push_back( ctx );
	return ctx;
}

static void ReturnTraceCtx( TraceCtx *ctx )
{
	std::lock_guard<std::mutex> lock( g_clMutex );
	g_ctxFree.push_back( ctx );
}

static void ReleaseGatherBuffers()
{
	if ( g_dEmit ) { clReleaseMemObject( g_dEmit ); g_dEmit = nullptr; }
	if ( g_dRefl ) { clReleaseMemObject( g_dRefl ); g_dRefl = nullptr; }
	if ( g_dTrans ) { clReleaseMemObject( g_dTrans ); g_dTrans = nullptr; }
	if ( g_dOff ) { clReleaseMemObject( g_dOff ); g_dOff = nullptr; }
	if ( g_dAdd ) { clReleaseMemObject( g_dAdd ); g_dAdd = nullptr; }
	if ( g_dPatchEnv ) { clReleaseMemObject( g_dPatchEnv ); g_dPatchEnv = nullptr; }
	if ( g_dVolTint ) { clReleaseMemObject( g_dVolTint ); g_dVolTint = nullptr; }
	if ( g_dVolUseColor ) { clReleaseMemObject( g_dVolUseColor ); g_dVolUseColor = nullptr; }
	if ( g_dVolUseBright ) { clReleaseMemObject( g_dVolUseBright ); g_dVolUseBright = nullptr; }
	g_nPatchesCached = 0;
	g_nTransCached = 0;
}

static bool WriteBufferChunked( cl_mem buf, size_t totalBytes, const void *data, const char *what )
{
	const char *p = (const char *)data;
	for ( size_t off = 0; off < totalBytes; off += kUploadChunkBytes )
	{
		const size_t n = (std::min)( (size_t)kUploadChunkBytes, totalBytes - off );
		cl_int err = clEnqueueWriteBuffer( g_queue, buf, CL_TRUE, off, n, p + off, 0, nullptr, nullptr );
		if ( !CheckCL( err, what ) )
			return false;
	}
	return true;
}

static bool FitsDeviceAlloc( size_t bytes, const char *label )
{
	if ( g_deviceMaxAlloc == 0 )
		return true;
	// Leave headroom - some ICDs reject near-limit COPY/WRITE sizes.
	const cl_ulong limit = (cl_ulong)( g_deviceMaxAlloc * 0.85 );
	if ( bytes > (size_t)limit )
	{
		Warning( "[VRAD-GPU] %s is %zu MB (device max alloc ~%llu MB) - skipping GPU scene.\n",
				 label, bytes / ( 1024 * 1024 ),
				 (unsigned long long)( g_deviceMaxAlloc / ( 1024 * 1024 ) ) );
		return false;
	}
	return true;
}

static bool UploadSceneToGPU()
{
	ReleaseSceneBuffers();
	if ( !g_bCaptureDone || g_hostTris.empty() )
		return false;

	const int nCaptured = (int)g_hostTris.size();
	if ( g_nMaxTris > 0 && nCaptured > g_nMaxTris )
	{
		Warning( "[VRAD-GPU] Scene has %d tris (limit %d). GPU BVH skipped - bounce still on GPU; "
				 "occlusion/closest use CPU. Raise with -gpu_maxtris N or drop -StaticPropPolys.\n",
				 nCaptured, g_nMaxTris );
		g_hostTris.clear();
		g_hostTris.shrink_to_fit();
		g_bCaptureDone = false;
		return false;
	}

	BuildBVH();
	if ( g_hostBVH.empty() || g_primIndices.empty() )
		return false;

	g_nTris = (int)g_hostTris.size();
	g_nBVH = (int)g_hostBVH.size();

	const size_t triBytes = sizeof( TriGPU ) * (size_t)g_nTris;
	const size_t bvhBytes = sizeof( BVHNodeGPU ) * (size_t)g_nBVH;
	const size_t primBytes = sizeof( int ) * g_primIndices.size();
	if ( !FitsDeviceAlloc( triBytes, "triangle buffer" ) ||
		 !FitsDeviceAlloc( bvhBytes, "BVH buffer" ) ||
		 !FitsDeviceAlloc( primBytes, "prim index buffer" ) )
	{
		g_hostTris.clear();
		g_hostBVH.clear();
		g_primIndices.clear();
		g_hostTris.shrink_to_fit();
		g_hostBVH.shrink_to_fit();
		g_primIndices.shrink_to_fit();
		g_bCaptureDone = false;
		return false;
	}

	std::vector<TriGPU> tris( g_nTris );
	for ( int i = 0; i < g_nTris; ++i )
	{
		tris[i].a = ToF3( g_hostTris[i].a );
		tris[i].b = ToF3( g_hostTris[i].b );
		tris[i].c = ToF3( g_hostTris[i].c );
		tris[i].flags = g_hostTris[i].flags;
		tris[i].pad0 = tris[i].pad1 = tris[i].pad2 = 0;
	}

	// Free host triangle soup before device upload to cut peak RAM.
	g_hostTris.clear();
	g_hostTris.shrink_to_fit();
	g_bCaptureDone = false;

	cl_int err = CL_SUCCESS;
	g_dTris = clCreateBuffer( g_context, CL_MEM_READ_ONLY, triBytes, nullptr, &err );
	if ( !CheckCL( err, "tris buffer alloc" ) ) { ReleaseSceneBuffers(); return false; }
	g_dBVH = clCreateBuffer( g_context, CL_MEM_READ_ONLY, bvhBytes, nullptr, &err );
	if ( !CheckCL( err, "bvh buffer alloc" ) ) { ReleaseSceneBuffers(); return false; }
	g_dPrims = clCreateBuffer( g_context, CL_MEM_READ_ONLY, primBytes, nullptr, &err );
	if ( !CheckCL( err, "prims buffer alloc" ) ) { ReleaseSceneBuffers(); return false; }

	if ( !WriteBufferChunked( g_dTris, triBytes, tris.data(), "tris upload" ) ||
		 !WriteBufferChunked( g_dBVH, bvhBytes, g_hostBVH.data(), "bvh upload" ) ||
		 !WriteBufferChunked( g_dPrims, primBytes, g_primIndices.data(), "prims upload" ) )
	{
		ReleaseSceneBuffers();
		return false;
	}

	tris.clear();
	tris.shrink_to_fit();
	g_hostBVH.clear();
	g_primIndices.clear();
	g_hostBVH.shrink_to_fit();
	g_primIndices.shrink_to_fit();

	err = clFinish( g_queue );
	if ( !CheckCL( err, "scene upload finish" ) )
	{
		ReleaseSceneBuffers();
		return false;
	}

	g_bSceneUploaded = true;
	return true;
}

static void ShutdownUnlocked()
{
	g_bActive = false;
	ReleaseGatherBuffers();
	ReleaseTraceCtxPool();
	ReleaseSceneBuffers();
	if ( g_kOcclusion ) { clReleaseKernel( g_kOcclusion ); g_kOcclusion = nullptr; }
	if ( g_kClosest ) { clReleaseKernel( g_kClosest ); g_kClosest = nullptr; }
	if ( g_kGather ) { clReleaseKernel( g_kGather ); g_kGather = nullptr; }
	if ( g_program ) { clReleaseProgram( g_program ); g_program = nullptr; }
	if ( g_queue ) { clReleaseCommandQueue( g_queue ); g_queue = nullptr; }
	if ( g_context ) { clReleaseContext( g_context ); g_context = nullptr; }
	g_device = nullptr;
	g_platform = nullptr;
	g_deviceMaxAlloc = 0;
	g_deviceGlobalMem = 0;
	g_hostTris.clear();
	g_hostBVH.clear();
	g_primIndices.clear();
	g_bCaptureDone = false;
}

static bool FillRays( const Vector *pStarts, const Vector *pEnds, int offset, int count, std::vector<RayGPU> &rays )
{
	rays.resize( count );
	for ( int i = 0; i < count; ++i )
	{
		const int src = offset + i;
		rays[i].origin = ToF3( pStarts[src] );
		rays[i].dir = ToF3( pEnds[src] - pStarts[src] );
	}
	return true;
}

} // namespace

void VRadGPU_SetRequested( bool bRequested )
{
	g_bVRadGPURequested = bRequested;
}

void VRadGPU_SetMaxTris( int nMaxTris )
{
	g_nMaxTris = ( nMaxTris < 0 ) ? 0 : nMaxTris;
}

void VRadGPU_SetRayBatchSize( int nBatch )
{
	if ( nBatch < 1024 )
		nBatch = 1024;
	if ( nBatch > 262144 )
		nBatch = 262144;
	g_nMaxRayBatch = nBatch;
}

bool VRadGPU_IsActive()
{
	return g_bActive && !g_bGPUFailed;
}

bool VRadGPU_HasScene()
{
	return g_bSceneUploaded && !g_bGPUFailed;
}

void VRadGPU_CaptureScene( RayTracingEnvironment &rtEnv )
{
	g_hostTris.clear();
	g_bCaptureDone = false;

	const int n = rtEnv.OptimizedTriangleList.Count();
	g_hostTris.reserve( n );
	for ( int i = 0; i < n; ++i )
	{
		const CacheOptimizedTriangle &tri = rtEnv.OptimizedTriangleList[i];
		HostTri h;
		h.a = tri.Vertex( 0 );
		h.b = tri.Vertex( 1 );
		h.c = tri.Vertex( 2 );
		h.flags = tri.m_Data.m_GeometryData.m_nTriangleID;
		g_hostTris.push_back( h );
	}
	g_bCaptureDone = true;
	Msg( "[VRAD-GPU] Captured %d triangles for GPU BVH\n", n );
}

bool VRadGPU_InitAfterScene()
{
	g_bActive = false;
	g_bSceneUploaded = false;
	g_bGPUFailed = false;
	if ( !g_bVRadGPURequested )
		return false;

	std::lock_guard<std::mutex> lock( g_clMutex );

	cl_int err = CL_SUCCESS;
	cl_uint nPlat = 0;
	err = clGetPlatformIDs( 0, nullptr, &nPlat );
	if ( err != CL_SUCCESS || nPlat == 0 )
	{
		Warning( "[VRAD-GPU] No OpenCL platform; using CPU.\n" );
		return false;
	}
	std::vector<cl_platform_id> plats( nPlat );
	clGetPlatformIDs( nPlat, plats.data(), nullptr );

	cl_device_id bestDev = nullptr;
	cl_platform_id bestPlat = nullptr;
	cl_ulong bestMem = 0;
	for ( cl_uint p = 0; p < nPlat; ++p )
	{
		cl_uint nDev = 0;
		if ( clGetDeviceIDs( plats[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &nDev ) != CL_SUCCESS || nDev == 0 )
			continue;
		std::vector<cl_device_id> devs( nDev );
		clGetDeviceIDs( plats[p], CL_DEVICE_TYPE_GPU, nDev, devs.data(), nullptr );
		for ( cl_uint d = 0; d < nDev; ++d )
		{
			cl_ulong mem = 0;
			clGetDeviceInfo( devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( mem ), &mem, nullptr );
			if ( mem >= bestMem )
			{
				bestMem = mem;
				bestDev = devs[d];
				bestPlat = plats[p];
			}
		}
	}
	if ( !bestDev )
	{
		Warning( "[VRAD-GPU] No OpenCL GPU device; using CPU.\n" );
		return false;
	}

	g_platform = bestPlat;
	g_device = bestDev;
	g_deviceGlobalMem = bestMem;
	clGetDeviceInfo( g_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( g_deviceMaxAlloc ), &g_deviceMaxAlloc, nullptr );

	char name[256] = {};
	clGetDeviceInfo( g_device, CL_DEVICE_NAME, sizeof( name ), name, nullptr );
	Msg( "[VRAD-GPU] Using device: %s (%.1f GB, max alloc %.0f MB, ray batch %d, max tris %d)\n",
		 name,
		 (double)g_deviceGlobalMem / ( 1024.0 * 1024.0 * 1024.0 ),
		 (double)g_deviceMaxAlloc / ( 1024.0 * 1024.0 ),
		 g_nMaxRayBatch,
		 g_nMaxTris );

	g_context = clCreateContext( nullptr, 1, &g_device, nullptr, nullptr, &err );
	if ( err != CL_SUCCESS )
	{
		Warning( "[VRAD-GPU] clCreateContext failed; using CPU.\n" );
		return false;
	}

	g_queue = clCreateCommandQueue( g_context, g_device, 0, &err );
	if ( err != CL_SUCCESS )
	{
		Warning( "[VRAD-GPU] clCreateCommandQueue failed; using CPU.\n" );
		ShutdownUnlocked();
		return false;
	}

	const char *src = g_kernelSource;
	size_t srcLen = strlen( src );
	g_program = clCreateProgramWithSource( g_context, 1, &src, &srcLen, &err );
	if ( err != CL_SUCCESS )
	{
		Warning( "[VRAD-GPU] clCreateProgramWithSource failed; using CPU.\n" );
		ShutdownUnlocked();
		return false;
	}

	// Avoid -cl-fast-relaxed-math (can worsen numerical edge cases on some ICDs).
	err = clBuildProgram( g_program, 1, &g_device, "-cl-mad-enable", nullptr, nullptr );
	if ( err != CL_SUCCESS )
	{
		size_t logSize = 0;
		clGetProgramBuildInfo( g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize );
		std::vector<char> log( logSize + 1 );
		clGetProgramBuildInfo( g_program, g_device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr );
		Warning( "[VRAD-GPU] Kernel build failed:\n%s\n", log.data() );
		ShutdownUnlocked();
		return false;
	}

	g_kOcclusion = clCreateKernel( g_program, "kOcclusion", &err );
	CheckCL( err, "kOcclusion" );
	g_kClosest = clCreateKernel( g_program, "kClosest", &err );
	CheckCL( err, "kClosest" );
	g_kGather = clCreateKernel( g_program, "kGatherBounce", &err );
	CheckCL( err, "kGatherBounce" );
	if ( !g_kOcclusion || !g_kClosest || !g_kGather )
	{
		ShutdownUnlocked();
		return false;
	}

	bool bSceneOk = false;
	if ( g_bCaptureDone && !g_hostTris.empty() )
	{
		Msg( "[VRAD-GPU] Building BVH...\n" );
		bSceneOk = UploadSceneToGPU();
		if ( !bSceneOk )
			Warning( "[VRAD-GPU] BVH upload skipped/failed; occlusion/closest stay on CPU.\n" );
	}
	else
	{
		Msg( "[VRAD-GPU] No scene captured before KD build; occlusion/closest stay on CPU.\n" );
	}

	g_bActive = true;
	if ( bSceneOk )
	{
		Msg( "[VRAD-GPU] Ready - bounce gather + occlusion/closest (%d tris, %d BVH nodes).\n",
			 g_nTris, g_nBVH );
	}
	else
	{
		Msg( "[VRAD-GPU] Ready - bounce gather accelerated (CPU traces).\n" );
	}
	return true;
}

void VRadGPU_Shutdown()
{
	std::lock_guard<std::mutex> lock( g_clMutex );
	ShutdownUnlocked();
}

bool VRadGPU_TraceOcclusion( const Vector *pStarts, const Vector *pEnds, unsigned char *pVisible, int nRays )
{
	if ( !VRadGPU_HasScene() || nRays < kMinGPURays )
		return false;

	TraceCtx *ctx = AcquireTraceCtx();
	if ( !ctx )
		return false;

	bool bOk = true;
	std::vector<RayGPU> rays;
	for ( int offset = 0; offset < nRays && bOk; offset += ctx->capacity )
	{
		const int count = (std::min)( ctx->capacity, nRays - offset );
		FillRays( pStarts, pEnds, offset, count, rays );

		cl_int err = clEnqueueWriteBuffer( ctx->queue, ctx->dRays, CL_FALSE, 0,
										   sizeof( RayGPU ) * count, rays.data(), 0, nullptr, nullptr );
		if ( !CheckCL( err, "write rays" ) )
		{
			MarkGPUFailed( "occlusion ray upload" );
			bOk = false;
			break;
		}

		clSetKernelArg( ctx->kOcclusion, 5, sizeof( int ), &count );

		size_t global = (size_t)count;
		err = clEnqueueNDRangeKernel( ctx->queue, ctx->kOcclusion, 1, nullptr, &global, nullptr, 0, nullptr, nullptr );
		if ( !CheckCL( err, "kOcclusion enqueue" ) )
		{
			MarkGPUFailed( "occlusion kernel" );
			bOk = false;
			break;
		}

		err = clEnqueueReadBuffer( ctx->queue, ctx->dVis, CL_TRUE, 0, count, pVisible + offset, 0, nullptr, nullptr );
		if ( !CheckCL( err, "occlusion read" ) )
		{
			MarkGPUFailed( "occlusion readback" );
			bOk = false;
			break;
		}
	}

	ReturnTraceCtx( ctx );
	return bOk;
}

bool VRadGPU_TraceClosest( const Vector *pStarts, const Vector *pEnds,
						   float *pHitT, int *pHitFlags, int nRays )
{
	if ( !VRadGPU_HasScene() || nRays < kMinGPURays )
		return false;

	TraceCtx *ctx = AcquireTraceCtx();
	if ( !ctx )
		return false;

	bool bOk = true;
	std::vector<RayGPU> rays;
	for ( int offset = 0; offset < nRays && bOk; offset += ctx->capacity )
	{
		const int count = (std::min)( ctx->capacity, nRays - offset );
		FillRays( pStarts, pEnds, offset, count, rays );

		cl_int err = clEnqueueWriteBuffer( ctx->queue, ctx->dRays, CL_FALSE, 0,
										   sizeof( RayGPU ) * count, rays.data(), 0, nullptr, nullptr );
		if ( !CheckCL( err, "write rays" ) )
		{
			MarkGPUFailed( "closest ray upload" );
			bOk = false;
			break;
		}

		clSetKernelArg( ctx->kClosest, 6, sizeof( int ), &count );

		size_t global = (size_t)count;
		err = clEnqueueNDRangeKernel( ctx->queue, ctx->kClosest, 1, nullptr, &global, nullptr, 0, nullptr, nullptr );
		if ( !CheckCL( err, "kClosest enqueue" ) )
		{
			MarkGPUFailed( "closest kernel" );
			bOk = false;
			break;
		}

		err = clEnqueueReadBuffer( ctx->queue, ctx->dHitT, CL_FALSE, 0,
								   sizeof( float ) * count, pHitT + offset, 0, nullptr, nullptr );
		if ( !CheckCL( err, "closest hitT read" ) )
		{
			MarkGPUFailed( "closest readback" );
			bOk = false;
			break;
		}
		err = clEnqueueReadBuffer( ctx->queue, ctx->dHitFlags, CL_TRUE, 0,
								   sizeof( int ) * count, pHitFlags + offset, 0, nullptr, nullptr );
		if ( !CheckCL( err, "closest flags read" ) )
		{
			MarkGPUFailed( "closest readback" );
			bOk = false;
			break;
		}
	}

	ReturnTraceCtx( ctx );
	return bOk;
}

bool VRadGPU_UploadTransferGraph( const Vector *pReflectivity,
								  const VRadGPUTransfer_t *pTransfers,
								  const int *pOffsets,
								  const int *pPatchEnvIds,
								  int nPatches )
{
	if ( !VRadGPU_IsActive() || nPatches <= 0 )
		return false;

	std::lock_guard<std::mutex> lock( g_clMutex );
	if ( g_bGPUFailed )
		return false;

	const int nTrans = pOffsets[nPatches];
	if ( nTrans <= 0 )
		return false;

	ReleaseGatherBuffers();

	cl_int err = CL_SUCCESS;
	const size_t reflBytes = sizeof( Vector ) * (size_t)nPatches;
	const size_t transBytes = sizeof( VRadGPUTransfer_t ) * (size_t)nTrans;
	const size_t offBytes = sizeof( int ) * (size_t)( nPatches + 1 );
	const size_t envBytes = sizeof( int ) * (size_t)nPatches;

	if ( !FitsDeviceAlloc( reflBytes, "reflectivity" ) ||
		 !FitsDeviceAlloc( transBytes, "transfers" ) )
	{
		Warning( "[VRAD-GPU] Transfer graph too large for device; bounce gather stays on CPU.\n" );
		return false;
	}

	g_dRefl = clCreateBuffer( g_context, CL_MEM_READ_ONLY, reflBytes, nullptr, &err );
	if ( !CheckCL( err, "dRefl alloc" ) ) { ReleaseGatherBuffers(); return false; }
	g_dTrans = clCreateBuffer( g_context, CL_MEM_READ_ONLY, transBytes, nullptr, &err );
	if ( !CheckCL( err, "dTrans alloc" ) ) { ReleaseGatherBuffers(); return false; }
	g_dOff = clCreateBuffer( g_context, CL_MEM_READ_ONLY, offBytes, nullptr, &err );
	if ( !CheckCL( err, "dOff alloc" ) ) { ReleaseGatherBuffers(); return false; }
	g_dPatchEnv = clCreateBuffer( g_context, CL_MEM_READ_ONLY, envBytes, nullptr, &err );
	if ( !CheckCL( err, "dPatchEnv alloc" ) ) { ReleaseGatherBuffers(); return false; }
	g_dEmit = clCreateBuffer( g_context, CL_MEM_READ_ONLY, reflBytes, nullptr, &err );
	if ( !CheckCL( err, "dEmit alloc" ) ) { ReleaseGatherBuffers(); return false; }
	g_dAdd = clCreateBuffer( g_context, CL_MEM_WRITE_ONLY, reflBytes, nullptr, &err );
	if ( !CheckCL( err, "dAdd alloc" ) ) { ReleaseGatherBuffers(); return false; }

	if ( !WriteBufferChunked( g_dRefl, reflBytes, pReflectivity, "refl upload" ) ||
		 !WriteBufferChunked( g_dTrans, transBytes, pTransfers, "trans upload" ) ||
		 !WriteBufferChunked( g_dOff, offBytes, pOffsets, "off upload" ) ||
		 !WriteBufferChunked( g_dPatchEnv, envBytes, pPatchEnvIds, "env upload" ) )
	{
		ReleaseGatherBuffers();
		return false;
	}

	const int nVol = LightEnv_VolumeCount();
	const int nTint = nVol + 1;
	std::vector<Float4Host> tints( nTint );
	std::vector<int> useColor( nTint, 0 );
	std::vector<int> useBright( nTint, 0 );
	memset( tints.data(), 0, sizeof( Float4Host ) * nTint );
	for ( int e = 1; e <= nVol; ++e )
	{
		const LightEnvVolumeInfo_t *v = LightEnv_GetVolumeInfo( e );
		if ( !v )
			continue;
		tints[e].x = v->bounceTint.x;
		tints[e].y = v->bounceTint.y;
		tints[e].z = v->bounceTint.z;
		tints[e].w = v->bounceIntensity;
		useColor[e] = v->bInboundBounceUsesVolumeColor ? 1 : 0;
		useBright[e] = v->bInboundBounceUsesVolumeBrightness ? 1 : 0;
	}
	g_flDefaultBounceIntensity = LightEnv_GetDefaultBounceIntensity();
	g_dVolTint = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								 sizeof( Float4Host ) * nTint, tints.data(), &err );
	g_dVolUseColor = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									 sizeof( int ) * nTint, useColor.data(), &err );
	g_dVolUseBright = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									  sizeof( int ) * nTint, useBright.data(), &err );

	if ( !g_dVolTint || !g_dVolUseColor || !g_dVolUseBright )
	{
		Warning( "[VRAD-GPU] UploadTransferGraph volume buffers failed\n" );
		ReleaseGatherBuffers();
		return false;
	}

	g_nPatchesCached = nPatches;
	g_nTransCached = nTrans;
	Msg( "[VRAD-GPU] Uploaded transfer graph: %d patches, %d transfers\n", nPatches, nTrans );
	return true;
}

bool VRadGPU_GatherBounce( const Vector *pEmitLight, Vector *pAddLight, int nPatches )
{
	if ( !VRadGPU_IsActive() || nPatches != g_nPatchesCached || !g_dEmit )
		return false;

	std::lock_guard<std::mutex> lock( g_clMutex );
	if ( g_bGPUFailed )
		return false;

	cl_int err = clEnqueueWriteBuffer( g_queue, g_dEmit, CL_FALSE, 0,
									   sizeof( Vector ) * nPatches, pEmitLight, 0, nullptr, nullptr );
	if ( !CheckCL( err, "emit upload" ) )
	{
		MarkGPUFailed( "gather emit upload" );
		return false;
	}

	clSetKernelArg( g_kGather, 0, sizeof( cl_mem ), &g_dEmit );
	clSetKernelArg( g_kGather, 1, sizeof( cl_mem ), &g_dRefl );
	clSetKernelArg( g_kGather, 2, sizeof( cl_mem ), &g_dTrans );
	clSetKernelArg( g_kGather, 3, sizeof( cl_mem ), &g_dOff );
	clSetKernelArg( g_kGather, 4, sizeof( cl_mem ), &g_dPatchEnv );
	clSetKernelArg( g_kGather, 5, sizeof( cl_mem ), &g_dVolTint );
	clSetKernelArg( g_kGather, 6, sizeof( cl_mem ), &g_dVolUseColor );
	clSetKernelArg( g_kGather, 7, sizeof( cl_mem ), &g_dVolUseBright );
	clSetKernelArg( g_kGather, 8, sizeof( cl_mem ), &g_dAdd );
	clSetKernelArg( g_kGather, 9, sizeof( int ), &nPatches );
	clSetKernelArg( g_kGather, 10, sizeof( float ), &g_flDefaultBounceIntensity );

	size_t global = (size_t)nPatches;
	err = clEnqueueNDRangeKernel( g_queue, g_kGather, 1, nullptr, &global, nullptr, 0, nullptr, nullptr );
	if ( !CheckCL( err, "kGatherBounce enqueue" ) )
	{
		MarkGPUFailed( "gather kernel" );
		return false;
	}

	err = clEnqueueReadBuffer( g_queue, g_dAdd, CL_TRUE, 0,
							   sizeof( Vector ) * nPatches, pAddLight, 0, nullptr, nullptr );
	if ( !CheckCL( err, "gather readback" ) )
	{
		MarkGPUFailed( "gather readback" );
		return false;
	}
	return true;
}
