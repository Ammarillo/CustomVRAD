//========= Copyright Valve Corporation, All rights reserved. ============//
//
// OpenCL acceleration for VRAD (-gpu): BVH occlusion/closest + bounce gather.
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

bool g_bVRadGPURequested = false;
bool g_bVRadGPUTransfers = false;
bool g_bVRadCoarsePatches = false;
float g_flMaxTransferDist = 0.0f;

namespace
{
std::mutex g_clMutex;
bool g_bActive = false;
bool g_bSceneUploaded = false;
bool g_bCaptureDone = false;

cl_platform_id g_platform = nullptr;
cl_device_id g_device = nullptr;
cl_context g_context = nullptr;
cl_command_queue g_queue = nullptr;
cl_program g_program = nullptr;
cl_kernel g_kOcclusion = nullptr;
cl_kernel g_kClosest = nullptr;
cl_kernel g_kGather = nullptr;

// Scene BVH
cl_mem g_dTris = nullptr;
cl_mem g_dBVH = nullptr;
cl_mem g_dPrims = nullptr;
int g_nTris = 0;
int g_nBVH = 0;

// Persistent reusable ray buffers (grow-only)
cl_mem g_dRays = nullptr;
cl_mem g_dVis = nullptr;
cl_mem g_dHitT = nullptr;
cl_mem g_dHitFlags = nullptr;
int g_nRayCapacity = 0;

// Persistent gather buffers
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

struct Float4Host { float x, y, z, w; };

#pragma pack(push, 4)
struct Float3 { float x, y, z, w; };
struct TriGPU { Float3 a, b, c; int flags; int pad0, pad1, pad2; };
struct BVHNodeGPU { Float3 bmin, bmax; int left, right, firstPrim, primCount; };
struct RayGPU { Float3 origin, dir; }; // dir = end-start (unnormalized); tmax = length
#pragma pack(pop)

struct HostTri
{
	Vector a, b, c;
	int flags;
};

std::vector<HostTri> g_hostTris;
std::vector<BVHNodeGPU> g_hostBVH;
std::vector<int> g_primIndices;

enum { kMinGPURays = 32 };

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

// Möller–Trumbore; returns t>0 on hit (ray = o + t*d with |d|=1)
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
      Tint4 tint = volTint[recvEnv];
      if (volUseBright[recvEnv] != 0) {
        float scale = tint.w / defaultIntensity;
        vx = tint.x * (lum * scale);
        vy = tint.y * (lum * scale);
        vz = tint.z * (lum * scale);
      } else {
        vx = tint.x * lum;
        vy = tint.y * lum;
        vz = tint.z * lum;
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

static void CheckCL( cl_int err, const char *what )
{
	if ( err != CL_SUCCESS )
		Warning( "[VRAD-GPU] %s failed (cl_int %d)\n", what, (int)err );
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
	g_hostBVH.push_back( node ); // placeholder

	if ( count <= 4 || depth > 48 )
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

static void ReleaseRayBuffers()
{
	if ( g_dRays ) { clReleaseMemObject( g_dRays ); g_dRays = nullptr; }
	if ( g_dVis ) { clReleaseMemObject( g_dVis ); g_dVis = nullptr; }
	if ( g_dHitT ) { clReleaseMemObject( g_dHitT ); g_dHitT = nullptr; }
	if ( g_dHitFlags ) { clReleaseMemObject( g_dHitFlags ); g_dHitFlags = nullptr; }
	g_nRayCapacity = 0;
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

static bool EnsureRayBuffers( int nRays )
{
	if ( nRays <= g_nRayCapacity && g_dRays && g_dVis && g_dHitT && g_dHitFlags )
		return true;

	ReleaseRayBuffers();

	const int capacity = ( nRays < 256 ) ? 256 : nRays;
	cl_int err = CL_SUCCESS;
	g_dRays = clCreateBuffer( g_context, CL_MEM_READ_ONLY, sizeof( RayGPU ) * capacity, nullptr, &err );
	CheckCL( err, "dRays" );
	g_dVis = clCreateBuffer( g_context, CL_MEM_WRITE_ONLY, capacity, nullptr, &err );
	CheckCL( err, "dVis" );
	g_dHitT = clCreateBuffer( g_context, CL_MEM_WRITE_ONLY, sizeof( float ) * capacity, nullptr, &err );
	CheckCL( err, "dHitT" );
	g_dHitFlags = clCreateBuffer( g_context, CL_MEM_WRITE_ONLY, sizeof( int ) * capacity, nullptr, &err );
	CheckCL( err, "dHitFlags" );

	if ( !g_dRays || !g_dVis || !g_dHitT || !g_dHitFlags )
	{
		ReleaseRayBuffers();
		return false;
	}

	g_nRayCapacity = capacity;
	return true;
}

static bool UploadSceneToGPU()
{
	ReleaseSceneBuffers();
	if ( !g_bCaptureDone || g_hostTris.empty() )
		return false;

	BuildBVH();
	if ( g_hostBVH.empty() || g_primIndices.empty() )
		return false;

	g_nTris = (int)g_hostTris.size();
	g_nBVH = (int)g_hostBVH.size();

	std::vector<TriGPU> tris( g_nTris );
	for ( int i = 0; i < g_nTris; ++i )
	{
		tris[i].a = ToF3( g_hostTris[i].a );
		tris[i].b = ToF3( g_hostTris[i].b );
		tris[i].c = ToF3( g_hostTris[i].c );
		tris[i].flags = g_hostTris[i].flags;
		tris[i].pad0 = tris[i].pad1 = tris[i].pad2 = 0;
	}

	cl_int err = CL_SUCCESS;
	g_dTris = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
							  sizeof( TriGPU ) * g_nTris, tris.data(), &err );
	CheckCL( err, "tris buffer" );
	g_dBVH = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
							 sizeof( BVHNodeGPU ) * g_nBVH, g_hostBVH.data(), &err );
	CheckCL( err, "bvh buffer" );
	g_dPrims = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
							   sizeof( int ) * g_primIndices.size(), g_primIndices.data(), &err );
	CheckCL( err, "prims buffer" );

	if ( !g_dTris || !g_dBVH || !g_dPrims )
	{
		ReleaseSceneBuffers();
		return false;
	}

	g_bSceneUploaded = true;
	return true;
}

// Caller must hold g_clMutex (or be single-threaded during init failure).
static void ShutdownUnlocked()
{
	g_bActive = false;
	ReleaseGatherBuffers();
	ReleaseRayBuffers();
	ReleaseSceneBuffers();
	if ( g_kOcclusion ) { clReleaseKernel( g_kOcclusion ); g_kOcclusion = nullptr; }
	if ( g_kClosest ) { clReleaseKernel( g_kClosest ); g_kClosest = nullptr; }
	if ( g_kGather ) { clReleaseKernel( g_kGather ); g_kGather = nullptr; }
	if ( g_program ) { clReleaseProgram( g_program ); g_program = nullptr; }
	if ( g_queue ) { clReleaseCommandQueue( g_queue ); g_queue = nullptr; }
	if ( g_context ) { clReleaseContext( g_context ); g_context = nullptr; }
	g_device = nullptr;
	g_platform = nullptr;
	g_hostTris.clear();
	g_hostBVH.clear();
	g_primIndices.clear();
	g_bCaptureDone = false;
}

} // namespace

void VRadGPU_SetRequested( bool bRequested )
{
	g_bVRadGPURequested = bRequested;
}

void VRadGPU_SetTransfersRequested( bool bRequested )
{
	g_bVRadGPUTransfers = bRequested;
}

bool VRadGPU_IsActive()
{
	return g_bActive;
}

bool VRadGPU_HasScene()
{
	return g_bSceneUploaded;
}

void VRadGPU_CaptureScene( RayTracingEnvironment &rtEnv )
{
	g_hostTris.clear();
	g_bCaptureDone = false;

	// Capture while Vertex() is still valid (before KD convert / intersection format).
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

	bool found = false;
	for ( cl_uint p = 0; p < nPlat && !found; ++p )
	{
		cl_uint nDev = 0;
		if ( clGetDeviceIDs( plats[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &nDev ) != CL_SUCCESS || nDev == 0 )
			continue;
		std::vector<cl_device_id> devs( nDev );
		clGetDeviceIDs( plats[p], CL_DEVICE_TYPE_GPU, nDev, devs.data(), nullptr );
		g_platform = plats[p];
		g_device = devs[0];
		found = true;
	}
	if ( !found )
	{
		Warning( "[VRAD-GPU] No OpenCL GPU device; using CPU.\n" );
		return false;
	}

	char name[256] = {};
	clGetDeviceInfo( g_device, CL_DEVICE_NAME, sizeof( name ), name, nullptr );
	Msg( "[VRAD-GPU] Using device: %s\n", name );

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

	err = clBuildProgram( g_program, 1, &g_device, "-cl-fast-relaxed-math", nullptr, nullptr );
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
			Warning( "[VRAD-GPU] BVH upload failed; occlusion/closest stay on CPU.\n" );
	}
	else
	{
		Msg( "[VRAD-GPU] No scene captured before KD build; occlusion/closest stay on CPU.\n" );
	}

	g_bActive = true;
	if ( bSceneOk )
	{
		Msg( "[VRAD-GPU] Ready — bounce gather + occlusion/closest (%d tris, %d BVH nodes).\n",
			 g_nTris, g_nBVH );
	}
	else
	{
		Msg( "[VRAD-GPU] Ready — bounce gather accelerated.\n" );
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
	if ( !g_bActive || !g_bSceneUploaded || nRays < kMinGPURays )
		return false;

	std::lock_guard<std::mutex> lock( g_clMutex );

	if ( !EnsureRayBuffers( nRays ) )
		return false;

	std::vector<RayGPU> rays( nRays );
	for ( int i = 0; i < nRays; ++i )
	{
		rays[i].origin = ToF3( pStarts[i] );
		rays[i].dir = ToF3( pEnds[i] - pStarts[i] );
	}

	cl_int err = clEnqueueWriteBuffer( g_queue, g_dRays, CL_FALSE, 0,
									   sizeof( RayGPU ) * nRays, rays.data(), 0, nullptr, nullptr );
	if ( err != CL_SUCCESS )
	{
		CheckCL( err, "write rays" );
		return false;
	}

	clSetKernelArg( g_kOcclusion, 0, sizeof( cl_mem ), &g_dRays );
	clSetKernelArg( g_kOcclusion, 1, sizeof( cl_mem ), &g_dVis );
	clSetKernelArg( g_kOcclusion, 2, sizeof( cl_mem ), &g_dTris );
	clSetKernelArg( g_kOcclusion, 3, sizeof( cl_mem ), &g_dBVH );
	clSetKernelArg( g_kOcclusion, 4, sizeof( cl_mem ), &g_dPrims );
	clSetKernelArg( g_kOcclusion, 5, sizeof( int ), &nRays );

	size_t global = (size_t)nRays;
	err = clEnqueueNDRangeKernel( g_queue, g_kOcclusion, 1, nullptr, &global, nullptr, 0, nullptr, nullptr );
	if ( err != CL_SUCCESS )
	{
		CheckCL( err, "kOcclusion enqueue" );
		return false;
	}

	err = clEnqueueReadBuffer( g_queue, g_dVis, CL_TRUE, 0, nRays, pVisible, 0, nullptr, nullptr );
	return err == CL_SUCCESS;
}

bool VRadGPU_TraceClosest( const Vector *pStarts, const Vector *pEnds,
						   float *pHitT, int *pHitFlags, int nRays )
{
	if ( !g_bActive || !g_bSceneUploaded || nRays < kMinGPURays )
		return false;

	std::lock_guard<std::mutex> lock( g_clMutex );

	if ( !EnsureRayBuffers( nRays ) )
		return false;

	std::vector<RayGPU> rays( nRays );
	for ( int i = 0; i < nRays; ++i )
	{
		rays[i].origin = ToF3( pStarts[i] );
		rays[i].dir = ToF3( pEnds[i] - pStarts[i] );
	}

	cl_int err = clEnqueueWriteBuffer( g_queue, g_dRays, CL_FALSE, 0,
									   sizeof( RayGPU ) * nRays, rays.data(), 0, nullptr, nullptr );
	if ( err != CL_SUCCESS )
	{
		CheckCL( err, "write rays" );
		return false;
	}

	clSetKernelArg( g_kClosest, 0, sizeof( cl_mem ), &g_dRays );
	clSetKernelArg( g_kClosest, 1, sizeof( cl_mem ), &g_dHitT );
	clSetKernelArg( g_kClosest, 2, sizeof( cl_mem ), &g_dHitFlags );
	clSetKernelArg( g_kClosest, 3, sizeof( cl_mem ), &g_dTris );
	clSetKernelArg( g_kClosest, 4, sizeof( cl_mem ), &g_dBVH );
	clSetKernelArg( g_kClosest, 5, sizeof( cl_mem ), &g_dPrims );
	clSetKernelArg( g_kClosest, 6, sizeof( int ), &nRays );

	size_t global = (size_t)nRays;
	err = clEnqueueNDRangeKernel( g_queue, g_kClosest, 1, nullptr, &global, nullptr, 0, nullptr, nullptr );
	if ( err != CL_SUCCESS )
	{
		CheckCL( err, "kClosest enqueue" );
		return false;
	}

	err = clEnqueueReadBuffer( g_queue, g_dHitT, CL_FALSE, 0,
							   sizeof( float ) * nRays, pHitT, 0, nullptr, nullptr );
	if ( err != CL_SUCCESS )
		return false;
	err = clEnqueueReadBuffer( g_queue, g_dHitFlags, CL_TRUE, 0,
							   sizeof( int ) * nRays, pHitFlags, 0, nullptr, nullptr );
	return err == CL_SUCCESS;
}

bool VRadGPU_UploadTransferGraph( const Vector *pReflectivity,
								  const VRadGPUTransfer_t *pTransfers,
								  const int *pOffsets,
								  const int *pPatchEnvIds,
								  int nPatches )
{
	if ( !g_bActive || nPatches <= 0 )
		return false;

	std::lock_guard<std::mutex> lock( g_clMutex );

	const int nTrans = pOffsets[nPatches];
	if ( nTrans <= 0 )
		return false;

	ReleaseGatherBuffers();

	cl_int err;
	g_dRefl = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
							  sizeof( Vector ) * nPatches, (void *)pReflectivity, &err );
	g_dTrans = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
							   sizeof( VRadGPUTransfer_t ) * nTrans, (void *)pTransfers, &err );
	g_dOff = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
							 sizeof( int ) * ( nPatches + 1 ), (void *)pOffsets, &err );
	g_dPatchEnv = clCreateBuffer( g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								  sizeof( int ) * nPatches, (void *)pPatchEnvIds, &err );
	g_dEmit = clCreateBuffer( g_context, CL_MEM_READ_ONLY, sizeof( Vector ) * nPatches, nullptr, &err );
	g_dAdd = clCreateBuffer( g_context, CL_MEM_WRITE_ONLY, sizeof( Vector ) * nPatches, nullptr, &err );

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

	if ( !g_dRefl || !g_dTrans || !g_dOff || !g_dPatchEnv || !g_dEmit || !g_dAdd ||
		 !g_dVolTint || !g_dVolUseColor || !g_dVolUseBright )
	{
		Warning( "[VRAD-GPU] UploadTransferGraph buffer alloc failed\n" );
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
	if ( !g_bActive || nPatches != g_nPatchesCached || !g_dEmit )
		return false;

	std::lock_guard<std::mutex> lock( g_clMutex );

	cl_int err = clEnqueueWriteBuffer( g_queue, g_dEmit, CL_FALSE, 0,
									   sizeof( Vector ) * nPatches, pEmitLight, 0, nullptr, nullptr );

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
	if ( err != CL_SUCCESS )
	{
		CheckCL( err, "kGatherBounce enqueue" );
		return false;
	}
	err = clEnqueueReadBuffer( g_queue, g_dAdd, CL_TRUE, 0,
							   sizeof( Vector ) * nPatches, pAddLight, 0, nullptr, nullptr );
	return err == CL_SUCCESS;
}
