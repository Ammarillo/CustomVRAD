//========= Copyright CustomVRAD contributors. ============//
//
// D3D12 + DXR device / BLAS / TLAS / batched closest-hit queries.
//
//=============================================================================//

#include "pathtrace_dxr.h"
#include "pathtrace_dxr_device.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#define DXC_API_IMPORT
#include <dxcapi.h>
#include <wrl/client.h>

#include <vector>
#include <mutex>
#include <cstring>

#include "tier0/dbg.h"
#include "mathlib/vector.h"
#include "raytrace.h"
#include "vrad.h"
#include "worldsize.h"
#include "envvolume.h"
#include "bounce_vol.h"

#pragma comment( lib, "d3d12.lib" )
#pragma comment( lib, "dxgi.lib" )
#pragma comment( lib, "dxguid.lib" )
#pragma comment( lib, "d3dcompiler.lib" )

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Host triangle capture (before KD convert)
// ---------------------------------------------------------------------------
struct PtHostTri
{
	Vector	a, b, c;
	uint32	flags;	// TRACE_ID_*
};

static std::vector<PtHostTri>	g_ptTris;
static bool						g_bPtCaptureDone = false;

void PathTraceDXR_CaptureScene( RayTracingEnvironment &rtEnv )
{
	g_ptTris.clear();
	g_bPtCaptureDone = false;
	const int n = rtEnv.OptimizedTriangleList.Count();
	g_ptTris.reserve( n );
	for ( int i = 0; i < n; ++i )
	{
		const CacheOptimizedTriangle &tri = rtEnv.OptimizedTriangleList[i];
		PtHostTri h;
		h.a = tri.Vertex( 0 );
		h.b = tri.Vertex( 1 );
		h.c = tri.Vertex( 2 );
		h.flags = (uint32)tri.m_Data.m_GeometryData.m_nTriangleID;
		g_ptTris.push_back( h );
	}
	g_bPtCaptureDone = true;
	Msg( "[PathTrace-DXR] Captured %d triangles for DXR AS\n", n );
}

// ---------------------------------------------------------------------------
// GPU ray / hit records (must match HLSL)
// ---------------------------------------------------------------------------
struct PtGpuRay
{
	float origin[3];
	float tmin;
	float direction[3];
	float tmax;
};

struct PtGpuHit
{
	float	t;
	uint32	flags;
	uint32	prim;
	uint32	hit;	// 1 = triangle hit, 0 = miss
};

static const char *kPtRayQueryCS = R"HLSL(
RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<float4> Rays : register(t1);   // origin.xyz, tmin | direction.xyz, tmax  (2 x float4 per ray)
RWStructuredBuffer<float4> Hits : register(u0); // t, asfloat(flags), asfloat(prim), asfloat(hit)
ByteAddressBuffer TriFlags : register(t2);
cbuffer CB : register(b0) { uint RayCount; uint Pad0; uint Pad1; uint Pad2; };

[numthreads(64, 1, 1)]
void CSMain( uint3 dtid : SV_DispatchThreadID )
{
	uint i = dtid.x;
	if ( i >= RayCount )
		return;

	float4 o = Rays[i * 2 + 0];
	float4 d = Rays[i * 2 + 1];

	RayDesc ray;
	ray.Origin = o.xyz;
	ray.TMin = o.w;
	ray.Direction = d.xyz;
	ray.TMax = d.w;

	RayQuery<RAY_FLAG_NONE> q;
	q.TraceRayInline( Scene, RAY_FLAG_NONE, 0xFF, ray );
	q.Proceed();

	float4 hit = float4( -1, 0, 0, 0 );
	if ( q.CommittedStatus() == COMMITTED_TRIANGLE_HIT )
	{
		uint prim = q.CommittedPrimitiveIndex();
		uint flags = TriFlags.Load( prim * 4 );
		hit = float4( q.CommittedRayT(), asfloat( flags ), asfloat( prim ), asfloat( 1u ) );
	}
	Hits[i] = hit;
}
)HLSL";

// ---------------------------------------------------------------------------
// Device state
// ---------------------------------------------------------------------------
struct PtDevice
{
	ComPtr<ID3D12Device5>				device;
	ComPtr<ID3D12CommandQueue>			queue;
	ComPtr<ID3D12CommandAllocator>		alloc;
	ComPtr<ID3D12GraphicsCommandList4>	list;
	ComPtr<ID3D12Fence>					fence;
	UINT64								fenceValue = 0;
	HANDLE								fenceEvent = nullptr;

	ComPtr<ID3D12Resource>				blas;
	ComPtr<ID3D12Resource>				tlas;
	ComPtr<ID3D12Resource>				vertexBuffer;
	ComPtr<ID3D12Resource>				flagBuffer;
	ComPtr<ID3D12DescriptorHeap>		srvUavHeap;
	D3D12_GPU_DESCRIPTOR_HANDLE			tlasSrvGpu = {};
	D3D12_GPU_VIRTUAL_ADDRESS			tlasVA = 0;

	ComPtr<ID3D12RootSignature>			rootSig;
	ComPtr<ID3D12PipelineState>			pso;

	ComPtr<ID3D12Resource>				rayUpload;
	ComPtr<ID3D12Resource>				rayDefault;
	ComPtr<ID3D12Resource>				hitDefault;
	ComPtr<ID3D12Resource>				hitReadback;
	ComPtr<ID3D12Resource>				cbUpload;
	uint32								rayCapacity = 0;

	bool								ready = false;
	std::mutex							mutex;
};

static PtDevice g_ptDev;
static HMODULE g_hDxil = nullptr;
static HMODULE g_hDxc = nullptr;

static void PtWaitGPU()
{
	const UINT64 v = ++g_ptDev.fenceValue;
	g_ptDev.queue->Signal( g_ptDev.fence.Get(), v );
	if ( g_ptDev.fence->GetCompletedValue() < v )
	{
		g_ptDev.fence->SetEventOnCompletion( v, g_ptDev.fenceEvent );
		WaitForSingleObject( g_ptDev.fenceEvent, INFINITE );
	}
}

static bool PtCreateBuffer( UINT64 size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
							D3D12_RESOURCE_STATES state, ID3D12Resource **ppOut, const wchar_t *name )
{
	D3D12_HEAP_PROPERTIES hp = {};
	hp.Type = heap;
	D3D12_RESOURCE_DESC rd = {};
	rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	rd.Width = size;
	rd.Height = 1;
	rd.DepthOrArraySize = 1;
	rd.MipLevels = 1;
	rd.SampleDesc.Count = 1;
	rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	rd.Flags = flags;
	HRESULT hr = g_ptDev.device->CreateCommittedResource( &hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS( ppOut ) );
	if ( FAILED( hr ) )
		return false;
	if ( name )
		( *ppOut )->SetName( name );
	return true;
}

static bool PtCompileCS( ID3DBlob **ppBlob )
{
	*ppBlob = nullptr;

	const wchar_t *paths[] = {
		L"dxcompiler.dll",
		L"C:\\Windows\\System32\\dxcompiler.dll",
	};

	for ( int i = 0; i < 2 && !g_hDxc; ++i )
		g_hDxc = LoadLibraryW( paths[i] );

	if ( !g_hDxc )
	{
		wchar_t sdkRoot[MAX_PATH] = {};
		DWORD n = GetEnvironmentVariableW( L"WindowsSdkDir", sdkRoot, MAX_PATH );
		if ( n == 0 || n >= MAX_PATH )
			wcscpy_s( sdkRoot, L"C:\\Program Files (x86)\\Windows Kits\\10\\" );
		const wchar_t *vers[] = { L"10.0.26100.0", L"10.0.22621.0", L"10.0.22000.0", L"10.0.19041.0" };
		for ( int v = 0; v < 4 && !g_hDxc; ++v )
		{
			wchar_t path[MAX_PATH];
			_snwprintf_s( path, _TRUNCATE, L"%sbin\\%s\\x64\\dxcompiler.dll", sdkRoot, vers[v] );
			g_hDxc = LoadLibraryW( path );
		}
	}

	if ( !g_hDxc )
	{
		Warning( "[PathTrace-DXR] dxcompiler.dll not found; cannot compile RayQuery CS.\n" );
		return false;
	}

	if ( !g_hDxil )
	{
		g_hDxil = LoadLibraryW( L"dxil.dll" );
		if ( !g_hDxil )
		{
			wchar_t dxcPath[MAX_PATH] = {};
			GetModuleFileNameW( g_hDxc, dxcPath, MAX_PATH );
			wchar_t *slash = wcsrchr( dxcPath, L'\\' );
			if ( slash )
			{
				wcscpy_s( slash + 1, MAX_PATH - (int)( slash + 1 - dxcPath ), L"dxil.dll" );
				g_hDxil = LoadLibraryW( dxcPath );
			}
		}
	}

	DxcCreateInstanceProc createFn = (DxcCreateInstanceProc)GetProcAddress( g_hDxc, "DxcCreateInstance" );
	if ( !createFn )
	{
		Warning( "[PathTrace-DXR] DxcCreateInstance missing.\n" );
		return false;
	}

	ComPtr<IDxcUtils> utils;
	ComPtr<IDxcCompiler3> compiler;
	if ( FAILED( createFn( CLSID_DxcUtils, IID_PPV_ARGS( &utils ) ) ) ||
		 FAILED( createFn( CLSID_DxcCompiler, IID_PPV_ARGS( &compiler ) ) ) )
	{
		Warning( "[PathTrace-DXR] Failed to create DXC objects.\n" );
		return false;
	}

	const UINT32 size = (UINT32)strlen( kPtRayQueryCS );
	DxcBuffer srcBuf = {};
	srcBuf.Ptr = kPtRayQueryCS;
	srcBuf.Size = size;
	srcBuf.Encoding = DXC_CP_UTF8;

	LPCWSTR args[] = {
		L"-T", L"cs_6_5",
		L"-E", L"CSMain",
		L"-HV", L"2021",
		L"-Wno-ignored-attributes",
	};

	ComPtr<IDxcResult> result;
	HRESULT hr = compiler->Compile( &srcBuf, args, _countof( args ), nullptr, IID_PPV_ARGS( &result ) );
	if ( FAILED( hr ) || !result )
	{
		Warning( "[PathTrace-DXR] DXC Compile call failed.\n" );
		return false;
	}

	HRESULT status = S_OK;
	result->GetStatus( &status );
	if ( FAILED( status ) )
	{
		ComPtr<IDxcBlobEncoding> errors;
		result->GetErrorBuffer( &errors );
		if ( errors && errors->GetBufferPointer() )
			Warning( "[PathTrace-DXR] Shader compile error:\n%s\n", (const char *)errors->GetBufferPointer() );
		else
			Warning( "[PathTrace-DXR] Shader compile failed (no log).\n" );
		return false;
	}

	ComPtr<IDxcBlob> dxil;
	result->GetResult( &dxil );
	if ( !dxil )
	{
		ComPtr<IDxcBlob> obj;
		ComPtr<IDxcBlobUtf16> name;
		if ( SUCCEEDED( result->GetOutput( DXC_OUT_OBJECT, IID_PPV_ARGS( &obj ), &name ) ) )
			dxil = obj;
	}
	if ( !dxil )
		return false;

	ID3DBlob *blob = nullptr;
	hr = D3DCreateBlob( dxil->GetBufferSize(), &blob );
	if ( FAILED( hr ) || !blob )
		return false;
	memcpy( blob->GetBufferPointer(), dxil->GetBufferPointer(), dxil->GetBufferSize() );
	*ppBlob = blob;
	return true;
}

static bool PtEnsureRayCapacity( uint32 nRays )
{
	if ( nRays <= g_ptDev.rayCapacity && g_ptDev.rayDefault )
		return true;

	g_ptDev.rayUpload.Reset();
	g_ptDev.rayDefault.Reset();
	g_ptDev.hitDefault.Reset();
	g_ptDev.hitReadback.Reset();

	const uint32 cap = ( nRays < 4096u ) ? 4096u : nRays;
	const UINT64 rayBytes = (UINT64)cap * 2 * sizeof( float ) * 4; // 2 float4
	const UINT64 hitBytes = (UINT64)cap * sizeof( float ) * 4;

	if ( !PtCreateBuffer( rayBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
						  D3D12_RESOURCE_STATE_GENERIC_READ, &g_ptDev.rayUpload, L"PT_RayUpload" ) )
		return false;
	if ( !PtCreateBuffer( rayBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
						  D3D12_RESOURCE_STATE_COPY_DEST, &g_ptDev.rayDefault, L"PT_Rays" ) )
		return false;
	if ( !PtCreateBuffer( hitBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
						  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &g_ptDev.hitDefault, L"PT_Hits" ) )
		return false;
	if ( !PtCreateBuffer( hitBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
						  D3D12_RESOURCE_STATE_COPY_DEST, &g_ptDev.hitReadback, L"PT_HitReadback" ) )
		return false;

	g_ptDev.rayCapacity = cap;
	return true;
}

bool PathTraceDXR_DeviceInit( int adapterIndex )
{
	PathTraceDXR_DeviceShutdown();

	if ( !g_bPtCaptureDone || g_ptTris.empty() )
	{
		Warning( "[PathTrace-DXR] No captured triangles.\n" );
		return false;
	}

	UINT dxgiFlags = 0;
#if defined( _DEBUG )
	ComPtr<ID3D12Debug> debug;
	if ( SUCCEEDED( D3D12GetDebugInterface( IID_PPV_ARGS( &debug ) ) ) )
	{
		debug->EnableDebugLayer();
		dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
	}
#endif

	ComPtr<IDXGIFactory4> factory;
	if ( FAILED( CreateDXGIFactory2( dxgiFlags, IID_PPV_ARGS( &factory ) ) ) )
	{
		Warning( "[PathTrace-DXR] CreateDXGIFactory2 failed.\n" );
		return false;
	}

	ComPtr<IDXGIAdapter1> adapter;
	ComPtr<IDXGIAdapter1> chosen;
	for ( UINT i = 0; factory->EnumAdapters1( i, &adapter ) != DXGI_ERROR_NOT_FOUND; ++i )
	{
		DXGI_ADAPTER_DESC1 desc = {};
		adapter->GetDesc1( &desc );
		if ( desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE )
			continue;

		ComPtr<ID3D12Device> testDev;
		if ( FAILED( D3D12CreateDevice( adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS( &testDev ) ) ) )
			continue;

		D3D12_FEATURE_DATA_D3D12_OPTIONS5 opt5 = {};
		if ( FAILED( testDev->CheckFeatureSupport( D3D12_FEATURE_D3D12_OPTIONS5, &opt5, sizeof( opt5 ) ) ) )
			continue;
		if ( opt5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED )
			continue;

		if ( adapterIndex >= 0 && (int)i != adapterIndex )
			continue;

		chosen = adapter;
		char name[256];
		WideCharToMultiByte( CP_UTF8, 0, desc.Description, -1, name, sizeof( name ), nullptr, nullptr );
		Msg( "[PathTrace-DXR] Using adapter %u: %s (RT tier %u)\n", i, name, (unsigned)opt5.RaytracingTier );
		break;
	}

	if ( !chosen )
	{
		Warning( "[PathTrace-DXR] No DXR-capable GPU found; falling back to stock radiosity.\n" );
		return false;
	}

	ComPtr<ID3D12Device> deviceBase;
	if ( FAILED( D3D12CreateDevice( chosen.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS( &deviceBase ) ) ) )
	{
		Warning( "[PathTrace-DXR] D3D12CreateDevice failed.\n" );
		return false;
	}
	if ( FAILED( deviceBase.As( &g_ptDev.device ) ) )
	{
		Warning( "[PathTrace-DXR] ID3D12Device5 not available.\n" );
		return false;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS5 opt5 = {};
	g_ptDev.device->CheckFeatureSupport( D3D12_FEATURE_D3D12_OPTIONS5, &opt5, sizeof( opt5 ) );
	if ( opt5.RaytracingTier < D3D12_RAYTRACING_TIER_1_1 )
	{
		// RayQuery needs 1.1; still try — some drivers report 1.0 but support inline
		Msg( "[PathTrace-DXR] Warning: driver reports RT tier %u (RayQuery prefers 1.1).\n",
			 (unsigned)opt5.RaytracingTier );
	}

	D3D12_COMMAND_QUEUE_DESC qd = {};
	qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	if ( FAILED( g_ptDev.device->CreateCommandQueue( &qd, IID_PPV_ARGS( &g_ptDev.queue ) ) ) )
		return false;
	if ( FAILED( g_ptDev.device->CreateCommandAllocator( D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS( &g_ptDev.alloc ) ) ) )
		return false;
	if ( FAILED( g_ptDev.device->CreateCommandList( 0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_ptDev.alloc.Get(), nullptr,
													IID_PPV_ARGS( &g_ptDev.list ) ) ) )
		return false;
	g_ptDev.list->Close();

	if ( FAILED( g_ptDev.device->CreateFence( 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS( &g_ptDev.fence ) ) ) )
		return false;
	g_ptDev.fenceEvent = CreateEventW( nullptr, FALSE, FALSE, nullptr );
	if ( !g_ptDev.fenceEvent )
		return false;

	// ---- Upload vertices + flags ----
	const uint32 nTris = (uint32)g_ptTris.size();
	const UINT64 vbBytes = (UINT64)nTris * 3 * sizeof( float ) * 3;
	const UINT64 flagBytes = (UINT64)nTris * sizeof( uint32 );

	ComPtr<ID3D12Resource> vbUpload, flagUpload;
	if ( !PtCreateBuffer( vbBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
						  D3D12_RESOURCE_STATE_GENERIC_READ, &vbUpload, L"PT_VBUpload" ) )
		return false;
	if ( !PtCreateBuffer( vbBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
						  D3D12_RESOURCE_STATE_COMMON, &g_ptDev.vertexBuffer, L"PT_VB" ) )
		return false;
	if ( !PtCreateBuffer( flagBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
						  D3D12_RESOURCE_STATE_GENERIC_READ, &flagUpload, L"PT_FlagUpload" ) )
		return false;
	// ALLOW_UNORDERED_ACCESS required for ByteAddressBuffer RAW SRV (RayQuery / baker).
	if ( !PtCreateBuffer( flagBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
						  D3D12_RESOURCE_STATE_COMMON, &g_ptDev.flagBuffer, L"PT_Flags" ) )
		return false;

	{
		float *pV = nullptr;
		vbUpload->Map( 0, nullptr, (void **)&pV );
		uint32 *pF = nullptr;
		flagUpload->Map( 0, nullptr, (void **)&pF );
		for ( uint32 i = 0; i < nTris; ++i )
		{
			const PtHostTri &t = g_ptTris[i];
			pV[i * 9 + 0] = t.a.x; pV[i * 9 + 1] = t.a.y; pV[i * 9 + 2] = t.a.z;
			pV[i * 9 + 3] = t.b.x; pV[i * 9 + 4] = t.b.y; pV[i * 9 + 5] = t.b.z;
			pV[i * 9 + 6] = t.c.x; pV[i * 9 + 7] = t.c.y; pV[i * 9 + 8] = t.c.z;
			pF[i] = t.flags;
		}
		vbUpload->Unmap( 0, nullptr );
		flagUpload->Unmap( 0, nullptr );
	}

	g_ptDev.alloc->Reset();
	g_ptDev.list->Reset( g_ptDev.alloc.Get(), nullptr );
	g_ptDev.list->CopyResource( g_ptDev.vertexBuffer.Get(), vbUpload.Get() );
	g_ptDev.list->CopyResource( g_ptDev.flagBuffer.Get(), flagUpload.Get() );

	D3D12_RESOURCE_BARRIER barriers[2] = {};
	barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[0].Transition.pResource = g_ptDev.vertexBuffer.Get();
	barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	barriers[1] = barriers[0];
	barriers[1].Transition.pResource = g_ptDev.flagBuffer.Get();
	g_ptDev.list->ResourceBarrier( 2, barriers );

	// ---- BLAS ----
	D3D12_RAYTRACING_GEOMETRY_DESC geo = {};
	geo.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
	geo.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
	geo.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
	geo.Triangles.VertexCount = nTris * 3;
	geo.Triangles.VertexBuffer.StartAddress = g_ptDev.vertexBuffer->GetGPUVirtualAddress();
	geo.Triangles.VertexBuffer.StrideInBytes = sizeof( float ) * 3;

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasIn = {};
	blasIn.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
	blasIn.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	blasIn.NumDescs = 1;
	blasIn.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	blasIn.pGeometryDescs = &geo;

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPre = {};
	g_ptDev.device->GetRaytracingAccelerationStructurePrebuildInfo( &blasIn, &blasPre );

	ComPtr<ID3D12Resource> blasScratch;
	if ( !PtCreateBuffer( blasPre.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
						  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
						  &blasScratch, L"PT_BLASScratch" ) )
		return false;
	if ( !PtCreateBuffer( blasPre.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
						  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
						  D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
						  &g_ptDev.blas, L"PT_BLAS" ) )
		return false;

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuild = {};
	blasBuild.Inputs = blasIn;
	blasBuild.DestAccelerationStructureData = g_ptDev.blas->GetGPUVirtualAddress();
	blasBuild.ScratchAccelerationStructureData = blasScratch->GetGPUVirtualAddress();
	g_ptDev.list->BuildRaytracingAccelerationStructure( &blasBuild, 0, nullptr );

	D3D12_RESOURCE_BARRIER uavBarrier = {};
	uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	uavBarrier.UAV.pResource = g_ptDev.blas.Get();
	g_ptDev.list->ResourceBarrier( 1, &uavBarrier );

	// ---- TLAS ----
	D3D12_RAYTRACING_INSTANCE_DESC inst = {};
	inst.Transform[0][0] = inst.Transform[1][1] = inst.Transform[2][2] = 1.0f;
	inst.InstanceMask = 0xFF;
	inst.AccelerationStructure = g_ptDev.blas->GetGPUVirtualAddress();

	ComPtr<ID3D12Resource> instUpload;
	if ( !PtCreateBuffer( sizeof( inst ), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
						  D3D12_RESOURCE_STATE_GENERIC_READ, &instUpload, L"PT_InstUpload" ) )
		return false;
	{
		void *p = nullptr;
		instUpload->Map( 0, nullptr, &p );
		memcpy( p, &inst, sizeof( inst ) );
		instUpload->Unmap( 0, nullptr );
	}

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasIn = {};
	tlasIn.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
	tlasIn.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	tlasIn.NumDescs = 1;
	tlasIn.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	tlasIn.InstanceDescs = instUpload->GetGPUVirtualAddress();

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPre = {};
	g_ptDev.device->GetRaytracingAccelerationStructurePrebuildInfo( &tlasIn, &tlasPre );

	ComPtr<ID3D12Resource> tlasScratch;
	if ( !PtCreateBuffer( tlasPre.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
						  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
						  &tlasScratch, L"PT_TLASScratch" ) )
		return false;
	if ( !PtCreateBuffer( tlasPre.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
						  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
						  D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
						  &g_ptDev.tlas, L"PT_TLAS" ) )
		return false;

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuild = {};
	tlasBuild.Inputs = tlasIn;
	tlasBuild.DestAccelerationStructureData = g_ptDev.tlas->GetGPUVirtualAddress();
	tlasBuild.ScratchAccelerationStructureData = tlasScratch->GetGPUVirtualAddress();
	g_ptDev.list->BuildRaytracingAccelerationStructure( &tlasBuild, 0, nullptr );

	uavBarrier.UAV.pResource = g_ptDev.tlas.Get();
	g_ptDev.list->ResourceBarrier( 1, &uavBarrier );

	g_ptDev.list->Close();
	ID3D12CommandList *lists[] = { g_ptDev.list.Get() };
	g_ptDev.queue->ExecuteCommandLists( 1, lists );
	PtWaitGPU();
	g_ptDev.tlasVA = g_ptDev.tlas->GetGPUVirtualAddress();

	// Keep upload alive until GPU done — already waited.

	// ---- Root signature + PSO ----
	D3D12_DESCRIPTOR_RANGE ranges[3] = {};
	ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; // t0 TLAS
	ranges[0].NumDescriptors = 1;
	ranges[0].BaseShaderRegister = 0;
	ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
	ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; // t1 rays, t2 flags
	ranges[1].NumDescriptors = 2;
	ranges[1].BaseShaderRegister = 1;
	ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
	ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; // u0 hits
	ranges[2].NumDescriptors = 1;
	ranges[2].BaseShaderRegister = 0;
	ranges[2].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

	D3D12_ROOT_PARAMETER params[2] = {};
	params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	params[0].DescriptorTable.NumDescriptorRanges = 3;
	params[0].DescriptorTable.pDescriptorRanges = ranges;
	params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	params[1].Constants.Num32BitValues = 4;
	params[1].Constants.ShaderRegister = 0;
	params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
	rsDesc.NumParameters = 2;
	rsDesc.pParameters = params;
	rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

	ComPtr<ID3DBlob> rsBlob, rsErr;
	if ( FAILED( D3D12SerializeRootSignature( &rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr ) ) )
	{
		Warning( "[PathTrace-DXR] SerializeRootSignature failed.\n" );
		return false;
	}
	if ( FAILED( g_ptDev.device->CreateRootSignature( 0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
													  IID_PPV_ARGS( &g_ptDev.rootSig ) ) ) )
		return false;

	ID3DBlob *csBlob = nullptr;
	if ( !PtCompileCS( &csBlob ) )
		return false;

	D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = g_ptDev.rootSig.Get();
	psoDesc.CS.pShaderBytecode = csBlob->GetBufferPointer();
	psoDesc.CS.BytecodeLength = csBlob->GetBufferSize();
	HRESULT hrPso = g_ptDev.device->CreateComputePipelineState( &psoDesc, IID_PPV_ARGS( &g_ptDev.pso ) );
	csBlob->Release();
	if ( FAILED( hrPso ) )
	{
		Warning( "[PathTrace-DXR] CreateComputePipelineState failed (0x%08X). Need SM 6.5 / DXR 1.1.\n", (unsigned)hrPso );
		return false;
	}

	// Descriptor heap: 0=TLAS, 1=rays, 2=flags, 3=hits UAV — rebuilt each dispatch for rays/hits
	D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heapDesc.NumDescriptors = 4;
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if ( FAILED( g_ptDev.device->CreateDescriptorHeap( &heapDesc, IID_PPV_ARGS( &g_ptDev.srvUavHeap ) ) ) )
		return false;

	const UINT incr = g_ptDev.device->GetDescriptorHandleIncrementSize( D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV );
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_ptDev.srvUavHeap->GetCPUDescriptorHandleForHeapStart();

	D3D12_SHADER_RESOURCE_VIEW_DESC tlasSrv = {};
	tlasSrv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
	tlasSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	tlasSrv.RaytracingAccelerationStructure.Location = g_ptDev.tlasVA;
	g_ptDev.device->CreateShaderResourceView( nullptr, &tlasSrv, cpu );

	g_ptDev.tlasSrvGpu = g_ptDev.srvUavHeap->GetGPUDescriptorHandleForHeapStart();

	if ( !PtEnsureRayCapacity( 8192 ) )
		return false;

	g_ptDev.ready = true;
	Msg( "[PathTrace-DXR] Device ready (%u tris). Large ray batches use DXR RayQuery; tiny batches stay on SSE Trace4Rays.\n", nTris );
	return true;
}

void PathTraceDXR_DeviceShutdown()
{
	PathTraceDXR_GpuBakeEnd();

	if ( g_ptDev.queue && g_ptDev.fence && g_ptDev.fenceEvent )
		PtWaitGPU();

	g_ptDev.ready = false;
	g_ptDev.pso.Reset();
	g_ptDev.rootSig.Reset();
	g_ptDev.srvUavHeap.Reset();
	g_ptDev.rayUpload.Reset();
	g_ptDev.rayDefault.Reset();
	g_ptDev.hitDefault.Reset();
	g_ptDev.hitReadback.Reset();
	g_ptDev.cbUpload.Reset();
	g_ptDev.blas.Reset();
	g_ptDev.tlas.Reset();
	g_ptDev.vertexBuffer.Reset();
	g_ptDev.flagBuffer.Reset();
	g_ptDev.list.Reset();
	g_ptDev.alloc.Reset();
	g_ptDev.queue.Reset();
	g_ptDev.fence.Reset();
	g_ptDev.device.Reset();
	g_ptDev.rayCapacity = 0;
	if ( g_ptDev.fenceEvent )
	{
		CloseHandle( g_ptDev.fenceEvent );
		g_ptDev.fenceEvent = nullptr;
	}
	g_ptDev.fenceValue = 0;
}

bool PathTraceDXR_DeviceReady()
{
	return g_ptDev.ready;
}

// Path integration issues millions of rays. One DXR dispatch per ray under a global
// mutex serializes all worker threads onto the GPU and looks "stuck". Use the same
// threaded SSE KD-tree as stock VRAD (g_RtEnv) for queries; DXR AS remains built for
// future large-batch acceleration.
static bool PathTraceDXR_TraceClosest_CPU( const Vector *origins, const Vector *dirs,
										   const float *tmins, const float *tmaxs,
										   float *outT, uint32 *outFlags, uint32 *outHit,
										   Vector *outNormal, int nRays )
{
	for ( int base = 0; base < nRays; base += 4 )
	{
		const int count = min( 4, nRays - base );
		FourRays myrays;
		float tminLocal[4];
		float tmaxLocal[4];

		for ( int i = 0; i < 4; ++i )
		{
			const int src = base + ( ( i < count ) ? i : ( count - 1 ) );
			Vector o = origins[src];
			Vector d = dirs[src];
			float tmin = tmins ? tmins[src] : 0.01f;
			float tmax = tmaxs ? tmaxs[src] : (float)MAX_TRACE_LENGTH;
			if ( tmin < 0.0f )
				tmin = 0.0f;
			if ( tmax < tmin )
				tmax = tmin;

			// Degenerate / non-finite rays → force a miss (avoids KD AV / NaN storms).
			const float d2 = d.x * d.x + d.y * d.y + d.z * d.z;
			if ( d2 < 1e-20f || !_finite( d.x ) || !_finite( d.y ) || !_finite( d.z ) ||
				 !_finite( o.x ) || !_finite( o.y ) || !_finite( o.z ) || ( tmax - tmin ) < 1e-8f )
			{
				tminLocal[i] = tmin;
				tmaxLocal[i] = 0.0f;
				myrays.origin.X( i ) = o.x;
				myrays.origin.Y( i ) = o.y;
				myrays.origin.Z( i ) = o.z;
				myrays.direction.X( i ) = 0.0f;
				myrays.direction.Y( i ) = 0.0f;
				myrays.direction.Z( i ) = 1.0f;
				continue;
			}

			// Advance origin by tmin so Trace4Rays can use TMin=0.
			o.x += d.x * tmin;
			o.y += d.y * tmin;
			o.z += d.z * tmin;
			tmaxLocal[i] = tmax - tmin;
			tminLocal[i] = tmin;

			myrays.origin.X( i ) = o.x;
			myrays.origin.Y( i ) = o.y;
			myrays.origin.Z( i ) = o.z;
			myrays.direction.X( i ) = d.x;
			myrays.direction.Y( i ) = d.y;
			myrays.direction.Z( i ) = d.z;
		}

		fltx4 len = LoadUnalignedSIMD( tmaxLocal );
		RayTracingResult rt_result;
		g_RtEnv.Trace4Rays( myrays, Four_Zeros, len, &rt_result, TRACE_ID_STATICPROP | (unsigned)-1, nullptr );

		const int nTris = g_RtEnv.OptimizedTriangleList.Count();
		for ( int i = 0; i < count; ++i )
		{
			const int dst = base + i;
			const float hitT = rt_result.HitDistance.m128_f32[i];
			const int tri = rt_result.HitIds[i];
			if ( tri >= 0 && tri < nTris && hitT < tmaxLocal[i] && tmaxLocal[i] > 0.0f )
			{
				outT[dst] = hitT + tminLocal[i];
				outFlags[dst] = (uint32)g_RtEnv.OptimizedTriangleList[tri].m_Data.m_IntersectData.m_nTriangleID;
				outHit[dst] = 1;
				if ( outNormal )
				{
					Vector n;
					n.x = rt_result.surface_normal.X( i );
					n.y = rt_result.surface_normal.Y( i );
					n.z = rt_result.surface_normal.Z( i );
					VectorNormalize( n );
					// Face the ray origin (two-sided lightmaps).
					const Vector &d = dirs[dst];
					if ( DotProduct( n, d ) > 0.0f )
						n = -n;
					outNormal[dst] = n;
				}
			}
			else
			{
				outT[dst] = -1.0f;
				outFlags[dst] = 0;
				outHit[dst] = 0;
				if ( outNormal )
					outNormal[dst].Init( 0, 0, 1 );
			}
		}
	}
	return true;
}

static const uint32 kPtGpuMinRays = 64;

static void PtFillHitNormal( uint32 prim, const Vector &dir, Vector &outN )
{
	outN.Init( 0, 0, 1 );
	if ( prim >= (uint32)g_ptTris.size() )
		return;
	const PtHostTri &t = g_ptTris[prim];
	Vector e1 = t.b - t.a;
	Vector e2 = t.c - t.a;
	CrossProduct( e1, e2, outN );
	VectorNormalize( outN );
	if ( DotProduct( outN, dir ) > 0.0f )
		outN = -outN;
}

static bool PathTraceDXR_TraceClosest_GPU( const Vector *origins, const Vector *dirs,
										   const float *tmins, const float *tmaxs,
										   float *outT, uint32 *outFlags, uint32 *outHit,
										   Vector *outNormal, int nRays )
{
	if ( !g_ptDev.ready || !g_ptDev.pso || nRays <= 0 )
		return false;

	std::lock_guard<std::mutex> lock( g_ptDev.mutex );

	if ( !PtEnsureRayCapacity( (uint32)nRays ) )
		return false;

	{
		float *p = nullptr;
		if ( FAILED( g_ptDev.rayUpload->Map( 0, nullptr, (void **)&p ) ) )
			return false;
		for ( int i = 0; i < nRays; ++i )
		{
			float tmin = tmins ? tmins[i] : 0.01f;
			float tmax = tmaxs ? tmaxs[i] : (float)MAX_TRACE_LENGTH;
			if ( tmin < 0.0f ) tmin = 0.0f;
			if ( tmax < tmin ) tmax = tmin;
			p[i * 8 + 0] = origins[i].x;
			p[i * 8 + 1] = origins[i].y;
			p[i * 8 + 2] = origins[i].z;
			p[i * 8 + 3] = tmin;
			p[i * 8 + 4] = dirs[i].x;
			p[i * 8 + 5] = dirs[i].y;
			p[i * 8 + 6] = dirs[i].z;
			p[i * 8 + 7] = tmax;
		}
		g_ptDev.rayUpload->Unmap( 0, nullptr );
	}

	g_ptDev.alloc->Reset();
	g_ptDev.list->Reset( g_ptDev.alloc.Get(), nullptr );

	g_ptDev.list->CopyResource( g_ptDev.rayDefault.Get(), g_ptDev.rayUpload.Get() );

	D3D12_RESOURCE_BARRIER b0 = {};
	b0.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b0.Transition.pResource = g_ptDev.rayDefault.Get();
	b0.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	b0.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	b0.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	g_ptDev.list->ResourceBarrier( 1, &b0 );

	const UINT incr = g_ptDev.device->GetDescriptorHandleIncrementSize( D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV );
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_ptDev.srvUavHeap->GetCPUDescriptorHandleForHeapStart();

	// slot0 TLAS already set at init; refresh rays/flags/hits each dispatch
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC raySrv = {};
		raySrv.Format = DXGI_FORMAT_UNKNOWN;
		raySrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		raySrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		raySrv.Buffer.FirstElement = 0;
		raySrv.Buffer.NumElements = g_ptDev.rayCapacity * 2;
		raySrv.Buffer.StructureByteStride = sizeof( float ) * 4;
		D3D12_CPU_DESCRIPTOR_HANDLE c1 = cpu;
		c1.ptr += incr;
		g_ptDev.device->CreateShaderResourceView( g_ptDev.rayDefault.Get(), &raySrv, c1 );

		D3D12_SHADER_RESOURCE_VIEW_DESC flagSrv = {};
		flagSrv.Format = DXGI_FORMAT_R32_TYPELESS;
		flagSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		flagSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		flagSrv.Buffer.FirstElement = 0;
		flagSrv.Buffer.NumElements = (UINT)g_ptTris.size();
		flagSrv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
		D3D12_CPU_DESCRIPTOR_HANDLE c2 = cpu;
		c2.ptr += incr * 2;
		g_ptDev.device->CreateShaderResourceView( g_ptDev.flagBuffer.Get(), &flagSrv, c2 );

		D3D12_UNORDERED_ACCESS_VIEW_DESC hitUav = {};
		hitUav.Format = DXGI_FORMAT_UNKNOWN;
		hitUav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		hitUav.Buffer.FirstElement = 0;
		hitUav.Buffer.NumElements = g_ptDev.rayCapacity;
		hitUav.Buffer.StructureByteStride = sizeof( float ) * 4;
		D3D12_CPU_DESCRIPTOR_HANDLE c3 = cpu;
		c3.ptr += incr * 3;
		g_ptDev.device->CreateUnorderedAccessView( g_ptDev.hitDefault.Get(), nullptr, &hitUav, c3 );
	}

	ID3D12DescriptorHeap *heaps[] = { g_ptDev.srvUavHeap.Get() };
	g_ptDev.list->SetDescriptorHeaps( 1, heaps );
	g_ptDev.list->SetComputeRootSignature( g_ptDev.rootSig.Get() );
	g_ptDev.list->SetPipelineState( g_ptDev.pso.Get() );
	g_ptDev.list->SetComputeRootDescriptorTable( 0, g_ptDev.tlasSrvGpu );
	UINT consts[4] = { (UINT)nRays, 0, 0, 0 };
	g_ptDev.list->SetComputeRoot32BitConstants( 1, 4, consts, 0 );
	g_ptDev.list->Dispatch( ( nRays + 63 ) / 64, 1, 1 );

	D3D12_RESOURCE_BARRIER uavB = {};
	uavB.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	uavB.UAV.pResource = g_ptDev.hitDefault.Get();
	g_ptDev.list->ResourceBarrier( 1, &uavB );

	D3D12_RESOURCE_BARRIER b1 = {};
	b1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b1.Transition.pResource = g_ptDev.hitDefault.Get();
	b1.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	b1.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	b1.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	g_ptDev.list->ResourceBarrier( 1, &b1 );
	g_ptDev.list->CopyResource( g_ptDev.hitReadback.Get(), g_ptDev.hitDefault.Get() );

	// restore hit buffer to UAV for next dispatch
	D3D12_RESOURCE_BARRIER b2 = b1;
	b2.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
	b2.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	g_ptDev.list->ResourceBarrier( 1, &b2 );

	// restore rays to COPY_DEST
	D3D12_RESOURCE_BARRIER b3 = b0;
	b3.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	b3.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
	g_ptDev.list->ResourceBarrier( 1, &b3 );

	g_ptDev.list->Close();
	ID3D12CommandList *lists[] = { g_ptDev.list.Get() };
	g_ptDev.queue->ExecuteCommandLists( 1, lists );
	PtWaitGPU();

	{
		float *p = nullptr;
		D3D12_RANGE range = { 0, (SIZE_T)nRays * sizeof( float ) * 4 };
		if ( FAILED( g_ptDev.hitReadback->Map( 0, &range, (void **)&p ) ) )
			return false;
		for ( int i = 0; i < nRays; ++i )
		{
			const float t = p[i * 4 + 0];
			const uint32 flags = *(const uint32 *)&p[i * 4 + 1];
			const uint32 prim = *(const uint32 *)&p[i * 4 + 2];
			const uint32 hit = *(const uint32 *)&p[i * 4 + 3];
			outT[i] = t;
			outFlags[i] = flags;
			outHit[i] = hit;
			if ( outNormal )
			{
				if ( hit )
					PtFillHitNormal( prim, dirs[i], outNormal[i] );
				else
					outNormal[i].Init( 0, 0, 1 );
			}
		}
		D3D12_RANGE empty = {};
		g_ptDev.hitReadback->Unmap( 0, &empty );
	}
	return true;
}


bool PathTraceDXR_TraceClosest( const Vector *origins, const Vector *dirs, const float *tmins, const float *tmaxs,
								float *outT, uint32 *outFlags, uint32 *outHit, int nRays, Vector *outNormal )
{
	if ( nRays <= 0 )
		return false;
	// Allow CPU traces once the stock KD-tree exists (DeviceInit may still be warming DXR).
	if ( g_RtEnv.OptimizedTriangleList.Count() <= 0 )
		return false;

	// Large batches: DXR RayQuery (Frostbite/Bakery style). Tiny batches: SSE (avoids submit overhead).
	if ( g_ptDev.ready && nRays >= (int)kPtGpuMinRays )
	{
		if ( PathTraceDXR_TraceClosest_GPU( origins, dirs, tmins, tmaxs, outT, outFlags, outHit, outNormal, nRays ) )
			return true;
	}
	return PathTraceDXR_TraceClosest_CPU( origins, dirs, tmins, tmaxs, outT, outFlags, outHit, outNormal, nRays );
}

#include "pathtrace_dxr_gpu_bake.inl"
