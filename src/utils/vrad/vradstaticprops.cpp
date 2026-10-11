//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
// $Revision: $
// $NoKeywords: $
//
// This file contains code to allow us to associate client data with bsp leaves.
//
//=============================================================================//

#include "vrad.h"
#include "light_projection.h"
#include "mathlib/vector.h"
#include "UtlBuffer.h"
#include "utlvector.h"
#include "GameBSPFile.h"
#include "BSPTreeData.h"
#include "VPhysics_Interface.h"
#include "Studio.h"
#include "Optimize.h"
#include "Bsplib.h"
#include "CModel.h"
#include "PhysDll.h"
#include "phyfile.h"
#include "collisionutils.h"
#include "tier1/KeyValues.h"
#include "pacifier.h"
#include "materialsystem/imaterial.h"
#include "materialsystem/hardwareverts.h"
#include "materialsystem/hardwaretexels.h"
#include "byteswap.h"
#include "mpivrad.h"
#include "vtf/vtf.h"
#include "vtf_sdk_compat.h"
#include "bake_volume.h"
#include "tier1/utldict.h"
#include "tier1/utlsymbol.h"
#include "bitmap/tgawriter.h"

#include "prop_uv_split.h"
#include "pathtrace_dxr.h"
#include "pathtrace_dxr_device.h"
#include "bounce_albedo.h"
#include "messbuf.h"
#include "vmpi.h"
#include "vmpi_distribute_work.h"


#define ALIGN_TO_POW2(x,y) (((x)+(y-1))&~(y-1))

// identifies a vertex embedded in solid
// lighting will be copied from nearest valid neighbor
struct badVertex_t
{
	int		m_ColorVertex;
	Vector	m_Position;
	Vector	m_Normal;
};

// a final colored vertex
struct colorVertex_t
{
	Vector	m_Color;
	Vector	m_Position;
	bool	m_bValid;
};

// a texel suitable for a model
struct colorTexel_t
{
	Vector		m_Color;
	Vector		m_WorldPosition;
	Vector		m_WorldNormal;
	float		m_fDistanceToTri; // If we are outside of the triangle, how far away is it?
	bool		m_bValid;
	bool		m_bPossiblyInteresting;
	Vector		m_AxisU;		// world/model units per lightmap texel
	Vector		m_AxisV;
	float		m_LuxelWorld;	// texel footprint (0 = unknown)

};

class CComputeStaticPropLightingResults
{
public:
	~CComputeStaticPropLightingResults()
	{
		m_ColorVertsArrays.PurgeAndDeleteElements();
		m_ColorTexelsArrays.PurgeAndDeleteElements();
		m_TexelColors.PurgeAndDeleteElements();
		m_TexelFlags.PurgeAndDeleteElements();
	}
	
	CUtlVector< CUtlVector<colorVertex_t>* > m_ColorVertsArrays;
	CUtlVector< CUtlVector<colorTexel_t>* > m_ColorTexelsArrays;
	// Compact bake store. The fat colorTexel_t grid (positions, normals, axes)
	// is rebuilt from the shared template for one prop at a time, then freed.
	// Flag bits: 1 = valid, 2 = possibly interesting.
	CUtlVector< CUtlVector<Vector>* > m_TexelColors;
	CUtlVector< CUtlVector<unsigned char>* > m_TexelFlags;
};

//-----------------------------------------------------------------------------
struct Rasterizer
{
	struct Location
	{
		Vector barycentric;
		Vector2D uv;
		bool   insideTriangle;
	};

	Rasterizer(Vector2D t0, Vector2D t1, Vector2D t2, size_t resX, size_t resY)
	: mT0(t0)
	, mT1(t1)
	, mT2(t2)
	, mResX(resX)
	, mResY(resY)
	, mUvStepX(1.0f / resX)
	, mUvStepY(1.0f / resY)
	{ 
		Build();
	}

	CUtlVector< Location >::iterator begin() { return mRasterizedLocations.begin(); }
	CUtlVector< Location >::iterator end() { return mRasterizedLocations.end(); }

	void Build();

	inline size_t GetRow(float y) const { return size_t(y * mResY); }
	inline size_t GetCol(float x) const { return size_t(x * mResX); }

	inline size_t GetLinearPos( const CUtlVector< Location >::iterator& it ) const
	{
		// Given an iterator, return what the linear position in the buffer would be for the data.
		return (size_t)(GetRow(it->uv.y) * mResX)
			 + (size_t)(GetCol(it->uv.x));
	}
	
private:
	const Vector2D mT0, mT1, mT2;
	const size_t mResX, mResY;
	const float mUvStepX, mUvStepY;

	// Right now, we just fill this out and directly iterate over it. 
	// It could be large. This is a memory/speed tradeoff. We could instead generate them
	// on demand. 
	CUtlVector< Location > mRasterizedLocations;
};

//-----------------------------------------------------------------------------
inline Vector ComputeBarycentric( Vector2D _edgeC, Vector2D _edgeA, Vector2D _edgeB, float _dAA, float _dAB, float _dBB, float _invDenom )
{
	float dCA = _edgeC.Dot(_edgeA);
	float dCB = _edgeC.Dot(_edgeB);
	
	Vector retVal;
	retVal.y = (_dBB * dCA - _dAB * dCB) * _invDenom;
	retVal.z = (_dAA * dCB - _dAB * dCA) * _invDenom;
	retVal.x = 1.0f - retVal.y - retVal.z;

	return retVal;
}

//-----------------------------------------------------------------------------
void Rasterizer::Build()
{
	// For now, use the barycentric method. It's easy, I'm lazy. 
	// We can optimize later if it's a performance issue.
	const float baseX = mUvStepX / 2.0f;
	const float baseY = mUvStepY / 2.0f;


	float fMinX = min(min(mT0.x, mT1.x), mT2.x);
	float fMinY = min(min(mT0.y, mT1.y), mT2.y);
	float fMaxX = max(max(mT0.x, mT1.x), mT2.x);
	float fMaxY = max(max(mT0.y, mT1.y), mT2.y);

	// Degenerate. Consider warning about these, but otherwise no problem.
	if (fMinX == fMaxX || fMinY == fMaxY)
		return;

	// Clamp to 0..1
	fMinX = max(0, fMinX);
	fMinY = max(0, fMinY);
	fMaxX = min(1.0f, fMaxX);
	fMaxY = min(1.0f, fMaxY);

	// We puff the interesting area up by 1 so we can hit an inflated region for the necessary bilerp data.
	// If we wanted to support better texturing (almost definitely unnecessary), we'd change this to a larger size.
	const int kFilterSampleRadius = 1;

	int iMinX = GetCol(fMinX) - kFilterSampleRadius;
	int iMinY = GetRow(fMinY) - kFilterSampleRadius;
	int iMaxX = GetCol(fMaxX) + 1 + kFilterSampleRadius;
	int iMaxY = GetRow(fMaxY) + 1 + kFilterSampleRadius;

	// Clamp to valid texture (integer) locations
	iMinX = max(0, iMinX);
	iMinY = max(0, iMinY);
	iMaxX = min(iMaxX, mResX - 1);
	iMaxY = min(iMaxY, mResY - 1);

	// Set the size to be as expected. 
	// TODO: Pass this in from outside to minimize allocations
	int count = (iMaxY - iMinY + 1) 
		      * (iMaxX - iMinX + 1);
	mRasterizedLocations.EnsureCount(count);
	memset( mRasterizedLocations.Base(), 0, mRasterizedLocations.Count() * sizeof( Location ) );
	
	// Computing Barycentrics adapted from here http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
	Vector2D edgeA = mT1 - mT0;
	Vector2D edgeB = mT2 - mT0;

	float dAA = edgeA.Dot(edgeA);
	float dAB = edgeA.Dot(edgeB);
	float dBB = edgeB.Dot(edgeB);
	float invDenom = 1.0f / (dAA * dBB - dAB * dAB);

	int linearPos = 0; 
	for (int j = iMinY; j <= iMaxY; ++j) {
		for (int i = iMinX; i <= iMaxX; ++i) {
			Vector2D testPt( i * mUvStepX + baseX, j * mUvStepY + baseY );
			Vector barycentric = ComputeBarycentric( testPt - mT0, edgeA, edgeB, dAA, dAB, dBB, invDenom );

			// Test whether the point is inside the triangle. 
			// MCJOHNTODO: Edge rules and whatnot--right now we re-rasterize points on the edge.
			Location& newLoc = mRasterizedLocations[linearPos++];
			newLoc.barycentric = barycentric;
			newLoc.uv = testPt;

			newLoc.insideTriangle = (barycentric.x >= 0.0f && barycentric.x <= 1.0f && barycentric.y >= 0.0f && barycentric.y <= 1.0f && barycentric.z >= 0.0f && barycentric.z <= 1.0f);
		}
	}
}


//-----------------------------------------------------------------------------
// Globals
//-----------------------------------------------------------------------------
CUtlSymbolTable g_ForcedTextureShadowsModels;

// DON'T USE THIS FROM WITHIN A THREAD.  THERE IS A THREAD CONTEXT CREATED 
// INSIDE PropTested_t.  USE THAT INSTEAD.
IPhysicsCollision *s_pPhysCollision = NULL;

static void ConvertTexelDataToTexture(unsigned int _resX, unsigned int _resY, ImageFormat _destFmt, const CUtlVector<colorTexel_t>& _srcTexels, CUtlMemory<byte>* _outTexture);

// Such a monstrosity. :(
static void GenerateLightmapSamplesForMesh( const matrix3x4_t& _matPos, const matrix3x4_t& _matNormal, int _iThread, int _skipProp, int _nFlags, int _lightmapResX, int _lightmapResY, 
											studiohdr_t* _pStudioHdr, mstudiomodel_t* _pStudioModel, OptimizedModel::ModelHeader_t* _pVtxModel, int _meshID, 
											CComputeStaticPropLightingResults *_pResults, bool _bComputeLighting = true,
											const Vector *_pPropOrigin = NULL, const char *_pPropModel = NULL,
											bool _bResetTexels = true,
											const PropUvSplitPlan *_pUvPlan = NULL, int _uvBin = -1, int _uvBody = 0, int _uvModel = 0 );

// Debug function, converts lightmaps to linear space then dumps them out. 
// TODO: Write out the file in a .dds instead of a .tga, in whatever format we're supposed to use.
static void DumpLightmapLinear( const char* _dstFilename, const CUtlVector<colorTexel_t>& _srcTexels, int _width, int _height );


//-----------------------------------------------------------------------------
// Vrad's static prop manager
//-----------------------------------------------------------------------------

class CVradStaticPropMgr : public IVradStaticPropMgr
{
public:
	// constructor, destructor
	CVradStaticPropMgr();
	virtual ~CVradStaticPropMgr();

	// methods of IStaticPropMgr
	void Init();
	void Shutdown();

	// iterate all the instanced static props and compute their vertex lighting
	void ComputeLighting( int iThread );
	void AddAmbientSamplePositions( int leafIndex, CUtlVector<Vector> &positions, int maxPositions ) const;
	bool IsPointInsideStaticProp( int iThread, int leafIndex, const Vector &position ) const;

private:
	// VMPI stuff.
#ifdef MPI
	static void VMPI_ProcessStaticProp_Static( int iThread, uint64 iStaticProp, MessageBuffer *pBuf );
	static void VMPI_ReceiveStaticPropResults_Static( uint64 iStaticProp, MessageBuffer *pBuf, int iWorker );
	void VMPI_ProcessStaticProp( int iThread, int iStaticProp, MessageBuffer *pBuf );
	void VMPI_ReceiveStaticPropResults( int iStaticProp, MessageBuffer *pBuf, int iWorker );
#endif
	
	// local thread version
	static void ThreadComputeStaticPropLighting( int iThread, void *pUserData );
	void ComputeLightingForProp( int iThread, int iStaticProp );

	// Methods associated with unserializing static props
	void UnserializeModelDict( CUtlBuffer& buf );
	void UnserializeModels( CUtlBuffer& buf );
	void UnserializeStaticProps();

	// Creates a collision model
	void CreateCollisionModel( char const* pModelName );

private:
	// Unique static prop models
	struct StaticPropDict_t
	{
		vcollide_t		m_loadedModel;
		CPhysCollide*	m_pModel;
		Vector			m_Mins;			// Bounding box is in local coordinates
		Vector			m_Maxs;
		studiohdr_t*	m_pStudioHdr;
		CUtlBuffer		m_VtxBuf;
		char			m_Name[STATIC_PROP_NAME_LENGTH];
		CUtlVector<int>	m_textureShadowIndex;	// each texture has an index if this model casts texture shadows
		CUtlVector<int>	m_triangleMaterialIndex;// each triangle has an index if this model casts texture shadows
	};

	struct MeshData_t
	{
		CUtlVector<Vector>	m_VertexColors;
		CUtlVector<byte>	m_KeepCached;	// 1 = write prior VHV bytes (outside bake_volume)
		CUtlMemory<byte>	m_TexelsEncoded;
		int					m_nLod;
	};

	// A static prop instance
	struct CStaticProp
	{
		Vector					m_Origin;
		QAngle					m_Angles;
		Vector					m_mins;
		Vector					m_maxs;
		Vector					m_LightingOrigin;
		int						m_ModelIdx;
		unsigned short			m_FirstLeaf;
		unsigned short			m_LeafCount;
		unsigned char			m_Solid;
		BSPTreeDataHandle_t		m_Handle;
		CUtlVector<MeshData_t>	m_MeshData;
		int                     m_Flags;
		bool					m_bLightingOriginValid;

		// Note that all lightmaps for a given prop share the same resolution (and format)--and there can be multiple lightmaps
		// per prop (if there are multiple pieces--the watercooler is an example).
		// This is effectively because there's not a good way in hammer for a prop to say "this should be the resolution
		// of each of my sub-pieces."
		ImageFormat				m_LightmapImageFormat;
		unsigned int			m_LightmapImageWidth;
		unsigned int			m_LightmapImageHeight;
		// -1 keeps every material. Otherwise this instance is one material of a
		// lightmapped prop that was split so the engine's single prop lightmap
		// does not overlap the other materials' UVs.
		int						m_OnlyMaterial;
		// -1 keeps every UV island. Otherwise this instance keeps one bin of a
		// plan that packs overlapping islands into as few models as possible.
		int						m_UvBin = -1;
		int						m_UvPlan = -1;
		int						m_EmitModelIdx;		// dictionary slot written into the BSP; -1 = m_ModelIdx
		int						m_ShadowSkipProp;	// ray-trace prop to skip for self-shadow; -1 = this prop
		int						m_Skin;
		float					m_FadeMinDist;
		float					m_FadeMaxDist;
		float					m_flForcedFadeScale;
		unsigned short			m_nMinDXLevel;
		unsigned short			m_nMaxDXLevel;

	};

	struct LightmapMaterialModel_t
	{
		int m_SrcDict = -1;
		int m_Material = -1;
		int m_UvBin = -1;
		int m_UvPlan = -1;
		int m_EmitDict = -1;
	};

	// Enumeration context
	struct EnumContext_t
	{
		PropTested_t* m_pPropTested;
		Ray_t const* m_pRay;
	};

	// The list of all static props
	CUtlVector <StaticPropDict_t>	m_StaticPropDict;
	CUtlVector <CStaticProp>		m_StaticProps;
	CUtlVector<StaticPropLeafLump_t> m_StaticPropLeaves;
	CUtlVector< CUtlVector<int> >	m_LeafStaticProps;
	CUtlVector<LightmapMaterialModel_t> m_LightmapMaterialModels;
	CUtlVector<PropUvSplitPlan> m_UvPlans;

	bool m_bIgnoreStaticPropTrace;

	void ComputeLighting( CStaticProp &prop, int iThread, int prop_index, CComputeStaticPropLightingResults *pResults );
	bool ComputeLightingPathTraceGPU();
	void ApplyLightingToStaticProp( int iStaticProp, CStaticProp &prop, const CComputeStaticPropLightingResults *pResults );

	void SerializeLighting();
	void SplitLightmappedMaterialProps();
	void SplitLightmappedUvProps();
	void PrintLightmapSplitStats( int nOriginalProps, int nOriginalModels );
	void WriteStaticPropGameLump();
	int EnsureLightmapMaterialModel( int srcDict, int material, int uvBin = -1, int uvPlan = -1 );
	int FindLightmapEmit( int srcDict, int material, int uvBin ) const;
	void NoteLightmapUvFilter( int emitDict, int uvBin, int uvPlan );
	const PropUvSplitPlan *PropUvPlan( int planIndex ) const;
	const char *PropBakeName( const CStaticProp &prop ) const;
	int PropSelfShadowSkip( const CStaticProp &prop, int propIndex ) const;
	void AddPolysForRayTrace();
	void BuildTriList( CStaticProp &prop );
};


//-----------------------------------------------------------------------------
// Expose IVradStaticPropMgr to vrad
//-----------------------------------------------------------------------------

static CVradStaticPropMgr	g_StaticPropMgr;
IVradStaticPropMgr* StaticPropMgr()
{
	return &g_StaticPropMgr;
}

void CVradStaticPropMgr::AddAmbientSamplePositions( int leafIndex, CUtlVector<Vector> &positions, int maxPositions ) const
{
	if ( leafIndex < 0 || leafIndex >= m_LeafStaticProps.Count() || positions.Count() >= maxPositions )
		return;

	const CUtlVector<int> &leafProps = m_LeafStaticProps[leafIndex];
	const int totalSlots = leafProps.Count() * 8;
	const int wanted = min( maxPositions - positions.Count(), totalSlots );
	static const int sideAxis[6] = { 2, 2, 0, 0, 1, 1 };
	static const int sideSign[6] = { 1, -1, 1, -1, 1, -1 };
	const float clearance = 16.0f;

	for ( int i = 0; i < wanted; ++i )
	{
		const int slot = (int)( (int64)i * totalSlots / wanted );
		const CStaticProp &prop = m_StaticProps[leafProps[slot / 8]];
		const int propSlot = slot % 8;

		Vector sample = ( prop.m_mins + prop.m_maxs ) * 0.5f;
		if ( propSlot == 0 )
		{
			sample = prop.m_Origin;
			sample.z = prop.m_mins.z + 36.0f;
		}
		else if ( propSlot >= 2 )
		{
			const int side = propSlot - 2;
			const int axis = sideAxis[side];
			if ( sideSign[side] > 0 )
				sample[axis] = prop.m_maxs[axis] - clearance;
			else
				sample[axis] = prop.m_mins[axis] + clearance;
		}
		if ( PointLeafnum( sample ) == leafIndex )
			positions.AddToTail( sample );
	}
}

bool CVradStaticPropMgr::IsPointInsideStaticProp( int iThread, int leafIndex, const Vector &position ) const
{
	NOTE_UNUSED( iThread );
	if ( !s_pPhysCollision || leafIndex < 0 || leafIndex >= m_LeafStaticProps.Count() )
		return false;

	const CUtlVector<int> &leafProps = m_LeafStaticProps[leafIndex];
	for ( int i = 0; i < leafProps.Count(); ++i )
	{
		const CStaticProp &prop = m_StaticProps[leafProps[i]];
		if ( prop.m_Solid == 0 )
			continue;
		if ( prop.m_ModelIdx < 0 || prop.m_ModelIdx >= m_StaticPropDict.Count() )
			continue;
		if ( position.x < prop.m_mins.x || position.x > prop.m_maxs.x ||
			 position.y < prop.m_mins.y || position.y > prop.m_maxs.y ||
			 position.z < prop.m_mins.z || position.z > prop.m_maxs.z )
			continue;

		const StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
		Ray_t ray;
		ray.Init( position, position, vec3_origin, vec3_origin );

		bool tested = false;
		for ( int solid = 0; solid < dict.m_loadedModel.solidCount; ++solid )
		{
			const CPhysCollide *pModel = dict.m_loadedModel.solids
				? dict.m_loadedModel.solids[solid] : NULL;
			if ( !pModel )
				continue;
			tested = true;
			trace_t trace;
			Q_memset( &trace, 0, sizeof( trace ) );
			s_pPhysCollision->TraceBox( ray, pModel, prop.m_Origin, prop.m_Angles, &trace );
			if ( trace.startsolid || trace.allsolid )
				return true;
		}
		if ( !tested && dict.m_pModel )
		{
			trace_t trace;
			Q_memset( &trace, 0, sizeof( trace ) );
			s_pPhysCollision->TraceBox( ray, dict.m_pModel, prop.m_Origin, prop.m_Angles, &trace );
			if ( trace.startsolid || trace.allsolid )
				return true;
		}
	}
	return false;
}


//-----------------------------------------------------------------------------
// constructor, destructor
//-----------------------------------------------------------------------------

CVradStaticPropMgr::CVradStaticPropMgr()
{
	// set to ignore static prop traces
	m_bIgnoreStaticPropTrace = false;
}

CVradStaticPropMgr::~CVradStaticPropMgr()
{
}

//-----------------------------------------------------------------------------
// Makes sure the studio model is a static prop
//-----------------------------------------------------------------------------

bool IsStaticProp( studiohdr_t* pHdr )
{
	if (!(pHdr->flags & STUDIOHDR_FLAGS_STATIC_PROP))
		return false;

	return true;
}


//-----------------------------------------------------------------------------
// Load a file into a Utlbuf
//-----------------------------------------------------------------------------
static bool LoadFile( char const* pFileName, CUtlBuffer& buf )
{
	if ( !g_pFullFileSystem )
		return false;

	return g_pFullFileSystem->ReadFile( pFileName, NULL, buf );
}


//-----------------------------------------------------------------------------
// Constructs the file name from the model name
//-----------------------------------------------------------------------------
static char const* ConstructFileName( char const* pModelName )
{
	static char buf[1024];
	sprintf( buf, "%s%s", gamedir, pModelName );
	return buf;
}


//-----------------------------------------------------------------------------
// Computes a convex hull from a studio mesh
//-----------------------------------------------------------------------------
static CPhysConvex* ComputeConvexHull( mstudiomesh_t* pMesh, studiohdr_t *pStudioHdr  )
{
	const mstudio_meshvertexdata_t *vertData = pMesh->GetVertexData( (void *)pStudioHdr );
	Assert( vertData ); // This can only return NULL on X360 for now

	// Generate a list of all verts in the mesh
	Vector** ppVerts = (Vector**)_alloca(pMesh->numvertices * sizeof(Vector*) );
	for (int i = 0; i < pMesh->numvertices; ++i)
	{
		ppVerts[i] = vertData->Position(i);
	}

	// Generate a convex hull from the verts
	return s_pPhysCollision->ConvexFromVerts( ppVerts, pMesh->numvertices );
}


//-----------------------------------------------------------------------------
// Computes a convex hull from the studio model
//-----------------------------------------------------------------------------
CPhysCollide* ComputeConvexHull( studiohdr_t* pStudioHdr )
{
	CUtlVector<CPhysConvex*>	convexHulls;

	for (int body = 0; body < pStudioHdr->numbodyparts; ++body )
	{
		mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( body );
		for( int model = 0; model < pBodyPart->nummodels; ++model )
		{
			mstudiomodel_t *pStudioModel = pBodyPart->pModel( model );
			for( int mesh = 0; mesh < pStudioModel->nummeshes; ++mesh )
			{
				// Make a convex hull for each mesh
				// NOTE: This won't work unless the model has been compiled
				// with $staticprop
				mstudiomesh_t *pStudioMesh = pStudioModel->pMesh( mesh );
				convexHulls.AddToTail( ComputeConvexHull( pStudioMesh, pStudioHdr ) );
			}
		}
	}

	// Convert an array of convex elements to a compiled collision model
	// (this deletes the convex elements)
	return s_pPhysCollision->ConvertConvexToCollide( convexHulls.Base(), convexHulls.Size() );
}


//-----------------------------------------------------------------------------
// Load studio model vertex data from a file...
//-----------------------------------------------------------------------------

bool LoadStudioModel( char const* pModelName, CUtlBuffer& buf )
{
	// No luck, gotta build it	
	// Construct the file name...
	if (!LoadFile( pModelName, buf ))
	{
		Warning("Error! Unable to load model \"%s\"\n", pModelName );
		return false;
	}

	// Check that it's valid
	if (strncmp ((const char *) buf.PeekGet(), "IDST", 4) &&
		strncmp ((const char *) buf.PeekGet(), "IDAG", 4))
	{
		Warning("Error! Invalid model file \"%s\"\n", pModelName );
		return false;
	}

	studiohdr_t* pHdr = (studiohdr_t*)buf.PeekGet();

	Studio_ConvertStudioHdrToNewVersion( pHdr );

	if (pHdr->version != STUDIO_VERSION)
	{
		Warning("Error! Invalid model version \"%s\"\n", pModelName );
		return false;
	}

	if (!IsStaticProp(pHdr))
	{
		Warning("Error! To use model \"%s\"\n"
			"      as a static prop, it must be compiled with $staticprop!\n", pModelName );
		return false;
	}

	// ensure reset
	pHdr->SetVertexBase( NULL );
	pHdr->SetIndexBase( NULL );

	return true;
}

bool LoadStudioCollisionModel( char const* pModelName, CUtlBuffer& buf )
{
	char tmp[1024];
	Q_strncpy( tmp, pModelName, sizeof( tmp ) );
	Q_SetExtension( tmp, ".phy", sizeof( tmp ) );
	// No luck, gotta build it	
	if (!LoadFile( tmp, buf ))
	{
		// this is not an error, the model simply has no PHY file
		return false;
	}

	phyheader_t *header = (phyheader_t *)buf.PeekGet();

	if ( header->size != sizeof(*header) || header->solidCount <= 0 )
		return false;

	return true;
}

bool LoadVTXFile( char const* pModelName, const studiohdr_t *pStudioHdr, CUtlBuffer& buf )
{
	// GMod / modern studiomdl often ships only .dx90.vtx. Stock VRAD
	// hard-coded .dx80.vtx and treated a miss as a hard error.
	static const char *const s_pExts[] = { ".dx90.vtx", ".dx80.vtx", ".sw.vtx" };

	char filename[MAX_PATH];
	for ( int i = 0; i < ARRAYSIZE( s_pExts ); ++i )
	{
		Q_strncpy( filename, pModelName, sizeof( filename ) );
		Q_SetExtension( filename, s_pExts[i], sizeof( filename ) );

		buf.Purge();
		if ( !LoadFile( filename, buf ) || buf.Size() < (int)sizeof( OptimizedModel::FileHeader_t ) )
			continue;

		OptimizedModel::FileHeader_t* pVtxHdr = (OptimizedModel::FileHeader_t *)buf.Base();
		if ( pVtxHdr->version != OPTIMIZED_MODEL_FILE_VERSION )
		{
			Warning( "Error! Invalid VTX file version: %d, expected %d \"%s\"\n",
				pVtxHdr->version, OPTIMIZED_MODEL_FILE_VERSION, filename );
			continue;
		}
		if ( pVtxHdr->checkSum != pStudioHdr->checksum )
		{
			Warning( "Error! Invalid VTX file checksum: %d, expected %d \"%s\"\n",
				pVtxHdr->checkSum, pStudioHdr->checksum, filename );
			continue;
		}

		return true;
	}

	Warning( "Error! Unable to load VTX for \"%s\" (tried .dx90.vtx, .dx80.vtx, .sw.vtx)\n", pModelName );
	return false;
}

//-----------------------------------------------------------------------------
// Gets a vertex position from a strip index
//-----------------------------------------------------------------------------
inline static Vector* PositionFromIndex( const mstudio_meshvertexdata_t *vertData, mstudiomesh_t* pMesh, OptimizedModel::StripGroupHeader_t* pStripGroup, int i )
{
	OptimizedModel::Vertex_t* pVert = pStripGroup->pVertex( i );
	return vertData->Position( pVert->origMeshVertID );
}


//-----------------------------------------------------------------------------
// Purpose: Writes a glview text file containing the collision surface in question
// Input  : *pCollide - 
//			*pFilename - 
//-----------------------------------------------------------------------------
void DumpCollideToGlView( vcollide_t *pCollide, const char *pFilename )
{
	if ( !pCollide )
		return;

	Msg("Writing %s...\n", pFilename );

	FILE *fp = fopen( pFilename, "w" );
	for (int i = 0; i < pCollide->solidCount; ++i)
	{
		Vector *outVerts;
		int vertCount = s_pPhysCollision->CreateDebugMesh( pCollide->solids[i], &outVerts );
		int triCount = vertCount / 3;
		int vert = 0;

		unsigned char r = (i & 1) * 64 + 64;
		unsigned char g = (i & 2) * 64 + 64;
		unsigned char b = (i & 4) * 64 + 64;

		float fr = r / 255.0f;
		float fg = g / 255.0f;
		float fb = b / 255.0f;

		for ( int i = 0; i < triCount; i++ )
		{
			fprintf( fp, "3\n" );
			fprintf( fp, "%6.3f %6.3f %6.3f %.2f %.3f %.3f\n", 
				outVerts[vert].x, outVerts[vert].y, outVerts[vert].z, fr, fg, fb );
			vert++;
			fprintf( fp, "%6.3f %6.3f %6.3f %.2f %.3f %.3f\n", 
				outVerts[vert].x, outVerts[vert].y, outVerts[vert].z, fr, fg, fb );
			vert++;
			fprintf( fp, "%6.3f %6.3f %6.3f %.2f %.3f %.3f\n", 
				outVerts[vert].x, outVerts[vert].y, outVerts[vert].z, fr, fg, fb );
			vert++;
		}
		s_pPhysCollision->DestroyDebugMesh( vertCount, outVerts );
	}
	fclose( fp );
}


static bool PointInTriangle( const Vector2D &p, const Vector2D &v0, const Vector2D &v1, const Vector2D &v2 )
{
	float coords[3];
	GetBarycentricCoords2D( v0, v1, v2, p, coords );
	for ( int i = 0; i < 3; i++ )
	{
		if ( coords[i] < 0.0f || coords[i] > 1.0f )
			return false;
	}
	float sum = coords[0] + coords[1] + coords[2];
	if ( sum > 1.0f )
		return false;
	return true;
}

bool LoadFileIntoBuffer( CUtlBuffer &buf, const char *pFilename )
{
	FileHandle_t fileHandle = g_pFileSystem->Open( pFilename, "rb" );
	if ( !fileHandle )
		return false;

	// Get the file size
	int texSize = g_pFileSystem->Size( fileHandle );
	buf.EnsureCapacity( texSize );
	int nBytesRead = g_pFileSystem->Read( buf.Base(), texSize, fileHandle );
	g_pFileSystem->Close( fileHandle );
	buf.SeekPut( CUtlBuffer::SEEK_HEAD, nBytesRead );
	buf.SeekGet( CUtlBuffer::SEEK_HEAD, 0 );
	return true;
}

// keeps a list of all textures that cast shadows via alpha channel
class CShadowTextureList
{
public:
	// This loads a vtf and converts it to RGB8888 format
	unsigned char *LoadVTFRGB8888( const char *pName, int *pWidth, int *pHeight, bool *pClampU, bool *pClampV )
	{
		char szPath[MAX_PATH];
		Q_strncpy( szPath, "materials/", sizeof( szPath ) );
		Q_strncat( szPath, pName, sizeof( szPath ), COPY_ALL_CHARACTERS );
		Q_strncat( szPath, ".vtf", sizeof( szPath ), COPY_ALL_CHARACTERS );
		Q_FixSlashes( szPath, CORRECT_PATH_SEPARATOR );

		CUtlBuffer buf;
		if ( !LoadFileIntoBuffer( buf, szPath ) )
			return NULL;
		PatchVtfForSdkLoader( buf );
		IVTFTexture *pTex = CreateVTFTexture();
		if (!pTex->Unserialize( buf ))
		{
			DestroyVTFTexture( pTex );
			return NULL;
		}
		Msg("Loaded alpha texture %s\n", szPath );
		unsigned char *pSrcImage = pTex->ImageData( 0, 0, 0, 0, 0, 0 );
		int iWidth = pTex->Width();
		int iHeight = pTex->Height();
		ImageFormat dstFormat = IMAGE_FORMAT_RGBA8888;
		ImageFormat srcFormat = pTex->Format();
		*pClampU = (pTex->Flags() & TEXTUREFLAGS_CLAMPS) ? true : false;
		*pClampV = (pTex->Flags() & TEXTUREFLAGS_CLAMPT) ? true : false;
		unsigned char *pDstImage = new unsigned char[ImageLoader::GetMemRequired( iWidth, iHeight, 1, dstFormat, false )];

		if( !ImageLoader::ConvertImageFormat( pSrcImage, srcFormat, 
			pDstImage, dstFormat, iWidth, iHeight, 0, 0 ) )
		{
			delete[] pDstImage;
			DestroyVTFTexture( pTex );
			return NULL;
		}

		DestroyVTFTexture( pTex );
		*pWidth = iWidth;
		*pHeight = iHeight;
		return pDstImage;
	}

	// Checks the database for the material and loads if necessary
	// returns true if found and pIndex will be the index, -1 if no alpha shadows
	bool FindOrLoadIfValid( const char *pMaterialName, int *pIndex )
	{
		*pIndex = -1;
		int index = m_Textures.Find(pMaterialName);
		bool bFound = false;
		if ( index != m_Textures.InvalidIndex() )
		{
			bFound = true;
			*pIndex = index;
		}
		else
		{
			KeyValues *pVMT = new KeyValues("vmt");
			CUtlBuffer buf(0,0,CUtlBuffer::TEXT_BUFFER);
			LoadFileIntoBuffer( buf, pMaterialName );
			if ( pVMT->LoadFromBuffer( pMaterialName, buf ) )
			{
				bFound = true;
				KeyValues *pTrans = pVMT->FindKey( "$translucent" );
				KeyValues *pAlphaTest = pVMT->FindKey( "$alphatest" );
				const bool bTranslucent = pTrans && ( pTrans->GetInt() != 0 || pTrans->GetFloat() > 0.0f );
				const bool bAlphaTest = pAlphaTest && ( pAlphaTest->GetInt() != 0 || pAlphaTest->GetFloat() > 0.0f );
				if ( bTranslucent || bAlphaTest )
				{
					KeyValues *pBaseTexture = pVMT->FindKey("$basetexture");
					if ( pBaseTexture )
					{
						const char *pBaseTextureName = pBaseTexture->GetString();
						if ( pBaseTextureName )
						{
							int w, h;
							bool bClampU = false;
							bool bClampV = false;
							unsigned char *pImageBits = LoadVTFRGB8888( pBaseTextureName, &w, &h, &bClampU, &bClampV );
							if ( pImageBits )
							{
								int index = m_Textures.Insert( pMaterialName );
								m_Textures[index].InitFromRGB8888( w, h, pImageBits );
								m_Textures[index].isAlphaTest = bAlphaTest; // cutout; else soft $translucent
								*pIndex = index;
								if ( pVMT->FindKey("$nocull") )
								{
									// UNDONE: Support this? Do we need to emit two triangles?
									m_Textures[index].allowBackface = true;
								}
								m_Textures[index].clampU = bClampU;
								m_Textures[index].clampV = bClampV;
								delete[] pImageBits;
							}
						}
					}
				}

			}
			pVMT->deleteThis();
		}

		return bFound;
	}


	// iterate the textures for the model and load each one into the database
	// this is used on models marked to cast texture shadows
	void LoadAllTexturesForModel( studiohdr_t *pHdr, int *pTextureList )
	{
		for ( int i = 0; i < pHdr->numtextures; i++ )
		{
			int textureIndex = -1;
			// try to add each texture to the transparent shadow manager
			char szPath[MAX_PATH];

			// iterate quietly through all specified directories until a valid material is found
			for ( int j = 0; j < pHdr->numcdtextures; j++ )
			{
				Q_strncpy( szPath, "materials/", sizeof( szPath ) );
				Q_strncat( szPath, pHdr->pCdtexture( j ), sizeof( szPath ) );
				const char *textureName = pHdr->pTexture( i )->pszName();
				Q_strncat( szPath, textureName, sizeof( szPath ), COPY_ALL_CHARACTERS );
				Q_strncat( szPath, ".vmt", sizeof( szPath ), COPY_ALL_CHARACTERS );
				Q_FixSlashes( szPath, CORRECT_PATH_SEPARATOR );
				if ( FindOrLoadIfValid( szPath, &textureIndex ) )
					break;
			}

			pTextureList[i] = textureIndex;
		}
	}
	
	int AddMaterialEntry( int shadowTextureIndex, const Vector2D &t0, const Vector2D &t1, const Vector2D &t2 )
	{
		int index = m_MaterialEntries.AddToTail();
		m_MaterialEntries[index].textureIndex = shadowTextureIndex;
		m_MaterialEntries[index].uv[0] = t0;
		m_MaterialEntries[index].uv[1] = t1;
		m_MaterialEntries[index].uv[2] = t2;
		return index;
	}

	// HACKHACK: Compute the average coverage for this triangle by sampling the AABB of its texture space
	float ComputeCoverageForTriangle( int shadowTextureIndex, const Vector2D &t0, const Vector2D &t1, const Vector2D &t2 )
	{
		float umin = min(t0.x, t1.x);
		umin = min(umin, t2.x);
		float umax = max(t0.x, t1.x);
		umax = max(umax, t2.x);

		float vmin = min(t0.y, t1.y);
		vmin = min(vmin, t2.y);
		float vmax = max(t0.y, t1.y);
		vmax = max(vmax, t2.y);

		// UNDONE: Do something about tiling
		umin = clamp(umin, 0, 1);
		umax = clamp(umax, 0, 1);
		vmin = clamp(vmin, 0, 1);
		vmax = clamp(vmax, 0, 1);
		Assert(umin>=0.0f && umax <= 1.0f);
		Assert(vmin>=0.0f && vmax <= 1.0f);
		const alphatexture_t &tex = m_Textures.Element(shadowTextureIndex);
		int u0 = umin * (tex.width-1);
		int u1 = umax * (tex.width-1);
		int v0 = vmin * (tex.height-1);
		int v1 = vmax * (tex.height-1);

		int total = 0;
		int count = 0;
		for ( int v = v0; v <= v1; v++ )
		{
			int row = (v * tex.width);
			for ( int u = u0; u <= u1; u++ )
			{
				total += tex.pAlphaTexels[row + u];
				count++;
			}
		}
		if ( count )
		{
			float coverage = float(total) / (count * 255.0f);
			return coverage;
		}
		return 1.0f;
	}
	
	int SampleMaterial( int materialIndex, const Vector &coords, bool bBackface )
	{
		const materialentry_t &mat = m_MaterialEntries[materialIndex];
		const alphatexture_t &tex = m_Textures.Element(m_MaterialEntries[materialIndex].textureIndex);
		if ( bBackface && !tex.allowBackface )
			return 0;
		Vector2D uv = coords.x * mat.uv[0] + coords.y * mat.uv[1] + coords.z * mat.uv[2];
		int u = RoundFloatToInt( uv[0] * tex.width );
		int v = RoundFloatToInt( uv[1] * tex.height );
		
		// asume power of 2, clamp or wrap
		// UNDONE: Support clamp?  This code should work
#if 0
		u = tex.clampU ? clamp(u,0,(tex.width-1)) : (u & (tex.width-1));
		v = tex.clampV ? clamp(v,0,(tex.height-1)) : (v & (tex.height-1));
#else
		// for now always wrap
		u &= (tex.width-1);
		v &= (tex.height-1);
#endif

		return tex.pAlphaTexels[v * tex.width + u];
	}

	bool BuildGpuUpload( const int *matIndexPerTri, uint32_t nTris,
						 ShadowAlphaGpuArray_t &outArray,
						 std::vector<float> &outMetaFloat4s )
	{
		outArray.width = 1;
		outArray.height = 1;
		outArray.layers = 1;
		outArray.rgba.assign( 4, 255 );
		outMetaFloat4s.assign( (size_t)nTris * (size_t)kShadowAlphaMetaFloat4s * 4u, 0.0f );
		if ( !matIndexPerTri || nTris == 0 )
			return true;

		const int kMaxLayers = 256;
		const int kMaxDim = 2048;
		CUtlDict<int, int> layerOf;
		CUtlVector<int> layerTexIdx;

		auto GetLayer = [&]( int textureIndex ) -> int
		{
			if ( !m_Textures.IsValidIndex( textureIndex ) )
				return -1;
			const alphatexture_t &tex = m_Textures[textureIndex];
			if ( !tex.pAlphaTexels || tex.width <= 0 || tex.height <= 0 )
				return -1;
			if ( tex.width > kMaxDim || tex.height > kMaxDim )
				return -1;
			char key[32];
			Q_snprintf( key, sizeof( key ), "%d", textureIndex );
			unsigned short idx = layerOf.Find( key );
			if ( layerOf.IsValidIndex( idx ) )
				return layerOf[idx];
			if ( layerTexIdx.Count() >= kMaxLayers )
				return -1;
			const int layer = layerTexIdx.Count();
			layerTexIdx.AddToTail( textureIndex );
			layerOf.Insert( key, layer );
			return layer;
		};

		int nAlpha = 0;
		int maxW = 1, maxH = 1;
		for ( uint32_t i = 0; i < nTris; ++i )
		{
			float *m = &outMetaFloat4s[(size_t)i * (size_t)kShadowAlphaMetaFloat4s * 4u];
			const int matIdx = matIndexPerTri[i];
			if ( matIdx < 0 || matIdx >= m_MaterialEntries.Count() )
				continue;
			const materialentry_t &mat = m_MaterialEntries[matIdx];
			const int layer = GetLayer( mat.textureIndex );
			if ( layer < 0 )
				continue;
			const alphatexture_t &tex = m_Textures[mat.textureIndex];
			// 1 = $alphatest stochastic cutout, 2 = $translucent fractional T*=(1-a)
			m[0] = tex.isAlphaTest ? 1.0f : 2.0f;
			m[1] = (float)layer;
			m[2] = (float)tex.width;
			m[3] = (float)tex.height;
			m[4] = mat.uv[0].x; m[5] = mat.uv[0].y;
			m[6] = mat.uv[1].x; m[7] = mat.uv[1].y;
			m[8] = mat.uv[2].x; m[9] = mat.uv[2].y;
			m[10] = 0.0f; m[11] = 0.0f;
			maxW = max( maxW, (int)tex.width );
			maxH = max( maxH, (int)tex.height );
			++nAlpha;
		}

		if ( layerTexIdx.Count() == 0 || nAlpha == 0 )
			return true;

		outArray.width = maxW;
		outArray.height = maxH;
		outArray.layers = layerTexIdx.Count();
		outArray.rgba.assign( (size_t)maxW * (size_t)maxH * 4u * (size_t)outArray.layers, 0 );
		for ( int layer = 0; layer < layerTexIdx.Count(); ++layer )
		{
			const alphatexture_t &tex = m_Textures[layerTexIdx[layer]];
			unsigned char *dstBase = &outArray.rgba[(size_t)layer * (size_t)maxW * (size_t)maxH * 4u];
			for ( int v = 0; v < tex.height; ++v )
			{
				for ( int u = 0; u < tex.width; ++u )
				{
					const unsigned char a = tex.pAlphaTexels[v * tex.width + u];
					unsigned char *p = dstBase + ( (size_t)v * (size_t)maxW + (size_t)u ) * 4u;
					p[0] = a; p[1] = a; p[2] = a; p[3] = a;
				}
			}
		}
		Msg( "[PathTrace-DXR] Texture shadows GPU: %d alpha tris, %d atlas layer(s) %dx%d.\n",
			 nAlpha, outArray.layers, outArray.width, outArray.height );
		return true;
	}

	struct alphatexture_t 
	{
		short width;
		short height;
		bool allowBackface;
		bool clampU;
		bool clampV;
		bool isAlphaTest;	// true=$alphatest cutout; false=$translucent soft alpha
		unsigned char *pAlphaTexels;

		void InitFromRGB8888( int w, int h, unsigned char *pTexels )
		{
			width = w;
			height = h;
			allowBackface = false;
			clampU = false;
			clampV = false;
			isAlphaTest = false;
			pAlphaTexels = new unsigned char[w*h];
			for ( int i = 0; i < h; i++ )
			{
				for ( int j = 0; j < w; j++ )
				{
					int index = (i*w) + j;
					pAlphaTexels[index] = pTexels[index*4 + 3];
				}
			}
		}
	};
	struct materialentry_t
	{
		int textureIndex;
		Vector2D uv[3];
	};
	// this is the list of textures we've loaded
	// only load each one once
	CUtlDict< alphatexture_t, unsigned short > m_Textures;
	CUtlVector<materialentry_t> m_MaterialEntries;
};

// global to keep the shadow-casting texture list and their alpha bits
CShadowTextureList g_ShadowTextureList;

float ComputeCoverageFromTexture( float b0, float b1, float b2, int32 hitID )
{
	const float alphaScale = 1.0f / 255.0f;
	// UNDONE: Pass ray down to determine backfacing?
	//Vector normal( tri.m_flNx, tri.m_flNy, tri.m_flNz );
	//bool bBackface = DotProduct(delta, tri.N) > 0 ? true : false;
	Vector coords(b0,b1,b2);
	return alphaScale * g_ShadowTextureList.SampleMaterial( g_RtEnv.GetTriangleMaterial(hitID), coords, false );
}

bool ShadowTexture_BuildGpuUpload( const int *matIndexPerTri, uint32_t nTris,
								   ShadowAlphaGpuArray_t &outArray,
								   std::vector<float> &outMetaFloat4s )
{
	return g_ShadowTextureList.BuildGpuUpload( matIndexPerTri, nTris, outArray, outMetaFloat4s );
}

// this is here to strip models/ or .mdl or whatnot
void CleanModelName( const char *pModelName, char *pOutput, int outLen )
{
	// strip off leading models/ if it exists
	const char *pModelDir = "models/";
	int modelLen = Q_strlen(pModelDir);

	if ( !Q_strnicmp(pModelName, pModelDir, modelLen ) )
	{
		pModelName += modelLen;
	}
	Q_strncpy( pOutput, pModelName, outLen );

	// truncate any .mdl extension
	char *dot = strchr(pOutput,'.');
	if ( dot )
	{
		*dot = 0;
	}

}


void ForceTextureShadowsOnModel( const char *pModelName )
{
	char buf[1024];
	CleanModelName( pModelName, buf, sizeof(buf) );
	if ( !g_ForcedTextureShadowsModels.Find(buf).IsValid())
	{
		g_ForcedTextureShadowsModels.AddString(buf);
	}
}

bool IsModelTextureShadowsForced( const char *pModelName )
{
	char buf[1024];
	CleanModelName( pModelName, buf, sizeof(buf) );
	return g_ForcedTextureShadowsModels.Find(buf).IsValid();
}


//-----------------------------------------------------------------------------
// Creates a collision model (based on the render geometry!)
//-----------------------------------------------------------------------------
void CVradStaticPropMgr::CreateCollisionModel( char const* pModelName )
{
	CUtlBuffer buf;
	CUtlBuffer bufvtx;
	CUtlBuffer bufphy;

	int i = m_StaticPropDict.AddToTail();
	m_StaticPropDict[i].m_pModel = NULL;
	m_StaticPropDict[i].m_pStudioHdr = NULL;
	Q_memset( &m_StaticPropDict[i].m_loadedModel, 0, sizeof( m_StaticPropDict[i].m_loadedModel ) );
	Q_strncpy( m_StaticPropDict[i].m_Name, pModelName ? pModelName : "", sizeof( m_StaticPropDict[i].m_Name ) );

	if ( !LoadStudioModel( pModelName, buf ) )
	{
		VectorCopy( vec3_origin, m_StaticPropDict[i].m_Mins );
		VectorCopy( vec3_origin, m_StaticPropDict[i].m_Maxs );
		return;
	}

	studiohdr_t* pHdr = (studiohdr_t*)buf.Base();

	VectorCopy( pHdr->hull_min, m_StaticPropDict[i].m_Mins );
	VectorCopy( pHdr->hull_max, m_StaticPropDict[i].m_Maxs );

	if ( LoadStudioCollisionModel( pModelName, bufphy ) )
	{
		phyheader_t header;
		bufphy.Get( &header, sizeof(header) );

		vcollide_t *pCollide = &m_StaticPropDict[i].m_loadedModel;
		s_pPhysCollision->VCollideLoad( pCollide, header.solidCount, (const char *)bufphy.PeekGet(), bufphy.TellPut() - bufphy.TellGet() );
		m_StaticPropDict[i].m_pModel = m_StaticPropDict[i].m_loadedModel.solids[0];

		/*
		static int propNum = 0;
		char tmp[128];
		sprintf( tmp, "staticprop%03d.txt", propNum );
		DumpCollideToGlView( pCollide, tmp );
		++propNum;
		*/
	}
	else
	{
		// mark this as unused
		m_StaticPropDict[i].m_loadedModel.solidCount = 0;

		// CPhysCollide* pPhys = CreatePhysCollide( pHdr, pVtxHdr );
		m_StaticPropDict[i].m_pModel = ComputeConvexHull( pHdr );
	}

	// clone it
	m_StaticPropDict[i].m_pStudioHdr = (studiohdr_t *)malloc( buf.Size() );
	memcpy( m_StaticPropDict[i].m_pStudioHdr, (studiohdr_t*)buf.Base(), buf.Size() );

	if ( !LoadVTXFile( pModelName, m_StaticPropDict[i].m_pStudioHdr, m_StaticPropDict[i].m_VtxBuf ) )
	{
		// failed, leave state identified as disabled
		m_StaticPropDict[i].m_VtxBuf.Purge();
	}

	if ( g_bTextureShadows )
	{
		// Always probe materials when -textureshadows is set. FindOrLoadIfValid only
		// keeps $alphatest / $translucent maps, so opaque props stay opaque. Do not
		// require STUDIOHDR_FLAGS_CAST_TEXTURE_SHADOWS or lights.rad forcetextureshadow
		// (those remain as optional optimizers / explicit overrides).
		m_StaticPropDict[i].m_textureShadowIndex.RemoveAll();
		m_StaticPropDict[i].m_triangleMaterialIndex.RemoveAll();
		m_StaticPropDict[i].m_textureShadowIndex.AddMultipleToTail( pHdr->numtextures );
		g_ShadowTextureList.LoadAllTexturesForModel( pHdr, m_StaticPropDict[i].m_textureShadowIndex.Base() );
	}
}


//-----------------------------------------------------------------------------
// Unserialize static prop model dictionary
//-----------------------------------------------------------------------------
void CVradStaticPropMgr::UnserializeModelDict( CUtlBuffer& buf )
{
	int count = buf.GetInt();
	while ( --count >= 0 )
	{
		StaticPropDictLump_t lump;
		buf.Get( &lump, sizeof(StaticPropDictLump_t) );
		
		CreateCollisionModel( lump.m_Name );
	}
}

void CVradStaticPropMgr::UnserializeModels( CUtlBuffer& buf )
{
	int count = buf.GetInt();


	m_StaticProps.AddMultipleToTail(count);
	for ( int i = 0; i < count; ++i )				  
	{
		StaticPropLump_t lump;
		buf.Get( &lump, sizeof(StaticPropLump_t) );
		
		VectorCopy( lump.m_Origin, m_StaticProps[i].m_Origin );
		VectorCopy( lump.m_Angles, m_StaticProps[i].m_Angles );
		VectorCopy( lump.m_LightingOrigin, m_StaticProps[i].m_LightingOrigin );
		m_StaticProps[i].m_bLightingOriginValid = ( lump.m_Flags & STATIC_PROP_USE_LIGHTING_ORIGIN ) > 0;
		m_StaticProps[i].m_ModelIdx = lump.m_PropType;
		m_StaticProps[i].m_FirstLeaf = lump.m_FirstLeaf;
		m_StaticProps[i].m_LeafCount = lump.m_LeafCount;
		m_StaticProps[i].m_Solid = lump.m_Solid;
		m_StaticProps[i].m_Handle = TREEDATA_INVALID_HANDLE;
		m_StaticProps[i].m_Flags = lump.m_Flags;
		m_StaticProps[i].m_OnlyMaterial = -1;
		m_StaticProps[i].m_UvBin = -1;
		m_StaticProps[i].m_UvPlan = -1;
		m_StaticProps[i].m_EmitModelIdx = -1;
		m_StaticProps[i].m_ShadowSkipProp = -1;
		m_StaticProps[i].m_Skin = lump.m_Skin;
		m_StaticProps[i].m_FadeMinDist = lump.m_FadeMinDist;
		m_StaticProps[i].m_FadeMaxDist = lump.m_FadeMaxDist;
		m_StaticProps[i].m_flForcedFadeScale = lump.m_flForcedFadeScale;
		m_StaticProps[i].m_nMinDXLevel = lump.m_nMinDXLevel;
		m_StaticProps[i].m_nMaxDXLevel = lump.m_nMaxDXLevel;

		// Cache an oriented world AABB. Ambient probes use its faces as cheap
		// surface-guided candidates; the final placement test uses real geometry.
		m_StaticProps[i].m_mins = lump.m_Origin;
		m_StaticProps[i].m_maxs = lump.m_Origin;
		if ( lump.m_PropType >= 0 && lump.m_PropType < m_StaticPropDict.Count() )
		{
			const StaticPropDict_t &dict = m_StaticPropDict[lump.m_PropType];
			matrix3x4_t matrix;
			AngleMatrix( lump.m_Angles, lump.m_Origin, matrix );
			m_StaticProps[i].m_mins.Init( FLT_MAX, FLT_MAX, FLT_MAX );
			m_StaticProps[i].m_maxs.Init( -FLT_MAX, -FLT_MAX, -FLT_MAX );
			for ( int corner = 0; corner < 8; ++corner )
			{
				Vector local(
					( corner & 1 ) ? dict.m_Maxs.x : dict.m_Mins.x,
					( corner & 2 ) ? dict.m_Maxs.y : dict.m_Mins.y,
					( corner & 4 ) ? dict.m_Maxs.z : dict.m_Mins.z );
				Vector world;
				VectorTransform( local, matrix, world );
				VectorMin( m_StaticProps[i].m_mins, world, m_StaticProps[i].m_mins );
				VectorMax( m_StaticProps[i].m_maxs, world, m_StaticProps[i].m_maxs );
			}
			// Collision can extend beyond studio hull bounds. Expand the broad phase
			// so every VPhysics solid still reaches the exact point-containment test.
			if ( s_pPhysCollision && dict.m_loadedModel.solids )
			{
				for ( int solid = 0; solid < dict.m_loadedModel.solidCount; ++solid )
				{
					if ( !dict.m_loadedModel.solids[solid] )
						continue;
					Vector collisionMins, collisionMaxs;
					s_pPhysCollision->CollideGetAABB( &collisionMins, &collisionMaxs,
						dict.m_loadedModel.solids[solid], lump.m_Origin, lump.m_Angles );
					VectorMin( m_StaticProps[i].m_mins, collisionMins, m_StaticProps[i].m_mins );
					VectorMax( m_StaticProps[i].m_maxs, collisionMaxs, m_StaticProps[i].m_maxs );
				}
			}
		}

		// Changed this from using DXT1 to RGB888 because the compression artifacts were pretty nasty. 
		// TODO: Consider changing back or basing this on user selection in hammer.
		m_StaticProps[i].m_LightmapImageFormat = IMAGE_FORMAT_RGB888;
		m_StaticProps[i].m_LightmapImageWidth = lump.m_nLightmapResolutionX;
		m_StaticProps[i].m_LightmapImageHeight = lump.m_nLightmapResolutionY;
	}
}

//-----------------------------------------------------------------------------
// Unserialize static props
//-----------------------------------------------------------------------------

void CVradStaticPropMgr::UnserializeStaticProps()
{
	// Unserialize static props, insert them into the appropriate leaves
	GameLumpHandle_t handle = g_GameLumps.GetGameLumpHandle( GAMELUMP_STATIC_PROPS );
	int size = g_GameLumps.GameLumpSize( handle );
	if (!size)
		return;

	if ( g_GameLumps.GetGameLumpVersion( handle ) != GAMELUMP_STATIC_PROPS_VERSION )
	{
		Error( "Cannot load the static props... encountered a stale map version. Re-vbsp the map." );
	}

	if ( g_GameLumps.GetGameLump( handle ) )
	{
		CUtlBuffer buf( g_GameLumps.GetGameLump(handle), size, CUtlBuffer::READ_ONLY );
		UnserializeModelDict( buf );

		// Keep the leaf list so ambient probes can be seeded from nearby props.
		int count = buf.GetInt();
		m_StaticPropLeaves.SetCount( count );
		if ( count )
			buf.Get( m_StaticPropLeaves.Base(), count * sizeof(StaticPropLeafLump_t) );

		UnserializeModels( buf );

		m_LeafStaticProps.SetCount( numleafs );
		for ( int propIndex = 0; propIndex < m_StaticProps.Count(); ++propIndex )
		{
			const CStaticProp &prop = m_StaticProps[propIndex];
			const int end = min( (int)prop.m_FirstLeaf + (int)prop.m_LeafCount, m_StaticPropLeaves.Count() );
			for ( int i = prop.m_FirstLeaf; i < end; ++i )
			{
				const int leaf = m_StaticPropLeaves[i].m_Leaf;
				if ( leaf >= 0 && leaf < numleafs )
					m_LeafStaticProps[leaf].AddToTail( propIndex );
			}
		}
	}
}

//-----------------------------------------------------------------------------
// Level init, shutdown
//-----------------------------------------------------------------------------

void CVradStaticPropMgr::Init()
{
	CreateInterfaceFn physicsFactory = GetPhysicsFactory();
	if ( !physicsFactory )
		Error( "Unable to load vphysics DLL." );
		
	s_pPhysCollision = (IPhysicsCollision *)physicsFactory( VPHYSICS_COLLISION_INTERFACE_VERSION, NULL );
	if( !s_pPhysCollision )
	{
		Error( "Unable to get '%s' for physics interface.", VPHYSICS_COLLISION_INTERFACE_VERSION );
		return;
	}

	// Read in static props that have been compiled into the bsp file
	UnserializeStaticProps();
	const int nOriginalProps = m_StaticProps.Count();
	const int nOriginalModels = m_StaticPropDict.Count();
	// Before raytrace polys, so material and UV clones are not added as extra occluders.
	SplitLightmappedMaterialProps();
	SplitLightmappedUvProps();
	PrintLightmapSplitStats( nOriginalProps, nOriginalModels );
}

void CVradStaticPropMgr::Shutdown()
{
	// Remove all static prop model data
	for (int i = m_StaticPropDict.Size(); --i >= 0; )
	{
		studiohdr_t *pStudioHdr = m_StaticPropDict[i].m_pStudioHdr;
		if ( pStudioHdr )
		{
			if ( pStudioHdr->VertexBase() )
			{
				free( pStudioHdr->VertexBase() );
			}
			free( pStudioHdr );
		}
	}

	m_StaticProps.Purge();
	m_StaticPropDict.Purge();
	m_StaticPropLeaves.Purge();
	m_LeafStaticProps.Purge();
}

void ComputeLightmapColor( dface_t* pFace, Vector &color )
{
	texinfo_t* pTex = &texinfo[pFace->texinfo];
	if ( pTex->flags & SURF_SKY )
	{
		// sky ambient already accounted for in direct component
		return;
	}
}

bool PositionInSolid( Vector &position )
{
	int ndxLeaf = PointLeafnum( position );
	if ( dleafs[ndxLeaf].contents & CONTENTS_SOLID )
	{
		// position embedded in solid
		return true;
	}

	return false;
}

//-----------------------------------------------------------------------------
// Trace from a vertex to each direct light source, accumulating its contribution.
//-----------------------------------------------------------------------------
void ComputeDirectLightingAtPoint( Vector &position, Vector &normal, Vector &outColor, int iThread,
								   int static_prop_id_to_skip, int nLFlags )
{
	SSE_sampleLightOutput_t	sampleOutput;

	outColor.Init();

	// Iterate over all direct lights and accumulate their contribution
	int cluster = ClusterFromPoint( position );
	for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
	{
		if ( dl->light.style )
		{
			// skip lights with style
			continue;
		}

		// is this lights cluster visible?
		if ( !PVSCheck( dl->pvs, cluster ) )
			continue;

		// push the vertex towards the light to avoid surface acne
		Vector adjusted_pos = position;
		float flEpsilon = 0.0;

		if  (dl->light.type != emit_skyambient)
		{
			// push towards the light
			Vector fudge;
			if ( dl->light.type == emit_skylight )
				fudge = -( dl->light.normal);
			else
			{
				fudge = dl->light.origin-position;
				VectorNormalize( fudge );
			}
			fudge *= 4.0;
			adjusted_pos += fudge;
		}
		else 
		{
			// push out along normal
			adjusted_pos += 4.0 * normal;
//			flEpsilon = 1.0;
		}

		FourVectors adjusted_pos4;
		FourVectors normal4;
		adjusted_pos4.DuplicateVector( adjusted_pos );
		normal4.DuplicateVector( normal );

		GatherSampleLightSSE( sampleOutput, dl, -1, adjusted_pos4, &normal4, 1, iThread, nLFlags | GATHERLFLAGS_FORCE_FAST,
		                      static_prop_id_to_skip, flEpsilon );
		
		{
			const float amt = sampleOutput.m_flFalloff.m128_f32[0] * sampleOutput.m_flDot[0].m128_f32[0];
			Vector intensity = dl->light.intensity;
			if ( dl->m_pProj && amt > 0.0f )
			{
				Vector lightToSample = adjusted_pos - dl->light.origin;
				Vector rgb = Proj_EvalLightRGB( dl, lightToSample );
				intensity.x *= rgb.x;
				intensity.y *= rgb.y;
				intensity.z *= rgb.z;
			}
			Vector contrib = intensity * amt;
			contrib = IES_ClampDirectSample( dl, contrib );
			outColor += contrib;
		}
	}
}

//-----------------------------------------------------------------------------
// Same as ComputeDirectLightingAtPoint, but lights up to 4 vertices per
// gather call - one traced ray set serves 4 verts instead of duplicating
// one vertex across all SSE lanes (~4x fewer traces for prop vertices).
//-----------------------------------------------------------------------------
static void ComputeDirectLightingAtPoints4( const Vector *positions, const Vector *normals,
											Vector *outColors, int nCount, int iThread,
											int static_prop_id_to_skip = -1, int nLFlags = 0 )
{
	SSE_sampleLightOutput_t	sampleOutput;

	Vector pos[4], nrm[4];
	int clusters[4];
	for ( int i = 0; i < 4; i++ )
	{
		const int src = ( i < nCount ) ? i : 0;
		pos[i] = positions[src];
		nrm[i] = normals[src];
	}
	for ( int i = 0; i < nCount; i++ )
		outColors[i].Init();
	for ( int i = 0; i < 4; i++ )
		clusters[i] = ClusterFromPoint( pos[i] );

	for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
	{
		if ( dl->light.style )
			continue;

		float pvsMask[4];
		bool bAnyVisible = false;
		for ( int i = 0; i < 4; i++ )
		{
			pvsMask[i] = PVSCheck( dl->pvs, clusters[i] ) ? 1.0f : 0.0f;
			if ( i < nCount && pvsMask[i] > 0.0f )
				bAnyVisible = true;
		}
		if ( !bAnyVisible )
			continue;

		// push each vertex towards the light to avoid surface acne
		Vector adjusted[4];
		for ( int i = 0; i < 4; i++ )
		{
			adjusted[i] = pos[i];
			if ( dl->light.type != emit_skyambient )
			{
				Vector fudge;
				if ( dl->light.type == emit_skylight )
					fudge = -( dl->light.normal );
				else
				{
					fudge = dl->light.origin - pos[i];
					VectorNormalize( fudge );
				}
				adjusted[i] += fudge * 4.0f;
			}
			else
			{
				adjusted[i] += 4.0f * nrm[i];
			}
		}

		FourVectors adjusted_pos4;
		FourVectors normal4;
		adjusted_pos4.LoadAndSwizzle( adjusted[0], adjusted[1], adjusted[2], adjusted[3] );
		normal4.LoadAndSwizzle( nrm[0], nrm[1], nrm[2], nrm[3] );

		GatherSampleLightSSE( sampleOutput, dl, -1, adjusted_pos4, &normal4, 1, iThread,
							  nLFlags | GATHERLFLAGS_FORCE_FAST, static_prop_id_to_skip, 0.0f );

		for ( int i = 0; i < nCount; i++ )
		{
			float scale = pvsMask[i] * sampleOutput.m_flFalloff.m128_f32[i] * sampleOutput.m_flDot[0].m128_f32[i];
			if ( scale > 0.0f )
			{
				Vector intensity = dl->light.intensity;
				if ( dl->m_pProj )
				{
					Vector lightToSample = adjusted[i] - dl->light.origin;
					Vector rgb = Proj_EvalLightRGB( dl, lightToSample );
					intensity.x *= rgb.x;
					intensity.y *= rgb.y;
					intensity.z *= rgb.z;
				}
				Vector contrib = intensity * scale;
				contrib = IES_ClampDirectSample( dl, contrib );
				outColors[i] += contrib;
			}
		}
	}
}

//-----------------------------------------------------------------------------
// Takes the results from a ComputeLighting call and applies it to the static prop in question.
// Vertex colors are computed once against the shared studio vertex pool; every VTX LOD
// inherits those colors via origMeshVertID (same light for LOD0 / LOD1 / ...).
//-----------------------------------------------------------------------------
// The engine samples one lightmap for every material on a prop. After a split,
// copy the material's texels onto every mesh so whichever mesh the engine binds
// still shows that material.
static void MergeSplitLightmaps( CComputeStaticPropLightingResults *pRes )
{
	if ( !pRes )
		return;

	int n = 0;
	for ( int i = 0; i < pRes->m_ColorTexelsArrays.Count(); ++i )
	{
		CUtlVector<colorTexel_t> *a = pRes->m_ColorTexelsArrays[i];
		if ( a && a->Count() > n )
			n = a->Count();
	}
	if ( n <= 0 )
		return;

	CUtlVector<colorTexel_t> merged;
	merged.SetCount( n );
	memset( merged.Base(), 0, n * sizeof( colorTexel_t ) );
	for ( int t = 0; t < n; ++t )
		merged[t].m_fDistanceToTri = FLT_MAX;

	for ( int i = 0; i < pRes->m_ColorTexelsArrays.Count(); ++i )
	{
		CUtlVector<colorTexel_t> *a = pRes->m_ColorTexelsArrays[i];
		if ( !a || a->Count() != n )
			continue;
		for ( int t = 0; t < n; ++t )
		{
			const colorTexel_t &s = ( *a )[t];
			colorTexel_t &d = merged[t];
			if ( d.m_bValid )
				continue;
			if ( s.m_bValid || ( s.m_bPossiblyInteresting && !d.m_bPossiblyInteresting ) )
				d = s;
		}
	}

	for ( int i = 0; i < pRes->m_ColorTexelsArrays.Count(); ++i )
	{
		CUtlVector<colorTexel_t> *a = pRes->m_ColorTexelsArrays[i];
		if ( !a )
			continue;
		a->SetCount( n );
		Q_memcpy( a->Base(), merged.Base(), n * sizeof( colorTexel_t ) );
	}
}

void CVradStaticPropMgr::ApplyLightingToStaticProp( int iStaticProp, CStaticProp &prop, const CComputeStaticPropLightingResults *pResults )
{
	if ( pResults->m_ColorVertsArrays.Count() == 0 && pResults->m_ColorTexelsArrays.Count() == 0 )
		return;

	if ( prop.m_OnlyMaterial >= 0 )
		MergeSplitLightmaps( const_cast< CComputeStaticPropLightingResults * >( pResults ) );

	StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
	studiohdr_t	*pStudioHdr = dict.m_pStudioHdr;
	OptimizedModel::FileHeader_t *pVtxHdr = (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base();
	if ( !pStudioHdr || !pVtxHdr )
		return;

	prop.m_MeshData.RemoveAll();

	int iCurColorVertsArray = 0;
	int iCurColorTexelsArray = 0;

	for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts; ++bodyID )
	{
		OptimizedModel::BodyPartHeader_t* pVtxBodyPart = pVtxHdr->pBodyPart( bodyID );
		mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );

		for ( int modelID = 0; modelID < pBodyPart->nummodels; ++modelID )
		{
			OptimizedModel::ModelHeader_t* pVtxModel = pVtxBodyPart->pModel( modelID );
			mstudiomodel_t *pStudioModel = pBodyPart->pModel( modelID );
						
			const CUtlVector<colorVertex_t> *colorVerts = nullptr;
			if ( iCurColorVertsArray < pResults->m_ColorVertsArrays.Count() )
				colorVerts = pResults->m_ColorVertsArrays[iCurColorVertsArray++];
			const CUtlVector<colorTexel_t> *colorTexels = nullptr;
			if ( iCurColorTexelsArray < pResults->m_ColorTexelsArrays.Count() )
				colorTexels = pResults->m_ColorTexelsArrays[iCurColorTexelsArray++];

			const int nLods = min( pVtxHdr->numLODs, pVtxModel->numLODs );
			for ( int nLod = 0; nLod < nLods; nLod++ )
			{
				OptimizedModel::ModelLODHeader_t *pVtxLOD = pVtxModel->pLOD( nLod );
				const int nMeshes = min( pStudioModel->nummeshes, pVtxLOD->numMeshes );

				for ( int nMesh = 0; nMesh < nMeshes; ++nMesh )
				{
					mstudiomesh_t* pMesh = pStudioModel->pMesh( nMesh );
					OptimizedModel::MeshHeader_t* pVtxMesh = pVtxLOD->pMesh( nMesh );

					for ( int nGroup = 0; nGroup < pVtxMesh->numStripGroups; ++nGroup )
					{
						OptimizedModel::StripGroupHeader_t* pStripGroup = pVtxMesh->pStripGroup( nGroup );
						int nMeshIdx = prop.m_MeshData.AddToTail();

						if (colorVerts)
						{
							prop.m_MeshData[nMeshIdx].m_VertexColors.AddMultipleToTail( pStripGroup->numVerts );
							prop.m_MeshData[nMeshIdx].m_KeepCached.AddMultipleToTail( pStripGroup->numVerts );
							prop.m_MeshData[nMeshIdx].m_nLod = nLod;

							for ( int nVertex = 0; nVertex < pStripGroup->numVerts; ++nVertex )
							{
								int nIndex = pMesh->vertexoffset + pStripGroup->pVertex( nVertex )->origMeshVertID;

								Assert( nIndex < pStudioModel->numvertices );
								Vector worldPos( 0, 0, 0 );
								if ( nIndex >= 0 && nIndex < colorVerts->Count() )
								{
									prop.m_MeshData[nMeshIdx].m_VertexColors[nVertex] = (*colorVerts)[nIndex].m_Color;
									worldPos = (*colorVerts)[nIndex].m_Position;
								}
								else
									prop.m_MeshData[nMeshIdx].m_VertexColors[nVertex].Init();

								byte keep = 0;
								if ( BakeVolume_Active() &&
									 !BakeVolume_ShouldBakePropPoint( prop.m_Origin, PropBakeName( prop ), worldPos ) )
									keep = 1;
								prop.m_MeshData[nMeshIdx].m_KeepCached[nVertex] = keep;
							}
						}

						if (colorTexels)
						{
							// Same lightmap for every LOD - lower LODs inherit LOD0 texel lighting.
							CUtlVector<colorTexel_t> mergedTexels;
							mergedTexels.SetCount( colorTexels->Count() );
							Q_memcpy( mergedTexels.Base(), colorTexels->Base(), colorTexels->Count() * sizeof( colorTexel_t ) );
							if ( BakeVolume_Active() )
							{
								const int w = (int)prop.m_LightmapImageWidth;
								const int h = (int)prop.m_LightmapImageHeight;
								for ( int ti = 0; ti < mergedTexels.Count(); ++ti )
								{
									if ( BakeVolume_ShouldBakePropPoint( prop.m_Origin, PropBakeName( prop ), mergedTexels[ti].m_WorldPosition ) )
										continue;
									Vector cached;
									if ( BakeVolume_CachedPropTexelLinear( prop.m_Origin, PropBakeName( prop ), nMeshIdx, ti, w, h, cached ) )
										mergedTexels[ti].m_Color = cached;
								}
							}
							ConvertTexelDataToTexture(prop.m_LightmapImageWidth, prop.m_LightmapImageHeight, prop.m_LightmapImageFormat, mergedTexels, &prop.m_MeshData[nMeshIdx].m_TexelsEncoded);

							if (g_bDumpPropLightmaps)
							{
								char buffer[_MAX_PATH];
								V_snprintf( 
									buffer, 
									_MAX_PATH - 1, 
									"staticprop_lightmap_%d_%.0f_%.0f_%.0f_%s_%d_%d_%d_%d_%d.tga", 
									iStaticProp, 
									prop.m_Origin.x, 
									prop.m_Origin.y,
									prop.m_Origin.z,
									dict.m_pStudioHdr->pszName(), 
									bodyID, 
									modelID, 
									nLod, 
									nMesh, 
									nGroup 
								);

								for ( int i = 0; buffer[i]; ++i ) 
								{
									if (buffer[i] == '/' || buffer[i] == '\\')
										buffer[i] = '-';
								}
								DumpLightmapLinear( buffer, mergedTexels, prop.m_LightmapImageWidth, prop.m_LightmapImageHeight );
							}
						}
					}
				}
			}
		}
	}
}

//-----------------------------------------------------------------------------
// Trace rays from each unique vertex, accumulating direct and indirect
// sources at each ray termination. Use the winding data to distribute the unique vertexes
// into the rendering layout.
//-----------------------------------------------------------------------------
void CVradStaticPropMgr::ComputeLighting( CStaticProp &prop, int iThread, int prop_index, CComputeStaticPropLightingResults *pResults )
{
	CUtlVector<badVertex_t>		badVerts;

	StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
	studiohdr_t	*pStudioHdr = dict.m_pStudioHdr;
	OptimizedModel::FileHeader_t *pVtxHdr = (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base();
	if ( !pStudioHdr || !pVtxHdr )
	{
		// must have model and its verts for lighting computation
		// game will fallback to fullbright
		return;
	}

	const bool withVertexLighting = (prop.m_Flags & STATIC_PROP_NO_PER_VERTEX_LIGHTING) == 0;
	const bool withTexelLighting = (prop.m_Flags & STATIC_PROP_NO_PER_TEXEL_LIGHTING) == 0;

	if (!withVertexLighting && !withTexelLighting)
		return;

	const int skip_prop = PropSelfShadowSkip( prop, prop_index );
	const int nFlags = ( prop.m_Flags & STATIC_PROP_IGNORE_NORMALS ) ? GATHERLFLAGS_IGNORE_NORMALS : 0;
	const PropUvSplitPlan *pUvPlan = PropUvPlan( prop.m_UvPlan );
	const int uvBin = ( pUvPlan && prop.m_UvBin >= 0 ) ? prop.m_UvBin : -1;

#ifdef MPI
	VMPI_SetCurrentStage( "ComputeLighting" );
#endif

	matrix3x4_t	matPos, matNormal;
	AngleMatrix(prop.m_Angles, prop.m_Origin, matPos);
	AngleMatrix(prop.m_Angles, matNormal);
	
	for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts; ++bodyID )
	{
		OptimizedModel::BodyPartHeader_t* pVtxBodyPart = pVtxHdr->pBodyPart( bodyID );
		mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );

		for ( int modelID = 0; modelID < pBodyPart->nummodels; ++modelID )
		{
			OptimizedModel::ModelHeader_t* pVtxModel = pVtxBodyPart->pModel(modelID);
			mstudiomodel_t *pStudioModel = pBodyPart->pModel( modelID );

			CUtlVector<colorTexel_t> *pSectionTexels = NULL;
			bool resetTexels = true;
			if (withTexelLighting)
			{
				pSectionTexels = new CUtlVector<colorTexel_t>;
				pResults->m_ColorTexelsArrays.AddToTail(pSectionTexels);
			}
			
			// light all unique vertexes
			CUtlVector<colorVertex_t> *pColorVertsArray = new CUtlVector<colorVertex_t>;
			pResults->m_ColorVertsArrays.AddToTail( pColorVertsArray );
						
			CUtlVector<colorVertex_t> &colorVerts = *pColorVertsArray; 
			colorVerts.EnsureCount( pStudioModel->numvertices );
			memset( colorVerts.Base(), 0, colorVerts.Count() * sizeof(colorVertex_t) );

			int numVertexes = 0;
			for ( int meshID = 0; meshID < pStudioModel->nummeshes; ++meshID )
			{
				mstudiomesh_t *pStudioMesh = pStudioModel->pMesh( meshID );
				const mstudio_meshvertexdata_t *vertData = pStudioMesh->GetVertexData((void *)pStudioHdr);

				Assert(vertData); // This can only return NULL on X360 for now

				const bool meshKept = ( prop.m_OnlyMaterial < 0 || pStudioMesh->material == prop.m_OnlyMaterial );
				// TODO: Move this into its own function. In fact, refactor this whole function.
				if (withTexelLighting && meshKept)
				{
					GenerateLightmapSamplesForMesh( matPos, matNormal, iThread, skip_prop, nFlags, prop.m_LightmapImageWidth, prop.m_LightmapImageHeight, pStudioHdr, pStudioModel, pVtxModel, meshID, pResults, true, &prop.m_Origin, PropBakeName( prop ), resetTexels, ( uvBin >= 0 ) ? pUvPlan : NULL, uvBin, bodyID, modelID );
					resetTexels = false;
				}
				if ( !meshKept )
				{
					numVertexes += pStudioMesh->numvertices;
					continue;
				}

				// If we do lightmapping, we also do vertex lighting as a potential fallback. This may change.
				// Collect valid vertices first, then light them 4 per SSE gather call.
				CUtlVector<int> validIdx;
				CUtlVector<Vector> validPos;
				CUtlVector<Vector> validNrm;
				validIdx.EnsureCapacity( pStudioMesh->numvertices );
				validPos.EnsureCapacity( pStudioMesh->numvertices );
				validNrm.EnsureCapacity( pStudioMesh->numvertices );

				for ( int vertexID = 0; vertexID < pStudioMesh->numvertices; ++vertexID )
				{
					Vector sampleNormal;
					Vector samplePosition;
					// transform position and normal into world coordinate system
					VectorTransform(*vertData->Position(vertexID), matPos, samplePosition);
					VectorTransform(*vertData->Normal(vertexID), matNormal, sampleNormal);

					colorVerts[numVertexes].m_Position = samplePosition;

					if ( !BakeVolume_ShouldBakePropPoint( prop.m_Origin, PropBakeName( prop ), samplePosition ) )
					{
						numVertexes++;
						continue;
					}

					if ( PositionInSolid( samplePosition ) )
					{
						// vertex is in solid, add to the bad list, and recover later
						badVertex_t badVertex;
						badVertex.m_ColorVertex = numVertexes;
						badVertex.m_Position = samplePosition;
						badVertex.m_Normal = sampleNormal;
						badVerts.AddToTail( badVertex );			
					}
					else
					{
						validIdx.AddToTail( numVertexes );
						validPos.AddToTail( samplePosition );
						validNrm.AddToTail( sampleNormal );
					}

					numVertexes++;
				}

				for ( int base = 0; base < validIdx.Count(); base += 4 )
				{
					const int n = ( validIdx.Count() - base < 4 ) ? ( validIdx.Count() - base ) : 4;
					Vector directColors[4];

					if ( g_bShowStaticPropNormals )
					{
						for ( int i = 0; i < n; i++ )
						{
							directColors[i] = validNrm[base + i];
							directColors[i] += Vector( 1.0, 1.0, 1.0 );
							directColors[i] *= 50.0;
						}
					}
					else
					{
						ComputeDirectLightingAtPoints4( &validPos[base], &validNrm[base],
														directColors, n, iThread, skip_prop, nFlags );
					}

					for ( int i = 0; i < n; i++ )
					{
						const int ci = validIdx[base + i];
						Vector indirectColor( 0, 0, 0 );
						if ( !g_bShowStaticPropNormals && numbounce >= 1 )
							ComputeIndirectLightingAtPoint(
								validPos[base + i], validNrm[base + i],
								indirectColor, iThread, true,
								( prop.m_Flags & STATIC_PROP_IGNORE_NORMALS) != 0 );

						colorVerts[ci].m_bValid = true;
						colorVerts[ci].m_Position = validPos[base + i];
						VectorAdd( directColors[i], indirectColor, colorVerts[ci].m_Color );
					}
				}
			}
			
			// color in the bad vertexes
			// when entire model has no lighting origin and no valid neighbors
			// must punt, leave black coloring
			if ( badVerts.Count() && ( prop.m_bLightingOriginValid || badVerts.Count() != numVertexes ) )
			{
				for ( int nBadVertex = 0; nBadVertex < badVerts.Count(); nBadVertex++ )
				{		
					Vector bestPosition;
					if ( prop.m_bLightingOriginValid )
					{
						// use the specified lighting origin
						VectorCopy( prop.m_LightingOrigin, bestPosition );
					}
					else
					{
						// find the closest valid neighbor
						int best = 0;
						float closest = FLT_MAX;
						for ( int nColorVertex = 0; nColorVertex < numVertexes; nColorVertex++ )
						{
							if ( !colorVerts[nColorVertex].m_bValid )
							{
								// skip invalid neighbors
								continue;
							}
							Vector delta;
							VectorSubtract( colorVerts[nColorVertex].m_Position, badVerts[nBadVertex].m_Position, delta );
							float distance = VectorLength( delta );
							if ( distance < closest )
							{
								closest = distance;
								best    = nColorVertex;
							}
						}

						// use the best neighbor as the direction to crawl
						VectorCopy( colorVerts[best].m_Position, bestPosition );
					}

					// crawl toward best position
					// sudivide to determine a closer valid point to the bad vertex, and re-light
					Vector midPosition;
					int numIterations = 20;
					while ( --numIterations > 0 )
					{
						VectorAdd( bestPosition, badVerts[nBadVertex].m_Position, midPosition );
						VectorScale( midPosition, 0.5f, midPosition );
						if ( PositionInSolid( midPosition ) )
							break;
						bestPosition = midPosition;
					}

					// re-light from better position
					Vector directColor;
					ComputeDirectLightingAtPoint( bestPosition, badVerts[nBadVertex].m_Normal, directColor, iThread );

					Vector indirectColor;
					ComputeIndirectLightingAtPoint( bestPosition, badVerts[nBadVertex].m_Normal,
													indirectColor, iThread, true );

					// save results, not changing valid status
					// to ensure this offset position is not considered as a viable candidate
					colorVerts[badVerts[nBadVertex].m_ColorVertex].m_Position = bestPosition;
					VectorAdd( directColor, indirectColor, colorVerts[badVerts[nBadVertex].m_ColorVertex].m_Color );
				}
			}
			
			// discard bad verts
			badVerts.Purge();
		}
	}
}

//-----------------------------------------------------------------------------
// Write the lighitng to bsp pak lump
//-----------------------------------------------------------------------------
void CVradStaticPropMgr::SerializeLighting()
{
	char		filename[MAX_PATH];
	CUtlBuffer	utlBuf;

	// illuminate them all
	int count = m_StaticProps.Count();
	if ( !count )
	{
		// nothing to do
		return;
	}

	char mapName[MAX_PATH];
	Q_FileBase( source, mapName, sizeof( mapName ) );

	WriteStaticPropGameLump();

	int size;
	for (int i = 0; i < count; ++i)
	{
		// no need to write this file if we didn't compute the data
		// props marked this way will not load the info anyway 
		if ( m_StaticProps[i].m_Flags & STATIC_PROP_NO_PER_VERTEX_LIGHTING )
			continue;
		if ( !m_StaticPropDict[m_StaticProps[i].m_ModelIdx].m_pStudioHdr )
			continue;

		CStaticProp &prop = m_StaticProps[i];
		StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
		if ( !BakeVolume_ShouldBakePropOriented( prop.m_Origin, prop.m_Angles, dict.m_Mins, dict.m_Maxs ) )
		{
			if ( BakeVolume_ReinjectProp( i, prop.m_Origin, PropBakeName( prop ) ) )
				continue;
			continue;
		}

		if (g_bHDR)
		{
			sprintf( filename, "sp_hdr_%d.vhv", i );
		}
		else
		{
			sprintf( filename, "sp_%d.vhv", i );
		}

		int totalVertexes = 0;
		for ( int j=0; j<m_StaticProps[i].m_MeshData.Count(); j++ )
		{
			totalVertexes += m_StaticProps[i].m_MeshData[j].m_VertexColors.Count();
		}

		// allocate a buffer with enough padding for alignment
		size = sizeof( HardwareVerts::FileHeader_t ) + 
				m_StaticProps[i].m_MeshData.Count()*sizeof(HardwareVerts::MeshHeader_t) +
				totalVertexes*4 + 2*512;
		utlBuf.EnsureCapacity( size );
		Q_memset( utlBuf.Base(), 0, size );

		HardwareVerts::FileHeader_t *pVhvHdr = (HardwareVerts::FileHeader_t *)utlBuf.Base();

		// align to start of vertex data
		const size_t vertexDataOff = ALIGN_TO_POW2(
			sizeof( HardwareVerts::FileHeader_t ) + m_StaticProps[i].m_MeshData.Count() * sizeof( HardwareVerts::MeshHeader_t ),
			512 );
		unsigned char *pVertexData = (unsigned char *)pVhvHdr + vertexDataOff;
		
		// construct header
		pVhvHdr->m_nVersion     = VHV_VERSION;
		pVhvHdr->m_nChecksum    = m_StaticPropDict[m_StaticProps[i].m_ModelIdx].m_pStudioHdr->checksum;
		pVhvHdr->m_nVertexFlags = VERTEX_COLOR;
		pVhvHdr->m_nVertexSize  = 4;
		pVhvHdr->m_nVertexes    = totalVertexes;
		pVhvHdr->m_nMeshes      = m_StaticProps[i].m_MeshData.Count();

		for (int n=0; n<pVhvHdr->m_nMeshes; n++)
		{
			// construct mesh dictionary
			HardwareVerts::MeshHeader_t *pMesh = pVhvHdr->pMesh( n );
			pMesh->m_nLod      = m_StaticProps[i].m_MeshData[n].m_nLod;
			pMesh->m_nVertexes = m_StaticProps[i].m_MeshData[n].m_VertexColors.Count();
			pMesh->m_nOffset   = (unsigned int)( pVertexData - (unsigned char *)pVhvHdr );

			// construct vertexes
			for (int k=0; k<pMesh->m_nVertexes; k++)
			{
				if ( k < m_StaticProps[i].m_MeshData[n].m_KeepCached.Count() &&
					 m_StaticProps[i].m_MeshData[n].m_KeepCached[k] )
				{
					if ( BakeVolume_CopyCachedPropVertBGRA( m_StaticProps[i].m_Origin,
							PropBakeName( m_StaticProps[i] ),
							n, k, pVertexData ) )
					{
						pVertexData += 4;
						continue;
					}
				}

				Vector &vertexColor = m_StaticProps[i].m_MeshData[n].m_VertexColors[k];

				ColorRGBExp32 rgbColor;
				VectorToColorRGBExp32( vertexColor, rgbColor );
				unsigned char dstColor[4];
				ConvertRGBExp32ToRGBA8888( &rgbColor, dstColor );

				// b,g,r,a order
				pVertexData[0] = dstColor[2];
				pVertexData[1] = dstColor[1];
				pVertexData[2] = dstColor[0];
				pVertexData[3] = dstColor[3];
				pVertexData += 4;
			}
		}

		// align to end of file
		const size_t endOff = ALIGN_TO_POW2( (size_t)( pVertexData - (unsigned char *)pVhvHdr ), 512 );
		AddBufferToPak( GetPakFile(), filename, (void*)pVhvHdr, (int)endOff, false );
		BakeVolume_NoteProp( m_StaticProps[i].m_Origin, PropBakeName( m_StaticProps[i] ),
							 pVhvHdr, (int)endOff, NULL, 0 );
	}

	for (int i = 0; i < count; ++i)
	{
		const int kAlignment = 512;
		// no need to write this file if we didn't compute the data
		// props marked this way will not load the info anyway 
		if (m_StaticProps[i].m_Flags & STATIC_PROP_NO_PER_TEXEL_LIGHTING)
			continue;
		if ( !m_StaticPropDict[m_StaticProps[i].m_ModelIdx].m_pStudioHdr )
			continue;
		if ( !BakeVolume_ShouldBakePropOriented( m_StaticProps[i].m_Origin, m_StaticProps[i].m_Angles,
												 m_StaticPropDict[m_StaticProps[i].m_ModelIdx].m_Mins,
												 m_StaticPropDict[m_StaticProps[i].m_ModelIdx].m_Maxs ) )
			continue;

		sprintf(filename, "texelslighting_%d.ppl", i);

		ImageFormat fmt = m_StaticProps[i].m_LightmapImageFormat;

		unsigned int totalTexelSizeBytes = 0;
		for (int j = 0; j < m_StaticProps[i].m_MeshData.Count(); j++)
		{
			totalTexelSizeBytes += m_StaticProps[i].m_MeshData[j].m_TexelsEncoded.Count();
		}

		// allocate a buffer with enough padding for alignment
		size = sizeof(HardwareTexels::FileHeader_t) 
			 + m_StaticProps[i].m_MeshData.Count() * sizeof(HardwareTexels::MeshHeader_t) 
			 + totalTexelSizeBytes
			 + 2 * kAlignment;
		
		utlBuf.EnsureCapacity(size);
		Q_memset(utlBuf.Base(), 0, size);

		HardwareTexels::FileHeader_t *pVhtHdr = (HardwareTexels::FileHeader_t *)utlBuf.Base();

		const size_t texelDataOff = ALIGN_TO_POW2(
			sizeof(HardwareTexels::FileHeader_t) + m_StaticProps[i].m_MeshData.Count() * sizeof(HardwareTexels::MeshHeader_t),
			kAlignment );
		unsigned char *pTexelData = (unsigned char *)pVhtHdr + texelDataOff;

		pVhtHdr->m_nVersion	    = VHT_VERSION;
		pVhtHdr->m_nChecksum    = m_StaticPropDict[m_StaticProps[i].m_ModelIdx].m_pStudioHdr->checksum;
		pVhtHdr->m_nTexelFormat = fmt;
		pVhtHdr->m_nMeshes      = m_StaticProps[i].m_MeshData.Count();

		for (int n = 0; n < pVhtHdr->m_nMeshes; n++)
		{
			HardwareTexels::MeshHeader_t *pMesh = pVhtHdr->pMesh(n);
			pMesh->m_nLod = m_StaticProps[i].m_MeshData[n].m_nLod;
			pMesh->m_nOffset = (unsigned int)( pTexelData - (unsigned char *)pVhtHdr );
			pMesh->m_nBytes = m_StaticProps[i].m_MeshData[n].m_TexelsEncoded.Count();
			pMesh->m_nWidth = m_StaticProps[i].m_LightmapImageWidth;
			pMesh->m_nHeight = m_StaticProps[i].m_LightmapImageHeight;

			Q_memcpy(pTexelData, m_StaticProps[i].m_MeshData[n].m_TexelsEncoded.Base(), m_StaticProps[i].m_MeshData[n].m_TexelsEncoded.Count());
			pTexelData += m_StaticProps[i].m_MeshData[n].m_TexelsEncoded.Count();
		}

		const size_t endOff = ALIGN_TO_POW2( (size_t)( pTexelData - (unsigned char *)pVhtHdr ), kAlignment );
		AddBufferToPak(GetPakFile(), filename, (void*)pVhtHdr, (int)endOff, false);
		BakeVolume_NoteProp( m_StaticProps[i].m_Origin, PropBakeName( m_StaticProps[i] ),
							 NULL, 0, pVhtHdr, (int)endOff );
	}
}

#ifdef MPI
void CVradStaticPropMgr::VMPI_ProcessStaticProp_Static( int iThread, uint64 iStaticProp, MessageBuffer *pBuf )
{
	g_StaticPropMgr.VMPI_ProcessStaticProp( iThread, iStaticProp, pBuf );
}

void CVradStaticPropMgr::VMPI_ReceiveStaticPropResults_Static( uint64 iStaticProp, MessageBuffer *pBuf, int iWorker )
{
	g_StaticPropMgr.VMPI_ReceiveStaticPropResults( iStaticProp, pBuf, iWorker );
}
	
//-----------------------------------------------------------------------------
// Called on workers to do the computation for a static prop and send
// it to the master.
//-----------------------------------------------------------------------------
void CVradStaticPropMgr::VMPI_ProcessStaticProp( int iThread, int iStaticProp, MessageBuffer *pBuf )
{
	// Compute the lighting.
	CComputeStaticPropLightingResults results;
	ComputeLighting( m_StaticProps[iStaticProp], iThread, iStaticProp, &results );

#ifdef MPI
	VMPI_SetCurrentStage( "EncodeLightingResults" );
#endif
	
	// Encode the results.
	int nLists = results.m_ColorVertsArrays.Count();
	pBuf->write( &nLists, sizeof( nLists ) );
	
	for ( int i=0; i < nLists; i++ )
	{
		CUtlVector<colorVertex_t> &curList = *results.m_ColorVertsArrays[i];
		int count = curList.Count();
		pBuf->write( &count, sizeof( count ) );
		pBuf->write( curList.Base(), curList.Count() * sizeof( colorVertex_t ) );
	}

	nLists = results.m_ColorTexelsArrays.Count();
	pBuf->write(&nLists, sizeof(nLists));

	for (int i = 0; i < nLists; i++)
	{
		CUtlVector<colorTexel_t> &curList = *results.m_ColorTexelsArrays[i];
		int count = curList.Count();
		pBuf->write(&count, sizeof(count));
		pBuf->write(curList.Base(), curList.Count() * sizeof(colorTexel_t));
	}
}

//-----------------------------------------------------------------------------
// Called on the master when a worker finishes processing a static prop.
//-----------------------------------------------------------------------------
void CVradStaticPropMgr::VMPI_ReceiveStaticPropResults( int iStaticProp, MessageBuffer *pBuf, int iWorker )
{
	// Read in the results.
	CComputeStaticPropLightingResults results;
	
	int nLists;
	pBuf->read( &nLists, sizeof( nLists ) );
	
	for ( int i=0; i < nLists; i++ )
	{
		CUtlVector<colorVertex_t> *pList = new CUtlVector<colorVertex_t>;
		results.m_ColorVertsArrays.AddToTail( pList );
		
		int count;
		pBuf->read( &count, sizeof( count ) );
		pList->SetSize( count );
		pBuf->read( pList->Base(), count * sizeof( colorVertex_t ) );
	}

	pBuf->read(&nLists, sizeof(nLists));

	for (int i = 0; i < nLists; i++)
	{
		CUtlVector<colorTexel_t> *pList = new CUtlVector<colorTexel_t>;
		results.m_ColorTexelsArrays.AddToTail(pList);

		int count;
		pBuf->read(&count, sizeof(count));
		pList->SetSize(count);
		pBuf->read(pList->Base(), count * sizeof(colorTexel_t));
	}
	
	// Apply the results.
	ApplyLightingToStaticProp( iStaticProp, m_StaticProps[iStaticProp], &results );
}
#endif

void CVradStaticPropMgr::ComputeLightingForProp( int iThread, int iStaticProp )
{
	CStaticProp &prop = m_StaticProps[iStaticProp];
	StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
	if ( !BakeVolume_ShouldBakePropOriented( prop.m_Origin, prop.m_Angles, dict.m_Mins, dict.m_Maxs ) )
		return;

	CComputeStaticPropLightingResults results;
	ComputeLighting( prop, iThread, iStaticProp, &results );
	ApplyLightingToStaticProp( iStaticProp, prop, &results );
}

void CVradStaticPropMgr::ThreadComputeStaticPropLighting( int iThread, void *pUserData )
{
	while (1)
	{
		int j = GetThreadWork ();
		if (j == -1)
			break;
		CComputeStaticPropLightingResults results;
		g_StaticPropMgr.ComputeLightingForProp( iThread, j );
	}
}

//-----------------------------------------------------------------------------
// Pathtrace GPU bake for static props (same PathLi as world luxels).
//-----------------------------------------------------------------------------
struct PtPropVertTemplate
{
	CUtlVector<Vector> localPos;
	CUtlVector<Vector> localNrm;
};

struct PtPropTexelTemplate
{
	int resX = 0;
	int resY = 0;
	int material = -1;
	int uvBin = -1;
	CUtlVector<Vector> localPos;
	CUtlVector<Vector> localNrm;
	CUtlVector<Vector> localAxisU;
	CUtlVector<Vector> localAxisV;
	CUtlVector<float> luxelWorld;
	CUtlVector<bool> valid;
	CUtlVector<bool> interesting;
	// 1 = path-trace this texel. Interiors always. Gutter texels only when no
	// same-chart interior is next to them (a triangle that missed the grid).
	CUtlVector<unsigned char> bake;
};

struct PtPropModelTemplate
{
	CUtlVector<PtPropVertTemplate> vertSections; // one per bodyxmodel
	// Per section, one rasterization per lightmap resolution (instances of the
	// same model often have different baked lightmap sizes).
	CUtlVector<CUtlVector<PtPropTexelTemplate>> texelBySection;
};

static unsigned int PtPropSampleSeed( const Vector &pos, int propIndex )
{
	unsigned int h = (unsigned int)propIndex * 747796405u;
	h ^= (unsigned int)( pos.x * 12.9898f + pos.y * 78.233f + pos.z * 37.719f );
	h *= 1597334677u;
	return h;
}

// Same quantized world hash as brush luxels, so neighboring prop texels
// don't share a correlated sample sequence.
static unsigned int PtPropLightmapSeed( const Vector &pos, float luxelWorld )
{
	const float cell = ( luxelWorld > 0.5f ) ? luxelWorld : 2.0f;
	const int qx = (int)floorf( pos.x / cell );
	const int qy = (int)floorf( pos.y / cell );
	const int qz = (int)floorf( pos.z / cell );
	unsigned int h = (unsigned int)qx * 73856093u ^ (unsigned int)qy * 19349663u ^ (unsigned int)qz * 83492791u;
	h ^= h >> 16;
	return h ? h : 1u;
}

static void PtPackBakeLuxel( PtGpuBakeLuxel &job, const Vector &pos, const Vector &nrm, int skipProp, int propIndex )
{
	memset( &job, 0, sizeof( job ) );
	job.pos[0] = pos.x; job.pos[1] = pos.y; job.pos[2] = pos.z;
	job.luxelWorld = 2.0f;
	Vector n = nrm;
	if ( n.LengthSqr() < 1e-8f )
		n.Init( 0, 0, 1 );
	else
		n.NormalizeInPlace();
	job.normal[0] = n.x; job.normal[1] = n.y; job.normal[2] = n.z;
	job.seed = PtPropSampleSeed( pos, propIndex );
	job.faceNum = -1;
	job.aaN = 1;
	job.skipPropIndex = skipProp;
	job.sampleScale = 1;
}

// Prop lightmap texels: same footprint AA and luxel size as world faces.
// bFullSelfSkip: drop every hit on meshProp (no-self-shadow flag).
// Otherwise padU is a near distance: shadow rays ignore same-facing hits on
// this prop inside that radius so tessellation doesn't speckle the lightmap.
static void PtPackBakeLuxelLightmap( PtGpuBakeLuxel &job, const Vector &pos, const Vector &nrm,
									  const Vector &axisU, const Vector &axisV, float luxelWorld,
									  int meshProp, int propIndex, bool bFullSelfSkip )
{
	PtPackBakeLuxel( job, pos, nrm, meshProp, propIndex );
	if ( luxelWorld < 0.25f )
		luxelWorld = 2.0f;
	job.luxelWorld = luxelWorld;
	job.axisU[0] = axisU.x; job.axisU[1] = axisU.y; job.axisU[2] = axisU.z;
	job.axisV[0] = axisV.x; job.axisV[1] = axisV.y; job.axisV[2] = axisV.z;
	job.skipPropIndex = meshProp;
	job.padU = bFullSelfSkip ? 0.0f : max( luxelWorld * 8.0f, 8.0f );
	job.seed = PtPropLightmapSeed( pos - nrm * luxelWorld, luxelWorld );
	int aaN = g_nPathTraceLuxelAA;
	if ( aaN < 1 ) aaN = 1;
	if ( aaN > 5 ) aaN = 5;
	if ( axisU.LengthSqr() < 1e-8f || axisV.LengthSqr() < 1e-8f )
		aaN = 1;
	job.aaN = aaN;
}

// Lower LODs share the studio vertex pool. Verts never hit by the LOD0 sample grid
// (or only used by higher LODs) inherit color from the nearest lit vertex.
static void PtInheritUnlitPropVertColors( CUtlVector<colorVertex_t> &verts,
										  const CUtlVector<float> &weights,
										  const CUtlVector<Vector> &localPos )
{
	const int n = verts.Count();
	if ( n <= 0 || weights.Count() < n || localPos.Count() < n )
		return;

	CUtlVector<int> lit;
	lit.EnsureCapacity( n );
	for ( int i = 0; i < n; ++i )
	{
		if ( weights[i] > 1e-8f && verts[i].m_Color.LengthSqr() > 0.0f )
			lit.AddToTail( i );
	}
	if ( lit.Count() <= 0 )
		return;

	for ( int i = 0; i < n; ++i )
	{
		if ( weights[i] > 1e-8f )
			continue;
		float best = FLT_MAX;
		int bestIdx = lit[0];
		const Vector &p = localPos[i];
		for ( int k = 0; k < lit.Count(); ++k )
		{
			const int j = lit[k];
			const float d = ( localPos[j] - p ).LengthSqr();
			if ( d < best )
			{
				best = d;
				bestIdx = j;
			}
		}
		verts[i].m_Color = verts[bestIdx].m_Color;
		verts[i].m_bValid = verts[bestIdx].m_bValid;
	}
}

static float PropTexelLuma( const Vector &v )
{
	return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

// Neighboring lightmap texels are one chart when they sit on the same surface.
// A different UV island packed next door is far away in world space, and a hard
// edge (the rock cracks in a lightmapped prop) has a different normal. The 3x3
// encode filter, dilate, and denoise must not blend those.
static bool PropTexelChartsMatch( const colorTexel_t &a, const colorTexel_t &b )
{
	float luxel = 0.0f;
	if ( a.m_LuxelWorld > 0.25f )
		luxel = a.m_LuxelWorld;
	if ( b.m_LuxelWorld > luxel )
		luxel = b.m_LuxelWorld;
	if ( luxel < 0.25f )
		luxel = 2.0f;

	const Vector delta = a.m_WorldPosition - b.m_WorldPosition;
	const float maxDist = luxel * 3.0f;
	if ( delta.LengthSqr() > maxDist * maxDist )
		return false;

	if ( a.m_WorldNormal.LengthSqr() > 1e-6f && b.m_WorldNormal.LengthSqr() > 1e-6f )
	{
		// About 60 degrees. Shallower bends still filter; a corner does not.
		if ( a.m_WorldNormal.Dot( b.m_WorldNormal ) < 0.50f )
			return false;
	}
	return true;
}

// Prop texels inside a brush stay marked valid with a black bake, and denoise
// then smears that black onto the visible chart. Drop them from the chart and
// lift the one straddling texel from the visible side. Dilate fills the hole.
static void PtHealPropBrushIntersection( CUtlVector<colorTexel_t> &texels, int resX, int resY )
{
	if ( resX <= 1 || resY <= 1 || texels.Count() != resX * resY )
		return;

	const int n = resX * resY;
	CUtlVector<char> embedded;
	CUtlVector<char> partial;
	embedded.SetCount( n );
	partial.SetCount( n );
	int nEmb = 0;
	for ( int i = 0; i < n; ++i )
	{
		embedded[i] = 0;
		partial[i] = 0;
		colorTexel_t &ct = texels[i];
		if ( !ct.m_bValid )
			continue;

		Vector p = ct.m_WorldPosition;
		const bool centerIn = PositionInSolid( p );
		int cornersIn = 0;
		if ( ct.m_AxisU.LengthSqr() > 1e-8f && ct.m_AxisV.LengthSqr() > 1e-8f )
		{
			const Vector hu = ct.m_AxisU * 0.5f;
			const Vector hv = ct.m_AxisV * 0.5f;
			Vector c0 = p + hu + hv;
			Vector c1 = p + hu - hv;
			Vector c2 = p - hu + hv;
			Vector c3 = p - hu - hv;
			if ( PositionInSolid( c0 ) ) ++cornersIn;
			if ( PositionInSolid( c1 ) ) ++cornersIn;
			if ( PositionInSolid( c2 ) ) ++cornersIn;
			if ( PositionInSolid( c3 ) ) ++cornersIn;
		}
		if ( centerIn || cornersIn >= 2 )
		{
			embedded[i] = 1;
			++nEmb;
		}
		else if ( cornersIn == 1 )
		{
			partial[i] = 1;
		}
	}
	if ( !nEmb )
		return;

	const int kDx[4] = { 1, -1, 0, 0 };
	const int kDy[4] = { 0, 0, 1, -1 };
	for ( int y = 0; y < resY; ++y )
	{
		for ( int x = 0; x < resX; ++x )
		{
			const int idx = y * resX + x;
			if ( embedded[idx] || !texels[idx].m_bValid )
				continue;
			if ( !partial[idx] )
			{
				bool beside = false;
				for ( int k = 0; k < 4 && !beside; ++k )
				{
					const int nx = x + kDx[k];
					const int ny = y + kDy[k];
					if ( nx < 0 || ny < 0 || nx >= resX || ny >= resY )
						continue;
					if ( embedded[ny * resX + nx] )
						beside = true;
				}
				if ( !beside )
					continue;
			}

			float refL = -1.0f;
			Vector ref( 0, 0, 0 );
			bool agree = true;
			int nRef = 0;
			bool touch = false;
			for ( int k = 0; k < 4; ++k )
			{
				const int nx = x + kDx[k];
				const int ny = y + kDy[k];
				if ( nx < 0 || ny < 0 || nx >= resX || ny >= resY )
					continue;
				const int ni = ny * resX + nx;
				if ( embedded[ni] )
				{
					touch = true;
					const int ox = x - kDx[k];
					const int oy = y - kDy[k];
					if ( ox < 0 || oy < 0 || ox >= resX || oy >= resY )
						continue;
					const int oi = oy * resX + ox;
					if ( embedded[oi] || partial[oi] || !texels[oi].m_bValid )
						continue;
					if ( !PropTexelChartsMatch( texels[idx], texels[oi] ) )
						continue;
					const float ol = PropTexelLuma( texels[oi].m_Color );
					if ( !nRef )
					{
						refL = ol;
						ref = texels[oi].m_Color;
					}
					else if ( fabsf( ol - refL ) / max( max( ol, refL ), 1e-4f ) > 0.35f )
						agree = false;
					++nRef;
					continue;
				}
				if ( !partial[idx] || partial[ni] || !texels[ni].m_bValid )
					continue;
				if ( !PropTexelChartsMatch( texels[idx], texels[ni] ) )
					continue;
				const float ol = PropTexelLuma( texels[ni].m_Color );
				if ( !nRef )
				{
					refL = ol;
					ref = texels[ni].m_Color;
				}
				else if ( fabsf( ol - refL ) / max( max( ol, refL ), 1e-4f ) > 0.35f )
					agree = false;
				++nRef;
			}
			if ( partial[idx] )
				touch = true;
			if ( !touch || !agree || !nRef || refL <= 0.02f )
				continue;
			if ( PropTexelLuma( texels[idx].m_Color ) < refL * 0.90f )
				texels[idx].m_Color = ref;
		}
	}

	for ( int i = 0; i < n; ++i )
	{
		if ( !embedded[i] )
			continue;
		texels[i].m_bValid = false;
		texels[i].m_Color.Init();
		texels[i].m_bPossiblyInteresting = true;
	}
}

static void PtDilatePropTexels( CUtlVector<colorTexel_t> &texels, int resX, int resY )
{
	if ( resX <= 0 || resY <= 0 || texels.Count() != resX * resY )
		return;

	// Two passes: fill interesting neighbors from valid, then any remaining holes from lit neighbors.
	for ( int pass = 0; pass < 2; ++pass )
	{
		CUtlVector<Vector> staged;
		CUtlVector<int> stagedIdx;
		for ( int y = 0; y < resY; ++y )
		{
			for ( int x = 0; x < resX; ++x )
			{
				const int i = y * resX + x;
				colorTexel_t &ct = texels[i];
				if ( ct.m_bValid )
					continue;
				if ( pass == 0 && !ct.m_bPossiblyInteresting )
					continue;
				if ( ct.m_Color.LengthSqr() > 1e-12f )
					continue;

				Vector sum( 0, 0, 0 );
				int n = 0;
				for ( int dy = -1; dy <= 1; ++dy )
				{
					for ( int dx = -1; dx <= 1; ++dx )
					{
						if ( !dx && !dy )
							continue;
						const int nx = x + dx, ny = y + dy;
						if ( nx < 0 || ny < 0 || nx >= resX || ny >= resY )
							continue;
						const colorTexel_t &nb = texels[ny * resX + nx];
						if ( !nb.m_bValid && nb.m_Color.LengthSqr() < 1e-12f )
							continue;
						if ( !PropTexelChartsMatch( ct, nb ) )
							continue;
						sum += nb.m_Color;
						++n;
					}
				}
				if ( n > 0 )
				{
					staged.AddToTail( sum * ( 1.0f / (float)n ) );
					stagedIdx.AddToTail( i );
				}
			}
		}
		for ( int k = 0; k < stagedIdx.Count(); ++k )
			texels[stagedIdx[k]].m_Color = staged[k];
	}
}

// Same bright-spike clamp brush luxels get before denoise. Dense grid (padding included).
static void PtSuppressDenseOutliers( Vector *grid, int width, int height )
{
	if ( !grid || width < 2 || height < 2 )
		return;

	for ( int t = 0; t < height; ++t )
	{
		for ( int s = 0; s < width; ++s )
		{
			const int idx = s + t * width;
			float neighL[8];
			int nNeigh = 0;
			for ( int dt = -1; dt <= 1; ++dt )
			{
				const int nt = t + dt;
				if ( nt < 0 || nt >= height )
					continue;
				for ( int ds = -1; ds <= 1; ++ds )
				{
					if ( !ds && !dt )
						continue;
					const int ns = s + ds;
					if ( ns < 0 || ns >= width )
						continue;
					neighL[nNeigh++] = PropTexelLuma( grid[ns + nt * width] );
				}
			}
			if ( nNeigh < 5 )
				continue;

			for ( int a = 1; a < nNeigh; ++a )
			{
				const float v = neighL[a];
				int b = a;
				while ( b > 0 && neighL[b - 1] > v )
				{
					neighL[b] = neighL[b - 1];
					--b;
				}
				neighL[b] = v;
			}
			const float med = neighL[nNeigh / 2];
			float maxN = 0.0f;
			for ( int k = 0; k < nNeigh; ++k )
				maxN = max( maxN, neighL[k] );

			const float L = PropTexelLuma( grid[idx] );
			if ( med > 0.02f && L > med * 3.0f && L > maxN * 2.0f )
				grid[idx] *= ( med * 1.05f ) / max( L, 1e-4f );
		}
	}
}

// Same one-sided high-contrast AA as brush luxels, after denoise.
static void PtAntialiasDense( Vector *grid, int width, int height )
{
	if ( !grid || width < 2 || height < 2 )
		return;

	const int n = width * height;
	CUtlVector<Vector> out;
	CUtlVector<float> luma;
	out.SetCount( n );
	luma.SetCount( n );

	for ( int pass = 0; pass < 2; ++pass )
	{
		for ( int i = 0; i < n; ++i )
			luma[i] = PropTexelLuma( grid[i] );

		for ( int y = 0; y < height; ++y )
		{
			for ( int x = 0; x < width; ++x )
			{
				const int ci = y * width + x;
				out[ci] = grid[ci];
				const float lc = luma[ci];
				float maxRel = 0.0f;
				Vector sum( 0, 0, 0 );
				float wsum = 0.0f;
				for ( int dy = -1; dy <= 1; ++dy )
				{
					const int ny = y + dy;
					if ( ny < 0 || ny >= height )
						continue;
					for ( int dx = -1; dx <= 1; ++dx )
					{
						const int nx = x + dx;
						if ( nx < 0 || nx >= width )
							continue;
						const int ni = ny * width + nx;
						const float ln = luma[ni];
						const float denom = max( max( lc, ln ), 1e-4f );
						const float rel = fabsf( lc - ln ) / denom;
						if ( rel > maxRel )
							maxRel = rel;
						const float spat = ( dx == 0 && dy == 0 ) ? 1.0f
							: ( ( dx == 0 || dy == 0 ) ? 0.85f : 0.45f );
						sum += grid[ni] * spat;
						wsum += spat;
					}
				}
				float aa = ( maxRel - 0.25f ) / 0.30f;
				if ( aa < 0.0f )
					aa = 0.0f;
				if ( aa > 1.0f )
					aa = 1.0f;
				aa *= aa;
				if ( aa < 1e-4f || wsum < 1e-6f )
					continue;
				const Vector avg = sum * ( 1.0f / wsum );
				if ( PropTexelLuma( avg ) < lc * 0.92f )
					continue;
				out[ci] = grid[ci] + ( avg - grid[ci] ) * ( aa * 0.65f );
			}
		}
		for ( int i = 0; i < n; ++i )
			grid[i] = out[i];
	}
}

// Denoise each UV island on its own, edge-extended so gutters stay out of the filter.
// A single atlas pass would mix unrelated charts and darken them against empty padding.
static void PtDenoisePropLightmap( CUtlVector<colorTexel_t> &texels, int resX, int resY )
{
	if ( !g_bPathTraceDenoise || !PathTraceDenoise_IsReady() )
		return;
	if ( resX <= 1 || resY <= 1 || texels.Count() != resX * resY )
		return;

	const int n = resX * resY;
	const float blend = min( 1.0f, max( 0.0f, g_flPathTraceDenoiseStrength ) );
	CUtlVector<int> comp;
	comp.SetCount( n );
	memset( comp.Base(), 0, n * sizeof( int ) );

	auto inChart = [&]( int i ) -> bool
	{
		if ( texels[i].m_bValid )
			return true;
		return texels[i].m_bPossiblyInteresting && texels[i].m_Color.LengthSqr() > 1e-12f;
	};

	CUtlVector<int> stack;
	stack.EnsureCapacity( 1024 );
	int nextId = 1;

	for ( int seed = 0; seed < n; ++seed )
	{
		if ( comp[seed] != 0 || !inChart( seed ) )
			continue;

		const int id = nextId++;
		int count = 0;
		int minX = resX, minY = resY, maxX = 0, maxY = 0;
		int sp = 0;
		stack.RemoveAll();
		stack.AddToTail( seed );
		comp[seed] = id;
		while ( sp < stack.Count() )
		{
			const int i = stack[sp++];
			const int x = i % resX;
			const int y = i / resX;
			if ( x < minX ) minX = x;
			if ( y < minY ) minY = y;
			if ( x > maxX ) maxX = x;
			if ( y > maxY ) maxY = y;
			++count;

			const int nb[4] = {
				( x > 0 ) ? i - 1 : -1,
				( x + 1 < resX ) ? i + 1 : -1,
				( y > 0 ) ? i - resX : -1,
				( y + 1 < resY ) ? i + resX : -1
			};
			for ( int k = 0; k < 4; ++k )
			{
				const int j = nb[k];
				if ( j < 0 || comp[j] != 0 || !inChart( j ) )
					continue;
				if ( !PropTexelChartsMatch( texels[i], texels[j] ) )
					continue;
				comp[j] = id;
				stack.AddToTail( j );
			}
		}

		const int pad = 8;
		const int bw = ( maxX - minX + 1 ) + pad * 2;
		const int bh = ( maxY - minY + 1 ) + pad * 2;
		if ( bw < 4 || bh < 4 )
			continue;

		CUtlVector<Vector> img;
		CUtlVector<unsigned char> filled;
		img.SetCount( bw * bh );
		filled.SetCount( bw * bh );
		memset( img.Base(), 0, img.Count() * sizeof( Vector ) );
		memset( filled.Base(), 0, filled.Count() );

		for ( int i = 0; i < stack.Count(); ++i )
		{
			const int src = stack[i];
			if ( comp[src] != id )
				continue;
			const int x = src % resX;
			const int y = src / resX;
			const int dx = ( x - minX ) + pad;
			const int dy = ( y - minY ) + pad;
			const int di = dx + dy * bw;
			img[di] = texels[src].m_Color;
			filled[di] = 1;
		}

		CUtlVector<Vector> next;
		CUtlVector<unsigned char> nextF;
		next.SetCount( bw * bh );
		nextF.SetCount( bw * bh );
		// Fill the whole pad. A short pass count leaves black corners, and the
		// denoiser then pulls that black into a chart that is only a few texels.
		for ( int pass = 0; pass < bw + bh; ++pass )
		{
			memcpy( next.Base(), img.Base(), img.Count() * sizeof( Vector ) );
			memcpy( nextF.Base(), filled.Base(), filled.Count() );
			bool any = false;
			for ( int y = 0; y < bh; ++y )
			{
				for ( int x = 0; x < bw; ++x )
				{
					const int i = x + y * bw;
					if ( filled[i] )
						continue;
					Vector sum( 0, 0, 0 );
					int nn = 0;
					const int ox[4] = { -1, 1, 0, 0 };
					const int oy[4] = { 0, 0, -1, 1 };
					for ( int k = 0; k < 4; ++k )
					{
						const int nx = x + ox[k];
						const int ny = y + oy[k];
						if ( nx < 0 || ny < 0 || nx >= bw || ny >= bh )
							continue;
						const int j = nx + ny * bw;
						if ( !filled[j] )
							continue;
						sum += img[j];
						++nn;
					}
					if ( nn > 0 )
					{
						next[i] = sum * ( 1.0f / (float)nn );
						nextF[i] = 1;
						any = true;
					}
				}
			}
			img.Swap( next );
			filled.Swap( nextF );
			if ( !any )
				break;
		}

		PtSuppressDenseOutliers( img.Base(), bw, bh );

		CUtlVector<float> flat;
		flat.SetCount( bw * bh * 3 );
		for ( int i = 0; i < bw * bh; ++i )
		{
			flat[i * 3 + 0] = img[i].x;
			flat[i * 3 + 1] = img[i].y;
			flat[i * 3 + 2] = img[i].z;
		}
		if ( PathTraceDenoise_DenoiseRGB( flat.Base(), bw, bh ) )
		{
			for ( int i = 0; i < bw * bh; ++i )
				img[i].Init( flat[i * 3 + 0], flat[i * 3 + 1], flat[i * 3 + 2] );
		}
		PtAntialiasDense( img.Base(), bw, bh );

		for ( int i = 0; i < stack.Count(); ++i )
		{
			const int src = stack[i];
			if ( comp[src] != id )
				continue;
			const int x = src % resX;
			const int y = src / resX;
			const int dx = ( x - minX ) + pad;
			const int dy = ( y - minY ) + pad;
			const int di = dx + dy * bw;
			Vector d = img[di];
			if ( !_finite( d.x ) || !_finite( d.y ) || !_finite( d.z ) )
				continue;
			if ( blend >= 0.999f )
				texels[src].m_Color = d;
			else
				texels[src].m_Color = texels[src].m_Color * ( 1.0f - blend ) + d * blend;
		}
	}
}

// Prop vertex atlas: 8x8 cell per triangle, 64x64 cells/page, 1-texel inset (6x6 content).
static const int kPtVertCell = 8;
static const int kPtVertAtlasGrid = 64;
static const int kPtVertPage = kPtVertCell * kPtVertAtlasGrid; // 512
static const int kPtVertInset = 1;
static const int kPtVertContent = kPtVertCell - 2 * kPtVertInset; // 6
static const int kPtVertCellsPerPage = kPtVertAtlasGrid * kPtVertAtlasGrid; // 4096
// Right-triangle lattice inside content: G = content-1 => (G+1)(G+2)/2 samples.
static const int kPtVertContentG = kPtVertContent - 1; // 5
static const int kPtVertSamplesPerTri = ( kPtVertContentG + 1 ) * ( kPtVertContentG + 2 ) / 2; // 21

struct PtVertAtlasCell_t
{
	int		prop;
	int		section;
	int		v0, v1, v2;
	bool	used;
};

static void PtVertAtlasHoleFill( Vector *rgb, const unsigned char *bakeMask, int pageW, int pageH )
{
	const int n = pageW * pageH;
	CUtlVector<Vector> tmp;
	CUtlVector<unsigned char> wgt;
	tmp.SetCount( n );
	wgt.SetCount( n );
	for ( int i = 0; i < n; ++i )
	{
		tmp[i] = rgb[i];
		wgt[i] = bakeMask[i] ? 1 : 0;
	}

	CUtlVector<Vector> next;
	CUtlVector<unsigned char> nextW;
	next.SetCount( n );
	nextW.SetCount( n );

	for ( int pass = 0; pass < 8; ++pass )
	{
		bool any = false;
		for ( int i = 0; i < n; ++i )
		{
			next[i] = tmp[i];
			nextW[i] = wgt[i];
		}
		for ( int y = 0; y < pageH; ++y )
		{
			for ( int x = 0; x < pageW; ++x )
			{
				const int idx = x + y * pageW;
				if ( wgt[idx] )
					continue;
				Vector sum( 0, 0, 0 );
				int sw = 0;
				for ( int dy = -1; dy <= 1; ++dy )
				{
					const int ny = y + dy;
					if ( ny < 0 || ny >= pageH )
						continue;
					for ( int dx = -1; dx <= 1; ++dx )
					{
						const int nx = x + dx;
						if ( nx < 0 || nx >= pageW )
							continue;
						const int nidx = nx + ny * pageW;
						if ( !wgt[nidx] )
							continue;
						sum += tmp[nidx];
						++sw;
					}
				}
				if ( sw > 0 )
				{
					next[idx] = sum * ( 1.0f / (float)sw );
					nextW[idx] = 1;
					any = true;
				}
			}
		}
		for ( int i = 0; i < n; ++i )
		{
			tmp[i] = next[i];
			wgt[i] = nextW[i];
		}
		if ( !any )
			break;
	}
	for ( int i = 0; i < n; ++i )
		rgb[i] = tmp[i];
}

static void PtVertAtlasDenoisePage( Vector *rgb, int pageW, int pageH )
{
	if ( !g_bPathTraceDenoise || !PathTraceDenoise_IsReady() )
		return;
	if ( pageW * pageH < 32 )
		return;

	const int n = pageW * pageH;
	const float blend = min( 1.0f, max( 0.0f, g_flPathTraceDenoiseStrength ) );
	CUtlVector<float> flat;
	flat.SetCount( n * 3 );
	for ( int i = 0; i < n; ++i )
	{
		flat[i * 3 + 0] = rgb[i].x;
		flat[i * 3 + 1] = rgb[i].y;
		flat[i * 3 + 2] = rgb[i].z;
	}
	if ( !PathTraceDenoise_DenoiseRGB( flat.Base(), pageW, pageH ) )
		return;
	for ( int i = 0; i < n; ++i )
	{
		Vector d( flat[i * 3 + 0], flat[i * 3 + 1], flat[i * 3 + 2] );
		if ( !_finite( d.x ) || !_finite( d.y ) || !_finite( d.z ) )
			continue;
		if ( blend >= 0.999f )
			rgb[i] = d;
		else
			rgb[i] = rgb[i] * ( 1.0f - blend ) + d * blend;
	}
}

static void PtVertAtlasSmoothCell( Vector *rgb, const unsigned char *bakeMask, int cellIndex )
{
	// Box-filter bake texels inside one cell only (no cross-triangle bleed).
	const int cellCol = cellIndex % kPtVertAtlasGrid;
	const int cellRow = cellIndex / kPtVertAtlasGrid;
	const int ox = cellCol * kPtVertCell;
	const int oy = cellRow * kPtVertCell;

	Vector staged[kPtVertCell * kPtVertCell];
	unsigned char has[kPtVertCell * kPtVertCell];
	memset( has, 0, sizeof( has ) );

	for ( int ly = 0; ly < kPtVertCell; ++ly )
	{
		for ( int lx = 0; lx < kPtVertCell; ++lx )
		{
			const int idx = ( ox + lx ) + ( oy + ly ) * kPtVertPage;
			if ( !bakeMask[idx] )
				continue;
			Vector sum( 0, 0, 0 );
			int n = 0;
			for ( int dy = -1; dy <= 1; ++dy )
			{
				const int yy = ly + dy;
				if ( yy < 0 || yy >= kPtVertCell )
					continue;
				for ( int dx = -1; dx <= 1; ++dx )
				{
					const int xx = lx + dx;
					if ( xx < 0 || xx >= kPtVertCell )
						continue;
					const int nidx = ( ox + xx ) + ( oy + yy ) * kPtVertPage;
					if ( !bakeMask[nidx] )
						continue;
					sum += rgb[nidx];
					++n;
				}
			}
			if ( n > 0 )
			{
				staged[ly * kPtVertCell + lx] = sum * ( 1.0f / (float)n );
				has[ly * kPtVertCell + lx] = 1;
			}
		}
	}
	for ( int ly = 0; ly < kPtVertCell; ++ly )
	{
		for ( int lx = 0; lx < kPtVertCell; ++lx )
		{
			if ( !has[ly * kPtVertCell + lx] )
				continue;
			rgb[( ox + lx ) + ( oy + ly ) * kPtVertPage] = staged[ly * kPtVertCell + lx];
		}
	}
}

static void PtVertAtlasGatherCell( const PtVertAtlasCell_t &cell, int cellIndex,
								   const Vector *rgb, const Vector *worldPos, const unsigned char *bakeMask,
								   CUtlVector<CUtlVector<CUtlVector<Vector>> *> &vertColorSum,
								   CUtlVector<CUtlVector<CUtlVector<float>> *> &vertWeightSum,
								   CUtlVector<CComputeStaticPropLightingResults *> &allResults )
{
	if ( !cell.used )
		return;
	if ( !vertColorSum[cell.prop] || cell.section >= vertColorSum[cell.prop]->Count() )
		return;
	if ( cell.section >= allResults[cell.prop]->m_ColorVertsArrays.Count() )
		return;

	CUtlVector<Vector> &sum = ( *vertColorSum[cell.prop] )[cell.section];
	CUtlVector<float> &w = ( *vertWeightSum[cell.prop] )[cell.section];
	const CUtlVector<colorVertex_t> &verts = *allResults[cell.prop]->m_ColorVertsArrays[cell.section];

	const int cellCol = cellIndex % kPtVertAtlasGrid;
	const int cellRow = cellIndex / kPtVertAtlasGrid;
	const int ox = cellCol * kPtVertCell;
	const int oy = cellRow * kPtVertCell;

	// Inverse-distance weights favor the closest sample but blend nearby chart
	// texels so single-sample fireflies don't dominate (old bary scatter averaged many).
	const float kEps = 1.0f; // hammer units^2 floor
	const int corners[3] = { cell.v0, cell.v1, cell.v2 };
	for ( int c = 0; c < 3; ++c )
	{
		const int vi = corners[c];
		if ( vi < 0 || vi >= sum.Count() || vi >= verts.Count() )
			continue;

		const Vector &vp = verts[vi].m_Position;
		Vector colSum( 0, 0, 0 );
		float wSum = 0.0f;
		float dMin = FLT_MAX;
		for ( int ly = 0; ly < kPtVertCell; ++ly )
		{
			for ( int lx = 0; lx < kPtVertCell; ++lx )
			{
				const int idx = ( ox + lx ) + ( oy + ly ) * kPtVertPage;
				if ( !bakeMask[idx] )
					continue;
				const float d2 = ( worldPos[idx] - vp ).LengthSqr();
				if ( d2 < dMin )
					dMin = d2;
			}
		}
		if ( dMin >= FLT_MAX * 0.5f )
			continue;

		// Ignore samples much farther than the closest (keeps "closest" intent).
		const float dCut = dMin * 9.0f + kEps; // ~3x distance
		for ( int ly = 0; ly < kPtVertCell; ++ly )
		{
			for ( int lx = 0; lx < kPtVertCell; ++lx )
			{
				const int idx = ( ox + lx ) + ( oy + ly ) * kPtVertPage;
				if ( !bakeMask[idx] )
					continue;
				const float d2 = ( worldPos[idx] - vp ).LengthSqr();
				if ( d2 > dCut )
					continue;
				const float bw = 1.0f / ( d2 + kEps );
				colSum += rgb[idx] * bw;
				wSum += bw;
			}
		}
		if ( wSum > 1e-8f )
		{
			sum[vi] += colSum * ( 1.0f / wSum );
			w[vi] += 1.0f;
		}
	}
}

bool CVradStaticPropMgr::ComputeLightingPathTraceGPU()
{
	if ( !PathTraceDXR_CanBakeProps() )
		return false;

	const int count = m_StaticProps.Count();
	if ( count <= 0 )
		return false;

	Msg( "[PathTrace-DXR] Static prop lighting via GPU pathtrace...\n" );
	fflush( stdout );

	CUtlVector<PtPropModelTemplate *> templates;
	templates.SetSize( m_StaticPropDict.Count() );
	for ( int i = 0; i < templates.Count(); ++i )
		templates[i] = nullptr;

	CUtlVector<CComputeStaticPropLightingResults *> allResults;
	allResults.SetSize( count );
	for ( int i = 0; i < count; ++i )
		allResults[i] = new CComputeStaticPropLightingResults;

	CUtlVector<PtGpuBakeLuxel> texelJobs;
	CUtlVector<PtGpuBakeLuxel> vertJobs;
	struct SampleMap_t { int prop; int section; int index; bool texel; };
	struct VertScatter_t { int prop; int section; int v0, v1, v2; float b0, b1, b2; };
	struct VertAtlasJobMap_t { int pixel; Vector worldPos; };
	CUtlVector<SampleMap_t> texelMaps;
	CUtlVector<VertScatter_t> vertScatters;
	CUtlVector<VertAtlasJobMap_t> atlasJobMaps;
	// Cap resident jobs - 17M verts * ~72B ~= 1.2GB and was OOMing into Map() AVs.
	const int kJobChunk = 262144;
	texelJobs.EnsureCapacity( kJobChunk );
	vertJobs.EnsureCapacity( kJobChunk );
	texelMaps.EnsureCapacity( kJobChunk );
	vertScatters.EnsureCapacity( kJobChunk );
	atlasJobMaps.EnsureCapacity( kJobChunk );

	// Vertex color accumulators: legacy bary scatter weights, or atlas closest-sample counts.
	CUtlVector<CUtlVector<CUtlVector<Vector>> *> vertColorSum;
	CUtlVector<CUtlVector<CUtlVector<float>> *> vertWeightSum;
	vertColorSum.SetSize( count );
	vertWeightSum.SetSize( count );
	for ( int i = 0; i < count; ++i )
	{
		vertColorSum[i] = nullptr;
		vertWeightSum[i] = nullptr;
	}

	matrix3x4_t identityPos, identityNrm;
	SetIdentityMatrix( identityPos );
	SetIdentityMatrix( identityNrm );

	int sessionKind = -1; // -1 none, 0 verts, 1 lightmap texels

	auto ensureVertAccum = [&]( int propIndex, int section, int nVerts )
	{
		CComputeStaticPropLightingResults *pRes = allResults[propIndex];
		while ( pRes->m_ColorVertsArrays.Count() <= section )
		{
			CUtlVector<colorVertex_t> *pArr = new CUtlVector<colorVertex_t>;
			pRes->m_ColorVertsArrays.AddToTail( pArr );
		}
		CUtlVector<colorVertex_t> &verts = *pRes->m_ColorVertsArrays[section];
		if ( verts.Count() < nVerts )
		{
			verts.SetCount( nVerts );
			memset( verts.Base(), 0, verts.Count() * sizeof( colorVertex_t ) );
		}
		if ( !vertColorSum[propIndex] )
		{
			vertColorSum[propIndex] = new CUtlVector<CUtlVector<Vector>>;
			vertWeightSum[propIndex] = new CUtlVector<CUtlVector<float>>;
		}
		CUtlVector<CUtlVector<Vector>> &sums = *vertColorSum[propIndex];
		CUtlVector<CUtlVector<float>> &ws = *vertWeightSum[propIndex];
		while ( sums.Count() <= section )
		{
			sums.AddToTail();
			ws.AddToTail();
		}
		if ( sums[section].Count() < nVerts )
		{
			sums[section].SetCount( nVerts );
			ws[section].SetCount( nVerts );
			memset( sums[section].Base(), 0, nVerts * sizeof( Vector ) );
			memset( ws[section].Base(), 0, nVerts * sizeof( float ) );
		}
	};

	auto writeBackTexels = [&]( const CUtlVector<SampleMap_t> &maps, const CUtlVector<PtGpuBakeResult> &results )
	{
		for ( int i = 0; i < maps.Count(); ++i )
		{
			const SampleMap_t &m = maps[i];
			const PtGpuBakeResult &r = results[i];
			Vector col( r.radiance[0], r.radiance[1], r.radiance[2] );
			if ( !_finite( col.x ) || !_finite( col.y ) || !_finite( col.z ) )
				col.Init();
			CComputeStaticPropLightingResults *pRes = allResults[m.prop];
			if ( m.section < pRes->m_TexelColors.Count() && pRes->m_TexelColors[m.section] )
				( *pRes->m_TexelColors[m.section] )[m.index] = col;
		}
	};

	auto writeBackVertScatters = [&]( const CUtlVector<VertScatter_t> &scatters, const CUtlVector<PtGpuBakeResult> &results )
	{
		for ( int i = 0; i < scatters.Count(); ++i )
		{
			const VertScatter_t &s = scatters[i];
			const PtGpuBakeResult &r = results[i];
			Vector col( r.radiance[0], r.radiance[1], r.radiance[2] );
			if ( !_finite( col.x ) || !_finite( col.y ) || !_finite( col.z ) )
				col.Init();
			if ( !vertColorSum[s.prop] || s.section >= vertColorSum[s.prop]->Count() )
				continue;
			CUtlVector<Vector> &sum = ( *vertColorSum[s.prop] )[s.section];
			CUtlVector<float> &w = ( *vertWeightSum[s.prop] )[s.section];
			auto add = [&]( int vi, float bw )
			{
				if ( vi < 0 || vi >= sum.Count() || bw <= 1e-8f )
					return;
				sum[vi] += col * bw;
				w[vi] += bw;
			};
			add( s.v0, s.b0 );
			add( s.v1, s.b1 );
			add( s.v2, s.b2 );
		}
	};

	// Active vertex atlas page (only used when vertGrid > 0).
	CUtlVector<Vector> pageRgb;
	CUtlVector<Vector> pageWorldPos;
	CUtlVector<unsigned char> pageBakeMask;
	CUtlVector<PtVertAtlasCell_t> pageCells;
	int pageCellCount = 0;
	bool bAtlasMode = false;

	auto resetAtlasPage = [&]()
	{
		const int nPix = kPtVertPage * kPtVertPage;
		pageRgb.SetCount( nPix );
		pageWorldPos.SetCount( nPix );
		pageBakeMask.SetCount( nPix );
		pageCells.SetCount( kPtVertCellsPerPage );
		memset( pageRgb.Base(), 0, nPix * sizeof( Vector ) );
		memset( pageWorldPos.Base(), 0, nPix * sizeof( Vector ) );
		memset( pageBakeMask.Base(), 0, nPix * sizeof( unsigned char ) );
		memset( pageCells.Base(), 0, kPtVertCellsPerPage * sizeof( PtVertAtlasCell_t ) );
		pageCellCount = 0;
	};

	auto writeBackAtlasJobs = [&]( const CUtlVector<VertAtlasJobMap_t> &maps, const CUtlVector<PtGpuBakeResult> &results )
	{
		for ( int i = 0; i < maps.Count(); ++i )
		{
			const VertAtlasJobMap_t &m = maps[i];
			const PtGpuBakeResult &r = results[i];
			if ( m.pixel < 0 || m.pixel >= pageRgb.Count() )
				continue;
			Vector col( r.radiance[0], r.radiance[1], r.radiance[2] );
			if ( !_finite( col.x ) || !_finite( col.y ) || !_finite( col.z ) )
				col.Init();
			pageRgb[m.pixel] = col;
			pageWorldPos[m.pixel] = m.worldPos;
			pageBakeMask[m.pixel] = 1;
		}
	};

	auto flushTexelJobs = [&]( bool bEndSession, unsigned &doneOut, unsigned progressTotal ) -> bool
	{
		const char *tag = "prop lightmaps";
		const int kind = 1;
		if ( texelJobs.Count() <= 0 )
		{
			if ( bEndSession && PathTraceDXR_GpuBakeIsActive() && sessionKind == kind )
			{
				PathTraceDXR_PropBakeClose( tag, progressTotal > 0 ? progressTotal : doneOut );
				sessionKind = -1;
			}
			return true;
		}
		if ( PathTraceDXR_GpuBakeIsActive() && sessionKind != kind )
		{
			PathTraceDXR_PropBakeSuspend( sessionKind == 1 ? "prop lightmaps" : "prop verts" );
			sessionKind = -1;
		}
		CUtlVector<PtGpuBakeResult> results;
		results.SetCount( texelJobs.Count() );
		const unsigned base = doneOut;
		const bool baked = PathTraceDXR_BakePropSamples( texelJobs.Base(), (unsigned)texelJobs.Count(),
														 results.Base(), true, bEndSession,
														 base, progressTotal );
		if ( baked )
		{
			writeBackTexels( texelMaps, results );
			doneOut += (unsigned)texelJobs.Count();
		}
		if ( !baked || bEndSession )
			sessionKind = -1;
		else
			sessionKind = kind;
		texelJobs.RemoveAll();
		texelMaps.RemoveAll();
		return baked;
	};

	auto flushVertJobs = [&]( bool bEndSession, unsigned &doneOut, unsigned progressTotal ) -> bool
	{
		const char *tag = "prop verts";
		const int kind = 0;
		if ( vertJobs.Count() <= 0 )
		{
			if ( bEndSession && PathTraceDXR_GpuBakeIsActive() && sessionKind == kind )
			{
				PathTraceDXR_PropBakeClose( tag, progressTotal > 0 ? progressTotal : doneOut );
				sessionKind = -1;
			}
			return true;
		}
		if ( PathTraceDXR_GpuBakeIsActive() && sessionKind != kind )
		{
			PathTraceDXR_PropBakeSuspend( sessionKind == 1 ? "prop lightmaps" : "prop verts" );
			sessionKind = -1;
		}
		CUtlVector<PtGpuBakeResult> results;
		results.SetCount( vertJobs.Count() );
		const unsigned base = doneOut;
		const bool baked = PathTraceDXR_BakePropSamples( vertJobs.Base(), (unsigned)vertJobs.Count(),
														 results.Base(), false, bEndSession,
														 base, progressTotal );
		if ( baked )
		{
			if ( bAtlasMode )
				writeBackAtlasJobs( atlasJobMaps, results );
			else
				writeBackVertScatters( vertScatters, results );
			doneOut += (unsigned)vertJobs.Count();
		}
		if ( !baked || bEndSession )
			sessionKind = -1;
		else
			sessionKind = kind;
		vertJobs.RemoveAll();
		vertScatters.RemoveAll();
		atlasJobMaps.RemoveAll();
		return baked;
	};

	auto finalizeAtlasPage = [&]( unsigned &doneOut, unsigned progressTotal ) -> bool
	{
		if ( pageCellCount <= 0 )
			return true;
		if ( !flushVertJobs( false, doneOut, progressTotal ) )
			return false;

		PtVertAtlasHoleFill( pageRgb.Base(), pageBakeMask.Base(), kPtVertPage, kPtVertPage );
		PtVertAtlasDenoisePage( pageRgb.Base(), kPtVertPage, kPtVertPage );

		// Per-cell blur after denoise: kills residual fireflies and limits atlas OIDN bleed.
		for ( int pass = 0; pass < 2; ++pass )
		{
			for ( int ci = 0; ci < pageCellCount; ++ci )
			{
				if ( pageCells[ci].used )
					PtVertAtlasSmoothCell( pageRgb.Base(), pageBakeMask.Base(), ci );
			}
		}

		for ( int ci = 0; ci < pageCellCount; ++ci )
		{
			PtVertAtlasGatherCell( pageCells[ci], ci, pageRgb.Base(), pageWorldPos.Base(),
								   pageBakeMask.Base(), vertColorSum, vertWeightSum, allResults );
		}
		resetAtlasPage();
		return true;
	};

	// Rough totals for one progress line (solid-culled samples may be slightly lower).
	unsigned estTexels = 0;
	unsigned estVerts = 0;
	for ( int propIndex = 0; propIndex < count; ++propIndex )
	{
		CStaticProp &prop = m_StaticProps[propIndex];
		StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
		studiohdr_t *pStudioHdr = dict.m_pStudioHdr;
		if ( !pStudioHdr )
			continue;
		const bool withVertexLighting = ( prop.m_Flags & STATIC_PROP_NO_PER_VERTEX_LIGHTING ) == 0;
		const bool withTexelLighting = ( prop.m_Flags & STATIC_PROP_NO_PER_TEXEL_LIGHTING ) == 0;
		if ( withVertexLighting )
		{
			for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts; ++bodyID )
			{
				mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );
				for ( int modelID = 0; modelID < pBodyPart->nummodels; ++modelID )
				{
					unsigned nVerts = (unsigned)pBodyPart->pModel( modelID )->numvertices;
					const PropUvSplitPlan *pPlan = PropUvPlan( prop.m_UvPlan );
					if ( pPlan && prop.m_UvBin >= 0 && pPlan->m_NumBins > 1 )
						nVerts = nVerts / (unsigned)pPlan->m_NumBins;
					estVerts += nVerts;
				}
			}
		}
	}
	if ( g_nPathTracePropVertGrid > 0 )
	{
		// Rough: ~2 tris/vert x atlas samples/tri.
		estVerts = estVerts * 2u * (unsigned)kPtVertSamplesPerTri / 3u;
	}

	bool ok = true;
	int totalTexelJobs = 0;
	int totalVertJobs = 0;
	unsigned doneTexels = 0;
	unsigned doneVerts = 0;

	auto ensureTemplate = [&]( int modelIdx, studiohdr_t *pStudioHdr,
							   OptimizedModel::FileHeader_t *pVtxHdr )
	{
		if ( templates[modelIdx] )
			return;
		if ( !pStudioHdr || !pVtxHdr )
			return;
		PtPropModelTemplate *tmpl = new PtPropModelTemplate;
		templates[modelIdx] = tmpl;
		for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts; ++bodyID )
		{
			mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );
			for ( int modelID = 0; modelID < pBodyPart->nummodels; ++modelID )
			{
				mstudiomodel_t *pStudioModel = pBodyPart->pModel( modelID );

				int sec = tmpl->vertSections.AddToTail();
				tmpl->texelBySection.AddToTail();
				PtPropVertTemplate &vt = tmpl->vertSections[sec];
				vt.localPos.SetCount( pStudioModel->numvertices );
				vt.localNrm.SetCount( pStudioModel->numvertices );
				int vertCursor = 0;
				for ( int meshID = 0; meshID < pStudioModel->nummeshes; ++meshID )
				{
					mstudiomesh_t *pStudioMesh = pStudioModel->pMesh( meshID );
					const mstudio_meshvertexdata_t *vertData = pStudioMesh->GetVertexData( (void *)pStudioHdr );
					if ( !vertData )
					{
						vertCursor += pStudioMesh->numvertices;
						continue;
					}
					for ( int vertexID = 0; vertexID < pStudioMesh->numvertices; ++vertexID )
					{
						if ( vertCursor >= vt.localPos.Count() )
							break;
						vt.localPos[vertCursor] = *vertData->Position( vertexID );
						vt.localNrm[vertCursor] = *vertData->Normal( vertexID );
						++vertCursor;
					}
				}
			}
		}
	};

	auto getOrBuildTexels = [&]( PtPropModelTemplate *tmpl, int section,
								 studiohdr_t *pStudioHdr, mstudiomodel_t *pStudioModel,
								 OptimizedModel::ModelHeader_t *pVtxModel,
								 int lightmapW, int lightmapH, int materialFilter,
								 const PropUvSplitPlan *pUvPlan, int uvBin, int uvBody, int uvModel ) -> PtPropTexelTemplate *
	{
		if ( !tmpl || section < 0 || section >= tmpl->texelBySection.Count() )
			return NULL;
		if ( lightmapW <= 0 || lightmapH <= 0 )
			return NULL;

		CUtlVector<PtPropTexelTemplate> &vars = tmpl->texelBySection[section];
		for ( int i = 0; i < vars.Count(); ++i )
		{
			if ( vars[i].resX == lightmapW && vars[i].resY == lightmapH && vars[i].material == materialFilter && vars[i].uvBin == uvBin )
				return &vars[i];
		}

		int slot = vars.AddToTail();
		PtPropTexelTemplate &tt = vars[slot];
		tt.material = materialFilter;
		tt.uvBin = uvBin;
		CComputeStaticPropLightingResults tmpRes;
		CUtlVector<colorTexel_t> *pArr = new CUtlVector<colorTexel_t>;
		tmpRes.m_ColorTexelsArrays.AddToTail( pArr );
		bool resetTexels = true;
		for ( int meshID = 0; meshID < pStudioModel->nummeshes; ++meshID )
		{
			if ( materialFilter >= 0 )
			{
				mstudiomesh_t *pMesh = pStudioModel->pMesh( meshID );
				if ( !pMesh || pMesh->material != materialFilter )
					continue;
			}
			GenerateLightmapSamplesForMesh( identityPos, identityNrm, 0, -1, 0,
				lightmapW, lightmapH,
				pStudioHdr, pStudioModel, pVtxModel, meshID, &tmpRes, false, NULL, NULL, resetTexels,
				pUvPlan, uvBin, uvBody, uvModel );
			resetTexels = false;
		}
		CUtlVector<colorTexel_t> &texels = *tmpRes.m_ColorTexelsArrays[0];
		tt.resX = lightmapW;
		tt.resY = lightmapH;
		tt.localPos.SetCount( texels.Count() );
		tt.localNrm.SetCount( texels.Count() );
		tt.localAxisU.SetCount( texels.Count() );
		tt.localAxisV.SetCount( texels.Count() );
		tt.luxelWorld.SetCount( texels.Count() );
		tt.valid.SetCount( texels.Count() );
		tt.interesting.SetCount( texels.Count() );
		tt.bake.SetCount( texels.Count() );
		for ( int i = 0; i < texels.Count(); ++i )
		{
			tt.localPos[i] = texels[i].m_WorldPosition;
			tt.localNrm[i] = texels[i].m_WorldNormal;
			tt.localAxisU[i] = texels[i].m_AxisU;
			tt.localAxisV[i] = texels[i].m_AxisV;
			tt.luxelWorld[i] = texels[i].m_LuxelWorld;
			tt.valid[i] = texels[i].m_bValid;
			tt.interesting[i] = texels[i].m_bPossiblyInteresting;
			tt.bake[i] = texels[i].m_bValid ? 1 : 0;
		}
		const int resX = tt.resX;
		const int resY = tt.resY;
		for ( int y = 0; y < resY; ++y )
		{
			for ( int x = 0; x < resX; ++x )
			{
				const int i = y * resX + x;
				if ( !tt.interesting[i] || tt.valid[i] || tt.bake[i] )
					continue;
				if ( tt.localNrm[i].LengthSqr() < 1e-6f )
					continue;
				bool beside = false;
				for ( int dy = -1; dy <= 1 && !beside; ++dy )
				{
					for ( int dx = -1; dx <= 1 && !beside; ++dx )
					{
						if ( !dx && !dy )
							continue;
						const int nx = x + dx, ny = y + dy;
						if ( nx < 0 || ny < 0 || nx >= resX || ny >= resY )
							continue;
						const int j = ny * resX + nx;
						if ( !tt.valid[j] )
							continue;
						colorTexel_t a, b;
						a.m_WorldPosition = tt.localPos[i];
						a.m_WorldNormal = tt.localNrm[i];
						a.m_LuxelWorld = tt.luxelWorld[i];
						b.m_WorldPosition = tt.localPos[j];
						b.m_WorldNormal = tt.localNrm[j];
						b.m_LuxelWorld = tt.luxelWorld[j];
						if ( PropTexelChartsMatch( a, b ) )
							beside = true;
					}
				}
				if ( !beside )
					tt.bake[i] = 1;
			}
		}
		return &tt;
	};

	// Count texels that will actually be traced (not the full lightmap square).
	// UV-split pieces only cover their own islands, so width*height is several times too high.
	for ( int propIndex = 0; propIndex < count; ++propIndex )
	{
		CStaticProp &prop = m_StaticProps[propIndex];
		if ( prop.m_Flags & STATIC_PROP_NO_PER_TEXEL_LIGHTING )
			continue;
		if ( prop.m_LightmapImageWidth == 0 || prop.m_LightmapImageHeight == 0 )
			continue;
		StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
		studiohdr_t *pStudioHdr = dict.m_pStudioHdr;
		OptimizedModel::FileHeader_t *pVtxHdr = (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base();
		if ( !pStudioHdr || !pVtxHdr )
			continue;
		ensureTemplate( prop.m_ModelIdx, pStudioHdr, pVtxHdr );
		PtPropModelTemplate *tmpl = templates[prop.m_ModelIdx];
		if ( !tmpl )
			continue;
		const PropUvSplitPlan *pUvPlan = PropUvPlan( prop.m_UvPlan );
		const int uvBin = ( pUvPlan && prop.m_UvBin >= 0 ) ? prop.m_UvBin : -1;
		int section = 0;
		for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts; ++bodyID )
		{
			OptimizedModel::BodyPartHeader_t *pVtxBodyPart = pVtxHdr->pBodyPart( bodyID );
			mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );
			for ( int modelID = 0; modelID < pBodyPart->nummodels; ++modelID, ++section )
			{
				if ( section >= tmpl->vertSections.Count() )
					break;
				PtPropTexelTemplate *tt = getOrBuildTexels( tmpl, section, pStudioHdr, pBodyPart->pModel( modelID ),
					pVtxBodyPart->pModel( modelID ),
					(int)prop.m_LightmapImageWidth, (int)prop.m_LightmapImageHeight, prop.m_OnlyMaterial,
					( uvBin >= 0 ) ? pUvPlan : NULL, uvBin, bodyID, modelID );
				if ( !tt )
					continue;
				for ( int i = 0; i < tt->bake.Count(); ++i )
				{
					if ( tt->bake[i] )
						++estTexels;
				}
			}
		}
	}

	Msg( "[PathTrace-DXR] Prop lighting: ~%u texel samples, ~%u vert samples (chunks of %d)...\n",
		 estTexels, estVerts, kJobChunk );
	fflush( stdout );

	// ----- Phase 1: lightmap texels only (never start vert GPU session) -----
	for ( int propIndex = 0; propIndex < count && ok; ++propIndex )
	{
		CStaticProp &prop = m_StaticProps[propIndex];
		StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
		studiohdr_t *pStudioHdr = dict.m_pStudioHdr;
		OptimizedModel::FileHeader_t *pVtxHdr = (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base();
		if ( !pStudioHdr || !pVtxHdr )
			continue;
		if ( !BakeVolume_ShouldBakePropOriented( prop.m_Origin, prop.m_Angles, dict.m_Mins, dict.m_Maxs ) )
			continue;

		const bool withVertexLighting = ( prop.m_Flags & STATIC_PROP_NO_PER_VERTEX_LIGHTING ) == 0;
		const bool withTexelLighting = ( prop.m_Flags & STATIC_PROP_NO_PER_TEXEL_LIGHTING ) == 0;
		if ( !withVertexLighting && !withTexelLighting )
			continue;

		const int skip_prop = PropSelfShadowSkip( prop, propIndex );
		ensureTemplate( prop.m_ModelIdx, pStudioHdr, pVtxHdr );

		PtPropModelTemplate *tmpl = templates[prop.m_ModelIdx];
		CComputeStaticPropLightingResults *pResults = allResults[propIndex];
		matrix3x4_t matPos, matNormal;
		AngleMatrix( prop.m_Angles, prop.m_Origin, matPos );
		AngleMatrix( prop.m_Angles, matNormal );

		int section = 0;
		for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts && ok; ++bodyID )
		{
			OptimizedModel::BodyPartHeader_t *pVtxBodyPart = pVtxHdr->pBodyPart( bodyID );
			mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );
			for ( int modelID = 0; modelID < pBodyPart->nummodels && ok; ++modelID, ++section )
			{
				if ( section >= tmpl->vertSections.Count() )
					break;

				if ( withTexelLighting )
				{
					OptimizedModel::ModelHeader_t *pVtxModel = pVtxBodyPart->pModel( modelID );
					mstudiomodel_t *pStudioModel = pBodyPart->pModel( modelID );
					const PropUvSplitPlan *pUvPlan = PropUvPlan( prop.m_UvPlan );
					const int uvBin = ( pUvPlan && prop.m_UvBin >= 0 ) ? prop.m_UvBin : -1;
					PtPropTexelTemplate *tt = getOrBuildTexels( tmpl, section, pStudioHdr, pStudioModel, pVtxModel,
						(int)prop.m_LightmapImageWidth, (int)prop.m_LightmapImageHeight, prop.m_OnlyMaterial,
						( uvBin >= 0 ) ? pUvPlan : NULL, uvBin, bodyID, modelID );
					if ( !tt )
						continue;

					const int nTex = tt->localPos.Count();
					CUtlVector<Vector> *pColors = new CUtlVector<Vector>;
					CUtlVector<unsigned char> *pFlags = new CUtlVector<unsigned char>;
					pColors->SetCount( nTex );
					pFlags->SetCount( nTex );
					memset( pColors->Base(), 0, nTex * sizeof( Vector ) );
					memset( pFlags->Base(), 0, nTex * sizeof( unsigned char ) );
					pResults->m_TexelColors.AddToTail( pColors );
					pResults->m_TexelFlags.AddToTail( pFlags );
					for ( int i = 0; i < nTex; ++i )
					{
						Vector worldPos, worldNrm;
						VectorTransform( tt->localPos[i], matPos, worldPos );
						VectorRotate( tt->localNrm[i], matNormal, worldNrm );
						worldNrm.NormalizeInPlace();
						unsigned char flags = 0;
						if ( tt->valid[i] )
							flags |= 1;
						if ( tt->interesting[i] )
							flags |= 2;
						float luxelWorld = 0.0f;
						Vector axisU( 0, 0, 0 ), axisV( 0, 0, 0 );
						if ( i < tt->luxelWorld.Count() && tt->luxelWorld[i] > 0.05f )
						{
							luxelWorld = tt->luxelWorld[i];
							VectorRotate( tt->localAxisU[i], matNormal, axisU );
							VectorRotate( tt->localAxisV[i], matNormal, axisV );
						}
						( *pFlags )[i] = flags;
						// Interiors, plus a gutter only when this triangle owns no neighboring
						// texel. The 1px ring around a covered chart is dilated afterwards.
						if ( i < tt->bake.Count() && tt->bake[i] )
						{
							if ( worldNrm.LengthSqr() < 1e-6f )
								continue;
							float luxel = ( luxelWorld > 0.05f ) ? luxelWorld : 2.0f;
							// Same crease bias as world luxels, but ignore this prop so its own
							// tessellation is not treated as a side wall.
							const int meshProp = ( prop.m_ShadowSkipProp >= 0 ) ? prop.m_ShadowSkipProp : propIndex;
							const bool bFullSelfSkip = skip_prop >= 0;
							Vector samplePos;
							PathTraceDXR_LuxelOriginInFront( worldPos, worldNrm, luxel, samplePos,
								TRACE_ID_STATICPROP | meshProp );
							if ( PositionInSolid( samplePos ) )
								samplePos = worldPos;
							if ( PositionInSolid( samplePos ) )
							{
								Vector nrm = worldNrm;
								VectorNormalize( nrm );
								bool found = false;
								for ( int step = 1; step <= 6 && !found; ++step )
								{
									Vector p = worldPos + nrm * ( luxel * (float)step );
									if ( !PositionInSolid( p ) )
									{
										samplePos = p;
										found = true;
									}
								}
								if ( !found )
								{
									// Same-chart dilate can still fill this from a lit neighbor.
									// Leaving it valid and black is what paints a whole triangle black.
									( *pFlags )[i] = 2;
									continue;
								}
							}

							if ( !BakeVolume_ShouldBakePropPoint( prop.m_Origin, PropBakeName( prop ), worldPos ) )
							{
								Vector cached;
								if ( BakeVolume_CachedPropTexelLinear( prop.m_Origin, PropBakeName( prop ), 0, i,
										(int)prop.m_LightmapImageWidth, (int)prop.m_LightmapImageHeight, cached ) )
									( *pColors )[i] = cached;
								continue;
							}
							SampleMap_t m = { propIndex, section, i, true };
							texelMaps.AddToTail( m );
							PtGpuBakeLuxel job;
							PtPackBakeLuxelLightmap( job, samplePos, worldNrm, axisU, axisV, luxel, meshProp, propIndex, bFullSelfSkip );
							texelJobs.AddToTail( job );
							++totalTexelJobs;
							if ( texelJobs.Count() >= kJobChunk )
								ok = flushTexelJobs( false, doneTexels, estTexels );
						}
					}
				}
				// Vert color arrays are allocated in phase 2 - keeps ~18M verts off the heap
				// during lightmap GpuBakeBegin (was causing intermittent Map/OOM AVs).
			}
		}
	}

	if ( totalTexelJobs > 0 )
		estTexels = (unsigned)totalTexelJobs;
	if ( ok )
		ok = flushTexelJobs( true, doneTexels, estTexels );

	const int vertGrid = g_nPathTracePropVertGrid;
	bAtlasMode = ( vertGrid > 0 );
	if ( bAtlasMode )
	{
		Msg( "[PathTrace-DXR] Prop vertex lighting: %dx%d tri charts in %dx%d atlas (inset %d, ~%d samples/tri, near-vert IDW gather%s).\n",
			 kPtVertCell, kPtVertCell, kPtVertAtlasGrid, kPtVertAtlasGrid, kPtVertInset, kPtVertSamplesPerTri,
			 ( g_bPathTraceDenoise && PathTraceDenoise_IsReady() ) ? ", denoise on" : "" );
		fflush( stdout );
		resetAtlasPage();
	}

	// ----- Phase 2: allocate + bake vertex samples (chunked) -----
	for ( int propIndex = 0; propIndex < count && ok; ++propIndex )
	{
		CStaticProp &prop = m_StaticProps[propIndex];
		StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
		studiohdr_t *pStudioHdr = dict.m_pStudioHdr;
		OptimizedModel::FileHeader_t *pVtxHdr = (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base();
		if ( !pStudioHdr || !pVtxHdr )
			continue;
		if ( !BakeVolume_ShouldBakePropOriented( prop.m_Origin, prop.m_Angles, dict.m_Mins, dict.m_Maxs ) )
			continue;
		const bool withVertexLighting = ( prop.m_Flags & STATIC_PROP_NO_PER_VERTEX_LIGHTING ) == 0;
		const bool withTexelLighting = ( prop.m_Flags & STATIC_PROP_NO_PER_TEXEL_LIGHTING ) == 0;
		if ( !withVertexLighting )
			continue;

		ensureTemplate( prop.m_ModelIdx, pStudioHdr, pVtxHdr );
		if ( !templates[prop.m_ModelIdx] )
			continue;

		const int skip_prop = PropSelfShadowSkip( prop, propIndex );
		PtPropModelTemplate *tmpl = templates[prop.m_ModelIdx];
		CComputeStaticPropLightingResults *pResults = allResults[propIndex];
		matrix3x4_t matPos, matNormal;
		AngleMatrix( prop.m_Angles, prop.m_Origin, matPos );
		AngleMatrix( prop.m_Angles, matNormal );

		int section = 0;
		for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts && ok; ++bodyID )
		{
			OptimizedModel::BodyPartHeader_t *pVtxBodyPart = pVtxHdr->pBodyPart( bodyID );
			mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );
			for ( int modelID = 0; modelID < pBodyPart->nummodels && ok; ++modelID, ++section )
			{
				if ( section >= tmpl->vertSections.Count() )
					break;

				OptimizedModel::ModelHeader_t *pVtxModel = pVtxBodyPart->pModel( modelID );
				mstudiomodel_t *pStudioModel = pBodyPart->pModel( modelID );
				PtPropVertTemplate &vt = tmpl->vertSections[section];
				ensureVertAccum( propIndex, section, vt.localPos.Count() );

				CUtlVector<colorVertex_t> &colorVerts = *pResults->m_ColorVertsArrays[section];
				for ( int i = 0; i < vt.localPos.Count(); ++i )
				{
					colorVertex_t &cv = colorVerts[i];
					VectorTransform( vt.localPos[i], matPos, cv.m_Position );
					cv.m_bValid = !PositionInSolid( cv.m_Position );
					cv.m_Color.Init();
				}

				if ( !bAtlasMode )
				{
					// Legacy: one PathLi sample per unique vertex.
					for ( int i = 0; i < vt.localPos.Count() && ok; ++i )
					{
						colorVertex_t &cv = colorVerts[i];
						if ( !cv.m_bValid )
							continue;
						if ( !BakeVolume_ShouldBakePropPoint( prop.m_Origin, PropBakeName( prop ), cv.m_Position ) )
							continue;
						Vector nrm;
						VectorRotate( vt.localNrm[i], matNormal, nrm );
						nrm.NormalizeInPlace();
						VertScatter_t sc = { propIndex, section, i, i, i, 1.0f, 0.0f, 0.0f };
						vertScatters.AddToTail( sc );
						PtGpuBakeLuxel job;
						PtPackBakeLuxel( job, cv.m_Position, nrm, skip_prop, propIndex );
						vertJobs.AddToTail( job );
						++totalVertJobs;
						if ( vertJobs.Count() >= kJobChunk )
							ok = flushVertJobs( false, doneVerts, estVerts );
					}
					continue;
				}

				// 8x8 charts packed into atlas pages; closest denoised sample -> verts.
				OptimizedModel::ModelLODHeader_t *pVtxLOD = pVtxModel->pLOD( 0 );
				const int nMeshes = min( pStudioModel->nummeshes, pVtxLOD->numMeshes );
				int meshVertBase = 0;
				for ( int meshID = 0; meshID < nMeshes && ok; ++meshID )
				{
					mstudiomesh_t *pStudioMesh = pStudioModel->pMesh( meshID );
					OptimizedModel::MeshHeader_t *pVtxMesh = pVtxLOD->pMesh( meshID );
					if ( prop.m_OnlyMaterial >= 0 && pStudioMesh->material != prop.m_OnlyMaterial )
					{
						meshVertBase += pStudioMesh->numvertices;
						continue;
					}
					const mstudio_meshvertexdata_t *vertData = pStudioMesh->GetVertexData( (void *)pStudioHdr );
					if ( !vertData )
					{
						meshVertBase += pStudioMesh->numvertices;
						continue;
					}

					for ( int nGroup = 0; nGroup < pVtxMesh->numStripGroups && ok; ++nGroup )
					{
						OptimizedModel::StripGroupHeader_t *pStripGroup = pVtxMesh->pStripGroup( nGroup );
						for ( int nStrip = 0; nStrip < pStripGroup->numStrips && ok; ++nStrip )
						{
							OptimizedModel::StripHeader_t *pStrip = pStripGroup->pStrip( nStrip );
							if ( !( pStrip->flags & OptimizedModel::STRIP_IS_TRILIST ) )
								continue;

							const PropUvSplitPlan *pUvPlan = PropUvPlan( prop.m_UvPlan );
							const int uvBin = ( pUvPlan && prop.m_UvBin >= 0 ) ? prop.m_UvBin : -1;
							for ( int ii = 0; ii < pStrip->numIndices && ok; ii += 3 )
							{
								const int idx = pStrip->indexOffset + ii;
								if ( pUvPlan && uvBin >= 0 &&
									 !PropUvTriKept( *pUvPlan, uvBin, bodyID, modelID, 0, meshID, nGroup, idx ) )
									continue;
								const int mv0 = pStripGroup->pVertex( *pStripGroup->pIndex( idx ) )->origMeshVertID;
								const int mv1 = pStripGroup->pVertex( *pStripGroup->pIndex( idx + 1 ) )->origMeshVertID;
								const int mv2 = pStripGroup->pVertex( *pStripGroup->pIndex( idx + 2 ) )->origMeshVertID;
								const int v0 = meshVertBase + mv0;
								const int v1 = meshVertBase + mv1;
								const int v2 = meshVertBase + mv2;
								if ( v0 < 0 || v1 < 0 || v2 < 0 ||
									 v0 >= vt.localPos.Count() || v1 >= vt.localPos.Count() || v2 >= vt.localPos.Count() )
									continue;

								Vector lp0 = vt.localPos[v0], lp1 = vt.localPos[v1], lp2 = vt.localPos[v2];
								Vector ln0 = vt.localNrm[v0], ln1 = vt.localNrm[v1], ln2 = vt.localNrm[v2];
								Vector wp0, wp1, wp2, wn0, wn1, wn2;
								VectorTransform( lp0, matPos, wp0 );
								VectorTransform( lp1, matPos, wp1 );
								VectorTransform( lp2, matPos, wp2 );
								VectorRotate( ln0, matNormal, wn0 );
								VectorRotate( ln1, matNormal, wn1 );
								VectorRotate( ln2, matNormal, wn2 );

								if ( pageCellCount >= kPtVertCellsPerPage )
								{
									ok = finalizeAtlasPage( doneVerts, estVerts );
									if ( !ok )
										break;
								}

								const int cellIndex = pageCellCount++;
								PtVertAtlasCell_t &cell = pageCells[cellIndex];
								cell.prop = propIndex;
								cell.section = section;
								cell.v0 = v0;
								cell.v1 = v1;
								cell.v2 = v2;
								cell.used = true;

								const int cellCol = cellIndex % kPtVertAtlasGrid;
								const int cellRow = cellIndex / kPtVertAtlasGrid;
								const int ox = cellCol * kPtVertCell;
								const int oy = cellRow * kPtVertCell;
								const float invG = 1.0f / (float)kPtVertContentG;

								for ( int bi = 0; bi <= kPtVertContentG && ok; ++bi )
								{
									for ( int bj = 0; bj <= kPtVertContentG - bi && ok; ++bj )
									{
										const float b0 = (float)bi * invG;
										const float b1 = (float)bj * invG;
										const float b2 = 1.0f - b0 - b1;
										Vector pos = wp0 * b0 + wp1 * b1 + wp2 * b2;
										Vector nrm = wn0 * b0 + wn1 * b1 + wn2 * b2;
										if ( nrm.LengthSqr() < 1e-10f )
											continue;
										nrm.NormalizeInPlace();
										pos += nrm * 0.25f;
										if ( PositionInSolid( pos ) )
											continue;
										if ( !BakeVolume_ShouldBakePropPoint( prop.m_Origin, PropBakeName( prop ), pos ) )
											continue;

										// Place lattice into centered inset: content (bi,bj) -> cell (inset+bi, inset+bj).
										const int lx = kPtVertInset + bi;
										const int ly = kPtVertInset + bj;
										const int pixel = ( ox + lx ) + ( oy + ly ) * kPtVertPage;

										VertAtlasJobMap_t jm;
										jm.pixel = pixel;
										jm.worldPos = pos;
										atlasJobMaps.AddToTail( jm );
										PtGpuBakeLuxel job;
										PtPackBakeLuxel( job, pos, nrm, skip_prop, propIndex );
										vertJobs.AddToTail( job );
										++totalVertJobs;
										if ( vertJobs.Count() >= kJobChunk )
											ok = flushVertJobs( false, doneVerts, estVerts );
									}
								}
							}
						}
					}
					meshVertBase += pStudioMesh->numvertices;
				}
			}
		}
	}

	if ( ok && bAtlasMode && pageCellCount > 0 )
		ok = finalizeAtlasPage( doneVerts, estVerts );

	if ( totalVertJobs > 0 )
		estVerts = (unsigned)totalVertJobs;
	if ( ok )
		ok = flushVertJobs( true, doneVerts, estVerts );
	if ( PathTraceDXR_GpuBakeIsActive() )
	{
		PathTraceDXR_PropBakeClose( sessionKind == 1 ? "prop lightmaps" : "prop verts",
									sessionKind == 1 ? doneTexels : doneVerts );
		sessionKind = -1;
	}

	// Normalize vertex colors; unlit verts inherit nearest lit
	// (covers verts only referenced by lower LODs / missed by the LOD0 sample grid).
	if ( ok )
	{
		for ( int pi = 0; pi < count; ++pi )
		{
			if ( !vertColorSum[pi] )
				continue;
			CComputeStaticPropLightingResults *pRes = allResults[pi];
			CUtlVector<CUtlVector<Vector>> &sums = *vertColorSum[pi];
			CUtlVector<CUtlVector<float>> &ws = *vertWeightSum[pi];
			PtPropModelTemplate *tmpl = templates[m_StaticProps[pi].m_ModelIdx];
			for ( int s = 0; s < sums.Count() && s < pRes->m_ColorVertsArrays.Count(); ++s )
			{
				CUtlVector<colorVertex_t> &verts = *pRes->m_ColorVertsArrays[s];
				for ( int vi = 0; vi < sums[s].Count() && vi < verts.Count(); ++vi )
				{
					if ( ws[s][vi] > 1e-8f )
						verts[vi].m_Color = sums[s][vi] * ( 1.0f / ws[s][vi] );
				}
				if ( tmpl && s < tmpl->vertSections.Count() )
					PtInheritUnlitPropVertColors( verts, ws[s], tmpl->vertSections[s].localPos );
			}
		}
	}

	if ( ok )
		Msg( "[PathTrace-DXR] Prop lighting baked: %d lightmap texels, %d vertex samples.\n",
			 totalTexelJobs, totalVertJobs );

	if ( ok && totalTexelJobs == 0 && totalVertJobs == 0 )
		ok = true;

	if ( ok )
	{
		Msg( "Applying static prop lighting...\n" );
		fflush( stdout );
		for ( int pi = 0; pi < count; ++pi )
		{
			CComputeStaticPropLightingResults *pRes = allResults[pi];
			if ( !pRes )
				continue;
			CStaticProp &prop = m_StaticProps[pi];
			// Rebuild one prop's fat grids from the shared template, filter, encode, free.
			// Holding every prop's positions and normals at once was paging the machine.
			if ( pRes->m_TexelColors.Count() > 0 )
			{
				StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
				studiohdr_t *pStudioHdr = dict.m_pStudioHdr;
				OptimizedModel::FileHeader_t *pVtxHdr = (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base();
				PtPropModelTemplate *tmpl = templates[prop.m_ModelIdx];
				matrix3x4_t matPos, matNormal;
				AngleMatrix( prop.m_Angles, prop.m_Origin, matPos );
				AngleMatrix( prop.m_Angles, matNormal );
				const PropUvSplitPlan *pUvPlan = PropUvPlan( prop.m_UvPlan );
				const int uvBin = ( pUvPlan && prop.m_UvBin >= 0 ) ? prop.m_UvBin : -1;
				int section = 0;
				int colorSlot = 0;
				if ( pStudioHdr && pVtxHdr && tmpl )
				{
					for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts; ++bodyID )
					{
						OptimizedModel::BodyPartHeader_t *pVtxBodyPart = pVtxHdr->pBodyPart( bodyID );
						mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );
						for ( int modelID = 0; modelID < pBodyPart->nummodels; ++modelID, ++section )
						{
							if ( section >= tmpl->vertSections.Count() )
								break;
							PtPropTexelTemplate *tt = getOrBuildTexels( tmpl, section, pStudioHdr, pBodyPart->pModel( modelID ),
								pVtxBodyPart->pModel( modelID ),
								(int)prop.m_LightmapImageWidth, (int)prop.m_LightmapImageHeight, prop.m_OnlyMaterial,
								( uvBin >= 0 ) ? pUvPlan : NULL, uvBin, bodyID, modelID );
							if ( !tt )
								continue;
							if ( colorSlot >= pRes->m_TexelColors.Count() )
								break;
							CUtlVector<Vector> *pColors = pRes->m_TexelColors[colorSlot];
							CUtlVector<unsigned char> *pFlags = ( colorSlot < pRes->m_TexelFlags.Count() ) ? pRes->m_TexelFlags[colorSlot] : NULL;
							++colorSlot;
							if ( !pColors || tt->localPos.Count() != pColors->Count() )
								continue;
							CUtlVector<colorTexel_t> *grid = new CUtlVector<colorTexel_t>;
							grid->SetCount( pColors->Count() );
							for ( int i = 0; i < pColors->Count(); ++i )
							{
								colorTexel_t &ct = ( *grid )[i];
								memset( &ct, 0, sizeof( ct ) );
								VectorTransform( tt->localPos[i], matPos, ct.m_WorldPosition );
								VectorRotate( tt->localNrm[i], matNormal, ct.m_WorldNormal );
								ct.m_WorldNormal.NormalizeInPlace();
								ct.m_Color = ( *pColors )[i];
								ct.m_fDistanceToTri = 0;
								const unsigned char flags = ( pFlags && i < pFlags->Count() ) ? ( *pFlags )[i] : 0;
								ct.m_bValid = ( flags & 1 ) != 0;
								ct.m_bPossiblyInteresting = ( flags & 2 ) != 0;
								if ( i < tt->luxelWorld.Count() && tt->luxelWorld[i] > 0.05f )
								{
									ct.m_LuxelWorld = tt->luxelWorld[i];
									VectorRotate( tt->localAxisU[i], matNormal, ct.m_AxisU );
									VectorRotate( tt->localAxisV[i], matNormal, ct.m_AxisV );
								}
							}
							pRes->m_ColorTexelsArrays.AddToTail( grid );
							PtHealPropBrushIntersection( *grid, (int)prop.m_LightmapImageWidth, (int)prop.m_LightmapImageHeight );
							PtDilatePropTexels( *grid, (int)prop.m_LightmapImageWidth, (int)prop.m_LightmapImageHeight );
							PtDenoisePropLightmap( *grid, (int)prop.m_LightmapImageWidth, (int)prop.m_LightmapImageHeight );
						}
					}
				}
				pRes->m_TexelColors.PurgeAndDeleteElements();
				pRes->m_TexelFlags.PurgeAndDeleteElements();
			}
			ApplyLightingToStaticProp( pi, prop, pRes );
			delete pRes;
			allResults[pi] = NULL;
		}
		Msg( "Serializing static prop lighting...\n" );
		fflush( stdout );
		SerializeLighting();
		Msg( "Static prop lighting serialized.\n" );
		fflush( stdout );
	}
	else
	{
		for ( int i = 0; i < count; ++i )
			delete allResults[i];
	}

	for ( int i = 0; i < templates.Count(); ++i )
		delete templates[i];
	for ( int i = 0; i < count; ++i )
	{
		delete vertColorSum[i];
		delete vertWeightSum[i];
	}

	return ok;
}


//-----------------------------------------------------------------------------
// Computes lighting for the static props.
// Must be after all other surface lighting has been computed for the indirect sampling.
//-----------------------------------------------------------------------------
static void GatherDefaultBodyMaterials( studiohdr_t *pHdr, OptimizedModel::FileHeader_t *pVtx, CUtlVector<int> &mats )
{
	mats.RemoveAll();
	if ( !pHdr || !pVtx )
		return;

	const int nBody = min( pHdr->numbodyparts, pVtx->numBodyParts );
	for ( int bodyID = 0; bodyID < nBody; ++bodyID )
	{
		mstudiobodyparts_t *pBody = pHdr->pBodypart( bodyID );
		OptimizedModel::BodyPartHeader_t *pVtxBody = pVtx->pBodyPart( bodyID );
		if ( !pBody || pBody->nummodels <= 0 || !pVtxBody || pVtxBody->numModels <= 0 )
			continue;

		// Static props draw body 0, which is model 0 of each bodypart.
		mstudiomodel_t *pModel = pBody->pModel( 0 );
		OptimizedModel::ModelHeader_t *pVtxModel = pVtxBody->pModel( 0 );
		if ( !pModel || !pVtxModel || pVtxModel->numLODs <= 0 )
			continue;

		OptimizedModel::ModelLODHeader_t *pLod = pVtxModel->pLOD( 0 );
		const int nMesh = min( pModel->nummeshes, pLod->numMeshes );
		for ( int meshID = 0; meshID < nMesh; ++meshID )
		{
			OptimizedModel::MeshHeader_t *pVtxMesh = pLod->pMesh( meshID );
			bool hasStrips = false;
			for ( int group = 0; group < pVtxMesh->numStripGroups; ++group )
			{
				if ( pVtxMesh->pStripGroup( group )->numStrips > 0 )
				{
					hasStrips = true;
					break;
				}
			}
			if ( !hasStrips )
				continue;

			const int material = pModel->pMesh( meshID )->material;
			bool seen = false;
			for ( int i = 0; i < mats.Count(); ++i )
			{
				if ( mats[i] == material )
				{
					seen = true;
					break;
				}
			}
			if ( !seen )
				mats.AddToTail( material );
		}
	}
}

static bool StaticPropModelFilesPresent( const char *pModelName, const studiohdr_t *pHdr )
{
	if ( !pModelName || !pModelName[0] || !pHdr )
		return false;

	CUtlBuffer mdl;
	if ( !LoadFile( pModelName, mdl ) || mdl.TellPut() <= 0 )
		return false;

	char filename[MAX_PATH];
	Q_strncpy( filename, pModelName, sizeof( filename ) );
	Q_SetExtension( filename, ".vvd", sizeof( filename ) );
	CUtlBuffer vvd;
	if ( !LoadFile( filename, vvd ) || vvd.TellPut() <= 0 )
		return false;

	static const char *const exts[] = { ".dx90.vtx", ".dx80.vtx", ".sw.vtx" };
	for ( int i = 0; i < ARRAYSIZE( exts ); ++i )
	{
		Q_strncpy( filename, pModelName, sizeof( filename ) );
		Q_SetExtension( filename, exts[i], sizeof( filename ) );
		CUtlBuffer vtx;
		if ( !LoadFile( filename, vtx ) || vtx.TellPut() < (int)sizeof( OptimizedModel::FileHeader_t ) )
			continue;
		OptimizedModel::FileHeader_t *pVtx = (OptimizedModel::FileHeader_t *)vtx.Base();
		if ( pVtx->version == OPTIMIZED_MODEL_FILE_VERSION && pVtx->checkSum == pHdr->checksum )
			return true;
	}
	return false;
}

static void StripVtxToMaterial( OptimizedModel::FileHeader_t *pVtx, studiohdr_t *pHdr, int keepMaterial )
{
	if ( !pVtx || !pHdr )
		return;

	const int nBody = min( pVtx->numBodyParts, pHdr->numbodyparts );
	for ( int bodyID = 0; bodyID < nBody; ++bodyID )
	{
		OptimizedModel::BodyPartHeader_t *pVtxBody = pVtx->pBodyPart( bodyID );
		mstudiobodyparts_t *pBody = pHdr->pBodypart( bodyID );
		const int nModel = min( pVtxBody->numModels, pBody->nummodels );
		for ( int modelID = 0; modelID < nModel; ++modelID )
		{
			OptimizedModel::ModelHeader_t *pVtxModel = pVtxBody->pModel( modelID );
			mstudiomodel_t *pModel = pBody->pModel( modelID );
			for ( int lod = 0; lod < pVtxModel->numLODs; ++lod )
			{
				OptimizedModel::ModelLODHeader_t *pLod = pVtxModel->pLOD( lod );
				const int nMesh = min( pLod->numMeshes, pModel->nummeshes );
				for ( int meshID = 0; meshID < nMesh; ++meshID )
				{
					if ( pModel->pMesh( meshID )->material == keepMaterial )
						continue;
					OptimizedModel::MeshHeader_t *pVtxMesh = pLod->pMesh( meshID );
					for ( int group = 0; group < pVtxMesh->numStripGroups; ++group )
						pVtxMesh->pStripGroup( group )->numStrips = 0;
				}
			}
		}
	}
}

static int BufferPutBytes( CUtlBuffer &buf )
{
	const int put = buf.TellPut();
	return ( put > 0 ) ? put : buf.Size();
}

static bool PakModelFile( const char *pPakName, CUtlBuffer &buf )
{
	const int n = BufferPutBytes( buf );
	if ( n <= 0 || !GetPakFile() )
		return false;
	AddBufferToPak( GetPakFile(), pPakName, buf.Base(), n, false );
	return true;
}

const char *CVradStaticPropMgr::PropBakeName( const CStaticProp &prop ) const
{
	const int idx = ( prop.m_EmitModelIdx >= 0 ) ? prop.m_EmitModelIdx : prop.m_ModelIdx;
	if ( idx >= 0 && idx < m_StaticPropDict.Count() )
		return m_StaticPropDict[idx].m_Name;
	return "";
}

int CVradStaticPropMgr::PropSelfShadowSkip( const CStaticProp &prop, int propIndex ) const
{
	const int id = ( prop.m_ShadowSkipProp >= 0 ) ? prop.m_ShadowSkipProp : propIndex;
	if ( g_bDisablePropSelfShadowing || ( prop.m_Flags & STATIC_PROP_NO_SELF_SHADOWING ) )
		return id;
	return -1;
}

int CVradStaticPropMgr::FindLightmapEmit( int srcDict, int material, int uvBin ) const
{
	for ( int i = 0; i < m_LightmapMaterialModels.Count(); ++i )
	{
		if ( m_LightmapMaterialModels[i].m_SrcDict == srcDict &&
			m_LightmapMaterialModels[i].m_Material == material &&
			m_LightmapMaterialModels[i].m_UvBin == uvBin )
			return m_LightmapMaterialModels[i].m_EmitDict;
	}
	return -1;
}

void CVradStaticPropMgr::NoteLightmapUvFilter( int emitDict, int uvBin, int uvPlan )
{
	for ( int i = 0; i < m_LightmapMaterialModels.Count(); ++i )
	{
		if ( m_LightmapMaterialModels[i].m_EmitDict != emitDict )
			continue;
		m_LightmapMaterialModels[i].m_UvBin = uvBin;
		m_LightmapMaterialModels[i].m_UvPlan = uvPlan;
		return;
	}
}

const PropUvSplitPlan *CVradStaticPropMgr::PropUvPlan( int planIndex ) const
{
	if ( planIndex < 0 || planIndex >= m_UvPlans.Count() )
		return NULL;
	return &m_UvPlans[planIndex];
}

int CVradStaticPropMgr::EnsureLightmapMaterialModel( int srcDict, int material, int uvBin, int uvPlan )
{
	for ( int i = 0; i < m_LightmapMaterialModels.Count(); ++i )
	{
		if ( m_LightmapMaterialModels[i].m_SrcDict == srcDict &&
			m_LightmapMaterialModels[i].m_Material == material &&
			m_LightmapMaterialModels[i].m_UvBin == uvBin )
		{
			if ( uvPlan >= 0 )
				m_LightmapMaterialModels[i].m_UvPlan = uvPlan;
			return m_LightmapMaterialModels[i].m_EmitDict;
		}
	}

	if ( srcDict < 0 || srcDict >= m_StaticPropDict.Count() )
		return -1;

	const char *pSrcName = m_StaticPropDict[srcDict].m_Name;
	char stem[STATIC_PROP_NAME_LENGTH];
	Q_StripExtension( pSrcName, stem, sizeof( stem ) );

	char name[STATIC_PROP_NAME_LENGTH];
	int wrote = -1;
	if ( uvBin >= 0 && material >= 0 )
		wrote = Q_snprintf( name, sizeof( name ), "%s_lm%d_uv%d.mdl", stem, material, uvBin );
	else if ( uvBin >= 0 )
		wrote = Q_snprintf( name, sizeof( name ), "%s_uv%d.mdl", stem, uvBin );
	else
		wrote = Q_snprintf( name, sizeof( name ), "%s_lm%d.mdl", stem, material );
	bool collided = ( wrote < 0 || wrote >= STATIC_PROP_NAME_LENGTH );
	if ( !collided )
	{
		for ( int i = 0; i < m_StaticPropDict.Count(); ++i )
		{
			if ( !Q_stricmp( m_StaticPropDict[i].m_Name, name ) )
			{
				collided = true;
				break;
			}
		}
	}
	if ( collided )
	{
		unsigned int hash = 2166136261u;
		for ( const char *s = pSrcName; s && *s; ++s )
		{
			hash ^= (unsigned char)*s;
			hash *= 16777619u;
		}
		if ( uvBin >= 0 )
			Q_snprintf( name, sizeof( name ), "models/lmuv_%08x_%d_%d.mdl", hash, material + 1, uvBin );
		else
			Q_snprintf( name, sizeof( name ), "models/lm_%08x_%d.mdl", hash, material );
	}
	for ( char *s = name; *s; ++s )
	{
		if ( *s == '\\' )
			*s = '/';
	}

	const int emit = m_StaticPropDict.AddToTail();
	StaticPropDict_t &entry = m_StaticPropDict[emit];
	entry.m_pModel = NULL;
	entry.m_pStudioHdr = NULL;
	Q_memset( &entry.m_loadedModel, 0, sizeof( entry.m_loadedModel ) );
	entry.m_Mins.Init();
	entry.m_Maxs.Init();
	Q_strncpy( entry.m_Name, name, sizeof( entry.m_Name ) );

	LightmapMaterialModel_t rec;
	rec.m_SrcDict = srcDict;
	rec.m_Material = material;
	rec.m_UvBin = uvBin;
	rec.m_UvPlan = uvPlan;
	rec.m_EmitDict = emit;
	m_LightmapMaterialModels.AddToTail( rec );
	return emit;
}

void CVradStaticPropMgr::SplitLightmappedMaterialProps()
{
	const int srcCount = m_StaticProps.Count();

	for ( int i = 0; i < srcCount; ++i )
	{
		CStaticProp &prop = m_StaticProps[i];
		if ( prop.m_Flags & STATIC_PROP_NO_PER_TEXEL_LIGHTING )
			continue;
		if ( prop.m_LightmapImageWidth == 0 || prop.m_LightmapImageHeight == 0 )
			continue;
		if ( prop.m_ModelIdx < 0 || prop.m_ModelIdx >= m_StaticPropDict.Count() )
			continue;

		StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
		studiohdr_t *pHdr = dict.m_pStudioHdr;
		OptimizedModel::FileHeader_t *pVtx = ( dict.m_VtxBuf.Base() && dict.m_VtxBuf.TellPut() > 0 )
			? (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base()
			: NULL;
		if ( !pHdr || !pVtx )
			continue;

		CUtlVector<int> mats;
		GatherDefaultBodyMaterials( pHdr, pVtx, mats );
		if ( mats.Count() < 2 )
			continue;

		if ( !StaticPropModelFilesPresent( dict.m_Name, pHdr ) )
		{
			Warning( "Static prop \"%s\" has %d lightmapped materials, but its mdl/vvd/vtx could not be read. Leaving it as one prop.\n",
				dict.m_Name, mats.Count() );
			continue;
		}

		CUtlVector<int> emits;
		emits.SetCount( mats.Count() );
		bool ok = true;
		for ( int m = 0; m < mats.Count(); ++m )
		{
			emits[m] = EnsureLightmapMaterialModel( prop.m_ModelIdx, mats[m] );
			if ( emits[m] < 0 )
				ok = false;
		}
		if ( !ok )
			continue;

		// Re-fetch after Ensure may have grown the dict. The prop reference is
		// still valid: clones are appended after this assignment.
		m_StaticProps[i].m_OnlyMaterial = mats[0];
		m_StaticProps[i].m_EmitModelIdx = emits[0];

		for ( int m = 1; m < mats.Count(); ++m )
		{
			const int ni = m_StaticProps.AddToTail();
			const CStaticProp &srcProp = m_StaticProps[i];
			CStaticProp &clone = m_StaticProps[ni];
			clone.m_Origin = srcProp.m_Origin;
			clone.m_Angles = srcProp.m_Angles;
			clone.m_mins = srcProp.m_mins;
			clone.m_maxs = srcProp.m_maxs;
			clone.m_LightingOrigin = srcProp.m_LightingOrigin;
			clone.m_ModelIdx = srcProp.m_ModelIdx;
			clone.m_FirstLeaf = srcProp.m_FirstLeaf;
			clone.m_LeafCount = srcProp.m_LeafCount;
			clone.m_Solid = 0;
			clone.m_Handle = TREEDATA_INVALID_HANDLE;
			clone.m_Flags = srcProp.m_Flags;
			clone.m_bLightingOriginValid = srcProp.m_bLightingOriginValid;
			clone.m_LightmapImageFormat = srcProp.m_LightmapImageFormat;
			clone.m_LightmapImageWidth = srcProp.m_LightmapImageWidth;
			clone.m_LightmapImageHeight = srcProp.m_LightmapImageHeight;
			clone.m_OnlyMaterial = mats[m];
			clone.m_EmitModelIdx = emits[m];
			clone.m_ShadowSkipProp = i;
			clone.m_Skin = srcProp.m_Skin;
			clone.m_FadeMinDist = srcProp.m_FadeMinDist;
			clone.m_FadeMaxDist = srcProp.m_FadeMaxDist;
			clone.m_flForcedFadeScale = srcProp.m_flForcedFadeScale;
			clone.m_nMinDXLevel = srcProp.m_nMinDXLevel;
			clone.m_nMaxDXLevel = srcProp.m_nMaxDXLevel;
		}
	}
}

void CVradStaticPropMgr::SplitLightmappedUvProps()
{
	const int srcCount = m_StaticProps.Count();

	for ( int i = 0; i < srcCount; ++i )
	{
		if ( m_StaticProps[i].m_Flags & STATIC_PROP_NO_PER_TEXEL_LIGHTING )
			continue;
		if ( m_StaticProps[i].m_LightmapImageWidth == 0 || m_StaticProps[i].m_LightmapImageHeight == 0 )
			continue;
		const int src = m_StaticProps[i].m_ModelIdx;
		if ( src < 0 || src >= m_StaticPropDict.Count() )
			continue;

		const int material = m_StaticProps[i].m_OnlyMaterial;
		int planIndex = -1;
		for ( int p = 0; p < m_UvPlans.Count(); ++p )
		{
			if ( m_UvPlans[p].m_SrcDict == src && m_UvPlans[p].m_Material == material )
			{
				planIndex = p;
				break;
			}
		}

		StaticPropDict_t &dict = m_StaticPropDict[src];
		if ( planIndex < 0 )
		{
			studiohdr_t *pHdr = dict.m_pStudioHdr;
			OptimizedModel::FileHeader_t *pVtx = ( dict.m_VtxBuf.Base() && dict.m_VtxBuf.TellPut() > 0 )
				? (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base()
				: NULL;
			planIndex = m_UvPlans.AddToTail();
			PropUvSplitPlan &built = m_UvPlans[planIndex];
			built.m_SrcDict = src;
			built.m_Material = material;
			built.m_NumBins = 1;
			built.m_IslandCount = 0;
			if ( pHdr && pVtx && !PropUvBuildPlan( pHdr, pVtx, material, built ) )
			{
				Warning( "Static prop \"%s\" could not be split along UV islands. Leaving it as one prop.\n",
					dict.m_Name );
				built.m_NumBins = 1;
				built.m_Tris.Purge();
			}
			else if ( pHdr && built.m_NumBins > 1 && !StaticPropModelFilesPresent( dict.m_Name, pHdr ) )
			{
				Warning( "Static prop \"%s\" has overlapping UV islands, but its mdl/vvd/vtx could not be read. Leaving it as one prop.\n",
					dict.m_Name );
				built.m_NumBins = 1;
				built.m_Tris.Purge();
			}
		}

		if ( planIndex < 0 || m_UvPlans[planIndex].m_NumBins <= 1 )
			continue;

		const int nBins = m_UvPlans[planIndex].m_NumBins;
		if ( m_UvPlans[planIndex].m_EmitDict.Count() != nBins )
		{
			CUtlVector<int> emits;
			emits.SetCount( nBins );
			int reused = FindLightmapEmit( src, material, 0 );
			if ( reused < 0 )
				reused = FindLightmapEmit( src, material, -1 );
			if ( reused >= 0 )
				emits[0] = reused;
			else
				emits[0] = EnsureLightmapMaterialModel( src, material, 0, planIndex );

			bool ok = emits[0] >= 0;
			for ( int b = 1; b < nBins && ok; ++b )
			{
				emits[b] = EnsureLightmapMaterialModel( src, material, b, planIndex );
				if ( emits[b] < 0 )
					ok = false;
			}
			if ( !ok )
				continue;
			NoteLightmapUvFilter( emits[0], 0, planIndex );
			m_UvPlans[planIndex].m_EmitDict = emits;
		}

		const CUtlVector<int> &emits = m_UvPlans[planIndex].m_EmitDict;
		if ( emits.Count() != nBins )
			continue;

		const int skip = ( m_StaticProps[i].m_ShadowSkipProp >= 0 ) ? m_StaticProps[i].m_ShadowSkipProp : i;
		m_StaticProps[i].m_UvBin = 0;
		m_StaticProps[i].m_UvPlan = planIndex;
		m_StaticProps[i].m_EmitModelIdx = emits[0];

		for ( int b = 1; b < nBins; ++b )
		{
			const int ni = m_StaticProps.AddToTail();
			m_StaticProps[ni] = m_StaticProps[i];
			m_StaticProps[ni].m_Solid = 0;
			m_StaticProps[ni].m_Handle = TREEDATA_INVALID_HANDLE;
			m_StaticProps[ni].m_UvBin = b;
			m_StaticProps[ni].m_UvPlan = planIndex;
			m_StaticProps[ni].m_EmitModelIdx = emits[b];
			m_StaticProps[ni].m_ShadowSkipProp = skip;
			m_StaticProps[ni].m_MeshData.RemoveAll();
		}
	}
}

void CVradStaticPropMgr::PrintLightmapSplitStats( int nOriginalProps, int nOriginalModels )
{
	const int nProps = m_StaticProps.Count();
	const int nModels = m_StaticPropDict.Count();
	const int nExtraProps = nProps - nOriginalProps;
	const int nExtraModels = nModels - nOriginalModels;

	int nUvExtraProps = 0;
	for ( int i = nOriginalProps; i < nProps; ++i )
	{
		if ( m_StaticProps[i].m_UvBin > 0 )
			++nUvExtraProps;
	}
	const int nMatExtraProps = nExtraProps - nUvExtraProps;

	CUtlVector<int> srcs;
	for ( int i = 0; i < m_LightmapMaterialModels.Count(); ++i )
	{
		const int src = m_LightmapMaterialModels[i].m_SrcDict;
		if ( srcs.Find( src ) < 0 )
			srcs.AddToTail( src );
	}

	Msg( "Static props: %d original + %d from splits = %d.\n", nOriginalProps, nExtraProps, nProps );
	Msg( "Models: %d original + %d from splits = %d.\n", nOriginalModels, nExtraModels, nModels );

	if ( srcs.Count() <= 0 )
	{
		Msg( "Lightmap splits: none.\n" );
		return;
	}

	int nMatModels = 0;
	int nUvModels = 0;
	for ( int s = 0; s < srcs.Count(); ++s )
	{
		bool anyUv = false;
		CUtlVector<int> mats;
		for ( int i = 0; i < m_LightmapMaterialModels.Count(); ++i )
		{
			if ( m_LightmapMaterialModels[i].m_SrcDict != srcs[s] )
				continue;
			if ( m_LightmapMaterialModels[i].m_UvBin >= 0 )
				anyUv = true;
			if ( mats.Find( m_LightmapMaterialModels[i].m_Material ) < 0 )
				mats.AddToTail( m_LightmapMaterialModels[i].m_Material );
		}
		if ( mats.Count() >= 2 )
			++nMatModels;
		if ( anyUv )
			++nUvModels;
	}

	Msg( "Lightmap splits: %d models into %d (%d by material, +%d props; %d by UV islands, +%d props).\n",
		srcs.Count(), m_LightmapMaterialModels.Count(), nMatModels, nMatExtraProps, nUvModels, nUvExtraProps );

	for ( int s = 0; s < srcs.Count(); ++s )
	{
		const int src = srcs[s];
		int pieces = 0;
		bool anyUv = false;
		CUtlVector<int> mats;
		for ( int i = 0; i < m_LightmapMaterialModels.Count(); ++i )
		{
			if ( m_LightmapMaterialModels[i].m_SrcDict != src )
				continue;
			++pieces;
			if ( m_LightmapMaterialModels[i].m_UvBin >= 0 )
				anyUv = true;
			if ( mats.Find( m_LightmapMaterialModels[i].m_Material ) < 0 )
				mats.AddToTail( m_LightmapMaterialModels[i].m_Material );
		}

		int placed = 0;
		int copies = 0;
		for ( int i = 0; i < nProps; ++i )
		{
			if ( m_StaticProps[i].m_ModelIdx != src || m_StaticProps[i].m_EmitModelIdx < 0 )
				continue;
			if ( i < nOriginalProps )
				++placed;
			else
				++copies;
		}

		const char *pWhy = "material";
		if ( anyUv && mats.Count() >= 2 )
			pWhy = "material + UV";
		else if ( anyUv )
			pWhy = "UV islands";

		const char *pName = ( src >= 0 && src < m_StaticPropDict.Count() ) ? m_StaticPropDict[src].m_Name : "";
		Msg( "  \"%s\" -> %d models (%s, %d placed, +%d props).\n", pName, pieces, pWhy, placed, copies );
	}
}

void CVradStaticPropMgr::WriteStaticPropGameLump()
{
	if ( m_LightmapMaterialModels.Count() <= 0 )
		return;

	for ( int i = 0; i < m_LightmapMaterialModels.Count(); ++i )
	{
		const LightmapMaterialModel_t &rec = m_LightmapMaterialModels[i];
		if ( rec.m_SrcDict < 0 || rec.m_SrcDict >= m_StaticPropDict.Count() )
			continue;
		if ( rec.m_EmitDict < 0 || rec.m_EmitDict >= m_StaticPropDict.Count() )
			continue;

		StaticPropDict_t &src = m_StaticPropDict[rec.m_SrcDict];
		const char *pEmitName = m_StaticPropDict[rec.m_EmitDict].m_Name;
		studiohdr_t *pHdr = src.m_pStudioHdr;
		if ( !pHdr || !pEmitName[0] )
			continue;

		CUtlBuffer mdl;
		if ( !LoadFile( src.m_Name, mdl ) || !PakModelFile( pEmitName, mdl ) )
		{
			Warning( "Failed to pack material model \"%s\".\n", pEmitName );
			continue;
		}

		char filename[MAX_PATH];
		char pakName[MAX_PATH];
		Q_strncpy( filename, src.m_Name, sizeof( filename ) );
		Q_SetExtension( filename, ".vvd", sizeof( filename ) );
		Q_strncpy( pakName, pEmitName, sizeof( pakName ) );
		Q_SetExtension( pakName, ".vvd", sizeof( pakName ) );
		CUtlBuffer vvd;
		if ( !LoadFile( filename, vvd ) || !PakModelFile( pakName, vvd ) )
			Warning( "Failed to pack \"%s\".\n", pakName );

		Q_strncpy( filename, src.m_Name, sizeof( filename ) );
		Q_SetExtension( filename, ".phy", sizeof( filename ) );
		Q_strncpy( pakName, pEmitName, sizeof( pakName ) );
		Q_SetExtension( pakName, ".phy", sizeof( pakName ) );
		CUtlBuffer phy;
		if ( LoadFile( filename, phy ) && BufferPutBytes( phy ) > 0 )
			PakModelFile( pakName, phy );

		static const char *const exts[] = { ".dx90.vtx", ".dx80.vtx", ".sw.vtx" };
		for ( int ext = 0; ext < ARRAYSIZE( exts ); ++ext )
		{
			Q_strncpy( filename, src.m_Name, sizeof( filename ) );
			Q_SetExtension( filename, exts[ext], sizeof( filename ) );
			CUtlBuffer vtx;
			if ( !LoadFile( filename, vtx ) || vtx.TellPut() < (int)sizeof( OptimizedModel::FileHeader_t ) )
				continue;
			OptimizedModel::FileHeader_t *pVtx = (OptimizedModel::FileHeader_t *)vtx.Base();
			if ( pVtx->version != OPTIMIZED_MODEL_FILE_VERSION || pVtx->checkSum != pHdr->checksum )
				continue;
			if ( rec.m_Material >= 0 )
				StripVtxToMaterial( pVtx, pHdr, rec.m_Material );
			if ( rec.m_UvBin >= 0 )
			{
				const PropUvSplitPlan *pPlan = PropUvPlan( rec.m_UvPlan );
				if ( pPlan )
					PropUvDegenerateOutsideBin( pHdr, pVtx, *pPlan, rec.m_UvBin );
			}
			Q_strncpy( pakName, pEmitName, sizeof( pakName ) );
			Q_SetExtension( pakName, exts[ext], sizeof( pakName ) );
			if ( !PakModelFile( pakName, vtx ) )
				Warning( "Failed to pack \"%s\".\n", pakName );
		}
	}

	CUtlVector<StaticPropDictLump_t> dicts;
	dicts.SetCount( m_StaticPropDict.Count() );
	for ( int i = 0; i < dicts.Count(); ++i )
	{
		Q_memset( &dicts[i], 0, sizeof( dicts[i] ) );
		Q_strncpy( dicts[i].m_Name, m_StaticPropDict[i].m_Name, sizeof( dicts[i].m_Name ) );
	}

	CUtlVector<StaticPropLump_t> props;
	props.SetCount( m_StaticProps.Count() );
	for ( int i = 0; i < props.Count(); ++i )
	{
		const CStaticProp &prop = m_StaticProps[i];
		StaticPropLump_t &lump = props[i];
		Q_memset( &lump, 0, sizeof( lump ) );
		lump.m_Origin = prop.m_Origin;
		lump.m_Angles = prop.m_Angles;
		const int model = ( prop.m_EmitModelIdx >= 0 ) ? prop.m_EmitModelIdx : prop.m_ModelIdx;
		lump.m_PropType = (unsigned short)model;
		lump.m_FirstLeaf = prop.m_FirstLeaf;
		lump.m_LeafCount = prop.m_LeafCount;
		lump.m_Solid = prop.m_Solid;
		lump.m_Skin = prop.m_Skin;
		lump.m_FadeMinDist = prop.m_FadeMinDist;
		lump.m_FadeMaxDist = prop.m_FadeMaxDist;
		lump.m_LightingOrigin = prop.m_LightingOrigin;
		lump.m_flForcedFadeScale = prop.m_flForcedFadeScale;
		lump.m_nMinDXLevel = prop.m_nMinDXLevel;
		lump.m_nMaxDXLevel = prop.m_nMaxDXLevel;
		lump.m_Flags = prop.m_Flags;
		lump.m_nLightmapResolutionX = (unsigned short)prop.m_LightmapImageWidth;
		lump.m_nLightmapResolutionY = (unsigned short)prop.m_LightmapImageHeight;
	}

	const int dictBytes = dicts.Count() * (int)sizeof( StaticPropDictLump_t );
	const int leafBytes = m_StaticPropLeaves.Count() * (int)sizeof( StaticPropLeafLump_t );
	const int propBytes = props.Count() * (int)sizeof( StaticPropLump_t );
	const int size = dictBytes + leafBytes + propBytes + 3 * (int)sizeof( int );

	GameLumpHandle_t handle = g_GameLumps.GetGameLumpHandle( GAMELUMP_STATIC_PROPS );
	if ( handle != g_GameLumps.InvalidGameLump() )
		g_GameLumps.DestroyGameLump( handle );
	handle = g_GameLumps.CreateGameLump( GAMELUMP_STATIC_PROPS, size, 0, GAMELUMP_STATIC_PROPS_VERSION );

	CUtlBuffer buf( g_GameLumps.GetGameLump( handle ), size );
	buf.PutInt( dicts.Count() );
	if ( dictBytes )
		buf.Put( dicts.Base(), dictBytes );
	buf.PutInt( m_StaticPropLeaves.Count() );
	if ( leafBytes )
		buf.Put( m_StaticPropLeaves.Base(), leafBytes );
	buf.PutInt( props.Count() );
	if ( propBytes )
		buf.Put( props.Base(), propBytes );
}

void CVradStaticPropMgr::ComputeLighting( int iThread )
{
	// illuminate them all
	int count = m_StaticProps.Count();
	if ( !count )
	{
		// nothing to do
		return;
	}

	StartPacifier( "Computing static prop lighting : " );

	// ensure any traces against us are ignored because we have no inherit lighting contribution
	m_bIgnoreStaticPropTrace = true;

	bool bGpuProps = false;
#ifdef MPI
	if ( !g_bUseMPI )
#endif
	{
		if ( PathTraceDXR_CanBakeProps() )
		{
			SuppressPacifier( true );
			bGpuProps = ComputeLightingPathTraceGPU();
			SuppressPacifier( false );
			if ( !bGpuProps )
				Warning( "[PathTrace-DXR] Prop GPU bake failed - falling back to stock CPU prop lighting.\n" );
		}
	}

	if ( !bGpuProps )
	{
#ifdef MPI
	if ( g_bUseMPI )
	{
		// Distribute the work among the workers.
		VMPI_SetCurrentStage( "CVradStaticPropMgr::ComputeLighting" );
		
		DistributeWork( 
			count, 
			&CVradStaticPropMgr::VMPI_ProcessStaticProp_Static, 
			&CVradStaticPropMgr::VMPI_ReceiveStaticPropResults_Static );
	}
	else
#endif
	{
		RunThreadsOn(count, true, ThreadComputeStaticPropLighting);
	}

	// restore default
	m_bIgnoreStaticPropTrace = false;

	// save data to bsp
	SerializeLighting();
	}
	else
	{
		m_bIgnoreStaticPropTrace = false;
	}

	EndPacifier( true );
}

//-----------------------------------------------------------------------------
// Adds all static prop polys to the ray trace store.
//-----------------------------------------------------------------------------
void CVradStaticPropMgr::AddPolysForRayTrace( void )
{
	int count = m_StaticProps.Count();
	if ( !count )
	{
		// nothing to do
		return;
	}

	PathTraceDXR_ClearPropInstances();
	const bool bRegisterDxr = PathTraceDXR_IsRequested();
	static CUtlVector<int> s_registeredPhys;
	static CUtlVector<int> s_registeredStudio;
	s_registeredPhys.RemoveAll();
	s_registeredStudio.RemoveAll();

	// Triangle coverage of 1 (full coverage)
	Vector fullCoverage;
	fullCoverage.x = 1.0f;

	for ( int nProp = 0; nProp < count; ++nProp )
	{
		CStaticProp &prop = m_StaticProps[nProp];
		StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];

		if ( prop.m_Flags & STATIC_PROP_NO_SHADOW )
			continue;

		// Extra materials are their own props in the BSP. Occlusion stays on the
		// primary, which still references the full studio mesh.
		if ( prop.m_ShadowSkipProp >= 0 )
			continue;

		// Phase-2 DXR: one local BLAS per unique model + TLAS instance.
		if ( bRegisterDxr )
		{
			matrix3x4_t matrix;
			AngleMatrix( prop.m_Angles, prop.m_Origin, matrix );
			PathTraceDXR_RegisterPropInstance( nProp, prop.m_ModelIdx, matrix.m_flMatVal );
		}

		// If not using static prop polys, use AABB
		if ( !g_bStaticPropPolys )
		{
			if ( dict.m_pModel )
			{
				if ( bRegisterDxr )
				{
					// Register local-space VPhysics tris once per model.
					bool already = false;
					for ( int i = 0; i < s_registeredPhys.Count(); ++i )
					{
						if ( s_registeredPhys[i] == prop.m_ModelIdx ) { already = true; break; }
					}
					if ( !already )
					{
						CUtlVector<Vector> localVerts;
						ICollisionQuery *queryModel = s_pPhysCollision->CreateQueryModel( dict.m_pModel );
						for ( int nConvex = 0; nConvex < queryModel->ConvexCount(); ++nConvex )
						{
							for ( int nTri = 0; nTri < queryModel->TriangleCount( nConvex ); ++nTri )
							{
								Vector verts[3];
								queryModel->GetTriangleVerts( nConvex, nTri, verts );
								localVerts.AddToTail( verts[0] );
								localVerts.AddToTail( verts[1] );
								localVerts.AddToTail( verts[2] );
							}
						}
						s_pPhysCollision->DestroyQueryModel( queryModel );
						if ( localVerts.Count() >= 3 )
						{
							PathTraceDXR_RegisterPropModel( prop.m_ModelIdx, localVerts.Base(), localVerts.Count() / 3 );
							s_registeredPhys.AddToTail( prop.m_ModelIdx );
						}
					}
				}
				VMatrix xform;
				xform.SetupMatrixOrgAngles ( prop.m_Origin, prop.m_Angles );
				ICollisionQuery *queryModel = s_pPhysCollision->CreateQueryModel( dict.m_pModel );
				for ( int nConvex = 0; nConvex < queryModel->ConvexCount(); ++nConvex )
				{
					for ( int nTri = 0; nTri < queryModel->TriangleCount( nConvex ); ++nTri )
					{
						Vector verts[3];
						queryModel->GetTriangleVerts( nConvex, nTri, verts );
						for ( int nVert = 0; nVert < 3; ++nVert )
							verts[nVert] = xform.VMul4x3(verts[nVert]);
						g_RtEnv.AddTriangle ( TRACE_ID_STATICPROP | nProp, verts[0], verts[1], verts[2], fullCoverage );
					}
				}
				s_pPhysCollision->DestroyQueryModel( queryModel );
			}
			else
			{
				g_RtEnv.AddAxisAlignedRectangularSolid ( TRACE_ID_STATICPROP | nProp, prop.m_mins, prop.m_maxs, fullCoverage );
			}
			
			continue;
		}

		studiohdr_t	*pStudioHdr = dict.m_pStudioHdr;
		OptimizedModel::FileHeader_t *pVtxHdr = (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base();
		if ( !pStudioHdr || !pVtxHdr )
		{
			// must have model and its verts for decoding triangles
			return;
		}
		// only init the triangle table the first time
		bool bInitTriangles = dict.m_triangleMaterialIndex.Count() ? false : true;
		int triangleIndex = 0;

		CUtlVector<Vector> *pLocalDxr = nullptr;
		CUtlVector<int> *pLocalDxrMats = nullptr;
		CUtlVector<Vector> *pLocalDxrAlbedo = nullptr;
		CUtlVector<Vector> localDxrVerts;
		CUtlVector<int> localDxrMats;
		CUtlVector<Vector> localDxrAlbedo;
		struct PropBrdfName_t { char s[160]; };
		CUtlVector<float> localDxrUV;
		CUtlVector<PropBrdfName_t> localDxrMat;
		if ( bRegisterDxr )
		{
			bool already = false;
			for ( int i = 0; i < s_registeredStudio.Count(); ++i )
			{
				if ( s_registeredStudio[i] == prop.m_ModelIdx ) { already = true; break; }
			}
			if ( !already )
			{
				pLocalDxr = &localDxrVerts;
				pLocalDxrMats = &localDxrMats;
				pLocalDxrAlbedo = &localDxrAlbedo;
				s_registeredStudio.AddToTail( prop.m_ModelIdx );
			}
		}

		// meshes are deeply hierarchial, divided between three stores, follow the white rabbit
		// body parts -> models -> lod meshes -> strip groups -> strips
		// the vertices and indices are pooled, the trick is knowing the offset to determine your indexed base 
		for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts; ++bodyID )
		{
			OptimizedModel::BodyPartHeader_t* pVtxBodyPart = pVtxHdr->pBodyPart( bodyID );
			mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );

			for ( int modelID = 0; modelID < pBodyPart->nummodels; ++modelID )
			{
				OptimizedModel::ModelHeader_t* pVtxModel = pVtxBodyPart->pModel( modelID );
				mstudiomodel_t *pStudioModel = pBodyPart->pModel( modelID );

				// assuming lod 0, could iterate if required
				int nLod = 0;
				OptimizedModel::ModelLODHeader_t *pVtxLOD = pVtxModel->pLOD( nLod );

				for ( int nMesh = 0; nMesh < pStudioModel->nummeshes; ++nMesh )
				{
					// check if this mesh's material is in the no shadow material name list
					mstudiomesh_t* pMesh = pStudioModel->pMesh( nMesh );
					mstudiotexture_t *pTxtr=pStudioHdr->pTexture(pMesh->material);
					//printf("mat idx=%d mat name=%s\n",pMesh->material,pTxtr->pszName());
					bool bSkipThisMesh = false;
					for(int check=0; check<g_NonShadowCastingMaterialStrings.Count(); check++)
					{
						if ( Q_stristr( pTxtr->pszName(),
										g_NonShadowCastingMaterialStrings[check] ) )
						{
							//printf("skip mat name=%s\n",pTxtr->pszName());
							bSkipThisMesh = true;
							break;
						}
					}
					if ( bSkipThisMesh)
						continue;

					int shadowTextureIndex = -1;
					if ( dict.m_textureShadowIndex.Count() )
					{
						shadowTextureIndex = dict.m_textureShadowIndex[pMesh->material];
					}


					OptimizedModel::MeshHeader_t* pVtxMesh = pVtxLOD->pMesh( nMesh );
					const mstudio_meshvertexdata_t *vertData = pMesh->GetVertexData( (void *)pStudioHdr );
					Assert( vertData ); // This can only return NULL on X360 for now

					for ( int nGroup = 0; nGroup < pVtxMesh->numStripGroups; ++nGroup )
					{
						OptimizedModel::StripGroupHeader_t* pStripGroup = pVtxMesh->pStripGroup( nGroup );

						int nStrip;
						for ( nStrip = 0; nStrip < pStripGroup->numStrips; nStrip++ )
						{
							OptimizedModel::StripHeader_t *pStrip = pStripGroup->pStrip( nStrip );

							if ( pStrip->flags & OptimizedModel::STRIP_IS_TRILIST )
							{
								for ( int i = 0; i < pStrip->numIndices; i += 3 )
								{
									int idx = pStrip->indexOffset + i;

									unsigned short i1 = *pStripGroup->pIndex( idx );
									unsigned short i2 = *pStripGroup->pIndex( idx + 1 );
									unsigned short i3 = *pStripGroup->pIndex( idx + 2 );

									int vertex1 = pStripGroup->pVertex( i1 )->origMeshVertID;
									int vertex2 = pStripGroup->pVertex( i2 )->origMeshVertID;
									int vertex3 = pStripGroup->pVertex( i3 )->origMeshVertID;

									// transform position into world coordinate system
									matrix3x4_t	matrix;
									AngleMatrix( prop.m_Angles, prop.m_Origin, matrix );

									Vector position1;
									Vector position2;
									Vector position3;
									VectorTransform( *vertData->Position( vertex1 ), matrix, position1 );
									VectorTransform( *vertData->Position( vertex2 ), matrix, position2 );
									VectorTransform( *vertData->Position( vertex3 ), matrix, position3 );
									unsigned short flags = 0;
									int materialIndex = -1;
									Vector color = vec3_origin;
									if ( shadowTextureIndex >= 0 )
									{
										if ( bInitTriangles )
										{
											// Always register alpha/translucent tris for per-texel sampling.
											// (Old coverage==1 skip treated translucent leaf cards as opaque.)
											float coverage = g_ShadowTextureList.ComputeCoverageForTriangle(shadowTextureIndex, *vertData->Texcoord(vertex1), *vertData->Texcoord(vertex2), *vertData->Texcoord(vertex3) );
											materialIndex = g_ShadowTextureList.AddMaterialEntry( shadowTextureIndex, *vertData->Texcoord(vertex1), *vertData->Texcoord(vertex2), *vertData->Texcoord(vertex3) );
											color.x = coverage;
											dict.m_triangleMaterialIndex.AddToTail(materialIndex);
										}
										else
										{
											materialIndex = dict.m_triangleMaterialIndex[triangleIndex];
											triangleIndex++;
										}
										if ( materialIndex >= 0 )
										{
											flags = FCACHETRI_TRANSPARENT;
										}
									}
									// Include alpha-tested cards in DXR AS; GPU VisRGB samples
									// coverage (stochastic alpha). Keep material index for atlas.
									if ( pLocalDxr )
									{
										pLocalDxr->AddToTail( *vertData->Position( vertex1 ) );
										pLocalDxr->AddToTail( *vertData->Position( vertex2 ) );
										pLocalDxr->AddToTail( *vertData->Position( vertex3 ) );
										if ( pLocalDxrMats )
											pLocalDxrMats->AddToTail( materialIndex );
										const Vector2D uv0 = *vertData->Texcoord( vertex1 );
										const Vector2D uv1 = *vertData->Texcoord( vertex2 );
										const Vector2D uv2 = *vertData->Texcoord( vertex3 );
										localDxrUV.AddToTail( uv0.x ); localDxrUV.AddToTail( uv0.y );
										localDxrUV.AddToTail( uv1.x ); localDxrUV.AddToTail( uv1.y );
										localDxrUV.AddToTail( uv2.x ); localDxrUV.AddToTail( uv2.y );
										PropBrdfName_t nm;
										nm.s[0] = 0;
										if ( pTxtr && pTxtr->pszName() )
											Q_strncpy( nm.s, pTxtr->pszName(), sizeof( nm.s ) );
										localDxrMat.AddToTail( nm );
										if ( pLocalDxrAlbedo )
										{
											Vector alb( -1.0f, -1.0f, -1.0f );
											if ( pTxtr && pTxtr->pszName() )
											{
												const Vector2D uvs[3] = { uv0, uv1, uv2 };
												Vector acc( 0, 0, 0 );
												int ns = 0;
												for ( int s = 0; s < 3; ++s )
												{
													Vector c;
													if ( BounceAlbedo_SampleMaterialUV( pTxtr->pszName(), uvs[s].x, uvs[s].y, c ) )
													{
														acc += c;
														++ns;
													}
												}
												if ( ns > 0 )
													alb = acc * ( 1.0f / (float)ns );
											}
											pLocalDxrAlbedo->AddToTail( alb );
										}
									}
// 		printf( "\ngl 3\n" );
// 		printf( "gl %6.3f %6.3f %6.3f 1 0 0\n", XYZ(position1));
// 		printf( "gl %6.3f %6.3f %6.3f 0 1 0\n", XYZ(position2));
// 		printf( "gl %6.3f %6.3f %6.3f 0 0 1\n", XYZ(position3));
									g_RtEnv.AddTriangle( TRACE_ID_STATICPROP | nProp,
														 position1, position2, position3,
														 color, flags, materialIndex);
								}
							}
							else
							{
								// all tris expected to be discrete tri lists
								// must fixme if stripping ever occurs
								printf( "unexpected strips found\n" );
								Assert( 0 );
								return;
							}
						}
					}
				}
			}
		}
		if ( pLocalDxr && localDxrVerts.Count() >= 3 )
		{
			const int nLocalTris = localDxrVerts.Count() / 3;
			CUtlVector<const char *> matPtrs;
			if ( localDxrMat.Count() == nLocalTris )
			{
				matPtrs.SetCount( nLocalTris );
				for ( int i = 0; i < nLocalTris; ++i )
					matPtrs[i] = localDxrMat[i].s;
			}
			PathTraceDXR_RegisterPropModel( prop.m_ModelIdx, localDxrVerts.Base(), nLocalTris,
											( localDxrMats.Count() == nLocalTris ) ? localDxrMats.Base() : nullptr,
											( localDxrAlbedo.Count() == nLocalTris ) ? localDxrAlbedo.Base() : nullptr,
											( localDxrUV.Count() == nLocalTris * 6 ) ? localDxrUV.Base() : nullptr,
											( matPtrs.Count() == nLocalTris ) ? matPtrs.Base() : nullptr );
		}
	}
}

struct tl_tri_t
{
	Vector	p0;
	Vector	p1;
	Vector	p2;
	Vector	n0;
	Vector	n1;
	Vector	n2;

	bool operator == (const tl_tri_t &t) const 
	{ 
		return ( p0 == t.p0 && 
				p1 == t.p1 && 
				p2 == t.p2 &&
				n0 == t.n0 &&
				n1 == t.n1 &&
				n2 == t.n2 );
	}
};

struct tl_vert_t
{
	Vector m_position;
	CUtlLinkedList< tl_tri_t, int > m_triList;
};

void AddTriVertsToList( CUtlVector< tl_vert_t > &triListVerts, int vertIndex, Vector vertPosition, Vector p0, Vector p1, Vector p2, Vector n0, Vector n1, Vector n2 )
{
	tl_tri_t tlTri;

	tlTri.p0 = p0;
	tlTri.p1 = p1;
	tlTri.p2 = p2;
	tlTri.n0 = n0;
	tlTri.n1 = n1;
	tlTri.n2 = n2;

	triListVerts.EnsureCapacity( vertIndex+1 );

	triListVerts[vertIndex].m_position = vertPosition;

	int index = triListVerts[vertIndex].m_triList.Find( tlTri );
	if ( !triListVerts[vertIndex].m_triList.IsValidIndex( index ) )
	{
		// not in list, add to list of triangles
		triListVerts[vertIndex].m_triList.AddToTail( tlTri );
	}
}

//-----------------------------------------------------------------------------
// Builds a list of tris for every vertex
//-----------------------------------------------------------------------------
void CVradStaticPropMgr::BuildTriList( CStaticProp &prop )
{
	// the generated list will consist of a list of verts
	// each vert will have a linked list of triangles that it belongs to
	CUtlVector< tl_vert_t >	triListVerts;

	StaticPropDict_t &dict = m_StaticPropDict[prop.m_ModelIdx];
	studiohdr_t	*pStudioHdr = dict.m_pStudioHdr;
	OptimizedModel::FileHeader_t *pVtxHdr = (OptimizedModel::FileHeader_t *)dict.m_VtxBuf.Base();
	if ( !pStudioHdr || !pVtxHdr )
	{
		// must have model and its verts for decoding triangles
		return;
	}

	// meshes are deeply hierarchial, divided between three stores, follow the white rabbit
	// body parts -> models -> lod meshes -> strip groups -> strips
	// the vertices and indices are pooled, the trick is knowing the offset to determine your indexed base 
	for ( int bodyID = 0; bodyID < pStudioHdr->numbodyparts; ++bodyID )
	{
		OptimizedModel::BodyPartHeader_t* pVtxBodyPart = pVtxHdr->pBodyPart( bodyID );
		mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart( bodyID );

		for ( int modelID = 0; modelID < pBodyPart->nummodels; ++modelID )
		{
			OptimizedModel::ModelHeader_t* pVtxModel = pVtxBodyPart->pModel( modelID );
			mstudiomodel_t *pStudioModel = pBodyPart->pModel( modelID );

			// get the specified lod, assuming lod 0
			int nLod = 0;
			OptimizedModel::ModelLODHeader_t *pVtxLOD = pVtxModel->pLOD( nLod );

			// must reset because each model has their own vertexes [0..n]
			// in order for this to be monolithic for the entire prop the list must be segmented
			triListVerts.Purge();

			for ( int nMesh = 0; nMesh < pStudioModel->nummeshes; ++nMesh )
			{
				mstudiomesh_t* pMesh = pStudioModel->pMesh( nMesh );
				OptimizedModel::MeshHeader_t* pVtxMesh = pVtxLOD->pMesh( nMesh );
				const mstudio_meshvertexdata_t *vertData = pMesh->GetVertexData( (void *)pStudioHdr );
				Assert( vertData ); // This can only return NULL on X360 for now

				for ( int nGroup = 0; nGroup < pVtxMesh->numStripGroups; ++nGroup )
				{
					OptimizedModel::StripGroupHeader_t* pStripGroup = pVtxMesh->pStripGroup( nGroup );

					int nStrip;
					for ( nStrip = 0; nStrip < pStripGroup->numStrips; nStrip++ )
					{
						OptimizedModel::StripHeader_t *pStrip = pStripGroup->pStrip( nStrip );

						if ( pStrip->flags & OptimizedModel::STRIP_IS_TRILIST )
						{
							for ( int i = 0; i < pStrip->numIndices; i += 3 )
							{
								int idx = pStrip->indexOffset + i;

								unsigned short i1 = *pStripGroup->pIndex( idx );
								unsigned short i2 = *pStripGroup->pIndex( idx + 1 );
								unsigned short i3 = *pStripGroup->pIndex( idx + 2 );

								int vertex1 = pStripGroup->pVertex( i1 )->origMeshVertID;
								int vertex2 = pStripGroup->pVertex( i2 )->origMeshVertID;
								int vertex3 = pStripGroup->pVertex( i3 )->origMeshVertID;

								// transform position into world coordinate system
								matrix3x4_t	matrix;
								AngleMatrix( prop.m_Angles, prop.m_Origin, matrix );

								Vector position1;
								Vector position2;
								Vector position3;
								VectorTransform( *vertData->Position( vertex1 ), matrix, position1 );
								VectorTransform( *vertData->Position( vertex2 ), matrix, position2 );
								VectorTransform( *vertData->Position( vertex3 ), matrix, position3 );

								Vector normal1;
								Vector normal2;
								Vector normal3;
								VectorTransform( *vertData->Normal( vertex1 ), matrix, normal1 );
								VectorTransform( *vertData->Normal( vertex2 ), matrix, normal2 );
								VectorTransform( *vertData->Normal( vertex3 ), matrix, normal3 );

								AddTriVertsToList( triListVerts, pMesh->vertexoffset + vertex1, position1, position1, position2, position3, normal1, normal2, normal3 );
								AddTriVertsToList( triListVerts, pMesh->vertexoffset + vertex2, position2, position1, position2, position3, normal1, normal2, normal3 );
								AddTriVertsToList( triListVerts, pMesh->vertexoffset + vertex3, position3, position1, position2, position3, normal1, normal2, normal3 );
							}
						}
						else
						{
							// all tris expected to be discrete tri lists
							// must fixme if stripping ever occurs
							printf( "unexpected strips found\n" );
							Assert( 0 );
							return;
						}
					}
				}
			}
		}
	}
}

const vertexFileHeader_t * mstudiomodel_t::CacheVertexData( void *pModelData )
{
	studiohdr_t *pActiveStudioHdr = static_cast<studiohdr_t *>(pModelData);
	Assert( pActiveStudioHdr );

	if ( pActiveStudioHdr->VertexBase() )
	{
		return (vertexFileHeader_t *)pActiveStudioHdr->VertexBase();
	}

	// mandatory callback to make requested data resident
	// load and persist the vertex file
	char fileName[MAX_PATH];
	strcpy( fileName, "models/" );	
	strcat( fileName, pActiveStudioHdr->pszName() );
	Q_StripExtension( fileName, fileName, sizeof( fileName ) );
	strcat( fileName, ".vvd" );

	// load the model
	FileHandle_t fileHandle = g_pFileSystem->Open( fileName, "rb" );
	if ( !fileHandle )
	{
		Error( "Unable to load vertex data \"%s\"\n", fileName );
	}

	// Get the file size
	int vvdSize = g_pFileSystem->Size( fileHandle );
	if ( vvdSize == 0 )
	{
		g_pFileSystem->Close( fileHandle );
		Error( "Bad size for vertex data \"%s\"\n", fileName );
	}

	vertexFileHeader_t *pVvdHdr = (vertexFileHeader_t *)malloc( vvdSize );
	g_pFileSystem->Read( pVvdHdr, vvdSize, fileHandle );
	g_pFileSystem->Close( fileHandle );

	// check header
	if ( pVvdHdr->id != MODEL_VERTEX_FILE_ID )
	{
		Error("Error Vertex File %s id %d should be %d\n", fileName, pVvdHdr->id, MODEL_VERTEX_FILE_ID);
	}
	if ( pVvdHdr->version != MODEL_VERTEX_FILE_VERSION )
	{
		Error("Error Vertex File %s version %d should be %d\n", fileName, pVvdHdr->version, MODEL_VERTEX_FILE_VERSION);
	}
	if ( pVvdHdr->checksum != pActiveStudioHdr->checksum )
	{
		Error("Error Vertex File %s checksum %d should be %d\n", fileName, pVvdHdr->checksum, pActiveStudioHdr->checksum);
	}

	// need to perform mesh relocation fixups
	// allocate a new copy
	vertexFileHeader_t *pNewVvdHdr = (vertexFileHeader_t *)malloc( vvdSize );
	if ( !pNewVvdHdr )
	{
		Error( "Error allocating %d bytes for Vertex File '%s'\n", vvdSize, fileName );
	}

	// load vertexes and run fixups
	Studio_LoadVertexes( pVvdHdr, pNewVvdHdr, 0, true );

	// discard original
	free( pVvdHdr );
	pVvdHdr = pNewVvdHdr;

	pActiveStudioHdr->SetVertexBase( (void*)pVvdHdr );
	return pVvdHdr;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------
struct ColorTexelValue
{
	Vector mLinearColor;	// Linear color value for this texel
	bool mValidData;		// Whether there is valid data in this texel.
	size_t mTriangleIndex;	// Which triangle we used to generate the texel.
};

// ------------------------------------------------------------------------------------------------
inline int ComputeLinearPos( int _x, int _y, int _resX, int _resY )
{
	return Min( Max( 0, _y ), _resY - 1 ) * _resX
		 + Min( Max( 0, _x ), _resX - 1 );
}

// ------------------------------------------------------------------------------------------------
inline float ComputeBarycentricDistanceToTri( Vector _barycentricCoord, Vector2D _v[3] )
{
	Vector2D realPos = _barycentricCoord.x * _v[0]
		             + _barycentricCoord.y * _v[1]
					 + _barycentricCoord.z * _v[2];

	int minIndex = 0;
	float minVal = _barycentricCoord[0];
	for (int i = 1; i < 3; ++i) {
		if (_barycentricCoord[i] < minVal) {
			minVal = _barycentricCoord[i];
			minIndex = i;
		}
	}

	Vector2D& first  = _v[ (minIndex + 1) % 3];
	Vector2D& second = _v[ (minIndex + 2) % 3];

	return CalcDistanceToLineSegment2D( realPos, first, second );
}

// ------------------------------------------------------------------------------------------------
static void GenerateLightmapSamplesForMesh( const matrix3x4_t& _matPos, const matrix3x4_t& _matNormal, int _iThread, int _skipProp, int _flags, int _lightmapResX, int _lightmapResY, studiohdr_t* _pStudioHdr, mstudiomodel_t* _pStudioModel, OptimizedModel::ModelHeader_t* _pVtxModel, int _meshID, CComputeStaticPropLightingResults *_outResults, bool _bComputeLighting, const Vector *_pPropOrigin, const char *_pPropModel, bool _bResetTexels, const PropUvSplitPlan *_pUvPlan, int _uvBin, int _uvBody, int _uvModel )
{
	// Could iterate and gen this if needed.
	int nLod = 0;

	OptimizedModel::ModelLODHeader_t *pVtxLOD = _pVtxModel->pLOD(nLod);

	CUtlVector<colorTexel_t> &colorTexels = (*_outResults->m_ColorTexelsArrays.Tail());
	const int cTotalPixelCount = _lightmapResX * _lightmapResY;
	const bool alreadySized = ( colorTexels.Count() == cTotalPixelCount );
	colorTexels.EnsureCount(cTotalPixelCount);
	// A split prop accumulates every mesh of one material into the same image.
	// Resetting on each mesh would keep only the last mesh.
	if ( _bResetTexels || !alreadySized )
	{
		memset(colorTexels.Base(), 0, colorTexels.Count() * sizeof(colorTexel_t));
		for (int i = 0; i < colorTexels.Count(); ++i)
			colorTexels[i].m_fDistanceToTri = FLT_MAX;
	}

	mstudiomesh_t* pMesh = _pStudioModel->pMesh(_meshID);
	OptimizedModel::MeshHeader_t* pVtxMesh = pVtxLOD->pMesh(_meshID);
	const mstudio_meshvertexdata_t *vertData = pMesh->GetVertexData((void *)_pStudioHdr);
	Assert(vertData); // This can only return NULL on X360 for now

	for (int nGroup = 0; nGroup < pVtxMesh->numStripGroups; ++nGroup)
	{
		OptimizedModel::StripGroupHeader_t* pStripGroup = pVtxMesh->pStripGroup(nGroup);

		int nStrip;
		for (nStrip = 0; nStrip < pStripGroup->numStrips; nStrip++)
		{
			OptimizedModel::StripHeader_t *pStrip = pStripGroup->pStrip(nStrip);

			// If this hits, re-factor the code to iterate over triangles, and build the triangles
			// from the underlying structures.
			Assert((pStrip->flags & OptimizedModel::STRIP_IS_TRISTRIP) == 0);

			if (pStrip->flags & OptimizedModel::STRIP_IS_TRILIST)
			{
				for (int i = 0; i < pStrip->numIndices; i += 3)
				{
					int idx = pStrip->indexOffset + i;

					unsigned short i1 = *pStripGroup->pIndex(idx);
					unsigned short i2 = *pStripGroup->pIndex(idx + 1);
					unsigned short i3 = *pStripGroup->pIndex(idx + 2);

					int vertex1 = pStripGroup->pVertex(i1)->origMeshVertID;
					int vertex2 = pStripGroup->pVertex(i2)->origMeshVertID;
					int vertex3 = pStripGroup->pVertex(i3)->origMeshVertID;

					if ( _pUvPlan && _uvBin >= 0 &&
						!PropUvTriKept( *_pUvPlan, _uvBin, _uvBody, _uvModel, nLod, _meshID, nGroup, idx ) )
						continue;

					Vector modelPos[3] = {
						*vertData->Position(vertex1),
						*vertData->Position(vertex2),
						*vertData->Position(vertex3)
					};

					Vector modelNormal[3] = {
						*vertData->Normal(vertex1),
						*vertData->Normal(vertex2),
						*vertData->Normal(vertex3)
					};

					Vector worldPos[3];
					Vector worldNormal[3];

					VectorTransform(modelPos[0], _matPos, worldPos[0]);
					VectorTransform(modelPos[1], _matPos, worldPos[1]);
					VectorTransform(modelPos[2], _matPos, worldPos[2]);

					VectorTransform(modelNormal[0], _matNormal, worldNormal[0]);
					VectorTransform(modelNormal[1], _matNormal, worldNormal[1]);
					VectorTransform(modelNormal[2], _matNormal, worldNormal[2]);

					Vector2D texcoord[3] = { 
						*vertData->Texcoord(vertex1),
						*vertData->Texcoord(vertex2),
						*vertData->Texcoord(vertex3)
					};

					Rasterizer rasterizer(texcoord[0], texcoord[1], texcoord[2],
					                      _lightmapResX, _lightmapResY);

					for (auto it = rasterizer.begin(); it != rasterizer.end(); ++it)
					{
						size_t linearPos = rasterizer.GetLinearPos(it);
						Assert(linearPos < cTotalPixelCount);

						if ( colorTexels[linearPos].m_bValid )
						{
							continue;
						}						

						float ourDistancetoTri = ComputeBarycentricDistanceToTri( it->barycentric, texcoord );

						bool doWrite =  it->insideTriangle
							        || !colorTexels[linearPos].m_bPossiblyInteresting
									||  colorTexels[linearPos].m_fDistanceToTri > ourDistancetoTri;

						if (doWrite)
						{
							Vector itWorldPos = worldPos[0] * it->barycentric.x
											  + worldPos[1] * it->barycentric.y
											  + worldPos[2] * it->barycentric.z;

							Vector itWorldNormal = worldNormal[0] * it->barycentric.x
												 + worldNormal[1] * it->barycentric.y
												 + worldNormal[2] * it->barycentric.z;
							itWorldNormal.NormalizeInPlace();

							colorTexels[linearPos].m_WorldPosition = itWorldPos;
							colorTexels[linearPos].m_WorldNormal = itWorldNormal;
							colorTexels[linearPos].m_bValid = it->insideTriangle;
							colorTexels[linearPos].m_bPossiblyInteresting = true;
							colorTexels[linearPos].m_fDistanceToTri = ourDistancetoTri;

							const float sx = (float)_lightmapResX;
							const float sy = (float)_lightmapResY;
							const Vector2D d1( ( texcoord[1].x - texcoord[0].x ) * sx, ( texcoord[1].y - texcoord[0].y ) * sy );
							const Vector2D d2( ( texcoord[2].x - texcoord[0].x ) * sx, ( texcoord[2].y - texcoord[0].y ) * sy );
							const float det = d1.x * d2.y - d1.y * d2.x;
							Vector axisU( 0, 0, 0 ), axisV( 0, 0, 0 );
							float luxel = 0.0f;
							if ( fabsf( det ) > 1e-6f )
							{
								const float invDet = 1.0f / det;
								const Vector e1 = worldPos[1] - worldPos[0];
								const Vector e2 = worldPos[2] - worldPos[0];
								axisU = ( e1 * d2.y - e2 * d1.y ) * invDet;
								axisV = ( e2 * d1.x - e1 * d2.x ) * invDet;
								luxel = sqrtf( 0.5f * ( axisU.LengthSqr() + axisV.LengthSqr() ) );
								if ( luxel < 0.25f )
									luxel = 0.25f;
								if ( luxel > 48.0f )
									luxel = 48.0f;
							}
							colorTexels[linearPos].m_AxisU = axisU;
							colorTexels[linearPos].m_AxisV = axisV;
							colorTexels[linearPos].m_LuxelWorld = luxel;
						}
					}
				}
			}
		}
	}

	// Process neighbors to the valid region. Walk through the existing array, look for samples that
	// are not valid but are adjacent to valid samples. Works if we are only bilinearly sampling
	// on the other side.
	// First attempt: Just pretend the triangle was larger and cast a ray from this new world pos 
	// as above.
	int linearPos = 0;
	for ( int j = 0; j < _lightmapResY; ++j )
	{
		for (int i = 0; i < _lightmapResX; ++i )
		{
			bool shouldProcess = colorTexels[linearPos].m_bValid;
			// Are any of the eight neighbors valid??
			if ( colorTexels[linearPos].m_bPossiblyInteresting )
			{
				// Look at our neighborhood (3x3 centerd on us). 
				shouldProcess = shouldProcess
				             || colorTexels[ComputeLinearPos( i - 1, j - 1, _lightmapResX, _lightmapResY )].m_bValid  // TL
							 || colorTexels[ComputeLinearPos( i    , j - 1, _lightmapResX, _lightmapResY )].m_bValid  // T
							 || colorTexels[ComputeLinearPos( i + 1, j - 1, _lightmapResX, _lightmapResY )].m_bValid  // TR

							 || colorTexels[ComputeLinearPos( i - 1, j    , _lightmapResX, _lightmapResY )].m_bValid  // L
							 || colorTexels[ComputeLinearPos( i + 1, j    , _lightmapResX, _lightmapResY )].m_bValid  // R

							 || colorTexels[ComputeLinearPos( i - 1, j + 1, _lightmapResX, _lightmapResY )].m_bValid  // BL
							 || colorTexels[ComputeLinearPos( i    , j + 1, _lightmapResX, _lightmapResY )].m_bValid  // B
							 || colorTexels[ComputeLinearPos( i + 1, j + 1, _lightmapResX, _lightmapResY )].m_bValid; // BR
			}

			if (shouldProcess)
			{
				if ( _bComputeLighting )
				{
					if ( _pPropOrigin && _pPropModel &&
						 !BakeVolume_ShouldBakePropPoint( *_pPropOrigin, _pPropModel, colorTexels[linearPos].m_WorldPosition ) )
					{
						++linearPos;
						continue;
					}

					Vector directColor(0, 0, 0),
						   indirectColor(0, 0, 0);


					ComputeDirectLightingAtPoint( colorTexels[linearPos].m_WorldPosition, colorTexels[linearPos].m_WorldNormal, directColor, _iThread, _skipProp, _flags);

					if (numbounce >= 1) {
						ComputeIndirectLightingAtPoint( colorTexels[linearPos].m_WorldPosition, colorTexels[linearPos].m_WorldNormal, indirectColor, _iThread, true, (_flags & GATHERLFLAGS_IGNORE_NORMALS) != 0 );
					}

					VectorAdd(directColor, indirectColor, colorTexels[linearPos].m_Color);
				}
			}

			++linearPos;
		}
	}
}

// ------------------------------------------------------------------------------------------------
static int GetTexelCount(unsigned int _resX, unsigned int _resY, bool _mipmaps)
{
	// Because they are unsigned, this is a != check--but if we were to change to ints, this would be
	// the right assert (and it's no worse than != now). 
	Assert(_resX > 0 && _resY > 0);

	if (_mipmaps == false)
		return _resX * _resY;

	int retVal = 0;
	while (_resX > 1 || _resY > 1) 
	{
		retVal += _resX * _resY;
		_resX = max(1, _resX >> 1);
		_resY = max(1, _resY >> 1);
	}

	// Add in the 1x1 mipmap level, which wasn't hit above. This could be done in the initializer of 
	// retVal, but it's more obvious here. 
	retVal += 1;

	return retVal;
}

// ------------------------------------------------------------------------------------------------
static void FilterFineMipmap(unsigned int _resX, unsigned int _resY, const CUtlVector<colorTexel_t>& _srcTexels, CUtlVector<Vector>* _outLinear)
{
	Assert(_outLinear);
	// We can't filter in place, so go ahead and create a linear buffer here.
	CUtlVector<Vector> filterSrc;
	filterSrc.EnsureCount(_srcTexels.Count());

	for (int i = 0; i < _srcTexels.Count(); ++i)
	{
		ColorRGBExp32 rgbColor;
		VectorToColorRGBExp32(_srcTexels[i].m_Color, rgbColor);
		ConvertRGBExp32ToLinear( &rgbColor, &(filterSrc[i]) );
	}

	const int cRadius = 1;
	const float cOneOverDiameter = 1.0f / pow(2.0f * cRadius + 1.0f, 2.0f) ;
	// Filter here.
	for (int j = 0; j < _resY; ++j) 
	{
		for (int i = 0; i < _resX; ++i)
		{
			Vector value(0, 0, 0);
			int thisIndex = ComputeLinearPos(i, j, _resX, _resY);

			if (!_srcTexels[thisIndex].m_bValid)
			{
				(*_outLinear)[thisIndex] = filterSrc[thisIndex];
				continue;
			}

			// TODO: Check ASM for this, unroll by hand if needed.
			for ( int offsetJ = -cRadius; offsetJ <= cRadius; ++offsetJ )
			{
				for ( int offsetI = -cRadius; offsetI <= cRadius; ++offsetI )
				{
					int finalIndex = ComputeLinearPos( i + offsetI, j + offsetJ, _resX, _resY );
					if ( !_srcTexels[finalIndex].m_bValid ||
						!PropTexelChartsMatch( _srcTexels[thisIndex], _srcTexels[finalIndex] ) )
					{
						finalIndex = thisIndex;
					}
						
					value += filterSrc[finalIndex];
				}
			}

			(*_outLinear)[thisIndex] = value * cOneOverDiameter;
		}
	}
}

// ------------------------------------------------------------------------------------------------
static void BuildFineMipmap(unsigned int _resX, unsigned int _resY, bool _applyFilter, const CUtlVector<colorTexel_t>& _srcTexels, CUtlVector<RGB888_t>* _outTexelsRGB888, CUtlVector<Vector>* _outLinear)
{
	// At least one of these needs to be non-null, otherwise what are we doing here?
	Assert(_outTexelsRGB888 || _outLinear);
	Assert(!_applyFilter || _outLinear);
	Assert(_srcTexels.Count() == GetTexelCount(_resX, _resY, false));

	int texelCount = GetTexelCount(_resX, _resY, true);

	if (_outTexelsRGB888)
		(*_outTexelsRGB888).EnsureCount(texelCount);

	if (_outLinear)
		(*_outLinear).EnsureCount(GetTexelCount(_resX, _resY, false));

	// This code can take awhile, so minimize the branchiness of the inner-loop. 
	if (_applyFilter)
	{

		FilterFineMipmap(_resX, _resY, _srcTexels, _outLinear);

		if ( _outTexelsRGB888 )
		{
			for (int i = 0; i < _srcTexels.Count(); ++i) 
			{
				RGBA8888_t encodedColor;

				Vector linearColor = (*_outLinear)[i];

				ConvertLinearToRGBA8888( &linearColor, (unsigned char*)&encodedColor );
				(*_outTexelsRGB888)[i].r = encodedColor.r;
				(*_outTexelsRGB888)[i].g = encodedColor.g;
				(*_outTexelsRGB888)[i].b = encodedColor.b;
			}
		}
	}
	else
	{
		for (int i = 0; i < _srcTexels.Count(); ++i) 
		{
			ColorRGBExp32 rgbColor;
			RGBA8888_t encodedColor;
			VectorToColorRGBExp32(_srcTexels[i].m_Color, rgbColor);
			ConvertRGBExp32ToRGBA8888(&rgbColor, (unsigned char*)&encodedColor, (_outLinear ? (&(*_outLinear)[i]) : NULL) );
			// We drop alpha on the floor here, if this were to fire we'd need to consider using a different compressed format.
			Assert(encodedColor.a == 0xFF);

			if (_outTexelsRGB888)
			{
				(*_outTexelsRGB888)[i].r = encodedColor.r;
				(*_outTexelsRGB888)[i].g = encodedColor.g;
				(*_outTexelsRGB888)[i].b = encodedColor.b;
			}
		}
	}
}

// ------------------------------------------------------------------------------------------------
void ConvertLinearToRGB888( const Vector* pColorIn, RGB888_t* pColorOut )
{
	RGBA8888_t outPixel;
	ConvertLinearToRGBA8888( pColorIn, ( unsigned char* ) &outPixel );
	pColorOut->r = outPixel.r;
	pColorOut->g = outPixel.g;
	pColorOut->b = outPixel.b;
}

// ------------------------------------------------------------------------------------------------
static void FilterCoarserMipmaps( unsigned int _resX, unsigned int _resY, CUtlVector<Vector>* _scratchLinear, CUtlVector<RGB888_t>* _outTexelsRGB888 )
{
	Assert( _outTexelsRGB888 );

	int srcResX = _resX;
	int srcResY = _resY;
	int dstResX = max( 1, ( srcResX >> 1 ) );
	int dstResY = max( 1, ( srcResY >> 1 ) );
	int dstOffset = GetTexelCount( srcResX, srcResY, false );

	// Build mipmaps here, after being converted to linear space. 
	// TODO: Should do better filtering for downsampling. But this will work for now.
	while ( srcResX > 1 || srcResY > 1 )
	{
		int maxDestColumn = dstResX - 1;
		int maxDestRow = dstResY - 1;

		int maxColumn = srcResX - 1;
		int maxRow = srcResY - 1;
		for ( int j = 0; j < srcResY; j += 2 ) {
			for ( int i = 0; i < srcResX; i += 2 ) {
				int srcCol0 = i;
				int srcCol1 = i + 1;
				int srcRow0 = j;
				int srcRow1 = j + 1;
				srcCol1 = min( srcCol1, maxColumn );
				srcRow1 = min( srcRow1, maxRow );

				int dstCol = min( i >> 1, maxDestColumn );
				int dstRow = min( j >> 1, maxDestRow );

				const Vector& tl = ( *_scratchLinear )[srcCol0 + ( srcRow0 * srcResX )];
				const Vector& tr = ( *_scratchLinear )[srcCol1 + ( srcRow0 * srcResX )];
				const Vector& bl = ( *_scratchLinear )[srcCol0 + ( srcRow1 * srcResX )];
				const Vector& br = ( *_scratchLinear )[srcCol1 + ( srcRow1 * srcResX )];

				Vector sample = ( tl + tr + bl + br ) / 4.0f;

				ConvertLinearToRGB888( &sample, &( *_outTexelsRGB888 )[dstOffset + dstCol + dstRow * dstResX] );
				// Also overwrite the srcBuffer to filter the next loop. This is safe because we won't be reading this source value
				// again during this mipmap level.
				( *_scratchLinear )[dstCol + dstRow * dstResX] = sample;
			}
		}

		srcResX = dstResX;
		srcResY = dstResY;
		dstResX = max( 1, ( srcResX >> 1 ) );
		dstResY = max( 1, ( srcResY >> 1 ) );
		dstOffset += GetTexelCount( srcResX, srcResY, false );
	}
}

// ------------------------------------------------------------------------------------------------
static void ConvertToDestinationFormat(unsigned int _resX, unsigned int _resY, ImageFormat _destFmt, const CUtlVector<RGB888_t>& _scratchRBG888, CUtlMemory<byte>* _outTexture)
{
	const ImageFormat cSrcImageFormat = IMAGE_FORMAT_RGB888;

	// Converts from the scratch RGB888 buffer, which should be fully filled out to the output texture.
	int destMemoryUsage = ImageLoader::GetMemRequired(_resX, _resY, 1, _destFmt, true);
	(*_outTexture).EnsureCapacity(destMemoryUsage);

	int srcResX = _resX;
	int srcResY = _resY;
	int srcOffset = 0;
	int dstOffset = 0;

	// The usual case--that they'll be different.
	if (cSrcImageFormat != _destFmt)
	{
		while (srcResX > 1 || srcResY > 1)
		{
			// Convert this mipmap level.
			ImageLoader::ConvertImageFormat((unsigned char*)(&_scratchRBG888[srcOffset]), cSrcImageFormat, (*_outTexture).Base() + dstOffset, _destFmt, srcResX, srcResY);

			// Then update offsets for the next mipmap level.
			srcOffset += GetTexelCount(srcResX, srcResY, false);
			dstOffset += ImageLoader::GetMemRequired(srcResX, srcResY, 1, _destFmt, false);

			srcResX = max(1, (srcResX >> 1));
			srcResY = max(1, (srcResY >> 1));
		}

		// Do the 1x1 level also.
		ImageLoader::ConvertImageFormat((unsigned char*)_scratchRBG888.Base() + srcOffset, cSrcImageFormat, (*_outTexture).Base() + dstOffset, _destFmt, srcResX, srcResY);
	} else {
		// But sometimes (particularly for debugging) they will be the same.
		Q_memcpy( (*_outTexture).Base(), _scratchRBG888.Base(), destMemoryUsage );
	}
}

// ------------------------------------------------------------------------------------------------
static void ConvertTexelDataToTexture(unsigned int _resX, unsigned int _resY, ImageFormat _destFmt, const CUtlVector<colorTexel_t>& _srcTexels, CUtlMemory<byte>* _outTexture)
{
	if ( !_outTexture || _resX == 0 || _resY == 0 )
		return;
	if ( _srcTexels.Count() != (int)( _resX * _resY ) )
	{
		Warning( "static prop lightmap size mismatch (%d texels vs %ux%u), skipping.\n",
			_srcTexels.Count(), _resX, _resY );
		return;
	}

	CUtlVector<RGB888_t> scratchRGB888;
	CUtlVector<Vector> scratchLinear;

	BuildFineMipmap(_resX, _resY, true, _srcTexels, &scratchRGB888, &scratchLinear);
	FilterCoarserMipmaps(_resX, _resY, &scratchLinear, &scratchRGB888 );
	ConvertToDestinationFormat(_resX, _resY, _destFmt, scratchRGB888, _outTexture);
}

// ------------------------------------------------------------------------------------------------
static void DumpLightmapLinear( const char* _dstFilename, const CUtlVector<colorTexel_t>& _srcTexels, int _width, int _height )
{
	CUtlVector< Vector > linearFloats;
	CUtlVector< BGR888_t > linearBuffer;
	BuildFineMipmap( _width, _height, true, _srcTexels, NULL, &linearFloats );
	linearBuffer.SetCount( linearFloats.Count() );

	for ( int i = 0; i < linearFloats.Count(); ++i ) {
		linearBuffer[i].b = RoundFloatToByte(linearFloats[i].z * 255.0f);
		linearBuffer[i].g = RoundFloatToByte(linearFloats[i].y * 255.0f);
		linearBuffer[i].r = RoundFloatToByte(linearFloats[i].x * 255.0f);
	}
	
	TGAWriter::WriteTGAFile( _dstFilename, _width, _height, IMAGE_FORMAT_BGR888, (uint8*)(linearBuffer.Base()), _width * ImageLoader::SizeInBytes(IMAGE_FORMAT_BGR888) );
}
