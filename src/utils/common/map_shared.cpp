//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
// $NoKeywords: $
//=============================================================================//

#include "map_shared.h"
#include "bsplib.h"
#include "cmdlib.h"


CMapError g_MapError;
int g_nMapFileVersion;


//-----------------------------------------------------------------------------
// Purpose: 
// Input  : *szKey - 
//			*szValue - 
//			*pLoadEntity - 
// Output : ChunkFileResult_t
//-----------------------------------------------------------------------------
ChunkFileResult_t LoadEntityKeyCallback(const char *szKey, const char *szValue, LoadEntity_t *pLoadEntity)
{
	if (!stricmp(szKey, "classname"))
	{
		if (!stricmp(szValue, "func_detail"))
		{
			pLoadEntity->nBaseContents = CONTENTS_DETAIL;
		}
		else if (!stricmp(szValue, "func_ladder"))
		{
			pLoadEntity->nBaseContents = CONTENTS_LADDER;
		}
		else if (!stricmp(szValue, "func_water"))
		{
			pLoadEntity->nBaseContents = CONTENTS_WATER;
		}
	}
	else if (!stricmp(szKey, "id"))
	{
		// UNDONE: flag entity errors by ID instead of index
		//g_MapError.EntityState( atoi( szValue ) );
		// rename this field since DME code uses this name
		SetKeyValue( pLoadEntity->pEntity, "hammerid", szValue );
		return(ChunkFile_Ok);
	}
	else if( !stricmp( szKey, "mapversion" ) )
	{
		// .vmf map revision number
		g_MapRevision = atoi( szValue );
		SetKeyValue( pLoadEntity->pEntity, szKey, szValue );
		return ( ChunkFile_Ok );
	}

	SetKeyValue( pLoadEntity->pEntity, szKey, szValue );

	return(ChunkFile_Ok);
}


static ChunkFileResult_t LoadEntityCallback( CChunkFile *pFile, int nParam )
{
	if (num_entities == MAX_MAP_ENTITIES)
	{
		// Exits.
		g_MapError.ReportError ("num_entities == MAX_MAP_ENTITIES");
	}

	entity_t *mapent = &entities[num_entities];
	num_entities++;
	memset(mapent, 0, sizeof(*mapent));
	mapent->numbrushes = 0;

	LoadEntity_t LoadEntity;
	LoadEntity.pEntity = mapent;

	// No default flags/contents
	LoadEntity.nBaseFlags = 0;
	LoadEntity.nBaseContents = 0;

	//
	// Read the entity chunk.
	//
	ChunkFileResult_t eResult = pFile->ReadChunk((KeyHandler_t)LoadEntityKeyCallback, &LoadEntity);

	return eResult;
}


bool LoadEntsFromMapFile( char const *pFilename )
{
	//
	// Dummy this up for the texture handling. This can be removed when old .MAP file
	// support is removed.
	//
	g_nMapFileVersion = 400;

	//
	// Open the file.
	//
	CChunkFile File;
	ChunkFileResult_t eResult = File.Open( pFilename, ChunkFile_Read );

	if(eResult == ChunkFile_Ok)
	{
		num_entities = 0;

		//
		// Set up handlers for the subchunks that we are interested in.
		//
		CChunkHandlerMap Handlers;
		Handlers.AddHandler("entity", (ChunkHandler_t)LoadEntityCallback, 0);

		File.PushHandlers(&Handlers);

		//
		// Read the sub-chunks. We ignore keys in the root of the file.
		//
		while (eResult == ChunkFile_Ok)
		{
			eResult = File.ReadChunk();
		}

		File.PopHandlers();
		return true;
	}
	else
	{
		Error("Error in LoadEntsFromMapFile (in-memory file): %s.\n", File.GetErrorText(eResult));
		return false;
	}
}

static void FreeEntityEpairs( entity_t *pEnt )
{
	if ( !pEnt )
		return;
	epair_t *ep = pEnt->epairs;
	while ( ep )
	{
		epair_t *pNext = ep->next;
		free( ep->key );
		free( ep->value );
		free( ep );
		ep = pNext;
	}
	pEnt->epairs = NULL;
}

static float LightEnvBrightness( entity_t *pEnt )
{
	// "_light" "r g b intensity"
	char *p = ValueForKey( pEnt, "_light" );
	if ( !p || !p[0] )
		return 0.0f;
	float r = 0, g = 0, b = 0, i = 0;
	sscanf( p, "%f %f %f %f", &r, &g, &b, &i );
	return ( 0.2126f * r + 0.7152f * g + 0.0722f * b ) * i;
}

struct LoadLightEnv_t
{
	entity_t	bestVisible;
	entity_t	bestHidden;
	bool		bHaveVisible;
	bool		bHaveHidden;
	float		flBestVisibleBright;
	float		flBestHiddenBright;
};

static void ConsiderLightEnvCandidate( LoadLightEnv_t *pCtx, entity_t *pScratch, bool bHidden )
{
	const char *pszClass = ValueForKey( pScratch, "classname" );
	if ( !pszClass || Q_stricmp( pszClass, "light_environment" ) )
	{
		FreeEntityEpairs( pScratch );
		return;
	}

	const float bright = LightEnvBrightness( pScratch );
	if ( !bHidden )
	{
		if ( !pCtx->bHaveVisible || bright >= pCtx->flBestVisibleBright )
		{
			if ( pCtx->bHaveVisible )
				FreeEntityEpairs( &pCtx->bestVisible );
			pCtx->bestVisible = *pScratch;
			pScratch->epairs = NULL;
			pCtx->bHaveVisible = true;
			pCtx->flBestVisibleBright = bright;
		}
		else
		{
			FreeEntityEpairs( pScratch );
		}
	}
	else
	{
		if ( !pCtx->bHaveHidden || bright >= pCtx->flBestHiddenBright )
		{
			if ( pCtx->bHaveHidden )
				FreeEntityEpairs( &pCtx->bestHidden );
			pCtx->bestHidden = *pScratch;
			pScratch->epairs = NULL;
			pCtx->bHaveHidden = true;
			pCtx->flBestHiddenBright = bright;
		}
		else
		{
			FreeEntityEpairs( pScratch );
		}
	}
}

static ChunkFileResult_t LoadLightEnvEntityCallback( CChunkFile *pFile, LoadLightEnv_t *pCtx );

// Depth: 0 = top-level entity, >0 = under hidden (Hammer visgroup-hidden ents).
static int g_nLightEnvHiddenDepth = 0;

static ChunkFileResult_t LoadLightEnvEntityCallback( CChunkFile *pFile, LoadLightEnv_t *pCtx )
{
	entity_t scratch;
	memset( &scratch, 0, sizeof( scratch ) );

	LoadEntity_t load;
	load.pEntity = &scratch;
	load.nBaseFlags = 0;
	load.nBaseContents = 0;

	ChunkFileResult_t eResult = pFile->ReadChunk( (KeyHandler_t)LoadEntityKeyCallback, &load );
	if ( eResult != ChunkFile_Ok && eResult != ChunkFile_EOF )
	{
		FreeEntityEpairs( &scratch );
		return eResult;
	}

	ConsiderLightEnvCandidate( pCtx, &scratch, g_nLightEnvHiddenDepth > 0 );
	return ChunkFile_Ok;
}

// Nested Hammer "hidden { }" (visgroups / cordons). Do not PushHandlers per nest
// (overflows MAX_INDENT_DEPTH) and do not loop ReadChunk — one ReadChunk consumes
// this block; looping past EndOfChunk desyncs the parser and crashes on large .vmf.
static ChunkFileResult_t LoadLightEnvHiddenCallbackDepth( CChunkFile *pFile, LoadLightEnv_t *pCtx )
{
	++g_nLightEnvHiddenDepth;
	const ChunkFileResult_t eResult = pFile->ReadChunk();
	--g_nLightEnvHiddenDepth;
	return eResult;
}

bool LoadLightEnvironmentEntityFromVmf( const char *pFilename, entity_t *pOut )
{
	if ( !pFilename || !pFilename[0] || !pOut )
		return false;

	g_nMapFileVersion = 400;
	g_nLightEnvHiddenDepth = 0;

	CChunkFile File;
	ChunkFileResult_t eResult = File.Open( pFilename, ChunkFile_Read );
	if ( eResult != ChunkFile_Ok )
		return false;

	LoadLightEnv_t ctx;
	memset( &ctx, 0, sizeof( ctx ) );

	CChunkHandlerMap Handlers;
	Handlers.AddHandler( "entity", (ChunkHandler_t)LoadLightEnvEntityCallback, &ctx );
	Handlers.AddHandler( "hidden", (ChunkHandler_t)LoadLightEnvHiddenCallbackDepth, &ctx );
	File.PushHandlers( &Handlers );

	while ( eResult == ChunkFile_Ok )
		eResult = File.ReadChunk();

	File.PopHandlers();
	g_nLightEnvHiddenDepth = 0;

	memset( pOut, 0, sizeof( *pOut ) );
	if ( ctx.bHaveVisible )
	{
		*pOut = ctx.bestVisible;
		if ( ctx.bHaveHidden )
			FreeEntityEpairs( &ctx.bestHidden );
		return true;
	}
	if ( ctx.bHaveHidden )
	{
		*pOut = ctx.bestHidden;
		return true;
	}
	return false;
}


