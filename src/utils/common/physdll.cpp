//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
// $NoKeywords: $
//
//=============================================================================//
#include <stdio.h>
#include "physdll.h"
#include "filesystem_tools.h"
#include "tier1/strtools.h"
#include "tier1/interface.h"
#ifdef _WIN32
#include <windows.h>
#endif

static CSysModule *pPhysicsModule = NULL;

static CSysModule *TryLoadPhysicsModule( const char *pName )
{
	if ( !pName || !pName[0] )
		return NULL;

	CSysModule *pMod = NULL;
	if ( g_pFullFileSystem )
		pMod = g_pFullFileSystem->LoadModule( pName );
	if ( !pMod )
		pMod = Sys_LoadModule( pName );
	return pMod;
}

static CSysModule *TryLoadPhysicsBesideExe( const char *pFileName )
{
#ifdef _WIN32
	char exePath[MAX_PATH];
	if ( !GetModuleFileNameA( NULL, exePath, sizeof( exePath ) ) )
		return NULL;

	char exeDir[MAX_PATH];
	Q_strncpy( exeDir, exePath, sizeof( exeDir ) );
	Q_StripFilename( exeDir );

	char full[MAX_PATH];
	Q_snprintf( full, sizeof( full ), "%s\\%s", exeDir, pFileName );
	CSysModule *pMod = Sys_LoadModule( full );
	if ( pMod )
		return pMod;

	// PLATFORM_BIN_DIR layout: <exedir>\bin\x64\<dll>
	Q_snprintf( full, sizeof( full ), "%s\\bin\\x64\\%s", exeDir, pFileName );
	return Sys_LoadModule( full );
#else
	return NULL;
#endif
}

static bool LoadPhysicsModuleInternal( const char *pPreferredPath )
{
	if ( pPhysicsModule )
		return true;

	// Prefer the tools stub when present — GMod's full vphysics.dll does not
	// match the SDK2013 tier0 this VRAD is linked against.
	const char *candidates[] =
	{
		pPreferredPath,
		"vphysics_stub.dll",
		"vphysics.dll",
		"VPHYSICS.DLL",
		NULL
	};

	for ( int i = 0; candidates[i]; ++i )
	{
		if ( !candidates[i][0] )
			continue;

		pPhysicsModule = TryLoadPhysicsBesideExe( candidates[i] );
		if ( pPhysicsModule )
			return true;

		pPhysicsModule = TryLoadPhysicsModule( candidates[i] );
		if ( pPhysicsModule )
			return true;
	}

	return false;
}

CreateInterfaceFn GetPhysicsFactory( void )
{
	if ( !pPhysicsModule )
	{
		if ( !LoadPhysicsModuleInternal( NULL ) )
			return NULL;
	}

	return Sys_GetFactory( pPhysicsModule );
}

void PhysicsDLLPath( const char *pPathname )
{
	if ( !pPhysicsModule )
	{
		LoadPhysicsModuleInternal( pPathname );
	}
}
