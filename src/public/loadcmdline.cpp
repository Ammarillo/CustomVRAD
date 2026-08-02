//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: loads additional command line options from a config file
//
// $NoKeywords: $
//=============================================================================//

#include "loadcmdline.h"
#include "KeyValues.h"
#include "tier1/strtools.h"
#include "FileSystem_Tools.h"
#include "tier1/utlstring.h"
#include "tier1/utlvector.h"
#include "tier0/dbg.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <stdio.h>
#include <string.h>

// So we know whether or not we own argv's memory
static bool s_bOwnArgv = false;

static void FreeOwnedArgv( int argc, char **argv )
{
	if ( !s_bOwnArgv || !argv )
		return;
	for ( int i = 0; i < argc; ++i )
		delete[] argv[i];
	delete[] argv;
	s_bOwnArgv = false;
}

static char *DupArg( const char *s )
{
	return V_strdup( s ? s : "" );
}

static void AdoptArgv( int &argc, char **&argv, CUtlVector<char *> &args )
{
	const int n = args.Count();
	char **out = new char *[n];
	for ( int i = 0; i < n; ++i )
		out[i] = args[i];

	FreeOwnedArgv( argc, argv );
	argc = n;
	argv = out;
	s_bOwnArgv = true;
}

//-----------------------------------------------------------------------------
// Purpose: Parses arguments and inserts them before the last argv entry (map path).
//-----------------------------------------------------------------------------
static void AddArguments( int &argc, char **&argv, const char *str )
{
	if ( !str || !str[0] || argc < 1 )
		return;

	CUtlVector<char *> tokens;
	char *argList = V_strdup( str );
	char *token = strtok( argList, " \t\r\n" );
	while ( token )
	{
		tokens.AddToTail( DupArg( token ) );
		token = strtok( NULL, " \t\r\n" );
	}
	delete[] argList;

	if ( tokens.Count() <= 0 )
		return;

	CUtlVector<char *> args;
	args.EnsureCapacity( argc + tokens.Count() );

	// Keep everything except the last original arg, then insert, then map path.
	const int keepTail = ( argc >= 1 ) ? 1 : 0;
	const int headCount = argc - keepTail;
	for ( int i = 0; i < headCount; ++i )
		args.AddToTail( DupArg( argv[i] ) );
	for ( int i = 0; i < tokens.Count(); ++i )
		args.AddToTail( tokens[i] );
	if ( keepTail )
		args.AddToTail( DupArg( argv[argc - 1] ) );

	AdoptArgv( argc, argv, args );
}

static bool FileExistsLocal( const char *path )
{
	FILE *fp = fopen( path, "rb" );
	if ( !fp )
		return false;
	fclose( fp );
	return true;
}

static bool ResolveConfigPath( const char *inPath, char *outPath, int outSize )
{
	if ( !inPath || !inPath[0] || !outPath || outSize <= 0 )
		return false;

	static const char *exts[] = { "", ".cfg", ".txt", NULL };

	// 1) As given (+ optional extension)
	for ( int e = 0; exts[e] != NULL; ++e )
	{
		Q_snprintf( outPath, outSize, "%s%s", inPath, exts[e] );
		if ( FileExistsLocal( outPath ) )
			return true;
	}

#ifdef _WIN32
	// 2) Next to vrad.exe / vrad_dll.dll
	char modulePath[MAX_PATH];
	modulePath[0] = '\0';
	if ( GetModuleFileNameA( NULL, modulePath, MAX_PATH ) > 0 )
	{
		char dir[MAX_PATH];
		Q_strncpy( dir, modulePath, sizeof( dir ) );
		Q_StripFilename( dir );

		for ( int e = 0; exts[e] != NULL; ++e )
		{
			Q_snprintf( outPath, outSize, "%s\\%s%s", dir, inPath, exts[e] );
			if ( FileExistsLocal( outPath ) )
				return true;
			Q_snprintf( outPath, outSize, "%s\\configs\\%s%s", dir, inPath, exts[e] );
			if ( FileExistsLocal( outPath ) )
				return true;
		}
	}
#endif

	Q_strncpy( outPath, inPath, outSize );
	return false;
}

static void StripInlineComment( char *line )
{
	bool inQuote = false;
	for ( char *p = line; *p; ++p )
	{
		if ( *p == '"' )
			inQuote = !inQuote;
		if ( inQuote )
			continue;
		if ( *p == '#' || ( p[0] == '/' && p[1] == '/' ) )
		{
			*p = '\0';
			return;
		}
	}
}

static void TokenizeConfigLine( const char *line, CUtlVector<char *> &out )
{
	const char *p = line;
	while ( *p )
	{
		while ( *p == ' ' || *p == '\t' || *p == '\r' )
			++p;
		if ( !*p )
			break;

		char token[1024];
		int n = 0;
		if ( *p == '"' )
		{
			++p;
			while ( *p && *p != '"' && n < (int)sizeof( token ) - 1 )
				token[n++] = *p++;
			if ( *p == '"' )
				++p;
		}
		else
		{
			while ( *p && *p != ' ' && *p != '\t' && *p != '\r' && n < (int)sizeof( token ) - 1 )
				token[n++] = *p++;
		}
		token[n] = '\0';
		if ( n > 0 )
			out.AddToTail( DupArg( token ) );
	}
}

static bool LoadConfigTokens( const char *path, CUtlVector<char *> &tokens )
{
	FILE *fp = fopen( path, "rb" );
	if ( !fp )
		return false;

	char line[2048];
	while ( fgets( line, sizeof( line ), fp ) )
	{
		StripInlineComment( line );
		// trim trailing whitespace / newline
		int len = (int)strlen( line );
		while ( len > 0 && ( line[len - 1] == '\n' || line[len - 1] == '\r' ||
							 line[len - 1] == ' ' || line[len - 1] == '\t' ) )
			line[--len] = '\0';

		// skip blank / full-line comments already stripped
		const char *p = line;
		while ( *p == ' ' || *p == '\t' )
			++p;
		if ( !*p )
			continue;

		TokenizeConfigLine( p, tokens );
	}

	fclose( fp );
	return true;
}

//-----------------------------------------------------------------------------
// Purpose: Expand -config / -cfg <path> into argv. Runs before filesystem init.
//-----------------------------------------------------------------------------
void ExpandConfigArgs( int &argc, char **&argv )
{
	if ( argc < 2 || !argv )
		return;

	// Collect first (allow multiple -config, expand left-to-right).
	bool bAny = false;
	for ( ;; )
	{
		int cfgIdx = -1;
		for ( int i = 1; i < argc; ++i )
		{
			if ( !argv[i] )
				continue;
			if ( !Q_stricmp( argv[i], "-config" ) || !Q_stricmp( argv[i], "-cfg" ) )
			{
				cfgIdx = i;
				break;
			}
		}
		if ( cfgIdx < 0 )
			break;
		if ( cfgIdx + 1 >= argc )
		{
			Warning( "ExpandConfigArgs: -config requires a file path.\n" );
			break;
		}

		const char *inPath = argv[cfgIdx + 1];
		char resolved[1024];
		if ( !ResolveConfigPath( inPath, resolved, sizeof( resolved ) ) )
		{
			Warning( "ExpandConfigArgs: config not found: \"%s\"\n", inPath );
			// Drop the bad -config pair so ParseCommandLine doesn't choke on it.
			CUtlVector<char *> stripped;
			for ( int i = 0; i < argc; ++i )
			{
				if ( i == cfgIdx || i == cfgIdx + 1 )
					continue;
				stripped.AddToTail( DupArg( argv[i] ) );
			}
			AdoptArgv( argc, argv, stripped );
			continue;
		}

		CUtlVector<char *> cfgTokens;
		if ( !LoadConfigTokens( resolved, cfgTokens ) )
		{
			Warning( "ExpandConfigArgs: failed to read \"%s\"\n", resolved );
			CUtlVector<char *> stripped;
			for ( int i = 0; i < argc; ++i )
			{
				if ( i == cfgIdx || i == cfgIdx + 1 )
					continue;
				stripped.AddToTail( DupArg( argv[i] ) );
			}
			AdoptArgv( argc, argv, stripped );
			continue;
		}

		Msg( "[config] loaded \"%s\" (%d args)\n", resolved, cfgTokens.Count() );

		CUtlVector<char *> args;
		args.EnsureCapacity( argc + cfgTokens.Count() );
		for ( int i = 0; i < cfgIdx; ++i )
			args.AddToTail( DupArg( argv[i] ) );
		for ( int i = 0; i < cfgTokens.Count(); ++i )
			args.AddToTail( cfgTokens[i] );
		for ( int i = cfgIdx + 2; i < argc; ++i )
			args.AddToTail( DupArg( argv[i] ) );

		AdoptArgv( argc, argv, args );
		bAny = true;
	}

	if ( bAny )
	{
		Msg( "[config] effective command line:\n" );
		for ( int i = 1; i < argc; ++i )
			Msg( "  %s\n", argv[i] );
	}
}

//-----------------------------------------------------------------------------
// Purpose: Loads additional commandline arguments from a config file for an app.
//			Filesystem must be initialized before calling this function.
//-----------------------------------------------------------------------------
void LoadCmdLineFromFile( int &argc, char **&argv, const char *keyname, const char *appname )
{
	assert( g_pFileSystem );
	if( !g_pFileSystem )
		return;

	// Load the cfg file, and find the keyname
	KeyValues *kv = new KeyValues( "CommandLine" );

	char filename[512];
	Q_snprintf( filename, sizeof( filename ), "%s/cfg/commandline.cfg", gamedir );

	if ( kv->LoadFromFile( g_pFileSystem, filename ) )
	{
		// Load the commandline arguments for this app
		KeyValues  *appKey	= kv->FindKey( keyname );
		if( appKey )
		{
			const char *str	= appKey->GetString( appname );
			Msg( "Command Line found: %s\n", str );

			AddArguments( argc, argv, str );
		}
	}

	kv->deleteThis();
}

//-----------------------------------------------------------------------------
// Purpose: Cleans up any memory allocated for the new argv.  Pass in the app's
// argc and argv - this is safe even if no extra arguments were loaded.
//-----------------------------------------------------------------------------
void DeleteCmdLine( int argc, char **argv )
{
	FreeOwnedArgv( argc, argv );
}
