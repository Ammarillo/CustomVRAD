//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
// $Workfile:     $
// $Date:         $
//
//-----------------------------------------------------------------------------
// $Log: $
//
// $NoKeywords: $
//=============================================================================//

#define	USED

#include <windows.h>
#include "cmdlib.h"
#define NO_THREAD_NAMES
#include "threads.h"
#include "pacifier.h"

class CRunThreadsData
{
public:
	int m_iThread;
	void *m_pUserData;
	RunThreadsFn m_Fn;
};

CRunThreadsData g_RunThreadsData[MAX_TOOL_THREADS];


volatile LONG	dispatch;
int		workcount;
qboolean		pacifier;

qboolean	threaded;
bool g_bLowPriorityThreads = false;

HANDLE g_ThreadHandles[MAX_TOOL_THREADS];
static CRITICAL_SECTION s_PacifierCrit;
static bool s_bPacifierCritInit = false;

static void EnsurePacifierCrit()
{
	if ( !s_bPacifierCritInit )
	{
		InitializeCriticalSection( &s_PacifierCrit );
		s_bPacifierCritInit = true;
	}
}



/*
=============
GetThreadWork

Lock-free work steal via InterlockedIncrement. Pacifier updates use a separate
short critical section so Msg() does not serialize the whole work queue.
=============
*/
int	GetThreadWork (void)
{
	LONG r = InterlockedIncrement( &dispatch ) - 1;
	if ( r >= workcount )
		return -1;

	if ( pacifier && workcount > 0 )
	{
		EnsurePacifierCrit();
		EnterCriticalSection( &s_PacifierCrit );
		UpdatePacifier( (float)r / (float)workcount );
		LeaveCriticalSection( &s_PacifierCrit );
	}

	return (int)r;
}


ThreadWorkerFn workfunction;

void ThreadWorkerFunction( int iThread, void *pUserData )
{
	int		work;

	while (1)
	{
		work = GetThreadWork ();
		if (work == -1)
			break;
		 
		workfunction( iThread, work );
	}
}

void RunThreadsOnIndividual (int workcnt, qboolean showpacifier, ThreadWorkerFn func)
{
	if (numthreads == -1)
		ThreadSetDefault ();
	
	workfunction = func;
	RunThreadsOn (workcnt, showpacifier, ThreadWorkerFunction);
}


/*
===================================================================

WIN32

===================================================================
*/

int		numthreads = -1;
CRITICAL_SECTION		crit;
static int enter;


class CCritInit
{
public:
	CCritInit()
	{
		InitializeCriticalSection (&crit);
	}
} g_CritInit;



void SetLowPriority()
{
	SetPriorityClass( GetCurrentProcess(), IDLE_PRIORITY_CLASS );
}


static int DetectHardwareThreadCount()
{
	// Prefer group-aware count so machines with >64 logical processors report correctly.
	HMODULE hKernel = GetModuleHandleA( "kernel32.dll" );
	if ( hKernel )
	{
		typedef DWORD (WINAPI *GetActiveProcessorCountFn)( WORD );
		GetActiveProcessorCountFn pGetActiveProcessorCount =
			(GetActiveProcessorCountFn)GetProcAddress( hKernel, "GetActiveProcessorCount" );
		if ( pGetActiveProcessorCount )
		{
			DWORD n = pGetActiveProcessorCount( ALL_PROCESSOR_GROUPS );
			if ( n >= 1 )
				return (int)n;
		}
	}

	SYSTEM_INFO info;
	GetSystemInfo( &info );
	return (int)info.dwNumberOfProcessors;
}


void ThreadSetDefault (void)
{
	if (numthreads == -1)	// not set manually
	{
		numthreads = DetectHardwareThreadCount();
		if ( numthreads < 1 )
			numthreads = 1;
		if ( numthreads > MAX_TOOL_THREADS )
			numthreads = MAX_TOOL_THREADS;
	}
	else
	{
		if ( numthreads < 1 )
			numthreads = 1;
		if ( numthreads > MAX_TOOL_THREADS )
		{
			Msg( "Clamping -threads from %d to %d\n", numthreads, MAX_TOOL_THREADS );
			numthreads = MAX_TOOL_THREADS;
		}
	}

	Msg ("%i threads\n", numthreads);
}


void ThreadLock (void)
{
	if (!threaded)
		return;
	EnterCriticalSection (&crit);
	if (enter)
		Error ("Recursive ThreadLock\n");
	enter = 1;
}

void ThreadUnlock (void)
{
	if (!threaded)
		return;
	if (!enter)
		Error ("ThreadUnlock without lock\n");
	enter = 0;
	LeaveCriticalSection (&crit);
}


// This runs in the thread and dispatches a RunThreadsFn call.
DWORD WINAPI InternalRunThreadsFn( LPVOID pParameter )
{
	CRunThreadsData *pData = (CRunThreadsData*)pParameter;
	pData->m_Fn( pData->m_iThread, pData->m_pUserData );
	return 0;
}


void RunThreads_Start( RunThreadsFn fn, void *pUserData, ERunThreadsPriority ePriority )
{
	Assert( numthreads > 0 );
	threaded = true;

	if ( numthreads > MAX_TOOL_THREADS )
		numthreads = MAX_TOOL_THREADS;

	for ( int i=0; i < numthreads ;i++ )
	{
		g_RunThreadsData[i].m_iThread = i;
		g_RunThreadsData[i].m_pUserData = pUserData;
		g_RunThreadsData[i].m_Fn = fn;

		DWORD dwDummy;
		g_ThreadHandles[i] = CreateThread(
		   NULL,	// LPSECURITY_ATTRIBUTES lpsa,
		   0,		// DWORD cbStack,
		   InternalRunThreadsFn,	// LPTHREAD_START_ROUTINE lpStartAddr,
		   &g_RunThreadsData[i],	// LPVOID lpvThreadParm,
		   0,			// DWORD fdwCreate,
		   &dwDummy );

		if ( ePriority == k_eRunThreadsPriority_UseGlobalState )
		{
			if( g_bLowPriorityThreads )
				SetThreadPriority( g_ThreadHandles[i], THREAD_PRIORITY_LOWEST );
		}
		else if ( ePriority == k_eRunThreadsPriority_Idle )
		{
			SetThreadPriority( g_ThreadHandles[i], THREAD_PRIORITY_IDLE );
		}
	}
}


static void WaitForAllThreadHandles( int nThreads )
{
	// WaitForMultipleObjects is limited to MAXIMUM_WAIT_OBJECTS (64) handles.
	int remaining = nThreads;
	int offset = 0;
	while ( remaining > 0 )
	{
		int batch = remaining;
		if ( batch > MAXIMUM_WAIT_OBJECTS )
			batch = MAXIMUM_WAIT_OBJECTS;

		DWORD result = WaitForMultipleObjects( batch, &g_ThreadHandles[offset], TRUE, INFINITE );
		if ( result == WAIT_FAILED )
			Error( "WaitForMultipleObjects failed (%u)\n", GetLastError() );

		offset += batch;
		remaining -= batch;
	}
}


void RunThreads_End()
{
	WaitForAllThreadHandles( numthreads );
	for ( int i=0; i < numthreads; i++ )
		CloseHandle( g_ThreadHandles[i] );

	threaded = false;
}
	

/*
=============
RunThreadsOn
=============
*/
void RunThreadsOn( int workcnt, qboolean showpacifier, RunThreadsFn fn, void *pUserData )
{
	int		start, end;

	start = Plat_FloatTime();
	InterlockedExchange( &dispatch, 0 );
	workcount = workcnt;
	StartPacifier("");
	pacifier = showpacifier;
	EnsurePacifierCrit();

#ifdef _PROFILE
	threaded = false;
	(*func)( 0 );
	return;
#endif

	
	RunThreads_Start( fn, pUserData );
	RunThreads_End();


	end = Plat_FloatTime();
	if (pacifier)
	{
		EndPacifier(false);
		printf (" (%i)\n", end-start);
	}
}

