//========= Copyright Valve Corporation, All rights reserved. ============//
//
// light_portal — linked-space radiosity portals (CustomVRAD).
//
//=============================================================================//

#include "vrad.h"
#include "portal.h"
#include "mathlib/vector.h"

extern int total_transfer;
extern int max_transfer;

#define PORTAL_MAX			64
#define PORTAL_NEAR_DIST	512.0f
#define PORTAL_TRANSFER_EPS	1e-6f

struct PortalBrush_t
{
	char	targetname[64];
	char	target[64];
	Vector	center;
	Vector	forward;
	Vector	right;
	Vector	up;
	float	halfW;
	float	halfH;
	int		link;		// index of linked portal, or -1
	bool	bMutual;	// both sides target each other
};

static PortalBrush_t s_Portals[PORTAL_MAX];
static int s_nPortals = 0;

void Portal_Clear()
{
	s_nPortals = 0;
}

bool Portal_HasPairs()
{
	for ( int i = 0; i < s_nPortals; ++i )
	{
		if ( s_Portals[i].link >= 0 )
			return true;
	}
	return false;
}

int Portal_PairCount()
{
	int n = 0;
	for ( int i = 0; i < s_nPortals; ++i )
	{
		if ( s_Portals[i].link > i )
			++n;
	}
	return n;
}

void Portal_ParseEntity( entity_t *e )
{
	char *pModel = ValueForKey( e, "model" );
	if ( !pModel || pModel[0] != '*' )
	{
		Warning( "WARNING: light_portal without brush model ignored\n" );
		return;
	}
	int modelIndex = atoi( pModel + 1 );
	if ( modelIndex <= 0 || modelIndex >= nummodels )
	{
		Warning( "WARNING: light_portal has invalid model \"%s\"\n", pModel );
		return;
	}
	if ( s_nPortals >= PORTAL_MAX )
	{
		Warning( "WARNING: too many light_portal entities (max %d)\n", PORTAL_MAX );
		return;
	}

	dmodel_t *pm = &dmodels[modelIndex];
	PortalBrush_t &p = s_Portals[s_nPortals];
	memset( &p, 0, sizeof( p ) );
	p.link = -1;

	p.center = ( pm->mins + pm->maxs ) * 0.5f;
	Vector size = pm->maxs - pm->mins;

	Vector angles;
	GetVectorForKey( e, "angles", angles );
	float pitch = FloatForKey( e, "pitch" );
	QAngle q( angles.x, angles.y, angles.z );
	if ( pitch != 0.0f )
		q.x = pitch;
	AngleVectors( q, &p.forward, &p.right, &p.up );

	// Aperture: extents projected onto portal plane axes.
	p.halfW = 0.5f * ( fabsf( DotProduct( size, p.right ) ) );
	p.halfH = 0.5f * ( fabsf( DotProduct( size, p.up ) ) );
	if ( p.halfW < 1.0f ) p.halfW = 0.5f * max( size.x, max( size.y, size.z ) );
	if ( p.halfH < 1.0f ) p.halfH = p.halfW;
	if ( p.halfW < 1.0f ) p.halfW = 8.0f;
	if ( p.halfH < 1.0f ) p.halfH = 8.0f;

	Q_strncpy( p.targetname, ValueForKey( e, "targetname" ), sizeof( p.targetname ) );
	Q_strncpy( p.target, ValueForKey( e, "target" ), sizeof( p.target ) );

	Msg( "light_portal '%s' -> '%s'  size=%.0fx%.0f\n",
		 p.targetname[0] ? p.targetname : "(unnamed)",
		 p.target[0] ? p.target : "(none)",
		 p.halfW * 2.0f, p.halfH * 2.0f );
	++s_nPortals;
}

void Portal_LinkPairs()
{
	for ( int i = 0; i < s_nPortals; ++i )
	{
		s_Portals[i].link = -1;
		s_Portals[i].bMutual = false;
	}

	for ( int i = 0; i < s_nPortals; ++i )
	{
		if ( !s_Portals[i].target[0] )
			continue;
		for ( int j = 0; j < s_nPortals; ++j )
		{
			if ( i == j )
				continue;
			if ( !s_Portals[j].targetname[0] )
				continue;
			if ( Q_stricmp( s_Portals[i].target, s_Portals[j].targetname ) != 0 )
				continue;
			s_Portals[i].link = j;
			if ( s_Portals[j].target[0] &&
				 s_Portals[i].targetname[0] &&
				 Q_stricmp( s_Portals[j].target, s_Portals[i].targetname ) == 0 )
			{
				s_Portals[i].bMutual = true;
				s_Portals[j].bMutual = true;
				s_Portals[j].link = i;
			}
			break;
		}
	}

	int pairs = Portal_PairCount();
	if ( pairs > 0 )
		Msg( "light_portal: %d linked pair(s)\n", pairs );
}

static bool TraceClear( const Vector &start, const Vector &stop )
{
	FourVectors s, e;
	s.DuplicateVector( start );
	e.DuplicateVector( stop );
	fltx4 vis;
	TestLine( s, e, &vis, -1 );
	return SubFloat( vis, 0 ) > 0.5f;
}

static bool ProjectOntoPortal( const Vector &from, const PortalBrush_t &portal, Vector &outHit, float &u, float &v )
{
	float denom = DotProduct( portal.forward, portal.forward ); // 1
	(void)denom;
	float dist = DotProduct( portal.center - from, portal.forward );
	if ( dist <= 0.5f )
		return false; // behind or on portal

	// Aim toward portal center projected on plane.
	Vector aim = portal.center - from;
	float ad = DotProduct( aim, portal.forward );
	if ( ad <= 1e-4f )
		return false;
	Vector hit = from + aim * ( dist / ad );

	Vector rel = hit - portal.center;
	u = DotProduct( rel, portal.right );
	v = DotProduct( rel, portal.up );
	if ( fabsf( u ) > portal.halfW || fabsf( v ) > portal.halfH )
	{
		// Clamp to aperture and re-test (soft edge accept near rim).
		u = clamp( u, -portal.halfW, portal.halfW );
		v = clamp( v, -portal.halfH, portal.halfH );
		hit = portal.center + portal.right * u + portal.up * v;
	}

	outHit = hit;
	return true;
}

static Vector MapUV( const PortalBrush_t &src, const PortalBrush_t &dst, float u, float v )
{
	// Flip U so facing portals map mirror-correct through the link.
	float nu = ( src.halfW > 1e-4f ) ? ( u / src.halfW ) : 0.0f;
	float nv = ( src.halfH > 1e-4f ) ? ( v / src.halfH ) : 0.0f;
	return dst.center - dst.right * ( nu * dst.halfW ) + dst.up * ( nv * dst.halfH );
}

static bool PortalPathClear( const Vector &from, const Vector &to,
							 const PortalBrush_t &src, const PortalBrush_t &dst )
{
	Vector hitA;
	float u, v;
	if ( !ProjectOntoPortal( from, src, hitA, u, v ) )
		return false;

	// Receiver should be in front of destination portal.
	if ( DotProduct( to - dst.center, dst.forward ) <= 0.5f )
		return false;

	Vector hitB = MapUV( src, dst, u, v );

	Vector a0 = hitA - src.forward * 1.0f;
	Vector b0 = hitB + dst.forward * 1.0f;

	if ( !TraceClear( from, a0 ) )
		return false;
	if ( !TraceClear( b0, to ) )
		return false;
	return true;
}

static float PortalFormFactor( const CPatch *p1, const CPatch *p2,
							   const PortalBrush_t &src, const PortalBrush_t &dst )
{
	// Warp p2 into src's space for form-factor (angle rotation via basis).
	Vector rel = p2->origin - dst.center;
	float x = DotProduct( rel, dst.right );
	float y = DotProduct( rel, dst.up );
	float z = DotProduct( rel, dst.forward );
	Vector p2w = src.center - src.right * x + src.up * y - src.forward * z;

	Vector nrel = p2->normal;
	float nx = DotProduct( nrel, dst.right );
	float ny = DotProduct( nrel, dst.up );
	float nz = DotProduct( nrel, dst.forward );
	Vector n2w = -src.right * nx + src.up * ny - src.forward * nz;
	VectorNormalize( n2w );

	Vector vDelta;
	VectorSubtract( p1->origin, p2w, vDelta );
	float flLength = VectorNormalize( vDelta );
	if ( flLength < 1.0f )
		flLength = 1.0f;

	float scale = -DotProduct( vDelta, p1->normal ) * DotProduct( vDelta, n2w ) / ( flLength * flLength );
	if ( scale <= 0.0f )
		return 0.0f;

	float trans = p2->area * scale;
	if ( trans <= PORTAL_TRANSFER_EPS )
		return 0.0f;

	// Soft aperture attenuation by solid angle of portal from p1.
	float portalArea = ( 2.0f * src.halfW ) * ( 2.0f * src.halfH );
	Vector toPortal = src.center - p1->origin;
	float distP = VectorNormalize( toPortal );
	if ( distP < 1.0f ) distP = 1.0f;
	float cosP = DotProduct( toPortal, p1->normal );
	if ( cosP < 0.0f ) cosP = 0.0f;
	float aperture = ( portalArea * cosP ) / ( M_PI * distP * distP );
	if ( aperture > 1.0f ) aperture = 1.0f;
	trans *= max( 0.05f, aperture );

	return trans;
}

static void AppendTransfer( int ndxPatch1, int ndxPatch2, float rawTrans )
{
	if ( ndxPatch1 == g_Patches.InvalidIndex() || ndxPatch2 == g_Patches.InvalidIndex() )
		return;
	if ( rawTrans <= PORTAL_TRANSFER_EPS )
		return;

	CPatch *p = &g_Patches.Element( ndxPatch1 );
	if ( p->sky )
		return;
	if ( p->numtransfers >= MAX_PATCHES )
		return;

	for ( int i = 0; i < p->numtransfers; ++i )
	{
		if ( p->transfers[i].patch == ndxPatch2 )
			return;
	}

	// Post-MakeScales units: raw form-factor energy scaled like MakeScales (1/PI).
	float scaled = rawTrans * ( 1.0f / M_PI );

	transfer_t *neu = (transfer_t *)realloc( p->transfers, ( p->numtransfers + 1 ) * sizeof( transfer_t ) );
	if ( !neu )
		return;
	p->transfers = neu;
	p->transfers[p->numtransfers].patch = ndxPatch2;
	p->transfers[p->numtransfers].transfer = scaled;
	p->numtransfers++;

	ThreadLock();
	total_transfer++;
	if ( p->numtransfers > max_transfer )
		max_transfer = p->numtransfers;
	ThreadUnlock();
}

static bool PatchNearPortal( const CPatch *patch, const PortalBrush_t &portal )
{
	if ( patch->child1 != g_Patches.InvalidIndex() )
		return false;
	Vector d = patch->origin - portal.center;
	if ( DotProduct( d, d ) > PORTAL_NEAR_DIST * PORTAL_NEAR_DIST )
		return false;
	// Prefer patches facing the portal.
	Vector toP = portal.center - patch->origin;
	VectorNormalize( toP );
	if ( DotProduct( toP, patch->normal ) < 0.0f )
		return false;
	if ( DotProduct( patch->origin - portal.center, portal.forward ) <= 0.0f )
		return false;
	return true;
}

void Portal_BuildExtraTransfers()
{
	if ( !Portal_HasPairs() )
		return;

	Msg( "Building light_portal transfers...\n" );

	CUtlVector<int> nearA;
	CUtlVector<int> nearB;
	int nLinks = 0;
	unsigned int uiPatchCount = g_Patches.Count();

	for ( int i = 0; i < s_nPortals; ++i )
	{
		int j = s_Portals[i].link;
		if ( j < 0 )
			continue;

		const PortalBrush_t &A = s_Portals[i];
		const PortalBrush_t &B = s_Portals[j];

		nearA.RemoveAll();
		nearB.RemoveAll();
		for ( unsigned int p = 0; p < uiPatchCount; ++p )
		{
			CPatch *patch = &g_Patches[p];
			if ( PatchNearPortal( patch, A ) )
				nearA.AddToTail( (int)p );
			if ( PatchNearPortal( patch, B ) )
				nearB.AddToTail( (int)p );
		}

		// Directed transfers for this portal's target link (A -> B).
		for ( int a = 0; a < nearA.Count(); ++a )
		{
			for ( int b = 0; b < nearB.Count(); ++b )
			{
				CPatch *pa = &g_Patches[nearA[a]];
				CPatch *pb = &g_Patches[nearB[b]];
				if ( !PortalPathClear( pa->origin, pb->origin, A, B ) )
					continue;
				float t = PortalFormFactor( pa, pb, A, B );
				if ( t > PORTAL_TRANSFER_EPS )
				{
					AppendTransfer( nearA[a], nearB[b], t );
					++nLinks;
				}
			}
		}
	}

	Msg( "light_portal: %d extra transfer link(s)\n", nLinks );
}
