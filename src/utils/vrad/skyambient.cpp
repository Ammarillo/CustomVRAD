//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Shared sky ambient evaluation (flat ambient + volume blend + sun).
//
//=============================================================================//

#include "vrad.h"
#include "skyambient.h"
#include "envvolume.h"

static directlight_t *FindSkylightForEnv( int envId )
{
	for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
	{
		if ( dl->light.type == emit_skylight && dl->m_nEnvId == envId )
			return dl;
	}
	return NULL;
}

void SkyAmbient_GetParams( const directlight_t *pAmbient, SkyAmbientParams_t &out )
{
	out.ambient = pAmbient->light.intensity;
}

void SkyAmbient_ComputeAtPos( const Vector &pos, Vector &intensity )
{
	intensity.Init();

	if ( !LightEnv_HasVolumes() )
	{
		directlight_t *pAmb = NULL;
		for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
		{
			if ( dl->light.type == emit_skyambient && dl->m_nEnvId == LIGHTENV_ID_DEFAULT )
			{
				pAmb = dl;
				break;
			}
		}
		if ( !pAmb )
		{
			for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
			{
				if ( dl->light.type == emit_skyambient )
				{
					pAmb = dl;
					break;
				}
			}
		}
		if ( !pAmb )
			return;

		intensity = pAmb->light.intensity;
		return;
	}

	for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
	{
		if ( dl->light.type != emit_skyambient )
			continue;

		float flWeight = ( dl->m_nEnvId == LIGHTENV_ID_NONE ) ?
			1.0f : LightEnv_GetWeight( dl->m_nEnvId, pos );
		if ( flWeight <= 0.0f )
			continue;

		VectorMA( intensity, flWeight, dl->light.intensity, intensity );
	}
}

bool SkyAmbient_ComputeSunAtPos( const Vector &pos, Vector &sunDir, Vector &sunIntensity )
{
	sunDir.Init( 0, 0, 1 );
	sunIntensity.Init();
	bool bAny = false;

	if ( !LightEnv_HasVolumes() )
	{
		directlight_t *pSun = FindSkylightForEnv( LIGHTENV_ID_DEFAULT );
		if ( !pSun )
		{
			for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
			{
				if ( dl->light.type == emit_skylight )
				{
					pSun = dl;
					break;
				}
			}
		}
		if ( !pSun )
			return false;
		VectorScale( pSun->light.normal, -1.0f, sunDir );
		VectorNormalize( sunDir );
		sunIntensity = pSun->light.intensity;
		return true;
	}

	Vector dirAccum( 0, 0, 0 );
	for ( directlight_t *dl = activelights; dl != NULL; dl = dl->next )
	{
		if ( dl->light.type != emit_skylight )
			continue;

		float flWeight = ( dl->m_nEnvId == LIGHTENV_ID_NONE ) ?
			1.0f : LightEnv_GetWeight( dl->m_nEnvId, pos );
		if ( flWeight <= 0.0f )
			continue;

		Vector d;
		VectorScale( dl->light.normal, -1.0f, d );
		VectorNormalize( d );
		VectorMA( dirAccum, flWeight, d, dirAccum );
		VectorMA( sunIntensity, flWeight, dl->light.intensity, sunIntensity );
		bAny = true;
	}

	if ( !bAny )
		return false;

	float len = VectorNormalize( dirAccum );
	if ( len > 1e-6f )
		sunDir = dirAccum;
	return true;
}
