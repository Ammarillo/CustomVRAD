//========= Copyright Valve Corporation, All rights reserved. ============//
//
// light_portal — linked-space radiosity portals (CustomVRAD).
//
//=============================================================================//

#ifndef VRAD_PORTAL_H
#define VRAD_PORTAL_H
#pragma once

#include "mathlib/vector.h"

struct entity_t;
struct transfer_t;

void Portal_Clear();
void Portal_ParseEntity( entity_t *e );
void Portal_LinkPairs(); // call after all entities parsed
bool Portal_HasPairs();
int  Portal_PairCount();

// Extra patch transfers through portals (call after BuildVisLeafs).
void Portal_BuildExtraTransfers();

#endif // VRAD_PORTAL_H
