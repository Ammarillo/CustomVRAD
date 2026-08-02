// GPU luxel path baker — included at end of pathtrace_dxr_device.cpp

static const char *kPtLuxelBakeCS =
R"HLSL(
RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<float3> Verts : register(t1);
ByteAddressBuffer TriFlags : register(t2);
StructuredBuffer<float3> TriAlbedo : register(t3);
StructuredBuffer<float4> FilterMeta : register(t18); // 5 x float4 per tri
Texture2DArray<float4> FilterArray : register(t19);
StructuredBuffer<float4> LightsA : register(t4);
StructuredBuffer<float4> LightsB : register(t5);
StructuredBuffer<float4> LightsC : register(t6);
StructuredBuffer<int4>   LightsD : register(t7);
StructuredBuffer<float4> LuxelsA : register(t8);
StructuredBuffer<float4> LuxelsB : register(t9);
StructuredBuffer<int4>   LuxelsC : register(t10);
StructuredBuffer<uint>   LocalIdx : register(t11);
StructuredBuffer<float>  LocalCdf : register(t12);
StructuredBuffer<float4> EmitA : register(t13); // 4 x float4 per emit tri
StructuredBuffer<float4> EmitE : register(t14); // 3 x float4 per emit tri (e0,e1,e2)
StructuredBuffer<float>  EmitCdf : register(t15);
StructuredBuffer<float4> EnvVols : register(t16); // 4 x float4 per light_env_vol
StructuredBuffer<float4> BounceVols : register(t17); // 3 x float4 per light_bounce_vol
RWStructuredBuffer<float4> OutRGB : register(u0);

cbuffer CB : register(b0)
{
	uint LuxelCount;
	uint Spp;
	uint Bounces;
	uint LightCount;
	uint LightSamples;
	uint SoftSamplesMax;
	uint SoftMode;
	uint TriCount;
	float OccludeBias;
	float MaxTrace;
	float FireflyCap;
	float PenumbraScale;
	float SkyAmbX;
	float SkyAmbY;
	float SkyAmbZ;
	uint LocalCount;
	uint SampleOffset;
	uint EmitTriCount;
	uint EmitSamples;
	uint EnvVolCount;
	float DefaultBounceIntensity;
	uint BounceVolCount;
	float CliBounceBoost;
	float CliBounceChroma;
};

#ifndef PT_ENABLE_VOLUMES
#define PT_ENABLE_VOLUMES 0
#endif

static const uint TRACE_ID_SKY = 0x01000000u;
static const uint TRACE_ID_FILTER = 0x08000000u;
static const float PI = 3.14159265f;
static const float INV_PI = 0.318309886f;
static const int MAX_FILTER_HITS = 64;
// Advance past a hit; sheet exit dedup handles thin brush backs separately.
static const float FILTER_ADVANCE = 0.15f;
// Max world-unit gap between enter/exit of the same thin brush sheet.
static const float FILTER_SHEET_THICK = 12.0f;

// ---- Oklab (Björn Ottosson) https://bottosson.github.io/posts/oklab/ ----
float3 LinearToOklab( float3 c )
{
	c = max( c, 0.0f );
	float l = 0.4122214708f * c.x + 0.5363325363f * c.y + 0.0514459929f * c.z;
	float m = 0.2119034982f * c.x + 0.6806995451f * c.y + 0.1073969566f * c.z;
	float s = 0.0883024619f * c.x + 0.2817188376f * c.y + 0.6299787005f * c.z;
	float l_ = pow( l, 1.0f / 3.0f );
	float m_ = pow( m, 1.0f / 3.0f );
	float s_ = pow( s, 1.0f / 3.0f );
	return float3(
		0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
		1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
		0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_ );
}

float3 OklabToLinear( float3 lab )
{
	float l_ = lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z;
	float m_ = lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z;
	float s_ = lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z;
	float l = l_ * l_ * l_;
	float m = m_ * m_ * m_;
	float s = s_ * s_ * s_;
	return float3(
		+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
		-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
		-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s );
}

float3 OklabLerpLin( float3 c0, float3 c1, float t )
{
	float3 a = LinearToOklab( c0 );
	float3 b = LinearToOklab( c1 );
	return saturate( OklabToLinear( lerp( a, b, saturate( t ) ) ) );
}

float3 OklabStackFilter( float3 accumT, float3 filterT )
{
	float3 t = LinearToOklab( max( accumT, 1e-4f ) );
	float3 f = LinearToOklab( max( filterT, 1e-4f ) );
	float3 o = float3( max( t.x * f.x, 1e-4f ), t.y + f.y, t.z + f.z );
	return saturate( OklabToLinear( o ) );
}

float3 OklabScaleChroma( float3 c, float sat )
{
	if ( sat <= 1.0001f )
		return c;
	float3 o = LinearToOklab( max( c, 0.0f ) );
	o.y *= sat;
	o.z *= sat;
	return max( OklabToLinear( o ), 0.0f );
}

float3 OklabApplyTintPreserveL( float3 light, float3 tint, float lumScale )
{
	float3 ol = LinearToOklab( max( light, 0.0f ) );
	float3 ot = LinearToOklab( max( tint, 0.0f ) );
	ol.x *= max( lumScale, 0.0f );
	ol.y = ot.y;
	ol.z = ot.z;
	return max( OklabToLinear( ol ), 0.0f );
}

uint WangHash( uint x )
{
	x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
	return x;
}
float Hash01( uint x )
{
	return ( WangHash( x ) & 0xFFFFFFu ) / 16777216.0f;
}

float3 ClampFirefly( float3 v, float cap )
{
	float m = max( v.x, max( v.y, v.z ) );
	if ( m > cap && m > 0 )
		v *= ( cap / m );
	return v;
}

void Orthonormal( float3 n, out float3 t, out float3 b )
{
	if ( abs( n.x ) > 0.1f )
		t = float3( n.y, -n.x, 0 );
	else
		t = float3( 0, n.z, -n.y );
	t = normalize( t );
	b = cross( n, t );
}

float3 CosineHemi( float3 n, float u1, float u2 )
{
	float r = sqrt( u1 );
	float phi = 2.0f * PI * u2;
	float3 t, b;
	Orthonormal( n, t, b );
	float3 d = t * ( r * cos( phi ) ) + b * ( r * sin( phi ) ) + n * sqrt( max( 0.0f, 1.0f - u1 ) );
	return normalize( d );
}

bool TraceClosest( float3 o, float3 d, float tmin, float tmax, out float tHit, out uint flags, out uint prim, out float3 nHit )
{
	RayDesc ray;
	ray.Origin = o;
	ray.Direction = d;
	ray.TMin = tmin;
	ray.TMax = tmax;
	RayQuery<RAY_FLAG_NONE> q;
	q.TraceRayInline( Scene, RAY_FLAG_NONE, 0xFF, ray );
	q.Proceed();
	tHit = -1;
	flags = 0;
	prim = 0;
	nHit = float3( 0, 0, 1 );
	if ( q.CommittedStatus() != COMMITTED_TRIANGLE_HIT )
		return false;
	tHit = q.CommittedRayT();
	prim = q.CommittedPrimitiveIndex();
	flags = TriFlags.Load( prim * 4 );
	float3 a = Verts[prim * 3 + 0];
	float3 b = Verts[prim * 3 + 1];
	float3 c = Verts[prim * 3 + 2];
	nHit = normalize( cross( b - a, c - a ) );
	if ( dot( nHit, d ) > 0 )
		nHit = -nHit;
	return true;
}

#if PT_ENABLE_VOLUMES
// Any-hit shadow: much cheaper than full closest (Frostbite/DXR practice).
float SoftMin3( float a, float b, float c, float k )
{
	if ( k < 1e-4f )
		return min( a, min( b, c ) );
	float m = min( a, min( b, c ) );
	float ea = exp( -( a - m ) / k );
	float eb = exp( -( b - m ) / k );
	float ec = exp( -( c - m ) / k );
	return ( a * ea + b * eb + c * ec ) / ( ea + eb + ec );
}

float SmootherStep01( float t )
{
	t = saturate( t );
	return t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
}

float SignedBlendDist( float3 p, float3 mins, float3 maxs, float softK )
{
	float dx = min( p.x - mins.x, maxs.x - p.x );
	float dy = min( p.y - mins.y, maxs.y - p.y );
	float dz = min( p.z - mins.z, maxs.z - p.z );
	if ( dx >= 0.0f && dy >= 0.0f && dz >= 0.0f )
		return SoftMin3( dx, dy, dz, softK );
	float3 o = max( mins - p, 0.0f ) + max( p - maxs, 0.0f );
	return -length( o );
}

float4 EnvVolA( uint i ) { return EnvVols[i * 4 + 0]; }
float4 EnvVolB( uint i ) { return EnvVols[i * 4 + 1]; }
float4 EnvVolC( uint i ) { return EnvVols[i * 4 + 2]; }
float4 EnvVolD( uint i ) { return EnvVols[i * 4 + 3]; }

float TerritoryFactor( uint idx, float3 pos, float signedDist )
{
	float factor = 1.0f;
	float4 va = EnvVolA( idx );
	float4 vc = EnvVolC( idx );
	float blendV = va.w;
	float prioV = vc.x;
	for ( uint j = 0; j < EnvVolCount; ++j )
	{
		if ( j == idx ) continue;
		float4 oa = EnvVolA( j );
		float4 ob = EnvVolB( j );
		float4 oc = EnvVolC( j );
		float softKJ = ( oa.w > 0.0f ) ? ( oa.w * 0.15f ) : 0.0f;
		float distJ = SignedBlendDist( pos, oa.xyz, ob.xyz, softKJ );
		float advantage = distJ - signedDist;
		if ( advantage <= 0.0f ) continue;
		float handoff = 0.25f * ( blendV + oa.w );
		if ( oc.x > prioV ) handoff *= 0.5f;
		else if ( prioV > oc.x ) handoff *= 2.0f;
		handoff = max( handoff, 8.0f );
		factor *= SmootherStep01( 1.0f - advantage / handoff );
		if ( factor <= 1e-5f ) return 0.0f;
	}
	return factor;
}

float RawVolumeWeight( uint idx, float3 pos )
{
	float4 a = EnvVolA( idx );
	float4 b = EnvVolB( idx );
	float blend = a.w;
	int mode = (int)b.w;
	if ( blend <= 0.0f )
	{
		if ( pos.x < a.x || pos.x > b.x || pos.y < a.y || pos.y > b.y || pos.z < a.z || pos.z > b.z )
			return 0.0f;
		return 1.0f;
	}
	float softK = blend * 0.15f;
	float dist = SignedBlendDist( pos, a.xyz, b.xyz, softK );
	float t;
	if ( mode == 0 ) // INSIDE
	{
		if ( dist <= 0.0f ) return 0.0f;
		t = dist / blend;
	}
	else if ( mode == 1 ) // OUTSIDE
	{
		if ( dist >= 0.0f ) return 1.0f;
		t = 1.0f + dist / blend;
	}
	else // CENTER
	{
		t = ( dist + blend ) / ( 2.0f * blend );
	}
	float w = SmootherStep01( t );
	if ( w <= 0.0f ) return 0.0f;
	return w * TerritoryFactor( idx, pos, dist );
}

// Weight for sky/ambient light envId at pos (0=default, 1..N=volumes, <0=always 1).
float EnvWeight( int envId, float3 pos )
{
	if ( envId < 0 )
		return 1.0f;
	if ( EnvVolCount == 0 )
		return ( envId == 0 ) ? 1.0f : 0.0f;

	int saturatedIdx = -1;
	float bestPrio = -1e30f;
	float bestSize = 1e30f;
	float sum = 0.0f;
	float matchW = 0.0f;
	bool haveMatch = false;
	for ( uint i = 0; i < EnvVolCount; ++i )
	{
		float w = RawVolumeWeight( i, pos );
		sum += w;
		float4 c = EnvVolC( i );
		if ( w >= 1.0f - 1e-4f )
		{
			if ( saturatedIdx < 0 || c.x > bestPrio || ( c.x == bestPrio && c.y < bestSize ) )
			{
				saturatedIdx = (int)i;
				bestPrio = c.x;
				bestSize = c.y;
			}
		}
		if ( (int)c.z == envId )
		{
			matchW = w;
			haveMatch = true;
		}
	}
	if ( saturatedIdx >= 0 )
	{
		int satEnv = (int)EnvVolC( (uint)saturatedIdx ).z;
		return ( envId == satEnv ) ? 1.0f : 0.0f;
	}
	if ( envId == 0 )
		return ( sum <= 1.0f ) ? ( 1.0f - sum ) : 0.0f;
	if ( !haveMatch )
		return 0.0f;
	if ( sum <= 1.0f )
		return matchW;
	return matchW / max( sum, 1e-8f );
}

int DominantEnvId( float3 pos )
{
	if ( EnvVolCount == 0 )
		return 0;
	int best = 0;
	float bestW = EnvWeight( 0, pos );
	for ( uint i = 0; i < EnvVolCount; ++i )
	{
		int eid = (int)EnvVolC( i ).z;
		float w = EnvWeight( eid, pos );
		if ( w > bestW )
		{
			bestW = w;
			best = eid;
		}
	}
	return best;
}

void TintInboundBounce( int receiverEnvId, int emitterEnvId, inout float3 light )
{
	if ( receiverEnvId <= 0 || EnvVolCount == 0 )
		return;
	// volumes are envId 1..N stored at index envId-1
	uint idx = (uint)( receiverEnvId - 1 );
	if ( idx >= EnvVolCount )
		return;
	int f = (int)EnvVolC( idx ).w;
	if ( ( f & 4 ) == 0 )
		return; // BounceVolColor off
	if ( emitterEnvId == receiverEnvId )
		return;
	float lum = light.x + light.y + light.z;
	if ( lum < 1e-10f )
		return;
	float3 tint = EnvVolD( idx ).xyz;
	float bounceInt = EnvVolD( idx ).w;
	float scale = 1.0f;
	if ( ( f & 8 ) != 0 )
		scale = bounceInt / max( DefaultBounceIntensity, 1e-6f );
	light = OklabApplyTintPreserveL( light, tint, scale );
}

float4 BVolA( uint i ) { return BounceVols[i * 3 + 0]; }
float4 BVolB( uint i ) { return BounceVols[i * 3 + 1]; }
float4 BVolC( uint i ) { return BounceVols[i * 3 + 2]; }

float BounceVolMembership( uint idx, float3 pos )
{
	float4 a = BVolA( idx );
	float4 b = BVolB( idx );
	float blend = a.w;
	int mode = (int)b.w;
	if ( blend <= 0.0f )
	{
		if ( pos.x < a.x || pos.x > b.x || pos.y < a.y || pos.y > b.y || pos.z < a.z || pos.z > b.z )
			return 0.0f;
		return 1.0f;
	}
	float softK = blend * 0.15f;
	float dist = SignedBlendDist( pos, a.xyz, b.xyz, softK );
	float t;
	if ( mode == 0 )
	{
		if ( dist <= 0.0f ) return 0.0f;
		t = dist / blend;
	}
	else if ( mode == 1 )
	{
		if ( dist >= 0.0f ) return 1.0f;
		t = 1.0f + dist / blend;
	}
	else
		t = ( dist + blend ) / ( 2.0f * blend );
	return SmootherStep01( t );
}

void ResolveBounceVol( float3 pos, out float boost, out float chroma )
{
	boost = CliBounceBoost;
	chroma = CliBounceChroma;
	if ( BounceVolCount == 0 )
		return;
	int best = -1;
	float bestW = 0.0f;
	for ( uint i = 0; i < BounceVolCount; ++i )
	{
		float w = BounceVolMembership( i, pos );
		if ( w <= 1e-6f ) continue;
		float4 c = BVolC( i );
		if ( best < 0 )
		{
			best = (int)i;
			bestW = w;
			continue;
		}
		float4 cb = BVolC( (uint)best );
		if ( (int)c.x != (int)cb.x )
		{
			if ( c.x > cb.x ) { best = (int)i; bestW = w; }
			continue;
		}
		if ( w > bestW + 1e-5f )
		{
			best = (int)i;
			bestW = w;
		}
		else if ( abs( w - bestW ) <= 1e-5f && c.y < cb.y )
		{
			best = (int)i;
			bestW = w;
		}
	}
	if ( best < 0 || bestW <= 1e-6f )
		return;
	float4 c = BVolC( (uint)best );
	if ( c.z >= 0.0f )
		boost = CliBounceBoost + ( c.z - CliBounceBoost ) * bestW;
	if ( c.w >= 0.0f )
		chroma = CliBounceChroma + ( c.w - CliBounceChroma ) * bestW;
}

float3 ApplyBounceVolAlbedo( float3 hitPos, float3 albedo, uint bounce )
{
	// Fast path: no volumes and identity CLI → skip resolve entirely.
	if ( BounceVolCount == 0 && CliBounceBoost == 1.0f && CliBounceChroma <= 0.0f )
		return albedo;
	float boost, chroma;
	ResolveBounceVol( hitPos, boost, chroma );
	albedo *= boost;
	if ( chroma > 0.0f )
	{
		float falloff = 1.0f / ( 1.0f + (float)bounce );
		float sat = 1.0f + chroma * falloff;
		albedo = OklabScaleChroma( albedo, sat );
	}
	return albedo;
}

bool PointInEnvVol( uint i, float3 p )
{
	float4 a = EnvVolA( i );
	float4 b = EnvVolB( i );
	return p.x >= a.x && p.x <= b.x && p.y >= a.y && p.y <= b.y && p.z >= a.z && p.z <= b.z;
}

bool IgnoreSkyOccluder( float3 samplePos, float3 hitPos )
{
	if ( EnvVolCount == 0 )
		return false;
	for ( uint i = 0; i < EnvVolCount; ++i )
	{
		int f = (int)EnvVolC( i ).w;
		bool outsideCast = ( f & 1 ) != 0;
		bool insideCast = ( f & 2 ) != 0;
		if ( outsideCast && insideCast )
			continue;
		bool sampleIn = PointInEnvVol( i, samplePos );
		bool hitIn = PointInEnvVol( i, hitPos );
		if ( sampleIn && !hitIn && !outsideCast )
			return true;
		if ( !sampleIn && hitIn && !insideCast )
			return true;
	}
	return false;
}

#else
// Lean path: compile-time strip of env/bounce volume helpers (no register bloat).
float EnvWeight( int envId, float3 pos )
{
	return ( envId <= 0 ) ? 1.0f : 0.0f;
}
float3 ApplyBounceVolAlbedo( float3 hitPos, float3 albedo, uint bounce )
{
	albedo *= CliBounceBoost;
	if ( CliBounceChroma > 0.0f )
	{
		float falloff = 1.0f / ( 1.0f + (float)bounce );
		float sat = 1.0f + CliBounceChroma * falloff;
		albedo = OklabScaleChroma( albedo, sat );
	}
	return albedo;
}
#endif
)HLSL"
R"HLSL(
// Sample $vrad_filter transmittance at a world hit (matches CPU SampleAtPos).
float3 SampleFilterT( uint prim, float3 hitPos )
{
	if ( prim >= TriCount )
		return 0;
	uint base = prim * 5u;
	float4 m0 = FilterMeta[base + 0];
	float mode = m0.w;
	if ( mode < 0.5f )
		return 0; // opaque
	if ( mode < 1.5f )
		return m0.xyz; // avg / flat fallback

	float4 p = FilterMeta[base + 1];
	float4 uvec = FilterMeta[base + 2];
	float4 vvec = FilterMeta[base + 3];
	float4 info = FilterMeta[base + 4]; // x=layer, y=uvScale (mip/full), z=mipW, w=mipH
	float scale = max( info.y, 1e-6f );
	float s = ( dot( hitPos, uvec.xyz ) + uvec.w ) * scale;
	float t = ( dot( hitPos, vvec.xyz ) + vvec.w ) * scale;
	float tw = max( info.z, 1.0f );
	float th = max( info.w, 1.0f );
	uint flags = (uint)p.w;
	if ( ( flags & 1u ) != 0 )
		s = clamp( s, 0.0f, tw - 1.0f );
	else
	{
		s = fmod( s, tw );
		if ( s < 0.0f ) s += tw;
	}
	if ( ( flags & 2u ) != 0 )
		t = clamp( t, 0.0f, th - 1.0f );
	else
	{
		t = fmod( t, th );
		if ( t < 0.0f ) t += th;
	}
	int layer = (int)info.x;
	float4 c = FilterArray.Load( int4( int2( s, t ), layer, 0 ) );
	float3 rgb = c.rgb * c.rgb; // sRGB-ish → linear (matches CPU)
	float thick = p.z;
	if ( abs( thick - 1.0f ) > 1e-3f )
		rgb = pow( max( rgb, 1e-6f ), thick );
	float str = p.x;
	// Strength: Oklab lerp white → filter (perceptual).
	float3 tint = OklabLerpLin( float3( 1, 1, 1 ), rgb, saturate( str ) );
	return saturate( tint * ( 1.0f - p.y ) );
}

bool IsFilterPrim( uint prim )
{
	if ( prim >= TriCount )
		return false;
	return FilterMeta[prim * 5u].w >= 0.5f;
}

float3 FilterFaceN( uint prim )
{
	float3 a = Verts[prim * 3 + 0];
	float3 b = Verts[prim * 3 + 1];
	float3 c = Verts[prim * 3 + 2];
	return normalize( cross( b - a, c - a ) );
}

// Default two-sided (bit4). Onesided materials only tint on front-face hits.
bool FilterShouldApply( uint prim, float3 rayDir )
{
	if ( prim >= TriCount )
		return false;
	uint f = (uint)FilterMeta[prim * 5u + 1u].w;
	if ( ( f & 4u ) != 0 )
		return true;
	return dot( FilterFaceN( prim ), rayDir ) < 0.0f;
}

float FilterAdvancePast( float t )
{
	return t + max( FILTER_ADVANCE, abs( t ) * 1e-5f + 1e-3f );
}

// True if this hit is the exit face of the sheet we just entered (avoid double tint).
bool FilterIsSheetExit( float t, float3 faceN, float lastT, float3 lastN )
{
	if ( lastT < -1e20f )
		return false;
	if ( ( t - lastT ) > FILTER_SHEET_THICK )
		return false;
	return dot( faceN, lastN ) < -0.85f;
}

// RGB transmittance (1 = clear). Stacked panes: photographic gel multiply T0*T1*…
float3 VisRGB( float3 from, float3 to )
{
	float3 delta = to - from;
	float dist = length( delta );
	if ( dist <= OccludeBias * 2.0f )
		return float3( 1, 1, 1 );
	delta /= dist;
	float tmin = OccludeBias;
	float tmax = dist - min( OccludeBias, dist * 0.25f );
	if ( tmax <= tmin )
		return float3( 1, 1, 1 );
	float3 T = 1;
	uint lastPrim = 0xFFFFFFFFu;
	uint lastFace = 0xFFFFFFFFu;
	float lastFilterT = -1e30f;
	float3 lastFilterN = float3( 0, 0, 0 );
	[loop] for ( int pass = 0; pass < MAX_FILTER_HITS + 1; ++pass )
	{
		if ( tmax <= tmin + 1e-4f )
			break;
		float t; uint flags, prim; float3 nHit;
		if ( !TraceClosest( from, delta, tmin, tmax, t, flags, prim, nHit ) )
			return T;
		if ( flags & TRACE_ID_SKY )
			return T;
#if PT_ENABLE_VOLUMES
		if ( EnvVolCount > 0 && IgnoreSkyOccluder( from, from + delta * t ) )
			return T;
#endif
		if ( ( flags & TRACE_ID_FILTER ) != 0 || IsFilterPrim( prim ) )
		{
			uint faceId = ( flags & TRACE_ID_FILTER ) != 0 ? ( flags & 0x00FFFFFFu ) : 0xFFFFFFFEu;
			float3 faceN = FilterFaceN( prim );
			if ( prim == lastPrim || faceId == lastFace || FilterIsSheetExit( t, faceN, lastFilterT, lastFilterN ) )
			{
				tmin = FilterAdvancePast( t );
				continue;
			}
			lastPrim = prim;
			lastFace = faceId;
			if ( !FilterShouldApply( prim, delta ) )
			{
				tmin = FilterAdvancePast( t );
				continue;
			}
			float3 F = SampleFilterT( prim, from + delta * t );
			// Meta miss (mode 0) must not hard-block filter hits.
			if ( max( F.x, max( F.y, F.z ) ) < 1e-5f && ( flags & TRACE_ID_FILTER ) != 0 )
				F = float3( 1, 1, 1 );
			T = OklabStackFilter( T, max( F, float3( 1e-4f, 1e-4f, 1e-4f ) ) );
			lastFilterT = t;
			lastFilterN = faceN;
			tmin = FilterAdvancePast( t );
			continue;
		}
		return 0;
	}
	return T;
}

bool Occluded( float3 from, float3 to )
{
	float3 T = VisRGB( from, to );
	return max( T.x, max( T.y, T.z ) ) < 1e-5f;
}

float3 SunVisRGB( float3 pos, float3 axis )
{
	float tmin = OccludeBias;
	float3 T = 1;
	uint lastPrim = 0xFFFFFFFFu;
	uint lastFace = 0xFFFFFFFFu;
	float lastFilterT = -1e30f;
	float3 lastFilterN = float3( 0, 0, 0 );
	[loop] for ( int pass = 0; pass < MAX_FILTER_HITS + 1; ++pass )
	{
		float t; uint flags, prim; float3 nHit;
		if ( !TraceClosest( pos, axis, tmin, MaxTrace, t, flags, prim, nHit ) )
			return T;
		if ( flags & TRACE_ID_SKY )
			return T;
#if PT_ENABLE_VOLUMES
		if ( EnvVolCount > 0 && IgnoreSkyOccluder( pos, pos + axis * t ) )
			return T;
#endif
		if ( ( flags & TRACE_ID_FILTER ) != 0 || IsFilterPrim( prim ) )
		{
			uint faceId = ( flags & TRACE_ID_FILTER ) != 0 ? ( flags & 0x00FFFFFFu ) : 0xFFFFFFFEu;
			float3 faceN = FilterFaceN( prim );
			if ( prim == lastPrim || faceId == lastFace || FilterIsSheetExit( t, faceN, lastFilterT, lastFilterN ) )
			{
				tmin = FilterAdvancePast( t );
				continue;
			}
			lastPrim = prim;
			lastFace = faceId;
			if ( !FilterShouldApply( prim, axis ) )
			{
				tmin = FilterAdvancePast( t );
				continue;
			}
			float3 F = SampleFilterT( prim, pos + axis * t );
			if ( max( F.x, max( F.y, F.z ) ) < 1e-5f && ( flags & TRACE_ID_FILTER ) != 0 )
				F = float3( 1, 1, 1 );
			T = OklabStackFilter( T, max( F, float3( 1e-4f, 1e-4f, 1e-4f ) ) );
			lastFilterT = t;
			lastFilterN = faceN;
			tmin = FilterAdvancePast( t );
			continue;
		}
		return 0;
	}
	return T;
}

bool SunBlocked( float3 pos, float3 axis )
{
	float3 T = SunVisRGB( pos, axis );
	return max( T.x, max( T.y, T.z ) ) < 1e-5f;
}

float ApplyFade( float falloff, float dist, float fadeS, float fadeE )
{
	if ( fadeE <= fadeS )
		return falloff;
	if ( dist > fadeE )
		return 0;
	float t = saturate( ( dist - fadeS ) / ( fadeE - fadeS ) );
	t = 1.0f - t;
	float mult = t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
	return falloff * mult;
}

float4 LA0( uint i ) { return LightsA[i * 2 + 0]; }
float4 LA1( uint i ) { return LightsA[i * 2 + 1]; }
float4 LB0( uint i ) { return LightsB[i * 2 + 0]; }
float4 LB1( uint i ) { return LightsB[i * 2 + 1]; }
float4 LC0( uint i ) { return LightsC[i * 2 + 0]; }
float4 LC1( uint i ) { return LightsC[i * 2 + 1]; }

float PointFalloff( uint i, float dist )
{
	float4 B1 = LB1( i );
	float4 C0 = LC0( i );
	float c = B1.z, l = B1.w, q = C0.x;
	float fadeS = C0.y, fadeE = C0.z, cap = C0.w;
	dist = max( dist, 1.0f );
	float d = min( dist, ( cap > 0.0f ) ? cap : 1e20f );
	float denom = c + l * d + q * d * d;
	if ( denom <= 1e-8f )
		return 0;
	return ApplyFade( 1.0f / denom, dist, fadeS, fadeE );
}

float SpotCone( uint i, float3 lightToSample )
{
	float3 Ldir = LB0( i ).xyz;
	float3 dir = normalize( lightToSample );
	float dotp = dot( -dir, normalize( Ldir ) );
	float stop = LB0( i ).w;
	float stop2 = LB1( i ).x;
	float expv = LB1( i ).y;
	if ( dotp <= stop2 )
		return 0;
	if ( dotp >= stop )
		return 1;
	float t = ( dotp - stop2 ) / max( 1e-6f, stop - stop2 );
	return pow( t, max( expv, 1.0f ) );
}


int EstimatePenumbraSamples( float lightSize, float distLight, float tHit, bool blocked )
{
	uint cap = max( SoftSamplesMax, 1u );
	if ( lightSize <= 1e-6f || PenumbraScale <= 1e-6f )
		return 1;
	if ( !blocked )
	{
		float ang = lightSize / max( distLight, 1.0f );
		int n = 1 + (int)( ang * 10.0f * PenumbraScale );
		int litCap = min( 4, (int)cap );
		return clamp( n, 1, litCap );
	}
	float dBlocker = max( distLight - tHit, 1e-3f );
	float wPen = ( tHit * lightSize * PenumbraScale ) / dBlocker;
	int n = 1 + (int)( wPen * 0.35f + 0.5f );
	return clamp( n, 1, (int)cap );
}
)HLSL"
R"HLSL(
void SoftDiskOffset( float3 diskN, float radius, uint i, uint seed, out float3 offsetOut )
{
	offsetOut = 0;
	if ( radius <= 0 )
		return;
	float3 t, b;
	Orthonormal( normalize( diskN ), t, b );
	float u = Hash01( seed + i * 19u + 5u );
	float r = radius * sqrt( u );
	float phi = Hash01( seed + i * 41u + 13u ) * 2.0f * PI;
	offsetOut = t * ( r * cos( phi ) ) + b * ( r * sin( phi ) );
}

void SoftSunDir( float3 axis, float sinExtent, uint i, uint nSoft, uint seed, out float3 outDir )
{
	float3 t1, t2;
	Orthonormal( axis, t1, t2 );
	float cosThetaMax = sqrt( max( 0.0f, 1.0f - sinExtent * sinExtent ) );
	float uRot = Hash01( seed + 17u );
	float phiRot = Hash01( seed + 91u ) * 2.0f * PI;
	float u = ( nSoft <= 1 ) ? 0.0f : ( (float)i + uRot ) / (float)nSoft;
	float cosTheta = 1.0f - u * ( 1.0f - cosThetaMax );
	float sinTheta = sqrt( max( 0.0f, 1.0f - cosTheta * cosTheta ) );
	float phi = (float)i * 2.3999632f + phiRot;
	outDir = normalize( axis * cosTheta + ( t1 * cos( phi ) + t2 * sin( phi ) ) * sinTheta );
}

// Shadow that also returns hit distance (for PCSS). Skips $vrad_filter panes.
bool OccludedT( float3 from, float3 to, out float tHit )
{
	tHit = 0;
	float3 delta = to - from;
	float dist = length( delta );
	if ( dist <= OccludeBias * 2.0f )
		return false;
	delta /= dist;
	float tmin = OccludeBias;
	float tmax = dist - min( OccludeBias, dist * 0.25f );
	if ( tmax <= tmin )
		return false;
	uint lastPrim = 0xFFFFFFFFu;
	[loop] for ( int pass = 0; pass < MAX_FILTER_HITS + 1; ++pass )
	{
		if ( tmax <= tmin + 1e-4f )
			return false;
		float t; uint flags, prim; float3 nHit;
		if ( !TraceClosest( from, delta, tmin, tmax, t, flags, prim, nHit ) )
			return false;
		if ( flags & TRACE_ID_SKY )
			return false;
#if PT_ENABLE_VOLUMES
		if ( EnvVolCount > 0 && IgnoreSkyOccluder( from, from + delta * t ) )
			return false;
#endif
		if ( ( flags & TRACE_ID_FILTER ) != 0 || IsFilterPrim( prim ) )
		{
			if ( prim == lastPrim )
			{
				tmin = FilterAdvancePast( t );
				continue;
			}
			lastPrim = prim;
			tmin = FilterAdvancePast( t );
			continue;
		}
		tHit = t;
		return true;
	}
	return false;
}

bool SunBlockedT( float3 pos, float3 axis, out float tHit )
{
	tHit = MaxTrace;
	float tmin = OccludeBias;
	uint lastPrim = 0xFFFFFFFFu;
	[loop] for ( int pass = 0; pass < MAX_FILTER_HITS + 1; ++pass )
	{
		float t; uint flags, prim; float3 nHit;
		if ( !TraceClosest( pos, axis, tmin, MaxTrace, t, flags, prim, nHit ) )
			return false;
		if ( flags & TRACE_ID_SKY )
			return false;
#if PT_ENABLE_VOLUMES
		if ( EnvVolCount > 0 && IgnoreSkyOccluder( pos, pos + axis * t ) )
			return false;
#endif
		if ( ( flags & TRACE_ID_FILTER ) != 0 || IsFilterPrim( prim ) )
		{
			if ( prim == lastPrim )
			{
				tmin = FilterAdvancePast( t );
				continue;
			}
			lastPrim = prim;
			tmin = FilterAdvancePast( t );
			continue;
		}
		tHit = t;
		return true;
	}
	return false;
}

float3 EvalLight( uint i, float3 pos, float3 normal, uint seed, int faceNum, bool allowSoft, inout float sunAmt )
{
	float4 a0 = LA0( i );
	float4 a1 = LA1( i );
	float type = a0.w;
	float3 intensity = a1.xyz;
	float3 sum = 0;

	if ( type < 0.5f || ( type > 0.5f && type < 1.5f ) )
	{
		float3 origin = a0.xyz;
		float radius = LC1( i ).z; // volumeRadius / -pt_lightradius
		float3 toC = origin - pos;
		float distC = length( toC );
		if ( distC < 1e-3f )
			return 0;
		toC /= distC;
)HLSL"
R"HLSL(
		int ns = 1;
		if ( allowSoft && radius > 0.0f && SoftSamplesMax > 1 && distC > OccludeBias + 1.0f )
		{
			float tHit = distC;
			bool blocked = OccludedT( pos, origin, tHit );
			ns = EstimatePenumbraSamples( radius, distC, tHit, blocked );
		}

		float3 diskN = -toC;
		float3 accum = 0;
		for ( int s = 0; s < ns; ++s )
		{
			float3 offset = 0;
			if ( radius > 0.0f && ns > 1 )
				SoftDiskOffset( diskN, radius, (uint)s, seed + (uint)s * 53u, offset );
			float3 lightPos = origin + offset;
			float3 delta = lightPos - pos;
			float dist = length( delta );
			if ( dist < 1e-3f )
				continue;
			delta /= dist;
			float ndl = dot( normal, delta );
			if ( ndl <= 0 )
				continue;
			if ( type > 0.5f )
			{
				float cone = SpotCone( i, delta );
				if ( cone <= 0 )
					continue;
				ndl *= cone;
			}
			float falloff = PointFalloff( i, dist );
			if ( falloff <= 0 )
				continue;
			float3 vis = VisRGB( pos, lightPos );
			if ( max( vis.x, max( vis.y, vis.z ) ) < 1e-5f )
				continue;
			accum += intensity * ( ndl * falloff ) * vis;
		}
		sum = accum * ( 1.0f / (float)ns );
	}
	else if ( type > 1.5f && type < 2.5f )
	{
#if PT_ENABLE_VOLUMES
		float wEnv = EnvWeight( LightsD[i].y, pos );
		if ( wEnv <= 0.0f )
			return 0;
#else
		float wEnv = 1.0f;
#endif

		float3 axis = normalize( LB0( i ).xyz );
		float ndl0 = dot( normal, axis );
		if ( ndl0 <= 0 )
			return 0;

		float sinExt = LC1( i ).y; // sunExtent
		int nSoft = 1;
		if ( allowSoft && sinExt > 0.0f && SoftSamplesMax > 1 )
		{
			float tHit = MaxTrace;
			bool blocked = SunBlockedT( pos, axis, tHit );
			float refDist = max( tHit, 64.0f );
			float lightSize = sinExt * refDist;
			nSoft = EstimatePenumbraSamples( lightSize, refDist, tHit, blocked );
			// legacy soft-sun style cap
			float angleDeg = degrees( asin( saturate( sinExt ) ) );
			int legacy = min( 40, 8 + (int)( angleDeg * 0.6f ) );
			nSoft = min( nSoft, legacy );
			nSoft = min( nSoft, (int)SoftSamplesMax );
			nSoft = max( nSoft, 1 );
		}

		float cosThetaMax = sqrt( max( 0.0f, 1.0f - sinExt * sinExt ) );
		float sunSolidAngle = max( 1e-6f, 2.0f * PI * ( 1.0f - cosThetaMax ) );
		float pdfLight = 1.0f / sunSolidAngle;

		float3 visAccum = 0;
		float ndlAccum = 0;
		float misAccum = 0;
		float nVis = 0;
		for ( int s = 0; s < nSoft; ++s )
		{
			float3 dir = axis;
			if ( nSoft > 1 && sinExt > 0 )
				SoftSunDir( axis, sinExt, (uint)s, (uint)nSoft, seed + (uint)s * 97u, dir );
			float ndlS = dot( normal, dir );
			if ( ndlS <= 0 )
				continue;
			float3 visS = SunVisRGB( pos, dir );
			if ( max( visS.x, max( visS.y, visS.z ) ) < 1e-5f )
				continue;
			float pdfBsdf = ndlS * INV_PI;
			float mis = ( pdfLight * pdfLight ) / max( 1e-8f, pdfLight * pdfLight + pdfBsdf * pdfBsdf );
			visAccum += visS;
			ndlAccum += ndlS;
			misAccum += mis;
			nVis += 1.0f;
		}
		if ( nVis > 0 )
		{
			float3 vis = visAccum / (float)nSoft;
			float ndlSoft = ndlAccum / nVis;
			float mis = misAccum / nVis;
			sum = intensity * ( ndlSoft * vis * mis * wEnv );
			sunAmt += ( ( vis.x + vis.y + vis.z ) * ( 1.0f / 3.0f ) ) * wEnv;
		}
	}
	else if ( type > 2.5f && type < 3.5f )
	{
		if ( faceNum >= 0 && LightsD[i].x == faceNum )
			return 0;
)HLSL"
R"HLSL(		// emit_surface / $vrad_emit*: hard center NEE + area falloff.
		// Do NOT soft-sample with radius sqrt(area/π) — that disk often sits in
		// solids for medium/large faces and zeros the light via occlusion.
		float3 origin = a0.xyz;
		float3 Lnor = normalize( LB0( i ).xyz );
		float areaR2 = max( LC1( i ).x, 0.0f );
		float softRadius = 0.0f;
		int ns = 1;
		if ( allowSoft && SoftSamplesMax > 1 )
		{
			softRadius = min( 8.0f, sqrt( areaR2 ) * 0.15f );
			if ( softRadius > 0.5f )
			{
				float3 toL = origin - pos;
				float distL = length( toL );
				if ( distL > OccludeBias + 1.0f )
				{
					toL /= distL;
					float tHit = distL;
					bool blocked = OccludedT( pos, origin, tHit );
					ns = EstimatePenumbraSamples( softRadius, distL, tHit, blocked );
				}
			}
		}
		float3 accum = 0;
		for ( int s = 0; s < ns; ++s )
		{
			float3 offset = 0;
			if ( s > 0 && softRadius > 0.5f )
				SoftDiskOffset( Lnor, softRadius, (uint)s, seed + (uint)s * 31u, offset );
			float3 lightPos = origin + offset + Lnor * 1.0f;
			float3 delta = lightPos - pos;
			float dist = length( delta );
			if ( dist < 1e-3f )
				continue;
			delta /= dist;
			float ndl = dot( normal, delta );
			if ( ndl <= 0 )
				continue;
			float emitterCos = -dot( delta, Lnor );
			if ( emitterCos <= 0 )
				continue;
			float denom = dist * dist + areaR2;
			if ( denom <= 1e-8f )
				continue;
			float4 c0 = LC0( i );
			float falloff = ApplyFade( emitterCos / denom, dist, c0.y, c0.z );
			if ( falloff <= 0 )
				continue;
			float3 vis = VisRGB( pos, lightPos );
			if ( max( vis.x, max( vis.y, vis.z ) ) < 1e-5f )
				continue;
			accum += intensity * ( ndl * falloff ) * vis;
		}
		sum = accum * ( 1.0f / (float)ns );
	}
	return sum;
}

float3 SampleSkyAmb( float3 pos )
{
#if PT_ENABLE_VOLUMES
	if ( EnvVolCount == 0 )
		return float3( SkyAmbX, SkyAmbY, SkyAmbZ );

	float3 amb = 0;
	for ( uint i = 0; i < LightCount; ++i )
	{
		float type = LA0( i ).w;
		if ( type <= 3.5f )
			continue;
		float w = EnvWeight( LightsD[i].y, pos );
		if ( w <= 0.0f )
			continue;
		amb += LA1( i ).xyz * w;
	}
	return amb;
#else
	return float3( SkyAmbX, SkyAmbY, SkyAmbZ );
#endif
}

uint SampleLocalCdf( float u )
{
	uint lo = 0, hi = LocalCount;
	while ( lo + 1 < hi )
	{
		uint mid = ( lo + hi ) >> 1;
		if ( LocalCdf[mid] < u )
			lo = mid;
		else
			hi = mid;
	}
	return lo;
}

float3 SampleDirect( float3 pos, float3 normal, uint seed, int faceNum, bool allowSoft, inout float sunAmt )
{
	float3 sum = 0;
	const bool powerSample = ( LightSamples > 0 ) && ( LocalCount > LightSamples );

	for ( uint i = 0; i < LightCount; ++i )
	{
		float type = LA0( i ).w;
		if ( type > 3.5f )
			continue;
		if ( powerSample && LightsD[i].z != 0 )
			continue;
		sum += EvalLight( i, pos, normal, seed + i * 17u, faceNum, allowSoft, sunAmt );
	}

	if ( powerSample )
	{
		for ( uint s = 0; s < LightSamples; ++s )
		{
			float u = Hash01( seed + s * 97u + 3u );
			if ( u < 1e-6f ) u = 1e-6f;
			uint li = SampleLocalCdf( u );
			uint idx = LocalIdx[li];
			float cdf0 = ( li == 0 ) ? 0.0f : LocalCdf[li - 1];
			float pdf = max( LocalCdf[li] - cdf0, 1e-8f );
			float3 col = EvalLight( idx, pos, normal, seed + ( s + 1 ) * 131u, faceNum, allowSoft, sunAmt );
			sum += col * ( 1.0f / ( pdf * (float)LightSamples ) );
		}
	}

	// PBRT textured area lights ($vrad_emit*). EmitSamples=0 → all tris; else stratified power picks.
	if ( EmitTriCount > 0 )
	{
		bool sampleAll = ( EmitSamples == 0 ) || ( EmitSamples >= EmitTriCount );
		uint k = sampleAll ? EmitTriCount : max( EmitSamples, 1u );
		float3 areaSum = 0;
		for ( uint s = 0; s < k; ++s )
		{
			uint ti;
			float pPick;
			if ( sampleAll )
			{
				ti = s;
				pPick = 1.0f;
			}
			else
			{
				float uPick = ( (float)s + Hash01( seed + s * 91u + 7u ) ) / (float)k;
				if ( uPick < 1e-6f ) uPick = 1e-6f;
				if ( uPick > 0.999999f ) uPick = 0.999999f;
				uint lo = 0, hi = EmitTriCount;
				while ( lo + 1 < hi )
				{
					uint mid = ( lo + hi ) >> 1;
					if ( EmitCdf[mid] < uPick ) lo = mid; else hi = mid;
				}
				ti = lo;
				float cdf0 = ( ti == 0 ) ? 0.0f : EmitCdf[ti - 1];
				pPick = max( EmitCdf[ti] - cdf0, 1e-12f );
			}
			float4 a0 = EmitA[ti * 4 + 0];
			float4 a1 = EmitA[ti * 4 + 1];
			float4 a2 = EmitA[ti * 4 + 2];
			float4 a3 = EmitA[ti * 4 + 3];
			int eFace = (int)a2.w;
			if ( faceNum >= 0 && eFace == faceNum )
				continue;
			float area = max( a0.w, 1e-8f );
			float u0 = Hash01( seed + s * 97u + 11u );
			float u1 = Hash01( seed + s * 131u + 23u );
			float su0 = sqrt( max( u0, 0.0f ) );
			float b0 = 1.0f - su0;
			float b1 = u1 * su0;
			float b2 = 1.0f - b0 - b1;
			float3 lp = a0.xyz * b0 + a1.xyz * b1 + a2.xyz * b2;
			float3 Le = EmitE[ti * 3 + 0].xyz * b0 + EmitE[ti * 3 + 1].xyz * b1 + EmitE[ti * 3 + 2].xyz * b2;
			float3 Ln = normalize( a3.xyz );
			lp += Ln * 1.0f;
			float3 delta = lp - pos;
			float dist = length( delta );
			if ( dist < 1e-3f ) continue;
			delta /= dist;
			float ndl = dot( normal, delta );
			if ( ndl <= 0 ) continue;
			float eCos = -dot( delta, Ln );
			if ( eCos <= 0 ) continue;
			float pdfA = 1.0f / area;
			float pdf = sampleAll ? pdfA : ( pPick * pdfA );
			if ( pdf <= 1e-20f ) continue;
			float3 vis = VisRGB( pos, lp );
			if ( max( vis.x, max( vis.y, vis.z ) ) < 1e-5f ) continue;
			areaSum += Le * ( ndl * eCos / ( dist * dist * pdf ) ) * vis;
		}
		if ( !sampleAll )
			areaSum *= ( 1.0f / (float)k );
		sum += areaSum;
	}
	return sum;
}

float3 PathLi( float3 posIn, float3 normalIn, uint seed, uint sppIndex, int faceNum, out float sunAmt )
{
	float3 radiance = 0;
	float3 throughput = 1;
	float3 pos = posIn;
	float3 normal = normalIn;
	int skipFace = faceNum;
	sunAmt = 0;
	uint pathDepth = Bounces + 1;
	if ( pathDepth < 1 ) pathDepth = 1;

	for ( uint bounce = 0; bounce < pathDepth; ++bounce )
	{
		bool allowSoft = ( bounce == 0 ) || ( SoftMode != 0 );
		float sunB = 0;
		float3 nee = SampleDirect( pos, normal, seed + bounce * 31u + sppIndex * 17u, skipFace, allowSoft, sunB );
		nee = ClampFirefly( nee, bounce == 0 ? FireflyCap : FireflyCap * 0.16f );
		radiance += throughput * nee;
		if ( bounce == 0 )
			sunAmt = sunB;

		float u1 = Hash01( seed + bounce * 13u + sppIndex * 7u + 11u );
		float u2 = Hash01( seed + bounce * 17u + sppIndex * 11u + 13u );
		float3 dir = CosineHemi( normal, u1, u2 );

		float tmin = 0.25f;
		float t = 0; uint flags = 0, prim = 0; float3 hitN = float3( 0, 0, 1 );
		bool hit = false;
		uint lastPrim = 0xFFFFFFFFu;
		uint lastFace = 0xFFFFFFFFu;
		float lastFilterT = -1e30f;
		float3 lastFilterN = float3( 0, 0, 0 );
		[loop] for ( int fpass = 0; fpass < MAX_FILTER_HITS + 1; ++fpass )
		{
			float t0; uint f0, p0; float3 n0;
			bool h0 = TraceClosest( pos, dir, tmin, MaxTrace, t0, f0, p0, n0 );
			if ( !h0 || ( f0 & TRACE_ID_SKY ) )
			{
				hit = false;
				flags = h0 ? f0 : 0;
				break;
			}
			if ( ( f0 & TRACE_ID_FILTER ) != 0 || IsFilterPrim( p0 ) )
			{
				uint faceId = ( f0 & TRACE_ID_FILTER ) != 0 ? ( f0 & 0x00FFFFFFu ) : 0xFFFFFFFEu;
				float3 faceN = FilterFaceN( p0 );
				if ( p0 == lastPrim || faceId == lastFace || FilterIsSheetExit( t0, faceN, lastFilterT, lastFilterN ) )
				{
					tmin = FilterAdvancePast( t0 );
					continue;
				}
				lastPrim = p0;
				lastFace = faceId;
				if ( FilterShouldApply( p0, dir ) )
				{
					float3 F = SampleFilterT( p0, pos + dir * t0 );
					if ( max( F.x, max( F.y, F.z ) ) < 1e-5f && ( f0 & TRACE_ID_FILTER ) != 0 )
						F = float3( 1, 1, 1 );
					throughput = OklabStackFilter( throughput, max( F, float3( 1e-4f, 1e-4f, 1e-4f ) ) );
					lastFilterT = t0;
					lastFilterN = faceN;
				}
				tmin = FilterAdvancePast( t0 );
				continue;
			}
			hit = true;
			t = t0; flags = f0; prim = p0; hitN = n0;
			break;
		}
		if ( !hit )
		{
			float3 amb = SampleSkyAmb( pos );
			radiance += ClampFirefly( throughput * amb, FireflyCap * 0.16f );
			break;
		}

		float3 hitPos = pos + dir * t;
		if ( dot( hitN, dir ) > 0 )
			hitN = -hitN;

		float3 albedo = ( prim < TriCount ) ? TriAlbedo[prim] : float3( 0.45, 0.45, 0.45 );
		albedo = ApplyBounceVolAlbedo( hitPos, albedo, bounce );
		albedo = clamp( albedo, 0.0f, 0.92f );
		throughput *= albedo;
#if PT_ENABLE_VOLUMES
		if ( EnvVolCount > 0 )
		{
			int recvEnv = DominantEnvId( posIn );
			int emitEnv = DominantEnvId( hitPos );
			TintInboundBounce( recvEnv, emitEnv, throughput );
		}
#endif

		if ( bounce >= 1 )
		{
			float q = min( 0.95f, max( throughput.x, max( throughput.y, throughput.z ) ) );
			float r = Hash01( seed + bounce * 41u + 99u + sppIndex );
			if ( r > q )
				break;
			throughput *= ( 1.0f / max( q, 1e-3f ) );
		}

		pos = hitPos + hitN * 0.5f;
		normal = hitN;
		skipFace = -1;
		if ( throughput.x + throughput.y + throughput.z < 1e-4f )
			break;
	}
	return radiance;
}

[numthreads(64, 1, 1)]
void CSMain( uint3 dtid : SV_DispatchThreadID )
{
	uint i = dtid.x;
	if ( i >= LuxelCount )
		return;

	float4 a0 = LuxelsA[i * 2 + 0];
	float4 a1 = LuxelsA[i * 2 + 1];
	float3 pos0 = a0.xyz;
	float luxelWorld = a0.w;
	float3 normal = normalize( a1.xyz );
	uint seed = asuint( a1.w );
	float3 axisU = LuxelsB[i * 2 + 0].xyz;
	float3 axisV = LuxelsB[i * 2 + 1].xyz;
	int faceNum = LuxelsC[i].x;
	int aaN = max( LuxelsC[i].y, 1 );

	float3 sum = 0;
	float sunSum = 0;
	uint spp = max( Spp, 1u );
	for ( uint s = 0; s < spp; ++s )
	{
		uint sGlobal = SampleOffset + s;
		float3 samplePos = pos0;
		if ( aaN > 1 )
		{
			int tap = (int)( sGlobal % (uint)( aaN * aaN ) );
			int tx = tap % aaN;
			int ty = tap / aaN;
			float span = 0.66f;
			float u = ( (float)tx / (float)( aaN - 1 ) - 0.5f ) * span;
			float v = ( (float)ty / (float)( aaN - 1 ) - 0.5f ) * span;
			samplePos += axisU * u + axisV * v;
		}
		float sunAmt = 0;
		float3 li = PathLi( samplePos, normal, seed, sGlobal, faceNum, sunAmt );
		li = ClampFirefly( li, FireflyCap );
		sum += li;
		sunSum += sunAmt;
	}
	float inv = 1.0f / (float)spp;
	OutRGB[i] = float4( sum * inv, sunSum * inv );
}
)HLSL";


// ---- Baker device extras (extends g_ptDev usage) ----
struct PtBakeGpu
{
	ComPtr<ID3D12RootSignature>	rootSig;
	ComPtr<ID3D12PipelineState>	pso;
	ComPtr<ID3D12DescriptorHeap> heap;
	ComPtr<ID3D12Resource>		lightsA, lightsB, lightsC, lightsD;
	ComPtr<ID3D12Resource>		localIdx, localCdf;
	ComPtr<ID3D12Resource>		emitA, emitE, emitCdf;
	ComPtr<ID3D12Resource>		envVols;
	ComPtr<ID3D12Resource>		bounceVols;
	ComPtr<ID3D12Resource>		albedo;
	ComPtr<ID3D12Resource>		triFilter; // FilterMeta: 5 float4 / tri
	ComPtr<ID3D12Resource>		filterAtlas; // Texture2DArray
	uint32						filterAtlasW = 1;
	uint32						filterAtlasH = 1;
	uint32						filterAtlasLayers = 1;
	uint32						localCount = 0;
	uint32						emitTriCount = 0;
	uint32						envVolCount = 0;
	uint32						bounceVolCount = 0;
	uint32						sampleOffset = 0;
	ComPtr<ID3D12Resource>		luxelA, luxelB, luxelC;
	ComPtr<ID3D12Resource>		luxelAUp, luxelBUp, luxelCUp;
	ComPtr<ID3D12Resource>		outBuf, outRead;
	ComPtr<ID3D12Resource>		vertSrvUpload; // unused — verts already on GPU
	uint32						luxelCap = 0;
	uint32						lightCount = 0;
	uint32						triCount = 0;
	PtGpuBakeParams				params = {};
	bool						active = false;
};
static PtBakeGpu g_ptBake;

static bool PtCompileCSNamed( const char *src, const char *entry, bool enableVolumes, ID3DBlob **ppBlob )
{
	// Reuse PtCompileCS pattern with custom source — local copy
	if ( !g_hDxc )
	{
		g_hDxil = LoadLibraryA( "dxil.dll" );
		g_hDxc = LoadLibraryA( "dxcompiler.dll" );
		if ( !g_hDxc )
			return false;
	}
	using DxcCreateInstanceFn = HRESULT( __stdcall * )( REFCLSID, REFIID, LPVOID * );
	auto create = (DxcCreateInstanceFn)GetProcAddress( g_hDxc, "DxcCreateInstance" );
	if ( !create )
		return false;
	ComPtr<IDxcUtils> utils;
	ComPtr<IDxcCompiler3> compiler;
	if ( FAILED( create( CLSID_DxcUtils, IID_PPV_ARGS( &utils ) ) ) )
		return false;
	if ( FAILED( create( CLSID_DxcCompiler, IID_PPV_ARGS( &compiler ) ) ) )
		return false;
	ComPtr<IDxcBlobEncoding> source;
	if ( FAILED( utils->CreateBlob( src, (UINT32)strlen( src ), CP_UTF8, &source ) ) )
		return false;
	// PT_ENABLE_VOLUMES=0 strips env/bounce vol helpers at DXC preprocess time so the
	// no-volume case does not pay register pressure from dead volume code.
	LPCWSTR args[] = {
		L"-E", L"CSMain", L"-T", L"cs_6_5", L"-HV", L"2021",
		enableVolumes ? L"-DPT_ENABLE_VOLUMES=1" : L"-DPT_ENABLE_VOLUMES=0"
	};
	(void)entry;
	DxcBuffer buf = {};
	buf.Ptr = source->GetBufferPointer();
	buf.Size = source->GetBufferSize();
	buf.Encoding = DXC_CP_UTF8;
	ComPtr<IDxcResult> result;
	if ( FAILED( compiler->Compile( &buf, args, _countof( args ), nullptr, IID_PPV_ARGS( &result ) ) ) )
		return false;
	HRESULT hrStatus = S_OK;
	result->GetStatus( &hrStatus );
	if ( FAILED( hrStatus ) )
	{
		ComPtr<IDxcBlobUtf8> errors;
		result->GetOutput( DXC_OUT_ERRORS, IID_PPV_ARGS( &errors ), nullptr );
		if ( errors && errors->GetStringLength() )
			Warning( "[PathTrace-DXR] Bake CS compile error:\n%s\n", errors->GetStringPointer() );
		return false;
	}
	ComPtr<IDxcBlob> dxil;
	result->GetOutput( DXC_OUT_OBJECT, IID_PPV_ARGS( &dxil ), nullptr );
	if ( !dxil )
		return false;
	ID3DBlob *blob = nullptr;
	if ( FAILED( D3DCreateBlob( dxil->GetBufferSize(), &blob ) ) )
		return false;
	memcpy( blob->GetBufferPointer(), dxil->GetBufferPointer(), dxil->GetBufferSize() );
	*ppBlob = blob;
	return true;
}

static bool PtEnsureLuxelCap( uint32 n )
{
	if ( n <= g_ptBake.luxelCap && g_ptBake.luxelA )
		return true;
	g_ptBake.luxelA.Reset(); g_ptBake.luxelB.Reset(); g_ptBake.luxelC.Reset();
	g_ptBake.luxelAUp.Reset(); g_ptBake.luxelBUp.Reset(); g_ptBake.luxelCUp.Reset();
	g_ptBake.outBuf.Reset(); g_ptBake.outRead.Reset();
	uint32 cap = ( n < 8192u ) ? 8192u : n;
	UINT64 aBytes = (UINT64)cap * 2 * sizeof( float ) * 4;
	UINT64 cBytes = (UINT64)cap * sizeof( int ) * 4;
	UINT64 oBytes = (UINT64)cap * sizeof( float ) * 4;
	if ( !PtCreateBuffer( aBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &g_ptBake.luxelAUp, L"PT_LuxAUp" ) ) return false;
	if ( !PtCreateBuffer( aBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &g_ptBake.luxelBUp, L"PT_LuxBUp" ) ) return false;
	if ( !PtCreateBuffer( cBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &g_ptBake.luxelCUp, L"PT_LuxCUp" ) ) return false;
	if ( !PtCreateBuffer( aBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.luxelA, L"PT_LuxA" ) ) return false;
	if ( !PtCreateBuffer( aBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.luxelB, L"PT_LuxB" ) ) return false;
	if ( !PtCreateBuffer( cBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.luxelC, L"PT_LuxC" ) ) return false;
	if ( !PtCreateBuffer( oBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &g_ptBake.outBuf, L"PT_Out" ) ) return false;
	if ( !PtCreateBuffer( oBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.outRead, L"PT_OutRB" ) ) return false;
	g_ptBake.luxelCap = cap;
	return true;
}

uint32_t PathTraceDXR_CapturedTriCount()
{
	return (uint32_t)g_ptTris.size();
}

uint32_t PathTraceDXR_CapturedTriFlags( uint32_t triIndex )
{
	if ( triIndex >= (uint32_t)g_ptTris.size() )
		return 0;
	return g_ptTris[triIndex].flags;
}

void PathTraceDXR_GpuBakeEnd()
{
	g_ptBake.active = false;
	g_ptBake.rootSig.Reset();
	g_ptBake.pso.Reset();
	g_ptBake.heap.Reset();
	g_ptBake.lightsA.Reset(); g_ptBake.lightsB.Reset(); g_ptBake.lightsC.Reset(); g_ptBake.lightsD.Reset();
	g_ptBake.localIdx.Reset(); g_ptBake.localCdf.Reset(); g_ptBake.localCount = 0; g_ptBake.sampleOffset = 0;
	g_ptBake.emitA.Reset(); g_ptBake.emitE.Reset(); g_ptBake.emitCdf.Reset(); g_ptBake.emitTriCount = 0;
	g_ptBake.envVols.Reset(); g_ptBake.envVolCount = 0;
	g_ptBake.bounceVols.Reset(); g_ptBake.bounceVolCount = 0;
	g_ptBake.albedo.Reset();
	g_ptBake.triFilter.Reset();
	g_ptBake.filterAtlas.Reset();
	g_ptBake.filterAtlasW = 1;
	g_ptBake.filterAtlasH = 1;
	g_ptBake.filterAtlasLayers = 1;
	g_ptBake.luxelA.Reset(); g_ptBake.luxelB.Reset(); g_ptBake.luxelC.Reset();
	g_ptBake.luxelAUp.Reset(); g_ptBake.luxelBUp.Reset(); g_ptBake.luxelCUp.Reset();
	g_ptBake.outBuf.Reset(); g_ptBake.outRead.Reset();
	g_ptBake.luxelCap = 0;
	g_ptBake.lightCount = 0;
	g_ptBake.triCount = 0;
}

bool PathTraceDXR_GpuBakeBegin( const PtGpuBakeLight *lights, uint32_t nLights,
								const float *albedoRGB, uint32_t nTris,
								const PtGpuBakeParams &params,
								const PtGpuEmitTri *emitTris, uint32_t nEmitTris,
								const float *emitCdf,
								const float *filterMeta,
								const unsigned char *filterArrayRGBA,
								uint32_t filterArrayW, uint32_t filterArrayH,
								uint32_t filterArrayLayers )
{
	PathTraceDXR_GpuBakeEnd();
	if ( !g_ptDev.ready || !lights || !albedoRGB || nTris == 0 || nTris != (uint32)g_ptTris.size() )
		return false;

	std::lock_guard<std::mutex> lock( g_ptDev.mutex );

	const bool enableVolumes = ( LightEnv_VolumeCount() > 0 ) || ( BounceVol_VolumeCount() > 0 );
	ID3DBlob *csBlob = nullptr;
	if ( !PtCompileCSNamed( kPtLuxelBakeCS, "CSMain", enableVolumes, &csBlob ) )
		return false;
	Msg( "[PathTrace-DXR] GPU luxel CS: %s\n", enableVolumes ? "volumes" : "lean (no volumes)" );

	// Root: table t0..t19 + u0, then 24 constants
	D3D12_DESCRIPTOR_RANGE ranges[2] = {};
	ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	ranges[0].NumDescriptors = 20;
	ranges[0].BaseShaderRegister = 0;
	ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
	ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	ranges[1].NumDescriptors = 1;
	ranges[1].BaseShaderRegister = 0;
	ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

	D3D12_ROOT_PARAMETER paramsRS[2] = {};
	paramsRS[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	paramsRS[0].DescriptorTable.NumDescriptorRanges = 2;
	paramsRS[0].DescriptorTable.pDescriptorRanges = ranges;
	paramsRS[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	paramsRS[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	paramsRS[1].Constants.Num32BitValues = 24;
	paramsRS[1].Constants.ShaderRegister = 0;
	paramsRS[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
	rsDesc.NumParameters = 2;
	rsDesc.pParameters = paramsRS;
	ComPtr<ID3DBlob> rsBlob, rsErr;
	if ( FAILED( D3D12SerializeRootSignature( &rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr ) ) )
	{
		csBlob->Release();
		return false;
	}
	if ( FAILED( g_ptDev.device->CreateRootSignature( 0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
													  IID_PPV_ARGS( &g_ptBake.rootSig ) ) ) )
	{
		csBlob->Release();
		return false;
	}

	D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = g_ptBake.rootSig.Get();
	psoDesc.CS.pShaderBytecode = csBlob->GetBufferPointer();
	psoDesc.CS.BytecodeLength = csBlob->GetBufferSize();
	HRESULT hr = g_ptDev.device->CreateComputePipelineState( &psoDesc, IID_PPV_ARGS( &g_ptBake.pso ) );
	csBlob->Release();
	if ( FAILED( hr ) )
	{
		Warning( "[PathTrace-DXR] Bake PSO failed (0x%08X)\n", (unsigned)hr );
		return false;
	}

	D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heapDesc.NumDescriptors = 24;
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if ( FAILED( g_ptDev.device->CreateDescriptorHeap( &heapDesc, IID_PPV_ARGS( &g_ptBake.heap ) ) ) )
		return false;

	// Upload lights (2 float4 each for A/B/C) + int4 D + albedo + filter meta
	const UINT64 lightF4 = (UINT64)nLights * 2 * sizeof( float ) * 4;
	const UINT64 lightI4 = (UINT64)nLights * sizeof( int ) * 4;
	const UINT64 albBytes = (UINT64)nTris * sizeof( float ) * 3;
	const UINT64 filtMetaBytes = (UINT64)nTris * 5u * sizeof( float ) * 4;

	ComPtr<ID3D12Resource> upA, upB, upC, upD, upAlb, upFilt;
	if ( !PtCreateBuffer( lightF4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upA, L"upLA" ) ) return false;
	if ( !PtCreateBuffer( lightF4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upB, L"upLB" ) ) return false;
	if ( !PtCreateBuffer( lightF4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upC, L"upLC" ) ) return false;
	if ( !PtCreateBuffer( lightI4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upD, L"upLD" ) ) return false;
	if ( !PtCreateBuffer( albBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upAlb, L"upAlb" ) ) return false;
	if ( !PtCreateBuffer( filtMetaBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upFilt, L"upFiltMeta" ) ) return false;
	if ( !PtCreateBuffer( lightF4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.lightsA, L"LA" ) ) return false;
	if ( !PtCreateBuffer( lightF4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.lightsB, L"LB" ) ) return false;
	if ( !PtCreateBuffer( lightF4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.lightsC, L"LC" ) ) return false;
	if ( !PtCreateBuffer( lightI4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.lightsD, L"LD" ) ) return false;
	if ( !PtCreateBuffer( albBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.albedo, L"Alb" ) ) return false;
	if ( !PtCreateBuffer( filtMetaBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.triFilter, L"FiltMeta" ) ) return false;

	{
		float *pa = nullptr, *pb = nullptr, *pc = nullptr; int *pd = nullptr; float *pal = nullptr; float *pf = nullptr;
		upA->Map( 0, nullptr, (void **)&pa );
		upB->Map( 0, nullptr, (void **)&pb );
		upC->Map( 0, nullptr, (void **)&pc );
		upD->Map( 0, nullptr, (void **)&pd );
		upAlb->Map( 0, nullptr, (void **)&pal );
		upFilt->Map( 0, nullptr, (void **)&pf );
		for ( uint32_t i = 0; i < nLights; ++i )
		{
			const PtGpuBakeLight &L = lights[i];
			pa[i * 8 + 0] = L.origin[0]; pa[i * 8 + 1] = L.origin[1]; pa[i * 8 + 2] = L.origin[2]; pa[i * 8 + 3] = L.type;
			pa[i * 8 + 4] = L.intensity[0]; pa[i * 8 + 5] = L.intensity[1]; pa[i * 8 + 6] = L.intensity[2]; pa[i * 8 + 7] = 0;
			pb[i * 8 + 0] = L.dir[0]; pb[i * 8 + 1] = L.dir[1]; pb[i * 8 + 2] = L.dir[2]; pb[i * 8 + 3] = L.stopdot;
			pb[i * 8 + 4] = L.stopdot2; pb[i * 8 + 5] = L.exponent; pb[i * 8 + 6] = L.constant_attn; pb[i * 8 + 7] = L.linear_attn;
			pc[i * 8 + 0] = L.quadratic_attn; pc[i * 8 + 1] = L.fadeStart; pc[i * 8 + 2] = L.fadeEnd; pc[i * 8 + 3] = L.capDist;
			pc[i * 8 + 4] = L.areaR2; pc[i * 8 + 5] = L.sunExtent; pc[i * 8 + 6] = L.volumeRadius; pc[i * 8 + 7] = L.power;
			pd[i * 4 + 0] = L.facenum; pd[i * 4 + 1] = L.envId; pd[i * 4 + 2] = L.isLocal; pd[i * 4 + 3] = 0;
		}
		memcpy( pal, albedoRGB, (size_t)albBytes );
		if ( filterMeta )
			memcpy( pf, filterMeta, (size_t)filtMetaBytes );
		else
			memset( pf, 0, (size_t)filtMetaBytes );
		upA->Unmap( 0, nullptr ); upB->Unmap( 0, nullptr ); upC->Unmap( 0, nullptr );
		upD->Unmap( 0, nullptr ); upAlb->Unmap( 0, nullptr ); upFilt->Unmap( 0, nullptr );
	}

	// Filter Texture2DArray (1x1x1 white if unused) — upload kept until GPU copy below
	ComPtr<ID3D12Resource> upArray;
	{
		const uint32 aw = max( filterArrayW, 1u );
		const uint32 ah = max( filterArrayH, 1u );
		const uint32 nLayers = max( filterArrayLayers, 1u );
		g_ptBake.filterAtlasW = aw;
		g_ptBake.filterAtlasH = ah;
		g_ptBake.filterAtlasLayers = nLayers;
		const UINT64 rowPitch = ( (UINT64)aw * 4u + 255u ) & ~255ull;
		const UINT64 slicePitch = rowPitch * ah;
		const UINT64 arrayBytes = slicePitch * nLayers;

		D3D12_HEAP_PROPERTIES hp = {};
		hp.Type = D3D12_HEAP_TYPE_DEFAULT;
		D3D12_RESOURCE_DESC rd = {};
		rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		rd.Width = aw;
		rd.Height = ah;
		rd.DepthOrArraySize = (UINT16)nLayers;
		rd.MipLevels = 1;
		rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		rd.SampleDesc.Count = 1;
		rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		if ( FAILED( g_ptDev.device->CreateCommittedResource( &hp, D3D12_HEAP_FLAG_NONE, &rd,
															  D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
															  IID_PPV_ARGS( &g_ptBake.filterAtlas ) ) ) )
			return false;

		if ( !PtCreateBuffer( arrayBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
							  D3D12_RESOURCE_STATE_GENERIC_READ, &upArray, L"upFiltArray" ) )
			return false;
		{
			unsigned char *p = nullptr;
			upArray->Map( 0, nullptr, (void **)&p );
			memset( p, 255, (size_t)arrayBytes );
			if ( filterArrayRGBA && filterArrayW > 0 && filterArrayH > 0 && filterArrayLayers > 0 )
			{
				for ( uint32 layer = 0; layer < nLayers; ++layer )
				{
					const unsigned char *srcLayer = filterArrayRGBA
						+ (size_t)layer * (size_t)aw * (size_t)ah * 4u;
					unsigned char *dstLayer = p + layer * slicePitch;
					for ( uint32 y = 0; y < ah; ++y )
						memcpy( dstLayer + y * rowPitch, srcLayer + (size_t)y * (size_t)aw * 4u, (size_t)aw * 4u );
				}
			}
			upArray->Unmap( 0, nullptr );
		}
	}

	// Local light CDF for -pt_lights power sampling
	std::vector<uint32> localIdx;
	std::vector<float> localCdf;
	localIdx.reserve( nLights );
	float powerSum = 0.0f;
	for ( uint32_t i = 0; i < nLights; ++i )
	{
		if ( !lights[i].isLocal )
			continue;
		float p = lights[i].power;
		if ( p < 1e-8f )
			p = 1e-8f;
		powerSum += p;
		localIdx.push_back( i );
		localCdf.push_back( powerSum );
	}
	if ( powerSum > 0.0f )
	{
		for ( size_t i = 0; i < localCdf.size(); ++i )
			localCdf[i] /= powerSum;
	}
	if ( localIdx.empty() )
	{
		localIdx.push_back( 0 );
		localCdf.push_back( 1.0f );
	}
	g_ptBake.localCount = (uint32)localIdx.size();

	const UINT64 locBytes = (UINT64)localIdx.size() * sizeof( uint32 );
	const UINT64 cdfBytes = (UINT64)localCdf.size() * sizeof( float );
	ComPtr<ID3D12Resource> upLoc, upCdf;
	if ( !PtCreateBuffer( locBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upLoc, L"upLoc" ) ) return false;
	if ( !PtCreateBuffer( cdfBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upCdf, L"upCdf" ) ) return false;
	if ( !PtCreateBuffer( locBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.localIdx, L"LocIdx" ) ) return false;
	if ( !PtCreateBuffer( cdfBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.localCdf, L"LocCdf" ) ) return false;
	{
		void *p = nullptr;
		upLoc->Map( 0, nullptr, &p );
		memcpy( p, localIdx.data(), (size_t)locBytes );
		upLoc->Unmap( 0, nullptr );
		upCdf->Map( 0, nullptr, &p );
		memcpy( p, localCdf.data(), (size_t)cdfBytes );
		upCdf->Unmap( 0, nullptr );
	}

	// $vrad_emit textured area tris (dummy 1-entry buffers if none)
	const uint32 nEmit = ( emitTris && nEmitTris > 0 && emitCdf ) ? nEmitTris : 0;
	g_ptBake.emitTriCount = nEmit;
	const uint32 nEmitUpload = max( nEmit, 1u );
	const UINT64 emitABytes = (UINT64)nEmitUpload * 4 * sizeof( float ) * 4;
	const UINT64 emitEBytes = (UINT64)nEmitUpload * 3 * sizeof( float ) * 4;
	const UINT64 emitCdfBytes = (UINT64)nEmitUpload * sizeof( float );
	ComPtr<ID3D12Resource> upEmitA, upEmitE, upEmitCdf;
	if ( !PtCreateBuffer( emitABytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upEmitA, L"upEmitA" ) ) return false;
	if ( !PtCreateBuffer( emitEBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upEmitE, L"upEmitE" ) ) return false;
	if ( !PtCreateBuffer( emitCdfBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upEmitCdf, L"upEmitCdf" ) ) return false;
	if ( !PtCreateBuffer( emitABytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.emitA, L"EmitA" ) ) return false;
	if ( !PtCreateBuffer( emitEBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.emitE, L"EmitE" ) ) return false;
	if ( !PtCreateBuffer( emitCdfBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.emitCdf, L"EmitCdf" ) ) return false;
	{
		float *pa = nullptr, *pe = nullptr, *pc = nullptr;
		upEmitA->Map( 0, nullptr, (void **)&pa );
		upEmitE->Map( 0, nullptr, (void **)&pe );
		upEmitCdf->Map( 0, nullptr, (void **)&pc );
		memset( pa, 0, (size_t)emitABytes );
		memset( pe, 0, (size_t)emitEBytes );
		memset( pc, 0, (size_t)emitCdfBytes );
		if ( nEmit > 0 )
		{
			for ( uint32 i = 0; i < nEmit; ++i )
			{
				const PtGpuEmitTri &T = emitTris[i];
				pa[i * 16 + 0] = T.v0[0]; pa[i * 16 + 1] = T.v0[1]; pa[i * 16 + 2] = T.v0[2]; pa[i * 16 + 3] = T.area;
				pa[i * 16 + 4] = T.v1[0]; pa[i * 16 + 5] = T.v1[1]; pa[i * 16 + 6] = T.v1[2]; pa[i * 16 + 7] = T.power;
				pa[i * 16 + 8] = T.v2[0]; pa[i * 16 + 9] = T.v2[1]; pa[i * 16 + 10] = T.v2[2]; pa[i * 16 + 11] = T.facenum;
				pa[i * 16 + 12] = T.n[0]; pa[i * 16 + 13] = T.n[1]; pa[i * 16 + 14] = T.n[2]; pa[i * 16 + 15] = 0;
				pe[i * 12 + 0] = T.e0[0]; pe[i * 12 + 1] = T.e0[1]; pe[i * 12 + 2] = T.e0[2]; pe[i * 12 + 3] = 0;
				pe[i * 12 + 4] = T.e1[0]; pe[i * 12 + 5] = T.e1[1]; pe[i * 12 + 6] = T.e1[2]; pe[i * 12 + 7] = 0;
				pe[i * 12 + 8] = T.e2[0]; pe[i * 12 + 9] = T.e2[1]; pe[i * 12 + 10] = T.e2[2]; pe[i * 12 + 11] = 0;
				pc[i] = emitCdf[i];
			}
		}
		else
		{
			pc[0] = 1.0f;
		}
		upEmitA->Unmap( 0, nullptr );
		upEmitE->Unmap( 0, nullptr );
		upEmitCdf->Unmap( 0, nullptr );
	}

	// light_env_vol AABBs for GPU EnvWeight / shadow filters
	const uint32 nVol = (uint32)max( 0, LightEnv_VolumeCount() );
	g_ptBake.envVolCount = nVol;
	const uint32 nVolUpload = max( nVol, 1u );
	const UINT64 envBytes = (UINT64)nVolUpload * 4 * sizeof( float ) * 4;
	ComPtr<ID3D12Resource> upEnv;
	if ( !PtCreateBuffer( envBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upEnv, L"upEnv" ) ) return false;
	if ( !PtCreateBuffer( envBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.envVols, L"EnvVols" ) ) return false;
	{
		float *pe = nullptr;
		upEnv->Map( 0, nullptr, (void **)&pe );
		memset( pe, 0, (size_t)envBytes );
		for ( uint32 i = 0; i < nVol; ++i )
		{
			const LightEnvVolumeInfo_t *v = LightEnv_GetVolumeInfo( (int)i + 1 );
			if ( !v ) continue;
			pe[i * 16 + 0] = v->mins.x; pe[i * 16 + 1] = v->mins.y; pe[i * 16 + 2] = v->mins.z; pe[i * 16 + 3] = v->blendDistance;
			pe[i * 16 + 4] = v->maxs.x; pe[i * 16 + 5] = v->maxs.y; pe[i * 16 + 6] = v->maxs.z; pe[i * 16 + 7] = (float)v->blendMode;
			pe[i * 16 + 8] = (float)v->priority; pe[i * 16 + 9] = v->volumeSize; pe[i * 16 + 10] = (float)v->envId;
			int flags = ( v->bOutsideCastShadow ? 1 : 0 ) | ( v->bInsideCastShadow ? 2 : 0 )
				| ( v->bInboundBounceUsesVolumeColor ? 4 : 0 ) | ( v->bInboundBounceUsesVolumeBrightness ? 8 : 0 );
			pe[i * 16 + 11] = (float)flags;
			pe[i * 16 + 12] = v->bounceTint.x; pe[i * 16 + 13] = v->bounceTint.y; pe[i * 16 + 14] = v->bounceTint.z;
			pe[i * 16 + 15] = v->bounceIntensity;
		}
		upEnv->Unmap( 0, nullptr );
	}

	const uint32 nBVol = (uint32)max( 0, BounceVol_VolumeCount() );
	g_ptBake.bounceVolCount = nBVol;
	const uint32 nBVolUpload = max( nBVol, 1u );
	const UINT64 bvolBytes = (UINT64)nBVolUpload * 3 * sizeof( float ) * 4;
	ComPtr<ID3D12Resource> upBVol;
	if ( !PtCreateBuffer( bvolBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &upBVol, L"upBVol" ) ) return false;
	if ( !PtCreateBuffer( bvolBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &g_ptBake.bounceVols, L"BounceVols" ) ) return false;
	{
		float *pb = nullptr;
		upBVol->Map( 0, nullptr, (void **)&pb );
		memset( pb, 0, (size_t)bvolBytes );
		for ( uint32 i = 0; i < nBVol; ++i )
		{
			BounceVolDesc_t v;
			if ( !BounceVol_GetVolume( (int)i, v ) ) continue;
			pb[i * 12 + 0] = v.mins.x; pb[i * 12 + 1] = v.mins.y; pb[i * 12 + 2] = v.mins.z; pb[i * 12 + 3] = v.blendDistance;
			pb[i * 12 + 4] = v.maxs.x; pb[i * 12 + 5] = v.maxs.y; pb[i * 12 + 6] = v.maxs.z; pb[i * 12 + 7] = (float)v.blendMode;
			pb[i * 12 + 8] = (float)v.priority; pb[i * 12 + 9] = v.volumeSize;
			pb[i * 12 + 10] = v.bounceBoost; pb[i * 12 + 11] = v.bounceChroma;
		}
		upBVol->Unmap( 0, nullptr );
	}

	g_ptDev.alloc->Reset();
	g_ptDev.list->Reset( g_ptDev.alloc.Get(), nullptr );
	g_ptDev.list->CopyResource( g_ptBake.lightsA.Get(), upA.Get() );
	g_ptDev.list->CopyResource( g_ptBake.lightsB.Get(), upB.Get() );
	g_ptDev.list->CopyResource( g_ptBake.lightsC.Get(), upC.Get() );
	g_ptDev.list->CopyResource( g_ptBake.lightsD.Get(), upD.Get() );
	g_ptDev.list->CopyResource( g_ptBake.albedo.Get(), upAlb.Get() );
	g_ptDev.list->CopyResource( g_ptBake.triFilter.Get(), upFilt.Get() );
	{
		const UINT64 rowPitch = ( (UINT64)g_ptBake.filterAtlasW * 4u + 255u ) & ~255ull;
		const UINT64 slicePitch = rowPitch * g_ptBake.filterAtlasH;
		for ( uint32 layer = 0; layer < g_ptBake.filterAtlasLayers; ++layer )
		{
			D3D12_TEXTURE_COPY_LOCATION dst = {};
			dst.pResource = g_ptBake.filterAtlas.Get();
			dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			dst.SubresourceIndex = layer;
			D3D12_TEXTURE_COPY_LOCATION src = {};
			src.pResource = upArray.Get();
			src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
			src.PlacedFootprint.Offset = slicePitch * layer;
			src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			src.PlacedFootprint.Footprint.Width = g_ptBake.filterAtlasW;
			src.PlacedFootprint.Footprint.Height = g_ptBake.filterAtlasH;
			src.PlacedFootprint.Footprint.Depth = 1;
			src.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;
			g_ptDev.list->CopyTextureRegion( &dst, 0, 0, 0, &src, nullptr );
		}
	}
	g_ptDev.list->CopyResource( g_ptBake.localIdx.Get(), upLoc.Get() );
	g_ptDev.list->CopyResource( g_ptBake.localCdf.Get(), upCdf.Get() );
	g_ptDev.list->CopyResource( g_ptBake.emitA.Get(), upEmitA.Get() );
	g_ptDev.list->CopyResource( g_ptBake.emitE.Get(), upEmitE.Get() );
	g_ptDev.list->CopyResource( g_ptBake.emitCdf.Get(), upEmitCdf.Get() );
	g_ptDev.list->CopyResource( g_ptBake.envVols.Get(), upEnv.Get() );
	g_ptDev.list->CopyResource( g_ptBake.bounceVols.Get(), upBVol.Get() );
	D3D12_RESOURCE_BARRIER bars[14] = {};
	for ( int i = 0; i < 14; ++i )
	{
		bars[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		bars[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		bars[i].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
		bars[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	}
	bars[0].Transition.pResource = g_ptBake.lightsA.Get();
	bars[1].Transition.pResource = g_ptBake.lightsB.Get();
	bars[2].Transition.pResource = g_ptBake.lightsC.Get();
	bars[3].Transition.pResource = g_ptBake.lightsD.Get();
	bars[4].Transition.pResource = g_ptBake.albedo.Get();
	bars[5].Transition.pResource = g_ptBake.triFilter.Get();
	bars[6].Transition.pResource = g_ptBake.filterAtlas.Get();
	bars[7].Transition.pResource = g_ptBake.localIdx.Get();
	bars[8].Transition.pResource = g_ptBake.localCdf.Get();
	bars[9].Transition.pResource = g_ptBake.emitA.Get();
	bars[10].Transition.pResource = g_ptBake.emitE.Get();
	bars[11].Transition.pResource = g_ptBake.emitCdf.Get();
	bars[12].Transition.pResource = g_ptBake.envVols.Get();
	bars[13].Transition.pResource = g_ptBake.bounceVols.Get();
	g_ptDev.list->ResourceBarrier( 14, bars );
	g_ptDev.list->Close();
	ID3D12CommandList *lists[] = { g_ptDev.list.Get() };
	g_ptDev.queue->ExecuteCommandLists( 1, lists );
	PtWaitGPU();

	g_ptBake.lightCount = nLights;
	g_ptBake.triCount = nTris;
	g_ptBake.params = params;
	g_ptBake.params.emitTriCount = nEmit;
	g_ptBake.params.envVolCount = nVol;
	g_ptBake.params.defaultBounceIntensity = LightEnv_GetDefaultBounceIntensity();
	g_ptBake.params.bounceVolCount = nBVol;
	g_ptBake.params.cliBounceBoost = g_flBounceBoost;
	g_ptBake.params.cliBounceChroma = g_flBounceChroma;
	g_ptBake.sampleOffset = 0;
	g_ptBake.active = true;
	Msg( "[PathTrace-DXR] GPU luxel baker ready (%u lights, %u locals, %u emit-tris, %u env vols, %u bounce vols, %u scene tris, pt_lights=%u).\n",
		 nLights, g_ptBake.localCount, nEmit, nVol, nBVol, nTris, params.lightSamples );
	fflush( stdout );
	return true;
}

void PathTraceDXR_GpuBakeConfigurePass( uint32_t spp, uint32_t sampleOffset )
{
	if ( spp < 1 )
		spp = 1;
	g_ptBake.params.spp = spp;
	g_ptBake.sampleOffset = sampleOffset;
}

bool PathTraceDXR_GpuBakeLuxels( const PtGpuBakeLuxel *luxels, uint32_t nLuxels,
								 PtGpuBakeResult *outResults )
{
	if ( !g_ptBake.active || !g_ptDev.ready || !luxels || !outResults || nLuxels == 0 )
		return false;

	std::lock_guard<std::mutex> lock( g_ptDev.mutex );
	if ( !PtEnsureLuxelCap( nLuxels ) )
		return false;

	{
		float *pa = nullptr, *pb = nullptr; int *pc = nullptr;
		g_ptBake.luxelAUp->Map( 0, nullptr, (void **)&pa );
		g_ptBake.luxelBUp->Map( 0, nullptr, (void **)&pb );
		g_ptBake.luxelCUp->Map( 0, nullptr, (void **)&pc );
		for ( uint32_t i = 0; i < nLuxels; ++i )
		{
			const PtGpuBakeLuxel &L = luxels[i];
			pa[i * 8 + 0] = L.pos[0]; pa[i * 8 + 1] = L.pos[1]; pa[i * 8 + 2] = L.pos[2]; pa[i * 8 + 3] = L.luxelWorld;
			pa[i * 8 + 4] = L.normal[0]; pa[i * 8 + 5] = L.normal[1]; pa[i * 8 + 6] = L.normal[2];
			*(uint32 *)&pa[i * 8 + 7] = L.seed;
			pb[i * 8 + 0] = L.axisU[0]; pb[i * 8 + 1] = L.axisU[1]; pb[i * 8 + 2] = L.axisU[2]; pb[i * 8 + 3] = 0;
			pb[i * 8 + 4] = L.axisV[0]; pb[i * 8 + 5] = L.axisV[1]; pb[i * 8 + 6] = L.axisV[2]; pb[i * 8 + 7] = 0;
			pc[i * 4 + 0] = L.faceNum; pc[i * 4 + 1] = L.aaN; pc[i * 4 + 2] = 0; pc[i * 4 + 3] = 0;
		}
		g_ptBake.luxelAUp->Unmap( 0, nullptr );
		g_ptBake.luxelBUp->Unmap( 0, nullptr );
		g_ptBake.luxelCUp->Unmap( 0, nullptr );
	}

	g_ptDev.alloc->Reset();
	g_ptDev.list->Reset( g_ptDev.alloc.Get(), nullptr );
	g_ptDev.list->CopyResource( g_ptBake.luxelA.Get(), g_ptBake.luxelAUp.Get() );
	g_ptDev.list->CopyResource( g_ptBake.luxelB.Get(), g_ptBake.luxelBUp.Get() );
	g_ptDev.list->CopyResource( g_ptBake.luxelC.Get(), g_ptBake.luxelCUp.Get() );

	D3D12_RESOURCE_BARRIER toSrv[3] = {};
	for ( int i = 0; i < 3; ++i )
	{
		toSrv[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		toSrv[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		toSrv[i].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
		toSrv[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	}
	toSrv[0].Transition.pResource = g_ptBake.luxelA.Get();
	toSrv[1].Transition.pResource = g_ptBake.luxelB.Get();
	toSrv[2].Transition.pResource = g_ptBake.luxelC.Get();
	g_ptDev.list->ResourceBarrier( 3, toSrv );

	const UINT incr = g_ptDev.device->GetDescriptorHandleIncrementSize( D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV );
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_ptBake.heap->GetCPUDescriptorHandleForHeapStart();
	auto at = [&]( UINT idx ) { D3D12_CPU_DESCRIPTOR_HANDLE h = cpu; h.ptr += incr * idx; return h; };

	// t0 TLAS
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC tlasSrv = {};
		tlasSrv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
		tlasSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		tlasSrv.RaytracingAccelerationStructure.Location = g_ptDev.tlasVA;
		g_ptDev.device->CreateShaderResourceView( nullptr, &tlasSrv, at( 0 ) );
	}
	// t1 verts
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_UNKNOWN;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.NumElements = g_ptBake.triCount * 3;
		srv.Buffer.StructureByteStride = sizeof( float ) * 3;
		g_ptDev.device->CreateShaderResourceView( g_ptDev.vertexBuffer.Get(), &srv, at( 1 ) );
	}
	// t2 flags raw
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_R32_TYPELESS;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.NumElements = g_ptBake.triCount;
		srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
		g_ptDev.device->CreateShaderResourceView( g_ptDev.flagBuffer.Get(), &srv, at( 2 ) );
	}
	auto makeF3 = [&]( ID3D12Resource *res, UINT n, UINT slot ) {
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_UNKNOWN;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.NumElements = n;
		srv.Buffer.StructureByteStride = sizeof( float ) * 3;
		g_ptDev.device->CreateShaderResourceView( res, &srv, at( slot ) );
	};
	auto makeF4 = [&]( ID3D12Resource *res, UINT n, UINT slot ) {
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_UNKNOWN;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.NumElements = n;
		srv.Buffer.StructureByteStride = sizeof( float ) * 4;
		g_ptDev.device->CreateShaderResourceView( res, &srv, at( slot ) );
	};
	auto makeI4 = [&]( ID3D12Resource *res, UINT n, UINT slot ) {
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_UNKNOWN;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.NumElements = n;
		srv.Buffer.StructureByteStride = sizeof( int ) * 4;
		g_ptDev.device->CreateShaderResourceView( res, &srv, at( slot ) );
	};
	makeF3( g_ptBake.albedo.Get(), g_ptBake.triCount, 3 );
	makeF4( g_ptBake.lightsA.Get(), g_ptBake.lightCount * 2, 4 );
	makeF4( g_ptBake.lightsB.Get(), g_ptBake.lightCount * 2, 5 );
	makeF4( g_ptBake.lightsC.Get(), g_ptBake.lightCount * 2, 6 );
	makeI4( g_ptBake.lightsD.Get(), g_ptBake.lightCount, 7 );
	makeF4( g_ptBake.luxelA.Get(), nLuxels * 2, 8 );
	makeF4( g_ptBake.luxelB.Get(), nLuxels * 2, 9 );
	makeI4( g_ptBake.luxelC.Get(), nLuxels, 10 );
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_UNKNOWN;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.NumElements = ( g_ptBake.localCount > 0 ) ? g_ptBake.localCount : 1u;
		srv.Buffer.StructureByteStride = sizeof( uint32 );
		g_ptDev.device->CreateShaderResourceView( g_ptBake.localIdx.Get(), &srv, at( 11 ) );
		srv.Buffer.StructureByteStride = sizeof( float );
		g_ptDev.device->CreateShaderResourceView( g_ptBake.localCdf.Get(), &srv, at( 12 ) );
	}
	{
		const uint32 nE = max( g_ptBake.emitTriCount, 1u );
		makeF4( g_ptBake.emitA.Get(), nE * 4, 13 );
		makeF4( g_ptBake.emitE.Get(), nE * 3, 14 );
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_UNKNOWN;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.NumElements = nE;
		srv.Buffer.StructureByteStride = sizeof( float );
		g_ptDev.device->CreateShaderResourceView( g_ptBake.emitCdf.Get(), &srv, at( 15 ) );
	}
	{
		const uint32 nV = max( g_ptBake.envVolCount, 1u );
		makeF4( g_ptBake.envVols.Get(), nV * 4, 16 );
	}
	{
		const uint32 nB = max( g_ptBake.bounceVolCount, 1u );
		makeF4( g_ptBake.bounceVols.Get(), nB * 3, 17 );
	}
	makeF4( g_ptBake.triFilter.Get(), g_ptBake.triCount * 5, 18 );
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Texture2DArray.MipLevels = 1;
		srv.Texture2DArray.FirstArraySlice = 0;
		srv.Texture2DArray.ArraySize = g_ptBake.filterAtlasLayers;
		g_ptDev.device->CreateShaderResourceView( g_ptBake.filterAtlas.Get(), &srv, at( 19 ) );
	}
	{
		D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
		uav.Format = DXGI_FORMAT_UNKNOWN;
		uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uav.Buffer.NumElements = g_ptBake.luxelCap;
		uav.Buffer.StructureByteStride = sizeof( float ) * 4;
		g_ptDev.device->CreateUnorderedAccessView( g_ptBake.outBuf.Get(), nullptr, &uav, at( 20 ) );
	}

	ID3D12DescriptorHeap *heaps[] = { g_ptBake.heap.Get() };
	g_ptDev.list->SetDescriptorHeaps( 1, heaps );
	g_ptDev.list->SetComputeRootSignature( g_ptBake.rootSig.Get() );
	g_ptDev.list->SetPipelineState( g_ptBake.pso.Get() );
	g_ptDev.list->SetComputeRootDescriptorTable( 0, g_ptBake.heap->GetGPUDescriptorHandleForHeapStart() );

	UINT cb[24] = {};
	cb[0] = nLuxels;
	cb[1] = g_ptBake.params.spp;
	cb[2] = g_ptBake.params.bounces;
	cb[3] = g_ptBake.lightCount;
	cb[4] = g_ptBake.params.lightSamples;
	cb[5] = g_ptBake.params.softSamplesMax;
	cb[6] = g_ptBake.params.softMode;
	cb[7] = g_ptBake.triCount;
	*(float *)&cb[8] = g_ptBake.params.occludeBias;
	*(float *)&cb[9] = g_ptBake.params.maxTraceLen;
	*(float *)&cb[10] = g_ptBake.params.fireflyCap;
	*(float *)&cb[11] = g_ptBake.params.penumbraScale;
	*(float *)&cb[12] = g_ptBake.params.skyAmb[0];
	*(float *)&cb[13] = g_ptBake.params.skyAmb[1];
	*(float *)&cb[14] = g_ptBake.params.skyAmb[2];
	cb[15] = g_ptBake.localCount;
	cb[16] = g_ptBake.sampleOffset;
	cb[17] = g_ptBake.emitTriCount;
	cb[18] = g_ptBake.params.emitSamples;
	cb[19] = g_ptBake.envVolCount;
	*(float *)&cb[20] = g_ptBake.params.defaultBounceIntensity;
	cb[21] = g_ptBake.bounceVolCount;
	*(float *)&cb[22] = g_ptBake.params.cliBounceBoost;
	*(float *)&cb[23] = g_ptBake.params.cliBounceChroma;
	g_ptDev.list->SetComputeRoot32BitConstants( 1, 24, cb, 0 );
	g_ptDev.list->Dispatch( ( nLuxels + 63 ) / 64, 1, 1 );

	D3D12_RESOURCE_BARRIER uavB = {};
	uavB.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	uavB.UAV.pResource = g_ptBake.outBuf.Get();
	g_ptDev.list->ResourceBarrier( 1, &uavB );

	D3D12_RESOURCE_BARRIER toCopy = {};
	toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	toCopy.Transition.pResource = g_ptBake.outBuf.Get();
	toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	toCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	g_ptDev.list->ResourceBarrier( 1, &toCopy );
	g_ptDev.list->CopyResource( g_ptBake.outRead.Get(), g_ptBake.outBuf.Get() );
	D3D12_RESOURCE_BARRIER back = toCopy;
	back.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
	back.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	g_ptDev.list->ResourceBarrier( 1, &back );

	// luxels back to COPY_DEST
	D3D12_RESOURCE_BARRIER toCopyD[3] = {};
	for ( int i = 0; i < 3; ++i )
	{
		toCopyD[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		toCopyD[i].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
		toCopyD[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
		toCopyD[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	}
	toCopyD[0].Transition.pResource = g_ptBake.luxelA.Get();
	toCopyD[1].Transition.pResource = g_ptBake.luxelB.Get();
	toCopyD[2].Transition.pResource = g_ptBake.luxelC.Get();
	g_ptDev.list->ResourceBarrier( 3, toCopyD );

	g_ptDev.list->Close();
	ID3D12CommandList *lists[] = { g_ptDev.list.Get() };
	g_ptDev.queue->ExecuteCommandLists( 1, lists );
	PtWaitGPU();

	{
		float *p = nullptr;
		D3D12_RANGE range = { 0, (SIZE_T)nLuxels * sizeof( float ) * 4 };
		if ( FAILED( g_ptBake.outRead->Map( 0, &range, (void **)&p ) ) )
			return false;
		float maxL = 0.0f;
		uint32_t nLit = 0;
		for ( uint32_t i = 0; i < nLuxels; ++i )
		{
			outResults[i].radiance[0] = p[i * 4 + 0];
			outResults[i].radiance[1] = p[i * 4 + 1];
			outResults[i].radiance[2] = p[i * 4 + 2];
			outResults[i].sunAmt = p[i * 4 + 3];
			const float L = outResults[i].radiance[0] + outResults[i].radiance[1] + outResults[i].radiance[2];
			if ( L > maxL ) maxL = L;
			if ( L > 1e-4f ) ++nLit;
		}
		D3D12_RANGE empty = {};
		g_ptBake.outRead->Unmap( 0, &empty );
		static bool s_logged = false;
		static bool s_hadGoodBatch = false;
		if ( !s_logged )
		{
			Msg( "[PathTrace-DXR] GPU bake batch: %u/%u luxels lit, maxRGB sum=%.4f\n", nLit, nLuxels, maxL );
			s_logged = true;
		}
		if ( nLit == 0 || maxL < 1e-6f )
		{
			HRESULT removed = S_OK;
			if ( g_ptDev.device )
				removed = g_ptDev.device->GetDeviceRemovedReason();
			if ( FAILED( removed ) )
			{
				Warning( "[PathTrace-DXR] GPU device lost during bake (0x%08X) — aborting GPU baker.\n",
						 (unsigned)removed );
				return false;
			}
			// Fully shaded / sealed faces can legitimately return all-black. Only treat
			// as a hard failure if we have never seen a lit luxel (shader/upload broken).
			if ( !s_hadGoodBatch )
			{
				Warning( "[PathTrace-DXR] GPU bake produced black results on first batch — check DXR RayQuery / buffer uploads.\n" );
				return false;
			}
			Msg( "[PathTrace-DXR] GPU bake batch all-dark (%u luxels) — keeping (likely fully shadowed faces).\n",
				 nLuxels );
		}
		else
		{
			s_hadGoodBatch = true;
		}
	}
	return true;
}
