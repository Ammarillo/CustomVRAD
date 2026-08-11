// CustomVRAD volumetric fog — pixel shader (ps_3_0)
// s0 = $basetexture (light-grid atlas), s1 = $texture1 (depth)
// s2 = $texture2 (baked tileable 3D noise atlas, 32^3 Z-sliced)
//
// Soft edges follow Lengyel, "Unified Distance Formulas for Halfspace Fog"
// (JGT 2007 §4): density is 0 on the volume face and rises with distance into
// the fog. Constant density at the boundary produces a hard silhouette even
// with a wide BlendDistance, because Beer-Lambert saturates through the shell
// (τ ≈ density * BlendDistance / 2 >> 1). We use a quadratic Lengyel profile
// and keep the shell optically soft for any Density.
//
// c1 = mins.xyz, maxs.z
// c2 = maxs.xy, NoiseScale (full float), CurTime()*WindSpeed (full float)
// c3 = forward * tan(horizFov/2), packFloat
// c3.w pack (integer, float-safe < 2^24):
//   bits  0..2  log2(nz)-2
//   bits  3..4  blendMode (0 Inside, 1 Outside, 2 Center)
//   bits  5..8  NoiseCoverage (0..15 → /15)
//   bits  9..11 WindYaw/45° (0..7)
//   bits 12..22 BlendDistance (0..2047 world units)
// nx/ny from TexBaseSize (c4); nz from pack; aspect from Tex1Size (c5)
// Wind is world-space: np = (p + windDir * c2.w) * c2.z
// Noise UVW = frac(np / 32) into baked FBM atlas ($texture2)
// Outer fade (Outside/Center): raymarch + light-grid UV use brush±BlendDistance
// (VRAD bakes the grid over that expanded AABB so the shell has real samples).

sampler LightGridSampler    : register( s0 );
sampler DepthTextureSampler : register( s1 );
sampler NoiseAtlasSampler   : register( s2 );

float4 c0 : register( c0 ); // xyz = EyePos, w = density
float4 c1 : register( c1 ); // xyz = mins,   w = maxs.z
float4 c2 : register( c2 ); // xy = maxs.xy, z = NoiseScale, w = CurTime()*WindSpeed
float4 c3 : register( c3 ); // xyz = forward * thx, w = pack

// screenspace_general: 1/width, 1/height of s0 / s1
float2 TexBaseSize : register( c4 );
float2 Tex1Size    : register( c5 );

static const float kNoiseRes = 32.0;

struct PS_INPUT
{
	float2 uv : TEXCOORD0;
};

float2 RayBox( float3 ro, float3 rd, float3 bmin, float3 bmax )
{
	float3 inv = 1.0 / rd;
	float3 t0 = ( bmin - ro ) * inv;
	float3 t1 = ( bmax - ro ) * inv;
	float3 tmin = min( t0, t1 );
	float3 tmax = max( t0, t1 );
	float tEnter = max( max( tmin.x, tmin.y ), tmin.z );
	float tExit  = min( min( tmax.x, tmax.y ), tmax.z );
	return float2( tEnter, tExit );
}

// Signed distance to AABB: >0 inside, <0 outside (Euclidean outside).
float SignedBlendDistance( float3 p, float3 bmin, float3 bmax )
{
	float dx = min( p.x - bmin.x, bmax.x - p.x );
	float dy = min( p.y - bmin.y, bmax.y - p.y );
	float dz = min( p.z - bmin.z, bmax.z - p.z );
	if ( dx >= 0.0 && dy >= 0.0 && dz >= 0.0 )
		return min( dx, min( dy, dz ) );

	float3 o = max( bmin - p, p - bmax );
	o = max( o, float3( 0, 0, 0 ) );
	return -length( o );
}

// Map 0..1 ramp parameter to a Lengyel-style density scale.
// u=0 at the unfogged side of the blend, u=1 at full density.
// Quadratic (u^2) → zero derivative at the face (IQ soft boundary) and
// optical depth through the shell is density*BlendDistance/3 instead of /2.
// edgeKeep caps shell optical depth so high Density cannot hard-clip the fade.
float LengyelDensityScale( float u, float density, float blendDist )
{
	u = saturate( u );
	float shape = u * u;
	float shellTau = density * max( blendDist, 1.0 ) * ( 1.0 / 3.0 );
	float edgeKeep = min( 1.0, 1.25 / max( shellTau, 1e-3 ) );
	// Rise from soft edge toward full Density; continuous at u=1.
	return shape * lerp( edgeKeep, 1.0, u );
}

// Returns local density multiplier for this sample (0 = no fog).
float FogDensityAt( float3 p, float3 bmin, float3 bmax,
					float density, float blendDist, float blendMode )
{
	float dist = SignedBlendDistance( p, bmin, bmax );

	if ( blendDist <= 1e-3 )
		return ( dist >= 0.0 ) ? 1.0 : 0.0;

	float u;
	if ( blendMode < 0.5 )
	{
		// Inside: 0 on face, 1 at BlendDistance inward (Lengyel halfspace).
		if ( dist <= 0.0 )
			return 0.0;
		u = dist / blendDist;
	}
	else if ( blendMode < 1.5 )
	{
		// Outside: full inside, 0 at BlendDistance outward.
		if ( dist >= 0.0 )
			return 1.0;
		u = 1.0 + dist / blendDist;
	}
	else
	{
		// Center: fade straddles face (BlendDistance each side).
		u = ( dist + blendDist ) / ( 2.0 * blendDist );
	}

	return LengyelDensityScale( u, density, blendDist );
}

float Lum3( float3 c )
{
	return dot( c, float3( 0.2126, 0.7152, 0.0722 ) );
}

// Wrap index into [0, n).
float WrapIndex( float i, float n )
{
	return i - n * floor( i / n );
}

// Point-sample baked noise cell (32^3 atlas, Z-sliced like the light grid).
float FetchNoiseCell( float ix, float iy, float iz )
{
	ix = WrapIndex( ix, kNoiseRes );
	iy = WrapIndex( iy, kNoiseRes );
	iz = WrapIndex( iz, kNoiseRes );
	float atlasW = kNoiseRes;
	float atlasH = kNoiseRes * kNoiseRes;
	float2 uv = float2( ( ix + 0.5 ) / atlasW,
	                    ( iy + iz * kNoiseRes + 0.5 ) / atlasH );
	return tex2Dlod( NoiseAtlasSampler, float4( uv, 0, 0 ) ).r;
}

// Trilinear sample of tileable noise. uvw in [0,1) over one 32^3 period.
float SampleNoiseAtlas( float3 uvw )
{
	uvw = frac( uvw );
	float fx = uvw.x * kNoiseRes;
	float fy = uvw.y * kNoiseRes;
	float fz = uvw.z * kNoiseRes;
	float x0 = floor( fx ); float x1 = x0 + 1.0; float tx = fx - x0;
	float y0 = floor( fy ); float y1 = y0 + 1.0; float ty = fy - y0;
	float z0 = floor( fz ); float z1 = z0 + 1.0; float tz = fz - z0;

	float c000 = FetchNoiseCell( x0, y0, z0 );
	float c100 = FetchNoiseCell( x1, y0, z0 );
	float c010 = FetchNoiseCell( x0, y1, z0 );
	float c110 = FetchNoiseCell( x1, y1, z0 );
	float c001 = FetchNoiseCell( x0, y0, z1 );
	float c101 = FetchNoiseCell( x1, y0, z1 );
	float c011 = FetchNoiseCell( x0, y1, z1 );
	float c111 = FetchNoiseCell( x1, y1, z1 );

	float nx00 = lerp( c000, c100, tx );
	float nx10 = lerp( c010, c110, tx );
	float nx01 = lerp( c001, c101, tx );
	float nx11 = lerp( c011, c111, tx );
	float nxy0 = lerp( nx00, nx10, ty );
	float nxy1 = lerp( nx01, nx11, ty );
	return lerp( nxy0, nxy1, tz );
}

// Point-sample one grid cell (integer indices). rgb = lighting, a = fog allow.
float4 FetchLightCell( float ix, float iy, float iz, float nx, float ny, float nz )
{
	ix = clamp( ix, 0.0, max( nx - 1.0, 0.0 ) );
	iy = clamp( iy, 0.0, max( ny - 1.0, 0.0 ) );
	iz = clamp( iz, 0.0, max( nz - 1.0, 0.0 ) );
	float atlasW = nx;
	float atlasH = ny * nz;
	float2 uv = float2( ( ix + 0.5 ) / atlasW,
	                    ( iy + iz * ny + 0.5 ) / atlasH );
	return tex2Dlod( LightGridSampler, float4( uv, 0, 0 ) );
}

// Trilinear light-grid sample that down-weights near-black cells (wall /
// solid holes) so fog does not go black when interpolating near geometry.
// .a = fog allow from fog_volume_blocker (1 = fog, 0 = blocked).
float4 SampleLightGrid( float3 p, float3 bmin, float3 bmax, float nx, float ny, float nz )
{
	float3 ext = max( bmax - bmin, float3( 1e-3, 1e-3, 1e-3 ) );
	float3 uvw = saturate( ( p - bmin ) / ext );

	float fx = uvw.x * max( nx - 1.0, 0.0 );
	float fy = uvw.y * max( ny - 1.0, 0.0 );
	float fz = uvw.z * max( nz - 1.0, 0.0 );
	float x0 = floor( fx ); float x1 = min( x0 + 1.0, max( nx - 1.0, 0.0 ) ); float tx = fx - x0;
	float y0 = floor( fy ); float y1 = min( y0 + 1.0, max( ny - 1.0, 0.0 ) ); float ty = fy - y0;
	float z0 = floor( fz ); float z1 = min( z0 + 1.0, max( nz - 1.0, 0.0 ) ); float tz = fz - z0;

	float3 acc = float3( 0, 0, 0 );
	float wsum = 0.0;
	float3 bright = float3( 0, 0, 0 );
	float brightLum = 0.0;

	float4 c000 = FetchLightCell( x0, y0, z0, nx, ny, nz );
	float4 c100 = FetchLightCell( x1, y0, z0, nx, ny, nz );
	float4 c010 = FetchLightCell( x0, y1, z0, nx, ny, nz );
	float4 c110 = FetchLightCell( x1, y1, z0, nx, ny, nz );
	float4 c001 = FetchLightCell( x0, y0, z1, nx, ny, nz );
	float4 c101 = FetchLightCell( x1, y0, z1, nx, ny, nz );
	float4 c011 = FetchLightCell( x0, y1, z1, nx, ny, nz );
	float4 c111 = FetchLightCell( x1, y1, z1, nx, ny, nz );

	float w000 = ( 1.0 - tx ) * ( 1.0 - ty ) * ( 1.0 - tz );
	float w100 = tx * ( 1.0 - ty ) * ( 1.0 - tz );
	float w010 = ( 1.0 - tx ) * ty * ( 1.0 - tz );
	float w110 = tx * ty * ( 1.0 - tz );
	float w001 = ( 1.0 - tx ) * ( 1.0 - ty ) * tz;
	float w101 = tx * ( 1.0 - ty ) * tz;
	float w011 = ( 1.0 - tx ) * ty * tz;
	float w111 = tx * ty * tz;

	// Fog allow is always trilinear (blockers carve density, not only lighting).
	float allow = c000.a * w000 + c100.a * w100 + c010.a * w010 + c110.a * w110
				+ c001.a * w001 + c101.a * w101 + c011.a * w011 + c111.a * w111;

	float l000 = Lum3( c000.rgb ); float l100 = Lum3( c100.rgb );
	float l010 = Lum3( c010.rgb ); float l110 = Lum3( c110.rgb );
	float l001 = Lum3( c001.rgb ); float l101 = Lum3( c101.rgb );
	float l011 = Lum3( c011.rgb ); float l111 = Lum3( c111.rgb );

	float s000 = saturate( l000 * 32.0 ); float s100 = saturate( l100 * 32.0 );
	float s010 = saturate( l010 * 32.0 ); float s110 = saturate( l110 * 32.0 );
	float s001 = saturate( l001 * 32.0 ); float s101 = saturate( l101 * 32.0 );
	float s011 = saturate( l011 * 32.0 ); float s111 = saturate( l111 * 32.0 );

	acc += c000.rgb * ( w000 * s000 ); wsum += w000 * s000;
	acc += c100.rgb * ( w100 * s100 ); wsum += w100 * s100;
	acc += c010.rgb * ( w010 * s010 ); wsum += w010 * s010;
	acc += c110.rgb * ( w110 * s110 ); wsum += w110 * s110;
	acc += c001.rgb * ( w001 * s001 ); wsum += w001 * s001;
	acc += c101.rgb * ( w101 * s101 ); wsum += w101 * s101;
	acc += c011.rgb * ( w011 * s011 ); wsum += w011 * s011;
	acc += c111.rgb * ( w111 * s111 ); wsum += w111 * s111;

	bright = c000.rgb; brightLum = l000;
	if ( l100 > brightLum ) { bright = c100.rgb; brightLum = l100; }
	if ( l010 > brightLum ) { bright = c010.rgb; brightLum = l010; }
	if ( l110 > brightLum ) { bright = c110.rgb; brightLum = l110; }
	if ( l001 > brightLum ) { bright = c001.rgb; brightLum = l001; }
	if ( l101 > brightLum ) { bright = c101.rgb; brightLum = l101; }
	if ( l011 > brightLum ) { bright = c011.rgb; brightLum = l011; }
	if ( l111 > brightLum ) { bright = c111.rgb; brightLum = l111; }

	float3 lighting = ( wsum > 1e-5 ) ? ( acc / wsum ) : bright;
	return float4( lighting, allow );
}

float4 main( PS_INPUT i ) : COLOR
{
	float density = max( c0.w, 0.0 );
	if ( density <= 1e-6 )
		return float4( 0, 0, 0, 0 );

	float3 eye = c0.xyz;
	float3 bmin = c1.xyz;
	float3 bmax = float3( c2.xy, c1.w );
	float noiseScale = max( c2.z, 0.0 );
	float windPhase = c2.w; // CurTime() * WindSpeed

	float thx = max( length( c3.xyz ), 1e-4 );
	float3 forward = c3.xyz / thx;

	float aspect = 1.777;
	if ( Tex1Size.x > 1e-8 && Tex1Size.y > 1e-8 )
		aspect = Tex1Size.y / Tex1Size.x;
	float thy = thx / max( aspect, 1e-4 );

	float pack = floor( c3.w + 0.5 );
	float lognz = fmod( pack, 8.0 );
	float blendMode = fmod( floor( pack / 8.0 ), 4.0 );
	float coverage = fmod( floor( pack / 32.0 ), 16.0 ) / 15.0;
	float windYawQ = fmod( floor( pack / 512.0 ), 8.0 );
	float blendDist = fmod( floor( pack / 4096.0 ), 2048.0 );
	float windYaw = windYawQ * 0.78539816; // 45 deg

	float nz = exp2( lognz + 2.0 );
	float atlasW = ( TexBaseSize.x > 1e-8 ) ? ( 1.0 / TexBaseSize.x ) : 4.0;
	float atlasH = ( TexBaseSize.y > 1e-8 ) ? ( 1.0 / TexBaseSize.y ) : nz * 4.0;
	float nx = max( floor( atlasW + 0.5 ), 1.0 );
	float ny = max( floor( atlasH / max( nz, 1.0 ) + 0.5 ), 1.0 );

	float3 right = cross( forward, float3( 0, 0, 1 ) );
	if ( dot( right, right ) < 1e-6 )
		right = float3( 0, 1, 0 );
	right = normalize( right );
	float3 up = normalize( cross( right, forward ) );

	float2 ndc = float2( i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0 );
	float3 rd = normalize( forward + right * ( ndc.x * thx ) + up * ( ndc.y * thy ) );

	float depthSample = tex2D( DepthTextureSampler, i.uv ).r;
	float viewZ = depthSample * 4000.0;
	float cosF = max( dot( rd, forward ), 0.05 );
	float sceneDist = min( viewZ / cosF, 8192.0 );
	if ( depthSample <= 1e-6 )
		sceneDist = 8192.0;

	float expand = 0.0;
	if ( blendDist > 1e-3 && blendMode > 0.5 )
		expand = blendDist;
	float3 bminR = bmin - expand;
	float3 bmaxR = bmax + expand;

	float2 hit = RayBox( eye, rd, bminR, bmaxR );
	float t0 = max( hit.x, 0.0 );
	float t1 = min( hit.y, sceneDist );
	if ( t1 <= t0 )
		return float4( 0, 0, 0, 0 );

	const int STEPS = 48;
	float dt = ( t1 - t0 ) / float( STEPS );

	float3 scatter = float3( 0, 0, 0 );
	float transmittance = 1.0;
	float3 windDir = float3( cos( windYaw ), sin( windYaw ), 0.15 );

	[loop]
	for ( int s = 0; s < STEPS; ++s )
	{
		float t = t0 + ( float( s ) + 0.5 ) * dt;
		float3 p = eye + rd * t;

		float densScale = FogDensityAt( p, bmin, bmax, density, blendDist, blendMode );
		float contact = saturate( ( sceneDist - t ) / 24.0 );
		densScale *= contact;

		if ( noiseScale > 1e-8 && densScale > 1e-5 )
		{
			// World-space wind; baked 32^3 FBM tiles via frac(np/32).
			float3 np = ( p + windDir * windPhase ) * noiseScale;
			float n = SampleNoiseAtlas( np / kNoiseRes );
			float cover = saturate( ( n - ( 1.0 - coverage ) ) / max( coverage, 1e-3 ) );
			densScale *= cover * lerp( 0.35, 1.0, n );
		}

		if ( densScale <= 1e-5 )
			continue;

		// Light grid covers brush±expand (baked); sample at the real march point
		// so the outer fade has its own lighting instead of stretched edge texels.
		float4 grid = SampleLightGrid( p, bminR, bmaxR, nx, ny, nz );
		densScale *= saturate( grid.a ); // fog_volume_blocker allow mask
		if ( densScale <= 1e-5 )
			continue;

		float3 lighting = grid.rgb * 1.5;

		// Beer-Lambert with local Lengyel density.
		float optical = density * densScale * dt;
		float absorb = 1.0 - exp( -optical );
		scatter += transmittance * absorb * lighting;
		transmittance *= ( 1.0 - absorb );
		if ( transmittance < 0.01 )
			break;
	}

	float alpha = saturate( 1.0 - transmittance );
	return float4( scatter, alpha );
}
