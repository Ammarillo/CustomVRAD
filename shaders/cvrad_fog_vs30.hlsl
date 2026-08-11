// cvrad_fog_vs30.hlsl - fullscreen for screenspace_general (SM3.0)
struct VS_INPUT {
	float3 pos : POSITION;
	float2 uv : TEXCOORD0;
	float4 color : COLOR0;
};
struct VS_OUTPUT {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
	float4 color : COLOR0;
};
VS_OUTPUT main(const VS_INPUT v) {
	VS_OUTPUT o;
	// Prefer UV for clip pos - DrawScreenQuad POSITION can be unreliable in 2D hooks
	o.pos = float4(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0, 0.0, 1.0);
	o.uv = v.uv;
	o.color = v.color;
	return o;
}