// FRAGMENT SHADER: one axis of a separable Gaussian blur, run twice per frame
// (H then V) on the quarter-res bright-pass output. A 2D Gaussian separates
// into two 1D passes (2*N taps instead of N*N); post.blurDir picks the axis.
//
//   uv          from Post.vert
//   post (set 0)  per draw: source texel size, blurDir (1,0) H or (0,1) V
//   srcTex       previous stage: BloomBright.frag for H, this shader's H result for V

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform PostUniformBufferObject {
	vec2  texelSize;
	vec2  blurDir;
	float threshold;
	float knee;
	float bloomIntensity;
	float exposure;
	int   debugFlags;
	float time;
	float escapeFlash;
} post;

layout(binding = 1, set = 0) uniform sampler2D srcTex;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

void main() {
	// 9 plain fetches, not the 5-fetch bilinear-tap trick: that needs each
	// pair's weight and sample point solved by hand, and getting it wrong
	// silently biases the blur. The kernel is small and runs at quarter res,
	// so the extra four fetches are cheap.
	//
	// Weights: a radius-4 discrete Gaussian, symmetric, normalized to 1.0 so
	// the blur only spreads brightness, never changes the total.
	float weights[9] = float[](
		0.016216, 0.054054, 0.1216216, 0.1945946, 0.2270270,
		0.1945946, 0.1216216, 0.054054, 0.016216
	);

	// Clamp range: half a texel in from each border. Needed by hand because
	// Starter.hpp builds framebuffer-attachment samplers with REPEAT, so an
	// off-edge tap wraps and an overbright pixel at the top of the frame
	// prints a bright band along the bottom. This is CLAMP_TO_EDGE done in
	// the shader, so the fix stays inside the bloom chain rather than
	// changing a sampler default every model texture relies on for tiling.
	vec2 lo = post.texelSize * 0.5;
	vec2 hi = vec2(1.0) - lo;

	vec3 sum = vec3(0.0);
	for(int i = 0; i < 9; i++) {
		float offset = float(i - 4);	// -4..4, centred on this fragment
		vec2 sampleUV = clamp(uv + post.texelSize * post.blurDir * offset, lo, hi);
		sum += texture(srcTex, sampleUV).rgb * weights[i];
	}

	// Unclamped: still an HDR intermediate, not a displayable colour yet.
	outColor = vec4(sum, 1.0);
}
