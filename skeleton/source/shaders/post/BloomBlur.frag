// FRAGMENT SHADER: one axis of a separable Gaussian blur, run twice per frame
// (H then V) on the quarter-res bright-pass output. post.blurDir picks the axis.

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
	// 9 plain fetches (not the 5-tap bilinear trick, to avoid hand-solving
	// weights). Radius-4 discrete Gaussian, normalized to 1.0.
	float weights[9] = float[](
		0.016216, 0.054054, 0.1216216, 0.1945946, 0.2270270,
		0.1945946, 0.1216216, 0.054054, 0.016216
	);

	// Clamp half a texel in: attachments use REPEAT samplers (Starter.hpp),
	// so an off-edge tap would wrap and print a bright band on the opposite edge.
	vec2 lo = post.texelSize * 0.5;
	vec2 hi = vec2(1.0) - lo;

	vec3 sum = vec3(0.0);
	for(int i = 0; i < 9; i++) {
		float offset = float(i - 4);	// -4..4, centred on this fragment
		vec2 sampleUV = clamp(uv + post.texelSize * post.blurDir * offset, lo, hi);
		sum += texture(srcTex, sampleUV).rgb * weights[i];
	}

	// Unclamped HDR intermediate.
	outColor = vec4(sum, 1.0);
}
