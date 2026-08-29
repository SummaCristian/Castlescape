// FRAGMENT SHADER: one axis of a separable Gaussian blur, run twice per
// frame (horizontal then vertical) on the quarter-res bright-pass output.
// A true 2D Gaussian blur is separable into two 1D passes with identical
// per-pixel cost to a single N-tap 2D kernel of the same radius but far
// fewer taps overall (2*N vs N*N) -- this shader is that 1D pass, and which
// axis it runs along for a given draw is entirely decided by post.blurDir.
//
// Where its inputs come from:
//   uv              from Post.vert, the shared full-screen-quad vertex shader.
//   post (set 0)    written per draw by main.cpp: the source texture's texel
//                   size and blurDir, (1,0) for the horizontal pass and (0,1)
//                   for the vertical one.
//   srcTex          the previous stage's output: BloomBright.frag's result
//                   for the horizontal pass, this shader's own horizontal
//                   result for the vertical pass.

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
	// Straightforward 9 independent fetches, not the 5-fetch bilinear-tap
	// trick (pairing weights and sampling between texel centres to get two
	// taps for the price of one). That trick needs each pair's two weights
	// to be combined and its sample point solved for by hand per kernel, and
	// getting that arithmetic wrong silently biases the blur rather than
	// erroring out. This bloom's kernel is small and runs at quarter
	// resolution, so the extra four fetches are cheap enough that the risk
	// isn't worth taking.
	//
	// Weights are a standard radius-4 discrete Gaussian, symmetric around
	// the centre tap and normalized to sum to 1.0 so the blur cannot change
	// the image's total brightness, only spread it out.
	float weights[9] = float[](
		0.016216, 0.054054, 0.1216216, 0.1945946, 0.2270270,
		0.1945946, 0.1216216, 0.054054, 0.016216
	);

	vec3 sum = vec3(0.0);
	for(int i = 0; i < 9; i++) {
		float offset = float(i - 4);	// -4..4, centred on this fragment
		vec2 sampleUV = uv + post.texelSize * post.blurDir * offset;
		sum += texture(srcTex, sampleUV).rgb * weights[i];
	}

	// Unclamped: still an HDR intermediate, read by another blur pass or by
	// Composite.frag, not a final displayable colour yet.
	outColor = vec4(sum, 1.0);
}
