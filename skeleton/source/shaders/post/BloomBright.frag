// FRAGMENT SHADER: first stage of the bloom chain. Reads full-res HDR scene,
// writes quarter-res image of only bright-enough parts, rest zeroed.
// Threshold and downsample combined: the four taps below are already a box filter.

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

// Rec. 709 luma, used only as a scalar brightness.
float luma(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Soft-knee threshold curve (Karis, SIGGRAPH 2014). Avoids the flicker a hard
// cutoff causes as sparks cross the line each frame.
vec3 softThreshold(vec3 c) {
	float br = luma(c);
	float soft = br - post.threshold + post.knee;
	soft = clamp(soft, 0.0, 2.0 * post.knee);
	soft = soft * soft / max(4.0 * post.knee, 1e-4);
	float contribution = max(soft, br - post.threshold);
	// Rescale by surviving brightness fraction, not flat grey, to keep hue/sat.
	return c * (contribution / max(br, 1e-4)); // Keep color
}

void main() {
	// Four taps offset half a texel diagonally; bilinear averages each over a
	// 2x2 block, giving a box downsample for 4 fetches instead of 16.
	// Clamped half a texel in (REPEAT samplers, see BloomBlur.frag).

	// 2x2
	vec2 texelOffset = post.texelSize;
	vec2 lo = texelOffset * 0.5;
	vec2 hi = vec2(1.0) - lo;
	// 4 Tap sample
	vec3 tapTL = texture(srcTex, clamp(uv + vec2(-texelOffset.x, -texelOffset.y), lo, hi)).rgb;
	vec3 tapTR = texture(srcTex, clamp(uv + vec2( texelOffset.x, -texelOffset.y), lo, hi)).rgb;
	vec3 tapBL = texture(srcTex, clamp(uv + vec2(-texelOffset.x,  texelOffset.y), lo, hi)).rgb;
	vec3 tapBR = texture(srcTex, clamp(uv + vec2( texelOffset.x,  texelOffset.y), lo, hi)).rgb;

	// Karis average: weight each tap by 1/(1+luma), else a bright firefly
	// pixel dominates the mean and the downsampled quad flickers per spark.
	float weightTL = 1.0 / (1.0 + luma(tapTL));
	float weightTR = 1.0 / (1.0 + luma(tapTR));
	float weightBL = 1.0 / (1.0 + luma(tapBL));
	float weightBR = 1.0 / (1.0 + luma(tapBR));
	vec3 avg = (tapTL * weightTL + tapTR * weightTR + tapBL * weightBL + tapBR * weightBR)
	         / max(weightTL + weightTR + weightBL + weightBR, 1e-4);

	// Never clamp: HDR target, blur passes need true overbright values.
	outColor = vec4(softThreshold(avg), 1.0);
}
