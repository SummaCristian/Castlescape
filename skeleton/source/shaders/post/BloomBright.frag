// FRAGMENT SHADER: first stage of the bloom chain. Reads the full-res HDR
// scene and writes a quarter-res image of only the parts bright enough to
// bloom, everything else zeroed.
//
//   uv          from Post.vert
//   post (set 0)  per frame: source texel size, threshold, knee
//   srcTex       the scene's HDR offscreen target (RGBA16F, unclamped)
//
// Threshold and downsample in one pass: the four taps below are already a box
// filter, so a dedicated downsample pass would have nothing to do.

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

// Soft-knee threshold curve (Karis, SIGGRAPH 2014). A hard cutoff makes the
// mask flicker as sparks cross the line each frame; the quadratic ramp
// through [threshold-knee, threshold+knee] fades a pixel in as it brightens.
vec3 softThreshold(vec3 c) {
	float br = luma(c);
	float soft = br - post.threshold + post.knee;
	soft = clamp(soft, 0.0, 2.0 * post.knee);
	soft = soft * soft / max(4.0 * post.knee, 1e-4);
	float contribution = max(soft, br - post.threshold);
	// Rescale the colour by the fraction of its brightness that survives,
	// rather than a flat grey, so hue and saturation stay intact.
	return c * (contribution / max(br, 1e-4));
}

void main() {
	// Four taps offset half a source texel diagonally. Bilinear filtering
	// averages each over a 2x2 block, so the four cover the full source area
	// mapping onto this quarter-res texel -- a box downsample for 4 fetches
	// instead of 16. Clamped half a texel in, like BloomBlur.frag: these
	// attachments carry a REPEAT sampler, so an off-edge tap wraps.
	vec2 texelOffset = post.texelSize;
	vec2 lo = texelOffset * 0.5;
	vec2 hi = vec2(1.0) - lo;
	vec3 tapTL = texture(srcTex, clamp(uv + vec2(-texelOffset.x, -texelOffset.y), lo, hi)).rgb;
	vec3 tapTR = texture(srcTex, clamp(uv + vec2( texelOffset.x, -texelOffset.y), lo, hi)).rgb;
	vec3 tapBL = texture(srcTex, clamp(uv + vec2(-texelOffset.x,  texelOffset.y), lo, hi)).rgb;
	vec3 tapBR = texture(srcTex, clamp(uv + vec2( texelOffset.x,  texelOffset.y), lo, hi)).rgb;

	// Karis average: weight each tap by 1/(1+luma). Otherwise a stray firefly
	// pixel (sparks are many times brighter than their neighbours) dominates
	// the mean and the downsampled quad flickers per spark.
	float weightTL = 1.0 / (1.0 + luma(tapTL));
	float weightTR = 1.0 / (1.0 + luma(tapTR));
	float weightBL = 1.0 / (1.0 + luma(tapBL));
	float weightBR = 1.0 / (1.0 + luma(tapBR));
	vec3 avg = (tapTL * weightTL + tapTR * weightTR + tapBL * weightBL + tapBR * weightBR)
	         / max(weightTL + weightTR + weightBL + weightBR, 1e-4);

	// Never clamp: HDR (RGBA16F) target, and the blur passes need the true
	// overbright values to sum correctly.
	outColor = vec4(softThreshold(avg), 1.0);
}
