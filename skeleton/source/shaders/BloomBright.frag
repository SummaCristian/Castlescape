// FRAGMENT SHADER: first stage of the bloom chain. Reads the full-res HDR
// scene target and writes a quarter-res image containing only the parts
// bright enough to bloom, everything else zeroed.
//
// Where its inputs come from:
//   uv              from Post.vert, the shared full-screen-quad vertex shader.
//   post (set 0)    written once per frame by main.cpp: the source texture's
//                   texel size and the threshold/knee for this pass.
//   srcTex          the scene's HDR offscreen target (RGBA16F, unclamped).
//
// Doing the threshold and the downsample in the same pass, rather than one
// full-res threshold pass followed by a separate downsample, halves the
// texture traffic for free -- the four taps below are already a box filter,
// so there is nothing left for a dedicated downsample pass to do.

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

// Rec. 709 luma. The bloom pass only ever needs this as a scalar brightness
// to threshold and weight by, never as a colour, so no separate "luminance
// vs luma" distinction matters here.
float luma(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Soft-knee threshold curve (Karis, SIGGRAPH 2014 "Next Generation Post
// Processing in Call of Duty: Advanced Warfare"). A hard cutoff at threshold
// makes the bloom mask flicker pixel-to-pixel as sparks cross the line each
// frame; the quadratic ramp through [threshold-knee, threshold+knee] instead
// fades a pixel in smoothly as it brightens.
vec3 softThreshold(vec3 c) {
	float br = luma(c);
	float soft = br - post.threshold + post.knee;
	soft = clamp(soft, 0.0, 2.0 * post.knee);
	soft = soft * soft / max(4.0 * post.knee, 1e-4);
	float contribution = max(soft, br - post.threshold);
	// Rescale the original colour by the fraction of its brightness that
	// survives thresholding, rather than returning a flat "contribution"
	// grey -- this keeps hue and saturation intact on the parts that bloom.
	return c * (contribution / max(br, 1e-4));
}

void main() {
	// Four taps offset by half a source texel in each diagonal direction.
	// Bilinear filtering then averages each tap over a 2x2 block of source
	// texels, so these four samples between them cover the full source area
	// that maps onto this (quarter-res) output texel -- a cheap box downsample
	// for the price of 4 fetches instead of the naive 16.
	vec2 o = post.texelSize;
	vec3 c0 = texture(srcTex, uv + vec2(-o.x, -o.y)).rgb;
	vec3 c1 = texture(srcTex, uv + vec2( o.x, -o.y)).rgb;
	vec3 c2 = texture(srcTex, uv + vec2(-o.x,  o.y)).rgb;
	vec3 c3 = texture(srcTex, uv + vec2( o.x,  o.y)).rgb;

	// Karis average: weight each tap by 1/(1+luma) instead of averaging them
	// evenly. A single stray firefly pixel (a spark can be many times
	// brighter than its neighbours) would otherwise dominate the mean and
	// make the downsampled quad flicker in step with individual sparks
	// rather than the flame's overall brightness.
	float w0 = 1.0 / (1.0 + luma(c0));
	float w1 = 1.0 / (1.0 + luma(c1));
	float w2 = 1.0 / (1.0 + luma(c2));
	float w3 = 1.0 / (1.0 + luma(c3));
	vec3 avg = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) / max(w0 + w1 + w2 + w3, 1e-4);

	// Never clamp: this writes into an HDR (RGBA16F) target, and the blur
	// passes after this one need the true overbright values to still sum
	// correctly.
	outColor = vec4(softThreshold(avg), 1.0);
}
