// Daylight behind the exit door (custom/ExitGlow.hpp).
//   fragCorner  from ExitGlow.vert, -1..1 across the quad
//   ubo         hue, peak radiance (driven by door swing), time, falloff shape
// Output can exceed 1.0 on purpose: RGBA16F target doesn't clip, and
// BloomBright.frag thresholds at 1.55, turning this into glare on the stone.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform ExitGlowUniformBufferObject {
	mat4  mvpMat;
	vec3  color;
	float intensity;
	float time;
	float softness;
} ubo;

layout(location = 0) in vec2 fragCorner;

layout(location = 0) out vec4 outColor;

void main() {
	// Shut door: draw call can't be skipped, so discard here (not black --
	// depthWriteEnable is on, alpha-0 would still occlude).
	if(ubo.intensity <= 0.001) {
		discard;
	}

	// Rectangular (per-axis) distance, not radial: worst viewing angle
	// projects the opening onto a quad corner, which length() would reach
	// but radial falloff wouldn't fully use.
	float axisDist = max(abs(fragCorner.x), abs(fragCorner.y));

	// Plateau, not falloff: flat/blown out, only fading in the last `softness`.
	// A centre rolloff would read as a hot spot implying a surface -- avoid that.
	float field = 1.0 - smoothstep(1.0 - ubo.softness, 1.0, axisDist);

	// Slow breathing, two non-dividing periods so it never visibly repeats.
	float breathe = 0.97 + 0.02 * sin(ubo.time * 0.53) + 0.01 * sin(ubo.time * 0.31);

	// Warmer at the edge (bloom smears it onto stonework), white elsewhere.
	vec3 tint = mix(ubo.color, ubo.color * vec3(1.0, 0.88, 0.70), smoothstep(0.72, 1.0, axisDist));

	float brightness = field * breathe;

	// Alpha collapses faster than brightness in the outer fade sliver, staying
	// opaque across the plateau so the ground behind never shows through.
	float alpha = smoothstep(0.0, 0.25, field);
	if(alpha < 0.004) {
		discard;
	}

	// Unclamped: HDR target, bloom needs the true value.
	outColor = vec4(tint * ubo.intensity * brightness, alpha);
}
