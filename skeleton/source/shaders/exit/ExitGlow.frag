// FRAGMENT SHADER for the daylight behind the exit door (custom/ExitGlow.hpp).
//
//   fragCorner   from ExitGlow.vert, -1..1 across the quad
//   ubo (set 0)  per frame: hue, peak radiance (driven by the door's swing),
//                time, falloff shape
//
// The output is far above 1.0 on purpose: the RGBA16F target doesn't clip, and
// BloomBright.frag thresholds at 1.55, so the bloom chain turns this bright
// rectangle into a glare that washes the stone around the opening. No glow
// billboard or halo geometry, same as Flame.hpp.

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
	// A shut door emits nothing and the draw call can't be skipped, so handle
	// "off" here. Discard, not write black: depthWriteEnable is hardcoded on,
	// so an alpha-0 fragment would still occlude the ground outside.
	if(ubo.intensity <= 0.001) {
		discard;
	}

	// RECTANGULAR distance, not radial: the larger axis distance, so the level
	// sets are the quad's outline, not an inscribed ellipse. The worst viewing
	// angle through the arch projects the opening onto a CORNER of the quad --
	// the farthest point under length() -- so a radial fade reached it while
	// the middle went spare. Per-axis puts the whole rectangle to work.
	float axisDist = max(abs(fragCorner.x), abs(fragCorner.y));

	// A PLATEAU, not a falloff: flat and blown out across the whole quad,
	// letting go only in the last `softness`. Rolling off from the centre read
	// as a hot spot on a darker surround -- which tells the player there's a
	// surface out there, the one thing this must never do. The border fade
	// stays because the quads overhang the opening on every side; it never
	// appears in the doorway itself.
	float field = 1.0 - smoothstep(1.0 - ubo.softness, 1.0, axisDist);

	// Very slow breathing from two non-dividing periods, so the sum never
	// visibly repeats. Small: outdoor daylight doesn't gutter, and at this
	// radiance a pulse would only show in the bloom around the frame.
	float breathe = 0.97 + 0.02 * sin(ubo.time * 0.53) + 0.01 * sin(ubo.time * 0.31);

	// Warmer at the very edge, white elsewhere. The bloom smears this band
	// over the stonework, giving the glare a colour instead of a grey-white
	// hole; inside the arch it's out of frame.
	vec3 tint = mix(ubo.color, ubo.color * vec3(1.0, 0.88, 0.70), smoothstep(0.72, 1.0, axisDist));

	float brightness = field * breathe;

	// Opaque across the plateau, alpha collapsing only in the outer sliver of
	// the fade, faster than brightness does. The daylight must HIDE what's
	// behind it: any translucency and the ground plane shows through the glare.
	float alpha = smoothstep(0.0, 0.25, field);
	if(alpha < 0.004) {
		discard;
	}

	// Never clamped: HDR scene target, the bloom pass needs the true value.
	outColor = vec4(tint * ubo.intensity * brightness, alpha);
}
