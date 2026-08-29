// FRAGMENT SHADER for the daylight behind the exit door (see custom/ExitGlow.hpp).
//
// Where its inputs come from:
//   fragCorner   from ExitGlow.vert, -1..1 across the quad in both axes.
//   ubo (set 0)  written once per frame by main.cpp: hue, peak radiance (which
//                the door's swing drives, so a shut door emits nothing), time
//                and the falloff shape.
//
// The output is deliberately far above 1.0. The scene pass renders into an
// RGBA16F target, so nothing clips here, and BloomBright.frag thresholds at
// 1.55 -- which every pixel of this clears by a wide margin. That is the whole
// mechanism: this shader draws a bright ellipse, and the bloom chain turns it
// into a glare that bleeds out over the door frame and washes the stone around
// the opening. There is no glow billboard, no halo geometry and no second pass
// here, for the same reason Flame.hpp has none.

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
	// A shut door emits nothing, and there is no way to skip the draw call
	// (the command buffer is recorded once and replayed -- see
	// ExitGlow::populateCommandBuffer), so the "off" state has to be handled
	// here. Discarding rather than writing black also keeps the quad from
	// writing depth over the ground outside while it is invisible:
	// Starter.hpp hardcodes depthWriteEnable, so an alpha-0 fragment would
	// otherwise still occlude.
	if(ubo.intensity <= 0.001) {
		discard;
	}

	// Distance from the centre of the QUAD, not of a circle in world space:
	// main.cpp gives the quad different half-width and half-height, so this
	// is an ellipse taller than it is wide -- which is the shape of the
	// doorway it has to fill, and cheaper than correcting for the aspect and
	// then re-imposing it.
	float r = length(fragCorner);

	// A PLATEAU, not a falloff. The field is flat and fully blown out across
	// the whole body of the quad and only lets go in the last `softness` of
	// the radius. An earlier version rolled off from the centre outward, and
	// through the arch that read as a hot spot floating on a visibly darker
	// surround -- which told the player there was a surface out there, the one
	// thing this must never do. Nothing beyond the doorway is allowed to have
	// shape, so the light has none either.
	//
	// The rim fade still exists because the quad overhangs the arch on every
	// side: it is what stops an edge showing if a viewing angle ever catches
	// one, and it never appears in the opening itself.
	float field = 1.0 - smoothstep(1.0 - ubo.softness, 1.0, r);

	// Very slow breathing, from two periods that don't divide into each other
	// so the sum never visibly repeats. Small on purpose, and now smaller
	// still: at this radiance the tone map has flattened everything near 1
	// anyway, so a visible pulse here would only show up in the bloom around
	// the frame. Outdoor daylight doesn't gutter.
	float breathe = 0.97 + 0.02 * sin(ubo.time * 0.53) + 0.01 * sin(ubo.time * 0.31);

	// Warmer at the very rim, white everywhere else. Kept because the bloom
	// smears this band out over the stonework around the opening, so it is
	// what gives the glare a colour rather than making it a grey-white hole;
	// inside the arch it is out of frame.
	vec3 tint = mix(ubo.color, ubo.color * vec3(1.0, 0.88, 0.70), smoothstep(0.72, 1.0, r));

	float brightness = field * breathe;

	// Fully opaque across the entire plateau, with alpha collapsing only in
	// the outer sliver of the rim fade -- much faster than the brightness
	// does. The daylight has to HIDE what is behind it: any translucency at
	// all and the ground plane outside shows through the glare, which is
	// exactly the giveaway the plateau above is there to prevent.
	float alpha = smoothstep(0.0, 0.25, field);
	if(alpha < 0.004) {
		discard;
	}

	// Never clamped: this writes into the HDR scene target, and the bloom
	// pass downstream needs the true overbright value.
	outColor = vec4(tint * ubo.intensity * brightness, alpha);
}
