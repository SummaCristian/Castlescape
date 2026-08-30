#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConsts {
	vec4 color;
	// Optional circular clip, used by the minimap. xy = centre in pixels,
	// z = radius in pixels, w = feather width in pixels. Disabled when z <= 0,
	// which is how every other UiQuad user (the cheat HUD, the crosshair)
	// leaves it.
	vec4 circle;
} pushConsts;

void main() {
	vec4 c = pushConsts.color;

	if (pushConsts.circle.z > 0.0) {
		float d = distance(gl_FragCoord.xy, pushConsts.circle.xy);
		float edge0 = pushConsts.circle.z - max(pushConsts.circle.w, 0.001);
		float a = 1.0 - smoothstep(edge0, pushConsts.circle.z, d);
		if (a <= 0.0) {
			discard;
		}
		c.a *= a;
	}

	outColor = c;
}
