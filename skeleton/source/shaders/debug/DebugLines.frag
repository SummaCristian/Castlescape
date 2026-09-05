// FRAGMENT SHADER for the light/shadow debug line overlay. Nothing to shade:
// DebugLines.vert already resolved each vertex's color, so this just carries
// the interpolated value to the framebuffer.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(fragColor, 1.0);
}
