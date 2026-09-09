// Shared by every bloom chain pass (BloomBright, BloomBlur x2, Composite):
// same full-screen quad, only the fragment shader differs. No uniform block.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPos;

layout(location = 0) out vec2 uv;

void main() {
	// inPos already NDC, straight to gl_Position (no camera/model).
	gl_Position = vec4(inPos, 0.0, 1.0);

	// Remap to 0..1; Vulkan texture origin matches NDC +Y-down, no flip needed.
	uv = inPos * 0.5 + 0.5;
}
