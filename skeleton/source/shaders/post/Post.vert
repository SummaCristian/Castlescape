// VERTEX SHADER shared by every pass of the bloom chain (BloomBright,
// BloomBlur x2, Composite). All four draw the same full-screen quad; only the
// fragment shader changes. No uniform block: a post-process quad's vertices
// never move, so every pass reads its inputs in the fragment stage.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPos;

layout(location = 0) out vec2 uv;

void main() {
	// inPos is already clip-space NDC ((-1,-1)..(1,1)), so it goes straight to
	// gl_Position -- no camera or model in a full-screen pass.
	gl_Position = vec4(inPos, 0.0, 1.0);

	// Remap to 0..1 for texture sampling. Vulkan's texture origin is top-left,
	// matching NDC's +Y-down, so no flip.
	uv = inPos * 0.5 + 0.5;
}
