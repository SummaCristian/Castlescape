// VERTEX SHADER shared by every pass of the bloom post-processing chain
// (BloomBright.frag, BloomBlur.frag x2, Composite.frag). All four passes draw
// the exact same thing -- a single triangle-strip quad that covers the whole
// screen -- so one vertex shader does for all of them; only the fragment
// shader changes between passes.
//
// There is deliberately no uniform block here: every pass reads its own
// per-fragment inputs (source texel size, blur direction, exposure, ...) in
// the fragment stage instead, since none of that affects where a vertex ends
// up. A post-process quad's vertices never move.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPos;

layout(location = 0) out vec2 uv;

void main() {
	// inPos is already clip-space NDC (the corners are (-1,-1)..(1,1)), so it
	// goes straight to gl_Position with no matrix -- there is no camera or
	// model to transform by in a full-screen pass.
	gl_Position = vec4(inPos, 0.0, 1.0);

	// uv remaps that same NDC square to 0..1 so the fragment shader can sample
	// textures directly with it. Vulkan's texture origin is the top-left,
	// matching NDC's own +Y-down convention, so no flip is needed here.
	uv = inPos * 0.5 + 0.5;
}
