// FRAGMENT SHADER for the 2D shadow pass. Writes nothing: RPShadow has no
// color attachment, only depth, written automatically from gl_Position.z.
// It still has to exist -- Pipeline::init always links a fragment module.

#version 450
#extension GL_ARB_separate_shader_objects : enable

void main() {
}
