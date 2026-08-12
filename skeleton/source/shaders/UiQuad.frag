#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConsts {
	vec4 color;
} pushConsts;

void main() {
	outColor = pushConsts.color;
}
