#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(binding = 1, set = 1) uniform sampler2D albedoMap;

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 lightDir;
    vec4 lightColor;
    vec3 eyePos;
} gubo;

void main() {
    // Interpolated per-vertex normal. Renormalized because linear interpolation
    // across a triangle shortens the vector wherever the three corner normals
    // diverge, which is exactly where smooth shading matters most.
    vec3 N = normalize(fragNorm);

    vec3 albedo = pow(texture(albedoMap, fragUV).rgb, vec3(2.2));

    vec3 V = normalize(gubo.eyePos - fragPos);
    vec3 L = normalize(-gubo.lightDir);
    vec3 H = normalize(V + L);
    vec3 radiance = gubo.lightColor.rgb;

    float NdotL = max(dot(N, L), 0.0);
    float HdotN = max(dot(H, N), 0.0);
    vec3 Lo = (albedo * NdotL + vec3(pow(HdotN, 150.0))) * radiance;

    vec3 ambient = 0.015 * albedo;
    vec3 color = ambient + Lo;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}
