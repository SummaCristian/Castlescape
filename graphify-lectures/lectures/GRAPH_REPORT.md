# Graph Report - C:\Users\Simo\Documenti_locale\Simo_Poli_passato\5_Anno\2 Semestre\Computer Graphics\_text  (2026-08-02)

## Corpus Check
- 29 files · ~75,907 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 618 nodes · 1165 edges · 36 communities (34 shown, 2 thin omitted)
- Extraction: 86% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 157 edges (avg confidence: 0.86)
- Token cost: 786,433 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_3D Transforms and Projections|3D Transforms and Projections]]
- [[_COMMUNITY_Rendering Equation and BRDF Theory|Rendering Equation and BRDF Theory]]
- [[_COMMUNITY_Project Topics and Requirements|Project Topics and Requirements]]
- [[_COMMUNITY_Meshes and Depth Testing|Meshes and Depth Testing]]
- [[_COMMUNITY_Mesh Construction and Clipping|Mesh Construction and Clipping]]
- [[_COMMUNITY_Course Intro and Raster Basics|Course Intro and Raster Basics]]
- [[_COMMUNITY_Pipelines and Shading Stages|Pipelines and Shading Stages]]
- [[_COMMUNITY_Textures, Samplers and Descriptors|Textures, Samplers and Descriptors]]
- [[_COMMUNITY_Exam Rules and Grading|Exam Rules and Grading]]
- [[_COMMUNITY_Keyframe Animation and Interpolation|Keyframe Animation and Interpolation]]
- [[_COMMUNITY_UV Mapping and Texture Filtering|UV Mapping and Texture Filtering]]
- [[_COMMUNITY_Render Passes and Attachments|Render Passes and Attachments]]
- [[_COMMUNITY_Cook-Torrance and Microfacet BRDFs|Cook-Torrance and Microfacet BRDFs]]
- [[_COMMUNITY_Graphics Pipeline Fixed State|Graphics Pipeline Fixed State]]
- [[_COMMUNITY_Vertex Formats and Model Loading|Vertex Formats and Model Loading]]
- [[_COMMUNITY_Uniform Buffers and Descriptor Sets|Uniform Buffers and Descriptor Sets]]
- [[_COMMUNITY_Projection Matrices in GLM|Projection Matrices in GLM]]
- [[_COMMUNITY_GLM and GLSL Types|GLM and GLSL Types]]
- [[_COMMUNITY_Camera Navigation Models|Camera Navigation Models]]
- [[_COMMUNITY_View and World Matrices in GLM|View and World Matrices in GLM]]
- [[_COMMUNITY_Normal Maps and PBR Materials|Normal Maps and PBR Materials]]
- [[_COMMUNITY_Command Buffer Binding and Instancing|Command Buffer Binding and Instancing]]
- [[_COMMUNITY_Input Handling and Six-Axis Controls|Input Handling and Six-Axis Controls]]
- [[_COMMUNITY_Oren-Nayar and Toon Shading|Oren-Nayar and Toon Shading]]
- [[_COMMUNITY_Ambient and Image Based Lighting|Ambient and Image Based Lighting]]
- [[_COMMUNITY_Axonometric Projections and Rotation|Axonometric Projections and Rotation]]
- [[_COMMUNITY_Shader Interpolation Qualifiers|Shader Interpolation Qualifiers]]
- [[_COMMUNITY_Quaternion Rotations|Quaternion Rotations]]
- [[_COMMUNITY_Ward Anisotropic and Tangent Space|Ward Anisotropic and Tangent Space]]
- [[_COMMUNITY_PBR Metalness Workflow|PBR Metalness Workflow]]
- [[_COMMUNITY_Vertex Normal Encoding|Vertex Normal Encoding]]
- [[_COMMUNITY_Vertex and Fragment Shader Basics|Vertex and Fragment Shader Basics]]
- [[_COMMUNITY_Cube Maps and Skyboxes|Cube Maps and Skyboxes]]
- [[_COMMUNITY_Command Buffer Recording|Command Buffer Recording]]
- [[_COMMUNITY_GLSL Control Flow on GPU|GLSL Control Flow on GPU]]
- [[_COMMUNITY_GLSL Literals and Casting|GLSL Literals and Casting]]

## God Nodes (most connected - your core abstractions)
1. `Computer Graphics exam project descriptions (Projects rules document)` - 42 edges
2. `L09 - Light Models and basic BRDFs` - 37 edges
3. `L06 - Depth testing and Meshes` - 30 edges
4. `L08 - Pipelines` - 30 edges
5. `L11 - UV mapping and Textures` - 27 edges
6. `L05 - View and World Matrices` - 25 edges
7. `Computer Graphics exam rules (document)` - 23 edges
8. `L01 - Graphics Adapters, Colors and 2D Drawings` - 22 edges
9. `L07 - Rendering` - 22 edges
10. `L03 - Advanced Transforms and Parallel Projections` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Cone Tip Normal Workaround` --references--> `L10 - Smooth shading`  [INFERRED]
  Excercises/E06 - Smooth surfaces  Advanced BRDF Techniques.txt → Lessons/L10 - Smooth shading.txt
- `getSixAxis() in Starter.hpp` --conceptually_related_to--> `Computer Graphics exam project descriptions (Projects rules document)`  [AMBIGUOUS]
  Excercises/E04 - Controls and Navigation.txt → Project/2 - Projects rules.txt
- `Local coordinates mapped to screen through local, view and projection transforms` --references--> `L02 - 3D coordinates and transforms`  [INFERRED]
  Project/2 - Projects rules.txt → Lessons/L02 - 3D coordinates and transforms.txt
- `Orthogonal projection matrix` --references--> `L03 - Advanced Transforms and Parallel Projections`  [INFERRED]
  Excercises/E01 - GLM and Projections Types.txt → Lessons/L03 - Advanced Transforms and Parallel Projections.txt
- `Projections parallel or perspective, with the scene viewable from different points` --references--> `L03 - Advanced Transforms and Parallel Projections`  [INFERRED]
  Project/2 - Projects rules.txt → Lessons/L03 - Advanced Transforms and Parallel Projections.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Local-to-Screen Vertex Transform Chain** — lessons_l05___view_and_world_matrices_local_coordinates, lessons_l05___view_and_world_matrices_world_matrix, lessons_l05___view_and_world_matrices_view_matrix, lessons_l04___axonometric_and_perspective_projections_perspective_projection_matrix, lessons_l03___advanced_transforms_and_parallel_projections_3d_normalized_screen_coordinates, lessons_l01___graphics_adapters__colors_and_2d_drawings_pixel_coordinates [INFERRED 0.85]
- **Three-Step Construction of the Orthogonal Projection Matrix** — lessons_l02___3d_coordinates_and_transforms_translation, lessons_l02___3d_coordinates_and_transforms_scaling, lessons_l02___3d_coordinates_and_transforms_mirroring, lessons_l03___advanced_transforms_and_parallel_projections_orthogonal_projection_matrix [EXTRACTED 1.00]
- **Look-At Camera Axis Construction** — lessons_l05___view_and_world_matrices_look_at, lessons_l05___view_and_world_matrices_up_vector, lessons_l05___view_and_world_matrices_cross_product, lessons_l05___view_and_world_matrices_camera_matrix, lessons_l05___view_and_world_matrices_orthonormal_inverse [EXTRACTED 1.00]
- **Scan-line rendering realized by the graphics pipeline stages** — lessons_l08___pipelines_graphics_pipeline, lessons_l08___pipelines_vertex_shader, lessons_l08___pipelines_rasterization, lessons_l08___pipelines_fragment_shader, lessons_l06___depth_testing_and__meshes_z_buffer, lessons_l08___pipelines_scan_line_rendering [EXTRACTED 1.00]
- **Standard real-time material: light model plus diffuse and specular BRDF terms** — lessons_l09___light_models_and_basic_brdfs_light_model, lessons_l09___light_models_and_basic_brdfs_lambert_reflection, lessons_l09___light_models_and_basic_brdfs_phong_specular_reflection, lessons_l09___light_models_and_basic_brdfs_blinn_specular_reflection, lessons_l09___light_models_and_basic_brdfs_diffuse_color, lessons_l09___light_models_and_basic_brdfs_specular_color [EXTRACTED 1.00]
- **Texture sampling flow from interpolated UV to filtered texel** — lessons_l11___uv_mapping_and_textures_uv_coordinates, lessons_l10___smooth_shading_perspective_correct_interpolation, lessons_l11___uv_mapping_and_textures_texel, lessons_l11___uv_mapping_and_textures_texture_filtering, lessons_l11___uv_mapping_and_textures_mip_mapping [INFERRED 0.85]
- **Uniform data path from application to shader** — lessons_l13___vulkan_rendering_part_ii_uniform_block, lessons_l13___vulkan_rendering_part_ii_descriptor_set_layout, lessons_l13___vulkan_rendering_part_ii_descriptor_set, lessons_l13___vulkan_rendering_part_ii_descriptor_pool, lessons_l13___vulkan_rendering_part_ii_pipeline_layout, lessons_l15___pipelines__render_passes_and_command_buffers_vkcmdbinddescriptorsets [EXTRACTED 1.00]
- **Vertex data path from C++ struct to vertex shader** — lessons_l12___vulkan_rendering_part_i_vertexdescriptor, lessons_l12___vulkan_rendering_part_i_vertex_binding, lessons_l12___vulkan_rendering_part_i_location, lessons_l12___vulkan_rendering_part_i_input_assembler, lessons_l12___vulkan_rendering_part_i_vertex_shader, lessons_l15___pipelines__render_passes_and_command_buffers_vertex_buffer [EXTRACTED 1.00]
- **Render pass creation flow** — lessons_l15___pipelines__render_passes_and_command_buffers_attachment_description, lessons_l15___pipelines__render_passes_and_command_buffers_attachment_reference, lessons_l15___pipelines__render_passes_and_command_buffers_subpass_description, lessons_l15___pipelines__render_passes_and_command_buffers_subpass_dependency, lessons_l15___pipelines__render_passes_and_command_buffers_vkcreaterenderpass [EXTRACTED 1.00]
- **World-View-Projection transform chain from model to pixel** — excercises_e02___matrices_wrap_up_world_matrix, excercises_e02___matrices_wrap_up_look_in_direction, excercises_e01___glm_and_projections_types_perspective_projection_matrix, excercises_e02___matrices_wrap_up_world_view_projection_matrix, excercises_e02___matrices_wrap_up_perspective_division, excercises_e02___matrices_wrap_up_screen_transform [EXTRACTED 1.00]
- **Walk navigation per-frame update loop** — excercises_e04___controls_and_navigation_getsixaxis, excercises_e04___controls_and_navigation_six_axes, excercises_e04___controls_and_navigation_linear_and_angular_speed, excercises_e04___controls_and_navigation_unitary_movement_vectors, excercises_e04___controls_and_navigation_walk_update_cycle, excercises_e02___matrices_wrap_up_look_in_direction [EXTRACTED 1.00]
- **GLSL to SPIR-V toolchain and shader interfaces** — excercises_e05___glsl_and_shaders_glsl, excercises_e05___glsl_and_shaders_glslc, excercises_e05___glsl_and_shaders_spirv, excercises_e05___glsl_and_shaders_vertex_shader, excercises_e05___glsl_and_shaders_uniform_blocks, excercises_e05___glsl_and_shaders_gl_position [EXTRACTED 1.00]
- **Cook-Torrance Specular as Product of D, F and G** — excercises_e06___smooth_surfaces__advanced_brdf_techniques_cook_torrance_reflection_model, excercises_e06___smooth_surfaces__advanced_brdf_techniques_distribution_term, excercises_e06___smooth_surfaces__advanced_brdf_techniques_fresnel_term, excercises_e06___smooth_surfaces__advanced_brdf_techniques_geometric_term, excercises_e06___smooth_surfaces__advanced_brdf_techniques_roughness_parameter, excercises_e06___smooth_surfaces__advanced_brdf_techniques_half_vector [EXTRACTED 1.00]
- **PBR + IBL Specular Pipeline (split-sum)** — excercises_e08___image_based_lighting_split_sum_approximation, excercises_e08___image_based_lighting_prefiltered_term, excercises_e08___image_based_lighting_brdf_term, excercises_e08___image_based_lighting_cube_maps, excercises_e08___image_based_lighting_environment_filtering, excercises_e07___advanced_texturing_pbr [EXTRACTED 1.00]
- **Tangent Space Normal Mapping Frame Construction** — excercises_e07___advanced_texturing_tangent_space_normal_maps, excercises_e07___advanced_texturing_tbn_frame_matrix, excercises_e07___advanced_texturing_tangent_and_bitangent_vectors, excercises_e07___advanced_texturing_gram_schmidt_orthonormalization, excercises_e07___advanced_texturing_on_the_fly_tangent_bitangent, excercises_e07___advanced_texturing_normal_map_color_encoding [EXTRACTED 1.00]
- **Mandatory Vulkan setup sections the project code must contain** — project_2___projects_rules_explicit_geometry_and_texture_loading, project_2___projects_rules_shaders_written_by_students, project_2___projects_rules_vertex_formats_and_uniform_blocks, project_2___projects_rules_pipelines_and_render_passes, project_2___projects_rules_command_buffers_and_draw_calls, project_2___projects_rules_mandatory_cpp_vulkan_starter [EXTRACTED 1.00]
- **Mandatory local-to-screen transform chain** — project_2___projects_rules_transform_chain, project_2___projects_rules_world_matrix_composition, project_2___projects_rules_rotations_euler_or_quaternions, project_2___projects_rules_view_matrix_lookat_or_lookindirection, project_2___projects_rules_projections_and_multiple_viewpoints [EXTRACTED 1.00]
- **Complete registration procedure required to be admitted to the exam** — project_1___computer_graphics_exam_rules_two_or_three_registrations, project_1___computer_graphics_exam_rules_online_services_registration, project_1___computer_graphics_exam_rules_extra_days_form_per_group, project_2___projects_rules_microsoft_form_registration, project_2___projects_rules_individual_registration_per_member, project_2___projects_rules_unregistered_not_admitted, project_3___project_registration_link_form_url [INFERRED 0.95]

## Communities (36 total, 2 thin omitted)

### Community 0 - "3D Transforms and Projections"
Cohesion: 0.08
Nodes (57): Pixel Aspect Ratio, L02 - 3D coordinates and transforms, 4x4 Matrix Transforms, Affine Transforms, Composition of Transformations, Flattening (Zero Scale Factor), Homogeneous Coordinates, Identity Transform (+49 more)

### Community 1 - "Rendering Equation and BRDF Theory"
Cohesion: 0.09
Nodes (53): L07 - Rendering, BRDF (Bidirectional Reflectance Distribution Function), BSDF (Bidirectional Scattering Distribution Function), BSSRDF, BTDF (Bidirectional Transmittance Distribution Function), Emission Term Le, Energy, Energy Conservation (+45 more)

### Community 2 - "Project Topics and Requirements"
Cohesion: 0.07
Nodes (52): E08 - Image Based Lighting, No deadline, but all the process must be completed before the last call in February 2027, Project topic and student names must be entered into a specific form, Computer Graphics exam project topics (document), Topic: Animal Herding Simulator, Topic: Dungeon Tavern NPC, Topic: Enchanted Forest Wanderer, Topic: Haunted Castle Explorer (+44 more)

### Community 3 - "Meshes and Depth Testing"
Cohesion: 0.10
Nodes (41): L06 - Depth testing and Meshes, 2-Manifold Topology, Back-Face Culling, Edge, Vertex Reuse Constraint, Face, Hidden Surfaces Elimination, Index Array (+33 more)

### Community 4 - "Mesh Construction and Clipping"
Cohesion: 0.06
Nodes (39): Clipping coordinates, glm::cross, glm::normalize and column constructor, Perspective division (normalization step), Screen transform to pixel coordinates, E03 - Mesh construction, Back-face culling, Clipping, Cost of clipping (+31 more)

### Community 5 - "Course Intro and Raster Basics"
Cohesion: 0.09
Nodes (33): L00 - Introduction, Application Fields of Computer Graphics, Course Structure, C++ as the Host Language, Exam: Project and Oral Discussion, Hardware Requirements, Steep Learning Curve of CG Applications, Vulkan (+25 more)

### Community 6 - "Pipelines and Shading Stages"
Cohesion: 0.18
Nodes (26): Visibility Term V(x,y), L08 - Pipelines, Anti-Aliasing, Color Blending, Compute Pipeline, Direct Light Sources, Fragment, Fragment Shader (+18 more)

### Community 7 - "Textures, Samplers and Descriptors"
Cohesion: 0.12
Nodes (25): Descriptor Pool, Descriptor Set, DescriptorSet class (Starter.hpp), DPSZs (Descriptor Pool SiZes), map() method, L14 - Vulkan Rendering Part III, Sampler Address Modes, Anisotropic Filtering (+17 more)

### Community 8 - "Exam Rules and Grading"
Cohesion: 0.11
Nodes (25): Computer Graphics exam rules (document), Every group component must be present for the project presentation, The exam day and room in the system are only one of the possible presentation days, Extra dates for special needs, but the mark is only registered on an official exam date, Form defining extra days and rooms, each group selects one available day, Final mark is the sum of project score and answers score, with laude in special circumstances, Form based registration system for choosing the presentation date, The 10-minute deadline is hard (+17 more)

### Community 9 - "Keyframe Animation and Interpolation"
Cohesion: 0.10
Nodes (23): Gram-Schmidt Re-orthogonalisation of the TBN, E09 - Animation, 3D Animation, Animation Channels, Animation Curves and Timeline Authoring, The Animator's Survival Kit (Richard Williams), Baked Animation Playback, Animation Baking (+15 more)

### Community 10 - "UV Mapping and Texture Filtering"
Cohesion: 0.22
Nodes (21): Diffuse Color mD, L11 - UV mapping and Textures, Anisotropic Filtering, Clamp Addressing, Constant (Border) Addressing, (Bi)linear Interpolation Filter, Magnification Filtering, Minification Filtering (+13 more)

### Community 11 - "Render Passes and Attachments"
Cohesion: 0.12
Nodes (21): Attachment, Deferred Rendering, Depth Buffer Attachment, G-Buffer, Render Pass, Render Subpass, RenderPass object (Starter.hpp), Shadow Map Pass (+13 more)

### Community 12 - "Cook-Torrance and Microfacet BRDFs"
Cohesion: 0.17
Nodes (16): Beckmann Distribution Version, Blinn 1977 - Model of Light Reflection for Computer Synthesized Pictures, Blinn Distribution Version, Cook-Torrance Reflection Model, Distribution Term D, Fresnel Principle, Fresnel Term F, Geometric Term G (+8 more)

### Community 13 - "Graphics Pipeline Fixed State"
Cohesion: 0.19
Nodes (16): Input Assembler, Pipeline object (Starter.hpp), Triangle Lists and Strips, vkCmdDraw(), L15 - Pipelines, Render Passes and Command Buffers, Back-face Culling (cullMode, frontFace), Color Blending, Depth and Stencil State (VkPipelineDepthStencilStateCreateInfo) (+8 more)

### Community 14 - "Vertex Formats and Model Loading"
Cohesion: 0.21
Nodes (14): L12 - Vulkan Rendering Part I, Index Buffer, Layout (Vulkan term), Location (shader slot), Model (Starter.hpp), Model file formats (OBJ, GLTF, MGCG), Starter.hpp wrapper library, Vertex Input Descriptor (+6 more)

### Community 15 - "Uniform Buffers and Descriptor Sets"
Cohesion: 0.20
Nodes (14): The Rendering Process (nested loops), L13 - Vulkan Rendering Part II, Uniform Alignment Requirements (alignas), Descriptor Set Layout, Material (BRDF parameter set), Pipeline Layout, Push Constants, Descriptor Set ID (Set) (+6 more)

### Community 16 - "Projection Matrices in GLM"
Cohesion: 0.20
Nodes (11): E01 - GLM and Projections Types, Cavalier and Cabinet projections, FOV / aspect ratio to frustum boundaries, glm::frustum, glm::ortho, glm::perspective, Oblique projections, Orthogonal projection matrix (+3 more)

### Community 17 - "GLM and GLSL Types"
Cohesion: 0.20
Nodes (11): GLM library, GLM header inclusion (glm.hpp, matrix_transform.hpp), glm::inverse and glm::transpose, glm::mat4, glm::scale, glm::translate, glm::vec3 and glm::vec4, Recovering the viewer position (+3 more)

### Community 18 - "Camera Navigation Models"
Cohesion: 0.29
Nodes (11): E04 - Controls and Navigation, Camera navigation models, Damped character facing direction, Third-person character camera, Camera damping, Fly navigation model, Ground motion model, Third-person spaceship camera (+3 more)

### Community 19 - "View and World Matrices in GLM"
Cohesion: 0.27
Nodes (10): E02 - Matrices Wrap up, Euler angles and glm::eulerAngleYXZ, Five transform steps (World, View, Projection, Normalization, Screen), glm::lookAt, Look-at view matrix, Look-in-direction view matrix in GLM, Complete projection example (tetrahedron starship), Roll in look-at matrices (+2 more)

### Community 20 - "Normal Maps and PBR Materials"
Cohesion: 0.20
Nodes (10): Indirect Lighting, Normal Map Colour Encoding and Decoding, Normal Maps, Object Space Normal Maps, PBR Material Libraries (freepbr.com), PBR Material Texture Set and Channel Packing, Ambient Occlusion, cmftStudio Environment Filtering Tool (+2 more)

### Community 21 - "Command Buffer Binding and Instancing"
Cohesion: 0.20
Nodes (9): Vertex Binding, vkCmdDrawIndexed(), Binding (uniform resource index), gl_InstanceIndex, Instancing, Resource Binding in the Command Buffer, Vertex Buffer / vkCmdBindVertexBuffers(), vkCmdBindDescriptorSets() (+1 more)

### Community 22 - "Input Handling and Six-Axis Controls"
Cohesion: 0.28
Nodes (9): Control axis normalization to [-1, +1], Frame delta time with <chrono>, getSixAxis() in Starter.hpp, GLFW, GLFW input functions (glfwGetKey, glfwGetCursorPos, glfwGetMouseButton), Linear and angular speed with delta time, Converting mouse motion to axis values, Six-axis control values (mx, my, mz, rx, ry, rz) (+1 more)

### Community 23 - "Oren-Nayar and Toon Shading"
Cohesion: 0.29
Nodes (8): E06 - Smooth surfaces and Advanced BRDF Techniques, Material Emission, Oren-Nayar A and B Parameters, Oren-Nayar Diffuse Model, Retroreflection, 1D Texture Look-Up for Toon Ramps, Toon Shading, Smooth Shading

### Community 24 - "Ambient and Image Based Lighting"
Cohesion: 0.32
Nodes (8): Ambient Lighting, Ambient Light Reflection Colour mA, E07 - Advanced Texturing, Hemispheric Lighting, Cubic Mapping Encoding of lA(x), Image Based Lighting, Six Direction Ambient Lighting, Spherical Harmonics

### Community 25 - "Axonometric Projections and Rotation"
Cohesion: 0.29
Nodes (6): Axonometric projections, Dimetric projection, glm::rotate, Isometric projection, Trimetric projection, Unitary movement vectors ux, uy, uz

### Community 26 - "Shader Interpolation Qualifiers"
Cohesion: 0.29
Nodes (7): CPU to Shader Data, flat and noperspective qualifiers, Intra-Shader Data, Perspective Correct Interpolation, Pipeline Interfaces, Vertex Attributes, Vertex Data

### Community 27 - "Quaternion Rotations"
Cohesion: 0.40
Nodes (6): glm::quat, Quaternion rotations, Quaternion Euler-order caveat, World matrix from a quaternion, World-space vs local-space quaternion rotation, Local/global coordinates model in quaternion form

### Community 28 - "Ward Anisotropic and Tangent Space"
Cohesion: 0.33
Nodes (6): Anisotropic Materials, Half-Vector, Tangent and Bi-Tangent Vectors, Ward Anisotropic Specular Model, Tangent and Bitangent Vectors (UV aligned), Tangent Storage vs On-the-Fly Computation Trade-off

### Community 29 - "PBR Metalness Workflow"
Cohesion: 0.40
Nodes (6): Albedo, Metallic vs Dielectric Materials, Metalness, PBR - Physically Based Rendering, IBL Diffuse Approximation, PBR combined with IBL

### Community 30 - "Vertex Normal Encoding"
Cohesion: 0.40
Nodes (5): Cone Tip Normal Workaround, Cube Mesh Encoding (24 vertices, 12 triangles), Cylinder Mesh Encoding (34 vertices, 32 triangles), Sphere Mesh Encoding (26 vertices, 48 triangles), Vertex Normal Vector Encoding

### Community 31 - "Vertex and Fragment Shader Basics"
Cohesion: 0.50
Nodes (5): Draw Call, Fragment Shader, gl_Position, gl_VertexIndex, Vertex Shader

### Community 32 - "Cube Maps and Skyboxes"
Cohesion: 0.67
Nodes (4): Cube Maps, Environment Maps, Perfect Fake Mirrors, Skybox

### Community 33 - "Command Buffer Recording"
Cohesion: 0.50
Nodes (3): Command Buffer, Queue submission, Command buffer population in Starter.hpp

## Ambiguous Edges - Review These
- `getSixAxis() in Starter.hpp` → `Computer Graphics exam project descriptions (Projects rules document)`  [AMBIGUOUS]
  Excercises/E04 - Controls and Navigation.txt · relation: conceptually_related_to

## Knowledge Gaps
- **73 isolated node(s):** `Course Structure`, `Exam: Project and Oral Discussion`, `Application Fields of Computer Graphics`, `Pixel`, `Raster Graphics` (+68 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `getSixAxis() in Starter.hpp` and `Computer Graphics exam project descriptions (Projects rules document)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Computer Graphics exam project descriptions (Projects rules document)` connect `Project Topics and Requirements` to `Mesh Construction and Clipping`, `Exam Rules and Grading`, `Graphics Pipeline Fixed State`, `Uniform Buffers and Descriptor Sets`, `Input Handling and Six-Axis Controls`?**
  _High betweenness centrality (0.526) - this node is a cross-community bridge._
- **Why does `Projections parallel or perspective, with the scene viewable from different points` connect `Project Topics and Requirements` to `Projection Matrices in GLM`, `3D Transforms and Projections`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `L03 - Advanced Transforms and Parallel Projections` connect `3D Transforms and Projections` to `Projection Matrices in GLM`, `Keyframe Animation and Interpolation`, `Project Topics and Requirements`, `Course Intro and Raster Basics`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `L09 - Light Models and basic BRDFs` (e.g. with `Recovering the viewer position` and `Material (BRDF parameter set)`) actually correct?**
  _`L09 - Light Models and basic BRDFs` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `L06 - Depth testing and Meshes` (e.g. with `E03 - Mesh construction` and `Depth and Stencil State (VkPipelineDepthStencilStateCreateInfo)`) actually correct?**
  _`L06 - Depth testing and Meshes` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `L08 - Pipelines` (e.g. with `Shader-pipeline communication: in and out variables` and `Input Assembler`) actually correct?**
  _`L08 - Pipelines` has 4 INFERRED edges - model-reasoned connections that need verification._