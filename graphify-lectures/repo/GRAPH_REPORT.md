# Graph Report - Computer_Graphics  (2026-08-02)

## Corpus Check
- Corpus is ~31,578 words - fits in a single context window. You may not need a graph.

## Summary
- 919 nodes · 1899 edges · 73 communities (63 shown, 10 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 70 edges (avg confidence: 0.86)
- Token cost: 66,359 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Skeletal Animation and Character Chain|Skeletal Animation and Character Chain]]
- [[_COMMUNITY_BaseProject Vulkan Core State|BaseProject Vulkan Core State]]
- [[_COMMUNITY_TextMaker HUD Rendering|TextMaker HUD Rendering]]
- [[_COMMUNITY_Graphics Pipeline Object|Graphics Pipeline Object]]
- [[_COMMUNITY_Vertex Descriptor and Bindings|Vertex Descriptor and Bindings]]
- [[_COMMUNITY_Collider Debug Visualizer|Collider Debug Visualizer]]
- [[_COMMUNITY_Collision Intersection Tests|Collision Intersection Tests]]
- [[_COMMUNITY_Model Loading and GLTF Nodes|Model Loading and GLTF Nodes]]
- [[_COMMUNITY_Collider Construction and Types|Collider Construction and Types]]
- [[_COMMUNITY_Text Block Layout|Text Block Layout]]
- [[_COMMUNITY_Physical Device and Swapchain|Physical Device and Swapchain]]
- [[_COMMUNITY_Scene Asset Registry|Scene Asset Registry]]
- [[_COMMUNITY_App Class and Camera State|App Class and Camera State]]
- [[_COMMUNITY_Asset File Types OBJ GLTF|Asset File Types OBJ GLTF]]
- [[_COMMUNITY_Vulkan Init and Lifecycle|Vulkan Init and Lifecycle]]
- [[_COMMUNITY_Named Command Buffers|Named Command Buffers]]
- [[_COMMUNITY_Render Pass Attachments|Render Pass Attachments]]
- [[_COMMUNITY_Framebuffer Attachments|Framebuffer Attachments]]
- [[_COMMUNITY_Texture Sampler|Texture Sampler]]
- [[_COMMUNITY_Render Pass Object|Render Pass Object]]
- [[_COMMUNITY_Movement Cheats and Build Setup|Movement Cheats and Build Setup]]
- [[_COMMUNITY_Texture Creation and Mipmaps|Texture Creation and Mipmaps]]
- [[_COMMUNITY_Descriptor Layouts and Shader Modules|Descriptor Layouts and Shader Modules]]
- [[_COMMUNITY_Course Corpus and Framework Shaders|Course Corpus and Framework Shaders]]
- [[_COMMUNITY_Scene Instance Record|Scene Instance Record]]
- [[_COMMUNITY_Exam Rules and Project Topics|Exam Rules and Project Topics]]
- [[_COMMUNITY_Font Definition and Atlas|Font Definition and Atlas]]
- [[_COMMUNITY_Image Layout Transitions|Image Layout Transitions]]
- [[_COMMUNITY_Buffer and Image Allocation|Buffer and Image Allocation]]
- [[_COMMUNITY_Text Colors and Alignment|Text Colors and Alignment]]
- [[_COMMUNITY_Technique Definitions|Technique Definitions]]
- [[_COMMUNITY_Descriptor Set|Descriptor Set]]
- [[_COMMUNITY_Light Models and Main Shaders|Light Models and Main Shaders]]
- [[_COMMUNITY_Scene Lifecycle Hooks|Scene Lifecycle Hooks]]
- [[_COMMUNITY_OBJ Loading and Screenshot|OBJ Loading and Screenshot]]
- [[_COMMUNITY_Command Buffer Submission|Command Buffer Submission]]
- [[_COMMUNITY_Vulkan Debug Messenger|Vulkan Debug Messenger]]
- [[_COMMUNITY_Descriptor Layout Bindings|Descriptor Layout Bindings]]
- [[_COMMUNITY_Font Character Metrics|Font Character Metrics]]
- [[_COMMUNITY_Model View Projection Uniforms|Model View Projection Uniforms]]
- [[_COMMUNITY_Global Uniform and Lighting|Global Uniform and Lighting]]
- [[_COMMUNITY_Main Entry and Vertex Format|Main Entry and Vertex Format]]
- [[_COMMUNITY_Collider Definitions in Scene|Collider Definitions in Scene]]
- [[_COMMUNITY_AABB Extents|AABB Extents]]
- [[_COMMUNITY_Cleanup Paths|Cleanup Paths]]
- [[_COMMUNITY_Command Buffer Initializers|Command Buffer Initializers]]
- [[_COMMUNITY_Image Memory Barriers|Image Memory Barriers]]
- [[_COMMUNITY_Application Main Loop|Application Main Loop]]
- [[_COMMUNITY_Descriptor Pool Sizes|Descriptor Pool Sizes]]
- [[_COMMUNITY_Shown Collider Record|Shown Collider Record]]
- [[_COMMUNITY_Vertex Descriptor Reference|Vertex Descriptor Reference]]
- [[_COMMUNITY_Texture Definitions|Texture Definitions]]
- [[_COMMUNITY_Validation Debug Callback|Validation Debug Callback]]
- [[_COMMUNITY_E13 Command Buffer Lesson|E13 Command Buffer Lesson]]
- [[_COMMUNITY_Tangent Space Chain E14 E15|Tangent Space Chain E14 E15]]
- [[_COMMUNITY_Technique Instances|Technique Instances]]
- [[_COMMUNITY_Instance Creation and Validation|Instance Creation and Validation]]
- [[_COMMUNITY_Depth Format Selection|Depth Format Selection]]
- [[_COMMUNITY_Text Vertex Format|Text Vertex Format]]
- [[_COMMUNITY_Window Resize Handling|Window Resize Handling]]
- [[_COMMUNITY_Stock Attachment Configurations|Stock Attachment Configurations]]
- [[_COMMUNITY_ColliderShow Push Constant|ColliderShow Push Constant]]
- [[_COMMUNITY_Swapchain Extent|Swapchain Extent]]
- [[_COMMUNITY_Image View Creation|Image View Creation]]
- [[_COMMUNITY_TextMaker Model Pair|TextMaker Model Pair]]
- [[_COMMUNITY_Mat3 Printing|Mat3 Printing]]
- [[_COMMUNITY_Run Script|Run Script]]
- [[_COMMUNITY_Vec2 Printing|Vec2 Printing]]
- [[_COMMUNITY_Vec4 Printing|Vec4 Printing]]
- [[_COMMUNITY_Image Create Info|Image Create Info]]
- [[_COMMUNITY_Memory Allocate Info|Memory Allocate Info]]
- [[_COMMUNITY_E09 Texture Mapping Lesson|E09 Texture Mapping Lesson]]
- [[_COMMUNITY_E16 Rendering Part 3 Lesson|E16 Rendering Part 3 Lesson]]

## God Nodes (most connected - your core abstractions)
1. `BaseProject` - 167 edges
2. `Collider` - 56 edges
3. `TextMaker` - 52 edges
4. `Scene` - 51 edges
5. `Pipeline` - 50 edges
6. `Skeleton26ReplaceName` - 47 edges
7. `ColliderShow` - 43 edges
8. `Model` - 41 edges
9. `RenderPass` - 38 edges
10. `size` - 36 edges

## Surprising Connections (you probably didn't know these)
- `E13 - Setting up rendering Part 2 (UBO, Global UBO, command buffer)` --conceptually_related_to--> `GlobalUniformBufferObject`  [INFERRED]
  docs/riassunto-corso.md → skeleton/source/src/main.cpp
- `CheatFlags struct (grouped debug/cheat toggles, no persistence)` --semantically_similar_to--> `Libs.cpp compiled with -g0 to cut build cost`  [AMBIGUOUS] [semantically similar]
  notes.md → skeleton/CMakeLists.txt
- `E06 - Motion systems (skeletal animation)` --conceptually_related_to--> `AnimBlender`  [INFERRED]
  docs/riassunto-corso.md → skeleton/source/include/modules/Animations.hpp
- `Optional collision detection (no-clip cheat)` --references--> `collidesWith`  [INFERRED]
  notes.md → skeleton/source/include/modules/Colliders.hpp
- `Grounded state computed against colliders, not from zero vertical velocity` --references--> `getExtents`  [INFERRED]
  notes.md → skeleton/source/include/modules/Colliders.hpp

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **First-person movement system (input, gravity, jump, sprint, ground contact)** — notes_camera, notes_gravity, notes_jump, notes_sprint, notes_grounded_detection, skeleton_source_src_main_skeleton26replacename_gamelogic, skeleton_source_include_modules_starter_baseproject_getsixaxis [INFERRED 0.90]
- **Cheat toggles vs numeric tuning split on the app class** — notes_cheatflags, notes_movementparams, skeleton_source_src_main_skeleton26replacename_cheats, skeleton_source_src_main_skeleton26replacename_movement, skeleton_source_src_main_skeleton26replacename [INFERRED 0.90]
- **Main object render pipeline (vert + frag + UBO/GUBO + shader build step)** — skeleton_source_shaders_tochangesimplepos_vert_main, skeleton_source_shaders_tochangeblinnfrompos_frag_main, skeleton_source_src_main_uniformbufferobject, skeleton_source_src_main_globaluniformbufferobject, skeleton_source_src_main_skeleton26replacename_updateuniformbuffer, skeleton_cmakelists_shaders_target [INFERRED 0.90]

## Communities (73 total, 10 thin omitted)

### Community 0 - "Skeletal Animation and Character Chain"
Cohesion: 0.05
Nodes (64): Animated character chain (E05 -> E06 -> E07, one shared pipeline), CookTorranceForCharacter.frag (Cook-Torrance BRDF), E05 - Build the models, E06 - Motion systems (skeletal animation), E07 - Shadow map, PosNormUvTanWeights.vert (skinning vertex shader), Animations, AF (+56 more)

### Community 1 - "BaseProject Vulkan Core State"
Cohesion: 0.05
Nodes (42): BaseProject, commandPool, currentFrame, debugMessenger, descriptorPool, device, DPSZs, framebufferResized (+34 more)

### Community 2 - "TextMaker HUD Rendering"
Cohesion: 0.09
Nodes (37): unordered_map, VkCommandBuffer, TextMaker, atlasToUV, Blocks, BP, commandBufferMustUpdate, createTextDescriptorSetAndVertexLayout (+29 more)

### Community 3 - "Graphics Pipeline Object"
Cohesion: 0.07
Nodes (36): Pipeline, BP, cleanup, CM, compareOp, D, fragShaderModule, graphicsPipeline (+28 more)

### Community 4 - "Vertex Descriptor and Bindings"
Cohesion: 0.06
Nodes (33): VertexBindingDescriptorElement, binding, inputRate, stride, VertexComponent, hasIt, offset, VertexDescriptor (+25 more)

### Community 5 - "Collider Debug Visualizer"
Cohesion: 0.08
Nodes (17): ColliderShow, BP, clds, commandBufferMustUpdate, DS, DSL, M, P (+9 more)

### Community 6 - "Collision Intersection Tests"
Cohesion: 0.22
Nodes (25): checkIntersectAxis, collidesWith, collisionAABBAABB, collisionOOBBAABB, collisionOOBBOOBB, collisionPointAABB, collisionPointOOBB, collisionPointPoint (+17 more)

### Community 7 - "Model Loading and GLTF Nodes"
Cohesion: 0.12
Nodes (26): Node, Primitive, printMat4, printQuat, mat4, quat, Model, BP (+18 more)

### Community 8 - "Collider Construction and Types"
Cohesion: 0.14
Nodes (26): Collider, children, collisionBVHCollider, fitAABB, fitOOBB, getModelExtentsAABB, initAABB, initBVH (+18 more)

### Community 9 - "Text Block Layout"
Cohesion: 0.08
Nodes (25): TextBlock, Alignment, Bold, Fill, FontFace, fontId, h, Italic (+17 more)

### Community 10 - "Physical Device and Swapchain"
Cohesion: 0.12
Nodes (24): deviceReport, optional, checkDeviceExtensionSupport, checkIfItHasDeviceExtension, chooseSwapPresentMode, chooseSwapSurfaceFormat, createCommandPool, createSwapChain (+16 more)

### Community 11 - "Scene Asset Registry"
Cohesion: 0.08
Nodes (24): unordered_map, Scene, As, AsIds, AssetFileCount, BP, ColShow, GlobalColliders (+16 more)

### Community 12 - "App Class and Camera State"
Cohesion: 0.10
Nodes (18): First-person camera notes, GlobalUniformBufferObject in fragment shader (lightDir, lightColor, eyePos), vector, Skeleton26ReplaceName, camPitch, camPos, camYaw, DSglobal (+10 more)

### Community 13 - "Asset File Types OBJ GLTF"
Cohesion: 0.12
Nodes (21): material_t, ModelType, AssetFile, attrib, cleanup, GLTFmeshes, GLTFnodes, init (+13 more)

### Community 14 - "Vulkan Init and Lifecycle"
Cohesion: 0.23
Nodes (20): createDescriptorPool, createImageViews, createLogicalDevice, createSurface, createSyncObjects, initVulkan, pipelinesAndDescriptorSetsInit, recreateSwapChain (+12 more)

### Community 15 - "Named Command Buffers"
Cohesion: 0.12
Nodes (20): NamedCommandBuffersStates, pNCBfree, pNCBfunc, clearNamedCommandBuffer, clearNamedCommandBufferForImage, createCommandBuffer, submitCommandBuffer, updateCommandBuffers (+12 more)

### Community 16 - "Render Pass Attachments"
Cohesion: 0.11
Nodes (19): AttchmentType, AttachmentProperties, aspect, clearValue, doDepthTransition, finalLayout, format, initialLayout (+11 more)

### Community 17 - "Framebuffer Attachments"
Cohesion: 0.12
Nodes (18): FrameBufferAttachment, cleanup, createDescriptionAndReference, descr, freeSampler, getView, image, init (+10 more)

### Community 18 - "Texture Sampler"
Cohesion: 0.14
Nodes (17): VkBool32, getViewAndSampler, VkDescriptorImageInfo, getViewAndSampler, setSampler, TextureSampler, BP, cleanup (+9 more)

### Community 19 - "Render Pass Object"
Cohesion: 0.12
Nodes (17): RenderPass, attachments, BP, clearValues, colorAttchementsCount, dependencies, depthAttIdx, firstColorAttIdx (+9 more)

### Community 20 - "Movement Cheats and Build Setup"
Cohesion: 0.15
Nodes (16): CheatFlags struct (grouped debug/cheat toggles, no persistence), Gravity handled in the camera (optional, enables fly cheat), Grounded state computed against colliders, not from zero vertical velocity, Jump as upward velocity impulse on spacebar (fire output of getSixAxis), MovementParams struct (numeric tuning separated from on/off flags), Sprint on Ctrl polled via glfwGetKey (no momentum through a jump), POST_BUILD asset directory copy, Libs.cpp compiled with -g0 to cut build cost (+8 more)

### Community 21 - "Texture Creation and Mipmaps"
Cohesion: 0.23
Nodes (16): Texture, BP, createTexture, createTextureImage, createTextureImageView, imgs, init, initCubic (+8 more)

### Community 22 - "Descriptor Layouts and Shader Modules"
Cohesion: 0.18
Nodes (15): DescriptorSetLayout, Bindings, BP, cleanup, descriptorSetLayout, imgInfoSize, vector, createShaderModule (+7 more)

### Community 23 - "Course Corpus and Framework Shaders"
Cohesion: 0.19
Nodes (14): Course Knowledge Graph (E02-E17 corpus), E02 - Setup the environment, E03 (reduced variant of E02), E04 - 3D projections in Excel (WVP chain without code), E17 - Maze (instanced rendering, per-instance push constants, offscreen minimap), Optional collision detection (no-clip cheat), shaders target (glslc compile + SPIR-V copy stamps), ColliderShow.frag main (flat pass-through color) (+6 more)

### Community 24 - "Scene Instance Record"
Cohesion: 0.14
Nodes (14): mat4, Instance, C, D, DS, id, Iid, Mid (+6 more)

### Community 25 - "Exam Rules and Project Topics"
Cohesion: 0.18
Nodes (12): Exam rules and project constraints, First-person camera fits the three exploration topics, Mandatory technical content (Vulkan pipeline setup, WVP chain, vertex formats, navigation, light models and BRDF, direct/indirect lighting, materials and textures), The 8 official project topics, Starter.hpp must not be modified, Hardcoded input handling in Starter.hpp (WASD, arrows, mouse only with left button), Computer Graphics Final Project (PoliMi a.y. 2025-2026), getSixAxis (+4 more)

### Community 26 - "Font Definition and Atlas"
Cohesion: 0.18
Nodes (12): Font, faces, maxChar, minChar, texH, textureFile, texW, FontDef (+4 more)

### Community 27 - "Image Layout Transitions"
Cohesion: 0.27
Nodes (11): beginSingleTimeCommands, copyBufferToImage, endSingleTimeCommands, generateMipmaps, hasStencilComponent, transitionImageLayout, bind, VkCommandBuffer (+3 more)

### Community 28 - "Buffer and Image Allocation"
Cohesion: 0.24
Nodes (11): createBuffer, createImage, findMemoryType, VkBuffer, VkBufferUsageFlags, VkDeviceMemory, VkDeviceSize, VkImageCreateFlags (+3 more)

### Community 29 - "Text Colors and Alignment"
Cohesion: 0.20
Nodes (11): string, vec4, TextColorPushConstant, Fill, Shadow, Stroke, measureText, print (+3 more)

### Community 30 - "Technique Definitions"
Cohesion: 0.27
Nodes (10): vector, PipelineAndTexturesDefs, P, texDefs, TechniqueRef, id, init, Ntextures (+2 more)

### Community 31 - "Descriptor Set"
Cohesion: 0.20
Nodes (10): DescriptorSet, BP, cleanup, descriptorSets, Layout, map, toFree, uniformBuffers (+2 more)

### Community 32 - "Light Models and Main Shaders"
Cohesion: 0.22
Nodes (9): E08 - Light Models (Lambert, Blinn-Phong), E10 - Mesh Normals and Smoothing, E11 - Advanced BRDFs, E12 - Setting up rendering Part 1, Light models chain (E08 -> E12 -> E11), albedoMap sampler (set 1, binding 1) with sRGB decode, Screen-space normal reconstruction via dFdx/dFdy cross product, toChangeBlinnFromPos.frag main (Blinn-Phong from world position) (+1 more)

### Community 33 - "Scene Lifecycle Hooks"
Cohesion: 0.22
Nodes (8): vec4, VkCommandBuffer, localCleanup, pipelinesAndDescriptorSetsCleanup, pipelinesAndDescriptorSetsInit, populateCommandBuffer, refreshColliderVisualizer, setColliderStroke

### Community 34 - "OBJ Loading and Screenshot"
Cohesion: 0.29
Nodes (8): attrib_t, shape_t, removeBuffer, saveScreenshot, loadModelOBJ, makeOBJMesh, begin, end

### Community 35 - "Command Buffer Submission"
Cohesion: 0.25
Nodes (8): clearCommandBuffers, vks_initializers_fenceCreateInfo, vks_initializers_submitInfo, vulkanDevice_flushCommandBuffer, VkFenceCreateFlags, VkFenceCreateInfo, VkQueue, VkSubmitInfo

### Community 36 - "Vulkan Debug Messenger"
Cohesion: 0.36
Nodes (8): populateDebugMessengerCreateInfo, setupDebugMessenger, CreateDebugUtilsMessengerEXT(), DestroyDebugUtilsMessengerEXT(), VkAllocationCallbacks, VkDebugUtilsMessengerCreateInfoEXT, VkDebugUtilsMessengerEXT, VkInstance

### Community 37 - "Descriptor Layout Bindings"
Cohesion: 0.25
Nodes (8): DescriptorSetLayoutBinding, binding, count, flags, linkSize, type, VkDescriptorType, VkShaderStageFlags

### Community 38 - "Font Character Metrics"
Cohesion: 0.25
Nodes (8): CharData, height, width, x, xadvance, xoffset, y, yoffset

### Community 39 - "Model View Projection Uniforms"
Cohesion: 0.29
Nodes (8): ColliderShow UniformBufferObject (vpMat + strokeColors[MAX_COLLIDERS]), UniformBufferObject in vertex shader (mvpMat, mMat), mat4, Ar, ViewPrj, UniformBufferObject, mMat, mvpMat

### Community 40 - "Global Uniform and Lighting"
Cohesion: 0.32
Nodes (6): vec3, vec4, GlobalUniformBufferObject, eyePos, lightColor, lightDir

### Community 41 - "Main Entry and Vertex Format"
Cohesion: 0.29
Nodes (6): json, vec2, main(), Vertex, pos, UV

### Community 42 - "Collider Definitions in Scene"
Cohesion: 0.29
Nodes (7): AABB-based ground detection declared in scene.json, ColliderDef, children, hasCollider, params, type, visible

### Community 43 - "AABB Extents"
Cohesion: 0.29
Nodes (7): AABBextents, xMax, xMin, yMax, yMin, zMax, zMin

### Community 44 - "Cleanup Paths"
Cohesion: 0.29
Nodes (7): cleanup, cleanupSwapChain, localCleanup, pipelinesAndDescriptorSetsCleanup, destroy, cleanup, cleanup

### Community 45 - "Command Buffer Initializers"
Cohesion: 0.38
Nodes (7): vks_initializers_commandBufferAllocateInfo, vks_initializers_commandBufferBeginInfo, vulkanDevice_createCommandBuffer, VkCommandBufferAllocateInfo, VkCommandBufferBeginInfo, VkCommandBufferLevel, VkCommandPool

### Community 46 - "Image Memory Barriers"
Cohesion: 0.29
Nodes (7): vks_initializers_imageMemoryBarrier, vks_tools_insertImageMemoryBarrier, VkAccessFlags, VkImageLayout, VkImageMemoryBarrier, VkImageSubresourceRange, VkPipelineStageFlags

### Community 47 - "Application Main Loop"
Cohesion: 0.33
Nodes (6): drawFrame, initWindow, mainLoop, run, setWindowParameters, updateUniformBuffer

### Community 48 - "Descriptor Pool Sizes"
Cohesion: 0.33
Nodes (6): PoolSizes, sampledImagesInPool, samplersInPool, setsInPool, texturesInPool, uniformBlocksInPool

### Community 49 - "Shown Collider Record"
Cohesion: 0.40
Nodes (5): ShownCollider, c, len, start, Stroke

### Community 50 - "Vertex Descriptor Reference"
Cohesion: 0.40
Nodes (5): string, VertexDescriptorRef, id, init, VD

### Community 51 - "Texture Definitions"
Cohesion: 0.40
Nodes (5): VkDescriptorImageInfo, TextureDefs, fromInstance, info, pos

### Community 52 - "Validation Debug Callback"
Cohesion: 0.40
Nodes (5): debugCallback, VKAPI_ATTR, VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, VkDebugUtilsMessengerCallbackDataEXT

### Community 54 - "Tangent Space Chain E14 E15"
Cohesion: 0.50
Nodes (4): E14 - Advanced texturing (tangent space), E15 - Image Based Lighting, MeshTBN.vert (tangent-space vertex pipeline), Tangent space chain (E14 + E15 share MeshTBN.vert)

### Community 55 - "Technique Instances"
Cohesion: 0.50
Nodes (4): TechniqueInstances, I, InstanceCount, T

### Community 56 - "Instance Creation and Validation"
Cohesion: 0.50
Nodes (4): checkIfItHasExtension, checkValidationLayerSupport, createInstance, getRequiredExtensions

### Community 57 - "Depth Format Selection"
Cohesion: 0.50
Nodes (4): findDepthFormat, findSupportedFormat, VkFormatFeatureFlags, VkImageTiling

### Community 58 - "Text Vertex Format"
Cohesion: 0.50
Nodes (4): vec2, TextVertex, pos, texCoord

### Community 59 - "Window Resize Handling"
Cohesion: 0.67
Nodes (3): GLFWwindow, framebufferResizeCallback, onWindowResize

### Community 61 - "ColliderShow Push Constant"
Cohesion: 0.67
Nodes (3): ColliderShowPushConstant, colliderIndex, colliderType

### Community 62 - "Swapchain Extent"
Cohesion: 0.67
Nodes (3): chooseSwapExtent, VkExtent2D, VkSurfaceCapabilitiesKHR

### Community 63 - "Image View Creation"
Cohesion: 0.67
Nodes (3): createImageView, VkImageAspectFlags, VkImageViewType

### Community 64 - "TextMaker Model Pair"
Cohesion: 0.67
Nodes (3): TextMakerAndModel, M, txt

## Ambiguous Edges - Review These
- `Instance` → `albedoMap sampler (set 1, binding 1) with sRGB decode`  [AMBIGUOUS]
  skeleton/source/shaders/toChangeBlinnFromPos.frag · relation: shares_data_with
- `CheatFlags struct (grouped debug/cheat toggles, no persistence)` → `Libs.cpp compiled with -g0 to cut build cost`  [AMBIGUOUS]
  notes.md · relation: semantically_similar_to

## Knowledge Gaps
- **385 isolated node(s):** `run.sh script`, `time`, `T`, `Q`, `S` (+380 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Instance` and `albedoMap sampler (set 1, binding 1) with sRGB decode`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **What is the exact relationship between `CheatFlags struct (grouped debug/cheat toggles, no persistence)` and `Libs.cpp compiled with -g0 to cut build cost`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `BaseProject` connect `BaseProject Vulkan Core State` to `TextMaker HUD Rendering`, `Graphics Pipeline Object`, `Vertex Descriptor and Bindings`, `Collider Debug Visualizer`, `Model Loading and GLTF Nodes`, `Collider Construction and Types`, `Physical Device and Swapchain`, `Scene Asset Registry`, `App Class and Camera State`, `Asset File Types OBJ GLTF`, `Vulkan Init and Lifecycle`, `Named Command Buffers`, `Framebuffer Attachments`, `Texture Sampler`, `Render Pass Object`, `Texture Creation and Mipmaps`, `Descriptor Layouts and Shader Modules`, `Course Corpus and Framework Shaders`, `Exam Rules and Project Topics`, `Image Layout Transitions`, `Buffer and Image Allocation`, `Descriptor Set`, `OBJ Loading and Screenshot`, `Command Buffer Submission`, `Vulkan Debug Messenger`, `Cleanup Paths`, `Command Buffer Initializers`, `Image Memory Barriers`, `Application Main Loop`, `Descriptor Pool Sizes`, `Validation Debug Callback`, `Instance Creation and Validation`, `Depth Format Selection`, `Window Resize Handling`, `Stock Attachment Configurations`, `Swapchain Extent`, `Image View Creation`, `Mat3 Printing`, `Vec2 Printing`, `Vec4 Printing`, `Image Create Info`, `Memory Allocate Info`?**
  _High betweenness centrality (0.278) - this node is a cross-community bridge._
- **Why does `TextMaker` connect `TextMaker HUD Rendering` to `TextMaker Model Pair`, `BaseProject Vulkan Core State`, `Skeletal Animation and Character Chain`, `Graphics Pipeline Object`, `Vertex Descriptor and Bindings`, `Model Loading and GLTF Nodes`, `Text Block Layout`, `Main Entry and Vertex Format`, `App Class and Camera State`, `Render Pass Object`, `Texture Creation and Mipmaps`, `Descriptor Layouts and Shader Modules`, `Font Definition and Atlas`, `Text Colors and Alignment`, `Descriptor Set`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `Scene` connect `Scene Asset Registry` to `Skeletal Animation and Character Chain`, `Scene Lifecycle Hooks`, `BaseProject Vulkan Core State`, `Vertex Descriptor and Bindings`, `Collider Debug Visualizer`, `Model Loading and GLTF Nodes`, `Collider Construction and Types`, `Main Entry and Vertex Format`, `Collider Definitions in Scene`, `App Class and Camera State`, `Asset File Types OBJ GLTF`, `Vertex Descriptor Reference`, `Texture Creation and Mipmaps`, `Technique Instances`, `Scene Instance Record`, `Technique Definitions`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **What connects `run.sh script`, `time`, `T` to the rest of the system?**
  _385 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Skeletal Animation and Character Chain` be split into smaller, more focused modules?**
  _Cohesion score 0.05174825174825175 - nodes in this community are weakly interconnected._