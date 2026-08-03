# Graph Report - .  (2026-08-03)

## Corpus Check
- 16 files · ~30,780 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 242 nodes · 484 edges · 19 communities detected
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 43 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Collision Detection (AABBOOBBBVH)|Collision Detection (AABB/OOBB/BVH)]]
- [[_COMMUNITY_Framework Bootstrap (BaseProjectStarter)|Framework Bootstrap (BaseProject/Starter)]]
- [[_COMMUNITY_Vulkan Resource Creation|Vulkan Resource Creation]]
- [[_COMMUNITY_Debug Collider Rendering & Cheat Flags|Debug Collider Rendering & Cheat Flags]]
- [[_COMMUNITY_Vulkan Device & Swapchain Setup|Vulkan Device & Swapchain Setup]]
- [[_COMMUNITY_App Lifecycle (initmain loopcleanup)|App Lifecycle (init/main loop/cleanup)]]
- [[_COMMUNITY_Text Rendering (TextMaker)|Text Rendering (TextMaker)]]
- [[_COMMUNITY_Skeletal Animation|Skeletal Animation]]
- [[_COMMUNITY_Mesh Loading (glTFOBJ)|Mesh Loading (glTF/OBJ)]]
- [[_COMMUNITY_Render Pass & Vertex Layout|Render Pass & Vertex Layout]]
- [[_COMMUNITY_Command Buffers & Draw Loop|Command Buffers & Draw Loop]]
- [[_COMMUNITY_Command Buffer Helpers (vks)|Command Buffer Helpers (vks)]]
- [[_COMMUNITY_DepthFormat Support Queries|Depth/Format Support Queries]]
- [[_COMMUNITY_Command Buffer Allocation (vks)|Command Buffer Allocation (vks)]]
- [[_COMMUNITY_Project README|Project README]]
- [[_COMMUNITY_Libs.cpp Entry|Libs.cpp Entry]]
- [[_COMMUNITY_GLFW Dependency|GLFW Dependency]]
- [[_COMMUNITY_GLM Dependency|GLM Dependency]]
- [[_COMMUNITY_Asset Copy Build Step|Asset Copy Build Step]]

## God Nodes (most connected - your core abstractions)
1. `PrintVkError()` - 19 edges
2. `init()` - 18 edges
3. `initVulkan()` - 13 edges
4. `collidesWith()` - 12 edges
5. `cleanup()` - 10 edges
6. `createSwapChain()` - 9 edges
7. `end()` - 9 edges
8. `recreateSwapChain()` - 8 edges
9. `saveScreenshot()` - 8 edges
10. `initFromAsset()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `ColliderShow.vert main()` --semantically_similar_to--> `AABB Floor Collider (Colliders.hpp / Scene.hpp)`  [INFERRED] [semantically similar]
  skeleton/source/shaders/framework/ColliderShow.vert → notes.md
- `ColliderShow PushConsts (colliderIndex)` --semantically_similar_to--> `Collision Detection Toggle (no-clip cheat)`  [INFERRED] [semantically similar]
  skeleton/source/shaders/framework/ColliderShow.vert → notes.md
- `toChangeBlinnFromPos.frag main()` --semantically_similar_to--> `toChangeSimplePos UniformBufferObject (mvpMat, mMat)`  [AMBIGUOUS] [semantically similar]
  skeleton/source/shaders/toChangeBlinnFromPos.frag → skeleton/source/shaders/toChangeSimplePos.vert
- `CreateColliderRecursive()` --calls--> `initPoint()`  [INFERRED]
  skeleton/source/include/modules/Scene.hpp → skeleton/source/include/modules/Colliders.hpp
- `CreateColliderRecursive()` --calls--> `initSphere()`  [INFERRED]
  skeleton/source/include/modules/Scene.hpp → skeleton/source/include/modules/Colliders.hpp

## Hyperedges (group relationships)
- **Blinn-Phong Shading Pipeline (position-based normals)** — tochangesimplepos_main, tochangesimplepos_uniformbufferobject, tochangeblinnfrompos_main, tochangeblinnfrompos_globaluniformbufferobject [INFERRED 0.85]
- **Collider Debug Visualization Pipeline** — collidershow_main_vert, collidershow_main_frag, collidershow_uniformbufferobject, collidershow_pushconsts [INFERRED 0.85]
- **Cheat/Debug Flags System** — notes_cheatflags_struct, notes_gravity_cheat, notes_jump_cheat, notes_sprint_cheat, notes_collision_cheat [EXTRACTED 1.00]

## Communities

### Community 0 - "Collision Detection (AABB/OOBB/BVH)"
Cohesion: 0.14
Nodes (32): checkIntersectAxis(), Collider, ColliderShow, collidesWith(), collisionAABBAABB(), collisionBVHCollider(), collisionOOBBAABB(), collisionOOBBOOBB() (+24 more)

### Community 1 - "Framework Bootstrap (BaseProject/Starter)"
Cohesion: 0.08
Nodes (8): AssetFile, BaseProject, checkIfItHasExtension(), deviceReport, getRequiredExtensions(), getSixAxis(), handleGamePad(), Model

### Community 2 - "Vulkan Resource Creation"
Cohesion: 0.12
Nodes (27): beginSingleTimeCommands(), copyBufferToImage(), createBuffer(), createImage(), createImageView(), createIndexBuffer(), createResources(), createShaderModule() (+19 more)

### Community 3 - "Debug Collider Rendering & Cheat Flags"
Cohesion: 0.09
Nodes (27): Shader Compilation Pipeline (glslc, GLOB_RECURSE), Vulkan::Vulkan Dependency, ColliderShow.frag main(), ColliderShow.vert main(), ColliderShow PushConsts (colliderIndex), ColliderShow UniformBufferObject (vpMat, strokeColors[20]), AABB Floor Collider (Colliders.hpp / Scene.hpp), CheatFlags Struct (+19 more)

### Community 4 - "Vulkan Device & Swapchain Setup"
Cohesion: 0.12
Nodes (26): checkIfItHasDeviceExtension(), checkValidationLayerSupport(), chooseSwapExtent(), chooseSwapPresentMode(), chooseSwapSurfaceFormat(), createCommandPool(), CreateDebugUtilsMessengerEXT(), createDescriptorPool() (+18 more)

### Community 5 - "App Lifecycle (init/main loop/cleanup)"
Cohesion: 0.09
Nodes (20): main(), Skeleton26ReplaceName, init(), localCleanup(), pipelinesAndDescriptorSetsCleanup(), pipelinesAndDescriptorSetsInit(), populateCommandBuffer(), Scene (+12 more)

### Community 6 - "Text Rendering (TextMaker)"
Cohesion: 0.13
Nodes (18): setCompareOp(), setCullMode(), setTransparency(), submitCommandBuffer(), atlasToUV(), createTextDescriptorSetAndVertexLayout(), createTextDescriptorSets(), createTextMesh() (+10 more)

### Community 7 - "Skeletal Animation"
Cohesion: 0.17
Nodes (9): Animations, Blend(), getAnim(), getSampleTransforms(), init(), Sample(), SkeletalAnimation, getGLTFnodeTransforms() (+1 more)

### Community 8 - "Mesh Loading (glTF/OBJ)"
Cohesion: 0.31
Nodes (10): begin(), checkDeviceExtensionSupport(), end(), initFromAsset(), makeGLTFMesh(), makeOBJMesh(), removeBuffer(), saveScreenshot() (+2 more)

### Community 9 - "Render Pass & Vertex Layout"
Cohesion: 0.29
Nodes (7): create(), createDescriptionAndReference(), createFramebuffers(), createRenderPass(), getAttributeDescriptions(), getBindingDescription(), getView()

### Community 10 - "Command Buffers & Draw Loop"
Cohesion: 0.4
Nodes (6): clearNamedCommandBuffer(), clearNamedCommandBufferForImage(), createCommandBuffer(), drawFrame(), mainLoop(), updateCommandBuffers()

### Community 11 - "Command Buffer Helpers (vks)"
Cohesion: 0.5
Nodes (4): clearCommandBuffers(), vks_initializers_fenceCreateInfo(), vks_initializers_submitInfo(), vulkanDevice_flushCommandBuffer()

### Community 12 - "Depth/Format Support Queries"
Cohesion: 0.67
Nodes (3): findDepthFormat(), findSupportedFormat(), getStandardAttchmentsProperties()

### Community 13 - "Command Buffer Allocation (vks)"
Cohesion: 0.67
Nodes (3): vks_initializers_commandBufferAllocateInfo(), vks_initializers_commandBufferBeginInfo(), vulkanDevice_createCommandBuffer()

### Community 14 - "Project README"
Cohesion: 1.0
Nodes (2): Politecnico di Milano CG Exam 2025-2026, Computer_Graphics Final Project

### Community 15 - "Libs.cpp Entry"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "GLFW Dependency"
Cohesion: 1.0
Nodes (1): GLFW FetchContent Dependency

### Community 17 - "GLM Dependency"
Cohesion: 1.0
Nodes (1): GLM FetchContent Dependency

### Community 18 - "Asset Copy Build Step"
Cohesion: 1.0
Nodes (1): Asset Copy Post-Build Step

## Ambiguous Edges - Review These
- `toChangeSimplePos UniformBufferObject (mvpMat, mMat)` → `toChangeBlinnFromPos.frag main()`  [AMBIGUOUS]
  skeleton/source/shaders/toChangeBlinnFromPos.frag · relation: semantically_similar_to

## Knowledge Gaps
- **26 isolated node(s):** `SkeletalAnimation`, `Animations`, `ColliderShow`, `Collider`, `Scene` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Project README`** (2 nodes): `Politecnico di Milano CG Exam 2025-2026`, `Computer_Graphics Final Project`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Libs.cpp Entry`** (1 nodes): `Libs.cpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `GLFW Dependency`** (1 nodes): `GLFW FetchContent Dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `GLM Dependency`** (1 nodes): `GLM FetchContent Dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Asset Copy Build Step`** (1 nodes): `Asset Copy Post-Build Step`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `toChangeSimplePos UniformBufferObject (mvpMat, mMat)` and `toChangeBlinnFromPos.frag main()`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `init()` connect `App Lifecycle (init/main loop/cleanup)` to `Collision Detection (AABB/OOBB/BVH)`, `Mesh Loading (glTF/OBJ)`?**
  _High betweenness centrality (0.187) - this node is a cross-community bridge._
- **Why does `initFromAsset()` connect `Mesh Loading (glTF/OBJ)` to `Framework Bootstrap (BaseProject/Starter)`, `Vulkan Resource Creation`, `App Lifecycle (init/main loop/cleanup)`, `Skeletal Animation`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `setWorldMatrix()` connect `Collision Detection (AABB/OOBB/BVH)` to `App Lifecycle (init/main loop/cleanup)`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `cleanup()` (e.g. with `pipelinesAndDescriptorSetsCleanup()` and `localCleanup()`) actually correct?**
  _`cleanup()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `SkeletalAnimation`, `Animations`, `ColliderShow` to the rest of the system?**
  _26 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Collision Detection (AABB/OOBB/BVH)` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._