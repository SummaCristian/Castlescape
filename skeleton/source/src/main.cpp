// THIS IS THE FILE YOU MUST START FROM!

// This has been adapted from the Vulkan tutorial
#include <sstream>
#include <limits>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#include <json.hpp>

#include "modules/Starter.hpp"
#include "modules/TextMaker.hpp"
#include "modules/Scene.hpp"
#include "custom/UiQuad.hpp"
#include "custom/CheatHud.hpp"
#include "custom/PauseMenu.hpp"
#include "custom/StartScreen.hpp"
#include "custom/SettingsMenu.hpp"
#include "custom/SceneColliders.hpp"
#include "custom/SceneMaterials.hpp"
#include "custom/SceneLights.hpp"
#include "custom/Flame.hpp"
#include "custom/ExitGlow.hpp"
#include "custom/CubeShadowMap.hpp"
#include "custom/DebugLines.hpp"
#include "custom/HuntCycle.hpp"

// Scene is data-driven: assets/scenes/*.json (scene, colliders, materials,
// lights, flames, gameplay). Shaders in source/shaders/. See OVERVIEW.md.

// The uniform buffer object used in this example
struct UniformBufferObject {
	alignas(16) glm::mat4 mvpMat;
	alignas(16) glm::mat4 mMat;
	// inverse-transpose of mMat: a non-uniform scale would tilt normals off the
	// surface. A mat4 and not a mat3, to dodge std140's column padding.
	alignas(16) glm::mat4 nMat;
	// Cook-Torrance material; mD isn't here, it's the albedo texture. Must match
	// field for field the block the four shaders that see it declare. No explicit
	// padding: the scalars below fill std140's vec4 slots exactly, 240 bytes.
	alignas(16) glm::vec3 mS;	// specular color
	float roughness;			// rho: width of the microfacet distribution
	float F0;					// reflectance seen head-on
	float k;					// diffuse share of the BRDF
	int flatNormals;			// 1: derive the face normal in the shader
	int interiorAmbient;		// 1: use hemispheric ambient for a vertical (interior) surface
	// Seconds since startup; only Flame shaders read it. Per-object, not global,
	// to avoid shifting LightData[]'s offset.
	float time;
	float ambientWeight;		// this model's indirect-light share; negative = inherit gubo.ambientWeight
	float glow;					// 0..1 focus highlight, per-instance, set from gazedInstance
	int metallic;				// 1: no diffuse lobe, indirect term reflects the room instead
};

// Per-frame data shared by every draw (vs. the per-instance UBO above).
// Fixed-size light array + live count, since uniform blocks need a compile-time size.
struct GlobalUniformBufferObject {
	alignas(16) glm::vec3 eyePos;
	int lightCount;
	int debugFlags;			// LIGHT_DEBUG_* bits from LightConstants.glsl
	float time;					// seconds since startup, animates the held torch's flame on the GPU
	float ambientWeight;		// default indirect-light share, 0..1, from lights.json
	float ambientBounce;		// point/spot radiance returned as indirect light; see AmbientLight::bounce
	// IMPORTANT: debugFlags/time/ambientWeight/ambientBounce/fogDensity fill std140's
	// padding before lights[] (offsets 20/24/28/32/36); lights[] itself is alignas(16)-
	// pinned to 48. Adding another scalar here shifts lights[] and every shader offset --
	// and every shader below declares this struct too and MUST be edited to match, since
	// glslc reports no cross-shader layout mismatch of its own (see LightConstants.glsl's
	// header-sharing comment for the same problem one level up).
	float fogDensity;			// exp(-(fogDensity*dist)^2) distance fog, see CookTorrance.frag
	LightData lights[MAX_LIGHTS];
};

// One torch's cube shadow capture data (ShadowCube.vert/frag, PShadowCube, set 1).
// IMPORTANT: UBO, not a push constant: command buffer is recorded once and reused,
// a push constant would freeze the moving held torch.
struct ShadowCubeUniformBufferObject {
	alignas(16) glm::mat4 lightViewProj[6];
	alignas(16) glm::vec4 lightPos;	// xyz used, w is padding
};

// Which of the 6 faces a draw call is for. Safe as a push constant: fixed by
// where the draw sits in the recorded command buffer.
struct ShadowCubeFacePushConstant {
	int32_t face;
};

// Vertex format "VDposNormUV". Starter.hpp fills the normal from the glTF/MGCG
// file automatically once the layout declares one.
struct Vertex {
	glm::vec3 pos;
	glm::vec3 norm;
	glm::vec2 UV;
};

// Shared by all four post passes; each fills only the fields it needs.
struct PostUniformBufferObject {
	glm::vec2 texelSize;	// 1/width, 1/height of the SOURCE texture
	glm::vec2 blurDir;		// (1,0) or (0,1); read by BloomBlur.frag only
	float threshold;		// bright pass: luminance above which anything blooms
	float knee;				// bright pass: how soft the threshold's shoulder is
	float bloomIntensity;	// composite: how much bloom is added back
	float exposure;			// composite
	int debugFlags;			// composite: LIGHT_DEBUG_NO_TONEMAP
	float time;
	// composite: 0->1 white flash on escape, applied after tonemapping.
	// Can't just crank exposure: tonemap approaches white but never fully reaches it.
	float escapeFlash;
	float spectralVeil;		// composite: ramps 0->1 as camera sinks into a ghost, see SPECTRAL_VEIL_OUTER
};

// A full-screen quad vertex; Post.vert derives UV from position alone.
struct PostVertex {
	glm::vec2 pos;
};

// Cheap 1D noise for the torch flicker, computed on CPU so flame, sparks
// and light all share the same value. // Integer hash, not fract(sin(x)*...): that trick gives poor randomness for small x.
static float fireHash(int32_t n) {
	uint32_t h = (uint32_t)n;
	h = (h ^ 61u) ^ (h >> 16);
	h *= 9u;
	h = h ^ (h >> 4);
	h *= 0x27d4eb2du;
	h = h ^ (h >> 15);
	return (float)(h & 0xFFFFFFu) / (float)0xFFFFFF;	// [0, 1]
}

static float fireNoise(float x) {
	float fi = std::floor(x);
	float f = x - fi;
	int32_t i = (int32_t)fi;
	// Smoothstep between the two lattice points, so the result is C1 and the
	// flame's brightness never steps.
	float u = f * f * (3.0f - 2.0f * f);
	return fireHash(i) * (1.0f - u) + fireHash(i + 1) * u;
}

// Three octaves: enough that no single frequency is audible in the result,
// few enough that this stays a rounding error next to the rest of the frame.
static float fireFbm(float x) {
	return fireNoise(x) * 0.55f
		 + fireNoise(x * 2.3f + 17.0f) * 0.30f
		 + fireNoise(x * 4.9f + 53.0f) * 0.15f;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// The whole game lives in one class because BaseProject (modules/Starter.hpp)
// drives it through virtual overrides: localInit / pipelinesAndDescriptorSets*
// / populateCommandBuffer / updateUniformBuffer / GameLogic / localCleanup.
// Anything that could stand on its own was already pulled out into
// include/custom/ (Flame, CubeShadowMap, HuntCycle, SceneLights, ...); what is
// left here is the glue between those, so it cannot be split across files.
//
// To keep it navigable, the body is cut into the sections below, in this
// order, each opened by a ==== banner. State, its tuning constants and its
// helpers stay together inside a section, so one theme reads top to bottom.
//
//   DECLARATIONS AND STATE
//     Vulkan objects: layouts, pipelines, render passes
//     Cube shadow maps: Vulkan objects
//     Cube shadow maps: per-frame submission
//     HDR post-processing chain
//     Scene, text and UI overlays
//     Camera and view state
//     Colliders, materials and lights
//     Cheats and movement tuning
//     Doors
//     Pickups, gaze targeting and the key ring
//     Held items: grip, wall tuck and raise animations
//     Flames, candles and wall torches
//     Visibility culling and candidate priority
//     Fire envelope: flicker, glare and lean
//     Animation time and walk bob
//     Ghosts
//     Hunt cycle and run state
//     The way out: exit box, door, daylight and whiteout
//     Spawn pose, stepping and movement state
//
//   ENGINE OVERRIDES AND LOGIC
//     Window, fullscreen and render-pass rebuild
//     localInit(): scene load and world setup
//     Cube shadow maps: slot assignment and capture
//     Pipelines, descriptor sets and cleanup
//     populateCommandBuffer(): draw order
//     updateUniformBuffer(): per-frame uniforms and HUD
//     Ghost navigation
//     Run lifecycle: restart and hunt phase
//     GameLogic(): input, physics and interaction
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
class Castlescape : public BaseProject {
	protected:

	// =====================================================================
	// Vulkan objects: layouts, pipelines, render passes
	// =====================================================================
	// Here you list all the Vulkan objects you need:

	// Descriptor Layouts [what will be passed to the shaders]
	DescriptorSetLayout DSLlocal, DSLglobal;

	// Vertex formants, Pipelines [Shader couples] and Render passes
	VertexDescriptor VD;
	// The scene pass. Renders into an offscreen floating-point colour
	// attachment (not the swapchain), which is what makes bloom possible.
	RenderPass RP;
	Pipeline P;

	// The ghosts' pipeline, same pass as P right after it. Separate because a
	// ghost needs alpha blending and no BRDF/shadow sampling (Spectral.frag).
	Pipeline Pspectral;

	// Ghosts' depth prepass (SpectralDepth.frag), run right before Pspectral.
	// IMPORTANT: uses VK_COMPARE_OP_LESS (not Pspectral's LESS_OR_EQUAL) so it
	// writes the nearest ghost surface's depth, letting the colour pass reject
	// the ghost's own interior instead of blending it under the body.
	Pipeline PspectralDepth;

	// =====================================================================
	// Cube shadow maps: Vulkan objects
	// =====================================================================
	// Shadow mapping, cube branch (the torches): a real 6-face cube map per
	// point light (CubeShadowMap.hpp: linear-distance storage, one flat bias).
	//
	// IMPORTANT: // // RPShadowCubeCompat: hack to get a valid VkRenderPass for the per-face framebuffers.
	// - createRenderPass() is private -> can't call it directly
	// - so we build a full RenderPass via .init()/.create() and just keep .renderPass
	// - its own attachment/image is never actually used
	// Framebuffers themselves: built by hand in createCubeShadowMaps(), since the
	// engine's framebuffer helper can't target a single face of a cube image.
	RenderPass RPShadowCubeCompat;
	Pipeline PShadowCube;
	CubeShadowMap torchCube[NUM_SHADOW_CUBES];
	// Shared by every torch's cube view: all R32_SFLOAT at one resolution.
	TextureSampler cubeShadowSampler;

	// set 1 for the cube capture pass: one UBO per cube slot (6 face
	// view-projections + world position).
	DescriptorSetLayout DSLshadowCubeCapture;
	DescriptorSet DSshadowCube[NUM_SHADOW_CUBES];

	// set 2 for the main pass's shadow sampling, read by CookTorrance.frag's
	// shadowFactor(). No DescriptorSet member: rides Scene's per-instance
	// machinery, so every instance gets a redundant but cheap copy rather than
	// a third hand-rolled binding path.
	DescriptorSetLayout DSLshadowSample;
	// Six face view-projections per torch cube, index-matched
	// [LightData::shadowIndex][face] (see CUBE_FACE_DIR in CubeShadowMap.hpp).
	glm::mat4 torchFaceMatrices[NUM_SHADOW_CUBES][6];
	// World position of each cube-mapped torch, index-matched to shadowIndex.
	glm::vec3 torchLightPos[NUM_SHADOW_CUBES];
	// How many of the above are populated (lights.json may author fewer shadow
	// point lights than slots).
	int activeCubeShadows = 0;
	// Must match SHADOW_CUBE_RES in LightConstants.glsl: CookTorrance.frag
	// derives the cube path's depth bias from this map's texel size.
	static constexpr int SHADOW_MAP_RES = SHADOW_CUBE_RES;
	// IMPORTANT: far clip and clear value for every torch cube face. Must stay
	// past anything shadowFromCube() can query, or geometry beyond it reads the
	// clear value as its nearest occluder and renders falsely shadowed.
	static constexpr float TORCH_SHADOW_FAR_CONST = 60.0f;
	// Near clip: close enough that only the torch fixture falls inside it.
	static constexpr float TORCH_SHADOW_NEAR_CONST = 0.05f;
	// Slot reserved for the held torch, which never goes through lights.json
	// (its Wm is rewritten live, so it can't get a fixed shadowIndex).
	static constexpr int HAND_TORCH_SHADOW_INDEX = NUM_SHADOW_CUBES - 1;

	// Start of the dynamic pool (between lights.json's fixed slots and the held
	// torch's reserved last one), counted at startup.
	int dynamicShadowSlotBase = 0;
	// Index into torchFlames for whichever flame holds each dynamic slot, -1 if empty.
	std::array<int, NUM_SHADOW_CUBES> dynamicSlotOccupant{};

	// Occupant identity each slot last rendered for; a mismatch queues all six
	// faces, so static point lights render exactly once for the program's life.
	// IMPORTANT: SHADOW_SLOT_UNSET forces one render even for a slot that stays
	// empty forever -- otherwise its image stays VK_IMAGE_LAYOUT_UNDEFINED,
	// which a samplerCube descriptor may not be bound against.
	static constexpr int SHADOW_SLOT_UNSET = -2;
	std::array<int, NUM_SHADOW_CUBES> lastRenderedOccupant;
	// Bitmask of stale FACES per slot (not per slot) since a cube map is six
	// independent images and redrawing all six to fix one is mostly clear cost.
	std::array<uint8_t, NUM_SHADOW_CUBES> pendingFaceMask{};
	static constexpr uint8_t ALL_CUBE_FACES = 0x3F;

	// =====================================================================
	// Cube shadow maps: per-frame submission
	// =====================================================================
	// IMPORTANT: its own pool -- the framework's commandPool has flags = 0, and
	// a buffer from a pool without RESET_COMMAND_BUFFER_BIT may not be re-recorded.
	VkCommandPool shadowCommandPool = VK_NULL_HANDLE;
	// One buffer and fence per swapchain image (re-recording a buffer the GPU
	// still reads is UB). Own fences, not inFlightFences: those are signalled by
	// the main submit, which finishing doesn't prove an earlier shadow submit did.
	std::vector<VkCommandBuffer> shadowCB;
	std::vector<VkFence> shadowCBFence;
	// Held torch's light position/occupied state at its last capture; faces are
	// only redrawn on change, so a stationary player costs no shadow work.
	// Infinity sentinel forces a capture on the first frame.
	glm::vec3 lastHandTorchCapturePos{std::numeric_limits<float>::infinity()};
	bool lastHandTorchCaptureOccupied = false;

	// The occupant diff only catches "did this slot's light change hands" --
	// not enough for ghosts/doors, which move while keeping the same light.
	// Candidates are ghosts and door leaves (materials.json castsShadow), not
	// every moving instance, to avoid re-rendering on every spinning pickup.
	std::vector<Instance *> movingOccluders;
	// Parallel to movingOccluders: true for a door leaf. A door's swing carves
	// a wedge out of a torch's cube map that the per-face sphere test in
	// queueMoverCubeSlotRenders() can miss (the leaf sweeps through faces its
	// resting bounding sphere doesn't touch), so a door forces every face of
	// every torch in reach ONCE, on the frame it stops moving, instead of
	// trusting that test for its final resting state. Forcing all six faces
	// every frame of the swing too (not just the settle frame) is what used
	// to make opening a door near several torches stutter -- the settle-frame
	// force is a one-off per open/close, so it costs nothing repeatable.
	std::vector<bool> movingOccluderIsDoor;
	// Parallel to movingOccluders: whether that mover was moving last frame,
	// so a door's settle frame (moving -> still) can be told apart from every
	// other still frame and get its one-off full-face force.
	std::vector<bool> movingOccluderWasMoving;
	// Wm each occluder had at its last capture, so a stationary mover is one
	// mat4 compare and no draws.
	std::vector<glm::mat4> movingOccluderWm;
	bool moverListBuilt = false;
	// Faces that had a mover at last capture -- needed to clear the faces a
	// mover walked OUT of, since its new position alone doesn't tell us that.
	std::array<uint8_t, NUM_SHADOW_CUBES> slotFaceHadMover{};
	// How far each slot's light still matters, from its own falloff.
	std::array<float, NUM_SHADOW_CUBES> torchShadowReach{};

	// Interval between updateDynamicShadowSlots() calls. Starts equal to the
	// interval so it fires on frame one instead of using uninitialised matrices.
	float shadowReassignTimer = 0.3f;
	static constexpr float SHADOW_REASSIGN_INTERVAL = 0.3f;
	// A waiting candidate must beat the occupant by this factor to take its
	// slot, so standing on the boundary between two candidates doesn't flip
	// (and re-render) on every re-evaluation.
	static constexpr float SHADOW_SWAP_MARGIN = 1.15f;
	// Same margin, for the view-cone cull: an existing occupant near the cone
	// edge shouldn't surrender and re-render every re-evaluation.
	static constexpr float SHADOW_VIEW_KEEP_MARGIN = 1.15f;

	// Models, textures and Descriptors (values assigned to the uniforms)
	DescriptorSet DSglobal;

	// =====================================================================
	// HDR post-processing chain
	// =====================================================================
	// scene (RGBA16F, MSAA + resolve) -> bright pass -> blur H -> blur V (all
	// quarter res) -> composite (swapchain: scene + bloom, then tone map).
	RenderPass RPbright, RPblurH, RPblurV, RPcomposite;
	VertexDescriptor VDpost;
	// One layout for single-texture passes, one for composite (scene + bloom).
	DescriptorSetLayout DSLpost1, DSLpost2;
	Pipeline Pbright, PblurH, PblurV, Pcomposite;
	DescriptorSet DSbright, DSblurH, DSblurV, DScomposite;
	Model *Mpost = nullptr;

	// Attachment descriptions each pass is built from. Members, not locals:
	// RenderPass::init keeps pointers into this vector's storage for the pass's
	// lifetime, and onWindowResize() rebuilds them at the new size.
	std::vector<AttachmentProperties> hdrAtt, brightAtt, blurHAtt, blurVAtt, compositeAtt;

	// Bloom chain runs at 1/BLOOM_DIV per axis, so a cheap 9-tap gaussian on a
	// quarter-res image still covers a wide soft halo in the final picture.
	static constexpr int BLOOM_DIV = 4;

	// 3D scene renders at this fraction of window resolution; Composite.frag
	// upsamples it back. UI stays at real resolution.
	float renderScale = 0.8f;

	// Clamped to >= 1 (a minimised window can't ask for a zero-sized image).
	// Takes windowW as a parameter: onWindowResize() needs the NEW size before
	// swapChainExtent catches up.
	int renderWidth(int windowW) const {
		return std::max(1, (int)std::lround(windowW * renderScale));
	}
	int renderHeight(int windowH) const {
		return std::max(1, (int)std::lround(windowH * renderScale));
	}

	// MSAA level as log2 (0 -> 1x, 2 -> 4x), since the slider's +/-1 step can't
	// express the doubling Vulkan sample counts need.
	float msaaLevel = 2.0f;
	// Upper bound from getMaxUsableSampleCount(): never offer a level the GPU can't do.
	float maxMsaaLevel = 2.0f;
	// The two above are driven by SettingsMenu; its callbacks live with the
	// rest of the UI, in "Scene, text and UI overlays" below.

	// IMPORTANT: threshold set above 1.0 -- at 1.0 a sunlit pale wall haloes
	// too, since it lands right at that luminance once sun+ambient are added.
	static constexpr float BLOOM_THRESHOLD = 1.55f;
	static constexpr float BLOOM_KNEE = 0.45f;
	static constexpr float BLOOM_INTENSITY = 0.65f;
	static constexpr float SCENE_EXPOSURE = 1.0f;

	// =====================================================================
	// Scene, text and UI overlays
	// =====================================================================
	// To support loading assets from a scene.json file
	Scene SC;
	std::vector<VertexDescriptorRef>  VDRs;
	std::vector<TechniqueRef> PRs;

	// to provide textual feedback
	TextMaker txt;

	// Flat-colored quads: background/highlight panel behind the cheat HUD's text.
	UiQuad uiQuad;

	// Center-screen dot for aiming look-based interactions (GameLogic()'s gaze test).
	UiQuad crosshair;

	// (Re)builds the crosshair dot centered on the window. Called at init and on resize.
	void setCrosshairQuad() {
		const float dotSize = 4.0f;
		float cx = (float)windowWidth / 2.0f;
		float cy = (float)windowHeight / 2.0f;
		crosshair.setQuads({
			UiRect{cx - dotSize / 2.0f, cy - dotSize / 2.0f, dotSize, dotSize,
				   glm::vec4(1.0f, 1.0f, 1.0f, 0.5f)}
		});
	}

	// Toggle-based pause menu for the cheats below, opened/closed with L.
	CheatHud hud;

	UiQuad pauseQuad;
	// Dims the screen and freezes GameLogic() while open (overlayOpen() gating). ESC toggles it.
	PauseMenu pauseMenu;

	UiQuad startScreenQuad;
	// Launch screen: open from the first frame so the app boots here instead
	// of into the castle. Reopened when Quit abandons a run.
	StartScreen startScreen;

	UiQuad settingsQuad;
	// Render Scale / MSAA sliders, reachable from StartScreen or PauseMenu.
	// settingsFromPause remembers which to reopen on Back.
	SettingsMenu settingsMenu;
	bool settingsFromPause = false;

	// The settings sliders' callbacks. The values they drive (renderScale,
	// msaaLevel) belong to the HDR chain and are declared up there.

	// Replays the resize rebuild path at the current window size. Skipped while
	// a rebuild is already pending, to avoid a render pass targeting a size
	// newer than its framebuffer.
	void applyRenderScaleChange() {
		if(!framebufferResized) {
			onWindowResize((int)windowWidth, (int)windowHeight);
			RebuildPipeline();
		}
	}

	// v is unused; always formats the current renderScale/renderWidth/renderHeight.
	std::string formatRenderScale(float /*v*/) {
		char buf[32];
		snprintf(buf, sizeof(buf), "< %d%% (%dx%d) >",
				 (int)std::lround(renderScale * 100.0f),
				 renderWidth((int)windowWidth), renderHeight((int)windowHeight));
		return std::string(buf);
	}

	// Unlike renderScale, a sample-count change has to rebuild hdrAtt itself.
	void applyMsaaChange() {
		if(!framebufferResized) {
			msaaSamples = static_cast<VkSampleCountFlagBits>(1 << (int)std::lround(msaaLevel));
			initRenderPasses();
			RebuildPipeline();
		}
	}

	std::string formatMsaaLevel(float level) {
		char buf[16];
		snprintf(buf, sizeof(buf), "< %dx >", 1 << (int)std::lround(level));
		return std::string(buf);
	}

	// =====================================================================
	// Camera and view state
	// =====================================================================
	// Other application parameters
	float Ar;	// Aspect ratio

	glm::mat4 ViewPrj;
	glm::mat4 View;

	// Free-look camera state, persisted across frames. Spawns inside the
	// dungeon hall, facing +X down the hall toward the far door.
	glm::vec3 camPos = glm::vec3(-20.2f, 1.8f, 18.0f);
	float camYaw = 0.0f;			// degrees around world up; 0 faces +X
	float camPitch = -10.0f;		// degrees, -90 down to +90 up
	float camVerticalVelocity = 0.0f;	// gravity speed, units/s, reset by ground clamp

	// Debug spectator orbit (cheats.debugCam): snaps behind the player when
	// enabled, then nudged by IJKL (orbit) and U/O (dolly).
	float dbgOrbitYaw = 0.0f;
	float dbgOrbitPitch = DEBUG_CAM_PITCH0;
	float dbgOrbitDist = DEBUG_CAM_DIST0;
	bool dbgCamWasOn = false;

	// Top of the "floor" instance; last-resort clamp so no-clip can't fall through the map.
	float worldFloorY = 0.0f;

	// =====================================================================
	// Colliders, materials and lights
	// =====================================================================
	// Hand-authored collision geometry from colliders.json, merged with what
	// scene.json built.
	SceneColliders colliderSet;

	// Per-model BRDF parameters from materials.json. See SceneMaterials.hpp.
	SceneMaterials materials;

	// The scene's light sources from lights.json. See SceneLights.hpp.
	SceneLights sceneLights;

	// Flat list of every collider gameplay collides against, from colliderSet after load.
	std::vector<Collider *> allColliders;

	// Uniform XZ grid over allColliders, built once, so ghost queries test only
	// nearby colliders instead of scanning the whole castle.
	struct ColliderGrid {
		// Bigger than a ghost's radius so the 3x3 neighbourhood query always
		// covers a ghost-sized radius test.
		static constexpr float CELL_SIZE = 3.0f;

		struct Entry {
			AABBextents E;
		};
		std::vector<Entry> entries;
		std::unordered_map<int64_t, std::vector<int>> cells;

		static int cellCoord(float v) { return (int)std::floor(v / CELL_SIZE); }
		static int64_t cellKey(int cx, int cz) {
			return (int64_t(uint32_t(cx)) << 32) | uint32_t(cz);
		}

		void build(const std::vector<Collider *> &colliders) {
			entries.clear();
			cells.clear();
			entries.reserve(colliders.size());
			for(Collider *c : colliders) {
				int idx = (int)entries.size();
				entries.push_back({c->getExtents()});
				const AABBextents &E = entries.back().E;
				int cx0 = cellCoord(E.xMin), cx1 = cellCoord(E.xMax);
				int cz0 = cellCoord(E.zMin), cz1 = cellCoord(E.zMax);
				for(int cx = cx0; cx <= cx1; cx++) {
					for(int cz = cz0; cz <= cz1; cz++) {
						cells[cellKey(cx, cz)].push_back(idx);
					}
				}
			}
		}

		// Visits cached extents in the 3x3 cell neighbourhood around `p`
		// (a collider one cell over can still be within query radius). May
		// visit a wide collider more than once; cheaper than a dedup pass.
		template<typename F>
		void forEachNear(const glm::vec3 &p, F &&fn) const {
			int cx = cellCoord(p.x), cz = cellCoord(p.z);
			for(int dx = -1; dx <= 1; dx++) {
				for(int dz = -1; dz <= 1; dz++) {
					auto it = cells.find(cellKey(cx + dx, cz + dz));
					if(it == cells.end()) continue;
					for(int idx : it->second) {
						fn(entries[idx].E);
					}
				}
			}
		}
	};
	ColliderGrid ghostColliderGrid;

	// Colliders left out of the grid because their Wm changes every frame
	// (door leaves/lock hardware) -- a cached AABB would keep blocking a ghost
	// after the door swung open, so these are tested live instead.
	std::vector<Collider *> ghostDynamicColliders;

	// Every ghost collider query goes through here: static grid, then movers.
	template<typename F>
	void ghostForEachNearbyCollider(const glm::vec3 &p, F &&fn) const {
		ghostColliderGrid.forEachNear(p, fn);
		for(Collider *c : ghostDynamicColliders) {
			fn(c->getExtents());
		}
	}

	// =====================================================================
	// Cheats and movement tuning
	// =====================================================================
	// Debug/cheat toggles, isolated in a utility struct.
	// Not persisted across runs, reset to default values on launch.
	struct CheatFlags {
		bool collisionEnabled = true;   // false = no-clip
		bool showCoordinates = false;   // live camera position/yaw readout, for placing scene.json objects
		bool ghostsCanCatch = true;     // off: hunt still plays out but can't end the run
		bool jumpEnabled = false;       // off by default: player can't jump

		// Flame switches, read by flameBurning(). Not in SceneLights: flame
		// lights are appended into gubo directly, never via lights.json.
		bool roomTorchesEnabled = true;
		bool handTorchEnabled = true;   // off also hides the held torch model

		// Lighting debug views, resolved into gubo.debugFlags for CookTorrance.frag.
		bool unlit = false;          // albedo only, no lighting
		bool showNormals = false;    // shading normal as color
		bool focusGlowEnabled = true;   // aura on the gazed target ([E] prompt stays either way)
		bool specularEnabled = true; // off forces k to 1
		bool toneMapEnabled = true;  // off clips overexposure to white
		bool shadowsEnabled = true;  // off forces shadowFactor() to 1 (shading vs geometry diagnostic)

		// Off means that flame category never competes for a cube-shadow slot
		// at all, unlike shadowsEnabled which just blanks an already-cast one.
		bool torchShadowsEnabled = true;
		bool candleShadowsEnabled = true;

		// Off by default: the held torch has no arm/body to anchor a shadow of
		// its own mesh, so it would look like the torch floating mid-air.
		bool handTorchModelCastsShadowWhenHeld = false;

		// Geometry overlays (DebugLines.hpp).
		bool showLightGizmos = false;
		bool showShadowFrustums = false;
		bool showColliders = false;     // wireframe box per collider/ramp quad
		// Third-person spectator view; visibility culls still run from the real
		// first-person eye, so the cull boundary becomes visible from outside.
		bool debugCam = false;
		bool showLightHeatmap = false;  // recolor surfaces by incoming light intensity
		// False-color cube-shadow diagnostic (LIGHT_DEBUG_SHADOW_GAP): green
		// means the map says a fragment is unoccluded yet it read as shadowed
		// upstream (stale/wrong cube data), blue means a tap genuinely found
		// an occluder there. See LightConstants.glsl for the full legend.
		bool showShadowGap = false;
	} cheats;

	// Numeric tuning for the movement cheats -- "how strong", not "on/off", so
	// a separate struct. Also not persisted.
	struct MovementParams {
		// World units traveled per second
		float moveSpeed = 3.0f;
		// Multiplier applied to moveSpeed while sprinting
		float sprintMultiplier = 2.0f;
		float jumpSpeed = 5.0f;				// initial upward velocity, world units/second
		float gravity = -9.81f;				// world units/second^2
	} movement;

	// =====================================================================
	// Doors
	// =====================================================================
	// A door leaf (its own instance) that swings around a vertical hinge on E.
	// IMPORTANT: SM_Door_01's local origin is at the hinge edge, so the
	// instance's authored transform already is the closed-door hinge frame;
	// "open" is just one extra rotation about world Y appended to it.
	struct Door {
		std::string instanceId;
		Instance *inst = nullptr;
		glm::mat4 baseWm{1.0f};	// authored (closed) world matrix
		glm::vec3 promptPos{0.0f};	// interact-range point: the doorway centre, not the hinge
		// Same point in the leaf's local frame; leafPos() tracks where the
		// panel has swung to, so a wide-open leaf can still be aimed at to close it.
		glm::vec3 promptOffset{0.0f};
		glm::vec3 leafPos() const {
			return inst == nullptr ? promptPos
								   : glm::vec3(inst->Wm * glm::vec4(promptOffset, 1.0f));
		}
		// Open travel magnitude; direction is decided per-opening by swingSignAwayFrom().
		float openAngleDeg = 100.0f;
		bool open = false;
		float angle = 0.0f;	// current animated angle, eases toward the target
		// Swing direction sign, only recomputed while fully closed (flipping
		// mid-swing would sweep the leaf through its own frame).
		float swingSign = 1.0f;
		// Empty lockKeyId = no lock. Matched by id, not "any key", so a
		// two-key level can't be opened in the wrong order.
		std::string lockKeyId;
		std::string lockLabel;	// human-readable name for the locked prompt, defaults to lockKeyId
		// Per-door locked-prompt wording; empty uses the generic padlock lines.
		std::string promptReady;	// carrying what it wants
		std::string promptMissing;	// not carrying it
		std::string promptBlocked;	// standing on the far side of the lock
		// While set and still locked without the key, this door isn't even a
		// gaze target (doorIsHidden): no aura, no prompt. Used for the secret
		// bookcase, where a red aura would give the trick away.
		bool secret = false;
		// True while the padlock holds; cleared for good once the key is
		// spent -- keys are one-shot, an unlocked door must never re-lock.
		bool locked = false;
		// Visible padlock hardware, in the leaf's local frame (world = leaf Wm * local).
		struct LockProp {
			Instance *inst;
			glm::mat4 local;	// identity, or a half turn for a flipped door (addLockProp's `flip`)
			// Inverted lifecycle: hardware that appears once the lock is OFF.
			// Used by the secret bookcase (a returned book fills the shelf gap).
			bool whenUnlocked = false;
		};
		std::vector<LockProp> lockProps;
		// Extra collider stand-off on the hardware face, keeping the held torch out of the chains.
		static constexpr float LOCK_PROP_KEEPOUT = 0.45f;
		// Which face the hardware is on: +1 as make_door_lock.py exports, -1 for
		// the flipped copy. Reachable only from its own face.
		// one that won't move, key or no key.
		float lockFaceSign = 1.0f;
		// True when `p` stands on the padlock's face. Against baseWm, not the
		// live Wm, since a locked door never swings. XZ only.
		bool onLockSide(const glm::vec3 &p) const {
			glm::vec3 axis(baseWm[0].x, 0.0f, baseWm[0].z);	// leaf local +X, in world
			if(glm::length(axis) < 1e-6f) return true;	// degenerate: don't lock anyone out
			glm::vec3 d(p.x - promptPos.x, 0.0f, p.z - promptPos.z);
			return glm::dot(d, glm::normalize(axis)) * lockFaceSign > 0.0f;
		}
		// Sign on |openAngleDeg| that swings the leaf away from a player at
		// `p`, so the door always opens outward. Rotates a panel point the
		// positive way and checks which side it lands on, rather than reasoning
		// about a cross product through an arbitrary transform. XZ only.
		float swingSignAwayFrom(const glm::vec3 &p) const {
			float authored = openAngleDeg < 0.0f ? -1.0f : 1.0f;
			glm::vec3 n(baseWm[0].x, 0.0f, baseWm[0].z);	// leaf local +X, in world
			if(glm::length(n) < 1e-6f) return authored;	// degenerate
			n = glm::normalize(n);

			glm::vec3 d(p.x - promptPos.x, 0.0f, p.z - promptPos.z);
			float playerSide = glm::dot(d, n);
			// Standing in the doorway itself: no "away" to pick.
			if(std::abs(playerSide) < 1e-4f) return authored;

			glm::vec3 hinge(baseWm[3]);
			glm::vec4 tip = baseWm
						  * glm::rotate(glm::mat4(1.0f), glm::radians(std::abs(openAngleDeg)), glm::vec3(0.0f, 1.0f, 0.0f))
						  * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f);	// a panel point, swung the positive way
			float tipSide = glm::dot(glm::vec3(tip) - hinge, n);
			if(std::abs(tipSide) < 1e-6f) return authored;	// swings flat
			return tipSide * playerSide > 0.0f ? -1.0f : 1.0f;
		}
	};
	std::vector<Door> doors;

	// How close (XZ, to the doorway centre) before a door's prompt appears.
	static constexpr float DOOR_INTERACT_RADIUS = 6.0f;
	static constexpr float DOOR_OPEN_SPEED = 120.0f;	// degrees/second
	// Gaze-candidate range; DOOR_INTERACT_RADIUS still gates actual interaction.
	static constexpr float DOOR_LOOK_DISTANCE = 9.0f;
	static constexpr float DOOR_AIM_RADIUS = 1.2f;		// doorway half-width as aiming tolerance

	// Edge-detection for E, so holding it doesn't toggle every frame.
	bool interactKeyWasPressed = false;
	// Index into `doors` of the one in range, or -1; read by updateUniformBuffer()
	// for the "[E] Interact" prompt.
	int nearbyDoor = -1;

	// =====================================================================
	// Pickups, gaze targeting and the key ring
	// =====================================================================
	// A world object collected with [E]. Not strictly one-way: can be dropped again (G).
	struct Pickup {
		std::string instanceId;
		Instance *inst = nullptr;
		glm::vec3 worldPos{0.0f};	// measured at load, for the in-range check
		bool collected = false;
		// Authored pose, so restartRun() can reset a picked-up/dropped item
		// (worldPos gets overwritten on drop).
		glm::mat4 spawnWm{1.0f};
		glm::vec3 spawnPos{0.0f};
		// Non-empty => this pickup is a key, opening every Door with a matching lockKeyId.
		std::string keyId;
		// Spent on a lock and gone for the run, distinct from `collected`
		// (a collected key can still be dropped; a consumed one can't come back).
		bool consumed = false;
		float worldScale = 1.0f;		// uniform world scale, read from the authored matrix at load
		// Euler angles (deg) orienting the item in hand, per-pickup since grip
		// depends on mesh axes (key stands on its long axis, book tilts flat-up).
		glm::vec3 handTiltDeg{0.0f};
		// Item position relative to the eye, camera space; per-pickup since it
		// positions the mesh origin, which differs per model.
		glm::vec3 handOffset{0.0f};
	};
	std::vector<Pickup> pickups;
	// Key ring: indices into `pickups`, in collection order (a vector, not an
	// id set, since two keys can share an id). keyRing.back() is drawn in the hand.
	std::vector<int> keyRing;
	static constexpr float PICKUP_INTERACT_RADIUS = 4.0f;	// 3D check, unlike doors' XZ one
	// Index into `pickups` of the one in range, or -1; checked before nearbyDoor
	// so grabbing wins over interacting with what's behind it.
	int nearbyPickup = -1;
	static constexpr float PICKUP_LOOK_DISTANCE = 6.0f;	// gaze-candidate range
	static constexpr float PICKUP_AIM_RADIUS = 0.35f;		// tight, they're small props

	// Base half-angle of the aiming cone, before an object's aim radius widens it. Shared by doors/pickups.
	static constexpr float GAZE_CONE_DEG = 9.0f;

	// Instance the crosshair rests on within interact range, or nullptr.
	// Resolved once per frame in GameLogic(), read by updateUniformBuffer() for ubo.glow.
	Instance *gazedInstance = nullptr;
	// True when gazedInstance is aimed at but [E] would do nothing (locked
	// door, wrong side, etc). Flips ubo.glow's sign, swapping the aura gold -> red.
	bool gazedInteractionDisabled = false;
	// Category of gazedInstance, so CookTorrance.frag can pick a distinct aura
	// color per kind. Encoded as ubo.glow's magnitude alongside the
	// gazedInteractionDisabled sign; see the assignment in updateUniformBuffer().
	enum class GlowKind { Door = 1, Pickup = 2, Candle = 3, WallTorch = 4 };
	GlowKind gazedGlowKind = GlowKind::Door;

	// True if `front` is aimed closely enough at `target`: within lookDist and
	// a cone widened by the angular size aimRadius subtends at that distance.
	bool isGazedAt(const glm::vec3 &front, const glm::vec3 &target,
				   float lookDist, float aimRadius, float &cosAngleOut) const {
		glm::vec3 to = target - camPos;
		float dist = glm::length(to);
		if(dist < 1e-4f || dist > lookDist) return false;
		float cosAngle = glm::dot(front, to / dist);
		float angularTolerance = std::atan(aimRadius / dist);
		float minCos = std::cos(glm::radians(GAZE_CONE_DEG) + angularTolerance);
		if(cosAngle < minCos) return false;
		cosAngleOut = cosAngle;
		return true;
	}

	// True for a secret door the player has no business seeing yet.
	bool doorIsHidden(const Door &d) const {
		return d.secret && d.locked && findKeyInRing(d.lockKeyId) < 0;
	}

	// Index into `doors` aimed at within look range, or -1. Does not check
	// DOOR_INTERACT_RADIUS; the caller does that separately.
	int findGazedDoor(const glm::vec3 &front) const {
		int best = -1;
		float bestCos = -1.0f;
		for(int i = 0; i < (int)doors.size(); i++) {
			if(doorIsHidden(doors[i])) continue;
			// Two points: the fixed doorway, and the leaf in its live pose. An
			// open door swings out of its own doorway, so aiming at the panel
			// to shut it used to select nothing. The one aimed at more squarely
			// wins.
			float cosAngle = -1.0f;
			bool hit = isGazedAt(front, doors[i].promptPos, DOOR_LOOK_DISTANCE,
								 DOOR_AIM_RADIUS, cosAngle);
			float leafCos = -1.0f;
			if(isGazedAt(front, doors[i].leafPos(), DOOR_LOOK_DISTANCE,
						 DOOR_AIM_RADIUS, leafCos) && (!hit || leafCos > cosAngle)) {
				cosAngle = leafCos;
				hit = true;
			}
			if(hit && (best < 0 || cosAngle > bestCos)) {
				best = i;
				bestCos = cosAngle;
			}
		}
		return best;
	}
	// How far the player is from a door for the interact-range gate: the
	// NEARER of its doorway and its leaf, matching the two points
	// findGazedDoor aims at. XZ only, as this check has always been.
	float doorDistance(const Door &d, const glm::vec3 &p) const {
		auto flat = [&](const glm::vec3 &q) {
			float dx = p.x - q.x, dz = p.z - q.z;
			return std::sqrt(dx * dx + dz * dz);
		};
		return std::min(flat(d.promptPos), flat(d.leafPos()));
	}

	// Same as findGazedDoor, for pickups. Skips already-collected ones, same
	// as the old proximity scan did.
	int findGazedPickup(const glm::vec3 &front) const {
		int best = -1;
		float bestCos = -1.0f;
		for(int i = 0; i < (int)pickups.size(); i++) {
			if(pickups[i].collected) continue;
			float cosAngle;
			if(isGazedAt(front, pickups[i].worldPos, PICKUP_LOOK_DISTANCE,
						 PICKUP_AIM_RADIUS, cosAngle)) {
				if(best < 0 || cosAngle > bestCos) {
					best = i;
					bestCos = cosAngle;
				}
			}
		}
		return best;
	}

	// Edge-detection for the drop key, same reason as interactKeyWasPressed.
	bool dropKeyWasPressed = false;

	// The key drawn in the hand: the most recently collected one, or -1.
	// Index into `pickups` -- picking up doesn't spawn a second model, it just
	// draws the same instance off the camera instead of on the table.
	int heldKeyIdx() const {
		return keyRing.empty() ? -1 : keyRing.back();
	}
	// Slot in keyRing of a carried key matching `id`, or -1. Empty `id` = "any
	// key". Back to front, so the key in hand is spent first.
	int findKeyInRing(const std::string &id) const {
		for(int slot = (int)keyRing.size() - 1; slot >= 0; slot--) {
			if(id.empty() || pickups[keyRing[slot]].keyId == id) return slot;
		}
		return -1;
	}
	// Spend a carried key permanently for this run. `slot` indexes keyRing, not
	// pickups. Parking below the map is how a mesh is hidden (no visibility flag exists).
	void consumeKey(int slot) {
		if(slot < 0 || slot >= (int)keyRing.size()) return;
		int idx = keyRing[slot];
		Pickup &p = pickups[idx];
		p.consumed = true;
		p.collected = true;
		keyRing.erase(keyRing.begin() + slot);
		// Not parked yet: it sinks out of frame first, parked once the
		// animation ends in GameLogic(). Only one key can animate at a time.
		if(keyLowerIdx >= 0) {
			pickups[keyLowerIdx].inst->Wm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
		}
		keyLowerIdx = idx;
		keyLowerElapsed = 0.0f;
	}
	// Puts a carried key back in the world, lying flat in front of the player.
	// `fwdDist`: arm's length for manual drop (G), closer for the automatic one.
	void dropKeyFromRing(int slot, const glm::vec3 &camPos, const glm::vec3 &front, float fwdDist) {
		if(slot < 0 || slot >= (int)keyRing.size()) return;
		Pickup &p = pickups[keyRing[slot]];
		const float EYE_HEIGHT = 1.8f;	// same eye height used throughout GameLogic()

		glm::vec2 faceDir(front.x, front.z);
		if(glm::length(faceDir) > 0.0001f) faceDir = glm::normalize(faceDir);
		else faceDir = glm::vec2(0.0f, 1.0f);
		// 0.03 above the feet: exactly at floor height would z-fight the floor mesh.
		glm::vec3 dropPos(camPos.x + faceDir.x * fwdDist,
						   camPos.y - EYE_HEIGHT + 0.03f,
						   camPos.z + faceDir.y * fwdDist);
		float yaw = std::atan2(faceDir.x, faceDir.y);

		p.worldPos = dropPos;
		p.collected = false;
		keyRing.erase(keyRing.begin() + slot);
		p.inst->Wm = glm::translate(glm::mat4(1.0f), dropPos)
					* glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f))
					* glm::scale(glm::mat4(1.0f), glm::vec3(p.worldScale));
	}
	// =====================================================================
	// Held items: grip, wall tuck and raise animations
	// =====================================================================
	// Default held pose; negative X = left hand (torch owns the right). Each
	// pickup can override via Pickup::handOffset/handTiltDeg.
	static constexpr glm::vec3 HAND_KEY_OFFSET = glm::vec3(-0.40f, -0.4f, -0.9f);
	// X = 90 swings the key's long axis (local Z) onto world Y, tip up.
	static constexpr glm::vec3 HAND_KEY_TILT_DEG = glm::vec3(90.0f, -20.0f, 0.0f);
	// Shared by rise and sink animations so the two can't drift apart.
	static glm::mat4 handGrip(const glm::vec3 &tiltDeg, float bobRollDeg) {
		return glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.x), glm::vec3(1.0f, 0.0f, 0.0f))
			 * glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.y), glm::vec3(0.0f, 1.0f, 0.0f))
			 * glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.z + bobRollDeg), glm::vec3(0.0f, 0.0f, 1.0f));
	}

	// ---- Held-item wall tuck ----
	// IMPORTANT: held items sit ~1 unit out from the camera, past the 0.3
	// PLAYER_RADIUS clearance, so at a wall the item pokes through it. For the
	// torch this is a lighting bug, not just a clipping one: its point light
	// and cube shadow ride the flame at the torch head, so a head through a
	// wall darkens the near face and lights the room beyond through solid
	// geometry. Fixed by retracting the item geometrically (probe how far it
	// can reach right now, place it exactly there) rather than blending to a
	// fixed "tucked" pose, which would be wrong at any distance but the one it
	// was tuned for.
	//
	// Everything below is a REACH FRACTION: 1 extended, HAND_TUCK_MIN_REACH against the chest.

	// How far the item's body reaches past the grip, camera-space (rough; HAND_TUCK_SKIN dwarfs the error).
	static constexpr float HAND_TUCK_TORCH_PAD = 0.30f;
	static constexpr float HAND_TUCK_KEY_PAD = 0.15f;

	// Collider growth for the probe only, covering the coarse collision shell
	// and giving the tuck's drop/rotation a safety margin. Keep small, or it folds the item away even in open rooms.
	static constexpr float HAND_TUCK_SKIN = 0.30f;

	// Reach never retracts past this: a light at the eye flattens shading and
	// blows out what's closest, so a few cm of torch in a wall reads better.
	// Also the probe's start point (everything nearer is assumed clear).
	static constexpr float HAND_TUCK_MIN_REACH = 0.35f;

	// Samples across the probed stretch; ten over ~1 unit gives ~0.1 granularity.
	static constexpr int HAND_TUCK_SAMPLES = 10;

	// Tuck in fast, out slow -- one constant can't avoid both strobing the
	// torchlight and letting the item dip into the wall.
	static constexpr float HAND_TUCK_TAU_IN = 0.05f;
	static constexpr float HAND_TUCK_TAU_OUT = 0.18f;

	// Cancels the upward drift retraction otherwise causes (shortening Y along
	// with reach), so the tuck doesn't read as "raised to face".
	static constexpr float HAND_TUCK_DROP = 0.28f;
	// Extra grip rotation sold as part of the tuck gesture; rotating about the
	// grip barely moves the light. Negate a component if an item rotates the wrong way.
	static constexpr glm::vec3 HAND_TUCK_TILT_DEG = glm::vec3(40.0f, 30.0f, 0.0f);

	// Live reach fraction per hand, advanced by advanceReach() each frame
	// (torch and key retract independently).
	float handTorchReach = 1.0f;
	float handKeyReach = 1.0f;

	// Like ghostPointBlocked but with a stand-off; kept separate since that one
	// answers a sightline question where inflating the world would be wrong.
	bool handPointBlocked(const glm::vec3 &p, float skin) const {
		for(Collider *C : allColliders) {
			AABBextents E = C->getExtents();
			if(p.x < E.xMin - skin || p.x > E.xMax + skin) continue;
			if(p.z < E.zMin - skin || p.z > E.zMax + skin) continue;
			if(p.y < E.yMin - skin || p.y > E.yMax + skin) continue;
			return true;
		}
		return false;
	}

	// How far the item at `camOffset` may reach this frame, as a fraction of
	// its authored offset. Marches outward along the eye->item line and stops
	// at the first solid sample.
	float handFreeReach(const glm::mat4 &camWm, const glm::vec3 &camOffset, float pad) const {
		float reach = glm::length(camOffset);
		if(reach < 1e-4f) {
			return 1.0f;
		}
		// The grip pushed out to the leading edge, camera space. Fractions
		// below are of this, so 1 leaves the edge where it was authored.
		glm::vec3 tip = camOffset * ((reach + pad) / reach);

		const float span = 1.0f - HAND_TUCK_MIN_REACH;
		for(int i = 1; i <= HAND_TUCK_SAMPLES; i++) {
			float t = HAND_TUCK_MIN_REACH + span * (float)i / (float)HAND_TUCK_SAMPLES;
			if(handPointBlocked(glm::vec3(camWm * glm::vec4(tip * t, 1.0f)), HAND_TUCK_SKIN)) {
				return HAND_TUCK_MIN_REACH + span * (float)(i - 1) / (float)HAND_TUCK_SAMPLES;
			}
		}
		return 1.0f;
	}

	// One asymmetric exponential smoothing step; tucking in means reach
	// decreasing, so the comparison is inverted against the tau names.
	static void advanceReach(float &state, float target, float deltaT) {
		float tau = (target < state) ? HAND_TUCK_TAU_IN : HAND_TUCK_TAU_OUT;
		state += (target - state) * (1.0f - std::exp(-deltaT / tau));
	}

	// Builds the tucked pose in place, so callers can still compose the walk
	// bob on top. "Inward" is read off the offset's sign (torch +X, key -X).
	static void applyTuck(float reachFrac, glm::vec3 &offset, glm::vec3 &tiltDeg) {
		if(reachFrac >= 1.0f) {
			return;
		}
		float inward = (offset.x >= 0.0f) ? -1.0f : 1.0f;
		// 0..1 into the tuck, normalised so retuning HAND_TUCK_MIN_REACH
		// doesn't rescale the drop and rotation.
		float tuck = (1.0f - reachFrac) / (1.0f - HAND_TUCK_MIN_REACH);

		offset *= reachFrac;
		offset.y -= HAND_TUCK_DROP * tuck;
		tiltDeg.x += HAND_TUCK_TILT_DEG.x * tuck;
		tiltDeg.y += inward * HAND_TUCK_TILT_DEG.y * tuck;
	}
	// ---- Key raise/sink animation ----
	// Pick-up animation: the key rises into frame from below over
	// KEY_RAISE_DURATION. A translation on camera-local Y added to
	// HAND_KEY_OFFSET, so it composes with the walk bob.
	static constexpr float KEY_RAISE_DURATION = 0.35f;
	// How far below the final pose the key starts, camera-local -- roughly out
	// the bottom of the frame, so it reads as "raised into view".
	static constexpr float KEY_RAISE_DROP = 0.8f;
	// Seconds since the key in hand was picked up, saturating at
	// KEY_RAISE_DURATION. Reset on every pickup (including a swap).
	float keyRaiseElapsed = KEY_RAISE_DURATION;
	// Index into `pickups` of a spent key still sinking out of frame, or -1.
	// Already off the ring, so only this block draws it; parks it below the
	// map when the fall finishes.
	int keyLowerIdx = -1;
	float keyLowerElapsed = 0.0f;

	// ---- The held torch ----
	// The torch held in the right hand. A scene instance whose Wm is rebuilt
	// every frame from the camera basis, so it follows the view like a
	// viewmodel. Null if the instance isn't found.
	Instance *handTorchInst = nullptr;
	// Index into `torchFlames` of the held torch's flame, or -1. Authored
	// UNLIT; lit from a burning wall torch with [E]. hasBurningTorch() reads
	// its `burning`.
	int handFlameIdx = -1;
	// True once the torch has been picked up. Until then it sits at its floor
	// pose as a pickup target, no flame or hand model is drawn, and
	// hasBurningTorch() is false. restartRun() clears it.
	bool handTorchCollected = false;
	// The torch's authored floor pose, captured in localInit before the
	// held-torch code can overwrite Wm. For the gaze check on the ground and
	// to put it back.
	glm::mat4 handTorchSpawnWm{1.0f};
	glm::vec3 handTorchWorldPos{0.0f};
	// True when the crosshair is on the floor torch within reach this
	// frame. Mirrors nearbyPickup; set every frame in GameLogic().
	bool nearbyHandTorch = false;
	// Torch position relative to the eye, camera space. Low and close, so the
	// handle crops off the bottom and only the torch shows, like a viewmodel.
	static constexpr glm::vec3 HAND_TORCH_OFFSET = glm::vec3(0.5f, -0.35f, -1.1f);
	static constexpr glm::vec3 HAND_TORCH_TILT_DEG = glm::vec3(-15.0f, 20.0f, 0.0f);
	static constexpr float HAND_TORCH_SCALE = 0.35f;	// mesh is sized for a wall mount
	// Pick-up animation, mirror of the key's: torch rises into frame from below.
	static constexpr float TORCH_RAISE_DURATION = 0.35f;
	static constexpr float TORCH_RAISE_DROP = 0.8f;
	// Seconds since pickup, or >= TORCH_RAISE_DURATION once done. Starts
	// saturated so an already-collected torch doesn't replay it.
	float torchRaiseElapsed = TORCH_RAISE_DURATION;

	// =====================================================================
	// Flames, candles and wall torches
	// =====================================================================
	// The flame at a torch's head (Flame.hpp); one instance drives every torch, held one included.
	Flame flame;

	// Daylight outside the exit door (ExitGlow.hpp): one overbright quad, its
	// glare done entirely by the bloom chain.
	ExitGlow exitGlow;

	// Line renderer for cheat-gated debug overlays; see DebugLines.hpp.
	DebugLines debugLines;

	// One entry per torch with a flame (localInit's addTorchFlame). `anchor` is
	// in the torch model's local space; inst->Wm * vec4(anchor,1) is world position.
	// The rest is live fire state, simulated on the CPU since billboards,
	// sparks and the cast point light all need to agree on one signal.
	struct TorchFlame {
		Instance *inst;
		int flameId;
		glm::vec3 anchor;

		// True only for the held torch: it's anchored in camera space and
		// tilts with the view, so its flame must ride the camera's actual
		// up/right rather than the ordinary cylindrical (world-up) billboard.
		bool heldByCamera = false;

		float phase = 0.0f;		// noise-field phase offset, so torches don't gutter in sync

		// Brightness envelope (~0.30 gutter to ~1.40 flare), chased by a
		// critically-damped spring so it glides instead of jumping per noise sample.
		float intensity = 1.0f;
		float intensityVel = 0.0f;

		// Height envelope (~0.78..1.09): same signal as intensity but
		// low-passed harder, since the fuel column follows light output late.
		float heightScale = 1.0f;

		float glare = 0.0f;		// smoothed stare-at factor, 0..1, overdrives HDR output

		// Low-passed anchor velocity; the flame leans against its own motion (air drag).
		glm::vec3 prevPos = glm::vec3(0.0f);
		glm::vec3 smoothedVel = glm::vec3(0.0f);
		bool velPrimed = false;

		// Fire colour driving both the flame gradient and the cast light.
		// Overwritten every frame by the hunt cycle (dragged toward violet).
		glm::vec3 color = TORCH_LIGHT_COLOR;
		// Colour with nothing hunting, captured at spawn -- `color` can't be
		// its own base since mixing toward violet and storing back converges on violet.
		glm::vec3 baseColor = TORCH_LIGHT_COLOR;

		// Multiplies FLAME_HEIGHT/HALF_WIDTH on top of instance scale; candles pass smaller.
		float sizeScale = 1.0f;
		// Multiplies point light colour only, independent of sizeScale, so
		// candles can cast their (small but not necessarily dim) light.
		float lightScale = 1.0f;

		// True for a candle, false for a torch (held included); from
		// flames.json's "isCandle". Only used by the HUD's Torch/Candle Shadows toggles.
		bool isCandle = false;

		// Read through flameBurning(): off takes light, billboard, sparks,
		// shadow candidacy and glare all at once, nothing spawned/destroyed at runtime.
		bool burning = true;
		// Authored value of `burning`, so restartRun() re-darkens candles the player lit.
		bool spawnBurning = true;

		// Catching-fire envelope: 0 unlit, ~1.05 flare, 1 lit. Scales billboard
		// and light colour together so it doesn't pop to size before lighting up.
		float ignitionScale = 1.0f;
		float ignitionVel = 0.0f;

		// World-space anchor computed at spawn; only meaningful for a static
		// flame (held torch recomputes it every frame). Aim target for findGazedCandle().
		glm::vec3 anchorWorld = glm::vec3(0.0f);

		// True for every flame competing for a dynamic shadow-cube slot
		// (everything except the held torch, which has its own fixed slot).
		bool shadowCandidate = false;
		int shadowSlot = -1;		// absolute cube-shadow slot if held, else -1

		// Six face view-projections, computed once at registration since this
		// flame is static like the lights.json torches.
		std::array<glm::mat4, 6> shadowFaceMatrices{};

		glm::vec2 lean = glm::vec2(0.0f);	// resolved into billboard axes, uploaded to the shader
	};
	std::vector<TorchFlame> torchFlames;

	// A candle needs no list of its own: it IS a TorchFlame authored unlit, so
	// `nearbyCandle` indexes torchFlames. 3D radii like a pickup's, tighter
	// since lighting one means holding a torch close.
	static constexpr float CANDLE_INTERACT_RADIUS = 3.5f;
	static constexpr float CANDLE_LOOK_DISTANCE = 6.5f;
	static constexpr float CANDLE_AIM_RADIUS = 0.55f;	// the wick is small, but don't demand pixel-perfect aim
	int nearbyCandle = -1;		// index into `torchFlames` of the unlit candle in range, or -1

	// Index into `torchFlames` of the unlit candle aimed at within look range,
	// or -1. Lit candles are skipped, so they neither glow nor steal aim.
	int findGazedCandle(const glm::vec3 &front) const {
		int best = -1;
		float bestCos = -1.0f;
		for(int i = 0; i < (int)torchFlames.size(); i++) {
			const TorchFlame &tf = torchFlames[i];
			if(!tf.isCandle || tf.burning || tf.heldByCamera) continue;
			float cosAngle;
			if(isGazedAt(front, tf.anchorWorld, CANDLE_LOOK_DISTANCE,
						 CANDLE_AIM_RADIUS, cosAngle)) {
				if(best < 0 || cosAngle > bestCos) {
					best = i;
					bestCos = cosAngle;
				}
			}
		}
		return best;
	}

	// Wall-torch lighting mirrors the candle interaction: aim at a burning wall
	// torch with the (unlit) held torch and press [E] to light it.
	// INTERACT_RADIUS is measured horizontally (XZ) in GameLogic(), since wall
	// torches mount well above eye level and a 3D check would spend most of the
	// budget on the vertical gap, forcing the player to stand unnaturally close.
	static constexpr float WALL_TORCH_INTERACT_RADIUS = 4.0f;
	static constexpr float WALL_TORCH_LOOK_DISTANCE = 8.0f;
	static constexpr float WALL_TORCH_AIM_RADIUS = 1.0f;
	int nearbyWallTorch = -1;	// index into `torchFlames` of the burning wall torch in range, or -1

	// Non-negative only while the held torch is unlit and in hand.
	int findGazedWallTorch(const glm::vec3 &front) const {
		if(handFlameIdx < 0 || !handTorchCollected ||
		   torchFlames[handFlameIdx].burning) return -1;
		int best = -1;
		float bestCos = -1.0f;
		for(int i = 0; i < (int)torchFlames.size(); i++) {
			const TorchFlame &tf = torchFlames[i];
			if(tf.isCandle || tf.heldByCamera || !flameBurning(tf)) continue;
			float cosAngle;
			if(isGazedAt(front, tf.anchorWorld, WALL_TORCH_LOOK_DISTANCE,
						 WALL_TORCH_AIM_RADIUS, cosAngle)) {
				if(best < 0 || cosAngle > bestCos) {
					best = i;
					bestCos = cosAngle;
				}
			}
		}
		return best;
	}

	// The floor torch, before pickup: mirrors findGazedPickup for the one
	// prop that doesn't ride the Pickup/keyRing machinery.
	bool findGazedHandTorch(const glm::vec3 &front) const {
		if(handTorchInst == nullptr || handTorchCollected) return false;
		float cosAngle;
		return isGazedAt(front, handTorchWorldPos, PICKUP_LOOK_DISTANCE,
						 PICKUP_AIM_RADIUS, cosAngle);
	}

	// Is this flame burning? The one question the two flame cheats are asked
	// through, so every consequence of a flame switches off together. Candles
	// use their own runtime `burning` (no cheat claims them -- lighting one is
	// gameplay).
	bool flameBurning(const TorchFlame &tf) const {
		if(tf.heldByCamera) return cheats.handTorchEnabled && tf.burning;
		if(tf.isCandle)     return tf.burning;
		return cheats.roomTorchesEnabled;
	}

	// Does the player have fire in hand? False until the torch is picked up
	// (handTorchCollected) AND lit from a wall torch, and false with "Holding
	// Torch" off. The one place that decides "there is fire in hand".
	bool hasBurningTorch() const {
		return handTorchInst != nullptr && handTorchCollected &&
			   cheats.handTorchEnabled &&
			   handFlameIdx >= 0 && torchFlames[handFlameIdx].burning;
	}

	// Global glare, 0..1: the max over every wall torch's stare-at factor,
	// smoothed asymmetrically. Feeds the post chain's exposure/bloom; a member
	// so the smoothing survives between frames.
	float glareSmoothed = 0.0f;

	// One anchor for every torch (wall and held meshes share layout). Y sits
	// below the rim since the cup has depth and the flame's field fades in
	// gradually. Also flames.json's fallback default.
	static constexpr glm::vec3 TORCH_FLAME_ANCHOR = glm::vec3(-0.384f, 0.30f, 0.0f);

	// Flame size in the torch model's local units, so it scales with each
	// instance (full on wall torches, shrunk on the held one). HALF_WIDTH is
	// wider than the visible flame since the noise field eats into its own silhouette.
	static constexpr float FLAME_HEIGHT = 0.95f;
	static constexpr float FLAME_HALF_WIDTH = 0.20f;

	// Torch flame's point light: one colour/falloff for every torch, tighter than gate lanterns.
	static constexpr glm::vec3 TORCH_LIGHT_COLOR = glm::vec3(1.0f, 0.5f, 0.16f);
	static constexpr float TORCH_LIGHT_G = 2.1f;
	static constexpr float TORCH_LIGHT_BETA = 1.4f;
	// Candles also shrink g (reach) on top of flames.json's lightScale (peak
	// brightness) so a dim candle doesn't still carry into the next room.
	static constexpr float CANDLE_LIGHT_G_SCALE = 0.35f;
	// Fraction of peak below which a moving occluder's shadow isn't worth
	// re-capturing (shadowRelevantReach()). Lower if a shadow stops following its ghost.
	static constexpr float SHADOW_REACH_CUTOFF = 0.02f;
	// Fraction of peak below which lightReachesViewCone() treats a light as
	// lighting nothing and drops it (and its shadow slot). Higher than
	// SHADOW_REACH_CUTOFF: a lagging shadow reads worse than a faint light
	// winking out. Raise to cull harder; watch the boundary via cheats.debugCam.
	static constexpr float LIGHT_CULL_REACH_CUTOFF = 0.04f;

	// How far a torch light is still uploaded, and how many may be live at
	// once -- each costs a full GGX eval per sample, the dominant GPU cost.
	static constexpr float TORCH_LIGHT_CULL_DIST = 55.0f;
	static constexpr int TORCH_LIGHT_MAX_LIVE = 32;

	// =====================================================================
	// Visibility culling and candidate priority
	// =====================================================================
	// Geometry visibility cull: radius around the player plus a longer view
	// cone. Geometry only, never the light list.
	static constexpr float GEOM_CULL_RADIUS = 12.0f;  // always drawn this close, any facing
	static constexpr float GEOM_CULL_CONE_DIST = 50.0f;  // dungeon's longest sightline is ~60
	// IMPORTANT: keeps a drawn torch bracket always a lit one -- if the light
	// cull were shorter than the geometry cone a visible torch model could render unlit.
	static_assert(TORCH_LIGHT_CULL_DIST >= GEOM_CULL_CONE_DIST,
				  "a torch light cull shorter than the geometry cone would "
				  "leave a visible torch model unlit");
	static constexpr float GEOM_CULL_CONE_COS = 0.34f;	// ~70 degree half-angle as a cosine
	// IMPORTANT: instances are tested by centre, and this pack's meshes aren't
	// centre-pivoted, so a long wall could measure outside the cone with half
	// still on screen (the real cause of edge-of-screen popping). Fixed by
	// treating every instance as a bounding sphere: collider extents where
	// available, else this flat fallback radius.
	static constexpr float GEOM_CULL_FALLBACK_RADIUS = 4.0f;

	// ---- Third-person debug camera ----
	// Third-person debug camera (cheats.debugCam) starting spherical coords
	// (I/K pitch, J/; yaw, U/O dolly at runtime). Only ViewPrj changes; every
	// cull still runs from the real first-person eye, so culled instances
	// visibly wink out when watched from here.
	static constexpr float DEBUG_CAM_DIST0 = 17.0f;   // metres from the player
	static constexpr float DEBUG_CAM_PITCH0 = 20.0f;  // degrees above the player
	static constexpr float DEBUG_CAM_ORBIT_SPEED = 90.0f;  // deg/s, IJKL
	static constexpr float DEBUG_CAM_DOLLY_SPEED = 14.0f;  // m/s, U/O
	// Clear space the debug camera's near plane leaves after clipping the room shell.
	static constexpr float DEBUG_CAM_CLIP_MARGIN = 4.0f;
	// Instances above this height (ceilings, at y=6.2) are skipped so the
	// spectator sees an open-top dollhouse instead of a roof.
	static constexpr float DEBUG_CAM_ROOF_CUT = 5.0f;

	// True if the instance at worldPos (bounding radius objRadius) is close or
	// aimed-at enough to draw. eyePos/forward are computed once per frame in
	// updateUniformBuffer() and passed through.
	static bool geometryVisible(const glm::vec3 &worldPos, float objRadius,
								const glm::vec3 &eyePos, const glm::vec3 &forward) {
		glm::vec3 d = worldPos - eyePos;
		float dist = glm::length(d);
		// Subtract the object's own half-size from the judged distance, so a
		// big/near object counts as "at" the camera before its centre is.
		float effDist = std::max(0.0f, dist - objRadius);
		if(effDist <= GEOM_CULL_RADIUS) {
			return true;
		}
		if(effDist > GEOM_CULL_CONE_DIST) {
			return false;
		}
		if(dist <= 1e-4f) {
			return true;	// camera at the object's centre
		}
		float facing = glm::dot(d / dist, forward);
		// Angular size ~objRadius/dist, so a near/large object needs less
		// head-on facing to count as in the cone; clamped for tiny/far ones.
		float angularSlack = glm::clamp(objRadius / dist, 0.0f, 1.0f) * 0.5f;
		return facing >= (GEOM_CULL_CONE_COS - angularSlack);
	}

	// True if a point light's illumination sphere overlaps the geometry-visible
	// region (the GEOM_CULL_RADIUS ball, or the view cone). Used to drop torch
	// lights/shadow cubes that contribute nothing on screen.
	// IMPORTANT: deliberately conservative -- inflates the cone by the full
	// `reach` rather than testing exactly, so it only ever culls a light that
	// truly reaches nothing drawn. Perpendicular/axial distances are both
	// 1-Lipschitz in world position, so evaluating the cone's flare at the
	// farthest axial point the sphere can touch (axial + reach) bounds every
	// lit point inside it -- one sqrt, no acos/tan.
	static bool lightReachesViewCone(const glm::vec3 &lightPos, float reach,
									 const glm::vec3 &eyePos, const glm::vec3 &forward) {
		glm::vec3 d = lightPos - eyePos;

		// Overlaps the always-drawn bubble (any facing).
		float nearR = GEOM_CULL_RADIUS + reach;
		if(glm::dot(d, d) <= nearR * nearR) {
			return true;
		}

		float axial = glm::dot(d, forward);
		// Sphere entirely past the cone's far cap.
		if(axial - reach > GEOM_CULL_CONE_DIST) {
			return false;
		}
		float perp = glm::length(d - axial * forward);

		// Widest the cone gets over the axial span the sphere reaches into.
		float tHi = glm::clamp(axial + reach, 0.0f, GEOM_CULL_CONE_DIST);
		float tanHalf = std::sqrt(std::max(0.0f, 1.0f - GEOM_CULL_CONE_COS * GEOM_CULL_CONE_COS))
						/ GEOM_CULL_CONE_COS;
		return perp <= tHi * tanHalf + reach;
	}

	// Skews candidate priority (live-light cut, shadow pool) toward what's
	// ahead of the player. 0 = pure nearest-first; at 0.6 a torch 8 units
	// behind loses to one 10 units in view.
	static constexpr float SHADOW_FACING_BIAS_WEIGHT = 0.6f;

	// Squared eye->pos distance, inflated for anything behind `forward` but
	// still monotonic in real distance for a fixed alignment.
	static float facingBiasedDistSq(const glm::vec3 &pos, const glm::vec3 &eyePos,
									 const glm::vec3 &forward) {
		glm::vec3 d = pos - eyePos;
		float distSq = glm::dot(d, d);
		if(distSq < 1e-6f) {
			return distSq;
		}
		float alignment = glm::dot(d, forward) / std::sqrt(distSq);	// cos(angle)
		return distSq * (1.0f + SHADOW_FACING_BIAS_WEIGHT * (1.0f - alignment));
	}

	// =====================================================================
	// Fire envelope: flicker, glare and lean
	// =====================================================================
	// Fire envelope. The fast flicker band lives in Flame.frag as a per-pixel
	// shimmer; the CPU keeps a slower 7 Hz term at reduced weight so the cast light dances too.
	static constexpr float FLAME_FLICKER_HZ = 7.0f;
	// Critically-damped spring rate for the brightness envelope; settles in ~0.15s.
	static constexpr float FLAME_BRIGHT_OMEGA = 14.0f;
	// One-pole rate for the height envelope: a flame shortens over ~1/3s, doesn't teleport.
	static constexpr float FLAME_HEIGHT_TAU = 0.35f;
	// Guttering: brief irregular collapses, driven by slow noise crossing a high threshold.
	static constexpr float FLAME_GUTTER_SPEED = 0.85f;
	static constexpr float FLAME_GUTTER_LO = 0.66f;	// noise below this: no gutter
	static constexpr float FLAME_GUTTER_HI = 0.82f;	// above this: full gutter
	static constexpr float FLAME_GUTTER_DEPTH = 0.48f;	// how far it ducks

	// Catching fire: underdamped spring (ZETA < 1) chasing 1/0, so it flares
	// slightly past resting size on the way up, reading as "catching" rather than "fading in".
	static constexpr float FLAME_IGNITION_OMEGA = 9.0f;
	static constexpr float FLAME_IGNITION_ZETA = 0.55f;

	// Stare-at glare: centring a wall torch in view swells exposure/bloom and
	// that torch's HDR output, masking the billboard's flat-card look exactly when it would show.
	static constexpr float GLARE_COS_MIN = 0.90f;	// facing cosine: below, no glare
	static constexpr float GLARE_COS_MAX = 0.985f;	// above, full glare
	static constexpr float GLARE_DIST_NEAR = 1.0f;	// fades in past this...
	static constexpr float GLARE_DIST_FAR = 9.0f;	// ...and is gone by here
	// Asymmetric smoothing: dazzle arrives fast, eye recovers slower (also
	// prevents pumping when the view strafes across the facing threshold).
	static constexpr float GLARE_TAU_RISE = 0.15f;
	static constexpr float GLARE_TAU_FALL = 0.40f;
	static constexpr float GLARE_EXPOSURE_GAIN = 0.30f;	// exposure *= 1 + gain*glare
	static constexpr float GLARE_BLOOM_GAIN = 0.90f;	// bloomIntensity likewise
	static constexpr float GLARE_FLAME_GAIN = 0.50f;	// that flame's own HDR boost

	// Lean: TAU is how fast smoothed velocity chases the real one; PER_SPEED
	// scales tip offset with hand speed; MAX stops sprinting folding the flame onto its side.
	static constexpr float TORCH_LEAN_TAU = 0.14f;
	static constexpr float TORCH_LEAN_PER_SPEED = 0.055f;
	static constexpr float TORCH_LEAN_MAX = 0.30f;

	// =====================================================================
	// Animation time and walk bob
	// =====================================================================
	// Seconds since startup, uploaded as gubo.time; free-running so sway/flicker never visibly repeats.
	float animTime = 0.0f;

	// Walking sway: lateral swing once per stride, vertical bounce at twice
	// that, from one accumulating phase. walkBobBlend eases 0..1 amplitude with walk state.
	float walkBobPhase = 0.0f;
	float walkBobBlend = 0.0f;
	static constexpr float WALK_BOB_SPEED = 7.0f;			// rad/s at normal walking speed
	static constexpr float WALK_BOB_VERTICAL = 0.035f;		// amplitude before walkBobBlend scales it
	static constexpr float WALK_BOB_LATERAL = 0.02f;
	static constexpr float WALK_BOB_BLEND_TAU = 0.15f;
	// Vertical head-bob applied to the view itself, kept well under the hand's
	// bob so the world only just nods.
	static constexpr float CAM_BOB_VERTICAL = 0.012f;

	// =====================================================================
	// Ghosts
	// =====================================================================
	// Three-state machine driven by huntCycle.hunting(). Return walks the chase's
	// breadcrumb trail backwards, which is why there is no pathfinding here.
	enum class GhostMode {
		Patrol,
		Chase,
		Return
	};

	struct Ghost {
		std::string instanceId;
		Instance *inst = nullptr;
		std::vector<glm::vec3> waypoints;	// XZ used for the path; Y is the resting hover height
		int targetIdx = 1;					// waypoints[] index currently being approached
		float distAlongSegment = 0.0f;		// world units already covered on the current leg
		float bobPhase = 0.0f;
		float speed = 1.5f;					// patrol pace, from gameplay.json
		float chaseSpeed = 3.4f;			// hunting pace, from gameplay.json

		GhostMode mode = GhostMode::Patrol;
		// Live position without the bob (folding it in would pump the collision slab).
		glm::vec3 pos{0.0f};
		float yaw = 0.0f;			// eased not snapped: spinning on the spot reads as a bug
		float turnBias = 1.0f;		// which way it went round an obstacle last frame, kept until clear

		std::vector<glm::vec3> trail;	// breadcrumbs, oldest first; trail[0] is where the chase began
		int resumeIdx = 1;				// patrol progress saved when chase started
		float resumeDist = 0.0f;

		// Where the ghost last saw the player; chase steers toward this, not the live position.
		glm::vec3 lastKnownPlayerPos{0.0f};
		bool hasLastKnown = false;

		// Stuck detection for Chase: pinned-against-a-door and idling-at-a-stale-target
		// look the same; GHOST_GIVEUP_TIME turns either into giving up.
		glm::vec3 stuckCheckPos{0.0f};
		float stuckTimer = 0.0f;

		// 0..1 chase progress, eased not switched (`mode` flips in one frame,
		// and Chase is entered/left repeatedly per hunt). Read by Spectral.frag
		// as ubo.F0 to shift the apparition toward hunt red/brighter.
		float chaseBlend = 0.0f;
	};
	std::vector<Ghost> ghosts;

	// chaseBlend's ease time constants: lighting up faster than calming down,
	// so the telegraph lands while it still matters.
	static constexpr float GHOST_CHASE_FADE_IN_TAU = 0.25f;
	static constexpr float GHOST_CHASE_FADE_OUT_TAU = 1.1f;

	static constexpr float GHOST_BOB_SPEED = 1.6f;
	static constexpr float GHOST_BOB_AMPLITUDE = 0.3f;
	// Faster than patrol/chase: the return is dead time, no reason to dawdle back.
	static constexpr float GHOST_RETURN_SPEED = 4.5f;
	static constexpr float GHOST_TURN_SPEED = 9.0f;	// yaw easing, rad/s

	// Vertical cylinder, not a box: box corners project further on a diagonal
	// and snag on jambs. Radius shrunk below the real mesh fit (0.45 vs 0.65)
	// to stop the clear/blocked test flipping every frame and the ghost vibrating.
	static constexpr float ghostXZFitShrink = 0.45f;
	static constexpr float ghostSteerMargin = 1.0f;
	float ghostRadius = 0.5f;
	float ghostBodyBottom = -1.80f;
	float ghostBodyTop = 0.83f;
	static constexpr float GHOST_PROBE_DIST = 1.2f;	// wall-probe lookahead
	// Horizontal catch distance plus vertical slack: ghosts hover at 2.2, player eyes at 1.8.
	static constexpr float GHOST_CATCH_RADIUS = 0.85f;
	static constexpr float GHOST_CATCH_VERTICAL = 2.5f;

	// The spectral veil, the screen half of the ghost fade (SpectralFade.glsl).
	// IMPORTANT: must match SPECTRAL_INSIDE_* in that shader file -- a gap
	// between the ramps is a moment with no ghost and no wash.
	static constexpr float SPECTRAL_VEIL_OUTER = 2.00f;
	static constexpr float SPECTRAL_VEIL_INNER = 1.05f;
	static constexpr float SPECTRAL_VEIL_FADE_Y = 0.60f;	// margin around model Y bounds, avoids bob flicker
	// How far up the veil's ramp a non-hunting ghost snuffs the held torch.
	static constexpr float SPECTRAL_VEIL_SNUFF_AT = 0.5f;
	// Prune radius must exceed trail spacing, or consecutive crumbs prune each other.
	static constexpr float GHOST_TRAIL_SPACING = 0.75f;
	static constexpr float GHOST_TRAIL_PRUNE_RADIUS = 1.4f;

	// Padded above idle drift so easing a yaw doesn't reset the stuck clock.
	static constexpr float GHOST_STUCK_EPS = 0.08f;
	static constexpr float GHOST_GIVEUP_TIME = 3.0f;

	// =====================================================================
	// Hunt cycle and run state
	// =====================================================================
	// The hunt cycle: the clock that decides when the torches change colour and
	// the ghosts come for the player. Owns no scene state of its own, see
	// custom/HuntCycle.hpp.
	HuntCycle huntCycle;

	// How the current run ended, or Running if it hasn't. Freezes GameLogic()
	// the same way an open cheat HUD does, and R starts a fresh one.
	enum class RunState {
		Running,
		Caught,
		Escaped
	};
	RunState runState = RunState::Running;
	bool restartKeyWasPressed = false;

	// Edge-detection for the ESC key, which now opens/closes the pause menu
	// instead of closing the window outright (see updateUniformBuffer()).
	bool escKeyWasPressed = false;

	// True whenever any modal overlay is open. GameLogic() freezes
	// camera/movement/physics/hunt clock behind this.
	bool overlayOpen() const {
		return hud.isOpen() || pauseMenu.isOpen() || startScreen.isOpen() || settingsMenu.isOpen();
	}

	// =====================================================================
	// The way out: exit box, door, daylight and whiteout
	// =====================================================================
	// Where the exit is, from gameplay.json's "exit.box". World-space and
	// axis-aligned: the player wins by standing inside it. The exit door
	// itself gates access with its own padlock (see exitDoorIndex below),
	// so this box has no separate key check.
	glm::vec3 exitBoxMin{0.0f};
	glm::vec3 exitBoxMax{0.0f};
	bool exitHasBox = false;

	// ---- The way out, as a door ----
	// Index into `doors` of the exit leaf (hbDoorE), or -1. Daylight is tied
	// to how far this door has swung, not the exit box, since the light
	// outside has to arrive while the leaf is still moving.
	int exitDoorIndex = -1;

	// Where the daylight stands: two quads because the door swings OUTWARD, so
	// nothing can be parked right behind the opening without the leaf sweeping through it.
	// IMPORTANT: half-extents (4.4/4.6) and positions are derived from the
	// worst-case sightline from the dv room (reaching back to x 12.8) and the
	// door leaf's swing geometry (hinge x 19.283, 2.50 diagonal) so the arch's
	// projected silhouette always falls inside the quad's flat middle, not its
	// border fade (which starts at 78% of the half-extent in ExitGlow.frag).
	// Changing the door/hub geometry requires re-deriving these.
	// Hub shrunk from 4x4 to 3x3 (tools/build_scene.py) shifted every X here by -7.2.
	static constexpr glm::vec3 EXIT_GLOW_CENTER = glm::vec3(23.6f, 2.8f, 10.79f);
	static constexpr float EXIT_GLOW_HALF_WIDTH = 4.4f;		// along world Z
	static constexpr float EXIT_GLOW_HALF_HEIGHT = 4.6f;	// along world Y
	static constexpr glm::vec3 EXIT_GLOW_NORMAL = glm::vec3(-1.0f, 0.0f, 0.0f);	// faces west, into the castle
	// Ground quad: bridges under the wall (hides its border fade in stone) out
	// to behind the upright quad (avoids a seam). 6cm up: clears z-fighting,
	// stays below the door's swept bottom edge (y 0.2).
	static constexpr glm::vec3 EXIT_GLOW_FLOOR_CENTER = glm::vec3(22.7f, 0.06f, 10.79f);
	static constexpr float EXIT_GLOW_FLOOR_HALF_X = 1.6f;
	static constexpr float EXIT_GLOW_FLOOR_HALF_Z = 3.6f;
	static constexpr glm::vec3 EXIT_GLOW_FLOOR_NORMAL = glm::vec3(0.0f, 1.0f, 0.0f);
	// Third quad, same trick upside down: the upright wall of light is finite,
	// so looking up at the threshold would otherwise see over it into the
	// skybox (no EXIT_GLOW_HALF_HEIGHT fixes this -- the sightline through the
	// arch top diverges as the player nears the wall). y = 4.9 sits just above
	// the doorway hole (0..4.85) so every upward ray through the arch hits a
	// white quad before escaping. Same X extent as the floor quad, same reasoning.
	static constexpr glm::vec3 EXIT_GLOW_CEILING_CENTER = glm::vec3(22.7f, 4.90f, 10.79f);
	static constexpr float EXIT_GLOW_CEILING_HALF_X = 1.6f;
	static constexpr float EXIT_GLOW_CEILING_HALF_Z = 3.6f;
	static constexpr glm::vec3 EXIT_GLOW_CEILING_NORMAL = glm::vec3(0.0f, -1.0f, 0.0f);
	// Quad ids, in write order. Named, not 0/1/2, because they have different
	// bases and half-extents.
	static constexpr int EXIT_GLOW_UPRIGHT = 0;
	static constexpr int EXIT_GLOW_FLOOR = 1;
	static constexpr int EXIT_GLOW_CEILING = 2;
	static constexpr int EXIT_GLOW_COUNT = 3;
	// Peak radiance. Absurd (BLOOM_THRESHOLD is 1.55) and has to be: the tone
	// map divides by (Y + 1), so 9 lands at 0.90 on screen and 60 at 0.984 --
	// "cannot look at it" costs an order of magnitude. The bloom chain then
	// floods the stonework with the unclamped value.
	static constexpr float EXIT_GLOW_INTENSITY = 60.0f;
	// Daylight, warmed slightly. Pure white reads as a hole, not sky.
	static constexpr glm::vec3 EXIT_GLOW_COLOR = glm::vec3(1.0f, 0.97f, 0.90f);
	// The light the doorway throws BACK into the room, as a spot appended
	// straight into gubo (the same thing the torch loop does with its flames),
	// not as a lights.json entry: its brightness is a function of the door's
	// angle, and lights.json has no way to say that. A spot rather than a
	// point because the light has to come through the opening -- a point light
	// out there would wrap round and light the outside face of the east wall
	// as brightly as the floor inside.
	static constexpr glm::vec3 EXIT_SPILL_POS = glm::vec3(29.5f, 2.6f, 10.79f);
	// Well over 1: this is a doorway onto open daylight standing in a room lit
	// by torches, and a spill light that merely matched them would leave the
	// stone around the opening looking like it was lit by another torch. The
	// scene target is HDR, so overbright light colours are as legitimate here
	// as they are on the flames -- and the bloom chain treats what this lights
	// up the same way it treats the quad itself.
	static constexpr glm::vec3 EXIT_SPILL_COLOR = glm::vec3(3.4f, 3.26f, 3.0f);
	static constexpr float EXIT_SPILL_G = 9.0f;		// reaches across the dv room
	static constexpr float EXIT_SPILL_BETA = 1.0f;	// inverse-linear, so it carries
	// Narrow, because this light has no shadow map: a wide cone would light
	// the east wall's inner face right across the room, reading as the wall
	// gone transparent. Aimed down the doorway axis, the leak stays where the
	// light would honestly be.
	static constexpr float EXIT_SPILL_INNER_DEG = 55.0f;
	static constexpr float EXIT_SPILL_OUTER_DEG = 105.0f;

	// 0 while the exit door is shut, 1 once it's finished swinging. Drives the
	// glow and spill light. Smoothstepped so they don't snap on when the
	// padlock comes off and the leaf twitches.
	float exitOpenFrac = 0.0f;

	// The whiteout. Once the run is won this ramps 0 -> 1 and multiplies the
	// post chain's exposure and bloom, so the last thing seen is the frame
	// blowing out, not a text banner. Rides the same two knobs the stare-at
	// glare uses -- the only two post values that mean "brighter".
	float escapeFlash = 0.0f;
	static constexpr float ESCAPE_FLASH_SECONDS = 1.6f;
	static constexpr float ESCAPE_EXPOSURE_GAIN = 7.0f;
	static constexpr float ESCAPE_BLOOM_GAIN = 3.0f;

	// =====================================================================
	// Spawn pose, stepping and movement state
	// =====================================================================
	// The authored starting pose, captured in localInit() before anything
	// moves it, so restartRun() has one source of truth.
	glm::vec3 spawnPos{0.0f};
	float spawnYaw = 0.0f;
	float spawnPitch = 0.0f;

	// Tallest surface the player walks straight onto, from the feet. One shared
	// constant: the two collision passes must agree or they contradict each
	// other -- the wall pass skips anything at/below this (a step, not a wall),
	// the ground pass accepts the same as standable. Two copies once turned
	// every low collider into an unclimbable wall.
	static constexpr float MAX_STEP_HEIGHT = 0.5f;

	// Vertical view smoothing: the ground clamp moves the camera up instantly
	// on a step-up, this offset absorbs the jump and decays back so the VIEW
	// eases while the physics position stays exact. Sloped ground is already
	// smooth; this is for crates, ledges and landings.
	float eyeStepOffset = 0.0f;
	// Past MAX_EYE_STEP_OFFSET the view trails so far it reads as sinking.
	static constexpr float EYE_SMOOTH_TAU = 0.06f;
	static constexpr float MAX_EYE_STEP_OFFSET = 0.6f;

	// Edge-detection for the jump key, so holding it doesn't re-trigger.
	bool jumpKeyWasPressed = false;
	// Feet resting on a collider, refreshed by the floor check. Starts true so
	// a jump is available immediately.
	bool grounded = true;
	// Stored rather than read from Ctrl directly: a sprint can only START while
	// grounded, but can be stopped mid-air.
	bool sprinting = false;

	// =====================================================================
	// Window, fullscreen and render-pass rebuild
	// =====================================================================

	// Here you set the main application parameters
	void setWindowParameters() {
		// window size, title and initial background
		windowWidth = 800;
		windowHeight = 600;
		windowTitle = "Castlescape";
		windowResizable = GLFW_TRUE;

		// Initial aspect ratio
		Ar = 4.0f / 3.0f;
	}


	// What to do when the window changes size
	void onWindowResize(int w, int h) {
		std::cout << "Window resized to: " << w << " x " << h << "\n";
		Ar = (float)w / (float)h;
		// Composite follows the window; the scene follows it scaled by
		// renderScale; the bloom targets follow that, divided by BLOOM_DIV.
		RP.width = renderWidth(w);
		RP.height = renderHeight(h);
		RPcomposite.width = w;
		RPcomposite.height = h;
		// After RP.width/height: bloomWidth()/Height() read those.
		RPbright.width = RPblurH.width = RPblurV.width = bloomWidth();
		RPbright.height = RPblurH.height = RPblurV.height = bloomHeight();

		windowWidth = (uint32_t)w;
		windowHeight = (uint32_t)h;

		txt.resizeScreen(w, h);
		uiQuad.resizeScreen(w, h);
		crosshair.resizeScreen(w, h);
		pauseQuad.resizeScreen(w, h);
		startScreenQuad.resizeScreen(w, h);
		settingsQuad.resizeScreen(w, h);
		setCrosshairQuad();
		// IMPORTANT: the collider visualizer owns a swapchain-attached render
		// pass too; without this it rebuilds its framebuffers at the old size
		// on resize (VUID-...-04533).
		SC.ColShow.resizeScreen(w, h);
	}

	// ---- F11 fullscreen toggle ----
	// Starter.hpp forces GLFW_RESIZABLE=FALSE, but glfwSetWindowMonitor() still
	// works: switching monitors fires GLFW's framebuffer-resize callback, which
	// makes BaseProject recreate the swapchain and call onWindowResize() for us.
	// Polled from GameLogic() once per frame.
	bool fullscreen = false;
	bool f11WasDown = false;
	int savedWinX = 0, savedWinY = 0, savedWinW = 0, savedWinH = 0;

	void toggleFullscreen() {
		if(!fullscreen) {
			glfwGetWindowPos(window, &savedWinX, &savedWinY);
			glfwGetWindowSize(window, &savedWinW, &savedWinH);
			GLFWmonitor* mon = glfwGetPrimaryMonitor();
			const GLFWvidmode* mode = glfwGetVideoMode(mon);
			glfwSetWindowMonitor(window, mon, 0, 0,
			                     mode->width, mode->height, mode->refreshRate);
			fullscreen = true;
		} else {
			glfwSetWindowMonitor(window, nullptr,
			                     savedWinX, savedWinY, savedWinW, savedWinH, 0);
			fullscreen = false;
		}
	}

	// Edge-triggered: fires once per F11 press.
	void pollFullscreenToggle() {
		bool f11Down = glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS;
		if(f11Down && !f11WasDown) toggleFullscreen();
		f11WasDown = f11Down;
	}

	// Fills the HDR chain's attachment descriptions at the current swapchain
	// size. IMPORTANT: spelled out rather than using
	// getStandardAttchmentsProperties() because no stock config is
	// floating-point -- AT_SURFACE_AA_DEPTH renders into the 8-bit sRGB
	// swapchain, capping every pixel at 1.0, which makes bloom impossible.
	void buildPostAttachments() {
		const VkFormat HDR = VK_FORMAT_R16G16B16A16_SFLOAT;
		// Black, not the old cyan: this is what shows past the geometry cull's
		// cone, and black reads as darkness/distance rather than a bright wall.
		const VkClearValue SKY = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
		const VkClearValue BLACK = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

		// --- the scene pass: multisampled HDR colour, depth, and a resolve
		// target the bloom chain and the composite can both sample.
		hdrAtt = {
			// Multisampled colour; storeOp DONT_CARE since only the resolve is read.
			{COLOR_AT, HDR,
				VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT, false, false,
				SKY,
				msaaSamples,
				VK_ATTACHMENT_LOAD_OP_CLEAR,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
			{DEPTH_AT, findDepthFormat(),
				VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
				VK_IMAGE_ASPECT_DEPTH_BIT, true, false,
				{.depthStencil = {1.0f, 0}},
				msaaSamples,
				VK_ATTACHMENT_LOAD_OP_CLEAR,
				VK_ATTACHMENT_STORE_OP_STORE,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
			// The resolve: an ordinary sampled texture, not the swapchain
			// image, so later passes can read the scene back.
			{RESOLVE_AT, HDR,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT, false, false,
				BLACK,
				VK_SAMPLE_COUNT_1_BIT,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_STORE,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
		};

		// Three identical bloom targets: single-sampled HDR colour, no depth
		// (full-screen quad, nothing to test). loadOp DONT_CARE since the quad covers every pixel.
		std::vector<AttachmentProperties> bloomTarget = {
			{COLOR_AT, HDR,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT, false, false,
				BLACK,
				VK_SAMPLE_COUNT_1_BIT,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_STORE,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
		};
		brightAtt = bloomTarget;
		blurHAtt = bloomTarget;
		blurVAtt = bloomTarget;

		// Composite writes straight to the swapchain image. IMPORTANT:
		// finalLayout PRESENT_SRC_KHR because the text/HUD passes that run
		// after this one declare it as their initialLayout and load what's already there.
		compositeAtt = {
			{COLOR_AT, swapChainImageFormat,
				VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT, false, true,
				BLACK,
				VK_SAMPLE_COUNT_1_BIT,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_STORE,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
		};
	}

	// Rebuilds attachment lists then re-inits the five render passes -- scene
	// at renderWidth()/Height(), bloom chain at bloomWidth()/Height(),
	// composite at window size. Called from localInit() and whenever
	// msaaSamples changes (a new sample count changes hdrAtt itself, which
	// only .init() re-copies). Caller must follow with RebuildPipeline().
	void initRenderPasses() {
		buildPostAttachments();

		// ATDEP_SIMPLE, not ATDEP_SURFACE_ONLY: the scene's output is sampled
		// by a later pass, so it needs dependency ordering against that read.
		// renderWidth()/Height(), not -1,-1, is what lets renderScale differ from the window.
		RP.init(this, renderWidth(swapChainExtent.width), renderHeight(swapChainExtent.height), -1, &hdrAtt,
				RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);

		RPbright.init(this, bloomWidth(), bloomHeight(), -1, &brightAtt,
					  RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		RPblurH.init(this, bloomWidth(), bloomHeight(), -1, &blurHAtt,
					 RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		RPblurV.init(this, bloomWidth(), bloomHeight(), -1, &blurVAtt,
					 RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		// Composite writes the swapchain and is read by nobody, so plain surface dependency is right.
		RPcomposite.init(this, -1, -1, -1, &compositeAtt,
						 RenderPass::getStandardDependencies(ATDEP_SURFACE_ONLY), false);
	}

	// Bloom target size, derived from the scene pass's own resolution
	// (already scaled by renderScale), not the swapchain's, since bloom reads
	// the scene's resolve target. Clamped at 1 for a minimised window.
	int bloomWidth() const {
		return std::max(1, RP.width / BLOOM_DIV);
	}
	int bloomHeight() const {
		return std::max(1, RP.height / BLOOM_DIV);
	}

	// =====================================================================
	// localInit(): scene load and world setup
	// =====================================================================
	// Here you load and setup all your Vulkan Models and Textures.
	// Here you also create your Descriptor set layouts and load the shaders for the pipelines
	// Runs once, in eight stages: Vulkan objects first, then the scene file,
	// then everything the game reads out of it. The numbered sub-banners below
	// mark the boundaries; the order between them is load-bearing (each stage
	// reads what the previous one resolved).
	void localInit() {
		// ---- 1. Render passes, layouts and vertex descriptors ----
		// IMPORTANT: windowWidth/windowHeight still hold setWindowParameters()'s
		// requested size in window points, not the real HiDPI framebuffer size
		// in pixels (the swapchain itself is sized correctly independently, via
		// glfwGetFramebufferSize() in chooseSwapExtent()). Left uncorrected,
		// every UI widget below initializes against the stale smaller size and
		// stays wrong until the first real resize -- this is why the launch
		// screen used to not look fullscreen until the window was resized.
		// Corrected here, once, rather than per-widget.
		{
			int realW = 0, realH = 0;
			glfwGetFramebufferSize(window, &realW, &realH);
			if(realW > 0 && realH > 0) {
				windowWidth = (uint32_t)realW;
				windowHeight = (uint32_t)realH;
				// Ar also defaults to a hardcoded 4/3 guess; corrected here so
				// the first frame's projection matrix is right even if the
				// real framebuffer isn't 4:3.
				Ar = (float)realW / (float)realH;
			}
		}

		// Descriptor Layouts [what will be passed to the shaders]
		DSLlocal.init(this, {
					// this array contains the binding:
					// first  element : the binding number
					// second element : the type of element (buffer or texture)
					// third  element : the pipeline stage where it will be used
					// ALL_GRAPHICS, not VERTEX_BIT: this buffer carries the
					// material too, which the fragment shader reads.
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(UniformBufferObject), 1},
					{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1}
				  });
		DSLglobal.init(this, {
					// this array contains the binding:
					// first  element : the binding number
					// second element : the type of element (buffer or texture)
					// third  element : the pipeline stage where it will be used
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(GlobalUniformBufferObject), 1}
				  });
		// Shadow sampling (set 2 of P, CookTorrance.frag): one separate sampler
		// binding per cube map. IMPORTANT: order/numbering here must agree with
		// CookTorrance.frag's shadowCube* bindings -- nothing else enforces it.
		std::vector<DescriptorSetLayoutBinding> shadowSampleBindings;
		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			shadowSampleBindings.push_back({(uint32_t)i,
											VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
											VK_SHADER_STAGE_FRAGMENT_BIT, i, 1});
		}
		DSLshadowSample.init(this, shadowSampleBindings);
		VD.init(this, {
				  {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}
				}, {
				  {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos),
				         sizeof(glm::vec3), POSITION},
				  {0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, norm),
				         sizeof(glm::vec3), NORMAL},
				  {0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, UV),
				         sizeof(glm::vec2), UV}
				});

		// IMPORTANT: Starter forces sampleShadingEnable, so MSAA becomes full
		// supersampling (fragment shader runs per sample). At the GPU's max
		// (16), CookTorrance.frag's full light loop ran 25 FPS on this Iris Xe
		// vs 94 at 4x, hence 4 as the default; live-adjustable via the MSAA slider.
		msaaSamples = VK_SAMPLE_COUNT_4_BIT;
		maxMsaaLevel = std::log2((float)getMaxUsableSampleCount());	// slider's upper bound: the device's real cap

		initRenderPasses();

		// Cube shadow render pass (torches); see RPShadowCubeCompat's member
		// comment. createCubeShadowMaps() builds the 36 real per-face framebuffers against it.
		std::vector<AttachmentProperties> cubeShadowAtt = {
			{COLOR_AT, VK_FORMAT_R32_SFLOAT,
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT, false, false,
				{.color = {.float32 = {TORCH_SHADOW_FAR_CONST, 0.0f, 0.0f, 0.0f}}},
				VK_SAMPLE_COUNT_1_BIT,
				VK_ATTACHMENT_LOAD_OP_CLEAR,
				VK_ATTACHMENT_STORE_OP_STORE,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
			{DEPTH_AT, findDepthFormat(),
				VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
				VK_IMAGE_ASPECT_DEPTH_BIT, true, false,
				{.depthStencil = {1.0f, 0}},
				VK_SAMPLE_COUNT_1_BIT,
				VK_ATTACHMENT_LOAD_OP_CLEAR,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}
		};
		// Like ATDEP_DEPTH_TRANS but for a color attachment: external->0 puts
		// the image into COLOR_ATTACHMENT_OPTIMAL before the write, 0->external
		// makes the main pass's fragment read wait for it.
		std::vector<VkSubpassDependency> cubeShadowDeps = {
			{
				VK_SUBPASS_EXTERNAL, 0,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				VK_DEPENDENCY_BY_REGION_BIT
			},
			{
				0, VK_SUBPASS_EXTERNAL,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				VK_DEPENDENCY_BY_REGION_BIT
			}
		};
		RPShadowCubeCompat.init(this, SHADOW_MAP_RES, SHADOW_MAP_RES, -1, &cubeShadowAtt, &cubeShadowDeps, true);
		RPShadowCubeCompat.create();

		createCubeShadowMaps();

		// ---- 2. Pipelines ----
		// Pipelines [Shader couples]
		// The last array, is a vector of pointer to the layouts of the sets that will
		// be used in this pipeline. The first element will be set 0, and so on..

		P.init(this, &VD, "shaders/scene/PosNormUV.vert.spv",
						  "shaders/scene/CookTorrance.frag.spv",
						  {&DSLglobal, &DSLlocal, &DSLshadowSample});

		// Ghosts: two sets, not three -- Spectral.frag is unlit and samples no shadow map.
		Pspectral.init(this, &VD, "shaders/scene/PosNormUV.vert.spv",
							  "shaders/spectral/Spectral.frag.spv",
							  {&DSLglobal, &DSLlocal});
		Pspectral.setTransparency(true);
		// IMPORTANT: back-face culling kept on, unlike Flame/ExitGlow's flat
		// billboards -- the ghost is a closed mesh with depthWriteEnable forced
		// on, so drawing both faces would blend two unsorted layers.
		// LESS_OR_EQUAL is load-bearing with the depth prepass in front: the
		// prepass leaves the nearest ghost depth, and EQUAL lets exactly those fragments through.
		Pspectral.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

		// Prepass matches Pspectral but for shader/compare op. Transparency on:
		// it emits alpha 0, so the blend returns dst and writes no colour.
		PspectralDepth.init(this, &VD, "shaders/scene/PosNormUV.vert.spv",
								   "shaders/spectral/SpectralDepth.frag.spv",
								   {&DSLglobal, &DSLlocal});
		PspectralDepth.setTransparency(true);

		// Post passes: one set layout for single-image passes, one for the
		// composite (mixes two). Starter reuses `linkSize` as the index into
		// the image-info vector, not a byte size.
		DSLpost1.init(this, {
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS,
						sizeof(PostUniformBufferObject), 1},
					{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1}
				  });
		DSLpost2.init(this, {
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS,
						sizeof(PostUniformBufferObject), 1},
					{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1},
					{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1}
				  });

		VDpost.init(this, {
					  {0, sizeof(PostVertex), VK_VERTEX_INPUT_RATE_VERTEX}
					}, {
					  {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(PostVertex, pos),
							 sizeof(glm::vec2), OTHER}
					});

		// All four share Post.vert, which is nothing but a pass-through of the
		// quad's own corners; only the fragment stage differs.
		Pbright.init(this, &VDpost, "shaders/post/Post.vert.spv", "shaders/post/BloomBright.frag.spv",
					 {&DSLpost1});
		PblurH.init(this, &VDpost, "shaders/post/Post.vert.spv", "shaders/post/BloomBlur.frag.spv",
					{&DSLpost1});
		PblurV.init(this, &VDpost, "shaders/post/Post.vert.spv", "shaders/post/BloomBlur.frag.spv",
					{&DSLpost1});
		Pcomposite.init(this, &VDpost, "shaders/post/Post.vert.spv", "shaders/post/Composite.frag.spv",
						{&DSLpost2});
		// A screen-filling quad has no meaningful facing and nothing to depth
		// test against, so culling it is one more way to end up with a black
		// screen for no benefit.
		Pbright.setCullMode(VK_CULL_MODE_NONE);
		PblurH.setCullMode(VK_CULL_MODE_NONE);
		PblurV.setCullMode(VK_CULL_MODE_NONE);
		Pcomposite.setCullMode(VK_CULL_MODE_NONE);

		// The quad itself, in NDC: Post.vert maps it straight through and
		// derives the UV from the same corners.
		static const PostVertex postCorners[4] = {
			{{-1.0f, -1.0f}}, {{1.0f, -1.0f}}, {{1.0f, 1.0f}}, {{-1.0f, 1.0f}}
		};
		Mpost = new Model();
		Mpost->indices = {0, 1, 2, 0, 2, 3};
		Mpost->vertices.resize(sizeof(postCorners));
		memcpy(Mpost->vertices.data(), postCorners, sizeof(postCorners));
		Mpost->initMesh(this, &VDpost, false);

		// Cube shadow pass's pipeline (torches). Set 0 reuses the main pass's
		// per-instance buffer to read Wm again for a different projection. Set
		// 1 is DSLshadowCubeCapture, one UBO per cube slot with that light's
		// view-projections + world position (not a push constant -- see
		// ShadowCubeUniformBufferObject's comment). Only the face index push constant is truly fixed at record time.
		DSLshadowCubeCapture.init(this, {
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
						sizeof(ShadowCubeUniformBufferObject), 1}
				  });
		VkPushConstantRange shadowCubeFacePushConstant{};
		shadowCubeFacePushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		shadowCubeFacePushConstant.offset = 0;
		shadowCubeFacePushConstant.size = sizeof(ShadowCubeFacePushConstant);
		PShadowCube.init(this, &VD, "shaders/shadow/ShadowCube.vert.spv",
								"shaders/shadow/ShadowCube.frag.spv",
								{&DSLlocal, &DSLshadowCubeCapture}, {shadowCubeFacePushConstant});
		// IMPORTANT: front faces culled, so each occluder records the side
		// turned away from the torch -- a lit surface is never in the map, so
		// shadow acne defence in shadowFromCube() needs only floating-point
		// noise slack instead of the usual bias-vs-detachment tradeoff. Cost:
		// an occluder leaks light by its own thickness (its far side is recorded).
		PShadowCube.setCullMode(VK_CULL_MODE_FRONT_BIT);
		PShadowCube.create(&RPShadowCubeCompat);

		// ---- 3. Descriptor pool and Scene load ----
		// Descriptor pool size: 4 post sets (one block each, 5 textures
		// between them) + NUM_SHADOW_CUBES for DSshadowCube[].
		DPSZs.uniformBlocksInPool = 2 + 4 + NUM_SHADOW_CUBES;
		DPSZs.texturesInPool = 1 + 5;
		DPSZs.setsInPool = 2 + 4 + NUM_SHADOW_CUBES;

		// to support scene
		VDRs.resize(1);
		VDRs[0].init("VDposNormUV",  &VD);

		// DSLshadowSample textures: none are "fromInstance" -- the shadow maps
		// are the same fixed images for every instance, unlike DSLlocal's
		// per-instance albedo. Same order as the binding list above.
		std::vector<TextureDefs> shadowMapDefs;
		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			shadowMapDefs.push_back({false, 0,
				{cubeShadowSampler.getSampler(), torchCube[i].cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
		}

		// IMPORTANT: order matters -- Scene walks techniques in registration
		// order, so "Spectral" must come second, after every opaque wall a
		// ghost could be seen through.
		PRs.resize(2);
		PRs[0].init("CookTorrance", {
							{&P, {//Pipeline and DSL for the main pass
							 /*DSLglobal*/{},
							 /*DSLlocal*/{
									/*t0*/{true,  0, {}}
								  },
							 /*DSLshadowSample*/ shadowMapDefs
								 }
								}
						  }, /*TotalNtextures*/1, &VD);

		// DSLlocal takes the ghost's albedo at slot 0 (Spectral.frag reads it
		// as a density mask). IMPORTANT: the pipeline named here is the depth
		// prepass, not Pspectral -- naming it makes SC.init() build this
		// technique's descriptor sets (shared by both passes), then it's
		// nulled below so Scene skips it; both passes are issued by hand in
		// populateCommandBuffer() in order: dungeon, flames, ghost prepass, ghost colour.
		PRs[1].init("Spectral", {
							{&PspectralDepth, {
							 /*DSLglobal*/{},
							 /*DSLlocal*/{
									/*t0*/{true,  0, {}}
								  }
								 }
								}
						  }, /*TotalNtextures*/1, &VD);

		if(SC.init(this, 1, VDRs, PRs, "assets/scenes/scene.json") != 0) {
			std::cout << "ERROR LOADING THE SCENE\n";
			exit(0);
		}

		// Unhook the ghost depth prepass from Scene's walk: flames must draw
		// between the dungeon and the prepass, or a ghost's occluding depth
		// would erase a flame poking into its body. Descriptor sets stay valid.
		SC.TI[1].T->PT[0].P = nullptr;

		// ---- 4. World: floor, colliders, doors, pickups ----
		// Cache the floor's top Y for the no-clip under-the-map clamp.
		auto floorIt = SC.InstanceIds.find("floor");
		if(floorIt != SC.InstanceIds.end() && SC.I[floorIt->second]->C != nullptr) {
			worldFloorY = SC.I[floorIt->second]->C->getExtents().yMax;
		}

		// Gameplay collision list: scene.json's auto-fit boxes, plus
		// hand-authored geometry for models an auto-fit box gets wrong.
		colliderSet.init(&SC, "assets/scenes/colliders.json");
		allColliders = colliderSet.list();

		// Interactable doors. promptOffset is the doorway's centre in the
		// leaf's local frame (the origin sits at the hinge), measured off the
		// arched SM_WallDoor_Hole_01 geometry -- re-measure if that asset changes.
		// openAngleDeg is how far the leaf swings; direction is decided at
		// runtime by Door::swingSignAwayFrom, so the sign here is only a fallback.
		// lockKeyId names the pickup that opens the padlock ("" = no lock);
		// lockLabel is the prompt's name for it, defaulting to the id.
		auto addDoor = [&](const char *id, glm::vec3 promptOffset, float openAngleDeg,
						   const char *lockKeyId = "", const char *lockLabel = "") {
			auto it = SC.InstanceIds.find(id);
			if(it == SC.InstanceIds.end()) {
				std::cout << "Door instance '" << id << "' not found, skipping\n";
				return;
			}
			Door d;
			d.instanceId = id;
			d.inst = SC.I[it->second];
			d.baseWm = d.inst->Wm;
			d.promptOffset = promptOffset;
			d.promptPos = glm::vec3(d.baseWm * glm::vec4(promptOffset, 1.0f));
			d.openAngleDeg = openAngleDeg;
			d.swingSign = openAngleDeg < 0.0f ? -1.0f : 1.0f;
			d.lockKeyId = lockKeyId;
			d.lockLabel = (lockLabel[0] != '\0') ? lockLabel : lockKeyId;
			d.locked = !d.lockKeyId.empty();
			doors.push_back(d);
		};
		// Finds a door by instance id, for the helpers below that decorate one
		// after addDoor() creates it (kept separate so ordinary doors don't
		// need to read past unused arguments).
		auto findDoor = [&](const char *id) -> Door * {
			auto it = std::find_if(doors.begin(), doors.end(),
								   [&](const Door &x) { return x.instanceId == id; });
			return it == doors.end() ? nullptr : &*it;
		};
		// Replaces the generic padlock prompts and marks a door secret.
		// `blocked` may be empty.
		auto setSecretDoor = [&](const char *id, const char *ready,
								 const char *missing, const char *blocked = "") {
			Door *d = findDoor(id);
			if(d == nullptr) {
				std::cout << "Door '" << id << "' not found, prompts not set\n";
				return;
			}
			d->promptReady = ready;
			d->promptMissing = missing;
			d->promptBlocked = blocked;
			d->secret = true;
		};
		// The level's doors, west to east. All leaves share the same asset,
		// hinge geometry and doorway centre, so promptOffset/openAngleDeg is
		// the same for each; only wall/yaw and key differ. See tools/build_scene.py.
		addDoor("iaDoorPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f);	// intro corridor -> hub, unlocked teaching door
		addDoor("hbDoorNPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f, "iron", "iron key");		// -> room A
		addDoor("hbDoorSPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f, "bronze", "bronze key");	// -> room C
		// The way out. IMPORTANT: this hole wall carries no yaw, so the leaf's
		// local +X points outward (unlike every other leaf here) -- hence the
		// flipped lock props and the negative open angle for this mirrored frame.
		addDoor("hbDoorEPanel", glm::vec3(0.0f, 2.52f, -1.231f), -100.0f, "gold", "gold key");
		// Secret bookcase (-> room B). promptOffset X is 0.25, not 0, so the
		// range check measures from the shelf face, not the hinge plane behind it.
		addDoor("hbShelfPanel", glm::vec3(0.25f, 2.20f, -1.231f), 100.0f, "book", "old book");
		// Only glows/prompts with the book in hand; no blocked line, since the
		// far side is unreachable before it's opened and it never re-locks.
		setSecretDoor("hbShelfPanel",
					  "[E] Slide the book into the gap",
					  "A book is missing from this shelf");
		// Hangs a scene instance on a door as lock hardware; separate from
		// addDoor() since a door can carry several (chains + padlock).
		// IMPORTANT: their scene.json colliders are synced off the prop's Wm
		// by the prop loop in GameLogic(), not static.
		// `flip` puts the hardware on the leaf's other face (make_door_lock.py
		// only builds one face). A rotation, not a mirror matrix: mirroring
		// flips the winding and turns the piece inside out under backface culling.
		auto addLockProp = [&](const char *doorId, const char *propId, bool flip = false) {
			auto d = std::find_if(doors.begin(), doors.end(),
								  [&](const Door &x) { return x.instanceId == doorId; });
			auto it = SC.InstanceIds.find(propId);
			if(d == doors.end() || it == SC.InstanceIds.end()) {
				std::cout << "Lock prop '" << propId << "' or its door '" << doorId
						  << "' not found, skipping\n";
				return;
			}
			const glm::vec3 pivot = glm::vec3(0.1965f, 0.0f, -1.231f);
			glm::mat4 local(1.0f);
			if(flip) {
				local = glm::translate(glm::mat4(1.0f), pivot)
					  * glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f))
					  * glm::translate(glm::mat4(1.0f), -pivot);
			}
			d->lockProps.push_back({SC.I[it->second], local});

			// Grow the auto-fit box outward (xMax only, model space) so the
			// player stops before the hardware is in the torch's reach.
			Collider *propC = SC.I[it->second]->C;
			if(propC != nullptr) {
				// getExtents() is world-space; at identity it reads back the model box.
				propC->setWorldMatrix(glm::mat4(1.0f));
				AABBextents L = propC->getExtents();
				propC->initAABB(L.xMin, L.yMin, L.zMin,
								L.xMax + Door::LOCK_PROP_KEEPOUT, L.yMax, L.zMax);
			}

			// Same flag decides where the hardware is drawn and which side E
			// works from, so the prompt can't disagree with the screen.
			d->lockFaceSign = flip ? -1.0f : 1.0f;
		};
		// All three locked leaves take flip=true: the hardware exports on the
		// leaf's local +X face, and the player always stands on the other side of that.
		addLockProp("hbDoorNPanel", "hbDoorNChains", true);
		addLockProp("hbDoorNPanel", "hbDoorNPadlock", true);
		addLockProp("hbDoorSPanel", "hbDoorSChains", true);
		addLockProp("hbDoorSPanel", "hbDoorSPadlock", true);
		addLockProp("hbDoorEPanel", "hbDoorEChains", true);
		addLockProp("hbDoorEPanel", "hbDoorEPadlock", true);

		// Cache which door is the way out, so the light outside is driven from
		// its swing without a string compare per frame.
		for(size_t i = 0; i < doors.size(); i++) {
			if(doors[i].instanceId == "hbDoorEPanel") {
				exitDoorIndex = (int)i;
				break;
			}
		}
		if(exitDoorIndex < 0) {
			std::cout << "Exit door 'hbDoorEPanel' not found: no daylight outside it\n";
		}

		// Key graph is strictly linear: iron -> hbDoorN -> book -> hbShelfPanel
		// -> bronze -> hbDoorS -> gold -> hbDoorE -> win.

		// World pickups. worldPos is read from the instance's own Wm (already
		// in scene.json). Passing a keyId makes the pickup a key: it goes on
		// the ring when collected, opens any Door with a matching lockKeyId
		// (or the exit), and is spent on first use. handTiltDeg/handOffset
		// default to the key pose (HAND_KEY_TILT_DEG/HAND_KEY_OFFSET).
		auto addPickup = [&](const char *id, const char *keyId = "",
							 glm::vec3 handTiltDeg = HAND_KEY_TILT_DEG,
							 glm::vec3 handOffset = HAND_KEY_OFFSET) {
			auto it = SC.InstanceIds.find(id);
			if(it == SC.InstanceIds.end()) {
				std::cout << "Pickup instance '" << id << "' not found, skipping\n";
				return;
			}
			Pickup p;
			p.instanceId = id;
			p.inst = SC.I[it->second];
			p.worldPos = glm::vec3(p.inst->Wm[3]);
			p.spawnWm = p.inst->Wm;
			p.spawnPos = p.worldPos;
			p.keyId = keyId;
			p.handTiltDeg = handTiltDeg;
			p.handOffset = handOffset;
			// Uniform scale from column 0's length, so scene.json's "scale" stays the one source.
			p.worldScale = glm::length(glm::vec3(p.inst->Wm[0]));
			pickups.push_back(p);
		};
		addPickup("hbKeyIron", "iron");     // on the hub table
		addPickup("rbKeyBronze", "bronze"); // dark NW corner of room B
		addPickup("rcKeyGold", "gold");     // atop the barrels in room C
		// The book: rides the same keyRing/findKeyInRing/consumeKey machinery
		// as keys, opens the bookcase only. Tilt stands its flat mesh up
		// cover-first; offset is HAND_KEY_OFFSET raised 0.16 since the book's
		// origin (mesh centre) sits lower relative to its grip than a key's does.
		addPickup("raBook", "book", glm::vec3(84.0f, -24.0f, 0.0f),
				  HAND_KEY_OFFSET + glm::vec3(0.0f, 0.16f, 0.0f));

		// Where the book ends up once spent: the gap on the bookcase's third
		// shelf, in the leaf's local frame (see Door::LockProp::whenUnlocked).
		// IMPORTANT: these three numbers also live in make_bookshelf.py's
		// BOOK_SLOT_* -- move one, move the other. rotY(180)*rotX(-90) stands
		// the book up spine-out without mirroring its winding.
		auto addSlotProp = [&](const char *doorId, const char *pickupId,
							   const glm::mat4 &local) {
			Door *d = findDoor(doorId);
			auto p = std::find_if(pickups.begin(), pickups.end(),
								  [&](const Pickup &x) { return x.instanceId == pickupId; });
			if(d == nullptr || p == pickups.end()) {
				std::cout << "Slot prop '" << pickupId << "' or its door '" << doorId
						  << "' not found, skipping\n";
				return;
			}
			// Scale from the pickup's authored matrix, so a resized book still
			// lands in its gap at world size.
			d->lockProps.push_back({p->inst,
									local * glm::scale(glm::mat4(1.0f), glm::vec3(p->worldScale)),
									true});
		};
		addSlotProp("hbShelfPanel", "raBook",
					glm::translate(glm::mat4(1.0f), glm::vec3(0.454f, 1.730f, -1.266f))
				  * glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f))
				  * glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)));

		// Now that every door and lock prop exists, split allColliders into
		// the movers (excluded from the grid, tested live every query) and
		// everything else (cached in the grid, built once).
		ghostDynamicColliders.clear();
		for(Door &d : doors) {
			if(d.inst != nullptr && d.inst->C != nullptr) {
				ghostDynamicColliders.push_back(d.inst->C);
			}
			for(Door::LockProp &prop : d.lockProps) {
				if(prop.inst->C != nullptr) {
					ghostDynamicColliders.push_back(prop.inst->C);
				}
			}
		}
		std::vector<Collider *> staticColliders;
		staticColliders.reserve(allColliders.size());
		for(Collider *c : allColliders) {
			bool isMover = std::find(ghostDynamicColliders.begin(), ghostDynamicColliders.end(), c)
						 != ghostDynamicColliders.end();
			if(!isMover) staticColliders.push_back(c);
		}
		ghostColliderGrid.build(staticColliders);

		// ---- 5. Spawn pose and the rules of the run ----
		// Player's spawn pose, captured before anything can move it; restartRun() uses this.
		spawnPos = camPos;
		spawnYaw = camYaw;
		spawnPitch = camPitch;

		// The rules of the game: hunt timings, win box, ghost patrols.
		{
			std::ifstream ifs("assets/scenes/gameplay.json");
			if(!ifs.is_open()) {
				std::cout << "gameplay.json not found: default hunt timings, no ghosts\n";
				// Still has to be armed, or a default HuntCycle's zero phase timer falls straight into a hunt.
				huntCycle.init(nlohmann::json::object());
			} else {
				nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);
				huntCycle.init(js.value("hunt", nlohmann::json::object()));

				if(js.contains("exit")) {
					const nlohmann::json &e = js["exit"];
					if(e.contains("box") && e["box"].size() == 6) {
						glm::vec3 a(e["box"][0].get<float>(), e["box"][1].get<float>(), e["box"][2].get<float>());
						glm::vec3 b(e["box"][3].get<float>(), e["box"][4].get<float>(), e["box"][5].get<float>());
						// min/max, not authored order, so swapped corners still describe the same box.
						exitBoxMin = glm::min(a, b);
						exitBoxMax = glm::max(a, b);
						exitHasBox = true;
					} else {
						std::cout << "gameplay.json: \"exit\" needs a 6-number \"box\", the run can't be won\n";
					}
				}

				for(const auto &g : js.value("ghosts", nlohmann::json::array())) {
					std::string id = g.value("instance", std::string(""));
					auto it = SC.InstanceIds.find(id);
					if(it == SC.InstanceIds.end()) {
						std::cout << "gameplay.json: no scene instance '" << id
								  << "', ghost skipped\n";
						continue;
					}

					Ghost gh;
					gh.instanceId = id;
					gh.inst = SC.I[it->second];
					gh.speed = g.value("speed", gh.speed);
					gh.chaseSpeed = g.value("chaseSpeed", gh.chaseSpeed);
					for(const auto &w : g.value("waypoints", nlohmann::json::array())) {
						if(w.size() != 3) {
							std::cout << "gameplay.json: ghost '" << id
									  << "' has a waypoint that isn't 3 numbers, skipped\n";
							continue;
						}
						gh.waypoints.push_back(glm::vec3(w[0].get<float>(), w[1].get<float>(), w[2].get<float>()));
					}
					// One waypoint is a ghost standing still, almost certainly a typo.
					if(gh.waypoints.size() < 2) {
						std::cout << "gameplay.json: ghost '" << id
								  << "' needs at least 2 waypoints, skipped\n";
						continue;
					}
					gh.pos = gh.waypoints[0];
					gh.bobPhase = (float)ghosts.size() * 2.399963f;	// staggered so identical ghosts don't hover in lockstep
					ghosts.push_back(gh);
				}
				std::cout << "gameplay.json: " << ghosts.size() << " ghosts loaded\n";

				// Fit the ghost's collision size from the mesh (all ghosts share Ghost.gltf).
				if(!ghosts.empty()) {
					Collider fit;
					fit.fitAABB(SC.M[ghosts[0].inst->Mid]);
					AABBextents E = fit.getExtents();	// fit's Wm is identity: local space

					// fitAABB is blind to scene.json's "scale"; multiply it back in.
					float instScale = glm::length(glm::vec3(ghosts[0].inst->Wm[0]));

					ghostBodyBottom = E.yMin * instScale;
					ghostBodyTop = E.yMax * instScale;

					float halfX = 0.5f * (E.xMax - E.xMin);
					float halfZ = 0.5f * (E.zMax - E.zMin);
					ghostRadius = ghostXZFitShrink * 0.5f * (halfX + halfZ) * instScale;

					std::cout << "Ghost collision fitted from mesh (scale " << instScale
							  << "): radius " << ghostRadius
							  << ", vertical [" << ghostBodyBottom << ", " << ghostBodyTop << "]\n";

					// Patrol legs have no wall resolution, so a leg authored
					// through a wall doesn't fail loudly; warn here instead.
					for(const Ghost &g : ghosts) {
						int n = (int)g.waypoints.size();
						for(int i = 0; i < n; i++) {
							const glm::vec3 &from = g.waypoints[i];
							const glm::vec3 &to = g.waypoints[(i + 1) % n];
							glm::vec2 delta(to.x - from.x, to.z - from.z);
							float len = glm::length(delta);
							if(len < 1e-4f) continue;
							if(!ghostPathClear(from, delta / len, len, ghostRadius)) {
								std::cout << "gameplay.json: ghost '" << g.instanceId
										  << "' patrols through a collider between waypoint "
										  << i << " and " << ((i + 1) % n)
										  << " -- it will clip that geometry and can wedge itself there on a hunt\n";
							}
						}
					}
				}
			}
		}

		// ---- 6. Torches, candles and flames ----
		// Held torch. Starts on the floor at its authored pose; picked up with
		// [E], after which GameLogic rebuilds its Wm from the camera.
		{
			auto it = SC.InstanceIds.find("handTorch");
			if(it == SC.InstanceIds.end()) {
				std::cout << "Hand torch instance 'handTorch' not found, skipping\n";
			} else {
				handTorchInst = SC.I[it->second];
				handTorchSpawnWm = handTorchInst->Wm;
				// Lifted off the instance origin (below the mesh, torch lies
				// on its side) so the crosshair lands on the torch body.
				handTorchWorldPos = glm::vec3(handTorchInst->Wm[3]) +
									glm::vec3(0.0f, 0.35f, 0.0f);
			}
		}

		// maxInstances covers every torch and candle mesh in the level, set
		// comfortably above the ~13 wall torches + held torch + ~8 candles the
		// level runs; spawn() past the cap fails silently (see addTorchFlame).
		flame.init(this, &DSLglobal, &DSglobal, 28);

		// No DSLglobal/DSglobal here: the daylight quads are unshaded and bind
		// sets of their own. Two quads (upright wall + ground); see
		// EXIT_GLOW_CENTER for why the outward-swinging door needs both.
		exitGlow.init(this, EXIT_GLOW_COUNT);

		debugLines.init(this);

		auto addTorchFlame = [&](const char *id, glm::vec3 anchor, bool heldByCamera = false,
								 glm::vec3 color = TORCH_LIGHT_COLOR, float sizeScale = 1.0f,
								 float lightScale = 1.0f, bool isCandle = false,
								 bool burning = true) {
			auto it = SC.InstanceIds.find(id);
			if(it == SC.InstanceIds.end()) {
				std::cout << "Torch instance '" << id << "' not found, skipping its flame\n";
				return;
			}
			Instance *inst = SC.I[it->second];
			float seed = (float)torchFlames.size() * 2.3971f;	// irrational-ish stride avoids phase collisions
			int flameId = flame.spawn(seed);
			if(flameId < 0) {
				return;
			}
			TorchFlame tf{};
			tf.inst = inst;
			tf.flameId = flameId;
			tf.anchor = anchor;
			tf.heldByCamera = heldByCamera;
			tf.color = color;
			tf.baseColor = color;
			tf.sizeScale = sizeScale;
			tf.lightScale = lightScale;
			tf.isCandle = isCandle;
			tf.burning = burning;
			tf.spawnBurning = burning;
			tf.ignitionScale = burning ? 1.0f : 0.0f;	// at rest if burning, 0 to play in full when lit
			if(!heldByCamera) {	// held torch's anchor is recomputed every frame instead
				tf.anchorWorld = glm::vec3(inst->Wm * glm::vec4(anchor, 1.0f));
			}
			tf.shadowCandidate = !heldByCamera;
			if(tf.shadowCandidate) {
				tf.shadowFaceMatrices = cubeFaceMatricesFor(tf.anchorWorld);	// static: computed once, not per frame
			}
			tf.phase = seed * 37.0f;	// scaled up since fireNoise hashes on the integer lattice
			torchFlames.push_back(tf);
		};

		if(handTorchInst != nullptr) {
			// Spawned UNLIT; lit from a burning wall torch with [E].
			// handFlameIdx is the slot it lands in.
			handFlameIdx = (int)torchFlames.size();
			addTorchFlame("handTorch", TORCH_FLAME_ANCHOR, true,
						  TORCH_LIGHT_COLOR, 1.0f, 1.0f, /*isCandle=*/false,
						  /*burning=*/false);
		}

		// Every OTHER flame: from flames.json, keyed by MODEL, so a level's
		// torches/candles get fire just by using the standard meshes.
		// "handTorch" above is the one literal-id exception (a gameplay
		// singleton).
		{
			std::ifstream ifs("assets/scenes/flames.json");
			if(!ifs.is_open()) {
				std::cout << "flames.json not found, no flames beyond the held torch\n";
			} else {
				nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);

				struct FlameDef {
					glm::vec3 anchor = TORCH_FLAME_ANCHOR;
					glm::vec3 color = TORCH_LIGHT_COLOR;
					float sizeScale = 1.0f;
					float lightScale = 1.0f;
					bool isCandle = false;
					// Default true, so torches are alight at load; candles set
					// it false and wait for the player.
					bool burning = true;
				};
				// Only overwrites what's present, so a def can start from
				// another (a model's defaults, for an override to build on).
				auto readVec3 = [](const nlohmann::json &j, const glm::vec3 &fallback) {
					if(!j.is_array() || j.size() != 3) return fallback;
					return glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
				};
				auto applyFlameDef = [&](const nlohmann::json &j, FlameDef def) {
					if(j.contains("anchor"))     def.anchor = readVec3(j["anchor"], def.anchor);
					if(j.contains("color"))      def.color = readVec3(j["color"], def.color);
					if(j.contains("sizeScale"))  def.sizeScale = j["sizeScale"].get<float>();
					if(j.contains("lightScale")) def.lightScale = j["lightScale"].get<float>();
					if(j.contains("isCandle"))   def.isCandle = j["isCandle"].get<bool>();
					if(j.contains("burning"))    def.burning = j["burning"].get<bool>();
					return def;
				};

				// Resolve each model name to its Mid once, for O(1) lookup per instance.
				std::unordered_map<int, FlameDef> byModel;
				if(js.contains("models")) {
					for(auto it = js["models"].begin(); it != js["models"].end(); ++it) {
						auto mit = SC.MeshIds.find(it.key());
						if(mit == SC.MeshIds.end()) {
							std::cout << "flames.json: unknown model '" << it.key() << "', skipped\n";
							continue;
						}
						byModel[mit->second] = applyFlameDef(it.value(), FlameDef{});
					}
				}

				const nlohmann::json *overrides = js.contains("overrides") ? &js["overrides"] : nullptr;

				for(const auto &kv : SC.InstanceIds) {
					const std::string &id = kv.first;
					if(id == "handTorch") {
						continue;
					}
					Instance *inst = SC.I[kv.second];
					auto dit = byModel.find(inst->Mid);
					if(dit == byModel.end()) {
						continue;
					}
					FlameDef def = dit->second;
					if(overrides != nullptr && overrides->contains(id)) {
						def = applyFlameDef((*overrides)[id], def);
					}
					addTorchFlame(id.c_str(), def.anchor, false, def.color, def.sizeScale,
								  def.lightScale, def.isCandle, def.burning);
				}
			}
		}

		// ---- 7. Materials, lights and cube-shadow slots ----
		// Surface parameters for the BRDF, one per model.
		materials.init(&SC, "assets/scenes/materials.json");

		// After Scene::init: a light can be anchored to an instance and needs
		// that instance's world matrix.
		sceneLights.init(&SC, "assets/scenes/lights.json");

		// After sceneLights.init(): needs the resolved world position of every
		// shadow-casting light, which instance+offset lights only have once
		// SceneLights has read scene.json's world matrices.
		computeShadowMatrices();

		// Dynamic pool: everything past lights.json's fixed slots and before the held torch's reserved last one.
		dynamicShadowSlotBase = activeCubeShadows;
		dynamicSlotOccupant.fill(-1);
		lastRenderedOccupant.fill(SHADOW_SLOT_UNSET);	// first frame's diff queues every slot for its one render
		// The render loop walks [0, activeCubeShadows) every frame, so the
		// dynamic pool must be counted even while empty.
		activeCubeShadows = std::max(activeCubeShadows, HAND_TORCH_SHADOW_INDEX);

		// Held torch's slot isn't in lights.json, so computeShadowMatrices()
		// never counts it; its matrices come from updateHandTorchShadow() instead.
		if(handTorchInst != nullptr) {
			activeCubeShadows = std::max(activeCubeShadows, HAND_TORCH_SHADOW_INDEX + 1);
		}

		// ---- 8. Text, overlays and the cheat menu ----
		// initializes the textual output
		txt.init(this, windowWidth, windowHeight);
		// initializes the flat-quad background/highlight layer for the cheat HUD
		uiQuad.init(this, windowWidth, windowHeight);
		// Distinct submitOrder/buffer name from uiQuad so they don't collide; same for the three below.
		crosshair.init(this, windowWidth, windowHeight, 9002, "crosshair");
		setCrosshairQuad();
		pauseQuad.init(this, windowWidth, windowHeight, 9003, "pause_quad");
		startScreenQuad.init(this, windowWidth, windowHeight, 9004, "start_screen_quad");
		settingsQuad.init(this, windowWidth, windowHeight, 9005, "settings_quad");

		// submits the main command buffer
		submitCommandBuffer("main", 0, populateCommandBufferAccess, this);

		// The FPS counter (text id 1). Never removed for the rest of the run,
		// which also keeps TextMaker's live block list from ever going empty --
		// createTextMesh() (unpatchable skeleton code) does M->vertices[0] with
		// no guard for zero total characters, which crashed when an earlier
		// version removed this while the launch screen was open.
		txt.print(1.0f, 1.0f, "FPS:",1,"CO",false,false,true,TAL_RIGHT,TRH_RIGHT,TRV_BOTTOM,{1.0f,0.0f,0.0f,1.0f},{0.8f,0.8f,0.0f,1.0f});

		hud.init(&txt, &uiQuad);
		pauseMenu.init(&txt, &pauseQuad);
		startScreen.init(&txt, &startScreenQuad, windowTitle);	// reuses windowTitle, one source for the launch screen's title
		startScreen.setOpen(true, windowWidth, windowHeight);
		settingsMenu.init(&txt, &settingsQuad);
		// Same named methods as the cheat HUD's copies below, so the two can't drift.
		settingsMenu.addSlider("Render Scale", &renderScale, 0.4f, 1.0f, 0.05f,
							   [this]() { applyRenderScaleChange(); },
							   [this](float v) { return formatRenderScale(v); });
		settingsMenu.addSlider("MSAA", &msaaLevel, 0.0f, maxMsaaLevel, 1.0f,
							   [this]() { applyMsaaChange(); },
							   [this](float v) { return formatMsaaLevel(v); });
		hud.addToggle("Collision", &cheats.collisionEnabled);
		hud.addToggle("Show Coordinates", &cheats.showCoordinates);
		hud.addToggle("Jump", &cheats.jumpEnabled);

		hud.addToggle("Hunt", &huntCycle.forceHunt);
		hud.addToggle("Ghosts Can Catch", &cheats.ghostsCanCatch);

		// Lighting rows: sources first, then shading. Spotlight/Ambient point
		// into sceneLights (which owns them); Torches/Holding Torch into
		// cheats (flame lights never go through SceneLights).
		hud.addToggle("Torches", &cheats.roomTorchesEnabled);
		hud.addToggle("Holding Torch", &cheats.handTorchEnabled);
		hud.addToggle("Spotlight", &sceneLights.spotEnabled);
		hud.addToggle("Torch Bounce", &sceneLights.bounceEnabled);
		hud.addToggle("Shadows", &cheats.shadowsEnabled);
		hud.addToggle("Torch Shadows", &cheats.torchShadowsEnabled);
		hud.addToggle("Candle Shadows", &cheats.candleShadowsEnabled);
		hud.addToggle("Held Torch Casts Shadow", &cheats.handTorchModelCastsShadowWhenHeld);
		hud.addToggle("Specular", &cheats.specularEnabled);
		hud.addToggle("Tone Mapping", &cheats.toneMapEnabled);
		hud.addToggle("Fullbright", &cheats.unlit);
		hud.addToggle("Show Normals", &cheats.showNormals);
		hud.addToggle("Focus Glow", &cheats.focusGlowEnabled);
		hud.addToggle("Light Gizmos", &cheats.showLightGizmos);
		hud.addToggle("Shadow Frustums", &cheats.showShadowFrustums);
		hud.addToggle("Show Colliders", &cheats.showColliders);
		hud.addToggle("Debug Camera", &cheats.debugCam);
		hud.addToggle("Light Heatmap", &cheats.showLightHeatmap);
		hud.addToggle("Shadow Gap", &cheats.showShadowGap);

		hud.addSlider("Render Scale", &renderScale, 0.4f, 1.0f, 0.05f,
					  [this]() { applyRenderScaleChange(); },
					  [this](float v) { return formatRenderScale(v); });
		hud.addSlider("MSAA", &msaaLevel, 0.0f, maxMsaaLevel, 1.0f,
					  [this]() { applyMsaaChange(); },
					  [this](float v) { return formatMsaaLevel(v); });
	}

	// =====================================================================
	// Cube shadow maps: slot assignment and capture
	// =====================================================================
	// Does slot t currently have a light in it? A lights.json slot is occupied
	// by construction; a dynamic-pool slot only once given a flame; the held
	// torch's only while the torch exists.
	bool cubeSlotOccupied(int t) const {
		if(t == HAND_TORCH_SHADOW_INDEX) {
			return HAND_TORCH_SHADOW_INDEX < activeCubeShadows && cheats.torchShadowsEnabled;
		}
		return (t < dynamicShadowSlotBase) || (dynamicSlotOccupant[t] != -1);
	}

	// The g (falloff reference distance) a flame's point light is uploaded
	// with; shared by the light-append loop and the shadow-reach computation.
	static float flameLightG(bool isCandle, float intensity) {
		return TORCH_LIGHT_G * (isCandle ? CANDLE_LIGHT_G_SCALE : 1.0f)
			   * (0.88f + 0.12f * intensity);
	}

	// How far from a point light a moving occluder can still cast a shadow
	// worth capturing. Not a radius: solving (g/d)^beta = SHADOW_REACH_CUTOFF
	// for d gives the distance where that light is down to a cutoff fraction of
	// peak. A bigger g gets a proportionally bigger answer, so this is correct
	// for candles as well as torches. Clamped to the cube's far plane.
	static float lightReachForCutoff(float g, float beta, float cutoff) {
		const float d = g * std::pow(cutoff, -1.0f / beta);
		return std::min(d, TORCH_SHADOW_FAR_CONST);
	}
	static float shadowRelevantReach(float g, float beta) {
		return lightReachForCutoff(g, beta, SHADOW_REACH_CUTOFF);
	}

	// Reach fed to lightReachesViewCone() for a flame: the distance its light is
	// down to LIGHT_CULL_REACH_CUTOFF of peak, at full (un-flickered) intensity
	// so a one-frame gutter can't drop and re-acquire the light or its shadow.
	static float flameLightCullReach(const TorchFlame &tf) {
		return lightReachForCutoff(flameLightG(tf.isCandle, 1.0f), TORCH_LIGHT_BETA,
								   LIGHT_CULL_REACH_CUTOFF);
	}

	std::array<glm::mat4, 6> cubeFaceMatricesFor(const glm::vec3 &pos) const {
		std::array<glm::mat4, 6> out{};
		for(int face = 0; face < 6; face++) {
			glm::mat4 view = glm::lookAt(pos, pos + CUBE_FACE_DIR[face], CUBE_FACE_UP[face]);
			glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f,
											  TORCH_SHADOW_NEAR_CONST, TORCH_SHADOW_FAR_CONST);
			out[face] = proj * view;
		}
		return out;
	}

	// Builds the view-projection matrix each shadow pass renders with, in
	// LightData::shadowIndex order. Called once from localInit(), since
	// torches never move. Reads sceneLights.all() rather than re-deriving a
	// point light's instance+offset world position here.
	void computeShadowMatrices() {
		for(const LightData &L : sceneLights.all()) {
			if(L.shadowIndex < 0) {
				continue;
			}

			// Six 90-degree perspective faces, axis-aligned on world X/Y/Z,
			// cover the whole sphere with no seam and no per-torch aim tuning.
			// IMPORTANT: no Y-flip here -- a cube map is sampled by direction
			// (samplerCube), never rasterized, so there's no Vulkan-vs-GL NDC
			// mismatch to correct; flipping would mis-rotate which face's texels land where.
			std::array<glm::mat4, 6> faces = cubeFaceMatricesFor(L.pos);
			for(int face = 0; face < 6; face++) {
				torchFaceMatrices[L.shadowIndex][face] = faces[face];
			}
			torchLightPos[L.shadowIndex] = L.pos;
			torchShadowReach[L.shadowIndex] = shadowRelevantReach(L.g, L.beta);
			activeCubeShadows = std::max(activeCubeShadows, L.shadowIndex + 1);
		}
	}

	// cubeFaceMatricesFor() for HAND_TORCH_SHADOW_INDEX, every frame instead of
	// once: the held torch moves with the camera. Runs before
	// populateCommandBuffer() records this frame's cube passes.
	void updateHandTorchShadow(const glm::vec3 &lightPos) {
		std::array<glm::mat4, 6> faces = cubeFaceMatricesFor(lightPos);
		for(int face = 0; face < 6; face++) {
			torchFaceMatrices[HAND_TORCH_SHADOW_INDEX][face] = faces[face];
		}
		torchLightPos[HAND_TORCH_SHADOW_INDEX] = lightPos;
		// Same reach every other torch gets, so the mover test treats it alike.
		torchShadowReach[HAND_TORCH_SHADOW_INDEX] =
				shadowRelevantReach(flameLightG(false, 1.0f), TORCH_LIGHT_BETA);
	}

	// Priority-based reassignment for the dynamic shadow-cube pool. Empty
	// slots go to candidates first; the contest logic below only fires once
	// there are more shadow-worthy lights than slots.
	// IMPORTANT: a plain "N nearest per tick" rule thrashes at the boundary
	// between two candidates, forcing a fresh render on every flip.
	// SHADOW_SWAP_MARGIN (swap only if genuinely closer) and
	// SHADOW_REASSIGN_INTERVAL fix that. `forward` only reorders candidates
	// (facingBiasedDistSq()), never decides whether one gets a slot.
	void updateDynamicShadowSlots(const glm::vec3 &eyePos, const glm::vec3 &forward) {
		const int base = dynamicShadowSlotBase;
		const int count = HAND_TORCH_SHADOW_INDEX - base;
		if(count <= 0) {
			return;
		}

		auto distSqTo = [&](const TorchFlame &tf) {
			glm::vec3 pos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			return facingBiasedDistSq(pos, eyePos, forward);
		};

		auto assignSlot = [&](int slot, int flameIdx) {
			dynamicSlotOccupant[slot] = flameIdx;
			TorchFlame &tf = torchFlames[flameIdx];
			tf.shadowSlot = slot;
			for(int face = 0; face < 6; face++) {
				torchFaceMatrices[slot][face] = tf.shadowFaceMatrices[face];
			}
			torchLightPos[slot] = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			// Full intensity, not this frame's flicker: a torch guttering for a
			// frame shouldn't drop and re-acquire a ghost at the edge of its light.
			torchShadowReach[slot] = shadowRelevantReach(flameLightG(tf.isCandle, 1.0f),
														TORCH_LIGHT_BETA);
		};

		// Torch/Candle Shadows HUD toggles: a disabled category never
		// candidates and drops any slot it held this tick.
		auto categoryEnabled = [&](const TorchFlame &tf) {
			if(!flameBurning(tf)) return false;
			return tf.isCandle ? cheats.candleShadowsEnabled : cheats.torchShadowsEnabled;
		};

		// A torch whose light reaches nothing drawn gives its slot back.
		// Existing occupants get a margin (SHADOW_VIEW_KEEP_MARGIN) so hovering
		// on the cone boundary doesn't re-render every pass; a fresh candidate
		// must clear the un-margined bar.
		auto lightsView = [&](const TorchFlame &tf, float reachMargin) {
			if(tf.heldByCamera) return true;
			glm::vec3 p = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			return lightReachesViewCone(p, flameLightCullReach(tf) * reachMargin,
										eyePos, forward);
		};

		for(int s = base; s < base + count; s++) {
			int idx = dynamicSlotOccupant[s];
			if(idx != -1 && (!categoryEnabled(torchFlames[idx])
							 || !lightsView(torchFlames[idx], SHADOW_VIEW_KEEP_MARGIN))) {
				torchFlames[idx].shadowSlot = -1;
				dynamicSlotOccupant[s] = -1;
			}
		}

		struct Cand { int flameIdx; float distSq; };
		std::vector<Cand> waiting;
		for(size_t i = 0; i < torchFlames.size(); i++) {
			const TorchFlame &tf = torchFlames[i];
			if(!tf.shadowCandidate || tf.shadowSlot >= 0 || !categoryEnabled(tf)
			   || !lightsView(tf, 1.0f)) {
				continue;
			}
			waiting.push_back({(int)i, distSqTo(tf)});
		}
		std::sort(waiting.begin(), waiting.end(),
				  [](const Cand &a, const Cand &b) { return a.distSq < b.distSq; });

		// Empty slots first, unconditionally.
		size_t wi = 0;
		for(int s = base; s < base + count && wi < waiting.size(); s++) {
			if(dynamicSlotOccupant[s] != -1) {
				continue;
			}
			assignSlot(s, waiting[wi].flameIdx);
			wi++;
		}

		// Occupied slots, farthest occupant first, contested against the best
		// remaining waiter; stops at the first failed margin since every later pairing fails too.
		std::vector<int> occupied;
		for(int s = base; s < base + count; s++) {
			if(dynamicSlotOccupant[s] != -1) {
				occupied.push_back(s);
			}
		}
		std::sort(occupied.begin(), occupied.end(), [&](int a, int b) {
			return distSqTo(torchFlames[dynamicSlotOccupant[a]])
				 > distSqTo(torchFlames[dynamicSlotOccupant[b]]);
		});

		const float marginSq = SHADOW_SWAP_MARGIN * SHADOW_SWAP_MARGIN;
		for(size_t oi = 0; wi < waiting.size() && oi < occupied.size(); oi++, wi++) {
			int slot = occupied[oi];
			float occupantDistSq = distSqTo(torchFlames[dynamicSlotOccupant[slot]]);
			if(waiting[wi].distSq * marginSq >= occupantDistSq) {
				break;
			}
			torchFlames[dynamicSlotOccupant[slot]].shadowSlot = -1;
			assignSlot(slot, waiting[wi].flameIdx);
		}
	}

	// Local-space bounding sphere per model index (xyz centre, w radius),
	// filled on first use. w < 0 = not fitted. Feeds recordCubeSlotFaces()'s
	// per-face cull: cheap enough to test 6x per instance per slot, and a
	// loose (sphere-around-AABB) cull only costs a draw, never a missing shadow.
	std::vector<glm::vec4> modelSphereCache;
	// Scratch for the same cull, a member so it reuses its allocation.
	struct ShadowCaster {
		Instance *inst;
		glm::vec3 centre;
		float radius;
	};
	std::vector<ShadowCaster> shadowCasterScratch;

	const glm::vec4 &modelSphere(int mid) {
		if((int)modelSphereCache.size() <= mid) {
			modelSphereCache.resize(mid + 1, glm::vec4(0.0f, 0.0f, 0.0f, -1.0f));
		}
		glm::vec4 &s = modelSphereCache[mid];
		if(s.w < 0.0f) {
			Collider fit;
			fit.fitAABB(SC.M[mid]);
			const AABBextents E = fit.getExtents();	// fit's Wm is identity, so local space
			const glm::vec3 centre((E.xMin + E.xMax) * 0.5f,
								   (E.yMin + E.yMax) * 0.5f,
								   (E.zMin + E.zMax) * 0.5f);
			const glm::vec3 half((E.xMax - E.xMin) * 0.5f,
								 (E.yMax - E.yMin) * 0.5f,
								 (E.zMax - E.zMin) * 0.5f);
			s = glm::vec4(centre, glm::length(half));
		}
		return s;
	}

	// Is a world-space sphere inside one cube face's 90-degree frustum? `rel`
	// is its centre relative to the light. By hand, not plane extraction: the
	// faces are axis-aligned and square, so it's a handful of compares, run once per instance per face.
	bool sphereInCubeFace(const glm::vec3 &rel, float radius, int face) const {
		const glm::vec3 &d = CUBE_FACE_DIR[face];
		const float along = glm::dot(rel, d);
		// Behind the far plane, measured along the axis (cheaper, never
		// over-rejects since |rel| >= along).
		if(along - radius > TORCH_SHADOW_FAR_CONST) {
			return false;
		}
		// Side planes at 45 degrees: a radius-r sphere sticks past until its
		// centre is r*sqrt(2) behind.
		const float slack = radius * 1.41421356f;
		for(int ax = 0; ax < 3; ax++) {
			if(std::abs(d[ax]) > 0.5f) {
				continue;	// the face's own axis, handled above
			}
			if(along + rel[ax] < -slack || along - rel[ax] < -slack) {
				return false;
			}
		}
		return true;
	}

	// One cube slot's six-face render, called via submitCubeShadowCaptures():
	// the held torch every frame, others when their occupant changes or a mover invalidates them.
	// ownerInst: excluded from its own shadow pass.
	// cullPerFace: drop instances outside the face being drawn (~6x fewer
	// draws); only sound for a buffer recorded and submitted the same frame.
	// faceMask: bit f = draw face f; a face left out keeps its old texels, but
	// IMPORTANT: must be set at least once per slot or the image stays
	// VK_IMAGE_LAYOUT_UNDEFINED and can't be bound.
	void recordCubeSlotFaces(VkCommandBuffer commandBuffer, int currentImage, int t,
							 bool slotOccupied, Instance *ownerInst, bool cullPerFace = false,
							 uint8_t faceMask = ALL_CUBE_FACES) {
		// Built once for all six faces (the world spheres don't depend on which
		// face is being drawn), and only when there's anything to draw.
		if(slotOccupied && faceMask != 0) {
			shadowCasterScratch.clear();
			for(int j = 0; j < SC.TI[0].InstanceCount; j++) {
				Instance &inst = SC.TI[0].I[j];
				if(!materials.forModel(inst.Mid).castsShadow) {
					continue;
				}
				if(&inst == ownerInst) {
					continue;
				}
				// The held torch's mesh: an occluder on the floor, not once in
				// hand (cheats.handTorchModelCastsShadowWhenHeld).
				if(&inst == handTorchInst && handTorchCollected && cheats.handTorchEnabled
				   && !cheats.handTorchModelCastsShadowWhenHeld) {
					continue;
				}
				ShadowCaster sc;
				sc.inst = &inst;
				if(cullPerFace) {
					const glm::vec4 &local = modelSphere(inst.Mid);
					sc.centre = glm::vec3(inst.Wm * glm::vec4(glm::vec3(local), 1.0f));
					// Largest column length, so a non-uniform scale doesn't miss the stretched axis.
					const float scale = std::max({glm::length(glm::vec3(inst.Wm[0])),
												  glm::length(glm::vec3(inst.Wm[1])),
												  glm::length(glm::vec3(inst.Wm[2]))});
					sc.radius = local.w * scale;
				} else {
					sc.centre = glm::vec3(0.0f);
					sc.radius = 0.0f;
				}
				shadowCasterScratch.push_back(sc);
			}
		}
		const glm::vec3 lightPos = torchLightPos[t];

		for(int face = 0; face < 6; face++) {
			if((faceMask & (1u << face)) == 0) {
				continue;
			}

			VkClearValue clearValues[2];
			clearValues[0].color = {{TORCH_SHADOW_FAR_CONST, 0.0f, 0.0f, 0.0f}};
			clearValues[1].depthStencil = {1.0f, 0};

			VkRenderPassBeginInfo rpInfo{};
			rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			rpInfo.renderPass = RPShadowCubeCompat.renderPass;
			rpInfo.framebuffer = torchCube[t].faceFramebuffers[face];
			rpInfo.renderArea.offset = {0, 0};
			rpInfo.renderArea.extent = {(uint32_t)SHADOW_MAP_RES, (uint32_t)SHADOW_MAP_RES};
			rpInfo.clearValueCount = 2;
			rpInfo.pClearValues = clearValues;
			vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			// Draw loop below is skipped for an unoccupied slot, but begin/end
			// still runs to clear and transition the image: the samplerCube
			// array requires every slot in SHADER_READ_ONLY_OPTIMAL even unsampled.
			if(slotOccupied) {
				PShadowCube.bind(commandBuffer);
				// Set 1: this torch's current matrices/position, mapped fresh
				// every frame (not a push constant, see ShadowCubeUniformBufferObject).
				DSshadowCube[t].bind(commandBuffer, PShadowCube, 1, currentImage);
				ShadowCubeFacePushConstant facePc{};
				facePc.face = face;
				vkCmdPushConstants(commandBuffer, PShadowCube.pipelineLayout,
								   VK_SHADER_STAGE_VERTEX_BIT, 0,
								   sizeof(ShadowCubeFacePushConstant), &facePc);

				for(const ShadowCaster &sc : shadowCasterScratch) {
					if(cullPerFace && !sphereInCubeFace(sc.centre - lightPos, sc.radius, face)) {
						continue;
					}
					Instance &inst = *sc.inst;
					inst.DS[0][1]->bind(commandBuffer, PShadowCube, 0, currentImage);
					SC.M[inst.Mid]->bind(commandBuffer);
					vkCmdDrawIndexed(commandBuffer,
									 static_cast<uint32_t>(SC.M[inst.Mid]->indices.size()), 1, 0, 0, 0);
				}
			}

			vkCmdEndRenderPass(commandBuffer);
		}
	}

	// Marks, in pendingFaceMask, every cube face a moving occluder has
	// invalidated: one that holds a mover that moved since last drawn, or held
	// one last capture and doesn't now. Range is each light's own reach vs.
	// the mover's bounding sphere. Called from updateUniformBuffer() after the
	// occupant diff, once GameLogic() and updateDynamicShadowSlots() have settled positions.
	void queueMoverCubeSlotRenders() {
		if(!moverListBuilt) {
			auto addMover = [&](Instance *inst, bool isDoor) {
				if(inst != nullptr && materials.forModel(inst->Mid).castsShadow) {
					movingOccluders.push_back(inst);
					movingOccluderIsDoor.push_back(isDoor);
				}
			};
			for(const Ghost &g : ghosts) {
				addMover(g.inst, false);
			}
			for(const Door &d : doors) {
				addMover(d.inst, true);
			}
			// Held torch: static on the floor until pickup jumps its Wm, which
			// erases the stale floor-shadow it left in whichever torch captured it.
			addMover(handTorchInst, false);
			// Zero matrix, not identity, so an authored-identity pose still gets its first capture.
			movingOccluderWm.assign(movingOccluders.size(), glm::mat4(0.0f));
			// First tick reads as "was moving" for everyone, so a door that is
			// somehow already at rest on frame one still gets its settle force.
			movingOccluderWasMoving.assign(movingOccluders.size(), true);
			moverListBuilt = true;
		}

		// Which faces of each slot hold a mover this frame, and which of those
		// hold one that has moved since that face was last drawn.
		std::array<uint8_t, NUM_SHADOW_CUBES> facesNow{};
		std::array<uint8_t, NUM_SHADOW_CUBES> facesStale{};

		for(size_t m = 0; m < movingOccluders.size(); m++) {
			const glm::mat4 &wm = movingOccluders[m]->Wm;

			// Held torch in hand jumps every frame but isn't drawn as an
			// occluder; skip it here, the pickup-frame jump is handled by "held a mover, now doesn't".
			if(movingOccluders[m] == handTorchInst && handTorchCollected
			   && cheats.handTorchEnabled && !cheats.handTorchModelCastsShadowWhenHeld) {
				movingOccluderWm[m] = wm;
				movingOccluderWasMoving[m] = false;
				continue;
			}

			const bool moved = (wm != movingOccluderWm[m]);
			// A door's settle frame: it was moving last frame and has just
			// stopped. That is the one frame worth paying for every face,
			// since it's the frame where the leaf's final resting silhouette
			// has to replace whatever the swing's per-face test left behind.
			const bool justSettled = movingOccluderIsDoor[m] && !moved && movingOccluderWasMoving[m];
			movingOccluderWasMoving[m] = moved;
			// World-space bounding sphere, so the mover's full extent (not just
			// its origin) is what's checked against a light's reach.
			const glm::vec4 &local = modelSphere(movingOccluders[m]->Mid);
			const glm::vec3 p = glm::vec3(wm * glm::vec4(glm::vec3(local), 1.0f));
			const float scale = std::max({glm::length(glm::vec3(wm[0])),
										  glm::length(glm::vec3(wm[1])),
										  glm::length(glm::vec3(wm[2]))});
			const float r = local.w * scale;

			// Held torch's slot is included too: a ghost walking into its beam has to mark it.
			for(int t = 0; t <= HAND_TORCH_SHADOW_INDEX; t++) {
				if(!cubeSlotOccupied(t)) {
					continue;
				}
				if(glm::distance(p, torchLightPos[t]) - r > torchShadowReach[t]) {
					continue;
				}
				const glm::vec3 rel = p - torchLightPos[t];
				uint8_t moverFaces = 0;
				for(int face = 0; face < 6; face++) {
					if(sphereInCubeFace(rel, r, face)) {
						moverFaces |= (uint8_t)(1u << face);
					}
				}
				facesNow[t] |= moverFaces;
				if(justSettled) {
					// The swing itself used the cheap per-face test above and
					// that is fine mid-swing (a face lagging by one frame for
					// a fraction of a second isn't visible); but that test
					// checks the CURRENT sphere, not the swept volume, so it
					// can leave a face holding the leaf's pre-swing silhouette
					// forever once the door stops. Pay for all six faces this
					// one settle frame to guarantee the final state is right.
					facesStale[t] |= ALL_CUBE_FACES;
				} else if(moved) {
					facesStale[t] |= moverFaces;
				}
			}
			movingOccluderWm[m] = wm;
		}

		for(int t = 0; t <= HAND_TORCH_SHADOW_INDEX; t++) {
			// Faces where something moved, plus faces a mover just left, ORed
			// into what the occupant diff already asked for.
			pendingFaceMask[t] |= facesStale[t] | (uint8_t)(slotFaceHadMover[t] & ~facesNow[t]);
			slotFaceHadMover[t] = facesNow[t];
		}
	}

	// Allocates the per-frame shadow command buffers, pool and fences.
	// Sized off swapChainImages since this runs again after every swapchain
	// recreation; matching destroy in pipelinesAndDescriptorSetsCleanup().
	void createShadowCommandBuffers() {
		destroyShadowCommandBuffers();

		QueueFamilyIndices qfi = findQueueFamilies(physicalDevice);
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = qfi.graphicsFamily.value();
		// IMPORTANT: without this bit the buffers below could be recorded exactly once each.
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		if(vkCreateCommandPool(device, &poolInfo, nullptr, &shadowCommandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create the shadow command pool!");
		}

		const uint32_t count = (uint32_t)swapChainImages.size();
		shadowCB.resize(count);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = shadowCommandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = count;
		if(vkAllocateCommandBuffers(device, &allocInfo, shadowCB.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate the shadow command buffers!");
		}

		// IMPORTANT: created signalled, or the first frame's wait on a fence nothing submitted yet hangs forever.
		shadowCBFence.resize(count);
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		for(uint32_t i = 0; i < count; i++) {
			if(vkCreateFence(device, &fenceInfo, nullptr, &shadowCBFence[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create a shadow command buffer fence!");
			}
		}
	}

	// Safe to call with nothing allocated. Both call sites run with the device
	// idle (recreateSwapChain() and mainLoop() before cleanup()).
	void destroyShadowCommandBuffers() {
		for(VkFence f : shadowCBFence) {
			vkDestroyFence(device, f, nullptr);
		}
		shadowCBFence.clear();
		// Destroying the pool frees every buffer allocated from it.
		if(shadowCommandPool != VK_NULL_HANDLE) {
			vkDestroyCommandPool(device, shadowCommandPool, nullptr);
			shadowCommandPool = VK_NULL_HANDLE;
		}
		shadowCB.clear();
	}

	// Records and submits this frame's cube shadow captures: the held torch
	// (never cacheable) plus every slot pendingFaceMask marks. Called at the
	// end of updateUniformBuffer(), after DSshadowCube[t] and every instance's
	// DS[0][1] have this frame's matrices.
	// IMPORTANT: its own submission, not part of "main" (recorded once and
	// replayed, so it can't express "these slots this frame"). Correct
	// because it's submitted before "main" on the same queue with a
	// barrier ordering its writes against later sampling (replacing a
	// mid-frame vkQueueWaitIdle), and each swapchain image has its own
	// buffer/fence, waited on before re-recording (returns immediately in the steady state).
	void submitCubeShadowCaptures(int currentImage) {
		if(shadowCB.empty()) {
			return;	// nothing allocated yet (first frames of a swapchain rebuild)
		}

		// Held torch's staleness, settled before recording so "any work" sees
		// it: all six faces if its light moved, otherwise only what a mover
		// marked. Occupancy is compared too, so toggling torch shadows off clears the map.
		const bool handOccupied = cubeSlotOccupied(HAND_TORCH_SHADOW_INDEX);
		if(HAND_TORCH_SHADOW_INDEX < activeCubeShadows) {
			const glm::vec3 &handPos = torchLightPos[HAND_TORCH_SHADOW_INDEX];
			if(handPos != lastHandTorchCapturePos || handOccupied != lastHandTorchCaptureOccupied) {
				pendingFaceMask[HAND_TORCH_SHADOW_INDEX] = ALL_CUBE_FACES;
				lastHandTorchCapturePos = handPos;
				lastHandTorchCaptureOccupied = handOccupied;
			}
		} else {
			pendingFaceMask[HAND_TORCH_SHADOW_INDEX] = 0;
		}

		// Nothing stale: no buffer, no submit, no fence traffic (steady state when player and ghosts are still).
		bool anyWork = false;
		for(int t = 0; t <= HAND_TORCH_SHADOW_INDEX && !anyWork; t++) {
			anyWork = (pendingFaceMask[t] != 0);
		}
		if(!anyWork) {
			return;
		}

		vkWaitForFences(device, 1, &shadowCBFence[currentImage], VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &shadowCBFence[currentImage]);

		VkCommandBuffer cb = shadowCB[currentImage];
		vkResetCommandBuffer(cb, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		if(vkBeginCommandBuffer(cb, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin the shadow command buffer!");
		}

		for(int t = 0; t <= HAND_TORCH_SHADOW_INDEX; t++) {
			if(pendingFaceMask[t] == 0) {
				continue;
			}
			// The instance a slot's flame is anchored to, excluded from its own
			// capture: it sits centimetres from the light and would fill
			// several faces with its silhouette.
			Instance *ownerInst = nullptr;
			if(t == HAND_TORCH_SHADOW_INDEX) {
				ownerInst = handTorchInst;
			} else if(t >= dynamicShadowSlotBase && dynamicSlotOccupant[t] != -1) {
				ownerInst = torchFlames[dynamicSlotOccupant[t]].inst;
			}
			recordCubeSlotFaces(cb, currentImage, t, cubeSlotOccupied(t), ownerInst, true,
								pendingFaceMask[t]);
		}

		// The dependency described above. A global memory barrier, not one per
		// cube: the layouts are already correct, only the writes'
		// availability/visibility is missing, which one barrier covers.
		VkMemoryBarrier memBarrier{};
		memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		memBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
								   | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cb,
							 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
								 | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
							 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

		if(vkEndCommandBuffer(cb) != VK_SUCCESS) {
			throw std::runtime_error("failed to record the shadow command buffer!");
		}

		// No semaphores: submission order plus the barrier above is the whole
		// synchronisation, and the fence is only for the re-recording above.
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cb;
		if(vkQueueSubmit(graphicsQueue, 1, &submitInfo, shadowCBFence[currentImage]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit the shadow command buffer!");
		}
	}

	// Builds every torch's CubeShadowMap (colour cube image + face views +
	// per-torch depth + the 36 face framebuffers) plus the shared sampler.
	// Called right after RPShadowCubeCompat.create(), since the framebuffers
	// are built against its .renderPass handle. Lives here, not in
	// CubeShadowMap.hpp, because createImage/createImageView/findDepthFormat
	// are protected members of BaseProject.
	void createCubeShadowMaps() {
		// Shared by every torch, no mipmaps (a shadow lookup always samples level 0).
		// IMPORTANT: NEAREST, not LINEAR -- these cubes store a distance, and
		// averaging four distances across a silhouette returns one that
		// belongs to neither surface, causing false shadowing/unshadowing that
		// used to be papered over with a 0.35 grazing bias (and let light leak
		// through closed doors). Nearest keeps the comparison honest.
		cubeShadowSampler.init(this, VK_FILTER_NEAREST, VK_FILTER_NEAREST,
								VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
								VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
								VK_SAMPLER_MIPMAP_MODE_LINEAR,
								VK_FALSE, 1.0f, 1.0f);

		const VkFormat colorFmt = VK_FORMAT_R32_SFLOAT;
		const VkFormat depthFmt = findDepthFormat();

		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			CubeShadowMap &c = torchCube[i];

			// CUBE_COMPATIBLE_BIT lets the cube view below treat its 6 layers as faces, not an array.
			createImage(SHADOW_MAP_RES, SHADOW_MAP_RES, 1, 6,
						VK_SAMPLE_COUNT_1_BIT, colorFmt, VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						c.colorImage, c.colorMemory);

			// The one view CookTorrance.frag's samplerCube reads.
			c.cubeView = createImageView(c.colorImage, colorFmt, VK_IMAGE_ASPECT_COLOR_BIT,
										  1, VK_IMAGE_VIEW_TYPE_CUBE, 6);

			// One 2D view per layer for the framebuffers: a cube view can't be
			// a render target, and createImageView fixes baseArrayLayer at 0.
			for(int face = 0; face < 6; face++) {
				VkImageViewCreateInfo faceInfo{};
				faceInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				faceInfo.image = c.colorImage;
				faceInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
				faceInfo.format = colorFmt;
				faceInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				faceInfo.subresourceRange.baseMipLevel = 0;
				faceInfo.subresourceRange.levelCount = 1;
				faceInfo.subresourceRange.baseArrayLayer = face;
				faceInfo.subresourceRange.layerCount = 1;
				VkResult result = vkCreateImageView(device, &faceInfo, nullptr, &c.faceViews[face]);
				if(result != VK_SUCCESS) {
					PrintVkError(result);
					throw std::runtime_error("failed to create cube shadow face view!");
				}
			}

			// One depth image per torch, reused across its 6 faces; never sampled, so plain 2D.
			createImage(SHADOW_MAP_RES, SHADOW_MAP_RES, 1, 1,
						VK_SAMPLE_COUNT_1_BIT, depthFmt, VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						c.depthImage, c.depthMemory);
			c.depthView = createImageView(c.depthImage, depthFmt, VK_IMAGE_ASPECT_DEPTH_BIT,
										   1, VK_IMAGE_VIEW_TYPE_2D, 1);

			// The 6 real framebuffers: colour face view + the shared depth
			// view, against RPShadowCubeCompat.renderPass.
			for(int face = 0; face < 6; face++) {
				VkImageView attachments[2] = {c.faceViews[face], c.depthView};

				VkFramebufferCreateInfo fbInfo{};
				fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				fbInfo.renderPass = RPShadowCubeCompat.renderPass;
				fbInfo.attachmentCount = 2;
				fbInfo.pAttachments = attachments;
				fbInfo.width = SHADOW_MAP_RES;
				fbInfo.height = SHADOW_MAP_RES;
				fbInfo.layers = 1;
				VkResult result = vkCreateFramebuffer(device, &fbInfo, nullptr, &c.faceFramebuffers[face]);
				if(result != VK_SUCCESS) {
					PrintVkError(result);
					throw std::runtime_error("failed to create cube shadow face framebuffer!");
				}
			}
		}
	}

	void destroyCubeShadowMaps() {
		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			CubeShadowMap &c = torchCube[i];
			for(int face = 0; face < 6; face++) {
				vkDestroyFramebuffer(device, c.faceFramebuffers[face], nullptr);
				vkDestroyImageView(device, c.faceViews[face], nullptr);
			}
			vkDestroyImageView(device, c.cubeView, nullptr);
			vkDestroyImage(device, c.colorImage, nullptr);
			vkFreeMemory(device, c.colorMemory, nullptr);
			vkDestroyImageView(device, c.depthView, nullptr);
			vkDestroyImage(device, c.depthImage, nullptr);
			vkFreeMemory(device, c.depthMemory, nullptr);
		}
		cubeShadowSampler.cleanup();
	}

	// =====================================================================
	// Pipelines, descriptor sets and cleanup
	// =====================================================================
	// Here you create your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsInit() {
		// All render passes first: RenderPass::create() allocates each
		// attachment's image/sampler, which the descriptor sets below point at.
		RP.create();
		RPbright.create();
		RPblurH.create();
		RPblurV.create();
		RPcomposite.create();

		P.create(&RP);
		// Ghosts draw inside the same scene pass, sharing its depth buffer and HDR attachment.
		Pspectral.create(&RP);
		PspectralDepth.create(&RP);
		Pbright.create(&RPbright);
		PblurH.create(&RPblurH);
		PblurV.create(&RPblurV);
		Pcomposite.create(&RPcomposite);

		DSglobal.init(this, &DSLglobal, {});
		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			DSshadowCube[i].init(this, &DSLshadowCubeCapture, {});
		}

		// Wire the chain: each pass reads the previous pass's colour
		// attachment as a texture. RP's sampled output is its resolve
		// attachment (the multisampled one is discarded).
		VkDescriptorImageInfo sceneTex = RP.attachments[RP.resolveAttIdx].getViewAndSampler();
		VkDescriptorImageInfo brightTex = RPbright.attachments[0].getViewAndSampler();
		VkDescriptorImageInfo blurHTex = RPblurH.attachments[0].getViewAndSampler();
		VkDescriptorImageInfo blurVTex = RPblurV.attachments[0].getViewAndSampler();

		DSbright.init(this, &DSLpost1, {sceneTex});
		DSblurH.init(this, &DSLpost1, {brightTex});
		DSblurV.init(this, &DSLpost1, {blurHTex});
		// Composite is the only pass needing two textures: full-brightness scene plus blurred bloom.
		DScomposite.init(this, &DSLpost2, {sceneTex, blurVTex});

		// Here you define the data set
		// If the scene has textures coming from a render pass, the corresponding element of the technique must be
		// updated before calling SC.pipelinesAndDescriptorSetsInit();

		SC.pipelinesAndDescriptorSetsInit();
		txt.pipelinesAndDescriptorSetsInit();
		uiQuad.pipelinesAndDescriptorSetsInit();
		crosshair.pipelinesAndDescriptorSetsInit();
		pauseQuad.pipelinesAndDescriptorSetsInit();
		startScreenQuad.pipelinesAndDescriptorSetsInit();
		settingsQuad.pipelinesAndDescriptorSetsInit();
		// Same RP as the scene: flame, exit glow and debug lines draw inside
		// it too, sharing the depth buffer and writing over-1.0 colours for bloom.
		flame.pipelinesAndDescriptorSetsInit(&RP);
		exitGlow.pipelinesAndDescriptorSetsInit(&RP);
		debugLines.pipelinesAndDescriptorSetsInit(&RP);

		createShadowCommandBuffers();	// one per swapchain image, rebuilt whenever the swapchain is
	}

	// Here you destroy your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsCleanup() {
		// Safe here: both call sites (recreateSwapChain(), cleanup()) wait for the device to be idle first.
		destroyShadowCommandBuffers();

		P.cleanup();
		Pspectral.cleanup();
		PspectralDepth.cleanup();
		Pbright.cleanup();
		PblurH.cleanup();
		PblurV.cleanup();
		Pcomposite.cleanup();

		RP.cleanup();
		RPbright.cleanup();
		RPblurH.cleanup();
		RPblurV.cleanup();
		RPcomposite.cleanup();

		DSglobal.cleanup();
		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			DSshadowCube[i].cleanup();
		}
		DSbright.cleanup();
		DSblurH.cleanup();
		DSblurV.cleanup();
		DScomposite.cleanup();

		SC.pipelinesAndDescriptorSetsCleanup();
		txt.pipelinesAndDescriptorSetsCleanup();
		uiQuad.pipelinesAndDescriptorSetsCleanup();
		crosshair.pipelinesAndDescriptorSetsCleanup();
		pauseQuad.pipelinesAndDescriptorSetsCleanup();
		startScreenQuad.pipelinesAndDescriptorSetsCleanup();
		settingsQuad.pipelinesAndDescriptorSetsCleanup();
		flame.pipelinesAndDescriptorSetsCleanup();
		exitGlow.pipelinesAndDescriptorSetsCleanup();
		debugLines.pipelinesAndDescriptorSetsCleanup();
	}

	// Here you destroy all the Models, Texture and Desc. Set Layouts you created!
	// You also have to destroy the pipelines
	void localCleanup() {
		DSLlocal.cleanup();
		DSLglobal.cleanup();
		DSLpost1.cleanup();
		DSLpost2.cleanup();
		DSLshadowSample.cleanup();
		DSLshadowCubeCapture.cleanup();

		if(Mpost != nullptr) {
			Mpost->cleanup();
		}

		P.destroy();
		Pspectral.destroy();
		PspectralDepth.destroy();
		Pbright.destroy();
		PblurH.destroy();
		PblurV.destroy();
		Pcomposite.destroy();

		// PShadowCube/RPShadowCubeCompat/torchCube[] never go through
		// pipelinesAndDescriptorSetsCleanup (see the member declaration for
		// why -- they don't depend on the swapchain, so a resize never tears
		// them down), which is where P/RP normally get their .cleanup() half.
		// Both halves have to happen somewhere, so both happen here instead.
		PShadowCube.cleanup();
		PShadowCube.destroy();
		destroyCubeShadowMaps();
		RPShadowCubeCompat.cleanup();
		RPShadowCubeCompat.destroy();

		RP.destroy();
		RPbright.destroy();
		RPblurH.destroy();
		RPblurV.destroy();
		RPcomposite.destroy();

		// Before SC.localCleanup(), which frees scene.json's colliders that colliderSet also points at.
		colliderSet.cleanup();
		allColliders.clear();

		SC.localCleanup();
		txt.localCleanup();
		uiQuad.localCleanup();
		crosshair.localCleanup();
		pauseQuad.localCleanup();
		startScreenQuad.localCleanup();
		settingsQuad.localCleanup();
		flame.localCleanup();
		exitGlow.localCleanup();
		debugLines.localCleanup();
	}

	// =====================================================================
	// populateCommandBuffer(): draw order
	// =====================================================================
	static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void *Params) {
		Castlescape *T = (Castlescape *)Params;
		T->populateCommandBuffer(commandBuffer, currentImage);
	}

	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
		// The whole HDR chain goes into this one command buffer, in order;
		// ordering between passes comes from their render pass dependencies.
		// The text and HUD passes are separate buffers submitted after, so
		// they draw on top of the composited frame.

		// IMPORTANT: no cube shadow pass here -- those are recorded/submitted
		// per frame by submitCubeShadowCaptures() instead, since this buffer
		// is recorded once per swapchain image and replayed unmodified, which
		// can't express "only the slots that went stale this frame" or the
		// per-face culling that makes re-captures affordable.
		//
		// 1. The scene, into the offscreen HDR target.
		RP.begin(commandBuffer, currentImage);
		// The dungeon only: the Spectral depth prepass was unhooked from this
		// walk in localInit() so the flames can slot in ahead of it.
		SC.populateCommandBuffer(commandBuffer, 0, currentImage);

		// Flames before the ghosts: they depth-test against the dungeon but
		// land before any ghost depth, so a flame poking into a ghost's body
		// isn't erased and stays visible right up until it snuffs.
		flame.populateCommandBuffer(commandBuffer, currentImage);

		// Ghost depth prepass, by hand: nearest ghost-surface depth only
		// (LESS, no colour), so the colour pass keeps just that layer.
		PspectralDepth.bind(commandBuffer);
		for(int i = 0; i < SC.TI[1].InstanceCount; i++) {
			Instance &inst = SC.TI[1].I[i];
			SC.M[inst.Mid]->bind(commandBuffer);
			for(int j = 0; j < inst.NDs[0]; j++) {
				inst.DS[0][j]->bind(commandBuffer, PspectralDepth, j, currentImage);
			}
			vkCmdDrawIndexed(commandBuffer,
							 static_cast<uint32_t>(SC.M[inst.Mid]->indices.size()), 1, 0, 0, 0);
		}

		// Ghosts' colour pass, by hand: same instances through Pspectral;
		// LESS_OR_EQUAL against the prepass keeps only the nearest layer per pixel.
		Pspectral.bind(commandBuffer);
		for(int i = 0; i < SC.TI[1].InstanceCount; i++) {
			Instance &inst = SC.TI[1].I[i];
			SC.M[inst.Mid]->bind(commandBuffer);
			for(int j = 0; j < inst.NDs[0]; j++) {
				inst.DS[0][j]->bind(commandBuffer, Pspectral, j, currentImage);
			}
			vkCmdDrawIndexed(commandBuffer,
							 static_cast<uint32_t>(SC.M[inst.Mid]->indices.size()), 1, 0, 0, 0);
		}

		// After the scene (so castle depth masks this quad to the arch) and the flames.
		exitGlow.populateCommandBuffer(commandBuffer, currentImage);
		debugLines.populateCommandBuffer(commandBuffer, currentImage);
		RP.end(commandBuffer);

		// 2-5. Four full-screen quads: threshold, blur H, blur V, then mix the
		// result back over the scene and tone map.
		auto fullScreenPass = [&](RenderPass &pass, Pipeline &pipe, DescriptorSet &ds) {
			pass.begin(commandBuffer, currentImage);
			pipe.bind(commandBuffer);
			Mpost->bind(commandBuffer);
			ds.bind(commandBuffer, pipe, 0, currentImage);
			vkCmdDrawIndexed(commandBuffer, (uint32_t)Mpost->indices.size(), 1, 0, 0, 0);
			pass.end(commandBuffer);
		};

		fullScreenPass(RPbright, Pbright, DSbright);
		fullScreenPass(RPblurH, PblurH, DSblurH);
		fullScreenPass(RPblurV, PblurV, DSblurV);
		fullScreenPass(RPcomposite, Pcomposite, DScomposite);
	}

	// =====================================================================
	// updateUniformBuffer(): per-frame uniforms and HUD
	// =====================================================================
	// Here is where you update the uniforms.
	// Very likely this will be where you will be writing the logic of your application.
	// Everything the GPU reads this frame, in dependency order: clock, then the
	// global UBO (lights), then the shadow slots and billboard bases that the
	// fire simulation needs, then the post chain, then one local UBO per
	// instance, and finally the shadow capture submission and the HUD text.
	void updateUniformBuffer(uint32_t currentImage) {
		// ---- Overlays and the frame clock ----
		static bool debounce = false;
		static int curDebounce = 0;

		// ESC opens/closes the pause menu, edge-triggered. Ignored while
		// another modal is open (settingsMenu closes PauseMenu, so ESC would
		// otherwise reopen it underneath).
		bool escPressed = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
		if(escPressed && !escKeyWasPressed && !hud.isOpen() && !startScreen.isOpen()
		   && !settingsMenu.isOpen()) {
			pauseMenu.setOpen(!pauseMenu.isOpen(), windowWidth, windowHeight);
		}
		escKeyWasPressed = escPressed;

		// moves the view
		float deltaT = GameLogic();

		// Zeroing deltaT freezes every deltaT-driven animation (flicker,
		// shadow reassignment, ...) while a modal is open. The cheat HUD does
		// not zero it -- torches flickering while flipping a debug flag is fine.
		if(pauseMenu.isOpen() || startScreen.isOpen() || settingsMenu.isOpen()) {
			deltaT = 0.0f;
		}

		// Free-running clock for shader-side animation; unlike elapsedT below this never resets.
		static float simTime = 0.0f;
		simTime += deltaT;

		// ---- Global UBO: the lights from lights.json ----
		// defines the global parameters for the uniform
		GlobalUniformBufferObject gubo{};

		// Every light comes from lights.json. No intensity factor: with a BRDF
		// returning [0,1] a white source is (1,1,1); strength is g and beta instead.
		const std::vector<LightData> &lights = sceneLights.update(deltaT);
		gubo.lightCount = (int)lights.size();
		for(int i = 0; i < gubo.lightCount; i++) {
			gubo.lights[i] = lights[i];
		}

		// Camera world position/look direction, used to cull torch lights and
		// orient flame billboards. `forward` biases the light/shadow pool toward what's in view.
		const glm::mat4 camToWorld = glm::inverse(View);
		const glm::vec3 eyePos = glm::vec3(camToWorld[3]);
		const glm::vec3 forward = -glm::vec3(camToWorld[2]);

		// ---- Cube-shadow slot bookkeeping ----
		// Re-decides the dynamic cube-shadow pool's slots, at most every
		// SHADOW_REASSIGN_INTERVAL, before the light-append loop and
		// DSshadowCube mapping loop read the (possibly changed) slots.
		shadowReassignTimer += deltaT;
		if(shadowReassignTimer >= SHADOW_REASSIGN_INTERVAL) {
			shadowReassignTimer = 0.0f;
			updateDynamicShadowSlots(eyePos, forward);
		}

		// Diffs every static cube slot's current occupant against the one it
		// last rendered for, queuing changes into pendingFaceMask.
		// HAND_TORCH_SHADOW_INDEX is excluded (rendered fresh every frame, never cached).
		for(int t = 0; t < dynamicShadowSlotBase; t++) {
			// A fixed lights.json slot's identity never changes, so this fires once, on frame one.
			if(lastRenderedOccupant[t] != t) {
				pendingFaceMask[t] = ALL_CUBE_FACES;
				lastRenderedOccupant[t] = t;
			}
		}
		for(int t = dynamicShadowSlotBase; t < HAND_TORCH_SHADOW_INDEX; t++) {
			if(lastRenderedOccupant[t] != dynamicSlotOccupant[t]) {
				pendingFaceMask[t] = ALL_CUBE_FACES;
				lastRenderedOccupant[t] = dynamicSlotOccupant[t];
			}
		}

		// The diff above only sees a slot whose light changed; this adds slots
		// whose light stayed put while a ghost/door moved inside it.
		queueMoverCubeSlotRenders();

		// ---- Billboard bases for the flames ----
		// IMPORTANT: cylindrical billboard basis from the camera's right axis,
		// not each flame's eye->anchor direction. The per-flame version breaks
		// for the held torch: its anchor is barely a unit away in camera
		// space, so pitching swings it around the eye and the derived yaw
		// whips through 180 (visible spin). The camera's right axis stays
		// horizontal at every pitch and turns only with yaw.
		glm::vec3 bbRight = glm::vec3(camToWorld[0]);
		bbRight.y = 0.0f;
		if(glm::dot(bbRight, bbRight) > 1e-8f) {
			bbRight = glm::normalize(bbRight);
		} else {
			// Unreachable while pitch is clamped, but a zero-length basis
			// would collapse the billboard to a line.
			bbRight = glm::vec3(1.0f, 0.0f, 0.0f);
		}
		const glm::vec3 bbUp = glm::vec3(0.0f, 1.0f, 0.0f);
		const glm::vec3 bbFwd = glm::cross(bbRight, bbUp);	// points back toward the eye

		// Second basis for the held torch only: the camera's actual
		// up/right/forward (pitch included), since the torch is welded to the
		// camera and a world-vertical flame would swing out of alignment as it tilts.
		const glm::vec3 handBbRight = glm::normalize(glm::vec3(camToWorld[0]));
		const glm::vec3 handBbUp    = glm::normalize(glm::vec3(camToWorld[1]));
		const glm::vec3 handBbFwd   = glm::normalize(glm::vec3(camToWorld[2]));

		animTime += deltaT;

		// ---- Fire simulation and the torch/candle lights ----
		// Advance every torch's fire state before anything reads it, so flame,
		// sparks and light are all driven by the same envelope this frame.
		for(TorchFlame &tf : torchFlames) {
			// Fast term (a minor share -- Flame.frag's per-pixel shimmer owns
			// the fast twitch), slow term for breathing over seconds, middle term to bridge them.
			float t = animTime + tf.phase;
			float fast   = fireFbm(t * FLAME_FLICKER_HZ);
			float middle = fireFbm(t * 3.1f + 7.0f);
			float slow   = fireFbm(t * 0.6f + 91.0f);
			float n = 0.25f * fast + 0.30f * middle + 0.45f * slow;	// [0,1]

			// Guttering, applied on top: a separate slow noise crossing a high
			// threshold, so a flame occasionally ducks hard and recovers
			// rather than dipping on any regular beat.
			float gut = fireFbm(t * FLAME_GUTTER_SPEED + 311.0f);
			float gutter = glm::smoothstep(FLAME_GUTTER_LO, FLAME_GUTTER_HI, gut);

			// This is the TARGET, not the value: assigning it raw was the old
			// whole-flame "jump" every time the noise stepped.
			float target = glm::clamp((0.78f + 0.50f * n) * (1.0f - FLAME_GUTTER_DEPTH * gutter),
									  0.30f, 1.40f);

			// Critically-damped spring toward the target: continuous in value
			// AND slope, so brightness never steps frame-to-frame, yet a
			// gutter still ducks in ~0.2 s (see FLAME_BRIGHT_OMEGA).
			float w0 = FLAME_BRIGHT_OMEGA;
			tf.intensityVel += ((target - tf.intensity) * w0 * w0
								- 2.0f * w0 * tf.intensityVel) * deltaT;
			tf.intensity += tf.intensityVel * deltaT;

			// Hunt cycle's colour, recomputed from the authored base every
			// frame: `color` feeds both light and billboard, so tinting it
			// here turns torch, light, shadow and bloom violet together.
			// IMPORTANT: rides this value, not `intensity` (spring-driven) --
			// a 4Hz pulse written into a ~0.2s spring would be damped away.
			tf.color = huntCycle.flameColor(tf.baseColor)
					   * huntCycle.warningPulse() * huntCycle.lightScale();

			// Slow hue wander on top of the brightness spring, so a guttering
			// flame shifts warm/cool rather than only dimming. Scaled by
			// `gutter` so a steady flame and the hunt-cycle tint stay clean.
			float hueWander = fireFbm(t * 0.9f + 53.0f) - 0.5f;
			tf.color.r *= 1.0f + hueWander * 0.10f * gutter;
			tf.color.b *= 1.0f - hueWander * 0.12f * gutter;

			float hTarget = 0.78f + 0.31f * (target - 0.30f) / 1.10f;	// ~0.78..1.09
			tf.heightScale += (hTarget - tf.heightScale)
							  * (1.0f - std::exp(-deltaT / FLAME_HEIGHT_TAU));

			// Catching fire / going out: spring toward 1 while lit, 0 while
			// not. Advanced unconditionally (not gated on flameBurning) so a
			// just-switched-off flame relaxes to 0 instead of freezing.
			// IMPORTANT: underdamped ZETA only on the way up (the flare past
			// resting size reads as "catching"); on the way down it would ring
			// past 0, bounce, and sink again -- visible as off/on/off when a
			// ghost snuffs the held torch. Critically damp the extinguish instead.
			float wi = FLAME_IGNITION_OMEGA;
			bool igniting = flameBurning(tf);
			float ignitionTarget = igniting ? 1.0f : 0.0f;
			float zi = igniting ? FLAME_IGNITION_ZETA : 1.0f;
			tf.ignitionVel += ((ignitionTarget - tf.ignitionScale) * wi * wi
								- 2.0f * zi * wi * tf.ignitionVel) * deltaT;
			tf.ignitionScale += tf.ignitionVel * deltaT;
			// Overshoot is the point (the catching flare), but must not go
			// negative or a negative billboard size flips the quads inside out for a frame.
			tf.ignitionScale = std::max(tf.ignitionScale, 0.0f);

			// A flame leans against its own velocity (dragged by the air it
			// moves through); horizontal only, so a vertical lift doesn't bend it sideways.
			glm::vec3 pos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			// IMPORTANT: a switched-off flame drops its velocity history rather
			// than differencing against it -- the held torch is parked 1000
			// units below the map while off, which would otherwise read as an
			// enormous velocity on both the off and on frames.
			if(!flameBurning(tf)) {
				tf.velPrimed = false;
				tf.smoothedVel = glm::vec3(0.0f);
				tf.lean = glm::vec2(0.0f);
				continue;
			}
			if(!tf.velPrimed) {
				// First frame: no previous position, and the instance may
				// still sit at its scene.json placeholder, so skip the jump.
				tf.prevPos = pos;
				tf.velPrimed = true;
			}
			glm::vec3 vel = (pos - tf.prevPos) / std::max(deltaT, 1e-4f);
			tf.prevPos = pos;

			// Framerate-independent exponential smoothing, or the flame would
			// twitch on every single-frame jolt in the walking bob.
			float a = 1.0f - std::exp(-deltaT / TORCH_LEAN_TAU);
			tf.smoothedVel += (vel - tf.smoothedVel) * a;

			glm::vec3 leanWorld = -tf.smoothedVel;	// trails behind the hand carrying it
			leanWorld.y = 0.0f;

			// IMPORTANT: resolved into the same basis this torch's quads are
			// actually built with below (not a second per-flame copy), or the
			// lean would point in the wrong direction relative to the billboard.
			const glm::vec3 &leanRight = tf.heldByCamera ? handBbRight : bbRight;
			const glm::vec3 &leanFwd   = tf.heldByCamera ? handBbFwd   : bbFwd;

			// IMPORTANT: deliberately not divided by the flame's world
			// half-width -- the held torch's is ~0.07 units, so dividing would
			// multiply velocity ~14x and saturate the lean permanently from
			// just turning on the spot. It would also make the small held
			// torch react harder than a full-size wall one.
			tf.lean = glm::vec2(glm::dot(leanWorld, leanRight), glm::dot(leanWorld, leanFwd))
					  * TORCH_LEAN_PER_SPEED;
			if(glm::length(tf.lean) > TORCH_LEAN_MAX) {
				tf.lean = glm::normalize(tf.lean) * TORCH_LEAN_MAX;
			}
		}

		// Torch flames' point lights, appended straight into gubo every frame
		// from the flame's live Wm -- not through lights.json's
		// instance+offset anchoring, which reads Wm once at init and would
		// freeze the held torch's light at the origin.
		// Culled by real distance (never by facing) and capped in count, in
		// facingBiasedDistSq() order; TORCH_LIGHT_MAX_LIVE is above the
		// scene's flame count so the cap doesn't bite in practice.
		{
			std::vector<std::pair<float, const TorchFlame *>> nearest;
			nearest.reserve(torchFlames.size());
			const float cullSq = TORCH_LIGHT_CULL_DIST * TORCH_LIGHT_CULL_DIST;
			for(const TorchFlame &tf : torchFlames) {
				if(!flameBurning(tf)) {	// switched-off flames never enter the contest at all
					continue;
				}
				glm::vec3 worldPos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
				glm::vec3 d = worldPos - eyePos;
				float dSq = glm::dot(d, d);
				if(dSq > cullSq) {
					continue;
				}
				// View-cone cull: drop a torch whose whole lit sphere sits
				// outside the frame before it costs a live slot or BRDF eval.
				// Held torch rides the camera, so it always reaches the view.
				if(!tf.heldByCamera
				   && !lightReachesViewCone(worldPos, flameLightCullReach(tf), eyePos, forward)) {
					continue;
				}
				nearest.push_back({facingBiasedDistSq(worldPos, eyePos, forward), &tf});
			}
			std::sort(nearest.begin(), nearest.end(),
					  [](const auto &a, const auto &b) { return a.first < b.first; });

			int live = 0;
			for(const auto &entry : nearest) {
				if(gubo.lightCount >= MAX_LIGHTS || live >= TORCH_LIGHT_MAX_LIVE) {
					break;
				}
				const TorchFlame &tf = *entry.second;

				LightData L{};
				L.pos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
				L.dir = glm::vec3(0.0f, -1.0f, 0.0f);	// unused for a point light
				// Same envelope the flame is drawn with: brightness on colour,
				// reach on g, and ignitionScale so light brightens with a catching flame.
				L.color = tf.color * tf.intensity * tf.lightScale * tf.ignitionScale;
				L.g = flameLightG(tf.isCandle, tf.intensity);
				L.beta = TORCH_LIGHT_BETA;
				L.cosIn = 1.0f;
				L.cosOut = 0.0f;
				L.type = LIGHT_POINT;
				// IMPORTANT: explicit -1 for a non-candidate matters -- slot 0
				// would otherwise read as the sun's shadow map indoors.
				if(tf.heldByCamera) {
					if(cheats.torchShadowsEnabled) {
						L.shadowIndex = HAND_TORCH_SHADOW_INDEX;
						updateHandTorchShadow(L.pos);
					} else {
						L.shadowIndex = -1;
					}
				} else if(tf.shadowCandidate) {
					L.shadowIndex = tf.shadowSlot;
				} else {
					L.shadowIndex = -1;
				}

				gubo.lights[gubo.lightCount++] = L;
				live++;
			}
		}

		// ---- Exit spill, fog and the debug flags ----
		// Light the open exit throws back into the room, tracking the leaf's
		// swing: the counterpart to the ExitGlow quad (which is the daylight
		// you look at; this is the daylight on the stone). No shadow map: the
		// wall already shapes it, and a moving-door occluder would re-render every frame.
		if(exitOpenFrac > 0.001f && gubo.lightCount < MAX_LIGHTS) {
			LightData L{};
			L.pos = EXIT_SPILL_POS;
			L.dir = glm::vec3(-1.0f, 0.0f, 0.0f);	// aimed due west, same axis the glow quad faces
			L.color = EXIT_SPILL_COLOR * exitOpenFrac;
			L.g = EXIT_SPILL_G;
			L.beta = EXIT_SPILL_BETA;
			L.cosIn = std::cos(glm::radians(EXIT_SPILL_INNER_DEG * 0.5f));
			L.cosOut = std::cos(glm::radians(EXIT_SPILL_OUTER_DEG * 0.5f));
			L.type = LIGHT_SPOT;
			L.shadowIndex = -1;
			gubo.lights[gubo.lightCount++] = L;
		}

		// Always as authored -- the Torch Bounce cheat zeroes the term in the
		// shader instead (LIGHT_DEBUG_NO_BOUNCE below). amb.upper/lower/dir
		// are no longer uploaded: nothing in CookTorrance.frag reads them
		// since metalAmbient() was rewired to reflect bounce instead of an
		// authored environment (see notes.md). Left in AmbientLight/
		// lights.json rather than deleted outright -- SceneLights.hpp's
		// comment on the struct explains why (a future exterior level).
		const AmbientLight amb = sceneLights.ambient();
		gubo.ambientWeight = amb.weight;
		gubo.ambientBounce = amb.bounce;

		// Distance fog density, derived from GEOM_CULL_CONE_DIST so the two
		// can't drift: solved so fog reaches 1% brightness somewhat past where
		// the cull stops drawing. FOG_REFERENCE_DIST_SCALE is the knob (2.5
		// keeps the foreground unfogged but lets a little pop peek through the cull edge).
		constexpr float FOG_RESIDUAL_AT_CULL_DIST = 0.01f;
		constexpr float FOG_REFERENCE_DIST_SCALE = 2.5f;
		gubo.fogDensity = std::sqrt(-std::log(FOG_RESIDUAL_AT_CULL_DIST))
						/ (GEOM_CULL_CONE_DIST * FOG_REFERENCE_DIST_SCALE);

		// The lighting debug cheats, packed into the one int the shader reads.
		// Note the two inversions: the cheat says what the frame should still
		// have, the flag says what the shader should drop.
		gubo.debugFlags = 0;
		if(cheats.unlit)            gubo.debugFlags |= LIGHT_DEBUG_UNLIT;
		if(cheats.showNormals)      gubo.debugFlags |= LIGHT_DEBUG_NORMALS;
		if(!cheats.specularEnabled) gubo.debugFlags |= LIGHT_DEBUG_NO_SPECULAR;
		if(!cheats.toneMapEnabled)  gubo.debugFlags |= LIGHT_DEBUG_NO_TONEMAP;
		if(!cheats.shadowsEnabled)  gubo.debugFlags |= LIGHT_DEBUG_NO_SHADOWS;
		if(cheats.showLightHeatmap) gubo.debugFlags |= LIGHT_DEBUG_HEATMAP;
		if(cheats.showShadowGap)    gubo.debugFlags |= LIGHT_DEBUG_SHADOW_GAP;
		if(!sceneLights.bounceEnabled)  gubo.debugFlags |= LIGHT_DEBUG_NO_BOUNCE;

		// Both computed further up, before the torch fire state that needs them.
		gubo.eyePos = eyePos;
		gubo.time = animTime;

		DSglobal.map(currentImage, &gubo, 0);

		// ---- Glare, exposure and the post-processing passes ----
		// Stare-at glare (GLARE_*): how squarely and closely the camera looks
		// at each WALL torch. The held torch is excluded -- it's near screen
		// centre by construction, so its glare would be a permanent bias.
		// Purely geometric, so raising the exposure below can't feed on itself.
		{
			// camToWorld column 2 points from the scene back to the eye, so
			// the look direction is its negation.
			const glm::vec3 camFwd = -handBbFwd;
			float glareTarget = 0.0f;
			for(TorchFlame &tf : torchFlames) {
				float tfTarget = 0.0f;
				// An extinguished torch doesn't dazzle: without flameBurning()
				// here the exposure/bloom would still ramp up when the player
				// stares at a torch that is no longer drawn at all.
				if(!tf.heldByCamera && flameBurning(tf)) {
					glm::vec3 fpos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
					glm::vec3 to = fpos - eyePos;
					float dist = glm::length(to);
					if(dist > 1e-4f && dist < GLARE_DIST_FAR) {
						// Facing band: a smoothstep over the view-to-flame
						// cosine, so glare ramps in as the flame approaches
						// screen centre instead of switching.
						float facing = glm::dot(camFwd, to / dist);
						float f = glm::smoothstep(GLARE_COS_MIN, GLARE_COS_MAX, facing);
						// Distance window: nothing right at the anchor (the
						// flame fills the view anyway there, and the numbers
						// go degenerate), full inside a few units, fading
						// out entirely by GLARE_DIST_FAR.
						float d = glm::smoothstep(GLARE_DIST_NEAR, GLARE_DIST_NEAR + 0.8f, dist)
								* (1.0f - glm::smoothstep(GLARE_DIST_FAR * 0.5f, GLARE_DIST_FAR, dist));
						// A guttering flame dazzles less: ride the same
						// envelope everything else does.
						tfTarget = f * d * glm::clamp(tf.intensity, 0.0f, 1.2f);
					}
				}
				// Per-flame smoothed copy, feeding that flame's own HDR
				// boost (fubo.glareBoost) further down.
				float tauF = tfTarget > tf.glare ? GLARE_TAU_RISE : GLARE_TAU_FALL;
				tf.glare += (tfTarget - tf.glare) * (1.0f - std::exp(-deltaT / tauF));
				glareTarget = std::max(glareTarget, tfTarget);
			}
			// max(), not sum: two torches lined up in view should read as one
			// dazzle, not stack into double exposure.
			float tau = glareTarget > glareSmoothed ? GLARE_TAU_RISE : GLARE_TAU_FALL;
			glareSmoothed += (glareTarget - glareSmoothed) * (1.0f - std::exp(-deltaT / tau));
		}

		// The four post-processing passes. Each one's texelSize is that of the texture it reads, not writes.
		{
			// RP.width/height, not swapChainExtent: the scene's resolve target
			// is sized to renderScale's scaled-down resolution.
			const glm::vec2 fullTexel = glm::vec2(1.0f / (float)RP.width,
												  1.0f / (float)RP.height);
			const glm::vec2 bloomTexel = glm::vec2(1.0f / (float)bloomWidth(),
												   1.0f / (float)bloomHeight());

			PostUniformBufferObject post{};
			post.time = animTime;
			post.threshold = BLOOM_THRESHOLD;
			post.knee = BLOOM_KNEE;
			// Stare-at glare rides both bloom and exposure; smoothed upstream so this never pumps.
			post.bloomIntensity = BLOOM_INTENSITY * (1.0f + GLARE_BLOOM_GAIN * glareSmoothed);
			post.exposure = SCENE_EXPOSURE * (1.0f + GLARE_EXPOSURE_GAIN * glareSmoothed);
			// Escape whiteout multiplies the same two knobs, cubed so the ramp
			// weights toward the end rather than fading evenly.
			if(escapeFlash > 0.0f) {
				const float f = escapeFlash * escapeFlash * escapeFlash;
				post.exposure *= 1.0f + ESCAPE_EXPOSURE_GAIN * f;
				post.bloomIntensity *= 1.0f + ESCAPE_BLOOM_GAIN * f;
			}
			// Held back until the exposure ramp has had most of the flash, so
			// the frame is already blowing out when the white arrives.
			post.escapeFlash = glm::smoothstep(0.45f, 1.0f, escapeFlash);

			// Spectral veil's ramp: here, not in the ghost loop (which runs on
			// the game clock and would freeze while paused). max() over ghosts,
			// reads the drawn matrix so the bob counts.
			{
				float veil = 0.0f;
				for(const Ghost &g : ghosts) {
					if(g.inst == nullptr) continue;
					const glm::vec3 gp = glm::vec3(g.inst->Wm[3]);

					// Vertical first: rejects most ghosts cheaply before the horizontal square root.
					const float dy = eyePos.y - gp.y;
					const float below = ghostBodyBottom - SPECTRAL_VEIL_FADE_Y;
					const float above = ghostBodyTop + SPECTRAL_VEIL_FADE_Y;
					if(dy <= below || dy >= above) continue;
					const float vy = glm::smoothstep(below, ghostBodyBottom, dy) *
									 (1.0f - glm::smoothstep(ghostBodyTop, above, dy));

					const float dx = eyePos.x - gp.x;
					const float dz = eyePos.z - gp.z;
					const float horiz = std::sqrt(dx * dx + dz * dz);
					const float vxz = 1.0f - glm::smoothstep(SPECTRAL_VEIL_INNER,
															 SPECTRAL_VEIL_OUTER, horiz);

					veil = std::max(veil, vxz * vy);
				}
				post.spectralVeil = veil;
			}

			post.debugFlags = gubo.debugFlags;

			// Bright pass: reads the full-resolution scene and writes the
			// quarter-res bloom target, so it steps in full-res texels.
			post.texelSize = fullTexel;
			post.blurDir = glm::vec2(0.0f);
			DSbright.map(currentImage, &post, 0);

			// The two blur directions. Same shader, same kernel; the only
			// difference between them is this vector, which is why there is one
			// BloomBlur.frag rather than two nearly identical ones.
			post.texelSize = bloomTexel;
			post.blurDir = glm::vec2(1.0f, 0.0f);
			DSblurH.map(currentImage, &post, 0);

			post.blurDir = glm::vec2(0.0f, 1.0f);
			DSblurV.map(currentImage, &post, 0);

			// Composite: samples with plain UVs, texel size unread. Tone map
			// happens here, hence debugFlags has to reach this pass.
			post.texelSize = fullTexel;
			post.blurDir = glm::vec2(0.0f);
			DScomposite.map(currentImage, &post, 0);
		}

		// ---- Flame and daylight quad matrices ----
		// Each flame's render matrix is one of two billboard bases translated
		// and scaled to its anchor: cylindrical for wall torches (upright at
		// any pitch), camera-locked for the held one. Shader local space: x
		// +/-1 across half-width, y 0 wick to 1 tip, z toward camera.
		for(const TorchFlame &tf : torchFlames) {
			glm::vec3 anchorWorld = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			float instScale = glm::length(glm::vec3(tf.inst->Wm[0]));	// uniform scale

			// An off/unlit flame collapses to a point rather than being
			// skipped (its descriptor set is recorded once and replayed).
			// ignitionScale already carries the on/off state, so no separate flameBurning() gate.
			float sizeScale = tf.sizeScale * tf.ignitionScale;
			float halfWidth = FLAME_HALF_WIDTH * instScale * sizeScale;
			float height = FLAME_HEIGHT * instScale * sizeScale;

			const glm::vec3 &right = tf.heldByCamera ? handBbRight : bbRight;
			const glm::vec3 &up    = tf.heldByCamera ? handBbUp    : bbUp;
			const glm::vec3 &fwd   = tf.heldByCamera ? handBbFwd   : bbFwd;

			glm::mat4 billboard = glm::mat4(
				glm::vec4(right * halfWidth, 0.0f),
				glm::vec4(up * height, 0.0f),
				glm::vec4(fwd * halfWidth, 0.0f),
				glm::vec4(anchorWorld, 1.0f)
			);

			flame.update(tf.flameId, ViewPrj * billboard, tf.intensity, tf.heightScale,
						 tf.lean, 1.0f + GLARE_FLAME_GAIN * tf.glare, tf.color, currentImage);
		}

		// Daylight outside the exit door: fixed planes, not billboards, so
		// their bases are world axes. Same column convention as Flame's
		// billboard, so one mesh/pipeline serves both the upright and flat quads.
		{
			// Squared so the light builds late in the swing: barely ajar shows a crack, not half the glare.
			const float glow = EXIT_GLOW_INTENSITY * exitOpenFrac * exitOpenFrac;

			// Wall of light across the doorway; in-plane axes world Z (across), world Y (up).
			const glm::mat4 uprightBasis = glm::mat4(
				glm::vec4(glm::vec3(0.0f, 0.0f, 1.0f) * EXIT_GLOW_HALF_WIDTH, 0.0f),
				glm::vec4(glm::vec3(0.0f, 1.0f, 0.0f) * EXIT_GLOW_HALF_HEIGHT, 0.0f),
				glm::vec4(EXIT_GLOW_NORMAL, 0.0f),
				glm::vec4(EXIT_GLOW_CENTER, 1.0f)
			);
			exitGlow.update(EXIT_GLOW_UPRIGHT, ViewPrj * uprightBasis,
							EXIT_GLOW_COLOR, glow, animTime, currentImage);

			// Ground quad, covering the strip left visible under the arch;
			// in-plane axes world X (out from threshold), world Z (across).
			const glm::mat4 floorBasis = glm::mat4(
				glm::vec4(glm::vec3(1.0f, 0.0f, 0.0f) * EXIT_GLOW_FLOOR_HALF_X, 0.0f),
				glm::vec4(glm::vec3(0.0f, 0.0f, 1.0f) * EXIT_GLOW_FLOOR_HALF_Z, 0.0f),
				glm::vec4(EXIT_GLOW_FLOOR_NORMAL, 0.0f),
				glm::vec4(EXIT_GLOW_FLOOR_CENTER, 1.0f)
			);
			exitGlow.update(EXIT_GLOW_FLOOR, ViewPrj * floorBasis,
							EXIT_GLOW_COLOR, glow, animTime, currentImage);

			// Lid: same plane as the floor quad, lifted above the arch and
			// facing down, so looking up finds daylight instead of the skybox.
			const glm::mat4 ceilingBasis = glm::mat4(
				glm::vec4(glm::vec3(1.0f, 0.0f, 0.0f) * EXIT_GLOW_CEILING_HALF_X, 0.0f),
				glm::vec4(glm::vec3(0.0f, 0.0f, 1.0f) * EXIT_GLOW_CEILING_HALF_Z, 0.0f),
				glm::vec4(EXIT_GLOW_CEILING_NORMAL, 0.0f),
				glm::vec4(EXIT_GLOW_CEILING_CENTER, 1.0f)
			);
			exitGlow.update(EXIT_GLOW_CEILING, ViewPrj * ceilingBasis,
							EXIT_GLOW_COLOR, glow, animTime, currentImage);
		}

		// ---- Per-instance local UBOs ----
		// defines the local parameters for the uniforms
		UniformBufferObject ubo{};

		// DSshadowCube[t] feeds the shadow capture pass via a mapped uniform
		// buffer rather than a push constant, so the held torch's slot
		// (refreshed above by updateHandTorchShadow()) takes effect every
		// frame. Static torches don't strictly need the re-map, but mapping
		// all of them uniformly is simpler and costs nothing worth avoiding.
		for(int t = 0; t < activeCubeShadows; t++) {
			ShadowCubeUniformBufferObject cubeUbo{};
			for(int face = 0; face < 6; face++) {
				cubeUbo.lightViewProj[face] = torchFaceMatrices[t][face];
			}
			cubeUbo.lightPos = glm::vec4(torchLightPos[t], 0.0f);
			DSshadowCube[t].map(currentImage, &cubeUbo, 0);
		}

		// Debug overlay (DebugLines.hpp, cheat-gated), placed here so gubo.lights[]/torchLightPos[] are current.
		std::vector<glm::vec4> dbgPos, dbgColor;
		if(cheats.showLightGizmos) {
			for(int i = 0; i < gubo.lightCount; i++) {
				const LightData &L = gubo.lights[i];
				glm::vec4 color = glm::vec4(L.color, 1.0f);
				if(L.type == LIGHT_DIRECT) {
					// The sun has no position, so its gizmo is an arrow near the player, not a cross.
					glm::vec3 anchor = eyePos + glm::vec3(0.0f, 2.0f, 0.0f);
					glm::vec3 tip = anchor + L.dir * 3.0f;
					DebugLines::PushLine(anchor, tip, color, dbgPos, dbgColor);
					glm::vec3 upHint = (std::abs(L.dir.y) > 0.99f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
					glm::vec3 side = glm::normalize(glm::cross(L.dir, upHint)) * 0.3f;
					glm::vec3 back = -L.dir * 0.3f;
					DebugLines::PushLine(tip, tip + back + side, color, dbgPos, dbgColor);
					DebugLines::PushLine(tip, tip + back - side, color, dbgPos, dbgColor);
				} else {
					DebugLines::PushCross(L.pos, 0.3f, color, dbgPos, dbgColor);
				}
			}
		}
		if(cheats.showShadowFrustums) {
			for(int t = 0; t < activeCubeShadows; t++) {
				DebugLines::PushBox(torchLightPos[t], TORCH_SHADOW_NEAR_CONST,
									glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), dbgPos, dbgColor);
				DebugLines::PushBox(torchLightPos[t], TORCH_SHADOW_FAR_CONST,
									glm::vec4(1.0f, 0.5f, 0.0f, 1.0f), dbgPos, dbgColor);
			}
		}
		// Collision geometry (showColliders). Read off the same list and
		// accessors the collision loops use, so it can't drift from what
		// blocks the player.
		if(cheats.showColliders) {
			// Grown outward before drawing: a box sits exactly ON its surface,
			// and coincident depth z-fights, flickering the overlay. The
			// pipeline depth-tests on purpose (an overlay through every wall is
			// unreadable), so the fix is the nudge.
			const float COLLIDER_DRAW_EPS = 0.01f;
			for(Collider *C : allColliders) {
				AABBextents E = C->getExtents();
				DebugLines::PushAABB(glm::vec3(E.xMin, E.yMin, E.zMin) - COLLIDER_DRAW_EPS,
									 glm::vec3(E.xMax, E.yMax, E.zMax) + COLLIDER_DRAW_EPS,
									 glm::vec4(0.0f, 1.0f, 0.2f, 1.0f), dbgPos, dbgColor);
			}
			// Ramps in a different color: ground-only, never walls, and their
			// surface is the drawn quad itself.
			for(const GroundVolume &G : colliderSet.ramps()) {
				// The four corners of the walkable surface, built in the model's
				// local space (where the footprint is axis-aligned whatever the
				// instance's rotation, exactly as groundAt() tests it) and then
				// sent through the same world matrix.
				glm::vec3 corners[4];
				const float runs[4]   = {G.runFrom, G.runTo, G.runTo, G.runFrom};
				const float rises[4]  = {G.riseFrom, G.riseTo, G.riseTo, G.riseFrom};
				const float widths[4] = {G.widthFrom, G.widthFrom, G.widthTo, G.widthTo};
				for(int i = 0; i < 4; i++) {
					glm::vec3 p(0.0f);
					p[G.runAxis]   = runs[i];
					p[G.riseAxis]  = rises[i];
					p[G.widthAxis] = widths[i];
					corners[i] = glm::vec3(G.Wm * glm::vec4(p, 1.0f));
				}
				DebugLines::PushQuad(corners[0], corners[1], corners[2], corners[3],
									 glm::vec4(0.2f, 0.6f, 1.0f, 1.0f), dbgPos, dbgColor);
			}
		}
		// Debug-camera companion overlay: draws the geometry cull's shape in
		// world space, so the spectator view shows exactly where an instance winks out.
		if(cheats.debugCam) {
			const glm::vec4 nearCol(0.2f, 0.9f, 1.0f, 1.0f);   // always-drawn bubble
			const glm::vec4 coneCol(1.0f, 0.8f, 0.15f, 1.0f);  // view cone
			glm::vec3 f = forward;
			glm::vec3 rr = glm::cross(f, glm::vec3(0.0f, 1.0f, 0.0f));
			rr = glm::length(rr) > 1e-4f ? glm::normalize(rr) : glm::vec3(1, 0, 0);
			glm::vec3 uu = glm::normalize(glm::cross(rr, f));
			DebugLines::PushCross(eyePos, 0.4f, coneCol, dbgPos, dbgColor);

			auto ring = [&](const glm::vec3 &c, float rad, const glm::vec3 &ax0,
							const glm::vec3 &ax1, const glm::vec4 &col) {
				const int N = 32;
				glm::vec3 prev = c + ax0 * rad;
				for(int i = 1; i <= N; i++) {
					float a = (float)i / N * 2.0f * 3.14159265f;
					glm::vec3 p = c + (ax0 * std::cos(a) + ax1 * std::sin(a)) * rad;
					DebugLines::PushLine(prev, p, col, dbgPos, dbgColor);
					prev = p;
				}
			};

			// Always-drawn radius: a sphere sketched as three great circles.
			ring(eyePos, GEOM_CULL_RADIUS, rr, uu, nearCol);
			ring(eyePos, GEOM_CULL_RADIUS, rr, f, nearCol);
			ring(eyePos, GEOM_CULL_RADIUS, uu, f, nearCol);

			// View cone: half-angle from the cull's cosine, edges to the cone distance plus a cap ring.
			float half = std::acos(GEOM_CULL_CONE_COS);
			float capR = GEOM_CULL_CONE_DIST * std::tan(half);
			glm::vec3 capC = eyePos + f * GEOM_CULL_CONE_DIST;
			for(int i = 0; i < 16; i++) {
				float a = (float)i / 16 * 2.0f * 3.14159265f;
				glm::vec3 edge = capC + (rr * std::cos(a) + uu * std::sin(a)) * capR;
				DebugLines::PushLine(eyePos, edge, coneCol, dbgPos, dbgColor);
			}
			ring(capC, capR, rr, uu, coneCol);

			// Torch-light cull companion: cross at every burning wall flame,
			// green if lightReachesViewCone() keeps it, red if dropped.
			const glm::vec4 liveCol(0.3f, 1.0f, 0.35f, 1.0f);
			const glm::vec4 deadCol(1.0f, 0.25f, 0.2f, 1.0f);
			for(const TorchFlame &tf : torchFlames) {
				if(tf.heldByCamera || !flameBurning(tf)) {
					continue;
				}
				glm::vec3 p = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
				bool live = lightReachesViewCone(p, flameLightCullReach(tf), eyePos, forward);
				DebugLines::PushCross(p, 0.5f, live ? liveCol : deadCol, dbgPos, dbgColor);
			}
		}

		debugLines.update(currentImage, ViewPrj, dbgPos, dbgColor);

		// The gazed door's chains/padlock glow along with the leaf. Empty with
		// Focus Glow off (only the aura goes, gaze itself is untouched).
		std::vector<Instance *> glowingInstances;
		if(cheats.focusGlowEnabled && gazedInstance != nullptr) {
			glowingInstances.push_back(gazedInstance);
			for(const Door &d : doors) {
				if(d.inst == gazedInstance) {
					if(d.locked) {
						for(const Door::LockProp &prop : d.lockProps) {
							glowingInstances.push_back(prop.inst);
						}
					}
					break;
				}
			}
		}

		// Over every technique, not just the first, since all instances need the same per-object uniforms.
		for(int techniqueId = 0; techniqueId < SC.TechniqueInstanceCount; techniqueId++) {
			for(int instanceId = 0; instanceId < SC.TI[techniqueId].InstanceCount; instanceId++) {
				glm::mat4 renderWm = SC.TI[techniqueId].I[instanceId].Wm;
				// Geometry visibility cull: an instance outside radius+cone
				// gets an invertible off-map translate (not a zero matrix,
				// since nMat is its inverse-transpose); the collider is
				// untouched. Tested as a bounding sphere from collider extents
				// (this pack isn't centre-pivoted), else GEOM_CULL_FALLBACK_RADIUS.
				Instance &cullInst = SC.TI[techniqueId].I[instanceId];
				glm::vec3 instPos = glm::vec3(renderWm[3]);
				float instRadius = GEOM_CULL_FALLBACK_RADIUS;
				if(cullInst.C != nullptr) {
					AABBextents e = cullInst.C->getExtents();
					instPos = glm::vec3((e.xMin + e.xMax) * 0.5f,
										 (e.yMin + e.yMax) * 0.5f,
										 (e.zMin + e.zMax) * 0.5f);
					instRadius = 0.5f * glm::length(glm::vec3(e.xMax - e.xMin,
															   e.yMax - e.yMin,
															   e.zMax - e.zMin));
				}
				if(!geometryVisible(instPos, instRadius, eyePos, forward)) {
					renderWm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
				}
				// Open-top dollhouse for the spectator view (see DEBUG_CAM_ROOF_CUT).
				if(cheats.debugCam && instPos.y > DEBUG_CAM_ROOF_CUT) {
					renderWm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
				}
				ubo.mMat = renderWm;
				ubo.mvpMat = ViewPrj * ubo.mMat;
				ubo.nMat = glm::inverse(glm::transpose(ubo.mMat));

				// By Mid rather than by name, so no string hashing per frame;
				// forInstance also checks Iid for a per-instance override (the
				// three door keys, one "key" model shaded as three metals).
				const Material &m = materials.forInstance(SC.TI[techniqueId].I[instanceId].Iid,
														 SC.TI[techniqueId].I[instanceId].Mid);
				ubo.mS = m.specularColor;
				ubo.roughness = m.roughness;
				ubo.F0 = m.F0;
				ubo.k = m.k;
				ubo.flatNormals = m.flatNormals;
				ubo.interiorAmbient = m.interiorAmbient;
				ubo.ambientWeight = m.ambientWeight;
				ubo.metallic = m.metallic;
				ubo.time = simTime;

				Instance &inst = SC.TI[techniqueId].I[instanceId];
				// Gazed instance (plus lock hardware) glows; other instances
				// of the same model don't, hence per-instance not per-Material.
				bool glow = false;
				for(Instance *g : glowingInstances) {
					if(g == &inst) {
						glow = true;
						break;
					}
				}
				// Sign = "would [E] do anything", magnitude = aura colour (GlowKind), packed into one scalar.
				float kindMag = static_cast<float>(gazedGlowKind);
				ubo.glow = glow ? (gazedInteractionDisabled ? -kindMag : kindMag) : 0.0f;

				// Ghosts' chase telegraph rides F0 (Spectral computes no BRDF).
				// Linear scan since a flag on Instance is off-limits and there are only three ghosts.
				for(const Ghost &g : ghosts) {
					if(g.inst == &inst) {
						ubo.F0 = g.chaseBlend;
						break;
					}
				}
				// DS[1] = Pchar pass (main render): set0=DSLglobal, set1=DSLlocal
				inst.DS[0][0]->map(currentImage, &gubo, 0); // global (light/camera)
				inst.DS[0][1]->map(currentImage, &ubo, 0); // camera MVPs
				// set2=DSLshadowSample is never mapped here: fixed samplerCube
				// bindings, set once at descriptor-set creation, no per-frame buffer.
			}
		}

		// ---- Shadow capture submission and HUD text ----
		// Records and submits this frame's cube shadow captures, now that both
		// DSshadowCube[t] and every instance's DS[0][1] are current --
		// recording earlier would bake a stale Wm into the map.
		submitCubeShadowCaptures(currentImage);
		pendingFaceMask.fill(0);

		// FPS counter, left visible even over the launch screen. Timed off
		// glfwGetTime(), not deltaT, since deltaT is forced to 0 while paused
		// but frames still render.
		static float elapsedT = 0.0f;
		static int countedFrames = 0;
		static double lastFpsTime = glfwGetTime();

		double nowFpsTime = glfwGetTime();
		countedFrames++;
		elapsedT += (float)(nowFpsTime - lastFpsTime);
		lastFpsTime = nowFpsTime;
		if(elapsedT > 1.0f) {
			float Fps = (float)countedFrames / elapsedT;

			std::ostringstream oss;
			oss << "FPS: " << Fps << "\n";

			txt.print(1.0f, 1.0f, oss.str(), 1, "CO", false, false, true,TAL_RIGHT,TRH_RIGHT,TRV_BOTTOM,{1.0f,0.0f,0.0f,1.0f},{0.8f,0.8f,0.0f,1.0f});

			elapsedT = 0.0f;
			countedFrames = 0;
		}

		if(!startScreen.isOpen() && !settingsMenu.isOpen()) {
			// Coordinates overlay, throttled to 10Hz since print() always dirties the text buffer.
			static float coordsElapsedT = 0.0f;
			static bool coordsShown = false;
			coordsElapsedT += deltaT;
			if(cheats.showCoordinates) {
				if(!coordsShown || coordsElapsedT > 0.1f) {
					std::ostringstream coss;
					coss << "X: " << camPos.x << "  Y: " << camPos.y << "  Z: " << camPos.z
						 << "  Yaw: " << camYaw << "\n";

					// A fixed pixel offset above the FPS line so the two don't overlap.
					float sx, sy;
					txt.pixelToScr((float)windowWidth - 1.0f, (float)windowHeight - 40.0f, sx, sy);
					txt.print(sx, sy, coss.str(), 2, "CO", false, false, true,
							  TAL_RIGHT, TRH_RIGHT, TRV_BOTTOM,
							  {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});

					coordsShown = true;
					coordsElapsedT = 0.0f;
				}
			} else if(coordsShown) {
				txt.removeText(2);
				coordsShown = false;
			}

			// The "[E] ..." prompt, shown while a door/pickup/candle/torch is
			// in range; suppressed once a run ends since GameLogic() freezes the nearby* fields.
			static bool interactPromptShown = false;
			static std::string interactPromptText;
			bool showInteractPrompt = runState == RunState::Running &&
									  (nearbyDoor >= 0 || nearbyPickup >= 0 ||
									   nearbyCandle >= 0 || nearbyWallTorch >= 0 ||
									   nearbyHandTorch);
			// Door prompts split four ways: openable padlock, unopenable
			// (names the key by lockLabel), wrong side (no key visible), plain door.
			std::string wantedPromptText;
			if(nearbyHandTorch) {
				wantedPromptText = "[E] Pick up torch";
			} else if(nearbyPickup >= 0) {
				wantedPromptText = "[E] Pick up";
			} else if(nearbyWallTorch >= 0) {
				wantedPromptText = "[E] Light your torch";
			} else if(nearbyCandle >= 0) {
				// Two ways, matching the aura the candle is wearing right now:
				// with fire in hand it's an invitation, without it's the reason
				// the candle won't take.
				wantedPromptText = hasBurningTorch()
								 ? "[E] Light the candle"
								 : "You need a lit torch to light this";
			} else if(nearbyDoor >= 0 && doors[nearbyDoor].locked) {
				// Per-door overrides (Door::promptReady etc.) win when set. Only
				// the bookcase sets them -- a padlock wants the generic wording,
				// a bookcase must never say "locked".
				const Door &d = doors[nearbyDoor];
				if(!d.onLockSide(camPos)) {
					wantedPromptText = d.promptBlocked.empty() ? "This door is blocked"
															   : d.promptBlocked;
				} else if(findKeyInRing(d.lockKeyId) >= 0) {
					wantedPromptText = d.promptReady.empty()
									 ? "[E] Unlock (uses the " + d.lockLabel + ")"
									 : d.promptReady;
				} else {
					wantedPromptText = d.promptMissing.empty()
									 ? "Locked - needs the " + d.lockLabel
									 : d.promptMissing;
				}
			} else {
				wantedPromptText = "[E] Interact";
			}
			if(showInteractPrompt && (!interactPromptShown || wantedPromptText != interactPromptText)) {
				if(interactPromptShown) txt.removeText(3);
				float sx, sy;
				txt.pixelToScr((float)windowWidth / 2.0f, (float)windowHeight - 60.0f, sx, sy);
				txt.print(sx, sy, wantedPromptText, 3, "CO", false, true, false,
						  TAL_CENTER, TRH_CENTER, TRV_BOTTOM,
						  {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
				interactPromptShown = true;
				interactPromptText = wantedPromptText;
			} else if(!showInteractPrompt && interactPromptShown) {
				txt.removeText(3);
				interactPromptShown = false;
			}

			// Hunt banner, re-printed only when the text changes; countdown rounded to whole seconds.
			static bool huntBannerShown = false;
			static std::string huntBannerText;
			std::string wantedHuntText;
			if(runState == RunState::Running) {
				if(huntCycle.phase() == HuntPhase::Warning) {
					std::ostringstream hoss;
					hoss << "THE LIGHTS ARE TURNING - "
						 << (int)std::ceil(huntCycle.timeLeftInPhase()) << "\n";
					wantedHuntText = hoss.str();
				} else if(huntCycle.phase() == HuntPhase::Hunt) {
					wantedHuntText = "RUN\n";
				}
			}
			if(!wantedHuntText.empty() && (!huntBannerShown || wantedHuntText != huntBannerText)) {
				if(huntBannerShown) txt.removeText(4);
				float sx, sy;
				txt.pixelToScr((float)windowWidth / 2.0f, 60.0f, sx, sy);
				txt.print(sx, sy, wantedHuntText, 4, "CO", false, true, false,
						  TAL_CENTER, TRH_CENTER, TRV_TOP,
						  {1.0f, 0.85f, 0.2f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
				huntBannerShown = true;
				huntBannerText = wantedHuntText;
			} else if(wantedHuntText.empty() && huntBannerShown) {
				txt.removeText(4);
				huntBannerShown = false;
			}

			// End of run: printed once on the transition, left alone until restart.
			static bool endBannerShown = false;
			static std::string endBannerText;
			std::string wantedEndText;
			if(runState == RunState::Caught) {
				wantedEndText = "CAUGHT\n[R] Try again\n";
			} else if(runState == RunState::Escaped) {
				wantedEndText = "YOU ESCAPED THE CASTLE\n[R] Play again\n";
			}
			if(!wantedEndText.empty() && (!endBannerShown || wantedEndText != endBannerText)) {
				if(endBannerShown) txt.removeText(5);
				float sx, sy;
				txt.pixelToScr((float)windowWidth / 2.0f, (float)windowHeight / 2.0f, sx, sy);
				txt.print(sx, sy, wantedEndText, 5, "CO", false, true, false,
						  TAL_CENTER, TRH_CENTER, TRV_MIDDLE,
						  {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
				endBannerShown = true;
				endBannerText = wantedEndText;
			} else if(wantedEndText.empty() && endBannerShown) {
				txt.removeText(5);
				endBannerShown = false;
			}
		}

		txt.updateCommandBuffer();
		uiQuad.updateCommandBuffer();
		crosshair.updateCommandBuffer();
		pauseQuad.updateCommandBuffer();
		startScreenQuad.updateCommandBuffer();
		settingsQuad.updateCommandBuffer();
	}
	
	// =====================================================================
	// Ghost navigation
	// =====================================================================
	// Ghosts obey the same walls the player does. Not pathfinding: a wall
	// test, a push-out like the player's, and a fan of candidate headings,
	// enough for rooms and corridors given the return trail.

	// True if a ghost-sized cylinder at `p` overlaps a wall. `radius` is a
	// parameter so ghostPathClear/ghostSteer can pad it while ghostResolveWalls uses the real body size.
	bool ghostBlockedAt(const glm::vec3 &p, float radius) const {
		bool blocked = false;
		ghostForEachNearbyCollider(p, [&](const AABBextents &E) {
			if(blocked) return;
			if(E.yMax < p.y + ghostBodyBottom) return;	// entirely underneath: floated over
			if(E.yMin > p.y + ghostBodyTop) return;	// entirely overhead: passed under
			float closestX = glm::clamp(p.x, E.xMin, E.xMax);
			float closestZ = glm::clamp(p.z, E.zMin, E.zMax);
			float dx = p.x - closestX;
			float dz = p.z - closestZ;
			if(dx * dx + dz * dz < radius * radius) {
				blocked = true;
			}
		});
		return blocked;
	}

	// Pushes `p` horizontally out of anything it's inside, along the shortest
	// escape, so the wall-parallel component survives (sliding for free).
	void ghostResolveWalls(glm::vec3 &p) const {
		// 3x3 neighbourhood computed once from `p` on entry, before any push
		// below moves it; a push is at most ghostRadius, well inside the CELL_SIZE margin.
		ghostForEachNearbyCollider(p, [&](const AABBextents &E) {
			if(E.yMax < p.y + ghostBodyBottom) return;
			if(E.yMin > p.y + ghostBodyTop) return;

			float closestX = glm::clamp(p.x, E.xMin, E.xMax);
			float closestZ = glm::clamp(p.z, E.zMin, E.zMax);
			float dx = p.x - closestX;
			float dz = p.z - closestZ;
			float dist = std::sqrt(dx * dx + dz * dz);
			if(dist >= ghostRadius) return;

			if(dist > 1e-5f) {
				float push = (ghostRadius - dist) / dist;
				p.x += dx * push;
				p.z += dz * push;
			} else {
				// Centre exactly inside the footprint: out along the nearest side.
				float pushXNeg = p.x - E.xMin, pushXPos = E.xMax - p.x;
				float pushZNeg = p.z - E.zMin, pushZPos = E.zMax - p.z;
				float minX = std::min(pushXNeg, pushXPos);
				float minZ = std::min(pushZNeg, pushZPos);
				if(minX < minZ) {
					p.x += (pushXNeg < pushXPos ? -1.0f : 1.0f) * (ghostRadius + minX);
				} else {
					p.z += (pushZNeg < pushZPos ? -1.0f : 1.0f) * (ghostRadius + minZ);
				}
			}
		});
	}

	// Whether a ghost at `from` could travel `dist` along `dir` unobstructed.
	// Sampled, not swept: point tests spaced under the ghost's radius so nothing thinner slips through.
	bool ghostPathClear(const glm::vec3 &from, const glm::vec2 &dir, float dist, float radius) const {
		const float step = ghostRadius * 0.8f;
		int samples = std::max(1, (int)std::ceil(dist / step));
		for(int i = 1; i <= samples; i++) {
			float t = dist * (float)i / (float)samples;
			glm::vec3 p = from + glm::vec3(dir.x * t, 0.0f, dir.y * t);
			if(ghostBlockedAt(p, radius)) {
				return false;
			}
		}
		return true;
	}

	// True if `p` sits inside any collider's box, testing the real Y of `p` --
	// a table only blocks a sightline if the line is actually low enough to clip it.
	bool ghostPointBlocked(const glm::vec3 &p) const {
		bool blocked = false;
		ghostForEachNearbyCollider(p, [&](const AABBextents &E) {
			if(blocked) return;
			if(p.x < E.xMin || p.x > E.xMax) return;
			if(p.z < E.zMin || p.z > E.zMax) return;
			if(p.y < E.yMin || p.y > E.yMax) return;
			blocked = true;
		});
		return blocked;
	}

	// Ghost "eye" height above its hover pivot, used only for the sightline below.
	static constexpr float GHOST_EYE_OFFSET = 0.5f;

	// Whether a ghost at `from` can see the player at `eyeTarget`. A 3D ray,
	// not ghostPathClear's flat XZ probe, so a hovering ghost clears a table
	// but walls/closed doors still block.
	bool ghostHasLineOfSight(const glm::vec3 &from, const glm::vec3 &eyeTarget) const {
		glm::vec3 eyeFrom = from + glm::vec3(0.0f, GHOST_EYE_OFFSET, 0.0f);
		glm::vec3 delta = eyeTarget - eyeFrom;
		float d = glm::length(delta);
		if(d < 1e-4f) return true;
		glm::vec3 dir = delta / d;
		const float step = ghostRadius * 0.8f;
		int samples = std::max(1, (int)std::ceil(d / step));
		// i in [1, samples): skip both endpoints (their own footprints).
		for(int i = 1; i < samples; i++) {
			float t = d * (float)i / (float)samples;
			if(ghostPointBlocked(eyeFrom + dir * t)) {
				return false;
			}
		}
		return true;
	}

	// Picks the heading a chasing ghost takes toward `desired`. Fans out in
	// widening steps and takes the first with a clear probe ahead.
	// IMPORTANT: g.turnBias prevents dithering -- without it a ghost facing a
	// pillar swaps left/right every frame as geometry shifts by centimetres;
	// the bias remembers the chosen side and re-tries it first.
	// Zero vector = boxed in, read by the caller as "don't move".
	glm::vec2 ghostSteer(Ghost &g, const glm::vec2 &desired) const {
		// Probed with a little more than the ghost's radius, so a choke point
		// only slightly wider than the body doesn't flip "clear?" every frame.
		const float steerRadius = ghostRadius * ghostSteerMargin;

		if(ghostPathClear(g.pos, desired, GHOST_PROBE_DIST, steerRadius)) {
			return desired;
		}

		// Deviations shallowest first, stopping just short of 180 (a dead straight retreat).
		static constexpr float FAN[] = {25.0f, 50.0f, 75.0f, 100.0f, 125.0f, 150.0f};
		for(float deg : FAN) {
			// The remembered side first, then the other one.
			for(int s = 0; s < 2; s++) {
				float sign = (s == 0) ? g.turnBias : -g.turnBias;
				float a = glm::radians(deg) * sign;
				float c = std::cos(a), sn = std::sin(a);
				glm::vec2 cand(desired.x * c - desired.y * sn,
							   desired.x * sn + desired.y * c);
				if(ghostPathClear(g.pos, cand, GHOST_PROBE_DIST, steerRadius)) {
					g.turnBias = sign;
					return cand;
				}
			}
		}
		return glm::vec2(0.0f);
	}

	// =====================================================================
	// Run lifecycle: restart and hunt phase
	// =====================================================================
	// Puts everything a run touches back to its authored state. No file
	// reload -- every "authored" value was captured in localInit(), so a
	// restart can't disagree with the scene the game started from.
	void restartRun() {
		camPos = spawnPos;
		camYaw = spawnYaw;
		camPitch = spawnPitch;
		camVerticalVelocity = 0.0f;
		eyeStepOffset = 0.0f;
		grounded = true;
		sprinting = false;
		walkBobPhase = 0.0f;
		walkBobBlend = 0.0f;

		huntCycle.reset();

		for(Ghost &g : ghosts) {
			g.mode = GhostMode::Patrol;
			g.targetIdx = 1;
			g.distAlongSegment = 0.0f;
			g.resumeIdx = 1;
			g.resumeDist = 0.0f;
			g.trail.clear();
			g.turnBias = 1.0f;
			g.hasLastKnown = false;
			g.stuckTimer = 0.0f;
			if(!g.waypoints.empty()) {
				g.pos = g.waypoints[0];
			}
		}

		for(Door &d : doors) {
			d.open = false;
			d.angle = 0.0f;
			d.swingSign = d.openAngleDeg < 0.0f ? -1.0f : 1.0f;
			d.locked = !d.lockKeyId.empty();	// padlocks come back with the run, or it gets easier each restart
			d.inst->Wm = d.baseWm;
			if(d.inst->C != nullptr) {
				d.inst->C->setWorldMatrix(d.inst->Wm);
			}
		}

		for(Pickup &p : pickups) {
			p.collected = false;
			p.consumed = false;
			p.worldPos = p.spawnPos;
			p.inst->Wm = p.spawnWm;
		}
		keyRing.clear();
		keyLowerIdx = -1;	// else a key mid-fall keeps drawing off camera before being parked

		// Torch back on the floor, every lit candle back out; Wm restored next frame.
		handTorchCollected = false;
		for(TorchFlame &tf : torchFlames) {
			tf.burning = tf.spawnBurning;
			tf.ignitionScale = tf.spawnBurning ? 1.0f : 0.0f;	// snap to rest, not mid-spring
			tf.ignitionVel = 0.0f;
		}

		nearbyDoor = -1;
		nearbyPickup = -1;
		nearbyCandle = -1;
		nearbyWallTorch = -1;
		nearbyHandTorch = false;
		gazedInstance = nullptr;
		exitOpenFrac = 0.0f;	// the way out closes with the rest of the doors, taking its daylight/whiteout
		escapeFlash = 0.0f;

		runState = RunState::Running;
	}

	// Called once on the frame the hunt phase changes; just the announcement
	// (the one place a music track would be swapped). No audio backend yet, so it prints.
	void onHuntPhaseChanged() {
		switch(huntCycle.phase()) {
			case HuntPhase::Calm:
				std::cout << "[hunt] calm\n";
				break;
			case HuntPhase::Warning:
				std::cout << "[hunt] warning: the lights are turning\n";
				break;
			case HuntPhase::Hunt:
				std::cout << "[hunt] HUNT\n";
				break;
		}
	}

	// =====================================================================
	// GameLogic(): input, physics and interaction
	// =====================================================================
	float GameLogic() {
		// Camera FOV-y, Near Plane and Far Plane
		const float FOVy = glm::radians(45.0f);
		const float nearPlane = 0.1f;
		const float farPlane = 100.f;

		// Camera movement controls
		// FOV degrees rotated per second
		const float ROT_SPEED = 90.0f;

		// ---- Frame timing and input ----
		// Integration with the timers and the controllers
		float deltaT;
		glm::vec3 m = glm::vec3(0.0f), r = glm::vec3(0.0f);
		bool fire = false;

		pollFullscreenToggle();

		// Poll the overlays BEFORE getSixAxis, which turns on
		// GLFW_STICKY_MOUSE_BUTTONS (a one-shot read): whoever polls the click
		// first gets the authoritative read. The HUD only reacts while no other
		// overlay is open -- one modal at a time.
		if(!pauseMenu.isOpen() && !startScreen.isOpen() && !settingsMenu.isOpen()) {
			hud.update(window, windowWidth, windowHeight);
		}

		startScreen.update(window, windowWidth, windowHeight);
		if(startScreen.playClicked()) {
			startScreen.setOpen(false, windowWidth, windowHeight);
		}
		if(startScreen.settingsClicked()) {
			// Remember to come back HERE (not PauseMenu) when Back is pressed.
			startScreen.setOpen(false, windowWidth, windowHeight);
			settingsFromPause = false;
			settingsMenu.setOpen(true, windowWidth, windowHeight);
		}
		if(startScreen.quitClicked()) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}

		pauseMenu.update(window, windowWidth, windowHeight);
		if(pauseMenu.resumeClicked()) {
			pauseMenu.setOpen(false, windowWidth, windowHeight);
		}
		if(pauseMenu.settingsClicked()) {
			// Same as StartScreen's above, but remembers to return to PauseMenu instead.
			pauseMenu.setOpen(false, windowWidth, windowHeight);
			settingsFromPause = true;
			settingsMenu.setOpen(true, windowWidth, windowHeight);
		}
		if(pauseMenu.quitClicked()) {
			// Abandon the run and drop back to the launch screen; restartRun() resets all world state.
			pauseMenu.setOpen(false, windowWidth, windowHeight);
			restartRun();
			startScreen.setOpen(true, windowWidth, windowHeight);
		}

		settingsMenu.update(window, windowWidth, windowHeight);
		if(settingsMenu.backClicked()) {
			// Reopen whichever menu sent the player here (settingsFromPause tracks which).
			settingsMenu.setOpen(false, windowWidth, windowHeight);
			if(settingsFromPause) {
				pauseMenu.setOpen(true, windowWidth, windowHeight);
			} else {
				startScreen.setOpen(true, windowWidth, windowHeight);
			}
		}

		getSixAxis(deltaT, m, r, fire);

		// Clamped after getSixAxis, before physics: Euler integration on a
		// stalled frame's deltaT spike would overshoot colliders and fall
		// through the floor. 1/20s caps it, the standard fix for a variable timestep.
		const float MAX_DELTA_T = 1.0f / 20.0f;
		if(deltaT > MAX_DELTA_T) {
			deltaT = MAX_DELTA_T;
		}

		if(overlayOpen()) {
			// HUD or pause menu is open: discard camera-look/move/fire input
			// this frame so a click or drag on either can't also spin the
			// camera underneath it.
			m = glm::vec3(0.0f);
			r = glm::vec3(0.0f);
			fire = false;
		}

		// ---- Projection and camera orientation ----
		// Projection
		glm::mat4 Prj = glm::perspective(FOVy, Ar, nearPlane, farPlane);
		Prj[1][1] *= -1;

		// Control Camera rotation
		// Yaw: left-right
		camYaw += r.y * ROT_SPEED * deltaT;
		// Pitch: up-down
		camPitch += -r.x * ROT_SPEED * deltaT;
		// Cap pitch to avoid full rotations, limits are full-down and full-up
		camPitch = glm::clamp(camPitch, -89.0f, 89.0f);

		// Convert Yaw and Pitch into Cartesian coordinates
		// Define global UP vector
		const glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
		// Define current FRONT vector
		glm::vec3 front;
		// Update FRONT based on new YAW and PITCH
		front.x = cos(glm::radians(camYaw)) * cos(glm::radians(camPitch));
		front.y = sin(glm::radians(camPitch));
		front.z = sin(glm::radians(camYaw)) * cos(glm::radians(camPitch));
		// Normalize values
		front = glm::normalize(front);
		// Compute RIGHT and UP vectors
		glm::vec3 right = glm::normalize(glm::cross(front, worldUp));
		glm::vec3 up = glm::normalize(glm::cross(right, front));

		// ---- Restart, whiteout and the hunt clock ----
		// Restart: only offered once a run has ended, edge-triggered so holding it doesn't restart every frame.
		bool restartKey = glfwGetKey(window, GLFW_KEY_R);
		if(runState != RunState::Running && restartKey && !restartKeyWasPressed && !overlayOpen()) {
			restartRun();
		}
		restartKeyWasPressed = restartKey;

		// Whiteout ramped here, not in the frozen-movement block, since it
		// must keep running after the run ends. Held while an overlay is open.
		if(!overlayOpen()) {
			float flashTarget = (runState == RunState::Escaped) ? 1.0f : 0.0f;
			if(flashTarget > escapeFlash) {
				escapeFlash = std::min(escapeFlash + deltaT / ESCAPE_FLASH_SECONDS, 1.0f);
			} else {
				escapeFlash = flashTarget;
			}
		}

		// The hunt clock, gated like the movement block below: a cycle counting
		// down behind an overlay would drop the player into a phase they never
		// saw start.
		if(!overlayOpen() && runState == RunState::Running) {
			huntCycle.update(deltaT);
			if(huntCycle.phaseJustChanged()) {
				onHuntPhaseChanged();
			}
		}

		// ---- Movement, collision, interaction and ghosts ----
		// Freeze all movement/physics while an overlay is open or a run has ended
		if(!overlayOpen() && runState == RunState::Running) {
			// Sprint -> Ctrl
			bool ctrlHeld = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL);
			if(!ctrlHeld) {
				sprinting = false;
			} else if(grounded) {
				sprinting = true;
			}
			float moveSpeed = movement.moveSpeed;
			if(sprinting) {
				moveSpeed *= movement.sprintMultiplier;
			}

			// -- Walking, wall collision and jumping --
			// WASD: m.y (fly) dropped, vertical movement only from jump/gravity.
			// Flattened to yaw only (not pitched `front`), so looking up while pressing W doesn't push through the floor.
			glm::vec3 frontFlat = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
			camPos += (right * m.x - frontFlat * m.z) * moveSpeed * deltaT;

			// Wall collision: push out of any collider too tall to step onto
			// and low enough for the body to reach. The ground pass handles
			// the rest, which is what makes steps and crates walkable.
			if(cheats.collisionEnabled) {
				const float EYE_HEIGHT = 1.8f;
				const float PLAYER_HEIGHT = 1.8f;
				const float PLAYER_RADIUS = 0.3f;
				const float VERTICAL_MARGIN = 0.1f;	// so grazing the gate lintel's underside isn't "inside" it
				float feetY = camPos.y - EYE_HEIGHT;
				float headY = feetY + PLAYER_HEIGHT;
				for(Collider *C : allColliders) {
					AABBextents E = C->getExtents();

					// Low enough to step onto. The ground pass lifts the player onto it.
					if(E.yMax <= feetY + MAX_STEP_HEIGHT) continue;

					// A wall only where the body reaches it: walk under a lintel above head height.
					if(headY <= E.yMin + VERTICAL_MARGIN) continue;

					// Closest point on the collider's XZ footprint to the camera
					float closestX = glm::clamp(camPos.x, E.xMin, E.xMax);
					float closestZ = glm::clamp(camPos.z, E.zMin, E.zMax);
					float dx = camPos.x - closestX;
					float dz = camPos.z - closestZ;
					float dist = std::sqrt(dx * dx + dz * dz);

					if(dist < PLAYER_RADIUS) {
						if(dist > 1e-5f) {
							// Push out along the horizontal vector to the closest surface point.
							float push = (PLAYER_RADIUS - dist) / dist;
							camPos.x += dx * push;
							camPos.z += dz * push;
						} else {
							// XZ exactly inside the footprint: push out the nearest side.
							float pushXNeg = camPos.x - E.xMin, pushXPos = E.xMax - camPos.x;
							float pushZNeg = camPos.z - E.zMin, pushZPos = E.zMax - camPos.z;
							float minX = std::min(pushXNeg, pushXPos);
							float minZ = std::min(pushZNeg, pushZPos);
							if(minX < minZ) {
								camPos.x += (pushXNeg < pushXPos ? -1.0f : 1.0f) * (PLAYER_RADIUS + minX);
							} else {
								camPos.z += (pushZNeg < pushZPos ? -1.0f : 1.0f) * (PLAYER_RADIUS + minZ);
							}
						}
					}
				}

			}

			// Jump: spacebar (Starter's "fire"), only while grounded.
			if(cheats.jumpEnabled && fire && !jumpKeyWasPressed && grounded) {
				camVerticalVelocity = movement.jumpSpeed;
				eyeStepOffset = 0.0f;	// drop leftover step smoothing, or a jump after a step-up trails the view
			}
			jumpKeyWasPressed = fire;

			// -- [E] interaction: doors, pickups, candles, torches --
			// findGazedDoor picks the target; DOOR_INTERACT_RADIUS gates reachability.
			nearbyDoor = -1;
			{
				int gazed = findGazedDoor(front);
				if(gazed >= 0 && doorDistance(doors[gazed], camPos) < DOOR_INTERACT_RADIUS) {
					nearbyDoor = gazed;
				}
			}

			// Pickups: same gaze-then-proximity check; wins over a door via E-key priority order.
			nearbyPickup = -1;
			{
				int gazed = findGazedPickup(front);
				if(gazed >= 0 && !pickups[gazed].consumed) {
					const Pickup &p = pickups[gazed];
					float dx = camPos.x - p.worldPos.x;
					float dy = camPos.y - p.worldPos.y;
					float dz = camPos.z - p.worldPos.z;
					float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
					if(dist < PICKUP_INTERACT_RADIUS) {
						nearbyPickup = gazed;
					}
				}
			}

			// Candles: an unlit one aimed at within reach is lit by [E]. Not
			// gated on having fire here, so the disabled glow/prompt can explain why.
			nearbyCandle = -1;
			{
				int gazed = findGazedCandle(front);
				if(gazed >= 0) {
					const TorchFlame &tf = torchFlames[gazed];
					float dx = camPos.x - tf.anchorWorld.x;
					float dy = camPos.y - tf.anchorWorld.y;
					float dz = camPos.z - tf.anchorWorld.z;
					float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
					if(dist < CANDLE_INTERACT_RADIUS) {
						nearbyCandle = gazed;
					}
				}
			}

			// Wall torches: mirror of the candle block. findGazedWallTorch
			// already returns -1 once the held torch is burning.
			nearbyWallTorch = -1;
			{
				int gazed = findGazedWallTorch(front);
				if(gazed >= 0) {
					const TorchFlame &tf = torchFlames[gazed];
					float dx = camPos.x - tf.anchorWorld.x;
					float dz = camPos.z - tf.anchorWorld.z;
					float dist = std::sqrt(dx * dx + dz * dz);	// XZ only, see WALL_TORCH_INTERACT_RADIUS
					if(dist < WALL_TORCH_INTERACT_RADIUS) {
						nearbyWallTorch = gazed;
					}
				}
			}

			// The floor torch: standard pickup tolerances.
			nearbyHandTorch = false;
			if(findGazedHandTorch(front)) {
				float dx = camPos.x - handTorchWorldPos.x;
				float dy = camPos.y - handTorchWorldPos.y;
				float dz = camPos.z - handTorchWorldPos.z;
				if(std::sqrt(dx * dx + dy * dy + dz * dz) < PICKUP_INTERACT_RADIUS) {
					nearbyHandTorch = true;
				}
			}

			// Single targeted instance for the focus glow, in E-key priority order
			gazedInstance = nullptr;
			gazedInteractionDisabled = false;
			if(nearbyHandTorch) {
				gazedInstance = handTorchInst;
				gazedGlowKind = GlowKind::Pickup;
			} else if(nearbyPickup >= 0) {
				gazedInstance = pickups[nearbyPickup].inst;
				gazedGlowKind = GlowKind::Pickup;
			} else if(nearbyWallTorch >= 0) {
				gazedInstance = torchFlames[nearbyWallTorch].inst;
				gazedGlowKind = GlowKind::WallTorch;
			} else if(nearbyCandle >= 0) {
				gazedInstance = torchFlames[nearbyCandle].inst;
				gazedGlowKind = GlowKind::Candle;
				gazedInteractionDisabled = !hasBurningTorch();	// nothing to light it with: red aura
			} else if(nearbyDoor >= 0) {
				const Door &d = doors[nearbyDoor];
				gazedInstance = d.inst;
				gazedGlowKind = GlowKind::Door;
				// Wrong side of the padlock, or no matching key: [E] would do nothing.
				gazedInteractionDisabled =
					d.locked && (!d.onLockSide(camPos) || findKeyInRing(d.lockKeyId) < 0);
			}

			bool interactKey = glfwGetKey(window, GLFW_KEY_E);
			if(interactKey && !interactKeyWasPressed) {
				if(nearbyHandTorch) {
					// Into the hand, still unlit; next frame's updateUniformBuffer() rebuilds Wm off the camera.
					handTorchCollected = true;
					torchRaiseElapsed = 0.0f;	// restart the raise
					std::cout << "[torch] picked up hand torch\n";
					nearbyHandTorch = false;
					gazedInstance = nullptr;

					// Floor object disappear -> sweep occupied-slot faces once (shadow): queueMoverCubeSlotRenders() can miss it if too far.

					{
						const glm::vec4 &local = modelSphere(handTorchInst->Mid);
						const glm::vec3 centre = glm::vec3(handTorchSpawnWm * glm::vec4(glm::vec3(local), 1.0f));
						const float scale = std::max({glm::length(glm::vec3(handTorchSpawnWm[0])),
													  glm::length(glm::vec3(handTorchSpawnWm[1])),
													  glm::length(glm::vec3(handTorchSpawnWm[2]))});
						const float r = local.w * scale;
						for(int t = 0; t < HAND_TORCH_SHADOW_INDEX; t++) {
							if(!cubeSlotOccupied(t)) {
								continue;
							}
							const glm::vec3 rel = centre - torchLightPos[t];
							for(int face = 0; face < 6; face++) {
								if(sphereInCubeFace(rel, r, face)) {
									pendingFaceMask[t] |= (uint8_t)(1u << face);
								}
							}
						}
					}
				} else if(nearbyPickup >= 0) {
					Pickup &p = pickups[nearbyPickup];
					p.collected = true;
					// No visibility flag -> "removed" means parked below the
					// map; a key is redrawn in hand from here on.
					p.inst->Wm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
					if(!p.keyId.empty()) {
						// Hand already occupied -> drop item on floor and change it in hand
						if(!keyRing.empty()) {
							dropKeyFromRing((int)keyRing.size() - 1, camPos, front, 0.5f);
						}
						keyRing.push_back(nearbyPickup);
						keyRaiseElapsed = 0.0f;	// restart the raise
					}
				} else if(nearbyWallTorch >= 0) {
					// Light the hand torch from nearby wall torch.
					torchFlames[handFlameIdx].burning = true;
					std::cout << "[torch] lit hand torch from '"
							  << *torchFlames[nearbyWallTorch].inst->id << "'\n";
					nearbyWallTorch = -1;
					gazedInstance = nullptr;
				} else if(nearbyCandle >= 0) {
					// Light the candle from the held torch. Requires fire in hand.
					if(hasBurningTorch()) {
						torchFlames[nearbyCandle].burning = true;
						std::cout << "[candle] lit '"
								  << *torchFlames[nearbyCandle].inst->id << "'\n";
						// Lit now, so it stops being a target.
						nearbyCandle = -1;
						gazedInstance = nullptr;
					}
				} else if(nearbyDoor >= 0) {
					Door &d = doors[nearbyDoor];
					if(d.locked) {
						// Padlocked: E spends a matching key (destroyed) and swings the door open.
						// No match / wrong face of the door: nothing
						// happens (the prompt already says what's missing).
						int slot = d.onLockSide(camPos) ? findKeyInRing(d.lockKeyId) : -1;
						if(slot >= 0) {
							std::cout << "[door] unlocked '" << d.instanceId
									  << "' with key '" << d.lockKeyId << "'\n";
							Instance *spent = pickups[keyRing[slot]].inst;
							consumeKey(slot);
							// If item is a whenUnlocked prop of this door (book) ->
							// cancel the sink consumeKey() just queued ->
							// prop loop puts it onto the shelf this same frame (later in code).
							for(const Door::LockProp &prop : d.lockProps) {
								if(prop.whenUnlocked && prop.inst == spent) {
									keyLowerIdx = -1;
								}
							}
							d.locked = false;
							d.swingSign = d.swingSignAwayFrom(camPos);	// a locked door is always fully closed here
							d.open = true;
						}
					} else {
						// Only a fully-closed leaf picks a side; reversing one
						// still swinging shut would drag the panel through the player.
						if(!d.open && d.angle == 0.0f) {
							d.swingSign = d.swingSignAwayFrom(camPos);
						}
						d.open = !d.open;
					}
				}
			}
			interactKeyWasPressed = interactKey;

			// Drop key (G): puts the newest key down at arm's length; repeated presses drop the ring in reverse order.
			// (Current: just 1 kay at the time)
			bool dropKey = glfwGetKey(window, GLFW_KEY_G);
			if(heldKeyIdx() >= 0 && dropKey && !dropKeyWasPressed) {
				dropKeyFromRing((int)keyRing.size() - 1, camPos, front, 1.0f);
			}
			dropKeyWasPressed = dropKey;

			for(Door &d : doors) {
				// Magnitude from the scene, direction from whoever opened it (swingSign).
				float target = d.open ? std::abs(d.openAngleDeg) * d.swingSign : 0.0f;
				float maxStep = DOOR_OPEN_SPEED * deltaT;
				if(d.angle < target) d.angle = std::min(d.angle + maxStep, target); // Rotation
				else if(d.angle > target) d.angle = std::max(d.angle - maxStep, target); // Rotation

				// The leaf's local origin is its hinge, so opening it is one more rotation on the closed transform.
				// Order of operations: local -> world. Door_closed_mat * rotation_mat
				d.inst->Wm = d.baseWm * glm::rotate(glm::mat4(1.0f), glm::radians(d.angle), glm::vec3(0.0f, 1.0f, 0.0f));
				if(d.inst->C != nullptr) {
					d.inst->C->setWorldMatrix(d.inst->Wm);
				}

				// Chains/padlock follow the door while locked, hidden below the map once not.
				// whenUnlocked props (book in bookshelf) are the mirror: appear once unlocked, untouched while locked.
				// Driven every frame so restartRun() gets re-locking for free.
				for(const Door::LockProp &prop : d.lockProps) {
					if(prop.whenUnlocked) {
						if(!d.locked) prop.inst->Wm = d.inst->Wm * prop.local;
					} else {
						prop.inst->Wm = d.locked ? d.inst->Wm * prop.local
												 : glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
					}
					// Collider follows the same matrix (Pickup instances have none).
					if(prop.inst->C != nullptr) {
						prop.inst->C->setWorldMatrix(prop.inst->Wm);
					}
				}
			}

			// -- Exit door, ghosts and the win/lose checks --
			// How open the exit door is, 0 to 1, used to fade in its light as it swings.
			// Stops updating once you win.
			if(exitDoorIndex >= 0) {
				const Door &exitDoor = doors[exitDoorIndex];
				float span = std::abs(exitDoor.openAngleDeg);
				exitOpenFrac = span > 1e-4f
					? glm::smoothstep(0.0f, 1.0f, glm::clamp(std::abs(exitDoor.angle) / span, 0.0f, 1.0f))
					: 0.0f;
			}

			// Ghosts: see the Ghost struct for the three modes. Bob (vertical height) is added on
			// top of `pos` at the end, so it never feeds back into steering/collision.
			bool ghostsHunting = huntCycle.hunting();
			for(Ghost &g : ghosts) {
				if(g.inst == nullptr || g.waypoints.size() < 2) continue;

				// Checked every frame a hunt is on, regardless of mode: a
				// Patrol/Return ghost that spots the player starts a chase.
				if(ghostsHunting && ghostHasLineOfSight(g.pos, camPos)) {
					g.lastKnownPlayerPos = camPos;
					g.hasLastKnown = true;
				}

				// A ghost that never saw the player stays on patrol even during a hunt.
				if(ghostsHunting && g.hasLastKnown && g.mode != GhostMode::Chase) {
					if(g.mode == GhostMode::Patrol) {
						// Remember where on the loop we're leaving, start a fresh trail there.
						g.resumeIdx = g.targetIdx;
						g.resumeDist = g.distAlongSegment;
						g.trail.clear();
						g.trail.push_back(g.pos);
					}
					g.mode = GhostMode::Chase;	// out of Return, the existing trail is still the way home
					g.stuckCheckPos = g.pos;
					g.stuckTimer = 0.0f;
				} else if(!ghostsHunting && g.mode == GhostMode::Chase) {
					g.mode = g.trail.empty() ? GhostMode::Patrol : GhostMode::Return;	// empty trail = chase never went anywhere
				}

				// Move, per mode; each branch leaves a `moveDir` for facing below.
				glm::vec2 moveDir(0.0f);

				if(g.mode == GhostMode::Chase) {
					// Toward the last place the player was seen, not their live position.
					glm::vec2 toPlayer(g.lastKnownPlayerPos.x - g.pos.x, g.lastKnownPlayerPos.z - g.pos.z);
					float d = glm::length(toPlayer);
					if(d > 1e-4f) {
						moveDir = ghostSteer(g, toPlayer / d);
						if(moveDir != glm::vec2(0.0f)) {
							float step = std::min(g.chaseSpeed * deltaT, d);	// min(step, d): don't overshoot a reached player
							g.pos.x += moveDir.x * step;
							g.pos.z += moveDir.y * step;
						}
						ghostResolveWalls(g.pos);
					}

					// Giving up: pinned against a wall or a stale target returns to patrol after a timer.
					float moved = glm::length(glm::vec2(g.pos.x - g.stuckCheckPos.x, g.pos.z - g.stuckCheckPos.z));
					if(moved >= GHOST_STUCK_EPS) {
						g.stuckCheckPos = g.pos;
						g.stuckTimer = 0.0f;
					} else {
						g.stuckTimer += deltaT;
						if(g.stuckTimer >= GHOST_GIVEUP_TIME) {
							g.hasLastKnown = false;
							g.mode = g.trail.empty() ? GhostMode::Patrol : GhostMode::Return;
						}
					}

					// Breadcrumb dropped by distance travelled, not time, so trail density doesn't depend on frame rate.
					if(g.trail.empty()) {
						g.trail.push_back(g.pos);
					} else if(glm::length(glm::vec2(g.pos.x - g.trail.back().x,
													g.pos.z - g.trail.back().z)) >= GHOST_TRAIL_SPACING) {
						// Loop check: revisiting an earlier crumb drops everything
						// after it (oldest match = biggest cut). Last 2 crumbs skipped, always within prune radius.
						int cut = -1;
						for(int i = 0; i + 2 < (int)g.trail.size(); i++) {
							glm::vec2 delta(g.pos.x - g.trail[i].x, g.pos.z - g.trail[i].z);
							if(glm::length(delta) < GHOST_TRAIL_PRUNE_RADIUS) {
								cut = i;
								break;
							}
						}
						if(cut >= 0) {
							g.trail.resize((size_t)cut + 1);	// no new crumb: near enough to trail[cut] already
						} else {
							g.trail.push_back(g.pos);
						}
					}
				} else if(g.mode == GhostMode::Return) {
					// Walk the breadcrumbs backwards, popping each as reached (faster than chase, dead time for the player).
					glm::vec3 target = g.trail.back();
					glm::vec2 delta(target.x - g.pos.x, target.z - g.pos.z);
					float d = glm::length(delta);
					float step = GHOST_RETURN_SPEED * deltaT;
					if(d <= step) {
						// Reached: land exactly on it (a known-walkable point) and drop it.
						g.pos.x = target.x;
						g.pos.z = target.z;
						g.pos.y = target.y;
						if(d > 1e-4f) moveDir = delta / d;
						g.trail.pop_back();
						if(g.trail.empty()) {
							// Home: restoring the saved leg/distance resumes the loop mid-stride.
							g.mode = GhostMode::Patrol;
							g.targetIdx = g.resumeIdx;
							g.distAlongSegment = g.resumeDist;
						}
					} else {
						moveDir = delta / d;
						g.pos.x += moveDir.x * step;
						g.pos.z += moveDir.y * step;
						ghostResolveWalls(g.pos);
					}
				} else {
					// Patrol: walks the waypoint loop at constant speed (distance-based, so `speed` is real units/s).
					int n = (int)g.waypoints.size();
					glm::vec3 from = g.waypoints[g.targetIdx == 0 ? n - 1 : g.targetIdx - 1];
					glm::vec3 to = g.waypoints[g.targetIdx];
					float segLen = glm::length(glm::vec2(to.x - from.x, to.z - from.z));

					g.distAlongSegment += g.speed * deltaT;
					while(segLen > 0.0f && g.distAlongSegment >= segLen) {
						g.distAlongSegment -= segLen;
						g.targetIdx = (g.targetIdx + 1) % n;
						from = g.waypoints[g.targetIdx == 0 ? n - 1 : g.targetIdx - 1];
						to = g.waypoints[g.targetIdx];
						segLen = glm::length(glm::vec2(to.x - from.x, to.z - from.z));
					}

					float t = segLen > 0.0f ? g.distAlongSegment / segLen : 0.0f;
					// Patrol follows the authored waypoints exactly, no wall push-out a nearby collider can't nudge it off its loop.
					g.pos = glm::mix(from, to, t);
					if(segLen > 0.0f) {
						moveDir = glm::normalize(glm::vec2(to.x - from.x, to.z - from.z));
					}
				}

				// Facing, eased toward travel; stationary ghost keeps its yaw.
				// +M_PI: the mesh's front faces -Z, not atan2's +Z.
				if(moveDir != glm::vec2(0.0f)) {
					float targetYaw = std::atan2(moveDir.x, moveDir.y) + (float)M_PI;
					// Shortest way round, or easing +179 -> -179 spins the ghost.
					float dYaw = targetYaw - g.yaw;
					while(dYaw > (float)M_PI)  dYaw -= 2.0f * (float)M_PI;
					while(dYaw < -(float)M_PI) dYaw += 2.0f * (float)M_PI;
					float maxStep = GHOST_TURN_SPEED * deltaT;
					g.yaw += glm::clamp(dYaw, -maxStep, maxStep);
				}

				// // Chase indicator (e.g. glow), smoothed instead of snapping on/off.
				// Here, not in updateUniformBuffer(), because this loop owns `mode` and has a deltaT.
				{
					float target = (g.mode == GhostMode::Chase) ? 1.0f : 0.0f;
					float tau = (target > g.chaseBlend) ? GHOST_CHASE_FADE_IN_TAU
													    : GHOST_CHASE_FADE_OUT_TAU;
					g.chaseBlend += (target - g.chaseBlend) * (1.0f - std::exp(-deltaT / tau));
				}

				g.bobPhase += GHOST_BOB_SPEED * deltaT;
				glm::vec3 drawPos = g.pos;
				drawPos.y += std::sin(g.bobPhase) * GHOST_BOB_AMPLITUDE;

				g.inst->Wm = glm::translate(glm::mat4(1.0f), drawPos)
							* glm::rotate(glm::mat4(1.0f), g.yaw, glm::vec3(0.0f, 1.0f, 0.0f));

				// The catch. Only a hunting ghost ends the run.
				if(ghostsHunting && cheats.ghostsCanCatch && runState == RunState::Running) {
					float dx = camPos.x - g.pos.x;
					float dz = camPos.z - g.pos.z;
					// From the chest, not the eyes: the eye height is the top of the body.
					float dy = std::abs((camPos.y - 0.9f) - g.pos.y);
					if(dx * dx + dz * dz < GHOST_CATCH_RADIUS * GHOST_CATCH_RADIUS &&
					   dy < GHOST_CATCH_VERTICAL) {
						runState = RunState::Caught;
						std::cout << "[run] caught by '" << g.instanceId << "'\n";
					}
				}

				// Non-hunt encounter:
				// walking through a ghost doesn't end the run.
				// Snuffs the held torch (setting `burning` false plays the ignition spring in reverse).
				// Triggered by the spectral veil's own ramp, not GHOST_CATCH_RADIUS.
				if(!ghostsHunting && handTorchCollected && handFlameIdx >= 0 &&
				   torchFlames[handFlameIdx].burning) {
					const float dyv = camPos.y - drawPos.y;
					const float below = ghostBodyBottom - SPECTRAL_VEIL_FADE_Y;
					const float above = ghostBodyTop + SPECTRAL_VEIL_FADE_Y;
					if(dyv > below && dyv < above) {
						const float vy = glm::smoothstep(below, ghostBodyBottom, dyv) *
										 (1.0f - glm::smoothstep(ghostBodyTop, above, dyv));
						const float dxv = camPos.x - drawPos.x;
						const float dzv = camPos.z - drawPos.z;
						const float horiz = std::sqrt(dxv * dxv + dzv * dzv);
						const float vxz = 1.0f - glm::smoothstep(SPECTRAL_VEIL_INNER,
																 SPECTRAL_VEIL_OUTER, horiz);
						if(vxz * vy > SPECTRAL_VEIL_SNUFF_AT) {
							torchFlames[handFlameIdx].burning = false;
							std::cout << "[torch] hand torch snuffed by ghost '"
									  << g.instanceId << "'\n";
						}
					}
				}
			}

			// Standing in the exit box wins. Checked after the ghosts, so a
			// catch on the threshold beats reaching it.
			if(exitHasBox && runState == RunState::Running) {
				bool inside = camPos.x >= exitBoxMin.x && camPos.x <= exitBoxMax.x &&
							  camPos.y >= exitBoxMin.y && camPos.y <= exitBoxMax.y &&
							  camPos.z >= exitBoxMin.z && camPos.z <= exitBoxMax.z;
				if(inside) {
					runState = RunState::Escaped;
					std::cout << "[run] escaped\n";
				}
			}

			camVerticalVelocity += movement.gravity * deltaT;
			camPos.y += camVerticalVelocity * deltaT;

			// -- Ground collision --
			// Floor collision: the tallest surface under the player's XZ
			// within MAX_STEP_HEIGHT of the feet is the standing height (lets hole-shaped models be walked through).
			if(cheats.collisionEnabled) {
				const float EYE_HEIGHT = 1.8f;
				float feetY = camPos.y - EYE_HEIGHT;
				float groundY = -std::numeric_limits<float>::infinity();
				for(Collider *C : allColliders) {
					AABBextents E = C->getExtents();
					bool insideXZ = camPos.x >= E.xMin && camPos.x <= E.xMax &&
									camPos.z >= E.zMin && camPos.z <= E.zMax;
					// Near the feet, not towering overhead (a wall being walked past).
					bool nearFeet = E.yMax <= feetY + MAX_STEP_HEIGHT;
					if(insideXZ && nearFeet && E.yMax > groundY) {
						groundY = E.yMax;
					}
				}

				// Ramps: same MAX_STEP_HEIGHT rule, but height varies continuously along the slope.
				float rampY;
				for(const GroundVolume &G : colliderSet.ramps()) {
					if(G.groundAt(camPos, rampY) &&
					   rampY <= feetY + MAX_STEP_HEIGHT && rampY > groundY) {
						groundY = rampY;
					}
				}
				// Clamp height if clipping through the highest surface found
				if(feetY < groundY) {
					feetY = groundY;
					if(camVerticalVelocity < 0.0f) {
						camVerticalVelocity = 0.0f;	// landed: stop falling
					}
				}
				const float GROUND_EPSILON = 0.05f;	// tolerance so an irregular floor still counts as grounded
				grounded = feetY <= groundY + GROUND_EPSILON;
				float preClampY = camPos.y;
				camPos.y = feetY + EYE_HEIGHT;

				// Only an upward clamp is handed to the view smoothing below, so
				// landings/falls stay as sharp as gravity made them.
				float lifted = camPos.y - preClampY;
				if(lifted > 0.0f) {
					eyeStepOffset = std::min(eyeStepOffset + lifted, MAX_EYE_STEP_OFFSET);
				}
			} else {
				// No-clip: walls ignored, but the world floor still holds so you can't fall out the bottom.
				const float EYE_HEIGHT = 1.8f;
				if(camPos.y - EYE_HEIGHT < worldFloorY) {
					camPos.y = worldFloorY + EYE_HEIGHT;
					if(camVerticalVelocity < 0.0f) {
						camVerticalVelocity = 0.0f;
					}
				}
				grounded = camPos.y - EYE_HEIGHT <= worldFloorY + 0.05f;
			}
		}

		// ---- Eye smoothing, walk bob and the view matrix ----
		// Exponential decay: never overshoots, no "still stepping" state.
		eyeStepOffset *= std::exp(-deltaT / EYE_SMOOTH_TAU);

		// Walk-bob phase, shared by camera and hands: one value, eased so
		// start/stop doesn't snap.
		bool isWalking = grounded && (std::abs(m.x) > 0.01f || std::abs(m.z) > 0.01f);
		float bobTarget = isWalking ? 1.0f : 0.0f;
		walkBobBlend += (bobTarget - walkBobBlend) * (1.0f - std::exp(-deltaT / WALK_BOB_BLEND_TAU));
		if(isWalking) {
			walkBobPhase += WALK_BOB_SPEED * (sprinting ? 1.4f : 1.0f) * deltaT;
		}

		// Applies bob only to the rendered eye position, not camPos, so
		// collisions/gravity stay unaffected.
		float camBob = std::sin(walkBobPhase * 2.0f) * CAM_BOB_VERTICAL * walkBobBlend;
		glm::vec3 eyePos = camPos - glm::vec3(0.0f, eyeStepOffset - camBob, 0.0f);
		View = glm::lookAt(eyePos, eyePos + front, up);

		// IMPORTANT: View must stay first-person (billboards, held items, culling all read it).
		// Only ViewPrj, what's actually drawn, swaps to the debug camera.
		if(cheats.debugCam) {
			// Snap the debug orbit behind the player's current facing when it turns on.
			if(!dbgCamWasOn) {
				dbgOrbitYaw = camYaw + 180.0f;
				dbgOrbitPitch = DEBUG_CAM_PITCH0;
				dbgOrbitDist = DEBUG_CAM_DIST0;
			}
			if(!overlayOpen()) {
				// Yaw on J / ; -- not J / L, because L toggles the cheat HUD.
				if(glfwGetKey(window, GLFW_KEY_J))
					dbgOrbitYaw -= DEBUG_CAM_ORBIT_SPEED * deltaT;
				if(glfwGetKey(window, GLFW_KEY_SEMICOLON))
					dbgOrbitYaw += DEBUG_CAM_ORBIT_SPEED * deltaT;
				if(glfwGetKey(window, GLFW_KEY_I))
					dbgOrbitPitch += DEBUG_CAM_ORBIT_SPEED * deltaT;
				if(glfwGetKey(window, GLFW_KEY_K))
					dbgOrbitPitch -= DEBUG_CAM_ORBIT_SPEED * deltaT;
				if(glfwGetKey(window, GLFW_KEY_U))
					dbgOrbitDist -= DEBUG_CAM_DOLLY_SPEED * deltaT;
				if(glfwGetKey(window, GLFW_KEY_O))
					dbgOrbitDist += DEBUG_CAM_DOLLY_SPEED * deltaT;
			}
			// Clamp shy of straight up/down (lookAt gimbal) and keep dolly range sane.
			dbgOrbitPitch = glm::clamp(dbgOrbitPitch, -85.0f, 85.0f);
			dbgOrbitDist = glm::clamp(dbgOrbitDist, 3.0f, 90.0f);

			float oy = glm::radians(dbgOrbitYaw);
			float op = glm::radians(dbgOrbitPitch);
			glm::vec3 dbgOffset = glm::vec3(std::cos(oy) * std::cos(op),
										   std::sin(op),
										   std::sin(oy) * std::cos(op)) * dbgOrbitDist;
			glm::vec3 dbgEye = eyePos + dbgOffset;
			// Push the near plane out to just short of the player, clipping
			// away the wall/ceiling shell between the spectator and the room's interior.
			float dbgNear = glm::max(0.2f,
				glm::length(dbgEye - eyePos) - DEBUG_CAM_CLIP_MARGIN);
			glm::mat4 dbgPrj = glm::perspective(FOVy, Ar, dbgNear, farPlane);
			dbgPrj[1][1] *= -1;
			ViewPrj = dbgPrj * glm::lookAt(dbgEye, eyePos, worldUp);
		} else {
			ViewPrj = Prj * View;
		}
		dbgCamWasOn = cheats.debugCam;

		// ---- Held items: wall tuck, torch and key placement ----
		// Camera-space basis for anything rigidly attached to the view (held
		// torch, held key): right/up/-front columns, eyePos translation.
		glm::mat4 camWm = glm::mat4(
			glm::vec4(right, 0.0f),
			glm::vec4(up, 0.0f),
			glm::vec4(-front, 0.0f),
			glm::vec4(eyePos, 1.0f)
		);

		// Wall tuck for both hands, computed once here for both. Off with no collision enabled.
		{
			bool tuckActive = cheats.collisionEnabled;

			float torchTarget = 1.0f;
			if(tuckActive && handTorchInst != nullptr && handTorchCollected && cheats.handTorchEnabled) {
				torchTarget = handFreeReach(camWm, HAND_TORCH_OFFSET, HAND_TUCK_TORCH_PAD);
			}
			advanceReach(handTorchReach, torchTarget, deltaT);

			// Whichever key is drawn in the left hand: held, or sinking after a lock took it (never both in frame).
			int tuckKeyIdx = heldKeyIdx() >= 0 ? heldKeyIdx() : keyLowerIdx;
			float keyTarget = 1.0f;
			if(tuckActive && tuckKeyIdx >= 0) {
				keyTarget = handFreeReach(camWm, pickups[tuckKeyIdx].handOffset, HAND_TUCK_KEY_PAD);
			}
			advanceReach(handKeyReach, keyTarget, deltaT);
		}

		// Held torch: fixed camera-local offset once picked up; before that,
		// its authored floor pose. With "Holding Torch" off it's parked below the map.
		bool torchInHand = handTorchCollected && cheats.handTorchEnabled;
		if(handTorchInst != nullptr && !torchInHand) {
			handTorchInst->Wm = (!handTorchCollected && cheats.handTorchEnabled)
				? handTorchSpawnWm
				: glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
		} else if(handTorchInst != nullptr) {
			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			// Wall tuck first, walk bob on top, folded into a copy of the
			// constants so the two compose. Also fixes the held torch's light,
			// since the flame anchor rides this same Wm.
			glm::vec3 tuckedOffset = HAND_TORCH_OFFSET;
			glm::vec3 tuckedTilt = HAND_TORCH_TILT_DEG;
			applyTuck(handTorchReach, tuckedOffset, tuckedTilt);

			glm::mat4 grip = handGrip(tuckedTilt, bobRollDeg);

			// Pick-up rise, cubic ease-out: only Y moves, so it composes with the tuck and bob.
			torchRaiseElapsed = std::min(torchRaiseElapsed + deltaT, TORCH_RAISE_DURATION);
			float torchT = torchRaiseElapsed / TORCH_RAISE_DURATION;
			float torchEased = 1.0f - (1.0f - torchT) * (1.0f - torchT) * (1.0f - torchT);
			float torchRaiseY = -TORCH_RAISE_DROP * (1.0f - torchEased);

			glm::vec3 bobbedOffset = tuckedOffset + glm::vec3(bobLateral, bobVertical + torchRaiseY, 0.0f);

			handTorchInst->Wm = camWm
				* glm::translate(glm::mat4(1.0f), bobbedOffset)
				* grip
				* glm::scale(glm::mat4(1.0f), glm::vec3(HAND_TORCH_SCALE));
		}

		// Held key: same camera-anchored placement as the torch. Reuses the
		// world instance, pointed at the camera each frame. Only the newest
		// key on the ring is drawn.
		if(heldKeyIdx() >= 0) {
			Pickup &held = pickups[heldKeyIdx()];
			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			// Tilt/offset off the item itself: the ring can hold a key or a book, different origins.
			glm::vec3 tuckedOffset = held.handOffset;
			glm::vec3 tuckedTilt = held.handTiltDeg;
			applyTuck(handKeyReach, tuckedOffset, tuckedTilt);

			glm::mat4 grip = handGrip(tuckedTilt, bobRollDeg);

			// Pick-up rise: only Y moves. Cubic ease-out, not linear (reads mechanical).
			keyRaiseElapsed = std::min(keyRaiseElapsed + deltaT, KEY_RAISE_DURATION);
			float t = keyRaiseElapsed / KEY_RAISE_DURATION;
			float eased = 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
			float raiseY = -KEY_RAISE_DROP * (1.0f - eased);

			glm::vec3 bobbedOffset = tuckedOffset + glm::vec3(bobLateral, bobVertical + raiseY, 0.0f);

			held.inst->Wm = camWm
				* glm::translate(glm::mat4(1.0f), bobbedOffset)
				* grip
				* glm::scale(glm::mat4(1.0f), glm::vec3(held.worldScale));
		}

		// Key spent on a lock: mirror of the rise, played downward, cubic ease-in.
		// Drawn here since the key is already off the ring; parked below the map when the fall ends.
		if(keyLowerIdx >= 0) {
			keyLowerElapsed = std::min(keyLowerElapsed + deltaT, KEY_RAISE_DURATION);
			float t = keyLowerElapsed / KEY_RAISE_DURATION;
			float eased = t * t * t;
			float lowerY = -KEY_RAISE_DROP * eased;

			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			Pickup &sinking = pickups[keyLowerIdx];

			// Tucked too, off the same handKeyReach, or it snaps to the
			// extended pose through the door it was just used on.
			glm::vec3 tuckedOffset = sinking.handOffset;
			glm::vec3 tuckedTilt = sinking.handTiltDeg;
			applyTuck(handKeyReach, tuckedOffset, tuckedTilt);

			glm::mat4 grip = handGrip(tuckedTilt, bobRollDeg);
			glm::vec3 bobbedOffset = tuckedOffset + glm::vec3(bobLateral, bobVertical + lowerY, 0.0f);

			sinking.inst->Wm = camWm
				* glm::translate(glm::mat4(1.0f), bobbedOffset)
				* grip
				* glm::scale(glm::mat4(1.0f), glm::vec3(sinking.worldScale));

			if(keyLowerElapsed >= KEY_RAISE_DURATION) {
				sinking.inst->Wm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
				keyLowerIdx = -1;
			}
		}

		return deltaT;
	}
};


// Main
int main() {
    Castlescape app;

    try {
        app.run(false);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}