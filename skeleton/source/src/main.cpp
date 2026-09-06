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

// The scene is data-driven: these files in assets/scenes/ are read at startup,
// so it can be changed without touching C++.
//
//   scene.json      which models exist and where they are placed
//   colliders.json  collision shapes for models an auto-fitted box gets wrong
//   materials.json  surface parameters, one entry per model
//   lights.json     the light sources and the ambient light
//   flames.json     which models get a flame, and its shape
//   gameplay.json   hunt timings, ghost patrols, and where the run is won
//
// Shaders live in source/shaders/, one folder per job. OVERVIEW.md and notes.md
// at the repo root explain the reasoning behind all of it.

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
	// 1: use a vertical surface's hemispheric ambient instead of this normal's.
	// For interiors, where the sky/ground blend has no meaning.
	int interiorAmbient;
	// Seconds since startup. Rides the per-object UBO rather than the global one,
	// which would shift LightData[]. Only the Flame shaders read it.
	float time;
	// This model's share of indirect light. Negative means inherit
	// gubo.ambientWeight, which is the common case.
	float ambientWeight;
	// 0..1 focus highlight for the instance being looked at. Per-INSTANCE, unlike
	// the fields above, set against gazedInstance in updateUniformBuffer().
	float glow;
	// 1: shade as a metal -- no diffuse lobe, and an indirect term that reflects
	// the room instead of scattering it.
	int metallic;
};

// Everything that's the same for every object drawn this frame. Split from the
// per-instance UBO by change frequency: this is written once, that one 23 times.
// Fixed-size light array plus a live count, since a uniform block needs a
// compile-time size.
struct GlobalUniformBufferObject {
	alignas(16) glm::vec3 eyePos;
	int lightCount;
	// Hemispheric ambient. See AmbientLight in SceneLights.hpp.
	alignas(16) glm::vec3 ambientUpper;
	alignas(16) glm::vec3 ambientLower;
	alignas(16) glm::vec3 ambientDir;
	// LIGHT_DEBUG_* bits from LightConstants.glsl, built from the lighting
	// cheats below. Sits in the 4 bytes std140 pads ambientDir with, exactly
	// like lightCount after eyePos, so the light array still starts at 64.
	int debugFlags;
	// Seconds since startup, for the held torch's flame (Flame.hpp): the one
	// thing in the frame that animates on the GPU rather than being computed
	// here and uploaded.
	float time;
	// The scene's default share of indirect light, 0..1, from lights.json.
	// Rides in the same padding before lights[] that time and debugFlags do,
	// so the array's offset is unchanged.
	float ambientWeight;
	// How much of a point/spot light's radiance comes back as indirect light.
	// See AmbientLight::bounce in SceneLights.hpp.
	//
	// ambientDir ends at 60, debugFlags fills that slot, and
	// time/ambientWeight/this take 64, 68 and 72.
	float ambientBounce;
	// Exponential-squared distance fog (fogFactor = exp(-(fogDensity*dist)^2)
	// in CookTorrance.frag), meant to fade geometry toward black before the
	// GEOM_CULL_* visibility cull above stops drawing it, rather than let it
	// pop out of view -- see updateUniformBuffer() for how this is derived
	// from GEOM_CULL_CONE_DIST. The one scalar that still fits here for
	// free: ambientBounce ends at 76, and the array's own alignas(16) starts
	// it at 80 either way. A second one here would move lights[] and every
	// offset in four shaders with it.
	float fogDensity;
	LightData lights[MAX_LIGHTS];
};

// One torch's cube shadow CAPTURE data (ShadowCube.vert/frag, PShadowCube),
// set 1 there. A uniform buffer and not a push constant, re-mapped every frame
// for every torch including the static ones: the main command buffer is
// recorded once per swapchain image and reused, so a push constant would stay
// frozen at recording time -- fatal for the held torch, which moves each frame.
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

// Shared by all four post passes (bright pass, both blur directions,
// composite). Each fills the fields it cares about and leaves the rest unread.
// No alignas(): two vec2s then scalars is already what std140 lays out.
struct PostUniformBufferObject {
	glm::vec2 texelSize;	// 1/width, 1/height of the SOURCE texture
	glm::vec2 blurDir;		// (1,0) or (0,1); read by BloomBlur.frag only
	float threshold;		// bright pass: luminance above which anything blooms
	float knee;				// bright pass: how soft the threshold's shoulder is
	float bloomIntensity;	// composite: how much bloom is added back
	float exposure;			// composite
	int debugFlags;			// composite: LIGHT_DEBUG_NO_TONEMAP
	float time;
	// composite: 0 normally, ramping to 1 as the player escapes, blending the
	// frame towards white. A term of its own, not more `exposure`: the tone
	// map approaches 1 asymptotically, so exposure alone only greys the frame
	// out. The exposure ramp blows the scene out; this finishes it.
	float escapeFlash;
	// composite: 0 normally, ramping to 1 as the camera sinks into a ghost.
	// See SPECTRAL_VEIL_OUTER.
	float spectralVeil;
};

// A full-screen quad vertex for those passes. Only a position: Post.vert
// derives its UV from it.
struct PostVertex {
	glm::vec2 pos;
};

// Cheap 1D value noise for the torch fire envelope in updateUniformBuffer().
// On the CPU because the flicker is shared by the flame, its sparks and the
// cast point light (TorchFlame), and only the CPU sees all three. Integer
// hash, not fract(sin(x)*...): that one decorrelates badly at small inputs.
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

// MAIN !

class Castlescape : public BaseProject {
	protected:
	// Here you list all the Vulkan objects you need:
	
	// Descriptor Layouts [what will be passed to the shaders]
	DescriptorSetLayout DSLlocal, DSLglobal;

	// Vertex formants, Pipelines [Shader couples] and Render passes
	VertexDescriptor VD;
	// The scene pass. Renders into an offscreen FLOATING-POINT colour
	// attachment, not the swapchain, which is what makes bloom possible. See
	// buildHdrAttachments().
	RenderPass RP;
	Pipeline P;

	// The ghosts' pipeline, drawn into the SAME pass as P right after it.
	// Separate because a ghost is not a surface: Spectral.frag computes no BRDF
	// and reads no shadow map, and it needs alpha blending, a
	// pipeline-creation flag. Shares P's vertex shader and DSLlocal, so the
	// ghosts ride the same per-instance UBO as every other prop.
	Pipeline Pspectral;

	// The ghosts' DEPTH PREPASS (SpectralDepth.frag), drawn over the same
	// instances immediately before Pspectral. It writes the depth of the nearest
	// ghost surface and leaves the colour attachment untouched, so the colour
	// pass can reject the ghost's own interior -- the feet inside the robe --
	// instead of blending it under the body.
	//w
	// Same DSLs as Pspectral, and the default VK_COMPARE_OP_LESS rather than
	// Pspectral's LESS_OR_EQUAL, which is the whole point: LESS is what leaves
	// the minimum in the depth buffer.
	Pipeline PspectralDepth;

	// Shadow mapping, CUBE branch (the torches): a real 6-face cube map per
	// point light (CubeShadowMap.hpp: linear-distance storage, one flat bias).
	//
	// RPShadowCubeCompat exists ONLY to mint a VkRenderPass compatible with the
	// per-face framebuffers -- createRenderPass() is private, so a full
	// RenderPass builds one and we read .renderPass back out. Its own
	// attachment is never used. The per-face framebuffers are built by hand in
	// createCubeShadowMaps(): they attach single-layer views into a 6-layer
	// cube image, which FrameBufferAttachment can't do.
	RenderPass RPShadowCubeCompat;
	Pipeline PShadowCube;
	CubeShadowMap torchCube[NUM_SHADOW_CUBES];
	// Shared by every torch's cube view: all R32_SFLOAT at one resolution, so
	// one CLAMP_TO_EDGE/linear sampler suffices. Starter.hpp's TextureSampler
	// rather than a raw VkSampler.
	TextureSampler cubeShadowSampler;

	// set 1 for the cube capture pass -- one UBO per cube slot, holding that
	// torch's 6 face view-projections plus its world position. Its own
	// DescriptorSet member: per slot, not per scene instance.
	DescriptorSetLayout DSLshadowCubeCapture;
	DescriptorSet DSshadowCube[NUM_SHADOW_CUBES];

	// set 2 for the main pass's shadow sampling: one sampler binding per
	// torch cube map, read by CookTorrance.frag's shadowFactor().
	// DSLlocal/DSLglobal stay set 1/0.
	//
	// No DescriptorSet member of its own: this one rides Scene's ordinary
	// per-instance machinery, so every CookTorrance instance gets an identical,
	// redundant copy. Wasteful but cheap at this instance count, and it avoids
	// hand-rolling a THIRD way to bind a descriptor set.
	DescriptorSetLayout DSLshadowSample;
	// The six face view-projection matrices for each torch's cube map,
	// index-matched [LightData::shadowIndex][face] (face order: see
	// CUBE_FACE_DIR in CubeShadowMap.hpp). Computed once, same reasoning.
	glm::mat4 torchFaceMatrices[NUM_SHADOW_CUBES][6];
	// World position of each cube-mapped torch, index-matched to shadowIndex.
	// Handed to the push constant by populateCommandBuffer().
	glm::vec3 torchLightPos[NUM_SHADOW_CUBES];
	// How many of the above are populated -- fewer than NUM_SHADOW_CUBES if
	// lights.json authors fewer shadow point lights than slots. Set by
	// computeShadowMatrices(), bumped in localInit() for the held torch.
	int activeCubeShadows = 0;
	// SHADOW_CUBE_RES rather than a literal: CookTorrance.frag derives the cube
	// path's depth bias from the world size of one texel of this map, so the
	// shader has to know the same number. See LightConstants.glsl.
	static constexpr int SHADOW_MAP_RES = SHADOW_CUBE_RES;
	// Far clip for every torch cube map, and the clear value each face is reset
	// to, so "nothing drawn" reads as unlit. This must stay past anything
	// shadowFromCube() can be asked about, or a wall beyond it takes the clear
	// value as its nearest occluder and reads as falsely shadowed.
	static constexpr float TORCH_SHADOW_FAR_CONST = 60.0f;
	// Near clip: close enough that only the torch fixture falls inside it. A
	// member because updateHandTorchShadow() needs the same number every frame.
	static constexpr float TORCH_SHADOW_NEAR_CONST = 0.05f;
	// The slot reserved for the held torch, which never goes through lights.json:
	// its Wm means nothing until GameLogic() starts rewriting it every frame, so
	// it can get neither a fixed shadowIndex nor fixed face matrices.
	static constexpr int HAND_TORCH_SHADOW_INDEX = NUM_SHADOW_CUBES - 1;

	// Start of the DYNAMIC pool: everything between lights.json's fixed slots and
	// the held torch's reserved last one. Counted at startup, not a literal, so
	// it survives a change to lights.json's own torch count.
	int dynamicShadowSlotBase = 0;
	// Index into torchFlames for whichever flame holds each dynamic slot, or -1
	// if empty. Sized NUM_SHADOW_CUBES for simplicity; only the dynamic range
	// is ever touched.
	std::array<int, NUM_SHADOW_CUBES> dynamicSlotOccupant{};

	// The occupant identity each slot last rendered its six faces for. Diffed
	// against the current one every frame; a mismatch queues all six faces.
	// Static point lights therefore render exactly once for the program's life.
	// SHADOW_SLOT_UNSET means never rendered, which every slot starts as: the
	// image would otherwise stay in VK_IMAGE_LAYOUT_UNDEFINED, which a
	// samplerCube descriptor may not be bound against. Hence one render even for
	// a slot that stays empty forever.
	static constexpr int SHADOW_SLOT_UNSET = -2;
	std::array<int, NUM_SHADOW_CUBES> lastRenderedOccupant;
	// What the diffs found stale this frame, as a bitmask of FACES per slot.
	// Per face and not per slot because a cube map is six independent images: at
	// 1024^2 R32 each, redrawing all six to fix one is mostly clear cost. A slot
	// whose LIGHT changed still gets all six -- nothing about its old content
	// survives moving the camera it was shot from.
	std::array<uint8_t, NUM_SHADOW_CUBES> pendingFaceMask{};
	static constexpr uint8_t ALL_CUBE_FACES = 0x3F;

	// ---- The per-frame cube-shadow submission ----
	// Its own pool: the framework's commandPool has flags = 0, and a buffer from
	// a pool without RESET_COMMAND_BUFFER_BIT may not be re-recorded.
	VkCommandPool shadowCommandPool = VK_NULL_HANDLE;
	// One buffer and fence per swapchain image, since re-recording a buffer the
	// GPU is still reading is undefined behaviour. Ours rather than
	// inFlightFences: those are signalled by the MAIN submit, and a later submit
	// finishing does not prove an earlier one did.
	std::vector<VkCommandBuffer> shadowCB;
	std::vector<VkFence> shadowCBFence;
	// Where the held torch's light was, and whether it was casting at all, when
	// its cube was last captured. Its faces are only redrawn when one of the
	// two changed (or a mover marked them), which is what turns a player
	// standing still into no shadow work at all. The sentinel is a position
	// nothing can occupy, so the first frame always captures.
	glm::vec3 lastHandTorchCapturePos{std::numeric_limits<float>::infinity()};
	bool lastHandTorchCaptureOccupied = false;

	// The occupant diff only answers "did this slot's LIGHT change hands", which
	// is enough while every occluder is nailed down. Ghosts and swinging doors
	// are not, and with the sun gone the cube maps are the only shadow they cast.
	//
	// Candidates are the ghosts and door leaves, NOT every instance whose Wm
	// changes (a spinning pickup key would re-render every slot near it
	// forever). Filtered to the ones materials.json marks castsShadow, which
	// today means the doors -- the ghosts cost more than their shadows are
	// worth and are switched off in materials.json. Built once on first use.
	std::vector<Instance *> movingOccluders;
	// Wm each of those had when a slot was last re-captured for it, so a mover
	// standing still costs one mat4 compare and no draws.
	std::vector<glm::mat4> movingOccluderWm;
	bool moverListBuilt = false;
	// Which faces had a mover at their last capture. Needed for the faces a mover
	// LEAVES: nothing about a ghost's new position tells the face it walked out
	// of that it still has the ghost painted on it.
	std::array<uint8_t, NUM_SHADOW_CUBES> slotFaceHadMover{};
	// How far each slot's light still matters, derived from its OWN falloff
	// rather than picked as a radius. Written wherever torchLightPos[] is.
	std::array<float, NUM_SHADOW_CUBES> torchShadowReach{};

	// Time between updateDynamicShadowSlots() calls: the candidates are static,
	// only the player moves. Starts equal to the interval so the first frame
	// fires it immediately, rather than rendering uninitialised matrices.
	float shadowReassignTimer = 0.3f;
	static constexpr float SHADOW_REASSIGN_INTERVAL = 0.3f;
	// A waiting candidate must beat the current occupant by this factor to take
	// its slot. Without a margin a player standing on the boundary between two
	// candidates would flip it, and force a fresh render, on every re-evaluation.
	static constexpr float SHADOW_SWAP_MARGIN = 1.15f;
	// Reach margin an EXISTING shadow-cube occupant is judged by in the
	// view-cone cull (lightReachesViewCone), vs. 1.0 for a fresh claimant. Same
	// job SHADOW_SWAP_MARGIN does for the distance contest: a torch sitting on
	// the cone boundary while the player turns past it shouldn't surrender and
	// re-render its cube every re-evaluation.
	static constexpr float SHADOW_VIEW_KEEP_MARGIN = 1.15f;

	// Models, textures and Descriptors (values assigned to the uniforms)
	DescriptorSet DSglobal;

	// ---- HDR post-processing chain ----
	// scene (RGBA16F, MSAA + resolve) -> bright pass -> blur H -> blur V (all
	// quarter res) -> composite (swapchain: scene + bloom, then tone map).
	//
	// All recorded into the "main" buffer in that order; ordering between them
	// comes from ATDEP_SIMPLE's subpass dependencies, not manual barriers.
	RenderPass RPbright, RPblurH, RPblurV, RPcomposite;
	VertexDescriptor VDpost;
	// One layout for the passes that read a single texture, one for the
	// composite, which reads the scene and the bloom result.
	DescriptorSetLayout DSLpost1, DSLpost2;
	Pipeline Pbright, PblurH, PblurV, Pcomposite;
	DescriptorSet DSbright, DSblurH, DSblurV, DScomposite;
	Model *Mpost = nullptr;

	// The attachment descriptions each pass is built from. Members, not locals:
	// RenderPass::init copies the vector but the per-attachment code points
	// back at the copy for the pass's lifetime, and onWindowResize() rebuilds
	// them at the new size.
	std::vector<AttachmentProperties> hdrAtt, brightAtt, blurHAtt, blurVAtt, compositeAtt;

	// The bloom chain runs at 1/BLOOM_DIV per axis. A fixed-width gaussian on a
	// quarter-res image covers 4x more of the final picture, so this is how a
	// cheap 9-tap kernel makes a wide soft halo.
	static constexpr int BLOOM_DIV = 4;

	// The 3D scene renders at this fraction of the window resolution;
	// Composite.frag upsamples it back. The UI stays at real resolution.
	float renderScale = 0.8f;

	// Clamped to >= 1 (a minimised window can't ask for a zero-sized image).
	// Takes the window size as a parameter -- onWindowResize() computes it from
	// the NEW size before swapChainExtent catches up.
	int renderWidth(int windowW) const {
		return std::max(1, (int)std::lround(windowW * renderScale));
	}
	int renderHeight(int windowH) const {
		return std::max(1, (int)std::lround(windowH * renderScale));
	}

	// MSAA sample count as a log2 level (0 -> 1x, 2 -> 4x), because the slider's
	// fixed +/-1 step can't express the doubling Vulkan sample counts need.
	float msaaLevel = 2.0f;
	// Upper bound, from getMaxUsableSampleCount(): never offer 16x to a GPU
	// that can't do it.
	float maxMsaaLevel = 2.0f;

	// Replays a resize's rebuild path at the current window size, so only the
	// internal render resolution changes. Skipped while a rebuild is pending:
	// stacking a second target size once began a render pass against a size
	// newer than its framebuffer.
	void applyRenderScaleChange() {
		if(!framebufferResized) {
			onWindowResize((int)windowWidth, (int)windowHeight);
			RebuildPipeline();
		}
	}

	// Shows the real pixel resolution beside the percentage. Reads
	// renderWidth()/renderHeight(), so it can't drift from what is rendered.
	// v is unused: this always formats the current scale.
	std::string formatRenderScale(float /*v*/) {
		char buf[32];
		snprintf(buf, sizeof(buf), "< %d%% (%dx%d) >",
				 (int)std::lround(renderScale * 100.0f),
				 renderWidth((int)windowWidth), renderHeight((int)windowHeight));
		return std::string(buf);
	}

	// Converts the level back to a VkSampleCountFlagBits, then regenerates the
	// attachment properties -- a sample-count change, unlike renderScale's plain
	// width/height one, has to rebuild hdrAtt itself. Same pending guard.
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

	// Bright-pass threshold/knee in luminance. Above 1.0, not at it: a sunlit
	// pale wall lands either side of 1.0 once sun and ambient are added, so 1.0
	// haloed every bright surface. The flame writes ~6 and sparks higher, so
	// only genuinely emissive things reach the bloom buffer.
	static constexpr float BLOOM_THRESHOLD = 1.55f;
	static constexpr float BLOOM_KNEE = 0.45f;
	static constexpr float BLOOM_INTENSITY = 0.65f;
	static constexpr float SCENE_EXPOSURE = 1.0f;

	// To support loading assets from a scene.json file
	Scene SC;
	std::vector<VertexDescriptorRef>  VDRs;
	std::vector<TechniqueRef> PRs;

	// to provide textual feedback
	TextMaker txt;

	// Flat-colored quads: background/highlight panel behind the cheat HUD's text.
	UiQuad uiQuad;

	// Flat-colored quad: the center-screen dot for aiming look-based
	// interactions (GameLogic()'s gaze test). Its own UiQuad instance.
	UiQuad crosshair;

	// (Re)builds the crosshair dot centered on the window. Called at init and
	// on resize; its pixel position depends on screen size only.
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

	// Its own UiQuad instance: dim overlay + button backgrounds behind the
	// pause menu's text.
	UiQuad pauseQuad;
	// The pause menu: dims the screen and freezes GameLogic() while open
	// (overlayOpen() gating). ESC toggles it.
	PauseMenu pauseMenu;

	// Its own UiQuad instance: opaque backdrop + buttons for the launch screen.
	UiQuad startScreenQuad;
	// The launch screen: open from the first frame, so the app boots here
	// instead of into the castle. Also reopened when Quit abandons a run.
	// Folded into overlayOpen() like the pause menu.
	StartScreen startScreen;

	// Its own UiQuad instance: opaque backdrop + rows for the settings screen.
	UiQuad settingsQuad;
	// The settings screen: Render Scale / MSAA, the same two sliders CheatHud
	// controls. Reachable from StartScreen's or PauseMenu's "Settings" button;
	// settingsFromPause remembers which to reopen on Back. In overlayOpen().
	SettingsMenu settingsMenu;
	// True if settingsMenu was opened from PauseMenu (mid-run), false from
	// StartScreen.
	bool settingsFromPause = false;

	// Other application parameters
	float Ar;	// Aspect ratio

	glm::mat4 ViewPrj;
	glm::mat4 View;

	// Free-look camera state (position + orientation), persisted across frames.
	// Spawns inside the dungeon hall (dh), clear of the table and both torches,
	// now that the castle courtyard is gone -- there's no outdoor approach
	// to walk in from anymore.
	glm::vec3 camPos = glm::vec3(-20.2f, 1.8f, 18.0f);
	// Yaw: rotation around world up axis, in degrees.
	// yaw=0 faces +X; increasing yaw turns right, decreasing turns left.
	// Faces +X so spawning looks straight down the hall toward the far door.
	float camYaw = 0.0f;
	// Pitch, degrees: -90 down, +90 up.
	float camPitch = -10.0f;
	// Vertical speed from gravity, units/s. Negative = falling. Reset to 0 by
	// the ground clamp on landing.
	float camVerticalVelocity = 0.0f;

	// Debug spectator orbit around the player (cheats.debugCam). Snapped
	// behind the player each time the camera is switched on, then nudged by
	// IJKL (orbit) and U/O (dolly). Yaw is a world angle in degrees, pitch is
	// degrees above the player, dist is metres.
	float dbgOrbitYaw = 0.0f;
	float dbgOrbitPitch = DEBUG_CAM_PITCH0;
	float dbgOrbitDist = DEBUG_CAM_DIST0;
	bool dbgCamWasOn = false;

	// Top of the "floor" instance, cached after load. Last-resort clamp while
	// no-clipping, so falling under the map is never possible.
	float worldFloorY = 0.0f;

	// Hand-authored collision geometry from colliders.json, merged with what
	// scene.json built.
	SceneColliders colliderSet;

	// Per-model BRDF parameters from materials.json. See SceneMaterials.hpp.
	SceneMaterials materials;

	// The scene's light sources from lights.json. See SceneLights.hpp.
	SceneLights sceneLights;

	// Flat list of every collider gameplay collides against, taken from
	// colliderSet after load, so the per-frame loops read a plain vector.
	std::vector<Collider *> allColliders;

	// Uniform XZ grid over allColliders, built once (colliders never move).
	// Lets the ghost queries below test only nearby colliders instead of
	// scanning the whole castle every frame, per ghost.
	struct ColliderGrid {
		// World units per cell. Bigger than a ghost's radius so the 3x3
		// neighbourhood queried below always covers a ghost-sized radius test
		// without needing to grow each collider's footprint by the query
		// radius.
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

		// Visits the cached extents of every collider registered in the 3x3
		// cell neighbourhood around `p`. Always the 3x3 block rather than just
		// p's own cell: the callers ask "is anything within some radius of
		// p", and a collider registered one cell over can still be that
		// close. A collider that spans several of those cells is visited once
		// per cell it's in, which costs a few redundant (and cheap) AABB
		// tests rather than needing a per-query dedup pass.
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

	// Colliders left OUT of the grid above because their Wm changes every
	// frame -- door leaves and their lock hardware (see GameLogic()'s
	// d.inst->C->setWorldMatrix() calls). A cached AABB for one of these
	// would still be blocking a ghost long after the door swung open, so
	// they're tested live instead, on top of the grid lookup. Short list, so
	// a plain scan of it stays cheap.
	std::vector<Collider *> ghostDynamicColliders;

	// Every ghost collider query goes through here: the static grid first,
	// then the movers above, read fresh each time.
	template<typename F>
	void ghostForEachNearbyCollider(const glm::vec3 &p, F &&fn) const {
		ghostColliderGrid.forEachNear(p, fn);
		for(Collider *c : ghostDynamicColliders) {
			fn(c->getExtents());
		}
	}

	// Debug/cheat toggles, isolated in a utility struct.
	// Not persisted across runs, reset to default values on launch.
	struct CheatFlags {
		bool collisionEnabled = true;   // false = no-clip
		// Live readout of the camera's world position/yaw, for hand-placing
		// scene.json objects. No gameplay to preserve, so defaults off.
		bool showCoordinates = false;

		// Whether a hunting ghost ending the run. Off, the hunt still plays
		// out, you just can't lose -- useful while tuning it.
		bool ghostsCanCatch = true;

		// The two flame switches, read by flameBurning() (which every
		// flame-showing path goes through: point light, billboard, shadow
		// candidacy, glare). Here and not in SceneLights because the flames
		// never go through lights.json -- their lights are appended into gubo
		// from updateUniformBuffer() off the flame's own position. Torches =
		// the wall-mounted ones; candles are separate and stay lit.
		bool roomTorchesEnabled = true;
		// The held torch: off hides the model too, since a dark stick in front
		// of the camera isn't "no torch in hand".
		bool handTorchEnabled = true;

		// Lighting debug views, resolved into gubo.debugFlags and read by
		// CookTorrance.frag. No "legit" state to preserve, so each defaults to
		// leaving the picture as authored. The light SOURCE switches (sun,
		// spot, ambient) live in SceneLights instead.
		bool unlit = false;          // albedo only -- "dark texture" vs "no light"
		bool showNormals = false;    // the shading normal as a color; what flatNormals is for
		// Off suppresses the aura on the gazed target. The cue goes, but [E]
		// and the prompt stay -- for viewing the scene's own lighting on a
		// door without an aura on top. Defaults on.
		bool focusGlowEnabled = true;
		bool specularEnabled = true; // off forces k to 1 -- highlight vs bright surface
		bool toneMapEnabled = true;  // off clips overexposure to white instead of compressing
		// Off forces shadowFactor() to 1 while still rendering the shadow
		// passes: splits "shadow sampling artifact" from "geometry artifact".
		bool shadowsEnabled = true;

		// Per-flame-TYPE shadow casting, read by updateDynamicShadowSlots()
		// and the light-append loop in updateUniformBuffer(). Distinct from
		// shadowsEnabled above: that one forces shadowFactor() to 1 for
		// every already-cast shadow (a shading diagnostic), these two decide
		// which flames COMPETE for a cube-shadow slot in the first place --
		// off means that category never casts a shadow at all, not merely
		// that its shadow renders as if absent. Torches and candles rather
		// than per-instance: same reasoning as SceneLights' directEnabled/
		// pointEnabled/spotEnabled, there's no use case for singling out one
		// specific torch. Both default on, matching the authored scene.
		bool torchShadowsEnabled = true;
		bool candleShadowsEnabled = true;

		// Whether the held torch's own MESH occludes other lights' shadows
		// (the cube maps of the wall torches/candles it walks past) while
		// it's actually in the player's hand. Its floor pose is unaffected --
		// see the handTorchInst checks in recordCubeSlotFaces and
		// queueMoverCubeSlotRenders -- and always occludes normally there,
		// the same as any other static prop, because that pose is static and
		// nothing about it is camera-anchored.
		//
		// Off by default: nothing occupies the hand but the torch itself, so
		// a shadow it throws while carried would read as the torch floating
		// in mid-air with no arm or body to explain its shape -- worse than
		// no shadow at all. A HUD row so it can be compared without a recompile
		// once there's a player model/arm to anchor the shape.
		bool handTorchModelCastsShadowWhenHeld = false;

		// Geometry overlays (DebugLines.hpp): light-position gizmos, and
		// wireframe boxes at each torch's shadow-cube clip distances.
		bool showLightGizmos = false;
		bool showShadowFrustums = false;
		// Wireframe box around every gameplay collider plus every ramp's quad.
		// The one view that answers "is this wall solid where it looks solid"
		// without walking into it. Drawn from getExtents() (world-space
		// axis-aligned), so it shows the same envelope the ground pass
		// collides against -- an OOBB as its fattened box, not its true one.
		bool showColliders = false;
		// Third-person spectator view: pulls the rendered camera far back from
		// the player while every visibility cull (geometry cone, torch-light
		// list, shadow pool) keeps running from the real first-person eye. The
		// cull boundary -- instances popping in and out at the cone edge -- is
		// then visible from outside. DebugLines sketches the cone itself.
		bool debugCam = false;
		// Shader-side: recolors surfaces by incoming light intensity. See
		// LIGHT_DEBUG_HEATMAP.
		bool showLightHeatmap = false;
	} cheats;

	// Numeric tuning for the movement cheats -- "how strong", not "on/off", so
	// a separate struct. Also not persisted.
	struct MovementParams {
		// World units traveled per second
		float moveSpeed = 3.0f;
		// Multiplier applied to moveSpeed while sprinting
		float sprintMultiplier = 2.0f;
		// Initial upward velocity on jump, world units/second
		float jumpSpeed = 5.0f;
		// Downward acceleration, world units/second^2 (negative = down)
		float gravity = -9.81f;
	} movement;

	// A door leaf (its own instance) that swings around a vertical hinge on E.
	// A list, not one hardcoded door, so adding one is a single addDoor() call.
	//
	// SM_Door_01's local origin is AT the hinge edge (the panel hangs to one
	// side of Z=0), so the instance's authored transform already IS the
	// closed-door hinge frame and "open" is one extra rotation about world Y
	// appended to it -- no separate hinge point to compute.
	struct Door {
		std::string instanceId;
		Instance *inst = nullptr;
		glm::mat4 baseWm{1.0f};	// authored (closed) world matrix
		glm::vec3 promptPos{0.0f};	// interact-range point: the doorway centre, not the hinge
		// The same point in the leaf's LOCAL frame. promptPos is the fixed
		// doorway; leafPos() below is where the panel's centre has swung to,
		// and the two are tested as alternatives so a wide-open leaf can still
		// be aimed at to close it.
		glm::vec3 promptOffset{0.0f};
		glm::vec3 leafPos() const {
			return inst == nullptr ? promptPos
								   : glm::vec3(inst->Wm * glm::vec4(promptOffset, 1.0f));
		}
		// Open travel. Only the MAGNITUDE is used at runtime; the direction is
		// decided per opening by swingSignAwayFrom(). The authored sign is the
		// fallback for a leaf whose geometry can't be read.
		float openAngleDeg = 100.0f;
		bool open = false;
		float angle = 0.0f;	// current animated angle, eases toward the target
		// Current swing direction, a sign on |openAngleDeg|. Only recomputed
		// while the door is fully closed: flipping it mid-swing would sweep the
		// leaf through its own frame.
		float swingSign = 1.0f;
		// Padlock. Empty lockKeyId = no lock (E just toggles). A non-empty id
		// means the door won't budge without a carried key whose keyId matches
		// -- by id, not "any key", so a two-key level can't be opened in the
		// wrong order.
		std::string lockKeyId;
		// Human-readable name for the locked prompt. Defaults to lockKeyId.
		std::string lockLabel;
		// Per-door wording for the locked prompts, or empty for the generic
		// padlock lines. Empty on every chained castle door; set on the secret
		// bookcase, where "Locked - needs the iron key" would give the trick
		// away (the gap on the shelf IS the keyhole).
		std::string promptReady;	// carrying what it wants
		std::string promptMissing;	// not carrying it
		std::string promptBlocked;	// standing on the far side of the lock
		// A door that doesn't admit to being one until it can be opened. While
		// set AND still locked AND the player isn't carrying the key, it's not
		// a gaze target at all (doorIsHidden): no aura, no prompt. On a
		// padlock the red aura is fine (the player sees the padlock); on a
		// bookcase it would point straight at the secret.
		bool secret = false;
		// True while the padlock holds. Set from lockKeyId at load and in
		// restartRun(), cleared for good once a matching key is spent -- keys
		// are one-shot, so an unlocked door must never re-lock.
		bool locked = false;
		// The visible padlock (SM_DoorChains_01 + SM_Padlock_01, or anything
		// from addLockProp()). Modelled in the LEAF's local frame, so their
		// world matrix is the leaf's times `local`. Shown while locked, parked
		// below the map once the key is spent.
		struct LockProp {
			Instance *inst;
			// Extra transform in the leaf's local frame. Identity, or a half
			// turn for a door approached from the other side (addLockProp's `flip`).
			glm::mat4 local;
			// Inverts the effect: hardware that appears when the lock comes
			// OFF. One user, the secret bookcase -- a book handed over ends up
			// in the gap on the shelf. With it set the prop is left alone
			// while locked (it's still a Pickup the pickup code owns), not
			// parked underground.
			bool whenUnlocked = false;
		};
		std::vector<LockProp> lockProps;
		// How far past its mesh a lock prop's collider reaches on the hardware
		// face: the stand-off keeping the held torch out of the chains.
		static constexpr float LOCK_PROP_KEEPOUT = 0.45f;
		// Which face the hardware is on, a sign on the leaf's local X: +1 as
		// make_door_lock.py exports, -1 for the flipped copy. The padlock is
		// reachable only from its own face, so a player behind the door meets
		// one that won't move, key or no key.
		float lockFaceSign = 1.0f;
		// True when `p` stands on the padlock's face. Against baseWm, not the
		// live Wm: a locked door never swings. XZ only -- height has no say.
		bool onLockSide(const glm::vec3 &p) const {
			glm::vec3 axis(baseWm[0].x, 0.0f, baseWm[0].z);	// leaf local +X, in world
			if(glm::length(axis) < 1e-6f) return true;	// degenerate: don't lock anyone out
			glm::vec3 d(p.x - promptPos.x, 0.0f, p.z - promptPos.z);
			return glm::dot(d, glm::normalize(axis)) * lockFaceSign > 0.0f;
		}
		// Sign on |openAngleDeg| that swings the leaf AWAY from a player at
		// `p`, so the door always opens outward, never into their face.
		//
		// The closed leaf's plane has local +X as normal, which answers both
		// "which side is the player" and "which side did the panel swing to".
		// Rather than reason about a cross product through an arbitrary
		// transform, rotate a panel point the positive way and check where it
		// lands. XZ only, like onLockSide.
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
	// How far a door can still be a gaze candidate. Larger than
	// DOOR_INTERACT_RADIUS, which still gates actual interaction once aimed at.
	static constexpr float DOOR_LOOK_DISTANCE = 9.0f;
	// Doorway half-width, turning promptPos into an aiming tolerance -- wide.
	static constexpr float DOOR_AIM_RADIUS = 1.2f;

	// Edge-detection for E, so holding it doesn't toggle every frame.
	bool interactKeyWasPressed = false;
	// Index into `doors` of the one in range, or -1. Set in GameLogic(), read
	// by updateUniformBuffer() for the "[E] Interact" prompt.
	int nearbyDoor = -1;

	// A world object collected with [E]. Same list reasoning as Door. Not
	// strictly one-way: the key can be dropped again (G).
	struct Pickup {
		std::string instanceId;
		Instance *inst = nullptr;
		glm::vec3 worldPos{0.0f};	// measured at load, for the in-range check
		bool collected = false;
		// The authored pose, so restartRun() can put a picked-up/dropped item
		// back where scene.json placed it (worldPos gets overwritten on drop).
		glm::mat4 spawnWm{1.0f};
		glm::vec3 spawnPos{0.0f};
		// Non-empty => this pickup IS a key, opening every Door with a matching
		// lockKeyId (plus the exit). Empty => an ordinary carried item.
		std::string keyId;
		// Spent on a lock and gone for the run. Distinct from `collected`: a
		// collected key can still be dropped or spent, a consumed one is off
		// the board until restartRun().
		bool consumed = false;
		// Uniform world scale, read from the authored matrix at load, so
		// scene.json's "scale" stays the one place it's written. Per-pickup
		// (several keys need not be one size).
		float worldScale = 1.0f;
		// Euler angles (deg) orienting this item in the hand. Per-pickup
		// because the grip is a fact about the MESH's axes: the key's long
		// axis is local Z and swings upright, the book lies flat and tilts up
		// to be read. addPickup() defaults it to HAND_KEY_TILT_DEG.
		glm::vec3 handTiltDeg{0.0f};
		// Where the item hangs relative to the eye, camera space. Per-pickup:
		// it positions the mesh ORIGIN, and the key's is near one end while the
		// book's is at its middle, so the same offset rides them differently.
		// addPickup() defaults it to HAND_KEY_OFFSET.
		glm::vec3 handOffset{0.0f};
	};
	std::vector<Pickup> pickups;
	// The key ring: indices into `pickups`, in collection order. A vector, not
	// a set of ids, because two keys can share an id and the ring must
	// remember which instance each was. keyRing.back() is drawn in the hand.
	std::vector<int> keyRing;
	// 3D check (unlike DOOR_INTERACT_RADIUS's XZ): a pickup can be at table height.
	static constexpr float PICKUP_INTERACT_RADIUS = 4.0f;
	// Index into `pickups` of the one in range, or -1. Checked before
	// nearbyDoor -- grabbing should win over interacting with what's behind it.
	int nearbyPickup = -1;
	static constexpr float PICKUP_LOOK_DISTANCE = 6.0f;	// gaze-candidate range
	// Pickup half-width for aiming tolerance -- tight, they're small props.
	static constexpr float PICKUP_AIM_RADIUS = 0.35f;

	// Base half-angle of the aiming cone, before an object's aim radius widens
	// it at range. Shared by doors and pickups.
	static constexpr float GAZE_CONE_DEG = 7.0f;

	// The Instance* the crosshair rests on within interact range, or nullptr.
	// Resolved once per frame in GameLogic(), read by updateUniformBuffer() to
	// set ubo.glow.
	Instance *gazedInstance = nullptr;
	// True when gazedInstance is aimed at but [E] would do nothing (a locked
	// door with no key, or approached from the wrong side). General so a future
	// interactable can flip it too. Flips ubo.glow's sign, swapping the aura
	// gold -> red.
	bool gazedInteractionDisabled = false;
	// What kind of thing gazedInstance is, so CookTorrance.frag can pick a
	// distinct aura color per category (doors gold, pickups blue/purple)
	// instead of every interactable looking the same. Encoded as
	// ubo.glow's magnitude (1 = Door, 2 = Pickup, 3 = Candle, 4 = WallTorch)
	// alongside the sign for gazedInteractionDisabled -- see the assignment in
	// updateUniformBuffer(). A plain enum rather than a bool, which is what
	// let Candle and WallTorch slot in later without renaming anything.
	enum class GlowKind { Door = 1, Pickup = 2, Candle = 3, WallTorch = 4 };
	GlowKind gazedGlowKind = GlowKind::Door;

	// True if `front` is aimed closely enough at `target`: within lookDist and
	// within a cone whose half-angle is GAZE_CONE_DEG widened by the angular
	// size aimRadius subtends (so a wide door forgives worse aim than a small
	// pickup). cosAngleOut lets a caller track the best candidate.
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

	// True for a secret door the player has no business seeing yet. Gated
	// inside findGazedDoor so the aura, prompt and E key all read the one
	// nearbyDoor this decides.
	bool doorIsHidden(const Door &d) const {
		return d.secret && d.locked && findKeyInRing(d.lockKeyId) < 0;
	}

	// Index into `doors` aimed at within look range, or -1. Does NOT check
	// DOOR_INTERACT_RADIUS -- the caller does, keeping "targetable" and "close
	// enough" as two separate gates.
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
	// Slot IN keyRing of a carried key matching `id`, or -1. Empty `id` = "any
	// key" (the exit, when gameplay.json names none). Back to front, so the key
	// in hand is spent first.
	int findKeyInRing(const std::string &id) const {
		for(int slot = (int)keyRing.size() - 1; slot >= 0; slot--) {
			if(id.empty() || pickups[keyRing[slot]].keyId == id) return slot;
		}
		return -1;
	}
	// Spend a carried key: off the ring, off the board, permanently for this
	// run. `slot` indexes keyRing, not pickups. Parking the instance below the
	// map is how every mesh here is hidden (Starter draws every instance, no
	// visibility flag).
	void consumeKey(int slot) {
		if(slot < 0 || slot >= (int)keyRing.size()) return;
		int idx = keyRing[slot];
		Pickup &p = pickups[idx];
		p.consumed = true;
		p.collected = true;
		keyRing.erase(keyRing.begin() + slot);
		// Off the ring now, but not parked yet: it sinks out of frame first
		// (the mirror of the pick-up rise), and the lowering block in
		// GameLogic() parks it when the animation ends. Only one key can
		// animate, so an already-sinking one is parked now.
		if(keyLowerIdx >= 0) {
			pickups[keyLowerIdx].inst->Wm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
		}
		keyLowerIdx = idx;
		keyLowerElapsed = 0.0f;
	}
	// Put a carried key back in the world: off the ring, lying flat in front
	// of the player, so walking up shows the pick-up prompt again. `slot`
	// indexes keyRing. `fwdDist` is how far ahead it lands -- arm's length for
	// the manual drop (G), closer for the automatic one that frees the hand.
	// Re-parks the same instance the held-key block was drawing; here rather
	// than inline so both callers agree on the pose.
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
	// Held pose. Negative X = LEFT hand (the torch owns the right). Default
	// only: each pickup carries its own (Pickup::handOffset / handTiltDeg),
	// since these numbers describe the key mesh's axes.
	static constexpr glm::vec3 HAND_KEY_OFFSET = glm::vec3(-0.40f, -0.4f, -0.9f);
	// X = 90 swings the key's long axis (local Z) onto world Y -- upright, tip
	// up. Negate to -90 if a render shows it tip-down.
	static constexpr glm::vec3 HAND_KEY_TILT_DEG = glm::vec3(90.0f, -20.0f, 0.0f);
	// The held item's orientation: its own tilt, with the walk's roll on the
	// last axis. Shared by the rise and the sink so the two can't drift apart.
	static glm::mat4 handGrip(const glm::vec3 &tiltDeg, float bobRollDeg) {
		return glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.x), glm::vec3(1.0f, 0.0f, 0.0f))
			 * glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.y), glm::vec3(0.0f, 1.0f, 0.0f))
			 * glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.z + bobRollDeg), glm::vec3(0.0f, 0.0f, 1.0f));
	}

	// ---- Held-item wall tuck -------------------------------------------------
	//
	// Both hands are welded to the camera and hold their item ~1 unit out, past
	// the 0.3 PLAYER_RADIUS the body is held off walls by. Walk up to a wall
	// and the far end of what you carry is inside it. For the key that's ugly;
	// for the torch it's a lighting bug -- its point light and cube shadow ride
	// the flame at the torch head, so a head through a wall darkens the wall in
	// front and lights the room beyond through solid geometry. A depth-cleared
	// second pass wouldn't help (the light isn't in the depth buffer); the item
	// has to move -- lowered and turned across the body, not yanked to the face.
	//
	// Driven GEOMETRICALLY, not by blending toward a fixed "tucked" pose: a
	// fixed pull is the same whether the wall is at arm's length or the wrist.
	// The probe measures how far the item may reach RIGHT NOW, and the pose is
	// built to reach exactly that far -- looking down at a table (half an arm
	// away) is the case that proves a wall-tuned constant wrong.
	//
	// Everything below is a REACH FRACTION: 1 extended, HAND_TUCK_MIN_REACH
	// against the chest.

	// How far the item's body reaches past the grip, camera-space, so the probe
	// stops at its leading edge. Rough -- HAND_TUCK_SKIN dwarfs the error.
	static constexpr float HAND_TUCK_TORCH_PAD = 0.30f;
	static constexpr float HAND_TUCK_KEY_PAD = 0.15f;

	// Colliders grown by this for the probe only (the body still hits the real
	// boxes). Two jobs: the collision shell is coarse (walls are thin plates),
	// so this stops the item a stand-off short and covers jambs/pilasters no
	// one authored a box for; and it's the safety margin for the drop and
	// rotation the pose adds off the probed line. Can't just be large: it's a
	// stand-off from every surface, so the item folds away in merely narrow
	// rooms. Raise if something clips, lower if it tucks in open space.
	static constexpr float HAND_TUCK_SKIN = 0.30f;

	// The reach the item is never retracted past. A LIGHTING limit: a light at
	// the eye flattens all shading and blows out what's closest -- better a few
	// cm of torch in a wall. Also the probe's start: everything nearer is taken
	// as clear (the body's own 0.3 clearance guarantees it, and the inflated
	// skin would otherwise read "blocked" whenever you brush a wall).
	static constexpr float HAND_TUCK_MIN_REACH = 0.35f;

	// Samples across the probed stretch. Ten over ~1 unit -> ~0.1 granularity,
	// fine enough that the smoothing hides the steps.
	static constexpr int HAND_TUCK_SAMPLES = 10;

	// Tuck in fast, out slow: one time constant can't be both slow enough not
	// to strobe the torchlight and fast enough not to let the item dip in.
	static constexpr float HAND_TUCK_TAU_IN = 0.05f;
	static constexpr float HAND_TUCK_TAU_OUT = 0.18f;

	// Radial retraction shortens Y too, so the item drifts UP as it comes in,
	// reading as raised-to-face rather than tucked. This cancels that lift at
	// full retraction; not a motion of its own.
	static constexpr float HAND_TUCK_DROP = 0.28f;
	// Extra grip rotation on top of the item's tilt: nose down (X), across the
	// body (Y). This sells the tuck as a gesture, and it's free -- rotating
	// about the grip barely moves the light. Negate a component if a render
	// shows an item rotating the wrong way.
	static constexpr glm::vec3 HAND_TUCK_TILT_DEG = glm::vec3(40.0f, 30.0f, 0.0f);

	// Live reach fraction per hand, advanced by advanceReach() each frame. Two
	// values: a wall to your right reaches the torch before the key.
	float handTorchReach = 1.0f;
	float handKeyReach = 1.0f;

	// ghostPointBlocked's test with a stand-off. Separate, not a defaulted
	// argument: that one answers a sightline question where inflating the world
	// would be wrong.
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
	// its authored offset. Marches outward from HAND_TUCK_MIN_REACH along the
	// eye->item line, stops at the first solid sample; the last clear one is
	// where the leading edge goes. A distance, not a severity, so the caller
	// can place the item AT it -- the difference from the fixed-pose version.
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

	// One asymmetric exponential smoothing step of a reach fraction. Can't
	// overshoot, needs no "still tucking" state. Tucking IN means reach going
	// DOWN, so the comparison is inverted against the tau names.
	static void advanceReach(float &state, float target, float deltaT) {
		float tau = (target < state) ? HAND_TUCK_TAU_IN : HAND_TUCK_TAU_OUT;
		state += (target - state) * (1.0f - std::exp(-deltaT / tau));
	}

	// Builds the tucked pose in place, so callers still compose the walk bob on
	// top. Radial retraction along handFreeReach's line, then the gesture: a
	// drop to cancel the lift, and rotation. "Inward" is read off the offset's
	// sign (torch +X, key -X), so neither hand carries a mirrored constant.
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
	// Extra tilt so it looks gripped, not dead level.
	static constexpr glm::vec3 HAND_TORCH_TILT_DEG = glm::vec3(-15.0f, 20.0f, 0.0f);
	// Uniform scale: the mesh is sized for a wall mount.
	static constexpr float HAND_TORCH_SCALE = 0.35f;
	// Pick-up animation, mirror of the key's: the torch rises into frame from
	// below. Its own constants so tweaking one item doesn't retune the other.
	static constexpr float TORCH_RAISE_DURATION = 0.35f;
	static constexpr float TORCH_RAISE_DROP = 0.8f;
	// Seconds since pickup, or >= TORCH_RAISE_DURATION once the rise is over.
	// Starts saturated so an already-collected torch doesn't play it.
	float torchRaiseElapsed = TORCH_RAISE_DURATION;

	// The flame at a torch's head (Flame.hpp). One Flame instance drives every
	// torch, held one included.
	Flame flame;

	// The daylight outside the exit door (ExitGlow.hpp): one overbright quad,
	// its glare all the work of the bloom chain. Driven by exitDoorIndex/EXIT_GLOW_*.
	ExitGlow exitGlow;

	// Line renderer for the cheat-gated debug overlays. What each draws is
	// decided in updateUniformBuffer(). See DebugLines.hpp for vertex pulling.
	DebugLines debugLines;

	// One entry per torch with a flame, filled in localInit() (addTorchFlame)
	// and walked every frame in updateUniformBuffer(). `anchor` is in the TORCH
	// MODEL's local space; `inst->Wm * vec4(anchor,1)` is the flame's world
	// position.
	//
	// The rest is this torch's live fire state, simulated on the CPU because
	// three consumers must agree on it: the billboards, the sparks, and the
	// cast point light. One signal, produced once.
	struct TorchFlame {
		Instance *inst;
		int flameId;
		glm::vec3 anchor;

		// True only for the held torch. A wall torch billboards the ordinary
		// CYLINDRICAL way (spin about world up, stay upright). The held torch
		// is anchored in CAMERA space and tilts with the view like a weapon
		// viewmodel, so its flame must ride the camera's actual up/right
		// (pitch included) or it swings out of alignment with the shaft.
		bool heldByCamera = false;

		// Phase offset into the noise field, so no two torches gutter together.
		float phase = 0.0f;

		// BRIGHTNESS envelope, ~0.30 (gutter) to ~1.40 (flare). Scales
		// brightness, spark output, and the point light's colour/reach -- not
		// height (heightScale). Chased by a critically-damped spring
		// (intensityVel), not assigned raw, so brightness glides rather than
		// jumping to a fresh noise sample each frame.
		float intensity = 1.0f;
		float intensityVel = 0.0f;

		// HEIGHT envelope, ~0.78..1.09. The same signal as intensity, but
		// compressed and low-passed much harder (FLAME_HEIGHT_TAU): the fuel
		// column follows the light output late. Its own value so height glides
		// instead of teleporting at flicker rate.
		float heightScale = 1.0f;

		// This torch's smoothed stare-at factor, 0..1 (glare block in
		// updateUniformBuffer): overdrives its HDR output.
		float glare = 0.0f;

		// Last frame's anchor and a low-passed velocity from it. A flame is
		// dragged by the air, so it leans AGAINST its own motion.
		glm::vec3 prevPos = glm::vec3(0.0f);
		glm::vec3 smoothedVel = glm::vec3(0.0f);
		bool velPrimed = false;

		// This torch's fire colour: drives both the flame gradient and the
		// cast light (one signal). Overwritten every frame by the hunt cycle
		// (baseColor dragged toward violet); everything downstream reads this
		// field, so the colour change needed no other plumbing.
		glm::vec3 color = TORCH_LIGHT_COLOR;

		// The colour with nothing hunting (flames.json, or default orange),
		// captured at spawn. `color` can't double as its own base -- mixing
		// toward violet and storing back converges on violet.
		glm::vec3 baseColor = TORCH_LIGHT_COLOR;

		// Multiplies FLAME_HEIGHT/HALF_WIDTH on top of the instance scale. 1.0
		// for torches; candles pass smaller so the flame reads as a candle's.
		float sizeScale = 1.0f;

		// Multiplies the point light's colour only -- independent of
		// sizeScale, since a small flame isn't automatically dim. Candles pass
		// small so they cast the faint local light a candle actually does.
		float lightScale = 1.0f;

		// True for a candle, false for a torch (held included). The whole
		// distinction the HUD's separate "Torch/Candle Shadows" toggles need;
		// everything else treats the two identically. From flames.json's
		// per-model "isCandle", default false.
		bool isCandle = false;

		// Is this flame burning? Read through flameBurning(), so switching it
		// off takes the light, billboard, sparks, shadow candidacy and glare
		// in one go -- nothing is spawned/destroyed at runtime. False for a
		// candle authored unlit (lit with [E] off the held torch) and for the
		// held torch (lit from a burning wall torch). Wall torches are
		// authored burning and never go out.
		bool burning = true;
		// What `burning` was authored as, so restartRun() can blow out the
		// candles the player lit -- a dark run must start dark again.
		bool spawnBurning = true;

		// Catching-fire envelope: 0 (unlit) .. ~1.05 flare .. 1 (lit). Scales
		// the billboard AND the light colour (one signal, two consumers), so a
		// catching flame doesn't pop to size while its light is still dark.
		// Seeded 1 for anything authored burning, 0 for unlit, so nothing
		// plays this on its spawn frame.
		float ignitionScale = 1.0f;
		float ignitionVel = 0.0f;

		// This flame's anchor in WORLD space, computed at spawn. Only
		// meaningful for a static flame (!heldByCamera) -- the held torch's is
		// recomputed each frame. Aim target for findGazedCandle().
		glm::vec3 anchorWorld = glm::vec3(0.0f);

		// True for every flame that competes for a DYNAMIC shadow-cube slot.
		// Set in addTorchFlame() as !heldByCamera: every flame is either the
		// held torch (its own fixed slot) or a static object that belongs in
		// the pool, so a future torch/candle is a candidate for free.
		bool shadowCandidate = false;

		// This flame's absolute cube-shadow slot if it holds one, else -1.
		// Read by the light-append loop to fill LightData::shadowIndex.
		int shadowSlot = -1;

		// This candidate's 6 face view-projections, computed once at
		// registration (it's static, like the lights.json torches). Copied
		// into torchFaceMatrices[]/torchLightPos[] whenever it's given a slot.
		std::array<glm::mat4, 6> shadowFaceMatrices{};

		// The lean, resolved into the billboard's axes (x right, y forward),
		// in the flame's local units. Uploaded straight to the shader.
		glm::vec2 lean = glm::vec2(0.0f);
	};
	std::vector<TorchFlame> torchFlames;

	// Candle lighting: the third interactable, needing no list of its own -- a
	// candle IS a TorchFlame authored unlit, so `nearbyCandle` indexes
	// torchFlames. Same gaze-then-proximity shape as doors and keys. Radii in
	// 3D like a pickup's (a candle is on a table), tighter because lighting one
	// means holding a torch up to it -- close range.
	static constexpr float CANDLE_INTERACT_RADIUS = 2.5f;
	static constexpr float CANDLE_LOOK_DISTANCE = 5.0f;
	static constexpr float CANDLE_AIM_RADIUS = 0.35f;	// tight, the wick is small
	// Index into `torchFlames` of the unlit candle in range, or -1. Set in GameLogic().
	int nearbyCandle = -1;

	// Index into `torchFlames` of the unlit candle aimed at within look range,
	// or -1. Two-gate split like findGazedDoor. Lit candles are skipped, so
	// they neither glow nor steal aim.
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

	// Wall-torch lighting: the mirror of the candle interaction. The held torch
	// starts unlit; aiming at a burning wall torch and pressing [E] lights it,
	// for the run (restartRun blows it out). A wall torch is a bigger target
	// at head height, so the aim cone is more forgiving.
	static constexpr float WALL_TORCH_INTERACT_RADIUS = 3.0f;
	static constexpr float WALL_TORCH_LOOK_DISTANCE = 6.0f;
	static constexpr float WALL_TORCH_AIM_RADIUS = 0.6f;
	// Index into `torchFlames` of the burning wall torch in reach and aimed
	// at, or -1. Set in GameLogic().
	int nearbyWallTorch = -1;

	// Index into `torchFlames` of the burning wall torch aimed at within look
	// range, or -1. Non-negative only while the held torch is unlit AND in
	// hand. Two-gate split like findGazedCandle.
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

	// The floor torch, before it's picked up: a single gaze target at its
	// authored resting place. True when aimed at within look range and
	// still on the ground. The mirror of findGazedPickup for the one prop
	// that doesn't ride the Pickup/keyRing machinery -- it reuses the
	// pickup look/aim tolerances since it's the same kind of small floor
	// object.
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

	// One anchor for every torch (the wall and held meshes are identical,
	// confirmed by walking the POSITION accessors). X/Z at the centroid of the
	// mesh's top (a wide flat cup). Y 0.30, below the rim (0.413): the cup has
	// depth, so sitting the flame at the rim left it floating above the torch.
	// The flame's field fades in a few percent up the card, so the anchor
	// sinks that much further and the fade reads as the flame emerging.
	// Also flames.json's fallback default.
	static constexpr glm::vec3 TORCH_FLAME_ANCHOR = glm::vec3(-0.384f, 0.30f, 0.0f);

	// The flame's size in the torch model's local units, so it rides each
	// instance's uniform scale -- full size on the wall torches, shrunk on the
	// held one. A fixed world size looked right on one and wrong on the other.
	// HALF_WIDTH is wider than the visible flame: the noise field eats into
	// its own silhouette, so the quad must be bigger than the fire inside it.
	static constexpr float FLAME_HEIGHT = 0.95f;
	static constexpr float FLAME_HALF_WIDTH = 0.20f;

	// The torch flame's point light. One colour/falloff for every torch --
	// all the same kind of fire. Tighter than the gate lanterns: a flame is a
	// smaller, closer source than a lamp head.
	static constexpr glm::vec3 TORCH_LIGHT_COLOR = glm::vec3(1.0f, 0.5f, 0.16f);
	static constexpr float TORCH_LIGHT_G = 2.1f;
	static constexpr float TORCH_LIGHT_BETA = 1.4f;
	// Candles also shrink g (falloff reach) on top of flames.json's lightScale
	// (peak brightness) -- two independent knobs. Without it a dim candle would
	// still carry into the next room at torch range instead of a local pool.
	static constexpr float CANDLE_LIGHT_G_SCALE = 0.35f;
	// Fraction of peak below which a light's shadow isn't worth re-capturing
	// for a moving occluder (shadowRelevantReach() turns it into a distance).
	// 2%: invisible even in this dark, but a ghost at the far end doesn't keep
	// every torch re-rendering. Lower it if a shadow stops following its ghost.
	static constexpr float SHADOW_REACH_CUTOFF = 0.02f;
	// Fraction of a torch light's peak below which lightReachesViewCone() treats
	// it as lighting nothing: once that contour clears the drawn region (the
	// GEOM_CULL_RADIUS ball or the view cone) the light is dropped from gubo and
	// its shadow cube slot is freed. Higher than SHADOW_REACH_CUTOFF -- a shadow
	// that lags its caster reads worse than a faint light winking out -- but low
	// enough to stay invisible in this dark. Raise it to cull harder; watch the
	// boundary from cheats.debugCam (the per-flame live/dead crosses).
	static constexpr float LIGHT_CULL_REACH_CUTOFF = 0.04f;

	// How far a torch light is still uploaded, and how many may be live. Each
	// costs a full GGX eval per SAMPLE -- the dominant GPU cost. The
	// static_assert below ties it to the geometry cone.
	static constexpr float TORCH_LIGHT_CULL_DIST = 55.0f;
	static constexpr int TORCH_LIGHT_MAX_LIVE = 32;

	// Geometry visibility: a radius around the player plus a longer view cone.
	// GEOMETRY only, never the light list. Carries headroom past the real
	// frustum rather than tracking it.
	static constexpr float GEOM_CULL_RADIUS = 12.0f;  // always drawn this close, any facing
	static constexpr float GEOM_CULL_CONE_DIST = 50.0f;  // dungeon's longest sightline is ~60
	// Keeps a drawn torch bracket always a lit one.
	static_assert(TORCH_LIGHT_CULL_DIST >= GEOM_CULL_CONE_DIST,
				  "a torch light cull shorter than the geometry cone would "
				  "leave a visible torch model unlit");
	// Cone half-angle as a cosine (a plain dot product, no per-instance acos).
	// ~70 degrees, well past the widest FOV, plus the per-instance size allowance.
	static constexpr float GEOM_CULL_CONE_COS = 0.34f;
	// Instances are tested by CENTRE, and this pack's meshes aren't
	// centre-pivoted, so a long wall could measure outside the cone with half
	// of it still on screen -- the real cause of edge-of-screen popping. Fixed
	// by treating every instance as a bounding SPHERE: its collider's extents
	// where it has one, else this flat allowance (sized to a floor tile, the
	// largest un-collided thing).
	static constexpr float GEOM_CULL_FALLBACK_RADIUS = 4.0f;

	// Third-person debug camera (cheats.debugCam): the render view orbits the
	// player at these starting spherical coords (I/K pitch, J/; yaw, U/O
	// dolly at runtime). Only ViewPrj changes -- every cull runs from the real
	// first-person eye, so culled instances visibly wink out when watched
	// from here.
	static constexpr float DEBUG_CAM_DIST0 = 17.0f;   // metres from the player
	static constexpr float DEBUG_CAM_PITCH0 = 20.0f;  // degrees above the player
	static constexpr float DEBUG_CAM_ORBIT_SPEED = 90.0f;  // deg/s, IJKL
	static constexpr float DEBUG_CAM_DOLLY_SPEED = 14.0f;  // m/s, U/O
	// How much clear space the debug camera's near plane leaves in front of
	// the player after clipping away the room shell between them.
	static constexpr float DEBUG_CAM_CLIP_MARGIN = 4.0f;
	// With the debug camera on, instances whose origin sits above this height
	// (the ceilings at y=6.2, nothing else) are skipped, so the spectator
	// looks into an open-top dollhouse instead of at a roof. The near-plane
	// cutaway only clears the roof right over the player; this clears it over
	// every room the cone reaches.
	static constexpr float DEBUG_CAM_ROOF_CUT = 5.0f;

	// True if the instance at worldPos (bounding radius objRadius) is close or
	// aimed-at enough to draw. eyePos/forward are computed once per frame in
	// updateUniformBuffer() and passed through.
	static bool geometryVisible(const glm::vec3 &worldPos, float objRadius,
								const glm::vec3 &eyePos, const glm::vec3 &forward) {
		glm::vec3 d = worldPos - eyePos;
		float dist = glm::length(d);
		// The object's own half-size eaten out of the distance it's judged
		// by, so a big/near object (large objRadius relative to dist) is
		// effectively already "at" the camera well before its centre is.
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
		// The object's angular size is ~objRadius/dist, so a near or large one
		// needs less head-on facing to count as in the cone. Clamped so a
		// tiny/far one still needs the full cosine.
		float angularSlack = glm::clamp(objRadius / dist, 0.0f, 1.0f) * 0.5f;
		return facing >= (GEOM_CULL_CONE_COS - angularSlack);
	}

	// True if a point light at `lightPos` with effective reach `reach` could
	// light ANY geometry the frame draws -- i.e. its illumination sphere
	// overlaps the geometry-visible region: the always-drawn GEOM_CULL_RADIUS
	// ball around the eye, or the view cone (apex at the eye, axis `forward`,
	// half-angle from GEOM_CULL_CONE_COS, length GEOM_CULL_CONE_DIST). Torch
	// lights and their shadow cubes are dropped when this is false: a torch
	// whose whole lit sphere sits behind the player or down an off-cone
	// corridor contributes nothing on screen.
	//
	// Deliberately conservative -- it inflates the cone by the full `reach` and
	// skips geometryVisible()'s per-instance angular slack, so it only ever
	// culls a light that truly reaches nothing drawn. The perpendicular and
	// axial distances are both 1-Lipschitz in world position, so evaluating the
	// cone's flare at the farthest axial point the sphere can touch
	// (axial + reach) bounds every lit point inside it. One sqrt, no acos/tan.
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

	// How much a candidate's priority (live-light cut and shadow pool) is
	// skewed by being ahead of vs behind the player. 0 = pure nearest-first.
	// The (1 + W*(1 - alignment)) term is 1 dead ahead, 1+W to the side, 1+2W
	// dead behind. At 0.6 a torch 8 units behind loses to one 10 units in view.
	static constexpr float SHADOW_FACING_BIAS_WEIGHT = 0.6f;

	// Squared eye->pos distance, inflated for anything behind `forward`. Still
	// monotonic in real distance for a fixed alignment, so only the
	// front/behind axis gets a thumb on the scale. One sqrt (distSq is already
	// at hand at every call site).
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

	// Fire envelope. The fast flicker band lives in Flame.frag as a per-pixel
	// shimmer varying along the flame -- one twitch on every pixel at once
	// reads as a brightness dial. The CPU keeps a slower 7 Hz term at reduced
	// weight so the cast light still dances, layered under slower terms so it
	// doesn't read as a metronome.
	static constexpr float FLAME_FLICKER_HZ = 7.0f;
	// Spring rate for the brightness envelope (critically damped,
	// TorchFlame::intensityVel). 14 rad/s settles in ~0.15 s: a gutter still
	// ducks, no noise step survives as a jump.
	static constexpr float FLAME_BRIGHT_OMEGA = 14.0f;
	// One-pole rate for the HEIGHT envelope, seconds. A flame shortens over a
	// third of a second; it doesn't teleport between heights.
	static constexpr float FLAME_HEIGHT_TAU = 0.35f;
	// Guttering: brief irregular collapses. Driven by its own slow noise
	// crossing a high threshold, so it happens rarely and off-schedule.
	static constexpr float FLAME_GUTTER_SPEED = 0.85f;
	static constexpr float FLAME_GUTTER_LO = 0.66f;	// noise below this: no gutter
	static constexpr float FLAME_GUTTER_HI = 0.82f;	// above this: full gutter
	static constexpr float FLAME_GUTTER_DEPTH = 0.48f;	// how far it ducks

	// Catching fire: a flame grows in from nothing when lit, rather than
	// snapping to full size. An UNDERDAMPED spring
	// (TorchFlame::ignitionScale/ignitionVel) chasing 1/0 -- ZETA < 1 lets it
	// flare a little past resting size on the way up, which reads as
	// "catching" rather than "fading in". OMEGA sets the pace (~0.3 s).
	static constexpr float FLAME_IGNITION_OMEGA = 9.0f;
	static constexpr float FLAME_IGNITION_ZETA = 0.55f;

	// Stare-at glare: centring a wall torch in view swells the post chain's
	// exposure/bloom and that torch's HDR output. View-dependent, so it masks
	// the billboard's flat-card nature exactly when it shows -- face-on, close.
	// Purely geometric, no feedback through the renderer.
	static constexpr float GLARE_COS_MIN = 0.90f;	// facing cosine: below, no glare
	static constexpr float GLARE_COS_MAX = 0.985f;	// above, full glare
	static constexpr float GLARE_DIST_NEAR = 1.0f;	// fades in past this...
	static constexpr float GLARE_DIST_FAR = 9.0f;	// ...and is gone by here
	// Asymmetric smoothing: dazzle arrives fast, the eye recovers slower.
	// The asymmetry is also what prevents pumping when the view strafes back
	// and forth across the facing threshold.
	static constexpr float GLARE_TAU_RISE = 0.15f;
	static constexpr float GLARE_TAU_FALL = 0.40f;
	static constexpr float GLARE_EXPOSURE_GAIN = 0.30f;	// exposure *= 1 + gain*glare
	static constexpr float GLARE_BLOOM_GAIN = 0.90f;	// bloomIntensity likewise
	static constexpr float GLARE_FLAME_GAIN = 0.50f;	// that flame's own HDR boost

	// Lean. TAU: how fast the smoothed velocity chases the real one. PER_SPEED:
	// flame half-widths of tip offset per world-unit/s of hand speed (a brisk
	// walk ~3 u/s puts the tip over by under a fifth of a half-width). MAX
	// caps it so sprinting can't fold the flame onto its side.
	static constexpr float TORCH_LEAN_TAU = 0.14f;
	static constexpr float TORCH_LEAN_PER_SPEED = 0.055f;
	static constexpr float TORCH_LEAN_MAX = 0.30f;

	// Seconds since startup, uploaded as gubo.time. A free-running
	// accumulator, so the sway/flicker never repeats on a noticeable cycle.
	float animTime = 0.0f;

	// Walking sway: a lateral swing once per stride plus a vertical bounce at
	// twice that, from one accumulating phase. walkBobBlend is the 0..1
	// amplitude, eased toward 1 while walking, 0 at rest.
	float walkBobPhase = 0.0f;
	float walkBobBlend = 0.0f;
	// Radians/second the phase advances at normal walking speed, faster
	// while sprinting.
	static constexpr float WALK_BOB_SPEED = 7.0f;
	// Sway amplitude in world units, before walkBobBlend scales it down.
	static constexpr float WALK_BOB_VERTICAL = 0.035f;
	static constexpr float WALK_BOB_LATERAL = 0.02f;
	// How fast walkBobBlend eases toward its target.
	static constexpr float WALK_BOB_BLEND_TAU = 0.15f;
	// Vertical head-bob applied to the view itself, kept well under the hand's
	// bob so the world only just nods.
	static constexpr float CAM_BOB_VERTICAL = 0.012f;

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
		// Live position WITHOUT the bob (presentation only -- folding it in
		// would pump the collision slab). Maintained in every mode.
		glm::vec3 pos{0.0f};
		// Facing, eased not snapped: a ghost spinning on the spot reads as a bug.
		float yaw = 0.0f;
		// Which way it went round an obstacle last frame, +/-1. Kept until it
		// has a clear run again, so it commits to one side of a pillar.
		float turnBias = 1.0f;

		// Breadcrumbs, oldest first. trail[0] is where the chase began.
		std::vector<glm::vec3> trail;
		// Patrol progress saved when the chase started, restored on getting home.
		int resumeIdx = 1;
		float resumeDist = 0.0f;

		// Where the ghost last SAW the player, and whether it has this hunt.
		// Chase steers toward this, not the live position. Never seen = stays
		// on patrol.
		glm::vec3 lastKnownPlayerPos{0.0f};
		bool hasLastKnown = false;

		// Stuck detection for Chase: `pos` at the last check, and time since it
		// covered less than GHOST_STUCK_EPS. Pinned against a door and idling
		// at a stale target look the same; GHOST_GIVEUP_TIME turns either into
		// giving up.
		glm::vec3 stuckCheckPos{0.0f};
		float stuckTimer = 0.0f;

		// 0..1, how far into a chase, eased not switched. Presentational:
		// Spectral.frag reads it (as ubo.F0) to shift the apparition toward
		// hunt red and brighter, and it's the only thing on that shader that
		// changes over time -- so this ease IS the animation. Eased because
		// `mode` flips in one frame, and Chase is entered/left repeatedly
		// within a hunt (the stuck/give-up path).
		float chaseBlend = 0.0f;
	};
	std::vector<Ghost> ghosts;

	// Time constant of chaseBlend's ease, one per direction. Lighting up
	// faster than calming down: the telegraph has to arrive while it still
	// buys you something, the fade out sells the chase being let go of.
	static constexpr float GHOST_CHASE_FADE_IN_TAU = 0.25f;
	static constexpr float GHOST_CHASE_FADE_OUT_TAU = 1.1f;

	// Bob envelope: how far above/below the resting hover height (radians/sec, world units).
	static constexpr float GHOST_BOB_SPEED = 1.6f;
	static constexpr float GHOST_BOB_AMPLITUDE = 0.3f;
	// How fast a ghost walks its breadcrumbs home. Faster than patrol or chase:
	// the return is dead time, and a ghost drifting back at patrol speed would
	// be out of position for the next hunt.
	static constexpr float GHOST_RETURN_SPEED = 4.5f;
	static constexpr float GHOST_TURN_SPEED = 9.0f;	// yaw easing, rad/s (~half-turn in 0.3 s)

	// A vertical cylinder, not a box: a box's corners project further on a
	// diagonal and snag on jambs. Fitted from Ghost.gltf at load. The radius
	// is shrunk below the real fit -- 0.45 not 0.65, where the clear/blocked
	// test flipped every frame and the ghost vibrated.
	static constexpr float ghostXZFitShrink = 0.45f;
	// No extra steering padding. Tried with the 0.65 shrink and it made
	// "blocked" win the flip-flop more often. Kept in case it's worth revisiting.
	static constexpr float ghostSteerMargin = 1.0f;
	float ghostRadius = 0.5f;
	float ghostBodyBottom = -1.80f;
	float ghostBodyTop = 0.83f;
	// How far ahead a candidate heading is probed for walls. Long enough to
	// turn along a wall in time, short enough not to refuse a doorway.
	static constexpr float GHOST_PROBE_DIST = 1.2f;
	// Horizontal catch distance plus vertical slack. Split because the ghosts
	// hover at 2.2 and the player's eyes are at 1.8, so a 3D test would need an
	// unfair horizontal radius.
	static constexpr float GHOST_CATCH_RADIUS = 0.85f;
	static constexpr float GHOST_CATCH_VERTICAL = 2.5f;

	// THE SPECTRAL VEIL, the screen half of the ghost fade
	// (SpectralFade.glsl). A ghost that disappears when you reach it was never
	// a body, so Composite.frag washes the frame cold. These MUST match
	// SPECTRAL_INSIDE_* in that file -- a gap between the ramps is a moment
	// with no ghost and no wash.
	static constexpr float SPECTRAL_VEIL_OUTER = 2.00f;
	static constexpr float SPECTRAL_VEIL_INNER = 1.05f;
	// Margin around the model's Y bounds, so the bob doesn't blink the wash on
	// and off.
	static constexpr float SPECTRAL_VEIL_FADE_Y = 0.60f;
	// How far up the veil's ramp before a non-hunting ghost snuffs the held
	// torch. Sharing the ramp ties the flame going out to when "inside the
	// ghost" first reads, rather than only at dead centre.
	static constexpr float SPECTRAL_VEIL_SNUFF_AT = 0.5f;
	// Breadcrumb spacing, and the radius within which a new one counts as
	// revisiting an old one (and prunes the loop between). The prune radius
	// must be larger than the spacing, or consecutive crumbs prune each other.
	static constexpr float GHOST_TRAIL_SPACING = 0.75f;
	static constexpr float GHOST_TRAIL_PRUNE_RADIUS = 1.4f;

	// How little ground counts as "not moving" for the stuck check, and how
	// long a chasing ghost tolerates it. Padded above idle drift, so easing a
	// yaw or a ghostResolveWalls nudge doesn't reset the clock.
	static constexpr float GHOST_STUCK_EPS = 0.08f;
	static constexpr float GHOST_GIVEUP_TIME = 3.0f;

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

	// Where the exit is and whether it's locked, both from gameplay.json's
	// "exit" block. The box is world-space and axis-aligned: the player wins by
	// standing inside it.
	glm::vec3 exitBoxMin{0.0f};
	glm::vec3 exitBoxMax{0.0f};
	bool exitHasBox = false;
	bool exitRequiresKey = true;
	// Which key opens the way out, from gameplay.json's "exit"."keyId". Empty
	// (the default) means ANY key on the ring does, which is exactly what
	// requiresKey meant before doors had locks -- so a level that names no id
	// behaves as it always did. Name one here as soon as a door lock competes
	// for the same key: keys are one-shot, and a player who spends the only
	// key on a side door would otherwise reach an exit they can never open.
	// The exit does NOT consume the key it checks: the run is over the moment
	// it passes, so there's nothing left to spend it on.
	std::string exitKeyId;
	// Set every frame in GameLogic() when the player is standing in a locked
	// exit without the key, read by updateUniformBuffer() to explain why
	// nothing happened. Same pattern as nearbyDoor/nearbyPickup.
	bool atLockedExit = false;

	// --- The way out, as a door ------------------------------------------
	//
	// Index into `doors` of the exit leaf (hbDoorE), or -1 if the scene
	// didn't have it. Everything below rides on how far THAT door has swung,
	// rather than on the exit box or on the run state: the light outside is a
	// fact about the door being open, so it has to arrive while the leaf is
	// still moving and the player is still a few metres short of winning. It
	// is the swing that is the payoff -- by the time the box triggers, the run
	// is already over.
	int exitDoorIndex = -1;

	// Where the daylight stands. Two quads, and the reason there are two is
	// the door: it swings OUTWARD, so nothing can be parked right behind the
	// opening without the leaf sweeping through it.
	//
	// The wall slab is x 18.758..20 and the arch spans z 28.87..31.11. The
	// open leaf reaches x 21.78 at the widest point of its swing (its hinge is
	// at x 19.283 and its diagonal is 2.50 long), so the upright quad stands
	// at 22.0 -- past the leaf by 22cm, which is what lets the door open into
	// the light and be seen as a silhouette against it instead of being cut in
	// half by it.
	//
	// That distance is also what forces the second quad. Two units of open
	// ground between the threshold and the light are visible through the
	// bottom of the arch (the sill hides only what is within about 35cm of the
	// wall), and a strip of lit ground is exactly the "something out there"
	// this effect exists to deny. So the second quad lies FLAT, 6cm above the
	// ground plane, bridging from under the wall out to the upright one. The
	// arch then frames white above and white below, with the door swinging
	// between the two.
	//
	// Both overhang what they have to cover, generously, and the margins are
	// worked from the worst viewing angle rather than guessed. A player can
	// stand anywhere in the dv room, which reaches back to x 12.8, and the
	// extreme sightlines project the arch onto the upright quad's plane over
	// roughly z 27.0..33.0 and y -0.5..6.0 -- all of which has to fall inside
	// the quad's flat middle, not its border fade, which ExitGlow.frag starts
	// at 78% of the half-extent. Hence half-extents of 4.4 and 4.6 rather than
	// something that merely covers the opening head-on. Everything past the
	// arch is masked by the wall's own depth, and the part below y 0 is buried
	// under the ground plane outside.
	// Hub shrunk from 4x4 to 3x3 (see tools/build_scene.py): hbDoorE moved one
	// tile west, so every X here is shifted by -7.2 from what the comments
	// above still describe.
	static constexpr glm::vec3 EXIT_GLOW_CENTER = glm::vec3(23.6f, 2.8f, 10.79f);
	static constexpr float EXIT_GLOW_HALF_WIDTH = 4.4f;		// along world Z
	static constexpr float EXIT_GLOW_HALF_HEIGHT = 4.6f;	// along world Y
	// Faces back into the castle, i.e. west, so the player looking out through
	// the doorway sees it square on.
	static constexpr glm::vec3 EXIT_GLOW_NORMAL = glm::vec3(-1.0f, 0.0f, 0.0f);
	// The ground quad, x 19.5..22.7. Both ends are deliberately buried: the
	// near one runs back UNDER the wall slab, so its border fade (starting at
	// x 19.85) is hidden by stone and the light is already at full strength by
	// the time the threshold lets you see any of it; the far one passes behind
	// the upright quad, so the two overlap instead of meeting at a seam.
	//
	// 6cm above the ground plane: far enough not to z-fight it, low enough
	// that the door -- whose own bottom edge is at y 0.2 -- always sweeps
	// above it rather than through it.
	static constexpr glm::vec3 EXIT_GLOW_FLOOR_CENTER = glm::vec3(22.7f, 0.06f, 10.79f);
	static constexpr float EXIT_GLOW_FLOOR_HALF_X = 1.6f;
	static constexpr float EXIT_GLOW_FLOOR_HALF_Z = 3.6f;
	static constexpr glm::vec3 EXIT_GLOW_FLOOR_NORMAL = glm::vec3(0.0f, 1.0f, 0.0f);
	// The third quad: the same trick as the ground one, upside down, and it
	// exists for the same reason the ground one does -- the upright wall of
	// light is FINITE, and a player who walks up to the threshold and looks UP
	// sees over the top of it, straight into the skybox.
	//
	// Raising the upright quad cannot fix that, and no value of
	// EXIT_GLOW_HALF_HEIGHT can: the sightline through the top of the arch
	// hits the plane x = 22 at
	//     y = y_eye + (4.85 - y_eye) * (22 - x_eye) / (20 - x_eye)
	// which diverges as the player approaches the wall's outer face at x = 20.
	// The margins in EXIT_GLOW_CENTER's comment were worked from a player
	// standing back in the room, where the ratio is small; pressed against the
	// doorway it is unbounded. A ceiling closes the geometry instead of
	// chasing it -- every upward ray through the arch crosses y = 4.9 sooner
	// or later, and whichever of the two quads it reaches first is white.
	//
	// y = 4.9 is 5cm above the top of the doorway (the hole runs y 0..4.85,
	// see the collider boxes for dvDoor), so it is above the leaf's sweep and
	// buried in the stone over the arch for the whole stretch that lies inside
	// the wall. Same x extent as the floor quad, for the same two reasons: the
	// near end runs back under the wall so its border fade never shows, and
	// the far end passes behind the upright quad rather than meeting it at a
	// seam.
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
	// Decay time constant, and the largest snap it will absorb (past this the
	// view trails so far it reads as sinking).
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
		// Update render passes. Composite follows the window; the scene follows
		// it scaled by renderScale; the bloom targets follow that, divided by
		// BLOOM_DIV. Starter.hpp tears down and rebuilds the images around this.
		RP.width = renderWidth(w);
		RP.height = renderHeight(h);
		RPcomposite.width = w;
		RPcomposite.height = h;
		// After RP.width/height: bloomWidth()/Height() read those.
		RPbright.width = RPblurH.width = RPblurV.width = bloomWidth();
		RPbright.height = RPblurH.height = RPblurV.height = bloomHeight();

		// Refresh windowWidth/Height (only set in setWindowParameters()
		// otherwise); the cheat HUD needs the current size.
		windowWidth = (uint32_t)w;
		windowHeight = (uint32_t)h;

		// updates the textual output
		txt.resizeScreen(w, h);
		uiQuad.resizeScreen(w, h);
		crosshair.resizeScreen(w, h);
		pauseQuad.resizeScreen(w, h);
		startScreenQuad.resizeScreen(w, h);
		settingsQuad.resizeScreen(w, h);
		setCrosshairQuad();
		// The collider visualizer owns a swapchain-attached render pass too and
		// was never told about resizes, so shrinking the window rebuilt its
		// framebuffers at the old size (VUID-...-04533).
		SC.ColShow.resizeScreen(w, h);
	}
	
	// Fills the attachment descriptions for the HDR chain, at the current
	// swapchain size. Called from localInit() and onWindowResize().
	//
	// Spelled out rather than from getStandardAttchmentsProperties() because
	// no stock config is floating-point: AT_SURFACE_AA_DEPTH renders into the
	// 8-bit sRGB swapchain, capping every pixel at 1.0 -- which is what makes
	// bloom impossible. Everything here follows from wanting a 16-bit float
	// target.
	void buildPostAttachments() {
		const VkFormat HDR = VK_FORMAT_R16G16B16A16_SFLOAT;
		// Black, not the old cyan: with the geometry cull stopping walls past
		// its cone, THIS is what shows through instead of them, and black keeps
		// that as darkness/distance rather than a bright wall in the distance.
		const VkClearValue SKY = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
		const VkClearValue BLACK = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

		// --- the scene pass: multisampled HDR colour, depth, and a resolve
		// target the bloom chain and the composite can both sample.
		hdrAtt = {
			// Multisampled colour. storeOp DONT_CARE: nothing reads the
			// multisampled image, only the resolve, so writing it out would be
			// pure bandwidth.
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
			// The resolve. Unlike the stock AA config's, this one is NOT the
			// swapchain image (swapChain = false): it is an ordinary sampled
			// texture, which is what lets the passes below read the scene back.
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

		// --- the three bloom targets. Identical: one single-sampled HDR colour
		// attachment, no depth (a full-screen quad has nothing to test). loadOp
		// DONT_CARE: the quad covers every pixel, so a clear would write twice.
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

		// --- the composite, straight onto the swapchain image. finalLayout
		// PRESENT_SRC_KHR because the text and HUD passes that run after this
		// one both declare PRESENT_SRC_KHR as their initialLayout and load what
		// is already there.
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

	// Rebuilds the attachment lists via buildPostAttachments(), then re-inits
	// the five render passes -- scene at renderWidth()/Height(), bloom chain at
	// bloomWidth()/Height(), composite at the window size.
	//
	// Called from localInit(), and at runtime when msaaSamples changes: unlike
	// renderScale (which only pokes RP.width/height like a resize), a new
	// sample count changes the hdrAtt properties themselves, which only .init()
	// re-copies. The caller must follow this with RebuildPipeline().
	void initRenderPasses() {
		buildPostAttachments();

		// ATDEP_SIMPLE, not the default ATDEP_SURFACE_ONLY: the scene's output
		// is sampled by a later pass, so it needs the dependency ordering a
		// colour write against a later shader read (and the next frame's overwrite).
		// renderWidth()/Height() rather than -1,-1 ("match the swapchain"):
		// this is what lets renderScale differ from the window resolution.
		RP.init(this, renderWidth(swapChainExtent.width), renderHeight(swapChainExtent.height), -1, &hdrAtt,
				RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);

		RPbright.init(this, bloomWidth(), bloomHeight(), -1, &brightAtt,
					  RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		RPblurH.init(this, bloomWidth(), bloomHeight(), -1, &blurHAtt,
					 RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		RPblurV.init(this, bloomWidth(), bloomHeight(), -1, &blurVAtt,
					 RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		// The composite writes the swapchain and is read by nobody, so the
		// plain surface dependency is right.
		RPcomposite.init(this, -1, -1, -1, &compositeAtt,
						 RenderPass::getStandardDependencies(ATDEP_SURFACE_ONLY), false);
	}

	// Width/height of the bloom chain's targets, derived from the SCENE pass's
	// own current resolution (RP.width/height, already scaled by
	// renderScale) rather than the swapchain's -- bloom reads the scene's
	// resolve target, so its size should track what that target actually is,
	// not the window's. Clamped at 1 so a minimised or absurdly narrow window
	// can't ask for a zero-sized image.
	int bloomWidth() const {
		return std::max(1, RP.width / BLOOM_DIV);
	}
	int bloomHeight() const {
		return std::max(1, RP.height / BLOOM_DIV);
	}

	// Here you load and setup all your Vulkan Models and Textures.
	// Here you also create your Descriptor set layouts and load the shaders for the pipelines
	void localInit() {
		// windowWidth/windowHeight are still whatever setWindowParameters()
		// requested (800x600) at this point -- that's a size in WINDOW
		// POINTS, handed to glfwCreateWindow(), not necessarily the real
		// FRAMEBUFFER size in pixels. On an integer-scale display the two
		// are the same number; on a HiDPI/Retina one (2x, 3x, ...) the
		// framebuffer is that many times larger, and nothing has corrected
		// windowWidth/windowHeight to match it yet -- the actual swapchain
		// is sized correctly regardless (chooseSwapExtent() in Starter.hpp
		// queries glfwGetFramebufferSize() itself, independently), which is
		// why the 3D scene always renders at the right resolution, but every
		// UI widget below (txt/uiQuad/crosshair/pauseQuad/startScreenQuad/
		// settingsQuad, and StartScreen's own first setOpen()) is about to
		// be initialized against whatever windowWidth/windowHeight says NOW
		// -- the stale, too-small request -- and stays that way until an
		// actual window resize corrects it (onWindowResize() gets the real
		// framebuffer size from GLFW's own callback and forces everything to
		// rebuild against it). That mismatch is what read as "the launch
		// screen isn't really fullscreen until I resize the window": its own
		// render pass was built to only cover the smaller, wrong area.
		//
		// Corrected here, once, before anything below reads these two
		// fields, rather than chasing the same fix per-widget.
		{
			int realW = 0, realH = 0;
			glfwGetFramebufferSize(window, &realW, &realH);
			if(realW > 0 && realH > 0) {
				windowWidth = (uint32_t)realW;
				windowHeight = (uint32_t)realH;
				// Same story as windowWidth/windowHeight above: Ar defaults
				// to a hardcoded 4/3 guess in setWindowParameters() (matching
				// the REQUESTED 800x600, not necessarily the real
				// framebuffer), and is otherwise only ever recomputed by
				// onWindowResize(). A uniform HiDPI scale factor alone
				// wouldn't change the ratio, but there's no guarantee the
				// real framebuffer is 4:3 shaped at all -- correcting it here
				// on the same real dimensions keeps the very first frame's
				// 3D projection matrix right regardless.
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
		// Shadow sampling (set 2 of P, see CookTorrance.frag). One separate
		// sampler binding per cube map -- see the member declaration for why
		// not one array binding. linkSize on the samplers is their own index
		// into the flat VkDescriptorImageInfo list Scene builds per instance
		// (see the texDefs passed to PRs[0].init below), the same role it plays
		// for DSLlocal's single texture.
		//
		// Built in a loop rather than written out, so the count lives in
		// exactly one place: NUM_SHADOW_CUBES samplerCube bindings, the same
		// order and numbering CookTorrance.frag declares its shadowCube*
		// bindings with, which nothing but agreement here keeps true.
		// linkSize follows the same order (see shadowMapDefs below).
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

		// Anti-aliasing level, set before initRenderPasses() reads it. Starter's
		// default is the GPU's max (16 here), and Starter also forces
		// sampleShadingEnable, so MSAA becomes full supersampling: the fragment
		// shader runs per SAMPLE. At 16, CookTorrance.frag ran 16x per pixel
		// with the full light loop -- 25 FPS on this Iris Xe, vs 94 at 4x.
		// 4 is the compromise, live-adjustable via the "MSAA" slider.
		msaaSamples = VK_SAMPLE_COUNT_4_BIT;
		// The slider's upper bound: the device's real cap, not an assumed 16.
		maxMsaaLevel = std::log2((float)getMaxUsableSampleCount());

		initRenderPasses();

		// The cube shadow render pass (torches) -- see the RPShadowCubeCompat
		// member comment for why this is built once, shared, and only its
		// .renderPass field is used. createCubeShadowMaps() below builds the
		// 36 real per-face framebuffers against it.
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
		// ATDEP_DEPTH_TRANS, but for a COLOR attachment: external->0 puts the
		// image into COLOR_ATTACHMENT_OPTIMAL before the write, 0->external
		// makes the main pass's fragment read wait for the write.
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

		// Pipelines [Shader couples]
		// The last array, is a vector of pointer to the layouts of the sets that will
		// be used in this pipeline. The first element will be set 0, and so on..

		P.init(this, &VD, "shaders/scene/PosNormUV.vert.spv",
						  "shaders/scene/CookTorrance.frag.spv",
						  {&DSLglobal, &DSLlocal, &DSLshadowSample});

		// The ghosts. Two sets, not three: Spectral.frag is unlit and samples
		// no shadow map, so DSLshadowSample is left off entirely.
		Pspectral.init(this, &VD, "shaders/scene/PosNormUV.vert.spv",
							  "shaders/spectral/Spectral.frag.spv",
							  {&DSLglobal, &DSLlocal});
		// Alpha blending, the flag the whole effect hangs off and why the
		// ghosts need their own pipeline (blending is baked into a VkPipeline).
		Pspectral.setTransparency(true);
		// BACK-FACE CULLING KEPT ON, unlike Flame/ExitGlow (flat billboards):
		// the ghost is a closed mesh, and with depthWriteEnable forced on,
		// drawing both faces would blend two unsorted layers. One layer per
		// pixel has no order to get wrong; the Fresnel rim supplies the volume.
		// LESS_OR_EQUAL is load-bearing with the prepass in front: the prepass
		// leaves the nearest ghost depth, and EQUAL lets exactly those
		// fragments through.
		Pspectral.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

		// The prepass. Matches Pspectral but for the shader and compare op.
		// Transparency on: it emits alpha 0, so the blend returns dst and no
		// colour is written.
		PspectralDepth.init(this, &VD, "shaders/scene/PosNormUV.vert.spv",
								   "shaders/spectral/SpectralDepth.frag.spv",
								   {&DSLglobal, &DSLlocal});
		PspectralDepth.setTransparency(true);

		// The post passes. Two set layouts: one for the single-image passes,
		// one for the composite (which mixes two). Note: Starter reuses
		// `linkSize` as the INDEX into the image-info vector, not a byte size,
		// so binding 1 reads image 0 and binding 2 reads image 1.
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

		// The cube shadow pass's pipeline (torches). Same DSLlocal reuse
		// idea as the main pass: set 0 is the SAME per-instance buffer the
		// main pass's ubo.mMat comes from, reused here to read Wm again for a
		// different projection. Set 1 is DSLshadowCubeCapture,
		// one uniform buffer per torch cube slot (DSshadowCube[], mapped
		// fresh every frame in updateUniformBuffer()) carrying the light's
		// current view-projection matrices and world position -- NOT a push
		// constant, see ShadowCube.vert's header for why that would silently
		// freeze the held torch's shadow at whatever position it first
		// rendered from. The push constant that remains only ever carries
		// the face index, which genuinely is fixed at record time.
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
		// FRONT faces culled, so each occluder records the side turned AWAY
		// from the torch. Shadow acne is a surface comparing against its own
		// record; every defence against it is slack, and slack is what
		// detaches a shadow from its caster. Culling front faces removes the
		// premise -- a lit surface isn't in the map -- so the depth slack in
		// shadowFromCube() can be nothing but floating-point noise. The cost:
		// an occluder now leaks light by its own THICKNESS (its far side is
		// recorded), which every wall here dwarfs. Watch a single flat face,
		// which has no far side.
		PShadowCube.setCullMode(VK_CULL_MODE_FRONT_BIT);
		// Against RPShadowCubeCompat -- see its member comment.
		PShadowCube.create(&RPShadowCubeCompat);

		// Descriptor pool size (before loading the scene). The four post sets:
		// one block each, five textures between them. + NUM_SHADOW_CUBES for
		// DSshadowCube[], one block/set per cube slot.
		DPSZs.uniformBlocksInPool = 2 + 4 + NUM_SHADOW_CUBES;
		DPSZs.texturesInPool = 1 + 5;
		DPSZs.setsInPool = 2 + 4 + NUM_SHADOW_CUBES;

		// to support scene
		VDRs.resize(1);
		VDRs[0].init("VDposNormUV",  &VD);

		// DSLshadowSample: none of these are "fromInstance" -- the shadow maps
		// are the same fixed images for every instance, not per-instance
		// textures like DSLlocal's albedo map. pos is unused on a
		// non-fromInstance entry. In a loop for the same reason the layout
		// above is. Same order the binding list above and CookTorrance.frag's
		// declarations use.
		std::vector<TextureDefs> shadowMapDefs;
		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			shadowMapDefs.push_back({false, 0,
				{cubeShadowSampler.getSampler(), torchCube[i].cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
		}

		// ORDER MATTERS: Scene walks techniques in registration order, so
		// "Spectral" second puts every alpha-blended ghost after every opaque
		// wall it's seen through. Drawn first, a ghost would composite against
		// the clear colour and the dungeon would paint over it.
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

		// Same two entries minus the shadow maps, matching Pspectral's two-set
		// layout. DSLlocal takes the ghost's albedo at slot 0, which
		// Spectral.frag reads as a density mask.
		//
		// The pipeline named here is the DEPTH PREPASS, not Pspectral: naming
		// it (not nullptr) makes SC.init() build this technique's descriptor
		// sets, which then serve both passes (shared DSLs). Right after
		// SC.init() the pipeline is nulled so Scene skips it -- both prepass
		// and colour pass are issued by hand in populateCommandBuffer(), with
		// the flames between the dungeon and the prepass. Order: dungeon, then
		// flames, then ghost prepass, then ghost colour.
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

		// Unhook the ghost depth prepass from Scene's walk: the flames must be
		// drawn BETWEEN the dungeon and the prepass, or a ghost writing its
		// occluding depth first erases a flame poking into its body. Nulled
		// here (Scene skips a null pipeline); populateCommandBuffer() issues it
		// by hand after the flames. The descriptor sets Scene built stay valid.
		SC.TI[1].T->PT[0].P = nullptr;

		// Cache the floor's top Y for the no-clip under-the-map clamp. Falls
		// back to 0.0f (this scene's floor height) if there's no "floor".
		auto floorIt = SC.InstanceIds.find("floor");
		if(floorIt != SC.InstanceIds.end() && SC.I[floorIt->second]->C != nullptr) {
			worldFloorY = SC.I[floorIt->second]->C->getExtents().yMax;
		}

		// Gameplay collision list: scene.json's auto-fit boxes, plus the hand-authored
		// geometry for the models an auto-fit box gets wrong (the gate's archway, the
		// staircase's profile). See SceneColliders.hpp for why those live in their own
		// data file instead of scene.json or here.
		colliderSet.init(&SC, "assets/scenes/colliders.json");
		allColliders = colliderSet.list();

		// Interactable doors. Each door leaf instance's own origin sits at
		// its hinge (see the Door struct comment above), so promptOffset is
		// the doorway's *centre* in the leaf's local frame instead -- the
		// point the in-range check should measure from, not the jamb it
		// hinges on. Measured off the current SM_WallDoor_Hole_01 geometry by
		// rasterizing its triangles, not eyeballed: the opening is ARCHED, not
		// rectangular -- it spans local Z 2.42..4.77 and Y 0.18..4.10 as a
		// rectangle, then curves in (1.91 wide at Y 4.25, 1.29 at Y 4.75) and
		// closes at about Y 4.85. Re-measure and update this if the asset is
		// regenerated again.
		// openAngleDeg is how FAR the leaf swings open. Which way is no
		// longer a scene decision: every door swings away from whoever
		// opens it (see Door::swingSignAwayFrom), so the sign here only
		// survives as the fallback for a pose that can't be read.
		//
		// lockKeyId is the padlock: leave it "" for a door that just opens,
		// or name the keyId of the pickup that opens it (see addPickup
		// below). lockLabel is what the prompt calls that key and defaults to
		// the id itself. Locking a door is therefore one extra argument on
		// the addDoor() line, not new plumbing -- same as adding the door was.
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
		// Finds a door by its instance id, for the two helpers below that
		// decorate one after addDoor() has made it. Both could have been extra
		// arguments on addDoor instead; they are not, because a door that needs
		// neither -- which is every door but the bookcase -- should not have to
		// read past them on its own line.
		auto findDoor = [&](const char *id) -> Door * {
			auto it = std::find_if(doors.begin(), doors.end(),
								   [&](const Door &x) { return x.instanceId == id; });
			return it == doors.end() ? nullptr : &*it;
		};
		// Replaces the generic padlock prompts and marks a door secret -- one
		// call because it's one decision: a door with its own wording isn't a
		// door, and must not wear the aura. `blocked` may be empty.
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
		// The level's doors, west to east along the intended route. All leaves
		// share the same asset, hinge geometry and doorway centre, so the same
		// promptOffset/openAngleDeg apply to every one -- the only things that
		// differ per door are which wall it stands in (its yaw, carried from
		// scene.json) and what key, if any, it wants. See tools/build_scene.py
		// for the layout these ids come from.
		//
		// iaDoorPanel -- the threshold between the intro corridor and the hub.
		// Unlocked: it exists to be a door the player opens once (learning [E])
		// before any lock is in play, the same teaching role the hall door's
		// padlock used to have but without the dead end.
		addDoor("iaDoorPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f);
		// hbDoorN -- hub north wall, gates branch 1 / room A (the book). Opened
		// with the "iron" key that sits on the hub table.
		addDoor("hbDoorNPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f, "iron", "iron key");
		// hbDoorS -- hub south wall, gates branch 3 / room C (the final key).
		// Opened with "bronze", found in the dark of room B once the player has
		// the torch to see it by.
		addDoor("hbDoorSPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f, "bronze", "bronze key");
		// hbDoorE -- hub east wall, the way out. The only door that opens onto
		// the outside and the last thing between the player and the win box
		// (see gameplay.json's "exit", which does not check a key of its own --
		// this door does). Its hole wall carries no yaw (it stands in an EAST
		// wall), so the leaf's local +X points OUTWARD, unlike every other leaf
		// here -- hence the flipped lock props below and the NEGATIVE open
		// angle, which is what "swing away from the player, into the daylight"
		// means for this mirrored frame. Opened with "gold", the reward for
		// room C's jump puzzle.
		addDoor("hbDoorEPanel", glm::vec3(0.0f, 2.52f, -1.231f), -100.0f, "gold", "gold key");
		// hbShelfPanel -- the secret passage: a bookcase standing in the hub's
		// west wall (tile (0,0)), gating branch 2 / room B (the torch). It is a
		// Door and nothing else -- same hinge convention, same swing, same
		// padlock rule -- differing only in what pays it and what it says.
		//
		// tools/make_bookshelf.py builds SM_Bookshelf_01 in SM_Door_01's own
		// local frame: origin on the hinge, panel hanging to local -Z, a
		// silhouette that follows the hole wall's arch. It sits INSIDE the
		// wall's thickness like every other leaf, which is what lets it swing
		// either way. promptOffset X is 0.25, not 0: the range check should
		// measure from the shelf FACE, not the hinge plane behind it.
		//
		// Locked with "book", which no other lock takes and which the exit
		// does not accept (gameplay.json's exit.keyId is "gold"), so the one
		// book in the level can only ever be spent here.
		addDoor("hbShelfPanel", glm::vec3(0.25f, 2.20f, -1.231f), 100.0f, "book", "old book");
		// What the bookcase says, and only ever with the book in hand (see
		// Door::secret): until then the shelf is furniture and does not glow,
		// so "A book is missing from this shelf" agrees with the player once
		// they have worked the gap out, it does not lead them to it. No blocked
		// line: the far side is unreachable without having opened it, and it
		// never re-locks within a run.
		setSecretDoor("hbShelfPanel",
					  "[E] Slide the book into the gap",
					  "A book is missing from this shelf");
		// Hangs a scene instance on a door as lock hardware. Separate from
		// addDoor() because a door can carry several -- the chains and padlock
		// are two models with different textures.
		//
		// These instances DO carry a "collider" in scene.json: the hardware
		// hangs 0.225 in front of the leaf face. Safe only because the prop
		// loop in GameLogic() syncs that box off the prop's Wm.
		//
		// `flip` puts the hardware on the leaf's OTHER face. make_door_lock.py
		// builds against one face only, so a door approached from the other
		// side would show the bare leaf. A half turn about the leaf's vertical
		// axis through the middle of its thickness and the doorway centre lands
		// the pieces on the far face -- and since both models are symmetric
		// about that z, the turned copy is the mirror the script would have
		// exported. A rotation, not a mirror matrix: mirroring flips the
		// winding and turns the piece inside out under backface culling.
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

			// Grow the auto-fit box outward so the player stops before the
			// hardware is in the torch's reach. The tuck is already at its
			// floor (HAND_TUCK_MIN_REACH), so distance is the only lever.
			// xMax alone, in MODEL space: the hardware is modelled entirely on
			// +X, and `local`'s half turn carries mesh and margin together.
			// Inflating all six faces would push into the jambs.
			Collider *propC = SC.I[it->second]->C;
			if(propC != nullptr) {
				// getExtents() is world-space; at identity it reads back the
				// model box. initAABB resets Wm, which the prop loop rewrites.
				propC->setWorldMatrix(glm::mat4(1.0f));
				AABBextents L = propC->getExtents();
				propC->initAABB(L.xMin, L.yMin, L.zMin,
								L.xMax + Door::LOCK_PROP_KEEPOUT, L.yMax, L.zMax);
			}

			// Same flag decides where the hardware is drawn and which side E
			// works from, so the prompt can't disagree with the screen.
			d->lockFaceSign = flip ? -1.0f : 1.0f;
		};
		// Both instances carry the SAME translate/eulerAngles as the leaf in
		// scene.json (build_scene.py places them there), which is all the
		// placement they need: the models live in its local frame.
		//
		// All three locked leaves take flip=true. The hardware is exported on
		// the leaf's local +X face, and for each of these the player stands on
		// the OTHER side of that: hbDoorN (yaw 90) is approached from the hub
		// to its south, hbDoorS (yaw 270) from the hub to its north, hbDoorE
		// (yaw 0) from the hub to its west while its +X points outward. The
		// half turn lands the chains and lock on the face the player actually
		// sees, and sets the side [E] works from with them.
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

		// The key graph is strictly linear and every key opens exactly one
		// lock: iron -> hbDoorN -> book -> hbShelfPanel -> bronze -> hbDoorS
		// -> gold -> hbDoorE -> win. No spares, no soft-lock: each key is found
		// in the room the previous lock opens onto.

		// World pickups. worldPos is read from the instance's own Wm, since
		// each key's position already lives in scene.json and shouldn't be
		// repeated here.
		// Passing a keyId makes the pickup a key: it goes on the ring when
		// collected and opens any Door whose lockKeyId matches (and the exit,
		// if exit.keyId names it). It is spent on first use -- one lock per
		// key, no take-backs.
		// handTiltDeg/handOffset are how this item sits in the hand once
		// carried. Both have defaults because the key is the item this pose was
		// built for and three of the four calls below are keys -- see
		// HAND_KEY_TILT_DEG and HAND_KEY_OFFSET.
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
			// Uniform scale from column 0's length, so scene.json's "scale" is
			// the one place it lives -- the held/dropped poses take it back
			// from here.
			p.worldScale = glm::length(glm::vec3(p.inst->Wm[0]));
			pickups.push_back(p);
		};
		// Three keys, three distinct ids -- iron, bronze, gold -- one per
		// locked door, each found in the room the previous door opens onto.
		// They are the same mesh but NOT interchangeable ids, because here the
		// player always has exactly the key the next lock wants and never a
		// choice of which to try: there is no spare to mismatch. The scarcity
		// is the route, not the count.
		addPickup("hbKeyIron", "iron");     // on the hub table
		addPickup("rbKeyBronze", "bronze"); // dark NW corner of room B
		addPickup("rcKeyGold", "gold");     // atop the barrels in room C
		// The book on the hub table, which opens the bookcase and nothing
		// else. It rides the SAME machinery as the
		// keys (keyRing, findKeyInRing, consumeKey) on purpose -- "carry a
		// thing to the lock that wants it, and spend it there" is already the
		// rule of this level, and a second parallel system for one object would
		// only be a second place for it to go wrong.
		//
		// The rule it does inherit and is worth knowing about: one free hand.
		// Picking the book up while carrying a key puts that key on the floor
		// at the player's feet (see the pickup branch in GameLogic), which is
		// not a bug to fix here -- the torch owns the other hand, and a player
		// juggling the ring is the cost of that decision, not of this book.
		//
		// The tilt is the book's own: its mesh lies FLAT in its local frame (x
		// spine to fore-edge, z the height of the page, y the thickness, see
		// make_bookshelf.py), so ~90 about X is what stands it up with the
		// cover toward the camera, and the rest is the same eyeballed turn the
		// key carries.
		//
		// The offset is HAND_KEY_OFFSET raised by 0.16. Not a taste decision
		// about books: the key's origin sits near one end of its mesh so the
		// key hangs UP out of the anchor, while the book's sits at the middle
		// of its page height (local Z is -0.170..0.170, symmetric) so the book
		// straddles it and half of it hangs below. Same number, lower object.
		// The lift puts the two at roughly the same height on screen.
		addPickup("raBook", "book", glm::vec3(84.0f, -24.0f, 0.0f),
				  HAND_KEY_OFFSET + glm::vec3(0.0f, 0.16f, 0.0f));

		// Where the book ends up once spent: in the gap on the bookcase's
		// third shelf, riding the leaf's local frame like the padlocks so it
		// swings with the case. See Door::LockProp::whenUnlocked.
		//
		// The three numbers ALSO live in make_bookshelf.py's BOOK_SLOT_*,
		// which leaves the gap in the row of books -- move one, move the other:
		//   x 0.454  the surrounding spine plane minus the book's spine bulge
		//   y 1.730  the shelf face plus half the book's height (mesh centred on Z)
		//   z -1.266 the gap centre; on the case's centre line, so a wider gap
		//            doesn't move it. The book doesn't fill the gap -- the case
		//            swings open the moment it lands, so only the frame before
		//            is ever read.
		//
		// The rotation stands a flat book spine-out on the shelf:
		// rotY(180) * rotX(-90). A rotation, not a mirror (which would flip
		// the winding).
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

		// The player's spawn pose, captured before anything can move it. See
		// spawnPos's declaration: this is what restartRun() puts them back to.
		spawnPos = camPos;
		spawnYaw = camYaw;
		spawnPitch = camPitch;

		// The rules of the game: hunt timings, the win box, the ghost patrols.
		// A data file for the same reason lights.json is -- every number is a
		// tuning decision.
		{
			std::ifstream ifs("assets/scenes/gameplay.json");
			if(!ifs.is_open()) {
				std::cout << "gameplay.json not found: default hunt timings, no ghosts\n";
				// Still has to be armed: a default HuntCycle has a phase timer
				// of 0 and would fall straight into a hunt.
				huntCycle.init(nlohmann::json::object());
			} else {
				// ignore_comments, like every other scene file here.
				nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);

				// Unconditional, and with an empty object if the block is
				// absent: init() treats every key as optional and finishes by
				// arming the clock, which has to happen either way.
				huntCycle.init(js.value("hunt", nlohmann::json::object()));

				if(js.contains("exit")) {
					const nlohmann::json &e = js["exit"];
					if(e.contains("box") && e["box"].size() == 6) {
						glm::vec3 a(e["box"][0].get<float>(), e["box"][1].get<float>(), e["box"][2].get<float>());
						glm::vec3 b(e["box"][3].get<float>(), e["box"][4].get<float>(), e["box"][5].get<float>());
						// min/max rather than trusting the authored order, so
						// writing the two corners the other way round still
						// describes the same box instead of an empty one.
						exitBoxMin = glm::min(a, b);
						exitBoxMax = glm::max(a, b);
						exitHasBox = true;
					} else {
						std::cout << "gameplay.json: \"exit\" needs a 6-number \"box\", the run can't be won\n";
					}
					exitRequiresKey = e.value("requiresKey", true);
					exitKeyId = e.value("keyId", std::string(""));
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
					// Two is the minimum that describes a path at all; one
					// waypoint is a ghost standing still, which is almost
					// certainly a typo rather than an intention.
					if(gh.waypoints.size() < 2) {
						std::cout << "gameplay.json: ghost '" << id
								  << "' needs at least 2 waypoints, skipped\n";
						continue;
					}
					gh.pos = gh.waypoints[0];
					// Staggered so identically-authored ghosts don't hover in
					// lockstep, same trick as the torches' flicker phase.
					gh.bobPhase = (float)ghosts.size() * 2.399963f;
					ghosts.push_back(gh);
				}
				std::cout << "gameplay.json: " << ghosts.size() << " ghosts loaded\n";

				// Fit the ghost's collision size from the mesh, not a
				// hand-measured constant. All ghosts share Ghost.gltf, so one
				// fit covers them.
				if(!ghosts.empty()) {
					Collider fit;
					fit.fitAABB(SC.M[ghosts[0].inst->Mid]);
					AABBextents E = fit.getExtents();	// fit's Wm is identity: local space

					// fitAABB reads the raw mesh, blind to scene.json's "scale".
					// Column 0's length is the uniform scale factor, so
					// multiplying it in makes shrinking a ghost shrink what it
					// collides as.
					float instScale = glm::length(glm::vec3(ghosts[0].inst->Wm[0]));

					ghostBodyBottom = E.yMin * instScale;
					ghostBodyTop = E.yMax * instScale;

					float halfX = 0.5f * (E.xMax - E.xMin);
					float halfZ = 0.5f * (E.zMax - E.zMin);
					ghostRadius = ghostXZFitShrink * 0.5f * (halfX + halfZ) * instScale;

					std::cout << "Ghost collision fitted from mesh (scale " << instScale
							  << "): radius " << ghostRadius
							  << ", vertical [" << ghostBodyBottom << ", " << ghostBodyTop << "]\n";

					// Patrol legs are walked with no wall resolution, so a leg
					// authored through a wall doesn't fail loudly -- the ghost
					// glides through, and the next hunt wedges. Checked once
					// here now the radius is known.
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

		// Held torch. Starts on the FLOOR at its authored pose; picked up with
		// [E]. Only once handTorchCollected is set does GameLogic rebuild its
		// Wm from the camera. The authored pose is captured here first.
		{
			auto it = SC.InstanceIds.find("handTorch");
			if(it == SC.InstanceIds.end()) {
				std::cout << "Hand torch instance 'handTorch' not found, skipping\n";
			} else {
				handTorchInst = SC.I[it->second];
				handTorchSpawnWm = handTorchInst->Wm;
				// Aim point for the floor pickup, lifted off the instance
				// origin (which sits below the mesh once the torch is laid on
				// its side) so the crosshair lands on the torch body itself.
				handTorchWorldPos = glm::vec3(handTorchInst->Wm[3]) +
									glm::vec3(0.0f, 0.35f, 0.0f);
			}
		}

		// Torch flames. DSglobal isn't populated yet (that happens in
		// pipelinesAndDescriptorSetsInit(), after the descriptor pool
		// exists), but its address is stable, so capturing a pointer to it
		// now and reading through it later is safe -- same reasoning as
		// handTorchInst above.
		//
		// maxInstances covers every torch AND candle mesh in the level (each
		// dungeonCandle instance gets a TorchFlame too, burning or not -- see
		// flames.json). The rebuilt level runs ~13 wall torches + the held
		// torch + ~8 candles, so this is set well above that; spawn() past the
		// cap fails silently (see addTorchFlame below), leaving a torch/candle
		// with no fire and no light instead of an error. Still under MAX_LIGHTS
		// (32) and NUM_SHADOW_CUBES (32), which the dynamic pool arbitrates.
		flame.init(this, &DSLglobal, &DSglobal, 28);

		// No DSLglobal/DSglobal here: the daylight quads aren't shaded and read
		// nothing the app-wide uniform carries, so they bind sets of their
		// own. Two of them -- the upright wall of light and the ground it
		// stands on. See ExitGlow.hpp, and EXIT_GLOW_CENTER for why the
		// outward-swinging door makes the second one necessary.
		exitGlow.init(this, EXIT_GLOW_COUNT);

		debugLines.init(this);

		// seed just spreads each flame's sway/flicker phase: index * an
		// irrational-ish constant, no RNG needed for one call.
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
			// Irrational-ish stride, so phases never land on a common multiple.
			float seed = (float)torchFlames.size() * 2.3971f;
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
			// At rest for a burning flame (no catching-fire on load), 0 for
			// unlit so it plays in full when lit.
			tf.ignitionScale = burning ? 1.0f : 0.0f;
			// Static flames only: the held torch's anchor moves every frame.
			if(!heldByCamera) {
				tf.anchorWorld = glm::vec3(inst->Wm * glm::vec4(anchor, 1.0f));
			}
			// Automatic: every flame is the held torch or a static object, and
			// every static one belongs in the pool.
			tf.shadowCandidate = !heldByCamera;
			if(tf.shadowCandidate) {
				// Static, so its face matrices are computed once here, not
				// every frame like the held torch.
				tf.shadowFaceMatrices = cubeFaceMatricesFor(tf.anchorWorld);
			}
			// Into a different part of the CPU noise field. Scaled up because
			// fireNoise hashes on the integer lattice.
			tf.phase = seed * 37.0f;
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

				// Resolve each model NAME to its Mid once, so spawning is an
				// O(1) lookup per instance.
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

		// Surface parameters for the BRDF, one per model.
		materials.init(&SC, "assets/scenes/materials.json");

		// After Scene::init: a light can be anchored to an instance and needs
		// that instance's world matrix.
		sceneLights.init(&SC, "assets/scenes/lights.json");

		// After sceneLights.init(): needs the resolved world position of every
		// shadow-casting light, which instance+offset lights only have once
		// SceneLights has read scene.json's world matrices.
		computeShadowMatrices();

		// Everything past lights.json's fixed slots and before the held torch's
		// reserved last one is the dynamic pool. Captured once, not a literal,
		// so it survives a change to lights.json's torch count.
		dynamicShadowSlotBase = activeCubeShadows;
		dynamicSlotOccupant.fill(-1);
		// Nothing rendered yet: the first updateUniformBuffer() diff queues
		// every slot for its one render.
		lastRenderedOccupant.fill(SHADOW_SLOT_UNSET);
		// The render loop walks [0, activeCubeShadows) every frame, so the
		// dynamic pool must be counted even while empty -- a slot filled
		// mid-game would otherwise never render.
		activeCubeShadows = std::max(activeCubeShadows, HAND_TORCH_SHADOW_INDEX);

		// The held torch's slot isn't in lights.json, so
		// computeShadowMatrices() never counts it. Its matrices are filled
		// every frame by updateHandTorchShadow(), but the render loop still
		// needs to know the slot is in play.
		if(handTorchInst != nullptr) {
			activeCubeShadows = std::max(activeCubeShadows, HAND_TORCH_SHADOW_INDEX + 1);
		}

		// initializes the textual output
		txt.init(this, windowWidth, windowHeight);
		// initializes the flat-quad background/highlight layer for the cheat HUD
		uiQuad.init(this, windowWidth, windowHeight);
		// The center-screen crosshair dot. Distinct submitOrder/buffer name
		// from uiQuad, so the two don't collide over one named command buffer;
		// same for the three below.
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

		// Wires the cheat HUD to the actual cheat flags.
		hud.init(&txt, &uiQuad);
		pauseMenu.init(&txt, &pauseQuad);
		// windowTitle, reused so the launch screen's title lives in one place.
		startScreen.init(&txt, &startScreenQuad, windowTitle);
		startScreen.setOpen(true, windowWidth, windowHeight);
		settingsMenu.init(&txt, &settingsQuad);
		// The two sliders a normal player reaches. Same named methods as the
		// cheat HUD's copies, so the two can't drift.
		settingsMenu.addSlider("Render Scale", &renderScale, 0.4f, 1.0f, 0.05f,
							   [this]() { applyRenderScaleChange(); },
							   [this](float v) { return formatRenderScale(v); });
		settingsMenu.addSlider("MSAA", &msaaLevel, 0.0f, maxMsaaLevel, 1.0f,
							   [this]() { applyMsaaChange(); },
							   [this](float v) { return formatMsaaLevel(v); });
		hud.addToggle("Collision", &cheats.collisionEnabled);
		hud.addToggle("Show Coordinates", &cheats.showCoordinates);

		// Gameplay rows. "Hunt" forces and holds the hunt phase; "Ghosts Can
		// Catch" lets you watch it longer than the first ghost takes to reach you.
		hud.addToggle("Hunt", &huntCycle.forceHunt);
		hud.addToggle("Ghosts Can Catch", &cheats.ghostsCanCatch);

		// Lighting rows, in the order you'd reach for them: which sources are
		// on, then how they're shaded. Spotlight/Ambient point into sceneLights
		// (it owns them); Torches/Holding Torch into cheats (their lights never
		// go through SceneLights); the rest into cheats -> gubo.debugFlags. No
		// "Sun" row -- the scene has no direct light; add it back with the sun.
		hud.addToggle("Torches", &cheats.roomTorchesEnabled);
		hud.addToggle("Holding Torch", &cheats.handTorchEnabled);
		hud.addToggle("Spotlight", &sceneLights.spotEnabled);
		hud.addToggle("Ambient Light", &sceneLights.ambientEnabled);
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

		// Render Scale and MSAA, also on SettingsMenu with the same named
		// methods so the two copies can't drift.
		hud.addSlider("Render Scale", &renderScale, 0.4f, 1.0f, 0.05f,
					  [this]() { applyRenderScaleChange(); },
					  [this](float v) { return formatRenderScale(v); });
		hud.addSlider("MSAA", &msaaLevel, 0.0f, maxMsaaLevel, 1.0f,
					  [this]() { applyMsaaChange(); },
					  [this](float v) { return formatMsaaLevel(v); });
	}

	// Does slot t currently have a light in it (is anything sampling its cube
	// map)? One definition, because "occupied" means three things: a
	// lights.json slot by construction, a dynamic-pool one only while
	// updateDynamicShadowSlots() has given it a flame, the held torch's only
	// while the torch exists.
	bool cubeSlotOccupied(int t) const {
		if(t == HAND_TORCH_SHADOW_INDEX) {
			return HAND_TORCH_SHADOW_INDEX < activeCubeShadows && cheats.torchShadowsEnabled;
		}
		return (t < dynamicShadowSlotBase) || (dynamicSlotOccupant[t] != -1);
	}

	// The g (falloff reference distance) a flame's point light is uploaded
	// with. One definition, called by the light-append loop and the
	// shadow-reach computation, so the two can't drift.
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

	// Builds the view-projection matrix each of the shadow passes renders
	// with, in LightData::shadowIndex order. Called once, from localInit()
	// right after sceneLights.init(): the torches never move, so there is
	// nothing here that needs recomputing per frame.
	//
	// Reads sceneLights.all() rather than scene.json/InstanceIds directly: a
	// point light's world position is instance-plus-offset (see
	// SceneLights::init), and re-deriving that here would be a second copy of
	// logic that already lives in exactly one place.
	void computeShadowMatrices() {
		// A torch's far plane (cubeFaceMatricesFor() -> TORCH_SHADOW_FAR_CONST):
		// now sized to the dungeon's own footprint rather than "wherever the
		// torch has faded to nothing", see that constant's own comment for
		// why a falloff-sized far plane went wrong. createCubeShadowMaps()
		// needs the same number for the color attachment's clear value,
		// which is why it's a member and not a local here.

		for(const LightData &L : sceneLights.all()) {
			if(L.shadowIndex < 0) {
				continue;
			}

			// A point light's cube map: six 90-degree perspective faces,
			// axis-aligned on world X/Y/Z (CUBE_FACE_DIR/CUBE_FACE_UP,
			// CubeShadowMap.hpp), covering the WHOLE sphere with no seam and
			// no hand-tuned aim per torch -- unlike the old two-map
			// front/back workaround, this needs no per-torch authoring at
			// all, so it drops TORCH_SHADOW_DIR entirely.
			//
			// No Y-flip here: a cube map is sampled by direction
			// (samplerCube), never rasterized to the screen, so there is no
			// Vulkan-vs-GL NDC mismatch to correct for -- flipping would
			// only mis-rotate which face's texels land where.
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

	// Priority-based reassignment for the dynamic shadow-cube pool, among every
	// flame marked shadowCandidate. The empty-slot pass gives one to every
	// candidate first, so in this scene (12 candidates, 31 slots) they all keep
	// a permanent shadow; the contest logic only fires on a level with more
	// shadow-worthy lights than slots.
	//
	// A plain "N nearest per tick" rule thrashes on the boundary between two
	// candidates, and every flip forces a fresh render. SHADOW_SWAP_MARGIN
	// (only swap if genuinely closer) and SHADOW_REASSIGN_INTERVAL (revisited a
	// few times a second) fix that. `forward` only changes the ORDER candidates
	// are considered in (facingBiasedDistSq()), never whether one gets a slot.
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

		// The HUD's "Torch/Candle Shadows" toggles: a flame whose category is
		// off is never a waiting candidate, and a slot it held is freed here so
		// switching off drops its shadow this tick. A flame switched off
		// entirely goes the same way (it isn't burning, so it casts nothing).
		auto categoryEnabled = [&](const TorchFlame &tf) {
			if(!flameBurning(tf)) return false;
			return tf.isCandle ? cheats.candleShadowsEnabled : cheats.torchShadowsEnabled;
		};

		// A torch whose light no longer reaches anything the frame draws (same
		// test the light-append loop culls by) has no on-screen shadow to cast,
		// so it gives its cube slot back. An existing occupant is judged with a
		// reach margin (SHADOW_VIEW_KEEP_MARGIN) so a torch hovering on the cone
		// boundary while the player turns doesn't re-render its cube every pass;
		// a fresh candidate must clear the un-margined bar to claim one.
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

		// Empty slots first, unconditionally: an empty slot never "wins" over a
		// candidate, it just has nothing in it yet.
		size_t wi = 0;
		for(int s = base; s < base + count && wi < waiting.size(); s++) {
			if(dynamicSlotOccupant[s] != -1) {
				continue;
			}
			assignSlot(s, waiting[wi].flameIdx);
			wi++;
		}

		// Occupied slots, farthest occupant first, contested against the best
		// remaining waiter: once one pairing fails the margin, every later one
		// does too, so this stops at the first miss.
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

	// Local-space bounding sphere per MODEL index (xyz centre, w radius),
	// filled on first use (fitAABB() walks every vertex). w < 0 = not fitted.
	// Feeds the per-face cull in recordCubeSlotFaces(): a sphere is the only
	// bound cheap enough to test 6x per instance per slot, and the sphere
	// AROUND the AABB errs safe (a loose cull costs a draw, a tight one a shadow).
	std::vector<glm::vec4> modelSphereCache;
	// Scratch for the same cull: one slot's shadow casters with their
	// world-space spheres. A member so it reuses its allocation.
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
	// is its centre relative to the light (the frustum apex). By hand, not
	// plane extraction: the faces are axis-aligned and square at 90 degrees,
	// so the test is a handful of component compares -- and it runs once per
	// instance per face.
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

	// One cube slot's six-face render. Every caller goes through
	// submitCubeShadowCaptures(): the held torch each frame, others when their
	// occupant changes or a mover invalidates them.
	//
	// ownerInst: the instance this slot's flame is anchored to, excluded from
	// its own shadow pass.
	//
	// cullPerFace: drop instances outside the face being drawn -- a ~6x cut in
	// draws, and what makes a per-frame re-capture affordable. Only sound for a
	// buffer recorded and submitted the same frame (a replayed one would use a
	// stale visible set), so it's a flag, not an assumption.
	//
	// faceMask: which faces to draw, bit f for face f. A face left out keeps
	// its texels (the point, see pendingFaceMask) -- but must be in the mask
	// AT LEAST ONCE before anything samples the cube, or it stays
	// VK_IMAGE_LAYOUT_UNDEFINED and can't be bound. The occupant diff asks for
	// ALL_CUBE_FACES the first time it sees each slot.
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
					// Largest column length: a non-uniform scale's sphere must
					// take the biggest or it misses the stretched axis.
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

			// The draw loop below is the real cost -- skipped for an
			// unoccupied slot, but the begin/end above still runs to clear the
			// image and transition it to SHADER_READ_ONLY_OPTIMAL, which the
			// samplerCube array requires of every slot even when unsampled.
			if(slotOccupied) {
				PShadowCube.bind(commandBuffer);
				// Set 1: this torch's current matrices/position, from the
				// uniform buffer updateUniformBuffer() maps every frame --
				// see ShadowCube.vert's header for why this can't be a push
				// constant. Only the face INDEX is still one, since that
				// genuinely never changes once recorded.
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

	// Marks, in pendingFaceMask, every cube FACE whose cached capture a MOVING
	// occluder has invalidated -- the other half of the staleness question the
	// lastRenderedOccupant diff answers. A face is marked when it holds a mover
	// that has moved since it was last drawn, or held one last capture and
	// doesn't now (its texels still have the ghost on them). "In range" is each
	// light's OWN reach, measured against the mover's bounding sphere. No
	// per-frame budget -- a shadow that lags its ghost is visible.
	//
	// Called from updateUniformBuffer() after the occupant diff, after
	// GameLogic() has moved the ghosts and doors and
	// updateDynamicShadowSlots() has settled torchLightPos[].
	void queueMoverCubeSlotRenders() {
		if(!moverListBuilt) {
			// A non-occluder can't invalidate a capture. The ghosts are off in
			// materials.json and never reach this list.
			auto addMover = [&](Instance *inst) {
				if(inst != nullptr && materials.forModel(inst->Mid).castsShadow) {
					movingOccluders.push_back(inst);
				}
			};
			for(const Ghost &g : ghosts) {
				addMover(g.inst);
			}
			for(const Door &d : doors) {
				addMover(d.inst);
			}
			// The held torch: static while on the floor, so this costs nothing
			// until pickup jumps its Wm. That jump is what erases the stale
			// floor-shadow from whichever torch captured it; once held it's not
			// drawn as an occluder, so tracking it here only ever cleans up.
			addMover(handTorchInst);
			// Zero matrix, not identity: an authored-identity pose would read
			// as "hasn't moved" on frame one and never get its first capture.
			movingOccluderWm.assign(movingOccluders.size(), glm::mat4(0.0f));
			moverListBuilt = true;
		}

		// Which faces of each slot hold a mover this frame, and which of those
		// hold one that has moved since that face was last drawn.
		std::array<uint8_t, NUM_SHADOW_CUBES> facesNow{};
		std::array<uint8_t, NUM_SHADOW_CUBES> facesStale{};

		for(size_t m = 0; m < movingOccluders.size(); m++) {
			const glm::mat4 &wm = movingOccluders[m]->Wm;

			// The held torch in hand: it jumps every frame but isn't drawn as
			// an occluder, so letting it through would mark a face stale every
			// frame for an identical redraw. Skipping it loses no cleanup --
			// the pickup frame's Wm jump is handled by "held a mover last
			// capture and doesn't now".
			if(movingOccluders[m] == handTorchInst && handTorchCollected
			   && cheats.handTorchEnabled && !cheats.handTorchModelCastsShadowWhenHeld) {
				movingOccluderWm[m] = wm;
				continue;
			}

			const bool moved = (wm != movingOccluderWm[m]);
			// World-space bounding sphere: the mover's extent must be in the
			// range test, or a ghost whose ORIGIN is just outside a light's
			// reach while its body crosses in would go untracked.
			const glm::vec4 &local = modelSphere(movingOccluders[m]->Mid);
			const glm::vec3 p = glm::vec3(wm * glm::vec4(glm::vec3(local), 1.0f));
			const float scale = std::max({glm::length(glm::vec3(wm[0])),
										  glm::length(glm::vec3(wm[1])),
										  glm::length(glm::vec3(wm[2]))});
			const float r = local.w * scale;

			// The held torch's slot is included: a ghost walking into its beam
			// while the player stands still has to mark it too.
			for(int t = 0; t <= HAND_TORCH_SHADOW_INDEX; t++) {
				if(!cubeSlotOccupied(t)) {
					continue;
				}
				if(glm::distance(p, torchLightPos[t]) - r > torchShadowReach[t]) {
					continue;
				}
				// Which of the six directions the mover sits in -- the same
				// test the capture culls by.
				const glm::vec3 rel = p - torchLightPos[t];
				for(int face = 0; face < 6; face++) {
					if(!sphereInCubeFace(rel, r, face)) {
						continue;
					}
					facesNow[t] |= (uint8_t)(1u << face);
					if(moved) {
						facesStale[t] |= (uint8_t)(1u << face);
					}
				}
			}
			movingOccluderWm[m] = wm;
		}

		for(int t = 0; t <= HAND_TORCH_SHADOW_INDEX; t++) {
			// Faces where something moved, plus faces a mover has just left
			// (still drawn on them). ORed into what the occupant diff asked for.
			pendingFaceMask[t] |= facesStale[t] | (uint8_t)(slotFaceHadMover[t] & ~facesNow[t]);
			slotFaceHadMover[t] = facesNow[t];
		}
	}

	// Allocates the per-frame shadow command buffers, their pool and their
	// fences. Called from pipelinesAndDescriptorSetsInit(), which runs once at
	// startup and again after every swapchain recreation -- hence sized off
	// swapChainImages, and hence the matching destroy in
	// pipelinesAndDescriptorSetsCleanup().
	void createShadowCommandBuffers() {
		destroyShadowCommandBuffers();

		QueueFamilyIndices qfi = findQueueFamilies(physicalDevice);
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = qfi.graphicsFamily.value();
		// The whole reason for a pool of our own: without this bit the buffers
		// below could be recorded exactly once each.
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

		// Created signalled: the first frame waits on a fence nothing has
		// submitted yet, and an unsignalled one would hang there forever.
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

	// Safe to call with nothing allocated (it is, once, from
	// createShadowCommandBuffers()). Both call sites run with the device idle:
	// recreateSwapChain() waits before cleaning up, and so does mainLoop()
	// before cleanup().
	void destroyShadowCommandBuffers() {
		for(VkFence f : shadowCBFence) {
			vkDestroyFence(device, f, nullptr);
		}
		shadowCBFence.clear();
		// Destroying the pool frees every buffer allocated from it, so the
		// buffers themselves need no separate free.
		if(shadowCommandPool != VK_NULL_HANDLE) {
			vkDestroyCommandPool(device, shadowCommandPool, nullptr);
			shadowCommandPool = VK_NULL_HANDLE;
		}
		shadowCB.clear();
	}

	// Records and submits this frame's cube shadow captures: the held torch
	// (never cacheable) plus every slot pendingFaceMask marks (light changed
	// hands, or a ghost/door moved inside it).
	//
	// Called at the end of updateUniformBuffer(), after DSshadowCube[t] and
	// every instance's DS[0][1] have this frame's matrices -- this pass binds
	// the same per-instance set the main pass does.
	//
	// Its own submission, not part of the "main" buffer (recorded once and
	// replayed, can't express "these slots this frame"). Correct because:
	//  - it's submitted BEFORE the main buffer on the same queue, and the
	//    barrier at the end orders its writes against every later command
	//    including the main pass's sampling. Replaces a mid-frame vkQueueWaitIdle.
	//  - each swapchain image has its own buffer and fence, waited on before
	//    re-recording. In the steady state the wait returns immediately.
	void submitCubeShadowCaptures(int currentImage) {
		if(shadowCB.empty()) {
			return;	// nothing allocated yet (first frames of a swapchain rebuild)
		}

		// The held torch's staleness, settled before recording so the "any
		// work" test sees it. All six faces whenever its light MOVED (every
		// face was shot from that position); otherwise only what a mover
		// marked. Occupancy is in the comparison so switching torch shadows
		// off clears the map.
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

		// Nothing stale anywhere: no buffer, no submit, no fence traffic. This
		// is the steady state whenever the player and the ghosts are both still.
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
	// per-torch depth + the 36 face framebuffers), plus the one sampler they
	// all share. Called from pipelinesAndDescriptorSetsInit(), right after
	// RPShadowCubeCompat.create() -- these framebuffers are only valid once
	// that render pass exists, since they're built against its .renderPass
	// handle (see the RPShadowCubeCompat member comment for why that render
	// pass is only used for this, never rendered into itself).
	//
	// Lives here rather than in CubeShadowMap.hpp because createImage/
	// createImageView/findDepthFormat are PROTECTED members of BaseProject:
	// only this class's own methods can call them (see CubeShadowMap.hpp's
	// header comment), the same reason every other Vulkan resource in this
	// file -- RP, the post chain -- is built in a method here
	// rather than in a free-standing helper.
	void createCubeShadowMaps() {
		// Shared by every torch: same resolution and format. No mipmaps (a
		// shadow lookup always samples level 0).
		//
		// NEAREST, not LINEAR: these cubes store a DISTANCE, and averaging four
		// distances across a silhouette returns one that belongs to neither
		// surface -- an unoccluded fragment reads as shadowed, or vice versa.
		// The error scales with the depth GAP (metres, not texels), which is
		// what forced the old 0.35 grazing bias and let a torch light the
		// chains through a closed door. One texel makes the comparison honest.
		cubeShadowSampler.init(this, VK_FILTER_NEAREST, VK_FILTER_NEAREST,
								VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
								VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
								VK_SAMPLER_MIPMAP_MODE_LINEAR,
								VK_FALSE, 1.0f, 1.0f);

		const VkFormat colorFmt = VK_FORMAT_R32_SFLOAT;
		const VkFormat depthFmt = findDepthFormat();

		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			CubeShadowMap &c = torchCube[i];

			// The 6-layer colour image. CUBE_COMPATIBLE_BIT lets the CUBE view
			// below treat its 6 layers as faces, not an array.
			createImage(SHADOW_MAP_RES, SHADOW_MAP_RES, 1, 6,
						VK_SAMPLE_COUNT_1_BIT, colorFmt, VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						c.colorImage, c.colorMemory);

			// The one view CookTorrance.frag's samplerCube reads: all 6 layers,
			// TYPE_CUBE.
			c.cubeView = createImageView(c.colorImage, colorFmt, VK_IMAGE_ASPECT_COLOR_BIT,
										  1, VK_IMAGE_VIEW_TYPE_CUBE, 6);

			// One 2D view per layer for the framebuffers: a cube view can't be
			// a render target, and createImageView fixes baseArrayLayer at 0,
			// so these need a raw vkCreateImageView.
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

			// One depth image per torch, reused across its 6 faces (see the
			// CubeShadowMap::depthImage comment). Never sampled, so plain 2D.
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

	// Here you create your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsInit() {
		// creates the render passes. All of them first: RenderPass::create()
		// is what actually allocates each attachment's image and sampler, and
		// the descriptor sets below have to point at those, so nothing may be
		// bound until every pass exists.
		RP.create();
		RPbright.create();
		RPblurH.create();
		RPblurV.create();
		RPcomposite.create();

		P.create(&RP);
		// Same pass as P: the ghosts draw inside the scene pass, sharing its
		// depth buffer and HDR attachment (where bloom finds their rim).
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

		// Wire the chain together. Each pass reads the previous pass's colour
		// attachment as a texture. RP's sampled output is its RESOLVE
		// attachment (the colour one is multisampled and discarded).
		VkDescriptorImageInfo sceneTex = RP.attachments[RP.resolveAttIdx].getViewAndSampler();
		VkDescriptorImageInfo brightTex = RPbright.attachments[0].getViewAndSampler();
		VkDescriptorImageInfo blurHTex = RPblurH.attachments[0].getViewAndSampler();
		VkDescriptorImageInfo blurVTex = RPblurV.attachments[0].getViewAndSampler();

		DSbright.init(this, &DSLpost1, {sceneTex});
		DSblurH.init(this, &DSLpost1, {brightTex});
		DSblurV.init(this, &DSLpost1, {blurHTex});
		// The composite is the only pass that needs two: the scene at full
		// brightness, and the blurred bloom to add on top of it.
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
		// Same RP as the scene: flame, exit glow and debug lines all draw
		// inside it, sharing the depth buffer and writing over-1.0 colours into
		// the HDR attachment where bloom finds them.
		flame.pipelinesAndDescriptorSetsInit(&RP);
		exitGlow.pipelinesAndDescriptorSetsInit(&RP);
		debugLines.pipelinesAndDescriptorSetsInit(&RP);

		// One per swapchain image, so this has to be rebuilt whenever the
		// swapchain is. See submitCubeShadowCaptures().
		createShadowCommandBuffers();
	}

	// Here you destroy your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsCleanup() {
		// Before anything else, and safe here because both paths into this
		// (recreateSwapChain() and cleanup()) wait for the device to be idle
		// first -- so nothing this frees is still in flight.
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

		// Before SC.localCleanup(): that frees scene.json's colliders, which
		// colliderSet also points at. Drop the shared list first.
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
	
	static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void *Params) {
		Castlescape *T = (Castlescape *)Params;
		T->populateCommandBuffer(commandBuffer, currentImage);
	}

	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
		// The whole HDR chain goes into this one command buffer, in order;
		// ordering between passes comes from their render pass dependencies.
		// The text and HUD passes are separate buffers submitted after, so
		// they draw on top of the composited frame.

		// No cube shadow pass here at all, the held torch's included: every one
		// of them is recorded and submitted per frame by
		// submitCubeShadowCaptures() instead, just before this buffer is
		// submitted. This one is recorded once per swapchain image and replayed
		// unmodified afterwards (see ShadowCubeUniformBufferObject's comment),
		// which can express neither "only the slots that went stale this frame"
		// nor the per-face culling that makes those re-captures affordable --
		// a visible set decided at record time stops being true the moment the
		// light or the occluders move, and the held torch's light moves with
		// the camera every frame.
		//
		// 1. The scene, into the offscreen HDR target.
		RP.begin(commandBuffer, currentImage);
		// The dungeon only: the Spectral depth prepass was unhooked from this
		// walk in localInit() so the flames can slot in ahead of it.
		SC.populateCommandBuffer(commandBuffer, 0, currentImage);

		// Flames BEFORE the ghosts. They depth-test against the dungeon but
		// land before any ghost depth, so a flame poking into a ghost's body
		// isn't erased -- "the torch snuffs when you step inside a ghost" needs
		// you to SEE it lit right up until it goes out. A ghost over a flame
		// still blends in front; the flame just glows through the shell.
		flame.populateCommandBuffer(commandBuffer, currentImage);

		// The ghost DEPTH PREPASS, by hand (see localInit()). Nearest
		// ghost-surface depth only (LESS, no colour), so the colour pass keeps
		// just that layer. SpectralDepth.frag's near-fade lets the flames and
		// exit glow show through a ghost the player is standing in.
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

		// The ghosts' COLOUR pass, by hand for the same reason. The same
		// instances again through Pspectral; only the nearest layer per pixel
		// survives LESS_OR_EQUAL, keeping the feet from glowing through the
		// body. One pipeline bind covers all three ghosts.
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

		// After the scene (so the castle depth masks this quad to the arch)
		// and the flames (the only other thing it blends against).
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

	// Here is where you update the uniforms.
	// Very likely this will be where you will be writing the logic of your application.
	void updateUniformBuffer(uint32_t currentImage) {
		static bool debounce = false;
		static int curDebounce = 0;

		// ESC opens/closes the pause menu (Quit is reached through it).
		// Edge-triggered. Ignored while the HUD, launch screen or settings
		// screen is open -- one modal at a time, and settingsMenu specifically
		// closes PauseMenu, so ESC would otherwise reopen it underneath.
		bool escPressed = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
		if(escPressed && !escKeyWasPressed && !hud.isOpen() && !startScreen.isOpen()
		   && !settingsMenu.isOpen()) {
			pauseMenu.setOpen(!pauseMenu.isOpen(), windowWidth, windowHeight);
		}
		escKeyWasPressed = escPressed;

		// moves the view
		float deltaT = GameLogic();

		// Pause menu, launch screen and settings screen all mean "stop time":
		// zeroing deltaT here freezes every deltaT-driven animation below
		// (flicker, shadow reassignment, ...) at once. The cheat HUD does NOT
		// zero it -- torches flickering while flipping a debug flag is fine.
		if(pauseMenu.isOpen() || startScreen.isOpen() || settingsMenu.isOpen()) {
			deltaT = 0.0f;
		}

		// Free-running clock for shader-side animation (currently just the
		// flame's UV scroll). Unlike elapsedT below this never resets.
		static float simTime = 0.0f;
		simTime += deltaT;

		// defines the global parameters for the uniform
		GlobalUniformBufferObject gubo{};

		// Every light comes from lights.json, sun included. No intensity factor:
		// with a BRDF returning [0,1] (L09 s.42) a white source is (1,1,1) and
		// the tone map handles the range. Strength is g and beta instead.
		const std::vector<LightData> &lights = sceneLights.update(deltaT);
		gubo.lightCount = (int)lights.size();
		for(int i = 0; i < gubo.lightCount; i++) {
			gubo.lights[i] = lights[i];
		}

		// The camera's world position and look direction, used to cull torch
		// lights by distance and to orient the flame billboards. `forward`
		// biases the light cut and shadow pool toward what's in view -- a
		// light behind the player lights nothing the frame renders.
		const glm::mat4 camToWorld = glm::inverse(View);
		const glm::vec3 eyePos = glm::vec3(camToWorld[3]);
		const glm::vec3 forward = -glm::vec3(camToWorld[2]);

		// Re-decides the dynamic cube-shadow pool's slots, at most every
		// SHADOW_REASSIGN_INTERVAL. Before the light-append loop (reads
		// tf.shadowSlot) and the DSshadowCube mapping loop (reads the changed
		// slots' matrices).
		shadowReassignTimer += deltaT;
		if(shadowReassignTimer >= SHADOW_REASSIGN_INTERVAL) {
			shadowReassignTimer = 0.0f;
			updateDynamicShadowSlots(eyePos, forward);
		}

		// Diffs every static cube slot's current occupant against the one it
		// last rendered for and queues changes into pendingFaceMask. Cheap
		// enough to run every frame; only finds work on the frame something
		// changed. HAND_TORCH_SHADOW_INDEX is excluded (rendered fresh every
		// frame, never cached).
		for(int t = 0; t < dynamicShadowSlotBase; t++) {
			// A fixed lights.json slot's identity never changes, so using the
			// slot index as the marker means this fires once, on frame one.
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

		// The diff above only sees a slot whose LIGHT changed. This adds slots
		// whose light stayed put while a ghost or door moved inside -- without
		// it those movers cast a shadow frozen at capture time.
		queueMoverCubeSlotRenders();

		// The cylindrical billboard basis every wall flame uses this frame,
		// from the CAMERA's right axis, not each flame's eye->anchor direction.
		// The textbook per-flame version behaves for a wall torch but breaks
		// for the held one: its anchor is barely a unit away in CAMERA space,
		// so pitching swings it around the eye, its horizontal offset shrinks
		// to zero and flips sign, and the derived yaw whips through 180
		// (visible flame spin). The camera's right axis is exactly horizontal
		// at every pitch and turns only with yaw -- the one rotation a standing
		// flame should follow.
		glm::vec3 bbRight = glm::vec3(camToWorld[0]);
		bbRight.y = 0.0f;
		if(glm::dot(bbRight, bbRight) > 1e-8f) {
			bbRight = glm::normalize(bbRight);
		} else {
			// Unreachable while pitch is clamped, but a zero-length basis
			// vector collapses the billboard to a line -- so, an arbitrary
			// valid axis rather than a NaN.
			bbRight = glm::vec3(1.0f, 0.0f, 0.0f);
		}
		const glm::vec3 bbUp = glm::vec3(0.0f, 1.0f, 0.0f);
		// Points back toward the eye, so the flame's local +z is "toward camera".
		const glm::vec3 bbFwd = glm::cross(bbRight, bbUp);

		// A second basis for the HELD torch only: the camera's ACTUAL
		// up/right/forward, pitch included, straight off camToWorld's columns.
		// The held torch is welded to the camera and tilts with it; a
		// world-vertical flame on a tilting shaft swings out of alignment past
		// a small pitch. Tying it to the same basis the torch mesh rides keeps
		// the two aligned, and reading it off the view matrix it can't
		// degenerate.
		const glm::vec3 handBbRight = glm::normalize(glm::vec3(camToWorld[0]));
		const glm::vec3 handBbUp    = glm::normalize(glm::vec3(camToWorld[1]));
		const glm::vec3 handBbFwd   = glm::normalize(glm::vec3(camToWorld[2]));

		animTime += deltaT;

		// Advance every torch's fire state before anything reads it, so the
		// flame, its sparks and its light are all driven by the same envelope
		// within the same frame (see the TorchFlame struct).
		for(TorchFlame &tf : torchFlames) {
			// Flicker: a fast term (a quarter of the signal -- Flame.frag's
			// per-pixel shimmer owns the fast twitch), a slow one so the flame
			// breathes over seconds, a middle one so the two don't read apart.
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

			// The hunt cycle's colour, recomputed from the authored base every
			// frame. One line: `color` is what both the light and the billboard
			// read, so tinting it here turns the torch, its light, its shadows
			// and its bloom violet together.
			//
			// The warning pulse and the hunt's dimming ride on the same value
			// rather than on `intensity`, which is spring-driven state: a 4 Hz
			// pulse written into a spring with a ~0.2 s response would be
			// damped into nothing, and writing it into the spring's own value
			// would corrupt the flicker it exists to produce.
			tf.color = huntCycle.flameColor(tf.baseColor)
					   * huntCycle.warningPulse() * huntCycle.lightScale();

			// Combustion instability: a slow wander in hue on top of the
			// brightness spring, so a guttering flame shifts warm/cool the way
			// burning fuel does instead of only dimming. Scaled by `gutter` so
			// a steady flame is untouched and the hunt-cycle tint stays clean.
			float hueWander = fireFbm(t * 0.9f + 53.0f) - 0.5f;
			tf.color.r *= 1.0f + hueWander * 0.10f * gutter;
			tf.color.b *= 1.0f - hueWander * 0.12f * gutter;

			// Height: the same signal, compressed into a narrower band and
			// chased much more slowly -- see TorchFlame::heightScale.
			float hTarget = 0.78f + 0.31f * (target - 0.30f) / 1.10f;	// ~0.78..1.09
			tf.heightScale += (hTarget - tf.heightScale)
							  * (1.0f - std::exp(-deltaT / FLAME_HEIGHT_TAU));

			// Catching fire / going out: an underdamped spring toward 1 while
			// lit, 0 while not -- see TorchFlame::ignitionScale and
			// FLAME_IGNITION_OMEGA/ZETA. Advanced unconditionally (not just
			// while flameBurning(tf)) so a flame just switched off relaxes
			// back to 0 instead of freezing wherever it was.
			//
			// The underdamped ZETA is only wanted on the way UP -- the flare
			// past resting size is what reads as "catching". On the way DOWN
			// that same ring makes ignitionScale undershoot to 0 (clamped),
			// bounce back to ~0.1, and sink again: the flame and its point
			// light (L.color and the billboard size both scale by
			// ignitionScale) visibly go out, flicker back, then out -- which
			// is exactly the "off, on, off" seen when a ghost snuffs the held
			// torch, the first thing in the game that flips `burning` false at
			// runtime. Critically damp the extinguish so it eases out once.
			float wi = FLAME_IGNITION_OMEGA;
			bool igniting = flameBurning(tf);
			float ignitionTarget = igniting ? 1.0f : 0.0f;
			float zi = igniting ? FLAME_IGNITION_ZETA : 1.0f;
			tf.ignitionVel += ((ignitionTarget - tf.ignitionScale) * wi * wi
								- 2.0f * zi * wi * tf.ignitionVel) * deltaT;
			tf.ignitionScale += tf.ignitionVel * deltaT;
			// The overshoot is the point (the flare while catching), but it
			// must not go negative -- a negative billboard size would flip
			// the flame's quads inside out for a frame.
			tf.ignitionScale = std::max(tf.ignitionScale, 0.0f);

			// Lean. A flame is dragged by the air it moves through, so it
			// leans AGAINST its own velocity: the world-space lean vector is
			// just the smoothed velocity negated. Only the horizontal part --
			// riding a lift up or down doesn't bend a flame sideways.
			glm::vec3 pos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			// A flame that's currently switched off drops its velocity history
			// instead of differencing against it. The held torch is parked
			// 1000 units below the map while off (see GameLogic), so both the
			// frame it goes away and the frame it comes back would otherwise
			// read as an enormous velocity and bring the flame back folded
			// over at its lean cap.
			if(!flameBurning(tf)) {
				tf.velPrimed = false;
				tf.smoothedVel = glm::vec3(0.0f);
				tf.lean = glm::vec2(0.0f);
				continue;
			}
			if(!tf.velPrimed) {
				// First frame: no previous position to difference against, and
				// the instance may still be sitting at its scene.json
				// placeholder, which would read as an enormous velocity.
				tf.prevPos = pos;
				tf.velPrimed = true;
			}
			glm::vec3 vel = (pos - tf.prevPos) / std::max(deltaT, 1e-4f);
			tf.prevPos = pos;

			// Exponential smoothing, framerate-independent: without it the
			// flame would twitch on every single-frame jolt in the walking bob
			// rather than swinging through it.
			float a = 1.0f - std::exp(-deltaT / TORCH_LEAN_TAU);
			tf.smoothedVel += (vel - tf.smoothedVel) * a;

			// Against its own motion: a flame is bent by the air it is being
			// dragged through, so it trails behind the hand carrying it.
			glm::vec3 leanWorld = -tf.smoothedVel;
			leanWorld.y = 0.0f;

			// Resolve into the billboard's own axes -- the SAME basis this
			// torch's quads are actually built with below (cylindrical for a
			// wall torch, camera-locked for the held one via heldByCamera),
			// not a second per-flame copy. The lean is uploaded in
			// billboard-local units, so resolving it against a different
			// basis than the one the quads use would make it lean in the
			// wrong direction.
			const glm::vec3 &leanRight = tf.heldByCamera ? handBbRight : bbRight;
			const glm::vec3 &leanFwd   = tf.heldByCamera ? handBbFwd   : bbFwd;

			// Straight from speed to half-widths of tip offset.
			//
			// Deliberately does NOT divide by the flame's world half-width:
			// the held torch's half-width is about 0.07 world units, so
			// dividing by it would multiply every velocity by ~14 -- merely
			// turning on the spot swings the torch through roughly 2 units/s,
			// which saturates the lean to its cap and pins it there
			// permanently folded over. It would also make the effect
			// scale-dependent in the wrong direction: the small held torch
			// would react three times harder than a full-size wall one, when
			// they should behave identically.
			tf.lean = glm::vec2(glm::dot(leanWorld, leanRight), glm::dot(leanWorld, leanFwd))
					  * TORCH_LEAN_PER_SPEED;
			if(glm::length(tf.lean) > TORCH_LEAN_MAX) {
				tf.lean = glm::normalize(tf.lean) * TORCH_LEAN_MAX;
			}
		}

		// Torch flames' point lights, appended straight into gubo every frame
		// from the same Wm the flame rides -- NOT through lights.json's
		// "instance"+"offset" anchoring, which reads the Wm once at init time
		// and would freeze the held torch's light at the origin.
		//
		// Culled by REAL distance (never by facing -- a torch you turned away
		// from still lights the room behind you) and capped in count, in
		// facingBiasedDistSq() order. Every gubo light costs a full BRDF per
		// fragment per sample; TORCH_LIGHT_MAX_LIVE is above the scene's flame
		// count, so the cap doesn't bite in practice.
		{
			std::vector<std::pair<float, const TorchFlame *>> nearest;
			nearest.reserve(torchFlames.size());
			const float cullSq = TORCH_LIGHT_CULL_DIST * TORCH_LIGHT_CULL_DIST;
			for(const TorchFlame &tf : torchFlames) {
				// Switched-off flames never even enter the contest: not culled
				// late, just absent, so they can't take a live slot from a
				// torch that IS burning either.
				if(!flameBurning(tf)) {
					continue;
				}
				glm::vec3 worldPos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
				glm::vec3 d = worldPos - eyePos;
				float dSq = glm::dot(d, d);
				if(dSq > cullSq) {
					continue;
				}
				// View-cone cull: a wall torch whose whole lit sphere sits
				// outside what the frame draws lights nothing on screen -- drop
				// it before it takes a live slot or a BRDF eval. The held torch
				// rides the camera, so it always reaches the view.
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
				// The same envelope the flame is drawn with: brightness on
				// colour, reach on g (both needed -- colour alone pulses in
				// place, g alone grows without heating). tf.ignitionScale too,
				// so the light brightens in step with a catching flame.
				L.color = tf.color * tf.intensity * tf.lightScale * tf.ignitionScale;
				L.g = flameLightG(tf.isCandle, tf.intensity);
				L.beta = TORCH_LIGHT_BETA;
				L.cosIn = 1.0f;
				L.cosOut = 0.0f;
				L.type = LIGHT_POINT;
				// The held torch gets HAND_TORCH_SHADOW_INDEX, recomputed for
				// its current position here so this frame's cube pass renders
				// it right. Every other candidate reads whatever slot
				// updateDynamicShadowSlots() gave it, or -1 to skip the lookup.
				// The explicit -1 for a non-candidate matters: 0 is the sun's
				// map, and indoors that would read the light as fully shadowed.
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

		// The light the open exit throws back into the room. Appended into gubo
		// like the torch lights, and for the same reason: its brightness
		// tracks the leaf's swing. The counterpart to the ExitGlow quad -- the
		// quad is the daylight you LOOK at, this is the daylight on the stone;
		// neither reads right alone. No shadow map (shadowIndex -1): the wall
		// it shines through already shapes it, and a moving-door occluder would
		// re-render every frame.
		if(exitOpenFrac > 0.001f && gubo.lightCount < MAX_LIGHTS) {
			LightData L{};
			L.pos = EXIT_SPILL_POS;
			// Aimed back through the doorway, i.e. due west: the same axis the
			// glow quad faces along.
			L.dir = glm::vec3(-1.0f, 0.0f, 0.0f);
			L.color = EXIT_SPILL_COLOR * exitOpenFrac;
			L.g = EXIT_SPILL_G;
			L.beta = EXIT_SPILL_BETA;
			L.cosIn = std::cos(glm::radians(EXIT_SPILL_INNER_DEG * 0.5f));
			L.cosOut = std::cos(glm::radians(EXIT_SPILL_OUTER_DEG * 0.5f));
			L.type = LIGHT_SPOT;
			L.shadowIndex = -1;
			gubo.lights[gubo.lightCount++] = L;
		}

		// By value: with the Ambient Light cheat off there is no stored ambient
		// to hand back a reference to. See SceneLights::ambient().
		const AmbientLight amb = sceneLights.ambient();
		gubo.ambientUpper = amb.upper;
		gubo.ambientLower = amb.lower;
		gubo.ambientDir = amb.dir;
		// Uploaded unswitched: the cheat is applied in the shader's
		// ambientShare(), after the per-model override, not by zeroing this
		// (a model with its own materials.json weight never reads this field).
		// The cheat must zero the SHARE, not just black the colors -- under
		// E17's blend the direct half is scaled by (1 - weight), so blacking
		// the colors would DARKEN the scene instead of removing indirect light.
		gubo.ambientWeight = amb.weight;
		// Same bucket, so the same gate covers it. See CookTorrance.frag's blend.
		gubo.ambientBounce = amb.bounce;

		// Distance fog density, derived from GEOM_CULL_CONE_DIST so the two
		// can't drift. Solves exp(-(density*dist)^2) = FOG_RESIDUAL_AT_CULL_DIST
		// for dist == GEOM_CULL_CONE_DIST * FOG_REFERENCE_DIST_SCALE, so fog
		// reaches 1% brightness somewhat PAST where the cull stops drawing,
		// not exactly at it. FOG_REFERENCE_DIST_SCALE is the knob: 1.0 reads
		// dark well before the cull edge (an exponential drops fast early);
		// 2.5 keeps the foreground near unfogged but lets a little pop peek
		// through the cull edge (softened by the vignette and shared black).
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
		if(!sceneLights.ambientEnabled) gubo.debugFlags |= LIGHT_DEBUG_NO_AMBIENT;

		// Both computed further up, before the torch fire state that needs them.
		gubo.eyePos = eyePos;
		gubo.time = animTime;

		DSglobal.map(currentImage, &gubo, 0);

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

		// The four post-processing passes. Each one's texelSize is that of the
		// texture it READS, not the one it writes, since it is used to step
		// from one source texel to the next.
		{
			// RP.width/height, not swapChainExtent: the scene's resolve
			// target (what the bright pass below actually reads) is sized to
			// renderScale's scaled-down resolution, which can now differ
			// from the swapchain/window's.
			const glm::vec2 fullTexel = glm::vec2(1.0f / (float)RP.width,
												  1.0f / (float)RP.height);
			const glm::vec2 bloomTexel = glm::vec2(1.0f / (float)bloomWidth(),
												   1.0f / (float)bloomHeight());

			PostUniformBufferObject post{};
			post.time = animTime;
			post.threshold = BLOOM_THRESHOLD;
			post.knee = BLOOM_KNEE;
			// Stare-at glare rides the two post knobs: more bloom AND more
			// exposure when the player looks into a flame. Smoothed
			// asymmetrically upstream, so this never pumps.
			post.bloomIntensity = BLOOM_INTENSITY * (1.0f + GLARE_BLOOM_GAIN * glareSmoothed);
			post.exposure = SCENE_EXPOSURE * (1.0f + GLARE_EXPOSURE_GAIN * glareSmoothed);
			// The escape whiteout, multiplied on top of the same two knobs so
			// winning while staring into a torch gives one flash. Cubed: a
			// linear exposure ramp reads as an even fade to white; weighting
			// it to the end leaves the room there a moment before it's taken.
			if(escapeFlash > 0.0f) {
				const float f = escapeFlash * escapeFlash * escapeFlash;
				post.exposure *= 1.0f + ESCAPE_EXPOSURE_GAIN * f;
				post.bloomIntensity *= 1.0f + ESCAPE_BLOOM_GAIN * f;
			}
			// The whiteout's own term, applied by Composite.frag after the
			// tone map (see PostUniformBufferObject). Held back until the
			// exposure ramp has had most of the flash, so the frame is already
			// blowing out when the white arrives.
			post.escapeFlash = glm::smoothstep(0.45f, 1.0f, escapeFlash);

			// The spectral veil's ramp (SPECTRAL_VEIL_OUTER). Here, not in the
			// ghost loop, which runs on the game clock -- pausing inside a
			// ghost would freeze the wash. max() over the ghosts. Reads the
			// drawn matrix so the bob counts.
			{
				float veil = 0.0f;
				for(const Ghost &g : ghosts) {
					if(g.inst == nullptr) continue;
					const glm::vec3 gp = glm::vec3(g.inst->Wm[3]);

					// Vertical first: one subtraction rejects most ghosts, and
					// the horizontal test costs a square root.
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

			// Composite: samples with plain UVs, so the texel size is unread.
			// The tone map happens here, at the end of the chain, which is why
			// debugFlags has to reach this pass.
			post.texelSize = fullTexel;
			post.blurDir = glm::vec2(0.0f);
			DScomposite.map(currentImage, &post, 0);
		}

		// Each flame's render matrix is one of the two billboard bases,
		// translated and scaled to its anchor. Wall torches get the
		// CYLINDRICAL basis (faces the camera, stays upright at any pitch);
		// the held torch gets the CAMERA-LOCKED one (tilts with the camera,
		// like the shaft it burns on). What SHOULD respond to movement goes in
		// deliberately as tf.lean.
		//
		// Shader local space: x +/-1 across the half-width, y 0 wick to 1 tip,
		// z toward the camera (depth-layer separation only).
		for(const TorchFlame &tf : torchFlames) {
			glm::vec3 anchorWorld = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			// Uniform scale, so any basis column's length IS the scale factor.
			float instScale = glm::length(glm::vec3(tf.inst->Wm[0]));

			// An off (or unlit) flame collapses to a point rather than being
			// skipped: its descriptor set is recorded once and replayed, so
			// there's no "don't draw this one". tf.ignitionScale already
			// carries the on/off state (springs to 0/1), so no separate
			// flameBurning() gate -- that would reintroduce the snap this
			// animation replaces.
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

		// The daylight outside the exit door. NOT billboards: fixed planes, so
		// their bases are world axes. Same column convention as Flame's
		// billboard (in-plane axes scaled to the half-extents, normal, centre),
		// so one mesh and pipeline serve both an upright quad and one flat --
		// the difference is which world axes go in the first two columns.
		{
			// Squared, so the light builds late in the swing: a door barely
			// ajar shows a crack, not half the glare.
			const float glow = EXIT_GLOW_INTENSITY * exitOpenFrac * exitOpenFrac;

			// The wall of light, standing across the doorway past the leaf's
			// reach: in-plane axes are world Z (across) and world Y (up).
			const glm::mat4 uprightBasis = glm::mat4(
				glm::vec4(glm::vec3(0.0f, 0.0f, 1.0f) * EXIT_GLOW_HALF_WIDTH, 0.0f),
				glm::vec4(glm::vec3(0.0f, 1.0f, 0.0f) * EXIT_GLOW_HALF_HEIGHT, 0.0f),
				glm::vec4(EXIT_GLOW_NORMAL, 0.0f),
				glm::vec4(EXIT_GLOW_CENTER, 1.0f)
			);
			exitGlow.update(EXIT_GLOW_UPRIGHT, ViewPrj * uprightBasis,
							EXIT_GLOW_COLOR, glow, animTime, currentImage);

			// The ground it stands on, covering the strip of open earth the
			// upright quad leaves visible under the arch: in-plane axes are
			// world X (out from the threshold) and world Z (across), normal
			// straight up.
			const glm::mat4 floorBasis = glm::mat4(
				glm::vec4(glm::vec3(1.0f, 0.0f, 0.0f) * EXIT_GLOW_FLOOR_HALF_X, 0.0f),
				glm::vec4(glm::vec3(0.0f, 0.0f, 1.0f) * EXIT_GLOW_FLOOR_HALF_Z, 0.0f),
				glm::vec4(EXIT_GLOW_FLOOR_NORMAL, 0.0f),
				glm::vec4(EXIT_GLOW_FLOOR_CENTER, 1.0f)
			);
			exitGlow.update(EXIT_GLOW_FLOOR, ViewPrj * floorBasis,
							EXIT_GLOW_COLOR, glow, animTime, currentImage);

			// The lid: the same plane as the floor quad, lifted over the top of
			// the arch and turned to face down, so looking up from the
			// threshold finds daylight rather than the skybox. Axes are the
			// floor's, since the plane is the same one -- only the normal and
			// the height differ.
			const glm::mat4 ceilingBasis = glm::mat4(
				glm::vec4(glm::vec3(1.0f, 0.0f, 0.0f) * EXIT_GLOW_CEILING_HALF_X, 0.0f),
				glm::vec4(glm::vec3(0.0f, 0.0f, 1.0f) * EXIT_GLOW_CEILING_HALF_Z, 0.0f),
				glm::vec4(EXIT_GLOW_CEILING_NORMAL, 0.0f),
				glm::vec4(EXIT_GLOW_CEILING_CENTER, 1.0f)
			);
			exitGlow.update(EXIT_GLOW_CEILING, ViewPrj * ceilingBasis,
							EXIT_GLOW_COLOR, glow, animTime, currentImage);
		}

		// defines the local parameters for the uniforms
		UniformBufferObject ubo{};

		// DSshadowCube[t] feeds the shadow CAPTURE pass
		// (PShadowCube/ShadowCube.vert/frag) its matrices/position through a
		// mapped uniform buffer instead of a push constant, precisely so the
		// held torch's slot -- refreshed a few lines above in this same
		// function, by updateHandTorchShadow() -- actually takes effect every
		// frame instead of freezing at whatever the "main" command buffer's
		// one-time recording saw. The six static torches don't strictly need
		// the re-map (their matrices never change after computeShadowMatrices()
		// runs once), but mapping all of them uniformly is simpler than
		// special-casing the held one, and costs nothing worth avoiding.
		for(int t = 0; t < activeCubeShadows; t++) {
			ShadowCubeUniformBufferObject cubeUbo{};
			for(int face = 0; face < 6; face++) {
				cubeUbo.lightViewProj[face] = torchFaceMatrices[t][face];
			}
			cubeUbo.lightPos = glm::vec4(torchLightPos[t], 0.0f);
			DSshadowCube[t].map(currentImage, &cubeUbo, 0);
		}

		// Debug overlay (DebugLines.hpp, cheat-gated): light-position gizmos
		// and/or wireframe boxes at each torch's shadow-cube clip planes.
		// Here, so gubo.lights[] and torchLightPos[] are current.
		std::vector<glm::vec4> dbgPos, dbgColor;
		if(cheats.showLightGizmos) {
			for(int i = 0; i < gubo.lightCount; i++) {
				const LightData &L = gubo.lights[i];
				glm::vec4 color = glm::vec4(L.color, 1.0f);
				if(L.type == LIGHT_DIRECT) {
					// The sun has no position, so its gizmo is an arrow anchored
					// near the player, not a cross at a point.
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
		// Debug-camera companion overlay: the geometry cull's shape drawn in
		// world space, so from the pulled-back spectator view you can see
		// exactly which side of it an instance is on when it winks out. Same
		// eyePos/forward the cull itself uses (GEOM_CULL_*).
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

			// View cone: half-angle straight from the cull's cosine, edge
			// lines out to the cone distance plus a cap ring.
			float half = std::acos(GEOM_CULL_CONE_COS);
			float capR = GEOM_CULL_CONE_DIST * std::tan(half);
			glm::vec3 capC = eyePos + f * GEOM_CULL_CONE_DIST;
			for(int i = 0; i < 16; i++) {
				float a = (float)i / 16 * 2.0f * 3.14159265f;
				glm::vec3 edge = capC + (rr * std::cos(a) + uu * std::sin(a)) * capR;
				DebugLines::PushLine(eyePos, edge, coneCol, dbgPos, dbgColor);
			}
			ring(capC, capR, rr, uu, coneCol);

			// Torch-light / shadow cull companion: a cross at every burning wall
			// flame, green when its light survives lightReachesViewCone() and
			// red when it's dropped (light out of gubo, cube slot freed). Same
			// reach the cull feeds in, so the colour flips exactly on the
			// boundary you see the surrounding geometry wink out at.
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

		// The gazed door's chains/padlock (Door::LockProp) glow along with the
		// leaf -- they read as part of the door. Built once here. Empty with
		// the Focus Glow cheat off (the gaze itself is untouched; only the
		// aura goes).
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

		// Over every technique, not just the first: instances need the same
		// per-object uniforms filled in regardless of which technique they
		// belong to.
		for(int techniqueId = 0; techniqueId < SC.TechniqueInstanceCount; techniqueId++) {
			for(int instanceId = 0; instanceId < SC.TI[techniqueId].InstanceCount; instanceId++) {
				glm::mat4 renderWm = SC.TI[techniqueId].I[instanceId].Wm;
				// Geometry visibility cull (GEOM_CULL_*): an instance outside
				// the radius+cone gets a stand-in render matrix and draws off
				// the map -- an invertible translate, not a zero matrix (nMat
				// below is its inverse-transpose). Only this render copy
				// changes; the collider is left alone. Tested as a bounding
				// SPHERE from the collider's extents (this pack isn't
				// centre-pivoted, so a bare point clipped things still in
				// view), else GEOM_CULL_FALLBACK_RADIUS.
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

				// By Mid rather than by name, so no string hashing per frame.
				const Material &m = materials.forModel(SC.TI[techniqueId].I[instanceId].Mid);
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
				// The gazed instance (plus its lock hardware) glows; other
				// instances of the same model don't -- which is why this flag
				// is per-instance and not in Material.
				bool glow = false;
				for(Instance *g : glowingInstances) {
					if(g == &inst) {
						glow = true;
						break;
					}
				}
				// Sign = "would [E] do anything" (gazedInteractionDisabled),
				// magnitude = which aura colour (GlowKind), packed into one
				// scalar since the UBO's spare room is spent.
				float kindMag = static_cast<float>(gazedGlowKind);
				ubo.glow = glow ? (gazedInteractionDisabled ? -kindMag : kindMag) : 0.0f;

				// The ghosts' chase telegraph, riding F0 (Spectral computes no
				// BRDF). After the material copy, which just wrote F0. A linear
				// scan per instance: a flag on Instance is off-limits, and a
				// map lookup costs more for three ghosts.
				for(const Ghost &g : ghosts) {
					if(g.inst == &inst) {
						ubo.F0 = g.chaseBlend;
						break;
					}
				}
				// DS[1] = Pchar pass (main render): set0=DSLglobal, set1=DSLlocal
				inst.DS[0][0]->map(currentImage, &gubo, 0); // global (light/camera)
				inst.DS[0][1]->map(currentImage, &ubo, 0); // camera MVPs
				// set2=DSLshadowSample, on techniques whose pipeline layout
				// declares a third set, is never mapped here: it holds only
				// fixed samplerCube bindings, set once at descriptor-set
				// creation (shadowMapDefs in localInit()), with no per-frame
				// host-visible buffer behind it.
			}
		}

		// Records and submits this frame's cube shadow captures, now that both
		// DSshadowCube[t] and every instance's inst.DS[0][1] (mapped in the
		// loop above) are current -- recording earlier would bake a stale Wm
		// into the map. See submitCubeShadowCaptures().
		submitCubeShadowCaptures(currentImage);
		pendingFaceMask.fill(0);

		// The FPS counter, left visible even over the launch screen. Everything
		// below it reads game state GameLogic() doesn't update while an overlay
		// is open, and TextMaker draws above the backdrop, so those blocks are
		// skipped rather than shown through it.
		//
		// Timed off glfwGetTime(), not deltaT: deltaT is forced to 0 while
		// paused, but frames are still rendered, so this readout shouldn't freeze.
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
			// Coordinates overlay (Show Coordinates cheat), above the FPS line.
			// Throttled to 10Hz: print() always dirties the text buffer.
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

			// The "[E] ..." prompt, shown while a door/pickup/candle/torch is in
			// range. Re-prints whenever the text changes, not just on a
			// show/hide toggle. The locked-exit line rides the same slot.
			// Suppressed once a run ends -- GameLogic() stops updating the
			// nearby* fields when it freezes.
			static bool interactPromptShown = false;
			static std::string interactPromptText;
			bool showInteractPrompt = runState == RunState::Running &&
									  (nearbyDoor >= 0 || nearbyPickup >= 0 ||
									   nearbyCandle >= 0 || nearbyWallTorch >= 0 ||
									   nearbyHandTorch || atLockedExit);
			// Door prompts split four ways: a padlock the player can open, one
			// they can't (name what to find, by lockLabel), one they're behind
			// (name no key -- there's no padlock in sight from that side), and
			// a plain door.
			std::string wantedPromptText;
			if(atLockedExit) {
				wantedPromptText = "The way out is locked - find the key";
			} else if(nearbyHandTorch) {
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

			// The hunt banner, high and centred: the words behind what the
			// torches say in colour. Re-printed only when the text changes;
			// the countdown is rounded to whole seconds so it changes once a
			// second, not once a frame.
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

			// End of run. Static text, so unlike the banner above it's printed once
			// on the transition and left alone until the run restarts.
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
	
	// --- Ghost navigation ---------------------------------------------------
	//
	// The ghosts obey the same walls the player does: a threat that ignores
	// geometry can't be played around. Not pathfinding -- a wall test, a
	// push-out identical to the player's, and a fan of candidate headings,
	// which is enough for rooms and corridors given the return trail.

	// True if a ghost-sized cylinder at `p` overlaps a wall. No
	// MAX_STEP_HEIGHT exemption: a ghost hovers over a low crate because the
	// crate's yMax falls below the slab tested here. `radius` is a parameter
	// so ghostPathClear/ghostSteer can ask with extra padding while
	// ghostResolveWalls asks with the real body size (see ghostSteer).
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

	// Pushes `p` horizontally out of anything it's inside. Same shape as the
	// player's wall block -- move first, then push out along the shortest
	// escape, so the wall-parallel component survives (sliding for free).
	void ghostResolveWalls(glm::vec3 &p) const {
		// The 3x3 neighbourhood is computed once, from `p` as it is on entry,
		// before any push below can move it -- exactly like the old
		// all-colliders scan, which also decided once which colliders exist
		// and then mutated p while walking that fixed list. A push here is at
		// most ghostRadius, well inside the CELL_SIZE margin, so p never
		// actually leaves the neighbourhood that was queried for it.
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
	// Sampled, not swept: point tests spaced under the ghost's radius, so
	// nothing thinner can slip between two.
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

	// True if `p` sits inside any collider's box, testing the real Y of `p`
	// (not a slab off another height) -- what a sightline needs, so a table
	// only blocks the ray if the line is actually low enough to clip it.
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

	// Roughly where a ghost's "eyes" are, relative to its hover pivot --
	// nearer the top of the body than the centre. Only used for the sightline
	// below; the movement/collision code has no use for it.
	static constexpr float GHOST_EYE_OFFSET = 0.5f;

	// Whether a ghost at `from` can see the player at `eyeTarget`. A 3D ray,
	// not ghostPathClear's flat XZ probe: a hovering ghost looking down clears
	// a waist-high table, while walls and closed doors (floor-to-ceiling
	// boxes) still block.
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

	// Picks the heading a chasing ghost takes, given where it WANTS to go
	// (straight at the player). Fans out from `desired` in widening steps and
	// takes the first with a clear probe ahead -- along a wall that's the
	// direction that slides down it, at a corner the one that turns it.
	// Backwards is valid too (giving up on a dead end).
	//
	// g.turnBias stops it dithering: without it a ghost facing a pillar swaps
	// left/right every frame as the geometry shifts by centimetres. The bias
	// remembers the chosen side and re-tries it first, re-examined only once
	// the ghost has a clear straight line again.
	//
	// Zero vector = boxed in, which the caller reads as "don't move".
	glm::vec2 ghostSteer(Ghost &g, const glm::vec2 &desired) const {
		// Probed with a little more than the ghost's radius: a choke point a
		// hair wider than the body flips "clear?" every frame, which reads as a
		// ghost snapping between headings. The padding makes it a threshold the
		// ghost commits to early, at the cost of refusing a gap slightly sooner.
		const float steerRadius = ghostRadius * ghostSteerMargin;

		// Straight there. Also the point at which the ghost stops having an
		// opinion about which way it went round the last obstacle.
		if(ghostPathClear(g.pos, desired, GHOST_PROBE_DIST, steerRadius)) {
			return desired;
		}

		// Deviations in degrees, shallowest first. Stops just short of 180: a
		// dead straight retreat is what the last pair already covers, and it
		// would be reached only when literally every other heading is blocked.
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
			// Padlocks come back with the run -- the spent key is back on its
			// table below, or the level would get easier every restart.
			d.locked = !d.lockKeyId.empty();
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
		// A key mid-fall would keep being drawn off the camera, then parked
		// below the map a few frames in, right after the loop put it back.
		keyLowerIdx = -1;

		// The torch goes back on the floor and every lit candle goes back out
		// -- same reason the doors re-lock: the player earns the light. Its Wm
		// is restored next frame once handTorchCollected is false.
		handTorchCollected = false;
		for(TorchFlame &tf : torchFlames) {
			tf.burning = tf.spawnBurning;
			// Snap to rest, not mid-spring: a re-lit flame plays its
			// catching-fire animation from a clean 0.
			tf.ignitionScale = tf.spawnBurning ? 1.0f : 0.0f;
			tf.ignitionVel = 0.0f;
		}

		nearbyDoor = -1;
		nearbyPickup = -1;
		nearbyCandle = -1;
		nearbyWallTorch = -1;
		nearbyHandTorch = false;
		gazedInstance = nullptr;
		atLockedExit = false;
		// The way out closes with the rest of the doors, so its daylight and
		// the whiteout go too.
		exitOpenFrac = 0.0f;
		escapeFlash = 0.0f;

		runState = RunState::Running;
	}

	// Called once on the frame the hunt phase changes. Everything the change
	// DOES is polled from huntCycle where needed, so this is just the
	// announcement -- its own function because it's the one place a music
	// track would be swapped. No audio backend yet, so it prints.
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

	// --- F11 fullscreen toggle ---------------------------------------------
	// Starter.hpp forces GLFW_RESIZABLE=FALSE, but glfwSetWindowMonitor() still
	// works: switching monitors fires GLFW's framebuffer-resize callback, which
	// makes BaseProject recreate the swapchain and call onWindowResize() for us.
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

	float GameLogic() {
		// Camera FOV-y, Near Plane and Far Plane
		const float FOVy = glm::radians(45.0f);
		const float nearPlane = 0.1f;
		const float farPlane = 100.f;

		// Camera movement controls
		// FOV degrees rotated per second
		const float ROT_SPEED = 90.0f;

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
			// Reached Settings from the launch screen (before/after a run):
			// close this, open settingsMenu, remember to come back HERE
			// (not to PauseMenu) when its Back is pressed.
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
			// Same as StartScreen's above, but remembering to come back to
			// PauseMenu (mid-run) instead.
			pauseMenu.setOpen(false, windowWidth, windowHeight);
			settingsFromPause = true;
			settingsMenu.setOpen(true, windowWidth, windowHeight);
		}
		if(pauseMenu.quitClicked()) {
			// Abandon the run and drop back to the launch screen. restartRun()
			// resets every piece of world state, so the next Play starts clean.
			pauseMenu.setOpen(false, windowWidth, windowHeight);
			restartRun();
			startScreen.setOpen(true, windowWidth, windowHeight);
		}

		settingsMenu.update(window, windowWidth, windowHeight);
		if(settingsMenu.backClicked()) {
			// Reopen whichever of PauseMenu/StartScreen sent the player here
			// -- settingsFromPause, set at the two settingsClicked() sites
			// above, is the only thing that remembers which.
			settingsMenu.setOpen(false, windowWidth, windowHeight);
			if(settingsFromPause) {
				pauseMenu.setOpen(true, windowWidth, windowHeight);
			} else {
				startScreen.setOpen(true, windowWidth, windowHeight);
			}
		}

		getSixAxis(deltaT, m, r, fire);

		// Clamped after getSixAxis (input timing untouched), before physics.
		// Gravity is plain Euler integration; on a stalled frame a deltaT spike
		// overshoots every collider and the player falls through the floor.
		// 1/20s caps one frame's fall -- the game briefly slows instead of
		// skipping physics, the standard fix for Euler on a variable timestep.
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

		// Restart. Only offered once a run has ended, so R is free to mean
		// something else during play, and edge-triggered like every other key
		// here so holding it doesn't restart every frame.
		bool restartKey = glfwGetKey(window, GLFW_KEY_R);
		if(runState != RunState::Running && restartKey && !restartKeyWasPressed && !overlayOpen()) {
			restartRun();
		}
		restartKeyWasPressed = restartKey;

		// The whiteout, ramped here not in the frozen-movement block: it has to
		// keep running after the run ends (RunState::Escaped starts it). Held
		// still while an overlay is open, so pausing mid-flash doesn't skip it.
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

		// Freeze all movement/physics while an overlay is open or a run has
		// ended, so the last frame the player saw is the one they keep looking
		// at.
		if(!overlayOpen() && runState == RunState::Running) {
			// Sprint: Ctrl multiplies move speed. Polled directly (Starter.hpp
			// doesn't wire Ctrl). Started only while grounded, stopped
			// immediately on release, air or not.
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

			// WASD: m.x strafe, m.z -forward. The Starter's m.y (R/F fly) is
			// dropped -- vertical movement only comes from jumping and gravity.
			// Flattened to yaw only, not the pitched `front`, so looking up and
			// pressing W doesn't push you up or down through the floor.
			glm::vec3 frontFlat = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
			camPos += (right * m.x - frontFlat * m.z) * moveSpeed * deltaT;

			// Wall collision: push the camera out of any collider that counts
			// as a wall -- too tall to step onto AND low enough for the body to
			// reach it. Everything else is left to the ground pass, which lifts
			// the player on top; that split makes steps and crates walkable.
			if(cheats.collisionEnabled) {
				const float EYE_HEIGHT = 1.8f;
				const float PLAYER_HEIGHT = 1.8f;
				const float PLAYER_RADIUS = 0.3f;
				// So grazing the gate lintel's underside doesn't count as being inside it.
				const float VERTICAL_MARGIN = 0.1f;
				float feetY = camPos.y - EYE_HEIGHT;
				float headY = feetY + PLAYER_HEIGHT;
				for(Collider *C : allColliders) {
					AABBextents E = C->getExtents();

					// Low enough to step onto: not a wall (the ground pass lifts
					// the player onto it). Also covers floors and anything below.
					if(E.yMax <= feetY + MAX_STEP_HEIGHT) continue;

					// A wall, but only for the part the body reaches: the gate
					// lintel's underside is above head height, so walk under it.
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

			// Jump: spacebar (Starter's "fire"). Edge-triggered, only while
			// grounded.
			if(fire && !jumpKeyWasPressed && grounded) {
				camVerticalVelocity = movement.jumpSpeed;
				// Drop leftover step smoothing, or a jump right after a step-up
				// starts from a trailing view.
				eyeStepOffset = 0.0f;
			}
			jumpKeyWasPressed = fire;

			// Interaction: findGazedDoor picks the target, DOOR_INTERACT_RADIUS
			// gates whether it's reachable, so a far door down a hall doesn't
			// light up early.
			nearbyDoor = -1;
			{
				int gazed = findGazedDoor(front);
				if(gazed >= 0 && doorDistance(doors[gazed], camPos) < DOOR_INTERACT_RADIUS) {
					nearbyDoor = gazed;
				}
			}

			// Pickups: same gaze-then-proximity check. A pickup wins over a
			// door via the E-key handling's priority order.
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

			// Candles: an unlit one aimed at within reach is lit by [E] off the
			// held torch. Whether the player HAS fire isn't checked here -- the
			// candle still targets, so the disabled glow and prompt can explain
			// why nothing happens.
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

			// Wall torches: a burning one within reach, aimed at, lights the
			// unlit torch in the player's hand ([E]). The mirror of the candle
			// block above. findGazedWallTorch already returns -1 once the held
			// torch is burning, so this quietly stops offering a target then.
			nearbyWallTorch = -1;
			{
				int gazed = findGazedWallTorch(front);
				if(gazed >= 0) {
					const TorchFlame &tf = torchFlames[gazed];
					float dx = camPos.x - tf.anchorWorld.x;
					float dy = camPos.y - tf.anchorWorld.y;
					float dz = camPos.z - tf.anchorWorld.z;
					float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
					if(dist < WALL_TORCH_INTERACT_RADIUS) {
						nearbyWallTorch = gazed;
					}
				}
			}

			// The floor torch: aimed at, within reach, not yet picked up.
			// Ordinary pickup tolerances -- it's a small floor object.
			nearbyHandTorch = false;
			if(findGazedHandTorch(front)) {
				float dx = camPos.x - handTorchWorldPos.x;
				float dy = camPos.y - handTorchWorldPos.y;
				float dz = camPos.z - handTorchWorldPos.z;
				if(std::sqrt(dx * dx + dy * dy + dz * dz) < PICKUP_INTERACT_RADIUS) {
					nearbyHandTorch = true;
				}
			}

			// The single targeted instance for the focus glow (ubo.glow), in
			// E-key priority order: floor torch and pickup, then wall torch,
			// candle, door. The smaller, nearer thing wins -- a candle usually
			// stands in front of a wall.
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
				// Nothing to light it with: red aura, and the prompt (see
				// updateUniformBuffer) says what's missing.
				gazedInteractionDisabled = !hasBurningTorch();
			} else if(nearbyDoor >= 0) {
				const Door &d = doors[nearbyDoor];
				gazedInstance = d.inst;
				gazedGlowKind = GlowKind::Door;
				// Same condition the locked-door prompt text above already
				// checks: wrong side of the padlock, or no matching key on
				// the ring -- either way [E] would do nothing right now.
				gazedInteractionDisabled =
					d.locked && (!d.onLockSide(camPos) || findKeyInRing(d.lockKeyId) < 0);
			}

			bool interactKey = glfwGetKey(window, GLFW_KEY_E);
			if(interactKey && !interactKeyWasPressed) {
				if(nearbyHandTorch) {
					// Into the hand, still unlit. From next frame
					// updateUniformBuffer() rebuilds its Wm off the camera.
					handTorchCollected = true;
					torchRaiseElapsed = 0.0f;	// restart the raise
					std::cout << "[torch] picked up hand torch\n";
					nearbyHandTorch = false;
					gazedInstance = nullptr;

					// The floor pose is about to vanish, and any shadow it was
					// baked into. queueMoverCubeSlotRenders() only tracks a
					// mover within its light's reach, and a pickup at a torch's
					// dim edge can be inside its cube face but outside the
					// cutoff. So this one-time reach-free sweep marks every
					// occupied-slot face the floor pose falls in.
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
					// No per-instance visibility flag, so "removed from the
					// world" means parked below the map. A key is redrawn in
					// the hand from here on; a plain pickup stays parked.
					p.inst->Wm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
					if(!p.keyId.empty()) {
						// One free hand: a key already in it goes back on the
						// floor (an unheld key on the ring would hang in
						// mid-air). Dropped short (0.5, not G's 1.0) so it
						// lands underfoot.
						if(!keyRing.empty()) {
							dropKeyFromRing((int)keyRing.size() - 1, camPos, front, 0.5f);
						}
						keyRing.push_back(nearbyPickup);
						keyRaiseElapsed = 0.0f;	// restart the raise
					}
				} else if(nearbyWallTorch >= 0) {
					// Lighting the held torch is one bool: flameBurning() turns
					// its billboard, sparks and light on this frame. From here
					// hasBurningTorch() is true, so candles can be lit.
					torchFlames[handFlameIdx].burning = true;
					std::cout << "[torch] lit hand torch from '"
							  << *torchFlames[nearbyWallTorch].inst->id << "'\n";
					nearbyWallTorch = -1;
					gazedInstance = nullptr;
				} else if(nearbyCandle >= 0) {
					// Lighting a candle is one bool: every consequence asks
					// through flameBurning() and turns itself on this frame.
					// Only with fire in hand; failure isn't reported (the aura
					// and prompt already say so).
					if(hasBurningTorch()) {
						torchFlames[nearbyCandle].burning = true;
						std::cout << "[candle] lit '"
								  << *torchFlames[nearbyCandle].inst->id << "'\n";
						// Lit now, so it stops being a target this frame.
						nearbyCandle = -1;
						gazedInstance = nullptr;
					}
				} else if(nearbyDoor >= 0) {
					Door &d = doors[nearbyDoor];
					if(d.locked) {
						// Padlocked: E spends a matching key (destroyed by the
						// unlock) and swings the door open in the same press.
						// No match, or the wrong face of the door: nothing
						// happens (the prompt already says what's missing).
						int slot = d.onLockSide(camPos) ? findKeyInRing(d.lockKeyId) : -1;
						if(slot >= 0) {
							std::cout << "[door] unlocked '" << d.instanceId
									  << "' with key '" << d.lockKeyId << "'\n";
							Instance *spent = pickups[keyRing[slot]].inst;
							consumeKey(slot);
							// If this item is a whenUnlocked prop of this door,
							// cancel the sink consumeKey() just queued -- the
							// prop loop takes it onto the shelf this same frame,
							// so "hand to gap" is one motion.
							for(const Door::LockProp &prop : d.lockProps) {
								if(prop.whenUnlocked && prop.inst == spent) {
									keyLowerIdx = -1;
								}
							}
							d.locked = false;
							// Pick the swing before opening, same as the unlocked
							// case below -- a locked door is by definition still
							// fully closed here.
							d.swingSign = d.swingSignAwayFrom(camPos);
							d.open = true;
						}
					} else {
						// Only a fully-closed leaf is free to pick a side.
						// Reversing one still swinging shut would drag the panel
						// back through the doorway and the player.
						if(!d.open && d.angle == 0.0f) {
							d.swingSign = d.swingSignAwayFrom(camPos);
						}
						d.open = !d.open;
					}
				}
			}
			interactKeyWasPressed = interactKey;

			// Drop key (G): puts the key in hand (the newest) down at arm's
			// length. Pressing G repeatedly drops the ring in reverse order.
			bool dropKey = glfwGetKey(window, GLFW_KEY_G);
			if(heldKeyIdx() >= 0 && dropKey && !dropKeyWasPressed) {
				dropKeyFromRing((int)keyRing.size() - 1, camPos, front, 1.0f);
			}
			dropKeyWasPressed = dropKey;

			for(Door &d : doors) {
				// Magnitude from the scene, direction from whoever opened it
				// (swingSign, set on the press): the leaf always travels away
				// from the player rather than into them.
				float target = d.open ? std::abs(d.openAngleDeg) * d.swingSign : 0.0f;
				float maxStep = DOOR_OPEN_SPEED * deltaT;
				if(d.angle < target) d.angle = std::min(d.angle + maxStep, target);
				else if(d.angle > target) d.angle = std::max(d.angle - maxStep, target);

				// The leaf's local origin IS its hinge, so opening it is one
				// more rotation on the closed-door transform.
				d.inst->Wm = d.baseWm * glm::rotate(glm::mat4(1.0f), glm::radians(d.angle), glm::vec3(0.0f, 1.0f, 0.0f));
				if(d.inst->C != nullptr) {
					d.inst->C->setWorldMatrix(d.inst->Wm);
				}

				// Chains and padlock: the leaf's matrix while locked (they're
				// in its local frame), parked below the map once not. Driven
				// every frame, so restartRun() re-locking brings them back for
				// free. A whenUnlocked prop is the mirror: it appears when the
				// lock comes off, and while locked it's left ALONE (it's still
				// a Pickup on its table).
				for(const Door::LockProp &prop : d.lockProps) {
					if(prop.whenUnlocked) {
						if(!d.locked) prop.inst->Wm = d.inst->Wm * prop.local;
					} else {
						prop.inst->Wm = d.locked ? d.inst->Wm * prop.local
												 : glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
					}
					// The hardware's collider, off the same matrix -- one left
					// behind would seal the doorway. Null for a whenUnlocked
					// prop (Pickup instances carry no collider).
					if(prop.inst->C != nullptr) {
						prop.inst->C->setWorldMatrix(prop.inst->Wm);
					}
				}
			}

			// How far the way out has swung, off the same animated angle the
			// leaf is drawn with, so the light outside can't disagree with the
			// door on screen. Read by updateUniformBuffer() for the daylight
			// quad and spill light. Smoothstepped so the light builds through
			// the swing and settles. Frozen at its last value once the run
			// ends -- an open door should keep lighting the escape banner.
			if(exitDoorIndex >= 0) {
				const Door &exitDoor = doors[exitDoorIndex];
				float span = std::abs(exitDoor.openAngleDeg);
				exitOpenFrac = span > 1e-4f
					? glm::smoothstep(0.0f, 1.0f, glm::clamp(std::abs(exitDoor.angle) / span, 0.0f, 1.0f))
					: 0.0f;
			}

			// Ghosts. See the Ghost struct for the three modes. Common to all:
			// the mode gives a direction and speed, and the bob, facing and
			// world matrix are shared. The bob is added on top of `pos` at the
			// end, so it never feeds back into steering or collision.
			bool ghostsHunting = huntCycle.hunting();
			for(Ghost &g : ghosts) {
				if(g.inst == nullptr || g.waypoints.size() < 2) continue;

				// Line of sight, every frame a hunt is on regardless of mode:
				// a Patrol or Return ghost that spots the player needs to start
				// a chase.
				if(ghostsHunting && ghostHasLineOfSight(g.pos, camPos)) {
					g.lastKnownPlayerPos = camPos;
					g.hasLastKnown = true;
				}

				// Mode transitions. Starting a chase needs g.hasLastKnown, not
				// just the Hunt phase: a ghost that never saw the player has
				// nothing to walk toward and stays on patrol.
				if(ghostsHunting && g.hasLastKnown && g.mode != GhostMode::Chase) {
					if(g.mode == GhostMode::Patrol) {
						// Remember where on the loop we're leaving, start a
						// fresh trail there.
						g.resumeIdx = g.targetIdx;
						g.resumeDist = g.distAlongSegment;
						g.trail.clear();
						g.trail.push_back(g.pos);
					}
					// Out of Return, the existing trail is still the way home.
					g.mode = GhostMode::Chase;
					g.stuckCheckPos = g.pos;
					g.stuckTimer = 0.0f;
				} else if(!ghostsHunting && g.mode == GhostMode::Chase) {
					// Empty trail = the chase never went anywhere.
					g.mode = g.trail.empty() ? GhostMode::Patrol : GhostMode::Return;
				}

				// Move, per mode. Each branch leaves a `moveDir` for the facing
				// code below.
				glm::vec2 moveDir(0.0f);

				if(g.mode == GhostMode::Chase) {
					// Toward the last place the player was SEEN, not their live
					// position.
					glm::vec2 toPlayer(g.lastKnownPlayerPos.x - g.pos.x, g.lastKnownPlayerPos.z - g.pos.z);
					float d = glm::length(toPlayer);
					if(d > 1e-4f) {
						moveDir = ghostSteer(g, toPlayer / d);
						if(moveDir != glm::vec2(0.0f)) {
							// min(step, d): stops the ghost overshooting
							// straight past a player it has already reached
							float step = std::min(g.chaseSpeed * deltaT, d);
							g.pos.x += moveDir.x * step;
							g.pos.z += moveDir.y * step;
						}
						// Out of the loop to avoid getting stuck in objects when hunt starts
						ghostResolveWalls(g.pos);
					}

					// Giving up: a ghost pinned against a closed wall or on the lastKnownPlayerPos
					// will return to normal patrol after a timer
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

					// Breadcrumb. Dropped by distance travelled, not by time, (trail density not dependant on frame rate)
					if(g.trail.empty()) {
						g.trail.push_back(g.pos);
					} else if(glm::length(glm::vec2(g.pos.x - g.trail.back().x,
													g.pos.z - g.trail.back().z)) >= GHOST_TRAIL_SPACING) {
						// Loop check: if this point revisits an earlier crumb, drop
						// everything after it (oldest match = biggest loop cut).
						// Skip last 2 crumbs, always within prune radius.
						int cut = -1;
						for(int i = 0; i + 2 < (int)g.trail.size(); i++) {
							glm::vec2 delta(g.pos.x - g.trail[i].x, g.pos.z - g.trail[i].z);
							if(glm::length(delta) < GHOST_TRAIL_PRUNE_RADIUS) {
								cut = i;
								break;
							}
						}
						if(cut >= 0) {
							// No new crumb: we're standing on trail[cut]
							// already, near enough for it to be the head.
							g.trail.resize((size_t)cut + 1);
						} else {
							g.trail.push_back(g.pos);
						}
					}
				} else if(g.mode == GhostMode::Return) {
					// Walk the breadcrumbs backwards, popping each as reached.
					// Faster than the chase -- dead time for the player.
					glm::vec3 target = g.trail.back();
					glm::vec2 delta(target.x - g.pos.x, target.z - g.pos.z);
					float d = glm::length(delta);
					float step = GHOST_RETURN_SPEED * deltaT;
					if(d <= step) {
						// Reached: land exactly on it (a known-walkable point)
						// and drop it.
						g.pos.x = target.x;
						g.pos.z = target.z;
						g.pos.y = target.y;
						if(d > 1e-4f) moveDir = delta / d;
						g.trail.pop_back();
						if(g.trail.empty()) {
							// Home. Restoring the saved leg and distance resumes
							// the loop mid-stride, not at the nearest waypoint.
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
					// Patrol: walks the waypoint loop at constant speed
					// (distance-based, so `speed` is a real units/s figure).
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
					// Authored, not steered: drives `pos` directly with no wall
					// resolution, so a nearby collider can't shove a ghost off
					// its loop.
					g.pos = glm::mix(from, to, t);
					if(segLen > 0.0f) {
						moveDir = glm::normalize(glm::vec2(to.x - from.x, to.z - from.z));
					}
				}

				// Facing, eased toward travel. A stationary ghost keeps its
				// yaw. +M_PI: the mesh's front faces -Z, not atan2's +Z.
				if(moveDir != glm::vec2(0.0f)) {
					float targetYaw = std::atan2(moveDir.x, moveDir.y) + (float)M_PI;
					// Shortest way round, or easing +179 -> -179 spins the ghost.
					float dYaw = targetYaw - g.yaw;
					while(dYaw > (float)M_PI)  dYaw -= 2.0f * (float)M_PI;
					while(dYaw < -(float)M_PI) dYaw += 2.0f * (float)M_PI;
					float maxStep = GHOST_TURN_SPEED * deltaT;
					g.yaw += glm::clamp(dYaw, -maxStep, maxStep);
				}

				// The chase telegraph, eased. Here, not in updateUniformBuffer(),
				// because this loop owns `mode` and has a deltaT.
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

				// The catch. Only a hunting ghost ends the run -- brushing one
				// on patrol would make the colour telegraph a lie.
				if(ghostsHunting && cheats.ghostsCanCatch && runState == RunState::Running) {
					float dx = camPos.x - g.pos.x;
					float dz = camPos.z - g.pos.z;
					// From the chest, not the eyes: the eye height is the top of
					// the body.
					float dy = std::abs((camPos.y - 0.9f) - g.pos.y);
					if(dx * dx + dz * dz < GHOST_CATCH_RADIUS * GHOST_CATCH_RADIUS &&
					   dy < GHOST_CATCH_VERTICAL) {
						runState = RunState::Caught;
						std::cout << "[run] caught by '" << g.instanceId << "'\n";
					}
				}

				// Non-hunt encounter: walking through a non-hunting ghost
				// doesn't end the run but snuffs the held torch. `burning` =
				// false is all it takes -- the ignition spring plays in reverse,
				// so it shrinks out rather than snapping dark. Triggered by the
				// spectral veil's own ramp (evaluated as updateUniformBuffer()
				// does), not GHOST_CATCH_RADIUS, so it fires when "inside the
				// ghost" first reads rather than at dead centre.
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

			// The way out. Standing in the exit box wins -- unless it's locked
			// and the key isn't in hand (atLockedExit flags it). Checked after
			// the ghosts, so a catch on the threshold beats reaching it.
			atLockedExit = false;
			if(exitHasBox && runState == RunState::Running) {
				bool inside = camPos.x >= exitBoxMin.x && camPos.x <= exitBoxMax.x &&
							  camPos.y >= exitBoxMin.y && camPos.y <= exitBoxMax.y &&
							  camPos.z >= exitBoxMin.z && camPos.z <= exitBoxMax.z;
				if(inside) {
					if(exitRequiresKey && findKeyInRing(exitKeyId) < 0) {
						atLockedExit = true;
					} else {
						runState = RunState::Escaped;
						std::cout << "[run] escaped\n";
					}
				}
			}

			// Gravity: constant downward acceleration integrated into a
			// velocity, resolved against the ground below (which zeroes it on
			// landing).
			camVerticalVelocity += movement.gravity * deltaT;
			camPos.y += camVerticalVelocity * deltaT;

			// Floor collision: the tallest surface under the player's XZ that
			// sits within MAX_STEP_HEIGHT of the feet is the standing height.
			// The step limit lets hole-shaped models (gates, doors) be walked
			// through.
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

				// Ramps (the staircase): same MAX_STEP_HEIGHT rule, but the
				// reported height varies continuously along the slope instead
				// of jumping a riser at a time.
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
					// Landed: stop falling instead of accumulating velocity forever
					if(camVerticalVelocity < 0.0f) {
						camVerticalVelocity = 0.0f;
					}
				}
				// Ground-contact for jumping. A little tolerance so an
				// irregular floor lifting the player slightly still counts.
				const float GROUND_EPSILON = 0.05f;
				grounded = feetY <= groundY + GROUND_EPSILON;
				// Set camera position to the new one + player height
				float preClampY = camPos.y;
				camPos.y = feetY + EYE_HEIGHT;

				// Whatever the clamp just pushed us up by is a snap: hand it to the
				// view smoothing below. Only upward, so landings and falls stay as
				// sharp as gravity made them.
				float lifted = camPos.y - preClampY;
				if(lifted > 0.0f) {
					eyeStepOffset = std::min(eyeStepOffset + lifted, MAX_EYE_STEP_OFFSET);
				}
			} else {
				// No-clip: walls are ignored, but the world floor still holds,
				// so you can walk through walls without falling out the bottom.
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

		// Decay the vertical view smoothing. Exponential: never overshoots, no
		// "still stepping" state -- a continuous climb settles at a small lag.
		eyeStepOffset *= std::exp(-deltaT / EYE_SMOOTH_TAU);

		// Walk-bob signal shared by the camera and both hands: one phase, eased
		// in/out by walkBobBlend so a start/stop doesn't snap the sway. Only
		// how each reads the phase differs.
		bool isWalking = grounded && (std::abs(m.x) > 0.01f || std::abs(m.z) > 0.01f);
		float bobTarget = isWalking ? 1.0f : 0.0f;
		walkBobBlend += (bobTarget - walkBobBlend) * (1.0f - std::exp(-deltaT / WALK_BOB_BLEND_TAU));
		if(isWalking) {
			walkBobPhase += WALK_BOB_SPEED * (sprinting ? 1.4f : 1.0f) * deltaT;
		}

		// View: rendered from the smoothed eye height; camPos is untouched so
		// collisions and gravity work on the exact position. A small head-bob
		// on the double-frequency signal, well under the hand's, so the world
		// barely nods. Presentation only, like eyeStepOffset.
		float camBob = std::sin(walkBobPhase * 2.0f) * CAM_BOB_VERTICAL * walkBobBlend;
		glm::vec3 eyePos = camPos - glm::vec3(0.0f, eyeStepOffset - camBob, 0.0f);
		View = glm::lookAt(eyePos, eyePos + front, up);

		// View stays the true first-person one -- billboards, held items and
		// the light/geometry culls all read it (via inverse(View)) and must
		// not follow the spectator. Only ViewPrj, what the frame is actually
		// rasterised with, is rebased onto the pulled-back debug camera.
		if(cheats.debugCam) {
			// On the frame it turns on, snap the orbit behind the player's
			// facing; after that IJKL/UO own it. camYaw 0 faces +X and the
			// orbit yaw is measured the same way, so +180 sits it behind.
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
			// Clamp shy of straight up/down (lookAt gimbal) and keep the
			// dolly range sane.
			dbgOrbitPitch = glm::clamp(dbgOrbitPitch, -85.0f, 85.0f);
			dbgOrbitDist = glm::clamp(dbgOrbitDist, 3.0f, 90.0f);

			float oy = glm::radians(dbgOrbitYaw);
			float op = glm::radians(dbgOrbitPitch);
			glm::vec3 dbgOffset = glm::vec3(std::cos(oy) * std::cos(op),
										   std::sin(op),
										   std::sin(oy) * std::cos(op)) * dbgOrbitDist;
			glm::vec3 dbgEye = eyePos + dbgOffset;
			// The spectator sits outside the room, so the wall and ceiling
			// between it and the player would fill the frame. Push its near
			// plane out to just short of the player: everything nearer is
			// clipped, leaving exactly the shell the real camera renders
			// from the inside. DEBUG_CAM_CLIP_MARGIN keeps a little air
			// around the player.
			float dbgNear = glm::max(0.2f,
				glm::length(dbgEye - eyePos) - DEBUG_CAM_CLIP_MARGIN);
			glm::mat4 dbgPrj = glm::perspective(FOVy, Ar, dbgNear, farPlane);
			dbgPrj[1][1] *= -1;
			ViewPrj = dbgPrj * glm::lookAt(dbgEye, eyePos, worldUp);
		} else {
			ViewPrj = Prj * View;
		}
		dbgCamWasOn = cheats.debugCam;

		// Camera-space basis for anything rigidly attached to the view (held
		// torch, held key): right/up/-front columns, eyePos translation.
		glm::mat4 camWm = glm::mat4(
			glm::vec4(right, 0.0f),
			glm::vec4(up, 0.0f),
			glm::vec4(-front, 0.0f),
			glm::vec4(eyePos, 1.0f)
		);

		// Wall tuck for both hands (see applyTuck). Resolved here, not in each
		// hand's block below: both probe against the same camWm this frame, and
		// both factors must advance every frame -- an un-advanced one would
		// stay pinned at whatever it read when the item left the hand. No-clip
		// keeps both hands out (nothing to rest an item against).
		{
			bool tuckActive = cheats.collisionEnabled;

			float torchTarget = 1.0f;
			if(tuckActive && handTorchInst != nullptr && handTorchCollected && cheats.handTorchEnabled) {
				torchTarget = handFreeReach(camWm, HAND_TORCH_OFFSET, HAND_TUCK_TORCH_PAD);
			}
			advanceReach(handTorchReach, torchTarget, deltaT);

			// Whichever key is drawn in the left hand: the held one, or the one
			// sinking after a lock took it. One factor for both -- same hand,
			// never both in frame.
			int tuckKeyIdx = heldKeyIdx() >= 0 ? heldKeyIdx() : keyLowerIdx;
			float keyTarget = 1.0f;
			if(tuckActive && tuckKeyIdx >= 0) {
				keyTarget = handFreeReach(camWm, pickups[tuckKeyIdx].handOffset, HAND_TUCK_KEY_PAD);
			}
			advanceReach(handKeyReach, keyTarget, deltaT);
		}

		// Held torch: a fixed camera-local offset, but only once picked up.
		// Before that it lies at its authored floor pose. With "Holding Torch"
		// off it's parked below the map. Flame and light drop via flameBurning().
		bool torchInHand = handTorchCollected && cheats.handTorchEnabled;
		if(handTorchInst != nullptr && !torchInHand) {
			// Floor pose if not yet picked up and not cheat-disabled; parked
			// below the map otherwise.
			handTorchInst->Wm = (!handTorchCollected && cheats.handTorchEnabled)
				? handTorchSpawnWm
				: glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
		} else if(handTorchInst != nullptr) {
			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			// Wall tuck first, walk bob on top -- folded into a copy of the
			// constants, not the finished matrix, so the two compose. This is
			// also the fix for the held torch's light: the flame anchor rides
			// this Wm, so pulling the model out of the wall pulls the light out
			// too.
			glm::vec3 tuckedOffset = HAND_TORCH_OFFSET;
			glm::vec3 tuckedTilt = HAND_TORCH_TILT_DEG;
			applyTuck(handTorchReach, tuckedOffset, tuckedTilt);

			glm::mat4 grip = handGrip(tuckedTilt, bobRollDeg);

			// Pick-up rise, cubic ease-out like the key's: only the Y offset
			// moves, so it composes with the tuck and bob.
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

		// Held key: same camera-anchored placement as the torch, live only once
		// picked up. Reuses the world instance -- pointing its Wm at the camera
		// every frame is the whole "in your hand" effect. Same bob phase and
		// sign as the torch (both hands swing together). Only the newest key on
		// the ring is drawn.
		if(heldKeyIdx() >= 0) {
			Pickup &held = pickups[heldKeyIdx()];
			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			// Tilt and offset off the item itself: the ring can hold a key or a
			// book, two shapes with their origins in different places. Same
			// wall tuck the torch gets; applyTuck reads the offset's sign for
			// which way "inward" is.
			glm::vec3 tuckedOffset = held.handOffset;
			glm::vec3 tuckedTilt = held.handTiltDeg;
			applyTuck(handKeyReach, tuckedOffset, tuckedTilt);

			glm::mat4 grip = handGrip(tuckedTilt, bobRollDeg);

			// Pick-up rise: only the Y offset moves, so the key slides up into
			// the pose it would have snapped to. Cubic ease-out, not linear
			// (which stops dead and reads mechanical). Clamped to 0 once over.
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

		// Key spent on a lock: the mirror of the rise, played downward, cubic
		// ease-IN. Drawn from here, not the held-key block, because the key is
		// already off the ring. When the fall ends the instance goes below the
		// map.
		if(keyLowerIdx >= 0) {
			keyLowerElapsed = std::min(keyLowerElapsed + deltaT, KEY_RAISE_DURATION);
			float t = keyLowerElapsed / KEY_RAISE_DURATION;
			float eased = t * t * t;
			float lowerY = -KEY_RAISE_DROP * eased;

			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			Pickup &sinking = pickups[keyLowerIdx];

			// Tucked too, off the same handKeyReach: without it the key snaps
			// out to the extended pose on the frame the lock takes it, through
			// the door it was just used on.
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


// This is the main: probably you do not need to touch this!
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