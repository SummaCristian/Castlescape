// THIS IS THE FILE YOU MUST START FROM!

// This has been adapted from the Vulkan tutorial
#include <sstream>
#include <limits>
#include <array>
#include <algorithm>

#include <json.hpp>

#include "modules/Starter.hpp"
#include "modules/TextMaker.hpp"
#include "modules/Scene.hpp"
#include "custom/UiQuad.hpp"
#include "custom/CheatHud.hpp"
#include "custom/PauseMenu.hpp"
#include "custom/StartScreen.hpp"
#include "custom/SceneColliders.hpp"
#include "custom/SceneMaterials.hpp"
#include "custom/SceneLights.hpp"
#include "custom/Flame.hpp"
#include "custom/ExitGlow.hpp"
#include "custom/CubeShadowMap.hpp"
#include "custom/DebugLines.hpp"
#include "custom/HuntCycle.hpp"

// Our own files, and where to start reading.
//
//   custom/SceneColliders.hpp   the boxes and ramps the player walks into
//   custom/SceneMaterials.hpp   what each surface is made of
//   custom/SceneLights.hpp      the scene's lights
//   custom/UiQuad.hpp           coloured rectangles for the HUD
//   custom/CheatHud.hpp         the cheat menu, opened with L
//
// Each of the first three reads its own data file from assets/scenes/ at
// startup, so the scene can be changed without touching C++:
//
//   scene.json      which models exist and where they are placed
//   colliders.json  hand-authored collision shapes for models an auto-fitted
//                   box gets wrong, like the gate's archway
//   materials.json  surface parameters, one entry per model
//   lights.json     the light sources and the ambient light
//   gameplay.json   the hunt cycle's timings, the ghosts' patrols, and where
//                   the run is won (see custom/HuntCycle.hpp)
//
// The shaders are in source/shaders/. PosNormUV.vert and CookTorrance.frag are
// the pair that draws the scene; the other two draw the HUD.
//
// notes.md at the repo root explains the reasoning behind all of it.

// The uniform buffer object used in this example
struct UniformBufferObject {
	alignas(16) glm::mat4 mvpMat;
	alignas(16) glm::mat4 mMat;
	// inverse-transpose of mMat. Normals can't ride the world matrix or a
	// non-uniform scale tilts them off the surface (the road is scaled [1,4,1]).
	// A mat4 rather than a mat3 to avoid std140's column-padding rules.
	alignas(16) glm::mat4 nMat;
	// Cook-Torrance material. mD isn't here, it's the albedo texture.
	//
	// Must match, field for field, the block declared by the four shaders that
	// see it: PosNormUV.vert and CookTorrance.frag at set 1, Shadow.vert and
	// ShadowCube.vert at set 0. (Flame.vert and Spark.vert sit at the same
	// binding but read a FlameUniformBufferObject of their own, so they are not
	// bound by this layout.)
	//
	// No explicit padding of ours: the scalars below fall into std140's vec4
	// slots on their own, as [mS.xyz | roughness], [F0 | k | flatNormals |
	// interiorAmbient], [time | ambientWeight | glow | metallic]. The struct's
	// 16-byte alignment rounds its size to those same 240 bytes, so C++ and
	// GLSL agree. `glow` and `metallic` were added into the third slot's spare
	// room, which is why the size did not move when they appeared.
	alignas(16) glm::vec3 mS;	// specular color
	float roughness;			// rho: width of the microfacet distribution
	float F0;					// reflectance seen head-on
	float k;					// diffuse share of the BRDF
	int flatNormals;			// 1: derive the face normal in the shader
	// 1: take the hemispheric ambient of a vertical surface instead of the one
	// this surface's normal implies. For interiors, where the sky/ground blend
	// the model is built on has no meaning. See SceneMaterials.hpp.
	int interiorAmbient;
	// Seconds since startup, the same value for every instance in a frame.
	// Piggybacks the per-object UBO instead of going in
	// GlobalUniformBufferObject, which would shift LightData[] off the offset
	// the comment there notes. Read only by the Flame shaders, which animate
	// the torch flames entirely on the GPU; the scene shaders declare it and
	// ignore it, since both pipelines share DSLlocal and so this one struct.
	float time;
	// This model's share of indirect light, overriding the scene's. Negative
	// means "inherit gubo.ambientWeight", and that is the common case: only
	// the interior models carry one. See Material::ambientWeight.
	float ambientWeight;
	// 0..1: this instance's currently-gazed-at focus highlight strength. Set
	// per-instance (not per-model, unlike the fields above) in
	// updateUniformBuffer()'s per-instance loop by comparing against
	// gazedInstance. Read only by CookTorrance.frag; the other three shaders
	// that share this layout declare and ignore it, same as time above.
	float glow;
	// 1: shade this model as a metal -- no diffuse lobe, and an indirect term
	// that reflects the room instead of scattering it. See Material::metallic
	// in SceneMaterials.hpp and metalAmbient() in CookTorrance.frag.
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
	// here and uploaded. LightData's own alignas(16) forces the compiler to
	// pad the array start to a 16-byte boundary regardless, so this scalar
	// just rides in front of that padding like debugFlags does above.
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

// Set 2: the shadow-sampling data, bound once and read by CookTorrance.frag.
// One matrix per 2D-shadow light (NUM_SHADOW_MAPS_2D, LightConstants.glsl --
// just the sun today), the SAME view-projection its own shadow pass rendered
// with (see computeShadowMatrices()). The torches don't need a matrix here
// any more: a cube map is sampled by direction, not by transforming into its
// clip space, so their light-space math never leaves computeShadowMatrices()/
// populateCommandBuffer(). Static for the life of the program, since the sun
// doesn't move, but still re-mapped every frame in updateUniformBuffer()
// rather than once at startup: map() writes into a per-swapchain-image
// buffer slot, and mapping only slot 0 would leave the others holding
// whatever was there at allocation time.
struct ShadowUniformBufferObject {
	alignas(16) glm::mat4 lightSpace[NUM_SHADOW_MAPS_2D];
};

// One torch's cube shadow CAPTURE data (ShadowCube.vert/frag, PShadowCube),
// set 1 there. Field-for-field the same layout those two shader stages
// declare. A uniform buffer, not a push constant, and re-mapped every frame
// in updateUniformBuffer() for every torch, including the six static ones --
// see ShadowCube.vert's header for why a push constant can't do this job:
// the "main" command buffer is recorded once per swapchain image and reused
// every frame after that (Starter.hpp's submitCommandBuffer()/
// updateCommandBuffers()), so a push constant's value would be frozen at
// whatever it was the moment that recording happened and never updated
// again -- fatal for the held torch, which moves every frame.
struct ShadowCubeUniformBufferObject {
	alignas(16) glm::mat4 lightViewProj[6];
	alignas(16) glm::vec4 lightPos;	// xyz used, w is padding
};

// Which of the 6 faces a draw call is for. Safe as a push constant unlike
// the matrix/position above: it's determined by WHERE in the recorded
// command buffer the draw sits (this loop iteration is always face i), not
// by anything that changes after the buffer is recorded.
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

// Shared by all four post-processing passes (bright pass, the two blur
// directions, composite), matching PostUniformBufferObject in BloomBright.frag
// / BloomBlur.frag / Composite.frag field for field. Each pass gets its own
// copy with the fields it cares about filled in; the rest are simply unread,
// which is cheaper than maintaining four nearly identical blocks.
//
// No alignas() needed anywhere here: two vec2s then six 4-byte scalars is
// already exactly what std140 lays out, with nothing to pad.
struct PostUniformBufferObject {
	glm::vec2 texelSize;	// 1/width, 1/height of the SOURCE texture
	glm::vec2 blurDir;		// (1,0) or (0,1); read by BloomBlur.frag only
	float threshold;		// bright pass: luminance above which anything blooms
	float knee;				// bright pass: how soft the threshold's shoulder is
	float bloomIntensity;	// composite: how much bloom is added back
	float exposure;			// composite
	int debugFlags;			// composite: LIGHT_DEBUG_NO_TONEMAP
	float time;
	// composite: 0 normally, ramping to 1 as the player escapes. Blends the
	// finished frame towards white.
	//
	// It has to be a term of its own rather than more `exposure`, because the
	// tone map (c / (Y + 1), see Composite.frag) approaches 1 asymptotically:
	// no exposure, however large, actually reaches white, and worse, it
	// reaches it at wildly different rates for a lit wall and for a dark
	// corner -- so cranking exposure alone doesn't white the frame out, it
	// flattens it into a grey with the bright parts still winning. The
	// exposure ramp is still there and still doing the work of blowing the
	// scene out; this is what finishes the job.
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

// Cheap 1D value noise, for the torch fire envelope in updateUniformBuffer().
// The same shape as the noise the flame shaders use, just on the CPU: the
// flicker has to be computed once and shared by the flame, its sparks and the
// point light it casts (see TorchFlame), and only the CPU sees all three.
//
// Integer hash rather than the usual fract(sin(x)*43758.5) trick: that one is
// famously driver-dependent in GLSL and, in C++ at double precision, simply
// does not decorrelate well enough at the small inputs used here.
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
	// The scene pass. No longer draws to the swapchain: it now renders into an
	// offscreen FLOATING-POINT colour attachment, which is what makes bloom
	// possible at all. See buildHdrAttachments() and the render graph comment
	// on populateCommandBuffer().
	RenderPass RP;
	Pipeline P;

	// The ghosts' pipeline, drawn into the SAME pass as P right after it (see
	// the "Spectral" technique in scene.json and PRs[1] below). Separate from P
	// for two reasons that both come down to a ghost not being a surface:
	// Spectral.frag computes no BRDF and reads no shadow map, so it needs
	// neither DSLshadowSample nor the lighting half of the fragment cost, and
	// the pipeline itself has to differ anyway -- alpha blending on, which is a
	// pipeline-creation flag and cannot be switched per draw.
	//
	// It shares P's vertex shader (PosNormUV.vert) and P's DSLlocal, so the
	// ghosts keep riding the same per-instance uniform buffer as every other
	// prop and updateUniformBuffer() needs no branch for them beyond the chase
	// blend (see ubo.F0 there).
	Pipeline Pspectral;

	// The ghosts' DEPTH PREPASS (shaders/SpectralDepth.frag), drawn over the
	// same instances immediately before Pspectral is. It writes the depth of
	// the nearest ghost surface and returns the colour attachment untouched, so
	// that the colour pass behind it can reject the ghost's own interior --
	// the feet inside the robe -- instead of blending it under the body. See
	// that shader's header for the mechanism and populateCommandBuffer() for
	// why THIS one is the pipeline registered with the technique while
	// Pspectral is the one issued by hand.
	//
	// Same DSLs as Pspectral (nothing here reads them, but the layout has to
	// match the sets Scene binds), same back-face culling, and the default
	// VK_COMPARE_OP_LESS rather than Pspectral's LESS_OR_EQUAL, which is the
	// whole point: LESS is what leaves the minimum in the depth buffer.
	Pipeline PspectralDepth;

	// Shadow mapping, 2D branch: one depth-only render pass per 2D
	// shadow-casting light (NUM_SHADOW_MAPS_2D, LightConstants.glsl -- just
	// the sun today) and ONE pipeline shared across all of them. Reusing
	// PShadow instead of one pipeline per pass relies on Vulkan's
	// render-pass-compatibility rule: RPShadow2D[i] all use the identical
	// AT_DEPTH_ONLY attachment configuration, so a pipeline created against
	// one of them works with any of the others. Unlike RP/P, both are
	// created once in localInit() and never touched by a resize: an
	// offscreen depth target doesn't depend on the window, so there's no
	// reason to tear it down and rebuild it the way the swapchain-sized
	// resources are.
	//
	// Only the CookTorrance technique is drawn into these (see
	// populateCommandBuffer()) -- the flames aren't occluders and shouldn't
	// occlude either, being translucent, so they're skipped rather than given
	// their own shadow logic. Same for the cube branch below.
	RenderPass RPShadow2D[NUM_SHADOW_MAPS_2D];
	Pipeline PShadow;

	// Shadow mapping, CUBE branch (the torches): a real 6-face cube map per
	// point light instead of the old two-perspective-map workaround -- see
	// CubeShadowMap.hpp for why (linear-distance storage, one flat bias).
	//
	// RPShadowCubeCompat exists ONLY to mint a VkRenderPass compatible with
	// every face framebuffer below: RenderPass::createRenderPass() is
	// private, so the sole way to obtain a spec-compatible VkRenderPass
	// through this class's public surface is to let a full RenderPass build
	// one for itself and read its .renderPass back out. Its own attachment
	// image/framebuffer (1-layer, SHADOW_MAP_RES sized) are never rendered
	// into or read -- unavoidable bookkeeping to stay inside RenderPass's
	// public API instead of duplicating vkCreateRenderPass by hand.
	//
	// The 36 real per-face framebuffers (one per torch per cube face) are
	// built manually in createCubeShadowMaps() against
	// RPShadowCubeCompat.renderPass, because they attach single-layer views
	// into a 6-layer cube image -- something FrameBufferAttachment has no
	// support for (it always creates a plain VK_IMAGE_VIEW_TYPE_2D, 1 layer).
	RenderPass RPShadowCubeCompat;
	Pipeline PShadowCube;
	CubeShadowMap torchCube[NUM_SHADOW_CUBES];
	// Shared by every torch's cube view: all render into R32_SFLOAT images at
	// the same resolution, so one CLAMP_TO_EDGE/linear sampler suffices.
	// Starter.hpp's own TextureSampler, same class FrameBufferAttachment
	// uses internally, rather than a raw VkSampler -- public API, no need to
	// hand-roll vkCreateSampler.
	TextureSampler cubeShadowSampler;

	// set 1 for the cube shadow CAPTURE pass (PShadowCube) -- one uniform
	// buffer per torch cube slot, holding that torch's 6 current face
	// view-projection matrices plus its world position. A DescriptorSet
	// member of its own, same reasoning as DSglobal (not per scene
	// instance), created/destroyed alongside it in
	// pipelinesAndDescriptorSetsInit()/Cleanup(). See ShadowCube.vert's
	// header for why this has to be a uniform buffer, re-mapped every frame,
	// rather than the push constant it replaced.
	DescriptorSetLayout DSLshadowCubeCapture;
	DescriptorSet DSshadowCube[NUM_SHADOW_CUBES];

	// set 2 for the main pass's shadow sampling: one UBO (the 2D light-space
	// matrices) plus one sampler binding per shadow map (2D then cube), read
	// by CookTorrance.frag's shadowFactor(). DSLlocal/DSLglobal stay set 1/0.
	//
	// No DescriptorSet member of its own: unlike DSglobal, this one goes
	// through Scene's ordinary per-instance machinery instead (P is given
	// this as a third layout below, so every CookTorrance instance gets its
	// own copy, same as its DSLlocal one). That means
	// NUM_SHADOW_MAPS_2D+NUM_SHADOW_CUBES+1 redundant, identical descriptor
	// sets per instance -- wasteful, but cheap at this instance count, and it
	// avoids hand-rolling a THIRD way to bind a descriptor set alongside
	// Scene's existing one.
	DescriptorSetLayout DSLshadowSample;
	// View-projection matrix each 2D shadow pass rendered with, index-matched
	// to LightData::shadowIndex for a direct/spot light. Computed once in
	// computeShadowMatrices() (the sun is static) and reused both as the push
	// constant Shadow.vert takes and as the UBO CookTorrance.frag samples
	// against.
	glm::mat4 shadowLightSpace2D[NUM_SHADOW_MAPS_2D];
	// The six face view-projection matrices for each torch's cube map,
	// index-matched [LightData::shadowIndex][face] (face order: see
	// CUBE_FACE_DIR in CubeShadowMap.hpp). Computed once, same reasoning.
	glm::mat4 torchFaceMatrices[NUM_SHADOW_CUBES][6];
	// World position of each cube-mapped torch, index-matched to
	// LightData::shadowIndex -- ShadowCube.frag needs it (the light to
	// measure distance from) and so does shadowFromCube() in
	// CookTorrance.frag, which reads it via gubo.lights[i].pos instead; this
	// copy is what populateCommandBuffer() hands to the push constant.
	glm::vec3 torchLightPos[NUM_SHADOW_CUBES];
	// How many of torchFaceMatrices/torchLightPos are actually populated --
	// fewer than NUM_SHADOW_CUBES if lights.json authors fewer shadow-casting
	// point lights than there are slots. Set once by computeShadowMatrices(),
	// then bumped once more in localInit() if the held torch exists, to also
	// cover HAND_TORCH_SHADOW_INDEX.
	int activeCubeShadows = 0;
	// SHADOW_CUBE_RES rather than a literal: CookTorrance.frag derives the cube
	// path's depth bias from the world size of one texel of this map, so the
	// shader has to know the same number. See LightConstants.glsl.
	static constexpr int SHADOW_MAP_RES = SHADOW_CUBE_RES;
	// Far clip for every torch's cube map (computeShadowMatrices()) and the
	// clear value ShadowCube.frag's output gets reset to before each face
	// pass: with nothing drawn a fragment's "distance" should read as
	// infinity/unlit, and any value >= this far plane does that -- PROVIDED
	// shadowFromCube() (CookTorrance.frag) never actually gets queried
	// beyond it, which was true back when this was 15: with the old g/beta
	// the torch's radiance was already down to a few percent by 15 units
	// out, invisibly below LIGHT_ATTEN_EPS's per-pixel skip soon after.
	//
	// That invariant broke once the falloff was retuned for a longer reach
	// (lower beta, higher g, a soft RADIANCE_CAP replacing the old
	// unbounded-then-culled shape): the torch now stays visibly bright well
	// past 15 units, so any wall farther than that from the torch WAS being
	// queried -- and got the clear value back as its "nearest occluder",
	// which is closer than the wall's own real distance, so it read as
	// falsely shadowed. That's what looked like the torch's light "only
	// reaching a fixed radius" with a hard edge at that radius, rather than
	// the shadow bug it actually was. Raised to comfortably cover the
	// dungeon's own ~60-unit footprint (TORCH_LIGHT_CULL_DIST's comment,
	// main.cpp) so the far plane stops being reachable during normal play.
	static constexpr float TORCH_SHADOW_FAR_CONST = 60.0f;
	// Near clip for every torch's cube map -- close enough that only the
	// torch fixture itself (not an occluder, Material::castsShadow) falls
	// inside it. A member (not a computeShadowMatrices() local) because
	// updateHandTorchShadow() needs the same number every frame.
	static constexpr float TORCH_SHADOW_NEAR_CONST = 0.05f;
	// The cube slot reserved for the held torch, one past the six lights.json
	// hands out (SceneLights::init, nextShadowIndexCube, torchW1/W2/E1/E2/DC/
	// DV): the held torch never goes through lights.json -- its Wm doesn't
	// exist in a meaningful form until GameLogic() starts overwriting it
	// every frame, so unlike the static torches it can't get a fixed
	// shadowIndex at SceneLights::init() time or fixed face matrices at
	// computeShadowMatrices() time. Instead updateHandTorchShadow() recomputes
	// torchFaceMatrices[HAND_TORCH_SHADOW_INDEX]/torchLightPos[..] every frame
	// in updateUniformBuffer(), before populateCommandBuffer() reads them.
	static constexpr int HAND_TORCH_SHADOW_INDEX = NUM_SHADOW_CUBES - 1;

	// The DYNAMIC shadow-cube pool: everything between lights.json's own
	// fixed slots (0..dynamicShadowSlotBase) and the held torch's reserved
	// last one (HAND_TORCH_SHADOW_INDEX). Set once in localInit(), right
	// after computeShadowMatrices() reports how many fixed slots
	// sceneLights.all() actually used -- not a literal 6, so this stays
	// correct if lights.json's own torch count ever changes.
	int dynamicShadowSlotBase = 0;
	// dynamicSlotOccupant[s] is an index into torchFlames for whichever
	// flame currently holds dynamic slot (dynamicShadowSlotBase + s), or -1
	// if the slot is empty (more slots than shadowCandidate flames). Sized
	// NUM_SHADOW_CUBES for simplicity -- only the entries covering the
	// dynamic range are ever touched -- rather than adding another
	// compile-time constant for the pool's width.
	std::array<int, NUM_SHADOW_CUBES> dynamicSlotOccupant{};

	// Caching for the static cube shadow slots (everything except
	// HAND_TORCH_SHADOW_INDEX, which moves every frame and is excluded from
	// this bookkeeping entirely): lastRenderedOccupant[t] is the "occupant
	// identity" the slot's image last actually had its 6 faces rendered
	// for -- t itself for a fixed lights.json slot (that identity never
	// changes once set), or dynamicSlotOccupant[t] for a dynamic-pool slot.
	// SHADOW_SLOT_UNSET means "never rendered", which every slot starts as:
	// the image is otherwise left at VK_IMAGE_LAYOUT_UNDEFINED, which the
	// samplerCube array descriptor is not allowed to be bound against, so
	// every slot needs exactly one render even if it stays empty forever
	// (see recordCubeSlotFaces()'s begin/end-only path for an unoccupied
	// slot). Compared every frame against the slot's CURRENT identity
	// (cheap: NUM_SHADOW_CUBES int compares) in updateUniformBuffer(); a
	// mismatch queues that slot's six faces into pendingFaceMask and updates the
	// stored identity, so a slot whose occupant hasn't changed since its
	// last render is never touched again -- static point lights in this
	// scene (all of them but the held torch: SceneLights::update() never
	// moves a point light, and updateDynamicShadowSlots() only reassigns a
	// dynamic slot when a nearer candidate genuinely outbids the current
	// one) end up rendered exactly once for the life of the program instead
	// of on every one of the ~9000+ per-frame draw calls the old
	// unconditional every-frame loop cost across all of them combined.
	static constexpr int SHADOW_SLOT_UNSET = -2;
	std::array<int, NUM_SHADOW_CUBES> lastRenderedOccupant;
	// What the diffs found stale this frame, as a BITMASK OF FACES per slot
	// (bit f = CUBE_FACE_DIR[f]), re-captured by submitCubeShadowCaptures()
	// right after the DSshadowCube mapping loop writes their fresh matrices/
	// position for currentImage (see updateUniformBuffer()) -- not inside the
	// "main" NamedCommandBuffer, which is recorded once per swapchain image and
	// replayed unmodified every frame after that (see
	// ShadowCubeUniformBufferObject's comment), so anything recorded into IT
	// would still redraw every frame regardless of this caching.
	//
	// Per FACE rather than per slot because a cube map is six independent
	// images that happen to share a handle: a face nobody re-renders keeps the
	// texels it was last drawn with, and for the static half of this scene
	// those texels stay correct forever. A ghost walking past a torch changes
	// what one, sometimes two of that torch's faces see -- redrawing all six
	// meant clearing and refilling 6 MB of render target to fix 1 MB of it, and
	// at ~1024^2 R32 per face the clears alone were the cost, not the draws.
	// A slot whose LIGHT changed still gets all six (ALL_CUBE_FACES): nothing
	// about its old content survives moving the camera it was shot from.
	std::array<uint8_t, NUM_SHADOW_CUBES> pendingFaceMask{};
	static constexpr uint8_t ALL_CUBE_FACES = 0x3F;

	// ---- The per-frame cube-shadow submission ----
	//
	// Every cube capture that can change from one frame to the next -- the held
	// torch, plus whatever faces pendingFaceMask marks -- is recorded into ONE
	// command buffer of our own, re-recorded and submitted every frame just
	// before Starter.hpp's drawFrame() submits the "main" one.
	//
	// Its own pool, because the framework's commandPool is created with flags =
	// 0 and a buffer allocated from a pool without
	// VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT may not be re-recorded.
	// Freeing and reallocating per frame would work but churns pool memory for
	// nothing.
	VkCommandPool shadowCommandPool = VK_NULL_HANDLE;
	// One buffer and one fence per swapchain image: the buffer for image i is
	// still executing until its fence signals, and re-recording a buffer the
	// GPU is reading is undefined behaviour. Ours rather than reusing
	// inFlightFences: those are signalled by the MAIN submit, and a later
	// submit finishing does not by itself prove an earlier one did.
	std::vector<VkCommandBuffer> shadowCB;
	std::vector<VkFence> shadowCBFence;
	// Where the held torch's light was, and whether it was casting at all, when
	// its cube was last captured. Its faces are only redrawn when one of the
	// two changed (or a mover marked them), which is what turns a player
	// standing still into no shadow work at all. The sentinel is a position
	// nothing can occupy, so the first frame always captures.
	glm::vec3 lastHandTorchCapturePos{std::numeric_limits<float>::infinity()};
	bool lastHandTorchCaptureOccupied = false;

	// The occupant diff above answers "did this slot's LIGHT change hands",
	// which is the only reason a slot's capture can go stale as long as every
	// occluder in the scene is nailed down. The ghosts are not, and neither is
	// a door mid-swing. While lights.json still had a sun that didn't show: a
	// mover's visible shadow was the SUN's 2D map, which is re-rendered inside
	// the "main" command buffer every frame and so followed it (see
	// Shadow.vert's header). With the sun gone the only shadow a ghost casts is
	// the torches' cube one -- and that one stayed frozen in whatever pose the
	// slot happened to be captured in.
	//
	// So movers get their own invalidation: queueMoverCubeSlotRenders(). The
	// candidates are the ghosts and the door leaves, NOT every instance whose
	// Wm changes -- a floating pickup key spins on the spot forever and would
	// keep every slot near it re-rendering for a shadow nobody can pick out,
	// which is exactly the per-frame cost this cache exists to avoid. Of those
	// candidates only the ones materials.json marks castsShadow actually make
	// the list, which today means the doors: the ghosts moved continuously
	// enough that capturing them cost more than their shadows were worth, and
	// they are switched off there rather than here (see materials.json's
	// "ghost" entry, and the filter in queueMoverCubeSlotRenders()). Built once
	// (the instances outlive the level) on first use.
	std::vector<Instance *> movingOccluders;
	// Wm each of those had when a slot was last re-captured for it, so a mover
	// standing still costs one mat4 compare and no draws.
	std::vector<glm::mat4> movingOccluderWm;
	bool moverListBuilt = false;
	// Which of slot t's faces had a mover in them at their last capture, same
	// bit layout as pendingFaceMask. Needed for the faces a mover LEAVES:
	// walking from a torch's +X face to its +Z one changes both, and nothing
	// about the ghost's new position tells +X that it still has the ghost
	// painted on it. Same for leaving the light's reach entirely.
	std::array<uint8_t, NUM_SHADOW_CUBES> slotFaceHadMover{};
	// How far each slot's light still matters, i.e. the distance past which a
	// mover cannot cast a shadow anyone could see through it. Derived from
	// that light's OWN falloff rather than picked as a radius -- see
	// shadowRelevantReach() -- and written wherever torchLightPos[] is, so the
	// two always describe the same light.
	std::array<float, NUM_SHADOW_CUBES> torchShadowReach{};

	// How much (in-game) time between updateDynamicShadowSlots() calls: the
	// candidates are static objects, only the player moves, so this doesn't
	// need a per-frame answer. Startup value equal to the interval so the
	// very first updateUniformBuffer() call fires it immediately (else the
	// dynamic slots would render whatever garbage torchFaceMatrices/
	// torchLightPos happen to start with).
	float shadowReassignTimer = 0.3f;
	static constexpr float SHADOW_REASSIGN_INTERVAL = 0.3f;
	// A waiting candidate must be closer than an occupied slot's current
	// occupant by more than this factor to take the slot. Without a margin,
	// a player standing near the distance boundary between two candidates
	// would flip the slot -- and force a fresh shadow render, since there's
	// no cross-fade between "has a shadow" and "doesn't" -- on essentially
	// every re-evaluation.
	static constexpr float SHADOW_SWAP_MARGIN = 1.15f;

	// Models, textures and Descriptors (values assigned to the uniforms)
	DescriptorSet DSglobal;

	// ---- HDR post-processing chain ----
	//
	// scene (RGBA16F, MSAA + resolve)
	//   -> bright pass  (quarter res: threshold + downsample)
	//   -> blur H       (quarter res: separable gaussian)
	//   -> blur V       (quarter res: separable gaussian)
	//   -> composite    (swapchain: scene + bloom, then tone map)
	//
	// All five are recorded into the same "main" command buffer in that order,
	// and every offscreen pass uses ATDEP_SIMPLE, whose second dependency
	// (subpass 0 -> EXTERNAL, COLOR_ATTACHMENT_OUTPUT -> FRAGMENT_SHADER) is
	// exactly the barrier that makes each pass's writes visible to the next
	// pass's texture reads. Its FIRST dependency matters just as much and is
	// easier to miss: it is a write-after-read barrier against the PREVIOUS
	// frame, which is what keeps frame N+1 from overwriting an offscreen
	// target while frame N is still sampling it. There are two frames in
	// flight and only one image per offscreen attachment, so without it that
	// race is real.
	RenderPass RPbright, RPblurH, RPblurV, RPcomposite;
	VertexDescriptor VDpost;
	// One layout for the passes that read a single texture, one for the
	// composite, which reads the scene and the bloom result.
	DescriptorSetLayout DSLpost1, DSLpost2;
	Pipeline Pbright, PblurH, PblurV, Pcomposite;
	DescriptorSet DSbright, DSblurH, DSblurV, DScomposite;
	Model *Mpost = nullptr;

	// The attachment descriptions each of the four passes is built from. Held
	// as members rather than locals because RenderPass::init copies the vector
	// but the per-attachment code then points back at the copy for the whole
	// pass's lifetime -- and because onWindowResize() has to rebuild them at
	// the new size.
	std::vector<AttachmentProperties> hdrAtt, brightAtt, blurHAtt, blurVAtt, compositeAtt;

	// The bloom chain runs at 1/BLOOM_DIV of the screen in each axis. A quarter
	// is the usual choice and it is not only about cost: a fixed-width gaussian
	// on a quarter-res image covers four times as much of the final picture, so
	// downsampling is how you get a WIDE soft halo out of a cheap 9-tap kernel
	// instead of a tight one.
	static constexpr int BLOOM_DIV = 4;

	// The 3D scene (RP, the hdrAtt chain CookTorrance.frag draws into) renders
	// at this fraction of the window's actual resolution in each axis, i.e.
	// renderScale^2 of its pixel count -- 0.8 is ~64%. Composite.frag then
	// upsamples it back up to the real window size through the same
	// bilinear sampler it already reads srcTex with, so the OUTPUT still
	// fills the window at full resolution; only the scene's own detail is
	// computed at fewer pixels. Free performance-wise in proportion to that
	// pixel-count cut (every fragment invocation this saves is one this
	// project's forced per-sample shading -- see msaaSamples' comment above
	// -- would otherwise have run the full light loop for), paid for in a
	// slightly softer scene, which fog/vignette/bloom/the general darkness of
	// a dungeon already hide well.
	//
	// The UI (txt/uiQuad/crosshair/hud/pauseMenu/startScreen) is NOT part of
	// this: those all render in their own separate command buffers/passes,
	// submitted after RPcomposite (see populateCommandBuffer()'s comment),
	// which stays at the window's real resolution unconditionally -- so text
	// and prompts stay perfectly sharp regardless of this value.
	//
	// A runtime member, not a compile-time constant: the "Render Scale"
	// slider in the cheat HUD (see its addSlider() call below) changes this
	// live, then replays the same rebuild path a real window resize already
	// uses -- onWindowResize(windowWidth, windowHeight) to recompute
	// RP/bloom sizes at the new scale, then RebuildPipeline() to actually
	// tear down and recreate the render targets at those sizes. Everything
	// that used to read the old compile-time constant (renderWidth()/
	// renderHeight(), bloomWidth()/bloomHeight(), and both call sites below)
	// reads this member instead, so nothing needed to change but this
	// declaration and its initial value.
	float renderScale = 0.8f;

	// The scene's own render resolution, some window dimension scaled by
	// renderScale and clamped to at least 1 (a minimised or absurdly narrow
	// window can't ask for a zero-sized image). Takes the window dimension
	// as a parameter rather than reading swapChainExtent directly, since
	// onWindowResize() needs to compute this from the NEW size it was just
	// handed, before swapChainExtent itself has necessarily been updated to
	// match.
	int renderWidth(int windowW) const {
		return std::max(1, (int)std::lround(windowW * renderScale));
	}
	int renderHeight(int windowH) const {
		return std::max(1, (int)std::lround(windowH * renderScale));
	}

	// The MSAA sample count as a log2 "level" (0 -> 1x, 1 -> 2x, 2 -> 4x, ...)
	// rather than the raw VkSampleCountFlagBits value, so the "MSAA" cheat
	// slider's fixed +/-1-per-press step (see CheatHud::addSlider) lands
	// exactly on the powers of two Vulkan sample counts have to be, instead
	// of needing a doubling step a plain additive slider can't express.
	// Starts at 2 (4x), matching msaaSamples' own initial value below.
	float msaaLevel = 2.0f;
	// The slider's upper bound, in the same units: set from
	// getMaxUsableSampleCount() once at startup (see localInit()) so a GPU
	// that can't do 16x is never offered it.
	float maxMsaaLevel = 2.0f;

	// Bright-pass threshold and knee, in luminance. Set high enough to clear
	// the SCENE's own peak radiance, not just 1.0: a sunlit wall with a pale
	// albedo lands a little either side of 1.0 once the sun and the ambient
	// term are added, so a threshold of 1.0 put a halo on every bright surface
	// in the castle and left the whole frame looking fogged. The flame writes
	// up to about 6, and the sparks higher, so there is a wide gap to sit in
	// and only things that are genuinely emissive get into the bloom buffer.
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

	// Flat-colored quad: the always-on center-screen dot used to aim
	// look-based interactions (see GameLogic()'s gaze test). Separate
	// UiQuad instance/named command buffer from uiQuad above, since the two
	// draw independent, unrelated content.
	UiQuad crosshair;

	// (Re)builds the crosshair dot centered on the current windowWidth/
	// windowHeight. Called once at init and again on every resize, since
	// the dot's pixel position depends on screen size; never needs to
	// change frame to frame otherwise, so it isn't called from the main
	// per-frame loop.
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

	// Flat-colored quads: dim overlay + button backgrounds behind the actual
	// pause menu's (ESC-triggered) text. Separate UiQuad instance/named
	// command buffer from uiQuad/crosshair above, same reasoning as
	// crosshair: independent, unrelated content.
	UiQuad pauseQuad;
	// The actual pause menu: dims the screen and freezes GameLogic() while
	// open (see GameLogic()'s overlayOpen() gating). ESC toggles it; see
	// updateUniformBuffer().
	PauseMenu pauseMenu;

	// Flat-colored quads: opaque backdrop + button backgrounds for the
	// launch screen. Separate UiQuad instance/named command buffer, same
	// reasoning as pauseQuad/crosshair above.
	UiQuad startScreenQuad;
	// The launch screen: open from the very first frame (see localInit()),
	// so the app boots into this instead of dropping the player straight
	// into the castle. Also reopened if the pause menu's Quit abandons the
	// current run. Folded into overlayOpen() like hud/pauseMenu, so it
	// freezes GameLogic() the same way.
	StartScreen startScreen;

	// Other application parameters
	float Ar;	// Aspect ratio

	glm::mat4 ViewPrj;
	glm::mat4 View;

	// Free-look camera state (position + orientation), persisted across frames.
	// Spawns inside the dungeon hall (dh), clear of the table and both torches,
	// now that the castle courtyard is gone -- there's no outdoor approach
	// to walk in from anymore.
	glm::vec3 camPos = glm::vec3(-33.5f, 1.8f, 29.0f);
	// Yaw: rotation around world up axis, in degrees.
	// yaw=0 faces +X; increasing yaw turns right, decreasing turns left.
	// Faces +X so spawning looks straight down the hall toward the far door.
	float camYaw = 0.0f;
	// Pitch: up-down, defined in degrees.
	// -90: looking down, +90: looking up
	float camPitch = -10.0f;
	// Vertical speed from gravity, in world units/second. Negative = falling.
	// Reset to 0 whenever the ground collision clamp catches us (i.e. we've landed).
	float camVerticalVelocity = 0.0f;

	// World floor height (top surface of the "floor" scene instance), cached once
	// after the scene loads. Used as a last-resort clamp while no-clipping through
	// walls, so falling under the map is never possible even with collisions off.
	float worldFloorY = 0.0f;

	// Owns the hand-authored collision geometry loaded from assets/scenes/colliders.json
	// and merges it with the colliders scene.json built.
	SceneColliders colliderSet;

	// Per-model BRDF parameters (specular color, roughness, F0, k), loaded from
	// assets/scenes/materials.json. See SceneMaterials.hpp.
	SceneMaterials materials;

	// The scene's light sources (direct, point and spot), loaded from
	// assets/scenes/lights.json. See SceneLights.hpp.
	SceneLights sceneLights;

	// Flat list of every collider gameplay collides against, taken from colliderSet
	// once the scene has loaded. Kept as its own member so the per-frame collision
	// loops below read a plain vector instead of going through the accessor.
	std::vector<Collider *> allColliders;

	// Debug/cheat toggles, isolated in a utility struct.
	// Not persisted across runs, reset to default values on launch.
	struct CheatFlags {
		// True: collisions (ground included), False: no-clip cheat
		bool collisionEnabled = true;
		// Debug overlay: continuously prints the camera's world-space
		// position/yaw in the bottom-right corner, meant as a live readout
		// for hand-placing scene.json objects (see notes.md). Unlike the
		// other flags, true isn't "the legit/no-cheat default": there's no
		// gameplay behavior to preserve here, so it defaults to off (hidden)
		// instead.
		bool showCoordinates = false;

		// Whether a hunting ghost touching the player ends the run. Off is the
		// cheat: the hunt still happens, the torches still turn, the ghosts
		// still come, you just can't lose to them -- which is what you want
		// while tuning any of it, since watching a chase play out is hard when
		// it's over the moment it starts working.
		bool ghostsCanCatch = true;

		// The two flame switches, read by flameBurning() -- which is what every
		// path that can show a flame goes through: its point light, its
		// billboard, its shadow-slot candidacy and its stare-at glare.
		//
		// They live HERE and not in SceneLights next to directEnabled/
		// spotEnabled because the flames never go through lights.json at all:
		// every torch light in this scene is appended straight into gubo from
		// updateUniformBuffer(), off the flame's own per-frame position (see
		// the "Torch flames' point lights" block there and lights.json's note
		// on why the static copies were removed). A switch in SceneLights
		// could only drop lights SceneLights owns, which is why the old
		// "Lanterns" row -- pointing at pointEnabled -- did nothing.
		//
		// Torches means the wall-mounted ones lighting the rooms; the candles
		// are a separate, much dimmer set of flames and stay lit. Both default
		// on, matching the authored scene.
		bool roomTorchesEnabled = true;
		// The torch in the player's hand: off hides the model too, not just
		// its flame and light, since a dark stick in front of the camera is
		// not what "no torch in hand" is meant to look like.
		bool handTorchEnabled = true;

		// Lighting debug views, all resolved into gubo.debugFlags in
		// updateUniformBuffer() and read by CookTorrance.frag. Same convention
		// as showCoordinates: these have no "legit" state to preserve, so each
		// one defaults to whatever leaves the picture as authored.
		//
		// The switches for the remaining light SOURCES (sun, spot, ambient)
		// aren't here: they live in SceneLights, next to the lights they drop,
		// and the HUD points straight at them.

		// Albedo only, nothing lit. Separates "this texture is dark" from
		// "no light is reaching this".
		bool unlit = false;
		// The shading normal as a color. The one view that shows normals
		// directly, which is what the flatNormals material flag exists for.
		bool showNormals = false;
		// Off suppresses the gold/blue aura on whatever the crosshair is aimed
		// at (see gazedInstance and ubo.glow in updateUniformBuffer()). Only
		// the visual cue goes: [E] still interacts with the same target and the
		// prompt still appears, so this is for looking at the scene's own
		// lighting on a door or a pickup without an aura sitting on top of it.
		// Defaults on, matching the authored gameplay.
		bool focusGlowEnabled = true;
		// Off kills the specular term (BRDF's k forced to 1), leaving pure
		// diffuse: tells a highlight apart from a genuinely bright surface.
		bool specularEnabled = true;
		// Off skips the tone map, so anything the tone map was pulling back
		// into range clips to flat white instead.
		bool toneMapEnabled = true;
		// Off forces every shadowFactor() to 1, i.e. renders as if no shadow
		// map existed, while still rendering the shadow passes themselves.
		// Diagnostic: it splits "this artifact comes from shadow sampling"
		// from "this artifact is in the geometry", which is otherwise hard to
		// tell apart by eye since both show up as flicker on a wall.
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

		// Geometry overlays (DebugLines.hpp), independent of the shading
		// debug views above: crosses/arrows at each active light's position,
		// and wireframe boxes at each torch's shadow-cube near/far clip
		// distance. Off by default, same reasoning as showCoordinates.
		bool showLightGizmos = false;
		bool showShadowFrustums = false;
		// Wireframe box around every collider the gameplay actually tests
		// against (SceneColliders::list(), i.e. scene.json's auto-fit boxes
		// plus colliders.json's authored ones), and the inclined quad of every
		// ramp. The one view that answers "is this wall solid where it looks
		// solid" without walking into it, and the fastest way to spot a box
		// that filled in an archway or a ramp that doesn't reach its landing.
		// Drawn from getExtents(), which is world-space axis-ALIGNED, so an
		// OOBB shows up as its fattened envelope rather than its true oriented
		// box -- the same envelope the ground pass itself uses (see the
		// collision loops in GameLogic()), so what's drawn is what's collided
		// against, not a prettier version of it.
		bool showColliders = false;
		// Shader-side, unlike the two above: recolors surfaces by incoming
		// light intensity instead of drawing extra geometry. See
		// LIGHT_DEBUG_HEATMAP in CookTorrance.frag.
		bool showLightHeatmap = false;
	} cheats;

	// Numeric tuning for the movement cheats above, isolated the same way but
	// kept as a separate struct since these aren't on/off switches: they're
	// "how strong", not "enabled or not". Also not persisted across runs.
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

	// A door leaf (its own instance, separate from the wall it's set into)
	// that swings open around a vertical hinge when the player interacts
	// with it nearby (E). Kept as a list rather than one hardcoded door,
	// since the brief calls for more interactables later (candles, secret
	// passages): adding one is one addDoor() call, no new per-object
	// plumbing.
	//
	// SM_Door_01 (unlike the wall-mounted SM_WallDoor_01 tried first, which
	// turned out to be a solid panel with no archway cut into it) is
	// authored with its local origin AT the hinge edge: its local bbox runs
	// Z 0.03..-2.43, i.e. the whole panel hangs to one side of Z=0. So the
	// instance's own authored transform (translate+eulerAngles in scene.json,
	// placing that origin at the door frame's hinge-side jamb) already IS
	// the closed-door hinge frame, and "open" is just one extra rotation
	// about world Y appended after it -- no separate hinge point to compute
	// or sandwich the rotation between.
	struct Door {
		std::string instanceId;
		Instance *inst = nullptr;
		glm::mat4 baseWm{1.0f};	// authored (closed, angle=0) world matrix
		glm::vec3 promptPos{0.0f};	// point used for the interact-range check (doorway centre, not the hinge)
		// The same point in the leaf's LOCAL frame, kept so it can be put
		// through the live Wm as well as through baseWm. promptPos is the
		// DOORWAY -- it is measured once, off the closed pose, and stays put
		// while the leaf swings. That is what it should be: a doorway does not
		// move. But it means a wide-open leaf is aimed at nowhere near it, and
		// a player standing in the passage looking straight at the panel could
		// not close it again. leafPos() below is the same offset carried by the
		// panel, and the two are tested as alternatives.
		glm::vec3 promptOffset{0.0f};
		// Where that offset has ended up this frame, i.e. the middle of the
		// leaf wherever it currently is. Equal to promptPos while the door is
		// shut, which is why the closed case needs no special handling.
		glm::vec3 leafPos() const {
			return inst == nullptr ? promptPos
								   : glm::vec3(inst->Wm * glm::vec4(promptOffset, 1.0f));
		}
		// How far the leaf travels when open. Only the MAGNITUDE is used at
		// runtime: the direction is decided per opening by swingSignAwayFrom()
		// below, so the authored sign here is just the fallback for a leaf
		// whose geometry can't be read (degenerate transform, player exactly
		// in the door's plane).
		float openAngleDeg = 100.0f;
		bool open = false;
		float angle = 0.0f;	// current animated angle, eases toward the target
		// Which way the leaf is currently swinging, as a sign on
		// |openAngleDeg|. Recomputed from the player's position only while
		// the door sits fully closed (see the toggle in GameLogic): flipping
		// it mid-swing would sweep the leaf back through the frame it is
		// hinged in.
		float swingSign = 1.0f;
		// Padlock. Empty lockKeyId = no lock at all, which is every door as
		// shipped: E just toggles it. A non-empty id means the door won't
		// budge until the player is carrying a key pickup whose keyId matches
		// (see Pickup::keyId and keyRing) -- matching by id rather than by
		// "any key" so a two-key level can't be opened in the wrong order.
		std::string lockKeyId;
		// Human-readable name for the locked prompt ("needs the <lockLabel>").
		// Defaults to lockKeyId in addDoor() when not given.
		std::string lockLabel;
		// Per-door wording for the three locked prompts, or empty to use the
		// generic padlock lines built from lockLabel (see the prompt block in
		// updateUniformBuffer). Empty on every chained door in the castle,
		// which is the point: "Locked - needs the iron key" is exactly right
		// for a door with a padlock hanging off it, and exactly wrong for a
		// bookcase, where the whole trick is that the player must work out that
		// the gap on the shelf IS a keyhole. A door that reads differently has
		// to be allowed to say so.
		std::string promptReady;	// carrying what it wants
		std::string promptMissing;	// not carrying it
		std::string promptBlocked;	// standing on the far side of the lock
		// A door that does not admit to being a door until it can be opened.
		// While this is set AND the lock still holds AND the player is not
		// carrying what it wants, the door is not offered as a gaze target at
		// all (see doorIsHidden): no focus aura, no prompt, nothing. It is
		// furniture.
		//
		// The red "you can't do this yet" aura every other lock wears is right
		// for a padlock -- the player can SEE the padlock, so the game telling
		// them it is shut is not telling them anything they didn't know. On a
		// bookcase it would be the opposite: the aura would be the game
		// pointing at the secret door and saying "here it is". The gap in the
		// row of books has to do that work on its own, and the reward for
		// noticing it is that the shelf lights up the moment you come back with
		// the book.
		bool secret = false;
		// Runtime state: true while the padlock still holds. Set from
		// lockKeyId at load and again in restartRun(), cleared for good (for
		// this run) the moment a matching key is spent on it -- keys are
		// one-shot, so a door that has been unlocked must never re-lock, or
		// the key would be gone with the door shut behind it.
		bool locked = false;
		// The visible padlock: SM_DoorChains_01 and SM_Padlock_01 (tools/
		// make_door_lock.py), or anything else registered with addLockProp().
		// Both are modelled in the LEAF's own local frame, so their world
		// matrix is the leaf's times `local` below -- which is why nothing
		// here reads their authored transform, and why they'd swing with the
		// door if a locked one ever could. Shown while locked, parked below
		// the map the moment the key is spent: that's the whole "the padlock
		// is off" effect.
		struct LockProp {
			Instance *inst;
			// Extra transform in the leaf's local frame. Identity for a door
			// approached from the side the models were built for; the half
			// turn below (see addLockProp's `flip`) for one approached from
			// the other.
			glm::mat4 local;
			// Inverts the whole thing: hardware that appears when the lock
			// COMES OFF instead of hardware that disappears with it. One user,
			// the secret bookcase (see addSlotProp): what pays for that door is
			// a book, and a book handed over does not vanish, it ends up in the
			// gap on the shelf the player was looking at. Same "an instance
			// riding the leaf's local frame" machinery either way, so this is a
			// flag rather than a second list.
			//
			// With it set, the prop is NOT parked below the map while the door
			// is locked -- it is simply left alone, because until it is spent
			// that instance is a Pickup lying on a table somewhere and the
			// pickup code owns its matrix. Taking it over from here would drag
			// the book underground the moment restartRun() re-locked the door.
			bool whenUnlocked = false;
		};
		std::vector<LockProp> lockProps;
		// How far past its own mesh a lock prop's collider reaches, on the face
		// the hardware hangs from: the stand-off that keeps the held torch out
		// of the chains. Raise if the flame still touches. See addLockProp().
		static constexpr float LOCK_PROP_KEEPOUT = 0.45f;
		// Which face of the leaf the hardware ended up on, as a sign on the
		// leaf's local X axis: +1 for the models as make_door_lock.py exports
		// them (FRONT_ON_PLUS_X), -1 for the half-turned copy. Set by
		// addLockProp() from its `flip`, and the reason a lock has a side at
		// all: the padlock is reachable only from the face it hangs on, so
		// the player standing behind the door meets a door that simply won't
		// move, key or no key.
		float lockFaceSign = 1.0f;
		// True when `p` stands on the padlock's face of the leaf. Measured
		// against baseWm rather than the live Wm because a locked door never
		// swings, so the closed pose is the only one this ever has to answer
		// for -- and it stays right even mid-animation on the frame the lock
		// comes off. XZ only: the sign shouldn't change with the player's
		// height (jumping, or the eye above a doorway's mid-plane).
		bool onLockSide(const glm::vec3 &p) const {
			glm::vec3 axis(baseWm[0].x, 0.0f, baseWm[0].z);	// leaf local +X, in world
			if(glm::length(axis) < 1e-6f) return true;	// degenerate: don't lock anyone out
			glm::vec3 d(p.x - promptPos.x, 0.0f, p.z - promptPos.z);
			return glm::dot(d, glm::normalize(axis)) * lockFaceSign > 0.0f;
		}
		// Sign on |openAngleDeg| that swings the leaf AWAY from a player at
		// `p` -- i.e. the door always opens outward with respect to whoever
		// is opening it, never into their face, whichever side they walked
		// up from.
		//
		// The closed leaf lies in the plane through its hinge with the leaf's
		// local +X as normal (the panel hangs along local Z; see the struct
		// comment), so the same normal answers both "which side is the
		// player on" and "which side did the panel end up on". Rather than
		// reasoning about the sign of a cross product through a transform
		// that may carry any rotation or mirroring, just rotate a point on
		// the panel the positive way and look at where it lands: if that is
		// the player's side, the negative way is the one wanted.
		//
		// XZ only, like onLockSide: a hinge is vertical, so the player's
		// height has no say in which way the door goes.
		float swingSignAwayFrom(const glm::vec3 &p) const {
			float authored = openAngleDeg < 0.0f ? -1.0f : 1.0f;
			glm::vec3 n(baseWm[0].x, 0.0f, baseWm[0].z);	// leaf local +X, in world
			if(glm::length(n) < 1e-6f) return authored;	// degenerate: keep the authored swing
			n = glm::normalize(n);

			glm::vec3 d(p.x - promptPos.x, 0.0f, p.z - promptPos.z);
			float playerSide = glm::dot(d, n);
			// Standing in the leaf's own plane (in the doorway itself): there
			// is no "away" to pick, so don't churn -- keep what the scene asked
			// for.
			if(std::abs(playerSide) < 1e-4f) return authored;

			glm::vec3 hinge(baseWm[3]);
			glm::vec4 tip = baseWm
						  * glm::rotate(glm::mat4(1.0f), glm::radians(std::abs(openAngleDeg)), glm::vec3(0.0f, 1.0f, 0.0f))
						  * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f);	// a point down the panel, swung the positive way
			float tipSide = glm::dot(glm::vec3(tip) - hinge, n);
			if(std::abs(tipSide) < 1e-6f) return authored;	// swings flat: nothing to choose between
			return tipSide * playerSide > 0.0f ? -1.0f : 1.0f;
		}
	};
	std::vector<Door> doors;

	// How close (world units, measured to the doorway centre) the player has
	// to be before a door's prompt appears and E does anything.
	static constexpr float DOOR_INTERACT_RADIUS = 6.0f;
	// Degrees/second the door animates open/closed at.
	static constexpr float DOOR_OPEN_SPEED = 120.0f;
	// How far away (world units) a door can still be picked as a gaze
	// candidate. Larger than DOOR_INTERACT_RADIUS, which still gates whether
	// it's actually close enough to interact with once aimed at -- see
	// findGazedDoor() and the gaze+proximity AND in GameLogic().
	static constexpr float DOOR_LOOK_DISTANCE = 9.0f;
	// Half-width (world units) of the doorway used to turn promptPos into an
	// angular aiming tolerance -- wide, since a doorway is a big target.
	static constexpr float DOOR_AIM_RADIUS = 1.2f;

	// Edge-detection for the interact key, same reason as jumpKeyWasPressed:
	// holding E shouldn't toggle the door every frame.
	bool interactKeyWasPressed = false;
	// Index into `doors` of whichever one is currently in range, or -1. Set
	// each frame in GameLogic(), read by updateUniformBuffer() to show/hide
	// the "[E] Interact" prompt.
	int nearbyDoor = -1;

	// A world object collected with [E], same list-of-interactables reasoning
	// as Door (the brief calls for several: the key here, more likely later).
	// Not strictly one-way: at least the key can be dropped again (G), which
	// just flips `collected` back and re-parks the instance in the world, so
	// the same entry keeps tracking it either way.
	struct Pickup {
		std::string instanceId;
		Instance *inst = nullptr;
		glm::vec3 worldPos{0.0f};	// measured once at load, used for the in-range check
		bool collected = false;
		// The authored pose, kept so restartRun() can put a picked-up or
		// dropped item back exactly where scene.json placed it. worldPos can't
		// serve: dropping the key overwrites it.
		glm::mat4 spawnWm{1.0f};
		glm::vec3 spawnPos{0.0f};
		// Non-empty => this pickup IS a key, and it opens every Door whose
		// lockKeyId is this same string (plus the exit, see exitKeyId). Empty
		// => an ordinary item that just gets carried.
		std::string keyId;
		// Spent on a lock and gone for the rest of the run. Distinct from
		// `collected`: a collected key is in hand and can still be dropped
		// (G) or spent, a consumed one is off the board entirely and only
		// restartRun() brings it back. Keys are usa e getta, so this is the
		// flag that enforces it.
		bool consumed = false;
		// Uniform world scale, read once at load out of the authored matrix
		// (column 0's length) so scene.json's "scale" stays the only place
		// that number is written. Per-pickup rather than one key-specific
		// constant: with several keys in a level they need not be one size,
		// and the held/dropped poses both rebuild the matrix from scratch and
		// so both need it back.
		float worldScale = 1.0f;
		// Euler angles (degrees, X then Y then Z) that orient this item in the
		// player's hand. Per-pickup and not one shared constant, because the
		// grip is a fact about the MESH's own axes rather than about hands: the
		// key's long axis is its local Z and has to be swung upright, the book
		// lies flat in its own frame and has to be tilted up to be read. Filled
		// by addPickup(), which defaults it to HAND_KEY_TILT_DEG -- declared
		// further down with the rest of the held pose, hence the plain zero
		// here rather than a default that would have to name it.
		glm::vec3 handTiltDeg{0.0f};
		// Where the item hangs relative to the eye, in camera space. Also
		// per-pickup, and for a reason that is easy to miss: this positions the
		// mesh's ORIGIN, and two meshes can carry their origin in quite
		// different places. The key's sits near one end, so the key mostly
		// hangs upward from it; the book's sits at the middle of its page
		// height, so the book straddles it and rides visibly lower from the
		// same offset. Same default treatment as the tilt: HAND_KEY_OFFSET,
		// filled by addPickup().
		glm::vec3 handOffset{0.0f};
	};
	std::vector<Pickup> pickups;
	// The player's key ring: indices into `pickups`, in the order collected.
	// A vector and not a set of ids because two keys can share an id (a level
	// with two identical padlocks and two identical keys is legal) and because
	// the ring has to remember WHICH instance each key was, to drop it or park
	// it on use. keyRing.back() is the one drawn in the hand.
	std::vector<int> keyRing;
	// Measured in 3D (unlike DOOR_INTERACT_RADIUS's XZ-only check): a pickup
	// can sit at table height, well above the player's feet.
	static constexpr float PICKUP_INTERACT_RADIUS = 4.0f;
	// Index into `pickups` of whichever one is currently in range, or -1.
	// Mirrors nearbyDoor; checked first in GameLogic() since grabbing
	// something should win over interacting with whatever's behind it.
	int nearbyPickup = -1;
	// How far away (world units) a pickup can still be picked as a gaze
	// candidate. See DOOR_LOOK_DISTANCE for the same reasoning.
	static constexpr float PICKUP_LOOK_DISTANCE = 6.0f;
	// Half-width (world units) of a pickup used for its angular aiming
	// tolerance -- tight, since pickups are small props, not doorways.
	static constexpr float PICKUP_AIM_RADIUS = 0.35f;

	// Base half-angle (degrees) of the aiming cone, before an object's own
	// aim radius widens it at range. Shared by doors and pickups.
	static constexpr float GAZE_CONE_DEG = 7.0f;

	// The single Instance* (a door's or a pickup's) the crosshair is
	// currently resting on within interact range, or nullptr. Resolved once
	// per frame in GameLogic() right after nearbyDoor/nearbyPickup, and read
	// by updateUniformBuffer()'s per-instance UBO loop to set ubo.glow.
	Instance *gazedInstance = nullptr;
	// True when gazedInstance is aimed at but [E] would currently do
	// nothing to it -- right now that's just a locked door with no matching
	// key on hand (or approached from the wrong side), but kept general
	// (rather than named e.g. doorLocked) so a future non-door interactable
	// with its own disabled state can flip it too. Flips ubo.glow's sign in
	// updateUniformBuffer(), which CookTorrance.frag reads to swap the aura
	// from gold to red -- see the color comment there.
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

	// True if `front` (the camera's normalized forward vector) is aimed
	// closely enough at the point `target` to select it: within lookDist,
	// and within a cone whose half-angle is GAZE_CONE_DEG widened by the
	// angular size aimRadius subtends at the target's distance (so a wide
	// door forgives worse aim than a small pickup, without needing a real
	// bounding box for either -- same "point + radius" trick already used
	// for interact ranges). smallerCosAngleWins lets a caller comparing
	// multiple candidates track the best (smallest-angle) one via cosAngle.
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

	// True for a secret door the player has no business seeing yet. Gating it
	// inside findGazedDoor rather than at each consumer is what stops the aura,
	// the prompt and the E key from ever disagreeing about whether the thing is
	// there: all three read the one nearbyDoor this decides.
	bool doorIsHidden(const Door &d) const {
		return d.secret && d.locked && findKeyInRing(d.lockKeyId) < 0;
	}

	// Index into `doors` the player is currently aiming at within look
	// range, or -1. Does NOT check DOOR_INTERACT_RADIUS -- that's re-checked
	// by the caller once a gaze candidate is found, keeping "can be
	// targeted at all" (this) and "close enough to actually interact"
	// (DOOR_INTERACT_RADIUS) as two separately named, separately tuned gates.
	int findGazedDoor(const glm::vec3 &front) const {
		int best = -1;
		float bestCos = -1.0f;
		for(int i = 0; i < (int)doors.size(); i++) {
			if(doorIsHidden(doors[i])) continue;
			// Two points, not one: the doorway (fixed, correct while the leaf
			// is shut or nearly so) and the leaf itself in its live pose. An
			// open door swings right out of its own doorway, so aiming at the
			// panel -- which is the obvious thing to do when you want to shut
			// it, and the ONLY thing in sight from inside a passage you have
			// just walked through -- used to select nothing at all. Whichever
			// of the two is aimed at more squarely wins.
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

	// Whichever key is currently drawn in the player's hand: the most
	// recently collected one, or -1 with an empty ring. Index into `pickups`,
	// never a copy of the Instance -- picking a key up doesn't spawn a second
	// model, it just stops drawing the world one where the table left it and
	// starts drawing it off the camera instead (see the held-key block in
	// GameLogic()).
	int heldKeyIdx() const {
		return keyRing.empty() ? -1 : keyRing.back();
	}
	// Position IN keyRing (not in `pickups`) of a carried key matching `id`,
	// or -1. An empty `id` means "any key at all", which is what the exit
	// uses when gameplay.json doesn't name one -- that's the pre-lock
	// behaviour of requiresKey, kept working unchanged.
	// Searches back to front so the key in hand is the one spent first: with
	// two interchangeable keys on the ring, spending the one you're visibly
	// holding is the only reading that isn't a surprise.
	int findKeyInRing(const std::string &id) const {
		for(int slot = (int)keyRing.size() - 1; slot >= 0; slot--) {
			if(id.empty() || pickups[keyRing[slot]].keyId == id) return slot;
		}
		return -1;
	}
	// Spend a carried key: off the ring, off the board, permanently for this
	// run. `slot` is an index INTO keyRing (what findKeyInRing returns), not
	// into pickups. Parking the instance below the map is how everything else
	// here hides a mesh (Starter draws every instance every frame, there's no
	// per-instance visibility flag), and it's what stops the held-key block
	// from drawing this one in the hand from the next frame on.
	void consumeKey(int slot) {
		if(slot < 0 || slot >= (int)keyRing.size()) return;
		int idx = keyRing[slot];
		Pickup &p = pickups[idx];
		p.consumed = true;
		p.collected = true;
		keyRing.erase(keyRing.begin() + slot);
		// Off the ring immediately (the lock is spent the moment E is pressed)
		// but NOT parked below the map yet: it sinks out of frame first, the
		// mirror image of the pick-up rise. The lowering block in GameLogic()
		// owns the instance until the animation ends and does the parking
		// there. Nothing can pick it back up meanwhile -- `consumed` already
		// took it out of the in-range scan.
		// If a key was already sinking, it gets parked now rather than left
		// mid-air: only one can be animating, and the newer one wins.
		if(keyLowerIdx >= 0) {
			pickups[keyLowerIdx].inst->Wm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
		}
		keyLowerIdx = idx;
		keyLowerElapsed = 0.0f;
	}
	// Put a carried key back in the world: off the ring, lying flat in front
	// of the player and facing the same way they are, so walking up to it
	// again shows the pick-up prompt like any other pickup. `slot` is an index
	// INTO keyRing, same as consumeKey. `fwdDist` is how far ahead of the feet
	// it lands: the manual drop (G) puts it at arm's length, the automatic one
	// that frees the hand for a newly grabbed key puts it closer so it doesn't
	// sail across the room.
	// Re-parks the SAME instance the held-key block was drawing in the hand,
	// same one-instance reasoning as pickup -- and doing it here rather than
	// inline in GameLogic() is what lets both callers agree on the pose.
	void dropKeyFromRing(int slot, const glm::vec3 &camPos, const glm::vec3 &front, float fwdDist) {
		if(slot < 0 || slot >= (int)keyRing.size()) return;
		Pickup &p = pickups[keyRing[slot]];
		const float EYE_HEIGHT = 1.8f;	// same eye height used throughout GameLogic()

		glm::vec2 faceDir(front.x, front.z);
		if(glm::length(faceDir) > 0.0001f) faceDir = glm::normalize(faceDir);
		else faceDir = glm::vec2(0.0f, 1.0f);
		// 0.03 above the feet: dropped exactly at floor height would
		// coincide with the floor mesh and z-fight (see notes.md on
		// the dungeon meshes' coplanar faces).
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
	// Held pose. Negative X puts it in the LEFT hand (mirrors
	// HAND_TORCH_OFFSET's +0.5, which is the right); the torch already owns
	// the right hand and a torch-carrying explorer would hold a found key in
	// the other one. Y raised from an earlier -0.45, which sat low enough to
	// be out of frame entirely. Y tilt is mirrored the same way as X for a
	// natural-looking grip -- unverified without a render, tune alongside
	// the offset if it looks wrong.
	static constexpr glm::vec3 HAND_KEY_OFFSET = glm::vec3(-0.40f, -0.4f, -0.9f);
	// X = 90: the key's long axis is local Z (see the raw-mesh-scale comment
	// below), and rotating 90 deg about X swings local Z onto world
	// Y -- i.e. upright, tip up. The mesh's Z range is asymmetric
	// (-90.84..20.07, in raw units), and the longer, more-negative side is
	// what maps to +Y at this angle, which is the assumption that it's the
	// bit/blade end rather than the bow/handle. If the render shows it
	// tip-down instead, negate this to -90.
	// It is the DEFAULT rather than the only one: each pickup carries its own
	// (Pickup::handTiltDeg, filled by addPickup), because the numbers above
	// describe the key mesh's axes and nothing else's.
	static constexpr glm::vec3 HAND_KEY_TILT_DEG = glm::vec3(90.0f, -20.0f, 0.0f);
	// The held item's orientation: the item's own tilt, with the walk's roll
	// added to the last axis so the hand sways with the step. Shared by the
	// rise and the sink so the two animations can't drift apart -- they are
	// meant to read as one motion played in both directions.
	static glm::mat4 handGrip(const glm::vec3 &tiltDeg, float bobRollDeg) {
		return glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.x), glm::vec3(1.0f, 0.0f, 0.0f))
			 * glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.y), glm::vec3(0.0f, 1.0f, 0.0f))
			 * glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg.z + bobRollDeg), glm::vec3(0.0f, 0.0f, 1.0f));
	}

	// ---- Held-item wall tuck -------------------------------------------------
	//
	// Both hands are welded rigidly to the camera (camWm, built in GameLogic),
	// and both hold their item roughly a unit out from the eye -- far past the
	// PLAYER_RADIUS of 0.3 the body itself is pushed out of walls by. Walk up to
	// a wall and the far end of whatever you're carrying is simply inside it.
	//
	// For the key that is merely ugly. For the torch it is a lighting bug: its
	// point light AND its cube shadow both sit on the flame anchor at the torch's
	// head (see updateHandTorchShadow), so a head that crosses a wall takes the
	// light through with it. The wall in front of you loses its N.L and goes
	// black at the exact moment you are closest to the flame, and the room on the
	// far side gets lit through solid geometry. The stock FPS fix -- draw the
	// viewmodel in a second pass over cleared depth -- would repair the pixels
	// and none of that, because the light doesn't live in the depth buffer. The
	// item itself has to move.
	//
	// It moves the way a person moves it: not yanked back into the face, which
	// drags the light onto the eye and flattens everything ahead of it, but
	// lowered and turned across the body, with only a modest pull. Most of the
	// tuck is rotation, and rotation costs the light almost no travel.

	// The tuck is driven GEOMETRICALLY, not by blending toward an authored
	// "tucked" pose. A fixed pose was the first attempt and it doesn't hold up:
	// whatever pull you bake into it is the same pull whether the wall is at
	// arm's length or at the wrist, so anything closer than that one distance
	// still ends up inside. What the probe measures is how far out the item may
	// reach RIGHT NOW, and the pose is then built to reach exactly that far --
	// so the fix scales with the obstacle instead of hoping one number covers
	// every case. Looking down at a table is the case that proves it: the
	// surface is barely half an arm away, far closer than any pose constant
	// tuned against a wall would ever have retracted.
	//
	// Everything below is expressed as a REACH FRACTION: 1 fully extended, down
	// to HAND_TUCK_MIN_REACH pressed in against the chest.

	// How far the item's own body reaches past the grip point, in camera-space
	// units, so the probe stops at the item's leading edge rather than at the
	// hand. Rough on purpose -- HAND_TUCK_SKIN below dwarfs the error.
	static constexpr float HAND_TUCK_TORCH_PAD = 0.30f;
	static constexpr float HAND_TUCK_KEY_PAD = 0.15f;

	// Colliders are grown by this much for the probe (and ONLY for the probe --
	// the body still walks into the real boxes). Two jobs, one constant:
	//
	//  - The collision geometry is a deliberately coarse shell. Walls are thin
	//    authored plates (see colliders.json: 1.242 through the wall and nothing
	//    else), so every jamb, pilaster and buttress that stands proud of that
	//    plate is invisible to any probe that trusts the boxes -- the exact
	//    "wall pieces with no collider" the item was still entering. The skin
	//    stops the item at a stand-off from the collision plane instead of at
	//    it, which covers protrusions up to roughly this deep without anyone
	//    having to author a box per moulding.
	//  - It doubles as the safety margin for the drop and rotation the pose adds
	//    AFTER the radial retraction, which push the item slightly off the line
	//    the probe measured along.
	//
	// It cannot simply be made large: it is a stand-off from every surface, and
	// past a point the item folds away in rooms that are merely narrow. This is
	// the number to raise if something still clips, and to lower if the item
	// tucks in open space.
	static constexpr float HAND_TUCK_SKIN = 0.30f;

	// The reach the item is never retracted past, even pressed into a corner.
	// This is a LIGHTING limit, not a comfort one: the torch's point light rides
	// on its head, and a light at the eye lights every surface head-on, flattens
	// all the shading and blows out whatever is closest. Better a few visible
	// centimetres of a torch in a wall than the whole room going flat.
	//
	// It doubles as the start of the probe. Everything nearer than this is taken
	// as clear without testing, which is not a shortcut but a correction: the
	// skin above inflates walls by 0.30, PLAYER_RADIUS only holds the body 0.30
	// off them, so a sample right at the eye reads "blocked" whenever you brush
	// a wall -- and the item would fold up every time you squeezed past one,
	// facing anywhere. The body's own clearance already guarantees this stretch.
	static constexpr float HAND_TUCK_MIN_REACH = 0.35f;

	// Samples across the probed stretch. Ten over ~1 unit resolves to ~0.1,
	// which is the granularity the retraction snaps to -- fine enough that the
	// smoothing below hides the steps completely.
	static constexpr int HAND_TUCK_SAMPLES = 10;

	// Tuck in fast, come back out slow. One shared time constant can't do both:
	// slow enough not to strobe the torchlight on a collider boundary is slow
	// enough to let the item dip into the wall when you walk at it, so the two
	// directions get their own.
	static constexpr float HAND_TUCK_TAU_IN = 0.05f;
	static constexpr float HAND_TUCK_TAU_OUT = 0.18f;

	// Retracting radially -- scaling the whole offset toward the eye -- shortens
	// its Y along with everything else, so the item drifts UP toward the middle
	// of the screen as it comes in, which reads like it is being raised to the
	// face rather than tucked away. This puts it back down. It is sized to
	// roughly cancel that lift at full retraction, not to be a motion of its own.
	static constexpr float HAND_TUCK_DROP = 0.28f;
	// Extra grip rotation on top of the item's own tilt: nose down (X) and swung
	// across the body (Y). This is the part that sells the tuck as a gesture,
	// and it is free -- rotating about the grip barely moves the torch's light
	// at all, unlike any amount of translation. Signs are the same kind of guess
	// HAND_KEY_TILT_DEG's are (they depend on which way each mesh's local axes
	// run), so if an item rotates the wrong way on a render, negate the offending
	// component rather than reworking the motion.
	static constexpr glm::vec3 HAND_TUCK_TILT_DEG = glm::vec3(40.0f, 30.0f, 0.0f);

	// Live reach fraction per hand, 1 (extended) down to HAND_TUCK_MIN_REACH,
	// advanced by advanceReach() every frame in GameLogic. Two independent
	// values, not one: the hands are on opposite sides of the eye and a wall to
	// your right reaches the torch long before it reaches the key.
	float handTorchReach = 1.0f;
	float handKeyReach = 1.0f;

	// ghostPointBlocked's test with a stand-off (see HAND_TUCK_SKIN). Separate
	// rather than a defaulted argument on that one: it answers a sightline
	// question where growing the world would be plainly wrong, and the two
	// should not be able to drift into each other.
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

	// How far out the item held at `camOffset` may reach this frame, as a
	// fraction of its authored offset: 1 nothing in the way, HAND_TUCK_MIN_REACH
	// something right at the hand.
	//
	// Marches outward from HAND_TUCK_MIN_REACH along the eye->item line and
	// stops at the first solid sample; the last clear one is where the item's
	// leading edge has to end up. Because the answer is a distance rather than a
	// severity, the caller can place the item AT it and know it is out -- which
	// is the whole difference from the fixed-pose version this replaced.
	float handFreeReach(const glm::mat4 &camWm, const glm::vec3 &camOffset, float pad) const {
		float reach = glm::length(camOffset);
		if(reach < 1e-4f) {
			return 1.0f;
		}
		// The grip point pushed out to the item's leading edge, still in camera
		// space: same direction from the eye, `pad` further along it. Fractions
		// below are of THIS, so 1 leaves the edge exactly where it was authored.
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

	// One asymmetric smoothing step of a reach fraction toward `target`.
	// Exponential like every other easing here (walkBobBlend, eyeStepOffset): it
	// cannot overshoot, and it needs no "am I still tucking" state. Note the
	// comparison is inverted against the tau names -- tucking IN means the reach
	// going DOWN.
	static void advanceReach(float &state, float target, float deltaT) {
		float tau = (target < state) ? HAND_TUCK_TAU_IN : HAND_TUCK_TAU_OUT;
		state += (target - state) * (1.0f - std::exp(-deltaT / tau));
	}

	// Builds the tucked pose for a reach fraction, in place, so callers keep
	// composing the walk bob on top exactly as before -- the tuck moves the POSE
	// the bob swings around, it doesn't fight it.
	//
	// The retraction is radial, along the same line handFreeReach measured, so
	// the item lands exactly where that said it could. Everything after is the
	// gesture: a drop to cancel the lift radial scaling causes, and rotation.
	//
	// Which way "inward" points is read off the offset's own sign rather than
	// passed in: the torch is the right hand (+X) and the key the left (-X), and
	// deriving it means neither hand carries a mirrored copy of these constants
	// that could drift from the other.
	static void applyTuck(float reachFrac, glm::vec3 &offset, glm::vec3 &tiltDeg) {
		if(reachFrac >= 1.0f) {
			return;
		}
		float inward = (offset.x >= 0.0f) ? -1.0f : 1.0f;
		// How far into the tuck we are, 0..1, independent of where the floor on
		// reach happens to sit -- so retuning HAND_TUCK_MIN_REACH doesn't
		// silently rescale the drop and the rotation with it.
		float tuck = (1.0f - reachFrac) / (1.0f - HAND_TUCK_MIN_REACH);

		offset *= reachFrac;
		offset.y -= HAND_TUCK_DROP * tuck;
		tiltDeg.x += HAND_TUCK_TILT_DEG.x * tuck;
		tiltDeg.y += inward * HAND_TUCK_TILT_DEG.y * tuck;
	}
	// Pick-up animation: the key doesn't snap into the hand, it rises into
	// frame from below over KEY_RAISE_DURATION seconds. Purely a translation
	// along camera-local Y added to HAND_KEY_OFFSET, so it composes with the
	// walk bob without either one knowing about the other.
	// Short: this is a flourish, not a cutscene -- long enough to read as
	// motion, short enough that a player grabbing a key mid-run doesn't wait.
	static constexpr float KEY_RAISE_DURATION = 0.35f;
	// How far below the final hand pose the key starts, in camera-local
	// units. Roughly out of the bottom of the frame at the default FOV, which
	// is what makes it read as "raised into view" rather than "nudged".
	static constexpr float KEY_RAISE_DROP = 0.8f;
	// Seconds elapsed since the key currently in hand was picked up, or
	// >= KEY_RAISE_DURATION once the rise is over (it just keeps counting, the
	// eased factor saturates at 1). Reset on every pickup, including the
	// swap that drops the old key, so each new key plays the animation.
	float keyRaiseElapsed = KEY_RAISE_DURATION;
	// Index into `pickups` of a key spent on a lock that is still sinking out
	// of frame, or -1. It is already off the ring (consumeKey erases it there
	// and then), so heldKeyIdx() no longer names it and the held-key block
	// below won't touch it -- this is the only thing still drawing it, and it
	// parks the instance below the map when the fall finishes.
	int keyLowerIdx = -1;
	float keyLowerElapsed = 0.0f;

	// No separate raw-mesh fix or hand-only size lives here: the held pose
	// reads the same size as the table/dropped one (Pickup::worldScale, read
	// once in addPickup() out of that key's own authored matrix -- see there). If a
	// held key ever needs to look bigger/smaller than the world one, that's
	// a multiplier to reintroduce here, not before.

	// The torch held in the player's right hand. A normal scene instance
	// (handTorch in scene.json) whose world matrix is rebuilt every frame
	// from the camera's position and basis vectors, so it follows the view
	// like a first-person weapon model. Null if the instance isn't found.
	Instance *handTorchInst = nullptr;
	// Index into `torchFlames` of the held torch's flame, or -1 if the
	// instance wasn't found. The held torch is authored UNLIT (see the
	// addTorchFlame call in localInit): the player lights it by holding it up
	// to a burning wall torch and pressing [E], the mirror of the candle
	// interaction. Everything that asks "is there fire in hand" goes through
	// hasBurningTorch(), which reads this flame's `burning` flag.
	int handFlameIdx = -1;
	// True once the player has picked the torch up off the floor. Until
	// then handTorchInst sits at its authored scene.json pose as a pickup
	// target (findGazedHandTorch / nearbyHandTorch), no flame or hand model
	// is drawn, and hasBurningTorch() is false so no candle can be lit.
	// restartRun() clears it -- every run begins with the torch back on the
	// ground, the first thing the player earns before the light.
	bool handTorchCollected = false;
	// The torch's authored floor pose and position, captured in localInit
	// before the per-frame held-torch code can overwrite Wm. Used for the
	// gaze/proximity check while it's on the ground and to put it back
	// there (updateUniformBuffer's not-collected branch, restartRun).
	glm::mat4 handTorchSpawnWm{1.0f};
	glm::vec3 handTorchWorldPos{0.0f};
	// True when the crosshair is on the floor torch within reach this
	// frame. Mirrors nearbyPickup; set every frame in GameLogic().
	bool nearbyHandTorch = false;
	// Torch position relative to the eye, in camera space (right, up,
	// -front). Low and close enough that the handle crops off the bottom
	// of the screen, so only the torch itself is visible, like a held
	// weapon in a first-person view.
	static constexpr glm::vec3 HAND_TORCH_OFFSET = glm::vec3(0.5f, -0.35f, -1.1f);
	// Extra tilt on top of the model's own orientation, so it looks gripped
	// rather than floating dead level.
	static constexpr glm::vec3 HAND_TORCH_TILT_DEG = glm::vec3(-15.0f, 20.0f, 0.0f);
	// Uniform scale: SM_Torch_01 is sized for a wall mount, shrunk to look
	// right at arm's length.
	static constexpr float HAND_TORCH_SCALE = 0.35f;

	// The flame at a torch's head. See custom/Flame.hpp: camera-facing
	// billboard layers shaded by a procedural fire field, drawn as one more
	// object inside the main pass. One Flame instance drives every torch in
	// the scene (Flame::spawn), held one included -- that's what the reusable
	// design was for.
	Flame flame;

	// The daylight standing outside the exit door. See custom/ExitGlow.hpp:
	// one overbright quad on the open ground past the doorway, drawn in the
	// main pass like the flames, whose glare is entirely the work of the
	// bloom chain downstream. Driven by exitDoorIndex/EXIT_GLOW_* below.
	ExitGlow exitGlow;

	// Line renderer for the cheat-menu-gated debug overlays: crosses/arrows at
	// each active light, wireframe boxes at each torch's shadow-cube near/far
	// clip distance, and the collider wireframes. What each overlay draws is
	// decided in updateUniformBuffer(), not here. See custom/DebugLines.hpp for
	// why it draws via vertex pulling instead of a vertex buffer.
	DebugLines debugLines;

	// One entry per torch that got a flame, filled once in localInit() (see
	// addTorchFlame there) and walked every frame in updateUniformBuffer()
	// to update that flame's transform, its sparks and its point light.
	// `anchor` is in the TORCH MODEL's own local space (i.e. before whatever
	// instance transform places it in the world); `inst->Wm * vec4(anchor,1)`
	// gives the flame's world POSITION.
	//
	// The rest is this torch's live fire state, simulated on the CPU. It is
	// deliberately computed HERE rather than in the shader, because three
	// separate consumers have to agree on it: the flame billboards, the
	// sparks, and the point light the torch casts into the scene. A flame
	// that gutters while the light it throws on the wall holds perfectly
	// steady reads as broken, and the light is by far the more convincing
	// half of the effect -- so there is exactly one signal, produced once,
	// and everything downstream reads it.
	struct TorchFlame {
		Instance *inst;
		int flameId;
		glm::vec3 anchor;

		// True only for the held torch. Every wall-mounted torch stays put in
		// world space, so its flame should billboard the ordinary CYLINDRICAL
		// way: spin about world up to face the camera, but stay upright
		// however the camera pitches, same as a real flame would.
		//
		// The held torch breaks that assumption: it is not a world object the
		// camera walks past, it is rigidly anchored in CAMERA space
		// (HAND_TORCH_OFFSET/HAND_TORCH_TILT_DEG, rebuilt from the camera's
		// own front/right/up every frame in GameLogic) and visibly tilts as
		// the camera pitches, exactly like a weapon viewmodel. A
		// world-vertical flame sitting on top of a shaft that tilts with your
		// view swings out of alignment with that shaft -- at enough pitch it
		// reads as sticking out sideways from the torch instead of burning at
		// its tip. Its flame needs to tilt WITH the torch, i.e. ride the
		// camera's actual up/right axes (pitch included) instead of the
		// flattened, pitch-independent ones the wall torches use.
		bool heldByCamera = false;

		// Phase offset into the noise field, so no two torches flicker or
		// gutter together.
		float phase = 0.0f;

		// BRIGHTNESS flicker/guttering envelope, ~0.30 (mid-gutter) to ~1.40
		// (a flare). Scales the flame's brightness, its spark output/rate, and
		// the point light's colour and reach -- but not the flame's height,
		// see heightScale. Chased toward its noise-driven target by a
		// critically-damped spring (intensityVel below) instead of being
		// assigned raw: the raw signal is a fresh noise sample every frame,
		// and slamming the whole flame to it at once would visibly "jump". A
		// spring is continuous in value AND slope, so brightness glides, yet
		// still ducks through a gutter in a couple tenths of a second.
		float intensity = 1.0f;
		float intensityVel = 0.0f;

		// HEIGHT envelope, ~0.78..1.09. Same underlying signal as intensity,
		// but compressed (a flame's height varies far less than its light
		// output) and low-passed much harder (FLAME_HEIGHT_TAU): light
		// responds to combustion instantly, the fuel column's height follows
		// it late. Kept as its own signal rather than sharing intensity's,
		// so the flame's height glides instead of teleporting at flicker rate.
		float heightScale = 1.0f;

		// This torch's own smoothed stare-at factor, 0..1 (see the glare
		// block in updateUniformBuffer): how squarely and closely the camera
		// is looking at this flame, used to overdrive its HDR output.
		float glare = 0.0f;

		// Where the anchor was last frame, and a low-passed velocity derived
		// from it. A flame is dragged by the air it moves through, so it
		// leans AGAINST its own motion -- lagging when you turn, streaming
		// backwards when you sprint, overshooting when you stop.
		glm::vec3 prevPos = glm::vec3(0.0f);
		glm::vec3 smoothedVel = glm::vec3(0.0f);
		bool velPrimed = false;

		// This torch's fire color: drives both the flame's own gradient
		// (Flame.frag hue-rotates onto it, see recolorStop() there) and the
		// point light it casts (see the light loop below) -- one signal for
		// both, same reasoning as `intensity`. Defaults to the realistic
		// torch orange, so any addTorchFlame() call that doesn't pass a
		// color is untouched by this.
		//
		// Overwritten every frame by the hunt cycle (see the flame envelope
		// block in updateUniformBuffer): during a hunt this is `baseColor`
		// dragged toward violet. Everything downstream -- the light, the
		// billboard, the bloom -- reads this one field, so the colour change
		// needed no plumbing beyond the mix itself.
		glm::vec3 color = TORCH_LIGHT_COLOR;

		// What this torch burns when nothing is hunting: the colour authored in
		// flames.json (or the default orange), captured once at spawn. `color`
		// can't double as its own base, because mixing a value toward violet
		// and storing the result back over the value you mixed FROM converges
		// on violet and never comes back.
		glm::vec3 baseColor = TORCH_LIGHT_COLOR;

		// Multiplies FLAME_HEIGHT/FLAME_HALF_WIDTH on top of the instance's
		// own uniform scale (see instScale below). 1.0 for every torch; the
		// candles pass a smaller value so their flame reads as a candle
		// flame rather than a torch flame that merely rode the candle
		// model's own (already small) instance scale.
		float sizeScale = 1.0f;

		// Multiplies the point light's colour only (see the light-append
		// loop below) -- independent of sizeScale, since a small flame
		// isn't automatically a dim one. 1.0 for every torch; the candles
		// pass a small value so they read as the faint, local light a
		// candle actually casts instead of a torch-strength light that
		// merely came out of a smaller flame.
		float lightScale = 1.0f;

		// True for a candle flame, false for a torch (the held torch
		// included). No third kind exists, so this is the whole
		// distinction the debug HUD's separate "Torch Shadows"/"Candle
		// Shadows" toggles need -- everything else about a flame (mesh,
		// light math, dynamic shadow pool membership) already treats the
		// two identically. Set from flames.json's per-model "isCandle"
		// (see FlameDef below); defaults to false so a model that omits
		// it is a torch, matching every model but dungeonCandle.
		bool isCandle = false;

		// Is this flame actually burning? Read through flameBurning() (see there),
		// so switching it off takes the point light, the billboard, the sparks,
		// the shadow-slot candidacy and the stare-at glare with it in one go --
		// nothing here is "spawned" or "destroyed" at runtime, since Flame.hpp
		// hands out its instance ids once at init (flame.spawn in
		// addTorchFlame) and has no way to give one back.
		//
		// False for a candle authored unlit in flames.json ("burning": false),
		// which is every candle as shipped: lighting one with [E] off the held
		// torch is the interaction (see nearbyCandle in GameLogic). Also false
		// for the held torch itself, which is spawned unlit and lit from a
		// burning wall torch (see nearbyWallTorch in GameLogic). Wall torches
		// are authored burning and no code ever puts them out.
		bool burning = true;
		// What `burning` was authored as, so restartRun() can blow the candles the
		// player lit back out -- a run that started dark has to start dark
		// again, same reasoning as re-locking the doors there.
		bool spawnBurning = true;

		// This flame's anchor in WORLD space, computed once at spawn. Only
		// meaningful for a static flame (!heldByCamera): the held torch's
		// anchor moves with the camera every frame and is recomputed in the
		// flame-update loop instead. Used by findGazedCandle() as the aim
		// target, and by the shadow face matrices below.
		glm::vec3 anchorWorld = glm::vec3(0.0f);

		// True for every flame that competes for a slot in the DYNAMIC
		// shadow-cube pool (updateDynamicShadowSlots()) -- set automatically
		// in addTorchFlame() as !heldByCamera, not passed in by callers:
		// every flame in this project is either the one held torch (its own
		// dedicated HAND_TORCH_SHADOW_INDEX slot, recomputed every frame
		// since it's the one that moves) or a static object, and every
		// static one belongs in the pool. A future addTorchFlame() call for
		// a new torch/candle is a shadow candidate for free, with nothing
		// to remember to flip on.
		bool shadowCandidate = false;

		// This flame's absolute cube-shadow slot if it currently holds one
		// of the dynamic pool's slots, else -1. Only meaningful when
		// shadowCandidate is true; read by the light-append loop below to
		// fill LightData::shadowIndex.
		int shadowSlot = -1;

		// This candidate's own six face view-projection matrices, computed
		// once when it's registered in addTorchFlame() below -- it's a
		// static object, same as the six lights.json torches, so unlike the
		// held torch it never needs recomputing after that. Copied into
		// torchFaceMatrices[]/torchLightPos[] whenever
		// updateDynamicShadowSlots() hands this flame a slot.
		std::array<glm::mat4, 6> shadowFaceMatrices{};

		// That lean, resolved into the billboard's own axes (x = the
		// billboard's right, y = its forward) and expressed in the same units
		// the flame's local geometry uses. Uploaded straight to the shader.
		glm::vec2 lean = glm::vec2(0.0f);
	};
	std::vector<TorchFlame> torchFlames;

	// Candle lighting: the third interactable, alongside Door and Pickup, and
	// the one that needed no list of its own -- a candle IS a TorchFlame that
	// happens to be authored unlit, so `nearbyCandle` indexes torchFlames
	// directly rather than duplicating anything into a parallel vector.
	//
	// Same gaze-then-proximity shape as the other two (findGazedCandle picks
	// the target, the radius below gates whether it's reachable), so the
	// crosshair, the focus glow and the [E] prompt all behave the way the
	// player already learned them on doors and keys.
	//
	// Radii: measured in 3D like a pickup's, not XZ-only like a door's -- a
	// candle sits on a table, well above the player's feet. Tighter than the
	// pickup's, because a candle is a small prop AND because reaching one
	// means holding a burning torch up to it, which is a close-range gesture,
	// not something done from across the room.
	static constexpr float CANDLE_INTERACT_RADIUS = 2.5f;
	static constexpr float CANDLE_LOOK_DISTANCE = 5.0f;
	// Half-width for the aiming cone. As tight as a pickup's: the wick is a
	// small target and the candle's own body is most of the model.
	static constexpr float CANDLE_AIM_RADIUS = 0.35f;
	// Index into `torchFlames` of the unlit candle currently in range, or -1.
	// Mirrors nearbyDoor/nearbyPickup, set every frame in GameLogic().
	int nearbyCandle = -1;

	// Index into `torchFlames` of the unlit candle the player is aiming at
	// within look range, or -1. Same two-gate split as findGazedDoor: this
	// answers "targeted at all", CANDLE_INTERACT_RADIUS answers "close enough
	// to light". Already-lit candles are skipped rather than reported and
	// rejected later, so a lit one neither glows nor steals the aim from an
	// unlit one behind it -- there is nothing left to do to it.
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

	// Wall-torch lighting: the mirror of the candle interaction above. The
	// held torch starts unlit; aiming at a burning wall torch within reach
	// and pressing [E] lights it. Once lit it stays lit for the run
	// (restartRun blows it out again, exactly like the candles the player
	// lit). A wall torch is a far bigger target than a wick and sits at head
	// height, so the aim cone is more forgiving than the candle's; the reach
	// is the same close-range gesture.
	static constexpr float WALL_TORCH_INTERACT_RADIUS = 3.0f;
	static constexpr float WALL_TORCH_LOOK_DISTANCE = 6.0f;
	static constexpr float WALL_TORCH_AIM_RADIUS = 0.6f;
	// Index into `torchFlames` of the burning wall torch currently in reach
	// and aimed at, or -1. Mirrors nearbyCandle, set every frame in GameLogic().
	int nearbyWallTorch = -1;

	// Index into `torchFlames` of the burning wall torch the player is aiming
	// at within look range, or -1. Only ever non-negative while the held
	// torch is unlit: a lit torch has nothing to gain from another, so no
	// wall torch is offered as a target once it's burning. Same two-gate
	// split as findGazedCandle: this answers "targeted at all",
	// WALL_TORCH_INTERACT_RADIUS answers "close enough to light from".
	// Also -1 until the torch is actually in hand: there's nothing to light
	// while it's still on the floor.
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

	// Is this flame currently burning at all? The single question the two
	// flame cheats (CheatFlags::roomTorchesEnabled/handTorchEnabled) are asked
	// through, so every consequence of a flame -- its point light, its
	// billboard, its shadow-slot candidacy, its stare-at glare -- switches off
	// together instead of each site testing a different flag and drifting.
	// Candles answer with their own runtime `burning` flag instead: no cheat row
	// claims them, and whether one burns is gameplay (the player lit it, see
	// the candle block in GameLogic) rather than a debug switch.
	bool flameBurning(const TorchFlame &tf) const {
		if(tf.heldByCamera) return cheats.handTorchEnabled && tf.burning;
		if(tf.isCandle)     return tf.burning;
		return cheats.roomTorchesEnabled;
	}

	// Does the player currently have fire in hand to light something WITH?
	// The held torch is a scene instance that always exists, but the player
	// starts without it: it has to be picked up off the floor
	// (handTorchCollected), then lit from a wall torch (see handFlameIdx) --
	// so this is false until both have happened, and false again if the
	// "Holding Torch" cheat is off. The candles ask through this rather than
	// reading the cheat or the flame directly, so there is one place that
	// decides "there is fire in hand".
	bool hasBurningTorch() const {
		return handTorchInst != nullptr && handTorchCollected &&
			   cheats.handTorchEnabled &&
			   handFlameIdx >= 0 && torchFlames[handFlameIdx].burning;
	}

	// Global glare level, 0..1: the max over every wall torch's own stare-at
	// factor, smoothed asymmetrically (GLARE_TAU_RISE/FALL). Feeds the post
	// chain's exposure/bloom every frame -- a member rather than a local so
	// the smoothing survives between frames.
	float glareSmoothed = 0.0f;

	// One anchor for every torch: SM_Torch_01 (wall-mounted) and
	// SM_Torch_Held_01 (held) turn out to be the identical mesh, confirmed
	// by walking their POSITION accessors directly rather than trusting the
	// two models' reported min/max (which, misleadingly, differ). X -0.544..
	// -0.224 (mid -0.384), Z -0.165..0.165 (mid 0) at the centroid of the
	// mesh's own top (a wide flat cup, not a point, so a bbox corner isn't
	// the right anchor). Y is 0.30, well BELOW that top (0.413): the cup has
	// depth, so sitting the flame's own base ring exactly at the rim left it
	// looking like it was floating just above the torch instead of coming
	// out of it. The flame's field fades out right at its own y=0
	// (Flame.frag's baseFade plus the alpha window), so its first VISIBLE
	// pixels sit a few percent up the card -- the anchor compensates by
	// sinking that much further into the cup, and the fade doubles as the
	// flame emerging from inside it rather than balancing on the rim.
	// Also flames.json's fallback default for a model with no "anchor" of
	// its own, and the value used for both "dungeonTorch" and
	// "dungeonTorchHeld" there (confirmed identical meshes the same way).
	static constexpr glm::vec3 TORCH_FLAME_ANCHOR = glm::vec3(-0.384f, 0.30f, 0.0f);

	// The flame's size, in the torch model's own local units, so it rides
	// each instance's uniform scale: full size on the wall-mounted torches,
	// shrunk to match on the held one (whose instance is scaled down to
	// arm's-length size, HAND_TORCH_SCALE). A single fixed world-space size
	// instead made the flame look right on the (small) held torch and
	// comically undersized on the (full-size) wall ones.
	//
	// HALF_WIDTH is deliberately wider than the visible flame ends up being:
	// the billboard's noise field eats into its own silhouette from the
	// edges in, so the quad has to be bigger than the fire that ends up
	// drawn inside it or the flame gets visibly clipped to a rectangle.
	static constexpr float FLAME_HEIGHT = 0.95f;
	static constexpr float FLAME_HALF_WIDTH = 0.20f;

	// The torch flame's point light. One color/falloff for every torch in
	// the scene (held and wall-mounted alike): they're all the same kind of
	// fire, so there's nothing to author per-instance. Tighter (lower g)
	// than the gate lanterns (SceneLights.hpp/lights.json): a torch flame is
	// a much smaller, closer source than a lamp head.
	static constexpr glm::vec3 TORCH_LIGHT_COLOR = glm::vec3(1.0f, 0.5f, 0.16f);
	// With the falloff (g/d)^beta, g = 2.1 makes the torches genuinely carry
	// into the room instead of only rimming their own wall.
	//
	// Both g and beta got tuned up (g to 3.5, beta down from 1.4) partway
	// through tracking down what turned out to be the real bug: torches
	// looked like they went dark past a hard-edged radius because
	// TORCH_SHADOW_FAR_CONST's shadow-cube far plane was too short, not
	// because of anything about this falloff curve. That's fixed now (see
	// TORCH_SHADOW_FAR_CONST's comment), so both are back to their original,
	// separately-authored values instead of ones that were compensating for
	// a shadow bug.
	static constexpr float TORCH_LIGHT_G = 2.1f;
	static constexpr float TORCH_LIGHT_BETA = 1.4f;
	// Candles additionally shrink g (their falloff reach), on top of
	// flames.json's own lightScale (their peak brightness): the two are
	// independent knobs the same way sizeScale/lightScale are (see
	// flames.json's dungeonCandle comment) -- a dim flame that still
	// reaches TORCH_LIGHT_G units out would carry into the next room at
	// torch-like range, just dimly, instead of reading as a small local
	// pool around the wick.
	static constexpr float CANDLE_LIGHT_G_SCALE = 0.35f;
	// The fraction of its peak contribution below which a light's shadow stops
	// being worth re-capturing for a moving occluder -- see
	// shadowRelevantReach(), which turns this into a per-light distance. 2%:
	// low enough that the cut is invisible even in a dungeon this dark (the
	// ambient term alone is worth more than that), high enough that a ghost
	// walking the far end of the level doesn't keep every torch in the place
	// re-rendering. This is the one number the mover tracking is tuned by; if a
	// shadow ever stops following its ghost too early, lower it.
	static constexpr float SHADOW_REACH_CUTOFF = 0.02f;

	// How far a torch light still gets uploaded, and how many may be live at
	// once. CookTorrance.frag loops over every light for every fragment
	// (times the MSAA sample count -- Starter.hpp turns on
	// sampleShadingEnable with minSampleShading 1.0, i.e. full per-SAMPLE
	// shading, and this project runs 4x MSAA -- see msaaSamples), so an
	// uploaded light costs a full GGX evaluation FOUR TIMES per pixel,
	// whether or not it can be seen. This is the actual dominant per-frame
	// GPU cost in this renderer, well above anything the geometry visibility
	// cull below touches (that one only trims draw calls, which were never
	// the bottleneck at this instance count).
	//
	// 25 used to be "generous" back when this was written against a
	// six-torch scene and a much smaller live-light/shadow-slot budget --
	// wrong for a while after that: the dungeon's own footprint is ~60 units
	// across, so a 25-unit radius dropped any torch in a room the player
	// wasn't standing in, VISIBLY (its flame billboard is unconditional, see
	// Flame.hpp, so it stayed lit-looking on screen while casting zero light
	// and shading its own surroundings pitch black -- exactly what a
	// purely-numeric "3% contribution, below what ambient hides" estimate
	// can't catch). 120 fixed that by comfortably covering the whole level
	// from any point in it, at the cost of uploading nearly every torch in
	// the dungeon nearly all the time.
	//
	// Tied to GEOM_CULL_CONE_DIST below (with a small margin) rather than to
	// its own flat number now that that geometry cull exists: anything whose
	// TORCH BRACKET is still being drawn is guaranteed to still be lit, so
	// the "visibly glowing but dark" case above can't reoccur for anything
	// with a visible model behind it. The residual case that can still
	// happen -- a flame's billboard alone, unconditional and undimmed,
	// rendering past both cutoffs with nothing lighting it -- is far less
	// noticeable than a fully modelled, clearly-visible dark torch was: a
	// small/distant glow with no bracket to contrast it against. Kept as a
	// static_assert right after GEOM_CULL_CONE_DIST is declared, so the two
	// can't drift out of sync by editing only one of them.
	static constexpr float TORCH_LIGHT_CULL_DIST = 55.0f;
	static constexpr int TORCH_LIGHT_MAX_LIVE = 32;

	// Geometry visibility: a radius around the player, plus a longer cone
	// down whatever direction the camera is actually facing. Replaces an
	// earlier room-graph approach (flood-filling through open doors from an
	// authored or collider-derived room box) that turned out to be too
	// fragile against this asset pack: wall pieces don't reliably tile a
	// room's full footprint with a scene.json "collider", so the derived
	// boxes had gaps big enough to hide the room the player was STANDING
	// in, not just the ones further away. A radius+cone test needs none of
	// that: it reads only the camera's live position/facing and each
	// instance's own position, so there is no room topology to get wrong
	// and no per-scene authoring to keep in sync as the level changes.
	//
	// The tradeoff, on purpose: unlike a room graph this has no idea a wall
	// is between the camera and something behind it, so a light-hearted
	// example would be an instance just past an open doorway showing up
	// slightly before the doorway itself is reached, if it happens to sit
	// inside the cone. That's an acceptable, cheap approximation for a
	// player-facing "what's worth drawing" cut, not a portal-correct
	// visibility system. Only applied to GEOMETRY (see the UBO loop in
	// updateUniformBuffer()) -- deliberately NOT applied to the torch light
	// list below, which stays on its own pure-distance cull: a light that
	// stops being uploaded to gubo while its flame's billboard (drawn
	// separately, unconditionally, see Flame.hpp) keeps rendering reads as
	// a burning torch that lights nothing, which is exactly the bug
	// TORCH_LIGHT_CULL_DIST's own comment above already had to fix once.
	//
	// GEOM_CULL_RADIUS: always draw anything this close, regardless of
	// facing -- so turning around, or a wall just to your side, doesn't pop.
	// Camera FOV is only 45 degrees vertical (see FOVy below), but on a wide
	// enough window the diagonal half-angle still stretches close to the
	// cone's own half-angle, so this and GEOM_CULL_CONE_COS both carry
	// comfortable headroom past the frustum's actual edges rather than
	// tracking them exactly -- cheaper than deriving the true frustum planes
	// every time the window resizes, for a cut that only has to be
	// approximately right.
	static constexpr float GEOM_CULL_RADIUS = 16.0f;
	// GEOM_CULL_CONE_DIST: how far the extended cone reaches down the view
	// direction -- long enough to see clear down the dungeon's own ~60-unit
	// longest sightline without popping the far wall into view a step at a
	// time.
	static constexpr float GEOM_CULL_CONE_DIST = 50.0f;
	// Keeps TORCH_LIGHT_CULL_DIST (declared above, before this one exists --
	// see its own comment for why) at least as far as this cone reaches, so
	// nothing whose torch bracket is still drawn can ever end up unlit.
	static_assert(TORCH_LIGHT_CULL_DIST >= GEOM_CULL_CONE_DIST,
				  "a torch light cull shorter than the geometry cone would "
				  "leave a visible torch model unlit");
	// GEOM_CULL_CONE_COS: half-angle of that cone, as a cosine (so the test
	// is a plain dot product, no acos per instance per frame). ~70 degrees
	// half-angle (140 total), well past the widest diagonal FOV this camera
	// can produce, plus the per-instance size allowance below on top.
	static constexpr float GEOM_CULL_CONE_COS = 0.34f;
	// A wall or piece of furniture is tested by its CENTRE, not its visual
	// extent, so a long wall segment whose pivot sits at one end (this asset
	// pack's meshes aren't consistently centre-pivoted) could measure as
	// outside the radius/cone while half of it is still plainly on screen --
	// this was the actual cause of things vanishing right at the screen's
	// edges, not the angle/distance numbers being too tight. Fixed by
	// pretending every instance is a bounding SPHERE instead of a point:
	// its collider's extents give a real radius when it has one (most
	// walls/furniture do, see scene.json's "collider": "AABB"); anything
	// without one (candles, skulls, floor/ceiling tiles, ...) falls back to
	// this flat allowance, sized to a floor/ceiling tile's own footprint
	// (the largest un-collided things in the scene) so it still errs toward
	// not culling rather than under-covering them.
	static constexpr float GEOM_CULL_FALLBACK_RADIUS = 4.0f;

	// True if the instance at worldPos, with bounding radius objRadius, is
	// close/aimed-at enough to draw, per GEOM_CULL_* above. eyePos/forward:
	// the same pair updateUniformBuffer() already computes once per frame
	// from the view matrix, passed through rather than recomputed per
	// instance.
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
			return true;	// camera exactly at the object's centre
		}
		float facing = glm::dot(d / dist, forward);
		// Same idea for the angle test: an object's own angular size is
		// roughly objRadius/dist (small-angle approximation), so a nearby or
		// large one needs proportionally less of a head-on facing to still
		// count as in the cone. Clamped so a tiny/far object still needs the
		// full GEOM_CULL_CONE_COS, and a huge/adjacent one doesn't get
		// forgiven into a 0 (i.e. treated as always in the cone) -- that's
		// what GEOM_CULL_RADIUS above is already for.
		float angularSlack = glm::clamp(objRadius / dist, 0.0f, 1.0f) * 0.5f;
		return facing >= (GEOM_CULL_CONE_COS - angularSlack);
	}

	// How much a candidate's priority (both the live-light cut above and the
	// shadow-cube pool below) is skewed by whether it's ahead of or behind
	// the player, as a fraction of its real distance. 0 disables this
	// entirely (pure nearest-first); 1.0 would let a light directly behind
	// the player be treated as twice as far as its real distance for
	// EQUAL alignment credit as one directly ahead, penalized by nothing.
	// facingBiasedDistSq()'s (1 + W * (1 - alignment)) term ranges
	// 1 (dead ahead, alignment 1) .. 1+W (directly to the side, alignment 0)
	// .. 1+2W (dead behind, alignment -1), so at 0.6: a torch 8 units away
	// but behind loses its slot to one 10 units away but in view (effective
	// 64*2.2=140.8 vs 100*1=100), while a torch merely off to the side still
	// mostly competes on real distance (effective 1.6x, not 2.2x).
	static constexpr float SHADOW_FACING_BIAS_WEIGHT = 0.6f;

	// Squared distance from eyePos to pos, inflated for anything not ahead
	// of `forward` -- see SHADOW_FACING_BIAS_WEIGHT above. Still monotonic in
	// real distance for a fixed alignment, so nearer-and-equally-in-view
	// still always beats farther-and-equally-in-view; only the front/behind
	// axis gets a thumb on the scale. One sqrt (for the cosine) instead of
	// two (glm::normalize would need its own): distSq is already at hand at
	// every call site.
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

	// Fire envelope. The fast, physically-right flicker band lives in
	// Flame.frag instead, as a per-pixel shimmer that varies along the flame
	// -- a WHOLE-FLAME envelope applying that twitch to every pixel at once
	// reads as a brightness dial being wiggled, since no real flame changes
	// uniformly. The CPU keeps a slower 7 Hz term at reduced weight purely so
	// the light the torch casts still dances a little, layered under slower
	// terms so the result doesn't read as a metronome (a pure sine period is
	// something the eye locks onto within two seconds).
	static constexpr float FLAME_FLICKER_HZ = 7.0f;
	// The spring rate the brightness envelope chases its target with
	// (critically damped, see TorchFlame::intensityVel). 14 rad/s settles in
	// roughly 2/14 ~ 0.15 s: fast enough that a gutter still visibly ducks,
	// slow enough that no frame-to-frame noise step survives as a jump.
	static constexpr float FLAME_BRIGHT_OMEGA = 14.0f;
	// How slowly the HEIGHT envelope chases the same signal (one-pole, in
	// seconds). A flame shortens over a third of a second; it doesn't
	// teleport between heights the way it can flicker in brightness.
	static constexpr float FLAME_HEIGHT_TAU = 0.35f;
	// Guttering: brief, irregular collapses where the flame ducks and dims
	// almost to nothing. Driven by its own slow noise crossing a high
	// threshold, so it happens rarely and never on a schedule.
	static constexpr float FLAME_GUTTER_SPEED = 0.85f;
	static constexpr float FLAME_GUTTER_LO = 0.66f;	// noise below this: no gutter
	static constexpr float FLAME_GUTTER_HI = 0.82f;	// above this: full gutter
	static constexpr float FLAME_GUTTER_DEPTH = 0.48f;	// how far it ducks

	// Stare-at glare: walking up to a wall torch and centring it in view
	// swells the post chain's exposure and bloom (and that torch's own HDR
	// output), like an eye caught by something too bright to look at. Doing
	// it view-DEPENDENT rather than just making flames brighter is what
	// masks the billboard's flat-card nature exactly when it is most
	// visible: face-on, up close. Purely geometric -- camera pose, anchor
	// position, envelope -- so there is no feedback through the renderer.
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

	// Lean. TAU is how fast the smoothed velocity chases the real one: too
	// short and the flame snaps about as rigidly as it did before, too long
	// and it keeps leaning after you've stopped. GAIN converts world units/
	// second into flame half-widths of lean, and MAX caps it so sprinting
	// can't fold the flame flat onto its side.
	static constexpr float TORCH_LEAN_TAU = 0.14f;
	// Half-widths of tip offset per world-unit/second of hand speed. A brisk
	// walk is around 3 units/s, so this puts the tip over by well under a
	// fifth of the flame's own half-width -- a lean you notice as the flame
	// reacting, not as the flame being knocked sideways.
	static constexpr float TORCH_LEAN_PER_SPEED = 0.055f;
	// Hard cap, in the same units. A flame that trails much further than this
	// stops reading as fire being dragged and starts reading as a rigid object
	// bolted to the top of the stick at an angle.
	static constexpr float TORCH_LEAN_MAX = 0.30f;

	// Seconds since startup, uploaded as gubo.time and read by Flame.vert/
	// .frag and Spark.vert. A free-running accumulator rather than a
	// frame-indexed value, so the sway/flicker never repeats on a
	// noticeable cycle.
	float animTime = 0.0f;

	// Walking sway: a lateral swing once per stride plus a vertical bounce
	// at twice that frequency (one bounce per footstep), both driven by a
	// single accumulating phase. walkBobBlend is the 0..1 sway amplitude,
	// eased toward 1 while walking and back to 0 at rest.
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

	// The dungeon ghost (assets/models/Entities/Ghost.gltf, "ghost" instance
	// in scene.json): a single-mesh, single-texture prop (see the model's own
	// notes.md-style history -- body and "orb" were originally two separate
	// pieces, joined in Blender and re-UV'd onto one flat-grey texture so the
	// engine's one-texture-per-instance loader could draw it as one entity).
	// Patrols a closed loop of waypoints at constant speed, facing its
	// direction of travel, with a sinusoidal bob layered on top of the Y
	// coordinate -- same idea as the torch's flame envelope, just applied to
	// world position instead of brightness.
	//
	// Drawn with the "Spectral" technique (Pspectral / shaders/Spectral.frag),
	// NOT with the CookTorrance one every other prop uses. It used to be the
	// latter, which bought it shadows in the sun's 2D map and the torches' cube
	// maps for free -- until the sun was removed (88e019e) and the cube
	// re-captures a continuously moving occluder forces turned out to cost more
	// than the shadow was worth (46777d5, and materials.json's "ghost" entry).
	// What was left was an opaque grey prop with no shadow under it, and the
	// missing shadow was the most visible thing about it.
	//
	// The Spectral technique answers that by making the ghost incorporeal
	// instead: alpha-blended, unlit, emissive, lit along its own silhouette by
	// a Fresnel rim -- which is why the model's ragged hem glows without the
	// shader knowing the hem exists. Something you can see the wall through has
	// no business casting a shadow, so the cheap path stops looking like a
	// compromise. See Spectral.frag's header for the effect itself and
	// Pspectral's declaration for the pipeline state it needs.
	//
	// Two consequences elsewhere, both wanted: the shadow passes in
	// populateCommandBuffer() walk SC.TI[0] only, so the ghosts are now outside
	// them structurally rather than by a castsShadow check, and the ghosts no
	// longer receive shadows either -- an unlit emitter has nothing to darken.
	//
	// Each ghost runs a three-state machine, driven entirely by
	// huntCycle.hunting():
	//
	//   Patrol   the original behaviour: walk the authored waypoint loop.
	//   Chase    drop the loop and steer toward the player, around walls.
	//   Return   the hunt is over, so walk BACK to where the chase started
	//            and pick the patrol up exactly where it was left.
	//
	// Return is the state that makes the whole thing work, and it exists
	// because of one specific problem: a ghost that can't pass through walls
	// can end a chase anywhere in the castle -- three rooms and two doorways
	// away from its loop -- and "walk back to a point you can no longer see"
	// is exactly the pathfinding problem we don't want to solve for a level
	// this small.
	//
	// So it isn't solved. During a chase the ghost drops a breadcrumb every
	// GHOST_TRAIL_SPACING units (`trail` below), and returning is just walking
	// that list backwards. The route home is guaranteed walkable because the
	// ghost physically walked it a moment ago, and it costs one vector push
	// every few frames instead of a nav mesh, an A* and a graph to run it on.
	//
	// The trail also self-prunes: a new breadcrumb landing near an older one
	// truncates everything after that older one (see the chase block in
	// GameLogic). A ghost that spends a hunt circling a table therefore walks
	// home in a straight-ish line rather than re-tracing every lap, and the
	// list stays bounded no matter how long a hunt runs.
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
		// Live world position WITHOUT the bob: the bob is presentation only,
		// and folding it in here would make the ghost's collision slab pump up
		// and down. Maintained in every mode, Patrol included, so a chase can
		// start from wherever the loop had got to.
		glm::vec3 pos{0.0f};
		// Facing, eased rather than snapped: steering around a corner changes
		// the heading in a single frame, and a ghost spinning on the spot
		// reads as a bug, not a haunting.
		float yaw = 0.0f;
		// Which way it went around an obstacle last frame, +1 or -1. Kept until
		// it gets a clear run at the player again, so it commits to one side of
		// a pillar instead of dithering in front of it.
		float turnBias = 1.0f;

		// Breadcrumbs, oldest first. trail[0] is where the chase began.
		std::vector<glm::vec3> trail;
		// Patrol progress saved the moment the chase started, restored when the
		// ghost gets back to trail[0].
		int resumeIdx = 1;
		float resumeDist = 0.0f;

		// Where the ghost last actually SAW the player, and whether that's ever
		// happened this hunt. Chase steers toward this, not toward the player's
		// live position -- see ghostHasLineOfSight. A ghost that has never seen
		// the player has nothing to chase and stays on patrol.
		glm::vec3 lastKnownPlayerPos{0.0f};
		bool hasLastKnown = false;

		// Stuck detection for Chase: `pos` at the last check, and how long
		// since then the ghost has covered less than GHOST_STUCK_EPS. A ghost
		// pressed against a closed door or idling at a stale lastKnownPlayerPos
		// looks identical from here -- either way it isn't getting anywhere,
		// and GHOST_GIVEUP_TIME is what turns that into giving up rather than
		// waiting out the rest of the hunt on the wrong side of a door.
		glm::vec3 stuckCheckPos{0.0f};
		float stuckTimer = 0.0f;

		// 0..1, how far this ghost is into a chase, eased rather than switched.
		// Purely presentational: Spectral.frag reads it (delivered as ubo.F0,
		// see updateUniformBuffer) to shift the apparition from its calm tint
		// towards the hunt red and to burn brighter. It is also the only thing
		// on that shader that changes over time at all -- nothing there idles
		// or pulses -- so this ease IS the animation. Eased because `mode`
		// flips in one frame and a ghost that changes colour instantly reads as
		// a texture swap rather than as something turning on you -- and because
		// Chase is entered and left repeatedly within one hunt (see the stuck /
		// give-up path), which as a hard switch would strobe.
		float chaseBlend = 0.0f;
	};
	std::vector<Ghost> ghosts;

	// Time constant of Ghost::chaseBlend's exponential ease, in seconds, one
	// per direction. Lighting up is faster than calming down on purpose: the
	// telegraph that a ghost has seen you has to arrive while it still buys you
	// something, and the fade back out is what sells the chase having been let
	// go of rather than merely toggled off.
	static constexpr float GHOST_CHASE_FADE_IN_TAU = 0.25f;
	static constexpr float GHOST_CHASE_FADE_OUT_TAU = 1.1f;

	// Bob envelope: how far above/below the resting hover height (radians/sec, world units).
	static constexpr float GHOST_BOB_SPEED = 1.6f;
	static constexpr float GHOST_BOB_AMPLITUDE = 0.3f;
	// How fast a ghost walks its breadcrumbs home. Faster than either patrol or
	// chase: the return is dead time for the player, and a ghost drifting back
	// across the map at patrol speed would still be out of position when the
	// next hunt starts.
	static constexpr float GHOST_RETURN_SPEED = 4.5f;
	// Yaw easing, radians/second. Roughly a half-turn in a third of a second.
	static constexpr float GHOST_TURN_SPEED = 9.0f;

	// Collision size: a vertical cylinder, not a box. A box would rotate with
	// the mesh's facing (or, left axis-aligned, would silently stop matching
	// it), and either way its corners project further out on a diagonal than
	// a circle of the same "radius" -- which is exactly how a box collider
	// snags on a doorway jamb or a corridor corner that a cylinder just slides
	// past. That snagging risk is why character/creature controllers use
	// capsules instead of boxes as a matter of course, and it's the reason
	// this stays a circle in XZ even though the ghost itself isn't round.
	//
	// Both numbers below are fitted from Ghost.gltf's own geometry at load
	// time (see the fit right after ghosts are read from gameplay.json)
	// instead of being hand-measured constants, so a model swap can't quietly
	// desync them again the way the old hardcoded values did.
	//
	// The vertical slab is taken at the mesh's exact fitted bounds: unlike the
	// radius there's no "getting stuck" failure mode to guard against by
	// shrinking it, and shrinking it is exactly what caused the ghost to float
	// over furniture it visibly clipped through (see ghostBlockedAt/
	// ghostResolveWalls). Ghost.gltf's local bounds run from -1.80 to +0.83
	// around its origin -- most of the body hangs below the pivot, not
	// centered on it -- which is also why this is two numbers and not one
	// symmetric half-height.
	//
	// The radius, by contrast, IS deliberately shrunk below the mesh's actual
	// footprint (ghostXZFitShrink), the same way it always was: a circle sized
	// to guarantee zero visual clipping would be wide enough to snag in a
	// doorway. Shrinking it off the real fit rather than picking an unrelated
	// number keeps it in the same ballpark as the mesh if the model changes.
	//
	// 0.45 lands the fitted radius close to 0.5 -- the value this project
	// shipped and navigated doorways with before any of this fitting existed.
	// A larger shrink (tried: 0.65, plus a steering margin on top of that)
	// pushed the effective radius close enough to half the doorway width that
	// ghostSteer's clear/blocked test started flipping every frame near a
	// threshold, which is worse than the clipping it was meant to fix: a
	// ghost that visibly clips a table is a minor visual issue, one that
	// visibly vibrates in a doorway is a broken one. Getting all the way back
	// to zero clipping isn't the goal here; not regressing movement is.
	static constexpr float ghostXZFitShrink = 0.45f;
	// Steering briefly probed with extra padding above the real radius, meant
	// to stop borderline gaps from flip-flopping every frame. Playtesting it
	// alongside the 0.65 shrink made things worse, not better -- padding a
	// radius that was already close to half the doorway width just made
	// "blocked" win the flip-flop more often. Back to 1.0 (no margin); the
	// parameter stays in ghostPathClear/ghostSteer in case it's worth
	// revisiting once the radius itself (ghostXZFitShrink, above) is confirmed
	// comfortable, rather than stacked on top of a radius that was still
	// riding the edge.
	static constexpr float ghostSteerMargin = 1.0f;
	float ghostRadius = 0.5f;
	float ghostBodyBottom = -1.80f;
	float ghostBodyTop = 0.83f;
	// How far ahead a candidate heading is tested for walls before the ghost
	// commits to it. Long enough to see a wall in time to turn along it, short
	// enough that it doesn't refuse to enter a doorway.
	static constexpr float GHOST_PROBE_DIST = 1.2f;
	// Horizontal distance at which a hunting ghost catches the player, plus the
	// vertical slack that catch allows. The vertical part matters: the ghosts
	// hover at 2.2 and the player's eyes are at 1.8, so a plain 3D distance
	// test would need a radius big enough to be unfair horizontally.
	static constexpr float GHOST_CATCH_RADIUS = 0.85f;
	static constexpr float GHOST_CATCH_VERTICAL = 2.5f;

	// THE SPECTRAL VEIL, the screen half of the ghost fade in
	// include/custom/SpectralFade.glsl. Walking through a ghost is a thing the
	// player does -- the catch above only fires while they are hunting -- and a
	// ghost that simply disappears when you reach it says it was never a body.
	// So Composite.frag washes the frame cold instead, and these numbers MUST
	// match SPECTRAL_INSIDE_* in that file: a gap between the two ramps is a
	// moment with no ghost and no wash.
	static constexpr float SPECTRAL_VEIL_OUTER = 2.00f;
	static constexpr float SPECTRAL_VEIL_INNER = 1.05f;
	// Margin above and below the model's Y bounds (ghostBodyBottom/Top), so the
	// bob doesn't blink the wash on and off from the edge of its reach.
	static constexpr float SPECTRAL_VEIL_FADE_Y = 0.60f;
	// Breadcrumb spacing, and the radius within which a new breadcrumb counts
	// as revisiting an old one (and prunes the loop between them). The prune
	// radius has to be comfortably larger than the spacing, or consecutive
	// breadcrumbs would prune each other and the trail could never grow.
	static constexpr float GHOST_TRAIL_SPACING = 0.75f;
	static constexpr float GHOST_TRAIL_PRUNE_RADIUS = 1.4f;

	// How little ground counts as "not really moving" for the stuck check
	// below, and how long a chasing ghost tolerates that before giving up.
	// Padded a bit above pure floating-point idle drift -- a ghost easing its
	// yaw or nudged half a centimetre by ghostResolveWalls shouldn't reset the
	// clock, only an actual stall (a closed door, an empty lastKnownPlayerPos)
	// should.
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

	// True whenever any modal overlay (cheat HUD, pause menu, or the launch
	// screen) is open. GameLogic() freezes camera/movement/physics/the hunt
	// clock behind this, same as it always did for hud.isOpen() alone.
	bool overlayOpen() const { return hud.isOpen() || pauseMenu.isOpen() || startScreen.isOpen(); }

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
	// Index into `doors` of the exit leaf (dvDoorPanel), or -1 if the scene
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
	static constexpr glm::vec3 EXIT_GLOW_CENTER = glm::vec3(22.0f, 2.8f, 29.99f);
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
	static constexpr glm::vec3 EXIT_GLOW_FLOOR_CENTER = glm::vec3(21.1f, 0.06f, 29.99f);
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
	static constexpr glm::vec3 EXIT_GLOW_CEILING_CENTER = glm::vec3(21.1f, 4.90f, 29.99f);
	static constexpr float EXIT_GLOW_CEILING_HALF_X = 1.6f;
	static constexpr float EXIT_GLOW_CEILING_HALF_Z = 3.6f;
	static constexpr glm::vec3 EXIT_GLOW_CEILING_NORMAL = glm::vec3(0.0f, -1.0f, 0.0f);
	// Quad ids, in the order updateUniformBuffer() writes them. Named rather
	// than passed as bare 0/1/2 because the three are not interchangeable:
	// they have different bases and different half-extents.
	static constexpr int EXIT_GLOW_UPRIGHT = 0;
	static constexpr int EXIT_GLOW_FLOOR = 1;
	static constexpr int EXIT_GLOW_CEILING = 2;
	static constexpr int EXIT_GLOW_COUNT = 3;
	// Peak radiance. Absurd on the face of it -- BLOOM_THRESHOLD is 1.55 --
	// and it has to be, because of the tone map: Composite.frag divides by
	// (Y + 1), so a value of 9 lands at 0.90 on screen and a value of 60 at
	// 0.984. Everything between "bright" and "cannot look at it" lives in that
	// last stretch, and reaching it costs an order of magnitude. The bloom
	// chain then takes the same unclamped value and floods the stonework
	// around the opening with it, which is the part that actually reads as
	// dazzle rather than as a white shape.
	static constexpr float EXIT_GLOW_INTENSITY = 60.0f;
	// Daylight, warmed very slightly. Pure white read as a hole in the render
	// rather than as sky.
	static constexpr glm::vec3 EXIT_GLOW_COLOR = glm::vec3(1.0f, 0.97f, 0.90f);
	// The light the doorway throws BACK into the room, as a spot appended
	// straight into gubo (the same thing the torch loop does with its flames),
	// not as a lights.json entry: its brightness is a function of the door's
	// angle, and lights.json has no way to say that. A spot rather than a
	// point because the light has to come through the opening -- a point light
	// out there would wrap round and light the outside face of the east wall
	// as brightly as the floor inside.
	static constexpr glm::vec3 EXIT_SPILL_POS = glm::vec3(20.7f, 2.6f, 29.99f);
	// Well over 1: this is a doorway onto open daylight standing in a room lit
	// by torches, and a spill light that merely matched them would leave the
	// stone around the opening looking like it was lit by another torch. The
	// scene target is HDR, so overbright light colours are as legitimate here
	// as they are on the flames -- and the bloom chain treats what this lights
	// up the same way it treats the quad itself.
	static constexpr glm::vec3 EXIT_SPILL_COLOR = glm::vec3(3.4f, 3.26f, 3.0f);
	static constexpr float EXIT_SPILL_G = 9.0f;		// reaches across the dv room
	static constexpr float EXIT_SPILL_BETA = 1.0f;	// inverse-linear, so it carries
	// Narrower than it looks like it should be, because this light has no
	// shadow map (see where it is appended): nothing stops it, so a wide cone
	// would light the east wall's inner face right across the room as evenly
	// as it lights the floor in front of the opening, which reads as the wall
	// having gone transparent. Kept to a cone aimed down the doorway's own
	// axis, the leak stays where the light would honestly be anyway.
	static constexpr float EXIT_SPILL_INNER_DEG = 55.0f;
	static constexpr float EXIT_SPILL_OUTER_DEG = 105.0f;

	// 0 while the exit door is shut, 1 once it has finished swinging. Drives
	// the glow, the spill light and nothing else. Smoothstepped rather than
	// linear so the light doesn't appear at full strength the instant the
	// padlock comes off and the leaf twitches.
	float exitOpenFrac = 0.0f;

	// The whiteout. Once the run is won, this ramps 0 -> 1 and multiplies the
	// post chain's exposure and bloom, so the last thing the player sees is
	// the whole frame blowing out rather than a text banner over a dungeon.
	// It rides the two knobs the stare-at glare already uses (see the GLARE_*
	// constants), for the same reason that effect does: they are the only two
	// values in the post chain that mean "brighter", and reusing them means
	// the tone map compresses the flash exactly as it compresses everything
	// else.
	float escapeFlash = 0.0f;
	static constexpr float ESCAPE_FLASH_SECONDS = 1.6f;
	static constexpr float ESCAPE_EXPOSURE_GAIN = 7.0f;
	static constexpr float ESCAPE_BLOOM_GAIN = 3.0f;

	// The player's authored starting pose, captured in localInit() before
	// anything moves it, so restarting a run puts them back where they spawned
	// instead of at a second set of hardcoded coordinates that could drift out
	// of step with the first.
	glm::vec3 spawnPos{0.0f};
	float spawnYaw = 0.0f;
	float spawnPitch = 0.0f;

	// Tallest surface the player can walk straight onto without jumping, measured
	// from the feet. Deliberately a single shared constant rather than a local in
	// each collision block: the two collision passes in GameLogic() must agree on
	// it or they contradict each other. The wall pass skips anything at or below
	// this height (it's a step, not a wall), the ground pass accepts exactly those
	// same colliders as standable ground. Two independent copies of this value is
	// what silently turned every low collider into an unclimbable wall before.
	static constexpr float MAX_STEP_HEIGHT = 0.5f;

	// Vertical view smoothing. The ground clamp moves the camera up instantly the
	// moment the player steps onto something; this offset absorbs that jump and
	// decays back to zero, so the *view* eases up over a fraction of a second
	// while the physics position stays exact. Sloped ground (see the ramps in
	// SceneColliders) is already smooth and barely feeds this; it's what keeps
	// crates, ledges and landings from snapping.
	float eyeStepOffset = 0.0f;
	// Time constant of that decay, and the largest single snap it will absorb:
	// past this the view would trail so far below the eyes it reads as sinking.
	static constexpr float EYE_SMOOTH_TAU = 0.06f;
	static constexpr float MAX_EYE_STEP_OFFSET = 0.6f;

	// Edge-detection for the jump key, so holding it down doesn't re-trigger
	// the jump every frame while airborne/grounded.
	bool jumpKeyWasPressed = false;
	// Whether the feet are resting on a collider, refreshed every frame by the
	// floor collision check. Starts true so a jump is available immediately.
	bool grounded = true;
	// Whether we're currently sprinting.
	// Instead of simply reading the Ctrl Key state, we store the state in this flag
	// so that we can apply some logic to it: in particular, we only allow
	// to start a sprint if the player is grounded, but allow to stop sprinting
	// while in the air during a jump.
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
		// Update Render Passes. The composite follows the window exactly (the
		// final image always fills it); the scene follows it scaled down by
		// renderScale; the three bloom targets follow THAT, divided down
		// again by BLOOM_DIV -- see bloomWidth()/bloomHeight() and
		// renderScale's own comment for why each of those reads what it
		// reads. Their attachment images are torn down and rebuilt around
		// this by pipelinesAndDescriptorSetsCleanup()/Init(), which
		// Starter.hpp calls on either side of a resize.
		RP.width = renderWidth(w);
		RP.height = renderHeight(h);
		RPcomposite.width = w;
		RPcomposite.height = h;
		// After RP.width/height above, not before: bloomWidth()/bloomHeight()
		// read those, so they only see the new scene size once it's set.
		RPbright.width = RPblurH.width = RPblurV.width = bloomWidth();
		RPbright.height = RPblurH.height = RPblurV.height = bloomHeight();

		// windowWidth/windowHeight are otherwise only set once in
		// setWindowParameters() and never refreshed here; the cheat HUD
		// needs the current size for its pixel-based layout math.
		windowWidth = (uint32_t)w;
		windowHeight = (uint32_t)h;

		// updates the textual output
		txt.resizeScreen(w, h);
		uiQuad.resizeScreen(w, h);
		crosshair.resizeScreen(w, h);
		pauseQuad.resizeScreen(w, h);
		startScreenQuad.resizeScreen(w, h);
		setCrosshairQuad();
		// The collider visualizer owns a swapchain-attached render pass too
		// (Colliders.hpp), and it was never being told about resizes -- its own
		// resizeScreen() carries a comment saying it is called when the window
		// changes size, but nothing called it. Shrinking the window below its
		// starting size therefore rebuilt that pass's framebuffers at the
		// original dimensions around the new, smaller swapchain images, which
		// the validation layer flags (VUID-VkFramebufferCreateInfo-flags-04533).
		// Unrelated to the flame work; found while testing this path.
		SC.ColShow.resizeScreen(w, h);
	}
	
	// Fills in the attachment descriptions for all four passes of the HDR
	// chain, at the current swapchain size. Called once from localInit() and
	// again from onWindowResize(), since three of the five render targets are
	// sized in pixels and have to be rebuilt when the window changes.
	//
	// These are spelled out here rather than taken from
	// RenderPass::getStandardAttchmentsProperties() because none of the stock
	// configurations is floating-point: AT_SURFACE_AA_DEPTH renders straight
	// into the 8-bit sRGB swapchain, which caps every pixel at 1.0. That cap is
	// precisely the thing that makes bloom impossible -- there is no such thing
	// as "brighter than white" to find and bleed. Everything else here follows
	// from wanting a 16-bit float target instead.
	void buildPostAttachments() {
		const VkFormat HDR = VK_FORMAT_R16G16B16A16_SFLOAT;
		// Was a bright cyan (0.0, 0.9, 1.0): with the dungeon sealed and no
		// real skybox/cubemap anywhere, this only ever showed through a gap
		// in the geometry, and the one deliberate gap -- the exit archway --
		// is covered by ExitGlow's own warm daylight quads and a closing
		// ceiling piece (see EXIT_GLOW_CEILING's comment), not by this raw
		// clear colour. So it was never actually meant to be seen. Now that
		// the geometry visibility cull (GEOM_CULL_* above) stops drawing
		// walls/ceiling past its radius+cone, THIS is what shows through
		// instead of them -- black keeps that masked as darkness/distance
		// rather than a jarring bright cyan wall in the distance.
		const VkClearValue SKY = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
		const VkClearValue BLACK = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

		// --- the scene pass: multisampled HDR colour, depth, and a resolve
		// target the bloom chain and the composite can both sample.
		hdrAtt = {
			// Multisampled colour. storeOp DONT_CARE on purpose: nothing ever
			// reads the multisampled image itself, only the resolve below, so
			// writing it out to memory would be pure bandwidth -- which is the
			// scarcest thing on the integrated GPU this runs on.
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

		// --- the three bloom-chain targets. All identical: one single-sampled
		// HDR colour attachment, no depth (a full-screen quad has nothing to
		// depth-test against, and Vulkan simply ignores a pipeline's depth
		// state when its subpass declares no depth attachment).
		//
		// loadOp DONT_CARE, not CLEAR: the quad covers every pixel, so clearing
		// first would be writing the whole target twice.
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

	// (Re)builds the attachment property lists via buildPostAttachments()
	// above, then (re)initializes the five render passes from them -- the
	// scene at renderWidth()/renderHeight(), the bloom chain at
	// bloomWidth()/bloomHeight() (which read the scene's own just-set
	// width/height), the composite at the window's real size.
	//
	// Called once from localInit(), and again whenever something that
	// changes what buildPostAttachments() produces needs to take effect at
	// runtime -- today, msaaSamples (see the "MSAA" slider below): unlike
	// renderScale, which only changes the WIDTH/HEIGHT each RenderPass is
	// initialized with (onWindowResize() pokes RP.width/height directly,
	// exactly like a real resize would), a new sample count changes the
	// hdrAtt COLOR/DEPTH ATTACHMENT PROPERTIES themselves, which only
	// buildPostAttachments() knows how to regenerate and only .init() (not
	// a direct member poke) re-copies into each RenderPass. Either way, the
	// caller still has to follow this with RebuildPipeline() to actually
	// tear down and recreate the underlying images/pipelines around the
	// new properties -- this only updates the C++-side description of what
	// they should look like.
	void initRenderPasses() {
		// initializes the render passes. The scene one no longer draws to the
		// screen: it renders into an offscreen floating-point target which the
		// bloom chain and the composite then read back. See
		// buildPostAttachments() for what each attachment is and why.
		buildPostAttachments();

		// ATDEP_SIMPLE rather than the default ATDEP_SURFACE_ONLY: the scene's
		// output is now sampled by a later pass, so it needs the dependency
		// pair that orders a colour write against a subsequent shader read
		// (and, in the other direction, against the NEXT frame overwriting it).
		//
		// renderWidth()/renderHeight() rather than -1,-1 (which would mean
		// "match the swapchain" -- see RenderPass::init): this is renderScale
		// above, the actual point of it being able to differ from the
		// window's own resolution at all.
		RP.init(this, renderWidth(swapChainExtent.width), renderHeight(swapChainExtent.height), -1, &hdrAtt,
				RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);

		RPbright.init(this, bloomWidth(), bloomHeight(), -1, &brightAtt,
					  RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		RPblurH.init(this, bloomWidth(), bloomHeight(), -1, &blurHAtt,
					 RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		RPblurV.init(this, bloomWidth(), bloomHeight(), -1, &blurVAtt,
					 RenderPass::getStandardDependencies(ATDEP_SIMPLE), true);
		// The composite writes the swapchain and is read by nobody, so the
		// plain surface dependency the main pass always used is right here.
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
		// Shadow sampling (set 2 of P, see CookTorrance.frag). One UBO, one
		// separate sampler binding per map -- see the member declaration for why
		// not one array binding. linkSize on the samplers is their own index
		// into the flat VkDescriptorImageInfo list Scene builds per instance
		// (see the texDefs passed to PRs[0].init below), the same role it plays
		// for DSLlocal's single texture.
		//
		// Built in a loop rather than written out, so the count lives in
		// exactly one place. Binding 0 is the UBO, then NUM_SHADOW_MAPS_2D
		// sampler2D bindings, then NUM_SHADOW_CUBES samplerCube bindings --
		// the same order and numbering CookTorrance.frag declares its
		// shadowMap2D_*/shadowCube* with, which nothing but agreement here
		// keeps true. linkSize follows the same 2D-then-cube order (see
		// shadowMapDefs below).
		std::vector<DescriptorSetLayoutBinding> shadowSampleBindings = {
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(ShadowUniformBufferObject), 1}
				  };
		for(int i = 0; i < NUM_SHADOW_MAPS_2D; i++) {
			shadowSampleBindings.push_back({(uint32_t)(i + 1),
											VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
											VK_SHADER_STAGE_FRAGMENT_BIT, i, 1});
		}
		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			shadowSampleBindings.push_back({(uint32_t)(NUM_SHADOW_MAPS_2D + i + 1),
											VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
											VK_SHADER_STAGE_FRAGMENT_BIT, NUM_SHADOW_MAPS_2D + i, 1});
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

		// Anti-aliasing level, set BEFORE initRenderPasses() below reads it
		// (via buildPostAttachments()).
		//
		// Starter.hpp's default is getMaxUsableSampleCount(), i.e. as many
		// samples as the GPU will admit to supporting -- 16 on this machine.
		// That would be merely wasteful on its own, but Starter.hpp also
		// builds every pipeline with sampleShadingEnable = VK_TRUE and
		// minSampleShading = 1.0f, which turns MSAA into full supersampling:
		// the fragment shader runs once per SAMPLE, not once per pixel. At 16
		// samples CookTorrance.frag was running 16 times per pixel, each time
		// looping over every light with a full GGX evaluation. Measured on the
		// Iris Xe this machine has: 25 FPS at 16 samples, 94 at 4, 175+ at 2.
		//
		// 4 is the compromise: still genuinely good antialiasing (and, with
		// per-sample shading forced on, better than 4x MSAA normally is),
		// for a bit under a quarter of the fragment cost.
		//
		// Assigning it here is legal without touching Starter.hpp: msaaSamples
		// is a protected member of BaseProject, and pickPhysicalDevice() (which
		// sets the default) runs earlier in initVulkan() than localInit() does.
		//
		// Now also changeable live via the "MSAA" slider (see its addSlider()
		// call below) for the same reason renderScale's is: rather than
		// guessing at this compromise, try it on the actual machine it's
		// running on.
		msaaSamples = VK_SAMPLE_COUNT_4_BIT;
		// The slider's own upper bound: the device's real cap
		// (getMaxUsableSampleCount(), queried once at startup by
		// pickPhysicalDevice()), not an assumed 16 -- a GPU that tops out
		// lower must not be offered a sample count it can't actually create
		// an image at.
		maxMsaaLevel = std::log2((float)getMaxUsableSampleCount());

		initRenderPasses();

		// The 2D shadow render passes -- the sun's today, in
		// LightData::shadowIndex order (see SceneLights::init). AT_DEPTH_ONLY
		// is a stock configuration built for exactly this: a D32_SFLOAT
		// attachment usable both as a depth target and, after
		// ATDEP_DEPTH_TRANS's barrier, as a sampled texture. initSampler=true
		// (the last argument) is what makes attachments[0].getViewAndSampler()
		// below valid -- without it there's no VkSampler to hand back.
		//
		// .create() runs right here rather than in
		// pipelinesAndDescriptorSetsInit() (where RP/P are created) for two
		// reasons: these don't need to survive a resize the way the
		// swapchain-sized passes do, and PRs[0].init() below needs the actual
		// VkImageView+sampler to exist already, to bind them into every
		// CookTorrance instance's shadow-sampling descriptor set.
		for(int i = 0; i < NUM_SHADOW_MAPS_2D; i++) {
			RPShadow2D[i].init(this, SHADOW_MAP_RES, SHADOW_MAP_RES, -1,
							  RenderPass::getStandardAttchmentsProperties(AT_DEPTH_ONLY, this),
							  RenderPass::getStandardDependencies(ATDEP_DEPTH_TRANS),
							  true);
			RPShadow2D[i].create();
		}

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
		// Mirrors ATDEP_DEPTH_TRANS (Starter.hpp), but for a COLOR attachment
		// going to COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
		// instead of a depth one: external->0 waits for nothing but puts the
		// image into COLOR_ATTACHMENT_OPTIMAL before the subpass writes it,
		// 0->external makes the main pass's fragment-shader read wait for
		// that write to finish.
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

		P.init(this, &VD, "shaders/PosNormUV.vert.spv",
						  "shaders/CookTorrance.frag.spv",
						  {&DSLglobal, &DSLlocal, &DSLshadowSample});

		// The ghosts. Two sets, not three: Spectral.frag samples no shadow map
		// (it is unlit -- see its header), so DSLshadowSample is left off the
		// layout entirely rather than bound and ignored. updateUniformBuffer()
		// already keys the shadow-set mapping off inst.NDs[0] >= 3, so the
		// shorter layout needs no special case there.
		Pspectral.init(this, &VD, "shaders/PosNormUV.vert.spv",
							  "shaders/Spectral.frag.spv",
							  {&DSLglobal, &DSLlocal});
		// Alpha blending: srcAlpha * src + (1 - srcAlpha) * dst, which is what
		// Starter.hpp's transparent path sets up. This is the flag the whole
		// effect hangs off, and it is why the ghosts need a pipeline of their
		// own -- blending is baked into a VkPipeline, not selectable per draw.
		Pspectral.setTransparency(true);
		// BACK-FACE CULLING KEPT ON, deliberately, where every other transparent
		// thing in this project (Flame, ExitGlow) turns it off. Those are flat
		// billboards with no back to cull; the ghost is a closed mesh, and with
		// depthWriteEnable hardcoded to VK_TRUE in Starter.hpp (see Flame.hpp's
		// createMesh() for the same constraint) drawing both faces would mean
		// blending two layers whose order nothing sorts -- the near one would
		// write depth and reject the far one per fragment, so the interior would
		// appear or vanish depending on which way the ghost happened to face.
		// One layer per pixel has no order to get wrong, and Spectral.frag's
		// Fresnel rim is what supplies the volume that the second layer would
		// otherwise have given.
		//
		// LESS_OR_EQUAL, and with the prepass below in front of it this is now
		// load-bearing rather than defensive: the prepass leaves the nearest
		// ghost depth in the buffer, and EQUAL is the comparison that lets
		// exactly the fragments which produced it through. Under a strict LESS
		// the ghost would vanish entirely.
		Pspectral.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

		// The prepass. Everything about it matches Pspectral except the shader
		// and the compare op -- see the member declaration, and
		// SpectralDepth.frag for what it is for. Transparency on for the blend,
		// which is how it avoids writing colour: it emits alpha 0, so
		// srcAlpha * src + (1 - srcAlpha) * dst returns dst.
		PspectralDepth.init(this, &VD, "shaders/PosNormUV.vert.spv",
								   "shaders/SpectralDepth.frag.spv",
								   {&DSLglobal, &DSLlocal});
		PspectralDepth.setTransparency(true);

		// The post-processing passes. Two set layouts, differing only in how
		// many textures they read: one for the passes that transform a single
		// image, one for the composite, which has to mix two.
		//
		// NOTE on the second number in a sampler binding: Starter.hpp reuses
		// `linkSize` as the INDEX into the image-info vector handed to
		// DescriptorSet::init, not as a byte size (see DescriptorSetLayout::
		// init). So binding 1 reads image 0 and binding 2 reads image 1.
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
		Pbright.init(this, &VDpost, "shaders/Post.vert.spv", "shaders/BloomBright.frag.spv",
					 {&DSLpost1});
		PblurH.init(this, &VDpost, "shaders/Post.vert.spv", "shaders/BloomBlur.frag.spv",
					{&DSLpost1});
		PblurV.init(this, &VDpost, "shaders/Post.vert.spv", "shaders/BloomBlur.frag.spv",
					{&DSLpost1});
		Pcomposite.init(this, &VDpost, "shaders/Post.vert.spv", "shaders/Composite.frag.spv",
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

		// The shadow pass's own pipeline (see the member declaration for why
		// one, shared, instead of one each). Its only set is DSLlocal -- the SAME
		// per-instance buffer the main pass's ubo.mMat comes from, reused
		// here at set 0 instead of set 1 to read Wm again for a different
		// projection; see Shadow.vert's header for why that's safe. The
		// light's own view-projection arrives separately, as a push constant,
		// since (unlike Wm) it never changes frame to frame.
		VkPushConstantRange shadowPushConstant{};
		shadowPushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		shadowPushConstant.offset = 0;
		shadowPushConstant.size = sizeof(glm::mat4);
		PShadow.init(this, &VD, "shaders/Shadow.vert.spv",
								"shaders/Shadow.frag.spv",
								{&DSLlocal}, {shadowPushConstant});
		// Created against RPShadow2D[0], but usable with all of them: they share
		// the identical AT_DEPTH_ONLY attachment layout, and Vulkan only requires
		// render-pass COMPATIBILITY (same attachment formats/samples/layouts)
		// between the render pass a pipeline was created with and the one
		// it's bound under at draw time, not the exact same object.
		PShadow.create(&RPShadow2D[0]);

		// The cube shadow pass's pipeline (torches). Same DSLlocal reuse as
		// PShadow, see Shadow.vert's header. Set 1 is DSLshadowCubeCapture,
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
		PShadowCube.init(this, &VD, "shaders/ShadowCube.vert.spv",
								"shaders/ShadowCube.frag.spv",
								{&DSLlocal, &DSLshadowCubeCapture}, {shadowCubeFacePushConstant});
		// FRONT faces culled, so each occluder records the side turned AWAY
		// from the torch. This is what lets the depth slack in
		// shadowFromCube() be nothing but floating-point noise.
		//
		// Shadow acne is a surface comparing against its own record: within a
		// texel that record varies by texelWorld * tan(incidence), so a lit
		// fragment can read as further from the light than the sample meant
		// to represent it. Every defence against it is slack of some kind --
		// a depth bias, a normal offset, a softening band -- and slack is
		// exactly what detaches a shadow from its caster: wherever an
		// occluder overhangs its own base the gap between it and the floor
		// grows slowly, so even a few millimetres of forgiveness spread into
		// centimetres of lit floor. Three rounds of tuning that number, in
		// two different files, moved the strip around without closing it.
		//
		// Culling front faces removes the premise instead. A surface facing
		// the light is not in the map at all, so it cannot fail a comparison
		// against itself, and there is no acne to buy off. The silhouette --
		// the set of directions the occluder covers, which is the only thing
		// that decides where the shadow falls -- is identical either way.
		//
		// The cost, and it is a real one: an occluder now leaks light by its
		// own THICKNESS, since what is recorded is its far side. That is
		// bounded by the geometry rather than by a constant, and every wall
		// and door leaf in this dungeon is far thicker than the slack this
		// replaces. The case to watch is the opposite one -- anything built
		// as a single flat face, which has no far side to record and so stops
		// casting from the side the light sees. See shadowFromCube().
		PShadowCube.setCullMode(VK_CULL_MODE_FRONT_BIT);
		// Created against RPShadowCubeCompat -- see that member's comment for
		// why it exists purely to be render-pass-compatible with the 36
		// manually built per-face framebuffers this pipeline actually draws
		// into.
		PShadowCube.create(&RPShadowCubeCompat);

		// sets the size of the Descriptor Set Pool (it MUST be done before loading the scene)
		// The four post-processing sets are counted in here too: one uniform
		// block each, and five sampled textures between them (one apiece for
		// the bright pass and the two blurs, two for the composite).
		// + NUM_SHADOW_CUBES: DSshadowCube[], one uniform block/set per torch
		// cube slot for the shadow capture pass (see its member comment).
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
		// above is. 2D maps first, then cube maps -- same order the binding
		// list above and CookTorrance.frag's declarations use.
		std::vector<TextureDefs> shadowMapDefs;
		for(int i = 0; i < NUM_SHADOW_MAPS_2D; i++) {
			shadowMapDefs.push_back({false, 0, RPShadow2D[i].attachments[0].getViewAndSampler()});
		}
		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			shadowMapDefs.push_back({false, 0,
				{cubeShadowSampler.getSampler(), torchCube[i].cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
		}

		// ORDER MATTERS, and it is the only thing keeping the ghosts readable:
		// Scene::populateCommandBuffer walks the techniques in the order they
		// are registered here, so registering "Spectral" second is what puts
		// every alpha-blended ghost after every opaque wall it can be seen
		// through. Blending is not commutative -- a ghost drawn first would be
		// composited against whatever was behind it at the time, which is the
		// clear colour, and the dungeon would then paint over it.
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

		// Same two entries as above minus the shadow maps, matching
		// Pspectral's two-set layout: the global set carries no textures, and
		// DSLlocal takes the instance's own albedo at texture slot 0 -- the
		// ghost's map, which Spectral.frag reads as a density mask rather than
		// as a colour.
		//
		// The pipeline named here is the DEPTH PREPASS, not Pspectral, and that
		// is the only way to get one pass between the walls and the ghosts:
		// Scene draws its techniques back to back with nothing to hook between
		// them, and the prepass must land after the dungeon (a ghost's depth
		// written before a wall behind it would reject that wall and leave a
		// ghost-shaped hole) and before the ghosts' own colour. So the slot
		// Scene owns goes to the prepass, and populateCommandBuffer() issues
		// the colour draws by hand right after SC.populateCommandBuffer()
		// returns. The descriptor sets Scene builds here serve both pipelines
		// unchanged -- they are built against the DSLs, which are identical.
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

		// Cache the floor's top Y once, for the no-clip under-the-map safety clamp below.
		// Falls back to 0.0f (this scene's actual floor height) if the scene has no
		// instance named "floor".
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
		// Replaces the generic padlock prompts on one door and marks it secret,
		// which are one call because they are one decision: a door that has its
		// own wording is a door that isn't a door, and a door that isn't a door
		// must not wear the focus aura either. See Door::promptReady and
		// Door::secret. `blocked` may be empty for the generic line.
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
		// The door at the player's back. They spawn at x = -33.5 facing +X and
		// this leaf sits at x = -36.883 in the hall's west wall, so it is the
		// first thing they see if they turn around and the only lock they can
		// meet before finding anything -- which is exactly why the chains go
		// here: a padlock teaches what a padlock is far better where the
		// player has no key yet and cannot try it.
		//
		// The models were built for the dl doors, which are approached from
		// the other side; the `true` on its addLockProp lines below is what
		// turns them round. See there.
		addDoor("dhDoorPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f, "iron", "iron key");
		// Second and third doors, gating the two new rooms (dl, dv) added east
		// of the antechamber. The full dc/dl boundary is two tiles wide, so it
		// took two hole-wall + leaf pairs, not one wall tile left solid next
		// to it -- a plain wall there would have blocked half the doorway
		// with no way through. Same leaf asset, same hinge geometry as the
		// first door, so the same promptOffset/openAngleDeg apply unchanged.
		addDoor("dlDoorPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f);
		// The other padlocked door. The northern of the pair, i.e.
		// the one the player walks straight into: they spawn at z = 29 facing
		// +X and this leaf sits at z = 28.779, while its twin is seven units
		// south. Which also means the lock costs them nothing if they'd rather
		// not look for the key -- the other doorway is open, and a chained
		// door with a way around it is the only kind that can't strand anyone.
		//
		// It takes "iron" like the hall door, and so do both pickups -- keys
		// in this level are interchangeable, which is the only honest rule
		// when both of them are the same mesh: nothing on screen could tell
		// the player which padlock wanted which, so no padlock is allowed to
		// care. What stays scarce is the COUNT: two keys, two chained doors
		// and the exit, and only the exit gives its key back.
		addDoor("dlDoorPanel2", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f, "iron", "iron key");
		// The way out, in the dv room's east wall -- the only door in the
		// castle that opens onto the outside, and the last thing between the
		// player and the win box (see gameplay.json's "exit", which no longer
		// checks a key of its own now that this door does).
		//
		// Where its wall came from, since scene.json can't carry a comment
		// saying so (Scene.hpp parses it strictly): that corner used to be
		// closed by dvCornSE, and a corner mesh has no hole to cut a doorway
		// into. So it is replaced by the two pieces that reproduce its two
		// arms exactly, one of which is a hole wall -- the same substitution
		// the dc/dl boundary already makes with dlDoor2 + dcWallS2 at one
		// shared translate. The two meshes do NOT share an origin convention
		// (a corner's arms run to local -Z, the straight and hole walls from
		// local 0 to +Z), so the numbers differ while the geometry doesn't:
		//   dvDoor  at [20.0, 26.4] rot 0    -> x 18.758..20,  z 26.4..33.6
		//   dvWallS at [20.0, 33.6] rot 270  -> x 12.8..20,    z 32.358..33.6
		// which is arm 1 and arm 2 of the old dvCornSE, unmoved. The leaf then
		// sits at the same offset from its hole wall that dlDoorPanel2 has
		// from dlDoor2, carried across through the two walls' differing yaw,
		// which puts the hinge on the jamb and the doorway centre at z 29.99
		// -- the middle of the arch, whose hole spans local z 2.47..4.71, i.e.
		// world z 28.87..31.11.
		//
		// Its hole wall carries no yaw, because it stands in an EAST wall
		// rather than a west-facing one, so the leaf carries none either --
		// and that flips which way its local +X points: outward, here, where
		// every other leaf in the castle has it pointing into the room. Hence
		// the flipped lock props below, which is what puts the chains on the
		// inside face and, with them, the side E works from. The promptOffset
		// is unchanged: the doorway centre sits at the same place in the
		// leaf's own frame no matter which way the frame points.
		//
		// It opens OUTWARD, into the daylight -- which is not a special case
		// any more but just what "away from the player" means for a leaf the
		// player can only reach from inside. For this leaf's mirrored frame
		// that is the NEGATIVE angle, which is what the sign below records.
		// The swing reaches x 21.78 at its widest, which is what sets where
		// the light behind it can stand: see EXIT_GLOW_CENTER, and the
		// second, ground-level quad it costs.
		addDoor("dvDoorPanel", glm::vec3(0.0f, 2.52f, -1.231f), -100.0f, "iron", "iron key");
		// The secret passage: a bookcase standing in the dl room's north wall,
		// where a plain wall tile (dlWallN2) used to be. It is a Door and
		// nothing else -- same hinge convention, same swing, same padlock rule
		// -- because it wants to behave exactly like one and the only thing
		// that differs is what it is paid with and what it says about itself.
		//
		// tools/make_bookshelf.py builds SM_Bookshelf_01 in SM_Door_01's own
		// local frame for that reason: origin on the hinge, panel hanging to
		// local -Z, and a silhouette that follows the hole wall's arch (the
		// carcass is rectangular to y 4.12, then an arched cap over it). A
		// rectangular case would have poked through the arch; a short one would
		// have left the lunette open and the secret room visible above the
		// books. It sits INSIDE the wall's thickness like every other leaf here
		// rather than in front of it, which is also what lets it swing either
		// way: a bookcase standing proud of the wall could only ever open into
		// the player's face.
		//
		// promptOffset X is 0.25 and not 0: the point the range check measures
		// from should be the shelf FACE the player is looking at, not the plane
		// of the hinge two thirds of the case's depth behind it.
		//
		// Locked with "book", which no other lock in the castle takes and which
		// the exit does not accept (gameplay.json's exit.keyId is "iron"), so
		// the one book in the level can only ever be spent here -- there is no
		// way to waste it and no way to strand a run on it.
		addDoor("dsShelfPanel", glm::vec3(0.25f, 2.20f, -1.231f), 100.0f, "book", "old book");
		// What the bookcase says, and when it says anything at all. Marking it
		// secret means both prompts below only ever appear with the book in
		// hand -- until then the shelf is furniture and does not glow (see
		// Door::secret), so "A book is missing from this shelf" is not a hint
		// that leads the player to the gap, it is the game agreeing with them
		// once they have already worked the gap out and gone and found the
		// book. Nothing here uses the word "locked" or names a key: a padlock
		// can afford to, because the player can see the padlock.
		//
		// No blocked line: it would be unreachable. The only way to stand on
		// the far side is to have opened the thing, and it never re-locks
		// within a run.
		setSecretDoor("dsShelfPanel",
					  "[E] Slide the book into the gap",
					  "A book is missing from this shelf");
		// Hangs a scene instance on a door as lock hardware. Separate from
		// addDoor() rather than another argument on it because a door can
		// carry several (the chains and the padlock are two models: the
		// loader allows one texture per file, and they want different ones --
		// door iron for the chains, the key's brass for the lock).
		//
		// These instances DO carry a "collider" in scene.json: the hardware hangs
		// 0.225 in front of the leaf face, so the leaf's own box leaves it in
		// open air. Safe only because the prop loop in GameLogic() syncs that box
		// off the prop's Wm -- a box left behind would seal an unlocked door.
		//
		// `flip` puts the hardware on the leaf's OTHER face. make_door_lock.py
		// builds chains and padlock against one face only (its FRONT_ON_PLUS_X),
		// so a door the player walks up to from the opposite side would show
		// them the bare leaf with the lock hidden behind it. A half turn about
		// the leaf's local vertical axis through the middle of its thickness
		// (x = (-0.037 + 0.430)/2) and the middle of the doorway (z = -1.231,
		// the same mid-plane promptOffset uses) lands the pieces on the far
		// face, and because both models are built symmetric about that same
		// z -- the padlock sits on it, the chain plates straddle it -- the
		// turned copy is the mirror image the script would have exported with
		// the flag the other way. A rotation and not an actual mirror matrix
		// on purpose: mirroring flips the winding, and the whole piece would
		// turn inside out under backface culling.
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

			// Grow the auto-fit box outward so the player is stopped before the
			// hardware is in the torch's reach. A box that hugs the mesh is not
			// enough: HAND_TUCK_MIN_REACH floors the tuck at 0.35, which still
			// leaves the grip 0.385 ahead of the eye plus the torch's own body,
			// against the 0.3 PLAYER_RADIUS holds the player off a collider. The
			// tuck is already at its floor, so distance is the only lever left.
			//
			// xMax alone, in MODEL space: the hardware is modelled entirely on
			// +X (chains 0.410..0.655, padlock 0.460..0.640, leaf ends at 0.430),
			// so `local`'s half turn carries mesh and margin to the far face
			// together. Inflating all six faces would push into the jambs and put
			// a keep-out on the bare face of the leaf.
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
			// works from, so the prompt can never disagree with what's on
			// screen. Every prop on one door is flipped the same way (they're
			// two halves of one lock), so the last one in wins harmlessly.
			d->lockFaceSign = flip ? -1.0f : 1.0f;
		};
		// Both instances carry the SAME translate/eulerAngles as the leaf in
		// scene.json, which is all the placement they need: the models live in
		// its local frame.
		addLockProp("dlDoorPanel2", "dlDoorChains2");
		addLockProp("dlDoorPanel2", "dlDoorPadlock2");
		// The hall door's set, flipped: the dl leaves are reached from the
		// west and this one from the east, and all three carry the same
		// eulerAngles, so the models as exported would hang on the side the
		// player never stands on.
		addLockProp("dhDoorPanel", "dhDoorChains", true);
		addLockProp("dhDoorPanel", "dhDoorPadlock", true);
		// The exit door's set, also flipped, but for the opposite reason to
		// dhDoorPanel's: that leaf is yawed 180 and approached from the east,
		// this one is yawed 0 and approached from the west. Either way the
		// models as exported end up on the face the player never stands on,
		// and either way the half turn is what fixes it. See the addDoor call
		// for this leaf above.
		addLockProp("dvDoorPanel", "dvDoorChains", true);
		addLockProp("dvDoorPanel", "dvDoorPadlock", true);

		// Cache which door is the way out, so the light outside can be driven
		// from its swing without a string compare every frame. Done here
		// rather than in the addDoor lambda because "which door is the exit"
		// is a property of the level, not of doors in general.
		for(size_t i = 0; i < doors.size(); i++) {
			if(doors[i].instanceId == "dvDoorPanel") {
				exitDoorIndex = (int)i;
				break;
			}
		}
		if(exitDoorIndex < 0) {
			std::cout << "Exit door 'dvDoorPanel' not found: no daylight outside it\n";
		}

		// dlDoorPanel, the southern leaf of the pair, is left unlocked on
		// purpose: it is the way around its chained twin. Two keys exist, both
		// are consumed on use and the exit needs one still on the ring, so
		// exactly one of the two padlocks can be paid for -- leaving the pair
		// half open is what keeps that from being a trap.

		// World pickups. worldPos is read from the instance's own Wm, since
		// dhKey's position already lives in scene.json and shouldn't be
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
			// Read the uniform scale straight out of the authored world
			// matrix -- column 0's length, which is the scale for any
			// instance scaled uniformly, rotated or not. This is the single
			// read that makes scene.json's "scale" the only place that number
			// has to live: the held and dropped poses rebuild the matrix from
			// scratch and take it back from here.
			p.worldScale = glm::length(glm::vec3(p.inst->Wm[0]));
			pickups.push_back(p);
		};
		// Three keys, and deliberately the SAME id: all three are the one key
		// mesh in the level, so a lock that accepted one and refused another
		// would read as a bug no matter how correct the rule was. Sharing an
		// id is what keyRing is built for (see its declaration) -- the ring
		// stores instances, not ids, so three "iron" keys are still three
		// distinct objects to pick up, drop and spend.
		// The scarcity is therefore arithmetic, not matching: three keys
		// against three padlocks, all of them consuming what they are paid
		// with (the exit no longer checks a key of its own -- its door does).
		// Two of those padlocks are optional, so the budget survives a player
		// spending both spares, and only just: waste all three and the way out
		// stays shut.
		addPickup("dhKey", "iron");
		addPickup("dcKey", "iron");
		// The third, in the coloured-torch room, added when the exit stopped
		// being a free box and became a chained door. See scene.json's dlKey.
		addPickup("dlKey", "iron");
		// The fourth pickup is not a key, or rather it is a key that does not
		// look like one: the book on the hall table, which opens the bookcase
		// three rooms east and nothing else. It rides the SAME machinery as the
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
		addPickup("dhBook", "book", glm::vec3(84.0f, -24.0f, 0.0f),
				  HAND_KEY_OFFSET + glm::vec3(0.0f, 0.16f, 0.0f));

		// Where the book ends up once it has been spent: standing in the gap on
		// the bookcase's third shelf, riding the leaf's local frame like the
		// padlocks do, so it swings with the case instead of hanging in the
		// doorway. See Door::LockProp::whenUnlocked.
		//
		// The three numbers are the fessura's, and they live in TWO places by
		// necessity: here, and in make_bookshelf.py's BOOK_SLOT_* which is what
		// actually leaves the gap in the row of books. Move one and move the
		// other.
		//   x 0.454  the spine plane of the surrounding books (0.470) minus the
		//            book's own 0.016 of spine bulge, so the new volume lines up
		//            with its neighbours instead of standing proud of them
		//   y 1.730  the shelf's face (1.56) plus half the book's height, since
		//            the mesh is centred on Z and Z is what becomes "up" here
		//   z -1.266 the 0.440-wide gap (-1.451..-1.011) minus the book's 0.070
		//            of thickness, halved: centred in the hole it was left.
		//            The gap is five or six volumes wide and sits on the case's
		//            own centre line, so this number does not move if it is
		//            widened again -- it is the centre line either way. Anything
		//            narrower read as just another seam between spines from
		//            across the room; see the note on BOOK_SLOT_W in
		//            make_bookshelf.py. The book does not fill the gap it goes
		//            into, and that is the intended trade: the case swings open
		//            the moment it lands, so the only frame anyone reads is the
		//            one before.
		//
		// The rotation is the one that stands a flat-lying book on a shelf with
		// its spine out: book x -> leaf -x (spine at the front, fore-edge going
		// back into the case), book y -> leaf z (thickness along the shelf), book
		// z -> leaf y (the page's height becomes the height). That is exactly
		// rotY(180) * rotX(-90), and it is a rotation and not a mirror -- a
		// mirror would flip the winding and turn the book inside out under
		// backface culling.
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
			// The scale comes from the pickup's own authored matrix, same
			// single read scene.json's "scale" feeds everywhere else, so a
			// resized book still lands in its gap at the size it has in the
			// world.
			d->lockProps.push_back({p->inst,
									local * glm::scale(glm::mat4(1.0f), glm::vec3(p->worldScale)),
									true});
		};
		addSlotProp("dsShelfPanel", "dhBook",
					glm::translate(glm::mat4(1.0f), glm::vec3(0.454f, 1.730f, -1.266f))
				  * glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f))
				  * glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)));

		// The player's spawn pose, captured before anything can move it. See
		// spawnPos's declaration: this is what restartRun() puts them back to.
		spawnPos = camPos;
		spawnYaw = camYaw;
		spawnPitch = camPitch;

		// The rules of the game: the hunt cycle's timings, where the run is
		// won, and the ghosts' patrols. All three used to be constants in this
		// file (the ghost's waypoint loop was written out right here); they're
		// a data file now for the same reason lights.json is one, which is that
		// every number in it is a tuning decision someone will want to change
		// without waiting for a rebuild.
		{
			std::ifstream ifs("assets/scenes/gameplay.json");
			if(!ifs.is_open()) {
				std::cout << "gameplay.json not found: default hunt timings, no ghosts\n";
				// Still has to be armed: a default-constructed HuntCycle has a
				// phase timer of 0 and would fall straight into a hunt on the
				// first frame. init() with no overrides is exactly "defaults,
				// then reset()".
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

				// Fit the ghost's collision size from the actual mesh instead of
				// hand-measuring it once and hoping nobody replaces the model.
				// Every ghost instance shares the same Ghost.gltf, so one fit
				// off the first one covers all of them.
				if(!ghosts.empty()) {
					Collider fit;
					fit.fitAABB(SC.M[ghosts[0].inst->Mid]);
					AABBextents E = fit.getExtents();	// fit's Wm is identity, so this is local space

					// fitAABB reads the raw mesh, which knows nothing about
					// scene.json's "scale" on the instance -- so a scaled-down
					// ghost would otherwise keep a full-size collider. Same
					// trick as Pickup::worldScale above: the length of the world
					// matrix's first column IS the instance's uniform scale
					// factor, so multiplying it in here is what makes shrinking
					// a ghost in scene.json actually shrink what it collides
					// as, not just what it looks like.
					float instScale = glm::length(glm::vec3(ghosts[0].inst->Wm[0]));

					ghostBodyBottom = E.yMin * instScale;
					ghostBodyTop = E.yMax * instScale;

					float halfX = 0.5f * (E.xMax - E.xMin);
					float halfZ = 0.5f * (E.zMax - E.zMin);
					ghostRadius = ghostXZFitShrink * 0.5f * (halfX + halfZ) * instScale;

					std::cout << "Ghost collision fitted from mesh (scale " << instScale
							  << "): radius " << ghostRadius
							  << ", vertical [" << ghostBodyBottom << ", " << ghostBodyTop << "]\n";

					// Patrol legs are walked without any wall resolution (see the
					// Patrol branch in GameLogic), so a leg authored through a wall
					// doesn't fail loudly: the ghost simply glides through it, and
					// only the next hunt shows the damage, as a Chase that finds
					// every heading blocked and stands still. Checked here, once,
					// now that the radius is known -- it costs nothing at load time
					// and turns that into a line anyone can read.
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

		// Held torch. Starts on the FLOOR at its authored scene.json pose
		// (translate/eulerAngles/scale): the player walks up to it and picks
		// it up with [E] (see nearbyHandTorch in GameLogic). Only once
		// handTorchCollected is set does GameLogic start rebuilding its Wm
		// from the camera every frame -- until then the authored pose is what
		// shows, so it's captured here before anything can overwrite it.
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
		// maxInstances raised past the default 8: held torch + 6 wall
		// torches + 4 colored dl torches + 2 candles is already 13, and
		// spawn() past this cap fails silently (see addTorchFlame below),
		// leaving a torch/candle mesh with no fire and no light instead of
		// an error.
		flame.init(this, &DSLglobal, &DSglobal, 16);

		// No DSLglobal/DSglobal here: the daylight quads aren't shaded and read
		// nothing the app-wide uniform carries, so they bind sets of their
		// own. Two of them -- the upright wall of light and the ground it
		// stands on. See ExitGlow.hpp, and EXIT_GLOW_CENTER for why the
		// outward-swinging door makes the second one necessary.
		exitGlow.init(this, EXIT_GLOW_COUNT);

		debugLines.init(this);

		// seed just spreads each flame's sway/flicker phase (see Flame.hpp),
		// not a real RNG: index * a large-ish irrational-ish constant keeps
		// them decorrelated without needing a seeded generator for one call.
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
			// An irrational-ish stride rather than a round number, so the
			// per-torch phases never land on a common multiple and start
			// flickering in step.
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
			// Static flames only: the held torch's anchor is rebuilt from the
			// camera every frame, so a world position captured here would be
			// wrong from the next one on. Nothing reads it for the held torch.
			if(!heldByCamera) {
				tf.anchorWorld = glm::vec3(inst->Wm * glm::vec4(anchor, 1.0f));
			}
			// Automatic, not a parameter: every flame this project has is
			// either the one held torch or a static object, and every
			// static one belongs in the dynamic shadow pool -- there is no
			// third kind, so a caller can never forget to opt a new torch/
			// candle in. See the field comment for what "candidate" means.
			tf.shadowCandidate = !heldByCamera;
			if(tf.shadowCandidate) {
				// Static object, so its face matrices are computed once
				// here, the same way computeShadowMatrices() used to for
				// the lights.json torches, rather than every frame like the
				// held torch.
				tf.shadowFaceMatrices = cubeFaceMatricesFor(tf.anchorWorld);
			}
			// Offsets this torch into a different part of the CPU noise field,
			// so no two gutter at the same moment. Scaled up because fireNoise
			// hashes on the integer lattice: a fractional offset would leave
			// neighbouring torches sampling the same two lattice points.
			tf.phase = seed * 37.0f;
			torchFlames.push_back(tf);
		};

		if(handTorchInst != nullptr) {
			// Spawned UNLIT: the player lights it from a burning wall torch
			// with [E] (see nearbyWallTorch in GameLogic). handFlameIdx is the
			// slot it lands in, captured before the push.
			handFlameIdx = (int)torchFlames.size();
			addTorchFlame("handTorch", TORCH_FLAME_ANCHOR, true,
						  TORCH_LIGHT_COLOR, 1.0f, 1.0f, /*isCandle=*/false,
						  /*burning=*/false);
		}

		// Every OTHER flame in the scene: read from flames.json instead of
		// listed here by instance id, so a level's torches/candles get fire
		// automatically just by using the standard meshes, with no main.cpp
		// change and no per-instance authoring -- see the file's own header
		// for why it's keyed by model. "handTorch" above is the one
		// exception, spawned by literal id: it's a gameplay singleton, not
		// level dressing.
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
					// Authored burning state. True for anything that doesn't
					// say otherwise, so every torch in the castle is alight at
					// load exactly as before; the candles set it false and
					// wait for the player (see TorchFlame::burning).
					bool burning = true;
				};
				// Same "only overwrite what's present" shape as
				// SceneLights::readVec3, so a def can start from another
				// def's values (a model's defaults, for an override to
				// build on) instead of always starting from scratch.
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

				// Resolve each listed model NAME to its Mid once (same
				// pattern as SceneMaterials.hpp), so spawning below is an
				// O(1) lookup per instance instead of a string compare
				// against every model name.
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

		// Everything past lights.json's own fixed slots (activeCubeShadows,
		// right now 6) and before the held torch's reserved last one is the
		// dynamic pool: captured here, once, rather than as a literal 6, so
		// this stays correct if lights.json's own torch count ever changes.
		// dynamicSlotOccupant starts empty; updateDynamicShadowSlots() fills
		// it in on the first updateUniformBuffer() call (shadowReassignTimer
		// starts already due, see its member comment).
		dynamicShadowSlotBase = activeCubeShadows;
		dynamicSlotOccupant.fill(-1);
		// Nothing has been rendered yet -- the very first updateUniformBuffer()
		// call's diff (see lastRenderedOccupant's member comment) queues every
		// fixed slot plus whatever updateDynamicShadowSlots() assigns on that
		// same tick (shadowReassignTimer already starts due) for its one and
		// only render.
		lastRenderedOccupant.fill(SHADOW_SLOT_UNSET);
		// The render loop below (populateCommandBuffer()) walks slots
		// [0, activeCubeShadows) every frame, so the dynamic pool has to be
		// counted in even before anything occupies it -- an empty dynamic
		// slot still needs to run its (harmless, nothing samples it) capture
		// pass, or a slot filled mid-game would never get rendered at all.
		activeCubeShadows = std::max(activeCubeShadows, HAND_TORCH_SHADOW_INDEX);

		// The held torch's cube slot isn't in sceneLights.all() (it's not in
		// lights.json), so computeShadowMatrices() never counts it into
		// activeCubeShadows. Its face matrices get filled in every frame by
		// updateHandTorchShadow() instead, but the cube shadow render loop
		// (populateCommandBuffer()) still needs to know HAND_TORCH_SHADOW_INDEX
		// is in play at all -- done once here, since whether the held torch
		// exists doesn't change after startup.
		if(handTorchInst != nullptr) {
			activeCubeShadows = std::max(activeCubeShadows, HAND_TORCH_SHADOW_INDEX + 1);
		}

		// initializes the textual output
		txt.init(this, windowWidth, windowHeight);
		// initializes the flat-quad background/highlight layer for the cheat HUD
		uiQuad.init(this, windowWidth, windowHeight);
		// initializes the always-on center-screen crosshair dot; distinct
		// submitOrder/buffer name from uiQuad above so the two don't collide
		// over the same named command buffer
		crosshair.init(this, windowWidth, windowHeight, 9002, "crosshair");
		setCrosshairQuad();
		// initializes the flat-quad dim overlay/button layer for the pause menu;
		// distinct submitOrder/buffer name for the same reason as crosshair above
		pauseQuad.init(this, windowWidth, windowHeight, 9003, "pause_quad");
		// initializes the flat-quad opaque backdrop/button layer for the
		// launch screen; distinct submitOrder/buffer name for the same
		// reason as crosshair/pauseQuad above
		startScreenQuad.init(this, windowWidth, windowHeight, 9004, "start_screen_quad");

		// submits the main command buffer
		submitCommandBuffer("main", 0, populateCommandBufferAccess, this);

		// Prepares for showing the FPS count. Left showing (on top of the
		// launch screen too) rather than hidden behind it: this block's text
		// (id 1) is never removed for the rest of the run, which is also
		// what keeps TextMaker's live block list from ever going fully
		// empty. TextMaker::createTextMesh() (skeleton code: see Libs.cpp's
		// "must not be modified" line, so this isn't ours to patch) does
		// M->vertices[0] unconditionally with no guard for zero total
		// characters across every live block -- an earlier version of this
		// code removed this text while the launch screen was open, and right
		// after Play closed it every other block (coordinates, interact
		// prompt, hunt/end banners, the menus) could also be momentarily
		// unset at once, crashing the very next updateCommandBuffer() call.
		txt.print(1.0f, 1.0f, "FPS:",1,"CO",false,false,true,TAL_RIGHT,TRH_RIGHT,TRV_BOTTOM,{1.0f,0.0f,0.0f,1.0f},{0.8f,0.8f,0.0f,1.0f});

		// Wires the cheat HUD to the actual cheat flags, so toggling a row
		// in the menu flips the exact same bools GameLogic() reads.
		hud.init(&txt, &uiQuad);
		pauseMenu.init(&txt, &pauseQuad);
		// windowTitle is set in setWindowParameters(), well before this runs
		// -- reused as-is so the launch screen's title only needs changing
		// in one place.
		startScreen.init(&txt, &startScreenQuad, windowTitle);
		startScreen.setOpen(true, windowWidth, windowHeight);
		hud.addToggle("Collision", &cheats.collisionEnabled);
		hud.addToggle("Show Coordinates", &cheats.showCoordinates);

		// Gameplay rows. "Hunt" forces the cycle into its hunt phase and holds
		// it there, so the mechanic can be watched without waiting out
		// calmDuration; "Ghosts Can Catch" is the row that lets you watch it
		// for longer than the first ghost takes to reach you.
		hud.addToggle("Hunt", &huntCycle.forceHunt);
		hud.addToggle("Ghosts Can Catch", &cheats.ghostsCanCatch);

		// Lighting rows. Listed after the movement ones and in the order you'd
		// use them: first which sources are on, then how they're being shaded.
		// "Spotlight"/"Ambient Light" point straight into sceneLights, which
		// owns those lights (see SceneLights.hpp); "Torches"/"Holding
		// Torch" into cheats, because the flames' point lights never go
		// through SceneLights at all (see CheatFlags and flameBurning()); the rest
		// into cheats too, where they become gubo.debugFlags.
		//
		// No "Sun" row: the scene has no direct light any more (lights.json), so
		// the toggle had nothing to switch. sceneLights.directEnabled is still
		// there and still filters LIGHT_DIRECT in update(), so pasting the sun
		// back into lights.json only needs this line back with it.
		hud.addToggle("Torches", &cheats.roomTorchesEnabled);
		hud.addToggle("Holding Torch", &cheats.handTorchEnabled);
		hud.addToggle("Spotlight", &sceneLights.spotEnabled);
		hud.addToggle("Ambient Light", &sceneLights.ambientEnabled);
		hud.addToggle("Shadows", &cheats.shadowsEnabled);
		hud.addToggle("Torch Shadows", &cheats.torchShadowsEnabled);
		hud.addToggle("Candle Shadows", &cheats.candleShadowsEnabled);
		hud.addToggle("Specular", &cheats.specularEnabled);
		hud.addToggle("Tone Mapping", &cheats.toneMapEnabled);
		hud.addToggle("Fullbright", &cheats.unlit);
		hud.addToggle("Show Normals", &cheats.showNormals);
		hud.addToggle("Focus Glow", &cheats.focusGlowEnabled);
		hud.addToggle("Light Gizmos", &cheats.showLightGizmos);
		hud.addToggle("Shadow Frustums", &cheats.showShadowFrustums);
		hud.addToggle("Show Colliders", &cheats.showColliders);
		hud.addToggle("Light Heatmap", &cheats.showLightHeatmap);

		// Render Scale: see renderScale's own declaration/comment above for
		// what this actually resizes. onChange replays the same rebuild path
		// a real window resize already goes through (see
		// framebufferResizeCallback/onWindowResize in Starter.hpp) at the
		// CURRENT window size, so only the internal render resolution
		// changes, nothing about the window itself. Floor of 0.4 (40%, i.e.
		// 16% of the pixel count): below that the upscale reliably reads as
		// blurry rather than atmospheric even with fog/vignette/bloom all
		// helping hide it, so there's little reason to let the slider go
		// lower than the point it stops being a useful comparison. 0.05 per
		// press (Left/Right on the selected row) gives 12 steps across the
		// full range -- fine enough to feel the difference between two
		// adjacent presses without needing dozens of them to cross the range.
		hud.addSlider("Render Scale", &renderScale, 0.4f, 1.0f, 0.05f, [this]() {
			// Skipped while a rebuild (this one, a previous slider press, or
			// an actual window resize) is still pending -- see
			// framebufferResized's own comment in Starter.hpp and
			// RebuildPipeline()'s. recreateSwapChain() only runs once, at
			// the very end of the CURRENT frame's drawFrame(); stacking a
			// second target size on top before that has happened is what
			// let a render pass get begun against a size newer than the
			// framebuffer it was actually bound to, which is what the
			// "renderArea... greater than framebuffer" validation errors
			// (and the crash that followed them) were. This can't fully
			// rule out the same race from resizing the WINDOW itself very
			// rapidly, since that path lives in the immutable Starter.hpp
			// and isn't something this guard touches -- but it stops our
			// own sliders from being an extra source of the same pileup.
			if(!framebufferResized) {
				onWindowResize((int)windowWidth, (int)windowHeight);
				RebuildPipeline();
			}
		}, [this](float /*scale*/) {
			// Shows the actual pixel resolution alongside the percentage
			// (e.g. "< 80% (1536x864) >") rather than switching to fixed
			// presets like 720p/1080p: those only mean one specific shape
			// (16:9) at one specific window size, while this scale has to
			// stay meaningful at whatever size/shape the window is
			// resized to. Reads renderWidth()/renderHeight() -- which
			// read the LIVE renderScale, not the parameter -- rather than
			// recomputing from scratch, so this can never drift from
			// what's actually being rendered.
			char buf[32];
			snprintf(buf, sizeof(buf), "< %d%% (%dx%d) >",
					 (int)std::lround(renderScale * 100.0f),
					 renderWidth((int)windowWidth), renderHeight((int)windowHeight));
			return std::string(buf);
		});

		// MSAA: see msaaLevel's own declaration for what the units are and
		// why (a log2 level, not the raw sample count -- a plain additive
		// slider step can't land on 1/2/4/8/16 otherwise). onChange converts
		// the level back to a real VkSampleCountFlagBits, then rebuilds the
		// render passes' attachment properties around it (initRenderPasses(),
		// since a sample-count change -- unlike renderScale's plain
		// width/height change -- has to regenerate hdrAtt itself) before
		// tearing down and recreating the actual GPU images/pipelines
		// (RebuildPipeline()). maxMsaaLevel: this device's real cap, set in
		// localInit() from getMaxUsableSampleCount(). Step 1.0 moves exactly
		// one power of two per press.
		hud.addSlider("MSAA", &msaaLevel, 0.0f, maxMsaaLevel, 1.0f, [this]() {
			// Same guard as Render Scale's onChange above, same reason.
			if(!framebufferResized) {
				msaaSamples = static_cast<VkSampleCountFlagBits>(1 << (int)std::lround(msaaLevel));
				initRenderPasses();
				RebuildPipeline();
			}
		}, [](float level) {
			char buf[16];
			snprintf(buf, sizeof(buf), "< %dx >", 1 << (int)std::lround(level));
			return std::string(buf);
		});
	}

	// Six 90-degree perspective faces covering a point light's whole sphere,
	// axis-aligned on world X/Y/Z (CUBE_FACE_DIR/CUBE_FACE_UP,
	// CubeShadowMap.hpp) -- the same math every point-light shadow camera in
	// this file needs, factored out so the six lights.json torches
	// (computeShadowMatrices(), once), the held torch
	// (updateHandTorchShadow(), every frame) and the dynamic shadow
	// candidates (addTorchFlame(), once each, they're static objects too)
	// share one implementation instead of three copies that could drift.
	// Does slot t currently have a light in it, i.e. is anything sampling its
	// cube map? One definition, because "occupied" means three different things
	// depending on which part of the slot range t falls in: a lights.json slot
	// is occupied by construction, a dynamic-pool one only while
	// updateDynamicShadowSlots() has given it a flame, and the held torch's
	// only while the torch itself exists (activeCubeShadows covers it).
	bool cubeSlotOccupied(int t) const {
		if(t == HAND_TORCH_SHADOW_INDEX) {
			return HAND_TORCH_SHADOW_INDEX < activeCubeShadows && cheats.torchShadowsEnabled;
		}
		return (t < dynamicShadowSlotBase) || (dynamicSlotOccupant[t] != -1);
	}

	// The g (falloff reference distance) a flame's point light is uploaded
	// with. One definition, called both by the light-append loop in
	// updateUniformBuffer() and by the shadow-reach computation, so what the
	// shadow logic believes about a light's size cannot drift from what the
	// shader is actually given.
	static float flameLightG(bool isCandle, float intensity) {
		return TORCH_LIGHT_G * (isCandle ? CANDLE_LIGHT_G_SCALE : 1.0f)
			   * (0.88f + 0.12f * intensity);
	}

	// How far from a point light a moving occluder can still cast a shadow
	// worth capturing.
	//
	// Not a hand-picked radius: CookTorrance.frag attenuates a point light by
	// (g/d)^beta, so solving (g/d)^beta = SHADOW_REACH_CUTOFF for d gives the
	// distance at which that specific light is down to a cutoff fraction of its
	// peak. Past it the light contributes less than the cutoff, so whether its
	// shadow map has the ghost in it or not is a difference nothing can
	// resolve on screen. A light with a bigger g gets a proportionally bigger
	// answer, which is what makes this correct for the candles (g scaled by
	// CANDLE_LIGHT_G_SCALE) as well as the torches, with no second number to
	// keep in step.
	//
	// Clamped to the cube's own far plane, past which the map records nothing
	// whatever the falloff says.
	static float shadowRelevantReach(float g, float beta) {
		const float d = g * std::pow(SHADOW_REACH_CUTOFF, -1.0f / beta);
		return std::min(d, TORCH_SHADOW_FAR_CONST);
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
	// right after sceneLights.init(): the sun and the torches never move, so
	// there is nothing here that needs recomputing per frame.
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

		// The sun has no position, only a travel direction (SceneLights.hpp),
		// so its shadow camera needs a stand-in position: back away from a
		// point roughly at the middle of the playable area, far enough that
		// an orthographic box this big (SUN_ORTHO_HALF_EXTENT) covers both
		// the castle courtyard and the dungeon under it. Hand-picked by
		// looking at the instance coordinates in scene.json, the same way
		// the collider and light offsets there were -- not derived from
		// anything, and the first thing to revisit if the shadow clips.
		const glm::vec3 SUN_TARGET(-8.0f, 0.0f, 15.0f);
		const float SUN_ORTHO_HALF_EXTENT = 55.0f;
		const float SUN_DISTANCE = 80.0f;

		for(const LightData &L : sceneLights.all()) {
			if(L.shadowIndex < 0) {
				continue;
			}

			if(L.type == LIGHT_DIRECT) {
				glm::vec3 pos = SUN_TARGET - L.dir * SUN_DISTANCE;
				glm::mat4 view = glm::lookAt(pos, SUN_TARGET, glm::vec3(0.0f, 1.0f, 0.0f));
				glm::mat4 proj = glm::ortho(-SUN_ORTHO_HALF_EXTENT, SUN_ORTHO_HALF_EXTENT,
											-SUN_ORTHO_HALF_EXTENT, SUN_ORTHO_HALF_EXTENT,
											1.0f, 200.0f);
				// Same Vulkan Y-flip as the main camera's projection (see
				// View/ViewPrj in GameLogic()); GLM assumes an OpenGL-handed
				// NDC otherwise.
				proj[1][1] *= -1;
				shadowLightSpace2D[L.shadowIndex] = proj * view;
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

	// Same cubeFaceMatricesFor() every point-light shadow camera uses, but
	// for HAND_TORCH_SHADOW_INDEX and called every frame from
	// updateUniformBuffer() instead of once from localInit(): the held torch's
	// world position moves with the camera, so its face matrices can't be
	// baked once like the static torches'/candidates' can. Runs before
	// populateCommandBuffer() records this frame's cube shadow passes
	// (updateUniformBuffer() precedes updateCommandBuffers() in Starter.hpp's
	// drawFrame()), so the push constants that pass reads are already current.
	void updateHandTorchShadow(const glm::vec3 &lightPos) {
		std::array<glm::mat4, 6> faces = cubeFaceMatricesFor(lightPos);
		for(int face = 0; face < 6; face++) {
			torchFaceMatrices[HAND_TORCH_SHADOW_INDEX][face] = faces[face];
		}
		torchLightPos[HAND_TORCH_SHADOW_INDEX] = lightPos;
		// Same reach every other torch gets, so the mover test treats this slot
		// like the rest of them (see queueMoverCubeSlotRenders): a torch is a
		// torch whether it's on a wall or in the player's hand.
		torchShadowReach[HAND_TORCH_SHADOW_INDEX] =
				shadowRelevantReach(flameLightG(false, 1.0f), TORCH_LIGHT_BETA);
	}

	// Priority-based reassignment for the dynamic shadow-cube pool
	// (dynamicShadowSlotBase..HAND_TORCH_SHADOW_INDEX), among every flame
	// marked shadowCandidate (every torch/candle but the held one): the
	// empty-slot pass below always gives one to every candidate first, no
	// contest involved, so in THIS scene (12 candidates against 31 dynamic
	// slots) every one of them keeps a real shadow permanently -- the
	// contest logic below only fires past that point, i.e. only on a level
	// authored with more shadow-worthy point lights than there are spare
	// cube slots, re-evaluated as the player moves so ones left behind lose
	// theirs to ones now more worth having.
	//
	// A plain "N nearest, recomputed every tick" rule thrashes: standing
	// near the distance boundary between two candidates flips the slot every
	// time position noise crosses it, and every flip forces a fresh shadow
	// render since there's no cross-fade between "has a shadow" and
	// "doesn't". SHADOW_SWAP_MARGIN fixes that -- a waiting candidate only
	// takes a slot from its current occupant if it's genuinely closer, not
	// marginally closer -- and SHADOW_REASSIGN_INTERVAL (called from
	// updateUniformBuffer(), not every frame) means the decision itself is
	// only revisited a few times a second: the candidates are static, only
	// the player moves, so nothing here needs a per-frame answer.
	// forward: world-space look direction, see its own comment where
	// updateUniformBuffer() derives it from camToWorld. Only changes the
	// ORDER candidates are considered in (both which empty slot gets which
	// waiting candidate first, and which occupied slot is worth contesting) --
	// see facingBiasedDistSq(). It never changes WHETHER a candidate gets a
	// slot: the empty-slot pass below is unconditional, so every candidate
	// still gets one as long as there are at least as many slots as
	// candidates, exactly as before this was added.
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
			// At full intensity, not this instant's flicker: the reach decides
			// whether a mover's shadow is tracked at all, and a torch guttering
			// for a frame shouldn't drop and re-acquire the ghost standing at
			// the edge of its light.
			torchShadowReach[slot] = shadowRelevantReach(flameLightG(tf.isCandle, 1.0f),
														TORCH_LIGHT_BETA);
		};

		// The debug HUD's "Torch Shadows"/"Candle Shadows" toggles: a flame
		// whose category is off is never a waiting candidate below, and any
		// slot it already held gets freed here so switching a category off
		// mid-game drops its shadow within this reassignment tick instead of
		// waiting for something else to outbid it.
		// A flame switched off entirely (the "Torches" row) is dropped by the
		// same path rather than by a second one: it casts nothing because it
		// isn't burning, and holding a cube slot for it would keep a shadow
		// rendering for a light that no longer reaches gubo.
		auto categoryEnabled = [&](const TorchFlame &tf) {
			if(!flameBurning(tf)) return false;
			return tf.isCandle ? cheats.candleShadowsEnabled : cheats.torchShadowsEnabled;
		};
		for(int s = base; s < base + count; s++) {
			int idx = dynamicSlotOccupant[s];
			if(idx != -1 && !categoryEnabled(torchFlames[idx])) {
				torchFlames[idx].shadowSlot = -1;
				dynamicSlotOccupant[s] = -1;
			}
		}

		struct Cand { int flameIdx; float distSq; };
		std::vector<Cand> waiting;
		for(size_t i = 0; i < torchFlames.size(); i++) {
			const TorchFlame &tf = torchFlames[i];
			if(!tf.shadowCandidate || tf.shadowSlot >= 0 || !categoryEnabled(tf)) {
				continue;
			}
			waiting.push_back({(int)i, distSqTo(tf)});
		}
		std::sort(waiting.begin(), waiting.end(),
				  [](const Cand &a, const Cand &b) { return a.distSq < b.distSq; });

		// Empty slots first, unconditionally: an empty slot never "wins"
		// over a candidate the way an occupied one does, it just has
		// nothing in it yet (startup, or fewer candidates than slots).
		size_t wi = 0;
		for(int s = base; s < base + count && wi < waiting.size(); s++) {
			if(dynamicSlotOccupant[s] != -1) {
				continue;
			}
			assignSlot(s, waiting[wi].flameIdx);
			wi++;
		}

		// Occupied slots, worst (farthest occupant) first, contested
		// against the best remaining waiter (nearest first, already
		// sorted): the moment one pairing fails the margin, every pairing
		// after it -- a worse waiter against a better-placed occupant --
		// fails it too, so this can stop at the first miss instead of
		// checking every combination.
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
	// filled in on first use -- the models never change, and fitAABB() walks
	// every vertex of the mesh, so this can't be done per frame. w < 0 marks a
	// slot that hasn't been fitted yet. Feeds the per-face cull in
	// recordCubeSlotFaces(): a sphere is the only bound cheap enough to test
	// six times per instance per slot, and it doesn't need to be tight -- a
	// cull that's too generous costs a draw, one that's too tight loses a
	// shadow, so this errs the safe way by construction (the sphere around the
	// AABB, not inside it).
	std::vector<glm::vec4> modelSphereCache;
	// Scratch for the same cull: the shadow-casting instances of one slot with
	// their world-space spheres, built once per slot render and then tested
	// against each of the six faces. A member so it reuses its allocation.
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

	// Is a world-space sphere inside one cube face's 90-degree frustum?
	// `rel` is its centre RELATIVE to the light, which is where every one of
	// these frustums has its apex.
	//
	// Done by hand rather than by extracting planes out of torchFaceMatrices:
	// the faces are axis-aligned (CUBE_FACE_DIR) and square at exactly 90
	// degrees, so the four side planes are just (d +- a) / sqrt(2) for the two
	// world axes a that aren't the face's own -- i.e. the whole test is a
	// handful of component compares with no matrix work at all. That matters,
	// since it runs once per instance per face.
	bool sphereInCubeFace(const glm::vec3 &rel, float radius, int face) const {
		const glm::vec3 &d = CUBE_FACE_DIR[face];
		const float along = glm::dot(rel, d);
		// Behind the far plane. Measured along the axis rather than by true
		// distance: cheaper, and never rejects something the true distance
		// would have kept (|rel| >= along).
		if(along - radius > TORCH_SHADOW_FAR_CONST) {
			return false;
		}
		// The side planes are at 45 degrees, so a sphere of radius r sticks
		// past one until its centre is r * sqrt(2) behind it.
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

	// One cube slot's six-face render. Every caller today goes through
	// submitCubeShadowCaptures(): the held torch each frame, and every other
	// slot when its occupant changes or a mover invalidates it (see
	// lastRenderedOccupant's and movingOccluders' member comments).
	//
	// ownerInst: the instance this slot's own flame is anchored to (nullptr
	// if none), excluded from its own shadow pass -- see the inline comment
	// this carried before extraction, preserved at the call sites' comments
	// instead of duplicated here.
	//
	// cullPerFace: drop instances that fall outside the face being drawn.
	// Roughly a 6x cut in draws, since a face covers a sixth of the sphere, and
	// it is what makes a per-frame re-capture affordable at all. It is only
	// sound for a command buffer recorded and submitted on the same frame,
	// which is the only kind that records this pass now -- a visible set
	// decided at record time stops being true the moment the light or the
	// occluders move, so a buffer replayed across frames (Starter.hpp's "main"
	// one) must not use it. The flag stays rather than being assumed, so that
	// constraint is stated where it is relied on.
	//
	// faceMask: which of the six faces to draw at all, bit f for face f. A face
	// left out is not cleared and not begun -- it keeps the texels it was last
	// drawn with, which is the whole point (see pendingFaceMask). Note this
	// means a face must be in the mask AT LEAST ONCE before anything samples
	// the cube: an image nobody ever rendered stays in VK_IMAGE_LAYOUT_UNDEFINED
	// and may not be bound to the samplerCube array. The occupant diff covers
	// that by asking for ALL_CUBE_FACES the first time it sees each slot.
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
				ShadowCaster sc;
				sc.inst = &inst;
				if(cullPerFace) {
					const glm::vec4 &local = modelSphere(inst.Mid);
					sc.centre = glm::vec3(inst.Wm * glm::vec4(glm::vec3(local), 1.0f));
					// The largest of the three column lengths: a non-uniformly
					// scaled instance has to take the biggest one or its sphere
					// stops enclosing the mesh on the stretched axis.
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

			// The scene-wide draw loop below is the actual cost (up to
			// SC.TI[0].InstanceCount draws, times 6 faces) -- skipped for a
			// slot nothing currently samples, while still leaving the
			// begin/end above to clear the image and run the render pass's
			// mandatory layout transition to SHADER_READ_ONLY_OPTIMAL (the
			// samplerCube array descriptor requires every slot to be in that
			// layout even if nothing samples it this frame -- skipping the
			// render pass entirely left it stuck at UNDEFINED forever and
			// triggered a validation error).
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

	// Marks, in pendingFaceMask, every cube FACE whose cached capture
	// a MOVING occluder has just invalidated -- the other half of the staleness
	// question the lastRenderedOccupant diff answers (see movingOccluders'
	// member comment for why the cache alone is not enough now that the sun is
	// gone from lights.json).
	//
	// A face is marked when either:
	//   - it holds a mover that has moved since that face was last drawn, so
	//     its texels still record the old pose; or
	//   - it held one at its last capture and doesn't now, so its texels still
	//     have a ghost on them that has walked into another face, or out of the
	//     light's reach entirely.
	// Everything else -- a face with no mover in it, or with a mover standing
	// perfectly still -- keeps what it already has and costs a distance test
	// plus a frustum test.
	//
	// "In range" is each light's OWN reach (torchShadowReach, from its falloff)
	// rather than one radius for all of them, and it is measured against the
	// mover's bounding sphere rather than its origin: both so that the only
	// captures skipped are ones whose result nothing could see, not ones that
	// happen to be inconvenient. Everything in range is refreshed on the frame
	// it moves -- there is no per-frame budget and no queue, because a shadow
	// that updates a frame or two late is a shadow that visibly lags its ghost.
	//
	// Called from updateUniformBuffer() right after the occupant diff, i.e.
	// after GameLogic() has moved the ghosts and the doors for this frame and
	// after updateDynamicShadowSlots() has settled torchLightPos[].
	void queueMoverCubeSlotRenders() {
		if(!moverListBuilt) {
			// Something that isn't an occluder can't invalidate a capture, so
			// it doesn't belong on this list: tracking it would mark faces,
			// clear them and redraw them to exactly the same texels. The ghosts
			// are off in materials.json and so never reach this list -- and
			// giving them their shadows back now takes more than that flag,
			// since they also draw with the Spectral technique and the shadow
			// passes walk SC.TI[0] alone. Nothing has to change HERE either
			// way; see materials.json's "ghost" entry.
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
			// Not identity: a mover whose authored pose IS the identity would
			// otherwise read as "hasn't moved" on the first frame and never get
			// its first capture. A zero matrix is a pose nothing can have.
			movingOccluderWm.assign(movingOccluders.size(), glm::mat4(0.0f));
			moverListBuilt = true;
		}

		// Which faces of each slot hold a mover this frame, and which of those
		// hold one that has moved since that face was last drawn.
		std::array<uint8_t, NUM_SHADOW_CUBES> facesNow{};
		std::array<uint8_t, NUM_SHADOW_CUBES> facesStale{};

		for(size_t m = 0; m < movingOccluders.size(); m++) {
			const glm::mat4 &wm = movingOccluders[m]->Wm;
			const bool moved = (wm != movingOccluderWm[m]);
			// The bounding sphere the cull already knows about, in world space:
			// the mover's own extent has to be part of the range test, or a
			// ghost whose ORIGIN sits just outside a light's reach while its
			// body still crosses into it would go untracked.
			const glm::vec4 &local = modelSphere(movingOccluders[m]->Mid);
			const glm::vec3 p = glm::vec3(wm * glm::vec4(glm::vec3(local), 1.0f));
			const float scale = std::max({glm::length(glm::vec3(wm[0])),
										  glm::length(glm::vec3(wm[1])),
										  glm::length(glm::vec3(wm[2]))});
			const float r = local.w * scale;

			// The held torch's slot is included: its light moving is not the
			// only thing that can change what it sees, and a ghost walking into
			// its beam while the player stands still has to mark it too.
			for(int t = 0; t <= HAND_TORCH_SHADOW_INDEX; t++) {
				if(!cubeSlotOccupied(t)) {
					continue;
				}
				if(glm::distance(p, torchLightPos[t]) - r > torchShadowReach[t]) {
					continue;
				}
				// Which of this light's six directions the mover actually sits
				// in -- the same test the capture itself culls by, so a face
				// marked here is exactly a face that would draw it.
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
			// Faces holding something that moved, plus faces that held a mover
			// at their last capture and don't any more -- those still have it
			// drawn on them and nothing else would ever come back to clean it
			// off. ORed into whatever the occupant diff already asked for.
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

	// Records and submits this frame's cube shadow captures: the held torch,
	// which moves with the camera and so is never cacheable, plus every slot
	// pendingFaceMask marks -- the faces whose light changed hands
	// (lastRenderedOccupant) and the ones a ghost or a door moved inside
	// (queueMoverCubeSlotRenders).
	//
	// Called at the very end of updateUniformBuffer(), i.e. after
	// DSshadowCube[t] has this frame's matrices and after every instance's
	// DS[0][1] has this frame's world matrix, both for currentImage: this pass
	// binds the SAME per-instance descriptor set the main pass does, so
	// recording any earlier would bake a stale Wm into the map.
	//
	// It is a submission of our own rather than part of Starter.hpp's "main"
	// one because that buffer is recorded once per swapchain image and replayed
	// unmodified afterwards, which cannot express "these slots, this frame".
	// Two things make that separate submission correct rather than merely
	// convenient:
	//
	//  - It is submitted BEFORE drawFrame() submits the main buffer, on the
	//    same queue, so it is earlier in submission order. The barrier at the
	//    end then orders these colour writes against every later command on the
	//    queue, the main pass's sampling of the cube maps included -- that is
	//    what makes the reads see finished captures. This is the piece that
	//    replaces the vkQueueWaitIdle this used to do (inside
	//    endSingleTimeCommands()): the GPU no longer has to drain mid-frame,
	//    it just has to order two stages against each other.
	//  - Each swapchain image has its own buffer and its own fence, waited on
	//    before re-recording, because re-recording a buffer still in flight is
	//    undefined behaviour. In the steady state that fence is long since
	//    signalled and the wait returns immediately.
	void submitCubeShadowCaptures(int currentImage) {
		if(shadowCB.empty()) {
			return;	// nothing allocated yet (first frames of a swapchain rebuild)
		}

		// The held torch's own staleness, settled before anything is recorded so
		// the "is there work at all" test below can see it. All six faces
		// whenever its light has MOVED, because every face was shot from that
		// position -- but only the faces a mover marked when it hasn't, which is
		// every frame the player stands still. Same rule as every other slot;
		// the only difference is that this light moves often rather than never.
		// The occupancy is part of the comparison because switching torch
		// shadows off has to clear the map it leaves behind.
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
			// The instance a slot's own flame is anchored to, excluded from its
			// own capture: it sits centimetres from the light, so it would
			// otherwise fill several faces with its own silhouette. For the held
			// torch that's handTorchInst, which never goes through the flame
			// pool (see HAND_TORCH_SHADOW_INDEX).
			Instance *ownerInst = nullptr;
			if(t == HAND_TORCH_SHADOW_INDEX) {
				ownerInst = handTorchInst;
			} else if(t >= dynamicShadowSlotBase && dynamicSlotOccupant[t] != -1) {
				ownerInst = torchFlames[dynamicSlotOccupant[t]].inst;
			}
			recordCubeSlotFaces(cb, currentImage, t, cubeSlotOccupied(t), ownerInst, true,
								pendingFaceMask[t]);
		}

		// The dependency described above. A global memory barrier rather than
		// one image barrier per cube: the layouts are already correct (each
		// face's render pass ends in SHADER_READ_ONLY_OPTIMAL), so all that is
		// missing is availability/visibility of the writes, which one barrier
		// covers for every image this buffer touched.
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
	// file -- RPShadow2D, RP, the post chain -- is built in a method here
	// rather than in a free-standing helper.
	void createCubeShadowMaps() {
		// Shared by every torch: same resolution, same format, so one
		// CLAMP_TO_EDGE sampler serves them all. No mipmaps (a shadow lookup
		// always samples level 0), hence maxLod = 1.
		//
		// NEAREST, not LINEAR. What these cubes store is a DISTANCE, not a
		// colour, and averaging four neighbouring distances is only meaningful
		// where they describe the same surface. Across a silhouette -- a texel
		// on the door and the texel next to it looking past its edge into the
		// far wall -- bilinear filtering returns a distance that belongs to
		// NEITHER, somewhere in between, and every fragment compared against it
		// gets the wrong answer: too near, and an unoccluded surface reads as
		// shadowed; too far, and a genuinely occluded one reads as lit. The
		// error is proportional to the depth GAP across the edge, i.e. metres,
		// not texels, which is why it used to need a bias of the same order to
		// paper over (the 0.35 grazing bias shadowFromCube() carried) -- and
		// that bias is what let a torch light the chains through a closed door,
		// since the door is only ~0.6 units in front of them. Sampling one
		// texel with no blending makes the comparison honest again and lets the
		// bias be what it should be: a texel-sized quantity.
		cubeShadowSampler.init(this, VK_FILTER_NEAREST, VK_FILTER_NEAREST,
								VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
								VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
								VK_SAMPLER_MIPMAP_MODE_LINEAR,
								VK_FALSE, 1.0f, 1.0f);

		const VkFormat colorFmt = VK_FORMAT_R32_SFLOAT;
		const VkFormat depthFmt = findDepthFormat();

		for(int i = 0; i < NUM_SHADOW_CUBES; i++) {
			CubeShadowMap &c = torchCube[i];

			// The 6-layer colour image itself: CUBE_COMPATIBLE_BIT is what
			// lets a VK_IMAGE_VIEW_TYPE_CUBE view (below) treat its 6 layers
			// as cube faces instead of an ordinary array.
			createImage(SHADOW_MAP_RES, SHADOW_MAP_RES, 1, 6,
						VK_SAMPLE_COUNT_1_BIT, colorFmt, VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						c.colorImage, c.colorMemory);

			// The one view CookTorrance.frag's samplerCube reads: all 6
			// layers, VK_IMAGE_VIEW_TYPE_CUBE. baseArrayLayer 0 (this
			// helper's only option) is correct here since it spans every
			// layer anyway.
			c.cubeView = createImageView(c.colorImage, colorFmt, VK_IMAGE_ASPECT_COLOR_BIT,
										  1, VK_IMAGE_VIEW_TYPE_CUBE, 6);

			// One VK_IMAGE_VIEW_TYPE_2D view per layer, for the framebuffers
			// below -- a full cube view cannot be a render target, and
			// BaseProject::createImageView always fixes baseArrayLayer at 0,
			// so these need a raw vkCreateImageView call to pick layer
			// `face` specifically.
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

			// One depth image/view PER TORCH, reused across its 6 faces --
			// see the CubeShadowMap::depthImage field comment for why that
			// reuse is safe. Never sampled (only used for the rasterizer's
			// z-test), so a single mip, single layer, plain 2D image.
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

		// This creates a new pipeline (with the current surface), using its shaders for the provided render pass
		P.create(&RP);
		// Same render pass as P: the ghosts draw inside the scene pass, sharing
		// its depth buffer (which is what lets a wall hide one) and writing into
		// the same HDR attachment, where the bloom chain can find their rim.
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
		// attachment as an ordinary texture; getViewAndSampler() hands back the
		// VkDescriptorImageInfo for it, already carrying the finalLayout the
		// render pass leaves the image in (SHADER_READ_ONLY_OPTIMAL).
		//
		// RP's own sampled output is its RESOLVE attachment, not its colour
		// one -- the colour attachment is multisampled and is discarded.
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
		// Same RP as the scene: the flame draws inside it, right after the
		// scene geometry, so it shares the depth buffer instead of needing its
		// own render pass the way UiQuad's 2D overlay does -- and so its
		// over-1.0 colours land in the HDR attachment where bloom can find
		// them.
		flame.pipelinesAndDescriptorSetsInit(&RP);
		// Same RP, and for the same reason: the exit's daylight has to write
		// its over-1.0 colours into the HDR attachment, which is where the
		// bloom chain can find them.
		exitGlow.pipelinesAndDescriptorSetsInit(&RP);
		// Same RP too, for the same reason -- see DebugLines.hpp's header.
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

		// PShadow/RPShadow2D never go through pipelinesAndDescriptorSetsCleanup
		// (see the member declaration for why -- they don't depend on the
		// swapchain, so a resize never tears them down), which is where P/RP
		// normally get their .cleanup() half. Both halves have to happen
		// somewhere, so both happen here instead. Same story for the cube
		// branch (PShadowCube/RPShadowCubeCompat/torchCube[]).
		PShadow.cleanup();
		PShadow.destroy();
		for(int i = 0; i < NUM_SHADOW_MAPS_2D; i++) {
			RPShadow2D[i].cleanup();
			RPShadow2D[i].destroy();
		}

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

		// Before SC.localCleanup(): that frees the colliders scene.json created, which
		// colliderSet also points at. It only deletes the ones it allocated itself, but
		// dropping the shared list first keeps the two ownership halves from overlapping.
		colliderSet.cleanup();
		allColliders.clear();

		SC.localCleanup();
		txt.localCleanup();
		uiQuad.localCleanup();
		crosshair.localCleanup();
		pauseQuad.localCleanup();
		startScreenQuad.localCleanup();
		flame.localCleanup();
		exitGlow.localCleanup();
		debugLines.localCleanup();
	}
	
	// Here it is the creation of the command buffer:
	// You send to the GPU all the objects you want to draw,
	// with their buffers and textures
	static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void *Params) {
		// Simple trick to avoid having always 'T->'
		// in che code that populates the command buffer!
		Castlescape *T = (Castlescape *)Params;
		T->populateCommandBuffer(commandBuffer, currentImage);
	}

	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
		// The whole HDR chain goes into this one command buffer, in order.
		// Ordering between the passes is handled by their render pass
		// dependencies rather than by explicit barriers -- see the comment on
		// the RPbright/RPblurH/RPblurV members. The text and HUD passes are
		// separate command buffers submitted after this one (submit orders
		// 10000 and 9000 against this one's 0), so they end up drawing on top
		// of the composited frame.

		// The shadow passes, all before the main pass they feed:
		// CookTorrance.frag samples these maps, so they have to be fully
		// rendered (and transitioned to a readable layout) before that draw
		// happens. Not Scene::populateCommandBuffer -- that walks every
		// technique including Flame, and the flames are deliberately not
		// occluders here (see the RPShadow2D member comment) -- so this
		// draws straight over the CookTorrance instances (technique 0 in
		// scene.json) itself, twice: once per 2D map, once per torch per
		// cube face.
		for(int i = 0; i < NUM_SHADOW_MAPS_2D; i++) {
			RPShadow2D[i].begin(commandBuffer, currentImage);
			PShadow.bind(commandBuffer);
			vkCmdPushConstants(commandBuffer, PShadow.pipelineLayout,
							   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
							   &shadowLightSpace2D[i]);
			for(int j = 0; j < SC.TI[0].InstanceCount; j++) {
				Instance &inst = SC.TI[0].I[j];

				// The light fixtures don't occlude (see Material::castsShadow).
				// Same forModel() lookup the main pass does for the BRDF, so no
				// extra per-frame work beyond the branch.
				if(!materials.forModel(inst.Mid).castsShadow) {
					continue;
				}

				// set 0 here is DSLlocal's per-instance buffer -- the SAME
				// descriptor set the main pass binds at set 1 (DS[0][1]),
				// re-mapped with this instance's current Wm every frame in
				// updateUniformBuffer() regardless of which pipeline reads
				// it. See Shadow.vert's header for why reusing it is safe.
				inst.DS[0][1]->bind(commandBuffer, PShadow, 0, currentImage);
				SC.M[inst.Mid]->bind(commandBuffer);
				vkCmdDrawIndexed(commandBuffer,
								 static_cast<uint32_t>(SC.M[inst.Mid]->indices.size()), 1, 0, 0, 0);
			}
			RPShadow2D[i].end(commandBuffer);
		}

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
		// Offscreen pass - always required
		// begin standard pass
		// 1. The scene, into the offscreen HDR target.
		RP.begin(commandBuffer, currentImage);
		SC.populateCommandBuffer(commandBuffer, 0, currentImage);

		// The ghosts' COLOUR pass, by hand, because Scene has already spent the
		// technique's own slot on the depth prepass that has to come between
		// the dungeon and this -- see PRs[1] in localInit() and
		// SpectralDepth.frag. Everything the prepass just wrote depth for is
		// now drawn again through Pspectral, and only its nearest layer per
		// pixel survives the LESS_OR_EQUAL test, which is what keeps the feet
		// inside the robe from glowing through the body.
		//
		// Technique 1 is the "Spectral" block in scene.json; the shadow loop
		// above indexes TI[0] the same way and for the same reason. Same
		// binding sequence Scene::populateCommandBuffer() uses, minus the
		// per-instance pipeline rebind: one bind covers all three ghosts.
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

		flame.populateCommandBuffer(commandBuffer, currentImage);
		// After the scene, so the castle has already written the depth that
		// masks this quad down to the shape of the doorway's arch, and after
		// the flames, which are the only other thing it can blend against.
		exitGlow.populateCommandBuffer(commandBuffer, currentImage);
		debugLines.populateCommandBuffer(commandBuffer, currentImage);
		RP.end(commandBuffer);

		// 2-5. Four full-screen quads: threshold, blur across, blur down,
		// then mix the result back over the scene and tone map.
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

		// ESC now opens/closes the pause menu instead of closing the window
		// outright -- Quit is reached through the menu instead (see
		// PauseMenu.hpp). Edge-triggered so holding ESC down doesn't reopen
		// the menu the instant Resume closes it. Ignored while the cheat HUD
		// or the launch screen is open: only one modal overlay at a time,
		// and L/Play already own closing those.
		bool escPressed = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
		if(escPressed && !escKeyWasPressed && !hud.isOpen() && !startScreen.isOpen()) {
			pauseMenu.setOpen(!pauseMenu.isOpen(), windowWidth, windowHeight);
		}
		escKeyWasPressed = escPressed;

		// moves the view
		float deltaT = GameLogic();

		// The pause menu and the launch screen both mean "stop time", full
		// stop: not just player movement/physics/the hunt clock (which
		// GameLogic() already gates on overlayOpen(), same as the cheat HUD
		// always did), but every other deltaT-driven animation below too
		// (torch flicker, flame UV scroll, shadow reassignment, ...) -- they
		// all thread through this one deltaT, so zeroing it here freezes all
		// of them at once instead of gating each site individually. The
		// cheat HUD deliberately does NOT also zero this: watching torches
		// keep flickering while flipping a debug flag is fine, only an
		// actual freeze-frame (paused, or not even started yet) needs to
		// look like one.
		if(pauseMenu.isOpen() || startScreen.isOpen()) {
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

		// The camera's world position, needed below to cull torch lights by
		// distance, and again by the flame billboards to work out which way to
		// face.
		const glm::mat4 camToWorld = glm::inverse(View);
		const glm::vec3 eyePos = glm::vec3(camToWorld[3]);
		// World-space look direction, for the same reason: camToWorld's column
		// 2 is the camera's local +Z expressed in world space, and the camera
		// looks down its own local -Z (glm::lookAt convention), hence the
		// negation. Used below to bias both the live-torch-light cut and the
		// shadow-cube pool toward whatever's actually in view over whatever's
		// merely close -- a light directly behind the player lights nothing
		// the frame renders, no matter how near it is.
		const glm::vec3 forward = -glm::vec3(camToWorld[2]);

		// Re-decides which shadow candidates hold the dynamic cube-shadow
		// pool's slots, at most every SHADOW_REASSIGN_INTERVAL -- must run
		// before the light-append loop below (it reads tf.shadowSlot) and
		// before the DSshadowCube mapping loop further down (it writes
		// torchFaceMatrices/torchLightPos for whichever slots change hands).
		shadowReassignTimer += deltaT;
		if(shadowReassignTimer >= SHADOW_REASSIGN_INTERVAL) {
			shadowReassignTimer = 0.0f;
			updateDynamicShadowSlots(eyePos, forward);
		}

		// Diffs every static cube slot's CURRENT occupant identity against
		// the one it was last actually rendered for (see
		// lastRenderedOccupant's member comment) and queues anything that
		// changed into pendingFaceMask. Cheap enough (activeCubeShadows
		// int compares, at most NUM_SHADOW_CUBES - 1) to just run every
		// frame rather than special-casing "only right after a reassignment
		// tick" -- it only ever finds work on the frame something actually
		// changed (startup, or a genuine reassignment/category toggle
		// above), everything else is a no-op pass. HAND_TORCH_SHADOW_INDEX
		// is excluded: it's rendered fresh every frame in
		// populateCommandBuffer() instead, never through this cache.
		for(int t = 0; t < dynamicShadowSlotBase; t++) {
			// A fixed lights.json slot's identity never changes once set, so
			// using the slot index itself as the "occupant" marker means
			// this only ever fires once, on the very first frame.
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

		// The diff above only sees a slot whose LIGHT changed. This adds the
		// slots whose light stayed put while a ghost (or a swinging door) moved
		// inside it -- without it those movers cast a shadow frozen at the pose
		// the slot was captured in, which is what happened to the ghosts the
		// moment lights.json lost its sun. See queueMoverCubeSlotRenders().
		queueMoverCubeSlotRenders();

		// The cylindrical billboard basis every flame uses this frame, taken
		// from the CAMERA's own right axis rather than from each flame's
		// individual eye->anchor direction.
		//
		// Deriving it per-flame as cross(worldUp, eyePos - anchor) is the
		// textbook cylindrical billboard, and it behaves for a torch bolted to
		// a wall a few units away. It falls apart for the torch held in the
		// player's own hand. That anchor sits barely a unit from the eye and is
		// placed in CAMERA space (HAND_TORCH_OFFSET), so pitching up or down
		// swings it bodily around the eye: its HORIZONTAL offset from the eye
		// shrinks toward zero, and at the pitch where the torch passes directly
		// over (or under) the camera it changes sign. The yaw derived from it
		// therefore whips through 180 degrees -- which is the flame visibly
		// spinning when you look up or down -- and just either side of the
		// crossing that horizontal vector is near zero length, so its direction
		// is numerical noise before it even flips.
		//
		// The camera's right axis has neither problem. This camera is built
		// yaw-then-pitch with no roll (right = cross(front, worldUp) in
		// GameLogic), so it is exactly horizontal at every pitch, never
		// degenerates, and turns only with yaw -- which is the one rotation a
		// standing flame should follow. Facing the view PLANE rather than the
		// view POINT is the standard billboard choice anyway: it is what stops
		// a billboard from slowly rotating as it slides across the screen, and
		// at torch distances the two are visually indistinguishable.
		glm::vec3 bbRight = glm::vec3(camToWorld[0]);
		bbRight.y = 0.0f;
		if(glm::dot(bbRight, bbRight) > 1e-8f) {
			bbRight = glm::normalize(bbRight);
		} else {
			// Unreachable while pitch is clamped to +/-89 degrees, but a
			// billboard with a zero-length basis vector collapses to a line,
			// so it gets an arbitrary valid axis rather than a NaN.
			bbRight = glm::vec3(1.0f, 0.0f, 0.0f);
		}
		const glm::vec3 bbUp = glm::vec3(0.0f, 1.0f, 0.0f);
		// Points back toward the eye, so the flame's local +z is "toward the
		// camera" exactly as Flame.vert's layer offset assumes.
		const glm::vec3 bbFwd = glm::cross(bbRight, bbUp);

		// A second basis for the HELD torch only (TorchFlame::heldByCamera):
		// the camera's ACTUAL up/right/forward, pitch included, straight off
		// camToWorld's columns (View is built with no roll, so these are
		// already orthonormal: column 0 is the camera's right, column 1 its
		// up, column 2 the local +Z axis, which points from the scene back
		// toward the eye -- the same "toward camera" sense bbFwd above has).
		//
		// The held torch is not a world object the camera walks past, it is
		// welded to the camera itself (see TorchFlame::heldByCamera), and
		// visibly tilts as the camera pitches. A flame that stays
		// world-vertical on top of a shaft that tilts with the view swings out
		// of alignment with it -- past a fairly small pitch it reads as
		// sticking out sideways from the torch instead of burning at its tip,
		// which is worse than the spin the cylindrical basis was built to fix.
		// Tying the flame to the SAME basis the torch mesh itself rides
		// (camera-local, pitch included) keeps the two rigidly aligned at
		// every pitch, and since this basis is read directly off the view
		// matrix it can't degenerate or flip the way a per-flame eye-vector
		// could.
		const glm::vec3 handBbRight = glm::normalize(glm::vec3(camToWorld[0]));
		const glm::vec3 handBbUp    = glm::normalize(glm::vec3(camToWorld[1]));
		const glm::vec3 handBbFwd   = glm::normalize(glm::vec3(camToWorld[2]));

		animTime += deltaT;

		// Advance every torch's fire state before anything reads it, so the
		// flame, its sparks and its light are all driven by the same envelope
		// within the same frame (see the TorchFlame struct).
		for(TorchFlame &tf : torchFlames) {
			// Flicker: a fast term for the light's visible dance (demoted from
			// half the signal to a quarter -- the per-pixel shimmer in
			// Flame.frag owns the fast twitch now, see FLAME_FLICKER_HZ), a
			// slow term under it so the flame also breathes over seconds, and
			// a middle one to stop the two from reading as separate layers.
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
			// frame (see TorchFlame::baseColor). This one line is the whole
			// visual half of the mechanic: `color` is what both the point light
			// and the billboard read, so tinting it here turns the torch, its
			// light, its shadows and its bloom violet together, with no second
			// path to keep in step.
			//
			// The warning pulse and the hunt's dimming ride on the same value
			// rather than on `intensity`, which is spring-driven state: a 4 Hz
			// pulse written into a spring with a ~0.2 s response would be
			// damped into nothing, and writing it into the spring's own value
			// would corrupt the flicker it exists to produce.
			tf.color = huntCycle.flameColor(tf.baseColor)
					   * huntCycle.warningPulse() * huntCycle.lightScale();

			// Height: the same signal, compressed into a narrower band and
			// chased much more slowly -- see TorchFlame::heightScale.
			float hTarget = 0.78f + 0.31f * (target - 0.30f) / 1.10f;	// ~0.78..1.09
			tf.heightScale += (hTarget - tf.heightScale)
							  * (1.0f - std::exp(-deltaT / FLAME_HEIGHT_TAU));

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

		// Torch flames' point lights. NOT going through SceneLights/
		// lights.json's own "instance"+"offset" anchoring: that reads the
		// instance's Wm once, at SceneLights::init() time, which is exactly
		// wrong for the held torch (its Wm doesn't exist in any meaningful
		// form until GameLogic() starts overwriting it every frame -- an
		// anchor taken before that would freeze the light at whatever the
		// placeholder scene.json transform happened to be, typically the
		// origin). So instead: appended straight into gubo here, every
		// frame, from the same Wm the flame itself now rides.
		//
		// Culled by REAL distance (never by facing -- a torch you just
		// turned away from should keep lighting the room behind you, it just
		// shouldn't win a contested slot over one you're looking at) and
		// capped in count, priority order from facingBiasedDistSq() (i.e.
		// nearest-and-in-view first): every light in gubo costs a full BRDF
		// evaluation for every fragment of every object, multiplied by the
		// sample count. See TORCH_LIGHT_CULL_DIST on why dropping the far
		// ones is invisible. With TORCH_LIGHT_MAX_LIVE now well above the
		// scene's actual flame count, this cap doesn't bite in practice --
		// the ordering only matters if it ever does.
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
				if(dSq <= cullSq) {
					nearest.push_back({facingBiasedDistSq(worldPos, eyePos, forward), &tf});
				}
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
				// The same envelope the flame itself is drawn with. Brightness
				// rides on colour and reach rides on g, and both are needed:
				// colour alone makes the lit area pulse in place, g alone makes
				// it grow and shrink without changing how hot it looks.
				L.color = tf.color * tf.intensity * tf.lightScale;
				L.g = flameLightG(tf.isCandle, tf.intensity);
				L.beta = TORCH_LIGHT_BETA;
				L.cosIn = 1.0f;
				L.cosOut = 0.0f;
				L.type = LIGHT_POINT;
				// The held torch gets the reserved HAND_TORCH_SHADOW_INDEX
				// cube slot, recomputed for its current (camera-following)
				// position right here so populateCommandBuffer()'s cube
				// shadow pass -- later this same frame -- renders it from
				// the right place. Every other shadowCandidate flame reads
				// whatever slot (if any) updateDynamicShadowSlots() gave it
				// this reassignment tick -- -1 (skip shadowFactor()'s lookup
				// entirely) if it currently isn't nearest enough to hold
				// one. A non-candidate flame is never a shadow caster: left
				// unset this zero-initializes to 0, the sun's own shadow
				// map, and indoors -- in the sun's shadow -- that would read
				// its light as fully shadowed, hence the explicit -1.
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

		// The light the open exit throws back into the room. Appended straight
		// into gubo like the torch lights just above, and for the same reason
		// they are: its brightness is a function of something that changes
		// every frame -- how far the leaf has swung -- and lights.json has no
		// way to express that. A closed door contributes nothing and isn't
		// appended at all, so this costs a light slot only while it matters.
		//
		// It is the counterpart to the ExitGlow quad, not a substitute for it:
		// the quad is the daylight you LOOK at, this is the daylight that
		// lands on the stone. Neither reads right alone -- a lit floor with a
		// dark opening looks broken, and a blazing opening that throws no
		// light looks pasted on.
		//
		// No shadow map: shadowIndex is left at -1 (the "skip the lookup"
		// value, see the torch loop above). Giving it one would mean a 2D
		// shadow pass whose only occluder is a door that moves, i.e. a map
		// that would have to be re-rendered every frame of the swing, and the
		// wall it shines through already does the shaping for free.
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
		// Uploaded unswitched. The cheat is applied in the shader's
		// ambientShare() via LIGHT_DEBUG_NO_AMBIENT below, NOT by zeroing this:
		// a model that sets its own ambientWeight in materials.json (the floor,
		// 0.20) never reads this field, so zeroing it here left that model at
		// full indirect light with the cheat off -- the switch has to sit after
		// the per-model override, and only the shader is.
		//
		// Zeroing the SHARE, wherever it happens, is still what the cheat has to
		// do rather than just blacking the colors: under E17's blend the direct
		// half is scaled by (1 - weight), so it would keep giving up its share
		// to an ambient contributing nothing and the cheat would DARKEN the
		// scene instead of taking the indirect light out of it.
		gubo.ambientWeight = amb.weight;
		// Same bucket, so the same gate covers it. See CookTorrance.frag's blend.
		gubo.ambientBounce = amb.bounce;

		// Distance fog density: derived from GEOM_CULL_CONE_DIST rather than
		// its own hand-picked number, so the two can't drift apart the way
		// TORCH_LIGHT_CULL_DIST and GEOM_CULL_CONE_DIST would have without
		// their static_assert above. Solves exp(-(density*dist)^2) =
		// FOG_RESIDUAL_AT_CULL_DIST for dist == GEOM_CULL_CONE_DIST *
		// FOG_REFERENCE_DIST_SCALE, i.e. fog reaches (in this case) 1% of
		// unfogged brightness only some distance PAST where the geometry
		// cull would stop drawing something, rather than exactly at it --
		// the single knob to turn if the fog still reads as too heavy or too
		// light close up:
		//   1.0   the first version: fully faded right at the cull distance,
		//         which read as noticeably dark well before that (~60% by
		//         the middle of the cull's own range) -- an exponential
		//         curve's brightness already drops fast long before it
		//         visually "arrives" at its target residual.
		//   >1.0  gentler up close, and fully faded only somewhat past the
		//         cull distance instead of exactly at it -- some genuine
		//         geometry pop can peek through right at the true cull edge
		//         as a result, softened by the vignette below and by fog and
		//         background sharing the same black. 2.5 keeps close-up
		//         brightness within a few percent of unfogged out to the
		//         GEOM_CULL_RADIUS "always visible" ring, and still reaches
		//         roughly half brightness by the cull's own farthest reach.
		//   <1.0  the opposite tradeoff: fully hides the pop with room to
		//         spare, at the cost of a noticeably darker foreground.
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

		// Stare-at glare (see the GLARE_* constants): how squarely and how
		// closely the camera is looking at each WALL torch's flame. The held
		// torch is excluded on purpose -- it sits near screen centre by
		// construction, so glare from it would be a permanent bias rather
		// than an event. Purely geometric (camera pose, anchor, envelope):
		// nothing here reads back anything rendered, so raising the exposure
		// below cannot recruit new glare and feed on itself.
		{
			// camToWorld's column 2 points from the scene back toward the
			// eye (see the billboard bases above), so the camera's actual
			// look direction is its negation.
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
			// The stare-at glare rides the two knobs the post chain already
			// has: more bloom spread AND more exposure when the player looks
			// into a flame. Smoothed asymmetrically upstream, so this never
			// pumps frame-to-frame.
			post.bloomIntensity = BLOOM_INTENSITY * (1.0f + GLARE_BLOOM_GAIN * glareSmoothed);
			post.exposure = SCENE_EXPOSURE * (1.0f + GLARE_EXPOSURE_GAIN * glareSmoothed);
			// The escape whiteout, on the very same two knobs and multiplied
			// on top rather than replacing them, so a player who wins while
			// staring into a torch gets one flash and not a fight between two.
			// Cubed: the ramp is linear in time (see GameLogic), and a linear
			// ramp of EXPOSURE reads as the image getting evenly brighter,
			// which looks like a fade to white. Weighting it towards the end
			// instead gives the eye a moment of the room still being there
			// before it is taken, which is what makes it read as being blinded
			// by the doorway rather than as a screen transition.
			if(escapeFlash > 0.0f) {
				const float f = escapeFlash * escapeFlash * escapeFlash;
				post.exposure *= 1.0f + ESCAPE_EXPOSURE_GAIN * f;
				post.bloomIntensity *= 1.0f + ESCAPE_BLOOM_GAIN * f;
			}
			// The whiteout's own term, which Composite.frag applies after the
			// tone map -- see its declaration in PostUniformBufferObject for
			// why the exposure ramp above cannot do this on its own. Held back
			// until the exposure ramp has had most of the flash to work in, so
			// the frame is already blowing out by the time the white arrives
			// rather than being painted over while it is still readable.
			post.escapeFlash = glm::smoothstep(0.45f, 1.0f, escapeFlash);

			// The spectral veil's ramp -- see SPECTRAL_VEIL_OUTER. Here rather
			// than in the ghost loop because that loop runs on the game clock:
			// pausing inside a ghost would freeze the wash while the camera
			// kept moving. max() over the ghosts, not a sum; two ghosts on the
			// same square is still one player inside a ghost. Reads the drawn
			// matrix so the bob counts, which at this range is the difference
			// between being inside the body and under it.
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

			// Composite: samples both textures with plain normalised UVs, so
			// the texel size is not read at all. The tone map (see
			// CookTorrance.frag) happens here, at the end of the chain, which
			// is why debugFlags has to reach this pass -- the Tone Mapping
			// cheat toggles it.
			post.texelSize = fullTexel;
			post.blurDir = glm::vec2(0.0f);
			DScomposite.map(currentImage, &post, 0);
		}

		// Each flame's render matrix is one of the two billboard bases built
		// once near the top of this function, translated and scaled to that
		// flame's anchor:
		//
		//   - Wall torches (heldByCamera == false) get the CYLINDRICAL basis
		//     (bbRight/bbUp/bbFwd): it spins about world up so the quads
		//     always face the camera, but stays upright regardless of the
		//     camera's pitch, because a torch bolted to a wall doesn't tilt
		//     just because the player looks up or down.
		//   - The held torch (heldByCamera == true) gets the CAMERA-LOCKED
		//     basis (handBbRight/handBbUp/handBbFwd) instead: that torch tilts
		//     WITH the camera's pitch (it's anchored in camera-local space),
		//     so its flame has to ride the camera's actual up/right axes to
		//     stay lined up with the shaft it's burning on top of -- a
		//     world-vertical flame there swings out of alignment with the
		//     torch at exactly the pitch angles that made this worth splitting
		//     out. See TorchFlame::heldByCamera.
		//
		// A camera-facing billboard has no orientation to get wrong relative
		// to the player turning -- it presents the same face from every
		// angle by construction. What's left is the part that SHOULD respond
		// to movement, and that goes in deliberately as tf.lean.
		//
		// Local space for the shaders: x is +/-1 across the flame's half-width,
		// y is 0 at the wick and 1 at the natural tip, z is toward the camera
		// (used only to separate the three depth layers).
		for(const TorchFlame &tf : torchFlames) {
			glm::vec3 anchorWorld = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			// Every torch instance here uses a uniform scale (none has a
			// stretched axis), so its length alone -- taken off any one
			// basis column, rotation doesn't change a vector's length -- IS
			// the scale factor, with no need to fully decompose Wm.
			float instScale = glm::length(glm::vec3(tf.inst->Wm[0]));

			// A flame switched off by the cheat menu is collapsed to a point
			// rather than skipped: Flame's per-flame descriptor set is
			// recorded once into the command buffer and replayed every frame,
			// so there is no "don't draw this one" to take here. Zero-sized
			// basis columns put all three cards' vertices on the anchor, i.e.
			// zero-area triangles the rasterizer produces no fragments for --
			// well-defined, unlike leaving w degenerate.
			float sizeScale = flameBurning(tf) ? tf.sizeScale : 0.0f;
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

		// The daylight outside the exit door. Unlike the flames above these
		// are NOT billboards: they stand in fixed planes outside the doorway,
		// so their bases are built from world axes and not from the camera's.
		// Two consequences worth knowing: seen from far off to one side they
		// foreshorten, which is correct (you are looking along a doorway, not
		// at a lamp), and the arch is what limits how much of either one you
		// can ever see anyway.
		//
		// Columns are the same convention Flame's billboard uses -- the two
		// in-plane axes scaled to the quad's half-extents, the normal, then
		// the centre -- so ExitGlow.vert can keep its corner parameter in
		// -1..1 and let these matrices do the placing. It is also what lets
		// one mesh and one pipeline serve both an upright quad and one lying
		// flat: the difference is entirely in which world axes go in the
		// first two columns.
		{
			// Squared, so the light builds late in the swing rather than
			// tracking it: a door barely ajar should show a crack of light,
			// not half the glare. exitOpenFrac is already smoothstepped, so
			// this is the second shaping of the same signal and deliberately
			// so -- the first spreads it over the swing, this one weights it
			// towards the end of it.
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

		// Same matrices every CookTorrance instance's set 2 gets mapped
		// with below -- built once here rather than inside the loop since
		// it's identical for every one of them. Static content (see
		// computeShadowMatrices()), but still re-mapped every frame: map()
		// writes into a per-swapchain-image buffer slot, and mapping only the
		// slot for image 0 would leave the others holding whatever was there
		// at allocation time.
		ShadowUniformBufferObject shadowUbo{};
		for(int i = 0; i < NUM_SHADOW_MAPS_2D; i++) {
			shadowUbo.lightSpace[i] = shadowLightSpace2D[i];
		}

		// Same idea, cube side: DSshadowCube[t] feeds the shadow CAPTURE pass
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

		// Debug overlay (DebugLines.hpp, cheat-menu gated): gizmos at each
		// active light's position/direction, and/or wireframe boxes at each
		// torch's shadow-cube clip planes. Built here so gubo.lights[]/
		// activeCubeShadows/torchLightPos[] are all current for this frame,
		// including the held torch (refreshed by updateHandTorchShadow()
		// earlier in this function, same as the DSshadowCube[] loop above
		// relies on).
		std::vector<glm::vec4> dbgPos, dbgColor;
		if(cheats.showLightGizmos) {
			for(int i = 0; i < gubo.lightCount; i++) {
				const LightData &L = gubo.lights[i];
				glm::vec4 color = glm::vec4(L.color, 1.0f);
				if(L.type == LIGHT_DIRECT) {
					// The sun has no position, only a direction, so its gizmo
					// is an arrow anchored near the player instead of a cross
					// at a point in space.
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
		// Collision geometry (CheatFlags::showColliders). Read straight off the
		// same list and the same accessors the collision loops in GameLogic()
		// use, so the overlay can't drift from what actually blocks the player.
		if(cheats.showColliders) {
			// Every box is grown outward by this much before being drawn. A
			// collider box normally sits exactly ON the surface it was fitted
			// to, and coincident depth on a wall we already know z-fights
			// badly (see the dungeon meshes) would make the overlay flicker in
			// and out along its own edges. The DebugLines pipeline depth-tests
			// like the rest of the scene, on purpose -- an overlay that ignored
			// depth would show every collider in the castle through the walls,
			// which is unreadable -- so the fix is the nudge, not the test.
			const float COLLIDER_DRAW_EPS = 0.01f;
			for(Collider *C : allColliders) {
				AABBextents E = C->getExtents();
				DebugLines::PushAABB(glm::vec3(E.xMin, E.yMin, E.zMin) - COLLIDER_DRAW_EPS,
									 glm::vec3(E.xMax, E.yMax, E.zMax) + COLLIDER_DRAW_EPS,
									 glm::vec4(0.0f, 1.0f, 0.2f, 1.0f), dbgPos, dbgColor);
			}
			// Ramps in a different color because they behave differently: they
			// are ground-only, never walls, and their surface is the drawn quad
			// itself rather than the top of a box.
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
		debugLines.update(currentImage, ViewPrj, dbgPos, dbgColor);

		// The gazed door's chains/padlock are separate instances (see
		// Door::LockProp) but should glow along with the door leaf -- they
		// read as part of the door, not a prop sitting in front of it.
		// Built once here rather than re-searching `doors` per instance
		// below.
		// Left empty with the Focus Glow cheat off, which drops ubo.glow to 0
		// for every instance below -- the gaze itself is untouched, so [E] and
		// the prompt keep working, only the aura goes.
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
				// Geometry visibility cull (GEOM_CULL_* above): an instance
				// too far outside the radius+cone gets a stand-in render
				// matrix instead of its real one, so it draws nowhere the
				// camera can ever see it. Same "park it off the map" idiom
				// already used to hide a consumed key/prop/sinking object
				// elsewhere in this file (an invertible translate, not a
				// zero matrix -- nMat below is its inverse-transpose, and a
				// true zero matrix has none, which is a NaN CookTorrance.frag
				// has no guard against). Only this render-time copy changes:
				// SC.TI[...].Wm and the instance's collider are left alone,
				// so a culled instance still exists for gameplay/collision
				// purposes, it just isn't drawn.
				//
				// Tested as a bounding SPHERE, not a bare point: this
				// instance's own translation column is a fine stand-in for
				// its position, but a long wall segment's pivot can sit at
				// one end rather than its centre (this asset pack isn't
				// consistently centre-pivoted), so judging it by that point
				// alone clipped things right at the screen's edges that were
				// still plainly in view. Instances with a collider (most
				// walls/furniture, see scene.json's "collider": "AABB") get
				// their real world-space extents; everything else falls back
				// to GEOM_CULL_FALLBACK_RADIUS.
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
				// The instance the crosshair is currently aimed at (see
				// gazedInstance in GameLogic()), plus its lock hardware if
				// it's a locked door (see glowingInstances above), glows;
				// every other instance, including other instances of the
				// same model, doesn't -- this is why the flag lives here
				// per-instance rather than in Material, which is shared
				// per-model.
				bool glow = false;
				for(Instance *g : glowingInstances) {
					if(g == &inst) {
						glow = true;
						break;
					}
				}
				// Sign carries "would [E] do anything right now" (see
				// gazedInteractionDisabled) and magnitude carries which
				// aura color (see GlowKind) -- both piggybacked on this one
				// scalar rather than adding fields, since the UBO's spare
				// room is already spent (see the struct comment above).
				// CookTorrance.frag reads the magnitude to pick door-gold
				// vs. pickup-blue, then the sign to override that with red
				// if disabled.
				float kindMag = static_cast<float>(gazedGlowKind);
				ubo.glow = glow ? (gazedInteractionDisabled ? -kindMag : kindMag) : 0.0f;

				// The ghosts' chase telegraph, riding F0 -- which the Spectral
				// technique has no use for, computing no BRDF at all (see
				// Spectral.frag's header for why this field rather than a new
				// one). AFTER the material copy above, which has just written
				// materials.json's F0 into it.
				//
				// A linear scan of a 3-element vector per instance, deliberately:
				// the alternative is a flag on Instance, which lives in
				// Scene.hpp and is off limits, or a per-frame map lookup keyed
				// by pointer, which for three ghosts costs more than this does.
				// Same shape as the glowingInstances scan just above.
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
				// declares a third set.
				if(inst.NDs[0] >= 3) {
					inst.DS[0][2]->map(currentImage, &shadowUbo, 0);
				}
			}
		}

		// Records and submits this frame's cube shadow captures -- the held
		// torch plus whatever the diffs earlier in this function found stale --
		// now that BOTH their DSshadowCube[t] matrices/position AND every
		// instance's inst.DS[0][1] world matrix (mapped for currentImage in the
		// loop just above) are current: recordCubeSlotFaces() binds the SAME
		// per-instance descriptor set the main pass uses, so recording any
		// earlier than this would bake whatever stale/uninitialized Wm
		// currentImage's buffer slot last held (on the very first call, before
		// that loop has ever run for this image index, that's garbage) into the
		// map. See submitCubeShadowCaptures() for why this is a submission of
		// its own rather than part of the "main" command buffer.
		submitCubeShadowCaptures(currentImage);
		pendingFaceMask.fill(0);

		// updates the FPS. Left running/visible even over the launch screen
		// (see the seed txt.print() in localInit() for why) -- everything
		// below this, though, reads game state that GameLogic() never even
		// updates while the launch screen is open (nearbyDoor, the hunt
		// cycle, ...), and the opaque backdrop would hide it anyway, but
		// TextMaker draws after (submit order 10000, above every UiQuad
		// instance) so a still-live text block would show through the
		// backdrop regardless. Skipping the rest of this block leaves them
		// exactly as hidden as removeText() would.
		//
		// Timed off glfwGetTime() rather than accumulated deltaT: deltaT is
		// forced to 0 while the pause menu/launch screen is open (see
		// updateUniformBuffer()'s "stop time" comment), which is exactly
		// right for gameplay but would otherwise freeze this readout too --
		// an FPS counter that stops updating while paused is telling the
		// player the wrong thing, since frames are still being rendered.
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

		if(!startScreen.isOpen()) {
			// Coordinates debug overlay (Show Coordinates cheat). Sits just above
			// the FPS line, bottom-right. Throttled to 10Hz rather than every
			// frame: print() unconditionally marks the text command buffer
			// dirty, so printing every frame would force a mesh/command-buffer
			// rebuild every frame just to show a live camera readout.
			static float coordsElapsedT = 0.0f;
			static bool coordsShown = false;
			coordsElapsedT += deltaT;
			if(cheats.showCoordinates) {
				if(!coordsShown || coordsElapsedT > 0.1f) {
					std::ostringstream coss;
					coss << "X: " << camPos.x << "  Y: " << camPos.y << "  Z: " << camPos.z
						 << "  Yaw: " << camYaw << "\n";

					// FPS is anchored at pixel-NDC (1,1), i.e. the screen's
					// bottom-right corner; this line sits a fixed pixel offset
					// above it so the two never overlap, converted through the
					// same pixelToScr TextMaker uses internally.
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

			// "[E] Interact"/"[E] Pick up" prompt, shown while a door or a pickup
			// is in range (nearbyDoor/nearbyPickup, set every frame in
			// GameLogic() -- pickups take priority, same as the E handling
			// itself). Re-prints on top of a shown/hidden toggle whenever the
			// text itself changes (e.g. walking from a door straight to the key),
			// not just on the binary transition the door-only version needed.
			// The locked-exit line rides the same slot: it's the same kind of
			// message (a one-line explanation of what the thing in front of you
			// needs), it appears in the same place, and the two can't be in range
			// at once in any layout worth building.
			static bool interactPromptShown = false;
			static std::string interactPromptText;
			//
			// Suppressed once a run has ended: GameLogic() stops updating
			// nearbyDoor/nearbyPickup/atLockedExit when it freezes, so whatever was
			// in range on the last live frame would otherwise sit there under the
			// game-over text still inviting a keypress that does nothing.
			bool showInteractPrompt = runState == RunState::Running &&
									  (nearbyDoor >= 0 || nearbyPickup >= 0 ||
									   nearbyCandle >= 0 || nearbyWallTorch >= 0 ||
									   nearbyHandTorch || atLockedExit);
			// Door prompts split four ways once locks exist: a padlock the
			// player can open ("[E] Unlock", and the wording warns the key is
			// spent, since it can't be got back), one they can't (what to go find
			// -- by lockLabel, not the raw id), one they're standing behind (no
			// key named at all: from this side there is no padlock in sight, and
			// naming one would be telling them something they can't see), and a
			// plain door.
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
				// Each of the three has a per-door override (Door::promptReady and
				// friends) that wins when it isn't empty. Only the bookcase sets
				// them: a padlock explains itself and wants the generic wording,
				// while a bookcase must never say "locked" -- it is furniture until
				// the player decides it isn't.
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

			// The hunt banner, centred and high on the screen: the words behind
			// what the torches are already saying in colour. Two lines only,
			// because a player reading a paragraph is a player not running.
			//
			// Re-printed only when the text changes, not every frame -- print()
			// unconditionally dirties the text command buffer, and the countdown is
			// therefore deliberately rounded to whole seconds so it changes at most
			// once a second instead of once a frame.
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
	}
	
	// --- Ghost navigation ---------------------------------------------------
	//
	// The ghosts obey the same walls the player does, and for the same reason
	// the player does: a threat that ignores geometry can't be played around.
	// If a ghost could drift through the dungeon wall behind you there'd be no
	// point in running anywhere, no point in the doors, and no decision left in
	// the hunt beyond holding W. Solid ghosts turn every corner and doorway
	// into something the player can use.
	//
	// None of this is pathfinding. It's a wall test, a push-out identical to
	// the player's, and a fan of candidate headings -- which is enough for a
	// castle of rooms and corridors, and is all the return trail (see the Ghost
	// struct) leaves it needing to do.

	// True if a ghost-sized cylinder standing at `p` overlaps a wall. `p` is
	// the ghost's un-bobbed position, so the vertical slab it tests is fixed.
	//
	// Unlike the player's wall pass there's no MAX_STEP_HEIGHT exemption: a
	// ghost hovers, so a low crate isn't something it steps onto, it's
	// something it floats over -- and it floats over it precisely because the
	// crate's yMax falls below the slab tested here. Same test, different
	// consequence, no special case needed.
	// `radius` is a parameter rather than always `ghostRadius` so a caller
	// deciding WHERE to go (ghostPathClear/ghostSteer) can ask with a little
	// extra padding, while the caller deciding whether `p` is actually
	// physically stuck (ghostResolveWalls) keeps asking with the real body
	// size. See ghostSteer for why that distinction exists.
	bool ghostBlockedAt(const glm::vec3 &p, float radius) const {
		for(Collider *C : allColliders) {
			AABBextents E = C->getExtents();
			if(E.yMax < p.y + ghostBodyBottom) continue;	// entirely underneath: floated over
			if(E.yMin > p.y + ghostBodyTop) continue;	// entirely overhead: passed under
			float closestX = glm::clamp(p.x, E.xMin, E.xMax);
			float closestZ = glm::clamp(p.z, E.zMin, E.zMax);
			float dx = p.x - closestX;
			float dz = p.z - closestZ;
			if(dx * dx + dz * dz < radius * radius) {
				return true;
			}
		}
		return false;
	}

	// Pushes `p` back out of anything it has ended up inside, horizontally.
	// Deliberately the same shape as the player's wall-collision block: the
	// ghost is a circle in XZ against the same boxes, and the only difference
	// is the radius and which vertical slab counts. Sliding along a wall
	// instead of sticking to it falls out of this for free, exactly as it does
	// for the player -- move first, then get pushed out along the shortest
	// horizontal escape, and the component parallel to the wall survives.
	void ghostResolveWalls(glm::vec3 &p) const {
		for(Collider *C : allColliders) {
			AABBextents E = C->getExtents();
			if(E.yMax < p.y + ghostBodyBottom) continue;
			if(E.yMin > p.y + ghostBodyTop) continue;

			float closestX = glm::clamp(p.x, E.xMin, E.xMax);
			float closestZ = glm::clamp(p.z, E.zMin, E.zMax);
			float dx = p.x - closestX;
			float dz = p.z - closestZ;
			float dist = std::sqrt(dx * dx + dz * dz);
			if(dist >= ghostRadius) continue;

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
		}
	}

	// Whether a ghost at `from` could travel `dist` along `dir` without hitting
	// anything. Sampled rather than swept: a handful of point tests along the
	// segment, spaced under the ghost's own radius so nothing thinner than the
	// ghost can slip between two samples. A real swept test against every
	// collider would cost more and buy nothing at these speeds.
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

	// True if the exact point `p` sits inside any collider's box. Unlike
	// ghostBlockedAt this tests the real Y of `p`, not a vertical slab hung
	// off some other position's height -- what a sightline needs, since the
	// ray runs from a hovering ghost's eye down to a standing player's, and a
	// table or chair along the way should only block it if the line is
	// actually low enough to clip the furniture there.
	bool ghostPointBlocked(const glm::vec3 &p) const {
		for(Collider *C : allColliders) {
			AABBextents E = C->getExtents();
			if(p.x < E.xMin || p.x > E.xMax) continue;
			if(p.z < E.zMin || p.z > E.zMax) continue;
			if(p.y < E.yMin || p.y > E.yMax) continue;
			return true;
		}
		return false;
	}

	// Roughly where a ghost's "eyes" are, relative to its hover pivot --
	// nearer the top of the body than the centre. Only used for the sightline
	// below; the movement/collision code has no use for it.
	static constexpr float GHOST_EYE_OFFSET = 0.5f;

	// Whether a ghost at `from` can currently see the player at `eyeTarget`
	// (already an eye-height position -- see camPos). A straight 3D ray, not
	// the flat XZ probe ghostPathClear uses for walking: that one tests a
	// fixed vertical slab at the ghost's own height and would call a
	// waist-high table "blocking" even though a hovering ghost looking down
	// at a player clears right over it. Walls and closed doors still block --
	// their boxes run floor to ceiling, so no point on the ray between two
	// eye heights ever misses them.
	bool ghostHasLineOfSight(const glm::vec3 &from, const glm::vec3 &eyeTarget) const {
		glm::vec3 eyeFrom = from + glm::vec3(0.0f, GHOST_EYE_OFFSET, 0.0f);
		glm::vec3 delta = eyeTarget - eyeFrom;
		float d = glm::length(delta);
		if(d < 1e-4f) return true;
		glm::vec3 dir = delta / d;
		const float step = ghostRadius * 0.8f;
		int samples = std::max(1, (int)std::ceil(d / step));
		// i starts at 1 and stops short of `samples` so neither endpoint --
		// the ghost's own eye position, or the player's -- is tested against
		// their own collider footprint.
		for(int i = 1; i < samples; i++) {
			float t = d * (float)i / (float)samples;
			if(ghostPointBlocked(eyeFrom + dir * t)) {
				return false;
			}
		}
		return true;
	}

	// Picks the heading a chasing ghost should actually take, given the
	// direction it WANTS to go (straight at the player).
	//
	// Fans out from `desired` in widening steps and takes the first candidate
	// with a clear probe ahead of it. Straight at the player wins whenever it's
	// available; when it isn't, the fan finds the shallowest deviation that is,
	// which along a wall is the direction that slides down it and around a
	// corner is the one that turns the corner. Backwards (the last candidates)
	// is a valid answer too -- that's a ghost giving up on a dead end.
	//
	// g.turnBias is why it doesn't dither. Without it, a ghost facing a pillar
	// with the player behind it would evaluate left and right as equally good
	// every frame and, as the geometry shifted by centimetres, keep swapping --
	// vibrating in place instead of committing. The bias remembers the side it
	// chose and re-tries that side first, and is only re-examined once the
	// ghost gets a clear straight line again.
	//
	// Returns a zero vector if it's boxed in on every side, which the caller
	// reads as "don't move this frame".
	glm::vec2 ghostSteer(Ghost &g, const glm::vec2 &desired) const {
		// Probed with a little more than the ghost's actual radius, not the
		// exact physical size ghostResolveWalls uses to push it out of a wall.
		// A choke point only a hair wider than the body makes "clear?" flip
		// between true and false from one frame's worth of movement -- which,
		// fed straight into a direction, is a ghost snapping between two
		// headings every frame instead of walking through. The padding turns
		// that knife-edge into a threshold the ghost commits to well before it
		// physically has to, at the cost of refusing a gap slightly sooner
		// than it strictly needs to.
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

	// Puts everything a run touches back to its authored state: the player, the
	// ghosts, the hunt clock, the doors, and anything picked up or dropped.
	// Nothing here reloads a file -- every "authored" value was captured in
	// localInit() (spawnPos, Door::baseWm, Pickup::spawnWm), so a restart can't
	// disagree with the scene the game started from.
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
			// Padlocks come back with the run: a key spent last run is back
			// on its table below, so the lock it opened has to be shut again
			// or the level would get easier every restart.
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
		// A key caught mid-fall would otherwise keep being drawn off the
		// camera -- and worse, get parked below the map a few frames into the
		// new run, right after the loop above put it back on its table.
		keyLowerIdx = -1;

		// The torch goes back on the floor, and every candle the player lit
		// goes back out (the torch's flame with them -- spawnBurning == false
		// for it), for the same reason the doors above are re-locked: a
		// castle that keeps the light it was given gets brighter with every
		// restart, and the whole point is that the player earns the light --
		// starting with finding and lighting their own torch. The instance's
		// Wm is put back to the authored floor pose by updateUniformBuffer()
		// on the next frame, once handTorchCollected is false again.
		handTorchCollected = false;
		for(TorchFlame &tf : torchFlames) {
			tf.burning = tf.spawnBurning;
		}

		nearbyDoor = -1;
		nearbyPickup = -1;
		nearbyCandle = -1;
		nearbyWallTorch = -1;
		nearbyHandTorch = false;
		gazedInstance = nullptr;
		atLockedExit = false;
		// The way out closes again with the rest of the doors (the loop above
		// re-locks and re-shuts every one), so the daylight behind it has to
		// go with them -- otherwise a restarted run would begin with a lit
		// doorway and no door open to explain it. Same for the whiteout: it is
		// the end of a run, and this is a new one.
		exitOpenFrac = 0.0f;
		escapeFlash = 0.0f;

		runState = RunState::Running;
	}

	// Called once on the frame the hunt cycle changes phase. Everything the
	// phase change actually DOES (colour, ghost behaviour) is polled from
	// huntCycle where it's needed, so this is left with just the announcement
	// -- and it exists as its own function because that's the single place a
	// music track would be swapped from. The project has no audio backend yet
	// (nothing links an audio library, see CMakeLists.txt), so for now it
	// prints; when one is added, three calls go here and nothing else moves.
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

		// Poll/render the launch screen, cheat HUD and pause menu BEFORE
		// getSixAxis. getSixAxis turns on GLFW_STICKY_MOUSE_BUTTONS, which
		// makes glfwGetMouseButton a one-shot read (it flips back to
		// "released" once polled). Reading their own click hit-tests first
		// guarantees they get that one authoritative read of a click, not
		// getSixAxis's drag-look check.
		//
		// The HUD only gets to react while neither the pause menu nor the
		// launch screen is open -- one modal overlay at a time, and ESC
		// (updateUniformBuffer()) is the pause menu's own equivalent guard
		// against L while paused.
		if(!pauseMenu.isOpen() && !startScreen.isOpen()) {
			hud.update(window, windowWidth, windowHeight);
		}

		startScreen.update(window, windowWidth, windowHeight);
		if(startScreen.playClicked()) {
			startScreen.setOpen(false, windowWidth, windowHeight);
		}
		if(startScreen.quitClicked()) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}

		pauseMenu.update(window, windowWidth, windowHeight);
		if(pauseMenu.resumeClicked()) {
			pauseMenu.setOpen(false, windowWidth, windowHeight);
		}
		if(pauseMenu.quitClicked()) {
			// Abandon the current run and drop back to the launch screen --
			// restartRun() puts every piece of world state (camera, doors,
			// keys, ghosts, candles) back to spawn, the same reset a fresh
			// Play needs, so the next Play click starts clean either way.
			pauseMenu.setOpen(false, windowWidth, windowHeight);
			restartRun();
			startScreen.setOpen(true, windowWidth, windowHeight);
		}

		getSixAxis(deltaT, m, r, fire);

		// Clamped AFTER getSixAxis (so input timing itself is untouched) but
		// BEFORE anything below integrates physics with it. Gravity is
		// plain Euler integration, camPos.y += camVerticalVelocity * deltaT
		// with no clamp: on a slow or momentarily stalled frame (asset/
		// pipeline work, a GPU driver hitch) deltaT spikes, one frame's fall
		// overshoots every collider's AABB, and the ground pass below has
		// nothing "near the feet" left to catch it on -- the player falls
		// straight through the floor. 1/20s caps a single frame's fall to
		// what a normal frame would produce even during a multi-frame stall;
		// the game just briefly slows down instead of skipping physics
		// entirely, which is the standard fix for Euler integration on a
		// variable timestep.
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

		// The whiteout, ramped here rather than in the frozen-movement block
		// below precisely because this is the one thing that has to keep
		// running after the run is over: RunState::Escaped is what starts it.
		// Held still while the cheat HUD or pause menu is open, like
		// everything else, so pausing mid-flash doesn't skip past it.
		if(!overlayOpen()) {
			float flashTarget = (runState == RunState::Escaped) ? 1.0f : 0.0f;
			if(flashTarget > escapeFlash) {
				escapeFlash = std::min(escapeFlash + deltaT / ESCAPE_FLASH_SECONDS, 1.0f);
			} else {
				escapeFlash = flashTarget;
			}
		}

		// The hunt clock, gated on the same two conditions as the movement
		// block below. A cycle that kept counting down behind an open cheat
		// menu or pause menu, or over a game-over screen, would have the
		// player come back to a phase they never saw start.
		if(!overlayOpen() && runState == RunState::Running) {
			huntCycle.update(deltaT);
			if(huntCycle.phaseJustChanged()) {
				onHuntPhaseChanged();
			}
		}

		// Freeze all movement/physics while the cheat HUD or pause menu is
		// open, so opening either pauses the game exactly where it was
		// (camera included, since m/r were already zeroed above). A
		// finished run freezes the same way, for the same reason: the last
		// frame the player saw is the one they should keep looking at while
		// deciding whether to restart.
		if(!overlayOpen() && runState == RunState::Running) {
			// Sprint: Ctrl multiplies movement speed. Polled directly (not through
			// getSixAxis/"fire") since Starter.hpp doesn't wire Ctrl to anything.
			// Can only be started while grounded (no starting a sprint mid-jump), but
			// releasing Ctrl always stops it right away, air or not.
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

			// Update position from WASD: m.x = strafe, m.z = -forward. The m.y the
			// Starter fills in from R/F is deliberately dropped: that's the template's
			// free-camera fly control, and feeding it to a body that has gravity just
			// makes you hop while the ground snap yanks you back down. Vertical
			// movement only ever comes from jumping and gravity.
			// Forward/strafe movement is flattened to the horizontal plane (yaw only), not
			// the full pitch-tilted `front` used for looking around: otherwise looking up
			// and pressing W pushes you upward (feels like a jump), and looking up while
			// walking backward pushes you down through the floor.
			glm::vec3 frontFlat = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
			camPos += (right * m.x - frontFlat * m.z) * moveSpeed * deltaT;

			// (Wall) Collision resolution: push the camera back out of any collider that
			// counts as a wall, so walking into a wall/pillar/gate pier stops you
			// horizontally instead of clipping through.
			// A collider is a wall only if it's both too tall to step onto and low enough
			// for our body to reach it. Everything else is left to the ground pass further
			// down, which lifts us on top of it: that split is what makes low obstacles
			// (steps, crates) walkable instead of solid.
			if(cheats.collisionEnabled) {
				const float EYE_HEIGHT = 1.8f;
				const float PLAYER_HEIGHT = 1.8f;
				const float PLAYER_RADIUS = 0.3f;
				// Small vertical margin so a collider whose underside we're passing
				// right below (the gate's lintel) isn't treated as something we're
				// "inside" the moment our head grazes its lower bound.
				const float VERTICAL_MARGIN = 0.1f;
				float feetY = camPos.y - EYE_HEIGHT;
				float headY = feetY + PLAYER_HEIGHT;
				for(Collider *C : allColliders) {
					AABBextents E = C->getExtents();

					// Low enough to step onto: not a wall, so don't push away from it.
					// The ground pass further down uses the same MAX_STEP_HEIGHT to lift
					// the player on top of it instead. This also covers everything at or
					// below foot level (floors, and anything we're falling past above).
					if(E.yMax <= feetY + MAX_STEP_HEIGHT) continue;

					// Too tall to step onto, so it's a wall, but only for the part of it
					// our body actually reaches: the gate's lintel spans the archway, yet
					// its underside is above head height, so we walk through underneath.
					// (No feet-side test is needed here: getting past the check above
					// already means yMax is well over our feet.)
					if(headY <= E.yMin + VERTICAL_MARGIN) continue;

					// Closest point on the collider's XZ footprint to the camera
					float closestX = glm::clamp(camPos.x, E.xMin, E.xMax);
					float closestZ = glm::clamp(camPos.z, E.zMin, E.zMax);
					float dx = camPos.x - closestX;
					float dz = camPos.z - closestZ;
					float dist = std::sqrt(dx * dx + dz * dz);

					if(dist < PLAYER_RADIUS) {
						if(dist > 1e-5f) {
							// Push the camera away from the collider along the horizontal
							// vector to the closest surface point, just past the radius.
							float push = (PLAYER_RADIUS - dist) / dist;
							camPos.x += dx * push;
							camPos.z += dz * push;
						} else {
							// Camera's XZ is exactly inside the footprint (e.g. spawned
							// there): push out along whichever side is nearest.
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

			// Jump: spacebar (wired to "fire" in Starter.hpp) gives the camera an upward
			// velocity impulse. Edge-triggered (only on the frame the key goes down) and
			// only while grounded (refreshed each frame by the floor collision check
			// below).
			if(fire && !jumpKeyWasPressed && grounded) {
				camVerticalVelocity = movement.jumpSpeed;
				// Drop any leftover step smoothing: jumping right after
				// stepping up would otherwise start the jump from a view
				// still trailing below the real eye height.
				eyeStepOffset = 0.0f;
			}
			jumpKeyWasPressed = fire;

			// Interaction: find the door the player is aiming the crosshair
			// at (findGazedDoor), then re-confirm it's within
			// DOOR_INTERACT_RADIUS -- gaze picks the target, proximity still
			// gates whether it's actually reachable, so staring down a long
			// hallway at a far door doesn't light it up early. Toggle on E
			// (edge-triggered, same pattern as jump), then ease every door's
			// animated angle toward its target and push the result into both
			// the render transform and its collider, so an open door is
			// actually walkable and a closed one still blocks.
			nearbyDoor = -1;
			{
				int gazed = findGazedDoor(front);
				if(gazed >= 0 && doorDistance(doors[gazed], camPos) < DOOR_INTERACT_RADIUS) {
					nearbyDoor = gazed;
				}
			}

			// Pickups: same gaze-then-proximity check as the door, and same
			// edge-triggered E. Checked independently of the door above; a
			// pickup found here still wins over a door below via the E-key
			// handling's own priority order, same as before.
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

			// Candles: an unlit one within reach, aimed at, is lit by [E] off
			// the torch in the player's hand. Same gaze-then-proximity pair as
			// the two above, in 3D like the pickup's since a candle stands on
			// furniture. Whether the player actually HAS fire to light it with
			// isn't checked here: the candle still gets targeted, so the
			// disabled-glow and the prompt below can explain why nothing
			// happens, which is more use than the candle silently not
			// responding.
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

			// The floor torch: aimed at, within reach, and not yet picked
			// up. Uses the ordinary pickup look/interact/aim tolerances --
			// it's the same kind of small floor object. findGazedHandTorch
			// already returns false once it's been collected.
			nearbyHandTorch = false;
			if(findGazedHandTorch(front)) {
				float dx = camPos.x - handTorchWorldPos.x;
				float dy = camPos.y - handTorchWorldPos.y;
				float dz = camPos.z - handTorchWorldPos.z;
				if(std::sqrt(dx * dx + dy * dy + dz * dz) < PICKUP_INTERACT_RADIUS) {
					nearbyHandTorch = true;
				}
			}

			// Whichever single instance is currently targeted (floor torch
			// and pickup first, then wall torch, then candle, then door --
			// same priority as the E-key handling below), for the
			// per-instance focus glow -- see ubo.glow in
			// updateUniformBuffer(). Candles and wall torches sit above doors
			// because one stands in front of a wall or a doorway more often
			// than not, and the smaller, nearer thing is the one the player
			// means.
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
					// Into the hand it goes, still unlit. From the next frame
					// updateUniformBuffer() rebuilds its Wm off the camera
					// instead of leaving it at the authored floor pose, and
					// findGazedWallTorch() starts offering a flame to light
					// it from.
					handTorchCollected = true;
					std::cout << "[torch] picked up hand torch\n";
					nearbyHandTorch = false;
					gazedInstance = nullptr;
				} else if(nearbyPickup >= 0) {
					Pickup &p = pickups[nearbyPickup];
					p.collected = true;
					// No per-instance visibility flag exists (Starter draws
					// every instance every frame), so "removed from the
					// world" means parked far below the map instead. The key
					// specifically gets redrawn in the player's hand every
					// frame from here on (see the held-key block below);
					// a pickup with no such block just stays parked here.
					p.inst->Wm = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
					// Keys additionally go on the ring, which is what makes
					// the newest one show up in the hand and what the locks
					// below are checked against.
					if(!p.keyId.empty()) {
						// One free hand, one key in it: whatever was already
						// there goes back on the floor instead of staying
						// frozen wherever the hand last drew it. Only the
						// newest key on the ring is drawn (see the held-key
						// block below), so an older one left on the ring would
						// hang in mid-air and be unreachable -- dropping it
						// keeps it collectable. Dropped short (0.5 rather than
						// G's 1.0) so it lands underfoot instead of being
						// flung past the key just taken.
						if(!keyRing.empty()) {
							dropKeyFromRing((int)keyRing.size() - 1, camPos, front, 0.5f);
						}
						keyRing.push_back(nearbyPickup);
						// Restart the raise: the key is drawn from the next
						// frame on, and it should come up from below rather
						// than appear already in place.
						keyRaiseElapsed = 0.0f;
					}
				} else if(nearbyWallTorch >= 0) {
					// Lighting the held torch is the same single bool as a
					// candle: flip `burning` on the held flame and flameBurning()
					// turns its billboard, sparks and point light on this frame.
					// From here hasBurningTorch() is true, so candles can be lit.
					torchFlames[handFlameIdx].burning = true;
					std::cout << "[torch] lit hand torch from '"
							  << *torchFlames[nearbyWallTorch].inst->id << "'\n";
					nearbyWallTorch = -1;
					gazedInstance = nullptr;
				} else if(nearbyCandle >= 0) {
					// Lighting a candle is one bool: `burning` is what flameBurning()
					// answers with, and every consequence of a burning flame
					// -- the billboard, the sparks, the point light, the
					// dynamic shadow slot it now competes for -- already asks
					// through flameBurning() and turns itself on this frame. See
					// TorchFlame::burning for why nothing is spawned here.
					//
					// Only with fire in hand. Nothing is reported on failure:
					// the red aura and the prompt have been saying so for as
					// long as the player has been aiming at it.
					if(hasBurningTorch()) {
						torchFlames[nearbyCandle].burning = true;
						std::cout << "[candle] lit '"
								  << *torchFlames[nearbyCandle].inst->id << "'\n";
						// The candle is lit from here on, so it stops being a
						// target this frame rather than staying aimed at with
						// a prompt inviting a press that would do nothing.
						nearbyCandle = -1;
						gazedInstance = nullptr;
					}
				} else if(nearbyDoor >= 0) {
					Door &d = doors[nearbyDoor];
					if(d.locked) {
						// Padlocked: E spends a matching key instead of
						// toggling. The key is destroyed by the unlock (usa e
						// getta), and the door swings open in the same press
						// rather than needing a second one -- a player who
						// just gave up a key expects the door to move.
						// Without a match nothing happens at all: the prompt
						// (see updateUniformBuffer) is already telling them
						// what's missing, so there's nothing to report here.
						// ...and only from the face the padlock hangs on: a
						// key can't be turned in a lock on the far side of a
						// closed door. The prompt on the blind side says so
						// (see updateUniformBuffer), so again nothing to
						// report here.
						int slot = d.onLockSide(camPos) ? findKeyInRing(d.lockKeyId) : -1;
						if(slot >= 0) {
							std::cout << "[door] unlocked '" << d.instanceId
									  << "' with key '" << d.lockKeyId << "'\n";
							Instance *spent = pickups[keyRing[slot]].inst;
							consumeKey(slot);
							// A door that has this very item as a whenUnlocked
							// prop takes it IN rather than off the board, so
							// the sink-out-of-frame animation consumeKey() just
							// queued is wrong for it: the book would drop out
							// of the bottom of the screen and then wink into
							// existence on a shelf a third of a second later.
							// Cancelling the sink hands the instance to the
							// prop loop below on this same frame, and "it went
							// from your hand into the gap" is all one motion.
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
						// Opening from fully closed is the one moment the leaf is
						// free to pick a side, so that is where the player's own
						// side is read. Re-opening a leaf that is still swinging
						// shut keeps the sign it already has: reversing it there
						// would drag the panel back through the doorway -- and
						// through the player -- to reach the mirrored pose.
						if(!d.open && d.angle == 0.0f) {
							d.swingSign = d.swingSignAwayFrom(camPos);
						}
						d.open = !d.open;
					}
				}
			}
			interactKeyWasPressed = interactKey;

			// Drop key (G): puts the held key back down at arm's length, see
			// dropKeyFromRing() for the pose. Drops the key in hand (the last
			// one collected); the rest of the ring stays put, so pressing G
			// repeatedly puts them down one at a time in reverse order of
			// pickup.
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

				// The leaf's own local origin IS its hinge (see the Door
				// struct comment), so opening it is just one more rotation
				// tacked onto the authored closed-door transform -- no
				// separate world-space hinge point to sandwich it between.
				d.inst->Wm = d.baseWm * glm::rotate(glm::mat4(1.0f), glm::radians(d.angle), glm::vec3(0.0f, 1.0f, 0.0f));
				if(d.inst->C != nullptr) {
					d.inst->C->setWorldMatrix(d.inst->Wm);
				}

				// Chains and padlock: the leaf's own matrix while the lock
				// holds (they're modelled in its local frame, so that IS
				// their pose, bar the half turn a flipped set carries),
				// parked below the map once it doesn't. Driven
				// from here every frame rather than only on the unlock press,
				// so restartRun() re-locking a door brings them back with no
				// extra bookkeeping.
				//
				// A whenUnlocked prop is the mirror image and gets the mirrored
				// treatment: it appears when the lock comes off (the book that
				// paid for the bookcase, now standing in the gap on its shelf),
				// and while the door is still locked it is left ALONE rather
				// than parked -- see LockProp::whenUnlocked for why parking it
				// would bury the book that is still lying on its table.
				for(const Door::LockProp &prop : d.lockProps) {
					if(prop.whenUnlocked) {
						if(!d.locked) prop.inst->Wm = d.inst->Wm * prop.local;
					} else {
						prop.inst->Wm = d.locked ? d.inst->Wm * prop.local
												 : glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
					}
					// The hardware's own collider, off the same matrix. This is
					// what makes the box safe to have: one left behind would seal
					// the doorway, one that follows the prop below the map cannot.
					// Null for a whenUnlocked prop -- those are Pickup instances,
					// whose models carry no collider.
					if(prop.inst->C != nullptr) {
						prop.inst->C->setWorldMatrix(prop.inst->Wm);
					}
				}
			}

			// How far the way out has swung, off the same animated angle the
			// leaf is drawn with, so the light outside can never disagree with
			// the door on screen. Read by updateUniformBuffer() for both the
			// daylight quad and the spill light -- and by nothing else.
			//
			// Smoothstepped rather than linear because the two ends are what
			// matter: the light should stay out of it while the leaf is barely
			// moving, build through the middle of the swing, and settle
			// instead of arriving at full strength on the last degree.
			//
			// Left frozen at its last value once the run ends: this whole
			// block stops updating then, and a door that stayed open is
			// exactly what should still be lighting the room behind the
			// escape banner.
			if(exitDoorIndex >= 0) {
				const Door &exitDoor = doors[exitDoorIndex];
				float span = std::abs(exitDoor.openAngleDeg);
				exitOpenFrac = span > 1e-4f
					? glm::smoothstep(0.0f, 1.0f, glm::clamp(std::abs(exitDoor.angle) / span, 0.0f, 1.0f))
					: 0.0f;
			}

			// Ghosts. See the Ghost struct for the three modes and why Return
			// works the way it does.
			//
			// Common to all three: the mode decides a movement direction and a
			// speed, and everything after that -- the bob, the eased facing,
			// the world matrix -- is shared. The bob is added at the very end,
			// on top of `pos` rather than into it, so it never feeds back into
			// either the steering or the collision slab.
			bool ghostsHunting = huntCycle.hunting();
			for(Ghost &g : ghosts) {
				if(g.inst == nullptr || g.waypoints.size() < 2) continue;

				// --- Line of sight, checked every frame a hunt is on regardless
				// of mode: a Patrol or Return ghost that spots the player needs
				// to be able to start/resume a chase, not just a Chase one that
				// already has a target to refresh.
				if(ghostsHunting && ghostHasLineOfSight(g.pos, camPos)) {
					g.lastKnownPlayerPos = camPos;
					g.hasLastKnown = true;
				}

				// --- Mode transitions.
				//
				// Starting (or resuming) a chase now needs g.hasLastKnown, not
				// just the phase being Hunt: a ghost that has never seen the
				// player has nothing to walk toward, so it stays on patrol
				// instead of beelining for coordinates it was never shown --
				// which is exactly what used to put a ghost behind a door in
				// another room the player was about to open.
				if(ghostsHunting && g.hasLastKnown && g.mode != GhostMode::Chase) {
					if(g.mode == GhostMode::Patrol) {
						// Remember exactly where on the loop we're leaving
						// from, and start a fresh trail at that same point.
						g.resumeIdx = g.targetIdx;
						g.resumeDist = g.distAlongSegment;
						g.trail.clear();
						g.trail.push_back(g.pos);
					}
					// Coming back out of Return instead, the existing trail and
					// resume point are still the way home -- a second hunt
					// starting mid-return just extends the same trail.
					g.mode = GhostMode::Chase;
					g.stuckCheckPos = g.pos;
					g.stuckTimer = 0.0f;
				} else if(!ghostsHunting && g.mode == GhostMode::Chase) {
					// An empty trail means the chase never went anywhere, so
					// there's nothing to walk back.
					g.mode = g.trail.empty() ? GhostMode::Patrol : GhostMode::Return;
				}

				// --- Move, per mode. Each branch leaves a `moveDir` (XZ, unit
				// or zero) for the facing code below.
				glm::vec2 moveDir(0.0f);

				if(g.mode == GhostMode::Chase) {
					// Toward the last place the player was actually seen, not
					// their live position -- see ghostHasLineOfSight above.
					glm::vec2 toPlayer(g.lastKnownPlayerPos.x - g.pos.x, g.lastKnownPlayerPos.z - g.pos.z);
					float d = glm::length(toPlayer);
					if(d > 1e-4f) {
						moveDir = ghostSteer(g, toPlayer / d);
						if(moveDir != glm::vec2(0.0f)) {
							// min(step, d): stops the ghost overshooting
							// straight past a player it has already reached,
							// which at chase speed on a long frame it otherwise
							// can -- and overshooting is how you get a ghost
							// that passes THROUGH the player without the catch
							// test below ever seeing them close.
							float step = std::min(g.chaseSpeed * deltaT, d);
							g.pos.x += moveDir.x * step;
							g.pos.z += moveDir.y * step;
							ghostResolveWalls(g.pos);
						}
					}

					// Giving up: a ghost pinned against a closed door and one
					// standing on a stale lastKnownPlayerPos with nobody there
					// both look the same from here -- no meaningful ground
					// covered -- so one timer catches both instead of needing
					// an "arrived" check that the door case would never trip.
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

					// Breadcrumb. Dropped by distance travelled, not by time,
					// so the trail's density doesn't depend on the frame rate.
					if(g.trail.empty()) {
						g.trail.push_back(g.pos);
					} else if(glm::length(glm::vec2(g.pos.x - g.trail.back().x,
													g.pos.z - g.trail.back().z)) >= GHOST_TRAIL_SPACING) {
						// Before adding it: does this land back on a stretch we
						// already walked? If so the shortest way home from here
						// is that older crumb, and everything recorded since is
						// a detour worth throwing away. Searched oldest-first
						// so the biggest loop is the one that gets cut.
						//
						// The last two crumbs are excluded because they're
						// necessarily within the prune radius of where we are
						// (spacing is smaller than that radius, deliberately),
						// and matching them would prune the trail back to
						// nothing on every single step.
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
					// Walk the breadcrumbs backwards, popping each as it's
					// reached. Faster than the chase (see GHOST_RETURN_SPEED):
					// this is time the player isn't being threatened in.
					glm::vec3 target = g.trail.back();
					glm::vec2 delta(target.x - g.pos.x, target.z - g.pos.z);
					float d = glm::length(delta);
					float step = GHOST_RETURN_SPEED * deltaT;
					if(d <= step) {
						// Reached it: land exactly on it (so the next leg
						// starts from a point we know is walkable) and drop it.
						g.pos.x = target.x;
						g.pos.z = target.z;
						g.pos.y = target.y;
						if(d > 1e-4f) moveDir = delta / d;
						g.trail.pop_back();
						if(g.trail.empty()) {
							// Home. trail[0] was the patrol position at the
							// moment the chase started, so restoring the saved
							// leg and distance resumes the loop mid-stride
							// rather than snapping to the nearest waypoint.
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
					// Patrol: walks its waypoint loop at constant speed
					// (distance-based, not time-based, so `speed` is an actual
					// world-units/second figure regardless of leg length),
					// facing the leg it's currently on.
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
					// The patrol is authored, not steered: it's assumed clear,
					// and it drives `pos` directly with no wall resolution, so
					// a ghost can't be shoved off its own loop by a collider
					// somebody adds next to it.
					g.pos = glm::mix(from, to, t);
					if(segLen > 0.0f) {
						moveDir = glm::normalize(glm::vec2(to.x - from.x, to.z - from.z));
					}
				}

				// --- Facing, eased toward the direction of travel. A ghost
				// that isn't moving (boxed in, or standing on the player) keeps
				// the yaw it had rather than snapping to some default.
				//
				// +M_PI: the ghost mesh's modeled front faces -Z, not the +Z
				// atan2(x, z) assumes -- confirmed by it walking backwards
				// without this.
				if(moveDir != glm::vec2(0.0f)) {
					float targetYaw = std::atan2(moveDir.x, moveDir.y) + (float)M_PI;
					// Shortest way round: without this, easing from +179 to
					// -179 degrees takes the long way and spins the ghost.
					float dYaw = targetYaw - g.yaw;
					while(dYaw > (float)M_PI)  dYaw -= 2.0f * (float)M_PI;
					while(dYaw < -(float)M_PI) dYaw += 2.0f * (float)M_PI;
					float maxStep = GHOST_TURN_SPEED * deltaT;
					g.yaw += glm::clamp(dYaw, -maxStep, maxStep);
				}

				// --- The chase telegraph. Same exponential ease as walkBobBlend
				// above, and here rather than in updateUniformBuffer() because
				// this is the loop that owns `mode` and the one with a deltaT.
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

				// --- The catch. Only a hunting ghost can end the run: one
				// you've walked into on patrol is scenery, and killing the
				// player for brushing past a ghost that isn't chasing them
				// would make the colour telegraph a lie.
				if(ghostsHunting && cheats.ghostsCanCatch && runState == RunState::Running) {
					float dx = camPos.x - g.pos.x;
					float dz = camPos.z - g.pos.z;
					// Measured from the player's chest, not their eyes: the
					// eye height (1.8) is the top of the body, and comparing a
					// hovering ghost against it would read as half a metre
					// further away vertically than it is.
					float dy = std::abs((camPos.y - 0.9f) - g.pos.y);
					if(dx * dx + dz * dz < GHOST_CATCH_RADIUS * GHOST_CATCH_RADIUS &&
					   dy < GHOST_CATCH_VERTICAL) {
						runState = RunState::Caught;
						std::cout << "[run] caught by '" << g.instanceId << "'\n";
					}
				}
			}

			// The way out. Standing in the exit box wins the run -- unless it's
			// locked and the key isn't in hand, in which case atLockedExit
			// flags it so updateUniformBuffer() can say why. Checked after the
			// ghosts so a ghost catching the player on the threshold beats
			// reaching it, which is the reading that makes the last few metres
			// tense instead of a formality.
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

			// Gravity: constant downward acceleration, integrated into a vertical
			// velocity each frame. Resolved against the ground below (collision
			// block right after this), which zeroes the velocity out on landing.
			camVerticalVelocity += movement.gravity * deltaT;
			camPos.y += camVerticalVelocity * deltaT;

			// (Floor) Collision detection.
			// Wired-in using Scene.hpp and Colliders.hpp.
			// Look for every solid collider whose horizontal position is under the current position,
			// then update the camera's vertical position so that the player's feet don't clip into it.
			// In case of non-flat meshes, take the tallest surface as the standing height.
			// Same thing in case of multiple colliders, always take the tallest surface.
			// Only surfaces close to the feet (within MAX_STEP_HEIGHT) qualify as ground, so
			// gates, doors and other hole-shaped models allow the player to pass through.
			if(cheats.collisionEnabled) {
				const float EYE_HEIGHT = 1.8f;
				// Compute feet height from the (camera) eye height
				float feetY = camPos.y - EYE_HEIGHT;
				float groundY = -std::numeric_limits<float>::infinity();
				for(Collider *C : allColliders) {
					// for every collider, check collision
					AABBextents E = C->getExtents();
					bool insideXZ = camPos.x >= E.xMin && camPos.x <= E.xMax &&
									camPos.z >= E.zMin && camPos.z <= E.zMax;
					// Only consider surfaces near the feet (a small step up, or below), not
					// anything towering overhead (e.g. the top of a wall being walked past).
					bool nearFeet = E.yMax <= feetY + MAX_STEP_HEIGHT;
					// Update with the highest (max) surface found so far
					if(insideXZ && nearFeet && E.yMax > groundY) {
						groundY = E.yMax;
					}
				}

				// Sloped surfaces (the staircase). Same MAX_STEP_HEIGHT rule as the
				// boxes above, so a ramp still can't be entered from underneath, but
				// the height it reports varies continuously along the slope instead
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
				// Ground-contact test for jumping.
				// Allows to jump only when within a certain distance threshold from the ground.
				// Some tolerance allows to jump even when irregular floor slightly lifts the
				// player's model from the ground
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
				// No-clip: walls/objects are ignored entirely (handled by the blocks
				// above being skipped), but the world floor still acts as a hard
				// floor, so no-clipping lets you walk through walls without letting
				// you fall out of the map underneath it.
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

		// Decay the vertical view smoothing. Exponential rather than linear: it
		// never overshoots, and it needs no "am I still stepping" state, since a
		// continuous climb (walking up a ramp) just settles at a small constant
		// lag instead of oscillating.
		eyeStepOffset *= std::exp(-deltaT / EYE_SMOOTH_TAU);

		// View: rendered from the smoothed eye height. camPos itself is left
		// untouched, so collisions, gravity and ground contact all keep working
		// on the exact position; only what the player sees is eased.
		glm::vec3 eyePos = camPos - glm::vec3(0.0f, eyeStepOffset, 0.0f);
		View = glm::lookAt(eyePos, eyePos + front, up);

		// View-Projection
		ViewPrj = Prj * View;

		// Camera-space basis for anything rigidly attached to the view (held
		// torch, held key): right/up/-front as columns, eyePos as the
		// translation. front is negated since the camera looks down its own
		// local -Z. Shared by both blocks below so they stay in lockstep.
		glm::mat4 camWm = glm::mat4(
			glm::vec4(right, 0.0f),
			glm::vec4(up, 0.0f),
			glm::vec4(-front, 0.0f),
			glm::vec4(eyePos, 1.0f)
		);

		// Walk-bob signal shared by both hands: one accumulating phase, eased
		// in/out by walkBobBlend so a start/stop doesn't snap the sway.
		// Originally the torch's own state, now doubles for the key since
		// both hands swing with the same gait -- only how each hand reads
		// the phase (see bobLateral's sign below) differs between them.
		bool isWalking = grounded && (std::abs(m.x) > 0.01f || std::abs(m.z) > 0.01f);
		float bobTarget = isWalking ? 1.0f : 0.0f;
		walkBobBlend += (bobTarget - walkBobBlend) * (1.0f - std::exp(-deltaT / WALK_BOB_BLEND_TAU));
		if(isWalking) {
			walkBobPhase += WALK_BOB_SPEED * (sprinting ? 1.4f : 1.0f) * deltaT;
		}

		// Wall tuck for both hands (see applyTuck and the block of constants
		// around it). Resolved here rather than inside each hand's own block
		// below for two reasons: both targets get probed against the same camWm
		// on the same frame, and both factors have to advance on EVERY frame --
		// including the ones where a hand is empty. Left un-advanced, a factor
		// would stay pinned at whatever it read when the item left the hand, and
		// the next pickup would appear already folded into a wall pose.
		//
		// No-clip is the one case with no sensible answer: with
		// cheats.collisionEnabled off the player walks through walls deliberately,
		// so there is nothing to rest an item against and both hands stay out.
		{
			bool tuckActive = cheats.collisionEnabled;

			float torchTarget = 1.0f;
			if(tuckActive && handTorchInst != nullptr && handTorchCollected && cheats.handTorchEnabled) {
				torchTarget = handFreeReach(camWm, HAND_TORCH_OFFSET, HAND_TUCK_TORCH_PAD);
			}
			advanceReach(handTorchReach, torchTarget, deltaT);

			// Whichever key is actually drawn in the left hand drives it: the
			// held one, or -- while that animation runs -- the one sinking out
			// of frame after a lock took it. One factor for both, since they are
			// never in frame together and it is the same hand regardless.
			int tuckKeyIdx = heldKeyIdx() >= 0 ? heldKeyIdx() : keyLowerIdx;
			float keyTarget = 1.0f;
			if(tuckActive && tuckKeyIdx >= 0) {
				keyTarget = handFreeReach(camWm, pickups[tuckKeyIdx].handOffset, HAND_TUCK_KEY_PAD);
			}
			advanceReach(handKeyReach, keyTarget, deltaT);
		}

		// Held torch: sits at a fixed offset from the eye, in that camera-local space --
		// but only once the player has picked it up (handTorchCollected). Before
		// that it's left exactly where localInit captured it, lying on the floor
		// at its authored scene.json pose, for the player to walk up to.
		// With the "Holding Torch" cheat off it's parked below the map instead,
		// which is how everything else here hides a mesh (see consumeKey():
		// Starter draws every instance every frame, there's no per-instance
		// visibility flag). Its flame and light are dropped separately, through
		// flameBurning().
		bool torchInHand = handTorchCollected && cheats.handTorchEnabled;
		if(handTorchInst != nullptr && !torchInHand) {
			// On the floor if it hasn't been picked up and the torch isn't
			// cheat-disabled; parked below the map otherwise (collected but
			// cheat off).
			handTorchInst->Wm = (!handTorchCollected && cheats.handTorchEnabled)
				? handTorchSpawnWm
				: glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1000.0f, 0.0f));
		} else if(handTorchInst != nullptr) {
			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			// Wall tuck first, walk bob on top: the tuck decides the pose, the
			// bob swings around whatever pose that is. Folding it into a copy of
			// the constants (rather than into the finished matrix) is what lets
			// the two compose without either one knowing about the other.
			//
			// This is also the whole fix for the held torch's light: the flame
			// anchor rides on this very Wm, so pulling the model out of the wall
			// pulls the point light and its shadow cube out with it. Nothing
			// downstream in updateUniformBuffer needs to change.
			glm::vec3 tuckedOffset = HAND_TORCH_OFFSET;
			glm::vec3 tuckedTilt = HAND_TORCH_TILT_DEG;
			applyTuck(handTorchReach, tuckedOffset, tuckedTilt);

			// Was an inline copy of handGrip's three rotations; it is the same
			// expression, and sharing it keeps the two hands tilting alike.
			glm::mat4 grip = handGrip(tuckedTilt, bobRollDeg);
			glm::vec3 bobbedOffset = tuckedOffset + glm::vec3(bobLateral, bobVertical, 0.0f);

			handTorchInst->Wm = camWm
				* glm::translate(glm::mat4(1.0f), bobbedOffset)
				* grip
				* glm::scale(glm::mat4(1.0f), glm::vec3(HAND_TORCH_SCALE));
		}

		// Held key: same camera-anchored placement as the torch, live only
		// once picked up. Reuses the world instance (parked below the map on
		// pickup, see the interaction block above) instead of a second one --
		// pointing its Wm at the camera every frame from here on is the
		// entire "now it's in your hand" effect.
		//
		// Lateral bob, roll and vertical bob all read off the SAME phase with
		// the SAME sign as the torch's: both hands swing left together and
		// right together (not toward/away from center), which is what the
		// held-key sway is meant to match here.
		//
		// Only the newest key on the ring is drawn: there's one free hand,
		// and a key spent on a lock leaves the ring (and is parked below the
		// map by consumeKey) so it stops being drawn on the very next frame.
		if(heldKeyIdx() >= 0) {
			Pickup &held = pickups[heldKeyIdx()];
			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			// Tilt and offset off the item itself, not off one pair of
			// constants: the ring can hold a key and a book, and they are two
			// different shapes carrying their origins in two different places.
			// Same wall tuck the torch gets, on the item's OWN offset and tilt
			// rather than on one pair of constants -- for the same reason the
			// grip already reads them off the pickup: the ring can hold a key or
			// a book, and they are different shapes in different hands' worth of
			// space. applyTuck reads the offset's sign to work out which way
			// "inward" is, so the left hand tucks toward the centre unaided.
			glm::vec3 tuckedOffset = held.handOffset;
			glm::vec3 tuckedTilt = held.handTiltDeg;
			applyTuck(handKeyReach, tuckedOffset, tuckedTilt);

			glm::mat4 grip = handGrip(tuckedTilt, bobRollDeg);

			// Pick-up rise: only the Y offset moves, so the grip and the bob
			// above are untouched and the key simply slides up into the pose
			// it would otherwise have snapped to. Cubic ease-out (fast off the
			// floor, settling at the top) rather than linear, which stops dead
			// and reads mechanical. Clamped, so once the rise is over this is
			// exactly 0 and the pose is the plain held one.
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

		// Key spent on a lock: the exact mirror of the rise above, played
		// downward. Same pose, same duration, same drop distance, only the
		// easing is reversed (cubic ease-IN: it starts from the held pose and
		// accelerates away) so the two read as one motion run backwards.
		// Drawn from here and not from the held-key block because the key is
		// already off the ring -- the lock took it the moment E was pressed,
		// and only the model is still catching up. When the fall ends the
		// instance goes below the map, which is where consumeKey used to put
		// it immediately.
		if(keyLowerIdx >= 0) {
			keyLowerElapsed = std::min(keyLowerElapsed + deltaT, KEY_RAISE_DURATION);
			float t = keyLowerElapsed / KEY_RAISE_DURATION;
			float eased = t * t * t;
			float lowerY = -KEY_RAISE_DROP * eased;

			float bobLateral = sinf(walkBobPhase) * WALK_BOB_LATERAL * walkBobBlend;
			float bobVertical = sinf(walkBobPhase * 2.0f) * WALK_BOB_VERTICAL * walkBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			Pickup &sinking = pickups[keyLowerIdx];

			// Tucked too, off the same handKeyReach the held key uses (the block
			// above feeds it from whichever key is in frame). Without this the
			// key would snap back out to the extended pose on the single frame
			// the lock takes it -- straight through the door it was just used
			// on, which is by definition the wall you are standing against.
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