// THIS IS THE FILE YOU MUST START FROM!

// This has been adapted from the Vulkan tutorial
#include <sstream>
#include <limits>

#include <json.hpp>

#include "modules/Starter.hpp"
#include "modules/TextMaker.hpp"
#include "modules/Scene.hpp"
#include "custom/UiQuad.hpp"
#include "custom/CheatHud.hpp"
#include "custom/SceneColliders.hpp"
#include "custom/SceneMaterials.hpp"
#include "custom/SceneLights.hpp"

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
	// Must match the GLSL block field for field; the floats after the vec3 fill
	// std140's padding, so no explicit padding of ours is needed.
	alignas(16) glm::vec3 mS;	// specular color
	float roughness;			// rho: width of the microfacet distribution
	float F0;					// reflectance seen head-on
	float k;					// diffuse share of the BRDF
	int flatNormals;			// 1: derive the face normal in the shader
	// Seconds since startup, the same value for every instance in a frame.
	// Piggybacks the per-object UBO instead of going in
	// GlobalUniformBufferObject, which would shift LightData[] off the offset
	// the comment there notes. Read only by the Flame shaders, which animate
	// the torch flames entirely on the GPU; the scene shaders declare it and
	// ignore it, since both pipelines share DSLlocal and so this one struct.
	float time;
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
	LightData lights[MAX_LIGHTS];
};

// Set 2: the shadow-sampling data, bound once and read by CookTorrance.frag.
// One matrix per shadow-casting light (NUM_SHADOW_LIGHTS, LightConstants.glsl
// -- the sun plus the six torches), each the SAME view-projection its own
// shadow pass rendered with (see computeShadowMatrices()). Static for the
// life of the program, since none of those lights move, but still re-mapped
// every frame in updateUniformBuffer() rather than once at startup: map()
// writes into a per-swapchain-image buffer slot, and mapping only slot 0
// would leave the others holding whatever was there at allocation time.
struct ShadowUniformBufferObject {
	alignas(16) glm::mat4 lightSpace[NUM_SHADOW_LIGHTS];
};

// Vertex format "VDposNormUV". Starter.hpp fills the normal from the glTF/MGCG
// file automatically once the layout declares one.
struct Vertex {
	glm::vec3 pos;
	glm::vec3 norm;
	glm::vec2 UV;
};

// MAIN !

class Skeleton26ReplaceName : public BaseProject {
	protected:
	// Here you list all the Vulkan objects you need:
	
	// Descriptor Layouts [what will be passed to the shaders]
	DescriptorSetLayout DSLlocal, DSLglobal;

	// Vertex formants, Pipelines [Shader couples] and Render passes
	VertexDescriptor VD;
	RenderPass RP;
	Pipeline P;
	// Second pipeline, for the torch flames only. It exists because a flame
	// needs two pipeline-level settings P cannot have without breaking every
	// other object: alpha blending (a flame is a translucent volume, and an
	// opaque one reads as orange plastic however it is shaded) and no
	// back-face culling (it is seen from every side). Same vertex format and
	// same descriptor set layouts as P -- only the shaders and these two
	// switches differ.
	Pipeline PFlame;

	// Shadow mapping: one depth-only render pass per shadow-casting light
	// (NUM_SHADOW_LIGHTS = the sun plus the six torches, LightConstants.glsl)
	// and ONE pipeline shared across all of them. Reusing PShadow instead of
	// one pipeline per pass relies on Vulkan's render-pass-compatibility
	// rule: RPShadow[i] all use the identical AT_DEPTH_ONLY attachment
	// configuration, so a pipeline created against one of them works with any
	// of the others. Unlike RP/P, both are created once in localInit() and
	// never touched by a resize: an offscreen depth target doesn't depend on
	// the window, so there's no reason to tear it down and rebuild it the way
	// the swapchain-sized resources are.
	//
	// Only the CookTorrance technique is drawn into these (see
	// populateCommandBuffer()) -- the flames aren't occluders and shouldn't
	// occlude either, being translucent, so they're skipped rather than given
	// their own shadow logic.
	RenderPass RPShadow[NUM_SHADOW_LIGHTS];
	Pipeline PShadow;
	// set 2 for the main pass's shadow sampling: one UBO (the light-space
	// matrices) plus one sampler binding per shadow map, read by
	// CookTorrance.frag's shadowFactor(). DSLlocal/DSLglobal stay set 1/0.
	//
	// No DescriptorSet member of its own: unlike DSglobal, this one goes
	// through Scene's ordinary per-instance machinery instead (P is given
	// this as a third layout below, so every CookTorrance instance gets its
	// own copy, same as its DSLlocal one). That means NUM_SHADOW_LIGHTS+1
	// redundant, identical descriptor sets per instance -- wasteful, but
	// cheap at this instance count, and it avoids hand-rolling a THIRD way to
	// bind a descriptor set alongside Scene's existing one.
	DescriptorSetLayout DSLshadowSample;
	// View-projection matrix each shadow pass rendered with, index-matched to
	// LightData::shadowIndex. Computed once in computeShadowMatrices() (the
	// sun and the torches are static) and reused both as the push constant
	// Shadow.vert takes and as the UBO CookTorrance.frag samples against.
	glm::mat4 shadowLightSpace[NUM_SHADOW_LIGHTS];
	static constexpr int SHADOW_MAP_RES = 1024;

	// Models, textures and Descriptors (values assigned to the uniforms)
	DescriptorSet DSglobal;

	// To support loading assets from a scene.json file
	Scene SC;
	std::vector<VertexDescriptorRef>  VDRs;
	std::vector<TechniqueRef> PRs;

	// to provide textual feedback
	TextMaker txt;

	// Flat-colored quads: background/highlight panel behind the cheat HUD's text.
	UiQuad uiQuad;

	// Toggle-based pause menu for the cheats below, opened/closed with L.
	CheatHud hud;

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
		// Global gravity
		bool gravityEnabled = true;
		// True: collisions (ground included), False: no-clip cheat
		bool collisionEnabled = true;
		// Jump flag
		bool jumpEnabled = true;
		// Sprint flag
		bool sprintEnabled = true;
		// Debug overlay: continuously prints the camera's world-space
		// position/yaw in the bottom-right corner, meant as a live readout
		// for hand-placing scene.json objects (see notes.md). Unlike the
		// other flags, true isn't "the legit/no-cheat default": there's no
		// gameplay behavior to preserve here, so it defaults to off (hidden)
		// instead.
		bool showCoordinates = false;

		// Lighting debug views, all resolved into gubo.debugFlags in
		// updateUniformBuffer() and read by CookTorrance.frag. Same convention
		// as showCoordinates: these have no "legit" state to preserve, so each
		// one defaults to whatever leaves the picture as authored.
		//
		// The switches for the light SOURCES (sun, lanterns, spot, ambient,
		// sun orbit) aren't here: they live in SceneLights, next to the lights
		// they drop, and the HUD points straight at them.

		// Albedo only, nothing lit. Separates "this texture is dark" from
		// "no light is reaching this".
		bool unlit = false;
		// The shading normal as a color. The one view that shows normals
		// directly, which is what the flatNormals material flag exists for.
		bool showNormals = false;
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
		float openAngleDeg = 100.0f;	// target angle when open; sign picks swing direction
		bool open = false;
		float angle = 0.0f;	// current animated angle, eases toward the target
	};
	std::vector<Door> doors;

	// How close (world units, measured to the doorway centre) the player has
	// to be before a door's prompt appears and E does anything.
	static constexpr float DOOR_INTERACT_RADIUS = 3.5f;
	// Degrees/second the door animates open/closed at.
	static constexpr float DOOR_OPEN_SPEED = 120.0f;

	// Edge-detection for the interact key, same reason as jumpKeyWasPressed:
	// holding E shouldn't toggle the door every frame.
	bool interactKeyWasPressed = false;
	// Index into `doors` of whichever one is currently in range, or -1. Set
	// each frame in GameLogic(), read by updateUniformBuffer() to show/hide
	// the "[E] Interact" prompt.
	int nearbyDoor = -1;

	// A skull sitting in a torch's flame that yaws in place to face the
	// player, updated every frame in GameLogic() -- unlike the door's angle,
	// there's no authored target to ease toward, it just always points at
	// camPos. worldPos is captured once at init (these instances carry no
	// rotation/scale in scene.json, just translate), so Wm each frame is
	// nothing but that translation with a fresh yaw appended.
	struct WatchingSkull {
		Instance *inst = nullptr;
		glm::vec3 worldPos{0.0f};
	};
	std::vector<WatchingSkull> watchingSkulls;

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
		windowTitle = "Skeleton: place the name of your app here";
    	windowResizable = GLFW_TRUE;
		
		// Initial aspect ratio
		Ar = 4.0f / 3.0f;
	}
	
	// What to do when the window changes size
	void onWindowResize(int w, int h) {
		std::cout << "Window resized to: " << w << " x " << h << "\n";
		Ar = (float)w / (float)h;
		// Update Render Pass
		RP.width = w;
		RP.height = h;

		// windowWidth/windowHeight are otherwise only set once in
		// setWindowParameters() and never refreshed here; the cheat HUD
		// needs the current size for its pixel-based layout math.
		windowWidth = (uint32_t)w;
		windowHeight = (uint32_t)h;

		// updates the textual output
		txt.resizeScreen(w, h);
		uiQuad.resizeScreen(w, h);
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
		// for DSLlocal's single texture, just NUM_SHADOW_LIGHTS-wide here.
		//
		// Built in a loop rather than written out: at NUM_SHADOW_LIGHTS = 13
		// the literal list was getting long enough to hide a typo, and this
		// way the count lives in exactly one place. Binding 0 is the UBO, so
		// map i sits at binding i+1 -- the same numbering CookTorrance.frag
		// declares its shadowMap0..12 with, which nothing but agreement here
		// keeps true.
		std::vector<DescriptorSetLayoutBinding> shadowSampleBindings = {
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(ShadowUniformBufferObject), 1}
				  };
		for(int i = 0; i < NUM_SHADOW_LIGHTS; i++) {
			shadowSampleBindings.push_back({(uint32_t)(i + 1),
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

		// initializes the render passes
		RP.init(this);
		// sets the blue sky
		RP.properties[0].clearValue = {0.0f,0.9f,1.0f,1.0f};

		// The shadow render passes -- the sun's and each torch's, in
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
		for(int i = 0; i < NUM_SHADOW_LIGHTS; i++) {
			RPShadow[i].init(this, SHADOW_MAP_RES, SHADOW_MAP_RES, -1,
							  RenderPass::getStandardAttchmentsProperties(AT_DEPTH_ONLY, this),
							  RenderPass::getStandardDependencies(ATDEP_DEPTH_TRANS),
							  true);
			RPShadow[i].create();
		}

		// Pipelines [Shader couples]
		// The last array, is a vector of pointer to the layouts of the sets that will
		// be used in this pipeline. The first element will be set 0, and so on..

		P.init(this, &VD, "shaders/PosNormUV.vert.spv",
						  "shaders/CookTorrance.frag.spv",
						  {&DSLglobal, &DSLlocal, &DSLshadowSample});

		// The flame pipeline. setTransparency turns on the SRC_ALPHA /
		// ONE_MINUS_SRC_ALPHA blend Starter.hpp wires up, which is what makes
		// the alpha Flame.frag writes mean anything; cull NONE keeps the far
		// side of the flame from vanishing when you walk round the torch.
		// Depth WRITE stays on (Starter.hpp hard-codes it, and it is off
		// limits), so the flames are drawn last -- see scene.json, where their
		// technique block is the final one -- and nothing is drawn afterwards
		// that they could wrongly occlude.
		PFlame.init(this, &VD, "shaders/Flame.vert.spv",
							   "shaders/Flame.frag.spv",
							   {&DSLglobal, &DSLlocal});
		PFlame.setTransparency(true);
		PFlame.setCullMode(VK_CULL_MODE_NONE);

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
		// Created against RPShadow[0], but usable with all of them: they share the
		// identical AT_DEPTH_ONLY attachment layout, and Vulkan only requires
		// render-pass COMPATIBILITY (same attachment formats/samples/layouts)
		// between the render pass a pipeline was created with and the one
		// it's bound under at draw time, not the exact same object.
		PShadow.create(&RPShadow[0]);

		// sets the size of the Descriptor Set Pool (it MUST be done before loading the scene)
		DPSZs.uniformBlocksInPool = 2;
		DPSZs.texturesInPool = 1;
		DPSZs.setsInPool = 2;

		// to support scene
		VDRs.resize(1);
		VDRs[0].init("VDposNormUV",  &VD);

		// DSLshadowSample: none of these are "fromInstance" -- the shadow maps
		// are the same fixed images for every instance, not per-instance
		// textures like DSLlocal's albedo map. pos is unused on a
		// non-fromInstance entry. In a loop for the same reason the layout
		// above is.
		std::vector<TextureDefs> shadowMapDefs;
		for(int i = 0; i < NUM_SHADOW_LIGHTS; i++) {
			shadowMapDefs.push_back({false, 0, RPShadow[i].attachments[0].getViewAndSampler()});
		}

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
		// Same shape as above: the flames still declare one texture, because
		// DSLlocal has a sampler binding either way, even though Flame.frag
		// never samples it (its colour is generated, not painted).
		PRs[1].init("Flame", {
							{&PFlame, {
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
		// openAngleDeg's sign picks which way it swings open; chosen without
		// being able to see the render from here, so if it swings the wrong
		// way, negate it.
		auto addDoor = [&](const char *id, glm::vec3 promptOffset, float openAngleDeg) {
			auto it = SC.InstanceIds.find(id);
			if(it == SC.InstanceIds.end()) {
				std::cout << "Door instance '" << id << "' not found, skipping\n";
				return;
			}
			Door d;
			d.instanceId = id;
			d.inst = SC.I[it->second];
			d.baseWm = d.inst->Wm;
			d.promptPos = glm::vec3(d.baseWm * glm::vec4(promptOffset, 1.0f));
			d.openAngleDeg = openAngleDeg;
			doors.push_back(d);
		};
		addDoor("dhDoorPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f);
		// Second and third doors, gating the two new rooms (dl, dv) added east
		// of the antechamber. The full dc/dl boundary is two tiles wide, so it
		// took two hole-wall + leaf pairs, not one wall tile left solid next
		// to it -- a plain wall there would have blocked half the doorway
		// with no way through. Same leaf asset, same hinge geometry as the
		// first door, so the same promptOffset/openAngleDeg apply unchanged.
		addDoor("dlDoorPanel", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f);
		addDoor("dlDoorPanel2", glm::vec3(0.0f, 2.52f, -1.231f), 100.0f);

		// The five watching skulls, one per torch (see scene.json "torchSkull"
		// instances). One addWatchingSkull() call per skull, same reasoning as
		// addDoor() above: adding another is one line, not new plumbing.
		auto addWatchingSkull = [&](const char *id) {
			auto it = SC.InstanceIds.find(id);
			if(it == SC.InstanceIds.end()) {
				std::cout << "Watching skull instance '" << id << "' not found, skipping\n";
				return;
			}
			WatchingSkull s;
			s.inst = SC.I[it->second];
			s.worldPos = glm::vec3(s.inst->Wm[3]);
			watchingSkulls.push_back(s);
		};
		addWatchingSkull("dhSkullTorchW1");
		addWatchingSkull("dhSkullTorchW2");
		addWatchingSkull("dhSkullTorchE1");
		addWatchingSkull("dhSkullTorchE2");
		addWatchingSkull("dcSkullTorchE");

		// Surface parameters for the BRDF, one per model.
		materials.init(&SC, "assets/scenes/materials.json");

		// After Scene::init: a light can be anchored to an instance and needs
		// that instance's world matrix.
		sceneLights.init(&SC, "assets/scenes/lights.json");

		// After sceneLights.init(): needs the resolved world position of every
		// shadow-casting light, which instance+offset lights only have once
		// SceneLights has read scene.json's world matrices.
		computeShadowMatrices();

		// initializes the textual output
		txt.init(this, windowWidth, windowHeight);
		// initializes the flat-quad background/highlight layer for the cheat HUD
		uiQuad.init(this, windowWidth, windowHeight);

		// submits the main command buffer
		submitCommandBuffer("main", 0, populateCommandBufferAccess, this);

		// Prepares for showing the FPS count
		txt.print(1.0f, 1.0f, "FPS:",1,"CO",false,false,true,TAL_RIGHT,TRH_RIGHT,TRV_BOTTOM,{1.0f,0.0f,0.0f,1.0f},{0.8f,0.8f,0.0f,1.0f});

		// Wires the cheat HUD to the actual cheat flags, so toggling a row
		// in the menu flips the exact same bools GameLogic() reads.
		hud.init(&txt, &uiQuad);
		hud.addToggle("Gravity", &cheats.gravityEnabled);
		hud.addToggle("Collision", &cheats.collisionEnabled);
		hud.addToggle("Jump", &cheats.jumpEnabled);
		hud.addToggle("Sprint", &cheats.sprintEnabled);
		hud.addToggle("Show Coordinates", &cheats.showCoordinates);

		// Lighting rows. Listed after the movement ones and in the order you'd
		// use them: first which sources are on, then how they're being shaded.
		// The first five point straight into sceneLights, which owns them (see
		// SceneLights.hpp); the rest into cheats, which become gubo.debugFlags.
		hud.addToggle("Sun", &sceneLights.directEnabled);
		hud.addToggle("Lanterns", &sceneLights.pointEnabled);
		hud.addToggle("Spotlight", &sceneLights.spotEnabled);
		hud.addToggle("Ambient Light", &sceneLights.ambientEnabled);
		hud.addToggle("Sun Orbit", &sceneLights.orbitOverride);
		hud.addToggle("Shadows", &cheats.shadowsEnabled);
		hud.addToggle("Specular", &cheats.specularEnabled);
		hud.addToggle("Tone Mapping", &cheats.toneMapEnabled);
		hud.addToggle("Fullbright", &cheats.unlit);
		hud.addToggle("Show Normals", &cheats.showNormals);
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
		// Torch aim, hand-picked per wall side rather than read from
		// anywhere: point lights carry no direction (SceneLights.hpp), so
		// nothing already knows which way a torch should look. Aimed
		// horizontally INTO the room the torch is mounted on (matching the
		// +-X sign already used for the flame/light offsets in scene.json
		// and lights.json) with a slight downward tilt, so the shadow map
		// actually covers the floor and the opposite wall -- the case that
		// motivated this feature (see notes.md) was light bleeding through
		// exactly that wall.
		//
		// Index-matched to the SIX torches in the order they appear in
		// lights.json (torchW1, torchW2, torchE1, torchE2, torchDC, torchDV),
		// i.e. shadowIndex 1..6 once the sun takes 0.
		//
		// Each torch gets TWO maps, this direction and its exact opposite (see
		// SHADOW_MAPS_PER_LIGHT and shadowFactor() in CookTorrance.frag), so
		// the pair between them covers the whole sphere bar a thin band square
		// with the aim. Which of the two halves is which barely matters now
		// that both are rendered -- what the aim still buys is where the band
		// of uncovered directions falls, so pointing it INTO the room (the
		// +-X sign already used for the flame/light offsets in scene.json and
		// lights.json) puts that band flat along the mounting wall, where the
		// geometry it could leak through is furthest away.
		//
		// DEAD horizontal, no downward tilt: the band should lie in the plane
		// of the wall, and any tilt rotates it to slice diagonally through the
		// room instead.
		static const glm::vec3 TORCH_SHADOW_DIR[6] = {
			glm::vec3( 1.0f, 0.0f, 0.0f),	// torchW1: west wall, aims +X into the room
			glm::vec3( 1.0f, 0.0f, 0.0f),	// torchW2
			glm::vec3(-1.0f, 0.0f, 0.0f),	// torchE1: east wall, aims -X into the room
			glm::vec3(-1.0f, 0.0f, 0.0f),	// torchE2
			glm::vec3(-1.0f, 0.0f, 0.0f),	// torchDC
			glm::vec3(-1.0f, 0.0f, 0.0f),	// torchDV: east wall of the dv alcove
		};

		// Per torch map. Two back-to-back frusta this wide overlap everywhere
		// except a band of about +-20 degrees around the plane square with the
		// aim, and that band is the only place a torch can still light through
		// geometry. Wider would close it further, but a perspective projection
		// degenerates approaching 180 degrees: the same 1024 texels spread over
		// more angle and stretch brutally towards the edges, which is depth
		// precision the bias in CookTorrance.frag has to absorb. 140 is the
		// compromise; the leftover band lies flat along the wall, where in this
		// scene there is nothing close enough to leak into.
		const float TORCH_SHADOW_FOV = 140.0f;

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

		int torchSlot = 0;
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
				shadowLightSpace[L.shadowIndex] = proj * view;
				continue;
			}

			// A point light's two maps: the aim, then its opposite. Same
			// order SceneLights::init reserved the slots in and the same
			// order shadowFactor() tries them.
			//
			// The aim table is hand-written per torch and lights.json could
			// outgrow it (its entries are the only thing here that is not
			// derived from the light itself), so a torch past the end takes
			// the last direction rather than reading off the array. Wrong-
			// looking shadows on one torch beat undefined behaviour.
			const int aimIndex = std::min(torchSlot++,
										  (int)(sizeof(TORCH_SHADOW_DIR) / sizeof(TORCH_SHADOW_DIR[0])) - 1);
			const glm::vec3 aim = TORCH_SHADOW_DIR[aimIndex];
			for(int half = 0; half < SHADOW_MAPS_PER_LIGHT; half++) {
				const glm::vec3 dir = (half == 0) ? aim : -aim;

				// lookAt degenerates if the aim is parallel to `up`, so pick
				// an up that cannot be: world up unless the torch looks
				// straight up or down, in which case any horizontal axis does.
				// Every TORCH_SHADOW_DIR entry is horizontal today, so this
				// only guards a future edit.
				const glm::vec3 up = (std::abs(dir.y) > 0.99f)
									 ? glm::vec3(0.0f, 0.0f, 1.0f)
									 : glm::vec3(0.0f, 1.0f, 0.0f);

				glm::mat4 view = glm::lookAt(L.pos, L.pos + dir, up);
				glm::mat4 proj = glm::perspective(glm::radians(TORCH_SHADOW_FOV),
												  1.0f, 0.1f, 15.0f);
				proj[1][1] *= -1;

				shadowLightSpace[L.shadowIndex + half] = proj * view;
			}
		}
	}

	// Here you create your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsInit() {
		// creates the render passes
		RP.create();
		
		// This creates a new pipeline (with the current surface), using its shaders for the provided render pass
		P.create(&RP);
		PFlame.create(&RP);

		DSglobal.init(this, &DSLglobal, {});
		
		// Here you define the data set
		// If the scene has textures coming from a render pass, the corresponding element of the technique must be
		// updated before calling SC.pipelinesAndDescriptorSetsInit();

		SC.pipelinesAndDescriptorSetsInit();
		txt.pipelinesAndDescriptorSetsInit();
		uiQuad.pipelinesAndDescriptorSetsInit();
	}

	// Here you destroy your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsCleanup() {
		P.cleanup();
		PFlame.cleanup();

		RP.cleanup();
		
		DSglobal.cleanup();
		
		SC.pipelinesAndDescriptorSetsCleanup();
		txt.pipelinesAndDescriptorSetsCleanup();
		uiQuad.pipelinesAndDescriptorSetsCleanup();
	}

	// Here you destroy all the Models, Texture and Desc. Set Layouts you created!
	// You also have to destroy the pipelines
	void localCleanup() {
		DSLlocal.cleanup();
		DSLglobal.cleanup();
		DSLshadowSample.cleanup();

		P.destroy();
		PFlame.destroy();

		// PShadow/RPShadow never go through pipelinesAndDescriptorSetsCleanup
		// (see the member declaration for why -- they don't depend on the
		// swapchain, so a resize never tears them down), which is where P/RP
		// normally get their .cleanup() half. Both halves have to happen
		// somewhere, so both happen here instead.
		PShadow.cleanup();
		PShadow.destroy();
		for(int i = 0; i < NUM_SHADOW_LIGHTS; i++) {
			RPShadow[i].cleanup();
			RPShadow[i].destroy();
		}

		RP.destroy();

		// Before SC.localCleanup(): that frees the colliders scene.json created, which
		// colliderSet also points at. It only deletes the ones it allocated itself, but
		// dropping the shared list first keeps the two ownership halves from overlapping.
		colliderSet.cleanup();
		allColliders.clear();

		SC.localCleanup();
		txt.localCleanup();
		uiQuad.localCleanup();
	}
	
	// Here it is the creation of the command buffer:
	// You send to the GPU all the objects you want to draw,
	// with their buffers and textures
	static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void *Params) {
		// Simple trick to avoid having always 'T->'
		// in che code that populates the command buffer!
		Skeleton26ReplaceName *T = (Skeleton26ReplaceName *)Params;
		T->populateCommandBuffer(commandBuffer, currentImage);
	}

	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {

		// The shadow passes, one per shadow-casting light, all before the
		// main pass they feed: CookTorrance.frag samples these maps, so they
		// have to be fully rendered (and, thanks to RPShadow's
		// ATDEP_DEPTH_TRANS dependency, transitioned to a readable layout)
		// before that draw happens. Not Scene::populateCommandBuffer -- that
		// walks every technique including Flame, and the flames are
		// deliberately not occluders here (see the RPShadow member comment) --
		// so this is its own small loop straight over the CookTorrance
		// instances (technique 0 in scene.json).
		for(int i = 0; i < NUM_SHADOW_LIGHTS; i++) {
			RPShadow[i].begin(commandBuffer, currentImage);
			PShadow.bind(commandBuffer);
			vkCmdPushConstants(commandBuffer, PShadow.pipelineLayout,
							   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
							   &shadowLightSpace[i]);
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
			RPShadow[i].end(commandBuffer);
		}

		// Offscreen pass - always required
		// begin standard pass
		RP.begin(commandBuffer, currentImage);

		SC.populateCommandBuffer(commandBuffer, 0, currentImage);

		RP.end(commandBuffer);
	}

	// Here is where you update the uniforms.
	// Very likely this will be where you will be writing the logic of your application.
	void updateUniformBuffer(uint32_t currentImage) {
		static bool debounce = false;
		static int curDebounce = 0;

		// handle the ESC key to exit the app
		if(glfwGetKey(window, GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}

		// moves the view
		float deltaT = GameLogic();

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

		// By value: with the Ambient Light cheat off there is no stored ambient
		// to hand back a reference to. See SceneLights::ambient().
		const AmbientLight amb = sceneLights.ambient();
		gubo.ambientUpper = amb.upper;
		gubo.ambientLower = amb.lower;
		gubo.ambientDir = amb.dir;

		// The lighting debug cheats, packed into the one int the shader reads.
		// Note the two inversions: the cheat says what the frame should still
		// have, the flag says what the shader should drop.
		gubo.debugFlags = 0;
		if(cheats.unlit)            gubo.debugFlags |= LIGHT_DEBUG_UNLIT;
		if(cheats.showNormals)      gubo.debugFlags |= LIGHT_DEBUG_NORMALS;
		if(!cheats.specularEnabled) gubo.debugFlags |= LIGHT_DEBUG_NO_SPECULAR;
		if(!cheats.toneMapEnabled)  gubo.debugFlags |= LIGHT_DEBUG_NO_TONEMAP;
		if(!cheats.shadowsEnabled)  gubo.debugFlags |= LIGHT_DEBUG_NO_SHADOWS;

		gubo.eyePos = glm::vec3(glm::inverse(View)[3]);

		DSglobal.map(currentImage, &gubo, 0);

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
		for(int i = 0; i < NUM_SHADOW_LIGHTS; i++) {
			shadowUbo.lightSpace[i] = shadowLightSpace[i];
		}

		// Over every technique, not just the first: the flames are their own
		// technique (see PFlame / scene.json) and their instances need the
		// same per-object uniforms filled in as any other. Both techniques
		// share DSLlocal, so one struct serves both -- the Flame shaders just
		// read a different subset of it.
		for(int techniqueId = 0; techniqueId < SC.TechniqueInstanceCount; techniqueId++) {
			for(int instanceId = 0; instanceId < SC.TI[techniqueId].InstanceCount; instanceId++) {
				ubo.mMat = SC.TI[techniqueId].I[instanceId].Wm;
				ubo.mvpMat = ViewPrj * ubo.mMat;
				ubo.nMat = glm::inverse(glm::transpose(ubo.mMat));

				// By Mid rather than by name, so no string hashing per frame.
				const Material &m = materials.forModel(SC.TI[techniqueId].I[instanceId].Mid);
				ubo.mS = m.specularColor;
				ubo.roughness = m.roughness;
				ubo.F0 = m.F0;
				ubo.k = m.k;
				ubo.flatNormals = m.flatNormals;
				ubo.time = simTime;

				Instance &inst = SC.TI[techniqueId].I[instanceId];
				// DS[1] = Pchar pass (main render): set0=DSLglobal, set1=DSLlocal
				inst.DS[0][0]->map(currentImage, &gubo, 0); // global (light/camera)
				inst.DS[0][1]->map(currentImage, &ubo, 0); // camera MVPs
				// set2=DSLshadowSample, only on the CookTorrance technique
				// (PFlame's pipeline layout has no third set -- Flame.frag
				// doesn't read shadows, see its header). NDs[0] is 3 there
				// and 2 for Flame instances, which is what this checks
				// instead of comparing techniqueId to a hardcoded index.
				if(inst.NDs[0] >= 3) {
					inst.DS[0][2]->map(currentImage, &shadowUbo, 0);
				}
			}
		}
		
		// updates the FPS
		static float elapsedT = 0.0f;
		static int countedFrames = 0;
		
		countedFrames++;
		elapsedT += deltaT;
		if(elapsedT > 1.0f) {
			float Fps = (float)countedFrames / elapsedT;
			
			std::ostringstream oss;
			oss << "FPS: " << Fps << "\n";

			txt.print(1.0f, 1.0f, oss.str(), 1, "CO", false, false, true,TAL_RIGHT,TRH_RIGHT,TRV_BOTTOM,{1.0f,0.0f,0.0f,1.0f},{0.8f,0.8f,0.0f,1.0f});
			
			elapsedT = 0.0f;
		    countedFrames = 0;
		}

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

		// "[E] Interact" prompt, shown only while a door is in range
		// (nearbyDoor, set every frame in GameLogic()). A plain shown/hidden
		// toggle needs no throttling like the coordinates overlay does: it's
		// binary, so it only touches the text buffer on the frames the state
		// actually flips.
		static bool interactPromptShown = false;
		bool showInteractPrompt = (nearbyDoor >= 0);
		if(showInteractPrompt && !interactPromptShown) {
			float sx, sy;
			txt.pixelToScr((float)windowWidth / 2.0f, (float)windowHeight - 60.0f, sx, sy);
			txt.print(sx, sy, "[E] Interact", 3, "CO", false, true, false,
					  TAL_CENTER, TRH_CENTER, TRV_BOTTOM,
					  {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
			interactPromptShown = true;
		} else if(!showInteractPrompt && interactPromptShown) {
			txt.removeText(3);
			interactPromptShown = false;
		}

		txt.updateCommandBuffer();
		uiQuad.updateCommandBuffer();
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

		// Poll/render the cheat HUD BEFORE getSixAxis. getSixAxis turns on
		// GLFW_STICKY_MOUSE_BUTTONS, which makes glfwGetMouseButton a
		// one-shot read (it flips back to "released" once polled). Reading
		// the HUD's own click hit-test first guarantees the HUD gets that
		// one authoritative read of a click, not getSixAxis's drag-look check.
		hud.update(window, windowWidth, windowHeight);

		getSixAxis(deltaT, m, r, fire);

		if(hud.isOpen()) {
			// HUD is open: discard camera-look/move/fire input this frame so
			// a HUD click or drag can't also spin the camera underneath the
			// menu.
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

		// Freeze all movement/physics while the cheat HUD is open, so opening
		// it pauses the game exactly where it was (camera included, since m/r
		// were already zeroed above).
		if(!hud.isOpen()) {
			// Sprint: Ctrl multiplies movement speed, gated behind sprintEnabled like
			// the other cheats/debug toggles. Polled directly (not through getSixAxis/
			// "fire") since Starter.hpp doesn't wire Ctrl to anything.
			// Can only be started while grounded (no starting a sprint mid-jump), but
			// releasing Ctrl always stops it right away, air or not.
			bool ctrlHeld = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL);
			if(!cheats.sprintEnabled || !ctrlHeld) {
				sprinting = false;
			} else if(grounded) {
				sprinting = true;
			}
			float moveSpeed = movement.moveSpeed;
			if(sprinting) {
				moveSpeed *= movement.sprintMultiplier;
			}

			// Update position from WASD/R/F: m.x = strafe, m.z = -forward, m.y = world up/down
			// Forward/strafe movement is flattened to the horizontal plane (yaw only), not
			// the full pitch-tilted `front` used for looking around: otherwise looking up
			// and pressing W pushes you upward (feels like a jump), and looking up while
			// walking backward pushes you down through the floor. Vertical movement only
			// ever comes from R/F (m.y), jumping, and gravity.
			glm::vec3 frontFlat = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
			camPos += (right * m.x - frontFlat * m.z + worldUp * m.y) * moveSpeed * deltaT;

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
			// below). Gated behind jumpEnabled like the other cheats/debug toggles.
			// If gravity is off, the impulse gets reset straight back to 0 below, so
			// jumping naturally has no effect without gravity to bring us back down.
			if(cheats.jumpEnabled) {
				if(fire && !jumpKeyWasPressed && grounded) {
					camVerticalVelocity = movement.jumpSpeed;
					// Drop any leftover step smoothing: jumping right after
					// stepping up would otherwise start the jump from a view
					// still trailing below the real eye height.
					eyeStepOffset = 0.0f;
				}
			}
			jumpKeyWasPressed = fire;

			// Interaction: find the nearest door within range of its doorway
			// centre, toggle it open/closed on E (edge-triggered, same
			// pattern as jump), then ease every door's animated angle
			// toward its target and push the result into both the render
			// transform and its collider, so an open door is actually
			// walkable and a closed one still blocks.
			nearbyDoor = -1;
			float bestDoorDist = DOOR_INTERACT_RADIUS;
			for(int i = 0; i < (int)doors.size(); i++) {
				float dx = camPos.x - doors[i].promptPos.x;
				float dz = camPos.z - doors[i].promptPos.z;
				float dist = std::sqrt(dx * dx + dz * dz);
				if(dist < bestDoorDist) {
					bestDoorDist = dist;
					nearbyDoor = i;
				}
			}
			bool interactKey = glfwGetKey(window, GLFW_KEY_E);
			if(nearbyDoor >= 0 && interactKey && !interactKeyWasPressed) {
				doors[nearbyDoor].open = !doors[nearbyDoor].open;
			}
			interactKeyWasPressed = interactKey;

			for(Door &d : doors) {
				float target = d.open ? d.openAngleDeg : 0.0f;
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
			}

			// Watching skulls: yaw only (they stay upright), recomputed fresh
			// every frame from the current camera position -- there's no eased
			// "target" the way the doors have one, it's a straight look-at.
			// atan2(dx, dz) assumes the skull mesh's modeled front faces +Z; if
			// it turns out to face the camera backwards, add M_PI here.
			for(WatchingSkull &s : watchingSkulls) {
				float dx = camPos.x - s.worldPos.x;
				float dz = camPos.z - s.worldPos.z;
				float yaw = std::atan2(dx, dz);
				s.inst->Wm = glm::translate(glm::mat4(1.0f), s.worldPos)
							* glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
			}

			// Gravity: constant downward acceleration, integrated into a vertical
			// velocity each frame. Resolved against the ground below (collision
			// block right after this), which zeroes the velocity out on landing.
			if(cheats.gravityEnabled) {
				camVerticalVelocity += movement.gravity * deltaT;
				camPos.y += camVerticalVelocity * deltaT;
			} else {
				// Don't let velocity build up while gravity's off, so re-enabling
				// it later doesn't suddenly slam the camera down/up
				camVerticalVelocity = 0.0f;
			}

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

		return deltaT;
	}
};


// This is the main: probably you do not need to touch this!
int main() {
    Skeleton26ReplaceName app;

    try {
        app.run(false);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}