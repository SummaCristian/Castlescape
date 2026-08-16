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
#include "custom/Flame.hpp"

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
	LightData lights[MAX_LIGHTS];
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

	// Free-look camera state (position + orientation), persisted across frames
	glm::vec3 camPos = glm::vec3(0.0f, 1.8f, 5.0f);
	// Yaw: rotation around world up axis, in degrees.
	// yaw=0 faces +X; increasing yaw turns right, decreasing turns left.
	// Starts at -90 (faces -Z) to match the scene's original forward direction.
	float camYaw = -90.0f;
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

	// The torch held in the player's right hand. A normal scene instance
	// (handTorch in scene.json) whose world matrix is rebuilt every frame
	// from the camera's position and basis vectors, so it follows the view
	// like a first-person weapon model. Null if the instance isn't found.
	Instance *handTorchInst = nullptr;
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

	// The flame at a torch's head. See custom/Flame.hpp: a low-poly mesh,
	// entirely GPU-animated, drawn as one more object inside the main pass.
	// One Flame instance drives every torch in the scene (Flame::spawn),
	// held one included -- that's what the reusable design was for.
	Flame flame;

	// One entry per torch that got a flame, filled once in localInit() (see
	// addTorchFlame there) and walked every frame in updateUniformBuffer()
	// to update that flame's transform, its glow billboard and its point
	// light. `anchor` is in the TORCH MODEL's own local space (i.e. before
	// whatever instance transform places it in the world); `inst->Wm *
	// vec4(anchor,1)` gives the flame's world POSITION, but not its
	// orientation -- see FLAME_WORLD_SCALE below for why the flame doesn't
	// otherwise ride the instance's Wm the way the torch mesh itself does.
	struct TorchFlame {
		Instance *inst;
		int flameId;
		glm::vec3 anchor;
	};
	std::vector<TorchFlame> torchFlames;

	// One anchor for every torch: SM_Torch_01 (wall-mounted) and
	// SM_Torch_Held_01 (held) turn out to be the identical mesh, confirmed
	// by walking their POSITION accessors directly rather than trusting the
	// two models' reported min/max (which, misleadingly, differ). X -0.544..
	// -0.224 (mid -0.384), Z -0.165..0.165 (mid 0) at the centroid of the
	// mesh's own top (a wide flat cup, not a point, so a bbox corner isn't
	// the right anchor). Y is 0.38, BELOW that top (0.413): the cup has
	// depth, so sitting the flame's own base ring exactly at the rim left it
	// looking like it was floating just above the torch instead of coming
	// out of it -- nestling it down into the cup by that same margin reads
	// as rooted instead.
	static constexpr glm::vec3 TORCH_FLAME_ANCHOR = glm::vec3(-0.384f, 0.38f, 0.0f);

	// The flame's own local geometry (Flame.hpp's H[]/Rr[] arrays) was
	// authored by eye directly against this SAME model's proportions (it's
	// about 1.1 units tall unscaled), so riding each instance's own uniform
	// scale -- extracted below, since the render matrix otherwise carries
	// NO rotation (a flame stays vertical from its own buoyancy regardless
	// of how the torch holding it is tilted) and so carries no scale either
	// by default -- reproduces that proportion on every torch: full size on
	// the wall-mounted ones, shrunk to match on the held one (whose instance
	// is scaled down to arm's-length size, HAND_TORCH_SCALE). A single fixed
	// world-space size instead made the flame look right on the (small)
	// held torch and comically undersized on the (full-size) wall ones.

	// The torch flame's point light. One color/falloff for every torch in
	// the scene (held and wall-mounted alike): they're all the same kind of
	// fire, so there's nothing to author per-instance. Tighter (lower g)
	// than the gate lanterns (SceneLights.hpp/lights.json): a torch flame is
	// a much smaller, closer source than a lamp head.
	static constexpr glm::vec3 TORCH_LIGHT_COLOR = glm::vec3(1.0f, 0.5f, 0.16f);
	static constexpr float TORCH_LIGHT_G = 1.6f;
	static constexpr float TORCH_LIGHT_BETA = 1.4f;

	// The glow billboard (see Flame.hpp/FlameGlow.*): world-space half-size
	// of the quad, and how far above the flame's own anchor point its center
	// sits, so the haze is centered on the flame's visual mass (the crown)
	// rather than its base.
	static constexpr float TORCH_GLOW_RADIUS = 0.3f;
	static constexpr float TORCH_GLOW_HEIGHT_OFFSET = 0.12f;

	// Seconds since startup, uploaded as gubo.time and read by Flame.vert/
	// .frag and FlameGlow.frag. A free-running accumulator rather than a
	// frame-indexed value, so the sway/flicker never repeats on a
	// noticeable cycle.
	float animTime = 0.0f;

	// Walking sway: a lateral swing once per stride plus a vertical bounce
	// at twice that frequency (one bounce per footstep), both driven by a
	// single accumulating phase. torchBobBlend is the 0..1 sway amplitude,
	// eased toward 1 while walking and back to 0 at rest.
	float torchBobPhase = 0.0f;
	float torchBobBlend = 0.0f;
	// Radians/second the phase advances at normal walking speed, faster
	// while sprinting.
	static constexpr float TORCH_BOB_SPEED = 7.0f;
	// Sway amplitude in world units, before torchBobBlend scales it down.
	static constexpr float TORCH_BOB_VERTICAL = 0.035f;
	static constexpr float TORCH_BOB_LATERAL = 0.02f;
	// How fast torchBobBlend eases toward its target.
	static constexpr float TORCH_BOB_BLEND_TAU = 0.15f;

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

		// Pipelines [Shader couples]
		// The last array, is a vector of pointer to the layouts of the sets that will
		// be used in this pipeline. The first element will be set 0, and so on..
		
		P.init(this, &VD, "shaders/PosNormUV.vert.spv",
						  "shaders/CookTorrance.frag.spv",
						  {&DSLglobal, &DSLlocal});


		// sets the size of the Descriptor Set Pool (it MUST be done before loading the scene)
		DPSZs.uniformBlocksInPool = 2;
		DPSZs.texturesInPool = 1;
		DPSZs.setsInPool = 2;

		// to support scene
		VDRs.resize(1);
		VDRs[0].init("VDposNormUV",  &VD);

		PRs.resize(1);
		PRs[0].init("CookTorrance", {
							{&P, {//Pipeline and DSL for the main pass
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
		// hinges on. Measured off the current SM_WallDoor_Hole_01 geometry
		// (opening spans local Y 0.19..4.85, Z 2.47..4.71; the panel's own
		// origin sits at the hinge-side jamb, Y 0, Z ~4.82), not eyeballed --
		// re-measure and update this if the asset is regenerated again.
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

		// Held torch. Its Wm is overwritten every frame in GameLogic(), so
		// the placeholder transform in scene.json never actually shows.
		{
			auto it = SC.InstanceIds.find("handTorch");
			if(it == SC.InstanceIds.end()) {
				std::cout << "Hand torch instance 'handTorch' not found, skipping\n";
			} else {
				handTorchInst = SC.I[it->second];
			}
		}

		// Torch flames. DSglobal isn't populated yet (that happens in
		// pipelinesAndDescriptorSetsInit(), after the descriptor pool
		// exists), but its address is stable, so capturing a pointer to it
		// now and reading through it later is safe -- same reasoning as
		// handTorchInst above.
		flame.init(this, &DSLglobal, &DSglobal);

		// seed just spreads each flame's sway/flicker phase (see Flame.hpp),
		// not a real RNG: index * a large-ish irrational-ish constant keeps
		// them decorrelated without needing a seeded generator for one call.
		auto addTorchFlame = [&](const char *id, glm::vec3 anchor) {
			auto it = SC.InstanceIds.find(id);
			if(it == SC.InstanceIds.end()) {
				std::cout << "Torch instance '" << id << "' not found, skipping its flame\n";
				return;
			}
			Instance *inst = SC.I[it->second];
			int flameId = flame.spawn((float)torchFlames.size() * 2.3971f);
			if(flameId < 0) {
				return;
			}
			torchFlames.push_back({inst, flameId, anchor});
		};

		if(handTorchInst != nullptr) {
			addTorchFlame("handTorch", TORCH_FLAME_ANCHOR);
		}
		// The wall-mounted dungeonTorch instances (see scene.json): two pairs
		// flanking the hall's doorway plus one in the corridor.
		addTorchFlame("dhTorchW1", TORCH_FLAME_ANCHOR);
		addTorchFlame("dhTorchW2", TORCH_FLAME_ANCHOR);
		addTorchFlame("dhTorchE1", TORCH_FLAME_ANCHOR);
		addTorchFlame("dhTorchE2", TORCH_FLAME_ANCHOR);
		addTorchFlame("dcTorchE", TORCH_FLAME_ANCHOR);

		// Surface parameters for the BRDF, one per model.
		materials.init(&SC, "assets/scenes/materials.json");

		// After Scene::init: a light can be anchored to an instance and needs
		// that instance's world matrix.
		sceneLights.init(&SC, "assets/scenes/lights.json");

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
		hud.addToggle("Specular", &cheats.specularEnabled);
		hud.addToggle("Tone Mapping", &cheats.toneMapEnabled);
		hud.addToggle("Fullbright", &cheats.unlit);
		hud.addToggle("Show Normals", &cheats.showNormals);
	}
	
	// Here you create your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsInit() {
		// creates the render passes
		RP.create();
		
		// This creates a new pipeline (with the current surface), using its shaders for the provided render pass
		P.create(&RP);
		
		DSglobal.init(this, &DSLglobal, {});
		
		// Here you define the data set
		// If the scene has textures coming from a render pass, the corresponding element of the technique must be
		// updated before calling SC.pipelinesAndDescriptorSetsInit();

		SC.pipelinesAndDescriptorSetsInit();
		txt.pipelinesAndDescriptorSetsInit();
		uiQuad.pipelinesAndDescriptorSetsInit();
		// Same RP as the main pass: the flame draws inside it, right after
		// the scene, so it shares the depth buffer instead of needing its
		// own render pass the way UiQuad's 2D overlay does.
		flame.pipelinesAndDescriptorSetsInit(&RP);
	}

	// Here you destroy your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsCleanup() {
		P.cleanup();

		RP.cleanup();
		
		DSglobal.cleanup();
		
		SC.pipelinesAndDescriptorSetsCleanup();
		txt.pipelinesAndDescriptorSetsCleanup();
		uiQuad.pipelinesAndDescriptorSetsCleanup();
		flame.pipelinesAndDescriptorSetsCleanup();
	}

	// Here you destroy all the Models, Texture and Desc. Set Layouts you created!
	// You also have to destroy the pipelines
	void localCleanup() {
		DSLlocal.cleanup();
		DSLglobal.cleanup();

		P.destroy();

		RP.destroy();

		// Before SC.localCleanup(): that frees the colliders scene.json created, which
		// colliderSet also points at. It only deletes the ones it allocated itself, but
		// dropping the shared list first keeps the two ownership halves from overlapping.
		colliderSet.cleanup();
		allColliders.clear();

		SC.localCleanup();
		txt.localCleanup();
		uiQuad.localCleanup();
		flame.localCleanup();
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
		
		// Offscreen pass - always required
		// begin standard pass
		RP.begin(commandBuffer, currentImage);

		SC.populateCommandBuffer(commandBuffer, 0, currentImage);
		flame.populateCommandBuffer(commandBuffer, currentImage);

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

		// Torch flames' point lights. NOT going through SceneLights/
		// lights.json's own "instance"+"offset" anchoring: that reads the
		// instance's Wm once, at SceneLights::init() time, which is exactly
		// wrong for the held torch (its Wm doesn't exist in any meaningful
		// form until GameLogic() starts overwriting it every frame -- an
		// anchor taken before that would freeze the light at whatever the
		// placeholder scene.json transform happened to be, typically the
		// origin). So instead: appended straight into gubo here, every
		// frame, from the same Wm the flame itself now rides.
		for(const TorchFlame &tf : torchFlames) {
			if(gubo.lightCount >= MAX_LIGHTS) {
				break;
			}
			glm::vec3 worldPos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));

			LightData L{};
			L.pos = worldPos;
			L.dir = glm::vec3(0.0f, -1.0f, 0.0f);	// unused for a point light
			L.color = TORCH_LIGHT_COLOR;
			L.g = TORCH_LIGHT_G;
			L.beta = TORCH_LIGHT_BETA;
			L.cosIn = 1.0f;
			L.cosOut = 0.0f;
			L.type = LIGHT_POINT;

			gubo.lights[gubo.lightCount++] = L;
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

		gubo.eyePos = glm::vec3(glm::inverse(View)[3]);

		animTime += deltaT;
		gubo.time = animTime;

		DSglobal.map(currentImage, &gubo, 0);

		// Each flame's render matrix uses only its torch's WORLD POSITION
		// (inst->Wm * anchor), never its rotation: a flame stands upright
		// from its own buoyancy no matter how the torch holding it is tilted
		// or held, so inheriting the instance's full Wm here (as the flame
		// used to) dragged it sideways with the torch, most visibly on the
		// wall sconces (mounted at an angle) and the held one (grip tilt +
		// walking bob roll). FLAME_WORLD_SCALE stands in for the scale that
		// rotation-inheriting matrix would otherwise have carried.
		//
		// The glow billboard needs its own basis on top of that: it has to
		// face the camera rather than stand upright, so its right/up come
		// from View's own rotation (row 0/1 of a lookAt matrix are the
		// camera's world-space right/up -- glm is column-major, so that's
		// View[col][row] with row/col swapped from the usual read).
		glm::vec3 worldRight = glm::vec3(View[0][0], View[1][0], View[2][0]);
		glm::vec3 worldUp    = glm::vec3(View[0][1], View[1][1], View[2][1]);
		glm::vec3 worldFwd   = glm::vec3(View[0][2], View[1][2], View[2][2]);

		for(const TorchFlame &tf : torchFlames) {
			glm::vec3 anchorWorld = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
			// Every torch instance here uses a uniform scale (none has a
			// stretched axis), so its length alone -- taken off any one
			// basis column, rotation doesn't change a vector's length -- IS
			// the scale factor, with no need to fully decompose Wm.
			float instScale = glm::length(glm::vec3(tf.inst->Wm[0]));

			glm::mat4 flameWm = glm::translate(glm::mat4(1.0f), anchorWorld)
				* glm::scale(glm::mat4(1.0f), glm::vec3(instScale));
			flame.update(tf.flameId, ViewPrj * flameWm, currentImage);

			glm::vec3 glowCenter = anchorWorld + worldUp * (TORCH_GLOW_HEIGHT_OFFSET * instScale);
			float glowRadius = TORCH_GLOW_RADIUS * instScale;
			glm::mat4 glowM = glm::mat4(
				glm::vec4(worldRight * glowRadius, 0.0f),
				glm::vec4(worldUp * glowRadius, 0.0f),
				glm::vec4(worldFwd, 0.0f),
				glm::vec4(glowCenter, 1.0f)
			);
			flame.updateGlow(tf.flameId, ViewPrj * glowM, currentImage);
		}

		// defines the local parameters for the uniforms
		UniformBufferObject ubo{};		

		int instanceId;
		// character
		for(instanceId = 0; instanceId < SC.TI[0].InstanceCount; instanceId++) {
			ubo.mMat = SC.TI[0].I[instanceId].Wm;
			ubo.mvpMat = ViewPrj * ubo.mMat;
			ubo.nMat = glm::inverse(glm::transpose(ubo.mMat));

			// By Mid rather than by name, so no string hashing per frame.
			const Material &m = materials.forModel(SC.TI[0].I[instanceId].Mid);
			ubo.mS = m.specularColor;
			ubo.roughness = m.roughness;
			ubo.F0 = m.F0;
			ubo.k = m.k;
			ubo.flatNormals = m.flatNormals;
			
			// DS[1] = Pchar pass (main render): set0=DSLglobal, set1=DSLlocal
			SC.TI[0].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // global (light/camera)
			SC.TI[0].I[instanceId].DS[0][1]->map(currentImage, &ubo, 0); // camera MVPs
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

		// Held torch: sits at a fixed offset from the eye, in the camera's
		// own local space (right, up, -front; front is negated since the
		// camera looks down its own local -Z).
		if(handTorchInst != nullptr) {
			bool isWalking = grounded && (std::abs(m.x) > 0.01f || std::abs(m.z) > 0.01f);
			float bobTarget = isWalking ? 1.0f : 0.0f;
			torchBobBlend += (bobTarget - torchBobBlend) * (1.0f - std::exp(-deltaT / TORCH_BOB_BLEND_TAU));
			if(isWalking) {
				torchBobPhase += TORCH_BOB_SPEED * (sprinting ? 1.4f : 1.0f) * deltaT;
			}
			float bobLateral = sinf(torchBobPhase) * TORCH_BOB_LATERAL * torchBobBlend;
			float bobVertical = sinf(torchBobPhase * 2.0f) * TORCH_BOB_VERTICAL * torchBobBlend;
			float bobRollDeg = bobLateral * 90.0f;

			glm::mat4 camWm = glm::mat4(
				glm::vec4(right, 0.0f),
				glm::vec4(up, 0.0f),
				glm::vec4(-front, 0.0f),
				glm::vec4(eyePos, 1.0f)
			);
			glm::mat4 grip = glm::rotate(glm::mat4(1.0f), glm::radians(HAND_TORCH_TILT_DEG.x), glm::vec3(1.0f, 0.0f, 0.0f))
							* glm::rotate(glm::mat4(1.0f), glm::radians(HAND_TORCH_TILT_DEG.y), glm::vec3(0.0f, 1.0f, 0.0f))
							* glm::rotate(glm::mat4(1.0f), glm::radians(HAND_TORCH_TILT_DEG.z + bobRollDeg), glm::vec3(0.0f, 0.0f, 1.0f));
			glm::vec3 bobbedOffset = HAND_TORCH_OFFSET + glm::vec3(bobLateral, bobVertical, 0.0f);

			handTorchInst->Wm = camWm
				* glm::translate(glm::mat4(1.0f), bobbedOffset)
				* grip
				* glm::scale(glm::mat4(1.0f), glm::vec3(HAND_TORCH_SCALE));
		}

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