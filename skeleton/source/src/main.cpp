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

class Skeleton26ReplaceName : public BaseProject {
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

	// The flame at a torch's head. See custom/Flame.hpp: camera-facing
	// billboard layers shaded by a procedural fire field, drawn as one more
	// object inside the main pass. One Flame instance drives every torch in
	// the scene (Flame::spawn), held one included -- that's what the reusable
	// design was for.
	Flame flame;

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
		// the point light's colour and reach -- but NOT the flame's height any
		// more, see heightScale. Chased toward its noise-driven target by a
		// critically-damped spring (intensityVel below) instead of being
		// assigned raw: the raw signal is a fresh noise sample every frame,
		// and slamming the whole flame to it at once was exactly the visible
		// "jumping". A spring is continuous in value AND slope, so brightness
		// glides, yet still ducks through a gutter in a couple tenths of a
		// second.
		float intensity = 1.0f;
		float intensityVel = 0.0f;

		// HEIGHT envelope, ~0.78..1.09. Same underlying signal as intensity,
		// but compressed (a flame's height varies far less than its light
		// output) and low-passed much harder (FLAME_HEIGHT_TAU): light
		// responds to combustion instantly, the fuel column's height follows
		// it late. One shared signal for both was the other half of the old
		// "jumping" -- the flame teleported between heights at flicker rate.
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

		// That lean, resolved into the billboard's own axes (x = the
		// billboard's right, y = its forward) and expressed in the same units
		// the flame's local geometry uses. Uploaded straight to the shader.
		glm::vec2 lean = glm::vec2(0.0f);
	};
	std::vector<TorchFlame> torchFlames;

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
	// out of it. The old mesh flame only needed to nestle down to 0.38, but
	// the shader flame's field fades out right at its own y=0 (Flame.frag's
	// baseFade plus the alpha window), so its first VISIBLE pixels sit a few
	// percent up the card -- the anchor compensates by sinking that much
	// further into the cup, and the fade doubles as the flame emerging from
	// inside it rather than balancing on the rim.
	static constexpr glm::vec3 TORCH_FLAME_ANCHOR = glm::vec3(-0.384f, 0.30f, 0.0f);

	// The flame's size, in the torch model's own local units, so it rides
	// each instance's uniform scale: full size on the wall-mounted torches,
	// shrunk to match on the held one (whose instance is scaled down to
	// arm's-length size, HAND_TORCH_SCALE). A single fixed world-space size
	// instead made the flame look right on the (small) held torch and
	// comically undersized on the (full-size) wall ones.
	//
	// HALF_WIDTH is deliberately wider than the old mesh's widest ring: the
	// billboard's noise field eats into its own silhouette from the edges in,
	// so the quad has to be bigger than the fire that ends up drawn inside it
	// or the flame gets visibly clipped to a rectangle.
	static constexpr float FLAME_HEIGHT = 0.95f;
	static constexpr float FLAME_HALF_WIDTH = 0.20f;

	// The torch flame's point light. One color/falloff for every torch in
	// the scene (held and wall-mounted alike): they're all the same kind of
	// fire, so there's nothing to author per-instance. Tighter (lower g)
	// than the gate lanterns (SceneLights.hpp/lights.json): a torch flame is
	// a much smaller, closer source than a lamp head.
	static constexpr glm::vec3 TORCH_LIGHT_COLOR = glm::vec3(1.0f, 0.5f, 0.16f);
	// 2.1 rather than the original 1.6: with the falloff (g/d)^beta this
	// lifts the light by a flat ~45% at every distance, so the torches
	// genuinely carry into the room instead of only rimming their own wall.
	static constexpr float TORCH_LIGHT_G = 2.1f;
	static constexpr float TORCH_LIGHT_BETA = 1.4f;

	// How far a torch light still gets uploaded, and how many may be live at
	// once. CookTorrance.frag loops over every light for every fragment (times
	// the sample count, since Starter.hpp forces per-sample shading), so an
	// uploaded light costs a full GGX evaluation across the whole screen
	// whether or not it can be seen.
	//
	// The radius is set generously on purpose: at g = 2.1 and beta = 1.4 a
	// torch 25 units out contributes about 3% of its colour, which is still
	// below what the ambient term hides, so nothing you could actually notice
	// goes dark. This is headroom, not the fix -- the MSAA change in
	// localInit() is what actually bought the frame budget back, and all six
	// torches fit comfortably inside these limits in the current scene.
	static constexpr float TORCH_LIGHT_CULL_DIST = 25.0f;
	static constexpr int TORCH_LIGHT_MAX_LIVE = 8;

	// Fire envelope. The fast term used to be sampled at 12 Hz and weighted
	// half the whole signal, which is physically the right flicker band but
	// looked wrong for a structural reason: a WHOLE-FLAME envelope applies
	// that twitch to every pixel at once, and no real flame changes uniformly
	// -- that reads as a brightness dial being wiggled. The fast twitch now
	// lives in Flame.frag as a per-pixel shimmer that varies along the flame;
	// the CPU keeps a slower 7 Hz term at reduced weight purely so the light
	// the torch casts still dances a little, and the slower terms underneath
	// it stop the result from reading as uniform hash. (A plain sine was what
	// the first version used, and the ear-equivalent problem applies to the
	// eye: a pure period is a metronome you lock onto within two seconds.)
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
		// Update Render Passes. Every one of them: the scene and the composite
		// follow the window, the three bloom targets follow it divided down.
		// Their attachment images are torn down and rebuilt around this by
		// pipelinesAndDescriptorSetsCleanup()/Init(), which Starter.hpp calls
		// on either side of a resize.
		RP.width = w;
		RP.height = h;
		RPcomposite.width = w;
		RPcomposite.height = h;
		RPbright.width = RPblurH.width = RPblurV.width = std::max(1, w / BLOOM_DIV);
		RPbright.height = RPblurH.height = RPblurV.height = std::max(1, h / BLOOM_DIV);

		// windowWidth/windowHeight are otherwise only set once in
		// setWindowParameters() and never refreshed here; the cheat HUD
		// needs the current size for its pixel-based layout math.
		windowWidth = (uint32_t)w;
		windowHeight = (uint32_t)h;

		// updates the textual output
		txt.resizeScreen(w, h);
		uiQuad.resizeScreen(w, h);
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
		const VkClearValue SKY = {.color = {.float32 = {0.0f, 0.9f, 1.0f, 1.0f}}};
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

	// Width/height of the bloom chain's targets, derived from the swapchain.
	// Clamped at 1 so a minimised or absurdly narrow window can't ask for a
	// zero-sized image.
	int bloomWidth() const {
		return std::max(1, (int)swapChainExtent.width / BLOOM_DIV);
	}
	int bloomHeight() const {
		return std::max(1, (int)swapChainExtent.height / BLOOM_DIV);
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

		// Anti-aliasing level, set BEFORE RP.init() below reads it.
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
		msaaSamples = VK_SAMPLE_COUNT_4_BIT;

		// initializes the render passes. The scene one no longer draws to the
		// screen: it renders into an offscreen floating-point target which the
		// bloom chain and the composite then read back. See
		// buildPostAttachments() for what each attachment is and why.
		buildPostAttachments();

		// ATDEP_SIMPLE rather than the default ATDEP_SURFACE_ONLY: the scene's
		// output is now sampled by a later pass, so it needs the dependency
		// pair that orders a colour write against a subsequent shader read
		// (and, in the other direction, against the NEXT frame overwriting it).
		RP.init(this, -1, -1, -1, &hdrAtt,
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

		// Pipelines [Shader couples]
		// The last array, is a vector of pointer to the layouts of the sets that will
		// be used in this pipeline. The first element will be set 0, and so on..

		P.init(this, &VD, "shaders/PosNormUV.vert.spv",
						  "shaders/CookTorrance.frag.spv",
						  {&DSLglobal, &DSLlocal});

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

		// sets the size of the Descriptor Set Pool (it MUST be done before loading the scene)
		// The four post-processing sets are counted in here too: one uniform
		// block each, and five sampled textures between them (one apiece for
		// the bright pass and the two blurs, two for the composite).
		DPSZs.uniformBlocksInPool = 2 + 4;
		DPSZs.texturesInPool = 1 + 5;
		DPSZs.setsInPool = 2 + 4;

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
		auto addTorchFlame = [&](const char *id, glm::vec3 anchor, bool heldByCamera = false) {
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
			// Offsets this torch into a different part of the CPU noise field,
			// so no two gutter at the same moment. Scaled up because fireNoise
			// hashes on the integer lattice: a fractional offset would leave
			// neighbouring torches sampling the same two lattice points.
			tf.phase = seed * 37.0f;
			torchFlames.push_back(tf);
		};

		if(handTorchInst != nullptr) {
			addTorchFlame("handTorch", TORCH_FLAME_ANCHOR, true);
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
		Pbright.create(&RPbright);
		PblurH.create(&RPblurH);
		PblurV.create(&RPblurV);
		Pcomposite.create(&RPcomposite);

		DSglobal.init(this, &DSLglobal, {});

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
		// Same RP as the scene: the flame draws inside it, right after the
		// scene geometry, so it shares the depth buffer instead of needing its
		// own render pass the way UiQuad's 2D overlay does -- and so its
		// over-1.0 colours land in the HDR attachment where bloom can find
		// them.
		flame.pipelinesAndDescriptorSetsInit(&RP);
	}

	// Here you destroy your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsCleanup() {
		P.cleanup();
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
		DSbright.cleanup();
		DSblurH.cleanup();
		DSblurV.cleanup();
		DScomposite.cleanup();

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
		DSLpost1.cleanup();
		DSLpost2.cleanup();

		if(Mpost != nullptr) {
			Mpost->cleanup();
		}

		P.destroy();
		Pbright.destroy();
		PblurH.destroy();
		PblurV.destroy();
		Pcomposite.destroy();

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
		// The whole HDR chain goes into this one command buffer, in order.
		// Ordering between the passes is handled by their render pass
		// dependencies rather than by explicit barriers -- see the comment on
		// the RPbright/RPblurH/RPblurV members. The text and HUD passes are
		// separate command buffers submitted after this one (submit orders
		// 10000 and 9000 against this one's 0), so they end up drawing on top
		// of the composited frame.

		// 1. The scene, into the offscreen HDR target.
		RP.begin(commandBuffer, currentImage);
		SC.populateCommandBuffer(commandBuffer, 0, currentImage);
		flame.populateCommandBuffer(commandBuffer, currentImage);
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

		// The camera's world position, needed below to cull torch lights by
		// distance, and again by the flame billboards to work out which way to
		// face. Pulled up here from where it used to sit (just before the
		// DSglobal.map() call) so both have it.
		const glm::mat4 camToWorld = glm::inverse(View);
		const glm::vec3 eyePos = glm::vec3(camToWorld[3]);

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
			// This deliberately does NOT divide by the flame's world half-width,
			// which is what the first attempt did and why the flame ended up
			// permanently folded over. The held torch's half-width is about
			// 0.07 world units, so dividing by it multiplied every velocity by
			// ~14: merely turning on the spot swings the torch through roughly
			// 2 units/s, which saturated the lean to its cap and pinned it
			// there. It also made the effect scale-dependent in the wrong
			// direction -- the small held torch reacted three times harder than
			// a full-size wall one, when they should behave identically.
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
		// Culled by distance and capped in count, nearest first: every light
		// in gubo costs a full BRDF evaluation for every fragment of every
		// object, multiplied by the sample count. See TORCH_LIGHT_CULL_DIST
		// on why dropping the far ones is invisible.
		{
			// Index + squared distance, so the sort doesn't pay for a sqrt it
			// does not need.
			std::vector<std::pair<float, const TorchFlame *>> nearest;
			nearest.reserve(torchFlames.size());
			const float cullSq = TORCH_LIGHT_CULL_DIST * TORCH_LIGHT_CULL_DIST;
			for(const TorchFlame &tf : torchFlames) {
				glm::vec3 worldPos = glm::vec3(tf.inst->Wm * glm::vec4(tf.anchor, 1.0f));
				glm::vec3 d = worldPos - eyePos;
				float dSq = glm::dot(d, d);
				if(dSq <= cullSq) {
					nearest.push_back({dSq, &tf});
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
				L.color = TORCH_LIGHT_COLOR * tf.intensity;
				L.g = TORCH_LIGHT_G * (0.88f + 0.12f * tf.intensity);
				L.beta = TORCH_LIGHT_BETA;
				L.cosIn = 1.0f;
				L.cosOut = 0.0f;
				L.type = LIGHT_POINT;

				gubo.lights[gubo.lightCount++] = L;
				live++;
			}
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
				if(!tf.heldByCamera) {
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
			const glm::vec2 fullTexel = glm::vec2(1.0f / (float)swapChainExtent.width,
												  1.0f / (float)swapChainExtent.height);
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
			// the texel size is not read at all. The tone map that used to live
			// at the end of CookTorrance.frag now happens here instead, which
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
		// This is also what fixes the flame not following the player's
		// orientation at all. The old matrix was translate * scale with NO
		// rotation at all, so the mesh's local axes stayed welded to world
		// X/Z: turning on the spot swung the flame's own asymmetric crown
		// around relative to the torch holding it. A camera-facing billboard
		// has no such orientation to get wrong -- it presents the same face
		// from every angle by construction -- so the problem stops existing
		// rather than being corrected. What is left is the part that SHOULD
		// respond to movement, and that now goes in deliberately as tf.lean.
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

			float halfWidth = FLAME_HALF_WIDTH * instScale;
			float height = FLAME_HEIGHT * instScale;

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
						 tf.lean, 1.0f + GLARE_FLAME_GAIN * tf.glare, currentImage);
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