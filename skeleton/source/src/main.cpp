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

// The uniform buffer object used in this example
struct UniformBufferObject {
	alignas(16) glm::mat4 mvpMat;
	alignas(16) glm::mat4 mMat;
};

struct GlobalUniformBufferObject {
	alignas(16) glm::vec3 lightDir;
	alignas(16) glm::vec4 lightColor;
	alignas(16) glm::vec3 eyePos;
};

struct Vertex {
	glm::vec3 pos;
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
	glm::vec3 camPos = glm::vec3(0.0f, 1.0f, 5.0f);
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
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, sizeof(UniformBufferObject), 1},
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
				  {0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, UV),
				         sizeof(glm::vec2), UV}
				});

		// initializes the render passes
		RP.init(this);
		// sets the blue sky
		RP.properties[0].clearValue = {0.0f,0.9f,1.0f,1.0f};

		// Pipelines [Shader couples]
		// The last array, is a vector of pointer to the layouts of the sets that will
		// be used in this pipeline. The first element will be set 0, and so on..
		
		P.init(this, &VD, "shaders/toChangeSimplePos.vert.spv",
						  "shaders/toChangeBlinnFromPos.frag.spv",
						  {&DSLglobal, &DSLlocal});


		// sets the size of the Descriptor Set Pool (it MUST be done before loading the scene)
		DPSZs.uniformBlocksInPool = 2;
		DPSZs.texturesInPool = 1;
		DPSZs.setsInPool = 2;

		// to support scene
		VDRs.resize(1);
		VDRs[0].init("VDposUV",  &VD);

		PRs.resize(1);
		PRs[0].init("BlinnPos", {
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
	}

	// Here you destroy your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsCleanup() {
		P.cleanup();

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
		static float lightRotationAngle = 0.0f; // Static variable to keep track of rotation
		lightRotationAngle += -0.5f * deltaT; // Increment rotation angle based on time

		const glm::mat4 lightView = glm::rotate(glm::mat4(1), glm::radians(lightRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f)) * 
									glm::rotate(glm::mat4(1), glm::radians(-45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		const glm::vec3 lightDir =  glm::vec3(lightView * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));

		GlobalUniformBufferObject gubo{};

		gubo.lightDir = lightDir;
		gubo.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)*5.0f;
		gubo.eyePos = glm::vec3(glm::inverse(View)[3]);

		DSglobal.map(currentImage, &gubo, 0);

		// defines the local parameters for the uniforms
		UniformBufferObject ubo{};		

		int instanceId;
		// character
		for(instanceId = 0; instanceId < SC.TI[0].InstanceCount; instanceId++) {
			ubo.mMat = SC.TI[0].I[instanceId].Wm;
			ubo.mvpMat = ViewPrj * ubo.mMat;
			
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
				const float EYE_HEIGHT = 1.0f;
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
				const float EYE_HEIGHT = 1.0f;
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
				const float EYE_HEIGHT = 1.0f;
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