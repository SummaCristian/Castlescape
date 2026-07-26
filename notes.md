# Notes about the project
Here we are going to list all interesting/weird things we found/implemented/discovered during development.

This will help us remember them for the final oral presentation.

## Camera
- Input handling comes bundled in `Starter.hpp`. Everything is hardcoded. WASD for movement, arrow keys for camera yaw/pitch, and mouse cursor support as well. Mouse is ONLY supported with the **left button pressed**. We can't change that without either editing the Starter file (we can't!) or re-implementing it ourselves, which may not work with the prof's version of Starter...
- Gravity is handled in the camera. It's optional: allows flying as a cheat/debug option
- Jump is bound to spacebar, which `Starter.hpp` already wires to the `fire` output of `getSixAxis`, so no changes were needed there. It's a simple upward velocity impulse, gated behind its own optional flag (like the other cheats), and only triggers on a fresh key press while grounded.
- "Grounded" can't be inferred just by checking if vertical velocity is exactly `0`: looking up/down while walking bleeds a bit of the camera's front vector into vertical movement, which can leave you standing but with nonzero velocity. Ground contact is instead computed directly against the colliders every frame.
- Sprint is bound to Ctrl (not wired by `Starter.hpp`, so polled directly via `glfwGetKey`), gated behind its own flag too. Can only be started while grounded, but once sprinting, releasing Ctrl always stops it immediately even mid-air, since our movement isn't a persisted velocity. There's no "momentum" to preserve through a jump.

## Colliders
- Floor collider has `AABB`-based ground detection, provided by `Colliders.hpp` and `Scene.hpp` files. Simply saying 
`"collider": "AABB"` in the `scene.json` file will apply ground detection to whatever model is used as floor. 
- Collision detection is optional, controlled by a flag. Allows to have a `no-clip` cheat/debug option

## Cheats / debug toggles
- All debug/cheat flags (`gravityEnabled`, `collisionEnabled`, `jumpEnabled`, `sprintEnabled`) are grouped into one `CheatFlags` struct, held as a single `cheats` member on the app class, instead of being loose booleans scattered among the camera/physics state.
- No persistence on purpose: it's a plain struct with default member initializers, so it just resets to the "legit" defaults (everything `true`, i.e. cheats off) every time the app starts. No config file, no serialization needed.
- Grouping them like this makes them easy to find as a block, easy to reset all at once, and gives a natural place to hang a future debug UI or keybind toggles off of (e.g. iterate/reflect over the struct's fields) without polluting the rest of the class's member list.
- Call sites read through the struct, e.g. `if(cheats.gravityEnabled)`, `if(cheats.collisionEnabled)`, `if(cheats.jumpEnabled)`, `if(!cheats.sprintEnabled || !ctrlHeld)`.
- The numeric tuning behind those cheats (`moveSpeed`, `sprintMultiplier`, `jumpSpeed`, `gravity`) lives in a separate `MovementParams` struct (`movement` member), not in `CheatFlags`. Reasoning: those bools answer "is this cheat on/off", these floats answer "how strong is it", different concern, so a different struct, even though both sit next to each other on the class and follow the same no-persistence pattern.