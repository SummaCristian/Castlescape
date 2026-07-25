# Notes about the project
Here we are going to list all interesting/weird things we found/implemented/discovered during development.

This will help us remember them for the final oral presentation.

## Camera
- Input handling comes bundled in `Starter.hpp`. Everything is hardcoded. WASD for movement, arrow keys for camera yaw/pitch, and mouse cursor support as well. Mouse is ONLY supported with the **left button pressed**. We can't change that without either editing the Starter file (we can't!) or re-implementing it ourselves, which may not work with the prof's version of Starter...
- Gravity is handled in the camera. It's optional: allows flying as a cheat/debug option

## Colliders
- Floor collider has `AABB`-based ground detection, provided by `Colliders.hpp` and `Scene.hpp` files. Simply saying 
`"collider": "AABB"` in the `scene.json` file will apply ground detection to whatever model is used as floor. 
- Collision detection is optional, controlled by a flag. Allows to have a `no-clip` cheat/debug option