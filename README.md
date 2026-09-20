# Castlescape

A first-person horror-exploration game set in a haunted castle, written from scratch in
C++ and Vulkan.


## The game

You wake up inside a castle with a torch in your hand and only one way out. The place is
dark, the corridors are long, and something else lives there.

Every now and then something changes, and the castle knows you are there. Survive it, find
your way through, and walk out into the daylight.

## Gameplay video



## Gallery



## Under the hood

Everything is rendered by hand: no game engine, no renderer library, only the Vulkan API
and the course framework.

- **Physically based lighting:** Cook-Torrance with a GGX microfacet model, with every
  torch and candle lighting the room around it.
- **Real-time shadows:** each flame casts shadows in every direction, through cube shadow
  maps captured on the fly.
- **HDR and bloom:** the scene is rendered in high dynamic range, where fire can be far
  brighter than white, then bloomed and tone mapped the way a camera would see it.
- **Procedural fire:** the flames, their flicker, their sparks and the way they lean when
  you walk are computed in the shaders, not animated by hand.
- **Ghosts:** translucent, rim-lit silhouettes that patrol a route, chase you during a
  hunt, and find their own way back afterwards.
- **A data-driven level:** geometry, materials, lights, flames and gameplay timings all
  live in data files, so the castle can be retuned without recompiling.

## Controls

- `WASD` - move
- `Space` - jump
- `Ctrl` - sprint
- `E` - interact: open a door, pick up an object, light a candle
- `G` - drop the key you are holding
- `R` - restart the run
- `F11` - fullscreen
- `Esc` - pause menu

Render resolution and anti-aliasing can be changed in the Settings menu, reachable from
both the title screen and the pause menu.

## Building and running

You need a C++20 compiler, CMake 3.21 or newer, and the
[Vulkan SDK](https://vulkan.lunarg.com/). GLFW and GLM are downloaded automatically.

From the `skeleton` folder, `./run.sh` configures, builds and launches the game. By hand:

```bash
cd skeleton
cmake --preset default
cmake --build --preset default
cd build && ./Castlescape
```

The game has to be launched from the build folder, where assets and shaders are copied.

## Repository layout

- `skeleton/source/src/` - the application: game state, rendering, gameplay
- `skeleton/source/include/custom/` - our own modules (flames, shadows, lights, menus, HUD)
- `skeleton/source/include/modules/` - the course framework, used as provided
- `skeleton/source/shaders/` - the GLSL shaders
- `skeleton/source/assets/` - models, textures and the scene description

## Authors

- Simone Rodari ([@SimoPolimi](https://github.com/SimoPolimi))
- Cristian Summa ([@SummaCristian](https://github.com/SummaCristian))

## About

Castlescape is the final project for the Computer Graphics course at Politecnico di Milano,
a.y. 2025-2026, built on the Vulkan starter framework provided with the course.

## Asset credits

- **MGCG castle pack** (`assets/models/Castle`, `assets/textures/Castle`) - course material
  provided by the professor.
- **Dracula's Castle Assetpack** (`assets/models/Dracula`, `assets/textures/Dracula`) - by
  Bob Hoegen, https://bob-hoegen.itch.io/dracula-asset-pack. Free download; the itch.io page
  states no licence and the archive ships no licence file, so no terms have been granted in
  writing. Used here for a non-commercial university project, with credit. Converted from the
  original `.unitypackage` to glTF via `tools/convert_assets.py`; the source `.unitypackage` is not
  redistributed here.
