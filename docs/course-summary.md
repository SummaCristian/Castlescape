# Course material summary

What we extracted from the 17 lab projects (E02–E17), the professor's framework and the
exam-rules PDFs, condensed into something readable.

## The two graphs in this folder

**The course graph** — the 17 lab projects, the framework and the exam rules.
2469 nodes, 4055 edges, 187 communities, from 148 curated files (~90,000 words).
Static: the course material never changes, so it never needs rebuilding.

- `course-graph.html` — open in a browser, self-contained, no tooling needed
- `course-graph.json` — `graphify query "<question>" --graph docs/course-graph.json`

**The project graph** — our own code, as of the last rebuild.
919 nodes, 1899 edges, 73 communities, from 20 files.

- `project-graph.html` — same, open in a browser
- `project-graph.json` — `graphify query "<question>" --graph docs/project-graph.json`
- `project-graph-report.md` — god nodes, surprising connections, community breakdown

This one goes stale as we write code. Rebuild it with `/graphify` and re-copy it here, or
just work off your own `graphify-out/` (see the note at the bottom). The committed copy is
there so you get something useful on a fresh clone without running anything.

The rest of this file summarises the course graph in plain English.

---

## The lessons

The lab exercises in the corpus, in the order the graph links them:

`E02 - Setup the environment`
  The most complete reference project: app, camera, scene, assets, PBR, skinning. Second
  most connected node after the framework itself, with 101 edges.

`E03`
  Reduced variant of E02. The graph pairs them explicitly through a shared `ColliderShow.frag`.

`E04 - 3D projections in Excel`
  Projection matrices with no code. Useful for understanding the WVP chain before writing it.

`E05 - Build the models`
  Mesh construction. Introduces `PosNormUvTanWeights.vert` and `CookTorranceForCharacter.frag`.

`E06 - Motion systems`
  Skeletal animation. Reuses both of E05's shaders.

`E07 - Shadow map`
  Shadow mapping. Reuses the same two shaders again.

`E08 - Light Models`
  Lambert and Blinn-Phong. Start of the light-model chain.

`E09 - Texture Mapping`
  Texture mapping.

`E10 - Mesh Normals and Smoothing`
  Normals and smoothing. Relevant to us: our fragment shader currently rebuilds normals
  from `dFdx`/`dFdy` precisely because the vertex format carries none.

`E11 - Advanced BRDFs` (plus a `v2`)
  Advanced BRDFs. Closes the chain that started at E08.

`E12 - Setting up rendering Part 1`
  Incremental path: raw triangle, coloured triangle, geometry from code, 3D projection,
  loading objects.

`E13 - Setting up rendering Part 2`
  UBOs, global UBO, command buffers.

`E14 - Advanced texturing`
  Advanced texturing, tangent-space vertex pipeline `MeshTBN.vert`.

`E15 - Image Based Lighting`
  IBL. Shares `MeshTBN.vert` with E14.

`E16 - Setting up rendering Part 3`

`E17 - Maze`
  Instanced rendering, per-instance push constants, offscreen minimap.

---

## Three chains the graph found

Links between lessons you cannot see by looking at one folder at a time.

**Light models**: E08 (Lambert/Blinn) to E12 (fixed-light Lambert+Blinn) to E11 (advanced
BRDFs). If we have to justify a lighting choice at the oral exam, the story starts here.

**Animated character**: E05, E06 and E07 literally share the same two shaders,
`PosNormUvTanWeights.vert` for skinning and `CookTorranceForCharacter.frag` for
Cook-Torrance. Three different lessons, one pipeline.

**Tangent space**: E14 and E15 share `MeshTBN.vert`. Normal mapping and IBL rest on the
same geometric base.

---

## Most connected nodes

`BaseProject` (195 edges) dominates everything: it is the framework class every project
inherits from, ours included. Then come E02, E05, `Pipeline`, E10, `TextMaker`, `Scene`,
E06 and E07.

No import cycles anywhere in the corpus.

---

## What the exam rules say

Constraints extracted from the PDFs in this folder:

- The framework is mandatory: C++ plus Vulkan plus `Starter.hpp`. **`Starter.hpp` must not
  be modified** — the professor grades against his own copy.
- We write the shaders ourselves. AI may only be a minor contribution.
- Teams of 2 to 3, each member registers individually with a motivation.
- Grading: project up to 25 points plus 8 points of individual questions. We are expected
  to know the code in detail.
- Mandatory technical content: explicit Vulkan pipeline setup, WVP transform chain, mesh
  and vertex formats, navigation controls, light models and BRDFs, direct and indirect
  lighting, materials and textures.

## The 8 official topics

Haunted Castle Explorer · Dungeon Tavern NPC · Enchanted Forest Wanderer ·
Modular School Classroom Planner · Hospital Resource Mover · Tabletop Dice RPG Arena ·
Animal Herding Simulator · Vehicle Simulator

The graph links our first-person camera notes to the three exploration topics on its own
(Haunted Castle, Enchanted Forest, Dungeon Tavern) — that is where the current skeleton
is heading.

---

## Rebuilding the graph of our own code

`graphify-out/` is gitignored — it is generated output and it changes with every commit.
The snapshot in `docs/project-graph.*` is a convenience copy; regenerate your own with
`/graphify` whenever you want it current.

One thing is committed out of `graphify-out/`: `cache/semantic/` (80 KB). That cache is
keyed by file content plus repo-relative path, so it is machine-independent — your rebuild
reuses it and the semantic extraction costs zero tokens.

Two things that build needs, or it silently goes wrong:

**Widen the recognised extensions.** graphify does not know `.vert` / `.frag`, so all six
of our shaders get dropped without a word. Before scanning:

```python
from graphify import detect as D
D.DOC_EXTENSIONS = D.DOC_EXTENSIONS | {".vert", ".frag", ".comp", ".glsl"}
```

**Exclude the vendored headers.** `skeleton/source/include/` is third-party single-header
libraries (json, stb, tiny_gltf, plusaes, sdefl/sinfl) except for `modules/`, which is the
professor's framework and ours to read. Also skip `docs/`, the texture folder and
`graphify-out/` itself. Without this the corpus goes from ~32,000 words to ~640,000, which
is roughly a million tokens of extraction for a worse graph.
