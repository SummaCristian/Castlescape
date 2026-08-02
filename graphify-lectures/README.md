# Knowledge graphs of the Computer Graphics course

Every graphify artifact for this course lives under this one folder, one subfolder per
graph. `docs/` holds only the professor's PDFs. Each subfolder has the same three files,
so once you know one you know them all:

- `graph.html` — open it in a browser, self-contained, no tooling needed
- `graph.json` — `graphify query "<question>" --graph graphify-lectures/<folder>/graph.json`
- `GRAPH_REPORT.md` — god nodes, surprising connections, community breakdown

## `lectures/` — the professor's slides

L00–L15, E01–E09 and the project rules. 618 nodes, 1165 edges, 36 communities, from 29
decks (~76,000 words). Static, so it never needs rebuilding.

Nodes carry the slide number in `source_location`, so an answer points you at the exact
slide. One honest limit: the displayed formulas were images in the PDFs and did not
extract, so this graph holds the concepts and the vocabulary, not the maths. For the
formulas use `labs/` — there they exist as shader code.

`workspace/` next to it is gitignored: it is the text extracted from the slide PDFs plus
the graphify working directory that queries it. You only need it to re-query or `--update`
the graph, and the extracted slide text is the professor's material to distribute, not
ours. Rebuild recipe at the bottom of this file.

## `labs/` — the 17 lab projects

E02–E17, the framework and the exam rules, as code. 2469 nodes, 4055 edges, 187
communities, from 148 curated files (~90,000 words). Static too. The corpus it was built
from no longer exists; this graph is all that survives of it.

## `repo/` — our own code

919 nodes, 1899 edges, 73 communities, from 20 files, as of the last rebuild.

This one goes stale as we write code. It is a convenience snapshot so a fresh clone gets
something useful without running anything; for live work rebuild with `/graphify` from the
repo root and use your own `graphify-out/`, which stays gitignored.

The rest of this file summarises the `labs/` graph in plain English.

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

Constraints extracted from the PDFs in `docs/`:

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

## Rebuilding

### The lectures graph

Run graphify with `graphify-lectures/lectures/workspace/` as both the corpus root and the
working directory — graphify looks for its cache at `<corpus root>/graphify-out/cache/`,
so the extracted text and the working directory have to sit side by side. Get that wrong
and the cache misses on all 29 files and you pay for the whole extraction again.

The semantic cache in there is already populated: a rebuild recovers all 618 nodes and
1165 edges for zero tokens. When you are done, copy `graphify-out/graph.html`,
`graph.json` and `GRAPH_REPORT.md` up one level into `lectures/`.

Regenerating the extracted text is a separate step: reading PDFs needs poppler, which is
not always installed, so the text was pulled out with `pypdf` into one `.txt` per deck,
mirroring the `Lessons/` `Excercises/` `Project/` layout, with a `## Slide N` heading per
page. That is what makes `source_location` point at real slide numbers.

### The graph of our own code

`graphify-out/` at the repo root is gitignored — it is generated output and it changes
with every commit. The snapshot in `repo/` is a convenience copy; regenerate your own with
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
professor's framework and ours to read. Also skip `docs/`, `graphify-lectures/`, the
texture folder and `graphify-out/` itself. Without this the corpus goes from ~32,000 words
to ~640,000, which is roughly a million tokens of extraction for a worse graph.
