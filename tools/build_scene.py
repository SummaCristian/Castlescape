#!/usr/bin/env python3
"""
Generatore del livello: riscrive il blocco di istanze del dungeon dentro
scene.json e le collisioni autorate dentro colliders.json.

Il kit Dracula e' modulare su una griglia di 7.2 unita'. Le convenzioni qui
sotto sono ricavate dalla geometria reale dei modelli (accessor POSITION),
non assunte:

  piastrella (i,j) -> x in [7.2 i, 7.2 (i+1)], z in [7.2 j, 7.2 (j+1)]
  i cresce verso EST (+X), j cresce verso SUD (+Z), come il resto del motore.

  SM_StoneFloor_02    x [-7.2,0]  z [0,7.2]     rispetto al translate
  SM_StoneCeiling_02  idem, a y = 6.213
  SM_WallStraight_02  x [-1.242,0] z [0,7.2]    -> lastra sul lato EST della cella
  SM_WallCorner_02    x [-7.2,0]  z [-7.2,0]    -> due bracci, lato E e lato S
  SM_WallDoor_Hole_02 come lo straight ma con l'arcata scavata (z 2.47..4.71)
  SM_Door_01          pende da un cardine: z [-2.462, 0], x [-0.037, 0.430]

Rotazioni (eulerAngles Y): 0 = est, 90 = nord, 180 = ovest, 270 = sud.

STRUTTURA DEL LIVELLO (vedi memory/level-course-rebuild.md):

        ROOM A  (studio -- il LIBRO)
           |  porta N "iron"
  ROOM B --+---------- HUB 3x3 --- porta E "gold" -> USCITA (ExitGlow)
 (torcia,  | libreria "book"  (chiave "iron" sul tavolo)
  "bronze")|  porta S "bronze"
        ROOM C  (puzzle di salto -- la chiave "gold")
           ^
    ALLEY INTRO (spawn, un barile da scavalcare) entra da OVEST

Grafo chiavi (lineare, nessun soft-lock):
  iron (hub) -> porta N -> book (A) -> libreria -> torcia + bronze (B)
  -> porta S -> gold (C) -> porta E -> box di vittoria fuori.

Uso:  python tools/build_scene.py
"""

import json
import math
import os

TS = 7.2
Y = 0.02
CEIL_Y = 6.213

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "skeleton/source/assets/scenes/scene.json")
COLLIDERS = os.path.join(ROOT, "skeleton/source/assets/scenes/colliders.json")

# --- il kit: chiave logica -> (id modello, id texture) -----------------------
KIT = {
    "Floor":   ("dungeonFloor",     "dungeonFloorTex"),
    "Ceiling": ("dungeonCeiling",   "dungeonCeilingTex"),
    "Wall":    ("dungeonWall",      "dungeonWallTex"),
    "Corner":  ("dungeonCorner",    "dungeonCornerTex"),
    "DoorWall":("dungeonDoorWall",  "dungeonDoorWallTex"),
    "Panel":   ("dungeonDoorPanel", "dungeonDoorPanelTex"),
    "Chains":  ("doorChains01",     "dungeonDoorPanelTex"),
    "Padlock": ("padlock01",        "keyTex"),
    "Shelf":   ("bookshelf01",      "dungeonBookshelfTex"),
    "Book":    ("book01",           "dungeonBannerTex"),
    "Key":     ("key",              "keyTex"),
    "Barrel":  ("dungeonBarrel",    "dungeonBarrelTex"),
    "Torch":   ("dungeonTorch",     "dungeonTorchTex"),
    "HandTorch":("dungeonTorchHeld","dungeonTorchTex"),
}

inst = []          # CookTorrance elements
ghosts = []        # Spectral elements
colliders = {}     # colliders.json

_ids = set()


def add(id, key, pos, rot=None, scale=None, tex=None):
    assert id not in _ids, "id duplicato: %s" % id
    _ids.add(id)
    model, texture = KIT[key]
    e = {"id": id, "model": model, "texture": [tex or texture],
         "translate": [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)]}
    if rot is not None:
        e["eulerAngles"] = [0.0, float(rot), 0.0]
    if scale is not None:
        e["scale"] = [float(scale)] * 3
    inst.append(e)
    return e


# --- geometria delle celle ---------------------------------------------------
NEIGH = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
WALL_ROT = {"E": 0, "N": 90, "W": 180, "S": 270}


def wall_pos(i, j, side):
    if side == "E":
        return (TS * (i + 1), Y, TS * j)
    if side == "N":
        return (TS * i, Y, TS * j)
    if side == "W":
        return (TS * i, Y, TS * (j + 1))
    return (TS * (i + 1), Y, TS * (j + 1))          # S


CORNER = {"SE": (0,   (TS, TS)), "NE": (90, (TS, 0)),
          "NW": (180, (0, 0)),   "SW": (270, (0, TS))}
CORNER_SIDES = {"NE": ("N", "E"), "NW": ("N", "W"),
                "SE": ("S", "E"), "SW": ("S", "W")}


def corner_pos(i, j, kind):
    _, (ox, oz) = CORNER[kind]
    return (TS * i + ox, Y, TS * j + oz)


def rot_xz(dx, dz, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (dx * c + dz * s, -dx * s + dz * c)


# offset del battente dal proprio muro-arcata, nel frame locale del muro
# (verificato: dhDoor rot180 + questo == dhDoorPanel in scena)
PANEL_OFF = (-0.717, 4.821)
# La libreria segreta NON usa la stessa posa del battente: il suo mesh
# (make_bookshelf.py) ha il fronte -- dove stanno i libri e va infilato quello
# mancante -- sul +X locale, quindi va ruotata di 180 rispetto al muro perche'
# il fronte guardi DENTRO la stanza da cui la si apre, non il passaggio segreto.
# Offset e rotazione verificati contro il vecchio dsDoor(rot90)/dsShelfPanel(rot270).
SHELF_OFF = (-0.66, 2.359)
SHELF_ROT_ADD = 180

# SM_WallStraight_02 NON e' una lastra piatta: agli estremi ha due PILASTRI che
# sporgono (faccia a -1.242), in mezzo un pannello piatto RIENTRATO (~-0.855).
# Le torce vanno sul pannello piatto centrale, non sul pilastro: along = 3.6
# (centro tessera) le tiene lontane dai pilastri.
TORCH_ALONG = 3.6
# TORCH_EMBED: l'origine del mesh torcia arretra rispetto alla staffa, e in piu'
# il pannello centrale e' rientrato ~0.39 rispetto al piano nominale del muro.
# Questo e' l'UNICA manopola: spinge ogni torcia, di questa quantita', verso la
# pietra del pannello piatto. ~0.5 la mette a filo; alzare se resta staccata,
# abbassare se sprofonda.
TORCH_EMBED = 0.75

ARCH_BOXES = [
    [-1.242, 0.00, 0.00,  0.0, 6.19, 2.47],
    [-1.242, 0.00, 4.71,  0.0, 6.19, 7.20],
    [-1.242, 4.85, 2.47,  0.0, 6.19, 4.71],
]
CORNER_BOXES = [
    [-1.242, 0.00, -7.20,  0.0, 6.19, 0.0],
    [-7.20,  0.00, -1.242, 0.0, 6.19, 0.0],
]


# =========================================================================
#  DEFINIZIONE DEL LIVELLO
# =========================================================================
# ogni area: prefisso -> lista di celle (i,j)
AREAS = {
    "ia": [(i, 3) for i in range(-4, 0)],                       # alley intro (SO del hub)
    "hb": [(i, j) for i in range(0, 4) for j in range(0, 4)],   # hub 4x4
    "b1": [(1, -2), (1, -1)],                                   # alley nord
    "ra": [(i, j) for i in range(0, 4) for j in range(-5, -2)], # room A 4x3
    "b2": [(-2, 0), (-1, 0)],                                   # alley ovest
    "rb": [(i, j) for i in range(-6, -2) for j in range(-2, 2)],# room B 4x4
    "b3": [(1, 4), (1, 5)],                                     # alley sud
    "rc": [(i, j) for i in range(0, 4) for j in range(6, 9)],   # room C 4x3
}
UNION = {c for cells in AREAS.values() for c in cells}
assert len(UNION) == sum(len(c) for c in AREAS.values()), "celle sovrapposte fra aree"

# varchi: (cella, lato) del muro-arcata -> (prefisso id, tipo, keyId, label)
#   tipo: "open"  battente semplice, si apre e basta
#         "lock"  battente + catene + lucchetto
#         "exit"  come lock ma si apre verso l'ESTERNO (verso la luce)
#         "shelf" libreria segreta
PORTALS = {
    ((0, 3), "W"): ("iaDoor",  "open",  "",       ""),
    ((1, 0), "N"): ("hbDoorN", "lock",  "iron",   "iron key"),
    ((1, 3), "S"): ("hbDoorS", "lock",  "bronze", "bronze key"),
    ((3, 1), "E"): ("hbDoorE", "exit",  "gold",   "gold key"),
    ((0, 0), "W"): ("hbShelf", "shelf", "book",   "old book"),
}

# barili da scavalcare (callback: intro insegna il salto, room C lo richiede)
BARRELS = [
    ("iaJump",  (-14.4, Y, 25.2), 0.48),      # ostacolo tutorial, meta' alley
    ("rcStep1", (10.8,  Y, 50.4), 0.48),      # puzzle room C: appoggi verso la chiave
    ("rcStep2", (14.4,  Y, 52.0), 0.48),
    ("rcStep3", (18.0,  Y, 50.4), 0.48),
]

# torce a muro: (id, cella_i, cella_j, lato, along)  -- il muro su `lato` di quella
# tessera; along = offset lungo il muro dal suo spigolo (0..7.2). ~7.0 mette la
# torcia sul pilastro presso il giunto con la tessera successiva. Ogni (cella,lato)
# DEVE essere un bordo con muro (o un portale): validato in gen_props().
A = TORCH_ALONG
TORCHES = [
    ("iaTorch",   -3, 3, "S", A),              # unica luce dell'alley, fioca
    ("hbTorchN1",  0, 0, "N", A),
    ("hbTorchN2",  2, 0, "N", A),
    ("hbTorchW",   0, 1, "W", A),
    ("hbTorchE",   3, 2, "E", A),
    ("hbTorchS1",  0, 3, "S", A),
    ("hbTorchS2",  2, 3, "S", A),
    ("raTorchW",   0, -4, "W", A),
    ("raTorchE",   3, -4, "E", A),
    ("raTorchN1",  1, -5, "N", A),
    ("rbTorch",   -4, 1, "S", A),              # una sola, presso l'ingresso: l'angolo NO resta buio
    ("rcTorchW",   0, 7, "W", A),
    ("rcTorchE",   3, 7, "E", A),
]

# raccoglibili
PICKUPS = [
    ("hbKeyIron",   "Key", (14.4,  1.10, 14.4),  0.0032),   # hub (come su un tavolo)
    ("raBook",      "Book", (14.4,  1.10, -25.2), 1.0),     # room A
    ("rbKeyBronze", "Key", (-41.0, 1.05, -11.0), 0.0032),   # room B, angolo NO buio
    ("rcKeyGold",   "Key", (18.0,  2.30, 50.4),  0.0032),   # room C, in cima ai barili
]

GHOSTS = [
    ("ghost",  (14.4, 2.2, 14.4)),    # hub
    ("ghost2", (-28.8, 2.2, 0.0)),    # room B
    ("ghost3", (14.4, 2.2, 50.4)),    # room C
]

# camera di spawn: estremita' ovest dell'alley, guarda +X lungo il corridoio
SPAWN = (TS * -4 + 1.4, 1.8, TS * 3 + 3.6)      # (-27.4, 1.8, 25.2)
HAND_TORCH = ("handTorch", (-28.8, -0.12, 0.0), [0.0, 35.0, -90.0])


def torch_on(id, i, j, side, along=3.6, y=3.5):
    rot = WALL_ROT[side]
    if side == "E":
        pos = (TS * (i + 1) - 1.242 + TORCH_EMBED, y, TS * j + along)
    elif side == "W":
        pos = (TS * i + 1.242 - TORCH_EMBED, y, TS * j + along)
    elif side == "N":
        pos = (TS * i + along, y, TS * j + 1.242 - TORCH_EMBED)
    else:  # S
        pos = (TS * i + along, y, TS * (j + 1) - 1.242 + TORCH_EMBED)
    add(id, "Torch", pos, rot)


# =========================================================================
#  GENERAZIONE
# =========================================================================
def gen_structure():
    for i, j in sorted(UNION):
        floor_i = "%sF_%d_%d" % (area_of(i, j), i, j)
        add(floor_i, "Floor", (TS * (i + 1), Y, TS * j))
        add(floor_i.replace("F_", "C_"), "Ceiling", (TS * (i + 1), CEIL_Y, TS * j))

    covered = set()          # (i,j,side) gia' chiusi

    # 1) angoli: solo celle con ESATTAMENTE due lati di bordo adiacenti
    for i, j in sorted(UNION):
        b = boundary_sides(i, j)
        for kind, sides in CORNER_SIDES.items():
            if set(sides) == b and not any((i, j, s) in PORTALS_BY_CELL for s in sides):
                cid = "%sK_%d_%d" % (area_of(i, j), i, j)
                add(cid, "Corner", corner_pos(i, j, kind), CORNER[kind][0])
                colliders[cid] = {"boxes": [list(x) for x in CORNER_BOXES]}
                for s in sides:
                    covered.add((i, j, s))

    # 2) varchi
    for (i, j), side in PORTALS:
        gen_portal(i, j, side)
        covered.add((i, j, side))

    # 3) muri dritti su ogni bordo rimasto scoperto
    for i, j in sorted(UNION):
        for s in boundary_sides(i, j):
            if (i, j, s) in covered:
                continue
            add("%sW_%d_%d_%s" % (area_of(i, j), i, j, s), "Wall",
                wall_pos(i, j, s), WALL_ROT[s])
            covered.add((i, j, s))

    # verifica: nessun bordo del livello resta aperto
    missing = [(i, j, s) for (i, j) in UNION for s in NEIGH
               if (i + NEIGH[s][0], j + NEIGH[s][1]) not in UNION
               and (i, j, s) not in covered]
    assert not missing, "bordi esterni scoperti: %s" % missing


def gen_portal(i, j, side):
    prefix, kind, keyid, _ = PORTALS[((i, j), side)]
    rot = WALL_ROT[side]
    wp = wall_pos(i, j, side)
    wall_id = prefix + "Wall"
    add(wall_id, "DoorWall", wp, rot)
    colliders[wall_id] = {"boxes": [list(x) for x in ARCH_BOXES]}

    if kind == "shelf":
        dx, dz = rot_xz(SHELF_OFF[0], SHELF_OFF[1], rot)
        add(prefix + "Panel", "Shelf", (wp[0] + dx, wp[1], wp[2] + dz),
            (rot + SHELF_ROT_ADD) % 360)
        return
    dx, dz = rot_xz(PANEL_OFF[0], PANEL_OFF[1], rot)
    ppos = (wp[0] + dx, wp[1], wp[2] + dz)
    add(prefix + "Panel", "Panel", ppos, rot)
    if kind in ("lock", "exit"):
        add(prefix + "Chains", "Chains", ppos, rot)
        add(prefix + "Padlock", "Padlock", ppos, rot)


def gen_props():
    inst.insert(0, {"id": "floor", "model": "floor", "texture": ["groundTex"],
                    "translate": [0.0, 0.0, 0.0], "scale": [100.0, 100.0, 100.0]})
    _ids.add("floor")
    e = {"id": "handTorch", "model": "dungeonTorchHeld", "texture": ["dungeonTorchTex"],
         "translate": [round(v, 3) for v in HAND_TORCH[1]],
         "eulerAngles": HAND_TORCH[2], "scale": [1.0, 1.0, 1.0]}
    _ids.add("handTorch")
    inst.append(e)

    for id, pos, sc in BARRELS:
        add(id, "Barrel", pos, None, sc)
    for id, i, j, side, along in TORCHES:
        assert (i, j) in UNION, "%s: cella (%d,%d) non esiste" % (id, i, j)
        assert side in boundary_sides(i, j) or (i, j, side) in PORTALS_BY_CELL, \
            "%s: (%d,%d) lato %s non e' un muro (bordo interno: torcia nel vuoto)" \
            % (id, i, j, side)
        torch_on(id, i, j, side, along)
    for id, key, pos, sc in PICKUPS:
        add(id, key, pos, None, sc)

    for id, pos in GHOSTS:
        ghosts.append({"id": id, "model": "ghost", "texture": ["ghostTex"],
                       "translate": [round(v, 3) for v in pos]})


# --- utilita' --------------------------------------------------------------
PORTALS_BY_CELL = {(i, j, s) for ((i, j), s) in PORTALS}
_AREA_LOOKUP = {}
for _pref, _cells in AREAS.items():
    for _c in _cells:
        _AREA_LOOKUP[_c] = _pref


def area_of(i, j):
    return _AREA_LOOKUP[(i, j)]


def boundary_sides(i, j):
    return {s for s, (di, dj) in NEIGH.items() if (i + di, j + dj) not in UNION}


# --- scrittura ------------------------------------------------------------
def splice_array(text, marker, close, depth, entries):
    lines = text.split("\n")
    s = next(k for k, l in enumerate(lines) if marker in l)
    e = next(k for k in range(s + 1, len(lines)) if lines[k].strip() == close)
    pad = "\t" * depth
    block = [pad + json.dumps(x, ensure_ascii=False) + "," for x in entries]
    block[-1] = block[-1].rstrip(",")
    return "\n".join(lines[:s + 1] + block + lines[e:])


def main():
    gen_structure()
    gen_props()

    text = open(SCENE, encoding="utf-8").read()
    text = splice_array(text, '"technique": "CookTorrance"', "]},", 3, inst)
    text = splice_array(text, '"technique": "Spectral"', "]}", 3, ghosts)
    open(SCENE, "w", encoding="utf-8", newline="\n").write(text)
    json.load(open(SCENE, encoding="utf-8"))            # deve restare valido

    with open(COLLIDERS, "w", encoding="utf-8", newline="\n") as f:
        f.write("{\n")
        keys = sorted(colliders)
        for n, k in enumerate(keys):
            boxes = ",\n".join("\t\t\t[%s]" % ", ".join("%.3f" % v for v in b)
                               for b in colliders[k]["boxes"])
            f.write('\t"%s": {\n\t\t"boxes": [\n%s\n\t\t]\n\t}%s\n'
                    % (k, boxes, "," if n < len(keys) - 1 else ""))
        f.write("}\n")

    n_ceil = sum(1 for e in inst if e["model"] == "dungeonCeiling")
    n_wall = sum(1 for e in inst if e["model"] in
                 ("dungeonWall", "dungeonCorner", "dungeonDoorWall"))
    print("celle              : %d" % len(UNION))
    print("muri (+angoli+archi): %d" % n_wall)
    print("soffitti           : %d" % n_ceil)
    print("istanze CookTorrance: %d" % len(inst))
    print("fantasmi            : %d" % len(ghosts))
    print("collider autorati   : %d" % len(colliders))
    print("SPAWN camPos = (%.2f, %.2f, %.2f)  yaw 0" % SPAWN)


if __name__ == "__main__":
    main()
