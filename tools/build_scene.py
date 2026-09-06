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

HUB e ROOM B sono state rimpicciolite da 4x4 a 3x3 celle (il resto della
mappa -- ia/b3/room C -- e' stato traslato di una tessera per restare
attaccato all'hub) e gli arredi da riempimento (barili/teschi a raffica,
le due tavolate extra dell'hub, le file di barili lungo i muri generate
da _line()) sono stati tagliati per tenere la scena piccola e leggibile.

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
    "Carpet":  ("dungeonCarpet",    "dungeonCarpetTex"),
    "Table":   ("dungeonTable",     "dungeonTableTex"),
    "Chair":   ("dungeonChair",     "dungeonChairTex"),
    "Plate":   ("dungeonPlate",     "dungeonPlateTex"),
    "Candle":  ("dungeonCandle",    "dungeonCandleTex"),
    "Skull":   ("dungeonSkull",     "dungeonSkullTex"),
    "Banner":  ("dungeonBanner",    "dungeonBannerTex"),
}

# scala umana per i pezzi del kit modellati troppo grandi (dal vecchio build_scene)
SCALES = {"Table": 0.60, "Chair": 0.86, "Barrel": 0.48, "Skull": 0.46, "Candle": 0.41}

inst = []          # CookTorrance elements
ghosts = []        # Spectral elements
colliders = {}     # colliders.json

_ids = set()
_used_models = set()


def add(id, key, pos, rot=None, scale=None, tex=None):
    assert id not in _ids, "id duplicato: %s" % id
    _ids.add(id)
    model, texture = KIT[key]
    _used_models.add(model)
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
    "ia": [(i, 2) for i in range(-3, 0)],                       # alley intro (SO del hub)
    "hb": [(i, j) for i in range(0, 3) for j in range(0, 3)],   # hub 3x3
    "b1": [(1, -2), (1, -1)],                                   # alley nord
    "ra": [(i, j) for i in range(0, 4) for j in range(-5, -2)], # room A 4x3
    "b2": [(-2, 0), (-1, 0)],                                   # alley ovest
    "rb": [(i, j) for i in range(-5, -2) for j in range(-2, 1)],# room B 3x3
    "b3": [(1, 3), (1, 4)],                                     # alley sud
    "rc": [(i, j) for i in range(0, 4) for j in range(5, 8)],   # room C 4x3
}
UNION = {c for cells in AREAS.values() for c in cells}
assert len(UNION) == sum(len(c) for c in AREAS.values()), "celle sovrapposte fra aree"

# varchi: (cella, lato) del muro-arcata -> (prefisso id, tipo, keyId, label)
#   tipo: "open"  battente semplice, si apre e basta
#         "lock"  battente + catene + lucchetto
#         "exit"  come lock ma si apre verso l'ESTERNO (verso la luce)
#         "shelf" libreria segreta
PORTALS = {
    ((0, 2), "W"): ("iaDoor",  "open",  "",       ""),
    ((1, 0), "N"): ("hbDoorN", "lock",  "iron",   "iron key"),
    ((1, 2), "S"): ("hbDoorS", "lock",  "bronze", "bronze key"),
    ((2, 1), "E"): ("hbDoorE", "exit",  "gold",   "gold key"),
    ((0, 0), "W"): ("hbShelf", "shelf", "book",   "old book"),
}

# barili da scavalcare (callback: intro insegna il salto, room C lo richiede)
BARRELS = [
    ("iaJump",  (-14.4, Y, 18.0), 0.48),      # ostacolo tutorial, meta' alley
    ("rcStep1", (10.8,  Y, 43.2), 0.48),      # puzzle room C: appoggi verso la chiave
    ("rcStep2", (14.4,  Y, 44.8), 0.48),
    ("rcStep3", (18.0,  Y, 43.2), 0.48),
]

# torce a muro: (id, cella_i, cella_j, lato, along)  -- il muro su `lato` di quella
# tessera; along = offset lungo il muro dal suo spigolo (0..7.2). Ogni (cella,lato)
# DEVE essere un bordo con muro (o un portale): validato in gen_props().
A = TORCH_ALONG
TORCHES = [
    ("iaTorch",   -3, 2, "S", A),              # unica luce dell'alley, fioca
    ("hbTorchN1",  0, 0, "N", A),
    ("hbTorchN2",  2, 0, "N", A),
    ("hbTorchW",   0, 1, "W", A),
    ("hbTorchE",   2, 2, "E", A),
    ("raTorchW",   0, -4, "W", A),
    ("raTorchE",   3, -4, "E", A),
    ("raTorchN1",  1, -5, "N", A),
    ("rbTorch",   -4, 0, "S", A),              # una sola, presso l'ingresso: l'angolo NO resta buio
    ("rcTorchW",   0, 6, "W", A),
    ("rcTorchE",   3, 6, "E", A),
]

# raccoglibili
PICKUPS = [
    ("hbKeyIron",   "Key", (9.6,   1.28, 10.7),  0.0032),   # hub, posata dentro un piatto sul tavolo
    ("raBook",      "Book", (14.4,  1.22, -25.2), 1.0),     # room A, sullo scrittoio
    ("rbKeyBronze", "Key", (-33.0, 0.05, -11.0), 0.0032),   # room B, a terra nell'angolo NO buio
    ("rcKeyGold",   "Key", (18.0,  1.12, 43.2),  0.0032),   # room C, sopra il barile rcStep3
]

GHOSTS = [
    ("ghost",  (3.0, 2.2, 3.0)),      # hub (lontano dal tavolo centrale)
    ("ghost2", (-30.0, 2.2, 5.0)),    # room B
    ("ghost3", (20.0, 2.2, 39.8)),    # room C
]

# Superficie dei piani: TABLE scala 0.6, translate y 0.26 -> il ripiano sta a
# y ~= 1.20 e la sua impronta e' x +-1.36, z +-0.79. Quindi i piatti/candele/
# chiavi sopra vanno a y 1.20 ENTRO x +-1.1, z +-0.55 dal centro del tavolo,
# se no restano per aria oltre il bordo.
TTOP = 1.20
BTOP = 1.09       # cima di un barile (scala 0.48): ci si appoggia una candela

# arredi: (id, chiave_kit, x, y, z, rot, scala)  -- scala None = SCALES.get o 1.0
PROPS = [
    # ============================ HUB: sala del banchetto (3x3, rimpicciolita) =
    ("hbCarpet",  "Carpet", 10.8, 0.03, 10.8, 0, 1.0),
    ("hbTable",   "Table",  10.8, 0.26, 10.8, 0, None),
    ("hbChairW",  "Chair",  8.9,  0.02, 10.8, 90, None),
    ("hbChairE",  "Chair",  12.7, 0.02, 10.8, -90, None),
    ("hbChairN",  "Chair",  10.8, 0.02, 9.1,  0, None),
    ("hbChairS",  "Chair",  10.8, 0.02, 12.5, 180, None),
    ("hbPlate1",  "Plate",  10.8, TTOP, 10.4, 90, None),
    ("hbCandleC", "Candle", 10.8, TTOP, 10.8, 0, None),
    ("hbBarNW",   "Barrel", 4.0,  0.02, 4.5, 15, None),
    ("hbSkullNW", "Skull",  5.4,  0.02, 5.4, -30, None),
    ("hbBarSE",   "Barrel", 18.6, 0.02, 18.6, -20, None),
    ("hbSkullSE", "Skull",  17.2, 0.02, 17.4, 40, None),
    # ============================ ROOM A: lo studio (invariata) ============
    ("raCarpet",  "Carpet", 14.4, 0.03, -25.2, 0, 1.0),
    ("raTable",   "Table",  14.4, 0.26, -25.2, 0, None),
    ("raChair",   "Chair",  14.4, 0.02, -22.7, 180, None),
    ("raChair2",  "Chair",  16.6, 0.02, -25.2, -90, None),
    ("raCandleL", "Candle", 13.5, TTOP, -25.4, 0, None),
    ("raCandleR", "Candle", 15.3, TTOP, -25.0, 0, None),
    ("raPlate",   "Plate",  14.4, TTOP, -24.75, 0, None),
    ("raBar1",    "Barrel", 3.1,  0.02, -33.4, 0, None),
    ("raBar4",    "Barrel", 25.6, 0.02, -16.8, -30, None),
    ("raSkull1",  "Skull",  4.7,  0.02, -33.5, 25, None),
    ("raChairC",  "Chair",  6.5,  0.02, -18.5, 220, None),
    # ============================ ROOM B: cupa (3x3, rimpicciolita) ========
    ("rbCarpet",  "Carpet", -24.0, 0.03, 2.0, 90, 1.0),
    ("rbChair",   "Chair",  -19.5, 0.02, -6.0, 200, None),
    ("rbChair2",  "Chair",  -17.5, 0.02, -4.5, 20, None),
    ("rbBar1",    "Barrel", -16.4, 0.02, -2.0, 20, None),
    ("rbCandleB", "Candle", -16.4, BTOP, -2.0, 0, None),
    ("rbSkull1",  "Skull",  -18.2, 0.02, -1.0, 30, None),
    # ============================ ROOM C: puzzle (spostata) ================
    ("rcBarA",    "Barrel", 3.4,  0.02, 38.8, 0, None),
    ("rcBarC",    "Barrel", 25.0, 0.02, 48.0, 30, None),
    ("rcCandleB", "Candle", 3.4,  BTOP, 38.8, 0, None),
    ("rcChair",   "Chair",  6.4,  0.02, 46.4, 130, None),
    ("rcSkull1",  "Skull",  5.6,  0.02, 40.4, 25, None),
    # ============================ corridoi ================================
    ("iaBar",     "Barrel", -18.2, 0.02, 15.4, 10, None),
    ("iaSkull",   "Skull",  -14.6, 0.02, 20.6, -25, None),
    ("b1Bar",     "Barrel", 8.6,  0.02, -12.2, 0, None),
    ("b2Bar",     "Barrel", -11.0, 0.02, 4.8, -15, None),
    ("b3Bar",     "Barrel", 8.6,  0.02, 26.8, 0, None),
]

# stendardi a muro: (id, i, j, lato, along) -- come le torce ma appesi in alto
BANNERS = [
    ("raBanN",  2, -5, "N", A),
    ("raBanW",  0, -3, "W", A),
    ("raBanE",  3, -3, "E", A),
    ("raBanS1", 0, -3, "S", A),
    ("raBanS2", 3, -3, "S", A),
    ("rbBanS",  -5, 0, "S", A),
    ("rbBanE",  -3, -1, "E", A),
    ("rbBanN",  -5, -2, "N", A),
    ("rcBanW",  0, 7, "W", A),
    ("rcBanN",  2, 5, "N", A),
    ("rcBanE",  3, 7, "E", A),
]

# camera di spawn: estremita' ovest dell'alley, guarda +X lungo il corridoio
SPAWN = (TS * -3 + 1.4, 1.8, TS * 2 + 3.6)      # (-20.2, 1.8, 18.0)
HAND_TORCH = ("handTorch", (-28.8, -0.12, 0.0), [0.0, 35.0, -90.0])


def wall_mount(id, key, i, j, side, along, y, embed, rot_add=0, scale=None):
    """Appende un pezzo alla faccia interna del muro su `side`, spinto `embed`
    dentro la pietra."""
    rot = (WALL_ROT[side] + rot_add) % 360
    if side == "E":
        pos = (TS * (i + 1) - 1.242 + embed, y, TS * j + along)
    elif side == "W":
        pos = (TS * i + 1.242 - embed, y, TS * j + along)
    elif side == "N":
        pos = (TS * i + along, y, TS * j + 1.242 - embed)
    else:  # S
        pos = (TS * i + along, y, TS * (j + 1) - 1.242 + embed)
    add(id, key, pos, rot, scale)


BANNER_EMBED = 0.15


def torch_on(id, i, j, side, along=3.6, y=3.5):
    wall_mount(id, "Torch", i, j, side, along, y, TORCH_EMBED)


def banner_on(id, i, j, side, along=3.6, y=Y):
    wall_mount(id, "Banner", i, j, side, along, y, BANNER_EMBED)


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
        assert in_union(pos[0], pos[2]), "%s fuori dalle stanze: %s" % (id, pos)
        assert door_clear(pos[0], pos[2]), "%s davanti a un varco: %s" % (id, pos)
        add(id, "Barrel", pos, None, sc)

    for id, i, j, side, along in TORCHES + BANNERS:
        assert (i, j) in UNION, "%s: cella (%d,%d) non esiste" % (id, i, j)
        assert side in boundary_sides(i, j) or (i, j, side) in PORTALS_BY_CELL, \
            "%s: (%d,%d) lato %s non e' un muro (bordo interno: nel vuoto)" \
            % (id, i, j, side)
    for id, i, j, side, along in TORCHES:
        torch_on(id, i, j, side, along)
    for id, i, j, side, along in BANNERS:
        banner_on(id, i, j, side, along)

    for id, key, x, y, z, rot, sc in PROPS:
        assert in_union(x, z), "%s fuori dalle stanze: (%.1f, %.1f)" % (id, x, z)
        assert key == "Carpet" or door_clear(x, z), \
            "%s davanti a un varco: (%.1f, %.1f)" % (id, x, z)
        scale = sc if sc is not None else SCALES.get(key, 1.0)
        add(id, key, (x, y, z), rot if rot else None, scale)

    for id, key, pos, sc in PICKUPS:
        add(id, key, pos, None, sc)

    for id, pos in GHOSTS:
        ghosts.append({"id": id, "model": "ghost", "texture": ["ghostTex"],
                       "translate": [round(v, 3) for v in pos]})


# --- utilita' --------------------------------------------------------------
PORTALS_BY_CELL = {(i, j, s) for ((i, j), s) in PORTALS}

# centro di ogni varco, per tenere sgombra la soglia (tranne i tappeti)
def _door_center(i, j, side):
    if side == "E":
        return (TS * (i + 1), TS * j + 3.6)
    if side == "W":
        return (TS * i, TS * j + 3.6)
    if side == "N":
        return (TS * i + 3.6, TS * j)
    return (TS * i + 3.6, TS * (j + 1))

DOOR_CENTERS = [_door_center(i, j, s) for ((i, j), s) in PORTALS]


def door_clear(x, z, r=3.6):
    return all((x - cx) ** 2 + (z - cz) ** 2 > r * r for cx, cz in DOOR_CENTERS)
_AREA_LOOKUP = {}
for _pref, _cells in AREAS.items():
    for _c in _cells:
        _AREA_LOOKUP[_c] = _pref


def area_of(i, j):
    return _AREA_LOOKUP[(i, j)]


def boundary_sides(i, j):
    return {s for s, (di, dj) in NEIGH.items() if (i + di, j + dj) not in UNION}


def in_union(x, z, slack=0.6):
    """Vero se (x,z) sta dentro una tessera calpestabile (rete anti-vuoto, non
    un vero controllo di contenimento contro i muri)."""
    return any(TS * i - slack <= x <= TS * (i + 1) + slack
               and TS * j - slack <= z <= TS * (j + 1) + slack
               for (i, j) in UNION)


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
