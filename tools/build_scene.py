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

# --- kit "Dungeon" del professore (.mgcg, una texture condivisa) --------------
# nome logico -> (file senza estensione, collide?)
DK = {
    "Library03": ("library_03", True), "Library04": ("library_04", True),
    "WeaponRack": ("weaponrack", True),
    "Cupboard01": ("cupboard_01", True), "Cupboard02": ("cupboard_02", True),
    "TableA": ("table_01", True), "TableB": ("table_02", True),
    "Desk": ("table_desk", True), "AlchTable": ("table_alch", True),
    "Lectern": ("table_lectern", True),
    "Box01": ("box_01", True), "Box02": ("box_02", True), "Box03": ("box_03", True),
    "Chest": ("chest_01", True),
    "Rubble01": ("rubble_01", False), "Rubble02": ("rubble_02", False),
    "Rubble03": ("rubble_03", False), "Masonry": ("masonry_01", False),
    "Urn01": ("urn_01", True), "Urn02": ("urn_02", True),
    "Cauldron": ("cauldron", True), "BannerCloth": ("banner_cloth", False),
    "Jug": ("jug_01", False), "Log": ("log_01", True),
    "Skeleton": ("bones_01", False),           # scheletro in T-pose: va A MURO
    "Coffin01": ("coffin_01", True), "Coffin02": ("coffin_02", True),
    "Raven": ("raven", True), "Gravestone": ("gravestone", True),
    "Crystals": ("crystals", True),
    "Statue01": ("statue_01", True), "Statue02": ("statue_02", True),
    "CrossWood": ("cross_wood", False),
    "RackTorture": ("rack_torture", True), "CageHang": ("cage_hang", True),
    "ChairA": ("chair_a", True), "ChairB": ("chair_b", True), "Throne": ("throne", True),
}
for _k in DK:
    KIT[_k] = ("dk" + _k, "dungeonKitTex")

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
# tessera; along = offset lungo il muro dal suo spigolo (0..7.2). Ogni (cella,lato)
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
    ("hbKeyIron",   "Key", (13.4,  1.28, 14.3),  0.0032),   # hub, posata dentro un piatto sul tavolo
    ("raBook",      "Book", (14.4,  1.22, -25.2), 1.0),     # room A, sullo scrittoio
    ("rbKeyBronze", "Key", (-41.0, 0.05, -11.0), 0.0032),   # room B, a terra nell'angolo NO buio
    ("rcKeyGold",   "Key", (18.0,  1.12, 50.4),  0.0032),   # room C, sopra il barile rcStep3
]

GHOSTS = [
    ("ghost",  (7.0, 2.2, 7.0)),      # hub (lontano dal tavolo centrale)
    ("ghost2", (-30.0, 2.2, 5.0)),    # room B
    ("ghost3", (20.0, 2.2, 47.0)),    # room C
]

# Superficie dei piani: TABLE scala 0.6, translate y 0.26 -> il ripiano sta a
# y ~= 1.20 e la sua impronta e' x +-1.36, z +-0.79. Quindi i piatti/candele/
# chiavi sopra vanno a y 1.20 ENTRO x +-1.1, z +-0.55 dal centro del tavolo,
# se no restano per aria oltre il bordo.
TTOP = 1.20
BTOP = 1.09       # cima di un barile (scala 0.48): ci si appoggia una candela

# arredi: (id, chiave_kit, x, y, z, rot, scala)  -- scala None = SCALES.get o 1.0
PROPS = [
    # ============================ HUB: sala del banchetto ==================
    ("hbCarpet",  "Carpet", 14.4, 0.03, 14.4, 0, 1.0),
    ("hbTable",   "Table",  14.4, 0.26, 14.4, 0, None),
    ("hbChairW",  "Chair",  12.5, 0.02, 14.4, 90, None),
    ("hbChairE",  "Chair",  16.3, 0.02, 14.4, -90, None),
    ("hbChairN",  "Chair",  14.4, 0.02, 12.7, 0, None),
    ("hbChairS",  "Chair",  14.4, 0.02, 16.1, 180, None),
    ("hbChairNW", "Chair",  12.8, 0.02, 12.9, 45, None),
    ("hbPlate1",  "Plate",  13.4, TTOP, 14.3, 10, None),
    ("hbPlate2",  "Plate",  15.4, TTOP, 14.5, 190, None),
    ("hbPlate3",  "Plate",  14.4, TTOP, 13.95, 90, None),
    ("hbPlate4",  "Plate",  14.4, TTOP, 14.85, 270, None),
    ("hbCandleC", "Candle", 14.4, TTOP, 14.4, 0, None),
    ("hbBarNW1",  "Barrel", 5.2,  0.02, 3.0, 0, None),
    ("hbBarNW2",  "Barrel", 6.5,  0.02, 2.9, 22, None),
    ("hbBarNW3",  "Barrel", 5.4,  0.02, 4.4, -12, None),
    ("hbSkullNW", "Skull",  6.7,  0.02, 4.7, -30, None),
    ("hbBarNE1",  "Barrel", 25.7, 0.02, 3.0, -25, None),
    ("hbBarNE2",  "Barrel", 24.3, 0.02, 3.3, 10, None),
    ("hbSkullNE", "Skull",  25.2, 0.02, 4.6, 15, None),
    ("hbBarSW1",  "Barrel", 5.4,  0.02, 25.7, 15, None),
    ("hbBarSW2",  "Barrel", 6.7,  0.02, 25.9, -20, None),
    ("hbSkullSW", "Skull",  5.6,  0.02, 24.3, 60, None),
    ("hbBarSE1",  "Barrel", 25.7, 0.02, 24.8, 0, None),
    ("hbBarSE2",  "Barrel", 25.5, 0.02, 26.3, 30, None),
    ("hbBarSE3",  "Barrel", 24.1, 0.02, 25.6, -15, None),
    ("hbSkullSE", "Skull",  24.3, 0.02, 24.4, 200, None),
    ("hbBarWmid", "Barrel", 2.6,  0.02, 16.5, 8, None),
    ("hbBarWmid2","Barrel", 2.6,  0.02, 15.0, -14, None),
    ("hbSkullWmid","Skull", 3.6,  0.02, 17.6, 40, None),
    ("hbSkullEmid","Skull", 26.1, 0.02, 19.0, 40, None),
    ("hbChairStray","Chair", 8.4, 0.02, 21.6, 300, None),
    ("hbSkullFloor","Skull", 9.6, 0.02, 8.0, 120, None),
    # ============================ ROOM A: lo studio =======================
    ("raCarpet",  "Carpet", 14.4, 0.03, -25.2, 0, 1.0),
    ("raTable",   "Table",  14.4, 0.26, -25.2, 0, None),
    ("raChair",   "Chair",  14.4, 0.02, -22.7, 180, None),
    ("raChair2",  "Chair",  16.6, 0.02, -25.2, -90, None),
    ("raCandleL", "Candle", 13.5, TTOP, -25.4, 0, None),
    ("raCandleR", "Candle", 15.3, TTOP, -25.0, 0, None),
    ("raPlate",   "Plate",  14.4, TTOP, -24.75, 0, None),
    ("raBar1",    "Barrel", 3.1,  0.02, -33.4, 0, None),
    ("raBar2",    "Barrel", 4.5,  0.02, -33.2, 18, None),
    ("raBar3",    "Barrel", 25.6, 0.02, -33.0, -20, None),
    ("raBar4",    "Barrel", 3.3,  0.02, -16.6, 15, None),
    ("raBar5",    "Barrel", 25.6, 0.02, -16.8, -30, None),
    ("raSkull1",  "Skull",  4.7,  0.02, -33.5, 25, None),
    ("raSkull2",  "Skull",  24.4, 0.02, -17.4, 40, None),
    ("raSkull3",  "Skull",  14.4, 0.02, -33.6, -10, None),
    ("raChairC",  "Chair",  6.5,  0.02, -18.5, 220, None),
    # ============================ ROOM B: cupa ============================
    ("rbCarpet",  "Carpet", -24.0, 0.03, 2.0, 90, 1.0),
    ("rbChair",   "Chair",  -19.5, 0.02, 9.5, 200, None),
    ("rbChair2",  "Chair",  -17.5, 0.02, -9.0, 20, None),
    ("rbBar1",    "Barrel", -16.4, 0.02, 11.4, 20, None),
    ("rbBar2",    "Barrel", -16.8, 0.02, 9.8, 0, None),
    ("rbBar3",    "Barrel", -15.6, 0.02, 10.6, -30, None),
    ("rbBar4",    "Barrel", -16.6, 0.02, -11.2, -15, None),
    ("rbBar5",    "Barrel", -16.6, 0.02, 1.5, 5, None),
    ("rbCandleB", "Candle", -16.4, BTOP, 11.4, 0, None),
    ("rbSkull1",  "Skull",  -18.2, 0.02, 12.4, 30, None),
    ("rbSkull2",  "Skull",  -25.0, 0.02, 12.6, -40, None),
    ("rbSkull3",  "Skull",  -30.0, 0.02, -11.5, 70, None),
    # ============================ ROOM C: puzzle ==========================
    ("rcBarA",    "Barrel", 3.4,  0.02, 46.0, 0, None),
    ("rcBarA2",   "Barrel", 4.8,  0.02, 46.2, 20, None),
    ("rcBarB",    "Barrel", 25.0, 0.02, 46.0, -20, None),
    ("rcBarC",    "Barrel", 25.0, 0.02, 55.2, 30, None),
    ("rcBarD",    "Barrel", 3.4,  0.02, 55.2, 0, None),
    ("rcBarD2",   "Barrel", 4.8,  0.02, 55.0, -15, None),
    ("rcCandleB", "Candle", 3.4,  BTOP, 46.0, 0, None),
    ("rcChair",   "Chair",  6.4,  0.02, 53.6, 130, None),
    ("rcSkull1",  "Skull",  5.6,  0.02, 47.6, 25, None),
    ("rcSkull2",  "Skull",  23.2, 0.02, 54.2, -35, None),
    ("rcSkull3",  "Skull",  24.0, 0.02, 47.0, 60, None),
    # ============================ corridoi ================================
    ("iaBar",     "Barrel", -25.4, 0.02, 22.6, 10, None),
    ("iaBar2",    "Barrel", -24.0, 0.02, 22.8, -20, None),
    ("iaSkull",   "Skull",  -21.8, 0.02, 27.8, -25, None),
    ("b1Bar",     "Barrel", 8.6,  0.02, -12.2, 0, None),
    ("b1Bar2",    "Barrel", 12.9, 0.02, -9.0, 15, None),
    ("b1Skull",   "Skull",  8.3,  0.02, -8.5, 20, None),
    ("b2Bar",     "Barrel", -11.0, 0.02, 4.8, -15, None),
    ("b2Skull",   "Skull",  -4.5, 0.02, 5.0, 40, None),
    ("b3Bar",     "Barrel", 8.6,  0.02, 34.0, 0, None),
    ("b3Bar2",    "Barrel", 12.9, 0.02, 40.0, -10, None),
    ("b3Skull",   "Skull",  8.4,  0.02, 38.5, 40, None),
    # --- HUB: seconda tavolata in un angolo, per non lasciare vuoto il resto ---
    ("hbTable2",  "Table",  21.0, 0.26, 7.6, 25, None),
    ("hbT2ChA",   "Chair",  19.2, 0.02, 7.0, 55, None),
    ("hbT2ChB",   "Chair",  22.6, 0.02, 8.6, -125, None),
    ("hbT2Plate", "Plate",  21.0, TTOP, 7.6, 0, None),
    ("hbT2Cndl",  "Candle", 20.3, TTOP, 7.2, 0, None),
    ("hbStk1a",   "Barrel", 2.7,  0.02, 9.2, 0, None),
    ("hbStk1b",   "Barrel", 2.7,  1.03, 9.2, 40, None),
    ("hbStk2a",   "Barrel", 26.3, 0.02, 22.0, 10, None),
    ("hbStk2b",   "Barrel", 26.3, 1.03, 22.0, -35, None),
    ("hbStk3a",   "Barrel", 20.0, 0.02, 25.8, 0, None),
    ("hbStk3b",   "Barrel", 20.0, 1.03, 25.8, 60, None),
]

# --- extra generati: file di barili lungo i muri, teschi sparsi sul pavimento ---
def _line(prefix, key, p0, p1, n, y=0.02, scale=None):
    out = []
    keys = key if isinstance(key, (list, tuple)) else [key]
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0.5
        x = round(p0[0] + (p1[0] - p0[0]) * t, 2)
        z = round(p0[1] + (p1[1] - p0[1]) * t, 2)
        out.append(("%s%d" % (prefix, k), keys[k % len(keys)],
                    x, y, z, (k * 43) % 360, scale))
    return out

BONES = ["Bones01", "Bones02", "Bones03"]     # mucchi d'ossa del kit Dungeon

PROPS += _line("hbBwW", "Barrel", (2.7, 8.4), (2.7, 20.6), 5)
PROPS += _line("hbBwE1", "Barrel", (26.3, 1.6), (26.3, 6.6), 3)
PROPS += _line("hbBwE2", "Barrel", (26.3, 15.4), (26.3, 24.6), 4)
PROPS += _line("hbBwN", "Barrel", (16.0, 2.7), (26.8, 2.7), 4)
PROPS += _line("hbBwS", "Barrel", (16.0, 26.3), (24.4, 26.3), 3)
PROPS += _line("hbSkF", "Skull", (8.5, 8.0), (12.5, 22.5), 3)
PROPS += _line("hbSkF2", "Skull", (18.5, 5.0), (21.5, 20.0), 3)
PROPS += _line("raBwN", "Barrel", (7.5, -33.4), (20.5, -33.4), 4)
PROPS += _line("raBwW", "Barrel", (2.9, -29.0), (2.9, -21.0), 3)
PROPS += _line("raSkF", "Skull", (8.0, -20.0), (20.0, -30.0), 3)
PROPS += _line("rbBwN", "Barrel", (-35.5, -12.8), (-28.0, -12.8), 3)
PROPS += _line("rbBwW", "Barrel", (-41.0, -4.0), (-41.0, 7.0), 3)
PROPS += _line("rbSkF", "Skull", (-32.0, -6.0), (-22.0, 8.0), 3)
PROPS += _line("rcBwN", "Barrel", (7.5, 44.6), (21.0, 44.6), 4)
PROPS += _line("rcBwS", "Barrel", (7.5, 56.6), (21.0, 56.6), 4)
PROPS += _line("rcSkF", "Skull", (8.0, 47.0), (20.0, 54.0), 3)

# --- kit Dungeon: prima passata di calibrazione (scala/rot da verificare a video)
DK_PROPS = [
    # ROOM A -- lo studio
    ("dkLib1",   "Library03", 2.5,  0.02, -28.0, 0, 1.0),
    ("dkLib2",   "Library04", 2.5,  0.02, -22.0, 0, 1.0),
    ("dkDesk",   "Desk",      14.4, 0.02, -25.2, 0, 1.0),
    ("dkLectern","Lectern",   20.0, 0.02, -19.5, 200, 1.0),
    ("dkCupA",   "Cupboard01", 25.8, 0.02, -21.0, 180, 1.0),
    # HUB
    ("dkStatL",  "Statue01",  26.0, 0.02, 8.0, 180, 1.0),
    ("dkStatR",  "Statue02",  26.0, 0.02, 13.6, 180, 1.0),
    ("dkRaven",  "Raven",     6.5,  0.02, 6.5, 45, 1.0),
    ("dkTblA",   "TableA",    7.5,  0.02, 20.0, 20, 1.0),
    ("dkCauld",  "Cauldron",  21.0, 0.02, 21.5, 0, 1.0),
    ("dkBoxA",   "Box01",     24.2, 0.02, 4.2, 15, 1.0),
    ("dkBoxB",   "Box02",     25.0, 0.02, 5.6, -20, 1.0),
    ("dkChest",  "Chest",     5.5,  0.02, 11.0, 30, 1.0),
    # ROOM B
    ("dkCoffin", "Coffin01",  -30.0, 0.02, 8.0, 20, 1.0),
    ("dkCrystal","Crystals",  -40.5, 0.02, 8.5, 0, 1.0),
    ("dkCupB",   "Cupboard02", -16.5, 0.02, -8.0, 90, 1.0),
    # ROOM C
    ("dkRack",   "RackTorture", 5.5, 0.02, 46.0, 20, 1.0),
    ("dkCage",   "CageHang",  23.0, 0.02, 48.0, 0, 1.0),
    ("dkGrave",  "Gravestone", 24.0, 0.02, 55.0, 15, 1.0),
]

# stendardi a muro: (id, i, j, lato, along) -- come le torce ma appesi in alto
BANNERS = [
    ("hbBanN",  3, 0, "N", A),
    ("hbBanW",  0, 2, "W", A),
    ("hbBanE1", 3, 0, "E", A),
    ("hbBanE2", 3, 3, "E", A),
    ("hbBanS",  3, 3, "S", A),
    ("raBanN",  2, -5, "N", A),
    ("raBanW",  0, -3, "W", A),
    ("raBanE",  3, -3, "E", A),
    ("raBanS1", 0, -3, "S", A),
    ("raBanS2", 3, -3, "S", A),
    ("rbBanS",  -5, 1, "S", A),
    ("rbBanE",  -3, -1, "E", A),
    ("rbBanN",  -5, -2, "N", A),
    ("rcBanW",  0, 8, "W", A),
    ("rcBanN",  2, 6, "N", A),
    ("rcBanE",  3, 8, "E", A),
]

# scheletri in T-pose incatenati al muro (kit Dungeon, bones_01). y = altezza
# a cui pendono; embed li appoggia con la schiena alla pietra.
SKELETONS = [
    ("skB1", -6, -1, "W", 3.6),   # room B, cripta
    ("skB2", -6, 0,  "W", 3.6),
    ("skC1", 3, 6,  "E", 3.6),    # room C, sala della tortura
    ("skC2", 1, 8,  "S", 3.6),
    ("skIA", -2, 3, "N", 3.6),    # corridoio d'ingresso
]
SKELETON_Y = 0.35
SKELETON_EMBED = 0.55
SKELETON_ROT_ADD = -90        # le braccia in T-pose devono stendersi LUNGO il muro
SKELETON_SCALE = 1.9

# camera di spawn: estremita' ovest dell'alley, guarda +X lungo il corridoio
SPAWN = (TS * -4 + 1.4, 1.8, TS * 3 + 3.6)      # (-27.4, 1.8, 25.2)
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


def skeleton_on(id, i, j, side, along=3.6):
    wall_mount(id, "Skeleton", i, j, side, along, SKELETON_Y, SKELETON_EMBED,
               SKELETON_ROT_ADD, SKELETON_SCALE)


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

    for id, i, j, side, along in TORCHES + BANNERS + SKELETONS:
        assert (i, j) in UNION, "%s: cella (%d,%d) non esiste" % (id, i, j)
        assert side in boundary_sides(i, j) or (i, j, side) in PORTALS_BY_CELL, \
            "%s: (%d,%d) lato %s non e' un muro (bordo interno: nel vuoto)" \
            % (id, i, j, side)
    for id, i, j, side, along in TORCHES:
        torch_on(id, i, j, side, along)
    for id, i, j, side, along in BANNERS:
        banner_on(id, i, j, side, along)
    for id, i, j, side, along in SKELETONS:
        skeleton_on(id, i, j, side, along)

    for id, key, x, y, z, rot, sc in PROPS + DK_PROPS:
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


def splice_append(text, marker, close, depth, entries):
    """Inserisce `entries` appena PRIMA della riga di chiusura, tenendo il corpo."""
    if not entries:
        return text
    lines = text.split("\n")
    s = next(k for k, l in enumerate(lines) if marker in l)
    e = next(k for k in range(s + 1, len(lines)) if lines[k].strip() == close)
    if not lines[e - 1].rstrip().endswith(","):
        lines[e - 1] = lines[e - 1].rstrip() + ","
    pad = "\t" * depth
    block = [pad + json.dumps(x, ensure_ascii=False) + "," for x in entries]
    block[-1] = block[-1].rstrip(",")
    return "\n".join(lines[:e] + block + lines[e:])


def main():
    gen_structure()
    gen_props()

    dk_models = [{"id": "dk" + k, "VD": "VDposNormUV",
                  "model": "assets/models/Dungeon/%s.mgcg" % DK[k][0], "format": "MGCG",
                  **({"collider": "AABB"} if DK[k][1] else {})}
                 for k in DK if ("dk" + k) in _used_models]
    dk_tex = ([{"id": "dungeonKitTex",
                "texture": "assets/textures/Dungeon/Textures_Dungeon.png", "format": "C"}]
              if dk_models else [])

    text = open(SCENE, encoding="utf-8").read()
    text = splice_append(text, '"models": [', "],", 2, dk_models)
    text = splice_append(text, '"textures": [', "],", 2, dk_tex)
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
