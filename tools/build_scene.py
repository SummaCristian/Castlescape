#!/usr/bin/env python3
"""
Genera il blocco di istanze del dungeon Dracula dentro scene.json.

Il kit e' modulare su una griglia di 7.2 con muri spessi 1.24 appoggiati sul
bordo interno della piastrella. Le convenzioni sotto sono ricavate dalla
geometria reale dei modelli, non assunte:

  piastrella (i,j) -> x in [X0+7.2i, X0+7.2(i+1)], z in [Z0+7.2j, Z0+7.2(j+1)]

  SM_StoneFloor_01    occupa x [-7.2,0], z [0,7.2]      rispetto al translate
  SM_WallStraight_01  occupa x [-1.24,0], z [0,7.2]     -> lastra sul lato est
  SM_WallCorner_01    occupa la piastrella x [-7.2,0], z [-7.2,0] e fornisce
                      insieme il lato est e il lato sud di quella piastrella

Da cui le rotazioni: 0 = est, 90 = nord, 180 = ovest, 270 = sud.

Lo script controlla che ogni bordo esterno delle stanze sia coperto una volta
sola e che ogni prop stia dentro l'area calpestabile, poi riscrive in scene.json
tutte le istanze il cui modello inizia per "dungeon".

Uso:  python build_scene.py
"""

import glob
import json
import os

import numpy as np

TS = 7.2      # lato della piastrella
Y = 0.02      # alzata comune, per non litigare col piano grande a y=0
X0, Z0 = -21.6, -46.8

# La griglia sopra e' costruita a nord del castello; SHIFT trasla l'intera stanza
# (struttura, arredi e bounding degli interni) al momento della scrittura. Qui la
# porta in campo aperto a sud-ovest, a ~14 unita' dallo spawn (0,1,5): ci si gira
# di ~180 gradi e la si vede. A ovest di x=0 per non accavallarsi alla strada.
SHIFT = (-16.0, 66.0)

SCENE = "skeleton/source/assets/scenes/scene.json"
MODELS = "skeleton/source/assets/models/Dracula"

FILE = {
    "Floor": "SM_StoneFloor_01", "Wall": "SM_WallStraight_01",
    "Corner": "SM_WallCorner_01", "Door": "SM_WallDoor_01",
    "Carpet": "SM_Carpet_01", "Table": "SM_Table_01", "Chair": "SM_Chair_01",
    "Plate": "SM_PlateAndCutlery_01", "Candle": "SM_Candle_01",
    "Skull": "SM_Skull_01", "Barrel": "SM_Barrel_01", "Torch": "SM_Torch_01",
    "Banner": "SM_CastleBanners_01",
}
STRUCTURAL = ("Floor", "Wall", "Corner", "Door")

inst = []
edges = {}


def add(id, key, pos, rot=None, scale=None):
    e = {"id": id, "model": "dungeon" + key, "texture": ["dungeon" + key + "Tex"],
         "translate": [round(pos[0] + SHIFT[0], 2), round(pos[1], 2),
                       round(pos[2] + SHIFT[1], 2)]}
    if rot is not None:
        e["eulerAngles"] = [0.0, float(rot), 0.0]
    if scale is not None:
        e["scale"] = [float(scale)] * 3
    inst.append(e)


def floor(id, i, j):
    add(id, "Floor", (X0 + TS * (i + 1), Y, Z0 + TS * j))


def claim(i, j, side):
    key = (i, j, side)
    assert key not in edges, "bordo coperto due volte: %s" % (key,)
    edges[key] = True


def wall(id, i, j, side, key="Wall"):
    claim(i, j, side)
    if side == "E":
        pos, rot = (X0 + TS * (i + 1), Y, Z0 + TS * j), None
    elif side == "N":
        pos, rot = (X0 + TS * i, Y, Z0 + TS * j), 90
    elif side == "W":
        pos, rot = (X0 + TS * i, Y, Z0 + TS * (j + 1)), 180
    else:
        pos, rot = (X0 + TS * (i + 1), Y, Z0 + TS * (j + 1)), 270
    add(id, key, pos, rot)


CORNER = {"SE": (None, ("E", "S")), "NE": (90, ("N", "E")),
          "NW": (180, ("W", "N")), "SW": (270, ("S", "W"))}


def corner(id, i, j, kind):
    rot, sides = CORNER[kind]
    for s in sides:
        claim(i, j, s)
    if kind == "SE":
        pos = (X0 + TS * (i + 1), Y, Z0 + TS * (j + 1))
    elif kind == "NE":
        pos = (X0 + TS * (i + 1), Y, Z0 + TS * j)
    elif kind == "NW":
        pos = (X0 + TS * i, Y, Z0 + TS * j)
    else:
        pos = (X0 + TS * i, Y, Z0 + TS * (j + 1))
    add(id, "Corner", pos, rot)


# ---------------------------------------------------------------- grande sala
HALL = {(i, j) for i in range(3) for j in range(3)}
for i, j in sorted(HALL):
    floor("dhFloor%d%d" % (i, j), i, j)
corner("dhCornNW", 0, 0, "NW")
corner("dhCornNE", 2, 0, "NE")
corner("dhCornSW", 0, 2, "SW")
corner("dhCornSE", 2, 2, "SE")
wall("dhWallN", 1, 0, "N")
wall("dhWallS", 1, 2, "S")
# SM_WallDoor_01 e' un muro pieno con una porta chiusa modellata sopra, non un
# varco: verificato proiettando i triangoli sul piano del muro, copre tutta la
# superficie come SM_WallStraight_01. Va quindi su un muro esterno, come
# ingresso decorativo, e il collegamento fra le stanze si fa lasciando aperto
# il bordo condiviso.
wall("dhDoor", 0, 1, "W", "Door")

# ---------------------------------------------------------------- anticamera
CHAM = {(i, j) for i in (3, 4) for j in (0, 1)}
for i, j in sorted(CHAM):
    floor("dcFloor%d%d" % (i, j), i, j)
corner("dcCornNE", 4, 0, "NE")
corner("dcCornSE", 4, 1, "SE")
wall("dcWallN", 3, 0, "N")
wall("dcWallS", 3, 1, "S")

# il lato ovest dell'anticamera e' il lato est della sala: l'angolo NE della
# sala copre j=0, la porta copre j=1, quindi non servono pezzi dedicati.

ROOMS = HALL | CHAM
NEIGH = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
missing = [(i, j, s) for (i, j) in ROOMS for s, (di, dj) in NEIGH.items()
           if (i + di, j + dj) not in ROOMS and (i, j, s) not in edges]
assert not missing, "bordi esterni scoperti: %s" % missing

# ---------------------------------------------------------------- arredamento
# interni calpestabili, ricavati dai bordi meno lo spessore del muro, gia' shiftati
INT = {"dh": (-20.36 + SHIFT[0], -1.24 + SHIFT[0], -45.56 + SHIFT[1], -26.44 + SHIFT[1]),
       "dc": (0.0 + SHIFT[0], 13.16 + SHIFT[0], -45.56 + SHIFT[1], -33.64 + SHIFT[1])}

# Il kit Dracula e' modellato a una scala molto piu' grande dell'omino
# (EYE_HEIGHT del player e' 1.8): tavolo, sedia, barile e teschio grezzi
# arrivano quasi o oltre l'altezza occhi. Questi fattori li riportano a
# proporzioni umane realistiche (tavolo ~1.0m, sedia ~0.95m, barile ~0.9m,
# teschio ~0.22m, candela ~0.3m); il resto del kit (muri, porta, torcia,
# stendardo, tappeto, piatto) e' gia' in scala e resta a 1.0.
SCALE = {"Table": 0.60, "Chair": 0.86, "Barrel": 0.48, "Skull": 0.46, "Candle": 0.41}

PROPS = [
    # grande sala: tavolata al centro sul tappeto, torce e stendardi alle pareti
    ("dhCarpet",  "Carpet", -10.80, Y + 0.01,   -36.0, None),
    ("dhTable",   "Table",  -10.80, Y + 0.24,   -36.0, None),
    ("dhChairW",  "Chair",  -14.60, Y,          -36.0, 90),
    ("dhChairE",  "Chair",   -7.00, Y,          -36.0, -90),
    ("dhPlate1",  "Plate",  -12.00, Y + 1.2009, -36.0, None),
    ("dhPlate2",  "Plate",   -9.60, Y + 1.2009, -36.0, 180),
    ("dhCandle",  "Candle", -10.80, Y + 1.2009, -36.0, None),
    ("dhBarrel1", "Barrel", -18.50, Y,          -43.5, None),
    ("dhBarrel2", "Barrel", -18.50, Y,          -28.5, 25),
    ("dhBarrel3", "Barrel",  -3.50, Y,          -43.5, -40),
    ("dhSkull",   "Skull",  -17.00, Y,          -31.5, -35),
    ("dhTorchW1", "Torch",  -20.36, 3.5,        -42.0, 180),
    ("dhTorchW2", "Torch",  -20.36, 3.5,        -30.0, 180),
    ("dhTorchE1", "Torch",   -1.24, 3.5,        -42.0, None),
    ("dhTorchE2", "Torch",   -1.24, 3.5,        -30.0, None),
    ("dhBanner1", "Banner", -20.36, Y,          -40.0, 180),
    ("dhBanner2", "Banner", -20.36, Y,          -32.0, 180),
    # anticamera: candela sopra il barile, sedia rovesciata contro la parete
    ("dcCarpet",  "Carpet",   6.58, Y + 0.01,   -39.6, None),
    ("dcBarrel1", "Barrel",   2.00, Y,          -35.5, None),
    ("dcCandle",  "Candle",   2.00, Y + 1.0802, -35.5, None),
    ("dcBarrel2", "Barrel",  11.50, Y,          -44.0, 15),
    ("dcChair",   "Chair",   10.00, Y,          -36.5, 200),
    ("dcSkull",   "Skull",    3.00, Y,          -43.5, 20),
    ("dcTorchE",  "Torch",   13.16, 3.5,        -39.6, None),
    ("dcBanner",  "Banner",  13.16, Y,          -43.0, None),
]
for id, key, x, y, z, rot in PROPS:
    add(id, key, (x, y, z), rot, SCALE.get(key))

# controllo di contenimento: un prop che sfora finisce dentro o oltre il muro
bbox = {}
for g in glob.glob(os.path.join(MODELS, "*.gltf")):
    a = json.load(open(g))["accessors"][0]
    bbox[os.path.basename(g)[:-5]] = (np.array(a["min"]), np.array(a["max"]))

bad = []
for e in inst:
    key = e["model"].replace("dungeon", "")
    if key in STRUCTURAL:
        continue
    mn, mx = bbox[FILE[key]]
    sc = e.get("scale", [1.0, 1.0, 1.0])[0]
    mn, mx = mn * sc, mx * sc
    th = np.radians(e.get("eulerAngles", [0, 0, 0])[1])
    c, s = np.cos(th), np.sin(th)
    corners = [(px * c + pz * s, -px * s + pz * c)
               for px in (mn[0], mx[0]) for pz in (mn[2], mx[2])]
    t = e["translate"]
    xs = [t[0] + p[0] for p in corners]
    zs = [t[2] + p[1] for p in corners]
    ix0, ix1, iz0, iz1 = INT[e["id"][:2]]
    tol = 0.40   # torce e stendardi sono appesi e mordono il muro di proposito
    if min(xs) < ix0 - tol or max(xs) > ix1 + tol or min(zs) < iz0 - tol or max(zs) > iz1 + tol:
        bad.append(e["id"])
assert not bad, "prop fuori dall'area calpestabile: %s" % bad

# ---------------------------------------------------------------- scrittura
# Inserimento testuale invece di un json.dump: riserializzare toccherebbe anche
# le righe del castello, allineate a mano, sporcando il diff di tutto il file.
MODEL_ENTRIES = [
    {"id": "dungeon" + k, "VD": "VDposNormUV",
     "model": "assets/models/Dracula/%s.gltf" % FILE[k], "format": "GLTF",
     **({"collider": "AABB"} if k in ("Wall", "Door") else {})}
    for k in FILE
]
TEX_ENTRIES = [
    {"id": "dungeon" + k + "Tex",
     "texture": "assets/textures/Dracula/%s.png" % FILE[k], "format": "C"}
    for k in FILE
]


def splice(lines, open_marker, close_marker, depth, new_entries):
    """Sostituisce le righe dungeon di un array lasciando intatte le altre."""
    s = next(i for i, l in enumerate(lines) if open_marker in l)
    e = next(i for i in range(s + 1, len(lines)) if lines[i].strip() == close_marker)
    body = [l for l in lines[s + 1:e] if "dungeon" not in l]
    while body and not body[-1].strip():
        body.pop()
    if body and not body[-1].rstrip().endswith(","):
        body[-1] = body[-1].rstrip() + ","
    pad = "\t" * depth
    block = [""] + [pad + json.dumps(x, ensure_ascii=False) + ","
                    for x in new_entries]
    block[-1] = block[-1].rstrip(",")
    return lines[:s + 1] + body + block + lines[e:]


lines = open(SCENE, encoding="utf-8").read().split("\n")
lines = splice(lines, '"models": [', "],", 2, MODEL_ENTRIES)
lines = splice(lines, '"textures": [', "],", 2, TEX_ENTRIES)
lines = splice(lines, '"elements": [', "]}", 3, inst)
open(SCENE, "w", encoding="utf-8", newline="\n").write("\n".join(lines))

doc = json.load(open(SCENE, encoding="utf-8"))   # rilettura: deve restare valido

n_struct = sum(1 for e in inst if e["model"].replace("dungeon", "") in STRUCTURAL)
print("bordi esterni coperti : %d" % len(edges))
print("istanze strutturali   : %d" % n_struct)
print("prop                  : %d" % (len(inst) - n_struct))
print("istanze totali in scena: %d" % len(doc["instances"][0]["elements"]))
