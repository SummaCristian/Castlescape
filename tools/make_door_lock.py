#!/usr/bin/env python3
"""
Genera SM_DoorChains_01 e SM_Padlock_01: le catene e il lucchetto che chiudono
una porta finche' il giocatore non spende la chiave giusta (logica dei lucchetti
in main.cpp, Door::lockKeyId).

DUE file e non uno perche' il loader di Starter.hpp concatena tutte le primitive
in un unico vertex buffer con UNA sola texture: due materiali diversi devono per
forza stare in due modelli diversi. Il taglio pero' non e' solo tecnico, e' anche
quello che si vuole a vedersi:

  SM_DoorChains_01 -> SM_Door_01.png, la stessa texture del battente: le catene
                     sono ferro del dungeon e devono sparire nel legno scuro
  SM_Padlock_01    -> Key_Albedo.png, la stessa texture della chiave: l'ottone
                     e' l'unica cosa chiara sulla porta, prende la luce delle
                     torce e dice al giocatore "qui va quella cosa che hai in
                     mano". Nessuna texture nuova: entrambe sono gia' nella
                     scena, come vuole il kit.

Nessun materiale esportato: come tutti i pezzi del kit la texture arriva da
scene.json, qui ci sono solo le UV che pescano nella zona giusta dell'atlante.

Le quote sono tutte ricavate dal battente vero, non scelte a occhio:

  SM_Door_01  x [-0.037, 0.430]   spessore
              y [ 0.156, 4.921]   altezza
              z [-2.462, 0.0  ]   larghezza, cardine a z=0

Le mesh vivono nello STESSO spazio locale del battente, quindi in scene.json
prendono translate/eulerAngles identici a quelli del suo pannello
("dhDoorPanel" & co.) e si appoggiano da sole al posto giusto: nessuna quota da
ricalcolare a mano per ogni porta.

La faccia lavorata e' quella a x negative. Il pannello e' ruotato di 180 gradi
attorno a Y in scene.json, quindi la sua x locale negativa guarda le x positive
del mondo, cioe' il lato da cui arriva il giocatore (spawn a x=-33.5, porte a
x=-36.9 e x=-0.9: si arriva sempre da est). Se una porta futura venisse messa
girata, l'unica cosa da cambiare e' il segno di FACE_X.

Uso:
    blender --background --python tools/make_door_lock.py
    python tools/convert_assets.py <cartella>/SM_DoorChains_01.gltf
    python tools/convert_assets.py <cartella>/SM_Padlock_01.gltf \
        --models-out skeleton/source/assets/models/Miscellaneous \
        --tex-out   skeleton/source/assets/textures/Miscellaneous

Il secondo passaggio porta i file nella forma che Starter.hpp sa leggere (glTF
ASCII, attributi non interlacciati, un solo materiale).
"""

import math
import os
import tempfile

import bmesh
import bpy
from mathutils import Vector

NAME_CHAINS = "SM_DoorChains_01"
NAME_LOCK = "SM_Padlock_01"

# --- battente (SM_Door_01.gltf, spazio locale, coordinate glTF con Y in alto) ---
PANEL_X0, PANEL_X1 = -0.037, 0.430
FACE_X = PANEL_X0        # faccia su cui si costruisce
# Su quale delle due facce del battente finisce il pezzo. Il lavoro si fa sempre
# su quella a x negative e alla fine, se serve, si specchia: e' un solo passaggio
# in fondo a build(), invece di ogni singola quota scritta due volte col segno
# giusto.
#
# Quale delle due serve dipende da dove arriva il giocatore, non dalla porta in
# se': i pannelli in scene.json sono tutti ruotati di 180 gradi attorno a Y,
# quindi la loro x locale negativa guarda EST. Le porte di dl (x = -0.9) si
# raggiungono da ovest, venendo dall'anticamera, e vogliono True; quella in
# fondo alla sala iniziale (x = -36.9) si guarda da est e vuole False.
#
# In pratica serve UN solo export, quello con True: main.cpp sa girare le mesh
# a runtime per l'altra faccia (addLockProp(..., flip=true), che applica una
# rotazione di 180 gradi attorno alla verticale per (x medio spessore, z
# mezzeria del vano)). Funziona perche' entrambi i pezzi sono simmetrici
# rispetto a quella z. Se un pezzo futuro NON lo fosse, allora si', va
# riesportato con la flag girata.
FRONT_ON_PLUS_X = True
DOOR_Z0, DOOR_Z1 = -2.462, 0.0
DOOR_MID_Z = -1.231      # mezzeria del vano, la stessa che main.cpp usa per il prompt
DOOR_MID_Y = 2.52        # mezzeria in altezza del vano

# --- catena ---
LINK_R = 0.075           # raggio dei due semicerchi dell'anello
LINK_L = 0.150           # tratto dritto fra i due semicerchi
TUBE_R = 0.030           # raggio del tondino
LINK_PITCH = 0.170       # passo fra i centri: piu' corto della semilunghezza
                         # dell'anello (LINK_R + LINK_L/2 = 0.15), che e' quello
                         # che fa incastrare un anello nel successivo
# Segmenti del semicerchio e lati della sezione. NON sono una scelta di budget:
# le maglie sono lisciate (normali mediate), quindi ogni punto in cui la linea
# media cambia curvatura di scatto -- il passaggio fra cappello e tratto dritto
# dell'asola -- e' una discontinuita' di normale. Un lobo speculare stretto,
# quale ha un metallo, su una discontinuita' del genere produce una riga
# brillante larga un pixel, che antialiasata fra cresta calda e ombra fredda
# legge come frangia colorata sul bordo. Con 6 e 4 quelle righe si vedevano a
# occhio; a 8 e 6 la normale varia abbastanza dolcemente da non produrle.
# Costo: circa il doppio dei triangoli su un pezzo che ne aveva 2808.
LINK_SEG = 6             # segmenti per semicerchio della maglia
TUBE_SIDES = 8           # lati della sezione del tondino
# Gli anelli "di taglio" sporgono di LINK_R + TUBE_R = 0.105 fuori dal piano
# della catena: il piano sta 0.12 davanti alla porta, cosi' restano 0.015 liberi
# e nessuna maglia entra dentro il pannello.
CHAIN_X = FACE_X - 0.12

# Percorsi delle due catene, nel piano (z, y). Quella alta si affloscia verso il
# centro e passa dentro l'arco del lucchetto; quella bassa e' dritta e serve solo
# a dire "questa porta e' incatenata", non regge niente.
#
# Le piastre stanno alle quote qui sotto e il percorso finisce mezza maglia
# PRIMA, cioe' arretrato di LINK_HALF: cosi' la punta dell'ultimo anello cade
# esattamente sul centro della piastra e ci sparisce dentro. Facendo finire il
# percorso sulla piastra, invece, l'ultima maglia sporgeva per meta' oltre la
# piastra e la catena sembrava tagliata a meta' invece che ancorata.
PLATE_Z0, PLATE_Z1 = -0.22, -2.24
LINK_HALF = LINK_R + LINK_L / 2 + TUBE_R
CHAIN_END_Z0, CHAIN_END_Z1 = PLATE_Z0 - LINK_HALF, PLATE_Z1 + LINK_HALF
CHAIN_TOP_Y = 2.80
CHAIN_SAG_Y = 2.47       # quota a cui la catena alta attraversa l'arco
CHAIN_LOW_Y = 1.62

# --- piastre di ancoraggio alle estremita' ---
PLATE_W, PLATE_H, PLATE_T = 0.28, 0.32, 0.08
# Quanto la piastra AFFONDA nel battente. Non e' un dettaglio: appoggiarla
# esattamente sulla faccia x = FACE_X darebbe due quad complanari e coincidenti,
# cioe' lo z-fighting che le mesh del kit hanno gia' per conto loro (vedi
# notes.md). Compenetrare di un paio di centimetri nasconde la faccia posteriore
# dentro il legno e non lascia nessuna fessura da vedere in controluce.
PLATE_SINK = 0.02

# --- lucchetto ---
LOCK_Z = DOOR_MID_Z      # appeso in mezzeria del vano
BODY_W, BODY_H, BODY_T = 0.42, 0.44, 0.18
BODY_CORNER = 0.09
BODY_TOP_Y = 2.38
SHACKLE_R = 0.17         # raggio dell'arco: la luce interna (2*(R - r)) deve
                         # lasciar passare il tondino della catena
SHACKLE_TUBE_R = 0.035
SHACKLE_SIDES = 8
SHACKLE_SEG = 8
SHACKLE_CENTER_Y = 2.40  # centro dell'arco, appena sopra il corpo
KEYHOLE_R = 0.075
KEYHOLE_DEPTH = 0.055    # abbastanza fonda da farsi l'ombra da sola: la texture
                         # della chiave e' ottone piatto, il buco si legge solo
                         # se e' geometria
OUTLINE_STEPS = 5        # punti per quadrante del contorno del corpo e della toppa
# Corpo centrato sul piano della catena: cosi' l'arco e le maglie stanno sullo
# stesso piano e la maglia centrale lo attraversa davvero invece di sfiorarlo.
# Ne segue che il retro del corpo resta a 0.03 dal battente -- niente facce
# complanari con l'anta, che e' la condizione da evitare qui (vedi PLATE_SINK).
LOCK_FRONT_X = CHAIN_X - BODY_T / 2

# --- zone dell'atlante da cui pescare ---
# Coordinate immagine (v dall'alto, come si leggono in un editor), normalizzate
# sui 1024 px delle due texture. Non sono ritagli a caso: la prima e' la fascia
# di ferro battuto delle bande della porta (media 54/255, deviazione 5, cioe'
# variazione appena percettibile e nessun bordo dentro), la seconda e' ottone
# pieno della chiave (255,164,0 uniforme). Rimisurare se le texture cambiano.
IRON_UV = (676 / 1024.0, 30 / 1024.0, 720 / 1024.0, 250 / 1024.0)
GOLD_UV = (300 / 1024.0, 600 / 1024.0, 460 / 1024.0, 760 / 1024.0)


def uv_in(rect, s, t):
    """(s,t) in [0,1]^2 -> punto dentro il ritaglio, in coordinate immagine."""
    u0, v0, u1, v1 = rect
    return (u0 + (u1 - u0) * s, v0 + (v1 - v0) * t)


def emit(faces, verts, uvs, ref, toward=False, smooth=True):
    """Aggiunge una faccia orientando l'avvolgimento rispetto a `ref`.

    La normale deve puntare LONTANO da ref (il centro del pezzo) per una
    superficie esterna, e verso ref per una cava tipo il buco della serratura.
    Calcolarlo invece di ragionare sull'ordine dei vertici e' l'unico modo di
    non sbagliare un verso su un pezzo tubolare, dove "fuori" cambia a ogni
    segmento.
    """
    p = [Vector(v) for v in verts]
    n = (p[1] - p[0]).cross(p[2] - p[0])
    c = sum(p, Vector((0, 0, 0))) / len(p)
    outward = n.dot(c - Vector(ref))
    if (outward < 0) != toward:
        verts = list(reversed(verts))
        uvs = list(reversed(uvs))
    # smooth per faccia e non per oggetto: il tondino deve sembrare tondo con
    # sei lati, il corpo del lucchetto invece e' un blocco e con le normali
    # mediate diventerebbe un cuscino.
    faces.append((verts, uvs, smooth))


def planar_tube(faces, pts, plane_normal, r, sides, rect, closed, cap=False):
    """Tondino di raggio r lungo una spezzata PIANA.

    La sezione e' un poligono nel piano perpendicolare alla tangente, costruito
    su (plane_normal, plane_normal x tangente): con un percorso piano il
    riferimento non ruota mai, quindi non serve trasporto parallelo e la maglia
    non si attorciglia.
    """
    pts = [Vector(p) for p in pts]
    a = Vector(plane_normal).normalized()
    n = len(pts)

    rings = []
    for i in range(n):
        if closed:
            t = (pts[(i + 1) % n] - pts[(i - 1) % n])
        elif i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        t.normalize()
        b = t.cross(a).normalized()
        rings.append([pts[i] + (a * math.cos(th) + b * math.sin(th)) * r
                      for th in [2 * math.pi * j / sides for j in range(sides)]])

    span = n if closed else n - 1
    for i in range(span):
        i2 = (i + 1) % n
        centre = (pts[i] + pts[i2]) / 2.0
        for j in range(sides):
            j2 = (j + 1) % sides
            quad = [rings[i][j], rings[i2][j], rings[i2][j2], rings[i][j2]]
            uvq = [uv_in(rect, j / sides, i / max(1, span)),
                   uv_in(rect, j / sides, (i + 1) / max(1, span)),
                   uv_in(rect, j2 / sides, (i + 1) / max(1, span)),
                   uv_in(rect, j2 / sides, i / max(1, span))]
            emit(faces, [tuple(v) for v in quad], uvq, tuple(centre))

    if cap and not closed:
        for i, ref in ((0, pts[1]), (n - 1, pts[-2])):
            ring = rings[i] if i == 0 else list(reversed(rings[i]))
            emit(faces, [tuple(v) for v in ring],
                 [uv_in(rect, 0.5, 0.5)] * sides, tuple(ref), smooth=False)


def link_centreline(centre, tangent, plane_normal):
    """Asola (due semicerchi + due tratti dritti): la linea media di una maglia."""
    t = Vector(tangent).normalized()
    a = Vector(plane_normal).normalized()
    b = t.cross(a).normalized()
    # I due semicerchi hanno lo STESSO verso di percorrenza (theta che cresce):
    # cambia solo il centro, spostato di +-L/2. Ribaltare anche il seno e il
    # coseno insieme al centro chiuderebbe l'asola a farfalla invece che ad
    # anello -- il giro deve uscire dal cappello destro in alto e rientrare in
    # quello sinistro dallo stesso lato.
    pts = []
    for sign, base in ((1.0, -math.pi / 2), (-1.0, math.pi / 2)):
        for i in range(LINK_SEG + 1):
            th = base + math.pi * i / LINK_SEG
            pts.append(Vector(centre)
                       + t * (sign * LINK_L / 2 + LINK_R * math.cos(th))
                       + b * (LINK_R * math.sin(th)))
    # i due estremi di ogni semicerchio coincidono con l'inizio dell'altro
    out = []
    for p in pts:
        if not out or (p - out[-1]).length > 1e-6:
            out.append(p)
    if (out[0] - out[-1]).length < 1e-6:
        out.pop()
    return out


def sample_path(path, pitch):
    """Punti equidistanti lungo una spezzata, con la tangente locale."""
    pts = [Vector(p) for p in path]
    segs = [(pts[i], pts[i + 1], (pts[i + 1] - pts[i]).length) for i in range(len(pts) - 1)]
    total = sum(s[2] for s in segs)
    count = max(2, int(round(total / pitch)))
    step = total / count
    out = []
    for k in range(count + 1):
        d = k * step
        for p0, p1, ln in segs:
            if d <= ln or (p0, p1, ln) is segs[-1]:
                out.append((p0 + (p1 - p0) * (min(d, ln) / ln), (p1 - p0).normalized()))
                break
            d -= ln
    return out


def chain(faces, path, pitch, flip_parity):
    """Stende una catena lungo un percorso nel piano x = CHAIN_X.

    Le maglie alternano il piano di 90 gradi, come una catena vera: una di
    faccia (piano dell'anta, si vede l'asola) e una di taglio (sporge in x).
    flip_parity sceglie quale delle due capita in mezzo -- serve perche' la
    maglia che passa dentro l'arco del lucchetto deve essere quella di taglio,
    l'unica abbastanza stretta da starci dentro.
    """
    samples = sample_path(path, pitch)
    for i, (centre, tangent) in enumerate(samples):
        face_on = ((i + flip_parity) % 2 == 0)
        # di faccia: il piano della maglia contiene la tangente e la verticale
        # del percorso, quindi la sua normale e' x; di taglio: normale nel piano.
        if face_on:
            normal = Vector((1.0, 0.0, 0.0))
        else:
            normal = tangent.cross(Vector((1.0, 0.0, 0.0))).normalized()
        pts = link_centreline(centre, tangent, normal)
        planar_tube(faces, pts, normal, TUBE_R, TUBE_SIDES, IRON_UV, closed=True)


def box(faces, cx, cy, cz, sx, sy, sz, rect):
    """Scatola allineata agli assi, per le piastre di ancoraggio."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    c = (cx, cy, cz)
    corners = {}
    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-1, 1):
                corners[(i, j, k)] = (cx + i * hx, cy + j * hy, cz + k * hz)
    quads = [
        [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
        [(1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)],
        [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
        [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)],
        [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)],
        [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    ]
    for q in quads:
        emit(faces, [corners[k] for k in q],
             [uv_in(rect, 0.15, 0.15), uv_in(rect, 0.85, 0.15),
              uv_in(rect, 0.85, 0.85), uv_in(rect, 0.15, 0.85)], c, smooth=False)


def rounded_rect(cz, cy, w, h, radius, steps=OUTLINE_STEPS):
    """Contorno stondato nel piano (z, y), antiorario, 4*steps punti.

    Gli angoli sono campionati a meta' passo (i + 0.5) invece che sugli estremi:
    cosi' due quadranti non producono mai due punti coincidenti nemmeno quando i
    raggi mangiano tutto il lato, cioe' quando il rettangolo stondato degenera
    in un cerchio. E' quello che permette di usare la STESSA funzione per il
    contorno del corpo e per il cerchio della toppa e di ottenere due giri con
    lo stesso numero di punti, da cui una corona di quad regolari fra i due.
    """
    hz, hy = w / 2 - radius, h / 2 - radius
    pts = []
    for (sz, sy, base) in ((1, -1, -math.pi / 2), (1, 1, 0.0),
                           (-1, 1, math.pi / 2), (-1, -1, math.pi)):
        for i in range(steps):
            th = base + math.pi / 2 * (i + 0.5) / steps
            pts.append((cz + sz * hz + radius * math.cos(th),
                        cy + sy * hy + radius * math.sin(th)))
    return pts


def build_chains():
    faces = []

    upper = [(CHAIN_X, CHAIN_TOP_Y, CHAIN_END_Z0),
             (CHAIN_X, CHAIN_SAG_Y, LOCK_Z),
             (CHAIN_X, CHAIN_TOP_Y, CHAIN_END_Z1)]
    lower = [(CHAIN_X, CHAIN_LOW_Y, CHAIN_END_Z0),
             (CHAIN_X, CHAIN_LOW_Y, CHAIN_END_Z1)]

    # La maglia centrale della catena alta e' quella che entra nell'arco: con un
    # numero pari di campioni per meta' percorso capita di faccia, quindi la
    # parita' e' invertita di uno.
    chain(faces, upper, LINK_PITCH, flip_parity=1)
    chain(faces, lower, LINK_PITCH, flip_parity=0)

    for z, y in ((PLATE_Z0, CHAIN_TOP_Y), (PLATE_Z1, CHAIN_TOP_Y),
                 (PLATE_Z0, CHAIN_LOW_Y), (PLATE_Z1, CHAIN_LOW_Y)):
        box(faces, FACE_X + PLATE_SINK - PLATE_T / 2, y, z,
            PLATE_T, PLATE_H, PLATE_W, IRON_UV)

    return to_object(NAME_CHAINS, mirror(faces) if FRONT_ON_PLUS_X else faces)


def build_lock():
    faces = []
    x_front = LOCK_FRONT_X
    x_back = LOCK_FRONT_X + BODY_T
    body_cy = BODY_TOP_Y - BODY_H / 2

    outline = rounded_rect(LOCK_Z, body_cy, BODY_W, BODY_H, BODY_CORNER)
    n = len(outline)
    centre = (x_front + BODY_T / 2, body_cy, LOCK_Z)

    # fianco: una striscia di quad lungo il contorno
    for i in range(n):
        z0, y0 = outline[i]
        z1, y1 = outline[(i + 1) % n]
        emit(faces,
             [(x_front, y0, z0), (x_back, y0, z0), (x_back, y1, z1), (x_front, y1, z1)],
             [uv_in(GOLD_UV, i / n, 0.1), uv_in(GOLD_UV, i / n, 0.9),
              uv_in(GOLD_UV, (i + 1) / n, 0.9), uv_in(GOLD_UV, (i + 1) / n, 0.1)],
             centre, smooth=False)

    # retro: non si vede mai (guarda il battente) ma il pezzo resta chiuso
    emit(faces, [(x_back, y, z) for z, y in outline],
         [uv_in(GOLD_UV, 0.5, 0.5)] * n, centre, smooth=False)

    # fronte: anello dal contorno al cerchio della toppa, poi la cava. Il buco
    # e' modellato e non dipinto perche' la texture della chiave e' ottone
    # uniforme: qui l'unico dettaglio possibile e' quello che fa ombra da solo.
    # Il cerchio della toppa nasce dallo STESSO generatore del contorno (un
    # rettangolo stondato il cui raggio d'angolo e' meta' del lato e' un
    # cerchio): stesso numero di punti e stesso ordine, quindi la corona fra i
    # due e' una striscia di quad regolari. Contarli in modo indipendente
    # lasciava quad degeneri, cioe' lo strappo sulla faccia.
    hole = rounded_rect(LOCK_Z, body_cy - 0.03,
                        2 * KEYHOLE_R, 2 * KEYHOLE_R, KEYHOLE_R)
    assert len(hole) == n, "contorno e toppa devono avere lo stesso numero di punti"
    for i in range(n):
        z0, y0 = outline[i]
        z1, y1 = outline[(i + 1) % n]
        hz0, hy0 = hole[i]
        hz1, hy1 = hole[(i + 1) % n]
        emit(faces,
             [(x_front, y0, z0), (x_front, y1, z1),
              (x_front, hy1, hz1), (x_front, hy0, hz0)],
             [uv_in(GOLD_UV, i / n, 0.05), uv_in(GOLD_UV, (i + 1) / n, 0.05),
              uv_in(GOLD_UV, (i + 1) / n, 0.45), uv_in(GOLD_UV, i / n, 0.45)],
             centre, smooth=False)

    for i in range(n):
        z0, y0 = hole[i]
        z1, y1 = hole[(i + 1) % n]
        # parete della cava: la normale guarda l'asse del buco, non fuori
        emit(faces,
             [(x_front, y0, z0), (x_front, y1, z1),
              (x_front + KEYHOLE_DEPTH, y1, z1), (x_front + KEYHOLE_DEPTH, y0, z0)],
             [uv_in(GOLD_UV, 0.1, 0.1), uv_in(GOLD_UV, 0.9, 0.1),
              uv_in(GOLD_UV, 0.9, 0.9), uv_in(GOLD_UV, 0.1, 0.9)],
             (x_front + KEYHOLE_DEPTH, body_cy - 0.03, LOCK_Z),
             toward=True, smooth=False)
    emit(faces, [(x_front + KEYHOLE_DEPTH, y, z) for z, y in hole],
         [uv_in(GOLD_UV, 0.5, 0.5)] * n,
         (x_back, body_cy, LOCK_Z), smooth=False)

    # arco: due gambe dritte che affondano nel corpo piu' il semicerchio sopra
    x_mid = x_front + BODY_T / 2
    leg_bottom = BODY_TOP_Y - 0.10
    path = [(x_mid, leg_bottom, LOCK_Z - SHACKLE_R),
            (x_mid, SHACKLE_CENTER_Y, LOCK_Z - SHACKLE_R)]
    for i in range(SHACKLE_SEG + 1):
        th = math.pi - math.pi * i / SHACKLE_SEG
        path.append((x_mid,
                     SHACKLE_CENTER_Y + SHACKLE_R * math.sin(th),
                     LOCK_Z + SHACKLE_R * math.cos(th)))
    path += [(x_mid, SHACKLE_CENTER_Y, LOCK_Z + SHACKLE_R),
             (x_mid, leg_bottom, LOCK_Z + SHACKLE_R)]
    clean = []
    for p in path:
        if not clean or (Vector(p) - Vector(clean[-1])).length > 1e-6:
            clean.append(p)
    planar_tube(faces, clean, (1.0, 0.0, 0.0), SHACKLE_TUBE_R, SHACKLE_SIDES,
                GOLD_UV, closed=False, cap=True)

    return to_object(NAME_LOCK, mirror(faces) if FRONT_ON_PLUS_X else faces)


def mirror(faces):
    """Specchia il pezzo sull'altra faccia del battente.

    Il piano dello specchio e' la mezzeria dello spessore del pannello, cosi'
    quello che era appoggiato (o affondato) su una faccia resta appoggiato (o
    affondato dello stesso tanto) sull'altra. Lo specchio inverte
    l'avvolgimento, quindi ogni faccia va riletta al contrario o la normale
    finirebbe dentro al pezzo.
    """
    twice_mid = PANEL_X0 + PANEL_X1
    out = []
    for verts, uvs, smooth in faces:
        flipped = [(twice_mid - v[0], v[1], v[2]) for v in verts]
        out.append((list(reversed(flipped)), list(reversed(uvs)), smooth))
    return out


def to_object(name, faces):
    """glTF (x, y, z) -> Blender (x, -z, y); la V delle UV e' ribaltata."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for verts, uvs, smooth in faces:
        try:
            f = bm.faces.new([bm.verts.new((v[0], -v[2], v[1])) for v in verts])
        except ValueError:
            continue
        f.smooth = smooth
        for loop, uv in zip(f.loops, uvs):
            loop[uv_layer].uv = (uv[0], 1.0 - uv[1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()

    for o in [o for o in bpy.data.objects if o.name.startswith(name)]:
        bpy.data.objects.remove(o, do_unlink=True)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def check(obj):
    """UV dentro l'atlante e ingombro davanti al battente, non dentro."""
    me = obj.data
    uvs = [l.uv for l in me.uv_layers[0].data]
    assert all(-1e-6 <= u <= 1 + 1e-6 and -1e-6 <= v <= 1 + 1e-6 for u, v in uvs), \
        "UV fuori dalla texture"
    # bbox in coordinate glTF: Blender (x, y, z) -> glTF (x, z, -y)
    xs = [v.co[0] for v in me.vertices]
    ys = [v.co[2] for v in me.vertices]
    zs = [-v.co[1] for v in me.vertices]
    # Solo le piastre possono affondare nel legno, e solo di PLATE_SINK: tutto
    # il resto deve restare davanti alla faccia del battente, da qualunque
    # delle due parte sia finito il pezzo.
    if FRONT_ON_PLUS_X:
        assert min(xs) >= PANEL_X1 - PLATE_SINK - 1e-4, \
            "%s entra nel battente (x min %.3f < %.3f)" % (obj.name, min(xs), PANEL_X1)
    else:
        assert max(xs) <= PANEL_X0 + PLATE_SINK + 1e-4, \
            "%s entra nel battente (x max %.3f > %.3f)" % (obj.name, max(xs), PANEL_X0)
    assert DOOR_Z0 - 1e-4 <= min(zs) and max(zs) <= DOOR_Z1 + 1e-4, \
        "%s sborda in larghezza (z %.3f..%.3f)" % (obj.name, min(zs), max(zs))
    print("%-20s %5d facce  bbox x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f" %
          (obj.name, len(me.polygons), min(xs), max(xs),
           min(ys), max(ys), min(zs), max(zs)))


def export(objs):
    out_dir = os.path.join(tempfile.gettempdir(), "cg_lock_export")
    os.makedirs(out_dir, exist_ok=True)
    for obj in objs:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.export_scene.gltf(
            filepath=os.path.join(out_dir, obj.name + ".gltf"),
            export_format='GLTF_SEPARATE', use_selection=True, export_apply=True,
            export_yup=True, export_normals=True, export_texcoords=True,
            export_materials='NONE', export_extras=False)
    return out_dir


if __name__ == "__main__":
    built = [build_chains(), build_lock()]
    for o in built:
        check(o)
    out = export(built)
    print("\nesportato in %s" % out)
