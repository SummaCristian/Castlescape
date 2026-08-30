#!/usr/bin/env python3
"""
Genera SM_Bookshelf_01 e SM_Book_01: la libreria che nasconde il passaggio
segreto e il libro che la apre (logica in main.cpp, Door::lockKeyId con
lockKeyId = "book" e la LockProp montata su `whenUnlocked`).

DUE file e non uno per lo stesso motivo delle catene e del lucchetto: il loader
di Starter.hpp concatena tutte le primitive in un unico vertex buffer con UNA
sola texture, quindi due materiali diversi devono stare in due modelli diversi.
E anche qui il taglio tecnico coincide con quello che si vuole vedere:

  SM_Bookshelf_01 -> SM_Door_01.png, la stessa texture del battente. La libreria
                    DEVE essere fatta della stessa roba delle porte del dungeon:
                    e' un mobile che finge di essere un muro, e se il legno non
                    fosse quello di casa si vedrebbe da lontano che nasconde
                    qualcosa. I dorsi dei libri pescano tre zone diverse dello
                    stesso atlante (noce chiaro, noce scuro, ferro quasi nero,
                    pietra chiara) -- niente colori nuovi, solo tono.
  SM_Book_01      -> SM_CastleBanners_01.png, il cremisi degli stendardi con il
                    taglio dorato. Qui il contrasto serve: e' un pickup e deve
                    farsi trovare su un tavolo buio, esattamente come il
                    lucchetto prende l'ottone della chiave per dire "questa cosa
                    va con quest'altra". Il libro sullo scaffale, invece, deve
                    sparire nella libreria, ed e' per questo che la fessura
                    vuota e' l'unico indizio.

Nessun materiale esportato: come tutti i pezzi del kit la texture arriva da
scene.json, qui ci sono solo le UV che pescano nella zona giusta dell'atlante.

--- Perche' queste quote e non altre ---

La libreria e' un'ANTA a tutti gli effetti: main.cpp la fa girare con lo stesso
codice delle porte (addDoor), che assume una cosa sola sul mesh, ossia che
l'origine locale stia SUL CARDINE e che il pannello penda tutto da una parte di
z = 0. SM_Door_01 e' fatto cosi':

  SM_Door_01  x [-0.037, 0.430]   spessore
              y [ 0.156, 4.921]   altezza
              z [-2.462, 0.0  ]   larghezza, cardine a z=0

quindi la libreria copia z e x (piu' profonda: un mobile non e' un'anta) e
soprattutto copia la SAGOMA. Il vano di SM_WallDoor_Hole_01 non e'
rettangolare, e' un arco: rasterizzando i triangoli del muro sul piano (z, y) il
buco risulta un rettangolo z 2.420..4.780 fino a y 4.12, poi si chiude in arco e
finisce a y 4.92 (i numeri sono in HOLE_PROFILE piu' sotto, in coordinate locali
dell'anta). Una libreria rettangolare alta 4.9 sfonderebbe l'arco; una alta 4.12
lascerebbe scoperta la lunetta e da un metro di distanza si vedrebbe la stanza
segreta sopra i libri. Da qui il cappello: il fusto e' rettangolare fino a 4.12
e sopra c'e' un massello sagomato sull'arco piu' 0.055 di margine, che e' la
stessa sovrapposizione con cui l'anta di legno copre i propri stipiti.

--- Cosa NON e' un dettaglio ---

La fessura vuota su un ripiano. E' il terzo ripiano da sotto, y = 1.56, cioe'
appena sotto l'altezza degli occhi del giocatore (1.8): e' l'unica cosa che dice
"qui manca qualcosa" senza una riga di HUD, e va guardata, non cercata a
tastoni. Le costanti BOOK_SLOT_* qui sotto sono le stesse tre che main.cpp
riscrive nel proprio BOOK_SLOT_LOCAL per infilarci dentro il libro quando viene
speso: se si sposta la fessura vanno cambiate in DUE posti.

Uso:
    blender --background --python tools/make_bookshelf.py
    python tools/convert_assets.py <cartella>/SM_Bookshelf_01.gltf \
        --models-out skeleton/source/assets/models/Dracula \
        --tex-out   skeleton/source/assets/textures/Dracula
    python tools/convert_assets.py <cartella>/SM_Book_01.gltf \
        --models-out skeleton/source/assets/models/Dracula \
        --tex-out   skeleton/source/assets/textures/Dracula

Il secondo passaggio porta i file nella forma che Starter.hpp sa leggere (glTF
ASCII, attributi non interlacciati, un solo materiale).
"""

import math
import os
import random
import tempfile

import bmesh
import bpy
from mathutils import Vector

NAME_SHELF = "SM_Bookshelf_01"
NAME_BOOK = "SM_Book_01"

# --- l'anta (coordinate locali, glTF con Y in alto, cardine a z = 0) ---
SHELF_Z0, SHELF_Z1 = -2.462, 0.0      # larghezza, identica a SM_Door_01
BACK_X = -0.037                       # faccia posteriore, identica a SM_Door_01
FRONT_X = 0.523                       # faccia anteriore: 0.56 di profondita'
CASE_TOP_Y = 4.12                     # dove il vano smette di essere rettangolare
CAP_TOP_Y = 4.95                      # sopra il punto in cui il vano si chiude (4.92)
# Il mobile non parte da y = 0: appoggiarlo esattamente sul piano del pavimento
# (che sta anch'esso a 0.02, la quota comune di tutta la scena) darebbe due quad
# complanari e coincidenti, cioe' lo z-fighting che le mesh del kit hanno gia'
# per conto loro. Tre centimetri di luce sotto lo zoccolo non si vedono da in
# piedi e sono la stessa soluzione che SM_Door_01 adotta partendo da 0.156.
# Sta qui e non in scene.json apposta: cosi' l'istanza usa lo stesso
# "translate": [x, 0.02, z] di ogni altro pezzo del dungeon.
CASE_BOTTOM_Y = 0.03
CENTRE_Z = (SHELF_Z0 + SHELF_Z1) / 2  # -1.231

# Sagoma del vano di SM_WallDoor_Hole_01, in coordinate locali dell'anta:
# (y, mezza larghezza attorno a z = -1.221, che e' la mezzeria VERA del buco).
# Misurata rasterizzando i triangoli del muro, non a occhio -- e' la stessa
# misura che il commento di addDoor() in main.cpp cita per il promptOffset.
# Rifarla se l'asset del muro viene rigenerato.
HOLE_CENTRE_Z = -1.221
HOLE_PROFILE = [
    (4.120, 1.180), (4.167, 1.148), (4.256, 1.116), (4.344, 1.068),
    (4.433, 1.028), (4.522, 0.980), (4.610, 0.900), (4.699, 0.788),
    (4.787, 0.652), (4.876, 0.420), (4.920, 0.000),
]
# Quanto il cappello sborda oltre il buco. Stesso ordine di grandezza della
# sovrapposizione dell'anta sugli stipiti (0.061 per lato), abbastanza da non
# lasciare una fessura di luce al bordo dell'arco e poco abbastanza da non
# entrare nel muro quando l'anta ruota.
CAP_MARGIN = 0.055

# --- carcassa ---
BACK_T = 0.08                         # spessore dello schienale
SIDE_T = 0.09                         # spessore dei fianchi
BOARD_T = 0.05                        # spessore dei ripiani
PLINTH_Y = 0.28                       # zoccolo: quota della sua faccia superiore
# Quote della faccia SUPERIORE di ogni ripiano, cioe' dove appoggiano i libri.
# Sei campate da ~0.59 di luce: con 4.12 di altezza totale, campate piu' alte
# davano libri alti mezzo metro (il fusto e' scalato sul vano di una porta da
# 4.9, non su un mobile umano) e la libreria sembrava un magazzino.
BOARD_TOPS = [PLINTH_Y, 0.92, 1.56, 2.20, 2.84, 3.48]
TOP_BOARD_Y = 4.02                    # faccia superiore del ripiano piu' alto

# --- la fessura vuota (le stesse tre quote stanno in main.cpp, BOOK_SLOT_LOCAL) ---
BOOK_SLOT_BOARD = 2                   # indice in BOARD_TOPS: y = 1.56
BOOK_SLOT_Z = -1.300                  # bordo della fessura dal lato -z
BOOK_SLOT_W = 0.110                   # larghezza: il libro e' 0.070, resta gioco
BOOK_SPINE_X = 0.470                  # dove cadono i dorsi, appena rientrati

# --- i libri sugli scaffali ---
BOOK_H_MIN, BOOK_H_MAX = 0.34, 0.50   # altezza; l'ultima campata e' piu' bassa
BOOK_W_MIN, BOOK_W_MAX = 0.055, 0.135 # spessore del dorso
BOOK_DEPTH = 0.34                     # dal dorso verso lo schienale
BOOK_TILT_MAX = 11.0                  # gradi: qualche volume storto, non tutti
BOOK_GAP = 0.012                      # aria fra un volume e l'altro
BOOK_SEED = 20260830                  # deterministico: due export danno lo stesso mesh

# --- il libro pickup (spazio locale suo, appoggiato in piano) ---
# x da 0 (dorso) a 0.230 (taglio davanti), z lungo l'altezza delle pagine,
# y lo spessore a partire da 0: cosi' in scene.json basta un translate sul piano
# del tavolo piu' un'imbardata, senza rotazioni da indovinare. La posa in piedi
# nella fessura e quella in mano sono due matrici in main.cpp.
BOOK_LEN_X = 0.230
BOOK_LEN_Z = 0.340
BOOK_THICK = 0.070
BOOK_COVER_T = 0.011                  # spessore di un piatto
BOOK_SPINE_R = 0.018                  # sporgenza del dorso oltre i piatti
BOOK_PAGE_INSET = 0.014               # quanto le pagine rientrano sotto i piatti

# --- zone dell'atlante da cui pescare ---
# Coordinate immagine (v dall'alto, come si leggono in un editor), normalizzate
# sui 1024 px delle due texture. Misurate, non scelte a caso -- media e
# deviazione fra parentesi, rimisurare se le texture cambiano.
#
# SM_Door_01.png:
WOOD_UV = (430 / 1024.0, 400 / 1024.0, 540 / 1024.0, 450 / 1024.0)   # (69,49,31) s16
WOOD_DARK_UV = (440 / 1024.0, 468 / 1024.0, 540 / 1024.0, 496 / 1024.0)  # (59,41,25) s19
IRON_UV = (676 / 1024.0, 30 / 1024.0, 720 / 1024.0, 250 / 1024.0)    # (54,54,54) s5
STONE_UV = (858 / 1024.0, 620 / 1024.0, 882 / 1024.0, 940 / 1024.0)  # (116,111,101) s13
# SM_CastleBanners_01.png:
RED_UV = (430 / 1024.0, 750 / 1024.0, 640 / 1024.0, 880 / 1024.0)    # (172,72,72) s10
GOLD_UV = (802 / 1024.0, 235 / 1024.0, 838 / 1024.0, 345 / 1024.0)   # ottone/oro

# La libreria e' di legno; i dorsi alternano queste quattro. Pelle chiara e
# pergamena esistono perche' una parete di soli marroni scuri, in una stanza
# illuminata da una torcia, e' una macchia nera: servono due o tre volumi che
# prendano la luce per far leggere la libreria come libreria.
SPINE_UVS = [WOOD_UV, WOOD_DARK_UV, IRON_UV, STONE_UV, WOOD_DARK_UV, WOOD_UV]


def uv_in(rect, s, t):
    """(s,t) in [0,1]^2 -> punto dentro il ritaglio, in coordinate immagine."""
    u0, v0, u1, v1 = rect
    return (u0 + (u1 - u0) * s, v0 + (v1 - v0) * t)


def sub_rect(rect, rng, frac=0.45):
    """Ritaglio piu' piccolo, pescato a caso dentro `rect`.

    Serve solo ai dorsi: due libri che campionano lo stesso identico quadratino
    escono dello stesso identico colore e la fila diventa una striscia piatta.
    Pescando finestre diverse dentro la stessa zona ognuno prende la propria
    venatura, senza uscire dal tono deciso sopra.
    """
    u0, v0, u1, v1 = rect
    w, h = (u1 - u0) * frac, (v1 - v0) * frac
    return (u0 + rng.random() * (u1 - u0 - w), v0 + rng.random() * (v1 - v0 - h),
            u0 + rng.random() * (u1 - u0 - w) + w, v0 + rng.random() * (v1 - v0 - h) + h)


def emit(faces, verts, uvs, ref, toward=False, smooth=False):
    """Aggiunge una faccia orientando l'avvolgimento rispetto a `ref`.

    La normale deve puntare LONTANO da ref (il centro del pezzo) per una
    superficie esterna, e verso ref per una cava. Calcolarlo invece di ragionare
    sull'ordine dei vertici e' l'unico modo di non sbagliare un verso su un
    pezzo fatto di decine di scatole ruotate.

    smooth di default False, al contrario di make_door_lock.py: qui non c'e'
    niente di tondo. Un libro con le normali mediate diventa un cuscino e uno
    scaffale un materasso.
    """
    p = [Vector(v) for v in verts]
    n = (p[1] - p[0]).cross(p[2] - p[0])
    c = sum(p, Vector((0, 0, 0))) / len(p)
    outward = n.dot(c - Vector(ref))
    if (outward < 0) != toward:
        verts = list(reversed(verts))
        uvs = list(reversed(uvs))
    faces.append((verts, uvs, smooth))


def box(faces, x0, x1, y0, y1, z0, z1, rect, inset=0.15):
    """Scatola allineata agli assi: ripiani, fianchi, schienale, zoccolo."""
    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    corners = {(i, j, k): (x0 if i < 0 else x1, y0 if j < 0 else y1, z0 if k < 0 else z1)
               for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)}
    quads = [
        [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
        [(1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)],
        [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
        [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)],
        [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)],
        [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    ]
    lo, hi = inset, 1.0 - inset
    for q in quads:
        emit(faces, [corners[k] for k in q],
             [uv_in(rect, lo, lo), uv_in(rect, hi, lo),
              uv_in(rect, hi, hi), uv_in(rect, lo, hi)], c)


def tilted_book(faces, x0, x1, y0, height, z_lo, z_hi, rect, tilt_deg):
    """Un volume in piedi su un ripiano, eventualmente storto.

    La rotazione e' attorno all'asse x (la profondita' dello scaffale) e ha per
    centro la mezzeria del PIEDE del libro: cosi' un volume inclinato resta
    appoggiato invece di levitare. Lo spigolo basso che scende sotto il piano
    affonda al massimo di mezza larghezza per il seno dell'inclinazione, cioe'
    ~0.013 con i valori qui sopra -- meno dello spessore del ripiano (0.05),
    quindi sparisce dentro il legno invece di bucarlo.
    """
    zc = (z_lo + z_hi) / 2
    a = math.radians(tilt_deg)
    ca, sa = math.cos(a), math.sin(a)

    def place(z, y):
        dz, dy = z - zc, y - y0
        return (zc + dz * ca - dy * sa, y0 + dz * sa + dy * ca)

    y1 = y0 + height
    flat = {}
    for kz, z in ((-1, z_lo), (1, z_hi)):
        for ky, y in ((-1, y0), (1, y1)):
            flat[(kz, ky)] = place(z, y)
    corners = {}
    for kx, x in ((-1, x0), (1, x1)):
        for kz in (-1, 1):
            for ky in (-1, 1):
                z, y = flat[(kz, ky)]
                corners[(kx, ky, kz)] = (x, y, z)

    c = ((x0 + x1) / 2, (y0 + y1) / 2, zc)
    quads = [
        [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
        [(1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)],
        [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
        [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)],
        [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)],
        [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    ]
    # UV verticali sul dorso: il ritaglio e' alto e stretto quanto il libro, cosi'
    # la venatura del legno corre per il lungo come la pelle di una costola.
    for q in quads:
        emit(faces, [corners[k] for k in q],
             [uv_in(rect, 0.1, 0.05), uv_in(rect, 0.9, 0.05),
              uv_in(rect, 0.9, 0.95), uv_in(rect, 0.1, 0.95)], c)


def cap_half_width(y):
    """Mezza larghezza del cappello alla quota y, in coordinate dell'anta.

    E' la sagoma del buco piu' CAP_MARGIN, tagliata alla larghezza dell'anta:
    sopra i 4.12 il massello non puo' essere piu' largo del fusto o sporgerebbe
    oltre il cardine e, ruotando, entrerebbe nello stipite.
    """
    if y <= HOLE_PROFILE[0][0]:
        return (SHELF_Z1 - SHELF_Z0) / 2
    for (ya, ha), (yb, hb) in zip(HOLE_PROFILE, HOLE_PROFILE[1:]):
        if y <= yb:
            t = (y - ya) / (yb - ya)
            h = ha + (hb - ha) * t
            break
    else:
        h = 0.0
    # Il buco si chiude a 4.92 ma il cappello ci arriva sopra: una punta a zero
    # sarebbe un triangolo degenere, e sopra la chiusura non c'e' piu' niente da
    # coprire, quindi si taglia piatto.
    return min(max(h + CAP_MARGIN, 0.30), (SHELF_Z1 - SHELF_Z0) / 2)


def build_shelf():
    faces = []
    rng = random.Random(BOOK_SEED)

    inner_z0, inner_z1 = SHELF_Z0 + SIDE_T, SHELF_Z1 - SIDE_T
    back_face_x = BACK_X + BACK_T

    # schienale, fianchi, zoccolo, ripiani
    y_bot = CASE_BOTTOM_Y
    box(faces, BACK_X, back_face_x, y_bot, CASE_TOP_Y, SHELF_Z0, SHELF_Z1, WOOD_UV)
    box(faces, back_face_x, FRONT_X, y_bot, CASE_TOP_Y, SHELF_Z0, inner_z0, WOOD_UV)
    box(faces, back_face_x, FRONT_X, y_bot, CASE_TOP_Y, inner_z1, SHELF_Z1, WOOD_UV)
    box(faces, back_face_x, FRONT_X, y_bot, PLINTH_Y, inner_z0, inner_z1, WOOD_DARK_UV)
    for y in BOARD_TOPS[1:] + [TOP_BOARD_Y]:
        box(faces, back_face_x, FRONT_X, y - BOARD_T, y, inner_z0, inner_z1, WOOD_UV)

    # cappello: un massello a tutta profondita' sagomato sull'arco del vano.
    # Poligono nel piano (z, y), montante sinistro in salita e destro in discesa,
    # e i due profili estrusi fra BACK_X e FRONT_X.
    ys = [CASE_TOP_Y] + [y for y, _ in HOLE_PROFILE if y > CASE_TOP_Y] + [CAP_TOP_Y]
    left = [(CENTRE_Z - cap_half_width(y), y) for y in ys]
    right = [(CENTRE_Z + cap_half_width(y), y) for y in ys]
    outline = left + list(reversed(right))
    cap_centre = (0.5 * (BACK_X + FRONT_X), (CASE_TOP_Y + CAP_TOP_Y) / 2, CENTRE_Z)
    n = len(outline)
    for i in range(n):
        z0, y0 = outline[i]
        z1, y1 = outline[(i + 1) % n]
        emit(faces,
             [(BACK_X, y0, z0), (FRONT_X, y0, z0), (FRONT_X, y1, z1), (BACK_X, y1, z1)],
             [uv_in(WOOD_UV, 0.1, i / n), uv_in(WOOD_UV, 0.9, i / n),
              uv_in(WOOD_UV, 0.9, (i + 1) / n), uv_in(WOOD_UV, 0.1, (i + 1) / n)],
             cap_centre)
    for x in (BACK_X, FRONT_X):
        emit(faces, [(x, y, z) for z, y in outline],
             [uv_in(WOOD_UV, 0.2 + 0.6 * (z - SHELF_Z0) / (SHELF_Z1 - SHELF_Z0),
                    0.2 + 0.6 * (y - CASE_TOP_Y) / (CAP_TOP_Y - CASE_TOP_Y))
              for z, y in outline],
             cap_centre)

    # i volumi. Ogni campata si riempie da -z verso +z finche' c'e' posto,
    # saltando la fessura del ripiano scelto.
    tops = BOARD_TOPS + [TOP_BOARD_Y]
    for bay, y0 in enumerate(BOARD_TOPS):
        clear = tops[bay + 1] - BOARD_T - y0
        z = inner_z0 + BOOK_GAP
        idx = 0
        while z < inner_z1 - BOOK_W_MIN - BOOK_GAP:
            # la fessura: si salta e basta, il ripiano sotto resta a vista
            if bay == BOOK_SLOT_BOARD and z < BOOK_SLOT_Z + BOOK_SLOT_W and \
               z + BOOK_W_MAX > BOOK_SLOT_Z:
                z = BOOK_SLOT_Z + BOOK_SLOT_W + BOOK_GAP
                continue
            w = rng.uniform(BOOK_W_MIN, BOOK_W_MAX)
            if bay == BOOK_SLOT_BOARD and z < BOOK_SLOT_Z and z + w > BOOK_SLOT_Z:
                w = BOOK_SLOT_Z - z - BOOK_GAP
                if w < BOOK_W_MIN:
                    z = BOOK_SLOT_Z + BOOK_SLOT_W + BOOK_GAP
                    continue
            if z + w > inner_z1 - BOOK_GAP:
                break
            h = min(rng.uniform(BOOK_H_MIN, BOOK_H_MAX), clear - 0.03)
            # profondita' variabile: nessuna libreria vera ha i tagli allineati
            depth = BOOK_DEPTH * rng.uniform(0.82, 1.0)
            spine = BOOK_SPINE_X - rng.uniform(0.0, 0.035)
            # un volume storto ogni tanto, e mai due di fila dalla stessa parte
            tilt = rng.uniform(-BOOK_TILT_MAX, BOOK_TILT_MAX) if rng.random() < 0.22 else 0.0
            tilted_book(faces, spine - depth, spine, y0, h, z, z + w,
                        sub_rect(SPINE_UVS[idx % len(SPINE_UVS)], rng), tilt)
            z += w + BOOK_GAP * rng.uniform(0.6, 2.4)
            idx += 1

    return to_object(NAME_SHELF, faces)


def build_book():
    """Il libro pickup: due piatti, il blocco delle pagine, il dorso tondo.

    Il dorso e' l'unica parte che si vede quando il libro e' nella fessura, ed e'
    anche quella che il giocatore ha in mano: sporge di BOOK_SPINE_R oltre i
    piatti ed e' spezzato in tre facce invece di una, il minimo perche' prenda
    tre valori di luce diversi e non legga come un mattone.
    """
    faces = []
    x0, x1 = 0.0, BOOK_LEN_X
    z0, z1 = -BOOK_LEN_Z / 2, BOOK_LEN_Z / 2
    y0, y1 = 0.0, BOOK_THICK

    # piatti (sotto e sopra)
    box(faces, x0, x1, y0, y0 + BOOK_COVER_T, z0, z1, RED_UV, inset=0.08)
    box(faces, x0, x1, y1 - BOOK_COVER_T, y1, z0, z1, RED_UV, inset=0.08)
    # Blocco pagine, rientrato sotto i piatti in testa, al piede e sul taglio
    # davanti, ma NON sul dorso: li' arriva fino a x0 di proposito. Facendolo
    # partire da x0 + BOOK_SPINE_R restava un vuoto fra i due piatti proprio
    # sotto la costola, aperto in testa e al piede -- e da fuori si vedeva
    # dentro il libro.
    box(faces, x0, x1 - BOOK_PAGE_INSET,
        y0 + BOOK_COVER_T, y1 - BOOK_COVER_T,
        z0 + BOOK_PAGE_INSET, z1 - BOOK_PAGE_INSET, GOLD_UV, inset=0.1)

    # dorso: tre facce sulla curva, da y0 a y1, che sporgono di BOOK_SPINE_R
    steps = 3
    prof = []
    for i in range(steps + 1):
        t = i / steps
        y = y0 + (y1 - y0) * t
        # semicerchio schiacciato: massimo aggetto a meta' spessore
        prof.append((x0 - BOOK_SPINE_R * math.sin(math.pi * t), y))
    spine_centre = (x0 + 0.05, (y0 + y1) / 2, 0.0)
    for i in range(steps):
        xa, ya = prof[i]
        xb, yb = prof[i + 1]
        emit(faces, [(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)],
             [uv_in(RED_UV, 0.05 + 0.9 * i / steps, 0.05),
              uv_in(RED_UV, 0.05 + 0.9 * (i + 1) / steps, 0.05),
              uv_in(RED_UV, 0.05 + 0.9 * (i + 1) / steps, 0.95),
              uv_in(RED_UV, 0.05 + 0.9 * i / steps, 0.95)], spine_centre)
    # Tappi del dorso alle due teste: la falce fra l'arco e il piano x = x0.
    # Solo i punti dell'arco -- prof PARTE gia' da (x0, y0) e FINISCE su
    # (x0, y1), quindi il lato dritto e' il segmento di chiusura del poligono.
    # Aggiungerlo a mano rifarebbe due vertici doppi, che remove_doubles fonde
    # lasciando una faccia degenere: e' esattamente il buco che si vedeva sul
    # dorso.
    for z, ref in ((z0, (x0, (y0 + y1) / 2, z0 - 1.0)), (z1, (x0, (y0 + y1) / 2, z1 + 1.0))):
        emit(faces, [(x, y, z) for x, y in prof],
             [uv_in(RED_UV, 0.5, 0.5)] * len(prof), ref, toward=True)

    return to_object(NAME_BOOK, faces)


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


def gltf_bbox(obj):
    """bbox in coordinate glTF: Blender (x, y, z) -> glTF (x, z, -y)."""
    me = obj.data
    xs = [v.co[0] for v in me.vertices]
    ys = [v.co[2] for v in me.vertices]
    zs = [-v.co[1] for v in me.vertices]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def check_shelf(obj):
    """UV dentro l'atlante, ingombro dell'anta, e -- la sola che conta -- che il
    cappello copra davvero il vano a ogni quota della sagoma."""
    uvs = [l.uv for l in obj.data.uv_layers[0].data]
    assert all(-1e-6 <= u <= 1 + 1e-6 and -1e-6 <= v <= 1 + 1e-6 for u, v in uvs), \
        "UV fuori dalla texture"
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = gltf_bbox(obj)
    assert zmin >= SHELF_Z0 - 1e-4 and zmax <= SHELF_Z1 + 1e-4, \
        "%s sborda in larghezza (z %.3f..%.3f)" % (obj.name, zmin, zmax)
    assert xmin >= BACK_X - 1e-4 and xmax <= FRONT_X + 1e-4, \
        "%s sborda in profondita' (x %.3f..%.3f)" % (obj.name, xmin, xmax)
    for y, half in HOLE_PROFILE:
        if half <= 0.0:
            continue
        cap = cap_half_width(y)
        assert HOLE_CENTRE_Z - half >= CENTRE_Z - cap - 1e-6 and \
               HOLE_CENTRE_Z + half <= CENTRE_Z + cap + 1e-6, \
            "il cappello scopre il vano a y %.3f (%.3f < %.3f)" % (y, cap, half)
    assert ymax >= HOLE_PROFILE[-1][0], \
        "l'anta non arriva in cima al vano (%.3f < %.3f)" % (ymax, HOLE_PROFILE[-1][0])
    report(obj)


def check_book(obj):
    uvs = [l.uv for l in obj.data.uv_layers[0].data]
    assert all(-1e-6 <= u <= 1 + 1e-6 and -1e-6 <= v <= 1 + 1e-6 for u, v in uvs), \
        "UV fuori dalla texture"
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = gltf_bbox(obj)
    # Deve entrare nella fessura in piedi: lo spessore va nella larghezza del
    # vuoto, l'altezza delle pagine nella luce della campata.
    thickness = ymax - ymin
    height = zmax - zmin
    clear = BOARD_TOPS[BOOK_SLOT_BOARD + 1] - BOARD_T - BOARD_TOPS[BOOK_SLOT_BOARD]
    assert thickness <= BOOK_SLOT_W - 1e-3, \
        "il libro non entra nella fessura (%.3f > %.3f)" % (thickness, BOOK_SLOT_W)
    assert height <= clear - 1e-3, \
        "il libro non sta nella campata (%.3f > %.3f)" % (height, clear)
    report(obj)


def report(obj):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = gltf_bbox(obj)
    print("%-20s %5d facce  bbox x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f" %
          (obj.name, len(obj.data.polygons), xmin, xmax, ymin, ymax, zmin, zmax))


def export(objs):
    out_dir = os.path.join(tempfile.gettempdir(), "cg_bookshelf_export")
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
    shelf = build_shelf()
    book = build_book()
    check_shelf(shelf)
    check_book(book)
    out = export([shelf, book])
    print("\nesportato in %s" % out)
    print("fessura: ripiano y %.3f, z %.3f..%.3f, dorsi a x %.3f"
          % (BOARD_TOPS[BOOK_SLOT_BOARD], BOOK_SLOT_Z, BOOK_SLOT_Z + BOOK_SLOT_W,
             BOOK_SPINE_X))
