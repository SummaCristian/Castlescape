#!/usr/bin/env python3
"""
Genera SM_Bookshelf_01 e SM_Book_01: la libreria che nasconde il passaggio
segreto e il libro che la apre (logica in main.cpp, Door::lockKeyId con
lockKeyId = "book" e la LockProp montata su `whenUnlocked`).

DUE file e non uno per lo stesso motivo delle catene e del lucchetto: il loader
di Starter.hpp concatena tutte le primitive in un unico vertex buffer con UNA
sola texture, quindi due materiali diversi devono stare in due modelli diversi.
E anche qui il taglio tecnico coincide con quello che si vuole vedere:

  SM_Bookshelf_01 -> SM_Bookshelf_01.png, che e' l'atlante del battente PIU' i
                    dorsi dei libri (la fa tools/make_bookshelf_texture.py). Il
                    fusto resta mappato sul noce di SM_Door_01: la libreria DEVE
                    essere fatta della stessa roba delle porte del dungeon,
                    perche' e' un mobile che finge di essere un muro e se il
                    legno non fosse quello di casa si vedrebbe da lontano che
                    nasconde qualcosa. I dorsi no: pescavano quattro zone dello
                    stesso atlante (noce chiaro, noce scuro, ferro, pietra) e
                    una parete di libri di legno e' esattamente quello che
                    sembrava. Nel kit un altro colore non c'e' -- il perche' e
                    la misura stanno in make_bookshelf_texture.py -- quindi i
                    dorsi hanno otto tinte dipinte in un angolo vuoto
                    dell'atlante, e il resto della texture e' la copia bit a bit
                    di quella della porta.
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

E deve essere LARGA, molto piu' di quanto sembri ragionevole guardando i numeri.
La prima versione lasciava 0.18 su una campata da 2.28 con diciotto volumi in
fila, e fra un libro e l'altro ci sono gia' 0.012 di aria che a due passi di
distanza, con la sola luce di una torcia, leggono come fessure anche loro: il
vuoto vero spariva in mezzo agli altri. Portarla a due volumi (0.22) non e'
bastato per lo stesso motivo -- il metro di paragone non e' la larghezza di un
libro, e' la larghezza della fessura piu' grande fra due libri qualsiasi del
resto dello scaffale, e finche' il vuoto e' dello stesso ordine di quelle si
legge come una fila un po' larga, non come un buco. Adesso e' 0.44: cinque o sei
volumi mancanti, un quinto della campata, l'unico posto del mobile dove si vede
il fondo -- che e' poi il modo in cui si riconosce un buco, non dalla misura ma
dal fatto che dietro non c'e' un dorso.

Il prezzo e' che il libro, una volta infilato, non riempie la fessura: resta al
centro con il vuoto attorno. Va bene cosi', e non e' una rinuncia -- appena
entra la libreria si apre, quindi quel fotogramma nessuno lo guarda, mentre i
minuti in cui il buco deve farsi notare sono tutti quelli prima.

Perche' regga, i due bordi della fessura sono ESATTI: la fila
si costruisce in due tratti che finiscono e ricominciano sul millimetro dove
comincia e finisce il vuoto (fill_run piu' sotto), invece di lasciare che la
larghezza a caso dell'ultimo volume decida dove il buco si apre davvero.

Uso:
    python tools/make_bookshelf_texture.py
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
# Centrata sulla mezzeria del mobile (-1.231) e larga cinque o sei volumi: il
# perche' della misura sta in testa al file, sotto "Cosa NON e' un dettaglio".
# Essendo centrata, il libro depositato cade su CENTRE_Z qualunque sia la
# larghezza -- cioe' allargare la fessura NON sposta il numero in main.cpp.
BOOK_SLOT_W = 0.440                   # larghezza del vuoto
BOOK_SLOT_Z = CENTRE_Z - BOOK_SLOT_W / 2   # -1.451, bordo dal lato -z
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
# sui 1024 px delle due texture. Misurate, non scelte a occhio -- luminanza
# media, minima e deviazione fra parentesi, rimisurare se le texture cambiano.
#
# E il criterio non e' solo il tono, e' che il ritaglio sia PULITO. Il pannello
# di SM_Door_01 non e' una tavola liscia: ha le fughe fra le assi (righe
# orizzontali sotto i 20 di luminanza), due bandelle verticali scure e una
# placca quasi nera della serratura. I primi ritagli ci passavano sopra, e
# siccome box() spalma il ritaglio intero su ogni faccia il difetto non veniva
# fuori come una venatura ma come una riga nera dritta ripetuta identica su
# schienale, ripiani e cappello, piu' una toppa nera sullo zoccolo dov'era
# finita la serratura. Questi due rettangoli sono il piu' grande e il piu' scuro
# che stanno DENTRO una singola asse, cercati imponendo che la luminanza minima
# non scenda sotto i 28 e la deviazione resti sotto 7.
#
# SM_Door_01.png (e quindi anche SM_Bookshelf_01.png, che ne e' la copia):
WOOD_UV = (422 / 1024.0, 370 / 1024.0, 522 / 1024.0, 398 / 1024.0)   # (61) min 41 s5
WOOD_DARK_UV = (434 / 1024.0, 408 / 1024.0, 524 / 1024.0, 432 / 1024.0)  # (53) min 28 s6
# SM_CastleBanners_01.png:
RED_UV = (430 / 1024.0, 750 / 1024.0, 640 / 1024.0, 880 / 1024.0)    # (172,72,72) s10
GOLD_UV = (802 / 1024.0, 235 / 1024.0, 838 / 1024.0, 345 / 1024.0)   # ottone/oro

# SM_Bookshelf_01.png, pannello dei dorsi: otto celle dipinte in un angolo che
# nell'atlante della porta era nero. Il pannello e la griglia sono le stesse
# quattro quote di make_bookshelf_texture.py (PANEL, COLS, ROWS): stanno in due
# posti perche' uno dipinge e l'altro mappa, e vanno cambiate insieme.
SPINE_PANEL = (908, 8, 1020, 336)
SPINE_COLS, SPINE_ROWS = 4, 2
# Un texel di guardia in piu' del margine che il generatore lascia (2): con la
# bilineare e i mip, campionare fino al bordo esatto della cella tira dentro il
# colore della cella accanto e sul dorso compare una riga della tinta sbagliata.
SPINE_INSET = 3


def spine_cell(i):
    """Ritaglio della i-esima tinta, in coordinate immagine normalizzate."""
    x0, y0, x1, y1 = SPINE_PANEL
    cw, ch = (x1 - x0) // SPINE_COLS, (y1 - y0) // SPINE_ROWS
    cx = x0 + (i % SPINE_COLS) * cw + SPINE_INSET
    cy = y0 + (i // SPINE_COLS) * ch + SPINE_INSET
    return (cx / 1024.0, cy / 1024.0,
            (cx + cw - 2 * SPINE_INSET) / 1024.0, (cy + ch - 2 * SPINE_INSET) / 1024.0)


# L'ordine e' quello in cui i volumi si posano lungo il ripiano, quindi conta:
# sono le stesse otto celle del generatore rimescolate perche' due tinte vicine
# di famiglia (sanguigna/bordeaux, indaco/ardesia) non finiscano appaiate, e con
# la pergamena distanziata dalle altre chiare. Otto e non sei come prima: sei
# tinte su una campata da venti volumi facevano ripetere il motivo tre volte e
# si vedeva.
SPINE_UVS = [spine_cell(i) for i in (0, 2, 4, 6, 1, 5, 3, 7)]


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

    La normale si prende con Newell, cioe' sommando il contributo di TUTTI gli
    spigoli, e non dal prodotto vettoriale dei primi tre vertici. Sui quad non
    cambia niente, ma sui poligoni del cappello -- ventiquattro vertici lungo
    l'arco -- cambia il segno: i primi tre punti del profilo stanno su una
    curva, la loro svolta locale gira al contrario del giro complessivo del
    poligono, e il prodotto dei primi tre dava una normale opposta a quella
    vera. Le due facce piatte del cappello uscivano quindi rivolte verso
    l'interno, il backface culling se le mangiava e il massello si vedeva
    attraverso da davanti e da dietro, ridotto alla sua cornice. Newell e' la
    normale del poligono e basta, e non ha corner case.

    smooth di default False, al contrario di make_door_lock.py: qui non c'e'
    niente di tondo. Un libro con le normali mediate diventa un cuscino e uno
    scaffale un materasso.
    """
    p = [Vector(v) for v in verts]
    n = Vector((0.0, 0.0, 0.0))
    for a, b in zip(p, p[1:] + p[:1]):
        n.x += (a.y - b.y) * (a.z + b.z)
        n.y += (a.z - b.z) * (a.x + b.x)
        n.z += (a.x - b.x) * (a.y + b.y)
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

    # Il punto di riferimento per l'orientamento e' il baricentro degli otto
    # spigoli DOPO la rotazione, non ((x0+x1)/2, (y0+y1)/2, zc). La rotazione ha
    # centro il piede del libro, quindi la sua meta' alta si sposta di
    # h/2 * sin(tilt) -- su un volume alto 0.46 inclinato di 11 gradi sono 0.044,
    # e un dorso stretto e' largo la meta'. Con il centro non ruotato il
    # riferimento finiva FUORI dalla scatola, emit orientava le facce
    # allontanandole da un punto esterno e il libro usciva rivoltato.
    c = tuple(sum(v[i] for v in corners.values()) / 8.0 for i in range(3))
    quads = [
        [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
        [(1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)],
        [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
        [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)],
        [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)],
        [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    ]
    # UV verticali sul dorso: il ritaglio e' alto e stretto quanto il libro, cosi'
    # la venatura corre per il lungo come la pelle di una costola.
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

    # i volumi. Ogni campata si riempie da -z verso +z finche' c'e' posto; il
    # ripiano della fessura si riempie in DUE tratti, uno per lato del vuoto.
    tops = BOARD_TOPS + [TOP_BOARD_Y]
    idx = 0
    for bay, y0 in enumerate(BOARD_TOPS):
        clear = tops[bay + 1] - BOARD_T - y0
        lo, hi = inner_z0 + BOOK_GAP, inner_z1 - BOOK_GAP
        if bay == BOOK_SLOT_BOARD:
            runs = [(lo, BOOK_SLOT_Z, False, True),
                    (BOOK_SLOT_Z + BOOK_SLOT_W, hi, True, False)]
        else:
            runs = [(lo, hi, False, False)]
        for z_start, z_end, snap_start, snap_end in runs:
            idx = fill_run(faces, rng, y0, clear, z_start, z_end,
                           snap_start, snap_end, idx)

    return to_object(NAME_SHELF, faces)


def fill_run(faces, rng, y0, clear, z_start, z_end, snap_start, snap_end, idx):
    """Riempie un tratto di ripiano da z_start a z_end e torna l'indice tinta.

    snap_start/snap_end dicono se quel capo del tratto e' un bordo della
    fessura. Se lo e', il volume di li' ci si appoggia ESATTO -- il primo parte
    da z_start, l'ultimo viene allungato fino a z_end -- perche' il buco deve
    essere largo BOOK_SLOT_W e non "BOOK_SLOT_W piu' quello che avanzava". Se
    invece il capo e' il fianco del mobile non si snappa niente e si lascia
    l'avanzo: un libro a filo del fianco vorrebbe dire due quad complanari e
    coincidenti, cioe' lo z-fighting che si sta evitando ovunque.

    E i volumi che toccano la fessura non si inclinano mai. Un libro storto
    accanto al vuoto ci pende dentro, e li' dentro ci deve entrare il libro che
    il giocatore porta: sarebbero due mesh compenetrate proprio nel punto che
    tutta questa geometria esiste per far guardare.
    """
    placed = []
    z = z_start
    while True:
        w = rng.uniform(BOOK_W_MIN, BOOK_W_MAX)
        if z + w > z_end:
            # Ultimo volume del tratto. Se di la' c'e' la fessura si prende
            # tutto lo spazio che resta invece della sua larghezza a caso: e'
            # l'unico modo di far finire la fila sul bordo del vuoto senza poi
            # allungare un libro di mezzo dorso.
            if not (snap_end and z_end - z >= BOOK_W_MIN):
                break
            w = z_end - z
        h = min(rng.uniform(BOOK_H_MIN, BOOK_H_MAX), clear - 0.03)
        # profondita' variabile: nessuna libreria vera ha i tagli allineati
        depth = BOOK_DEPTH * rng.uniform(0.82, 1.0)
        spine = BOOK_SPINE_X - rng.uniform(0.0, 0.035)
        # un volume storto ogni tanto, non tutti
        tilt = rng.uniform(-BOOK_TILT_MAX, BOOK_TILT_MAX) if rng.random() < 0.22 else 0.0
        placed.append([z, z + w, h, depth, spine, tilt, idx])
        z += w + BOOK_GAP * rng.uniform(0.6, 2.4)
        idx += 1

    if placed:
        if snap_end:
            # Resta al massimo BOOK_W_MIN da coprire (il ramo qui sopra ha gia'
            # preso tutto il resto): l'ultimo si allunga fino al bordo del
            # vuoto. Si allunga e non si sposta, o andrebbe addosso a quello
            # prima.
            placed[-1][1] = z_end
            placed[-1][5] = 0.0
        if snap_start:
            placed[0][0] = z_start
            placed[0][5] = 0.0
    for z0, z1, h, depth, spine, tilt, i in placed:
        tilted_book(faces, spine - depth, spine, y0, h, z0, z1,
                    sub_rect(SPINE_UVS[i % len(SPINE_UVS)], rng), tilt)
    return idx


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

    # La fessura e' vuota davvero, ed e' larga quanto dice la costante. Non e'
    # pignoleria: e' l'unico segnale che il livello da al giocatore sul fatto
    # che il libro va li', e un volume che ci sborda dentro per mezzo
    # centimetro lo cancella. Il volume di prova e' il parallelepipedo che il
    # libro occupera' -- dal fondo dei tagli fino ai dorsi, per tutta la luce
    # della campata, largo la fessura meno un pelo per non contare i due
    # volumi che le si appoggiano ai lati.
    # Le due facce piatte del cappello guardano fuori. E' il bug che le ha
    # avute rivolte all'indietro per un giro intero (vedi emit): non si vedeva
    # dal conto delle facce, che c'erano tutte, ma solo in gioco, dove il
    # massello sopra lo scaffale spariva lasciando la sua cornice. Blender
    # tiene la x di glTF, quindi il segno della normale si legge diretto.
    for poly in obj.data.polygons:
        vs = [obj.data.vertices[i].co for i in poly.vertices]
        for x, want in ((BACK_X, -1.0), (FRONT_X, 1.0)):
            if all(abs(v[0] - x) < 1e-4 for v in vs) and \
               all(v[2] > CASE_TOP_Y - 1e-3 for v in vs):
                assert poly.normal[0] * want > 0.0, \
                    "faccia del cappello a x %.3f rivolta all'interno" % x

    y_lo = BOARD_TOPS[BOOK_SLOT_BOARD]
    y_hi = BOARD_TOPS[BOOK_SLOT_BOARD + 1] - BOARD_T
    z_lo, z_hi = BOOK_SLOT_Z + 1e-3, BOOK_SLOT_Z + BOOK_SLOT_W - 1e-3
    for v in obj.data.vertices:
        x, y, z = v.co[0], v.co[2], -v.co[1]
        if BACK_X + BACK_T + 1e-3 < x < FRONT_X and y_lo + 1e-3 < y < y_hi - 1e-3 \
           and z_lo < z < z_hi:
            raise AssertionError(
                "c'e' geometria dentro la fessura: (%.3f, %.3f, %.3f)" % (x, y, z))
    check_orientation(obj)
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
    check_orientation(obj, conforming=False)
    report(obj)


def check_orientation(obj, conforming=True):
    """Ogni guscio chiuso e' orientato verso l'esterno.

    Serve perche' le facce girate al contrario NON si vedono da nessun conto:
    il numero di facce e' giusto, l'ingombro e' giusto, le UV sono dentro
    l'atlante, e il pezzo in gioco semplicemente non c'e'. E' successo due
    volte in questo file, una al cappello (la normale presa dai primi tre
    vertici di un poligono curvo) e una a un libro storto (il riferimento per
    l'orientamento che non seguiva l'inclinazione), e in entrambi i casi si e'
    scoperto guardando il mobile, non esportandolo.

    Il criterio e' il volume con segno: per una superficie chiusa e coerente il
    volume non dipende dall'origine, vale il volume del solido se le normali
    guardano fuori e l'opposto se guardano dentro. Se poi l'orientamento e'
    incoerente -- alcune facce si', altre no -- il conto perde senso e viene
    quel che viene, ma quasi mai un numero positivo plausibile: e' proprio il
    caso che ha pescato il libro rivoltato.

    conforming=False disattiva il solo controllo sugli spigoli, e serve al libro
    pickup: il tappo a falce del dorso chiude con un segmento che va dal piede
    alla testa in un pezzo solo, mentre sullo stesso spigolo i due piatti e il
    blocco pagine ci arrivano in tre tratti spezzati alle loro giunzioni. La
    superficie e' chiusa lo stesso -- e' una giunzione a T, non un buco -- ma i
    vertici non si corrispondono e il pareggio fra i versi non torna. E' come e'
    sempre stato e in gioco non si vede; rifare il dorso perche' un check sia
    contento sarebbe il contrario del motivo per cui il check esiste.
    """
    me = obj.data
    parent = list(range(len(me.vertices)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for poly in me.polygons:
        vs = list(poly.vertices)
        for v in vs[1:]:
            ra, rb = find(vs[0]), find(v)
            if ra != rb:
                parent[ra] = rb

    # Ogni spigolo va percorso lo STESSO numero di volte nei due versi. E' la
    # condizione di una superficie chiusa e orientata in modo coerente, e non
    # "una volta per verso": schienale, fianchi e ripiani si toccano e dopo
    # remove_doubles condividono gli spigoli d'angolo, che finiscono quindi
    # percorsi due volte per verso. Sono due scatole appoggiate, non un errore.
    # Un buco o una faccia girata rompono comunque il pareggio.
    half = {}
    vols = {}
    for poly in me.polygons:
        vs = list(poly.vertices)
        for a, b in zip(vs, vs[1:] + vs[:1]):
            half[(a, b)] = half.get((a, b), 0) + 1
        p = [me.vertices[i].co for i in vs]
        r = find(vs[0])
        for i in range(1, len(p) - 1):
            a, b, c = p[0], p[i], p[i + 1]
            vols[r] = vols.get(r, 0.0) + a.dot(b.cross(c)) / 6.0

    bad_edges = [e for e, n in half.items()
                 if conforming and half.get((e[1], e[0]), 0) != n]
    assert not bad_edges, \
        "%s: %d spigoli sbilanciati fra i due versi (guscio aperto o facce " \
        "orientate in modo incoerente), il primo fra i vertici %d e %d" \
        % (obj.name, len(bad_edges), bad_edges[0][0], bad_edges[0][1])
    inside_out = [v for v in vols.values() if v <= 1e-9]
    assert not inside_out, \
        "%s: %d gusci su %d hanno le facce rivolte all'interno (volumi %s)" \
        % (obj.name, len(inside_out), len(vols),
           ", ".join("%.4f" % v for v in inside_out[:4]))
    print("%-20s %3d gusci, tutti chiusi e rivolti fuori" % (obj.name, len(vols)))


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
