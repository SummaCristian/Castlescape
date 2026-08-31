#!/usr/bin/env python3
"""
Genera SM_Bookshelf_01.png: l'atlante della porta con, in piu', i dorsi dei
libri.

--- Perche' una texture nuova e non una del kit ---

I dorsi erano quattro ritagli di SM_Door_01.png (noce chiaro, noce scuro, ferro,
pietra) e il risultato era esattamente quello che sembrava: una parete di libri
di legno. Il difetto non era la scelta dei ritagli, e' che nel kit non c'e' un
altro colore da pescare. Misurando la saturazione di tutte e sedici le texture
di Dracula, tredici stanno su un'unica tinta (il marrone del legno o il grigio
della pietra) e le uniche due che portano altro -- SM_CastleBanners_01 e
SM_Carpet_01 -- portano SOLO il cremisi e l'oro degli stendardi, senza un
centimetro di legno. E la libreria il legno ce l'ha per forza: e' un'anta, e
deve essere la stessa roba delle porte del dungeon o si capisce da lontano che
nasconde qualcosa (il ragionamento sta in make_bookshelf.py).

Il loader di Starter.hpp concatena tutte le primitive di un modello in un solo
vertex buffer con UNA sola texture, quindi "legno del kit + dorsi colorati" non
puo' venire da due immagini. Deve venire da una immagine che le ha entrambe, e
quella immagine non esiste: si fa.

--- Perche' una COPIA dell'atlante della porta ---

Il fusto della libreria (schienale, fianchi, ripiani, zoccolo, cappello) resta
mappato dove era: sul noce di SM_Door_01.png. Ripartendo da una copia bit a bit
di quell'atlante, tutte le UV gia' scritte in make_bookshelf.py -- WOOD_UV,
WOOD_DARK_UV, IRON_UV, STONE_UV -- restano valide senza toccare un numero, e la
libreria continua a essere legno identico a quello delle porte perche' e'
letteralmente lo stesso legno. Cambiano solo i dorsi, che vanno a pescare in un
angolo dell'atlante che prima era nero.

L'angolo non e' scelto a occhio: rasterizzando l'alpha... anzi, l'atlante e'
opaco, quindi si misura sul nero. Il rettangolo libero piu' grande in alto a
destra e' x 902..1024, y 0..347; il pannello qui sotto ci sta dentro con
margine. Se il kit venisse aggiornato con un atlante piu' pieno, rimisurarlo.

SM_Door_01.png NON viene modificata: la usano anche le ante, i muri con la porta
e il muro col vano, e ripassarci sopra dei dorsi colorati vorrebbe dire
colorare anche loro.

--- I colori ---

Otto tinte da biblioteca vecchia, non otto colori a caso: pelle sanguigna,
tela verde, indaco, ocra, pergamena, bordeaux, ardesia, prugna. Sono tenute
scure e poco sature apposta -- la stanza e' illuminata da una torcia e da un
paio di candele, e un dorso alla saturazione piena in quella luce diventa una
macchia fluorescente accanto alla pietra. La pergamena e' l'unica chiara, e c'e'
per lo stesso motivo per cui c'era la pietra prima: servono due o tre volumi che
prendano la luce, o la libreria in penombra legge come un rettangolo nero.

Uso:
    python tools/make_bookshelf_texture.py
"""

import os

import numpy as np
from PIL import Image

SRC = "skeleton/source/assets/textures/Dracula/SM_Door_01.png"
DST = "skeleton/source/assets/textures/Dracula/SM_Bookshelf_01.png"

# Il pannello dei dorsi, in pixel sull'atlante da 1024. Queste quattro quote e
# la griglia qui sotto stanno anche in make_bookshelf.py (SPINE_PANEL,
# SPINE_COLS, SPINE_ROWS): sono le due meta' della stessa cosa, si cambiano
# insieme.
PANEL = (908, 8, 1020, 336)     # x0, y0, x1, y1  -> 112 x 328
COLS, ROWS = 4, 2               # otto celle da 28 x 164, alte e strette come
                                # un dorso vero (rapporto ~1:6)
CELL_MARGIN = 2                 # texel di guardia: le UV pescano finestre a
                                # caso dentro la cella, non deve mai capitarne
                                # una a cavallo di due tinte

# (nome, colore base) -- vedi il perche' dei toni bassi nel commento in testa
SPINES = [
    ("pelle sanguigna", (112, 42, 38)),
    ("tela verde",      (48, 80, 56)),
    ("indaco",          (46, 56, 94)),
    ("ocra",            (150, 116, 50)),
    ("pergamena",       (196, 182, 148)),
    ("bordeaux",        (80, 30, 40)),
    ("ardesia",         (48, 76, 84)),
    ("prugna",          (90, 52, 98)),
]

GRAIN = 9.0        # ampiezza del rumore per texel: la grana della tela/pelle
FIBRE = 0.10       # ampiezza delle striature verticali, in frazione del colore
CORDS = 4          # nervi del dorso: le costole orizzontali della rilegatura
SEED = 20260831


def spine_swatch(w, h, base, rng):
    """Un dorso: colore base, fibra verticale, grana, nervi, bordi in ombra.

    Tutto quello che c'e' dentro deve reggere l'unica cosa che la mesh ne fa,
    cioe' pescarci dentro una finestra rettangolare a caso larga il 45%
    (sub_rect in make_bookshelf.py). Percio' niente disegni, niente scritte,
    niente etichette: solo variazione che resti plausibile comunque la si
    ritagli. I nervi sono orizzontali e ripetuti apposta -- una finestra che ne
    prende uno e mezzo sembra ancora un dorso rilegato.
    """
    a = np.empty((h, w, 3), np.float64)
    a[:] = np.asarray(base, np.float64)

    # fibra: colonne leggermente piu' chiare/scure, costanti in altezza, come la
    # trama di una tela tirata sul cartone
    fib = 1.0 + FIBRE * (rng.random(w) - 0.5)[None, :, None]
    a *= fib

    # nervi: bande scure con il filo di luce sopra, dove la cucitura alza la
    # pelle. Sinusoide e non righe nette: a questa risoluzione una riga da un
    # texel sparisce nel mip appena il libro si allontana.
    y = np.arange(h)[:, None, None] / h
    a *= 1.0 + 0.10 * np.sin(2.0 * np.pi * CORDS * y - 0.6)

    # bordi in ombra: il dorso e' tondo, i due lati vanno verso i piatti
    x = np.arange(w)[None, :, None] / max(w - 1, 1)
    a *= 0.80 + 0.20 * np.sin(np.pi * np.clip(x, 0.0, 1.0)) ** 0.5

    a += GRAIN * (rng.random((h, w, 3)) - 0.5)
    return np.clip(a, 0, 255)


def main():
    src = Image.open(SRC).convert("RGB")
    out = np.asarray(src).astype(np.float64).copy()

    x0, y0, x1, y1 = PANEL
    assert out[y0:y1, x0:x1].max() <= 8, \
        "il pannello dei dorsi cade su una zona gia' usata dell'atlante"
    cw, ch = (x1 - x0) // COLS, (y1 - y0) // ROWS

    rng = np.random.default_rng(SEED)
    for i, (name, base) in enumerate(SPINES):
        cx = x0 + (i % COLS) * cw
        cy = y0 + (i // COLS) * ch
        m = CELL_MARGIN
        # il margine di guardia prende il colore base piatto: se una finestra
        # ci finisce sopra vede comunque la tinta giusta, non il nero di sotto
        out[cy:cy + ch, cx:cx + cw] = np.asarray(base, np.float64)
        out[cy + m:cy + ch - m, cx + m:cx + cw - m] = \
            spine_swatch(cw - 2 * m, ch - 2 * m, base, rng)
        print("  %-16s %3d,%3d  %dx%d  rgb%s" % (name, cx, cy, cw, ch, base))

    img = Image.fromarray(out.round().astype(np.uint8))
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    img.save(DST, "PNG", optimize=True)
    print("scritta %s  (%d KB)" % (DST, os.path.getsize(DST) // 1024))
    print("pannello dorsi: x %d..%d  y %d..%d  griglia %dx%d celle %dx%d"
          % (x0, x1, y0, y1, COLS, ROWS, cw, ch))


if __name__ == "__main__":
    main()
