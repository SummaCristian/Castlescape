#!/usr/bin/env python3
"""
Genera SM_StoneCeiling_01.png ritagliandola dall'atlante del muro.

Il soffitto e' l'unico pezzo strutturale modellato da noi e non ha una texture
nel kit. Riusarne una esistente non basta: mappandolo dentro l'atlante del muro
la pietra e' quella giusta, ma resta piu' chiaro della parete. Il motivo e' che
la parete non e' uniforme -- la sua superficie visibile ha luminanza media 50,
mentre le facce piane dei suoi blocchi stanno a 63, e la differenza sono le
fughe e i fondi degli archi, che hanno l'occlusione ambientale cotta dentro la
texture. Il soffitto, che di quelle cavita' non ne ha, finisce inevitabilmente
sul valore alto.

E nell'atlante del muro non c'e' una via d'uscita: l'unica zona piana grande
abbastanza per una piastrella da 7.2 e' quella a 63, e le uniche zone piu'
scure sono il campo di pietrame in basso, con un disegno molto piu' grosso e
irregolare dei mattoni regolari che si vogliono sopra la testa.

Quindi: si ritaglia il campo di mattoni regolari del muro e lo si scurisce fino
alla media della parete. E' la stessa pietra, con lo stesso disegno, portata al
tono giusto.

Come la texture del pavimento, non e' affiancabile: due piastrelle adiacenti
ripetono la stessa immagine e il giunto si vede, esattamente come succede gia'
al pavimento del dungeon. Renderla senza cuciture vorrebbe dire inventare
pixel che nel kit non ci sono.

Uso:
    python tools/make_ceiling_texture.py
"""

import os

import numpy as np
from PIL import Image

SRC = "skeleton/source/assets/textures/Dracula/SM_WallStraight_01.png"
DST = "skeleton/source/assets/textures/Dracula/SM_StoneCeiling_01.png"

# Il pannello di muro liscio, misurato sul quad piatto di SM_WallStraight_01:
# quel quad mappa y [0,6.19] e z [0,7.2] esattamente su questo rettangolo.
RECT  = (0.0482, 0.2749, 0.3012, 0.5690)   # u0, v0, u1, v1
INSET = 4          # texel scartati sui bordi: attorno ci sono arco e pilastri
SIZE  = 512        # lato dell'immagine finale
TARGET = 50.0      # luminanza del piano di fondo del muro, misurata per area


def main():
    src = Image.open(SRC).convert("RGB")
    W, H = src.size
    x0 = round(RECT[0] * W) + INSET
    y0 = round(RECT[1] * H) + INSET
    x1 = round(RECT[2] * W) - INSET
    y1 = round(RECT[3] * H) - INSET

    # quadrato piu' grande dentro il pannello, centrato sul lato lungo
    side = min(x1 - x0, y1 - y0)
    x0 += (x1 - x0 - side) // 2
    y0 += (y1 - y0 - side) // 2
    crop = src.crop((x0, y0, x0 + side, y0 + side))

    a = np.asarray(crop).astype(np.float64)
    before = a.mean()
    # moltiplicazione, non gamma: si vuole la stessa pietra meno chiara, e il
    # prodotto conserva i rapporti fra i toni invece di schiacciarne il contrasto
    a = np.clip(a * (TARGET / before), 0, 255)

    out = Image.fromarray(a.round().astype(np.uint8)).resize(
        (SIZE, SIZE), Image.LANCZOS)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    out.save(DST, "PNG", optimize=True)

    after = np.asarray(out).astype(np.float64).mean()
    print("ritaglio %dx%d da (%d,%d), fattore %.3f" % (side, side, x0, y0, TARGET / before))
    print("luminanza %.1f -> %.1f  (obiettivo %.1f)" % (before, after, TARGET))
    print("scritta %s  (%d KB)" % (DST, os.path.getsize(DST) // 1024))


if __name__ == "__main__":
    main()
