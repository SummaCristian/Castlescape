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

Portarlo al tono giusto IN MEDIA pero' non basta, e questa e' la seconda cosa
che il ritaglio si porta dietro dalla parete: la luminanza non e' uniforme.
Sulla parete l'occlusione ambientale e' cotta dentro la texture, quindi verso la
lesena la pietra e' piu' scura, e nel ritaglio questo diventa un gradiente da
40 a 58 -- il 45% -- da un bordo all'altro. Su un muro quel gradiente e'
informazione; su una piastrella di soffitto che si ripete ogni 7.2 m e' un
errore, e si vede: ogni piastrella e' chiara da un lato e scura dall'altro, e
al passaggio da una all'altra il chiaro va contro lo scuro. La luce sul
soffitto la fanno le torce, non un'ombra dipinta buona per un altro pezzo.
Quindi il gradiente si toglie -- vedi flatten.

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
DEG = 3            # grado del polinomio con cui si modella il gradiente


def bands(l, n=8):
    """Luminanza media per fascia sui due assi: il gradiente si legge qui."""
    k = l.shape[0] // n
    return ([l[:, i * k:(i + 1) * k].mean() for i in range(n)],
            [l[i * k:(i + 1) * k].mean() for i in range(n)])


def flatten(a):
    """Toglie il gradiente d'ambiente, lasciando il disegno della pietra.

    Si DIVIDE per il gradiente, non lo si sottrae: la texture e' un albedo e
    l'occlusione ci e' entrata moltiplicando, quindi solo il quoziente conserva
    i rapporti fra i toni -- una fuga resta scura quanto lo era rispetto al suo
    concio, invece di diventare piu' chiara dove si schiarisce il fondo.

    Il modello e' un polinomio di terzo grado nelle due coordinate. Il grado non
    e' a caso: e' il piu' basso che riporta i due bordi opposti allo stesso tono
    -- ed e' quello il punto, perche' e' li' che due piastrella si toccano --
    mentre con una quadrica restava un dislivello dell'8% fra bordo e bordo,
    l'ombra della lesena non essendo una rampa dritta ma una rampa con un
    ginocchio. Piu' su non si va: i conci sono larghi un sesto del ritaglio, e
    un polinomio che cominci a seguirli li spiana, mentre e' proprio il loro
    chiaroscuro quello che si vuole tenere. Il gradiente scende dal 36% al 7%,
    che e' quanto varia la pietra da sola: il residuo sono conci piu' scuri di
    altri, non piu' un'ombra che attraversa la piastrella.
    """
    l = a @ np.array([0.2126, 0.7152, 0.0722])
    n = l.shape[0]
    y, x = np.mgrid[0:n, 0:n] / float(n - 1)
    cols = [x ** i * y ** j for i in range(DEG + 1) for j in range(DEG + 1 - i)]
    M = np.stack([c.ravel() for c in cols], 1)
    c, *_ = np.linalg.lstsq(M, l.ravel(), rcond=None)
    g = (M @ c).reshape(n, n)
    return a * (g.mean() / np.maximum(g, 1e-6))[:, :, None]


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
    u0, v0 = bands(a @ np.array([0.2126, 0.7152, 0.0722]))
    a = flatten(a)
    u1, v1 = bands(a @ np.array([0.2126, 0.7152, 0.0722]))
    # moltiplicazione, non gamma: si vuole la stessa pietra meno chiara, e il
    # prodotto conserva i rapporti fra i toni invece di schiacciarne il contrasto
    a = np.clip(a * (TARGET / a.mean()), 0, 255)

    out = Image.fromarray(a.round().astype(np.uint8)).resize(
        (SIZE, SIZE), Image.LANCZOS)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    out.save(DST, "PNG", optimize=True)

    after = np.asarray(out).astype(np.float64).mean()
    swing = lambda b: 100.0 * (max(b) - min(b)) / (sum(b) / len(b))
    print("ritaglio %dx%d da (%d,%d), fattore %.3f" % (side, side, x0, y0, TARGET / before))
    print("luminanza %.1f -> %.1f  (obiettivo %.1f)" % (before, after, TARGET))
    print("gradiente su u %.0f%% -> %.0f%%,  su v %.0f%% -> %.0f%%"
          % (swing(u0), swing(u1), swing(v0), swing(v1)))
    print("scritta %s  (%d KB)" % (DST, os.path.getsize(DST) // 1024))


if __name__ == "__main__":
    main()
