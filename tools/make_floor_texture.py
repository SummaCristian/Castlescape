#!/usr/bin/env python3
"""
Genera SM_StoneFloor_02.png: la texture del pavimento con le venature.

La pietra del kit e' uniforme sotto le macchie di degrado: da vicino, e con le
torce che la rasentano, i lastroni leggono come cemento colorato. Qui ogni
lastrone prende qualche vena di calcite -- chiara, sottile, che entra ed esce
dalla superficie invece di attraversarla tutta.

Tre scelte contano piu' delle altre.

Le vene sono generate lastrone per lastrone, con direzione, passo e fase
propri, non come un campo unico steso su tutta la piastrella. Un campo unico
farebbe combaciare le vene attraverso le fughe, e un pavimento in cui il
disegno prosegue da una pietra all'altra si legge subito come finto: i lastroni
sono blocchi tagliati separatamente. Per lo stesso motivo non entrano mai nella
fuga -- si spengono qualche texel prima del bordo del lastrone.

I lastroni tagliati dal bordo della piastrella sono ricuciti. La griglia si
richiude su se stessa (vedi floor_grid.py), quindi la colonna 0 e la 8 sono le
due meta' della stessa colonna, e cosi' la prima e l'ultima riga di ogni
colonna. Le vene si generano sul lastrone intero e poi si scrivono con gli
indici modulo il lato del ritaglio: affiancando due piastrelle una vena
attraversa il bordo invece di morirci contro, che sarebbe l'unico posto in cui
il modulo da 7.2 m si vedrebbe a occhio nudo.

Non tutti i lastroni sono venati e nessuna vena e' continua da bordo a bordo:
un secondo campo di rumore la accende e la spegne lungo il suo percorso. E'
quello che rende "qualche venatura" invece di una rigatura regolare.

Si scrive un file nuovo, non si tocca l'atlante del kit: cosi' la mesh
SM_StoneFloor_02 resta utilizzabile anche con la texture originale.

Uso:
    python tools/make_floor_texture.py
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floor_grid as G

SRC = "skeleton/source/assets/textures/Dracula/SM_StoneFloor_01.png"
DST = "skeleton/source/assets/textures/Dracula/SM_StoneFloor_02.png"

SEED = 20260901
MARGIN = 1.5        # texel di stacco fra la vena e il bordo del lastrone
PLAIN = 0.22        # frazione di lastroni lasciati senza vene
LIGHT = 0.28        # opacita' massima del nucleo chiaro della vena
HALO = 0.10         # opacita' massima dell'alone scuro che le da' rilievo


def value_noise(rng, shape, cells):
    """Rumore di valore: griglia casuale interpolata con smoothstep."""
    gh, gw = max(1, cells[0]), max(1, cells[1])
    g = rng.random((gh + 1, gw + 1))
    y = np.linspace(0, gh, shape[0], endpoint=False)
    x = np.linspace(0, gw, shape[1], endpoint=False)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - y0, x - x0
    sy = (fy * fy * (3 - 2 * fy))[:, None]
    sx = (fx * fx * (3 - 2 * fx))[None, :]
    top = g[np.ix_(y0, x0)] * (1 - sx) + g[np.ix_(y0, x0 + 1)] * sx
    bot = g[np.ix_(y0 + 1, x0)] * (1 - sx) + g[np.ix_(y0 + 1, x0 + 1)] * sx
    return top * (1 - sy) + bot * sy


def fbm(rng, shape, cells, octaves=3):
    out, amp, tot = np.zeros(shape), 1.0, 0.0
    for o in range(octaves):
        out += amp * value_noise(rng, shape, (cells[0] << o, cells[1] << o))
        tot, amp = tot + amp, amp * 0.5
    return out / tot


def stones():
    """I lastroni interi, in texel, con i bordi oltre il ritaglio dove si
    richiudono dall'altra parte. Uno per fuga verticale: il lastrone j va dalla
    fuga j alla j+1, e l'ultimo scavalca il bordo per ricongiungersi alla 0."""
    n = len(G.XJ)
    for j in range(n):
        (xa, wa), (xb, wb) = G.XJ[j], G.XJ[(j + 1) % n]
        if j == n - 1:
            xb += G.CROP
        zj = G.ZJ[j + 1]                       # la colonna a destra della fuga j
        m = len(zj)
        for i in range(m):
            (za, ha), (zb, hb) = zj[i], zj[(i + 1) % m]
            if i == m - 1:
                zb += G.CROP
            yield (xa + wa / 2 + MARGIN, xb - wb / 2 - MARGIN,
                   za + ha / 2 + MARGIN, zb - hb / 2 - MARGIN)


def one_vein(rng, h, w, yy, xx):
    """Una vena sola: la curva di livello di una coordinata deformata.

    Non una sinusoide periodica -- quella darebbe piu' vene tutte parallele e
    tutte equidistanti, cioe' una rigatura. Qui si prende una direzione a caso,
    la si deforma con del rumore e si tiene UN livello solo, scelto in modo che
    attraversi il lastrone: viene fuori una curva sola, che serpeggia.
    """
    theta = rng.uniform(0.0, np.pi)
    f = xx * np.cos(theta) + yy * np.sin(theta)
    f = f + (fbm(rng, (h, w), (2, 2), 3) - 0.5) * rng.uniform(0.14, 0.30) * max(h, w)
    d = np.abs(f - rng.uniform(f.min(), f.max()))

    # La larghezza varia lungo il percorso: una vena si assottiglia e si
    # ingrossa, un graffio no.
    width = rng.uniform(0.9, 1.8) * (0.55 + 0.9 * fbm(rng, (h, w), (3, 3), 2))
    core = np.exp(-(d / width) ** 2)
    halo = np.exp(-((d - 2.6 * width) / (1.3 * width)) ** 2)
    # Accesa e spenta lungo il percorso: nessuna vena corre da bordo a bordo.
    gate = np.clip((fbm(rng, (h, w), (1, 1), 2) - 0.44) * 3.4, 0.0, 1.0)
    return core * gate, halo * gate


def veins(rng, h, w):
    """Il campo di vene di un lastrone: 1 sul filo della vena, 0 altrove."""
    yy = np.arange(h, dtype=float)[:, None]
    xx = np.arange(w, dtype=float)[None, :]
    core = np.zeros((h, w))
    halo = np.zeros((h, w))
    for _ in range(int(rng.choice([1, 1, 1, 2, 2, 3]))):
        c, a = one_vein(rng, h, w, yy, xx)
        core, halo = np.maximum(core, c), np.maximum(halo, a)
    # Si spengono sul filo della fuga: la pietra e' tagliata li', la vena no.
    edge = np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx))
    fade = np.clip(edge / 2.5, 0.0, 1.0)
    return core * fade, halo * fade


def main():
    src = Image.open(SRC).convert("RGB")
    atlas = np.asarray(src).astype(np.float64).copy()
    x0, y0, n = G.CROP_X, G.CROP_Y, G.CROP
    crop = atlas[y0:y0 + n, x0:x0 + n]

    rng = np.random.default_rng(SEED)
    painted = wrapped = 0
    for xa, xb, za, zb in stones():
        xs = np.arange(int(np.ceil(xa)), int(np.floor(xb)) + 1)
        ys = np.arange(int(np.ceil(za)), int(np.floor(zb)) + 1)
        if len(xs) < 8 or len(ys) < 8:
            continue
        if rng.random() < PLAIN:               # qualche lastrone resta liscio
            continue
        core, halo = veins(rng, len(ys), len(xs))
        amp = rng.uniform(0.55, 1.0)
        idx = np.ix_(ys % n, xs % n)
        wrapped += int(xs[-1] >= n or ys[-1] >= n)

        stone = crop[idx]
        a = (LIGHT * amp * core)[:, :, None]
        b = (HALO * amp * halo)[:, :, None]
        # La calcite e' la stessa pietra schiarita, non un colore nuovo: il
        # prodotto conserva le macchie sotto la vena invece di coprirle.
        light = np.clip(stone * 1.38 + 15.0, 0, 255) * np.array([1.02, 1.0, 0.96])
        stone = stone * (1 - a) + light * a
        crop[idx] = np.clip(stone * (1 - b) + stone * 0.78 * b, 0, 255)
        painted += 1

    atlas[y0:y0 + n, x0:x0 + n] = crop
    out = Image.fromarray(np.clip(atlas, 0, 255).round().astype(np.uint8))
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    out.save(DST, "PNG", optimize=True)

    before = np.asarray(src).astype(np.float64)[y0:y0 + n, x0:x0 + n].mean()
    print("%d lastroni venati (%d a cavallo del bordo)" % (painted, wrapped))
    print("luminanza del quadrante %.1f -> %.1f" % (before, crop.mean()))
    print("scritta %s  (%d KB)" % (DST, os.path.getsize(DST) // 1024))


if __name__ == "__main__":
    main()
