#!/usr/bin/env python3
"""
Misura sulla texture la griglia dei conci di una faccia di muro.

Il pavimento aveva una griglia sola e floor_grid.py se la teneva in una tabella
scritta a mano. Qui non si puo': ogni faccia a vista dei pezzi del muro pesca da
un pezzo diverso dell'atlante, con un disegno diverso e un orientamento diverso
(su una faccia u cresce verso l'alto, sull'altra verso il basso), e i pezzi sono
tre. Otto tabelle scritte a mano sarebbero otto occasioni di sbagliare a mano,
quindi la griglia si misura, dalla texture, al momento di generare la mesh.

COME. Le fughe sono righe scure. Una soglia globale pero' non le trova: le
macchie di degrado dipinte sui conci sono scure quanto le fughe, e una soglia
che le eviti perde le fughe piu' chiare. Si usa allora un top-hat -- si
confronta ogni texel con la media locale su una finestra piu' larga di un
concio e si tiene solo quello che sta sotto di uno scarto -- che e' cieco alle
variazioni lente (le macchie, il gradiente di luce dipinto) e vede solo i
solchi stretti. E' lo stesso metodo di _floor_joints.py, per la stessa ragione.

Prima pero' si proietta: le fughe di letto attraversano la faccia da parte a
parte, quindi mediando lungo la larghezza restano e tutto il resto si spegne. E
si media SOLO sui texel che appartengono davvero alla faccia -- la maschera --
perche' il rettangolo UV di una faccia contiene anche quello che le sta intorno
nell'atlante: sul campo esterno del muro dritto ci sta dentro tutto il vano
dell'arcata, che e' nero, e senza maschera si porta via la misura.

Poi, dentro ogni corso cosi' trovato, si ripete la stessa cosa nell'altra
direzione e si trovano le fughe di testa di quel corso. Due passate, non una
griglia incrociata: i corsi sono sfalsati, ogni corso ha le sue fughe di testa e
una passata unica sull'intera faccia le cancellerebbe a vicenda.

Tutto in texel del ritaglio; chi chiama converte con la sua mappa affine.
"""

import numpy as np

ATLAS = 1024

BASE = 31        # finestra della media locale, in texel: piu' larga di un concio
DEPTH = 4.0      # di quanto un texel deve stare sotto la media locale
WIDE = (2, 11)   # larghezza ammessa di una fuga, in texel
COVER = 0.10     # frazione minima di texel della faccia perche' una riga conti
# Bassa apposta. Il campo del pezzo del vano e' un poligono che attraversa tutta
# la parete ma ne copre un terzo -- e' fatto a L intorno all'apertura -- quindi
# quasi nessuna sua riga arriva al quarto di copertura. A 0.25 se ne misuravano
# tre corsi su otto e mezza parete restava piatta; a 0.10 si misurano tutti, e
# sulle facce piene (il muro dritto, l'angolo) non cambia niente.
MIN_COURSE = 7   # sotto questi texel non e' un corso ma il bordo del ritaglio


def _smooth(v, n):
    return np.convolve(np.pad(v, n // 2, mode="edge"), np.ones(n) / n, "valid")[:len(v)]


def _runs(v, ok):
    """Le fughe in un profilo: tratti sotto la media locale, larghi il giusto.
    Restituisce (centro, larghezza) in texel."""
    d = _smooth(v, BASE) - v
    m = (d > DEPTH) & ok
    out, s = [], None
    for i, x in enumerate(list(m) + [False]):
        if x and s is None:
            s = i
        elif not x and s is not None:
            if WIDE[0] <= i - s <= WIDE[1]:
                out.append(((s + i) / 2.0, float(i - s)))
            s = None
    return out


def _profile(lum, mask, axis):
    """Media della luminanza lungo `axis`, sui soli texel della faccia."""
    w = mask.sum(axis=axis)
    tot = (lum * mask).sum(axis=axis)
    ok = w >= COVER * mask.shape[axis]
    prof = np.where(w > 0, tot / np.maximum(w, 1), 0.0)
    prof[~ok] = _smooth(prof, BASE)[~ok]     # righe scoperte: neutre, non fughe
    return prof, ok


def measure(lum, mask, h_axis):
    """La griglia della faccia.

    `lum` e `mask` sono il ritaglio del quadrante (righe = v, colonne = u).
    `h_axis` dice quale asse del ritaglio corre lungo l'ALTEZZA del muro: 1 se
    e' quello delle colonne (u), 0 se e' quello delle righe.

    Restituisce (bed, heads): bed sono le fughe di letto lungo h_axis, heads e'
    una lista parallela ai corsi con le fughe di testa di ciascuno, tutte come
    (centro, larghezza) in texel del ritaglio.
    """
    w_axis = 1 - h_axis
    prof, ok = _profile(lum, mask, w_axis)
    bed = _runs(prof, ok)

    n = lum.shape[h_axis]
    edges = [0.0] + [c for c, _ in bed] + [float(n)]
    heads = []
    for i in range(len(edges) - 1):
        a, b = int(round(edges[i])), int(round(edges[i + 1]))
        if b - a < MIN_COURSE:
            heads.append(None)               # non e' un corso: e' un avanzo
            continue
        # due texel di margine: i bordi del corso sono le fughe di letto, e
        # entrerebbero nella media come una riga scura che non c'entra
        band = (slice(None), slice(a + 2, b - 2)) if h_axis == 1 else \
               (slice(a + 2, b - 2), slice(None))
        p, o = _profile(lum[band], mask[band], h_axis)
        heads.append(_runs(p, o))
    return bed, heads


def courses(bed, heads, n):
    """Da (bed, heads) all'elenco dei conci, in texel del ritaglio.

    Ogni elemento e' (k, i, h0, h1, w0, w1, half, key) con half le semifughe
    sui quattro lati nell'ordine (w-, w+, h-, h+) e key l'identita' del giunto,
    None dove il lato e' il bordo del ritaglio.

    Le chiavi sono del GIUNTO, non del concio: due conci affacciati sulla stessa
    fuga -- comprese le due meta' di quello tagliato dal bordo della piastrella
    -- devono leggere lo stesso numero, o affiancando due muri la fuga cambia
    larghezza a meta'. E' la lezione del pavimento e vale identica.
    """
    edges = [0.0] + [c for c, _ in bed] + [float(n[0])]
    halves = [0.0] + [w / 2.0 for _, w in bed] + [0.0]
    out = []
    for k in range(len(edges) - 1):
        hs = heads[k]
        if hs is None:
            continue
        h0, h1 = edges[k], edges[k + 1]
        se = [0.0] + [c for c, _ in hs] + [float(n[1])]
        sh = [0.0] + [w / 2.0 for _, w in hs] + [0.0]
        for i in range(len(se) - 1):
            key = (None if i == 0 else ("W", k, i - 1),
                   None if i == len(se) - 2 else ("W", k, i),
                   None if k == 0 else ("B", k - 1),
                   None if k == len(edges) - 2 else ("B", k))
            out.append((k, i, h0, h1, se[i], se[i + 1],
                        (sh[i], sh[i + 1], halves[k], halves[k + 1]), key))
    return out
