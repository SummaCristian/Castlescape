#!/usr/bin/env python3
"""
Scava i conci dei muri del dungeon, come make_floor.py fa con i lastroni.

I pezzi del kit hanno l'arcata cieca, le lesene e il cornicione gia' modellati,
ma il CAMPO fra loro -- cioe' quasi tutta la parete -- e' un poligono piatto da
una ventina di triangoli: i conci e le fughe sono solo disegnati nella texture.
Con la Cook-Torrance e delle torce appese al muro questo si vede piu' che sul
pavimento, perche' la torcia sta a mezzo metro dalla parete e la illumina di
striscio: una superficie piatta ha una normale sola, quindi tutto il campo si
accende insieme come un foglio, e le fughe restano scure quanto il pittore le ha
dipinte anche dove la luce arriva parallela al muro. Il risultato e' che
l'arcata modellata galleggia su una carta da parati.

Qui i conci diventano geometria. La griglia non e' inventata: wall_grid.py la
misura sulla texture, faccia per faccia, quindi ogni fuga scavata cade sulla
fuga dipinta con la sua larghezza.

QUALI FACCE. Non quelle che sembrano: la faccia verso la stanza NON e' il quad
piatto da due triangoli a x=0. Quello e' il di dietro. Il pezzo e' una L cava --
il muro dritto occupa il bordo della piastrella del pavimento, il pezzo d'angolo
due bordi -- e la stanza sta dalla parte CONCAVA, cioe' dove ci sono l'arcata e
le lesene. Lo si legge in scene.json: dhWallN sta a (-30.4, 19.2) ruotato di 90
gradi, e con quella rotazione il piano locale x=0 finisce a z=19.2 mentre la
stanza sta oltre z=20.44. Percio' si scavano le facce con normale -x (e -z sul
pezzo d'angolo), e sono due per lato, non una: il campo esterno intorno
all'arcata e il fondo della nicchia dentro l'arcata, che stanno su due piani
diversi e pescano da due pezzi diversi dell'atlante.

UN MURO NON E' UN PAVIMENTO IN PIEDI. Le differenze non sono di gusto:

  - le fughe di LETTO (fra un corso e l'altro) sono quasi rette, quelle di TESTA
    (fra due conci dello stesso corso) serpeggiano. E' il modo in cui si tira su
    un muro: il muratore livella ogni corso prima di posare il successivo, e la
    fuga di testa e' semplicemente l'avanzo fra un concio e il seguente. Nel
    pavimento fanno serpeggiare tutti e quattro i lati allo stesso modo, e li'
    va bene perche' un lastrone non deve stare in piedi su quello sotto;
  - i conci non SPROFONDANO, si mettono FUORI PIOMBO. Un lastrone calpestato per
    tre secoli scende dentro il suo letto di sabbia e resta parallelo al piano;
    un concio no, si assesta ruotando, e la sua faccia smette di essere parallela
    al filo del muro. Percio' qui l'arretramento non e' costante sul concio ma e'
    un piano inclinato: ogni concio prende una normale sua e sotto la torcia si
    accende con la sua intensita'. E' questo, non la profondita' della fuga, a
    fare la differenza fra una parete e un foglio;
  - le fughe sono piu' profonde di quelle del pavimento (22 mm contro 13) e lo
    smusso piu' stretto. Su un pavimento la malta si consuma fino quasi a filo e
    piu' di 13 mm legge come ceramica bombata; su un muro non la calpesta
    nessuno, e sotto la luce radente e' solo quella profondita' a dare l'ombra
    che fa leggere il concio come un blocco.

Il resto e' la lezione del pavimento, e vale identica:

  - le quote non si toccano. La faccia a vista dei conci resta esattamente sul
    piano del poligono originale e tutto il rilievo va all'INDIETRO, dentro il
    muro, quindi l'ingombro della mesh, il filo della parete e scene.json non
    cambiano di un millimetro;
  - i conci appoggiano sulla malta invece di essere cuciti fra loro. I corsi sono
    sfalsati, quindi lungo una fuga di letto i due lati hanno suddivisioni
    diverse: cucirli darebbe T-junction lungo tutte le fughe di letto, e le
    T-junction fanno pinhole. Con il piano dietro, ogni concio e' indipendente e
    in fondo alla fuga si vede la malta -- che e' anche quello che c'e' davvero
    fra due pietre;
  - ai bordi della piastrella la semifuga vale 0: li' il concio e' tagliato e
    prosegue nel muro accanto, cosi' lo smusso degenera in una parete
    perpendicolare e le due meta' combaciano. Il serpeggiamento delle fughe di
    letto e' percio' PERIODICO sulla larghezza della faccia, e l'arretramento del
    concio e' indicizzato sul concio intero: se non lo fosse, ogni 7.2 m
    comparirebbe uno scalino in mezzo alla parete. C'e' un assert che lo verifica;
  - la malta e' l'ULTIMA cosa scritta nel buffer. Copre tutta la faccia ma se ne
    vede un quinto: disegnandola per prima ogni pixel di parete passerebbe due
    volte nel fragment shader, che qui cicla su tutte le luci con PCF su cubemap.
    In fondo al buffer i conci hanno gia' scritto la profondita' e l'early-Z
    scarta il resto.

I conci che non ci stanno per intero dentro il poligono della faccia non si
alzano: intorno all'arcata resta la malta piana. E' voluto -- li' il concio
sarebbe tagliato dalla ghiera, che gli sta davanti e lo copre -- ed e' anche
l'unica cosa onesta da fare senza ritagliare il contorno del concio contro un
poligono concavo, che costerebbe molto e non si vedrebbe.

Le UV sono la stessa proiezione della faccia originale, quindi la texture del
kit resta buona e non c'e' niente da rifare.

Uso:
    python tools/make_wall.py
"""

import math
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall_grid as G
from convert_assets import Asset, write_gltf

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "skeleton", "source", "assets", "models", "Dracula")
TEXTURES = os.path.join(HERE, "..", "skeleton", "source", "assets", "textures", "Dracula")

# I pezzi da trattare, con le direzioni in cui sta la stanza. SM_WallDoor_01 non
# e' in scene.json (per il vano si usa SM_WallDoor_Hole_01) e non si tocca.
PIECES = [
    ("SM_WallStraight_01", "SM_WallStraight_02", "SM_WallStraight_01", [(0, -1)], []),
    ("SM_WallCorner_01", "SM_WallCorner_02", "SM_WallCorner_01", [(0, -1), (2, -1)], []),
    ("SM_WallDoor_Hole_01", "SM_WallDoor_Hole_02", "SM_WallDoor_Hole_01", [(0, -1)], []),
    # Il soffitto guarda in giu': la "stanza" sta sotto. Il fondo del cassettone
    # e' un campo di conci come un muro e si misura sulla texture; l'intradosso
    # delle travi, a y=0, e' un campo di conci anche lui ma la sua griglia sulla
    # texture non c'e' -- il pittore ci ha steso gli stessi mattoni del fondo,
    # perche' e' l'atlante a decidere, non noi. Quel piano si dichiara qui e la
    # griglia gliela costruisce Beam.
    ("SM_StoneCeiling_01", "SM_StoneCeiling_02", "SM_StoneCeiling_01", [(1, -1)],
     [(1, -1, 0.0)]),
]

MIN_AREA = 8.0       # sotto questa un piano non e' un campo di conci
MIN_BLOCKS = 12      # ...e se non ci si misura una griglia, non lo era
MIN_PIECE = 0.05     # sotto questo un frammento di concio non e' un concio

DEPTH = 0.022        # quanto sta dietro il piano della malta
CHAMFER = 0.019      # rientro dello smusso del concio
KEEP = 0.30          # frazione di semifuga che resta piana sotto lo smusso
EDGE = 0.40          # ...e quanto puo' valere al massimo del lato del concio
JITTER = (0.80, 1.24)   # quanto varia una semifuga da un giunto all'altro
WOBBLE_HEAD = 0.42      # ...e quanto varia lungo una fuga di testa
WOBBLE_BED = 0.12       # ...e lungo una di letto, che il muratore ha livellato
RINGS = (3, 7)          # punti di controllo sul giro, due ottave
CHIP = (0.0, 0.055)     # quanto e' sbocconcellato un angolo, in pianta
CHIP_MIN = 0.018        # sotto questa soglia l'angolo si lascia netto
SET = 0.004             # arretramento del concio dal filo del muro
TILT = 0.006            # e di quanto e' fuori piombo da un capo all'altro
STEP = 0.40             # passo di campionatura del bordo del concio
BEAM_STONE = 1.25       # lunghezza di riferimento di un concio di trave

# RINGS e STEP vanno letti insieme, come nel pavimento: le onde sono lunghe 2.4 e
# 1.0 m contro un passo di 40 cm. Con un rumore a onda corta la campionatura lo
# cancella e i bordi tornano dritti -- e comunque una pietra squadrata a mano non
# ha il bordo frastagliato: ha il bordo dritto ma non parallelo a quello accanto,
# e la fuga che si allarga da un capo all'altro.

_rings = {}


def wobble(key, s, period):
    """Rumore 1D in [-1,1], periodico su `period`.

    Periodico e' il punto per le fughe di letto: s e' una coordinata lungo il
    muro, e wobble(key, 0) e wobble(key, W) devono dare lo stesso numero per
    costruzione, perche' i due lati di quel confine sono due meta' dello stesso
    concio in due istanze diverse della mesh.
    """
    total = 0.0
    for k, amp in zip(RINGS, (1.0, 0.42)):
        g = _rings.get((key, k))
        if g is None:
            r = random.Random(repr((key, k)))
            g = _rings[(key, k)] = [r.uniform(-1.0, 1.0) for _ in range(k)]
        u = (s / period) * k
        i = int(math.floor(u))
        f = u - math.floor(u)
        f = f * f * (3.0 - 2.0 * f)
        total += amp * (g[i % k] * (1.0 - f) + g[(i + 1) % k] * f)
    return total / 1.42


class NotAField(Exception):
    """Il piano c'e' ma non e' un campo di conci: le sue UV non sono una
    proiezione ortogonale del disegno, quindi non c'e' nessuna griglia da
    misurarci sopra. Capita sui pezzi ritagliati a mano, come il fianco del vano
    della porta, dove le UV sono state cucite a occhio."""


class Face(object):
    """Una faccia a vista, con il suo sistema locale e la sua griglia.

    (X, Y) sono metri sulla faccia: X corre LUNGO i corsi, Y li ATTRAVERSA. Su
    una parete questo vuol dire X orizzontale e Y in altezza, ed e' l'unica
    lettura possibile perche' i corsi li ha livellati il muratore. Su una faccia
    orizzontale -- il fondo del cassettone del soffitto -- non c'e' nessuna
    gravita' nel piano a dirlo, e quale dei due assi porti i corsi lo decide la
    misura sulla texture: vedi `flip` e find_faces.

    h e' quanto si sta dietro il filo, sempre >= 0. Tutto il rilievo si
    costruisce qui dentro e torna in coordinate di mondo solo alla fine, cosi'
    lo stesso codice vale per il campo del muro dritto, per le due facce del
    pezzo d'angolo -- che guardano in direzioni diverse e sono piu' corte -- per
    il fondo della nicchia, che sta su un altro piano e su un altro quadrante, e
    per il soffitto, che guarda in giu'.
    """

    def __init__(self, axis, sign, plane, tris, pos, uv, image, flip=False):
        self.axis, self.sign, self.plane, self.tris = axis, sign, plane, tris
        self.normal = np.zeros(3)
        self.normal[axis] = sign
        rest = [i for i in (0, 1, 2) if i != axis]
        if axis == 1:                       # faccia orizzontale: nessun "alto"
            self.xax, self.yax = (rest[1], rest[0]) if flip else (rest[0], rest[1])
        else:                               # faccia verticale: i corsi sono livellati
            self.xax = [i for i in rest if i != 1][0]
            self.yax = 1
        p = pos[tris.reshape(-1)]
        q = uv[tris.reshape(-1)]

        # uv = A @ (X, Y) + b. L'atlante e' allineato alla faccia, quindi A e'
        # diagonale o antidiagonale: lo si verifica invece di darlo per buono,
        # perche' tutto il resto ci conta.
        M = np.column_stack([p[:, self.xax], p[:, self.yax], np.ones(len(p))])
        A, *_ = np.linalg.lstsq(M, q, rcond=None)
        # Tolleranza di un texel e mezzo, non zero: sul pannello del vano della
        # porta c'e' un vertice spostato di 13 mm a cui non hanno aggiornato le
        # UV, e a tolleranza stretta l'intero pannello veniva scartato e restava
        # piatto. Sotto il texel non serve piu' precisione di cosi': la griglia
        # la misuriamo in texel comunque.
        # Il lato dell'atlante si legge dall'immagine e non si scrive a mano: le
        # texture dei muri sono 1024, quella del soffitto 512, e con un numero
        # fisso la griglia misurata finirebbe al doppio della scala giusta.
        self.atlas = np.array([image.shape[1], image.shape[0]], float)
        if not np.allclose(M @ A, q, atol=1.5 / self.atlas.min()):
            raise NotAField("le UV non sono affini")
        # L'altezza deve muovere un asse dell'atlante e la larghezza l'altro:
        # se l'atlante fosse storto rispetto alla faccia, le fughe scavate non
        # cadrebbero su quelle dipinte. Il confronto e' RELATIVO -- lo storto si
        # misura in frazione della scala, non in unita' UV -- perche' con una
        # soglia assoluta bastava un vertice fuori posto per scartare la faccia.
        self.h_uv = 0 if abs(A[1, 0]) > abs(A[1, 1]) else 1   # asse uv di Y
        self.w_uv = 1 - self.h_uv
        skew = max(abs(A[1, self.w_uv]) / abs(A[1, self.h_uv]),
                   abs(A[0, self.h_uv]) / max(abs(A[0, self.w_uv]), 1e-12))
        if skew > 0.02:
            raise NotAField("l'atlante e' storto del %.0f%% sulla faccia" % (100 * skew))
        self.A, self.b = A[:2], A[2]

        # Il ritaglio del quadrante, e la maschera di cosa appartiene davvero
        # alla faccia: il rettangolo UV di un campo contiene anche il vano
        # dell'arcata, che e' nero e senza maschera si mangia la misura.
        self.t0 = np.floor(q.min(0) * self.atlas).astype(int)
        self.t1 = np.ceil(q.max(0) * self.atlas).astype(int)
        crop = image[self.t0[1]:self.t1[1], self.t0[0]:self.t1[0]]
        mask = self._mask(q, crop.shape)
        # righe = v, colonne = u: h_uv==0 vuol dire che Y corre sulle colonne
        bed, heads = G.measure(crop, mask, 1 if self.h_uv == 0 else 0)
        self.cells = G.courses(bed, heads, self._crop_shape())
        self.nbed = len(bed)

        # scala texel -> metri sui due assi, e il verso: la mappa e' allineata,
        # quindi ogni asse del ritaglio dipende da una coordinata sola e si
        # inverte da solo. Il verso puo' essere negativo -- sul campo esterno u
        # cresce verso il BASSO -- e non e' lo stesso su tutte le facce, quindi
        # non si assume: si legge dal segno del coefficiente.
        self.mh = abs(1.0 / (A[1, self.h_uv] * self.atlas[self.h_uv]))
        self.mw = abs(1.0 / (A[0, self.w_uv] * self.atlas[self.w_uv]))
        self.fh = 1.0 if A[1, self.h_uv] > 0 else -1.0
        self.fw = 1.0 if A[0, self.w_uv] > 0 else -1.0
        lo, hi = p.min(0), p.max(0)
        self.X0, self.Y0 = lo[self.xax], lo[self.yax]
        self.W = hi[self.xax] - lo[self.xax]
        self.H = hi[self.yax] - lo[self.yax]
        self.poly = np.stack([p[:, self.xax], p[:, self.yax]], 1).reshape(-1, 3, 2)

    def _crop_shape(self):
        n = self.t1 - self.t0                       # (u, v)
        return (n[self.h_uv], n[self.w_uv])         # (altezza, larghezza)

    def _mask(self, q, shape):
        """I texel coperti dai triangoli della faccia."""
        m = np.zeros(shape, bool)
        t = q.reshape(-1, 3, 2) * self.atlas - self.t0
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        px, py = xx + 0.5, yy + 0.5
        for a, b, c in t:
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-9:
                continue
            w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
            w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
            m |= (w0 >= -0.02) & (w1 >= -0.02) & (w0 + w1 <= 1.02)
        return m

    def Y(self, ch):
        """Texel del ritaglio attraverso i corsi -> metri."""
        return (((self.t0[self.h_uv] + ch) / self.atlas[self.h_uv] - self.b[self.h_uv])
                / self.A[1, self.h_uv])

    def X(self, cw):
        """Texel del ritaglio lungo i corsi -> metri."""
        return (((self.t0[self.w_uv] + cw) / self.atlas[self.w_uv] - self.b[self.w_uv])
                / self.A[0, self.w_uv])

    def world(self, X, Y, h):
        out = np.empty(3)
        out[self.axis] = self.plane - self.sign * h     # h e' quanto si sta DIETRO
        out[self.xax] = X
        out[self.yax] = Y
        return out

    def uv(self, X, Y):
        return tuple(np.array([X, Y]) @ self.A + self.b)

    def inside(self, X, Y):
        """(X, Y) sta dentro il poligono della faccia? Test sui triangoli, non
        sul contorno: il campo esterno ha un buco in mezzo (l'arcata) e un
        contorno ricostruito sarebbe un'altra cosa da sbagliare."""
        a, b, c = self.poly[:, 0], self.poly[:, 1], self.poly[:, 2]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        d = np.where(abs(d) < 1e-12, 1e-12, d)
        w0 = ((b[:, 1] - c[:, 1]) * (X - c[:, 0]) + (c[:, 0] - b[:, 0]) * (Y - c[:, 1])) / d
        w1 = ((c[:, 1] - a[:, 1]) * (X - c[:, 0]) + (a[:, 0] - c[:, 0]) * (Y - c[:, 1])) / d
        return bool(np.any((w0 >= -1e-6) & (w1 >= -1e-6) & (w0 + w1 <= 1 + 1e-6)))

    def blocks(self):
        """I conci in metri: (identita', x0, x1, y0, y1, semifughe, chiavi).

        L'identita' e' del CONCIO, non della cella: il primo e l'ultimo di un
        corso sono le due meta' dello stesso concio, tagliato dal bordo della
        piastrella, e si incontrano solo affiancando due pezzi nella scena.
        Devono arretrare uguale, ed e' l'identita' comune a garantirlo.
        """
        out = []
        # Il ritaglio del quadrante e' arrotondato al texel, quindi la griglia
        # sborda della frazione di texel che avanza: sul fondo del cassettone
        # erano 3 mm, e bastavano a far cadere FUORI dal pannello tutti i conci
        # del bordo, che percio' non si alzavano -- restava un telaio piatto
        # intorno a una toppa di conci. Si aggancia all'ingombro della faccia:
        # un campo non puo' andare oltre se stesso, e sotto il texel non c'e'
        # niente di misurato da rispettare.
        lox, hix = self.X0, self.X0 + self.W
        loy, hiy = self.Y0, self.Y0 + self.H
        clamp = lambda v, a, b: min(max(v, a), b)
        for k, i, h0, h1, w0, w1, half, key in self.cells:
            y0, y1 = sorted((self.Y(h0), self.Y(h1)))
            x0, x1 = sorted((self.X(w0), self.X(w1)))
            x0, x1 = clamp(x0, lox, hix), clamp(x1, lox, hix)
            y0, y1 = clamp(y0, loy, hiy), clamp(y1, loy, hiy)
            if x1 - x0 < 1e-4 or y1 - y0 < 1e-4:
                continue
            hw0, hw1, hh0, hh1 = half
            if self.fw < 0:
                hw0, hw1 = hw1, hw0
            if self.fh < 0:
                hh0, hh1 = hh1, hh0
            kw0, kw1, kh0, kh1 = key
            if self.fw < 0:
                kw0, kw1 = kw1, kw0
            if self.fh < 0:
                kh0, kh1 = kh1, kh0
            last = max(c[1] for c in self.cells if c[0] == k)
            out.append(((k, 0 if i == last else i), x0, x1, y0, y1,
                        (hw0 * self.mw, hw1 * self.mw, hh0 * self.mh, hh1 * self.mh),
                        (kw0, kw1, kh0, kh1)))

        return out

    def recess(self, bkey, y0, y1):
        """L'arretramento della faccia del concio dal filo del muro: non un
        numero ma una funzione di (X, Y), perche' un concio si assesta ruotando
        e la sua faccia smette di essere parallela al muro.

        La parte lungo l'altezza e' lineare -- il concio pende da una parte -- e
        usa il corso, che non e' tagliato da niente. Quella lungo il muro e'
        invece l'onda periodica: un termine lineare in X non si richiuderebbe
        sul modulo, e le due meta' del concio tagliato dal bordo prenderebbero
        arretramenti diversi, cioe' uno scalino ogni 7.2 m sulla faccia.
        """
        r = random.Random(repr(("set", bkey)))
        base = SET * r.random()
        gy, gx = r.uniform(-1.0, 1.0), r.uniform(-1.0, 1.0)

        def at(X, Y):
            v = gy * (2.0 * (Y - y0) / max(y1 - y0, 1e-9) - 1.0)
            v += gx * wobble(("tilt", bkey), X, self.W)
            return base + 0.25 * TILT * (v + 2.0)      # sempre >= 0: mai fuori dal filo

        return at

    def what(self):
        n = len(set(c[0] for c in self.cells))
        return "%2d cors%s" % (n, "o" if n == 1 else "i")


class Beam(Face):
    """La cornice di travi del cassettone del soffitto: un campo di conci di cui
    pero' sulla texture non c'e' scritta la griglia.

    Le travi prendono dall'atlante gli stessi mattoni del fondo -- e' la
    proiezione dall'alto di make_ceiling.py, non una scelta -- ma una trave non
    e' muratura. La fascia e' larga 45 cm e i conci dipinti sono lunghi 1.15,
    quindi dentro la fascia non ci sta un corso: ci stanno dei pezzi di conci
    tagliati per il lungo, e alzando quelli si alzavano quattro frammenti sparsi
    con la trave che sembrava sbriciolata. Per questo prima restava piatta.

    Un concio di trave e' invece un pezzo unico che tiene tutta la larghezza
    della fascia, e i suoi giunti sono TRASVERSALI. Quella griglia non e'
    disegnata da nessuna parte e non c'e' niente da misurare, quindi la si
    costruisce -- ed e' la cornice stessa a darla, senza inventare: i quattro
    cantonali dove le due fasce si incrociano, e fra un cantonale e l'altro il
    numero di conci uguali che sta piu' vicino a BEAM_STONE. Dalla texture si
    prende solo la larghezza delle fughe, che e' l'unica cosa che deve restare
    la stessa del resto del soffitto.

    L'altra differenza sta nel bordo della piastrella. Sul muro il bordo taglia
    in due i conci di un corso e le due meta' stanno ai due capi dello stesso
    corso; qui taglia in DUE DIREZIONI, perche' una trave del mondo e' larga 90
    cm -- due fasce di due piastrelle affiancate -- e un cantonale e' un quarto
    di blocco diviso fra quattro piastrelle. Percio' le meta' che si devono
    richiudere sono la fascia bassa con quella alta e la sinistra con la destra,
    i quattro cantonali fra loro, e l'arretramento dev'essere periodico su
    tutt'e due gli assi invece che su uno solo: vedi recess.
    """

    def __init__(self, *args):
        Face.__init__(self, *args)
        self.band = self._band()
        # La fuga: quella misurata sulla texture della faccia stessa. Un numero
        # scritto a mano qui sarebbe l'unica quota del pezzo non presa dal kit,
        # e si vedrebbe -- la trave sta a un metro dal fondo del cassettone,
        # nello stesso colpo d'occhio.
        half = [v for b in Face.blocks(self) for v in b[5] if v > 1e-9]
        assert half, "nessuna fuga misurata sulla texture della cornice"
        self.joint = float(np.median(half))

    def _band(self):
        """La larghezza della fascia, misurata sul contorno libero della faccia.

        La cornice e' un anello, e i suoi bordi liberi sono due: l'ingombro
        della piastrella e il perimetro della luce interna. Quelli che non
        stanno sull'ingombro danno la luce, e la differenza e' la fascia. Le
        diagonali dei cantonali non entrano: sono spigoli interni, ci passano
        due triangoli.
        """
        count = {}
        for tri in self.poly:
            pts = [(round(x, 6), round(y, 6)) for x, y in tri]
            for i in range(3):
                a, b = pts[i], pts[(i + 1) % 3]
                e = (a, b) if a <= b else (b, a)
                count[e] = count.get(e, 0) + 1
        lo = (self.X0, self.Y0)
        hi = (self.X0 + self.W, self.Y0 + self.H)
        on = lambda p: any(abs(p[i] - lo[i]) < 1e-4 or abs(p[i] - hi[i]) < 1e-4
                           for i in (0, 1))
        inner = [p for e, n in count.items() if n == 1 for p in e if not on(p)]
        assert inner, "la faccia non e' una cornice: non ha luce interna"
        band = [min(p[0] for p in inner) - lo[0], hi[0] - max(p[0] for p in inner),
                min(p[1] for p in inner) - lo[1], hi[1] - max(p[1] for p in inner)]
        assert max(band) - min(band) < 1e-4, "cornice di larghezza disuguale: %s" % band
        return band[0]

    def _lines(self, i):
        """I tagli lungo una fascia, e la chiave del giunto su ognuno.

        Le due fasce parallele di una piastrella hanno gli stessi tagli e le
        stesse chiavi: sono le due meta' delle stesse travi del mondo, e un
        taglio in piu' su una sola vorrebbe dire un giunto che si ferma a meta'
        della trave.
        """
        lo = (self.X0, self.Y0)[i]
        L = (self.W, self.H)[i] - 2.0 * self.band
        n = max(1, int(round(L / BEAM_STONE)))
        xs = ([lo, lo + self.band] + [lo + self.band + L * j / n for j in range(1, n)]
              + [lo + self.band + L, lo + self.band * 2 + L])
        # ai due capi il bordo della piastrella, dove il concio non finisce: li'
        # non c'e' fuga, il concio prosegue nella piastrella accanto
        return xs, [None] + [("BJ", i, j) for j in range(n + 1)] + [None]

    def blocks(self):
        xs, xk = self._lines(0)
        ys, yk = self._lines(1)
        out = []

        def add(bkey, x0, x1, y0, y1, key):
            out.append((bkey, x0, x1, y0, y1,
                        tuple(0.0 if k is None else self.joint for k in key), key))

        # Le fasce lungo X si prendono anche i quattro cantonali: cosi' l'anello
        # si scompone in rettangoli senza sovrapporsi, e un cantonale e' un
        # concio come gli altri invece di un pezzo a L che outline non saprebbe
        # disegnare.
        for i in range(len(xs) - 1):
            corner = i in (0, len(xs) - 2)
            for lo, hi in ((0, 1), (-2, -1)):
                add(("BC",) if corner else ("BX", i),
                    xs[i], xs[i + 1], ys[lo], ys[hi],
                    (xk[i], xk[i + 1], yk[lo], yk[hi]))
        # ...e quelle lungo Y vanno da un cantonale all'altro
        for j in range(1, len(ys) - 2):
            for lo, hi in ((0, 1), (-2, -1)):
                add(("BY", j), xs[lo], xs[hi], ys[j], ys[j + 1],
                    (xk[lo], xk[hi], yk[j], yk[j + 1]))
        return out

    def recess(self, bkey, y0, y1):
        """Come Face.recess, ma periodico su tutt'e due gli assi.

        Il termine lineare lungo il corso, che sulla parete descrive il concio
        che pende, qui non si puo' usare: il concio e' tagliato in due anche
        nell'altra direzione, e un termine lineare darebbe alle due meta'
        arretramenti diversi, cioe' uno scalino in mezzo a ogni trave. Restano
        le due onde, che sul modulo si richiudono per costruzione. La faccia
        resta comunque fuori piano -- e' questo che serve, perche' sotto la
        torcia ogni concio prenda la sua intensita' -- solo che invece di pendere
        da un capo all'altro si assesta seguendo l'onda.
        """
        r = random.Random(repr(("set", bkey)))
        base = SET * r.random()
        gx, gy = r.uniform(-1.0, 1.0), r.uniform(-1.0, 1.0)

        def at(X, Y):
            v = gx * wobble(("tiltx", bkey), X, self.W)
            v += gy * wobble(("tilty", bkey), Y, self.H)
            return base + 0.25 * TILT * (v + 2.0)

        return at

    def what(self):
        return "fascia %.2f m" % self.band


def side(t, key, s, period, amp):
    """Semifuga di un lato nel punto s. Deterministica e indicizzata sul giunto,
    non sul concio: due conci affacciati sulla stessa fuga -- comprese le due
    meta' di uno tagliato dal bordo -- leggono lo stesso numero. Sul bordo della
    faccia vale 0 e il lato resta dritto."""
    if key is None:
        return 0.0
    base = t * random.Random(repr(key)).uniform(*JITTER)
    return base * (1.0 + amp * wobble(key, s, period))


def foot(t, cap):
    """Semifuga alla base del concio: lo smusso rientra di CHAMFER ma lascia
    sempre una striscia di malta piana, e non si mangia mai piu' di EDGE del
    concio -- altrimenti sui frammenti tagliati dal bordo della piastrella, che
    sono larghi pochi centimetri, i due smussi si incontrerebbero."""
    if t <= 0.0:
        return 0.0
    return max(t - min(CHAMFER, cap), KEEP * t)


def span(a, b):
    """Punti interni fra due estremi, estremi esclusi."""
    n = max(1, int(round(abs(b - a) / STEP)))
    return [a + (b - a) * i / float(n) for i in range(1, n)]


def outline(f, x0, x1, y0, y1, half, key):
    """Il contorno del concio: quattro lati che non sono rette.

    Ogni punto porta con se' anche di quanto va spostato in fuori alla base
    (ox, oy), cioe' il rientro dello smusso in quel punto: cosi' la faccia a
    vista e l'appoggio sulla malta sono lo stesso contorno a due quote e lo
    smusso viene da se', senza dover ricostruire gli angoli.
    """
    hm, hp, hb, ht = half
    km, kp, kb, kt = key
    W, H = f.W, f.H
    # le fughe di testa corrono lungo l'altezza, quelle di letto lungo il muro:
    # solo queste ultime devono richiudersi sul modulo, ed e' li' che il
    # serpeggiamento va tenuto corto perche' il corso legga livellato
    xm = lambda y: side(hm, km, y, H, WOBBLE_HEAD)
    xp = lambda y: side(hp, kp, y, H, WOBBLE_HEAD)
    yb = lambda x: side(hb, kb, x, W, WOBBLE_BED)
    yt = lambda x: side(ht, kt, x, W, WOBBLE_BED)

    cx = EDGE * (x1 - x0)
    cy = EDGE * (y1 - y0)
    run = lambda v, c: v - foot(v, c)
    on_yb = lambda x: (x, y0 + yb(x), 0.0, -run(yb(x), cy))
    on_xp = lambda y: (x1 - xp(y), y, +run(xp(y), cx), 0.0)
    on_yt = lambda x: (x, y1 - yt(x), 0.0, +run(yt(x), cy))
    on_xm = lambda y: (x0 + xm(y), y, -run(xm(y), cx), 0.0)

    # Gli angoli per primi, valutando ogni lato con la quota nominale dell'altro:
    # da li' in poi ogni lato e' funzione di una variabile sola e i suoi estremi
    # sono esattamente questi punti.
    ax, bx = x0 + xm(y0 + hb), x1 - xp(y0 + hb)
    dx, cx2 = x0 + xm(y1 - ht), x1 - xp(y1 - ht)
    ay, by = y0 + yb(ax), y0 + yb(bx)
    dy, cy2 = y1 - yt(dx), y1 - yt(cx2)
    lim = 0.22 * min(bx - ax, cy2 - by)

    def chip(k1, k2):
        """Di quanto e' sbocconcellato l'angolo, in pianta. Zero se uno dei due
        lati e' il bordo della piastrella: li' il concio e' segato, e il taglio
        e' netto -- oltre che da ricucire con il muro accanto."""
        if k1 is None or k2 is None:
            return 0.0
        v = random.Random(repr(("chip", k1, k2))).uniform(*CHIP)
        # Sotto soglia l'angolo resta netto invece di prendere uno smusso di
        # pochi millimetri: quello darebbe un quad a spillo che non si vede e si
        # paga, e soprattutto lascia qualche angolo vivo, che e' cio' che rende
        # la sbocconcellatura irregolare invece di una smussatura.
        return 0.0 if v < CHIP_MIN else max(0.0, min(v, lim))

    ca, cb = chip(km, kb), chip(kp, kb)
    cc, cd = chip(kp, kt), chip(km, kt)

    # Giro antiorario A(x-,y-) -> B(x+,y-) -> C(x+,y+) -> D(x-,y+). Su un angolo
    # smussato si entra da un lato e si esce dall'altro, quindi i punti sono due;
    # su un angolo netto e' uno solo, con il rientro di entrambi i lati.
    loop = []
    loop += ([on_xm(ay + ca), on_yb(ax + ca)] if ca else
             [(ax, ay, -run(xm(ay), cx), -run(yb(ax), cy))])
    loop += [on_yb(x) for x in span(ax + ca, bx - cb)]
    loop += ([on_yb(bx - cb), on_xp(by + cb)] if cb else
             [(bx, by, +run(xp(by), cx), -run(yb(bx), cy))])
    loop += [on_xp(y) for y in span(by + cb, cy2 - cc)]
    loop += ([on_xp(cy2 - cc), on_yt(cx2 - cc)] if cc else
             [(cx2, cy2, +run(xp(cy2), cx), +run(yt(cx2), cy))])
    loop += [on_yt(x) for x in span(cx2 - cc, dx + cd)]
    loop += ([on_yt(dx + cd), on_xm(dy - cd)] if cd else
             [(dx, dy, -run(xm(dy), cx), +run(yt(dx), cy))])
    loop += [on_xm(y) for y in span(dy - cd, ay + ca)]
    return loop


def relief(f):
    """La faccia a vista, scavata. Restituisce i poligoni (punti di mondo, uv)
    gia' orientati verso la stanza, piu' due contatori per il resoconto."""
    out, border = [], {}
    thin, raised = f.W, 0

    # Prima il contorno di tutti i conci, poi si alzano: serve la passata in
    # piu' perche' un concio troppo stretto va scartato INSIEME alla sua altra
    # meta', che sta dall'altro capo della piastrella e si incontra dopo.
    plan = []
    for bkey, x0, x1, y0, y1, half, key in f.blocks():
        loop = outline(f, x0, x1, y0, y1, half, key)
        w = max(p[0] for p in loop) - min(p[0] for p in loop)
        h = max(p[1] for p in loop) - min(p[1] for p in loop)
        plan.append((bkey, x0, x1, y0, y1, key, loop, min(w, h)))
    # Le due meta' del concio tagliato dal bordo devono comportarsi uguale --
    # arretrare uguale e, se sono schegge, sparire tutt'e due: buttarne via una
    # sola lascerebbe uno scalino sul giunto fra due muri affiancati.
    tiny = set(b[0] for b in plan if b[-1] < MIN_PIECE)
    total = len(plan)

    for bkey, x0, x1, y0, y1, key, loop, _ in plan:
        if bkey in tiny:
            continue
        depth = f.recess(bkey, y0, y1)
        # Un concio che sporge dal poligono della faccia non si alza: intorno
        # all'arcata resta la malta piana, che li' e' coperta dalla ghiera.
        if not all(f.inside(x, y) for x, y, _, _ in loop):
            continue
        raised += 1
        thin = min(thin, max(p[0] for p in loop) - min(p[0] for p in loop),
                   max(p[1] for p in loop) - min(p[1] for p in loop))
        # I lati che sono il bordo della piastrella. Quello che deve combaciare
        # con l'altra meta' dello stesso concio -- che sta all'altro capo della
        # mesh e si incontra solo affiancando due pezzi nella scena -- e' il
        # profilo su quel bordo, quote comprese. Si guardano tutt'e due gli
        # assi, non solo quello dei corsi: sulla cornice del soffitto un concio
        # e' tagliato in due direzioni e un cantonale in quattro.
        for a, (lo, hi, k0, k1) in enumerate(((x0, x1, key[0], key[1]),
                                              (y0, y1, key[2], key[3]))):
            if k0 is None or k1 is None:
                e = lo if k0 is None else hi
                border.setdefault((bkey, a, round(x0 if a else y0, 9),
                                   round(x1 if a else y1, 9)), []).append(sorted(
                    (round(p[1 - a], 9), round(depth(p[0], p[1]), 9))
                    for p in loop if abs(p[a] - e) < 1e-9))

        top = [(x, y, depth(x, y)) for x, y, _, _ in loop]
        bot = [(x + ox, y + oy, DEPTH) for x, y, ox, oy in loop]
        # La faccia a vista non e' piana (il concio e' fuori piombo), quindi
        # niente n-gon: ventaglio dal baricentro, che per questi contorni -- un
        # rettangolo con gli angoli smussati -- e' sempre stellato.
        gx = sum(p[0] for p in top) / len(top)
        gy = sum(p[1] for p in top) / len(top)
        c = (gx, gy, depth(gx, gy))
        for j in range(len(top)):
            out.append(poly(f, [c, top[j], top[(j + 1) % len(top)]]))
        # Lo smusso, una striscia di quad fra i due contorni. Sui lati che sono
        # il bordo della piastrella il rientro e' nullo, i due contorni
        # coincidono e il quad e' degenere: quello va buttato, non emesso.
        for j in range(len(top)):
            m = (j + 1) % len(top)
            quad = [top[j], top[m], bot[m], bot[j]]
            if area(f, quad) > 1e-9:
                out.append(poly(f, quad))

    for k, pair in border.items():
        # Se questi profili non coincidono, affiancando due pezzi il concio
        # tagliato dal bordo cambia larghezza o arretramento a meta': uno
        # scalino in mezzo alla parete, e ce n'e' uno ogni 7.2 m.
        assert all(p == pair[0] for p in pair), (k, pair)
    # Niente del rilievo deve stare DAVANTI al filo della parete: il muro
    # mangerebbe lo spazio della stanza e le quote di scene.json non sarebbero
    # piu' quelle. Si guarda qui e non sulla mesh finita, perche' li' davanti ci
    # stanno per progetto le lesene e la ghiera, che sono roba del kit.
    front = min([h for ws, _ in out for h in [0.0]] +
                [f.sign * (f.plane - w[f.axis]) for ws, _ in out for w in ws])
    return out, thin, raised, total, front


def poly(f, pts, want=None):
    ws = [f.world(x, y, h) for x, y, h in pts]
    uvs = [f.uv(x, y) for x, y, _ in pts]
    if float(np.dot(newell(ws), f.normal if want is None else want)) < 0:
        ws, uvs = ws[::-1], uvs[::-1]
    return ws, uvs


def rim(f, box):
    """Il raccordo con cui la malta risale a filo sul bordo interno del campo.

    Arretrando la malta di 22 mm il campo si stacca da quello che gli sta
    intorno. Sul pezzo d'angolo questo apriva un buco vero, e vale la pena
    scriverlo perche' non e' ovvio: le due facce a vista si toccano sullo
    spigolo concavo, e arretrandole tutt'e due resta in piedi un pilastrino di
    22x22 mm per tutta l'altezza. Le sue due facce laterali prima erano dentro
    la muratura e nessuno le aveva modellate; adesso danno sull'incavo, che e'
    aperto verso la stanza, e da dentro si vede fuori.

    Tapparle sarebbe una pezza. La cosa giusta e' non scavare fino al bordo:
    negli ultimi 22 mm la malta risale in rampa fino al filo, quindi l'incavo si
    chiude da solo e le due facce tornano a incontrarsi sullo spigolo esattamente
    dov'erano prima. Il pilastrino non esiste piu' perche' non c'e' piu' niente
    da cui sporgere. E' anche giusto di suo: la malta contro la ghiera
    dell'arcata o contro lo spigolo fa cordolo, non taglio netto.

    Non su tutti i bordi pero': dove il campo arriva all'ingombro del pezzo, di
    la' c'e' la piastrella accanto con la sua malta alla stessa quota, e una
    rampa li' sarebbe un cordolo di 22 mm in mezzo alla parete, ripetuto ogni
    7.2 m. Percio' si guarda l'ingombro: bordo sull'ingombro, niente rampa;
    bordo interno -- lo spigolo concavo dell'angolo, il giro dell'arcata, il
    perimetro del fondo della nicchia -- rampa.
    """
    count, apex = {}, {}
    for tri in f.poly:                      # (X, Y) dei tre vertici, nel piano
        pts = [(round(x, 6), round(y, 6)) for x, y in tri]
        for i in range(3):
            a, b = pts[i], pts[(i + 1) % 3]
            e = (a, b) if a <= b else (b, a)
            count[e] = count.get(e, 0) + 1
            apex.setdefault(e, pts[(i + 2) % 3])

    lo, hi = box
    onbox = lambda v, i: abs(v - lo[i]) < 1e-4 or abs(v - hi[i]) < 1e-4
    out = []
    for ((ax, ay), (bx, by)), n in count.items():
        if n != 1:
            continue
        if (ax == bx and onbox(ax, f.xax)) or (ay == by and onbox(ay, f.yax)):
            continue                        # e' il bordo della piastrella
        cx, cy = apex[((ax, ay), (bx, by))]
        # normale del bordo che punta FUORI dal campo, nel piano: la rampa
        # scende verso l'interno, quindi si va nel verso opposto
        ox, oy = ay - by, bx - ax
        if ox * (cx - ax) + oy * (cy - ay) > 0:
            ox, oy = -ox, -oy
        d = math.hypot(ox, oy) or 1.0
        ox, oy = ox * DEPTH / d, oy * DEPTH / d
        # a filo sul bordo, al fondo della malta 22 mm piu' dentro. Se i due
        # punti di dentro non cadono nel campo la rampa si salta: succede sui
        # triangoli a scheggia del contorno frastagliato del pezzo del vano,
        # dove il terzo vertice e' quasi allineato al lato e non dice piu' da
        # che parte sia il dentro. Meglio un bordo netto che una rampa a caso.
        if not (f.inside(ax - ox, ay - oy) and f.inside(bx - ox, by - oy)):
            continue
        out.append(poly(f, [(ax, ay, 0.0), (bx, by, 0.0),
                            (bx - ox, by - oy, DEPTH), (ax - ox, ay - oy, DEPTH)]))
    return out


def area(f, pts):
    return float(np.linalg.norm(newell([f.world(x, y, h) for x, y, h in pts]))) / 2.0


def newell(vs):
    n = np.zeros(3)
    for i, a in enumerate(vs):
        b = vs[(i + 1) % len(vs)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return n


def find_faces(pos, uv, idx, image, rooms, beams=()):
    """I campi di conci: i triangoli complanari con la normale verso la stanza,
    abbastanza grandi, su cui wall_grid riesce a misurare una griglia.

    `beams` sono i piani dichiarati come cornice di travi: quelli la griglia non
    la fanno misurare, se la fanno costruire (vedi Beam).

    Cercarli e non scriverli a mano e' il punto: il muro dritto ne ha due (il
    campo intorno all'arcata e il fondo della nicchia), il pezzo d'angolo
    quattro, e il pezzo del vano ha un contorno che nessuno ha voglia di
    trascrivere. Tutto il resto della mesh -- ghiera, lesene, cornicione,
    fianchi, il di dietro -- non passa questi filtri e non si tocca.
    """
    v = pos[idx]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    L = np.linalg.norm(n, axis=1)
    groups = {}
    for t in range(len(idx)):
        if L[t] < 1e-12:
            continue
        u = n[t] / L[t]
        for ax, sg in rooms:
            if u[ax] * sg > 0.999:
                groups.setdefault((ax, sg, round(float(v[t][:, ax].mean()), 4)),
                                  []).append(t)

    out = []
    for (ax, sg, plane), group in sorted(groups.items()):
        beam = any(a == ax and s == sg and abs(v - plane) < 1e-4 for a, s, v in beams)
        for ts in components(group, idx):
            if float(L[ts].sum()) / 2.0 < MIN_AREA:
                continue
            if beam:
                # Niente prova nei due versi e nessun minimo di conci: sulla
                # cornice non c'e' una griglia da riconoscere, e l'orientamento
                # non conta perche' l'anello e' simmetrico.
                f = Beam(ax, sg, plane, idx[ts], pos, uv, image)
                f.members = set(ts)
                out.append(f)
                continue
            # Su una faccia verticale i corsi sono livellati e non c'e' niente
            # da scegliere. Su una orizzontale -- il fondo del cassettone del
            # soffitto -- si prova nei due versi e vince quello che trova piu'
            # fughe passanti: le fughe di letto attraversano la faccia da parte
            # a parte e nel profilo mediato restano, quelle di testa sono
            # sfalsate da un corso all'altro e mediandole si cancellano. Quale
            # dei due assi porti i corsi lo dice il disegno, non noi.
            best = None
            for flip in ((False, True) if ax == 1 else (False,)):
                try:
                    f = Face(ax, sg, plane, idx[ts], pos, uv, image, flip)
                except NotAField:
                    continue
                if len(f.cells) < MIN_BLOCKS:
                    continue                # non e' un campo di conci: e' un taglio
                if best is None or f.nbed > best.nbed:
                    best = f
            if best is None:
                continue
            f = best
            f.members = set(ts)
            out.append(f)
    return out


def components(ts, idx):
    """Spezza un gruppo complanare nei suoi pezzi staccati.

    Serve per il pezzo del vano della porta: i due pannelli ai lati dell'apertura
    sono complanari ma pescano da due punti diversi dell'atlante, e un fit affine
    solo su tutt'e due non esiste -- infatti la faccia veniva scartata e meta'
    parete restava piatta. Sono pezzi di muro diversi e vanno misurati separati.
    """
    parent = {t: t for t in ts}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    seen = {}
    for t in ts:
        for v in idx[t]:
            if v in seen:
                a, b = find(t), find(seen[v])
                parent[a] = b
            else:
                seen[v] = t
    out = {}
    for t in ts:
        out.setdefault(find(t), []).append(t)
    return list(out.values())


def carve(src, dst, tex, rooms, beams):
    a = Asset(os.path.join(MODELS, src + ".gltf"))
    p = a.json["meshes"][0]["primitives"][0]
    pos = a.read_accessor(p["attributes"]["POSITION"]).astype(np.float64)
    nrm = a.read_accessor(p["attributes"]["NORMAL"]).astype(np.float64)
    uv = a.read_accessor(p["attributes"]["TEXCOORD_0"]).astype(np.float64)
    idx = a.read_accessor(p["indices"]).reshape(-1, 3).astype(np.int64)

    img = np.asarray(Image.open(os.path.join(TEXTURES, tex + ".png")).convert("RGB"))
    lum = img @ np.array([0.2126, 0.7152, 0.0722])
    faces = find_faces(pos, uv, idx, lum, rooms, beams)
    assert faces, "nessun campo di conci trovato in " + src
    print("%s: %d camp%s di conci" % (src, len(faces), "o" if len(faces) == 1 else "i"))

    drop, polys, thinnest = set(), [], 1e9
    for f in faces:
        drop.update(f.members)
        ps, thin, raised, total, front = relief(f)
        assert front >= -1e-4, "il rilievo esce dal filo del muro: %.4f" % front
        thinnest = min(thinnest, thin)
        polys += ps
        print("   %sx%+d a %7.4f  %5.2f x %5.2f m  %-12s %3d/%3d conci alzati"
              % ("xyz"[f.axis], f.sign, f.plane, f.W, f.H, f.what(), raised, total))

    # I triangoli che restano, cosi' come sono; poi il rilievo; poi la malta,
    # che e' la faccia originale arretrata ed e' l'ULTIMA cosa nel buffer.
    keep = [t for t in range(len(idx)) if t not in drop]
    P = [pos[idx[t]] for t in keep]
    N = [nrm[idx[t]] for t in keep]
    T = [uv[idx[t]] for t in keep]
    box = (pos.min(0), pos.max(0))
    for f in faces:
        polys += rim(f, box)
        for t in f.tris:
            q = pos[t].copy()
            q[:, f.axis] -= f.sign * DEPTH
            polys.append((list(q), list(uv[t])))
    for ws, uvs in polys:
        n = newell(ws)
        n = n / max(np.linalg.norm(n), 1e-12)
        for j in range(1, len(ws) - 1):        # ventaglio: i poligoni sono convessi
            tri = np.array([ws[0], ws[j], ws[j + 1]])
            # I triangoli di area nulla non si scrivono. Non disegnano niente ma
            # la loro normale e' spazzatura, e basta uno sullo spigolo per far
            # credere -- a un ray-cast come a un algoritmo di occlusione -- che
            # li' ci sia una superficie che invece non c'e'.
            if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) < 1e-9:
                continue
            P.append(tri)
            N.append(np.array([n, n, n]))
            T.append(np.array([uvs[0], uvs[j], uvs[j + 1]]))

    P = np.concatenate(P).astype(np.float32)
    N = np.concatenate(N).astype(np.float32)
    T = np.concatenate(T).astype(np.float32)
    key = np.concatenate([P, N, T], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)                  # tieni l'ordine dei triangoli
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    ind = rank[inv.reshape(-1)].astype(np.uint32)
    check(P, ind, pos, faces, thinnest)
    write_gltf(dst, P[first[order]], N[first[order]], T[first[order]], ind, MODELS)


def check(P, ind, src_pos, faces, thinnest):
    """L'ingombro deve restare quello del pezzo del kit: le istanze in
    scene.json stanno tutte a y=0.02 e affiancate sul passo da 7.2 m, e un
    millimetro di crescita qui e' un millimetro di compenetrazione li'."""
    assert np.allclose(P.min(0), src_pos.min(0), atol=1e-4), (P.min(0), src_pos.min(0))
    assert np.allclose(P.max(0), src_pos.max(0), atol=1e-4), (P.max(0), src_pos.max(0))
    assert thinnest > 0.02, "un concio si e' chiuso: %.4f" % thinnest
    print("   %d triangoli, %d vertici; fuga %.0f mm, smusso %.0f gradi, "
          "fuori piombo %.1f mm" % (
              len(ind) // 3, len(np.unique(ind)), DEPTH * 1000,
              math.degrees(math.atan2(DEPTH, CHAMFER)), (SET + TILT) * 1000))


def main():
    for src, dst, tex, rooms, beams in PIECES:
        carve(src, dst, tex, rooms, beams)
    print("\nora aggiorna scene.json: %s" % ", ".join(p[1] for p in PIECES))


if __name__ == "__main__":
    main()
