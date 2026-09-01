#!/usr/bin/env python3
"""
La griglia dei lastroni del pavimento del dungeon, misurata sulla sua texture.

La piastrella del kit e' una lastra liscia da 12 triangoli: i lastroni e le
fughe esistono solo come disegno dipinto dentro l'atlante SM_StoneFloor_01.png.
Sia la mesh scavata (make_floor.py) sia le venature (make_floor_texture.py)
devono cadere esattamente su quel disegno, altrimenti si vede una fuga
geometrica accanto a una fuga dipinta. Quindi la griglia non e' inventata: e'
misurata da _floor_joints.py con un top-hat sulla luminanza.

Un top-hat e non una soglia globale perche' la texture ha macchie di degrado
scure quanto le fughe e molto piu' larghe: una soglia le promuove a fughe
larghissime. Cio' che distingue una fuga e' che e' SOTTILE e piu' scura del suo
intorno immediato, cioe' esattamente quello che misura la differenza fra la
luminanza e una media mobile larga.

Coordinate: texel del ritaglio del quadrante, origine in alto a sinistra.

    cx 0..506  ->  x   0.0 .. -7.2      (u cresce verso le x negative)
    cy 0..506  ->  z   7.2 ..  0.0      (v cresce verso le z negative)

Il disegno si richiude su se stesso e nessuno dei quattro bordi della
piastrella cade su una fuga. Le colonne 0 e 8 sono percio' le due meta' della
stessa colonna di lastroni tagliata dal bordo (per questo condividono la lista
di fughe orizzontali), e la prima e l'ultima cella di ogni colonna sono le due
meta' dello stesso lastrone. E' la ragione per cui due piastrelle affiancate
non producono ne' un gradino ne' una fuga che nella texture non c'e'.
"""

# --- il quadrante del pavimento dentro l'atlante -----------------------------
# I quattro numeri sono le UV della faccia superiore della mesh originale, non
# una stima: SM_StoneFloor_01.gltf mappa (x=0,z=0) su (U0,V1) e (x=-7.2,z=7.2)
# su (U1,V0).
ATLAS = 1024
U0, V0, U1, V1 = 0.00324, 0.50324, 0.49676, 0.99676
CROP_X, CROP_Y, CROP = 3, 515, 506      # lo stesso ritaglio usato per misurare

TS = 7.2        # modulo della griglia del dungeon, cioe' il lato della piastrella
THICK = 0.2     # spessore della lastra, invariato rispetto all'asset del kit

TEXEL = TS / ((U1 - U0) * ATLAS)        # 0.014247 unita' di mondo per texel

# --- le fughe, come (centro, larghezza) in texel ------------------------------
# Fughe verticali: attraversano la piastrella da parte a parte.
XJ = [
    (32.5, 5.0), (111.5, 5.0), (161.0, 4.0), (239.0, 4.0),
    (288.5, 5.0), (367.5, 5.0), (417.0, 4.0), (495.0, 4.0),
]

# Fughe orizzontali: una lista per colonna, perche' i corsi sono sfalsati e ogni
# colonna ha i suoi. La 8 e' la meta' della 0 rimasta oltre il bordo, quindi
# ripete la 0. Escono piu' larghe delle verticali (6-8 texel contro 4-5) perche'
# nella texture lo sono davvero: i giunti di testa fra un lastrone e il
# successivo sono meno serrati di quelli di corso.
ZJ = [
    [(26.0, 6.0), (103.5, 7.0), (184.0, 8.0), (282.0, 6.0), (360.0, 8.0), (440.0, 8.0)],
    [(55.5, 7.0), (153.5, 7.0), (232.5, 7.0), (311.5, 7.0), (410.5, 7.0), (488.5, 7.0)],
    [(12.5, 7.0), (110.5, 7.0), (191.0, 8.0), (268.5, 7.0), (366.0, 8.0), (446.5, 7.0)],
    [(69.5, 7.0), (147.0, 6.0), (225.0, 8.0), (326.0, 8.0), (403.0, 6.0), (481.0, 8.0)],
    [(26.0, 6.0), (104.0, 8.0), (184.0, 8.0), (282.0, 6.0), (360.0, 8.0), (440.0, 8.0)],
    [(55.5, 7.0), (153.5, 7.0), (232.5, 7.0), (311.0, 8.0), (410.0, 8.0), (488.5, 7.0)],
    [(12.0, 8.0), (110.0, 8.0), (190.5, 7.0), (268.0, 6.0), (366.5, 7.0), (447.0, 8.0)],
    [(69.5, 7.0), (146.5, 7.0), (225.0, 8.0), (326.0, 8.0), (403.0, 6.0), (481.5, 7.0)],
]
ZJ.append(ZJ[0])                        # colonna 8: l'altra meta' della 0


def x_of_cx(cx):
    """Texel -> x di mondo. I bordi sono agganciati: il ritaglio e' arrotondato
    al texel e cadrebbe 4 mm fuori dalla piastrella."""
    u = (CROP_X + cx) / float(ATLAS)
    return min(0.0, max(-TS, -TS * (u - U0) / (U1 - U0)))


def z_of_cy(cy):
    v = (CROP_Y + cy) / float(ATLAS)
    return min(TS, max(0.0, TS * (V1 - v) / (V1 - V0)))


def uv_of(x, z):
    """La proiezione dall'alto della mesh originale, in convenzione glTF."""
    return (U0 + (-x / TS) * (U1 - U0), V1 - (z / TS) * (V1 - V0))


def cells():
    """Un lastrone per elemento, in coordinate di mondo.

    Restituisce (col, row, xlo, xhi, zlo, zhi, t, key) dove t e' la
    semilarghezza della fuga sui quattro lati nell'ordine (x-, x+, z-, z+).
    Sui lati che sono il bordo della piastrella t vale 0: li' il lastrone e'
    tagliato, non fugato, e prosegue nella piastrella accanto.

    key identifica il giunto e il lato, ed e' None sui bordi. Serve a chi vuole
    variare le fughe una per una: due celle che condividono lo stesso giunto
    ricevono la stessa chiave, comprese le due meta' dei lastroni tagliati dal
    bordo (colonne 0 e 8, prima e ultima riga). Sbagliare questo vuol dire un
    gradino di qualche millimetro sulla giunzione fra due piastrelle.
    """
    xe = [0.0] + [c for c, _ in XJ] + [float(CROP)]
    xh = [0.0] + [w / 2.0 for _, w in XJ] + [0.0]
    last = len(xe) - 2
    out = []
    for c in range(len(xe) - 1):
        # cx cresce verso le x negative: il bordo cx=xe[c] e' il lato x+
        xhi, xlo = x_of_cx(xe[c]), x_of_cx(xe[c + 1])
        tp, tm = xh[c] * TEXEL, xh[c + 1] * TEXEL
        cc = 0 if c == last else c          # la colonna 8 e' la meta' della 0
        ze = [0.0] + [j for j, _ in ZJ[c]] + [float(CROP)]
        zh = [0.0] + [w / 2.0 for _, w in ZJ[c]] + [0.0]
        nr = len(ze) - 2
        for r in range(len(ze) - 1):
            # cy cresce verso le z negative: il bordo cy=ze[r] e' il lato z+
            zhi, zlo = z_of_cy(ze[r]), z_of_cy(ze[r + 1])
            key = (None if c == last else ("V", c, "m"),
                   None if c == 0 else ("V", c - 1, "p"),
                   None if r == nr else ("H", cc, r, "m"),
                   None if r == 0 else ("H", cc, r - 1, "p"))
            out.append((c, r, xlo, xhi, zlo, zhi,
                        (tm, tp, zh[r + 1] * TEXEL, zh[r] * TEXEL), key))
    return out


if __name__ == "__main__":
    cs = cells()
    print("%d lastroni su %d colonne" % (len(cs), len(XJ) + 1))
    print("un texel vale %.5f unita'" % TEXEL)
    w = [(c[3] - c[2], c[5] - c[4]) for c in cs]
    print("lastrone piu' piccolo %.2f x %.2f, piu' grande %.2f x %.2f" % (
        min(a for a, _ in w), min(b for _, b in w),
        max(a for a, _ in w), max(b for _, b in w)))
    ks = set(k for c in cs for k in c[7] if k is not None)
    print("%d lati fugati, %d giunti distinti" % (
        sum(1 for c in cs for k in c[7] if k is not None), len(ks)))
    tt = [t for c in cs for t in c[6] if t > 0]
    print("semifughe da %.4f a %.4f unita' (%.1f-%.1f mm)" % (
        min(tt), max(tt), min(tt) * 1000, max(tt) * 1000))
