#!/usr/bin/env python3
"""
Genera SM_StoneFloor_02, il pavimento del dungeon con i lastroni davvero rilevati.

Il pezzo del kit, SM_StoneFloor_01, e' una lastra liscia da 12 triangoli: i
lastroni e le fughe sono solo disegnati nella texture. Con la Cook-Torrance e
delle torce che rasentano il pavimento questo si vede, perche' una superficie
piatta ha una normale sola: le fughe restano scure quanto il pittore le ha
dipinte anche quando la luce arriva parallela al pavimento, e il pavimento
legge come carta da parati stesa per terra.

Qui i lastroni diventano geometria. Ogni lastrone e' un tronco di piramide
schiacciato -- faccia a vista piana a y=0, smusso che scende allargandosi --
appoggiato su un unico piano dello stucco a -DEPTH. La griglia viene da
floor_grid.py, cioe' e' misurata sulla texture: ogni fuga scavata cade sulla
fuga dipinta, con la sua larghezza.

Perche' i lastroni appoggiano su un piano invece di essere cuciti fra loro. I
corsi sono sfalsati e ogni colonna ha le sue fughe orizzontali, quindi lungo
una fuga verticale i due lati hanno suddivisioni diverse: cucirli darebbe
T-junction lungo tutte le fughe verticali, e le T-junction fanno pinhole. Con
il piano sotto, invece, ogni lastrone e' indipendente da tutti gli altri, non
c'e' un solo vertice da far combaciare, e nel fondo della fuga si vede lo
stucco -- che e' anche quello che c'e' davvero fra due pietre. Il piano non e'
complanare con nessuna faccia dei lastroni, quindi non aggiunge z-fighting (il
difetto che [dracula-asset-zfighting] documenta per gli altri pezzi del kit).

Ai bordi della piastrella la semifuga vale 0: li' il lastrone e' tagliato e
prosegue nella piastrella accanto, cosi' lo smusso degenera in una parete
verticale che combacia con quella della piastrella vicina, esattamente come
gia' fa la lastra del kit. Le fughe che attraversano il bordo restano aperte in
quella parete, ed e' giusto: di la' la fuga continua.

Le quote non sono scelte a occhio:
  - la faccia a vista resta a y=0, quindi il piano di calpestio e l'ingombro
    della mesh sono identici a quelli dell'asset del kit e scene.json non si
    tocca (le istanze restano a y=0.02, l'alzata comune del dungeon);
  - DEPTH 13 mm su lastroni da 1.1-1.4 m e' quanto si consuma uno stucco
    vecchio, e basta a far cambiare intensita' allo smusso senza che i lastroni
    diventino piastrelle bombate: a 20 mm, provati, il pavimento legge come
    ceramica smussata invece che come pietra;
  - lo smusso rientra di 18 mm ma non oltre il 65% della semifuga, cosi' sotto
    resta sempre una striscia di stucco piana anche sulle fughe verticali, che
    nella texture sono le piu' strette.

Poi il pavimento e' vecchio, e su questo si gioca quasi tutto l'aspetto:

  - ogni fuga e' larga un po' diversa dalle altre (JITTER), e i due lati della
    stessa fuga sono diversi fra loro;
  - e nessun bordo di lastrone e' una retta: la semifuga varia anche LUNGO il
    giunto (WOBBLE), quindi ogni lato e' una spezzata che serpeggia e due
    lastroni affacciati non hanno mai lo stesso profilo;
  - qualche lastrone e' sprofondato di un paio di millimetri (SINK), cosi' le
    facce a vista non sono tutte sullo stesso piano e la luce radente lo mostra.

Senza queste tre cose i lastroni sono rettangoli identici con lo stesso smusso
sui quattro lati, e il pavimento legge come piastrella pressata: e' un castello
vecchio, non un bagno.

Tutte e tre sono pero' agganciate al giunto o al lastrone, mai alla cella, e il
serpeggiamento e' PERIODICO SUL MODULO da 7.2 m. E' la condizione che tiene in
piedi l'affiancamento: il bordo della piastrella taglia dei lastroni a meta',
e le due meta' vivono in due istanze diverse della stessa mesh. Se il profilo
non si richiudesse esattamente dopo 7.2 m, ogni 7.2 m comparirebbe uno scalino
nella fuga. Per questo il rumore e' un anello di punti di controllo campionato
in coordinate di mondo, non una funzione qualsiasi: a x=0 e a x=-7.2 vale per
costruzione lo stesso numero.

La texture e' SM_StoneFloor_02.png (make_floor_texture.py), che aggiunge le
venature. Le UV sono la stessa proiezione dall'alto dell'asset originale, sullo
stesso quadrante dell'atlante, quindi la mesh funziona anche con la texture del
kit se si vuole cambiare una cosa sola alla volta.

Uso:
    blender --background --python tools/make_floor.py
    python tools/convert_assets.py <cartella stampata dallo script>
"""

import math
import os
import random
import sys
import tempfile

import bmesh
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floor_grid as G

NAME = "SM_StoneFloor_02"

TS, THICK = G.TS, G.THICK
DEPTH = 0.013        # quanto sta sotto il piano dello stucco
CHAMFER = 0.018      # rientro orizzontale dello smusso del lastrone
KEEP = 0.35          # frazione di semifuga che resta piana sotto lo smusso
JITTER = (0.78, 1.24)   # quanto varia una semifuga da un giunto all'altro
WOBBLE = 0.45           # ...e quanto varia lungo il giunto stesso
RINGS = (3, 7)          # punti di controllo sul giro da 7.2 m, due ottave
CHIP = (0.0, 0.075)     # quanto e' sbocconcellato un angolo, in pianta
CHIP_MIN = 0.012        # sotto questa soglia l'angolo si lascia netto
SINK = 0.0035           # di quanto puo' essere sprofondato un lastrone
STEP = 0.30             # passo di campionatura del bordo del lastrone

# RINGS e STEP vanno letti insieme. Le onde sono lunghe 2.4 m e 1.03 m contro un
# passo di 30 cm: 8 e 3.4 campioni per onda. Con un rumore a onda corta (il primo
# tentativo era 31 e 15 cm) la campionatura lo cancella e i bordi tornano dritti
# -- e comunque non e' quello che si vuole. Una pietra tagliata a mano non ha il
# bordo frastagliato: ha il bordo dritto ma NON parallelo a quello accanto, e la
# fuga che si allarga e si stringe da un capo all'altro. Quello lo da' l'onda
# lunga, e costa pochi vertici.

UP = (0.0, 1.0, 0.0)

_rings = {}


def wobble(key, s):
    """Rumore 1D in [-1,1], periodico sul modulo da 7.2 m.

    Periodico e' il punto: s e' una coordinata di mondo, e wobble(key, 0) e
    wobble(key, -7.2) devono dare lo stesso numero per costruzione, perche' i
    due lati di quel confine sono due meta' dello stesso lastrone in due
    istanze diverse della mesh. Un anello di punti di controllo interpolato con
    smoothstep lo garantisce senza doverci pensare ogni volta.
    """
    total = 0.0
    for k, amp in zip(RINGS, (1.0, 0.42)):
        g = _rings.get((key, k))
        if g is None:
            r = random.Random(repr((key, k)))
            g = _rings[(key, k)] = [r.uniform(-1.0, 1.0) for _ in range(k)]
        u = (s / TS) * k
        i = int(math.floor(u))
        f = u - math.floor(u)
        f = f * f * (3.0 - 2.0 * f)
        total += amp * (g[i % k] * (1.0 - f) + g[(i + 1) % k] * f)
    return total / 1.42


def side(t, key, s):
    """Semifuga di un lato nel punto s. Deterministica e indicizzata sul
    giunto, non sulla cella: due celle affacciate sullo stesso giunto --
    comprese le due meta' di un lastrone tagliato dal bordo -- leggono lo
    stesso numero. Sul bordo della piastrella vale 0 e il lato resta dritto."""
    if key is None:
        return 0.0
    base = t * random.Random(repr(key)).uniform(*JITTER)
    return base * (1.0 + WOBBLE * wobble(key, s))


def sink(c, r):
    """Di quanto e' sprofondato il lastrone, indicizzato sul lastrone intero:
    le due meta' di uno tagliato dal bordo devono sprofondare uguale, o al
    confine fra due piastrelle si apre un gradino."""
    key = ((c - 1) % len(G.XJ), (r - 1) % len(G.ZJ[c]))
    return -SINK * random.Random(repr(("sink", key))).random()


def foot(t):
    """Semifuga alla base del lastrone: lo smusso rientra di CHAMFER ma lascia
    sempre una striscia di stucco piana. Sul bordo della piastrella (t=0) resta
    0 e lo smusso degenera in parete verticale."""
    return 0.0 if t <= 0.0 else max(t - CHAMFER, KEEP * t)


def span(a, b):
    """Punti interni fra due estremi, estremi esclusi."""
    n = max(1, int(round(abs(b - a) / STEP)))
    return [a + (b - a) * i / float(n) for i in range(1, n)]


def outline(xlo, xhi, zlo, zhi, t, key):
    """Il contorno del lastrone: quattro lati che non sono rette.

    Ogni punto porta con se' anche di quanto va spostato in fuori alla base
    (ox, oz), cioe' il rientro dello smusso in quel punto: cosi' la faccia a
    vista e l'appoggio sullo stucco sono lo stesso contorno a due quote e lo
    smusso viene da se', senza dover ricostruire gli angoli.
    """
    tm, tp, tzm, tzp = t
    km, kp, kzm, kzp = key
    xm = lambda z: side(tm, km, z)          # semifuga sul lato -x, lungo z
    xp = lambda z: side(tp, kp, z)
    zm = lambda x: side(tzm, kzm, x)        # semifuga sul lato -z, lungo x
    zp = lambda x: side(tzp, kzp, x)
    run = lambda v: v - foot(v)             # rientro orizzontale dello smusso

    # Un punto su ciascuno dei quattro lati, con il rientro dello smusso.
    on_zm = lambda x: (x, zlo + zm(x), 0.0, -run(zm(x)))
    on_xp = lambda z: (xhi - xp(z), z, +run(xp(z)), 0.0)
    on_zp = lambda x: (x, zhi - zp(x), 0.0, +run(zp(x)))
    on_xm = lambda z: (xlo + xm(z), z, -run(xm(z)), 0.0)

    # Gli angoli per primi, valutando ogni lato con la quota nominale
    # dell'altro: da li' in poi ogni lato e' una funzione di una variabile sola
    # e i suoi estremi sono esattamente questi punti.
    ax, bx = xlo + xm(zlo + tzm), xhi - xp(zlo + tzm)
    cx, dx = xhi - xp(zhi - tzp), xlo + xm(zhi - tzp)
    az, bz = zlo + zm(ax), zlo + zm(bx)
    cz, dz = zhi - zp(cx), zhi - zp(dx)
    lim = 0.22 * min(bx - ax, cz - bz)

    def chip(k1, k2):
        """Di quanto e' sbocconcellato l'angolo, in pianta. Zero se uno dei due
        lati e' il bordo della piastrella: li' il lastrone e' segato, e il
        taglio e' netto -- oltre che da ricucire con la piastrella accanto."""
        if k1 is None or k2 is None:
            return 0.0
        v = random.Random(repr(("chip", k1, k2))).uniform(*CHIP)
        # Sotto soglia l'angolo resta netto invece di prendere uno smusso di
        # pochi millimetri: quello darebbe un quad a spillo che non si vede e
        # si paga, e soprattutto lascia qualche angolo vivo, che e' cio' che
        # rende la sbocconcellatura irregolare invece di una smussatura.
        return 0.0 if v < CHIP_MIN else min(v, lim)

    ca, cb = chip(km, kzm), chip(kp, kzm)
    cc, cd = chip(kp, kzp), chip(km, kzp)

    # Giro antiorario A(-x,-z) -> B(+x,-z) -> C(+x,+z) -> D(-x,+z). Su un
    # angolo smussato si entra da un lato e si esce dall'altro, quindi i punti
    # sono due; su un angolo netto e' uno solo, con il rientro di entrambi i
    # lati (altrimenti i due punti coincidono e il quad dello smusso degenera).
    loop = []
    loop += ([on_xm(az + ca), on_zm(ax + ca)] if ca else
             [(ax, az, -run(xm(az)), -run(zm(ax)))])
    loop += [on_zm(x) for x in span(ax + ca, bx - cb)]
    loop += ([on_zm(bx - cb), on_xp(bz + cb)] if cb else
             [(bx, bz, +run(xp(bz)), -run(zm(bx)))])
    loop += [on_xp(z) for z in span(bz + cb, cz - cc)]
    loop += ([on_xp(cz - cc), on_zp(cx - cc)] if cc else
             [(cx, cz, +run(xp(cz)), +run(zp(cx)))])
    loop += [on_zp(x) for x in span(cx - cc, dx + cd)]
    loop += ([on_zp(dx + cd), on_xm(dz - cd)] if cd else
             [(dx, dz, -run(xm(dz)), +run(zp(dx)))])
    loop += [on_xm(z) for z in span(dz - cd, az + ca)]
    return loop


def newell(vs):
    n = [0.0, 0.0, 0.0]
    for i, a in enumerate(vs):
        b = vs[(i + 1) % len(vs)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return n


def face(out, vs, want):
    """Aggiunge un poligono orientandolo verso `want`. Avvolgere a mano ogni
    faccia sarebbe fragile: la mesh non e' un solido chiuso (i lastroni sono
    aperti sotto, appoggiati sul piano) e recalc_face_normals su una mesh non
    manifold e' un'euristica, non una risposta."""
    n = newell(vs)
    if sum(a * b for a, b in zip(n, want)) < 0:
        vs = vs[::-1]
    out.append(vs)


def build():
    D = DEPTH
    faces = []

    smallest, border, sunk = TS, {}, []
    for c, r, xlo, xhi, zlo, zhi, t, key in G.cells():
        loop = outline(xlo, xhi, zlo, zhi, t, key)
        dy = sink(c, r)
        sunk.append(dy)
        xs = [p[0] for p in loop]
        zs = [p[1] for p in loop]
        smallest = min(smallest, max(xs) - min(xs), max(zs) - min(zs))
        if c == 0 or c == len(G.XJ):
            # Le due meta' dello stesso lastrone, una per lato della
            # piastrella: quello che deve combaciare e' il profilo del bordo.
            edge = xhi if c == 0 else xlo
            border.setdefault(r, []).append(
                (round(dy, 9), sorted(round(p[1], 9) for p in loop
                                      if abs(p[0] - edge) < 1e-9)))

        top = [(x, dy, z) for x, z, _, _ in loop]
        bot = [(x + ox, -D, z + oz) for x, z, ox, oz in loop]
        face(faces, top, UP)
        # Lo smusso, una striscia di quad fra i due contorni. `want` punta via
        # dal baricentro del lastrone: cosi' vale sia per le facce inclinate
        # sia per quelle verticali sul bordo della piastrella, dove lo smusso
        # e' nullo e la normale e' orizzontale.
        gx, gz = sum(xs) / len(xs), sum(zs) / len(zs)
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            mx = (top[i][0] + top[j][0] + bot[i][0] + bot[j][0]) / 4.0
            mz = (top[i][2] + top[j][2] + bot[i][2] + bot[j][2]) / 4.0
            face(faces, [top[i], top[j], bot[j], bot[i]],
                 (mx - gx, 0.3, mz - gz))

    # Fianchi e fondo della lastra: sotto il piano dello stucco e' un
    # parallelepipedo come l'originale, e i fianchi combaciano con quelli della
    # piastrella accanto senza sporgere.
    face(faces, [(0, -D, 0), (0, -D, TS),
                 (0, -THICK, TS), (0, -THICK, 0)], (1, 0, 0))
    face(faces, [(-TS, -D, 0), (-TS, -D, TS),
                 (-TS, -THICK, TS), (-TS, -THICK, 0)], (-1, 0, 0))
    face(faces, [(0, -D, 0), (-TS, -D, 0),
                 (-TS, -THICK, 0), (0, -THICK, 0)], (0, 0, -1))
    face(faces, [(0, -D, TS), (-TS, -D, TS),
                 (-TS, -THICK, TS), (0, -THICK, TS)], (0, 0, 1))
    face(faces, [(0, -THICK, 0), (-TS, -THICK, 0),
                 (-TS, -THICK, TS), (0, -THICK, TS)], (0, -1, 0))

    # Il piano dello stucco, una faccia sola sotto tutta la piastrella, ed e'
    # l'ULTIMA cosa scritta nel buffer, non la prima. Copre tutti i 51.84 m2
    # della piastrella ma se ne vede il 15%: disegnandola per prima, ogni pixel
    # di pavimento passa due volte nel fragment shader -- che qui cicla su tutte
    # le luci con PCF su cubemap ed e' la cosa piu' cara del frame, mentre il
    # pavimento e' la superficie piu' grande a schermo. In fondo al buffer,
    # invece, i lastroni hanno gia' scritto la profondita' e l'early-Z scarta i
    # frammenti coperti prima di ombreggiarli. L'ordine delle facce qui e'
    # l'ordine dei triangoli nel .bin (l'esportatore glTF lo conserva, e si
    # verifica), quindi questa riga non e' stile: e' una scelta di prestazioni.
    face(faces, [(0, -D, 0), (-TS, -D, 0), (-TS, -D, TS), (0, -D, TS)], UP)

    assert smallest > 4 * G.TEXEL, "un lastrone si e' chiuso: %.4f" % smallest
    for r, pair in border.items():
        # Se queste due non coincidono, affiancando due piastrelle il lastrone
        # tagliato dal bordo cambia larghezza a meta': uno scalino nella fuga.
        assert len(pair) == 2 and pair[0] == pair[1], (r, pair)
    return faces, smallest, sunk


def uvs_of(vs):
    """Proiezione dall'alto, la stessa dell'asset del kit: la pietra corre
    continua su faccia, smusso e fondo della fuga. I fianchi della lastra sono
    verticali e la proiezione li schiaccerebbe su una riga, quindi li si
    proietta rientrando verso l'interno di quanto scendono: sono comunque
    sepolti fra due piastrelle, ma cosi' campionano texel veri e restano dentro
    il quadrante dell'atlante."""
    out = []
    for x, y, z in vs:
        if y < -DEPTH:                       # fianco o fondo della lastra
            d = -y - DEPTH
            x += d if x <= -TS + 1e-6 else (-d if x >= -1e-6 else 0.0)
            z += d if z <= 1e-6 else (-d if z >= TS - 1e-6 else 0.0)
        out.append(G.uv_of(x, z))
    return out


def to_blender(faces):
    me = bpy.data.meshes.new(NAME)
    bm = bmesh.new()
    layer = bm.loops.layers.uv.new("UVMap")
    for vs in faces:
        # glTF (x, y, z) -> Blender (x, -z, y); la V delle UV e' ribaltata
        f = bm.faces.new([bm.verts.new((v[0], -v[2], v[1])) for v in vs])
        f.smooth = False                     # spigoli vivi: normali per faccia
        for loop, (u, v) in zip(f.loops, uvs_of(vs)):
            loop[layer].uv = (u, 1.0 - v)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()

    for o in [o for o in bpy.data.objects if o.name.startswith(NAME)]:
        bpy.data.objects.remove(o, do_unlink=True)
    obj = bpy.data.objects.new(NAME, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def check(obj, smallest, sunk):
    """L'ingombro deve restare quello dell'asset del kit, le UV dentro il
    quadrante del pavimento, e nessuna faccia a vista sopra quota zero."""
    me = obj.data
    lo = [min(v.co[i] for v in me.vertices) for i in range(3)]
    hi = [max(v.co[i] for v in me.vertices) for i in range(3)]
    assert [round(v, 4) for v in lo] == [-TS, -TS, -THICK], lo
    assert [round(v, 4) for v in hi[:2]] == [0.0, 0.0], hi
    # I lastroni sprofondano, non si alzano: il piano di calpestio resta y=0 e
    # l'ingombro non cresce, cosi' scene.json e le quote del dungeon non
    # cambiano di una virgola.
    assert -SINK - 1e-6 <= hi[2] <= 0.0, hi

    for u, v in [l.uv for l in me.uv_layers[0].data]:
        assert G.U0 - 1e-6 <= u <= G.U1 + 1e-6, u
        assert 1.0 - G.V1 - 1e-6 <= v <= 1.0 - G.V0 + 1e-6, v

    top = sum(p.area for p in me.polygons
              if p.normal.z > 0.999 and p.center.z > -SINK - 1e-6)
    mortar = sum(p.area for p in me.polygons
                 if p.normal.z > 0.999 and abs(p.center.z + DEPTH) < 1e-6)
    assert abs(mortar - TS * TS) < 1e-3, mortar
    # Se questo salta, il piano dello stucco e' tornato davanti ai lastroni e
    # il pavimento costa il doppio in riempimento: vedi build().
    last = me.polygons[-1]
    assert last.normal.z > 0.999 and abs(last.center.z + DEPTH) < 1e-6, \
        "il piano dello stucco non e' l'ultima faccia"
    slope = math.degrees(math.atan2(DEPTH, CHAMFER))
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    print("%d facce (%d triangoli), %d vertici" % (
        len(me.polygons), tris, len(me.vertices)))
    print("lastroni a vista %.1f%% della piastrella, stucco %.1f%%" % (
        100 * top / (TS * TS), 100 * (1 - top / (TS * TS))))
    print("fuga profonda %.0f mm, smusso a %.0f gradi, lastrone minimo %.0f mm" % (
        DEPTH * 1000, slope, smallest * 1000))
    print("sprofondamento da %.1f a %.1f mm su %d lastroni" % (
        -min(sunk) * 1000, -max(sunk) * 1000, len(sunk)))


def export(obj):
    out_dir = os.path.join(tempfile.gettempdir(), "cg_floor_export")
    os.makedirs(out_dir, exist_ok=True)
    for o in bpy.context.view_layer.objects:
        o.select_set(o is obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(out_dir, NAME + ".gltf"),
        export_format='GLTF_SEPARATE', use_selection=True, export_apply=True,
        export_yup=True, export_normals=True, export_texcoords=True,
        export_materials='NONE', export_extras=False)
    return out_dir


def main():
    faces, smallest, sunk = build()
    obj = to_blender(faces)
    check(obj, smallest, sunk)
    out = export(obj)
    print("\nesportato in %s" % out)
    print("ora:  python tools/convert_assets.py \"%s\"" % os.path.join(out, NAME + ".gltf"))
    return out


if __name__ == "__main__":
    main()
