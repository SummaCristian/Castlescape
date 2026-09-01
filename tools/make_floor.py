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

Ogni fuga e' poi larga un po' diversa dalle altre (JITTER), e i due lati della
stessa fuga sono diversi fra loro: senza, tutti i lastroni hanno esattamente lo
stesso smusso su tutti e quattro i lati e il pavimento sembra fresato. La
variazione e' pero' legata al giunto, non alla cella, altrimenti le due meta'
di un lastrone tagliato dal bordo prenderebbero due larghezze diverse e sulla
giunzione fra due piastrelle comparirebbe uno scalino di qualche millimetro.

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
JITTER = (0.78, 1.24)   # quanto puo' variare una semifuga rispetto alla misura

UP = (0.0, 1.0, 0.0)


def widen(t, key):
    """Semifuga effettiva di un lato. Deterministica e indicizzata sul giunto,
    non sulla cella: due celle che si affacciano sullo stesso giunto -- comprese
    le due meta' di un lastrone tagliato dal bordo della piastrella -- devono
    leggere lo stesso numero."""
    if key is None:
        return 0.0
    return t * random.Random(repr(key)).uniform(*JITTER)


def foot(t):
    """Semifuga alla base del lastrone: lo smusso rientra di CHAMFER ma lascia
    sempre una striscia di stucco piana. Sul bordo della piastrella (t=0) resta
    0 e lo smusso degenera in parete verticale."""
    return 0.0 if t <= 0.0 else max(t - CHAMFER, KEEP * t)


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

    smallest, border = TS, {}
    for c, r, xlo, xhi, zlo, zhi, t, key in G.cells():
        t = tuple(widen(v, k) for v, k in zip(t, key))
        tm, tp, tzm, tzp = t
        bm_, bp, bzm, bzp = [foot(v) for v in t]
        xt0, xt1, zt0, zt1 = xlo + tm, xhi - tp, zlo + tzm, zhi - tzp
        xb0, xb1, zb0, zb1 = xlo + bm_, xhi - bp, zlo + bzm, zhi - bzp
        smallest = min(smallest, xt1 - xt0, zt1 - zt0)
        if c == 0 or c == len(G.XJ):
            # le due meta' dello stesso lastrone, una per lato della piastrella
            border.setdefault(r, []).append((tzm, tzp))

        face(faces, [(xt0, 0, zt0), (xt0, 0, zt1),
                     (xt1, 0, zt1), (xt1, 0, zt0)], UP)
        face(faces, [(xt1, 0, zt0), (xt1, 0, zt1),
                     (xb1, -D, zb1), (xb1, -D, zb0)], (1, 1, 0))
        face(faces, [(xt0, 0, zt0), (xt0, 0, zt1),
                     (xb0, -D, zb1), (xb0, -D, zb0)], (-1, 1, 0))
        face(faces, [(xt0, 0, zt1), (xt1, 0, zt1),
                     (xb1, -D, zb1), (xb0, -D, zb1)], (0, 1, 1))
        face(faces, [(xt0, 0, zt0), (xt1, 0, zt0),
                     (xb1, -D, zb0), (xb0, -D, zb0)], (0, 1, -1))

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
    return faces, smallest


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


def check(obj, smallest):
    """L'ingombro deve restare quello dell'asset del kit, le UV dentro il
    quadrante del pavimento, e le facce a vista tutte a quota zero."""
    me = obj.data
    lo = [min(v.co[i] for v in me.vertices) for i in range(3)]
    hi = [max(v.co[i] for v in me.vertices) for i in range(3)]
    assert [round(v, 4) for v in lo] == [-TS, -TS, -THICK], lo
    assert [round(v, 4) for v in hi] == [0.0, 0.0, 0.0], hi

    for u, v in [l.uv for l in me.uv_layers[0].data]:
        assert G.U0 - 1e-6 <= u <= G.U1 + 1e-6, u
        assert 1.0 - G.V1 - 1e-6 <= v <= 1.0 - G.V0 + 1e-6, v

    top = sum(p.area for p in me.polygons
              if p.normal.z > 0.999 and abs(p.center.z) < 1e-6)
    mortar = sum(p.area for p in me.polygons
                 if p.normal.z > 0.999 and abs(p.center.z + DEPTH) < 1e-6)
    assert abs(mortar - TS * TS) < 1e-3, mortar
    # Se questo salta, il piano dello stucco e' tornato davanti ai lastroni e
    # il pavimento costa il doppio in riempimento: vedi build().
    last = me.polygons[-1]
    assert last.normal.z > 0.999 and abs(last.center.z + DEPTH) < 1e-6, \
        "il piano dello stucco non e' l'ultima faccia"
    slope = math.degrees(math.atan2(DEPTH, CHAMFER))
    print("%d facce, %d vertici" % (len(me.polygons), len(me.vertices)))
    print("lastroni a vista %.1f%% della piastrella, stucco %.1f%%" % (
        100 * top / (TS * TS), 100 * (1 - top / (TS * TS))))
    print("fuga profonda %.0f mm, smusso a %.0f gradi, lastrone minimo %.0f mm" % (
        DEPTH * 1000, slope, smallest * 1000))


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
    faces, smallest = build()
    obj = to_blender(faces)
    check(obj, smallest)
    out = export(obj)
    print("\nesportato in %s" % out)
    print("ora:  python tools/convert_assets.py \"%s\"" % os.path.join(out, NAME + ".gltf"))
    return out


if __name__ == "__main__":
    main()
