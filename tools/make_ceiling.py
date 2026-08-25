#!/usr/bin/env python3
"""
Genera SM_StoneCeiling_01, la piastrella di soffitto del dungeon.

Il kit Dracula non ha un soffitto: questo e' l'unico pezzo strutturale
modellato da noi. Tutte le quote sotto sono ricavate dalla geometria reale
degli altri pezzi, non scelte a occhio:

  SM_StoneFloor_01    x [-7.2,0], z [0,7.2], spessore 0.2 -> modulo 7.2
  SM_WallStraight_01  alto 6.1929, spesso 1.2421

Da cui la piastrella copre esattamente l'impronta del pavimento e appoggia il
piano y=0 (la faccia inferiore delle travi) sulla sommita' del muro. In
scene.json questo si traduce in "translate": [x, 6.213, z] per ogni piastrella,
cioe' 0.02 (l'alzata comune del dungeon) + 6.1929: le due superfici combaciano
senza sovrapporsi, quindi non ci sono facce complanari coincidenti che possano
fare z-fighting con la faccia interna del muro.

La forma e' un cassettone: cornice di travi sul perimetro, smusso, pannello di
fondo rialzato. Simmetrico sui due assi, quindi affianca se stesso in qualsiasi
direzione senza orientamento obbligato, e lo smusso da' alle torce una
superficie inclinata su cui l'intensita' cambia, invece di un lastrone piatto
che sotto la Cook-Torrance resterebbe di un colore solo.

Nessun materiale: come tutti i pezzi del kit la texture arriva da scene.json,
e le UV sono mappate dentro l'atlante di SM_StoneFloor_01.png alla stessa
densita' del pavimento (0.4936 UV ogni 7.2 unita'), cosi' i corsi di pietra
hanno la stessa scala su pavimento e soffitto e non serve una texture nuova.

Uso:
    blender --background --python tools/make_ceiling.py
    python tools/convert_assets.py <cartella stampata dallo script>

Il secondo passaggio e' quello che porta il file nella forma che Starter.hpp sa
leggere (glTF ASCII, attributi non interlacciati, un solo materiale).
"""

import math
import os
import sys
import tempfile

import bmesh
import bpy

NAME = "SM_StoneCeiling_01"

TS   = 7.2      # modulo della griglia
BW   = 0.45     # larghezza della trave perimetrale
CH_H = 0.30     # altezza dello smusso
CH_W = 0.20     # rientro orizzontale dello smusso
SLAB = 0.55     # spessore totale della lastra

# L'atlante di SM_StoneFloor_01.png ha la pietra nei due quadranti di sinistra:
# quello in basso e' il piano di calpestio, quello in alto e' quello che il
# pavimento usa sui fianchi. Il quadrante in alto a destra e' nero e va evitato.
K      = 0.4936 / 7.2     # UV per unita', misurata sul top del pavimento
U0, V0 = 0.0032, 0.5032   # quadrante "piano"
U1, V1 = 0.0032, 0.0032   # quadrante "fianchi"


def build():
    """Costruisce la mesh. Coordinate glTF (Y in alto), convertite in fondo."""
    Ox0, Ox1, Oz0, Oz1 = -TS, 0.0, 0.0, TS                      # bordo esterno
    Ix0, Ix1, Iz0, Iz1 = -TS + BW, -BW, BW, TS - BW             # luce interna
    Px0, Px1 = Ix0 + CH_W, Ix1 - CH_W                           # fondo del cassettone
    Pz0, Pz1 = Iz0 + CH_W, Iz1 - CH_W

    def down(x, y, z):
        """Proiezione dall'alto: la pietra corre continua su trave, smusso e fondo."""
        return (U0 + (x + TS) * K, V0 + z * K)

    def side(u, y):
        return (U1 + u * K, V1 + (SLAB - y) * K)

    faces = []   # ogni quad e' gia' avvolto nel verso della sua normale

    # sottotrave perimetrale, y=0, guarda in basso
    for q in [[(Ox0, 0, Oz0), (Ox1, 0, Oz0), (Ix1, 0, Iz0), (Ix0, 0, Iz0)],
              [(Ix0, 0, Iz1), (Ix1, 0, Iz1), (Ox1, 0, Oz1), (Ox0, 0, Oz1)],
              [(Ox0, 0, Oz0), (Ix0, 0, Iz0), (Ix0, 0, Iz1), (Ox0, 0, Oz1)],
              [(Ix1, 0, Iz0), (Ox1, 0, Oz0), (Ox1, 0, Oz1), (Ix1, 0, Iz1)]]:
        faces.append((q, [down(*v) for v in q]))

    # smusso
    for q in [[(Ix0, 0, Iz0), (Ix1, 0, Iz0), (Px1, CH_H, Pz0), (Px0, CH_H, Pz0)],
              [(Px0, CH_H, Pz1), (Px1, CH_H, Pz1), (Ix1, 0, Iz1), (Ix0, 0, Iz1)],
              [(Ix0, 0, Iz0), (Px0, CH_H, Pz0), (Px0, CH_H, Pz1), (Ix0, 0, Iz1)],
              [(Px1, CH_H, Pz0), (Ix1, 0, Iz0), (Ix1, 0, Iz1), (Px1, CH_H, Pz1)]]:
        faces.append((q, [down(*v) for v in q]))

    # fondo del cassettone
    q = [(Px0, CH_H, Pz0), (Px1, CH_H, Pz0), (Px1, CH_H, Pz1), (Px0, CH_H, Pz1)]
    faces.append((q, [down(*v) for v in q]))

    # fianchi esterni: combaciano con la piastrella accanto, quindi ne resta
    # visibile una sola per giunto e non c'e' nulla che possa sfarfallare
    for q, ax in [([(Ox1, 0, Oz0), (Ox0, 0, Oz0), (Ox0, SLAB, Oz0), (Ox1, SLAB, Oz0)], 'x'),
                  ([(Ox0, 0, Oz1), (Ox1, 0, Oz1), (Ox1, SLAB, Oz1), (Ox0, SLAB, Oz1)], 'x'),
                  ([(Ox0, 0, Oz0), (Ox0, 0, Oz1), (Ox0, SLAB, Oz1), (Ox0, SLAB, Oz0)], 'z'),
                  ([(Ox1, 0, Oz1), (Ox1, 0, Oz0), (Ox1, SLAB, Oz0), (Ox1, SLAB, Oz1)], 'z')]:
        faces.append((q, [side(v[0] + TS if ax == 'x' else v[2], v[1]) for v in q]))

    # estradosso: non si vede mai, ma la mesh resta chiusa come il pavimento
    q = [(Ox0, SLAB, Oz0), (Ox0, SLAB, Oz1), (Ox1, SLAB, Oz1), (Ox1, SLAB, Oz0)]
    faces.append((q, [down(*v) for v in q]))

    # glTF (x, y, z) -> Blender (x, -z, y); la V delle UV e' ribaltata
    me = bpy.data.meshes.new(NAME)
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for verts, uvs in faces:
        f = bm.faces.new([bm.verts.new((v[0], -v[2], v[1])) for v in verts])
        f.smooth = False       # spigoli vivi: le normali restano per faccia
        for loop, uv in zip(f.loops, uvs):
            loop[uv_layer].uv = (uv[0], 1.0 - uv[1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()

    for o in [o for o in bpy.data.objects if o.name.startswith(NAME)]:
        bpy.data.objects.remove(o, do_unlink=True)
    obj = bpy.data.objects.new(NAME, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def check(obj):
    """La mesh deve essere chiusa e occupare esattamente l'impronta del pavimento."""
    me = obj.data
    counts = {}
    for p in me.polygons:
        for k in p.edge_keys:
            counts[k] = counts.get(k, 0) + 1
    assert all(v == 2 for v in counts.values()), "mesh non chiusa"
    lo = [min(v.co[i] for v in me.vertices) for i in range(3)]
    hi = [max(v.co[i] for v in me.vertices) for i in range(3)]
    assert [round(c, 4) for c in lo] == [-TS, -TS, 0.0], lo
    assert [round(c, 4) for c in hi] == [0.0, 0.0, SLAB], hi
    uvs = [l.uv for l in me.uv_layers[0].data]
    assert all(0.003 <= u <= 0.497 for u, _ in uvs), "UV fuori dalla pietra dell'atlante"
    print("mesh chiusa, %d facce, bbox e UV verificate" % len(me.polygons))


def export(obj):
    out_dir = os.path.join(tempfile.gettempdir(), "cg_ceiling_export")
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(out_dir, NAME + ".gltf"),
        export_format='GLTF_SEPARATE', use_selection=True, export_apply=True,
        export_yup=True, export_normals=True, export_texcoords=True,
        export_materials='NONE', export_extras=False)
    return out_dir


if __name__ == "__main__":
    obj = build()
    check(obj)
    out = export(obj)
    print("\nesportato in %s" % out)
    print("ora:  python tools/convert_assets.py \"%s\"" % os.path.join(out, NAME + ".gltf"))
