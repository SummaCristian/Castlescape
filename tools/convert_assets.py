#!/usr/bin/env python3
"""
Converte asset glTF/GLB esterni nella forma che Starter.hpp sa leggere.

Il loader del progetto ha tre limiti che questo script aggira:
  - legge solo .gltf ASCII (LoadASCIIFromFile), mai .glb
  - ignora bufferView.byteStride: gli attributi devono essere compatti e non interlacciati
  - concatena tutte le primitive in un unico vertex buffer con una sola texture,
    quindi ogni materiale deve stare in un file a se'

In piu' applica ai vertici le trasformazioni dei nodi (meno la traslazione della
radice), cosi' il posizionamento resta tutto nel scene.json: Scene.hpp scarta la
matrice del modello appena una entry dichiara translate/rotate/scale.

Uso:
    python convert_assets.py <input.glb|input.gltf|cartella> [opzioni]

Opzioni:
    --models-out DIR    default skeleton/source/assets/models/Dracula
    --tex-out DIR       default skeleton/source/assets/textures/Dracula
    --tex-size N        lato massimo delle texture, default 1024 (0 = non ridimensionare)
"""

import argparse
import base64
import hashlib
import json
import os
import re
import struct
import sys

import numpy as np
from PIL import Image

COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
COMPONENT_MAX = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def safe_name(name):
    """Nomi file ASCII: Unity lascia in giro & e spazi non-breaking (U+00A0)."""
    name = name.replace(" ", " ").replace("&", "And")  # U+00A0 = spazio non-breaking
    name = re.sub(r"\(Clone\)", "", name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_") or "mesh"


class Asset:
    """Un .gltf o .glb caricato, con i buffer risolti."""

    def __init__(self, path):
        self.path = path
        self.dir = os.path.dirname(os.path.abspath(path))
        self.glb_bin = None
        with open(path, "rb") as fh:
            head = fh.read(4)
            fh.seek(0)
            if head == b"glTF":
                self._read_glb(fh.read())
            else:
                self.json = json.load(open(path, "r", encoding="utf-8"))
        self.buffers = [self._resolve_buffer(b) for b in self.json.get("buffers", [])]

    def _read_glb(self, data):
        offset = 12
        while offset < len(data):
            length, kind = struct.unpack_from("<II", data, offset)
            chunk = data[offset + 8: offset + 8 + length]
            if kind == 0x4E4F534A:
                self.json = json.loads(chunk)
            elif kind == 0x004E4942:
                self.glb_bin = chunk
            offset += 8 + length

    def _resolve_buffer(self, buf):
        uri = buf.get("uri")
        if uri is None:
            return self.glb_bin
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        return open(os.path.join(self.dir, uri), "rb").read()

    def view_bytes(self, view_index):
        view = self.json["bufferViews"][view_index]
        data = self.buffers[view["buffer"]]
        start = view.get("byteOffset", 0)
        return data[start: start + view["byteLength"]], view

    def read_accessor(self, index):
        """Legge un accessor rispettando byteStride e normalized."""
        acc = self.json["accessors"][index]
        ncomp = TYPE_COMPONENTS[acc["type"]]
        dtype = np.dtype(COMPONENT_DTYPE[acc["componentType"]]).newbyteorder("<")
        count = acc["count"]

        if "bufferView" not in acc:
            out = np.zeros((count, ncomp), dtype=np.float32)
        else:
            blob, view = self.view_bytes(acc["bufferView"])
            item = dtype.itemsize * ncomp
            stride = view.get("byteStride") or item
            base = acc.get("byteOffset", 0)
            raw = np.frombuffer(blob, dtype=np.uint8)
            # gather esplicito: e' l'unico modo sicuro con dati interlacciati
            picks = (base + np.arange(count) * stride)[:, None] + np.arange(item)[None, :]
            out = np.frombuffer(raw[picks].tobytes(), dtype=dtype).reshape(count, ncomp)

        if acc.get("normalized") and acc["componentType"] in COMPONENT_MAX:
            out = out.astype(np.float32) / COMPONENT_MAX[acc["componentType"]]
            if acc["componentType"] in (5120, 5122):
                out = np.maximum(out, -1.0)
        return out


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = m @ np.diag(list(node["scale"]) + [1.0])
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ])
        m = r @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


def collect_instances(asset):
    """Percorre la scena e restituisce (nodo, mesh, matrice) per ogni mesh.

    La traslazione del nodo radice viene tolta: e' la posizione del GameObject
    in Unity, non fa parte del modello.
    """
    js = asset.json
    scene = js.get("scenes", [{}])[js.get("scene", 0)]
    roots = scene.get("nodes", list(range(len(js.get("nodes", [])))))
    found = []

    def walk(index, parent):
        node = js["nodes"][index]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            found.append((node, node["mesh"], world))
        for child in node.get("children", []):
            walk(child, world)

    for root in roots:
        walk(root, np.eye(4))

    if found:
        # ancora tutto rispetto alla radice del primo nodo con mesh
        origin = found[0][2][:3, 3].copy()
        shift = np.eye(4)
        shift[:3, 3] = -origin
        found = [(n, m, shift @ w) for n, m, w in found]
    return found


def base_color_source(asset, primitive):
    mat_index = primitive.get("material")
    if mat_index is None:
        return None
    mat = asset.json.get("materials", [])[mat_index]
    tex = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if tex is None:
        return None
    return asset.json["textures"][tex["index"]].get("source")


def export_texture(asset, source_index, out_dir, name, max_size, cache):
    if source_index is None:
        return None
    image = asset.json["images"][source_index]
    if "uri" in image:
        uri = image["uri"]
        if uri.startswith("data:"):
            blob = base64.b64decode(uri.split(",", 1)[1])
        else:
            blob = open(os.path.join(asset.dir, uri), "rb").read()
    else:
        blob, _ = asset.view_bytes(image["bufferView"])

    digest = hashlib.md5(blob).hexdigest()
    if digest in cache:
        return cache[digest]

    img = Image.open(__import__("io").BytesIO(blob))
    # le zone trasparenti dell'atlas sono UV inutilizzate: appiattirle evita
    # sorprese se lo shader campiona l'alpha
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (0, 0, 0))
        flat.paste(img, mask=img.split()[-1])
        img = flat
    else:
        img = img.convert("RGB")
    if max_size and max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize((round(img.width * ratio), round(img.height * ratio)), Image.LANCZOS)

    out = os.path.join(out_dir, name + ".png")
    img.save(out, "PNG", optimize=True)
    cache[digest] = out
    return out


def write_gltf(name, pos, nrm, uv, idx, models_dir):
    """Scrive .gltf + .bin con attributi compatti e non interlacciati."""
    chunks, views, accessors = [], [], []
    offset = 0

    def add(array, target, acc_type, comp_type, extra=None):
        nonlocal offset
        blob = np.ascontiguousarray(array).tobytes()
        pad = (-len(blob)) % 4
        chunks.append(blob + b"\x00" * pad)
        views.append({
            "buffer": 0, "byteOffset": offset, "byteLength": len(blob), "target": target,
        })
        acc = {
            "bufferView": len(views) - 1, "byteOffset": 0,
            "componentType": comp_type, "count": len(array), "type": acc_type,
        }
        if extra:
            acc.update(extra)
        accessors.append(acc)
        offset += len(blob) + pad
        return len(accessors) - 1

    a_pos = add(pos, 34962, "VEC3", 5126,
                {"min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
    a_nrm = add(nrm, 34962, "VEC3", 5126)
    a_uv = add(uv, 34962, "VEC2", 5126)
    a_idx = add(idx, 34963, "SCALAR", 5125)

    bin_name = name + ".bin"
    with open(os.path.join(models_dir, bin_name), "wb") as fh:
        fh.write(b"".join(chunks))

    doc = {
        "asset": {"version": "2.0", "generator": "convert_assets.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv},
            "indices": a_idx, "mode": 4,
        }]}],
        "buffers": [{"uri": bin_name, "byteLength": offset}],
        "bufferViews": views,
        "accessors": accessors,
    }
    path = os.path.join(models_dir, name + ".gltf")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return path


def convert(path, models_dir, tex_dir, max_size, tex_cache, entries):
    asset = Asset(path)
    stem = safe_name(os.path.splitext(os.path.basename(path))[0])
    instances = collect_instances(asset)

    for node, mesh_index, world in instances:
        mesh = asset.json["meshes"][mesh_index]
        normal_mat = np.linalg.inv(world[:3, :3]).T

        for prim_i, prim in enumerate(mesh["primitives"]):
            attrs = prim["attributes"]
            if "POSITION" not in attrs or prim.get("mode", 4) != 4:
                print("  salto una primitiva senza POSITION o non triangolare")
                continue

            pos = asset.read_accessor(attrs["POSITION"]).astype(np.float64)
            count = len(pos)
            nrm = (asset.read_accessor(attrs["NORMAL"]).astype(np.float64)
                   if "NORMAL" in attrs else np.tile([0.0, 1.0, 0.0], (count, 1)))
            uv = (asset.read_accessor(attrs["TEXCOORD_0"]).astype(np.float32)
                  if "TEXCOORD_0" in attrs else np.zeros((count, 2), np.float32))
            idx = (asset.read_accessor(prim["indices"]).reshape(-1).astype(np.uint32)
                   if "indices" in prim else np.arange(count, dtype=np.uint32))

            # le mesh multi-materiale condividono un unico pool di vertici fra le
            # primitive: tenendo solo quelli indicizzati il file non si porta dietro
            # la geometria degli altri materiali
            used, remap = np.unique(idx, return_inverse=True)
            if len(used) < count:
                pos, nrm, uv = pos[used], nrm[used], uv[used]
                idx = remap.astype(np.uint32)

            pos = (pos @ world[:3, :3].T) + world[:3, 3]
            nrm = nrm @ normal_mat.T
            lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.where(lengths == 0, 1, lengths)

            name = stem
            if len(mesh["primitives"]) > 1 or len(instances) > 1:
                mat_index = prim.get("material")
                label = (asset.json["materials"][mat_index].get("name")
                         if mat_index is not None else None) or ("part%d" % prim_i)
                name = "%s_%s" % (stem, safe_name(label))

            out = write_gltf(name, pos.astype(np.float32), nrm.astype(np.float32),
                             uv.astype(np.float32), idx, models_dir)
            tex = export_texture(asset, base_color_source(asset, prim),
                                 tex_dir, name, max_size, tex_cache)

            size = pos.max(0) - pos.min(0)
            print("  %-42s %5d tris  bbox %.2f x %.2f x %.2f" %
                  (os.path.basename(out), len(idx) // 3, size[0], size[1], size[2]))
            entries.append((name, out, tex, pos.min(0), pos.max(0)))


def rel_asset_path(path):
    """Percorso come lo vuole scene.json, cioe' relativo a source/."""
    path = os.path.abspath(path).replace("\\", "/")
    marker = "/assets/"
    return "assets/" + path.split(marker, 1)[1] if marker in path else path


def scene_id(name):
    """SM_WallStraight_01 -> wallStraight01, come gli id gia' usati in scene.json."""
    name = re.sub(r"^SM_", "", name)
    parts = [p for p in re.split(r"[_.-]+", name) if p]
    if not parts:
        return "model"
    head = parts[0][0].lower() + parts[0][1:]
    return head + "".join(p[0].upper() + p[1:] for p in parts[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--models-out", default="skeleton/source/assets/models/Dracula")
    ap.add_argument("--tex-out", default="skeleton/source/assets/textures/Dracula")
    ap.add_argument("--tex-size", type=int, default=1024)
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                 if f.lower().endswith((".glb", ".gltf"))]
        # una cartella per modello: prendi anche i .gltf annidati di un livello
        for sub in sorted(os.listdir(args.input)):
            full = os.path.join(args.input, sub)
            if os.path.isdir(full):
                files += [os.path.join(full, f) for f in sorted(os.listdir(full))
                          if f.lower().endswith((".glb", ".gltf"))]
    else:
        files = [args.input]

    if not files:
        sys.exit("nessun .glb o .gltf trovato in " + args.input)

    os.makedirs(args.models_out, exist_ok=True)
    os.makedirs(args.tex_out, exist_ok=True)

    tex_cache, entries = {}, []
    for f in files:
        print(os.path.basename(f))
        convert(f, args.models_out, args.tex_out, args.tex_size, tex_cache, entries)

    print("\n--- models per scene.json ---")
    for name, model, _, mn, mx in entries:
        print('{"id": "%s", "VD": "VDposNormUV", "model": "%s", "format":"GLTF"},'
              % (scene_id(name), rel_asset_path(model)))
    print("\n--- textures per scene.json ---")
    seen = set()
    for name, _, tex, _, _ in entries:
        if tex and tex not in seen:
            seen.add(tex)
            print('{"id": "%sTex", "texture": "%s", "format": "C"},'
                  % (scene_id(name), rel_asset_path(tex)))


if __name__ == "__main__":
    main()
