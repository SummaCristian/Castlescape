"""Scratch: find the mortar joints in the floor tile's quadrant of the atlas.

A global threshold does not work: the texture's dark weathering blotches are as
dark as the joints and far larger, so they come out as fat false joints. What
separates a joint from a stain is that a joint is THIN and darker than its own
immediate surroundings, so this uses a top-hat instead -- lum minus a wide
rolling baseline -- and keeps only runs a few texels wide.
"""
import numpy as np
from PIL import Image

W = H = 1024
U0, U1, V0, V1 = 0.0032, 0.4968, 0.5032, 0.9968
BASE = 41        # rolling-baseline window, texels: wider than any joint
DEPTH = 6.0      # how far below the baseline counts as a joint
WIDTH = (3, 13)  # plausible joint width in texels

im = Image.open(
    r"C:\Users\Simo\Repo\Computer_Graphics\skeleton\source\assets\textures\Dracula\SM_StoneFloor_01.png"
).convert("RGB")
a = np.asarray(im.crop((round(U0 * W), round(V0 * H), round(U1 * W), round(V1 * H)))).astype(float)
lum = a @ np.array([0.2126, 0.7152, 0.0722])
h, w = lum.shape


def baseline(v, n):
    pad = np.pad(v, n // 2, mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")[:len(v)]


def joints(v):
    d = baseline(v, BASE) - v
    mask = d > DEPTH
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if WIDTH[0] <= i - s <= WIDTH[1]:
                out.append((s, i))
            s = None
    return out


print("crop %dx%d" % (w, h))
vj = joints(lum.mean(axis=0))
print("\nvertical joints:", [(s, e) for s, e in vj])
centres = [(s + e) / 2 for s, e in vj]
print("centres:", ["%.1f" % c for c in centres])
print("spacing:", ["%.1f" % (centres[i + 1] - centres[i]) for i in range(len(centres) - 1)])

edges = [0.0] + centres + [float(w)]
print("\nhorizontal joints per column:")
for i in range(len(edges) - 1):
    x0, x1 = int(round(edges[i])), int(round(edges[i + 1]))
    if x1 - x0 < 12:
        print(" col %d  x %3d..%3d  (sliver, skipped)" % (i, x0, x1))
        continue
    hj = joints(lum[:, x0 + 3:x1 - 3].mean(axis=1))
    cs = [(s + e) / 2 for s, e in hj]
    print(" col %d  x %3d..%3d  w %3d   joints y: %s" % (
        i, x0, x1, x1 - x0, ", ".join("%.1f" % c for c in cs)))
    print("        spacing: %s" % ", ".join(
        "%.1f" % (cs[j + 1] - cs[j]) for j in range(len(cs) - 1)))
