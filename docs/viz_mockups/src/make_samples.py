# Generate ColoredMNIST-style sample tiles (real MNIST digits, DomainBed's colouring rule) for the mockups.
import base64, io, json, sys
import numpy as np
from PIL import Image
raw = sys.argv[1]
imgs = np.fromfile(f"{raw}/t10k-images-idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28)
digits = np.fromfile(f"{raw}/t10k-labels-idx1-ubyte", dtype=np.uint8)[8:]
rng = np.random.RandomState(7)

def tile(i, color):
    a = imgs[i].astype(np.float32)
    rgb = np.zeros((28, 28, 3), np.uint8)
    rgb[..., 0 if color == 0 else 1] = a.astype(np.uint8)       # channel 0 red, channel 1 green
    im = Image.fromarray(rgb).resize((56, 56), Image.NEAREST)
    b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def sample(e, n, start):
    out = []
    for k in range(n):
        i = start + k * 37
        d = int(digits[i]); y0 = int(d >= 5)
        flip = bool(rng.rand() < 0.25); y = y0 ^ flip
        c = y ^ int(rng.rand() < e)
        out.append({"img": tile(i, c), "digit": d, "label": y, "flipped": flip, "color": c})
    return out

intro = {name: sample(e, 6, s) for name, e, s in (("env0", 0.1, 11), ("env1", 0.2, 501), ("env2", 0.9, 1201))}
# main page: 4 groups x 8 tiles, prediction simulated for a model that uses reversed colour + shape
groups = {"A": (False, True), "B": (False, False), "C": (True, True), "D": (True, False)}  # (label flipped, colour agrees)
main = []
i = 2001
for g, (flip, agree) in groups.items():
    got = 0
    while got < 8:
        i += 13
        d = int(digits[i]); y0 = int(d >= 5); y = y0 ^ flip; c = y if agree else 1 - y
        # simulated SOTA: follows reversed colour 90% of the time, shape otherwise
        acc = {"A": 0.97, "B": 0.92, "C": 0.21, "D": 0.88}[g]      # same as the mockup's diagnostic bars
        pred = y if rng.rand() < acc else 1 - y
        main.append({"img": tile(i, c), "digit": d, "label": y, "flipped": flip, "color": c,
                     "pred": int(pred), "group": g, "p": round(0.55 + 0.44 * rng.rand(), 2)})
        got += 1
json.dump({"intro": intro, "main": main}, open(sys.argv[2], "w"))
print("ok", len(main))
