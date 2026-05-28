#!/usr/bin/env bash
# Generate a tiny PPM corpus for smoke-testing zjr-calibrate.
# Writes three 96x96 P6 PPM files into the first positional argument
# (defaults to ./refs).
set -euo pipefail
out="${1:-refs}"
mkdir -p "$out"

python3 - "$out" <<'PY'
import os, sys, struct, math
out = sys.argv[1]
W, H = 96, 96
def write_ppm(path, pixel_fn):
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (W, H))
        for y in range(H):
            row = bytearray()
            for x in range(W):
                row.extend(pixel_fn(x, y))
            f.write(bytes(row))
def stripes(x, y):
    r = (x * 7 + y * 3) % 240 + ((x ^ y) * 2654435761 & 0x0F)
    g = (x * 5 + y * 11) % 220 + (((x ^ y) * 2654435761) >> 4 & 0x1F)
    b = (x * 13 + y * 2) % 200 + (((x ^ y) * 2654435761) >> 9 & 0x3F)
    return (r & 0xFF, g & 0xFF, b & 0xFF)
def checker(x, y):
    on = ((x // 8) + (y // 8)) & 1
    base = 30 if on else 200
    return (base, (base + 40) & 0xFF, (base + 80) & 0xFF)
def noise(x, y):
    seed = (x * 1664525 + y * 1013904223) & 0xFFFFFFFF
    seed ^= seed >> 16
    return (seed & 0xFF, (seed >> 8) & 0xFF, (seed >> 16) & 0xFF)
write_ppm(os.path.join(out, "stripes.ppm"), stripes)
write_ppm(os.path.join(out, "checker.ppm"), checker)
write_ppm(os.path.join(out, "noise.ppm"), noise)
print(f"wrote 3 PPMs to {out}")
PY
