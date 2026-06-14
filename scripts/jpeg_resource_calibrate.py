#!/usr/bin/env python3
"""Calibrate JPEG encode/decode resource use (peak mem + wall + CPU time),
focused on the RD features the guessed-throughput heuristics ignore:
TRELLIS (on/off) and BOUNDARY-RD (on/off).

Drives `examples/jpeg_probe` (one process per op → clean per-op VmHWM peak).
Input is raw packed RGB8 (`PIL.tobytes()`), so the probe needs no image dep.
Single-thread (no `parallel` feature) so wall ≈ user.
"""
import argparse, subprocess, datetime, socket
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def gen_raw(src, n, outdir):
    im = Image.open(src)
    if max(im.size) < n:
        return None
    im = im.convert("RGB").resize((n, n), Image.LANCZOS)
    p = outdir / f"{Path(src).stem}_{n}.rgb"
    p.write_bytes(im.tobytes())
    return p


def run(b, raw, n, mode, q, trellis, brd, outp):
    out = subprocess.run([b, str(raw), str(n), str(n), mode, str(q), str(trellis), str(brd), str(outp)],
                         capture_output=True, text=True).stdout
    return {k: v for k, v in (t.split("=", 1) for t in out.split() if "=" in t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/examples/jpeg_probe")
    ap.add_argument("--sizes", default="512,1024,2048")
    ap.add_argument("--qualities", default="50,85")
    ap.add_argument("--content-file", default=None)
    ap.add_argument("--content", action="append", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    quals = [int(x) for x in a.qualities.split(",")]
    configs = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (trellis, boundary_rd)
    content = [c.split(":") for c in a.content]
    if a.content_file:
        for line in Path(a.content_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                content.append(line.split(":"))

    date = datetime.date.today().isoformat()
    out = Path(a.out or f"benchmarks/jpeg_resource_{date}.tsv")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tmp = Path("/tmp/_jpegcal"); tmp.mkdir(exist_ok=True)

    cells = []
    for src, cls in content:
        for n in sizes:
            v = gen_raw(src, n, tmp)
            if v:
                cells.append((v, n, cls))

    rows = []
    total = len(cells) * len(quals) * len(configs)
    i = 0
    for (raw, n, cls) in cells:
        for q in quals:
            for (tr, brd) in configs:
                i += 1
                op = tmp / f"{raw.stem}_{q}_{tr}{brd}.jpg"
                enc = run(a.bin, raw, n, "encode", q, tr, brd, op)
                dec = run(a.bin, raw, n, "decode", q, tr, brd, op)
                px = n * n
                if not enc.get("bytes"):
                    print(f"[{i}/{total}] {cls} {n} q{q} t{tr}b{brd} ENCODE FAILED", flush=True)
                    continue
                rows.append((cls, n, px, q, tr, brd, "encode", int(enc["peak_kb"]), int(enc["delta_kb"]),
                             float(enc["wall_ms"]), float(enc["user_ms"]), float(enc["sys_ms"]), int(enc["bytes"])))
                if dec.get("peak_kb"):
                    rows.append((cls, n, px, q, tr, brd, "decode", int(dec["peak_kb"]), int(dec["delta_kb"]),
                                 float(dec["wall_ms"]), float(dec["user_ms"]), float(dec["sys_ms"]), int(enc["bytes"])))
                print(f"[{i}/{total}] {cls} {n}^2 q{q} trellis={tr} brd={brd} -> "
                      f"enc {int(enc['delta_kb'])//1024}MB {float(enc['wall_ms']):.0f}ms "
                      f"({float(enc['wall_ms'])*1e3/px:.3f}us/px) | dec {float(dec.get('wall_ms',0)):.0f}ms", flush=True)

    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write("content\tsize\tpixels\tquality\ttrellis\tbrd\top\tpeak_kb\tdelta_kb\twall_ms\tuser_ms\tsys_ms\tbytes\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    with open(str(out) + ".meta", "w") as f:
        f.write(f"# jpeg_resource_calibrate\ncommit: {commit}\nhost: {socket.gethostname()}\ndate: {date}\n"
                f"bin: {a.bin}\nsizes: {sizes}\nqualities: {quals}\nconfigs(trellis,brd): {configs}\n"
                f"content_classes: {sorted(set(c[1] for c in content))}\n"
                f"measure: jpeg_probe VmHWM delta + wall (Instant) + user/sys (/proc/self/stat), single-thread, "
                f"one process per op; ycbcr 4:2:0, progressive default.\n")
    print(f"\nwrote {out} ({len(rows)} rows)")

    # ---- fit: time/mem per (trellis,brd), and the multipliers vs baseline ----
    med = lambda v: sorted(v)[len(v) // 2]
    from collections import defaultdict
    enc = [r for r in rows if r[6] == "encode" and r[2] >= 512 * 512]
    print("\n=== ENCODE wall us/px + mem B/px per (trellis,brd) [px>=512^2, both q] ===")
    print(f"{'trellis':>7} {'brd':>3} {'n':>3} {'wall us/px p50':>15} {'mem B/px p50':>13}")
    g = defaultdict(list)
    for r in enc:
        g[(r[4], r[5])].append(r)
    base = None
    for k in sorted(g):
        v = g[k]
        wpp = med([r[9] * 1e3 / r[2] for r in v])
        mpp = med([r[8] * 1024.0 / r[2] for r in v])
        if k == (0, 0):
            base = wpp
        print(f"{k[0]:>7} {k[1]:>3} {len(v):>3} {wpp:>15.4f} {mpp:>13.1f}")
    if base:
        print(f"\n=== TIME MULTIPLIERS vs baseline (trellis=0,brd=0) ===")
        for k in sorted(g):
            wpp = med([r[9] * 1e3 / r[2] for r in g[k]])
            print(f"  trellis={k[0]} brd={k[1]}: {wpp/base:.2f}x")
    dec = [r for r in rows if r[6] == "decode" and r[2] >= 512 * 512]
    if dec:
        print(f"\n=== DECODE: mem p50={med([r[8]*1024.0/r[2] for r in dec]):.1f} B/px | "
              f"wall p50={med([r[9]*1e3/r[2] for r in dec]):.4f} us/px (n={len(dec)}) ===")


if __name__ == "__main__":
    main()
