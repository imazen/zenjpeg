#!/usr/bin/env python3
"""vcpu_resource_sweep TSV -> vCPU-axis findings (codec-agnostic).

Per (path, effort, size) stratum reports the time speedup curve, the
memory-vs-threads behaviour (marginal delta + a linear gamma fit), and the
thread-independent estimate_encode prediction vs measurement. Reads the
`codec` column to pick each encoder's fixed-overhead constant.

Usage: vcpu_resource_fit.py <sweep.tsv>
"""
import sys
from collections import defaultdict

# heuristics.rs fixed-overhead per (codec, path), KiB.
FIXED_KB = {
    ("jxl-encoder", "lossy"): 16 * 1024,
    ("jxl-encoder", "lossless"): 20 * 1024,
    ("zenpng", "lossless"): 6 * 1024,
    ("zenavif", "lossy"): 8 * 1024,
    ("zenjpeg", "lossy"): 4 * 1024,
}


def load(fn):
    rows = []
    with open(fn) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(hdr)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(hdr):
                continue
            def fnum(k):
                try:
                    return float(c[ix[k]])
                except (ValueError, KeyError):
                    return None
            rows.append(dict(
                codec=c[ix["codec"]], path=c[ix["path"]], effort=int(c[ix["effort"]]),
                w=int(c[ix["width"]]), px=int(c[ix["pixels"]]), threads=int(c[ix["threads"]]),
                est_typ=fnum("est_typ_kb"), est_max=fnum("est_max_kb"), est_time=fnum("est_time_ms"),
                ph=fnum("meas_peak_heap_kb"), pr=fnum("meas_peak_rss_kb"),
                vmhwm=fnum("meas_vmhwm_kb"), delta=fnum("meas_delta_kb"),
                wall=fnum("meas_wall_ms"),
            ))
    return rows


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return None, None
    sx, sy, sxx, sxy = sum(xs), sum(ys), sum(x * x for x in xs), sum(x * y for x, y in zip(xs, ys))
    d = n * sxx - sx * sx
    if d == 0:
        return None, None
    b = (n * sxy - sx * sy) / d
    return (sy - b * sx) / n, b


def main():
    rows = load(sys.argv[1])
    by = defaultdict(list)
    for r in rows:
        by[(r["codec"], r["path"], r["effort"], r["w"])].append(r)
    print("=" * 78)
    print(f"vCPU RESOURCE SWEEP — {rows[0]['codec'] if rows else '?'}")
    print("=" * 78)
    for k in sorted(by):
        codec, path, effort, w = k
        g = sorted(by[k], key=lambda r: r["threads"])
        px = g[0]["px"]
        fixed = FIXED_KB.get((codec, path), 0)
        input_kb = px * 3 // 1024
        print(f"\n### {codec} {path} e{effort}  {w}x{w} ({px/1e6:.2f} MP)")
        print(f"  {'thr':>3} {'wall_ms':>9} {'speedup':>7} {'eff%':>5} "
              f"{'delta_MB':>8} {'peakRSS_MB':>10} {'peakHeap_MB':>11}")
        wall1 = next((r["wall"] for r in g if r["threads"] == 1 and r["wall"]), None)
        for r in g:
            sp = (wall1 / r["wall"]) if (wall1 and r["wall"]) else None
            eff = (sp / r["threads"] * 100) if sp else None
            ph = f"{r['ph']/1024:>11.1f}" if r["ph"] else f"{'—':>11}"
            print(f"  {r['threads']:>3} {r['wall'] or 0:>9.1f} "
                  f"{(f'{sp:.2f}x' if sp else '—'):>7} {(f'{eff:.0f}' if eff else '—'):>5} "
                  f"{(r['delta'] or 0)/1024:>8.1f} {(r['vmhwm'] or 0)/1024:>10.1f} {ph}")
        ts = [r["threads"] for r in g if r["delta"]]
        ds = [r["delta"] for r in g if r["delta"]]
        a, b = linfit(ts, ds)
        if b is not None:
            d1 = ds[0] if ds else 1
            print(f"  fit: delta_kb ≈ {a:.0f} + {b:.0f}·threads "
                  f"(γ = {b/1024:.2f} MB/thread; peak grows {b*max(ts)/max(d1,1)*100:.0f}% over 1→{max(ts)}T)")
        et, em = g[0]["est_typ"], g[0]["est_max"]
        if et:
            wp = et - fixed - input_kb
            d1 = next((r["delta"] for r in g if r["threads"] == 1), None)
            prN = max((r["vmhwm"] for r in g if r["vmhwm"]), default=None)
            print(f"  EST: typ={et/1024:.0f} MB (working_pred={wp/1024:.0f} MB), max={em/1024:.0f} MB")
            if d1:
                print(f"       working_pred/measured-delta_t1 = {wp/max(d1,1):.2f}×")
            if prN:
                print(f"       est_max/measured-peakRSS-maxThr = {em/max(prN,1):.2f}× (cover={'OK' if em>=prN else 'UNDER'})")
    print()


if __name__ == "__main__":
    main()
