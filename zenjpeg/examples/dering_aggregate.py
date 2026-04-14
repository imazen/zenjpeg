#!/usr/bin/env python3
"""Aggregate dering_corpus_sweep CSV into the headline tables.

Usage:
  python3 dering_aggregate.py <csv_path> [--pareto-names NAME1,NAME2,NAME3]

Produces:
  1. Headline table (mean delta across corpus at each Q)
  2. Photo vs graphic split
  3. Pareto curves for 3 representative images
  4. Per-image summary
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from statistics import mean, median


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(
                dict(
                    category=r["category"],
                    name=r["name"],
                    width=int(r["width"]),
                    height=int(r["height"]),
                    q=int(r["q"]),
                    dering=r["dering"] == "true",
                    bytes=int(r["bytes"]),
                    ssim2=float(r["ssim2"]),
                )
            )
    return rows


def index_paired(rows):
    """Return {(category, name, q): {'on': row, 'off': row}}."""
    by_key = defaultdict(dict)
    for r in rows:
        k = (r["category"], r["name"], r["q"])
        by_key[k]["on" if r["dering"] else "off"] = r
    return by_key


def pct(a, b):
    return 100.0 * (a - b) / b if b else 0.0


def verdict(size_pct, ssim_delta, n):
    """Return verdict for dering-on vs dering-off at a Q."""
    # dering adds cost (positive size_pct). SSIM delta positive = dering helps.
    if abs(size_pct) < 0.05 and abs(ssim_delta) < 0.01:
        return "tie"
    if ssim_delta > 0.10 and size_pct < 1.0:
        return "WORTH IT"
    if ssim_delta > 0.05 and size_pct < 0.5:
        return "marginal+"
    if ssim_delta < -0.05:
        return "HURTS QUALITY"
    if size_pct > 0.5 and ssim_delta < 0.05:
        return "WASTES BYTES"
    return "~tie"


def headline_table(paired, title, filter_cat=None):
    print(f"\n## {title}")
    print()
    print("| Q  | N imgs | dering size Δ | dering SSIM2 Δ | verdict |")
    print("|----|-------:|--------------:|---------------:|---------|")
    by_q = defaultdict(list)
    for (cat, name, q), pair in paired.items():
        if filter_cat and cat != filter_cat:
            continue
        if "on" not in pair or "off" not in pair:
            continue
        on, off = pair["on"], pair["off"]
        size_pct = pct(on["bytes"], off["bytes"])
        ssim_d = on["ssim2"] - off["ssim2"]
        by_q[q].append((size_pct, ssim_d, on["bytes"], off["bytes"]))

    for q in sorted(by_q):
        items = by_q[q]
        n = len(items)
        sp = mean(s for s, _, _, _ in items)
        sd = mean(d for _, d, _, _ in items)
        # absolute byte delta
        b_on = sum(on for _, _, on, _ in items)
        b_off = sum(of for _, _, _, of in items)
        v = verdict(sp, sd, n)
        print(
            f"| {q:>2} | {n:>6} | {sp:+7.3f}% ({b_on-b_off:+d} B) | {sd:+7.3f} pts | {v} |"
        )


def image_split(paired):
    print("\n## Image-type split summary")
    print()
    for cat in ("photo", "graphic", "frymire"):
        print(f"\n### {cat}")
        print()
        print("| Q  | N | mean size Δ | mean SSIM2 Δ | median size Δ | median SSIM2 Δ |")
        print("|----|--:|------------:|-------------:|--------------:|---------------:|")
        by_q = defaultdict(list)
        for (c, n, q), pair in paired.items():
            if c != cat or "on" not in pair or "off" not in pair:
                continue
            on, off = pair["on"], pair["off"]
            by_q[q].append((pct(on["bytes"], off["bytes"]), on["ssim2"] - off["ssim2"]))
        for q in sorted(by_q):
            xs = by_q[q]
            if not xs:
                continue
            sp_m = mean(s for s, _ in xs)
            sd_m = mean(d for _, d in xs)
            sp_md = median(s for s, _ in xs)
            sd_md = median(d for _, d in xs)
            print(
                f"| {q:>2} | {len(xs):>2} | {sp_m:+7.3f}% | {sd_m:+7.3f} pts | {sp_md:+7.3f}% | {sd_md:+7.3f} pts |"
            )


def pareto_lines(paired, names):
    print(f"\n## Pareto traces (size vs SSIM2) for {names}")
    for target_name in names:
        print(f"\n### {target_name}")
        print()
        print("| Q  | bytes on | ssim2 on | bytes off | ssim2 off | Δbytes | ΔSSIM2 |")
        print("|----|---------:|---------:|----------:|----------:|-------:|-------:|")
        qs = sorted({q for (_, n, q) in paired if n == target_name})
        for q in qs:
            # There may be multiple categories with same name; pick any
            for (cat, nm, qq), pair in paired.items():
                if nm == target_name and qq == q and "on" in pair and "off" in pair:
                    on, off = pair["on"], pair["off"]
                    dB = on["bytes"] - off["bytes"]
                    dS = on["ssim2"] - off["ssim2"]
                    print(
                        f"| {q:>2} | {on['bytes']:>8} | {on['ssim2']:>8.3f} | {off['bytes']:>9} | {off['ssim2']:>9.3f} | {dB:>+6d} | {dS:>+6.3f} |"
                    )
                    break


def worst_offenders(paired, top_n=10):
    """Find images where dering hurts most (high size cost, low ssim gain)."""
    print(f"\n## Worst dering offenders (per-image, across all Qs)")
    print()
    by_img = defaultdict(lambda: [0, 0, 0.0, 0])  # bytes_on, bytes_off, ssim_delta_sum, n
    for (cat, name, q), pair in paired.items():
        if "on" not in pair or "off" not in pair:
            continue
        on, off = pair["on"], pair["off"]
        rec = by_img[(cat, name)]
        rec[0] += on["bytes"]
        rec[1] += off["bytes"]
        rec[2] += on["ssim2"] - off["ssim2"]
        rec[3] += 1

    rows = []
    for (cat, name), (b_on, b_off, s_sum, n) in by_img.items():
        size_pct = pct(b_on, b_off)
        avg_ssim = s_sum / n
        # score: byte cost dominates when ssim delta is near zero
        rows.append((cat, name, size_pct, avg_ssim, b_on - b_off, n))
    # Sort by dering giving bad ratio: high cost, low/neg ssim gain
    rows.sort(key=lambda x: (x[3], -x[2]))  # ascending ssim_delta, then high cost

    print("| rank | category | image | size % Δ | avg SSIM2 Δ | bytes Δ | n Q |")
    print("|------|----------|-------|---------:|------------:|--------:|----:|")
    for i, r in enumerate(rows[:top_n]):
        print(f"| {i+1} | {r[0]} | {r[1]} | {r[2]:+7.3f}% | {r[3]:+7.3f} | {r[4]:>+d} | {r[5]} |")

    print("\n## Best dering wins (largest positive SSIM gain)")
    print()
    rows.sort(key=lambda x: -x[3])
    print("| rank | category | image | size % Δ | avg SSIM2 Δ | bytes Δ | n Q |")
    print("|------|----------|-------|---------:|------------:|--------:|----:|")
    for i, r in enumerate(rows[:top_n]):
        print(f"| {i+1} | {r[0]} | {r[1]} | {r[2]:+7.3f}% | {r[3]:+7.3f} | {r[4]:>+d} | {r[5]} |")


def main():
    if len(sys.argv) < 2:
        print("usage: dering_aggregate.py CSV [--pareto NAMES]")
        sys.exit(2)
    csv_path = sys.argv[1]
    pareto = ["frymire", "codec_wiki", "baby-lossless"]
    if "--pareto" in sys.argv:
        pareto = sys.argv[sys.argv.index("--pareto") + 1].split(",")

    rows = load(csv_path)
    paired = index_paired(rows)

    print(f"# Dering corpus sweep report")
    print(f"\nSource CSV: `{csv_path}`")
    print(f"Rows: {len(rows)}, paired (img,Q): {len(paired)}")
    cats = defaultdict(set)
    for r in rows:
        cats[r["category"]].add(r["name"])
    for c, names in cats.items():
        print(f"- {c}: {len(names)} images")

    headline_table(paired, "Headline: corpus-wide dering impact (ALL images)")
    headline_table(paired, "Photos only (CID22)", filter_cat="photo")
    headline_table(paired, "Graphics/screenshots only (gb82-sc)", filter_cat="graphic")
    headline_table(paired, "frymire only", filter_cat="frymire")

    image_split(paired)

    pareto_lines(paired, pareto)

    worst_offenders(paired, top_n=15)


if __name__ == "__main__":
    main()
