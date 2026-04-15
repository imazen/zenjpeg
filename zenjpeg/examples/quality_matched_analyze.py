#!/usr/bin/env python3
"""Analyze quality_matched_sweep.csv to find the matching distance offset per Q.

For each Q:
- Compute mean(cjpegli_ssim2), mean(cjpegli_butter), mean(cjpegli_size)
- For each offset Δ, compute mean(zen_ssim2(Δ)), mean(zen_butter(Δ)), mean(zen_size(Δ))
- Find the Δ whose zen mean matches cjpegli mean (separately for each metric)
- Report size delta at matched Δ
- Compute photo-only and graphic-only stats

Usage:
  python3 quality_matched_analyze.py benchmarks/quality_matched_2026-04-14.csv
"""

import csv
import sys
from collections import defaultdict


def interpolate_crossing(offsets, means, target):
    """Given sorted (offset, mean) pairs sorted by offset, find Δ where mean = target.

    Assume `means` is roughly monotonically DECREASING in offset (higher distance = lower quality).
    Returns the linearly-interpolated Δ that crosses `target`.
    """
    pairs = sorted(zip(offsets, means), key=lambda p: p[0])
    # Find bracket
    for i in range(len(pairs) - 1):
        o1, m1 = pairs[i]
        o2, m2 = pairs[i + 1]
        if (m1 >= target >= m2) or (m1 <= target <= m2):
            if m2 == m1:
                return o1
            frac = (target - m1) / (m2 - m1)
            return o1 + frac * (o2 - o1)
    # No crossing — return boundary
    if pairs[0][1] > target and pairs[-1][1] > target:
        return pairs[-1][0]  # mean always above target → use largest offset
    if pairs[0][1] < target and pairs[-1][1] < target:
        return pairs[0][0]
    return None


def interpolate_value(offsets, values, delta):
    """Interpolate `values` at offset `delta`."""
    pairs = sorted(zip(offsets, values), key=lambda p: p[0])
    if delta <= pairs[0][0]:
        return pairs[0][1]
    if delta >= pairs[-1][0]:
        return pairs[-1][1]
    for i in range(len(pairs) - 1):
        o1, v1 = pairs[i]
        o2, v2 = pairs[i + 1]
        if o1 <= delta <= o2:
            if o2 == o1:
                return v1
            frac = (delta - o1) / (o2 - o1)
            return v1 + frac * (v2 - v1)
    return pairs[-1][1]


def analyze(csv_path, categories=None):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if categories is not None and r["category"] not in categories:
                continue
            rows.append(r)

    # Group: (q, source, offset) -> list of (name, bytes, ssim2, butter)
    qdata = defaultdict(lambda: defaultdict(list))
    for r in rows:
        q = int(r["q"])
        src = r["source"]
        off = float(r["offset"])
        key = (src, off)
        qdata[q][key].append(
            (
                r["name"],
                int(r["bytes"]),
                float(r["ssim2"]),
                float(r["butter"]),
            )
        )

    # Per-image structure: q -> name -> {(src, off): (bytes, ssim2, butter)}
    per_image = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        q = int(r["q"])
        per_image[q][r["name"]][(r["source"], float(r["offset"]))] = (
            int(r["bytes"]),
            float(r["ssim2"]),
            float(r["butter"]),
        )

    results = []
    for q in sorted(qdata.keys()):
        # cjpegli stats
        cpp = qdata[q][("cjpegli", 0.0)]
        cpp_mean_size = sum(x[1] for x in cpp) / len(cpp)
        cpp_mean_ssim = sum(x[2] for x in cpp) / len(cpp)
        cpp_mean_butt = sum(x[3] for x in cpp) / len(cpp)

        # Gather zen offsets
        zen_offs = sorted({off for (src, off) in qdata[q] if src == "zen"})
        zen_ssim_by_off = {}
        zen_butt_by_off = {}
        zen_size_by_off = {}
        for off in zen_offs:
            samples = qdata[q][("zen", off)]
            # Pair with cjpegli to get matched-set mean
            zen_ssim_by_off[off] = sum(x[2] for x in samples) / len(samples)
            zen_butt_by_off[off] = sum(x[3] for x in samples) / len(samples)
            zen_size_by_off[off] = sum(x[1] for x in samples) / len(samples)

        # At Δ=0 (no offset)
        zen0_ssim = zen_ssim_by_off.get(0.0)
        zen0_butt = zen_butt_by_off.get(0.0)

        ssim_excess = zen0_ssim - cpp_mean_ssim  # positive = zen better
        butt_excess = cpp_mean_butt - zen0_butt  # positive = zen better (lower butteraugli)

        # Find Δ matching each metric
        delta_ssim = interpolate_crossing(
            zen_offs,
            [zen_ssim_by_off[o] for o in zen_offs],
            cpp_mean_ssim,
        )
        delta_butt = interpolate_crossing(
            zen_offs,
            [zen_butt_by_off[o] for o in zen_offs],
            cpp_mean_butt,
        )

        # Size at matched Δ
        size_at_ssim = (
            interpolate_value(zen_offs, [zen_size_by_off[o] for o in zen_offs], delta_ssim)
            if delta_ssim is not None
            else None
        )
        size_at_butt = (
            interpolate_value(zen_offs, [zen_size_by_off[o] for o in zen_offs], delta_butt)
            if delta_butt is not None
            else None
        )

        results.append(
            {
                "q": q,
                "n_images": len(cpp),
                "cpp_size": cpp_mean_size,
                "cpp_ssim": cpp_mean_ssim,
                "cpp_butt": cpp_mean_butt,
                "zen0_ssim": zen0_ssim,
                "zen0_butt": zen0_butt,
                "zen0_size": zen_size_by_off.get(0.0),
                "ssim_excess_at_0": ssim_excess,
                "butt_excess_at_0": butt_excess,
                "delta_ssim": delta_ssim,
                "size_at_ssim": size_at_ssim,
                "delta_butt": delta_butt,
                "size_at_butt": size_at_butt,
                "zen_ssim_by_off": zen_ssim_by_off,
                "zen_butt_by_off": zen_butt_by_off,
                "zen_size_by_off": zen_size_by_off,
                "per_image": per_image[q],
            }
        )

    return results


def per_image_deltas(results):
    """Find per-image matched-Δ variance (for global-curve viability)."""
    OFFSETS = None
    variance_by_q = {}
    for r in results:
        q = r["q"]
        zen_offs = sorted({off for (src, off) in [(s, o) for s, o in [(k[0], k[1]) for k in [k for k in [(k,) for k in []]]]]}) if False else sorted(r["zen_size_by_off"].keys())
        per_img = r["per_image"]
        image_deltas_ssim = []
        image_deltas_butt = []
        for name, enc in per_img.items():
            cpp = enc.get(("cjpegli", 0.0))
            if not cpp:
                continue
            cpp_ssim = cpp[1]
            cpp_butt = cpp[2]
            # Gather this image's zen offsets
            this_offsets = sorted(
                [(o, enc[("zen", o)]) for o in zen_offs if ("zen", o) in enc]
            )
            if not this_offsets:
                continue
            offs = [o for o, _ in this_offsets]
            ssim_list = [v[1] for _, v in this_offsets]
            butt_list = [v[2] for _, v in this_offsets]
            d_ssim = interpolate_crossing(offs, ssim_list, cpp_ssim)
            d_butt = interpolate_crossing(offs, butt_list, cpp_butt)
            if d_ssim is not None:
                image_deltas_ssim.append(d_ssim)
            if d_butt is not None:
                image_deltas_butt.append(d_butt)
        if image_deltas_ssim:
            image_deltas_ssim.sort()
            mid = len(image_deltas_ssim) // 2
            variance_by_q[q] = {
                "ssim_median": image_deltas_ssim[mid],
                "ssim_min": image_deltas_ssim[0],
                "ssim_max": image_deltas_ssim[-1],
                "ssim_p25": image_deltas_ssim[len(image_deltas_ssim) // 4],
                "ssim_p75": image_deltas_ssim[3 * len(image_deltas_ssim) // 4],
                "butt_median": sorted(image_deltas_butt)[len(image_deltas_butt) // 2] if image_deltas_butt else None,
                "butt_min": min(image_deltas_butt) if image_deltas_butt else None,
                "butt_max": max(image_deltas_butt) if image_deltas_butt else None,
            }
    return variance_by_q


def print_table(results, label=""):
    if label:
        print(f"\n=== {label} ===")
    print(f"{'Q':>3} {'N':>3} "
          f"{'cpp_ssim':>9} {'zen0_ssim':>9} {'ssim_ex':>8} "
          f"{'cpp_butt':>9} {'zen0_butt':>9} {'butt_ex':>8} "
          f"{'cpp_size':>9} {'zen0_size':>10} {'zen0/cpp':>9}")
    for r in results:
        print(
            f"{r['q']:>3} {r['n_images']:>3} "
            f"{r['cpp_ssim']:>9.3f} {r['zen0_ssim']:>9.3f} {r['ssim_excess_at_0']:>+8.3f} "
            f"{r['cpp_butt']:>9.3f} {r['zen0_butt']:>9.3f} {r['butt_excess_at_0']:>+8.3f} "
            f"{r['cpp_size']:>9.0f} {r['zen0_size']:>10.0f} "
            f"{r['zen0_size'] / r['cpp_size']:>9.3f}"
        )


def print_matched_table(results, metric="ssim"):
    print(f"\n=== Matched-quality by {metric} ===")
    print(f"{'Q':>3} {'Δ':>7} {'zen_size':>10} {'cpp_size':>10} {'ratio':>7} {'savings':>9}")
    for r in results:
        if metric == "ssim":
            d = r["delta_ssim"]
            s = r["size_at_ssim"]
        else:
            d = r["delta_butt"]
            s = r["size_at_butt"]
        if d is None or s is None:
            print(f"{r['q']:>3} {'N/A':>7} {'N/A':>10}")
            continue
        ratio = s / r["cpp_size"]
        savings = (1 - ratio) * 100
        print(
            f"{r['q']:>3} {d:>+7.3f} {s:>10.0f} {r['cpp_size']:>10.0f} "
            f"{ratio:>7.3f} {savings:>+8.2f}%"
        )


def main():
    csv_path = sys.argv[1]
    all_results = analyze(csv_path)
    photo_results = analyze(csv_path, {"photo"})
    graphic_results = analyze(csv_path, {"graphic", "frymire"})

    print("=" * 80)
    print("ALL IMAGES")
    print("=" * 80)
    print_table(all_results, "zen at Δ=0 vs cjpegli — means at equal user-Q")
    print_matched_table(all_results, "ssim")
    print_matched_table(all_results, "butt")

    print("\n" + "=" * 80)
    print("PHOTOS only")
    print("=" * 80)
    print_table(photo_results)
    print_matched_table(photo_results, "ssim")
    print_matched_table(photo_results, "butt")

    print("\n" + "=" * 80)
    print("GRAPHICS + frymire")
    print("=" * 80)
    print_table(graphic_results)
    print_matched_table(graphic_results, "ssim")
    print_matched_table(graphic_results, "butt")

    print("\n" + "=" * 80)
    print("PER-IMAGE matched-Δ VARIANCE (is a global curve viable?)")
    print("=" * 80)
    variance = per_image_deltas(all_results)
    print(f"{'Q':>3} {'ssim_min':>9} {'ssim_p25':>9} {'ssim_med':>9} {'ssim_p75':>9} {'ssim_max':>9} {'spread':>7}")
    for q in sorted(variance.keys()):
        v = variance[q]
        print(
            f"{q:>3} {v['ssim_min']:>+9.3f} {v['ssim_p25']:>+9.3f} {v['ssim_median']:>+9.3f} "
            f"{v['ssim_p75']:>+9.3f} {v['ssim_max']:>+9.3f} {(v['ssim_max']-v['ssim_min']):>7.3f}"
        )


if __name__ == "__main__":
    main()
