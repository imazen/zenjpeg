#!/usr/bin/env python3
"""Analyze saved source_aware_rd points without rerunning encodes or IQA.

BD-rate uses exact piecewise-linear integration of ln(bpp) over the common
distortion interval. This intentionally avoids cubic overshoot on quantization
plateaus. It is not the cubic implementation in zenjpeg-bench-utils::rd.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def frontier(points):
    """Nondominated (distortion, rate) points, increasing distortion."""
    best = math.inf
    hull = []
    for d, rate in sorted(points, key=lambda p: (p[1], p[0])):
        if not math.isfinite(d) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("invalid RD point")
        if d < best:
            hull.append((d, rate))
            best = d
    return list(reversed(hull))


def integral(curve, lo, hi):
    total = 0.0
    for (x0, r0), (x1, r1) in zip(curve, curve[1:]):
        a, b = max(lo, x0), min(hi, x1)
        if a >= b:
            continue
        y0, y1 = math.log(r0), math.log(r1)
        ya = y0 + (y1 - y0) * (a - x0) / (x1 - x0)
        yb = y0 + (y1 - y0) * (b - x0) / (x1 - x0)
        total += (ya + yb) * (b - a) / 2
    return total


def compare(baseline, candidate):
    b, c = frontier(baseline), frontier(candidate)
    result = {"base_points": len(b), "candidate_points": len(c),
              "bd_rate_percent": None, "overlap_low": None, "overlap_high": None,
              "baseline_coverage": 0.0, "candidate_coverage": 0.0,
              "sparse": len(b) < 4 or len(c) < 4}
    if len(b) < 2 or len(c) < 2:
        return result
    lo, hi = max(b[0][0], c[0][0]), min(b[-1][0], c[-1][0])
    if hi <= lo:
        return result
    result.update(
        bd_rate_percent=100 * math.expm1((integral(c, lo, hi) - integral(b, lo, hi)) / (hi - lo)),
        overlap_low=lo, overlap_high=hi,
        baseline_coverage=(hi - lo) / (b[-1][0] - b[0][0]),
        candidate_coverage=(hi - lo) / (c[-1][0] - c[0][0]),
    )
    return result


GROUP_FIELDS = ("image", "source_mode", "source_quality", "destination", "sampling")


def analyze(points):
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for p in points:
        group = tuple(p[k] for k in GROUP_FIELDS)
        identity = (*group, p["variant"], p["quality"])
        if identity in seen:
            raise ValueError(f"duplicate sweep cell: {identity}")
        seen.add(identity)
        groups[group][p["variant"]].append(p)
    rows = []
    for group, variants in sorted(groups.items()):
        for reference in ("generation", "cumulative"):
            for metric in ("ssim2", "butteraugli"):
                curves = {v: [(100 - p[reference][metric] if metric == "ssim2" else p[reference][metric], p["bpp"])
                              for p in ps] for v, ps in variants.items()}
                for base in ("generic", "exact"):
                    if base not in curves:
                        continue
                    for variant, curve in sorted(curves.items()):
                        if variant == base or (base == "exact" and variant == "generic"):
                            continue
                        rows.append(dict(zip(GROUP_FIELDS, group), reference=reference, metric=metric,
                                         baseline=base, candidate=variant, **compare(curves[base], curve)))
    return groups, rows


def write_summary(out, rows, complete):
    agg = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("source_mode", "destination", "reference", "metric", "baseline", "candidate"))
        agg[key].append(row)
    lines = ["# Source-aware quantization: experimental RD results", "",
             f"Run status: {'complete' if complete else 'INCOMPLETE — diagnostic only'}.", "",
             "Negative BD-rate means fewer bytes at equal measured quality. Integration is piecewise linear in log rate; no extrapolation. "
             "SSIM2 uses 100 − score, Butteraugli uses its raw distance. The metrics are reported separately.", "",
             "Aggregate rates are geometric means of per-image/source-quality/sampling rate ratios, not pooled curves. "
             "Cells with fewer than four Pareto points, or without overlap, are excluded from the aggregate. "
             "See bd_rate.csv for each interval and coverage; small overlaps do not support broad conclusions.", "",
             "Sources are zenjpeg's mode emulations. These results do not establish parity with external C encoders or held-out gains.", "",
             "| Source | Destination | Reference | Metric | Baseline | Candidate | Valid/total | BD-rate | Worst cell | Regressions |",
             "|---|---|---|---|---|---|---:|---:|---:|---:|"]
    for key, cells in sorted(agg.items()):
        valid = [c["bd_rate_percent"] for c in cells if c["bd_rate_percent"] is not None and not c["sparse"]]
        if valid:
            mean = 100 * math.expm1(sum(math.log1p(v / 100) for v in valid) / len(valid))
            values = [f"{mean:+.2f}%", f"{max(valid):+.2f}%", f"{sum(v > 0 for v in valid)}/{len(valid)}"]
        else:
            values = ["N/A", "N/A", "N/A"]
        lines.append("| " + " | ".join(map(str, (*key, f"{len(valid)}/{len(cells)}", *values))) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def goal_choices(out, groups):
    """Measured per-image oracle at each generic point's byte budget.

    Generation-reference selection could be used at runtime with an IQA
    budget. Cumulative-reference selection is offline-only (needs originals).
    Neither is evidence for an unmeasured one-shot selector.
    """
    rows = []
    for group, variants in sorted(groups.items()):
        pool = [p for ps in variants.values() for p in ps]
        for baseline in variants.get("generic", []):
            eligible = [p for p in pool if p["bytes"] <= baseline["bytes"]]
            for reference in ("generation", "cumulative"):
                for metric in ("ssim2", "butteraugli"):
                    oriented = lambda p: -p[reference][metric] if metric == "ssim2" else p[reference][metric]
                    best = min(eligible, key=lambda p: (oriented(p), p["bytes"], p["variant"]))
                    rows.append(dict(zip(GROUP_FIELDS, group), reference=reference, metric=metric,
                                     budget_quality=baseline["quality"], budget_bytes=baseline["bytes"],
                                     winner=best["variant"], winner_quality=best["quality"], winner_bytes=best["bytes"],
                                     ssim2=best[reference]["ssim2"], butteraugli=best[reference]["butteraugli"],
                                     improvement=oriented(baseline) - oriented(best), jpeg=best["jpeg"]))
    if rows:
        with (out / "goal_choices.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def joint_choices(out, groups):
    """Smallest measured output meeting both baseline metric floors."""
    rows = []
    for group, variants in sorted(groups.items()):
        pool = [p for ps in variants.values() for p in ps]
        for baseline in variants.get("generic", []):
            for reference in ("generation", "cumulative"):
                eligible = [p for p in pool if p["bytes"] <= baseline["bytes"]
                            and p[reference]["ssim2"] >= baseline[reference]["ssim2"]
                            and p[reference]["butteraugli"] <= baseline[reference]["butteraugli"]]
                best = min(eligible, key=lambda p: (p["bytes"], -p[reference]["ssim2"], p[reference]["butteraugli"], p["variant"]))
                rows.append(dict(zip(GROUP_FIELDS, group), reference=reference,
                                 budget_quality=baseline["quality"], budget_bytes=baseline["bytes"],
                                 winner=best["variant"], winner_quality=best["quality"], winner_bytes=best["bytes"],
                                 savings_percent=100 * (1 - best["bytes"] / baseline["bytes"]),
                                 ssim2_delta=best[reference]["ssim2"] - baseline[reference]["ssim2"],
                                 butteraugli_delta=best[reference]["butteraugli"] - baseline[reference]["butteraugli"],
                                 jpeg=best["jpeg"]))
    if rows:
        with (out / "joint_choices.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def plots(out, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for index, (group, variants) in enumerate(sorted(groups.items())):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for row, reference in enumerate(("generation", "cumulative")):
            for col, metric in enumerate(("ssim2", "butteraugli")):
                ax = axes[row, col]
                for variant, ps in sorted(variants.items()):
                    ps = sorted(ps, key=lambda p: p["bpp"])
                    ax.plot([p["bpp"] for p in ps], [p[reference][metric] for p in ps], ".-", label=variant)
                ax.set(xlabel="Output bits/pixel", ylabel="SSIMULACRA2 (higher better)" if metric == "ssim2" else "Butteraugli (lower better)", title=reference)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7)
        fig.suptitle(" / ".join(map(str, group)))
        fig.savefig(out / f"curves_{index:03}.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plots", action="store_true", help="write SVG plots (requires matplotlib)")
    args = parser.parse_args()
    points = [json.loads(line) for line in (args.run / "points.jsonl").read_text().splitlines() if line.strip()]
    if not points:
        raise ValueError("no saved points")
    groups, rows = analyze(points)
    if not rows:
        raise ValueError("no baseline/candidate curve pairs")
    with (args.run / "bd_rate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(args.run, rows, (args.run / "COMPLETE").exists())
    goal_choices(args.run, groups)
    joint_choices(args.run, groups)
    if args.plots:
        plots(args.run, groups)
    print(f"{len(points)} points; {len(rows)} comparisons; {args.run / 'summary.md'}")


if __name__ == "__main__":
    main()
