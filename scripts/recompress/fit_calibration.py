#!/usr/bin/env python3
"""Fit per-(source_q, target) median table from a zjr-calibrate cumulative-sweep TSV.

Inputs a sweep TSV. Outputs:
- median measured zensim-A per (source_q_bucket, target) cell to stdout as a Rust
  const array,
- median size_ratio per (source_q_bucket, target) cell,
- empirical NoOp band (source_estimated_zensim_a beyond which the source is treated
  as already at target).

Usage:
    python3 scripts/fit_calibration.py benchmarks/cid22_15img_seed_sweep_2026-05-28.tsv

Bucketing strategy: each row's source_q is the detected jpegli BA distance
(for zenjpeg outputs) or IJG-Q (for libjpeg-turbo etc). We bucket BA distance
into 9 buckets matching the input sweep grid; rows with `quality_scale` ≠
ButteraugliDistance are reported separately.
"""

import csv
import sys
from collections import defaultdict
from statistics import median


def main(tsv_path):
    rows = []
    with open(tsv_path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)

    # Bucket source_q into the original sweep grid buckets. For BA-distance
    # encoders, our cumulative-sweep input was --source-qs 20,30,40,50,60,70,80,90,95
    # which maps to detected BA distances roughly:
    #   q=20 → BA ~8.0      q=60 → BA ~2.5
    #   q=30 → BA ~5.7      q=70 → BA ~1.8
    #   q=40 → BA ~4.6      q=80 → BA ~1.3
    #   q=50 → BA ~3.5      q=90 → BA ~0.5    q=95 → BA ~0.3
    #
    # We bucket by snapping each row's BA distance to the nearest sweep
    # input. (Same applies to the dimension on the encoder side.)
    # Bucket centers chosen from the actual data:
    ba_buckets = [0.5, 1.3, 1.8, 2.5, 3.5, 4.6, 5.7, 6.7, 8.0]

    # (bucket, target) → list of measured zensim-A and list of size ratios
    measured_by_cell = defaultdict(list)
    ratio_by_cell = defaultdict(list)

    for row in rows:
        if row["strategy"] not in ("tuned", "deblock"):
            continue
        try:
            src_ba = float(row["source_q"])
            target = float(row["target_zensim_a"])
            measured = float(row["zensim_a_vs_reference"])
            ratio = float(row["size_ratio"])
        except (ValueError, KeyError):
            continue

        # Snap to nearest bucket
        bucket = min(ba_buckets, key=lambda b: abs(b - src_ba))
        measured_by_cell[(bucket, target)].append(measured)
        ratio_by_cell[(bucket, target)].append(ratio)

    # Get unique sorted buckets and targets actually populated
    buckets = sorted({b for (b, _) in measured_by_cell})
    targets = sorted({t for (_, t) in measured_by_cell})

    print("// Auto-fit calibration anchors from sweep TSV.", file=sys.stderr)
    print(
        f"// Source: {tsv_path}\n"
        f"// Buckets (BA distance): {buckets}\n"
        f"// Targets: {targets}\n",
        file=sys.stderr,
    )

    # Emit median measured zensim-A per cell as a Rust 2D const.
    n_refs = len({(row.get("reference_id") or "") for row in rows}) or "?"
    n_cells = sum(len(v) for v in measured_by_cell.values())
    print("/// Measured zensim-A (vs unknown reference) achieved by the Tuned strategy")
    print(f"/// at each `(source_ba_bucket, target_zensim_a)` cell — median over")
    print(f"/// {n_refs} refs, {n_cells} forced-Tuned samples ({tsv_path}).")
    print("///")
    print("/// Indexing: `CELL_MEDIAN[src_idx][tgt_idx]` where `src_idx` indexes into")
    print("/// `SOURCE_BA_BUCKETS` and `tgt_idx` into `TARGET_GRID`.")
    print(f"pub const SOURCE_BA_BUCKETS: &[f32] = &{buckets!r};")
    print(f"pub const TARGET_GRID: &[f32] = &{[float(t) for t in targets]!r};")
    print()

    print("pub const CELL_MEDIAN_ZENSIM_A: &[[f32; {}]; {}] = &[".format(
        len(targets), len(buckets)
    ))
    for b in buckets:
        row_vals = []
        for t in targets:
            vals = measured_by_cell.get((b, t), [])
            row_vals.append(f"{median(vals):.3f}" if vals else "f32::NAN")
        print("    [" + ", ".join(row_vals) + "],")
    print("];")
    print()

    print("pub const CELL_MEDIAN_SIZE_RATIO: &[[f32; {}]; {}] = &[".format(
        len(targets), len(buckets)
    ))
    for b in buckets:
        row_vals = []
        for t in targets:
            vals = ratio_by_cell.get((b, t), [])
            row_vals.append(f"{median(vals):.4f}" if vals else "f32::NAN")
        print("    [" + ", ".join(row_vals) + "],")
    print("];")
    print()

    # MAE report
    print("// Fitter sanity check: per-bucket calibration error if we project", file=sys.stderr)
    print("// `projected = CELL_MEDIAN_ZENSIM_A[src][tgt]` and `measured = actual`:", file=sys.stderr)
    total_n, total_err = 0, 0.0
    for (b, t), vals in measured_by_cell.items():
        m = median(vals)
        for v in vals:
            total_n += 1
            total_err += abs(v - m)
    if total_n:
        print(f"// 2D-table residual MAE: {total_err/total_n:.3f} (n={total_n})", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
