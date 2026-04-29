#!/usr/bin/env python3
"""
Offline Pareto-front analysis + regression fit for the Zq perceptual-target controller.

Input:
  - benchmarks/zq_pareto_<DATE>.tsv          : (image, size, config, q) → (bytes, zensim)
  - benchmarks/zq_pareto_<DATE>_features.tsv : (image, size) → zenanalyze features

Output (stdout):
  - Per-(image, size) Pareto front statistics
  - Optimal (config, q) labels at every target_zq for every (image, size)
  - Trained regression coefficients: (features, target_zq) → predicted_starting_q
    per config. Plus a config-classifier: (features, target_zq) → best_config_id.
  - Held-out validation error.

Usage:
  python3 scripts/zq_pareto_fit.py \
    --pareto benchmarks/zq_pareto_2026-04-28.tsv \
    --features benchmarks/zq_pareto_2026-04-28_features.tsv \
    [--zq-targets 40,50,60,70,80,85,90,95] \
    [--holdout-frac 0.2] \
    [--seed 0]
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Configuration (must mirror the Rust harness's ConfigSpec table) -------
CONFIGS = {
    0: "ycbcr_444_baseline",
    1: "ycbcr_420_baseline",
    2: "ycbcr_444_progressive",
    3: "ycbcr_420_progressive",
    4: "xyb_444_baseline",
    5: "xyb_420_baseline",
    6: "ycbcr_420_auto_optimize",
    7: "ycbcr_444_auto_optimize",
}

DEFAULT_ZQ_TARGETS = [40, 50, 60, 65, 70, 75, 80, 85, 88, 90, 92, 95]
SIZE_RANK = {"tiny": 0, "small": 1, "medium": 2, "large": 3}


def load_pareto(path):
    """Load the (image, size, config, q) → (bytes, zensim) TSV.

    Returns dict[(image, size_class)] -> list of (config_id, q, bytes, zensim).
    Skips rows where bytes/zensim are blank (encode failure)."""
    rows = defaultdict(list)
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            if not r["bytes"] or not r["zensim"]:
                continue
            try:
                key = (r["image_path"], r["size_class"])
                rows[key].append((
                    int(r["config_id"]),
                    int(r["q"]),
                    int(r["bytes"]),
                    float(r["zensim"]),
                ))
            except (ValueError, KeyError):
                continue
    return rows


def load_features(path):
    """Load per-(image, size_class) feature row.

    Returns dict[(image, size_class)] -> dict[feature_name -> float], plus
    the ordered list of feature column names."""
    feats = {}
    cols = []
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for fld in rdr.fieldnames or []:
            if fld.startswith("feat_"):
                cols.append(fld)
        for r in rdr:
            key = (r["image_path"], r["size_class"])
            row = {}
            for c in cols:
                v = r[c]
                try:
                    row[c] = float(v) if v else 0.0
                except ValueError:
                    row[c] = 0.0
            feats[key] = row
    return feats, cols


def pareto_front(points):
    """Given a list of (config_id, q, bytes, zensim), return the subset
    that's Pareto-optimal in the (-bytes, +zensim) sense.

    A point is dominated iff some other point has fewer-or-equal bytes
    AND >= zensim AND strict-better in at least one. We keep undominated."""
    pts = sorted(points, key=lambda p: (p[2], -p[3]))  # by bytes asc, zensim desc
    front = []
    best_z = -math.inf
    for p in pts:
        if p[3] > best_z:
            front.append(p)
            best_z = p[3]
    return front


def best_at_zq(points, target_zq):
    """Smallest-bytes (config, q) achieving zensim >= target_zq.

    Returns (config_id, q, bytes, zensim) or None if no point meets target."""
    feasible = [p for p in points if p[3] >= target_zq]
    if not feasible:
        return None
    return min(feasible, key=lambda p: (p[2], p[1]))  # by bytes, tiebreak by q


def lstsq(X, y):
    """Plain least-squares: returns w such that X @ w ≈ y.

    Closed-form normal-equation solver, no numpy dep."""
    n_rows = len(X)
    n_cols = len(X[0]) if n_rows else 0
    # XtX = X^T X
    XtX = [[0.0] * n_cols for _ in range(n_cols)]
    Xty = [0.0] * n_cols
    for r in range(n_rows):
        for i in range(n_cols):
            xi = X[r][i]
            Xty[i] += xi * y[r]
            for j in range(i, n_cols):
                XtX[i][j] += xi * X[r][j]
    for i in range(n_cols):
        for j in range(i):
            XtX[i][j] = XtX[j][i]
    # Solve via Cholesky of XtX + small ridge for numerical stability.
    eps = 1e-6
    for i in range(n_cols):
        XtX[i][i] += eps
    L = [[0.0] * n_cols for _ in range(n_cols)]
    for i in range(n_cols):
        for j in range(i + 1):
            s = XtX[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    return [0.0] * n_cols
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    # Forward solve L · v = Xty
    v = [0.0] * n_cols
    for i in range(n_cols):
        v[i] = (Xty[i] - sum(L[i][k] * v[k] for k in range(i))) / L[i][i]
    # Back-solve L^T · w = v
    w = [0.0] * n_cols
    for i in reversed(range(n_cols)):
        w[i] = (v[i] - sum(L[k][i] * w[k] for k in range(i + 1, n_cols))) / L[i][i]
    return w


def build_design_matrix(feature_rows, feature_cols, zq_target):
    """Build a feature matrix with light non-linearity.

    Columns:
      [0]   1.0 (intercept)
      [1]   zq_target
      [2]   zq_target ** 2
      [3..] selected features (raw + log1p of saturating features)
    """
    SELECTED = [
        "feat_variance",
        "feat_edge_density",
        "feat_chroma_complexity",
        "feat_uniformity",
        "feat_flat_color_block_ratio",
        "feat_high_freq_energy_ratio",
        "feat_luma_histogram_entropy",
        "feat_cb_peak_sharpness",
        "feat_cr_peak_sharpness",
        "feat_distinct_color_bins",
    ]
    SELECTED = [c for c in SELECTED if c in feature_cols]
    X = []
    for row in feature_rows:
        r = [
            1.0,
            zq_target / 100.0,
            (zq_target / 100.0) ** 2,
        ]
        for c in SELECTED:
            v = row.get(c, 0.0)
            r.append(v)
            r.append(math.log1p(max(0.0, v)))
        X.append(r)
    return X, SELECTED


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pareto", required=True)
    p.add_argument("--features", required=True)
    p.add_argument(
        "--zq-targets",
        default=",".join(str(z) for z in DEFAULT_ZQ_TARGETS),
        help="Comma-separated zq_targets to fit at",
    )
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--per-config-fit",
        action="store_true",
        help="Fit one regression per config (default: also fit a single shared model)",
    )
    args = p.parse_args()

    targets = [int(x) for x in args.zq_targets.split(",")]

    print(f"# Loading {args.pareto}...", file=sys.stderr)
    pareto = load_pareto(args.pareto)
    print(f"# Loaded {len(pareto)} (image, size) cells", file=sys.stderr)
    feats, feat_cols = load_features(args.features)
    print(f"# Loaded {len(feats)} feature rows × {len(feat_cols)} columns", file=sys.stderr)

    # ----- Build label set: per (image, size, target_zq) → (config*, q*).
    # Drop cells where features are missing OR no Pareto point meets any target.
    keys = sorted(set(pareto.keys()) & set(feats.keys()))
    print(f"# Joined keys: {len(keys)}", file=sys.stderr)

    labels = defaultdict(dict)  # (image, size) -> {zq_target -> (cfg, q, bytes, zensim)}
    pareto_sizes = []
    config_choices = defaultdict(int)
    for key in keys:
        front = pareto_front(pareto[key])
        pareto_sizes.append(len(front))
        for tz in targets:
            best = best_at_zq(pareto[key], tz)
            if best is not None:
                labels[key][tz] = best
                config_choices[best[0]] += 1

    print(
        f"\n## Pareto-front summary",
        f"\n- (image, size) cells: {len(keys)}",
        f"\n- median |Pareto front|: {median(pareto_sizes):.1f}  (a perfect single-config dominator would give 1)",
    )
    print("\n### Optimal-config frequency across all (image, size, zq_target) labels:")
    total_labels = sum(config_choices.values())
    for cfg_id in sorted(config_choices.keys(), key=lambda c: -config_choices[c]):
        n = config_choices[cfg_id]
        pct = 100 * n / total_labels if total_labels else 0
        print(f"- config {cfg_id} ({CONFIGS[cfg_id]}): {n} ({pct:.1f}%)")

    # Held-out split: split *images* (not (image, size, zq)) so a single
    # source can't leak between train and val.
    import random
    rng = random.Random(args.seed)
    images = sorted({k[0] for k in keys})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * args.holdout_frac))
    val_images = set(images[:n_val])
    train_images = set(images[n_val:])
    print(f"\n## Train/val split", file=sys.stderr)
    print(f"- val images: {len(val_images)}", file=sys.stderr)
    print(f"- train images: {len(train_images)}", file=sys.stderr)

    train_keys = [k for k in keys if k[0] in train_images]
    val_keys = [k for k in keys if k[0] in val_images]

    # ----- Per-config regression: predict optimal_q within each config for
    # the subset of (image, size, zq) where THAT config was on the Pareto
    # front (or close enough). Each config has its own fit so feature
    # interactions are kept inside one config.
    print(f"\n## Per-config regression — predict starting_q given (features, zq)\n")
    for cfg_id in sorted(CONFIGS.keys()):
        # Collect samples: for every (key, tz), if cfg_id appears in
        # pareto[key] AT zq_target ≥ tz, the smallest q within cfg_id
        # achieving zensim ≥ tz is our regression label.
        train_X_rows = []
        train_y = []
        for key in train_keys:
            pts_in_cfg = [(c, q, b, z) for (c, q, b, z) in pareto[key] if c == cfg_id]
            if not pts_in_cfg:
                continue
            for tz in targets:
                feasible = [p for p in pts_in_cfg if p[3] >= tz]
                if not feasible:
                    continue
                # Smallest-q-meeting-target within this config.
                best_q = min(feasible, key=lambda p: (p[1], p[2]))[1]
                fr = feats.get(key)
                if fr is None:
                    continue
                # Build single-row design.
                X1, _ = build_design_matrix([fr], feat_cols, tz)
                train_X_rows.append(X1[0])
                train_y.append(float(best_q))
        if not train_X_rows:
            print(f"### config {cfg_id} ({CONFIGS[cfg_id]}): NO TRAINING DATA")
            continue
        w = lstsq(train_X_rows, train_y)
        # Validation error.
        val_errs = []
        for key in val_keys:
            pts_in_cfg = [(c, q, b, z) for (c, q, b, z) in pareto[key] if c == cfg_id]
            if not pts_in_cfg:
                continue
            for tz in targets:
                feasible = [p for p in pts_in_cfg if p[3] >= tz]
                if not feasible:
                    continue
                best_q = min(feasible, key=lambda p: (p[1], p[2]))[1]
                fr = feats.get(key)
                if fr is None:
                    continue
                X1, _ = build_design_matrix([fr], feat_cols, tz)
                pred = sum(xi * wi for xi, wi in zip(X1[0], w))
                val_errs.append(pred - best_q)
        rmse = math.sqrt(sum(e * e for e in val_errs) / max(1, len(val_errs))) if val_errs else float("nan")
        bias = sum(val_errs) / max(1, len(val_errs)) if val_errs else float("nan")
        print(f"### config {cfg_id} ({CONFIGS[cfg_id]})")
        print(f"- train n={len(train_X_rows)}, val n={len(val_errs)}")
        print(f"- val RMSE: {rmse:.2f}  bias: {bias:+.2f}")
        print(f"- weights = {[round(x, 4) for x in w]}")

    # ----- Pareto-Pareto choice metric: at each (val image, val zq), measure
    # the bytes-cost of the per-config-best-q vs the actual Pareto best.
    print(f"\n## Pareto-overhead (bytes vs ideal, config-naive)\n")
    overhead_by_zq = defaultdict(list)
    for key in val_keys:
        for tz in targets:
            best = labels[key].get(tz)
            if not best:
                continue
            ideal_bytes = best[2]
            # Find smallest-bytes-meeting-tz across all configs (the actual
            # Pareto-optimal).
            actual_bytes = ideal_bytes  # by construction
            # Compare to the bytes of (any-config, smallest q meeting tz)
            # we'd ship if we used the bucket-naive zq_to_q heuristic.
            # The current scaffold uses ApproxJpegli(some_q), which lands
            # at config 1 (ycbcr_420_baseline); use that as the reference.
            scaffold_pts = [(c, q, b, z) for (c, q, b, z) in pareto[key] if c == 1 and z >= tz]
            if not scaffold_pts:
                continue
            scaffold_bytes = min(scaffold_pts, key=lambda p: p[2])[2]
            overhead_by_zq[tz].append((scaffold_bytes - actual_bytes) / actual_bytes)
    print("zq_tgt | n | mean overhead | p50 | p75 |")
    print("-------|---|---------------|-----|-----|")
    for tz in targets:
        vs = overhead_by_zq.get(tz, [])
        if not vs:
            continue
        vs_s = sorted(vs)
        n = len(vs_s)
        mean = sum(vs_s) / n
        p50 = vs_s[n // 2]
        p75 = vs_s[(n * 3) // 4]
        print(f"{tz:3d}    | {n:3d} | {100 * mean:+.1f}%       | {100 * p50:+.1f}% | {100 * p75:+.1f}% |")
    print(
        "\nInterpretation: overhead is bytes-vs-Pareto for the bucket-1-naive "
        "(ycbcr_420_baseline) controller. Positive = the naive controller ships "
        "more bytes than necessary; negative = the chosen target wasn't reached "
        "and the scaffold falls back."
    )


if __name__ == "__main__":
    main()
