#!/usr/bin/env python3
"""
Group-ablation companion to `zq_feature_ablation.py`.

Single-feature LOO is blind to correlated feature clusters: each
individual drop looks marginal because siblings compensate. This
script ablates *groups* of related features to test whether the
group is collectively load-bearing or not.

Groups (mapped to zenanalyze tiers):
  alpha             — alpha_present, alpha_used_fraction, alpha_bimodal_score
  chroma_sharpness  — 8 features: cb/cr × {sharpness, horiz, vert, peak}
  palette           — distinct_color_bins, flat_color_block_ratio
  tier3_dct         — high_freq_energy_ratio, luma_histogram_entropy
  tier1_basic       — variance, edge_density, uniformity, chroma_complexity

Same plumbing as `zq_feature_ablation.py`; only the masking changes.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PARETO = Path("benchmarks/zq_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zq_pareto_features_2026-04-29.tsv")
OUT_LOG = Path("benchmarks/zq_feature_group_ablation_2026-04-29.log")
OUT_JSON = Path("benchmarks/zq_feature_group_ablation_2026-04-29.json")

ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

HISTGB_KW = dict(
    max_iter=100,
    max_depth=4,
    learning_rate=0.1,
    l2_regularization=0.5,
    random_state=SEED,
)

GROUPS = {
    "alpha": [
        "feat_alpha_present",
        "feat_alpha_used_fraction",
        "feat_alpha_bimodal_score",
    ],
    "chroma_sharpness": [
        "feat_cb_sharpness",
        "feat_cr_sharpness",
        "feat_cb_horiz_sharpness",
        "feat_cb_vert_sharpness",
        "feat_cb_peak_sharpness",
        "feat_cr_horiz_sharpness",
        "feat_cr_vert_sharpness",
        "feat_cr_peak_sharpness",
    ],
    "palette": [
        "feat_distinct_color_bins",
        "feat_flat_color_block_ratio",
    ],
    "tier3_dct": [
        "feat_high_freq_energy_ratio",
        "feat_luma_histogram_entropy",
    ],
    "tier1_basic": [
        "feat_variance",
        "feat_edge_density",
        "feat_uniformity",
        "feat_chroma_complexity",
    ],
}

# Bonus: cumulative drops in the order LOO suggested would help.
# Test "drop everything LOO said was zero/negative simultaneously".
LOO_DROPS_NEGATIVE_OR_ZERO = [
    "feat_distinct_color_bins",
    "feat_cr_peak_sharpness",
    "feat_cb_vert_sharpness",
    "feat_flat_color_block_ratio",
    "feat_alpha_bimodal_score",
    "feat_alpha_used_fraction",
    "feat_alpha_present",
    "feat_variance",
]

CONFIG_NAMES: dict = {}


def load_pareto(path):
    rows = defaultdict(list)
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                cid = int(r["config_id"])
                bytes_v = int(r["bytes"])
                zensim_v = float(r["zensim"])
            except (ValueError, KeyError):
                continue
            CONFIG_NAMES.setdefault(cid, r["config_name"])
            key = (r["image_path"], r["size_class"], int(r["width"]), int(r["height"]))
            rows[key].append({"config_id": cid, "bytes": bytes_v, "zensim": zensim_v})
    return rows


def load_features(path):
    feats = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        for r in rdr:
            feats[(r["image_path"], r["size_class"])] = np.array(
                [float(r[c]) for c in cols], dtype=np.float32
            )
    return feats, cols


def build_dataset(pareto, feats, feat_cols):
    n_configs = max(CONFIG_NAMES) + 1
    X_rows, Y_rows, meta = [], [], []
    for (image, size, w, h), samples in pareto.items():
        feat_key = (image, size)
        if feat_key not in feats:
            continue
        f = feats[feat_key]
        log_px = math.log(max(1, w * h))
        size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float32)
        size_oh[SIZE_INDEX[size]] = 1.0

        per_cfg = defaultdict(lambda: defaultdict(lambda: math.inf))
        for s in samples:
            for zq in ZQ_TARGETS:
                if s["zensim"] >= zq and s["bytes"] < per_cfg[zq][s["config_id"]]:
                    per_cfg[zq][s["config_id"]] = s["bytes"]

        for zq in ZQ_TARGETS:
            if not per_cfg[zq]:
                continue
            zq_norm = zq / 100.0
            x = np.concatenate([f, size_oh, np.array([log_px, zq_norm], dtype=np.float32)])
            y = np.full(n_configs, np.nan, dtype=np.float32)
            for cfg, b in per_cfg[zq].items():
                if b > 0 and not math.isinf(b):
                    y[cfg] = math.log(b)
            X_rows.append(x)
            Y_rows.append(y)
            meta.append((image, size, zq))
    return np.stack(X_rows), np.stack(Y_rows), meta


def evaluate_argmin(Y_pred, Y_actual, meta, mask):
    n = Y_pred.shape[0]
    overheads = []
    correct = 0
    for i in range(n):
        actual = Y_actual[i]
        pred = Y_pred[i]
        m = (~np.isnan(actual)) & mask
        if not np.any(m):
            continue
        ab = np.where(m, np.exp(actual), np.inf)
        pb = np.where(m, np.exp(np.clip(pred, -30, 30)), np.inf)
        actual_best = int(np.argmin(ab))
        pred_best = int(np.argmin(pb))
        if pred_best == actual_best:
            correct += 1
        ov = (ab[pred_best] - ab[actual_best]) / ab[actual_best]
        overheads.append(ov)
    if not overheads:
        return None
    arr = np.array(overheads)
    return {
        "n": int(len(arr)),
        "argmin_acc": correct / len(arr),
        "mean_pct": float(100 * arr.mean()),
        "p50_pct": float(100 * np.percentile(arr, 50)),
        "p90_pct": float(100 * np.percentile(arr, 90)),
    }


def train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, mask_cols, n_configs):
    X_tr_m = X_tr[:, mask_cols]
    X_va_m = X_va[:, mask_cols]
    Y_pred = np.zeros_like(Y_va)
    for cfg in range(n_configs):
        mask = ~np.isnan(Y_tr[:, cfg])
        if mask.sum() < 50:
            Y_pred[:, cfg] = np.nanmean(Y_tr[:, cfg]) if mask.any() else 0.0
            continue
        gbm = HistGradientBoostingRegressor(**HISTGB_KW)
        gbm.fit(X_tr_m[mask], Y_tr[mask, cfg])
        Y_pred[:, cfg] = gbm.predict(X_va_m)
    return evaluate_argmin(Y_pred, Y_va, meta_va, np.ones(n_configs, dtype=bool))


def main():
    sys.stderr.write(f"Loading {PARETO}...\n")
    pareto = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES)
    sys.stderr.write(f"Loaded {len(pareto)} cells × {len(feat_cols)} features\n")

    X, Y, meta = build_dataset(pareto, feats, feat_cols)
    n_configs = Y.shape[1]
    n_total = X.shape[1]

    rng = np.random.default_rng(SEED)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * HOLDOUT_FRAC))
    val_set = set(images[:n_val])
    train_idx = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"Train rows: {len(train_idx)}, val rows: {len(val_idx)}\n")

    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]
    meta_va = [meta[i] for i in val_idx]

    feat_idx = {name: i for i, name in enumerate(feat_cols)}

    # Baseline.
    sys.stderr.write("\nBaseline (all features)...\n")
    baseline = train_and_eval(
        X_tr, Y_tr, X_va, Y_va, meta_va, np.ones(n_total, dtype=bool), n_configs
    )
    sys.stderr.write(
        f"  baseline: {baseline['mean_pct']:.2f}% argmin_acc {baseline['argmin_acc']:.1%}\n"
    )

    # Group ablations.
    sys.stderr.write("\nGroup ablations (drop entire group)...\n")
    group_results = {}
    for gname, gfeats in GROUPS.items():
        mask = np.ones(n_total, dtype=bool)
        missing = []
        for fname in gfeats:
            if fname in feat_idx:
                mask[feat_idx[fname]] = False
            else:
                missing.append(fname)
        if missing:
            sys.stderr.write(f"  WARN group '{gname}' missing features: {missing}\n")
        m = train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, mask, n_configs)
        group_results[gname] = m
        delta = m["mean_pct"] - baseline["mean_pct"]
        sys.stderr.write(
            f"  drop {gname:20s} ({len(gfeats)} feats): {m['mean_pct']:.2f}% (Δ {delta:+.2f}pp)\n"
        )

    # Combined LOO-suggested drops.
    sys.stderr.write("\nCombined drop: every LOO-zero-or-negative feature simultaneously...\n")
    mask = np.ones(n_total, dtype=bool)
    dropped = []
    for fname in LOO_DROPS_NEGATIVE_OR_ZERO:
        if fname in feat_idx:
            mask[feat_idx[fname]] = False
            dropped.append(fname)
    n_dropped = len(dropped)
    n_remaining = sum(1 for n in feat_cols if n not in dropped)
    combined = train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, mask, n_configs)
    sys.stderr.write(
        f"  drop {n_dropped} feats, keep {n_remaining}: "
        f"{combined['mean_pct']:.2f}% (Δ {combined['mean_pct'] - baseline['mean_pct']:+.2f}pp) "
        f"argmin_acc {combined['argmin_acc']:.1%}\n"
    )

    # Persist.
    out = {
        "baseline": baseline,
        "groups": {k: {"features": GROUPS[k], "metrics": v} for k, v in group_results.items()},
        "combined_loo_drops": {
            "features_dropped": dropped,
            "metrics": combined,
        },
        "histgb_params": HISTGB_KW,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = []

    def w(s):
        lines.append(s)
        sys.stderr.write(s + "\n")

    w("\n# Group ablation — Pareto controller HistGB")
    w(f"Train rows: {len(X_tr)}, val rows: {len(X_va)}")
    w(f"HistGB: {HISTGB_KW}")
    w("")
    w(f"## Baseline (all 19 features)")
    w(f"mean overhead: {baseline['mean_pct']:.2f}%  argmin_acc: {baseline['argmin_acc']:.1%}")
    w("")
    w("## Group ablations (drop each group; sorted by Δ vs baseline)")
    w(f"{'group':22s} {'n_feats':>7s} {'overhead':>10s} {'Δ vs base':>10s}")
    w("-" * 60)
    sorted_groups = sorted(group_results.items(), key=lambda kv: -kv[1]["mean_pct"])
    for gname, m in sorted_groups:
        delta = m["mean_pct"] - baseline["mean_pct"]
        w(f"{gname:22s} {len(GROUPS[gname]):>7d} {m['mean_pct']:>9.2f}%  {delta:>+8.2f}pp")
    w("")
    w(f"## Combined LOO-suggested drops ({n_dropped} features dropped)")
    w(f"dropped: {', '.join(dropped)}")
    w(
        f"overhead: {combined['mean_pct']:.2f}% "
        f"(Δ {combined['mean_pct'] - baseline['mean_pct']:+.2f}pp)  "
        f"argmin_acc: {combined['argmin_acc']:.1%}"
    )

    OUT_LOG.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
