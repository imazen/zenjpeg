#!/usr/bin/env python3
"""
Validate the 8-feature reduced schema on production HistGB.

The ablation work used a lighter HistGB (max_iter=100, max_depth=4)
for throughput across 19 LOO + 5 group + 4 sub-ablation points. The
ranking should be stable but absolute numbers shift. This script
re-runs at production hyperparameters (max_iter=400, max_depth=8 —
matching zq_bytes_distill.py's teacher) on:

  baseline (all 19 features)  — should reproduce ~6.8% mean overhead
  reduced  (8 features)       — must be ≤ baseline to ship

Reduced schema (8 features kept):
  Tier 1 basic:  variance, edge_density, uniformity, chroma_complexity
  Tier 1 chroma: cb_sharpness, cr_sharpness
  Tier 3 DCT:    high_freq_energy_ratio, luma_histogram_entropy

Dropped (11 features, 3 entire zenanalyze tiers):
  Alpha tier (3): alpha_present, alpha_used_fraction, alpha_bimodal_score
  Palette tier (2): distinct_color_bins, flat_color_block_ratio
  Tier 2 (6): cb/cr × {horiz, vert, peak}_sharpness

Output: benchmarks/zq_reduced_schema_validate_2026-04-29.{log,json}
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
OUT_LOG = Path("benchmarks/zq_reduced_schema_validate_2026-04-29.log")
OUT_JSON = Path("benchmarks/zq_reduced_schema_validate_2026-04-29.json")

ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

# Production HistGB hyperparameters — same as zq_bytes_distill.py teacher.
HISTGB_KW = dict(
    max_iter=400,
    max_depth=8,
    learning_rate=0.05,
    l2_regularization=0.5,
    random_state=SEED,
)

KEEP_FEATURES = [
    "feat_variance",
    "feat_edge_density",
    "feat_uniformity",
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    "feat_cr_sharpness",
    "feat_high_freq_energy_ratio",
    "feat_luma_histogram_entropy",
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


def evaluate(Y_pred, Y_actual, meta):
    overheads, correct = [], 0
    per_zq = defaultdict(list)
    for i in range(Y_pred.shape[0]):
        actual = Y_actual[i]
        pred = Y_pred[i]
        m = ~np.isnan(actual)
        if not np.any(m):
            continue
        ab = np.where(m, np.exp(actual), np.inf)
        pb = np.where(m, np.exp(np.clip(pred, -30, 30)), np.inf)
        a = int(np.argmin(ab))
        p = int(np.argmin(pb))
        if p == a:
            correct += 1
        ov = (ab[p] - ab[a]) / ab[a]
        overheads.append(ov)
        per_zq[meta[i][2]].append(ov)
    arr = np.array(overheads)
    return {
        "n": int(len(arr)),
        "mean_pct": float(100 * arr.mean()),
        "p50_pct": float(100 * np.percentile(arr, 50)),
        "p75_pct": float(100 * np.percentile(arr, 75)),
        "p90_pct": float(100 * np.percentile(arr, 90)),
        "argmin_acc": correct / len(arr),
        "per_zq": {
            tz: {
                "n": len(v),
                "mean": float(100 * np.mean(v)),
                "p50": float(100 * np.percentile(v, 50)),
                "p90": float(100 * np.percentile(v, 90)),
            }
            for tz, v in per_zq.items()
        },
    }


def train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, mask, n_configs, label):
    Xtr = X_tr[:, mask]
    Xva = X_va[:, mask]
    sys.stderr.write(f"  [{label}] training {n_configs} per-config HistGBs (n_inputs={Xtr.shape[1]})...\n")
    Y_pred = np.zeros_like(Y_va)
    for cfg in range(n_configs):
        m = ~np.isnan(Y_tr[:, cfg])
        if m.sum() < 50:
            Y_pred[:, cfg] = np.nanmean(Y_tr[:, cfg]) if m.any() else 0.0
            continue
        gbm = HistGradientBoostingRegressor(**HISTGB_KW)
        gbm.fit(Xtr[m], Y_tr[m, cfg])
        Y_pred[:, cfg] = gbm.predict(Xva)
        if (cfg + 1) % 30 == 0:
            sys.stderr.write(f"    {cfg + 1}/{n_configs} configs trained\n")
    return evaluate(Y_pred, Y_va, meta_va)


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
    sys.stderr.write(f"HistGB: {HISTGB_KW}\n")

    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]
    meta_va = [meta[i] for i in val_idx]
    feat_idx = {n: i for i, n in enumerate(feat_cols)}

    # Baseline (all 19 features).
    sys.stderr.write("\n[1/2] Baseline (all 19 features, production HistGB)...\n")
    baseline = train_and_eval(
        X_tr, Y_tr, X_va, Y_va, meta_va, np.ones(n_total, dtype=bool), n_configs, "baseline"
    )
    sys.stderr.write(
        f"  baseline: mean {baseline['mean_pct']:.2f}% p50 {baseline['p50_pct']:.2f}% "
        f"p90 {baseline['p90_pct']:.2f}% argmin_acc {baseline['argmin_acc']:.1%}\n"
    )

    # Reduced (8 features).
    keep_set = set(KEEP_FEATURES)
    mask = np.zeros(n_total, dtype=bool)
    kept = []
    missing = []
    for name in KEEP_FEATURES:
        if name in feat_idx:
            mask[feat_idx[name]] = True
            kept.append(name)
        else:
            missing.append(name)
    # Also keep size onehot + zq (always-on auxiliary inputs).
    mask[len(feat_cols):] = True

    if missing:
        sys.stderr.write(f"  WARN missing: {missing}\n")

    sys.stderr.write(f"\n[2/2] Reduced ({len(kept)} features, production HistGB)...\n")
    sys.stderr.write(f"  features kept: {kept}\n")
    reduced = train_and_eval(
        X_tr, Y_tr, X_va, Y_va, meta_va, mask, n_configs, "reduced"
    )
    sys.stderr.write(
        f"  reduced: mean {reduced['mean_pct']:.2f}% p50 {reduced['p50_pct']:.2f}% "
        f"p90 {reduced['p90_pct']:.2f}% argmin_acc {reduced['argmin_acc']:.1%}\n"
    )

    delta = reduced["mean_pct"] - baseline["mean_pct"]
    sys.stderr.write(f"\nΔ (reduced - baseline): {delta:+.2f}pp\n")

    out = {
        "baseline": baseline,
        "reduced": reduced,
        "kept_features": kept,
        "histgb_params": HISTGB_KW,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = []

    def w(s):
        lines.append(s)
        sys.stderr.write(s + "\n")

    w("\n# Reduced schema validation — production HistGB")
    w(f"Train rows: {len(X_tr)}, val rows: {len(X_va)}")
    w(f"HistGB: {HISTGB_KW}")
    w("")
    w("## Comparison")
    w(f"{'config':30s} {'mean':>10s} {'p50':>8s} {'p75':>8s} {'p90':>8s} {'argmin':>10s}")
    w("-" * 75)
    w(
        f"{'baseline (19 features)':30s} "
        f"{baseline['mean_pct']:>9.2f}% {baseline['p50_pct']:>7.2f}% "
        f"{baseline['p75_pct']:>7.2f}% {baseline['p90_pct']:>7.2f}% "
        f"{baseline['argmin_acc']:>9.1%}"
    )
    w(
        f"{'reduced (8 features)':30s} "
        f"{reduced['mean_pct']:>9.2f}% {reduced['p50_pct']:>7.2f}% "
        f"{reduced['p75_pct']:>7.2f}% {reduced['p90_pct']:>7.2f}% "
        f"{reduced['argmin_acc']:>9.1%}"
    )
    w(f"\nΔ mean: {delta:+.2f}pp")
    w("")
    w("## Per-zq comparison (mean overhead)")
    w(f"{'zq':>4s} {'baseline':>10s} {'reduced':>10s} {'Δ':>8s}")
    w("-" * 40)
    for tz in sorted(set(baseline["per_zq"]) & set(reduced["per_zq"])):
        b = baseline["per_zq"][tz]
        r = reduced["per_zq"][tz]
        d = r["mean"] - b["mean"]
        w(f"{tz:>4d} {b['mean']:>9.2f}% {r['mean']:>9.2f}% {d:>+7.2f}pp")

    OUT_LOG.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
