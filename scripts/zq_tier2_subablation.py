#!/usr/bin/env python3
"""
Targeted: can we drop Tier 2 entirely?

Tier 2 is the most expensive zenanalyze tier (full-image 3-row
sliding window for per-axis Cb/Cr sharpness — 6 features). The
group-ablation showed dropping ALL 8 chroma sharpness features
costs +0.36pp. But cb_sharpness/cr_sharpness are Tier 1 (free
piggyback). The question: drop just the 6 Tier 2 features
(horiz/vert/peak), keep the 2 Tier 1 features.

Tests two configurations on top of the alpha+palette drop:
  drop_t2_chroma : alpha + palette + 6 Tier 2 chroma sharpness
  drop_t2_chroma_minus_horiz : alpha + palette + just 4 vert/peak
                              (keep horiz, which LOO ranked as
                              relatively high-importance: +0.08-0.09)
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PARETO = Path("benchmarks/zq_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zq_pareto_features_2026-04-29.tsv")

ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

HISTGB_KW = dict(max_iter=100, max_depth=4, learning_rate=0.1, l2_regularization=0.5, random_state=SEED)

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
        overheads.append((ab[p] - ab[a]) / ab[a])
    arr = np.array(overheads)
    return {
        "mean_pct": float(100 * arr.mean()),
        "p50_pct": float(100 * np.percentile(arr, 50)),
        "p90_pct": float(100 * np.percentile(arr, 90)),
        "argmin_acc": correct / len(arr),
    }


def train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, mask, n_configs):
    Xtr = X_tr[:, mask]
    Xva = X_va[:, mask]
    Y_pred = np.zeros_like(Y_va)
    for cfg in range(n_configs):
        m = ~np.isnan(Y_tr[:, cfg])
        if m.sum() < 50:
            Y_pred[:, cfg] = np.nanmean(Y_tr[:, cfg]) if m.any() else 0.0
            continue
        gbm = HistGradientBoostingRegressor(**HISTGB_KW)
        gbm.fit(Xtr[m], Y_tr[m, cfg])
        Y_pred[:, cfg] = gbm.predict(Xva)
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
    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]
    meta_va = [meta[i] for i in val_idx]
    feat_idx = {n: i for i, n in enumerate(feat_cols)}

    SAFE_DROPS = [
        "feat_alpha_present",
        "feat_alpha_used_fraction",
        "feat_alpha_bimodal_score",
        "feat_distinct_color_bins",
        "feat_flat_color_block_ratio",
    ]
    TIER2_FULL = [
        "feat_cb_horiz_sharpness",
        "feat_cb_vert_sharpness",
        "feat_cb_peak_sharpness",
        "feat_cr_horiz_sharpness",
        "feat_cr_vert_sharpness",
        "feat_cr_peak_sharpness",
    ]
    TIER2_KEEP_HORIZ = [
        "feat_cb_vert_sharpness",
        "feat_cb_peak_sharpness",
        "feat_cr_vert_sharpness",
        "feat_cr_peak_sharpness",
    ]

    def make_mask(drops):
        m = np.ones(n_total, dtype=bool)
        for f in drops:
            if f in feat_idx:
                m[feat_idx[f]] = False
        return m

    sys.stderr.write("\nBaseline (all 19 features)...\n")
    base = train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, np.ones(n_total, dtype=bool), n_configs)
    sys.stderr.write(f"  baseline: {base['mean_pct']:.2f}% argmin {base['argmin_acc']:.1%}\n")

    sys.stderr.write("\nDrop alpha + palette only (5 features)...\n")
    a = train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, make_mask(SAFE_DROPS), n_configs)
    sys.stderr.write(
        f"  -> {a['mean_pct']:.2f}% (Δ {a['mean_pct'] - base['mean_pct']:+.2f}pp)  "
        f"argmin {a['argmin_acc']:.1%}\n"
    )

    sys.stderr.write("\nDrop alpha + palette + ALL 6 Tier 2 chroma (11 features)...\n")
    b = train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, make_mask(SAFE_DROPS + TIER2_FULL), n_configs)
    sys.stderr.write(
        f"  -> {b['mean_pct']:.2f}% (Δ {b['mean_pct'] - base['mean_pct']:+.2f}pp)  "
        f"argmin {b['argmin_acc']:.1%}\n"
    )

    sys.stderr.write("\nDrop alpha + palette + 4 Tier 2 (keep horiz, 9 dropped)...\n")
    c = train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, make_mask(SAFE_DROPS + TIER2_KEEP_HORIZ), n_configs)
    sys.stderr.write(
        f"  -> {c['mean_pct']:.2f}% (Δ {c['mean_pct'] - base['mean_pct']:+.2f}pp)  "
        f"argmin {c['argmin_acc']:.1%}\n"
    )

    sys.stderr.write("\nSummary:\n")
    sys.stderr.write(f"  baseline (all 19):              {base['mean_pct']:.2f}%\n")
    sys.stderr.write(f"  drop alpha+palette (5 dropped): {a['mean_pct']:.2f}% (Δ {a['mean_pct'] - base['mean_pct']:+.2f}pp)\n")
    sys.stderr.write(f"  + all 6 Tier 2 (11 dropped):    {b['mean_pct']:.2f}% (Δ {b['mean_pct'] - base['mean_pct']:+.2f}pp)\n")
    sys.stderr.write(f"  + 4 Tier 2 keep horiz (9):      {c['mean_pct']:.2f}% (Δ {c['mean_pct'] - base['mean_pct']:+.2f}pp)\n")


if __name__ == "__main__":
    main()
