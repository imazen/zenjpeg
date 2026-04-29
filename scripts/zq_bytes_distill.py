#!/usr/bin/env python3
"""
Distill HistGradientBoosting forest into the small shared MLP.

The HistGB model from `zq_bytes_regression_fit.py` ships at +9.0% mean
overhead but bakes to ~5-7 MB compressed — too big. The vanilla shared
MLP from `zq_bytes_mlp_shared.py` bakes to 59 KB but lands at +16.4%
mean overhead — too noisy.

Distillation uses HistGB's predicted log-bytes as soft targets for the
MLP. The MLP no longer has to learn the noisy raw byte function; it
only has to fit HistGB's already-smoothed surface. Same 59 KB artifact,
much closer to HistGB's accuracy.

Two MLP heads:
- Standard: trained on HistGB soft targets across all 120 configs
- Evaluated under all-configs and ycbcr-only masks at argmin time

Inputs:
- benchmarks/zq_pareto_2026-04-29.tsv
- benchmarks/zq_pareto_features_2026-04-29.tsv

Outputs:
- benchmarks/zq_bytes_distill_2026-04-29.{log,json}
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PARETO = Path("benchmarks/zq_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zq_pareto_features_2026-04-29.tsv")
OUT_LOG = Path("benchmarks/zq_bytes_distill_2026-04-29.log")
OUT_JSON = Path("benchmarks/zq_bytes_distill_2026-04-29.json")

ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

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
            feats[(r["image_path"], r["size_class"])] = np.array([float(r[c]) for c in cols], dtype=np.float32)
    return feats, cols


def build_dataset(pareto, feats, feat_cols):
    """Build (X_simple, X_eng, Y, meta).
    X_simple: 19 + 4 + 1 + 1 = 25 inputs (for HistGB teacher)
    X_eng: simple + 4 polynomial + 19 cross terms + 1 icc = 49 (for MLP student)
    """
    n_configs = max(CONFIG_NAMES) + 1

    Xs_rows, Xe_rows, Y_rows, meta = [], [], [], []
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
            cfg_to_bytes = per_cfg[zq]
            if not cfg_to_bytes:
                continue
            zq_norm = zq / 100.0
            xs = np.concatenate([
                f, size_oh, np.array([log_px, zq_norm], dtype=np.float32),
            ])
            xe = np.concatenate([
                f, size_oh,
                np.array([log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px], dtype=np.float32),
                zq_norm * f,
                np.array([0.0], dtype=np.float32),  # icc placeholder
            ])
            y = np.full(n_configs, np.nan, dtype=np.float32)
            for cfg, b in cfg_to_bytes.items():
                if b > 0 and not math.isinf(b):
                    y[cfg] = math.log(b)
            Xs_rows.append(xs); Xe_rows.append(xe); Y_rows.append(y); meta.append((image, size, zq))

    return np.stack(Xs_rows), np.stack(Xe_rows), np.stack(Y_rows), meta


def evaluate_argmin(Y_pred_log, Y_actual_log, meta, mask, name):
    n = Y_pred_log.shape[0]
    overheads = []
    correct = 0
    per_zq = defaultdict(list)
    unreach = 0
    for i in range(n):
        actual = Y_actual_log[i]
        pred = Y_pred_log[i]
        m = (~np.isnan(actual)) & mask
        if not np.any(m):
            unreach += 1
            continue
        ab = np.where(m, np.exp(actual), np.inf)
        # clip to avoid overflow on rough predictions outside training range
        pb = np.where(m, np.exp(np.clip(pred, -30, 30)), np.inf)
        actual_best = int(np.argmin(ab))
        pred_best = int(np.argmin(pb))
        if pred_best == actual_best:
            correct += 1
        ov = (ab[pred_best] - ab[actual_best]) / ab[actual_best]
        overheads.append(ov)
        per_zq[meta[i][2]].append(ov)
    if not overheads:
        return None
    overheads = np.array(overheads)
    return {
        "name": name,
        "n": int(len(overheads)),
        "unreachable": unreach,
        "argmin_accuracy": correct / len(overheads),
        "overhead_mean_pct": float(100 * np.mean(overheads)),
        "overhead_p50_pct": float(100 * np.percentile(overheads, 50)),
        "overhead_p75_pct": float(100 * np.percentile(overheads, 75)),
        "overhead_p90_pct": float(100 * np.percentile(overheads, 90)),
        "per_zq": {tz: {
            "n": len(v), "mean": float(100 * np.mean(v)),
            "p50": float(100 * np.percentile(v, 50)),
            "p90": float(100 * np.percentile(v, 90)),
        } for tz, v in per_zq.items()},
    }


def main():
    sys.stderr.write(f"Loading {PARETO}...\n")
    pareto = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES)
    sys.stderr.write(f"Loaded {len(pareto)} cells × {len(feat_cols)} features\n")

    Xs, Xe, Y, meta = build_dataset(pareto, feats, feat_cols)
    n_configs = Y.shape[1]
    sys.stderr.write(f"Decision rows: {len(Xs)} × {n_configs} configs (Xs:{Xs.shape[1]}, Xe:{Xe.shape[1]})\n")

    rng = np.random.default_rng(SEED)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * HOLDOUT_FRAC))
    val_set = set(images[:n_val])
    train_idx = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"Train rows: {len(train_idx)}, val rows: {len(val_idx)}\n")

    Xs_tr, Xs_va = Xs[train_idx], Xs[val_idx]
    Xe_tr, Xe_va = Xe[train_idx], Xe[val_idx]
    Y_tr, Y_va = Y[train_idx], Y[val_idx]
    meta_va = [meta[i] for i in val_idx]

    # ============ Step 1: train HistGB teacher per-config ============
    sys.stderr.write("\nTraining HistGB teacher (120 per-config models)...\n")
    teachers = []
    cfg_means = np.nanmean(Y_tr, axis=0)
    for cfg in range(n_configs):
        mask = ~np.isnan(Y_tr[:, cfg])
        if mask.sum() < 50:
            teachers.append(None)
            continue
        gbm = HistGradientBoostingRegressor(
            max_iter=400, max_depth=8, learning_rate=0.05,
            l2_regularization=0.5, random_state=SEED,
        )
        gbm.fit(Xs_tr[mask], Y_tr[mask, cfg])
        teachers.append(gbm)
        if (cfg + 1) % 20 == 0:
            sys.stderr.write(f"  {cfg+1}/{n_configs} teachers trained\n")

    # ============ Step 2: generate dense soft targets on full train+val ============
    sys.stderr.write("\nGenerating soft targets...\n")
    soft_tr = np.zeros((len(train_idx), n_configs), dtype=np.float32)
    soft_va = np.zeros((len(val_idx), n_configs), dtype=np.float32)
    for cfg in range(n_configs):
        if teachers[cfg] is None:
            soft_tr[:, cfg] = cfg_means[cfg]
            soft_va[:, cfg] = cfg_means[cfg]
        else:
            soft_tr[:, cfg] = teachers[cfg].predict(Xs_tr)
            soft_va[:, cfg] = teachers[cfg].predict(Xs_va)

    # Sanity check teacher quality on val
    all_mask = np.ones(n_configs, dtype=bool)
    ycbcr_mask = np.array([CONFIG_NAMES[c].startswith("ycbcr") for c in range(n_configs)], dtype=bool)
    sys.stderr.write("\nTeacher val metrics (sanity check)...\n")
    t_all = evaluate_argmin(soft_va, Y_va, meta_va, all_mask, "HistGB teacher all")
    t_ycb = evaluate_argmin(soft_va, Y_va, meta_va, ycbcr_mask, "HistGB teacher ycbcr")

    # ============ Step 3: train MLP student on soft targets ============
    sys.stderr.write("\nTraining MLP student on HistGB soft targets...\n")
    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_va_s = scaler.transform(Xe_va)
    student = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=2e-3,
        batch_size=512,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        tol=1e-6,
        random_state=SEED,
        verbose=False,
    )
    student.fit(Xe_tr_s, soft_tr)
    sys.stderr.write(f"  trained, final loss={student.loss_:.4f}, n_iter={student.n_iter_}\n")

    Y_va_pred = student.predict(Xe_va_s)

    s_all = evaluate_argmin(Y_va_pred, Y_va, meta_va, all_mask, "MLP student all")
    s_ycb = evaluate_argmin(Y_va_pred, Y_va, meta_va, ycbcr_mask, "MLP student ycbcr")

    # Save weights
    weights = {
        "n_inputs": int(Xe.shape[1]),
        "n_configs": int(n_configs),
        "config_names": {int(k): v for k, v in CONFIG_NAMES.items()},
        "feat_cols": feat_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(student.coefs_, student.intercepts_)
        ],
        "activation": "relu",
    }
    OUT_JSON.write_text(json.dumps(weights))
    n_params = sum(c.size + i.size for c, i in zip(student.coefs_, student.intercepts_))
    sys.stderr.write(f"  {n_params} weights, {OUT_JSON.stat().st_size} bytes JSON ({n_params*4/1024:.1f} KB f32)\n")

    # Report
    lines = []
    def w(s):
        lines.append(s); sys.stderr.write(s + "\n")

    w("# Distilled MLP — bytes-per-config regression (HistGB teacher → small MLP student)")
    w(f"Train rows: {len(Xs_tr)}, val rows: {len(Xs_va)}")
    w(f"Teacher: HistGB per-config × {n_configs}, simple inputs ({Xs.shape[1]})")
    w(f"Student: MLP {Xe.shape[1]} -> 64 -> 64 -> {n_configs}, engineered inputs, {n_params} params (~{n_params*4/1024:.1f} KB f32)")
    w("")

    def fmt(m):
        if m is None: return "n/a"
        return (f"argmin_acc={m['argmin_accuracy']:.1%}  n={m['n']}  unreach={m['unreachable']}  "
                f"overhead mean={m['overhead_mean_pct']:+.1f}% "
                f"p50={m['overhead_p50_pct']:+.1f}% "
                f"p75={m['overhead_p75_pct']:+.1f}% "
                f"p90={m['overhead_p90_pct']:+.1f}%")

    w("## Argmin metrics (held-out images)")
    w(f"HistGB teacher (all):    {fmt(t_all)}")
    w(f"HistGB teacher (ycbcr):  {fmt(t_ycb)}")
    w(f"MLP student (all):       {fmt(s_all)}")
    w(f"MLP student (ycbcr):     {fmt(s_ycb)}")
    w("")

    for label, m in [("Student all", s_all), ("Student ycbcr-only", s_ycb)]:
        if m is None: continue
        w(f"## {label} — per-zq overhead")
        w("zq  | n   |  mean  |  p50   |  p90   |")
        w("----|-----|--------|--------|--------|")
        for tz in sorted(m["per_zq"]):
            d = m["per_zq"][tz]
            w(f"{tz:>3} | {d['n']:>3} | {d['mean']:+5.1f}% | {d['p50']:+5.1f}% | {d['p90']:+5.1f}% |")
        w("")

    OUT_LOG.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
