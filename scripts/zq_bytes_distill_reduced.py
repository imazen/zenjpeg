#!/usr/bin/env python3
"""
8-feature reduced-schema distillation.

Wraps `zq_bytes_distill.py`'s training pipeline but filters the
feature columns at load time to the 8-feature picker schema
validated in `zq_reduced_schema_validate.py`.

The teacher (HistGB per-config × 120, simple inputs) and student
(shared MLP 64→64→120 with engineered cross-terms) are unchanged.
Only the input feature column subset differs.

Outputs:
- benchmarks/zq_bytes_distill_reduced_2026-04-29.json
- benchmarks/zq_bytes_distill_reduced_2026-04-29.log

The output JSON shape is identical to `zq_bytes_distill.py`'s, so
`tools/bake_picker.py` consumes it unchanged. The schema_hash will
differ (different feat_cols, different extra_axes).
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
OUT_LOG = Path("benchmarks/zq_bytes_distill_reduced_2026-04-29.log")
OUT_JSON = Path("benchmarks/zq_bytes_distill_reduced_2026-04-29.json")

ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

# 8-feature reduced schema (validated 2026-04-29 on production HistGB).
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


def load_features(path, keep_set):
    feats = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        all_cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        # Preserve KEEP_FEATURES order, not the TSV order, so the
        # baked schema is deterministic and matches what the codec
        # crate compiles in.
        ordered_cols = [c for c in KEEP_FEATURES if c in all_cols]
        missing = [c for c in KEEP_FEATURES if c not in all_cols]
        if missing:
            raise SystemExit(f"missing feature columns in TSV: {missing}")
        for r in rdr:
            feats[(r["image_path"], r["size_class"])] = np.array(
                [float(r[c]) for c in ordered_cols], dtype=np.float32
            )
    return feats, ordered_cols


def build_dataset(pareto, feats, feat_cols):
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
            xs = np.concatenate([f, size_oh, np.array([log_px, zq_norm], dtype=np.float32)])
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
        pb = np.where(m, np.exp(np.clip(pred, -30, 30)), np.inf)
        actual_best = int(np.argmin(ab))
        pred_best = int(np.argmin(pb))
        if pred_best == actual_best:
            correct += 1
        ov = (ab[pred_best] - ab[actual_best]) / ab[actual_best]
        overheads.append(ov)
        per_zq[meta[i][2]].append(ov)
    arr = np.array(overheads)
    return {
        "name": name,
        "n": int(len(arr)),
        "unreachable": unreach,
        "argmin_accuracy": correct / len(arr),
        "overhead_mean_pct": float(100 * arr.mean()),
        "overhead_p50_pct": float(100 * np.percentile(arr, 50)),
        "overhead_p75_pct": float(100 * np.percentile(arr, 75)),
        "overhead_p90_pct": float(100 * np.percentile(arr, 90)),
    }


def main():
    sys.stderr.write(f"Loading {PARETO}...\n")
    pareto = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES, set(KEEP_FEATURES))
    sys.stderr.write(f"Loaded {len(pareto)} cells × {len(feat_cols)} features (reduced schema)\n")
    sys.stderr.write(f"  feat_cols: {feat_cols}\n")

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
        if (cfg + 1) % 30 == 0:
            sys.stderr.write(f"  {cfg+1}/{n_configs} teachers trained\n")

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

    all_mask = np.ones(n_configs, dtype=bool)
    sys.stderr.write("\nTeacher val metrics (sanity check)...\n")
    t_all = evaluate_argmin(soft_va, Y_va, meta_va, all_mask, "HistGB teacher")

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
    s_all = evaluate_argmin(Y_va_pred, Y_va, meta_va, all_mask, "MLP student")

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

    lines = []

    def w(s):
        lines.append(s); sys.stderr.write(s + "\n")

    w("\n# Distilled MLP — 8-feature reduced schema")
    w(f"Train rows: {len(Xs_tr)}, val rows: {len(Xs_va)}")
    w(f"Teacher: HistGB per-config × {n_configs}, simple inputs ({Xs.shape[1]}: {len(feat_cols)} feats + 4 size + log_px + zq)")
    w(f"Student: MLP {Xe.shape[1]} -> 64 -> 64 -> {n_configs}, engineered inputs, {n_params} params (~{n_params*4/1024:.1f} KB f32)")
    w("")

    def fmt(m):
        return (f"argmin_acc={m['argmin_accuracy']:.1%}  n={m['n']}  unreach={m['unreachable']}  "
                f"overhead mean={m['overhead_mean_pct']:+.1f}% "
                f"p50={m['overhead_p50_pct']:+.1f}% "
                f"p75={m['overhead_p75_pct']:+.1f}% "
                f"p90={m['overhead_p90_pct']:+.1f}%")

    w("## Argmin metrics (held-out images, all configs allowed)")
    w(f"HistGB teacher: {fmt(t_all)}")
    w(f"MLP student:    {fmt(s_all)}")
    w("")
    w(f"## Features kept ({len(feat_cols)})")
    for c in feat_cols:
        w(f"  {c}")

    OUT_LOG.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
