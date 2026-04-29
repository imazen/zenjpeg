#!/usr/bin/env python3
"""
Hybrid-heads picker training (v0.2 design).

Splits the 120-config grid into:
  - 12 categorical cells: (color, sub, trellis_kind, SA)
    - color ∈ {ycbcr, xyb}
    - sub ∈ {444, 420}      (xyb's BQuarter is labeled "420" too)
    - trellis_kind ∈ {noT, hyb}
    - SA ∈ {true, false}    (only ycbcr; xyb cells have SA always false)
  - 2 continuous predictions per cell:
    - chroma_scale* (value in [0.6, 1.5])
    - lambda*       (value in {8.0, 14.5, 25.0} for hyb, sentinel=0.0 for noT)

For each (image, size, target_zq), compute the within-cell optimal:
  bytes(cell)     = min bytes over configs in cell that reach zq
  chroma_scale*   = chroma_scale of the within-cell optimal config
  lambda*         = lambda     of the within-cell optimal config

Train an MLP with 36 outputs (12 bytes + 12 chroma + 12 lambda).
Bytes head is log-space scalar regression (same as today). Scalar
heads are direct value regression.

At inference:
  Y = picker.predict(features)
  bytes_log = Y[0..12]; chroma = Y[12..24]; lam = Y[24..36]
  cell_idx = argmin(bytes_log, mask=allowed_cells)
  encoder_config = (
      cells[cell_idx].color,
      cells[cell_idx].sub,
      cells[cell_idx].sa,
      cells[cell_idx].trellis_on,
      lambda     = lam[cell_idx]    if trellis_on else None,
      chroma_scale = chroma[cell_idx],
  )

The model learns *Pareto-optimal scalars* the codec consumer can
clamp to caller constraints (chroma_scale ∈ [0.8, 1.2], etc.).

Output:
  benchmarks/zq_bytes_hybrid_2026-04-29.json   — model weights + manifest
  benchmarks/zq_bytes_hybrid_2026-04-29.log    — training summary
"""

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PARETO = Path("benchmarks/zq_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zq_pareto_features_2026-04-29.tsv")
OUT_LOG = Path("benchmarks/zq_bytes_hybrid_2026-04-29.log")
OUT_JSON = Path("benchmarks/zq_bytes_hybrid_2026-04-29.json")

ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

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

# Lambda value for the "no trellis" sentinel. Picker still emits a
# value at this output index; codec ignores it when the categorical
# cell is `trellis_on=false`. We pick 0.0 because it's clearly
# "out of band" relative to the real range {8, 14.5, 25}.
LAMBDA_NOTRELLIS_SENTINEL = 0.0

CONFIG_NAMES: dict = {}


# ---------- Config-name parser ----------

# Pattern examples:
#   ycbcr_444_noT_cs60        → ycbcr, 444, trellis_off, no SA, lambda=N/A, cs=0.60
#   ycbcr_444_noT_cs60_sa     → ycbcr, 444, trellis_off, SA on,  lambda=N/A, cs=0.60
#   ycbcr_444_hyb80_cs60      → ycbcr, 444, trellis_on,  no SA, lambda=8.0, cs=0.60
#   ycbcr_444_hyb145_cs100_sa → ycbcr, 444, trellis_on,  SA on,  lambda=14.5, cs=1.00
#   xyb_420_hyb250_cs150      → xyb, 420 (BQuarter), trellis_on, lambda=25.0, cs=1.50

CONFIG_RE = re.compile(
    r"^(?P<color>ycbcr|xyb)_(?P<sub>444|420)_"
    r"(?:noT|hyb(?P<lam>\d+))_cs(?P<cs>\d+)(?P<sa>_sa)?$"
)


def parse_config_name(name: str) -> dict:
    m = CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable config name: {name}")
    color = m.group("color")
    sub = m.group("sub")
    lam_raw = m.group("lam")
    cs_raw = m.group("cs")
    sa = m.group("sa") is not None
    trellis_on = lam_raw is not None
    lam_val = None
    if trellis_on:
        # hyb80 → 8.0, hyb145 → 14.5, hyb250 → 25.0
        # encoded as "lambda * 10" digits (8 → 80, 14.5 → 145, 25 → 250)
        lam_int = int(lam_raw)
        lam_val = lam_int / 10.0
    cs_int = int(cs_raw)
    cs_val = cs_int / 100.0
    return {
        "color": color,
        "sub": sub,
        "sa": sa,
        "trellis_on": trellis_on,
        "lambda": lam_val if lam_val is not None else LAMBDA_NOTRELLIS_SENTINEL,
        "chroma_scale": cs_val,
    }


def categorical_key(parsed: dict) -> tuple:
    """The (color, sub, trellis_on, sa) tuple. xyb cells always have sa=False."""
    return (parsed["color"], parsed["sub"], parsed["trellis_on"], parsed["sa"])


# ---------- Data loading ----------


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
        all_cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        cols = [c for c in KEEP_FEATURES if c in all_cols]
        for r in rdr:
            feats[(r["image_path"], r["size_class"])] = np.array(
                [float(r[c]) for c in cols], dtype=np.float32
            )
    return feats, cols


# ---------- Build categorical cell mapping ----------


def build_cell_index():
    """Return:
       cells: list of dicts describing each cell (in stable order)
       cell_id_by_key: {(color, sub, trellis_on, sa) -> int}
       config_to_cell: {config_id -> cell_id}
       config_to_parsed: {config_id -> parsed dict}
    """
    parsed_all = {}
    for cid, name in CONFIG_NAMES.items():
        parsed_all[cid] = parse_config_name(name)

    keys = sorted({categorical_key(p) for p in parsed_all.values()})
    cell_id_by_key = {k: i for i, k in enumerate(keys)}

    cells = []
    for k in keys:
        color, sub, trellis_on, sa = k
        # Pick a representative human-readable label
        sa_tag = "_sa" if sa else ""
        trel_tag = "trellis" if trellis_on else "noT"
        label = f"{color}_{sub}_{trel_tag}{sa_tag}"
        # Find member configs
        members = [cid for cid, p in parsed_all.items() if categorical_key(p) == k]
        cells.append(
            {
                "id": cell_id_by_key[k],
                "label": label,
                "color": color,
                "sub": sub,
                "trellis_on": trellis_on,
                "sa": sa,
                "member_config_ids": sorted(members),
            }
        )

    config_to_cell = {cid: cell_id_by_key[categorical_key(p)] for cid, p in parsed_all.items()}
    return cells, cell_id_by_key, config_to_cell, parsed_all


# ---------- Build training dataset ----------


def build_dataset(pareto, feats, feat_cols, cells, config_to_cell, parsed_all):
    """Per (image, size, zq) row, compute within-cell optimal:
       bytes_log[c]    = log(min bytes in cell c over configs that reach zq)
       chroma_scale[c] = chroma of the within-cell optimal
       lambda[c]       = lambda of the within-cell optimal
       reachable[c]    = 1 if any config in cell c reached zq, 0 otherwise
    """
    n_cells = len(cells)
    Xs_rows, Xe_rows = [], []
    bytes_log_rows, chroma_rows, lambda_rows, reach_rows = [], [], [], []
    meta = []

    for (image, size, w, h), samples in pareto.items():
        feat_key = (image, size)
        if feat_key not in feats:
            continue
        f = feats[feat_key]
        log_px = math.log(max(1, w * h))
        size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float32)
        size_oh[SIZE_INDEX[size]] = 1.0

        # Group samples by config to track per-config best.
        # (one config can have multiple q values; pareto-best for each
        # cell at each zq target is the cheapest config that crosses zq.)
        by_cfg = defaultdict(list)
        for s in samples:
            by_cfg[s["config_id"]].append(s)

        for zq in ZQ_TARGETS:
            cell_bytes = [math.inf] * n_cells
            cell_cs = [math.nan] * n_cells
            cell_lam = [math.nan] * n_cells
            cell_reach = [False] * n_cells

            for cfg_id, hits in by_cfg.items():
                # Cheapest sample for this config that reaches zq.
                best_b = math.inf
                for s in hits:
                    if s["zensim"] >= zq and s["bytes"] < best_b:
                        best_b = s["bytes"]
                if math.isinf(best_b):
                    continue
                c = config_to_cell[cfg_id]
                if best_b < cell_bytes[c]:
                    cell_bytes[c] = best_b
                    p = parsed_all[cfg_id]
                    cell_cs[c] = p["chroma_scale"]
                    cell_lam[c] = p["lambda"]
                    cell_reach[c] = True

            if not any(cell_reach):
                continue

            zq_norm = zq / 100.0
            # Engineered input vector — same as v1.1 student to keep
            # the comparison apples-to-apples.
            xs = np.concatenate([f, size_oh, np.array([log_px, zq_norm], dtype=np.float32)])
            xe = np.concatenate([
                f,
                size_oh,
                np.array(
                    [log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px],
                    dtype=np.float32,
                ),
                zq_norm * f,
                np.array([0.0], dtype=np.float32),  # icc placeholder
            ])

            bytes_log = np.array(
                [math.log(b) if not math.isinf(b) else math.nan for b in cell_bytes],
                dtype=np.float32,
            )
            chroma = np.array(cell_cs, dtype=np.float32)
            lam = np.array(cell_lam, dtype=np.float32)
            reach = np.array(cell_reach, dtype=bool)

            Xs_rows.append(xs)
            Xe_rows.append(xe)
            bytes_log_rows.append(bytes_log)
            chroma_rows.append(chroma)
            lambda_rows.append(lam)
            reach_rows.append(reach)
            meta.append((image, size, zq))

    return (
        np.stack(Xs_rows),
        np.stack(Xe_rows),
        np.stack(bytes_log_rows),
        np.stack(chroma_rows),
        np.stack(lambda_rows),
        np.stack(reach_rows),
        meta,
    )


# ---------- Evaluation ----------


def evaluate_argmin(pred_bytes_log, actual_bytes_log, reach, meta, mask):
    """Categorical argmin over allowed reachable cells."""
    n_rows = pred_bytes_log.shape[0]
    overheads, correct = [], 0
    for i in range(n_rows):
        actual = actual_bytes_log[i]
        pred = pred_bytes_log[i]
        m = reach[i] & mask
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
        "n": int(len(arr)),
        "argmin_acc": correct / len(arr),
        "mean_pct": float(100 * arr.mean()),
        "p50_pct": float(100 * np.percentile(arr, 50)),
        "p90_pct": float(100 * np.percentile(arr, 90)),
    }


def evaluate_scalars(pred_chroma, actual_chroma, pred_lam, actual_lam, reach):
    """RMSE on the chroma_scale and lambda predictions, computed over
    reachable cells only (where the targets exist).
    """
    rmse = {}
    cs_diff = []
    lam_diff = []
    for i in range(pred_chroma.shape[0]):
        for c in range(pred_chroma.shape[1]):
            if not reach[i, c]:
                continue
            cs_diff.append(pred_chroma[i, c] - actual_chroma[i, c])
            # lambda only meaningful when trellis_on (target != sentinel)
            if not math.isnan(actual_lam[i, c]) and actual_lam[i, c] > 0:
                lam_diff.append(pred_lam[i, c] - actual_lam[i, c])
    cs_arr = np.array(cs_diff, dtype=np.float64)
    lam_arr = np.array(lam_diff, dtype=np.float64) if lam_diff else np.array([0.0])
    rmse["chroma_scale"] = float(np.sqrt((cs_arr ** 2).mean()))
    rmse["chroma_scale_mae"] = float(np.abs(cs_arr).mean())
    rmse["lambda"] = float(np.sqrt((lam_arr ** 2).mean()))
    rmse["lambda_mae"] = float(np.abs(lam_arr).mean())
    return rmse


# ---------- Train ----------


def train_teacher_per_cell(Xs_tr, bytes_log_tr, chroma_tr, lam_tr, reach_tr, n_cells):
    """Per-cell HistGB regressors for: bytes_log, chroma_scale, lambda."""
    sys.stderr.write(f"\nTraining per-cell HistGB teachers ({n_cells} cells × 3 targets)...\n")

    teachers_bytes = []
    teachers_chroma = []
    teachers_lambda = []

    cs_means = np.nanmean(chroma_tr, axis=0)
    lam_means = np.nanmean(np.where(lam_tr > 0, lam_tr, np.nan), axis=0)

    for c in range(n_cells):
        # bytes head: regress log_bytes where reachable
        m = reach_tr[:, c]
        if m.sum() < 50:
            teachers_bytes.append(None)
            teachers_chroma.append(None)
            teachers_lambda.append(None)
            continue
        # bytes
        gbm = HistGradientBoostingRegressor(
            max_iter=400, max_depth=8, learning_rate=0.05,
            l2_regularization=0.5, random_state=SEED,
        )
        gbm.fit(Xs_tr[m], bytes_log_tr[m, c])
        teachers_bytes.append(gbm)
        # chroma
        gbm_cs = HistGradientBoostingRegressor(
            max_iter=400, max_depth=8, learning_rate=0.05,
            l2_regularization=0.5, random_state=SEED,
        )
        gbm_cs.fit(Xs_tr[m], chroma_tr[m, c])
        teachers_chroma.append(gbm_cs)
        # lambda — only train where trellis is actually on (target > 0)
        m_lam = m & (lam_tr[:, c] > 0)
        if m_lam.sum() < 50:
            teachers_lambda.append(None)
        else:
            gbm_lam = HistGradientBoostingRegressor(
                max_iter=400, max_depth=8, learning_rate=0.05,
                l2_regularization=0.5, random_state=SEED,
            )
            gbm_lam.fit(Xs_tr[m_lam], lam_tr[m_lam, c])
            teachers_lambda.append(gbm_lam)
        sys.stderr.write(f"  cell {c}: trained on n={m.sum()}\n")

    return teachers_bytes, teachers_chroma, teachers_lambda, cs_means, lam_means


def teacher_predict_all(teachers, Xs, fallback_means, n_cells):
    out = np.zeros((Xs.shape[0], n_cells), dtype=np.float32)
    for c in range(n_cells):
        if teachers[c] is None:
            out[:, c] = fallback_means[c] if not math.isnan(fallback_means[c]) else 0.0
        else:
            out[:, c] = teachers[c].predict(Xs)
    return out


def main():
    sys.stderr.write(f"Loading {PARETO}...\n")
    pareto = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES)
    sys.stderr.write(f"Loaded {len(pareto)} cells × {len(feat_cols)} features\n")

    cells, cell_id_by_key, config_to_cell, parsed_all = build_cell_index()
    n_cells = len(cells)
    sys.stderr.write(f"\nCategorical cells: {n_cells}\n")
    for c in cells:
        sys.stderr.write(f"  {c['id']:>2d}: {c['label']:30s}  ({len(c['member_config_ids'])} configs)\n")

    Xs, Xe, bytes_log, chroma, lam, reach, meta = build_dataset(
        pareto, feats, feat_cols, cells, config_to_cell, parsed_all
    )
    sys.stderr.write(
        f"\nDecision rows: {len(Xs)}; Xs={Xs.shape[1]}, Xe={Xe.shape[1]}, n_cells={n_cells}\n"
    )

    rng = np.random.default_rng(SEED)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * HOLDOUT_FRAC))
    val_set = set(images[:n_val])
    tr = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    va = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"Train rows: {len(tr)}, val rows: {len(va)}\n")

    Xs_tr, Xs_va = Xs[tr], Xs[va]
    Xe_tr, Xe_va = Xe[tr], Xe[va]
    bl_tr, bl_va = bytes_log[tr], bytes_log[va]
    cs_tr, cs_va = chroma[tr], chroma[va]
    lam_tr, lam_va = lam[tr], lam[va]
    rch_tr, rch_va = reach[tr], reach[va]
    meta_va = [meta[i] for i in va]

    # --- Teacher
    t_bytes, t_chroma, t_lambda, cs_means, lam_means = train_teacher_per_cell(
        Xs_tr, bl_tr, cs_tr, lam_tr, rch_tr, n_cells
    )
    sys.stderr.write("\nGenerating teacher soft targets (val + train)...\n")
    bytes_pred_tr = teacher_predict_all(t_bytes, Xs_tr, np.nanmean(bl_tr, axis=0), n_cells)
    bytes_pred_va = teacher_predict_all(t_bytes, Xs_va, np.nanmean(bl_tr, axis=0), n_cells)
    chroma_pred_tr = teacher_predict_all(t_chroma, Xs_tr, cs_means, n_cells)
    chroma_pred_va = teacher_predict_all(t_chroma, Xs_va, cs_means, n_cells)
    lam_pred_tr = teacher_predict_all(t_lambda, Xs_tr, lam_means, n_cells)
    lam_pred_va = teacher_predict_all(t_lambda, Xs_va, lam_means, n_cells)

    all_mask = np.ones(n_cells, dtype=bool)
    teacher_argmin = evaluate_argmin(bytes_pred_va, bl_va, rch_va, meta_va, all_mask)
    teacher_scalars = evaluate_scalars(chroma_pred_va, cs_va, lam_pred_va, lam_va, rch_va)
    sys.stderr.write(
        f"\nTeacher metrics: argmin mean overhead {teacher_argmin['mean_pct']:.2f}% "
        f"argmin_acc {teacher_argmin['argmin_acc']:.1%}\n"
    )
    sys.stderr.write(
        f"  scalar RMSE: chroma {teacher_scalars['chroma_scale']:.4f}  "
        f"lambda {teacher_scalars['lambda']:.3f}\n"
    )

    # --- Student
    # Soft targets: 12 bytes + 12 chroma + 12 lambda = 36 outputs
    soft_tr = np.concatenate([bytes_pred_tr, chroma_pred_tr, lam_pred_tr], axis=1)
    sys.stderr.write(f"\nTraining MLP student (hidden=128x2, output_dim={soft_tr.shape[1]})...\n")

    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_va_s = scaler.transform(Xe_va)
    student = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=2e-3,
        batch_size=512,
        max_iter=500,
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
    pred_bytes = Y_va_pred[:, :n_cells]
    pred_chroma = Y_va_pred[:, n_cells : 2 * n_cells]
    pred_lambda = Y_va_pred[:, 2 * n_cells : 3 * n_cells]

    student_argmin = evaluate_argmin(pred_bytes, bl_va, rch_va, meta_va, all_mask)
    student_scalars = evaluate_scalars(pred_chroma, cs_va, pred_lambda, lam_va, rch_va)
    sys.stderr.write(
        f"\nStudent metrics: argmin mean overhead {student_argmin['mean_pct']:.2f}% "
        f"argmin_acc {student_argmin['argmin_acc']:.1%}\n"
    )
    sys.stderr.write(
        f"  scalar RMSE: chroma {student_scalars['chroma_scale']:.4f}  "
        f"lambda {student_scalars['lambda']:.3f}\n"
    )

    # --- Persist
    n_params = sum(c.size + i.size for c, i in zip(student.coefs_, student.intercepts_))
    out = {
        "n_inputs": int(Xe.shape[1]),
        "n_outputs": 3 * n_cells,
        "n_cells": n_cells,
        "config_names": {int(k): v for k, v in CONFIG_NAMES.items()},
        "feat_cols": feat_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(student.coefs_, student.intercepts_)
        ],
        "activation": "relu",
        "hybrid_heads_manifest": {
            "n_cells": n_cells,
            "cells": cells,
            "output_layout": {
                "bytes_log": [0, n_cells],
                "chroma_scale": [n_cells, 2 * n_cells],
                "lambda": [2 * n_cells, 3 * n_cells],
            },
            "lambda_notrellis_sentinel": LAMBDA_NOTRELLIS_SENTINEL,
        },
        "teacher_metrics": {"argmin": teacher_argmin, "scalars": teacher_scalars},
        "student_metrics": {"argmin": student_argmin, "scalars": student_scalars},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    sys.stderr.write(
        f"\nWrote {OUT_JSON} ({n_params} weights, {n_params*2/1024:.1f} KB f16)\n"
    )

    # --- Report
    lines = []
    def w(s):
        lines.append(s)
        sys.stderr.write(s + "\n")

    w("\n# Hybrid-heads picker — categorical bytes + scalar (chroma_scale, lambda)")
    w(f"Train rows: {len(tr)}, val rows: {len(va)}")
    w(f"n_cells: {n_cells}, output_dim: {3 * n_cells}")
    w(f"Student: MLP {Xe.shape[1]} -> 128 -> 128 -> {3 * n_cells}, "
      f"{n_params} params (~{n_params*2/1024:.1f} KB f16)")
    w("")
    w("## Categorical cells")
    for c in cells:
        w(f"  {c['id']:>2d}: {c['label']:30s}  ({len(c['member_config_ids'])} member configs)")
    w("")
    w("## Argmin (categorical) — vs reachable per-row optimal")
    w(f"  Teacher: mean {teacher_argmin['mean_pct']:.2f}%  argmin_acc {teacher_argmin['argmin_acc']:.1%}")
    w(f"  Student: mean {student_argmin['mean_pct']:.2f}%  argmin_acc {student_argmin['argmin_acc']:.1%}")
    w("")
    w("## Scalar regression RMSE")
    w(f"  Teacher chroma_scale RMSE: {teacher_scalars['chroma_scale']:.4f}  "
      f"(MAE {teacher_scalars['chroma_scale_mae']:.4f}, range 0.6..1.5)")
    w(f"  Teacher lambda RMSE:       {teacher_scalars['lambda']:.3f}   "
      f"(MAE {teacher_scalars['lambda_mae']:.3f}, range 8..25)")
    w(f"  Student chroma_scale RMSE: {student_scalars['chroma_scale']:.4f}  "
      f"(MAE {student_scalars['chroma_scale_mae']:.4f})")
    w(f"  Student lambda RMSE:       {student_scalars['lambda']:.3f}   "
      f"(MAE {student_scalars['lambda_mae']:.3f})")

    OUT_LOG.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
