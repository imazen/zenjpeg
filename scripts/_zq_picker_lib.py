"""
Shared helpers for the Zq picker training pipeline. All speed-ups
live here so the per-script orchestrators (zq_bytes_distill.py,
zq_bytes_distill_reduced.py, zq_distill_capacity_sweep.py,
zq_hybrid_heads.py, zq_feature_*.py, …) stay short.

Three big wins encapsulated:

1. **Disk-cached dataset construction.** Loading the Pareto TSV +
   Features TSV and pivoting into (X, Y, meta) takes ~30 s in pure
   Python. After the first run we serialize to .npz and subsequent
   loads are ~100 ms. Cache key is the TSV mtimes + the
   `KEEP_FEATURES` list — invalidates correctly on any input or
   schema change.

2. **Parallel teacher training.** sklearn's HistGB is single-threaded.
   Per-config training is embarrassingly parallel — `joblib.Parallel`
   with `n_jobs=-1` gives ~10-16× speedup on a 16-core box. Drops the
   teacher step from 5-10 min to 30-60 s.

3. **HistGB fast/full presets.** `HISTGB_FAST` (max_iter=100,
   max_depth=4) is the iteration default; `HISTGB_FULL`
   (max_iter=400, max_depth=8) matches the production distill. The
   ablation work validated that ranking is stable between them — use
   FAST while iterating, FULL for the final bake.
"""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

ZQ_TARGETS_DEFAULT = list(range(0, 70, 5)) + list(range(70, 101, 2))
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC_DEFAULT = 0.20
SEED_DEFAULT = 0xCAFE

CACHE_DIR = Path("/tmp/zq_picker_cache")

# Production preset — matches zq_bytes_distill.py teacher.
HISTGB_FULL = dict(
    max_iter=400,
    max_depth=8,
    learning_rate=0.05,
    l2_regularization=0.5,
    random_state=SEED_DEFAULT,
)

# Iteration preset — quality loss ~1 pp vs FULL but ranking stable.
# Use this for ablation sweeps, hyperparameter searches, anything
# you'll re-run more than twice.
HISTGB_FAST = dict(
    max_iter=100,
    max_depth=4,
    learning_rate=0.1,
    l2_regularization=0.5,
    random_state=SEED_DEFAULT,
)


# ---------- Cache key + load ----------


def _cache_key(
    pareto_path: Path, features_path: Path, keep_features: list[str] | None
) -> str:
    """Stable hash for the cache filename. Invalidates on TSV mtime
    or KEEP_FEATURES change."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(pareto_path.resolve()).encode())
    h.update(str(int(pareto_path.stat().st_mtime)).encode())
    h.update(str(features_path.resolve()).encode())
    h.update(str(int(features_path.stat().st_mtime)).encode())
    if keep_features is not None:
        for f in keep_features:
            h.update(f.encode())
            h.update(b"\x00")
    return h.hexdigest()


def load_pareto_raw(path: Path) -> tuple[dict, dict]:
    """Slow path: parse the full Pareto TSV. Returns (rows, config_names)."""
    rows: dict = defaultdict(list)
    config_names: dict = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                cid = int(r["config_id"])
                bytes_v = int(r["bytes"])
                zensim_v = float(r["zensim"])
            except (ValueError, KeyError):
                continue
            config_names.setdefault(cid, r["config_name"])
            key = (
                r["image_path"],
                r["size_class"],
                int(r["width"]),
                int(r["height"]),
            )
            rows[key].append({"config_id": cid, "bytes": bytes_v, "zensim": zensim_v})
    return rows, config_names


def load_features_raw(
    path: Path, keep_features: list[str] | None
) -> tuple[dict, list[str]]:
    """Slow path: parse the features TSV. Optionally restrict + reorder columns."""
    feats: dict = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        all_cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        if keep_features is not None:
            cols = [c for c in keep_features if c in all_cols]
            missing = [c for c in keep_features if c not in all_cols]
            if missing:
                raise SystemExit(f"missing feature columns in TSV: {missing}")
        else:
            cols = all_cols
        for r in rdr:
            feats[(r["image_path"], r["size_class"])] = np.array(
                [float(r[c]) for c in cols], dtype=np.float32
            )
    return feats, cols


def build_dataset_simple(
    pareto: dict,
    config_names: dict,
    feats: dict,
    feat_cols: list[str],
    zq_targets: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Build (Xs, Y, meta) where Xs is the simple input vector
    (n_feats + 4 size onehot + log_pixels + zq_norm) and Y[i, c] is
    log(min bytes for config c at the given image+size+zq), or NaN
    if config c didn't reach the target on that cell.

    No cross-term engineering — that's the codec's choice. This
    function is the canonical "load + pivot" step shared by every
    fitter.
    """
    n_configs = max(config_names) + 1
    Xs_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []
    meta: list[tuple] = []
    for (image, size, w, h), samples in pareto.items():
        feat_key = (image, size)
        if feat_key not in feats:
            continue
        f = feats[feat_key]
        log_px = math.log(max(1, w * h))
        size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float32)
        size_oh[SIZE_INDEX[size]] = 1.0

        per_cfg: dict = defaultdict(lambda: defaultdict(lambda: math.inf))
        for s in samples:
            for zq in zq_targets:
                if (
                    s["zensim"] >= zq
                    and s["bytes"] < per_cfg[zq][s["config_id"]]
                ):
                    per_cfg[zq][s["config_id"]] = s["bytes"]

        for zq in zq_targets:
            if not per_cfg[zq]:
                continue
            zq_norm = zq / 100.0
            xs = np.concatenate(
                [f, size_oh, np.array([log_px, zq_norm], dtype=np.float32)]
            )
            y = np.full(n_configs, np.nan, dtype=np.float32)
            for cfg, b in per_cfg[zq].items():
                if b > 0 and not math.isinf(b):
                    y[cfg] = math.log(b)
            Xs_rows.append(xs)
            Y_rows.append(y)
            meta.append((image, size, zq))
    return np.stack(Xs_rows), np.stack(Y_rows), meta


def load_or_build_dataset(
    pareto_path: Path,
    features_path: Path,
    keep_features: list[str] | None = None,
    zq_targets: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple], list[str], dict[int, str]]:
    """Cached wrapper around `load_pareto_raw` + `load_features_raw` +
    `build_dataset_simple`. First call: ~30 s. Subsequent calls with
    the same inputs: ~100 ms.
    """
    if zq_targets is None:
        zq_targets = ZQ_TARGETS_DEFAULT
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(pareto_path, features_path, keep_features)
    cache_path = CACHE_DIR / f"dataset_{key}.npz"

    if cache_path.exists():
        sys.stderr.write(f"[cache] loading {cache_path.name}\n")
        z = np.load(cache_path, allow_pickle=True)
        Xs = z["Xs"]
        Y = z["Y"]
        meta = [tuple(m) for m in z["meta"].tolist()]
        feat_cols = [str(c) for c in z["feat_cols"].tolist()]
        config_names = {int(k): str(v) for k, v in z["config_names"].item().items()}
        return Xs, Y, meta, feat_cols, config_names

    sys.stderr.write(
        f"[cache miss] building dataset from {pareto_path} + {features_path}...\n"
    )
    pareto, config_names = load_pareto_raw(pareto_path)
    feats, feat_cols = load_features_raw(features_path, keep_features)
    Xs, Y, meta = build_dataset_simple(pareto, config_names, feats, feat_cols, zq_targets)
    np.savez(
        cache_path,
        Xs=Xs,
        Y=Y,
        meta=np.array(meta, dtype=object),
        feat_cols=np.array(feat_cols),
        config_names=np.array(dict(config_names), dtype=object),
    )
    sys.stderr.write(
        f"[cache] wrote {cache_path.name} ({Xs.shape[0]} rows × "
        f"{Xs.shape[1]} inputs × {Y.shape[1]} configs)\n"
    )
    return Xs, Y, meta, feat_cols, config_names


# ---------- Train/val split (stable across runs) ----------


def split_by_image(
    meta: list[tuple], holdout_frac: float = HOLDOUT_FRAC_DEFAULT, seed: int = SEED_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """Image-level holdout split. Same image's rows always go to the
    same side, regardless of size class or zq target."""
    rng = np.random.default_rng(seed)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * holdout_frac))
    val_set = set(images[:n_val])
    train_idx = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    return train_idx, val_idx


# ---------- Parallel teacher training ----------


def _fit_one_teacher(
    cfg: int, X_tr: np.ndarray, y_col: np.ndarray, params: dict
) -> tuple[int, Any]:
    """Worker function for `train_teachers_parallel`. Returns
    (cfg_idx, fitted_estimator_or_None)."""
    mask = ~np.isnan(y_col)
    if mask.sum() < 50:
        return cfg, None
    gbm = HistGradientBoostingRegressor(**params)
    gbm.fit(X_tr[mask], y_col[mask])
    return cfg, gbm


def train_teachers_parallel(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    params: dict | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> list[Any]:
    """Train one HistGB teacher per output column in parallel.

    With `n_jobs=-1` on a 16-core box, ~10× speedup over the serial
    loop. Returns a list of length `Y_tr.shape[1]`; entries are
    fitted estimators or `None` for columns with too few non-NaN
    rows (< 50)."""
    if params is None:
        params = HISTGB_FULL
    n_configs = Y_tr.shape[1]
    sys.stderr.write(
        f"[teachers] training {n_configs} per-config HistGB models in parallel "
        f"(n_jobs={n_jobs}, max_iter={params['max_iter']}, max_depth={params['max_depth']})\n"
    )
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(_fit_one_teacher)(c, X_tr, Y_tr[:, c], params)
        for c in range(n_configs)
    )
    teachers: list[Any] = [None] * n_configs
    for cfg, model in results:
        teachers[cfg] = model
    n_trained = sum(1 for t in teachers if t is not None)
    sys.stderr.write(f"[teachers] {n_trained}/{n_configs} models trained\n")
    return teachers


def teacher_predict_all(
    teachers: list[Any], X: np.ndarray, fallback_means: np.ndarray
) -> np.ndarray:
    """Stack predictions from `teachers[c]` for c in 0..n_configs into
    a (n_rows, n_configs) array. None entries get filled from
    `fallback_means[c]` (typically `np.nanmean(Y_tr[:, c])`)."""
    n_configs = len(teachers)
    out = np.zeros((X.shape[0], n_configs), dtype=np.float32)
    for c, t in enumerate(teachers):
        if t is None:
            fill = fallback_means[c]
            out[:, c] = (
                fill if (isinstance(fill, float) and not math.isnan(fill)) else 0.0
            )
        else:
            out[:, c] = t.predict(X)
    return out


# ---------- Argmin evaluation (shared) ----------


def evaluate_argmin(
    Y_pred: np.ndarray,
    Y_actual: np.ndarray,
    meta: list[tuple],
    mask: np.ndarray,
) -> dict:
    """Standard argmin-overhead evaluation used by every fitter.

    Returns mean / p50 / p75 / p90 mean overhead, argmin accuracy,
    and a per-zq breakdown.
    """
    n_rows = Y_pred.shape[0]
    overheads: list[float] = []
    correct = 0
    per_zq: dict = defaultdict(list)
    for i in range(n_rows):
        actual = Y_actual[i]
        pred = Y_pred[i]
        m = (~np.isnan(actual)) & mask
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
    if not overheads:
        return {"n": 0}
    arr = np.array(overheads)
    return {
        "n": int(len(arr)),
        "argmin_acc": correct / len(arr),
        "mean_pct": float(100 * arr.mean()),
        "p50_pct": float(100 * np.percentile(arr, 50)),
        "p75_pct": float(100 * np.percentile(arr, 75)),
        "p90_pct": float(100 * np.percentile(arr, 90)),
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
