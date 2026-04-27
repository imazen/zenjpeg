#!/usr/bin/env python3
"""
Emit Rust source for the Pareto-trained Zq controller.

Reads the JSON output of zq_pareto_fit.py (--emit-json mode) and writes
a Rust file that:
  1. Defines the 8-config enum (matches harness's ConfigSpec table).
  2. Bakes per-config regression coefficients as `const`.
  3. Provides `predict_zq(features, target_zq) -> (config_id, starting_q)`
     that runs the per-config regressors, classifies (smallest predicted
     bytes), returns (chosen_config, predicted_q).

Output is hand-pasted into `zenjpeg/src/encode/zq.rs`.
"""

import argparse
import json
import sys


def emit(json_path, out_path):
    with open(json_path) as f:
        fit = json.load(f)

    cfg_names = fit["configs"]
    feature_cols = fit["feature_cols"]
    weights = fit["weights"]  # cfg_id -> [w0, w1, ...]

    src = ["// Auto-generated from scripts/zq_emit_rust.py — do not hand-edit.",
           f"// Source: {json_path}",
           "",
           "/// Pareto-trained zq controller — predict (config, starting_q) from "
           "(features, zq_target).",
           "#[allow(dead_code)]",
           "pub(crate) struct ParetoZqModel;",
           ""]

    for cfg_id, cfg_name in cfg_names.items():
        ws = weights.get(str(cfg_id))
        if ws is None:
            continue
        src.append(f"// config {cfg_id}: {cfg_name}")
        src.append(f"const W_{cfg_name.upper()}: &[f32; {len(ws)}] = &[")
        for w in ws:
            src.append(f"    {w:+.6e},")
        src.append("];")
        src.append("")

    print("\n".join(src), file=open(out_path, "w") if out_path else sys.stdout)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    emit(a.json, a.out)
