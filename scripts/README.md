# zenjpeg scripts

Most picker training scripts moved to the canonical, codec-agnostic
location:

  **<https://github.com/imazen/zenanalyze>** → `zenpicker/tools/`

| zenjpeg path (removed)               | Canonical location                         |
|---|---|
| `scripts/zq_bytes_distill.py`        | `zenpicker/tools/train_distill.py`         |
| `scripts/zq_bytes_distill_reduced.py`| `zenpicker/tools/train_distill_reduced.py` |
| `scripts/zq_feature_ablation.py`     | `zenpicker/tools/feature_ablation.py`      |
| `scripts/zq_feature_group_ablation.py`| `zenpicker/tools/feature_group_ablation.py` |
| `scripts/zq_reduced_schema_validate.py`| `zenpicker/tools/validate_schema.py`     |

Run them against zenjpeg's TSV / config taxonomy via the codec
config module at
[`zenpicker/examples/zenjpeg_picker_config.py`](https://github.com/imazen/zenanalyze/blob/main/zenpicker/examples/zenjpeg_picker_config.py):

```bash
PYTHONPATH=<zenanalyze>/zenpicker/examples \
    python3 <zenanalyze>/zenpicker/tools/train_hybrid.py \
        --codec-config zenjpeg_picker_config
```

See the `zenpicker/tools/README.md` in zenanalyze for the file map
and the codec config contract.

## What stays here

| File | Purpose |
|---|---|
| `zq_emit_rust.py`       | Bakes Rust source for the legacy 8-config Zq controller (`src/encode/zq.rs`) — zenjpeg-specific |
| `zq_pareto_fit.py`      | Offline Pareto-front analysis for the legacy 8-config controller — zenjpeg-specific |
| `zq_tier2_subablation.py`| Targeted experiment: can we drop zenanalyze Tier 2 entirely? — kept for reference |
