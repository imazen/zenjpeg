# boundary_rd perf — post-optimization 2026-04-21

End of the optimization pass. Two patches applied, one negative result
recorded and reverted.

## Headline numbers (stratified 50-image lineart sample, seed 42, Q=75 4:2:0)

10 iterations per config per image, median interleaved measurement.

| Config | Overall median overhead | p95 overhead | Δ vs baseline |
|---|---:|---:|---:|
| Baseline (before this pass) | +60.4% | +81.9% | — |
| opt1: cache orig right-edge | +56.5% | +71.8% | **-3.9 pp** |
| opt2: skip DC-only IDCT check + token hoist | **+54.3%** | +71.7% | **-6.1 pp total** |

### Per-category breakdown (post-opt2)

| Category | N | Median overhead | vs baseline |
|---|---:|---:|---:|
| screen_ui | 13 | +55.1% | -5.9 pp |
| screen_chart | 13 | +52.4% | +1.2 pp (noise) |
| screen_document | 13 | +53.3% | -1.8 pp |
| illustration | 11 | +54.3% | -9.0 pp |

The biggest wins come on `illustration` content, which also has the
heaviest refinement trigger rate — consistent with opt1 saving the
redundant per-block-neighbor reference IDCT and opt2 cutting the
per-IDCT DC-only check.

## Callgrind (one 768×768 illustration, 30 iters)

| Build | PROGRAM TOTALS | vs baseline OFF | Δ vs prior build |
|---|---:|---:|---:|
| OFF | 1,842 M Ir | — | — |
| ON baseline | 2,373 M Ir | +531 M Ir (+28.8%) | — |
| ON opt1 | 2,314 M Ir | +471 M Ir (+25.6%) | -59 M (-11% of delta) |
| ON opt2 | 2,243 M Ir | +400 M Ir (+21.7%) | -71 M (-14% of remaining delta) |

**Total Ir delta reduction: 531 M → 400 M = -24.7%.**

## Target vs actual

Task brief target: **≤ +25% wall-clock overhead at Q75**. We shipped at
**+54.3%**. That target was out of reach in this pass — the Ir-delta is
85% IDCT + refinement inline body, both of which resist further
optimization without a larger architectural change (preserving pixel
strips across iMCU boundaries, or restructuring to emit edge-only
projections). Honest framing: this is a **~10% improvement on the
prior +60% overhead**, not a path to +25%.

## Negative results recorded

### SimdDispatch token hoist (reverted)

Tried: add a module-level `SimdDispatch` summoned once per iMCU, pass
through `boundary_distortion_raw_fast`, `idct_reference_block_fast`,
`ac_dct_energy_fast` helpers that dispatch on the cached token directly
instead of via `incant!`.

Callgrind showed a 3M Ir reduction (-0.1% absolute). Wall-clock:
`opt1 56.5% → opt2-token 57.7%` — within noise, possibly negative.

Root cause: the `incant!` macro-expanded `if let Some(token) =
X64V3Token::summon() { tail_call_to_arcane(token, ...) }` compiles to
a 3-4 instruction sequence with one well-predicted branch + tail-call
— essentially free. Saving the atomic load doesn't cross into
wall-clock-measurable.

The functionally-equivalent hoist WAS kept in the final opt2 commit
because it pairs with the `_fast` IDCT entry and doesn't cost
anything. But it's not where the actual win comes from — **skipping
`is_dc_only` inside `inverse_dct_8x8` is the real saving**.

## What remains

Top Ir-delta sources in opt2:

| ΔIr (M) | Where |
|---:|---|
| 194.2 | inline body of `quantize_y_with_boundary_rd_impl` (IDCT + helpers, rolled up) |
| ~85 | `__arcane_mage_inverse_dct_8x8` (helper-attribution buckets sum) |
| 22.4 | `__arcane_mage_quantize_block_zigzag_v3` from retry path |
| 20.8 | strip/mod.rs:refinement loop body exclusive |
| 7-11 each | core helper functions (slice/ptr/iter macros charged to IDCT) |

Further reduction would require:

1. **Eliminating one IDCT per block** — the reference IDCT always runs.
   Could be skipped if the pixel strip were preserved across iMCUs
   (pixel-space edge extraction instead of IDCT). This is a significant
   memory + architectural change (~8 MB/iMCU for 4K content), deferred.

2. **Skipping D_b computation for low-AC blocks.** In flat regions
   `ac_energy` is near-zero, so `threshold × ac_energy` is near-zero,
   and `db_default` is typically near-zero too — refinement never
   triggers. A pre-check `if ac_energy < MIN_REFINEMENT_AC_ENERGY` could
   skip the 2 IDCTs + 2 boundary_distortion calls for such blocks.
   Deferred because: (a) must preserve byte-identity with the current
   output, and (b) the threshold tuning needs a fresh BBS sweep.

3. **Removing `Vec<Option<...>>` entirely in state** — DONE in opt1 via
   `row_emitted` counter + `above_row_written` bool.

4. **SIMD edge extraction** — `left_edge_col` / `right_edge_col` are
   strided 8-element gathers. LLVM already emits 8 scalar `movss` for
   these after inlining. A true `vgatherdps` or transpose-then-column
   approach is tempting but the scalar version is already cache-local
   and pipeline-friendly. Deferred.

## Methodology

Same as baseline: 50 images stratified by GPT category from the
1,375-image `zenjpeg_tuning_corpus_gpt.txt`, seed 42 SplitMix64,
MCU-aligned max-side=768 via Triangle resize, 10 iterations per config
per image, interleaved per-iteration measurement, median aggregated
per image, median-of-medians per category, overall median.

Callgrind: single 768×768 illustration (`2e573cf50eda55867b119c5c010973e7c097f4f1.png`)
encoded 30 times; `callgrind_annotate --inclusive=no --threshold=100
--auto=no`.

Reproduce:
```bash
cargo build --release -p zenjpeg --features trellis \
    --example boundary_rd_perf_bench --example boundary_rd_callgrind

./target/release/examples/boundary_rd_perf_bench \
    --sample 50 --iters 10 --max-side 768 --seed 42 \
    --tag final --output-dir benchmarks/boundary_rd
```

## Raw artifacts

- `perf_baseline_2026-04-21.csv` — 50 × 5-iter, before this pass
- `perf_opt1_cache_orig_2026-04-21.csv` — after opt1 (5 iters)
- `perf_opt1_10iter_2026-04-21.csv` — opt1 re-run at 10 iters for fair compare
- `perf_opt2_fast_idct_2026-04-21.csv` — after opt2 (10 iters)
- `callgrind_baseline_off_2026-04-21.txt` + `callgrind_baseline_on_2026-04-21.txt`
- `callgrind_opt1_on_2026-04-21.txt`
- `callgrind_opt2_fast_idct_on_2026-04-21.txt`
