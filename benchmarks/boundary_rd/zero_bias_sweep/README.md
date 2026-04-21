# Zero-bias-shrink sweep (Task 6 of #102 rollup)

Date: 2026-04-21
Harness: `zenjpeg/examples/boundary_rd_zero_bias_sweep.rs`
Corpus manifest: `benchmarks/boundary_rd/sweep_corpus/manifest.tsv`

## Question

Does the newly-added `RetryPolicy::zero_bias_shrink` knob add signal
over the existing `aq_shrink` for boundary-RD's retry path? Is it
worth changing `BoundaryRdConfig::default()` to exploit it?

## Method

- **Grid**: `α × threshold × aq_shrink × zero_bias_shrink × max_retries`
  = 2×2×2×3×2 − 4 pure-identity cells = **40 configs**.
- **Qualities**: `{5, 15, 30, 45, 60, 75, 85, 95}` (8 levels).
- **Corpus**: 23 images across 6 classes — bilevel (5 gb82-sc),
  screencontent (4 gb82-sc), doc-text (6 Gutenberg/Brown-v-Board
  pages), palette-rich (4 pngsuite), mixed-vector (4 committed
  synthetic linearts).
- **Metrics**: BBS (block-boundary sensitivity) + SSIMULACRA2
  distortion.
- **Aggregation**: per-(class, q-range) BD-rate; composite score
  `-BD_BBS - 5 × max(0, BD_SSIM2)` (same weighting as the Phase-5
  low-Q sweep, see `../default_rationale_2026-04-21.md`).

Total encodes: 7360 candidate + 184 baseline = **7544 in 156 s**
on the Ryzen 9 7950X.

## Result: keep the default; keep the knob manual

`BoundaryRdConfig::default()` stays at `(α=2.0, threshold=0.02,
aq_shrink=0.5, zero_bias_shrink=1.0, max_retries=2, above=true)`.
The sweep does not reveal a strictly-dominant alternative.

### Per-cell best configs

From `best_per_class_per_q.csv`:

| class | q_range | winner | bd_bbs | bd_ssim2 |
|---|---|---|---|---|
| bilevel | low | a2_t0.10_aq0.5_**zb1.0**_r1 | −4.7 | +0.5 |
| bilevel | mid | a2_t0.10_aq0.5_**zb1.0**_r2 | −5.4 | −0.8 |
| bilevel | high | a2_t0.02_aq0.5_**zb0.5**_r1 | −5.9 | −1.3 |
| doc-text | low | a2_t0.02_aq0.5_**zb1.0**_r2 | −6.1 | +0.4 |
| doc-text | mid | a2_t0.10_aq0.5_**zb1.0**_r2 | −5.6 | +0.6 |
| doc-text | high | a2_t0.02_aq0.5_**zb0.5**_r2 | −23.4 | −8.9 |
| mixed-vector | low | a2_t0.02_aq0.5_**zb1.0**_r2 | −6.1 | +0.4 |
| mixed-vector | mid | a1_t0.10_aq0.5_**zb1.0**_r2 | −3.2 | 0.0 |
| mixed-vector | high | a2_t0.02_aq0.5_**zb0.3**_r1 | −6.3 | −0.6 |
| palette-rich | low | a2_t0.02_aq0.5_**zb1.0**_r1 | +0.3 | −0.5 |
| palette-rich | mid | a2_t0.02_aq0.5_**zb1.0**_r1 | +0.4 | +0.3 |
| palette-rich | high | a1_t0.10_aq0.5_**zb1.0**_r1 | +0.5 | +0.9 |
| screencontent | low | a1_t0.10_aq0.5_**zb1.0**_r1 | −5.0 | +0.7 |
| screencontent | mid | a2_t0.02_aq0.5_**zb1.0**_r1 | −4.2 | +1.1 |
| screencontent | high | a2_t0.02_aq0.5_**zb1.0**_r2 | −6.8 | −0.4 |

### Analysis

1. **Every winner has `aq_shrink=0.5`.** Pure zero-bias-shrink (i.e.
   `aq_shrink=1.0 + zero_bias_shrink<1.0`) never wins — it produces
   non-trivial BD_BBS gains (−4 to −8 %) but at a large SSIM2 cost
   (+4 to +5 %). The composite-score penalty wipes out the BBS win.

2. **`zero_bias_shrink<1.0` wins only at high Q, and only for
   line-art-dominated classes.** 3 of 15 cells:
   - `bilevel/high` (zb=0.5): +2.8 composite-score points over the
     current default.
   - `doc-text/high` (zb=0.5): +1.5 points (the large BD_BBS of −23
     is partly driven by one very-line-art Gutenberg page, not the
     default-change candidate).
   - `mixed-vector/high` (zb=0.3): +3.5 points.

3. **At low and mid Q, every winner has `zero_bias_shrink=1.0`.**
   At low Q, zero-bias-shrink preserves too many near-zero AC
   coefficients, inflating bytes faster than BBS improves.

4. **The two knobs are NOT collinear.** `aq_shrink` lowers the
   overall quant step (weakens quant globally); `zero_bias_shrink`
   only affects the zero-bias threshold (whether to zero-out
   below-threshold coefficients). At `aq_shrink=0.5`, pulling
   `zero_bias_shrink` below 1.0 adds additional AC coefficients
   on top of the aq-shrink effect — sometimes a win, sometimes
   a regression.

5. **palette-rich regresses for every On-config.** All `palette-rich`
   composite scores are negative — boundary-RD makes this class
   worse regardless of knobs. This is a pre-existing finding, not
   new. The feature is meant to be enabled per-class, not
   unconditionally (classifier deferred to #103).

### Why not bump the default

The current default `(α=2, t=0.02, aq_shrink=0.5, zb_shrink=1.0,
r=2, above=true)` is the winner in the `doc-text/low` and
`mixed-vector/low` cells and within 3 composite-score points of
every other zb_shrink=1.0 cell's winner. No single alternative
strictly dominates it:

- Switching to `zb_shrink=0.5` wins 3 high-Q cells but loses
  13-20 composite-score points at low-Q (large SSIM2 regression).
- Switching to `threshold=0.1` wins 4 low-to-mid cells but loses
  at doc-text/low and mixed-vector/low.
- Switching to `retries=1` wins the bilevel/low and bilevel/high
  cells by <1 point but loses at doc-text/high by 3.5 points.

The winner-per-cell variance means the right answer is per-class
tuning (tracked under #103), not another default bump. In the
meantime, the `zero_bias_shrink` knob stays as a manual-config
option for callers who already know their class.

### Recommended manual overrides

Until the per-class classifier lands in #103:

```rust
// Line-art / screen-content at mid-to-high Q: default is fine.
let cfg = BoundaryRdConfig::default();

// High-Q line-art or text documents (Q>=80):
let cfg = BoundaryRdConfig::default()
    .with_zero_bias_shrink(0.5);    // +1-3 composite pts on bilevel/high, doc-text/high

// Mid-to-high Q line-art vector content:
let cfg = BoundaryRdConfig::default()
    .with_zero_bias_shrink(0.3)
    .with_max_retries(1);           // mixed-vector/high winner

// palette-rich or screencontent: off (boundary-RD regresses).
```

### Overhead

`zero_bias_shrink=1.0` (default) is byte-identical to the pre-Task-3
retry path — verified by the unit test
`scaled_zero_bias_scale_one_is_identity` in `foundation::simd_types`.
Wall-clock overhead of the scaled kernel vs the unscaled kernel is
<1 % — one extra f32x8 splat + multiply per row — and the scaled
variant is only called on triggered blocks (a small fraction of the
total encode time).

## Files

- `grid.csv` — 7544 rows of per-(image, config, quality) measurements.
- `per_class_per_q.csv` — 600 rows of per-(class, q_range, config)
  BD-rate aggregates.
- `best_per_class_per_q.csv` — 15 rows of per-(class, q_range)
  composite-score winners (table above).

## Reproduction

```bash
cargo build --release -p zenjpeg --features "trellis decoder" --example boundary_rd_zero_bias_sweep
./target/release/examples/boundary_rd_zero_bias_sweep \
    --output benchmarks/boundary_rd/zero_bias_sweep
```

Runtime: ~3 minutes on a Ryzen 9 7950X. All corpus paths in the
manifest must be accessible (see `../sweep_corpus/manifest.tsv`).
