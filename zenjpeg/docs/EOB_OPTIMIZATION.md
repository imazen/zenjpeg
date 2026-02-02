# EOB Run Optimization: Investigation Notes

**Status**: BROKEN / DISABLED
**Date**: 2026-02-01
**Branch**: `feat/mozjpeg-mimic-tests`

This document records the investigation into EOB (end-of-block) run optimization and why the standalone implementation does not work.

---

## What is EOB Optimization?

EOB run optimization is a cross-block optimization for JPEG encoding. It finds opportunities to zero out trailing AC coefficients across multiple consecutive blocks to create longer EOBRUN codes, which compress more efficiently in progressive JPEG.

In mozjpeg, this provides **~0.5-1% file size reduction** with negligible quality impact.

---

## Implementation Attempt

We attempted to implement EOB optimization as a standalone post-processing step:

1. Run trellis quantization (produces quantized coefficients)
2. Apply EOB optimization to the quantized blocks
3. Encode the modified blocks

**Files created:**
- `trellis/eob.rs` — `estimate_block_eob_info()` and `optimize_eob_runs()`
- `encode/streaming.rs` — `apply_eob_optimization()` (now disabled)

**API added:**
- `TrellisConfig::eob_optimization(bool)` — accepted but has no effect

---

## Test Results

Using TRUE A/B test (mozjpeg-mimic mode, no AQ, no deringing):

| Metric | Without EOB | With EOB | Change |
|--------|-------------|----------|--------|
| File size | 73,856 bytes | 16,613 bytes | **-77%** |
| DSSIM | 0.0027 | 0.1098 | **40× worse** |

The algorithm destroyed the image by zeroing almost all coefficients.

---

## Root Cause: Unit Mismatch

### mozjpeg's Computation (Correct)

In `jcdctmgr.c:1148`, mozjpeg computes zero-block cost **during** trellis:

```c
// x = abs(src[bi][z]) — ORIGINAL unquantized coefficient (100-1000+)
// lambda — R-D tradeoff factor (~0.001-0.01)
// lambda_tbl[z] — per-coefficient weight (1/quant²)

accumulated_zero_dist[i] = x * x * lambda * lambda_tbl[z] + accumulated_zero_dist[i-1];
```

At line 1239:
```c
cost_all_zeros = accumulated_zero_dist[Se];
```

The zero-block cost is **lambda-weighted distortion**, in the same units as encoding cost.

### Our Computation (Broken)

In `trellis/eob.rs`, we compute cost **after** trellis:

```rust
// coef is QUANTIZED value (1-10)
zero_block_cost += (coef as f32) * (coef as f32);
```

**What's missing:**
- Original coefficients (destroyed by quantization)
- Lambda weighting
- Per-coefficient weighting

### Numerical Comparison

| Factor | mozjpeg | Our Implementation |
|--------|---------|-------------------|
| Coefficient source | Original (100-1000+) | Quantized (1-10) |
| Lambda weighting | Yes (~0.001-0.01) | **No** |
| Per-coef weighting | Yes (1/quant²) | **No** |
| Typical zero cost | 1000-5000 | 1-100 |
| Units | R-D weighted bits | Raw squared integers |

### Example

Block with one AC coefficient:
- Original value: 500
- Quantization step: 50
- Quantized value: 10
- Lambda: 0.005

**mozjpeg:**
```
zero_cost = 500² × 0.005 × (1/50²) = 0.5
encode_cost ≈ 4 bits
Decision: 0.5 < 4 → KEEP coefficient
```

**Our implementation:**
```
zero_cost = 10² = 100
encode_cost ≈ 4 bits
(units incompatible, algorithm misbehaves)
```

The algorithm compares `accumulated_zero_block_cost` (tiny values ~100) against `best_cost` (encoding cost ~10-50 bits) and decides zeroing entire block runs is "cheaper."

---

## Why Post-Trellis EOB Cannot Work

After trellis quantization:
1. **Original coefficients destroyed** — replaced with small integers
2. **Lambda not stored** — computed per-block and discarded
3. **Quant weights not recoverable** — baked into quantized values

There is no way to reconstruct proper R-D costs from quantized coefficients alone.

---

## Correct Implementation (If Ever Needed)

EOB optimization **must** be integrated into trellis quantization:

```rust
// During trellis AC coefficient loop:
let lambda = self.compute_lambda(block_norm);
let zero_dist = original_coef * original_coef * lambda * lambda_tbl[z];
accumulated_zero_dist[i] = accumulated_zero_dist[i-1] + zero_dist;

// After trellis, store for EOB:
block_eob_info.cost_all_zeros = accumulated_zero_dist[se];
```

Then pass `cost_all_zeros` to `optimize_eob_runs()`.

---

## Current State

- `TrellisConfig::eob_optimization(true)` — accepted, **no effect**
- `apply_eob_optimization()` — disabled with warning comments
- `trellis/eob.rs` — functions exist but are broken
- API preserved for future implementation

---

## Why This Doesn't Matter

zenjpeg with trellis **already beats** C mozjpeg with trellis:

| Quality | C mozjpeg+trellis | zenjpeg+trellis | Difference |
|---------|-------------------|-----------------|------------|
| Q50 | 45,617 | 45,333 | **-0.6%** |
| Q75 | 73,994 | 73,856 | **-0.2%** |
| Q90 | 130,585 | 130,459 | **-0.1%** |

EOB optimization is not needed — zenjpeg wins without it.

---

## Recommendation

**Do not pursue EOB optimization.**

1. Expected gain: only ~0.5-1%
2. zenjpeg already beats mozjpeg by 0.1-0.6%
3. Proper implementation requires trellis refactoring
4. Complexity/benefit ratio is poor

---

## References

**mozjpeg source:**
- `jcdctmgr.c:1148` — `accumulated_zero_dist` computation
- `jcdctmgr.c:1239` — `cost_all_zeros = accumulated_zero_dist[Se]`
- `jcdctmgr.c:1274-1308` — EOB run optimization loop
- `jpegint.h:106` — `trellis_eob_opt` flag

**Test command:**
```bash
cargo run --release -p zenjpeg --features mozjpeg-tables --example eob_mozjpeg_mimic
```
