# Context Handoff: EOB Optimization Investigation Results

Branch: `feat/mozjpeg-mimic-tests`
Date: 2026-02-01

## Summary: EOB Optimization is BROKEN

The standalone EOB optimization approach (`TrellisConfig::eob_optimization(true)`) **does not work** and has been disabled.

### Test Results

Using TRUE A/B test (mozjpeg-mimic mode, no AQ, no deringing):

| Metric | Without EOB | With EOB | Change |
|--------|-------------|----------|--------|
| File size | 73856 bytes | 16613 bytes | **-77%** |
| DSSIM | 0.0027 | 0.1098 | **40x worse** |

The "77% smaller" was achieved by destroying the image.

---

## Root Cause: Missing Lambda Weighting

### How mozjpeg Computes Zero-Block Cost (CORRECT)

In `jcdctmgr.c` line 1148, mozjpeg computes the cost of zeroing each coefficient **during** trellis:

```c
// x = abs(src[bi][z]) — ORIGINAL unquantized coefficient (100-1000+)
// lambda — R-D tradeoff factor (~0.001-0.01)
// lambda_tbl[z] — per-coefficient weight from quant table

accumulated_zero_dist[i] = x * x * lambda * lambda_tbl[z] + accumulated_zero_dist[i-1];
```

Then at line 1239:
```c
cost_all_zeros = accumulated_zero_dist[Se];  // Sum over all AC coefficients
```

**Key insight**: The zero-block cost is `lambda * Σ(original_coef²)`, which is **in the same units** as the encoding cost (both are R-D weighted).

### How Our Implementation Computes Zero-Block Cost (BROKEN)

In `trellis/eob.rs`, we compute the cost **after** trellis on already-quantized blocks:

```rust
// coef is QUANTIZED value (small integer, typically 1-10)
zero_block_cost += (coef as f32) * (coef as f32);
```

**What's missing:**
- Original coefficients are gone (we only have quantized values)
- No lambda weighting
- No per-coefficient weighting from quant table

### Numerical Comparison

| Factor | mozjpeg | Our Implementation |
|--------|---------|-------------------|
| Coefficient source | Original (100-1000+) | Quantized (1-10) |
| Lambda weighting | Yes (~0.001-0.01) | **No** |
| Per-coef weighting | Yes (1/quant²) | **No** |
| Typical zero cost | 1000-5000 | 1-100 |
| Units | R-D weighted bits | Raw squared integers |

### Why This Breaks Everything

Consider a block with a single AC coefficient:
- Original value: 500
- Quantization step: 50
- Quantized value: 10
- Lambda: 0.005

**mozjpeg's zero cost:**
```
cost = 500² × 0.005 × (1/50²) = 250000 × 0.005 × 0.0004 = 0.5
```

**mozjpeg's encoding cost:**
```
cost ≈ 4 bits (for value=10 with run=0)
```

Decision: 0.5 < 4, so **keep** the coefficient (zeroing costs more in R-D terms).

**Our zero cost:**
```
cost = 10² = 100
```

**Our encoding cost:**
```
cost ≈ 4 bits
```

Decision: 4 < 100, so **zero** the coefficient.

But wait—these aren't even in the same units! We're comparing bits to squared-integers. The algorithm sees "4 < 100" and always chooses to encode, right?

Actually no—the comparison goes the other way in `optimize_eob_runs`. The algorithm adds up `zero_block_cost` across blocks and compares to `best_cost` (encoding cost). Since our `zero_block_cost` values are tiny (sum of squares of 1-10), and encoding costs are in bits (10-50 per block), the algorithm decides that zeroing entire runs of blocks is "cheaper."

The fundamental problem: **incompatible units make the comparison meaningless**.

---

## Why EOB Helps mozjpeg (When Implemented Correctly)

With proper lambda weighting, the algorithm makes sensible tradeoffs:

1. **High-detail blocks**: Large original coefficients → high zero cost → keep them
2. **Low-detail blocks**: Small coefficients → low zero cost → candidates for zeroing
3. **Trailing coefficients**: High-frequency coefficients with small values may be worth zeroing if it creates longer EOBRUN codes

The savings are typically **0.5-1%** because:
- Most blocks have significant coefficients that shouldn't be zeroed
- The optimization only helps at block boundaries where EOBRUN coding applies
- Lambda ensures quality loss is always weighed against size savings

---

## Fix Required

To properly implement EOB optimization, it **must** be integrated into trellis:

### Option 1: Compute During Trellis (Recommended)

In `HybridQuantContext::quantize_row_trellis()`:

```rust
// During AC coefficient trellis loop:
let lambda = self.compute_lambda(block_norm);
let zero_dist = original_coef * original_coef * lambda * lambda_tbl[z];
accumulated_zero_dist[i] = accumulated_zero_dist[i-1] + zero_dist;

// After trellis:
let cost_all_zeros = accumulated_zero_dist[se];
```

Store `cost_all_zeros` in the output and pass to EOB optimization.

### Option 2: Store Original Coefficients

Keep original (unquantized) coefficients until after EOB optimization, then discard. This increases memory usage but allows standalone EOB.

### Why Post-Trellis EOB Cannot Work

After trellis quantization:
- Original coefficients are **destroyed** (replaced with quantized values)
- Lambda was computed per-block and **not stored**
- Quantization table weights were applied and **not recoverable**

There is no way to reconstruct proper R-D costs from quantized coefficients alone.

---

## Current State

- `TrellisConfig::eob_optimization(true)` is accepted but **has no effect**
- The `apply_eob_optimization()` function in `streaming.rs` is disabled
- The API is preserved for future implementation
- Detailed warning comments explain the issue

## Files Changed

| File | Change |
|------|--------|
| `encode/streaming.rs` | Disabled `apply_eob_optimization()` with detailed comments |
| `encode/mozjpeg_compat.rs` | `eob_optimization()` method exists but is no-op |
| `trellis/eob.rs` | Functions exist but `estimate_block_eob_info` is fundamentally broken |
| `examples/eob_mozjpeg_mimic.rs` | TRUE A/B test harness (now shows 0% delta) |

---

## What Still Works

zenjpeg with trellis **already beats** C mozjpeg with trellis:

| Quality | C mozjpeg+trellis | zenjpeg+trellis | Difference |
|---------|-------------------|-----------------|------------|
| Q50 | 45617 | 45333 | **-0.6%** |
| Q75 | 73994 | 73856 | **-0.2%** |
| Q90 | 130585 | 130459 | **-0.1%** |

**No EOB optimization needed** — zenjpeg is already winning without it.

---

## Recommendation

**Do not pursue EOB optimization further.**

Reasons:
1. Expected gain is only ~0.5-1% (mozjpeg's own benefit)
2. zenjpeg already beats mozjpeg by 0.1-0.6% without EOB
3. Proper implementation requires significant trellis refactoring
4. The complexity/benefit ratio is poor

If EOB is ever needed:
1. Store `accumulated_zero_dist` during trellis quantization
2. Pass lambda-weighted costs to EOB optimization
3. Apply as true R-D optimization with comparable units

---

## Test Command

```bash
cargo run --release -p zenjpeg --features mozjpeg-tables --example eob_mozjpeg_mimic
```

## Reference: mozjpeg Source Locations

- `jcdctmgr.c:1148` — `accumulated_zero_dist` computation
- `jcdctmgr.c:1239` — `cost_all_zeros = accumulated_zero_dist[Se]`
- `jcdctmgr.c:1274-1308` — EOB run optimization loop
- `jpegint.h:106` — `trellis_eob_opt` flag definition
