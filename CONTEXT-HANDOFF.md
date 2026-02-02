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

## Root Cause

The `estimate_block_eob_info` function in `trellis/eob.rs` compares:
- **Encoding cost** (in bits): the cost to Huffman-encode the non-zero coefficients
- **Zeroing cost** (sum of squared coefficients): `coef^2`

These quantities are in **incompatible units**. Without a lambda multiplier, the algorithm always decides that zeroing is "cheaper" because small quantized coefficients (1-10) have small squared values.

### What mozjpeg Does Correctly

In mozjpeg's `jcdctmgr.c`, the `cost_all_zeros` is computed **during** trellis quantization where:
- The original (unquantized) coefficients are available
- The lambda factor is available
- The cost is computed as `lambda * (original_coef)^2`

This gives a properly weighted rate-distortion tradeoff.

### Why the Rust Implementation Fails

The Rust `estimate_block_eob_info` function is called **after** trellis quantization on already-quantized blocks. At this point:
- Original coefficients are lost
- Lambda is not available
- We only have small integer coefficients (1-10)

Computing `coef^2` on these small values gives tiny "distortion costs" that are always smaller than encoding costs.

## Fix Required

To properly implement EOB optimization:

1. **Integrate into trellis pass**: Compute `cost_all_zeros` during `HybridQuantContext::quantize_row_trellis()` where lambda is available
2. **Store the cost**: Add a `cost_all_zeros: f32` field to `BlockEobInfo` populated during trellis
3. **Use during EOB optimization**: Pass the trellis-computed costs to `optimize_eob_runs`

This is a significant refactor - EOB cannot be a standalone post-processing step.

## Current State

- `TrellisConfig::eob_optimization(true)` is accepted but **has no effect**
- The `apply_eob_optimization()` function in `streaming.rs` is disabled with documentation
- The API is preserved for future implementation

## Files Changed

| File | Change |
|------|--------|
| `encode/streaming.rs` | Disabled `apply_eob_optimization()` with warning comments |
| `encode/mozjpeg_compat.rs` | `eob_optimization()` method exists but has no effect |
| `trellis/eob.rs` | Functions exist but `estimate_block_eob_info` is broken |
| `examples/eob_mozjpeg_mimic.rs` | TRUE A/B test harness (shows 0% delta now) |

## What Still Works

The mozjpeg-mimic mode comparison shows zenjpeg is **already competitive** with C mozjpeg:

| Quality | C mozjpeg+trellis | zenjpeg+trellis | Difference |
|---------|-------------------|-----------------|------------|
| Q50 | 45617 | 45333 | **-0.6%** |
| Q75 | 73994 | 73856 | **-0.2%** |
| Q90 | 130585 | 130459 | **-0.1%** |

zenjpeg with trellis is already **0.1-0.6% smaller** than C mozjpeg with trellis (no EOB needed).

## Recommendation

**Do not pursue EOB optimization further.** The gain in mozjpeg is minimal (~0.5-1%) and the implementation complexity is high. zenjpeg already beats C mozjpeg without it.

If EOB is ever needed:
1. Port the full trellis+EOB integration from mozjpeg
2. Store `cost_all_zeros` during trellis quantization
3. Apply EOB as a true rate-distortion optimization

## Test Command

```bash
cargo run --release -p zenjpeg --features mozjpeg-tables --example eob_mozjpeg_mimic
```
