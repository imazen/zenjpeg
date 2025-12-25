# Quality Investigation - RESOLVED

## Summary
Fixed: Rust now produces identical DSSIM to C++ at same Q value.

## Results After Fix

| Q | Rust Size | C++ Size | Rust DSSIM | C++ DSSIM | Size Δ |
|---|-----------|----------|------------|-----------|--------|
| 70 | 12,204 | 11,876 | 0.002846 | 0.002846 | +2.8% |
| 80 | 14,640 | 14,414 | 0.001837 | 0.001837 | +1.6% |
| 85 | 16,594 | 16,458 | 0.001225 | 0.001225 | +0.8% |
| 95 | 25,598 | 25,415 | 0.000263 | 0.000263 | +0.7% |

**DSSIM is now IDENTICAL between Rust and C++ at all tested quality levels.**

## Root Cause

Rust was using 2 quant tables (Y and Cb), applying the Cb matrix to both
chroma components. C++ cjpegli uses 3 separate tables.

Key difference at position 22:
- Cb matrix: 102.645 (very aggressive quantization)
- Cr matrix: 7.886 (much less aggressive)

Using Cb matrix for Cr caused excessive zeroing of Cr coefficients.

## Fix Applied

Commit: `fix: Use 3 separate quant tables for YCbCr like C++ cjpegli`

- `write_quant_tables` now writes 3 DQT tables (Y, Cb, Cr)
- `quantize_all_blocks` takes separate cb_quant and cr_quant parameters
- Frame header assigns Cr component to quant table 2 (was table 1)
- All encode paths updated (baseline, progressive standard, progressive optimized)

## Verified Components (All Match C++)

- [x] `quality_to_distance()` formula
- [x] `distance_to_scale()` formula
- [x] `FREQUENCY_EXPONENT[64]` array
- [x] `GLOBAL_SCALE_YCBCR` (1.73966010)
- [x] `BASE_QUANT_MATRIX_YCBCR[192]` values
- [x] `ZERO_BIAS_OFFSET_YCBCR_AC[3]`
- [x] `ZERO_BIAS_MUL_YCBCR_LQ/HQ[192]`
