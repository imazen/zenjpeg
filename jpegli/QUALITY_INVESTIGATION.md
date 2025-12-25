# Quality Investigation Progress

## Issue
Rust produces ~17% smaller files than C++ at same Q value, but with worse DSSIM.
At same file SIZE, Rust actually has better quality.

## Root Cause Found: Quant Table Count

### C++ (cjpegli) behavior with `add_two_chroma_tables=true`:
- Table 0: Y component (BASE_QUANT_MATRIX_YCBCR[0..64])
- Table 1: Cb component (BASE_QUANT_MATRIX_YCBCR[64..128])
- Table 2: Cr component (BASE_QUANT_MATRIX_YCBCR[128..192])

### Current Rust behavior (INCORRECT):
- Table 0: Y component (BASE_QUANT_MATRIX_YCBCR[0..64])
- Table 1: Cb+Cr components (BASE_QUANT_MATRIX_YCBCR[64..128])

### Why this causes smaller files with worse quality:

Position 22 in base matrix:
- Cb matrix: **102.645** (extremely aggressive)
- Cr matrix: **7.886** (13x less aggressive)

By using Cb matrix for Cr component, Rust quantizes Cr much more aggressively,
zeroing out more coefficients -> smaller file but worse Cr channel quality.

## Verified Components (These MATCH C++):
- [x] `quality_to_distance()` formula - IDENTICAL
- [x] `distance_to_scale()` formula - IDENTICAL
- [x] `FREQUENCY_EXPONENT[64]` array - IDENTICAL
- [x] `GLOBAL_SCALE_YCBCR` (1.73966010) - IDENTICAL
- [x] `BASE_QUANT_MATRIX_YCBCR[192]` values - IDENTICAL
- [x] `ZERO_BIAS_OFFSET_YCBCR_AC[3]` - IDENTICAL
- [x] `ZERO_BIAS_MUL_YCBCR_LQ[192]` - IDENTICAL
- [x] `ZERO_BIAS_MUL_YCBCR_HQ[192]` - IDENTICAL

## Fix Required
Change encode.rs to generate 3 quant tables instead of 2:
1. Y quant table (component 0)
2. Cb quant table (component 1)
3. Cr quant table (component 2)

Update `write_quant_tables` to accept 3 tables.
Update frame header to reference correct quant tables per component.
