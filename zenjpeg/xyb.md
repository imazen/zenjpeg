# XYB Quality Gap Investigation

## Current Issue
Rust XYB mode produces ~-48 mean diff in the B channel vs C++ jpegli when decoded by djpegli.

**Test output (q90, kodak/1.png):**
```
Per-channel differences (Rust - C++):
  R (X): mean=-0.01, abs=0.22, max=16.0   ✅ Essentially identical
  G (Y): mean=+0.00, abs=1.13, max=11.0   ✅ Essentially identical
  B (B): mean=-47.94, abs=48.87, max=110.0  ❌ HUGE DIFFERENCE
```

File size: Rust +5.9% larger than C++
Quality vs original: Rust is ~10 SSIMULACRA2 points worse

## Verified Components (NOT the issue)

### 0. Scaled XYB Values ✅ (NEW FINDING)
- **Test:** `cargo test --features ffi-tests --test xyb_cpp_comparison`
- **Result:** PASS - **ALL scaled values match C++ exactly**
- **Evidence:**
  ```
  (  0,  0,255) -> Rust: (0.353816, 0.329026, 1.000000), C++: (0.353816, 0.329026, 1.000000)
  (128,128,128) -> Rust: (0.353816, 0.529286, 0.417152), C++: (0.353816, 0.529286, 0.417152)
  ```
- **Conclusion:** The color conversion and scaling pipeline is correct!

### 1. Raw XYB Conversion ✅
- **Test:** `cargo test --features ffi-tests --test xyb_linear_cpp_parity`
- **Result:** PASS - both scalar and SIMD match C++ within 1e-7
- **Details:** `xyb_ulp_parity` example shows max error 3.58e-7, all errors ≤1e-6

### 2. Opsin Absorbance Matrix ✅
- **Rust values match C++ exactly:**
  - kM00=0.30, kM01=0.622, kM02=0.078
  - kM10=0.23, kM11=0.692, kM12=0.078
  - kM20=0.24342269, kM21=0.20476744, kM22=0.55180987
- **Bias:** 0.0037930733 (same for all channels)

### 3. Scaling Constants ✅
- **Scale:** [22.995789, 1.183, 1.5021414] (matches C++)
- **Offset:** [0.015386134, 0.0, 0.27770459] (matches C++)

### 4. Scaling Formula ✅
- **Rust:** `scaled_b = (b_xyb - y_xyb + offset_b) * scale_b`
- **C++:** `row2[x] = (row2[x] - row1[x] + kScaledXYBOffset[2]) * kScaledXYBScale[2]`
- **Identical formula**

### 5. sRGB→Linear Conversion ✅
- LUT matches exact sRGB formula within f32 precision
- C++ polynomial approximation differs by max 0.00005% (negligible)

### 6. Cube Root Precision ✅
- Scalar `cbrtf_fast`: 1-2 ULP error
- SIMD `cbrtf_x8` (f64 Newton): 0 ULP error
- Both acceptable precision

### 7. XYB Quantization Matrix ✅
- 192-element matrix (X/Y/B channels) matches C++ exactly
- Channel 2 (B) values verified

### 8. Component IDs ✅
- XYB uses 'R'=82, 'G'=71, 'B'=66 (matches C++)

## Remaining Suspects

### 1. Zero Bias Configuration
- XYB uses `mul=0.5, offset=0.5` for all coefficients
- Need to verify this matches C++ for XYB mode
- **However:** if this were wrong, X and Y would also be affected

### 2. Quantization Rounding
- How DCT coefficients are rounded during quantization
- Check if C++ uses different rounding for B channel

### 3. ICC Profile Differences
- Both encode XYB ICC profile, but are they identical?
- Decoder uses ICC profile for inverse transform

### 4. B Channel Subsampling
- `.xyb()` uses `BQuarter` subsampling
- Need to verify this is handled correctly vs C++

### 5. Encoding Pipeline Issues
- Something in the DCT → quantize → Huffman path
- Check if B channel uses different Huffman tables

## Next Steps

1. [ ] Compare actual scaled XYB values (after scale_xyb) between Rust and C++ FFI
2. [ ] Compare ICC profiles embedded in both JPEGs
3. [ ] Compare quantization tables in output JPEGs
4. [ ] Check zero-bias configuration for XYB in C++
5. [ ] Dump DCT coefficients for B channel and compare

## Commands

```bash
# Run XYB comparison
cargo run --release --example xyb_rust_vs_cpp_ssim2

# Run parity test with FFI
cargo run --release --example xyb_ulp_parity --features ffi-tests

# Run XYB parity test
cargo run --release --example xyb_parity_test

# Test XYB linear conversion parity
cargo test --features ffi-tests --test xyb_linear_cpp_parity
```
