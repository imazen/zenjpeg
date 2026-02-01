# XYB Size Difference Investigation Plan

## Current State

XYB mode produces **5-11% larger files** than C++ jpegli at equivalent perceptual quality.

- Q70: +6-11%
- Q80: +5-9%
- Q90: +5-7%

**What's confirmed identical:**
- AQ maps (100% match with distance-based encoding)
- Quantization table values (verified)
- XYB color conversion constants
- Subsampling structure (R:2×2 G:2×2 B:1×1)

**What's confirmed different:**
- DCT coefficients differ by ±1 (normal SIMD float rounding)
- File sizes differ by 5-11%

## Hypothesis Priority

1. **Huffman table efficiency** - Different coefficient distributions → different optimal tables
2. **Zero-bias threshold differences** - Might be quantizing fewer coefficients to zero
3. **DC coefficient prediction** - Could affect entropy encoding efficiency
4. **Coefficient distribution** - More non-zero coefficients = larger files

## Phase 1: Coefficient Analysis (Quantitative)

### 1.1 Count Non-Zero Coefficients

Compare total non-zero coefficient counts between Rust and C++:
```bash
# Per-component breakdown (X, Y, B channels)
# Rust should have SAME or FEWER non-zeros if AQ is working
```

**Expected outcome:** If Rust has more non-zero coefficients, zero-bias thresholds may be too low.

### 1.2 DC Coefficient Distribution

Compare DC coefficient values and prediction residuals:
- Extract DC values for each component
- Compare prediction residual magnitudes
- Check if DC differences accumulate

**Tool:** `xyb_dc_debug` example (exists)

### 1.3 AC Coefficient Histogram

For each component, histogram the AC coefficient magnitudes:
- Positions 1-63 in zigzag order
- Compare distributions between Rust and C++
- Look for systematic shifts

**New tool needed:** `xyb_coeff_histogram`

## Phase 2: Huffman Analysis

### 2.1 Compare Huffman Tables

Extract and compare:
- DC Huffman tables (3 tables for XYB: X, Y, B)
- AC Huffman tables (3 tables for XYB: X, Y, B)
- Code lengths for each symbol

**Tool:** `jpeg_inspect --validate` can decode, but need coefficient extraction

### 2.2 Entropy Calculation

Calculate theoretical minimum bits for each component's data:
```
H(X) = -Σ p(x) log2(p(x))
```

Compare actual bits used vs theoretical minimum.

## Phase 3: Zero-Bias Investigation

### 3.1 Verify Zero-Bias Constants

Check that XYB zero-bias matches C++:
```rust
// Rust (quant/mod.rs:167-175)
pub const ZERO_BIAS_MUL_XYB: f32 = 0.5;
pub const ZERO_BIAS_OFFSET_XYB: f32 = 0.5;
```

Compare with C++ jpegli:
```cpp
// lib/jpegli/quant.cc - find XYB zero-bias values
```

### 3.2 Zero-Bias Application

Trace the zero-bias application in quantization:
1. Coefficient value after DCT
2. Threshold = offset + mul × aq_strength
3. If |coeff| < threshold → quantize to 0

Verify formula matches C++ exactly.

## Phase 4: Scan Data Comparison

### 4.1 Extract Raw Scan Data

Strip markers, compare raw entropy-coded data:
- SOI to SOS: should be nearly identical (quant tables, Huffman tables)
- SOS to EOI: entropy-coded coefficients

**Size breakdown expected:**
- Headers/markers: ~500-1000 bytes (should match)
- Scan data: remaining (where the 5-11% difference is)

### 4.2 Progressive vs Baseline

If using progressive:
- Compare scan ordering
- Compare coefficient ranges per scan
- DC scans vs AC scans

## Phase 5: Component-Specific Analysis

### 5.1 Isolate Each Channel

Encode synthetic single-channel images:
- Uniform X channel, varying Y/B
- etc.

Identify which channel(s) contribute most to size difference.

### 5.2 Edge Block Analysis

The CLAUDE.md notes "block patterns in R/B channels" in visual diffs.
- Compare edge block handling
- Verify padding/replication behavior

## Diagnostic Commands

```bash
# Existing tools
cargo run --release --example xyb_parity_test
cargo run --release --example xyb_cpp_comparison
cargo run --release --example xyb_dc_debug
cargo run --release --example compare_xyb_constants

# Quick size comparison
just xyb-diff ~/work/codec-eval/codec-corpus/kodak/1.png

# C++ reference encoding
./internal/jpegli-cpp/build/tools/cjpegli -x -q 90 input.png output_cpp.jpg
```

## Success Criteria

Investigation complete when we can explain:
1. **What** exactly differs (coefficient counts, Huffman efficiency, etc.)
2. **Why** it differs (different algorithm, constants, edge cases)
3. **Impact** of each factor on file size
4. **Action** - fix vs accept as implementation difference

## Files to Examine

### Rust
- `zenjpeg/src/quant/mod.rs` - Quantization, zero-bias
- `zenjpeg/src/encode/strip/mod.rs` - Strip processing, quantize_pending_imcu
- `zenjpeg/src/foundation/simd_types.rs:346` - Zero-bias threshold formula
- `zenjpeg/src/huffman/` - Huffman table generation

### C++ Reference
- `internal/jpegli-cpp/lib/jpegli/quant.cc` - kGlobalScaleXYB, kBaseQuantMatrixXYB
- `internal/jpegli-cpp/lib/jpegli/encode.cc` - XYB sampling factors
- `internal/jpegli-cpp/lib/jpegli/adaptive_quantization.cc` - AQ for XYB

## Next Steps

1. Start with Phase 1.1 - count non-zero coefficients
2. If coefficient counts differ significantly, investigate zero-bias
3. If coefficient counts are similar, investigate Huffman efficiency
4. Document findings in this file as we go
