# Size Parity Investigation Plan

## Current Status (2025-01-04)

After fixing AC refinement encoding for djpeg compatibility, we have excellent parity in most configurations but some size gaps remain:

### Configurations with Perfect Parity (0.0% diff)
- YCbCr Progressive + Optimized Huffman (all sizes)
- YCbCr Baseline + Fixed Huffman (all sizes)

### Configurations with Size Gaps

| Config | Size Diff | Priority |
|--------|-----------|----------|
| YCbCr Baseline + Opt Huffman | +1.9% to +2.9% | Medium |
| XYB Baseline + Fixed Huffman | +14% to +15% (small images) | High |
| XYB Baseline + Opt Huffman | +0.0% to +1.1% | Low |
| XYB Progressive + Opt Huffman | +0.2% to +5.7% | Medium |

---

## Investigation 1: YCbCr Baseline Optimized Huffman (+1.9% to +2.9%)

### Hypothesis
The Huffman optimization algorithm produces slightly different code assignments than C++ jpegli, resulting in marginally larger files.

### Investigation Steps

1. **Compare Huffman table contents byte-by-byte**
   ```bash
   cargo run --release --example dump_huffman_codes
   ```
   - Extract DHT markers from both Rust and C++ output
   - Compare symbol counts per code length
   - Compare actual code assignments

2. **Compare histogram building**
   - Add instrumentation to `huffman_opt.rs` to dump symbol histograms
   - Compare with C++ histograms from instrumented build
   - Check if frequency counts match exactly

3. **Compare tree building algorithm**
   - Rust uses package-merge algorithm (matches C++)
   - Verify length-limited code generation produces same lengths
   - Check tie-breaking behavior when frequencies are equal

4. **Compare code assignment**
   - After tree building, codes are assigned by length
   - Check if symbol ordering within same length differs

### Files to Examine
- `jpegli-rs/src/huffman_opt.rs` - Rust optimization
- `jpegli-cpp/lib/jpegli/huffman.cc` - C++ reference
- `jpegli-rs/tests/huffman_cpp_comparison.rs` - Existing comparison tests

### Expected Outcome
Identify specific differences in Huffman table generation that cause the +2-3% gap.

---

## Investigation 2: XYB Baseline Fixed Huffman (+14-15% on small images)

### Hypothesis
XYB mode uses different quantization or coefficient distribution that interacts poorly with fixed Huffman tables designed for YCbCr.

### Investigation Steps

1. **Compare quantization tables**
   ```bash
   cargo run --release --example compare_cpp_quant -- --xyb
   ```
   - XYB uses different quant matrices than YCbCr
   - Verify Rust XYB quant tables match C++ exactly

2. **Compare coefficient distributions**
   - Dump DCT coefficient histograms for XYB mode
   - Compare AC/DC coefficient magnitudes between Rust and C++
   - Check if XYB scaling (`scale_xyb()`) matches C++

3. **Check XYB color conversion precision**
   - Compare intermediate values in XYB pipeline:
     - `srgb_to_linear()` output
     - `linear_rgb_to_xyb()` output
     - `scale_xyb()` output
   - Small differences accumulate across all pixels

4. **Verify ICC profile handling**
   - XYB embeds a 720-byte ICC profile
   - Check if profile bytes match exactly
   - Verify APP2 marker structure is identical

5. **Compare scan data sizes**
   - Break down file size by component:
     - Headers (SOI, DQT, SOF, DHT, APP2)
     - Scan data per component
   - Identify which component(s) contribute to bloat

### Files to Examine
- `jpegli-rs/src/xyb.rs` - XYB color conversion
- `jpegli-rs/src/quant.rs` - Quantization tables
- `jpegli-rs/src/icc.rs` - ICC profile embedding
- `jpegli-cpp/lib/jxl/enc_xyb.cc` - C++ XYB reference

### Expected Outcome
Identify whether the gap is from:
- Color conversion precision
- Quantization table differences
- Fixed Huffman table mismatch with XYB coefficient distribution

---

## Investigation 3: XYB Progressive (+0.2% to +5.7%)

### Hypothesis
Progressive XYB combines issues from both XYB color conversion and progressive scan organization.

### Investigation Steps

1. **Compare scan scripts**
   - Verify XYB progressive uses same scan structure as C++
   - Check spectral selection ranges (Ss, Se)
   - Check successive approximation values (Ah, Al)

2. **Compare per-scan sizes**
   ```bash
   cargo run --release --example analyze_scan_data -- --xyb
   ```
   - Break down size contribution per scan
   - Identify which scans are bloated

3. **Check DC coefficient handling**
   - XYB has 3 non-interleaved DC scans (vs 1 interleaved for YCbCr)
   - Verify DC prediction matches C++

4. **Verify AC refinement in XYB context**
   - Recent fix corrected tokenization order
   - Check if XYB-specific coefficient patterns expose edge cases

### Files to Examine
- `jpegli-rs/src/scan_script.rs` - Progressive scan organization
- `jpegli-rs/src/encode.rs` - XYB progressive encoding path

### Expected Outcome
Determine if XYB progressive gap is from:
- XYB-specific issues (Investigation 2)
- Progressive-specific issues
- Combination of both

---

## Tools to Create

### 1. `examples/compare_xyb_pipeline.rs`
Compare XYB color conversion step-by-step:
```rust
// For each pixel:
// 1. Compare srgb_to_linear output
// 2. Compare linear_rgb_to_xyb output
// 3. Compare scale_xyb output
// Report max/mean differences at each stage
```

### 2. `examples/compare_coefficient_histograms.rs`
Compare DCT coefficient distributions:
```rust
// Encode same image with Rust and C++
// Extract and compare:
// - DC coefficient histogram per component
// - AC coefficient histogram per component per frequency
// - Zero count per component
```

### 3. `examples/compare_huffman_tables_detailed.rs`
Detailed Huffman table comparison:
```rust
// For each table (DC0, DC1, AC0, AC1, etc.):
// - Symbol counts per length
// - Actual code assignments
// - Encoding efficiency (bits per symbol)
```

---

## Success Criteria

| Config | Current Gap | Target Gap |
|--------|-------------|------------|
| YCbCr Baseline Opt | +2-3% | <1% |
| XYB Baseline Fixed | +14-15% | <5% |
| XYB Baseline Opt | +0-1% | <1% |
| XYB Progressive Opt | +0.2-5.7% | <2% |

---

## Priority Order

1. **YCbCr Baseline Opt** - Most impactful, likely simple Huffman difference
2. **XYB Baseline Fixed** - Large gap on small images, likely quant/color issue
3. **XYB Progressive** - May be resolved by fixing #1 and #2

---

## Notes

- Progressive YCbCr with optimized Huffman already has 0.0% gap - this is our reference for "correct" behavior
- Fixed Huffman YCbCr baseline also has 0.0% gap - quantization and DCT are correct
- The gaps appear specifically when Huffman optimization is involved (YCbCr) or XYB mode is used
