# Size Parity Investigation Plan

## Current Status (2025-01-04)

After fixing AC refinement encoding for djpeg compatibility, we have excellent parity in most configurations but some size gaps remain:

### Configurations with Perfect Parity (0.0% diff)
- YCbCr Progressive + Optimized Huffman (all sizes)
- YCbCr Baseline + Fixed Huffman (all sizes)

### Configurations with Size Gaps

| Config | Size Diff | Priority |
|--------|-----------|----------|
| XYB Baseline + Fixed Huffman | +14% to +15% (small images) | **HIGH** |
| XYB Progressive + Opt Huffman | +0.2% to +5.7% | Medium |
| XYB Baseline + Opt Huffman | +0.0% to +1.1% | Low |
| YCbCr Baseline + Opt Huffman | +1.9% to +2.9% | Low |

---

## Investigation 1: XYB Color Conversion (FMA Issues) - **HIGH PRIORITY**

### Hypothesis
XYB color conversion involves many floating-point operations where FMA (fused multiply-add)
can produce slightly different results than separate multiply + add. This affects:
- `srgb_to_linear()` - gamma curve with pow()
- `linear_rgb_to_xyb()` - matrix multiplication with opsin bias
- `scale_xyb()` - scaling for JPEG range

Small per-pixel differences accumulate across the image, producing different DCT coefficients.

### Investigation Steps

1. **Check FMA usage in Rust vs C++**
   - Rust: Check if `#[cfg(target_feature = "fma")]` affects results
   - C++: Check Highway SIMD FMA usage in `enc_xyb.cc`
   - Compare with `-C target-cpu=native` vs without

2. **Compare XYB pipeline step-by-step**
   ```rust
   // For a few test pixels, compare:
   // Step 1: srgb_to_linear(r, g, b) -> (lr, lg, lb)
   // Step 2: linear_rgb_to_xyb(lr, lg, lb) -> (x, y, b)
   // Step 3: scale_xyb(x, y, b) -> (sx, sy, sb)
   // Report: max/mean diff at each stage
   ```

3. **Test specific FMA-sensitive operations**
   - Matrix multiply: `a*x + b*y + c*z + bias`
   - Cube root approximation
   - Gamma curve pow(x, 2.4) and pow(x, 1/2.4)

4. **Compare with forced scalar operations**
   - Disable SIMD in both Rust and C++
   - Compare scalar-only results

### Files to Examine
- `jpegli-rs/src/xyb.rs` - Rust XYB implementation
- `jpegli-cpp/lib/jxl/enc_xyb.cc` - C++ reference
- `jpegli-cpp/lib/jxl/enc_xyb-inl.h` - C++ SIMD implementation

### Tool to Create: `examples/compare_xyb_pipeline.rs`
```rust
// Compare XYB conversion step-by-step for specific pixel values
// Report differences at each stage to identify where divergence occurs
```

### Expected Outcome
Identify whether FMA or other floating-point differences cause XYB coefficient divergence.

---

## Investigation 2: YCbCr Baseline Optimized Huffman (+1.9% to +2.9%)

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

1. **XYB Color Conversion (FMA)** - Root cause of XYB gaps, affects all XYB modes
2. **XYB Baseline Fixed** - Large gap on small images, likely resolved by #1
3. **XYB Progressive** - May be resolved by fixing #1
4. **YCbCr Baseline Opt** - Low priority, small gap, likely Huffman tie-breaking

---

## Notes

- Progressive YCbCr with optimized Huffman already has 0.0% gap - this is our reference for "correct" behavior
- Fixed Huffman YCbCr baseline also has 0.0% gap - quantization and DCT are correct
- The gaps appear specifically when Huffman optimization is involved (YCbCr) or XYB mode is used

---

## Example Consolidation Plan

### Current State: 170 examples

| Category | Count | Notes |
|----------|-------|-------|
| debug_*.rs / trace_*.rs | 52 | Investigation artifacts |
| compare_*.rs | 27 | Many overlap in functionality |
| test_*.rs | 24 | Should be actual tests, not examples |
| xyb_*.rs | 14 | XYB-specific, some redundant |
| analyze_*.rs / check_*.rs | 12 | Various analysis tools |
| benchmark_*.rs | 6 | Performance measurement |
| Other | ~35 | Mixed utility |

### Essential Examples to Keep (~15)

**Core Comparison Tools:**
- `comprehensive_matrix.rs` - Full config matrix comparison (KEEP)
- `corpus_comparison.rs` - Multi-image corpus analysis (KEEP)
- `binary_compare.rs` - Byte-level diff for debugging (KEEP)

**Benchmarks:**
- `encode_benchmark.rs` - Encoding performance (KEEP)
- `decode_benchmark.rs` - Decoding performance (KEEP)
- `benchmark_decoders.rs` - Multi-decoder comparison (KEEP)

**Quality Analysis:**
- `pareto_comparison.rs` - Size vs quality tradeoff (KEEP)

**XYB Tools:**
- `xyb_cpp_comparison.rs` - XYB parity verification (KEEP)
- `encode_xyb.rs` - Simple XYB encoding example (KEEP)

**Huffman:**
- `dump_huffman_codes.rs` - DHT inspection (KEEP)
- `huffman_corpus_validation.rs` - Huffman correctness (KEEP)

**Utilities:**
- `roundtrip_corpus.rs` - Batch roundtrip testing (KEEP)

### Candidates for Removal (~100+)

**Debug artifacts (can be deleted):**
- `debug_50x50.rs`, `debug_compare_49_50.rs` - Size-specific debugging
- `debug_block_*.rs`, `debug_byte_trace.rs` - Low-level tracing
- `debug_decode_*.rs` (10+ files) - Decoder debugging
- `debug_refine_*.rs` - AC refinement debugging (fixed now)
- `trace_*.rs` (6 files) - Various tracing

**Redundant comparisons (consolidate):**
- `compare_baseline_dssim.rs` → merge into `comprehensive_matrix.rs`
- `compare_progressive_dssim.rs` → merge into `comprehensive_matrix.rs`
- `compare_baseline_progressive.rs` → covered by `comprehensive_matrix.rs`
- `compare_baseline_vs_progressive.rs` → duplicate
- `compare_progressive_decoders.rs` → merge into `benchmark_decoders.rs`

**Should be tests, not examples:**
- `test_*.rs` files → move to tests/ directory
- `validate_*.rs`, `verify_*.rs` → move to tests/

**XYB consolidation:**
- Keep: `xyb_cpp_comparison.rs`, `encode_xyb.rs`
- Merge others into a single `xyb_analysis.rs` or tests

### Action Items

1. [ ] Delete obvious debug artifacts (debug_50x50.rs, etc.)
2. [ ] Move test_*.rs files to tests/ directory
3. [ ] Consolidate compare_*_dssim.rs into comprehensive_matrix.rs
4. [ ] Consolidate XYB examples
5. [ ] Review remaining and delete unused
6. [ ] Update Cargo.toml to remove deleted examples

### Untracked Files Status

| File | Action | Reason |
|------|--------|--------|
| `compare_baseline_dssim.rs` | DELETE | Covered by comprehensive_matrix |
| `compare_progressive_dssim.rs` | DELETE | Covered by comprehensive_matrix |
| `trace_scan10.rs` (modified) | KEEP | Useful for scan debugging |
