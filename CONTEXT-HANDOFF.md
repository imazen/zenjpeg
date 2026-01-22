# Context Handoff - jpegli-rs Encoding Divergence Investigation

**Date**: 2026-01-22
**Last commit**: 8424b0c (fix: use Cr base matrix for 2-table mode)

## Session Summary

This session completed two tasks:

### 1. Fixed 2-table quant mode (jpeg_set_quality parity)

Added `separate_chroma_tables` flag to control 2 vs 3 quantization tables:
- `true` (default): 3 tables (Y, Cb, Cr) - matches `jpegli_set_distance()`
- `false`: 2 tables (Y, shared chroma) - matches `jpeg_set_quality()`

**Key fix**: When `separate_chroma_tables=false`, the Cb quant table now uses the **Cr base matrix** (not Cb), matching C++ jpegli behavior.

**Files changed**:
- `jpegli-rs/src/encode/streaming.rs` - Uses component index 2 (Cr) for Cb table when 2-table mode
- `jpegli-rs/src/encode/serialize.rs` - Writes 2 vs 3 DQT tables, sets correct SOF table IDs
- `jpegli-rs/src/encode/config.rs`, `encoder_config.rs`, `byte_encoders.rs` - Flag propagation

**Results after fix**:
| Mode | Size Δ | DSSIM Δ |
|------|--------|---------|
| 2-table (quality) | +0.3% to +1.8% | ±1-4% |
| 3-table (distance) | +0.3% to +2% | ±1-6% |

### 2. Investigated encoding divergence

**Verified identical**:
- AQ maps: 100% parity
- Quant tables: Identical with correct mode
- Zero-bias formulas: Match C++

**Remaining divergence** (±1 coefficient differences):
- 80% of Y blocks differ by ±1 in 1-3 AC coefficients
- 27-37% of chroma blocks differ by ±1
- Max diff: 6 (single DC outlier)
- Net effect: ~0.2% file size difference, ~1% butteraugli difference

**Root causes identified**:
1. **DCT SIMD differences**: Highway (C++) vs wide crate (Rust) produce slightly different floating-point intermediates
2. **Rounding mode**: Highway uses round-to-nearest-**even**, Rust uses round-to-nearest-**ties-away-from-zero**

## Key Code Locations

### Quantization
- `jpegli-rs/src/quant/mod.rs:594` - `quantize()` function uses `.round()` (ties-away-from-zero)
- `jpegli-rs/src/quant/mod.rs:667` - `quantize_block_with_zero_bias()` - zero-bias threshold logic
- C++ equivalent: `internal/jpegli-cpp/lib/jpegli/encode_finish.cc:55` - uses Highway `Round()` (ties-even)

### DCT
- Rust: `jpegli-rs/src/encode/dct.rs` or similar (uses wide crate)
- C++: Uses Highway SIMD

### Base quant matrices
- Rust: `jpegli-rs/src/foundation/consts.rs:192` - `BASE_QUANT_MATRIX_YCBCR[192]` (Y, Cb, Cr × 64)
- C++: `internal/jpegli-cpp/lib/jpegli/quant.cc:229` - `kBaseQuantMatrixYCbCr[]`

### 2-table mode logic
- C++: `internal/jpegli-cpp/lib/jpegli/quant.cc:686-697` - `add_two_chroma_tables` parameter
  - When false: `base_quant_matrix[1] = kBaseQuantMatrixYCbCr + 2 * DCTSIZE2` (uses Cr for both)
- Rust: `jpegli-rs/src/encode/streaming.rs:886` - `cb_component = if separate_chroma_tables { 1 } else { 2 }`

## Investigation Tools

```bash
# DCT coefficient comparison
cargo run --release --example compare_dct_coefficients -- [image] [distance]

# Synthetic coefficient test
cargo run --release --example coeff_synthetic_test

# AQ map comparison (set env var for both C++ and Rust)
DUMP_AQ_MAP=/tmp/cpp_aq.bin cjpegli input.png output.jpg
DUMP_AQ_MAP=/tmp/rust_aq.bin cargo run --example ...
cargo run --release --example compare_aq_maps -- /tmp/cpp_aq.bin /tmp/rust_aq.bin
```

## Next Steps to Pursue

1. **Compare pre-quantized DCT values** - Add instrumentation to dump floating-point DCT coefficients before quantization from both implementations

2. **Try round-to-even in Rust** - Implement and test banker's rounding to see if it reduces differences:
   ```rust
   fn round_ties_even(x: f32) -> f32 {
       let r = x.round();
       if (x - x.floor()) == 0.5 && (r as i32) % 2 != 0 {
           r - x.signum()
       } else {
           r
       }
   }
   ```

3. **Compare raw DCT implementations** - Feed identical 8×8 pixel blocks to both DCT implementations and compare outputs

4. **Investigate the DC outlier** - The single block with diff=6 is unusual; worth investigating if it's a bug or edge case

## Current Parity Status

| Component | Rust vs C++ | Notes |
|-----------|-------------|-------|
| AQ maps | 100% identical | Verified with DUMP_AQ_MAP |
| Quant tables | Identical | With matching mode (2 or 3 tables) |
| DCT coefficients | ±1 differences | 80% of Y blocks, floating-point variance |
| File size | +0.2% to +2% | Acceptable |
| Butteraugli | ~1% | Essentially identical perceptual quality |

## Files to Read First

1. `CLAUDE.md` - Full project guide with profiling results, known bugs, API rules
2. `jpegli-rs/src/quant/mod.rs` - Quantization and zero-bias implementation
3. `jpegli-rs/examples/compare_dct_coefficients.rs` - Coefficient comparison tool
4. `internal/jpegli-cpp/lib/jpegli/encode_finish.cc:40-58` - C++ quantization with zero-bias
