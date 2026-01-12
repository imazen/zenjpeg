# Code Analysis Notes

This file preserves investigation context that might otherwise be lost.

---

## Bug: HF Modulation Index Wrap in Rightmost Blocks

**File:** `jpegli-rs/src/quant/aq/simd.rs:566`
**Status:** FIXED
**Date:** 2025-01-10

### Summary

In `hf_modulation_sum_8x8`, the vertical difference SIMD path is missing a column validity check, causing rightmost partial blocks to read pixels from the next row.

### Root Cause

The horizontal SIMD path (line 541) correctly checks column bounds:
```rust
if row_start + 9 <= block.len() && block_x + 8 < img_width {
```

But the vertical SIMD path (line 566) only checks slice length:
```rust
if row_start + 8 <= block.len() && next_row_start + 8 <= block.len() {
```

**Missing:** `block_x + 8 <= img_width`

### Why This Causes Wrapping

For image width=140 (18 blocks, last block at x=136 with only 4 valid columns):

- `block` slice is contiguous memory starting at `(y_start, x_start)`
- `block[0]` = col 136, `block[3]` = col 139 (valid)
- `block[4]` = col 140 = **row y+1, column 0** (WRAPPED!)
- `load_f32x8(block, row_start)` reads positions 0-7
- Positions 4-7 are from the wrong row!

### C++ Reference

C++ `HfModulation` (adaptive_quantization.cc:295-324) uses `RowBuffer` with proper stride/padding, so each row access stays within its row. They also use a mask for the rightmost pixel:

```cpp
HWY_ALIGN constexpr uint32_t kMaskRight[8] = {~0u, ~0u, ~0u, ~0u,
                                              ~0u, ~0u, ~0u, 0};
```

### Proposed Fix

Change line 566 from:
```rust
if row_start + 8 <= block.len() && next_row_start + 8 <= block.len() {
```

To:
```rust
if row_start + 8 <= block.len() && next_row_start + 8 <= block.len() && block_x + 8 <= img_width {
```

### Testing

This affects images where `width % 8 != 0`. Test with:
- width=140 (4 leftover columns)
- width=67 (3 leftover columns)
- Any non-multiple-of-8 width

Compare AQ output before/after fix, and run C++ parity tests.

### Impact on Parity

This bug causes incorrect AQ strength values for rightmost blocks, which affects quantization decisions. Fixing it should improve C++ parity for images with non-multiple-of-8 widths.

### Test Results (2025-01-10)

**Test 1: Corpus images (widths divisible by 8)**
- No change - bug path not exercised

**Test 2: frymire.png (1118x1105, remainder 6 and 1)**
```
BEFORE FIX (frymire + 1001682):
  Size difference:       +0.39%
  DSSIM difference:      +0.19%
  Butteraugli difference: +0.06%

AFTER FIX (frymire + 1001682):
  Size difference:       +0.39%
  DSSIM difference:      +0.19%
  Butteraugli difference: +0.06%
```

**Why no measurable change?**
- Only rightmost block column affected (1 of 140 = 0.7% of blocks)
- Only 2 of 8 pixels per row read wrong data in affected blocks
- Net impact: ~0.7% × 25% = 0.18% of pixels, diluted in aggregate metrics
- Fix is correct; effect is below measurement threshold

**Verification:** 14/14 SIMD unit tests pass.

---

## Feature: Edge Handling for MCU Alignment (COMPLETE)

**Status:** COMPLETE - Strip encoder has excellent C++ parity
**Date:** 2026-01-10 (updated 2026-01-12)

### Summary

The streaming/strip encoder handles non-MCU-aligned dimensions correctly, achieving excellent C++ parity.

### Current Results

**Comprehensive parity test (10 images × 50 quality levels):**
- Size: +0.26% (Rust slightly larger)
- DSSIM: +0.15% (essentially identical)
- Butteraugli: -0.00% (identical)

All 50 quality levels within 5% tolerance on both metrics.

**Edge-specific tests (`strip_edge_cpp_comparison`):**
- All partial MCU widths (1-15) and heights (1-15) tested
- DSSIM within 0.6% of C++ for all edge cases

### Historical Note

The old full-plane encoder had edge handling issues and was removed in commit `0fd8adc`.
The `Encoder` type now wraps `StreamingEncoder` exclusively.

---

## Bug: Progressive Non-Interleaved Block Count Mismatch

**File:** `jpegli-rs/src/encode/progressive.rs`
**Status:** FIXED
**Date:** 2026-01-10

### Summary

Progressive JPEG encoding with subsampled modes (4:2:0, 4:2:2, 4:4:0) and non-MCU-aligned dimensions produced malformed JPEGs that failed to decode with zune-jpeg ("Marker missing where expected").

### Root Cause

For **non-interleaved progressive scans**, the JPEG spec requires block counts based on ORIGINAL dimensions, not MCU-padded dimensions:

- **Interleaved scans:** blocks = MCUs × blocks_per_MCU (uses MCU-aligned dimensions)
- **Non-interleaved scans:** blocks = ceil(width/8) × ceil(height/8) (uses original dimensions)

Example for 67×71 with 4:2:0:
- MCU-padded dimensions: 80×80
- Interleaved (baseline): 5×5 MCUs × 4 Y blocks = 100 Y blocks ✓
- Non-interleaved (progressive): ceil(67/8) × ceil(71/8) = 9×9 = 81 Y blocks

We were encoding 100 Y blocks but the SOF header said 67×71, so decoders expected only 81 blocks and failed when they didn't find the next marker.

### The Fix

After `quantize_all_blocks_subsampled()` returns blocks for the padded dimensions, filter them to keep only blocks that correspond to the original dimensions:

```rust
let orig_y_blocks_h = (width + 7) / 8;
let orig_y_blocks_v = (height + 7) / 8;
let padded_y_blocks_h = padded_w / 8;
let padded_y_blocks_v = padded_h / 8;

let y_blocks = if padded_y_blocks_h != orig_y_blocks_h
    || padded_y_blocks_v != orig_y_blocks_v
{
    // Filter Y blocks: extract blocks for original dimensions
    let mut filtered = Vec::with_capacity(orig_y_blocks_h * orig_y_blocks_v);
    for by in 0..orig_y_blocks_v {
        for bx in 0..orig_y_blocks_h {
            let padded_idx = by * padded_y_blocks_h + bx;
            filtered.push(y_blocks_padded[padded_idx]);
        }
    }
    filtered
} else {
    y_blocks_padded
};
```

The same filtering is applied to chroma blocks.

### Testing

Test cases verified:
- 67×71 (small, both dimensions non-aligned)
- 1118×1105 (frymire, both dimensions non-aligned)
- 100×100 (vertical edge: height non-aligned)
- 127×129 (both edges)

All progressive subsampled modes now decode successfully with zune-jpeg.

### Impact

- File sizes decreased slightly (encoding fewer padding blocks)
- Hash updates required for `progressive_420_opt` in locked tests
- Strip encoder unaffected (uses baseline which is interleaved)

---

## Template for Future Bugs

**File:**
**Status:** UNFIXED/FIXED/INVESTIGATING
**Date:**

### Summary

### Root Cause

### C++ Reference

### Proposed Fix

### Testing

---

## Performance Analysis: Rust vs C++ jpegli

**Date:** 2026-01-11
**Status:** Analysis complete, optimizations TBD

### Profiling Results

Profiled with `perf record` on 2048x2048 encoding, 5 iterations.

**Key Finding: Huffman encoding is NOT the bottleneck (only 3.59% of CPU time)**

#### CPU Time Breakdown

| Component | CPU % | Notes |
|-----------|-------|-------|
| **AQ (Adaptive Quant)** | 17.74% | Biggest bottleneck |
| ↳ per_block_modulations | 6.26% | |
| ↳ fuzzy_erosion | 5.89% | |
| ↳ pre_erosion | ~5% | |
| **DCT** | 14.35% | forward_dct_8x8 |
| **Memory ops** | 6.82% | memset/memmove from zeroed allocs |
| **Huffman table build** | 5.65% | Optimized tables construction |
| **Huffman encode** | 3.59% | The actual encoding |

#### Timing Matrix (Rust vs C++)

Overall: Rust is ~90% slower than C++ across configurations.

| Mode | Config | Rust Slowdown |
|------|--------|---------------|
| YUV/SEQ/FIX | Fixed Huffman | 133-184% |
| YUV/SEQ/OPT | Optimized Huffman | 155-241% |
| YUV/PROG/* | Progressive | 62-97% |

### C++ vs Rust Implementation Differences

1. **Horizontal SIMD reduction**: C++ uses `SumOfLanes(d, sum)` (single vectorized instruction). We extract to array and sum scalarly:
   ```rust
   // Our approach (slower)
   let arr: [f32; 8] = h_diff.into();
   sum += arr[0] + arr[1] + arr[2] + ... // 7 scalar adds

   // C++ approach (faster)
   sum = SumOfLanes(d, sum);  // Single instruction
   ```

2. **Buffer allocation**: Our `fuzzy_erosion_simd` allocates fresh tmp buffer (`try_alloc_zeroed`) every call. C++ reuses buffers via RowBuffer class.

3. **Per-block processing**: C++ keeps values in SIMD registers through ComputeMask→HfModulation→GammaModulation chain. We extract to scalar for each block.

4. **Selection sort in weighted_min4_of_9**: Pure scalar with 32 comparisons. C++ likely uses SIMD min operations.

### Optimization Opportunities

1. **Replace array sum with SIMD horizontal sum** - Use `wide` crate's `reduce_add()` on f32x8
2. **Reuse fuzzy_erosion tmp buffer** - Pass pre-allocated workspace, avoid zeroed alloc
3. **Batch multiple blocks** - Process 8 blocks in parallel (one per SIMD lane)
4. **Profile and optimize DCT** - Second largest cost (14.35%)

### Files Involved

- `jpegli-rs/src/quant/aq/simd.rs` - AQ SIMD implementation
- `jpegli-rs/src/dct/forward.rs` - Forward DCT
- `jpegli-rs/src/entropy/encoder.rs` - Huffman encoding
- `jpegli-rs/src/huffman/encode.rs` - Huffman table construction

---

## Performance Analysis: Decoder

**Date:** 2026-01-12
**Status:** Analysis complete, optimizations in progress

### Profiling Results

Profiled with `perf record` on 2048x2048 decoding, 10 iterations.

**Throughput:** ~38 MP/s (megapixels per second)

#### CPU Time Breakdown

| Component | CPU % | Notes |
|-----------|-------|-------|
| **Color conversion** | 15.83% | `ycbcr_planes_f32_to_rgb_u8` - MAIN BOTTLENECK |
| **Entropy decoding** | 6.01% | `EntropyDecoder::decode_block` |
| **Scan processing** | 4.89% | `JpegParser::decode_scan` |
| **IDCT (scalar)** | 3.07% | `inverse_dct_8x8` |
| **IDCT (SIMD)** | 0.71% | `inverse_dct_8x8_simd` |

### Color Conversion Bottleneck

The `ycbcr_planes_f32_to_rgb_u8` function does SIMD math but then **extracts to scalar arrays for interleaved RGB storage**:

```rust
// SIMD compute (fast)
let r = cr_to_r.mul_add(cr, y + offset).max(zero).min(max_val);
let g = cb_to_g.mul_add(cb, cr_to_g.mul_add(cr, y + offset)).max(zero).min(max_val);
let b = cb_to_b.mul_add(cb, y + offset).max(zero).min(max_val);

// Extract to scalar (slow!)
let r_arr: [f32; 8] = r.into();
let g_arr: [f32; 8] = g.into();
let b_arr: [f32; 8] = b.into();

// Scalar interleave loop (very slow!)
for j in 0..8 {
    let idx = (base + j) * 3;
    rgb[idx] = r_arr[j] as u8;
    rgb[idx + 1] = g_arr[j] as u8;
    rgb[idx + 2] = b_arr[j] as u8;
}
```

### Proposed Optimization: SIMD RGB Interleave

**Strategy 1: Pack and Shuffle**
1. Convert f32x8 → u8x8 (pack with saturation)
2. Use SIMD shuffle to interleave RGB
3. Store 24 bytes at once

**Strategy 2: Process 16 pixels → 48 bytes**
1. Load 16 pixels (2× f32x8 per channel)
2. Pack to i16x16, then u8x32
3. Use AVX2 VPERMD for interleave
4. Store aligned 48 bytes

### Files to Modify

- `jpegli-rs/src/color.rs:465` - `ycbcr_planes_f32_to_rgb_u8`

### Comparison: Encoder vs Decoder Bottlenecks

| Area | Encoder % | Decoder % |
|------|-----------|-----------|
| AQ/Quant | 17.74% | N/A |
| DCT/IDCT | 14.35% | 3.78% |
| Huffman | 9.24% | 6.01% |
| Color | <1% | **15.83%** |

The encoder spends most time on AQ and DCT; the decoder spends most time on color conversion.

