# Code Analysis Notes

This file preserves investigation context that might otherwise be lost.

---

## Bug: HF Modulation Index Wrap in Rightmost Blocks

**File:** `zenjpeg/src/quant/aq/simd.rs:566`
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

**File:** `zenjpeg/src/encode/progressive.rs`
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

- `zenjpeg/src/quant/aq/simd.rs` - AQ SIMD implementation
- `zenjpeg/src/dct/forward.rs` - Forward DCT
- `zenjpeg/src/entropy/encoder.rs` - Huffman encoding
- `zenjpeg/src/huffman/encode.rs` - Huffman table construction

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

- `zenjpeg/src/color.rs:465` - `ycbcr_planes_f32_to_rgb_u8`

### Comparison: Encoder vs Decoder Bottlenecks

| Area | Encoder % | Decoder % |
|------|-----------|-----------|
| AQ/Quant | 17.74% | N/A |
| DCT/IDCT | 14.35% | 3.78% |
| Huffman | 9.24% | 6.01% |
| Color | <1% | **15.83%** |

The encoder spends most time on AQ and DCT; the decoder spends most time on color conversion.

---

## Feature: f32 YCbCr Streaming API

**Date:** 2026-01-12
**Status:** COMPLETE

### Summary

Added streaming APIs for both encoder and decoder that accept/output f32 YCbCr data directly, bypassing the expensive color conversion step.

### API

**Decoder:**
```rust
pub fn decode_to_ycbcr_f32(&self, data: &[u8]) -> Result<DecodedYCbCr>

pub struct DecodedYCbCr {
    pub y: Vec<f32>,      // [-128, 127] range
    pub cb: Vec<f32>,     // [-128, 127] range
    pub cr: Vec<f32>,     // [-128, 127] range
    pub width: u32,
    pub height: u32,
    pub icc_profile: Option<Vec<u8>>,
}
```

**Encoder:**
```rust
// Full-resolution chroma (will be downsampled)
pub fn push_ycbcr_strip_f32(&mut self, y: &[f32], cb: &[f32], cr: &[f32], num_rows: usize)

// Pre-downsampled chroma
pub fn push_ycbcr_strip_f32_subsampled(&mut self, y: &[f32], cb: &[f32], cr: &[f32], num_rows: usize)
```

### Performance

Benchmark on 2048x2048 images (10 iterations):

| Path | Time (ms) | MP/s | Speedup |
|------|-----------|------|---------|
| RGB decode | 81.9 | 51.2 | baseline |
| YCbCr decode | 65.5 | 64.1 | **1.25x** |
| zune-jpeg | 3.7 | 1146 | 22x |

The YCbCr path is 25% faster by bypassing color conversion (which was 15.8% of decode time). zune-jpeg remains much faster due to comprehensive SIMD optimization.

### Use Cases

- Video pipelines that work in YCbCr space
- Re-encoding without color space round-trip
- Custom color space transformations
- Maximum performance when RGB output is not needed

### Files Modified

- `zenjpeg/src/decode/mod.rs` - Added `DecodedYCbCr`, `decode_to_ycbcr_f32()`, `to_ycbcr_planes_f32()`
- `zenjpeg/src/encode/strip.rs` - Added `process_strip_ycbcr_f32()`, `process_strip_ycbcr_f32_subsampled()`
- `zenjpeg/src/encode/streaming.rs` - Added `push_ycbcr_strip_f32()`, `push_ycbcr_strip_f32_subsampled()`
- `zenjpeg/src/lib.rs` - Exported `DecodedYCbCr`
- `zenjpeg/tests/ycbcr_f32_api.rs` - Test coverage
- `zenjpeg/examples/ycbcr_benchmark.rs` - Performance benchmark

---

## Performance Analysis: zune-jpeg vs jpegli-rs Decoder

**Date:** 2026-01-12
**Status:** Analysis complete, optimizations categorized

### Performance Gap

| Decoder | 2048×2048 | MP/s | Relative |
|---------|-----------|------|----------|
| zune-jpeg | 3.7ms | 1146 | 18× faster |
| jpegli-rs YCbCr | 65.5ms | 64 | 1.25× faster than RGB |
| jpegli-rs RGB | 81.9ms | 51 | baseline |

### zune-jpeg Optimization Techniques

#### 1. IDCT (Inverse DCT)

**Scalar implementation:**
- Integer-only arithmetic (no floating point)
- 16-bit intermediate values with 14-bit fixed-point coefficients
- DC-only shortcut: `if block[1..64] == 0` return `[dc_coeff * 8; 64]`
- 4×4 IDCT variant for blocks with only top-left 4×4 non-zero
- AAN algorithm with minimal multiplications

**AVX2 implementation (`idct_avx2`):**
- In-register 8×8 transpose using `_mm256_unpacklo/hi_epi16` + `_mm256_permute2x128_si256`
- No memory round-trip during row/column passes
- `_mm256_madd_epi16` for efficient multiply-accumulate
- Single function processes entire 8×8 block without function calls

#### 2. Color Conversion (YCbCr → RGB)

**Scalar implementation:**
- Fixed-point coefficients (14-bit precision)
- Input/output in i16 (no f32)
- `>> 14` for final denormalization
- Direct saturation via `.clamp(0, 255) as u8`

**AVX2 implementation (`ycbcr_to_rgb_avx`):**
- Processes 16 pixels per iteration (32 bytes RGB output)
- Uses `_mm256_madd_epi16` for coefficient multiply-add
- Clever RGB interleaving: `shuffle` → `blend` → `permute4x64`
- Single loop with no inner branches

#### 3. Entropy Decoding (Huffman)

**Key optimizations:**
- 9-bit lookahead table (`HUFF_LOOKAHEAD = 9`)
- Combined AC symbol + magnitude decoding (`ac_lookup` table)
- Fast 4-byte refill when no 0xFF markers present
- `has_byte()` bit-hack to detect 0xFF in u32
- Aligned buffer for faster peek operations
- MSB-aligned `aligned_buffer` avoids shifting on peek
- `decode_huff!` macro inlined at call sites

**AC coefficient acceleration:**
```rust
// Decodes symbol + magnitude in single lookup (when possible)
fast_ac[i] = (k << 8) + (run << 4) + (len + mag_bits);
```

### Categorized Optimization Opportunities

#### XYB-SAFE (Can apply without affecting XYB precision)

| Optimization | Component | Speedup Est. | Notes |
|--------------|-----------|--------------|-------|
| DC-only shortcut | IDCT | 5-15% | Already implemented in jpegli-rs |
| In-register transpose | IDCT | 10-20% | Avoid memory round-trip |
| Combined AC lookup | Entropy | 5-10% | Decode symbol+magnitude together |
| 4-byte fast refill | Entropy | 5-10% | Skip marker check for common case |
| Aligned buffer peek | Entropy | 2-5% | MSB-aligned avoids shifting |
| SIMD RGB interleave | Color | 10-15% | Pack f32→u8 with shuffle |
| `wide` reduce_add() | AQ | 3-5% | Replace scalar sum extraction |
| Buffer reuse | AQ | 2-5% | Avoid allocations in fuzzy_erosion |

#### STANDARD-ONLY (Would compromise XYB precision)

| Optimization | Component | Speedup Est. | Why it breaks XYB |
|--------------|-----------|--------------|-------------------|
| Integer IDCT | IDCT | 30-50% | XYB requires f32 for extended gamut |
| Fixed-point color | Color | 20-40% | XYB color transform needs f32 |
| i16 coefficients | All | 20-30% | XYB HDR values exceed i16 range |
| 4×4 IDCT shortcut | IDCT | 5-10% | XYB coefficient distribution differs |

#### WHY XYB NEEDS f32

1. **Extended gamut**: XYB encodes HDR content with values outside [0,255]
2. **Precision requirements**: XYB → linear RGB transform is lossy with integer math
3. **Non-standard coefficient patterns**: XYB doesn't have same sparsity as YCbCr

### Recommended Optimizations (Priority Order)

1. **SIMD RGB interleave** (color.rs:465) - 15.8% of decode time
   - Replace scalar loop with `wide` pack/shuffle operations
   - Use `i32x8` → `i16x8` → `u8x16` packing chain

2. **In-register IDCT transpose** (idct.rs)
   - Keep all 8 rows in SIMD registers through both passes
   - Use `wide::f32x8::transpose()` already available

3. **Combined AC lookup table** (entropy/decoder.rs)
   - Pre-compute symbol+magnitude for short codes
   - Fast path when code_len + mag_bits ≤ lookahead_bits

4. **Fast bitstream refill** (bitstream.rs)
   - 4-byte read when no 0xFF present
   - Use Stanford bit-hack for 0xFF detection

5. **AQ buffer reuse** (quant/aq/simd.rs)
   - Pass workspace buffer to fuzzy_erosion
   - Avoid zeroed allocation per call

### Implementation Notes

**For SIMD RGB interleave with `wide`:**
```rust
// Current (slow): extract to arrays, scalar interleave
let r_arr: [f32; 8] = r.into();
for j in 0..8 { rgb[idx+j*3] = r_arr[j] as u8; }

// Proposed (fast): pack and shuffle
// 1. Convert f32x8 → i32x8 (with clamp)
// 2. Pack pairs: i32x8 + i32x8 → i16x16
// 3. Pack pairs: i16x16 + i16x16 → u8x32
// 4. Shuffle to interleave RGB
```

**For `wide` horizontal sum:**
```rust
// Current: extract and sum scalarly
let arr: [f32; 8] = h_diff.into();
sum += arr[0] + arr[1] + ... + arr[7];

// Proposed: use wide's reduce
sum += h_diff.reduce_add();  // Single SIMD reduction
```

### Files to Modify

| File | Optimization |
|------|--------------|
| `src/color.rs:465` | SIMD RGB interleave |
| `src/idct.rs` | In-register transpose (already mostly done) |
| `src/entropy/decoder.rs` | Combined AC lookup |
| `src/bitstream.rs` | Fast 4-byte refill |
| `src/quant/aq/simd.rs` | wide reduce_add, buffer reuse |

---

## Bug: 1-Pixel Partial MCU Edge Quality Gap

**Status:** OPEN - Investigation in progress
**Date:** 2026-01-19

### Summary

When encoding images with partial MCU width of exactly 1 pixel (e.g., width=257), Rust produces significantly worse quality than C++ jpegli.

### Test Results (edge_tile_ssim2_comparison)

Test tiles the rightmost partial MCU pixels across the image to amplify edge effects:

| Image | Dims | Edge | Rust SSIM2 | C++ SSIM2 | Diff |
|-------|------|------|------------|-----------|------|
| 258947 | 257x256 | 1 | 62.34 | 85.27 | **-22.93** |
| 258947 | 259x256 | 3 | 77.02 | 76.43 | +0.59 |
| 258947 | 262x256 | 6 | 80.06 | 79.61 | +0.45 |
| 258947 | 257x129 | 1x1 | 59.74 | 89.19 | **-29.45** |

**Pattern:**
- 1-pixel edges: -15 to -35 SSIM2 (severe)
- 3+ pixel edges: ~0 (parity)
- Height-only edges: -1 to -3 (minor)

### Investigation Notes

1. **Edge padding logic is correct** - Both use Replicate strategy
2. **`hf_modulation_sum_8x8` scalar path** - For 1-pixel edge with `img_width=padded_width`:
   - `h_count = 7` (computes 7 differences)
   - But these differences are between replicated pixels (all 0)
3. **Potential issue in `per_block_modulations_row`**:
   - Line 680 passes `width` as both stride AND `img_width`
   - When called from streaming AQ, `width = padded_width`
   - This may cause incorrect boundary handling

### C++ Behavior

C++ `HfModulation` (adaptive_quantization.cc:295-324):
- Uses RowBuffer with implicit stride/padding
- Always does SIMD loads (relies on padding for safety)
- Uses mask `kMaskRight = [1,1,1,1,1,1,1,0]` to zero position 7
- Does NOT check `img_width` explicitly

### Possible Fixes

1. Pass actual `img_width` separately from stride to `hf_modulation_sum_8x8`
2. Match C++ behavior: always use SIMD with mask, rely on padding
3. Investigate if issue is in AQ, DCT, or quantization path

### Test Command

```bash
cargo test --release -p zenjpeg@0.9.0 --test edge_tile_ssim2_comparison -- --nocapture --ignored
```

---

