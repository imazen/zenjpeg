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

## Feature: EdgePadding for MCU Edge Handling (IMPLEMENTED)

**Files:** `types.rs`, `encode/config.rs`, `encode/mod.rs`, `encode/baseline.rs`, `encode/progressive.rs`, `encode/output.rs`
**Status:** IMPLEMENTED for full-plane encoder, PENDING for strip encoder
**Date:** 2026-01-10

### Summary

Implemented configurable edge padding for MCU alignment to match C++ jpegli's `RowBuffer` padding behavior. The full-plane encoder now pads YCbCr planes to MCU-aligned dimensions before processing, with the original dimensions stored in the JFIF header so decoders crop correctly.

### Implementation Details

**New Types (`types.rs`):**
```rust
pub enum EdgePadding {
    Replicate,  // Default - replicate edge pixel outward (matches C++)
    Mirror,     // Reflect at edge
    Wrap,       // Tile the image
}

pub struct EdgePaddingConfig {
    pub luma: EdgePadding,
    pub chroma: EdgePadding,
}
```

**New Config Fields (`encode/config.rs`):**
- `edge_padding: EdgePaddingConfig` - Per-channel padding strategy
- `original_width: Option<u32>` - Original dimensions for JFIF header
- `original_height: Option<u32>`

**New Functions (`encode/mod.rs`):**
- `get_padded_coord()` - Calculate source coordinate for padding strategies
- `pad_plane_f32()` - Pad single f32 plane to MCU-aligned dimensions
- `pad_ycbcr_planes_subsampled()` - Pad Y, Cb, Cr planes with proper chroma handling

**Integration:**
- `encode/baseline.rs`: Calls `pad_ycbcr_planes_subsampled()` before DCT
- `encode/progressive.rs`: Same padding for both optimized and non-optimized paths
- `encode/output.rs`: Uses `original_width/height` for frame headers

### Why Per-Channel Padding?

Different strategies work better for different channel types:
- **Luma (Y):** Mirror preserves gradients at edges
- **Chroma (Cb/Cr):** Replicate is safer since chroma is upsampled by decoders

### Results After Implementation

**Comprehensive parity test (10 images × 50 quality levels):**
- Size: +0.26% (Rust slightly larger)
- DSSIM: +0.15% (essentially identical)
- Butteraugli: -0.00% (identical)

All 50 quality levels within 5% tolerance on both metrics.

### Remaining Work

The **strip encoder** (`StripProcessor`) doesn't have edge padding yet. The `test_frymire_backend_parity` test skips configurations where dimensions don't align to MCU boundaries. Adding strip encoder padding would require:
1. Padding horizontal strips as they're processed
2. Special handling for bottom-most strip with partial rows
3. Proper chroma plane coordination

---

## Historical Context: Edge MCU Parity Gap Analysis

**Date:** 2026-01-10

### Original Problem

Edge-tiled test revealed **massive parity differences** for partial MCU handling:
- Size: +15-21% larger than C++
- DSSIM: +60-137% worse quality

The HF modulation fix did NOT address this issue.

### Test Results BEFORE Edge Padding (2026-01-10)

Using `edge_mcu_parity` example with frymire.png (1118x1105):

**Right-edge only (--mode=right --edge-width 6):** 518x513
```
Quality |  Rust Size   C++ Size  Size Δ% | Rust DSSIM  C++ DSSIM   DSSIM Δ%
    q50 |      33911      28454  +19.18% |   0.000176   0.000110    +59.56%
    q75 |      37949      31291  +21.28% |   0.000078   0.000033   +136.75%
    q90 |      41817      36367  +14.99% |   0.000006   0.000005    +19.38%
    q95 |      44807      39545  +13.31% |   0.000002   0.000005    -48.61%
```

**Bottom-edge only (--mode=bottom --edge-height 6):** 518x518
```
Quality |  Rust Size   C++ Size  Size Δ% | Rust DSSIM  C++ DSSIM   DSSIM Δ%
    q50 |      32974      28876  +14.19% |   0.000236   0.000463    -49.06%
    q75 |      36676      32322  +13.47% |   0.000087   0.000057    +53.94%
    q90 |      40822      37485   +8.90% |   0.000006   0.000046    -86.55%
    q95 |      43455      41603   +4.45% |   0.000003   0.000027    -87.98%
```

**Both edges (--edge-width 6 --edge-height 6):** 518x518
```
Quality |  Rust Size   C++ Size  Size Δ% | Rust DSSIM  C++ DSSIM   DSSIM Δ%
    q50 |      97463      69712  +39.81% |   0.003538   0.004668    -24.20%
    q75 |     133927      92723  +44.44% |   0.000504   0.002781    -81.88%
    q90 |     183910     147391  +24.78% |   0.000090   0.001686    -94.67%
    q95 |     214138     193622  +10.60% |   0.000014   0.000052    -72.08%
```

### Key Observations

1. **Size:** Rust consistently produces larger files (+10-44%)
2. **Right edge:** Worse at low quality, converges at high quality
3. **Bottom edge:** Better DSSIM at high quality, still larger files
4. **Both edges:** Largest size difference but Rust has BETTER DSSIM (-24 to -94%)

The DSSIM pattern suggests Rust is using MORE bits than necessary (less aggressive
quantization), resulting in better quality but larger files. This is NOT the same
bug as the HF modulation wrap - this is a fundamentally different quantization issue.

### Comparison with Normal Images

The comprehensive C++ parity test on non-edge-tiled images shows:
- **Size: +0.26%** (excellent)
- **DSSIM: +0.15%** (essentially identical)
- **Butteraugli: -0.00%** (identical)
- All 50 quality levels within 5% on both metrics

This proves the general encoding path has excellent parity. The **edge block handling
specifically** is causing the +10-44% size bloat in edge-tiled tests.

### Root Cause Analysis

The C++ jpegli uses a `RowBuffer<T>` class (`common_internal.h:92-125`) that provides:

1. **Pre-allocated padding**: Extra columns for border access
2. **Negative indexing**: `row[-1]` returns replicated edge value
3. **PadRow method**: Replicates `row[0]` to negative indices, `row[width-1]` to `row[width..]`

```cpp
// C++ RowBuffer allocation - includes offset for negative indexing
void Allocate(cinfo, num_rows, rowsize) {
    size_t alignment = max(HWY_ALIGNMENT, vec_size);
    size_t min_memstride = alignment + rowsize * sizeof(T) + vec_size;
    stride_ = RoundUpTo(min_memstride, alignment) / sizeof(T);
    offset_ = alignment / sizeof(T);  // Allows row[-border] access
}

// C++ PadRow - fills both left and right borders
void PadRow(size_t y, size_t from, int border) {
    float* row = Row(y);
    for (int offset = -border; offset < 0; ++offset)
        row[offset] = row[0];           // Left border
    float last_val = row[from - 1];
    for (size_t x = from; x < xsize_ + border; ++x)
        row[x] = last_val;              // Right border
}
```

**In Rust**, we use regular `Vec<f32>` without this padding infrastructure. When AQ and DCT
calculations run, C++ has already padded the buffer, so accessing beyond width gives the
replicated edge value. Rust clamps indices but this affects the computed values differently.

### Proposed Fix

1. Create a `PaddedBuffer` struct that allocates with border space
2. Implement `PadRow` equivalent that replicates edge values
3. Use padded buffers for all Y/Cb/Cr strips in `StripProcessor`
4. Add an `EdgeHandling` enum to configure behavior:
   - `Clamp` - Current behavior (clamp indices)
   - `Replicate` - Pad buffers with replicated edge values (match C++)
   - `Pad8` - Expand to multiple of 8, store crop dimensions

### Files to Modify

1. `jpegli-rs/src/encode/strip.rs` - StripProcessor buffer allocation
2. `jpegli-rs/src/quant/aq/streaming.rs` - StreamingAQ buffer handling
3. Create `jpegli-rs/src/buffer.rs` - PaddedBuffer implementation

### How to Reproduce

```bash
cargo run --release --example edge_mcu_parity
cargo run --release --example edge_mcu_parity -- --mode=right --edge-width 6
cargo run --release --example edge_mcu_parity -- --mode=bottom --edge-height 6
cargo run --release --example edge_mcu_parity -- --edge-width 6 --edge-height 6
```

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
