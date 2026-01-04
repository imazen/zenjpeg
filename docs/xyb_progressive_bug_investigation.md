# XYB Progressive Quality Bug Investigation

**Status:** PARTIALLY FIXED - corruption reduced from 93% to 50%, MCU ordering issue identified
**Severity:** High - 50% of pixels still decode incorrectly (down from 93%)
**Date:** 2026-01-04 (Updated)

## Bug Summary

XYB Progressive mode produces severely corrupted decoded images. When decoded with a standard JPEG decoder, 93% of pixels differ from what XYB Baseline produces, even though both modes should produce identical quantized DCT coefficients and thus identical decoded output.

### Symptoms

| Metric | Value | Expected |
|--------|-------|----------|
| Pixels differing from baseline | 11,445 / 12,288 (93%) | 0% |
| SSIMULACRA2 score | -102.15 | ~3.28 (like baseline) |
| Decoded pixel example | R=91 G=80 B=155 | R=91 G=36 B=191 |

**Example (Pixel 0 of 64x64 gradient at Q70):**
- Original input: R=0 G=0 B=128
- XYB Baseline decoded: R=91 G=36 B=191 ✅
- XYB Progressive decoded: R=91 G=80 B=155 ❌ (44-point error in G, 36-point error in B)
- C++ XYB Baseline: R=91 G=36 B=191 ✅
- C++ XYB Progressive: R=91 G=36 B=191 ✅

### Key Finding

**C++ behavior (correct):** XYB Baseline and Progressive produce IDENTICAL decoded output
**Rust behavior (broken):** XYB Baseline and Progressive produce DIFFERENT decoded output

This proves the bug is Rust-specific and NOT an inherent limitation of XYB progressive encoding.

## How to Reproduce

### Quick Test (64x64 gradient)

```bash
cargo run --release --example compare_xyb_decoded
```

**Expected output:** "0 pixels differ" between Rust baseline and progressive
**Actual output:** "11,445 pixels differ" (93% corruption)

### Comprehensive Matrix Test

```bash
cargo run --release --example comprehensive_matrix 2>&1 | grep "XYB.*Prog"
```

Look for SSIM2 scores around -102 (should be positive like +3).

### Manual Inspection

```bash
# Generate test JPEGs
cargo run --release --example diagnose_xyb_progressive

# Files saved to /tmp/:
#   xyb_baseline.jpg    - correct decoding
#   xyb_progressive.jpg - corrupted decoding

# View with any JPEG viewer (Firefox, etc.)
firefox /tmp/xyb_baseline.jpg /tmp/xyb_progressive.jpg
```

The progressive image will have incorrect colors (more cyan/blue tint).

## Investigation History

### Initial Discovery (2026-01-04)

Comprehensive matrix test revealed:
- YCbCr modes: Working correctly
- XYB Baseline: Working correctly (SSIM2 = 3.27)
- XYB Progressive: **SSIM2 = -102.15** ⚠️

### First Hypothesis: Missing Adaptive Quantization

**Investigation:** Compared quantization code between XYB Baseline and Progressive

**Finding:** XYB Progressive was NOT using Adaptive Quantization!

**XYB Baseline (encode.rs:1220):**
```rust
let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb_with_aq_simple(
    &x_plane, &y_plane, &b_downsampled,
    width, height, b_width, b_height,
    &x_quant, &y_quant, &b_quant,
    &aq_map,          // ← HAS AQ MAP
    &x_zero_bias,     // ← HAS ZERO BIAS
    &y_zero_bias,
    &b_zero_bias,
);
```

**XYB Progressive BEFORE fix (encode.rs:1403):**
```rust
let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb(
    &x_plane, &y_plane, &b_downsampled,
    width, height, b_width, b_height,
    &x_quant, &y_quant, &b_quant,
    // ← NO AQ MAP! NO ZERO BIAS!
);
```

**Fix Applied:** Added AQ computation to XYB Progressive (encode.rs:1402-1411):
```rust
// Compute AQ map from Y plane (same as baseline XYB)
let y_plane_scaled: Vec<f32> = y_plane.iter().map(|&v| v * 255.0).collect();
let y_quant_01 = y_quant.values[1];
let aq_map = compute_aq_strength_map(&y_plane_scaled, width, height, y_quant_01);

// Generate zero-bias parameters (same as baseline XYB)
let effective_distance = quant::quant_vals_to_distance(&x_quant, &y_quant, &b_quant);
let x_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
let b_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
```

**Result:** Bug persists! Still 93% pixel difference.

**Conclusion:** The bug is NOT in quantization. Both modes now use identical quantization logic, but decoded output is still different.

## Current Understanding

### What We Know

1. ✅ **Quantization is now correct** - Both modes use `quantize_all_blocks_xyb_with_aq_simple()` with same parameters
2. ✅ **Bug is Rust-specific** - C++ XYB Progressive works correctly
3. ✅ **Bug affects decoded output** - Not just a measurement issue
4. ✅ **Bug is in progressive encoding pipeline** - After quantization, before/during bitstream writing

### What We Don't Know

❓ **Where exactly are the DCT coefficients being corrupted?**

Possible locations:
- Progressive scan tokenization (Pass 1)
- Huffman table building from tokens
- Token replay during encoding (Pass 2)
- Progressive scan structure/script for XYB
- Huffman context assignments for XYB progressive

## Debugging Next Steps

### Step 1: Verify Quantized Coefficients Match

Add debug output to compare `x_blocks`, `y_blocks`, `b_blocks` between baseline and progressive AFTER quantization:

```rust
// In encode_progressive_xyb_optimized() after quantization:
eprintln!("PROGRESSIVE - First X block coefficients: {:?}", &x_blocks[0]);

// In encode_baseline_xyb() after quantization:
eprintln!("BASELINE - First X block coefficients: {:?}", &x_blocks[0]);
```

**Expected:** Coefficients should be IDENTICAL
**If different:** Bug is in quantization logic (something we missed)
**If identical:** Bug is in progressive encoding/replay

### Step 2: Add Tokenization Debug Logging

If coefficients match, add logging to the tokenization process:

```rust
// In encode_progressive_xyb_optimized() during tokenization:
for (scan_idx, scan) in scans.iter().enumerate() {
    eprintln!("Tokenizing scan {}: ss={} se={} ah={} al={}",
              scan_idx, scan.ss, scan.se, scan.ah, scan.al);

    // After tokenization:
    eprintln!("  Scan {} tokenized: {} tokens", scan_idx, token_count);
}
```

### Step 3: Compare Progressive Scan Scripts

Check if XYB and YCbCr use the same scan script:

```rust
// In get_progressive_scan_script():
let scans = self.get_progressive_scan_script(is_color);
eprintln!("XYB Progressive scan script: {} scans", scans.len());
for scan in &scans {
    eprintln!("  Scan: components={:?} ss={} se={} ah={} al={}",
              scan.components, scan.ss, scan.se, scan.ah, scan.al);
}
```

Compare with YCbCr progressive output.

### Step 4: Check Huffman Context Assignments

XYB progressive uses different context assignments than YCbCr:

```rust
// From encode.rs:1425-1429
let context = if scan.ss == 0 && scan.se == 0 {
    0 // DC: all components use context 0
} else {
    num_components as u8 // AC: all components use context 3
};
```

Verify this is correct. Maybe XYB should use different contexts?

### Step 5: Compare Token Replay

Add logging during token replay:

```rust
// During replay_progressive_scan():
eprintln!("Replaying scan {}: {} tokens", scan_idx, token_count);

// After replay:
eprintln!("  Generated {} bytes", scan_data.len());
```

Compare with baseline encoding.

## Relevant Code Locations

### Main Functions

| File | Function | Line | Description |
|------|----------|------|-------------|
| `encode.rs` | `encode_baseline_xyb()` | 1126 | XYB Baseline encoder (working) |
| `encode.rs` | `encode_progressive_xyb_optimized()` | 1380 | XYB Progressive encoder (broken) |
| `encode.rs` | `quantize_all_blocks_xyb_with_aq_simple()` | 4288 | Quantization with AQ (now used by both) |
| `encode.rs` | `get_progressive_scan_script()` | ~3900 | Defines progressive scan structure |
| `encode.rs` | `replay_progressive_scan()` | ~2800 | Replays tokens to bitstream |

### Diagnostic Tools

| File | Purpose |
|------|---------|
| `examples/compare_xyb_decoded.rs` | Compares decoded pixels between modes |
| `examples/diagnose_xyb_progressive.rs` | Analyzes XYB progressive quality |
| `examples/comprehensive_matrix.rs` | Full matrix with SSIM2 scores |

### Test Commands

```bash
# Quick pixel comparison
cargo run --release --example compare_xyb_decoded

# Detailed diagnostics
cargo run --release --example diagnose_xyb_progressive

# Full matrix test
cargo run --release --example comprehensive_matrix
```

## Technical Background

### XYB Color Space

XYB is a perceptual color space from JPEG XL:
- **X channel:** Color opponent axis (like Cb)
- **Y channel:** Luminance (like Y in YCbCr)
- **B channel:** Color opponent axis (like Cr), **downsampled 2x2**

XYB uses:
- sRGB → linear RGB → XYB color transform
- Scaling for JPEG sample range [0-255]
- ICC profile embedding (720 bytes) for proper decoding
- Separate quantization tables per component

### Progressive JPEG Level 2

Progressive encoding uses successive approximation:
1. **DC scan:** All DC coefficients at full precision
2. **AC first (1-2):** Low-frequency AC at full precision
3. **AC first (3-63):** High-frequency AC, top 3 bits only (ah=0, al=2)
4. **AC refine (3-63):** Bit 1 refinement (ah=2, al=1)
5. **AC refine (3-63):** Bit 0 refinement (ah=1, al=0)

Steps 3-5 repeat for each component.

### 2-Pass Huffman Optimization

XYB Progressive uses 2-pass encoding:

**Pass 1 - Tokenization:**
1. Quantize all blocks
2. Tokenize each progressive scan without writing bits
3. Collect symbol frequency statistics

**Pass 2 - Encoding:**
1. Build optimized Huffman tables from statistics
2. Write JPEG headers and Huffman tables
3. Replay tokens with optimized tables to generate bitstream

### Why This Matters

The bug suggests that somewhere between tokenization and replay, the DCT coefficients are being corrupted or encoded incorrectly. Since baseline works but progressive doesn't, and both use the same quantization, the bug must be in the progressive-specific code paths.

## Possible Root Causes (Hypotheses)

### Hypothesis 1: Incorrect Progressive Scan Script for XYB

XYB might need a different scan order or successive approximation schedule than YCbCr. Check if C++ uses the same script for XYB and YCbCr.

**Test:** Compare `get_progressive_scan_script()` output for XYB vs YCbCr.

### Hypothesis 2: Wrong Huffman Context for XYB

XYB progressive assigns all AC components to context 3 (line 1429). Maybe XYB needs component-specific contexts?

**Test:** Try using context 0/1/2 for components 0/1/2 instead of all using context 3.

### Hypothesis 3: Tokenization Bug for XYB Coefficients

The tokenization might handle XYB coefficients incorrectly (different range/scaling than YCbCr).

**Test:** Log token values during tokenization and compare with baseline encoding.

### Hypothesis 4: Replay Bug for Successive Approximation

The token replay might incorrectly handle the successive approximation bit planes for XYB.

**Test:** Add extensive logging to `replay_progressive_scan()` for XYB mode.

## Related Issues

### YCbCr Progressive - Working ✅

YCbCr progressive mode works correctly:
- Rust matches C++ within 0-7%
- SSIM2 scores are identical
- Decoded output is correct

This proves the progressive encoding infrastructure is fundamentally correct; the bug is specific to XYB.

### XYB Baseline - Working ✅

XYB Baseline mode works correctly:
- Rust matches C++ within 0-3%
- SSIM2 ~3.27 (matches C++)
- Decoded output is correct

This proves XYB color conversion, quantization, and baseline encoding are correct.

## C++ Reference Behavior

**C++ cjpegli XYB Progressive (CORRECT):**
- Baseline and Progressive produce IDENTICAL decoded output
- 0 pixels differ after decoding
- SSIM2 ~3.28 (positive, good quality)

**How to test C++:**
```bash
# Build C++ cjpegli
cd internal/jpegli-cpp/build
ninja cjpegli

# Encode with XYB Progressive
./tools/cjpegli --xyb -p 2 -q 70 input.ppm output_prog.jpg

# Encode with XYB Baseline
./tools/cjpegli --xyb -p 0 -q 70 input.ppm output_base.jpg

# Decode both and compare pixels
# (use jpeg-decoder or any standard JPEG decoder)
```

## Files Modified During Investigation

### Code Changes

- **encode.rs:1402-1435** - Added AQ, zero-bias, and aq_map to XYB Progressive
  - Status: Applied, but bug persists
  - Keep this change (it's correct even though it didn't fix the bug)

### Diagnostic Tools Created

- **examples/compare_xyb_decoded.rs** - Pixel-level comparison tool
- **examples/diagnose_xyb_progressive.rs** - Quality diagnostics
- **examples/comprehensive_matrix.rs** - Full test matrix with SSIM2

### Test Results Saved

When running diagnostics, JPEGs are saved to `/tmp/`:
- `xyb_baseline.jpg` - Correct (matches C++)
- `xyb_progressive.jpg` - Corrupted (93% wrong pixels)
- `cpp_xyb_base.jpg` - C++ baseline reference
- `cpp_xyb_prog.jpg` - C++ progressive reference

## Conclusion

The XYB Progressive bug is confirmed and partially diagnosed. Adaptive Quantization was missing and has been added, but this did not fix the issue. The bug lies somewhere in the progressive encoding pipeline after quantization - likely in tokenization, Huffman table context assignment, or token replay.

**Next investigator should:**
1. Start with Step 1 (verify quantized coefficients match)
2. Add extensive debug logging to trace where corruption occurs
3. Compare behavior with working YCbCr progressive mode
4. Check C++ source code for XYB-specific progressive handling

**Priority:** Medium-High - XYB Progressive is unusable in current state, but XYB Baseline works fine as a workaround.

---

## Update: 2026-01-04 (Later Same Day)

### Bugs Fixed

**Bug 1: Huffman Context Assignment (encode.rs:1442-1449)**
- **Problem**: All AC scans used context `num_components` (3 for XYB)
- **Fix**: Use component-specific contexts: `num_components + component_index` (3, 4, 5)
- **Matches**: YCbCr progressive (which works correctly)

**Bug 2: Scan Header Table Selectors (encode.rs:2499-2508)**
- **Problem**: All XYB components wrote `0x00` (DC table 0, AC table 0)
- **Fix**: Use luma/chroma split - component 0 uses table 0, components 1-2 use table 1
- **Matches**: YCbCr behavior and Huffman table generation

### Results After Fixes

- ✅ Progressive JPEG now decodes successfully (was failing completely before)
- ✅ Pixel corruption reduced from **93% (11,253/12,288)** to **50% (6,116/12,288)**
- ✅ First 4 pixels now match perfectly between baseline and progressive
- ✅ Quantized coefficients confirmed identical between modes

### Remaining 50% Corruption - Diagnosis

Created diagnostic tools (`analyze_pixel_differences.rs`, `test_ycbcr_progressive_pattern.rs`) which revealed:

**Checkerboard Pattern:**
```
..XXXXXX
XXXXXX..
..XXXXXX
XXXXXX..
```
- Blocks (0,0), (1,0), (0,1), (1,1) are correct (25% of blocks)
- All other blocks are corrupted (75% of blocks)

**Channel-Specific Corruption:**
- **B channel: ✅ Perfect** (0 differences!)
- **R channel: ❌ 3072 pixels corrupt**
- **G channel: ❌ 3044 pixels corrupt**

**Key Finding: YCbCr 4:4:4 Progressive Works Perfectly**
- YCbCr 4:4:4 progressive: 0 pixels differ (100% correct)
- Same tokenization/replay infrastructure as XYB
- Proves bug is XYB-specific, likely related to subsampling

### Root Cause Hypothesis: MCU Ordering Issue

XYB uses **2:2:1 subsampling** (frame header declares R:2×2, G:2×2, B:1×1):
- MCU size: 16×16 pixels
- X and Y: 2×2 blocks per MCU (64 blocks total in 8×8 grid)
- B: 1×1 block per MCU (16 blocks total in 4×4 grid)

**Theory:** Progressive non-interleaved scans should use **raster order** within each component, but the decoder may be expecting **MCU order** for components with 2×2 sampling factors.

**Supporting evidence:**
1. B channel (1×1 sampling) is **perfect** - uses simple raster order
2. X/Y channels (2×2 sampling) have **checkerboard corruption** - MCU order mismatch
3. Pattern matches MCU boundaries (every 2×2 block group)

### Next Steps

1. **Investigate C++ reference implementation:**
   - How does `cjpegli` order blocks for XYB progressive?
   - Does it use MCU order or raster order for 2×2 sampled components?
   - Check `lib/jpegli/encode.cc` progressive scan encoding

2. **Implement MCU ordering if needed:**
   - Convert block arrays from raster to MCU order before tokenization
   - OR adjust frame header sampling factors to use 1:1:1 (no subsampling)
   - Verify with C++ reference which approach is correct

3. **Test the fix:**
   - Run `compare_xyb_decoded.rs` - should show 0 pixels differ
   - Run `analyze_pixel_differences.rs` - should show no checkerboard pattern
   - Verify B channel remains perfect

---

**Investigation Date:** 2026-01-04
**Investigator:** Claude (Anthropic)
**Status:** Partially fixed - MCU ordering issue identified, requires C++ reference check
