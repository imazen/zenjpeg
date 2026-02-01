# Context Handoff: Catastrophic Encoder Regression + Trellis Analysis

## TL;DR

**The zenjpeg encoder is completely broken.** Output JPEGs decode to wrong pixels (solid yellow at Q75, random colored streaks at Q95). This is a regression caused by the SIMD refactor chain (commits `332bde1` through `9fcfe81`). The trellis quantization feature cannot be meaningfully evaluated until this is fixed.

## What We Found

### 1. Encoder Output Is Completely Wrong

Pixel-level proof (Kodak 1.png, Q75, decoded by ImageMagick/zune-jpeg/djpegli — all agree):

| Encoder | First pixel (orig=99,99,99) | Avg error | File size |
|---------|---------------------------|-----------|-----------|
| C++ jpegli | (103, 98, 102) | **4.3** | 86 KB |
| mozjpeg-rs | (101, 102, 107) | **5.1** | 74 KB |
| **zenjpeg** | **(0, 1, 2)** | **67.8** | 271 KB |

- zen_q75.jpg: **solid yellow** when displayed (confirmed by user)
- zen_q95.jpg: **random blocks of colored streaks** + ImageMagick reports "bad Huffman code" and "29504 extraneous bytes before marker"
- DSSIM is constant at ~0.2155 across ALL quality levels Q2-Q100 (quality parameter has no effect on perceptual quality)
- File sizes DO scale with quality (70KB-467KB) but pixel data is always wrong

### 2. The Comprehensive Parity Test Passes Despite 0/50 Levels Meeting Targets

`cargo test --release -p zenjpeg --test comprehensive_cpp_comparison -- --ignored` shows:
- Size: **+220%** (target: ~0%)
- DSSIM: **+8,717%** (target: ~0%)
- Butteraugli: **+1,478%** (target: ~0%)
- Quality levels within 5%: **0/50** (target: 50/50)

The test has NO assertions enforcing quality — it just prints results and returns ok.

### 3. Suspect Commits (SIMD Refactor Chain)

These commits rewrote the SIMD layer and are the most likely cause:

```
332bde1 refactor: replace unsafe_simd with magetypes in simd_types.rs and dct.rs
de546f8 refactor: migrate unsafe_simd to safe archmage #[arcane] tokens
0b1673c refactor: replace unsafe load/store with safe_unaligned_simd wrappers
79f3bcf refactor: update archmage 0.2.1→0.3.0, magetypes 0.1.0→0.3.0
f176071 refactor: eliminate all remaining unsafe blocks in zenjpeg/src
9dcb74b fix: update archmage-simd code for archmage 0.3.0 API changes
9fcfe81 refactor: consolidate SIMD features into single archmage-simd flag
```

The commit BEFORE this chain: `6d50d62` (the previous context handoff for this migration).
The last known-good state: `8a0313a` (rename baseline→sequential).

### 4. Root Cause Hypothesis

**Solid yellow** = systematic color conversion or DCT error. All pixels shifted to one color suggests:
- DCT coefficients being produced in wrong order (natural vs zigzag mismatch)
- Quantization multiplier/divisor inverted after SIMD migration
- Color space matrix (RGB→YCbCr) producing wrong coefficients
- A sign flip or scaling factor changed in the magetypes/archmage migration

**Random streaks at Q95** = Huffman/entropy stream corruption. At Q95, more non-zero coefficients survive quantization, amplifying any coefficient ordering or encoding bug.

**Quality parameter has no effect on DSSIM** = the bug dominates the signal. Quant tables change correctly (file sizes scale), but base pixel data is wrong enough that perceptual quality doesn't improve.

## How To Fix

### Step 1: Bisect the Regression

```bash
# Test at the pre-SIMD-refactor commit
git checkout 8a0313a
cargo run --release -p zenjpeg --example encode_simple -- \
    ~/work/codec-eval/codec-corpus/kodak/1.png /mnt/v/output/zenjpeg/trellis-validation/pre_simd.jpg 75
display /mnt/v/output/zenjpeg/trellis-validation/pre_simd.jpg
# If correct → bug is in 332bde1..9fcfe81
# Then bisect within that range
git checkout main
```

### Step 2: Look at DCT/Quantization SIMD Code

Key files changed in the refactor:
- `zenjpeg/src/encode/dct.rs` — forward DCT (SIMD migration to magetypes)
- `zenjpeg/src/encode/encode_simd.rs` — block extraction and quantization
- `zenjpeg/src/foundation/simd_types.rs` — SIMD type wrappers
- `zenjpeg/src/color/ycbcr.rs` — RGB↔YCbCr conversion

Check:
- Are DCT coefficients being written in the right order (natural vs zigzag)?
- Is the quantization multiplier/divisor inverted?
- Did the archmage 0.2.1→0.3.0 migration change any API semantics?
- Did `magetypes` transpose produce different layout than the old `unsafe_simd` transpose?

### Step 3: Add Regression Guard

The comprehensive parity test needs ASSERTIONS:
```rust
assert!(avg_size_diff < 5.0, "Size parity regression: {:.2}%", avg_size_diff);
assert!(avg_dssim_diff < 0.05, "DSSIM parity regression: {:.2}%", avg_dssim_diff * 100.0);
```

### Step 4: Try Disabling archmage-simd

Quick sanity check — does building WITHOUT the archmage-simd feature produce correct output?
```bash
cargo run --release -p zenjpeg --no-default-features --features "std,yuv" \
    --example encode_simple -- ~/work/codec-eval/codec-corpus/kodak/1.png /tmp/no_simd.jpg 75
display /tmp/no_simd.jpg
```
If this produces correct output, the bug is specifically in the archmage-simd code path.

## Trellis Analysis (For After Fix)

### What Works
- mozjpeg-rs 0.5.1 trellis code is structurally correct (own validation tests pass)
- Integration mechanics: DCT ×64 scaling bridges jpegli's 1/64-scale to mozjpeg's 8× divisor
- Trellis consistently produces smaller files (6-14% at Q50, 1-3% at Q95)
- Feature gating via `experimental-hybrid-trellis` works correctly
- All 6 integration tests pass

### What's Missing (For Proper mozjpeg Parity)
1. **Standard Huffman tables** used for rate estimation instead of image-specific tables
2. **No cross-block EOB optimization** — `optimize_eob_runs()` exported by mozjpeg-rs but not called
3. **No multi-pass trellis** — `num_loops` config exists but tables never updated between passes
4. **bench-utils feature propagation** — `zenjpeg-bench-utils` dev-dependency doesn't forward `experimental-hybrid-trellis`, so `quality_compare --encoder hybrid` fails at runtime

### Quality Can't Be Evaluated Until Encoder Is Fixed
Both trellis and non-trellis paths produce the same broken output. Trellis DOES reduce file size, confirming RD optimization works at the coefficient level, but quality measurement is meaningless with the base encoder broken.

## Visual Output

Output images at `/mnt/v/output/zenjpeg/trellis-validation/`:
- `original.png` — source Kodak 1
- `cpp_jpegli_q75.jpg` — C++ jpegli reference (correct, 86KB)
- `mozjpeg_rs_q75.jpg` — mozjpeg-rs reference (correct, 74KB)
- `zen_q75.jpg` — zenjpeg Q75 (**solid yellow**, 255KB)
- `zen_trellis_q75.jpg` — zenjpeg+trellis Q75 (also broken, 239KB)
- `zen_q50.jpg` — zenjpeg Q50 (broken, 195KB)
- `zen_q95.jpg` — zenjpeg Q95 (**corrupt Huffman, random streaks**, 410KB)
- `comparison_all.png` — labeled montage of all
- `zen_quality_sweep.png` — Q50/Q75/Q95 side-by-side

## Baseline Benchmarks (Pre-SIMD-Refactor, from previous handoff)

### Encode (512x512 reference)
- encode/rgb/512x512:    1.32 ms
- encode/rgb/1024x1024:  5.49 ms
- quality/q/75:          1.24 ms
- quality/q/90:          1.29 ms

### DCT (16384 blocks)
- dct/recursive:         545 µs (30.0 Melem/s)
- dct_single/recursive:  54.3 ns

### Decode
- decode/rgb/512x512:    482 µs
- decode/rgb/1024x1024:  1.88 ms
