# jpegli-rs Refactoring Plan

Goal: Organize by encoder/decoder separation, algorithm family, and intuitive structure.
Target: No file over 2000 lines.

## Current State

58 total .rs files, 4 files over 2k lines:
- `decode/mod.rs` (3416 lines)
- `encode_simd.rs` (3176 lines)
- `xyb.rs` (2301 lines)
- `encode/strip.rs` (2125 lines)

## Guiding Principles

1. **SIMD is "the way"** - Don't segregate by SIMD vs non-SIMD (wide types work everywhere)
2. **Unsafe SIMD** - Can go in sidecar files for clarity (e.g., `foo.rs` + `foo_unsafe.rs`)
3. **Organize by**: encoder/decoder, separation of concerns, algorithm family
4. **Shared code** - Gets appropriate root-level module (color/, huffman/, etc.)

## Complete File Inventory

### Root Level (22 files)

| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `lib.rs` | 327 | Public API | Keep |
| `error.rs` | 301 | Error types | Keep |
| `types.rs` | 682 | Core types | Keep |
| `pixel.rs` | 191 | Pixel types | Keep |
| `simd_types.rs` | 594 | SIMD type aliases | Keep (or → foundation/) |
| `test_utils.rs` | 970 | Test helpers | Keep |
| `icc.rs` | 227 | ICC profiles | Keep (shared) |
| `color.rs` | 1609 | RGB↔YCbCr | → `color/ycbcr.rs` |
| `xyb.rs` | 2301 | XYB color | → `color/xyb.rs` |
| `chroma.rs` | 1281 | Chroma downsample | → `encode/chroma.rs` |
| `dct.rs` | 1729 | Forward DCT | → `encode/dct.rs` |
| `idct.rs` | 884 | Inverse DCT (f32) | → `decode/idct.rs` |
| `idct_int.rs` | 744 | Inverse DCT (int) | → `decode/idct_int.rs` |
| `encode_simd.rs` | 3176 | Mixed SIMD | **Split** (see below) |
| `transfer_functions.rs` | 484 | PQ/HLG | Keep (future HDR) |
| `tone_mapping.rs` | 348 | Tone mapping | Keep (future HDR) |
| `quality_conversion.rs` | 451 | Quality mapping | Keep or → encode/ |
| `scan_script.rs` | 534 | Scan scripts | → `encode/scan_script.rs` |
| `simplified_quant.rs` | 371 | Simple quant | Keep or → quant/ |
| `aligned_alloc.rs` | 119 | Aligned alloc | → `foundation/` |
| `adaptive_quant.rs` | 10 | Re-export stub | **Backward compat** - re-exports from quant::aq |
| `huffman_opt.rs` | 11 | Re-export stub | **Backward compat** - re-exports from huffman::optimize |

### foundation/ (4 files) - Keep as-is ✓

| File | Lines |
|------|-------|
| `mod.rs` | ~50 |
| `alloc.rs` | 768 |
| `bitstream.rs` | 680 |
| `consts.rs` | 593 |

### huffman/ (9 files) - Keep as-is ✓

| File | Lines | Notes |
|------|-------|-------|
| `mod.rs` | 33 | |
| `encode.rs` | 725 | |
| `classic.rs` | 529 | |
| `types.rs` | 490 | |
| `optimize/mod.rs` | ~50 | |
| `optimize/cluster.rs` | 286 | |
| `optimize/frequency.rs` | 403 | |
| `optimize/progressive.rs` | 1198 | Borderline, could split later |
| `optimize/tokens.rs` | 393 | |

### quant/ (4 files) - Keep as-is ✓

| File | Lines | Notes |
|------|-------|-------|
| `mod.rs` | 1612 | Borderline |
| `aq/mod.rs` | 1036 | |
| `aq/simd.rs` | 1459 | |
| `aq/streaming.rs` | 707 | |

### entropy/ (3 files) - Keep as-is ✓

| File | Lines |
|------|-------|
| `mod.rs` | 204 |
| `encoder.rs` | 1064 |
| `decoder.rs` | 770 |

### encode/ (11 files)

| File | Lines | Action |
|------|-------|--------|
| `mod.rs` | 685 | Keep |
| `config.rs` | ~200 | Keep |
| `streaming.rs` | 1403 | Keep |
| `strip.rs` | 2125 | **Split** → strip/mod.rs + strip/convert.rs |
| `blocks.rs` | 714 | Keep |
| `serialize.rs` | 672 | Keep |
| `progressive.rs` | 421 | Keep |
| `parallel.rs` | 512 | Keep |
| `linear_lut.rs` | 390 | Keep |
| `hybrid.rs` | 252 | Keep |

### hybrid/ (3 files) - Experimental feature

| File | Lines | Action |
|------|-------|--------|
| `mod.rs` | ~20 | Keep |
| `config.rs` | 488 | Keep |
| `core.rs` | 366 | Keep |

**Note**: `hybrid_config.rs` (488 lines) at root is **IDENTICAL** to `hybrid/config.rs` - delete the root copy

### decode/ (2 files)

| File | Lines | Action |
|------|-------|--------|
| `mod.rs` | 3416 | **Split** → mod.rs + parser.rs + image.rs |
| `scanline.rs` | 1300 | Keep |

## Proposed Final Structure

```
src/
├── lib.rs                    # Public API re-exports
├── error.rs                  # Error types
├── types.rs                  # Core types (JpegMode, Subsampling, etc.)
├── pixel.rs                  # Pixel format types
├── icc.rs                    # ICC profile handling (shared)
├── test_utils.rs             # Test utilities
│
├── color/                    # Color space conversions (SHARED)
│   ├── mod.rs               # Re-exports
│   ├── ycbcr.rs             # ← color.rs (RGB↔YCbCr)
│   └── xyb.rs               # ← xyb.rs + XYB decode from encode_simd.rs
│
├── foundation/               # Low-level utilities (SHARED)
│   ├── mod.rs
│   ├── alloc.rs
│   ├── bitstream.rs
│   ├── consts.rs
│   ├── simd_types.rs        # ← simd_types.rs
│   └── aligned_alloc.rs     # ← aligned_alloc.rs
│
├── huffman/                  # Huffman coding (SHARED) - keep as-is
│   └── ...
│
├── quant/                    # Quantization (SHARED) - keep as-is
│   └── ...
│
├── entropy/                  # Entropy coding (SHARED) - keep as-is
│   └── ...
│
├── encode/                   # ENCODER
│   ├── mod.rs               # Encoder struct and public API
│   ├── config.rs            # EncoderConfig
│   ├── streaming.rs         # StreamingEncoder
│   ├── strip/               # Strip-based processing
│   │   ├── mod.rs           # StripProcessor core (~1200 lines)
│   │   └── convert.rs       # Color conversion for strips (~900 lines)
│   ├── dct.rs               # ← dct.rs (forward DCT)
│   ├── chroma.rs            # ← chroma.rs (downsampling)
│   ├── color.rs             # ← encoder parts of encode_simd.rs
│   ├── blocks.rs            # Block encoding
│   ├── serialize.rs         # JPEG structure writing
│   ├── progressive.rs       # Progressive encoding
│   ├── scan_script.rs       # ← scan_script.rs
│   ├── parallel.rs          # Parallel encoding
│   ├── linear_lut.rs        # Linear LUT
│   └── hybrid.rs            # Hybrid trellis (experimental)
│
├── decode/                   # DECODER
│   ├── mod.rs               # Decoder struct and public API (~1200 lines)
│   ├── parser.rs            # JPEG parsing (JpegParser) (~1500 lines)
│   ├── image.rs             # DecodedImage, DecodedImageF32, DecodedYCbCr (~500 lines)
│   ├── idct.rs              # ← idct.rs (inverse DCT f32)
│   ├── idct_int.rs          # ← idct_int.rs (inverse DCT int)
│   └── scanline.rs          # Scanline decoding
│
├── hybrid/                   # Experimental hybrid quantization
│   ├── mod.rs
│   ├── config.rs
│   └── core.rs
│
└── (future)
    ├── transfer_functions.rs # Keep at root for now
    ├── tone_mapping.rs       # Keep at root for now
    ├── quality_conversion.rs # Keep at root for now
    └── simplified_quant.rs   # Keep at root for now
```

## Execution Order

### Phase 1: Clean up duplicates
1. [ ] Delete `hybrid_config.rs` (identical to `hybrid/config.rs`)
2. [ ] Keep `adaptive_quant.rs` and `huffman_opt.rs` (backward compat re-exports)

### Phase 2: Create `color/` module
3. [ ] Create `color/mod.rs`
4. [ ] Move `color.rs` → `color/ycbcr.rs`, update imports
5. [ ] Move `xyb.rs` → `color/xyb.rs`, update imports
6. [ ] Move XYB decode functions from `encode_simd.rs` → `color/xyb.rs`

### Phase 3: Move encoder-only code to `encode/`
7. [ ] Move `chroma.rs` → `encode/chroma.rs`
8. [ ] Move `dct.rs` → `encode/dct.rs`
9. [ ] Move `scan_script.rs` → `encode/scan_script.rs`
10. [ ] Split `encode_simd.rs` encoder parts → `encode/color.rs`
11. [ ] Delete `encode_simd.rs` (after splitting)

### Phase 4: Split large `encode/strip.rs`
12. [ ] Create `encode/strip/mod.rs` with StripProcessor core
13. [ ] Create `encode/strip/convert.rs` with color conversion methods
14. [ ] Update imports

### Phase 5: Move decoder-only code to `decode/`
15. [ ] Move `idct.rs` → `decode/idct.rs`
16. [ ] Move `idct_int.rs` → `decode/idct_int.rs`

### Phase 6: Split large `decode/mod.rs`
17. [ ] Extract `JpegParser` → `decode/parser.rs`
18. [ ] Extract `DecodedImage*` types → `decode/image.rs`
19. [ ] Update imports

### Phase 7: Foundation cleanup (optional)
20. [ ] Move `simd_types.rs` → `foundation/simd_types.rs`
21. [ ] Move `aligned_alloc.rs` → `foundation/aligned_alloc.rs`

### Phase 8: Final verification
22. [ ] Run `cargo test --release`
23. [ ] Run `cargo clippy`
24. [ ] Verify no file exceeds 2000 lines

## Dependency Notes

- `color.rs` is used by: decode/, chroma.rs → must be shared
- `xyb.rs` is used by: encode/strip.rs, chroma.rs → shared (decoder needs XYB→RGB)
- `encode_simd.rs` is used by: encode/, decode/ (XYB decode) → must split
- `chroma.rs` is used by: encode/strip.rs only → encoder-only
- `dct.rs` is used by: encode/ only → encoder-only
- `idct.rs` is used by: decode/ only → decoder-only
- `idct_int.rs` is used by: decode/ only → decoder-only

## Files That Stay at Root

These files are either:
- Core types needed everywhere (lib.rs, error.rs, types.rs, pixel.rs)
- Future/experimental features not yet integrated (transfer_functions.rs, tone_mapping.rs)
- Test infrastructure (test_utils.rs)
- Shared utilities that don't fit elsewhere (icc.rs, quality_conversion.rs, simplified_quant.rs)
