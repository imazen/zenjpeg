# jpegli-rs Crate Refactoring Plan

## Goals
1. Split files over 2000 lines into logical modules
2. Separate responsibilities clearly
3. Enable optimal borrowing patterns for future buffer reuse
4. Make the codebase intuitive for contributors

## Current State

| File | Lines | Issue |
|------|-------|-------|
| encode.rs | 6317 | 3x over limit - monolithic |
| huffman_opt.rs | 2417 | ~20% over |
| decode.rs | 2301 | ~15% over |
| xyb.rs | 1912 | OK but large |
| encode_simd.rs | 1851 | OK |

**Current structure:** 32 flat .rs files in `src/`

---

## Final Directory Structure

```
src/
├── lib.rs                    # Public API re-exports
├── error.rs                  # Error types (226 lines, unchanged)
├── types.rs                  # Core types: JpegMode, Subsampling, Quality (502 lines)
├── pixel.rs                  # PixelFormat handling (191 lines, unchanged)
│
├── encode/                   # === ENCODER MODULE ===
│   ├── mod.rs               # Encoder struct, builder pattern, public encode()
│   ├── config.rs            # EncoderConfig, InternalPipeline, validation
│   ├── pipeline/
│   │   ├── mod.rs           # Pipeline orchestration
│   │   ├── baseline.rs      # encode_baseline_ycbcr, encode_baseline_xyb
│   │   ├── progressive.rs   # encode_progressive, encode_progressive_xyb
│   │   └── scan.rs          # Scan encoding, DC/AC passes, token replay
│   ├── color/
│   │   ├── mod.rs           # Color conversion dispatch
│   │   ├── ycbcr.rs         # RGB->YCbCr conversion (from color.rs)
│   │   ├── xyb.rs           # RGB->XYB conversion (from xyb.rs encode parts)
│   │   └── downsample.rs    # Chroma downsampling 4:2:0, 4:2:2, 4:4:0
│   ├── blocks/
│   │   ├── mod.rs           # Block operations
│   │   ├── extract.rs       # Block extraction from planes
│   │   ├── dct.rs           # Forward DCT (from dct.rs)
│   │   └── quantize.rs      # Quantization with AQ, zero-biasing
│   ├── output/
│   │   ├── mod.rs           # JPEG output
│   │   ├── markers.rs       # SOI, SOF, DQT, DHT, SOS, EOI writing
│   │   ├── tables.rs        # Quant table & Huffman table writing
│   │   └── bitstream.rs     # Entropy-coded data output
│   └── simd.rs              # SIMD optimizations (encode_simd.rs, trimmed)
│
├── decode/                   # === DECODER MODULE ===
│   ├── mod.rs               # Decoder struct, public decode()
│   ├── parser.rs            # JPEG marker parsing
│   ├── huffman.rs           # Huffman decoding
│   ├── dequant.rs           # Dequantization
│   ├── idct.rs              # Inverse DCT (from idct.rs)
│   ├── color.rs             # YCbCr->RGB, XYB->RGB
│   └── icc.rs               # ICC profile handling
│
├── quant/                    # === QUANTIZATION MODULE ===
│   ├── mod.rs               # Public quant API
│   ├── tables.rs            # QuantTable generation, BASE_QUANT_MATRIX
│   ├── aq/
│   │   ├── mod.rs           # Adaptive quantization API
│   │   ├── strength.rs      # AQ strength map computation
│   │   ├── erosion.rs       # Fuzzy erosion, pre-erosion
│   │   └── simd.rs          # SIMD AQ (adaptive_quant_simd.rs)
│   └── zero_bias.rs         # Zero-biasing parameters
│
├── huffman/                  # === HUFFMAN MODULE ===
│   ├── mod.rs               # Public Huffman API
│   ├── types.rs             # HuffmanTable, HuffmanEncodeTable
│   ├── encode.rs            # Huffman encoding
│   ├── decode.rs            # Huffman decoding
│   ├── optimize/
│   │   ├── mod.rs           # Optimization API, HuffmanOptimizer
│   │   ├── frequency.rs     # FrequencyCounter, symbol counting
│   │   ├── cluster.rs       # ClusterResult, cluster_histograms()
│   │   ├── tree.rs          # Huffman tree building
│   │   └── canonical.rs     # Canonical Huffman code generation
│   └── classic.rs           # Classic mozjpeg-style tables
│
├── entropy/                  # === ENTROPY CODING ===
│   ├── mod.rs               # EntropyEncoder, EntropyDecoder
│   ├── tokens.rs            # Token types, tokenization
│   ├── writer.rs            # BitWriter for encoding
│   └── reader.rs            # BitReader for decoding
│
├── color/                    # === SHARED COLOR UTILITIES ===
│   ├── mod.rs               # Color space utilities
│   ├── consts.rs            # BT.601 coefficients, color matrices
│   ├── transfer.rs          # sRGB gamma, linear conversion (transfer_functions.rs)
│   └── xyb_tables.rs        # XYB-specific LUTs and constants
│
├── foundation/               # === LOW-LEVEL UTILITIES ===
│   ├── mod.rs
│   ├── consts.rs            # JPEG markers, zigzag tables
│   ├── alloc.rs             # Memory allocation, fallible alloc
│   └── bitstream.rs         # Low-level bit I/O
│
├── scan_script.rs           # Progressive scan ordering (534 lines, unchanged)
├── tone_mapping.rs          # HDR tone mapping (348 lines, unchanged)
│
└── hybrid/                   # === EXPERIMENTAL (feature-gated) ===
    ├── mod.rs
    ├── trellis.rs           # Trellis quantization
    └── config.rs            # HybridConfig
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                          │
│                         &[u8] RGB pixels                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        encode/config.rs                                     │
│                   EncoderConfig validation                                  │
│            ┌──────────────┴──────────────┐                                  │
│            ▼                             ▼                                  │
│      YCbCr Mode                     XYB Mode                                │
└────────────┬─────────────────────────────┬──────────────────────────────────┘
             │                             │
             ▼                             ▼
┌────────────────────────┐    ┌────────────────────────┐
│  encode/color/ycbcr.rs │    │  encode/color/xyb.rs   │
│  RGB → YCbCr (f32)     │    │  RGB → linear → XYB    │
│  BT.601 matrix         │    │  (f32, scaled)         │
└────────────┬───────────┘    └────────────┬───────────┘
             │                             │
             ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      encode/color/downsample.rs                             │
│                                                                             │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                       │
│   │  4:4:4  │  │  4:2:2  │  │  4:2:0  │  │  4:4:0  │                       │
│   │  None   │  │  2x1    │  │  2x2    │  │  1x2    │                       │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘                       │
│                                                                             │
│   Methods: Box, BoxSmoothed, Sharp, GammaAware                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │   Y plane (f32)   │           │ Cb/Cr planes (f32)│
        │   full resolution │           │ (downsampled)     │
        └─────────┬─────────┘           └─────────┬─────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          quant/aq/                                          │
│                   Adaptive Quantization                                     │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │ pre_erosion  │ → │fuzzy_erosion │ → │  per_block   │                    │
│  │  (4x down)   │   │  (5x5 min)   │   │ modulations  │                    │
│  └──────────────┘   └──────────────┘   └──────────────┘                    │
│                                                                             │
│  Output: AQStrengthMap [f32 per 8x8 block]                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         quant/tables.rs                                     │
│                    Quantization Table Generation                            │
│                                                                             │
│  Quality → Distance → Per-frequency scaling → QuantTable [u16; 64]         │
│  + ZeroBiasParams { mul[64], offset[64] }                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        encode/blocks/                                       │
│                                                                             │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐              │
│  │  extract.rs    │ → │    dct.rs      │ → │  quantize.rs   │              │
│  │ plane→[f32;64] │   │ DCT transform  │   │ f32→i16 quant  │              │
│  │ (8x8 blocks)   │   │ (f32→f32)      │   │ +zero-biasing  │              │
│  └────────────────┘   └────────────────┘   └────────────────┘              │
│                                                                             │
│  Output: Vec<[i16; 64]> quantized blocks                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │  BASELINE PATH    │           │ PROGRESSIVE PATH  │
        │  (single scan)    │           │ (multi-scan)      │
        └─────────┬─────────┘           └─────────┬─────────┘
                  │                               │
                  ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        entropy/tokens.rs                                    │
│                         Tokenization                                        │
│                                                                             │
│  [i16; 64] → zigzag → DC diff encoding → AC run-length → Token stream      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │  Fixed Huffman    │           │ Optimized Huffman │
        │  (single pass)    │           │ (two-pass)        │
        └─────────┬─────────┘           └─────────┬─────────┘
                  │                               │
                  │                     ┌─────────┴─────────┐
                  │                     ▼                   │
                  │         ┌───────────────────┐           │
                  │         │ huffman/optimize/ │           │
                  │         │ Frequency count → │           │
                  │         │ Tree build →      │           │
                  │         │ Canonical codes   │           │
                  │         └─────────┬─────────┘           │
                  │                   │                     │
                  │                   ▼                     │
                  │         ┌───────────────────┐           │
                  │         │  Token replay     │←──────────┘
                  │         │  with optimal     │
                  │         │  tables           │
                  │         └─────────┬─────────┘
                  │                   │
                  └─────────┬─────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       entropy/writer.rs                                     │
│                        Bit Writing                                          │
│                                                                             │
│  Token → Huffman code lookup → write_bits() → byte buffer                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       encode/output/                                        │
│                      JPEG Assembly                                          │
│                                                                             │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────────┐ ┌─────┐          │
│  │ SOI │→│ APP │→│ DQT │→│ SOF │→│ DHT │→│ SOS + data  │→│ EOI │          │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────────────┘ └─────┘          │
│                                                                             │
│  (Progressive: multiple SOS + data sections)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                         │
│                          Vec<u8> JPEG                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### encode/
| File | Responsibility | Key Types |
|------|---------------|-----------|
| mod.rs | `Encoder` struct, builder pattern, `encode()` entry | `Encoder` |
| config.rs | Configuration validation, internal pipeline | `EncoderConfig`, `InternalPipeline` |
| pipeline/baseline.rs | Baseline encoding orchestration | - |
| pipeline/progressive.rs | Progressive multi-scan encoding | `ProgressiveScan` |
| pipeline/scan.rs | Individual scan encoding, token replay | - |
| color/ycbcr.rs | RGB→YCbCr with configurable method | - |
| color/xyb.rs | RGB→XYB perceptual conversion | - |
| color/downsample.rs | Chroma subsampling (box, sharp, gamma) | - |
| blocks/extract.rs | Extract 8x8 blocks from planes | - |
| blocks/dct.rs | Forward DCT transform | - |
| blocks/quantize.rs | Quantization with AQ and zero-bias | - |
| output/markers.rs | JPEG marker writing | - |
| output/tables.rs | DQT/DHT table writing | - |
| simd.rs | SIMD-optimized block operations | - |

### decode/
| File | Responsibility |
|------|---------------|
| mod.rs | `Decoder` struct, `decode()` entry |
| parser.rs | JPEG marker parsing |
| huffman.rs | Huffman table decoding |
| dequant.rs | Coefficient dequantization |
| idct.rs | Inverse DCT |
| color.rs | YCbCr/XYB→RGB conversion |
| icc.rs | ICC profile handling |

### quant/
| File | Responsibility |
|------|---------------|
| mod.rs | Public quantization API |
| tables.rs | `QuantTable` generation from quality |
| aq/strength.rs | AQ strength map computation |
| aq/erosion.rs | Pre-erosion, fuzzy erosion |
| aq/simd.rs | SIMD AQ implementations |
| zero_bias.rs | Zero-biasing parameters |

### huffman/
| File | Responsibility |
|------|---------------|
| mod.rs | Public Huffman API |
| types.rs | `HuffmanTable`, `HuffmanEncodeTable` |
| encode.rs | Huffman encoding |
| decode.rs | Huffman decoding |
| optimize/mod.rs | `HuffmanOptimizer` - two-pass optimization orchestrator |
| optimize/frequency.rs | `FrequencyCounter` - symbol frequency counting |
| optimize/cluster.rs | `ClusterResult`, `cluster_histograms()` - histogram clustering |
| optimize/tree.rs | Huffman tree construction from frequencies |
| optimize/canonical.rs | Canonical code generation (JPEG DHT format) |
| classic.rs | Pre-computed tables (mozjpeg style) |

---

## Migration Strategy

### Phase 0: Audit SIMD Branch Changes

**Net diff: +5,271 / -560 lines in src/*.rs**

| File | Change | Pattern |
|------|--------|---------|
| `encode_simd.rs` | +1761 NEW | Dedicated SIMD module |
| `adaptive_quant_simd.rs` | +1450 NEW | Dedicated SIMD module |
| `xyb.rs` | +482 | Added `*_simd` functions |
| `color.rs` | +418 | `#[cfg(feature = "simd")]` blocks |
| `dct.rs` | +377 | `mod simd` + `#[inline]` hints |
| `decode.rs` | +219 | SIMD conversion functions |
| `quant.rs` | +218 | SIMD quantization |
| `chroma.rs` | +169 | SIMD downsampling |
| `hybrid.rs` | +119 | SIMD hybrid quant |
| `encode.rs` | -124 net | Calls SIMD functions (not inline) |
| `idct.rs` | +39 | SIMD IDCT |
| Other | +small | Minor changes |

**SIMD Factoring Assessment:**

✅ **Cleanly factored (new files):**
- `encode_simd.rs` - Self-contained SIMD functions
- `adaptive_quant_simd.rs` - Self-contained SIMD AQ

✅ **Cleanly factored (cfg-gated in existing files):**
- `color.rs` - SIMD blocks wrapped in `#[cfg(feature = "simd")]`
- `dct.rs` - Separate `mod simd` submodule

✅ **Caller-side changes only:**
- `encode.rs` - Just calls `crate::encode_simd::*` and `crate::xyb::*_simd`
- No inline SIMD code in encode.rs

⚠️ **Needs review during refactor:**
- `xyb.rs` - Mixed scalar + SIMD functions (should move SIMD to `encode/color/xyb.rs`)
- `chroma.rs` - SIMD interspersed (should extract to `encode/color/downsample.rs`)

**Decision: Refactor with SIMD in place**
- SIMD code is already well-factored
- Move `encode_simd.rs` → `encode/simd.rs`
- Move `adaptive_quant_simd.rs` → `quant/aq/simd.rs`
- Extract `#[cfg(feature = "simd")]` blocks during module splits

### Phase 1: Create Directory Structure (no code changes)
```bash
mkdir -p src/{encode/{pipeline,color,blocks,output},decode,quant/aq,huffman/optimize,entropy,color,foundation,hybrid}
```

### Phase 2: Extract Foundation (lowest risk)
1. Move `consts.rs` → `foundation/consts.rs`
2. Move `alloc.rs` → `foundation/alloc.rs`
3. Move `bitstream.rs` → `foundation/bitstream.rs`
4. Update imports in all files

### Phase 3: Extract Huffman Module
1. Split `huffman_opt.rs` (2417 lines) into:
   - `huffman/optimize/mod.rs` (~300 lines) - HuffmanOptimizer, public API
   - `huffman/optimize/frequency.rs` (~350 lines) - FrequencyCounter, symbol counting
   - `huffman/optimize/cluster.rs` (~250 lines) - ClusterResult, cluster_histograms()
   - `huffman/optimize/tree.rs` (~400 lines) - Huffman tree building
   - `huffman/optimize/canonical.rs` (~300 lines) - Canonical code generation
2. Move `huffman.rs` → `huffman/encode.rs`
3. Move `huffman_types.rs` → `huffman/types.rs`
4. Move `huffman_classic.rs` → `huffman/classic.rs`
5. Create `huffman/mod.rs` with re-exports

### Phase 4: Extract Quantization Module
1. Move `quant.rs` → `quant/tables.rs`
2. Split `adaptive_quant.rs` + `adaptive_quant_simd.rs` into `quant/aq/`
3. Extract zero-bias logic to `quant/zero_bias.rs`

### Phase 5: Extract Entropy Module
1. Move `entropy.rs` → `entropy/mod.rs`
2. Extract token types to `entropy/tokens.rs`
3. Extract bit I/O to `entropy/writer.rs`, `entropy/reader.rs`

### Phase 6: Split encode.rs (largest change)
1. Extract `EncoderConfig` → `encode/config.rs` (~400 lines)
2. Extract baseline encoding → `encode/pipeline/baseline.rs` (~800 lines)
3. Extract progressive encoding → `encode/pipeline/progressive.rs` (~1000 lines)
4. Extract color conversion → `encode/color/` (~600 lines)
5. Extract block operations → `encode/blocks/` (~500 lines)
6. Extract JPEG writing → `encode/output/` (~800 lines)
7. Keep `Encoder` builder in `encode/mod.rs` (~500 lines)

### Phase 7: Split decode.rs
1. Extract parser → `decode/parser.rs`
2. Extract IDCT → `decode/idct.rs`
3. Extract color → `decode/color.rs`
4. Keep `Decoder` in `decode/mod.rs`

### Phase 8: Cleanup
1. Move remaining shared color code to `color/`
2. Move XYB tables to `color/xyb_tables.rs`
3. Update all imports
4. Run tests, fix breakages

---

## File Size Targets

| Module | Target Max Lines |
|--------|-----------------|
| Any single file | 800 |
| mod.rs files | 200 |
| Total per directory | 2000 |

---

## Borrowing Optimization Opportunities

After refactoring, these patterns will be cleaner:

1. **Buffer pools** in `encode/mod.rs`:
   ```rust
   struct EncoderBuffers {
       y_plane: Vec<f32>,
       cb_plane: Vec<f32>,
       cr_plane: Vec<f32>,
       blocks: Vec<[i16; 64]>,
       tokens: Vec<Token>,
   }
   ```

2. **Fallible allocation** in `foundation/alloc.rs`:
   ```rust
   pub fn try_alloc_f32(len: usize) -> Result<Vec<f32>, Error>
   ```

3. **Clear data flow** enables:
   - Pass `&mut Vec<f32>` instead of returning `Vec<f32>`
   - Reuse block buffers across MCUs
   - Pre-allocate based on image dimensions

---

## Progress Log

### Session 2026-01-06: Phase 6 partial

**Commit:** `34b06d0` on branch `simd`

**Completed:**

| Phase | Status | Details |
|-------|--------|---------|
| Phase 0 | ✅ Done | SIMD audit complete - code is well-factored |
| Phase 1 | ✅ Done | Directory structure created |
| Phase 2 | ✅ Done | Foundation module extracted |
| Phase 3 | ✅ Done | Huffman module fully extracted |
| Phase 4 | ✅ Done | Quantization module refactored |
| Phase 5 | ✅ Done | Entropy module split |
| Phase 6 | 🔄 Partial | Encode module started |

**Files created in encode/:**
```
src/encode/mod.rs       - Encoder struct and all encoding methods (~5900 lines)
src/encode/config.rs    - Configuration types (~350 lines)
```

**Structure:**
- encode.rs (6316 lines) converted to encode/ directory module
- Configuration types (EncoderConfig, InternalPipeline, etc.) in config.rs
- Main encoder implementation in mod.rs (still large, further splits possible)

**Next steps for Phase 6:**
- Optional: Further split encode/mod.rs into pipeline/, output/, blocks/ submodules
- Each submodule would be ~500-1000 lines for better maintainability

**Backward compatibility:**
- encode/ directory module replaces encode.rs (Rust module resolution)
- All 324 library tests pass

---

### Session 2026-01-06: Phase 5 complete

**Commit:** `970c519` on branch `simd`

**Completed:**

| Phase | Status | Details |
|-------|--------|---------|
| Phase 0 | ✅ Done | SIMD audit complete - code is well-factored |
| Phase 1 | ✅ Done | Directory structure created |
| Phase 2 | ✅ Done | Foundation module extracted |
| Phase 3 | ✅ Done | Huffman module fully extracted |
| Phase 4 | ✅ Done | Quantization module refactored |
| Phase 5 | ✅ Done | Entropy module split |

**Files created in entropy/:**
```
src/entropy/mod.rs      - shared utilities (category, decode_value, etc.) + re-exports
src/entropy/encoder.rs  - EntropyEncoder struct (~730 lines)
src/entropy/decoder.rs  - EntropyDecoder struct (~500 lines)
```

**Structure:**
- entropy.rs (1516 lines) split into 3 smaller files
- Shared utilities (category, additional_bits, decode_value) in mod.rs
- EntropyEncoder with baseline and progressive encoding in encoder.rs
- EntropyDecoder with baseline and progressive decoding in decoder.rs

**Backward compatibility:**
- entropy/ directory module replaces entropy.rs (Rust module resolution)
- No wrapper needed - lib.rs `pub mod entropy;` resolves to entropy/mod.rs
- All 324 library tests pass

---

### Session 2026-01-06: Phase 4 complete

**Commit:** `62e7446` on branch `simd`

**Completed:**

| Phase | Status | Details |
|-------|--------|---------|
| Phase 0 | ✅ Done | SIMD audit complete - code is well-factored |
| Phase 1 | ✅ Done | Directory structure created |
| Phase 2 | ✅ Done | Foundation module extracted |
| Phase 3 | ✅ Done | Huffman module fully extracted |
| Phase 4 | ✅ Done | Quantization module refactored |

**Files created in quant/aq/:**
```
src/quant/aq/mod.rs   - AQ implementation (from adaptive_quant.rs)
src/quant/aq/simd.rs  - SIMD AQ functions (from adaptive_quant_simd.rs)
```

**Updated files:**
```
src/quant.rs          - moved to src/quant/mod.rs (directory module)
src/quant/mod.rs      - added `pub mod aq;` to include aq submodule
src/adaptive_quant.rs - converted to thin re-export wrapper
src/adaptive_quant_simd.rs - converted to thin re-export wrapper
```

**Backward compatibility:**
- `adaptive_quant.rs` re-exports all types from `quant::aq`
- `adaptive_quant_simd.rs` re-exports all types from `quant::aq::simd`
- All 324 library tests pass

---

### Session 2026-01-06: Phase 3 complete

**Commit:** `055cfcc` on branch `simd`

**Completed:**

| Phase | Status | Details |
|-------|--------|---------|
| Phase 0 | ✅ Done | SIMD audit complete - code is well-factored |
| Phase 1 | ✅ Done | Directory structure created |
| Phase 2 | ✅ Done | Foundation module extracted |
| Phase 3 | ✅ Done | Huffman module fully extracted |

**Files created in huffman/optimize/:**
```
src/huffman/optimize/mod.rs        - module re-exports
src/huffman/optimize/frequency.rs  - FrequencyCounter, OptimizedTable, OptimizedHuffmanTables
src/huffman/optimize/tokens.rs     - Token, RefToken, ScanTokenInfo, TokenBuffer
src/huffman/optimize/cluster.rs    - ClusterResult, ContextConfig, cluster_histograms()
src/huffman/optimize/progressive.rs - ProgressiveTokenBuffer with tokenization methods
```

**Updated files:**
```
src/huffman/mod.rs     - added optimize submodule and re-exports
src/huffman_opt.rs     - converted to thin re-export wrapper for backward compatibility
```

**Backward compatibility:**
- `huffman_opt.rs` re-exports all types from `huffman::optimize`
- `huffman/mod.rs` re-exports optimize types at the huffman level
- All huffman-related tests pass

**Next steps:**
1. ✅ Phase 4: Extract Quantization module (`quant.rs`, `adaptive_quant*.rs`) - DONE
2. ✅ Phase 5: Extract Entropy module (`entropy.rs`) - DONE
3. Phase 6: Split encode.rs (6317 lines - the big one)
4. Phase 7: Split decode.rs (2301 lines)
5. Phase 8: Final cleanup and test

**Notes:**
- The SIMD code is well-factored and can move cleanly during later phases
- `encode_simd.rs` → `encode/simd.rs`
- `adaptive_quant_simd.rs` → `quant/aq/simd.rs`
- Backward-compatible re-exports prevent breaking changes during migration

---

### Session 2025-01-06: Phase 0-3 (partial)

**Commit:** `395571e` on branch `simd`

**Completed:**

| Phase | Status | Details |
|-------|--------|---------|
| Phase 0 | ✅ Done | SIMD audit complete - code is well-factored |
| Phase 1 | ✅ Done | Directory structure created |
| Phase 2 | ✅ Done | Foundation module extracted |
| Phase 3 | 🔄 Partial | Huffman module partially extracted |

**Files moved:**
```
src/consts.rs      → src/foundation/consts.rs
src/alloc.rs       → src/foundation/alloc.rs
src/bitstream.rs   → src/foundation/bitstream.rs
src/huffman.rs     → src/huffman/encode.rs
src/huffman_types.rs → src/huffman/types.rs
src/huffman_classic.rs → src/huffman/classic.rs
```

**New files created:**
```
src/foundation/mod.rs  - re-exports from foundation submodules
src/huffman/mod.rs     - re-exports from huffman submodules
```

**Backward compatibility:**
- `lib.rs` updated with `pub use foundation::{alloc, bitstream, consts}`
- `lib.rs` updated with `pub use huffman::{classic as huffman_classic, types as huffman_types}`
- All 324 tests pass

**Remaining for Phase 3:**
- `huffman_opt.rs` (2417 lines) NOT YET SPLIT
- Contains intertwined types: `FrequencyCounter`, `ClusterResult`, `Token`, `TokenBuffer`, `ProgressiveTokenBuffer`, `HuffmanOptimizer`
- Plan: Split into `huffman/optimize/{mod,frequency,cluster,tree,canonical}.rs`
