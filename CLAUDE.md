# zenjpeg Project Guide

Pure Rust port of Google's jpegli JPEG encoder/decoder from the JPEG XL project.

## API Stability Rules (CRITICAL)

**DO NOT change the public API without explicit approval:**

1. **No re-exports at crate root** - Types stay in their modules (`encoder::EncoderConfig`, not `EncoderConfig`)
2. **No new public types/functions** without approval
3. **No changes to existing function signatures**
4. **Doc links use full paths** - `[`encoder::EncoderConfig`]` not `[`EncoderConfig`]`

## Pixel Data API Rules (CRITICAL - NON-NEGOTIABLE)

**Every API that touches pixel data MUST follow these rules. NO EXCEPTIONS.**

### Type-Safe Pixel Formats (MANDATORY)

**NEVER use raw `&[u8]` or `&[f32]` for pixel data without explicit format information.**

Use the `rgb` crate types or equivalent to encode channel count, order, and bit depth.

**STRIDE IS ALWAYS REQUIRED. NO EXCEPTIONS.**

Use `imgref::ImgRef`/`ImgRefMut` (preferred) or explicit stride parameter (in pixels, not bytes):

```rust
// BEST - imgref handles stride, type encodes format
fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u8>>) -> Result<()>;
fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u16>>) -> Result<()>;
fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGBA<f32>>) -> Result<()>;
fn read_rows(&mut self, out: ImgRefMut<'_, rgb::RGB<u8>>) -> Result<usize>;

// GOOD - Explicit stride in pixels (not bytes!)
fn push_rows(&mut self, data: &[rgb::RGB<u16>], width: usize, stride_pixels: usize, count: usize) -> Result<()>;
fn read_rows(&mut self, out: &mut [rgb::RGB<u16>], width: usize, stride_pixels: usize, count: usize) -> Result<usize>;

// BAD - No stride, will break on padded buffers
fn push_rows(&mut self, data: &[rgb::RGB<u8>], count: usize) -> Result<()>;

// TERRIBLE - No format, no stride, completely ambiguous
fn push_rows(&mut self, data: &[u8], width: usize) -> Result<()>;
```

### Precision Requirements (MANDATORY)

**zenjpeg is a professional codec. 8-bit internal precision is UNACCEPTABLE.**

- **Internal processing**: 16-bit minimum, 32-bit float preferred
- **DCT/quantization**: f32 or i32, NEVER i16 for intermediates
- **Color conversion**: f32 linear light, NEVER gamma-encoded arithmetic
- **API input/output**: Support u8, u16, f32 - but DOCUMENT the precision implications

```rust
// GOOD - High precision internal, flexible external, stride via imgref
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u8>>) -> Result<()>;   // Converts to f32 internally
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u16>>) -> Result<()>; // Converts to f32 internally
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<f32>>) -> Result<()>; // Native precision

// GOOD - Explicit stride alternative
pub fn push_rows(&mut self, data: &[rgb::RGB<u16>], width: usize, stride: usize, count: usize) -> Result<()>;

// BAD - No stride, precision loss hidden
pub fn push_rows(&mut self, rows: &[u8]) -> Result<()>; // What happens to my 16-bit data? What's the stride?
```

### Color Space Awareness (MANDATORY)

**EVERY pixel-touching function must consider color space. Even if "agnostic", document it.**

Think about:
- **Working space**: What color space are we computing in? Linear? Gamma?
- **ICC profile**: Is there an embedded profile? Do we honor it?
- **CICP**: Are we writing/reading CICP tags? (primaries, transfer, matrix)
- **Primaries**: sRGB? P3? Rec.2020? BT.601 vs BT.709 YCbCr matrix?
- **Transfer function**: Linear? sRGB? PQ? HLG?

```rust
// GOOD - Color space explicit, stride via imgref
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<f32>>, colorspace: ColorSpace) -> Result<()>;

// GOOD - Explicit stride + colorspace
pub fn push_rows(&mut self, data: &[rgb::RGB<f32>], width: usize, stride: usize, count: usize, colorspace: ColorSpace) -> Result<()>;

// ACCEPTABLE - Documented assumption, still has stride
/// Assumes sRGB primaries with sRGB transfer function.
/// For wide-gamut input, convert to sRGB first or use `push_rows_with_colorspace`.
pub fn push_rows_srgb(&mut self, rows: ImgRef<'_, rgb::RGB<u8>>) -> Result<()>;

// BAD - No stride, no colorspace, no type safety
pub fn push_rows(&mut self, rows: &[u8]) -> Result<()>;
```

### Memory Architecture (MANDATORY)

**WHOLE-IMAGE BUFFERING IS FORBIDDEN. STREAMING ONLY.**

1. **No one-shot APIs** - All encode/decode must be streaming row-by-row
2. **No `Vec<Vec<T>>`** - Use flat buffers with stride, or ring buffers
3. **Borrow, don't clone** - Take `&[T]` not `Vec<T>`, return into caller's buffer
4. **Fallible allocation** - Use `try_reserve()`, return `Result` on OOM
5. **Buffer pool friendly** - Caller provides target buffers, we write into them
6. **Stride always explicit** - Via `imgref` or parameter

```rust
// GOOD - Streaming, borrows, stride via imgref, writes to caller's buffer
pub fn push_rows(&mut self, rows: ImgRef<'_, rgb::RGB<u16>>) -> Result<()>;
pub fn read_rows(&mut self, out: ImgRefMut<'_, rgb::RGB<u16>>) -> Result<usize>;

// GOOD - Explicit stride, caller's buffer
pub fn push_rows(&mut self, data: &[rgb::RGB<u16>], width: usize, stride: usize, count: usize) -> Result<()>;
pub fn read_rows(&mut self, out: &mut [rgb::RGB<u16>], width: usize, stride: usize, max_rows: usize) -> Result<usize>;

// GOOD - Caller provides output buffer, no internal allocation
pub fn finish_into(self, output: &mut Vec<u8>) -> Result<()>;

// BAD - Whole image, allocates, clones, no stride
pub fn encode(image: Vec<Vec<u8>>) -> Vec<u8>;

// BAD - Hidden allocation, not fallible, no stride
pub fn decode(&self) -> Vec<u8>;  // What if it's 100MP? OOM panic!
```

### Ring Buffer / Pool Architecture

Design for buffer reuse. Caller owns all buffers, encoder/decoder writes into them:

```rust
// GOOD - Caller owns input buffer with explicit stride
let mut encoder = StreamingEncoder::new(width, height, config)?;
let stride = (width + 15) & !15;  // Align to 16 pixels for SIMD
let mut row_buf = vec![rgb::RGB::<u16>::default(); stride * batch_size];

for chunk in source.chunks(batch_size) {
    // Copy into caller's buffer (or read directly if source is strided)
    for (i, row) in chunk.iter().enumerate() {
        row_buf[i * stride..(i * stride + width)].copy_from_slice(row);
    }
    encoder.push_rows(&row_buf, width, stride, chunk.len())?;
}

// GOOD - Caller owns output buffer
let mut output = Vec::new();
output.try_reserve(estimated_size)?;  // Fallible!
encoder.finish_into(&mut output)?;

// GOOD - Decoder writes into caller's buffer with stride
let mut decoder = StreamingDecoder::new(&jpeg_data)?;
let out_stride = (decoder.width() + 15) & !15;
let mut out_buf = vec![rgb::RGB::<u8>::default(); out_stride * batch_size];

while !decoder.is_finished() {
    let rows_read = decoder.read_rows(&mut out_buf, decoder.width(), out_stride, batch_size)?;
    process_rows(&out_buf[..rows_read * out_stride]);
}
```

### Summary: The Non-Negotiables

| Rule | Requirement |
|------|-------------|
| **Pixel format** | Type-safe (`rgb::RGB<T>`, `rgb::RGBA<T>`) - NEVER raw `&[u8]` |
| **Stride** | ALWAYS via `imgref` or explicit parameter (pixels, not bytes) - NEVER omitted |
| **Precision** | 16-32 bit internal - NEVER 8-bit arithmetic on pixels |
| **Color space** | Explicit parameter or documented assumption - NEVER silently ignored |
| **Streaming** | Row-by-row only - NEVER whole-image buffering |
| **Allocation** | Fallible (`try_reserve`) - NEVER panic on OOM |
| **Ownership** | Borrow input (`&[T]`), write to caller's output (`&mut [T]`) - NEVER clone |
| **Target buffers** | Caller provides output buffers - NEVER allocate internally when avoidable |

**If you're about to write an API that omits stride, STOP. Add stride. Always.**

**If you're about to write an API that allocates output, STOP. Take a target buffer.**

**If you're about to write an API with raw bytes, STOP. Use `rgb::RGB<T>` or similar.**

## Performance Rules (CRITICAL)

**Never compromise sequential performance for parallel gains:**

1. Sequential encoding is the default and most common path
2. Parallel refactors must not add overhead to the sequential path
3. Use `#[cfg(feature = "parallel")]` to isolate parallel-only code
4. Benchmark both paths before and after changes
5. Frequency counting and other setup work can be parallelized separately from encoding

## Context Preservation (CRITICAL)

**You may lose context at any time.** Always record findings immediately:

1. **Suspected bugs** → Add to "Known Bugs" section below with file:line references
2. **Code analysis/details** → Save to `CODE.md` with:
   - Relevant code snippets
   - Root cause analysis
   - Proposed fixes
   - C++ reference behavior
3. **Investigation progress** → Commit WIP notes before switching tasks

**Do this BEFORE continuing investigation.** Lost context = wasted work.

## Quick Start

```bash
cargo build --release
cargo test --release
```

## C++ Parity Verification (IMPORTANT)

All features enabled by default. Tests auto-find corpus at `~/work/codec-eval/codec-corpus/`.

### Quick Parity Check
```bash
# Comprehensive test: 10 images × 50 quality levels (live cjpegli FFI)
cargo test --release -p zenjpeg --test comprehensive_cpp_comparison -- --nocapture --ignored
```

Expected results (using `jpegli_set_distance` for 3-table parity):
- **Size**: ~0% (within ±1%)
- **DSSIM**: ~0% (within ±1%)
- **Butteraugli**: ~0% (within ±1%)

**Note:** Results now use distance-based encoding (`jpegli_set_distance()`) instead
of `jpeg_set_quality()`. This ensures both encoders use 3 quant tables (Y, Cb, Cr).
Previous results showed ~4% differences due to `jpeg_set_quality()` using 2 tables.

### All C++ Comparison Tests
```bash
# Run ALL comparison tests with live C++ FFI
cargo test --release -p zenjpeg -- comparison --nocapture --ignored

# Corpus-based comparison (CID22 images)
cargo test --release -p zenjpeg --test corpus_cpp_comparison -- --nocapture --ignored

# XYB mode comparison (larger differences expected: 0.2-3%)
cargo run --release --example xyb_parity_test

# SSIMULACRA2 comparison (synthetic images)
cargo run --release --example ssim2_comparison
```

### Key Parity Tests
| Test | Command | Expected |
|------|---------|----------|
| comprehensive | `--test comprehensive_cpp_comparison` | Size ~0%, DSSIM ~0% |
| corpus | `--test corpus_cpp_comparison` | Size ~0% |
| xyb | `--example xyb_parity_test` | Size 0.2-3% |
| locked | `--test cpp_parity_locked` | Hash-locked values |
| strip edges | `--test strip_edge_cpp_comparison` | DSSIM <0.6% diff |

## Tools

```bash
cargo run --release --example jpeg_inspect -- --validate image.jpg
```

## Project Structure

```
zenjpeg/
├── zenjpeg/           # Main library crate
│   ├── src/             # Encoder, decoder, color conversion
│   ├── examples/        # Debugging tools (see examples/README.md)
│   ├── tests/           # Integration tests
│   └── benches/         # Criterion benchmarks
├── zenjpeg-bench-utils/  # Shared utilities for benchmarks/examples
├── internal/jpegli-cpp/ # C++ jpegli submodule (for parity testing)
└── docs/                # Additional documentation
```

## Examples & Debugging Tools

See **`zenjpeg/examples/README.md`** for complete documentation.

### Most Useful Tools

| Tool | Purpose |
|------|---------|
| `jpeg_inspect --validate` | Validate JPEGs with multiple decoders |
| `jpeg_inspect --all` | Full JPEG structure analysis |
| `quality_compare` | Compare encoder quality/size metrics |
| `xyb_parity_test` | Compare Rust vs C++ XYB output |
| `test_libjpeg_compat` | Verify libjpeg decoder compatibility |
| `edge_mcu_parity` | Test partial MCU edge handling parity |

### Edge MCU Parity Testing

The `edge_mcu_parity` example and `jpegli_bench_utils` provide tools for testing
edge-case handling in images with non-8-aligned dimensions:

```rust
use jpegli_bench_utils::{tile_edge_columns, create_edge_mcu_test_image, McuEdgeInfo};

// Analyze image for edge characteristics
let info = McuEdgeInfo::analyze(1118, 1105);
// partial_mcu_width = 6, affected_block_pct = 0.71%

// Create test image that amplifies edge bugs
let tiled = tile_edge_columns(&source, 6, 518);
// Now 100% of content comes from edge strip
```

Use `--edge-width N` to test specific column widths (1-7).

### XYB Debugging (Size Gap Investigation)

XYB produces 5-11% larger files than C++ jpegli at equivalent quality (quality is identical):

```bash
# Compare file sizes and DSSIM
cargo run --release --example xyb_parity_test

# Check XYB conversion precision
cargo run --release --example xyb_ulp_parity

# Compare butteraugli scores
cargo run --release --example xyb_cpp_comparison

# Compare XYB vs YCbCr quality
cargo run --release --example xyb_vs_ycbcr_butteraugli
```

## Profiling Results (4K image, 2026-01-21)

Top encoder hotspots: AQ 14.5%, entropy encoding 11.7%, DCT+quant 8.6%, deringing 6.9%.
SIMD sorting network cut AQ from 27% to 14.5% (1.86x faster). Entropy encoding and
DCT+quantization already have parallel paths; AQ is not worth parallelizing (too fine-grained).
See `docs/TUNING_HISTORY.md` for full function-level flamegraph breakdown.

## Decoder Profiling (512x512 image, 2026-01-22)

Decoder went from 60.3M to 40.5M instructions (33% reduction, 1.17x vs zune-jpeg) via AVX2
upsampling (-23%) and AVX2 IDCT. Key lesson: tiered 4x4/8x8 IDCT was counterproductive --
always use AVX2 8x8 with built-in DC-only check. Baseline 512x512 decoding: 1.31ms -> 456us (65% faster).
See `docs/TUNING_HISTORY.md` for callgrind breakdown and per-function instruction counts.

## C++ Performance Gap (2026-02-04)

Rust is **~15-20% slower** than C++ jpegli (1.15x-1.2x median). Quality is effectively
identical (size +0.63%, DSSIM +0.41%, Butteraugli +0.19% mean).

**AVX-512 additions (2026-02-04):** Added `mage_pre_erosion_row_padded_v4()` with X64V4Token
dispatch - processes 16 pixels per iteration vs 8 for AVX2. Shows ~10% improvement in quick
benchmarks. DCT dual-block AVX-512 was attempted but failed (2.3x slower due to register
lane crossing overhead).

Remaining gap from C++ Highway AVX-512 vs Rust `wide` AVX2 in DCT (~8.6% of encode). Other
hot paths: pre_erosion (now AVX-512), per_block_modulations (8x8 block structure limits
AVX-512 benefit), entropy encoding (inherently serial). Allocations reduced 84% (33K to 5K
per 10 encodes). Run: `cargo bench --bench cpp_comparison`.
See `docs/TUNING_HISTORY.md` for allocation details, SIMD analysis, and autovectorization benchmarks.

## WASM SIMD128 Performance (2026-01-27)

WASM SIMD128 gives **1.6-1.7x encode** and **1.5-2.0x decode** speedup over scalar. The `wide`
crate f32x4 maps directly to v128 operations. Explicit WASM intrinsics only improve DCT transpose
by 7% (not worth the complexity). Build with `RUSTFLAGS="-C target-feature=+simd128"`.
Run: `just wasm-bench`. See `docs/TUNING_HISTORY.md` for full benchmark tables and intrinsics investigation.

## Decoder Performance (2026-02-06)

Scanline decoder matches or beats zune-jpeg. Buffered fast mode within 15%.
Progressive beats zune at ≤1024, within 9% at 4096.

**Wall-clock progressive (commit aba9777):**
| Size | zune-jpeg | zenjpeg prog | ratio | zenjpeg fast | ratio |
|------|-----------|-------------|-------|-------------|-------|
| 256 | 247µs | 164µs | **0.66x** | 143µs | **0.58x** |
| 512 | 914µs | 595µs | **0.65x** | 532µs | **0.58x** |
| 1024 | 2.68ms | 2.16ms | **0.81x** | 1.96ms | **0.73x** |
| 2048 | 8.85ms | 9.16ms | 1.03x | 8.36ms | **0.94x** |
| 4096 | 91.3ms | 99.1ms | 1.09x | 97.7ms | 1.07x |

**Wall-clock baseline/scanline (2048x2048, commit 4ae7ed6):**
| Mode | zune-jpeg | zenjpeg | Ratio |
|------|-----------|---------|-------|
| Baseline | 4.09ms | 5.51ms | 1.35x |
| Baseline-fast | 4.09ms | 4.72ms | 1.15x |
| **Scanline-420** | **4.09ms** | **4.03ms** | **0.99x** |
| **Baseline-444** | **6.34ms** | **5.64ms** | **0.89x** |
| **Scanline-444** | **6.34ms** | **5.78ms** | **0.91x** |

**Optimizations applied:**
1. Fused box-filter 4:2:0 upsample + YCbCr→RGB AVX2 kernel (`color/ycbcr.rs`)
2. Partial dequantize based on coeff_count (skip zero coefficients)
3. DC-only fast path bypassing dequant buffer entirely
4. Marker-based ICC profile scanning (was byte-by-byte O(n) scan)
5. Force-inline hot path BitReader and Huffman functions
6. 16-bit peek Huffman slow path with pre-shifted maxcode table
7. Fast AC refinement bit reads via `read_bit_refine()` (no ScanRead enum, no bit_buffer sync)
8. Natural-order dequant with sequential writes (30% reduction in dequant instructions)
9. AVX-512 dispatch for YCbCr→RGB (correct but no measurable benefit on Zen 4)
10. Nonzero coefficient bitmap (u64 per block) for AC refinement — skip zero
    positions via `trailing_zeros()`. 12% instruction reduction in AC refine.

**Fast mode** (`fancy_upsampling(false)`): Uses box-filter upsampling fused with
color conversion instead of bilinear. 5-10% faster, minimal quality difference.

### Parallel Decode (2026-02-14, `--features parallel`)

Fused parallel decode: entropy decode + IDCT + color convert in one pass per
restart segment using rayon. Requires MCU-row-aligned DRI (default `restart_mcu_rows=4`).

**Wall-clock baseline 4:2:0 + Triangle (default), commit 872561d:**

| Size | mozjpeg | zune | zenjpeg parallel | vs mozjpeg | vs sequential |
|------|---------|------|-----------------|------------|---------------|
| 256 | 207µs | 186µs | 184µs | 1.1x | ~1.0x |
| 512 | 720µs | 779µs | 683µs | 1.1x | 1.6x |
| 1024 | 3.01ms | 3.08ms | 1.35ms | **2.2x** | **2.3x** |
| 2048 | 13.31ms | 12.91ms | 2.69ms | **4.9x** | **4.8x** |
| 4096 | 78.1ms | 78.7ms | 38.5ms | **2.0x** | N/A |

Three fused paths: 4:4:4/gray (single-pass), 4:2:0+box (single-pass), 4:2:0+fancy
(single-pass with double-buffered extended chroma strips + boundary fixup). All
produce byte-identical output to sequential path (16 hash-lock tests verify this).

Previous broken approach (two-phase full-image planes) was 2-2.5x *slower* than
sequential. Fixed by replacing with strip-based single-pass in commit 872561d.

### Dequantization Bias (2026-02-06)

`Decoder::new().dequant_bias(true)` enables Laplacian dequantization biases
(Price & Rabbani 2000). Computes per-coefficient biases from DCT statistics
and applies them during f32 dequantization. Bypasses fast i16 IDCT path.
Default: off.

**Frymire quality sweep** (1118x1105 photograph, baseline 4:2:0, commit 86e3bef):

| Q | bytes | zenjpeg | zen+bias | cjpegli | zune-jpeg | bias-zen | bias-cpp | maxdif |
|---|-------|---------|----------|---------|-----------|----------|----------|--------|
| 10 | 116K | 5.25 | 1.88 | 1.99 | 5.25 | -3.37 | -0.11 | 1 |
| 20 | 171K | 21.35 | 18.54 | 18.61 | 21.35 | -2.82 | -0.07 | 1 |
| 30 | 219K | 30.59 | 28.57 | 28.59 | 30.59 | -2.03 | -0.03 | 1 |
| 40 | 243K | 34.21 | 32.37 | 32.44 | 34.21 | -1.84 | -0.07 | 1 |
| 50 | 271K | 37.28 | 35.95 | 36.01 | 37.28 | -1.32 | -0.06 | 1 |
| 60 | 309K | 41.07 | 40.07 | 40.10 | 41.07 | -0.99 | -0.03 | 1 |
| 70 | 362K | 45.00 | 44.24 | 44.31 | 45.00 | -0.76 | -0.07 | 1 |
| 80 | 438K | 48.72 | 48.25 | 48.32 | 48.72 | -0.47 | -0.07 | 1 |
| 85 | 494K | 50.45 | 50.18 | 50.21 | 50.45 | -0.27 | -0.03 | 1 |
| 90 | 583K | 51.94 | 51.81 | 51.83 | 51.94 | -0.14 | -0.02 | 1 |
| 95 | 742K | 53.28 | 53.25 | 53.27 | 53.28 | -0.03 | -0.02 | 1 |
| 97 | 848K | 53.71 | 53.68 | 53.73 | 53.71 | -0.03 | -0.05 | 1 |
| 99 | 1034K | 54.00 | 54.03 | 54.07 | 54.00 | +0.03 | -0.05 | 1 |

**CID22 mean** (10 images, 512px, baseline 4:2:0):

| Q | zenjpeg | zen+bias | cjpegli | zune-jpeg | bias-zen | bias-cpp |
|---|---------|----------|---------|-----------|----------|----------|
| 50 | 65.23 | 65.03 | 65.07 | 65.23 | -0.21 | -0.05 |
| 75 | 75.05 | 75.19 | 75.24 | 75.05 | +0.14 | -0.06 |
| 85 | 79.85 | 80.17 | 80.22 | 79.85 | +0.33 | -0.04 |
| 95 | 86.65 | 87.05 | 87.11 | 86.65 | +0.39 | -0.06 |

- `bias-zen`: SSIM2 pt gain over default (positive = better)
- `bias-cpp`: SSIM2 pt gap vs C++ jpegli (negative = C++ better)
- `maxdif`: max pixel diff between zen+bias and cjpegli

**Pairwise SSIMULACRA2** (between decoders, Q85, 6 CID22 images):

| | zenjpeg | zen+bias | cjpegli | zune-jpeg |
|---|---------|----------|---------|-----------|
| zenjpeg | - | 91.39 | 91.42 | 100.00 |
| zen+bias | | - | 94.31 | 91.23 |
| cjpegli | | | - | 91.26 |

**Key findings:**
- zenjpeg default == zune-jpeg (identical output, both integer IDCT)
- zen+bias↔cjpegli similarity: 94.31 vs default↔cjpegli 91.42 (3 pts closer)
- Max pixel diff between zen+bias and cjpegli: always 1 (IDCT rounding only)
- **Image-dependent quality tradeoff**: on CID22 (small, diverse), bias helps
  +0.14 to +0.39 at Q75+. On frymire (large photograph), default integer IDCT
  wins by 0.03-3.37 pts across all qualities. Bias only breaks even at Q99.
- C++ jpegli shows the same pattern: also behind integer IDCT on frymire.
  The f32 IDCT + bias path and integer IDCT path have different rounding
  characteristics; which wins depends on image content.
- bias-cpp gap is consistently tiny (0.02-0.11 pts), confirming zen+bias
  closely matches C++ jpegli decoder behavior regardless of image.

Run: `cargo test --release -p zenjpeg --test dequant_bias_comparison --features decoder -- --nocapture --ignored`

### Remaining Bottlenecks

**Buffered decoder 1.15x gap** (baseline-fast vs zune):
- Two-pass architecture (store all coefficients → separate output pass) causes
  extra cache misses vs zune's inline IDCT-during-decode approach
- Scanline decoder avoids this, which is why it matches/beats zune

**Progressive 1.09x gap at 4096x4096** (down from 1.92x originally):
- AC refinement still dominates at 67% of decode instructions (289M/~430M at 2048)
- Already heavily optimized: bitmap skip, `read_bit_refine()`, all-zeros early exit
- Size-dependent gap: at ≤1024, coefficient data fits L2 cache (two-pass is fine);
  at 4096+, coefficient data exceeds L2, causing cache misses in output pass
- Remaining overhead is Huffman decode interspersed with refinement bits (inherently serial)
- The tokenize_ac_refinement_scan encoder-side function also iterates coefficients
  similarly but is not part of the decode path

## Failed Explorations

### Parallel AQ (2026-01-17)

**Attempted:** Parallelize `per_block_modulations_row` using rayon.

**Why it failed:**
- Per-block AQ computation takes ~0.2 microseconds
- Far too small for rayon thread pool overhead to be worthwhile
- 4K benchmark with threshold=256: **5x slower** than sequential
- Even 8K (33M pixels, 518K blocks) wouldn't benefit

**Analysis:**
- AQ takes 26% of 8K encode time, but only ~15% is parallelizable
- `pre_erosion_row` (6%) has row-to-row accumulation dependency
- `fuzzy_erosion` (5%) needs 3x3 neighborhood lookahead
- Max theoretical speedup with 4 threads: ~10% overall
- After rayon overhead: ~6% realistic gain - not worth complexity

**Conclusion:** The SIMD-optimized sequential path is already efficient. Thread-level parallelism would need coarser granularity (e.g., multiple iMCU rows buffered) to overcome overhead, which conflicts with the streaming architecture.

### Fuzzy Erosion SIMD (2026-01-21)

**Attempted:** SIMD-optimize `compute_fuzzy_erosion_row_into` with archmage.

**Approaches tried:**
1. **Archmage with helper functions**: Created `mage_compute_fuzzy_erosion_row` with separate
   `weighted_4_smallest`, `gather_3x3_clamped`, `gather_3x3_interior` helpers.
   Result: **3x slower** (67ms vs 53ms) - `#[arcane]` prevents inlining, causing YMM register
   spills at every function call boundary.

2. **Massive inlined function**: ~350 lines with all 4 corners fully unrolled, no helper calls.
   Result: **Still slower** (68ms vs 53ms) - instruction cache pressure from code bloat.

**Root cause analysis:**
- The algorithm requires finding 4 smallest from 9 values with index tracking
- Scalar partial sort: `find min → replace with MAX → repeat 4×`
- This creates unpredictable branch patterns that SIMD doesn't help
- Code bloat from unrolling hurts icache more than SIMD helps

**What would actually help:**
- True SIMD sorting network (e.g., bitonic sort for 16 elements)
- Would need to process multiple blocks in parallel, not just vectorize one block
- Complexity not justified for ~5% of encode time

**Files:** `zenjpeg/src/quant/aq/simd.rs:1377` (massive version, unused),
`zenjpeg/src/quant/aq/streaming.rs:631` (original scalar, in use)

### AVX-512 Dual-Block DCT (2026-01-21)

**Attempted:** Process two 8x8 blocks simultaneously using AVX-512 (512-bit = 16 floats = 2 blocks).

**Implementation:** Pack two blocks into ZMM registers [A_row_i, B_row_i], do DCT butterflies
with AVX-512 arithmetic, transpose with extract/AVX2/insert pattern.

**Benchmark results:**
- AVX2 single-block: 41.19M blocks/sec
- AVX-512 dual-block: 17.58M blocks/sec (2.3x **slower**)

**Why it failed:**
1. **8x8 blocks fit AVX2 perfectly** - 8 floats = 256 bits, no wasted register space
2. **Transpose cannot be done natively in AVX-512** - `_mm512_unpacklo_ps` operates on 128-bit
   lanes, mixing data between blocks A and B
3. **Extract/insert workaround is expensive** - each transpose requires:
   - 8 `_mm512_extractf32x8_ps` to split ZMM→YMM
   - 48 AVX2 operations (two 8x8 transposes)
   - 16 `_mm512_insertf32x8` to recombine YMM→ZMM
4. **Two transposes per DCT** = 64 extra extract/insert operations
5. **AVX-512 frequency throttling** on some CPUs adds further penalty

**Conclusion:** AVX-512 benefits require naturally 16-wide workloads. 8x8 DCT is inherently
8-wide, making AVX2 the optimal register width. Dual-block packing just adds overhead.

**Files:** `zenjpeg/src/encode/mage_simd.rs:600-775` (kept for reference, not used in encoder)

### Decoder Zero-Copy Architecture (2026-01-22) - IMPLEMENTED

**Problem:** Original decoder returned `([i16; 64], u8)` by value, copying 128 bytes per block.
Smart zeroing alone didn't help because copy dominated memory bandwidth.

**Solution:** Zero-copy `decode_block_into` API where caller provides reusable buffer:
```rust
fn decode_block_into(
    &mut self,
    coeffs: &mut [i16; 64],      // Caller-provided buffer
    prev_coeff_count: u8,        // Zeroing hint from previous block
    component: usize,
    dc_table_idx: usize,
    ac_table_idx: usize,
) -> ScanResult<u8>              // Returns new coeff count
```

**Key insight:** Reusable buffers accumulate state from ALL previous blocks, not just the
immediately previous one. If block N-2 wrote to position X, block N-1 didn't, and block N
doesn't either, position X still has stale data. Fix: track MAXIMUM coefficient count since
last restart marker, not just previous block's count.

**Implementation:**
- `entropy/decoder.rs`: Added `decode_block_into` with smart zeroing
- `decode/parser.rs`: Added `prev_coeff_counts: [u8; 4]` per-component tracking
- `decode/scanline.rs`: Added `coeffs_buf` reusable buffer and max-tracking

**Results (2026-01-22):**
- 512x512: ~5% improvement
- 2048x2048: 6.5% improvement (17.9ms vs 19.2ms)

Memory bandwidth reduction per block:
- Before: 128 bytes zeroing + 128 bytes copy = 256 bytes
- After: ~20 bytes targeted zeroing + 0 bytes copy = ~20 bytes

## Investigation Notes

**DCT Coefficient Parity (VERIFIED):** Coefficient differences are normal +/-1 rounding from
SIMD float precision, not systematic bugs. AQ maps are 100% identical between Rust and C++
when using matching quant tables (`jpegli_set_distance()`). Remaining ~0.2% size difference
is solely from DCT rounding. Use `separate_chroma_tables(true)` for 3-table mode matching C++.

**Hybrid Trellis:** Negative coupling (`-4.0`) produces ~2% smaller files with ~3% DSSIM
degradation on photos. Use `aq_trellis_max_adjustment=1.0` for screenshots (prevents 552%
quality destruction). Texture-adaptive coupling via `adaptive_config(aq_mean, aq_std)` is
recommended. For optimizers, key parameters: `tables.quant` (192 vals), `lambda_log_scale1/2`,
`zero_bias_mul` (jpegli only). Run: `cargo run --release --example hybrid_parameter_sweep`.

See `docs/TUNING_HISTORY.md` for full investigation details: DCT parity verification, AQ map
comparison, visual diff interpretation, hybrid trellis R-D analysis, ExpertConfig parameter
sensitivity tables, and preset baselines.

## Known Bugs

1. **Trellis dead parameters (2026-02-02)** - Measured via parameter sensitivity test:
   - `trellis_use_lambda_weight_tbl`: Always uses flat 1/q² weights (`encode/trellis/ac.rs:52`)
   - `trellis_num_loops`: Stored but never read — single-pass only
   - `trellis_speed_mode`: Only affects search bounds, not output (same optimum found)
   - EOB optimization: deleted (was broken, destroyed quality). See commit history.
   - See `encode/search.rs` test `test_parameter_sensitivity` for measurements

2. **SA-optimized tables non-monotonic (2026-02-03)** - `optimized_tables.rs` anchor tables
   are non-monotonic between quality levels. Luma DC: q90=5, q95=37, q100=6. Each anchor
   was independently SA-optimized, finding different local optima. Results: ~10-20 SSIM2
   points worse than JpegliProg at matched BPP, non-monotonic BPP vs quality.
   - Root cause: Independent per-anchor SA without monotonicity constraints
   - Impact: Feature unusable as-is. Would need constrained optimization or post-smoothing.
   - Verified: `cargo run --release -p zenjpeg --example knobs_vs_jpegli --features optimized-tables`

3. **Parallel feature non-deterministic output (2026-02-06)** - `locked_values` test fails
   with `--features parallel` because the parallel encoding path produces slightly different
   output due to threading non-determinism. The locked hashes are generated with default
   (sequential) features. This is expected behavior for parallel encoding, but the test
   should either be skipped with parallel or have separate locked values.
   - Impact: `cargo test --release --all-features --test locked_values` fails
   - Workaround: Run without `parallel` feature, or skip the test

4. **Grayscale scanline reader panic (FIXED 2026-02-06)** - Streaming scanline reader methods
   (`read_rows_rgb8`, `read_rows_rgbx8`, `read_rows_rgba_f32`, `read_rows_ycbcr_planes`)
   panicked on grayscale (1-component) images because they called `row_planes()` which
   requires cb/cr buffers that are empty for grayscale. Fixed: commit be24fac.

### Fixed Bugs (historical reference)

- **4:2:0 scanline chroma upsampling at MCU bottom boundaries (FIXED 2026-02-09, commit bd0f8d7)** -
  Bilinear chroma upsampler used edge replication at MCU row bottom boundaries (max ~43
  pixel error for streaming, ~57 for coefficient/transform path). Fix: mirror the existing
  top-boundary fixup for the bottom edge. Coefficient path peeks ahead by IDCT'ing the first
  chroma block row of the next MCU. Streaming path pre-decodes the next MCU row and serves
  corrected chroma through deferred buffers. Boundary max diff now ≤4 (IDCT rounding only).

- **Scanline h2v2 boundary fixup buffer overflow (FIXED 2026-02-09, commit 8f1295f)** -
  `fixup_h2v2_row0()` used hardcoded `[i16; 4096]` stack buffers, panicking on any 4:2:0
  image wider than 8192px (chroma width > 4096). Fix: borrow disjoint struct fields directly
  instead of copying to temp buffers. Closes #1.
- **Progressive MCU-padded storage (FIXED 2026-02-09, commit 29d6d81)** - Progressive decoder
  allocated coefficients with component-based counts (ceil(scaled_w/8)) but output path reads
  with MCU-padded stride (mcu_cols * h_samp). For 4:2:0 with non-MCU-aligned width, caused
  1-block-per-row shift accumulating to max_diff=255. Affected ~20/543 web corpus files.
- **Progressive interleaved DC scan padding (FIXED 2026-02-09, commit 759a4a7)** - Skipping
  entropy data for out-of-bounds MCU padding blocks desynchronized Huffman decoder. Caused
  "invalid Huffman code" parse errors on 80/543 progressive 4:2:0 files.
- See `docs/TUNING_HISTORY.md` for older fixed bugs.

## Planned Features / TODO

### Make archmage-simd mandatory (not a feature flag)

Move `archmage`, `magetypes`, and `safe_unaligned_simd` from optional to required dependencies.
Remove the `archmage-simd` feature flag and all `#[cfg(feature = "archmage-simd")]` gates.
SIMD should always be compiled in — there's no reason to support a non-SIMD build.

### Needs Heavy Analysis: CMA-ES auto_optimize() (2026-02-04)

Merged from `feat/formula-optimization`. Adds `EncoderConfig::auto_optimize()` with CMA-ES
butteraugli-optimized scaling parameters:
- `OPTIMIZED_GLOBAL_SCALE` = 5.608994 (4:2:0), 5.101017 (4:4:4)
- `OPTIMIZED_FREQUENCY_EXPONENT[64]` — per-frequency non-linear scaling
- Quality-gated: q70+ for 4:2:0, q50+ for 4:4:4
- Claimed holdout: +0.46 mean Pareto, 76% wins

**TODO:**
- [x] Add `auto_optimize()` to `knobs_vs_jpegli` R-D comparison — commit 52d921c
- [x] Compare vs HybMax-L14.5: **identical** (auto_optimize uses HybMax-L14.5 internally)
- [x] Verify on CID22 corpus: confirmed +2-3 SSIM2 over cjpegli, +0.8-1.0 over JpegliProg
- [x] Test interaction with trellis: **exclusive, not stacking** — auto_optimize() sets
  hybrid_config and clears standalone trellis. If user calls `.trellis()` after
  `.auto_optimize(true)`, standalone trellis wins and hybrid is bypassed (streaming.rs:260-268).
- [x] Document recommended usage pattern (see below)

**R-D comparison results (gb82, 25 images, 2026-02-06):**

| BPP | cjpegli-444 | JpegliProg | AutoOptimize | HybMax-L14.5 |
|-----|-------------|------------|--------------|--------------|
| 0.8 | 71.9 SSIM2 | **73.5** | 73.7 | 74.0 |
| 1.0 | 77.0 | 77.9 | **78.5** | **78.6** |
| 1.5 | 83.8 | 84.7 | **85.2** | **85.3** |
| 2.0 | 87.4 | 88.0 | **88.5** | **88.5** |
| 2.5 | 89.6 | 89.9 | **90.4** | **90.4** |

AutoOptimize = HybMax-L14.5 (confirmed identical). Gains: +1.5 SSIM2 over cjpegli,
+0.6 over JpegliProg at 1.0 BPP. Consistent wins across all measured BPP levels.

**R-D comparison results (CID22, 20 images, 512px, 2026-02-06):**

| BPP | cjpegli-444 | JpegliProg | AutoOptimize | HybMax-L14.5 |
|-----|-------------|------------|--------------|--------------|
| 1.0 | 70.4 SSIM2 | 72.7 | 73.2 | **73.6** |
| 1.5 | 80.1 | 81.2 | 82.1 | **82.2** |
| 2.0 | 84.5 | 85.7 | **86.5** | **86.5** |
| 2.5 | 87.5 | 88.1 | **88.9** | **88.9** |

CID22 confirms gb82 findings. AutoOptimize within 0.1-0.4 of HybMax-L14.5.

**Recommended usage:**
```rust
// Best quality at given size (hybrid trellis λ=14.5 + progressive)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .auto_optimize(true);

// Do NOT combine with .trellis() — auto_optimize clears it and uses hybrid instead.
// If you call .trellis() after .auto_optimize(true), standalone trellis wins
// and hybrid is bypassed (streaming.rs:260-268).
```

Note: CMA-ES frequency scaling is separate from auto_optimize (which uses hybrid
trellis only). CMA-ES scaling modifies quant table generation and is described as
"incompatible" with auto_optimize's hybrid trellis approach.

### Remaining Hardening

- `serialize.rs::write_frame_header_xyb_ex()` still hardcodes 0x22/0x11 (low priority, always correct for XYB)

### API Improvements (Future)

**Issue #2: Top-level type re-exports for discoverability**

Currently, common types are in private/nested modules causing confusion:
- `zenjpeg::types::PixelFormat` - Module is private by default
- `zenjpeg::decoder::PixelFormat` - Counterintuitive location
- `zenjpeg::encoder::PixelLayout` - Different type, different location

**Proposed:** Re-export commonly used types at crate root for ergonomics:

```rust
// What users expect to write:
use zenjpeg::{PixelFormat, PixelLayout, ColorSpace, Limits, Dimensions};

// Instead of current workarounds:
use zenjpeg::decoder::PixelFormat;  // Why is it in decoder?
use zenjpeg::encoder::PixelLayout;  // Different module for similar type
```

**Types to re-export:**
- `PixelFormat` - Input/output pixel formats (RGB, RGBA, Gray, etc.)
- `PixelLayout` - Encoder pixel layouts (with transfer functions)
- `ColorSpace` - JPEG color space (YCbCr, Grayscale, RGB)
- `Subsampling` - Chroma subsampling modes (4:4:4, 4:2:0, etc.)
- `Limits` - Resource limits (max_pixels, max_memory, max_output)
- `Dimensions` - Width/height pair
- `JpegMode` - Baseline/Progressive

**Rationale:** Follows Rust conventions (tokio, serde, etc.). Makes API discoverable
via autocomplete. No breaking changes - existing paths still work.

**Status:** Deferred - requires careful consideration of public API surface.

Reference: api-feedback.md issue #2

### Resource Estimation API (docs/API_DESIGN.md)

For proxy server efficiency: accurate memory and compute cost estimation before encoding.

**Design:**
- `EncoderConfig` - dimension-independent config, reusable across images
- `ResourceEstimate` - returns `peak_bytes` (public), tracks internal metrics
- `InputMethod` enum - OneShot, Streaming, YCbCrDirect, YCbCrSubsampled
- `compute_cost_ms()` - approximate encoding time for current architecture
- `EncodeMetrics` - actual values returned from `finish_with_metrics()`

**Implementation TODO:**
- [ ] Extract `EncoderConfig` from `JpegEncoder` (dimension-independent)
- [ ] Add `InputMethod` enum for different input paths
- [ ] Implement `estimate_resources(width, height, input_method)` with accurate modeling
- [ ] Add allocation tracking (behind feature flag for zero overhead in production)
- [ ] Add `compute_cost_ms()` with CPU detection and calibration
- [ ] Add `EncodeMetrics` struct with actual peak_bytes, alloc_count, total_alloc_bytes, elapsed_ms
- [ ] Add `finish_with_metrics()` to return EncodeMetrics
- [ ] Add `finish_into(buffer)` for zero-copy output to pre-allocated buffer
- [ ] Benchmark to calibrate `compute_cost_ms()` estimates across architectures

**Internal tracking (not public API):**
- Total allocation count
- Total allocated bytes
- Max single allocation size

### CLI Parameter Support

- [x] Quality via `-q` / `--quality` (0-100, maps to quality internally)
- [x] Distance via `-d` / `--distance` (Butteraugli distance, lower = better)

**C++ cjpegli behavior differences based on quality vs distance:**

| Setting | C++ `-q` (quality) | C++ `-d` (distance) |
|---------|-------------------|---------------------|
| Quant tables | 2 tables (Y, shared Cb/Cr) | 3 tables (Y, Cb, Cr) |
| Quant values | Always clamped to 255 | Always clamped to 255 |
| Table selection | `jpeg_set_quality()` | `jpegli_set_distance()` |

**zenjpeg behavior:**
- Default: 3 separate quant tables (matches C++ distance mode)
- Use `separate_chroma_tables(false)` for 2-table mode matching C++ quality mode
- Quant values: Can exceed 255 when `allow_16bit_quant_tables = true` (default)

### 16-bit Quantization Tables (IMPORTANT)

**C++ jpegli API supports 16-bit but CLI forces baseline:**

```cpp
// API signature - force_baseline controls 8-bit vs 16-bit
void jpegli_set_distance(j_compress_ptr cinfo, float distance, boolean force_baseline);
void jpegli_set_quality(j_compress_ptr cinfo, int quality, boolean force_baseline);

// Quant value clamping in quant.cc:585
int quant_max = m->force_baseline ? 255 : 32767U;
```

The `force_baseline` parameter:
- `TRUE` (baseline): Clamp quant values to 255, use 8-bit DQT (SOF0)
- `FALSE` (extended): Allow values up to 32767, use 16-bit DQT when needed (SOF1)

**C++ cjpegli CLI always uses baseline:**
- `jxl::extras::EncodeJpeg()` hardcodes `TRUE` (lib/extras/enc/jpegli.cc:470-472)
- No CLI flag exists to change this
- Default in `jpegli_create_compress()` is `force_baseline = true`

**zenjpeg behavior:**
- `allow_16bit_quant_tables = true` (default): Matches `force_baseline = FALSE`
- `allow_16bit_quant_tables = false`: Matches `force_baseline = TRUE` (cjpegli CLI)
- Precision is auto-selected per-table: 8-bit if max ≤ 255, 16-bit if max > 255
- We DO automatically use 8-bit when no coefficient exceeds 255

**Quality threshold for 16-bit tables:**

The jpegli quant formulas produce chroma values that exceed 255 below Q87.
This is **quality-dependent, not image-dependent** - tested on Kodak corpus (24 images):

| Quality | Max Chroma | Tables |
|---------|------------|--------|
| Q90-100 | ≤200 | 8-bit (both Rust and C++) |
| Q87-89 | 218-254 | 8-bit (both Rust and C++) |
| Q86 | 272 | 16-bit (Rust) or clamped (C++) |
| Q70 | 560 | 16-bit (Rust) or clamped (C++) |
| Q50 | 919 | 16-bit (Rust) or clamped (C++) |

**Practical impact:**
- Default Q90: No difference between Rust and C++
- Web-quality Q70-Q85: Rust uses 16-bit, C++ CLI clamps to 255
- Low-quality Q50: Rust preserves precision, C++ CLI loses ~70% of chroma range

Note: The C++ API supports 16-bit via `force_baseline=FALSE`, but cjpegli CLI
doesn't expose this. Our `allow_16bit_quant_tables=true` default matches the
C++ API capability, not the CLI behavior.

**Why clamping doesn't hurt quality:**

The quant positions that exceed 255 are high-frequency chroma coefficients
(bottom-right corner of DCT matrix). These coefficients quantize to zero
regardless of whether dividing by 255 or 500+.

Tested on Kodak corpus at Q5-Q86:
- **Scan data is byte-for-byte identical** between 16-bit and clamped versions
- **SSIMULACRA2 delta: 0.000** at all quality levels (Q5, Q10, Q20, Q50...)
- Only difference is DQT marker overhead (~128 bytes)

C++ chose baseline compatibility with zero quality impact. Consider defaulting
`allow_16bit_quant_tables=false` to match, since 16-bit provides no benefit.

To match C++ cjpegli CLI, use `.allow_16bit_quant_tables(false)`.

## Running Tests

### Default Tests (no external dependencies)
```bash
cargo test --release                    # All non-ignored tests (~340 tests)
cargo test --release --test <name>      # Specific test file
```

### Full Test Suite (requires C++, testdata, corpus)

Prerequisites:
1. Build C++ jpegli: `cd internal/jpegli-cpp && mkdir -p build && cd build && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_TOOLS=ON .. && ninja cjpegli djpegli`
2. Generate testdata: `GENERATE_RUST_TEST_DATA=1 ./build/tools/cjpegli input.png output.jpg`
3. Corpus available at `~/work/codec-eval/codec-corpus/`

```bash
# Run ALL tests including ignored ones
cargo test --release -- --ignored

# C++ parity tests (most important)
cargo test --release --test parity_enforcement -- --ignored     # 7 tests
cargo test --release --test cpp_comparison -- --ignored         # 4 tests
cargo test --release --test cpp_filesize_comparison -- --ignored

# Corpus-based tests (need corpus-tests feature)
cargo test --release --features corpus-tests -- --ignored

# FFI tests (need ffi-tests feature + C++ build)
cargo test --release --features ffi-tests -- --ignored
```

### Test Categories

| Category | Command | Notes |
|----------|---------|-------|
| Unit tests | `cargo test --release --lib` | 324 tests, no deps |
| Integration | `cargo test --release` | Includes strip parity |
| C++ parity | `cargo test --release -- --ignored` | Needs C++ build |
| Corpus | `--features corpus-tests -- --ignored` | Needs image corpus |
| FFI | `--features ffi-tests` | Direct C++ bindings |

## Benchmarks

```bash
# Encoding benchmark
cargo bench --bench encode

# Decoding benchmark
cargo bench --bench decode

# Quick throughput check
cargo run --release --example benchmark_sharp_yuv
```

### Benchmark Output Rules (CRITICAL)

**NEVER re-run benchmarks just to parse output differently.** Criterion saves structured JSON:

```bash
# Results are stored in target/criterion/<group>/<bench>/new/estimates.json
# Extract results from JSON instead of re-running:
cat target/criterion/decode_compare/jpegli-baseline/512x512/new/estimates.json | jq '.mean.point_estimate'

# If you need terminal output, pipe to a file on FIRST run:
cargo bench --bench decode_compare 2>&1 | tee /tmp/bench-output.txt
```

**JSON structure:** `estimates.json` contains `mean`, `median`, `slope`, `std_dev` with `point_estimate` (nanoseconds) and `confidence_interval`.

**Example extraction:**
```bash
for d in target/criterion/decode_compare/*/512x512/new; do
  name=$(basename $(dirname $(dirname $d)))
  mean=$(cat "$d/estimates.json" | jq -r '.mean.point_estimate')
  echo "$name: $(echo "scale=2; $mean/1000" | bc) µs"
done
```

## C++ Parity Testing

Requires `cjpegli`/`djpegli` binaries from libjxl build:

```bash
# Build C++ tools (in internal/jpegli-cpp)
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_TOOLS=ON ..
ninja cjpegli djpegli

# Run parity tests
cargo test --release --features ffi-tests
```

## Color Space Architecture (CRITICAL)

### The Fundamental Truth

**Color space is NOT optional metadata. It defines the meaning of pixel values.**

A pixel value of `(255, 128, 0)` means COMPLETELY DIFFERENT COLORS in:
- sRGB (a specific orange)
- Display P3 (a more saturated orange)
- Rec.2020 (an even more saturated orange)
- Linear sRGB (a MUCH brighter orange, wrong gamma)

**If your code ignores color space, it is BROKEN. Period.**

### RGB→YCbCr Matrix: BT.601 Only (JPEG Constraint)

**CRITICAL**: JPEG encoding ALWAYS uses BT.601 RGB→YCbCr, regardless of input gamut.

```
BT.601: Y = 0.299R + 0.587G + 0.114B  (ALWAYS for JPEG)
```

The ICC profile embedded in the JPEG tells decoders what gamut the *decoded* RGB is in.
The YCbCr encoding math doesn't change - only the interpretation of the final RGB.

**Correct wide-gamut encoding pipeline:**
```
Input (any colorspace)
    │
    ▼ [1] Apply EOTF (linearize)
Linear RGB in source primaries
    │
    ▼ [2] Convert primaries (3x3 matrix)
Linear RGB in output primaries
    │
    ▼ [3] Apply OETF (e.g., sRGB gamma)
Gamma-encoded RGB in output primaries
    │
    ▼ [4] BT.601 RGB→YCbCr  ← ALWAYS BT.601!
YCbCr
    │
    ▼ [5] JPEG encode + embed ICC profile
JPEG file (ICC profile describes output primaries)
```

### Color Space Checklist

For EVERY pixel-processing function, answer:

1. **What primaries?** (sRGB/BT.709, P3, Rec.2020, AdobeRGB)
2. **What transfer function?** (Linear, sRGB, PQ, HLG, Gamma 2.2/2.4)
3. **What white point?** (D65, D50, DCI)
4. **Are we doing math?** → Convert to LINEAR first!
5. **Are we storing/transmitting?** → Apply appropriate OETF
6. **Is there an ICC profile?** → Honor it or document why not
7. **Should we write CICP?** → For video-derived content, yes

### UltraHDR Color Pipeline

For UltraHDR, the codec handles ONLY:
- JPEG encoding (SDR base + gain map)
- MPF structure (multi-picture format)
- XMP metadata embedding
- ICC profile embedding

The codec does NOT handle (caller's responsibility):
- HDR to SDR tonemapping
- Gain map computation
- Color space conversions
- ICC profile generation

```rust
// Caller does all color processing
let sdr = my_tonemapper.process(&hdr_linear_p3);  // Caller's tonemapper
let gainmap = compute_gainmap(&hdr, &sdr);         // Caller's gain map
let icc = p3_icc_profile();                        // Caller's ICC

// Codec just encodes and assembles - note stride is ALWAYS provided
let mut encoder = StreamingUltraHdrEncoder::new(w, h, gm_w, gm_h, config)?;
encoder.set_icc_profile(Some(&icc));

let sdr_stride = (w + 15) & !15;  // Caller's buffer layout
let gm_stride = (gm_w + 15) & !15;

for row in 0..height {
    // Push with explicit width and stride
    encoder.push_sdr_rows(&sdr_row_buf, w, sdr_stride, 1)?;
    if row % gm_scale == 0 {
        encoder.push_gainmap_rows(&gm_row_buf, gm_w, gm_stride, 1)?;
    }
}

// Caller provides output buffer
let mut output = Vec::new();
output.try_reserve(estimated_size)?;
encoder.finish_into(&mut output, &metadata)?;
```

### UltraHDR Conformance Notes

zenjpeg uses `ultrahdr-core` for XMP metadata parsing and gain map computation. There are known
differences from Google's libultrahdr reference implementation:

| Behavior | libultrahdr | zenjpeg/ultrahdr-core |
|----------|-------------|----------------------|
| `BaseRenditionIsHDR="True"` | Rejected | Accepted (bug) |
| Required XMP fields validation | Strict | Lenient |
| JPEG boundary detection | JpegScanner | MPF + SOI/EOI fallback |

**Practical impact**: Standard Ultra HDR files work correctly. Edge cases with `BaseRenditionIsHDR="True"`
or missing required XMP fields may behave incorrectly.

See the [ultrahdr README](https://github.com/imazen/ultrahdr#known-differences-from-libultrahdr) for details.

## Quality Metrics

**Use DSSIM or SSIMULACRA2, never PSNR.** PSNR doesn't correlate with perceptual quality.

```rust
// DSSIM (lower = better, 0 = identical)
use dssim::Dssim;

// SSIMULACRA2 (higher = better, 100 = identical)
use fast_ssim2::compute_frame_ssimulacra2;

// Butteraugli (lower = better, <1.0 = good)
use butteraugli::compute_butteraugli;
```

**Banned test images:**
- NEVER use the Kodak corpus. It's overfit by every codec and gives misleading results. Use CID22, CLIC, or screenshots instead.
- NEVER use smooth gradients for test image generators. Gradients produce degenerate DCT coefficients (0 or ±1) where arithmetic right shift is identity (`-1 >> n = -1`), making successive approximation levels indistinguishable and frequency-split comparisons meaningless. Use noise+patches, photographic content, or checkerboard patterns instead.

## Git Discipline

1. **Commit early, commit often** - Uncommitted work is invisible
2. **Run `cargo fmt` before changes** - Keep formatting commits separate
3. **Commit failing tests first** - Then fix in separate commit
4. **Never loosen test thresholds** - Find the real bug instead

## Feature Flags

```toml
[features]
default = ["std", "yuv", "archmage-simd", "trellis"]
trellis = []              # Rate-distortion trellis quantization (mozjpeg-style)
decoder = []              # Enable decoder (prerelease, API will change)
parallel = ["dep:rayon"]  # Multi-threaded DCT/quantization
archmage-simd = ["dep:archmage", "dep:magetypes", "dep:safe_unaligned_simd"]  # Token-based SIMD (~10-20% faster)
cms = ["cms-lcms2"]       # Color management
ultrahdr = ["dep:ultrahdr-core", "decoder"]  # UltraHDR HDR gain map support
ffi-tests = []            # C++ parity tests (requires jpegli-sys)
corpus-tests = []         # Corpus comparison tests
test-utils = []           # Testing utilities
```

**Decoder:** The decoder API is in prerelease. Enable with `features = ["decoder"]`.
API will have breaking changes.

**SIMD options:**
- Default: `wide` crate (portable, safe) - always enabled
- `archmage-simd` (default): Token-based safe intrinsics via archmage + magetypes - **~10-20% faster** on x86_64

## Key Files for Debugging

| File | Purpose |
|------|---------|
| `zenjpeg/src/encode/mod.rs` | Encoder pipeline |
| `zenjpeg/src/decode/mod.rs` | Decoder pipeline |
| `zenjpeg/src/color/xyb.rs` | XYB color conversion |
| `zenjpeg/src/quant/aq/mod.rs` | Adaptive quantization |
| `zenjpeg/src/huffman/mod.rs` | Huffman encoding |
| `zenjpeg/src/encode/search.rs` | ExpertConfig for external optimization |

## External Dependencies

- **Test images**: `internal/jpegli-cpp/testdata/`
- **Codec corpus**: `~/work/codec-eval/codec-corpus/` (Kodak, CID22)
- **C++ reference**: `internal/jpegli-cpp/` submodule

## Detailed Documentation

- **`zenjpeg/README.md` - API Reference** (encoder/decoder usage with examples)
- `zenjpeg/examples/README.md` - Examples and debugging tools
- `zenjpeg/docs/ADAPTIVE_QUANTIZATION.md` - AQ algorithm details
- `zenjpeg/docs/API_DESIGN.md` - Full API surface and proposed enhancements
- `docs/TUNING_HISTORY.md` - Performance tuning data, SIMD analysis, investigation notes, fixed bugs
- `docs/SECURITY.md` - Security considerations

## API Convergence TODOs

See `/home/lilith/work/zendiff/API_COMPARISON.md` for full cross-codec comparison.

**Three-layer pattern: EncoderConfig → EncodeRequest<'a> → Encoder (streaming only)**

**No backwards compatibility required** — we have no external users. Just bump the 0.x major version for breaking changes. No deprecation shims or legacy aliases — delete old APIs. Prefer one obvious way to do things — no duplicate entry points. Minimize API surface for forwards compatibility. Avoid free functions — use methods on types (Config, Request, Decoder) instead.

**Builder convention**: `with_` prefix for consuming builder setters, bare-name for getters.

**Licensing**: AGPL v3 / Commercial dual license. Cargo.toml uses `license = "AGPL-3.0-or-later"`. README must include the standard licensing text (see codec-design README).

**Project standards**: `#![forbid(unsafe_code)]` with default features. no_std+alloc (minimum: wasm32). CI with codecov. README with badges and usage examples. As of Rust 1.92, almost everything is in `core::` (including `Error`) — don't assume `std` is needed. Use `wasmtimer` crate for timing on wasm. Fuzz targets required (decode, roundtrip, limits, streaming). Codecs must be safe for malicious input on real-time image proxies — no amplification, bound memory/CPU, periodic DoS/security audits.

- [x] Add `EncodeRequest<'a>` intermediate between config and encoder — commit a4b7a01
- [ ] Move metadata (ICC/EXIF/XMP) from config to request (EncodeRequest supports them; config still has them too)
- [ ] Evaluate `RgbEncoder<P>` generic monomorphization cost — build example using 1 type vs 4 types (RGB8, RGBA8, RGB16, RGBF32), measure binary size delta and compile time. If significant, switch to `PixelLayout` enum internally with generic convenience at boundary
- [ ] Keep streaming push pattern but behind `request.build()` → `Encoder`
- [x] Add one-shot `encode()`/`encode_into()`/`encode_bytes()`/`encode_bytes_into()` — commit 9a388dc
- [x] Streaming keeps `finish()`/`finish_into()`/`finish_to()` (already correct)
- [x] `encode_to()`/`finish_to()` std-only (already gated with `#[cfg(feature = "std")]`)
- [x] Add `Limits` struct (all fields `Option<u64>`, default None = no limit) — commit ad91910
- [x] Rename `Error` → `EncodeError` (type alias, legacy re-export kept) — commit 0385d9f
- [ ] Switch from `impl Stop` per push to `&dyn Stop` on request
- [x] Resource estimation: `estimate_memory()` / `estimate_memory_ceiling()` already on config
- [ ] Factor metadata into `ImageMetadata` struct, move from config to request
- [x] Lossy/lossless split: N/A — JPEG is lossy-only, single `EncoderConfig` is correct
- [x] Standardize `AllocationStats` → `EncodeStats` — commit ef84b22
- [x] Add `DecodeError` type alias — commit 0385d9f
- [ ] Adopt `with_` prefix convention for all builder setters on Config/Request
- [x] Support `Rgba8` and `Bgra8` for encode and decode — encode: commit 44dcc4a, decode: commit 001319b
- [ ] Add probing: `ImageInfo::from_bytes(&[u8])` static probe with `PROBE_BYTES` constant
- [ ] Two-phase decoder: `build()` parses header → `info()` inspects → `decode()` continues without re-parsing
