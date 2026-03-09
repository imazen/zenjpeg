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
2. **Code analysis/details** → Add to "Investigation Notes" section below with:
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

## Decoder Performance (2026-02-15)

All benchmarks use noise+patches test patterns (not gradients). Noise+patches
produce realistic DCT coefficient distributions; gradients are degenerate.

**Wall-clock baseline 4:2:0 (sequential, streaming default, commit 39f012f):**

| Size | mozjpeg | zune | zenjpeg | zen-fast | cjpegli | zen/moz | zen/zune |
|------|---------|------|---------|----------|---------|---------|----------|
| 256 | 271µs | 252µs | 217µs | 207µs | ~360µs | **0.80x** | **0.86x** |
| 512 | 1.05ms | 1.06ms | 937µs | 895µs | 1.46ms | **0.89x** | **0.88x** |
| 1024 | 3.86ms | 4.03ms | 3.66ms | 3.48ms | 5.81ms | **0.95x** | **0.91x** |
| 2048 | 15.8ms | 16.6ms | 14.8ms | 14.8ms | 24.1ms | **0.94x** | **0.89x** |
| 4096 | 65.3ms | 67.8ms | 61.9ms | 59.1ms | 95.5ms | **0.95x** | **0.91x** |

zen-fast = `fancy_upsampling(false)` (box filter). mozjpeg = libjpeg-turbo with NASM SIMD (C).
All baseline 4:2:0 now uses streaming single-pass decode (entropy → IDCT → color convert →
output, no coefficient storage). zenjpeg beats mozjpeg at all sizes (0.80-0.95x) and
zune at all sizes (0.86-0.91x).

**Wall-clock baseline 4:4:4 (sequential, streaming):**

| Size | zune | zenjpeg | cjpegli | zen/zune |
|------|------|---------|---------|----------|
| 512 | 1.30ms | 1.14ms | 1.95ms | **0.88x** |
| 1024 | 5.15ms | 4.48ms | 7.64ms | **0.87x** |
| 2048 | 21.3ms | 18.6ms | 31.9ms | **0.87x** |

**Wall-clock progressive 4:2:0 (sequential, no DRI):**

| Size | mozjpeg | zune | zenjpeg | cjpegli | zen/moz | zen/zune |
|------|---------|------|---------|---------|---------|----------|
| 256 | 785µs | 877µs | 486µs | 988µs | **0.62x** | **0.55x** |
| 512 | 3.08ms | 3.39ms | 1.86ms | 3.70ms | **0.60x** | **0.55x** |
| 1024 | 12.1ms | 13.3ms | 7.34ms | 14.7ms | **0.61x** | **0.55x** |
| 2048 | 49.8ms | 54.3ms | 31.0ms | 59.6ms | **0.62x** | **0.57x** |
| 4096 | 225ms | 248ms | 156ms | 277ms | **0.69x** | **0.63x** |

zenjpeg is **1.4-1.6x faster** than mozjpeg, **1.6-1.8x faster** than zune,
and **~2x faster** than cjpegli on progressive. zune-jpeg 0.5.12 has a bug where
it silently skips AC refinement with DRI, so progressive benchmarks use NO DRI.
Without DRI, zune produces correct output.

**Callgrind (2048x2048, Q85 4:2:0 progressive, no DRI, commit 0c6d6ba):**
zenjpeg 415M Ir vs zune 580M Ir = zenjpeg is 28% fewer instructions when both correct.

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
11. Fused AC scan methods: `decode_ac_first_scan` and `decode_ac_refine_scan` process
    entire block grids without ScanResult enum wrapping or per-block function calls.
    Fast_ac combined 9-bit Huffman+value lookup for AC first scan. Partial peek
    fallback for end-of-scan with <9 bits. 33% entropy instruction reduction.
12. Split AC refine inner loop into separate ZRL and NEW_NZ code paths. Eliminates
    per-iteration `size == 0` check, Option<i16> wrapping, redundant termination
    checks. -6.1% AC refine instructions, -3% wall-clock.
13. Streaming single-pass decode for ALL baseline subsampling modes (4:4:4, 4:2:0,
    4:2:2). Eliminates coefficient storage (~100MB for 4096×4096 4:2:0) and the
    separate output pass. Entropy → IDCT → color convert → output in one MCU-row
    pass. Fancy h2v2 uses double-buffered chroma strips with 1-row lag for vertical
    context. 40% faster at 4096 (100ms → 60ms). `num_threads()` API for opt-in
    parallel control (0=auto, 1=sequential).

**Fast mode** (`fancy_upsampling(false)`): Uses box-filter upsampling fused with
color conversion instead of bilinear. 5-10% faster, minimal quality difference.

### Parallel Decode (2026-02-15, `--features parallel`)

Fused parallel decode: entropy decode + IDCT + color convert in one pass per
restart segment using rayon. Requires MCU-row-aligned DRI (default `restart_mcu_rows=4`).
Use `num_threads(1)` to force sequential; `num_threads(0)` = auto (default).

**Wall-clock baseline 4:2:0 (sequential vs parallel, commit 39f012f):**

| Size | mozjpeg | zune | zen seq | zen par | zen-fast par | speedup | vs moz |
|------|---------|------|---------|---------|-------------|---------|--------|
| 512 | 1.05ms | 1.07ms | 937µs | 796µs | 807µs | 1.2x | **0.77x** |
| 1024 | 3.86ms | 4.35ms | 3.66ms | 1.56ms | 1.41ms | **2.6x** | **0.37x** |
| 2048 | 15.8ms | 17.0ms | 14.8ms | 3.05ms | 2.57ms | **5.8x** | **0.16x** |
| 4096 | 65.3ms | 67.8ms | 61.9ms | 8.74ms | 6.52ms | **9.5x** | **0.10x** |

speedup = zen seq / zen par. vs moz = zen-fast par / mozjpeg. 256 omitted (too
small for parallel, MIN_BLOCKS=1024). Streaming decode + parallel eliminates both
the 2-pass bottleneck and the serial bottleneck. At 4096, parallel gets near-ideal
scaling: 62ms / 8.7ms = 7.1x on an 8-thread system.

**Parallel 4:4:4 (commit 39f012f):**

| Size | zune | zen seq | zen par | speedup |
|------|------|---------|---------|---------|
| 512 | 1.30ms | 1.14ms | 1.05ms | 1.1x |
| 1024 | 5.01ms | 4.48ms | 1.81ms | **2.5x** |
| 2048 | 22.1ms | 18.6ms | 3.03ms | **6.1x** |

Four fused paths: 4:4:4/gray (single-pass), 4:2:0+box (single-pass), 4:2:0+fancy
(single-pass with double-buffered extended chroma strips + boundary fixup), and
4:2:2/h2v1 (single-pass with horizontal-only chroma upsampling). All produce
byte-identical output to sequential path (23 hash-lock tests verify this).

Progressive images show no parallel speedup (no DRI = no restart segments to
parallelize). Progressive performance is the same with or without `--features parallel`.

**Benchmark methodology note:** To compare parallel vs sequential, must run bench
TWICE with different feature flags. Both "baseline" and "baseline-parallel" groups
use identical `Decoder::new()` code — fused parallel activates automatically when
compiled with `--features parallel` and DRI is present.

### Wave-Parallel Scanline Decode (2026-02-16, `--features parallel`)

Wave-parallel decode: decode `wave_size` restart segments at a time via rayon into
a reusable buffer, serve rows on demand, recycle. Activated automatically by
`scanline_reader()` when compiled with `--features parallel`, DRI is present, and
box filter is used (`fancy_upsampling(false)`). Uses a 6MB memory budget for the
wave buffer (vs 48MB full-image for full-buffer parallel at 4096).

**Wall-clock baseline 4:2:0 scanline paths (commit 3258bc2, Ryzen 9 7950X WSL2):**

| Size | seq scanline box | wave parallel | full-buf par | wave speedup | wave mem | full-buf mem |
|------|-----------------|--------------|-------------|-------------|---------|-------------|
| 512 | 1.13ms | 616µs | 527µs | **1.8x** | 768KB | 768KB |
| 1024 | 4.52ms | 1.50ms | 1.34ms | **3.0x** | 3MB | 3MB |
| 2048 | 17.7ms | 4.48ms | 2.41ms | **3.9x** | 6MB | 12MB |
| 4096 | 72.6ms | 19.7ms | 6.52ms | **3.7x** | 6MB | 48MB |

wave speedup = seq scanline box / wave parallel. Wave buffer capped at 6MB; at
4096 this is 8 segments (vs 64 total), giving 8x memory savings vs full-buffer.
At 512/1024 all segments fit in 6MB so no cap applies. Wave is 1.1-3.0x slower
than full-buffer parallel due to wave synchronization overhead and smaller
per-wave parallelism, but uses streaming API (rows on demand) with bounded memory.

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

**Baseline: RESOLVED** — Streaming decode eliminates the 2-pass bottleneck.
Sequential: zenjpeg beats mozjpeg at all sizes (0.80-0.95x) and zune (0.86-0.91x).
Parallel: at 4096, 10x faster than mozjpeg (6.5ms vs 65ms). Old buffered 4096 gap
(1.46x vs zune) is eliminated.

**Progressive: zenjpeg WINS 1.4-1.8x vs mozjpeg, 1.6-1.8x vs zune**:
- zune-jpeg 0.5.12 silently skips AC refinement with DRI (corrupt output, max_diff=224).
  Progressive benchmarks use NO DRI. Without DRI, zune produces correct output.
- Wall-clock at 2048: zenjpeg 31ms vs mozjpeg 50ms vs zune 54ms vs cjpegli 60ms
- Callgrind: zenjpeg 415M Ir vs zune 580M Ir = 28% fewer instructions

### Decoder Strictness Levels (2026-02-15)

Four levels controlling error tolerance during decode:

| Behavior | Strict | Balanced | Lenient | Permissive |
|----------|--------|----------|---------|------------|
| Non-JFIF markers | Error | Warn | Warn | Warn |
| Truncated data | Error | Pad zeros | Pad zeros | Pad zeros |
| Bad restart count | Error | Error | Warn | Resync fwd |
| RST sequence wrong | Error | Error | Error | Accept any |
| Zero quant value | Error | Error | Error | Clamp to 1 |
| Malformed segment | Error | Error | Error | Skip |
| Bad Huffman idx | Error | Error | Error | Clamp to 0 |
| Malformed DNL | Error | Error | Error | Skip |

Test results (commit 8d26d2c, 177-file conformance corpus):

| Decoder | Valid OK | Inv Rejected | Non-conf Accept |
|---------|----------|-------------|-----------------|
| zen-Strict | 39/41 | 100/116 | 8/20 |
| zen-Balanced | 41/41 | 92/116 | 14/20 |
| zen-Lenient | 41/41 | 88/116 | 14/20 |
| zen-Permissive | 41/41 | 80/116 | 14/20 |
| libjpeg-turbo | 37/41 | 75/116 | 14/20 |

Remaining 17-file gap vs libjpeg-turbo: all 613-byte fuzz-mutated files needing
scan-level longjmp recovery (diminishing returns). Non-conformant acceptance
matches libjpeg-turbo exactly (14/20).

Run: `cargo test --release -p zenjpeg --test decoder_leniency_comparison --features decoder -- compare_strictness --nocapture --ignored`

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

### Linear Iteration for AC Refinement (2026-02-14)

**Attempted:** Replace bitmap-accelerated inner scan loop in `decode_ac_refine` with linear
iteration (k from ss to se), matching zune-jpeg's approach. Goal was to eliminate the
`num_zeros_to_skip < zero_gap` branch that caused 1.08M mispredicts (25.5% of ALL mispredicts).

**Results:** WORSE. Instructions 449M → 469M (+4.4%), mispredicts 4.24M → 7.75M (+83%).

**Why it failed:** Linear iteration visits EVERY position from k to se (~49 positions per block
for band [15,63]), while bitmap visits only nonzero positions (~5-10). Even though individual
branches are more predictable (`coeffs[k] != 0` is 90% false for sparse blocks), the total
branch count is much higher: 49 × ~2.5 branches = ~123 per block vs bitmap's 10 × ~7 = ~70.
The unconditional refinement bit reads in the nonzero case happen the same number of times,
but the zero-position checking adds massive overhead for sparse progressive blocks.

**Conclusion:** Bitmap is fundamentally better for sparse coefficient data. The O(nonzero)
iteration count dominates the per-iteration branch cost.

### Unchecked Bit Reads for AC Refinement (2026-02-14)

**Attempted:** Add `read_bit_unchecked()` (no refill check) with `ensure_n_bits()` pre-fill
before bitmap loops. Save ~2 instructions per bit read by eliminating the `bits_in_buffer == 0`
check in the hot loop.

**Results:** Breaks restart marker handling. 5 test failures including "expected 0xFF for restart
marker" and "invalid Huffman code".

**Why it failed:** Near restart markers, `refill()` returns fewer bits than requested and sets
`marker_found`. The checked `read_bit_refine()` calls `refill()` when buffer empties, which
re-adds zero padding. Unchecked reads consume past the marker boundary. Even with
`saturating_sub` to prevent u8 underflow, the consumed bits corrupt the position for
subsequent Huffman decodes. Safe handling requires tracking available bits vs needed bits per
loop iteration, which adds complexity matching the cost of the original check.

**Conclusion:** The 2-instruction saving per bit read isn't worth the marker boundary complexity.

### Pre-refill AC First Scan (2026-02-15)

**Attempted:** Apply the same `ensure_bits()` + `peek_top(9)` pre-refill pattern (from AC
refine commit 43b24d6) to `decode_ac_first_scan`.

**Results:** Callgrind showed +11% regression (34.8M → 38.6M instructions). AC refine scan
(unchanged code) also regressed +6.4% (162.7M → 173.1M) due to code layout changes from
recompilation. Function is only 2.43% of total — even a 20% improvement saves <0.5%.

**Conclusion:** Not worth pursuing. The function is too small a fraction of total decode
time. Code layout effects from the change outweigh the algorithmic improvement.

### Conditional read_bit_fast in AC Refinement (2026-02-15)

**Attempted:** Add `read_bit_fast()` (no refill check) to bitstream.rs. Use `fast_bits`
boolean in AC refine to choose between `read_bit_fast()` and `read_bit_refine()` per
refinement bit read, based on whether `ensure_bits()` succeeded.

**Results:** All tests passed but callgrind showed 162.7M → 218.7M (+56M, +34%).

**Why it failed:** The per-read `if fast_bits { read_bit_fast() } else { read_bit_refine() }`
branch costs ~2 instructions — exactly the same as the refill check it replaces. Net effect
is zero benefit with added code complexity. The branch predictor handles the refill check
(`bits_in_buffer == 0` is rarely true) just as well as the `fast_bits` check.

**Conclusion:** Cannot eliminate per-bit overhead through branching. Would need fundamentally
different approach (e.g., reading multiple refinement bits in one operation).

### Branchy Coefficient Update in AC Refinement (2026-02-15)

**Attempted:** Replace branchless `c.wrapping_add((bit as i16) * not_set * sign * bit_val)`
with branchy `if bit != 0 && (c & bit_val) == 0 { if c > 0 { +bit_val } else { -bit_val } }`.

**Results:** Callgrind AC refine dropped from 225.2M to 184.5M (-18.1%). But wall-clock
was 5-21% WORSE across all sizes.

**Why it failed:** The `bit != 0` and `c > 0` branches are poorly predicted — coefficient
signs and refinement bits are effectively random. Each misprediction costs ~15 cycles but
counts as only 1 instruction in callgrind. The branchless version has more instructions but
is fully predictable (no branches = no mispredictions). Branch misprediction overhead
dominates instruction-count savings.

**Conclusion:** Callgrind instruction count can be misleading when branch prediction matters.
Branchless is correct for this hot path despite higher instruction count.

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

1. **Catastrophic 4:2:0 auto_optimize quality at specific Q levels (2026-02-19)** -
   `auto_optimize(true)` (hybrid trellis) with `ChromaSubsampling::Quarter` produces
   catastrophically degraded output (BA 20-43, visually destroyed) for certain images at
   specific quality levels. Neighboring quality levels are fine.
   - Affected images: bulb, baby, girl (from gb82 corpus). Also city/flowers at some Q levels.
   - Pattern: turbo Q90 source → zen Q75 (bulb: BA=32), Q97 (bulb: BA=43, baby: BA=32).
     cjpegli Q90 source → zen Q93 4:2:0 (bulb: BA=43), but zen Q93 4:4:4 is fine (BA=0.09).
   - The bug is quality-level-specific and non-monotonic: Q95 is fine but Q93 and Q97 are bad.
   - 4:4:4 also affected at extreme Q levels: waves Q97 4:4:4 auto_optimize scores 47.7.
     Previously believed 4:2:0-only; zensim regression tests found 4:4:4 cases too.
   - Impact: Contaminates reencode calibration grids (3/10 images have 16-42 BA deltas).
     Min_delta tables showed 0.55 at turbo Q90 instead of correct ~0.03.
   - Reproduction: `/mnt/v/output/zenjpeg/encoder_bug_420/` has source + re-encoded images.
     Raw data: `/mnt/v/output/zenjpeg/reencode_calibration/raw_data.csv`
   - **Root cause analysis (2026-03-09):** Trellis lambda weight is `1/(q*q)` per coefficient
     (`encode/trellis/ac.rs:48-72`). For 4:2:0, K420_RESCALE (`quant/mod.rs:549-566`) reduces
     chroma quant values to ~36-49% of luma. Smaller quant values → larger lambda weights →
     trellis zeros chroma AC coefficients more aggressively. This is *backwards*: K420_RESCALE
     means finer quantization (keep more detail), but the trellis interprets small quant values
     as "unimportant, zero them." At specific Q levels, the quant table scaling hits sweet spots
     where this inversion is catastrophic. Additionally, `auto_optimize` uses `aq_lambda_scale=0.0`
     (no AQ coupling), so there's no per-block adaptation to protect important chroma blocks.
     The block-norm-adaptive lambda (`scale1 / (scale2 + norm)`) amplifies this: 4:2:0 chroma
     blocks have smaller raw DCT coefficients → lower norm → even larger lambda → even more
     aggressive zeroing. This creates a feedback loop at certain Q levels.
   - **Fix candidates:** (a) Scale lambda weights by K420_RESCALE inverse for chroma components,
     (b) apply `chroma_scale < 1.0` in HybridConfig to reduce trellis aggression on chroma,
     (c) add AQ coupling (`aq_lambda_scale > 0`) to protect textured chroma regions.
   - Workaround: Calibration grids now use trimmed mean (drop top 20%) instead of mean.

2. **SA-optimized tables non-monotonic (2026-02-03)** - `optimized_tables.rs` anchor tables
   are non-monotonic between quality levels. Luma DC: q90=5, q95=37, q100=6. Each anchor
   was independently SA-optimized, finding different local optima. Results: ~10-20 SSIM2
   points worse than JpegliProg at matched BPP, non-monotonic BPP vs quality.
   - Root cause: Independent per-anchor SA without monotonicity constraints
   - Impact: Feature unusable as-is. Would need constrained optimization or post-smoothing.

3. **frymire_hash_locked XYB Q50 size mismatch (2026-03-08)** - `test_frymire_hashes_locked`
   fails on `baseline_xyb_opt Q50`: expected 292993 bytes, actual 293121 bytes (+128 bytes).
   Stale locked hash needing update after a previous encoder change.
   - Impact: `cargo test --release -p zenjpeg --test frymire_hash_locked` fails
   - Fix: Update locked hash value.

4. **Trellis dead parameters (2026-02-02, documented 2026-03-08)** - `trellis_use_lambda_weight_tbl`
   always uses flat 1/q² weights. `trellis_num_loops` stored but never read (single-pass only).
   Both documented in config doc comments. Low priority — parameters have no effect.

### Fixed / Resolved Bugs (historical reference)

- **Progressive decoder truncation near restart markers (FIXED 2026-03-09, commit 08ef601)** -
  Fused `decode_ac_first_scan` and `decode_ac_refine_scan` lacked a bit-by-bit Huffman
  fallback when `peek_bits_refill(16)` failed near restart marker boundaries. When a Huffman
  code > 9 bits occurred in the last 2-3 blocks before a restart marker (0xFF 0xDn), the
  16-bit peek failed because the marker interrupted bitstream refill. The function incorrectly
  treated this as scan truncation, zeroing all remaining AC coefficients. The standard
  `decode_huffman_symbol_lenient` had this fallback but the fused functions did not.
  Triggered at Q91-Q93 (where AC table had codes > 9 bits) with DRI=216 on 576x576 images.
  Fix: added bit-by-bit Huffman decode fallback matching the standard function.
  - Found during investigation of Known Bug #1 (catastrophic auto_optimize quality).
  - Test: `cargo test --release -p zenjpeg --test quality_regression --features decoder -- diagnostic_coefficient_comparison --nocapture --ignored`

- **Parallel feature skipping deringing (FIXED 2026-03-09)** - `parallel_dct_plane` in
  `encode/parallel.rs` did `extract_block → forward_dct` without applying deringing, while
  the sequential path in `strip/mod.rs:1027-1029` applied `preprocess_deringing_block` before
  DCT. This caused `locked_values` test failures with `--features parallel` — not
  non-determinism, but a deterministic quality regression (deringing silently skipped).
  Fix: pass `deringing: Option<u16>` (dc_quant when enabled) through `parallel_dct_y_blocks`
  into both parallel and sequential DCT plane functions. Deringing is block-local (no
  cross-block dependencies), so it parallelizes trivially.

- **zune-jpeg progressive decode issue (STALE, was Bug #5)** - Originally reported that
  zune-jpeg decoded zenjpeg progressive output as grayscale. Investigation (2026-03-09)
  found 70+ progressive encoding tests pass with zune-jpeg. The AC refinement trailing
  ZRL fix (commit d355648) likely resolved the underlying scan structure issue. The only
  remaining trace is a skip in `chroma_upsample_regression.rs:1038`. Note: zune-jpeg 0.5.12
  still has a separate bug silently skipping AC refinement with DRI (max_diff=224).

- **Grayscale scanline reader panic (FIXED 2026-02-06, commit be24fac)** - Streaming scanline
  reader methods panicked on grayscale images. Fixed by handling 1-component images.

- **XYB 4:2:0 encoder producing undecodable JPEGs (FIXED 2026-03-04, commit b0cafce)** -
  Frequency counter clamped DC categories to 11 (`.min(11)`) but encoder wrote unclamped
  categories. XYB produces DC differences > ±2047 at low quality (categories 12+). Huffman
  table lacked codes for those categories, writing (code=0, len=0) → corrupted bitstream.
  Fix: remove `.min(11)` from `collect_block_frequencies_simd`. Previously-encoded files
  in `testdata/decode_failures/` remain permanently corrupted (kept as ignored tests).
  - Test: `cargo test --release -p zenjpeg --test xyb_roundtrip --features decoder`

- **CMYK scanline transform panic (FIXED 2026-03-04, commit bde9f48)** -
  `scanline_reader_with_transform()` had no CMYK check. Non-dimension-swapping transforms
  (e.g., FlipHorizontal) fell through to `from_coefficients()` → `StripProcessor` with
  `[u8; 3]` arrays → index-out-of-bounds at `h_samp[3]`. Fix: route CMYK to buffered
  decode fallback, matching `scanline_reader()`.
  - Test: `cargo test --release -p zenjpeg --test cmyk_transform --features decoder`

- **False XYB ICC detection for cjpegli JPEGs (FIXED 2026-02-14, commit 744d38a)** -
  `is_xyb_profile()` checked for "jxl " CMM type (bytes 4-7) in ICC profiles, but cjpegli
  writes "jxl " for ALL ICC profiles (including standard sRGB), not just XYB ones. This caused
  every cjpegli JPEG with an ICC profile to be misidentified as XYB, bypassing the fast i16
  decode path and falling through to the f32 XYB→RGB conversion — producing completely wrong
  colors (max_diff=252). Fix: replace "jxl " CMM check with exact-match against the known
  720-byte XYB ICC profile, falling back to "XYB" text search in the profile description.
  Also affected baseline streaming and fused parallel paths (would have returned "no decoded
  data" error for cjpegli images).

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

### ~~Make archmage-simd mandatory (not a feature flag)~~ DONE

Completed: archmage, magetypes, safe_unaligned_simd are now mandatory dependencies.
`archmage-simd` feature flag is empty (kept for backwards compatibility).
All `#[cfg(feature = "archmage-simd")]` gates replaced with `#[cfg(target_arch = "...")]`.
DCT, transpose, and nonzero mask functions dispatch to archmage intrinsics at runtime.

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
- **Decoder marker validation parity with libjpeg-turbo** — see `docs/strictness.md` for
  the full comparison table. The goal: any malformed input that could crash *any* decoder
  (division by zero, out-of-bounds index, integer overflow) must be caught at parse time,
  regardless of strictness level. Structural validation should never be skippable.
  Specific gaps to close:
  - **DRI length validation**: currently reads 2 bytes without checking `length == 4`.
    Malformed DRI could desync the parser. (`markers.rs:parse_restart_interval`)
  - **SOS length validation**: doesn't check `length == 6 + 2*num_components`. A crafted
    SOS with wrong length could cause reads past the marker boundary. (`scan.rs:parse_scan`)
  - **Duplicate component in SOS**: not checked. libjpeg-turbo rejects duplicate component
    IDs within a single scan. A duplicate could cause the same coefficient buffer to be
    written twice, producing garbage. (`scan.rs:parse_scan`)
  - ~~**Ah/Al range validation**~~: FIXED (commit d12d699). Now rejects Ah/Al > 13.
    Also added Ss > Se validation. 5 regression tests in `decoder_error_handling.rs`.
  - **Extraneous inter-marker bytes**: silently skipped with no count or warning. Should
    at least count discarded bytes and emit a warning in Balanced mode, error in Strict.
    (`mod.rs:read_marker`)
  - **DHT symbol count vs remaining length**: validated via length arithmetic but should
    explicitly check `num_values <= 256` before reading. (`markers.rs:parse_huffman_table`)
  - **Restart marker resync**: no recovery strategy when the wrong RST marker appears.
    libjpeg-turbo has a 3-action resync (discard/scan forward/leave unread). zenjpeg
    should at minimum handle the "RST off by 1-2" case gracefully in Balanced/Lenient.
  - **Tables-only streams (EOI before SOS)**: currently fatal. Consider supporting in
    Balanced mode (return empty image or header-only result) like libjpeg-turbo does.

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

Reference: docs/api-feedback.md issue #2

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
- **XYB always uses extended sequential (SOF1)**: `EncoderConfig::xyb()` sets
  `allow_16bit_quant_tables = true`. Calling `allow_16bit_quant_tables(false)` or
  `force_baseline()` on an XYB config returns `Err`. XYB's wider dynamic range
  (scaling factors up to 23x) produces DC categories 12-15 that exceed baseline's
  limit of 11.
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
archmage-simd = []  # Legacy flag — archmage/magetypes are now mandatory dependencies.
cms = ["cms-lcms2"]       # Color management
ultrahdr = ["dep:ultrahdr-core", "decoder"]  # UltraHDR HDR gain map support
ffi-tests = []            # C++ parity tests (requires jpegli-sys)
corpus-tests = []         # Corpus comparison tests
test-utils = []           # Testing utilities
```

**Decoder:** The decoder API is in prerelease. Enable with `features = ["decoder"]`.
API will have breaking changes.

**SIMD:** archmage + magetypes are mandatory dependencies (token-based safe intrinsics).
`wide` crate provides portable fallback. On x86_64, archmage dispatches to AVX2/FMA/AVX-512
at runtime via `X64V3Token::summon()` / `X64V4Token::summon()`.

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
