# zenjpeg Project Guide

Pure Rust port of Google's jpegli JPEG encoder/decoder from the JPEG XL project.

## Canonical training data + indexes (added 2026-05-20)

**The canonical index for all ML data lives at `~/work/zen/DATA_PROVENANCE.md`.**

Quick paths:
- Trainer input: `s3://zentrain/canonical/2026-06-27/` (per `~/work/zen/DATA_PROVENANCE.md`; 2026-05-21 superseded)
- Master inventory: `~/work/zen/_ml-inventory-2026-05-20/00-MASTER-SYNTHESIS.md`
- Per-codec picker audit: `~/work/zen/_ml-inventory-2026-05-20/05-per-codec-pickers.md`

## ML/adaptive selector status (2026-05-20)

zenjpeg's adaptive selector lives in `~/work/coefficient/` (NOT in this repo):

- Substrate: `/home/lilith/oracle-d2-store/oracle-d2/` (75k JPEG encodings, 108k metrics, 90 source images)
- Trainer: `coefficient/scripts/fit_oracle_tree.py` (sklearn decision tree)
- Rules output: `selector_tree_rules.json` (~70 decisions per (bucket, q_bin, metric))
- Wired into `EncoderConfig::adaptive(image, quality)` in this repo

The `benchmarks/zenjpeg_picker_v0.3_2026-05-04.bin` (7.5 KB, ZNPR format) is a training artifact only — no encoder integration. `dev/picker_v0_3_holdout_ab.rs` is the eval harness for it. Both kept for reproducibility; do not delete.

**Future picker work pattern:** if a Rust-side picker is needed, follow zenavif (`~/work/zen/zenavif/src/auto_tune.rs` + `EncoderConfig::auto_tune()` public API). See `~/work/zen/_ml-inventory-2026-05-20/05-per-codec-pickers.md` for the cross-codec design discussion.

## BANNED: worktrees in this repo (CRITICAL)

**Do NOT create `git worktree` directories in this repo.** Claude Code
sessions have repeatedly desynced worktrees from `origin/main` across
force-pushes and history rewrites, stranding WIP and stale `main` refs
across multiple sibling directories.

**Superseded 2026-07-19: the global jj-on-main workflow (see
`~/.claude/CLAUDE.md`) is mandatory — no `git checkout` on tracked
branches; sibling `jj workspace add` only when physically necessary.**

**If you find yourself typing `git worktree add` — stop.** Work in
place on main via jj. The one exception is when a human
explicitly names a worktree path for a specific reason.

**Cleanup of existing worktrees requires rescue first:** copy every
modified and untracked file to `~/tmp/_rescued-worktrees/<timestamp>-<name>/`
preserving relative paths BEFORE running `git worktree remove` — even
if `git worktree remove` is supposed to refuse dirty trees. Belt and
suspenders.

### Why jj (incident context — jj is now mandatory, see above)

`jj` (https://jj-vcs.github.io/jj/) is a git-compatible VCS that
eliminates most of the footguns Claude has hit in this session:

- **Auto-snapshot on every command** — working tree always committed;
  nothing lost to `stash` confusion or forgotten `git add`.
- **`jj undo`** — undo any op, including force-push recovery locally.
- **Change IDs stable across rebases** — my history rewrite wouldn't
  have created ghost SHAs that left sibling worktrees holding bags.
- **First-class conflict markers in commits** — rebase that would
  fail in git succeeds with conflicts marked, fixable incrementally.
- **No staging area** — the class of bugs where Claude staged an
  unintended file (the `internal/jpegli-cpp` symlink typechange)
  don't exist.

This repo is already colocated (`.jj/` present). All existing git
tooling (GitHub, PRs, CI) keeps working.

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
(Price & Rabbani 2000) during f32 dequantization (bypasses the fast i16 IDCT
path). Default: off. Key findings (full sweep tables migrated to
`docs/TUNING_HISTORY.md`, 2026-07-13):

- Image-dependent tradeoff: helps +0.14..+0.39 SSIM2 on CID22 at Q75+,
  but the default integer IDCT wins on frymire at all qualities below Q99.
- zen+bias tracks C++ jpegli's decoder closely everywhere (gap 0.02-0.11
  SSIM2 pts; max pixel diff vs cjpegli always 1).
- Pairwise similarity at Q85: zen+bias<->cjpegli 94.31 vs default<->cjpegli 91.42.

Run: `cargo test --release -p zenjpeg --test dequant_bias_comparison -- --nocapture --ignored`

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
| Zero quant value | Error | Clamp to 1 | Clamp to 1 | Clamp to 1 |
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

Run: `cargo test --release -p zenjpeg --test decoder_leniency_comparison -- compare_strictness --nocapture --ignored`

## Failed Explorations

Measured dead ends — DO NOT RE-ATTEMPT without reading the full write-ups,
migrated to `docs/TUNING_HISTORY.md` (2026-07-13):

- **Parallel AQ (2026-01-17):** rayon per-block AQ = 5x SLOWER at 4K; max realistic gain ~6% — not worth it.
- **Fuzzy erosion SIMD (2026-01-21):** archmage helpers 3x slower (register spills); fully-inlined variant still slower (icache bloat).
- **AVX-512 dual-block DCT (2026-01-21):** 2.3x SLOWER — 8x8 DCT is inherently 8-wide; transpose can't cross 128-bit lanes cheaply. AVX2 is optimal.
- **Linear iteration for AC refinement (2026-02-14):** +4.4% instructions, +83% mispredicts vs the nonzero-bitmap loop.
- **Unchecked bit reads in AC refinement (2026-02-14):** breaks restart-marker boundary handling; safe handling costs as much as the check.
- **Pre-refill AC first scan (2026-02-15):** +11% callgrind regression; the function is only 2.43% of decode.
- **Conditional read_bit_fast in AC refinement (2026-02-15):** the per-read mode branch costs exactly what the refill check costs — zero net.
- **Branchy coefficient update in AC refinement (2026-02-15):** -18% instructions but 5-21% WORSE wall-clock — coefficient signs are unpredictable; branchless wins. (Callgrind counts mislead when prediction dominates.)
- **Decoder zero-copy architecture (2026-01-22): IMPLEMENTED** — `decode_block_into` with caller buffer + max-coeff-count-since-restart zeroing; 5-6.5% wall-clock win. Key insight: reusable buffers accumulate stale state from ALL prior blocks, so track the maximum coefficient count since the last restart, not just the previous block's.

## Investigation Notes

**Cross-backend dispatch parity tolerance (2026-04-21):**

`test_dispatch_parity` (zenjpeg/tests/encoder_regression.rs) tolerates up to
**64 bytes** of size divergence between archmage token permutations. The
April 1 ties-to-even fix (archmage c566f76) IS landed in magetypes 0.9.21,
so this is NOT the old scalar-vs-SSE ties-away rounding bug.

Observed failure was `progressive_444_opt Q90 ~41 bytes` on frymire when
v3/v3-Crypto/AVX-512/v4x are all disabled. zenjpeg only registers
`v3, neon, wasm128, scalar` tiers, so that permutation falls directly to
scalar. Baseline configs stay within a handful of bytes; progressive Q90
amplifies each boundary-flip through AC-refinement tokenization.

Source is NOT YET LOCALIZED. A few ULPs of FP intermediate difference
between the AVX2 and scalar paths cascade into occasional DCT coefficient
flips near the zero-bias threshold, but it is not currently proven whether
the divergence originates in magetypes or in zenjpeg's own SIMD code.

Before filing anything upstream: write a unit test against magetypes only
(no zenjpeg) that exercises the AVX2 vs scalar backends on the same input
and looks for ULP-level divergence in the specific operations we use (DCT
butterflies, quantization, AQ pre-erosion). If magetypes alone is
bit-identical across backends, the divergence is ours.

If the 64-byte threshold is exceeded again on baseline configs or at a
quality/subsampling combination that previously passed, localize the
regression to the specific encode-side SIMD change rather than raising
the tolerance further.

**Mozjpeg Parity Investigation (2026-03-26, commit 1aba86cf):**

`Quality::ApproxMozjpeg(q)` + `MozjpegRobidoux` tables had a double-conversion bug:
`to_internal()` remapped mozjpeg Q85→jpegli Q83, then that remapped value fed the
Robidoux table generator. Fix: `Quality::for_mozjpeg_tables()` returns original `q`
unchanged for `ApproxMozjpeg`, falls back to `to_internal()` for other variants.

After fix, measured on 25 gb82 images at Q50-Q98 (MozjpegProgressive preset, 4:2:0):
- **Size**: zen/moz ratio 0.99-1.01x (was 0.90-0.95x before fix)
- **Zensim vs mozjpeg decoded**: 84-93 mean (was 75-91, +5 pts from table fix)
- **Zensim vs original**: zen +0.01 to +0.66 better than mozjpeg (f32 DCT wins)
- **Wins/ties/losses**: 110/66/24 out of 200 comparisons (zen wins 55%)
- Integer DCT is NOT needed — f32 produces measurably better quality at same size

Remaining zen-vs-moz gap (zensim 84-93 instead of 100) is the f32 vs 13-bit fixed-point
DCT constant divergence. The integer constants differ by 10-160 ppm from f32, producing
systematically different coefficients — NOT just precision loss. Color conversion contributes
negligibly (only 0.06% of all RGB values differ by ±1 between f32 FMA and 16-bit fixed-point).

Examples: `mozjpeg_parity_regress`, `mozjpeg_parity_tuning`, `mozjpeg_quality_vs_original`.

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

Live bugs only. Fixed ones are one-liners under "Fixed / Resolved Bugs" below,
with full write-ups in `docs/TUNING_HISTORY.md` — do not let struck-through
entries accumulate here.

1. **SA-optimized quant tables are non-monotonic (2026-02-03, issue #12)** — the anchor
   tables are non-monotonic across quality levels. Quant values must be non-increasing as
   quality rises; these climb repeatedly.
   - **Location (re-verified 2026-07-15):** `encode/tables/sa_piecewise_v4_data.rs`.
     The old `optimized_tables.rs` no longer exists — the data was *restored* into
     `sa_piecewise_v4{,_data}.rs` and wired into production as
     `QuantTableConfig::PiecewiseV4` (`encode/plan.rs:182`), carried over **verbatim**,
     so the defect shipped with it.
   - **Measured (2026-07-15, read from the source data):** `ANCHOR_LUMA` DC q5→q100 =
     15, 15, 28, 20, 24, 12, 14, 18, 23, 26, 30, 12, 6, 7, 5, 3, 8, 5, **37**, 6 — i.e.
     the old q90=5 / q95=37 / q100=6 wobble is unchanged. `ANCHOR_CB` DC is worse: q100=81,
     *coarser than q5's 66*. Independently corroborated by #12's own comment (34 size-
     monotonicity violations q1–q99 vs 0 for Jpegli).
   - **Root cause:** each anchor was independently SA-optimized, finding a different local
     optimum; `tables_for_quality` (`sa_piecewise_v4.rs`) just lerps between anchors — no
     smoothing, no constraint pass, and **no monotonicity test** anywhere in the tables path.
   - **Impact:** not the default (`QuantTableConfig::default()` is Jpegli), so only callers
     explicitly selecting `PiecewiseV4` hit it — but it is public and production-wired.
     `sa_piecewise_v4.rs`'s own header doc recommends it as `EncoderConfig::adaptive`'s
     default and claims it "beats jpegli on 99/100 quality levels" without mentioning any
     of this — treat that doc as unreliable.
   - **Fix:** constrained optimization or a post-smoothing pass over the 20 anchors, plus a
     monotonicity regression test (natural home: `sa_piecewise_v4.rs`'s test module).

2. **Two `target-zq`-gated research examples don't compile against current APIs (found
   2026-07-23, pre-existing).** Neither is part of the normal build/test matrix — both
   require the `target-zq` feature, and per this doc's own Feature Flags section "no CI
   job enables `target-zq`" — but `--all-features` exposes them, and one (below) was
   additionally hidden behind a dead cfg condition until today.
   - **`examples/zq_calibrate.rs`**: references `AnalysisFeature::TextLikelihood` and
     `AnalysisFeature::ScreenContentLikelihood`, neither of which exist on the `zenanalyze`
     git dependency's current `AnalysisFeature` enum (`error[E0599]`). Needs the current
     variant names from `zenanalyze` (unpublished sibling, `~/work/zen/zenanalyze`).
   - **`examples/picker_v0_3_holdout_ab.rs`**: `#![cfg(all(feature = "target-zq", feature =
     "trellis"))]` referenced the removed `trellis` feature (see "Flags that no longer
     exist" above), so the file was UNCONDITIONALLY compiled out — the `cfg` could never be
     true, meaning nobody has actually built this file since `trellis` was dropped. Fixed
     the gate to plain `#![cfg(feature = "target-zq")]` (now consistent with the
     `[[example]]`'s own `required-features`), which reveals it's *also* stale: unresolved
     `zenjpeg::encode::trellis::HybridConfig` import, and `EncoderConfig` has no
     `chroma_distance_scale`/`hybrid_config` methods anymore (there's a `chroma_distance_scales`
     — plural — getter in `encode/encoder_types.rs`, but the setter-side API this file
     expects doesn't match current `EncoderConfig`). Needs a real pass against the current
     encoder builder API, not attempted here — out of scope for the day's actual task
     (removing the `image` dev-dependency to unblock a moxcms upgrade).

### Fixed / Resolved Bugs (historical reference)

One-line index; full write-ups migrated to `docs/TUNING_HISTORY.md` (2026-07-13).

- Progressive Q10 encoder ~2.8% larger than C++ jpegli (issue #23) — FIXED; issue closed 2026-04-15.
  **Verified 2026-07-15**: the 4 `quality_matrix` progressive tests that had been `#[ignore]`d citing
  #23 (444/422/420/440) all pass, and the ignores are removed. Q10 progressive size deltas are now
  +1.4..+2.4% *and* Rust scores +3.1..+4.1 SSIM2 better — i.e. it buys quality, not waste. The
  sibling baseline case is still live as #78 (`test_ycbcr_420_baseline`, still `#[ignore]`d).
- XYB bottom-partial-strip vertical padding stride (issue #186) — FIXED 2026-07-13 (0064e34a);
  regression test `tests/bundled/xyb_edge_padding.rs`.
- Catastrophic 4:2:0 auto_optimize quality at specific Q levels — FIXED 2026-03-09 (08ef601); root
  cause was progressive decoder truncation near restart markers, NOT trellis lambda weights.
- frymire_hash_locked XYB Q50 size mismatch — FIXED 2026-03-09 (stale hashes; 16-bit DQT at Q50).
- frymire_hash_locked XYB Q50 pre-existing failure — FIXED 2026-03-27 (XYB hashes relocked).
- Trellis dead parameters (`trellis_use_lambda_weight_tbl`, `trellis_num_loops`) — FIXED 2026-03-30 (d2a1af25).
- XYB linear-input encoder paths saturated to white — FIXED 2026-04-23 (28658af6 + 9e2348fe).
- Fused parallel decode bypassed coefficient storage (`DecodeMode::Coefficient`) — FIXED 2026-03-31 (c9b47ec1).
- Progressive decoder truncation near restart markers (missing bit-by-bit Huffman fallback) — FIXED 2026-03-09 (08ef601).
- `--features parallel` silently skipped deringing — FIXED 2026-03-09.
- zune-jpeg "decodes our progressive as grayscale" report — STALE; separate zune 0.5.12 bug (skips AC refinement with DRI) remains upstream.
- Grayscale scanline reader panic — FIXED 2026-02-06 (be24fac).
- XYB 4:2:0 undecodable JPEGs (DC category clamped in frequency counter but not encoder) — FIXED 2026-03-04 (b0cafce).
- CMYK scanline transform panic — FIXED 2026-03-04 (bde9f48).
- False XYB ICC detection for cjpegli JPEGs (matched "jxl " CMM instead of the exact XYB profile) — FIXED 2026-02-14 (744d38a).
- 4:2:0 scanline chroma upsampling at MCU bottom boundaries — FIXED 2026-02-09 (bd0f8d7).
- Scanline h2v2 boundary fixup buffer overflow (>8192px wide) — FIXED 2026-02-09 (8f1295f, closes #1).
- Progressive MCU-padded storage stride mismatch — FIXED 2026-02-09 (29d6d81).
- Progressive interleaved DC scan padding desync — FIXED 2026-02-09 (759a4a7).
- Older fixed bugs: `docs/TUNING_HISTORY.md`.

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
  - ~~**DRI length validation**~~: FIXED (already had length check + extra byte consumption + warning).
  - ~~**SOS length validation**~~: FIXED (already checked `length == 6 + 2*num_components` with warning).
  - ~~**Duplicate component in SOS**~~: FIXED (commit d88c3ad7). Rejects duplicate component
    IDs within a single scan, matching libjpeg-turbo behavior.
  - ~~**Ah/Al range validation**~~: FIXED (commit d12d699). Now rejects Ah/Al > 13.
    Also added Ss > Se validation. 5 regression tests in `decoder_error_handling.rs`.
  - ~~**Extraneous inter-marker bytes**~~: FIXED (commit d88c3ad7). Counts skipped bytes,
    errors in Strict mode, warns with `ExtraneousBytesSkipped` in Balanced/Lenient.
  - ~~**DHT symbol count vs remaining length**~~: FIXED (commit d88c3ad7). Explicit
    `num_values <= 256` check before allocation, preventing OOM on malicious input.
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
- **SOF1 vs quant precision are decoupled**: `allow_16bit_quant_tables` controls only
  DQT marker precision (8-bit vs 16-bit values). SOF1 frame type is controlled
  separately by an internal `force_sof1` flag.
- **XYB always uses SOF1**: `force_sof1` is set automatically for XYB color mode
  because XYB's wider dynamic range (scaling factors up to 23x) produces DC
  categories 12-15, exceeding baseline SOF0's limit of 11. This is independent of
  quant table precision — XYB defaults to `allow_16bit_quant_tables = false` since
  16-bit DQT provides no quality benefit.
- `allow_16bit_quant_tables()` and `force_baseline()` return `Self` (infallible)
  and work with any color mode including XYB.
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

C++ chose baseline compatibility with zero quality impact. zenjpeg now also
defaults `allow_16bit_quant_tables=true` for YCbCr (matching C++ API capability)
but `false` for XYB (where SOF1 is forced for DC categories, not quant precision).

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

NOTE 2026-07-19: NEW benches use zenbench (workspace mandate); the criterion notes below apply to legacy benches only.

**NEVER re-run benchmarks just to parse output differently.** Criterion saves structured JSON:

```bash
# Results are stored in target/criterion/<group>/<bench>/new/estimates.json
# Extract results from JSON instead of re-running:
cat target/criterion/decode_compare/jpegli-baseline/512x512/new/estimates.json | jq '.mean.point_estimate'

# If you need terminal output, pipe to a file on FIRST run:
cargo bench --bench decode_compare 2>&1 | tee ~/tmp/bench-output.txt
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
use fast_ssim2::compute_ssimulacra2;

// Butteraugli (lower = better, <1.0 = good)
use butteraugli::compute_butteraugli;
```

**Banned test images:**
- NEVER use the Kodak corpus. It's overfit by every codec and gives misleading results. Use CID22, CLIC, or screenshots instead.
- NEVER use smooth gradients for test image generators. Gradients produce degenerate DCT coefficients (0 or ±1) where arithmetic right shift is identity (`-1 >> n = -1`), making successive approximation levels indistinguishable and frequency-split comparisons meaningless. Use noise+patches, photographic content, or checkerboard patterns instead.

**REQUIRED test images: flat-chroma (DC-only) coverage for any decoder path test.**

The gradient ban above is about *luma frequency-split* comparisons. Do NOT
over-read it into "make everything high-frequency" — that is its own blind spot,
and it has cost us a real shipped-adjacent bug:

- A decoder's **DC-only** block paths (`coeff_count <= 1`, `idct_int_dc_only`)
  are **unreachable** with high-frequency content. Smooth chroma quantizes to
  DC-only, which is what essentially **every real photograph** produces.
- 2026-07-15: the coefficient path's vertical-context peek had a wrong DC-only
  IDCT (`(dc + 1024) >> 11` vs the correct `(dc + 4 + 1024) >> 3`). It was
  byte-exact vs real libjpeg-turbo on every synthetic generator in the suite, at
  every size × subsampling × path — because they all deliberately make chroma
  non-flat (`color_noise`: *"so chroma isn't flat"*). Only a corpus image caught
  it (waterhouse.jpg: 76/255 wrong on the last row of every MCU row). See
  `docs/DECODER_UNIFICATION_PLAN.md`.

So: any test that sweeps decode paths MUST cover **both** chroma regimes —
high-frequency chroma *and* chroma that is flat within each 8×8 chroma block
(varying across MCU boundaries so boundary errors stay visible). Canonical
generators: `smooth_chroma_image` in `tests/libjpeg_idct_all_paths_parity.rs`
(swept via `IMAGE_KINDS`) and `smooth_chroma_bands` in
`tests/bundled/coeff_unification.rs`. Both are verified to fail against the
pre-fix code — do not delete them citing the gradient ban; they are not
gradients, and they are the only reason those branches are covered.

## Git Discipline

1. **Commit early, commit often** - Uncommitted work is invisible
2. **Run `cargo fmt` before changes** - Keep formatting commits separate
3. **Commit failing tests first** - Then fix in separate commit
4. **Never loosen test thresholds** - Find the real bug instead

## Feature Flags

Verified against `zenjpeg/Cargo.toml` on 2026-07-15. `Cargo.toml` is the source
of truth — re-check it before relying on this list.

```toml
[features]
default = []                                   # everything below is opt-in

# ── User-facing ──
parallel = ["dep:rayon"]                       # multi-threaded DCT/quant + parallel decode
moxcms = ["dep:moxcms"]                        # color management. Required for XYB + .correct_color()
zencodec = ["dep:zenpixels-convert", "zenpixels-convert/icc-db"]  # zencodec trait impls + ICC synth
layout = ["dep:zenresize"]                     # lossless transforms + decode→resize→encode
ultrahdr = ["ultrahdr-core/std", "ultrahdr-core/tonemap", "dep:half", "dep:zentone"]
boundary-rd = []                               # boundary-continuity refinement (#91 / PR #102)
target-zq = ["dep:zensim", "dep:zenpredict", ...] # all Quality::Zq* — see below
recompress = []                                # JPEG→JPEG recompression (decoder only, cheap)
recompress-iqa = ["recompress", "dep:zensim"]  # + closed-loop IQA refinement (Budget::MaxIterations>1)
recompress-expert = ["recompress"]             # recompress::expert — unstable, not semver-covered

# ── Internal / dev-only (`__` prefix = don't use) ──
__ffi-tests = []                               # C++ parity (needs jpegli submodule + toolchain)
__corpus-tests = []                            # corpus comparison tests
__test-utils = []                              # image generation / quality verification
__expert = []                                  # InternalParams bundle (mirrors zenwebp's __expert)
__profile = []                                 # scope-timer instrumentation
__debug-tokens = []                            # token serialization for C++ comparison
__alloc-instrument = []                        # Vec utilization logging on drop
__bdrd-trace = ["boundary-rd"]                 # per-block BD-RD refinement trace sink
__wasm-simd = []                               # WASM SIMD128 tests
__picker-research = [...]                      # needs sibling ../../zenanalyze/zenpredict checkout
```

**Flags that no longer exist** (removed; do not use — they will fail the build,
not no-op): `trellis`, `decoder`, `cms`, `archmage-simd`, `yuv`. Trellis, the
decoder, icc-db synthesis, archmage/magetypes SIMD, and zenyuv are all
unconditionally compiled. zenyuv's old `yuv = []` gate selected between
zenyuv-via-`fast_yuv` and an in-crate magetypes scalar fallback; the scalar path
was deleted, so there is one code path now.

**Recompress split:** `recompress` needs only the decoder (no heavy deps, barely
moves compile time). The closed loop is split into `recompress-iqa` because it
pulls `zensim` — same reason `target-zq` does.

**`target-zq` is the umbrella for every `Quality::Zq*` variant** (2026-07-16):
- `Quality::Zq` / `Quality::ZqExplicit` — closed loop (encode→decode→zensim→correct). Needs `zensim`.
- `Quality::ZqPicker` — realtime one-shot: the distilled picker predicts the config, single encode, **no measurement** (`achieved_score` is `NaN`). Needs `zenpredict` + `zenanalyze-api`.

The one-shot and iterative paths are a caller choice: pick the `Quality` variant.
Both retain the source pixels (`Quality::is_perceptual_target()`). Per the
2026-07-16 decision this is ONE feature (`target-zq` pulls both the metric and
the picker) rather than a split `picker` flag — so a realtime-only consumer
still compiles `zensim` it doesn't call. `zenpredict` is an unpublished sibling
path dep (`../../zenanalyze/zenpredict`), which is why **no CI job enables
`target-zq`** and the i686 manifest-strip rewrites it to `["dep:zensim"]`; it
resolves to a normal version dep once the picker + new zensim publish.
`__picker-research` is now a thin alias for `target-zq` (kept for the
`picker_v0_3_holdout_ab` dev binary).

**Boundary-RD:** Adds the post-quantization refinement from issue #91.
Disabled by default — `cargo build -p zenjpeg` (with or without trellis)
produces bit-for-bit identical output to pre-boundary-RD `origin/main`
(enforced by `tests/boundary_rd_disabled_byte_identity.rs`). Enable with
`--features boundary-rd` to unlock `EncoderConfig::boundary_rd()` and the
`BoundaryRd`/`BoundaryRdConfig` types. The knob set and defaults are
evolvable — coefficient (#94, #103) drives parameter-space exploration
with GPU-backed metrics behind this flag.

**Decoder:** The decoder API is in prerelease (always compiled — the `decoder`
flag is gone entirely, not a no-op). API will have breaking changes.

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

## aarch64 / NEON dispatch audit (2026-07-31)

**Fixed:** `encode::dct::forward_dct_8x8` had an x86_64 arm and a wasm32 arm but
NO aarch64 arm, so ARM fell through to `forward_dct_8x8_scalar`. The NEON code
already existed — `forward_dct_8x8_simd_chained_fallback` carries
`#[magetypes(v3, neon, wasm128, scalar)]`, so a `_neon` variant was generated on
every ARM build and was simply unreachable. Wiring the arm: **3.24x** on the
kernel (95.2 -> 29.4 ns), end-to-end encode 1.23x -> 1.30x vs forced scalar.
Gate: `forward_dct_8x8_dispatched_matches_scalar`.

The pre-existing tests could not have caught it — they only ever exercised the
scalar path on ARM, so they stayed green through both the bug and the fix.

**The shape to grep for:** a dispatch fn with a `#[cfg(target_arch = "x86_64")]`
arm that terminates in a bare `*_scalar(...)` call instead of
`incant!(*_fallback(...))`. The `incant!` form dispatches NEON correctly (that
is why `forward_dct_8x8_wide` was fine); the bare-scalar form silently drops ARM.

**Still open — 4 portable-looking, 1 unclear.** These have an x86 arm, no NEON
arm, no `incant!`, and end in a bare scalar call:

| site | note |
|---|---|
| `encode/linear_lut.rs::linear_to_srgb_255_x8` | x86-only `_v3`, body is generic `mt_f32x8` |
| `encode/linear_lut.rs::linear_u16_to_srgb_255_x8` | ditto |
| `encode/linear_lut.rs::linear_rgb16_to_ycbcr_x8` | ditto |
| `encode/linear_lut.rs::linear_rgbf32_to_ycbcr_x8` | ditto |
| `decode/rst_scan.rs::scan_rst_markers` | byte scan, not obviously vectorizable the same way |

**DONE 2026-07-31.** All four now have an aarch64 arm:

    linear_to_srgb_255_x8       38.6 -> 11.2 ns   3.45x
    linear_u16_to_srgb_255_x8   35.3 ->  9.7 ns   3.64x
    linear_rgb16_to_ycbcr_x8   108.7 -> 20.0 ns   5.44x
    linear_rgbf32_to_ycbcr_x8   85.9 -> 16.2 ns   5.30x

They could NOT be fixed the way the DCT was, and the reason is worth keeping:
their only sRGB step is `linear_srgb::tokens::x8::linear_to_srgb_v3`, and the
whole `tokens::x8` module is `#[cfg(target_arch = "x86_64")]`. Adding
`#[magetypes(v3, neon, wasm128)]` to that body was tried and reverted — it
generates a `_neon` variant inside a module that does not exist on aarch64.

The ARM arms are instead built on `linear_srgb::tokens::x4` (not arch-gated,
NEON-capable), running the 4-wide transfer twice per 8 values. Because that is
a different decomposition from the x86 body rather than the same code on a
different token, each kernel is gated against the wrapper's own scalar fallback
(`arm_linear_lut_tests`), with inputs spanning the sRGB piecewise knee
(0.0031308), the linear segment below it, the power segment above, exact 0/1,
and HDR values > 1 that trigger Reinhard — a transfer curve can agree in
midtones while a branch or clamp is wrong at a boundary. Verified LIVE: scaling
the x255 step to 254 fails all four.

**Where ARM is already fine:** `color/ycbcr.rs`, `decode/idct_int.rs` and
`quant/aq/simd.rs` all dispatch through `incant!` with magetypes-generated NEON
variants — their extra `X64V3Token` hits are additional x86-only fast paths on
top, not a missing ARM path. `encode/mage_simd.rs` is x86-only by design
(callers pass `Option<Desktop64>`, which is None on ARM and falls back).

**Known unrelated breakage:** `jpegli-internals-sys`'s vendored archive contains
`adaptive_quantization.cc.o` that is "not a mach-o file" on this host, which
breaks any target depending on it (`cpp_comparison` bench, the `__test-utils`
integration tests). Pre-existing. `--lib` and `--bench tier_isolation` are
unaffected.
