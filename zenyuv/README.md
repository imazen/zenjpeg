# zenyuv ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjpeg/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenyuv?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenyuv?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenyuv) ![docs.rs](https://img.shields.io/docsrs/zenyuv?style=flat-square) ![license](https://img.shields.io/crates/l/zenyuv?style=flat-square)

SIMD-optimized YUV/YCbCr color matrix conversion for BT.601, BT.709, and BT.2020. Safe Rust (`#![forbid(unsafe_code)]`), `no_std + alloc`, with native AVX2, NEON, and WASM SIMD128 kernels via [archmage](https://lib.rs/crates/archmage) token-based dispatch. Dual-licensed MIT/Apache-2.0.

## Usage

```rust
use zenyuv::{YuvContext, Range, Matrix};

let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);

// Encode: RGB -> YCbCr 4:2:0
let rgb = vec![128u8; 640 * 480 * 3]; // packed R,G,B (see "Pixel layout" below)
let mut y  = vec![0u8; 640 * 480];        // Y  = width * height
let mut cb = vec![0u8; 320 * 240];        // Cb = ceil(width/2) * ceil(height/2)
let mut cr = vec![0u8; 320 * 240];        // Cr = ceil(width/2) * ceil(height/2)
ctx.encode_420_u8(&rgb, &mut y, &mut cb, &mut cr, 640, 480);
```

`YuvContext` is reusable across frames. Internal buffers are lazy-allocated on first use -- plain u8 box-average encode allocates nothing.

## Pixel layout: input is packed **R, G, B** (3 bytes per pixel, tightly packed)

The `&[u8]` passed to every encode method is interpreted as **R first, then G, then B** -- byte `i*3+0` is red, `i*3+1` is green, `i*3+2` is blue. There is no BGR variant: passing BGR data produces a swapped-but-silent result (Cb/Cr inverted), not an error. If your source is BGR(A), swizzle to RGB before calling. Both the AVX2 fast path and the scalar/NEON/WASM paths use the identical R,G,B order, so the channel mapping is the same on every CPU.

**Buffers are tightly packed -- there is no stride parameter.** Each input row is exactly `width * 3` bytes and each output plane row is exactly its plane-width bytes, with no inter-row padding. If you have strided/padded rows (SIMD-aligned buffers, `imgref::ImgRef` with a stride, sub-region crops), pack each row into a contiguous buffer before calling, or call once per contiguous row. A native strided entry point is not part of the current API.

## API

All public conversion is done through methods on `YuvContext` (no free functions are exported). Every method returns `()` and **panics** (via `assert!` on buffer length) if any output plane is too small for the given dimensions -- see "Buffer sizes & return contract" below. There is no `Result`-returning variant.

### Encode methods (RGB -> YCbCr)

| Method | Subsampling | Output type | Notes |
|--------|-------------|-------------|-------|
| `encode_444_u8` | 4:4:4 | `u8` | Full-resolution Cb/Cr |
| `encode_420_u8` | 4:2:0 | `u8` | Box-average chroma |
| `encode_420_f32` | 4:2:0 | `f32` | u8 values widened to `f32` (0.0..=255.0); for zenjpeg's DCT pipeline |
| `encode_420_y_only_u8` | 4:2:0 | `u8` | Y plane only; caller supplies chroma elsewhere |
| `encode_sharp_420_u8` | 4:2:0 | `u8` | Sharp YUV (see below); takes `&SharpYuvConfig` |
| `encode_sharp_420_f32` | 4:2:0 | `f32` | Sharp YUV, `f32` output; takes `&SharpYuvConfig` |

There is currently **no 4:2:2 encode** and **no 4:4:4 `f32` encode** -- only the rows above exist. (4:2:2 and 4:0:0 exist on the decode side; see "Decode".)

The `sharp` module additionally exposes the underlying Sharp YUV free functions for callers that manage their own workspace or pre-seed chroma: `sharp::rgb_to_yuv420_sharp`, `sharp::refine_chroma_420_u8`, `sharp::refine_chroma_420_u8_with_workspace`, `sharp::refine_y_420_u8`, and `sharp::rgb_to_yuv420_sharp_with_workspace` / `sharp::rgb_to_yuv420_sharp_f32`. These take the same packed-RGB input and the same panic-on-undersized-buffer contract.

### Buffer sizes & return contract

For an image of `width × height`, with `cw = ceil(width/2)` and `ch = ceil(height/2)`:

| Plane | 4:4:4 size | 4:2:0 size |
|-------|------------|------------|
| `rgb` (input) | `width * height * 3` | `width * height * 3` |
| `y` | `width * height` | `width * height` |
| `cb` | `width * height` | `cw * ch` |
| `cr` | `width * height` | `cw * ch` |

- Methods take `width` and `height` explicitly; buffer lengths are validated with `assert!(buf.len() >= required)`. **Undersized buffers panic; they do not return an error.** Over-sized buffers are fine (only the leading `required` bytes are written).
- **Odd dimensions round up** for chroma: a 5-wide image has `cw = 3` chroma columns; a 5-tall image has `ch = 3` chroma rows. Size your `cb`/`cr` planes with `ceil`, not integer-truncating division, or you will panic on odd-dimension images.
- These are not streaming APIs -- one call converts the whole image (or the whole strip) you pass in.

## Output range and neutral chroma

The `Range` argument fixes the numeric range of the YCbCr output (useful for validating results):

| `Range` | Y range | Cb/Cr range | Neutral (achromatic) Cb/Cr |
|---------|---------|-------------|----------------------------|
| `Range::Full` (JFIF) | `0..=255` | `0..=255` | `128` |
| `Range::Limited` (studio/VP8) | `16..=235` | `16..=240` | `128` |

A solid gray RGB input encodes to a flat Cb = Cr = 128 in both ranges; only the Y level and the Cb/Cr excursion scale differ. (This is why a grey-128 example can't reveal channel order -- R, G, and B are equal, so a channel swap is invisible.)

## Required arguments: `Range` and `Matrix` are not optional

`YuvContext::new(range, matrix)` takes both as **required enum arguments** -- there is no "default colorspace." This is deliberate: silently guessing the matrix (BT.601 vs 709 vs 2020) or the range (full vs limited) is exactly how YUV pipelines produce washed-out or color-shifted output. You choose explicitly:

- `Matrix` -- `Bt601` (SD video, JFIF JPEG), `Bt709` (HD), `Bt2020` (UHD/HDR), plus `WebpEncoder` (libwebp's empirical `kWebpMatrix`, limited-range only; pair it with `Range::Limited` for byte-identical WebP Y). `Matrix` is `#[non_exhaustive]`.
- `Range` -- `Full` or `Limited`, as above.

JPEG always uses `Matrix::Bt601`; WebP's VP8 uses `Range::Limited`. Pick the pair your container demands.

## Decode (YCbCr -> RGB)

Decode kernels (4:4:4, 4:2:0 nearest + bilinear, 4:2:2, and 4:0:0/grayscale) exist in the source and write packed **R, G, B** output (same byte order as the encode input). **They are not part of the public 0.1.x API yet** -- the inverse path is on the Phase 3 roadmap and the functions are currently crate-internal. If you need YCbCr -> RGB today, use the `yuv` crate; this section will list the public decode entry points once they land. The relevant internal names (for tracking) are `yuv444_to_rgb`, `yuv420_to_rgb`, `yuv420_to_rgb_bilinear`, `yuv422_to_rgb`, and `yuv400_to_rgb` (each with a `_with(range, matrix)` variant).

## Performance

Encode benchmarks on a Ryzen 9 7950X (WSL2), AVX2 dispatch. Compared against the `yuv` crate v0.8 in Professional mode.

### 4:4:4 RGB to YCbCr

| Size | zenyuv | yuv crate | Delta |
|------|--------|-----------|-------|
| 256 | 14.7 us | 16.5 us | **-11%** |
| 512 | 59.3 us | 62.8 us | **-6%** |
| 1024 | 239.8 us | 244.7 us | -2% |
| 2048 | 4.94 ms | 5.00 ms | -1% |
| 4096 | 20.44 ms | 20.74 ms | -1.5% |

### 4:2:0 RGB to YCbCr

| Size | zenyuv | yuv crate | Delta |
|------|--------|-----------|-------|
| 256 | 11.0 us | 11.4 us | **-3%** |
| 512 | 45.9 us | 45.9 us | 0% |
| 1024 | 184.6 us | 175.7 us | +5% |
| 2048 | 875.6 us | 870.2 us | 0% |
| 4096 | 4.66 ms | 4.84 ms | **-4%** |

## Sharp YUV

Sharp YUV minimizes chroma reconstruction error in gamma-encoded RGB space. Standard box-average chroma subsampling (4:2:0) discards spatial information, producing visible color bleeding at high-contrast edges. Sharp YUV iteratively adjusts Cb/Cr values to minimize the L2 error between the original RGB and the reconstructed RGB after chroma upsampling.

zenyuv uses an L2-optimal Newton step: for each 2x2 chroma block, compute the exact reconstruction error using the inverse color matrix Jacobian, then apply the analytically derived correction. Two iterations converge where traditional forward-matrix gradient methods need 4+. The iteration loop is vectorized across blocks via magetypes `f32x8`, achieving a **25x speedup** over the original scalar implementation with better quality (correct Jacobian vs hand-tuned damping constants).

Configure via `SharpYuvConfig`:

```rust
use zenyuv::{YuvContext, Range, Matrix, SharpYuvConfig};

let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
let config = SharpYuvConfig::default(); // 2 Newton iterations, Y refinement on

let rgb = vec![128u8; 640 * 480 * 3]; // packed R,G,B
let mut y  = vec![0u8; 640 * 480];
let mut cb = vec![0u8; 320 * 240];
let mut cr = vec![0u8; 320 * 240];
ctx.encode_sharp_420_u8(&rgb, &mut y, &mut cb, &mut cr, 640, 480, &config);
```

After the chroma iteration, a Y refinement pass (`SharpYuvConfig::refine_y`, on by default) adjusts Y to compensate for the luma error introduced by 4:2:0 chroma subsampling, matching libwebp's `SharpYuvUpdateY`. For callers that produce initial Cb/Cr with a different averaging model (e.g., gamma-corrected downsampling), `sharp::refine_chroma_420_u8` runs the Newton iteration on pre-seeded chroma without recomputing Y.

## Feature Matrix

### Encode (RGB to YCbCr)

| | 4:4:4 | 4:2:0 | 4:2:0 Sharp |
|---|---|---|---|
| BT.601 | yes | yes | yes |
| BT.709 | yes | yes | yes |
| BT.2020 | yes | yes | yes |
| Full range | yes | yes | yes |
| Limited range | yes | yes | yes |
| u8 output | yes | yes | yes |
| f32 output | no | yes | yes |

Per-cell: `u8` output is available for 4:4:4, 4:2:0, and 4:2:0 Sharp. `f32` output is available for **4:2:0 and 4:2:0 Sharp only** (no 4:4:4 `f32` method). 4:2:2 and 4:0:0 are decode-only (not yet public; see "Decode").

## Platform Support

| Platform | ISA | Kernel | Pixels/iter |
|----------|-----|--------|-------------|
| x86-64 | AVX2+FMA | `#[arcane]` pmaddwd + pshufb deinterleave | 32 |
| aarch64 | NEON | `#[arcane]` vld3q_u8 deinterleave + vmulq | 16 |
| wasm32 | SIMD128 | `#[arcane]` i32x4_dot_i16x8 | 16 |
| All others | Scalar | magetypes `f32x8` auto-vectorized | 8 |

SIMD tier is selected at runtime via archmage token dispatch (`X64V3Token::summon()` etc.). No compile-time target features required.

## `no_std`

zenyuv is `#![no_std]` with `alloc`. The `std` feature (enabled by default) adds nothing to the public API -- disable it for embedded or WASM targets:

```toml
[dependencies]
zenyuv = { version = "0.1", default-features = false }
```

Core encode with u8 buffers requires zero heap allocation. The f32 output paths and Sharp YUV lazy-allocate internal workspace on first use.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
