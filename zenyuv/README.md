# zenyuv ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjpeg/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenyuv?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenyuv?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenyuv) ![docs.rs](https://img.shields.io/docsrs/zenyuv?style=flat-square) ![license](https://img.shields.io/crates/l/zenyuv?style=flat-square)

SIMD-optimized YUV/YCbCr color matrix conversion for BT.601, BT.709, and BT.2020. Safe Rust (`#![forbid(unsafe_code)]`), `no_std + alloc`, with native AVX2, NEON, and WASM SIMD128 kernels via [archmage](https://lib.rs/crates/archmage) token-based dispatch. Dual-licensed MIT/Apache-2.0.

## Usage

```rust
use zenyuv::{YuvContext, Range, Matrix};

let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);

// Encode: RGB -> YCbCr 4:2:0
let rgb = vec![128u8; 640 * 480 * 3];
let mut y  = vec![0u8; 640 * 480];
let mut cb = vec![0u8; 320 * 240];
let mut cr = vec![0u8; 320 * 240];
ctx.encode_420_u8(&rgb, &mut y, &mut cb, &mut cr, 640, 480);
```

`YuvContext` is reusable across frames. Internal buffers are lazy-allocated on first use -- plain u8 box-average encode allocates nothing.

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
let config = SharpYuvConfig::default(); // 2 Newton iterations

let rgb = vec![128u8; 640 * 480 * 3];
let mut y  = vec![0u8; 640 * 480];
let mut cb = vec![0u8; 320 * 240];
let mut cr = vec![0u8; 320 * 240];
ctx.encode_sharp_420_u8(&rgb, &mut y, &mut cb, &mut cr, 640, 480, &config);
```

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
| f32 output | yes | yes | yes |

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

Core encode/decode with u8 buffers requires zero heap allocation. The f32 output paths and Sharp YUV lazy-allocate internal workspace on first use.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
