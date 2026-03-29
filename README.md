# zenjpeg [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjpeg/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/imazen/zenjpeg/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenjpeg?style=flat-square)](https://crates.io/crates/zenjpeg) [![docs.rs](https://img.shields.io/docsrs/zenjpeg?style=flat-square)](https://docs.rs/zenjpeg) [![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zenjpeg?style=flat-square)](https://github.com/imazen/zenjpeg#license)

zenjpeg is a pure Rust JPEG encoder and decoder ported from Google's
[jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli).

It produces smaller files at the same visual quality as libjpeg-turbo and
mozjpeg through adaptive quantization, optional trellis optimization, and
XYB perceptual color space support. The decoder is faster than both mozjpeg
and zune-jpeg across all image sizes, with parallel decoding reaching 10x
the throughput of mozjpeg on 4K images. `#![forbid(unsafe_code)]`, no C
dependencies.

## Quick start

### Encode

```rust,ignore
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};

let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .progressive(true);

// One-shot encode from an RGB pixel buffer
let jpeg: Vec<u8> = config.encode(&rgb_pixels, width, height)?;
```

### Decode

Requires the `decoder` feature:

```toml
[dependencies]
zenjpeg = { version = "0.6", features = ["decoder"] }
```

```rust,ignore
use zenjpeg::decoder::{Decoder, DecodeResult};

let result: DecodeResult = Decoder::new()
    .decode(&jpeg_data, enough::Unstoppable)?;
let pixels: &[u8] = result.pixels_u8().expect("u8 output");
let (width, height) = (result.width, result.height);
```

### Streaming encode

For large images or memory-constrained environments, push rows incrementally:

```rust,ignore
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, PixelLayout, Unstoppable};

let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .progressive(true);

let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
for chunk in rgb_bytes.chunks(1920 * 3 * 16) {
    enc.push_packed(chunk, Unstoppable)?;
}
let jpeg: Vec<u8> = enc.finish()?;
```

## Performance

Measured on a Ryzen 9 7950X (WSL2), default compiler target (no
`-C target-cpu=native`), using noise+patches test patterns. Full benchmark
tables and methodology are in the
[CLAUDE.md project guide](https://github.com/imazen/zenjpeg/blob/main/CLAUDE.md).

### Encoder

The encoder is ~15-20% slower than C++ jpegli (median 1.15-1.2x), with
effectively identical output quality (size within +0.63%, DSSIM within
+0.41%). `auto_optimize(true)` adds hybrid trellis quantization for +1.5
SSIMULACRA2 over C++ jpegli at matched bit rate.

### Decoder (sequential, baseline 4:2:0)

| Image size | vs mozjpeg | vs zune-jpeg | vs C++ jpegli |
|------------|------------|--------------|---------------|
| 512x512    | **0.80x**  | **0.86x**    | 0.55x         |
| 2048x2048  | **0.94x**  | **0.89x**    | 0.61x         |
| 4096x4096  | **0.95x**  | **0.91x**    | 0.65x         |

Values < 1.0 mean zenjpeg is faster. mozjpeg here is libjpeg-turbo with
NASM SIMD (C).

### Decoder (progressive, sequential, no DRI)

| Image size | vs mozjpeg | vs zune-jpeg |
|------------|------------|--------------|
| 512x512    | **0.60x**  | **0.55x**    |
| 2048x2048  | **0.62x**  | **0.57x**    |
| 4096x4096  | **0.69x**  | **0.63x**    |

1.4-1.8x faster than mozjpeg, 1.6-1.8x faster than zune-jpeg on
progressive files.

### Decoder (parallel, `--features parallel`, baseline 4:2:0)

| Image size | vs mozjpeg | parallel speedup |
|------------|------------|------------------|
| 1024x1024  | **0.37x**  | 2.6x             |
| 2048x2048  | **0.16x**  | 5.8x             |
| 4096x4096  | **0.10x**  | 9.5x             |

Fused parallel decode (entropy + IDCT + color convert per restart segment
via rayon). Requires restart markers; progressive images show no parallel
benefit.

## Encoder features

### Color modes

- **YCbCr** -- Standard JPEG. Maximum decoder compatibility.
- **XYB** -- Perceptual color space from JPEG XL. Better quality per byte,
  but requires a jpegli-aware decoder (libjxl, zenjpeg).
- **Grayscale** -- Single-channel output.

```rust,ignore
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, XybSubsampling, Quality};

let ycbcr = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
let xyb   = EncoderConfig::xyb(85, XybSubsampling::BQuarter);
let gray  = EncoderConfig::grayscale(85);

// Quality scales: target a perceptual metric instead of an arbitrary number
let ssim  = EncoderConfig::ycbcr(Quality::ApproxSsim2(90.0), ChromaSubsampling::None);
let btrgl = EncoderConfig::ycbcr(Quality::ApproxButteraugli(1.0), ChromaSubsampling::Quarter);
let moz   = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(80), ChromaSubsampling::Quarter);
```

### Optimization presets

Eight presets covering the jpegli, mozjpeg, and hybrid optimization
families:

| Preset | Scan type | AQ | Trellis | Tables | Notes |
|--------|-----------|-----|---------|--------|-------|
| `JpegliBaseline` | Baseline | Yes | No | 3 | Matches `cjpegli --jpeg_type baseline` |
| `JpegliProgressive` | Progressive | Yes | No | 3 | Default. Matches `cjpegli` output. |
| `MozjpegBaseline` | Baseline | No | Full | 2 | Matches C mozjpeg baseline profile |
| `MozjpegProgressive` | Progressive | No | Full | 2 | Matches C mozjpeg progressive profile |
| `MozjpegMaxCompression` | Progressive | No | Full | 2 | Matches `JCP_MAX_COMPRESSION` + scan search |
| `HybridBaseline` | Baseline | Yes | Adaptive | 3 | Best of both: jpegli AQ + trellis |
| `HybridProgressive` | Progressive | Yes | Adaptive | 3 | Best quality/size tradeoff |
| `HybridMaxCompression` | Progressive | Yes | Full | 3 | Slowest, smallest files |

Or skip presets and use `auto_optimize(true)` for CMA-ES butteraugli-tuned
parameters (equivalent to `HybridMaxCompression` with lambda 14.5):

```rust,ignore
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .auto_optimize(true);
```

### Chroma subsampling

4:4:4, 4:2:0, 4:2:2, and 4:4:0. Downsampling methods: `Box`,
`GammaAware`, and `SharpYUV` (via the `yuv` feature, default).

### Other encoder capabilities

- **Progressive JPEG** with jpegli, mozjpeg, and exhaustive scan scripts
- **Streaming row-by-row API** -- no whole-image buffering
- **Parallel encoding** via rayon (`parallel` feature)
- **19 pixel input layouts** including RGB/BGR/RGBA/BGRA in 8-bit, 16-bit,
  and f32
- **ICC profile embedding** and EXIF metadata builder
- **Per-image metadata** via the three-layer pattern
  (`EncoderConfig` -> `EncodeRequest` -> encoder)
- **Memory estimation** -- `estimate_memory()` and
  `estimate_memory_ceiling()` before encoding
- **Cancellation** -- cooperative `Stop` token on all push methods

## Decoder features

The decoder API is in prerelease. Enable with `features = ["decoder"]`.

- **Streaming decode** -- single-pass for baseline, multi-pass for
  progressive
- **Scanline reader** -- `ScanlineReader` for row-by-row output with
  bounded memory
- **4 strictness levels** -- `Strict`, `Balanced`, `Lenient`, `Permissive`
- **Parallel decode** -- wave-parallel and full-buffer modes via rayon
  (`parallel` feature)
- **Fancy upsampling** -- bilinear chroma upsampling (disable with
  `fancy_upsampling(false)` for ~5-10% speed gain)
- **Crop regions** -- decode a sub-rectangle without decoding the full
  image
- **Auto-orient** -- lossless DCT-domain rotation from EXIF orientation
- **CMYK support** and raw coefficient access (`DecodedCoefficients`)
- **Metadata preservation** -- EXIF, ICC, XMP, JFIF, Adobe, MPF segments
- **Deblocking filters** -- optional H.264-style / Knusperli post-processing
- **Dequantization bias** -- Laplacian bias for improved decode quality at
  low Q (matches C++ jpegli decoder behavior)
- **Resource limits** -- `max_pixels`, `max_memory` for DoS protection

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `decoder` | No | JPEG decoding (enables `zenjpeg::decoder` module) |
| `parallel` | No | Multi-threaded encode/decode via rayon |
| `trellis` | Yes | Rate-distortion trellis quantization (mozjpeg-style Viterbi) |
| `yuv` | Yes | Fast SIMD RGB-to-YCbCr via the `yuv` crate (10-150x faster) |
| `cms-lcms2` | No | ICC color management via lcms2 (needed for XYB decode) |
| `cms-moxcms` | No | Pure Rust ICC color management (alternative to lcms2) |
| `ultrahdr` | No | UltraHDR HDR gain map encode/decode (implies `decoder`) |
| `layout` | No | Lossless transforms + decode-resize-encode pipeline |

### Common configurations

```toml
# Encode only (default features)
zenjpeg = "0.6"

# Decode + encode
zenjpeg = { version = "0.6", features = ["decoder"] }

# Server workload (parallel encode + decode)
zenjpeg = { version = "0.6", features = ["decoder", "parallel"] }

# Minimal (no trellis, no SharpYUV)
zenjpeg = { version = "0.6", default-features = false }

# UltraHDR
zenjpeg = { version = "0.6", features = ["ultrahdr"] }
```

## UltraHDR

With `features = ["ultrahdr"]`, zenjpeg can encode and decode HDR gain maps
following the Ultra HDR specification. The codec handles JPEG assembly,
MPF structure, and XMP metadata; the caller provides tonemapped SDR, gain
map data, and ICC profiles.

## Lossless transforms

With `features = ["decoder"]`, DCT-domain rotate/flip operations are
available with zero generation loss. The `layout` feature adds a
decode-resize-encode pipeline integrating `zenlayout` and `zenresize`.

## SIMD and platform support

SIMD dispatch is handled by [archmage](https://github.com/imazen/archmage)
at runtime -- no compile-time target flags required. Supported
instruction sets:

- **x86_64**: AVX2/FMA, AVX-512
- **wasm32**: SIMD128 (1.6-1.7x encode, 1.5-2.0x decode speedup)
- Scalar fallback on all other targets

CI tests on 6 OS targets: Ubuntu x86_64, Ubuntu ARM64, macOS ARM64,
macOS Intel, Windows x86_64, and Windows ARM64.

## Limitations

- The encoder is ~15-20% slower than C++ jpegli. The gap is mostly in the
  DCT where C++ Highway AVX-512 outperforms Rust AVX2. Other hot paths
  (AQ, entropy coding) are at parity.
- XYB mode produces 5-11% larger files than C++ jpegli at equivalent
  visual quality. The quality is identical; the size gap is under
  investigation.
- The decoder API is in prerelease and will have breaking changes.
- Progressive images do not benefit from parallel decoding (no restart
  segments to parallelize).
- NEON SIMD is not yet implemented. ARM targets use the scalar fallback.

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs<sup>[1]</sup> | **zenjpeg** · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sup>[1]</sup> <sub>as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software -- and the 40+
library ecosystem it depends on -- full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** -- $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key](https://www.imazen.io/pricing)
- **Commercial subscription** -- Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial](https://www.imazen.io/pricing)
- **AGPL v3** -- Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
