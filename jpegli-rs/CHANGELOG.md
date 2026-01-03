# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Multi-decoder compatibility test: validates jpegli-rs output works with jpeg-decoder, zune-jpeg, and mozjpeg
- Butteraugli-based quality thresholds in decoder compatibility tests

### Changed

- Updated butteraugli to 0.3.1 with `unsafe-perf` feature for 1.5x faster quality metrics
- Updated zune-jpeg to 0.5 (API changes for ZCursor wrapper)
- Updated mozjpeg-rs to 0.2.5

## [0.3.0] - 2026-01-02

### Added

- XYB progressive mode: proper support with APP14 Adobe marker and ICC profile embedding
- Restart marker sequence validation (RST0-RST7 cycling) matching libjpeg behavior

### Changed

- **BREAKING**: Renamed feature `hybrid-trellis` to `experimental-hybrid-trellis`
  - This feature is experimental and its parameters are not statistically validated
- Renamed dependency `mozjpeg-oxide` to `mozjpeg-rs`
- Updated hybrid trellis documentation with caveats about limited testing

### Fixed

- Restart marker decoding: explicit marker validation instead of silent skip
- Decoder now properly resets DC predictors and aligns to byte boundary at restart intervals

## [0.2.0] - 2025-12-28

### Added

- APP14 Adobe marker for XYB mode, improving decoder compatibility ([google/jpegli#135](https://github.com/google/jpegli/pull/135))
- `#[non_exhaustive]` attribute on all public enums for API stability:
  - `ColorSpace`, `PixelFormat`, `SampleDepth`, `Subsampling`, `JpegMode`, `Quality`
- `DecodedImage` helper methods: `dimensions()`, `bytes_per_pixel()`, `stride()`
- Re-exports at crate root: `JpegMode`, `Subsampling`, `DecodedImage`

### Changed

- Internal modules (`huffman`, `dct`, `entropy`, etc.) are now `#[doc(hidden)]`
  - Still accessible but not part of stable public API
- Renamed dependency from `butteraugli-oxide` to `butteraugli`

### Fixed

- Progressive JPEG decoder: AC refinement ZRL handling
- DC prediction in XYB Huffman table optimization

## [0.1.0] - 2025-12-27

### Added

- Initial release
- Pure Rust JPEG encoder with jpegli-compatible output
- Baseline and progressive JPEG encoding
- Adaptive quantization for perceptual optimization
- XYB color space support with ICC profile embedding
- Chroma subsampling: 4:4:4, 4:2:2, 4:2:0, 4:4:0
- Optimized Huffman table generation
- JPEG decoder with ICC profile support
- Optional CMS backends: lcms2, moxcms
- SIMD acceleration via `wide` crate
- Butteraugli quality metric integration

[0.3.0]: https://github.com/imazen/jpegli-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/imazen/jpegli-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/imazen/jpegli-rs/releases/tag/v0.1.0
