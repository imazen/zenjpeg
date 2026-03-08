# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Progressive JPEG is now the default encoding mode** for both YCbCr and XYB color spaces
  - Progressive encoding produces 3-7% smaller files with no quality loss
  - XYB mode benefits even more: Rust now matches or beats C++ jpegli file sizes
    (previously 2-3% larger in baseline mode, now -0.3% to -4.3% smaller)
  - Use `.progressive(false)` to restore baseline mode if needed
  - Note: Progressive mode requires optimized Huffman tables (the default)

### Known Issues

- Progressive decoder has issues with very small images (<64 pixels) and some
  subsampled configurations. Use baseline mode (`.progressive(false)`) for reliable
  small image roundtrips. Encoder is unaffected.

## [0.3.1] - 2026-02-01

### Changed

- **`allow_16bit_quant_tables` now defaults to `false`** to match C++ cjpegli CLI behavior
  - At very low quality (Q<25), quantization values can exceed 255
  - With `false` (new default): values are clamped to 255, producing baseline JPEG (SOF0)
  - With `true`: values up to 32767 are allowed, producing extended JPEG (SOF1) when needed
  - C++ cjpegli CLI hardcodes `force_baseline=TRUE`, so this matches that behavior
  - Use `.allow_16bit_quant_tables(true)` to restore previous behavior

### Added

- **Locked values test infrastructure** (`tests/locked_values.rs`)
  - SHA-256 protected CSV files track expected encoder output hashes
  - Separate files per SIMD variant (`values_archmage.csv`, `values_wide.csv`)
  - Compile-time `#[cfg]` selection ensures each build validates its own variant
  - Requires `REGENERATE_LOCKED_VALUES=1` env var to update (always fails to force hash update)

## [0.2.0] - 2026-01-23

Renamed from `zenjpeg` with Quick Start documentation.

### Changed

- **Crate renamed from `zenjpeg` to `zenjpeg`**
  - Import path changed from `use jpegli::` to `use zenjpeg::`
  - After six rewrites and significant divergence from the original jpegli, the new name better reflects this is an independent project

- **`archmage-simd` feature is now enabled by default** for improved performance
  - Provides ~6% faster encoding on x86_64 via token-based safe SIMD intrinsics
  - *Note: As of v0.6+, archmage is a mandatory dependency and this feature flag is a no-op*

### Removed

- **Removed `spin` crate dependency** - Huffman encode tables now use `const fn` initialization
  - Reduces dependency count and improves compile times
  - No functional change; tables are computed at compile time instead of lazily at runtime

---

# Historical changelog from zenjpeg

## [0.11.1] - 2026-01-22

### Fixed

- Updated mozjpeg-rs to 0.5.1

## [0.11.0] - 2026-01-22

### Added

- **UltraHDR support** via `ultrahdr` feature
  - `ultrahdr::encode_ultrahdr()` - Encode HDR images with gainmaps
  - `ultrahdr::reencode_ultrahdr()` - Re-encode existing UltraHDR JPEGs
  - Streaming interfaces for low-memory processing
  - Requires `decoder` feature

### Fixed

- WASM compatibility: replaced `ln()` with polynomial approximation for browser support
- Various clippy warnings

## [0.10.1] - 2026-01-21

### Performance

- Removed unnecessary fuzzy erosion zeroing in AQ computation

## [0.10.0] - 2026-01-21

### Added

- WASM SIMD128 support - builds and runs with `RUSTFLAGS="-C target-feature=+simd128"`
- AArch64 NEON support via `#[multiversion]` runtime dispatch

### Changed

- All SIMD functions now use `#[multiversed]` for portable acceleration across x86_64, AArch64, and WASM

## [0.9.0] - 2026-01-20

### Breaking Changes

- **`EncoderConfig` API redesigned with separate constructors per color mode**
  - New constructors: `EncoderConfig::ycbcr(quality, subsampling)`, `EncoderConfig::xyb(quality, b_subsampling)`, `EncoderConfig::grayscale(quality)`
  - Removed `EncoderConfig::new()` - use the explicit constructors above
  - Removed `.xyb()`, `.grayscale()`, and `.ycbcr(subsampling)` builder methods
  - **Migration**: Replace `EncoderConfig::new(q, sub)` with `EncoderConfig::ycbcr(q, sub)`
  - **Migration**: Replace `EncoderConfig::new(q, sub).xyb()` with `EncoderConfig::xyb(q, XybSubsampling::BQuarter)`
  - **Migration**: Replace `EncoderConfig::new(q, sub).grayscale()` with `EncoderConfig::grayscale(q)`

- **New `XybSubsampling` enum for XYB B-channel subsampling**
  - `XybSubsampling::Full` - No B-channel subsampling (4:4:4)
  - `XybSubsampling::BQuarter` - B-channel at 2x2 downsampled (matches C++ default)

- **Removed deprecated `Quality` methods**
  - Removed `Quality::from_quality()`, `Quality::from_distance()`, `Quality::Traditional()`
  - Use `Quality::ApproxJpegli(f32)`, `Quality::ApproxButteraugli(f32)`, or `Quality::ApproxSsim2(f32)` instead
  - Quality can be passed as `f32` or `u8` directly to constructors (converts to `ApproxJpegli`)

- **`MozjpegTables::generate()` now returns `Box<EncodingTables>`** (was `QuantTableConfig`)
  - Uses `ScalingParams::Exact` since tables are already quality-scaled
  - **Migration**: Use `.tables(MozjpegTables::generate(...))` directly on `EncoderConfig`

- **Deprecated `ChromaDownsampling` in favor of `DownsamplingMethod`**
  - Both enums have the same variants, but `DownsamplingMethod` is the preferred type
  - `ChromaDownsampling` will be removed in a future version

- **Old `Encoder` and `StreamingEncoder` types made internal**
  - These are now `pub(crate)` instead of `pub`
  - Use `EncoderConfig` and its `encode_from_*()` methods instead

### Added

- **Overshoot deringing (enabled by default)**
  - Eliminates ringing artifacts on documents, screenshots, and graphics
  - No quality penalty for photographic content
  - Algorithm pioneered by [@kornel](https://github.com/kornelski) in [mozjpeg](https://github.com/mozilla/mozjpeg)
  - Uses Catmull-Rom spline interpolation to smooth hard edges
  - Disable with `.deringing(false)` if needed (not recommended)
  - See README "Overshoot Deringing" section for technical details

- **`TrellisConfig` API for mozjpeg-compatible trellis quantization** (requires `experimental-hybrid-trellis` feature)
  - `TrellisConfig::default()` - Standard trellis with AC and DC optimization
  - `TrellisConfig::disabled()` - No trellis (fastest)
  - `TrellisConfig::favor_size()` / `TrellisConfig::favor_quality()` - Presets
  - Builder methods: `.ac_trellis()`, `.dc_trellis()`, `.speed_level()`, `.rd_factor()`
  - Use via `EncoderConfig::ycbcr(...).trellis(TrellisConfig::default())`

- **`archmage-simd` feature for token-based safe SIMD**
  - Alternative to `unsafe_simd` using the archmage crate
  - Provides AVX2+FMA intrinsics with compile-time capability tokens
  - New SIMD functions for DCT, color conversion, and AQ computation
  - *Note: As of v0.6+, archmage is a mandatory dependency and this feature flag is a no-op*

- **New encoder methods**
  - `.force_baseline()` - Convenience method for maximum compatibility (SOF0)
  - `.allow_16bit_quant_tables(bool)` - Control extended sequential (SOF1) vs baseline
  - `.get_trellis()` - Accessor for trellis configuration

- **`EncodingTables` struct for quantization table experimentation**
  - `PerComponent<T>` wrapper with named accessors for YCbCr (Y/Cb/Cr) and XYB (X/Y/B)
  - `ScalingParams` enum: `Exact` (no scaling) or `Scaled { global_scale, frequency_exponents }`
  - `EncodingTables::default_ycbcr()` and `::default_xyb()` factory methods
  - Helper methods: `scale_quant()`, `blend()`, `to_quant_config()`, `to_zero_bias_config()`
  - `dct` module with `freq_distance()`, `IMPORTANCE_ORDER`, `to_zigzag()` helpers
  - Use via `EncoderConfig::ycbcr(...).tables(my_tables)` for custom tables
  - See README "Table Optimization" section for research methodology

### Changed

- **SIMD function names now match actual instruction sets used**
  - `extract_r/g/b_sse` → `extract_r/g/b_ssse3` (uses `_mm_shuffle_epi8`)
  - `transpose_8x8_avx2` → `transpose_8x8_avx` (uses only AVX)
  - `forward_dct_8x8_avx2` → `forward_dct_8x8_fma` (uses FMA)
  - `rgb_to_ycbcr_8px_avx2` → `rgb_to_ycbcr_8px_fma` (uses FMA)

- **AQ computation converted to FMA operations**
  - Patterns like `a * b + c` now use `mul_add()` for better precision
  - Tests use relative epsilon (1e-6) to handle FMA hardware differences

### Performance

- **Padded AQ buffers optimization** - 24-50% faster on some encoding paths
  - StreamingAQ now uses MCU-aligned `padded_width` for buffer stride
  - Enables full SIMD processing without edge case scalar fallbacks

### Fixed

- Separated `stride` and `img_width` parameters in AQ modulation for correct edge handling
- `hf_modulation_sum_8x8` now uses consistent 8-pixel rows (buffer is MCU-padded)

## [0.8.1] - 2026-01-19

### Added

- **`YCbCrPlanarEncoder` now fully implemented** for video pipeline integration
  - Previously `finish()` returned an error - was a stub
  - Uses streaming architecture with internal partial strip buffering
  - Push any number of rows - buffers internally until MCU-aligned (8 rows for 4:4:4, 16 for 4:2:0)
  - Automatic chroma downsampling when pushing full-resolution Cb/Cr planes
  - Supports ICC profile, EXIF, and XMP metadata injection
  - 13 comprehensive tests added

### Fixed

- **`ZeroBiasConfig::Default` now properly auto-selects based on color mode**
  - Previously always used YCbCr perceptual tables even in XYB mode
  - Now correctly uses: YCbCr tables for YCbCr mode, XYB tables (0.5/0.5) for XYB mode
  - This matches C++ jpegli behavior exactly

## [0.8.0] - 2026-01-18

### Breaking Changes

- **`ZeroBiasConfig` enum redesigned for clarity**
  - `Perceptual` renamed to `Default` - uses YCbCr perceptual tables
  - New `YCbCr` variant - alias for Default (explicit clarity)
  - New `Xyb` variant - use XYB 0.5 tables (for experimentation)
  - `Disabled` and `Custom` remain unchanged
  - **Migration**: Replace `ZeroBiasConfig::Perceptual` with `ZeroBiasConfig::Default`

## [0.7.3] - 2026-01-18

### Fixed

- **Critical XYB color conversion bug in SIMD inplace functions**
  - Three SIMD inplace functions had incorrect scaling formulas causing severe color distortion
  - Affected: `srgb_to_scaled_xyb_planes_simd_inplace`, `srgb_to_scaled_xyb_planes_simd_rgba_inplace`, `srgb_to_scaled_xyb_planes_simd_bgra_inplace`
  - Bug: Used `val * scale + offset` instead of correct `(val + offset) * scale`
  - B channel additionally missing `- y` term: used `b * scale + offset` instead of `(b - y + offset) * scale`
  - Impact: X channel error up to 33%, B channel (blue) significantly darkened
  - Non-inplace SIMD functions (`srgb_to_scaled_xyb_planes_simd`, etc.) were already correct

### Added

- Unit tests for B channel scaling verification (`test_b_channel_*`)
  - `test_b_channel_scaling_formula` - verifies correct scaling math
  - `test_b_channel_simd_inplace_vs_scalar` - SIMD matches scalar reference
  - `test_b_channel_rgba_bgra_inplace_vs_scalar` - RGBA/BGRA variants
  - `test_b_channel_blue_heavy_colors` - tests where bug was most visible

## [0.7.2] - 2026-01-18

### Breaking Changes

- **`QuantTableConfig` now has separate `cb` and `cr` fields** instead of shared `chroma`
  - `CustomBase { luma, chroma }` → `CustomBase { luma, cb, cr }`
  - `Exact { luma, chroma }` → `Exact { luma, cb, cr }`
  - Allows independent control of blue and red chroma quantization

### Added

- **`ZeroBiasConfig` API** for controlling coefficient rounding during quantization
  - `ZeroBiasConfig::Perceptual` (default) - quality-aware perceptual tuning
  - `ZeroBiasConfig::Disabled` - no zero biasing
  - `ZeroBiasConfig::Custom { luma, cb, cr }` - full control with per-component (mul, offset) arrays
  - New `.zero_bias(config)` builder method on `EncoderConfig`

- **`encoder::tables` module** exposing default quantization and zero-bias tables
  - `BASE_QUANT_YCBCR`, `BASE_QUANT_XYB`, `BASE_QUANT_STD` - base quantization matrices
  - `ZERO_BIAS_MUL_YCBCR_LQ/HQ`, `ZERO_BIAS_OFFSET_*` - zero-bias tables
  - Helper functions: `luma_from_192()`, `cb_from_192()`, `cr_from_192()`, `pack_192()`
  - Enables users to modify defaults rather than starting from scratch

## [0.7.1] - 2026-01-18

### Performance

- **59% faster linear RGB encoding** (RgbF32Linear, Rgb16Linear, etc.)
  - SIMD-accelerated linear→sRGB conversion using `linear-srgb` crate
  - 8-wide f32x8 processing with `#[inline(always)]` for hot paths
  - Optimized memory loads with `bytemuck::pod_read_unaligned`
  - 1024×1024 RgbF32Linear: 41ms → 17ms (2.4x faster)

### Changed

- Updated `linear-srgb` to 0.3.1 (improved SIMD performance)
- Updated `whereat` to 0.1.3

## [0.7.0] - 2026-01-18

### Breaking Changes

- **`EncoderConfig::new()` now requires quality and subsampling parameters**
  - Before: `EncoderConfig::new().quality(85)`
  - After: `EncoderConfig::new(85, ChromaSubsampling::Quarter)`
  - This makes quality and subsampling explicit required choices

- **Error type restructured with hierarchical categories**
  - Errors now use `thiserror` with location tracking (`#[track_caller]`)
  - Error variants reorganized into logical groups
  - No longer `Clone` or `PartialEq` (now contains stack traces)

- **`.exif()` method signature changed**
  - Before: `.exif(raw_bytes)` accepting `impl Into<Vec<u8>>`
  - After: `.exif(Exif::raw(bytes))` or `.exif(Exif::build().orientation(...))`
  - Compile-time separation between raw bytes and field-based building

### Added

- **Type-safe EXIF builder** with compile-time separation of raw vs field modes
  - `Exif::raw(bytes)` - use raw TIFF bytes
  - `Exif::build().orientation(Orientation::Rotate90).copyright("© 2024")` - build from fields
  - `Orientation` enum with all 8 EXIF orientation values
  - `ExifFields` builder for orientation and copyright tags

- **XMP metadata support**
  - `.xmp(data)` method on `EncoderConfig` for embedding XMP metadata
  - Proper APP1 marker with Adobe XMP namespace

- **Performance improvements**
  - Optimized Huffman encoding hot path (SIMD frequency collection)
  - Reduced memcpy overhead in streaming encoder
  - Lazy error evaluation in entropy encoder (13% speedup)

### Fixed

- Dead code warning for prerelease decoder API
- Build failures across targets and features
- Quality formula alignment with C++ jpegli

## [0.6.0] - 2026-01-15

_Internal refactoring release - no public API changes_

## [0.5.0] - 2026-01-14

_Internal refactoring release - no public API changes_

## [0.4.1] - 2026-01-11

### Added

- CI workflow testing on 6 platform targets (Linux/macOS/Windows x x64/ARM64)
- Benchmark tracking workflow with regression detection

## [0.4.0] - 2026-01-11

### Added

- **JpegEncoder API**: New recommended encoder with row-by-row input
  - `JpegEncoder::new(width, height).start()` - Builder pattern for configuration
  - `push_row()` / `push_rows()` - Incremental row input
  - `push_row_with_stop()` / `push_rows_with_stop()` - With cancellation support
  - `encode_all()` - Convenience method for single-call encoding
  - `estimate_memory_usage()` - Predict peak memory before encoding
  - ~50% lower peak memory vs legacy Encoder
  - 16-20% faster than legacy Encoder at 1080p+ resolutions
  - Full progressive mode support
- Multi-decoder compatibility test: validates zenjpeg output works with jpeg-decoder, zune-jpeg, and mozjpeg
- Butteraugli-based quality thresholds in decoder compatibility tests
- **Quality conversion API**: `QualityConversion` and `QualityComparisonMetric` for matching other encoders
  - `QualityConversion::mozjpeg_equivalent()` - Convert mozjpeg quality to equivalent jpegli quality
  - `QualityConversion::try_mozjpeg_equivalent()` - Same but returns `None` for unmapped values
  - Supports DSSIM, SSIMULACRA2, and Butteraugli metrics
  - Pre-computed tables for 4:4:4 and 4:2:0 subsampling modes
- `Encoder::jpegli_quality()` - Explicit method for setting jpegli native quality
- `Encoder::equivalent_quality()` - Set quality by matching another encoder

### Changed

- Updated butteraugli to 0.3.1 with `unsafe-perf` feature for 1.5x faster quality metrics
- Updated zune-jpeg to 0.5 (API changes for ZCursor wrapper)
- Updated mozjpeg-rs to 0.2.5
- Updated `enough` crate to 0.2.0 (Stopper moved to `almost-enough` crate)
- All encoder allocations are now fallible (returns `Error::AllocationFailed` instead of panicking)
- Eliminated HuffmanEncodeTable/HuffmanDecodeTable clones in hot paths

### Deprecated

- **`Encoder` struct** - Use `JpegEncoder` instead for better performance and lower memory usage
- `Encoder::quality()` - Use `jpegli_quality()` or `equivalent_quality()` instead

### Fixed

- Strip encoder edge padding for grayscale parity with full-plane encoder
- Progressive non-interleaved block count for non-MCU-aligned images
- Handle UnexpectedEof in progressive DC scans

### Notes

- **Decoder Laplacian biases (expected behavior)**: The zenjpeg decoder produces slightly
  different output than standard decoders (jpeg-decoder, zune-jpeg) because it uses Laplacian
  dequantization biases matching C++ djpegli. This shifts reconstructed values toward zero
  and typically improves quality for photographic content. For synthetic test images, this
  may result in higher (worse) butteraugli scores vs the original, but this matches the
  intended C++ jpegli behavior.

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

[0.4.0]: https://github.com/imazen/zenjpeg/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/imazen/zenjpeg/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/imazen/zenjpeg/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/imazen/zenjpeg/releases/tag/v0.1.0
