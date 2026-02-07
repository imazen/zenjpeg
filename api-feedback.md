# API Feedback - zencodecs Integration

**Date:** 2026-02-06
**Context:** Implementing JPEG codec adapter in zencodecs (src/codecs/jpeg.rs)

## Issues Encountered

### 1. Feature gate not obvious
**Issue:** Compilation error "could not find `decoder` in `zenjpeg`"
**Solution:** Had to enable `features = ["decoder"]` in Cargo.toml
**Feedback:** The error message didn't make it clear that a feature gate was the problem. Consider mentioning required features in compilation errors or top-level docs.

### 2. Private modules confusion
**Issue:** Tried to use `zenjpeg::types::PixelFormat` but module is private
**Solution:** Use `zenjpeg::decoder::PixelFormat` and `zenjpeg::encoder::PixelLayout` instead
**Feedback:** Having `PixelFormat` in decoder and `PixelLayout` in encoder is confusing when they represent the same concept. Consider unifying or re-exporting at top level.

### 3. Type inconsistency: PixelFormat vs PixelLayout
**Issue:** Decoder result has `PixelFormat`, but encoder expects `encoder::PixelLayout`
**Solution:** Had to write separate conversion functions for each
**Feedback:** These should be the same type or have explicit conversion methods. Current approach requires maintainers to keep both enums in sync manually.

### 4. No ICC profile extraction
**Issue:** Expected `icc_profile()` method on decoder result, doesn't exist
**Solution:** Return `None` with TODO comment in adapter
**Feedback:** JPEG files commonly have ICC profiles. Would be useful to expose this metadata from the decoder.

### 5. Limits/Stop integration unclear
**Issue:** Couldn't figure out how to pass zencodecs `Limits` or `Stop` token through to decoder/encoder
**Solution:** Simplified to `Decoder::new().decode(data, enough::Unstoppable)` without configuration
**Feedback:** The three-layer pattern (Config → Request → Encoder/Decoder) is powerful but lacks simple usage examples. Consider adding a "Quick Start" section showing the simplest decode/encode path.

## Current Implementation

```rust
// Decode
let decoder = zenjpeg::decoder::Decoder::new();
let result = decoder.decode(data, enough::Unstoppable)?;
let pixels = result.pixels_u8()?.to_vec();

// Encode
let quality = quality.unwrap_or(85.0).clamp(0.0, 100.0) as u8;
let config = zenjpeg::encoder::EncoderConfig::ycbcr(
    quality,
    zenjpeg::encoder::ChromaSubsampling::Quarter,
);
let pixel_layout = to_jpeg_encoder_layout(layout);
let mut encoder = config.encode_from_bytes(width, height, pixel_layout)?;
encoder.push_packed(pixels, enough::Unstoppable)?;
let jpeg_data = encoder.finish()?;
```

## Recommendations

1. **Add simple usage examples** - Show the minimal decode/encode path in README or module docs
2. **Unify pixel format types** - Either use the same enum for encoder/decoder or provide explicit conversions
3. **Expose ICC profiles** - Add method to extract ICC profile from decoded JPEG
4. **Document feature flags** - Clearly list all available features and when they're needed
5. **Consider convenience functions** - Like `decode_to_rgb(data, quality)` for common cases

## What Worked Well

- The `enough::Unstoppable` pattern for cancellation is clean
- Quality parameter being 0-100 (native JPEG scale) is intuitive
- `ChromaSubsampling` enum is clear and self-documenting
- `encode_from_bytes()` method makes pixel ownership straightforward
