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

---

## Round 2: Limits/Stop/Config Wiring (2026-02-07)

**Context:** Wiring up Limits enforcement, Stop forwarding, and codec config passthrough in all zencodecs adapters.

### 6. Probe requires full decode

**Issue:** `read_info()` returns `JpegInfo` but doesn't extract ICC/EXIF/XMP. The adapter has to do a full `decode()` just for `probe()` (metadata-only inspection).

**Fix:** `read_info()` should parse APP markers and expose ICC/EXIF/XMP, or there should be a `read_info_with_extras()` that does.

### 7. Limits are public fields, not builder methods

**Issue:** `DecodeConfig` uses builder methods for everything (`output_format()`, `chroma_upsampling()`, `fancy_upsampling()`, etc.) but `max_pixels` and `max_memory` are bare `pub` fields:
```rust
dc.max_pixels = max_px;
dc.max_memory = max_mem as usize;
```
Every other setting uses `self` consuming builders. Inconsistent.

**Fix:** Add `with_max_pixels()` and `with_max_memory()` builder methods to match the rest of the API.

### 8. `max_memory` is `usize`, not `u64`

**Issue:** zenwebp, zengif, and zencodecs all use `u64` for memory limits. zenjpeg uses `usize`. Requires a cast in the adapter (`max_mem as usize`). On 32-bit targets this silently truncates.

**Fix:** Change to `u64` for consistency across the ecosystem.

### 9. Encoder metadata methods require owned `Vec<u8>`

**Issue:** `EncoderConfig::icc_profile()`, `xmp()` take owned `Vec<u8>`. The adapter has borrowed `&[u8]` from `ImageMetadata` and must clone:
```rust
config = config.icc_profile(icc.to_vec());
config = config.xmp(xmp.to_vec());
```
`EncodeRequest` has `_owned` variants, but `EncoderConfig` doesn't have borrowed variants.

**Fix:** Accept `impl Into<Cow<'_, [u8]>>` or add `icc_profile_ref(&[u8])` methods, similar to how `EncodeRequest` has both borrowed and owned variants.

### 10. `Decoder` vs `DecodeConfig` naming ambiguity

**Issue:** The main decode type is `DecodeConfig` but it has methods like `decode()`, `scanline_reader()`, `ultrahdr_reader()` — it *is* the decoder. The old name `Decoder` exists as a type alias. Both are in scope causing confusion.

**Fix:** Pick one name. If it's the config+decoder combined, `Decoder` is more natural (users call `.decode()` on it). If splitting config from execution, make them separate types.

### 11. `decode_f32()` is redundant

**Issue:** `decode_f32()` is identical to `decode()` with `.output_target(OutputTarget::SrgbF32)`. It's extra API surface with no additional capability.

**Suggestion:** Deprecate `decode_f32()` in favor of the `OutputTarget` approach.

## What Worked Well (Round 2)

- `DecodedExtras` API for segment preservation is excellent — `segments()`, `mpf()`, `secondary_images()`, `gainmap()`, `to_encoder_segments()` is a clean roundtrip story
- `GainMapHandling` enum gives fine-grained control over gain map processing cost
- `PreserveConfig` letting callers choose what to preserve is good
- The `EncodeRequest` → `RgbEncoder` → `push_packed()` → `finish()` streaming encode pipeline is clean
- `EncoderSegments::add_gainmap()` makes gain map roundtrip trivial
