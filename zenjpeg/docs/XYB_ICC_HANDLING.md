# XYB color handling and ICC profile flow in zenjpeg

This document explains how zenjpeg's XYB-encoded JPEGs work end to end:
how the encoder embeds the XYB ICC profile, what's actually stored in the
8-bit JPEG channels, and how the decoder produces correct sRGB output
both with and without an explicit color-management call.

## Why XYB JPEGs need an ICC profile

A JPEG container only knows about Y/Cb/Cr or R/G/B channels — the JPEG
spec has no concept of a perceptual color space like XYB. To smuggle XYB
data through an off-the-shelf JPEG decoder, encoders use the same trick
libjxl/jpegli use:

1. Convert RGB → linear → XYB (libjxl opsin matrix + cube root)
2. Scale and offset XYB into the 0..255 byte range
3. Store the scaled XYB as if it were R/G/B channels in the JPEG
4. Embed an ICC profile that *describes the RGB→XYB transform*, so a
   color-managed decoder will undo it back to display-referred sRGB

A naive decoder that ignores ICC sees garbage colors (because it
interprets scaled XYB as RGB). A color-managed decoder sees correct
sRGB. zenjpeg's decoder takes a third path — it recognizes the XYB ICC,
skips the standard JPEG color-conversion, and runs its own
SIMD-optimized inverse-XYB → sRGB transform.

## The encoder side

### What gets embedded

For every XYB encode, zenjpeg writes the following marker sequence
(see `encode/streaming.rs::encode_sequential_xyb` and
`encode/progressive.rs::encode_progressive_with_scans`):

| Marker | Content | Purpose |
|--------|---------|---------|
| SOI | `FF D8` | Standard JPEG start |
| APP14 (Adobe) | transform = 0 (RGB) | Tells generic decoders this JPEG uses RGB component IDs, not YCbCr |
| APP2 (ICC_PROFILE) | The 720-byte canonical XYB ICC profile | Color-managed decoders apply this to convert "RGB" → sRGB |
| DQT × 3 | Y / Cb / Cr quant tables (XYB-tuned, treated as X/Y/B) | One per component |
| SOF1 (extended sequential) or SOF2 (progressive) | R/G/B component IDs (`82 71 66`), sampling factors per `XybSubsampling` | SOF1 is forced because XYB DC categories can exceed baseline's limit of 11 |
| DHT × N | Optimized or fixed Huffman tables | XYB shares one DC + one AC table across components by default |
| SOS | Component-to-table assignments | Per `XybSubsampling` and scan strategy |

### The XYB ICC profile

`zenjpeg::foundation::consts::XYB_ICC_PROFILE` is a 720-byte v2 ICC
profile, byte-identical to the one cjpegli embeds. Detection is exact-
match in the fast path with a description-string fallback:

```rust
// zenjpeg/src/color/icc.rs
pub fn is_xyb_profile(icc_data: &[u8]) -> bool {
    use crate::foundation::consts::XYB_ICC_PROFILE;
    if icc_data == XYB_ICC_PROFILE {
        return true;
    }
    // Fallback: check for "XYB" / UTF-16BE "XYB" in profile description.
    // NOTE: the "jxl " CMM type (bytes 4-7) is NOT sufficient — cjpegli
    // writes "jxl " for ALL ICC profiles (including standard sRGB),
    // not just XYB ones.
    icc_data.windows(XYB_PROFILE_MARKER.len()).any(|w| w == XYB_PROFILE_MARKER)
        || icc_data.windows(6).any(|w| w == XYB_UTF16BE_MARKER)
}
```

The 6-byte UTF-16BE fallback is there because some renderers store the
description as UTF-16BE, not ASCII.

### XYB subsampling layout

The SOF sampling-factor bytes encode the `XybSubsampling` variant:

| Variant | R sampling | G sampling | B sampling | Notes |
|---------|-----------|-----------|-----------|-------|
| `BQuarter` (default) | `0x22` (2×2) | `0x22` (2×2) | `0x11` (1×1) | B is at quarter resolution |
| `Full` | `0x11` (1×1) | `0x11` (1×1) | `0x11` (1×1) | All components at full resolution |

The BQuarter layout matches what cjpegli emits; Full is zenjpeg-specific
(implemented in commit `25ea06e4`, fixed in `daf52508`).

The MCU layout in the entropy stream MUST match the SOF declaration —
the original Full implementation broke this and was caught only by the
quadrant-color test, not by the existing XYB roundtrip test.

## The decoder side

### XYB detection

`decode/parser/mod.rs::JpegInfo` flags the file as XYB when the embedded
ICC matches `is_xyb_profile`:

```rust
// JpegInfo::color_space derivation
let is_xyb = self.icc_profile.as_ref()
    .map(|p| is_xyb_profile(p))
    .unwrap_or(false);
let color_space = if is_xyb {
    ColorSpace::Xyb
} else if /* RGB component IDs */ { ColorSpace::Rgb }
  else if /* 1 component */ { ColorSpace::Grayscale }
  else if /* 4 component */ { ColorSpace::Cmyk }
  else { ColorSpace::YCbCr };
```

This XYB flag drives several decode-pipeline decisions:

- `decode_baseline_streaming_rgb` (the fast streaming path) refuses XYB,
  forcing the buffered-coefficient path that has access to the f32 IDCT
  output (`scan.rs:534-545`)
- `can_use_fast_i16_path` returns `false` for XYB so the i16 fast IDCT
  is skipped in favor of the f32 IDCT (`output.rs:133`)
- The output stage runs `xyb_planes_to_rgb_u8_simd` instead of the
  YCbCr→RGB conversion (`output.rs:1241`)

### The scaled-XYB → sRGB inverse transform

zenjpeg's decoder does the inverse of the encoder transform itself —
no ICC application required to get correct sRGB output:

```rust
// decode/parser/output.rs (paraphrased)
if is_xyb {
    crate::color::xyb::xyb_planes_to_rgb_u8_simd(
        &planes_f32[0],  // X plane (came in as "R")
        &planes_f32[1],  // Y plane (came in as "G")
        &planes_f32[2],  // B plane (came in as "B")
        &mut rgb,
    );
}
```

`xyb_planes_to_rgb_u8_simd` (see `color/xyb.rs:344` for the scalar
reference `scaled_xyb_to_srgb`) un-scales, runs the inverse opsin
matrix, undoes the cube root, applies the sRGB transfer curve, and
clamps to u8. This is symmetric with the encoder side's
`srgb_to_scaled_xyb_planes_*` family.

The output is sRGB-encoded u8 RGB. Calling
`Decoder::correct_color(Some(TargetColorSpace::Srgb))` is therefore
**unnecessary for XYB JPEGs** — you'd be asking moxcms to apply the
embedded XYB ICC to data that is already sRGB. zenjpeg correctly skips
that round-trip.

For non-XYB JPEGs with an embedded ICC (e.g. Display P3 photos),
`correct_color(Some(Srgb))` does its normal job: moxcms transforms the
decoded RGB through the embedded profile to the requested target.

### Comparison with other decoders

| Decoder | Behavior on XYB JPEG |
|---------|---------------------|
| **zenjpeg (default)** | Detects XYB ICC, runs inverse-XYB → sRGB internally, returns correct sRGB pixels |
| **zenjpeg + correct_color(Srgb)** | Same as default — `correct_color` is a no-op when the source is already XYB-sRGB transformed |
| **cjpegli reference (libjxl)** | Same as zenjpeg — detects XYB and runs its own inverse transform |
| **mozjpeg / libjpeg-turbo (no CMS)** | Reads scaled-XYB as raw RGB → wrong colors (looks washed/blue-shifted) |
| **mozjpeg + ICC-aware host** | Host applies the embedded XYB ICC profile → correct sRGB |
| **zune-jpeg** | Same as raw libjpeg-turbo — does NOT apply ICC profiles. Wrong colors on XYB. This was the trap behind `zenjpeg-bench-utils::decode_jpeg_to_rgb` (now `#[deprecated]`); use `decode_jpeg_with_icc` (which routes through zenjpeg's decoder) for any code path that may see XYB input. |

### Color round-trip fidelity caveat

XYB encoding is lossy at every quality. Even at Q95 with correct
layout, saturated primaries (pure red, blue, yellow) decode with
noticeable cast — for example a (220, 40, 40) red source pixel in the
4-quadrant test image decodes to roughly (226, 131, 100) at Q85
BQuarter. The dominant channel relationships are preserved (R is
clearly the largest component) but absolute saturation is reduced.

This is a property of the XYB color space, not a bug. The pixel-
correctness regression test therefore checks dominant-channel
relationships, not absolute color values.

## Test coverage

`zenjpeg/tests/bundled/xyb_roundtrip.rs` covers (all on stable, no
`--ignored`):

| Test | What it verifies |
|------|------------------|
| `xyb_420_roundtrip_all_qualities` | XYB BQuarter encode succeeds and decodes at Q15-Q95 |
| `xyb_full_roundtrip` | XYB Full encode/decode succeeds at Q15/50/85 |
| `xyb_full_baseline_pixel_correctness` | Strict regression: baseline XYB Full produces correct quadrant colors (catches the daf52508 bug) |
| `xyb_roundtrip_matrix_pixel_correctness` | Boxed sweep: every (BQuarter\|Full) × (progressive\|baseline) × (Q15\|50\|85\|95) combo passes the quadrant test |
| `xyb_encoder_embeds_xyb_icc_profile` | Every XYB encode has the canonical XYB ICC in its APP2 segments |
| `xyb_decoder_detects_xyb_color_space` | `is_xyb_profile` correctly flags every encoded XYB JPEG |
| `xyb_decode_produces_srgb_without_correct_color_call` | Default decode (no `correct_color`) produces correct sRGB — the inverse XYB transform runs internally |
| `xyb_sequential_uses_sof1` | Baseline XYB writes SOF1, never SOF0 (XYB DC categories can exceed baseline limit) |
| `xyb_allows_baseline_quant` | `force_baseline()` and `allow_16bit_quant_tables()` are accepted on XYB and don't break SOF1 emission |

## Open issues / known limitations

- **`Decoder::correct_color(Some(...))` is a no-op for XYB.** Documented
  here, but the API doesn't communicate this clearly to users — they
  may reasonably expect `correct_color(Some(DisplayP3))` to convert
  XYB → P3 via the embedded ICC. Currently it just returns sRGB regardless
  because the XYB inverse transform runs first. Fix would be to route
  XYB decode through moxcms when `correct_color` targets a non-sRGB
  space, or to document the limitation in the `correct_color` rustdoc.
- **No XYB → wide-gamut path.** XYB internally has a wider gamut than
  sRGB but our inverse transform clamps to sRGB u8. Decoding to f32
  in linear-Rec.2020 would preserve more range. Out of scope for the
  current decoder API.
- **The 720-byte canonical XYB ICC is hard-coded.** If JPEG XL evolves
  the XYB profile, we'll need to update `XYB_ICC_PROFILE` and the
  exact-match check in `is_xyb_profile`.
