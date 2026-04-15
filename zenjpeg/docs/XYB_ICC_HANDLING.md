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
SIMD-optimized inverse-XYB → sRGB transform
([`xyb_planes_to_srgb_u8_simd`] / [`xyb_planes_to_srgb_f32_simd`] in
`color/xyb.rs`). This SIMD kernel handles the full inverse pipeline
(unscale → cube → inverse opsin → sRGB OETF) with no external CMS
dependency, so XYB JPEGs decode to correct sRGB pixels **without
requiring the `moxcms` feature**.

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
- The output stage runs `xyb_planes_to_srgb_u8_simd` (the full inverse
  XYB transform) instead of the YCbCr→RGB conversion (`output.rs`)

### The scaled-XYB → sRGB inverse transform

The output stage emits visibly-correct sRGB directly — no CMS required.
The decoder recognizes XYB via `is_xyb_profile(icc)` and routes through
the built-in SIMD inverse in `color/xyb.rs`:

```rust
// decode/parser/output.rs
if is_xyb {
    // Run the full inverse XYB → sRGB transform in-decoder.
    crate::color::xyb::xyb_planes_to_srgb_u8_simd(
        &planes_f32[0],  // scaled-X plane (pre-level-shift)
        &planes_f32[1],  // scaled-Y plane
        &planes_f32[2],  // scaled-B plane
        &mut rgb,
    );
}
```

The kernel pipeline, per 8 lanes:

1. `+128` then `/255` — undo level shift, recover scaled XYB
2. Unscale via `SCALED_XYB_OFFSET` / `SCALED_XYB_SCALE` (with `b += y`)
3. Inverse cube (`v*v*v` is sign-preserving) and subtract bias
4. Multiply by `XYB_OPSIN_INVERSE_MATRIX` (3×3 FMA)
5. Clamp to `[0, 1]` and apply sRGB OETF (scalar per lane;
   `linear_to_srgb` is a rational polynomial, no `powf`)

Dispatched via `#[magetypes(v3, neon, wasm128, scalar)]` — AVX2+FMA on
x86_64, NEON on aarch64, SIMD128 on WASM, scalar elsewhere.

Because the output is sRGB, `correct_color(Some(target))` on an XYB
source is skipped (applying the embedded XYB ICC here would double-
transform — the ICC describes scaled-XYB → sRGB, but pixels are
already sRGB). For non-sRGB targets on XYB sources, the correct
pipeline is sRGB → target as a separate step (future work — see TODO).

So for XYB:
- **Default decode** (`Decoder::new().decode(...)`): correct sRGB.
- **`correct_color(Some(Srgb))`**: no-op; result is byte-identical to
  default decode (verified by `xyb_decode_byte_equal_with_and_without_correct_color`).
- **`correct_color(Some(DisplayP3))`** / **`Rec2020`**: currently
  produces sRGB (not P3/Rec.2020). TODO: wire an sRGB → target step
  for this case.

For non-XYB JPEGs with an embedded ICC (e.g. Display P3 photos),
`correct_color(Some(Srgb))` retains its normal opt-in behavior.

### Without the `moxcms` feature

XYB JPEGs decode to correct sRGB with or without `moxcms` — the in-decoder
SIMD kernel carries the entire inverse transform. Verified on every
(BQuarter|Full) × (progressive|baseline) × (Q50|85|95) combination by
the `xyb_decode_correct_without_moxcms` test.

For decoding to non-sRGB target color spaces (Display P3, Rec.2020) on
XYB sources, the sRGB → target step is not yet implemented. Enable
`moxcms` for ICC-based workflows on non-XYB sources or use the
pre-SIMD-kernel behavior (revert c5f2... on a dev branch).

### Comparison with other decoders

| Decoder | Behavior on XYB JPEG |
|---------|---------------------|
| **zenjpeg (default)** | Detects XYB ICC, runs inverse-XYB → sRGB internally via SIMD, returns correct sRGB pixels. Works in all feature configurations, including `--no-default-features`. |
| **zenjpeg + correct_color(Srgb)** | Byte-identical to default — `correct_color(Srgb)` is a no-op when the source is XYB (pixels are already sRGB after the SIMD kernel). |
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
| `xyb_decode_correct_without_moxcms` | 4-quadrant correctness across every (subsampling, scan, quality) combo through the default decode path. With `--no-default-features --features "trellis decoder"` this verifies the SIMD kernel carries the full load. |
| `xyb_decode_byte_equal_with_and_without_correct_color` | `correct_color(Some(Srgb))` is byte-identical to default decode on an XYB source — guards against ICC double-transform regressions. |
| `xyb_sequential_uses_sof1` | Baseline XYB writes SOF1, never SOF0 (XYB DC categories can exceed baseline limit) |
| `xyb_allows_baseline_quant` | `force_baseline()` and `allow_16bit_quant_tables()` are accepted on XYB and don't break SOF1 emission |
| `simd_xyb_to_srgb_u8_matches_scalar` (lib test) | SIMD kernel matches scalar reference within 1 ULP across a 64-sample sweep. Runs on `cargo test --lib` under any feature config. |
| `simd_xyb_to_srgb_u8_roundtrip_primaries` (lib test) | Dominant-channel round-trip for R/G/B/yellow primaries. |

## Architectural follow-up: built-in XYB profile in zenpixels-convert

### Option 1: XYB-specific SIMD kernel in zenjpeg (SHIPPED)

Implemented as `xyb_planes_to_srgb_u8_simd` / `xyb_planes_to_srgb_f32_simd`
in `color/xyb.rs`. Pipeline: unscale → cube → inverse opsin matrix →
sRGB OETF. Dispatched via `#[magetypes(v3, neon, wasm128, scalar)]` for
x86_64 AVX2+FMA, aarch64 NEON, wasm32 SIMD128, and a scalar fallback.
Lives alongside the encode-side `srgb_to_scaled_xyb_planes_simd` family
(same file, symmetric API shape).

Coverage:
- Every subsampling × scan × quality combination round-trips correctly
  in both `--features moxcms` and `--no-default-features --features
  "trellis decoder"` builds (see `tests/bundled/xyb_roundtrip.rs`).
- Library-level kernel vs scalar parity tests in `color::xyb::tests`
  that run on `cargo test --lib` regardless of feature flags.

Remaining:
- `correct_color(Some(DisplayP3))` / `Rec2020` on XYB sources produces
  sRGB instead of the requested target. The post-kernel step needs
  either a moxcms sRGB-source transform or a native sRGB → target kernel
  in zenpixels-convert. Tracked alongside option 2 below.

### Option 2: register the XYB ICC as a built-in profile in zenpixels-convert

Since our XYB profile is a fixed 720-byte blob (`XYB_ICC_PROFILE`),
zenpixels-convert could ship the bytes + a hand-coded SIMD inverse:

```rust
// In zenpixels-convert
pub fn maybe_builtin_profile(icc: &[u8]) -> Option<BuiltinProfile> {
    if icc == zenpixels_convert::profiles::XYB_ICC_BYTES {
        return Some(BuiltinProfile::XybScaled);
    }
    None
}

pub fn convert_via_builtin(profile: BuiltinProfile, target: Cicp, ...) {
    match profile {
        BuiltinProfile::XybScaled => incant!(xyb_to_target_v3(target, ...)),
    }
}
```

A CMS-aware caller (e.g. zenjpeg's decoder) checks `maybe_builtin_profile`
first, falls back to moxcms for unknown ICCs. Saves the cost of moxcms
parsing/transform construction for the most common XYB case, AND
removes the moxcms dependency for XYB-only consumers.

Recommendation: **start with Option 1** (XYB-specific kernel in
zenjpeg's `color/xyb.rs`) since it's localized and unblocks no-moxcms
XYB decoding. Migrate to Option 2 if/when zenpixels-convert grows a
"known profile fast path" registry — XYB would be the first entry.

## Open issues / known limitations

- **`correct_color(Some(non_srgb))` on XYB sources** currently produces
  sRGB instead of the requested target color space. The SIMD kernel
  already converts XYB to sRGB, so the missing piece is an sRGB → target
  step. For now the XYB ICC apply is skipped entirely to avoid double-
  transform; this behaviour is verified by
  `xyb_decode_byte_equal_with_and_without_correct_color`.
- **XYB color loss at saturated primaries**: empirically (see
  `examples/xyb_color_loss.rs`) the worst-case round-trip ΔE is ~21
  even at Q100 XYB Full. Compare to YCbCr 4:4:4 Q100 which achieves
  max ΔE=1. Pattern: dark red (R≤8) gets pushed up by ~+20; some
  blue/cyan triplets lose B precision. This is fundamental to XYB's
  perceptual encoding, not a bug.
- **The 720-byte canonical XYB ICC is hard-coded.** If JPEG XL evolves
  the XYB profile, we'll need to update `XYB_ICC_PROFILE` and the
  exact-match check in `is_xyb_profile`.

## Quantified color loss (xyb_color_loss example output)

Brute-force sweep over a 16×16×16 sRGB grid, encoded at the highest
zenjpeg setting and decoded back. Numbers from
`examples/xyb_color_loss.rs` after Option 1 shipped:

| Encode | RMSE | MAE R/G/B | max ΔE | within ΔE≤1 | within ΔE≤4 |
|--------|------|-----------|--------|-------------|-------------|
| YCbCr 4:4:4 Q100 progressive | 0.99 | 0.36/0.19/0.44 | 1 | 100.0% | 100.0% |
| XYB Full Q100 progressive (SIMD kernel, default decode) | 0.27 | 0.03/0.00/0.03 | 2 | 99.7% | 100.0% |
| XYB Full Q100 + `correct_color(Srgb)` | 0.27 | 0.03/0.00/0.03 | 2 | 99.7% | 100.0% |

Notes:

- The SIMD inverse kernel reconstructs sRGB substantially more
  accurately than the prior moxcms-via-ICC path, which routed through
  the lossy 720-byte v2 XYB ICC profile (max ΔE was previously 21 on
  the same sweep).
- `correct_color(Srgb)` is byte-identical to default decode, as
  expected — the ICC apply is skipped for XYB sources.
- Max ΔE of 2 at Q100 is within IDCT rounding tolerance; the encoder
  itself is XYB-lossy at extreme dark primaries even at Q100, but the
  decoder no longer adds its own noticeable ΔE on top.
