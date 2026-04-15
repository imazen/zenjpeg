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

The output stage emits **raw scaled-XYB bytes**, not sRGB. The function
called for XYB output (`output.rs:1241`) is misleadingly named:

```rust
// decode/parser/output.rs (paraphrased)
if is_xyb {
    // Output raw level-shifted values, NO XYB→RGB conversion.
    // The XYB values are stored in JPEG sample positions but are NOT RGB.
    // The ICC profile transforms these directly to sRGB.
    crate::color::xyb::xyb_planes_to_rgb_u8_simd(
        &planes_f32[0],  // X plane
        &planes_f32[1],  // Y plane
        &planes_f32[2],  // B plane
        &mut rgb,
    );
}
```

Despite its name, `xyb_planes_to_rgb_u8_simd` only does `+128` level
shift and clamp to u8 — no inverse opsin matrix, no cube unwinding,
no sRGB OETF. The scalar `scaled_xyb_to_srgb` function (`color/xyb.rs:344`)
*does* the full inverse transform but **is not called from the decoder**.

This means the embedded XYB ICC profile is the **only** path to visible
sRGB. The decoder defaults to applying it when the source is XYB:

```rust
// decode/mod.rs (post-fix, both u8 and f32 output paths)
let effective_target = self.correct_color.or_else(|| {
    parser.icc_profile.as_ref().and_then(|icc| {
        if crate::color::icc::is_xyb_profile(icc) {
            Some(TargetColorSpace::Srgb)  // default-apply for XYB
        } else {
            None  // non-XYB: only apply if user opted in
        }
    })
});
if let Some(target) = effective_target { /* run moxcms ICC transform */ }
```

So for XYB:
- **Default decode** (`Decoder::new().decode(...)`): correct sRGB output
  (because we default-apply the XYB ICC when moxcms is enabled).
- **`correct_color(Some(Srgb))`**: same as default — explicit confirmation.
- **`correct_color(Some(DisplayP3))`**: applies the XYB ICC with target
  P3, giving correctly P3-converted output.

For non-XYB JPEGs with an embedded ICC (e.g. Display P3 photos),
`correct_color(Some(Srgb))` retains its normal opt-in behavior.

### Without the `moxcms` feature

If the `moxcms` feature is disabled, **XYB JPEGs decode to garbage colors**
because the inverse XYB transform isn't built into the decoder — only the
ICC profile carries the inverse transform, and there's no CMS to apply it.

Future work could embed a SIMD scaled-XYB → sRGB inverse directly in
`color/xyb.rs` so XYB works without `moxcms`. See
"Architectural follow-up: built-in XYB profile in zenpixels-convert"
below.

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

## Architectural follow-up: built-in XYB profile in zenpixels-convert

Today, XYB JPEGs require the `moxcms` feature to decode visibly because
the inverse XYB transform lives only in the embedded ICC. Two cleaner
options are open:

### Option 1: hard-code XYB recognition in zenpixels-convert

`zenpixels-convert/src/fast_gamut.rs` already has a `stamp_trc_kernels!`
macro that emits SIMD `linearize → 3x3 matrix → encode` pipelines per
(src TRC, dst TRC) pair, dispatched through magetypes / `#[arcane]`.

XYB *almost* fits this model:

| Stage | Maps to |
|-------|---------|
| 1. unscale bytes | per-channel "linearize" (with cross-channel B+=Y bias) |
| 2. cube (inverse cube root) | per-channel |
| 3. inverse opsin matrix | 3x3 mat |
| 4. linear → sRGB OETF | per-channel "encode" |

The cross-channel `b += y` bias in stage 1 doesn't fit the existing
1D-TRC slot, but it can be folded into a 5-stage variant of the
macro: `unscale → bias → cube → matrix → OETF`. Or the bias can be
absorbed into a synthetic 3x3 `[[1,0,0],[0,1,0],[0,1,1]]` between
stages 1 and 2 — at the cost of being unable to use the existing
`fused_8px_*` kernel layout directly.

Cleanest implementation: add a dedicated `xyb_to_rgb_x8_v3`
`#[arcane]` function in `color/xyb.rs` that hand-codes the full
fused pipeline using `magetypes::simd::f32x8`. Lives in zenjpeg
since the encoder side already has `srgb_to_scaled_xyb_planes_simd`
in the same file. Output stage in `decode/parser/output.rs:1241`
calls this instead of the misleadingly-named level-shift function,
and the moxcms ICC-apply path becomes the fallback for users
wanting non-sRGB targets.

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

- **`xyb_planes_to_rgb_u8_simd` is misleadingly named** — it's just a
  level-shift, not an XYB→RGB transform. Should be renamed to
  `xyb_planes_level_shift_to_u8` in a follow-up to make the call site
  in `decode/parser/output.rs` self-explanatory.
- **No `moxcms`-free XYB path.** Without `moxcms`, XYB JPEGs decode to
  garbage. Fix: see Option 1 above (built-in SIMD inverse in
  `color/xyb.rs`).
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
`examples/xyb_color_loss.rs` on this branch:

| Encode | Distinct outputs | RMSE | MAE R/G/B | max ΔE | within ΔE≤1 | within ΔE≤4 |
|--------|------------------|------|-----------|--------|-------------|-------------|
| YCbCr 4:4:4 Q100 progressive | 4096 / 4096 (100%) | 0.99 | 0.36/0.19/0.44 | 1 | 100.0% | 100.0% |
| XYB Full Q100 progressive (default decode after fix) | 4086 / 4096 (99.8%) | 4.50 | 1.98/0.52/1.90 | 21 | 34.6% | 80.4% |

Worst XYB sRGB samples concentrate in the dark-red corner (R≤8). The
encoder's RGB→XYB pipeline (`srgb_to_scaled_xyb`) loses precision in
the toe of the cube root + scale, and the inverse can't recover it.
Bias is uniformly negative-R / negative-B (decoded values come out
slightly darker than source for R and B) at -1.22 / -1.16 mean.
