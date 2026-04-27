//! Source-direct HDR / bit-depth tier.
//!
//! All other tiers read RGB8 (via `RowStream::Native` or row-by-row
//! conversion through `zenpixels-convert::RowConverter`). That works
//! for SDR features whose threshold contract is calibrated on display-
//! space bytes, but it **destroys** the HDR signal on PQ / HLG / linear-
//! light f32 sources because `RowConverter` does not tonemap — it
//! clips into the [0, 1] sRGB-display range. The bytes that reach the
//! standard tiers from a 4000-nit PQ HDR image are bit-for-bit
//! identical to the bytes from a 100-nit-clipped SDR image.
//!
//! This tier reads the source samples directly via `PixelSlice::row`
//! (same pattern as the alpha pass), decodes them through the
//! descriptor's `TransferFunction` to linear nits, and measures HDR-
//! aware statistics:
//!
//! - **Peak luminance** (max nits across sampled pixels)
//! - **P99 luminance** (robust against single hot pixels — single-bin
//!   histogram resolves to ~3% of full range)
//! - **HDR headroom in stops** (`log2(peak / sdr_reference)`)
//! - **HDR pixel fraction** (sampled pixels above the SDR threshold)
//! - **Wide-gamut peak** (per-channel max in linear light — feeds
//!   future gamut-clipping decisions)
//! - **Effective bit depth** (sample-distribution probe: how many
//!   bottom bits carry information vs being u8-promotion zeros)
//!
//! ## Reference white convention
//!
//! "Nits" here is a **convention**, not a measurement — we don't have
//! pixel-level mastering metadata. The convention tracks the transfer
//! function's de-facto signal-to-nits mapping:
//!
//! | Transfer | Linear 1.0 maps to | Rationale |
//! |---|---|---|
//! | `Srgb` / `Bt709` / `Gamma22` | 80 nits | sRGB display reference (IEC 61966-2-1). |
//! | `Linear` | 80 nits | Treat scene-referred f32 as display-referred without metadata. |
//! | `Pq` | 10 000 nits | SMPTE ST 2084 absolute reference. |
//! | `Hlg` | 1 000 nits | Nominal peak for typical HLG broadcasts. |
//! | `Unknown` | 80 nits | Conservative SDR fallback. |
//!
//! These numbers are stable across the 0.1.x line; if a downstream
//! consumer needs a different mapping, file an issue — at that point
//! we'd thread metadata through the descriptor rather than changing
//! the convention silently.
//!
//! ## Why a separate tier
//!
//! Other tiers iterate over RGB8 rows from `RowStream`. This one
//! iterates over the **source bytes**, decodes per channel-type
//! (u8 / u16 / f32), and applies the transfer EOTF in f32. There's
//! no shared inner loop with Tier 1/2/3 — the data layout, sample
//! type, and math are all different. Sharing a pass would either
//! require RGB8 conversion (defeats the purpose) or specializing the
//! existing tiers across 4 channel types × 5 transfers (massive
//! monomorphization explosion). One small dedicated tier is the
//! clean answer.

use linear_srgb::tf;
use zenpixels::{ChannelType, ColorPrimaries, PixelSlice, TransferFunction};

/// Reference peak luminance per transfer function. See module docs for
/// the convention.
const PEAK_SRGB_NITS: f32 = 80.0;
const PEAK_PQ_NITS: f32 = 10_000.0;
const PEAK_HLG_NITS: f32 = 1_000.0;

/// Threshold above which a sample counts as "HDR" — a rendered
/// approximation of the SDR boundary on a typical sRGB display.
const SDR_THRESHOLD_NITS: f32 = 100.0;

/// Output of the depth tier. Default (all-zero) is what gets written
/// when the tier is gated off, so every field's "absent" semantics
/// must read as "no signal" (zero peak / zero headroom / zero
/// fraction).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DepthStats {
    /// Peak luminance in nits over sampled pixels.
    pub peak_nits: f32,
    /// 99th-percentile luminance in nits.
    pub p99_nits: f32,
    /// HDR headroom in stops: `log2(peak_nits / 80)`. SDR ⇒ ~0.
    pub headroom_stops: f32,
    /// Fraction of sampled pixels above [`SDR_THRESHOLD_NITS`].
    pub hdr_pixel_fraction: f32,
    /// Largest single-channel linear value across sampled pixels.
    /// Useful as a "this image will clip on sRGB primaries" signal —
    /// values >> 1.0 indicate wide-gamut content.
    pub wide_gamut_peak: f32,
    /// Fraction of sampled pixels with at least one channel > 1.0
    /// in linear light (i.e., outside the source-primaries' gamut
    /// that maps to sRGB-display [0, 255]).
    pub wide_gamut_fraction: f32,
    /// Effective bit depth: smallest power-of-2 quantization grid
    /// the sampled values populate. Common values: 8, 10, 12, 14, 16.
    /// For u8 sources always 8. For u16 sources, probes the
    /// low-byte distribution to detect u8-promoted content.
    pub effective_bit_depth: u32,
    /// `true` iff [`peak_nits`] >> the SDR threshold AND the source
    /// transfer function is genuinely HDR-capable (PQ / HLG /
    /// Linear-with-out-of-range values). Catches the hard case the
    /// standard tiers miss: a PQ-encoded image whose tonemapped
    /// rendition looks like SDR but whose source carries far more
    /// dynamic range.
    pub hdr_present: bool,
    /// Fraction of sampled pixels whose linear-RGB, when projected
    /// from the source primaries into BT.709 / sRGB primaries, has
    /// every channel within `[-ε, 1 + ε]`. `1.0` ⇒ image is safely
    /// downcastable to sRGB primaries (codecs save bits by encoding
    /// nclx-Bt709 / sRGB ICC instead of the wider declared gamut).
    /// For sRGB-declared sources this is trivially `1.0`; the signal
    /// is load-bearing for P3 / Rec.2020 / AdobeRGB sources whose
    /// pixels happen to all live in the sRGB sub-gamut.
    pub gamut_coverage_srgb: f32,
    /// Same as [`gamut_coverage_srgb`] for the Display P3 sub-gamut.
    /// Useful when the source is declared Rec.2020 — `1.0` here
    /// means the source is downcastable to P3 (a smaller container
    /// than Rec.2020 but wider than sRGB).
    pub gamut_coverage_p3: f32,
}

/// Histogram-bin count for percentile estimation. 256 bins on a
/// linear nits scale gives ~3% relative resolution at the SDR
/// threshold and ~40-nit absolute resolution at 10 000 nits — fine
/// for codec dispatch decisions.
const HIST_BINS: usize = 256;

/// Logarithmic histogram bin: maps `nits ∈ [0, ~10 000]` to
/// `[0, 256)` via `log2(1 + nits)` so SDR detail isn't squashed by
/// the HDR tail.
#[inline]
fn nits_to_bin(nits: f32) -> usize {
    if !nits.is_finite() || nits <= 0.0 {
        return 0;
    }
    // log2(1 + 10000) ≈ 13.29; scale to [0, HIST_BINS).
    let v = (1.0_f32 + nits).log2() / 14.0;
    let i = (v * HIST_BINS as f32) as usize;
    i.min(HIST_BINS - 1)
}

#[inline]
fn bin_to_nits(bin: usize) -> f32 {
    // Inverse of nits_to_bin: nits = exp2((bin / HIST_BINS) * 14) - 1.
    (((bin as f32 + 0.5) / HIST_BINS as f32) * 14.0).exp2() - 1.0
}

/// Apply a transfer function's EOTF (signal → linear) to a sample
/// already normalized to `[0, 1]` (or wider, for HDR-out-of-range
/// f32). Output is normalized linear (1.0 = the transfer's reference
/// peak — see `peak_nits_for`).
#[inline]
fn eotf(tf_kind: TransferFunction, signal: f32) -> f32 {
    match tf_kind {
        TransferFunction::Linear => signal,
        TransferFunction::Srgb | TransferFunction::Unknown => tf::srgb_to_linear(signal),
        TransferFunction::Bt709 => tf::bt709_to_linear(signal),
        TransferFunction::Gamma22 => signal.max(0.0).powf(2.2),
        TransferFunction::Pq => tf::pq_to_linear(signal),
        TransferFunction::Hlg => tf::hlg_to_linear(signal),
        _ => signal, // Defensive: non_exhaustive enum ⇒ any future variant.
    }
}

/// Reference peak in nits for a transfer function — see module docs.
#[inline]
fn peak_nits_for(tf_kind: TransferFunction) -> f32 {
    match tf_kind {
        TransferFunction::Pq => PEAK_PQ_NITS,
        TransferFunction::Hlg => PEAK_HLG_NITS,
        _ => PEAK_SRGB_NITS,
    }
}

/// 3×3 matrices that take linear RGB in the source primaries and
/// project it into linear sRGB-primaries RGB. Pre-computed from the
/// ITU-R / SMPTE primaries chromaticities + D65 whitepoint via
/// `M = M_xyz_to_srgb · M_src_to_xyz`. Source: standard derivation;
/// the same numbers appear in libjxl, ffmpeg, and `colour-science`.
///
/// Values stored row-major: `[r_out, g_out, b_out]` from
/// `(r_lin, g_lin, b_lin)` via `out = M · in`.
const M_DISPLAYP3_TO_SRGB: [[f32; 3]; 3] = [
    [1.224_940_2, -0.224_940_4, 0.000_000_0],
    [-0.042_056_9, 1.042_057_1, 0.000_000_0],
    [-0.019_637_6, -0.078_636_1, 1.098_273_7],
];
const M_BT2020_TO_SRGB: [[f32; 3]; 3] = [
    [1.660_491_0, -0.587_641_1, -0.072_849_9],
    [-0.124_550_5, 1.132_899_9, -0.008_349_4],
    [-0.018_150_8, -0.100_578_9, 1.118_729_7],
];
const M_ADOBERGB_TO_SRGB: [[f32; 3]; 3] = [
    [1.398_287_7, -0.398_287_8, 0.000_000_0],
    [0.000_000_0, 1.000_000_0, 0.000_000_0],
    [0.000_000_0, -0.042_969_2, 1.042_969_3],
];
/// Same shape, projecting wider primaries into Display P3 (the
/// "smallest wide gamut" — useful when the source is BT.2020 and we
/// want to know if a P3-down container would suffice).
const M_BT2020_TO_DISPLAYP3: [[f32; 3]; 3] = [
    [1.343_578_8, -0.282_855_8, -0.060_722_6],
    [-0.077_876_4, 1.083_393_2, -0.005_516_5],
    [0.000_307_5, -0.027_209_2, 1.026_901_8],
];

/// Identity 3×3.
const M_IDENTITY: [[f32; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// Multiply 3×3 matrix `m` by 3-vector `(r, g, b)`.
#[inline]
fn mat3_mul(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )
}

/// Pick the matrix that takes source-primaries linear RGB into sRGB
/// (BT.709) primaries linear RGB. Returns `None` if the source is
/// already in sRGB primaries (caller's hot path skips the multiply).
#[inline]
fn primaries_to_srgb_matrix(src: ColorPrimaries) -> Option<&'static [[f32; 3]; 3]> {
    match src {
        ColorPrimaries::Bt709 => None,
        ColorPrimaries::DisplayP3 => Some(&M_DISPLAYP3_TO_SRGB),
        ColorPrimaries::Bt2020 => Some(&M_BT2020_TO_SRGB),
        ColorPrimaries::AdobeRgb => Some(&M_ADOBERGB_TO_SRGB),
        _ => None, // Unknown / future ⇒ no projection (assume already in target gamut).
    }
}

/// Pick the matrix that takes source-primaries linear RGB into
/// Display P3 primaries linear RGB. P3 is the "next-smallest" gamut
/// after sRGB; useful for "is this Rec.2020 source actually just P3
/// content?" downcast detection.
#[inline]
fn primaries_to_displayp3_matrix(src: ColorPrimaries) -> &'static [[f32; 3]; 3] {
    match src {
        ColorPrimaries::DisplayP3 => &M_IDENTITY,
        ColorPrimaries::Bt2020 => &M_BT2020_TO_DISPLAYP3,
        // sRGB / Bt709 / AdobeRGB are subsets of P3 in practice
        // (some AdobeRGB greens push slightly outside P3, but the
        // common case is fully covered). For codec dispatch we
        // treat anything sRGB-or-narrower as P3-coverable.
        ColorPrimaries::Bt709 | ColorPrimaries::AdobeRgb => &M_IDENTITY,
        _ => &M_IDENTITY,
    }
}

/// `true` iff `tf` can carry above-SDR signal levels. Used to gate
/// `hdr_present` so a sRGB image with a single hot pixel from
/// rounding doesn't trip the flag.
#[inline]
fn is_hdr_capable_tf(tf_kind: TransferFunction) -> bool {
    matches!(
        tf_kind,
        TransferFunction::Pq | TransferFunction::Hlg | TransferFunction::Linear
    )
}

/// Decode one source sample to `[0, 1]`-normalized signal (may be
/// wider than 1.0 for f32 HDR linear inputs). `bytes` points at the
/// start of the sample; caller guarantees `bytes.len() >= ch.byte_size()`.
#[inline]
fn read_sample(ch: ChannelType, bytes: &[u8]) -> f32 {
    match ch {
        ChannelType::U8 => bytes[0] as f32 / 255.0,
        ChannelType::U16 => u16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 65535.0,
        ChannelType::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        ChannelType::F16 => {
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
            f16_to_f32(raw)
        }
        // ChannelType is non_exhaustive — defensive fallback.
        _ => 0.0,
    }
}

/// Bit-for-bit IEEE 754 binary16 → binary32 conversion. Inlined to
/// keep the hot loop tight; called only on the (rare) F16 path.
#[inline]
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 0x1) as u32;
    let exp = ((half >> 10) & 0x1f) as u32;
    let mant = (half & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal — renormalize.
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf / NaN.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp + 127 - 15) & 0xff) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Walk source samples directly, accumulating depth statistics. Stride-
/// samples rows so total walked pixels ≈ `pixel_budget`.
pub(crate) fn scan_depth(slice: &PixelSlice<'_>, pixel_budget: usize) -> DepthStats {
    let desc = slice.descriptor();
    let width = slice.width() as usize;
    let height = slice.rows() as usize;
    if width == 0 || height == 0 {
        return DepthStats::default();
    }

    let layout = desc.layout();
    let ch = desc.channel_type();
    let color_channels = desc.color_model().color_channels() as usize;
    if color_channels == 0 {
        return DepthStats::default();
    }
    let total_channels = layout.channels();
    let bpp = total_channels * ch.byte_size();
    let ch_bytes = ch.byte_size();

    let tf_kind = desc.transfer();
    let peak_nits_unit = peak_nits_for(tf_kind);
    let hdr_capable = is_hdr_capable_tf(tf_kind);

    // SDR-fast-path: u8 source with an SDR-only transfer function
    // can never produce HDR / wide-gamut signal, and the bit depth is
    // trivially 8. Walking through the EOTF for every sample just to
    // confirm that is wasteful — return the canonical SDR profile
    // directly.
    if matches!(ch, ChannelType::U8) && !hdr_capable {
        // Conservative SDR-display peak — encode-orchestrators see a
        // sensible non-zero peak for "what's the brightness" queries
        // without having to walk pixels.
        //
        // Gamut-coverage fast path: u8 + sRGB-class transfer cannot
        // carry values that exceed the source primaries (linear ∈
        // [0, 1] always). Whether the image fits in a NARROWER
        // gamut still depends on its actual pixel content — but the
        // detection requires the matrix walk we're avoiding here.
        // For sRGB-declared sources gamut_coverage_srgb is trivially
        // 1.0; for wider declared primaries the SDR fast path skips
        // the detection. Codecs needing the downcast signal must
        // request a query that disables the fast path
        // (e.g., include any HDR-tier feature that forces the walk).
        let trivial_srgb_cover =
            if matches!(desc.primaries, ColorPrimaries::Bt709) { 1.0 } else { 0.0 };
        let trivial_p3_cover =
            if matches!(desc.primaries, ColorPrimaries::Bt709 | ColorPrimaries::DisplayP3) {
                1.0
            } else {
                0.0
            };
        return DepthStats {
            peak_nits: PEAK_SRGB_NITS,
            p99_nits: PEAK_SRGB_NITS,
            headroom_stops: 0.0,
            hdr_pixel_fraction: 0.0,
            wide_gamut_peak: 1.0,
            wide_gamut_fraction: 0.0,
            effective_bit_depth: 8,
            hdr_present: false,
            gamut_coverage_srgb: trivial_srgb_cover,
            gamut_coverage_p3: trivial_p3_cover,
        };
    }

    // Stride-sample rows, same shape as alpha pass. Within a row we
    // also stride-sample pixels for very wide images so the budget
    // covers a representative slice of the height too.
    let pixels_per_row = width.max(1);
    let target_rows = (pixel_budget / pixels_per_row).max(1).min(height);
    let row_step = (height / target_rows).max(1);

    let mut hist = [0u32; HIST_BINS];
    let mut total: u32 = 0;
    let mut hdr_pixels: u32 = 0;
    let mut wide_gamut_pixels: u32 = 0;
    let mut peak_nits: f32 = 0.0;
    let mut wide_gamut_peak: f32 = 0.0;

    // Gamut downcast counters. For each pixel, project the linear
    // RGB from source primaries to {sRGB, P3} and count pixels whose
    // every channel stays within [GAMUT_LO, GAMUT_HI] = [-0.005, 1.005].
    // Tolerance absorbs small numerical noise from the matrix walk.
    let m_to_srgb = primaries_to_srgb_matrix(desc.primaries);
    let m_to_p3 = primaries_to_displayp3_matrix(desc.primaries);
    let mut srgb_in: u32 = 0;
    let mut p3_in: u32 = 0;
    const GAMUT_LO: f32 = -0.005;
    const GAMUT_HI: f32 = 1.005;
    #[inline]
    fn in_gamut(r: f32, g: f32, b: f32) -> bool {
        let range = GAMUT_LO..=GAMUT_HI;
        range.contains(&r) && range.contains(&g) && range.contains(&b)
    }

    // For BT.2020-ish luma weights — close enough for cross-primary
    // luminance approximation. The exact primaries vary, but the
    // relative weight on green dominates so the choice barely shifts
    // the peak / P99.
    const WL_R: f32 = 0.2627;
    const WL_G: f32 = 0.6780;
    const WL_B: f32 = 0.0593;

    // Effective-bit-depth probe (integer sources only): track the OR
    // of the low byte across samples. If it stays at 0 the source is
    // u8-promoted; otherwise we have at least 9-bit content. Counts
    // distinct low-byte values via a 256-bin presence flag for a
    // sharper estimate up to ~10/12-bit.
    let mut low_byte_seen = [false; 256];
    let mut low_byte_distinct: u32 = 0;
    let probe_bits = matches!(ch, ChannelType::U16);

    let mut y = 0usize;
    while y < height {
        let row = slice.row(y as u32);
        // Pixel stride within a row: 1 for narrow images, larger for
        // very wide images so we don't blow past the budget on a
        // single row.
        let row_pixel_stride = ((width as u32) / 1024).max(1) as usize;
        let mut x = 0usize;
        while x < width {
            let off = x * bpp;
            if off + color_channels * ch_bytes > row.len() {
                break;
            }
            // Read each colour channel's sample.
            let mut linear_max: f32 = 0.0;
            let mut linears = [0.0_f32; 4]; // up to 4 colour channels
            for c in 0..color_channels.min(4) {
                let s = read_sample(ch, &row[off + c * ch_bytes..]);
                let l = eotf(tf_kind, s);
                linears[c] = l;
                if l > linear_max {
                    linear_max = l;
                }
            }
            let linear_luma = if color_channels >= 3 {
                WL_R * linears[0] + WL_G * linears[1] + WL_B * linears[2]
            } else {
                // Grayscale — luma = the single channel, already linear.
                linears[0]
            };

            // Probe low-byte for u16 sources.
            if probe_bits && color_channels >= 1 {
                let low = row[off]; // little-endian: byte 0 is the low byte
                if !low_byte_seen[low as usize] {
                    low_byte_seen[low as usize] = true;
                    low_byte_distinct += 1;
                }
            }

            let nits = linear_luma * peak_nits_unit;
            if nits > peak_nits {
                peak_nits = nits;
            }
            if linear_max > wide_gamut_peak {
                wide_gamut_peak = linear_max;
            }
            if linear_max > 1.0 {
                wide_gamut_pixels += 1;
            }
            if nits > SDR_THRESHOLD_NITS {
                hdr_pixels += 1;
            }
            // Gamut-coverage projections (only meaningful for ≥ 3
            // colour channels — grayscale by construction has the
            // pixel sitting on the achromatic axis, in every gamut).
            if color_channels >= 3 {
                let (sr_r, sr_g, sr_b) = match m_to_srgb {
                    Some(m) => mat3_mul(m, linears[0], linears[1], linears[2]),
                    None => (linears[0], linears[1], linears[2]),
                };
                if in_gamut(sr_r, sr_g, sr_b) {
                    srgb_in += 1;
                }
                let (p3_r, p3_g, p3_b) =
                    mat3_mul(m_to_p3, linears[0], linears[1], linears[2]);
                if in_gamut(p3_r, p3_g, p3_b) {
                    p3_in += 1;
                }
            } else {
                // Achromatic pixel — by definition it sits in every
                // gamut. Counting both buckets keeps the fraction
                // meaningful for grayscale-in-RGB sources.
                srgb_in += 1;
                p3_in += 1;
            }
            hist[nits_to_bin(nits)] += 1;
            total += 1;

            x += row_pixel_stride;
        }
        y += row_step;
    }

    if total == 0 {
        return DepthStats::default();
    }
    let total_f = total as f32;

    // P99: walk the histogram from the top; pick the bin where the
    // cumulative count first reaches 1% of total.
    let target = (total / 100).max(1);
    let mut cum: u32 = 0;
    let mut p99_bin = 0usize;
    for b in (0..HIST_BINS).rev() {
        cum += hist[b];
        if cum >= target {
            p99_bin = b;
            break;
        }
    }
    let p99_nits = bin_to_nits(p99_bin).min(peak_nits);

    let headroom_stops = if peak_nits > 0.0 {
        (peak_nits / PEAK_SRGB_NITS).max(1.0).log2()
    } else {
        0.0
    };

    // Effective bit depth.
    let effective_bit_depth = match ch {
        ChannelType::U8 => 8,
        ChannelType::F32 | ChannelType::F16 => {
            // f32 / f16: report the storage depth. A finer probe
            // (sample-quantization grid analysis) is deferred — the
            // storage depth is the more useful per-codec signal
            // (drives jxl modular bit width and avif encode_depth).
            if matches!(ch, ChannelType::F32) { 32 } else { 16 }
        }
        ChannelType::U16 => effective_depth_from_low_byte(low_byte_distinct, total),
        // Defensive: ChannelType is non_exhaustive.
        _ => 0,
    };

    // hdr_present: peak well above SDR AND a non-trivial fraction of
    // pixels are above the threshold AND the transfer function can
    // carry HDR. Counting pixels avoids tripping on a single rounding
    // outlier.
    let hdr_pixel_fraction = hdr_pixels as f32 / total_f;
    let wide_gamut_fraction = wide_gamut_pixels as f32 / total_f;
    let hdr_present = hdr_capable
        && peak_nits > 1.5 * SDR_THRESHOLD_NITS
        && hdr_pixel_fraction > 0.001;
    let gamut_coverage_srgb = srgb_in as f32 / total_f;
    let gamut_coverage_p3 = p3_in as f32 / total_f;

    DepthStats {
        peak_nits,
        p99_nits,
        headroom_stops,
        hdr_pixel_fraction,
        wide_gamut_peak,
        wide_gamut_fraction,
        effective_bit_depth,
        hdr_present,
        gamut_coverage_srgb,
        gamut_coverage_p3,
    }
}

/// Map low-byte-distinct count to effective bit depth. Reasoning: a
/// u8-promoted u16 has every low byte equal to its high byte, so the
/// distinct low-byte count caps at 256 — but it only reaches 256 if
/// the high byte is uniformly distributed. Genuine 10/12/14/16-bit
/// content, by contrast, has the low byte sweeping uniformly through
/// 256 values regardless of high-byte distribution, so a small image
/// region exhibits ~256 distinct low bytes quickly.
///
/// In practice the discriminator that catches u8-promoted-to-u16 is
/// simpler: in u8-promoted u16 every low byte is *also* a high byte
/// of the same sample, so `low == high` for every sample. Detecting
/// that is a separate test we don't do here — the distinct count is
/// the conservative signal: <16 ⇒ 8-bit-class content, 16..64 ⇒
/// 10-bit, 64..192 ⇒ 12-bit, ≥192 ⇒ 14-bit-or-finer.
#[inline]
fn effective_depth_from_low_byte(distinct: u32, total: u32) -> u32 {
    // Tiny samples can't tell us anything — fall back to storage depth.
    if total < 64 {
        return 16;
    }
    match distinct {
        0..=15 => 8,
        16..=63 => 10,
        64..=191 => 12,
        _ => 14, // ≥ 192 distinct low-byte values ⇒ effectively 14-bit or finer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::PixelDescriptor;

    #[test]
    fn solid_srgb_u8_takes_fast_path_canonical_sdr_profile() {
        // SDR fast-path: u8 + sRGB-class transfer cannot carry HDR, so
        // we skip the per-pixel EOTF walk and return the canonical SDR
        // profile (peak = display reference 80 nits, no HDR/wide-gamut
        // signal, depth 8). Callers that want the actual brightness
        // distribution read Variance / LumaHistogramEntropy from Tier
        // 1 / Tier 3.
        let buf = vec![128u8; 32 * 32 * 3];
        let s = PixelSlice::new(&buf, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert_eq!(d.peak_nits, PEAK_SRGB_NITS);
        assert_eq!(d.headroom_stops, 0.0);
        assert!(!d.hdr_present);
        assert_eq!(d.effective_bit_depth, 8);
        assert_eq!(d.wide_gamut_fraction, 0.0);
    }

    #[test]
    fn solid_white_srgb_u8_at_sdr_reference() {
        // Same fast-path; sRGB code 255 still returns the canonical
        // 80-nit SDR profile. (The pixel-walking version would have
        // returned exactly 80 nits too, since linear 1.0 = 80 nits.)
        let buf = vec![255u8; 32 * 32 * 3];
        let s = PixelSlice::new(&buf, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert_eq!(d.peak_nits, PEAK_SRGB_NITS);
        assert_eq!(d.hdr_pixel_fraction, 0.0);
        assert!(!d.hdr_present);
    }

    #[test]
    fn solid_pq_full_signal_is_high_dynamic_range() {
        // PQ code 1.0 ⇒ linear 1.0 ⇒ 10000 nits per ST 2084.
        let mut buf = vec![0u8; 32 * 32 * 3 * 4];
        let one = 1.0_f32.to_le_bytes();
        for px in buf.chunks_exact_mut(12) {
            px[0..4].copy_from_slice(&one);
            px[4..8].copy_from_slice(&one);
            px[8..12].copy_from_slice(&one);
        }
        let desc = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Pq);
        let s = PixelSlice::new(&buf, 32, 32, 32 * 12, desc).unwrap();
        let d = scan_depth(&s, 100_000);
        assert!(
            (d.peak_nits - 10_000.0).abs() < 5.0,
            "peak={} expected ~10000",
            d.peak_nits
        );
        assert!(d.headroom_stops > 6.0, "headroom={}", d.headroom_stops);
        assert_eq!(d.hdr_pixel_fraction, 1.0);
        assert!(d.hdr_present);
    }

    #[test]
    fn solid_hlg_full_signal_is_hdr_at_1000_nits() {
        // HLG signal 1.0 ⇒ linear 1.0 ⇒ 1000 nits (our convention).
        let mut buf = vec![0u8; 16 * 16 * 12];
        let one = 1.0_f32.to_le_bytes();
        for px in buf.chunks_exact_mut(12) {
            px[0..4].copy_from_slice(&one);
            px[4..8].copy_from_slice(&one);
            px[8..12].copy_from_slice(&one);
        }
        let desc = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Hlg);
        let s = PixelSlice::new(&buf, 16, 16, 16 * 12, desc).unwrap();
        let d = scan_depth(&s, 100_000);
        assert!(
            (d.peak_nits - 1_000.0).abs() < 1.0,
            "peak={} expected ~1000",
            d.peak_nits
        );
        assert!(d.hdr_present);
    }

    #[test]
    fn linear_f32_above_one_is_wide_gamut_signal() {
        // Linear-light f32 with values above 1.0 ⇒ wide-gamut peak.
        let mut buf = vec![0u8; 16 * 16 * 12];
        let two = 2.0_f32.to_le_bytes();
        for px in buf.chunks_exact_mut(12) {
            px[0..4].copy_from_slice(&two);
            px[4..8].copy_from_slice(&two);
            px[8..12].copy_from_slice(&two);
        }
        let s = PixelSlice::new(&buf, 16, 16, 16 * 12, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let d = scan_depth(&s, 100_000);
        assert!((d.wide_gamut_peak - 2.0).abs() < 1e-3);
        assert_eq!(d.wide_gamut_fraction, 1.0);
    }

    #[test]
    fn u8_promoted_u16_reads_as_8bit_effective_depth() {
        // u16 samples = u8 * 257 with very few distinct high bytes.
        // The distinct-low-byte probe should map to 8-bit.
        let mut buf = vec![0u8; 16 * 16 * 6];
        for (i, px) in buf.chunks_exact_mut(2).enumerate() {
            // Cycle through only 4 high-byte values across the image.
            let v = ((i % 4) * 64) as u8;
            let u = (v as u16) * 257;
            px.copy_from_slice(&u.to_le_bytes());
        }
        let s = PixelSlice::new(&buf, 16, 16, 16 * 6, PixelDescriptor::RGB16_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert_eq!(d.effective_bit_depth, 8);
    }

    #[test]
    fn genuine_16bit_u16_reads_as_high_effective_depth() {
        // Sweep u16 values uniformly — many distinct low bytes ⇒
        // effective_bit_depth ≥ 14.
        let mut buf = vec![0u8; 64 * 64 * 6];
        let mut state = 0xC001_u32;
        for px in buf.chunks_exact_mut(2) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let u = (state & 0xFFFF) as u16;
            px.copy_from_slice(&u.to_le_bytes());
        }
        let s = PixelSlice::new(&buf, 64, 64, 64 * 6, PixelDescriptor::RGB16_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert!(
            d.effective_bit_depth >= 14,
            "expected ≥14, got {}",
            d.effective_bit_depth
        );
    }

    #[test]
    fn empty_slice_returns_default_stats() {
        let buf: Vec<u8> = Vec::new();
        let s = PixelSlice::new(&buf, 0, 0, 0, PixelDescriptor::RGB8_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert_eq!(d.peak_nits, 0.0);
        assert_eq!(d.effective_bit_depth, 0);
        assert!(!d.hdr_present);
    }
}
