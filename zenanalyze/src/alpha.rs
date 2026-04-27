//! Alpha-channel analysis — histogram bimodality + non-opaque fraction.
//!
//! Mirrors `zenwebp::encoder::analysis::classifier`'s alpha-histogram
//! shape detector: an alpha distribution with significant mass at both
//! ends (`alpha ∈ [0, 64)` AND `alpha ∈ [192, 256)`) is a strong
//! signal for text-on-transparent / sprites / synthetic-content
//! masks. Photo-class content typically has alpha = 255 everywhere or
//! a gradual roll-off, not bimodal.
//!
//! Independent of the RGB pipeline (RowStream drops alpha), so this
//! module reads the source [`PixelSlice`] directly via raw byte
//! offsets keyed off the descriptor's [`PixelFormat`].

use zenpixels::{AlphaMode, PixelFormat, PixelSlice};

/// Output of an alpha scan. All fields are zero on inputs without a
/// straight-alpha channel (no alpha at all, or premultiplied alpha —
/// premul's alpha is already baked into the colour and the histogram
/// shape stops being meaningful).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AlphaStats {
    pub present: bool,
    pub used_fraction: f32,
    pub bimodal_score: f32,
}

/// Match the descriptor's `PixelFormat` to `(alpha_byte_offset_in_pixel,
/// channel_byte_size, pixel_byte_size)`. Returns `None` for formats
/// without an alpha channel — same shape as
/// `coefficient::analysis::feature_extract::extract_features_from_slice_with`.
fn alpha_layout(fmt: PixelFormat) -> Option<(usize, usize, usize)> {
    match fmt {
        PixelFormat::Rgba8 => Some((3, 1, 4)),
        PixelFormat::Bgra8 => Some((3, 1, 4)),
        PixelFormat::Rgba16 => Some((6, 2, 8)),
        PixelFormat::RgbaF32 => Some((12, 4, 16)),
        PixelFormat::GrayA8 => Some((1, 1, 2)),
        PixelFormat::GrayA16 => Some((2, 2, 4)),
        PixelFormat::GrayAF32 => Some((4, 4, 8)),
        // Rgbx8 / Bgrx8 are padding bytes, not alpha — descriptor.has_alpha()
        // returns false for them. Other formats (Rgb8 / Gray8 / OklabF32 /
        // CMYK / F16-anything) similarly have no straight-alpha to scan.
        _ => None,
    }
}

/// Convert a single pixel's alpha sample to an 8-bit histogram bin.
/// `ch_bytes` is 1 (u8), 2 (u16, little-endian), or 4 (f32 LE in
/// `[0.0, 1.0]`). Out-of-range f32 values are clamped.
#[inline(always)]
fn alpha_byte_at(pixel: &[u8], a_off: usize, ch_bytes: usize) -> u8 {
    match ch_bytes {
        1 => pixel[a_off],
        2 => pixel[a_off + 1], // u16 LE high byte ≡ value >> 8
        4 => {
            let v = f32::from_le_bytes([
                pixel[a_off],
                pixel[a_off + 1],
                pixel[a_off + 2],
                pixel[a_off + 3],
            ]);
            (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        }
        _ => 255,
    }
}

/// Walk the source slice's alpha channel into a 256-bin histogram and
/// derive `AlphaStats`. Stride-samples rows so total work stays within
/// `pixel_budget` (same budget unit as Tier 1 / Tier 2 — alpha scan
/// shares the budget with them by sharing the budget *value*, not by
/// taking from a single pool).
///
/// Returns `AlphaStats::default()` (i.e., `present = false`) for
/// inputs without straight alpha — premultiplied is treated as
/// opaque per coefficient's convention.
pub(crate) fn scan_alpha(slice: &PixelSlice<'_>, pixel_budget: usize) -> AlphaStats {
    let desc = slice.descriptor();
    let alpha_mode = desc.alpha;
    if !matches!(alpha_mode, Some(AlphaMode::Straight)) {
        return AlphaStats::default();
    }
    let (a_off, ch_bytes, bpp) = match alpha_layout(desc.format) {
        Some(t) => t,
        None => return AlphaStats::default(),
    };

    let width = slice.width() as usize;
    let height = slice.rows() as usize;
    if width == 0 || height == 0 {
        return AlphaStats::default();
    }

    // Stride sampling — same shape as Tier 1 / Tier 2: pick a row step
    // so total scanned pixels ≈ pixel_budget.
    let pixels_per_row = width.max(1);
    let target_rows = (pixel_budget / pixels_per_row).max(1).min(height);
    let row_step = (height / target_rows).max(1);

    let mut histogram = [0u32; 256];
    let mut total: u32 = 0;
    let mut y = 0usize;
    while y < height {
        let row = slice.row(y as u32);
        let mut x = 0usize;
        while x < width {
            let off = x * bpp;
            // Defensive: row may be shorter than width*bpp on a tail row.
            if off + a_off + ch_bytes > row.len() {
                break;
            }
            let pixel = &row[off..off + bpp];
            let a = alpha_byte_at(pixel, a_off, ch_bytes);
            histogram[a as usize] += 1;
            total += 1;
            x += 1;
        }
        y += row_step;
    }

    if total == 0 {
        return AlphaStats {
            present: true,
            ..AlphaStats::default()
        };
    }

    let total_f = total as f32;
    let low_quarter: u32 = histogram[..64].iter().sum();
    let high_quarter: u32 = histogram[192..].iter().sum();
    let fully_opaque: u32 = histogram[255];

    let low_frac = low_quarter as f32 / total_f;
    let high_frac = high_quarter as f32 / total_f;
    let used_fraction = 1.0 - (fully_opaque as f32 / total_f);
    let bimodal_score = low_frac.min(high_frac);

    AlphaStats {
        present: true,
        used_fraction,
        bimodal_score,
    }
}
