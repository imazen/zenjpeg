//! ICC color management module.
//!
//! Provides unified ICC profile detection and color conversion for JPEG decoding.
//! Automatically detects XYB profiles and applies appropriate color transformation.
//!
//! # Features
//!
//! - `moxcms`: Enable color management via moxcms (pure Rust with SIMD)
//!
//! # Example
//!
//! ```ignore
//! use zenjpeg::icc::IccDecoder;
//!
//! let decoder = IccDecoder::new();
//! let (rgb, width, height) = decoder.decode_jpeg(&jpeg_data)?;
//! // ICC profile is automatically applied if present
//! ```

#![allow(unused_imports)] // Imports used by decoder-only functions

use crate::error::{Error, Result};

/// ICC profile signature in APP2 marker
pub const ICC_PROFILE_SIGNATURE: &[u8; 12] = b"ICC_PROFILE\0";

/// Target color space for ICC profile conversion during decoding.
///
/// When [`Decoder::correct_color`](crate::decode::Decoder::correct_color) is set
/// to `Some(target)`, embedded ICC profiles are converted to this target color space.
///
/// When the `moxcms` feature is disabled, the type is still available for API
/// compatibility but color conversion is a no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TargetColorSpace {
    /// sRGB (IEC 61966-2-1). The universal web/display standard.
    #[default]
    Srgb,
    /// Display P3 (DCI-P3 primaries, sRGB transfer function, D65 white point).
    DisplayP3,
    /// ITU-R BT.2020 (wide gamut, 2.4 gamma, D65 white point).
    Rec2020,
}

/// XYB profile description substring for detection
const XYB_PROFILE_MARKER: &[u8] = b"XYB";

/// Extract ICC profile from JPEG data.
///
/// ICC profiles are stored in APP2 markers with signature "ICC_PROFILE\0".
/// Large profiles may be split across multiple APP2 markers.
///
/// Scans marker-to-marker (not byte-by-byte), stopping at SOS.
///
/// The cumulative profile size is checked **incrementally** against
/// [`MAX_ICC_PROFILE_SIZE`](crate::foundation::alloc::MAX_ICC_PROFILE_SIZE).
/// As soon as a chunk would push the running total over the cap, we abort
/// and return `None` without copying that chunk — bounding peak memory
/// usage during extraction to roughly the cap size rather than the full
/// length of the input JPEG.
pub fn extract_icc_profile(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut running_total: usize = 0;

    // Skip SOI (0xFF 0xD8)
    let mut i = 2;

    // Scan marker-by-marker until SOS or end of markers
    while i + 3 < jpeg_data.len() {
        // Look for marker prefix
        if jpeg_data[i] != 0xFF {
            break; // Not a marker, likely scan data
        }

        // Skip any fill bytes (0xFF padding)
        while i + 1 < jpeg_data.len() && jpeg_data[i + 1] == 0xFF {
            i += 1;
        }
        if i + 3 >= jpeg_data.len() {
            break;
        }

        let marker_type = jpeg_data[i + 1];

        // Stop at SOS (start of scan) or EOI
        if marker_type == 0xDA || marker_type == 0xD9 {
            break;
        }

        // Markers without length (RST, SOI, EOI, TEM)
        if marker_type == 0x00 || marker_type == 0x01 || (0xD0..=0xD7).contains(&marker_type) {
            i += 2;
            continue;
        }

        let length = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
        if length < 2 || i + 2 + length > jpeg_data.len() {
            break;
        }

        // Check for APP2 with ICC_PROFILE signature
        if marker_type == 0xE2 && length >= 16 && &jpeg_data[i + 4..i + 16] == ICC_PROFILE_SIGNATURE
        {
            let chunk_num = jpeg_data[i + 16];
            let _total_chunks = jpeg_data[i + 17];
            let chunk_payload = &jpeg_data[i + 18..i + 2 + length];

            // Enforce the cap incrementally so a JPEG with many APP2/ICC
            // chunks (each individually under u16::MAX bytes) cannot
            // accumulate gigabytes of u8 vectors into `chunks` before the
            // post-loop size check fires.
            let next_total = running_total.checked_add(chunk_payload.len())?;
            if next_total > crate::foundation::alloc::MAX_ICC_PROFILE_SIZE {
                return None;
            }
            running_total = next_total;

            chunks.push((chunk_num, chunk_payload.to_vec()));
        }

        i += 2 + length;
    }

    if chunks.is_empty() {
        return None;
    }

    // Sort by chunk number and concatenate. The per-chunk incremental
    // cap above already guarantees `running_total` <= MAX_ICC_PROFILE_SIZE.
    chunks.sort_by_key(|(num, _)| *num);

    let mut profile: Vec<u8> = Vec::new();
    profile.try_reserve_exact(running_total).ok()?;
    for (_, data) in chunks {
        profile.extend_from_slice(&data);
    }

    Some(profile)
}

/// Check if an ICC profile is an XYB profile from jpegli/JPEG XL.
pub fn is_xyb_profile(icc_data: &[u8]) -> bool {
    // Fast path: exact match against the known XYB ICC profile (720 bytes).
    // This is the profile embedded by zenjpeg and jpegli for XYB-encoded JPEGs.
    use crate::foundation::consts::XYB_ICC_PROFILE;
    if icc_data == XYB_ICC_PROFILE {
        return true;
    }

    // Fallback: check for "XYB" in profile description (ASCII or UTF-16BE).
    // XYB profiles have "XYB_Per" as description text.
    //
    // NOTE: The "jxl " CMM type (bytes 4-7) is NOT sufficient — cjpegli writes
    // "jxl " for ALL ICC profiles (including standard sRGB), not just XYB ones.
    // We must check for the XYB description text to avoid false positives.
    const XYB_UTF16BE: [u8; 6] = [0, b'X', 0, b'Y', 0, b'B'];
    icc_data
        .windows(XYB_PROFILE_MARKER.len())
        .any(|w| w == XYB_PROFILE_MARKER)
        || icc_data.windows(6).any(|w| w == XYB_UTF16BE)
}

/// Apply ICC profile transformation to RGB image data.
///
/// Converts from the input profile's color space to the specified target.
/// Apply ICC profile transformation using moxcms (pure Rust).
#[cfg(feature = "moxcms")]
pub fn apply_icc_transform(
    rgb_data: &[u8],
    _width: usize,
    _height: usize,
    icc_profile: &[u8],
    target: TargetColorSpace,
) -> Result<Vec<u8>> {
    use moxcms::{ColorProfile, Layout};

    let input_profile = ColorProfile::new_from_slice(icc_profile)
        .map_err(|e| Error::icc_error(format!("moxcms: {e:?}")))?;

    let output_profile = make_moxcms_target(target);

    let transform = input_profile
        .create_transform_8bit(
            Layout::Rgb,
            &output_profile,
            Layout::Rgb,
            moxcms_transform_opts(),
        )
        .map_err(|e| Error::icc_error(format!("moxcms transform: {e:?}")))?;

    let mut output = vec![0u8; rgb_data.len()];
    transform
        .transform(rgb_data, &mut output)
        .map_err(|e| Error::icc_error(format!("moxcms transform execution: {e:?}")))?;

    Ok(output)
}

/// Fallback when no CMS feature is enabled.
#[cfg(not(feature = "moxcms"))]
pub fn apply_icc_transform(
    rgb_data: &[u8],
    _width: usize,
    _height: usize,
    _icc_profile: &[u8],
    _target: TargetColorSpace,
) -> Result<Vec<u8>> {
    // No CMS available - return data unchanged
    // User should enable moxcms feature for ICC support
    Ok(rgb_data.to_vec())
}

// ============================================================================
// f32 ICC transform variants
// ============================================================================

/// Apply ICC profile transformation to f32 RGB image data.
///
/// Input and output are interleaved RGB f32 in [0.0, 1.0] range.
/// Converts from the input profile's color space to the specified target.
/// Apply ICC profile transformation to f32 using moxcms (pure Rust).
#[cfg(feature = "moxcms")]
pub fn apply_icc_transform_f32(
    rgb_data: &[f32],
    _width: usize,
    _height: usize,
    icc_profile: &[u8],
    target: TargetColorSpace,
) -> Result<Vec<f32>> {
    use moxcms::{ColorProfile, Layout};

    let input_profile = ColorProfile::new_from_slice(icc_profile)
        .map_err(|e| Error::icc_error(format!("moxcms: {e:?}")))?;

    let output_profile = make_moxcms_target(target);

    let transform = input_profile
        .create_transform_f32(
            Layout::Rgb,
            &output_profile,
            Layout::Rgb,
            moxcms_transform_opts(),
        )
        .map_err(|e| Error::icc_error(format!("moxcms f32 transform: {e:?}")))?;

    let mut output = vec![0f32; rgb_data.len()];
    transform
        .transform(rgb_data, &mut output)
        .map_err(|e| Error::icc_error(format!("moxcms f32 transform execution: {e:?}")))?;

    Ok(output)
}

/// Fallback when no CMS feature is enabled.
#[cfg(not(feature = "moxcms"))]
pub fn apply_icc_transform_f32(
    rgb_data: &[f32],
    _width: usize,
    _height: usize,
    _icc_profile: &[u8],
    _target: TargetColorSpace,
) -> Result<Vec<f32>> {
    Ok(rgb_data.to_vec())
}

/// Standard moxcms transform options for ICC color transforms.
#[cfg(feature = "moxcms")]
fn moxcms_transform_opts() -> moxcms::TransformOptions {
    use moxcms::{BarycentricWeightScale, InterpolationMethod, TransformOptions};
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

/// Create a moxcms `ColorProfile` for the given target.
#[cfg(feature = "moxcms")]
fn make_moxcms_target(target: TargetColorSpace) -> moxcms::ColorProfile {
    match target {
        TargetColorSpace::Srgb => moxcms::ColorProfile::new_srgb(),
        TargetColorSpace::DisplayP3 => moxcms::ColorProfile::new_display_p3(),
        TargetColorSpace::Rec2020 => moxcms::ColorProfile::new_bt2020(),
    }
}

// ============================================================================
// sRGB → Linear transfer function
// ============================================================================

/// Convert sRGB gamma-encoded f32 values to linear light.
///
/// Applies the sRGB EOTF (Electro-Optical Transfer Function) per IEC 61966-2-1.
/// Input values should be in [0.0, 1.0] nominal range (may exceed for unclamped data).
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    linear_srgb::default::srgb_to_linear(v)
}

/// Convert an entire f32 RGB pixel buffer from sRGB gamma to linear light.
///
/// Operates in-place for efficiency. Each channel is independently linearized.
/// Uses SIMD-accelerated batch conversion when available.
pub fn srgb_to_linear_inplace(pixels: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_slice(pixels);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_xyb_profile() {
        // XYB profile should contain "XYB" marker
        let xyb_profile = b"...XYB_Per...";
        assert!(is_xyb_profile(xyb_profile));

        // Known XYB ICC profile constant should match
        assert!(is_xyb_profile(&crate::foundation::consts::XYB_ICC_PROFILE));

        // Regular sRGB shouldn't match
        let srgb = b"sRGB IEC61966-2.1";
        assert!(!is_xyb_profile(srgb));

        // Regression: "jxl " CMM type alone must NOT trigger XYB detection.
        // cjpegli writes "jxl " for ALL ICC profiles including standard sRGB.
        let mut jxl_srgb = vec![0u8; 128];
        jxl_srgb[4..8].copy_from_slice(b"jxl ");
        jxl_srgb[8..23].copy_from_slice(b"sRGB IEC61966-2");
        assert!(
            !is_xyb_profile(&jxl_srgb),
            "jxl CMM type alone should not be detected as XYB"
        );
    }

    #[test]
    fn test_extract_icc_profile_empty() {
        let no_icc = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]; // JFIF
        assert!(extract_icc_profile(&no_icc).is_none());
    }

    /// Regression for the security audit's H4 finding: many APP2/ICC
    /// chunks whose individual lengths fit in u16 but whose sum exceeds
    /// `MAX_ICC_PROFILE_SIZE` must be rejected without buffering hundreds
    /// of MB of u8 vectors first. We construct a JPEG-shaped byte sequence
    /// with enough chunks to (in aggregate) exceed the 16 MB cap and
    /// assert `extract_icc_profile` returns `None`.
    #[test]
    fn test_extract_icc_profile_oversize_rejected_incrementally() {
        use crate::foundation::alloc::MAX_ICC_PROFILE_SIZE;

        // Build a JPEG-like buffer: SOI then a long string of APP2 ICC
        // chunks. Each chunk: 0xFF 0xE2 [len_hi len_lo] "ICC_PROFILE\0"
        // [chunk_num total_chunks] [payload..]. We pick payload_len so
        // that ~2x MAX_ICC_PROFILE_SIZE worth of payload would be
        // accumulated if the cap weren't enforced incrementally.
        let payload_len: usize = 65500 - 16; // marker payload after sig+chunk hdrs
        let mut buf: Vec<u8> = Vec::with_capacity(MAX_ICC_PROFILE_SIZE * 2);
        buf.extend_from_slice(&[0xFF, 0xD8]); // SOI
        let mut total = 0usize;
        let mut chunk_num: u8 = 1;
        while total < MAX_ICC_PROFILE_SIZE * 2 {
            let length = 2 + 12 + 2 + payload_len;
            buf.push(0xFF);
            buf.push(0xE2);
            buf.push((length >> 8) as u8);
            buf.push((length & 0xFF) as u8);
            buf.extend_from_slice(ICC_PROFILE_SIGNATURE);
            buf.push(chunk_num);
            buf.push(0xFF); // total_chunks (unused in our scanner)
            buf.extend(std::iter::repeat_n(0u8, payload_len));
            total += payload_len;
            chunk_num = chunk_num.wrapping_add(1);
        }
        // Trailing EOI to terminate the marker walk cleanly
        buf.extend_from_slice(&[0xFF, 0xD9]);

        // The function must reject without exploding peak memory. We
        // can't directly assert peak memory in a unit test, but we *can*
        // assert it returns None, and the buffer-building above runs in
        // bounded ~2x MAX_ICC_PROFILE_SIZE bytes — meaning the function
        // iterates over our crafted input and refuses, which is the
        // intended fix.
        assert!(extract_icc_profile(&buf).is_none());
    }
}
