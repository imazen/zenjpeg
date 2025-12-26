//! ICC color management module.
//!
//! Provides unified ICC profile detection and color conversion for JPEG decoding.
//! Automatically detects XYB profiles and applies appropriate color transformation.
//!
//! # Features
//!
//! - `cms-lcms2`: Use Little CMS 2 (mature C library bindings)
//! - `cms-moxcms`: Use moxcms (pure Rust with SIMD)
//!
//! # Example
//!
//! ```ignore
//! use jpegli::icc::IccDecoder;
//!
//! let decoder = IccDecoder::new();
//! let (rgb, width, height) = decoder.decode_jpeg(&jpeg_data)?;
//! // ICC profile is automatically applied if present
//! ```

use crate::jpegli::error::{Error, Result};

/// ICC profile signature in APP2 marker
pub const ICC_PROFILE_SIGNATURE: &[u8; 12] = b"ICC_PROFILE\0";

/// XYB profile description substring for detection
const XYB_PROFILE_MARKER: &[u8] = b"XYB";

/// Extract ICC profile from JPEG data.
///
/// ICC profiles are stored in APP2 markers with signature "ICC_PROFILE\0".
/// Large profiles may be split across multiple APP2 markers.
pub fn extract_icc_profile(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut i = 0;

    while i < jpeg_data.len().saturating_sub(1) {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xE2 {
            // APP2 marker
            if i + 4 > jpeg_data.len() {
                break;
            }
            let length = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
            if i + 2 + length > jpeg_data.len() {
                break;
            }

            // Check for ICC_PROFILE signature
            if length >= 16 && &jpeg_data[i + 4..i + 16] == ICC_PROFILE_SIGNATURE {
                let chunk_num = jpeg_data[i + 16];
                let _total_chunks = jpeg_data[i + 17];
                let icc_data = jpeg_data[i + 18..i + 2 + length].to_vec();
                chunks.push((chunk_num, icc_data));
            }
            i += 2 + length;
        } else {
            i += 1;
        }
    }

    if chunks.is_empty() {
        return None;
    }

    // Sort by chunk number and concatenate
    chunks.sort_by_key(|(num, _)| *num);
    let profile: Vec<u8> = chunks.into_iter().flat_map(|(_, data)| data).collect();

    Some(profile)
}

/// Check if an ICC profile is an XYB profile.
pub fn is_xyb_profile(icc_data: &[u8]) -> bool {
    // Check for "XYB" in profile description
    // The XYB profile typically has "XYB_Per" in the description tag
    icc_data
        .windows(XYB_PROFILE_MARKER.len())
        .any(|w| w == XYB_PROFILE_MARKER)
}

/// Apply ICC profile transformation to RGB image data.
///
/// Converts from the input profile's color space to sRGB.
#[cfg(feature = "cms-lcms2")]
pub fn apply_icc_transform(
    rgb_data: &[u8],
    width: usize,
    height: usize,
    icc_profile: &[u8],
) -> Result<Vec<u8>> {
    use lcms2::{Intent, PixelFormat, Profile, Transform};

    let input_profile =
        Profile::new_icc(icc_profile).map_err(|e| Error::IccError(format!("lcms2: {e}")))?;

    let srgb = Profile::new_srgb();

    // Use RelativeColorimetric for best accuracy with XYB profiles
    // Testing showed Perceptual intent adds ~14% more error vs RelativeColorimetric
    let transform = Transform::new(
        &input_profile,
        PixelFormat::RGB_8,
        &srgb,
        PixelFormat::RGB_8,
        Intent::RelativeColorimetric,
    )
    .map_err(|e| Error::IccError(format!("lcms2 transform: {e}")))?;

    let pixels: Vec<[u8; 3]> = rgb_data
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    let mut output = vec![[0u8; 3]; pixels.len()];
    transform.transform_pixels(&pixels, &mut output);

    Ok(output.into_iter().flat_map(|p| p).collect())
}

/// Apply ICC profile transformation using moxcms (pure Rust).
#[cfg(all(feature = "cms-moxcms", not(feature = "cms-lcms2")))]
pub fn apply_icc_transform(
    rgb_data: &[u8],
    _width: usize,
    _height: usize,
    icc_profile: &[u8],
) -> Result<Vec<u8>> {
    use moxcms::{ColorProfile, Layout, TransformOptions};

    let input_profile = ColorProfile::new_from_slice(icc_profile)
        .map_err(|e| Error::IccError(format!("moxcms: {e:?}")))?;

    let srgb = ColorProfile::new_srgb();

    let transform = input_profile
        .create_transform_8bit(Layout::Rgb, &srgb, Layout::Rgb, TransformOptions::default())
        .map_err(|e| Error::IccError(format!("moxcms transform: {e:?}")))?;

    let mut output = vec![0u8; rgb_data.len()];
    transform
        .transform(rgb_data, &mut output)
        .map_err(|e| Error::IccError(format!("moxcms transform execution: {e:?}")))?;

    Ok(output)
}

/// Fallback when no CMS feature is enabled.
#[cfg(not(any(feature = "cms-lcms2", feature = "cms-moxcms")))]
pub fn apply_icc_transform(
    rgb_data: &[u8],
    _width: usize,
    _height: usize,
    _icc_profile: &[u8],
) -> Result<Vec<u8>> {
    // No CMS available - return data unchanged
    // User should enable cms-lcms2 or cms-moxcms feature for ICC support
    Ok(rgb_data.to_vec())
}

/// Decode JPEG with automatic ICC profile application.
///
/// This is a convenience function that:
/// 1. Decodes the JPEG using jpeg-decoder
/// 2. Extracts any embedded ICC profile
/// 3. Applies the ICC transform if present and CMS is available
///
/// Available when `cms-lcms2` or `cms-moxcms` feature is enabled.
#[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
pub fn decode_jpeg_with_icc(jpeg_data: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    // Extract ICC profile first
    let icc_profile = extract_icc_profile(jpeg_data);

    // Decode JPEG
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let pixels = decoder
        .decode()
        .map_err(|e| Error::DecodeError(format!("jpeg decode: {e}")))?;

    let info = decoder
        .info()
        .ok_or_else(|| Error::DecodeError("no image info".to_string()))?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Apply ICC if present
    let output = if let Some(ref profile) = icc_profile {
        apply_icc_transform(&pixels, width, height, profile)?
    } else {
        pixels
    };

    Ok((output, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_xyb_profile() {
        // XYB profile should contain "XYB" marker
        let xyb_profile = b"...XYB_Per...";
        assert!(is_xyb_profile(xyb_profile));

        // Regular sRGB shouldn't match
        let srgb = b"sRGB IEC61966-2.1";
        assert!(!is_xyb_profile(srgb));
    }

    #[test]
    fn test_extract_icc_profile_empty() {
        let no_icc = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]; // JFIF
        assert!(extract_icc_profile(&no_icc).is_none());
    }
}
