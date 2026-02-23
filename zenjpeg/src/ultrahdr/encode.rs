//! UltraHDR encoding workflow helpers.
//!
//! This module provides high-level functions for encoding UltraHDR JPEGs
//! from HDR source images.

use crate::encode::extras::EncoderSegments;
use crate::encoder::{EncoderConfig, PixelLayout};
use crate::error::{Error, Result};
use enough::Stop;
use ultrahdr_core::{
    ColorGamut, ColorTransfer, GainMap, GainMapMetadata, PixelFormat as UhdrPixelFormat, RawImage,
    color::tonemap::{AdaptiveTonemapper, ToneMapConfig, tonemap_to_sdr},
    gainmap::{GainMapConfig, RowEncoder, compute_gainmap},
    metadata::xmp::generate_xmp,
};

/// Encode an HDR image as UltraHDR JPEG.
///
/// This performs the full UltraHDR encoding workflow:
/// 1. Tonemap HDR to SDR using the provided config
/// 2. Compute gain map from HDR/SDR pair
/// 3. Encode SDR base image with jpegli
/// 4. Encode gain map as grayscale JPEG
/// 5. Generate XMP metadata
/// 6. Assemble final UltraHDR JPEG with MPF structure
///
/// # Arguments
///
/// * `hdr` - Source HDR image (linear float, PQ, or HLG)
/// * `gainmap_config` - Configuration for gain map computation
/// * `tonemap_config` - Configuration for HDR→SDR tonemapping
/// * `encoder_config` - jpegli encoder configuration for the base image
/// * `gainmap_quality` - JPEG quality for the gain map (typically 75)
/// * `stop` - Cooperative cancellation token
///
/// # Returns
///
/// Complete UltraHDR JPEG bytes ready for writing to disk or network.
pub fn encode_ultrahdr(
    hdr: &RawImage,
    gainmap_config: &GainMapConfig,
    tonemap_config: &ToneMapConfig,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    // Step 1: Tonemap HDR to SDR
    let sdr = tonemap_hdr_to_sdr(hdr, tonemap_config)?;
    stop.check()?;

    // Step 2: Compute gain map
    let (gainmap, metadata) = compute_gainmap(hdr, &sdr, gainmap_config, &stop)?;
    stop.check()?;

    // Step 3-6: Encode and assemble
    encode_with_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        encoder_config,
        gainmap_quality,
        stop,
    )
}

/// Encode UltraHDR using a pre-learned adaptive tonemapper.
///
/// Use this when re-encoding edited HDR content to preserve the original
/// tonemapping relationship. The adaptive tonemapper learns the HDR→SDR
/// curve from an existing pair and can reproduce it for modified content.
///
/// # Arguments
///
/// * `hdr` - Modified HDR image
/// * `tonemapper` - Adaptive tonemapper learned from original HDR/SDR pair
/// * `gainmap_config` - Configuration for gain map computation
/// * `encoder_config` - jpegli encoder configuration
/// * `gainmap_quality` - JPEG quality for the gain map
/// * `stop` - Cooperative cancellation token
pub fn encode_ultrahdr_with_tonemapper(
    hdr: &RawImage,
    tonemapper: &AdaptiveTonemapper,
    gainmap_config: &GainMapConfig,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    // Step 1: Apply adaptive tonemapper
    let sdr = tonemapper.apply(hdr).map_err(ultrahdr_to_jpegli_error)?;
    stop.check()?;

    // Step 2: Compute gain map
    let (gainmap, metadata) = compute_gainmap(hdr, &sdr, gainmap_config, &stop)?;
    stop.check()?;

    // Step 3-6: Encode and assemble
    encode_with_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        encoder_config,
        gainmap_quality,
        stop,
    )
}

/// Create a streaming gain map computer for row-by-row processing.
///
/// This is more memory-efficient than the full-image [`compute_gainmap`] for large images,
/// as it processes rows in batches rather than loading the entire image.
///
/// Input data must be **linear f32 RGB** for both HDR and SDR. The caller is
/// responsible for converting encoded formats (sRGB, PQ, HLG) to linear f32
/// before feeding rows to the encoder.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `config` - Gain map computation configuration
/// * `hdr_gamut` - HDR color gamut
///
/// # Returns
///
/// A [`RowEncoder`] that can process HDR/SDR linear f32 row pairs.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::ultrahdr::{create_gainmap_computer, GainMapConfig, UhdrColorGamut};
///
/// let mut computer = create_gainmap_computer(
///     width, height,
///     &GainMapConfig::default(),
///     UhdrColorGamut::Bt709,
/// )?;
///
/// // Process rows in batches (both HDR and SDR must be linear f32 RGB)
/// for batch_start in (0..height).step_by(16) {
///     let batch_height = 16.min(height - batch_start);
///     let gm_rows = computer.process_rows(&hdr_linear_f32, &sdr_linear_f32, batch_height)?;
///     // gm_rows contains completed gainmap rows (if any)
/// }
///
/// // Finish and get the complete gainmap
/// let (gainmap, metadata) = computer.finish()?;
/// ```
pub fn create_gainmap_computer(
    width: u32,
    height: u32,
    config: &GainMapConfig,
    hdr_gamut: ColorGamut,
) -> Result<RowEncoder> {
    RowEncoder::new(width, height, config.clone(), hdr_gamut, ColorGamut::Bt709)
        .map_err(ultrahdr_to_jpegli_error)
}

/// Encode SDR image with pre-computed gain map.
///
/// Lower-level function for when you already have the SDR and gain map.
pub fn encode_with_gainmap(
    sdr: &RawImage,
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    // Encode gain map as grayscale JPEG
    let gainmap_jpeg = encode_gainmap_jpeg(gainmap, gainmap_quality, &stop)?;
    stop.check()?;

    // Generate XMP metadata
    let xmp = generate_xmp(metadata, gainmap_jpeg.len());

    // Create encoder segments with XMP and gain map (chained builder pattern)
    let segments = EncoderSegments::new()
        .set_xmp(&xmp)
        .add_mpf_image(gainmap_jpeg, crate::encode::extras::MpfImageType::Undefined);

    // Encode base SDR image with the segments
    let base_jpeg = encode_sdr_base(sdr, encoder_config, segments, stop)?;

    Ok(base_jpeg)
}

/// Encode the gain map as a grayscale JPEG.
fn encode_gainmap_jpeg(gainmap: &GainMap, quality: f32, stop: &impl Stop) -> Result<Vec<u8>> {
    let config = EncoderConfig::grayscale(quality);

    let mut encoder = config.encode_from_bytes(
        gainmap.width,
        gainmap.height,
        if gainmap.channels == 1 {
            PixelLayout::Gray8Srgb
        } else {
            PixelLayout::Rgb8Srgb
        },
    )?;

    encoder.push_packed(&gainmap.data, stop)?;
    encoder.finish()
}

/// Encode the SDR base image.
fn encode_sdr_base(
    sdr: &RawImage,
    config: &EncoderConfig,
    segments: EncoderSegments,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    // Determine pixel layout from SDR format
    let layout = match sdr.format {
        UhdrPixelFormat::Rgba8 => PixelLayout::Rgba8Srgb,
        UhdrPixelFormat::Rgb8 => PixelLayout::Rgb8Srgb,
        _ => {
            return Err(Error::unsupported_feature(
                "SDR image must be Rgba8 or Rgb8 for UltraHDR encoding",
            ));
        }
    };

    let config_with_segments = config.clone().with_segments(segments);

    let mut encoder = config_with_segments.encode_from_bytes(sdr.width, sdr.height, layout)?;

    encoder.push_packed(&sdr.data, stop)?;
    encoder.finish()
}

/// Tonemap HDR to SDR using the provided config.
fn tonemap_hdr_to_sdr(hdr: &RawImage, config: &ToneMapConfig) -> Result<RawImage> {
    let width = hdr.width;
    let height = hdr.height;

    // Validate input buffer size
    let bytes_per_pixel = match hdr.format {
        UhdrPixelFormat::Rgba32F => 16,
        UhdrPixelFormat::Rgba16F => 8,
        UhdrPixelFormat::Rgba8 => 4,
        UhdrPixelFormat::Rgb8 => 3,
        _ => {
            return Err(Error::unsupported_feature(
                "Unsupported HDR pixel format for tonemapping",
            ));
        }
    };
    let expected_size = (height * hdr.stride) as usize;
    if hdr.data.len() < expected_size {
        return Err(Error::invalid_buffer_size(expected_size, hdr.data.len()));
    }

    // Create output SDR image
    let mut sdr =
        RawImage::new(width, height, UhdrPixelFormat::Rgba8).map_err(ultrahdr_to_jpegli_error)?;
    sdr.gamut = ultrahdr_core::ColorGamut::Bt709;
    sdr.transfer = ColorTransfer::Srgb;

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let hdr_linear = get_linear_rgb_safe(hdr, x, y, bytes_per_pixel);

            // Tonemap using ultrahdr-core's unified function
            let sdr_linear = tonemap_to_sdr(hdr_linear, hdr.transfer, config);

            // Apply sRGB OETF and write (bounds already validated by RawImage::new)
            let out_idx = (y * sdr.stride + x * 4) as usize;
            if let Some(slice) = sdr.data.get_mut(out_idx..out_idx + 4) {
                slice[0] = (srgb_oetf(sdr_linear[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
                slice[1] = (srgb_oetf(sdr_linear[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
                slice[2] = (srgb_oetf(sdr_linear[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
                slice[3] = 255;
            }
        }
    }

    Ok(sdr)
}

/// Extract linear RGB from an HDR image at the given pixel position.
/// Uses bounds-checked access to avoid panics on malformed data.
fn get_linear_rgb_safe(img: &RawImage, x: u32, y: u32, bytes_per_pixel: usize) -> [f32; 3] {
    let idx = (y * img.stride + x * bytes_per_pixel as u32) as usize;

    match img.format {
        UhdrPixelFormat::Rgba32F => {
            // Need 12 bytes for RGB (skip alpha)
            if let Some(slice) = img.data.get(idx..idx + 12) {
                let r = f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
                let g = f32::from_le_bytes([slice[4], slice[5], slice[6], slice[7]]);
                let b = f32::from_le_bytes([slice[8], slice[9], slice[10], slice[11]]);
                [r, g, b]
            } else {
                [0.18, 0.18, 0.18] // Mid-gray fallback
            }
        }
        UhdrPixelFormat::Rgba16F => {
            // Need 6 bytes for RGB (skip alpha)
            if let Some(slice) = img.data.get(idx..idx + 6) {
                let r = half_to_f32_safe(slice.get(0..2));
                let g = half_to_f32_safe(slice.get(2..4));
                let b = half_to_f32_safe(slice.get(4..6));
                [r, g, b]
            } else {
                [0.18, 0.18, 0.18]
            }
        }
        UhdrPixelFormat::Rgba8 | UhdrPixelFormat::Rgb8 => {
            // Need 3 bytes for RGB
            if let Some(slice) = img.data.get(idx..idx + 3) {
                let r = slice[0] as f32 / 255.0;
                let g = slice[1] as f32 / 255.0;
                let b = slice[2] as f32 / 255.0;
                // Assume sRGB for 8-bit, apply EOTF
                [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
            } else {
                [0.18, 0.18, 0.18]
            }
        }
        _ => [0.18, 0.18, 0.18], // Fallback to mid-gray
    }
}

/// Convert half-precision float bytes to f32 (bounds-checked).
fn half_to_f32_safe(bytes: Option<&[u8]>) -> f32 {
    let Some(bytes) = bytes else {
        return 0.0;
    };
    let Some(&b0) = bytes.first() else {
        return 0.0;
    };
    let Some(&b1) = bytes.get(1) else {
        return 0.0;
    };
    let bits = u16::from_le_bytes([b0, b1]);
    // Manual half-float conversion (avoiding dependency on half crate)
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Denormalized or zero
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormalized
            let e = (mant as f32).log2().floor() as i32;
            let m = ((mant as f32) / (1 << (e + 1)) as f32 - 0.5) * 2.0;
            let result = (1.0 + m) * 2.0f32.powi(-14 + e);
            if sign == 1 { -result } else { result }
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        // Normalized
        let exp32 = exp + 127 - 15;
        let mant32 = mant << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
    }
}

/// sRGB OETF (linear to gamma)
fn srgb_oetf(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB EOTF (gamma to linear)
fn srgb_eotf(gamma: f32) -> f32 {
    if gamma <= 0.04045 {
        gamma / 12.92
    } else {
        ((gamma + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert ultrahdr_core::Error to jpegli Error.
fn ultrahdr_to_jpegli_error(e: ultrahdr_core::Error) -> Error {
    Error::decode_error(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_roundtrip() {
        for i in 0..256 {
            let gamma = i as f32 / 255.0;
            let linear = srgb_eotf(gamma);
            let back = srgb_oetf(linear);
            assert!(
                (gamma - back).abs() < 0.001,
                "Failed at {}: {} -> {} -> {}",
                i,
                gamma,
                linear,
                back
            );
        }
    }

    #[test]
    fn test_half_to_f32_safe() {
        // Test None
        assert_eq!(half_to_f32_safe(None), 0.0);

        // Test zero
        assert_eq!(half_to_f32_safe(Some(&[0, 0])), 0.0);

        // Test one (0x3C00)
        let one = half_to_f32_safe(Some(&[0x00, 0x3C]));
        assert!((one - 1.0).abs() < 0.001);

        // Test negative one (0xBC00)
        let neg_one = half_to_f32_safe(Some(&[0x00, 0xBC]));
        assert!((neg_one + 1.0).abs() < 0.001);

        // Test incomplete slice (should return 0.0)
        assert_eq!(half_to_f32_safe(Some(&[0x00])), 0.0);
        assert_eq!(half_to_f32_safe(Some(&[])), 0.0);
    }
}
