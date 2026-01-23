//! UltraHDR encoding workflow helpers.
//!
//! This module provides high-level functions for encoding UltraHDR JPEGs
//! from HDR source images.

use crate::encode::extras::EncoderSegments;
use crate::encoder::{EncoderConfig, PixelLayout};
use crate::error::{Error, Result};
use enough::Stop;
use ultrahdr_core::{
    color::tonemap::{tonemap_to_sdr, AdaptiveTonemapper, ToneMapConfig},
    gainmap::{compute_gainmap, GainMapConfig},
    metadata::xmp::generate_xmp,
    ColorTransfer, GainMap, GainMapMetadata, PixelFormat as UhdrPixelFormat, RawImage,
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

/// Encode SDR image with pre-computed gain map.
///
/// Lower-level function for when you already have the SDR and gain map.
fn encode_with_gainmap(
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
        UhdrPixelFormat::Rgba8 => PixelLayout::Rgbx8Srgb,
        UhdrPixelFormat::Rgb8 => PixelLayout::Rgb8Srgb,
        _ => {
            return Err(Error::unsupported_feature(
                "SDR image must be Rgba8 or Rgb8 for UltraHDR encoding",
            ))
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

    // Create output SDR image
    let mut sdr =
        RawImage::new(width, height, UhdrPixelFormat::Rgba8).map_err(ultrahdr_to_jpegli_error)?;
    sdr.gamut = ultrahdr_core::ColorGamut::Bt709;
    sdr.transfer = ColorTransfer::Srgb;

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let hdr_linear = get_linear_rgb(hdr, x, y);

            // Tonemap using ultrahdr-core's unified function
            let sdr_linear = tonemap_to_sdr(hdr_linear, hdr.transfer, config);

            // Apply sRGB OETF and write
            let out_idx = (y * sdr.stride + x * 4) as usize;
            sdr.data[out_idx] = (srgb_oetf(sdr_linear[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
            sdr.data[out_idx + 1] =
                (srgb_oetf(sdr_linear[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
            sdr.data[out_idx + 2] =
                (srgb_oetf(sdr_linear[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
            sdr.data[out_idx + 3] = 255;
        }
    }

    Ok(sdr)
}

/// Extract linear RGB from an HDR image at the given pixel position.
fn get_linear_rgb(img: &RawImage, x: u32, y: u32) -> [f32; 3] {
    match img.format {
        UhdrPixelFormat::Rgba32F => {
            let idx = (y * img.stride + x * 16) as usize;
            let r = f32::from_le_bytes([
                img.data[idx],
                img.data[idx + 1],
                img.data[idx + 2],
                img.data[idx + 3],
            ]);
            let g = f32::from_le_bytes([
                img.data[idx + 4],
                img.data[idx + 5],
                img.data[idx + 6],
                img.data[idx + 7],
            ]);
            let b = f32::from_le_bytes([
                img.data[idx + 8],
                img.data[idx + 9],
                img.data[idx + 10],
                img.data[idx + 11],
            ]);
            [r, g, b]
        }
        UhdrPixelFormat::Rgba16F => {
            let idx = (y * img.stride + x * 8) as usize;
            let r = half_to_f32(&img.data[idx..idx + 2]);
            let g = half_to_f32(&img.data[idx + 2..idx + 4]);
            let b = half_to_f32(&img.data[idx + 4..idx + 6]);
            [r, g, b]
        }
        UhdrPixelFormat::Rgba8 | UhdrPixelFormat::Rgb8 => {
            let bpp = if img.format == UhdrPixelFormat::Rgba8 {
                4
            } else {
                3
            };
            let idx = (y * img.stride + x * bpp as u32) as usize;
            let r = img.data[idx] as f32 / 255.0;
            let g = img.data[idx + 1] as f32 / 255.0;
            let b = img.data[idx + 2] as f32 / 255.0;
            // Assume sRGB for 8-bit, apply EOTF
            [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
        }
        _ => [0.18, 0.18, 0.18], // Fallback to mid-gray
    }
}

/// Convert half-precision float bytes to f32.
fn half_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
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
            if sign == 1 {
                -result
            } else {
                result
            }
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
    fn test_half_to_f32() {
        // Test zero
        assert_eq!(half_to_f32(&[0, 0]), 0.0);

        // Test one (0x3C00)
        let one = half_to_f32(&[0x00, 0x3C]);
        assert!((one - 1.0).abs() < 0.001);

        // Test negative one (0xBC00)
        let neg_one = half_to_f32(&[0x00, 0xBC]);
        assert!((neg_one + 1.0).abs() < 0.001);
    }
}
