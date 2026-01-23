//! UltraHDR decoding workflow helpers.
//!
//! This module provides high-level functions for decoding UltraHDR JPEGs
//! and reconstructing HDR content.

use crate::decode::DecodedExtras;
use crate::decoder::Decoder;
use crate::error::{Error, Result};
use enough::Stop;
use ultrahdr_core::{
    color::tonemap::AdaptiveTonemapper,
    gainmap::{apply_gainmap, HdrOutputFormat},
    metadata::xmp::parse_xmp,
    GainMap, GainMapMetadata, RawImage,
};

/// Extension trait for [`DecodedExtras`] to check for UltraHDR content.
///
/// This trait provides methods to detect UltraHDR images and extract
/// their gain map metadata without fully decoding the gain map.
pub trait UltraHdrExtras {
    /// Check if this JPEG contains UltraHDR gain map metadata.
    ///
    /// Returns `true` if the XMP contains HDR gain map attributes
    /// (`hdrgm:Version` or `hdrgm:GainMapMax`).
    fn is_ultrahdr(&self) -> bool;

    /// Parse and return the gain map metadata from XMP.
    ///
    /// Returns `None` if no XMP is present or it's not UltraHDR.
    /// Returns `Some(Err(...))` if XMP is present but parsing fails.
    fn ultrahdr_metadata(&self) -> Option<Result<(GainMapMetadata, Option<usize>)>>;

    /// Decode the gain map JPEG from MPF secondary images.
    ///
    /// Returns `None` if no gain map is present.
    /// Returns `Some(Err(...))` if gain map is present but decoding fails.
    fn decode_gainmap(&self) -> Option<Result<GainMap>>;
}

impl UltraHdrExtras for DecodedExtras {
    fn is_ultrahdr(&self) -> bool {
        self.xmp()
            .map(|xmp: &str| xmp.contains("hdrgm:Version") || xmp.contains("hdrgm:GainMapMax"))
            .unwrap_or(false)
    }

    fn ultrahdr_metadata(&self) -> Option<Result<(GainMapMetadata, Option<usize>)>> {
        let xmp = self.xmp()?;
        Some(parse_xmp(xmp).map_err(ultrahdr_to_jpegli_error))
    }

    fn decode_gainmap(&self) -> Option<Result<GainMap>> {
        // Get the gain map JPEG from secondary images
        let gainmap_jpeg = self.gainmap()?;

        // Decode it
        Some(decode_gainmap_jpeg(gainmap_jpeg))
    }
}

/// Reconstruct HDR from decoded UltraHDR JPEG.
///
/// This takes the decoded SDR pixels and extras (containing gain map),
/// and produces an HDR image using the gain map reconstruction algorithm.
///
/// # Arguments
///
/// * `sdr_pixels` - Decoded SDR pixels from the base JPEG (RGB8 or RGBA8)
/// * `width` - Image width
/// * `height` - Image height
/// * `extras` - Decoded extras containing gain map and XMP metadata
/// * `display_boost` - Target display capability (1.0=SDR, 4.0=typical HDR)
/// * `output_format` - Desired HDR output format
/// * `stop` - Cooperative cancellation token
///
/// # Returns
///
/// HDR image in the requested format.
///
/// # Errors
///
/// Returns an error if:
/// - The image is not UltraHDR (no metadata or gain map)
/// - Metadata parsing fails
/// - Gain map decoding fails
/// - HDR reconstruction fails
pub fn reconstruct_hdr(
    sdr_pixels: &[u8],
    width: u32,
    height: u32,
    extras: &DecodedExtras,
    display_boost: f32,
    output_format: HdrOutputFormat,
    stop: impl Stop,
) -> Result<RawImage> {
    // Parse metadata
    let (metadata, _) = extras
        .ultrahdr_metadata()
        .ok_or_else(|| Error::decode_error("Not an UltraHDR image".to_string()))??;

    // Decode gain map
    let gainmap = extras
        .decode_gainmap()
        .ok_or_else(|| Error::decode_error("No gain map found".to_string()))??;

    // Create SDR RawImage from pixels
    let sdr = create_sdr_rawimage(sdr_pixels, width, height)?;

    // Apply gain map to reconstruct HDR
    apply_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        display_boost,
        output_format,
        stop,
    )
    .map_err(ultrahdr_to_jpegli_error)
}

/// Extract an adaptive tonemapper from an UltraHDR image.
///
/// This creates an [`AdaptiveTonemapper`] from the gain map metadata,
/// which can be used to reproduce the same tonemapping curve when
/// re-encoding edited HDR content.
///
/// # Example
///
/// ```rust,ignore
/// use jpegli::decoder::Decoder;
/// use jpegli::ultrahdr::{tonemapper_from_ultrahdr, encode_ultrahdr_with_tonemapper};
///
/// // Decode original UltraHDR
/// let decoded = Decoder::new().decode(&original_jpeg)?;
/// let extras = decoded.extras().unwrap();
///
/// // Extract tonemapper
/// let tonemapper = tonemapper_from_ultrahdr(extras)?;
///
/// // Edit the HDR...
/// let edited_hdr = edit_hdr(&original_hdr);
///
/// // Re-encode with same tonemapping
/// let new_jpeg = encode_ultrahdr_with_tonemapper(
///     &edited_hdr, &tonemapper, &gainmap_config, &encoder_config, 75.0, Unstoppable,
/// )?;
/// ```
pub fn tonemapper_from_ultrahdr(extras: &DecodedExtras) -> Result<AdaptiveTonemapper> {
    let (metadata, _) = extras
        .ultrahdr_metadata()
        .ok_or_else(|| Error::decode_error("Not an UltraHDR image".to_string()))??;

    Ok(AdaptiveTonemapper::from_gainmap(&metadata))
}

/// Full roundtrip: decode UltraHDR, reconstruct HDR, apply transform, re-encode.
///
/// This is a convenience function for editing workflows where you want to:
/// 1. Decode an existing UltraHDR JPEG
/// 2. Optionally modify the HDR content
/// 3. Re-encode as UltraHDR with the same tonemapping relationship
///
/// # Arguments
///
/// * `jpeg_data` - Original UltraHDR JPEG bytes
/// * `display_boost` - HDR reconstruction boost (4.0 typical)
/// * `transform` - Optional closure to transform the HDR image (None = passthrough)
/// * `gainmap_config` - Configuration for gain map computation
/// * `encoder_config` - jpegli encoder configuration
/// * `gainmap_quality` - JPEG quality for the gain map
/// * `stop` - Cooperative cancellation token
///
/// # Example
///
/// ```rust,ignore
/// use jpegli::ultrahdr::{reencode_ultrahdr, GainMapConfig, Unstoppable};
/// use jpegli::encoder::{EncoderConfig, ChromaSubsampling};
///
/// // Simple re-encode (no modification)
/// let new_jpeg = reencode_ultrahdr(
///     &original_jpeg,
///     4.0,
///     None::<fn(&mut UhdrRawImage)>,
///     &GainMapConfig::default(),
///     &EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter),
///     75.0,
///     Unstoppable,
/// )?;
///
/// // Re-encode with HDR adjustment
/// let new_jpeg = reencode_ultrahdr(
///     &original_jpeg,
///     4.0,
///     Some(|hdr: &mut UhdrRawImage| {
///         // Apply brightness boost, color grade, etc.
///     }),
///     &GainMapConfig::default(),
///     &EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter),
///     75.0,
///     Unstoppable,
/// )?;
/// ```
pub fn reencode_ultrahdr<F>(
    jpeg_data: &[u8],
    display_boost: f32,
    transform: Option<F>,
    gainmap_config: &ultrahdr_core::gainmap::GainMapConfig,
    encoder_config: &crate::encoder::EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>>
where
    F: FnOnce(&mut RawImage),
{
    use crate::ultrahdr::encode::encode_ultrahdr_with_tonemapper;

    // Decode
    let decoded = Decoder::new().decode(jpeg_data)?;
    let extras = decoded
        .extras()
        .ok_or_else(|| Error::decode_error("No extras preserved during decode".to_string()))?;

    // Extract tonemapper before reconstruction
    let tonemapper = tonemapper_from_ultrahdr(extras)?;

    // Reconstruct HDR
    let mut hdr = reconstruct_hdr(
        decoded.pixels(),
        decoded.width(),
        decoded.height(),
        extras,
        display_boost,
        HdrOutputFormat::LinearFloat,
        &stop,
    )?;
    stop.check()?;

    // Apply optional transform
    if let Some(f) = transform {
        f(&mut hdr);
    }

    // Re-encode with preserved tonemapping
    encode_ultrahdr_with_tonemapper(
        &hdr,
        &tonemapper,
        gainmap_config,
        encoder_config,
        gainmap_quality,
        stop,
    )
}

/// Decode a gain map JPEG to GainMap struct.
fn decode_gainmap_jpeg(jpeg_data: &[u8]) -> Result<GainMap> {
    let decoded = Decoder::new().decode(jpeg_data)?;

    let width = decoded.width();
    let height = decoded.height();
    let pixels = decoded.pixels().to_vec();

    // Determine if single-channel or multi-channel based on decoded format
    // The decoder typically outputs RGB, so we take the R channel for grayscale
    // or all channels for multi-channel
    let channels = if is_grayscale_content(&pixels) { 1 } else { 3 };

    let data = if channels == 1 {
        // Extract just the R (or first) channel
        pixels.chunks(3).map(|p| p[0]).collect()
    } else {
        pixels
    };

    Ok(GainMap {
        width,
        height,
        channels,
        data,
    })
}

/// Check if decoded RGB content is actually grayscale (R==G==B for all pixels).
fn is_grayscale_content(pixels: &[u8]) -> bool {
    pixels
        .chunks(3)
        .take(100) // Sample first 100 pixels
        .all(|p| p[0] == p[1] && p[1] == p[2])
}

/// Create an SDR RawImage from pixel bytes.
fn create_sdr_rawimage(pixels: &[u8], width: u32, height: u32) -> Result<RawImage> {
    let expected_rgb = (width * height * 3) as usize;
    let expected_rgba = (width * height * 4) as usize;

    let (format, data) = if pixels.len() == expected_rgba {
        (ultrahdr_core::PixelFormat::Rgba8, pixels.to_vec())
    } else if pixels.len() == expected_rgb {
        // Convert RGB to RGBA
        let mut rgba = Vec::with_capacity(expected_rgba);
        for chunk in pixels.chunks(3) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(chunk[2]);
            rgba.push(255);
        }
        (ultrahdr_core::PixelFormat::Rgba8, rgba)
    } else {
        return Err(Error::invalid_buffer_size(expected_rgb, pixels.len()));
    };

    RawImage::from_data(
        width,
        height,
        format,
        ultrahdr_core::ColorGamut::Bt709,
        ultrahdr_core::ColorTransfer::Srgb,
        data,
    )
    .map_err(ultrahdr_to_jpegli_error)
}

/// Convert ultrahdr_core::Error to jpegli Error.
fn ultrahdr_to_jpegli_error(e: ultrahdr_core::Error) -> Error {
    Error::decode_error(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::SegmentType;

    fn make_test_extras_with_xmp(xmp: &str) -> DecodedExtras {
        let mut extras = DecodedExtras::new();
        let xmp_data = format!("http://ns.adobe.com/xap/1.0/\0{}", xmp);
        extras.add_segment(0xE1, xmp_data.into_bytes(), SegmentType::Xmp);
        extras
    }

    #[test]
    fn test_is_ultrahdr_positive() {
        let extras = make_test_extras_with_xmp(
            r#"<x:xmpmeta><rdf:RDF><rdf:Description hdrgm:Version="1.0"/></rdf:RDF></x:xmpmeta>"#,
        );
        assert!(extras.is_ultrahdr());
    }

    #[test]
    fn test_is_ultrahdr_negative() {
        let extras = make_test_extras_with_xmp(
            r#"<x:xmpmeta><rdf:RDF><rdf:Description dc:creator="Test"/></rdf:RDF></x:xmpmeta>"#,
        );
        assert!(!extras.is_ultrahdr());
    }

    #[test]
    fn test_is_ultrahdr_no_xmp() {
        let extras = DecodedExtras::new();
        assert!(!extras.is_ultrahdr());
    }

    #[test]
    fn test_is_grayscale_content() {
        // Grayscale content
        let gray = vec![128, 128, 128, 64, 64, 64, 200, 200, 200];
        assert!(is_grayscale_content(&gray));

        // Color content
        let color = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        assert!(!is_grayscale_content(&color));
    }
}
