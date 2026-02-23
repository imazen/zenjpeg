//! UltraHDR decoding workflow helpers.
//!
//! This module provides high-level functions for decoding UltraHDR JPEGs
//! and reconstructing HDR content.

use crate::decode::DecodedExtras;
use crate::decoder::Decoder;
use crate::error::{Error, Result};
use ultrahdr_core::{
    ColorGamut, GainMap, GainMapMetadata, color::tonemap::AdaptiveTonemapper, gainmap::RowDecoder,
    metadata::xmp::parse_xmp,
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

/// Create a streaming HDR reconstructor for row-by-row processing.
///
/// This is more memory-efficient than full-image reconstruction for large images,
/// as it processes rows in batches rather than loading the entire image.
///
/// The reconstructor accepts **linear f32 RGB** input and produces **linear f32 RGBA**
/// output. The caller must convert sRGB u8 decoder output to linear f32 before
/// calling `process_rows`.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `extras` - Decoded extras containing gain map and XMP metadata
/// * `display_boost` - Target display capability (1.0=SDR, 4.0=typical HDR)
///
/// # Returns
///
/// A [`RowDecoder`] that can process linear f32 SDR rows into linear f32 HDR rows.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::ultrahdr::create_hdr_reconstructor;
///
/// let mut reconstructor = create_hdr_reconstructor(
///     width, height, extras, 4.0,
/// )?;
///
/// // Process rows in batches (input must be linear f32 RGB)
/// for batch_start in (0..height).step_by(16) {
///     let batch_height = 16.min(height - batch_start);
///     let sdr_batch = &sdr_linear_f32[batch_start as usize * row_stride..];
///     let hdr_rows = reconstructor.process_rows(sdr_batch, batch_height as u32)?;
///     // hdr_rows is linear f32 RGBA
/// }
/// ```
pub fn create_hdr_reconstructor(
    width: u32,
    height: u32,
    extras: &DecodedExtras,
    display_boost: f32,
) -> Result<RowDecoder> {
    // Parse metadata
    let (metadata, _) = extras
        .ultrahdr_metadata()
        .ok_or_else(|| Error::decode_error("Not an UltraHDR image".to_string()))??;

    // Decode gain map
    let gainmap = extras
        .decode_gainmap()
        .ok_or_else(|| Error::decode_error("No gain map found".to_string()))??;

    // Create reconstructor (expects linear f32 RGB input, outputs linear f32 RGBA)
    RowDecoder::new(
        gainmap,
        metadata,
        width,
        height,
        display_boost,
        ColorGamut::Bt709,
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
/// use zenjpeg::decoder::Decoder;
/// use zenjpeg::ultrahdr::{tonemapper_from_ultrahdr, encode_ultrahdr_with_tonemapper};
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

/// Decode a gain map JPEG to GainMap struct.
fn decode_gainmap_jpeg(jpeg_data: &[u8]) -> Result<GainMap> {
    let decoded = Decoder::new().decode(jpeg_data, enough::Unstoppable)?;

    let width = decoded.width();
    let height = decoded.height();
    let pixels = decoded.pixels_u8().unwrap().to_vec();

    // Determine if single-channel or multi-channel based on decoded format
    // The decoder typically outputs RGB, so we take the R channel for grayscale
    // or all channels for multi-channel
    let channels = if is_grayscale_content(&pixels) { 1 } else { 3 };

    let data = if channels == 1 {
        // Extract just the R (or first) channel
        // Use chunks_exact to avoid panic on incomplete final chunk
        pixels.chunks_exact(3).map(|p| p[0]).collect()
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
    // Use chunks_exact to avoid incomplete final chunk
    pixels
        .chunks_exact(3)
        .take(100) // Sample first 100 pixels
        .all(|p| p[0] == p[1] && p[1] == p[2])
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
