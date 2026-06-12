//! UltraHDR decoding workflow helpers.
//!
//! This module provides high-level functions for decoding UltraHDR JPEGs
//! and reconstructing HDR content.

use crate::container::xmp::parse_xmp;
use crate::decode::DecodedExtras;
use crate::decoder::Decoder;
use crate::error::{Error, Result};
use crate::types::PixelFormat as JpegPixelFormat;
use enough::Unstoppable;
use ultrahdr_core::{
    ColorPrimaries, ColorPrimaries as ColorGamut, GainMap, GainMapMetadata, PixelBuffer,
    PixelFormat, TransferFunction,
    color::tonemap::AdaptiveTonemapper,
    gainmap::{HdrOutputFormat, RowDecoder, apply_gainmap},
    pixel_buffer_from_vec,
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
        // First try primary XMP (legacy format: all metadata in primary).
        // Field shape changed in 0.5: gain ranges live on channel records now,
        // not on a flat `gain_map_max: [f32; 3]`.
        if let Some(xmp) = self.xmp()
            && let Ok((metadata, len)) = parse_xmp(xmp)
            && (metadata.channels.iter().any(|c| c.max != 0.0)
                || metadata.alternate_hdr_headroom != 0.0)
        {
            return Some(Ok((metadata, len)));
        }

        // Then try gain map JPEG's XMP (modern format: metadata in secondary)
        if let Some(gainmap_jpeg) = self.gainmap()
            && let Some(gm_xmp) = extract_xmp_from_jpeg(gainmap_jpeg)
        {
            return Some(parse_xmp(&gm_xmp).map_err(xmp_to_jpegli_error));
        }

        // Fall back to primary XMP even if values are all-default
        let xmp = self.xmp()?;
        Some(parse_xmp(xmp).map_err(xmp_to_jpegli_error))
    }

    fn decode_gainmap(&self) -> Option<Result<GainMap>> {
        // Get the gain map JPEG from secondary images
        let gainmap_jpeg = self.gainmap()?;

        // Channel form comes from the hdrgm/ISO metadata, not pixel
        // inspection (the ultrahdr-rs#27 model; zenjpeg#152).
        let single_channel = self
            .ultrahdr_metadata()
            .and_then(|m| m.ok())
            .map(|(m, _)| m.is_single_channel());
        Some(decode_gainmap_jpeg(gainmap_jpeg, single_channel))
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

// ===========================================================================
// One-call convenience entry points (libultrahdr parity surface)
// ===========================================================================

/// Decode an Ultra HDR JPEG to its SDR base image.
///
/// One-call equivalent of libultrahdr's `uhdr_decode` followed by
/// `uhdr_get_decoded_image` with `display_boost = 1.0`. Returns the SDR
/// base image as `Rgba8` with sRGB transfer — what every viewer can show
/// without an HDR-aware compositor. For HDR output, use
/// [`decode_ultrahdr_hdr`].
///
/// Works on any JPEG (Ultra HDR or not) since it returns the primary
/// rendition. Use `DecodeResult::extras().is_ultrahdr()` first if you
/// need to discriminate.
pub fn decode_ultrahdr(bytes: &[u8]) -> Result<PixelBuffer> {
    let decoded = Decoder::new().decode(bytes, Unstoppable)?;
    decode_result_to_pixel_buffer(decoded)
}

/// Decode an Ultra HDR JPEG to HDR pixels using its gain map.
///
/// One-call equivalent of libultrahdr's `uhdr_decode` →
/// `uhdr_dec_set_out_max_display_boost` → `uhdr_get_decoded_image` flow.
///
/// `display_boost` is the linear HDR capacity:
/// - `1.0` = SDR (gain map ignored)
/// - `2.0` = 2× headroom
/// - `4.0` = typical HDR display
/// - `8.0`+ = full reconstruction at PQ peak
///
/// `format` picks the output pixel layout. See [`HdrOutputFormat`] —
/// `LinearFloat`/`LinearF16` produce linear HDR (1.0 = SDR white),
/// `Srgb8` clips to SDR for fallback rendering.
///
/// Returns an error if the input isn't an Ultra HDR image (no gain map
/// or metadata).
pub fn decode_ultrahdr_hdr(
    bytes: &[u8],
    display_boost: f32,
    format: HdrOutputFormat,
) -> Result<PixelBuffer> {
    let mut decoded = Decoder::new().decode(bytes, Unstoppable)?;
    let extras = decoded
        .take_extras()
        .ok_or_else(|| Error::decode_error("decoded JPEG has no extras".to_string()))?;
    let (metadata, _) = extras
        .ultrahdr_metadata()
        .ok_or_else(|| Error::decode_error("not an Ultra HDR image".to_string()))??;
    let gainmap = extras
        .decode_gainmap()
        .ok_or_else(|| Error::decode_error("not an Ultra HDR image".to_string()))??;

    let sdr = decode_result_to_pixel_buffer(decoded)?;
    apply_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        display_boost,
        format,
        Unstoppable,
    )
    .map_err(ultrahdr_to_jpegli_error)
}

/// Convert a [`crate::decode::DecodeResult`] to a `PixelBuffer` with sRGB
/// transfer + Bt709 primaries. Maps the codec-internal [`JpegPixelFormat`]
/// to [`zenpixels::PixelFormat`]; expands 3-channel RGB to 4-channel Rgba8
/// when the codec returned packed RGB (the common case).
fn decode_result_to_pixel_buffer(decoded: crate::decode::DecodeResult) -> Result<PixelBuffer> {
    let width = decoded.width;
    let height = decoded.height;
    let format = decoded.format;
    let (pixels_u8, _pixels_f32, _w, _h, _f, _extras) = decoded.into_parts();
    let data = pixels_u8.ok_or_else(|| {
        Error::unsupported_feature("decode_ultrahdr: decoder produced no u8 pixels")
    })?;

    let (bytes, zp_format) = match format {
        JpegPixelFormat::Rgb => (rgb_to_rgba8(&data), PixelFormat::Rgba8),
        JpegPixelFormat::Rgba => (data, PixelFormat::Rgba8),
        JpegPixelFormat::Gray => (data, PixelFormat::Gray8),
        other => {
            return Err(Error::decode_error(format!(
                "decode_ultrahdr: codec returned {other:?}; expected Rgb / Rgba / Gray"
            )));
        }
    };
    pixel_buffer_from_vec(
        bytes,
        width,
        height,
        zp_format,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .map_err(ultrahdr_to_jpegli_error)
}

/// Pad packed RGB bytes to RGBA8 with `A=0xFF`. Done in-place via Vec growth.
fn rgb_to_rgba8(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks_exact(3) {
        out.extend_from_slice(px);
        out.push(0xFF);
    }
    out
}

/// Decode a gain map JPEG to GainMap struct (#152).
///
/// `single_channel` is the metadata's channel form (`None` when no metadata
/// parsed). Single (or unknown): decode with native Gray output — the exact
/// luma plane (RGB-then-take-R rounds ±1 differently and promotes 4:2:0
/// chroma noise on color-encoded maps). Multi-channel: decode RGB. The map
/// decodes in stored orientation; callers orient it to the base (#151).
fn decode_gainmap_jpeg(jpeg_data: &[u8], single_channel: Option<bool>) -> Result<GainMap> {
    if single_channel != Some(false) {
        // Some color encodings can't produce Gray output ("unsupported color
        // conversion"); those fall through to the RGB path below.
        if let Ok(decoded) = Decoder::new()
            .output_format(JpegPixelFormat::Gray)
            .auto_orient(false)
            .decode(jpeg_data, enough::Unstoppable)
            && let Some(px) = decoded.pixels_u8()
        {
            return Ok(GainMap {
                width: decoded.width(),
                height: decoded.height(),
                channels: 1,
                data: px.to_vec(),
            });
        }
    }

    let decoded = Decoder::new()
        .output_format(JpegPixelFormat::Rgb)
        .auto_orient(false)
        .decode(jpeg_data, enough::Unstoppable)?;
    let width = decoded.width();
    let height = decoded.height();
    let rgb = decoded
        .pixels_u8()
        .ok_or_else(|| Error::decode_error("gain-map decode produced no u8 pixels".into()))?
        .to_vec();
    let collapse = match single_channel {
        // Metadata says luma-only: collapse regardless of decode noise.
        Some(true) => true,
        // Metadata says per-channel: keep all three.
        Some(false) => false,
        // No metadata: collapse only when provably achromatic (full scan,
        // no sampling).
        None => rgb
            .chunks_exact(3)
            .all(|px| px[0] == px[1] && px[1] == px[2]),
    };
    let (data, channels) = if collapse {
        // BT.709 luma — the same weighting the Gray decode applies.
        (
            rgb.chunks_exact(3)
                .map(|px| {
                    (0.2126_f32 * f32::from(px[0])
                        + 0.7152 * f32::from(px[1])
                        + 0.0722 * f32::from(px[2]))
                    .clamp(0.0, 255.0) as u8
                })
                .collect(),
            1,
        )
    } else {
        (rgb, 3)
    };

    Ok(GainMap {
        width,
        height,
        channels,
        data,
    })
}

/// Extract XMP string from a JPEG's APP1 segment.
fn extract_xmp_from_jpeg(jpeg: &[u8]) -> Option<String> {
    let xmp_ns = b"http://ns.adobe.com/xap/1.0/\0";
    let idx = jpeg.windows(xmp_ns.len()).position(|w| w == xmp_ns)?;
    let xmp_start = idx + xmp_ns.len();
    // Find the APP1 marker to get the segment length
    // Walk backwards from idx to find FF E1
    let marker_pos = idx.checked_sub(4)?;
    if jpeg.get(marker_pos)? != &0xFF || jpeg.get(marker_pos + 1)? != &0xE1 {
        return None;
    }
    let length = u16::from_be_bytes([jpeg[marker_pos + 2], jpeg[marker_pos + 3]]) as usize;
    let xmp_end = marker_pos + 2 + length;
    let xmp_bytes = jpeg.get(xmp_start..xmp_end)?;
    String::from_utf8(xmp_bytes.to_vec()).ok()
}

/// Convert ultrahdr_core::Error to jpegli Error.
fn ultrahdr_to_jpegli_error(e: ultrahdr_core::Error) -> Error {
    Error::decode_error(e.to_string())
}

/// Convert XMP parse errors (from `crate::container::xmp::parse_xmp`) to jpegli Error.
fn xmp_to_jpegli_error(e: crate::container::xmp::XmpError) -> Error {
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

}
