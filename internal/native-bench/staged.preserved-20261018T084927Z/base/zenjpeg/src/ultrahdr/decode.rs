//! UltraHDR decoding workflow helpers.
//!
//! This module provides high-level functions for decoding UltraHDR JPEGs
//! and reconstructing HDR content.

use crate::container::xmp::parse_xmp;
use crate::container::{iso21496_from_gainmap_jpeg, iso21496_record};
use crate::decode::DecodedExtras;
use crate::decoder::Decoder;
use crate::error::{Error, Result};
use crate::types::PixelFormat as JpegPixelFormat;
use enough::Unstoppable;
use ultrahdr_core::{
    ColorPrimaries, GainMap, GainMapMetadata, PixelBuffer, PixelFormat, TransferFunction,
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
    /// (`hdrgm:Version` or `hdrgm:GainMapMax`), or an ISO 21496-1 APP2 exists.
    fn is_ultrahdr(&self) -> bool;

    /// Read gain-map parameters, preferring the secondary image's ISO 21496-1
    /// APP2 record over XMP. Invalid ISO falls back to XMP; without XMP its
    /// parse error is returned.
    ///
    /// Returns `None` if neither supported metadata form is present.
    /// Ordinary XMP without gain-map properties returns `None`.
    /// Returns `Some(Err(...))` if gain-map metadata is present but invalid.
    fn ultrahdr_metadata(&self) -> Option<Result<(GainMapMetadata, Option<usize>)>>;

    /// Decode the gain map JPEG from MPF secondary images.
    ///
    /// Returns `None` if no gain map is present.
    /// Returns `Some(Err(...))` if gain map is present but decoding fails.
    fn decode_gainmap(&self) -> Option<Result<GainMap>>;
}

impl UltraHdrExtras for DecodedExtras {
    fn is_ultrahdr(&self) -> bool {
        self.ultrahdr_metadata()
            .is_some_and(|result| result.is_ok())
            || self.xmp().is_some_and(|xmp| {
                xmp.contains("hdrgm:Version") || xmp.contains("hdrgm:GainMapMax")
            })
            || self
                .segments_by_marker(0xE2)
                .any(|seg| seg.data.starts_with(zencodec::ISO_21496_1_URN))
            || self
                .gainmap()
                .is_some_and(|jpeg| iso21496_record(jpeg).is_some())
    }

    fn ultrahdr_metadata(&self) -> Option<Result<(GainMapMetadata, Option<usize>)>> {
        metadata_with_xmp(self, true)
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

/// Resolve gain-map metadata without consulting XMP when extraction is denied.
pub(crate) fn metadata_with_xmp(
    extras: &DecodedExtras,
    allow_xmp: bool,
) -> Option<Result<(GainMapMetadata, Option<usize>)>> {
    let iso_error = match extras.gainmap().and_then(iso21496_from_gainmap_jpeg) {
        Some(Ok(metadata)) => return Some(Ok((metadata, None))),
        Some(Err(error)) => Some(error),
        None => None,
    };
    if allow_xmp {
        xmp_metadata(extras)
    } else {
        None
    }
    .or_else(|| iso_error.map(Err))
}

fn xmp_metadata(extras: &DecodedExtras) -> Option<Result<(GainMapMetadata, Option<usize>)>> {
    // First try primary XMP (legacy format: all metadata in primary).
    if let Some(found) = primary_xmp_gain_map(extras) {
        return Some(Ok(found));
    }

    // Then try gain map JPEG's XMP (modern format: metadata in secondary)
    if let Some(gainmap_jpeg) = extras.gainmap()
        && let Some(gm_xmp) = extract_xmp_from_jpeg(gainmap_jpeg)
        && let Some(metadata) = optional_gainmap_xmp(&gm_xmp)
    {
        return Some(metadata);
    }

    // Fall back to primary XMP even if values are all-default
    let xmp = extras.xmp()?;
    optional_gainmap_xmp(xmp)
}

/// The parser distinguishes absent gain-map properties from invalid ones.
/// Preserve that distinction through the optional metadata API.
fn optional_gainmap_xmp(xmp: &str) -> Option<Result<(GainMapMetadata, Option<usize>)>> {
    match parse_xmp(xmp) {
        Err(crate::container::xmp::XmpError::NotUltraHdr) => None,
        result => Some(result.map_err(xmp_to_zenjpeg_error)),
    }
}

/// The body of a gain-map JPEG's ISO 21496-1 APP2 record (URN included), if
/// it has one before its first scan.
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
        base_color(extras.icc_profile())?.0,
    )
    .map_err(ultrahdr_to_zenjpeg_error)
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
/// base image with its resolved primaries and transfer function. Unknown ICC
/// profiles require a color-managed decode and return an error here. For HDR output, use
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
    let decoded = Decoder::new().decode(bytes, Unstoppable)?;
    let extras = decoded
        .extras()
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
    .map_err(ultrahdr_to_zenjpeg_error)
}

/// Convert a [`crate::decode::DecodeResult`] to a `PixelBuffer`, preserving
/// the decoded primaries and transfer. Maps the codec-internal [`JpegPixelFormat`]
/// to [`zenpixels::PixelFormat`]; expands 3-channel RGB to 4-channel Rgba8
/// when the codec returned packed RGB (the common case).
fn decode_result_to_pixel_buffer(decoded: crate::decode::DecodeResult) -> Result<PixelBuffer> {
    let (primaries, transfer) = base_color(decoded.extras().and_then(DecodedExtras::icc_profile))?;
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
    pixel_buffer_from_vec(bytes, width, height, zp_format, primaries, transfer)
        .map_err(ultrahdr_to_zenjpeg_error)
}

// The decoder inverts XYB to sRGB. For other ICC profiles, resolve only
// color spaces the shared resolver can represent without a CMS.
pub(crate) fn base_color(icc: Option<&[u8]>) -> Result<(ColorPrimaries, TransferFunction)> {
    let Some(icc) = icc else {
        return Ok((ColorPrimaries::Bt709, TransferFunction::Srgb));
    };
    if crate::color::icc::is_xyb_profile(icc) {
        return Ok((ColorPrimaries::Bt709, TransferFunction::Srgb));
    }
    zenpixels::ColorProfileSource::Icc(icc)
        .resolve()
        .ok_or_else(|| {
            Error::unsupported_feature("Ultra HDR base ICC requires color-managed conversion")
        })
}

/// Pad packed RGB bytes to RGBA8 with `A=0xFF`. Done in-place via Vec growth.
fn rgb_to_rgba8(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.as_chunks::<3>().0 {
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
            .as_chunks::<3>()
            .0
            .iter()
            .all(|px| px[0] == px[1] && px[1] == px[2]),
    };
    let (data, channels) = if collapse {
        // BT.709 luma — the same weighting the Gray decode applies.
        (
            rgb.as_chunks::<3>()
                .0
                .iter()
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
pub(crate) fn extract_xmp_from_jpeg(jpeg: &[u8]) -> Option<String> {
    let segment = xmp_segment_range(jpeg)?;
    let xmp_bytes = jpeg.get(segment.start + 4 + XMP_NS.len()..segment.end)?;
    String::from_utf8(xmp_bytes.to_vec()).ok()
}

const XMP_NS: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

/// The APP1 segment (marker included) that [`extract_xmp_from_jpeg`] reads:
/// the one around the first occurrence of the XMP namespace, when `FF E1`
/// and a length that covers the namespace precede it.
pub(crate) fn xmp_segment_range(jpeg: &[u8]) -> Option<core::ops::Range<usize>> {
    let idx = jpeg.windows(XMP_NS.len()).position(|w| w == XMP_NS)?;
    let xmp_start = idx + XMP_NS.len();
    // Find the APP1 marker to get the segment length
    // Walk backwards from idx to find FF E1
    let marker_pos = idx.checked_sub(4)?;
    if jpeg.get(marker_pos)? != &0xFF || jpeg.get(marker_pos + 1)? != &0xE1 {
        return None;
    }
    let length = u16::from_be_bytes([jpeg[marker_pos + 2], jpeg[marker_pos + 3]]) as usize;
    let xmp_end = marker_pos + 2 + length;
    jpeg.get(xmp_start..xmp_end)?;
    Some(marker_pos..xmp_end)
}

/// XMP fallback branch: gain-map parameters from the
/// primary XMP, when it carries non-default ones.
pub(crate) fn primary_xmp_gain_map(
    extras: &DecodedExtras,
) -> Option<(GainMapMetadata, Option<usize>)> {
    // Field shape changed in 0.5: gain ranges live on channel records now,
    // not on a flat `gain_map_max: [f32; 3]`.
    let (metadata, len) = parse_xmp(extras.xmp()?).ok()?;
    (metadata.channels.iter().any(|c| c.max != 0.0) || metadata.alternate_hdr_headroom != 0.0)
        .then_some((metadata, len))
}

/// Convert an ultrahdr_core error to a zenjpeg Error.
///
/// Generic over `Display` so it accepts both bare `ultrahdr_core::Error` and
/// the `At<Error>` location wrapper (ultrahdr-core #31) — zenjpeg stays
/// compatible across the ultrahdr-core error-type change without pinning.
fn ultrahdr_to_zenjpeg_error(e: impl core::fmt::Display) -> Error {
    Error::decode_error(e.to_string())
}

/// Convert XMP parse errors (from `crate::container::xmp::parse_xmp`) to a zenjpeg Error.
fn xmp_to_zenjpeg_error(e: crate::container::xmp::XmpError) -> Error {
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
    fn denied_xmp_is_not_used_for_gainmap_parameters() {
        let xmp = crate::container::xmp::generate_xmp(&metadata(2.0, 3.0), 12);
        let extras = make_test_extras_with_xmp(&xmp);
        assert!(metadata_with_xmp(&extras, true).unwrap().is_ok());
        assert!(metadata_with_xmp(&extras, false).is_none());
        let secondary = {
            let mut jpeg = vec![0xff, 0xd8];
            jpeg.extend(crate::container::xmp::create_xmp_app1_marker(&xmp));
            jpeg.extend([0xff, 0xd9]);
            jpeg
        };
        let mut extras = DecodedExtras::new();
        extras.add_secondary_image(1, crate::decode::MpfImageType::Undefined, secondary);
        assert!(metadata_with_xmp(&extras, true).unwrap().is_ok());
        assert!(metadata_with_xmp(&extras, false).is_none());
    }

    #[test]
    fn test_is_ultrahdr_no_xmp() {
        let extras = DecodedExtras::new();
        assert!(!extras.is_ultrahdr());
    }
    fn metadata(max: f64, headroom: f64) -> GainMapMetadata {
        let mut m = GainMapMetadata::default();
        for c in &mut m.channels {
            c.min = 0.0;
            c.max = max;
            c.gamma = 1.0;
            c.base_offset = 1.0 / 64.0;
            c.alternate_offset = 1.0 / 64.0;
        }
        m.base_hdr_headroom = 0.0;
        m.alternate_hdr_headroom = headroom;
        m
    }

    /// A stand-in gain-map image: SOI, one APP2 with `body`, EOI. The
    /// metadata path reads its header only.
    fn gainmap_jpeg_with_app2(body: &[u8]) -> Vec<u8> {
        let len = u16::try_from(body.len() + 2).unwrap().to_be_bytes();
        [&[0xFF, 0xD8, 0xFF, 0xE2], &len[..], body, &[0xFF, 0xD9]].concat()
    }

    fn iso_body(m: &GainMapMetadata) -> Vec<u8> {
        zencodec::gainmap::serialize_iso21496_fmt(
            m,
            ultrahdr_core::Iso21496Format::JpegApp2BodyWithUrn,
        )
    }

    /// What the ISO 21496-1 record of `m` decodes to (its fractions round).
    fn iso_round_trip(m: &GainMapMetadata) -> GainMapMetadata {
        zencodec::gainmap::parse_iso21496_fmt(
            &iso_body(m),
            ultrahdr_core::Iso21496Format::JpegApp2BodyWithUrn,
        )
        .unwrap()
    }

    fn extras_with(
        primary_xmp: Option<&GainMapMetadata>,
        gainmap: Option<Vec<u8>>,
    ) -> DecodedExtras {
        let mut extras = DecodedExtras::new();
        if let Some(m) = primary_xmp {
            let xmp = crate::container::xmp::generate_gainmap_xmp(m);
            extras.add_segment(
                0xE1,
                [&b"http://ns.adobe.com/xap/1.0/\0"[..], xmp.as_bytes()].concat(),
                SegmentType::Xmp,
            );
        }
        if let Some(jpeg) = gainmap {
            extras.add_secondary_image(1, crate::decode::MpfImageType::Undefined, jpeg);
        }
        extras
    }

    #[test]
    fn iso_only_gain_map_metadata_is_read() {
        // zenjpeg#216: no XMP at all, the ISO 21496-1 record in the gain-map
        // image.
        let iso = metadata(2.0, 2.0);
        let extras = extras_with(None, Some(gainmap_jpeg_with_app2(&iso_body(&iso))));
        assert!(extras.is_ultrahdr());
        let (got, len) = extras.ultrahdr_metadata().unwrap().unwrap();
        assert_eq!(got, iso_round_trip(&iso));
        assert_eq!(len, None);
    }

    #[test]
    fn iso_metadata_wins_over_disagreeing_xmp() {
        // Ultra HDR v1.1: prefer ISO 21496-1 when both are present.
        let xmp = metadata(1.0, 1.0);
        let iso = metadata(3.0, 3.0);
        let extras = extras_with(Some(&xmp), Some(gainmap_jpeg_with_app2(&iso_body(&iso))));
        let (got, _) = extras.ultrahdr_metadata().unwrap().unwrap();
        assert_eq!(got, iso_round_trip(&iso));
        assert_ne!(got.alternate_hdr_headroom, xmp.alternate_hdr_headroom);
    }

    #[test]
    fn malformed_iso_record_falls_back_to_xmp_or_is_the_error() {
        // A version-only record (what the primary image carries) where the
        // gain-map image should hold the full metadata.
        let truncated = [zencodec::ISO_21496_1_URN, &[0u8, 0, 0, 0][..]].concat();
        let xmp = metadata(2.0, 2.0);
        let extras = extras_with(Some(&xmp), Some(gainmap_jpeg_with_app2(&truncated)));
        let (got, _) = extras.ultrahdr_metadata().unwrap().unwrap();
        assert_eq!(got.alternate_hdr_headroom, xmp.alternate_hdr_headroom);
        let extras = extras_with(None, Some(gainmap_jpeg_with_app2(&truncated)));
        assert!(matches!(extras.ultrahdr_metadata(), Some(Err(_))));
        // Its presence still marks the file as a gain-map image.
        assert!(extras.is_ultrahdr());
    }

    #[test]
    fn xmp_only_gain_map_metadata_is_unchanged() {
        let xmp = metadata(2.0, 2.0);
        let extras = extras_with(Some(&xmp), Some(gainmap_jpeg_with_app2(b"not iso")));
        let (got, _) = extras.ultrahdr_metadata().unwrap().unwrap();
        assert_eq!(got.alternate_hdr_headroom, xmp.alternate_hdr_headroom);
        assert_eq!(got.channels[0].max, xmp.channels[0].max);
    }
}
