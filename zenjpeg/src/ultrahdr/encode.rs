//! UltraHDR encoding workflow helpers.
//!
//! Provides high-level functions for encoding UltraHDR JPEGs from HDR
//! source images. Inputs are [`zenpixels::PixelBuffer`] (re-exported here
//! as [`super::UhdrRawImage`] for back-compat).

use crate::container::xmp::{create_xmp_app1_marker, generate_gainmap_xmp, generate_primary_xmp};
use crate::encode::extras::EncoderSegments;
use crate::encoder::{EncoderConfig, PixelLayout};
use crate::error::{Error, Result};
use enough::Stop;
use ultrahdr_core::{
    ColorPrimaries, GainMap, GainMapEncodingFormat, GainMapMetadata, PixelBuffer, PixelFormat,
    TransferFunction,
    color::tonemap::{AdaptiveTonemapper, ToneMapConfig, tonemap_image_to_srgb8},
    gainmap::{GainMapConfig, RowEncoder, compute_gainmap},
    pixel_buffer_from_vec,
};
use zencodec::Iso21496Format;
use zencodec::gainmap::{ISO_21496_1_PRIMARY_APP2_BODY, serialize_iso21496_fmt};

/// Encode an HDR image as UltraHDR JPEG.
///
/// Performs the full UltraHDR encoding workflow:
/// 1. Tonemap HDR to SDR via `ultrahdr_core::color::tonemap::tonemap_image_to_srgb8`
/// 2. Compute gain map from HDR/SDR pair
/// 3. Encode SDR base image with jpegli
/// 4. Encode gain map as grayscale JPEG
/// 5. Generate XMP metadata
/// 6. Assemble final UltraHDR JPEG with MPF structure
///
/// `tonemap_config` is currently a no-op placeholder — `tonemap_image_to_srgb8`
/// uses `ToneMapConfig::default()` internally. Issue #71 will replace this
/// signature with one that takes `&dyn LumaToneMap` for full curve choice.
pub fn encode_ultrahdr(
    hdr: &PixelBuffer,
    gainmap_config: &GainMapConfig,
    _tonemap_config: &ToneMapConfig,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    let sdr = tonemap_to_pixel_buffer(hdr, ColorPrimaries::Bt709)?;
    stop.check()?;

    let (gainmap, metadata) =
        compute_gainmap(hdr, &sdr, gainmap_config, &stop).map_err(ultrahdr_to_jpegli_error)?;
    stop.check()?;

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
/// HDR→SDR relationship. The adaptive tonemapper learns the curve from an
/// existing pair and reproduces it for modified content.
pub fn encode_ultrahdr_with_tonemapper(
    hdr: &PixelBuffer,
    tonemapper: &AdaptiveTonemapper,
    gainmap_config: &GainMapConfig,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    let sdr = tonemapper.apply(hdr).map_err(ultrahdr_to_jpegli_error)?;
    stop.check()?;

    let (gainmap, metadata) =
        compute_gainmap(hdr, &sdr, gainmap_config, &stop).map_err(ultrahdr_to_jpegli_error)?;
    stop.check()?;

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
/// More memory-efficient than the full-image [`compute_gainmap`] for large
/// images. Inputs are linear f32 RGB for both HDR and SDR.
pub fn create_gainmap_computer(
    width: u32,
    height: u32,
    config: &GainMapConfig,
    hdr_gamut: ColorPrimaries,
) -> Result<RowEncoder> {
    RowEncoder::new(
        width,
        height,
        config.clone(),
        hdr_gamut,
        ColorPrimaries::Bt709,
    )
    .map_err(ultrahdr_to_jpegli_error)
}

/// Encode SDR image with pre-computed gain map.
///
/// Uses [`GainMapEncodingFormat::Both`] for maximum compatibility.
pub fn encode_with_gainmap(
    sdr: &PixelBuffer,
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    encode_with_gainmap_format(
        sdr,
        gainmap,
        metadata,
        encoder_config,
        gainmap_quality,
        GainMapEncodingFormat::Both,
        stop,
    )
}

/// Encode SDR image with pre-computed gain map and metadata format control.
pub fn encode_with_gainmap_format(
    sdr: &PixelBuffer,
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    metadata_format: GainMapEncodingFormat,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    let gainmap_jpeg = encode_gainmap_jpeg(gainmap, gainmap_quality, &stop)?;
    stop.check()?;

    let metadata_markers = build_gainmap_metadata_markers(metadata, metadata_format);
    let mut gainmap_final = gainmap_jpeg;
    for marker in metadata_markers.iter().rev() {
        gainmap_final = inject_marker_after_soi(&gainmap_final, marker)?;
    }

    let primary_xmp = generate_primary_xmp(gainmap_final.len());

    let mut segments = EncoderSegments::new().set_xmp(&primary_xmp).add_mpf_image(
        gainmap_final,
        crate::encode::extras::MpfImageType::Undefined,
    );

    let include_iso = matches!(
        metadata_format,
        GainMapEncodingFormat::Iso21496 | GainMapEncodingFormat::Both
    );
    if include_iso {
        segments = segments.add_raw(0xE2, ISO_21496_1_PRIMARY_APP2_BODY.to_vec());
    }

    encode_sdr_base(sdr, encoder_config, segments, stop)
}

/// Tonemap HDR to a Bt709/Srgb `PixelBuffer`. Centralizes the conversion to
/// the pattern ultrahdr-rs uses (`tonemap_image_to_srgb8` + `pixel_buffer_from_vec`).
fn tonemap_to_pixel_buffer(hdr: &PixelBuffer, target_gamut: ColorPrimaries) -> Result<PixelBuffer> {
    let bytes = tonemap_image_to_srgb8(hdr, target_gamut).map_err(ultrahdr_to_jpegli_error)?;
    pixel_buffer_from_vec(
        bytes,
        hdr.width(),
        hdr.height(),
        PixelFormat::Rgba8,
        target_gamut,
        TransferFunction::Srgb,
    )
    .map_err(ultrahdr_to_jpegli_error)
}

/// Build the gain-map secondary's metadata APP markers (XMP APP1 and/or
/// ISO 21496-1 APP2) in canonical order. Mirrors ultrahdr-rs's local helper
/// since `ultrahdr_core::metadata::xmp::build_gainmap_metadata_markers` was
/// retired in 0.5.
fn build_gainmap_metadata_markers(
    metadata: &GainMapMetadata,
    format: GainMapEncodingFormat,
) -> Vec<Vec<u8>> {
    let mut markers = Vec::with_capacity(2);
    if matches!(
        format,
        GainMapEncodingFormat::Xmp | GainMapEncodingFormat::Both
    ) {
        let xmp = generate_gainmap_xmp(metadata);
        markers.push(create_xmp_app1_marker(&xmp));
    }
    if matches!(
        format,
        GainMapEncodingFormat::Iso21496 | GainMapEncodingFormat::Both
    ) {
        let iso_body = serialize_iso21496_fmt(metadata, Iso21496Format::JpegApp2BodyWithUrn);
        markers.push(wrap_app2(&iso_body));
    }
    markers
}

/// Wrap a body in a JPEG APP2 marker (`FF E2` + big-endian length-with-self
/// + body). Mirrors ultrahdr-rs's local `wrap_app2`.
fn wrap_app2(body: &[u8]) -> Vec<u8> {
    let total_length = 2 + body.len();
    let mut marker = Vec::with_capacity(4 + body.len());
    marker.extend_from_slice(&[
        0xFF,
        0xE2,
        ((total_length >> 8) & 0xFF) as u8,
        (total_length & 0xFF) as u8,
    ]);
    marker.extend_from_slice(body);
    marker
}

/// Inject a complete JPEG marker segment after SOI.
fn inject_marker_after_soi(jpeg: &[u8], marker: &[u8]) -> Result<Vec<u8>> {
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return Err(Error::unsupported_feature("gain map JPEG missing SOI"));
    }
    let mut result = Vec::with_capacity(jpeg.len() + marker.len());
    result.extend_from_slice(&jpeg[..2]); // SOI
    result.extend_from_slice(marker); // marker segment
    result.extend_from_slice(&jpeg[2..]); // rest of JPEG
    Ok(result)
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

/// Encode the SDR base image. The SDR `PixelBuffer` must be `Rgba8` or `Rgb8`.
fn encode_sdr_base(
    sdr: &PixelBuffer,
    config: &EncoderConfig,
    segments: EncoderSegments,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    let format = sdr.descriptor().pixel_format();
    let layout = match format {
        PixelFormat::Rgba8 => PixelLayout::Rgba8Srgb,
        PixelFormat::Rgb8 => PixelLayout::Rgb8Srgb,
        _ => {
            return Err(Error::unsupported_feature(
                "SDR PixelBuffer must be Rgba8 or Rgb8 for UltraHDR encoding",
            ));
        }
    };

    let config_with_segments = config.clone().with_segments(segments);
    let mut encoder = config_with_segments.encode_from_bytes(sdr.width(), sdr.height(), layout)?;

    encoder.push_packed(sdr.as_slice().as_strided_bytes(), stop)?;
    encoder.finish()
}

/// Convert ultrahdr_core::Error to jpegli Error.
fn ultrahdr_to_jpegli_error(e: ultrahdr_core::Error) -> Error {
    Error::decode_error(e.to_string())
}
