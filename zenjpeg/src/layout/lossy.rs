//! Lossy path: decode → resize → encode.
//!
//! Uses streaming decode + streaming resize + streaming encode for bounded memory.
//! Builds a [`zenresize::StreamingResize`] from the local [`Plan`] — which owns
//! the `ResizeConfig` (including any `source_region` crop) and an `OrientOutput`
//! applied post-resize.

use alloc::vec;
use alloc::vec::Vec;

use enough::Stop;
use imgref::ImgRefMut;
use whereat::ResultAtExt;
use zenresize::StreamingResize;

use crate::decode::{ChromaUpsampling, DecodeConfig, JpegInfo};
use crate::encode::encoder_types::PixelLayout as EncPixelLayout;
use crate::encode::exif::Exif;
use crate::error::Result;

use super::LayoutConfig;
use super::plan::Plan;

/// Execute the lossy decode → resize → encode path.
///
/// Decodes the JPEG to RGB8 via scanline reader, resizes using zenresize's
/// streaming API for bounded memory, and re-encodes with the configured settings.
///
/// When `reset_orientation` is true, the EXIF orientation tag is set to 1
/// (normal) because the pixels have already been oriented by the pipeline.
pub(crate) fn execute_lossy(
    jpeg_data: &[u8],
    info: &JpegInfo,
    config: &LayoutConfig,
    plan: &Plan,
    reset_orientation: bool,
    force_baseline: bool,
    stop: &dyn Stop,
) -> Result<Vec<u8>> {
    if plan.is_identity {
        // No resize, no crop, no orient — just re-encode at source dimensions.
        let src_w = info.dimensions.width;
        let src_h = info.dimensions.height;
        return decode_reencode(
            jpeg_data,
            info,
            config,
            src_w,
            src_h,
            reset_orientation,
            force_baseline,
            stop,
        );
    }
    decode_resize_encode(
        jpeg_data,
        info,
        config,
        plan,
        reset_orientation,
        force_baseline,
        stop,
    )
}

/// Decode → resize → encode with streaming for bounded memory.
fn decode_resize_encode(
    jpeg_data: &[u8],
    info: &JpegInfo,
    config: &LayoutConfig,
    plan: &Plan,
    reset_orientation: bool,
    force_baseline: bool,
    stop: &dyn Stop,
) -> Result<Vec<u8>> {
    let src_w = info.dimensions.width;
    let out_w = plan.final_w;
    let out_h = plan.final_h;

    // Build streaming resizer from the planned ResizeConfig; apply post-resize orient.
    let batch = 8u32;
    let mut resizer =
        StreamingResize::with_batch_hint(&plan.resize, batch).with_orientation(plan.orient);

    // Build encoder with metadata from source.
    // force_baseline overrides progressive AFTER auto_optimize (which enables progressive).
    let encoder_config = if force_baseline {
        config.build_encoder_config().progressive(false)
    } else {
        config.build_encoder_config()
    };
    let mut request = encoder_config.request();
    request = attach_metadata(request, info, reset_orientation);
    request = request.stop(stop);

    let mut encoder = request.encode_from_bytes(out_w, out_h, EncPixelLayout::Rgb8Srgb)?;

    // Streaming pipeline: decode rows → push to resizer → pull output → push to encoder
    let decoder = {
        let mut dc = DecodeConfig::new();
        dc.chroma_upsampling = if config.fancy_upsampling {
            ChromaUpsampling::Triangle
        } else {
            ChromaUpsampling::NearestNeighbor
        };
        dc
    };
    let mut reader = decoder.scanline_reader(jpeg_data)?;

    let row_bytes = src_w as usize * 3;
    let batch = batch as usize;
    let mut buf = vec![0u8; row_bytes * batch];

    while !reader.is_finished() {
        stop.check()?;

        let img = ImgRefMut::new(&mut buf, src_w as usize * 3, batch);
        let rows_read = reader.read_rows_rgb8(img)?;
        if rows_read == 0 {
            break;
        }

        let available = resizer
            .push_rows(&buf[..row_bytes * rows_read], row_bytes, rows_read as u32)
            // `push_rows` returns `At<StreamingError>`; map the inner variant to our
            // `ErrorKind` with `map_err_at` so the resize-call trace is carried through
            // (the trailing `?` then wraps the resulting `At<ErrorKind>` into `Error`).
            .map_err_at(|inner| crate::error::ErrorKind::InternalError {
                reason: match inner {
                    zenresize::StreamingError::AlreadyFinished => "resize: push after finish",
                    zenresize::StreamingError::InputTooShort => "resize: input row too short",
                    zenresize::StreamingError::RingBufferOverflow => "resize: ring buffer overflow",
                    _ => "resize: unknown streaming error",
                },
            })?;
        drain_resizer(&mut resizer, available, &mut encoder, stop)?;
    }

    // Flush remaining rows from resizer
    let remaining = resizer.finish();
    drain_resizer(&mut resizer, remaining, &mut encoder, stop)?;

    encoder.finish()
}

/// Decode → transform → encode a secondary image (an Ultra HDR gain map)
/// with `plan`, the primary's plan scaled to it ([`Plan::for_secondary`]).
///
/// The map keeps its channel count (a one-component map stays grayscale),
/// its own metadata (ICC, EXIF, XMP) and its ISO 21496-1 records, which
/// carry its gain parameters. A plan that changes nothing returns the map's
/// bytes as they are.
pub(crate) fn transform_secondary(
    jpeg_data: &[u8],
    info: &JpegInfo,
    config: &LayoutConfig,
    plan: &Plan,
    stop: &dyn Stop,
) -> Result<Vec<u8>> {
    if plan.is_identity {
        return Ok(jpeg_data.to_vec());
    }
    let gray = info.num_components == 1;
    let (channels, layout) = if gray {
        (1usize, EncPixelLayout::Gray8Srgb)
    } else {
        (3usize, EncPixelLayout::Rgb8Srgb)
    };
    let src_w = info.dimensions.width;
    let batch = 8u32;
    let mut resizer =
        StreamingResize::with_batch_hint(&plan.resize, batch).with_orientation(plan.orient);

    let encoder_config = if gray {
        crate::encode::encoder_config::EncoderConfig::grayscale(config.quality)
            .progressive(config.progressive)
    } else {
        config.build_encoder_config()
    };
    let mut segments = crate::encode::EncoderSegments::new();
    for span in crate::container::iter(jpeg_data) {
        if matches!(span.kind, crate::container::MarkerKind::Sos) {
            break;
        }
        if matches!(span.kind, crate::container::MarkerKind::App(2))
            && span.payload.starts_with(zencodec::ISO_21496_1_URN)
        {
            segments = segments.add_raw(0xE2, span.payload.to_vec());
        }
    }
    let mut request = encoder_config.request().segments(segments);
    request = attach_metadata(request, info, false);
    request = request.stop(stop);
    let mut encoder = request.encode_from_bytes(plan.final_w, plan.final_h, layout)?;

    let mut reader = DecodeConfig::new().scanline_reader(jpeg_data)?;
    let row_bytes = src_w as usize * channels;
    let batch = batch as usize;
    let mut buf = vec![0u8; row_bytes * batch];
    while !reader.is_finished() {
        stop.check()?;
        let img = ImgRefMut::new(&mut buf, row_bytes, batch);
        let rows_read = if gray {
            reader.read_rows_gray8(img)?
        } else {
            reader.read_rows_rgb8(img)?
        };
        if rows_read == 0 {
            break;
        }
        let available = resizer
            .push_rows(&buf[..row_bytes * rows_read], row_bytes, rows_read as u32)
            .map_err_at(|_| crate::error::ErrorKind::InternalError {
                reason: "resize: gain-map rows rejected",
            })?;
        drain_resizer(&mut resizer, available, &mut encoder, stop)?;
    }
    let remaining = resizer.finish();
    drain_resizer(&mut resizer, remaining, &mut encoder, stop)?;
    encoder.finish()
}

/// Decode and re-encode without resize (for recompression or metadata update).
fn decode_reencode(
    jpeg_data: &[u8],
    info: &JpegInfo,
    config: &LayoutConfig,
    width: u32,
    height: u32,
    reset_orientation: bool,
    force_baseline: bool,
    stop: &dyn Stop,
) -> Result<Vec<u8>> {
    let encoder_config = if force_baseline {
        config.build_encoder_config().progressive(false)
    } else {
        config.build_encoder_config()
    };
    let mut request = encoder_config.request();
    request = attach_metadata(request, info, reset_orientation);
    request = request.stop(stop);

    let mut encoder = request.encode_from_bytes(width, height, EncPixelLayout::Rgb8Srgb)?;

    let decoder = {
        let mut dc = DecodeConfig::new();
        dc.chroma_upsampling = if config.fancy_upsampling {
            ChromaUpsampling::Triangle
        } else {
            ChromaUpsampling::NearestNeighbor
        };
        dc
    };
    let mut reader = decoder.scanline_reader(jpeg_data)?;

    let row_bytes = width as usize * 3;
    let batch = 8usize;
    let mut buf = vec![0u8; row_bytes * batch];

    while !reader.is_finished() {
        stop.check()?;

        let img = ImgRefMut::new(&mut buf, width as usize * 3, batch);
        let rows_read = reader.read_rows_rgb8(img)?;
        if rows_read == 0 {
            break;
        }

        encoder.push_packed(&buf[..row_bytes * rows_read], stop)?;
    }

    encoder.finish()
}

/// The `Exif\0\0` prefix length in APP1 EXIF segment data.
const EXIF_PREFIX_LEN: usize = 6;

/// Attach source metadata (ICC, EXIF, XMP) to the encode request.
///
/// When `reset_orientation` is true, clones the EXIF data and resets the
/// orientation tag to 1 (Normal) before attaching. This prevents double-rotation
/// when the pipeline has already oriented the pixels.
///
/// Note: `JpegInfo.exif` includes the `Exif\0\0` APP1 prefix, but `Exif::Raw()`
/// expects raw TIFF bytes without it. We strip the prefix before passing through.
fn attach_metadata<'a>(
    mut request: crate::encode::request::EncodeRequest<'a>,
    info: &'a JpegInfo,
    reset_orientation: bool,
) -> crate::encode::request::EncodeRequest<'a> {
    if let Some(ref icc) = info.icc_profile {
        request = request.icc_profile(icc);
    }
    if let Some(ref exif) = info.exif
        && exif.len() > EXIF_PREFIX_LEN
        && exif.starts_with(b"Exif\0\0")
    {
        if reset_orientation {
            let mut exif_copy = exif.clone();
            crate::lossless::set_exif_orientation(&mut exif_copy, 1);
            request = request.exif(Exif::Raw(exif_copy[EXIF_PREFIX_LEN..].to_vec()));
        } else {
            request = request.exif(Exif::Raw(exif[EXIF_PREFIX_LEN..].to_vec()));
        }
    }
    if let Some(ref xmp) = info.xmp {
        request = request.xmp(xmp.as_bytes());
    }
    request
}

/// Pull available output rows from the resizer and push them to the encoder.
fn drain_resizer(
    resizer: &mut StreamingResize,
    available: u32,
    encoder: &mut crate::encode::byte_encoders::BytesEncoder,
    stop: &dyn Stop,
) -> Result<()> {
    for _ in 0..available {
        if let Some(row) = resizer.next_output_row() {
            encoder.push_packed(row, stop)?;
        }
    }
    Ok(())
}
