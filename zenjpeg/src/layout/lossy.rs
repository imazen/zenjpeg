//! Lossy path: decode → resize → encode.
//!
//! Uses streaming decode + streaming resize + streaming encode for bounded memory.
//! Uses [`zenresize::streaming_from_plan()`] to build the streaming resizer from a
//! [`zenlayout::LayoutPlan`], automatically handling crop, resize, and canvas padding.

use alloc::vec;
use alloc::vec::Vec;

use enough::Stop;
use imgref::ImgRefMut;
use zenresize::{PixelDescriptor, StreamingResize};

use crate::decode::{DecodeConfig, JpegInfo};
use crate::encode::encoder_types::PixelLayout as EncPixelLayout;
use crate::encode::exif::Exif;
use crate::error::Result;

use super::LayoutConfig;

/// Execute the lossy decode → resize → encode path.
///
/// Decodes the JPEG to RGB8 via scanline reader, resizes using zenresize's
/// streaming API for bounded memory, and re-encodes with the configured settings.
///
/// Uses `streaming_from_plan()` to build the streaming resizer from the
/// layout plan, automatically handling crop, resize, and canvas padding.
///
/// When `reset_orientation` is true, the EXIF orientation tag is set to 1
/// (normal) because the pixels have already been oriented by the pipeline.
pub(crate) fn execute_lossy(
    jpeg_data: &[u8],
    info: &JpegInfo,
    config: &LayoutConfig,
    plan: &zenlayout::LayoutPlan,
    reset_orientation: bool,
    force_baseline: bool,
    stop: &dyn Stop,
) -> Result<Vec<u8>> {
    let src_w = info.dimensions.width;
    let src_h = info.dimensions.height;
    let target_w = plan.canvas.width;
    let target_h = plan.canvas.height;

    let needs_resize =
        !plan.resize_is_identity || plan.trim.is_some() || target_w != src_w || target_h != src_h;

    if needs_resize {
        decode_resize_encode(
            jpeg_data,
            info,
            config,
            plan,
            reset_orientation,
            force_baseline,
            stop,
        )
    } else {
        decode_reencode(
            jpeg_data,
            info,
            config,
            src_w,
            src_h,
            reset_orientation,
            force_baseline,
            stop,
        )
    }
}

/// Decode → resize → encode with streaming for bounded memory.
///
/// Uses `config_from_plan()` to build the resize config from the layout plan,
/// which handles crop (trim), resize, and canvas padding automatically.
fn decode_resize_encode(
    jpeg_data: &[u8],
    info: &JpegInfo,
    config: &LayoutConfig,
    plan: &zenlayout::LayoutPlan,
    reset_orientation: bool,
    force_baseline: bool,
    stop: &dyn Stop,
) -> Result<Vec<u8>> {
    let src_w = info.dimensions.width;
    let src_h = info.dimensions.height;
    let out_w = plan.canvas.width;
    let out_h = plan.canvas.height;

    // Build streaming resizer from layout plan (handles crop, resize, pad, orient).
    let batch = 8u32;
    let mut resizer = zenresize::streaming_from_plan_batched(
        src_w,
        src_h,
        plan,
        PixelDescriptor::RGB8_SRGB,
        config.filter,
        batch,
    );

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
    let decoder = DecodeConfig::new().fancy_upsampling(config.fancy_upsampling);
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
            .map_err(|e| {
                crate::error::Error::new(crate::error::ErrorKind::InternalError {
                    reason: match e.error() {
                        zenresize::StreamingError::AlreadyFinished => "resize: push after finish",
                        zenresize::StreamingError::InputTooShort => "resize: input row too short",
                        zenresize::StreamingError::RingBufferOverflow => {
                            "resize: ring buffer overflow"
                        }
                        _ => "resize: unknown streaming error",
                    },
                })
            })?;
        drain_resizer(&mut resizer, available, &mut encoder, stop)?;
    }

    // Flush remaining rows from resizer
    let remaining = resizer.finish();
    drain_resizer(&mut resizer, remaining, &mut encoder, stop)?;

    encoder.finish()
}

/// Simple decode → resize → encode without a layout plan.
///
/// Used for gain map proportional resize where no crop/pad/orient is needed.
pub(crate) fn resize_simple(
    jpeg_data: &[u8],
    info: &JpegInfo,
    config: &LayoutConfig,
    dst_w: u32,
    dst_h: u32,
    stop: &dyn Stop,
) -> Result<Vec<u8>> {
    let src_w = info.dimensions.width;
    let src_h = info.dimensions.height;

    if dst_w == src_w && dst_h == src_h {
        return decode_reencode(jpeg_data, info, config, src_w, src_h, false, false, stop);
    }

    let resize_config = zenresize::ResizeConfig::builder(src_w, src_h, dst_w, dst_h)
        .filter(config.filter)
        .format(PixelDescriptor::RGB8_SRGB)
        .linear()
        .build();

    let batch = 8u32;
    let mut resizer = StreamingResize::with_batch_hint(&resize_config, batch);

    let encoder_config = config.build_encoder_config();
    let mut request = encoder_config.request();
    request = attach_metadata(request, info, false);
    request = request.stop(stop);

    let mut encoder = request.encode_from_bytes(dst_w, dst_h, EncPixelLayout::Rgb8Srgb)?;

    let decoder = DecodeConfig::new().fancy_upsampling(config.fancy_upsampling);
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
            .map_err(|e| {
                crate::error::Error::new(crate::error::ErrorKind::InternalError {
                    reason: match e.error() {
                        zenresize::StreamingError::AlreadyFinished => "resize: push after finish",
                        zenresize::StreamingError::InputTooShort => "resize: input row too short",
                        zenresize::StreamingError::RingBufferOverflow => {
                            "resize: ring buffer overflow"
                        }
                        _ => "resize: unknown streaming error",
                    },
                })
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

    let decoder = DecodeConfig::new().fancy_upsampling(config.fancy_upsampling);
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
            // Strip the Exif\0\0 prefix — Exif::Raw expects raw TIFF bytes
            request = request.exif(Exif::Raw(exif_copy[EXIF_PREFIX_LEN..].to_vec()));
        } else {
            // Strip the Exif\0\0 prefix — Exif::Raw expects raw TIFF bytes
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
