//! Shared helpers converting zenjpeg header info to zencodec types.

use zencodec::{ImageFormat, ImageInfo};
use zenpixels::PixelDescriptor;

use super::decode::{select_decode_descriptor, will_auto_orient};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert JpegInfo to zc ImageInfo.
pub(super) fn to_image_info(info: &crate::decode::JpegInfo) -> ImageInfo {
    let mut img_info = ImageInfo::new(
        info.dimensions.width,
        info.dimensions.height,
        ImageFormat::Jpeg,
    )
    .with_bit_depth(info.precision)
    .with_channel_count(info.num_components);

    if let Some(ref icc) = info.icc_profile {
        img_info = img_info.with_icc_profile(icc.clone());
    }
    if let Some(ref exif) = info.exif {
        if let Some(orient) = crate::lossless::parse_exif_orientation(exif) {
            img_info = img_info
                .with_orientation(zencodec::Orientation::from_exif(orient).unwrap_or_default());
        }
        img_info = img_info.with_exif(exif.clone());
    }
    if let Some(ref xmp) = info.xmp {
        img_info = img_info.with_xmp(xmp.as_bytes().to_vec());
    }

    if let Some(ref jfif) = info.jfif
        && let Some(resolution) = jfif_to_resolution(jfif)
    {
        img_info = img_info.with_resolution(resolution);
    }

    img_info = img_info.with_progressive(matches!(
        info.mode,
        crate::types::JpegMode::Progressive | crate::types::JpegMode::ArithmeticProgressive
    ));

    img_info
}

/// Build a [`SourceColor`] from a JPEG header for descriptor derivation.
pub(super) fn source_color_from_header(
    info: &crate::decode::JpegInfo,
) -> zencodec::decode::SourceColor {
    let mut sc = zencodec::decode::SourceColor::default();
    if let Some(ref icc) = info.icc_profile {
        sc = sc.with_icc_profile(icc.clone());
    }
    sc
}

/// Derive the correct [`PixelDescriptor`] for decoded JPEG pixels.
///
/// Uses the shared zencodec utility to map source color metadata to a
/// descriptor that accurately reflects the pixel data's color space.
pub(super) fn decode_descriptor(
    preferred: &[PixelDescriptor],
    header: &crate::decode::JpegInfo,
    correct_color: Option<&crate::color::icc::TargetColorSpace>,
) -> PixelDescriptor {
    let base = select_decode_descriptor(preferred, header.num_components);
    let sc = source_color_from_header(header);
    let corrected_cicp =
        correct_color.map(|_| zenpixels::ColorProfileSource::Cicp(zenpixels::Cicp::SRGB));
    zencodec::helpers::descriptor_for_decoded_pixels_v2(
        base.pixel_format(),
        &sc,
        corrected_cicp.as_ref(),
    )
}

/// Populate [`ImageInfo`] metadata (ICC / EXIF + orientation / XMP / JFIF
/// resolution) from decoded JPEG extras. Shared by the normal decode path and
/// the Ultra HDR reconstruction path.
pub(super) fn populate_info_from_jpeg_extras(
    mut info: ImageInfo,
    extras: &crate::decode::DecodedExtras,
    orientation: zencodec::OrientationHint,
) -> ImageInfo {
    if let Some(icc) = extras.icc_profile() {
        info = info.with_icc_profile(icc.to_vec());
    }
    if let Some(exif) = extras.exif() {
        if let Some(orient) = crate::lossless::parse_exif_orientation(exif) {
            // If auto-orient was applied, report Identity; else source
            if will_auto_orient(orientation) {
                info = info.with_orientation(zencodec::Orientation::Identity);
            } else {
                info = info
                    .with_orientation(zencodec::Orientation::from_exif(orient).unwrap_or_default());
            }
        }
        info = info.with_exif(exif.to_vec());
    }
    if let Some(xmp) = extras.xmp() {
        info = info.with_xmp(xmp.as_bytes().to_vec());
    }
    // Populate resolution from JFIF APP0 density.
    if let Some(jfif) = extras.jfif()
        && let Some(resolution) = jfif_to_resolution(&jfif)
    {
        info = info.with_resolution(resolution);
    }
    // Ultra HDR container signal (cheap XMP check, no MPF decode) — adapter
    // parity with heic's `Supplements` population so callers can gate a
    // ReconstructHdr pass on the base decode's info alone.
    #[cfg(feature = "ultrahdr")]
    {
        use crate::ultrahdr::UltraHdrExtras;
        info.supplements.gain_map = extras.is_ultrahdr();
    }
    info
}

/// Convert JFIF density info to a zencodec [`Resolution`](zencodec::Resolution).
///
/// Returns `None` for aspect-ratio-only density (unit = 0) or zero densities.
pub(super) fn jfif_to_resolution(
    jfif: &crate::encode::extras::JfifInfo,
) -> Option<zencodec::Resolution> {
    use crate::encode::extras::DensityUnits;
    use zencodec::ResolutionUnit;

    if jfif.x_density == 0 || jfif.y_density == 0 {
        return None;
    }

    let unit = match jfif.density_units {
        DensityUnits::PixelsPerInch => ResolutionUnit::Inch,
        DensityUnits::PixelsPerCm => ResolutionUnit::Centimeter,
        DensityUnits::None => return None, // aspect ratio only, not real DPI
    };

    Some(zencodec::Resolution {
        x: jfif.x_density as f64,
        y: jfif.y_density as f64,
        unit,
    })
}
