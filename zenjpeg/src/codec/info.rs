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
        // The zencodec EXIF blob is the TIFF structure (module docs).
        img_info = img_info.with_exif(crate::encode::exif::strip_app1_exif_header(exif).to_vec());
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

/// Populate known JPEG auxiliary scope after full primary framing validation.
/// The existing container probe detects all top-level JPEG boundaries, MPF and
/// supported gain-map signals. SCAN_COUNT keeps discovery beyond the first SOS.
/// Recognized auxiliary content stays Unknown until its parameters are parsed;
/// it is never relabeled Absent just because pixel reconstruction was disabled.
pub(super) fn populate_complete_container_scope(data: &[u8], info: &mut ImageInfo) {
    use crate::container::{Wants, probe};
    let probe = probe(
        data,
        Wants::ULTRAHDR_DETECT | Wants::IMAGE_RANGES | Wants::SCAN_COUNT,
    );
    populate_probe_scope(&probe, info);
}

fn populate_probe_scope(probe: &crate::container::ContainerProbe, info: &mut ImageInfo) {
    use crate::container::GainMapPresence;
    let multiple = probe.image_ranges().len() > 1;
    let gain_map = probe.gainmap_presence() != GainMapPresence::None;
    let auxiliary = multiple || probe.mpf().is_some();
    info.supplements.gain_map |= gain_map;
    info.supplements.auxiliary |= auxiliary;
    if multiple {
        info.sequence = zencodec::ImageSequence::Multi {
            image_count: (probe.image_ranges().len() < 8)
                .then_some(probe.image_ranges().len() as u32),
            random_access: true,
        };
    }
    if !probe.truncated() && !gain_map && !auxiliary && probe.image_ranges().len() == 1 {
        info.gain_map = zencodec::GainMapPresence::Absent;
    }
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

/// Whether reads for `descriptor` need the strip's planes (gray and f32
/// reads of YCbCr frames), which the wave-only scanline reader does not keep.
pub(super) fn needs_plane_reads(descriptor: PixelDescriptor) -> bool {
    use zenpixels::{ChannelLayout, ChannelType};
    descriptor.channel_type() == ChannelType::F32 || descriptor.layout() == ChannelLayout::Gray
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
    // XYB is inverted to sRGB by the decoder even without optional ICC
    // correction. The source ICC describes encoded XYB, not the returned RGB.
    let corrected_cicp = (header.is_xyb || correct_color.is_some())
        .then_some(zenpixels::ColorProfileSource::Cicp(zenpixels::Cicp::SRGB));
    let decoded = zencodec::helpers::descriptor_for_decoded_pixels_v2(
        base.pixel_format(),
        &sc,
        corrected_cicp.as_ref(),
    );
    if header.is_xyb || base.channel_type() == zenpixels::ChannelType::F32 {
        // Float rows return linear-light samples; source transfer describes the
        // encoded samples, not these returned pixels.
        decoded.with_transfer(base.transfer())
    } else {
        decoded
    }
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
        // The zencodec EXIF blob is the TIFF structure (module docs).
        info = info.with_exif(crate::encode::exif::strip_app1_exif_header(exif).to_vec());
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
/// Retain non-square unknown-unit aspect separately from physical density.
/// Square unknown-unit pixels need no override of JPEG's ordinary square-pixel
/// interpretation. Zero density fields remain unavailable.
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
        DensityUnits::None if jfif.x_density == jfif.y_density => return None,
        DensityUnits::None => ResolutionUnit::Unknown,
    };

    Some(zencodec::Resolution {
        x: jfif.x_density as f64,
        y: jfif.y_density as f64,
        unit,
    })
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    #[test]
    fn incomplete_and_empty_containers_do_not_prove_gain_map_absence() {
        let truncated = crate::container::ContainerProbe {
            truncated: true,
            ..Default::default()
        };
        let mut info = ImageInfo::new(1, 1, ImageFormat::Jpeg);
        populate_probe_scope(&truncated, &mut info);
        assert_eq!(info.gain_map, zencodec::GainMapPresence::Unknown);
        for data in [&[][..], &[0xff, 0xd8][..]] {
            let mut info = ImageInfo::new(1, 1, ImageFormat::Jpeg);
            populate_complete_container_scope(data, &mut info);
            assert_eq!(info.gain_map, zencodec::GainMapPresence::Unknown);
            assert!(!info.supplements.auxiliary);
            assert!(!info.supplements.gain_map);
        }
    }
}
