//! Lossless pipeline fuzzer: `transform`, `apply_exif_orientation` and
//! `restructure` on arbitrary input must not panic, hang or amplify.
//!
//! They carry MPF secondary images (gain maps, thumbnails) from the input, so
//! a crafted container is the interesting input: repeated, overlapping or
//! nested MPF entries once multiplied the work and the output. A successful
//! output is bounded by the input (secondaries are disjoint ranges of it)
//! plus the re-encoded images.
//!
//! The selector (transform, edge policy, scan mode) comes from the input's
//! length and last byte, so plain JPEG files work as seeds unchanged.

use enough::Unstoppable;
use zenjpeg::container::find_jpeg_boundaries;
use zenjpeg::decoder::Decoder;
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, OutputMode, RestructureConfig, TransformConfig,
    apply_exif_orientation_with_edge_handling, restructure, transform,
};

/// The other decode targets' pixel cap. The lossless API takes no limits,
/// so inputs holding a larger image (primary or any embedded JPEG) are
/// skipped rather than reported as slow.
const MAX_PX: u64 = 4_000_000;

pub fn check(data: &[u8]) {
    let Some(padded_px) = padded_pixels(data) else {
        return;
    };
    let selector = (data.len() as u8).wrapping_mul(31) ^ data.last().copied().unwrap_or(0);
    let transforms = [
        LosslessTransform::None,
        LosslessTransform::FlipHorizontal,
        LosslessTransform::FlipVertical,
        LosslessTransform::Rotate180,
        LosslessTransform::Rotate90,
        LosslessTransform::Rotate270,
        LosslessTransform::Transpose,
        LosslessTransform::Transverse,
    ];
    let edge_handling = if selector & 8 == 0 {
        EdgeHandling::RejectPartialBlocks
    } else {
        EdgeHandling::TrimPartialBlocks
    };
    let config = TransformConfig {
        transform: transforms[usize::from(selector & 7)],
        edge_handling,
    };
    // Secondaries are carried from disjoint ranges of the input; the primary
    // and each transformed secondary are re-encoded. A re-encoded block costs
    // at most 256 bytes (64 coefficients, each a 16-bit code plus 15 extra
    // bits), 4 bytes per pixel per component; four components and 0xFF
    // stuffing on every byte make 32 per pixel of the MCU-padded area.
    // Anything beyond that is amplification.
    let bound = 2 * data.len() as u128 + 32 * u128::from(padded_px) + 4096;
    let check = |what: &str, out: &[u8]| {
        assert!(
            out.len() as u128 <= bound,
            "{what}: {} output bytes from {} input bytes",
            out.len(),
            data.len()
        );
    };
    if let Ok(out) = transform(data, &config, Unstoppable) {
        check("transform", &out);
    }
    if let Ok(out) = apply_exif_orientation_with_edge_handling(data, edge_handling, Unstoppable) {
        check("apply_exif_orientation", &out);
    }
    let restructure_config = RestructureConfig {
        output_mode: if selector & 16 == 0 {
            OutputMode::Sequential
        } else {
            OutputMode::Progressive
        },
        transform: (selector & 32 != 0).then_some(config),
        ..RestructureConfig::default()
    };
    if let Ok(out) = restructure(data, &restructure_config, Unstoppable) {
        check("restructure", &out);
    }
}

/// Pixels of every image the input holds, each padded to a 32x32 MCU (the
/// largest sampling factor), or `None` when the input has no readable primary
/// or holds an image over [`MAX_PX`]. The primary is read from the whole
/// input: a truncated scan with no EOI still decodes (padded), so it is not
/// always among `find_jpeg_boundaries`; when it is (a range at 0), it counts
/// once.
fn padded_pixels(data: &[u8]) -> Option<u64> {
    let primary = Decoder::new().read_info(data).ok()?.dimensions;
    let others = find_jpeg_boundaries(data)
        .into_iter()
        .filter(|range| range.start != 0)
        .filter_map(|range| Decoder::new().read_info(&data[range]).ok())
        .map(|info| info.dimensions);
    let mut total = 0u64;
    for d in core::iter::once(primary).chain(others) {
        let (w, h) = (u64::from(d.width), u64::from(d.height));
        if w * h > MAX_PX {
            return None;
        }
        total += w.next_multiple_of(32) * h.next_multiple_of(32);
    }
    Some(total)
}
