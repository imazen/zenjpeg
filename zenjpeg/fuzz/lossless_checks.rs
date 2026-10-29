//! Lossless pipeline fuzzer: `transform`, `apply_exif_orientation` and
//! `restructure` on arbitrary input must not panic, hang or amplify.
//!
//! They carry MPF secondary images (gain maps, thumbnails) and any trailing
//! bytes from the input, so a crafted container is the interesting input:
//! repeated, overlapping or nested MPF entries once multiplied the work and
//! the output. A successful output is bounded by the input (secondaries and
//! the trailer are disjoint ranges of it) plus the re-encoded images; it holds
//! no more complete JPEG streams than the primary plus every image the input
//! could carry (a damaged secondary comes back complete, so the input's own
//! complete streams are not the bound); and without a transform its primary
//! has the input's coefficients.
//!
//! The selector (transform, edge policy, scan mode) comes from the input's
//! length and last byte, so plain JPEG files work as seeds unchanged.

use enough::Unstoppable;
use zenjpeg::container::{find_jpeg_boundaries, parse_mpf, primary_bounds};
use zenjpeg::decoder::Decoder;
use zenjpeg::decoder::{DecodeConfig, DecodedCoefficients};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, OutputMode, RestructureConfig, TransformConfig,
    apply_exif_orientation_with_edge_handling, restructure, transform,
};

/// The other decode targets' pixel cap. The lossless API takes no limits,
/// so inputs whose primary is larger are skipped rather than reported as
/// slow. Secondaries are not capped here: the pipeline bounds the ones it
/// transforms by the primary (each at most its pixels, all at most four
/// times), so a large declared secondary must fail fast, not be skipped.
const MAX_PX: u64 = 4_000_000;

pub fn check(data: &[u8]) {
    let Some((padded_px, images)) = padded_pixels(data) else {
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
    let streams = images;
    let check = |what: &str, out: &[u8]| {
        assert!(
            out.len() as u128 <= bound,
            "{what}: {} output bytes from {} input bytes",
            out.len(),
            data.len()
        );
        let out_streams = find_jpeg_boundaries(out).len();
        assert!(
            out_streams <= streams,
            "{what}: {out_streams} JPEG streams out of an input holding {streams}"
        );
    };
    // Without a transform the primary's coefficients pass through unchanged.
    let input_coefficients = DecodeConfig::new()
        .decode_coefficients(data, Unstoppable)
        .ok();
    let same_coefficients = |what: &str, out: &[u8]| {
        let Some(input) = &input_coefficients else {
            return;
        };
        let output = DecodeConfig::new()
            .decode_coefficients(out, Unstoppable)
            .unwrap_or_else(|e| panic!("{what}: output does not decode: {e}"));
        assert!(
            coefficients_equal(input, &output),
            "{what}: coefficients changed"
        );
    };
    if let Ok(out) = transform(data, &config, Unstoppable) {
        check("transform", &out);
        if config.transform == LosslessTransform::None
            && config.edge_handling == EdgeHandling::RejectPartialBlocks
        {
            same_coefficients("transform(None)", &out);
        }
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
        if restructure_config.transform.is_none() {
            same_coefficients("restructure", &out);
        }
    }
}

/// Same frame and the same coefficients in every block the image covers.
/// Blocks in the MCU padding are not compared: a progressive output codes a
/// component's own block grid in its single-component scans, as JPEG
/// requires, so padding blocks come back zero whatever the input held there.
fn coefficients_equal(a: &DecodedCoefficients, b: &DecodedCoefficients) -> bool {
    if (a.width, a.height) != (b.width, b.height) || a.components.len() != b.components.len() {
        return false;
    }
    let h_max = a
        .components
        .iter()
        .map(|c| c.h_samp)
        .max()
        .unwrap_or(1)
        .max(1);
    let v_max = a
        .components
        .iter()
        .map(|c| c.v_samp)
        .max()
        .unwrap_or(1)
        .max(1);
    a.components.iter().zip(&b.components).all(|(x, y)| {
        if (x.blocks_wide, x.blocks_high, x.h_samp, x.v_samp)
            != (y.blocks_wide, y.blocks_high, y.h_samp, y.v_samp)
        {
            return false;
        }
        // The component's own size: ceil(width * h / h_max), in blocks.
        let px = |dim: u32, samp: u8, max: u8| {
            (u64::from(dim) * u64::from(samp)).div_ceil(u64::from(max))
        };
        let wide = (px(a.width, x.h_samp, h_max).div_ceil(8) as usize).min(x.blocks_wide);
        let high = (px(a.height, x.v_samp, v_max).div_ceil(8) as usize).min(x.blocks_high);
        (0..high).all(|row| {
            let start = row * x.blocks_wide * 64;
            let end = start + wide * 64;
            x.coeffs.get(start..end) == y.coeffs.get(start..end)
        })
    })
}

/// Pixels of every image the input holds, each padded to a 32x32 MCU (the
/// largest sampling factor), or `None` when the input has no readable primary
/// or its primary is over [`MAX_PX`]. The primary is read from the whole
/// input: a truncated scan with no EOI still decodes (padded), so it is not
/// always among `find_jpeg_boundaries`; when it is (a range at 0), it counts
/// once. The others are every complete JPEG after it and every MPF entry
/// that `read_info` accepts (an MPF secondary without an EOI is transformed
/// too), each counted once by its start. Also returns how many complete
/// JPEG streams the output may hold: the primary, plus one per distinct start
/// of an MPF entry or of an `FF D8` after the primary (bytes after a damaged
/// secondary are carried verbatim, and a stream there that the input's
/// boundary scan never reached becomes visible once the secondary is
/// re-encoded whole).
fn padded_pixels(data: &[u8]) -> Option<(u64, usize)> {
    let primary = Decoder::new().read_info(data).ok()?.dimensions;
    let (w, h) = (u64::from(primary.width), u64::from(primary.height));
    if w * h > MAX_PX {
        return None;
    }
    let mut ranges: Vec<core::ops::Range<usize>> = find_jpeg_boundaries(data)
        .into_iter()
        .filter(|range| range.start != 0)
        .collect();
    for entry in parse_mpf(data).unwrap_or_default().iter().skip(1) {
        let end = entry.offset.saturating_add(entry.size).min(data.len());
        if entry.offset < end && !ranges.iter().any(|r| r.start == entry.offset) {
            ranges.push(entry.offset..end);
        }
    }
    let after_primary = primary_bounds(data).map_or(data.len(), |b| b.end);
    let mut starts: Vec<usize> = ranges.iter().map(|r| r.start).collect();
    starts.extend(
        data.windows(2)
            .enumerate()
            .skip(after_primary)
            .filter(|(_, w)| *w == [0xFF, 0xD8])
            .map(|(i, _)| i),
    );
    starts.sort_unstable();
    starts.dedup();
    let images = 1 + starts.len();
    let mut total = w.next_multiple_of(32) * h.next_multiple_of(32);
    for range in ranges {
        if let Ok(info) = Decoder::new().read_info(&data[range]) {
            let (w, h) = (
                u64::from(info.dimensions.width),
                u64::from(info.dimensions.height),
            );
            total = total.saturating_add(w.next_multiple_of(32) * h.next_multiple_of(32));
        }
    }
    Some((total, images))
}
