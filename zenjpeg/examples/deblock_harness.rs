//! Deblocking experiment harness.
//!
//! Encodes source images with libjpeg-turbo, mozjpeg, and cjpegli at multiple
//! quality levels, caching the encoded JPEGs to disk. Then measures decode quality
//! with pluggable deblocking strategies.
//!
//! The cache is reusable across runs — encode once, measure many times.
//!
//! Usage:
//! ```bash
//! # Generate cached encoded JPEGs (first run)
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --generate
//!
//! # Measure baseline (no deblocking)
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --measure
//!
//! # Both in one shot
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder
//!
//! # Limit to N images for quick testing
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --images 3
//!
//! # Use specific corpus
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --corpus cid22
//! ```

use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use zenjpeg::detect::{self, EncoderFamily};
use zenjpeg_bench_utils::{decode_jpeg_with_icc, ImageData, QualityMetrics, RgbImage};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const CACHE_DIR: &str = "/mnt/v/output/zenjpeg/deblock";
const RESULTS_DIR: &str = "/mnt/v/output/zenjpeg/deblock/results";

const QUALITY_LEVELS: [u8; 17] = [
    5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 93, 95, 97,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Encoder {
    Turbo420,
    Mozjpeg420,
    Cjpegli,
}

impl Encoder {
    fn dir_name(self) -> &'static str {
        match self {
            Self::Turbo420 => "turbo-420",
            Self::Mozjpeg420 => "mozjpeg-420",
            Self::Cjpegli => "cjpegli",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Turbo420 => "libjpeg-turbo 4:2:0",
            Self::Mozjpeg420 => "mozjpeg 4:2:0",
            Self::Cjpegli => "cjpegli",
        }
    }

    fn expected_family(self) -> EncoderFamily {
        match self {
            Self::Turbo420 => EncoderFamily::LibjpegTurbo,
            Self::Mozjpeg420 => EncoderFamily::Mozjpeg,
            Self::Cjpegli => EncoderFamily::CjpegliYcbcr,
        }
    }

    fn all() -> &'static [Encoder] {
        &[Self::Turbo420, Self::Mozjpeg420, Self::Cjpegli]
    }
}

/// A deblocking strategy. Takes JPEG bytes, returns decoded (possibly enhanced) RGB.
trait DeblockStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage>;
}

/// Baseline: standard zenjpeg integer IDCT decode, no enhancements.
struct BaselineDecode;

impl DeblockStrategy for BaselineDecode {
    fn name(&self) -> &str {
        "baseline"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        decode_jpeg_with_icc(jpeg_bytes).ok()
    }
}

/// Dequant bias: zenjpeg f32 IDCT with Laplacian dequantization biases.
struct DequantBiasDecode;

impl DeblockStrategy for DequantBiasDecode {
    fn name(&self) -> &str {
        "dequant_bias"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;

        let decoded = Decoder::new()
            .dequant_bias(true)
            .decode(jpeg_bytes, Unstoppable)
            .ok()?;

        let w = decoded.width() as usize;
        let h = decoded.height() as usize;

        // dequant_bias uses SrgbF32Precise → f32 output in [0.0, 1.0], convert to u8
        let f32_pixels = decoded.pixels_f32()?;
        let u8_pixels: Vec<u8> = f32_pixels
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            .collect();
        Some(zenjpeg_bench_utils::bytes_to_rgb(&u8_pixels, w, h))
    }
}

// ---------------------------------------------------------------------------
// Deblocking helpers
// ---------------------------------------------------------------------------

/// JPEG zigzag order: maps natural (raster) index → zigzag index.
/// natural_to_zigzag[natural_pos] = zigzag_pos
const NATURAL_TO_ZIGZAG: [usize; 64] = [
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
];

/// Dequantize and unzigzag: takes zigzag-order coefficients and NATURAL-order quant table
/// (as returned by DecodedCoefficients::quant_tables), returns natural (raster) order
/// f32 coefficients ready for IDCT.
///
/// Note: the decoder parser converts quant tables from zigzag to natural order during
/// DQT parsing (markers.rs:267-273), so quant_tables are always in natural order.
fn dequantize_unzigzag(
    zigzag_coeffs: &[i16; 64],
    natural_quant: &[u16; 64],
) -> [f32; 64] {
    let mut result = [0.0f32; 64];
    for nat in 0..64 {
        let zi = NATURAL_TO_ZIGZAG[nat];
        result[nat] = zigzag_coeffs[zi] as f32 * natural_quant[nat] as f32;
    }
    result
}

/// Get the DC quantization step for the luma component.
/// Used to scale filter strength — larger quant step = more blocking = stronger filter.
fn get_luma_dc_quant(jpeg_bytes: &[u8]) -> Option<u16> {
    use enough::Unstoppable;
    use zenjpeg::decoder::Decoder;
    let coeffs = Decoder::new()
        .decode_coefficients(jpeg_bytes, Unstoppable)
        .ok()?;
    if coeffs.components.is_empty() {
        return None;
    }
    let qt_idx = coeffs.components[0].quant_table_idx as usize;
    let qt = coeffs.quant_tables.get(qt_idx)?.as_ref()?;
    Some(qt[0]) // DC value is zigzag position 0
}

/// Decode JPEG to planar Y, Cb, Cr (each as Vec<f32> in 0-255 range).
/// For 4:2:0: cb/cr are half-resolution.
#[allow(clippy::type_complexity)]
fn decode_to_coeff_planes(
    jpeg_bytes: &[u8],
) -> Option<CoeffPlanes> {
    use enough::Unstoppable;
    use zenjpeg::decoder::Decoder;

    let coeffs = Decoder::new()
        .decode_coefficients(jpeg_bytes, Unstoppable)
        .ok()?;

    if coeffs.components.len() < 3 {
        return None;
    }

    let w = coeffs.width as usize;
    let h = coeffs.height as usize;

    let mut planes = Vec::with_capacity(3);

    for ci in 0..3 {
        let comp = &coeffs.components[ci];
        let qt_idx = comp.quant_table_idx as usize;
        let qt = coeffs.quant_tables[qt_idx].as_ref()?;
        let bw = comp.blocks_wide;
        let bh = comp.blocks_high;
        let pw = bw * 8;
        let ph = bh * 8;

        let mut plane = vec![0.0f32; pw * ph];

        for by in 0..bh {
            for bx in 0..bw {
                let block_zigzag = comp.block_at(bx, by);
                let block_arr: [i16; 64] = block_zigzag.try_into().unwrap();
                // dequantize + convert zigzag → natural order for IDCT
                let dequant = dequantize_unzigzag(&block_arr, qt);
                let pixels = zenjpeg::decode::idct::inverse_dct_8x8(&dequant);

                for row in 0..8 {
                    for col in 0..8 {
                        let px = bx * 8 + col;
                        let py = by * 8 + row;
                        if px < pw && py < ph {
                            plane[py * pw + px] = pixels[row * 8 + col] + 128.0;
                        }
                    }
                }
            }
        }

        planes.push(ComponentPlane {
            data: plane,
            width: pw,
            height: ph,
            blocks_wide: bw,
            blocks_high: bh,
            quant_table: *qt,
        });
    }

    Some(CoeffPlanes {
        planes,
        image_width: w,
        image_height: h,
    })
}

struct ComponentPlane {
    data: Vec<f32>,
    width: usize,
    height: usize,
    blocks_wide: usize,
    blocks_high: usize,
    quant_table: [u16; 64],
}

struct CoeffPlanes {
    planes: Vec<ComponentPlane>,
    image_width: usize,
    image_height: usize,
}

/// Simple bilinear 2x upsample for chroma planes (4:2:0 → full res).
fn upsample_2x(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dw * dh];
    for dy in 0..dh {
        let sy_f = (dy as f32 + 0.5) * (sh as f32 / dh as f32) - 0.5;
        let sy0 = (sy_f.floor() as isize).max(0) as usize;
        let sy1 = (sy0 + 1).min(sh - 1);
        let fy = sy_f - sy0 as f32;

        for dx in 0..dw {
            let sx_f = (dx as f32 + 0.5) * (sw as f32 / dw as f32) - 0.5;
            let sx0 = (sx_f.floor() as isize).max(0) as usize;
            let sx1 = (sx0 + 1).min(sw - 1);
            let fx = sx_f - sx0 as f32;

            let v00 = src[sy0 * sw + sx0];
            let v10 = src[sy0 * sw + sx1];
            let v01 = src[sy1 * sw + sx0];
            let v11 = src[sy1 * sw + sx1];

            dst[dy * dw + dx] =
                v00 * (1.0 - fx) * (1.0 - fy) +
                v10 * fx * (1.0 - fy) +
                v01 * (1.0 - fx) * fy +
                v11 * fx * fy;
        }
    }
    dst
}

/// Convert YCbCr planes (f32, 0-255 range) to RGB ImgVec.
fn ycbcr_planes_to_rgb(
    y: &[f32], cb: &[f32], cr: &[f32],
    w: usize, h: usize,
) -> RgbImage {
    use imgref::ImgVec;
    use rgb::RGB8;

    let mut pixels = vec![RGB8::default(); w * h];
    for i in 0..w * h {
        let yv = y[i];
        let cbv = cb[i] - 128.0;
        let crv = cr[i] - 128.0;
        let r = (yv + 1.402 * crv).round().clamp(0.0, 255.0) as u8;
        let g = (yv - 0.344136 * cbv - 0.714136 * crv).round().clamp(0.0, 255.0) as u8;
        let b = (yv + 1.772 * cbv).round().clamp(0.0, 255.0) as u8;
        pixels[i] = RGB8 { r, g, b };
    }
    ImgVec::new(pixels, w, h)
}

/// Standard reconstruction from coefficient planes (no deblocking).
fn planes_to_rgb(cp: &CoeffPlanes) -> RgbImage {
    let w = cp.image_width;
    let h = cp.image_height;

    let y_plane = &cp.planes[0];
    let cb_plane = &cp.planes[1];
    let cr_plane = &cp.planes[2];

    // Trim Y to image dimensions
    let mut y = vec![0.0f32; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = y_plane.data[row * y_plane.width + col];
        }
    }

    // Upsample chroma if needed
    let cb_up = if cb_plane.width < w || cb_plane.height < h {
        upsample_2x(&cb_plane.data, cb_plane.width, cb_plane.height, w, h)
    } else {
        let mut out = vec![0.0f32; w * h];
        for row in 0..h {
            out[row * w..row * w + w]
                .copy_from_slice(&cb_plane.data[row * cb_plane.width..row * cb_plane.width + w]);
        }
        out
    };
    let cr_up = if cr_plane.width < w || cr_plane.height < h {
        upsample_2x(&cr_plane.data, cr_plane.width, cr_plane.height, w, h)
    } else {
        let mut out = vec![0.0f32; w * h];
        for row in 0..h {
            out[row * w..row * w + w]
                .copy_from_slice(&cr_plane.data[row * cr_plane.width..row * cr_plane.width + w]);
        }
        out
    };

    ycbcr_planes_to_rgb(&y, &cb_up, &cr_up, w, h)
}

// ---------------------------------------------------------------------------
// Strategy: Reconstruct (verify pipeline - should match baseline)
// ---------------------------------------------------------------------------

/// Verification: coefficient decode → IDCT → color convert, NO filtering.
/// This should produce results very close to baseline. If it doesn't,
/// the reconstruction pipeline is broken.
struct ReconstructVerify;

impl DeblockStrategy for ReconstructVerify {
    fn name(&self) -> &str {
        "reconstruct_verify"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        let cp = decode_to_coeff_planes(jpeg_bytes)?;
        Some(planes_to_rgb(&cp))
    }
}

// ---------------------------------------------------------------------------
// Strategy: Boundary 4-tap filter (H.264-inspired)
// ---------------------------------------------------------------------------

/// H.264-style boundary filter. Applies 4-tap weighted average at 8x8 block
/// boundaries. Strength proportional to quantization step.
struct Boundary4Tap;

impl Boundary4Tap {
    /// Apply boundary filter to a single plane.
    /// `strength` controls maximum pixel adjustment (derived from quant step).
    fn filter_plane(plane: &mut [f32], w: usize, h: usize, strength: f32) {
        if strength < 0.5 {
            return;
        }

        // Threshold: only filter if boundary discontinuity exceeds this
        let thresh = strength * 0.4;

        // Vertical boundaries (columns at 8, 16, 24, ...)
        for y in 0..h {
            for bx in 1..(w / 8) {
                let col = bx * 8;
                if col + 1 >= w || col < 2 {
                    continue;
                }

                let p1 = plane[y * w + col - 2];
                let p0 = plane[y * w + col - 1];
                let q0 = plane[y * w + col];
                let q1 = plane[y * w + col + 1];

                let disc = (p0 - q0).abs();
                if disc < thresh {
                    continue;
                }

                // 4-tap: [1, 3, 3, 1] / 8 weighted average across boundary
                let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
                let delta_p = (avg - p0).clamp(-strength, strength);
                let delta_q = (avg - q0).clamp(-strength, strength);

                plane[y * w + col - 1] = (p0 + delta_p).clamp(0.0, 255.0);
                plane[y * w + col] = (q0 + delta_q).clamp(0.0, 255.0);
            }
        }

        // Horizontal boundaries (rows at 8, 16, 24, ...)
        for x in 0..w {
            for by in 1..(h / 8) {
                let row = by * 8;
                if row + 1 >= h || row < 2 {
                    continue;
                }

                let p1 = plane[(row - 2) * w + x];
                let p0 = plane[(row - 1) * w + x];
                let q0 = plane[row * w + x];
                let q1 = plane[(row + 1) * w + x];

                let disc = (p0 - q0).abs();
                if disc < thresh {
                    continue;
                }

                let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
                let delta_p = (avg - p0).clamp(-strength, strength);
                let delta_q = (avg - q0).clamp(-strength, strength);

                plane[(row - 1) * w + x] = (p0 + delta_p).clamp(0.0, 255.0);
                plane[row * w + x] = (q0 + delta_q).clamp(0.0, 255.0);
            }
        }
    }
}

impl DeblockStrategy for Boundary4Tap {
    fn name(&self) -> &str {
        "boundary_4tap"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        let mut cp = decode_to_coeff_planes(jpeg_bytes)?;

        // Filter each component at its own block boundaries
        for plane in &mut cp.planes {
            // Strength = DC quant / 4, capped at 12.0
            let dc_quant = plane.quant_table[0] as f32;
            let strength = (dc_quant * 0.25).min(12.0);
            Self::filter_plane(
                &mut plane.data, plane.width, plane.height, strength
            );
        }

        Some(planes_to_rgb(&cp))
    }
}

// ---------------------------------------------------------------------------
// Strategy: CDEF-style directional filter
// ---------------------------------------------------------------------------

/// CDEF-inspired directional constrained nonlinear filter.
/// For each 8x8 block, finds the best direction, then applies a tap filter
/// along that direction with strength derived from the quant table.
struct CDEFDirection;

impl CDEFDirection {
    /// AV1 CDEF directions: 8 angles, each defined by (dx, dy) offsets.
    const DIRECTIONS: [(i32, i32); 8] = [
        (1, 0),   // 0°: horizontal
        (1, 1),   // 45°
        (0, 1),   // 90°: vertical
        (-1, 1),  // 135°
        (1, 0),   // 180° = 0° (same as 0 but we use abs for symmetry)
        (1, -1),  // 225° = 315°
        (0, -1),  // 270° = 90°
        (-1, -1), // 315° = 135°
    ];

    /// Find the best direction for a block by minimizing variance along the direction.
    fn find_direction(plane: &[f32], w: usize, bx: usize, by: usize) -> usize {
        let mut best_dir = 0;
        let mut best_var = f32::MAX;

        // Only check the 4 unique directions (0°, 45°, 90°, 135°)
        for dir in 0..4 {
            let (dx, dy) = Self::DIRECTIONS[dir];
            let mut sum_diff_sq = 0.0f32;
            let mut count = 0;

            // For each pixel in the block, measure orthogonal variance
            for iy in 0..8 {
                for ix in 0..8 {
                    let px = bx * 8 + ix;
                    let py = by * 8 + iy;

                    // Sample along the direction
                    let nx = px as i32 + dx;
                    let ny = py as i32 + dy;

                    if nx >= 0 && (nx as usize) < w && ny >= 0 {
                        let ny_u = ny as usize;
                        let h = plane.len() / w;
                        if ny_u < h {
                            let cur = plane[py * w + px];
                            let next = plane[ny_u * w + nx as usize];
                            sum_diff_sq += (cur - next) * (cur - next);
                            count += 1;
                        }
                    }
                }
            }

            let var = if count > 0 { sum_diff_sq / count as f32 } else { f32::MAX };
            if var < best_var {
                best_var = var;
                best_dir = dir;
            }
        }

        best_dir
    }

    /// Constrain function: clip large differences (reject outliers).
    #[inline]
    fn constrain(diff: f32, strength: f32, damping: f32) -> f32 {
        let abs_diff = diff.abs();
        if abs_diff >= strength {
            return 0.0;
        }
        let sign = diff.signum();
        // Smooth rolloff near strength threshold
        let dampened = abs_diff.max(0.0) - (abs_diff * abs_diff / (damping * 2.0)).min(abs_diff);
        sign * dampened
    }

    /// Apply CDEF filter to boundary pixels of a plane.
    fn filter_plane(plane: &mut [f32], w: usize, h: usize, strength: f32) {
        if strength < 0.5 {
            return;
        }

        let damping = (strength * 3.0).max(3.0);
        let bw = w / 8;
        let bh = h / 8;

        // Work on a copy so we read original values
        let orig = plane.to_vec();

        for by in 0..bh {
            for bx in 0..bw {
                let dir = Self::find_direction(&orig, w, bx, by);
                let (dx, dy) = Self::DIRECTIONS[dir];

                // Only filter pixels near block boundaries (2 pixels each side)
                for iy in 0..8 {
                    for ix in 0..8 {
                        // Skip interior pixels (not near any block boundary)
                        let near_h_boundary = ix <= 1 || ix >= 6;
                        let near_v_boundary = iy <= 1 || iy >= 6;
                        if !near_h_boundary && !near_v_boundary {
                            continue;
                        }

                        let px = bx * 8 + ix;
                        let py = by * 8 + iy;
                        if px >= w || py >= h {
                            continue;
                        }

                        let center = orig[py * w + px];
                        let mut sum = 0.0f32;

                        // Primary taps: along direction, weights [2, 2]
                        for dist in [1i32, 2] {
                            let tap_weight = if dist == 1 { 2.0 } else { 1.0 };

                            for sign in [-1i32, 1] {
                                let tx = px as i32 + dx * dist * sign;
                                let ty = py as i32 + dy * dist * sign;

                                if tx >= 0 && (tx as usize) < w && ty >= 0 && (ty as usize) < h {
                                    let tap = orig[ty as usize * w + tx as usize];
                                    sum += Self::constrain(tap - center, strength, damping) * tap_weight;
                                }
                            }
                        }

                        // Apply: center + round(sum / 8)
                        let delta = (sum / 8.0).clamp(-strength, strength);
                        plane[py * w + px] = (center + delta).clamp(0.0, 255.0);
                    }
                }
            }
        }
    }
}

impl DeblockStrategy for CDEFDirection {
    fn name(&self) -> &str {
        "cdef_direction"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        let mut cp = decode_to_coeff_planes(jpeg_bytes)?;

        for plane in &mut cp.planes {
            let dc_quant = plane.quant_table[0] as f32;
            let strength = (dc_quant * 0.3).min(15.0);
            Self::filter_plane(
                &mut plane.data, plane.width, plane.height, strength
            );
        }

        Some(planes_to_rgb(&cp))
    }
}

// ---------------------------------------------------------------------------
// Strategy: Coefficient smoothing (single-pass boundary minimization)
// ---------------------------------------------------------------------------

/// Single-pass coefficient refinement: adjusts low-frequency AC coefficients
/// within their quantization intervals to minimize boundary discontinuity.
struct CoeffSmooth;

impl CoeffSmooth {
    /// Single-pass DCT-domain coefficient smoothing across ALL AC frequencies.
    ///
    /// Like CoeffRefineTV but smooths all frequencies (not just low ones).
    /// Uses cardinal neighbor averaging with a single pass, small alpha.
    /// More aggressive than CoeffRefineTV (touches HF) but single-pass (no
    /// iteration), making it faster and less prone to over-smoothing.
    fn process_component(
        zigzag_coeffs: &[i16],
        blocks_wide: usize,
        blocks_high: usize,
        quant_table: &[u16; 64],
    ) -> Vec<f32> {
        let num_blocks = blocks_wide * blocks_high;

        // Dequantize with interval tracking
        let mut dequant = vec![[0.0f32; 64]; num_blocks];
        let mut block_min = vec![[0.0f32; 64]; num_blocks];
        let mut block_max = vec![[0.0f32; 64]; num_blocks];

        for bi in 0..num_blocks {
            let block: [i16; 64] = zigzag_coeffs[bi * 64..(bi + 1) * 64]
                .try_into().unwrap();
            for nat in 0..64 {
                let zi = NATURAL_TO_ZIGZAG[nat];
                let q = quant_table[nat] as f32;
                let mid = block[zi] as f32 * q;
                dequant[bi][nat] = mid;
                block_min[bi][nat] = mid - q * 0.5;
                block_max[bi][nat] = mid + q * 0.5;
            }
        }

        let alpha = 0.10;
        let snapshot = dequant.clone();

        for bi in 0..num_blocks {
            let by = bi / blocks_wide;
            let bx = bi % blocks_wide;

            // Smooth ALL AC coefficients (skip DC)
            for nat in 1..64 {
                let center = snapshot[bi][nat];

                let mut sum = 0.0f32;
                let mut count = 0u32;

                if by > 0 {
                    sum += snapshot[(by - 1) * blocks_wide + bx][nat];
                    count += 1;
                }
                if by + 1 < blocks_high {
                    sum += snapshot[(by + 1) * blocks_wide + bx][nat];
                    count += 1;
                }
                if bx > 0 {
                    sum += snapshot[by * blocks_wide + bx - 1][nat];
                    count += 1;
                }
                if bx + 1 < blocks_wide {
                    sum += snapshot[by * blocks_wide + bx + 1][nat];
                    count += 1;
                }

                if count > 0 {
                    let avg = sum / count as f32;
                    let new_val = center + alpha * (avg - center);
                    dequant[bi][nat] = new_val.clamp(
                        block_min[bi][nat], block_max[bi][nat],
                    );
                }
            }
        }

        // Final IDCT
        let pw = blocks_wide * 8;
        let ph = blocks_high * 8;
        let mut output = vec![0.0f32; pw * ph];
        for bi in 0..num_blocks {
            let by = bi / blocks_wide;
            let bx = bi % blocks_wide;
            let pixels = zenjpeg::decode::idct::inverse_dct_8x8(&dequant[bi]);
            for row in 0..8 {
                for col in 0..8 {
                    output[(by * 8 + row) * pw + bx * 8 + col] =
                        pixels[row * 8 + col] + 128.0;
                }
            }
        }

        output
    }
}

impl DeblockStrategy for CoeffSmooth {
    fn name(&self) -> &str {
        "coeff_smooth"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;

        let coeffs = Decoder::new()
            .decode_coefficients(jpeg_bytes, Unstoppable)
            .ok()?;

        let w = coeffs.width as usize;
        let h = coeffs.height as usize;

        let mut planes = Vec::with_capacity(3);
        for ci in 0..coeffs.components.len().min(3) {
            let comp = &coeffs.components[ci];
            let qt_idx = comp.quant_table_idx as usize;
            let qt = coeffs.quant_tables[qt_idx].as_ref()?;
            let bw = comp.blocks_wide;
            let bh = comp.blocks_high;
            let pw = bw * 8;
            let ph = bh * 8;

            let plane_data = Self::process_component(&comp.coeffs, bw, bh, qt);

            planes.push(ComponentPlane {
                data: plane_data,
                width: pw,
                height: ph,
                blocks_wide: bw,
                blocks_high: bh,
                quant_table: *qt,
            });
        }

        let cp = CoeffPlanes {
            planes,
            image_width: w,
            image_height: h,
        };
        Some(planes_to_rgb(&cp))
    }
}

// ---------------------------------------------------------------------------
// Strategy: Iterative low-frequency DCT coefficient smoothing
// ---------------------------------------------------------------------------

/// Iterative low-frequency DCT coefficient smoothing.
///
/// For each iteration, averages low-frequency AC coefficients with cardinal
/// neighbors (4-connected: up/down/left/right blocks), clamping to quantization
/// intervals. Only modifies coefficients where row+col < 4 in the DCT matrix
/// (the lowest 10 frequencies), leaving high-frequency detail untouched.
///
/// Works purely in DCT domain — no forward DCT needed, avoiding numerical
/// noise from FDCT/IDCT mismatch that destroyed the POCS approach.
struct CoeffRefineTV {
    iterations: usize,
}

impl CoeffRefineTV {
    fn new(iterations: usize) -> Self {
        Self { iterations }
    }

    fn process_component(
        zigzag_coeffs: &[i16],
        blocks_wide: usize,
        blocks_high: usize,
        quant_table: &[u16; 64],
        iterations: usize,
    ) -> Vec<f32> {
        let num_blocks = blocks_wide * blocks_high;

        // Dequantize with interval tracking
        let mut dequant = vec![[0.0f32; 64]; num_blocks];
        let mut block_min = vec![[0.0f32; 64]; num_blocks];
        let mut block_max = vec![[0.0f32; 64]; num_blocks];

        for bi in 0..num_blocks {
            let block: [i16; 64] = zigzag_coeffs[bi * 64..(bi + 1) * 64]
                .try_into().unwrap();
            for nat in 0..64 {
                let zi = NATURAL_TO_ZIGZAG[nat];
                let q = quant_table[nat] as f32;
                let mid = block[zi] as f32 * q;
                dequant[bi][nat] = mid;
                block_min[bi][nat] = mid - q * 0.5;
                block_max[bi][nat] = mid + q * 0.5;
            }
        }

        for iter in 0..iterations {
            let alpha = 0.15 / (1.0 + iter as f32);

            let snapshot = dequant.clone();

            for bi in 0..num_blocks {
                let by = bi / blocks_wide;
                let bx = bi % blocks_wide;

                // Only smooth low-frequency AC coefficients (skip DC)
                for nat in 1..64 {
                    let row = nat / 8;
                    let col = nat % 8;
                    if row + col >= 4 { continue; }

                    let center = snapshot[bi][nat];

                    // Average with cardinal neighbors only
                    let mut sum = 0.0f32;
                    let mut count = 0u32;

                    if by > 0 {
                        sum += snapshot[(by - 1) * blocks_wide + bx][nat];
                        count += 1;
                    }
                    if by + 1 < blocks_high {
                        sum += snapshot[(by + 1) * blocks_wide + bx][nat];
                        count += 1;
                    }
                    if bx > 0 {
                        sum += snapshot[by * blocks_wide + bx - 1][nat];
                        count += 1;
                    }
                    if bx + 1 < blocks_wide {
                        sum += snapshot[by * blocks_wide + bx + 1][nat];
                        count += 1;
                    }

                    if count > 0 {
                        let avg = sum / count as f32;
                        let new_val = center + alpha * (avg - center);
                        dequant[bi][nat] = new_val.clamp(
                            block_min[bi][nat], block_max[bi][nat],
                        );
                    }
                }
            }
        }

        // Final IDCT
        let pw = blocks_wide * 8;
        let ph = blocks_high * 8;
        let mut output = vec![0.0f32; pw * ph];
        for bi in 0..num_blocks {
            let by = bi / blocks_wide;
            let bx = bi % blocks_wide;
            let pixels = zenjpeg::decode::idct::inverse_dct_8x8(&dequant[bi]);
            for row in 0..8 {
                for col in 0..8 {
                    output[(by * 8 + row) * pw + bx * 8 + col] =
                        pixels[row * 8 + col] + 128.0;
                }
            }
        }

        output
    }
}

impl DeblockStrategy for CoeffRefineTV {
    fn name(&self) -> &str {
        "coeff_refine_tv"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;

        let coeffs = Decoder::new()
            .decode_coefficients(jpeg_bytes, Unstoppable)
            .ok()?;

        let w = coeffs.width as usize;
        let h = coeffs.height as usize;

        let mut planes = Vec::with_capacity(3);
        for ci in 0..coeffs.components.len().min(3) {
            let comp = &coeffs.components[ci];
            let qt_idx = comp.quant_table_idx as usize;
            let qt = coeffs.quant_tables[qt_idx].as_ref()?;
            let bw = comp.blocks_wide;
            let bh = comp.blocks_high;
            let pw = bw * 8;
            let ph = bh * 8;

            let plane_data = Self::process_component(
                &comp.coeffs, bw, bh, qt, self.iterations,
            );

            planes.push(ComponentPlane {
                data: plane_data,
                width: pw,
                height: ph,
                blocks_wide: bw,
                blocks_high: bh,
                quant_table: *qt,
            });
        }

        let cp = CoeffPlanes {
            planes,
            image_width: w,
            image_height: h,
        };
        Some(planes_to_rgb(&cp))
    }
}

// ---------------------------------------------------------------------------
// Strategy: Knusperli (Google) — single-pass DCT-domain boundary correction
// ---------------------------------------------------------------------------

/// Knusperli-style deblocking. For each pair of adjacent 8x8 blocks, analytically
/// computes the boundary discontinuity in DCT space, then applies a linear gradient
/// correction to both blocks. The correction is accumulated in a separate buffer
/// and applied once at the end, preventing cascading artifacts.
///
/// All arithmetic uses f32 (the original uses 10-bit fixed point integers).
///
/// Reference: google/knusperli output_image.cc:CopyFromJpegComponent()
struct Knusperli;

impl Knusperli {
    /// DCT representation of a linear ramp from 0 to 1 across 8 pixels.
    /// Only the first 4 coefficients are non-zero — high-frequency corrections
    /// would introduce ringing rather than smooth the boundary.
    /// Original C++ uses 10-bit FP: [318, -285, 81, -32, 0, 0, 0, 0]
    /// We use f32: [0.3105, -0.2783, 0.0791, -0.0313, 0, 0, 0, 0]
    const LINEAR_GRADIENT: [f32; 8] = [
        318.0 / 1024.0,  // 0.3105
        -285.0 / 1024.0, // -0.2783
        81.0 / 1024.0,   // 0.0791
        -32.0 / 1024.0,  // -0.0313
        0.0,
        0.0,
        0.0,
        0.0,
    ];

    /// Alpha coefficients: α(0) = 1/√2, α(k>0) = 1.0.
    /// Multiplied by √2 to get: α(0)*√2 = 1.0, α(k>0)*√2 = √2.
    const ALPHA_SQRT2: [f32; 8] = [
        1.0,
        std::f32::consts::SQRT_2,
        std::f32::consts::SQRT_2,
        std::f32::consts::SQRT_2,
        std::f32::consts::SQRT_2,
        std::f32::consts::SQRT_2,
        std::f32::consts::SQRT_2,
        std::f32::consts::SQRT_2,
    ];

    /// Process one component (Y, Cb, or Cr independently).
    /// Coefficients are in zigzag order, quant table in natural order.
    /// Returns dequantized natural-order coefficients with Knusperli correction.
    fn process_component(
        zigzag_coeffs: &[i16],   // flat: num_blocks * 64, zigzag order
        blocks_wide: usize,
        blocks_high: usize,
        quant_table: &[u16; 64],
    ) -> Vec<f32> {
        let num_blocks = blocks_wide * blocks_high;

        // Dequantize all blocks to natural order
        let mut blocks_mid = vec![[0.0f32; 64]; num_blocks];
        let mut blocks_min = vec![[0.0f32; 64]; num_blocks];
        let mut blocks_max = vec![[0.0f32; 64]; num_blocks];
        let mut blocks_off = vec![[0.0f32; 64]; num_blocks];

        for bi in 0..num_blocks {
            let block: [i16; 64] = zigzag_coeffs[bi * 64..(bi + 1) * 64]
                .try_into().unwrap();
            for nat in 0..64 {
                let zi = NATURAL_TO_ZIGZAG[nat];
                let q = quant_table[nat] as f32;
                let coeff = block[zi] as f32;
                let mid = coeff * q;
                blocks_mid[bi][nat] = mid;
                blocks_min[bi][nat] = mid - q * 0.5;
                blocks_max[bi][nat] = mid + q * 0.5;
                blocks_off[bi][nat] = 0.0;
            }
        }

        // Horizontal pass: correct vertical boundaries between adjacent blocks.
        // For each row v (0..4 = low frequencies only), compute the pixel
        // discontinuity between the right edge of block_i and left edge of block_j.
        for by in 0..blocks_high {
            for bx in 0..(blocks_wide.saturating_sub(1)) {
                let bi = by * blocks_wide + bx;       // left block
                let bj = by * blocks_wide + bx + 1;   // right block

                for v in 0..4 {
                    // Compute boundary discontinuity delta_v.
                    // Right edge of left block: sum_u α(u) * (-1)^u * coeff_i[v,u]
                    // Left edge of right block: sum_u α(u) * coeff_j[v,u]
                    // delta_v = left_edge_of_right - right_edge_of_left
                    let mut delta_v = 0.0f32;
                    let mut hf_penalty = 0.0f32;

                    for u in 0..8 {
                        let pos = v * 8 + u;
                        let gi = blocks_mid[bi][pos];
                        let gj = blocks_mid[bj][pos];
                        let sign = if u % 2 == 0 { 1.0f32 } else { -1.0 };

                        delta_v += Self::ALPHA_SQRT2[u] * (gj - sign * gi);
                        hf_penalty += (u * u) as f32 * (gi * gi + gj * gj);
                    }

                    // Distribute correction using linear gradient basis.
                    // The HF penalty halving is applied inside the inner loop
                    // (matching C++ behavior — delta_v gets halved per-frequency).
                    // The correction sign for the right block is OPPOSITE to the
                    // delta sign: even u → -1, odd u → +1 (C++: u&1 ? 1 : -1).
                    for u in 0..8 {
                        if hf_penalty > 400.0 {
                            delta_v *= 0.5;
                        }
                        let corr_sign = if u % 2 == 0 { -1.0f32 } else { 1.0 };
                        let correction = delta_v * Self::LINEAR_GRADIENT[u];
                        blocks_off[bi][v * 8 + u] += correction;
                        blocks_off[bj][v * 8 + u] += correction * corr_sign;
                    }
                }
            }
        }

        // Vertical pass: correct horizontal boundaries between adjacent blocks.
        // Same logic but transposed: iterate u=0..4, sum over v=0..7.
        for by in 0..(blocks_high.saturating_sub(1)) {
            for bx in 0..blocks_wide {
                let bi = by * blocks_wide + bx;       // top block
                let bj = (by + 1) * blocks_wide + bx; // bottom block

                for u in 0..4 {
                    let mut delta_u = 0.0f32;
                    let mut hf_penalty = 0.0f32;

                    for v in 0..8 {
                        let pos = v * 8 + u;
                        let gi = blocks_mid[bi][pos];
                        let gj = blocks_mid[bj][pos];
                        let sign = if v % 2 == 0 { 1.0f32 } else { -1.0 };

                        delta_u += Self::ALPHA_SQRT2[v] * (gj - sign * gi);
                        hf_penalty += (v * v) as f32 * (gi * gi + gj * gj);
                    }

                    // Same as horizontal: HF penalty inside loop, opposite sign
                    // for bottom block correction.
                    for v in 0..8 {
                        if hf_penalty > 400.0 {
                            delta_u *= 0.5;
                        }
                        let corr_sign = if v % 2 == 0 { -1.0f32 } else { 1.0 };
                        let correction = delta_u * Self::LINEAR_GRADIENT[v];
                        blocks_off[bi][v * 8 + u] += correction;
                        blocks_off[bj][v * 8 + u] += correction * corr_sign;
                    }
                }
            }
        }

        // Apply offsets: scale by 1/(2√2) to balance H and V corrections,
        // then clamp to quantization intervals.
        let half_sqrt2_inv = 1.0 / (2.0 * std::f32::consts::SQRT_2);

        for bi in 0..num_blocks {
            for k in 0..64 {
                blocks_mid[bi][k] += blocks_off[bi][k] * half_sqrt2_inv;
                blocks_mid[bi][k] = blocks_mid[bi][k]
                    .clamp(blocks_min[bi][k], blocks_max[bi][k]);
            }
        }

        // IDCT all blocks to pixel plane
        let pw = blocks_wide * 8;
        let ph = blocks_high * 8;
        let mut plane = vec![0.0f32; pw * ph];

        for by in 0..blocks_high {
            for bx in 0..blocks_wide {
                let bi = by * blocks_wide + bx;
                let pixels = zenjpeg::decode::idct::inverse_dct_8x8(&blocks_mid[bi]);
                for row in 0..8 {
                    for col in 0..8 {
                        plane[(by * 8 + row) * pw + bx * 8 + col] =
                            pixels[row * 8 + col] + 128.0;
                    }
                }
            }
        }

        plane
    }
}

impl DeblockStrategy for Knusperli {
    fn name(&self) -> &str {
        "knusperli"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;

        let coeffs = Decoder::new()
            .decode_coefficients(jpeg_bytes, Unstoppable)
            .ok()?;

        let w = coeffs.width as usize;
        let h = coeffs.height as usize;

        let mut planes = Vec::with_capacity(3);
        for ci in 0..coeffs.components.len().min(3) {
            let comp = &coeffs.components[ci];
            let qt_idx = comp.quant_table_idx as usize;
            let qt = coeffs.quant_tables[qt_idx].as_ref()?;
            let bw = comp.blocks_wide;
            let bh = comp.blocks_high;
            let pw = bw * 8;
            let ph = bh * 8;

            let plane_data = Self::process_component(
                &comp.coeffs, bw, bh, qt,
            );

            planes.push(ComponentPlane {
                data: plane_data,
                width: pw,
                height: ph,
                blocks_wide: bw,
                blocks_high: bh,
                quant_table: *qt,
            });
        }

        let cp = CoeffPlanes {
            planes,
            image_width: w,
            image_height: h,
        };
        Some(planes_to_rgb(&cp))
    }
}

// ---------------------------------------------------------------------------
// Strategy: QuantSmooth bilateral (jpeg-quantsmooth low-quality mode)
// ---------------------------------------------------------------------------

/// DCT-domain coefficient smoothing (jpeg-quantsmooth algorithm).
///
/// For each AC coefficient position, compute a bilateral-weighted average
/// from the 3x3 block neighborhood (same coefficient in neighboring blocks).
/// Clamp to quantization interval. Iterate.
///
/// This is fundamentally different from pixel-domain approaches: it smooths
/// individual DCT frequencies across block boundaries using spatial coherence.
struct QuantSmoothBilateral;

impl QuantSmoothBilateral {
    const ITERATIONS: usize = 4;

    fn process_component(
        zigzag_coeffs: &[i16],
        blocks_wide: usize,
        blocks_high: usize,
        quant_table: &[u16; 64],
    ) -> Vec<f32> {
        let pw = blocks_wide * 8;
        let ph = blocks_high * 8;
        let num_blocks = blocks_wide * blocks_high;

        // Dequantize all blocks with interval tracking
        let mut dequant = vec![[0.0f32; 64]; num_blocks];
        let mut block_min = vec![[0.0f32; 64]; num_blocks];
        let mut block_max = vec![[0.0f32; 64]; num_blocks];

        for bi in 0..num_blocks {
            let block: [i16; 64] = zigzag_coeffs[bi * 64..(bi + 1) * 64]
                .try_into().unwrap();
            for nat in 0..64 {
                let zi = NATURAL_TO_ZIGZAG[nat];
                let q = quant_table[nat] as f32;
                let mid = block[zi] as f32 * q;
                dequant[bi][nat] = mid;
                block_min[bi][nat] = mid - q * 0.5;
                block_max[bi][nat] = mid + q * 0.5;
            }
        }

        for iter in 0..Self::ITERATIONS {
            // Decreasing blend strength — conservative to avoid
            // over-smoothing at low quality where quant steps are large
            let alpha = 0.25 / (1.0 + iter as f32 * 0.3);

            // Snapshot for bilateral weight computation (avoid read-write race)
            let snapshot = dequant.clone();

            for bi in 0..num_blocks {
                let by = bi / blocks_wide;
                let bx = bi % blocks_wide;

                // Only smooth mid/high-frequency AC coefficients.
                // Low-frequency coefficients carry structural info and
                // shouldn't be averaged across blocks.
                for nat in 1..64 {
                    let row = nat / 8;
                    let col = nat % 8;
                    if row + col < 2 { continue; } // skip DC and two lowest AC
                    let q = quant_table[nat] as f32;
                    if q < 1.0 { continue; }
                    let center = snapshot[bi][nat];

                    // Bilateral range = 2.0 * quant step for this coefficient
                    let range = q * 2.0;

                    // Weighted average from 3×3 block neighborhood
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 { continue; }
                            let nx = bx as i32 + dx;
                            let ny = by as i32 + dy;
                            if nx < 0 || nx >= blocks_wide as i32
                                || ny < 0 || ny >= blocks_high as i32
                            {
                                continue;
                            }
                            let ni = ny as usize * blocks_wide + nx as usize;
                            let neighbor = snapshot[ni][nat];

                            // Bilateral weight: high when neighbor is similar
                            let diff = (center - neighbor).abs();
                            let t = (range - diff).max(0.0);
                            // Distance weight: cardinal=1.0, diagonal=0.707
                            let dw = if dx == 0 || dy == 0 { 1.0 } else {
                                1.0 / std::f32::consts::SQRT_2
                            };
                            let w = t * t * dw;

                            sum += neighbor * w;
                            weight_sum += w;
                        }
                    }

                    if weight_sum > 0.0 {
                        let smoothed = sum / weight_sum;
                        // Blend toward smoothed value
                        let new_val = center + alpha * (smoothed - center);
                        // Clamp to quantization interval
                        dequant[bi][nat] = new_val.clamp(
                            block_min[bi][nat], block_max[bi][nat],
                        );
                    }
                }
            }
        }

        // Final IDCT
        let mut output = vec![0.0f32; pw * ph];
        for bi in 0..num_blocks {
            let by = bi / blocks_wide;
            let bx = bi % blocks_wide;
            let pixels = zenjpeg::decode::idct::inverse_dct_8x8(&dequant[bi]);
            for row in 0..8 {
                for col in 0..8 {
                    output[(by * 8 + row) * pw + bx * 8 + col] =
                        pixels[row * 8 + col] + 128.0;
                }
            }
        }

        output
    }
}

impl DeblockStrategy for QuantSmoothBilateral {
    fn name(&self) -> &str {
        "quantsmooth_bilateral"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;

        let coeffs = Decoder::new()
            .decode_coefficients(jpeg_bytes, Unstoppable)
            .ok()?;

        let w = coeffs.width as usize;
        let h = coeffs.height as usize;

        let mut planes = Vec::with_capacity(3);
        for ci in 0..coeffs.components.len().min(3) {
            let comp = &coeffs.components[ci];
            let qt_idx = comp.quant_table_idx as usize;
            let qt = coeffs.quant_tables[qt_idx].as_ref()?;
            let bw = comp.blocks_wide;
            let bh = comp.blocks_high;
            let pw = bw * 8;
            let ph = bh * 8;

            let plane_data = Self::process_component(
                &comp.coeffs, bw, bh, qt,
            );

            planes.push(ComponentPlane {
                data: plane_data,
                width: pw,
                height: ph,
                blocks_wide: bw,
                blocks_high: bh,
                quant_table: *qt,
            });
        }

        let cp = CoeffPlanes {
            planes,
            image_width: w,
            image_height: h,
        };
        Some(planes_to_rgb(&cp))
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    generate: bool,
    measure: bool,
    corpus: String,
    max_images: usize,
    strategies: Vec<Box<dyn DeblockStrategy>>,
    verbose: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        generate: false,
        measure: false,
        corpus: "gb82+cid22".to_string(),
        max_images: usize::MAX,
        strategies: vec![
            Box::new(BaselineDecode),
            Box::new(DequantBiasDecode),
            Box::new(Boundary4Tap),
            Box::new(CDEFDirection),
            Box::new(Knusperli),
            Box::new(QuantSmoothBilateral),
            Box::new(CoeffSmooth),
            Box::new(CoeffRefineTV::new(4)),
        ],
        verbose: false,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--generate" => args.generate = true,
            "--measure" => args.measure = true,
            "--corpus" => {
                if let Some(s) = iter.next() {
                    args.corpus = s;
                }
            }
            "--images" => {
                args.max_images = iter
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(usize::MAX);
            }
            "--verbose" | "-v" => args.verbose = true,
            "--help" | "-h" => {
                eprintln!("Usage: deblock_harness [OPTIONS]");
                eprintln!("  --generate       Encode cached JPEGs (skip if already cached)");
                eprintln!("  --measure        Run measurements with deblock strategies");
                eprintln!("  --corpus <name>  gb82, cid22, gb82-sc, or gb82+cid22 (default)");
                eprintln!("  --images <N>     Max images per corpus");
                eprintln!("  --verbose        Per-image output");
                eprintln!();
                eprintln!("With no flags, runs both --generate and --measure.");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    // Default: both phases
    if !args.generate && !args.measure {
        args.generate = true;
        args.measure = true;
    }

    args
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

fn load_corpus_images(corpus_name: &str, max_images: usize) -> Vec<ImageData> {
    let cc = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    let mut images = Vec::new();

    let corpora: Vec<&str> = match corpus_name {
        "gb82+cid22" => vec!["gb82", "cid22"],
        other => vec![other],
    };

    for name in corpora {
        let dir = match name {
            "gb82" => cc.get("gb82").expect("gb82 not found"),
            "cid22" => cc
                .get("CID22")
                .expect("CID22 not found")
                .join("CID22-512/validation"),
            "gb82-sc" => cc.get("gb82-sc").expect("gb82-sc not found"),
            other => {
                eprintln!("Unknown corpus: {other}");
                continue;
            }
        };

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("cannot read corpus dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .is_some_and(|ext| ext == "png" || ext == "PNG")
            })
            .collect();
        paths.sort();

        for path in paths.into_iter().take(max_images) {
            match ImageData::from_path(&path) {
                Some(img) => images.push(img),
                None => eprintln!("  skip {}: load failed", path.display()),
            }
        }
    }

    images
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Encode with libjpeg-turbo cjpeg CLI (4:2:0). Returns JPEG bytes.
fn encode_turbo_420(ppm_path: &Path, quality: u8) -> Option<Vec<u8>> {
    let output = Command::new("cjpeg")
        .arg("-quality")
        .arg(quality.to_string())
        .arg(ppm_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(output.stdout)
}

/// Encode with mozjpeg-rs (in-process, progressive 4:2:0).
fn encode_mozjpeg_420(pixels: &[u8], w: usize, h: usize, quality: u8) -> Option<Vec<u8>> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w as u32, h as u32)
        .ok()
}

/// Encode with cjpegli CLI. Returns JPEG bytes.
fn encode_cjpegli(png_path: &Path, quality: u8, tmp_dir: &Path) -> Option<Vec<u8>> {
    let stem = png_path.file_stem().unwrap_or_default().to_string_lossy();
    let tmp_out = tmp_dir.join(format!("cjpegli_{stem}_q{quality}.jpg"));
    let output = Command::new("cjpegli")
        .arg(png_path)
        .arg(&tmp_out)
        .arg("-q")
        .arg(quality.to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let data = std::fs::read(&tmp_out).ok()?;
    std::fs::remove_file(&tmp_out).ok();
    Some(data)
}

/// Write PPM for cjpeg CLI input.
fn write_ppm(path: &Path, pixels: &[u8], w: usize, h: usize) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(pixels)?;
    Ok(())
}

/// Cache key: encoder/image_name_qXX.jpg
fn cache_path(encoder: Encoder, image_name: &str, quality: u8) -> PathBuf {
    Path::new(CACHE_DIR)
        .join("sources")
        .join(encoder.dir_name())
        .join(format!("{image_name}_q{quality}.jpg"))
}

/// Generate all cached encoded JPEGs. Skips files that already exist.
fn generate_cache(images: &[ImageData]) {
    let sources_dir = Path::new(CACHE_DIR).join("sources");

    // Create directories
    for enc in Encoder::all() {
        let dir = sources_dir.join(enc.dir_name());
        std::fs::create_dir_all(&dir).expect("cannot create cache dir");
    }

    let tmp_dir = Path::new(CACHE_DIR).join("tmp");
    std::fs::create_dir_all(&tmp_dir).expect("cannot create tmp dir");

    // Count what needs encoding
    let mut needed = 0u64;
    let mut cached = 0u64;
    for img in images {
        for &enc in Encoder::all() {
            for &q in &QUALITY_LEVELS {
                if cache_path(enc, &img.name, q).exists() {
                    cached += 1;
                } else {
                    needed += 1;
                }
            }
        }
    }

    eprintln!(
        "Cache: {cached} existing, {needed} to encode ({} images x {} encoders x {} qualities)",
        images.len(),
        Encoder::all().len(),
        QUALITY_LEVELS.len()
    );

    if needed == 0 {
        eprintln!("All JPEGs cached, nothing to encode.");
        return;
    }

    let progress = AtomicUsize::new(0);
    let total = needed as usize;
    let start = Instant::now();

    // Build work items
    struct EncodeJob {
        encoder: Encoder,
        quality: u8,
        image_idx: usize,
    }

    let mut jobs = Vec::new();
    for (i, img) in images.iter().enumerate() {
        for &enc in Encoder::all() {
            for &q in &QUALITY_LEVELS {
                if !cache_path(enc, &img.name, q).exists() {
                    jobs.push(EncodeJob {
                        encoder: enc,
                        quality: q,
                        image_idx: i,
                    });
                }
            }
        }
    }

    // Pre-write PPM files for turbo (cjpeg needs file input)
    let ppm_dir = tmp_dir.join("ppm");
    std::fs::create_dir_all(&ppm_dir).expect("cannot create ppm dir");

    let needs_ppm: Vec<usize> = jobs
        .iter()
        .filter(|j| j.encoder == Encoder::Turbo420)
        .map(|j| j.image_idx)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for &idx in &needs_ppm {
        let img = &images[idx];
        let ppm_path = ppm_dir.join(format!("{}.ppm", img.name));
        if !ppm_path.exists() {
            write_ppm(&ppm_path, &img.pixels, img.width, img.height)
                .unwrap_or_else(|e| panic!("write PPM {}: {e}", ppm_path.display()));
        }
    }

    // Encode in parallel
    jobs.par_iter().for_each(|job| {
        let img = &images[job.image_idx];
        let out_path = cache_path(job.encoder, &img.name, job.quality);

        let jpeg = match job.encoder {
            Encoder::Turbo420 => {
                let ppm_path = ppm_dir.join(format!("{}.ppm", img.name));
                encode_turbo_420(&ppm_path, job.quality)
            }
            Encoder::Mozjpeg420 => {
                encode_mozjpeg_420(&img.pixels, img.width, img.height, job.quality)
            }
            Encoder::Cjpegli => {
                // cjpegli needs PNG input — find the original
                let cc = codec_corpus::Corpus::new().unwrap();
                let png_path = find_source_png(&cc, &img.name);
                match png_path {
                    Some(p) => encode_cjpegli(&p, job.quality, &tmp_dir),
                    None => {
                        eprintln!("  cannot find PNG for {}", img.name);
                        None
                    }
                }
            }
        };

        if let Some(data) = jpeg {
            if let Err(e) = std::fs::write(&out_path, &data) {
                eprintln!("  write error {}: {e}", out_path.display());
            }
        } else {
            eprintln!(
                "  encode failed: {} q{} {}",
                img.name,
                job.quality,
                job.encoder.display_name()
            );
        }

        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 50 == 0 || done == total {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed;
            let remaining = (total - done) as f64 / rate;
            eprint!(
                "\r  Encoded {done}/{total} ({:.0}/s, {:.0}s remaining)    ",
                rate, remaining
            );
        }
    });

    eprintln!("\n  Done in {:.1}s", start.elapsed().as_secs_f64());

    // Cleanup PPMs
    std::fs::remove_dir_all(&ppm_dir).ok();
}

/// Find the original PNG source for an image name.
/// `name` may include the .png extension (e.g., "baby-lossless.png").
fn find_source_png(cc: &codec_corpus::Corpus, name: &str) -> Option<PathBuf> {
    // Strip .png/.PNG extension if present (ImageData.name includes extension)
    let stem = name
        .strip_suffix(".png")
        .or_else(|| name.strip_suffix(".PNG"))
        .unwrap_or(name);

    // Try gb82
    if let Ok(gb82) = cc.get("gb82") {
        let path = gb82.join(format!("{stem}.png"));
        if path.exists() {
            return Some(path);
        }
    }

    // Try CID22
    if let Ok(cid22) = cc.get("CID22") {
        let path = cid22.join(format!("CID22-512/validation/{stem}.png"));
        if path.exists() {
            return Some(path);
        }
        // Some CID22 images may use .PNG extension
        let path = cid22.join(format!("CID22-512/validation/{stem}.PNG"));
        if path.exists() {
            return Some(path);
        }
    }

    // Try gb82-sc
    if let Ok(sc) = cc.get("gb82-sc") {
        let path = sc.join(format!("{stem}.png"));
        if path.exists() {
            return Some(path);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Boundary discontinuity metric: mean |p0-q0| at 8x8 block boundaries.
/// Computed on the Y (luma) channel. Returns (smooth_mean, edge_mean, overall_mean).
fn boundary_discontinuity(img: &RgbImage) -> (f64, f64, f64) {
    let w = img.width();
    let h = img.height();
    let buf = img.buf();

    // Convert to Y channel (BT.601)
    let mut y = vec![0f32; w * h];
    for row in 0..h {
        for col in 0..w {
            let px = buf[row * img.stride() + col];
            y[row * w + col] = 0.299 * px.r as f32 + 0.587 * px.g as f32 + 0.114 * px.b as f32;
        }
    }

    let mut smooth_sum = 0.0f64;
    let mut smooth_count = 0u64;
    let mut edge_sum = 0.0f64;
    let mut edge_count = 0u64;

    // Edge threshold: if local gradient > this, it's an edge region
    let edge_thresh = 20.0f32;

    // Vertical boundaries (between columns 7-8, 15-16, etc.)
    for by in 0..h {
        for bx_idx in 1..(w / 8) {
            let col = bx_idx * 8;
            if col >= w {
                break;
            }
            let p0 = y[by * w + col - 1];
            let q0 = y[by * w + col];
            let disc = (p0 - q0).abs();

            // Check if this is an edge region: gradient magnitude around boundary
            let p1 = if col >= 2 { y[by * w + col - 2] } else { p0 };
            let q1 = if col + 1 < w { y[by * w + col + 1] } else { q0 };
            let grad = ((p0 - p1).abs() + (q1 - q0).abs()) * 0.5;

            if grad > edge_thresh {
                edge_sum += disc as f64;
                edge_count += 1;
            } else {
                smooth_sum += disc as f64;
                smooth_count += 1;
            }
        }
    }

    // Horizontal boundaries (between rows 7-8, 15-16, etc.)
    for bx in 0..w {
        for by_idx in 1..(h / 8) {
            let row = by_idx * 8;
            if row >= h {
                break;
            }
            let p0 = y[(row - 1) * w + bx];
            let q0 = y[row * w + bx];
            let disc = (p0 - q0).abs();

            let p1 = if row >= 2 { y[(row - 2) * w + bx] } else { p0 };
            let q1 = if row + 1 < h {
                y[(row + 1) * w + bx]
            } else {
                q0
            };
            let grad = ((p0 - p1).abs() + (q1 - q0).abs()) * 0.5;

            if grad > edge_thresh {
                edge_sum += disc as f64;
                edge_count += 1;
            } else {
                smooth_sum += disc as f64;
                smooth_count += 1;
            }
        }
    }

    let smooth_mean = if smooth_count > 0 {
        smooth_sum / smooth_count as f64
    } else {
        0.0
    };
    let edge_mean = if edge_count > 0 {
        edge_sum / edge_count as f64
    } else {
        0.0
    };
    let total = smooth_sum + edge_sum;
    let total_count = smooth_count + edge_count;
    let overall = if total_count > 0 {
        total / total_count as f64
    } else {
        0.0
    };

    (smooth_mean, edge_mean, overall)
}

#[derive(Debug)]
struct Measurement {
    image: String,
    encoder: Encoder,
    quality: u8,
    strategy: String,
    ssim2: f64,
    butteraugli: f64,
    boundary_smooth: f64,
    boundary_edge: f64,
    boundary_overall: f64,
    file_size: usize,
    detected_encoder: String,
    detected_quality: String,
}

/// Run measurements for all cached JPEGs with all strategies.
fn run_measurements(
    images: &[ImageData],
    strategies: &[Box<dyn DeblockStrategy>],
    verbose: bool,
) -> Vec<Measurement> {
    let results_dir = Path::new(RESULTS_DIR);
    std::fs::create_dir_all(results_dir).expect("cannot create results dir");

    // Build work items: (image_idx, encoder, quality)
    struct MeasureJob {
        image_idx: usize,
        encoder: Encoder,
        quality: u8,
    }

    let mut jobs = Vec::new();
    for (i, img) in images.iter().enumerate() {
        for &enc in Encoder::all() {
            for &q in &QUALITY_LEVELS {
                let path = cache_path(enc, &img.name, q);
                if path.exists() {
                    jobs.push(MeasureJob {
                        image_idx: i,
                        encoder: enc,
                        quality: q,
                    });
                }
            }
        }
    }

    eprintln!(
        "Measuring {} cached JPEGs x {} strategies = {} decode+measure ops",
        jobs.len(),
        strategies.len(),
        jobs.len() * strategies.len()
    );

    let progress = AtomicUsize::new(0);
    let total = jobs.len() * strategies.len();
    let start = Instant::now();

    // Measure in parallel over jobs (strategies are fast, parallelize over images)
    let measurements: Vec<Vec<Measurement>> = jobs
        .par_iter()
        .map(|job| {
            let img = &images[job.image_idx];
            let jpeg_path = cache_path(job.encoder, &img.name, job.quality);
            let jpeg_bytes = match std::fs::read(&jpeg_path) {
                Ok(b) => b,
                Err(_) => return vec![],
            };

            // Probe encoder detection
            let probe = detect::probe(&jpeg_bytes).ok();
            let detected_encoder = probe
                .as_ref()
                .map(|p| format!("{:?}", p.encoder))
                .unwrap_or_else(|| "ProbeError".to_string());
            let detected_quality = probe
                .as_ref()
                .map(|p| format!("{:?}", p.quality))
                .unwrap_or_else(|| "?".to_string());

            // Verify encoder detection
            if let Some(ref p) = probe {
                let expected = job.encoder.expected_family();
                let actual = p.encoder;
                if actual != expected {
                    eprintln!(
                        "  DETECTION MISMATCH: {} q{} {} — expected {:?}, got {:?}",
                        img.name,
                        job.quality,
                        job.encoder.display_name(),
                        expected,
                        actual
                    );
                }
            }

            // Build reference RgbImage from source pixels
            let reference = {
                use imgref::ImgVec;
                use rgb::RGB8;
                let px: Vec<RGB8> = img
                    .pixels
                    .chunks_exact(3)
                    .map(|c| RGB8 {
                        r: c[0],
                        g: c[1],
                        b: c[2],
                    })
                    .collect();
                ImgVec::new(px, img.width, img.height)
            };

            let mut results = Vec::with_capacity(strategies.len());

            for strategy in strategies {
                let decoded = match strategy.decode(&jpeg_bytes) {
                    Some(d) => d,
                    None => {
                        eprintln!(
                            "  decode failed: {} q{} {} [{}]",
                            img.name,
                            job.quality,
                            job.encoder.display_name(),
                            strategy.name()
                        );
                        continue;
                    }
                };

                // Quality metrics vs source
                let ssim2 = QualityMetrics::ssimulacra2(reference.as_ref(), decoded.as_ref());
                let ba = QualityMetrics::butteraugli(reference.as_ref(), decoded.as_ref());

                // Boundary discontinuity
                let (bsmooth, bedge, boverall) = boundary_discontinuity(&decoded);

                if verbose {
                    eprintln!(
                        "  {:<20} {:>10} q{:<3} [{:<13}] SS2={:6.2} BA={:5.2} BD={:.2}",
                        img.name,
                        job.encoder.display_name(),
                        job.quality,
                        strategy.name(),
                        ssim2,
                        ba,
                        bsmooth
                    );
                }

                results.push(Measurement {
                    image: img.name.clone(),
                    encoder: job.encoder,
                    quality: job.quality,
                    strategy: strategy.name().to_string(),
                    ssim2,
                    butteraugli: ba,
                    boundary_smooth: bsmooth,
                    boundary_edge: bedge,
                    boundary_overall: boverall,
                    file_size: jpeg_bytes.len(),
                    detected_encoder: detected_encoder.clone(),
                    detected_quality: detected_quality.clone(),
                });

                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 20 == 0 || done == total {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = done as f64 / elapsed;
                    let remaining = (total - done) as f64 / rate;
                    eprint!(
                        "\r  Measured {done}/{total} ({:.1}/s, {:.0}s remaining)    ",
                        rate, remaining
                    );
                }
            }

            results
        })
        .collect();

    let measurements: Vec<Measurement> = measurements.into_iter().flatten().collect();
    eprintln!(
        "\n  Done: {} measurements in {:.1}s",
        measurements.len(),
        start.elapsed().as_secs_f64()
    );

    measurements
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn write_csv(measurements: &[Measurement], path: &Path) {
    let mut f = std::fs::File::create(path).expect("cannot create CSV");
    writeln!(
        f,
        "image,encoder,quality,strategy,ssim2,butteraugli,bd_smooth,bd_edge,bd_overall,\
         file_size,detected_encoder,detected_quality"
    )
    .unwrap();

    for m in measurements {
        writeln!(
            f,
            "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{}",
            m.image,
            m.encoder.dir_name(),
            m.quality,
            m.strategy,
            m.ssim2,
            m.butteraugli,
            m.boundary_smooth,
            m.boundary_edge,
            m.boundary_overall,
            m.file_size,
            m.detected_encoder,
            m.detected_quality,
        )
        .unwrap();
    }
}

/// Print summary table: mean metrics per encoder × quality × strategy.
fn print_summary(measurements: &[Measurement]) {
    // Group by (strategy, encoder, quality) → collect metrics
    let mut groups: BTreeMap<(String, Encoder, u8), Vec<(f64, f64, f64)>> = BTreeMap::new();

    for m in measurements {
        groups
            .entry((m.strategy.clone(), m.encoder, m.quality))
            .or_default()
            .push((m.ssim2, m.butteraugli, m.boundary_smooth));
    }

    // Get all strategies
    let strategies: Vec<String> = measurements
        .iter()
        .map(|m| m.strategy.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for strategy in &strategies {
        eprintln!("\n=== Strategy: {} ===", strategy);
        eprintln!(
            "{:<14} {:>3}  {:>7} {:>7} {:>7}  {:>3}",
            "Encoder", "Q", "SS2", "BA", "BD_sm", "N"
        );
        eprintln!("{}", "-".repeat(52));

        for &enc in Encoder::all() {
            for &q in &QUALITY_LEVELS {
                if let Some(vals) = groups.get(&(strategy.clone(), enc, q)) {
                    let n = vals.len();
                    let mean_ss2: f64 = vals.iter().map(|v| v.0).sum::<f64>() / n as f64;
                    let mean_ba: f64 = vals.iter().map(|v| v.1).sum::<f64>() / n as f64;
                    let mean_bd: f64 = vals.iter().map(|v| v.2).sum::<f64>() / n as f64;

                    eprintln!(
                        "{:<14} {:>3}  {:>7.2} {:>7.2} {:>7.2}  {:>3}",
                        enc.dir_name(),
                        q,
                        mean_ss2,
                        mean_ba,
                        mean_bd,
                        n
                    );
                }
            }
        }
    }

    // Delta table: strategy improvements over baseline
    if strategies.len() > 1 {
        eprintln!("\n=== Deltas vs baseline ===");
        for strategy in strategies.iter().filter(|s| *s != "baseline") {
            eprintln!("\n--- {} vs baseline ---", strategy);
            eprintln!(
                "{:<14} {:>3}  {:>8} {:>8} {:>8}",
                "Encoder", "Q", "dSS2", "dBA", "dBD_sm"
            );
            eprintln!("{}", "-".repeat(55));

            for &enc in Encoder::all() {
                for &q in &QUALITY_LEVELS {
                    let baseline_key = ("baseline".to_string(), enc, q);
                    let strategy_key = (strategy.clone(), enc, q);

                    if let (Some(base), Some(strat)) =
                        (groups.get(&baseline_key), groups.get(&strategy_key))
                    {
                        let n = base.len().min(strat.len());
                        let d_ss2 = strat.iter().map(|v| v.0).sum::<f64>() / n as f64
                            - base.iter().map(|v| v.0).sum::<f64>() / n as f64;
                        let d_ba = strat.iter().map(|v| v.1).sum::<f64>() / n as f64
                            - base.iter().map(|v| v.1).sum::<f64>() / n as f64;
                        let d_bd = strat.iter().map(|v| v.2).sum::<f64>() / n as f64
                            - base.iter().map(|v| v.2).sum::<f64>() / n as f64;

                        eprintln!(
                            "{:<14} {:>3}  {:>+8.3} {:>+8.3} {:>+8.3}",
                            enc.dir_name(),
                            q,
                            d_ss2,
                            d_ba,
                            d_bd,
                        );
                    }
                }
            }
        }
    }

    // Detection accuracy
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut mismatches = Vec::new();
    // Deduplicate: check once per (encoder, quality, image)
    let mut seen = std::collections::HashSet::new();
    for m in measurements {
        if m.strategy != "baseline" {
            continue;
        }
        let key = (m.encoder, m.quality, m.image.clone());
        if !seen.insert(key) {
            continue;
        }
        total += 1;
        let expected = format!("{:?}", m.encoder.expected_family());
        if m.detected_encoder == expected {
            correct += 1;
        } else {
            mismatches.push(format!(
                "  {} q{} {}: expected {}, got {}",
                m.image,
                m.quality,
                m.encoder.dir_name(),
                expected,
                m.detected_encoder
            ));
        }
    }

    eprintln!("\n=== Encoder Detection ===");
    eprintln!("Correct: {correct}/{total}");
    if !mismatches.is_empty() {
        eprintln!("Mismatches:");
        for mm in &mismatches[..mismatches.len().min(20)] {
            eprintln!("{mm}");
        }
        if mismatches.len() > 20 {
            eprintln!("  ... and {} more", mismatches.len() - 20);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    eprintln!("Loading corpus: {}", args.corpus);
    let images = load_corpus_images(&args.corpus, args.max_images);
    eprintln!("Loaded {} images", images.len());

    if images.is_empty() {
        eprintln!("No images found!");
        return;
    }

    if args.generate {
        eprintln!("\n--- Phase 1: Generate cached JPEGs ---");
        generate_cache(&images);
    }

    if args.measure {
        eprintln!("\n--- Phase 2: Measure with strategies ---");
        let measurements = run_measurements(&images, &args.strategies, args.verbose);

        if measurements.is_empty() {
            eprintln!("No measurements collected! Run --generate first.");
            return;
        }

        // Write CSV
        let csv_path = Path::new(RESULTS_DIR).join("baseline.csv");
        write_csv(&measurements, &csv_path);
        eprintln!("\nCSV written to: {}", csv_path.display());

        // Print summary
        print_summary(&measurements);
    }
}
