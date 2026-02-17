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
    /// Forward DCT for a single 8x8 block (reference implementation).
    /// Output is scaled to match zenjpeg's IDCT convention: forward_dct * 8 → inverse_dct = identity.
    /// (The raw DCT produces 1/64 scale; IDCT expects 1/8 scale. The ×8 bridges this gap.)
    fn forward_dct_8x8(input: &[f32; 64]) -> [f32; 64] {
        // Separable 1D DCT: rows then columns
        let mut temp = [0.0f32; 64];
        let mut output = [0.0f32; 64];

        // DCT on rows
        for row in 0..8 {
            for k in 0..8 {
                let mut sum = 0.0f32;
                for n in 0..8 {
                    sum += input[row * 8 + n]
                        * ((2 * n + 1) as f32 * k as f32 * std::f32::consts::PI / 16.0).cos();
                }
                let ck = if k == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 };
                temp[row * 8 + k] = sum * ck * 0.5;
            }
        }

        // DCT on columns
        for col in 0..8 {
            for k in 0..8 {
                let mut sum = 0.0f32;
                for n in 0..8 {
                    sum += temp[n * 8 + col]
                        * ((2 * n + 1) as f32 * k as f32 * std::f32::consts::PI / 16.0).cos();
                }
                let ck = if k == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 };
                output[k * 8 + col] = sum * ck * 0.5;
            }
        }

        // Scale ×8 to match IDCT convention (raw DCT is 1/64 scale, IDCT expects 1/8)
        for v in &mut output {
            *v *= 8.0;
        }

        output
    }

    /// Compute boundary cost: sum of squared differences at block boundary.
    fn boundary_cost_h(
        left_pixels: &[f32; 64], right_pixels: &[f32; 64],
    ) -> f32 {
        // Vertical boundary: compare column 7 of left with column 0 of right
        let mut cost = 0.0f32;
        for row in 0..8 {
            let diff = left_pixels[row * 8 + 7] - right_pixels[row * 8];
            cost += diff * diff;
        }
        cost
    }

    fn boundary_cost_v(
        top_pixels: &[f32; 64], bottom_pixels: &[f32; 64],
    ) -> f32 {
        // Horizontal boundary: compare row 7 of top with row 0 of bottom
        let mut cost = 0.0f32;
        for col in 0..8 {
            let diff = top_pixels[7 * 8 + col] - bottom_pixels[col];
            cost += diff * diff;
        }
        cost
    }

    /// Adjust coefficients for a component to minimize boundary discontinuity.
    fn smooth_component(
        coeffs: &mut [i16],
        blocks_wide: usize,
        blocks_high: usize,
        quant_table: &[u16; 64],
    ) {
        // Coefficients are stored in zigzag order. We adjust the first few zigzag
        // positions (lowest frequencies) that most affect block boundaries.
        // Zigzag positions 1-3 correspond to the 3 lowest-frequency AC coefficients.
        // Conservative: only 3 positions to limit cascading modifications.
        let ac_positions: [usize; 3] = [1, 2, 3];

        // For each pair of horizontally adjacent blocks
        for by in 0..blocks_high {
            for bx in 0..blocks_wide.saturating_sub(1) {
                let left_idx = by * blocks_wide + bx;
                let right_idx = by * blocks_wide + bx + 1;

                // Get current pixels for both blocks
                let left_arr: [i16; 64] = coeffs[left_idx * 64..(left_idx + 1) * 64]
                    .try_into().unwrap();
                let right_arr: [i16; 64] = coeffs[right_idx * 64..(right_idx + 1) * 64]
                    .try_into().unwrap();

                let left_dq = dequantize_unzigzag(&left_arr, quant_table);
                let right_dq = dequantize_unzigzag(&right_arr, quant_table);

                let mut left_pixels = zenjpeg::decode::idct::inverse_dct_8x8(&left_dq);
                let mut right_pixels = zenjpeg::decode::idct::inverse_dct_8x8(&right_dq);

                // Level shift
                for p in left_pixels.iter_mut() { *p += 128.0; }
                for p in right_pixels.iter_mut() { *p += 128.0; }

                let mut current_cost = Self::boundary_cost_h(&left_pixels, &right_pixels);
                if current_cost < 8.0 {
                    continue; // Already smooth enough
                }

                // Try adjusting each AC coefficient by ±1 quantization step
                for &ac_pos in &ac_positions {
                    let q = quant_table[ac_pos];
                    if q == 0 { continue; }

                    // Try adjusting left block, then right block
                    for side in 0..2 {
                        let block_idx = if side == 0 { left_idx } else { right_idx };
                        let orig_val = coeffs[block_idx * 64 + ac_pos];

                        for delta in [-1i16, 1] {
                            let new_val = orig_val + delta;

                            coeffs[block_idx * 64 + ac_pos] = new_val;

                            // Recompute pixels
                            let l_arr: [i16; 64] = coeffs[left_idx * 64..(left_idx + 1) * 64]
                                .try_into().unwrap();
                            let r_arr: [i16; 64] = coeffs[right_idx * 64..(right_idx + 1) * 64]
                                .try_into().unwrap();

                            let l_dq = dequantize_unzigzag(&l_arr, quant_table);
                            let r_dq = dequantize_unzigzag(&r_arr, quant_table);

                            let mut lp = zenjpeg::decode::idct::inverse_dct_8x8(&l_dq);
                            let mut rp = zenjpeg::decode::idct::inverse_dct_8x8(&r_dq);
                            for p in lp.iter_mut() { *p += 128.0; }
                            for p in rp.iter_mut() { *p += 128.0; }

                            let new_cost = Self::boundary_cost_h(&lp, &rp);

                            if new_cost < current_cost * 0.7 {
                                // Keep the change — update cost baseline
                                current_cost = new_cost;
                                break;
                            } else {
                                coeffs[block_idx * 64 + ac_pos] = orig_val;
                            }
                        }
                    }
                }
            }
        }

        // Same for vertically adjacent blocks
        for by in 0..blocks_high.saturating_sub(1) {
            for bx in 0..blocks_wide {
                let top_idx = by * blocks_wide + bx;
                let bot_idx = (by + 1) * blocks_wide + bx;

                let top_arr: [i16; 64] = coeffs[top_idx * 64..(top_idx + 1) * 64]
                    .try_into().unwrap();
                let bot_arr: [i16; 64] = coeffs[bot_idx * 64..(bot_idx + 1) * 64]
                    .try_into().unwrap();

                let top_dq = dequantize_unzigzag(&top_arr, quant_table);
                let bot_dq = dequantize_unzigzag(&bot_arr, quant_table);

                let mut top_pixels = zenjpeg::decode::idct::inverse_dct_8x8(&top_dq);
                let mut bot_pixels = zenjpeg::decode::idct::inverse_dct_8x8(&bot_dq);
                for p in top_pixels.iter_mut() { *p += 128.0; }
                for p in bot_pixels.iter_mut() { *p += 128.0; }

                let mut current_cost = Self::boundary_cost_v(&top_pixels, &bot_pixels);
                if current_cost < 8.0 {
                    continue;
                }

                for &ac_pos in &ac_positions {
                    let q = quant_table[ac_pos];
                    if q == 0 { continue; }

                    for side in 0..2 {
                        let block_idx = if side == 0 { top_idx } else { bot_idx };
                        let orig_val = coeffs[block_idx * 64 + ac_pos];

                        for delta in [-1i16, 1] {
                            let new_val = orig_val + delta;
                            coeffs[block_idx * 64 + ac_pos] = new_val;

                            let t_arr: [i16; 64] = coeffs[top_idx * 64..(top_idx + 1) * 64]
                                .try_into().unwrap();
                            let b_arr: [i16; 64] = coeffs[bot_idx * 64..(bot_idx + 1) * 64]
                                .try_into().unwrap();

                            let t_dq = dequantize_unzigzag(&t_arr, quant_table);
                            let b_dq = dequantize_unzigzag(&b_arr, quant_table);

                            let mut tp = zenjpeg::decode::idct::inverse_dct_8x8(&t_dq);
                            let mut bp = zenjpeg::decode::idct::inverse_dct_8x8(&b_dq);
                            for p in tp.iter_mut() { *p += 128.0; }
                            for p in bp.iter_mut() { *p += 128.0; }

                            let new_cost = Self::boundary_cost_v(&tp, &bp);

                            if new_cost < current_cost * 0.7 {
                                // Keep the change — update cost baseline
                                current_cost = new_cost;
                                break;
                            } else {
                                coeffs[block_idx * 64 + ac_pos] = orig_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl DeblockStrategy for CoeffSmooth {
    fn name(&self) -> &str {
        "coeff_smooth"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;

        let mut coeffs = Decoder::new()
            .decode_coefficients(jpeg_bytes, Unstoppable)
            .ok()?;

        // Smooth each component
        for ci in 0..coeffs.components.len().min(3) {
            let qt_idx = coeffs.components[ci].quant_table_idx as usize;
            let qt = *coeffs.quant_tables[qt_idx].as_ref()?;
            let bw = coeffs.components[ci].blocks_wide;
            let bh = coeffs.components[ci].blocks_high;
            Self::smooth_component(
                &mut coeffs.components[ci].coeffs,
                bw, bh, &qt,
            );
        }

        // Reconstruct from modified coefficients
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

            let mut plane = vec![0.0f32; pw * ph];
            for by in 0..bh {
                for bx in 0..bw {
                    let block_zigzag = comp.block_at(bx, by);
                    let block_arr: [i16; 64] = block_zigzag.try_into().unwrap();
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

        let cp = CoeffPlanes {
            planes,
            image_width: w,
            image_height: h,
        };
        Some(planes_to_rgb(&cp))
    }
}

// ---------------------------------------------------------------------------
// Strategy: Iterative coefficient refinement with TV regularization (POCS)
// ---------------------------------------------------------------------------

/// Iterative coefficient refinement within quantization intervals.
/// Uses Projection Onto Convex Sets (POCS) approach:
/// 1. Reconstruct pixels from coefficients
/// 2. Apply total variation denoising to smooth block boundaries
/// 3. Forward DCT the smoothed pixels
/// 4. Project back onto quantization intervals (clamp each coefficient)
/// 5. Repeat
struct CoeffRefineTV {
    iterations: usize,
}

impl CoeffRefineTV {
    fn new(iterations: usize) -> Self {
        Self { iterations }
    }

    /// Project coefficients onto quantization intervals.
    /// Each coefficient c must stay within [original*q - q/2, original*q + q/2]
    /// in the dequantized domain. In the quantized domain, this means the
    /// coefficient can only change by ±0 (it stays at the same quantized level).
    /// But we allow fractional adjustment by working in the dequantized domain.
    fn project_onto_intervals(
        new_dct: &[f32; 64],
        original_coeffs: &[i16; 64],
        quant_table: &[u16; 64],
    ) -> [f32; 64] {
        // zigzag → natural order mapping
        let zigzag_to_natural: [usize; 64] = [
            0,  1,  8, 16,  9,  2,  3, 10,
            17, 24, 32, 25, 18, 11,  4,  5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13,  6,  7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63,
        ];

        let mut projected = [0.0f32; 64];

        for zi in 0..64 {
            let ni = zigzag_to_natural[zi];
            // quant_table is in NATURAL order (parser converts during DQT parsing)
            let q = quant_table[ni] as f32;
            let original_quantized = original_coeffs[zi] as f32;

            // Quantization interval in dequantized domain:
            // center = original_quantized * q
            // valid range = [center - q/2, center + q/2]
            let center = original_quantized * q;
            let lo = center - q * 0.5;
            let hi = center + q * 0.5;

            // new_dct is in natural order — project onto interval
            projected[ni] = new_dct[ni].clamp(lo, hi);
        }

        projected
    }

    /// Apply TV smoothing to boundary pixels only.
    /// Returns smoothed plane (does not modify original).
    fn tv_smooth_boundaries(plane: &[f32], w: usize, h: usize, lambda: f32) -> Vec<f32> {
        let mut smoothed = plane.to_vec();

        // Only smooth pixels immediately adjacent to block boundaries
        for y in 0..h {
            for x in 0..w {
                let at_v = (x % 8) == 0 || (x % 8) == 7;
                let at_h = (y % 8) == 0 || (y % 8) == 7;
                if !at_v && !at_h {
                    continue;
                }

                let center = plane[y * w + x];
                let mut grad = 0.0f32;
                let mut count = 0.0f32;

                // 4-connected neighbors
                if x > 0 {
                    grad += plane[y * w + x - 1] - center;
                    count += 1.0;
                }
                if x + 1 < w {
                    grad += plane[y * w + x + 1] - center;
                    count += 1.0;
                }
                if y > 0 {
                    grad += plane[(y - 1) * w + x] - center;
                    count += 1.0;
                }
                if y + 1 < h {
                    grad += plane[(y + 1) * w + x] - center;
                    count += 1.0;
                }

                if count > 0.0 {
                    smoothed[y * w + x] = center + lambda * grad / count;
                }
            }
        }

        smoothed
    }

    /// Run POCS iteration on a single component.
    fn refine_component(
        coeffs: &[i16],
        blocks_wide: usize,
        blocks_high: usize,
        quant_table: &[u16; 64],
        iterations: usize,
    ) -> Vec<f32> {
        let pw = blocks_wide * 8;
        let ph = blocks_high * 8;

        // Store original coefficients for projection
        let num_blocks = blocks_wide * blocks_high;
        let mut current_dequant = vec![[0.0f32; 64]; num_blocks];

        // Initial dequantization + IDCT
        for bi in 0..num_blocks {
            let block_arr: [i16; 64] = coeffs[bi * 64..(bi + 1) * 64]
                .try_into().unwrap();
            current_dequant[bi] = dequantize_unzigzag(&block_arr, quant_table);
        }

        // Compute adaptive lambda from quant table — more quantization = more smoothing
        let dc_quant = quant_table[0] as f32;
        let lambda = (dc_quant / 40.0).clamp(0.02, 0.2);

        for _iter in 0..iterations {
            // 1. IDCT all blocks to pixel plane
            let mut plane = vec![0.0f32; pw * ph];
            for by in 0..blocks_high {
                for bx in 0..blocks_wide {
                    let bi = by * blocks_wide + bx;
                    let pixels = zenjpeg::decode::idct::inverse_dct_8x8(&current_dequant[bi]);
                    for row in 0..8 {
                        for col in 0..8 {
                            plane[(by * 8 + row) * pw + bx * 8 + col] = pixels[row * 8 + col] + 128.0;
                        }
                    }
                }
            }

            // 2. TV-smooth the boundary pixels
            let smoothed = Self::tv_smooth_boundaries(&plane, pw, ph, lambda);

            // 3. Forward DCT the smoothed blocks + project onto quantization intervals
            for by in 0..blocks_high {
                for bx in 0..blocks_wide {
                    let bi = by * blocks_wide + bx;

                    // Extract smoothed 8x8 block
                    let mut block = [0.0f32; 64];
                    for row in 0..8 {
                        for col in 0..8 {
                            block[row * 8 + col] = smoothed[(by * 8 + row) * pw + bx * 8 + col] - 128.0;
                        }
                    }

                    // Forward DCT
                    let new_dct = CoeffSmooth::forward_dct_8x8(&block);

                    // Project onto quantization intervals
                    let original_block: [i16; 64] = coeffs[bi * 64..(bi + 1) * 64]
                        .try_into().unwrap();
                    current_dequant[bi] = Self::project_onto_intervals(
                        &new_dct, &original_block, quant_table,
                    );
                }
            }
        }

        // Final reconstruction
        let mut plane = vec![0.0f32; pw * ph];
        for by in 0..blocks_high {
            for bx in 0..blocks_wide {
                let bi = by * blocks_wide + bx;
                let pixels = zenjpeg::decode::idct::inverse_dct_8x8(&current_dequant[bi]);
                for row in 0..8 {
                    for col in 0..8 {
                        plane[(by * 8 + row) * pw + bx * 8 + col] = pixels[row * 8 + col] + 128.0;
                    }
                }
            }
        }

        plane
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

            let plane_data = Self::refine_component(
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
            Box::new(CoeffSmooth),
            Box::new(CoeffRefineTV::new(2)),
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
