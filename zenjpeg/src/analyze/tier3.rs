//! Tier 3: luma histogram entropy + high-frequency DCT energy ratio
//! + derived likelihoods (text / screen-content / natural).
//!
//! Direct ports of `coefficient::analysis::evalchroma_ext` (Tier 3)
//! and `image_adaptive::compute_derived_likelihoods`. Numbers are
//! kept identical so the parity example matches within an f32
//! epsilon.

use super::AnalyzerOutput;
use super::row_stream::RowStream;

/// Cap on sampled 8×8 luma blocks for `high_freq_energy_ratio`.
/// 256 blocks ≈ 1 ms even at 4K with the naive O(N⁴) DCT.
const HF_MAX_BLOCKS: usize = 256;

/// Fill in `high_freq_energy_ratio` and `luma_histogram_entropy`.
pub fn populate_tier3(out: &mut AnalyzerOutput, stream: &mut RowStream<'_>) {
    out.luma_histogram_entropy = luma_histogram_entropy(stream);
    out.high_freq_energy_ratio = high_freq_energy_ratio(stream, HF_MAX_BLOCKS);
}

/// Shannon entropy of a 32-bin BT.601 luma histogram, in bits.
/// Range `[0, log2(32)] = [0, 5]`. Samples every 4th pixel in raster
/// order — exactly matching the reference's `rgb.chunks_exact(3).
/// step_by(4)` semantics, including the cross-row stride carry when
/// `width % 4 != 0`.
fn luma_histogram_entropy(stream: &mut RowStream<'_>) -> f32 {
    let width = stream.width() as usize;
    let height = stream.height() as usize;
    if width == 0 || height == 0 {
        return 0.0;
    }
    let mut bins = [0u32; 32];
    let mut n = 0u32;
    let mut carry: usize = 0; // pixel index modulo 4 entering this row
    for yy in 0..height {
        let row = stream.borrow_row(yy as u32);
        // First sampled pixel in this row is at index `start`; subsequent ones at +4.
        let start = (4 - carry) % 4;
        let mut x = start;
        while x < width {
            let off = x * 3;
            let p = &row[off..off + 3];
            let y = ((66 * p[0] as u32 + 129 * p[1] as u32 + 25 * p[2] as u32 + 128) >> 8) as u8;
            bins[(y >> 3) as usize] += 1;
            n += 1;
            x += 4;
        }
        carry = (carry + width) % 4;
    }
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f32;
    let mut h = 0.0f32;
    for &c in &bins {
        if c > 0 {
            let p = c as f32 / n_f;
            h -= p * p.log2();
        }
    }
    h
}

/// Ratio of high-frequency to low-frequency AC DCT energy on sampled
/// 8×8 luma blocks. `Σ AC[k≥16] / max(1, Σ AC[k∈1..16])`. Naive
/// separable 1D DCT — exactness isn't required for a feature.
///
/// Pulls 8 rows at a time (one block-row's worth) and samples the
/// `bx` columns selected by `block_idx % stride`. Keeps memory at
/// 8 × width × 3 bytes regardless of image size.
fn high_freq_energy_ratio(stream: &mut RowStream<'_>, max_blocks: usize) -> f32 {
    let width = stream.width() as usize;
    let height = stream.height() as usize;
    if width < 8 || height < 8 {
        return 0.0;
    }
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let total_blocks = blocks_x * blocks_y;
    if total_blocks == 0 {
        return 0.0;
    }
    let stride = (total_blocks / max_blocks).max(1);

    let mut low_energy = 0.0f64;
    let mut high_energy = 0.0f64;
    let row_bytes = width * 3;
    let mut block_buf = vec![0u8; 8 * row_bytes]; // 8 rows of one block-row
    let mut block_idx = 0usize;

    for by in 0..blocks_y {
        // Determine whether any sampled block lives in this block-row.
        let row_start = by * blocks_x;
        let row_end = row_start + blocks_x;
        let any_sampled = (row_start..row_end).any(|k| k % stride == 0);
        if !any_sampled {
            block_idx += blocks_x;
            continue;
        }

        // Pull 8 contiguous rows for the block-row.
        for i in 0..8 {
            stream.fetch_into(
                (by * 8 + i) as u32,
                &mut block_buf[i * row_bytes..(i + 1) * row_bytes],
            );
        }

        for bx in 0..blocks_x {
            if block_idx % stride != 0 {
                block_idx += 1;
                continue;
            }
            block_idx += 1;

            let mut blk = [[0.0f32; 8]; 8];
            for y in 0..8 {
                let row = &block_buf[y * row_bytes..(y + 1) * row_bytes];
                for x in 0..8 {
                    let off = (bx * 8 + x) * 3;
                    let p = &row[off..off + 3];
                    let l = (66 * p[0] as u32 + 129 * p[1] as u32 + 25 * p[2] as u32 + 128) >> 8;
                    blk[y][x] = l as f32 - 128.0;
                }
            }

            let mut after_rows = [[0.0f32; 8]; 8];
            for y in 0..8 {
                for u in 0..8 {
                    let cu = if u == 0 {
                        core::f32::consts::FRAC_1_SQRT_2
                    } else {
                        1.0
                    };
                    let mut s = 0.0f32;
                    for x in 0..8 {
                        s += blk[y][x]
                            * ((core::f32::consts::PI * (2.0 * x as f32 + 1.0) * u as f32) / 16.0)
                                .cos();
                    }
                    after_rows[y][u] = 0.5 * cu * s;
                }
            }
            let mut coeffs = [[0.0f32; 8]; 8];
            for u in 0..8 {
                for v in 0..8 {
                    let cv = if v == 0 {
                        core::f32::consts::FRAC_1_SQRT_2
                    } else {
                        1.0
                    };
                    let mut s = 0.0f32;
                    for y in 0..8 {
                        s += after_rows[y][u]
                            * ((core::f32::consts::PI * (2.0 * y as f32 + 1.0) * v as f32) / 16.0)
                                .cos();
                    }
                    coeffs[v][u] = 0.5 * cv * s;
                }
            }

            for k in 1..64 {
                let u = k % 8;
                let v = k / 8;
                let e = (coeffs[v][u] * coeffs[v][u]) as f64;
                if k < 16 {
                    low_energy += e;
                } else {
                    high_energy += e;
                }
            }
        }
    }

    if low_energy < 1e-6 {
        return 0.0;
    }
    (high_energy / low_energy) as f32
}

/// Populate the three derived likelihood scores. Run after Tier 1+2+3
/// numerics are filled.
pub fn compute_derived_likelihoods(out: &mut AnalyzerOutput) {
    let chroma_sh = out.cb_sharpness + out.cr_sharpness;

    let entropy_low = (4.0 - out.luma_histogram_entropy).clamp(0.0, 4.0) / 4.0;
    let edge_hi = (out.edge_density / 0.25).min(1.0);
    let chroma_lo = (0.005 - chroma_sh).clamp(0.0, 0.005) / 0.005;
    out.text_likelihood = (entropy_low * 0.4 + edge_hi * 0.3 + chroma_lo * 0.3).clamp(0.0, 1.0);

    let palette_small = if out.distinct_color_bins == 0 {
        0.0
    } else {
        (1.0 - (out.distinct_color_bins as f32 / 4000.0).min(1.0)).clamp(0.0, 1.0)
    };
    let flat_high = (out.flat_color_block_ratio / 0.5).min(1.0);
    out.screen_content_likelihood =
        (flat_high * 0.6 + palette_small * 0.3 + chroma_lo * 0.1).clamp(0.0, 1.0);

    let entropy_hi = (out.luma_histogram_entropy - 3.5).clamp(0.0, 1.5) / 1.5;
    let palette_large = if out.distinct_color_bins < 2000 {
        0.0
    } else {
        ((out.distinct_color_bins as f32 - 2000.0) / 8000.0).clamp(0.0, 1.0)
    };
    let chroma_moderate = (chroma_sh / 0.012).min(1.0);
    let not_flat = (1.0 - (out.flat_color_block_ratio / 0.3).min(1.0)).clamp(0.0, 1.0);
    out.natural_likelihood =
        (entropy_hi * 0.3 + palette_large * 0.25 + chroma_moderate * 0.2 + not_flat * 0.25)
            .clamp(0.0, 1.0);
}
