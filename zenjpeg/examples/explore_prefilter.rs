//! Exploration: pre-encode denoising, noise-aware zero-bias, and perceptual loops.
//!
//! Tests three compression improvement strategies:
//! 1. Pre-encode noise-gated smoothing (reduce source entropy in flat regions)
//! 2. Noise-aware zero-bias (per-block noise energy → modulate zero_bias_mul)
//! 3. Encode-decode-measure loop (perceptual feedback for zero-bias redistribution)
//!
//! Also benchmarks XYB with per-component zero-bias vs the hardcoded 0.5.
//!
//! Usage:
//!   cargo run --release --example explore_prefilter -- [image.png ...]
//!   cargo run --release --example explore_prefilter -- ~/work/codec-corpus/cid22/*.png

use std::env;
use std::path::Path;
use std::time::Instant;

use zenjpeg::encoder::{EncoderConfig, PixelLayout, Quality, XybSubsampling};

// ============================================================================
// Gaussian blur (separable, f32, single-channel)
// ============================================================================

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0f32; size];
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0f32;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        kernel[i] = (-x * x * inv_2s2).exp();
        sum += kernel[i];
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Separable 2D Gaussian blur on a single f32 plane.
/// `buf` is width×height in row-major order.
fn gaussian_blur_2d(buf: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;

    // Horizontal pass
    let mut h_out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for ki in 0..kernel.len() {
                let sx = (x as isize + ki as isize - radius as isize)
                    .clamp(0, width as isize - 1) as usize;
                sum += buf[y * width + sx] * kernel[ki];
            }
            h_out[y * width + x] = sum;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for ki in 0..kernel.len() {
                let sy = (y as isize + ki as isize - radius as isize)
                    .clamp(0, height as isize - 1) as usize;
                sum += h_out[sy * width + x] * kernel[ki];
            }
            out[y * width + x] = sum;
        }
    }
    out
}

// ============================================================================
// Idea 1: Pre-encode noise-gated smoothing
// ============================================================================

/// Noise-gated smoothing on an f32 plane (one channel).
///
/// Smooths flat regions where noise dominates, preserves textured regions.
/// Based on zenfilter's AdaptiveSharpen noise gate pattern.
fn noise_gated_smooth(
    plane: &[f32],
    width: usize,
    height: usize,
    sigma: f32,
    noise_floor: f32,
) -> Vec<f32> {
    // 1. Blur to get local average
    let blurred = gaussian_blur_2d(plane, width, height, sigma);

    // 2. Compute detail (high-frequency content)
    let mut detail_sq = vec![0.0f32; width * height];
    for i in 0..plane.len() {
        let d = plane[i] - blurred[i];
        detail_sq[i] = d * d;
    }

    // 3. Estimate local energy = blur(detail^2, sigma*3)
    let energy = gaussian_blur_2d(&detail_sq, width, height, sigma * 3.0);

    // 4. Apply noise gate: gate = sqrt(energy) / (sqrt(energy) + noise_floor)
    //    gate ≈ 0 → flat region → use blurred (smooth)
    //    gate ≈ 1 → textured → keep original
    let mut out = vec![0.0f32; width * height];
    for i in 0..plane.len() {
        let e = energy[i].sqrt();
        let gate = e / (e + noise_floor);
        out[i] = gate * plane[i] + (1.0 - gate) * blurred[i];
    }
    out
}

/// Apply noise-gated smoothing to RGB pixels (operates per-channel).
/// Returns modified pixel buffer. `strength` scales noise_floor inversely.
fn prefilter_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    sigma: f32,
    noise_floor: f32,
) -> Vec<u8> {
    let npix = width * height;

    // Separate into channels (f32, 0..255 range)
    let mut r = vec![0.0f32; npix];
    let mut g = vec![0.0f32; npix];
    let mut b = vec![0.0f32; npix];
    for i in 0..npix {
        r[i] = pixels[i * 3] as f32;
        g[i] = pixels[i * 3 + 1] as f32;
        b[i] = pixels[i * 3 + 2] as f32;
    }

    // Apply noise-gated smoothing to each channel
    let r_f = noise_gated_smooth(&r, width, height, sigma, noise_floor);
    let g_f = noise_gated_smooth(&g, width, height, sigma, noise_floor);
    let b_f = noise_gated_smooth(&b, width, height, sigma, noise_floor);

    // Recombine
    let mut out = vec![0u8; npix * 3];
    for i in 0..npix {
        out[i * 3] = r_f[i].round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = g_f[i].round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = b_f[i].round().clamp(0.0, 255.0) as u8;
    }
    out
}

// ============================================================================
// Idea 2: Noise-aware zero-bias modulation for XYB
// ============================================================================

use zenjpeg::encode::tuning::EncodingTables;

/// Generate per-component, per-frequency XYB zero-bias tables.
///
/// Instead of flat 0.5 for all AC, use frequency-dependent values:
/// - Low AC frequencies (perceptually important): lower mul → preserve more
/// - High AC frequencies (noise-like): higher mul → zero more aggressively
/// - B channel (subsampled, less sensitive): higher mul overall
/// - Y channel (most sensitive): lower mul
fn xyb_tuned_zero_bias(distance: f32) -> EncodingTables {
    let mut tables = EncodingTables::default_xyb();

    // Frequency importance weights (row-major 8x8 DCT)
    // Lower = more perceptually important = preserve more
    // Arranged as 8 rows × 8 columns, top-left = DC (lowest freq)
    #[rustfmt::skip]
    let freq_importance: [f32; 64] = [
        0.0,  0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65,
        0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70,
        0.35, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
        0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    ];

    // Component sensitivity multipliers
    // X (red-green): moderate sensitivity → moderately aggressive zero-bias
    // Y (intensity): highest sensitivity → least aggressive zero-bias
    // B (blue-yellow, subsampled): lowest sensitivity → most aggressive zero-bias
    //
    // These scale AROUND 0.5 (jpegli default), not from 0.
    // >1.0 = more aggressive than default, <1.0 = less aggressive
    let component_scale = [1.0f32, 0.85, 1.25]; // X, Y, B

    // Quality blending: at low quality (high distance), be more aggressive
    let quality_factor = 0.8 + 0.4 * ((distance - 0.5) / 4.0).clamp(0.0, 1.0);

    for c in 0..3 {
        let mul = tables.zero_bias_mul.get_mut(c);
        for k in 1..64 {
            // Scale around baseline 0.5, modulated by frequency, component, quality
            let freq_scale = 0.7 + 0.6 * freq_importance[k]; // 0.7 to 1.3
            mul[k] = 0.5 * freq_scale * component_scale[c] * quality_factor;
        }
        mul[0] = 0.0; // DC always 0
    }

    // Per-component offsets: B is less sensitive, can use higher offset
    tables.zero_bias_offset_ac = [0.50, 0.45, 0.58]; // X, Y, B

    tables
}

// ============================================================================
// Idea 3: Perceptual feedback loop
// ============================================================================

/// Simple encode-decode-measure loop for zero-bias redistribution.
///
/// 1. Encode with current zero-bias
/// 2. Decode
/// 3. Compute per-block MSE (proxy for perceptual error)
/// 4. Redistribute zero-bias: high-error blocks → lower mul (keep more coeffs)
///
/// Returns the adjusted EncodingTables after `iters` iterations.
fn perceptual_loop_xyb(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    iters: usize,
) -> EncodingTables {
    let mut tables = xyb_tuned_zero_bias(quality_to_distance(quality));

    let bw = (width + 7) / 8;
    let bh = (height + 7) / 8;
    let num_blocks = bw * bh;

    // Per-block zero-bias scale factors (start at 1.0)
    let mut block_scales = vec![1.0f32; num_blocks];

    for iter in 0..iters {
        // Build tables with per-block scaling baked into component 1 (Y) mul
        // We can't actually do per-block tables in JPEG, so we adjust the
        // GLOBAL tables based on the AVERAGE error pattern from the previous encode.
        // The per-block scales inform which frequencies need adjustment.

        // Encode
        let jpeg = encode_xyb_with_tables(pixels, width, height, quality, &tables);

        // Decode
        let decoded = decode_jpeg_to_rgb_u8(&jpeg);
        if decoded.len() != pixels.len() {
            eprintln!(
                "  loop iter {}: decode size mismatch ({} vs {}), skipping",
                iter,
                decoded.len(),
                pixels.len()
            );
            break;
        }

        // Compute per-block MSE
        let block_mse = compute_block_mse(pixels, &decoded, width, height);

        // Compute statistics
        let avg_mse: f32 = block_mse.iter().sum::<f32>() / block_mse.len() as f32;
        let max_mse = block_mse.iter().copied().fold(0.0f32, f32::max);

        eprintln!(
            "  loop iter {}: avg_mse={:.1} max_mse={:.1} jpeg_size={}",
            iter,
            avg_mse,
            max_mse,
            jpeg.len()
        );

        if avg_mse < 1.0 {
            break; // Good enough
        }

        // Sum-preserving redistribution (zensim style)
        // High MSE blocks → lower zero-bias (keep more coefficients)
        // Low MSE blocks → higher zero-bias (zero more coefficients)
        let k_alpha = 0.15;
        let mut new_scales = vec![0.0f32; num_blocks];
        let mut sum_before = 0.0f32;
        let mut sum_after = 0.0f32;

        for bi in 0..num_blocks {
            sum_before += block_scales[bi];

            let ratio = if avg_mse > 0.0 {
                block_mse[bi] / avg_mse
            } else {
                1.0
            };
            // High error → factor > 1 → LOWER zero-bias mul (1/factor applied to mul)
            let factor = (1.0 + k_alpha * (ratio - 1.0)).clamp(0.5, 2.0);
            // Invert: high error should reduce zero-bias (keep more coeffs)
            new_scales[bi] = block_scales[bi] / factor;
            sum_after += new_scales[bi];
        }

        // Renormalize to preserve sum (keeps overall file size stable)
        if sum_after > 0.0 {
            let renorm = sum_before / sum_after;
            for v in &mut new_scales {
                *v *= renorm;
            }
        }
        block_scales = new_scales;

        // Aggregate block_scales into frequency-domain adjustments
        // Group blocks by error quartile and adjust per-frequency mul
        adjust_tables_from_block_scales(&mut tables, &block_scales, &block_mse, avg_mse);
    }

    tables
}

/// Adjust EncodingTables based on spatial error pattern.
///
/// Blocks with high error tend to have specific frequency patterns.
/// We measure which frequencies contribute most to high-error blocks
/// and reduce zero-bias mul for those frequencies.
fn adjust_tables_from_block_scales(
    tables: &mut EncodingTables,
    block_scales: &[f32],
    block_mse: &[f32],
    avg_mse: f32,
) {
    // Split blocks into high-error and low-error groups
    let mut high_error_scale_sum = 0.0f32;
    let mut low_error_scale_sum = 0.0f32;
    let mut high_count = 0usize;
    let mut low_count = 0usize;

    for (bi, &mse) in block_mse.iter().enumerate() {
        if mse > avg_mse * 1.5 {
            high_error_scale_sum += block_scales[bi];
            high_count += 1;
        } else if mse < avg_mse * 0.5 {
            low_error_scale_sum += block_scales[bi];
            low_count += 1;
        }
    }

    if high_count == 0 || low_count == 0 {
        return;
    }

    let high_avg = high_error_scale_sum / high_count as f32;
    let low_avg = low_error_scale_sum / low_count as f32;

    // If high-error blocks have lower scales (need less zeroing),
    // globally reduce zero-bias for mid-high frequencies
    let adjustment = (high_avg / low_avg).clamp(0.8, 1.2);

    for c in 0..3 {
        let mul = tables.zero_bias_mul.get_mut(c);
        for k in 8..64 {
            // Adjust mid-high frequencies based on error pattern
            mul[k] *= adjustment;
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn quality_to_distance(quality: u8) -> f32 {
    // Approximate jpegli quality → butteraugli distance mapping
    if quality >= 100 {
        0.01
    } else if quality >= 95 {
        0.1 + (100.0 - quality as f32) * 0.08
    } else {
        0.5 + (95.0 - quality as f32) * 0.1
    }
}

fn encode_ycbcr(pixels: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(
        Quality::ApproxJpegli(quality as f32),
        zenjpeg::encoder::ChromaSubsampling::Quarter,
    );
    let mut enc = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, enough::Unstoppable)
        .expect("encode");
    enc.finish().expect("finish")
}

fn encode_xyb(pixels: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::xyb(
        Quality::ApproxJpegli(quality as f32),
        XybSubsampling::BQuarter,
    );
    let mut enc = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, enough::Unstoppable)
        .expect("encode");
    enc.finish().expect("finish")
}

fn encode_xyb_with_tables(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    tables: &EncodingTables,
) -> Vec<u8> {
    let config = EncoderConfig::xyb(
        Quality::ApproxJpegli(quality as f32),
        XybSubsampling::BQuarter,
    )
    .tables(Box::new(tables.clone()));
    let mut enc = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, enough::Unstoppable)
        .expect("encode");
    enc.finish().expect("finish")
}

fn decode_jpeg_to_rgb_u8(jpeg: &[u8]) -> Vec<u8> {
    let decoder = zenjpeg::decoder::Decoder::new();
    let result = decoder.decode(jpeg, enough::Unstoppable).expect("decode");
    result.pixels_u8().expect("u8 pixels").to_vec()
}

fn compute_block_mse(
    original: &[u8],
    decoded: &[u8],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let bw = (width + 7) / 8;
    let bh = (height + 7) / 8;
    let mut block_mse = vec![0.0f32; bw * bh];

    for by in 0..bh {
        for bx in 0..bw {
            let mut sum_sq = 0.0f64;
            let mut count = 0u32;
            for dy in 0..8 {
                let y = by * 8 + dy;
                if y >= height {
                    break;
                }
                for dx in 0..8 {
                    let x = bx * 8 + dx;
                    if x >= width {
                        break;
                    }
                    let idx = (y * width + x) * 3;
                    for c in 0..3 {
                        let diff = original[idx + c] as f64 - decoded[idx + c] as f64;
                        sum_sq += diff * diff;
                    }
                    count += 1;
                }
            }
            block_mse[by * bw + bx] = if count > 0 {
                (sum_sq / (count as f64 * 3.0)) as f32
            } else {
                0.0
            };
        }
    }
    block_mse
}

fn compute_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    let mut sum_sq = 0.0f64;
    for i in 0..original.len() {
        let diff = original[i] as f64 - decoded[i] as f64;
        sum_sq += diff * diff;
    }
    let mse = sum_sq / original.len() as f64;
    if mse < 0.001 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn load_png(path: &str) -> (Vec<u8>, usize, usize) {
    let file = std::fs::File::open(path).expect("open png");
    let reader = std::io::BufReader::new(file);
    let decoder = png::Decoder::new(reader);
    let mut reader = decoder.read_info().expect("read info");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("output size")];
    let info = reader.next_frame(&mut buf).expect("read frame");
    buf.truncate(info.buffer_size());

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB if needed
    match info.color_type {
        png::ColorType::Rgb => (buf, width, height),
        png::ColorType::Rgba => {
            let mut rgb = vec![0u8; width * height * 3];
            for i in 0..(width * height) {
                rgb[i * 3] = buf[i * 4];
                rgb[i * 3 + 1] = buf[i * 4 + 1];
                rgb[i * 3 + 2] = buf[i * 4 + 2];
            }
            (rgb, width, height)
        }
        png::ColorType::Grayscale => {
            let mut rgb = vec![0u8; width * height * 3];
            for i in 0..(width * height) {
                rgb[i * 3] = buf[i];
                rgb[i * 3 + 1] = buf[i];
                rgb[i * 3 + 2] = buf[i];
            }
            (rgb, width, height)
        }
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    }
}

// ============================================================================
// Main benchmark
// ============================================================================

fn run_benchmark(path: &str, quality: u8) {
    let name = Path::new(path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let (pixels, width, height) = load_png(path);

    eprintln!(
        "\n=== {} ({}x{}, Q{}) ===",
        name, width, height, quality
    );

    // Baseline: standard YCbCr encode
    let t0 = Instant::now();
    let jpeg_ycbcr = encode_ycbcr(&pixels, width, height, quality);
    let t_ycbcr = t0.elapsed();
    let dec_ycbcr = decode_jpeg_to_rgb_u8(&jpeg_ycbcr);
    let psnr_ycbcr = compute_psnr(&pixels, &dec_ycbcr);

    // Baseline: standard XYB encode
    let t0 = Instant::now();
    let jpeg_xyb = encode_xyb(&pixels, width, height, quality);
    let t_xyb = t0.elapsed();
    let dec_xyb = decode_jpeg_to_rgb_u8(&jpeg_xyb);
    let psnr_xyb = compute_psnr(&pixels, &dec_xyb);

    // Idea 1: Pre-encode noise-gated smoothing (light)
    // sigma=1.0 (small kernel), noise_floor=5.0 (high threshold = only smooth very flat)
    let t0 = Instant::now();
    let filtered_light = prefilter_rgb(&pixels, width, height, 1.0, 5.0);
    let jpeg_prefilter_light = encode_ycbcr(&filtered_light, width, height, quality);
    let t_prefilter_light = t0.elapsed();
    let dec_prefilter_light = decode_jpeg_to_rgb_u8(&jpeg_prefilter_light);
    let psnr_prefilter_light = compute_psnr(&pixels, &dec_prefilter_light);

    // Idea 1b: Pre-encode noise-gated smoothing (medium)
    let filtered = prefilter_rgb(&pixels, width, height, 1.5, 3.0);
    let jpeg_prefilter = encode_ycbcr(&filtered, width, height, quality);
    let dec_prefilter = decode_jpeg_to_rgb_u8(&jpeg_prefilter);
    // Compare decoded prefiltered against ORIGINAL (not filtered) pixels
    let psnr_prefilter = compute_psnr(&pixels, &dec_prefilter);

    // Idea 1c: Pre-filter + XYB
    let jpeg_prefilter_xyb = encode_xyb(&filtered_light, width, height, quality);
    let dec_prefilter_xyb = decode_jpeg_to_rgb_u8(&jpeg_prefilter_xyb);
    let psnr_prefilter_xyb = compute_psnr(&pixels, &dec_prefilter_xyb);

    // Idea 2: XYB with tuned zero-bias (per-component, per-frequency)
    let distance = quality_to_distance(quality);
    let tuned_tables = xyb_tuned_zero_bias(distance);
    let t0 = Instant::now();
    let jpeg_xyb_tuned = encode_xyb_with_tables(&pixels, width, height, quality, &tuned_tables);
    let t_xyb_tuned = t0.elapsed();
    let dec_xyb_tuned = decode_jpeg_to_rgb_u8(&jpeg_xyb_tuned);
    let psnr_xyb_tuned = compute_psnr(&pixels, &dec_xyb_tuned);

    // Idea 3: Perceptual feedback loop (XYB, 2 iterations)
    let t0 = Instant::now();
    let loop_tables = perceptual_loop_xyb(&pixels, width, height, quality, 2);
    let jpeg_loop = encode_xyb_with_tables(&pixels, width, height, quality, &loop_tables);
    let t_loop = t0.elapsed();
    let dec_loop = decode_jpeg_to_rgb_u8(&jpeg_loop);
    let psnr_loop = compute_psnr(&pixels, &dec_loop);

    // Report
    let base_size = jpeg_ycbcr.len();
    let report = |label: &str, size: usize, psnr: f64, ms: f64| {
        let delta_pct = (size as f64 / base_size as f64 - 1.0) * 100.0;
        let delta_psnr = psnr - psnr_ycbcr;
        println!(
            "{:<30} {:>8} ({:+5.1}%)  PSNR {:.2} ({:+.2})  {:.0}ms",
            label, size, delta_pct, psnr, delta_psnr, ms
        );
    };

    println!(
        "{:<30} {:>8} {:>8}  {:>10} {:>8}  {:>6}",
        "Mode", "Size", "Δ%", "PSNR", "ΔPSNR", "ms"
    );
    println!("{}", "-".repeat(78));
    report(
        "YCbCr baseline",
        jpeg_ycbcr.len(),
        psnr_ycbcr,
        t_ycbcr.as_secs_f64() * 1000.0,
    );
    report(
        "XYB baseline",
        jpeg_xyb.len(),
        psnr_xyb,
        t_xyb.as_secs_f64() * 1000.0,
    );
    report(
        "1a: Prefilter light+YCbCr",
        jpeg_prefilter_light.len(),
        psnr_prefilter_light,
        t_prefilter_light.as_secs_f64() * 1000.0,
    );
    report(
        "1b: Prefilter med+YCbCr",
        jpeg_prefilter.len(),
        psnr_prefilter,
        0.0,
    );
    report(
        "1c: Prefilter light+XYB",
        jpeg_prefilter_xyb.len(),
        psnr_prefilter_xyb,
        0.0,
    );
    report(
        "2: XYB tuned zero-bias",
        jpeg_xyb_tuned.len(),
        psnr_xyb_tuned,
        t_xyb_tuned.as_secs_f64() * 1000.0,
    );
    report(
        "3: XYB percept loop (2 iter)",
        jpeg_loop.len(),
        psnr_loop,
        t_loop.as_secs_f64() * 1000.0,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: explore_prefilter <image.png> [image2.png ...]");
        eprintln!("       explore_prefilter ~/work/codec-corpus/cid22/*.png");
        std::process::exit(1);
    }

    let qualities = [75, 85, 95];

    for path in &args[1..] {
        if !Path::new(path).exists() {
            eprintln!("Skipping {}: not found", path);
            continue;
        }
        for &q in &qualities {
            run_benchmark(path, q);
        }
    }
}
