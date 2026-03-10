//! Exploration: XYB zero-bias tuning with SSIMULACRA2 evaluation.
//!
//! Tests improved XYB zero-bias tables against the hardcoded 0.5 baseline.
//! Measures quality with SSIMULACRA2 (perceptually calibrated) instead of PSNR.
//!
//! Three modes:
//! 1. `sweep` - Systematic per-component mul sweep to find optimal values
//! 2. `bench` - Compare pre-encode denoising, tuned tables, and perceptual loop
//! 3. `prefilter` - Test pre-encode noise-gated smoothing
//!
//! Usage:
//!   cargo run --release --example explore_prefilter -- sweep image1.png [image2.png ...]
//!   cargo run --release --example explore_prefilter -- bench image1.png [image2.png ...]
//!   cargo run --release --example explore_prefilter -- prefilter image1.png [image2.png ...]

use std::env;
use std::path::Path;
use std::time::Instant;

use fast_ssim2::{LinearRgbImage, compute_frame_ssimulacra2, srgb_u8_to_linear};
use zenjpeg::encode::tuning::EncodingTables;
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

fn gaussian_blur_2d(buf: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;

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
// Pre-encode noise-gated smoothing
// ============================================================================

fn noise_gated_smooth(
    plane: &[f32],
    width: usize,
    height: usize,
    sigma: f32,
    noise_floor: f32,
) -> Vec<f32> {
    let blurred = gaussian_blur_2d(plane, width, height, sigma);

    let mut detail_sq = vec![0.0f32; width * height];
    for i in 0..plane.len() {
        let d = plane[i] - blurred[i];
        detail_sq[i] = d * d;
    }

    let energy = gaussian_blur_2d(&detail_sq, width, height, sigma * 3.0);

    let mut out = vec![0.0f32; width * height];
    for i in 0..plane.len() {
        let e = energy[i].sqrt();
        let gate = e / (e + noise_floor);
        out[i] = gate * plane[i] + (1.0 - gate) * blurred[i];
    }
    out
}

fn prefilter_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    sigma: f32,
    noise_floor: f32,
) -> Vec<u8> {
    let npix = width * height;

    let mut r = vec![0.0f32; npix];
    let mut g = vec![0.0f32; npix];
    let mut b = vec![0.0f32; npix];
    for i in 0..npix {
        r[i] = pixels[i * 3] as f32;
        g[i] = pixels[i * 3 + 1] as f32;
        b[i] = pixels[i * 3 + 2] as f32;
    }

    let r_f = noise_gated_smooth(&r, width, height, sigma, noise_floor);
    let g_f = noise_gated_smooth(&g, width, height, sigma, noise_floor);
    let b_f = noise_gated_smooth(&b, width, height, sigma, noise_floor);

    let mut out = vec![0u8; npix * 3];
    for i in 0..npix {
        out[i * 3] = r_f[i].round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = g_f[i].round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = b_f[i].round().clamp(0.0, 255.0) as u8;
    }
    out
}

// ============================================================================
// SSIMULACRA2 quality metric
// ============================================================================

fn bytes_to_linear(data: &[u8], width: usize, height: usize) -> LinearRgbImage {
    let pixels: Vec<[f32; 3]> = data
        .chunks_exact(3)
        .map(|rgb| {
            [
                srgb_u8_to_linear(rgb[0]),
                srgb_u8_to_linear(rgb[1]),
                srgb_u8_to_linear(rgb[2]),
            ]
        })
        .collect();
    LinearRgbImage::new(pixels, width, height)
}

fn compute_ssim2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let orig = bytes_to_linear(original, width, height);
    let dec = bytes_to_linear(decoded, width, height);
    compute_frame_ssimulacra2(orig, dec).unwrap_or(-99.0)
}

// ============================================================================
// XYB zero-bias table generators
// ============================================================================

/// Frequency-dependent XYB zero-bias tables, v3.
///
/// Sweep results (13 CID22 images) showed:
/// - Y channel mul is by far the most impactful (±5 SSIM2 range)
/// - X and B channels have minimal impact (±0.5 SSIM2, B is subsampled)
/// - At Q75/low quality: higher mul (0.5-0.7) improves SSIM2 by zeroing noise
/// - At Q95/high quality: lower mul (0.3-0.5) preserves detail
/// - DC-adjacent coefficients (positions [0,1] and [1,0]) must be very low
///
/// Design: HQ tables are LESS aggressive than baseline 0.5 (preserve detail),
/// LQ tables are MORE aggressive (zero noise). Quality blending interpolates.
fn xyb_tuned_zero_bias_v2(distance: f32) -> EncodingTables {
    let mut tables = EncodingTables::default_xyb();

    // Quality blending: 0.0 = HQ (distance ≤ 1.0), 1.0 = LQ (distance ≥ 3.0)
    let lq_mix = ((distance - 1.0) / 2.0).clamp(0.0, 1.0);

    // --- Y channel (component 1 in XYB) ---
    // Most sensitive. DC-adjacent must be very low.
    // HQ: below 0.5 baseline to preserve detail at high quality.
    // LQ: above 0.5 baseline to zero noise at low quality.
    #[rustfmt::skip]
    let y_hq: [f32; 64] = [
        0.00, 0.01, 0.08, 0.20, 0.30, 0.35, 0.38, 0.40,
        0.01, 0.15, 0.25, 0.32, 0.35, 0.38, 0.40, 0.42,
        0.08, 0.25, 0.35, 0.38, 0.40, 0.42, 0.44, 0.45,
        0.20, 0.32, 0.38, 0.42, 0.44, 0.45, 0.46, 0.48,
        0.30, 0.35, 0.40, 0.44, 0.45, 0.46, 0.48, 0.48,
        0.35, 0.38, 0.42, 0.45, 0.46, 0.48, 0.48, 0.50,
        0.38, 0.40, 0.44, 0.46, 0.48, 0.48, 0.50, 0.50,
        0.40, 0.42, 0.45, 0.48, 0.48, 0.50, 0.50, 0.50,
    ];
    #[rustfmt::skip]
    let y_lq: [f32; 64] = [
        0.00, 0.05, 0.25, 0.45, 0.55, 0.58, 0.62, 0.65,
        0.05, 0.35, 0.48, 0.55, 0.58, 0.62, 0.65, 0.68,
        0.25, 0.48, 0.55, 0.58, 0.62, 0.65, 0.68, 0.70,
        0.45, 0.55, 0.58, 0.62, 0.65, 0.68, 0.70, 0.72,
        0.55, 0.58, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75,
        0.58, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.75,
        0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.75, 0.78,
        0.65, 0.68, 0.70, 0.72, 0.75, 0.75, 0.78, 0.78,
    ];

    // --- X channel (component 0 in XYB) ---
    // Red-green difference, chroma-like. X has very little impact (sweep shows
    // ~0.5 SSIM2 range over the full 0.3-1.2 mul sweep), so keep close to 0.5.
    #[rustfmt::skip]
    let x_hq: [f32; 64] = [
        0.00, 0.05, 0.20, 0.35, 0.42, 0.45, 0.48, 0.50,
        0.05, 0.25, 0.38, 0.42, 0.45, 0.48, 0.50, 0.50,
        0.20, 0.38, 0.45, 0.48, 0.50, 0.50, 0.52, 0.52,
        0.35, 0.42, 0.48, 0.50, 0.50, 0.52, 0.52, 0.55,
        0.42, 0.45, 0.50, 0.50, 0.52, 0.55, 0.55, 0.55,
        0.45, 0.48, 0.50, 0.52, 0.55, 0.55, 0.55, 0.55,
        0.48, 0.50, 0.52, 0.52, 0.55, 0.55, 0.55, 0.58,
        0.50, 0.50, 0.52, 0.55, 0.55, 0.55, 0.58, 0.58,
    ];
    #[rustfmt::skip]
    let x_lq: [f32; 64] = [
        0.00, 0.08, 0.28, 0.45, 0.52, 0.55, 0.58, 0.60,
        0.08, 0.35, 0.48, 0.52, 0.55, 0.58, 0.60, 0.62,
        0.28, 0.48, 0.55, 0.58, 0.60, 0.62, 0.62, 0.65,
        0.45, 0.52, 0.58, 0.60, 0.62, 0.65, 0.65, 0.68,
        0.52, 0.55, 0.60, 0.62, 0.65, 0.65, 0.68, 0.68,
        0.55, 0.58, 0.62, 0.65, 0.65, 0.68, 0.68, 0.70,
        0.58, 0.60, 0.62, 0.65, 0.68, 0.68, 0.70, 0.70,
        0.60, 0.62, 0.65, 0.68, 0.68, 0.70, 0.70, 0.72,
    ];

    // --- B channel (component 2 in XYB) ---
    // Blue-yellow, subsampled, least sensitive. Sweep shows B has the least
    // impact (~0.4 SSIM2 range). Keep slightly above 0.5 to save a few bytes.
    #[rustfmt::skip]
    let b_hq: [f32; 64] = [
        0.00, 0.10, 0.30, 0.42, 0.48, 0.50, 0.52, 0.55,
        0.10, 0.35, 0.45, 0.48, 0.50, 0.52, 0.55, 0.55,
        0.30, 0.45, 0.50, 0.52, 0.55, 0.55, 0.58, 0.58,
        0.42, 0.48, 0.52, 0.55, 0.55, 0.58, 0.58, 0.60,
        0.48, 0.50, 0.55, 0.55, 0.58, 0.58, 0.60, 0.60,
        0.50, 0.52, 0.55, 0.58, 0.58, 0.60, 0.60, 0.62,
        0.52, 0.55, 0.58, 0.58, 0.60, 0.60, 0.62, 0.62,
        0.55, 0.55, 0.58, 0.60, 0.60, 0.62, 0.62, 0.65,
    ];
    #[rustfmt::skip]
    let b_lq: [f32; 64] = [
        0.00, 0.15, 0.40, 0.55, 0.62, 0.68, 0.72, 0.75,
        0.15, 0.45, 0.58, 0.62, 0.68, 0.72, 0.75, 0.78,
        0.40, 0.58, 0.65, 0.68, 0.72, 0.75, 0.78, 0.80,
        0.55, 0.62, 0.68, 0.72, 0.75, 0.78, 0.80, 0.82,
        0.62, 0.68, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85,
        0.68, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.85,
        0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.85, 0.88,
        0.75, 0.78, 0.80, 0.82, 0.85, 0.85, 0.88, 0.88,
    ];

    let channel_tables = [
        (&x_hq, &x_lq), // component 0: X
        (&y_hq, &y_lq), // component 1: Y
        (&b_hq, &b_lq), // component 2: B
    ];

    for (c, (hq, lq)) in channel_tables.iter().enumerate() {
        let mul = tables.zero_bias_mul.get_mut(c);
        for k in 0..64 {
            mul[k] = hq[k] * (1.0 - lq_mix) + lq[k] * lq_mix;
        }
    }

    // Per-component offsets (like YCbCr's ~0.58-0.59)
    tables.zero_bias_offset_ac = [0.50, 0.48, 0.55]; // X, Y, B

    tables
}

/// Simple uniform per-component scaling for sweep tests.
fn xyb_uniform_bias(x_mul: f32, y_mul: f32, b_mul: f32) -> EncodingTables {
    let mut tables = EncodingTables::default_xyb();
    for k in 1..64 {
        tables.zero_bias_mul.get_mut(0)[k] = x_mul;
        tables.zero_bias_mul.get_mut(1)[k] = y_mul;
        tables.zero_bias_mul.get_mut(2)[k] = b_mul;
    }
    tables
}

// ============================================================================
// Encode/decode helpers
// ============================================================================

fn quality_to_distance(quality: u8) -> f32 {
    // Match jpegli's quality_to_distance exactly
    let q = quality as f32;
    if q >= 100.0 {
        0.01
    } else if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
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
// Mode 1: Per-component mul sweep
// ============================================================================

fn run_sweep(paths: &[String]) {
    println!("image\tquality\tmode\tx_mul\ty_mul\tb_mul\tsize\tssim2");

    // Sweep values: test a grid of uniform per-component multipliers
    let mul_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    let qualities = [75u8, 85, 95];

    for path in paths {
        if !Path::new(path).exists() {
            eprintln!("Skipping {}: not found", path);
            continue;
        }

        let name = Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let (pixels, width, height) = load_png(path);
        eprintln!("Loaded {} ({}x{})", name, width, height);

        for &q in &qualities {
            // Baselines
            let jpeg_ycbcr = encode_ycbcr(&pixels, width, height, q);
            let dec_ycbcr = decode_jpeg_to_rgb_u8(&jpeg_ycbcr);
            let ss2_ycbcr = compute_ssim2(&pixels, &dec_ycbcr, width, height);
            println!(
                "{}\t{}\tycbcr\t-\t-\t-\t{}\t{:.2}",
                name,
                q,
                jpeg_ycbcr.len(),
                ss2_ycbcr
            );

            let jpeg_xyb = encode_xyb(&pixels, width, height, q);
            let dec_xyb = decode_jpeg_to_rgb_u8(&jpeg_xyb);
            let ss2_xyb = compute_ssim2(&pixels, &dec_xyb, width, height);
            println!(
                "{}\t{}\txyb_0.5\t0.5\t0.5\t0.5\t{}\t{:.2}",
                name,
                q,
                jpeg_xyb.len(),
                ss2_xyb
            );

            // Tuned v2 (frequency-dependent)
            let tuned = xyb_tuned_zero_bias_v2(quality_to_distance(q));
            let jpeg_tuned = encode_xyb_with_tables(&pixels, width, height, q, &tuned);
            let dec_tuned = decode_jpeg_to_rgb_u8(&jpeg_tuned);
            let ss2_tuned = compute_ssim2(&pixels, &dec_tuned, width, height);
            println!(
                "{}\t{}\txyb_tuned_v2\t-\t-\t-\t{}\t{:.2}",
                name,
                q,
                jpeg_tuned.len(),
                ss2_tuned
            );

            // Sweep: vary Y with X=0.6, B=0.9 (based on YCbCr patterns)
            for &y_mul in &mul_values {
                let tables = xyb_uniform_bias(0.6, y_mul, 0.9);
                let jpeg = encode_xyb_with_tables(&pixels, width, height, q, &tables);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                println!(
                    "{}\t{}\tsweep_y\t0.6\t{:.1}\t0.9\t{}\t{:.2}",
                    name,
                    q,
                    y_mul,
                    jpeg.len(),
                    ss2
                );
            }

            // Sweep: vary X with Y=0.5, B=0.9
            for &x_mul in &mul_values {
                let tables = xyb_uniform_bias(x_mul, 0.5, 0.9);
                let jpeg = encode_xyb_with_tables(&pixels, width, height, q, &tables);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                println!(
                    "{}\t{}\tsweep_x\t{:.1}\t0.5\t0.9\t{}\t{:.2}",
                    name,
                    q,
                    x_mul,
                    jpeg.len(),
                    ss2
                );
            }

            // Sweep: vary B with X=0.6, Y=0.5
            for &b_mul in &mul_values {
                let tables = xyb_uniform_bias(0.6, 0.5, b_mul);
                let jpeg = encode_xyb_with_tables(&pixels, width, height, q, &tables);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                println!(
                    "{}\t{}\tsweep_b\t0.6\t0.5\t{:.1}\t{}\t{:.2}",
                    name,
                    q,
                    b_mul,
                    jpeg.len(),
                    ss2
                );
            }

            eprintln!("  Q{} done", q);
        }
    }
}

// ============================================================================
// Mode 2: Full benchmark (prefilter + tuned tables + loop)
// ============================================================================

fn run_benchmark(paths: &[String]) {
    println!("image\tquality\tmode\tsize\tssim2\tsize_vs_ycbcr\tssim2_vs_ycbcr\tms");

    let qualities = [75u8, 85, 95];

    for path in paths {
        if !Path::new(path).exists() {
            eprintln!("Skipping {}: not found", path);
            continue;
        }

        let name = Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let (pixels, width, height) = load_png(path);
        eprintln!("\n=== {} ({}x{}) ===", name, width, height);

        for &q in &qualities {
            let report = |label: &str, size: usize, ss2: f64, base_size: usize, base_ss2: f64, ms: f64| {
                let delta_pct = (size as f64 / base_size as f64 - 1.0) * 100.0;
                let delta_ss2 = ss2 - base_ss2;
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{:+.1}%\t{:+.2}\t{:.0}",
                    name, q, label, size, ss2, delta_pct, delta_ss2, ms
                );
            };

            // YCbCr baseline
            let t0 = Instant::now();
            let jpeg_ycbcr = encode_ycbcr(&pixels, width, height, q);
            let t_ycbcr = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_ycbcr = decode_jpeg_to_rgb_u8(&jpeg_ycbcr);
            let ss2_ycbcr = compute_ssim2(&pixels, &dec_ycbcr, width, height);
            report("ycbcr", jpeg_ycbcr.len(), ss2_ycbcr, jpeg_ycbcr.len(), ss2_ycbcr, t_ycbcr);

            // XYB baseline (flat 0.5)
            let t0 = Instant::now();
            let jpeg_xyb = encode_xyb(&pixels, width, height, q);
            let t_xyb = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_xyb = decode_jpeg_to_rgb_u8(&jpeg_xyb);
            let ss2_xyb = compute_ssim2(&pixels, &dec_xyb, width, height);
            report("xyb_0.5", jpeg_xyb.len(), ss2_xyb, jpeg_ycbcr.len(), ss2_ycbcr, t_xyb);

            // XYB tuned v2 (frequency-dependent, YCbCr-inspired)
            let t0 = Instant::now();
            let tuned = xyb_tuned_zero_bias_v2(quality_to_distance(q));
            let jpeg_tuned = encode_xyb_with_tables(&pixels, width, height, q, &tuned);
            let t_tuned = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_tuned = decode_jpeg_to_rgb_u8(&jpeg_tuned);
            let ss2_tuned = compute_ssim2(&pixels, &dec_tuned, width, height);
            report("xyb_tuned_v2", jpeg_tuned.len(), ss2_tuned, jpeg_ycbcr.len(), ss2_ycbcr, t_tuned);

            // Prefilter + XYB baseline
            let t0 = Instant::now();
            let filtered = prefilter_rgb(&pixels, width, height, 1.0, 5.0);
            let jpeg_pf_xyb = encode_xyb(&filtered, width, height, q);
            let t_pf = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_pf_xyb = decode_jpeg_to_rgb_u8(&jpeg_pf_xyb);
            // Compare decoded against ORIGINAL pixels
            let ss2_pf_xyb = compute_ssim2(&pixels, &dec_pf_xyb, width, height);
            report("prefilter+xyb", jpeg_pf_xyb.len(), ss2_pf_xyb, jpeg_ycbcr.len(), ss2_ycbcr, t_pf);

            // Prefilter + tuned v2
            let t0 = Instant::now();
            let jpeg_pf_tuned = encode_xyb_with_tables(&filtered, width, height, q, &tuned);
            let t_pf_tuned = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_pf_tuned = decode_jpeg_to_rgb_u8(&jpeg_pf_tuned);
            let ss2_pf_tuned = compute_ssim2(&pixels, &dec_pf_tuned, width, height);
            report("prefilter+tuned_v2", jpeg_pf_tuned.len(), ss2_pf_tuned, jpeg_ycbcr.len(), ss2_ycbcr, t_pf_tuned);

            // Perceptual loop (2 iterations, using tuned v2 as starting point)
            let t0 = Instant::now();
            let loop_tables = perceptual_loop_xyb(&pixels, width, height, q, 2);
            let jpeg_loop = encode_xyb_with_tables(&pixels, width, height, q, &loop_tables);
            let t_loop = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_loop = decode_jpeg_to_rgb_u8(&jpeg_loop);
            let ss2_loop = compute_ssim2(&pixels, &dec_loop, width, height);
            report("percept_loop_2", jpeg_loop.len(), ss2_loop, jpeg_ycbcr.len(), ss2_ycbcr, t_loop);

            eprintln!("  Q{} done", q);
        }
    }
}

// ============================================================================
// Mode 3: Prefilter-only comparison
// ============================================================================

fn run_prefilter(paths: &[String]) {
    println!("image\tquality\tmode\tsize\tssim2\tsize_vs_base\tssim2_vs_base");

    let qualities = [75u8, 85, 95];
    let configs = [
        ("light", 1.0f32, 5.0f32),
        ("medium", 1.5, 3.0),
        ("heavy", 2.0, 2.0),
    ];

    for path in paths {
        if !Path::new(path).exists() {
            continue;
        }

        let name = Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let (pixels, width, height) = load_png(path);
        eprintln!("Loaded {} ({}x{})", name, width, height);

        for &q in &qualities {
            // Baselines
            let jpeg_ycbcr = encode_ycbcr(&pixels, width, height, q);
            let dec_ycbcr = decode_jpeg_to_rgb_u8(&jpeg_ycbcr);
            let ss2_ycbcr = compute_ssim2(&pixels, &dec_ycbcr, width, height);
            println!(
                "{}\t{}\tycbcr\t{}\t{:.2}\t-\t-",
                name,
                q,
                jpeg_ycbcr.len(),
                ss2_ycbcr
            );

            let jpeg_xyb = encode_xyb(&pixels, width, height, q);
            let dec_xyb = decode_jpeg_to_rgb_u8(&jpeg_xyb);
            let ss2_xyb = compute_ssim2(&pixels, &dec_xyb, width, height);
            println!(
                "{}\t{}\txyb\t{}\t{:.2}\t-\t-",
                name,
                q,
                jpeg_xyb.len(),
                ss2_xyb
            );

            for &(label, sigma, noise_floor) in &configs {
                let filtered = prefilter_rgb(&pixels, width, height, sigma, noise_floor);

                // Prefilter + YCbCr
                let jpeg = encode_ycbcr(&filtered, width, height, q);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                let d_size = (jpeg.len() as f64 / jpeg_ycbcr.len() as f64 - 1.0) * 100.0;
                let d_ss2 = ss2 - ss2_ycbcr;
                println!(
                    "{}\t{}\tpf_{}_ycbcr\t{}\t{:.2}\t{:+.1}%\t{:+.2}",
                    name,
                    q,
                    label,
                    jpeg.len(),
                    ss2,
                    d_size,
                    d_ss2
                );

                // Prefilter + XYB
                let jpeg = encode_xyb(&filtered, width, height, q);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                let d_size = (jpeg.len() as f64 / jpeg_xyb.len() as f64 - 1.0) * 100.0;
                let d_ss2 = ss2 - ss2_xyb;
                println!(
                    "{}\t{}\tpf_{}_xyb\t{}\t{:.2}\t{:+.1}%\t{:+.2}",
                    name,
                    q,
                    label,
                    jpeg.len(),
                    ss2,
                    d_size,
                    d_ss2
                );
            }
        }
    }
}

// ============================================================================
// Perceptual feedback loop
// ============================================================================

fn perceptual_loop_xyb(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    iters: usize,
) -> EncodingTables {
    let mut tables = xyb_tuned_zero_bias_v2(quality_to_distance(quality));

    let bw = (width + 7) / 8;
    let bh = (height + 7) / 8;
    let num_blocks = bw * bh;

    let mut block_scales = vec![1.0f32; num_blocks];

    for iter in 0..iters {
        let jpeg = encode_xyb_with_tables(pixels, width, height, quality, &tables);
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
        let avg_mse: f32 = block_mse.iter().sum::<f32>() / block_mse.len() as f32;
        let max_mse = block_mse.iter().copied().fold(0.0f32, f32::max);

        eprintln!(
            "  loop iter {}: avg_mse={:.1} max_mse={:.1} jpeg_size={}",
            iter, avg_mse, max_mse, jpeg.len()
        );

        if avg_mse < 1.0 {
            break;
        }

        // Sum-preserving redistribution
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
            let factor = (1.0 + k_alpha * (ratio - 1.0)).clamp(0.5, 2.0);
            new_scales[bi] = block_scales[bi] / factor;
            sum_after += new_scales[bi];
        }

        if sum_after > 0.0 {
            let renorm = sum_before / sum_after;
            for v in &mut new_scales {
                *v *= renorm;
            }
        }
        block_scales = new_scales;

        adjust_tables_from_block_scales(&mut tables, &block_scales, &block_mse, avg_mse);
    }

    tables
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

fn adjust_tables_from_block_scales(
    tables: &mut EncodingTables,
    block_scales: &[f32],
    block_mse: &[f32],
    avg_mse: f32,
) {
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
    let adjustment = (high_avg / low_avg).clamp(0.8, 1.2);

    for c in 0..3 {
        let mul = tables.zero_bias_mul.get_mut(c);
        for k in 8..64 {
            mul[k] *= adjustment;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: explore_prefilter <mode> <image.png> [image2.png ...]");
        eprintln!("Modes:");
        eprintln!("  sweep     - Per-component zero-bias multiplier sweep");
        eprintln!("  bench     - Full benchmark (prefilter + tuned tables + loop)");
        eprintln!("  prefilter - Pre-encode noise-gated smoothing comparison");
        std::process::exit(1);
    }

    let mode = &args[1];
    let paths: Vec<String> = args[2..].to_vec();

    match mode.as_str() {
        "sweep" => run_sweep(&paths),
        "bench" => run_benchmark(&paths),
        "prefilter" => run_prefilter(&paths),
        _ => {
            eprintln!("Unknown mode: {}. Use sweep, bench, or prefilter.", mode);
            std::process::exit(1);
        }
    }
}
