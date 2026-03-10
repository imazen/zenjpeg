//! Exploration: XYB zero-bias tuning with SSIMULACRA2 evaluation.
//!
//! Tests improved XYB zero-bias tables against the hardcoded 0.5 baseline.
//! Measures quality with SSIMULACRA2 (perceptually calibrated) instead of PSNR.
//!
//! Four modes:
//! 1. `sweep` - Systematic per-component mul sweep to find optimal values
//! 2. `bench` - Compare tuned tables, MSE loop, and butteraugli-guided loop
//! 3. `prefilter` - Test pre-encode noise-gated smoothing
//! 4. `dqt` - Per-image DQT seeding from spectral analysis
//!
//! Usage:
//!   cargo run --release --example explore_prefilter -- sweep image1.png [image2.png ...]
//!   cargo run --release --example explore_prefilter -- bench image1.png [image2.png ...]
//!   cargo run --release --example explore_prefilter -- prefilter image1.png [image2.png ...]
//!   cargo run --release --example explore_prefilter -- dqt image1.png [image2.png ...]

use std::env;
use std::path::Path;
use std::time::Instant;

use butteraugli::ButteraugliParams;
use fast_ssim2::{LinearRgbImage, compute_frame_ssimulacra2, srgb_u8_to_linear};
use zenjpeg::color::xyb::srgb_to_xyb;
use zenjpeg::encode::dct::forward_dct_8x8;
use zenjpeg::encode::tuning::{EncodingTables, ScalingParams};
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
                let sx = (x as isize + ki as isize - radius as isize).clamp(0, width as isize - 1)
                    as usize;
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
                let sy = (y as isize + ki as isize - radius as isize).clamp(0, height as isize - 1)
                    as usize;
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
// Butteraugli quality metric
// ============================================================================

fn compute_butteraugli(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let orig_pixels: Vec<rgb::RGB8> = original
        .chunks_exact(3)
        .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
        .collect();
    let dec_pixels: Vec<rgb::RGB8> = decoded
        .chunks_exact(3)
        .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = imgref::Img::new(&orig_pixels[..], width, height);
    let dec_img = imgref::Img::new(&dec_pixels[..], width, height);
    let params = ButteraugliParams::default();
    match butteraugli::butteraugli(orig_img, dec_img, &params) {
        Ok(result) => result.score,
        Err(_) => f64::NAN,
    }
}

/// Compute butteraugli at a downscaled resolution (e.g., scale=2 means half-size).
/// This tests whether denoising looks better at typical viewing distances.
fn compute_butteraugli_scaled(
    original: &[u8],
    decoded: &[u8],
    width: usize,
    height: usize,
    scale: usize,
) -> f64 {
    if scale <= 1 {
        return compute_butteraugli(original, decoded, width, height);
    }
    let sw = width / scale;
    let sh = height / scale;
    if sw < 8 || sh < 8 {
        return f64::NAN;
    }
    let orig_down = box_downsample(original, width, height, scale);
    let dec_down = box_downsample(decoded, width, height, scale);
    compute_butteraugli(&orig_down, &dec_down, sw, sh)
}

/// Simple box-filter downsampling (average NxN blocks).
fn box_downsample(pixels: &[u8], width: usize, height: usize, scale: usize) -> Vec<u8> {
    let sw = width / scale;
    let sh = height / scale;
    let mut out = vec![0u8; sw * sh * 3];
    let area = (scale * scale) as f32;
    for sy in 0..sh {
        for sx in 0..sw {
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            for dy in 0..scale {
                for dx in 0..scale {
                    let idx = ((sy * scale + dy) * width + sx * scale + dx) * 3;
                    r += pixels[idx] as f32;
                    g += pixels[idx + 1] as f32;
                    b += pixels[idx + 2] as f32;
                }
            }
            let oi = (sy * sw + sx) * 3;
            out[oi] = (r / area + 0.5) as u8;
            out[oi + 1] = (g / area + 0.5) as u8;
            out[oi + 2] = (b / area + 0.5) as u8;
        }
    }
    out
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
///
/// Now uses the library's v3 tables (ZERO_BIAS_MUL_XYB_HQ/LQ) with blending.
fn xyb_tuned_zero_bias_v2(distance: f32) -> EncodingTables {
    let mut tables = EncodingTables::default_xyb();
    // default_xyb() stores LQ tables. Blend toward HQ based on distance.
    let hq = EncodingTables::xyb_hq_zero_bias_mul();
    let lq = EncodingTables::xyb_lq_zero_bias_mul();
    // t = 1.0 at distance <= 1.0 (full HQ), t = 0.0 at distance >= 3.0 (full LQ)
    let t = 1.0 - ((distance - 1.0) / 2.0).clamp(0.0, 1.0);
    tables.zero_bias_mul = lq.blend(&hq, t);
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
    println!("image\tquality\tmode\tx_mul\ty_mul\tb_mul\tsize\tssim2\tbfly");

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
            let bfly_ycbcr = compute_butteraugli(&pixels, &dec_ycbcr, width, height);
            println!(
                "{}\t{}\tycbcr\t-\t-\t-\t{}\t{:.2}\t{:.4}",
                name,
                q,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr
            );

            let jpeg_xyb = encode_xyb(&pixels, width, height, q);
            let dec_xyb = decode_jpeg_to_rgb_u8(&jpeg_xyb);
            let ss2_xyb = compute_ssim2(&pixels, &dec_xyb, width, height);
            let bfly_xyb = compute_butteraugli(&pixels, &dec_xyb, width, height);
            println!(
                "{}\t{}\txyb_0.5\t0.5\t0.5\t0.5\t{}\t{:.2}\t{:.4}",
                name,
                q,
                jpeg_xyb.len(),
                ss2_xyb,
                bfly_xyb
            );

            // Tuned v3 (frequency-dependent)
            let tuned = xyb_tuned_zero_bias_v2(quality_to_distance(q));
            let jpeg_tuned = encode_xyb_with_tables(&pixels, width, height, q, &tuned);
            let dec_tuned = decode_jpeg_to_rgb_u8(&jpeg_tuned);
            let ss2_tuned = compute_ssim2(&pixels, &dec_tuned, width, height);
            let bfly_tuned = compute_butteraugli(&pixels, &dec_tuned, width, height);
            println!(
                "{}\t{}\txyb_tuned_v3\t-\t-\t-\t{}\t{:.2}\t{:.4}",
                name,
                q,
                jpeg_tuned.len(),
                ss2_tuned,
                bfly_tuned
            );

            // Sweep: vary Y with X=0.6, B=0.9 (based on YCbCr patterns)
            for &y_mul in &mul_values {
                let tables = xyb_uniform_bias(0.6, y_mul, 0.9);
                let jpeg = encode_xyb_with_tables(&pixels, width, height, q, &tables);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                let bfly = compute_butteraugli(&pixels, &dec, width, height);
                println!(
                    "{}\t{}\tsweep_y\t0.6\t{:.1}\t0.9\t{}\t{:.2}\t{:.4}",
                    name,
                    q,
                    y_mul,
                    jpeg.len(),
                    ss2,
                    bfly
                );
            }

            // Sweep: vary X with Y=0.5, B=0.9
            for &x_mul in &mul_values {
                let tables = xyb_uniform_bias(x_mul, 0.5, 0.9);
                let jpeg = encode_xyb_with_tables(&pixels, width, height, q, &tables);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                let bfly = compute_butteraugli(&pixels, &dec, width, height);
                println!(
                    "{}\t{}\tsweep_x\t{:.1}\t0.5\t0.9\t{}\t{:.2}\t{:.4}",
                    name,
                    q,
                    x_mul,
                    jpeg.len(),
                    ss2,
                    bfly
                );
            }

            // Sweep: vary B with X=0.6, Y=0.5
            for &b_mul in &mul_values {
                let tables = xyb_uniform_bias(0.6, 0.5, b_mul);
                let jpeg = encode_xyb_with_tables(&pixels, width, height, q, &tables);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                let bfly = compute_butteraugli(&pixels, &dec, width, height);
                println!(
                    "{}\t{}\tsweep_b\t0.6\t0.5\t{:.1}\t{}\t{:.2}\t{:.4}",
                    name,
                    q,
                    b_mul,
                    jpeg.len(),
                    ss2,
                    bfly
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
    println!(
        "image\tquality\tmode\tsize\tssim2\tbfly\tsize_vs_ycbcr\tssim2_vs_ycbcr\tbfly_vs_ycbcr\tms"
    );

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
            let report = |label: &str,
                          size: usize,
                          ss2: f64,
                          bfly: f64,
                          base_size: usize,
                          base_ss2: f64,
                          base_bfly: f64,
                          ms: f64| {
                let delta_pct = (size as f64 / base_size as f64 - 1.0) * 100.0;
                let delta_ss2 = ss2 - base_ss2;
                let delta_bfly = if base_bfly > 0.0 {
                    (bfly / base_bfly - 1.0) * 100.0
                } else {
                    0.0
                };
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{:.4}\t{:+.1}%\t{:+.2}\t{:+.1}%\t{:.0}",
                    name, q, label, size, ss2, bfly, delta_pct, delta_ss2, delta_bfly, ms
                );
            };

            // YCbCr baseline
            let t0 = Instant::now();
            let jpeg_ycbcr = encode_ycbcr(&pixels, width, height, q);
            let t_ycbcr = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_ycbcr = decode_jpeg_to_rgb_u8(&jpeg_ycbcr);
            let ss2_ycbcr = compute_ssim2(&pixels, &dec_ycbcr, width, height);
            let bfly_ycbcr = compute_butteraugli(&pixels, &dec_ycbcr, width, height);
            report(
                "ycbcr",
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_ycbcr,
            );

            // XYB baseline (flat 0.5)
            let t0 = Instant::now();
            let jpeg_xyb = encode_xyb(&pixels, width, height, q);
            let t_xyb = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_xyb = decode_jpeg_to_rgb_u8(&jpeg_xyb);
            let ss2_xyb = compute_ssim2(&pixels, &dec_xyb, width, height);
            let bfly_xyb = compute_butteraugli(&pixels, &dec_xyb, width, height);
            report(
                "xyb_0.5",
                jpeg_xyb.len(),
                ss2_xyb,
                bfly_xyb,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_xyb,
            );

            // XYB tuned v3 (frequency-dependent)
            let t0 = Instant::now();
            let tuned = xyb_tuned_zero_bias_v2(quality_to_distance(q));
            let jpeg_tuned = encode_xyb_with_tables(&pixels, width, height, q, &tuned);
            let t_tuned = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_tuned = decode_jpeg_to_rgb_u8(&jpeg_tuned);
            let ss2_tuned = compute_ssim2(&pixels, &dec_tuned, width, height);
            let bfly_tuned = compute_butteraugli(&pixels, &dec_tuned, width, height);
            report(
                "xyb_tuned_v3",
                jpeg_tuned.len(),
                ss2_tuned,
                bfly_tuned,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_tuned,
            );

            // Prefilter + XYB baseline
            let t0 = Instant::now();
            let filtered = prefilter_rgb(&pixels, width, height, 1.0, 5.0);
            let jpeg_pf_xyb = encode_xyb(&filtered, width, height, q);
            let t_pf = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_pf_xyb = decode_jpeg_to_rgb_u8(&jpeg_pf_xyb);
            // Compare decoded against ORIGINAL pixels
            let ss2_pf_xyb = compute_ssim2(&pixels, &dec_pf_xyb, width, height);
            let bfly_pf_xyb = compute_butteraugli(&pixels, &dec_pf_xyb, width, height);
            report(
                "prefilter+xyb",
                jpeg_pf_xyb.len(),
                ss2_pf_xyb,
                bfly_pf_xyb,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_pf,
            );

            // Prefilter + tuned v3
            let t0 = Instant::now();
            let jpeg_pf_tuned = encode_xyb_with_tables(&filtered, width, height, q, &tuned);
            let t_pf_tuned = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_pf_tuned = decode_jpeg_to_rgb_u8(&jpeg_pf_tuned);
            let ss2_pf_tuned = compute_ssim2(&pixels, &dec_pf_tuned, width, height);
            let bfly_pf_tuned = compute_butteraugli(&pixels, &dec_pf_tuned, width, height);
            report(
                "prefilter+tuned_v3",
                jpeg_pf_tuned.len(),
                ss2_pf_tuned,
                bfly_pf_tuned,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_pf_tuned,
            );

            // MSE-based perceptual loop (2 iterations)
            let t0 = Instant::now();
            let loop_tables = perceptual_loop_xyb(&pixels, width, height, q, 2);
            let jpeg_loop = encode_xyb_with_tables(&pixels, width, height, q, &loop_tables);
            let t_loop = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_loop = decode_jpeg_to_rgb_u8(&jpeg_loop);
            let ss2_loop = compute_ssim2(&pixels, &dec_loop, width, height);
            let bfly_loop = compute_butteraugli(&pixels, &dec_loop, width, height);
            report(
                "mse_loop_2",
                jpeg_loop.len(),
                ss2_loop,
                bfly_loop,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_loop,
            );

            // Butteraugli-guided perceptual loop (2 iterations)
            let t0 = Instant::now();
            let bfly_loop_tables = perceptual_loop_butteraugli(&pixels, width, height, q, 2);
            let jpeg_bfly_loop =
                encode_xyb_with_tables(&pixels, width, height, q, &bfly_loop_tables);
            let t_bfly_loop = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_bfly_loop = decode_jpeg_to_rgb_u8(&jpeg_bfly_loop);
            let ss2_bfly_loop = compute_ssim2(&pixels, &dec_bfly_loop, width, height);
            let bfly_bfly_loop = compute_butteraugli(&pixels, &dec_bfly_loop, width, height);
            report(
                "bfly_loop_2",
                jpeg_bfly_loop.len(),
                ss2_bfly_loop,
                bfly_bfly_loop,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_bfly_loop,
            );

            // Butteraugli-guided loop with 4 iterations
            let t0 = Instant::now();
            let bfly_loop4_tables = perceptual_loop_butteraugli(&pixels, width, height, q, 4);
            let jpeg_bfly_loop4 =
                encode_xyb_with_tables(&pixels, width, height, q, &bfly_loop4_tables);
            let t_bfly_loop4 = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_bfly_loop4 = decode_jpeg_to_rgb_u8(&jpeg_bfly_loop4);
            let ss2_bfly_loop4 = compute_ssim2(&pixels, &dec_bfly_loop4, width, height);
            let bfly_bfly_loop4 = compute_butteraugli(&pixels, &dec_bfly_loop4, width, height);
            report(
                "bfly_loop_4",
                jpeg_bfly_loop4.len(),
                ss2_bfly_loop4,
                bfly_bfly_loop4,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_bfly_loop4,
            );

            // Zensim-guided perceptual loop (2 iterations)
            let t0 = Instant::now();
            let zen_loop_tables = perceptual_loop_zensim(&pixels, width, height, q, 2);
            let jpeg_zen_loop =
                encode_xyb_with_tables(&pixels, width, height, q, &zen_loop_tables);
            let t_zen_loop = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_zen_loop = decode_jpeg_to_rgb_u8(&jpeg_zen_loop);
            let ss2_zen_loop = compute_ssim2(&pixels, &dec_zen_loop, width, height);
            let bfly_zen_loop = compute_butteraugli(&pixels, &dec_zen_loop, width, height);
            report(
                "zensim_loop_2",
                jpeg_zen_loop.len(),
                ss2_zen_loop,
                bfly_zen_loop,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_zen_loop,
            );

            eprintln!("  Q{} done", q);
        }
    }
}

// ============================================================================
// Mode 3: Prefilter-only comparison
// ============================================================================

fn run_prefilter(paths: &[String]) {
    println!(
        "image\tquality\tmode\tsize\tssim2\tbfly\tbfly_x2\tbfly_x4\tsize_vs_base\tssim2_vs_base\tbfly_vs_base"
    );

    let qualities = [75u8, 85, 95];
    let configs = [
        ("vlight", 0.5f32, 8.0f32),
        ("light", 1.0, 5.0),
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
            let bfly_ycbcr = compute_butteraugli(&pixels, &dec_ycbcr, width, height);
            let bfly_ycbcr_x2 = compute_butteraugli_scaled(&pixels, &dec_ycbcr, width, height, 2);
            let bfly_ycbcr_x4 = compute_butteraugli_scaled(&pixels, &dec_ycbcr, width, height, 4);
            println!(
                "{}\t{}\tycbcr\t{}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t-\t-\t-",
                name,
                q,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                bfly_ycbcr_x2,
                bfly_ycbcr_x4
            );

            let jpeg_xyb = encode_xyb(&pixels, width, height, q);
            let dec_xyb = decode_jpeg_to_rgb_u8(&jpeg_xyb);
            let ss2_xyb = compute_ssim2(&pixels, &dec_xyb, width, height);
            let bfly_xyb = compute_butteraugli(&pixels, &dec_xyb, width, height);
            let bfly_xyb_x2 = compute_butteraugli_scaled(&pixels, &dec_xyb, width, height, 2);
            let bfly_xyb_x4 = compute_butteraugli_scaled(&pixels, &dec_xyb, width, height, 4);
            println!(
                "{}\t{}\txyb\t{}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t-\t-\t-",
                name,
                q,
                jpeg_xyb.len(),
                ss2_xyb,
                bfly_xyb,
                bfly_xyb_x2,
                bfly_xyb_x4
            );

            for &(label, sigma, noise_floor) in &configs {
                let filtered = prefilter_rgb(&pixels, width, height, sigma, noise_floor);

                // Prefilter + YCbCr
                let jpeg = encode_ycbcr(&filtered, width, height, q);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                let bfly = compute_butteraugli(&pixels, &dec, width, height);
                let bfly_x2 = compute_butteraugli_scaled(&pixels, &dec, width, height, 2);
                let bfly_x4 = compute_butteraugli_scaled(&pixels, &dec, width, height, 4);
                let d_size = (jpeg.len() as f64 / jpeg_ycbcr.len() as f64 - 1.0) * 100.0;
                let d_ss2 = ss2 - ss2_ycbcr;
                let d_bfly = if bfly_ycbcr > 0.0 {
                    (bfly / bfly_ycbcr - 1.0) * 100.0
                } else {
                    0.0
                };
                println!(
                    "{}\t{}\tpf_{}_ycbcr\t{}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:+.1}%\t{:+.2}\t{:+.1}%",
                    name,
                    q,
                    label,
                    jpeg.len(),
                    ss2,
                    bfly,
                    bfly_x2,
                    bfly_x4,
                    d_size,
                    d_ss2,
                    d_bfly
                );

                // Prefilter + XYB
                let jpeg = encode_xyb(&filtered, width, height, q);
                let dec = decode_jpeg_to_rgb_u8(&jpeg);
                let ss2 = compute_ssim2(&pixels, &dec, width, height);
                let bfly = compute_butteraugli(&pixels, &dec, width, height);
                let bfly_x2 = compute_butteraugli_scaled(&pixels, &dec, width, height, 2);
                let bfly_x4 = compute_butteraugli_scaled(&pixels, &dec, width, height, 4);
                let d_size = (jpeg.len() as f64 / jpeg_xyb.len() as f64 - 1.0) * 100.0;
                let d_ss2 = ss2 - ss2_xyb;
                let d_bfly = if bfly_xyb > 0.0 {
                    (bfly / bfly_xyb - 1.0) * 100.0
                } else {
                    0.0
                };
                println!(
                    "{}\t{}\tpf_{}_xyb\t{}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:+.1}%\t{:+.2}\t{:+.1}%",
                    name,
                    q,
                    label,
                    jpeg.len(),
                    ss2,
                    bfly,
                    bfly_x2,
                    bfly_x4,
                    d_size,
                    d_ss2,
                    d_bfly
                );
            }
        }
    }
}

// ============================================================================
// Perceptual feedback loop (MSE-based, original)
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
                "  mse loop iter {}: decode size mismatch ({} vs {}), skipping",
                iter,
                decoded.len(),
                pixels.len()
            );
            break;
        }

        let block_mse = compute_block_mse(pixels, &decoded, width, height);
        let avg_mse: f32 = block_mse.iter().sum::<f32>() / block_mse.len() as f32;
        let max_mse = block_mse.iter().copied().fold(0.0f32, f32::max);

        eprintln!(
            "  mse loop iter {}: avg_mse={:.1} max_mse={:.1} jpeg_size={}",
            iter,
            avg_mse,
            max_mse,
            jpeg.len()
        );

        if avg_mse < 1.0 {
            break;
        }

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

// ============================================================================
// Butteraugli-guided perceptual feedback loop
// ============================================================================

/// Compute per-block error from butteraugli diffmap using L4 norm.
/// L4 emphasizes blocks with high peak error (like libjxl's L16 but less extreme).
fn block_errors_from_diffmap(
    diffmap: &imgref::ImgRef<'_, f32>,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let bw = (width + 7) / 8;
    let bh = (height + 7) / 8;
    let mut block_errors = vec![0.0f32; bw * bh];

    for by in 0..bh {
        for bx in 0..bw {
            let mut sum4 = 0.0f64;
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
                    let v = diffmap[(x, y)] as f64;
                    let v2 = v * v;
                    sum4 += v2 * v2; // L4 norm
                    count += 1;
                }
            }
            block_errors[by * bw + bx] = if count > 0 {
                (sum4 / count as f64).powf(0.25) as f32
            } else {
                0.0
            };
        }
    }
    block_errors
}

/// Butteraugli-guided perceptual loop.
///
/// Like the MSE loop but uses actual butteraugli diffmap for per-block error.
/// This gives perceptually weighted spatial error: blocks with high butteraugli
/// error get more precision (lower zero-bias mul), blocks with low error get
/// more aggressive zeroing.
///
/// The adjustment is sum-preserving: total zero-bias energy is held constant
/// to avoid file size drift.
fn perceptual_loop_butteraugli(
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

    let params = ButteraugliParams::default().with_compute_diffmap(true);

    // Build reference image once (sRGB u8 → RGB8 pixels for butteraugli)
    let orig_pixels: Vec<rgb::RGB8> = pixels
        .chunks_exact(3)
        .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = imgref::Img::new(&orig_pixels[..], width, height);

    for iter in 0..iters {
        let jpeg = encode_xyb_with_tables(pixels, width, height, quality, &tables);
        let decoded = decode_jpeg_to_rgb_u8(&jpeg);
        if decoded.len() != pixels.len() {
            eprintln!("  bfly loop iter {}: decode size mismatch, skipping", iter,);
            break;
        }

        // Compute butteraugli with diffmap
        let dec_pixels: Vec<rgb::RGB8> = decoded
            .chunks_exact(3)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect();
        let dec_img = imgref::Img::new(&dec_pixels[..], width, height);
        let result = match butteraugli::butteraugli(orig_img, dec_img, &params) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("  bfly loop iter {}: butteraugli failed, stopping", iter);
                break;
            }
        };

        let diffmap = match &result.diffmap {
            Some(dm) => dm,
            None => {
                eprintln!("  bfly loop iter {}: no diffmap returned", iter);
                break;
            }
        };

        let block_errors = block_errors_from_diffmap(&diffmap.as_ref(), width, height);
        let avg_error: f32 = block_errors.iter().sum::<f32>() / block_errors.len() as f32;
        let max_error = block_errors.iter().copied().fold(0.0f32, f32::max);

        eprintln!(
            "  bfly loop iter {}: score={:.4} avg_block={:.4} max_block={:.4} size={}",
            iter,
            result.score,
            avg_error,
            max_error,
            jpeg.len()
        );

        if avg_error < 0.01 {
            break;
        }

        // Sum-preserving redistribution of zero-bias mul tables.
        // High-error blocks → lower mul (keep more coefficients).
        // Low-error blocks → higher mul (zero more aggressively).
        // K_ALPHA controls aggressiveness. Start mild (0.10), like zensim.
        let k_alpha = 0.10;

        // Compute per-block adjustment factors
        let mut factors = vec![1.0f32; num_blocks];
        for bi in 0..num_blocks {
            let ratio = if avg_error > 0.0 {
                block_errors[bi] / avg_error
            } else {
                1.0
            };
            // High error → factor > 1 → we want LOWER mul → divide by factor
            factors[bi] = (1.0 + k_alpha * (ratio - 1.0)).clamp(0.7, 1.5);
        }

        // Aggregate block factors into frequency-band adjustments.
        // JPEG has global tables, so we can't adjust per-block.
        // Strategy: partition blocks by error level, compute average factor
        // per frequency band for high-error vs low-error regions, then
        // shift the global table toward preserving high-error frequencies.
        //
        // Split into 3 frequency bands:
        //   - Low (positions 1-7): DC-adjacent, most important
        //   - Mid (positions 8-31): mid frequencies
        //   - High (positions 32-63): high frequencies, least important
        let error_threshold_high = avg_error * 1.3;
        let error_threshold_low = avg_error * 0.7;

        let mut high_factor_sum = 0.0f32;
        let mut low_factor_sum = 0.0f32;
        let mut high_count = 0usize;
        let mut low_count = 0usize;

        for bi in 0..num_blocks {
            if block_errors[bi] > error_threshold_high {
                high_factor_sum += factors[bi];
                high_count += 1;
            } else if block_errors[bi] < error_threshold_low {
                low_factor_sum += factors[bi];
                low_count += 1;
            }
        }

        if high_count == 0 || low_count == 0 {
            continue;
        }

        let high_avg_factor = high_factor_sum / high_count as f32;
        let low_avg_factor = low_factor_sum / low_count as f32;

        // The ratio tells us how much more precision high-error blocks need.
        // Apply this as a frequency-dependent adjustment: high frequencies
        // get a stronger shift because they're where zero-bias has the most effect.
        let ratio = high_avg_factor / low_avg_factor;

        // Apply per-band: low-freq gets mild adjustment, high-freq gets stronger
        for c in 0..3 {
            let mul = tables.zero_bias_mul.get_mut(c);
            let sum_before: f32 = mul[1..].iter().sum();

            // Low band (1-7): mild — these are critical, don't change much
            for k in 1..8 {
                mul[k] /= 1.0 + 0.3 * (ratio - 1.0);
            }
            // Mid band (8-31): moderate
            for k in 8..32 {
                mul[k] /= 1.0 + 0.6 * (ratio - 1.0);
            }
            // High band (32-63): strongest
            for k in 32..64 {
                mul[k] /= ratio;
            }

            // Renormalize to preserve total zero-bias energy (controls file size)
            let sum_after: f32 = mul[1..].iter().sum();
            if sum_after > 0.0 {
                let renorm = sum_before / sum_after;
                for k in 1..64 {
                    mul[k] *= renorm;
                }
            }
        }
    }

    tables
}

// ============================================================================
// Zensim-guided perceptual feedback loop
// ============================================================================

/// Compute per-block L4 error from a flat diffmap (row-major f32 slice).
///
/// Same approach as `block_errors_from_diffmap` but works with zensim's flat
/// `&[f32]` diffmap instead of butteraugli's `ImgRef<f32>`.
fn block_errors_from_flat_diffmap(
    diffmap: &[f32],
    dm_width: usize,
    _dm_height: usize,
    img_width: usize,
    img_height: usize,
) -> Vec<f32> {
    let bw = (img_width + 7) / 8;
    let bh = (img_height + 7) / 8;
    let mut block_errors = vec![0.0f32; bw * bh];

    for by in 0..bh {
        for bx in 0..bw {
            let mut sum4 = 0.0f64;
            let mut count = 0u32;
            for dy in 0..8 {
                let y = by * 8 + dy;
                if y >= img_height {
                    break;
                }
                for dx in 0..8 {
                    let x = bx * 8 + dx;
                    if x >= img_width {
                        break;
                    }
                    let v = diffmap[y * dm_width + x] as f64;
                    let v2 = v * v;
                    sum4 += v2 * v2; // L4 norm
                    count += 1;
                }
            }
            block_errors[by * bw + bx] = if count > 0 {
                (sum4 / count as f64).powf(0.25) as f32
            } else {
                0.0
            };
        }
    }
    block_errors
}

/// Zensim-guided perceptual loop.
///
/// Uses zensim's psychovisual diffmap for per-block error instead of
/// butteraugli or MSE. Key advantages:
/// - Much faster than butteraugli (~2-3x)
/// - Optimizes for SSIM2 (the metric we care about most)
/// - Includes edge artifact and HF texture features
///
/// Same sum-preserving redistribution as the butteraugli loop.
fn perceptual_loop_zensim(
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

    // Set up zensim with precomputed reference
    let z = zensim::Zensim::new(zensim::ZensimProfile::latest()).with_parallel(false);
    let stride = width * 3;
    let ref_img = zensim::StridedBytes::new(
        pixels,
        width,
        height,
        stride,
        zensim::PixelFormat::Srgb8Rgb,
    );
    let precomputed = match z.precompute_reference(&ref_img) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  zensim precompute failed: {}", e);
            return tables;
        }
    };

    let diffmap_opts = zensim::DiffmapOptions {
        weighting: zensim::DiffmapWeighting::Trained,
        masking_strength: Some(8.0),
        sqrt: false,
        include_hf: true,
        include_edge_mse: true,
    };

    for iter in 0..iters {
        let jpeg = encode_xyb_with_tables(pixels, width, height, quality, &tables);
        let decoded = decode_jpeg_to_rgb_u8(&jpeg);
        if decoded.len() != pixels.len() {
            eprintln!(
                "  zensim loop iter {}: decode size mismatch, skipping",
                iter
            );
            break;
        }

        // Compute zensim diffmap
        let dec_img = zensim::StridedBytes::new(
            &decoded,
            width,
            height,
            stride,
            zensim::PixelFormat::Srgb8Rgb,
        );
        let dm_result =
            match z.compute_with_ref_and_diffmap(&precomputed, &dec_img, diffmap_opts) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  zensim loop iter {}: diffmap failed: {}", iter, e);
                    break;
                }
            };

        let score = dm_result.score();
        let diffmap = dm_result.diffmap();

        // Per-block L4 error from diffmap
        let block_errors = block_errors_from_flat_diffmap(
            diffmap,
            dm_result.width(),
            dm_result.height(),
            width,
            height,
        );
        let avg_error: f32 = block_errors.iter().sum::<f32>() / block_errors.len() as f32;
        let max_error = block_errors.iter().copied().fold(0.0f32, f32::max);

        eprintln!(
            "  zensim loop iter {}: score={:.2} avg_block={:.4} max_block={:.4} size={}",
            iter, score, avg_error, max_error, jpeg.len()
        );

        // No early exit — zensim diffmap values are small (0.0001-0.002 range)
        // but the relative distribution (ratio high/low) is what drives redistribution.

        // Sum-preserving redistribution of zero-bias mul tables.
        // Same approach as butteraugli loop: partition blocks by error level,
        // adjust frequency bands differently.
        let k_alpha = 0.15; // Slightly more aggressive than bfly (0.10) since zensim is SSIM-tuned
        let mut factors = vec![1.0f32; num_blocks];
        for bi in 0..num_blocks {
            let ratio = if avg_error > 0.0 {
                block_errors[bi] / avg_error
            } else {
                1.0
            };
            factors[bi] = (1.0 + k_alpha * (ratio - 1.0)).clamp(0.7, 1.5);
        }

        let error_threshold_high = avg_error * 1.3;
        let error_threshold_low = avg_error * 0.7;

        let mut high_factor_sum = 0.0f32;
        let mut low_factor_sum = 0.0f32;
        let mut high_count = 0usize;
        let mut low_count = 0usize;

        for bi in 0..num_blocks {
            if block_errors[bi] > error_threshold_high {
                high_factor_sum += factors[bi];
                high_count += 1;
            } else if block_errors[bi] < error_threshold_low {
                low_factor_sum += factors[bi];
                low_count += 1;
            }
        }

        if high_count == 0 || low_count == 0 {
            continue;
        }

        let high_avg_factor = high_factor_sum / high_count as f32;
        let low_avg_factor = low_factor_sum / low_count as f32;
        let ratio = high_avg_factor / low_avg_factor;

        // Apply per-band adjustment (same as bfly loop)
        for c in 0..3 {
            let mul = tables.zero_bias_mul.get_mut(c);
            let sum_before: f32 = mul[1..].iter().sum();

            // Low band (1-7): mild — critical frequencies
            for k in 1..8 {
                mul[k] /= 1.0 + 0.3 * (ratio - 1.0);
            }
            // Mid band (8-31): moderate
            for k in 8..32 {
                mul[k] /= 1.0 + 0.6 * (ratio - 1.0);
            }
            // High band (32-63): strongest
            for k in 32..64 {
                mul[k] /= ratio;
            }

            // Renormalize to preserve total zero-bias energy
            let sum_after: f32 = mul[1..].iter().sum();
            if sum_after > 0.0 {
                let renorm = sum_before / sum_after;
                for k in 1..64 {
                    mul[k] *= renorm;
                }
            }
        }
    }

    tables
}

fn compute_block_mse(original: &[u8], decoded: &[u8], width: usize, height: usize) -> Vec<f32> {
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
// Mode 4: Per-image DQT seeding from spectral analysis
// ============================================================================

/// Compute per-position DCT coefficient statistics in XYB color space.
///
/// Returns (median_abs, mad) for each of 3 channels × 64 positions.
/// median_abs[c][k] = median of |coeff_k| across all blocks for channel c.
/// This is the Laplacian scale parameter b_k (up to ln(2) factor).
fn compute_xyb_dct_statistics(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> ([[f32; 64]; 3], [[f32; 64]; 3]) {
    let bw = (width + 7) / 8;
    let bh = (height + 7) / 8;
    let num_blocks = bw * bh;

    // Convert entire image to XYB planes (f32)
    let npix = width * height;
    let mut x_plane = vec![0.0f32; npix];
    let mut y_plane = vec![0.0f32; npix];
    let mut b_plane = vec![0.0f32; npix];

    for i in 0..npix {
        let r = pixels[i * 3];
        let g = pixels[i * 3 + 1];
        let b = pixels[i * 3 + 2];
        let (x, y, bv) = srgb_to_xyb(r, g, b);
        x_plane[i] = x;
        y_plane[i] = y;
        b_plane[i] = bv;
    }

    let planes = [&x_plane, &y_plane, &b_plane];

    // Collect |coeff_k| for all blocks, per channel, per position
    // Use a flat buffer: coeffs[c][k] = Vec of |values| across all blocks
    let mut coeffs: [Vec<Vec<f32>>; 3] = [
        (0..64).map(|_| Vec::with_capacity(num_blocks)).collect(),
        (0..64).map(|_| Vec::with_capacity(num_blocks)).collect(),
        (0..64).map(|_| Vec::with_capacity(num_blocks)).collect(),
    ];

    for c in 0..3 {
        let plane = planes[c];
        for by in 0..bh {
            for bx in 0..bw {
                // Extract 8x8 block (with edge replication)
                let mut block = [0.0f32; 64];
                for dy in 0..8 {
                    let y = (by * 8 + dy).min(height - 1);
                    for dx in 0..8 {
                        let x = (bx * 8 + dx).min(width - 1);
                        block[dy * 8 + dx] = plane[y * width + x];
                    }
                }

                // Forward DCT
                let dct = forward_dct_8x8(&block);

                // Collect absolute values (skip DC for AC statistics)
                for k in 0..64 {
                    coeffs[c][k].push(dct[k].abs());
                }
            }
        }
    }

    // Compute median and MAD for each position
    let mut median_abs = [[0.0f32; 64]; 3];
    let mut mad = [[0.0f32; 64]; 3];

    for c in 0..3 {
        for k in 0..64 {
            let vals = &mut coeffs[c][k];
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let n = vals.len();
            if n == 0 {
                continue;
            }
            let med = if n % 2 == 0 {
                (vals[n / 2 - 1] + vals[n / 2]) / 2.0
            } else {
                vals[n / 2]
            };
            median_abs[c][k] = med;

            // MAD (median absolute deviation from median)
            let mut deviations: Vec<f32> = vals.iter().map(|&v| (v - med).abs()).collect();
            deviations.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            mad[c][k] = if n % 2 == 0 {
                (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0
            } else {
                deviations[n / 2]
            };
        }
    }

    (median_abs, mad)
}

/// Per-image DQT seeding using spectral analysis.
///
/// Mathematical foundation: For Laplacian-distributed DCT coefficients
/// with scale b_k and perceptual weight w_k, the RD-optimal quantization
/// step is q_k = C * b_k / w_k.
///
/// We use a hybrid approach: scale the corpus-tuned tables by the ratio
/// of this image's spectral profile to the average corpus profile.
///
/// seed_q[k] = corpus_q[k] * (b_k / b_k_ref)^alpha
///
/// alpha controls adaptation strength:
///   0.0 = pure corpus (ignore image statistics)
///   0.5 = geometric mean (balanced)
///   1.0 = full per-image (maximum adaptation)
fn per_image_dqt_seed(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    alpha: f32,
) -> EncodingTables {
    let distance = quality_to_distance(quality);

    // Step 1: Compute this image's DCT coefficient statistics
    let (median_abs, _mad) = compute_xyb_dct_statistics(pixels, width, height);

    // Step 2: Get the corpus base tables and compute what DQT values they produce
    let corpus_tables = EncodingTables::default_xyb();
    let (qt0, qt1, qt2) = corpus_tables.generate_quant_tables(distance, false); // XYB uses S444

    let corpus_dqt: [Vec<u16>; 3] = [
        qt0.values.to_vec(),
        qt1.values.to_vec(),
        qt2.values.to_vec(),
    ];

    // Step 3: Compute reference spectral profile (what the corpus tables "expect").
    // We derive b_k_ref from the corpus quant values and CSF weights.
    // Since corpus_q[k] = base[k] * scale, and optimal q_k ∝ b_k / w_k,
    // we can use the corpus median as the reference.
    // For simplicity: use the geometric mean of all images' median_abs as reference.
    // Since we don't have the corpus statistics, we use a heuristic:
    // b_k_ref = corpus_dqt[k] * normalization_factor
    // This means the adaptation ratio becomes b_k_actual / (corpus_dqt[k] * norm).
    //
    // Better approach: use the image's own statistics directly.
    // ratio[k] = median_abs[k] / geomean(median_abs), then apply to corpus DQT.
    // This preserves the corpus's absolute calibration while adapting the shape.

    // Start from tuned tables (not default_xyb) so zero-bias matches baseline
    let mut tables = xyb_tuned_zero_bias_v2(distance);

    for c in 0..3 {
        // Compute geometric mean of this image's coefficient magnitudes (AC only)
        let mut log_sum = 0.0f64;
        let mut count = 0;
        for k in 1..64 {
            let v = median_abs[c][k] as f64;
            if v > 1e-10 {
                log_sum += v.ln();
                count += 1;
            }
        }
        let geomean = if count > 0 {
            (log_sum / count as f64).exp() as f32
        } else {
            1.0
        };

        // Similarly for the corpus DQT values (geometric mean of AC)
        let mut log_sum_corpus = 0.0f64;
        let mut count_corpus = 0;
        for k in 1..64 {
            let v = corpus_dqt[c][k] as f64;
            if v > 1.0 {
                log_sum_corpus += v.ln();
                count_corpus += 1;
            }
        }
        let geomean_corpus = if count_corpus > 0 {
            (log_sum_corpus / count_corpus as f64).exp() as f32
        } else {
            1.0
        };

        // Compute per-position adaptation ratio.
        // ratio[k] = (median_abs[k] / geomean) / (corpus_dqt[k] / geomean_corpus)
        // This normalizes both profiles to their geometric means, so we're comparing
        // the *shape* of the spectral distribution, not the absolute level.
        let mut adapted_quant = [0.0f32; 64];

        // DC: keep corpus value (DC is well-calibrated in corpus tables)
        adapted_quant[0] = corpus_dqt[c][0] as f32;

        for k in 1..64 {
            let b_k = median_abs[c][k];
            let b_k_norm = if geomean > 1e-10 { b_k / geomean } else { 1.0 };

            let corpus_k = corpus_dqt[c][k] as f32;
            let corpus_k_norm = if geomean_corpus > 1.0 {
                corpus_k / geomean_corpus
            } else {
                1.0
            };

            // Spectral shape ratio: how much more energy this image has at position k
            // relative to the corpus average at position k
            let shape_ratio = if corpus_k_norm > 0.01 {
                b_k_norm / corpus_k_norm
            } else {
                1.0
            };

            // Apply adaptation: q_new = q_corpus * ratio^alpha
            // alpha=0: no adaptation (shape_ratio^0 = 1)
            // alpha=0.5: moderate adaptation
            // alpha=1: full adaptation
            let factor = shape_ratio.powf(alpha).clamp(0.3, 3.0);
            adapted_quant[k] = (corpus_k * factor).round().clamp(1.0, 65535.0);
        }

        // Store as exact quant values (bypass scaling)
        tables.quant.get_mut(c).copy_from_slice(&adapted_quant);
    }

    // Use Exact scaling since we've already computed final DQT values
    tables.scaling = ScalingParams::Exact;

    tables
}

/// Per-image DQT with iterative refinement.
///
/// Starts from spectral-seeded DQT, then uses encode→measure→adjust loop
/// to refine. The per-coefficient error contribution guides adjustment.
fn per_image_dqt_iterative(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    alpha: f32,
    iters: usize,
) -> EncodingTables {
    let mut tables = per_image_dqt_seed(pixels, width, height, quality, alpha);

    // Set up zensim for error measurement
    let z = zensim::Zensim::new(zensim::ZensimProfile::latest()).with_parallel(false);
    let stride = width * 3;
    let ref_img = zensim::StridedBytes::new(
        pixels,
        width,
        height,
        stride,
        zensim::PixelFormat::Srgb8Rgb,
    );
    let precomputed = match z.precompute_reference(&ref_img) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  zensim precompute failed: {}", e);
            return tables;
        }
    };
    let diffmap_opts = zensim::DiffmapOptions {
        weighting: zensim::DiffmapWeighting::Trained,
        masking_strength: Some(8.0),
        sqrt: false,
        include_hf: true,
        include_edge_mse: true,
    };

    for iter in 0..iters {
        // Encode with current tables
        let jpeg = encode_xyb_with_tables(pixels, width, height, quality, &tables);
        let decoded = decode_jpeg_to_rgb_u8(&jpeg);
        if decoded.len() != pixels.len() {
            eprintln!("  dqt iter {}: decode size mismatch", iter);
            break;
        }

        // Get per-pixel error via zensim diffmap
        let dec_img = zensim::StridedBytes::new(
            &decoded,
            width,
            height,
            stride,
            zensim::PixelFormat::Srgb8Rgb,
        );
        let dm_result =
            match z.compute_with_ref_and_diffmap(&precomputed, &dec_img, diffmap_opts) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  dqt iter {}: diffmap failed: {}", iter, e);
                    break;
                }
            };

        let score = dm_result.score();
        eprintln!(
            "  dqt iter {}: score={:.2} size={}",
            iter,
            score,
            jpeg.len()
        );

        // Compute per-block error and per-position coefficient error contribution.
        // Strategy: blocks with high error → their dominant frequencies need smaller quant steps.
        // Compute per-position "error weight" by correlating block error with coefficient energy.
        let bw = (width + 7) / 8;
        let bh = (height + 7) / 8;
        let diffmap = dm_result.diffmap();

        let block_errors = block_errors_from_flat_diffmap(
            diffmap,
            dm_result.width(),
            dm_result.height(),
            width,
            height,
        );
        let avg_error: f32 = block_errors.iter().sum::<f32>() / block_errors.len() as f32;

        // For each frequency position, compute the weighted error contribution:
        // error_weight[k] = Σ (block_error[b] * |coeff_b_k|) / Σ |coeff_b_k|
        // This tells us which frequency positions contribute most to perceptual error.
        // Then adjust: positions with error_weight > average → decrease quant step.
        let planes = {
            let npix = width * height;
            let mut x_plane = vec![0.0f32; npix];
            let mut y_plane = vec![0.0f32; npix];
            let mut b_plane = vec![0.0f32; npix];
            for i in 0..npix {
                let (x, y, bv) = srgb_to_xyb(pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]);
                x_plane[i] = x;
                y_plane[i] = y;
                b_plane[i] = bv;
            }
            [x_plane, y_plane, b_plane]
        };

        for c in 0..3 {
            let plane = &planes[c];
            let mut error_energy = [0.0f64; 64]; // sum of (block_error * |coeff|)
            let mut total_energy = [0.0f64; 64]; // sum of |coeff|

            for by in 0..bh {
                for bx in 0..bw {
                    let bi = by * bw + bx;
                    let be = block_errors[bi] as f64;

                    let mut block = [0.0f32; 64];
                    for dy in 0..8 {
                        let y = (by * 8 + dy).min(height - 1);
                        for dx in 0..8 {
                            let x = (bx * 8 + dx).min(width - 1);
                            block[dy * 8 + dx] = plane[y * width + x];
                        }
                    }
                    let dct = forward_dct_8x8(&block);

                    for k in 1..64 {
                        let abs_c = dct[k].abs() as f64;
                        error_energy[k] += be * abs_c;
                        total_energy[k] += abs_c;
                    }
                }
            }

            // Compute per-position error weight
            let mut error_weight = [0.0f32; 64];
            for k in 1..64 {
                error_weight[k] = if total_energy[k] > 1e-10 {
                    (error_energy[k] / total_energy[k]) as f32
                } else {
                    avg_error
                };
            }

            let avg_ew: f32 = error_weight[1..].iter().sum::<f32>() / 63.0;

            // Adjust quant values: positions with above-average error → decrease step
            let quant = tables.quant.get_mut(c);
            let k_adjust = 0.15; // conservative per iteration

            for k in 1..64 {
                let ratio = if avg_ew > 0.0 {
                    error_weight[k] / avg_ew
                } else {
                    1.0
                };
                // ratio > 1: this position contributes more error → need finer quant
                // ratio < 1: this position contributes less → can coarsen
                let factor = 1.0 / (1.0 + k_adjust * (ratio - 1.0));
                let factor = factor.clamp(0.85, 1.15);
                quant[k] = (quant[k] * factor).round().clamp(1.0, 65535.0);
            }
        }
    }

    tables
}

fn run_dqt_benchmark(paths: &[String]) {
    println!(
        "image\tquality\tmode\tsize\tssim2\tbfly\tsize_vs_ycbcr\tssim2_vs_ycbcr\tbfly_vs_ycbcr\tms"
    );

    let qualities = [75u8, 85, 95];
    let alphas = [0.0f32, 0.3, 0.5, 0.7, 1.0];

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
            let report = |label: &str,
                          size: usize,
                          ss2: f64,
                          bfly: f64,
                          base_size: usize,
                          base_ss2: f64,
                          base_bfly: f64,
                          ms: f64| {
                let delta_pct = (size as f64 / base_size as f64 - 1.0) * 100.0;
                let delta_ss2 = ss2 - base_ss2;
                let delta_bfly = if base_bfly > 0.0 {
                    (bfly / base_bfly - 1.0) * 100.0
                } else {
                    0.0
                };
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{:.4}\t{:+.1}%\t{:+.2}\t{:+.1}%\t{:.0}",
                    name, q, label, size, ss2, bfly, delta_pct, delta_ss2, delta_bfly, ms
                );
            };

            // YCbCr baseline
            let t0 = Instant::now();
            let jpeg_ycbcr = encode_ycbcr(&pixels, width, height, q);
            let t_ycbcr = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_ycbcr = decode_jpeg_to_rgb_u8(&jpeg_ycbcr);
            let ss2_ycbcr = compute_ssim2(&pixels, &dec_ycbcr, width, height);
            let bfly_ycbcr = compute_butteraugli(&pixels, &dec_ycbcr, width, height);
            report(
                "ycbcr",
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_ycbcr,
            );

            // XYB tuned v3 (corpus tables, our current best)
            let t0 = Instant::now();
            let tuned = xyb_tuned_zero_bias_v2(quality_to_distance(q));
            let jpeg_tuned = encode_xyb_with_tables(&pixels, width, height, q, &tuned);
            let t_tuned = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_tuned = decode_jpeg_to_rgb_u8(&jpeg_tuned);
            let ss2_tuned = compute_ssim2(&pixels, &dec_tuned, width, height);
            let bfly_tuned = compute_butteraugli(&pixels, &dec_tuned, width, height);
            report(
                "xyb_tuned_v3",
                jpeg_tuned.len(),
                ss2_tuned,
                bfly_tuned,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_tuned,
            );

            // Exact-mode reproduction of tuned_v3 (should match exactly)
            {
                let t0 = Instant::now();
                let distance = quality_to_distance(q);
                let mut exact_tables = xyb_tuned_zero_bias_v2(distance);
                let (qt0, qt1, qt2) = exact_tables.generate_quant_tables(distance, false);
                for (k, &v) in qt0.values.iter().enumerate() {
                    exact_tables.quant.get_mut(0)[k] = v as f32;
                }
                for (k, &v) in qt1.values.iter().enumerate() {
                    exact_tables.quant.get_mut(1)[k] = v as f32;
                }
                for (k, &v) in qt2.values.iter().enumerate() {
                    exact_tables.quant.get_mut(2)[k] = v as f32;
                }
                exact_tables.scaling = ScalingParams::Exact;
                let jpeg_exact =
                    encode_xyb_with_tables(&pixels, width, height, q, &exact_tables);
                let t_exact = t0.elapsed().as_secs_f64() * 1000.0;
                let dec_exact = decode_jpeg_to_rgb_u8(&jpeg_exact);
                let ss2_exact = compute_ssim2(&pixels, &dec_exact, width, height);
                let bfly_exact = compute_butteraugli(&pixels, &dec_exact, width, height);
                report(
                    "exact_tuned",
                    jpeg_exact.len(),
                    ss2_exact,
                    bfly_exact,
                    jpeg_ycbcr.len(),
                    ss2_ycbcr,
                    bfly_ycbcr,
                    t_exact,
                );
            }

            // Per-image DQT at various alpha values
            for &alpha in &alphas {
                let label = format!("dqt_a{:.1}", alpha);
                let t0 = Instant::now();
                let dqt_tables = per_image_dqt_seed(&pixels, width, height, q, alpha);
                let jpeg_dqt =
                    encode_xyb_with_tables(&pixels, width, height, q, &dqt_tables);
                let t_dqt = t0.elapsed().as_secs_f64() * 1000.0;
                let dec_dqt = decode_jpeg_to_rgb_u8(&jpeg_dqt);
                let ss2_dqt = compute_ssim2(&pixels, &dec_dqt, width, height);
                let bfly_dqt = compute_butteraugli(&pixels, &dec_dqt, width, height);
                report(
                    &label,
                    jpeg_dqt.len(),
                    ss2_dqt,
                    bfly_dqt,
                    jpeg_ycbcr.len(),
                    ss2_ycbcr,
                    bfly_ycbcr,
                    t_dqt,
                );
            }

            // Per-image DQT + iterative refinement (alpha=0.5, 2 iterations)
            let t0 = Instant::now();
            let dqt_iter_tables =
                per_image_dqt_iterative(&pixels, width, height, q, 0.5, 2);
            let jpeg_dqt_iter =
                encode_xyb_with_tables(&pixels, width, height, q, &dqt_iter_tables);
            let t_dqt_iter = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_dqt_iter = decode_jpeg_to_rgb_u8(&jpeg_dqt_iter);
            let ss2_dqt_iter = compute_ssim2(&pixels, &dec_dqt_iter, width, height);
            let bfly_dqt_iter =
                compute_butteraugli(&pixels, &dec_dqt_iter, width, height);
            report(
                "dqt_iter_a0.5",
                jpeg_dqt_iter.len(),
                ss2_dqt_iter,
                bfly_dqt_iter,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_dqt_iter,
            );

            // Per-image DQT + iterative refinement (alpha=0.7, 3 iterations)
            let t0 = Instant::now();
            let dqt_iter3_tables =
                per_image_dqt_iterative(&pixels, width, height, q, 0.7, 3);
            let jpeg_dqt_iter3 =
                encode_xyb_with_tables(&pixels, width, height, q, &dqt_iter3_tables);
            let t_dqt_iter3 = t0.elapsed().as_secs_f64() * 1000.0;
            let dec_dqt_iter3 = decode_jpeg_to_rgb_u8(&jpeg_dqt_iter3);
            let ss2_dqt_iter3 = compute_ssim2(&pixels, &dec_dqt_iter3, width, height);
            let bfly_dqt_iter3 =
                compute_butteraugli(&pixels, &dec_dqt_iter3, width, height);
            report(
                "dqt_iter_a0.7x3",
                jpeg_dqt_iter3.len(),
                ss2_dqt_iter3,
                bfly_dqt_iter3,
                jpeg_ycbcr.len(),
                ss2_ycbcr,
                bfly_ycbcr,
                t_dqt_iter3,
            );

            eprintln!("  Q{} done", q);
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
        eprintln!("  dqt       - Per-image DQT seeding from spectral analysis");
        std::process::exit(1);
    }

    let mode = &args[1];
    let paths: Vec<String> = args[2..].to_vec();

    match mode.as_str() {
        "sweep" => run_sweep(&paths),
        "bench" => run_benchmark(&paths),
        "prefilter" => run_prefilter(&paths),
        "dqt" => run_dqt_benchmark(&paths),
        _ => {
            eprintln!(
                "Unknown mode: {}. Use sweep, bench, prefilter, or dqt.",
                mode
            );
            std::process::exit(1);
        }
    }
}
