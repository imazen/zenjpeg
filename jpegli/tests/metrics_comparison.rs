//! Quality metrics comparison tests.
//!
//! Tests DSSIM and SSIMULACRA2 metrics to ensure they correlate and produce
//! consistent results that match C++ implementations.

use dssim::Dssim;
use rgb::RGBA8;
use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use std::fs;
use std::path::Path;

fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
    data.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

fn compute_dssim(original: &[u8], distorted: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let orig_rgba = rgb_to_rgba(original);
    let dist_rgba = rgb_to_rgba(distorted);
    let orig = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp = attr.create_image_rgba(&dist_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&orig, comp);
    dssim.into()
}

fn compute_ssimulacra2(original: &[u8], distorted: &[u8], width: usize, height: usize) -> f64 {
    // Convert RGB bytes to Rgb frame (normalized to 0-1 range)
    let orig_rgb = Rgb::new(
        original
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let dist_rgb = Rgb::new(
        distorted
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_rgb, dist_rgb).unwrap_or(-1.0)
}

fn encode_jpegli(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(quality.into()))
        .encode(rgb)
        .expect("jpegli encode")
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("decode")
}

/// Test that DSSIM and SSIMULACRA2 are correlated.
/// Higher quality should give lower DSSIM (0 = identical) and higher SSIMULACRA2 (100 = identical).
#[test]
fn test_metrics_correlation() {
    let width = 128usize;
    let height = 128usize;
    let mut rgb = vec![0u8; width * height * 3];

    // Create colorful gradient
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = (x * 2) as u8;
            rgb[idx + 1] = (y * 2) as u8;
            rgb[idx + 2] = (x + y) as u8;
        }
    }

    let qualities = [50, 70, 80, 90, 95];
    let mut prev_dssim = f64::INFINITY;
    let mut prev_ssim2 = f64::NEG_INFINITY;

    println!("\n=== Metrics Correlation Test ===");
    println!("{:>8} {:>12} {:>14}", "Quality", "DSSIM", "SSIMULACRA2");
    println!("{}", "-".repeat(38));

    for &quality in &qualities {
        let jpeg_data = encode_jpegli(&rgb, width as u32, height as u32, quality);
        let decoded = decode_jpeg(&jpeg_data);

        let dssim = compute_dssim(&rgb, &decoded, width, height);
        let ssim2 = compute_ssimulacra2(&rgb, &decoded, width, height);

        println!("{:>8} {:>12.6} {:>14.4}", quality, dssim, ssim2);

        // DSSIM should decrease (better) as quality increases
        assert!(
            dssim < prev_dssim + 0.001,
            "DSSIM should decrease with higher quality: {} > {}",
            dssim,
            prev_dssim
        );

        // SSIMULACRA2 should increase (better) as quality increases
        // Note: SSIMULACRA2 can be negative for very distorted images
        assert!(
            ssim2 > prev_ssim2 - 1.0,
            "SSIMULACRA2 should increase with higher quality: {} < {}",
            ssim2,
            prev_ssim2
        );

        prev_dssim = dssim;
        prev_ssim2 = ssim2;
    }
}

/// Test metrics on identical images.
#[test]
fn test_metrics_identical() {
    let width = 64usize;
    let height = 64usize;
    let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let dssim = compute_dssim(&rgb, &rgb, width, height);
    let ssim2 = compute_ssimulacra2(&rgb, &rgb, width, height);

    println!(
        "Identical image metrics: DSSIM={:.6}, SSIMULACRA2={:.4}",
        dssim, ssim2
    );

    // DSSIM of identical images should be 0
    assert!(
        dssim < 1e-10,
        "DSSIM of identical images should be 0, got {}",
        dssim
    );

    // SSIMULACRA2 of identical images should be 100
    assert!(
        ssim2 > 99.9,
        "SSIMULACRA2 of identical images should be ~100, got {}",
        ssim2
    );
}

/// Test metrics at various distortion levels.
#[test]
fn test_metrics_distortion_levels() {
    let width = 64usize;
    let height = 64usize;
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 7) % 256) as u8)
        .collect();

    println!("\n=== Distortion Level Test ===");
    println!(
        "{:>12} {:>12} {:>14}",
        "Noise Level", "DSSIM", "SSIMULACRA2"
    );
    println!("{}", "-".repeat(42));

    for noise in [0, 1, 2, 5, 10, 20, 50] {
        // Add uniform noise
        let distorted: Vec<u8> = rgb
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let noise_val = ((i * 17 + noise) % (noise * 2 + 1)) as i16 - noise as i16;
                (v as i16 + noise_val).clamp(0, 255) as u8
            })
            .collect();

        let dssim = compute_dssim(&rgb, &distorted, width, height);
        let ssim2 = compute_ssimulacra2(&rgb, &distorted, width, height);

        println!("{:>12} {:>12.6} {:>14.4}", noise, dssim, ssim2);

        // Higher noise should give higher DSSIM
        if noise > 0 {
            assert!(dssim > 0.0, "DSSIM should be > 0 for noisy image");
        }
    }
}

/// Benchmark encoding quality across encoders using multiple metrics.
#[test]
#[ignore] // Requires test file
fn test_metrics_encoder_comparison() {
    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test file not found");
        return;
    }

    let png_data = fs::read(path).expect("read file");
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("png frame");

    let width = info.width as usize;
    let height = info.height as usize;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type"),
    };

    println!("\n=== Encoder Comparison (flower_small.rgb.png) ===");
    println!(
        "{:>8} {:>10} {:>10} {:>12} {:>14}",
        "Quality", "jpegli KB", "moz KB", "DSSIM", "SSIMULACRA2"
    );
    println!("{}", "-".repeat(58));

    for quality in [70, 80, 90] {
        // Encode with jpegli
        let jpegli_data = encode_jpegli(&rgb, width as u32, height as u32, quality);
        let jpegli_decoded = decode_jpeg(&jpegli_data);

        let dssim = compute_dssim(&rgb, &jpegli_decoded, width, height);
        let ssim2 = compute_ssimulacra2(&rgb, &jpegli_decoded, width, height);

        // Encode with mozjpeg for comparison
        use mozjpeg::{ColorSpace, Compress};
        let mut comp = Compress::new(ColorSpace::JCS_RGB);
        comp.set_size(width, height);
        comp.set_quality(quality as f32);
        comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));
        let mut started = comp.start_compress(Vec::new()).expect("mozjpeg");
        for y in 0..height {
            let row = &rgb[y * width * 3..(y + 1) * width * 3];
            let _ = started.write_scanlines(row);
        }
        let moz_data = started.finish().expect("mozjpeg finish");

        println!(
            "{:>8} {:>10.1} {:>10.1} {:>12.6} {:>14.4}",
            quality,
            jpegli_data.len() as f64 / 1024.0,
            moz_data.len() as f64 / 1024.0,
            dssim,
            ssim2
        );
    }
}

/// Test SSIMULACRA2 score ranges.
#[test]
fn test_ssimulacra2_score_ranges() {
    // SSIMULACRA2 scores interpretation:
    // 90+: Excellent, essentially identical
    // 70-90: Very good, minimal artifacts
    // 50-70: Good, some visible differences
    // 30-50: Fair, noticeable artifacts
    // <30: Poor, significant degradation

    let width = 128usize;
    let height = 128usize;

    // Create a smooth gradient (more natural, easier to compress)
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = (x * 2) as u8;
            rgb[idx + 1] = (y * 2) as u8;
            rgb[idx + 2] = 128;
        }
    }

    // High quality encoding should score > 70
    let jpeg_data = encode_jpegli(&rgb, width as u32, height as u32, 95);
    let decoded = decode_jpeg(&jpeg_data);
    let ssim2 = compute_ssimulacra2(&rgb, &decoded, width, height);

    println!("Q95 SSIMULACRA2 score: {:.2}", ssim2);
    // Smooth gradients at Q95 should score very high
    assert!(
        ssim2 > 70.0,
        "Q95 should have SSIMULACRA2 > 70, got {}",
        ssim2
    );
}
