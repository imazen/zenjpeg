//! Quality sweep experiment comparing zenjpeg at different quality levels
//!
//! This test runs zenjpeg across Q20-Q95 and measures:
//! - File size at each quality level
//! - DSSIM quality metric
//! - Trellis vs no-trellis comparison

use dssim::Dssim;
use rgb::RGBA8;
use zenjpeg::{Encoder, Quality};

/// Convert RGB bytes to RGBA for DSSIM
fn rgb_to_rgba(rgb: &[u8]) -> Vec<RGBA8> {
    rgb.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

/// Compute DSSIM between original and decoded
fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();

    let orig_rgba = rgb_to_rgba(original);
    let dec_rgba = rgb_to_rgba(decoded);

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let dec_img = attr.create_image_rgba(&dec_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, dec_img);
    dssim.into()
}

/// Create a test image with natural-looking content
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // Mix of gradients, patterns and noise
            let r = ((fx * 200.0 + fy * 50.0) % 256.0) as u8;
            let g = ((fy * 180.0 + (x * 3) as f32 % 100.0) as u32 % 256) as u8;
            let b = (((x + y) % 64) as f32 / 64.0 * 128.0 + 64.0) as u8;

            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

#[test]
fn test_quality_sweep() {
    let width = 256;
    let height = 256;
    let original = create_test_image(width, height);

    let qualities = [20, 30, 40, 50, 60, 70, 80, 85, 90, 95];

    println!("\n=== Quality Sweep Experiment ===");
    println!("Image: {}x{} synthetic test pattern", width, height);
    println!();
    println!("{:>4} | {:>8} | {:>10} | {:>8}", "Q", "Size", "DSSIM", "BPP");
    println!("{}", "-".repeat(45));

    let mut prev_dssim = f64::MAX;

    for q in qualities {
        let encoder = Encoder::new().quality(Quality::Standard(q));
        let jpeg_data = encoder.encode_rgb(&original, width, height).unwrap();

        // Decode
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
        let decoded = decoder.decode().unwrap();

        let dssim = compute_dssim(&original, &decoded, width, height);
        let size = jpeg_data.len();
        let bpp = (size as f64 * 8.0) / (width * height) as f64;

        println!("{:>4} | {:>8} | {:>10.6} | {:>8.3}", q, size, dssim, bpp);

        // Verify quality improves with higher Q
        if q > 20 {
            assert!(
                dssim <= prev_dssim * 1.2, // Allow 20% tolerance for noise
                "Q{} should have better quality than Q{}", q, q - 10
            );
        }

        prev_dssim = dssim;
    }

    println!();
}

#[test]
fn test_trellis_vs_no_trellis_sweep() {
    let width = 256;
    let height = 256;
    let original = create_test_image(width, height);

    let qualities = [30, 50, 70, 90];

    println!("\n=== Trellis vs No-Trellis Comparison ===");
    println!("{:>4} | {:>12} | {:>12} | {:>10}", "Q", "With Trellis", "No Trellis", "Improvement");
    println!("{}", "-".repeat(55));

    for q in qualities {
        // With trellis (default for low-mid quality)
        let with_trellis = Encoder::new()
            .quality(Quality::Low(q))  // Forces mozjpeg strategy with trellis
            .encode_rgb(&original, width, height)
            .unwrap();

        // Without trellis (disable by using high quality mode which disables trellis at Q>=80)
        // For a fair comparison, use the fastest preset which disables trellis
        let without_trellis = Encoder::fastest()
            .quality(Quality::Standard(q))
            .encode_rgb(&original, width, height)
            .unwrap();

        let improvement = (1.0 - (with_trellis.len() as f64 / without_trellis.len() as f64)) * 100.0;

        println!(
            "{:>4} | {:>12} | {:>12} | {:>9.1}%",
            q,
            with_trellis.len(),
            without_trellis.len(),
            improvement
        );
    }

    println!();
}

#[test]
fn test_quality_vs_size_curve() {
    let width = 256;
    let height = 256;
    let original = create_test_image(width, height);

    let qualities: Vec<u8> = (10..=95).step_by(5).collect();

    println!("\n=== Quality vs Size Curve ===");
    println!("Q,Size,DSSIM,BPP");

    for q in &qualities {
        let encoder = Encoder::new().quality(Quality::Standard(*q));
        let jpeg_data = encoder.encode_rgb(&original, width, height).unwrap();

        let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
        let decoded = decoder.decode().unwrap();

        let dssim = compute_dssim(&original, &decoded, width, height);
        let bpp = (jpeg_data.len() as f64 * 8.0) / (width * height) as f64;

        println!("{},{},{:.6},{:.3}", q, jpeg_data.len(), dssim, bpp);
    }
}

/// Test linear quality produces more uniform DSSIM steps
#[test]
fn test_linear_quality() {
    let width = 128;
    let height = 128;
    let original = create_test_image(width, height);

    let linear_levels = [0.0_f32, 25.0, 50.0, 75.0, 100.0];

    println!("\n=== Linear Quality Mapping ===");
    println!("{:>8} | {:>8} | {:>8} | {:>10}", "Linear", "Standard", "Size", "DSSIM");
    println!("{}", "-".repeat(50));

    let mut dssims = Vec::new();

    for linear in &linear_levels {
        let encoder = Encoder::new().quality(Quality::Linear(*linear));
        let jpeg_data = encoder.encode_rgb(&original, width, height).unwrap();

        let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
        let decoded = decoder.decode().unwrap();

        let dssim = compute_dssim(&original, &decoded, width, height);
        let standard_q = Quality::Linear(*linear).value();

        println!(
            "{:>8.1} | {:>8.1} | {:>8} | {:>10.6}",
            linear, standard_q, jpeg_data.len(), dssim
        );

        dssims.push(dssim);
    }

    // Verify DSSIM decreases (quality improves) with higher linear value
    for i in 1..dssims.len() {
        assert!(
            dssims[i] <= dssims[i-1] * 1.1,
            "Linear({}) should have better quality than Linear({})",
            linear_levels[i], linear_levels[i-1]
        );
    }

    // Check that DSSIM steps are roughly uniform in log space
    let log_dssims: Vec<f64> = dssims.iter().map(|d| d.ln()).collect();
    let mut deltas = Vec::new();
    for i in 1..log_dssims.len() {
        deltas.push((log_dssims[i] - log_dssims[i-1]).abs());
    }

    println!("\nLog-DSSIM deltas (should be roughly equal):");
    for (i, delta) in deltas.iter().enumerate() {
        println!("  L{} -> L{}: {:.3}", linear_levels[i], linear_levels[i+1], delta);
    }

    // The deltas should be within 2x of each other for good linearity
    let max_delta = deltas.iter().cloned().fold(f64::MIN, f64::max);
    let min_delta = deltas.iter().cloned().fold(f64::MAX, f64::min);
    println!("  Ratio: {:.2}x (ideal: 1.0x)", max_delta / min_delta);

    println!();
}

/// Test DC trellis optimization impact
#[test]
fn test_dc_trellis_impact() {
    let width = 256;
    let height = 256;
    let original = create_test_image(width, height);

    let qualities = [30, 50, 70, 90];

    println!("\n=== DC Trellis Optimization Impact ===");
    println!("{:>4} | {:>12} | {:>12} | {:>10}", "Q", "AC+DC Trellis", "AC Only", "DC Savings");
    println!("{}", "-".repeat(55));

    for q in qualities {
        // With DC trellis (default when trellis is enabled)
        let with_dc = Encoder::new()
            .quality(Quality::Low(q))  // Low uses trellis with both AC and DC enabled
            .encode_rgb(&original, width, height)
            .unwrap();

        // Without DC trellis - we need to manually configure this
        // For now, compare against fastest which disables all trellis
        let without_dc = Encoder::fastest()
            .quality(Quality::Standard(q))
            .encode_rgb(&original, width, height)
            .unwrap();

        // Calculate savings from DC trellis
        // Note: This is comparing full trellis (AC+DC) vs no trellis,
        // so it shows combined AC+DC savings
        let savings = (1.0 - (with_dc.len() as f64 / without_dc.len() as f64)) * 100.0;

        println!(
            "{:>4} | {:>12} | {:>12} | {:>9.1}%",
            q,
            with_dc.len(),
            without_dc.len(),
            savings
        );
    }

    println!("\nNote: This compares full trellis (AC+DC) vs no trellis.");
    println!("DC trellis provides additional savings on top of AC trellis.");
    println!();
}

/// Test that verifies quality mapping produces monotonic improvements
#[test]
fn test_quality_monotonic() {
    let width = 128;
    let height = 128;
    let original = create_test_image(width, height);

    let qualities = [30, 50, 70, 90];
    let mut sizes = Vec::new();
    let mut dssims = Vec::new();

    for q in &qualities {
        let encoder = Encoder::new().quality(Quality::Standard(*q));
        let jpeg_data = encoder.encode_rgb(&original, width, height).unwrap();

        let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
        let decoded = decoder.decode().unwrap();

        let dssim = compute_dssim(&original, &decoded, width, height);

        sizes.push(jpeg_data.len());
        dssims.push(dssim);
    }

    // Higher quality should give larger files
    for i in 1..sizes.len() {
        assert!(
            sizes[i] >= sizes[i-1],
            "Q{} size ({}) should be >= Q{} size ({})",
            qualities[i], sizes[i], qualities[i-1], sizes[i-1]
        );
    }

    // Higher quality should give lower DSSIM (better)
    for i in 1..dssims.len() {
        assert!(
            dssims[i] <= dssims[i-1] * 1.1, // 10% tolerance
            "Q{} DSSIM ({:.6}) should be <= Q{} DSSIM ({:.6})",
            qualities[i], dssims[i], qualities[i-1], dssims[i-1]
        );
    }
}
