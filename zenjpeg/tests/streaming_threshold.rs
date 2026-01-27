//! Test streaming encoder transition thresholds.
//!
//! Measures file size overhead at different transition points to find
//! the minimum data needed for near-optimal Huffman tables.
//!
//! Requires `test-utils` feature to access internal streaming API.

#![cfg(feature = "test-utils")]

use std::path::Path;
use zenjpeg::encode::streaming::StreamingEncoder;
use zenjpeg::types::Subsampling;

/// Test streaming threshold for bounded-memory encoding.
/// This test is ignored by default as it requires the codec-corpus.
#[test]
#[ignore]
fn test_streaming_transition_thresholds() {
    let img_path = std::env::var("STREAMING_TEST_IMAGE")
        .unwrap_or_else(|_| "/home/lilith/work/codec-corpus/clic2025/validation/2c1f84548ef99faec2b4f9bf12227c83.png".to_string());

    if !Path::new(&img_path).exists() {
        eprintln!("Skipping test: image not found at {}", img_path);
        return;
    }

    eprintln!("Loading: {}", img_path);

    // Load image
    let decoder = png::Decoder::new(std::fs::File::open(&img_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let pixels = &buf[..info.buffer_size()];

    let width = info.width;
    let height = info.height;
    let total_pixels = width as usize * height as usize;

    eprintln!(
        "Image: {}x{} ({:.2} MP)",
        width,
        height,
        total_pixels as f64 / 1e6
    );

    // Baseline: full optimization (no streaming transition)
    let baseline = StreamingEncoder::new(width, height)
        .quality(zenjpeg::encode::encoder_types::Quality::ApproxJpegli(85.0))
        .subsampling(Subsampling::S420)
        .progressive(false)
        .encode(pixels)
        .unwrap();

    let baseline_size = baseline.len();
    eprintln!("\nBaseline (full optimization): {} bytes", baseline_size);

    // Test different transition points (as percentage of rows)
    let percentages = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100];

    eprintln!("\n{:>6} {:>10} {:>8}", "%rows", "size", "overhead");
    eprintln!("{}", "-".repeat(30));

    let mut found_threshold = false;

    for pct in percentages {
        // Use the internal streaming API with transition_after_percent
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(zenjpeg::encode::encoder_types::Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .transition_after_percent(pct)
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder.push_row(&pixels[start..end]).unwrap();
        }
        let result = encoder.finish().unwrap();

        let size = result.len();
        let overhead = 100.0 * (size as f64 - baseline_size as f64) / baseline_size as f64;

        eprintln!(
            "{:>5}% {:>8} KB {:>7.2}%",
            pct,
            size / 1024,
            overhead
        );

        // Report when we first hit the threshold
        if !found_threshold && overhead <= 4.0 {
            found_threshold = true;
            eprintln!("\n✓ Found threshold: {}% of rows gives {:.2}% overhead", pct, overhead);
        }
    }

    if !found_threshold {
        eprintln!("\n✗ No threshold found within 4% overhead");
    }
}

/// Comprehensive test across all CLIC 2025 validation images.
/// Tests multiple quality levels and threshold percentages.
#[test]
#[ignore]
fn test_streaming_threshold_clic2025_comprehensive() {
    use zenjpeg::encode::encoder_types::Quality;

    let clic_dir = "/home/lilith/work/codec-corpus/clic2025/validation";

    // Get all PNG files in the directory
    let test_images: Vec<_> = std::fs::read_dir(clic_dir)
        .expect("Failed to read CLIC validation directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "png" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    eprintln!("Found {} images in CLIC 2025 validation set", test_images.len());

    let qualities = [70.0, 85.0, 95.0];
    let thresholds = [15, 20, 25, 50];

    // Results: (quality, threshold, overhead)
    let mut all_results: Vec<(f32, usize, f64)> = Vec::new();

    // Per-image worst cases for reporting
    let mut worst_cases: std::collections::HashMap<(i32, usize), (String, f64)> = std::collections::HashMap::new();

    for (img_idx, img_path) in test_images.iter().enumerate() {
        let img_name = img_path.file_name().unwrap().to_str().unwrap();

        // Load image
        let decoder = png::Decoder::new(std::fs::File::open(img_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let pixels = &buf[..info.buffer_size()];

        let width = info.width;
        let height = info.height;

        if img_idx % 8 == 0 {
            eprintln!("Processing image {}/{}: {} ({}x{})",
                img_idx + 1, test_images.len(), img_name, width, height);
        }

        for &quality in &qualities {
            // Baseline for this quality
            let baseline = StreamingEncoder::new(width, height)
                .quality(Quality::ApproxJpegli(quality))
                .subsampling(Subsampling::S420)
                .progressive(false)
                .encode(pixels)
                .unwrap();
            let baseline_size = baseline.len();

            for &pct in &thresholds {
                let mut encoder = StreamingEncoder::new(width, height)
                    .quality(Quality::ApproxJpegli(quality))
                    .subsampling(Subsampling::S420)
                    .progressive(false)
                    .transition_after_percent(pct)
                    .start()
                    .unwrap();

                let row_size = width as usize * 3;
                for y in 0..height as usize {
                    let start = y * row_size;
                    let end = start + row_size;
                    encoder.push_row(&pixels[start..end]).unwrap();
                }
                let result = encoder.finish().unwrap();

                let overhead = 100.0 * (result.len() as f64 - baseline_size as f64) / baseline_size as f64;
                all_results.push((quality, pct, overhead));

                // Track worst case
                let key = (quality as i32, pct);
                let entry = worst_cases.entry(key).or_insert((String::new(), f64::MIN));
                if overhead > entry.1 {
                    *entry = (img_name.to_string(), overhead);
                }
            }
        }
    }

    // Print summary statistics
    eprintln!("\n{}", "=".repeat(70));
    eprintln!("SUMMARY: {} images × {} qualities × {} thresholds",
        test_images.len(), qualities.len(), thresholds.len());
    eprintln!("{}", "=".repeat(70));

    eprintln!("\n{:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Quality", "Thresh%", "Mean", "Median", "Max", "Worst Image");
    eprintln!("{}", "-".repeat(70));

    for &quality in &qualities {
        for &pct in &thresholds {
            let mut overheads: Vec<f64> = all_results.iter()
                .filter(|(q, p, _)| *q == quality && *p == pct)
                .map(|(_, _, o)| *o)
                .collect();

            overheads.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean = overheads.iter().sum::<f64>() / overheads.len() as f64;
            let median = overheads[overheads.len() / 2];
            let max = overheads.last().unwrap();

            let (worst_img, _) = worst_cases.get(&(quality as i32, pct)).unwrap();
            let worst_img_short: String = worst_img.chars().take(12).collect();

            let flag = if *max <= 4.0 { "✓" } else { "✗" };

            eprintln!("{:>7.0} {:>7}% {:>9.2}% {:>9.2}% {:>9.2}% {} {}",
                quality, pct, mean, median, max, flag, worst_img_short);
        }
        eprintln!();
    }

    // Print recommendation
    eprintln!("\nRECOMMENDATION:");
    for &quality in &qualities {
        for &pct in &thresholds {
            let overheads: Vec<f64> = all_results.iter()
                .filter(|(q, p, _)| *q == quality && *p == pct)
                .map(|(_, _, o)| *o)
                .collect();
            let max = overheads.iter().cloned().fold(f64::MIN, f64::max);
            if max <= 4.0 {
                eprintln!("  Q{}: {}% threshold achieves <4% overhead (max: {:.2}%)",
                    quality as i32, pct, max);
                break;
            }
        }
    }
}

/// Investigate outlier images with extremely high overhead.
#[test]
#[ignore]
fn test_streaming_outlier_investigation() {
    use zenjpeg::encode::encoder_types::Quality;

    let outlier_images = [
        "/home/lilith/work/codec-corpus/clic2025/validation/11f2b039b293758398b1a7a8afa64bb2.png",
        "/home/lilith/work/codec-corpus/clic2025/validation/aed95e005df28e790519eefb6eb1e565.png",
    ];

    let thresholds = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90];

    for img_path in &outlier_images {
        let img_name = img_path.split('/').last().unwrap();
        eprintln!("\n{}", "=".repeat(60));
        eprintln!("Image: {}", img_name);

        // Load image
        let decoder = png::Decoder::new(std::fs::File::open(img_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let pixels = &buf[..info.buffer_size()];

        let width = info.width;
        let height = info.height;
        eprintln!("Dimensions: {}x{}", width, height);

        // Baseline
        let baseline = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .encode(pixels)
            .unwrap();
        let baseline_size = baseline.len();
        eprintln!("Baseline size: {} bytes", baseline_size);

        eprintln!("\n{:>8} {:>10} {:>10} {:>8}", "Thresh%", "Size", "Delta", "Overhead");
        eprintln!("{}", "-".repeat(42));

        for &pct in &thresholds {
            let mut encoder = StreamingEncoder::new(width, height)
                .quality(Quality::ApproxJpegli(85.0))
                .subsampling(Subsampling::S420)
                .progressive(false)
                .transition_after_percent(pct)
                .start()
                .unwrap();

            let row_size = width as usize * 3;
            for y in 0..height as usize {
                let start = y * row_size;
                let end = start + row_size;
                encoder.push_row(&pixels[start..end]).unwrap();
            }
            let result = encoder.finish().unwrap();

            let size = result.len();
            let delta = size as i64 - baseline_size as i64;
            let overhead = 100.0 * delta as f64 / baseline_size as f64;

            let flag = if overhead <= 4.0 { "✓" } else { "✗" };
            eprintln!("{:>7}% {:>10} {:>+10} {:>7.2}% {}",
                pct, size, delta, overhead, flag);
        }
    }
}

/// Test with memory-limit based transition (approximation of row percentage).
#[test]
#[ignore]
fn test_memory_limit_streaming_thresholds() {
    let img_path = std::env::var("STREAMING_TEST_IMAGE")
        .unwrap_or_else(|_| "/home/lilith/work/codec-corpus/clic2025/validation/2c1f84548ef99faec2b4f9bf12227c83.png".to_string());

    if !Path::new(&img_path).exists() {
        eprintln!("Skipping test: image not found at {}", img_path);
        return;
    }

    eprintln!("Loading: {}", img_path);

    // Load image
    let decoder = png::Decoder::new(std::fs::File::open(&img_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let pixels = &buf[..info.buffer_size()];

    let width = info.width;
    let height = info.height;
    let total_pixels = width as usize * height as usize;

    eprintln!(
        "Image: {}x{} ({:.2} MP)",
        width,
        height,
        total_pixels as f64 / 1e6
    );

    // Baseline: full optimization (no streaming transition)
    let baseline = StreamingEncoder::new(width, height)
        .quality(zenjpeg::encode::encoder_types::Quality::ApproxJpegli(85.0))
        .subsampling(Subsampling::S420)
        .progressive(false)
        .encode(pixels)
        .unwrap();

    let baseline_size = baseline.len();
    eprintln!("\nBaseline (full optimization): {} bytes", baseline_size);

    // Calculate approximate memory usage per MCU row
    // For 4:2:0: each MCU row is 16 pixels tall
    let blocks_per_mcu_row = (width as usize + 7) / 8 * 2  // Y blocks (2 rows per MCU)
        + (width as usize + 15) / 16 * 2;  // Cb + Cr blocks
    let bytes_per_mcu_row = blocks_per_mcu_row * 128 + width as usize * 3 * 16;

    eprintln!("Estimated bytes per MCU row: {}", bytes_per_mcu_row);

    // Test different transition points (as percentage of rows)
    let percentages = [5, 10, 15, 20, 25, 30, 40, 50, 75];

    eprintln!("\n{:>6} {:>12} {:>10} {:>8}", "%rows", "mem_limit", "size", "overhead");
    eprintln!("{}", "-".repeat(42));

    for pct in percentages {
        // Calculate memory limit to trigger transition at approximately this percentage
        let mcu_rows = (height as usize + 15) / 16;
        let target_mcu_rows = (mcu_rows * pct) / 100;
        let mem_limit = target_mcu_rows.max(1) * bytes_per_mcu_row;

        // Encode with this memory limit
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(zenjpeg::encode::encoder_types::Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(mem_limit)
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder.push_row(&pixels[start..end]).unwrap();
        }
        let result = encoder.finish().unwrap();

        let size = result.len();
        let overhead = 100.0 * (size as f64 - baseline_size as f64) / baseline_size as f64;

        eprintln!(
            "{:>5}% {:>10} KB {:>8} KB {:>7.2}%",
            pct,
            mem_limit / 1024,
            size / 1024,
            overhead
        );

        // Early exit if we've found a good threshold
        if overhead <= 4.0 {
            eprintln!("\n✓ Found threshold: {}% of rows gives {:.2}% overhead", pct, overhead);
        }
    }
}
