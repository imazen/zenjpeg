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

/// Test on multiple images to verify threshold consistency.
#[test]
#[ignore]
fn test_streaming_threshold_multiple_images() {
    let test_images = [
        "/home/lilith/work/codec-corpus/clic2025/validation/2c1f84548ef99faec2b4f9bf12227c83.png",
        "/home/lilith/work/codec-corpus/clic2025/validation/097cb426910ba8ce2525dd8bb7fb1777.png",
        "/home/lilith/work/codec-corpus/clic2025/validation/100a02c269c5948392f283b2aa3bb4da.png",
        "/home/lilith/work/codec-corpus/kodak/10.png",
        "/home/lilith/work/codec-corpus/kodak/23.png",
    ];

    let mut results = Vec::new();

    for img_path in &test_images {
        if !std::path::Path::new(img_path).exists() {
            eprintln!("Skipping: {} (not found)", img_path);
            continue;
        }

        // Load image
        let decoder = png::Decoder::new(std::fs::File::open(img_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let pixels = &buf[..info.buffer_size()];

        let width = info.width;
        let height = info.height;

        // Baseline
        let baseline = StreamingEncoder::new(width, height)
            .quality(zenjpeg::encode::encoder_types::Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .encode(pixels)
            .unwrap();
        let baseline_size = baseline.len();

        // Test 15% and 25% thresholds
        for pct in [15, 20, 25] {
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

            let overhead = 100.0 * (result.len() as f64 - baseline_size as f64) / baseline_size as f64;
            results.push((img_path.split('/').last().unwrap(), pct, overhead));
        }
    }

    // Print summary
    eprintln!("\nSummary:");
    eprintln!("{:>35} {:>5} {:>8}", "Image", "%", "Overhead");
    eprintln!("{}", "-".repeat(52));
    for (img, pct, overhead) in &results {
        eprintln!("{:>35} {:>4}% {:>7.2}%", img, pct, overhead);
    }

    // Calculate average overhead per percentage
    for pct in [15, 20, 25] {
        let avg: f64 = results.iter()
            .filter(|(_, p, _)| *p == pct)
            .map(|(_, _, o)| o)
            .sum::<f64>() / results.iter().filter(|(_, p, _)| *p == pct).count() as f64;
        eprintln!("\n{}% average overhead: {:.2}%", pct, avg);
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
