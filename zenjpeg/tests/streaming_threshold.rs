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
        // New outliers that heuristics don't help
        "/home/lilith/work/codec-corpus/clic2025/validation/d79d465ac77c36518e0f0d626bf97ec4.png",
        "/home/lilith/work/codec-corpus/clic2025/validation/5e5ce43575fa67fdc0dd37146d7f479e.png",
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

/// Test heuristics for detecting pathological frequency distributions.
#[test]
#[ignore]
fn test_streaming_heuristics() {
    use zenjpeg::encode::encoder_types::Quality;

    // Test both a "normal" image and the pathological ones
    let test_cases = [
        ("/home/lilith/work/codec-corpus/clic2025/validation/100a02c269c5948392f283b2aa3bb4da.png", "normal"),
        ("/home/lilith/work/codec-corpus/clic2025/validation/11f2b039b293758398b1a7a8afa64bb2.png", "pathological-1"),
        ("/home/lilith/work/codec-corpus/clic2025/validation/aed95e005df28e790519eefb6eb1e565.png", "pathological-2"),
        ("/home/lilith/work/codec-corpus/clic2025/validation/d79d465ac77c36518e0f0d626bf97ec4.png", "outlier-no-help"),
        ("/home/lilith/work/codec-corpus/clic2025/validation/5e5ce43575fa67fdc0dd37146d7f479e.png", "outlier-barely-over"),
    ];

    for (img_path, label) in &test_cases {
        eprintln!("\n{}", "=".repeat(70));
        eprintln!("Image: {} ({})", img_path.split('/').last().unwrap(), label);

        // Load image
        let decoder = png::Decoder::new(std::fs::File::open(img_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let pixels = &buf[..info.buffer_size()];

        let width = info.width;
        let height = info.height;
        eprintln!("Dimensions: {}x{}", width, height);

        // First, get baseline size
        let baseline = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .encode(pixels)
            .unwrap();
        let baseline_size = baseline.len();

        // Measure heuristics at different points by encoding partway and checking
        eprintln!("\nHeuristics at different row percentages:");
        eprintln!("{:>8} {:>10} {:>12}", "Rows%", "AC Cov%", "AC Entropy");
        eprintln!("{}", "-".repeat(35));

        for pct in [10, 15, 20, 25, 30, 40, 50] {
            let rows_to_process = (height as usize * pct) / 100;

            let mut encoder = StreamingEncoder::new(width, height)
                .quality(Quality::ApproxJpegli(85.0))
                .subsampling(Subsampling::S420)
                .progressive(false)
                .start()
                .unwrap();

            let row_size = width as usize * 3;
            // Push rows up to the threshold
            for y in 0..rows_to_process.min(height as usize) {
                let start = y * row_size;
                let end = start + row_size;
                encoder.push_row(&pixels[start..end]).unwrap();
            }

            // Get heuristics at this point
            let (ac_cov, ac_ent, _, _) = encoder.frequency_heuristics();
            let stable = encoder.is_distribution_stable(4.0, 30.0);

            let flag = if stable { "✓" } else { "✗" };
            eprintln!("{:>7}% {:>9.1}% {:>11.2} {}",
                pct, ac_cov, ac_ent, flag);
        }

        // Now test overhead with and without heuristic gating
        eprintln!("\nOverhead comparison (memory_limit=1MB):");
        eprintln!("{:>20} {:>10} {:>8}", "Mode", "Size", "Overhead");
        eprintln!("{}", "-".repeat(42));

        // Without heuristics
        let mut encoder_no_heur = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(1024 * 1024) // 1MB
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder_no_heur.push_row(&pixels[start..end]).unwrap();
        }
        let result_no_heur = encoder_no_heur.finish().unwrap();
        let overhead_no_heur = 100.0 * (result_no_heur.len() as f64 - baseline_size as f64) / baseline_size as f64;

        let flag = if overhead_no_heur <= 4.0 { "✓" } else { "✗" };
        eprintln!("{:>20} {:>8} KB {:>7.2}% {}", "No heuristics",
            result_no_heur.len() / 1024, overhead_no_heur, flag);

        // With heuristics (entropy=4.0, coverage=30%)
        let mut encoder_heur = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(1024 * 1024) // 1MB
            .require_stable_distribution() // entropy=4.0, coverage=30%
            .start()
            .unwrap();

        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder_heur.push_row(&pixels[start..end]).unwrap();
        }
        // Get transition info before finish() consumes the encoder
        let transition_pct = encoder_heur.transition_percent();
        let result_heur = encoder_heur.finish().unwrap();
        let overhead_heur = 100.0 * (result_heur.len() as f64 - baseline_size as f64) / baseline_size as f64;

        let flag = if overhead_heur <= 4.0 { "✓" } else { "✗" };
        let trans_str = transition_pct
            .map(|p| format!("@{:.0}%", p))
            .unwrap_or_else(|| "N/A".to_string());
        eprintln!("{:>20} {:>8} KB {:>7.2}% {} ({})",
            "With heuristics", result_heur.len() / 1024, overhead_heur, flag, trans_str);

        let improvement = overhead_no_heur - overhead_heur;
        if improvement > 0.5 {
            eprintln!("  → Heuristics improved overhead by {:.1}%", improvement);
        }
    }
}

/// Comprehensive test with heuristics across full CLIC 2025 corpus.
/// Compares overhead with and without heuristic gating.
#[test]
#[ignore]
fn test_streaming_heuristics_clic2025_comprehensive() {
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

    // Track results: (image_name, overhead_flat25, overhead_heur25, trans_flat_info, trans_heur_info)
    let mut results: Vec<(String, f64, f64, String, String)> = Vec::new();

    // Memory limit for testing (1MB is reasonable for bounded-memory streaming)
    let memory_limit = 1024 * 1024;

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

        // Baseline for this image
        let baseline = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .encode(pixels)
            .unwrap();
        let baseline_size = baseline.len();

        let row_size = width as usize * 3;

        // Flat 25% - just min_transition_percent, no entropy/coverage checks
        let mut encoder_flat = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(memory_limit)
            .min_transition_percent(25)
            .start()
            .unwrap();

        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder_flat.push_row(&pixels[start..end]).unwrap();
        }
        let trans_flat_info = encoder_flat.transition_info();
        let result_flat = encoder_flat.finish().unwrap();
        let overhead_flat = 100.0 * (result_flat.len() as f64 - baseline_size as f64) / baseline_size as f64;

        // 25% + entropy/coverage heuristics
        let mut encoder_heur = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(memory_limit)
            .min_transition_percent(25)
            .min_entropy(4.0)
            .min_coverage(30.0)
            .start()
            .unwrap();

        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder_heur.push_row(&pixels[start..end]).unwrap();
        }
        let trans_heur_info = encoder_heur.transition_info();
        let result_heur = encoder_heur.finish().unwrap();
        let overhead_heur = 100.0 * (result_heur.len() as f64 - baseline_size as f64) / baseline_size as f64;

        results.push((img_name.to_string(), overhead_flat, overhead_heur, trans_flat_info, trans_heur_info));
    }

    // Print summary
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("COMPARISON: Flat 25% vs 25% + entropy/coverage heuristics");
    eprintln!("(Shows value of heuristics at lower minimum percentage)");
    eprintln!("{}", "=".repeat(80));

    // Calculate statistics - results are (name, flat_overhead, heur_overhead, trans_flat_info, trans_heur_info)
    let flat_overheads: Vec<f64> = results.iter().map(|(_, o, _, _, _)| *o).collect();
    let heur_overheads: Vec<f64> = results.iter().map(|(_, _, o, _, _)| *o).collect();

    let mean_flat = flat_overheads.iter().sum::<f64>() / flat_overheads.len() as f64;
    let mean_heur = heur_overheads.iter().sum::<f64>() / heur_overheads.len() as f64;

    let max_flat = flat_overheads.iter().cloned().fold(f64::MIN, f64::max);
    let max_heur = heur_overheads.iter().cloned().fold(f64::MIN, f64::max);

    let mut sorted_flat = flat_overheads.clone();
    sorted_flat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_flat = sorted_flat[sorted_flat.len() / 2];

    let mut sorted_heur = heur_overheads.clone();
    sorted_heur.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_heur = sorted_heur[sorted_heur.len() / 2];

    // Count images within 4% threshold
    let within_4_flat = flat_overheads.iter().filter(|&&o| o <= 4.0).count();
    let within_4_heur = heur_overheads.iter().filter(|&&o| o <= 4.0).count();

    // Count transition reasons
    let heur_passed = results.iter().filter(|(_, _, _, _, t)| t.contains("heuristics")).count();
    let safety_valve = results.iter().filter(|(_, _, _, _, t)| t.contains("safety")).count();
    let min_pct = results.iter().filter(|(_, _, _, _, t)| t.contains("min%")).count();

    eprintln!("\n{:>25} {:>12} {:>18}", "Metric", "Flat 25%", "25% + Heuristics");
    eprintln!("{}", "-".repeat(60));
    eprintln!("{:>25} {:>11.2}% {:>17.2}%", "Mean overhead", mean_flat, mean_heur);
    eprintln!("{:>25} {:>11.2}% {:>17.2}%", "Median overhead", median_flat, median_heur);
    eprintln!("{:>25} {:>11.2}% {:>17.2}%", "Max overhead", max_flat, max_heur);
    eprintln!("{:>25} {:>8}/{:<4} {:>14}/{:<4}", "Within 4%",
        within_4_flat, test_images.len(), within_4_heur, test_images.len());

    eprintln!("\nTransition reasons (with heuristics):");
    eprintln!("  Heuristics passed at 25%: {} images", heur_passed);
    eprintln!("  Waited for safety (50%):  {} images", safety_valve);
    eprintln!("  Min% only (no heuristics): {} images", min_pct);

    // Show worst cases - results are (name, flat_overhead, heur_overhead, trans_flat_info, trans_heur_info)
    eprintln!("\nWorst 5 images (by flat 25% overhead):");
    let mut by_flat: Vec<_> = results.iter().collect();
    by_flat.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("{:>40} {:>10} {:>10} {:>10} {:>15}", "Image", "Flat 25%", "w/Heur", "Improve", "Transition");
    eprintln!("{}", "-".repeat(90));
    for (name, flat, heur, _, trans_heur_info) in by_flat.iter().take(5) {
        let improvement = flat - heur;
        let name_short: String = name.chars().take(38).collect();
        eprintln!("{:>40} {:>9.2}% {:>9.2}% {:>+9.1}% {:>15}",
            name_short, flat, heur, improvement, trans_heur_info);
    }

    // Show images where heuristics helped most
    let mut by_improvement: Vec<_> = results.iter()
        .map(|(n, flat, heur, _, trans)| (n, flat - heur, trans.as_str()))
        .collect();
    by_improvement.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\nTop 5 images where heuristics helped most:");
    eprintln!("{:>40} {:>12} {:>15}", "Image", "Improvement", "Transition");
    eprintln!("{}", "-".repeat(72));
    for (name, improvement, trans) in by_improvement.iter().take(5) {
        let name_short: String = name.chars().take(38).collect();
        eprintln!("{:>40} {:>+11.1}% {:>15}", name_short, improvement, trans);
    }

    // Final verdict
    eprintln!("\n{}", "=".repeat(80));
    if max_heur <= 4.0 {
        eprintln!("✓ 25% + heuristics: ALL {} images within 4% overhead (max: {:.2}%)",
            test_images.len(), max_heur);
    } else {
        let over_4: Vec<_> = results.iter()
            .filter(|(_, _, h, _, _)| *h > 4.0)
            .collect();
        eprintln!("✗ 25% + heuristics: {} images exceed 4% overhead", over_4.len());
        for (name, _, h, _, trans) in over_4.iter().take(5) {
            eprintln!("    {} → {:.2}% [{}]", name, h, trans);
        }
    }

    // Compare to flat 25%
    if max_flat <= 4.0 {
        eprintln!("✓ Flat 25%: ALL {} images within 4% overhead (max: {:.2}%)",
            test_images.len(), max_flat);
    } else {
        let over_4_flat: Vec<_> = results.iter()
            .filter(|(_, f, _, _, _)| *f > 4.0)
            .collect();
        eprintln!("✗ Flat 25%: {} images exceed 4% overhead", over_4_flat.len());
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
