//! Deringing Impact Benchmark
//!
//! Tests how overshoot deringing affects quality metrics and file size.
//! Deringing is most effective on images with:
//! - White backgrounds
//! - Sharp edges (text, graphics)
//! - Saturated regions (pixels at 0 or 255)
//!
//! Usage:
//!   cargo run --release --example deringing_benchmark

use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Calculate DSSIM between two RGB images
fn calculate_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use dssim::Dssim;
    use imgref::ImgVec;
    use rgb::RGB;

    // dssim requires f32 images with values in 0.0-1.0 range
    let orig_rgb: Vec<RGB<f32>> = original
        .chunks_exact(3)
        .map(|c| {
            RGB::new(
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            )
        })
        .collect();
    let dec_rgb: Vec<RGB<f32>> = decoded
        .chunks_exact(3)
        .map(|c| {
            RGB::new(
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            )
        })
        .collect();

    let orig_img = ImgVec::new(orig_rgb, width, height);
    let dec_img = ImgVec::new(dec_rgb, width, height);

    let attr = Dssim::new();
    let orig_ref = attr
        .create_image(&orig_img)
        .expect("Failed to create orig image");
    let dec_ref = attr
        .create_image(&dec_img)
        .expect("Failed to create dec image");

    let (val, _) = attr.compare(&orig_ref, dec_ref);
    val.into()
}

#[derive(Debug)]
struct DeringResult {
    image: String,
    quality: u8,
    with_dering_size: usize,
    without_dering_size: usize,
    with_dering_dssim: f64,
    without_dering_dssim: f64,
    size_diff_pct: f64,
    dssim_diff: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = PathBuf::from("comparison_outputs");
    fs::create_dir_all(&output_dir)?;

    // Quality levels to test
    let quality_levels: Vec<u8> = vec![30, 50, 70, 85, 95];

    println!("=== DERINGING IMPACT BENCHMARK ===");
    println!("Testing quality levels: {:?}\n", quality_levels);

    // Find test images
    let corpus_dir = PathBuf::from("../codec-eval/codec-corpus/kodak");
    let test_images: Vec<PathBuf> = if corpus_dir.exists() {
        let mut images: Vec<PathBuf> = fs::read_dir(&corpus_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "png"))
            .collect();
        images.sort();
        // Use first 4 images for quick benchmark
        images.into_iter().take(4).collect()
    } else {
        println!("Kodak corpus not found at {:?}", corpus_dir);
        vec![]
    };

    let mut all_results: Vec<DeringResult> = Vec::new();

    for img_path in &test_images {
        let img_name = img_path.file_stem().unwrap().to_string_lossy().to_string();
        println!("Processing: {}", img_name);

        // Load PNG
        let decoder = png::Decoder::new(fs::File::open(img_path)?);
        let mut reader = decoder.read_info()?;
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf)?;
        let bytes = &buf[..info.buffer_size()];

        let rgb_data = match info.color_type {
            png::ColorType::Rgb => bytes.to_vec(),
            png::ColorType::Rgba => bytes
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect(),
            _ => {
                println!("  Skipping unsupported color type: {:?}", info.color_type);
                continue;
            }
        };

        let width = info.width as usize;
        let height = info.height as usize;

        // Analyze image for saturated pixels (deringing is most effective here)
        let saturated_count = rgb_data.iter().filter(|&&p| p == 0 || p == 255).count();
        let saturated_pct = saturated_count as f64 / rgb_data.len() as f64 * 100.0;
        println!("  Saturated pixels: {:.1}%", saturated_pct);

        for &quality in &quality_levels {
            // Encode with deringing
            let with_dering = mozjpeg_oxide::Encoder::max_compression()
                .quality(quality)
                .subsampling(mozjpeg_oxide::Subsampling::S420)
                .overshoot_deringing(true)
                .optimize_scans(quality < 100)
                .encode_rgb(&rgb_data, width as u32, height as u32)?;

            // Encode without deringing
            let without_dering = mozjpeg_oxide::Encoder::max_compression()
                .quality(quality)
                .subsampling(mozjpeg_oxide::Subsampling::S420)
                .overshoot_deringing(false)
                .optimize_scans(quality < 100)
                .encode_rgb(&rgb_data, width as u32, height as u32)?;

            // Decode both
            let with_decoded = decode_jpeg(&with_dering)?;
            let without_decoded = decode_jpeg(&without_dering)?;

            // Calculate DSSIM
            let with_dssim = calculate_dssim(&rgb_data, &with_decoded, width, height);
            let without_dssim = calculate_dssim(&rgb_data, &without_decoded, width, height);

            let size_diff_pct = (with_dering.len() as f64 - without_dering.len() as f64)
                / without_dering.len() as f64
                * 100.0;
            let dssim_diff = with_dssim - without_dssim;

            all_results.push(DeringResult {
                image: img_name.clone(),
                quality,
                with_dering_size: with_dering.len(),
                without_dering_size: without_dering.len(),
                with_dering_dssim: with_dssim,
                without_dering_dssim: without_dssim,
                size_diff_pct,
                dssim_diff,
            });
        }
    }

    // Generate synthetic test image with white background (deringing should help here)
    println!("\nProcessing: synthetic_white_bg");
    let (width, height) = (256, 256);
    let mut pixels = vec![255u8; width * height * 3]; // White background
                                                      // Add a dark rectangle
    for y in 64..192 {
        for x in 64..192 {
            let idx = (y * width + x) * 3;
            pixels[idx] = 50;
            pixels[idx + 1] = 50;
            pixels[idx + 2] = 50;
        }
    }

    let saturated_count = pixels.iter().filter(|&&p| p == 0 || p == 255).count();
    let saturated_pct = saturated_count as f64 / pixels.len() as f64 * 100.0;
    println!("  Saturated pixels: {:.1}%", saturated_pct);

    for &quality in &quality_levels {
        let with_dering = mozjpeg_oxide::Encoder::max_compression()
            .quality(quality)
            .subsampling(mozjpeg_oxide::Subsampling::S420)
            .overshoot_deringing(true)
            .optimize_scans(quality < 100)
            .encode_rgb(&pixels, width as u32, height as u32)?;

        let without_dering = mozjpeg_oxide::Encoder::max_compression()
            .quality(quality)
            .subsampling(mozjpeg_oxide::Subsampling::S420)
            .overshoot_deringing(false)
            .optimize_scans(quality < 100)
            .encode_rgb(&pixels, width as u32, height as u32)?;

        let with_decoded = decode_jpeg(&with_dering)?;
        let without_decoded = decode_jpeg(&without_dering)?;

        let with_dssim = calculate_dssim(&pixels, &with_decoded, width, height);
        let without_dssim = calculate_dssim(&pixels, &without_decoded, width, height);

        let size_diff_pct = (with_dering.len() as f64 - without_dering.len() as f64)
            / without_dering.len() as f64
            * 100.0;
        let dssim_diff = with_dssim - without_dssim;

        all_results.push(DeringResult {
            image: "synthetic_white_bg".to_string(),
            quality,
            with_dering_size: with_dering.len(),
            without_dering_size: without_dering.len(),
            with_dering_dssim: with_dssim,
            without_dering_dssim: without_dssim,
            size_diff_pct,
            dssim_diff,
        });
    }

    // Generate synthetic test with many sharp lines (worst case for ringing)
    println!("\nProcessing: synthetic_lines");
    let (width, height) = (512, 512);
    let mut pixels = vec![255u8; width * height * 3]; // White background

    // Add many horizontal black lines (1px wide, spaced 8px apart)
    for y in (0..height).step_by(8) {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            pixels[idx] = 0;
            pixels[idx + 1] = 0;
            pixels[idx + 2] = 0;
        }
    }
    // Add many vertical black lines
    for y in 0..height {
        for x in (0..width).step_by(8) {
            let idx = (y * width + x) * 3;
            pixels[idx] = 0;
            pixels[idx + 1] = 0;
            pixels[idx + 2] = 0;
        }
    }

    let saturated_count = pixels.iter().filter(|&&p| p == 0 || p == 255).count();
    let saturated_pct = saturated_count as f64 / pixels.len() as f64 * 100.0;
    println!("  Saturated pixels: {:.1}%", saturated_pct);

    // Test at low quality levels for lines (Q10 can cause decoder issues)
    let low_quality_levels: Vec<u8> = vec![15, 20, 25, 30, 40];
    println!("  Testing at low quality: {:?}", low_quality_levels);

    for &quality in &low_quality_levels {
        // Use baseline encoding (no progressive) for synthetic tests to avoid decoder issues
        let with_dering = mozjpeg_oxide::Encoder::max_compression()
            .quality(quality)
            .subsampling(mozjpeg_oxide::Subsampling::S420)
            .overshoot_deringing(true)
            .optimize_scans(false)  // Baseline for compatibility
            .encode_rgb(&pixels, width as u32, height as u32)?;

        let without_dering = mozjpeg_oxide::Encoder::max_compression()
            .quality(quality)
            .subsampling(mozjpeg_oxide::Subsampling::S420)
            .overshoot_deringing(false)
            .optimize_scans(false)  // Baseline for compatibility
            .encode_rgb(&pixels, width as u32, height as u32)?;

        let with_decoded = decode_jpeg(&with_dering)?;
        let without_decoded = decode_jpeg(&without_dering)?;

        let with_dssim = calculate_dssim(&pixels, &with_decoded, width, height);
        let without_dssim = calculate_dssim(&pixels, &without_decoded, width, height);

        let size_diff_pct = (with_dering.len() as f64 - without_dering.len() as f64)
            / without_dering.len() as f64
            * 100.0;
        let dssim_diff = with_dssim - without_dssim;

        all_results.push(DeringResult {
            image: "synthetic_lines".to_string(),
            quality,
            with_dering_size: with_dering.len(),
            without_dering_size: without_dering.len(),
            with_dering_dssim: with_dssim,
            without_dering_dssim: without_dssim,
            size_diff_pct,
            dssim_diff,
        });
    }

    // Generate synthetic test with text-like patterns (alternating blocks)
    println!("\nProcessing: synthetic_text");
    let (width, height) = (512, 512);
    let mut pixels = vec![255u8; width * height * 3]; // White background

    // Simulate text: small black rectangles on white
    for block_y in 0..(height / 16) {
        for block_x in 0..(width / 8) {
            // Every other block is "text"
            if (block_y + block_x) % 2 == 0 {
                for dy in 2..14 {
                    for dx in 1..7 {
                        let y = block_y * 16 + dy;
                        let x = block_x * 8 + dx;
                        if y < height && x < width {
                            let idx = (y * width + x) * 3;
                            pixels[idx] = 0;
                            pixels[idx + 1] = 0;
                            pixels[idx + 2] = 0;
                        }
                    }
                }
            }
        }
    }

    let saturated_count = pixels.iter().filter(|&&p| p == 0 || p == 255).count();
    let saturated_pct = saturated_count as f64 / pixels.len() as f64 * 100.0;
    println!("  Saturated pixels: {:.1}%", saturated_pct);

    for &quality in &low_quality_levels {
        // Use baseline encoding (no progressive) for synthetic tests to avoid decoder issues
        let with_dering = mozjpeg_oxide::Encoder::max_compression()
            .quality(quality)
            .subsampling(mozjpeg_oxide::Subsampling::S420)
            .overshoot_deringing(true)
            .optimize_scans(false)  // Baseline for compatibility
            .encode_rgb(&pixels, width as u32, height as u32)?;

        let without_dering = mozjpeg_oxide::Encoder::max_compression()
            .quality(quality)
            .subsampling(mozjpeg_oxide::Subsampling::S420)
            .overshoot_deringing(false)
            .optimize_scans(false)  // Baseline for compatibility
            .encode_rgb(&pixels, width as u32, height as u32)?;

        let with_decoded = decode_jpeg(&with_dering)?;
        let without_decoded = decode_jpeg(&without_dering)?;

        let with_dssim = calculate_dssim(&pixels, &with_decoded, width, height);
        let without_dssim = calculate_dssim(&pixels, &without_decoded, width, height);

        let size_diff_pct = (with_dering.len() as f64 - without_dering.len() as f64)
            / without_dering.len() as f64
            * 100.0;
        let dssim_diff = with_dssim - without_dssim;

        all_results.push(DeringResult {
            image: "synthetic_text".to_string(),
            quality,
            with_dering_size: with_dering.len(),
            without_dering_size: without_dering.len(),
            with_dering_dssim: with_dssim,
            without_dering_dssim: without_dssim,
            size_diff_pct,
            dssim_diff,
        });
    }

    // Print summary
    println!("\n=== DERINGING IMPACT SUMMARY ===");
    println!("Negative DSSIM diff = deringing IMPROVES quality (lower DSSIM = better)");
    println!("Positive size diff = deringing increases file size\n");

    println!(
        "{:>5} {:>12} {:>12} {:>10} {:>12} {:>12} {:>10}",
        "Q", "Size+Dering", "Size-Dering", "Size%", "DSSIM+Dering", "DSSIM-Dering", "DSSIM+/-"
    );
    println!("{}", "-".repeat(85));

    // Group by quality
    for &q in &quality_levels {
        let q_results: Vec<_> = all_results.iter().filter(|r| r.quality == q).collect();
        if q_results.is_empty() {
            continue;
        }

        let avg_size_diff: f64 =
            q_results.iter().map(|r| r.size_diff_pct).sum::<f64>() / q_results.len() as f64;
        let avg_dssim_diff: f64 =
            q_results.iter().map(|r| r.dssim_diff).sum::<f64>() / q_results.len() as f64;
        let avg_with_size: f64 = q_results
            .iter()
            .map(|r| r.with_dering_size as f64)
            .sum::<f64>()
            / q_results.len() as f64;
        let avg_without_size: f64 = q_results
            .iter()
            .map(|r| r.without_dering_size as f64)
            .sum::<f64>()
            / q_results.len() as f64;
        let avg_with_dssim: f64 =
            q_results.iter().map(|r| r.with_dering_dssim).sum::<f64>() / q_results.len() as f64;
        let avg_without_dssim: f64 = q_results
            .iter()
            .map(|r| r.without_dering_dssim)
            .sum::<f64>()
            / q_results.len() as f64;

        println!(
            "{:>5} {:>12.0} {:>12.0} {:>+9.2}% {:>12.6} {:>12.6} {:>+10.6}",
            q,
            avg_with_size,
            avg_without_size,
            avg_size_diff,
            avg_with_dssim,
            avg_without_dssim,
            avg_dssim_diff
        );
    }

    // Per-image summary for synthetic tests
    println!("\n=== SYNTHETIC WHITE BACKGROUND ===");
    let synthetic_results: Vec<_> = all_results
        .iter()
        .filter(|r| r.image == "synthetic_white_bg")
        .collect();
    for r in &synthetic_results {
        println!(
            "Q{:>2}: Size {:>+5.1}%, DSSIM {:>+.6} ({})",
            r.quality,
            r.size_diff_pct,
            r.dssim_diff,
            if r.dssim_diff < 0.0 {
                "BETTER"
            } else {
                "worse"
            }
        );
    }

    println!("\n=== SYNTHETIC LINES (grid pattern, 100% saturated) ===");
    let lines_results: Vec<_> = all_results
        .iter()
        .filter(|r| r.image == "synthetic_lines")
        .collect();
    for r in &lines_results {
        println!(
            "Q{:>2}: Size {:>+5.1}%, DSSIM {:>+.6} ({})",
            r.quality,
            r.size_diff_pct,
            r.dssim_diff,
            if r.dssim_diff < 0.0 {
                "BETTER"
            } else {
                "worse"
            }
        );
    }

    println!("\n=== SYNTHETIC TEXT (checkerboard blocks, 100% saturated) ===");
    let text_results: Vec<_> = all_results
        .iter()
        .filter(|r| r.image == "synthetic_text")
        .collect();
    for r in &text_results {
        println!(
            "Q{:>2}: Size {:>+5.1}%, DSSIM {:>+.6} ({})",
            r.quality,
            r.size_diff_pct,
            r.dssim_diff,
            if r.dssim_diff < 0.0 {
                "BETTER"
            } else {
                "worse"
            }
        );
    }

    // Write CSV
    let csv_path = output_dir.join("deringing_impact.csv");
    let mut csv = fs::File::create(&csv_path)?;
    writeln!(
        csv,
        "image,quality,with_dering_size,without_dering_size,size_diff_pct,with_dering_dssim,without_dering_dssim,dssim_diff"
    )?;
    for r in &all_results {
        writeln!(
            csv,
            "{},{},{},{},{:.4},{:.8},{:.8},{:.8}",
            r.image,
            r.quality,
            r.with_dering_size,
            r.without_dering_size,
            r.size_diff_pct,
            r.with_dering_dssim,
            r.without_dering_dssim,
            r.dssim_diff
        )?;
    }
    println!("\nWrote results to {:?}", csv_path);

    println!("\n=== INTERPRETATION ===");
    println!("Deringing is most effective when:");
    println!("  - Images have high % saturated pixels (0 or 255)");
    println!("  - Quality is low-to-medium (more aggressive quantization)");
    println!("  - Sharp edges on white backgrounds");
    println!("\nExpected behavior:");
    println!("  - File size: Usually slightly larger (+0-2%) due to overshoot values");
    println!("  - DSSIM: Should be LOWER (better) with deringing for high-saturation images");

    Ok(())
}

fn decode_jpeg(data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use jpeg_decoder::Decoder;
    let mut decoder = Decoder::new(data);
    let pixels = decoder.decode()?;
    Ok(pixels)
}
