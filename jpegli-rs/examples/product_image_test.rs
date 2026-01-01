//! Test jpegli behavior on product-style images with white backgrounds
//!
//! Compares quality and size for different encoder strategies on
//! images with large uniform (white) regions.

use butteraugli::{compute_butteraugli, ButteraugliParams};
use dssim::Dssim;
use jpegli::encode::detect_uniform_block;

use std::fs;
use std::path::PathBuf;

fn main() {
    println!("=== Product Image Analysis ===\n");

    // Quality levels to test (sorted high to low for display)
    let quality_levels = [96, 91, 85, 80, 73, 60, 52, 34, 20, 15, 10, 5];

    // Create output directory for comparisons
    let output_dir = PathBuf::from("product_comparison_outputs");
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    println!("Saving comparison outputs to: {}\n", output_dir.display());
    println!("Testing quality levels: {:?}\n", quality_levels);

    // Try to find real product images, otherwise generate synthetic ones
    let corpus_dir = PathBuf::from("/home/lilith/work/codec-eval/corpus/sharpened-800px");

    if corpus_dir.exists() {
        analyze_real_images(&corpus_dir, &output_dir, &quality_levels);
    }

    // Always run synthetic tests
    analyze_synthetic_products(&output_dir, &quality_levels);
}

fn analyze_real_images(corpus_dir: &PathBuf, output_dir: &PathBuf, quality_levels: &[u8]) {
    println!("--- Real Image Analysis ---\n");

    let attr = Dssim::new();

    let mut files: Vec<PathBuf> = fs::read_dir(corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "png") &&
            p.file_name().unwrap().to_string_lossy().contains("product")
        })
        .take(10)
        .collect();

    if files.is_empty() {
        // Fall back to any images
        files = fs::read_dir(corpus_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|e| e == "png") &&
                !p.file_name().unwrap().to_string_lossy().starts_with("bpp_")
            })
            .take(5)
            .collect();
    }

    for file in &files {
        let filename = file.file_name().unwrap().to_string_lossy();
        let base_name = filename.trim_end_matches(".png");

        let Ok(f) = fs::File::open(file) else { continue };
        let decoder = png::Decoder::new(f);
        let Ok(mut reader) = decoder.read_info() else { continue };
        let mut buf = vec![0; reader.output_buffer_size()];
        let Ok(info) = reader.next_frame(&mut buf) else { continue };

        if info.color_type != png::ColorType::Rgb { continue }

        let pixels = &buf[..info.buffer_size()];
        let width = info.width as usize;
        let height = info.height as usize;

        // Copy original PNG to output directory for easy comparison
        let orig_dest = output_dir.join(format!("{}_original.png", base_name));
        let _ = fs::copy(file, &orig_dest);

        // Create reference for quality measurement
        let orig_rgba: Vec<rgb::RGBA<u8>> = pixels
            .chunks(3)
            .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
            .collect();
        let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

        println!("\n{}", filename);
        println!("{:>3} {:>8} {:>8} {:>7} {:>9} {:>9} {:>7} {:>7} {:>8}",
            "Q", "jpegli", "mozjpeg", "j_win%", "j_dssim", "m_dssim", "j_ba", "m_ba", "ba_win");
        println!("{}", "-".repeat(78));

        for &quality in quality_levels {
            let total_pixels = (width * height) as f64;

            // Encode with jpegli
            let jpegli_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_quality(quality as f32))
                .encode(pixels)
                .unwrap();

            // Calculate bpp for filename
            let j_bpp = (jpegli_result.len() * 8) as f64 / total_pixels;
            let j_bpp_100 = (j_bpp * 100.0).round() as u32;

            // Save jpegli JPEG with bpp prefix
            fs::write(output_dir.join(format!("{}_{:04}_jpegli_q{}.jpg", base_name, j_bpp_100, quality)), &jpegli_result)
                .expect("Failed to write jpegli output");

            // Encode with mozjpeg
            let mozjpeg_result = encode_mozjpeg(pixels, width, height, quality);

            // Calculate bpp for filename
            let m_bpp = (mozjpeg_result.len() * 8) as f64 / total_pixels;
            let m_bpp_100 = (m_bpp * 100.0).round() as u32;

            // Save mozjpeg JPEG with bpp prefix
            fs::write(output_dir.join(format!("{}_{:04}_mozjpeg_q{}.jpg", base_name, m_bpp_100, quality)), &mozjpeg_result)
                .expect("Failed to write mozjpeg output");

            // Measure DSSIM
            let j_dssim = compute_dssim(&attr, &orig_img, &jpegli_result, width, height);
            let m_dssim = compute_dssim(&attr, &orig_img, &mozjpeg_result, width, height);

            // Measure Butteraugli
            let j_ba = compute_butter(pixels, &jpegli_result, width, height);
            let m_ba = compute_butter(pixels, &mozjpeg_result, width, height);

            let size_diff = ((jpegli_result.len() as f64 / mozjpeg_result.len() as f64) - 1.0) * 100.0;

            // Butteraugli: lower is better, so negative diff means jpegli is better
            let ba_winner = if j_ba < m_ba { "jpegli" } else { "mozjpeg" };

            println!("{:>3} {:>8} {:>8} {:>+6.1}% {:>9.5} {:>9.5} {:>7.2} {:>7.2} {:>8}",
                quality, jpegli_result.len(), mozjpeg_result.len(), size_diff,
                j_dssim, m_dssim, j_ba, m_ba, ba_winner);
        }
    }
    println!();
}

fn analyze_image_blocks(pixels: &[u8], width: usize, height: usize) -> (f64, f64) {
    let blocks_h = (width + 7) / 8;
    let blocks_v = (height + 7) / 8;
    let mut uniform_count = 0usize;
    let mut white_count = 0usize;
    let mut total_blocks = 0usize;

    // Convert to Y plane for analysis
    let y_plane: Vec<f32> = pixels.chunks(3)
        .map(|rgb| 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32)
        .collect();

    for by in 0..blocks_v {
        for bx in 0..blocks_h {
            // Extract block
            let mut block = [0.0f32; 64];
            let mut is_white = true;

            for y in 0..8 {
                for x in 0..8 {
                    let px = (bx * 8 + x).min(width - 1);
                    let py = (by * 8 + y).min(height - 1);
                    let idx = py * width + px;
                    let val = y_plane[idx];
                    block[y * 8 + x] = val - 128.0; // Level shift

                    // Check if pixel is white (Y > 250)
                    if val < 250.0 {
                        is_white = false;
                    }
                }
            }

            total_blocks += 1;

            let result = detect_uniform_block(&block, 2.0); // Threshold of 2 for near-uniform
            if result.is_uniform {
                uniform_count += 1;
            }
            if is_white {
                white_count += 1;
            }
        }
    }

    let uniform_pct = 100.0 * uniform_count as f64 / total_blocks as f64;
    let white_pct = 100.0 * white_count as f64 / total_blocks as f64;
    (uniform_pct, white_pct)
}

fn analyze_synthetic_products(output_dir: &PathBuf, quality_levels: &[u8]) {
    println!("--- Synthetic Product Tests ---\n");

    // Test various product-like scenarios
    let scenarios = [
        ("Small product, 90% white bg", "small_90white", 200, 200, 0.90),
        ("Medium product, 80% white bg", "medium_80white", 400, 400, 0.80),
        ("Large product, 50% white bg", "large_50white", 600, 600, 0.50),
        ("Full frame product, 10% white bg", "full_10white", 800, 600, 0.10),
    ];

    let attr = Dssim::new();

    for (name, filename, width, height, white_fraction) in scenarios {
        let pixels = create_product_image(width, height, white_fraction);

        // Save original PNG
        save_png(&pixels, width, height, &output_dir.join(format!("{}_original.png", filename)));

        // Create reference for quality measurement
        let orig_rgba: Vec<rgb::RGBA<u8>> = pixels
            .chunks(3)
            .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
            .collect();
        let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

        println!("\n{}", name);
        println!("{:>3} {:>8} {:>8} {:>7} {:>9} {:>9} {:>7} {:>7} {:>8}",
            "Q", "jpegli", "mozjpeg", "j_win%", "j_dssim", "m_dssim", "j_ba", "m_ba", "ba_win");
        println!("{}", "-".repeat(78));

        for &quality in quality_levels {
            let total_pixels = (width * height) as f64;

            // Encode with jpegli
            let jpegli_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_quality(quality as f32))
                .encode(&pixels)
                .unwrap();

            // Calculate bpp for filename
            let j_bpp = (jpegli_result.len() * 8) as f64 / total_pixels;
            let j_bpp_100 = (j_bpp * 100.0).round() as u32;

            // Save jpegli JPEG with bpp prefix
            fs::write(output_dir.join(format!("{}_{:04}_jpegli_q{}.jpg", filename, j_bpp_100, quality)), &jpegli_result)
                .expect("Failed to write jpegli output");

            // Encode with mozjpeg
            let mozjpeg_result = encode_mozjpeg(&pixels, width, height, quality);

            // Calculate bpp for filename
            let m_bpp = (mozjpeg_result.len() * 8) as f64 / total_pixels;
            let m_bpp_100 = (m_bpp * 100.0).round() as u32;

            // Save mozjpeg JPEG with bpp prefix
            fs::write(output_dir.join(format!("{}_{:04}_mozjpeg_q{}.jpg", filename, m_bpp_100, quality)), &mozjpeg_result)
                .expect("Failed to write mozjpeg output");

            // Measure DSSIM
            let j_dssim = compute_dssim(&attr, &orig_img, &jpegli_result, width, height);
            let m_dssim = compute_dssim(&attr, &orig_img, &mozjpeg_result, width, height);

            // Measure Butteraugli
            let j_ba = compute_butter(&pixels, &jpegli_result, width, height);
            let m_ba = compute_butter(&pixels, &mozjpeg_result, width, height);

            let size_diff = ((jpegli_result.len() as f64 / mozjpeg_result.len() as f64) - 1.0) * 100.0;

            // Butteraugli: lower is better, so negative diff means jpegli is better
            let ba_winner = if j_ba < m_ba { "jpegli" } else { "mozjpeg" };

            println!("{:>3} {:>8} {:>8} {:>+6.1}% {:>9.5} {:>9.5} {:>7.2} {:>7.2} {:>8}",
                quality, jpegli_result.len(), mozjpeg_result.len(), size_diff,
                j_dssim, m_dssim, j_ba, m_ba, ba_winner);
        }
    }

    println!("\n=== Key Insight ===\n");
    println!("For images with large white backgrounds:");
    println!("  - mozjpeg is often SMALLER (better at uniform regions)");
    println!("  - jpegli preserves EDGES better (AQ allocates bits to product edges)");
    println!("\nUniform block detection could help jpegli:");
    println!("  - Skip DCT for uniform blocks (minor speedup)");
    println!("  - Use zero AQ for uniform blocks (no adaptation needed)");
    println!("  - Improve Huffman coding (guaranteed zero AC coefficients)");
}

fn create_product_image(width: usize, height: usize, white_fraction: f64) -> Vec<u8> {
    let mut pixels = vec![255u8; width * height * 3]; // Start all white

    // Calculate product region (centered)
    let product_area = (1.0 - white_fraction) * (width * height) as f64;
    let product_size = (product_area.sqrt()) as usize;

    // Clamp product size to fit within image bounds
    let product_width = product_size.min(width);
    let product_height = product_size.min(height);

    let x_start = (width - product_width) / 2;
    let y_start = (height - product_height) / 2;

    // Create a colorful "product" with texture
    for y in y_start..(y_start + product_height) {
        for x in x_start..(x_start + product_width) {
            let idx = (y * width + x) * 3;

            // Create a gradient with some texture
            let r = (128 + ((x - x_start) * 127 / product_width.max(1))) as u8;
            let g = (64 + ((y - y_start) * 127 / product_height.max(1))) as u8;
            let b = (180 - ((x + y) % 80)) as u8;

            // Add some noise/texture
            let noise = ((x * 17 + y * 31) % 20) as u8;

            pixels[idx] = r.saturating_add(noise / 2);
            pixels[idx + 1] = g.saturating_add(noise / 3);
            pixels[idx + 2] = b.saturating_sub(noise / 4);
        }
    }

    pixels
}

fn compute_dssim(
    attr: &Dssim,
    orig: &dssim::DssimImage<f32>,
    jpeg_data: &[u8],
    width: usize,
    height: usize,
) -> f64 {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let decoded = decoder.decode().expect("decode");

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr.create_image_rgba(&decoded_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(orig, decoded_img);
    dssim.into()
}

fn encode_mozjpeg(pixels: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
    std::panic::catch_unwind(|| {
        use mozjpeg::{ColorSpace, Compress};

        let mut comp = Compress::new(ColorSpace::JCS_RGB);
        comp.set_size(width, height);
        comp.set_quality(quality as f32);
        comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1)); // 4:4:4

        let mut started = comp.start_compress(Vec::new()).expect("start");

        let row_stride = width * 3;
        for y in 0..height {
            let row_start = y * row_stride;
            let row = &pixels[row_start..row_start + row_stride];
            let _ = started.write_scanlines(row);
        }

        started.finish().expect("finish")
    })
    .unwrap_or_default()
}

fn compute_butter(orig: &[u8], jpeg_data: &[u8], width: usize, height: usize) -> f64 {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let decoded = decoder.decode().expect("decode");

    let params = ButteraugliParams::default();
    compute_butteraugli(orig, &decoded, width, height, &params)
        .map(|r| r.score)
        .unwrap_or(999.0)
}

fn save_png(pixels: &[u8], width: usize, height: usize, path: &PathBuf) {
    use std::io::BufWriter;
    let file = fs::File::create(path).expect("Failed to create PNG file");
    let writer = BufWriter::new(file);

    let mut encoder = png::Encoder::new(writer, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header().expect("Failed to write PNG header");
    writer.write_image_data(pixels).expect("Failed to write PNG data");
}
