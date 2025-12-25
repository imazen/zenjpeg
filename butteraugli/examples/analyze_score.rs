//! Analyze butteraugli scores in detail.

use butteraugli_oxide::{compute_butteraugli, ButteraugliParams, ImageF};
use std::fs;
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let (width, height) = (info.width as usize, info.height as usize);

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..width * height]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn analyze_diffmap(diffmap: &ImageF) {
    let width = diffmap.width();
    let height = diffmap.height();
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut sum = 0.0f64;
    let mut nonzero_count = 0usize;

    for y in 0..height {
        for x in 0..width {
            let v = diffmap.get(x, y);
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            sum += v as f64;
            if v > 0.0001 {
                nonzero_count += 1;
            }
        }
    }

    let mean = sum / (width * height) as f64;

    println!("  Diffmap stats:");
    println!("    size: {}x{}", width, height);
    println!("    min: {:.6}", min_val);
    println!("    max: {:.6}", max_val);
    println!("    mean: {:.6}", mean);
    println!(
        "    nonzero pixels: {} ({:.1}%)",
        nonzero_count,
        100.0 * nonzero_count as f64 / (width * height) as f64
    );
}

fn main() {
    // Test 1: Synthetic images with known difference
    println!("=== Test 1: Synthetic checkerboard patterns ===");
    let width = 64;
    let height = 64;

    let mut rgb1: Vec<u8> = vec![0; width * height * 3];
    let mut rgb2: Vec<u8> = vec![0; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let v1 = if (x / 8 + y / 8) % 2 == 0 {
                200u8
            } else {
                50u8
            };
            let v2 = if (x / 8 + y / 8) % 2 == 1 {
                200u8
            } else {
                50u8
            };
            rgb1[idx] = v1;
            rgb1[idx + 1] = v1;
            rgb1[idx + 2] = v1;
            rgb2[idx] = v2;
            rgb2[idx + 1] = v2;
            rgb2[idx + 2] = v2;
        }
    }

    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params);
    println!("Inverse block checkerboard:");
    println!("  Score: {:.6}", result.score);
    if let Some(ref diffmap) = result.diffmap {
        analyze_diffmap(diffmap);
    }

    // Test 2: Uniform colors
    println!("\n=== Test 2: Black vs White uniform ===");
    let rgb_black: Vec<u8> = vec![0; width * height * 3];
    let rgb_white: Vec<u8> = vec![255; width * height * 3];

    let result = compute_butteraugli(&rgb_black, &rgb_white, width, height, &params);
    println!("Black vs White:");
    println!("  Score: {:.6}", result.score);
    if let Some(ref diffmap) = result.diffmap {
        analyze_diffmap(diffmap);
    }

    // Test 3: Slight uniform shift
    println!("\n=== Test 3: Uniform gray vs shifted gray ===");
    let rgb_gray: Vec<u8> = vec![128; width * height * 3];
    let rgb_gray2: Vec<u8> = vec![138; width * height * 3];

    let result = compute_butteraugli(&rgb_gray, &rgb_gray2, width, height, &params);
    println!("Gray 128 vs Gray 138:");
    println!("  Score: {:.6}", result.score);
    if let Some(ref diffmap) = result.diffmap {
        analyze_diffmap(diffmap);
    }

    // Test 4: Real image with JPEG compression
    println!("\n=== Test 4: Real image JPEG roundtrip ===");
    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    if path.exists() {
        let (original, w, h) = load_png(path).expect("load png");

        // Create a heavily degraded version (simulate very low quality)
        // by averaging 8x8 blocks
        let mut degraded = original.clone();
        for by in 0..(h / 8) {
            for bx in 0..(w / 8) {
                // Calculate block average
                let mut sum_r = 0u32;
                let mut sum_g = 0u32;
                let mut sum_b = 0u32;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let x = bx * 8 + dx;
                        let y = by * 8 + dy;
                        let idx = (y * w + x) * 3;
                        sum_r += original[idx] as u32;
                        sum_g += original[idx + 1] as u32;
                        sum_b += original[idx + 2] as u32;
                    }
                }
                let avg_r = (sum_r / 64) as u8;
                let avg_g = (sum_g / 64) as u8;
                let avg_b = (sum_b / 64) as u8;

                // Fill block with average
                for dy in 0..8 {
                    for dx in 0..8 {
                        let x = bx * 8 + dx;
                        let y = by * 8 + dy;
                        let idx = (y * w + x) * 3;
                        degraded[idx] = avg_r;
                        degraded[idx + 1] = avg_g;
                        degraded[idx + 2] = avg_b;
                    }
                }
            }
        }

        let result = compute_butteraugli(&original, &degraded, w, h, &params);
        println!("Original vs 8x8 block-averaged (simulating extreme blur):");
        println!("  Score: {:.6}", result.score);
        if let Some(ref diffmap) = result.diffmap {
            analyze_diffmap(diffmap);
        }

        // Also test with actual JPEG compression at very low quality
        println!("\n=== Test 5: JPEG Q10 compression ===");
        let jpeg_data = encode_jpeg(&original, w as u32, h as u32, 10);
        let decoded = decode_jpeg(&jpeg_data);

        if decoded.len() == original.len() {
            let result = compute_butteraugli(&original, &decoded, w, h, &params);
            println!("Original vs JPEG Q10:");
            println!("  JPEG size: {} bytes", jpeg_data.len());
            println!("  Score: {:.6}", result.score);
            if let Some(ref diffmap) = result.diffmap {
                analyze_diffmap(diffmap);
            }
        }
    } else {
        println!("Test image not found: {:?}", path);
    }
}

fn encode_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use std::io::Cursor;

    let mut output = Vec::new();
    let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
    comp.set_size(width as usize, height as usize);
    comp.set_quality(quality as f32);

    let mut started = comp
        .start_compress(Cursor::new(&mut output))
        .expect("start");
    let row_stride = width as usize * 3;
    for row in rgb.chunks(row_stride) {
        started.write_scanlines(row).expect("write");
    }
    started.finish().expect("finish");
    output
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(data)
        .decode()
        .unwrap_or_default()
}
