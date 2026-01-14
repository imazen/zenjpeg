//! Edge-specific SSIMULACRA2 comparison: Rust vs C++ jpegli
//!
//! Isolates edge handling quality by:
//! 1. Extracting partial MCU edge pixels (rightmost columns / bottom rows)
//! 2. Tiling them to create a full image of just edge content
//! 3. Encoding/decoding with both Rust and C++
//! 4. Comparing SSIMULACRA2 of the tiled edge regions
//!
//! Run with: cargo test --release -p jpegli-rs --test edge_tile_ssim2_comparison -- --nocapture --ignored

use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::path::PathBuf;
use std::process::Command;

fn find_corpus_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!(
            "{}/work/codec-eval/codec-corpus/CID22/CID22-512/training",
            home
        ),
        format!("{}/work/codec-eval/codec-corpus/kodak", home),
    ];
    for p in candidates {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn cjpegli_path() -> Option<PathBuf> {
    let candidates = [
        "internal/jpegli-cpp/build/tools/cjpegli",
        "../internal/jpegli-cpp/build/tools/cjpegli",
    ];
    for path in &candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    // Try from CARGO_MANIFEST_DIR
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(&manifest).join("../internal/jpegli-cpp/build/tools/cjpegli");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn djpegli_path() -> Option<PathBuf> {
    let candidates = [
        "internal/jpegli-cpp/build/tools/djpegli",
        "../internal/jpegli-cpp/build/tools/djpegli",
    ];
    for path in &candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(&manifest).join("../internal/jpegli-cpp/build/tools/djpegli");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Load PNG and return RGB data with dimensions
fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
            for chunk in rgba.chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

/// Crop image to specified dimensions (top-left origin)
fn crop_image(
    rgb: &[u8],
    src_width: u32,
    src_height: u32,
    new_width: u32,
    new_height: u32,
) -> Vec<u8> {
    let new_width = new_width.min(src_width) as usize;
    let new_height = new_height.min(src_height) as usize;
    let src_width = src_width as usize;

    let mut cropped = Vec::with_capacity(new_width * new_height * 3);
    for y in 0..new_height {
        let start = y * src_width * 3;
        cropped.extend_from_slice(&rgb[start..start + new_width * 3]);
    }
    cropped
}

/// Extract rightmost N columns and tile them leftward to fill width
fn tile_right_edge(rgb: &[u8], width: usize, height: usize, edge_width: usize) -> Vec<u8> {
    let mut tiled = vec![0u8; width * height * 3];

    for y in 0..height {
        // Extract edge pixels for this row
        let row_start = y * width * 3;
        let edge_start = row_start + (width - edge_width) * 3;
        let edge_pixels = &rgb[edge_start..row_start + width * 3];

        // Tile edge pixels across the full width
        let dst_row_start = y * width * 3;
        for x in 0..width {
            let src_x = x % edge_width;
            let src_idx = src_x * 3;
            let dst_idx = dst_row_start + x * 3;
            tiled[dst_idx] = edge_pixels[src_idx];
            tiled[dst_idx + 1] = edge_pixels[src_idx + 1];
            tiled[dst_idx + 2] = edge_pixels[src_idx + 2];
        }
    }
    tiled
}

/// Extract bottom N rows and tile them upward to fill height
fn tile_bottom_edge(rgb: &[u8], width: usize, height: usize, edge_height: usize) -> Vec<u8> {
    let mut tiled = vec![0u8; width * height * 3];

    // Extract edge rows
    let edge_start_row = height - edge_height;

    for y in 0..height {
        let src_y = edge_start_row + (y % edge_height);
        let src_row_start = src_y * width * 3;
        let dst_row_start = y * width * 3;
        tiled[dst_row_start..dst_row_start + width * 3]
            .copy_from_slice(&rgb[src_row_start..src_row_start + width * 3]);
    }
    tiled
}

/// Encode with Rust jpegli
fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::new()
        .quality(quality)
        .ycbcr(ChromaSubsampling::Full)
        .progressive(true)
        .optimize_huffman(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(rgb, enough::Unstoppable)
        .expect("push data");
    enc.finish().expect("finish")
}

/// Decode JPEG to RGB
fn decode_jpeg(jpeg: &[u8]) -> Vec<u8> {
    let decoded = jpegli::decoder::Decoder::new()
        .decode(jpeg)
        .expect("decode failed");
    decoded.data
}

/// Encode with C++ cjpegli, decode with djpegli, return RGB
fn encode_decode_cpp(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    cjpegli: &PathBuf,
    djpegli: &PathBuf,
) -> Option<Vec<u8>> {
    let tmp_dir = std::env::temp_dir();
    let ppm_path = tmp_dir.join("edge_test_input.ppm");
    let jpg_path = tmp_dir.join("edge_test_output.jpg");
    let out_ppm_path = tmp_dir.join("edge_test_decoded.ppm");

    // Write PPM
    let mut ppm = format!("P6\n{} {}\n255\n", width, height).into_bytes();
    ppm.extend_from_slice(rgb);
    std::fs::write(&ppm_path, &ppm).ok()?;

    // Encode with cjpegli
    let status = Command::new(cjpegli)
        .args([
            ppm_path.to_str()?,
            jpg_path.to_str()?,
            "-q",
            &quality.to_string(),
            "--chroma_subsampling",
            "444",
            "--progressive_level=2",
        ])
        .output()
        .ok()?;

    if !status.status.success() {
        return None;
    }

    // Decode with djpegli
    let status = Command::new(djpegli)
        .args([jpg_path.to_str()?, out_ppm_path.to_str()?])
        .output()
        .ok()?;

    if !status.status.success() {
        return None;
    }

    // Read decoded PPM
    let ppm_data = std::fs::read(&out_ppm_path).ok()?;

    // Parse PPM header to find RGB data start
    let mut i = 0;
    let mut newlines = 0;
    while newlines < 3 && i < ppm_data.len() {
        if ppm_data[i] == b'\n' {
            newlines += 1;
        }
        i += 1;
    }

    Some(ppm_data[i..].to_vec())
}

/// Calculate SSIMULACRA2 between two RGB images
fn calculate_ssim2(rgb1: &[u8], rgb2: &[u8], width: usize, height: usize) -> f64 {
    use fast_ssim2::{compute_frame_ssimulacra2, srgb_u8_to_linear, LinearRgbImage};

    // Convert to linear RGB
    let src: Vec<[f32; 3]> = rgb1
        .chunks(3)
        .map(|c| {
            [
                srgb_u8_to_linear(c[0]),
                srgb_u8_to_linear(c[1]),
                srgb_u8_to_linear(c[2]),
            ]
        })
        .collect();
    let dst: Vec<[f32; 3]> = rgb2
        .chunks(3)
        .map(|c| {
            [
                srgb_u8_to_linear(c[0]),
                srgb_u8_to_linear(c[1]),
                srgb_u8_to_linear(c[2]),
            ]
        })
        .collect();

    let src_img = LinearRgbImage::new(src, width, height);
    let dst_img = LinearRgbImage::new(dst, width, height);

    compute_frame_ssimulacra2(src_img, dst_img).unwrap_or(-1.0)
}

#[test]
#[ignore]
fn test_edge_tile_ssim2_comparison() {
    let corpus_path = find_corpus_path().expect("Corpus not found");
    let cjpegli = cjpegli_path().expect("cjpegli not found");
    let djpegli = djpegli_path().expect("djpegli not found");

    // Find test images
    let mut images: Vec<PathBuf> = std::fs::read_dir(&corpus_path)
        .expect("Failed to read corpus")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "png").unwrap_or(false))
        .take(3) // Use 3 images for reasonable test time
        .collect();
    images.sort();

    if images.is_empty() {
        panic!("No PNG images found in corpus");
    }

    println!("\n=== EDGE TILE SSIMULACRA2 COMPARISON ===\n");
    println!("Testing edge handling quality by tiling partial MCU pixels\n");

    let quality = 85.0;

    // Test configurations: (crop_width, crop_height, edge_width, edge_height)
    // edge_width/height is how many pixels are in the partial MCU
    let test_configs = [
        // Width edge tests (1-7 partial pixels)
        (257, 256, 1, 0), // 1 pixel partial width
        (259, 256, 3, 0), // 3 pixels partial width
        (262, 256, 6, 0), // 6 pixels partial width
        // Height edge tests (1-7 partial pixels)
        (256, 129, 0, 1), // 1 pixel partial height
        (256, 131, 0, 3), // 3 pixels partial height
        (256, 134, 0, 6), // 6 pixels partial height
        // Combined tests
        (257, 129, 1, 1), // Both edges minimal
        (262, 134, 6, 6), // Both edges maximal
    ];

    println!(
        "{:>20} {:>10} {:>8} {:>8} {:>12} {:>12} {:>10}",
        "Image", "Dims", "EdgeW", "EdgeH", "Rust SSIM2", "C++ SSIM2", "Diff"
    );
    println!("{}", "-".repeat(90));

    let mut total_rust_ssim2 = 0.0;
    let mut total_cpp_ssim2 = 0.0;
    let mut count = 0;

    for image_path in &images {
        let (rgb, orig_width, orig_height) = match load_png(image_path) {
            Some(data) => data,
            None => continue,
        };

        let image_name = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        for &(crop_w, crop_h, edge_w, edge_h) in &test_configs {
            if crop_w > orig_width || crop_h > orig_height {
                continue;
            }

            // Crop image
            let cropped = crop_image(&rgb, orig_width, orig_height, crop_w, crop_h);
            let width = crop_w as usize;
            let height = crop_h as usize;

            // Encode and decode
            let rust_jpeg = encode_rust(&cropped, crop_w, crop_h, quality);
            let rust_decoded = decode_jpeg(&rust_jpeg);

            let cpp_decoded =
                match encode_decode_cpp(&cropped, crop_w, crop_h, quality, &cjpegli, &djpegli) {
                    Some(d) => d,
                    None => {
                        println!(
                            "{:>20} {:>10} - C++ encode/decode failed",
                            image_name,
                            format!("{}x{}", crop_w, crop_h)
                        );
                        continue;
                    }
                };

            // Calculate SSIM2 based on edge type
            let (rust_ssim2, cpp_ssim2) = if edge_w > 0 && edge_h == 0 {
                // Width edge: tile rightmost columns
                let orig_tiled = tile_right_edge(&cropped, width, height, edge_w as usize);
                let rust_tiled = tile_right_edge(&rust_decoded, width, height, edge_w as usize);
                let cpp_tiled = tile_right_edge(&cpp_decoded, width, height, edge_w as usize);

                (
                    calculate_ssim2(&orig_tiled, &rust_tiled, width, height),
                    calculate_ssim2(&orig_tiled, &cpp_tiled, width, height),
                )
            } else if edge_h > 0 && edge_w == 0 {
                // Height edge: tile bottom rows
                let orig_tiled = tile_bottom_edge(&cropped, width, height, edge_h as usize);
                let rust_tiled = tile_bottom_edge(&rust_decoded, width, height, edge_h as usize);
                let cpp_tiled = tile_bottom_edge(&cpp_decoded, width, height, edge_h as usize);

                (
                    calculate_ssim2(&orig_tiled, &rust_tiled, width, height),
                    calculate_ssim2(&orig_tiled, &cpp_tiled, width, height),
                )
            } else if edge_w > 0 && edge_h > 0 {
                // Both edges: test right edge (could also test bottom, but pick one for clarity)
                let orig_tiled = tile_right_edge(&cropped, width, height, edge_w as usize);
                let rust_tiled = tile_right_edge(&rust_decoded, width, height, edge_w as usize);
                let cpp_tiled = tile_right_edge(&cpp_decoded, width, height, edge_w as usize);

                (
                    calculate_ssim2(&orig_tiled, &rust_tiled, width, height),
                    calculate_ssim2(&orig_tiled, &cpp_tiled, width, height),
                )
            } else {
                continue;
            };

            let diff = rust_ssim2 - cpp_ssim2;
            let diff_str = if diff.abs() < 0.01 {
                format!("{:+.2}", diff)
            } else if diff > 0.0 {
                format!("{:+.2} ✓", diff) // Rust better
            } else {
                format!("{:+.2} ✗", diff) // C++ better
            };

            println!(
                "{:>20} {:>10} {:>8} {:>8} {:>12.2} {:>12.2} {:>10}",
                image_name,
                format!("{}x{}", crop_w, crop_h),
                if edge_w > 0 {
                    edge_w.to_string()
                } else {
                    "-".to_string()
                },
                if edge_h > 0 {
                    edge_h.to_string()
                } else {
                    "-".to_string()
                },
                rust_ssim2,
                cpp_ssim2,
                diff_str
            );

            total_rust_ssim2 += rust_ssim2;
            total_cpp_ssim2 += cpp_ssim2;
            count += 1;
        }
    }

    println!("{}", "-".repeat(90));
    if count > 0 {
        let avg_rust = total_rust_ssim2 / count as f64;
        let avg_cpp = total_cpp_ssim2 / count as f64;
        let avg_diff = avg_rust - avg_cpp;
        println!(
            "{:>20} {:>10} {:>8} {:>8} {:>12.2} {:>12.2} {:>10}",
            "AVERAGE",
            "",
            "",
            "",
            avg_rust,
            avg_cpp,
            format!("{:+.2}", avg_diff)
        );
    }

    println!("\nNote: SSIMULACRA2 scores are for TILED EDGE PIXELS ONLY");
    println!("Higher = better quality. Positive diff = Rust handles edges better.\n");
}
