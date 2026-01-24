//! AC Refinement Scan Comparison Across Image Types
//!
//! Tests Rust vs C++ progressive encoding on various image types to identify
//! patterns in where differences occur.

use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};
use std::fs;
use std::path::Path;
use std::process::Command;

fn encode_rgb_progressive(width: u32, height: u32, data: &[u8], quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).progressive(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(data, enough::Unstoppable)
        .expect("push data");
    enc.finish().expect("finish")
}

/// Write PPM file for C++ cjpegli
fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

/// Generate solid color image
fn generate_solid(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for i in 0..(width * height) {
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    rgb
}

/// Generate horizontal gradient
fn generate_gradient_h(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let t = (x * 255 / width.max(1)) as u8;
            rgb[idx] = t;
            rgb[idx + 1] = 255 - t;
            rgb[idx + 2] = 128;
        }
    }
    rgb
}

/// Generate vertical gradient
fn generate_gradient_v(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let t = (y * 255 / height.max(1)) as u8;
            rgb[idx] = t;
            rgb[idx + 1] = t;
            rgb[idx + 2] = 255 - t;
        }
    }
    rgb
}

/// Generate diagonal gradient
fn generate_gradient_diag(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let t = ((x + y) * 255 / (width + height).max(1)) as u8;
            rgb[idx] = t;
            rgb[idx + 1] = 128;
            rgb[idx + 2] = 255 - t;
        }
    }
    rgb
}

/// Generate checkerboard pattern
fn generate_checkerboard(width: usize, height: usize, block_size: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let bx = x / block_size;
            let by = y / block_size;
            let is_white = (bx + by) % 2 == 0;
            let v = if is_white { 255 } else { 0 };
            rgb[idx] = v;
            rgb[idx + 1] = v;
            rgb[idx + 2] = v;
        }
    }
    rgb
}

/// Generate random noise
fn generate_noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    // Simple LCG for deterministic "random" values
    let mut state = seed;
    for i in 0..(width * height * 3) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        rgb[i] = (state >> 33) as u8;
    }
    rgb
}

/// Generate smooth color bands
fn generate_bands(width: usize, height: usize, num_bands: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    let band_height = height / num_bands.max(1);
    for y in 0..height {
        let band = y / band_height.max(1);
        let hue = (band * 360 / num_bands.max(1)) as f32;
        let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
    rgb
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Generate edge pattern (sharp transitions)
fn generate_edges(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Create sharp vertical and horizontal edges
            let edge_x = (x % 32) < 2;
            let edge_y = (y % 32) < 2;
            let v = if edge_x || edge_y { 255 } else { 0 };
            rgb[idx] = v;
            rgb[idx + 1] = v;
            rgb[idx + 2] = v;
        }
    }
    rgb
}

/// Generate natural-like texture (simple Perlin-ish noise)
fn generate_texture(width: usize, height: usize, scale: f32) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Simple sine-based texture
            let fx = x as f32 * scale;
            let fy = y as f32 * scale;
            let v1 = ((fx.sin() + 1.0) * 127.5) as u8;
            let v2 = ((fy.cos() + 1.0) * 127.5) as u8;
            let v3 = (((fx + fy).sin() + 1.0) * 127.5) as u8;
            rgb[idx] = v1;
            rgb[idx + 1] = v2;
            rgb[idx + 2] = v3;
        }
    }
    rgb
}

struct TestImage {
    name: &'static str,
    width: usize,
    height: usize,
    data: Vec<u8>,
}

fn get_test_images() -> Vec<TestImage> {
    let size = 128; // Use 128x128 for faster testing
    vec![
        TestImage {
            name: "solid_gray",
            width: size,
            height: size,
            data: generate_solid(size, size, 128, 128, 128),
        },
        TestImage {
            name: "solid_red",
            width: size,
            height: size,
            data: generate_solid(size, size, 255, 0, 0),
        },
        TestImage {
            name: "gradient_h",
            width: size,
            height: size,
            data: generate_gradient_h(size, size),
        },
        TestImage {
            name: "gradient_v",
            width: size,
            height: size,
            data: generate_gradient_v(size, size),
        },
        TestImage {
            name: "gradient_diag",
            width: size,
            height: size,
            data: generate_gradient_diag(size, size),
        },
        TestImage {
            name: "checkerboard_8",
            width: size,
            height: size,
            data: generate_checkerboard(size, size, 8),
        },
        TestImage {
            name: "checkerboard_16",
            width: size,
            height: size,
            data: generate_checkerboard(size, size, 16),
        },
        TestImage {
            name: "noise_1",
            width: size,
            height: size,
            data: generate_noise(size, size, 12345),
        },
        TestImage {
            name: "noise_2",
            width: size,
            height: size,
            data: generate_noise(size, size, 67890),
        },
        TestImage {
            name: "bands_5",
            width: size,
            height: size,
            data: generate_bands(size, size, 5),
        },
        TestImage {
            name: "bands_10",
            width: size,
            height: size,
            data: generate_bands(size, size, 10),
        },
        TestImage {
            name: "edges",
            width: size,
            height: size,
            data: generate_edges(size, size),
        },
        TestImage {
            name: "texture_fine",
            width: size,
            height: size,
            data: generate_texture(size, size, 0.1),
        },
        TestImage {
            name: "texture_coarse",
            width: size,
            height: size,
            data: generate_texture(size, size, 0.02),
        },
    ]
}

#[derive(Debug)]
struct ComparisonResult {
    name: String,
    cpp_size: usize,
    rust_size: usize,
    diff_pct: f64,
    rust_is_progressive: bool,
}

fn encode_cpp_progressive(ppm_path: &str, quality: u32) -> Option<Vec<u8>> {
    let cjpegli_path = zenjpeg::test_utils::find_cjpegli()?;
    let output_path = format!("/tmp/cpp_test_{}.jpg", std::process::id());

    let output = Command::new(cjpegli_path)
        .args([
            "-p",
            "2",
            ppm_path,
            &output_path,
            "-q",
            &quality.to_string(),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    fs::read(&output_path).ok()
}

#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_ac_refinement_across_image_types() {
    let images = get_test_images();
    let quality = 90;

    println!("\n=== AC Refinement Comparison Across Image Types ===\n");
    println!("Quality: {}", quality);
    println!("All images: 128x128\n");

    let mut results: Vec<ComparisonResult> = Vec::new();

    for img in &images {
        let ppm_path = format!("/tmp/test_{}.ppm", img.name);
        write_ppm(&ppm_path, &img.data, img.width, img.height).unwrap();

        // Encode with C++
        let cpp_jpeg = match encode_cpp_progressive(&ppm_path, quality) {
            Some(j) => j,
            None => {
                println!("{}: C++ encoding failed", img.name);
                continue;
            }
        };

        // Encode with Rust
        let rust_jpeg = encode_rgb_progressive(
            img.width as u32,
            img.height as u32,
            &img.data,
            quality as f32,
        );

        // Check if Rust produced progressive
        let is_progressive = rust_jpeg.windows(2).any(|w| w == [0xFF, 0xC2]);

        let cpp_size = cpp_jpeg.len();
        let rust_size = rust_jpeg.len();
        let diff_pct = 100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64;

        results.push(ComparisonResult {
            name: img.name.to_string(),
            cpp_size,
            rust_size,
            diff_pct,
            rust_is_progressive: is_progressive,
        });
    }

    // Sort by difference percentage (worst first)
    results.sort_by(|a, b| b.diff_pct.partial_cmp(&a.diff_pct).unwrap());

    // Print results table
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>8}",
        "Image", "C++ Size", "Rust Size", "Diff %", "Prog?"
    );
    println!("{:-<60}", "");

    for r in &results {
        let prog_str = if r.rust_is_progressive { "yes" } else { "NO!" };
        let diff_str = format!("{:+.1}%", r.diff_pct);
        println!(
            "{:<20} {:>10} {:>10} {:>10} {:>8}",
            r.name, r.cpp_size, r.rust_size, diff_str, prog_str
        );
    }

    // Summary statistics
    let total_cpp: usize = results.iter().map(|r| r.cpp_size).sum();
    let total_rust: usize = results.iter().map(|r| r.rust_size).sum();
    let avg_diff: f64 = results.iter().map(|r| r.diff_pct).sum::<f64>() / results.len() as f64;
    let max_diff = results
        .iter()
        .map(|r| r.diff_pct)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_diff = results
        .iter()
        .map(|r| r.diff_pct)
        .fold(f64::INFINITY, f64::min);

    println!("{:-<60}", "");
    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "TOTAL",
        total_cpp,
        total_rust,
        format!(
            "{:+.1}%",
            100.0 * (total_rust as f64 - total_cpp as f64) / total_cpp as f64
        )
    );
    println!("\nStatistics:");
    println!("  Average diff: {:+.2}%", avg_diff);
    println!("  Max diff (worst): {:+.2}%", max_diff);
    println!("  Min diff (best): {:+.2}%", min_diff);

    // Identify patterns
    println!("\n=== Pattern Analysis ===\n");

    // Group by diff severity
    let severe: Vec<_> = results.iter().filter(|r| r.diff_pct > 10.0).collect();
    let moderate: Vec<_> = results
        .iter()
        .filter(|r| r.diff_pct > 5.0 && r.diff_pct <= 10.0)
        .collect();
    let good: Vec<_> = results.iter().filter(|r| r.diff_pct <= 5.0).collect();

    if !severe.is_empty() {
        println!("SEVERE (>10% bloat):");
        for r in &severe {
            println!("  - {} ({:+.1}%)", r.name, r.diff_pct);
        }
    }

    if !moderate.is_empty() {
        println!("\nMODERATE (5-10% bloat):");
        for r in &moderate {
            println!("  - {} ({:+.1}%)", r.name, r.diff_pct);
        }
    }

    if !good.is_empty() {
        println!("\nGOOD (<=5% diff):");
        for r in &good {
            println!("  - {} ({:+.1}%)", r.name, r.diff_pct);
        }
    }
}

/// Test with real photo images if available
#[test]
#[ignore = "requires C++ cjpegli build and test images"]
fn test_ac_refinement_real_images() {
    let testdata = Path::new(env!("CARGO_MANIFEST_DIR")).join("../internal/jpegli-cpp/testdata");

    let real_images = [(
        "flower_small",
        testdata.join("jxl/flower/flower_small.rgb.png"),
    )];

    println!("\n=== AC Refinement Comparison - Real Images ===\n");

    for (name, path) in &real_images {
        if !path.exists() {
            println!("{}: Image not found at {:?}", name, path);
            continue;
        }

        // Load PNG
        let decoder = png::Decoder::new(fs::File::open(path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();

        let bytes = &buf[..info.buffer_size()];
        let rgb: Vec<u8> = match info.color_type {
            png::ColorType::Rgb => bytes.to_vec(),
            png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
            _ => {
                println!("{}: Unsupported color type", name);
                continue;
            }
        };

        let ppm_path = format!("/tmp/test_{}.ppm", name);
        write_ppm(&ppm_path, &rgb, info.width as usize, info.height as usize).unwrap();

        println!("{} ({}x{}):", name, info.width, info.height);

        for quality in [90, 80, 70, 60] {
            let cpp_jpeg = match encode_cpp_progressive(&ppm_path, quality) {
                Some(j) => j,
                None => continue,
            };

            let rust_jpeg = encode_rgb_progressive(info.width, info.height, &rgb, quality as f32);

            let diff_pct =
                100.0 * (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64;

            println!(
                "  Q{}: C++={:>6} Rust={:>6} ({:+.1}%)",
                quality,
                cpp_jpeg.len(),
                rust_jpeg.len(),
                diff_pct
            );
        }
        println!();
    }
}

/// Test different quality levels to see if bloat varies
#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_ac_refinement_quality_levels() {
    let size = 128;
    let test_images = [
        ("gradient", generate_gradient_h(size, size)),
        ("noise", generate_noise(size, size, 42)),
        ("edges", generate_edges(size, size)),
    ];

    println!("\n=== AC Refinement Bloat vs Quality Level ===\n");

    for (name, data) in &test_images {
        let ppm_path = format!("/tmp/test_ql_{}.ppm", name);
        write_ppm(&ppm_path, data, size, size).unwrap();

        println!("{}:", name);
        println!(
            "{:<8} {:>10} {:>10} {:>10}",
            "Quality", "C++ Size", "Rust Size", "Diff %"
        );
        println!("{:-<42}", "");

        for quality in [95, 90, 85, 80, 75, 70, 60, 50] {
            let cpp_jpeg = match encode_cpp_progressive(&ppm_path, quality) {
                Some(j) => j,
                None => continue,
            };

            let rust_jpeg = encode_rgb_progressive(size as u32, size as u32, data, quality as f32);

            let diff_pct =
                100.0 * (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64;

            println!(
                "{:<8} {:>10} {:>10} {:>10}",
                quality,
                cpp_jpeg.len(),
                rust_jpeg.len(),
                format!("{:+.1}%", diff_pct)
            );
        }
        println!();
    }
}
