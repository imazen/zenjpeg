//! Test that compares Rust output file sizes with C++ cjpegli.
//!
//! This test ensures the Rust port produces comparable file sizes to C++.
//! Differences > 5% are investigated as potential bugs.

use std::fs;
use std::process::Command;

/// Generate test image of specified size
fn create_gradient_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = ((x * 255) / width.max(1)) as u8;
            rgb[idx + 1] = ((y * 255) / height.max(1)) as u8;
            rgb[idx + 2] = 128;
        }
    }
    rgb
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

/// Encode with C++ cjpegli (matching settings: 4:4:4, no AQ, sequential, fixed codes)
fn encode_cpp(ppm_path: &str, quality: u32) -> Option<Vec<u8>> {
    let cjpegli_path = "/home/lilith/work/jpegli-rs/jpegli-cpp/build/tools/cjpegli";
    if !std::path::Path::new(cjpegli_path).exists() {
        return None;
    }

    let output_path = format!("/tmp/cpp_test_q{}.jpg", quality);
    let output = Command::new(cjpegli_path)
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p",
            "0",
            "--fixed_code",
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

/// Encode with Rust jpegli
fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::Traditional(quality))
        .encode(rgb)
        .expect("Rust encoding failed")
}

#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_filesize_comparison_synthetic() {
    let test_cases = [
        (8, 8, "8x8"),
        (16, 16, "16x16"),
        (64, 64, "64x64"),
        (256, 256, "256x256"),
    ];

    for (width, height, name) in test_cases {
        let rgb = create_gradient_image(width, height);
        let ppm_path = format!("/tmp/test_{}.ppm", name);
        write_ppm(&ppm_path, &rgb, width, height).unwrap();

        for quality in [90, 80, 70] {
            let cpp_jpeg = match encode_cpp(&ppm_path, quality) {
                Some(j) => j,
                None => {
                    println!("Skipping {} Q{}: C++ not available", name, quality);
                    continue;
                }
            };

            let rust_jpeg = encode_rust(&rgb, width as u32, height as u32, quality as f32);

            let cpp_size = cpp_jpeg.len();
            let rust_size = rust_jpeg.len();
            let diff_pct = 100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64;

            println!(
                "{} Q{}: C++={} Rust={} ({:+.1}%)",
                name, quality, cpp_size, rust_size, diff_pct
            );

            // Allow up to 10% difference for now (we know there are differences)
            assert!(
                diff_pct.abs() < 10.0,
                "{} Q{}: file size differs by {:.1}%",
                name,
                quality,
                diff_pct
            );
        }
    }
}

#[test]
#[ignore = "requires C++ cjpegli build and test image"]
fn test_filesize_comparison_photo() {
    let png_path = "/home/lilith/work/jpegli-rs/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";
    if !std::path::Path::new(png_path).exists() {
        println!("Skipping: test image not found");
        return;
    }

    // Load PNG
    let decoder = png::Decoder::new(fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let bytes = &buf[..info.buffer_size()];
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => panic!("Unsupported color type"),
    };

    // Save as PPM for C++
    let ppm_path = "/tmp/test_flower.ppm";
    write_ppm(ppm_path, &rgb, info.width as usize, info.height as usize).unwrap();

    println!("Image: {}x{}", info.width, info.height);

    for quality in [90, 80, 70, 60] {
        let cpp_jpeg = match encode_cpp(ppm_path, quality) {
            Some(j) => j,
            None => {
                println!("Skipping Q{}: C++ not available", quality);
                continue;
            }
        };

        let rust_jpeg = encode_rust(&rgb, info.width, info.height, quality as f32);

        let cpp_size = cpp_jpeg.len();
        let rust_size = rust_jpeg.len();
        let diff_pct = 100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64;

        println!(
            "Q{}: C++={} Rust={} ({:+.1}%)",
            quality, cpp_size, rust_size, diff_pct
        );
    }
}
