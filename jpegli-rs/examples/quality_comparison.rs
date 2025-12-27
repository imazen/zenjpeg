//! Quality comparison between Rust and C++ jpegli

use dssim::Dssim;
use rgb::RGBA8;
use std::fs;
use std::process::Command;

fn compute_dssim(original: &[u8], distorted: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let orig_rgba: Vec<RGBA8> = original
        .chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let dist_rgba: Vec<RGBA8> = distorted
        .chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let orig = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp = attr.create_image_rgba(&dist_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&orig, comp);
    dssim.into()
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("decode")
}

/// Decode XYB JPEG using djpegli (handles ICC profile correctly)
fn decode_xyb_jpeg(jpeg_path: &str, ppm_path: &str) -> Option<Vec<u8>> {
    let output = Command::new("/home/lilith/work/jpegli/build/tools/djpegli")
        .args([jpeg_path, ppm_path])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Read PPM file
    let ppm_data = fs::read(ppm_path).ok()?;
    parse_ppm(&ppm_data)
}

fn parse_ppm(data: &[u8]) -> Option<Vec<u8>> {
    let data_str = String::from_utf8_lossy(data);
    let mut lines = data_str.lines();

    // Skip P6 header
    let magic = lines.next()?;
    if magic != "P6" {
        return None;
    }

    // Get dimensions
    let dims = lines.next()?;
    let mut parts = dims.split_whitespace();
    let _width: usize = parts.next()?.parse().ok()?;
    let _height: usize = parts.next()?.parse().ok()?;

    // Skip max value
    let _ = lines.next()?;

    // Find binary data start
    let header_len = data
        .iter()
        .enumerate()
        .filter(|(_, &b)| b == b'\n')
        .take(3)
        .last()
        .map(|(i, _)| i + 1)?;

    Some(data[header_len..].to_vec())
}

fn main() {
    let png_path =
        "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";

    // Load PNG
    let decoder = png::Decoder::new(fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let bytes = &buf[..info.buffer_size()];
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => panic!("Unsupported"),
    };

    let width = info.width as usize;
    let height = info.height as usize;

    println!("Image: {}x{}\n", info.width, info.height);

    // Low quality YCbCr comparison with DSSIM
    println!("=== YCbCr Mode: Low Quality (Q20-Q60) ===");
    println!("| Quality | C++ Size | Rust Size | Diff    | C++ DSSIM | Rust DSSIM |");
    println!("|---------|----------|-----------|---------|-----------|------------|");

    for q in [20u8, 30, 40, 50, 60] {
        let rust_jpeg = jpegli::encode::Encoder::new()
            .width(info.width)
            .height(info.height)
            .quality(jpegli::quant::Quality::Traditional(q as f32))
            .encode(&rgb)
            .unwrap();

        let rust_size = rust_jpeg.len();
        let rust_decoded = decode_jpeg(&rust_jpeg);
        let rust_dssim = compute_dssim(&rgb, &rust_decoded, width, height);

        let ppm_path = "/tmp/test.ppm";
        let jpg_path = format!("/tmp/cpp_q{}.jpg", q);

        let output =
            Command::new("/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli")
                .args([
                    "--chroma_subsampling=444",
                    "-p",
                    "0",
                    "--fixed_code",
                    ppm_path,
                    &jpg_path,
                    "-q",
                    &q.to_string(),
                ])
                .output()
                .expect("cjpegli");

        if output.status.success() {
            let cpp_data = fs::read(&jpg_path).unwrap();
            let cpp_size = cpp_data.len();
            let cpp_decoded = decode_jpeg(&cpp_data);
            let cpp_dssim = compute_dssim(&rgb, &cpp_decoded, width, height);
            let diff = (rust_size as f64 - cpp_size as f64) / cpp_size as f64 * 100.0;
            println!(
                "| Q{:<6} | {:>8} | {:>9} | {:>+6.1}% | {:>9.6} | {:>10.6} |",
                q, cpp_size, rust_size, diff, cpp_dssim, rust_dssim
            );
        }
    }

    println!();
    println!("=== YCbCr Mode: High Quality (Q70-Q95) ===");
    println!("| Quality | C++ Size | Rust Size | Diff    | C++ DSSIM | Rust DSSIM |");
    println!("|---------|----------|-----------|---------|-----------|------------|");

    for q in [70u8, 80, 90, 95] {
        let rust_jpeg = jpegli::encode::Encoder::new()
            .width(info.width)
            .height(info.height)
            .quality(jpegli::quant::Quality::Traditional(q as f32))
            .encode(&rgb)
            .unwrap();

        let rust_size = rust_jpeg.len();
        let rust_decoded = decode_jpeg(&rust_jpeg);
        let rust_dssim = compute_dssim(&rgb, &rust_decoded, width, height);

        let ppm_path = "/tmp/test.ppm";
        let jpg_path = format!("/tmp/cpp_q{}.jpg", q);

        let output =
            Command::new("/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli")
                .args([
                    "--chroma_subsampling=444",
                    "-p",
                    "0",
                    "--fixed_code",
                    ppm_path,
                    &jpg_path,
                    "-q",
                    &q.to_string(),
                ])
                .output()
                .expect("cjpegli");

        if output.status.success() {
            let cpp_data = fs::read(&jpg_path).unwrap();
            let cpp_size = cpp_data.len();
            let cpp_decoded = decode_jpeg(&cpp_data);
            let cpp_dssim = compute_dssim(&rgb, &cpp_decoded, width, height);
            let diff = (rust_size as f64 - cpp_size as f64) / cpp_size as f64 * 100.0;
            println!(
                "| Q{:<6} | {:>8} | {:>9} | {:>+6.1}% | {:>9.6} | {:>10.6} |",
                q, cpp_size, rust_size, diff, cpp_dssim, rust_dssim
            );
        }
    }

    // XYB Mode comparison using djpegli for proper decoding
    println!();
    println!("=== XYB Mode (Q30-Q95) - decoded with djpegli ===");
    println!("| Quality | C++ Size | Rust Size | Diff    | C++ DSSIM | Rust DSSIM |");
    println!("|---------|----------|-----------|---------|-----------|------------|");

    for q in [30u8, 40, 50, 60, 70, 80, 90, 95] {
        // Rust XYB encode
        let rust_result = jpegli::encode::Encoder::new()
            .width(info.width)
            .height(info.height)
            .quality(jpegli::quant::Quality::Traditional(q as f32))
            .use_xyb(true)
            .encode(&rgb);

        let ppm_path = "/tmp/test.ppm";
        let cpp_jpg_path = format!("/tmp/cpp_xyb_q{}.jpg", q);
        let rust_jpg_path = format!("/tmp/rust_xyb_q{}.jpg", q);
        let cpp_decoded_ppm = format!("/tmp/cpp_xyb_q{}_decoded.ppm", q);
        let rust_decoded_ppm = format!("/tmp/rust_xyb_q{}_decoded.ppm", q);

        // C++ XYB encode
        let output =
            Command::new("/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli")
                .args([
                    "--xyb",
                    "-p",
                    "0",
                    "--fixed_code",
                    ppm_path,
                    &cpp_jpg_path,
                    "-q",
                    &q.to_string(),
                ])
                .output()
                .expect("cjpegli");

        let cpp_result = if output.status.success() {
            fs::read(&cpp_jpg_path).ok()
        } else {
            None
        };

        match (rust_result, cpp_result) {
            (Ok(rust_jpeg), Some(cpp_data)) => {
                let rust_size = rust_jpeg.len();
                let cpp_size = cpp_data.len();
                let diff = (rust_size as f64 - cpp_size as f64) / cpp_size as f64 * 100.0;

                // Save Rust JPEG for djpegli decoding
                fs::write(&rust_jpg_path, &rust_jpeg).unwrap();

                // Decode with djpegli
                let cpp_decoded = decode_xyb_jpeg(&cpp_jpg_path, &cpp_decoded_ppm);
                let rust_decoded = decode_xyb_jpeg(&rust_jpg_path, &rust_decoded_ppm);

                match (cpp_decoded, rust_decoded) {
                    (Some(cpp_rgb), Some(rust_rgb)) => {
                        let cpp_dssim = compute_dssim(&rgb, &cpp_rgb, width, height);
                        let rust_dssim = compute_dssim(&rgb, &rust_rgb, width, height);
                        println!(
                            "| Q{:<6} | {:>8} | {:>9} | {:>+6.1}% | {:>9.6} | {:>10.6} |",
                            q, cpp_size, rust_size, diff, cpp_dssim, rust_dssim
                        );
                    }
                    _ => {
                        println!(
                            "| Q{:<6} | {:>8} | {:>9} | {:>+6.1}% | decode err | decode err |",
                            q, cpp_size, rust_size, diff
                        );
                    }
                }
            }
            (Ok(rust_jpeg), None) => {
                println!(
                    "| Q{:<6} | {:>8} | {:>9} |         | C++ fail   |            |",
                    q,
                    "N/A",
                    rust_jpeg.len()
                );
            }
            (Err(e), Some(cpp_data)) => {
                println!(
                    "| Q{:<6} | {:>8} | {:>9} |         |            | Err: {:?}   |",
                    q,
                    cpp_data.len(),
                    "ERR",
                    e
                );
            }
            (Err(e), None) => {
                println!(
                    "| Q{:<6} | {:>8} | {:>9} |         |            | Err: {:?}   |",
                    q, "N/A", "ERR", e
                );
            }
        }
    }
}
