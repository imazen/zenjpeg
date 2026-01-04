//! Comprehensive matrix: File size + Quality (SSIMULACRA2) comparison
//!
//! Tests all combinations and measures both efficiency and quality

use jpegli::{Encoder, PixelFormat};
use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use std::fs;
use std::process::Command;

fn create_gradient(width: usize, height: usize) -> Vec<u8> {
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

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

fn compute_ssim2(orig: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let orig_f32: Vec<[f32; 3]> = orig
        .chunks(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();

    let dec_f32: Vec<[f32; 3]> = decoded
        .chunks(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();

    let orig_rgb = Rgb::new(
        orig_f32,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let dec_rgb = Rgb::new(
        dec_f32,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_rgb, dec_rgb).unwrap_or(0.0)
}

fn decode_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    jpeg_decoder::Decoder::new(data).decode().ok()
}

struct Result {
    size: usize,
    ssim2: f64,
}

fn test_rust(
    rgb: &[u8],
    width: usize,
    height: usize,
    progressive: bool,
    use_xyb: bool,
    optimize: bool,
    quality: u8,
) -> Option<Result> {
    let jpeg = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
        .mode(if progressive {
            jpegli::types::JpegMode::Progressive
        } else {
            jpegli::types::JpegMode::Baseline
        })
        .use_xyb(use_xyb)
        .optimize_huffman(optimize)
        .encode(rgb)
        .ok()?;

    let decoded = decode_jpeg(&jpeg)?;
    let ssim2 = compute_ssim2(rgb, &decoded, width, height);

    Some(Result {
        size: jpeg.len(),
        ssim2,
    })
}

fn test_cpp(
    ppm_path: &str,
    progressive: bool,
    use_xyb: bool,
    optimize: bool,
    quality: u8,
    orig_rgb: &[u8],
    width: usize,
    height: usize,
) -> Option<Result> {
    let cjpegli = jpegli::test_utils::find_cjpegli()?;
    let output = "/tmp/cpp_matrix_test.jpg";

    let mut args = vec![
        ppm_path.to_string(),
        output.to_string(),
        "-p".to_string(),
        if progressive { "2" } else { "0" }.to_string(),
        "-q".to_string(),
        quality.to_string(),
    ];

    if use_xyb {
        args.push("--xyb".to_string());
    }
    if !optimize {
        args.push("--fixed_code".to_string());
    }

    let status = Command::new(cjpegli).args(&args).output().ok()?;
    if !status.status.success() {
        return None;
    }

    let jpeg = fs::read(output).ok()?;
    let decoded = decode_jpeg(&jpeg)?;
    let ssim2 = compute_ssim2(orig_rgb, &decoded, width, height);

    Some(Result {
        size: jpeg.len(),
        ssim2,
    })
}

fn main() {
    println!("\n{}", "=".repeat(160));
    println!(" COMPREHENSIVE MATRIX: File Size + Quality (SSIMULACRA2)");
    println!("{}\n", "=".repeat(160));

    if jpegli::test_utils::find_cjpegli().is_none() {
        println!("ERROR: C++ cjpegli not found");
        return;
    }

    println!(
        "{:10} | {:6} | {:5} | {:8} | {:4} | {:30} | {:30} | {:20}",
        "Size", "Mode", "Color", "Huffman", "Q", "Rust", "C++", "Comparison"
    );
    println!(
        "{:10} | {:6} | {:5} | {:8} | {:4} | {:14} {:14} | {:14} {:14} | {:8} {:8}",
        "", "", "", "", "", "Size", "SSIM2", "Size", "SSIM2", "Size Δ%", "SSIM2 Δ"
    );
    println!("{}", "-".repeat(160));

    let test_configs = [
        // Small image
        (64, 64, "64x64"),
        // Medium image
        (512, 512, "512x512"),
        // Large image
        (2048, 2048, "2048x2048"),
    ];

    let qualities = [70, 90];
    let modes = [(false, "Base"), (true, "Prog")];
    let colors = [(false, "YCbCr"), (true, "XYB")];
    let huffmans = [(false, "Fixed"), (true, "Opt")];

    for (width, height, size_label) in &test_configs {
        let rgb = create_gradient(*width, *height);
        let ppm_path = format!("/tmp/test_{}x{}.ppm", width, height);
        write_ppm(&ppm_path, &rgb, *width, *height).ok();

        println!("\n--- {} ---", size_label);

        for &quality in &qualities {
            for &(progressive, mode_label) in &modes {
                for &(use_xyb, color_label) in &colors {
                    for &(optimize, huff_label) in &huffmans {
                        let rust_result = test_rust(
                            &rgb,
                            *width,
                            *height,
                            progressive,
                            use_xyb,
                            optimize,
                            quality,
                        );

                        let cpp_result = test_cpp(
                            &ppm_path,
                            progressive,
                            use_xyb,
                            optimize,
                            quality,
                            &rgb,
                            *width,
                            *height,
                        );

                        match (rust_result, cpp_result) {
                            (Some(r), Some(c)) => {
                                let size_diff =
                                    100.0 * (r.size as f64 - c.size as f64) / c.size as f64;
                                let ssim2_diff = r.ssim2 - c.ssim2;

                                // Flag issues
                                let size_flag = if size_diff.abs() > 20.0 {
                                    "⚠️ "
                                } else {
                                    ""
                                };
                                let ssim_flag = if ssim2_diff.abs() > 2.0 {
                                    "⚠️ "
                                } else {
                                    ""
                                };

                                println!(
                                    "{:10} | {:6} | {:5} | {:8} | {:4} | {:14} {:14.2} | {:14} {:14.2} | {:>+7.1}% {} | {:>+6.2} {}",
                                    size_label,
                                    mode_label,
                                    color_label,
                                    huff_label,
                                    quality,
                                    r.size,
                                    r.ssim2,
                                    c.size,
                                    c.ssim2,
                                    size_diff,
                                    size_flag,
                                    ssim2_diff,
                                    ssim_flag
                                );
                            }
                            (Some(r), None) => {
                                println!(
                                    "{:10} | {:6} | {:5} | {:8} | {:4} | {:14} {:14.2} | {:14} {:14} | {:20}",
                                    size_label,
                                    mode_label,
                                    color_label,
                                    huff_label,
                                    quality,
                                    r.size,
                                    r.ssim2,
                                    "N/A",
                                    "N/A",
                                    "C++ FAILED"
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    println!("\n{}", "=".repeat(160));
    println!("KEY FINDINGS:");
    println!("1. ⚠️  = Size differs by >20% or SSIM2 differs by >2.0 (investigate!)");
    println!("2. Huffman optimization validation: Compare 'Fixed' vs 'Opt' rows");
    println!("3. Progressive overhead: Compare 'Base' vs 'Prog' at different sizes");
    println!("4. Quality parity: SSIM2 should be nearly identical between Rust and C++");
    println!("{}\n", "=".repeat(160));
}
