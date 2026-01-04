//! Quick comparison of XYB vs YCbCr encoding modes.
//! Shows file size (bpp) and both Butteraugli and SSIMULACRA2 quality metrics.

use butteraugli::{compute_butteraugli, ButteraugliParams};
use jpegli::{Decoder, Encoder, PixelFormat};
use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::process::Command;

fn load_png(path: &std::path::Path) -> (Vec<u8>, usize, usize) {
    let decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let bytes = &buf[..info.buffer_size()];

    // Convert to RGB if needed
    let rgb = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes
            .chunks(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect(),
        _ => panic!("Unsupported color type"),
    };

    (rgb, info.width as usize, info.height as usize)
}

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

fn encode_with_cjpegli(
    rgb_data: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    use_xyb: bool,
) -> Option<Vec<u8>> {
    let cjpegli_path = match jpegli::test_utils::find_cjpegli() {
        Some(p) => p,
        None => return None,
    };

    let ppm_path = "/tmp/xyb_test_input.ppm";
    let jpg_path = if use_xyb {
        "/tmp/xyb_test_cpp_xyb.jpg"
    } else {
        "/tmp/xyb_test_cpp_ycbcr.jpg"
    };

    write_ppm(ppm_path, rgb_data, width, height).ok()?;

    let quality_str = quality.to_string();
    let mut args = vec![ppm_path, jpg_path, "-q", &quality_str];
    if use_xyb {
        args.push("--xyb");
    }

    let output = Command::new(&cjpegli_path).args(&args).output().ok()?;

    if !output.status.success() {
        eprintln!(
            "cjpegli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    fs::read(jpg_path).ok()
}

fn find_corpus_images() -> Vec<PathBuf> {
    // Look for codec-corpus in common locations
    let search_paths = [
        PathBuf::from("../codec-comparison/codec-corpus"),
        PathBuf::from("../codec-corpus"),
        PathBuf::from("./codec-corpus"),
    ];

    for base in &search_paths {
        if base.exists() {
            // Try kodak first (24 high-quality test images)
            let kodak_dir = base.join("kodak");
            if kodak_dir.exists() {
                let mut images: Vec<_> = std::fs::read_dir(kodak_dir)
                    .unwrap()
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "png")
                            .unwrap_or(false)
                    })
                    .map(|e| e.path())
                    .collect();
                images.sort();
                return images;
            }
        }
    }

    Vec::new()
}

fn main() {
    let corpus_images = find_corpus_images();

    let test_images = if !corpus_images.is_empty() {
        println!("Found {} images in codec-corpus/kodak", corpus_images.len());
        // Use first 3 images for quick test
        corpus_images.into_iter().take(3).collect::<Vec<_>>()
    } else {
        println!("codec-corpus not found, using synthetic test image");
        vec![]
    };

    if test_images.is_empty() {
        // Fallback to synthetic image
        let width = 256;
        let height = 256;
        let mut rgb_data = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                rgb_data[idx] = (x % 256) as u8;
                rgb_data[idx + 1] = (y % 256) as u8;
                rgb_data[idx + 2] = ((x + y) / 2 % 256) as u8;
            }
        }
        run_comparison("synthetic_gradient", rgb_data, width, height);
        return;
    }

    // Test with real images
    for image_path in test_images {
        let (rgb_data, width, height) = load_png(&image_path);
        let name = image_path.file_stem().unwrap().to_str().unwrap();
        println!("\n==========================");
        run_comparison(name, rgb_data, width, height);
    }
}

fn run_comparison(name: &str, rgb_data: Vec<u8>, width: usize, height: usize) {
    println!("=== {} ({}x{}) ===\n", name, width, height);

    // Check if C++ cjpegli is available
    let has_cpp = jpegli::test_utils::find_cjpegli().is_some();
    if has_cpp {
        println!("C++ cjpegli available - comparing all 4 modes\n");
    } else {
        println!("C++ cjpegli not found - comparing Rust only\n");
    }

    for quality in [70, 80, 90] {
        println!("--- Quality {} ---", quality);

        // Encode YCbCr
        let ycbcr_jpeg = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
            .use_xyb(false)
            .encode(&rgb_data)
            .expect("YCbCr encode failed");

        // Encode XYB
        let xyb_jpeg = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
            .use_xyb(true)
            .encode(&rgb_data)
            .expect("XYB encode failed");

        // Decode both
        let ycbcr_decoded = Decoder::new()
            .apply_icc(false)
            .decode(&ycbcr_jpeg)
            .expect("YCbCr decode failed");

        let xyb_decoded = Decoder::new()
            .apply_icc(true)
            .decode(&xyb_jpeg)
            .expect("XYB decode failed");

        // Compute bpp
        let total_pixels = (width * height) as f64;
        let ycbcr_bpp = (ycbcr_jpeg.len() as f64 * 8.0) / total_pixels;
        let xyb_bpp = (xyb_jpeg.len() as f64 * 8.0) / total_pixels;

        // Convert to SSIMULACRA2 format
        let orig_rgb = Rgb::new(
            rgb_data
                .chunks(3)
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ]
                })
                .collect(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .expect("create orig rgb");

        let ycbcr_rgb = Rgb::new(
            ycbcr_decoded
                .data
                .chunks(3)
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ]
                })
                .collect(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .expect("create ycbcr rgb");

        let xyb_rgb = Rgb::new(
            xyb_decoded
                .data
                .chunks(3)
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ]
                })
                .collect(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .expect("create xyb rgb");

        // Compute SSIMULACRA2
        let ycbcr_ssim2 =
            compute_frame_ssimulacra2(orig_rgb.clone(), ycbcr_rgb).expect("compute ycbcr ssim2");
        let xyb_ssim2 =
            compute_frame_ssimulacra2(orig_rgb.clone(), xyb_rgb).expect("compute xyb ssim2");

        // Compute Butteraugli
        let params = ButteraugliParams::default();
        let ycbcr_butteraugli =
            compute_butteraugli(&rgb_data, &ycbcr_decoded.data, width, height, &params)
                .expect("compute ycbcr butteraugli")
                .score;
        let xyb_butteraugli =
            compute_butteraugli(&rgb_data, &xyb_decoded.data, width, height, &params)
                .expect("compute xyb butteraugli")
                .score;

        println!(
            "  Rust YCbCr: {:.2} bpp | SSIM2 {:.2} | Bfly {:.3}",
            ycbcr_bpp, ycbcr_ssim2, ycbcr_butteraugli
        );
        println!(
            "  Rust XYB:   {:.2} bpp | SSIM2 {:.2} | Bfly {:.3}",
            xyb_bpp, xyb_ssim2, xyb_butteraugli
        );

        // Encode and test with C++ if available
        if has_cpp {
            if let Some(cpp_ycbcr_jpeg) =
                encode_with_cjpegli(&rgb_data, width, height, quality, false)
            {
                if let Some(cpp_xyb_jpeg) =
                    encode_with_cjpegli(&rgb_data, width, height, quality, true)
                {
                    // Decode C++ versions
                    let cpp_ycbcr_decoded = Decoder::new()
                        .apply_icc(false)
                        .decode(&cpp_ycbcr_jpeg)
                        .expect("C++ YCbCr decode failed");

                    let cpp_xyb_decoded = Decoder::new()
                        .apply_icc(true)
                        .decode(&cpp_xyb_jpeg)
                        .expect("C++ XYB decode failed");

                    // Compute C++ metrics
                    let cpp_ycbcr_bpp = (cpp_ycbcr_jpeg.len() as f64 * 8.0) / total_pixels;
                    let cpp_xyb_bpp = (cpp_xyb_jpeg.len() as f64 * 8.0) / total_pixels;

                    let cpp_ycbcr_rgb = Rgb::new(
                        cpp_ycbcr_decoded
                            .data
                            .chunks(3)
                            .map(|c| {
                                [
                                    c[0] as f32 / 255.0,
                                    c[1] as f32 / 255.0,
                                    c[2] as f32 / 255.0,
                                ]
                            })
                            .collect(),
                        width,
                        height,
                        TransferCharacteristic::SRGB,
                        ColorPrimaries::BT709,
                    )
                    .expect("create cpp ycbcr rgb");

                    let cpp_xyb_rgb = Rgb::new(
                        cpp_xyb_decoded
                            .data
                            .chunks(3)
                            .map(|c| {
                                [
                                    c[0] as f32 / 255.0,
                                    c[1] as f32 / 255.0,
                                    c[2] as f32 / 255.0,
                                ]
                            })
                            .collect(),
                        width,
                        height,
                        TransferCharacteristic::SRGB,
                        ColorPrimaries::BT709,
                    )
                    .expect("create cpp xyb rgb");

                    let cpp_ycbcr_ssim2 =
                        compute_frame_ssimulacra2(orig_rgb.clone(), cpp_ycbcr_rgb)
                            .expect("compute cpp ycbcr ssim2");
                    let cpp_xyb_ssim2 = compute_frame_ssimulacra2(orig_rgb.clone(), cpp_xyb_rgb)
                        .expect("compute cpp xyb ssim2");

                    let cpp_ycbcr_butteraugli = compute_butteraugli(
                        &rgb_data,
                        &cpp_ycbcr_decoded.data,
                        width,
                        height,
                        &params,
                    )
                    .expect("compute cpp ycbcr butteraugli")
                    .score;

                    let cpp_xyb_butteraugli = compute_butteraugli(
                        &rgb_data,
                        &cpp_xyb_decoded.data,
                        width,
                        height,
                        &params,
                    )
                    .expect("compute cpp xyb butteraugli")
                    .score;

                    println!(
                        "  C++  YCbCr: {:.2} bpp | SSIM2 {:.2} | Bfly {:.3}",
                        cpp_ycbcr_bpp, cpp_ycbcr_ssim2, cpp_ycbcr_butteraugli
                    );
                    println!(
                        "  C++  XYB:   {:.2} bpp | SSIM2 {:.2} | Bfly {:.3}",
                        cpp_xyb_bpp, cpp_xyb_ssim2, cpp_xyb_butteraugli
                    );
                }
            }
        }

        // Show Rust deltas
        let bpp_diff = xyb_bpp - ycbcr_bpp;
        let ssim2_diff = xyb_ssim2 - ycbcr_ssim2;
        let bfly_diff = xyb_butteraugli - ycbcr_butteraugli;
        println!(
            "  Rust Δ:     {:+.2} bpp ({:+.1}%) | {:+.2} SSIM2 | {:+.3} Bfly",
            bpp_diff,
            (bpp_diff / ycbcr_bpp) * 100.0,
            ssim2_diff,
            bfly_diff
        );
        println!();
    }
}
