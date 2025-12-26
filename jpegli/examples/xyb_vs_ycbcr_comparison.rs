//! XYB vs YCbCr comparison: Rust vs C++ implementation gaps.
//!
//! This example compares:
//! 1. Rust XYB vs C++ XYB (implementation gap)
//! 2. Rust YCbCr vs C++ YCbCr (implementation gap)
//! 3. XYB vs YCbCr (color space efficiency gap)
//!
//! Usage: cargo run --release --example xyb_vs_ycbcr_comparison

use dssim::Dssim;
use rgb::RGBA8;
use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;
use std::process::Command;

const CJPEGLI_PATH: &str = "/home/lilith/work/jpegli/build/tools/cjpegli";

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
        png::ColorType::GrayscaleAlpha => buf[..width * height * 2]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
    data.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let orig_rgba = rgb_to_rgba(original);
    let dec_rgba = rgb_to_rgba(decoded);
    let orig = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp = attr.create_image_rgba(&dec_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&orig, comp);
    dssim.into()
}

fn compute_ssim2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let orig_rgb = Rgb::new(
        original
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
    .unwrap();

    let dec_rgb = Rgb::new(
        decoded
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
    .unwrap();

    compute_frame_ssimulacra2(orig_rgb, dec_rgb).unwrap_or(0.0)
}

fn encode_rust_jpegli(rgb: &[u8], width: u32, height: u32, quality: u8, use_xyb: bool) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(quality.into()))
        .use_xyb(use_xyb)
        .encode(rgb)
        .expect("jpegli encode")
}

fn encode_cpp_cjpegli(ppm_path: &str, quality: u32, use_xyb: bool) -> Option<Vec<u8>> {
    if !Path::new(CJPEGLI_PATH).exists() {
        return None;
    }

    let output_path = format!(
        "/tmp/cpp_{}_{}.jpg",
        if use_xyb { "xyb" } else { "ycbcr" },
        quality
    );

    let mut args = vec![
        "--chroma_subsampling=444".to_string(),
        "-p".to_string(),
        "0".to_string(), // Sequential mode
    ];

    if use_xyb {
        args.push("--xyb".to_string());
    }

    args.push(ppm_path.to_string());
    args.push(output_path.clone());
    args.push("-q".to_string());
    args.push(quality.to_string());

    let output = Command::new(CJPEGLI_PATH).args(&args).output().ok()?;

    if !output.status.success() {
        eprintln!(
            "cjpegli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    fs::read(&output_path).ok()
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("decode")
}

/// Decode XYB JPEG with ICC profile applied
fn decode_xyb_with_icc(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
    {
        if let Ok((pixels, _, _)) = jpegli::icc::decode_jpeg_with_icc(jpeg_data) {
            return Some(pixels);
        }
    }

    // Fallback to Python/Pillow
    use std::process::Stdio;

    let jpeg_path = "/tmp/xyb_decode_temp.jpg";
    let output_path = "/tmp/xyb_decode_temp.bin";
    fs::write(jpeg_path, jpeg_data).ok()?;

    let script = r#"
import io, sys
from PIL import Image, ImageCms
img = Image.open(sys.argv[1])
if 'icc_profile' in img.info and len(img.info['icc_profile']) > 0:
    input_profile = ImageCms.ImageCmsProfile(io.BytesIO(img.info['icc_profile']))
    srgb = ImageCms.createProfile('sRGB')
    transform = ImageCms.buildTransformFromOpenProfiles(input_profile, srgb, 'RGB', 'RGB')
    img = ImageCms.applyTransform(img, transform)
with open(sys.argv[2], 'wb') as f:
    f.write(bytes(img.convert('RGB').tobytes()))
"#;

    let status = Command::new("python3")
        .args(["-c", script, jpeg_path, output_path])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?;

    let _ = fs::remove_file(jpeg_path);
    if !status.success() {
        return None;
    }

    let data = fs::read(output_path).ok()?;
    let _ = fs::remove_file(output_path);
    Some(data)
}

#[derive(Debug, Clone)]
struct EncoderResult {
    name: String,
    quality: u8,
    size: usize,
    bpp: f64,
    dssim: f64,
    ssim2: f64,
}

fn main() {
    println!("=== XYB vs YCbCr Comparison: Rust vs C++ ===\n");

    // Check if cjpegli exists
    if !Path::new(CJPEGLI_PATH).exists() {
        eprintln!("ERROR: cjpegli not found at {}", CJPEGLI_PATH);
        eprintln!("Build it with: cd /home/lilith/work/jpegli && mkdir -p build && cd build && cmake -G Ninja -DJPEGXL_ENABLE_TOOLS=ON .. && ninja cjpegli");
        return;
    }

    // Test images
    let test_images = [
        "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png",
        "/home/lilith/work/jpegli/testdata/jxl/flower/flower.png",
    ];

    // Quality levels to test
    let quality_levels = [70, 80, 85, 90, 95];

    for image_path in &test_images {
        let path = Path::new(image_path);
        if !path.exists() {
            println!("Skipping {}: file not found", image_path);
            continue;
        }

        println!(
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        println!("Image: {}", path.file_name().unwrap().to_string_lossy());
        println!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );

        let (rgb, width, height) = match load_png(path) {
            Some(d) => d,
            None => {
                println!("Failed to load image");
                continue;
            }
        };

        let pixels = width * height;
        println!("Dimensions: {}x{} ({} pixels)", width, height, pixels);

        // Write PPM for C++
        let ppm_path = "/tmp/test_image.ppm";
        write_ppm(ppm_path, &rgb, width, height).unwrap();

        println!(
            "\n{:>5} {:>12} {:>10} {:>8} {:>8} {:>10} {:>10}",
            "Q", "Encoder", "Size", "bpp", "DSSIM", "SSIM2", "vs Best"
        );
        println!("{}", "-".repeat(80));

        for &quality in &quality_levels {
            let mut results: Vec<EncoderResult> = Vec::new();

            // 1. Rust YCbCr
            let rust_ycbcr = encode_rust_jpegli(&rgb, width as u32, height as u32, quality, false);
            let rust_ycbcr_dec = decode_jpeg(&rust_ycbcr);
            results.push(EncoderResult {
                name: "Rust YCbCr".to_string(),
                quality,
                size: rust_ycbcr.len(),
                bpp: rust_ycbcr.len() as f64 * 8.0 / pixels as f64,
                dssim: compute_dssim(&rgb, &rust_ycbcr_dec, width, height),
                ssim2: compute_ssim2(&rgb, &rust_ycbcr_dec, width, height),
            });

            // 2. C++ YCbCr
            if let Some(cpp_ycbcr) = encode_cpp_cjpegli(ppm_path, quality as u32, false) {
                let cpp_ycbcr_dec = decode_jpeg(&cpp_ycbcr);
                results.push(EncoderResult {
                    name: "C++ YCbCr".to_string(),
                    quality,
                    size: cpp_ycbcr.len(),
                    bpp: cpp_ycbcr.len() as f64 * 8.0 / pixels as f64,
                    dssim: compute_dssim(&rgb, &cpp_ycbcr_dec, width, height),
                    ssim2: compute_ssim2(&rgb, &cpp_ycbcr_dec, width, height),
                });
            }

            // 3. Rust XYB
            let rust_xyb = encode_rust_jpegli(&rgb, width as u32, height as u32, quality, true);
            let rust_xyb_dec = decode_xyb_with_icc(&rust_xyb).unwrap_or_else(|| {
                // Fallback: decode without ICC (will have color cast)
                decode_jpeg(&rust_xyb)
            });
            results.push(EncoderResult {
                name: "Rust XYB".to_string(),
                quality,
                size: rust_xyb.len(),
                bpp: rust_xyb.len() as f64 * 8.0 / pixels as f64,
                dssim: compute_dssim(&rgb, &rust_xyb_dec, width, height),
                ssim2: compute_ssim2(&rgb, &rust_xyb_dec, width, height),
            });

            // 4. C++ XYB
            if let Some(cpp_xyb) = encode_cpp_cjpegli(ppm_path, quality as u32, true) {
                let cpp_xyb_dec = decode_xyb_with_icc(&cpp_xyb).unwrap_or_else(|| {
                    decode_jpeg(&cpp_xyb)
                });
                results.push(EncoderResult {
                    name: "C++ XYB".to_string(),
                    quality,
                    size: cpp_xyb.len(),
                    bpp: cpp_xyb.len() as f64 * 8.0 / pixels as f64,
                    dssim: compute_dssim(&rgb, &cpp_xyb_dec, width, height),
                    ssim2: compute_ssim2(&rgb, &cpp_xyb_dec, width, height),
                });
            }

            // Find best SSIM2 for comparison
            let best_ssim2 = results.iter().map(|r| r.ssim2).fold(0.0, f64::max);

            for r in &results {
                let diff = if r.ssim2 >= best_ssim2 - 0.01 {
                    "best".to_string()
                } else {
                    format!("{:+.2}", r.ssim2 - best_ssim2)
                };
                println!(
                    "{:>5} {:>12} {:>10} {:>8.3} {:>8.6} {:>10.2} {:>10}",
                    r.quality, r.name, r.size, r.bpp, r.dssim, r.ssim2, diff
                );
            }
            println!();
        }

        // Summary: compute gaps
        println!("\n=== Gap Analysis ===");

        for &quality in &[80, 90] {
            println!("\nAt Q{}:", quality);

            // Encode all
            let rust_ycbcr = encode_rust_jpegli(&rgb, width as u32, height as u32, quality, false);
            let rust_xyb = encode_rust_jpegli(&rgb, width as u32, height as u32, quality, true);
            let cpp_ycbcr = encode_cpp_cjpegli(ppm_path, quality as u32, false);
            let cpp_xyb = encode_cpp_cjpegli(ppm_path, quality as u32, true);

            // YCbCr implementation gap (Rust vs C++)
            if let Some(ref cpp) = cpp_ycbcr {
                let size_diff_pct =
                    100.0 * (rust_ycbcr.len() as f64 - cpp.len() as f64) / cpp.len() as f64;
                println!(
                    "  YCbCr Rust vs C++: {:+.1}% size ({} vs {} bytes)",
                    size_diff_pct,
                    rust_ycbcr.len(),
                    cpp.len()
                );
            }

            // XYB implementation gap (Rust vs C++)
            if let (Some(ref cpp_x), _rust_x) = (&cpp_xyb, &rust_xyb) {
                let size_diff_pct =
                    100.0 * (rust_xyb.len() as f64 - cpp_x.len() as f64) / cpp_x.len() as f64;
                println!(
                    "  XYB Rust vs C++:   {:+.1}% size ({} vs {} bytes)",
                    size_diff_pct,
                    rust_xyb.len(),
                    cpp_x.len()
                );
            }

            // Color space gap (XYB vs YCbCr) - for C++
            if let (Some(ref cpp_y), Some(ref cpp_x)) = (&cpp_ycbcr, &cpp_xyb) {
                let cpp_ycbcr_dec = decode_jpeg(cpp_y);
                let cpp_xyb_dec = decode_xyb_with_icc(cpp_x).unwrap_or_else(|| decode_jpeg(cpp_x));
                let ssim2_y = compute_ssim2(&rgb, &cpp_ycbcr_dec, width, height);
                let ssim2_x = compute_ssim2(&rgb, &cpp_xyb_dec, width, height);

                let size_diff_pct =
                    100.0 * (cpp_x.len() as f64 - cpp_y.len() as f64) / cpp_y.len() as f64;
                println!(
                    "  C++ XYB vs YCbCr:  {:+.1}% size, {:+.2} SSIM2 ({:.2} vs {:.2})",
                    size_diff_pct,
                    ssim2_x - ssim2_y,
                    ssim2_x,
                    ssim2_y
                );
            }

            // Color space gap for Rust
            {
                let rust_ycbcr_dec = decode_jpeg(&rust_ycbcr);
                let rust_xyb_dec =
                    decode_xyb_with_icc(&rust_xyb).unwrap_or_else(|| decode_jpeg(&rust_xyb));
                let ssim2_y = compute_ssim2(&rgb, &rust_ycbcr_dec, width, height);
                let ssim2_x = compute_ssim2(&rgb, &rust_xyb_dec, width, height);

                let size_diff_pct =
                    100.0 * (rust_xyb.len() as f64 - rust_ycbcr.len() as f64) / rust_ycbcr.len() as f64;
                println!(
                    "  Rust XYB vs YCbCr: {:+.1}% size, {:+.2} SSIM2 ({:.2} vs {:.2})",
                    size_diff_pct,
                    ssim2_x - ssim2_y,
                    ssim2_x,
                    ssim2_y
                );
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Legend:");
    println!("  - Size: smaller is better (more efficient compression)");
    println!("  - DSSIM: lower is better (0 = identical)");
    println!("  - SSIM2: higher is better (100 = identical)");
    println!("  - Implementation gap: Rust vs C++ with same color space");
    println!("  - Color space gap: XYB vs YCbCr (perceptual vs traditional)");
}
