//! Compare XYB vs YCbCr using Butteraugli (the metric XYB is designed for).
//!
//! Usage: cargo run --release --example xyb_butteraugli_comparison

use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};
use dssim::Dssim;
use rgb::RGBA8;
use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;
use std::process::{Command, Stdio};

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

fn compute_butteraugli_dist(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let params = ButteraugliParams::default();
    compute_butteraugli(original, decoded, width, height, &params)
        .map(|r| r.score)
        .unwrap_or(999.0)
}

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8, use_xyb: bool) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(quality.into()))
        .use_xyb(use_xyb)
        .encode(rgb)
        .expect("encode")
}

fn encode_cpp(ppm_path: &str, quality: u32, use_xyb: bool) -> Option<Vec<u8>> {
    if !Path::new(CJPEGLI_PATH).exists() {
        return None;
    }

    let output_path = format!(
        "/tmp/cpp_butter_{}_{}.jpg",
        if use_xyb { "xyb" } else { "ycbcr" },
        quality
    );

    let mut args = vec!["--chroma_subsampling=444", "-p", "0"];
    if use_xyb {
        args.push("--xyb");
    }
    args.push(ppm_path);
    args.push(&output_path);
    args.push("-q");
    let q_str = quality.to_string();
    args.push(&q_str);

    let output = Command::new(CJPEGLI_PATH)
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    fs::read(&output_path).ok()
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(data).decode().expect("decode")
}

fn decode_xyb_with_icc(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
    {
        if let Ok((pixels, _, _)) = jpegli::icc::decode_jpeg_with_icc(jpeg_data) {
            return Some(pixels);
        }
    }

    let jpeg_path = "/tmp/xyb_decode_butter.jpg";
    let output_path = "/tmp/xyb_decode_butter.bin";
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

fn main() {
    println!("=== XYB vs YCbCr: Butteraugli Comparison ===\n");
    println!("Butteraugli is the metric XYB is optimized for.");
    println!("Lower Butteraugli = better perceptual quality.\n");

    let image_path = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";
    let path = Path::new(image_path);

    let (rgb, width, height) = match load_png(path) {
        Some(d) => d,
        None => {
            eprintln!("Failed to load image");
            return;
        }
    };

    let pixels = width * height;
    println!(
        "Image: {} ({}x{})\n",
        path.file_name().unwrap().to_string_lossy(),
        width,
        height
    );

    let ppm_path = "/tmp/test_butter.ppm";
    write_ppm(ppm_path, &rgb, width, height).unwrap();

    let quality_levels: Vec<u8> = vec![70, 80, 85, 90, 95];

    println!(
        "{:>5} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "Q", "Encoder", "Size", "bpp", "Butter", "DSSIM"
    );
    println!("{}", "-".repeat(70));

    for &q in &quality_levels {
        let mut results: Vec<(&str, usize, f64, f64)> = Vec::new();

        // Rust YCbCr
        let data = encode_rust(&rgb, width as u32, height as u32, q, false);
        let dec = decode_jpeg(&data);
        let butter = compute_butteraugli_dist(&rgb, &dec, width, height);
        let dssim = compute_dssim(&rgb, &dec, width, height);
        results.push(("Rust YCbCr", data.len(), butter, dssim));

        // C++ YCbCr
        if let Some(data) = encode_cpp(ppm_path, q as u32, false) {
            let dec = decode_jpeg(&data);
            let butter = compute_butteraugli_dist(&rgb, &dec, width, height);
            let dssim = compute_dssim(&rgb, &dec, width, height);
            results.push(("C++ YCbCr", data.len(), butter, dssim));
        }

        // Rust XYB
        let data = encode_rust(&rgb, width as u32, height as u32, q, true);
        let dec = decode_xyb_with_icc(&data).unwrap_or_else(|| decode_jpeg(&data));
        let butter = compute_butteraugli_dist(&rgb, &dec, width, height);
        let dssim = compute_dssim(&rgb, &dec, width, height);
        results.push(("Rust XYB", data.len(), butter, dssim));

        // C++ XYB
        if let Some(data) = encode_cpp(ppm_path, q as u32, true) {
            let dec = decode_xyb_with_icc(&data).unwrap_or_else(|| decode_jpeg(&data));
            let butter = compute_butteraugli_dist(&rgb, &dec, width, height);
            let dssim = compute_dssim(&rgb, &dec, width, height);
            results.push(("C++ XYB", data.len(), butter, dssim));
        }

        for (name, size, butter, dssim) in &results {
            let bpp = *size as f64 * 8.0 / pixels as f64;
            println!(
                "{:>5} {:>12} {:>10} {:>10.3} {:>10.4} {:>10.6}",
                q, name, size, bpp, butter, dssim
            );
        }
        println!();
    }

    // Summary analysis
    println!("=== Analysis at Q90 ===\n");

    let rust_ycbcr = encode_rust(&rgb, width as u32, height as u32, 90, false);
    let rust_xyb = encode_rust(&rgb, width as u32, height as u32, 90, true);
    let cpp_ycbcr = encode_cpp(ppm_path, 90, false);
    let cpp_xyb = encode_cpp(ppm_path, 90, true);

    let rust_ycbcr_dec = decode_jpeg(&rust_ycbcr);
    let rust_xyb_dec = decode_xyb_with_icc(&rust_xyb).unwrap_or_else(|| decode_jpeg(&rust_xyb));

    let rust_ycbcr_butter = compute_butteraugli_dist(&rgb, &rust_ycbcr_dec, width, height);
    let rust_xyb_butter = compute_butteraugli_dist(&rgb, &rust_xyb_dec, width, height);

    println!(
        "Rust YCbCr: {} bytes, Butteraugli {:.4}",
        rust_ycbcr.len(),
        rust_ycbcr_butter
    );
    println!(
        "Rust XYB:   {} bytes, Butteraugli {:.4}",
        rust_xyb.len(),
        rust_xyb_butter
    );
    println!();

    let size_diff =
        100.0 * (rust_xyb.len() as f64 - rust_ycbcr.len() as f64) / rust_ycbcr.len() as f64;
    let butter_diff = rust_xyb_butter - rust_ycbcr_butter;

    println!("XYB vs YCbCr:");
    println!("  Size: {:+.1}%", size_diff);
    println!("  Butteraugli: {:+.4} (negative = XYB better)", butter_diff);

    if let (Some(cpp_y), Some(cpp_x)) = (&cpp_ycbcr, &cpp_xyb) {
        let cpp_ycbcr_dec = decode_jpeg(cpp_y);
        let cpp_xyb_dec = decode_xyb_with_icc(cpp_x).unwrap_or_else(|| decode_jpeg(cpp_x));

        let cpp_ycbcr_butter = compute_butteraugli_dist(&rgb, &cpp_ycbcr_dec, width, height);
        let cpp_xyb_butter = compute_butteraugli_dist(&rgb, &cpp_xyb_dec, width, height);

        println!(
            "\nC++ YCbCr: {} bytes, Butteraugli {:.4}",
            cpp_y.len(),
            cpp_ycbcr_butter
        );
        println!(
            "C++ XYB:   {} bytes, Butteraugli {:.4}",
            cpp_x.len(),
            cpp_xyb_butter
        );

        let cpp_size_diff = 100.0 * (cpp_x.len() as f64 - cpp_y.len() as f64) / cpp_y.len() as f64;
        let cpp_butter_diff = cpp_xyb_butter - cpp_ycbcr_butter;

        println!("\nC++ XYB vs YCbCr:");
        println!("  Size: {:+.1}%", cpp_size_diff);
        println!(
            "  Butteraugli: {:+.4} (negative = XYB better)",
            cpp_butter_diff
        );

        // Rust vs C++ XYB gap
        let rust_cpp_size =
            100.0 * (rust_xyb.len() as f64 - cpp_x.len() as f64) / cpp_x.len() as f64;
        let rust_cpp_butter = rust_xyb_butter - cpp_xyb_butter;

        println!("\nRust XYB vs C++ XYB:");
        println!("  Size: {:+.1}%", rust_cpp_size);
        println!(
            "  Butteraugli: {:+.4} (negative = Rust better)",
            rust_cpp_butter
        );
    }
}
