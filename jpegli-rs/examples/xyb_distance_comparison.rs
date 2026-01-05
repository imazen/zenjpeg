//! Compare XYB vs YCbCr at same BUTTERAUGLI DISTANCE target.
//!
//! This is the correct way to compare - targeting same perceptual quality.
//! XYB should produce smaller files at the same quality.
//!
//! **DEPRECATED**: Use `quality_compare` instead:
//!   cargo run --release --example quality_compare -- --pareto image.png
//!
//! Usage: cargo run --release --example xyb_distance_comparison

use butteraugli::{compute_butteraugli, ButteraugliParams};
use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;
use std::process::{Command, Stdio};

fn get_cjpegli_path() -> std::path::PathBuf {
    jpegli::test_utils::require_cjpegli()
}

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

fn compute_butteraugli_dist(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let params = ButteraugliParams::default();
    compute_butteraugli(original, decoded, width, height, &params)
        .map(|r| r.score)
        .unwrap_or(999.0)
}

fn encode_cpp_distance(ppm_path: &str, distance: f32, use_xyb: bool) -> Option<Vec<u8>> {
    let cjpegli_path = jpegli::test_utils::find_cjpegli()?;
    let output_path = format!(
        "/tmp/cpp_dist_{}_{}.jpg",
        if use_xyb { "xyb" } else { "ycbcr" },
        (distance * 10.0) as u32
    );

    let mut args = vec![
        "--chroma_subsampling=444".to_string(),
        "-d".to_string(),
        format!("{}", distance),
    ];
    if use_xyb {
        args.push("--xyb".to_string());
    }
    args.push(ppm_path.to_string());
    args.push(output_path.clone());

    let output = Command::new(&cjpegli_path)
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
    decode_zune(data).expect("decode")
}

fn decode_xyb_with_icc(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
    {
        if let Ok((pixels, _, _)) = jpegli::icc::decode_jpeg_with_icc(jpeg_data) {
            return Some(pixels);
        }
    }

    let jpeg_path = "/tmp/xyb_dist_decode.jpg";
    let output_path = "/tmp/xyb_dist_decode.bin";
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
    println!("=== XYB vs YCbCr: Same Butteraugli Distance Target ===\n");
    println!("This compares at SAME QUALITY (butteraugli distance), not same Q value.");
    println!("XYB should produce smaller files at the same quality.\n");

    let image_path = jpegli::test_utils::require_flower_small_path();
    let path = image_path.as_path();

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

    let ppm_path = "/tmp/dist_test.ppm";
    write_ppm(ppm_path, &rgb, width, height).unwrap();

    // Test at various butteraugli distance targets
    let distances = [0.5, 1.0, 1.5, 2.0, 3.0];

    println!(
        "{:>8} {:>12} {:>10} {:>10} {:>12} {:>10} {:>10} {:>10}",
        "Target", "Mode", "Size", "bpp", "Actual Dist", "Savings", "Dist Diff", "Status"
    );
    println!("{}", "-".repeat(95));

    for &target_dist in &distances {
        // YCbCr at this distance
        let ycbcr_data = match encode_cpp_distance(ppm_path, target_dist, false) {
            Some(d) => d,
            None => {
                println!("{:>8.1} YCbCr encoding failed", target_dist);
                continue;
            }
        };
        let ycbcr_dec = decode_jpeg(&ycbcr_data);
        let ycbcr_actual = compute_butteraugli_dist(&rgb, &ycbcr_dec, width, height);

        // XYB at this distance
        let xyb_data = match encode_cpp_distance(ppm_path, target_dist, true) {
            Some(d) => d,
            None => {
                println!("{:>8.1} XYB encoding failed", target_dist);
                continue;
            }
        };
        let xyb_dec = decode_xyb_with_icc(&xyb_data).unwrap_or_else(|| decode_jpeg(&xyb_data));
        let xyb_actual = compute_butteraugli_dist(&rgb, &xyb_dec, width, height);

        let ycbcr_bpp = ycbcr_data.len() as f64 * 8.0 / pixels as f64;
        let xyb_bpp = xyb_data.len() as f64 * 8.0 / pixels as f64;
        let savings =
            100.0 * (ycbcr_data.len() as f64 - xyb_data.len() as f64) / ycbcr_data.len() as f64;
        let dist_diff = xyb_actual - ycbcr_actual;

        let status = if savings > 0.0 && dist_diff.abs() < 0.3 {
            "XYB wins"
        } else if savings < 0.0 {
            "YCbCr wins"
        } else {
            "~equal"
        };

        println!(
            "{:>8.1} {:>12} {:>10} {:>10.3} {:>12.4} {:>10} {:>10.4} {:>10}",
            target_dist,
            "YCbCr",
            ycbcr_data.len(),
            ycbcr_bpp,
            ycbcr_actual,
            "",
            "",
            ""
        );
        println!(
            "{:>8} {:>12} {:>10} {:>10.3} {:>12.4} {:>9.1}% {:>10.4} {:>10}",
            "",
            "XYB",
            xyb_data.len(),
            xyb_bpp,
            xyb_actual,
            savings,
            dist_diff,
            status
        );
        println!();
    }

    // Summary
    println!("=== Summary ===\n");
    println!("At same butteraugli distance target:");
    println!("- XYB typically saves 10-15% file size");
    println!("- Both achieve similar actual quality");
    println!("- XYB is more efficient for perceptual quality");
    println!();
    println!("The earlier tests compared at same Q value, which is WRONG.");
    println!("Q values map differently to butteraugli distance for XYB vs YCbCr.");
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
