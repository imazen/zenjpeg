//! Compare XYB vs YCbCr at SAME FILE SIZE to find quality difference.
//!
//! This is the correct way to measure XYB efficiency - same file size, compare quality.
//!
//! Usage: cargo run --release --example xyb_same_size_comparison --features cms-lcms2

use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
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

fn encode_cpp_quality(ppm_path: &str, quality: u32, use_xyb: bool) -> Option<Vec<u8>> {
    if !Path::new(CJPEGLI_PATH).exists() {
        return None;
    }
    let output_path = format!(
        "/tmp/samesize_{}_{}.jpg",
        if use_xyb { "xyb" } else { "ycbcr" },
        quality
    );

    let mut args = vec![
        "--chroma_subsampling=444".to_string(),
        "-q".to_string(),
        format!("{}", quality),
    ];
    if use_xyb {
        args.push("--xyb".to_string());
    }
    args.push(ppm_path.to_string());
    args.push(output_path.clone());

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

fn decode_jpeg_simple(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(data).decode().expect("decode")
}

fn decode_xyb_with_icc(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    // Try Rust CMS first
    #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
    {
        if let Ok((pixels, _, _)) = jpegli::icc::decode_jpeg_with_icc(jpeg_data) {
            return Some(pixels);
        }
    }

    // Fallback to Python with Pillow
    let jpeg_path = "/tmp/xyb_samesize_decode.jpg";
    let output_path = "/tmp/xyb_samesize_decode.bin";
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

fn compute_ssimulacra2(orig: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
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

    compute_frame_ssimulacra2(orig_rgb, dec_rgb).unwrap_or(-1.0)
}

fn main() {
    println!("=== XYB vs YCbCr: Same File Size Comparison ===\n");
    println!("For each quality level, find comparable file sizes and compare SSIMULACRA2.\n");

    let image_path = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";
    let path = Path::new(image_path);

    let (rgb, width, height) = match load_png(path) {
        Some(d) => d,
        None => {
            eprintln!("Failed to load image");
            return;
        }
    };

    println!(
        "Image: {} ({}x{})\n",
        path.file_name().unwrap().to_string_lossy(),
        width,
        height
    );

    let ppm_path = "/tmp/samesize_test.ppm";
    write_ppm(ppm_path, &rgb, width, height).unwrap();

    // Strategy: For each YCbCr quality level, find XYB quality that produces similar file size
    let ycbcr_qualities = [50, 60, 70, 80, 90];

    println!(
        "{:>6} {:>8} {:>12} {:>6} {:>8} {:>12} {:>10}",
        "YCbCr", "Size", "SSIMULACRA2", "XYB", "Size", "SSIMULACRA2", "Diff"
    );
    println!("{}", "-".repeat(72));

    for &yq in &ycbcr_qualities {
        // Encode YCbCr at this quality
        let ycbcr_data = match encode_cpp_quality(ppm_path, yq, false) {
            Some(d) => d,
            None => continue,
        };
        let ycbcr_size = ycbcr_data.len();
        let ycbcr_dec = decode_jpeg_simple(&ycbcr_data);
        let ycbcr_ssim = compute_ssimulacra2(&rgb, &ycbcr_dec, width, height);

        // Binary search to find XYB quality that produces similar file size
        let mut best_xyb_q = yq;
        let mut best_size_diff = i64::MAX;
        let mut best_xyb_data = None;

        for xq in 30..=100 {
            if let Some(xyb_data) = encode_cpp_quality(ppm_path, xq, true) {
                let size_diff = (xyb_data.len() as i64 - ycbcr_size as i64).abs();
                if size_diff < best_size_diff {
                    best_size_diff = size_diff;
                    best_xyb_q = xq;
                    best_xyb_data = Some(xyb_data);
                }
            }
        }

        if let Some(xyb_data) = best_xyb_data {
            let xyb_size = xyb_data.len();
            let xyb_dec = decode_xyb_with_icc(&xyb_data).unwrap_or_else(|| decode_jpeg_simple(&xyb_data));
            let xyb_ssim = compute_ssimulacra2(&rgb, &xyb_dec, width, height);

            let ssim_diff = xyb_ssim - ycbcr_ssim;
            let size_diff_pct =
                100.0 * (xyb_size as f64 - ycbcr_size as f64) / ycbcr_size as f64;

            let status = if ssim_diff > 0.5 {
                "XYB better"
            } else if ssim_diff < -0.5 {
                "YCbCr better"
            } else {
                "~equal"
            };

            println!(
                "Q{:>4} {:>8} {:>12.2} Q{:>4} {:>8} {:>12.2} {:>+7.2} {}",
                yq, ycbcr_size, ycbcr_ssim, best_xyb_q, xyb_size, xyb_ssim, ssim_diff, status
            );
        }
    }

    println!("\n=== Summary ===\n");
    println!("At same file size (within 1-2%), compare SSIMULACRA2 scores.");
    println!("Positive diff = XYB is better, Negative = YCbCr is better.");
}
