//! XYB vs YCbCr comparison with proper ICC handling

use dssim::Dssim;
use rgb::RGBA8;
use std::fs;
use std::io::Write;
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

/// Decode JPEG using Python with ICC profile handling
fn decode_with_icc(jpeg_path: &str) -> Option<(Vec<u8>, usize, usize)> {
    // Write a Python script to decode with ICC
    let script = format!(
        r#"
from PIL import Image, ImageCms
import io
import sys

img = Image.open('{}')
icc = img.info.get('icc_profile')
if icc:
    inp = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    out = ImageCms.createProfile('sRGB')
    img = ImageCms.profileToProfile(img, inp, out)
img = img.convert('RGB')
w, h = img.size
print(f"{{w}} {{h}}")
sys.stdout.buffer.write(img.tobytes())
"#,
        jpeg_path
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Parse dimensions from first line
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line_end = stdout.find('\n')?;
    let dims: Vec<usize> = stdout[..first_line_end]
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    if dims.len() != 2 {
        return None;
    }

    let (width, height) = (dims[0], dims[1]);
    let pixel_data = output.stdout[(first_line_end + 1)..].to_vec();

    Some((pixel_data, width, height))
}

fn main() {
    let png_path = "/home/lilith/work/jpegli-rs/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";

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

    println!("Image: {}x{}\n", width, height);

    println!("=== C++ XYB vs C++ YCbCr (DSSIM with ICC correction) ===");
    println!("| Quality | XYB Size | YCbCr Size | XYB DSSIM | YCbCr DSSIM | Notes |");
    println!("|---------|----------|------------|-----------|-------------|-------|");

    for q in [50u8, 60, 70, 80, 90, 95] {
        // Encode C++ XYB
        let xyb_path = format!("/tmp/cpp_xyb_{}.jpg", q);
        Command::new("/home/lilith/work/jpegli-rs/jpegli-cpp/build/tools/cjpegli")
            .args([
                "--xyb",
                "-p",
                "0",
                "--fixed_code",
                "/tmp/test.ppm",
                &xyb_path,
                "-q",
                &q.to_string(),
            ])
            .output()
            .expect("cjpegli xyb");

        // Encode C++ YCbCr
        let ycbcr_path = format!("/tmp/cpp_ycbcr_{}.jpg", q);
        Command::new("/home/lilith/work/jpegli-rs/jpegli-cpp/build/tools/cjpegli")
            .args([
                "--chroma_subsampling=444",
                "-p",
                "0",
                "--fixed_code",
                "/tmp/test.ppm",
                &ycbcr_path,
                "-q",
                &q.to_string(),
            ])
            .output()
            .expect("cjpegli ycbcr");

        let xyb_size = fs::metadata(&xyb_path).map(|m| m.len()).unwrap_or(0);
        let ycbcr_size = fs::metadata(&ycbcr_path).map(|m| m.len()).unwrap_or(0);

        // Decode with ICC
        let xyb_decoded = decode_with_icc(&xyb_path);
        let ycbcr_decoded = decode_with_icc(&ycbcr_path);

        match (xyb_decoded, ycbcr_decoded) {
            (Some((xyb_rgb, _, _)), Some((ycbcr_rgb, _, _))) => {
                let xyb_dssim = compute_dssim(&rgb, &xyb_rgb, width, height);
                let ycbcr_dssim = compute_dssim(&rgb, &ycbcr_rgb, width, height);

                let notes = if xyb_dssim < ycbcr_dssim {
                    "XYB better"
                } else if xyb_size < ycbcr_size as u64 {
                    "XYB smaller"
                } else {
                    ""
                };

                println!(
                    "| Q{:<6} | {:>8} | {:>10} | {:>9.6} | {:>11.6} | {:6} |",
                    q, xyb_size, ycbcr_size, xyb_dssim, ycbcr_dssim, notes
                );
            }
            _ => {
                println!(
                    "| Q{:<6} | {:>8} | {:>10} | decode err | decode err |       |",
                    q, xyb_size, ycbcr_size
                );
            }
        }
    }

    println!();
    println!("=== Rust XYB vs Rust YCbCr (DSSIM with ICC correction) ===");
    println!("| Quality | XYB Size | YCbCr Size | XYB DSSIM | YCbCr DSSIM | Notes |");
    println!("|---------|----------|------------|-----------|-------------|-------|");

    for q in [50u8, 60, 70, 80, 90, 95] {
        // Encode Rust XYB
        let rust_xyb = jpegli::encode::Encoder::new()
            .width(info.width)
            .height(info.height)
            .quality(jpegli::quant::Quality::Traditional(q as f32))
            .use_xyb(true)
            .encode(&rgb);

        // Encode Rust YCbCr
        let rust_ycbcr = jpegli::encode::Encoder::new()
            .width(info.width)
            .height(info.height)
            .quality(jpegli::quant::Quality::Traditional(q as f32))
            .encode(&rgb);

        match (rust_xyb, rust_ycbcr) {
            (Ok(xyb_data), Ok(ycbcr_data)) => {
                let xyb_size = xyb_data.len();
                let ycbcr_size = ycbcr_data.len();

                // Save and decode with ICC
                fs::write("/tmp/rust_xyb.jpg", &xyb_data).unwrap();
                fs::write("/tmp/rust_ycbcr.jpg", &ycbcr_data).unwrap();

                let xyb_decoded = decode_with_icc("/tmp/rust_xyb.jpg");
                let ycbcr_decoded = decode_with_icc("/tmp/rust_ycbcr.jpg");

                match (xyb_decoded, ycbcr_decoded) {
                    (Some((xyb_rgb, _, _)), Some((ycbcr_rgb, _, _))) => {
                        let xyb_dssim = compute_dssim(&rgb, &xyb_rgb, width, height);
                        let ycbcr_dssim = compute_dssim(&rgb, &ycbcr_rgb, width, height);

                        let notes = if xyb_dssim < ycbcr_dssim {
                            "XYB better"
                        } else if xyb_size < ycbcr_size {
                            "XYB smaller"
                        } else {
                            ""
                        };

                        println!(
                            "| Q{:<6} | {:>8} | {:>10} | {:>9.6} | {:>11.6} | {:6} |",
                            q, xyb_size, ycbcr_size, xyb_dssim, ycbcr_dssim, notes
                        );
                    }
                    _ => {
                        println!(
                            "| Q{:<6} | {:>8} | {:>10} | decode err | decode err |       |",
                            q, xyb_size, ycbcr_size
                        );
                    }
                }
            }
            (Err(e), _) => println!("| Q{:<6} | XYB err: {:?} |", q, e),
            (_, Err(e)) => println!("| Q{:<6} | YCbCr err: {:?} |", q, e),
        }
    }
}
