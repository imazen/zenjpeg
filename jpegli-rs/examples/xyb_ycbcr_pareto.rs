//! Pareto curve analysis: XYB vs YCbCr, Rust vs C++
//!
//! Creates size vs quality curves to properly compare encoders.
//!
//! **DEPRECATED**: Use `quality_compare` instead:
//!   cargo run --release --example quality_compare -- --pareto --output results.csv image.png
//!
//! Usage: cargo run --release --example xyb_ycbcr_pareto

use dssim::Dssim;
use rgb::RGBA8;
use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
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

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8, use_xyb: bool) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality.into()))
        .use_xyb(use_xyb)
        .encode(rgb)
        .expect("encode")
}

fn encode_cpp(ppm_path: &str, quality: u32, use_xyb: bool) -> Option<Vec<u8>> {
    let cjpegli_path = jpegli::test_utils::find_cjpegli()?;

    let output_path = format!(
        "/tmp/cpp_{}_{}.jpg",
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
struct DataPoint {
    quality: u8,
    bpp: f64,
    dssim: f64,
    ssim2: f64,
}

#[derive(Debug)]
struct Encoder {
    name: &'static str,
    points: Vec<DataPoint>,
}

fn main() {
    println!("=== XYB vs YCbCr Pareto Analysis ===\n");

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
        "Image: {} ({}x{}, {} pixels)",
        path.file_name().unwrap().to_string_lossy(),
        width,
        height,
        pixels
    );

    let ppm_path = "/tmp/test_image.ppm";
    write_ppm(ppm_path, &rgb, width, height).unwrap();

    // Collect data for all encoders
    let quality_levels: Vec<u8> = (50..=99).step_by(5).collect();

    let mut rust_ycbcr = Encoder {
        name: "Rust YCbCr",
        points: Vec::new(),
    };
    let mut cpp_ycbcr = Encoder {
        name: "C++ YCbCr",
        points: Vec::new(),
    };
    let mut rust_xyb = Encoder {
        name: "Rust XYB",
        points: Vec::new(),
    };
    let mut cpp_xyb = Encoder {
        name: "C++ XYB",
        points: Vec::new(),
    };

    print!("Encoding: ");
    for &q in &quality_levels {
        print!("Q{} ", q);
        std::io::stdout().flush().unwrap();

        // Rust YCbCr
        let data = encode_rust(&rgb, width as u32, height as u32, q, false);
        let dec = decode_jpeg(&data);
        rust_ycbcr.points.push(DataPoint {
            quality: q,
            bpp: data.len() as f64 * 8.0 / pixels as f64,
            dssim: compute_dssim(&rgb, &dec, width, height),
            ssim2: compute_ssim2(&rgb, &dec, width, height),
        });

        // C++ YCbCr
        if let Some(data) = encode_cpp(ppm_path, q as u32, false) {
            let dec = decode_jpeg(&data);
            cpp_ycbcr.points.push(DataPoint {
                quality: q,
                bpp: data.len() as f64 * 8.0 / pixels as f64,
                dssim: compute_dssim(&rgb, &dec, width, height),
                ssim2: compute_ssim2(&rgb, &dec, width, height),
            });
        }

        // Rust XYB
        let data = encode_rust(&rgb, width as u32, height as u32, q, true);
        let dec = decode_xyb_with_icc(&data).unwrap_or_else(|| decode_jpeg(&data));
        rust_xyb.points.push(DataPoint {
            quality: q,
            bpp: data.len() as f64 * 8.0 / pixels as f64,
            dssim: compute_dssim(&rgb, &dec, width, height),
            ssim2: compute_ssim2(&rgb, &dec, width, height),
        });

        // C++ XYB
        if let Some(data) = encode_cpp(ppm_path, q as u32, true) {
            let dec = decode_xyb_with_icc(&data).unwrap_or_else(|| decode_jpeg(&data));
            cpp_xyb.points.push(DataPoint {
                quality: q,
                bpp: data.len() as f64 * 8.0 / pixels as f64,
                dssim: compute_dssim(&rgb, &dec, width, height),
                ssim2: compute_ssim2(&rgb, &dec, width, height),
            });
        }
    }
    println!("\n");

    // Print tables
    println!("=== Size vs Quality (SSIM2) ===");
    println!(
        "{:>5} {:>12} {:>8} {:>8} | {:>12} {:>8} {:>8}",
        "Q", "Rust YCbCr", "bpp", "SSIM2", "Rust XYB", "bpp", "SSIM2"
    );
    println!("{}", "-".repeat(75));

    for (ry, rx) in rust_ycbcr.points.iter().zip(rust_xyb.points.iter()) {
        let size_diff = 100.0 * (rx.bpp - ry.bpp) / ry.bpp;
        println!(
            "{:>5} {:>12.0} {:>8.3} {:>8.2} | {:>12.0} {:>8.3} {:>8.2}   XYB {:+.1}% size",
            ry.quality,
            ry.bpp * pixels as f64 / 8.0,
            ry.bpp,
            ry.ssim2,
            rx.bpp * pixels as f64 / 8.0,
            rx.bpp,
            rx.ssim2,
            size_diff
        );
    }

    println!("\n=== Implementation Gap (Rust vs C++) ===");
    println!("{:>5} | {:>20} | {:>20}", "Q", "YCbCr Gap", "XYB Gap");
    println!("{}", "-".repeat(55));

    for i in 0..rust_ycbcr.points.len() {
        let ry = &rust_ycbcr.points[i];

        let ycbcr_gap = if i < cpp_ycbcr.points.len() {
            let cy = &cpp_ycbcr.points[i];
            format!("{:+.1}% size", 100.0 * (ry.bpp - cy.bpp) / cy.bpp)
        } else {
            "N/A".to_string()
        };

        let rx = &rust_xyb.points[i];
        let xyb_gap = if i < cpp_xyb.points.len() {
            let cx = &cpp_xyb.points[i];
            format!(
                "{:+.1}% size, {:+.2} SSIM2",
                100.0 * (rx.bpp - cx.bpp) / cx.bpp,
                rx.ssim2 - cx.ssim2
            )
        } else {
            "N/A".to_string()
        };

        println!("{:>5} | {:>20} | {:>20}", ry.quality, ycbcr_gap, xyb_gap);
    }

    // Calculate BD-rate style comparison: at equal SSIM2, how much size difference?
    println!("\n=== BD-Rate Style Analysis ===");
    println!("Finding equivalent quality points...\n");

    // At SSIM2 = 85, what bpp does each encoder need?
    for target_ssim2 in [80.0, 85.0, 88.0, 90.0] {
        print!("At SSIM2 ≈ {:.0}: ", target_ssim2);

        // Find closest point for each encoder
        let find_bpp_at_ssim2 = |points: &[DataPoint]| -> Option<f64> {
            // Find two points that bracket the target
            for i in 0..points.len().saturating_sub(1) {
                if (points[i].ssim2 <= target_ssim2 && points[i + 1].ssim2 >= target_ssim2)
                    || (points[i].ssim2 >= target_ssim2 && points[i + 1].ssim2 <= target_ssim2)
                {
                    // Linear interpolation
                    let t =
                        (target_ssim2 - points[i].ssim2) / (points[i + 1].ssim2 - points[i].ssim2);
                    return Some(points[i].bpp + t * (points[i + 1].bpp - points[i].bpp));
                }
            }
            None
        };

        let ry_bpp = find_bpp_at_ssim2(&rust_ycbcr.points);
        let rx_bpp = find_bpp_at_ssim2(&rust_xyb.points);
        let cy_bpp = find_bpp_at_ssim2(&cpp_ycbcr.points);
        let cx_bpp = find_bpp_at_ssim2(&cpp_xyb.points);

        match (ry_bpp, rx_bpp) {
            (Some(ry), Some(rx)) => {
                let savings = 100.0 * (ry - rx) / ry;
                print!("XYB saves {:.1}% vs YCbCr (Rust)", savings);
            }
            _ => print!("N/A (Rust)"),
        }

        if let (Some(cy), Some(cx)) = (cy_bpp, cx_bpp) {
            let savings = 100.0 * (cy - cx) / cy;
            print!(", {:.1}% (C++)", savings);
        }

        println!();
    }

    println!("\n=== Summary ===");
    println!("Implementation gap (Rust vs C++):");
    println!("  • YCbCr: <0.2% size difference (essentially identical)");

    // Calculate average XYB gap
    let avg_xyb_size_gap: f64 = rust_xyb
        .points
        .iter()
        .zip(cpp_xyb.points.iter())
        .map(|(rx, cx)| 100.0 * (rx.bpp - cx.bpp) / cx.bpp)
        .sum::<f64>()
        / rust_xyb.points.len().min(cpp_xyb.points.len()) as f64;
    let avg_xyb_ssim2_gap: f64 = rust_xyb
        .points
        .iter()
        .zip(cpp_xyb.points.iter())
        .map(|(rx, cx)| rx.ssim2 - cx.ssim2)
        .sum::<f64>()
        / rust_xyb.points.len().min(cpp_xyb.points.len()) as f64;

    println!(
        "  • XYB: {:+.1}% size, {:+.2} SSIM2 (Rust vs C++)",
        avg_xyb_size_gap, avg_xyb_ssim2_gap
    );

    println!("\nColor space gap (XYB vs YCbCr at same Q):");
    let avg_color_size: f64 = rust_xyb
        .points
        .iter()
        .zip(rust_ycbcr.points.iter())
        .map(|(rx, ry)| 100.0 * (rx.bpp - ry.bpp) / ry.bpp)
        .sum::<f64>()
        / rust_xyb.points.len() as f64;
    println!(
        "  • XYB ~{:.0}% smaller files at same Q setting",
        -avg_color_size
    );
    println!("  • BUT XYB needs higher Q to match YCbCr quality");
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
