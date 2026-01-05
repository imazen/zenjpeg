//! Compare XYB vs YCbCr at SAME FILE SIZE to find quality difference.
//!
//! This is the correct way to measure XYB efficiency - same file size, compare quality.
//! Uses all three metrics: DSSIM, SSIMULACRA2, and Butteraugli.
//!
//! Usage: cargo run --release --example xyb_same_size_comparison

use jpegli::icc::{apply_icc_transform, extract_icc_profile};
use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
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

fn encode_cpp_quality(input_path: &Path, quality: u32, use_xyb: bool) -> Option<Vec<u8>> {
    let cjpegli_path = jpegli::test_utils::find_cjpegli()?;
    let output_path = std::env::temp_dir().join(format!(
        "samesize_{}_{}.jpg",
        if use_xyb { "xyb" } else { "ycbcr" },
        quality
    ));

    let mut cmd = Command::new(&cjpegli_path);
    cmd.arg(input_path)
        .arg(&output_path)
        .arg(format!("--quality={}", quality));

    if use_xyb {
        cmd.arg("--xyb");
    }

    let output = cmd
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }
    let data = fs::read(&output_path).ok()?;
    let _ = fs::remove_file(&output_path);
    Some(data)
}

fn decode_jpeg_simple(data: &[u8]) -> Vec<u8> {
    decode_zune(data).expect("decode")
}

fn decode_xyb_with_icc(jpeg_data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let icc_profile = extract_icc_profile(jpeg_data);

    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(jpeg_data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.dimensions()?;

    let rgb = match info.pixel_format {
        3 /* RGB */ => pixels,
        1 /* Grayscale */ => pixels.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => return None,
    };

    let width = info.width as usize;
    let height = info.height as usize;

    let output = if let Some(ref profile) = icc_profile {
        apply_icc_transform(&rgb, width, height, profile).unwrap_or(rgb)
    } else {
        rgb
    };

    Some((output, width, height))
}

fn compute_dssim(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use dssim::Dssim;
    use rgb::RGBA;

    let attr = Dssim::new();

    let orig_rgba: Vec<RGBA<u8>> = orig
        .chunks(3)
        .map(|c| RGBA::new(c[0], c[1], c[2], 255))
        .collect();
    let comp_rgba: Vec<RGBA<u8>> = comp
        .chunks(3)
        .map(|c| RGBA::new(c[0], c[1], c[2], 255))
        .collect();

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp_img = attr.create_image_rgba(&comp_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, comp_img);
    dssim.into()
}

fn compute_butteraugli(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use butteraugli::{compute_butteraugli, ButteraugliParams};

    let params = ButteraugliParams::default();
    match compute_butteraugli(orig, comp, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => f64::NAN,
    }
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

/// Binary search to find XYB quality that matches target file size
fn find_matching_xyb_quality(input_path: &Path, target_size: usize) -> Option<(u32, Vec<u8>)> {
    let mut best_quality = 50u32;
    let mut best_data: Option<Vec<u8>> = None;
    let mut best_diff = i64::MAX;

    // Linear search (could optimize with binary search)
    for quality in (30..=100).step_by(2) {
        if let Some(data) = encode_cpp_quality(input_path, quality, true) {
            let diff = (data.len() as i64 - target_size as i64).abs();
            if diff < best_diff {
                best_diff = diff;
                best_quality = quality;
                best_data = Some(data);
            }
        }
    }

    best_data.map(|d| (best_quality, d))
}

fn main() {
    let testdata_dir = jpegli::test_utils::get_testdata_dir();
    let flower_dir = testdata_dir.join("jxl/flower");

    let corpus_dir = if flower_dir.exists() {
        Some(flower_dir)
    } else {
        None
    }
    .or_else(|| {
        std::env::var("CORPUS_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|d| d.exists())
    })
    .or_else(|| {
        let candidates = [
            "../codec-eval/codec-corpus/kodak",
            "../codec-corpus/kodak",
            "codec-corpus/kodak",
        ];
        candidates
            .iter()
            .find(|p| Path::new(p).exists())
            .map(PathBuf::from)
    })
    .expect("No corpus found. Set CORPUS_DIR or JPEGLI_TESTDATA env var.");

    let ycbcr_qualities: Vec<u32> = vec![50, 70, 80, 90, 95];

    eprintln!("=== XYB vs YCbCr: Same File Size Comparison ===");
    eprintln!("Adjusting XYB quality to match YCbCr file size, then comparing quality metrics.\n");
    eprintln!("cjpegli: {}", get_cjpegli_path().display());
    eprintln!("corpus: {}\n", corpus_dir.display());

    println!(
        "{:<8} {:>4} {:>4} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6}",
        "Image",
        "Q_Y",
        "Q_X",
        "Size_Y",
        "Size_X",
        "DSSIM_Y",
        "DSSIM_X",
        "Butt_Y",
        "Butt_X",
        "SSIM_Y",
        "SSIM_X"
    );
    println!("{}", "-".repeat(110));

    let mut png_files: Vec<_> = fs::read_dir(&corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .map(|e| e.path())
        .collect();
    png_files.sort();

    let max_images = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    png_files.truncate(max_images);

    let mut total_ycbcr_dssim = 0.0;
    let mut total_xyb_dssim = 0.0;
    let mut total_ycbcr_butt = 0.0;
    let mut total_xyb_butt = 0.0;
    let mut total_ycbcr_ssim = 0.0;
    let mut total_xyb_ssim = 0.0;
    let mut count = 0;

    for png_path in &png_files {
        let name = png_path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Some((orig_rgb, width, height)) = load_png(png_path) else {
            continue;
        };

        for &ycbcr_quality in &ycbcr_qualities {
            // Encode with YCbCr
            let Some(ycbcr_jpeg) = encode_cpp_quality(png_path, ycbcr_quality, false) else {
                continue;
            };
            let ycbcr_size = ycbcr_jpeg.len();

            // Find XYB quality that matches this file size
            let Some((xyb_quality, xyb_jpeg)) = find_matching_xyb_quality(png_path, ycbcr_size)
            else {
                continue;
            };
            let xyb_size = xyb_jpeg.len();

            // Decode YCbCr
            let ycbcr_decoded = decode_jpeg_simple(&ycbcr_jpeg);

            // Decode XYB with ICC
            let Some((xyb_decoded, _, _)) = decode_xyb_with_icc(&xyb_jpeg) else {
                continue;
            };

            // Compute all three metrics
            let ycbcr_dssim = compute_dssim(&orig_rgb, &ycbcr_decoded, width, height);
            let xyb_dssim = compute_dssim(&orig_rgb, &xyb_decoded, width, height);

            let ycbcr_butt = compute_butteraugli(&orig_rgb, &ycbcr_decoded, width, height);
            let xyb_butt = compute_butteraugli(&orig_rgb, &xyb_decoded, width, height);

            let ycbcr_ssim = compute_ssimulacra2(&orig_rgb, &ycbcr_decoded, width, height);
            let xyb_ssim = compute_ssimulacra2(&orig_rgb, &xyb_decoded, width, height);

            println!(
                "{:<8} {:>4} {:>4} {:>8} {:>8} {:>.6} {:>.6} {:>6.2} {:>6.2} {:>6.1} {:>6.1}",
                name,
                ycbcr_quality,
                xyb_quality,
                ycbcr_size,
                xyb_size,
                ycbcr_dssim,
                xyb_dssim,
                ycbcr_butt,
                xyb_butt,
                ycbcr_ssim,
                xyb_ssim
            );

            total_ycbcr_dssim += ycbcr_dssim;
            total_xyb_dssim += xyb_dssim;
            total_ycbcr_butt += ycbcr_butt;
            total_xyb_butt += xyb_butt;
            total_ycbcr_ssim += ycbcr_ssim;
            total_xyb_ssim += xyb_ssim;
            count += 1;
        }
    }

    println!("{}", "-".repeat(110));

    if count > 0 {
        let avg_ycbcr_dssim = total_ycbcr_dssim / count as f64;
        let avg_xyb_dssim = total_xyb_dssim / count as f64;
        let avg_ycbcr_butt = total_ycbcr_butt / count as f64;
        let avg_xyb_butt = total_xyb_butt / count as f64;
        let avg_ycbcr_ssim = total_ycbcr_ssim / count as f64;
        let avg_xyb_ssim = total_xyb_ssim / count as f64;

        eprintln!("\n=== SUMMARY (at matched file sizes) ===");
        eprintln!("Total comparisons: {}\n", count);

        eprintln!("DSSIM (lower = better):");
        eprintln!("  YCbCr avg: {:.6}", avg_ycbcr_dssim);
        eprintln!("  XYB avg:   {:.6}", avg_xyb_dssim);
        let dssim_diff = ((avg_xyb_dssim - avg_ycbcr_dssim) / avg_ycbcr_dssim) * 100.0;
        if avg_xyb_dssim < avg_ycbcr_dssim {
            eprintln!("  Winner: XYB ({:.1}% better)\n", -dssim_diff);
        } else {
            eprintln!("  Winner: YCbCr ({:.1}% better)\n", dssim_diff);
        }

        eprintln!("Butteraugli (lower = better):");
        eprintln!("  YCbCr avg: {:.2}", avg_ycbcr_butt);
        eprintln!("  XYB avg:   {:.2}", avg_xyb_butt);
        let butt_diff = ((avg_xyb_butt - avg_ycbcr_butt) / avg_ycbcr_butt) * 100.0;
        if avg_xyb_butt < avg_ycbcr_butt {
            eprintln!("  Winner: XYB ({:.1}% better)\n", -butt_diff);
        } else {
            eprintln!("  Winner: YCbCr ({:.1}% better)\n", butt_diff);
        }

        eprintln!("SSIMULACRA2 (higher = better):");
        eprintln!("  YCbCr avg: {:.2}", avg_ycbcr_ssim);
        eprintln!("  XYB avg:   {:.2}", avg_xyb_ssim);
        let ssim_diff = avg_xyb_ssim - avg_ycbcr_ssim;
        if avg_xyb_ssim > avg_ycbcr_ssim {
            eprintln!("  Winner: XYB ({:+.2} points)\n", ssim_diff);
        } else {
            eprintln!("  Winner: YCbCr ({:+.2} points)\n", -ssim_diff);
        }
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
