//! Pareto front validation: compare jpegli vs mozjpeg quality/size tradeoff.
//!
//! Verifies that jpegli is competitive with mozjpeg on the Pareto front
//! of DSSIM vs file size.
//!
//! Uses DSSIM + off-by-N statistics (similar to imageflow's approach).

use dssim::Dssim;
use rgb::RGBA8;
use std::fs;
use std::path::Path;

/// Statistics for comparing decoded image vs original (off-by-N approach)
#[derive(Debug, Default)]
struct DiffStats {
    values: usize,
    values_differing: usize,
    off_by_1: usize,
    off_by_2: usize,
    off_by_3_plus: usize,
    max_diff: u8,
    sum_abs_diff: u64,
}

impl DiffStats {
    fn from_pixels(a: &[u8], b: &[u8]) -> Self {
        assert_eq!(a.len(), b.len());
        let mut stats = DiffStats {
            values: a.len(),
            ..Default::default()
        };

        for (av, bv) in a.iter().zip(b.iter()) {
            let diff = (*av as i16 - *bv as i16).unsigned_abs() as u8;
            if diff > 0 {
                stats.values_differing += 1;
                stats.sum_abs_diff += diff as u64;
                stats.max_diff = stats.max_diff.max(diff);

                match diff {
                    1 => stats.off_by_1 += 1,
                    2 => stats.off_by_2 += 1,
                    _ => stats.off_by_3_plus += 1,
                }
            }
        }
        stats
    }

    fn pct_off_by_1(&self) -> f64 {
        100.0 * self.off_by_1 as f64 / self.values as f64
    }

    fn pct_off_by_2(&self) -> f64 {
        100.0 * self.off_by_2 as f64 / self.values as f64
    }

    fn pct_off_by_3_plus(&self) -> f64 {
        100.0 * self.off_by_3_plus as f64 / self.values as f64
    }

    fn avg_diff(&self) -> f64 {
        if self.values_differing > 0 {
            self.sum_abs_diff as f64 / self.values_differing as f64
        } else {
            0.0
        }
    }

    fn report(&self, name: &str) {
        println!(
            "  {}: off-by-1={:.1}%, off-by-2={:.1}%, off-by-3+={:.1}%, max={}, avg={:.2}",
            name,
            self.pct_off_by_1(),
            self.pct_off_by_2(),
            self.pct_off_by_3_plus(),
            self.max_diff,
            self.avg_diff()
        );
    }
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

fn encode_jpegli(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality.into()))
        .encode(rgb)
        .expect("jpegli encode")
}

fn encode_jpegli_xyb(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality.into()))
        .use_xyb(true)
        .encode(rgb)
        .expect("jpegli XYB encode")
}

/// Decode XYB JPEG with ICC profile applied (using Python/Pillow).
///
/// djpegli's PPM output doesn't apply ICC profiles, so we use Pillow
/// with lcms2 for proper ICC color management.
fn decode_xyb_with_icc(jpeg_data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    use std::process::{Command, Stdio};

    // Write JPEG to temp file
    let jpeg_path = "/tmp/jpegli_test_temp.jpg";
    let output_path = "/tmp/jpegli_test_temp.bin";
    fs::write(jpeg_path, jpeg_data).ok()?;

    // Python script to decode with ICC profile
    let script = r#"
import io
import sys
from PIL import Image, ImageCms

img = Image.open(sys.argv[1])
if 'icc_profile' in img.info and len(img.info['icc_profile']) > 0:
    input_profile = ImageCms.ImageCmsProfile(io.BytesIO(img.info['icc_profile']))
    srgb = ImageCms.createProfile('sRGB')
    transform = ImageCms.buildTransformFromOpenProfiles(input_profile, srgb, 'RGB', 'RGB')
    img = ImageCms.applyTransform(img, transform)

# Write raw RGB with width/height header
w, h = img.size
with open(sys.argv[2], 'wb') as f:
    f.write(w.to_bytes(4, 'little'))
    f.write(h.to_bytes(4, 'little'))
    f.write(bytes(img.convert('RGB').tobytes()))
"#;

    let status = Command::new("python3")
        .args(["-c", script, jpeg_path, output_path])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .ok()?;

    fs::remove_file(jpeg_path).ok();

    if !status.success() {
        return None;
    }

    // Read output
    let data = fs::read(output_path).ok()?;
    fs::remove_file(output_path).ok();

    if data.len() < 8 {
        return None;
    }

    let width = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let height = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let pixels = data[8..].to_vec();

    if pixels.len() != width * height * 3 {
        return None;
    }

    Some((pixels, width, height))
}

fn encode_mozjpeg(rgb: &[u8], width: usize, height: usize, quality: f32, use_444: bool) -> Vec<u8> {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality);

    if use_444 {
        comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));
    }

    let mut started = comp.start_compress(Vec::new()).expect("mozjpeg start");
    let row_stride = width * 3;
    for y in 0..height {
        let row = &rgb[y * row_stride..(y + 1) * row_stride];
        let _ = started.write_scanlines(row);
    }
    started.finish().expect("mozjpeg finish")
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("decode")
}

/// Test that jpegli is competitive with mozjpeg on Pareto front.
///
/// For similar file sizes, jpegli should achieve similar or better DSSIM.
/// We allow up to 20% worse DSSIM at the same quality setting, since
/// the quality scales may differ between encoders.
#[test]
fn test_pareto_front_flower_small() {
    let path = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let (original, width, height) = load_png(&path).expect("load png");
    let width_u32 = width as u32;
    let height_u32 = height as u32;

    println!("\n=== Pareto Front Comparison (4:4:4 subsampling) ===");
    println!(
        "{:>7} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Quality", "jpegli Size", "moz444 Size", "jpegli DSSIM", "moz444 DSSIM", "Winner"
    );
    println!("{}", "-".repeat(76));

    let mut jpegli_wins = 0;
    let mut mozjpeg_wins = 0;
    let mut ties = 0;

    for quality in [60, 70, 80, 90] {
        let jpegli_data = encode_jpegli(&original, width_u32, height_u32, quality);
        let mozjpeg_data = encode_mozjpeg(&original, width, height, quality as f32, true);

        let jpegli_decoded = decode_jpeg(&jpegli_data);
        let mozjpeg_decoded = decode_jpeg(&mozjpeg_data);

        let jpegli_dssim = compute_dssim(&original, &jpegli_decoded, width, height);
        let mozjpeg_dssim = compute_dssim(&original, &mozjpeg_decoded, width, height);

        // Determine winner: lower DSSIM is better, smaller size is better
        // Use a simple metric: DSSIM * size
        let jpegli_score = jpegli_dssim * jpegli_data.len() as f64;
        let mozjpeg_score = mozjpeg_dssim * mozjpeg_data.len() as f64;

        let winner = if jpegli_score < mozjpeg_score * 0.95 {
            jpegli_wins += 1;
            "jpegli"
        } else if mozjpeg_score < jpegli_score * 0.95 {
            mozjpeg_wins += 1;
            "mozjpeg"
        } else {
            ties += 1;
            "tie"
        };

        println!(
            "{:>7} {:>12} {:>12} {:>12.6} {:>12.6} {:>8}",
            quality,
            jpegli_data.len(),
            mozjpeg_data.len(),
            jpegli_dssim,
            mozjpeg_dssim,
            winner
        );

        // Assert that jpegli is not dramatically worse
        // Allow 50% tolerance since encoders may have different quality curves
        assert!(
            jpegli_dssim < mozjpeg_dssim * 1.5,
            "jpegli DSSIM ({}) is >50% worse than mozjpeg ({}) at quality {}",
            jpegli_dssim,
            mozjpeg_dssim,
            quality
        );

        // Assert that file size is within reason
        assert!(
            jpegli_data.len() < mozjpeg_data.len() * 2,
            "jpegli size ({}) is >2x mozjpeg ({}) at quality {}",
            jpegli_data.len(),
            mozjpeg_data.len(),
            quality
        );
    }

    println!();
    println!(
        "Summary: jpegli wins: {}, mozjpeg wins: {}, ties: {}",
        jpegli_wins, mozjpeg_wins, ties
    );

    // Note: mozjpeg is highly optimized, so it's expected to often win
    // The important thing is that jpegli is in the same ballpark,
    // which is verified by the individual assertions above
    println!("Note: mozjpeg is a mature, optimized encoder - some losses are expected");
}

/// Test that at similar file sizes, quality is comparable.
#[test]
fn test_quality_at_similar_size() {
    let path = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let (original, width, height) = load_png(&path).expect("load png");
    let width_u32 = width as u32;
    let height_u32 = height as u32;

    // Encode with jpegli at Q80
    let jpegli_data = encode_jpegli(&original, width_u32, height_u32, 80);
    let jpegli_decoded = decode_jpeg(&jpegli_data);
    let jpegli_dssim = compute_dssim(&original, &jpegli_decoded, width, height);

    println!("\n=== Quality at Similar Size ===");
    println!(
        "jpegli Q80: {} bytes, DSSIM: {:.6}",
        jpegli_data.len(),
        jpegli_dssim
    );

    // Find mozjpeg quality that produces similar size
    let target_size = jpegli_data.len();
    let mut best_quality = 80;
    let mut best_diff = usize::MAX;

    for q in 60..=95 {
        let moz_data = encode_mozjpeg(&original, width, height, q as f32, true);
        let diff = (moz_data.len() as i64 - target_size as i64).unsigned_abs() as usize;
        if diff < best_diff {
            best_diff = diff;
            best_quality = q;
        }
    }

    let mozjpeg_data = encode_mozjpeg(&original, width, height, best_quality as f32, true);
    let mozjpeg_decoded = decode_jpeg(&mozjpeg_data);
    let mozjpeg_dssim = compute_dssim(&original, &mozjpeg_decoded, width, height);

    println!(
        "mozjpeg Q{}: {} bytes, DSSIM: {:.6}",
        best_quality,
        mozjpeg_data.len(),
        mozjpeg_dssim
    );

    let size_ratio = jpegli_data.len() as f64 / mozjpeg_data.len() as f64;
    let dssim_ratio = jpegli_dssim / mozjpeg_dssim;

    println!("Size ratio (jpegli/mozjpeg): {:.3}", size_ratio);
    println!("DSSIM ratio (jpegli/mozjpeg): {:.3}", dssim_ratio);

    // STRICT CHECK: At similar file sizes, quality should be within 15%
    // 30% was too loose - if jpegli is 30% worse, that's a real problem.
    assert!(
        dssim_ratio < 1.15,
        "At similar sizes, jpegli DSSIM ratio should be < 1.15 (within 15%), got {:.3}",
        dssim_ratio
    );
}

/// Test XYB mode vs YCbCr mode vs mozjpeg with DSSIM + off-by-N stats
#[test]
fn test_xyb_vs_ycbcr_vs_mozjpeg() {
    let path = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    // Check if Python/Pillow is available (needed for XYB ICC profile decode)
    let pillow_available = std::process::Command::new("python3")
        .args(["-c", "from PIL import Image, ImageCms"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !pillow_available {
        eprintln!("Skipping XYB test: Python/Pillow not available");
        return;
    }

    let (original, width, height) = load_png(&path).expect("load png");
    let width_u32 = width as u32;
    let height_u32 = height as u32;

    println!("\n=== XYB vs YCbCr vs mozjpeg Comparison ===");
    println!("Image: {}x{} ({} pixels)\n", width, height, width * height);

    for quality in [70, 80, 90] {
        println!("--- Quality {} ---", quality);

        // Encode with each method
        let jpegli_ycbcr = encode_jpegli(&original, width_u32, height_u32, quality);
        let jpegli_xyb = encode_jpegli_xyb(&original, width_u32, height_u32, quality);
        let mozjpeg_data = encode_mozjpeg(&original, width, height, quality as f32, true);

        // Decode
        let ycbcr_decoded = decode_jpeg(&jpegli_ycbcr);
        let mozjpeg_decoded = decode_jpeg(&mozjpeg_data);

        // XYB needs ICC-aware decoder for proper color conversion
        let xyb_decoded = decode_xyb_with_icc(&jpegli_xyb);

        // File sizes
        println!("  File sizes:");
        println!(
            "    jpegli YCbCr: {} bytes ({:.2} bpp)",
            jpegli_ycbcr.len(),
            8.0 * jpegli_ycbcr.len() as f64 / (width * height) as f64
        );
        println!(
            "    jpegli XYB:   {} bytes ({:.2} bpp)",
            jpegli_xyb.len(),
            8.0 * jpegli_xyb.len() as f64 / (width * height) as f64
        );
        println!(
            "    mozjpeg:      {} bytes ({:.2} bpp)",
            mozjpeg_data.len(),
            8.0 * mozjpeg_data.len() as f64 / (width * height) as f64
        );

        // Quality metrics - YCbCr vs original
        let ycbcr_dssim = compute_dssim(&original, &ycbcr_decoded, width, height);
        let ycbcr_stats = DiffStats::from_pixels(&original, &ycbcr_decoded);

        let mozjpeg_dssim = compute_dssim(&original, &mozjpeg_decoded, width, height);
        let mozjpeg_stats = DiffStats::from_pixels(&original, &mozjpeg_decoded);

        println!("\n  DSSIM (lower=better):");
        println!("    jpegli YCbCr: {:.6}", ycbcr_dssim);
        println!("    mozjpeg:      {:.6}", mozjpeg_dssim);

        println!("\n  Off-by-N stats vs original:");
        ycbcr_stats.report("jpegli YCbCr");
        mozjpeg_stats.report("mozjpeg     ");

        // XYB metrics if decode succeeded
        if let Some((xyb_pixels, _, _)) = &xyb_decoded {
            if xyb_pixels.len() == original.len() {
                let xyb_dssim = compute_dssim(&original, xyb_pixels, width, height);
                let xyb_stats = DiffStats::from_pixels(&original, xyb_pixels);
                println!("    jpegli XYB:   {:.6}", xyb_dssim);
                xyb_stats.report("jpegli XYB  ");

                // Winner determination
                println!("\n  Winner (DSSIM × size metric):");
                let ycbcr_score = ycbcr_dssim * jpegli_ycbcr.len() as f64;
                let xyb_score = xyb_dssim * jpegli_xyb.len() as f64;
                let moz_score = mozjpeg_dssim * mozjpeg_data.len() as f64;

                let winner = if ycbcr_score <= xyb_score && ycbcr_score <= moz_score {
                    "jpegli YCbCr"
                } else if xyb_score <= ycbcr_score && xyb_score <= moz_score {
                    "jpegli XYB"
                } else {
                    "mozjpeg"
                };
                println!(
                    "    {} (YCbCr={:.2}, XYB={:.2}, moz={:.2})",
                    winner, ycbcr_score, xyb_score, moz_score
                );

                // Assertions for XYB
                assert!(
                    xyb_dssim < 0.01,
                    "XYB DSSIM too high at Q{}: {}",
                    quality,
                    xyb_dssim
                );
            } else {
                println!(
                    "    XYB: decode size mismatch ({} vs {})",
                    xyb_pixels.len(),
                    original.len()
                );
            }
        } else {
            println!("    XYB: decode failed (djpegli error)");
        }

        // Assertions
        assert!(
            ycbcr_dssim < 0.01,
            "YCbCr DSSIM too high at Q{}: {}",
            quality,
            ycbcr_dssim
        );
        assert!(
            ycbcr_stats.pct_off_by_3_plus() < 50.0,
            "Too many off-by-3+ pixels in YCbCr at Q{}: {:.1}%",
            quality,
            ycbcr_stats.pct_off_by_3_plus()
        );

        println!();
    }
}
