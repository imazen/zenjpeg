//! Corpus-wide comparison of jpegli vs mozjpeg with multiple metrics.
//!
//! Generates an HTML chart showing quality/size tradeoff across quality levels.
//! Compares: jpegli (YCbCr), jpegli (XYB), and mozjpeg (all using 4:4:4 subsampling)
//! Metrics: DSSIM and SSIMULACRA2
//!
//! XYB mode uses Python/Pillow with ICC profile for color conversion.
//!
//! Usage: cargo run --release --example corpus_comparison -- <corpus_dir> <output.html>

use dssim::Dssim;
use rgb::RGBA8;
use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

// Version for cache invalidation - bump when encoder changes
const CACHE_VERSION: &str = "v1";

fn get_cache_path(cache_dir: &Path, filename: &str, encoder: &str, quality: u8) -> PathBuf {
    cache_dir.join(format!(
        "{}_{}_q{}_{}.jpg",
        filename, encoder, quality, CACHE_VERSION
    ))
}

fn load_cached_or_encode<F>(cache_path: &Path, encode_fn: F) -> (Vec<u8>, bool)
// (data, was_cached)
where
    F: FnOnce() -> Vec<u8>,
{
    if cache_path.exists() {
        if let Ok(data) = fs::read(cache_path) {
            return (data, true);
        }
    }

    let data = encode_fn();

    // Save to cache
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(cache_path, &data);

    (data, false)
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

fn encode_jpegli(rgb: &[u8], width: u32, height: u32, quality: u8, use_xyb: bool) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality.into()))
        .use_xyb(use_xyb)
        .encode(rgb)
        .expect("jpegli encode")
}

/// Decode XYB JPEG with ICC profile applied
///
/// Uses jpegli::icc module when cms feature is enabled, falls back to Python/Pillow otherwise.
fn decode_xyb_with_icc(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    // Try using the native icc module first
    #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
    {
        if let Ok((pixels, _, _)) = jpegli::icc::decode_jpeg_with_icc(jpeg_data) {
            return Some(pixels);
        }
    }

    // Fallback to Python/Pillow
    use std::process::{Command, Stdio};

    let jpeg_path = "/tmp/corpus_cmp_xyb.jpg";
    let output_path = "/tmp/corpus_cmp_xyb.bin";
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

    fs::remove_file(jpeg_path).ok();
    if !status.success() {
        return None;
    }

    let data = fs::read(output_path).ok()?;
    fs::remove_file(output_path).ok();
    Some(data)
}

fn encode_mozjpeg(rgb: &[u8], width: usize, height: usize, quality: f32) -> Vec<u8> {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality);
    comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));

    let mut started = comp.start_compress(Vec::new()).expect("mozjpeg start");
    let row_stride = width * 3;
    for y in 0..height {
        let row = &rgb[y * row_stride..(y + 1) * row_stride];
        let _ = started.write_scanlines(row);
    }
    started.finish().expect("mozjpeg finish")
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
    decoder.decode().expect("decode")
}

#[derive(Debug, Clone)]
struct EncoderResult {
    name: String,
    color: String,
    points: Vec<DataPoint>,
}

#[derive(Debug, Clone)]
struct DataPoint {
    quality: u8,
    bpp: f64,
    dssim: f64,
    ssim2: f64,
}

#[derive(Debug, Clone)]
struct PerImageResult {
    filename: String,
    // jpegli (YCbCr)
    jpegli_dssim: f64,
    jpegli_ssim2: f64,
    jpegli_bpp: f64,
    // jpegli XYB
    xyb_dssim: f64,
    xyb_ssim2: f64,
    xyb_bpp: f64,
    // mozjpeg
    moz_dssim: f64,
    moz_ssim2: f64,
    moz_bpp: f64,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <corpus_dir> <output.html>", args[0]);
        eprintln!(
            "Example: {} /mnt/v/work/corpus/CID22-512 comparison.html",
            args[0]
        );
        std::process::exit(1);
    }

    let corpus_dir = &args[1];
    let output_path = &args[2];

    // Cache directory (alongside output)
    let cache_dir = Path::new(output_path)
        .parent()
        .unwrap_or(Path::new("."))
        .join("jpeg_cache");
    let use_cache = env::var("NO_CACHE").is_err();
    if use_cache {
        println!(
            "Using cache at {} (set NO_CACHE=1 to disable, CACHE_VERSION={})",
            cache_dir.display(),
            CACHE_VERSION
        );
    }

    let mut files: Vec<_> = fs::read_dir(corpus_dir)
        .expect("Failed to read corpus directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext.to_ascii_lowercase() == "png")
                .unwrap_or(false)
        })
        .collect();

    files.sort_by_key(|e| e.path());

    let max_files: usize = env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    if max_files > 0 && files.len() > max_files {
        println!("Limiting to {} files (set MAX_FILES=0 for all)", max_files);
        files.truncate(max_files);
    }

    println!("Processing {} PNG files...", files.len());

    // Include low-Q values for hypothesis testing (5-step increments below Q60)
    let quality_levels = [
        10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 85, 90, 95,
    ];

    // Store per-image results for analysis
    let mut low_q_results: Vec<PerImageResult> = Vec::new();

    let mut jpegli_ycbcr = EncoderResult {
        name: "jpegli".to_string(),
        color: "#2196F3".to_string(),
        points: Vec::new(),
    };
    let mut jpegli_xyb = EncoderResult {
        name: "jpegli-XYB".to_string(),
        color: "#9C27B0".to_string(),
        points: Vec::new(),
    };
    let mut mozjpeg = EncoderResult {
        name: "mozjpeg".to_string(),
        color: "#4CAF50".to_string(),
        points: Vec::new(),
    };

    for &quality in &quality_levels {
        print!("Q{}: ", quality);
        std::io::stdout().flush().unwrap();

        let mut totals = [(0usize, 0.0f64, 0.0f64); 3]; // (size, dssim*pixels, ssim2*pixels)
        let mut total_pixels = 0usize;

        for entry in &files {
            let path = entry.path();
            if let Some((rgb, width, height)) = load_png(&path) {
                let pixels = width * height;
                let filename = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                // Encode with all three (with caching)
                let (jpegli_data, _cached1) = if use_cache {
                    let cache_path = get_cache_path(&cache_dir, &filename, "jpegli", quality);
                    load_cached_or_encode(&cache_path, || {
                        encode_jpegli(&rgb, width as u32, height as u32, quality, false)
                    })
                } else {
                    (
                        encode_jpegli(&rgb, width as u32, height as u32, quality, false),
                        false,
                    )
                };

                let (xyb_data, _cached2) = if use_cache {
                    let cache_path = get_cache_path(&cache_dir, &filename, "xyb", quality);
                    load_cached_or_encode(&cache_path, || {
                        encode_jpegli(&rgb, width as u32, height as u32, quality, true)
                    })
                } else {
                    (
                        encode_jpegli(&rgb, width as u32, height as u32, quality, true),
                        false,
                    )
                };

                let (moz_data, _cached3) = if use_cache {
                    let cache_path = get_cache_path(&cache_dir, &filename, "mozjpeg", quality);
                    load_cached_or_encode(&cache_path, || {
                        encode_mozjpeg(&rgb, width, height, quality as f32)
                    })
                } else {
                    (encode_mozjpeg(&rgb, width, height, quality as f32), false)
                };

                // Decode
                let jpegli_dec = decode_jpeg(&jpegli_data);
                // XYB needs ICC-aware decoder for proper color conversion
                let xyb_dec =
                    decode_xyb_with_icc(&xyb_data).unwrap_or_else(|| decode_jpeg(&xyb_data));
                let moz_dec = decode_jpeg(&moz_data);

                // Compute metrics
                let jpegli_dssim = compute_dssim(&rgb, &jpegli_dec, width, height);
                let jpegli_ssim2 = compute_ssim2(&rgb, &jpegli_dec, width, height);

                let xyb_dssim = compute_dssim(&rgb, &xyb_dec, width, height);
                let xyb_ssim2 = compute_ssim2(&rgb, &xyb_dec, width, height);

                let moz_dssim = compute_dssim(&rgb, &moz_dec, width, height);
                let moz_ssim2 = compute_ssim2(&rgb, &moz_dec, width, height);

                totals[0].0 += jpegli_data.len();
                totals[0].1 += jpegli_dssim * pixels as f64;
                totals[0].2 += jpegli_ssim2 * pixels as f64;

                totals[1].0 += xyb_data.len();
                totals[1].1 += xyb_dssim * pixels as f64;
                totals[1].2 += xyb_ssim2 * pixels as f64;

                totals[2].0 += moz_data.len();
                totals[2].1 += moz_dssim * pixels as f64;
                totals[2].2 += moz_ssim2 * pixels as f64;

                total_pixels += pixels;

                // Collect per-image results at Q30 for low-Q analysis
                if quality == 30 {
                    let jpegli_bpp_img = jpegli_data.len() as f64 / pixels as f64 * 8.0;
                    let xyb_bpp_img = xyb_data.len() as f64 / pixels as f64 * 8.0;
                    let moz_bpp_img = moz_data.len() as f64 / pixels as f64 * 8.0;
                    low_q_results.push(PerImageResult {
                        filename: path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string(),
                        jpegli_dssim,
                        jpegli_ssim2,
                        jpegli_bpp: jpegli_bpp_img,
                        xyb_dssim,
                        xyb_ssim2,
                        xyb_bpp: xyb_bpp_img,
                        moz_dssim,
                        moz_ssim2,
                        moz_bpp: moz_bpp_img,
                    });
                }
            }
        }

        if total_pixels > 0 {
            let tp = total_pixels as f64;

            jpegli_ycbcr.points.push(DataPoint {
                quality,
                bpp: totals[0].0 as f64 / tp * 8.0,
                dssim: totals[0].1 / tp,
                ssim2: totals[0].2 / tp,
            });

            jpegli_xyb.points.push(DataPoint {
                quality,
                bpp: totals[1].0 as f64 / tp * 8.0,
                dssim: totals[1].1 / tp,
                ssim2: totals[1].2 / tp,
            });

            mozjpeg.points.push(DataPoint {
                quality,
                bpp: totals[2].0 as f64 / tp * 8.0,
                dssim: totals[2].1 / tp,
                ssim2: totals[2].2 / tp,
            });

            let jp = jpegli_ycbcr.points.last().unwrap();
            let xyb = jpegli_xyb.points.last().unwrap();
            let mz = mozjpeg.points.last().unwrap();
            println!(
                "jpegli {:.2}bpp/{:.4}dssim/{:.1}ssim2 | XYB {:.2}bpp/{:.4}dssim/{:.1}ssim2 | moz {:.2}bpp/{:.4}dssim/{:.1}ssim2",
                jp.bpp, jp.dssim, jp.ssim2,
                xyb.bpp, xyb.dssim, xyb.ssim2,
                mz.bpp, mz.dssim, mz.ssim2
            );
        }
    }

    let encoders = vec![jpegli_ycbcr, jpegli_xyb, mozjpeg];

    // Print low-Q analysis
    println!("\n=== Low-Q (Q30) Per-Image Analysis ===");
    println!("Comparing jpegli, XYB, and mozjpeg at Q30");

    if !low_q_results.is_empty() {
        // Calculate advantages
        let calc_dssim_adv_jpegli = |r: &PerImageResult| r.moz_dssim - r.jpegli_dssim;
        let calc_ssim2_adv_jpegli = |r: &PerImageResult| r.jpegli_ssim2 - r.moz_ssim2;
        let calc_dssim_adv_xyb = |r: &PerImageResult| r.moz_dssim - r.xyb_dssim;
        let calc_ssim2_adv_xyb = |r: &PerImageResult| r.xyb_ssim2 - r.moz_ssim2;

        // Sort by SSIM2 advantage (more perceptually accurate)
        let mut by_ssim2: Vec<_> = low_q_results.iter().collect();
        by_ssim2.sort_by(|a, b| {
            calc_ssim2_adv_jpegli(b)
                .partial_cmp(&calc_ssim2_adv_jpegli(a))
                .unwrap()
        });

        println!("\n--- SSIM2 Analysis (jpegli vs mozjpeg) ---");
        println!(
            "{:>35} {:>10} {:>10} {:>10} {:>10}",
            "Filename", "jpegli", "XYB", "mozjpeg", "Winner"
        );

        println!("\nTop 5 where jpegli beats mozjpeg (SSIM2):");
        for r in by_ssim2.iter().take(5) {
            let winner = if r.jpegli_ssim2 > r.moz_ssim2 + 0.5 {
                "jpegli"
            } else if r.moz_ssim2 > r.jpegli_ssim2 + 0.5 {
                "mozjpeg"
            } else {
                "~tie"
            };
            println!(
                "{:>35} {:>10.2} {:>10.2} {:>10.2} {:>10}",
                &r.filename[..r.filename.len().min(35)],
                r.jpegli_ssim2,
                r.xyb_ssim2,
                r.moz_ssim2,
                winner
            );
        }

        println!("\nTop 5 where mozjpeg beats jpegli (SSIM2):");
        for r in by_ssim2.iter().rev().take(5) {
            let winner = if r.jpegli_ssim2 > r.moz_ssim2 + 0.5 {
                "jpegli"
            } else if r.moz_ssim2 > r.jpegli_ssim2 + 0.5 {
                "mozjpeg"
            } else {
                "~tie"
            };
            println!(
                "{:>35} {:>10.2} {:>10.2} {:>10.2} {:>10}",
                &r.filename[..r.filename.len().min(35)],
                r.jpegli_ssim2,
                r.xyb_ssim2,
                r.moz_ssim2,
                winner
            );
        }

        // XYB vs mozjpeg analysis
        println!("\n--- XYB vs mozjpeg Analysis ---");
        let mut by_xyb_ssim2: Vec<_> = low_q_results.iter().collect();
        by_xyb_ssim2.sort_by(|a, b| {
            calc_ssim2_adv_xyb(b)
                .partial_cmp(&calc_ssim2_adv_xyb(a))
                .unwrap()
        });

        println!("\nTop 5 where XYB beats mozjpeg (SSIM2):");
        for r in by_xyb_ssim2.iter().take(5) {
            let winner = if r.xyb_ssim2 > r.moz_ssim2 + 0.5 {
                "XYB"
            } else if r.moz_ssim2 > r.xyb_ssim2 + 0.5 {
                "mozjpeg"
            } else {
                "~tie"
            };
            println!(
                "{:>35} {:>10.2} {:>10.2} {:>10.2} {:>10}",
                &r.filename[..r.filename.len().min(35)],
                r.jpegli_ssim2,
                r.xyb_ssim2,
                r.moz_ssim2,
                winner
            );
        }

        // Summary statistics
        println!("\n=== Summary at Q30 ===");

        // DSSIM stats
        let jpegli_dssim_wins = low_q_results
            .iter()
            .filter(|r| calc_dssim_adv_jpegli(r) > 0.0001)
            .count();
        let xyb_dssim_wins = low_q_results
            .iter()
            .filter(|r| calc_dssim_adv_xyb(r) > 0.0001)
            .count();

        // SSIM2 stats
        let jpegli_ssim2_wins = low_q_results
            .iter()
            .filter(|r| calc_ssim2_adv_jpegli(r) > 0.5)
            .count();
        let xyb_ssim2_wins = low_q_results
            .iter()
            .filter(|r| calc_ssim2_adv_xyb(r) > 0.5)
            .count();
        let moz_ssim2_wins_vs_jpegli = low_q_results
            .iter()
            .filter(|r| calc_ssim2_adv_jpegli(r) < -0.5)
            .count();
        let moz_ssim2_wins_vs_xyb = low_q_results
            .iter()
            .filter(|r| calc_ssim2_adv_xyb(r) < -0.5)
            .count();

        let n = low_q_results.len();
        println!("DSSIM (lower = better):");
        println!(
            "  jpegli beats moz: {} ({:.1}%)",
            jpegli_dssim_wins,
            100.0 * jpegli_dssim_wins as f64 / n as f64
        );
        println!(
            "  XYB beats moz:    {} ({:.1}%)",
            xyb_dssim_wins,
            100.0 * xyb_dssim_wins as f64 / n as f64
        );

        println!("\nSSIMULACRA2 (higher = better, threshold 0.5):");
        println!(
            "  jpegli > moz: {} ({:.1}%)",
            jpegli_ssim2_wins,
            100.0 * jpegli_ssim2_wins as f64 / n as f64
        );
        println!(
            "  moz > jpegli: {} ({:.1}%)",
            moz_ssim2_wins_vs_jpegli,
            100.0 * moz_ssim2_wins_vs_jpegli as f64 / n as f64
        );
        println!(
            "  XYB > moz:    {} ({:.1}%)",
            xyb_ssim2_wins,
            100.0 * xyb_ssim2_wins as f64 / n as f64
        );
        println!(
            "  moz > XYB:    {} ({:.1}%)",
            moz_ssim2_wins_vs_xyb,
            100.0 * moz_ssim2_wins_vs_xyb as f64 / n as f64
        );

        // Averages
        let avg_jpegli_ssim2: f64 =
            low_q_results.iter().map(|r| r.jpegli_ssim2).sum::<f64>() / n as f64;
        let avg_xyb_ssim2: f64 = low_q_results.iter().map(|r| r.xyb_ssim2).sum::<f64>() / n as f64;
        let avg_moz_ssim2: f64 = low_q_results.iter().map(|r| r.moz_ssim2).sum::<f64>() / n as f64;

        let avg_jpegli_bpp: f64 =
            low_q_results.iter().map(|r| r.jpegli_bpp).sum::<f64>() / n as f64;
        let avg_xyb_bpp: f64 = low_q_results.iter().map(|r| r.xyb_bpp).sum::<f64>() / n as f64;
        let avg_moz_bpp: f64 = low_q_results.iter().map(|r| r.moz_bpp).sum::<f64>() / n as f64;

        println!("\nAverages:");
        println!(
            "  SSIM2: jpegli={:.2}, XYB={:.2}, mozjpeg={:.2}",
            avg_jpegli_ssim2, avg_xyb_ssim2, avg_moz_ssim2
        );
        println!(
            "  bpp:   jpegli={:.3}, XYB={:.3}, mozjpeg={:.3}",
            avg_jpegli_bpp, avg_xyb_bpp, avg_moz_bpp
        );
    }

    generate_html_chart(&encoders, &low_q_results, output_path);
    println!("\nChart saved to: {}", output_path);
}

fn generate_html_chart(
    encoders: &[EncoderResult],
    low_q_results: &[PerImageResult],
    output_path: &str,
) {
    // Generate two charts: DSSIM and SSIMULACRA2
    let dssim_svg = generate_svg_chart(encoders, "dssim", "DSSIM (lower = better)", true);
    let ssim2_svg = generate_svg_chart(encoders, "ssim2", "SSIMULACRA2 (higher = better)", false);

    let table_rows = generate_table_rows(encoders);
    let low_q_table = generate_low_q_table(low_q_results);

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>jpegli vs mozjpeg Comparison</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .charts {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .chart {{ flex: 1; min-width: 400px; }}
        table {{ border-collapse: collapse; margin-top: 20px; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
        th {{ background: #f0f0f0; }}
        td:first-child {{ text-align: left; }}
        .better {{ background: #e8f5e9; }}
        .worse {{ background: #ffebee; }}
        .note {{ color: #666; font-size: 14px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>jpegli-rs vs mozjpeg Quality/Size Comparison</h1>
        <p>Compares jpegli (YCbCr), jpegli (XYB quant tables), and mozjpeg. All use 4:4:4 subsampling.</p>
        <div class="charts">
            <div class="chart">
                <h3>DSSIM Metric (lower-left is better)</h3>
                {}
            </div>
            <div class="chart">
                <h3>SSIMULACRA2 Metric (upper-left is better)</h3>
                {}
            </div>
        </div>
        <h2>Data Table</h2>
        <table>
            <tr>
                <th>Quality</th>
                <th>Encoder</th>
                <th>bpp</th>
                <th>DSSIM</th>
                <th>SSIMULACRA2</th>
            </tr>
{}
        </table>
        <h2>Low-Q (Q30) Per-Image Analysis</h2>
        <p>Identifies images where each encoder excels.</p>
        <table>
            <tr>
                <th>Filename</th>
                <th>jpegli bpp</th>
                <th>jpegli SSIM2</th>
                <th>XYB bpp</th>
                <th>XYB SSIM2</th>
                <th>moz bpp</th>
                <th>moz SSIM2</th>
            </tr>
{}
        </table>
        <p class="note">
            <strong>bpp</strong> = bits per pixel (lower = smaller file)<br>
            <strong>DSSIM</strong> = structural dissimilarity (lower = better, 0 = identical)<br>
            <strong>SSIMULACRA2</strong> = perceptual quality (higher = better, 100 = identical)<br>
            <strong>XYB</strong> = jpegli with XYB-optimized quantization tables<br>
            Green cells = best performer for that image
        </p>
    </div>
</body>
</html>"#,
        dssim_svg, ssim2_svg, table_rows, low_q_table
    );

    fs::write(output_path, html).expect("Failed to write HTML");
}

fn generate_low_q_table(results: &[PerImageResult]) -> String {
    let mut rows = Vec::new();

    for r in results {
        // Find best SSIM2
        let best_ssim2 = r.jpegli_ssim2.max(r.xyb_ssim2).max(r.moz_ssim2);
        let jpegli_best = (r.jpegli_ssim2 - best_ssim2).abs() < 0.1;
        let xyb_best = (r.xyb_ssim2 - best_ssim2).abs() < 0.1;
        let moz_best = (r.moz_ssim2 - best_ssim2).abs() < 0.1;

        let jpegli_class = if jpegli_best { " class=\"better\"" } else { "" };
        let xyb_class = if xyb_best { " class=\"better\"" } else { "" };
        let moz_class = if moz_best { " class=\"better\"" } else { "" };

        rows.push(format!(
            "            <tr><td>{}</td><td>{:.3}</td><td{}>{:.2}</td><td>{:.3}</td><td{}>{:.2}</td><td>{:.3}</td><td{}>{:.2}</td></tr>",
            r.filename,
            r.jpegli_bpp, jpegli_class, r.jpegli_ssim2,
            r.xyb_bpp, xyb_class, r.xyb_ssim2,
            r.moz_bpp, moz_class, r.moz_ssim2
        ));
    }

    rows.join("\n")
}

fn generate_svg_chart(
    encoders: &[EncoderResult],
    metric: &str,
    y_label: &str,
    lower_better: bool,
) -> String {
    let width = 450.0;
    let height = 350.0;
    let margin = 55.0;
    let plot_width = width - 2.0 * margin;
    let plot_height = height - 2.0 * margin;

    // Find ranges
    let min_bpp = encoders
        .iter()
        .flat_map(|e| e.points.iter().map(|p| p.bpp))
        .fold(f64::INFINITY, f64::min);
    let max_bpp = encoders
        .iter()
        .flat_map(|e| e.points.iter().map(|p| p.bpp))
        .fold(0.0, f64::max);

    let get_metric = |p: &DataPoint| if metric == "dssim" { p.dssim } else { p.ssim2 };

    let min_m = encoders
        .iter()
        .flat_map(|e| e.points.iter().map(|p| get_metric(p)))
        .fold(f64::INFINITY, f64::min);
    let max_m = encoders
        .iter()
        .flat_map(|e| e.points.iter().map(|p| get_metric(p)))
        .fold(0.0, f64::max);

    let bpp_range = max_bpp - min_bpp;
    let m_range = max_m - min_m;
    let min_bpp = min_bpp - bpp_range * 0.1;
    let max_bpp = max_bpp + bpp_range * 0.1;
    let min_m = (min_m - m_range * 0.1).max(0.0);
    let max_m = max_m + m_range * 0.1;

    let scale_x = |bpp: f64| margin + (bpp - min_bpp) / (max_bpp - min_bpp) * plot_width;
    let scale_y = |m: f64| {
        if lower_better {
            margin + plot_height - (m - min_m) / (max_m - min_m) * plot_height
        } else {
            margin + (max_m - m) / (max_m - min_m) * plot_height
        }
    };

    let mut svg = format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis {{ stroke: #333; stroke-width: 1; }}
    .grid {{ stroke: #eee; stroke-width: 0.5; }}
    .label {{ font-family: sans-serif; font-size: 10px; }}
    .legend {{ font-family: sans-serif; font-size: 10px; }}
  </style>
"#,
        width, height
    );

    // Axes
    svg.push_str(&format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>
"#,
        margin,
        margin,
        margin,
        height - margin,
        margin,
        height - margin,
        width - margin,
        height - margin
    ));

    // Axis labels
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" class="label" text-anchor="middle">bpp</text>
  <text x="12" y="{}" class="label" text-anchor="middle" transform="rotate(-90, 12, {})">{}</text>
"#,
        width / 2.0,
        height - 8.0,
        height / 2.0,
        height / 2.0,
        y_label
    ));

    // Grid and ticks
    for i in 0..=4 {
        let x = margin + plot_width * i as f64 / 4.0;
        let y = margin + plot_height * i as f64 / 4.0;
        svg.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
"#,
            x,
            margin,
            x,
            height - margin,
            margin,
            y,
            width - margin,
            y
        ));

        let bpp = min_bpp + (max_bpp - min_bpp) * i as f64 / 4.0;
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="middle">{:.2}</text>
"#,
            x,
            height - margin + 12.0,
            bpp
        ));

        let m_val = if lower_better {
            max_m - (max_m - min_m) * i as f64 / 4.0
        } else {
            min_m + (max_m - min_m) * i as f64 / 4.0
        };
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="end">{:.3}</text>
"#,
            margin - 4.0,
            y + 3.0,
            m_val
        ));
    }

    // Draw each encoder
    for encoder in encoders {
        if encoder.points.is_empty() {
            continue;
        }

        let mut path = String::new();
        for (i, p) in encoder.points.iter().enumerate() {
            let x = scale_x(p.bpp);
            let y = scale_y(get_metric(p));
            if i == 0 {
                path.push_str(&format!("M {} {}", x, y));
            } else {
                path.push_str(&format!(" L {} {}", x, y));
            }
        }
        svg.push_str(&format!(
            r#"  <path d="{}" stroke="{}" fill="none" stroke-width="2"/>
"#,
            path, encoder.color
        ));

        for p in &encoder.points {
            let x = scale_x(p.bpp);
            let y = scale_y(get_metric(p));
            svg.push_str(&format!(
                r#"  <circle cx="{}" cy="{}" r="3" fill="{}"/>
"#,
                x, y, encoder.color
            ));
        }
    }

    // Legend
    let legend_x = width - 90.0;
    let legend_y = 10.0;
    svg.push_str(&format!(
        r##"  <rect x="{}" y="{}" width="85" height="{}" fill="white" stroke="#ccc" rx="3"/>
"##,
        legend_x,
        legend_y,
        15.0 * encoders.len() as f64 + 8.0
    ));

    for (i, encoder) in encoders.iter().enumerate() {
        let y = legend_y + 14.0 + 15.0 * i as f64;
        svg.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>
  <circle cx="{}" cy="{}" r="3" fill="{}"/>
  <text x="{}" y="{}" class="legend">{}</text>
"#,
            legend_x + 5.0,
            y,
            legend_x + 20.0,
            y,
            encoder.color,
            legend_x + 12.5,
            y,
            encoder.color,
            legend_x + 25.0,
            y + 3.0,
            encoder.name
        ));
    }

    svg.push_str("</svg>");
    svg
}

fn generate_table_rows(encoders: &[EncoderResult]) -> String {
    let mut rows = Vec::new();

    if let Some(first) = encoders.first() {
        for (i, p) in first.points.iter().enumerate() {
            for encoder in encoders {
                if let Some(ep) = encoder.points.get(i) {
                    rows.push(format!(
                        "            <tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.6}</td><td>{:.2}</td></tr>",
                        ep.quality, encoder.name, ep.bpp, ep.dssim, ep.ssim2
                    ));
                }
            }
        }
    }

    rows.join("\n")
}
