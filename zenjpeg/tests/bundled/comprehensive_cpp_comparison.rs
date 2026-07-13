//! Comprehensive comparison of Rust vs C++ jpegli across corpora.
//!
//! Measures: timing, file size, DSSIM, butteraugli at distance levels.
//!
//! Uses FFI to call C++ jpegli directly (no subprocess overhead).
//!
//! IMPORTANT: Uses `jpegli_set_distance()` for C++ (not `jpeg_set_quality()`)
//! to ensure matching quant table configuration (3 tables for both).

use std::fs;
use std::path::Path;
use std::time::Instant;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};
use zenjpeg_bench_utils::{
    ChromaSubsampling as BenchChromaSubsampling, ColorMode, EncoderConfig as BenchEncoderConfig,
    EncoderImpl, ImageData, ScanMode,
};

/// Convert quality (0-100) to butteraugli distance.
/// Same formula as C++ jpegli_quality_to_distance.
fn quality_to_distance(q: f32) -> f32 {
    if q >= 100.0 {
        0.01
    } else if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
    }
}

fn encode_rust_progressive(
    width: u32,
    height: u32,
    data: &[u8],
    distance: f32,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let config = EncoderConfig::ycbcr(
        Quality::ApproxButteraugli(distance),
        ChromaSubsampling::Quarter,
    )
    .progressive(true)
    .optimize_huffman(true);
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_cpp_ffi_progressive(width: u32, height: u32, data: &[u8], distance: f32) -> Vec<u8> {
    let img = ImageData {
        name: "test".to_string(),
        pixels: data.to_vec(),
        width: width as usize,
        height: height as usize,
    };
    BenchEncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Progressive)
        .subsampling(BenchChromaSubsampling::S420)
        .distance(distance) // Use distance for 3-table parity
        .encode(&img)
        .expect("C++ jpegli FFI encode failed")
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = zenjpeg_bench_utils::load_png(path).ok()?;
    let width = img.width() as u32;
    let height = img.height() as u32;
    let rgb: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    Some((rgb, width, height))
}

fn rgb_to_imgvec(data: &[u8], width: usize, height: usize) -> imgref::ImgVec<rgb::RGB8> {
    let pixels: Vec<rgb::RGB8> = data
        .chunks_exact(3)
        .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
        .collect();
    imgref::ImgVec::new(pixels, width, height)
}

fn compute_dssim(orig: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = dssim_core::Dssim::new();

    let orig_rgba: Vec<rgb::RGBA8> = orig
        .chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let dec_rgba: Vec<rgb::RGBA8> = decoded
        .chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let dec_img = attr.create_image_rgba(&dec_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, dec_img);
    dssim.into()
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    decode_zune(data).unwrap()
}

#[allow(dead_code)] // Fields used in Debug output and extended reporting
struct ComparisonResult {
    quality: u8,
    distance: f32,
    rust_size: usize,
    cpp_size: usize,
    rust_time_ms: f64,
    cpp_time_ms: f64,
    rust_dssim: f64,
    cpp_dssim: f64,
    rust_butteraugli: f64,
    cpp_butteraugli: f64,
}

fn compare_image(rgb: &[u8], width: u32, height: u32, quality: u8) -> Option<ComparisonResult> {
    // Convert quality to distance for both encoders
    let distance = quality_to_distance(quality as f32);

    // Rust encoding with timing
    let rust_start = Instant::now();
    let rust_jpeg = encode_rust_progressive(width, height, rgb, distance).ok()?;
    let rust_time_ms = rust_start.elapsed().as_secs_f64() * 1000.0;
    let rust_size = rust_jpeg.len();

    // Decode Rust JPEG for quality metrics
    let rust_decoded = decode_jpeg(&rust_jpeg);
    let rust_dssim = compute_dssim(rgb, &rust_decoded, width as usize, height as usize);

    // Rust butteraugli
    let bfly_params = butteraugli::ButteraugliParams::default();
    let orig_img = rgb_to_imgvec(rgb, width as usize, height as usize);
    let rust_dec_img = rgb_to_imgvec(&rust_decoded, width as usize, height as usize);
    let rust_butteraugli =
        butteraugli::butteraugli(orig_img.as_ref(), rust_dec_img.as_ref(), &bfly_params)
            .expect("butteraugli")
            .score;

    // C++ encoding via FFI with timing (using same distance)
    let cpp_start = Instant::now();
    let cpp_jpeg = encode_cpp_ffi_progressive(width, height, rgb, distance);
    let cpp_time_ms = cpp_start.elapsed().as_secs_f64() * 1000.0;
    let cpp_size = cpp_jpeg.len();

    // Decode C++ JPEG for quality metrics
    let cpp_decoded = decode_jpeg(&cpp_jpeg);
    let cpp_dssim = compute_dssim(rgb, &cpp_decoded, width as usize, height as usize);

    // C++ butteraugli
    let cpp_dec_img = rgb_to_imgvec(&cpp_decoded, width as usize, height as usize);
    let cpp_butteraugli =
        butteraugli::butteraugli(orig_img.as_ref(), cpp_dec_img.as_ref(), &bfly_params)
            .expect("butteraugli")
            .score;

    Some(ComparisonResult {
        quality,
        distance,
        rust_size,
        cpp_size,
        rust_time_ms,
        cpp_time_ms,
        rust_dssim,
        cpp_dssim,
        rust_butteraugli,
        cpp_butteraugli,
    })
}

fn find_corpus_images(max_images: usize) -> Vec<std::path::PathBuf> {
    let mut images = Vec::new();

    // PRIORITY: Add frymire.png first - it has odd dimensions (1118x1105)
    // which exercises edge cases in AQ code (1118%8=6, 1105%8=1)
    let frymire_paths = [
        std::path::PathBuf::from("zenjpeg/tests/images/frymire.png"),
        std::path::PathBuf::from("tests/images/frymire.png"),
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/images/frymire.png"),
    ];
    for frymire in &frymire_paths {
        if frymire.exists() {
            println!("Adding frymire.png (1118x1105 - odd dimensions for edge case testing)");
            images.push(frymire.clone());
            break;
        }
    }

    // Try CID22-512
    if let Ok(entries) = fs::read_dir(zenjpeg_bench_utils::cid22_512_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "png") {
                images.push(path);
                if images.len() >= max_images {
                    break;
                }
            }
        }
    }

    // Try testdata
    if images.len() < max_images {
        let testdata_flower = zenjpeg::test_utils::get_testdata_dir().join("jxl/flower");
        if let Ok(entries) = fs::read_dir(&testdata_flower) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "png") {
                    images.push(path);
                    if images.len() >= max_images {
                        break;
                    }
                }
            }
        }
    }

    images
}

#[ignore = "requires external test resources"]
#[test] // Requires C++ jpegli FFI build and corpus images
fn test_comprehensive_cpp_comparison() {
    let max_images = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let images = find_corpus_images(max_images);
    assert!(
        !images.is_empty(),
        "missing test prerequisite: no corpus images found"
    );

    // Quality levels: 2, 4, 6, ..., 100
    let qualities: Vec<u8> = (1..=50).map(|i| i * 2).collect();

    println!("\n{}", "=".repeat(120));
    println!(" COMPREHENSIVE RUST vs C++ JPEGLI COMPARISON (FFI) ");
    println!("{}\n", "=".repeat(120));
    println!("Images: {}", images.len());
    println!("Quality levels: {:?}\n", qualities);

    // Aggregate results per quality level
    let mut aggregated: std::collections::HashMap<u8, Vec<ComparisonResult>> =
        std::collections::HashMap::new();

    for (img_idx, img_path) in images.iter().enumerate() {
        let img_name = img_path.file_name().unwrap().to_str().unwrap();
        println!(
            "[{}/{}] Processing: {}",
            img_idx + 1,
            images.len(),
            img_name
        );

        let (rgb, width, height) = match load_png(img_path) {
            Some(data) => data,
            None => {
                println!("  Skipping: failed to load");
                continue;
            }
        };

        for &q in &qualities {
            if let Some(result) = compare_image(&rgb, width, height, q) {
                aggregated.entry(q).or_default().push(result);
            }
        }
    }

    // Print summary table
    println!("\n{}", "=".repeat(140));
    println!(" SUMMARY (averaged across {} images) ", images.len());
    println!("{}\n", "=".repeat(140));

    println!(
        "{:>4} | {:>10} {:>10} {:>7} | {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7}",
        "Q",
        "Rust Size",
        "C++ Size",
        "Δ%",
        "Rust ms",
        "C++ ms",
        "Δ%",
        "Rust DSSIM",
        "C++ DSSIM",
        "Δ%",
        "Rust Bfly",
        "C++ Bfly",
        "Δ%"
    );
    println!("{:-<140}", "");

    let mut all_size_diffs = Vec::new();
    let mut all_dssim_diffs = Vec::new();
    let mut all_bfly_diffs = Vec::new();
    let mut all_time_diffs = Vec::new();

    for q in &qualities {
        if let Some(results) = aggregated.get(q) {
            let n = results.len() as f64;

            let avg_rust_size: f64 = results.iter().map(|r| r.rust_size as f64).sum::<f64>() / n;
            let avg_cpp_size: f64 = results.iter().map(|r| r.cpp_size as f64).sum::<f64>() / n;
            let size_diff = (avg_rust_size - avg_cpp_size) / avg_cpp_size * 100.0;

            let avg_rust_time: f64 = results.iter().map(|r| r.rust_time_ms).sum::<f64>() / n;
            let avg_cpp_time: f64 = results.iter().map(|r| r.cpp_time_ms).sum::<f64>() / n;
            let time_diff = (avg_rust_time - avg_cpp_time) / avg_cpp_time * 100.0;

            let avg_rust_dssim: f64 = results.iter().map(|r| r.rust_dssim).sum::<f64>() / n;
            let avg_cpp_dssim: f64 = results.iter().map(|r| r.cpp_dssim).sum::<f64>() / n;
            let dssim_diff = if avg_cpp_dssim > 0.0 {
                (avg_rust_dssim - avg_cpp_dssim) / avg_cpp_dssim * 100.0
            } else {
                0.0
            };

            let avg_rust_bfly: f64 = results.iter().map(|r| r.rust_butteraugli).sum::<f64>() / n;
            let avg_cpp_bfly: f64 = results.iter().map(|r| r.cpp_butteraugli).sum::<f64>() / n;
            let bfly_diff = if avg_cpp_bfly > 0.0 {
                (avg_rust_bfly - avg_cpp_bfly) / avg_cpp_bfly * 100.0
            } else {
                0.0
            };

            all_size_diffs.push(size_diff);
            all_dssim_diffs.push(dssim_diff);
            all_bfly_diffs.push(bfly_diff);
            all_time_diffs.push(time_diff);

            println!(
                "{:>4} | {:>10.0} {:>10.0} {:>+6.1}% | {:>8.2} {:>8.2} {:>+6.1}% | {:>8.6} {:>8.6} {:>+6.1}% | {:>8.4} {:>8.4} {:>+6.1}%",
                q,
                avg_rust_size,
                avg_cpp_size,
                size_diff,
                avg_rust_time,
                avg_cpp_time,
                time_diff,
                avg_rust_dssim,
                avg_cpp_dssim,
                dssim_diff,
                avg_rust_bfly,
                avg_cpp_bfly,
                bfly_diff
            );
        }
    }

    println!("{:-<140}", "");

    // Overall summary
    let avg_size_diff: f64 = all_size_diffs.iter().sum::<f64>() / all_size_diffs.len() as f64;
    let avg_dssim_diff: f64 = all_dssim_diffs.iter().sum::<f64>() / all_dssim_diffs.len() as f64;
    let avg_bfly_diff: f64 = all_bfly_diffs.iter().sum::<f64>() / all_bfly_diffs.len() as f64;
    let avg_time_diff: f64 = all_time_diffs.iter().sum::<f64>() / all_time_diffs.len() as f64;

    println!("\nOVERALL AVERAGES:");
    println!(
        "  Size difference:       {:>+.2}% (positive = Rust larger)",
        avg_size_diff
    );
    println!(
        "  Time difference:       {:>+.2}% (positive = Rust slower)",
        avg_time_diff
    );
    println!(
        "  DSSIM difference:      {:>+.2}% (positive = Rust worse)",
        avg_dssim_diff
    );
    println!(
        "  Butteraugli difference: {:>+.2}% (positive = Rust worse)",
        avg_bfly_diff
    );

    // Quality parity assessment
    println!("\nQUALITY PARITY ASSESSMENT:");
    let dssim_match_count = all_dssim_diffs.iter().filter(|d| d.abs() < 5.0).count();
    let bfly_match_count = all_bfly_diffs.iter().filter(|d| d.abs() < 5.0).count();
    println!(
        "  DSSIM within 5%: {}/{} quality levels",
        dssim_match_count,
        all_dssim_diffs.len()
    );
    println!(
        "  Butteraugli within 5%: {}/{} quality levels",
        bfly_match_count,
        all_bfly_diffs.len()
    );
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
