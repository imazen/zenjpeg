//! Comprehensive comparison of jpegli-rs vs C++ jpegli across all encoding modes.
//!
//! Tests: YCbCr, XYB, Grayscale × Sequential/Progressive × Subsampling × Quality
//! Metrics: File size, SSIMULACRA2
//!
//! Run with:
//! ```
//! cargo run --release --example comprehensive_mode_comparison
//! ```
//!
//! Environment variables:
//!   MAX_IMAGES=N     - Limit images to test (default: 20)
//!   CORPUS_DIR=/path - Override corpus directory

use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_id() -> u64 {
    COUNTER.fetch_add(1, Ordering::SeqCst)
}

// Encoding mode configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct EncodingMode {
    color_mode: ColorMode,
    progressive: bool,
    subsampling: SubsamplingMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ColorMode {
    YCbCr,
    XYB,
    Grayscale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum SubsamplingMode {
    S444,
    S422,
    S420,
    S440,
}

impl std::fmt::Display for EncodingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let color = match self.color_mode {
            ColorMode::YCbCr => "YCbCr",
            ColorMode::XYB => "XYB",
            ColorMode::Grayscale => "Gray",
        };
        let mode = if self.progressive { "prog" } else { "seq" };
        let sub = match self.subsampling {
            SubsamplingMode::S444 => "444",
            SubsamplingMode::S422 => "422",
            SubsamplingMode::S420 => "420",
            SubsamplingMode::S440 => "440",
        };
        write!(f, "{}/{}/{}", color, mode, sub)
    }
}

impl EncodingMode {
    fn is_valid(&self) -> bool {
        // XYB only supports 4:4:4
        if self.color_mode == ColorMode::XYB && self.subsampling != SubsamplingMode::S444 {
            return false;
        }
        // Grayscale ignores subsampling, only test 4:4:4
        if self.color_mode == ColorMode::Grayscale && self.subsampling != SubsamplingMode::S444 {
            return false;
        }
        true
    }

    fn all_modes() -> Vec<Self> {
        let mut modes = Vec::new();
        for &color_mode in &[ColorMode::YCbCr, ColorMode::XYB, ColorMode::Grayscale] {
            for &progressive in &[false, true] {
                for &subsampling in &[
                    SubsamplingMode::S444,
                    SubsamplingMode::S422,
                    SubsamplingMode::S420,
                    SubsamplingMode::S440,
                ] {
                    let mode = EncodingMode {
                        color_mode,
                        progressive,
                        subsampling,
                    };
                    if mode.is_valid() {
                        modes.push(mode);
                    }
                }
            }
        }
        modes
    }

    fn to_rust_subsampling(&self) -> Subsampling {
        match self.subsampling {
            SubsamplingMode::S444 => Subsampling::S444,
            SubsamplingMode::S422 => Subsampling::S422,
            SubsamplingMode::S420 => Subsampling::S420,
            SubsamplingMode::S440 => Subsampling::S440,
        }
    }

    fn cpp_args(&self, quality: u8) -> Vec<String> {
        let mut args = vec!["-q".to_string(), quality.to_string()];

        // Progressive level
        if self.progressive {
            args.push("--progressive_level=2".to_string());
        } else {
            args.push("--progressive_level=0".to_string());
        }

        // Subsampling
        let sub = match self.subsampling {
            SubsamplingMode::S444 => "444",
            SubsamplingMode::S422 => "422",
            SubsamplingMode::S420 => "420",
            SubsamplingMode::S440 => "440",
        };
        args.push(format!("--chroma_subsampling={}", sub));

        // XYB mode
        if self.color_mode == ColorMode::XYB {
            args.push("--xyb".to_string());
        }

        args
    }
}

struct ComparisonResult {
    rust_size: usize,
    cpp_size: usize,
    rust_ssim2: f64,
    cpp_ssim2: f64,
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let width = info.width;
    let height = info.height;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn rgb_to_gray(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks(3)
        .map(|c| (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32).round() as u8)
        .collect()
}

fn gray_to_rgb(gray: &[u8]) -> Vec<u8> {
    gray.iter().flat_map(|&g| [g, g, g]).collect()
}

fn compute_ssim2(orig_rgb: &[u8], decoded_rgb: &[u8], width: usize, height: usize) -> f64 {
    let expected_len = width * height * 3;
    if orig_rgb.len() != expected_len {
        eprintln!(
            "SSIM ERROR: orig_rgb.len()={} expected={}",
            orig_rgb.len(),
            expected_len
        );
        return -999.0;
    }
    if decoded_rgb.len() != expected_len {
        eprintln!(
            "SSIM ERROR: decoded_rgb.len()={} expected={}",
            decoded_rgb.len(),
            expected_len
        );
        return -999.0;
    }

    let orig = Rgb::new(
        orig_rgb
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

    let dec = Rgb::new(
        decoded_rgb
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

    compute_frame_ssimulacra2(orig, dec).unwrap_or(-1.0)
}

fn decode_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    // Use jpegli's ICC-aware decoder which handles XYB profile transformation
    let (pixels, _width, _height) = jpegli::icc::decode_jpeg_with_icc(data).ok()?;
    Some(pixels)
}

fn decode_jpeg_simple(data: &[u8]) -> Option<Vec<u8>> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder.decode().ok()?;

    // Get actual output colorspace - zune-jpeg may override our request for grayscale images
    let colorspace = decoder.output_colorspace()?;
    match colorspace {
        ColorSpace::Luma => {
            // Convert grayscale to RGB
            Some(pixels.iter().flat_map(|&g| [g, g, g]).collect())
        }
        ColorSpace::RGB => Some(pixels),
        _ => {
            // For other colorspaces (YCbCr, etc), zune should have converted to RGB
            Some(pixels)
        }
    }
}

fn encode_rust(
    rgb: &[u8],
    width: u32,
    height: u32,
    mode: &EncodingMode,
    quality: u8,
) -> Option<Vec<u8>> {
    let (pixels, pixel_format) = if mode.color_mode == ColorMode::Grayscale {
        (rgb_to_gray(rgb), PixelFormat::Gray)
    } else {
        (rgb.to_vec(), PixelFormat::Rgb)
    };

    let jpeg_mode = if mode.progressive {
        JpegMode::Progressive
    } else {
        JpegMode::Baseline
    };

    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(pixel_format)
        .mode(jpeg_mode)
        .subsampling(mode.to_rust_subsampling())
        .use_xyb(mode.color_mode == ColorMode::XYB)
        .optimize_huffman(true)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .encode(&pixels)
        .ok()
}

fn encode_cpp(
    png_path: &Path,
    mode: &EncodingMode,
    quality: u8,
    cjpegli: &Path,
) -> Option<Vec<u8>> {
    let out_path = format!("/tmp/cpp_{}_{}.jpg", std::process::id(), unique_id());

    let mut args = mode.cpp_args(quality);
    args.insert(0, png_path.to_str()?.to_string());
    args.insert(1, out_path.clone());

    let status = Command::new(cjpegli).args(&args).output().ok()?;

    if !status.status.success() {
        return None;
    }

    let data = fs::read(&out_path).ok();
    let _ = fs::remove_file(&out_path);
    data
}

fn find_cjpegli() -> Option<PathBuf> {
    let paths = [
        "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli",
        "../internal/jpegli-cpp/build/tools/cjpegli",
    ];
    for p in paths {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn find_corpus() -> Vec<PathBuf> {
    let max_images: usize = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let mut images = Vec::new();

    // Try user-specified directory
    if let Ok(dir) = std::env::var("CORPUS_DIR") {
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "png") {
                    images.push(path);
                    if images.len() >= max_images {
                        return images;
                    }
                }
            }
        }
    }

    // Try CID22-512
    if let Ok(entries) = fs::read_dir("/mnt/v/work/corpus/CID22-512") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "png") {
                images.push(path);
                if images.len() >= max_images {
                    return images;
                }
            }
        }
    }

    // Try testdata
    if images.len() < max_images {
        let testdata = PathBuf::from("/home/lilith/work/jpegli-rs/testdata/jxl/flower");
        if let Ok(entries) = fs::read_dir(&testdata) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "png") {
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

fn compare_mode(
    rgb: &[u8],
    width: u32,
    height: u32,
    png_path: &Path,
    mode: &EncodingMode,
    quality: u8,
    cjpegli: &Path,
) -> Option<ComparisonResult> {
    // Encode with Rust
    let rust_jpeg = encode_rust(rgb, width, height, mode, quality)?;
    let rust_size = rust_jpeg.len();

    // Decode and compute SSIM2
    // decode_jpeg already returns RGB (handles grayscale-to-RGB conversion internally)
    let rust_decoded_rgb = decode_jpeg(&rust_jpeg)?;
    let rust_ssim2 = compute_ssim2(rgb, &rust_decoded_rgb, width as usize, height as usize);

    // Encode with C++
    // For grayscale, we need to prepare a grayscale PNG
    let cpp_result = if mode.color_mode == ColorMode::Grayscale {
        // Create temporary grayscale PNG
        let gray_png_path = format!("/tmp/gray_{}_{}.png", std::process::id(), unique_id());
        let gray_data = rgb_to_gray(rgb);
        {
            let file = fs::File::create(&gray_png_path).ok()?;
            let mut encoder = png::Encoder::new(file, width, height);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().ok()?;
            writer.write_image_data(&gray_data).ok()?;
        }
        let result = encode_cpp(Path::new(&gray_png_path), mode, quality, cjpegli);
        let _ = fs::remove_file(&gray_png_path);
        result
    } else {
        encode_cpp(png_path, mode, quality, cjpegli)
    };

    let cpp_jpeg = cpp_result?;
    let cpp_size = cpp_jpeg.len();

    // decode_jpeg already returns RGB (handles grayscale-to-RGB conversion internally)
    let cpp_decoded_rgb = decode_jpeg(&cpp_jpeg)?;
    let cpp_ssim2 = compute_ssim2(rgb, &cpp_decoded_rgb, width as usize, height as usize);

    Some(ComparisonResult {
        rust_size,
        cpp_size,
        rust_ssim2,
        cpp_ssim2,
    })
}

fn main() {
    println!("\n{}", "=".repeat(120));
    println!(" COMPREHENSIVE jpegli-rs vs C++ COMPARISON");
    println!(" All Encoding Modes × Quality Levels × SSIMULACRA2 + File Size");
    println!("{}\n", "=".repeat(120));

    let cjpegli = match find_cjpegli() {
        Some(p) => p,
        None => {
            eprintln!("ERROR: cjpegli not found. Build it first:");
            eprintln!("  cd internal/jpegli-cpp && mkdir -p build && cd build");
            eprintln!("  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_TOOLS=ON ..");
            eprintln!("  ninja cjpegli");
            std::process::exit(1);
        }
    };
    println!("Using cjpegli: {}", cjpegli.display());

    let images = find_corpus();
    if images.is_empty() {
        eprintln!("ERROR: No test images found");
        std::process::exit(1);
    }
    println!("Found {} test images", images.len());

    let modes = EncodingMode::all_modes();
    println!("Testing {} encoding modes\n", modes.len());

    let qualities = [50, 75, 90];

    // Aggregate results: mode -> quality -> Vec<results>
    let mut aggregated: BTreeMap<EncodingMode, BTreeMap<u8, Vec<ComparisonResult>>> =
        BTreeMap::new();

    for (idx, img_path) in images.iter().enumerate() {
        let name = img_path.file_name().unwrap().to_str().unwrap();
        print!("\r[{}/{}] Processing: {:<40}", idx + 1, images.len(), name);

        let (rgb, width, height) = match load_png(img_path) {
            Some(d) => d,
            None => continue,
        };

        // Save as PNG for C++ (it needs a file path)
        let tmp_png = format!("/tmp/cmp_{}_{}.png", std::process::id(), idx);
        {
            let file = fs::File::create(&tmp_png).unwrap();
            let mut encoder = png::Encoder::new(file, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&rgb).unwrap();
        }

        for mode in &modes {
            for &q in &qualities {
                if let Some(result) =
                    compare_mode(&rgb, width, height, Path::new(&tmp_png), mode, q, &cjpegli)
                {
                    aggregated
                        .entry(*mode)
                        .or_default()
                        .entry(q)
                        .or_default()
                        .push(result);
                }
            }
        }

        let _ = fs::remove_file(&tmp_png);
    }
    println!("\n");

    // Print results table
    println!("{}", "=".repeat(120));
    println!(" RESULTS SUMMARY (averaged across {} images)", images.len());
    println!("{}\n", "=".repeat(120));

    for q in &qualities {
        println!("\n### Quality {} ###\n", q);
        println!(
            "{:<20} | {:>10} {:>10} {:>10} {:>10} | {:>10} {:>10} {:>8}",
            "Mode", "Rust B", "C++ B", "Δ Bytes", "Δ Size", "Rust S2", "C++ S2", "Δ S2"
        );
        println!("{:-<110}", "");

        for mode in &modes {
            if let Some(q_results) = aggregated.get(mode).and_then(|m| m.get(q)) {
                let n = q_results.len() as f64;
                if n == 0.0 {
                    continue;
                }

                let avg_rust_size: f64 =
                    q_results.iter().map(|r| r.rust_size as f64).sum::<f64>() / n;
                let avg_cpp_size: f64 =
                    q_results.iter().map(|r| r.cpp_size as f64).sum::<f64>() / n;
                let size_diff = (avg_rust_size - avg_cpp_size) / avg_cpp_size * 100.0;

                let avg_rust_ssim2: f64 = q_results.iter().map(|r| r.rust_ssim2).sum::<f64>() / n;
                let avg_cpp_ssim2: f64 = q_results.iter().map(|r| r.cpp_ssim2).sum::<f64>() / n;
                let ssim2_diff = avg_rust_ssim2 - avg_cpp_ssim2;

                let byte_diff = avg_rust_size - avg_cpp_size;
                println!(
                    "{:<20} | {:>10.0} {:>10.0} {:>+10.0} {:>+9.3}% | {:>10.2} {:>10.2} {:>+7.2}",
                    mode.to_string(),
                    avg_rust_size,
                    avg_cpp_size,
                    byte_diff,
                    size_diff,
                    avg_rust_ssim2,
                    avg_cpp_ssim2,
                    ssim2_diff
                );
            }
        }
    }

    // Overall summary
    println!("\n{}", "=".repeat(120));
    println!(" OVERALL SUMMARY BY MODE TYPE");
    println!("{}\n", "=".repeat(120));

    // Group by color mode
    for color in &[ColorMode::YCbCr, ColorMode::XYB, ColorMode::Grayscale] {
        let color_name = match color {
            ColorMode::YCbCr => "YCbCr",
            ColorMode::XYB => "XYB",
            ColorMode::Grayscale => "Grayscale",
        };

        let mut total_rust_size = 0.0;
        let mut total_cpp_size = 0.0;
        let mut total_rust_ssim2 = 0.0;
        let mut total_cpp_ssim2 = 0.0;
        let mut count = 0.0;

        for (mode, q_map) in &aggregated {
            if mode.color_mode != *color {
                continue;
            }
            for results in q_map.values() {
                for r in results {
                    total_rust_size += r.rust_size as f64;
                    total_cpp_size += r.cpp_size as f64;
                    total_rust_ssim2 += r.rust_ssim2;
                    total_cpp_ssim2 += r.cpp_ssim2;
                    count += 1.0;
                }
            }
        }

        if count > 0.0 {
            let size_diff = (total_rust_size - total_cpp_size) / total_cpp_size * 100.0;
            let ssim2_diff = (total_rust_ssim2 - total_cpp_ssim2) / count;

            println!(
                "{:<12}: Size {:>+6.2}% vs C++, SSIM2 {:>+.2} vs C++  (n={})",
                color_name, size_diff, ssim2_diff, count as usize
            );
        }
    }

    // Group by sequential vs progressive
    println!();
    for &prog in &[false, true] {
        let mode_name = if prog { "Progressive" } else { "Sequential" };

        let mut total_rust_size = 0.0;
        let mut total_cpp_size = 0.0;
        let mut total_rust_ssim2 = 0.0;
        let mut total_cpp_ssim2 = 0.0;
        let mut count = 0.0;

        for (mode, q_map) in &aggregated {
            if mode.progressive != prog {
                continue;
            }
            for results in q_map.values() {
                for r in results {
                    total_rust_size += r.rust_size as f64;
                    total_cpp_size += r.cpp_size as f64;
                    total_rust_ssim2 += r.rust_ssim2;
                    total_cpp_ssim2 += r.cpp_ssim2;
                    count += 1.0;
                }
            }
        }

        if count > 0.0 {
            let size_diff = (total_rust_size - total_cpp_size) / total_cpp_size * 100.0;
            let ssim2_diff = (total_rust_ssim2 - total_cpp_ssim2) / count;

            println!(
                "{:<12}: Size {:>+6.2}% vs C++, SSIM2 {:>+.2} vs C++  (n={})",
                mode_name, size_diff, ssim2_diff, count as usize
            );
        }
    }

    // Grand total
    println!();
    let mut total_rust_size = 0.0;
    let mut total_cpp_size = 0.0;
    let mut total_rust_ssim2 = 0.0;
    let mut total_cpp_ssim2 = 0.0;
    let mut count = 0.0;

    for q_map in aggregated.values() {
        for results in q_map.values() {
            for r in results {
                total_rust_size += r.rust_size as f64;
                total_cpp_size += r.cpp_size as f64;
                total_rust_ssim2 += r.rust_ssim2;
                total_cpp_ssim2 += r.cpp_ssim2;
                count += 1.0;
            }
        }
    }

    if count > 0.0 {
        let size_diff = (total_rust_size - total_cpp_size) / total_cpp_size * 100.0;
        let ssim2_diff = (total_rust_ssim2 - total_cpp_ssim2) / count;

        println!("{}", "-".repeat(60));
        println!(
            "GRAND TOTAL: Size {:>+6.2}% vs C++, SSIM2 {:>+.2} vs C++",
            size_diff, ssim2_diff
        );
        println!(
            "             ({} total comparisons across {} images)",
            count as usize,
            images.len()
        );
    }

    println!("\n{}", "=".repeat(120));
    println!("LEGEND:");
    println!("  Size Δ: positive = Rust larger, negative = Rust smaller");
    println!("  SSIM2 Δ: positive = Rust better quality, negative = C++ better");
    println!("  SSIM2 scale: 100 = identical, higher = better (typical good: 70-90)");
    println!("{}", "=".repeat(120));
}
