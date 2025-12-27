//! Generate C++ jpegli reference data for parity testing.
//!
//! This tool runs cjpegli on a corpus of images and captures:
//! - File sizes at various quality levels
//! - DSSIM scores
//! - SSIMULACRA2 scores
//! - Butteraugli scores
//!
//! Output is JSON that can be committed to the repo for standalone testing.
//!
//! Usage:
//!   cargo run --example generate_cpp_reference [CORPUS_DIR] [OUTPUT.json]
//!
//! Environment variables:
//!   CJPEGLI_PATH - path to cjpegli binary (default: /home/lilith/work/jpegli/build/tools/cjpegli)
//!   MAX_IMAGES - limit number of images (default: 24)

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Reference data for a single image at a single quality level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPoint {
    pub quality: u8,
    pub file_size: usize,
    pub dssim: f64,
    pub ssimulacra2: f64,
    pub butteraugli: f64,
}

/// Reference data for a single image across all quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageReference {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub original_size: usize,
    pub points: Vec<QualityPoint>,
}

/// Complete reference dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CppReferenceData {
    pub version: String,
    pub cjpegli_version: String,
    pub generated_at: String,
    pub images: Vec<ImageReference>,
}

fn get_cjpegli_version(cjpegli_path: &str) -> String {
    Command::new(cjpegli_path)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string())
        .trim()
        .to_string()
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    // Convert to RGB if needed
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buf.chunks(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

fn encode_with_cjpegli(cjpegli_path: &str, input_path: &Path, quality: u8) -> Option<Vec<u8>> {
    let temp_output = std::env::temp_dir().join(format!(
        "cjpegli_ref_{}_{}.jpg",
        input_path.file_stem()?.to_str()?,
        quality
    ));

    let status = Command::new(cjpegli_path)
        .arg(input_path)
        .arg(&temp_output)
        .arg(format!("--quality={}", quality))
        .status()
        .ok()?;

    if !status.success() {
        return None;
    }

    let data = fs::read(&temp_output).ok()?;
    let _ = fs::remove_file(&temp_output);
    Some(data)
}

fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;

    let rgb = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => pixels.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => return None,
    };

    Some((rgb, info.width as u32, info.height as u32))
}

fn compute_dssim(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use dssim::Dssim;
    use rgb::RGBA;

    let attr = Dssim::new();

    // Convert to RGBA
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
    let val: f64 = dssim.into();
    val
}

fn compute_ssimulacra2(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

    let orig_rgb: Vec<[f32; 3]> = orig
        .chunks(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();
    let comp_rgb: Vec<[f32; 3]> = comp
        .chunks(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();

    let orig_frame = Rgb::new(
        orig_rgb,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let comp_frame = Rgb::new(
        comp_rgb,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_frame, comp_frame).unwrap_or(0.0)
}

fn compute_butteraugli_score(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};

    let params = ButteraugliParams::default();
    match compute_butteraugli(orig, comp, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => f64::NAN,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let cjpegli_path = std::env::var("CJPEGLI_PATH")
        .map(PathBuf::from)
        .or_else(|_| jpegli::test_utils::find_cjpegli().ok_or(()))
        .expect("cjpegli not found. Set CJPEGLI_PATH or build internal/jpegli-cpp");
    let cjpegli_path = cjpegli_path.to_string_lossy().to_string();

    // Default corpus locations to check
    let testdata_dir = jpegli::test_utils::get_testdata_dir();
    let mut corpus_paths: Vec<PathBuf> = Vec::new();
    let flower_dir = testdata_dir.join("jxl/flower");
    if flower_dir.exists() {
        corpus_paths.push(flower_dir);
    }
    // Check common relative locations
    let candidates = [
        "../codec-eval/codec-corpus/kodak",
        "../codec-corpus/kodak",
        "codec-corpus/kodak",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            corpus_paths.push(p);
        }
    }

    let corpus_dir = args
        .get(1)
        .map(PathBuf::from)
        .or_else(|| {
            corpus_paths
                .into_iter()
                .find(|p| p.exists())
        })
        .expect("No corpus directory found. Set JPEGLI_TESTDATA env var or pass CORPUS_DIR argument");

    let output_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cpp_reference_data.json"));

    // Quality levels to test
    let qualities: Vec<u8> = vec![30, 50, 70, 80, 85, 90, 95, 100];

    eprintln!("Generating C++ reference data...");
    eprintln!("  cjpegli: {}", cjpegli_path);
    eprintln!("  corpus: {}", corpus_dir.display());
    eprintln!("  output: {}", output_path.display());
    eprintln!("  qualities: {:?}", qualities);

    let mut images = Vec::new();

    // Find all PNG files
    let mut png_files: Vec<_> = fs::read_dir(&corpus_dir)
        .expect("Failed to read corpus directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .map(|e| e.path())
        .collect();

    png_files.sort();

    // Limit for reasonable runtime
    let max_images = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);

    if png_files.len() > max_images {
        eprintln!(
            "  Limiting to {} images (set MAX_IMAGES to change)",
            max_images
        );
        png_files.truncate(max_images);
    }

    for (idx, png_path) in png_files.iter().enumerate() {
        let name = png_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        eprint!("[{}/{}] {} ... ", idx + 1, png_files.len(), name);
        std::io::stderr().flush().unwrap();

        let Some((orig_rgb, width, height)) = load_png(png_path) else {
            eprintln!("SKIP (failed to load)");
            continue;
        };

        let original_size = fs::metadata(png_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        let mut points = Vec::new();

        for &quality in &qualities {
            let Some(jpeg_data) = encode_with_cjpegli(&cjpegli_path, png_path, quality) else {
                eprintln!("SKIP Q{} (encode failed)", quality);
                continue;
            };

            let file_size = jpeg_data.len();

            let Some((decoded_rgb, _, _)) = decode_jpeg(&jpeg_data) else {
                eprintln!("SKIP Q{} (decode failed)", quality);
                continue;
            };

            let dssim = compute_dssim(&orig_rgb, &decoded_rgb, width as usize, height as usize);
            let ssimulacra2 =
                compute_ssimulacra2(&orig_rgb, &decoded_rgb, width as usize, height as usize);
            let butteraugli =
                compute_butteraugli_score(&orig_rgb, &decoded_rgb, width as usize, height as usize);

            points.push(QualityPoint {
                quality,
                file_size,
                dssim,
                ssimulacra2,
                butteraugli,
            });
        }

        eprintln!("OK ({} points)", points.len());

        images.push(ImageReference {
            name,
            width,
            height,
            original_size,
            points,
        });
    }

    let reference = CppReferenceData {
        version: "1.0".to_string(),
        cjpegli_version: get_cjpegli_version(&cjpegli_path),
        generated_at: chrono::Utc::now().to_rfc3339(),
        images,
    };

    let json = serde_json::to_string_pretty(&reference).expect("Failed to serialize");
    fs::write(&output_path, &json).expect("Failed to write output");

    eprintln!(
        "\nGenerated {} bytes to {}",
        json.len(),
        output_path.display()
    );

    // Print summary
    eprintln!("\nSummary:");
    for img in &reference.images {
        eprintln!("  {}: {}x{}", img.name, img.width, img.height);
        for pt in &img.points {
            eprintln!(
                "    Q{:3}: {:6} bytes, DSSIM={:.6}, SSIM2={:.2}, Butteraugli={:.4}",
                pt.quality, pt.file_size, pt.dssim, pt.ssimulacra2, pt.butteraugli
            );
        }
    }
}
