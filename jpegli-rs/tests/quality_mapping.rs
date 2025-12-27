//! Quality mapping between jpegli and mozjpeg.
//!
//! Finds the jpegli quality value that produces the same DSSIM as a given mozjpeg quality.
//! This allows users to translate mozjpeg quality settings to jpegli equivalents.

use dssim::Dssim;
use rgb::RGBA8;
use std::fs;
use std::path::Path;

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

fn encode_jpegli(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(quality))
        .encode(rgb)
        .expect("jpegli encode")
}

fn encode_mozjpeg(rgb: &[u8], width: usize, height: usize, quality: f32) -> Vec<u8> {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality);
    comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1)); // 4:4:4

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

/// Finds jpegli quality that produces the same DSSIM as mozjpeg at given quality.
fn find_matching_jpegli_quality(
    rgb: &[u8],
    width: usize,
    height: usize,
    moz_quality: f32,
) -> (f32, f64, f64) {
    let moz_data = encode_mozjpeg(rgb, width, height, moz_quality);
    let moz_decoded = decode_jpeg(&moz_data);
    let moz_dssim = compute_dssim(rgb, &moz_decoded, width, height);

    // Binary search for matching jpegli quality
    let mut low = 1.0f32;
    let mut high = 100.0f32;
    let mut best_quality = 50.0f32;
    let mut best_diff = f64::INFINITY;
    let mut best_dssim = 0.0;

    for _ in 0..20 {
        // 20 iterations for good precision
        let mid = (low + high) / 2.0;
        let jpegli_data = encode_jpegli(rgb, width as u32, height as u32, mid);
        let jpegli_decoded = decode_jpeg(&jpegli_data);
        let jpegli_dssim = compute_dssim(rgb, &jpegli_decoded, width, height);

        let diff = (jpegli_dssim - moz_dssim).abs();
        if diff < best_diff {
            best_diff = diff;
            best_quality = mid;
            best_dssim = jpegli_dssim;
        }

        // Lower DSSIM = better quality, so if jpegli_dssim > moz_dssim, need higher quality
        if jpegli_dssim > moz_dssim {
            low = mid;
        } else {
            high = mid;
        }
    }

    (best_quality, moz_dssim, best_dssim)
}

/// Test quality mapping on a real image.
#[test]
fn test_quality_mapping() {
    let path = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let (rgb, width, height) = load_png(&path).expect("load png");

    println!("\n=== Quality Mapping: mozjpeg -> jpegli (4:4:4) ===");
    println!(
        "{:>12} {:>12} {:>12} {:>12}",
        "mozjpeg Q", "jpegli Q", "moz DSSIM", "jpegli DSSIM"
    );
    println!("{}", "-".repeat(52));

    for moz_q in [50, 60, 70, 75, 80, 85, 90, 95] {
        let (jpegli_q, moz_dssim, jpegli_dssim) =
            find_matching_jpegli_quality(&rgb, width, height, moz_q as f32);

        println!(
            "{:>12} {:>12.1} {:>12.6} {:>12.6}",
            moz_q, jpegli_q, moz_dssim, jpegli_dssim
        );
    }
}

/// Test quality mapping across a corpus.
#[test]
#[ignore] // Requires corpus directory
fn test_quality_mapping_corpus() {
    let corpus_dir = std::env::var("CORPUS_DIR").unwrap_or_else(|_| {
        eprintln!("Set CORPUS_DIR environment variable to run this test");
        return "/mnt/v/work/corpus/CID22-512".to_string();
    });

    let corpus_path = Path::new(&corpus_dir);
    if !corpus_path.exists() {
        eprintln!("Corpus directory not found: {}", corpus_dir);
        return;
    }

    // Find PNG files
    let files: Vec<_> = fs::read_dir(corpus_path)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext.to_ascii_lowercase() == "png")
                .unwrap_or(false)
        })
        .take(50) // Limit for speed
        .collect();

    println!("\n=== Corpus Quality Mapping ({} images) ===", files.len());

    let moz_qualities = [60, 70, 80, 90];
    let mut mappings: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();

    for entry in &files {
        let path = entry.path();
        if let Some((rgb, width, height)) = load_png(&path) {
            for &moz_q in &moz_qualities {
                let (jpegli_q, _, _) =
                    find_matching_jpegli_quality(&rgb, width, height, moz_q as f32);
                mappings.entry(moz_q).or_default().push(jpegli_q);
            }
        }
    }

    println!(
        "{:>12} {:>12} {:>12} {:>12}",
        "mozjpeg Q", "jpegli Q avg", "min", "max"
    );
    println!("{}", "-".repeat(52));

    for moz_q in moz_qualities {
        if let Some(jpegli_qs) = mappings.get(&moz_q) {
            let avg: f32 = jpegli_qs.iter().sum::<f32>() / jpegli_qs.len() as f32;
            let min = jpegli_qs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = jpegli_qs.iter().cloned().fold(0.0, f32::max);
            println!("{:>12} {:>12.1} {:>12.1} {:>12.1}", moz_q, avg, min, max);
        }
    }
}

/// Generates a quality mapping table that can be used programmatically.
#[test]
#[ignore] // Requires test image
fn generate_quality_mapping_table() {
    let path = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let (rgb, width, height) = load_png(&path).expect("load png");

    println!("\n// Quality mapping table: mozjpeg Q -> jpegli Q (for same DSSIM)");
    println!("// Generated from flower_small.rgb.png test image");
    println!("const MOZJPEG_TO_JPEGLI: &[(u8, u8)] = &[");

    for moz_q in (10..=100).step_by(5) {
        let (jpegli_q, _, _) = find_matching_jpegli_quality(&rgb, width, height, moz_q as f32);
        println!("    ({}, {}),", moz_q, jpegli_q.round() as u8);
    }

    println!("];");
}
