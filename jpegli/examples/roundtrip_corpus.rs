//! Roundtrip test for corpus images.
//!
//! Encodes with jpegli-rs, decodes with jpeg-decoder (reference), measures DSSIM.
//!
//! Usage: cargo run --example roundtrip_corpus --release -- <input_dir> <output_dir> [quality]

use dssim::Dssim;
use rgb::RGBA8;
use std::env;
use std::fs;
use std::path::Path;

use jpegli::{Encoder, PixelFormat};
use jpegli::quant::Quality;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input_dir> <output_dir> [quality]", args[0]);
        eprintln!("Example: {} ./testdata ./output 90", args[0]);
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let output_dir = &args[2];
    let quality: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(90);

    // Create output directory
    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    fs::create_dir_all(format!("{}/jpeg", output_dir)).expect("Failed to create jpeg directory");

    println!("Input: {}", input_dir);
    println!("Output: {}", output_dir);
    println!("Quality: {}", quality);
    println!("Encoder: jpegli-rs");
    println!("Decoder: jpeg-decoder (reference)");
    println!("Metric: DSSIM (lower is better)");
    println!();

    // Find all PNG files
    let mut files: Vec<_> = fs::read_dir(input_dir)
        .expect("Failed to read input directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext.to_ascii_lowercase() == "png")
                .unwrap_or(false)
        })
        .collect();

    files.sort_by_key(|e| e.path());

    if files.is_empty() {
        eprintln!("No PNG files found in {}", input_dir);
        std::process::exit(1);
    }

    println!("Found {} PNG files", files.len());
    println!();
    println!(
        "{:>4} {:>30} {:>10} {:>10} {:>12}",
        "#", "Filename", "Size", "JPEG", "DSSIM"
    );
    println!("{}", "-".repeat(72));

    let mut success_count = 0;
    let mut error_count = 0;
    let mut total_dssim = 0.0;
    let mut total_original_size = 0usize;
    let mut total_jpeg_size = 0usize;

    for (i, entry) in files.iter().enumerate() {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy();

        match process_image(&path, output_dir, quality) {
            Ok((dssim, original_size, jpeg_size)) => {
                println!(
                    "{:>4} {:>30} {:>10} {:>10} {:>12.6}",
                    i + 1,
                    truncate_str(&filename, 30),
                    format_size(original_size),
                    format_size(jpeg_size),
                    dssim
                );
                success_count += 1;
                total_dssim += dssim;
                total_original_size += original_size;
                total_jpeg_size += jpeg_size;
            }
            Err(e) => {
                println!(
                    "{:>4} {:>30} ERROR: {}",
                    i + 1,
                    truncate_str(&filename, 30),
                    e
                );
                error_count += 1;
            }
        }
    }

    println!();
    println!("=== Summary ===");
    println!("Success: {}", success_count);
    println!("Errors: {}", error_count);
    if success_count > 0 {
        println!("Average DSSIM: {:.6}", total_dssim / success_count as f64);
        println!(
            "Total size: {} -> {} ({:.1}%)",
            format_size(total_original_size),
            format_size(total_jpeg_size),
            100.0 * total_jpeg_size as f64 / total_original_size as f64
        );
    }
}

fn process_image(
    input_path: &Path,
    output_dir: &str,
    quality: u8,
) -> Result<(f64, usize, usize), String> {
    // Read PNG
    let png_data = fs::read(input_path).map_err(|e| format!("Read: {}", e))?;
    let original_size = png_data.len();

    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("PNG decode: {}", e))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("PNG frame: {}", e))?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB
    let rgb_data = match info.color_type {
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
        _ => return Err(format!("Unsupported color type: {:?}", info.color_type)),
    };

    // Encode with jpegli
    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(quality as f32));

    let jpeg_data = encoder
        .encode(&rgb_data)
        .map_err(|e| format!("JPEG encode: {}", e))?;

    let jpeg_size = jpeg_data.len();

    // Save JPEG
    let filename = input_path.file_stem().unwrap().to_string_lossy();
    let jpeg_path = format!("{}/jpeg/{}.jpg", output_dir, filename);
    fs::write(&jpeg_path, &jpeg_data).map_err(|e| format!("Write: {}", e))?;

    // Decode with reference decoder (jpeg-decoder)
    let mut jpeg_dec = jpeg_decoder::Decoder::new(&jpeg_data[..]);
    let decoded_rgb = jpeg_dec
        .decode()
        .map_err(|e| format!("JPEG decode: {}", e))?;

    // Calculate DSSIM
    let dssim = compute_dssim(&rgb_data, &decoded_rgb, width, height);

    Ok((dssim, original_size, jpeg_size))
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();

    let orig_rgba: Vec<RGBA8> = original
        .chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let dec_rgba: Vec<RGBA8> = decoded
        .chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();

    let orig = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp = attr.create_image_rgba(&dec_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig, comp);
    dssim.into()
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{}B", bytes)
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max_len + 3..])
    }
}
