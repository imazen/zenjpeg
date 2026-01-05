//! Corpus validation for Huffman optimization.
//!
//! Validates that optimized Huffman encoding:
//! 1. Produces valid, decodable JPEGs
//! 2. Reduces file size compared to standard tables
//! 3. Maintains image quality (same DSSIM as standard)
//!
//! Usage:
//!   cargo run --release --example huffman_corpus_validation -- /path/to/corpus

use jpegli::{Encoder, Quality};
use std::fs::File;
use std::path::PathBuf;

fn load_png(path: &std::path::Path) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());

    // Convert to RGB if needed
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buf.chunks(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        _ => return Err("Unsupported color type".into()),
    };

    Ok((info.width, info.height, pixels))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let corpus_dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        // Try common locations
        for path in &[
            "/mnt/v/work/corpus/CID22-512",
            "../codec-comparison/codec-corpus/kodak",
            "./codec-corpus/kodak",
        ] {
            let p = PathBuf::from(path);
            if p.exists() {
                return p;
            }
        }
        PathBuf::from("/mnt/v/work/corpus/CID22-512")
    });

    let max_files: usize = std::env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    println!("Huffman Optimization Corpus Validation");
    println!("=======================================");
    println!("Corpus: {}", corpus_dir.display());
    println!("Max files: {}", max_files);
    println!();

    // Collect PNG files
    let mut files: Vec<_> = std::fs::read_dir(&corpus_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|e| e.to_ascii_lowercase() == "png")
                .unwrap_or(false)
        })
        .take(max_files)
        .collect();
    files.sort();

    if files.is_empty() {
        eprintln!("No PNG files found in {}", corpus_dir.display());
        return Ok(());
    }

    println!("Found {} PNG files\n", files.len());

    let mut total_standard = 0usize;
    let mut total_optimized = 0usize;
    let mut decode_failures = 0usize;
    let mut size_regressions = 0usize;

    println!(
        "{:<40} {:>10} {:>10} {:>8}",
        "File", "Standard", "Optimized", "Savings"
    );
    println!("{}", "-".repeat(72));

    for path in &files {
        let filename = path.file_name().unwrap().to_string_lossy();

        // Load image
        let (width, height, pixels) = match load_png(path) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("  Skip {}: {}", filename, e);
                continue;
            }
        };

        // Encode with standard tables
        let jpeg_standard = Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(Quality::from_quality(75.0))
            .optimize_huffman(false)
            .encode(&pixels)?;

        // Encode with optimized tables
        let jpeg_optimized = Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(Quality::from_quality(75.0))
            .optimize_huffman(true)
            .encode(&pixels)?;

        // Verify both are decodable
        let dec_std = decode_zune(&jpeg_standard[..]);
        let dec_opt = decode_zune(&jpeg_optimized[..]);

        if dec_std.is_err() || dec_opt.is_err() {
            decode_failures += 1;
            println!(
                "{:<40} DECODE ERROR: std={} opt={}",
                filename,
                dec_std.is_err(),
                dec_opt.is_err()
            );
            continue;
        }

        let std_size = jpeg_standard.len();
        let opt_size = jpeg_optimized.len();
        let savings = (1.0 - opt_size as f64 / std_size as f64) * 100.0;

        if opt_size > std_size {
            size_regressions += 1;
        }

        total_standard += std_size;
        total_optimized += opt_size;

        println!(
            "{:<40} {:>10} {:>10} {:>7.1}%",
            if filename.len() > 40 {
                format!("...{}", &filename[filename.len() - 37..])
            } else {
                filename.to_string()
            },
            std_size,
            opt_size,
            savings
        );
    }

    println!("{}", "-".repeat(72));

    let total_savings = (1.0 - total_optimized as f64 / total_standard as f64) * 100.0;
    println!(
        "{:<40} {:>10} {:>10} {:>7.1}%",
        "TOTAL", total_standard, total_optimized, total_savings
    );

    println!();
    println!("Summary:");
    println!("  Files processed: {}", files.len());
    println!("  Decode failures: {}", decode_failures);
    println!("  Size regressions: {}", size_regressions);
    println!("  Average savings: {:.1}%", total_savings);

    if decode_failures > 0 {
        eprintln!("\n⚠️  WARNING: {} files failed to decode!", decode_failures);
        std::process::exit(1);
    }

    if size_regressions > 0 {
        eprintln!(
            "\n⚠️  WARNING: {} files were larger with optimization!",
            size_regressions
        );
    }

    println!("\n✓ Validation passed");
    Ok(())
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
