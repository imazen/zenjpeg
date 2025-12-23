//! Compare jpegli-rs encoder against mozjpeg.
//!
//! Usage: cargo run --example compare_encoders --release -- <input_dir> <output_dir> [quality]

use std::env;
use std::fs;
use std::path::Path;

use jpegli::{Encoder, PixelFormat, Quality};
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input_dir> <output_dir> [quality]", args[0]);
        eprintln!("Example: {} /mnt/v/work/corpus/CID22-512 ./compare_out 90", args[0]);
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let output_dir = &args[2];
    let quality: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(90);

    // Create output directories
    fs::create_dir_all(format!("{}/jpegli_rs", output_dir)).expect("Failed to create jpegli_rs dir");
    fs::create_dir_all(format!("{}/mozjpeg", output_dir)).expect("Failed to create mozjpeg dir");

    println!("Input: {}", input_dir);
    println!("Output: {}", output_dir);
    println!("Quality: {}", quality);
    println!();

    // Find all PNG files
    let mut files: Vec<_> = fs::read_dir(input_dir)
        .expect("Failed to read input directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .map(|ext| ext.to_ascii_lowercase() == "png")
                .unwrap_or(false)
        })
        .collect();

    files.sort_by_key(|e| e.path());

    println!("Found {} PNG files", files.len());
    println!();
    println!("{:<40} {:>10} {:>10} {:>12} {:>12}", "File", "jpegli KB", "moz KB", "jpegli DSSIM", "moz DSSIM");
    println!("{}", "-".repeat(96));

    let mut jpegli_total_size = 0usize;
    let mut moz_total_size = 0usize;
    let mut jpegli_total_dssim = 0.0f64;
    let mut moz_total_dssim = 0.0f64;
    let mut count = 0usize;

    for entry in files.iter() {
        let path = entry.path();
        let filename = path.file_stem().unwrap().to_string_lossy();

        let jpegli_jpeg = format!("{}/jpegli_rs/{}.jpg", output_dir, filename);
        let moz_jpeg = format!("{}/mozjpeg/{}.jpg", output_dir, filename);

        // Load PNG
        let (rgb_data, width, height) = match load_png(&path) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("{}: Load error: {}", filename, e);
                continue;
            }
        };

        // Encode with jpegli-rs
        let jpegli_result = encode_with_jpegli(&rgb_data, width, height, &jpegli_jpeg, quality);

        // Encode with mozjpeg
        let moz_result = encode_with_mozjpeg(&rgb_data, width, height, &moz_jpeg, quality);

        match (jpegli_result, moz_result) {
            (Ok(jpegli_size), Ok(moz_size)) => {
                // Calculate DSSIM for both using CLI tool
                let jpegli_dssim = calculate_dssim(&path, &jpegli_jpeg);
                let moz_dssim = calculate_dssim(&path, &moz_jpeg);

                match (jpegli_dssim, moz_dssim) {
                    (Ok(jd), Ok(md)) => {
                        println!("{:<40} {:>10.1} {:>10.1} {:>12.6} {:>12.6}",
                            filename,
                            jpegli_size as f64 / 1024.0,
                            moz_size as f64 / 1024.0,
                            jd, md);

                        jpegli_total_size += jpegli_size;
                        moz_total_size += moz_size;
                        jpegli_total_dssim += jd;
                        moz_total_dssim += md;
                        count += 1;
                    }
                    (Err(e), _) => eprintln!("{}: jpegli DSSIM error: {}", filename, e),
                    (_, Err(e)) => eprintln!("{}: mozjpeg DSSIM error: {}", filename, e),
                }
            }
            (Err(e), _) => eprintln!("{}: jpegli encode error: {}", filename, e),
            (_, Err(e)) => eprintln!("{}: mozjpeg encode error: {}", filename, e),
        }
    }

    println!("{}", "-".repeat(96));
    println!();
    println!("=== Summary ({} files) ===", count);
    println!("jpegli-rs:");
    println!("  Total size: {:.1} KB ({:.2} KB avg)",
        jpegli_total_size as f64 / 1024.0,
        jpegli_total_size as f64 / 1024.0 / count as f64);
    println!("  Avg DSSIM:  {:.6} (lower is better)", jpegli_total_dssim / count as f64);
    println!();
    println!("mozjpeg:");
    println!("  Total size: {:.1} KB ({:.2} KB avg)",
        moz_total_size as f64 / 1024.0,
        moz_total_size as f64 / 1024.0 / count as f64);
    println!("  Avg DSSIM:  {:.6} (lower is better)", moz_total_dssim / count as f64);
    println!();

    let size_ratio = jpegli_total_size as f64 / moz_total_size as f64;
    let dssim_ratio = jpegli_total_dssim / moz_total_dssim;
    println!("jpegli/mozjpeg size ratio:  {:.3}x", size_ratio);
    println!("jpegli/mozjpeg DSSIM ratio: {:.3}x", dssim_ratio);

    println!();
    if size_ratio < 1.0 && dssim_ratio <= 1.0 {
        println!("Result: jpegli-rs produces smaller files with equal or better quality!");
    } else if size_ratio <= 1.05 && dssim_ratio < 1.0 {
        println!("Result: jpegli-rs produces better quality with similar file sizes!");
    } else if size_ratio > 1.0 && dssim_ratio < 1.0 {
        println!("Result: jpegli-rs produces better quality but larger files");
    } else if size_ratio < 1.0 && dssim_ratio > 1.0 {
        println!("Result: jpegli-rs produces smaller files but lower quality");
    } else {
        println!("Result: mozjpeg currently produces better results");
    }
}

fn load_png(path: &Path) -> Result<(Vec<u8>, usize, usize), String> {
    let png_data = fs::read(path).map_err(|e| format!("Read error: {}", e))?;

    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().map_err(|e| format!("PNG decode error: {}", e))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("PNG frame error: {}", e))?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB
    let rgb_data = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(width * height * 3);
            for chunk in rgba.chunks(4) {
                rgb.push(chunk[0]);
                rgb.push(chunk[1]);
                rgb.push(chunk[2]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let gray = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(width * height * 3);
            for &g in gray {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let ga = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(width * height * 3);
            for chunk in ga.chunks(2) {
                rgb.push(chunk[0]);
                rgb.push(chunk[0]);
                rgb.push(chunk[0]);
            }
            rgb
        }
        _ => return Err(format!("Unsupported color type: {:?}", info.color_type)),
    };

    Ok((rgb_data, width, height))
}

fn encode_with_jpegli(rgb_data: &[u8], width: usize, height: usize, output: &str, quality: u8) -> Result<usize, String> {
    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(quality as f32));

    let jpeg_data = encoder.encode(rgb_data)
        .map_err(|e| format!("JPEG encode error: {}", e))?;

    let size = jpeg_data.len();
    fs::write(output, &jpeg_data).map_err(|e| format!("Write error: {}", e))?;

    Ok(size)
}

fn encode_with_mozjpeg(rgb_data: &[u8], width: usize, height: usize, output: &str, quality: u8) -> Result<usize, String> {
    use mozjpeg::{Compress, ColorSpace};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality as f32);

    // Start compression to Vec<u8>
    let mut started = comp.start_compress(Vec::new())
        .map_err(|e| format!("mozjpeg start error: {}", e))?;

    // Write scanlines row by row
    let row_stride = width * 3;
    for y in 0..height {
        let row_start = y * row_stride;
        let row = &rgb_data[row_start..row_start + row_stride];
        started.write_scanlines(row);
    }

    let jpeg_data = started.finish()
        .map_err(|e| format!("mozjpeg finish error: {}", e))?;

    let size = jpeg_data.len();
    fs::write(output, &jpeg_data).map_err(|e| format!("Write error: {}", e))?;

    Ok(size)
}

fn calculate_dssim(original_path: &Path, compressed_path: &str) -> Result<f64, String> {
    let result = Command::new("dssim")
        .arg(original_path)
        .arg(compressed_path)
        .output()
        .map_err(|e| format!("dssim execution error: {}", e))?;

    if !result.status.success() {
        return Err(format!("dssim failed: {}", String::from_utf8_lossy(&result.stderr)));
    }

    let output = String::from_utf8_lossy(&result.stdout);
    // dssim output format: "0.001234\tfilename"
    let dssim_str = output.split_whitespace().next()
        .ok_or("Failed to parse dssim output")?;

    dssim_str.parse::<f64>()
        .map_err(|e| format!("Failed to parse DSSIM value: {}", e))
}
