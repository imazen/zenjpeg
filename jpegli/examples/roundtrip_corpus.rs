//! Roundtrip test for corpus images.
//!
//! Usage: cargo run --example roundtrip_corpus --release -- <input_dir> <output_dir> [quality]

use std::env;
use std::fs;
use std::path::Path;

use jpegli::{Encoder, Decoder, PixelFormat, Quality};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input_dir> <output_dir> [quality]", args[0]);
        eprintln!("Example: {} /mnt/v/work/corpus/CID22-512 ./output 90", args[0]);
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let output_dir = &args[2];
    let quality: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(90);

    // Create output directory
    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    fs::create_dir_all(format!("{}/jpeg", output_dir)).expect("Failed to create jpeg directory");
    fs::create_dir_all(format!("{}/roundtrip", output_dir)).expect("Failed to create roundtrip directory");

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

    let mut success_count = 0;
    let mut error_count = 0;
    let mut total_psnr = 0.0;
    let mut total_original_size = 0usize;
    let mut total_jpeg_size = 0usize;

    for (i, entry) in files.iter().enumerate() {
        let path = entry.path();
        let filename = path.file_stem().unwrap().to_string_lossy();

        print!("[{}/{}] {} ... ", i + 1, files.len(), filename);

        match process_image(&path, output_dir, quality) {
            Ok((psnr, original_size, jpeg_size)) => {
                println!("OK (PSNR: {:.2} dB, {:.1}% size)", psnr, 100.0 * jpeg_size as f64 / original_size as f64);
                success_count += 1;
                total_psnr += psnr;
                total_original_size += original_size;
                total_jpeg_size += jpeg_size;
            }
            Err(e) => {
                println!("ERROR: {}", e);
                error_count += 1;
            }
        }
    }

    println!();
    println!("=== Summary ===");
    println!("Success: {}", success_count);
    println!("Errors: {}", error_count);
    if success_count > 0 {
        println!("Average PSNR: {:.2} dB", total_psnr / success_count as f64);
        println!("Compression: {:.1}% of original", 100.0 * total_jpeg_size as f64 / total_original_size as f64);
    }
}

fn process_image(input_path: &Path, output_dir: &str, quality: u8) -> Result<(f64, usize, usize), String> {
    // Read PNG
    let png_data = fs::read(input_path).map_err(|e| format!("Read error: {}", e))?;
    let original_size = png_data.len();

    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().map_err(|e| format!("PNG decode error: {}", e))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("PNG frame error: {}", e))?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB if needed
    let (rgb_data, format) = match info.color_type {
        png::ColorType::Rgb => (buf[..info.buffer_size()].to_vec(), PixelFormat::Rgb),
        png::ColorType::Rgba => {
            // Strip alpha channel
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(width * height * 3);
            for chunk in rgba.chunks(4) {
                rgb.push(chunk[0]);
                rgb.push(chunk[1]);
                rgb.push(chunk[2]);
            }
            (rgb, PixelFormat::Rgb)
        }
        png::ColorType::Grayscale => {
            // Convert to RGB
            let gray = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(width * height * 3);
            for &g in gray {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            (rgb, PixelFormat::Rgb)
        }
        png::ColorType::GrayscaleAlpha => {
            let ga = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(width * height * 3);
            for chunk in ga.chunks(2) {
                rgb.push(chunk[0]);
                rgb.push(chunk[0]);
                rgb.push(chunk[0]);
            }
            (rgb, PixelFormat::Rgb)
        }
        _ => return Err(format!("Unsupported color type: {:?}", info.color_type)),
    };

    // Encode to JPEG
    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(format)
        .quality(Quality::from_quality(quality as f32));

    let jpeg_data = encoder.encode(&rgb_data)
        .map_err(|e| format!("JPEG encode error: {}", e))?;

    let jpeg_size = jpeg_data.len();

    // Save JPEG
    let filename = input_path.file_stem().unwrap().to_string_lossy();
    let jpeg_path = format!("{}/jpeg/{}.jpg", output_dir, filename);
    fs::write(&jpeg_path, &jpeg_data).map_err(|e| format!("Write JPEG error: {}", e))?;

    // Decode JPEG
    let mut decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg_data)
        .map_err(|e| format!("JPEG decode error: {}", e))?;

    // Calculate PSNR
    let psnr = calculate_psnr(&rgb_data, &decoded.data);

    // Save roundtrip PNG
    let rt_path = format!("{}/roundtrip/{}.png", output_dir, filename);
    save_png(&rt_path, &decoded.data, decoded.width as u32, decoded.height as u32)?;

    Ok((psnr, original_size, jpeg_size))
}

fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    if original.len() != decoded.len() {
        return 0.0;
    }

    let mse: f64 = original.iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>() / original.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (255.0 * 255.0 / mse).log10()
}

fn save_png(path: &str, data: &[u8], width: u32, height: u32) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|e| format!("Create file error: {}", e))?;
    let w = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header().map_err(|e| format!("PNG header error: {}", e))?;
    writer.write_image_data(data).map_err(|e| format!("PNG write error: {}", e))?;

    Ok(())
}
