//! Compare jpegli-rs decoder output against jpeg-decoder.
//!
//! Usage: cargo run --example decoder_compare --release -- <jpeg_file>

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <jpeg_file>", args[0]);
        return;
    }

    let jpeg_data = fs::read(&args[1]).expect("Failed to read file");
    println!("File: {}", args[1]);
    println!("Size: {} bytes", jpeg_data.len());
    println!();

    // Decode with jpegli-rs
    println!("=== jpegli-rs ===");
    let jpegli_result = jpegli::Decoder::new().decode(&jpeg_data);
    let jpegli_pixels = match &jpegli_result {
        Ok(img) => {
            println!(
                "Decoded: {}x{}, {} bytes",
                img.width,
                img.height,
                img.data.len()
            );
            Some(&img.data)
        }
        Err(e) => {
            println!("Error: {}", e);
            None
        }
    };

    // Decode with jpeg-decoder
    println!("\n=== jpeg-decoder ===");
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&jpeg_data[..]));
    let jpeg_decoder_result = decoder.decode();
    let jpeg_decoder_pixels = match &jpeg_decoder_result {
        Ok(pixels) => {
            let info = decoder.dimensions().unwrap();
            println!(
                "Decoded: {}x{}, {} bytes",
                info.0,
                info.1,
                pixels.len()
            );
            println!("Pixel format: {:?}", info.pixel_format);
            Some(pixels)
        }
        Err(e) => {
            println!("Error: {:?}", e);
            None
        }
    };

    // Compare if both succeeded
    if let (Some(jpegli), Some(jpeg_dec)) = (jpegli_pixels, jpeg_decoder_pixels) {
        println!("\n=== Comparison ===");

        if jpegli.len() != jpeg_dec.len() {
            println!(
                "Size mismatch: jpegli={}, jpeg-decoder={}",
                jpegli.len(),
                jpeg_dec.len()
            );
            return;
        }

        // Calculate statistics
        let mut max_diff = 0i32;
        let mut total_diff = 0u64;
        let mut diff_count = 0usize;

        for (i, (&a, &b)) in jpegli.iter().zip(jpeg_dec.iter()).enumerate() {
            let diff = (a as i32 - b as i32).abs();
            if diff > 0 {
                diff_count += 1;
                total_diff += diff as u64;
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        let avg_diff = if diff_count > 0 {
            total_diff as f64 / diff_count as f64
        } else {
            0.0
        };

        println!(
            "Pixels with differences: {} / {} ({:.2}%)",
            diff_count,
            jpegli.len(),
            100.0 * diff_count as f64 / jpegli.len() as f64
        );
        println!("Max difference: {}", max_diff);
        println!("Avg difference (of non-zero): {:.2}", avg_diff);

        // Show first few differences
        if diff_count > 0 && diff_count <= 20 {
            println!("\nFirst differences:");
            let mut shown = 0;
            for (i, (&a, &b)) in jpegli.iter().zip(jpeg_dec.iter()).enumerate() {
                if a != b && shown < 10 {
                    let px = i / 3;
                    let ch = i % 3;
                    let channel = ["R", "G", "B"][ch];
                    println!(
                        "  Pixel {} {}: jpegli={}, jpeg-decoder={}, diff={}",
                        px,
                        channel,
                        a,
                        b,
                        (a as i32 - b as i32).abs()
                    );
                    shown += 1;
                }
            }
        }

        // Show sample pixels from different areas
        println!("\nSample pixels (R,G,B):");
        let total_px = jpegli.len() / 3;
        let samples = [
            0,
            total_px / 4,
            total_px / 2,
            3 * total_px / 4,
            total_px - 1,
        ];
        for px in samples {
            if px * 3 + 2 < jpegli.len() {
                let j = (jpegli[px * 3], jpegli[px * 3 + 1], jpegli[px * 3 + 2]);
                let d = (jpeg_dec[px * 3], jpeg_dec[px * 3 + 1], jpeg_dec[px * 3 + 2]);
                println!(
                    "  Pixel {:6}: jpegli=({:3},{:3},{:3}) jpeg-dec=({:3},{:3},{:3})",
                    px, j.0, j.1, j.2, d.0, d.1, d.2
                );
            }
        }
    }
}
