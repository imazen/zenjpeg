//! Benchmark decoder performance: zune-jpeg vs jpegli-rs vs mozjpeg

use jpegli::{Decoder, Encoder, PixelFormat};
use std::io::Cursor;
use std::time::Instant;

fn benchmark_decoder(name: &str, jpeg_data: &[u8], iterations: usize) -> Option<(f64, usize)> {
    match name {
        "jpegli-rs" => {
            let start = Instant::now();
            let mut total_pixels = 0;
            for _ in 0..iterations {
                let decoded = Decoder::new().decode(jpeg_data).ok()?;
                total_pixels += decoded.data.len();
            }
            let elapsed = start.elapsed();
            Some((
                elapsed.as_secs_f64() / iterations as f64,
                total_pixels / iterations,
            ))
        }
        "zune-jpeg" => {
            let start = Instant::now();
            let mut total_pixels = 0;
            for _ in 0..iterations {
                let mut decoder = zune_jpeg::JpegDecoder::new(Cursor::new(jpeg_data));
                let pixels = decoder.decode().ok()?;
                total_pixels += pixels.len();
            }
            let elapsed = start.elapsed();
            Some((
                elapsed.as_secs_f64() / iterations as f64,
                total_pixels / iterations,
            ))
        }
        "mozjpeg" => {
            let start = Instant::now();
            let mut total_pixels = 0;
            for _ in 0..iterations {
                let decoder = mozjpeg::Decompress::new_mem(jpeg_data).ok()?;
                let image = decoder.rgb().ok()?;
                total_pixels += image.width() * image.height() * 3;
            }
            let elapsed = start.elapsed();
            Some((
                elapsed.as_secs_f64() / iterations as f64,
                total_pixels / iterations,
            ))
        }
        "jpeg-decoder" => {
            let start = Instant::now();
            let mut total_pixels = 0;
            for _ in 0..iterations {
                let mut decoder = zune_jpeg::JpegDecoder::new(
                    zune_jpeg::zune_core::bytestream::ZCursor::new(jpeg_data),
                );
                let pixels = decoder.decode().ok()?;
                total_pixels += pixels.len();
            }
            let elapsed = start.elapsed();
            Some((
                elapsed.as_secs_f64() / iterations as f64,
                total_pixels / iterations,
            ))
        }
        _ => None,
    }
}

fn format_throughput(size_bytes: usize, time_secs: f64) -> String {
    let mpixels_per_sec = (size_bytes as f64 / 3.0) / time_secs / 1_000_000.0;
    format!("{:.1} MP/s", mpixels_per_sec)
}

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];

    println!(
        "Test image: {}x{} RGB ({} bytes)\n",
        info.width,
        info.height,
        rgb.len()
    );

    // Encode with standard Huffman
    let jpeg_std = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(false)
        .encode(rgb)
        .unwrap();

    // Encode with optimized Huffman
    let jpeg_opt = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();

    let iterations = 100;
    let decoders = ["jpegli-rs", "zune-jpeg", "mozjpeg", "jpeg-decoder"];

    println!(
        "=== Standard Huffman ({} bytes, {} iterations) ===\n",
        jpeg_std.len(),
        iterations
    );
    for decoder_name in &decoders {
        if let Some((time, pixel_count)) = benchmark_decoder(decoder_name, &jpeg_std, iterations) {
            let throughput = format_throughput(rgb.len(), time);
            println!(
                "{:15} {:8.3} ms/decode  {}  ({} pixels)",
                decoder_name,
                time * 1000.0,
                throughput,
                pixel_count
            );
        } else {
            println!("{:15} FAILED", decoder_name);
        }
    }

    println!(
        "\n=== Optimized Huffman ({} bytes, {} iterations) ===\n",
        jpeg_opt.len(),
        iterations
    );
    for decoder_name in &decoders {
        if let Some((time, pixel_count)) = benchmark_decoder(decoder_name, &jpeg_opt, iterations) {
            let throughput = format_throughput(rgb.len(), time);
            println!(
                "{:15} {:8.3} ms/decode  {}  ({} pixels)",
                decoder_name,
                time * 1000.0,
                throughput,
                pixel_count
            );
        } else {
            println!("{:15} FAILED", decoder_name);
        }
    }

    // Relative performance
    println!("\n=== Relative Performance (Standard Huffman) ===\n");
    let mut times = Vec::new();
    for decoder_name in &decoders {
        if let Some((time, _)) = benchmark_decoder(decoder_name, &jpeg_std, iterations) {
            times.push((*decoder_name, time));
        }
    }
    times.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let fastest_time = times[0].1;
    for (name, time) in &times {
        let relative = time / fastest_time;
        println!(
            "{:15} {:5.2}x {}",
            name,
            relative,
            if *name == times[0].0 {
                "← fastest"
            } else {
                ""
            }
        );
    }
}
