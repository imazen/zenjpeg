//! Create a test JPEG with mozjpeg and compare decoders.
//!
//! Usage: cargo run --example create_test_jpeg --release -- <png_file>

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <png_file>", args[0]);
        return;
    }

    // Load PNG
    let png_data = fs::read(&args[1]).expect("Failed to read PNG");
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().expect("PNG decode error");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("PNG frame error");

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB
    let rgb_data: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => {
            eprintln!("Unsupported color type: {:?}", info.color_type);
            return;
        }
    };

    println!("PNG: {}x{}, {} bytes RGB", width, height, rgb_data.len());
    println!(
        "First few pixels: {:?}",
        &rgb_data[..12.min(rgb_data.len())]
    );

    // Encode with mozjpeg
    println!("\n=== Encoding with mozjpeg ===");
    let moz_jpeg = encode_mozjpeg(&rgb_data, width, height, 90);
    println!("mozjpeg output: {} bytes", moz_jpeg.len());
    fs::write("/tmp/test_mozjpeg.jpg", &moz_jpeg).unwrap();

    // Decode with jpeg-decoder
    println!("\n=== Decoding mozjpeg output with jpeg-decoder ===");
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&moz_jpeg[..]));
    let moz_decoded = decoder.decode().expect("jpeg-decoder failed");
    let info = decoder.dimensions().unwrap();
    println!(
        "jpeg-decoder: {}x{}, {} bytes",
        info.0,
        info.1,
        moz_decoded.len()
    );
    println!(
        "First few pixels: {:?}",
        &moz_decoded[..12.min(moz_decoded.len())]
    );

    // Decode with jpegli-rs
    println!("\n=== Decoding mozjpeg output with jpegli-rs ===");
    let jpegli_decoded = jpegli::Decoder::new()
        .decode(&moz_jpeg)
        .expect("jpegli-rs failed");
    println!(
        "jpegli-rs: {}x{}, {} bytes",
        jpegli_decoded.width,
        jpegli_decoded.height,
        jpegli_decoded.data.len()
    );
    println!(
        "First few pixels: {:?}",
        &jpegli_decoded.data[..12.min(jpegli_decoded.data.len())]
    );

    // Compare
    println!("\n=== Comparison (jpeg-decoder vs jpegli-rs on mozjpeg output) ===");
    compare_pixels(&moz_decoded, &jpegli_decoded.data);

    // Now test the roundtrip with our encoder
    println!("\n=== Encoding with jpegli-rs ===");
    let jpegli_jpeg = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .jpegli_quality(jpegli::Quality::from_quality(90.0))
        .encode(&rgb_data)
        .expect("jpegli encode failed");
    println!("jpegli-rs output: {} bytes", jpegli_jpeg.len());
    fs::write("/tmp/test_jpegli.jpg", &jpegli_jpeg).unwrap();

    // Decode our output with jpeg-decoder
    println!("\n=== Decoding jpegli-rs output with jpeg-decoder ===");
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&jpegli_jpeg[..]));
    match decoder.decode() {
        Ok(decoded) => {
            let info = decoder.dimensions().unwrap();
            println!(
                "jpeg-decoder: {}x{}, {} bytes",
                info.0,
                info.1,
                decoded.len()
            );
            println!("First few pixels: {:?}", &decoded[..12.min(decoded.len())]);
        }
        Err(e) => {
            println!("jpeg-decoder FAILED: {:?}", e);
        }
    }

    // Decode our output with jpegli-rs
    println!("\n=== Decoding jpegli-rs output with jpegli-rs ===");
    let jpegli_rt = jpegli::Decoder::new()
        .decode(&jpegli_jpeg)
        .expect("jpegli-rs failed");
    println!(
        "jpegli-rs: {}x{}, {} bytes",
        jpegli_rt.width,
        jpegli_rt.height,
        jpegli_rt.data.len()
    );
    println!(
        "First few pixels: {:?}",
        &jpegli_rt.data[..12.min(jpegli_rt.data.len())]
    );

    println!("\nTest files written to /tmp/test_mozjpeg.jpg and /tmp/test_jpegli.jpg");
}

fn encode_mozjpeg(rgb_data: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality as f32);

    let mut started = comp
        .start_compress(Vec::new())
        .expect("mozjpeg start error");

    let row_stride = width * 3;
    for y in 0..height {
        let row_start = y * row_stride;
        let row = &rgb_data[row_start..row_start + row_stride];
        started.write_scanlines(row);
    }

    started.finish().expect("mozjpeg finish error")
}

fn compare_pixels(a: &[u8], b: &[u8]) {
    if a.len() != b.len() {
        println!("Size mismatch: {} vs {}", a.len(), b.len());
        return;
    }

    let mut max_diff = 0i32;
    let mut total_diff = 0u64;
    let mut diff_count = 0usize;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = (x as i32 - y as i32).abs();
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
        a.len(),
        100.0 * diff_count as f64 / a.len() as f64
    );
    println!(
        "Max difference: {}, Avg difference: {:.2}",
        max_diff, avg_diff
    );
}
