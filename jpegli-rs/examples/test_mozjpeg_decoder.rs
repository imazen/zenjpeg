//! Test if mozjpeg (libjpeg-turbo) can decode our optimized Huffman tables

use jpegli::{Encoder, PixelFormat};

fn test_with_mozjpeg(data: &[u8], label: &str) -> bool {
    println!("  Attempting mozjpeg::Decompress::new_mem...");
    match mozjpeg::Decompress::new_mem(data) {
        Ok(decoder) => {
            println!("  Decompress created, attempting rgb()...");
            match decoder.rgb() {
                Ok(_pixels) => {
                    println!("✓ {}: mozjpeg SUCCESS", label);
                    true
                }
                Err(e) => {
                    println!("✗ {}: mozjpeg decode FAILED - {:?}", label, e);
                    false
                }
            }
        }
        Err(e) => {
            println!("✗ {}: mozjpeg decompress init FAILED - {:?}", label, e);
            false
        }
    }
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

    println!("=== Test 1: Standard (IJG) Huffman tables ===");
    let jpeg_std = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(false)
        .encode(rgb)
        .unwrap();

    println!("Standard JPEG: {} bytes", jpeg_std.len());
    test_with_mozjpeg(&jpeg_std, "Standard/mozjpeg");

    println!("\n=== Test 2: Optimized Huffman tables ===");
    let jpeg_opt = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();

    println!("Optimized JPEG: {} bytes", jpeg_opt.len());
    test_with_mozjpeg(&jpeg_opt, "Optimized/mozjpeg");

    println!("\n=== Test 3: Our own decoder ===");
    match jpegli::Decoder::new().decode(&jpeg_opt) {
        Ok(decoded) => {
            println!(
                "✓ Optimized/jpegli-rs: SUCCESS ({}x{})",
                decoded.width, decoded.height
            );
        }
        Err(e) => {
            println!("✗ Optimized/jpegli-rs: FAILED - {:?}", e);
        }
    }

    println!("\n=== Test 4: jpeg-decoder (pure Rust) ===");
    let mut jpeg_decoder = jpeg_decoder::Decoder::new(&jpeg_opt[..]);
    match jpeg_decoder.decode() {
        Ok(_pixels) => {
            println!("✓ Optimized/jpeg-decoder: SUCCESS");
        }
        Err(e) => {
            println!("✗ Optimized/jpeg-decoder: FAILED - {:?}", e);
        }
    }
}
