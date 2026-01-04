//! Debug exactly which Huffman table zune-jpeg rejects

use jpegli::{Encoder, PixelFormat};
use std::io::Cursor;

fn test_jpeg(data: &[u8], label: &str) -> bool {
    let mut decoder = zune_jpeg::JpegDecoder::new(Cursor::new(data));
    match decoder.decode() {
        Ok(_) => {
            println!("✓ {}: zune-jpeg SUCCESS", label);
            true
        }
        Err(e) => {
            println!("✗ {}: zune-jpeg FAILED - {:?}", label, e);
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

    // Create a simple 2x2 solid color image for minimal Huffman tables
    let simple_rgb = vec![128u8; 2 * 2 * 3];

    println!("=== Test 1: Simple image, standard Huffman ===");
    let simple_std = Encoder::new()
        .width(2)
        .height(2)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(false)
        .encode(&simple_rgb)
        .unwrap();
    test_jpeg(&simple_std, "Simple/Standard");

    println!("\n=== Test 2: Simple image, optimized Huffman ===");
    let simple_opt = Encoder::new()
        .width(2)
        .height(2)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true)
        .encode(&simple_rgb)
        .unwrap();
    test_jpeg(&simple_opt, "Simple/Optimized");

    println!("\n=== Test 3: Full image, standard Huffman ===");
    let full_std = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(false)
        .encode(rgb)
        .unwrap();
    test_jpeg(&full_std, "Full/Standard");

    println!("\n=== Test 4: Full image, optimized Huffman ===");
    let full_opt = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();
    test_jpeg(&full_opt, "Full/Optimized");
}
