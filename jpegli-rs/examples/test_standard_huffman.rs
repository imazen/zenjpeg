//! Test if zune-jpeg can decode JPEGs with standard (non-optimized) Huffman tables

use jpegli::{Encoder, PixelFormat};

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    // Load PNG
    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];

    println!("=== Test 1: Standard (IJG) Huffman Tables ===");
    let jpeg_standard = Encoder::new()
        .width(info.0)
        .height(info.1)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(false) // Use standard IJG tables
        .encode(rgb)
        .unwrap();

    println!("Standard Huffman JPEG: {} bytes", jpeg_standard.len());

    let mut zune_decoder1 = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&jpeg_standard));
    match zune_decoder1.decode() {
        Ok(_) => println!("✓ zune-jpeg can decode STANDARD (IJG) Huffman\n"),
        Err(e) => println!("✗ zune-jpeg FAILED on standard: {:?}\n", e),
    }

    println!("=== Test 2: Optimized Huffman Tables ===");
    let jpeg_optimized = Encoder::new()
        .width(info.0)
        .height(info.1)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true) // Build image-specific tables
        .encode(rgb)
        .unwrap();

    println!("Optimized Huffman JPEG: {} bytes", jpeg_optimized.len());

    let mut zune_decoder2 = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&jpeg_optimized));
    match zune_decoder2.decode() {
        Ok(_) => println!("✓ zune-jpeg can decode OPTIMIZED Huffman\n"),
        Err(e) => println!("✗ zune-jpeg FAILED on optimized: {:?}\n", e),
    }
}
