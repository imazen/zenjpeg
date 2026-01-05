//! Test decoding with standard vs optimized Huffman tables

use jpegli::decode::Decoder;
use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn main() {
    let width = 64u32;
    let height = 64u32;

    // Create gradient image that fails with XYB
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = ((x * 4) % 256) as u8;
            rgb[i + 1] = ((y * 4) % 256) as u8;
            rgb[i + 2] = 128;
        }
    }

    println!("Testing XYB encoding with different Huffman settings:\n");

    // Test 1: XYB with optimized Huffman (default)
    println!("1. XYB + optimized Huffman (default):");
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true);
    let jpeg = encoder.encode(&rgb).expect("encode");
    println!("   Encoded: {} bytes", jpeg.len());
    print!("   jpeg-decoder: ");
    match decode_zune(&jpeg[..]) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }
    print!("   Native decoder: ");
    match Decoder::new().decode(&jpeg) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }

    // Test 2: XYB with standard Huffman tables
    println!("\n2. XYB + standard Huffman (optimize_huffman=false):");
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .optimize_huffman(false);
    let jpeg = encoder.encode(&rgb).expect("encode");
    println!("   Encoded: {} bytes", jpeg.len());
    print!("   jpeg-decoder: ");
    match decode_zune(&jpeg[..]) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }
    print!("   Native decoder: ");
    match Decoder::new().decode(&jpeg) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }

    // Test 3: YCbCr (no XYB) with optimized Huffman
    println!("\n3. YCbCr + optimized Huffman:");
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(false);
    let jpeg = encoder.encode(&rgb).expect("encode");
    println!("   Encoded: {} bytes", jpeg.len());
    print!("   jpeg-decoder: ");
    match decode_zune(&jpeg[..]) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }
    print!("   Native decoder: ");
    match Decoder::new().decode(&jpeg) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }

    // Test 4: YCbCr with standard Huffman
    println!("\n4. YCbCr + standard Huffman:");
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(false)
        .optimize_huffman(false);
    let jpeg = encoder.encode(&rgb).expect("encode");
    println!("   Encoded: {} bytes", jpeg.len());
    print!("   jpeg-decoder: ");
    match decode_zune(&jpeg[..]) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }
    print!("   Native decoder: ");
    match Decoder::new().decode(&jpeg) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
