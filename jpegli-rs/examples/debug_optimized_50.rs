// Test 50x50 with optimized Huffman tables
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn gray_photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn main() {
    let data = gray_photo_like(50, 50);

    // With optimize_huffman = false (direct encoding)
    let jpeg_no_opt = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    // With optimize_huffman = true (token replay)
    let jpeg_opt = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    println!("No optimization: {} bytes", jpeg_no_opt.len());
    match decode_zune(&jpeg_no_opt[..]) {
        Ok(_) => println!("  Decode: OK"),
        Err(e) => println!("  Decode: FAILED - {:?}", e),
    }

    println!("With optimization: {} bytes", jpeg_opt.len());
    match decode_zune(&jpeg_opt[..]) {
        Ok(_) => println!("  Decode: OK"),
        Err(e) => println!("  Decode: FAILED - {:?}", e),
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
