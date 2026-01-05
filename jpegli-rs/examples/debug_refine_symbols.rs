// Debug what symbols are being encoded in the AC refinement scan
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn gray_photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn main() {
    // Set debug environment variable to trigger debug output
    std::env::set_var("DEBUG_REFINE_SYMBOLS", "1");

    let data_49 = gray_photo_like(49, 49);
    let data_50 = gray_photo_like(50, 50);

    println!("=== 49x49 (working) ===\n");
    let jpeg_49 = Encoder::new()
        .width(49)
        .height(49)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data_49)
        .expect("encode failed");

    println!("\n=== 50x50 (failing) ===\n");
    let jpeg_50 = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data_50)
        .expect("encode failed");

    // Test decoding
    println!("\n=== Decode Test ===");
    match decode_zune(&jpeg_49[..]) {
        Ok(_) => println!("49x49: decode OK ({} bytes)", jpeg_49.len()),
        Err(e) => println!("49x49: decode FAILED - {:?}", e),
    }

    match decode_zune(&jpeg_50[..]) {
        Ok(_) => println!("50x50: decode OK ({} bytes)", jpeg_50.len()),
        Err(e) => println!("50x50: decode FAILED - {:?}", e),
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
