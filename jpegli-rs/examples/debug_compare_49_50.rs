// Compare 49x49 vs 50x50 grayscale progressive encoding
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn gray_photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn encode(width: u32, height: u32, data: &[u8]) -> Vec<u8> {
    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(data)
        .expect("encode failed")
}

fn main() {
    // Generate and save both
    let data_49 = gray_photo_like(49, 49);
    let jpeg_49 = encode(49, 49, &data_49);
    std::fs::write("/tmp/gray_49x49.jpg", &jpeg_49).unwrap();

    let data_50 = gray_photo_like(50, 50);
    let jpeg_50 = encode(50, 50, &data_50);
    std::fs::write("/tmp/gray_50x50.jpg", &jpeg_50).unwrap();

    println!("49x49: {} bytes", jpeg_49.len());
    println!("50x50: {} bytes", jpeg_50.len());

    // Try decoding
    match decode_zune(&jpeg_49[..]) {
        Ok(_) => println!("49x49 decode: OK"),
        Err(e) => println!("49x49 decode: FAILED - {:?}", e),
    }

    match decode_zune(&jpeg_50[..]) {
        Ok(_) => println!("50x50 decode: OK"),
        Err(e) => println!("50x50 decode: FAILED - {:?}", e),
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
