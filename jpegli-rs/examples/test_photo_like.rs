// Test photo-like progressive encoding
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                let r = ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8;
                let g = ((x.wrapping_mul(13) ^ y.wrapping_mul(23)) % 256) as u8;
                let b = ((x.wrapping_mul(11) ^ y.wrapping_mul(19)) % 256) as u8;
                [r, g, b]
            })
        })
        .collect()
}

fn main() {
    let width = 128u32;
    let height = 96u32;
    let data = photo_like(width, height);

    // Test with optimize_huffman = false first (direct encoding)
    println!("Testing with optimize_huffman=false:");
    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    std::fs::write("/tmp/photo_like_no_opt.jpg", &jpeg_data).ok();
    println!("  Encoded {} bytes", jpeg_data.len());

    match decode_zune(&jpeg_data[..]) {
        Ok(_) => println!("  Decode: OK"),
        Err(e) => println!("  Decode: FAILED - {:?}", e),
    }

    // Test with optimize_huffman = true (two-pass encoding)
    println!("\nTesting with optimize_huffman=true:");
    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    std::fs::write("/tmp/photo_like_opt.jpg", &jpeg_data).ok();
    println!("  Encoded {} bytes", jpeg_data.len());

    match decode_zune(&jpeg_data[..]) {
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
