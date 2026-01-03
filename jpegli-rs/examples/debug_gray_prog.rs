// Debug grayscale progressive encoding
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn gray_photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn main() {
    // Test various sizes to find where failure starts
    for size in 48..=52 {
        let data = gray_photo_like(size, size);
        test_gray(size, size, &data);
    }
}

fn test_gray(width: u32, height: u32, data: &[u8]) {
    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(data)
        .expect("encode failed");

    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("Gray {}x{}: OK ({} bytes)", width, height, jpeg_data.len()),
        Err(e) => println!("Gray {}x{}: FAILED - {:?}", width, height, e),
    }
}
