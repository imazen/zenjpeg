// Test progressive encoding at various sizes
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

fn test_size(width: u32, height: u32) {
    let data = photo_like(width, height);

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(false) // Test direct encoding
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("{}x{}: OK ({} bytes)", width, height, jpeg_data.len()),
        Err(e) => println!("{}x{}: FAILED - {:?}", width, height, e),
    }
}

fn main() {
    // Test various sizes - narrow down failure threshold
    test_size(48, 48);
    test_size(49, 49);
    test_size(50, 50);
    test_size(51, 51);
    test_size(52, 52);
    test_size(53, 53);
    test_size(54, 54);
    test_size(55, 55);
    test_size(56, 56);
}
