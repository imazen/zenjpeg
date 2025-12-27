// Debug progressive encoding with simple patterns
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn solid_gray(w: u32, h: u32, value: u8) -> Vec<u8> {
    vec![value; (w * h * 3) as usize]
}

fn gradient(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                let v = ((x + y) % 256) as u8;
                [v, v, v]
            })
        })
        .collect()
}

fn test_pattern(name: &str, data: &[u8], width: u32, height: u32) {
    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(data)
        .expect("encode failed");

    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("{}: OK ({} bytes)", name, jpeg_data.len()),
        Err(e) => println!("{}: FAILED - {:?}", name, e),
    }
}

fn main() {
    // Test solid gray - should be simple (all zeros after DC)
    test_pattern("solid_128 50x50", &solid_gray(50, 50, 128), 50, 50);
    test_pattern("solid_128 56x56", &solid_gray(56, 56, 128), 56, 56);

    // Test gradient - more complexity
    test_pattern("gradient 50x50", &gradient(50, 50), 50, 50);
    test_pattern("gradient 56x56", &gradient(56, 56), 56, 56);

    // Test photo-like pattern (from test_size_scaling)
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
    test_pattern("photo_like 50x50", &photo_like(50, 50), 50, 50);
    test_pattern("photo_like 56x56", &photo_like(56, 56), 56, 56);
}
