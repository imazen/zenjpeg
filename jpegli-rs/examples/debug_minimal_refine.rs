// Minimal test case for AC refinement encoding
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn solid_gray(w: u32, h: u32, value: u8) -> Vec<u8> {
    vec![value; (w * h) as usize]
}

fn gradient(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x + y) % 256) as u8))
        .collect()
}

fn test_pattern(name: &str, data: &[u8], width: u32, height: u32) {
    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(data)
        .expect("encode failed");

    match jpeg_decoder::Decoder::new(&jpeg[..]).decode() {
        Ok(_) => println!("{}: OK ({} bytes)", name, jpeg.len()),
        Err(e) => println!("{}: FAILED - {:?}", name, e),
    }
}

fn main() {
    // Test solid patterns (should be all zeros after DC, minimal refinement needed)
    for size in [49u32, 50, 51, 56, 64] {
        test_pattern(
            &format!("solid_128 {}x{}", size, size),
            &solid_gray(size, size, 128),
            size,
            size,
        );
    }

    println!();

    // Test gradients (more AC content, but regular pattern)
    for size in [49u32, 50, 51, 56, 64] {
        test_pattern(
            &format!("gradient {}x{}", size, size),
            &gradient(size, size),
            size,
            size,
        );
    }

    println!();

    // Test the problematic pseudo-random pattern
    fn photo_like(w: u32, h: u32) -> Vec<u8> {
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8)
            })
            .collect()
    }

    for size in [48u32, 49, 50, 51, 52, 53, 54, 55, 56] {
        test_pattern(
            &format!("photo_like {}x{}", size, size),
            &photo_like(size, size),
            size,
            size,
        );
    }
}
