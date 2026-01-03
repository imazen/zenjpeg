// Debug what blocks look like at encode time for 49x49 vs 50x50
// Focus on blocks in the rightmost column which have different padding

use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn encode_test(size: u32) {
    let data = photo_like(size, size);
    let jpeg = Encoder::new()
        .width(size)
        .height(size)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    let result = jpeg_decoder::Decoder::new(&jpeg[..]).decode();
    eprintln!(
        "{}x{}: {} ({} bytes)",
        size,
        size,
        if result.is_ok() { "OK" } else { "FAIL" },
        jpeg.len()
    );
}

fn main() {
    // Set DEBUG_REFINE_SYMBOLS=1 in the environment before running this

    eprintln!("=== Testing 49x49 (should work) ===");
    encode_test(49);

    // Reset the block counter (it's a static, so we need to restart)
    eprintln!("\n\n=== Testing 50x50 (should fail) ===");
    encode_test(50);
}
