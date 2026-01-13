//! Profile target for flamegraph.
//!
//! Run with: cargo flamegraph --release --example profile_encode -o encode.svg

use jpegli::{JpegEncoder, Quality};

fn main() {
    let (width, height) = (2048, 2048);

    // Create test image
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    // Run enough iterations for good sampling
    for _ in 0..50 {
        let result = JpegEncoder::new(width as u32, height as u32)
            .quality(Quality::from_quality(85.0))
            .encode(&pixels)
            .unwrap();
        std::hint::black_box(&result);
    }
}
