//! Profile target for flamegraph.
//!
//! Run with: cargo flamegraph --release --example profile_encode -o encode.svg

use enough::Unstoppable;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let (width, height) = (2048usize, 2048usize);

    // Create test image
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    let config = EncoderConfig::new(85.0, ChromaSubsampling::Quarter);

    // Run enough iterations for good sampling
    for _ in 0..50 {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        let result = enc.finish().unwrap();
        std::hint::black_box(&result);
    }
}
