//! Minimal WASM SIMD128 test for jpegli encoder/decoder
//!
//! Run with:
//! ```sh
//! CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
//! RUSTFLAGS="-C target-feature=+simd128" \
//! cargo run --release -p zenjpeg --example wasm_simd_test \
//!     --target wasm32-wasip1 --no-default-features --features "std,decoder"
//! ```

use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    println!("Creating test image...");

    // Create a simple 64x64 RGB gradient
    let width = 64u32;
    let height = 64u32;
    let mut input = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            input[idx] = (x * 4) as u8; // R
            input[idx + 1] = (y * 4) as u8; // G
            input[idx + 2] = 128; // B
        }
    }

    println!("Encoding JPEG...");
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(&input, Unstoppable).expect("push failed");
    let jpeg = enc.finish().expect("encoding failed");
    println!("Encoded to {} bytes", jpeg.len());

    println!("Decoding JPEG...");
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let decoded = decoder.decode(&jpeg, Unstoppable).expect("decoding failed");

    let pixels = decoded.pixels_u8().unwrap();
    println!(
        "Decoded {}x{} image with {} bytes",
        decoded.width,
        decoded.height,
        pixels.len()
    );

    // Verify dimensions match
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(pixels.len(), (width * height * 3) as usize);

    // Check pixel values are reasonable
    let mut max_diff = 0i32;
    for i in 0..input.len() {
        let diff = (input[i] as i32 - pixels[i] as i32).abs();
        max_diff = max_diff.max(diff);
    }
    println!("Max pixel difference: {}", max_diff);
    assert!(max_diff < 30, "max_diff {} too large", max_diff);

    println!("✓ WASM SIMD128 encode/decode test passed!");
}
