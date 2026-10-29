//! WASM SIMD128 performance benchmark
//!
//! Measures encode/decode performance on WASM.
//!
//! Run with SIMD:
//! ```sh
//! CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
//! RUSTFLAGS="-C target-feature=+simd128" \
//! cargo run --release -p zenjpeg --example wasm_bench \
//!     --target wasm32-wasip1 --no-default-features
//! ```
//!
//! Run without SIMD (scalar):
//! ```sh
//! CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime" \
//! cargo run --release -p zenjpeg --example wasm_bench \
//!     --target wasm32-wasip1 --no-default-features
//! ```

use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    // Report SIMD status
    #[cfg(target_feature = "simd128")]
    println!("Mode: WASM SIMD128 enabled");
    #[cfg(not(target_feature = "simd128"))]
    println!("Mode: WASM scalar (no SIMD)");

    // Test different image sizes
    for &(width, height) in &[(64, 64), (256, 256), (512, 512), (1024, 1024)] {
        benchmark_size(width, height);
    }
}

fn benchmark_size(width: u32, height: u32) {
    let iterations = if width <= 256 { 20 } else { 5 };
    let pixels = width as usize * height as usize;

    // Create test image - gradient pattern
    let mut input = vec![0u8; pixels * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            input[idx] = ((x * 255) / width as usize) as u8;
            input[idx + 1] = ((y * 255) / height as usize) as u8;
            input[idx + 2] = 128;
        }
    }

    println!("\n=== {}x{} ({} pixels) ===", width, height, pixels);

    // Benchmark encoding
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);

    let start = Instant::now();
    let mut total_size = 0;
    for _ in 0..iterations {
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder creation failed");
        enc.push_packed(&input, Unstoppable).expect("push failed");
        let jpeg = enc.finish().expect("encoding failed");
        total_size += jpeg.len();
    }
    let encode_time = start.elapsed();
    let encode_per_iter = encode_time / iterations as u32;
    let encode_mpixels_per_sec =
        (pixels as f64 * iterations as f64) / encode_time.as_secs_f64() / 1_000_000.0;

    println!(
        "Encode: {:?}/iter, {:.2} MP/s, avg size: {} bytes",
        encode_per_iter,
        encode_mpixels_per_sec,
        total_size / iterations
    );

    // Get a JPEG for decode benchmark
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(&input, Unstoppable).expect("push failed");
    let jpeg = enc.finish().expect("encoding failed");

    // Benchmark decoding
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);

    let start = Instant::now();
    for _ in 0..iterations {
        let _decoded = decoder.decode(&jpeg, Unstoppable).expect("decoding failed");
    }
    let decode_time = start.elapsed();
    let decode_per_iter = decode_time / iterations as u32;
    let decode_mpixels_per_sec =
        (pixels as f64 * iterations as f64) / decode_time.as_secs_f64() / 1_000_000.0;

    println!(
        "Decode: {:?}/iter, {:.2} MP/s",
        decode_per_iter, decode_mpixels_per_sec
    );
}
