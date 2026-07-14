//! WASM decoder tests - verify grayscale JPEG decoding works in WASM.
//!
//! This test was created to reproduce a browser WASM crash where:
//! - RGB JPEG decode works fine
//! - Grayscale JPEG decode crashes with "unreachable" trap
//!
//! The gain map file is extracted from an UltraHDR JPEG and is a real-world
//! grayscale image that triggers the crash in browser WASM environments.
//!
//! Run with:
//!   wasm-pack test --node
//!   wasm-pack test --headless --chrome
//!
//! NOTE: Node.js WASM may pass while browser WASM crashes - this is the bug.

#![cfg(target_arch = "wasm32")]

use enough::Unstoppable;
use wasm_bindgen_test::*;

// Enable browser testing to reproduce the crash
wasm_bindgen_test_configure!(run_in_browser);

use zenjpeg::decoder::{Decoder, PixelFormat};

/// Set up panic hook for better error messages in WASM.
fn setup() {
    console_error_panic_hook::set_once();
}

/// UltraHDR gain map - 64x64 grayscale JPEG extracted from test_ultrahdr.jpg.
/// This is a real gain map from an HDR photo that crashes browser WASM decode.
const GAINMAP_GRAY_64X64: &[u8] = include_bytes!("../../fuzz/corpus/seed/gainmap_gray_64x64.jpg");

/// Test RGB decode works in WASM - this should always pass.
#[wasm_bindgen_test]
fn test_wasm_decode_rgb() {
    setup();

    // Create a simple RGB JPEG for testing
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");

    // 8x8 red image
    let pixels = vec![255u8, 0, 0].repeat(64);
    enc.push_packed(&pixels, enough::Unstoppable).expect("push");
    let jpeg_data = enc.finish().expect("finish");

    // Decode RGB
    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg_data, Unstoppable)
        .expect("RGB decode should work in WASM");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    assert_eq!(decoded.format, PixelFormat::Rgb);
}

/// Test grayscale decode in WASM - THIS CRASHES IN BROWSER WASM.
///
/// The gain map from UltraHDR is a grayscale JPEG that decodes fine in:
/// - Native (x86_64, aarch64)
/// - Node.js WASM (wasm-pack test --node)
///
/// But crashes with "unreachable" trap in:
/// - Chrome browser WASM
/// - Firefox browser WASM
///
/// This test exists to track when the fix lands.
#[wasm_bindgen_test]
fn test_wasm_decode_grayscale_gainmap() {
    setup();
    let decoder = Decoder::new();

    // This line crashes in browser WASM with "RuntimeError: unreachable"
    let decoded = decoder
        .decode(GAINMAP_GRAY_64X64, Unstoppable)
        .expect("Grayscale gain map decode should work in WASM");

    // Verify dimensions match expected gain map size
    assert_eq!(decoded.width, 64, "Expected 64x64 gain map");
    assert_eq!(decoded.height, 64, "Expected 64x64 gain map");

    // Output should be grayscale (single channel) or expanded to RGB
    // The decoder may expand grayscale to RGB for compatibility
    let expected_pixels = 64 * 64;
    assert!(
        decoded.pixels_u8().unwrap().len() == expected_pixels
            || decoded.pixels_u8().unwrap().len() == expected_pixels * 3,
        "Expected {} or {} bytes, got {}",
        expected_pixels,
        expected_pixels * 3,
        decoded.pixels_u8().unwrap().len()
    );
}

/// Test requesting grayscale output format explicitly.
/// This is an alternative path that may behave differently.
#[wasm_bindgen_test]
fn test_wasm_decode_grayscale_explicit_format() {
    setup();
    let decoder = Decoder::new().output_format(PixelFormat::Gray);

    let decoded = decoder
        .decode(GAINMAP_GRAY_64X64, Unstoppable)
        .expect("Explicit grayscale decode should work in WASM");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.format, PixelFormat::Gray);
    assert_eq!(
        decoded.pixels_u8().unwrap().len(),
        64 * 64,
        "Grayscale should be 1 byte per pixel"
    );
}

/// Test synthetic grayscale encode/decode roundtrip.
/// This tests if the issue is specific to the gain map file or general grayscale.
#[wasm_bindgen_test]
fn test_wasm_grayscale_roundtrip() {
    setup();
    use zenjpeg::encoder::{EncoderConfig, PixelLayout};

    // Create simple grayscale gradient
    let mut gray_data = vec![0u8; 64 * 64];
    for y in 0..64 {
        for x in 0..64 {
            gray_data[y * 64 + x] = ((x + y) * 2) as u8;
        }
    }

    // Encode as grayscale
    let config = EncoderConfig::grayscale(90.0);
    let mut enc = config
        .encode_from_bytes(64, 64, PixelLayout::Gray8Srgb)
        .expect("encoder setup");
    enc.push_packed(&gray_data, enough::Unstoppable)
        .expect("push");
    let jpeg_data = enc.finish().expect("finish encode");

    // Decode - this may crash in browser WASM
    let decoder = Decoder::new().output_format(PixelFormat::Gray);
    let decoded = decoder
        .decode(&jpeg_data, Unstoppable)
        .expect("Synthetic grayscale roundtrip should work");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.format, PixelFormat::Gray);
}

/// Test decoding the existing flower_gray.jpg from fuzz corpus.
/// This is a larger grayscale image that may trigger different code paths.
#[wasm_bindgen_test]
fn test_wasm_decode_flower_gray() {
    setup();

    // Load the flower_gray.jpg from fuzz corpus
    const FLOWER_GRAY: &[u8] = include_bytes!("../../fuzz/corpus/seed/flower_gray.jpg");

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(FLOWER_GRAY)
        .expect("Flower gray decode should work in WASM");

    assert!(decoded.width > 0, "width should be positive");
    assert!(decoded.height > 0, "height should be positive");
    assert!(
        !decoded.pixels_u8().unwrap().is_empty(),
        "data should not be empty"
    );
}
