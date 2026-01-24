//! Fuzz target for JPEG decoding.
//!
//! This is the primary fuzz target - it tests the decoder's ability to handle
//! arbitrary (potentially malformed) JPEG data without panicking or crashing.
//!
//! Security-critical: The decoder must gracefully reject malformed input.

#![no_main]

use zenjpeg::decode::Decoder;
use zenjpeg::types::PixelFormat;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test default decoding
    let decoder = Decoder::new();
    let _ = decoder.decode(data);

    // Test with different output formats
    for format in [
        PixelFormat::Gray,
        PixelFormat::Rgb,
        PixelFormat::Rgba,
        PixelFormat::Bgr,
        PixelFormat::Bgra,
    ] {
        let decoder = Decoder::new().output_format(format);
        let _ = decoder.decode(data);
    }

    // Test with various decoder options
    let decoder = Decoder::new()
        .fancy_upsampling(true)
        .block_smoothing(true);
    let _ = decoder.decode(data);

    let decoder = Decoder::new()
        .fancy_upsampling(false)
        .block_smoothing(false);
    let _ = decoder.decode(data);
});
