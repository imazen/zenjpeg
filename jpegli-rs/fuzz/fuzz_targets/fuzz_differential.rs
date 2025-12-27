//! Differential fuzz target comparing jpegli against reference decoders.
//!
//! This target decodes the same JPEG data with multiple decoders and checks
//! for consistency. It helps find cases where jpegli behaves differently
//! from established decoders.
//!
//! Note: This requires zune-jpeg as a dependency.

#![no_main]

use jpegli::decode::Decoder;
use jpegli::types::PixelFormat;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Decode with jpegli
    let jpegli_result = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(data);

    // Decode with zune-jpeg (reference)
    let zune_result = decode_with_zune(data);

    match (&jpegli_result, &zune_result) {
        // Both succeed - check for reasonable similarity
        (Ok(jpegli_img), Ok(zune_img)) => {
            // Dimensions must match exactly
            assert_eq!(
                jpegli_img.width, zune_img.width,
                "Width mismatch: jpegli={} zune={}",
                jpegli_img.width, zune_img.width
            );
            assert_eq!(
                jpegli_img.height, zune_img.height,
                "Height mismatch: jpegli={} zune={}",
                jpegli_img.height, zune_img.height
            );

            // Pixel values should be close (allowing for decoder differences)
            // JPEG decoding can have small differences due to IDCT implementations
            let max_diff = compute_max_diff(&jpegli_img.data, &zune_img.data);
            assert!(
                max_diff <= 4,
                "Pixel values differ too much: max_diff={}",
                max_diff
            );
        }

        // Both fail - acceptable
        (Err(_), Err(_)) => {}

        // One succeeds, one fails - potential issue
        // Note: Different decoders have different strictness, so we only log
        (Ok(_), Err(_)) => {
            // jpegli accepted what zune rejected - jpegli might be more lenient
            // This is not necessarily a bug, but worth noting
        }
        (Err(_), Ok(_)) => {
            // jpegli rejected what zune accepted - jpegli might be stricter
            // This could indicate a parsing bug in jpegli
        }
    }
});

/// Decode JPEG using zune-jpeg.
fn decode_with_zune(data: &[u8]) -> Result<DecodedImage, ()> {
    use zune_jpeg::JpegDecoder;

    let mut decoder = JpegDecoder::new(data);

    // Get dimensions
    decoder.decode_headers().map_err(|_| ())?;
    let info = decoder.info().ok_or(())?;
    let width = info.width as u32;
    let height = info.height as u32;

    // Decode pixels
    let pixels = decoder.decode().map_err(|_| ())?;

    Ok(DecodedImage {
        width,
        height,
        data: pixels,
    })
}

struct DecodedImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

/// Compute maximum absolute difference between two pixel buffers.
fn compute_max_diff(a: &[u8], b: &[u8]) -> u8 {
    if a.len() != b.len() {
        return 255;
    }

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}
