//! Differential fuzz target comparing zenjpeg against reference decoders.
//!
//! This target decodes the same JPEG data with multiple decoders and checks
//! for consistency. It helps find cases where zenjpeg behaves differently
//! from established decoders.
//!
//! Note: This requires zune-jpeg as a dependency.

#![no_main]

use zenjpeg::decoder::{Decoder, PixelFormat};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Decode with zenjpeg
    let zenjpeg_result = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .max_pixels(4_000_000)
        .decode(data, enough::Unstoppable);

    // Decode with zune-jpeg (reference)
    let zune_result = decode_with_zune(data);

    match (&zenjpeg_result, &zune_result) {
        // Both succeed - check for reasonable similarity
        (Ok(zenjpeg_img), Ok(zune_img)) => {
            let zj_pixels = match zenjpeg_img.pixels_u8() {
                Some(p) => p,
                None => return,
            };
            // Dimensions must match exactly
            assert_eq!(
                zenjpeg_img.width, zune_img.width,
                "Width mismatch: zenjpeg={} zune={}",
                zenjpeg_img.width, zune_img.width
            );
            assert_eq!(
                zenjpeg_img.height, zune_img.height,
                "Height mismatch: zenjpeg={} zune={}",
                zenjpeg_img.height, zune_img.height
            );

            // Normalize zune output to RGB so we can compare with zenjpeg's RGB output.
            // zune-jpeg returns grayscale as 1 byte/pixel for grayscale JPEGs even when
            // RGB is requested via DecoderOptions, so we expand it here.
            let pixel_count = (zenjpeg_img.width as usize) * (zenjpeg_img.height as usize);
            let zune_rgb: Vec<u8> = if zune_img.data.len() == pixel_count {
                // Grayscale → expand each Y to (Y, Y, Y)
                zune_img.data.iter().flat_map(|&y| [y, y, y]).collect()
            } else if zune_img.data.len() == pixel_count * 3 {
                zune_img.data.clone()
            } else {
                // Some other layout (CMYK, YCbCr) — skip the comparison rather than
                // raise a false positive on a format mismatch.
                return;
            };

            // Pixel values should be close (allowing for decoder differences)
            // JPEG decoding can have small differences due to IDCT implementations
            let max_diff = compute_max_diff(zj_pixels, &zune_rgb);
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
            // zenjpeg accepted what zune rejected - might be more lenient
        }
        (Err(_), Ok(_)) => {
            // zenjpeg rejected what zune accepted - might be stricter
        }
    }
});

/// Decode JPEG using zune-jpeg as a reference decoder.
fn decode_with_zune(data: &[u8]) -> Result<DecodedImage, ()> {
    use zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;

    let mut decoder = JpegDecoder::new(ZCursor::new(data));

    // Decode (headers + pixels in one call)
    let pixels = decoder.decode().map_err(|_| ())?;
    let info = decoder.info().ok_or(())?;
    let width = info.width as u32;
    let height = info.height as u32;

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
