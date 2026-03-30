//! Fuzz target for the push/streaming decode paths.
//!
//! Exercises decode_rows (u8) and decode_rows_f32 (f32) callbacks,
//! which use the ScanlineReader internally but manage buffers for the caller.
#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::decoder::{Decoder, PixelFormat};

fuzz_target!(|data: &[u8]| {
    let max_px = 4_000_000u64;

    // u8 row callbacks — all formats
    for format in [
        PixelFormat::Rgb, PixelFormat::Bgr,
        PixelFormat::Rgba, PixelFormat::Bgra,
        PixelFormat::Gray,
    ] {
        let _ = Decoder::new().max_pixels(max_px)
            .decode_rows(data, format, |_row| Ok(()), enough::Unstoppable);
    }

    // f32 row callbacks
    for format in [PixelFormat::RgbaF32, PixelFormat::GrayF32] {
        let _ = Decoder::new().max_pixels(max_px)
            .decode_rows_f32(data, format, |_row| Ok(()), enough::Unstoppable);
    }
});
