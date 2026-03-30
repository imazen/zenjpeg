//! Primary JPEG decode fuzzer — exercises all pixel formats, output targets,
//! strictness levels, and decode modes on every input.
#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::decoder::{
    ChromaUpsampling, Decoder, OutputTarget, PixelFormat, Strictness,
};

fuzz_target!(|data: &[u8]| {
    let max_px = 4_000_000u64;

    // 1. Default decode
    let _ = Decoder::new().max_pixels(max_px).decode(data, enough::Unstoppable);

    // 2. All u8 pixel formats
    for format in [
        PixelFormat::Gray, PixelFormat::Rgb, PixelFormat::Rgba,
        PixelFormat::Bgr, PixelFormat::Bgra,
    ] {
        let _ = Decoder::new().output_format(format).max_pixels(max_px)
            .decode(data, enough::Unstoppable);
    }

    // 3. f32 output targets
    for target in [
        OutputTarget::SrgbF32, OutputTarget::LinearF32,
        OutputTarget::SrgbF32Precise, OutputTarget::LinearF32Precise,
    ] {
        let _ = Decoder::new().output_target(target).max_pixels(max_px)
            .decode(data, enough::Unstoppable);
    }

    // 4. NearestNeighbor upsampling
    let _ = Decoder::new().chroma_upsampling(ChromaUpsampling::NearestNeighbor)
        .max_pixels(max_px).decode(data, enough::Unstoppable);

    // 5. Strict + Permissive (different error paths)
    let _ = Decoder::new().strictness(Strictness::Strict).max_pixels(max_px)
        .decode(data, enough::Unstoppable);
    let _ = Decoder::new().strictness(Strictness::Permissive).max_pixels(max_px)
        .decode(data, enough::Unstoppable);

    // 6. Coefficient extraction (DCT domain, no IDCT)
    let _ = Decoder::new().max_pixels(max_px)
        .decode_coefficients(data, enough::Unstoppable);

    // 7. YCbCr f32 planes (no RGB conversion)
    let _ = Decoder::new().max_pixels(max_px)
        .decode_to_ycbcr_f32(data, enough::Unstoppable);

    // 8. Auto-orient (EXIF DCT-domain transform)
    let _ = Decoder::new().auto_orient(true).max_pixels(max_px)
        .decode(data, enough::Unstoppable);

    // 9. Push-model row callbacks (streaming path)
    let _ = Decoder::new().max_pixels(max_px)
        .decode_rows(data, PixelFormat::Rgb, |_row| Ok(()), enough::Unstoppable);
    let _ = Decoder::new().max_pixels(max_px)
        .decode_rows(data, PixelFormat::Gray, |_row| Ok(()), enough::Unstoppable);
    let _ = Decoder::new().max_pixels(max_px)
        .decode_rows_f32(data, PixelFormat::RgbaF32, |_row| Ok(()), enough::Unstoppable);
});
