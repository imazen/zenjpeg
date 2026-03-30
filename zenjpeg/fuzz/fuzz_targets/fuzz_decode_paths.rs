//! Structured fuzzer that varies both the JPEG data AND the decoder configuration.
//!
//! Uses `arbitrary` to generate random decode options, ensuring all combinations
//! of output format, target, strictness, and upsampling are explored.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use zenjpeg::decoder::{
    ChromaUpsampling, Decoder, OutputTarget, PixelFormat, Strictness,
};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    pixel_format: u8,
    output_target: u8,
    strictness: u8,
    upsampling: u8,
    auto_orient: bool,
    use_decode_rows: bool,
    data: Vec<u8>,
}

fuzz_target!(|input: FuzzInput| {
    let pixel_format = match input.pixel_format % 5 {
        0 => PixelFormat::Gray,
        1 => PixelFormat::Rgb,
        2 => PixelFormat::Rgba,
        3 => PixelFormat::Bgr,
        _ => PixelFormat::Bgra,
    };

    let output_target = match input.output_target % 5 {
        0 => OutputTarget::Srgb8,
        1 => OutputTarget::SrgbF32,
        2 => OutputTarget::LinearF32,
        3 => OutputTarget::SrgbF32Precise,
        _ => OutputTarget::LinearF32Precise,
    };

    let strictness = match input.strictness % 4 {
        0 => Strictness::Strict,
        1 => Strictness::Balanced,
        2 => Strictness::Lenient,
        _ => Strictness::Permissive,
    };

    let upsampling = match input.upsampling % 2 {
        0 => ChromaUpsampling::Triangle,
        _ => ChromaUpsampling::NearestNeighbor,
    };

    let config = Decoder::new()
        .output_format(pixel_format)
        .output_target(output_target)
        .strictness(strictness)
        .chroma_upsampling(upsampling)
        .auto_orient(input.auto_orient)
        .max_pixels(4_000_000);

    if input.use_decode_rows {
        let _ = config.decode_rows(&input.data, pixel_format, |_row| Ok(()), enough::Unstoppable);
    } else {
        let _ = config.decode(&input.data, enough::Unstoppable);
    }
});
