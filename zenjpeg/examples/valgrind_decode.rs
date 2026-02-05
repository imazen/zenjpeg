//! Single-decode test for valgrind profiling (callgrind/cachegrind)
//!
//! Usage:
//!   cargo build --release --example valgrind_decode
//!   valgrind --tool=callgrind --cache-sim=yes ./target/release/examples/valgrind_decode jpegli
//!   valgrind --tool=callgrind --cache-sim=yes ./target/release/examples/valgrind_decode zune
//!   kcachegrind callgrind.out.*  # To visualize

use enough::Unstoppable;
use std::env;
use std::hint::black_box;
use zenjpeg::{
    decoder::{Decoder, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

fn create_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = ((x * 255) / width as usize) as u8;
            data[idx + 1] = ((y * 255) / height as usize) as u8;
            data[idx + 2] = 128;
        }
    }
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn decode_jpegli(jpeg_data: &[u8]) {
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(jpeg_data, Unstoppable).expect("jpegli decode failed");
    black_box(result);
}

fn decode_zune(jpeg_data: &[u8]) {
    let options = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGB);
    let cursor = ZCursor::new(jpeg_data);
    let mut decoder = JpegDecoder::new_with_options(cursor, options);
    let result = decoder.decode().expect("zune decode failed");
    black_box(result);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <jpegli|zune> [size]", args[0]);
        eprintln!("  size: 512 (default), 1024, or 2048");
        std::process::exit(1);
    }

    let decoder_type = &args[1];
    let size: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);

    eprintln!("Creating {}x{} test JPEG...", size, size);
    let jpeg_data = create_test_jpeg(size, size);
    eprintln!("JPEG size: {} bytes", jpeg_data.len());

    eprintln!("Decoding with {}...", decoder_type);

    match decoder_type.as_str() {
        "jpegli" => decode_jpegli(&jpeg_data),
        "zune" => decode_zune(&jpeg_data),
        _ => {
            eprintln!("Unknown decoder: {}. Use 'jpegli' or 'zune'", decoder_type);
            std::process::exit(1);
        }
    }

    eprintln!("Done.");
}
