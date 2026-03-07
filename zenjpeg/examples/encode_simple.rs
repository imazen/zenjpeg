//! Simple JPEG encoding example for AQ map comparison testing.
//!
//! Usage:
//! ```bash
//! cargo run --release --example encode_simple -- input.png output.jpg 75
//! ```

use std::env;
use std::fs;
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = zenjpeg_bench_utils::load_png(path).ok()?;
    let bytes: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    Some((bytes, img.width() as u32, img.height() as u32))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <input.png> <output.jpg> <quality>", args[0]);
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    let quality: f32 = args[3].parse().expect("Invalid quality");

    let (rgb, width, height) = load_png(input_path).expect("Failed to load PNG");

    eprintln!("Loaded {}x{} image from {:?}", width, height, input_path);

    use enough::Unstoppable;
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

    // Use baseline mode with no Huffman optimization for fair comparison with C++ cjpegli -p 0
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .progressive(false)
        .optimize_huffman(false);

    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("Failed to create encoder");

    encoder.push_packed(&rgb, Unstoppable).expect("Push failed");
    let jpeg_data = encoder.finish().expect("Encode failed");

    fs::write(output_path, &jpeg_data).expect("Failed to write JPEG");
    eprintln!("Wrote {} bytes to {:?}", jpeg_data.len(), output_path);
}
