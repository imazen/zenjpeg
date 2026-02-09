//! Profile real encoder allocations with actual images.
//!
//! Run with:
//!   cargo run --release --example real_alloc_profile --features alloc-instrument

use std::fs::File;
use std::io::BufReader;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    Some((buf, info.width, info.height))
}

fn encode_and_report(pixels: &[u8], width: u32, height: u32, quality: u8) {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("Failed to create encoder");

    encoder
        .push_packed(pixels, enough::Unstoppable)
        .expect("Failed to push pixels");

    // Print allocation stats BEFORE finish (which consumes encoder)
    let stats = encoder.encode_stats();
    eprintln!("  Strip allocations: {}", stats.summary());
    eprintln!("{}", stats.by_context_summary());

    let mut output = Vec::new();
    encoder.finish_into(&mut output).expect("Failed to finish");

    let blocks =
        ((width + 7) / 8) * ((height + 7) / 8) + 2 * ((width + 15) / 16) * ((height + 15) / 16);

    eprintln!(
        "  Output: {} bytes ({:.2} bytes/block)",
        output.len(),
        output.len() as f64 / blocks as f64
    );
}

fn main() {
    let corpus = codec_corpus::Corpus::new().ok();

    eprintln!("=== Real Encoder Allocation Profile (CLIC 2025) ===\n");

    // CLIC 2025 validation images - realistic modern photos
    let clic_images = [
        "097cb426910ba8ce2525dd8bb7fb1777", // 1507x2048
        "0c49a5cce349020bbba2f97ae41e90ba",
        "100a02c269c5948392f283b2aa3bb4da",
        "11f2b039b293758398b1a7a8afa64bb2",
    ];

    let clic_dir = corpus
        .as_ref()
        .and_then(|c| c.get("clic2025/validation").ok());

    for img_hash in clic_images {
        let path = clic_dir
            .as_ref()
            .map(|d| {
                d.join(format!("{}.png", img_hash))
                    .to_string_lossy()
                    .to_string()
            })
            .unwrap_or_default();
        if let Some((pixels, width, height)) = load_png(&path) {
            let short_hash = &img_hash[..8];
            eprintln!("=== CLIC {}.. ({}x{}) ===", short_hash, width, height);
            for quality in [75, 90] {
                eprintln!("Q{}:", quality);
                encode_and_report(&pixels, width, height, quality);
            }
            eprintln!();
        }
    }
}
