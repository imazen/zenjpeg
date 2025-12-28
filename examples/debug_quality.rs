//! Debug quality size behavior

use zenjpeg::{Encoder, EncodingStrategy, Quality};

fn main() {
    let width = 128;
    let height = 128;

    // Create a test image with content (same as aq_locked_tests test)
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 13 + i / 7) % 256) as u8)
        .collect();

    println!("Sizes with AUTO strategy:");
    for q in [30, 40, 50, 60, 70, 75, 80, 85, 90, 95] {
        let jpeg = Encoder::new()
            .quality(Quality::Standard(q))
            .encode_rgb(&rgb, width, height)
            .expect("encoding failed");
        println!("Q{}: {} bytes", q, jpeg.len());
    }

    println!("\nSizes with MOZJPEG strategy:");
    for q in [30, 40, 50, 60, 70, 75, 80, 85, 90, 95] {
        let jpeg = Encoder::new()
            .quality(Quality::Standard(q))
            .strategy(EncodingStrategy::Mozjpeg)
            .encode_rgb(&rgb, width, height)
            .expect("encoding failed");
        println!("Q{}: {} bytes", q, jpeg.len());
    }

    println!("\nSizes with JPEGLI strategy:");
    for q in [30, 40, 50, 60, 70, 75, 80, 85, 90, 95] {
        let jpeg = Encoder::new()
            .quality(Quality::Standard(q))
            .strategy(EncodingStrategy::Jpegli)
            .encode_rgb(&rgb, width, height)
            .expect("encoding failed");
        println!("Q{}: {} bytes", q, jpeg.len());
    }
}
