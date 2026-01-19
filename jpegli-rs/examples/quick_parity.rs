//! Quick parity test - verify encoder output is decodable and consistent.
//!
//! Run with: cargo run --release --example quick_parity

use enough::Unstoppable;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let w = 256;
    let h = 256;

    // Simple gradient image
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            data[idx] = x as u8;
            data[idx + 1] = y as u8;
            data[idx + 2] = 128;
        }
    }

    // Encode
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();

    println!("JPEG size: {} bytes", jpeg.len());

    // Save for external inspection
    std::fs::write("/tmp/quick_parity_test.jpg", &jpeg).unwrap();
    println!("Saved to /tmp/quick_parity_test.jpg");

    // Decode with zune-jpeg
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);

    let decoded = {
        let cursor = ZCursor::new(&jpeg);
        let mut dec = JpegDecoder::new_with_options(cursor, opts);
        dec.decode().unwrap()
    };

    println!("Decoded {} bytes", decoded.len());

    // Compare with original (accounting for lossy compression)
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (orig, dec) in data.iter().zip(decoded.iter()) {
        let d = (*orig as i32 - *dec as i32).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let avg_diff = sum_diff as f64 / data.len() as f64;

    println!(
        "Original vs decoded: max_diff={}, avg_diff={:.4}",
        max_diff, avg_diff
    );

    // Check first few pixels
    println!("First 9 pixels original: {:?}", &data[0..27]);
    println!("First 9 pixels decoded:  {:?}", &decoded[0..27]);

    // Sanity check - q90 compression should be within ~10 for most pixels
    if max_diff > 30 {
        println!("WARNING: Large difference detected, may indicate encoding issue");
    } else {
        println!("✓ Output looks reasonable for quality 90 compression");
    }
}
