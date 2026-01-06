//! Compare XYB vs YCbCr using butteraugli quality metric.
//!
//! Note: This comparison requires proper ICC profile handling for XYB images.
//! Without proper ICC transform, XYB decoded images are in wrong color space.
//!
//! For accurate XYB comparison, decode with an ICC-aware tool like:
//!   - Pillow (Python) with ImageCms
//!   - ImageMagick with -profile option
//!
//! Run with: cargo run --release --example xyb_vs_ycbcr_butteraugli

use butteraugli::{compute_butteraugli, ButteraugliParams};
use std::path::Path;
use std::process::Command;

fn main() {
    let image_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png".to_string());

    // Load image
    let file = std::fs::File::open(&image_path).expect("Failed to open image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    if info.color_type != png::ColorType::Rgb {
        eprintln!("Image must be RGB");
        return;
    }

    let pixels = &buf[..info.buffer_size()];
    let width = info.width as usize;
    let height = info.height as usize;

    let img_name = Path::new(&image_path)
        .file_name()
        .unwrap()
        .to_string_lossy();

    println!("XYB vs YCbCr Comparison (Butteraugli)");
    println!("Image: {} ({}x{})", img_name, width, height);
    println!();
    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}",
        "Quality", "XYB bytes", "YCbCr bytes", "Size diff", "XYB BA", "YCbCr BA", "BA diff"
    );
    println!("{}", "-".repeat(85));

    let qualities = [50, 60, 70, 80, 85, 90, 95];

    for &q in &qualities {
        // Encode XYB
        let xyb_jpeg = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .jpegli_quality(jpegli::quant::Quality::from_quality(q as f32))
            .use_xyb(true)
            .encode(pixels)
            .expect("XYB encode");
        let xyb_bytes = xyb_jpeg.len();

        // Encode YCbCr
        let ycbcr_jpeg = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .jpegli_quality(jpegli::quant::Quality::from_quality(q as f32))
            .use_xyb(false)
            .encode(pixels)
            .expect("YCbCr encode");
        let ycbcr_bytes = ycbcr_jpeg.len();

        // Decode both
        let xyb_decoded = decode_jpeg(&xyb_jpeg);
        let ycbcr_decoded = decode_jpeg(&ycbcr_jpeg);

        // Compute butteraugli
        let xyb_ba = compute_butteraugli_score(pixels, &xyb_decoded, width, height);
        let ycbcr_ba = compute_butteraugli_score(pixels, &ycbcr_decoded, width, height);

        let size_diff = 100.0 * (xyb_bytes as f64 - ycbcr_bytes as f64) / ycbcr_bytes as f64;
        let ba_diff = xyb_ba - ycbcr_ba;

        println!(
            "{:<8} {:<12} {:<12} {:+.1}%{:<7} {:<12.4} {:<10.4} {:+.4}",
            q, xyb_bytes, ycbcr_bytes, size_diff, "", xyb_ba, ycbcr_ba, ba_diff
        );
    }

    println!();
    println!("Note: Lower butteraugli = better quality");
    println!("      Negative BA diff = XYB has better quality");
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    // Use jpegli decoder which applies ICC profiles for XYB images
    let decoder = jpegli::decode::Decoder::new().apply_icc(true);
    match decoder.decode(data) {
        Ok(img) => {
            // eprintln!("  jpegli decode OK, {} bytes", img.data.len());
            img.data
        }
        Err(e) => {
            eprintln!("  jpegli decode failed: {:?}, using jpeg-decoder", e);
            // Fallback to jpeg-decoder if jpegli decode fails
            let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
            decoder.decode().expect("decode")
        }
    }
}

fn compute_butteraugli_score(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let params = ButteraugliParams::default();
    match compute_butteraugli(original, decoded, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => f64::NAN,
    }
}
