//! Test XYB encode/decode quality with a real image

use jpegli::{Encoder, Decoder, Quality};
use std::fs;

fn main() {
    // Load a test image
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png");
    let png_data = fs::read(png_path).expect("Failed to read PNG");

    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");
    let rgb_data = &buf[..info.buffer_size()];

    println!("Image: {}x{}", info.width, info.height);
    println!("Input RGB sample (first pixel): [{}, {}, {}]", rgb_data[0], rgb_data[1], rgb_data[2]);

    // Encode with XYB at Q90
    let jpeg_xyb = Encoder::new()
        .width(info.width)
        .height(info.height)
        .use_xyb(true)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(rgb_data)
        .expect("XYB encoding failed");

    println!("XYB JPEG size: {} bytes", jpeg_xyb.len());

    // Save for inspection
    fs::write("/tmp/test_xyb.jpg", &jpeg_xyb).ok();
    println!("Saved to /tmp/test_xyb.jpg");

    // Decode with zune-jpeg (external decoder, doesn't know about XYB)
    println!("\nTrying zune-jpeg decoder on XYB JPEG...");
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(&jpeg_xyb[..]);
    let mut zune_dec = JpegDecoder::new(cursor);
    match zune_dec.decode() {
        Ok(zune_data) => {
            let zune_data: Vec<u8> = zune_data;
            println!("zune-jpeg decoded: {} bytes", zune_data.len());
            if zune_data.len() >= 3 {
                println!("zune RGB sample (first pixel): [{}, {}, {}]",
                         zune_data[0], zune_data[1], zune_data[2]);
            }

            // This compares raw DCT output with original - will be way off since
            // zune doesn't know about XYB inverse transform
            let mut max_diff_z = 0i32;
            let mut sum_diff_z = 0u64;
            for i in 0..rgb_data.len().min(zune_data.len()) {
                let diff = (rgb_data[i] as i32 - zune_data[i] as i32).abs();
                max_diff_z = max_diff_z.max(diff);
                sum_diff_z += diff as u64;
            }
            let avg_diff_z = sum_diff_z as f64 / rgb_data.len() as f64;
            println!("zune XYB (no inverse XYB) max diff: {}, avg diff: {:.2}", max_diff_z, avg_diff_z);
            println!("NOTE: High diff expected since zune doesn't apply inverse XYB transform!");
        }
        Err(e) => println!("zune-jpeg failed: {:?}", e),
    }

    // Try our decoder
    println!("\nTrying jpegli-rs decoder...");
    let jpegli_decoder = Decoder::new();
    match jpegli_decoder.decode(&jpeg_xyb) {
        Ok(decoded) => {
            println!("jpegli-rs decoded: {} bytes", decoded.data.len());
            println!("Decoded dimensions: {}x{}", decoded.width, decoded.height);
            println!("Decoded RGB sample (first pixel): [{}, {}, {}]",
                     decoded.data[0], decoded.data[1], decoded.data[2]);

            let mut max_diff = 0i32;
            let mut sum_diff = 0u64;
            for i in 0..rgb_data.len().min(decoded.data.len()) {
                let diff = (rgb_data[i] as i32 - decoded.data[i] as i32).abs();
                max_diff = max_diff.max(diff);
                sum_diff += diff as u64;
            }
            let avg_diff = sum_diff as f64 / rgb_data.len() as f64;

            println!("\nXYB Quality:");
            println!("  Max pixel diff: {}", max_diff);
            println!("  Avg pixel diff: {:.2}", avg_diff);
        }
        Err(e) => println!("jpegli-rs decoder failed: {:?}", e),
    }

    // Also try YCbCr for comparison
    println!("\n--- YCbCr comparison ---");
    let jpeg_ycbcr = Encoder::new()
        .width(info.width)
        .height(info.height)
        .use_xyb(false)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(rgb_data)
        .expect("YCbCr encoding failed");

    println!("YCbCr JPEG size: {} bytes", jpeg_ycbcr.len());

    let decoder2 = Decoder::new();
    let decoded2 = decoder2.decode(&jpeg_ycbcr).expect("YCbCr decoding failed");

    let mut max_diff2 = 0i32;
    let mut sum_diff2 = 0u64;
    for i in 0..rgb_data.len().min(decoded2.data.len()) {
        let diff = (rgb_data[i] as i32 - decoded2.data[i] as i32).abs();
        max_diff2 = max_diff2.max(diff);
        sum_diff2 += diff as u64;
    }
    let avg_diff2 = sum_diff2 as f64 / rgb_data.len() as f64;

    println!("YCbCr max pixel diff: {}", max_diff2);
    println!("YCbCr avg pixel diff: {:.2}", avg_diff2);
}
