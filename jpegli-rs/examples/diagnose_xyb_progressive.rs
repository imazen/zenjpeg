//! Diagnose XYB Progressive quality issue (SSIM2 -102 instead of +3)

use jpegli::{Encoder, PixelFormat};

fn main() {
    // Small test image
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = 128;
        }
    }

    println!("Testing XYB encoding quality (64x64 gradient at Q70)");

    // XYB Baseline (works correctly)
    let xyb_baseline = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(70.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    println!("\nXYB Baseline: {} bytes", xyb_baseline.len());

    // Decode and check quality
    if let Ok(decoded) = decode_zune(&xyb_baseline[..]) {
        println!("  Decoded successfully: {} bytes", decoded.len());
        println!("  First 12 pixels (RGB):");
        for i in 0..4 {
            let idx = i * 3;
            println!(
                "    Pixel {}: R={:3} G={:3} B={:3} (orig: R={:3} G={:3} B={:3})",
                i,
                decoded[idx],
                decoded[idx + 1],
                decoded[idx + 2],
                data[idx],
                data[idx + 1],
                data[idx + 2]
            );
        }
    } else {
        println!("  ⚠️  Failed to decode!");
    }

    // XYB Progressive (broken)
    let xyb_progressive = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(70.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    println!("\nXYB Progressive: {} bytes", xyb_progressive.len());

    // Decode and check quality
    match decode_zune(&xyb_progressive[..]) {
        Ok(decoded) => {
            println!("  Decoded successfully: {} bytes", decoded.len());
            println!("  First 12 pixels (RGB):");
            for i in 0..4 {
                let idx = i * 3;
                println!(
                    "    Pixel {}: R={:3} G={:3} B={:3} (orig: R={:3} G={:3} B={:3})",
                    i,
                    decoded[idx],
                    decoded[idx + 1],
                    decoded[idx + 2],
                    data[idx],
                    data[idx + 1],
                    data[idx + 2]
                );
            }

            // Check if decoded values are completely wrong
            let mut huge_errors = 0;
            for i in 0..decoded.len() {
                let diff = (decoded[i] as i32 - data[i] as i32).abs();
                if diff > 100 {
                    huge_errors += 1;
                }
            }

            if huge_errors > 0 {
                println!(
                    "  ⚠️  {} pixels have errors > 100 (out of {}) - SEVERE CORRUPTION!",
                    huge_errors,
                    decoded.len()
                );
            }
        }
        Err(e) => {
            println!("  ⚠️  Failed to decode: {:?}", e);
        }
    }

    // Save both for manual inspection
    std::fs::write("/tmp/xyb_baseline.jpg", xyb_baseline).ok();
    std::fs::write("/tmp/xyb_progressive.jpg", xyb_progressive).ok();

    println!("\nFiles saved to /tmp/ for manual inspection");
    println!("  XYB Baseline:    /tmp/xyb_baseline.jpg");
    println!("  XYB Progressive: /tmp/xyb_progressive.jpg");
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
