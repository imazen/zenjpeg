//! Debug progressive + subsampling issue
//!
//! Tests which progressive + subsampling combinations produce valid output

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::fs;

fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            data.push((x * 255 / width) as u8); // R
            data.push((y * 255 / height) as u8); // G
            data.push(128); // B
        }
    }
    data
}

fn decode_with_mozjpeg(data: &[u8]) -> Result<Vec<u8>, String> {
    let decompress =
        mozjpeg::Decompress::new_mem(data).map_err(|e| format!("decompress: {:?}", e))?;
    let mut decompressor = decompress.rgb().map_err(|e| format!("rgb: {:?}", e))?;
    let width = decompressor.width();
    let height = decompressor.height();
    let mut pixels = vec![0u8; width * height * 3];
    #[allow(deprecated)]
    let _ = decompressor.read_scanlines_into::<u8>(&mut pixels);
    Ok(pixels)
}

fn decode_with_jpegli(data: &[u8]) -> Result<Vec<u8>, String> {
    jpegli::Decoder::new()
        .decode(data)
        .map(|img| img.data)
        .map_err(|e| format!("{:?}", e))
}

fn decode_with_zune(data: &[u8]) -> Result<Vec<u8>, String> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().map_err(|e| format!("{:?}", e))
}

fn test_config(
    width: u32,
    height: u32,
    mode: JpegMode,
    subsampling: Subsampling,
    quality: u8,
) -> (usize, bool, bool, bool) {
    let rgb = generate_gradient(width as usize, height as usize);

    let jpeg = match Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .mode(mode)
        .subsampling(subsampling)
        .optimize_huffman(true)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .encode(&rgb)
    {
        Ok(j) => j,
        Err(e) => {
            println!("  ENCODE FAILED: {}", e);
            return (0, false, false, false);
        }
    };

    let moz_ok = decode_with_mozjpeg(&jpeg).is_ok();
    let jpegli_ok = decode_with_jpegli(&jpeg).is_ok();
    let zune_ok = decode_with_zune(&jpeg).is_ok();

    (jpeg.len(), moz_ok, jpegli_ok, zune_ok)
}

fn main() {
    println!("\n=== Progressive + Subsampling Debug ===\n");

    let width = 256;
    let height = 256;
    let quality = 85;

    let modes = [
        ("Baseline", JpegMode::Baseline),
        ("Progressive", JpegMode::Progressive),
    ];

    let subsamplings = [
        ("444", Subsampling::S444),
        ("422", Subsampling::S422),
        ("420", Subsampling::S420),
        ("440", Subsampling::S440),
    ];

    println!(
        "{:<12} {:<6} {:>8} {:>8} {:>8} {:>8}",
        "Mode", "Sub", "Size", "mozjpeg", "jpegli", "zune"
    );
    println!("{:-<60}", "");

    for (mode_name, mode) in &modes {
        for (sub_name, sub) in &subsamplings {
            let (size, moz, jpegli, zune) = test_config(width, height, *mode, *sub, quality);

            let status = |ok: bool| if ok { "✓" } else { "✗" };

            println!(
                "{:<12} {:<6} {:>8} {:>8} {:>8} {:>8}",
                mode_name,
                sub_name,
                size,
                status(moz),
                status(jpegli),
                status(zune)
            );

            // If any decoder fails, save the JPEG for inspection
            if !moz || !jpegli || !zune {
                let filename = format!("/tmp/broken_{}_{}.jpg", mode_name.to_lowercase(), sub_name);
                let rgb = generate_gradient(width as usize, height as usize);
                if let Ok(jpeg) = Encoder::new()
                    .width(width)
                    .height(height)
                    .pixel_format(PixelFormat::Rgb)
                    .mode(*mode)
                    .subsampling(*sub)
                    .optimize_huffman(true)
                    .jpegli_quality(Quality::from_quality(quality as f32))
                    .encode(&rgb)
                {
                    let _ = fs::write(&filename, &jpeg);
                    println!("  -> Saved to {}", filename);

                    // Try to get more details
                    if let Err(e) = decode_with_mozjpeg(&jpeg) {
                        println!("  -> mozjpeg error: {}", e);
                    }
                    if let Err(e) = decode_with_jpegli(&jpeg) {
                        println!("  -> jpegli error: {}", e);
                    }
                    if let Err(e) = decode_with_zune(&jpeg) {
                        println!("  -> zune error: {}", e);
                    }
                }
            }
        }
    }

    println!("\n=== Size comparison (should be similar for same subsampling) ===\n");

    for (sub_name, sub) in &subsamplings {
        let (baseline_size, _, _, _) =
            test_config(width, height, JpegMode::Baseline, *sub, quality);
        let (prog_size, _, _, _) = test_config(width, height, JpegMode::Progressive, *sub, quality);

        let diff_pct = if baseline_size > 0 {
            (prog_size as f64 - baseline_size as f64) / baseline_size as f64 * 100.0
        } else {
            0.0
        };

        println!(
            "{}: Baseline={} bytes, Progressive={} bytes ({:+.1}%)",
            sub_name, baseline_size, prog_size, diff_pct
        );
    }
}
