// Test roundtrip: encode -> decode -> compare
use jpegli::{types::JpegMode, Decoder, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn mse(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as f64 - y as f64;
            diff * diff
        })
        .sum();
    sum / a.len() as f64
}

fn main() {
    for size in [49u32, 50, 51, 52, 53] {
        let original = photo_like(size, size);

        // First test baseline
        let jpeg_baseline = Encoder::new()
            .width(size)
            .height(size)
            .pixel_format(PixelFormat::Gray)
            .quality(Quality::from_quality(75.0))
            .mode(JpegMode::Baseline)
            .encode(&original)
            .expect("baseline encode failed");

        let baseline_decoded = match Decoder::new().decode(&jpeg_baseline) {
            Ok(result) => {
                let bytes_per_pixel = match result.format {
                    PixelFormat::Rgb | PixelFormat::Bgr => 3,
                    PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Cmyk => 4,
                    PixelFormat::Gray => 1,
                };
                if bytes_per_pixel == 1 {
                    result.data
                } else {
                    result
                        .data
                        .iter()
                        .step_by(bytes_per_pixel)
                        .copied()
                        .collect()
                }
            }
            Err(e) => {
                println!(
                    "{}x{}: OUR DECODER FAILED on baseline - {:?}",
                    size, size, e
                );
                continue;
            }
        };

        let baseline_mse = mse(&original, &baseline_decoded);

        // Encode with progressive
        let jpeg = Encoder::new()
            .width(size)
            .height(size)
            .pixel_format(PixelFormat::Gray)
            .quality(Quality::from_quality(75.0))
            .optimize_huffman(false)
            .mode(JpegMode::Progressive)
            .encode(&original)
            .expect("encode failed");

        // Decode with our decoder
        let decoded = match Decoder::new().decode(&jpeg) {
            Ok(result) => {
                // Our decoder returns RGB, extract Y channel
                let bytes_per_pixel = match result.format {
                    PixelFormat::Rgb | PixelFormat::Bgr => 3,
                    PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Cmyk => 4,
                    PixelFormat::Gray => 1,
                };
                if bytes_per_pixel == 1 {
                    result.data
                } else {
                    // RGB or RGBA - just take first channel (should be R=G=B for grayscale)
                    result
                        .data
                        .iter()
                        .step_by(bytes_per_pixel)
                        .copied()
                        .collect()
                }
            }
            Err(e) => {
                println!("{}x{}: OUR DECODER FAILED - {:?}", size, size, e);
                continue;
            }
        };

        // Compare MSE
        let error = mse(&original, &decoded);

        // Also try jpeg_decoder
        let jpeg_decoder_ok = jpeg_decoder::Decoder::new(&jpeg[..]).decode().is_ok();

        println!(
            "{}x{}: baseline_MSE={:.2}, prog_MSE={:.2}, jpeg_decoder={}",
            size,
            size,
            baseline_mse,
            error,
            if jpeg_decoder_ok { "OK" } else { "FAIL" }
        );

        // If MSE is high, something is wrong with our roundtrip
        if error > 100.0 {
            println!("  WARNING: High MSE indicates decoder mismatch!");
            // Print first few pixel differences
            for i in 0..10.min(decoded.len()) {
                if (original[i] as i16 - decoded[i] as i16).abs() > 10 {
                    println!(
                        "    pixel[{}]: original={}, decoded={}, diff={}",
                        i,
                        original[i],
                        decoded[i],
                        original[i] as i16 - decoded[i] as i16
                    );
                }
            }
        }
    }
}
