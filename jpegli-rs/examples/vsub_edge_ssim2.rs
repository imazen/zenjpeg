//! Compare vertical edge replication strategies using SSIMULACRA2
use fast_ssim2::{compute_frame_ssimulacra2, srgb_u8_to_linear, LinearRgbImage};
use jpegli::Quality;
#[allow(deprecated)]
use jpegli::{Encoder, PixelFormat, Subsampling};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let debug = args.iter().any(|a| a == "--debug");

    println!("Vertical edge replication test - S420 (16-pixel MCU)");
    println!("Comparing Strip vs Fullplane encoder\n");
    println!(
        "{:>6} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "Height", "H%16", "Pad", "Strip", "SSIM2", "Full", "SSIM2"
    );
    println!("{}", "-".repeat(80));

    let width = 64usize;

    for height in 56..=72 {
        let pad_rows = (16 - (height % 16)) % 16;

        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                rgb[idx] = ((x * 4) % 256) as u8;
                rgb[idx + 1] = ((y * 4) % 256) as u8;
                rgb[idx + 2] = (((x + y) * 2) % 256) as u8;
            }
        }

        // Strip encoder (default for deprecated Encoder)
        #[allow(deprecated)]
        let jpeg_strip = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .subsampling(Subsampling::S420)
            .jpegli_quality(Quality::from_quality(85.0))
            .optimize_huffman(true)
            .encode(&rgb)
            .unwrap();

        // Fullplane encoder
        #[allow(deprecated)]
        let jpeg_full = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .subsampling(Subsampling::S420)
            .jpegli_quality(Quality::from_quality(85.0))
            .optimize_huffman(true)
            .encode_fullplane(&rgb)
            .unwrap();

        let decoded_strip = jpegli::Decoder::new().decode(&jpeg_strip).unwrap();
        let decoded_full = jpegli::Decoder::new().decode(&jpeg_full).unwrap();

        // Debug output for specific heights
        if debug && (height == 64 || height == 65) {
            println!("\n=== Height {} debug ===", height);
            // Show bottom edge pixels
            for y in height.saturating_sub(3)..height {
                let x = 0;
                let orig_idx = (y * width + x) * 3;
                let dec_idx = orig_idx;
                println!(
                    "  Row {}: orig=({},{},{}) dec=({},{},{})",
                    y,
                    rgb[orig_idx],
                    rgb[orig_idx + 1],
                    rgb[orig_idx + 2],
                    decoded_strip.data[dec_idx],
                    decoded_strip.data[dec_idx + 1],
                    decoded_strip.data[dec_idx + 2]
                );
            }
            // Show max diff with location
            let mut max_diff = 0i32;
            let mut max_y = 0;
            let mut max_x = 0;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    for c in 0..3 {
                        let diff = (rgb[idx + c] as i32 - decoded_strip.data[idx + c] as i32).abs();
                        if diff > max_diff {
                            max_diff = diff;
                            max_y = y;
                            max_x = x;
                        }
                    }
                }
            }
            println!("  Max diff: {} at ({},{})", max_diff, max_x, max_y);

            // Save the JPEG for external analysis
            let filename = format!("/tmp/test_h{}.jpg", height);
            std::fs::write(&filename, &jpeg_strip).unwrap();
            println!("  Saved to: {}", filename);

            // Try decoding with libjpeg as well
            let libjpeg_decoded = std::process::Command::new("identify")
                .args(["-verbose", &filename])
                .output();
            if let Ok(output) = libjpeg_decoded {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.contains("Geometry:") || line.contains("Type:") {
                        println!("  ImageMagick: {}", line.trim());
                    }
                }
            }
        }

        let ssim2_strip = if decoded_strip.data.len() == width * height * 3 {
            compute_ssim2(&rgb, &decoded_strip.data, width, height)
        } else {
            -1.0
        };

        let ssim2_full = if decoded_full.data.len() == width * height * 3 {
            compute_ssim2(&rgb, &decoded_full.data, width, height)
        } else {
            -1.0
        };

        let marker = if ssim2_strip < 90.0 && ssim2_full >= 90.0 {
            " <-- STRIP BUG"
        } else if ssim2_strip < 90.0 {
            " <-- BAD"
        } else {
            ""
        };

        println!(
            "{:>6} {:>8} {:>6} {:>10} {:>10.2} {:>10} {:>10.2}{}",
            height,
            height % 16,
            pad_rows,
            jpeg_strip.len(),
            ssim2_strip,
            jpeg_full.len(),
            ssim2_full,
            marker
        );
    }
}

fn bytes_to_linear(data: &[u8], width: usize, height: usize) -> LinearRgbImage {
    let pixels: Vec<[f32; 3]> = data
        .chunks_exact(3)
        .map(|rgb| {
            [
                srgb_u8_to_linear(rgb[0]),
                srgb_u8_to_linear(rgb[1]),
                srgb_u8_to_linear(rgb[2]),
            ]
        })
        .collect();
    LinearRgbImage::new(pixels, width, height)
}

fn compute_ssim2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let orig = bytes_to_linear(original, width, height);
    let dec = bytes_to_linear(decoded, width, height);
    compute_frame_ssimulacra2(orig, dec).unwrap_or(0.0)
}
