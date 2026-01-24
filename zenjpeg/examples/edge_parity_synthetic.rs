//! Test edge padding with a synthetic image that has varied edge content

use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::fs;

fn save_png(path: &str, rgb: &[u8], width: usize, height: usize) {
    use std::io::BufWriter;
    let file = fs::File::create(path).expect("Failed to create file");
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("Failed to write header");
    writer.write_image_data(rgb).expect("Failed to write data");
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("Decode failed")
}

fn compute_psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn create_gradient_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Diagonal gradient with some noise/variation
            let r = ((x * 255 / width) + (y * 128 / height)) as u8;
            let g = ((y * 255 / height) + (x * 64 / width)) as u8;
            let b = (((x + y) * 128 / (width + height)) + ((x * y) % 128)) as u8;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }
    pixels
}

fn main() {
    // Create a gradient image with non-8-aligned dimensions
    // 67 = 8*8 + 3 (3 partial columns)
    // 71 = 8*8 + 7 (7 partial rows)
    let width = 67;
    let height = 71;

    println!("=== Edge Padding Parity Test (Synthetic Gradient) ===\n");
    println!(
        "Image: {}x{} (partial MCU: {} cols, {} rows)",
        width,
        height,
        width % 8,
        height % 8
    );

    let pixels = create_gradient_image(width, height);

    // Save source
    save_png("/mnt/v/gradient_source.png", &pixels, width, height);
    println!("Saved source to /mnt/v/gradient_source.png");

    // Test at different quality levels with baseline mode
    println!("\n{:>7} | {:>10} | {:>10}", "Quality", "Size", "PSNR");
    println!("{}", "-".repeat(35));

    for quality in [50, 75, 90, 95] {
        let config = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::None)
            .progressive(false)
            .optimize_huffman(true);
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .expect("encoder setup");
        enc.push_packed(&pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("Encode failed");

        let decoded = decode_jpeg(&jpeg);
        let psnr = compute_psnr(&pixels, &decoded);

        println!(
            "{:>7} | {:>10} | {:>10.2}",
            format!("q{}", quality),
            jpeg.len(),
            psnr
        );

        // Save q75 and q90
        if quality == 75 {
            fs::write("/mnt/v/gradient_q75.jpg", &jpeg).ok();
            save_png("/mnt/v/gradient_decoded_q75.png", &decoded, width, height);
        }
        if quality == 90 {
            fs::write("/mnt/v/gradient_q90.jpg", &jpeg).ok();
            save_png("/mnt/v/gradient_decoded_q90.png", &decoded, width, height);
        }
    }

    // Also test with 4:2:0
    println!("\n=== 4:2:0 Subsampling (16x16 MCU) ===");
    println!("Partial MCU: {} cols, {} rows", width % 16, height % 16);
    println!("\n{:>7} | {:>10} | {:>10}", "Quality", "Size", "PSNR");
    println!("{}", "-".repeat(35));

    for quality in [50, 75, 90, 95] {
        let config = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter)
            .progressive(false)
            .optimize_huffman(true);
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .expect("encoder setup");
        enc.push_packed(&pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("Encode failed");

        let decoded = decode_jpeg(&jpeg);
        let psnr = compute_psnr(&pixels, &decoded);

        println!(
            "{:>7} | {:>10} | {:>10.2}",
            format!("q{}", quality),
            jpeg.len(),
            psnr
        );
    }

    println!("\n=== Progressive Mode ===");
    for quality in [50, 75, 90, 95] {
        let config = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::None)
            .progressive(true)
            .optimize_huffman(true);
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .expect("encoder setup");
        enc.push_packed(&pixels, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("Encode failed");

        let decoded = decode_jpeg(&jpeg);
        let psnr = compute_psnr(&pixels, &decoded);

        println!(
            "{:>7} | {:>10} | {:>10.2}",
            format!("q{}", quality),
            jpeg.len(),
            psnr
        );
    }

    println!("\nImages saved to /mnt/v/");
}
