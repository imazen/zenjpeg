//! Test XYB mode edge handling with partial MCU dimensions

use enough::Unstoppable;
use fast_ssim2::{compute_frame_ssimulacra2, srgb_u8_to_linear, LinearRgbImage};
use jpegli::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = (64.0 + (x as f32 / width as f32) * 128.0) as u8;
            rgb[idx + 1] = (64.0 + (y as f32 / height as f32) * 128.0) as u8;
            rgb[idx + 2] = (64.0 + ((x + y) as f32 / (width + height) as f32) * 128.0) as u8;
        }
    }
    rgb
}

fn encode_xyb(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::new()
        .quality(quality)
        .ycbcr(ChromaSubsampling::Full) // XYB uses custom subsampling internally
        .optimize_huffman(true)
        .xyb();
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(rgb, Unstoppable).expect("push");
    enc.finish().expect("XYB encode failed")
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

fn main() {
    println!("XYB Mode Edge Handling Test");
    println!("============================\n");

    // XYB uses 16x16 MCUs for the B channel (2x2 downsampled)
    let mcu_size = 16;
    let base_width = 256;
    let base_height = 128;
    let quality = 85.0;

    println!("Testing partial MCU dimensions with XYB mode");
    println!("MCU size for B channel: {}x{}", mcu_size, mcu_size);
    println!();

    // Test width edge cases
    println!("=== WIDTH EDGE CASES ===");
    println!(
        "{:>8} {:>8} {:>10} {:>10} {:>8}",
        "Width", "W%16", "Size", "SSIM2", "Status"
    );
    println!("{}", "-".repeat(50));

    let mut all_pass = true;
    for remainder in [1, 4, 7, 15] {
        let width = base_width + remainder;
        let height = base_height;
        let rgb = create_test_image(width, height);

        match std::panic::catch_unwind(|| encode_xyb(&rgb, width as u32, height as u32, quality)) {
            Ok(jpeg) => {
                let decoded = jpegli::Decoder::new().decode(&jpeg).expect("decode failed");
                let ssim2 = compute_ssim2(&rgb, &decoded.data, width, height);
                let status = if ssim2 >= 85.0 { "OK" } else { "FAIL" };
                if ssim2 < 85.0 {
                    all_pass = false;
                }
                println!(
                    "{:>8} {:>8} {:>10} {:>10.2} {:>8}",
                    width,
                    width % 16,
                    jpeg.len(),
                    ssim2,
                    status
                );
            }
            Err(_) => {
                println!(
                    "{:>8} {:>8} {:>10} {:>10} {:>8}",
                    width,
                    width % 16,
                    "-",
                    "-",
                    "PANIC"
                );
                all_pass = false;
            }
        }
    }

    println!();

    // Test height edge cases
    println!("=== HEIGHT EDGE CASES ===");
    println!(
        "{:>8} {:>8} {:>10} {:>10} {:>8}",
        "Height", "H%16", "Size", "SSIM2", "Status"
    );
    println!("{}", "-".repeat(50));

    for remainder in [1, 4, 7, 15] {
        let width = base_width;
        let height = base_height + remainder;
        let rgb = create_test_image(width, height);

        match std::panic::catch_unwind(|| encode_xyb(&rgb, width as u32, height as u32, quality)) {
            Ok(jpeg) => {
                let decoded = jpegli::Decoder::new().decode(&jpeg).expect("decode failed");
                let ssim2 = compute_ssim2(&rgb, &decoded.data, width, height);
                let status = if ssim2 >= 85.0 { "OK" } else { "FAIL" };
                if ssim2 < 85.0 {
                    all_pass = false;
                }
                println!(
                    "{:>8} {:>8} {:>10} {:>10.2} {:>8}",
                    height,
                    height % 16,
                    jpeg.len(),
                    ssim2,
                    status
                );
            }
            Err(_) => {
                println!(
                    "{:>8} {:>8} {:>10} {:>10} {:>8}",
                    height,
                    height % 16,
                    "-",
                    "-",
                    "PANIC"
                );
                all_pass = false;
            }
        }
    }

    println!();

    // Test corner cases (both edges partial)
    println!("=== CORNER CASES ===");
    println!(
        "{:>12} {:>8} {:>8} {:>10} {:>10} {:>8}",
        "WxH", "W%16", "H%16", "Size", "SSIM2", "Status"
    );
    println!("{}", "-".repeat(60));

    for &w_rem in &[1, 7, 15] {
        for &h_rem in &[1, 7, 15] {
            let width = base_width + w_rem;
            let height = base_height + h_rem;
            let rgb = create_test_image(width, height);

            match std::panic::catch_unwind(|| {
                encode_xyb(&rgb, width as u32, height as u32, quality)
            }) {
                Ok(jpeg) => {
                    let decoded = jpegli::Decoder::new().decode(&jpeg).expect("decode failed");
                    let ssim2 = compute_ssim2(&rgb, &decoded.data, width, height);
                    let status = if ssim2 >= 85.0 { "OK" } else { "FAIL" };
                    if ssim2 < 85.0 {
                        all_pass = false;
                    }
                    println!(
                        "{:>12} {:>8} {:>8} {:>10} {:>10.2} {:>8}",
                        format!("{}x{}", width, height),
                        width % 16,
                        height % 16,
                        jpeg.len(),
                        ssim2,
                        status
                    );
                }
                Err(_) => {
                    println!(
                        "{:>12} {:>8} {:>8} {:>10} {:>10} {:>8}",
                        format!("{}x{}", width, height),
                        width % 16,
                        height % 16,
                        "-",
                        "-",
                        "PANIC"
                    );
                    all_pass = false;
                }
            }
        }
    }

    println!();
    if all_pass {
        println!("PASS: All XYB edge cases passed");
    } else {
        println!("FAIL: Some XYB edge cases failed");
        std::process::exit(1);
    }
}
