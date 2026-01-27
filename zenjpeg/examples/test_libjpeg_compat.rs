use enough::Unstoppable;
use std::process::Command;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn test_pattern(name: &str, data: &[u8], width: u32, height: u32, quality: f32) {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .optimize_huffman(true)
        .progressive(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(data, Unstoppable).expect("push");
    let jpeg_data = enc.finish().expect("Encoding should succeed");

    let path = format!("/tmp/test_{}.jpg", name);
    std::fs::write(&path, &jpeg_data).unwrap();

    // Test with djpeg
    let output = Command::new("djpeg")
        .args(["-outfile", "/dev/null", &path])
        .output()
        .expect("djpeg should run");

    let status = if output.status.success() {
        "OK"
    } else {
        "FAIL"
    };

    println!(
        "{:20} Q{:<3} {:6} bytes - djpeg: {}",
        name,
        quality as i32,
        jpeg_data.len(),
        status
    );

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.is_empty() {
            println!("  Error: {}", stderr.lines().next().unwrap_or(""));
        }
    }
}

fn main() {
    println!("Testing various patterns with djpeg...\n");

    // Test simple patterns
    for q in [30.0, 50.0, 75.0, 90.0] {
        // Solid color
        let solid: Vec<u8> = (0..64 * 64).flat_map(|_| [128, 128, 128]).collect();
        test_pattern("solid", &solid, 64, 64, q);

        // Gradient
        let gradient: Vec<u8> = (0..64)
            .flat_map(|y| (0..64).flat_map(move |x| [(x * 4) as u8, (y * 4) as u8, 128]))
            .collect();
        test_pattern("gradient", &gradient, 64, 64, q);

        // Checkerboard
        let check: Vec<u8> = (0..64)
            .flat_map(|y| {
                (0..64).flat_map(move |x| {
                    let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                    [v, v, v]
                })
            })
            .collect();
        test_pattern("checkerboard", &check, 64, 64, q);

        // Noise-like (the failing pattern)
        let noise: Vec<u8> = (0..64)
            .flat_map(|y| {
                (0..64).flat_map(move |x| {
                    let r = (((x * 17) ^ (y * 31)) % 256) as u8;
                    let g = (((x * 13) ^ (y * 23)) % 256) as u8;
                    let b = (((x * 11) ^ (y * 19)) % 256) as u8;
                    [r, g, b]
                })
            })
            .collect();
        test_pattern("noise64", &noise, 64, 64, q);

        println!();
    }

    // Test larger sizes with noise
    println!("Testing larger noise images...\n");
    for size in [32, 64, 96, 128] {
        let noise: Vec<u8> = (0..size)
            .flat_map(|y| {
                (0..size).flat_map(move |x| {
                    let r = (((x * 17) ^ (y * 31)) % 256) as u8;
                    let g = (((x * 13) ^ (y * 23)) % 256) as u8;
                    let b = (((x * 11) ^ (y * 19)) % 256) as u8;
                    [r, g, b]
                })
            })
            .collect();
        let name = format!("noise{}x{}", size, size);
        test_pattern(&name, &noise, size as u32, size as u32, 50.0);
    }

    // Test grayscale
    println!("\nTesting grayscale...\n");
    let gray_noise: Vec<u8> = (0..64)
        .flat_map(|y| (0..64).map(move |x| (((x * 17) ^ (y * 31)) % 256) as u8))
        .collect();

    let gray_config = EncoderConfig::grayscale(50.0)
        .optimize_huffman(true)
        .progressive(true);
    let mut gray_enc = gray_config
        .encode_from_bytes(64, 64, PixelLayout::Gray8Srgb)
        .expect("encoder setup");
    gray_enc
        .push_packed(&gray_noise, Unstoppable)
        .expect("push");
    let gray_jpeg = gray_enc.finish().expect("Encoding should succeed");

    let path = "/tmp/test_gray64.jpg";
    std::fs::write(path, &gray_jpeg).unwrap();

    let output = Command::new("djpeg")
        .args(["-outfile", "/dev/null", path])
        .output()
        .expect("djpeg should run");

    println!(
        "grayscale 64x64 Q50: {} bytes - djpeg: {}",
        gray_jpeg.len(),
        if output.status.success() {
            "OK"
        } else {
            "FAIL"
        }
    );
}
