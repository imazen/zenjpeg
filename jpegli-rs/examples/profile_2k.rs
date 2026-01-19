//! Simple 2K profiling target for samply
use enough::Unstoppable;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;
            rgb[idx] = ((fx * 255.0) + (fx * fy * 50.0).sin() * 30.0).clamp(0.0, 255.0) as u8;
            rgb[idx + 1] = ((fy * 255.0) + (fx * fy * 100.0).cos() * 40.0).clamp(0.0, 255.0) as u8;
            rgb[idx + 2] = (128.0 + ((fx + fy) * 50.0).sin() * 50.0).clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

fn main() {
    let width = 2048u32;
    let height = 2048u32;

    eprintln!("Generating {}x{} test image...", width, height);
    let rgb = generate_test_image(width as usize, height as usize);

    eprintln!("Running 50 encode iterations (q90 4:4:4 YCbCr for profiling...");
    let config_444 = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
        .progressive(false)
        .optimize_huffman(true);
    for i in 0..50 {
        let mut enc = config_444
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder setup");
        enc.push_packed(&rgb, Unstoppable).expect("push");
        let _jpeg = enc.finish().expect("encode failed");
        if i % 5 == 0 {
            eprintln!("  iteration {}/50", i + 1);
        }
    }
    eprintln!("Running 50 encode iterations (q70 4:2:0 YCbCr for profiling...");
    let config_420 = EncoderConfig::ycbcr(70.0, ChromaSubsampling::Quarter)
        .progressive(true);
    for i in 0..50 {
        let mut enc = config_420
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder setup");
        enc.push_packed(&rgb, Unstoppable).expect("push");
        let _jpeg = enc.finish().expect("encode failed");
        if i % 5 == 0 {
            eprintln!("  iteration {}/50", i + 1);
        }
    }
    eprintln!("Done!");
}
