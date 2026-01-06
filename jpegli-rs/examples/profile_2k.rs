//! Simple 2K profiling target for samply
use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};

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

    eprintln!("Running 20 encode iterations for profiling...");
    for i in 0..20 {
        let _jpeg = Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(90.0))
            .mode(JpegMode::Baseline)
            .optimize_huffman(true)
            .subsampling(Subsampling::S420)
            .use_xyb(false)
            .encode(&rgb)
            .expect("encode failed");
        if i % 5 == 0 {
            eprintln!("  iteration {}/20", i + 1);
        }
    }
    eprintln!("Done!");
}
