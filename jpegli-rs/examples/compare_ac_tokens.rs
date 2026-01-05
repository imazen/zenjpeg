//! Compare AC refinement tokens between Rust and C++

use jpegli::encode::Encoder;
use jpegli::quant::Quality;
use jpegli::types::JpegMode;
use std::fs;

fn generate_texture_fine(size: usize) -> Vec<u8> {
    let scale = 0.1f32;
    let mut rgb = vec![0u8; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let idx = (y * size + x) * 3;
            let fx = x as f32 * scale;
            let fy = y as f32 * scale;
            let v1 = ((fx.sin() + 1.0) * 127.5) as u8;
            let v2 = ((fy.cos() + 1.0) * 127.5) as u8;
            let v3 = (((fx + fy).sin() + 1.0) * 127.5) as u8;
            rgb[idx] = v1;
            rgb[idx + 1] = v2;
            rgb[idx + 2] = v3;
        }
    }
    rgb
}

fn main() {
    let size = 128;
    let rgb = generate_texture_fine(size);

    let jpeg = Encoder::new()
        .width(size as u32)
        .height(size as u32)
        .jpegli_quality(Quality::Traditional(90.0))
        .mode(JpegMode::Progressive)
        .encode(&rgb)
        .unwrap();

    println!("Rust JPEG size: {} bytes", jpeg.len());
    fs::write("/tmp/rust_texture_test.jpg", &jpeg).unwrap();

    // Write PPM for C++ comparison
    let mut ppm = format!("P6\n{} {}\n255\n", size, size).into_bytes();
    ppm.extend_from_slice(&rgb);
    fs::write("/tmp/rust_texture_ppm.ppm", &ppm).unwrap();
    println!("Wrote PPM for C++ comparison to /tmp/rust_texture_ppm.ppm");
}
