use jpegli::encode::Encoder;
use jpegli::quant::Quality;
use jpegli::types::JpegMode;

fn main() {
    let width = 64u32;
    let height = 64u32;
    let noise_mul = 13u32;

    // Generate RGB image
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let noise = ((x * noise_mul + y * noise_mul) % 64) as u8;
            rgb.push(((x * 4) as u8).wrapping_add(noise));
            rgb.push(((y * 4) as u8).wrapping_add(noise / 2));
            rgb.push(128u8.wrapping_add(noise));
        }
    }

    // Save as PNG for C++ to use
    let png_path = "/tmp/test_noise13.png";
    let file = std::fs::File::create(png_path).unwrap();
    let ref mut w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&rgb).unwrap();
    println!("Saved PNG: {}", png_path);

    // Encode with Rust jpegli
    let result = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(10.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&rgb)
        .unwrap();

    std::fs::write("/tmp/rust_noise13.jpg", &result).unwrap();
    println!("Rust JPEG: {} bytes", result.len());
}
