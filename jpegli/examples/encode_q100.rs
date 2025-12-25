use jpegli::types::{JpegMode, PixelFormat};
use jpegli::{Encoder, Quality};
use std::fs;

fn main() {
    // Load PNG from argument or default
    let png_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/realistic_test.png".to_string());

    let png_data = fs::read(&png_path).expect("read png");
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().expect("read info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("read frame");

    let width = info.width;
    let height = info.height;

    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    println!("Image: {}x{} from {}", width, height, png_path);

    // Encode at Q100 with progressive like C++
    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(100.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&rgb)
        .expect("encode");

    fs::write("/tmp/q100_rust.jpg", &jpeg).expect("write");
    println!("Rust Q100: {} bytes", jpeg.len());
}
