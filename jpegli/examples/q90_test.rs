use jpegli::{Encoder, Quality};
use jpegli::types::{JpegMode, PixelFormat};
use std::process::Command;

fn main() {
    // Load test image
    let test_img = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";
    let png_data = std::fs::read(test_img).unwrap();
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let width = info.width;
    let height = info.height;
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported"),
    };

    println!("=== Q90 Comparison ===");
    
    // Rust Q90
    let rust_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Baseline)
        .encode(&rgb)
        .expect("encode");
    
    // C++ Q90
    std::fs::write("/tmp/q90_input.png", &png_data).unwrap();
    let status = Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
        .args(["/tmp/q90_input.png", "/tmp/q90_cpp.jpg", "-q", "90", "--progressive_level=0"])
        .output()
        .unwrap();
    assert!(status.status.success());
    let cpp_jpeg = std::fs::read("/tmp/q90_cpp.jpg").unwrap();
    
    let size_diff = (rust_jpeg.len() as f64 / cpp_jpeg.len() as f64 - 1.0) * 100.0;
    println!("  Rust: {} bytes", rust_jpeg.len());
    println!("  C++:  {} bytes", cpp_jpeg.len());
    println!("  Size diff: {:+.1}%", size_diff);
    
    std::fs::write("/tmp/q90_rust.jpg", &rust_jpeg).unwrap();
}
