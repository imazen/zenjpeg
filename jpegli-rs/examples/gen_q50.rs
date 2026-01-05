use jpegli::quant::Quality;
use jpegli::types::JpegMode;
use jpegli::{Encoder, PixelFormat};

fn main() {
    let noise: Vec<u8> = (0..64)
        .flat_map(|y| {
            (0..64).flat_map(move |x| {
                let r = ((x * 17 ^ y * 31) % 256) as u8;
                let g = ((x * 13 ^ y * 23) % 256) as u8;
                let b = ((x * 11 ^ y * 19) % 256) as u8;
                [r, g, b]
            })
        })
        .collect();

    let jpeg_data = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(50.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&noise)
        .expect("Encoding should succeed");

    std::fs::write("/tmp/noise64_q50.jpg", &jpeg_data).unwrap();
    println!("Wrote {} bytes to /tmp/noise64_q50.jpg", jpeg_data.len());
}
