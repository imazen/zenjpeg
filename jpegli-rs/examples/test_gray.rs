use jpegli::encode::Encoder;
use jpegli::quant::Quality;
use jpegli::types::{JpegMode, PixelFormat};

fn main() {
    let width = 64u32;
    let height = 64u32;
    let mut gray = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let noise = ((x * 7 + y * 13) % 64) as u8;
            gray.push(128u8.wrapping_add(noise));
        }
    }

    let result = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(10.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&gray);

    match result {
        Ok(jpeg) => {
            println!("Grayscale Q10 encoded: {} bytes", jpeg.len());
            std::fs::write("/tmp/gray_q10.jpg", &jpeg).unwrap();
            use std::process::Command;
            let output = Command::new("djpeg")
                .args(["-pnm", "/tmp/gray_q10.jpg"])
                .output();
            match output {
                Ok(o) if o.status.success() => println!("djpeg: OK"),
                Ok(o) => println!("djpeg: FAIL - {}", String::from_utf8_lossy(&o.stderr).trim()),
                Err(e) => println!("djpeg error: {}", e),
            }
        }
        Err(e) => println!("Encode failed: {:?}", e),
    }
}
