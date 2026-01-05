use jpegli::encode::Encoder;
use jpegli::quant::Quality;
use jpegli::types::JpegMode;

fn main() {
    // Same failing image
    let width = 64u32;
    let height = 64u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let noise = ((x * 7 + y * 13) % 64) as u8;
            rgb.push(((x * 4) as u8).wrapping_add(noise));
            rgb.push(((y * 4) as u8).wrapping_add(noise / 2));
            rgb.push(128u8.wrapping_add(noise));
        }
    }

    // Enable context map debugging
    std::env::set_var("DUMP_CONTEXT_MAP", "1");

    let result = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(10.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&rgb);

    match result {
        Ok(jpeg) => {
            println!("Encoded {} bytes", jpeg.len());
            std::fs::write("/tmp/fail_q10.jpg", &jpeg).unwrap();
            use std::process::Command;
            let output = Command::new("djpeg")
                .args(["-pnm", "/tmp/fail_q10.jpg"])
                .output();
            match output {
                Ok(o) if o.status.success() => println!("djpeg: OK"),
                Ok(o) => println!(
                    "djpeg: FAIL - {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                ),
                Err(e) => println!("djpeg error: {}", e),
            }
        }
        Err(e) => println!("Encode failed: {:?}", e),
    }
}
