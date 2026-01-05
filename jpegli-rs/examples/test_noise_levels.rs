use jpegli::encode::Encoder;
use jpegli::quant::Quality;
use jpegli::types::JpegMode;

fn main() {
    let width = 64u32;
    let height = 64u32;
    
    for noise_mul in [1, 4, 7, 13, 20, 32] {
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let noise = ((x * noise_mul + y * noise_mul) % 64) as u8;
                rgb.push(((x * 4) as u8).wrapping_add(noise));
                rgb.push(((y * 4) as u8).wrapping_add(noise / 2));
                rgb.push(128u8.wrapping_add(noise));
            }
        }

        let result = Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(Quality::from_quality(10.0))
            .optimize_huffman(true)
            .mode(JpegMode::Progressive)
            .encode(&rgb);

        match result {
            Ok(jpeg) => {
                let path = format!("/tmp/noise_{}.jpg", noise_mul);
                std::fs::write(&path, &jpeg).unwrap();
                use std::process::Command;
                let output = Command::new("djpeg")
                    .args(["-pnm", &path])
                    .output();
                match output {
                    Ok(o) if o.status.success() => println!("noise_mul={}: {} bytes - OK", noise_mul, jpeg.len()),
                    Ok(o) => println!("noise_mul={}: {} bytes - FAIL: {}", 
                                      noise_mul, jpeg.len(), 
                                      String::from_utf8_lossy(&o.stderr).trim()),
                    Err(e) => println!("noise_mul={}: djpeg error: {}", noise_mul, e),
                }
            }
            Err(e) => println!("noise_mul={}: Encode failed: {:?}", noise_mul, e),
        }
    }
}
