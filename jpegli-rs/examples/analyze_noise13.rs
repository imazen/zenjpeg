use jpegli::encode::Encoder;
use jpegli::quant::Quality;
use jpegli::types::JpegMode;

fn main() {
    let width = 64u32;
    let height = 64u32;
    let noise_mul = 13u32;
    
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let noise = ((x * noise_mul + y * noise_mul) % 64) as u8;
            rgb.push(((x * 4) as u8).wrapping_add(noise));
            rgb.push(((y * 4) as u8).wrapping_add(noise / 2));
            rgb.push(128u8.wrapping_add(noise));
        }
    }

    // Enable all debugging
    std::env::set_var("DUMP_CONTEXT_MAP", "1");
    std::env::set_var("DUMP_RUST_AC_REFINEMENT", "1");
    std::env::set_var("DUMP_HUFFMAN_CODES", "1");

    let result = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(10.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&rgb);

    match result {
        Ok(jpeg) => {
            std::fs::write("/tmp/noise13.jpg", &jpeg).unwrap();
            println!("Encoded {} bytes", jpeg.len());
        }
        Err(e) => println!("Encode failed: {:?}", e),
    }
}
