// Debug the 50x50 progressive encoding failure
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                let r = ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8;
                let g = ((x.wrapping_mul(13) ^ y.wrapping_mul(23)) % 256) as u8;
                let b = ((x.wrapping_mul(11) ^ y.wrapping_mul(19)) % 256) as u8;
                [r, g, b]
            })
        })
        .collect()
}

fn main() {
    let width = 50u32;
    let height = 50u32;
    let data = photo_like(width, height);

    println!("Testing {}x{} progressive encoding", width, height);

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    // Save to file for analysis
    std::fs::write("/tmp/debug_50x50.jpg", &jpeg_data).unwrap();
    println!("Saved {} bytes to /tmp/debug_50x50.jpg", jpeg_data.len());

    // Try to decode it
    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("Decode: OK"),
        Err(e) => {
            println!("Decode: FAILED - {:?}", e);
            // Also try djpeg
            println!("\nTrying djpeg...");
            let status = std::process::Command::new("djpeg")
                .arg("-outfile")
                .arg("/tmp/debug_50x50.ppm")
                .arg("/tmp/debug_50x50.jpg")
                .status();
            match status {
                Ok(s) => println!("djpeg exit status: {}", s),
                Err(e) => println!("djpeg error: {:?}", e),
            }
        }
    }

    // Also test 49x49 which works
    let width = 49u32;
    let height = 49u32;
    let data = photo_like(width, height);

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    std::fs::write("/tmp/debug_49x49.jpg", &jpeg_data).unwrap();
    println!("\n49x49: {} bytes", jpeg_data.len());
    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("49x49 Decode: OK"),
        Err(e) => println!("49x49 Decode: FAILED - {:?}", e),
    }
}
