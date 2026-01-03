// Trace a single failing case to understand the bug
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn main() {
    // 11x11 with gradient pattern fails
    let size = 11u32;
    let data: Vec<u8> = (0..size * size)
        .map(|i| {
            let x = i % size;
            let y = i / size;
            ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8
        })
        .collect();

    println!("Testing {}x{} progressive encoding", size, size);
    println!(
        "Blocks: {}x{} = {}",
        (size + 7) / 8,
        (size + 7) / 8,
        ((size + 7) / 8).pow(2)
    );

    std::env::set_var("DEBUG_REFINE_SYMBOLS", "1");
    std::env::set_var("DEBUG_EOB", "1");

    let jpeg = Encoder::new()
        .width(size)
        .height(size)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    println!("\nEncoded {} bytes", jpeg.len());

    // Save to file
    let filename = format!("/tmp/fail_{}x{}.jpg", size, size);
    std::fs::write(&filename, &jpeg).expect("write failed");
    println!("Saved to {}", filename);

    // Try to decode
    match jpeg_decoder::Decoder::new(&jpeg[..]).decode() {
        Ok(_) => println!("Decode: OK"),
        Err(e) => println!("Decode: FAIL - {}", e),
    }

    // Detailed djpeg output
    let output = std::process::Command::new("djpeg")
        .arg("-verbose")
        .arg(&filename)
        .output();
    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        println!("\ndjpeg stdout:\n{}", stdout);
        println!("\ndjpeg stderr:\n{}", stderr);
    }
}
