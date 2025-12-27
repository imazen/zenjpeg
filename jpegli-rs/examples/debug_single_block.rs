// Debug a single block to trace encoder vs decoder expectations
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn main() {
    // Very small test: 9x9 should have 4 blocks (2x2)
    // Pattern that creates interesting AC coefficients
    let mut data = vec![0u8; 9 * 9];
    for y in 0..9 {
        for x in 0..9 {
            data[y * 9 + x] = ((x * 17 ^ y * 31) % 256) as u8;
        }
    }

    // First, try without progressive to verify basic encoding works
    println!("=== Sequential encoding (baseline) ===");
    let jpeg_seq = Encoder::new()
        .width(9)
        .height(9)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(75.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .encode(&data)
        .expect("encode failed");

    match jpeg_decoder::Decoder::new(&jpeg_seq[..]).decode() {
        Ok(_) => println!("Sequential: OK ({} bytes)", jpeg_seq.len()),
        Err(e) => println!("Sequential: FAILED - {:?}", e),
    }

    // Now try progressive
    println!("\n=== Progressive encoding ===");
    std::env::set_var("DEBUG_REFINE_SYMBOLS", "1");

    let jpeg_prog = Encoder::new()
        .width(9)
        .height(9)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    println!("\nDecode result:");
    match jpeg_decoder::Decoder::new(&jpeg_prog[..]).decode() {
        Ok(_) => println!("Progressive: OK ({} bytes)", jpeg_prog.len()),
        Err(e) => println!("Progressive: FAILED - {:?}", e),
    }

    // Try PIL
    std::fs::write("/tmp/test_9x9_prog.jpg", &jpeg_prog).unwrap();
    println!("\nFile saved to /tmp/test_9x9_prog.jpg");
    println!(
        "Try: python3 -c \"from PIL import Image; Image.open('/tmp/test_9x9_prog.jpg').show()\""
    );
}
