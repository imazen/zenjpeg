use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn main() {
    // Create test data
    let rgb_grad: Vec<u8> = (0..64).flat_map(|i| vec![i as u8 * 4, 128, 64]).collect();

    let encoder = Encoder::new()
        .width(8)
        .height(8)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&rgb_grad).expect("encode failed");
    std::fs::write("/tmp/test_decoder.jpg", &jpeg_data).ok();

    println!(
        "Encoded {} bytes, saved to /tmp/test_decoder.jpg",
        jpeg_data.len()
    );

    // Try jpeg-decoder
    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("jpeg-decoder: OK"),
        Err(e) => println!("jpeg-decoder: FAILED - {:?}", e),
    }

    // Try image crate
    match image::load_from_memory(&jpeg_data) {
        Ok(_) => println!("image crate: OK"),
        Err(e) => println!("image crate: FAILED - {:?}", e),
    }
}
