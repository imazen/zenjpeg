use jpegli::{Decoder, Encoder, PixelFormat};

fn main() {
    // Create small test image (uniform gray)
    let width = 16;
    let height = 16;
    let data = vec![128u8; width * height * 3];

    println!("Encoding {}x{} XYB Progressive...", width, height);

    // Encode XYB + Progressive
    let jpeg = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .encode(&data)
        .unwrap();

    println!("Encoded {} bytes", jpeg.len());

    // Write to file for external inspection
    std::fs::write("/tmp/xyb_progressive_test.jpg", &jpeg).unwrap();
    println!("Wrote to /tmp/xyb_progressive_test.jpg");

    // Try to decode with our decoder
    println!("\nDecoding with jpegli-rs...");
    match Decoder::new().decode(&jpeg) {
        Ok(decoded) => {
            println!("✓ Decoded successfully: {}x{}", decoded.width, decoded.height);
            println!("  First pixel: {:?}", &decoded.data[0..3]);
        }
        Err(e) => {
            println!("✗ Decode failed: {:?}", e);
            eprintln!("\nFull error: {:#?}", e);
        }
    }

    // Try with external decoders for comparison
    println!("\nTesting with zune-jpeg...");
    use std::io::Cursor;
    let mut zune_decoder = zune_jpeg::JpegDecoder::new(Cursor::new(&jpeg));
    match zune_decoder.decode() {
        Ok(_) => println!("✓ zune-jpeg decoded successfully"),
        Err(e) => println!("✗ zune-jpeg failed: {:?}", e),
    }

    println!("\nTesting with mozjpeg...");
    match mozjpeg::Decompress::new_mem(&jpeg) {
        Ok(decoder) => match decoder.rgb() {
            Ok(_) => println!("✓ mozjpeg decoded successfully"),
            Err(e) => println!("✗ mozjpeg decode failed: {:?}", e),
        },
        Err(e) => println!("✗ mozjpeg decompress failed: {:?}", e),
    }

    println!("\nTesting with jpeg-decoder...");
    let mut jpeg_dec = jpeg_decoder::Decoder::new(&jpeg[..]);
    match jpeg_dec.decode() {
        Ok(_) => println!("✓ jpeg-decoder decoded successfully"),
        Err(e) => println!("✗ jpeg-decoder failed: {:?}", e),
    }
}
