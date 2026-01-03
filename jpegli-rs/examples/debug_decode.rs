use jpegli::decode::Decoder;
use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn main() {
    // Create a simple 16x16 test image
    let width = 16u32;
    let height = 16u32;
    let mut rgb = vec![128u8; (width * height * 3) as usize];
    for i in 0..rgb.len() {
        rgb[i] = (i % 256) as u8;
    }

    // Encode with XYB
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true);

    let jpeg = encoder.encode(&rgb).expect("encode");
    println!("Encoded {} bytes (XYB mode)", jpeg.len());

    // Print first 200 bytes to see markers
    println!("\nJPEG header:");
    for (i, byte) in jpeg.iter().take(200).enumerate() {
        if i % 16 == 0 {
            print!("\n{:04x}: ", i);
        }
        print!("{:02x} ", byte);
    }
    println!("\n");

    // Try decoding with native decoder
    println!("Attempting native decode (XYB)...");
    let decoder = Decoder::new();
    match decoder.decode(&jpeg) {
        Ok(img) => println!("Native decode SUCCESS: {}x{}", img.width, img.height),
        Err(e) => println!("Native decode FAILED: {:?}", e),
    }

    // Try decoding with jpeg-decoder
    println!("\nAttempting jpeg-decoder (XYB)...");
    match jpeg_decoder::Decoder::new(&jpeg[..]).decode() {
        Ok(pixels) => println!("jpeg-decoder SUCCESS: {} bytes", pixels.len()),
        Err(e) => println!("jpeg-decoder FAILED: {:?}", e),
    }

    // Now try YCbCr mode
    println!("\n--- YCbCr mode ---");
    let encoder_ycbcr = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(false);

    let jpeg_ycbcr = encoder_ycbcr.encode(&rgb).expect("encode ycbcr");
    println!("Encoded {} bytes (YCbCr)", jpeg_ycbcr.len());

    println!("\nAttempting native decode (YCbCr)...");
    let decoder = Decoder::new();
    match decoder.decode(&jpeg_ycbcr) {
        Ok(img) => println!("Native decode SUCCESS: {}x{}", img.width, img.height),
        Err(e) => println!("Native decode FAILED: {:?}", e),
    }

    // Also try a larger image to rule out size issues
    println!("\n--- Larger image (64x64) ---");
    let width = 64u32;
    let height = 64u32;
    let mut rgb = vec![128u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = ((x * 4) % 256) as u8;
            rgb[i + 1] = ((y * 4) % 256) as u8;
            rgb[i + 2] = 128;
        }
    }

    let encoder_xyb = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true);

    let jpeg_xyb = encoder_xyb.encode(&rgb).expect("encode xyb 64x64");
    println!("Encoded {} bytes (XYB 64x64)", jpeg_xyb.len());

    println!("\nAttempting native decode (XYB 64x64)...");
    match Decoder::new().decode(&jpeg_xyb) {
        Ok(img) => println!("Native decode SUCCESS: {}x{}", img.width, img.height),
        Err(e) => println!("Native decode FAILED: {:?}", e),
    }

    println!("\nAttempting jpeg-decoder (XYB 64x64)...");
    match jpeg_decoder::Decoder::new(&jpeg_xyb[..]).decode() {
        Ok(pixels) => println!("jpeg-decoder SUCCESS: {} bytes", pixels.len()),
        Err(e) => println!("jpeg-decoder FAILED: {:?}", e),
    }
}
