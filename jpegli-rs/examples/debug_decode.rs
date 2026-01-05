fn main() {
    let jpeg = std::fs::read("/tmp/noise13.jpg").unwrap();
    println!("Read {} bytes", jpeg.len());

    // Try jpegli decode
    let decoder = jpegli::decode::Decoder::new();
    match decoder.decode(&jpeg) {
        Ok(img) => println!("jpegli decode OK: {}x{}", img.width, img.height),
        Err(e) => println!("jpegli decode FAIL: {:?}", e),
    }

    // Try jpeg-decoder
    let mut decoder =
        zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&jpeg[..]));
    match decoder.decode() {
        Ok(pixels) => println!("jpeg-decoder OK: {} pixels", pixels.len()),
        Err(e) => println!("jpeg-decoder FAIL: {}", e),
    }
}
