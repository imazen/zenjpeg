use jpegli::{
    encode::Encoder,
    types::{JpegMode, PixelFormat, Subsampling},
    Quality,
};

fn main() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S444)
        .mode(JpegMode::Progressive);

    let encoded = encoder.encode(&pixels).expect("encode");
    std::fs::write("/tmp/s444_prog.jpg", &encoded).unwrap();
    println!("Wrote {} bytes to /tmp/s444_prog.jpg", encoded.len());

    // Test with external decoder
    let output = std::process::Command::new("djpeg")
        .arg("-outfile")
        .arg("/tmp/s444_prog.ppm")
        .arg("/tmp/s444_prog.jpg")
        .output()
        .expect("djpeg");

    if output.status.success() {
        println!("djpeg: OK");
    } else {
        println!("djpeg: {}", String::from_utf8_lossy(&output.stderr));
    }

    // Also test with jpeg-decoder crate
    match decode_zune(&encoded[..]) {
        Ok(_) => println!("jpeg-decoder: OK"),
        Err(e) => println!("jpeg-decoder: FAIL - {}", e),
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
