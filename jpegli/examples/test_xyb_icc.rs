#[allow(deprecated)]
fn main() {
    use jpegli::encode::Encoder;
    use jpegli::quant::Quality;
    use jpegli::types::PixelFormat;

    // Create a 16x16 test image
    let mut data = vec![0u8; 16 * 16 * 3];
    for y in 0..16usize {
        for x in 0..16usize {
            let idx = (y * 16 + x) * 3;
            data[idx] = (x * 16) as u8;
            data[idx + 1] = (y * 16) as u8;
            data[idx + 2] = 128;
        }
    }

    let jpeg = Encoder::new()
        .width(16)
        .height(16)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&data)
        .expect("Encoding failed");

    // Check for ICC profile signature
    let icc_sig = b"ICC_PROFILE\0";
    let mut found_icc = false;
    for i in 0..jpeg.len().saturating_sub(12) {
        if &jpeg[i..i + 12] == icc_sig {
            found_icc = true;
            println!("Found ICC profile at offset {}", i);
            break;
        }
    }

    println!("JPEG size: {} bytes", jpeg.len());
    if found_icc {
        println!("SUCCESS: XYB JPEG contains ICC profile");
    } else {
        println!("ERROR: No ICC profile found");
    }

    std::fs::write("/tmp/test_xyb_output.jpg", &jpeg).unwrap();
    println!("Saved to /tmp/test_xyb_output.jpg");
}
