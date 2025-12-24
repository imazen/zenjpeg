fn main() {
    // Create 16x16 blue image (sRGB 0,0,128)
    let mut data = vec![0u8; 16 * 16 * 3];
    for i in 0..(16*16) {
        data[i*3] = 0;
        data[i*3+1] = 0;
        data[i*3+2] = 128;
    }

    let jpeg = jpegli::Encoder::new()
        .width(16)
        .height(16)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&data)
        .unwrap();

    std::fs::write("/tmp/rust_xyb_blue.jpg", &jpeg).unwrap();
    println!("Wrote {} bytes to /tmp/rust_xyb_blue.jpg", jpeg.len());

    // Check for APP2 marker (ICC profile)
    for i in 0..jpeg.len()-1 {
        if jpeg[i] == 0xFF && jpeg[i+1] == 0xE2 {
            let len = ((jpeg[i+2] as usize) << 8) | (jpeg[i+3] as usize);
            println!("Found ICC APP2 at offset {}, length {}", i, len);
            break;
        }
    }
}
