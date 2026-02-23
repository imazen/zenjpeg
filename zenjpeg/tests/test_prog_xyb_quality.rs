use enough::Unstoppable;
use zenjpeg::{
    decoder::Decoder,
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling},
};

#[test]
fn test_progressive_xyb_all_quality_levels() {
    // Create a test image (512x512)
    let width = 512u32;
    let height = 512u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 255) / width) as u8); // R gradient
            rgb.push(((y * 255) / height) as u8); // G gradient
            rgb.push(128u8); // B constant
        }
    }

    for &quality in &[10u8, 30, 50, 70, 85, 95] {
        // Encode progressive XYB
        let config = EncoderConfig::xyb(quality as f32, XybSubsampling::BQuarter).progressive(true);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder setup");
        enc.push_packed(&rgb, enough::Unstoppable).expect("push");
        let jpeg = enc
            .finish()
            .unwrap_or_else(|_| panic!("encode Q{} failed", quality));

        println!("Q{}: encoded {} bytes", quality, jpeg.len());

        // Decode
        let result = Decoder::new().apply_icc(true).decode(&jpeg, Unstoppable);

        match result {
            Ok(img) => println!("Q{}: decoded {}x{}", quality, img.width, img.height),
            Err(e) => panic!("Q{}: DECODE ERROR: {:?}", quality, e),
        }
    }
}

/// Test with non-8-aligned dimensions (like frymire.png: 1118x1105)
#[test]
fn test_progressive_xyb_non_aligned_dimensions() {
    // Use frymire.png dimensions - non-8-aligned
    let width = 1118u32;
    let height = 1105u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 255) / width) as u8); // R gradient
            rgb.push(((y * 255) / height) as u8); // G gradient
            rgb.push(((x + y) % 256) as u8); // B pattern
        }
    }

    // Just test one quality level for debugging
    let quality = 10u8;
    println!(
        "\nTesting Q{} with {}x{} progressive XYB...",
        quality, width, height
    );

    // Encode progressive XYB
    let config = EncoderConfig::xyb(quality as f32, XybSubsampling::BQuarter).progressive(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&rgb, enough::Unstoppable).expect("push");
    let jpeg = enc
        .finish()
        .unwrap_or_else(|_| panic!("encode Q{} failed", quality));

    println!("Q{}: encoded {} bytes", quality, jpeg.len());

    // Save for debugging
    std::fs::write("/tmp/test_prog_xyb.jpg", &jpeg).ok();
    println!("Saved to /tmp/test_prog_xyb.jpg");

    // First try decoding with zune-jpeg to verify the JPEG is valid
    {
        use zune_jpeg::JpegDecoder;
        use zune_jpeg::zune_core::bytestream::ZCursor;

        let cursor = ZCursor::new(&jpeg);
        let mut decoder = JpegDecoder::new(cursor);
        match decoder.decode() {
            Ok(data) => println!(
                "Q{}: zune-jpeg decoded {} bytes (RGB wrong but JPEG valid)",
                quality,
                data.len()
            ),
            Err(e) => println!("Q{}: zune-jpeg also failed: {:?}", quality, e),
        }
    }

    // Decode with zenjpeg
    let result = Decoder::new().apply_icc(true).decode(&jpeg, Unstoppable);

    match result {
        Ok(img) => println!("Q{}: zenjpeg decoded {}x{}", quality, img.width, img.height),
        Err(e) => panic!("Q{}: DECODE ERROR: {:?}", quality, e),
    }
}

/// Test YCbCr progressive with non-8-aligned dimensions
#[test]
fn test_progressive_ycbcr_non_aligned_dimensions() {
    // Use frymire.png dimensions - non-8-aligned
    let width = 1118u32;
    let height = 1105u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 255) / width) as u8);
            rgb.push(((y * 255) / height) as u8);
            rgb.push(((x + y) % 256) as u8);
        }
    }

    for &quality in &[10u8, 30, 50, 70, 85, 95] {
        println!(
            "\nTesting Q{} with {}x{} progressive YCbCr...",
            quality, width, height
        );

        // Encode progressive YCbCr
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
            .progressive(true)
            .quality(quality as f32);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder setup");
        enc.push_packed(&rgb, enough::Unstoppable).expect("push");
        let jpeg = enc
            .finish()
            .unwrap_or_else(|_| panic!("encode Q{} failed", quality));

        println!("Q{}: encoded {} bytes", quality, jpeg.len());

        // Decode with zune-jpeg (doesn't need ICC)
        use zune_jpeg::JpegDecoder;
        use zune_jpeg::zune_core::bytestream::ZCursor;

        let cursor = ZCursor::new(&jpeg);
        let mut decoder = JpegDecoder::new(cursor);
        let result = decoder.decode();

        match result {
            Ok(data) => println!("Q{}: decoded {} bytes", quality, data.len()),
            Err(e) => panic!("Q{}: DECODE ERROR: {:?}", quality, e),
        }
    }
}
