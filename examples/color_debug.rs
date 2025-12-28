//! Debug color conversion differences between zenjpeg and mozjpeg-oxide

fn main() {
    // Test color values
    let test_colors: Vec<(u8, u8, u8)> = vec![
        (0, 0, 0),       // Black
        (255, 255, 255), // White
        (255, 0, 0),     // Red
        (0, 255, 0),     // Green
        (0, 0, 255),     // Blue
        (128, 128, 128), // Gray
        (200, 50, 100),  // Random color
    ];

    println!("=== Color Conversion Comparison ===");
    println!("zenjpeg uses JFIF/BT.601 coefficients:");
    println!("  Y  =  0.299*R + 0.587*G + 0.114*B");
    println!("  Cb = -0.169*R - 0.331*G + 0.500*B + 128");
    println!("  Cr =  0.500*R - 0.419*G - 0.081*B + 128");
    println!();

    for (r, g, b) in test_colors {
        let (y, cb, cr) = zenjpeg_rgb_to_ycbcr(r, g, b);
        println!(
            "RGB({:3},{:3},{:3}) -> YCbCr({:3},{:3},{:3})",
            r, g, b, y, cb, cr
        );
    }

    // Now test a larger image and compare decoded output
    println!();
    println!("=== Roundtrip Test (encode RGB, decode back to RGB) ===");

    // Create test pattern - use Y-only (gray) to isolate luma issues
    let width = 8usize;
    let height = 8usize;

    // Pure gray gradient (should have Cb=Cr=128 exactly)
    let mut gray_rgb = Vec::with_capacity(width * height * 3);
    for i in 0..64 {
        let v = (i * 4) as u8; // 0 to 252
        gray_rgb.push(v);
        gray_rgb.push(v);
        gray_rgb.push(v);
    }

    // Encode and decode with both
    let zen = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(80))
        .encode_rgb(&gray_rgb, width, height)
        .unwrap();

    let moz = mozjpeg_oxide::Encoder::new()
        .quality(80)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&gray_rgb, width as u32, height as u32)
        .unwrap();

    let zen_dec = jpeg_decoder::Decoder::new(&zen[..]).decode().unwrap();
    let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

    println!("First 8 pixels (gray gradient 0-28):");
    println!("Orig | zen_R  zen_G  zen_B | moz_R  moz_G  moz_B");
    println!("{}", "-".repeat(55));

    for i in 0..8 {
        let orig = gray_rgb[i * 3];
        let zen_r = zen_dec[i * 3];
        let zen_g = zen_dec[i * 3 + 1];
        let zen_b = zen_dec[i * 3 + 2];
        let moz_r = moz_dec[i * 3];
        let moz_g = moz_dec[i * 3 + 1];
        let moz_b = moz_dec[i * 3 + 2];

        let zen_diff = (orig as i32 - zen_r as i32).abs();
        let moz_diff = (orig as i32 - moz_r as i32).abs();

        println!(
            "{:4} |  {:3}   {:3}   {:3}  |  {:3}   {:3}   {:3}   (err: zen={}, moz={})",
            orig, zen_r, zen_g, zen_b, moz_r, moz_g, moz_b, zen_diff, moz_diff
        );
    }

    // Total error
    let mut zen_err = 0i64;
    let mut moz_err = 0i64;
    for i in 0..gray_rgb.len() {
        zen_err += (gray_rgb[i] as i64 - zen_dec[i] as i64).abs();
        moz_err += (gray_rgb[i] as i64 - moz_dec[i] as i64).abs();
    }
    println!();
    println!("Total absolute error: zen={}, moz={}", zen_err, moz_err);

    // Now test with actual color image
    println!();
    println!("=== Color Roundtrip Test ===");

    // Red gradient
    let mut red_rgb = Vec::with_capacity(width * height * 3);
    for i in 0..64 {
        let v = (i * 4) as u8;
        red_rgb.push(v); // R varies
        red_rgb.push(0); // G = 0
        red_rgb.push(0); // B = 0
    }

    let zen = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(80))
        .encode_rgb(&red_rgb, width, height)
        .unwrap();

    let moz = mozjpeg_oxide::Encoder::new()
        .quality(80)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&red_rgb, width as u32, height as u32)
        .unwrap();

    let zen_dec = jpeg_decoder::Decoder::new(&zen[..]).decode().unwrap();
    let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

    println!("First 8 pixels (red gradient):");
    println!("Orig | zen_R  zen_G  zen_B | moz_R  moz_G  moz_B");
    println!("{}", "-".repeat(55));

    for i in 0..8 {
        let orig_r = red_rgb[i * 3];
        let zen_r = zen_dec[i * 3];
        let zen_g = zen_dec[i * 3 + 1];
        let zen_b = zen_dec[i * 3 + 2];
        let moz_r = moz_dec[i * 3];
        let moz_g = moz_dec[i * 3 + 1];
        let moz_b = moz_dec[i * 3 + 2];

        println!(
            "{:4} |  {:3}   {:3}   {:3}  |  {:3}   {:3}   {:3}",
            orig_r, zen_r, zen_g, zen_b, moz_r, moz_g, moz_b
        );
    }
}

/// zenjpeg's RGB to YCbCr conversion
fn zenjpeg_rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}
