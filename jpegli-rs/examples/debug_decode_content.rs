use jpegli::decode::Decoder;
use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn test_content(width: u32, height: u32, name: &str, fill_fn: impl Fn(u32, u32, usize) -> u8) {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = fill_fn(x, y, 0);
            rgb[i + 1] = fill_fn(x, y, 1);
            rgb[i + 2] = fill_fn(x, y, 2);
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true);

    let jpeg = encoder.encode(&rgb).expect("encode");

    let result = Decoder::new().decode(&jpeg);
    let status = match &result {
        Ok(_) => "OK".to_string(),
        Err(e) => format!("FAIL: {:?}", e),
    };

    println!(
        "{}x{} {}: {} ({} bytes)",
        width,
        height,
        name,
        status,
        jpeg.len()
    );
}

fn main() {
    println!("Testing XYB decode with different content at 64x64...\n");

    // Gradient (the original failing pattern)
    test_content(64, 64, "gradient", |x, y, c| match c {
        0 => ((x * 4) % 256) as u8,
        1 => ((y * 4) % 256) as u8,
        _ => 128,
    });

    // Solid color
    test_content(64, 64, "solid", |_, _, _| 128);

    // Checkerboard
    test_content(64, 64, "checker", |x, y, _| {
        if (x / 8 + y / 8) % 2 == 0 {
            255
        } else {
            0
        }
    });

    // Random-ish
    test_content(64, 64, "random", |x, y, c| {
        ((x * 17 + y * 31 + c as u32 * 7) % 256) as u8
    });

    // Horizontal stripes
    test_content(
        64,
        64,
        "h_stripes",
        |_, y, _| {
            if y % 8 < 4 {
                255
            } else {
                0
            }
        },
    );

    // Vertical stripes
    test_content(
        64,
        64,
        "v_stripes",
        |x, _, _| {
            if x % 8 < 4 {
                255
            } else {
                0
            }
        },
    );

    println!("\nTesting at 32x32 (failed earlier):");
    test_content(32, 32, "gradient", |x, y, c| match c {
        0 => ((x * 8) % 256) as u8,
        1 => ((y * 8) % 256) as u8,
        _ => 128,
    });
    test_content(32, 32, "solid", |_, _, _| 128);
    test_content(32, 32, "checker", |x, y, _| {
        if (x / 4 + y / 4) % 2 == 0 {
            255
        } else {
            0
        }
    });

    println!("\nTesting at 128x128 (failed earlier):");
    test_content(128, 128, "gradient", |x, y, c| match c {
        0 => (x % 256) as u8,
        1 => (y % 256) as u8,
        _ => 128,
    });
    test_content(128, 128, "solid", |_, _, _| 128);
}
