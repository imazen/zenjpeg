use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn test_size(width: u32, height: u32, label: &str) {
    let mut data = vec![128u8; (width * height) as usize];

    // Set a few pixels to create known DCT coefficients
    data[0] = 180;
    data[1] = 170;
    if width >= 8 {
        data[width as usize] = 170;
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive);

    match encoder.encode(&data) {
        Ok(jpeg_data) => match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
            Ok(_) => println!("{}: OK ({} bytes)", label, jpeg_data.len()),
            Err(e) => println!("{}: ENCODE OK, DECODE FAILED: {:?}", label, e),
        },
        Err(e) => println!("{}: ENCODE FAILED: {:?}", label, e),
    }
}

fn main() {
    println!("Testing progressive paths:");
    println!();

    // Test cases
    let test_cases = vec![
        (
            "8x8 RGB gradient",
            8u32,
            8u32,
            PixelFormat::Rgb,
            (0..64)
                .flat_map(|i| vec![i as u8 * 4, 128, 64])
                .collect::<Vec<_>>(),
        ),
        (
            "16x16 gray gradient",
            16u32,
            16u32,
            PixelFormat::Gray,
            (0..256).map(|i| i as u8).collect::<Vec<_>>(),
        ),
    ];

    for (name, width, height, format, data) in &test_cases {
        // Test with optimize_huffman=false (direct path)
        let encoder = Encoder::new()
            .width(*width)
            .height(*height)
            .pixel_format(*format)
            .jpegli_quality(Quality::from_quality(90.0))
            .optimize_huffman(false)
            .mode(JpegMode::Progressive);

        match encoder.encode(data) {
            Ok(jpeg_data) => match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
                Ok(_) => println!("{} (opt=false): OK ({} bytes)", name, jpeg_data.len()),
                Err(e) => println!("{} (opt=false): DECODE FAILED: {:?}", name, e),
            },
            Err(e) => println!("{} (opt=false): ENCODE FAILED: {:?}", name, e),
        }

        // Test with optimize_huffman=true (two-pass path)
        let encoder = Encoder::new()
            .width(*width)
            .height(*height)
            .pixel_format(*format)
            .jpegli_quality(Quality::from_quality(90.0))
            .optimize_huffman(true)
            .mode(JpegMode::Progressive);

        match encoder.encode(data) {
            Ok(jpeg_data) => match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
                Ok(_) => println!("{} (opt=true): OK ({} bytes)", name, jpeg_data.len()),
                Err(e) => println!("{} (opt=true): DECODE FAILED: {:?}", name, e),
            },
            Err(e) => println!("{} (opt=true): ENCODE FAILED: {:?}", name, e),
        }
        println!();
    }

    return;

    // This is the exact test case from test_progressive_optimized_single_block
    let width = 8u32;
    let height = 8u32;
    let data: Vec<u8> = (0..64).flat_map(|i| [i as u8 * 4, 128, 64]).collect();

    // Test with optimize_huffman=false (direct path)
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    match encoder.encode(&data) {
        Ok(jpeg_data) => {
            // Save to file for analysis
            std::fs::write("/tmp/test_prog_refine.jpg", &jpeg_data).ok();
            println!(
                "Saved to /tmp/test_prog_refine.jpg ({} bytes)",
                jpeg_data.len()
            );

            // Parse and show markers
            let mut i = 0;
            while i < jpeg_data.len() - 1 {
                if jpeg_data[i] == 0xFF && jpeg_data[i + 1] != 0x00 {
                    let marker = jpeg_data[i + 1];
                    match marker {
                        0xD8 => println!("  SOI at {}", i),
                        0xD9 => println!("  EOI at {}", i),
                        0xC2 => println!("  SOF2 (progressive) at {}", i),
                        0xC4 => {
                            let len =
                                ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
                            println!("  DHT at {} (len={})", i, len);
                            i += 2 + len;
                            continue;
                        }
                        0xDA => {
                            let len =
                                ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
                            let num_comp = jpeg_data[i + 4];
                            let ss = jpeg_data[i + 5 + num_comp as usize * 2];
                            let se = jpeg_data[i + 6 + num_comp as usize * 2];
                            let ah_al = jpeg_data[i + 7 + num_comp as usize * 2];
                            let scan_start = i + 2 + len;

                            // Find scan end
                            let mut scan_end = scan_start;
                            while scan_end < jpeg_data.len() - 1 {
                                if jpeg_data[scan_end] == 0xFF && jpeg_data[scan_end + 1] != 0x00 {
                                    break;
                                }
                                scan_end += 1;
                            }
                            let scan_len = scan_end - scan_start;

                            println!(
                                "  SOS at {}: {} comp, Ss={}, Se={}, Ah={}, Al={} -> {} data bytes",
                                i,
                                num_comp,
                                ss,
                                se,
                                ah_al >> 4,
                                ah_al & 0xF,
                                scan_len
                            );

                            // Show first few bytes of scan data
                            let preview_len = scan_len.min(16);
                            let preview = &jpeg_data[scan_start..scan_start + preview_len];
                            println!("      First bytes: {:02X?}", preview);

                            i = scan_end;
                            continue;
                        }
                        _ => {}
                    }
                }
                i += 1;
            }

            match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
                Ok(_) => println!("\n8x8 RGB gradient (opt=false): OK"),
                Err(e) => println!("\n8x8 RGB gradient (opt=false): DECODE FAILED: {:?}", e),
            }
        }
        Err(e) => println!("8x8 RGB gradient (opt=false): ENCODE FAILED: {:?}", e),
    }

    // Stop here for now
    return;

    println!("Testing various image sizes for progressive encoding:");
    println!();

    // Single block tests
    test_size(8, 8, "8x8 (1 block, perfect)");

    // Multiple blocks - horizontal
    test_size(16, 8, "16x8 (2 blocks wide, perfect)");
    test_size(9, 8, "9x8 (2 blocks wide, padded)");
    test_size(10, 8, "10x8 (2 blocks wide, padded)");
    test_size(17, 8, "17x8 (3 blocks wide, padded)");

    // Multiple blocks - vertical
    test_size(8, 16, "8x16 (2 blocks tall, perfect)");
    test_size(8, 9, "8x9 (2 blocks tall, padded)");
    test_size(8, 17, "8x17 (3 blocks tall, padded)");

    // Square sizes
    test_size(16, 16, "16x16 (4 blocks, perfect)");
    test_size(24, 24, "24x24 (9 blocks, perfect)");
    test_size(32, 32, "32x32 (16 blocks, perfect)");

    // Non-multiples of 8
    test_size(9, 9, "9x9 (4 blocks, both padded)");
    test_size(15, 15, "15x15 (4 blocks, both padded)");
    test_size(17, 17, "17x17 (9 blocks, both padded)");

    println!();

    // Now detailed test of failing case
    let width = 9u32;
    let height = 8u32;
    let mut data = vec![128u8; (width * height) as usize];
    data[0] = 180;
    data[1] = 170;
    data[width as usize] = 170;

    println!("Detailed test of 9x8 image (known failing case)");

    // Test with optimize_huffman=true only for now
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&data).expect("Encoding should succeed");
    println!("Encoded {} bytes", jpeg_data.len());

    // Parse SOS markers
    let mut i = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF {
            let marker = jpeg_data[i + 1];
            if marker == 0xDA {
                let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
                let num_comp = jpeg_data[i + 4];
                let ss = jpeg_data[i + 5 + num_comp as usize * 2];
                let se = jpeg_data[i + 6 + num_comp as usize * 2];
                let ah_al = jpeg_data[i + 7 + num_comp as usize * 2];
                println!(
                    "SOS at {}: {} comp, Ss={}, Se={}, Ah={}, Al={}",
                    i,
                    num_comp,
                    ss,
                    se,
                    ah_al >> 4,
                    ah_al & 0xF
                );

                // Find scan data
                let data_start = i + 2 + len;
                let mut data_end = data_start;
                while data_end < jpeg_data.len() - 1 {
                    if jpeg_data[data_end] == 0xFF && jpeg_data[data_end + 1] != 0x00 {
                        break;
                    }
                    data_end += 1;
                }
                let scan_data = &jpeg_data[data_start..data_end];
                println!("  Data: {} bytes, bits:", scan_data.len());

                // Show first 64 bits
                let mut bits = String::new();
                for (byte_idx, &byte) in scan_data.iter().take(8).enumerate() {
                    for bit in 0..8 {
                        bits.push(if (byte >> (7 - bit)) & 1 == 1 {
                            '1'
                        } else {
                            '0'
                        });
                    }
                    bits.push(' ');
                }
                println!("  {}", bits);
            }
        }
        i += 1;
    }

    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("\nDecode: OK"),
        Err(e) => println!("\nDecode FAILED: {:?}", e),
    }
}
