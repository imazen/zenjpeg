use jpegli::{Decoder, Encoder, PixelFormat};

fn main() {
    // Create minimal XYB progressive JPEG
    let width = 16;
    let height = 16;
    let data = vec![128u8; width * height * 3];

    println!("Creating {}x{} XYB Progressive JPEG...", width, height);

    let jpeg = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .encode(&data)
        .unwrap();

    println!("Created {} bytes", jpeg.len());
    std::fs::write("/tmp/xyb_progressive_8x8.jpg", &jpeg).unwrap();

    // Now create equivalent YCbCr progressive for comparison
    println!("\nCreating {}x{} YCbCr Progressive JPEG...", width, height);

    let jpeg_ycbcr = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(false)
        .encode(&data)
        .unwrap();

    println!("Created {} bytes", jpeg_ycbcr.len());
    std::fs::write("/tmp/ycbcr_progressive_8x8.jpg", &jpeg_ycbcr).unwrap();

    // Try decoding both
    println!("\n=== Decoding YCbCr Progressive ===");
    match Decoder::new().decode(&jpeg_ycbcr) {
        Ok(decoded) => println!("✓ YCbCr decoded: {}x{}", decoded.width, decoded.height),
        Err(e) => println!("✗ YCbCr failed: {:?}", e),
    }

    println!("\n=== Decoding XYB Progressive ===");
    match Decoder::new().decode(&jpeg) {
        Ok(decoded) => println!("✓ XYB decoded: {}x{}", decoded.width, decoded.height),
        Err(e) => {
            println!("✗ XYB failed: {:?}", e);
            eprintln!("\nFull error details: {:#?}", e);
        }
    }

    // Dump scan structure for comparison
    println!("\n=== Scan Structure Comparison ===");
    println!("YCbCr: {} scans", count_scans(&jpeg_ycbcr));
    println!("XYB:   {} scans", count_scans(&jpeg));

    // Dump first few bytes of each scan data
    println!("\n=== First SOS marker details ===");
    dump_first_sos("YCbCr", &jpeg_ycbcr);
    dump_first_sos("XYB", &jpeg);
}

fn count_scans(jpeg_data: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + 1 < jpeg_data.len() {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            count += 1;
            i += 2;
            while i < jpeg_data.len() {
                if jpeg_data[i] == 0xFF && i + 1 < jpeg_data.len() {
                    let next = jpeg_data[i + 1];
                    if next != 0x00 && next != 0xFF {
                        break;
                    }
                }
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    count
}

fn dump_first_sos(label: &str, jpeg_data: &[u8]) {
    let mut i = 0;
    while i + 1 < jpeg_data.len() {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            // Found SOS marker
            let length = u16::from_be_bytes([jpeg_data[i + 2], jpeg_data[i + 3]]) as usize;
            println!("{}: SOS at offset {}, length {}", label, i, length);

            let num_components = jpeg_data[i + 4];
            println!("  Components: {}", num_components);

            let mut offset = i + 5;
            for comp in 0..num_components {
                let comp_id = jpeg_data[offset];
                let tables = jpeg_data[offset + 1];
                let dc_table = tables >> 4;
                let ac_table = tables & 0x0F;
                println!("    Component {}: ID={} ('{}'/'0x{:02X}'), DC table={}, AC table={}",
                    comp, comp_id, comp_id as char, comp_id, dc_table, ac_table);
                offset += 2;
            }

            let ss = jpeg_data[offset];
            let se = jpeg_data[offset + 1];
            let ah_al = jpeg_data[offset + 2];
            let ah = ah_al >> 4;
            let al = ah_al & 0x0F;
            println!("  Spectral: Ss={}, Se={}", ss, se);
            println!("  Successive approx: Ah={}, Al={}", ah, al);

            // Show first 32 bytes of scan data
            let scan_start = i + 2 + length;
            let scan_preview_end = (scan_start + 32).min(jpeg_data.len());
            print!("  First bytes of scan data: ");
            for b in &jpeg_data[scan_start..scan_preview_end] {
                print!("{:02X} ", b);
            }
            println!();

            break;
        }
        i += 1;
    }
}
