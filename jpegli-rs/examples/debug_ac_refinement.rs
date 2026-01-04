use jpegli::{Encoder, PixelFormat};

fn main() {
    // Simple 64x64 gradient - same as our test
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;     // R gradient
            data[idx + 1] = ((y * 255) / height) as u8; // G gradient
            data[idx + 2] = 128;                        // B constant
        }
    }

    println!("=== Encoding 64x64 gradient with XYB Progressive ===\n");

    let jpeg = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .encode(&data)
        .unwrap();

    println!("Encoded {} bytes\n", jpeg.len());

    // Analyze scan structure
    analyze_scans(&jpeg);
}

fn analyze_scans(data: &[u8]) {
    let mut i = 0;
    let mut scan_num = 0;

    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            scan_num += 1;

            let num_components = data[i + 4];
            let mut offset = i + 5;

            let comp_id = data[offset];
            offset += 2 * num_components as usize;

            let ss = data[offset];
            let se = data[offset + 1];
            let ah_al = data[offset + 2];
            let ah = ah_al >> 4;
            let al = ah_al & 0x0F;

            // Find scan data
            let scan_start = i + 2 + length;
            let mut scan_end = scan_start;
            while scan_end + 1 < data.len() {
                if data[scan_end] == 0xFF && data[scan_end + 1] != 0x00 && data[scan_end + 1] != 0xFF {
                    break;
                }
                scan_end += 1;
            }
            let scan_size = scan_end - scan_start;

            // Show ALL scans for AC coefficients (Ss >= 1)
            if ss >= 1 {
                let scan_type = if ah == 0 {
                    "First"
                } else {
                    "Refine"
                };

                println!("Scan #{}: {} Ss={}, Se={}, Ah={}, Al={} → {} bytes",
                    scan_num, scan_type, ss, se, ah, al, scan_size);

                if scan_size > 100 {
                    println!("  ⚠️  HUGE SCAN! Expected ~2-30 bytes for gradient");

                    // Count byte values
                    let scan_data = &data[scan_start..scan_end];
                    let zeros = scan_data.iter().filter(|&&b| b == 0x00).count();
                    let ffs = scan_data.iter().filter(|&&b| b == 0xFF).count();
                    let others = scan_size - zeros - ffs;

                    println!("  Byte distribution: {} zeros, {} 0xFF, {} others", zeros, ffs, others);

                    // Show first 64 bytes
                    println!("  First 64 bytes:");
                    for chunk in scan_data[..scan_data.len().min(64)].chunks(16) {
                        print!("    ");
                        for &b in chunk {
                            print!("{:02X} ", b);
                        }
                        println!();
                    }

                    // Show last 16 bytes
                    if scan_size > 64 {
                        println!("  Last 16 bytes:");
                        print!("    ");
                        for &b in &scan_data[scan_size.saturating_sub(16)..] {
                            print!("{:02X} ", b);
                        }
                        println!();
                    }
                }
            }

            i = scan_end;
        } else {
            i += 1;
        }
    }
}
