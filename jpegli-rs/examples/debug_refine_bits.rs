// Detailed debug of progressive AC refinement scan bit patterns
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn gray_photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn find_scan_markers(data: &[u8]) -> Vec<(usize, &'static str, usize)> {
    let mut markers = Vec::new();
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            let marker = data[i + 1];
            let name = match marker {
                0xD8 => "SOI",
                0xD9 => "EOI",
                0xC0 => "SOF0",
                0xC2 => "SOF2",
                0xC4 => "DHT",
                0xDA => "SOS",
                0xDB => "DQT",
                0xDD => "DRI",
                0xE0..=0xEF => "APPx",
                0xFE => "COM",
                0xD0..=0xD7 => "RSTx",
                _ => "OTHER",
            };

            // Get marker length if applicable
            let length = if marker >= 0xC0
                && marker <= 0xFE
                && marker != 0xD8
                && marker != 0xD9
                && !(0xD0..=0xD7).contains(&marker)
            {
                if i + 3 < data.len() {
                    ((data[i + 2] as usize) << 8) | (data[i + 3] as usize)
                } else {
                    0
                }
            } else {
                0
            };

            markers.push((i, name, length));
        }
        i += 1;
    }
    markers
}

fn dump_sos_header(data: &[u8], pos: usize) {
    if pos + 10 < data.len() && data[pos] == 0xFF && data[pos + 1] == 0xDA {
        let len = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
        let ns = data[pos + 4]; // Number of components
        println!("  SOS: len={}, components={}", len, ns);

        let mut idx = pos + 5;
        for c in 0..ns {
            if idx + 1 < data.len() {
                let comp_id = data[idx];
                let table_sel = data[idx + 1];
                println!(
                    "    Component {}: id={}, DC table={}, AC table={}",
                    c,
                    comp_id,
                    table_sel >> 4,
                    table_sel & 0x0F
                );
                idx += 2;
            }
        }

        if idx + 2 < data.len() {
            let ss = data[idx];
            let se = data[idx + 1];
            let ahl = data[idx + 2];
            let ah = ahl >> 4;
            let al = ahl & 0x0F;
            println!("    Spectral: ss={}, se={}, ah={}, al={}", ss, se, ah, al);

            if ss == 0 && se == 0 {
                if ah == 0 {
                    println!("    → DC first scan");
                } else {
                    println!("    → DC refinement scan");
                }
            } else {
                if ah == 0 {
                    println!("    → AC first scan");
                } else {
                    println!("    → AC REFINEMENT scan (ah={}, al={})", ah, al);
                }
            }
        }
    }
}

fn find_entropy_data(data: &[u8], sos_pos: usize) -> (usize, usize) {
    // Find start of entropy data (after SOS header)
    let len = ((data[sos_pos + 2] as usize) << 8) | (data[sos_pos + 3] as usize);
    let start = sos_pos + 2 + len;

    // Find end (next marker)
    let mut end = start;
    while end < data.len() - 1 {
        if data[end] == 0xFF && data[end + 1] != 0x00 && data[end + 1] != 0xFF {
            break;
        }
        end += 1;
    }

    (start, end)
}

fn dump_first_bytes(data: &[u8], start: usize, end: usize, count: usize) {
    let actual_end = (start + count).min(end);
    print!("    First {} bytes: ", actual_end - start);
    for i in start..actual_end {
        print!("{:02X} ", data[i]);
    }
    if actual_end < end {
        print!("... ({} more bytes)", end - actual_end);
    }
    println!();
}

fn dump_last_bytes(data: &[u8], start: usize, end: usize, count: usize) {
    let actual_start = if end - start > count {
        end - count
    } else {
        start
    };
    print!("    Last {} bytes: ", end - actual_start);
    for i in actual_start..end {
        print!("{:02X} ", data[i]);
    }
    println!();
}

fn analyze_jpeg(name: &str, data: &[u8]) {
    println!("\n=== {} ({} bytes) ===", name, data.len());

    let markers = find_scan_markers(data);
    let mut scan_num = 0;

    for (pos, marker_name, length) in &markers {
        if *marker_name == "SOS" {
            scan_num += 1;
            println!("\nScan #{} at offset 0x{:04X}:", scan_num, pos);
            dump_sos_header(data, *pos);

            let (start, end) = find_entropy_data(data, *pos);
            println!(
                "    Entropy data: {} bytes (0x{:04X} - 0x{:04X})",
                end - start,
                start,
                end
            );
            dump_first_bytes(data, start, end, 32);
            dump_last_bytes(data, start, end, 32);
        }
    }
}

fn main() {
    // Test 49x49 (working) vs 50x50 (failing)
    let data_49 = gray_photo_like(49, 49);
    let data_50 = gray_photo_like(50, 50);

    println!("Encoding 49x49 (should work)...");
    let jpeg_49 = Encoder::new()
        .width(49)
        .height(49)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data_49)
        .expect("encode failed");

    println!("Encoding 50x50 (fails to decode)...");
    let jpeg_50 = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data_50)
        .expect("encode failed");

    analyze_jpeg("49x49 (working)", &jpeg_49);
    analyze_jpeg("50x50 (failing)", &jpeg_50);

    // Attempt to decode
    println!("\n=== Decode Test ===");
    match jpeg_decoder::Decoder::new(&jpeg_49[..]).decode() {
        Ok(_) => println!("49x49: decode OK"),
        Err(e) => println!("49x49: decode FAILED - {:?}", e),
    }

    match jpeg_decoder::Decoder::new(&jpeg_50[..]).decode() {
        Ok(_) => println!("50x50: decode OK"),
        Err(e) => println!("50x50: decode FAILED - {:?}", e),
    }

    // Save files for external analysis
    std::fs::write("/tmp/gray_49x49_prog.jpg", &jpeg_49).unwrap();
    std::fs::write("/tmp/gray_50x50_prog.jpg", &jpeg_50).unwrap();
    println!("\nFiles saved to /tmp/gray_49x49_prog.jpg and /tmp/gray_50x50_prog.jpg");
    println!("Try: djpeg -verbose /tmp/gray_50x50_prog.jpg > /dev/null");
}
