use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn main() {
    let width = 32u32;
    let height = 32u32;
    let mut data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 8 + y * 8) % 256) as u8);
        }
    }

    // Test with optimize_huffman=false first (direct encoding path)
    let encoder_direct = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    let direct_data = encoder_direct
        .encode(&data)
        .expect("Direct encoding should succeed");
    match jpeg_decoder::Decoder::new(&direct_data[..]).decode() {
        Ok(_) => println!(
            "Direct (optimize_huffman=false): OK - {} bytes",
            direct_data.len()
        ),
        Err(e) => println!("Direct (optimize_huffman=false) FAILED: {:?}", e),
    }

    // Now test with optimize_huffman=true (tokenize+replay path)
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&data).expect("Encoding should succeed");

    println!("Encoded {} bytes", jpeg_data.len());
    std::fs::write("/tmp/test_prog.jpg", &jpeg_data).unwrap();
    println!("Wrote to /tmp/test_prog.jpg");

    // Parse markers
    let mut i = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF {
            let marker = jpeg_data[i + 1];
            match marker {
                0xD8 => println!("SOI at {}", i),
                0xD9 => println!("EOI at {}", i),
                0xC2 => println!("SOF2 (progressive) at {}", i),
                0xC4 => {
                    let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
                    println!("DHT marker at {} len={}", i, len);

                    // Parse all tables in this DHT segment
                    let mut pos = i + 4;
                    while pos < i + 2 + len {
                        let class_id = jpeg_data[pos];
                        let class = (class_id >> 4) & 0xF;
                        let id = class_id & 0xF;
                        let type_name = if class == 0 { "DC" } else { "AC" };

                        let bits = &jpeg_data[pos + 1..pos + 17];
                        let total_symbols: usize = bits.iter().map(|&b| b as usize).sum();

                        println!("  {} table {}: {} symbols", type_name, id, total_symbols);

                        let values = &jpeg_data[pos + 17..pos + 17 + total_symbols.min(30)];
                        println!("    Values: {:02X?}", values);
                        pos += 17 + total_symbols;
                    }
                    i += 2 + len;
                    continue;
                }
                0xDA => {
                    let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
                    let num_comp = jpeg_data[i + 4];
                    let ss = jpeg_data[i + 5 + num_comp as usize * 2];
                    let se = jpeg_data[i + 6 + num_comp as usize * 2];
                    let ah_al = jpeg_data[i + 7 + num_comp as usize * 2];
                    println!(
                        "SOS: {} comp, Ss={}, Se={}, Ah={}, Al={}",
                        num_comp,
                        ss,
                        se,
                        ah_al >> 4,
                        ah_al & 0xF
                    );
                }
                _ => {}
            }
        }
        i += 1;
    }

    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("Decode: OK"),
        Err(e) => println!("Decode FAILED: {:?}", e),
    }

    // Dump hex around each SOS marker to see scan data
    let mut i = 0;
    let mut scan_num = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
            let data_start = i + 2 + len;

            // Find end of scan data (next marker)
            let mut data_end = data_start;
            while data_end < jpeg_data.len() - 1 {
                if jpeg_data[data_end] == 0xFF && jpeg_data[data_end + 1] != 0x00 {
                    break;
                }
                data_end += 1;
            }

            let scan_len = data_end - data_start;
            println!(
                "\nScan {}: data at {} ({} bytes)",
                scan_num, data_start, scan_len
            );
            let preview_len = scan_len.min(40);
            let preview = &jpeg_data[data_start..data_start + preview_len];
            println!("  First bytes: {:02X?}", preview);

            scan_num += 1;
            i = data_end;
        } else {
            i += 1;
        }
    }
}
