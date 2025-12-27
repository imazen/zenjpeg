// Trace exactly what happens in AC refinement encoding
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn main() {
    // Create the RGB gradient that fails
    let rgb_grad: Vec<u8> = (0..64).flat_map(|i| vec![i as u8 * 4, 128, 64]).collect();

    println!("Input image: 8x8 RGB gradient");
    println!("First 8 pixels: {:?}", &rgb_grad[..24]);

    // Encode with progressive mode - this should trigger the debug output
    let encoder = Encoder::new()
        .width(8)
        .height(8)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&rgb_grad).expect("encode failed");
    std::fs::write("/tmp/trace_refine.jpg", &jpeg_data).ok();

    println!("\nEncoded {} bytes", jpeg_data.len());

    // Parse and show DHT (Huffman table) markers
    println!("\n=== Huffman Tables ===");
    let mut i = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xC4 {
            // DHT marker
            let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
            let mut pos = i + 4;
            while pos < i + 2 + len {
                let tc_th = jpeg_data[pos];
                let tc = tc_th >> 4; // Table class: 0=DC, 1=AC
                let th = tc_th & 0xF; // Table ID

                // Read BITS
                let bits: Vec<u8> = jpeg_data[pos + 1..pos + 17].to_vec();
                let total_codes: usize = bits.iter().map(|&b| b as usize).sum();

                // Read HUFFVAL
                let huffval: Vec<u8> = jpeg_data[pos + 17..pos + 17 + total_codes].to_vec();

                println!(
                    "Table {}{}: {} codes",
                    if tc == 0 { "DC" } else { "AC" },
                    th,
                    total_codes
                );

                if tc == 1 {
                    // Show AC table symbols
                    let refinement_syms: Vec<u8> = huffval
                        .iter()
                        .filter(|&&s| s == 0x00 || s == 0xF0 || (s & 0x0F) == 1)
                        .copied()
                        .collect();
                    println!("  Refinement symbols: {:02X?}", refinement_syms);

                    // Check for specific symbols
                    for sym in [0x00u8, 0x01, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0xF0, 0xF1] {
                        if huffval.contains(&sym) {
                            // Find code for this symbol
                            let mut code = 0u16;
                            let mut code_len = 0u8;
                            let mut idx = 0;
                            'outer: for (len, &count) in bits.iter().enumerate() {
                                for _ in 0..count {
                                    if huffval[idx] == sym {
                                        code_len = (len + 1) as u8;
                                        break 'outer;
                                    }
                                    code += 1;
                                    idx += 1;
                                }
                                code <<= 1;
                            }
                            if code_len > 0 {
                                let code_str: String = (0..code_len)
                                    .rev()
                                    .map(|b| if (code >> b) & 1 == 1 { '1' } else { '0' })
                                    .collect();
                                println!("    0x{:02X}: {} ({} bits)", sym, code_str, code_len);
                            }
                        } else {
                            println!("    0x{:02X}: NOT IN TABLE", sym);
                        }
                    }
                }

                pos += 17 + total_codes;
            }
            i = i + 2 + len;
        } else {
            i += 1;
        }
    }

    // Parse scans and show AC refinement data
    println!("\n=== AC Refinement Scans ===");
    i = 0;
    let mut scan_num = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
            let num_comp = jpeg_data[i + 4];

            let mut comp_info = Vec::new();
            for c in 0..num_comp as usize {
                let comp_id = jpeg_data[i + 5 + c * 2];
                let table_sel = jpeg_data[i + 6 + c * 2];
                comp_info.push((comp_id, table_sel >> 4, table_sel & 0xF));
            }

            let base = i + 5 + num_comp as usize * 2;
            let ss = jpeg_data[base];
            let se = jpeg_data[base + 1];
            let ah_al = jpeg_data[base + 2];
            let ah = ah_al >> 4;
            let al = ah_al & 0xF;

            let scan_start = i + 2 + len;
            let mut scan_end = scan_start;
            while scan_end < jpeg_data.len() - 1 {
                if jpeg_data[scan_end] == 0xFF && jpeg_data[scan_end + 1] != 0x00 {
                    break;
                }
                scan_end += 1;
            }

            if ah > 0 {
                let data = &jpeg_data[scan_start..scan_end];
                println!(
                    "Scan {} (AC refine): comp={:?} Ss={}-{} Ah={} Al={}",
                    scan_num, comp_info, ss, se, ah, al
                );
                println!("  Data: {} bytes", data.len());

                // Show bit-by-bit
                let bits: String = data
                    .iter()
                    .flat_map(|b| {
                        (0..8)
                            .rev()
                            .map(move |bit| if (b >> bit) & 1 == 1 { '1' } else { '0' })
                    })
                    .collect();
                println!("  Bits: {}", bits);

                // Try to decode using chroma AC table for components 2,3
                let ac_table_idx = if comp_info[0].0 > 1 { 1 } else { 0 };
                println!("  Using AC table {}", ac_table_idx);
            }

            scan_num += 1;
            i = scan_end;
        } else {
            i += 1;
        }
    }

    // Try decoding
    println!("\n=== Decode ===");
    match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAILED: {:?}", e),
    }
}
