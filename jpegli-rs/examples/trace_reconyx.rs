use jpegli::entropy::EntropyDecoder;
use jpegli::huffman::HuffmanDecodeTable;
use std::fs;

fn trace_decode(data: &[u8]) {
    println!("\n=== TRACING DECODE ===");

    // Find scan data start (after SOS header)
    let mut pos = 0;
    let mut scan_start = 0;
    let mut width = 0u16;
    let mut height = 0u16;
    let mut h_samp = [1u8; 4];
    let mut v_samp = [1u8; 4];
    let mut num_components = 0u8;
    let mut dc_tables: [Option<HuffmanDecodeTable>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<HuffmanDecodeTable>; 4] = [None, None, None, None];

    while pos < data.len() - 1 {
        if data[pos] == 0xFF && data[pos + 1] != 0x00 && data[pos + 1] != 0xFF {
            let marker = data[pos + 1];

            if marker == 0xC0 {
                // SOF0
                height = ((data[pos + 5] as u16) << 8) | (data[pos + 6] as u16);
                width = ((data[pos + 7] as u16) << 8) | (data[pos + 8] as u16);
                num_components = data[pos + 9];
                for i in 0..num_components as usize {
                    let sampling = data[pos + 11 + i * 3];
                    h_samp[i] = sampling >> 4;
                    v_samp[i] = sampling & 0xF;
                }
                println!("SOF0: {}x{}, {} components", width, height, num_components);
                for i in 0..num_components as usize {
                    println!("  Component {}: h={}, v={}", i + 1, h_samp[i], v_samp[i]);
                }
            }

            if marker == 0xC4 {
                // DHT
                let length = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
                let mut dht_pos = pos + 4;
                let dht_end = pos + 2 + length;

                while dht_pos < dht_end {
                    let info = data[dht_pos];
                    let table_class = (info >> 4) & 0x0F; // 0 = DC, 1 = AC
                    let table_id = info & 0x0F;
                    dht_pos += 1;

                    // Read bit counts
                    let mut bit_counts = [0u8; 16];
                    bit_counts.copy_from_slice(&data[dht_pos..dht_pos + 16]);
                    dht_pos += 16;

                    // Calculate total symbols
                    let total_symbols: usize = bit_counts.iter().map(|&c| c as usize).sum();

                    // Read symbols
                    let symbols: Vec<u8> = data[dht_pos..dht_pos + total_symbols].to_vec();
                    dht_pos += total_symbols;

                    println!(
                        "DHT: class={} ({}), id={}, {} symbols",
                        table_class,
                        if table_class == 0 { "DC" } else { "AC" },
                        table_id,
                        total_symbols
                    );

                    // Build decode table
                    match HuffmanDecodeTable::from_bits_values(&bit_counts, &symbols) {
                        Ok(table) => {
                            if table_class == 0 {
                                dc_tables[table_id as usize] = Some(table);
                            } else {
                                ac_tables[table_id as usize] = Some(table);
                            }
                        }
                        Err(e) => {
                            println!("ERROR building Huffman table: {:?}", e);
                        }
                    }
                }
            }

            if marker == 0xDA {
                // SOS
                let length = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
                scan_start = pos + 2 + length;
                println!("SOS: scan data starts at 0x{:06X}", scan_start);
                break;
            }

            // Skip marker with length
            if marker != 0xD8
                && marker != 0xD9
                && !(0xD0..=0xD7).contains(&marker)
                && pos + 3 < data.len()
            {
                let length = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
                pos += 2 + length;
            } else {
                pos += 2;
            }
        } else {
            pos += 1;
        }
    }

    if scan_start == 0 {
        println!("ERROR: Could not find SOS marker");
        return;
    }

    // Set up entropy decoder
    let scan_data = &data[scan_start..];
    println!("Scan data length: {} bytes", scan_data.len());
    let mut decoder = EntropyDecoder::new(scan_data);

    // Set up Huffman tables
    for i in 0..4 {
        if let Some(dc) = dc_tables[i].take() {
            decoder.set_dc_table(i, dc);
        } else if i == 0 {
            decoder.set_dc_table(i, HuffmanDecodeTable::std_dc_luminance());
        } else {
            decoder.set_dc_table(i, HuffmanDecodeTable::std_dc_chrominance());
        }

        if let Some(ac) = ac_tables[i].take() {
            decoder.set_ac_table(i, ac);
        } else if i == 0 {
            decoder.set_ac_table(i, HuffmanDecodeTable::std_ac_luminance());
        } else {
            decoder.set_ac_table(i, HuffmanDecodeTable::std_ac_chrominance());
        }
    }

    // Calculate MCU structure
    let max_h_samp = h_samp[0..num_components as usize]
        .iter()
        .max()
        .copied()
        .unwrap_or(1);
    let max_v_samp = v_samp[0..num_components as usize]
        .iter()
        .max()
        .copied()
        .unwrap_or(1);
    let mcu_width = max_h_samp as usize * 8;
    let mcu_height = max_v_samp as usize * 8;
    let mcu_cols = (width as usize + mcu_width - 1) / mcu_width;
    let mcu_rows = (height as usize + mcu_height - 1) / mcu_height;
    let total_mcus = mcu_cols * mcu_rows;

    println!(
        "MCU structure: {}x{} pixels, {}x{} MCUs = {} total MCUs",
        mcu_width, mcu_height, mcu_cols, mcu_rows, total_mcus
    );

    // Decode MCUs and track progress
    let mut mcu_count = 0;
    let mut last_report = 0;

    'outer: for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            // For each component
            for comp in 0..num_components as usize {
                let h = h_samp[comp] as usize;
                let v = v_samp[comp] as usize;
                let dc_idx = if comp == 0 { 0 } else { 1 };
                let ac_idx = if comp == 0 { 0 } else { 1 };

                // Decode h*v blocks for this component
                for _by in 0..v {
                    for _bx in 0..h {
                        match decoder.decode_block(comp, dc_idx, ac_idx) {
                            Ok(_) => {}
                            Err(e) => {
                                println!(
                                    "\nERROR at MCU ({}, {}), component {}: {:?}",
                                    mcu_x, mcu_y, comp, e
                                );
                                println!(
                                    "MCUs decoded: {} / {} ({:.2}%)",
                                    mcu_count,
                                    total_mcus,
                                    mcu_count as f64 / total_mcus as f64 * 100.0
                                );
                                println!(
                                    "Decoder position: {} / {} bytes",
                                    decoder.position(),
                                    scan_data.len()
                                );
                                break 'outer;
                            }
                        }
                    }
                }
            }

            mcu_count += 1;

            // Report every 10%
            if mcu_count * 10 / total_mcus > last_report {
                last_report = mcu_count * 10 / total_mcus;
                println!("Progress: {}% ({} MCUs)", last_report * 10, mcu_count);
            }
        }
    }

    if mcu_count == total_mcus {
        println!("SUCCESS: All {} MCUs decoded", total_mcus);
    }
}

fn main() {
    let path = "/home/lilith/work/codec-corpus/jpeg-conformance/valid/Reconyx_HC500_Hyperfire.jpg";
    let data = fs::read(path).expect("read file");
    println!("File size: {} bytes", data.len());

    // Manual decode to trace MCU progress
    trace_decode(&data);

    // Now try decoding
    println!("\n--- Attempting decode ---");
    let decoder = jpegli::Decoder::new();
    match decoder.decode(&data) {
        Ok(img) => {
            println!("SUCCESS: {}x{}", img.width, img.height);
        }
        Err(e) => {
            println!("FAILED: {:?}", e);
        }
    }

    // Compare with jpeg-decoder
    println!("\n--- jpeg-decoder ---");
    let mut ref_dec = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&data[..]));
    match ref_dec.decode() {
        Ok(pixels) => {
            let info = ref_dec.dimensions().unwrap();
            println!(
                "SUCCESS: {}x{}, {} bytes",
                info.0,
                info.1,
                pixels.len()
            );
        }
        Err(e) => {
            println!("FAILED: {:?}", e);
        }
    }
}
