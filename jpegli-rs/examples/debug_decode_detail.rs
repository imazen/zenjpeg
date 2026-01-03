use jpegli::encode::Encoder;
use jpegli::quant::Quality;

/// A minimal decoder that prints detailed debug info
fn debug_decode(jpeg: &[u8]) -> Result<(), String> {
    let mut pos = 0;

    // Check SOI
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return Err("Missing SOI".to_string());
    }
    pos = 2;

    let mut width = 0u32;
    let mut height = 0u32;
    let mut num_components = 0u8;
    let mut h_samp = [0u8; 4];
    let mut v_samp = [0u8; 4];

    // Parse markers
    while pos < jpeg.len() - 1 {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let marker = jpeg[pos + 1];
        pos += 2;

        match marker {
            0xD8 => continue, // SOI
            0xD9 => break,    // EOI
            0xDA => {
                // SOS - Start Of Scan
                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                let num_scan_components = jpeg[pos + 2];
                println!("SOS: {} components in scan", num_scan_components);

                let scan_start = pos + len;
                println!("Entropy data starts at: 0x{:04x}", scan_start);

                // Find end of entropy data
                let mut ecs_end = scan_start;
                while ecs_end < jpeg.len() - 1 {
                    if jpeg[ecs_end] == 0xFF
                        && jpeg[ecs_end + 1] != 0x00
                        && jpeg[ecs_end + 1] != 0xFF
                    {
                        break;
                    }
                    ecs_end += 1;
                }

                let ecs_len = ecs_end - scan_start;
                println!("Entropy data length: {} bytes", ecs_len);

                // Calculate expected data
                let mcu_width = ((width as usize) + 7) / 8;
                let mcu_height = ((height as usize) + 7) / 8;
                let total_mcus = mcu_width * mcu_height;
                let blocks_per_mcu: usize = (0..num_components as usize)
                    .map(|i| (h_samp[i] as usize) * (v_samp[i] as usize))
                    .sum();

                println!("Image: {}x{}", width, height);
                println!("MCUs: {}x{} = {} total", mcu_width, mcu_height, total_mcus);
                println!("Blocks per MCU: {}", blocks_per_mcu);
                println!("Total blocks to decode: {}", total_mcus * blocks_per_mcu);

                // Try to decode and trace
                debug_entropy_decode(
                    &jpeg[scan_start..ecs_end],
                    total_mcus,
                    blocks_per_mcu,
                    num_components,
                );

                return Ok(());
            }
            0xC0 | 0xC2 => {
                // SOF
                let _len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                let precision = jpeg[pos + 2];
                height = ((jpeg[pos + 3] as u32) << 8) | (jpeg[pos + 4] as u32);
                width = ((jpeg[pos + 5] as u32) << 8) | (jpeg[pos + 6] as u32);
                num_components = jpeg[pos + 7];

                println!(
                    "SOF: {}x{}, {} bits, {} components",
                    width, height, precision, num_components
                );

                for i in 0..num_components as usize {
                    let offset = pos + 8 + i * 3;
                    let _id = jpeg[offset];
                    let sampling = jpeg[offset + 1];
                    h_samp[i] = sampling >> 4;
                    v_samp[i] = sampling & 0x0F;
                    let _quant = jpeg[offset + 2];
                    println!("  Component {}: {}x{} sampling", i, h_samp[i], v_samp[i]);
                }

                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                pos += len;
            }
            0xC4 | 0xDB | 0xDD | 0xE0..=0xEF | 0xFE => {
                // Skip these markers
                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                pos += len;
            }
            _ => {
                if marker >= 0xC0 && marker <= 0xFE {
                    let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                    pos += len;
                }
            }
        }
    }

    Ok(())
}

fn debug_entropy_decode(data: &[u8], total_mcus: usize, blocks_per_mcu: usize, num_components: u8) {
    println!("\nDecoding entropy data ({} bytes)...", data.len());

    // Simulate the bit reader
    let mut pos = 0;
    let mut bit_buffer: u32 = 0;
    let mut bits_in_buffer: u8 = 0;
    let mut blocks_decoded = 0;
    let mut total_bits_read = 0u64;

    // Helper to read a byte with unstuffing
    let mut read_byte = |pos: &mut usize, data: &[u8]| -> Option<u8> {
        if *pos >= data.len() {
            return None;
        }
        let byte = data[*pos];
        *pos += 1;

        if byte == 0xFF {
            if *pos >= data.len() {
                return None;
            }
            let next = data[*pos];
            if next == 0x00 {
                *pos += 1; // Skip stuffed 0x00
            } else if (0xD0..=0xD7).contains(&next) {
                *pos += 1; // Restart marker, skip
            } else {
                // Found a marker, end of entropy data
                *pos -= 1;
                return None;
            }
        }
        Some(byte)
    };

    // Try to count how many complete bytes we can read
    let mut test_pos = 0;
    let mut readable_bytes = 0;
    while let Some(_) = read_byte(&mut test_pos, data) {
        readable_bytes += 1;
    }
    println!("Readable bytes (after unstuffing): {}", readable_bytes);
    println!("Available bits: {}", readable_bytes * 8);

    // Estimate bits needed
    // Very rough: DC needs ~8-12 bits, AC needs ~10-20 bits per non-zero,
    // typical block has maybe 10 non-zero coefficients
    // Let's assume ~100-200 bits per block as rough estimate
    let estimated_bits_needed = blocks_per_mcu * total_mcus * 100;
    println!("Estimated bits needed (rough): ~{}", estimated_bits_needed);

    // Reset and try to decode
    pos = 0;
    bit_buffer = 0;
    bits_in_buffer = 0;

    // Simulate reading blocks
    let mut failed_at_block = None;
    let mut last_bits_read = 0u64;

    for mcu in 0..total_mcus {
        for block in 0..blocks_per_mcu {
            // Try to read at least 16 bits to decode one Huffman symbol
            while bits_in_buffer < 16 && pos < data.len() {
                if let Some(byte) = read_byte(&mut pos, data) {
                    bit_buffer = (bit_buffer << 8) | (byte as u32);
                    bits_in_buffer += 8;
                    total_bits_read += 8;
                } else {
                    break;
                }
            }

            if bits_in_buffer < 8 {
                failed_at_block = Some((mcu, block, blocks_decoded, total_bits_read, pos));
                break;
            }

            // Consume some bits (simulating decoding)
            // Actual decode would be much more complex
            blocks_decoded += 1;
            last_bits_read = total_bits_read;
        }

        if failed_at_block.is_some() {
            break;
        }
    }

    if let Some((mcu, block, count, bits, byte_pos)) = failed_at_block {
        println!(
            "\nFailed at MCU {}, block {} ({} blocks decoded)",
            mcu, block, count
        );
        println!("Bits read: {}, byte position: {}", bits, byte_pos);
        println!("Data ends at byte: {}", data.len());
    } else {
        println!(
            "\nAll {} blocks could be partially processed",
            blocks_decoded
        );
    }
}

fn main() {
    // Create the 64x64 gradient image that fails
    let width = 64u32;
    let height = 64u32;
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = ((x * 4) % 256) as u8;
            rgb[i + 1] = ((y * 4) % 256) as u8;
            rgb[i + 2] = 128;
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true);

    let jpeg = encoder.encode(&rgb).expect("encode");
    println!("Encoded {} bytes\n", jpeg.len());

    debug_decode(&jpeg).expect("debug decode");
}
