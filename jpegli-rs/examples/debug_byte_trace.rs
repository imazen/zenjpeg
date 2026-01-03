// Trace exactly which block causes byte 63 to be written differently
// We need to find which block's encoding causes the divergence

use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn find_sos_markers(jpeg: &[u8]) -> Vec<(usize, u8, u8, u8, u8)> {
    let mut markers = Vec::new();
    let mut i = 0;
    while i < jpeg.len() - 10 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
            let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
            let num_components = jpeg[i + 4];
            let spec_offset = i + 5 + (num_components as usize * 2);
            if spec_offset + 2 < jpeg.len() {
                let ss = jpeg[spec_offset];
                let se = jpeg[spec_offset + 1];
                let ah_al = jpeg[spec_offset + 2];
                let ah = ah_al >> 4;
                let al = ah_al & 0x0F;
                let scan_start = i + 2 + len;
                markers.push((scan_start, ss, se, ah, al));
            }
            i += 2 + len;
        } else {
            i += 1;
        }
    }
    markers
}

fn main() {
    // Both 49x49 and 50x50 have 49 blocks but one works and one fails
    // Let's see what's different in the coefficient values

    // First, let's see what coefficients each block has
    // We'll use a simplified approach - look at what the first AC pass (al=2) produces

    for size in [49u32, 50] {
        println!("\n=== {}x{} ===", size, size);
        let data = photo_like(size, size);

        // The differences would be in the rightmost column of blocks
        // because that's where padding differs
        let block_cols = (size + 7) / 8;
        let block_rows = (size + 7) / 8;
        let last_pixel_col = (size % 8) as usize;
        let effective_last = if last_pixel_col == 0 {
            8
        } else {
            last_pixel_col
        };

        println!("Block grid: {}x{}", block_cols, block_rows);
        println!(
            "Last pixel column in rightmost block: {} (0-indexed)",
            last_pixel_col
        );
        println!("Rightmost blocks contain pixels up to index: {}", size - 1);

        // For the rightmost column (block x = 6):
        // - 49x49: pixels 48 (just 1), padded with 0
        // - 50x50: pixels 48, 49 (2 pixels), padded with 0

        // This would cause different DCT coefficients
        // Let's manually look at what pixel values go into the rightmost blocks

        println!("\nRightmost block column pixel values (first 3 rows):");
        for by in 0..3 {
            let bx = block_cols - 1;
            print!("Block ({},{}): [", bx, by);
            for dy in 0..8 {
                let y = by * 8 + dy;
                if y >= size {
                    continue;
                }
                for dx in 0..8 {
                    let x = bx * 8 + dx;
                    if x < size {
                        let idx = (y * size + x) as usize;
                        let val = data[idx];
                        print!("{:3},", val);
                    } else {
                        print!("  _,"); // padded
                    }
                }
                print!(" | ");
            }
            println!("]");
        }

        // Encode and check
        let jpeg = Encoder::new()
            .width(size)
            .height(size)
            .pixel_format(PixelFormat::Gray)
            .jpegli_quality(Quality::from_quality(75.0))
            .optimize_huffman(false)
            .mode(JpegMode::Progressive)
            .encode(&data)
            .expect("encode failed");

        let decode_result = jpeg_decoder::Decoder::new(&jpeg[..]).decode();
        println!(
            "\nDecode result: {}",
            if decode_result.is_ok() { "OK" } else { "FAIL" }
        );

        // Find the refinement scan
        let scans = find_sos_markers(&jpeg);
        for (i, (offset, ss, se, ah, al)) in scans.iter().enumerate() {
            if *ah == 2 && *al == 1 {
                // Find scan end
                let mut end = *offset;
                while end < jpeg.len() - 1 {
                    if jpeg[end] == 0xFF && jpeg[end + 1] != 0x00 && jpeg[end + 1] != 0xFF {
                        break;
                    }
                    end += 1;
                }
                println!(
                    "Refinement scan {}: {} bytes (offset {})",
                    i,
                    end - offset,
                    offset
                );

                // Show bytes 60-70 (around the divergence point at byte 63)
                println!("Bytes 60-80:");
                for b in 60..80.min(end - offset) {
                    print!("{:02X} ", jpeg[*offset + b]);
                }
                println!();
            }
        }
    }

    // Let's also test what happens if we use solid data (no variation)
    // to see if the issue is with the coefficient values or with the encoding logic
    println!("\n=== Solid 50x50 (should work) ===");
    let solid_data: Vec<u8> = vec![128; 50 * 50];
    let jpeg_solid = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&solid_data)
        .expect("encode failed");

    let solid_result = jpeg_decoder::Decoder::new(&jpeg_solid[..]).decode();
    println!(
        "Solid 50x50 decode: {}",
        if solid_result.is_ok() { "OK" } else { "FAIL" }
    );
}
