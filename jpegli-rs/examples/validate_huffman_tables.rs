//! Validate Huffman table validity
//!
//! Check if optimized Huffman tables satisfy the Kraft inequality
//! and other JPEG validity constraints.

use jpegli::{Encoder, PixelFormat};

/// Check if a Huffman table satisfies the Kraft inequality.
///
/// For a valid prefix-free code, we need:
/// sum(2^(-length_i)) <= 1
///
/// For JPEG with max length 16, this is equivalent to:
/// sum(bits[i] * 2^(16-i)) <= 2^16
fn check_kraft_inequality(bits: &[u8; 16]) -> (bool, u64, u64) {
    let mut sum: u64 = 0;
    for (i, &count) in bits.iter().enumerate() {
        let length = (i + 1) as u32;
        // 2^(16 - length) * count
        let codes_at_length = (count as u64) << (16 - length);
        sum += codes_at_length;
    }

    let max_codes = 1u64 << 16; // 2^16 = 65536
    (sum <= max_codes, sum, max_codes)
}

/// Parse DHT marker and extract tables.
fn parse_dht_tables(jpeg: &[u8]) -> Vec<(u8, u8, [u8; 16], Vec<u8>)> {
    let mut tables = Vec::new();
    let mut i = 0;

    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xC4 {
            // Found DHT marker
            if i + 4 > jpeg.len() {
                break;
            }

            let length = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            if i + 2 + length > jpeg.len() {
                break;
            }

            let mut offset = i + 4;
            let end = i + 2 + length;

            while offset < end {
                if offset + 17 > jpeg.len() {
                    break;
                }

                let tc_th = jpeg[offset];
                let tc = (tc_th >> 4) & 0x0F;
                let th = tc_th & 0x0F;

                let mut bits = [0u8; 16];
                bits.copy_from_slice(&jpeg[offset + 1..offset + 17]);

                let total_symbols: usize = bits.iter().map(|&b| b as usize).sum();

                if offset + 17 + total_symbols > jpeg.len() {
                    break;
                }

                let values = jpeg[offset + 17..offset + 17 + total_symbols].to_vec();

                tables.push((tc, th, bits, values));
                offset += 17 + total_symbols;
            }

            i += 2 + length;
        } else {
            i += 1;
        }
    }

    tables
}

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];

    println!("=== Validating Optimized Huffman Tables ===\n");

    let jpeg = Encoder::new()
        .width(info.0)
        .height(info.1)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();

    println!("JPEG size: {} bytes\n", jpeg.len());

    let tables = parse_dht_tables(&jpeg);
    println!("Found {} Huffman table(s)\n", tables.len());

    let mut all_valid = true;

    for (i, (tc, th, bits, values)) in tables.iter().enumerate() {
        let table_type = if *tc == 0 { "DC" } else { "AC" };
        println!("Table #{}: {} (class={}, id={})", i + 1, table_type, tc, th);

        // Check Kraft inequality
        let (kraft_ok, sum, max) = check_kraft_inequality(bits);
        println!(
            "  Kraft inequality: sum={}, max={}, valid={}",
            sum,
            max,
            if kraft_ok { "✓" } else { "✗" }
        );

        if !kraft_ok {
            all_valid = false;
            println!("  ERROR: Violates Kraft inequality! (sum > 2^16)");
        }

        // Check total symbols
        let total_symbols: usize = bits.iter().map(|&b| b as usize).sum();
        println!(
            "  Total symbols: {} (values.len={})",
            total_symbols,
            values.len()
        );

        if total_symbols != values.len() {
            all_valid = false;
            println!("  ERROR: Symbol count mismatch!");
        }

        // Check for invalid bit lengths (all zeros)
        if bits.iter().all(|&b| b == 0) {
            all_valid = false;
            println!("  ERROR: All bit lengths are zero!");
        }

        // Check for excessive codes at any length
        for (len_minus_1, &count) in bits.iter().enumerate() {
            let length = len_minus_1 + 1;
            let max_at_length = 1 << length; // 2^length
            if count as u32 > max_at_length {
                all_valid = false;
                println!(
                    "  ERROR: {} codes at length {} (max possible: {})",
                    count, length, max_at_length
                );
            }
        }

        // Display bit distribution
        print!("  Bit lengths: ");
        for &b in bits.iter() {
            print!("{} ", b);
        }
        println!();

        // Check for duplicate values
        let mut sorted_values = values.clone();
        sorted_values.sort_unstable();
        let orig_len = sorted_values.len();
        sorted_values.dedup();
        if sorted_values.len() != orig_len {
            all_valid = false;
            println!("  ERROR: Duplicate symbol values detected!");
        }

        println!();
    }

    if all_valid {
        println!("✓ All Huffman tables are VALID");
    } else {
        println!("✗ Some Huffman tables are INVALID");
    }

    // Test if zune-jpeg can decode it
    println!("\n=== Testing with zune-jpeg ===");
    let mut zune_decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&jpeg));
    match zune_decoder.decode() {
        Ok(_) => println!("✓ zune-jpeg can decode Rust JPEG"),
        Err(e) => println!("✗ zune-jpeg failed on Rust JPEG: {:?}", e),
    }

    // Compare with C++ tables
    if let Some(cjpegli) = jpegli::test_utils::find_cjpegli() {
        use std::fs;
        use std::io::Write;
        use std::process::Command;

        let ppm_path = "/tmp/validate_huff.ppm";
        let cpp_path = "/tmp/validate_huff.jpg";

        let mut ppm = fs::File::create(ppm_path).unwrap();
        writeln!(ppm, "P6").unwrap();
        writeln!(ppm, "{} {}", info.0, info.1).unwrap();
        writeln!(ppm, "255").unwrap();
        ppm.write_all(rgb).unwrap();
        drop(ppm);

        let output = Command::new(&cjpegli)
            .args([ppm_path, cpp_path, "-q", "90", "-p", "0"])
            .output()
            .unwrap();

        if output.status.success() {
            let cpp_jpeg = fs::read(cpp_path).unwrap();
            println!("\n=== C++ Huffman Tables ===\n");

            let cpp_tables = parse_dht_tables(&cpp_jpeg);
            println!("Found {} table(s)\n", cpp_tables.len());

            for (i, (tc, th, bits, _values)) in cpp_tables.iter().enumerate() {
                let table_type = if *tc == 0 { "DC" } else { "AC" };
                println!("Table #{}: {} (class={}, id={})", i + 1, table_type, tc, th);

                let (kraft_ok, sum, max) = check_kraft_inequality(bits);
                println!(
                    "  Kraft inequality: sum={}, max={}, valid={}",
                    sum,
                    max,
                    if kraft_ok { "✓" } else { "✗" }
                );

                print!("  Bit lengths: ");
                for &b in bits.iter() {
                    print!("{} ", b);
                }
                println!("\n");
            }

            let mut zune_decoder2 = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&cpp_jpeg));
            match zune_decoder2.decode() {
                Ok(_) => println!("✓ zune-jpeg can decode C++ JPEG"),
                Err(e) => println!("✗ zune-jpeg failed on C++ JPEG: {:?}", e),
            }
        }
    }
}
