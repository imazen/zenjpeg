//! Diagnose Huffman table encoding issues
//!
//! zune-jpeg reports "Bad Huffman Table" when trying to decode Rust JPEGs.
//! This tool compares Huffman table structures between Rust and C++ encoders.

use jpegli::{Encoder, PixelFormat};
use std::fs;

fn parse_dht_marker(data: &[u8], offset: usize) -> Option<(usize, Vec<u8>)> {
    if offset + 2 > data.len() {
        return None;
    }

    let length = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    if offset + length > data.len() {
        return None;
    }

    Some((length, data[offset..offset + length].to_vec()))
}

fn find_all_dht_markers(data: &[u8]) -> Vec<(usize, Vec<u8>)> {
    let mut dhts = Vec::new();
    let mut i = 0;

    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xC4 {
            // Found DHT marker
            if let Some((length, dht_data)) = parse_dht_marker(data, i + 2) {
                dhts.push((i, dht_data));
                i += 2 + length;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    dhts
}

fn parse_dht_table(dht_data: &[u8]) -> String {
    if dht_data.len() < 2 {
        return "Invalid DHT (too short)".to_string();
    }

    let mut result = String::new();
    let mut offset = 2; // Skip length bytes

    while offset < dht_data.len() {
        if offset >= dht_data.len() {
            break;
        }

        let tc_th = dht_data[offset];
        let tc = (tc_th >> 4) & 0x0F; // Table class (0=DC, 1=AC)
        let th = tc_th & 0x0F; // Table identifier

        result.push_str(&format!(
            "\n  Table class: {} ({}), ID: {}\n",
            tc,
            if tc == 0 { "DC" } else { "AC" },
            th
        ));

        offset += 1;

        // Read 16 bytes of bit lengths
        if offset + 16 > dht_data.len() {
            result.push_str("  ERROR: Truncated bit lengths\n");
            break;
        }

        let mut total_codes = 0u32;
        result.push_str("  Bit lengths: ");
        for i in 0..16 {
            let count = dht_data[offset + i];
            result.push_str(&format!("{} ", count));
            total_codes += count as u32;
        }
        result.push_str(&format!("(total: {})\n", total_codes));

        offset += 16;

        // Read symbol values
        if offset + total_codes as usize > dht_data.len() {
            result.push_str(&format!(
                "  ERROR: Expected {} symbols but only {} bytes remain\n",
                total_codes,
                dht_data.len() - offset
            ));
            break;
        }

        result.push_str(&format!("  Symbols ({} total): ", total_codes));
        for i in 0..total_codes.min(20) {
            result.push_str(&format!("{:02X} ", dht_data[offset + i as usize]));
        }
        if total_codes > 20 {
            result.push_str("...");
        }
        result.push_str("\n");

        offset += total_codes as usize;
    }

    result
}

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    // Load PNG
    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];
    let width = info.0 as u32;
    let height = info.1 as u32;

    println!("=== Encoding with Rust (Baseline) ===");
    let rust_baseline = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Baseline)
        .encode(rgb)
        .unwrap();

    println!("Rust baseline: {} bytes\n", rust_baseline.len());

    // Try to decode with zune-jpeg
    println!("=== Testing zune-jpeg decoder ===");
    let mut zune_decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&rust_baseline));
    match zune_decoder.decode() {
        Ok(_) => println!("✓ zune-jpeg can decode Rust JPEG\n"),
        Err(e) => println!("✗ zune-jpeg FAILED: {:?}\n", e),
    }

    // Find DHT markers in Rust JPEG
    println!("=== Rust JPEG DHT Markers ===");
    let rust_dhts = find_all_dht_markers(&rust_baseline);
    println!("Found {} DHT marker(s)", rust_dhts.len());

    for (i, (offset, dht)) in rust_dhts.iter().enumerate() {
        println!("\nDHT #{} at offset 0x{:04X}:", i + 1, offset);
        println!("  Length: {} bytes", dht.len());
        println!("{}", parse_dht_table(dht));
    }

    // Compare with C++ JPEG
    if let Some(cjpegli) = jpegli::test_utils::find_cjpegli() {
        use std::io::Write;
        use std::process::Command;

        let ppm_path = "/tmp/test_huffman.ppm";
        let cpp_jpeg_path = "/tmp/cpp_baseline.jpg";

        // Write PPM
        let mut ppm = fs::File::create(ppm_path).unwrap();
        writeln!(ppm, "P6").unwrap();
        writeln!(ppm, "{} {}", width, height).unwrap();
        writeln!(ppm, "255").unwrap();
        ppm.write_all(rgb).unwrap();
        drop(ppm);

        // Encode with C++ (baseline)
        let output = Command::new(&cjpegli)
            .args([ppm_path, cpp_jpeg_path, "-q", "90", "-p", "0"])
            .output()
            .unwrap();

        if !output.status.success() {
            eprintln!(
                "C++ encoding failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let cpp_jpeg = fs::read(cpp_jpeg_path).unwrap();
        println!("\n\n=== C++ JPEG DHT Markers ===");
        println!("C++ baseline: {} bytes", cpp_jpeg.len());

        let cpp_dhts = find_all_dht_markers(&cpp_jpeg);
        println!("Found {} DHT marker(s)", cpp_dhts.len());

        for (i, (offset, dht)) in cpp_dhts.iter().enumerate() {
            println!("\nDHT #{} at offset 0x{:04X}:", i + 1, offset);
            println!("  Length: {} bytes", dht.len());
            println!("{}", parse_dht_table(dht));
        }

        // Test C++ JPEG with zune-jpeg
        println!("\n=== Testing C++ JPEG with zune-jpeg ===");
        let mut zune_decoder2 = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&cpp_jpeg));
        match zune_decoder2.decode() {
            Ok(_) => println!("✓ zune-jpeg can decode C++ JPEG"),
            Err(e) => println!("✗ zune-jpeg FAILED: {:?}", e),
        }

        // Byte-level comparison of first DHT
        if !rust_dhts.is_empty() && !cpp_dhts.is_empty() {
            println!("\n=== Byte-level DHT Comparison ===");
            let rust_dht = &rust_dhts[0].1;
            let cpp_dht = &cpp_dhts[0].1;

            println!("Rust DHT length: {}", rust_dht.len());
            println!("C++  DHT length: {}", cpp_dht.len());

            if rust_dht.len() == cpp_dht.len() {
                let mut diffs = 0;
                for (i, (r, c)) in rust_dht.iter().zip(cpp_dht.iter()).enumerate() {
                    if r != c {
                        println!("  Diff at byte {}: Rust={:02X}, C++={:02X}", i, r, c);
                        diffs += 1;
                    }
                }
                if diffs == 0 {
                    println!("✓ First DHT markers are IDENTICAL");
                } else {
                    println!("✗ Found {} byte difference(s)", diffs);
                }
            } else {
                println!("✗ Different lengths");
            }
        }
    } else {
        println!("\nC++ cjpegli not found - skipping C++ comparison");
    }
}
