//! Analyze DHT (Huffman table) markers to find encoding differences.

use std::fs;

fn main() {
    let rust_jpg = fs::read("/tmp/binary_compare_rust.jpg").unwrap();
    let cpp_jpg = fs::read("/tmp/binary_compare_cpp.jpg").unwrap();

    println!("=== Rust DHT markers ===");
    analyze_dht_markers(&rust_jpg);

    println!("\n=== C++ DHT markers ===");
    analyze_dht_markers(&cpp_jpg);
}

fn analyze_dht_markers(data: &[u8]) {
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == 0xC4 {
            // DHT marker
            let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
            println!("\nDHT at 0x{:04X}, length {}", i, len);

            let mut pos = i + 4; // Skip marker and length
            let end = i + 2 + len;

            while pos < end && pos < data.len() {
                let table_spec = data[pos];
                let table_class = (table_spec >> 4) & 0x0F;
                let table_id = table_spec & 0x0F;

                println!(
                    "\n  Table: class={} ({}), id={}",
                    table_class,
                    if table_class == 0 { "DC" } else { "AC" },
                    table_id
                );

                pos += 1;

                // Read 16 bytes of code counts
                if pos + 16 > data.len() {
                    println!("  ERROR: Not enough data for counts");
                    break;
                }
                let counts = &data[pos..pos + 16];
                let total_symbols: usize = counts.iter().map(|&c| c as usize).sum();

                println!("  Counts (bits 1-16): {:?}", counts);
                println!("  Total symbols: {}", total_symbols);

                pos += 16;

                // Read symbol values
                if pos + total_symbols > data.len() {
                    println!("  ERROR: Not enough data for symbols");
                    break;
                }
                let symbols = &data[pos..pos + total_symbols];

                // Print symbols grouped by bit length
                let mut sym_idx = 0;
                for (bit_len, &count) in counts.iter().enumerate() {
                    if count > 0 {
                        print!("  {} bits: ", bit_len + 1);
                        for _ in 0..count {
                            print!("0x{:02X} ", symbols[sym_idx]);
                            sym_idx += 1;
                        }
                        println!();
                    }
                }

                pos += total_symbols;
            }
            i = end;
        } else {
            i += 1;
        }
    }
}
