//! Dump actual Huffman codes from DHT markers to compare Rust vs C++.

use std::fs;

fn main() {
    let rust_jpg = fs::read("/tmp/binary_compare_rust.jpg").unwrap();
    let cpp_jpg = fs::read("/tmp/binary_compare_cpp.jpg").unwrap();

    println!("=== Rust Huffman Codes ===");
    dump_huffman_codes(&rust_jpg);

    println!("\n=== C++ Huffman Codes ===");
    dump_huffman_codes(&cpp_jpg);
}

fn dump_huffman_codes(data: &[u8]) {
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == 0xC4 {
            // DHT marker
            let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);

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

                if pos + 16 > data.len() {
                    println!("  ERROR: Not enough data for counts");
                    break;
                }

                let counts = &data[pos..pos + 16];
                let total_symbols: usize = counts.iter().map(|&c| c as usize).sum();

                pos += 16;

                if pos + total_symbols > data.len() {
                    println!("  ERROR: Not enough data for symbols");
                    break;
                }

                let symbols = &data[pos..pos + total_symbols];

                // Build the code table
                let mut code: u32 = 0;
                let mut sym_idx = 0;

                for (bit_len, &count) in counts.iter().enumerate() {
                    let bits = bit_len + 1;
                    for _ in 0..count {
                        let symbol = symbols[sym_idx];
                        print!("  sym=0x{:02X} len={:2} code=", symbol, bits);
                        // Print code in binary
                        for j in (0..bits).rev() {
                            print!("{}", (code >> j) & 1);
                        }
                        println!(" (0x{:04X})", code);
                        sym_idx += 1;
                        code += 1;
                    }
                    code <<= 1;
                }

                pos += total_symbols;
            }
            i = end;
        } else {
            i += 1;
        }
    }
}
