use std::fs;

fn main() {
    let rust_path = "/tmp/xyb_matrix_rust_prog.jpg";
    let cpp_path = "/tmp/xyb_matrix_cpp.jpg";

    println!("=== Rust XYB Progressive DHT Analysis ===");
    analyze_dht(rust_path);

    println!("\n=== C++ XYB Progressive DHT Analysis ===");
    analyze_dht(cpp_path);
}

fn analyze_dht(path: &str) {
    let data = fs::read(path).unwrap();
    let mut i = 0;
    let mut dht_num = 0;

    while i + 3 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xC4 {
            // DHT marker
            let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            dht_num += 1;
            println!(
                "\nDHT #{} at offset {}, total length: {} bytes",
                dht_num, i, length
            );

            // Parse tables within this DHT
            let mut offset = i + 4;
            let end = i + 2 + length;
            let mut table_count = 0;

            while offset < end {
                table_count += 1;
                let info = data[offset];
                let table_class = info >> 4; // 0 = DC, 1 = AC
                let table_id = info & 0x0F;

                println!(
                    "  Table #{}: {} table {}",
                    table_count,
                    if table_class == 0 { "DC" } else { "AC" },
                    table_id
                );

                // Read lengths
                let mut total_symbols = 0u16;
                for i in 1..=16 {
                    total_symbols += data[offset + i] as u16;
                }

                let table_size = 1 + 16 + total_symbols as usize; // Info byte + 16 lengths + symbols
                println!("    {} symbols, {} bytes total", total_symbols, table_size);

                offset += table_size;
            }

            i += 2 + length;
        } else {
            i += 1;
        }
    }
}
