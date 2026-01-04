use std::fs;

fn main() {
    println!("=== Analyzing XYB Progressive Structure ===\n");

    let rust_path = "/tmp/xyb_matrix_rust_prog.jpg";
    let cpp_path = "/tmp/xyb_matrix_cpp.jpg";

    println!("Rust XYB Progressive:");
    analyze_jpeg(rust_path);

    println!("\n{}\n", "=".repeat(60));

    println!("C++ XYB Progressive:");
    analyze_jpeg(cpp_path);
}

fn analyze_jpeg(path: &str) {
    let data = fs::read(path).unwrap();
    println!("File size: {} bytes\n", data.len());

    let mut i = 0;
    let mut dht_count = 0;
    let mut dqt_count = 0;
    let mut sos_count = 0;
    let mut scan_sizes = Vec::new();

    while i + 1 < data.len() {
        if data[i] == 0xFF {
            let marker = data[i + 1];
            match marker {
                0xC0 | 0xC2 => {
                    // SOF
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    println!("SOF{} at {}: length {}", if marker == 0xC0 { "0" } else { "2" }, i, length);
                    i += 2 + length;
                }
                0xC4 => {
                    // DHT
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    dht_count += 1;
                    println!("DHT at {}: length {} bytes", i, length);
                    i += 2 + length;
                }
                0xDB => {
                    // DQT
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    dqt_count += 1;
                    println!("DQT at {}: length {} bytes", i, length);
                    i += 2 + length;
                }
                0xDA => {
                    // SOS
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    let num_components = data[i + 4];
                    sos_count += 1;

                    println!("\nSOS #{} at {}: {} component(s)", sos_count, i, num_components);

                    let mut offset = i + 5;
                    for comp in 0..num_components {
                        let comp_id = data[offset];
                        let tables = data[offset + 1];
                        let dc_table = tables >> 4;
                        let ac_table = tables & 0x0F;
                        println!("  Component {}: ID={} ('{}'), DC={}, AC={}",
                            comp, comp_id, comp_id as char, dc_table, ac_table);
                        offset += 2;
                    }

                    let ss = data[offset];
                    let se = data[offset + 1];
                    let ah_al = data[offset + 2];
                    let ah = ah_al >> 4;
                    let al = ah_al & 0x0F;
                    println!("  Spectral: Ss={}, Se={}", ss, se);
                    println!("  Successive approx: Ah={}, Al={}", ah, al);

                    // Find scan data end
                    let scan_start = i + 2 + length;
                    let mut scan_end = scan_start;
                    while scan_end + 1 < data.len() {
                        if data[scan_end] == 0xFF && data[scan_end + 1] != 0x00 && data[scan_end + 1] != 0xFF {
                            break;
                        }
                        scan_end += 1;
                    }

                    let scan_size = scan_end - scan_start;
                    scan_sizes.push(scan_size);
                    println!("  Scan data: {} bytes", scan_size);

                    i = scan_end;
                }
                0xD9 => {
                    // EOI
                    println!("\nEOI at {}", i);
                    break;
                }
                _ if marker >= 0xE0 && marker <= 0xEF => {
                    // APP markers
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    i += 2 + length;
                }
                _ => {
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
    }

    println!("\n=== Summary ===");
    println!("DHT markers: {}", dht_count);
    println!("DQT markers: {}", dqt_count);
    println!("Scans: {}", sos_count);
    println!("Total scan data: {} bytes", scan_sizes.iter().sum::<usize>());
    println!("Average scan size: {:.1} bytes", scan_sizes.iter().sum::<usize>() as f64 / scan_sizes.len() as f64);
    println!("Scan sizes: {:?}", scan_sizes);
}
