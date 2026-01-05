//! Analyze scan data between SOS markers to find extraneous bytes.
//!
//! **DEPRECATED**: Use `jpeg_inspect` instead:
//!   cargo run --release --example jpeg_inspect -- --scans --compare other.jpg image.jpg

use std::fs;

fn main() {
    let rust_path = std::env::args().nth(1).unwrap_or("/tmp/binary_compare_rust.jpg".to_string());
    let cpp_path = std::env::args().nth(2).unwrap_or("/tmp/binary_compare_cpp.jpg".to_string());

    let rust_jpg = fs::read(&rust_path).unwrap_or_else(|_| {
        eprintln!("Could not read {}", rust_path);
        vec![]
    });
    let cpp_jpg = fs::read(&cpp_path).unwrap_or_else(|_| {
        eprintln!("Could not read {}", cpp_path);
        vec![]
    });

    if !rust_jpg.is_empty() {
        println!("=== Analyzing {} ({} bytes) ===", rust_path, rust_jpg.len());
        analyze_scans(&rust_jpg);
    }

    if !cpp_jpg.is_empty() {
        println!("\n=== Analyzing {} ({} bytes) ===", cpp_path, cpp_jpg.len());
        analyze_scans(&cpp_jpg);
    }
}

fn analyze_scans(data: &[u8]) {
    let mut scan_num = 0;
    let mut i = 0;

    while i < data.len() - 1 {
        if data[i] == 0xFF {
            let marker = data[i + 1];

            if marker == 0xDA {
                // SOS marker - find scan parameters and data
                let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                let scan_start = i + 2 + len;

                // Find end of scan data (next marker)
                let mut scan_end = scan_start;
                while scan_end < data.len() - 1 {
                    if data[scan_end] == 0xFF && data[scan_end + 1] != 0x00 && data[scan_end + 1] != 0xFF {
                        // Found next marker (not stuffed 0xFF00 or padding 0xFFFF)
                        // But skip RST markers (D0-D7)
                        if data[scan_end + 1] >= 0xD0 && data[scan_end + 1] <= 0xD7 {
                            scan_end += 2;
                            continue;
                        }
                        break;
                    }
                    scan_end += 1;
                }

                let scan_data_len = scan_end - scan_start;

                // Get scan parameters
                let ns = data[i + 4] as usize;
                let ss = data[i + 4 + ns * 2 + 1];
                let se = data[i + 4 + ns * 2 + 2];
                let a = data[i + 4 + ns * 2 + 3];
                let ah = (a >> 4) & 0x0F;
                let al = a & 0x0F;

                // Check last few bytes of scan data for potential padding issues
                let last_bytes: Vec<u8> = data[scan_end.saturating_sub(8)..scan_end].to_vec();

                println!("Scan {}: Ss={} Se={} Ah={} Al={}", scan_num, ss, se, ah, al);
                println!("  Data: 0x{:04X} - 0x{:04X} ({} bytes)", scan_start, scan_end, scan_data_len);
                println!("  Last 8 bytes: {:02X?}", last_bytes);

                // Check for 0xFF bytes that might be problematic
                let ff_count = data[scan_start..scan_end].iter().filter(|&&b| b == 0xFF).count();
                let ff00_count = (0..data[scan_start..scan_end].len() - 1)
                    .filter(|&j| data[scan_start + j] == 0xFF && data[scan_start + j + 1] == 0x00)
                    .count();
                println!("  0xFF bytes: {}, 0xFF00 stuffed: {}", ff_count, ff00_count);

                scan_num += 1;
                i = scan_end;
            } else if marker == 0xD9 {
                println!("EOI at 0x{:04X}", i);
                break;
            } else {
                i += 2;
            }
        } else {
            i += 1;
        }
    }
}
