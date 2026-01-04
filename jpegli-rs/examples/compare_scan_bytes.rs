use std::fs;

fn main() {
    let data = fs::read("/tmp/xyb_matrix_rust_prog.jpg").unwrap();

    // Extract scan #6 and #7 data
    let scan6 = extract_scan(&data, 6);
    let scan7 = extract_scan(&data, 7);

    println!("Scan #6: {} bytes", scan6.len());
    println!("Scan #7: {} bytes", scan7.len());
    println!();

    if scan6 == scan7 {
        println!("✗ SCANS ARE IDENTICAL!");
    } else {
        println!("✓ Scans are different");

        // Find first difference
        for (i, (&b6, &b7)) in scan6.iter().zip(scan7.iter()).enumerate() {
            if b6 != b7 {
                println!("  First diff at byte {}: 0x{:02X} vs 0x{:02X}", i, b6, b7);
                break;
            }
        }
    }
}

fn extract_scan(data: &[u8], target_scan: usize) -> Vec<u8> {
    let mut i = 0;
    let mut scan_num = 0;

    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            scan_num += 1;

            if scan_num == target_scan {
                let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                let scan_start = i + 2 + length;
                let mut scan_end = scan_start;

                while scan_end + 1 < data.len() {
                    if data[scan_end] == 0xFF
                        && data[scan_end + 1] != 0x00
                        && data[scan_end + 1] != 0xFF
                    {
                        break;
                    }
                    scan_end += 1;
                }

                return data[scan_start..scan_end].to_vec();
            }

            i += 2;
        } else {
            i += 1;
        }
    }

    vec![]
}
