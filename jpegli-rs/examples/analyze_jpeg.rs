/// Analyze JPEG marker structure
fn main() {
    let path = std::env::args().nth(1).unwrap_or("/tmp/test_noise64.jpg".to_string());
    let data = std::fs::read(&path).expect("read file");

    println!("File: {} ({} bytes)\n", path, data.len());

    let mut i = 0;
    let mut scan_num = 0;
    let mut prev_end = 0;

    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            let marker = data[i + 1];
            let marker_name = match marker {
                0xD8 => "SOI",
                0xD9 => "EOI",
                0xC0 => "SOF0",
                0xC2 => "SOF2",
                0xC4 => "DHT",
                0xDA => "SOS",
                0xDB => "DQT",
                0xDD => "DRI",
                0xE0..=0xEF => "APPn",
                0xFE => "COM",
                0x00 => {
                    i += 1;
                    continue;
                }
                _ => "????",
            };

            if marker == 0xDA {
                scan_num += 1;
                if prev_end > 0 && i > prev_end + 2 {
                    let gap = i - prev_end;
                    println!(
                        "*** GAP: {} bytes between end of scan {} and SOS {}",
                        gap,
                        scan_num - 1,
                        scan_num
                    );
                }
            }

            // Parse length for variable-length markers
            if !matches!(marker, 0xD8 | 0xD9 | 0xD0..=0xD7) && i + 3 < data.len() {
                let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                print!("  {:04X}: {:02X} {} (len={})", i, marker, marker_name, len);

                // For SOS, print scan info
                if marker == 0xDA {
                    let num_comp = data[i + 4];
                    let ss = data[i + 5 + num_comp as usize * 2];
                    let se = data[i + 6 + num_comp as usize * 2];
                    let ahal = data[i + 7 + num_comp as usize * 2];
                    let ah = ahal >> 4;
                    let al = ahal & 0xF;
                    print!(" - Scan {}: {} comp, Ss={}, Se={}, Ah={}, Al={}",
                           scan_num, num_comp, ss, se, ah, al);

                    // Find end of scan data (next marker)
                    let scan_start = i + len + 2;
                    let mut scan_end = scan_start;
                    while scan_end < data.len() - 1 {
                        if data[scan_end] == 0xFF && data[scan_end + 1] != 0x00 && data[scan_end + 1] != 0xFF {
                            break;
                        }
                        scan_end += 1;
                    }
                    let scan_bytes = scan_end - scan_start;
                    print!(" ({} data bytes)", scan_bytes);
                    prev_end = scan_end;
                }
                println!();

                i += 2 + len;
            } else {
                println!("  {:04X}: {:02X} {}", i, marker, marker_name);
                i += 2;
            }
        } else {
            i += 1;
        }
    }
}
