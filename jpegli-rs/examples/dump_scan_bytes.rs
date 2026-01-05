/// Dump the actual bytes around scan boundaries
fn main() {
    let path = std::env::args().nth(1).unwrap_or("/tmp/noise64_q50.jpg".to_string());
    let data = std::fs::read(&path).expect("read file");

    println!("File: {} ({} bytes)\n", path, data.len());

    let mut i = 0;
    let mut scan_num = 0;

    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            scan_num += 1;
            // Parse SOS length
            let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
            let scan_start = i + 2 + len;

            // Find end of scan data
            let mut scan_end = scan_start;
            while scan_end < data.len() - 1 {
                if data[scan_end] == 0xFF && data[scan_end + 1] != 0x00 && data[scan_end + 1] != 0xFF {
                    break;
                }
                scan_end += 1;
            }

            let scan_bytes = scan_end - scan_start;
            let num_comp = data[i + 4];
            let ss = data[i + 5 + num_comp as usize * 2];
            let se = data[i + 6 + num_comp as usize * 2];
            let ahal = data[i + 7 + num_comp as usize * 2];
            let ah = ahal >> 4;
            let al = ahal & 0xF;

            println!(
                "=== Scan {} (Ss={}, Se={}, Ah={}, Al={}) - {} bytes ===",
                scan_num, ss, se, ah, al, scan_bytes
            );

            // Show last 20 bytes of scan data
            let start = scan_end.saturating_sub(20);
            print!("  Last 20 bytes: ");
            for j in start..scan_end {
                print!("{:02X} ", data[j]);
            }
            println!();

            // Show the FF marker after
            println!(
                "  Next marker: FF {:02X} at offset 0x{:04X}",
                data[scan_end + 1], scan_end
            );
            println!();

            i = scan_end;
        } else {
            i += 1;
        }
    }
}
