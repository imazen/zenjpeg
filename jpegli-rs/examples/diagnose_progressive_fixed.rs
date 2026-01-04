//! Diagnose why Progressive + Fixed Huffman produces huge files

use jpegli::{Encoder, PixelFormat};

fn analyze_jpeg(data: &[u8], label: &str) {
    println!("\n=== {} ===", label);
    println!("Total size: {} bytes", data.len());

    let mut i = 0;
    let mut scan_num = 0;
    let mut dht_count = 0;
    let mut dht_total_size = 0;

    while i + 1 < data.len() {
        if data[i] == 0xFF {
            let marker = data[i + 1];

            match marker {
                0xC4 => {
                    // DHT - Define Huffman Table
                    dht_count += 1;
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    dht_total_size += length + 2;
                    println!("  DHT #{}: {} bytes", dht_count, length + 2);
                    i += 2 + length;
                }
                0xDA => {
                    // SOS - Start of Scan
                    scan_num += 1;
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;

                    let num_components = data[i + 4];
                    let mut offset = i + 5;

                    // Skip component specifications
                    offset += num_components as usize * 2;

                    let ss = data[offset];
                    let se = data[offset + 1];
                    let ah_al = data[offset + 2];
                    let ah = ah_al >> 4;
                    let al = ah_al & 0x0F;

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

                    let scan_type = if ss == 0 && se == 0 {
                        "DC"
                    } else if ah == 0 {
                        "AC First"
                    } else {
                        "AC Refine"
                    };

                    println!(
                        "  Scan #{:2}: {} Ss={:2} Se={:2} Ah={} Al={} → {:7} bytes",
                        scan_num, scan_type, ss, se, ah, al, scan_size
                    );

                    if scan_size > 10000 {
                        println!("    ⚠️  HUGE SCAN! Sampling first 32 bytes:");
                        print!("      ");
                        for &b in &data[scan_start..scan_start + 32.min(scan_size)] {
                            print!("{:02X} ", b);
                        }
                        println!();

                        // Count patterns
                        let zeros = data[scan_start..scan_end].iter().filter(|&&b| b == 0x00).count();
                        let ffs = data[scan_start..scan_end].iter().filter(|&&b| b == 0xFF).count();
                        println!("      Zeros: {}, 0xFF: {}, Others: {}",
                                 zeros, ffs, scan_size - zeros - ffs);
                    }

                    i = scan_end;
                }
                0xD8 => {
                    // SOI
                    println!("  SOI");
                    i += 2;
                }
                0xD9 => {
                    // EOI
                    println!("  EOI");
                    i += 2;
                }
                _ if marker >= 0xE0 && marker <= 0xEF => {
                    // APP markers
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    println!("  APP{}: {} bytes", marker - 0xE0, length + 2);
                    i += 2 + length;
                }
                _ if marker >= 0xC0 && marker <= 0xCF => {
                    // SOF markers
                    let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                    println!("  SOF{}: {} bytes", marker - 0xC0, length + 2);
                    i += 2 + length;
                }
                _ => {
                    // Other markers
                    if marker != 0x00 && marker != 0xFF {
                        let length = if data.len() > i + 3 {
                            u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize
                        } else {
                            0
                        };
                        if length > 0 {
                            println!("  Marker 0x{:02X}: {} bytes", marker, length + 2);
                            i += 2 + length;
                        } else {
                            i += 2;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
        } else {
            i += 1;
        }
    }

    println!("\nSummary:");
    println!("  DHT tables: {} ({} bytes total)", dht_count, dht_total_size);
    println!("  Scans: {}", scan_num);
}

fn main() {
    // Small test image
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = 128;
        }
    }

    println!("Testing 64x64 gradient at Q90");

    // Baseline + Fixed
    let baseline_fixed = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .use_xyb(false)
        .optimize_huffman(false)
        .encode(&data)
        .unwrap();

    analyze_jpeg(&baseline_fixed, "Baseline + Fixed Huffman");

    // Progressive + Fixed
    let progressive_fixed = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(false)
        .optimize_huffman(false)
        .encode(&data)
        .unwrap();

    analyze_jpeg(&progressive_fixed, "Progressive + Fixed Huffman (BROKEN)");

    // Progressive + Optimized (for comparison)
    let progressive_opt = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(false)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    analyze_jpeg(&progressive_opt, "Progressive + Optimized (Working)");
}
