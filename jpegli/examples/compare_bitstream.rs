//! Compare actual bitstream bytes between C++ and Rust

use std::fs;
use std::process::Command;

fn main() {
    // Create 8x8 gray image (solid 128,128,128)
    let width = 8;
    let height = 8;
    let rgb: Vec<u8> = vec![128; width * height * 3];

    // Write PPM for C++
    let ppm_path = "/tmp/rust_gray8x8.ppm";
    {
        use std::io::Write;
        let mut f = fs::File::create(ppm_path).unwrap();
        writeln!(f, "P6").unwrap();
        writeln!(f, "{} {}", width, height).unwrap();
        writeln!(f, "255").unwrap();
        f.write_all(&rgb).unwrap();
    }

    // Encode with C++
    let cpp_path = "/tmp/cpp_gray8x8.jpg";
    let status = Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p", "0",
            "--fixed_code",
            ppm_path,
            cpp_path,
            "-q", "90",
        ])
        .output()
        .expect("cjpegli failed");

    if !status.status.success() {
        eprintln!("C++ failed: {}", String::from_utf8_lossy(&status.stderr));
        return;
    }

    // Encode with Rust
    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&rgb)
        .expect("Rust encoding failed");

    let cpp_jpeg = fs::read(cpp_path).unwrap();

    println!("=== 8x8 Gray Image Comparison ===");
    println!("C++ size: {} bytes", cpp_jpeg.len());
    println!("Rust size: {} bytes", rust_jpeg.len());
    println!("Difference: {} bytes ({:+.1}%)",
             rust_jpeg.len() as i64 - cpp_jpeg.len() as i64,
             100.0 * (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64);

    // Find first difference
    let min_len = cpp_jpeg.len().min(rust_jpeg.len());
    let mut first_diff = None;
    for i in 0..min_len {
        if cpp_jpeg[i] != rust_jpeg[i] {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(idx) = first_diff {
        println!("\nFirst difference at byte {:#x} ({}):", idx, idx);

        // Show context around the difference
        let start = idx.saturating_sub(8);
        let end = (idx + 16).min(min_len);

        println!("C++  @ {:#04x}: {:02x?}", start, &cpp_jpeg[start..end]);
        println!("Rust @ {:#04x}: {:02x?}", start, &rust_jpeg[start..end]);

        // Parse JPEG structure to identify what section this is
        describe_jpeg_position(&cpp_jpeg, idx);
    } else if cpp_jpeg.len() == rust_jpeg.len() {
        println!("\n✓ Files are identical!");
    } else {
        println!("\nFiles differ only in length");
    }

    // Dump JPEG structure
    println!("\n=== C++ JPEG Structure ===");
    dump_jpeg_structure(&cpp_jpeg);

    println!("\n=== Rust JPEG Structure ===");
    dump_jpeg_structure(&rust_jpeg);

    // Save Rust output for manual inspection
    fs::write("/tmp/rust_gray8x8.jpg", &rust_jpeg).unwrap();
    println!("\nSaved: /tmp/rust_gray8x8.jpg");
}

fn describe_jpeg_position(data: &[u8], pos: usize) {
    let mut i = 0;
    while i < data.len() && i < pos {
        if data[i] == 0xFF && i + 1 < data.len() {
            let marker = data[i + 1];
            let name = match marker {
                0xD8 => "SOI",
                0xE0 => "APP0",
                0xE1 => "APP1",
                0xDB => "DQT",
                0xC0 => "SOF0",
                0xC4 => "DHT",
                0xDA => "SOS",
                0xD9 => "EOI",
                _ => "?",
            };

            if marker == 0xD8 || marker == 0xD9 {
                i += 2;
            } else if marker >= 0xC0 {
                if i + 3 < data.len() {
                    let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                    if pos >= i && pos < i + 2 + len {
                        println!("Position is in {} segment (offset {:#x}, len {})",
                                 name, i, len);
                        return;
                    }
                    i += 2 + len;
                } else {
                    break;
                }
            } else {
                i += 2;
            }
        } else {
            i += 1;
        }
    }
    println!("Position is in scan data (after SOS)");
}

fn dump_jpeg_structure(data: &[u8]) {
    let mut i = 0;
    while i < data.len() {
        if data[i] == 0xFF && i + 1 < data.len() {
            let marker = data[i + 1];
            let name = match marker {
                0xD8 => "SOI",
                0xE0 => "APP0",
                0xE1 => "APP1",
                0xE2 => "APP2",
                0xDB => "DQT",
                0xC0 => "SOF0",
                0xC4 => "DHT",
                0xDA => "SOS",
                0xD9 => "EOI",
                0x00 => { i += 2; continue; }  // Stuffed byte
                _ => "?",
            };

            if marker == 0xD8 {
                println!("  {:#04x}: {} (Start of Image)", i, name);
                i += 2;
            } else if marker == 0xD9 {
                println!("  {:#04x}: {} (End of Image)", i, name);
                i += 2;
            } else if marker >= 0xC0 && i + 3 < data.len() {
                let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                println!("  {:#04x}: {} (len={})", i, name, len);

                if marker == 0xDA {
                    // SOS - scan data follows
                    i += 2 + len;
                    let scan_start = i;
                    // Find next marker (EOI)
                    while i < data.len() - 1 {
                        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
                            break;
                        }
                        i += 1;
                    }
                    let scan_len = i - scan_start;
                    println!("  {:#04x}: SCAN DATA (len={})", scan_start, scan_len);
                } else {
                    i += 2 + len;
                }
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}
