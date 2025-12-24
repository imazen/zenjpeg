//! Compare DCT coefficients between C++ and Rust encoded JPEGs

use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    let png_path = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";

    println!("=== DCT Coefficient Comparison ===\n");

    let (original, width, height) = load_png(png_path).expect("Failed to load PNG");

    let ppm_path = "/tmp/coeff_compare.ppm";
    write_ppm(ppm_path, &original, width as usize, height as usize).expect("Failed to write PPM");

    let cpp_jpg = "/tmp/coeff_cpp.jpg";
    Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p",
            "0",
            "--fixed_code",
            ppm_path,
            cpp_jpg,
            "-q",
            "90",
        ])
        .output()
        .expect("Failed to run cjpegli");

    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&original)
        .expect("Failed to encode");

    let rust_jpg = "/tmp/coeff_rust.jpg";
    fs::write(rust_jpg, &rust_jpeg).expect("Failed to write Rust JPEG");

    let cpp_data = fs::read(cpp_jpg).expect("Failed to read C++ JPEG");
    let rust_data = &rust_jpeg;

    println!("C++ JPEG: {} bytes", cpp_data.len());
    println!("Rust JPEG: {} bytes", rust_data.len());
    println!();

    let cpp_qtables = extract_quant_tables(&cpp_data);
    let rust_qtables = extract_quant_tables(rust_data);

    println!("=== Quantization Tables ===\n");

    for (idx, (cpp_qt, rust_qt)) in cpp_qtables.iter().zip(rust_qtables.iter()).enumerate() {
        let name = match idx {
            0 => "Y (Luminance)",
            1 => "Cb/Cr (Chrominance)",
            _ => "Unknown",
        };
        println!("Table {} ({}):", idx, name);

        let mut diffs = 0;
        for i in 0..64 {
            if cpp_qt[i] != rust_qt[i] {
                diffs += 1;
            }
        }

        if diffs == 0 {
            println!("  IDENTICAL");
        } else {
            println!("  {} differences:", diffs);
            for i in 0..64 {
                if cpp_qt[i] != rust_qt[i] {
                    println!("    [{}]: C++={}, Rust={}", i, cpp_qt[i], rust_qt[i]);
                }
            }
        }
        println!();
    }

    let (cpp_scan_start, cpp_scan_end) = find_scan_data(&cpp_data);
    let (rust_scan_start, rust_scan_end) = find_scan_data(rust_data);

    println!("=== Scan Data (Entropy-Coded Coefficients) ===\n");
    println!("C++ scan data: {} bytes", cpp_scan_end - cpp_scan_start);
    println!("Rust scan data: {} bytes", rust_scan_end - rust_scan_start);
    println!(
        "Difference: {:+} bytes ({:+.2}%)",
        (rust_scan_end - rust_scan_start) as i64 - (cpp_scan_end - cpp_scan_start) as i64,
        100.0
            * ((rust_scan_end - rust_scan_start) as f64 / (cpp_scan_end - cpp_scan_start) as f64
                - 1.0)
    );
    println!();

    println!("=== Huffman Table Comparison ===\n");
    let cpp_huff = extract_huffman_tables(&cpp_data);
    let rust_huff = extract_huffman_tables(rust_data);

    for (i, (cpp_ht, rust_ht)) in cpp_huff.iter().zip(rust_huff.iter()).enumerate() {
        let class = if i < 2 { "DC" } else { "AC" };
        let comp = if i % 2 == 0 { "Luma" } else { "Chroma" };

        if cpp_ht == rust_ht {
            println!(
                "{} {} table: IDENTICAL ({} codes)",
                class,
                comp,
                cpp_ht.len()
            );
        } else {
            println!("{} {} table: DIFFERENT", class, comp);
            println!(
                "  C++:  {} codes, first 10: {:?}",
                cpp_ht.len(),
                &cpp_ht[..10.min(cpp_ht.len())]
            );
            println!(
                "  Rust: {} codes, first 10: {:?}",
                rust_ht.len(),
                &rust_ht[..10.min(rust_ht.len())]
            );
        }
    }
}

fn extract_quant_tables(data: &[u8]) -> Vec<[u16; 64]> {
    let mut tables = Vec::new();
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xDB {
            let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
            let mut j = i + 4;
            while j < i + 2 + len {
                let pq_tq = data[j];
                let precision = (pq_tq >> 4) & 0x0F;
                j += 1;
                let mut table = [0u16; 64];
                for k in 0..64 {
                    if precision == 0 {
                        table[k] = data[j] as u16;
                        j += 1;
                    } else {
                        table[k] = ((data[j] as u16) << 8) | data[j + 1] as u16;
                        j += 2;
                    }
                }
                tables.push(table);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    tables
}

fn find_scan_data(data: &[u8]) -> (usize, usize) {
    let mut i = 0;
    let mut scan_start = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
            scan_start = i + 2 + len;
            break;
        }
        i += 1;
    }
    let mut scan_end = data.len();
    i = scan_start;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xD9 {
            scan_end = i;
            break;
        }
        i += 1;
    }
    (scan_start, scan_end)
}

fn extract_huffman_tables(data: &[u8]) -> Vec<Vec<u8>> {
    let mut tables = Vec::new();
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xC4 {
            let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
            let mut j = i + 4;
            while j < i + 2 + len {
                j += 1;
                let mut total_codes = 0usize;
                for k in 0..16 {
                    total_codes += data[j + k] as usize;
                }
                j += 16;
                let mut symbols = Vec::new();
                for _ in 0..total_codes {
                    symbols.push(data[j]);
                    j += 1;
                }
                tables.push(symbols);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    tables
}

fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let decoder = png::Decoder::new(fs::File::open(path).ok()?);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let bytes = &buf[..info.buffer_size()];
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6\n{} {}\n255", width, height)?;
    file.write_all(rgb)
}
