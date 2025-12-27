//! Compare quantization tables between Rust and C++ for XYB mode.
//!
//! Usage: cargo run --release --example compare_quant_tables

use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;
use std::process::{Command, Stdio};

const CJPEGLI_PATH: &str = "/home/lilith/work/jpegli-rs/jpegli-cpp/build/tools/cjpegli";

fn load_png(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let (width, height) = (info.width as usize, info.height as usize);
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => return None,
    };
    Some((rgb, width, height))
}

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: f32, use_xyb: bool) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(quality))
        .use_xyb(use_xyb)
        .encode(rgb)
        .expect("encode")
}

fn encode_cpp(ppm_path: &str, quality: u32, use_xyb: bool) -> Option<Vec<u8>> {
    if !Path::new(CJPEGLI_PATH).exists() {
        return None;
    }
    let output_path = format!(
        "/tmp/cpp_quant_{}_{}.jpg",
        if use_xyb { "xyb" } else { "ycbcr" },
        quality
    );
    let mut args = vec!["--chroma_subsampling=444", "-p", "0"];
    if use_xyb {
        args.push("--xyb");
    }
    args.push(ppm_path);
    args.push(&output_path);
    args.push("-q");
    let q_str = quality.to_string();
    args.push(&q_str);

    let output = Command::new(CJPEGLI_PATH)
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }
    fs::read(&output_path).ok()
}

fn extract_quant_tables(jpeg: &[u8]) -> Vec<(u8, Vec<u16>)> {
    let mut tables = Vec::new();
    let mut pos = 0;

    while pos < jpeg.len() - 2 {
        if jpeg[pos] == 0xFF {
            let marker = jpeg[pos + 1];
            if marker == 0xDB {
                // DQT marker
                let length = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
                let mut tpos = pos + 4;
                let end = pos + 2 + length;

                while tpos < end {
                    let pq_tq = jpeg[tpos];
                    let precision = (pq_tq >> 4) & 0x0F;
                    let table_id = pq_tq & 0x0F;
                    tpos += 1;

                    let table: Vec<u16> = if precision == 0 {
                        let t: Vec<u16> = jpeg[tpos..tpos + 64].iter().map(|&b| b as u16).collect();
                        tpos += 64;
                        t
                    } else {
                        let mut t = Vec::with_capacity(64);
                        for i in 0..64 {
                            t.push(u16::from_be_bytes([
                                jpeg[tpos + i * 2],
                                jpeg[tpos + i * 2 + 1],
                            ]));
                        }
                        tpos += 128;
                        t
                    };
                    tables.push((table_id, table));
                }
                pos = end;
            } else if marker == 0xD8 || marker == 0xD9 {
                pos += 2;
            } else if marker == 0x00 || marker == 0xFF {
                pos += 1;
            } else if pos + 4 <= jpeg.len() {
                let length = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
                pos += 2 + length;
            } else {
                pos += 2;
            }
        } else {
            pos += 1;
        }
    }
    tables
}

fn print_table(name: &str, table: &[u16]) {
    println!(
        "  {}: min={}, max={}, sum={}",
        name,
        table.iter().min().unwrap(),
        table.iter().max().unwrap(),
        table.iter().map(|&x| x as u32).sum::<u32>()
    );
    println!("  First 8: {:?}", &table[..8]);
}

fn main() {
    println!("=== Quantization Table Comparison ===\n");

    let image_path = "/home/lilith/work/jpegli-rs/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";
    let path = Path::new(image_path);

    let (rgb, width, height) = match load_png(path) {
        Some(d) => d,
        None => {
            eprintln!("Failed to load image");
            return;
        }
    };

    let ppm_path = "/tmp/quant_test.ppm";
    write_ppm(ppm_path, &rgb, width, height).unwrap();

    for quality in [70, 80, 90, 95] {
        println!("=== Quality {} ===\n", quality);

        // XYB mode
        println!("XYB Mode:");

        let rust_xyb = encode_rust(&rgb, width as u32, height as u32, quality as f32, true);
        let rust_tables = extract_quant_tables(&rust_xyb);
        println!("Rust XYB (size={})", rust_xyb.len());
        for (id, table) in &rust_tables {
            print_table(&format!("Table {}", id), table);
        }

        if let Some(cpp_xyb) = encode_cpp(ppm_path, quality, true) {
            let cpp_tables = extract_quant_tables(&cpp_xyb);
            println!("\nC++ XYB (size={})", cpp_xyb.len());
            for (id, table) in &cpp_tables {
                print_table(&format!("Table {}", id), table);
            }

            // Compare tables
            println!("\nTable differences (Rust - C++):");
            for ((rid, rt), (cid, ct)) in rust_tables.iter().zip(cpp_tables.iter()) {
                if rid == cid {
                    let diff: Vec<i32> = rt
                        .iter()
                        .zip(ct.iter())
                        .map(|(&r, &c)| r as i32 - c as i32)
                        .collect();
                    let max_diff = diff.iter().map(|&d| d.abs()).max().unwrap_or(0);
                    let sum_diff: i32 = diff.iter().sum();
                    println!(
                        "  Table {}: max_diff={}, sum_diff={}",
                        rid, max_diff, sum_diff
                    );
                    if max_diff > 0 {
                        println!("    First 8 Rust: {:?}", &rt[..8]);
                        println!("    First 8 C++:  {:?}", &ct[..8]);
                    }
                }
            }
        }

        // YCbCr mode
        println!("\nYCbCr Mode:");

        let rust_ycbcr = encode_rust(&rgb, width as u32, height as u32, quality as f32, false);
        let rust_tables = extract_quant_tables(&rust_ycbcr);
        println!("Rust YCbCr (size={})", rust_ycbcr.len());
        for (id, table) in &rust_tables {
            print_table(&format!("Table {}", id), table);
        }

        if let Some(cpp_ycbcr) = encode_cpp(ppm_path, quality, false) {
            let cpp_tables = extract_quant_tables(&cpp_ycbcr);
            println!("\nC++ YCbCr (size={})", cpp_ycbcr.len());
            for (id, table) in &cpp_tables {
                print_table(&format!("Table {}", id), table);
            }

            // Compare tables
            println!("\nTable differences (Rust - C++):");
            for ((rid, rt), (cid, ct)) in rust_tables.iter().zip(cpp_tables.iter()) {
                if rid == cid {
                    let diff: Vec<i32> = rt
                        .iter()
                        .zip(ct.iter())
                        .map(|(&r, &c)| r as i32 - c as i32)
                        .collect();
                    let max_diff = diff.iter().map(|&d| d.abs()).max().unwrap_or(0);
                    let sum_diff: i32 = diff.iter().sum();
                    println!(
                        "  Table {}: max_diff={}, sum_diff={}",
                        rid, max_diff, sum_diff
                    );
                }
            }
        }

        println!("\n{}\n", "=".repeat(60));
    }
}
