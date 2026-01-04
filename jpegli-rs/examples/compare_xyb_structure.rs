//! Compare JPEG structure between Rust and C++ XYB encodings

use jpegli::{Encoder, PixelFormat};
use std::fs;
use std::io::Write;
use std::process::Command;

fn dump_jpeg_structure(data: &[u8], label: &str) {
    println!("\n=== {} ===", label);
    println!("Total size: {} bytes\n", data.len());

    let mut i = 0;
    while i < data.len().saturating_sub(1) {
        if data[i] == 0xFF {
            let marker = data[i + 1];

            // Skip padding bytes
            if marker == 0xFF {
                i += 1;
                continue;
            }

            let marker_name = match marker {
                0xD8 => "SOI (Start of Image)",
                0xD9 => "EOI (End of Image)",
                0xC0 => "SOF0 (Start of Frame - Baseline)",
                0xC2 => "SOF2 (Start of Frame - Progressive)",
                0xC4 => "DHT (Define Huffman Table)",
                0xDB => "DQT (Define Quantization Table)",
                0xDD => "DRI (Define Restart Interval)",
                0xDA => "SOS (Start of Scan)",
                0xE0 => "APP0 (Application)",
                0xE1 => "APP1 (Application)",
                0xE2 => "APP2 (Application - ICC)",
                0xFE => "COM (Comment)",
                _ if marker >= 0xD0 && marker <= 0xD7 => "RST (Restart)",
                _ if marker >= 0xE0 && marker <= 0xEF => "APP (Application)",
                _ => "Unknown",
            };

            if marker == 0xD8 || marker == 0xD9 || (marker >= 0xD0 && marker <= 0xD7) {
                // No length field
                println!("  [0x{:04X}] FF {:02X} - {}", i, marker, marker_name);
                i += 2;
            } else if marker == 0x00 {
                // Stuffed byte
                i += 2;
            } else {
                // Has length field
                if i + 3 < data.len() {
                    let length = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                    println!(
                        "  [0x{:04X}] FF {:02X} - {} (length: {})",
                        i, marker, marker_name, length
                    );

                    // Detailed info for specific markers
                    if marker == 0xC0 || marker == 0xC2 {
                        // SOF
                        if i + 9 < data.len() {
                            let precision = data[i + 4];
                            let height = ((data[i + 5] as u16) << 8) | (data[i + 6] as u16);
                            let width = ((data[i + 7] as u16) << 8) | (data[i + 8] as u16);
                            let num_components = data[i + 9];
                            println!(
                                "      {}x{}, {} bits, {} components",
                                width, height, precision, num_components
                            );

                            let mut comp_idx = i + 10;
                            for comp_num in 0..num_components {
                                if comp_idx + 2 < data.len() {
                                    let id = data[comp_idx];
                                    let sampling = data[comp_idx + 1];
                                    let h_samp = (sampling >> 4) & 0x0F;
                                    let v_samp = sampling & 0x0F;
                                    let quant_tbl = data[comp_idx + 2];
                                    println!(
                                        "      Component {}: ID={}, sampling={}x{}, quant_tbl={}",
                                        comp_num, id, h_samp, v_samp, quant_tbl
                                    );
                                    comp_idx += 3;
                                }
                            }
                        }
                    } else if marker == 0xDA {
                        // SOS
                        if i + 5 < data.len() {
                            let num_components = data[i + 4];
                            println!("      {} components in scan", num_components);

                            let mut comp_idx = i + 5;
                            for comp_num in 0..num_components {
                                if comp_idx + 1 < data.len() {
                                    let comp_sel = data[comp_idx];
                                    let table_sel = data[comp_idx + 1];
                                    let dc_tbl = (table_sel >> 4) & 0x0F;
                                    let ac_tbl = table_sel & 0x0F;
                                    println!(
                                        "      Component {}: selector={}, DC_tbl={}, AC_tbl={}",
                                        comp_num, comp_sel, dc_tbl, ac_tbl
                                    );
                                    comp_idx += 2;
                                }
                            }

                            if comp_idx + 2 < data.len() {
                                let ss = data[comp_idx];
                                let se = data[comp_idx + 1];
                                let ah_al = data[comp_idx + 2];
                                let ah = (ah_al >> 4) & 0x0F;
                                let al = ah_al & 0x0F;
                                println!(
                                    "      Spectral: Ss={}, Se={}, Ah={}, Al={}",
                                    ss, se, ah, al
                                );
                            }
                        }
                    } else if marker == 0xE2 {
                        // APP2 - check for ICC
                        if i + 4 + 12 < data.len() {
                            let marker_data = &data[i + 4..];
                            if marker_data.len() >= 12 && &marker_data[..12] == b"ICC_PROFILE\0" {
                                if marker_data.len() >= 14 {
                                    let chunk_num = marker_data[12];
                                    let total_chunks = marker_data[13];
                                    println!(
                                        "      ICC Profile chunk {} of {}",
                                        chunk_num, total_chunks
                                    );
                                }
                            }
                        }
                    }

                    i += 2 + length;
                } else {
                    break;
                }
            }
        } else {
            i += 1;
        }
    }
}

fn main() {
    let png_path = "../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";

    // Load PNG
    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];
    let width = info.width as u32;
    let height = info.height as u32;

    // Encode with Rust XYB (progressive to match C++ default)
    let rust_xyb = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .use_xyb(true)
        .mode(jpegli::JpegMode::Progressive)
        .encode(rgb)
        .unwrap();

    // Encode with C++ XYB
    if let Some(cjpegli) = jpegli::test_utils::find_cjpegli() {
        let ppm_path = "/tmp/compare_xyb.ppm";
        let cpp_path = "/tmp/compare_xyb_cpp.jpg";

        let mut ppm = fs::File::create(ppm_path).unwrap();
        writeln!(ppm, "P6").unwrap();
        writeln!(ppm, "{} {}", width, height).unwrap();
        writeln!(ppm, "255").unwrap();
        ppm.write_all(rgb).unwrap();
        drop(ppm);

        Command::new(&cjpegli)
            .args([ppm_path, cpp_path, "-q", "90", "--xyb"])
            .output()
            .unwrap();

        let cpp_xyb = fs::read(cpp_path).unwrap();

        dump_jpeg_structure(&rust_xyb, "Rust XYB");
        dump_jpeg_structure(&cpp_xyb, "C++ XYB");
    }
}
