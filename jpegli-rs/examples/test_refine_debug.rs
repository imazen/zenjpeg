// Minimal test to debug progressive refinement encoding
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn test_case(name: &str, width: u32, height: u32, format: PixelFormat, data: &[u8]) {
    println!("\nTesting {}: {}x{} {:?}", name, width, height, format);

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(format)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    match encoder.encode(data) {
        Ok(jpeg_data) => {
            // Save for analysis
            let filename = format!("/tmp/test_refine_{}.jpg", name.replace(" ", "_"));
            std::fs::write(&filename, &jpeg_data).ok();

            // Try to decode
            match jpeg_decoder::Decoder::new(&jpeg_data[..]).decode() {
                Ok(_) => println!("  PASS: {} bytes", jpeg_data.len()),
                Err(e) => {
                    println!("  FAIL: {:?}", e);

                    // Dump ALL scans with details
                    let mut scan_num = 0;
                    let mut i = 0;
                    while i < jpeg_data.len() - 1 {
                        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
                            let len =
                                ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
                            let num_comp = jpeg_data[i + 4];

                            // Parse component info
                            let mut comp_info = Vec::new();
                            for c in 0..num_comp as usize {
                                let comp_id = jpeg_data[i + 5 + c * 2];
                                let table_sel = jpeg_data[i + 6 + c * 2];
                                let dc_table = table_sel >> 4;
                                let ac_table = table_sel & 0xF;
                                comp_info.push((comp_id, dc_table, ac_table));
                            }

                            let ss = jpeg_data[i + 5 + num_comp as usize * 2];
                            let se = jpeg_data[i + 6 + num_comp as usize * 2];
                            let ah_al = jpeg_data[i + 7 + num_comp as usize * 2];
                            let ah = ah_al >> 4;
                            let al = ah_al & 0xF;

                            let scan_start = i + 2 + len;
                            let mut scan_end = scan_start;
                            while scan_end < jpeg_data.len() - 1 {
                                if jpeg_data[scan_end] == 0xFF && jpeg_data[scan_end + 1] != 0x00 {
                                    break;
                                }
                                scan_end += 1;
                            }

                            let scan_type = if ss == 0 && se == 0 {
                                if ah > 0 {
                                    "DC refine"
                                } else {
                                    "DC first"
                                }
                            } else if ah > 0 {
                                "AC refine"
                            } else {
                                "AC first"
                            };

                            let data_bytes = &jpeg_data[scan_start..scan_end];
                            println!(
                                "  Scan {}: {} comp={:?} Ss={} Se={} Ah={} Al={} [{}] data={}B",
                                scan_num,
                                if ah > 0 { "*" } else { " " },
                                comp_info,
                                ss,
                                se,
                                ah,
                                al,
                                scan_type,
                                data_bytes.len()
                            );

                            if ah > 0 && data_bytes.len() <= 8 {
                                // Show raw bytes for short refinement scans
                                let bits: String = data_bytes
                                    .iter()
                                    .flat_map(|b| {
                                        (0..8).rev().map(move |bit| {
                                            if (b >> bit) & 1 == 1 {
                                                '1'
                                            } else {
                                                '0'
                                            }
                                        })
                                    })
                                    .collect();
                                println!("       Bits: {}", bits);
                            }

                            scan_num += 1;
                            i = scan_end;
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }
        Err(e) => println!("  ENCODE FAIL: {:?}", e),
    }
}

fn main() {
    // Just test the failing RGB case
    let rgb_grad: Vec<u8> = (0..64).flat_map(|i| vec![i as u8 * 4, 128, 64]).collect();
    test_case("8x8_rgb_seq", 8, 8, PixelFormat::Rgb, &rgb_grad);
}
