//! Dump Huffman table details from JPEG
//!
//! **DEPRECATED**: Use `jpeg_inspect` instead:
//!   cargo run --release --example jpeg_inspect -- --huffman image.jpg

use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn dump_huffman_tables(jpeg: &[u8], label: &str) {
    println!("\n=== {} ===", label);

    let mut pos = 2; // Skip SOI

    while pos < jpeg.len() - 1 {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let marker = jpeg[pos + 1];
        pos += 2;

        if marker == 0xC4 {
            // DHT
            let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
            let mut dht_pos = pos + 2;
            let dht_end = pos + len;

            while dht_pos < dht_end {
                let info = jpeg[dht_pos];
                let table_class = info >> 4; // 0 = DC, 1 = AC
                let table_idx = info & 0x0F;
                dht_pos += 1;

                let class_name = if table_class == 0 { "DC" } else { "AC" };
                print!("{} table {}: bits=[", class_name, table_idx);

                // Read BITS (16 bytes)
                let mut bits = [0u8; 16];
                let mut num_symbols = 0;
                for i in 0..16 {
                    bits[i] = jpeg[dht_pos + i];
                    num_symbols += bits[i] as usize;
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{}", bits[i]);
                }
                dht_pos += 16;

                print!("] values=[");
                for i in 0..num_symbols {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("0x{:02X}", jpeg[dht_pos + i]);
                }
                println!("]");

                dht_pos += num_symbols;
            }
            pos += len;
        } else if marker == 0xD9 {
            break;
        } else if marker >= 0xC0 && marker <= 0xFE && marker != 0xD8 {
            if pos + 1 < jpeg.len() {
                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                pos += len;
            } else {
                break;
            }
        }
    }
}

fn main() {
    let width = 64u32;
    let height = 64u32;

    // Create gradient image
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = ((x * 4) % 256) as u8;
            rgb[i + 1] = ((y * 4) % 256) as u8;
            rgb[i + 2] = 128;
        }
    }

    // XYB + optimized (FAILS)
    let jpeg_xyb = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&rgb)
        .expect("encode");

    // YCbCr + optimized (WORKS)
    let jpeg_ycbcr = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(false)
        .encode(&rgb)
        .expect("encode");

    // XYB + standard (WORKS)
    let jpeg_xyb_std = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .optimize_huffman(false)
        .encode(&rgb)
        .expect("encode");

    dump_huffman_tables(&jpeg_xyb, "XYB + optimized (FAILS)");
    dump_huffman_tables(&jpeg_ycbcr, "YCbCr + optimized (WORKS)");
    dump_huffman_tables(&jpeg_xyb_std, "XYB + standard (WORKS)");
}
