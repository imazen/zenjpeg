//! Compare JPEG headers between XYB and YCbCr encoding

use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn parse_jpeg_header(jpeg: &[u8], label: &str) {
    println!("\n=== {} ===", label);

    let mut pos = 2; // Skip SOI

    while pos < jpeg.len() - 1 {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let marker = jpeg[pos + 1];
        pos += 2;

        match marker {
            0xC0 | 0xC2 => {
                // SOF
                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                let height = ((jpeg[pos + 3] as u16) << 8) | (jpeg[pos + 4] as u16);
                let width = ((jpeg[pos + 5] as u16) << 8) | (jpeg[pos + 6] as u16);
                let num_components = jpeg[pos + 7];

                println!(
                    "SOF{}: {}x{}, {} components",
                    if marker == 0xC0 { 0 } else { 2 },
                    width,
                    height,
                    num_components
                );

                for i in 0..num_components as usize {
                    let offset = pos + 8 + i * 3;
                    let id = jpeg[offset];
                    let sampling = jpeg[offset + 1];
                    let quant = jpeg[offset + 2];
                    println!(
                        "  Component {}: id={}, sampling={}x{}, quant_table={}",
                        i,
                        id,
                        sampling >> 4,
                        sampling & 0xF,
                        quant
                    );
                }
                pos += len;
            }
            0xC4 => {
                // DHT
                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                let mut dht_pos = pos + 2;
                let dht_end = pos + len;

                while dht_pos < dht_end {
                    let info = jpeg[dht_pos];
                    let table_class = info >> 4; // 0 = DC, 1 = AC
                    let table_idx = info & 0x0F;
                    dht_pos += 1;

                    // Count symbols
                    let mut num_symbols = 0;
                    for i in 0..16 {
                        num_symbols += jpeg[dht_pos + i] as usize;
                    }

                    let class_name = if table_class == 0 { "DC" } else { "AC" };
                    println!(
                        "DHT: {} table {}, {} symbols",
                        class_name, table_idx, num_symbols
                    );

                    dht_pos += 16 + num_symbols;
                }
                pos += len;
            }
            0xDA => {
                // SOS
                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                let num_components = jpeg[pos + 2];

                println!("SOS: {} components", num_components);
                for i in 0..num_components as usize {
                    let offset = pos + 3 + i * 2;
                    let comp_id = jpeg[offset];
                    let tables = jpeg[offset + 1];
                    let dc_table = tables >> 4;
                    let ac_table = tables & 0x0F;
                    println!(
                        "  Component {}: id={}, DC_table={}, AC_table={}",
                        i, comp_id, dc_table, ac_table
                    );
                }

                let ss = jpeg[pos + 3 + num_components as usize * 2];
                let se = jpeg[pos + 4 + num_components as usize * 2];
                let ah_al = jpeg[pos + 5 + num_components as usize * 2];
                println!(
                    "  Spectral: ss={}, se={}, ah={}, al={}",
                    ss,
                    se,
                    ah_al >> 4,
                    ah_al & 0xF
                );

                // Count entropy data
                let ecs_start = pos + len;
                let mut ecs_end = ecs_start;
                while ecs_end < jpeg.len() - 1 {
                    if jpeg[ecs_end] == 0xFF
                        && jpeg[ecs_end + 1] != 0x00
                        && jpeg[ecs_end + 1] != 0xFF
                    {
                        if jpeg[ecs_end + 1] < 0xD0 || jpeg[ecs_end + 1] > 0xD7 {
                            break;
                        }
                    }
                    ecs_end += 1;
                }
                println!("  Entropy data: {} bytes", ecs_end - ecs_start);

                pos = ecs_end;
            }
            0xDB => {
                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                println!("DQT: {} bytes", len);
                pos += len;
            }
            0xD9 => {
                println!("EOI at 0x{:04x}", pos - 2);
                break;
            }
            0xE0..=0xEF => {
                let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                println!("APP{}: {} bytes", marker - 0xE0, len);
                pos += len;
            }
            _ => {
                if marker >= 0xC0 && marker <= 0xFE {
                    let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
                    pos += len;
                }
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

    parse_jpeg_header(&jpeg_xyb, "XYB + optimized (FAILS)");
    parse_jpeg_header(&jpeg_ycbcr, "YCbCr + optimized (WORKS)");
}
