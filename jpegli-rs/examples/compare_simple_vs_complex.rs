//! Compare Huffman tables: simple (works) vs complex (fails)

use jpegli::{Encoder, PixelFormat};

fn parse_dht_tables(jpeg: &[u8]) -> Vec<(u8, u8, [u8; 16])> {
    let mut tables = Vec::new();
    let mut i = 0;

    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xC4 {
            if i + 4 > jpeg.len() {
                break;
            }

            let length = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            if i + 2 + length > jpeg.len() {
                break;
            }

            let mut offset = i + 4;
            let end = i + 2 + length;

            while offset < end {
                if offset + 17 > jpeg.len() {
                    break;
                }

                let tc_th = jpeg[offset];
                let tc = (tc_th >> 4) & 0x0F;
                let th = tc_th & 0x0F;

                let mut bits = [0u8; 16];
                bits.copy_from_slice(&jpeg[offset + 1..offset + 17]);

                let total_symbols: usize = bits.iter().map(|&b| b as usize).sum();
                offset += 17 + total_symbols;

                tables.push((tc, th, bits));
            }

            i += 2 + length;
        } else {
            i += 1;
        }
    }

    tables
}

fn kraft_sum(bits: &[u8; 16]) -> u64 {
    let mut sum = 0u64;
    for (i, &count) in bits.iter().enumerate() {
        let length = (i + 1) as u32;
        sum += (count as u64) << (16 - length);
    }
    sum
}

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];

    let simple_rgb = vec![128u8; 2 * 2 * 3];

    println!("=== SIMPLE IMAGE (2x2, solid color) - zune-jpeg: OK ===\n");
    let simple_opt = Encoder::new()
        .width(2)
        .height(2)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true)
        .encode(&simple_rgb)
        .unwrap();

    let simple_tables = parse_dht_tables(&simple_opt);
    for (tc, th, bits) in &simple_tables {
        let table_type = if *tc == 0 { "DC" } else { "AC" };
        let sum = kraft_sum(bits);
        let total: usize = bits.iter().map(|&b| b as usize).sum();

        println!(
            "{} (id={}): {} symbols, Kraft sum={}",
            table_type, th, total, sum
        );
        print!("  Bits: ");
        for &b in bits.iter() {
            print!("{} ", b);
        }
        println!();
    }

    println!(
        "\n=== FULL IMAGE ({}x{}, complex) - zune-jpeg: FAIL ===\n",
        info.width, info.height
    );
    let full_opt = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();

    let full_tables = parse_dht_tables(&full_opt);
    for (tc, th, bits) in &full_tables {
        let table_type = if *tc == 0 { "DC" } else { "AC" };
        let sum = kraft_sum(bits);
        let total: usize = bits.iter().map(|&b| b as usize).sum();

        println!(
            "{} (id={}): {} symbols, Kraft sum={}",
            table_type, th, total, sum
        );
        print!("  Bits: ");
        for &b in bits.iter() {
            print!("{} ", b);
        }
        println!();
    }

    println!("\n=== DIFFERENCES ===");
    println!("Simple image: All tables have Kraft sum < 65536");
    println!("Full image:   All tables have Kraft sum = 65536 (FULL)");
    println!("\nHypothesis: zune-jpeg requires Kraft sum < 2^16 (strict inequality)");
}
