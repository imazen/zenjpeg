//! Check that Kraft sum is < 65536 with pseudo-symbol 256 fix

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

    println!("=== Optimized Huffman tables (WITH pseudo-symbol 256 fix) ===\n");
    let jpeg_opt = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();

    println!("JPEG size: {} bytes\n", jpeg_opt.len());

    let tables = parse_dht_tables(&jpeg_opt);
    for (tc, th, bits) in &tables {
        let table_type = if *tc == 0 { "DC" } else { "AC" };
        let sum = kraft_sum(bits);
        let total: usize = bits.iter().map(|&b| b as usize).sum();

        println!(
            "{} table (id={}): {} symbols, Kraft sum = {}",
            table_type, th, total, sum
        );
        print!("  Bits: ");
        for &b in bits.iter() {
            print!("{} ", b);
        }
        println!();

        if sum == 65536 {
            println!("  ⚠️  WARNING: Kraft sum = 65536 (completely full!)");
        } else if sum < 65536 {
            let slack = 65536 - sum;
            println!("  ✓ Slack space: {} (Kraft sum < 65536)", slack);
        } else {
            println!("  ✗ ERROR: Kraft sum > 65536 (invalid!)");
        }
        println!();
    }

    // Verify with mozjpeg
    println!("=== Testing with mozjpeg decoder ===");
    test_mozjpeg(&jpeg_opt);
}

fn test_mozjpeg(jpeg_data: &[u8]) {
    match mozjpeg::Decompress::new_mem(jpeg_data) {
        Ok(decoder) => {
            println!("✓ mozjpeg::Decompress::new_mem() succeeded");
            match decoder.rgb() {
                Ok(_pixels) => {
                    println!("✓ mozjpeg decoder SUCCESS!");
                }
                Err(e) => {
                    println!("✗ mozjpeg decoder FAILED: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ mozjpeg::Decompress::new_mem() FAILED: {:?}", e);
        }
    }
}
