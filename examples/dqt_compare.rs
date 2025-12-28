//! Compare quantization tables between zenjpeg and mozjpeg-oxide

fn main() {
    // Create a simple 16x16 test image
    let width = 16usize;
    let height = 16usize;
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let v = ((y * width + x) * 1) as u8;
            rgb.push(v);
            rgb.push(v);
            rgb.push(v);
        }
    }

    let q = 80;

    // Encode with both
    let zen = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(q))
        .encode_rgb(&rgb, width, height)
        .unwrap();

    let moz = mozjpeg_oxide::Encoder::new()
        .quality(q)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&rgb, width as u32, height as u32)
        .unwrap();

    println!("=== Quantization Table Comparison ===");
    println!("Quality: {}", q);
    println!();

    // Extract all DQT tables
    let zen_tables = extract_all_dqt(&zen);
    let moz_tables = extract_all_dqt(&moz);

    println!("zenjpeg has {} DQT tables", zen_tables.len());
    println!("mozjpeg has {} DQT tables", moz_tables.len());
    println!();

    for (i, (zt, mt)) in zen_tables.iter().zip(moz_tables.iter()).enumerate() {
        let mut diffs = 0;
        let mut total_diff = 0i32;
        for j in 0..64 {
            if zt[j] != mt[j] {
                diffs += 1;
                total_diff += (zt[j] as i32 - mt[j] as i32).abs();
            }
        }
        println!(
            "Table {}: {} differences, total delta = {}",
            i, diffs, total_diff
        );

        // Show first 16 values
        println!("  First 16 values:");
        println!("  Pos:  zen  moz  diff");
        for j in 0..16 {
            let diff = if zt[j] != mt[j] { "<--" } else { "" };
            println!("  {:2}:  {:3}  {:3}  {}", j, zt[j], mt[j], diff);
        }

        // Show ALL differences
        println!("  Differences (all positions):");
        for j in 0..64 {
            if zt[j] != mt[j] {
                println!(
                    "  pos {:2}: zen={:3} moz={:3} (delta={})",
                    j,
                    zt[j],
                    mt[j],
                    (zt[j] as i32 - mt[j] as i32).abs()
                );
            }
        }
        println!();
    }

    // Extract DCT coefficients by parsing the scan data
    println!("=== File Size ===");
    println!("zenjpeg: {} bytes", zen.len());
    println!("mozjpeg: {} bytes", moz.len());

    // Calculate theoretical maximum compression
    println!();
    println!("=== Error Analysis ===");

    let zen_dec = jpeg_decoder::Decoder::new(&zen[..]).decode().unwrap();
    let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

    let mut zen_err = 0i64;
    let mut moz_err = 0i64;
    for i in 0..rgb.len() {
        zen_err += (rgb[i] as i64 - zen_dec[i] as i64).abs();
        moz_err += (rgb[i] as i64 - moz_dec[i] as i64).abs();
    }
    println!("Total absolute error: zen={} moz={}", zen_err, moz_err);
    println!(
        "Ratio: zenjpeg has {:.1}x more error",
        zen_err as f64 / moz_err as f64
    );
}

fn extract_all_dqt(jpeg: &[u8]) -> Vec<[u16; 64]> {
    let mut tables = Vec::new();
    let mut i = 0;

    while i < jpeg.len() - 2 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDB {
            // DQT marker found
            let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
            let mut offset = i + 4;
            let end = i + 2 + len;

            while offset < end {
                let precision_and_id = jpeg[offset];
                let precision = precision_and_id >> 4;
                let _id = precision_and_id & 0x0F;
                offset += 1;

                let mut table = [0u16; 64];
                for j in 0..64 {
                    if precision == 0 {
                        table[j] = jpeg[offset] as u16;
                        offset += 1;
                    } else {
                        table[j] = ((jpeg[offset] as u16) << 8) | (jpeg[offset + 1] as u16);
                        offset += 2;
                    }
                }
                tables.push(table);
            }

            i = end;
        } else {
            i += 1;
        }
    }

    tables
}
