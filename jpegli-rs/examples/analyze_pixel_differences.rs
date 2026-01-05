// ! Analyze which pixels differ between XYB Baseline and Progressive

use jpegli::{Encoder, PixelFormat};

fn decode_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    decode_zune(data).ok()
}

fn main() {
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = 128;
        }
    }

    // Rust XYB Baseline
    let rust_xyb_base = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(70.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    // Rust XYB Progressive
    let rust_xyb_prog = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(70.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    let rust_base_decoded = decode_jpeg(&rust_xyb_base).unwrap();
    let rust_prog_decoded = decode_jpeg(&rust_xyb_prog).unwrap();

    println!("Analyzing pixel differences between Baseline and Progressive:");
    println!();

    // Check if differences are in specific channels
    let mut r_diffs = 0;
    let mut g_diffs = 0;
    let mut b_diffs = 0;
    let mut total_diffs = 0;

    for i in 0..rust_base_decoded.len() / 3 {
        let base_r = rust_base_decoded[i * 3];
        let base_g = rust_base_decoded[i * 3 + 1];
        let base_b = rust_base_decoded[i * 3 + 2];

        let prog_r = rust_prog_decoded[i * 3];
        let prog_g = rust_prog_decoded[i * 3 + 1];
        let prog_b = rust_prog_decoded[i * 3 + 2];

        if base_r != prog_r {
            r_diffs += 1;
        }
        if base_g != prog_g {
            g_diffs += 1;
        }
        if base_b != prog_b {
            b_diffs += 1;
        }
        if base_r != prog_r || base_g != prog_g || base_b != prog_b {
            total_diffs += 1;
        }
    }

    println!(
        "Total pixels differing: {}/{}",
        total_diffs,
        rust_base_decoded.len() / 3
    );
    println!("R channel differences: {}", r_diffs);
    println!("G channel differences: {}", g_diffs);
    println!("B channel differences: {}", b_diffs);
    println!();

    // Check if differences form a pattern (e.g., every other block)
    println!("Analyzing spatial pattern (8x8 blocks):");
    let blocks_x = width / 8;
    let blocks_y = height / 8;

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let mut block_has_diff = false;
            for py in 0..8 {
                for px in 0..8 {
                    let x = block_x * 8 + px;
                    let y = block_y * 8 + py;
                    let idx = (y * width + x) * 3;

                    if rust_base_decoded[idx] != rust_prog_decoded[idx]
                        || rust_base_decoded[idx + 1] != rust_prog_decoded[idx + 1]
                        || rust_base_decoded[idx + 2] != rust_prog_decoded[idx + 2]
                    {
                        block_has_diff = true;
                        break;
                    }
                }
                if block_has_diff {
                    break;
                }
            }
            print!("{}", if block_has_diff { "X" } else { "." });
        }
        println!();
    }
    println!();

    // Show first 10 differing pixels
    println!("First 10 pixels with differences:");
    let mut count = 0;
    for i in 0..(rust_base_decoded.len() / 3) {
        let base_r = rust_base_decoded[i * 3];
        let base_g = rust_base_decoded[i * 3 + 1];
        let base_b = rust_base_decoded[i * 3 + 2];

        let prog_r = rust_prog_decoded[i * 3];
        let prog_g = rust_prog_decoded[i * 3 + 1];
        let prog_b = rust_prog_decoded[i * 3 + 2];

        if base_r != prog_r || base_g != prog_g || base_b != prog_b {
            let x = i % width;
            let y = i / width;
            let block_x = x / 8;
            let block_y = y / 8;
            println!(
                "  Pixel {} (block {},{}): Baseline=({},{},{}) Progressive=({},{},{})",
                i, block_x, block_y, base_r, base_g, base_b, prog_r, prog_g, prog_b
            );
            count += 1;
            if count >= 10 {
                break;
            }
        }
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
