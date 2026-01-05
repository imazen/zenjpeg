//! Test if YCbCr 4:4:4 progressive has the same checkerboard pattern

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

    // YCbCr Baseline (4:4:4)
    let ycbcr_base = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(70.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .use_xyb(false)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    // YCbCr Progressive (4:4:4)
    let ycbcr_prog = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(70.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(false)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    let base_decoded = decode_jpeg(&ycbcr_base).unwrap();
    let prog_decoded = decode_jpeg(&ycbcr_prog).unwrap();

    println!("YCbCr 4:4:4 Progressive vs Baseline:");
    println!();

    // Check if differences are in specific channels
    let mut total_diffs = 0;

    for i in 0..base_decoded.len() / 3 {
        let base_r = base_decoded[i * 3];
        let base_g = base_decoded[i * 3 + 1];
        let base_b = base_decoded[i * 3 + 2];

        let prog_r = prog_decoded[i * 3];
        let prog_g = prog_decoded[i * 3 + 1];
        let prog_b = prog_decoded[i * 3 + 2];

        if base_r != prog_r || base_g != prog_g || base_b != prog_b {
            total_diffs += 1;
        }
    }

    println!(
        "Total pixels differing: {}/{}",
        total_diffs,
        base_decoded.len() / 3
    );
    println!();

    if total_diffs == 0 {
        println!("✅ YCbCr Progressive works correctly!");
        return;
    }

    // Check spatial pattern
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

                    if base_decoded[idx] != prog_decoded[idx]
                        || base_decoded[idx + 1] != prog_decoded[idx + 1]
                        || base_decoded[idx + 2] != prog_decoded[idx + 2]
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
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
