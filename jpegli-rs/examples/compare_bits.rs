//! Compare bit consumption between XYB and YCbCr encoding

use jpegli::encode::Encoder;
use jpegli::quant::Quality;

// Count unstuffed bytes in entropy data
fn count_unstuffed_bytes(jpeg: &[u8], start: usize, end: usize) -> usize {
    let mut count = 0;
    let mut i = start;
    while i < end {
        if jpeg[i] == 0xFF && i + 1 < end && jpeg[i + 1] == 0x00 {
            // This is a stuffed FF - only count the FF, skip the 00
            count += 1;
            i += 2;
        } else if jpeg[i] == 0xFF && i + 1 < end && (0xD0..=0xD7).contains(&jpeg[i + 1]) {
            // Restart marker - skip both bytes
            i += 2;
        } else {
            count += 1;
            i += 1;
        }
    }
    count
}

fn find_entropy_data(jpeg: &[u8]) -> (usize, usize) {
    let mut pos = 2;
    while pos < jpeg.len() - 1 {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = jpeg[pos + 1];
        pos += 2;

        if marker == 0xDA {
            let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
            let start = pos + len;
            let mut end = start;
            while end < jpeg.len() - 1 {
                if jpeg[end] == 0xFF && jpeg[end + 1] != 0x00 && jpeg[end + 1] != 0xFF {
                    if jpeg[end + 1] < 0xD0 || jpeg[end + 1] > 0xD7 {
                        break;
                    }
                }
                end += 1;
            }
            return (start, end);
        } else if marker >= 0xC0 && marker <= 0xFE && marker != 0xD8 {
            let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
            pos += len;
        }
    }
    (0, 0)
}

fn main() {
    let width = 64u32;
    let height = 64u32;

    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = ((x * 4) % 256) as u8;
            rgb[i + 1] = ((y * 4) % 256) as u8;
            rgb[i + 2] = 128;
        }
    }

    // XYB optimized
    println!("=== XYB + optimized (FAILS) ===");
    let jpeg_xyb = Encoder::new()
        .width(width)
        .height(height)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&rgb)
        .expect("encode");

    let (start, end) = find_entropy_data(&jpeg_xyb);
    let raw_bytes = end - start;
    let unstuffed = count_unstuffed_bytes(&jpeg_xyb, start, end);
    println!(
        "Entropy data: {} raw bytes, {} unstuffed bytes, {} bits",
        raw_bytes,
        unstuffed,
        unstuffed * 8
    );

    // Components: 2x2, 2x2, 1x1 => 64 + 64 + 16 = 144 blocks
    println!("Expected blocks: 144 (64 + 64 + 16)");
    println!("Bits per block avg: {:.1}", (unstuffed * 8) as f64 / 144.0);

    // YCbCr optimized
    println!("\n=== YCbCr + optimized (WORKS) ===");
    let jpeg_ycbcr = Encoder::new()
        .width(width)
        .height(height)
        .quality(Quality::from_quality(90.0))
        .use_xyb(false)
        .encode(&rgb)
        .expect("encode");

    let (start, end) = find_entropy_data(&jpeg_ycbcr);
    let raw_bytes = end - start;
    let unstuffed = count_unstuffed_bytes(&jpeg_ycbcr, start, end);
    println!(
        "Entropy data: {} raw bytes, {} unstuffed bytes, {} bits",
        raw_bytes,
        unstuffed,
        unstuffed * 8
    );

    // Components: 1x1, 1x1, 1x1 => 64 + 64 + 64 = 192 blocks
    println!("Expected blocks: 192 (64 + 64 + 64)");
    println!("Bits per block avg: {:.1}", (unstuffed * 8) as f64 / 192.0);

    // XYB standard Huffman
    println!("\n=== XYB + standard (WORKS) ===");
    let jpeg_xyb_std = Encoder::new()
        .width(width)
        .height(height)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .optimize_huffman(false)
        .encode(&rgb)
        .expect("encode");

    let (start, end) = find_entropy_data(&jpeg_xyb_std);
    let raw_bytes = end - start;
    let unstuffed = count_unstuffed_bytes(&jpeg_xyb_std, start, end);
    println!(
        "Entropy data: {} raw bytes, {} unstuffed bytes, {} bits",
        raw_bytes,
        unstuffed,
        unstuffed * 8
    );
    println!("Expected blocks: 144");
    println!("Bits per block avg: {:.1}", (unstuffed * 8) as f64 / 144.0);
}
