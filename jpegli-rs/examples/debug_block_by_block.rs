// Find exactly which block causes the failure
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn gray_photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn test_size(w: u32, h: u32) -> bool {
    let data = gray_photo_like(w, h);
    let jpeg = Encoder::new()
        .width(w)
        .height(h)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    jpeg_decoder::Decoder::new(&jpeg[..]).decode().is_ok()
}

fn main() {
    // Binary search on dimensions
    println!("Testing various sizes...\n");

    // Test widths from 48-56 with height 50
    println!("Width scan (height=50):");
    for w in 48..=56 {
        let ok = test_size(w, 50);
        println!("  {}x50: {}", w, if ok { "OK" } else { "FAIL" });
    }

    println!("\nHeight scan (width=50):");
    for h in 48..=56 {
        let ok = test_size(50, h);
        println!("  50x{}: {}", h, if ok { "OK" } else { "FAIL" });
    }

    // Check if it's related to block count or specific pixel content
    println!("\nBlock count analysis:");
    for size in 48..=57 {
        let blocks_h = (size + 7) / 8;
        let blocks = blocks_h * blocks_h;
        let last_block_cols = size % 8;
        let ok = test_size(size, size);
        println!(
            "  {}x{}: {} blocks, last_block_cols={}, {}",
            size,
            size,
            blocks,
            if last_block_cols == 0 {
                8
            } else {
                last_block_cols
            },
            if ok { "OK" } else { "FAIL" }
        );
    }
}
