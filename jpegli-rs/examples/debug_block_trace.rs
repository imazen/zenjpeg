// Trace which block causes the bitstream divergence
// We need to know: at what block/position does 49x49 and 50x50 diverge?

use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn main() {
    println!("Run with DEBUG_REFINE_SYMBOLS=1 and DEBUG_WRITER_BYTES=1 to trace\n");

    // Create minimal failing case
    let data = photo_like(50, 50);

    let jpeg = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    let result = jpeg_decoder::Decoder::new(&jpeg[..]).decode();
    println!(
        "50x50 decode: {}",
        if result.is_ok() { "OK" } else { "FAIL" }
    );

    // Let's also try to isolate which block is problematic
    // by encoding smaller regions and seeing when it fails

    println!("\nTesting smaller regions to isolate failure:");

    for size in [8, 16, 24, 32, 40, 48, 49, 50, 51, 52, 53, 54, 55, 56] {
        let data = photo_like(size, size);
        let jpeg = Encoder::new()
            .width(size)
            .height(size)
            .pixel_format(PixelFormat::Gray)
            .quality(Quality::from_quality(75.0))
            .optimize_huffman(false)
            .mode(JpegMode::Progressive)
            .encode(&data)
            .expect("encode failed");

        let result = jpeg_decoder::Decoder::new(&jpeg[..]).decode();
        let blocks = ((size + 7) / 8) * ((size + 7) / 8);
        let last_col = size % 8;
        println!(
            "{}x{}: {} ({} blocks, last_col={})",
            size,
            size,
            if result.is_ok() { "OK" } else { "FAIL" },
            blocks,
            last_col
        );
    }

    // The pattern: 50, 51, 52 fail. 49, 53+ work.
    // What's special about last_col being 2, 3, or 4?

    println!("\n--- Testing specific last_col values ---");

    // Keep height fixed, vary width
    for w in 48..=60 {
        let h = 50u32; // Fixed height
        let data = photo_like(w, h);
        let jpeg = Encoder::new()
            .width(w)
            .height(h)
            .pixel_format(PixelFormat::Gray)
            .quality(Quality::from_quality(75.0))
            .optimize_huffman(false)
            .mode(JpegMode::Progressive)
            .encode(&data)
            .expect("encode failed");

        let result = jpeg_decoder::Decoder::new(&jpeg[..]).decode();
        let blocks_x = (w + 7) / 8;
        let last_col = w % 8;
        println!(
            "{}x{}: {} (blocks_x={}, last_col={})",
            w,
            h,
            if result.is_ok() { "OK" } else { "FAIL" },
            blocks_x,
            last_col
        );
    }
}
