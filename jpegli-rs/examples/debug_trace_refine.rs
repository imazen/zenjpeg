// Trace exactly what happens during AC refinement encode/decode
// for a minimal failing case
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn main() {
    println!("Testing 49x49 (should work) vs 50x50 (should fail)\n");

    for size in [49u32, 50] {
        let data = photo_like(size, size);

        // Encode with progressive
        let jpeg = Encoder::new()
            .width(size)
            .height(size)
            .pixel_format(PixelFormat::Gray)
            .jpegli_quality(Quality::from_quality(75.0))
            .optimize_huffman(false)
            .mode(JpegMode::Progressive)
            .encode(&data)
            .expect("encode failed");

        // Try to decode with jpeg_decoder
        let result = decode_zune(&jpeg[..]);
        let status = if result.is_ok() { "OK" } else { "FAIL" };

        // Count blocks
        let blocks_x = (size + 7) / 8;
        let blocks_y = (size + 7) / 8;
        let total_blocks = blocks_x * blocks_y;

        println!(
            "{}x{}: {} ({} bytes, {}x{}={} blocks)",
            size,
            size,
            status,
            jpeg.len(),
            blocks_x,
            blocks_y,
            total_blocks
        );

        // If it failed, let's save the file for external analysis
        if result.is_err() {
            let filename = format!("/tmp/fail_{}x{}.jpg", size, size);
            std::fs::write(&filename, &jpeg).expect("write failed");
            println!("  Saved to: {}", filename);

            // Also try with djpeg
            let output = std::process::Command::new("djpeg")
                .arg("-v")
                .arg(&filename)
                .output();
            if let Ok(out) = output {
                if !out.status.success() {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    println!(
                        "  djpeg error: {}",
                        stderr.lines().take(3).collect::<Vec<_>>().join(" | ")
                    );
                }
            }
        }
    }

    // Let's also dump what block 49 (the last complete row for 49x49)
    // and block 49 (first incomplete row for 50x50) look like
    println!("\n--- Detailed block analysis ---");

    // For this we need to check if there's something special about
    // how the last_block_cols calculation affects refinement encoding
    for size in [49u32, 50, 51, 52, 53] {
        let blocks_x = (size + 7) / 8;
        let last_block_cols = (size % 8) as usize;
        let effective_last = if last_block_cols == 0 {
            8
        } else {
            last_block_cols
        };
        println!(
            "{}x{}: blocks_x={}, last_block_cols={} (effective {})",
            size, size, blocks_x, last_block_cols, effective_last
        );
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
