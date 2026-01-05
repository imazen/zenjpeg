// Test progressive encoding with and without refinement
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn main() {
    let sizes = [49u32, 50, 51, 52, 53];

    println!("Testing progressive modes:\n");

    for &size in &sizes {
        let data = photo_like(size, size);

        // Test with baseline (sequential)
        let jpeg_baseline = Encoder::new()
            .width(size)
            .height(size)
            .pixel_format(PixelFormat::Gray)
            .jpegli_quality(Quality::from_quality(75.0))
            .mode(JpegMode::Baseline)
            .encode(&data)
            .expect("encode failed");

        let baseline_ok = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&jpeg_baseline[..]))
            .decode()
            .is_ok();

        // Test with progressive (has refinement)
        let jpeg_prog = Encoder::new()
            .width(size)
            .height(size)
            .pixel_format(PixelFormat::Gray)
            .jpegli_quality(Quality::from_quality(75.0))
            .optimize_huffman(false) // Use standard tables
            .mode(JpegMode::Progressive)
            .encode(&data)
            .expect("encode failed");

        let prog_ok = decode_zune(&jpeg_prog[..]).is_ok();

        println!(
            "{}x{}: baseline={} ({} bytes), progressive={} ({} bytes)",
            size,
            size,
            if baseline_ok { "OK" } else { "FAIL" },
            jpeg_baseline.len(),
            if prog_ok { "OK" } else { "FAIL" },
            jpeg_prog.len()
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
