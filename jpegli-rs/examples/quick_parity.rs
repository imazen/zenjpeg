use jpegli::{Encoder, EncodingBackend, PixelFormat, Quality, Subsampling};

fn main() {
    let w = 256;
    let h = 256;

    // Simple gradient image
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            data[idx] = x as u8;
            data[idx + 1] = y as u8;
            data[idx + 2] = 128;
        }
    }

    // Encode both ways
    let full = Encoder::new()
        .width(w as u32)
        .height(h as u32)
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .jpegli_quality(Quality::from_quality(90.0))
        .encoding_backend(EncodingBackend::FullPlane)
        .encode(&data)
        .unwrap();

    let strip = Encoder::new()
        .width(w as u32)
        .height(h as u32)
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode_strip_based(&data)
        .unwrap();

    println!("Full: {} bytes, Strip: {} bytes", full.len(), strip.len());

    // Save for external inspection
    std::fs::write("/tmp/full_test.jpg", &full).unwrap();
    std::fs::write("/tmp/strip_test.jpg", &strip).unwrap();
    println!("Saved to /tmp/full_test.jpg and /tmp/strip_test.jpg");

    // Decode with zune-jpeg
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);

    let full_dec = {
        let cursor = ZCursor::new(&full);
        let mut dec = JpegDecoder::new_with_options(cursor, opts);
        dec.decode().unwrap()
    };

    let strip_dec = {
        let cursor = ZCursor::new(&strip);
        let mut dec = JpegDecoder::new_with_options(cursor, opts);
        dec.decode().unwrap()
    };

    // Compare
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (a, b) in full_dec.iter().zip(strip_dec.iter()) {
        let d = (*a as i32 - *b as i32).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let avg_diff = sum_diff as f64 / full_dec.len() as f64;

    println!("Max diff: {}, Avg diff: {:.4}", max_diff, avg_diff);

    // Check first few pixels
    println!("First 9 pixels full:  {:?}", &full_dec[0..27]);
    println!("First 9 pixels strip: {:?}", &strip_dec[0..27]);

    // Check first 10 DC coefficients from each file
    println!("\n=== Checking DC coefficients ===");

    // Parse JPEG structure to find SOS marker
    fn find_sos(data: &[u8]) -> Option<usize> {
        for i in 0..data.len() - 1 {
            if data[i] == 0xFF && data[i + 1] == 0xDA {
                return Some(i);
            }
        }
        None
    }

    let full_sos = find_sos(&full).unwrap();
    let strip_sos = find_sos(&strip).unwrap();
    println!(
        "Full SOS at byte {}, Strip SOS at byte {}",
        full_sos, strip_sos
    );

    // Show first 50 bytes of scan data (after SOS header)
    let full_scan_start = full_sos + 14; // SOS marker (2) + length (2) + header (10)
    let strip_scan_start = strip_sos + 14;
    println!(
        "Full scan bytes:  {:02x?}",
        &full[full_scan_start..full_scan_start + 50]
    );
    println!(
        "Strip scan bytes: {:02x?}",
        &strip[strip_scan_start..strip_scan_start + 50]
    );
}
