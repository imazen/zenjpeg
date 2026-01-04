//! Test if jpegli-rs decoder is at least self-consistent

use jpegli::{Decoder, Encoder, PixelFormat};

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    // Load PNG
    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];
    let width = info.width as u32;
    let height = info.height as u32;

    println!("Original: {}x{}, {} bytes\n", width, height, rgb.len());

    // Test 1: Baseline roundtrip - decode twice
    println!("=== Baseline Roundtrip ===");
    let baseline_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Baseline)
        .encode(rgb)
        .unwrap();

    println!("Encoded: {} bytes", baseline_jpeg.len());

    let decoded1 = Decoder::new().decode(&baseline_jpeg).unwrap();
    println!("Decode 1: {} bytes", decoded1.data.len());

    // Re-encode
    let reencoded = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Baseline)
        .encode(&decoded1.data)
        .unwrap();

    let decoded2 = Decoder::new().decode(&reencoded).unwrap();
    println!("Decode 2: {} bytes", decoded2.data.len());

    // Compare decoded1 vs decoded2
    let mut max_diff = 0i16;
    let mut diff_count = 0;

    for (d1, d2) in decoded1.data.iter().zip(decoded2.data.iter()) {
        let diff = (*d1 as i16 - *d2 as i16).abs();
        max_diff = max_diff.max(diff);
        if diff > 0 {
            diff_count += 1;
        }
    }

    println!("\nDecode 1 vs Decode 2:");
    println!("  Max diff: {}", max_diff);
    println!("  Pixels differ: {} / {} ({:.2}%)",
        diff_count, decoded1.data.len(),
        (diff_count as f64 / decoded1.data.len() as f64) * 100.0);

    if max_diff == 0 {
        println!("\n✓ DECODER IS DETERMINISTIC - same input produces same output");
    } else {
        println!("\n✗ DECODER IS NON-DETERMINISTIC - this is a bug!");
    }

    // Test 2: Progressive roundtrip
    println!("\n=== Progressive Roundtrip ===");
    let progressive_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Progressive)
        .encode(rgb)
        .unwrap();

    println!("Encoded: {} bytes", progressive_jpeg.len());

    let prog_decoded1 = Decoder::new().decode(&progressive_jpeg).unwrap();
    println!("Decode 1: {} bytes", prog_decoded1.data.len());

    let prog_reencoded = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Progressive)
        .encode(&prog_decoded1.data)
        .unwrap();

    let prog_decoded2 = Decoder::new().decode(&prog_reencoded).unwrap();
    println!("Decode 2: {} bytes", prog_decoded2.data.len());

    max_diff = 0;
    diff_count = 0;

    for (d1, d2) in prog_decoded1.data.iter().zip(prog_decoded2.data.iter()) {
        let diff = (*d1 as i16 - *d2 as i16).abs();
        max_diff = max_diff.max(diff);
        if diff > 0 {
            diff_count += 1;
        }
    }

    println!("\nDecode 1 vs Decode 2:");
    println!("  Max diff: {}", max_diff);
    println!("  Pixels differ: {} / {} ({:.2}%)",
        diff_count, prog_decoded1.data.len(),
        (diff_count as f64 / prog_decoded1.data.len() as f64) * 100.0);

    if max_diff == 0 {
        println!("\n✓ DECODER IS DETERMINISTIC");
    } else {
        println!("\n✗ DECODER IS NON-DETERMINISTIC");
    }
}
