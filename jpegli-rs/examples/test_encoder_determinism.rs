//! Test if encoder is deterministic (same input → same JPEG bytes)

use jpegli::{Encoder, PixelFormat};

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

    println!("Testing encoder determinism...\n");

    // Encode same input 3 times
    let jpeg1 = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Baseline)
        .encode(rgb)
        .unwrap();

    let jpeg2 = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Baseline)
        .encode(rgb)
        .unwrap();

    let jpeg3 = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Baseline)
        .encode(rgb)
        .unwrap();

    println!("Encode 1: {} bytes", jpeg1.len());
    println!("Encode 2: {} bytes", jpeg2.len());
    println!("Encode 3: {} bytes", jpeg3.len());

    if jpeg1 == jpeg2 && jpeg2 == jpeg3 {
        println!("\n✓ ENCODER IS DETERMINISTIC - same bytes every time");
    } else {
        println!("\n✗ ENCODER IS NON-DETERMINISTIC!");

        // Find first difference
        for (i, (b1, b2)) in jpeg1.iter().zip(jpeg2.iter()).enumerate() {
            if b1 != b2 {
                println!("  First diff at byte {}: {} vs {}", i, b1, b2);
                break;
            }
        }
    }

    // Test progressive too
    println!("\n=== Progressive Mode ===");
    let prog1 = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Progressive)
        .encode(rgb)
        .unwrap();

    let prog2 = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::JpegMode::Progressive)
        .encode(rgb)
        .unwrap();

    println!("Encode 1: {} bytes", prog1.len());
    println!("Encode 2: {} bytes", prog2.len());

    if prog1 == prog2 {
        println!("\n✓ PROGRESSIVE ENCODER IS DETERMINISTIC");
    } else {
        println!("\n✗ PROGRESSIVE ENCODER IS NON-DETERMINISTIC!");
    }
}
