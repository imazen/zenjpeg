//! Decode the same JPEG twice - should get identical pixels

use jpegli::{Decoder, Encoder, PixelFormat};

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    // Load PNG and encode
    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];

    let jpeg = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .encode(rgb)
        .unwrap();

    println!("Encoded JPEG: {} bytes\n", jpeg.len());

    // Decode same JPEG 3 times
    let decoded1 = Decoder::new().decode(&jpeg).unwrap();
    let decoded2 = Decoder::new().decode(&jpeg).unwrap();
    let decoded3 = Decoder::new().decode(&jpeg).unwrap();

    println!("Decoded 3 times, {} bytes each\n", decoded1.data.len());

    // Compare
    let mut diff_12 = 0;
    let mut diff_23 = 0;

    for i in 0..decoded1.data.len() {
        if decoded1.data[i] != decoded2.data[i] {
            diff_12 += 1;
        }
        if decoded2.data[i] != decoded3.data[i] {
            diff_23 += 1;
        }
    }

    println!("Decode 1 vs Decode 2: {} bytes differ", diff_12);
    println!("Decode 2 vs Decode 3: {} bytes differ", diff_23);

    if diff_12 == 0 && diff_23 == 0 {
        println!("\n✓ DECODER IS DETERMINISTIC");
    } else {
        println!("\n✗ DECODER IS NON-DETERMINISTIC - serious bug!");
    }
}
