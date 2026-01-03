//! XYB encoder with hybrid trellis for comparison testing.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: encode_xyb_hybrid <input.png> <output.jpg> <quality>");
        std::process::exit(1);
    }

    let input = &args[1];
    let output = &args[2];
    let quality: f32 = args[3].parse().expect("Invalid quality");

    // Load PNG
    let file = std::fs::File::open(input).expect("Failed to open input");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let pixels = &buf[..info.buffer_size()];
    let width = info.width;
    let height = info.height;

    // Encode XYB JPEG with hybrid
    #[cfg(feature = "experimental-hybrid-trellis")]
    let jpeg = {
        use jpegli::hybrid_config::HybridConfig;
        jpegli::Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(jpegli::quant::Quality::from_quality(quality))
            .use_xyb(true)
            .hybrid_config(HybridConfig::default())
            .encode(pixels)
            .expect("Encode failed")
    };

    #[cfg(not(feature = "experimental-hybrid-trellis"))]
    let jpeg = {
        jpegli::Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(jpegli::quant::Quality::from_quality(quality))
            .use_xyb(true)
            .encode(pixels)
            .expect("Encode failed")
    };

    std::fs::write(output, &jpeg).expect("Failed to write output");
    println!("Encoded {} bytes", jpeg.len());
}
