//! Test XYB encoding with adaptive quantization.
//!
//! Run with:
//! ```
//! cargo run --release --example xyb_aq_test --features hybrid-trellis
//! ```

fn main() {
    let image_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png".to_string());

    // Load image
    let file = std::fs::File::open(&image_path).expect("Failed to open image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    if info.color_type != png::ColorType::Rgb {
        eprintln!("Image must be RGB");
        return;
    }

    let pixels = &buf[..info.buffer_size()];
    let width = info.width as u32;
    let height = info.height as u32;

    println!("Testing XYB + AQ on {} ({}x{})", image_path, width, height);
    println!();

    // 1. Baseline jpegli (YCbCr, no hybrid)
    let baseline = jpegli::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::from_quality(80.0))
        .encode(pixels)
        .expect("baseline encode");
    println!("YCbCr baseline: {} bytes", baseline.len());

    // 2. XYB without hybrid
    let xyb_baseline = jpegli::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::from_quality(80.0))
        .use_xyb(true)
        .encode(pixels)
        .expect("XYB baseline encode");
    println!("XYB baseline:   {} bytes", xyb_baseline.len());

    // 3. XYB with hybrid trellis
    #[cfg(feature = "hybrid-trellis")]
    {
        use jpegli::hybrid_config::HybridConfig;

        let xyb_hybrid = jpegli::Encoder::new()
            .width(width)
            .height(height)
            .quality(jpegli::quant::Quality::from_quality(80.0))
            .use_xyb(true)
            .hybrid_config(HybridConfig::default())
            .encode(pixels)
            .expect("XYB hybrid encode");

        let diff_pct = 100.0 * (xyb_hybrid.len() as f64 - xyb_baseline.len() as f64)
            / xyb_baseline.len() as f64;
        println!(
            "XYB + hybrid:   {} bytes ({:+.1}% vs XYB baseline)",
            xyb_hybrid.len(),
            diff_pct
        );

        // Decode and check quality
        let decoded_baseline = decode_jpeg(&xyb_baseline);
        let decoded_hybrid = decode_jpeg(&xyb_hybrid);

        let dssim_baseline =
            compute_dssim(pixels, &decoded_baseline, width as usize, height as usize);
        let dssim_hybrid = compute_dssim(pixels, &decoded_hybrid, width as usize, height as usize);

        let quality_gain = 100.0 * (dssim_baseline - dssim_hybrid) / dssim_baseline;

        println!();
        println!("Quality comparison (DSSIM, lower is better):");
        println!("  XYB baseline: {:.6}", dssim_baseline);
        println!(
            "  XYB + hybrid: {:.6} ({:+.1}%)",
            dssim_hybrid, -quality_gain
        );
    }

    #[cfg(not(feature = "hybrid-trellis"))]
    {
        println!("(hybrid-trellis feature not enabled)");
    }
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("decode")
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use dssim::Dssim;

    let attr = Dssim::new();

    let orig_rgba: Vec<rgb::RGBA<u8>> = original
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr
        .create_image_rgba(&decoded_rgba, width, height)
        .unwrap();

    let (dssim, _) = attr.compare(&orig_img, decoded_img);
    dssim.into()
}
