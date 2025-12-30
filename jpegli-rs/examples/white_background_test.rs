//! Test jpegli behavior on white/uniform backgrounds
//!
//! Simulates background-removed product images with large uniform white areas

use butteraugli::{compute_butteraugli, ButteraugliParams};
use dssim::Dssim;

fn main() {
    println!("=== jpegli White Background Test ===\n");

    // Create test images with different amounts of white background
    let test_cases = [
        ("10% white bg", 0.10),
        ("50% white bg", 0.50),
        ("80% white bg", 0.80),
        ("95% white bg", 0.95),
    ];

    let width = 800usize;
    let height = 600usize;
    let total_pixels = width * height;

    let attr = Dssim::new();

    println!("{:20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Scenario", "jpegli", "mozjpeg", "j_dssim", "m_dssim", "j_butter");
    println!("{}", "-".repeat(75));

    for (name, white_fraction) in test_cases {
        // Create image: white background + colored product in center
        let mut pixels = vec![255u8; width * height * 3]; // Start all white

        // Add a "product" (colored rectangle) in the center
        let product_w = ((1.0f64 - white_fraction).sqrt() * width as f64) as usize;
        let product_h = ((1.0f64 - white_fraction).sqrt() * height as f64) as usize;
        let x_start = (width - product_w) / 2;
        let y_start = (height - product_h) / 2;

        for y in y_start..(y_start + product_h) {
            for x in x_start..(x_start + product_w) {
                let idx = (y * width + x) * 3;
                // Create a gradient product with texture
                pixels[idx] = (128 + (x % 64) as u8).min(230);     // R
                pixels[idx + 1] = (64 + (y % 32) as u8).min(200);  // G
                pixels[idx + 2] = (180 - (x % 50) as u8).max(100); // B
            }
        }

        // Encode with jpegli
        let jpegli_result = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .quality(jpegli::quant::Quality::from_quality(85.0))
            .encode(&pixels)
            .unwrap();

        // Encode with mozjpeg
        let mozjpeg_result = encode_mozjpeg(&pixels, width, height, 85);

        // Measure quality
        let j_dssim = compute_dssim(&attr, &pixels, &jpegli_result, width, height);
        let m_dssim = compute_dssim(&attr, &pixels, &mozjpeg_result, width, height);
        let j_butter = compute_butter(&pixels, &jpegli_result, width, height);

        println!("{:20} {:>10} {:>10} {:>10.5} {:>10.5} {:>10.2}",
            name, jpegli_result.len(), mozjpeg_result.len(), j_dssim, m_dssim, j_butter);
    }

    println!("\n=== Analysis ===\n");
    println!("White backgrounds are LOW variance → AQ treats them as 'simple' regions");
    println!("jpegli allocates FEWER bits to white areas (good for compression)");
    println!("Product edges against white may see different treatment:");
    println!("  - AQ sees high contrast → allocates MORE bits");
    println!("  - This preserves edge sharpness against white bg");
    println!("\nFor background-removed products:");
    println!("  - Standard AQ (scale 1.0) should work well");
    println!("  - AQ 0.25 (for sharpened) may help if product edges are sharpened");
}

fn compute_dssim(
    attr: &Dssim,
    orig: &[u8],
    jpeg_data: &[u8],
    width: usize,
    height: usize,
) -> f64 {
    let orig_rgba: Vec<rgb::RGBA<u8>> = orig
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let decoded = decoder.decode().expect("decode");

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr.create_image_rgba(&decoded_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, decoded_img);
    dssim.into()
}

fn compute_butter(orig: &[u8], jpeg_data: &[u8], width: usize, height: usize) -> f64 {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let decoded = decoder.decode().expect("decode");

    let params = ButteraugliParams::default();
    compute_butteraugli(orig, &decoded, width, height, &params)
        .map(|r| r.score)
        .unwrap_or(999.0)
}

fn encode_mozjpeg(pixels: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
    std::panic::catch_unwind(|| {
        use mozjpeg::{ColorSpace, Compress};

        let mut comp = Compress::new(ColorSpace::JCS_RGB);
        comp.set_size(width, height);
        comp.set_quality(quality as f32);
        comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));

        let mut started = comp.start_compress(Vec::new()).expect("start");

        let row_stride = width * 3;
        for y in 0..height {
            let row_start = y * row_stride;
            let row = &pixels[row_start..row_start + row_stride];
            let _ = started.write_scanlines(row);
        }

        started.finish().expect("finish")
    })
    .unwrap_or_default()
}
