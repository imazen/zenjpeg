//! Test decoder compatibility with different DCT scaling.

fn main() {
    // Create a simple solid gray image
    let width = 8;
    let height = 8;
    let gray_value = 200u8;

    // Create RGB data (all gray)
    let rgb: Vec<u8> = (0..width * height * 3).map(|_| gray_value).collect();

    println!("=== Encoder/Decoder Compatibility Test ===\n");
    println!(
        "Input: {}x{} solid gray (value {})",
        width, height, gray_value
    );

    // Encode with jpegli-rs
    let jpeg_data = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(90.0))
        .encode(&rgb)
        .expect("encode failed");

    println!("JPEG size: {} bytes\n", jpeg_data.len());

    // Decode with jpeg-decoder (standard decoder)
    let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
    let decoded_pixels = decoder.decode().expect("decode failed");
    let info = decoder.info().unwrap();

    println!("Decoded: {}x{}", info.width, info.height);
    println!("Pixel format: {:?}", info.pixel_format);

    // Check first few decoded pixel values
    println!("\nFirst 9 decoded pixels (expected ~{}):", gray_value);
    for i in 0..9 {
        let r = decoded_pixels[i * 3];
        let g = decoded_pixels[i * 3 + 1];
        let b = decoded_pixels[i * 3 + 2];
        println!("  Pixel {}: R={}, G={}, B={}", i, r, g, b);
    }

    // Check average value
    let avg: f64 =
        decoded_pixels.iter().map(|&x| x as f64).sum::<f64>() / decoded_pixels.len() as f64;
    println!(
        "\nAverage pixel value: {:.1} (expected: {})",
        avg, gray_value
    );

    // Calculate error
    let mse: f64 = decoded_pixels
        .iter()
        .map(|&x| (x as f64 - gray_value as f64).powi(2))
        .sum::<f64>()
        / decoded_pixels.len() as f64;
    let rmse = mse.sqrt();
    println!("RMSE: {:.2}", rmse);

    if rmse > 10.0 {
        println!("\n*** ERROR: RMSE too high! DCT scaling may be incompatible ***");
    } else {
        println!("\n*** OK: Encoder output is compatible with standard decoder ***");
    }
}
