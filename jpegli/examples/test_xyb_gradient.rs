//! Test XYB encoding with gradient image

use jpegli::{Encoder, Quality};

fn main() {
    // Create 32x32 gradient image
    let width = 32usize;
    let height = 32usize;
    let mut data = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            // Create RGB gradient
            data[i] = (x * 255 / (width - 1)) as u8; // R increases left to right
            data[i + 1] = (y * 255 / (height - 1)) as u8; // G increases top to bottom
            data[i + 2] = 128; // B constant
        }
    }

    // Encode with XYB mode
    #[allow(deprecated)]
    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(Quality::Traditional(90.0))
        .use_xyb(true);

    let jpeg_data = encoder.encode(&data).expect("Failed to encode");

    // Save for external testing
    std::fs::write("/tmp/rust_xyb_gradient.jpg", &jpeg_data).expect("write failed");
    println!("Encoded gradient to {} bytes", jpeg_data.len());
    println!("Saved to /tmp/rust_xyb_gradient.jpg");

    // Also generate input PPM for C++ comparison
    let mut ppm = format!("P6\n{} {}\n255\n", width, height).into_bytes();
    ppm.extend_from_slice(&data);
    std::fs::write("/tmp/gradient.ppm", &ppm).expect("write ppm failed");
    println!("Input saved to /tmp/gradient.ppm");
}
